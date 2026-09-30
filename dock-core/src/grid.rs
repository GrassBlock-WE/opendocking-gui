//! Precalculated 3-D affinity maps and trilinear interpolation with analytic gradients.
//!
//! # What a map holds
//!
//! A receptor is tabulated **once** onto a regular grid, after which scoring a
//! ligand atom costs eight table lookups instead of a loop over receptor
//! atoms. Each grid point stores four maps per element class:
//!
//! | map              | contains                          | written by receptor atoms that | read by ligand atoms that |
//! |------------------|-----------------------------------|--------------------------------|---------------------------|
//! | `Shape`          | Gaussian shape + steric repulsion  | every atom                     | every atom                |
//! | `HbFromDonor`    | hydrogen bond                     | can donate                     | can accept                |
//! | `HbFromAcceptor` | hydrogen bond                     | can accept                     | can donate                |
//! | `Hydrophobic`    | apolar contact                    | apolar                         | apolar                    |
//!
//! AutoDock Vina collapses the two hydrogen-bond maps into one, which lets a
//! donor–donor pair pick up a spurious hydrogen bond. Splitting them costs one
//! extra map per point and makes the decomposition exact — see
//! [`crate::scoring`] for the full rationale.
//!
//! # Gradients
//!
//! The interpolant is
//!
//! ```text
//! E(p) = Σ_k w_k(u) · F(g_k)
//! ```
//!
//! with `u` the fractional cell coordinate, `g_k` the eight cell corners and
//! `F` the tabulated field. Its **exact** analytic gradient is
//!
//! ```text
//! ∂E/∂p = Σ_k (∂w_k/∂u) · F(g_k) · ∂u/∂p
//! ```
//!
//! Note what is *not* there. A tempting "improvement" is to also carry the
//! radial derivative `F'` of the tabulated field and add
//! `w_k · F'(|p−g_k|) · (p−g_k)/|p−g_k|`. That term is **wrong**: the gradient
//! of a trilinear interpolant is not the trilinear interpolation of the
//! field's gradient, and a stored `F'` is the derivative with respect to the
//! *receptor* atom, not with respect to the probe. The engine therefore
//! computes the gradient of the interpolant it actually uses.
//!
//! The honest consequence is that this gradient is discontinuous across cell
//! boundaries, because a trilinear interpolant is only C⁰. That is inherent
//! to precalculated maps, and is why the default spacing is a fine 0.375 Å
//! and the local optimiser keeps a gradient-descent safeguard.
//! [`GridMaps::spacing`] is the knob: halving it quarters the discontinuity
//! at 8× the memory and precomputation cost.
//!
//! # Provenance
//!
//! The `e` / `e_hb` / `e_hyd` decomposition follows AutoDock Vina 1.2
//! (Trott & Olson, *J. Comput. Chem.* **31**, 455, 2010; Apache-2.0). The
//! trilinear scheme and the precalculated-grid idea originate with AutoDock
//! 4.2 (Morris et al., *J. Comput. Chem.* **29**, 2789, 2008; GPL-2.0). Both
//! are open source and were re-implemented here from their published
//! descriptions.

use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{DockError, Result};
use crate::scoring::{ScoringFunction, SpatialKernels};
use crate::types::{grid_type_index, Element, Molecule, Vec3, GRID_TYPE_COUNT};

/// Spatial resolution of the maps, in Ångström.
pub const DEFAULT_SPACING: f64 = 0.375;

/// Hard ceiling on the number of grid points, checked before allocation.
///
/// 2^28 points is 40 GiB of `f32` at the current stride, so this limit only
/// ever fires on a request that was never going to succeed anyway. It exists
/// so that an absurd spacing is reported as a parameter error rather than
/// aborting the process on a capacity overflow.
pub const MAX_GRID_POINTS: u64 = 1 << 28;

/// An axis-aligned cuboid in which a ligand is allowed to be placed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridBox {
    /// Minimum corner, inclusive.
    pub min: Vec3,
    /// Maximum corner, exclusive.
    pub max: Vec3,
}

impl GridBox {
    /// Build a box from two corners, validating that it is non-degenerate.
    ///
    /// The comparisons are written as `!(max > min)` on purpose: for `f64` this
    /// is the only form that also rejects `NaN`, which would otherwise sail
    /// through a `max <= min` check and poison every downstream index.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(min: Vec3, max: Vec3) -> Result<GridBox> {
        for k in 0..3 {
            // `!(a > b)` rather than `a <= b` so that a NaN corner is rejected
            // too: NaN fails every comparison, so the `else` branch would let it
            // through and it would poison every downstream index and length.
            if !min[k].is_finite() || !max[k].is_finite() || !(max[k] > min[k]) {
                return Err(DockError::param(
                    "grid box",
                    format!("{min:?} .. {max:?}"),
                    format!("axis {k} must be finite with a positive extent"),
                ));
            }
        }
        Ok(GridBox { min, max })
    }

    /// A box of the given size centred on `center`.
    pub fn centered(center: Vec3, size: Vec3) -> Result<GridBox> {
        let half = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0];
        GridBox::new(
            [
                center[0] - half[0],
                center[1] - half[1],
                center[2] - half[2],
            ],
            [
                center[0] + half[0],
                center[1] + half[1],
                center[2] + half[2],
            ],
        )
    }

    /// Geometric centre.
    pub fn center(&self) -> Vec3 {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Edge lengths.
    pub fn size(&self) -> Vec3 {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// True if `p` lies inside the box (inclusive of `min`, exclusive of `max`).
    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        (0..3).all(|k| p[k] >= self.min[k] && p[k] < self.max[k])
    }
}

/// The number of scalar maps stored per grid point.
///
/// One slot each for shape, hydrogen bond from a receptor donor, hydrogen bond
/// from a receptor acceptor, and hydrophobic contact. Vina uses three (shape,
/// `e_hb`, `e_hyd`) and resolves the donor/acceptor asymmetry at scoring time;
/// splitting the hydrogen-bond map in two here is a deliberate deviation, and
/// it is what removes the donor–donor false hydrogen bond a single `e_hb` map
/// produces. See the module documentation.
pub const MAPS_PER_TYPE: usize = 4;

/// Number of `f32` values stored per grid point, across every element type.
///
/// The WGSL kernel hard-codes the same number as `STRIDE`; the `gpu` feature's
/// test asserts the two agree, because a mismatch shifts every lookup without
/// failing anywhere.
pub const fn map_stride() -> usize {
    GRID_TYPE_COUNT * MAPS_PER_TYPE
}

/// Layout of the value block at one grid point, for one element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapSlot {
    /// Gaussian shape terms plus steric repulsion — every pair.
    Shape = 0,
    /// Hydrogen bond, tabulated from receptor **donors**; read by ligand acceptors.
    HbFromDonor = 1,
    /// Hydrogen bond, tabulated from receptor **acceptors**; read by ligand donors.
    HbFromAcceptor = 2,
    /// Apolar contact — apolar–apolar pairs only.
    Hydrophobic = 3,
}

impl MapSlot {
    /// All four slots, in storage order.
    pub const ALL: [MapSlot; MAPS_PER_TYPE] = [
        MapSlot::Shape,
        MapSlot::HbFromDonor,
        MapSlot::HbFromAcceptor,
        MapSlot::Hydrophobic,
    ];

    /// Dense index into the per-point block.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Number of grid points a box will be tabulated at, without allocating them.
///
/// Exposed so a caller can size a box before committing to the memory: a 40 Å
/// cube at 0.375 Å spacing is 38 million points, which is over 600 MB per map
/// set.
pub fn estimate_dims(box_: &GridBox, spacing: f64) -> [usize; 3] {
    let spacing = if spacing <= 0.0 {
        DEFAULT_SPACING
    } else {
        spacing
    };
    let s = box_.size();
    [
        ((s[0] / spacing).ceil() as usize + 1).max(2),
        ((s[1] / spacing).ceil() as usize + 1).max(2),
        ((s[2] / spacing).ceil() as usize + 1).max(2),
    ]
}

/// A precalculated set of affinity maps covering one [`GridBox`].
///
/// `data` is a flat array laid out as
/// `[((ix + nx * (iy + ny * iz)) * stride) + type * MAPS_PER_TYPE + slot]`,
/// where `stride = GRID_TYPE_COUNT * MAPS_PER_TYPE`. So the flat length is
/// exactly `stride * nx * ny * nz` — one `f32` per (point, element, slot) and
/// nothing else. In particular there is **no** stored radial derivative: the
/// gradient of a trilinear interpolant is not the interpolation of the field's
/// gradient, so keeping a per-point `F'` would invite a term that is not the
/// derivative of anything the caller evaluates. See the module documentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridMaps {
    /// Lower corner of the grid, which equals [`GridBox::min`].
    pub min: Vec3,
    /// Number of grid points along each axis.
    pub dims: [usize; 3],
    /// Spacing between grid points, in Ångström.
    pub spacing: [f64; 3],
    /// Flat map storage, laid out as documented on the struct.
    data: Vec<f32>,
    /// The box this grid covers.
    pub box_: GridBox,
    /// Heavy-atom coordinates of the receptor these maps were built from.
    ///
    /// Kept so a steric check can be made later without asking the caller for
    /// a receptor it already supplied to [`GridMaps::precalculate`]. Hydrogens
    /// are excluded: a polar hydrogen sits on its own heavy atom by
    /// construction, so it carries no independent steric information.
    receptor_heavy: Vec<Vec3>,
}

impl GridMaps {
    /// Number of grid points along each axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Total number of tabulated points.
    pub fn point_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Total number of scalar maps (elements × slots).
    pub fn map_count(&self) -> usize {
        GRID_TYPE_COUNT * MAPS_PER_TYPE
    }

    /// Number of `f32` values held by the grid.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Spacing in Ångström.
    pub fn spacing(&self) -> f64 {
        self.spacing[0]
    }

    /// True if p lies inside the tabulated volume.
    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        (0..3).all(|k| {
            let d = p[k] - self.min[k];
            d >= 0.0 && d < (self.dims[k] - 1) as f64 * self.spacing[k]
        })
    }

    /// The box this grid covers.
    pub fn grid_box(&self) -> GridBox {
        self.box_
    }

    #[inline]
    fn stride(&self) -> usize {
        map_stride()
    }

    /// Precalculate maps for `receptor` over `box_` at the given spacing.
    ///
    /// `scoring` decides which pair terms exist; `scoring.atom_weights(atom)`
    /// supplies the per-map multipliers.
    ///
    /// `n_threads == 0` means "use every core". The spacing test is written as
    /// `!(spacing > 0.0)` so a `NaN` spacing is rejected rather than dividing a
    /// dimension by it and allocating an absurd grid.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn precalculate(
        receptor: &Molecule,
        box_: &GridBox,
        scoring: &dyn ScoringFunction,
        spacing: f64,
        n_threads: usize,
    ) -> Result<GridMaps> {
        if !(spacing > 0.0) || spacing > 1.0 {
            return Err(DockError::param(
                "spacing",
                spacing,
                "must be in (0, 1] Ångström",
            ));
        }
        if receptor.is_empty() {
            return Err(DockError::molecule("receptor contains no atoms"));
        }

        // One extra point per axis so the last cell is fully bracketed.
        let dims = estimate_dims(box_, spacing);
        let step = [spacing, spacing, spacing];
        let stride = GRID_TYPE_COUNT * MAPS_PER_TYPE;

        // Reject an impossible grid *before* allocating.
        //
        // The spacing check above only rejects values outside (0, 1]; a spacing
        // of 1e-9 is inside that range and would ask for a 2×10^10-point axis.
        // `dims.iter().product()` then overflows, `vec!` aborts on capacity
        // overflow, and because the release profile is `panic = "abort"` that
        // kills the *host process* — a Python caller gets its interpreter
        // taken down, not an exception. Reporting the cost in megabytes is also
        // more useful than a capacity-overflow message.
        let mut points: u64 = 1;
        for d in dims {
            points = points.saturating_mul(d as u64);
            if points > MAX_GRID_POINTS {
                let mb = (points as f64 * stride as f64 * 4.0) / (1024.0 * 1024.0);
                return Err(DockError::param(
                    "spacing",
                    spacing,
                    format!(
                        "a {spacing} Ångström spacing over a {:?} Ångström box needs \
                         at least {mb:.0} MB of grid; the limit is {MAX_GRID_POINTS} \
                         points ({} MB). Use a coarser spacing or a smaller box.",
                        box_.size(),
                        MAX_GRID_POINTS * stride as u64 * 4 / (1024 * 1024),
                    ),
                ));
            }
        }
        let total = dims[0] * dims[1] * dims[2];
        let mut data = vec![0.0f32; total * stride];

        // Flatten the receptor into typed records so the inner loop stays tight.
        struct RecAtom {
            coord: Vec3,
            type_index: usize,
            /// Interaction radius, Ångström.
            radius: f64,
            /// Which of the three conditional maps this atom contributes to.
            donates: bool,
            accepts: bool,
            apolar: bool,
        }
        let atoms: Vec<RecAtom> = receptor
            .atoms
            .iter()
            .map(|a| RecAtom {
                coord: a.coord,
                type_index: grid_type_index(a.element),
                radius: a.element.interaction_radius(),
                donates: a.can_donate(),
                accepts: a.can_accept(),
                apolar: a.is_apolar(),
            })
            .collect();

        // The spatial kernels, evaluated once per (grid point, receptor atom).
        let kernels: SpatialKernels = scoring.spatial_kernels();

        // Tabulate. A z-slab is a contiguous run of the flat array, so
        // `par_chunks_mut` hands each worker a disjoint slab with no aliasing
        // and no `unsafe` — the crate forbids it outright.
        let slab_len = dims[0] * dims[1] * stride;
        let build = |slab: &mut [f32], iz: usize| {
            debug_assert_eq!(slab.len(), slab_len);
            let z = box_.min[2] + iz as f64 * step[2];
            for iy in 0..dims[1] {
                let y = box_.min[1] + iy as f64 * step[1];
                for ix in 0..dims[0] {
                    let x = box_.min[0] + ix as f64 * step[0];
                    let local = (ix + dims[0] * iy) * stride;
                    for a in &atoms {
                        let dx = x - a.coord[0];
                        let dy = y - a.coord[1];
                        let dz = z - a.coord[2];
                        let r2 = dx * dx + dy * dy + dz * dz;
                        // `kernels.cutoff` is a limit on the *surface*
                        // distance, but `r2` is a real distance, so the real
                        // distance limit is the cutoff plus both radii. Testing
                        // `r2` against the raw cutoff silently amputates the
                        // tail of the Gaussians — by 0.8 Å when every radius was
                        // 0.4 Å, and by ~3.8 Å now that the radii are the real
                        // per-element values.
                        let reach = kernels.cutoff + 2.0 * a.radius;
                        if r2 > reach * reach {
                            continue;
                        }
                        // Surface distance: both atoms inflated by the radius of
                        // the receptor atom's element.
                        let d = r2.sqrt() - 2.0 * a.radius;
                        let c = kernels.eval(d);
                        let t = a.type_index * MAPS_PER_TYPE;
                        slab[local + t + MapSlot::Shape.index()] += c.shape.0 as f32;
                        if a.donates {
                            slab[local + t + MapSlot::HbFromDonor.index()] += c.hbond.0 as f32;
                        }
                        if a.accepts {
                            slab[local + t + MapSlot::HbFromAcceptor.index()] += c.hbond.0 as f32;
                        }
                        if a.apolar {
                            slab[local + t + MapSlot::Hydrophobic.index()] +=
                                c.hydrophobic.0 as f32;
                        }
                    }
                }
            }
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(if n_threads == 0 {
                rayon::current_num_threads()
            } else {
                n_threads
            })
            .build();

        match pool {
            Ok(pool) => pool.install(|| {
                data.par_chunks_mut(slab_len)
                    .enumerate()
                    .for_each(|(iz, slab)| build(slab, iz));
            }),
            Err(_) => data
                .par_chunks_mut(slab_len)
                .enumerate()
                .for_each(|(iz, slab)| build(slab, iz)),
        }

        Ok(GridMaps {
            min: box_.min,
            dims,
            spacing: step,
            data,
            box_: *box_,
            receptor_heavy: receptor
                .atoms
                .iter()
                .filter(|a| a.element != Element::H)
                .map(|a| a.coord)
                .collect(),
        })
    }

    /// Heavy-atom coordinates of the receptor these maps were precalculated
    /// from. Empty only for maps built by hand rather than by
    /// [`GridMaps::precalculate`].
    pub fn receptor_atoms(&self) -> &[[f64; 3]] {
        &self.receptor_heavy
    }

    /// Fractional cell coordinate of `p`, or `None` if it is outside the grid.
    #[inline]
    fn fractional(&self, p: Vec3) -> Option<([f64; 3], [usize; 3])> {
        let u0 = (p[0] - self.min[0]) / self.spacing[0];
        let u1 = (p[1] - self.min[1]) / self.spacing[1];
        let u2 = (p[2] - self.min[2]) / self.spacing[2];
        let i0 = u0.floor();
        let i1 = u1.floor();
        let i2 = u2.floor();
        if i0 < 0.0
            || i1 < 0.0
            || i2 < 0.0
            || i0 as usize >= self.dims[0] - 1
            || i1 as usize >= self.dims[1] - 1
            || i2 as usize >= self.dims[2] - 1
        {
            return None;
        }
        Some((
            [u0 - i0, u1 - i1, u2 - i2],
            [i0 as usize, i1 as usize, i2 as usize],
        ))
    }

    /// Trilinearly interpolate the weighted map for one atom.
    ///
    /// `weights` are the per-slot multipliers for this atom's element and
    /// class. Returns the interpolated energy, or `None` when `p` lies outside
    /// the grid (the caller decides how to penalise that).
    pub fn interpolate(
        &self,
        type_index: usize,
        weights: &[f32; MAPS_PER_TYPE],
        p: Vec3,
    ) -> Option<f64> {
        self.interpolate_with_gradient(type_index, weights, p)
            .map(|(e, _)| e)
    }

    /// Trilinearly interpolate, returning the energy and the exact gradient of
    /// the trilinear interpolant.
    ///
    /// See the module documentation for why the gradient contains only the
    /// interpolation-weight term.
    pub fn interpolate_with_gradient(
        &self,
        type_index: usize,
        weights: &[f32; MAPS_PER_TYPE],
        p: Vec3,
    ) -> Option<(f64, Vec3)> {
        let (frac, cell) = self.fractional(p)?;
        let stride = self.stride();
        let [nx, ny, _] = self.dims;
        let base_idx = cell[0] + nx * (cell[1] + ny * cell[2]);
        let type_base = type_index * MAPS_PER_TYPE;

        // Trilinear weights for the eight corners.
        let (x, y, z) = (frac[0], frac[1], frac[2]);
        let w = [
            (1.0 - x) * (1.0 - y) * (1.0 - z),
            x * (1.0 - y) * (1.0 - z),
            (1.0 - x) * y * (1.0 - z),
            x * y * (1.0 - z),
            (1.0 - x) * (1.0 - y) * z,
            x * (1.0 - y) * z,
            (1.0 - x) * y * z,
            x * y * z,
        ];
        let corner = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];

        let mut energy = 0.0f64;
        let mut grad = [0.0f64; 3];

        for c in 0..8 {
            let idx = base_idx + corner[c][0] + nx * (corner[c][1] + ny * corner[c][2]);
            let value_base = idx * stride;

            // Weighted value of this corner for the probe's map set.
            let mut v = 0.0f64;
            for s in 0..MAPS_PER_TYPE {
                let wi = weights[s] as f64;
                if wi == 0.0 {
                    continue;
                }
                v += wi * self.data[value_base + type_base + s] as f64;
            }
            energy += w[c] * v;

            // ∂w_c/∂p in Cartesian units. With
            //   w_c = (c_x ? x : 1−x)(c_y ? y : 1−y)(c_z ? z : 1−z)
            // the x-derivative is (2c_x−1)·(c_y ? y : 1−y)·(c_z ? z : 1−z)/spacing.
            // Writing the per-axis factor as (2c−1)·u + (1−c) handles both
            // corners uniformly; using (c−1) instead would silently zero the
            // gradient contribution of every c=1 corner.
            let (cx, cy, cz) = (
                corner[c][0] as f64,
                corner[c][1] as f64,
                corner[c][2] as f64,
            );
            let fx = (2.0 * cx - 1.0) * x + (1.0 - cx);
            let fy = (2.0 * cy - 1.0) * y + (1.0 - cy);
            let fz = (2.0 * cz - 1.0) * z + (1.0 - cz);
            let dw0 = (2.0 * cx - 1.0) * fy * fz / self.spacing[0];
            let dw1 = (2.0 * cy - 1.0) * fx * fz / self.spacing[1];
            let dw2 = (2.0 * cz - 1.0) * fx * fy / self.spacing[2];
            grad[0] += dw0 * v;
            grad[1] += dw1 * v;
            grad[2] += dw2 * v;
        }
        Some((energy, grad))
    }

    /// Value of a single raw map at a grid point, for visualisation and tests.
    pub fn raw(&self, type_index: usize, slot: MapSlot, ix: usize, iy: usize, iz: usize) -> f32 {
        let idx = ix + self.dims[0] * (iy + self.dims[1] * iz);
        self.data[idx * self.stride() + type_index * MAPS_PER_TYPE + slot.index()]
    }

    /// The raw map values, for uploading to the GPU.
    ///
    /// The layout is exactly what the WGSL kernel expects, so no repacking is
    /// needed: `[(ix + nx·(iy + ny·iz))·STRIDE + type·SLOTS + slot]`.
    pub fn raw_slice(&self) -> &[f32] {
        &self.data
    }

    /// Write the maps to a `.map` file compatible with AutoDock's convention.
    ///
    /// One file per (element, slot) pair, named
    /// `<basename>.<element><suffix>.map`, e.g. `maps.C.map`, `maps.OA.map`.
    pub fn write_autodock_map_files(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        for t in 0..GRID_TYPE_COUNT {
            for slot in MapSlot::ALL {
                let path = dir.join(format!(
                    "maps.{}{}.map",
                    crate::types::grid_type_name(t),
                    slot_tag(slot)
                ));
                let mut text = String::with_capacity(128);
                text.push_str("GRID_PARAMETER_FILE opendocking\n");
                text.push_str("GRID_DATA_FILE_NONE\n");
                text.push_str("MACROMOLECULE\n");
                text.push_str("SPACING ");
                for s in self.spacing {
                    text.push_str(&format!("{s:.3} "));
                }
                text.push('\n');
                text.push_str(&format!("NELEMENTS {}\n", t + 1));
                text.push_str(&format!(
                    "CENTER {:.3} {:.3} {:.3}\n",
                    self.box_.center()[0],
                    self.box_.center()[1],
                    self.box_.center()[2]
                ));
                let mut data = String::new();
                for iz in 0..self.dims[2] {
                    for iy in 0..self.dims[1] {
                        for ix in 0..self.dims[0] {
                            data.push_str(&format!("{:.4}\n", self.raw(t, slot, ix, iy, iz)));
                        }
                    }
                }
                std::fs::write(path, format!("{text}{data}"))?;
            }
        }
        Ok(())
    }
}

/// Suffix used when writing AutoDock-compatible `.map` files.
fn slot_tag(slot: MapSlot) -> &'static str {
    match slot {
        MapSlot::Shape => "e",
        MapSlot::HbFromDonor => "hbdon",
        MapSlot::HbFromAcceptor => "hbacc",
        MapSlot::Hydrophobic => "hyd",
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::VinaScoring;
    use crate::types::{Atom, AtomKind, AtomType, Element};

    fn scoring() -> VinaScoring {
        VinaScoring::default()
    }

    fn carbon(cx: f64, cy: f64, cz: f64) -> Atom {
        let mut a = Atom::new(1, [cx, cy, cz], Element::C, AtomType::CH);
        a.kind = AtomKind::Hydrophobic;
        a
    }

    fn test_box() -> GridBox {
        GridBox::new([-6.0, -6.0, -6.0], [6.0, 6.0, 6.0]).unwrap()
    }

    #[test]
    fn box_validation() {
        assert!(GridBox::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]).is_ok());
        assert!(GridBox::new([0.0, 0.0, 0.0], [0.0, 10.0, 10.0]).is_err());
        assert!(GridBox::new([0.0, 0.0, 0.0], [-1.0, 10.0, 10.0]).is_err());
        let b = GridBox::centered([1.0, 2.0, 3.0], [10.0, 10.0, 10.0]).unwrap();
        assert!((b.center()[0] - 1.0).abs() < 1e-12);
        assert!((b.size()[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn grid_dims_and_bounds() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let maps = GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.5, 1).unwrap();
        // 12 Å / 0.5 = 24 cells, +1 point.
        assert_eq!(maps.dims, [25, 25, 25]);
        assert!(maps.contains([0.0, 0.0, 0.0]));
        assert!(!maps.contains([100.0, 0.0, 0.0]));
    }

    #[test]
    fn maps_are_finite_and_bounded() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0), carbon(2.0, 0.0, 0.0)]).unwrap();
        let maps = GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.5, 1).unwrap();
        for v in &maps.data {
            assert!(v.is_finite(), "map contains a non-finite value");
        }
    }

    #[test]
    fn interpolation_at_grid_point_matches_stored_value() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let maps = GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.5, 1).unwrap();
        let w = crate::scoring::weights_for_kind(rec.atoms[0].kind);
        let ti = grid_type_index(Element::C);
        // A grid point exactly on the corner must reproduce the stored value.
        let p = [
            maps.min[0] + 5.0 * maps.spacing[0],
            maps.min[1],
            maps.min[2],
        ];
        let e = maps.interpolate(ti, &w, p).expect("inside grid");
        let stored: f64 = (0..3)
            .map(|s| maps.raw(ti, MapSlot::ALL[s], 5, 0, 0) * w[s] as f32)
            .sum::<f32>() as f64;
        assert!((e - stored).abs() < 1e-4, "{e} vs {stored}");
    }

    /// Trilinear interpolation is linear along one axis — but that is a
    /// property of the *operator*, not of any particular field. Testing it
    /// against a tabulated field only checks the field happens to be flat
    /// between the sample points, which stops being true as soon as the
    /// scoring function changes. So compare against the stored node values.
    #[test]
    fn interpolation_is_linear_along_one_axis() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let maps = GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.5, 1).unwrap();
        let w = crate::scoring::weights_for_kind(rec.atoms[0].kind);
        let ti = grid_type_index(Element::C);
        // Sample exactly on grid nodes in y and z, so only x interpolates and
        // the y/z weights are a clean 1.0 on those nodes.
        let (iy, iz) = (3usize, 4usize);
        let y = maps.min[1] + iy as f64 * maps.spacing[1];
        let z = maps.min[2] + iz as f64 * maps.spacing[2];
        let node = |ix: usize| -> f64 {
            (0..4)
                .map(|s| maps.raw(ti, MapSlot::ALL[s], ix, iy, iz) as f64 * w[s] as f64)
                .sum()
        };
        // Nodes 3, 4 and 5 bracket the sample point; the x weights are then
        // 0.25, 0.5, 0.25, and the midpoint must be the plain average.
        let m = maps
            .interpolate(ti, &w, [maps.min[0] + 4.0 * maps.spacing[0], y, z])
            .unwrap();
        assert!(
            (m - 0.5 * (node(3) + node(5))).abs() < 1e-6,
            "midpoint of the bracket: {m} vs {}",
            0.5 * (node(3) + node(5))
        );
        let quarter = maps
            .interpolate(ti, &w, [maps.min[0] + 3.5 * maps.spacing[0], y, z])
            .unwrap();
        assert!(
            (quarter - (0.25 * node(3) + 0.75 * node(4))).abs() < 1e-6,
            "off-node sample: {quarter}"
        );
        // Whatever the field does, a point exactly on a node returns that node.
        let on_node = maps
            .interpolate(ti, &w, [maps.min[0] + 3.0 * maps.spacing[0], y, z])
            .unwrap();
        assert!(
            (on_node - node(3)).abs() < 1e-6,
            "on-node sample: {on_node}"
        );
    }

    #[test]
    fn outside_grid_returns_none() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let maps = GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.5, 1).unwrap();
        let w = crate::scoring::weights_for_kind(rec.atoms[0].kind);
        let ti = grid_type_index(Element::C);
        assert!(maps.interpolate(ti, &w, [100.0, 0.0, 0.0]).is_none());
        assert!(maps.interpolate(ti, &w, [0.0, 0.0, 0.0]).is_some());
    }

    /// The interpolated gradient must match central finite differences. This is
    /// the single most important test in the grid module: it validates the
    /// interpolation weights and the chain rule that turns them into a
    /// Cartesian gradient.
    ///
    /// Probe points are chosen strictly *inside* a cell. A trilinear
    /// interpolant is only C⁰, so a point exactly on a cell boundary has a
    /// genuine kink and a central difference across it would disagree with
    /// either one-sided derivative — that is a property of the method, not a
    /// bug, and it is what the module documentation warns about.
    #[test]
    fn interpolated_gradient_matches_finite_difference() {
        let rec = Molecule::from_atoms(vec![
            carbon(0.0, 0.0, 0.0),
            carbon(1.7, 0.3, -0.4),
            carbon(-0.9, 1.2, 0.8),
        ])
        .unwrap();
        let maps = GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.375, 1).unwrap();
        let probe = rec.atoms[0].clone();
        let w = crate::scoring::weights_for_kind(probe.kind);
        let ti = grid_type_index(Element::C);

        // The fractional offsets 0.37/0.61/0.23 keep every probe strictly
        // inside its cell, clear of the boundary kink.
        let s = maps.spacing[0];
        let point_at = |i: usize, j: usize, k: usize| {
            [
                maps.min[0] + (i as f64 + 0.37) * s,
                maps.min[1] + (j as f64 + 0.61) * s,
                maps.min[2] + (k as f64 + 0.23) * s,
            ]
        };
        let points = [
            point_at(12, 15, 16),
            point_at(17, 20, 13),
            point_at(9, 22, 18),
            point_at(15, 18, 11),
        ];
        for p in points {
            // Every coordinate must be off-grid in every axis.
            for k in 0..3 {
                let frac = (p[k] - maps.min[k]) / maps.spacing[k];
                let f = frac - frac.floor();
                assert!(
                    f > 0.2 && f < 0.8,
                    "test point {p:?} axis {k} is too close to a cell boundary ({f})"
                );
            }
            let (_e, g) = maps
                .interpolate_with_gradient(ti, &w, p)
                .expect("probe inside grid");

            let h = 1e-6;
            for k in 0..3 {
                let mut pp = p;
                let mut pm = p;
                pp[k] += h;
                pm[k] -= h;
                let numeric = (maps.interpolate(ti, &w, pp).unwrap()
                    - maps.interpolate(ti, &w, pm).unwrap())
                    / (2.0 * h);
                assert!(
                    (numeric - g[k]).abs() < 1e-3,
                    "axis {k} at {p:?}: numeric {numeric:.6} vs analytic {:.6}",
                    g[k]
                );
            }
        }
    }

    #[test]
    fn empty_receptor_is_rejected() {
        let rec = Molecule::from_atoms(Vec::new());
        assert!(rec.is_err());
    }

    #[test]
    fn bad_spacing_is_rejected() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        assert!(GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.0, 1).is_err());
        assert!(GridMaps::precalculate(&rec, &test_box(), &scoring(), 2.0, 1).is_err());
    }
}
