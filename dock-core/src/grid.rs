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
//! The element dimension is the **probe's** type, not a filter on the receptor.
//! Each block is the field for a hypothetical ligand atom *of that element*,
//! tabulated over every receptor atom with the two radii added, so a ligand
//! nitrogen's block holds what an oxygen receptor does to a nitrogen. Every
//! receptor atom writes into all ten blocks. That is what the "every atom"
//! column above means, and it is the one thing this table cannot express on its
//! own: a block is indexed by who is asking, not by who is answering.
//!
//! Getting it wrong is silent, which is why `a_receptor_atom_reaches_every_probe_type`
//! exists. An earlier version wrote each receptor atom only into its own
//! element's block, which made the element index a filter: cross-element pairs
//! scored **exactly zero at every separation**, so a nitrogen donor formed no
//! hydrogen bond with an oxygen acceptor, and — since `Shape` was partitioned
//! the same way — a ligand atom felt **no steric repulsion at all** against any
//! element missing from that partition.
//!
//! AutoDock Vina collapses the two hydrogen-bond maps into one, which lets a
//! donor–donor pair pick up a spurious hydrogen bond. Splitting them costs one
//! extra map per point and makes the decomposition exact — see
//! [`crate::scoring`] for the full rationale. It is exact for the *pair* formula;
//! the intramolecular sum calls `pair_energy` on atom kinds rather than reading
//! these maps, so the two halves of a total energy describe a hydrogen bond the
//! same way only now that the blocks are not filtered by element.
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
use crate::types::{
    grid_type_radius, Element, Molecule, Vec3, GRID_TYPE_COUNT, MAX_GRID_TYPE_RADIUS,
};

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
                        // `kernels.cutoff` is a limit on the *surface* distance,
                        // so the real-distance limit is the cutoff plus both
                        // radii. Testing `r2` against the raw cutoff silently
                        // amputates the tail of the Gaussians — by 0.8 Å when
                        // every radius was 0.4 Å, and by ~3.8 Å now that the
                        // radii are the real per-element values. This is the
                        // widest of the ten per-probe-type limits, used once so
                        // the square root below is computed once too.
                        let far = kernels.cutoff + MAX_GRID_TYPE_RADIUS + a.radius;
                        // One block per *probe* element, and every block is summed
                        // over **every** receptor atom. That is what the module
                        // table above says — the shape map is "written by every
                        // atom" — and it can only be true if an atom writes into
                        // all ten blocks rather than into its own.
                        //
                        // It used to write only into `a.type_index` and evaluate
                        // `d` as `r - 2 * a.radius`, which silently assumed the
                        // probe was the same element as the receptor atom. A
                        // ligand nitrogen therefore felt no oxygen at all: no
                        // hydrogen bond, and — far worse — **no steric
                        // repulsion either**, since `Shape` was partitioned the
                        // same way. A ligand could be pushed straight through the
                        // protein's oxygens and nitrogens and only feel carbon.
                        //
                        // The surface distance is therefore a function of both
                        // radii, and the probe's radius comes from
                        // `grid_type_radius` rather than from the receptor atom.
                        // The cutoff test moved inside the loop with it, and the
                        // single outer test below is the widest of the ten.
                        if r2 > far * far {
                            continue;
                        }
                        let r = r2.sqrt();
                        for p in 0..GRID_TYPE_COUNT {
                            let rp = grid_type_radius(p);
                            let d = r - (rp + a.radius);
                            if d > kernels.cutoff {
                                continue;
                            }
                            let c = kernels.eval(d);
                            let t = p * MAPS_PER_TYPE;
                            slab[local + t + MapSlot::Shape.index()] += c.shape.0 as f32;
                            if a.donates {
                                slab[local + t + MapSlot::HbFromDonor.index()] += c.hbond.0 as f32;
                            }
                            if a.accepts {
                                slab[local + t + MapSlot::HbFromAcceptor.index()] +=
                                    c.hbond.0 as f32;
                            }
                            if a.apolar {
                                slab[local + t + MapSlot::Hydrophobic.index()] +=
                                    c.hydrophobic.0 as f32;
                            }
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

    /// Trilinearly interpolate each of the four map slots separately.
    ///
    /// This is the **per-slot** decomposition, and it is a different question
    /// from the per-*term* one that [`crate::scoring::TermMaps`] answers. The
    /// four slots are what the grid physically stores; the five terms are what
    /// the scoring function is written in terms of. They do not correspond:
    /// slot 0 fuses `g1`, `g2` and `rep` into one `f32`, and the hydrogen bond
    /// is one term spread across **two** slots, one per receptor polarity. So
    /// four slots are not four terms in either direction, and a caller wanting
    /// "how much of this score is the hydrogen bond" cannot get it from here.
    ///
    /// What this *is* good for is attribution by map role: which of the four
    /// fields a given probe actually drew on, and by how much.
    ///
    /// The four values sum to exactly what [`Self::interpolate`] returns at the
    /// same point, because that is the same accumulation with the slot loop
    /// peeled apart — `interpolate_by_slot_sums_to_the_total` holds them to it.
    pub fn interpolate_by_slot(
        &self,
        type_index: usize,
        weights: &[f32; MAPS_PER_TYPE],
        p: Vec3,
    ) -> Option<[f64; MAPS_PER_TYPE]> {
        let (frac, cell) = self.fractional(p)?;
        let stride = self.stride();
        let [nx, ny, _] = self.dims;
        let base_idx = cell[0] + nx * (cell[1] + ny * cell[2]);
        let type_base = type_index * MAPS_PER_TYPE;
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
        let mut out = [0.0f64; MAPS_PER_TYPE];
        for c in 0..8 {
            let idx = base_idx + corner[c][0] + nx * (corner[c][1] + ny * corner[c][2]);
            let vb = idx * stride + type_base;
            for s in 0..MAPS_PER_TYPE {
                let wi = weights[s] as f64;
                if wi == 0.0 {
                    continue;
                }
                out[s] += w[c] * wi * self.data[vb + s] as f64;
            }
        }
        Some(out)
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

/// A second tabulation over the same geometry, carrying one field per Vina term.
///
/// # Why this is not a `GridMaps` with more slots
///
/// The production map's `Shape` slot is `w1·g1 + w2·g2 + w_rep·rep` summed and
/// rounded into a single `f32`, so the three terms cannot be recovered from it
/// afterwards — the information is gone, not merely unexposed. Widening
/// `MAPS_PER_TYPE` would change the flat stride, and the WGSL kernel hard-codes
/// that stride with a test asserting the two agree, so the split cannot be done
/// in place. This is therefore a parallel structure over the same box, built on
/// demand: the docking path never allocates one.
///
/// # What it costs
///
/// [`TERM_STRIDE`] = `GRID_TYPE_COUNT · TERM_FIELDS` = 60 `f32` per point,
/// against the production map's 40 — 1.5× the memory for a diagnostic. It is
/// a separate call precisely so that cost is opt-in.
///
/// See [`crate::scoring`] for where the weights are applied and why every term
/// here is already weighted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TermMaps {
    /// Lower corner of the grid, equal to [`GridBox::min`].
    pub min: Vec3,
    /// Number of grid points along each axis.
    pub dims: [usize; 3],
    /// Spacing between grid points, in Ångström.
    pub spacing: [f64; 3],
    /// Flat storage, laid out as
    /// `[((ix + nx·(iy + ny·iz)) · TERM_STRIDE) + type · TERM_FIELDS + field]`.
    data: Vec<f32>,
    /// The box this grid covers.
    pub box_: GridBox,
}

/// Number of `f32` values stored per grid point by [`TermMaps`].
pub const fn term_stride() -> usize {
    GRID_TYPE_COUNT * crate::scoring::TERM_FIELDS
}

impl TermMaps {
    /// The only backend that can produce a
    /// [`crate::scoring::TermBreakdown`]: **the CPU**.
    ///
    /// Exposed as a constant on the type that *has* to be built, so a caller
    /// can ask the question before paying 1.5x the memory rather than after
    /// reading a number that was quietly produced somewhere else. The same
    /// fact is carried on every [`crate::scoring::TermBreakdown`], in its
    /// `backend` field, so the answer cannot be separated from the value.
    ///
    /// The reason it is a fact rather than a convention is a type: the WGSL
    /// entry point is `GpuContext::score(&Batch, &GridMaps)`. It takes a
    /// [`GridMaps`] and there is no overload, no generic and no trait that
    /// would let a [`TermMaps`] be passed instead, so on a `gpu` build there is
    /// no path by which these five terms reach a device.
    pub const BACKEND: crate::scoring::TermBackend = crate::scoring::TermBackend::Cpu;

    /// Tabulate one field per Vina term over `box_`, on the same geometry
    /// [`GridMaps::precalculate`] would use.
    ///
    /// The loop is deliberately a copy of the production one — same radii, same
    /// cutoff test, same per-probe-type blocks — with six accumulators in place
    /// of four. Writing it as a call into the production loop is not possible:
    /// that loop's shape is that `Shape` is one sum, which is the thing being
    /// undone here. What keeps the two honest is
    /// `term_maps_agree_with_grid_maps_slot_by_slot`, which asserts the two
    /// agree on every field the two share.
    ///
    /// The spacing test is written as `!(spacing > 0.0)` for the reason
    /// [`GridMaps::precalculate`] gives: it is the only form that also rejects
    /// a `NaN` spacing, which would otherwise sail through and ask for an
    /// absurd grid.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn precalculate(
        receptor: &Molecule,
        box_: &GridBox,
        scoring: &dyn crate::scoring::ScoringFunction,
        spacing: f64,
        n_threads: usize,
    ) -> Result<TermMaps> {
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
        let dims = estimate_dims(box_, spacing);
        let step = [spacing, spacing, spacing];
        let stride = term_stride();
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
                         at least {mb:.0} MB of term maps; the limit is \
                         {MAX_GRID_POINTS} points",
                        box_.size(),
                    ),
                ));
            }
        }
        let total = dims[0] * dims[1] * dims[2];
        let mut data = vec![0.0f32; total * stride];

        struct RecAtom {
            coord: Vec3,
            radius: f64,
            mask: [bool; crate::scoring::TERM_FIELDS],
        }
        let atoms: Vec<RecAtom> = receptor
            .atoms
            .iter()
            .map(|a| RecAtom {
                coord: a.coord,
                radius: a.element.interaction_radius(),
                mask: crate::scoring::receptor_term_mask(a),
            })
            .collect();

        let kernels = scoring.spatial_kernels();
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
                        let far = kernels.cutoff + MAX_GRID_TYPE_RADIUS + a.radius;
                        if r2 > far * far {
                            continue;
                        }
                        let r = r2.sqrt();
                        for p in 0..GRID_TYPE_COUNT {
                            let rp = grid_type_radius(p);
                            let d = r - (rp + a.radius);
                            if d > kernels.cutoff {
                                continue;
                            }
                            let (v, _) = kernels.eval_terms(d);
                            let t = p * crate::scoring::TERM_FIELDS;
                            for f in 0..crate::scoring::TERM_FIELDS {
                                if a.mask[f] {
                                    slab[local + t + f] += v[f] as f32;
                                }
                            }
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

        Ok(TermMaps {
            min: box_.min,
            dims,
            spacing: step,
            data,
            box_: *box_,
        })
    }

    /// Number of grid points along each axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Spacing in Ångström.
    pub fn spacing(&self) -> f64 {
        self.spacing[0]
    }

    /// Number of `f32` values held.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// The box this grid covers.
    pub fn grid_box(&self) -> GridBox {
        self.box_
    }

    /// Raw value of one field at a grid point, for tests and diagnostics.
    pub fn raw(
        &self,
        type_index: usize,
        field: crate::scoring::TermField,
        ix: usize,
        iy: usize,
        iz: usize,
    ) -> f32 {
        let idx = ix + self.dims[0] * (iy + self.dims[1] * iz);
        self.data[idx * term_stride() + type_index * crate::scoring::TERM_FIELDS + field.index()]
    }

    /// The per-term energy a probe of `type_index` reads at `p`, in kcal/mol.
    ///
    /// `mask` is [`crate::scoring::probe_term_mask`] for the probe atom, so the
    /// hydrogen-bond halves are selected by the probe's class exactly as the
    /// production interpolation selects its slots. Returns `None` when `p` is
    /// outside the grid, matching [`GridMaps::interpolate`].
    pub fn terms_at(
        &self,
        type_index: usize,
        mask: &[bool; crate::scoring::TERM_FIELDS],
        p: Vec3,
    ) -> Option<crate::scoring::TermBreakdown> {
        let (frac, cell) = self.fractional(p)?;
        let stride = term_stride();
        let [nx, ny, _] = self.dims;
        let base_idx = cell[0] + nx * (cell[1] + ny * cell[2]);
        let type_base = type_index * crate::scoring::TERM_FIELDS;
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
        let mut acc = [0.0f64; crate::scoring::TERM_FIELDS];
        for c in 0..8 {
            let idx = base_idx + corner[c][0] + nx * (corner[c][1] + ny * corner[c][2]);
            let vb = idx * stride + type_base;
            for f in 0..crate::scoring::TERM_FIELDS {
                if mask[f] {
                    acc[f] += w[c] * self.data[vb + f] as f64;
                }
            }
        }
        use crate::scoring::TermField as F;
        let from_donor = acc[F::HbFromDonor.index()];
        let from_acceptor = acc[F::HbFromAcceptor.index()];
        Some(crate::scoring::TermBreakdown {
            backend: Self::BACKEND,
            gauss1: acc[F::Gauss1.index()],
            gauss2: acc[F::Gauss2.index()],
            repulsion: acc[F::Repulsion.index()],
            hbond: from_donor + from_acceptor,
            hydrophobic: acc[F::Hydrophobic.index()],
            hbond_from_donor: from_donor,
            hbond_from_acceptor: from_acceptor,
            out_of_box_penalty: 0.0,
        })
    }

    /// The per-term decomposition of a whole conformation's intermolecular
    /// energy, in kcal/mol.
    ///
    /// `coords` are the world-space coordinates forward kinematics produced for
    /// the conformation. An atom that left the grid contributes
    /// [`crate::search::OUT_OF_BOX_PENALTY`] to
    /// [`crate::scoring::TermBreakdown::out_of_box_penalty`] rather than to any
    /// term, so the penalty cannot be mistaken for a score.
    pub fn conformation_terms(
        &self,
        type_index: &[usize],
        masks: &[[bool; crate::scoring::TERM_FIELDS]],
        coords: &[Vec3],
    ) -> crate::scoring::TermBreakdown {
        let mut acc = crate::scoring::TermBreakdown::default();
        for i in 0..coords.len() {
            match self.terms_at(type_index[i], &masks[i], coords[i]) {
                Some(t) => {
                    acc.gauss1 += t.gauss1;
                    acc.gauss2 += t.gauss2;
                    acc.repulsion += t.repulsion;
                    acc.hbond += t.hbond;
                    acc.hydrophobic += t.hydrophobic;
                    acc.hbond_from_donor += t.hbond_from_donor;
                    acc.hbond_from_acceptor += t.hbond_from_acceptor;
                }
                None => acc.out_of_box_penalty += crate::search::OUT_OF_BOX_PENALTY,
            }
        }
        acc
    }

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
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::VinaScoring;
    use crate::types::{grid_type_index, grid_type_name, Atom, AtomKind, AtomType, Element};

    fn scoring() -> VinaScoring {
        VinaScoring::default()
    }

    fn carbon(cx: f64, cy: f64, cz: f64) -> Atom {
        let mut a = Atom::new(1, [cx, cy, cz], Element::C, AtomType::CH);
        a.kind = AtomKind::Hydrophobic;
        a
    }

    /// A receptor oxygen that accepts and nothing else. A lone acceptor, so
    /// anything a test sees in a block came from this atom and not from a
    /// neighbour's classification.
    fn oxygen(cx: f64, cy: f64, cz: f64) -> Atom {
        let mut a = Atom::new(1, [cx, cy, cz], Element::O, AtomType::OA);
        a.kind = AtomKind::Acceptor;
        a
    }

    /// Indices of the stored grid point nearest a world position, clamped into
    /// the grid so a query outside the box returns an edge rather than an index
    /// that would read past the array.
    fn nearest(maps: &GridMaps, p: [f64; 3]) -> (usize, usize, usize) {
        let at = |lo: f64, h: f64, want: f64, n: usize| {
            (((want - lo) / h).round().max(0.0) as usize).min(n.saturating_sub(1))
        };
        (
            at(maps.min[0], maps.spacing[0], p[0], maps.dims[0]),
            at(maps.min[1], maps.spacing[1], p[1], maps.dims[1]),
            at(maps.min[2], maps.spacing[2], p[2], maps.dims[2]),
        )
    }

    /// World position of a stored grid point, for recomputing what belongs there.
    fn point_of(maps: &GridMaps, i: (usize, usize, usize)) -> [f64; 3] {
        [
            maps.min[0] + i.0 as f64 * maps.spacing[0],
            maps.min[1] + i.1 as f64 * maps.spacing[1],
            maps.min[2] + i.2 as f64 * maps.spacing[2],
        ]
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

    /// Every receptor atom must reach **every** probe type's block.
    ///
    /// This is the assertion that says the element dimension of the grid is the
    /// *probe* type and not a filter on the receptor, which is what the module
    /// documentation's own table claims: the shape map is "written by every
    /// atom". It did not hold. An atom wrote only into its own element's block
    /// and the surface distance was `r − 2·R_self`, silently assuming the probe
    /// was the same element as the receptor atom. A ligand nitrogen therefore
    /// felt no oxygen at all — no hydrogen bond, and, because `Shape` was
    /// partitioned the same way, **no steric repulsion either**.
    ///
    /// The expected value is recomputed from the kernels rather than written as
    /// a literal, so this pins the *semantics* (which two radii, which slot) and
    /// not one number that could be re-derived and pasted.
    #[test]
    fn a_receptor_atom_reaches_every_probe_type() {
        let rec = Molecule::from_atoms(vec![oxygen(0.0, 0.0, 0.0)]).unwrap();
        let maps = GridMaps::precalculate(&rec, &test_box(), &scoring(), 0.375, 1).unwrap();
        let kernels = scoring().spatial_kernels();

        // A separation inside the hydrogen-bond window for the published radii
        // (R_N + R_O = 3.35 Å), so the term is unambiguously non-zero.
        let i = nearest(&maps, [3.0, 0.0, 0.0]);
        let p = point_of(&maps, i);
        let r = p[0].abs().max(p[1].abs()).max(p[2].abs());

        // Every block, including the ones whose probe element has no business
        // being anywhere near an oxygen, must carry this atom's shape term.
        for probe_type in 0..GRID_TYPE_COUNT {
            let shape = maps.raw(probe_type, MapSlot::Shape, i.0, i.1, i.2) as f64;
            assert!(
                (shape
                    - kernels
                        .eval(r - (grid_type_radius(probe_type) + 1.60))
                        .shape
                        .0)
                    .abs()
                    < 1e-5,
                "block {probe_type} ({} Å) holds {shape}, so a receptor oxygen is \
                 invisible to that probe type",
                grid_type_name(probe_type)
            );
        }

        // And the hydrogen-bond block must hold the value for *that probe's*
        // radius — so the blocks differ from one another, which is the whole
        // point: a nitrogen probe and a carbon probe 3 Å from the same oxygen
        // are at different surface distances and must not read the same number.
        //
        // What is stored is the *weighted* kernel output, `w_hb · hbond(d)`, so
        // a real bond is **negative** — which is why a lone oxygen's acceptor
        // slot bottoms out at exactly the `hb` weight rather than at 1.0.
        let hb_n = maps.raw(
            grid_type_index(Element::N),
            MapSlot::HbFromAcceptor,
            i.0,
            i.1,
            i.2,
        );
        let hb_c = maps.raw(
            grid_type_index(Element::C),
            MapSlot::HbFromAcceptor,
            i.0,
            i.1,
            i.2,
        );
        assert!(
            hb_n < 0.0,
            "an oxygen acceptor must attract a nitrogen donor through the grid, \
             but the nitrogen block holds {hb_n}"
        );
        assert!(
            (hb_n as f64 - kernels.eval(r - (1.75 + 1.60)).hbond.0).abs() < 1e-5,
            "nitrogen block holds {hb_n}, not the value for R_N + R_O"
        );
        assert!(
            (hb_c as f64 - kernels.eval(r - (1.90 + 1.60)).hbond.0).abs() < 1e-5,
            "carbon block holds {hb_c}, not the value for R_C + R_O"
        );
        assert_ne!(
            hb_n, hb_c,
            "two probe elements at the same point read the same hydrogen-bond \
             value, so the block is not per-probe-type at all"
        );

        // Reverse direction: an oxygen is not apolar, so no block may claim a
        // hydrophobic contact. A guard that only checked "non-zero" would pass
        // this even if every slot were filled with everything.
        for probe_type in 0..GRID_TYPE_COUNT {
            assert_eq!(
                maps.raw(probe_type, MapSlot::Hydrophobic, i.0, i.1, i.2),
                0.0,
                "block {probe_type} claims a hydrophobic contact with an oxygen"
            );
            assert_eq!(
                maps.raw(probe_type, MapSlot::HbFromDonor, i.0, i.1, i.2),
                0.0,
                "block {probe_type} claims a hydrogen bond *from* a receptor \
                 acceptor, which cannot donate"
            );
        }
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
        assert!(TermMaps::precalculate(&rec, &test_box(), &scoring(), 0.0, 1).is_err());
        assert!(TermMaps::precalculate(&rec, &test_box(), &scoring(), 2.0, 1).is_err());
    }

    // -----------------------------------------------------------------
    // The per-term decomposition
    // -----------------------------------------------------------------

    /// A receptor with one atom of each of the four interaction classes, so a
    /// decomposition test can see every term populated by something.
    fn mixed_receptor() -> Molecule {
        Molecule::from_atoms(vec![
            carbon(0.0, 0.0, 0.0),
            oxygen(4.0, 0.0, 0.0),
            donor(0.0, 4.0, 0.0),
            carbon(0.0, 0.0, 4.0),
        ])
        .unwrap()
    }

    /// A receptor nitrogen that donates and nothing else.
    fn donor(cx: f64, cy: f64, cz: f64) -> Atom {
        let mut a = Atom::new(1, [cx, cy, cz], Element::N, AtomType::NA);
        a.kind = AtomKind::Donor;
        a
    }

    /// An atom of the given class, for use as a probe.
    fn probe_atom(kind: AtomKind) -> Atom {
        let mut a = Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH);
        a.kind = kind;
        a
    }

    /// **The load-bearing identity: the five terms add up to the number the
    /// engine already returned.**
    ///
    /// This is the assertion the whole decomposition exists to support, and it
    /// is checked against [`GridMaps::interpolate`], not against
    /// `TermBreakdown::terms_total` — the latter would be a tautology, since
    /// `terms_total` is the same addition the test would perform. The reference
    /// has to be a number produced by a different route.
    ///
    /// **Why the bound is not zero, stated rather than assumed.** The production
    /// map rounds `w1·g1 + w2·g2 + w_rep·rep` into one `f32`; the term maps round
    /// each of the three separately. So the two disagree by the storage
    /// granularity of a `f32`, and pretending otherwise would make this test
    /// assert a property the data layout cannot have. `1e-4` is four orders of
    /// magnitude above that granularity and four below the 0.375 Å grid's own
    /// interpolation error, so it discriminates a real term change (which moves
    /// this by ~1e-1) while tolerating only the rounding it has to.
    #[test]
    fn the_five_terms_sum_to_the_engines_own_intermolecular_energy() {
        let rec = mixed_receptor();
        let box_ = test_box();
        let sc = scoring();
        let maps = GridMaps::precalculate(&rec, &box_, &sc, 0.375, 1).unwrap();
        let terms = TermMaps::precalculate(&rec, &box_, &sc, 0.375, 1).unwrap();
        assert_eq!(maps.dims(), terms.dims(), "the two must share a geometry");

        let mut worst: f64 = 0.0;
        let mut where_: Vec3 = [0.0; 3];
        for kind in [
            AtomKind::Hydrophobic,
            AtomKind::Donor,
            AtomKind::Acceptor,
            AtomKind::Other,
        ] {
            let a = probe_atom(kind);
            let ti = grid_type_index(a.element);
            let mask = crate::scoring::probe_term_mask(&a);
            let w = crate::scoring::weights_for_kind(kind);
            for k in 0..9 {
                let p = [1.5 + 0.5 * k as f64, -2.25, 0.75];
                let production = maps.interpolate(ti, &w, p).expect("inside grid");
                let t = terms.terms_at(ti, &mask, p).expect("inside grid");
                let delta = (t.terms_total() - production).abs();
                if delta > worst {
                    worst = delta;
                    where_ = p;
                }
            }
        }
        assert!(
            worst < 1e-4,
            "terms sum {worst:e} away from the production interpolation at {where_:?}, \
             which is above the f32 storage granularity that is the only reason \
             these two should ever differ"
        );
    }

    /// The two tabulations must agree on every field they share, and the
    /// hydrogen bond must be **split** in the term maps where the production map
    /// has two slots and the terms have one.
    ///
    /// This is what makes the sum identity above a statement about the *terms*
    /// rather than about two arbitrary numbers: it says the term maps are the
    /// same tabulation, unfolded.
    #[test]
    fn term_maps_agree_with_grid_maps_slot_by_slot() {
        use crate::scoring::TermField as F;
        let rec = mixed_receptor();
        let box_ = test_box();
        let sc = scoring();
        let maps = GridMaps::precalculate(&rec, &box_, &sc, 0.375, 1).unwrap();
        let terms = TermMaps::precalculate(&rec, &box_, &sc, 0.375, 1).unwrap();

        let mut worst_shape: f64 = 0.0;
        let mut worst_hb: f64 = 0.0;
        for ty in 0..GRID_TYPE_COUNT {
            for ix in 0..maps.dims()[0] {
                for iy in 0..maps.dims()[1] {
                    for iz in 0..maps.dims()[2] {
                        let shape = maps.raw(ty, MapSlot::Shape, ix, iy, iz) as f64;
                        let split = (terms.raw(ty, F::Gauss1, ix, iy, iz)
                            + terms.raw(ty, F::Gauss2, ix, iy, iz)
                            + terms.raw(ty, F::Repulsion, ix, iy, iz))
                            as f64;
                        worst_shape = worst_shape.max((shape - split).abs());
                        for slot in [MapSlot::HbFromDonor, MapSlot::HbFromAcceptor] {
                            let field = if slot == MapSlot::HbFromDonor {
                                F::HbFromDonor
                            } else {
                                F::HbFromAcceptor
                            };
                            let v = maps.raw(ty, slot, ix, iy, iz) as f64;
                            let t = terms.raw(ty, field, ix, iy, iz) as f64;
                            worst_hb = worst_hb.max((v - t).abs());
                        }
                    }
                }
            }
        }
        // Both differences are f32 rounding of the same quantity: the shape
        // field is one rounding of a three-term sum, the split is three
        // roundings. See the sum-identity test for why this is not zero.
        assert!(worst_shape < 1e-3, "shape split off by {worst_shape:e}");
        assert!(worst_hb < 1e-6, "hbond field off by {worst_hb:e}");
    }

    /// A single term being wrong must move **that term** and nothing else.
    ///
    /// Without this, a decomposition nobody compares is decoration: a bug that
    /// doubled `g1` and halved `rep` would leave every total-based check in the
    /// project green, because the sum is what the search optimises. The
    /// assertion is built by subtracting: with the other four terms held fixed,
    /// the reported `gauss1` is the whole of the difference.
    #[test]
    fn one_wrong_term_moves_only_that_term() {
        use crate::scoring::TermField as F;
        let rec = mixed_receptor();
        let box_ = test_box();
        // A scoring function whose `g1` is ten times too attractive.
        let mut w = crate::scoring::VinaWeights::default();
        w.gauss1 *= 10.0;
        let wrong = crate::scoring::VinaScoring { weights: w };
        let right = scoring();
        let p = [0.9, -1.1, 0.4];
        let a = probe_atom(AtomKind::Hydrophobic);
        let ti = grid_type_index(a.element);
        let mask = crate::scoring::probe_term_mask(&a);

        let mut diffs = [0.0f64; 5];
        let names = ["g1", "g2", "rep", "hb", "hyd"];
        let mut reference = [0.0f64; 5];
        for (which, sc) in [("right", right), ("wrong", wrong)] {
            let terms = TermMaps::precalculate(&rec, &box_, &sc, 0.375, 1).unwrap();
            let t = terms.terms_at(ti, &mask, p).expect("inside grid");
            let got = [t.gauss1, t.gauss2, t.repulsion, t.hbond, t.hydrophobic];
            if which == "right" {
                // The *signed* value. Taking the magnitude here would make the
                // subtraction below compare a negative term against its own
                // negation, and every term but the one under test would then
                // read as "moved" — a false red that would have had to be
                // silenced with a tolerance, which is the one thing a guard in
                // this position must never need.
                reference = got;
            } else {
                for i in 0..5 {
                    diffs[i] = (got[i] - reference[i]).abs();
                }
            }
        }
        // g1 moved, and it moved by roughly its own magnitude; nothing else did.
        let _ = F::Gauss1;
        let d1 = diffs[0];
        assert!(d1 > 1e-2, "perturbing g1 must move g1, got {d1:e}");
        for i in 1..5 {
            assert!(
                diffs[i] < 1e-6,
                "perturbing g1 also moved {} by {e}, and a term that moves when \
                 another one is wrong is a term nothing can be checked against",
                names[i],
                e = diffs[i]
            );
        }
    }

    /// The per-slot reading must add up to the total the same map returns, and
    /// must **not** be mistaken for a per-term one.
    ///
    /// The second half is the requirement-4 claim: slot 0 is the *fused* shape
    /// field, so a caller who wants `g1` cannot have it here, and the hydrogen
    /// bond is spread over two slots. Both are asserted rather than described,
    /// because "four slots, five terms" is exactly the kind of sentence that
    /// rots into a false equivalence.
    #[test]
    fn interpolate_by_slot_sums_to_the_total() {
        let rec = mixed_receptor();
        let box_ = test_box();
        let maps = GridMaps::precalculate(&rec, &box_, &scoring(), 0.375, 1).unwrap();
        for kind in [AtomKind::Hydrophobic, AtomKind::Donor, AtomKind::Acceptor] {
            let a = probe_atom(kind);
            let ti = grid_type_index(a.element);
            let w = crate::scoring::weights_for_kind(kind);
            for k in 0..7 {
                let p = [1.0 + 0.7 * k as f64, -2.0, 0.5];
                let by_slot = maps.interpolate_by_slot(ti, &w, p).expect("inside");
                let total = maps.interpolate(ti, &w, p).expect("inside");
                let summed: f64 = by_slot.iter().sum();
                assert!(
                    (summed - total).abs() < 1e-12,
                    "slots sum to {summed} against a total of {total}"
                );
                // A donor reads the acceptor slot and not the donor slot; that
                // is the whole of its selectivity, and it is a property of the
                // slot, not of the element.
                if kind == AtomKind::Donor {
                    assert_eq!(by_slot[MapSlot::HbFromDonor.index()], 0.0);
                }
            }
        }
        // The two halves of the claim: four slots are not the five terms. Slot 0
        // carries three of them, so no per-term value is recoverable from it.
        let terms = TermMaps::precalculate(&rec, &box_, &scoring(), 0.375, 1).unwrap();
        let a = probe_atom(AtomKind::Hydrophobic);
        let ti = grid_type_index(a.element);
        let p = [0.9, -1.1, 0.4];
        let t = terms
            .terms_at(ti, &crate::scoring::probe_term_mask(&a), p)
            .expect("inside");
        let shape = maps
            .interpolate_by_slot(
                ti,
                &crate::scoring::weights_for_kind(AtomKind::Hydrophobic),
                p,
            )
            .expect("inside")[MapSlot::Shape.index()];
        assert!(
            (t.shape() - shape).abs() < 1e-4,
            "slot 0 {} against the three shape terms {}",
            shape,
            t.shape()
        );
        // ... and slot 0 is strictly one number, so it cannot be the three.
        assert!(t.gauss1 != 0.0 && t.gauss2 != 0.0 && t.repulsion != 0.0);
    }

    /// An atom outside the grid must be reported as a penalty, not as a term.
    ///
    /// Folding it into the nearest term would make that term's value depend on
    /// whether the pose happened to be inside the box, and the sum would still
    /// come out right — which is precisely how a wrong decomposition can look
    /// correct.
    #[test]
    fn an_atom_outside_the_grid_is_a_penalty_not_a_term() {
        let rec = mixed_receptor();
        let box_ = test_box();
        let terms = TermMaps::precalculate(&rec, &box_, &scoring(), 0.375, 1).unwrap();
        let a = probe_atom(AtomKind::Hydrophobic);
        let ti = grid_type_index(a.element);
        let mask = crate::scoring::probe_term_mask(&a);
        let b = terms.conformation_terms(&[ti], &[mask], &[[100.0, 0.0, 0.0]]);
        assert_eq!(b.terms_total(), 0.0, "no term should be populated");
        assert_eq!(b.out_of_box_penalty, crate::search::OUT_OF_BOX_PENALTY);
        assert!((b.total() - crate::search::OUT_OF_BOX_PENALTY).abs() < 1e-12);

        // An inside atom contributes terms and no penalty.
        let g = terms.conformation_terms(&[ti], &[mask], &[[0.0, 0.0, 0.0]]);
        assert_eq!(g.out_of_box_penalty, 0.0);
        assert!(g.terms_total() != 0.0);
    }
}

/// The grid-side scoring constants, pinned against `docs/SCORING.md`.
///
/// The same two-part contract as `scoring::pinned_constants`: the equality, and
/// the documented line in the failure message so a constant and the sentence
/// that states it cannot drift apart. Three of these have **no line in
/// `SCORING.md`**, and their failure messages say so rather than quoting a
/// line that is not there.
#[cfg(test)]
mod pinned_grid_constants {
    use super::*;
    use crate::scoring::{TermBackend, TermBreakdown, TermField as F, VinaScoring, VinaWeights};
    use crate::types::{grid_type_index, Atom, AtomKind, AtomType, Element, GRID_TYPE_COUNT};

    /// A term's name and the edit that doubles its weight.
    type WeightDoubler = (&'static str, fn(&mut VinaWeights));

    /// Named rather than closed over, so all five share one `fn` type and the
    /// table below can be a plain list.
    fn double_gauss1(w: &mut VinaWeights) {
        w.gauss1 *= 2.0;
    }
    fn double_gauss2(w: &mut VinaWeights) {
        w.gauss2 *= 2.0;
    }
    fn double_repulsion(w: &mut VinaWeights) {
        w.repulsion *= 2.0;
    }
    fn double_hbond(w: &mut VinaWeights) {
        w.hbond *= 2.0;
    }
    fn double_hydrophobic(w: &mut VinaWeights) {
        w.hydrophobic *= 2.0;
    }

    const LAYOUT_DOC: &str = "SCORING.md §5.2: index = ((ix + nx·(iy + ny·iz)) · STRIDE) + type·4 + slot\n\
                                SCORING.md §5.2: STRIDE = GRID_TYPE_COUNT · MAPS_PER_TYPE = 10 · 4 = 40";
    const SLOT_DOC: &str =
        "SCORING.md §5.1 四个 slot: `Shape` (0) 高斯形状+排斥 / `HbFromDonor` (1) 氢键 / \
                            `HbFromAcceptor` (2) 氢键 / `Hydrophobic` (3) apolar 接触";
    const SPACING_DOC: &str = "SCORING.md §6.4: 默认间距取 0.375 Å（AutoDock 默认），足够细";
    const REACH_DOC: &str = "SCORING.md §5.3: `SpatialKernels::cutoff = 8.0` 是表面距离的上限，而内循环手里是真实距离，\
                              所以真实距离的截断应当是 `cutoff + 2*R`";

    fn assert_pinned<T: PartialEq + std::fmt::Debug>(
        constant: &str,
        found: T,
        documented: T,
        doc: &str,
    ) {
        if found != documented {
            panic!(
                "{constant} is {found:?}, but the specification documents {documented:?}.\n\
                 documented: {doc}\n\
                 The constant and the line that states it have to change together. \
                 A number nobody re-derived is worse than a number nobody wrote down."
            );
        }
    }

    fn carbon(cx: f64, cy: f64, cz: f64) -> Atom {
        let mut a = Atom::new(1, [cx, cy, cz], Element::C, AtomType::CH);
        a.kind = AtomKind::Hydrophobic;
        a
    }
    fn oxygen(cx: f64, cy: f64, cz: f64) -> Atom {
        let mut a = Atom::new(1, [cx, cy, cz], Element::O, AtomType::OA);
        a.kind = AtomKind::Acceptor;
        a
    }
    fn donor(cx: f64, cy: f64, cz: f64) -> Atom {
        let mut a = Atom::new(1, [cx, cy, cz], Element::N, AtomType::NA);
        a.kind = AtomKind::Donor;
        a
    }
    fn test_box() -> GridBox {
        GridBox::new([-6.0, -6.0, -6.0], [6.0, 6.0, 6.0]).unwrap()
    }
    /// Stored grid point nearest a world position, clamped into the grid.
    fn nearest(maps: &GridMaps, p: [f64; 3]) -> (usize, usize, usize) {
        let at = |lo: f64, h: f64, want: f64, n: usize| {
            (((want - lo) / h).round().max(0.0) as usize).min(n.saturating_sub(1))
        };
        (
            at(maps.min[0], maps.spacing[0], p[0], maps.dims[0]),
            at(maps.min[1], maps.spacing[1], p[1], maps.dims[1]),
            at(maps.min[2], maps.spacing[2], p[2], maps.dims[2]),
        )
    }

    /// `MAPS_PER_TYPE`, the element count and the flat stride: the memory
    /// layout of every score the program produces.
    ///
    /// The slot indices are pinned individually because §5.1's table names
    /// them, and because a *reordering* of the four slots keeps `map_stride()`
    /// correct while silently swapping the hydrogen-bond halves — the stride
    /// test in the `gpu` module could not see that.
    #[test]
    fn the_map_layout_is_the_documented_one() {
        assert_pinned("MAPS_PER_TYPE", MAPS_PER_TYPE, 4, LAYOUT_DOC);
        assert_pinned("GRID_TYPE_COUNT", GRID_TYPE_COUNT, 10, LAYOUT_DOC);
        assert_pinned("map_stride()", map_stride(), 40, LAYOUT_DOC);
        assert_pinned(
            "GridMaps::map_count() for a 1-point grid",
            MAPS_PER_TYPE * GRID_TYPE_COUNT,
            40,
            LAYOUT_DOC,
        );
        assert_pinned("MapSlot::Shape index", MapSlot::Shape.index(), 0, SLOT_DOC);
        assert_pinned(
            "MapSlot::HbFromDonor index",
            MapSlot::HbFromDonor.index(),
            1,
            SLOT_DOC,
        );
        assert_pinned(
            "MapSlot::HbFromAcceptor index",
            MapSlot::HbFromAcceptor.index(),
            2,
            SLOT_DOC,
        );
        assert_pinned(
            "MapSlot::Hydrophobic index",
            MapSlot::Hydrophobic.index(),
            3,
            SLOT_DOC,
        );
        assert_pinned(
            "MapSlot::ALL",
            MapSlot::ALL.map(|s| s.index()),
            [0, 1, 2, 3],
            SLOT_DOC,
        );
    }

    /// `DEFAULT_SPACING`, quoted from §6.4.
    ///
    /// The spacing is a default, not a tuned constant — it is a parameter — so
    /// what is pinned is that *this is the default*, not that every grid is
    /// built at it. §6.4's own argument is that it is a deliberate choice
    /// ("AutoDock 默认"), and a silent change would remove the argument.
    #[test]
    fn the_default_spacing_is_the_documented_one() {
        assert_pinned("DEFAULT_SPACING", DEFAULT_SPACING, 0.375, SPACING_DOC);
        // A spacing that is not a multiple of the default is a caller choice and
        // must survive, so the pin is on the constant, not on the estimate.
        let dims = estimate_dims(&test_box(), DEFAULT_SPACING);
        assert_pinned(
            "estimate_dims at the default spacing, first axis (12 A / 0.375 = 32, +1)",
            dims[0],
            33,
            SPACING_DOC,
        );
        assert_pinned(
            "a spacing <= 0 falls back to the default",
            estimate_dims(&test_box(), 0.0),
            dims,
            SPACING_DOC,
        );
    }

    /// The real-distance reach of the precalculation loop, which is the half of
    /// §5.3 that a constant in `scoring.rs` cannot see.
    ///
    /// **A documented-line drift, found by writing this assertion.** `SCORING.md`
    /// §5.3 states the reach as `kernels.cutoff + 2.0 * a.radius`. The code
    /// says `kernels.cutoff + MAX_GRID_TYPE_RADIUS + a.radius`, because the
    /// outer filter has to bound the **widest** of the ten per-probe-type
    /// blocks and the probe's radius is not the receptor atom's. The documented
    /// form is the same only when `MAX_GRID_TYPE_RADIUS == a.radius`, which is
    /// true for no element — it is the iodine radius, 2.350 Å. The documented
    /// form is therefore *narrower* than the code by `2.350 - R_atom`, i.e. it
    /// is the form that was correct when every heavy atom had the same 0.4 Å
    /// radius, and it is now stale in the conservative direction. The failure
    /// message quotes it, because that sentence is the thing that has to
    /// change: the code is right.
    #[test]
    fn the_precalculation_reach_is_the_cutoff_plus_both_radii() {
        let k = VinaScoring::new().spatial_kernels();
        let a = carbon(0.0, 0.0, 0.0);
        let reach = k.cutoff + MAX_GRID_TYPE_RADIUS + a.element.interaction_radius();
        // The code's own arithmetic, so a change to either term is visible.
        assert_pinned(
            "reach = cutoff + MAX_GRID_TYPE_RADIUS + receptor radius, for a carbon receptor atom",
            reach,
            8.0 + MAX_GRID_TYPE_RADIUS + 1.90,
            REACH_DOC,
        );
        // And the property the outer test actually has to guarantee: for **every**
        // probe type the reach must reach the furthest point the inner
        // `d <= cutoff` test can still accept, which is
        // `cutoff + grid_type_radius(p) + receptor radius`. If it ever stopped
        // reaching, the pre-filter would amputate a Gaussian tail for that
        // probe — the bug §5.3 describes, which cost 3.8 Å and turned no test
        // red. Equality is expected at the widest probe type, and being wider
        // anywhere is harmless.
        for p in 0..GRID_TYPE_COUNT {
            let rp = grid_type_radius(p);
            let needed = k.cutoff + rp + a.element.interaction_radius();
            assert!(
                reach >= needed,
                "probe type {p} (R = {rp}) is still contributing out to {needed} A \
                 but the loop's outer filter only reaches {reach} A; that is the \
                 amputation §5.3 describes.\ndocumented: {REACH_DOC}"
            );
        }
        // The widest probe type is the one the pre-filter is sized for, and
        // there it is exact rather than merely safe.
        let widest = (0..GRID_TYPE_COUNT)
            .map(grid_type_radius)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_pinned(
            "MAX_GRID_TYPE_RADIUS against the widest grid type radius",
            MAX_GRID_TYPE_RADIUS,
            widest,
            "SCORING.md §1: | I | 2.350 | ... (the widest of the ten per-probe-type radii)",
        );
        assert_pinned(
            "reach at the widest probe type, where the pre-filter is exact",
            k.cutoff + MAX_GRID_TYPE_RADIUS + a.element.interaction_radius(),
            k.cutoff + widest + a.element.interaction_radius(),
            REACH_DOC,
        );
        // Behaviourally, by **measuring** where the tabulated field stops rather
        // than sampling two points: walk outward along x and find the last
        // non-zero. Comparing the real distance against the raw cutoff truncated
        // the tabulation to `d <= 8.0 - 2R`, which for a carbon is `d <= 4.2`
        // instead of `d <= 8.0` — 3.8 Å of Gaussian tail gone, and no test in
        // the suite could see it.
        //
        // The file is read from disk rather than through `crate::gpu::ENERGY_WGSL`
        // below for the same reason the walk is a measurement: this test has to
        // run in the **default** configuration too, where the `gpu` module is not
        // compiled at all. `shader_and_rust_agree_on_the_grid_stride` covers the
        // `gpu` configuration's half of the same claim, through the `include_str!`
        // copy.
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let box_ = GridBox::new([-14.0, -14.0, -14.0], [14.0, 14.0, 14.0]).unwrap();
        let spacing = 0.5;
        let maps = GridMaps::precalculate(&rec, &box_, &VinaScoring::new(), spacing, 1).unwrap();
        let ti = grid_type_index(Element::C);
        let r_c = a.element.interaction_radius();
        let mut last: Option<f64> = None;
        for ix in 0..maps.dims()[0] {
            let x = maps.min[0] + ix as f64 * maps.spacing[0];
            let (jx, jy, jz) = nearest(&maps, [x, 0.0, 0.0]);
            if maps.raw(ti, MapSlot::Shape, jx, jy, jz) != 0.0f32 {
                last = Some(x - 2.0 * r_c);
            }
        }
        let last = last.expect("the shape field must be tabulated somewhere along x");
        // **The cutoff is not observable in the stored field, and that is worth
        // stating rather than working around.** Measured on this build, the
        // tabulated `Shape` support ends at surface distance **5.200 Å**, not
        // at the documented 8.0, and the reason is storage precision rather
        // than the cutoff: an `f32` cannot carry `w1 * g1` out there.
        //
        // The derivation, written out because it has to be reproducible rather
        // than memorable. `g1` falls off faster than an `f32` can follow it, so
        // the product has to be formed in `f64` and only then narrowed:
        //
        // ```text
        // g1      = exp(-((d - 0.5) / 0.5)^2) = 4.224152e-39     (f64)
        // w1*g1   = -0.035579 * 4.224152e-39    = -1.502911e-40   (f64)
        // narrowed to f32                      = -1.502907e-40   SUBNORMAL
        // ```
        //
        // Rounding `g1` to two significant figures *before* multiplying gives
        // -1.501434e-40 instead — a -0.10% error, small enough to look
        // harmless and irrelevant to every bound above, which is exactly why it
        // is worth naming: it is a step that lets a comment drift away from the
        // arithmetic while the drift stays invisible. An earlier version of this
        // comment did that *and* published `g1 = 5.1e-39`, which is not a
        // rounding of 4.224152e-39 at all but 21% high. That figure then left
        // this file in `docs/SCORING.md`, carried by a reader who had no way to
        // check it except against this comment, so the arithmetic is now pinned
        // by `the_shape_gauss_product_is_not_formed_from_a_rounded_gaussian`
        // and the classification by `the_outermost_stored_shape_values_are_unrepresentable`.
        //
        // So the map's effective support is set by storage precision, not by
        // the cutoff, and no test can find the cutoff by looking for the last
        // non-zero field value.
        //
        // What *can* be bounded is the direction that matters. The support must
        // reach at least as far as the terms are representable, i.e. it must sit
        // comfortably **inside** `cutoff`; and the §5.3 amputation — comparing
        // the real distance against the raw cutoff — truncates a carbon pair at
        // `8.0 - 2R = 4.200 Å`, which is *narrower* than the representable range
        // and so does show up here.
        assert!(
            last > 5.0 && last < k.cutoff,
            "the tabulated shape field ends at surface distance {last:.3} A. It \
             must reach past 5.0 A -- the range where w1*g1 is still a normal f32 \
             -- and stop inside the documented cutoff of {} A. A value of {:.3} A \
             is the signature of comparing the real distance against the raw \
             cutoff, which is the bug SCORING.md §5.3 describes.\n\
             documented: {REACH_DOC}",
            k.cutoff,
            k.cutoff - 2.0 * r_c
        );
        // And the cutoff therefore does real work only as a bound: the map's
        // support is {last:.1} A of the {k.cutoff} A available, so there is no
        // tuned constant in that gap for anyone to have tuned.
        assert!(
            k.cutoff - last > 2.0,
            "the gap between the f32 support ({last:.3} A) and the cutoff \
             ({} A) has collapsed to {} A, so the cutoff is no longer slack and \
             a change to the weights could start truncating a representable \
             term.\ndocumented: {REACH_DOC}",
            k.cutoff,
            k.cutoff - last
        );
        // And past the reach the field is *exactly* zero, not a rounding of
        // something tiny: the loop never evaluates those points.
        let (ix, iy, iz) = nearest(&maps, [k.cutoff + 1.0 + 2.0 * r_c, 0.0, 0.0]);
        assert_eq!(
            maps.raw(ti, MapSlot::Shape, ix, iy, iz),
            0.0f32,
            "a point beyond the cutoff of {} A must contribute exactly nothing.\n\
             documented: {REACH_DOC}",
            k.cutoff
        );
    }

    /// The two figures the `Shape` support comment publishes, pinned against
    /// the engine's own constants.
    ///
    /// A number in a comment has no compiler, and this one had already escaped:
    /// the comment's `g1 = 5.1e-39` was carried into `docs/SCORING.md` by a
    /// reader who had nothing to check it against except the comment. So the
    /// arithmetic behind both figures is asserted here, and asserted in the
    /// form that forbids the specific mistake — forming the product from a
    /// *rounded* `g1`, which is a -0.10% error that no bound in this file would
    /// ever notice and that is exactly how a figure goes stale.
    ///
    /// The band is 1e-6 relative. It has to admit the seven-significant-figure
    /// figures the comment prints, and it has to exclude both failure modes:
    /// the rounded form (-0.10% = 9.8e-4) and the wrong exponent that was there
    /// before (+21%). A band loose enough to pass the rounded form would be a
    /// guard that documents nothing.
    #[test]
    fn the_shape_gauss_product_is_not_formed_from_a_rounded_gaussian() {
        /// As published in the support comment above.
        const PUBLISHED_G1: f64 = 4.224_152e-39;
        const PUBLISHED_PRODUCT: f64 = -1.502_911e-40;
        /// The premature-rounding form, kept so the exclusion is demonstrable
        /// rather than asserted: -0.035579 * 4.22e-39.
        const ROUNDED_G1: f64 = 4.22e-39;
        const TOLERANCE: f64 = 1e-6;

        let k = SpatialKernels::from_weights(&VinaWeights::default());
        let d = 5.2f64;
        let (off, width) = k.gauss1;

        // Formed the way the comment says it is formed: the Gaussian in f64,
        // then the product in f64, and only then the narrowing to f32.
        let g1 = (-(((d - off) / width) * ((d - off) / width))).exp();
        let product = k.w1 * g1;

        assert!(
            (g1 - PUBLISHED_G1).abs() / PUBLISHED_G1 < TOLERANCE,
            "the engine computes g1 = {g1:.6e} at d = {d}, but the Shape support \
             comment publishes {PUBLISHED_G1:.6e}. The comment is the only place \
             this number exists, so it has to be re-derived when a weight or a \
             width changes."
        );
        assert!(
            (product - PUBLISHED_PRODUCT).abs() / PUBLISHED_PRODUCT.abs() < TOLERANCE,
            "the engine computes w1 * g1 = {product:.6e} at d = {d} \
             (w1 = {}), but the comment publishes {PUBLISHED_PRODUCT:.6e}.",
            k.w1
        );
        // The mutual consistency is the real guard, and the one that would have
        // caught the original defect: the two published figures must agree with
        // each other through the engine's own w1 to better than the rounding the
        // comment warns against.
        assert!(
            (k.w1 * PUBLISHED_G1 - PUBLISHED_PRODUCT).abs() / PUBLISHED_PRODUCT.abs() < TOLERANCE,
            "the two figures in the comment are not consistent with each other: \
             w1 * g1 = {} against a published product of {PUBLISHED_PRODUCT:.6e}. \
             A gap like that is what a rounded g1 looks like after the fact.",
            k.w1 * PUBLISHED_G1
        );
        // The exclusion, stated rather than implied: the rounded form is the
        // mistake, and it has to fail the same band the real figures pass.
        assert!(
            (k.w1 * ROUNDED_G1 - PUBLISHED_PRODUCT).abs() / PUBLISHED_PRODUCT.abs() >= TOLERANCE,
            "the tolerance is loose enough to admit the premature rounding \
             ({:.2e} relative), so it would not catch the error it exists to \
             catch",
            (k.w1 * ROUNDED_G1 - PUBLISHED_PRODUCT).abs() / PUBLISHED_PRODUCT.abs()
        );
        // And the narrowing really does lose it: the f32 image is subnormal,
        // which is the claim the whole comment rests on.
        let narrowed = product as f32;
        assert!(
            narrowed != 0.0 && narrowed.is_subnormal(),
            "narrowing w1 * g1 at d = {d} gives {narrowed:e}, which is {} the \
             support comment's central claim is that the stored value is not \
             normal. If this is normal, the comment's reason is wrong.",
            if narrowed == 0.0 { "exactly zero, and" } else { "normal, so" }
        );
    }

    /// What the stored field actually holds across the boundary where the
    /// `Shape` term stops being representable, measured rather than quoted.
    ///
    /// The phenomenon is real and the claim in `docs/SCORING.md` is about it, so
    /// this asserts the *classification* at three points and refuses to assert a
    /// distance. There is no number here that a reader could quote as "the
    /// cutoff": the support ends wherever storage precision ends, which depends
    /// on the probe radius, the grid spacing and the weights, all of which are
    /// separate parameters. A test that pinned a distance would be inventing a
    /// specification out of a build artefact.
    #[test]
    fn the_outermost_stored_shape_values_are_unrepresentable() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let box_ = GridBox::new([-14.0, -14.0, -14.0], [14.0, 14.0, 14.0]).unwrap();
        let maps = GridMaps::precalculate(&rec, &box_, &VinaScoring::new(), 0.5, 1).unwrap();
        let ti = grid_type_index(Element::C);
        let r_c = Element::C.interaction_radius();

        let mut band: Vec<(f64, f32, &'static str)> = Vec::new();
        for ix in 0..maps.dims()[0] {
            let x = maps.min[0] + ix as f64 * maps.spacing[0];
            let d = x - 2.0 * r_c;
            // The far side only. The walk starts at the near edge of the box,
            // where the surface distance is *negative* -- the probe point is
            // inside the sphere -- and the Gaussian is as tiny there as it is
            // in the far tail, so including the near side makes the "band"
            // span the whole atom and measures nothing. Only d >= 0 is a
            // surface distance in the sense the support claim is about.
            if d < 0.0 {
                continue;
            }
            let (jx, jy, jz) = nearest(&maps, [x, 0.0, 0.0]);
            let v = maps.raw(ti, MapSlot::Shape, jx, jy, jz);
            if v == 0.0 {
                continue;
            }
            band.push((d, v, if v.is_subnormal() { "subnormal" } else { "normal" }));
        }
        assert!(
            !band.is_empty(),
            "the shape field is entirely zero, so this test has nothing to measure"
        );

        // The band exists: at least one stored value is a normal f32 and at
        // least one is subnormal. Both halves, because "all subnormal" would
        // mean the tabulation is broken and "all normal" would mean there is
        // no boundary to report.
        let normals = band.iter().filter(|(_, _, c)| *c == "normal").count();
        let subnormals = band.iter().filter(|(_, _, c)| *c == "subnormal").count();
        assert!(
            normals > 0 && subnormals > 0,
            "expected both normal and subnormal stored values, found {normals} \
             normal and {subnormals} subnormal over {} non-zero points",
            band.len()
        );
        // The outermost non-zero stored value is subnormal on this build, which
        // is the whole point: the tail of the tabulated support is stored as
        // values that print as `-0.0000` and carry almost no mantissa.
        let (last_d, last_v, last_class) = *band.last().expect("band is non-empty");
        assert_eq!(
            last_class, "subnormal",
            "the outermost non-zero shape value is {last_v:e} at surface distance \
             {last_d:.3} A and is {last_class}. The claim under test is that the \
             outermost stored values are unrepresentable; if the tail is normal \
             then the support boundary moved and SCORING.md's claim needs \
             re-measuring rather than this test needing loosening."
        );
        // The band's two ends, in the units the claim is about. `f32::MIN_POSITIVE`
        // is the smallest *normal* value; a subnormal sits below it, and the
        // ratio is how far below the normal range the tail of the tabulated
        // support lives. Measured on this build that is 78x, which is worth
        // stating precisely because it is *modest*: the value is a perfectly
        // representable f32 with most of its mantissa intact. What is lost is
        // the guarantee of full relative precision and, far more visibly, the
        // ability to print it -- not the number itself.
        let smallest_normal = f32::MIN_POSITIVE;
        let below_normal = (smallest_normal / last_v.abs()) as f64;
        assert!(
            last_v.abs() < smallest_normal && below_normal > 10.0,
            "expected the outermost stored value to be below the smallest normal \
             f32 by more than an order of magnitude, got {last_v:e} at \
             {last_d:.3} A, which is only {below_normal:.1}x below \
             {smallest_normal:e}"
        );
        // And it is the *display* that misleads, not the arithmetic. A subnormal
        // f32 still holds a relative-precision mantissa -- halving it yields
        // another representable subnormal rather than a floor -- so the harm is
        // that a caller printing the field sees a signed zero. Asserted here
        // because an earlier draft of this test asserted the opposite
        // ("underflows when halved"), and the test caught me being wrong about
        // what subnormal means.
        assert_eq!(
            format!("{last_v:.4}"),
            "-0.0000",
            "the outermost stored value formats as {last_v:.4}, so the premise of \
             this test -- that the tail of the support is invisible in a printed \
             field -- no longer holds and the claim needs re-measuring"
        );
        // Past the band the field is exactly zero, so the support really does
        // end here rather than trailing off.
        let mut zeros_beyond = 0usize;
        for ix in 0..maps.dims()[0] {
            let x = maps.min[0] + ix as f64 * maps.spacing[0];
            let d = x - 2.0 * r_c;
            if d < 0.0 || d <= last_d {
                continue;
            }
            let (jx, jy, jz) = nearest(&maps, [x, 0.0, 0.0]);
            if maps.raw(ti, MapSlot::Shape, jx, jy, jz) == 0.0 {
                zeros_beyond += 1;
            }
        }
        assert!(
            zeros_beyond > 0,
            "every grid point beyond the outermost non-zero value is also \
             non-zero, so the field has no end and 'the support ends' is not what \
             is being measured"
        );
        // Nothing above asserts how far out this is. Report it, so the claim in
        // the document can be checked against a measurement rather than a
        // comment, and so a change in spacing or weights shows up here.
        let first_subnormal = band
            .iter()
            .find(|(_, _, c)| *c == "subnormal")
            .map(|(d, _, _)| *d)
            .expect("there is at least one subnormal value");
        let last_normal_d = band
            .iter()
            .filter(|(_, _, c)| *c == "normal")
            .map(|(d, _, _)| *d)
            .last()
            .unwrap_or(f64::NAN);
        println!(
            "SHAPE_SUPPORT: {} non-zero points, {zeros_beyond} exactly zero beyond; \
             last normal {last_normal_d:.3} A; subnormal {first_subnormal:.3}-{last_d:.3} A \
             ({:.3} A wide); outermost stored value {last_d:.3} A = {last_v:e}, {below_normal:.0}x \
             below the smallest normal f32, prints as {last_v:.4}",
            band.len(),
            last_d - first_subnormal,
        );
    }

    /// `MAX_GRID_POINTS` has **no line in `SCORING.md`**. It is a capacity
    /// ceiling, it is checked before allocation, and the reason it exists is
    /// written in its own doc comment rather than in the specification. Pinned
    /// against that, with the absence stated in the failure message: a scoring
    /// constant with no specification is a finding the reader should be handed,
    /// not a blank to fill in.
    #[test]
    fn the_grid_capacity_ceiling_is_what_its_own_comment_claims() {
        const DOC: &str = "grid.rs: '2^28 points is 40 GiB of f32 at the current stride, so this \
limit only ever fires on a request that was never going to succeed anyway' \
-- and SCORING.md has no line for MAX_GRID_POINTS.";
        assert_pinned("MAX_GRID_POINTS", MAX_GRID_POINTS, 1 << 28, DOC);
        // The claim the comment makes is checkable, so check it: 2^28 points at
        // the current stride is 40 GiB of f32.
        let bytes = (MAX_GRID_POINTS as f64) * (map_stride() as f64) * 4.0;
        let gib = bytes / (1024.0 * 1024.0 * 1024.0);
        assert!(
            (gib - 40.0).abs() < 0.5,
            "the comment says 2^28 points is 40 GiB at the current stride, but \
             {MAX_GRID_POINTS} points * {stride} f32 = {gib:.1} GiB\n\
             documented: {DOC}",
            stride = map_stride()
        );
    }

    /// **The CPU/GPU verdict, as a type rather than a sentence.**
    ///
    /// The per-term decomposition is CPU-only, and the reason is structural: the
    /// GPU entry point is `GpuContext::score(&Batch, &GridMaps)` — it takes a
    /// [`GridMaps`], and there is no overload or trait that would accept a
    /// [`TermMaps`]. So on a `gpu` build there is no path by which these five
    /// terms reach a device, and the question "do the terms agree between the
    /// two builds?" has the answer "there is nothing to compare", not "yes".
    ///
    /// What this test pins is that the *reason* stays true. The strides are
    /// different numbers, and a term-map kernel does not exist:
    ///
    /// * [`map_stride`] is 40 and [`term_stride`] is 60, so a kernel written for
    ///   one cannot read the other; and
    /// * the only WGSL in the crate declares the production stride.
    ///
    /// The surface makes the verdict visible in three places, none of them a
    /// document: [`TermMaps::BACKEND`] for a caller deciding whether to build
    /// the tabulation, `TermBreakdown::backend` on every value returned, and
    /// this test, which fails the moment a 60-float stride reaches a shader.
    ///
    /// The file is read from disk rather than through `crate::gpu::ENERGY_WGSL`
    /// so that this runs in the **default** configuration, where the `gpu`
    /// module is not compiled at all. `under_the_gpu_feature_the_terms_are_still_cpu_only`
    /// covers the `gpu` half, through the `include_str!` copy.
    #[test]
    fn the_per_term_decomposition_is_cpu_only_and_says_so() {
        assert_pinned(
            "TermMaps::BACKEND",
            TermMaps::BACKEND,
            TermBackend::Cpu,
            "scoring.rs TermBackend",
        );
        assert_pinned(
            "TermBackend::default()",
            TermBackend::default(),
            TermBackend::Cpu,
            "scoring.rs TermBackend",
        );
        // The two layouts are genuinely different numbers, which is the
        // mechanical reason the split could not have been done in place.
        assert_pinned("map_stride()", map_stride(), 40, LAYOUT_DOC);
        assert_pinned(
            "term_stride()",
            term_stride(),
            60,
            "scoring.rs TERM_FIELDS = 6; grid.rs term_stride() = GRID_TYPE_COUNT * TERM_FIELDS",
        );
        assert_ne!(map_stride(), term_stride());
        assert_pinned(
            "TERM_FIELDS",
            crate::scoring::TERM_FIELDS,
            6,
            "scoring.rs TERM_FIELDS = 6",
        );
        // The only shader in the crate knows the production stride, and the 60
        // never appears as one. If a term-maps kernel is ever added this turns
        // red, which is the moment the two backends would have to be proven
        // equal term by term before `TermBackend` could grow a variant.
        let wgsl =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/energy.wgsl"))
                .expect("energy.wgsl is include_str!d by the gpu module and must be present");
        assert!(
            wgsl.contains(&format!("const STRIDE: u32 = {}u", map_stride())),
            "energy.wgsl does not declare the production stride {}; the \
             per-term decomposition is CPU-only and the only shader must be the \
             production one",
            map_stride()
        );
        assert!(
            !wgsl.contains(&format!("const STRIDE: u32 = {}u", term_stride())),
            "energy.wgsl now declares a {}-float stride, which is the term-maps \
             layout. A per-term GPU path exists, so `TermBackend` needs a \
             second variant and the two backends have to be proven equal term \
             by term before it gets one",
            term_stride()
        );
        assert!(
            !wgsl.contains(&format!("{}.0", term_stride()))
                && !wgsl.contains(&format!("[{}]", term_stride()))
                && !wgsl.contains(&format!("<f32, {}>", term_stride())),
            "energy.wgsl mentions the {}-float term-map stride somewhere other \
             than a STRIDE const; find it and either delete it or finish the GPU \
             path deliberately",
            term_stride()
        );
    }

    /// The same verdict, in the `gpu` feature configuration.
    ///
    /// Runs the layout parity the `gpu` module already tests, and then asserts
    /// that nothing in that configuration can turn the decomposition into a
    /// device result. Without this, a `gpu` build would be the one place the
    /// per-term answer could differ — and it would differ *silently*, on
    /// whichever build a user happened to install, which is the worst possible
    /// shape for this bug class.
    #[cfg(feature = "gpu")]
    #[test]
    fn under_the_gpu_feature_the_terms_are_still_cpu_only() {
        use crate::gpu::ENERGY_WGSL;
        // The parity the gpu module already asserts: the shader's stride is the
        // production one. Unchanged from the non-gpu configuration, so the
        // energy the two builds produce is comparable.
        assert!(
            ENERGY_WGSL.contains(&format!("const STRIDE: u32 = {}u", map_stride())),
            "the shader stride no longer matches map_stride() = {}",
            map_stride()
        );
        // And there is no term-maps stride to compare against, in either
        // direction. The decomposition is not degraded on this build; it is
        // absent, and it says so.
        assert_eq!(TermMaps::BACKEND, TermBackend::Cpu);
        assert_ne!(map_stride(), term_stride());
        // A `gpu` build's own view of the layout agrees with the CPU's.
        assert_eq!(crate::gpu::ATOM_STRIDE, 8);
    }

    /// `conformation_terms` is the sum of its own `terms_at` calls.
    ///
    /// The per-term path's Rust-side coverage used to be a *single* atom, which
    /// cannot see an accumulation bug at all: with one atom the loop runs once
    /// and any indexing error is invisible. Three atoms of three different
    /// classes, at three different places, is the smallest input that makes the
    /// accumulation, the per-atom type index and the per-atom mask all matter.
    #[test]
    fn conformation_terms_is_the_sum_of_its_own_atoms() {
        let rec = Molecule::from_atoms(vec![
            carbon(0.0, 0.0, 0.0),
            oxygen(4.0, 0.0, 0.0),
            donor(0.0, 4.0, 0.0),
        ])
        .unwrap();
        let box_ = test_box();
        let terms = TermMaps::precalculate(&rec, &box_, &VinaScoring::new(), 0.375, 1).unwrap();

        let kinds = [AtomKind::Hydrophobic, AtomKind::Acceptor, AtomKind::Donor];
        let elements = [Element::C, Element::O, Element::N];
        let coords: [Vec3; 3] = [[0.9, -1.1, 0.4], [3.6, 0.2, -0.8], [-0.7, 3.5, 1.3]];
        let ti: Vec<usize> = elements.iter().map(|e| grid_type_index(*e)).collect();
        let masks: Vec<[bool; crate::scoring::TERM_FIELDS]> = kinds
            .iter()
            .map(|k| {
                let mut a = Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH);
                a.kind = *k;
                crate::scoring::probe_term_mask(&a)
            })
            .collect();

        let got = terms.conformation_terms(&ti, &masks, &coords);
        // Every term must be populated by this receptor, or the test below is
        // comparing zeros.
        assert!(
            got.gauss1 != 0.0
                && got.gauss2 != 0.0
                && got.repulsion != 0.0
                && got.hbond != 0.0
                && got.hydrophobic != 0.0,
            "this fixture is supposed to populate all five terms, got \
             g1={} g2={} rep={} hb={} hyd={}",
            got.gauss1,
            got.gauss2,
            got.repulsion,
            got.hbond,
            got.hydrophobic
        );

        // The reference has to be built by a different route: three separate
        // `terms_at` calls added by hand, not `terms_total`.
        let mut g1 = 0.0;
        let mut g2 = 0.0;
        let mut rep = 0.0;
        let mut hb = 0.0;
        let mut hyd = 0.0;
        let mut hbd = 0.0;
        let mut hba = 0.0;
        for i in 0..3 {
            let t = terms
                .terms_at(ti[i], &masks[i], coords[i])
                .expect("inside grid");
            g1 += t.gauss1;
            g2 += t.gauss2;
            rep += t.repulsion;
            hb += t.hbond;
            hyd += t.hydrophobic;
            hbd += t.hbond_from_donor;
            hba += t.hbond_from_acceptor;
        }
        for (name, got_v, want) in [
            ("gauss1", got.gauss1, g1),
            ("gauss2", got.gauss2, g2),
            ("repulsion", got.repulsion, rep),
            ("hbond", got.hbond, hb),
            ("hydrophobic", got.hydrophobic, hyd),
            ("hbond_from_donor", got.hbond_from_donor, hbd),
            ("hbond_from_acceptor", got.hbond_from_acceptor, hba),
        ] {
            assert_eq!(
                got_v, want,
                "conformation_terms must accumulate {name} atom by atom"
            );
        }
        // And the backend travels with the value.
        assert_eq!(got.backend, TermBackend::Cpu);
        let _ = TermBreakdown::default().backend;
    }

    /// **A wrong term in `conformation_terms` moves that term and no other.**
    ///
    /// This is the Rust-side version of the observation the per-term surface
    /// exists for. The measurement that motivated it: retuning the hydrophobic
    /// window from `1.5` to `1.45` moved the `hydrophobic` entry by
    /// **1.540e-03** — 1540x over a 1e-6 tolerance, red — while the *same*
    /// deviation on the sum was 16x **under** a 0.025 tolerance, green. A sum
    /// is structurally blind to which term is wrong, so the only defence is a
    /// test that perturbs one weight and demands that exactly one column move.
    ///
    /// All five weights are checked, over a whole three-atom conformation, so
    /// the accumulation and the per-atom masks are in scope too and not just
    /// the single-atom case.
    #[test]
    fn a_wrong_weight_moves_exactly_one_term_of_a_whole_conformation() {
        let rec = Molecule::from_atoms(vec![
            carbon(0.0, 0.0, 0.0),
            oxygen(4.0, 0.0, 0.0),
            donor(0.0, 4.0, 0.0),
        ])
        .unwrap();
        let box_ = test_box();
        let coords: [Vec3; 3] = [[0.9, -1.1, 0.4], [3.6, 0.2, -0.8], [-0.7, 3.5, 1.3]];
        let kinds = [AtomKind::Hydrophobic, AtomKind::Acceptor, AtomKind::Donor];
        let elements = [Element::C, Element::O, Element::N];
        let ti: Vec<usize> = elements.iter().map(|e| grid_type_index(*e)).collect();
        let masks: Vec<[bool; crate::scoring::TERM_FIELDS]> = kinds
            .iter()
            .map(|k| {
                let mut a = Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH);
                a.kind = *k;
                crate::scoring::probe_term_mask(&a)
            })
            .collect();

        let row = |sc: &VinaScoring| {
            let tm = TermMaps::precalculate(&rec, &box_, sc, 0.375, 1).unwrap();
            let t = tm.conformation_terms(&ti, &masks, &coords);
            [t.gauss1, t.gauss2, t.repulsion, t.hbond, t.hydrophobic]
        };
        let names = ["g1", "g2", "rep", "hb", "hyd"];
        let base = row(&VinaScoring::new());

        // Each weight is doubled in turn. Two is chosen because it moves a term
        // by its own magnitude, which is ~1e-1 here and four orders above the
        // 1e-6 the sum-level tolerance would hide.
        //
        // The labels are the *term* names, not the weight-field names: `g1` and
        // `gauss1` are the same term, and matching on the wrong one is a test
        // that passes for the wrong reason.
        let mutators: Vec<WeightDoubler> = vec![
            ("g1", double_gauss1),
            ("g2", double_gauss2),
            ("rep", double_repulsion),
            ("hb", double_hbond),
            ("hyd", double_hydrophobic),
        ];
        assert_eq!(mutators.len(), names.len());

        for (which, f) in mutators {
            let mut w = VinaWeights::default();
            f(&mut w);
            let got = row(&VinaScoring { weights: w });
            for j in 0..names.len() {
                let delta = (got[j] - base[j]).abs();
                if which == names[j] {
                    assert!(
                        delta > 1e-2,
                        "doubling {which} must move {which}, but it moved by \
                         only {delta:e}; the fixture is not populating it"
                    );
                } else {
                    assert!(
                        delta < 1e-9,
                        "doubling {which} also moved {} by {delta:e}. A term that \
                         moves when another one is wrong is a term nothing can be \
                         checked against -- the sum-level checks are all green \
                         either way, which is how a 1.5 -> 1.45 hydrophobic \
                         window change reached a release.",
                        names[j]
                    );
                }
            }
        }
        // Sanity: the reference row really does have five non-zero terms, so
        // the loop above is not comparing five zeros.
        for j in 0..names.len() {
            assert!(
                base[j] != 0.0,
                "{} is zero in the reference conformation",
                names[j]
            );
        }
    }

    /// The per-term tabulation is a *second* tabulation, not a widening of the
    /// first, and that has a measurable consequence a reader can check.
    ///
    /// 60 floats per point against 40 — 1.5x — for a diagnostic, and the
    /// production docking path never allocates one. Pinned so that "it is
    /// cheaper than it looks" cannot quietly become true.
    #[test]
    fn the_term_tabulation_costs_one_and_a_half_times_the_production_one() {
        assert_pinned(
            "term_stride()",
            term_stride(),
            60,
            "scoring.rs TERM_FIELDS = 6",
        );
        assert_pinned("map_stride()", map_stride(), 40, LAYOUT_DOC);
        let ratio = term_stride() as f64 / map_stride() as f64;
        assert!(
            (ratio - 1.5).abs() < 1e-12,
            "the term maps are {ratio} times the production stride, and the \
             documentation says 1.5x"
        );
        // And the two tabulations really are independent allocations: the
        // production map is built and the docking path scored without ever
        // touching one.
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let box_ = test_box();
        let maps = GridMaps::precalculate(&rec, &box_, &VinaScoring::new(), 0.375, 1).unwrap();
        assert_eq!(
            maps.data_len(),
            map_stride() * maps.point_count(),
            "the production map allocates nothing for the term decomposition"
        );
        let _ = F::Gauss1;
    }
}
