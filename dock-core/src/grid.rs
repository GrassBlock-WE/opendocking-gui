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
/// # This is an *indexing* limit, not a memory limit
///
/// It reads like a memory budget and is not one, and that distinction is the
/// whole reason the number is what it is.
///
/// `energy.wgsl` computes the flat index as
///
/// ```text
/// let idx = (cell.x + cx) + nx * ((cell.y + cy) + ny * (cell.z + cz));
/// let g = idx * STRIDE + type_base;   // then grid[g] .. grid[g + 3]
/// ```
///
/// and **every** term in that expression is a `u32`. The largest index the
/// kernel can read is `points * STRIDE - 1`: `cell` is at most `n - 2` because
/// the kernel clamps it there before it indexes (`clamp(floor(u), 0, n - 2)`),
/// so `cell + corner` is at most `n - 1`, and `type_base + 3` is the last of
/// the `STRIDE` values belonging to that point. So the addressable point count
/// is `(u32::MAX) / STRIDE` and nothing else.
///
/// That sentence is not trusted to a line number, and this file used to point
/// one at the shader: the expression moved and the pointer went on asserting
/// the old position afterwards. The test
/// `the_shader_flat_index_is_u32_so_this_limit_is_the_right_one` now checks the
/// claim against the shader's own text, and turns red if the expression is
/// renamed or retyped. A `u32` -> `u64` change is a behaviour change here and
/// should be a failing test, not a comment.
///
/// The previous value, `1 << 28`, was a memory number, and it advertised
/// 2.5x more points than the kernel can address: 268,435,456 against
/// 107,374,182. A grid between those two sizes passes every check in this
/// crate and then wraps its index arithmetic inside the kernel, so the last
/// maps in the grid are read from low addresses that hold something else. The
/// `u32` does not fault and the `nx`/`ny`/`nz` the host uploaded are narrowed
/// in the same expression, so the result is a plausible-looking number rather
/// than an error. The published limit was therefore a promise the engine could
/// not keep.
///
/// So this constant is deliberately **below** what the host could allocate.
/// 107,374,182 points is 16.0 GiB of `f32` at the current stride, and a caller
/// who asks for that much still has to get the memory: this number says the
/// arithmetic is sound, not that the allocation will succeed.
///
/// # What it costs a CPU-only caller
///
/// `precalculate` is the CPU path and the Rust-side indexing is `usize`, so
/// nothing here stops the CPU from tabulating a larger grid correctly. A
/// caller who only ever scores on the CPU, on a machine with enough RAM for a
/// 20-40 GiB grid, is refused something that would have worked for them. That
/// is a real narrowing and it is the intended one -- the engine's advertised
/// limit should be a limit the whole engine honours -- but it is a behaviour
/// change and not a no-op, and [`Self::precalculate`] reports it as a parameter
/// error naming this number.
pub const MAX_GRID_POINTS: u64 = (u32::MAX as u64) / (map_stride() as u64);

/// Pin [`MAX_GRID_POINTS`] to the kernel's `u32` index arithmetic.
///
/// Two assertions rather than one, because "the limit is small enough" and "the
/// limit is the *largest* small-enough value" are different claims and only
/// the second one keeps the constant tight. Together they say
/// `MAX_GRID_POINTS` is exactly `(u32::MAX) / STRIDE` for whatever
/// `map_stride()` currently returns, so a change to `MAPS_PER_TYPE` or
/// `GRID_TYPE_COUNT` moves this limit with it instead of leaving a stale
/// number behind.
///
/// This is a guard on the derivation, not a tripwire: it passes on the tree as
/// it stands, and it goes red only if someone changes one side of a relation
/// that this file derives from the other. An earlier version of this constant
/// was `1 << 28`, and an assert placed against *that* would have failed the
/// build of the whole crate -- which is why the constant was lowered first.
const _: () = {
    assert!(
        MAX_GRID_POINTS * (map_stride() as u64) <= u32::MAX as u64,
        "MAX_GRID_POINTS * map_stride() exceeds u32::MAX, so the last map value of \
         a grid at the limit wraps inside energy.wgsl's `idx * STRIDE`"
    );
    assert!(
        (MAX_GRID_POINTS + 1) * (map_stride() as u64) > u32::MAX as u64,
        "MAX_GRID_POINTS is now smaller than the largest point count energy.wgsl \
         can address; one more point would still fit, so the limit is needlessly \
         tight and should be re-derived from u32::MAX / map_stride()"
    );
};

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

/// The number of tabulated points a box will be tabulated at, without allocating them.
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

/// Refuse a set of grid dimensions the GPU kernel's `u32` index cannot address.
///
/// The free function behind [`GridMaps::check_gpu_index_range`], and separate
/// from it so the rule can be tested against dimensions no test can afford to
/// allocate: a grid at the limit is 16 GiB, so exercising the boundary through
/// a real [`GridMaps`] would mean building one, and exercising it through
/// [`GridMaps::precalculate`] would mean either skipping the interesting side
/// of the boundary or committing 16 GiB to find out.
///
/// Checked in `u64` with checked multiplication, so a `dims` whose product
/// overflows `usize` is reported as *over the limit* rather than as a second
/// and differently worded failure.
pub fn check_gpu_dims(dims: [usize; 3]) -> Result<()> {
    let stride = map_stride() as u64;
    let mut points: u64 = 1;
    for d in dims {
        points = match points.checked_mul(d as u64) {
            Some(p) => p,
            None => {
                return Err(DockError::param(
                    "grid dims",
                    format!("{dims:?}"),
                    format!(
                        "the point count overflows, so it is certainly above the \
                         {MAX_GRID_POINTS} points energy.wgsl can address with its \
                         u32 flat index"
                    ),
                ))
            }
        };
    }
    if points > MAX_GRID_POINTS {
        return Err(DockError::param(
            "grid dims",
            format!("{dims:?}"),
            format!(
                "this grid has {points} points, and energy.wgsl computes its flat index \
                 as `idx * {stride}` in u32, so the kernel can address at most \
                 {MAX_GRID_POINTS} points ({} values). A grid this size did not come \
                 from GridMaps::precalculate, which refuses anything larger before \
                 allocating.",
                points.saturating_mul(stride)
            ),
        ));
    }
    Ok(())
}

/// Refuse a `data` length that disagrees with the `dims` beside it.
///
/// The third constraint, and the only one of the three that is about the
/// *payload* rather than about the shape or about address arithmetic:
///
/// 1. [`check_gpu_dims`] -- the point count is one the kernel's `u32` flat
///    index can address at all.
/// 2. [`GridMaps::check_gpu_index_range`] -- every axis holds at least two
///    points, so the shader's `n - 2` clamp has a cell to clamp to.
/// 3. this one -- `data` is exactly `points * map_stride()` values long.
///
/// They are independent, and the point of listing them is that a grid can pass
/// the first two and still fail this one. It did: a document carrying `dims`
/// for 107,374,182 points beside 160 floats satisfied both addressability
/// rules, because neither of them looks at `data`, and `score()` then uploaded
/// 160 values for a grid the shader addresses up to index 4,294,967,279.
///
/// # Why `u128`, and why that is not decoration
///
/// The comparison is done in `u128` with checked multiplication, because a
/// check that can wrap is not a check: `dims` for 4,194,304 points per axis
/// times 40 overflows `u64`, and the wrapped result is a small number that a
/// short `data` could match by accident. `u128` holds the product of three
/// `usize` axes and one further factor of `map_stride()` for every `usize` a
/// target can represent; where even that is exceeded the implied length is
/// larger than any `data_len` that can exist on the machine, so the refusal
/// stays certain rather than becoming a guess.
///
/// Both `u128` overflow arms are **reachable**, not defensive decoration, and
/// both are exercised by name in
/// `tests/map_data_len.rs::the_u128_overflow_arms_are_reached_and_named`. The
/// arithmetic, for a 64-bit target where `usize::MAX = 2^64 - 1`:
///
/// * the *point* arm needs the product of the three axes alone to exceed
///   `u128::MAX = 2^128 - 1`. `usize::MAX` squared is `2^128 - 2^65 + 1`,
///   which still fits; the third axis is what overflows. So
///   `dims = [usize::MAX; 3]` reaches it, and on a 32-bit target it cannot,
///   which is why that test is `cfg`-gated rather than written as
///   "unreachable".
/// * the *value* arm needs `points` to fit while `points * 40` does not, so
///   `points` must exceed `u128::MAX / 40 ~ 8.5e36`. `2^126` does and `2^125`
///   does not, so `dims = [1 << 42; 3]` reaches it on any target wide enough
///   for `1 << 42` to be a `usize`.
///
/// Neither arm is reachable from a *real* map: `precalculate` refuses an
/// over-large box before it allocates, and a `data` of the implied length
/// cannot be built on the machine the implied length overflows. That is why
/// they are reached through `dims` rather than through a `Vec`, and it is why
/// "unreachable in practice" was the wrong thing to write down about them:
/// they are reachable through the only input the function has.
pub fn check_gpu_data_len(dims: [usize; 3], data_len: usize) -> Result<()> {
    check_payload_len(PayloadKind::Grid, dims, data_len)
}

/// The same rule for [`TermMaps`], on the same terms and through the same code.
///
/// See the [`TermMaps`] deserialiser for why a second `Vec<f32>` with a public
/// `dims` beside it needed the same closing.
pub fn check_term_data_len(dims: [usize; 3], data_len: usize) -> Result<()> {
    check_payload_len(PayloadKind::Term, dims, data_len)
}

/// Which of the two `Vec<f32>` payloads a `data` length is being checked
/// against.
///
/// `GridMaps` and `TermMaps` are the same shape under two different strides:
/// a private `data`, a public `dims`, a derived `Deserialize`, and a stride
/// that is implied rather than stored. So this is **one** rule with two
/// presentations, not two rules that happen to agree -- `check_payload_len`
/// does the arithmetic once, and this type says how the two readers fail
/// differently, which they genuinely do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PayloadKind {
    /// The production maps, read by `energy.wgsl` on a device.
    Grid,
    /// The per-term decomposition, read by `TermMaps::terms_at` on the CPU.
    Term,
}

impl PayloadKind {
    /// `f32` values stored per grid point.
    const fn stride(self) -> usize {
        match self {
            PayloadKind::Grid => map_stride(),
            PayloadKind::Term => term_stride(),
        }
    }

    /// How the refusal names the stride in the expected-length clause.
    const fn stride_fn_name(self) -> &'static str {
        match self {
            PayloadKind::Grid => "map_stride()",
            PayloadKind::Term => "term_stride()",
        }
    }

    /// The name the refusal gives the offending field.
    const fn field(self) -> &'static str {
        match self {
            PayloadKind::Grid => "grid data length",
            PayloadKind::Term => "term data length",
        }
    }
}

fn check_payload_len(kind: PayloadKind, dims: [usize; 3], data_len: usize) -> Result<()> {
    let stride = kind.stride() as u128;
    let mut points: u128 = 1;
    for d in dims {
        points = match points.checked_mul(d as u128) {
            Some(p) => p,
            None => {
                return Err(data_len_mismatch(
                    kind,
                    dims,
                    data_len,
                    None,
                    "the point count overflows u128, so it is larger than any array \
                     that can exist on this machine",
                ))
            }
        };
    }
    let expected = match points.checked_mul(stride) {
        Some(e) => e,
        None => {
            return Err(data_len_mismatch(
                kind,
                dims,
                data_len,
                Some(points),
                "the value count overflows u128, so it is larger than any array \
                 that can exist on this machine",
            ))
        }
    };
    if expected != data_len as u128 {
        return Err(data_len_mismatch(
            kind,
            dims,
            data_len,
            Some(points),
            &format!(
                "{expected} ({points} points x {} = {stride})",
                kind.stride_fn_name()
            ),
        ));
    }
    Ok(())
}

/// A byte count in whichever unit keeps it readable.
///
/// A fixed GiB was worse than useless at the small end: a 10 KiB overrun
/// printed as "0.0 GiB", which reads as "nothing" and is the one thing the
/// sentence must not say about a buffer the kernel is about to walk off.
fn magnitude_text(bytes: f64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    if bytes < MIB {
        format!("{:.1} KiB", bytes / KIB)
    } else if bytes < GIB {
        format!("{:.1} MiB", bytes / MIB)
    } else {
        format!("{:.1} GiB", bytes / GIB)
    }
}

/// The one refusal [`check_gpu_data_len`] can produce, in one place.
///
/// Names what was expected, what was found, and the field responsible, and
/// says how far the kernel would have read past the buffer -- because "your
/// array is the wrong length" is a worse message than "`score()` would have
/// uploaded 640 bytes and the shader would have read 16.0 GiB past them", and
/// only the second one tells a caller whether the file is damaged or the
/// reading of it is.
fn data_len_mismatch(
    kind: PayloadKind,
    dims: [usize; 3],
    data_len: usize,
    points: Option<u128>,
    expected: &str,
) -> DockError {
    let stride = kind.stride();
    let points_txt = match points {
        Some(p) => format!("{p} point(s)"),
        None => "more points than u128 can hold".to_string(),
    };
    // How far past the end of the uploaded buffer the kernel's last read
    // lands, and what that is in bytes. Only meaningful when the implied
    // length is a real number, so the overflow case says nothing rather than
    // printing a wrapped figure.
    let past = points.and_then(|p| p.checked_mul(stride as u128)).map(|e| {
        let last_read = e.saturating_sub(1);
        let ahead = last_read.saturating_sub(data_len as u128);
        (last_read, ahead)
    });
    // The two consequences are genuinely different facts about different
    // readers, and a refusal that blurred them would be a refusal that could
    // be true of neither. `energy.wgsl` reads the production payload through a
    // storage buffer the driver bounds, so a short one is a device-side read
    // past the end of an allocation. `terms_at` indexes a `Vec` with a plain
    // `[]`, so a short one is an index-out-of-bounds panic in the CPU
    // interpolator -- which is why the `TermMaps` copy of this rule is a
    // landmine defused rather than an exploit fixed, and why its text says
    // "panic" and not "reads garbage".
    let consequence = match (kind, past) {
        (PayloadKind::Grid, Some((last_read, ahead))) if ahead > 0 => format!(
            "energy.wgsl addresses a value as `grid[idx * {stride} + type_base + slot]`, \
             so for the last point it reads index {last_read}, and a buffer holding \
             {data_len} value(s) leaves {ahead} value(s) -- {magnitude} -- past the end \
             of what `score()` uploaded",
            magnitude = magnitude_text((ahead * 4) as f64)
        ),
        (PayloadKind::Grid, _) => format!(
            "energy.wgsl addresses a value as `grid[idx * {stride} + type_base + slot]`, \
             so a `data` of any other length makes it read either past the end of the \
             buffer or values belonging to a different point"
        ),
        (PayloadKind::Term, Some((last_read, ahead))) if ahead > 0 => format!(
            "TermMaps::terms_at indexes `data[idx * {stride} + type * TERM_FIELDS + field]`, \
             so for the last point it reads index {last_read}, and a `data` of \
             {data_len} value(s) leaves {ahead} value(s) -- {magnitude} -- past the end \
             of the vector. That index is not bounds-checked against `data.len()` -- \
             Rust checks it, so a short `data` is an index-out-of-bounds panic in the \
             interpolator, not a silent read past the allocation",
            magnitude = magnitude_text((ahead * 4) as f64)
        ),
        (PayloadKind::Term, _) => format!(
            "TermMaps::terms_at indexes `data[idx * {stride} + type * TERM_FIELDS + field]`, \
             so a `data` of any other length makes it index either past the end of the \
             vector or a value belonging to a different point"
        ),
    };
    // The closing sentence names the other rules this one is independent of,
    // and for `TermMaps` it has to be honest that one of them does not apply:
    // the point ceiling is a statement about `energy.wgsl`'s `u32` index, and
    // there is no shader that reads a `TermMaps`.
    let closing = match kind {
        PayloadKind::Grid => format!(
            "it is the grid. This is independent of the two addressability rules (the \
             {MAX_GRID_POINTS}-point ceiling and the two-points-per-axis shape rule), \
             both of which this grid satisfies."
        ),
        PayloadKind::Term => format!(
            "it is the tabulation. This is independent of the two-points-per-axis shape \
             rule, which this tabulation satisfies; the {MAX_GRID_POINTS}-point ceiling \
             is a statement about the `u32` flat index in energy.wgsl and says nothing \
             about a CPU-only vector, which is why it is not cited here as a rule this \
             one is independent of."
        ),
    };
    DockError::param(
        kind.field(),
        data_len,
        format!(
            "`data` holds {data_len} value(s) but dims {dims:?} imply {points_txt}, so it \
             must hold exactly {expected} -- {consequence}. A truncated or over-long \
             `data` has no reading that is correct, so the length is not padded, not \
             truncated and not tolerated: {closing}"
        ),
    )
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
#[serde(try_from = "GridMapsWire")]
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
    /// Receptor atoms whose PDBQT type this engine does not recognise.
    ///
    /// Carried here, and not only on [`crate::receptor::Receptor`], because the
    /// receptor is the object that goes *out of scope* the moment maps exist: a
    /// caller precalculates once and then holds only the maps, so a count that
    /// lives on the receptor alone cannot reach `dock()` and every energy the
    /// search returns is silently degraded with no way to tell. The engine does
    /// not refuse an unrecognised type — another tool may emit type names
    /// AutoDock never defined, and refusing would make the engine unusable with
    /// them — so carrying the number is the only thing standing between "the
    /// search worked" and "the search worked on 382 atoms whose types it could
    /// not read". See [`Receptor::unknown_atom_types`] for the policy this
    /// mirrors, and [`crate::docking::DockingResult::unknown_atom_types`] for
    /// where it surfaces to a caller.
    ///
    /// `#[serde(default)]` because a map file written before this field existed
    /// deserialises to 0, and "this map predates the counter" is not the same
    /// claim as "this map's receptor was fully recognised" — but refusing to
    /// read an older file would be worse than reporting the weaker claim.
    unknown_atom_types: usize,
}

/// The on-the-wire form of [`GridMaps`], and the reason `data` cannot lie.
///
/// `GridMaps` deserialises through this type rather than through its own fields
/// so that a map document whose `data` length disagrees with its `dims` is
/// rejected at load time instead of at `score()` time. `#[serde(try_from)]`
/// generates exactly the same `Deserialize` body the derive would have
/// written -- walk the map, fill these seven fields, build the value -- and
/// then hands the filled struct to [`TryFrom`], so the field-name contract is
/// the derive's contract and not a re-interpretation of it.
///
/// # The field contract is a product contract, and this struct has to keep it
///
/// Every field here has the same name, the same type and the same position as
/// the corresponding field of [`GridMaps`], in the same order, and
/// `unknown_atom_types` keeps the `#[serde(default)]` that lets a file written
/// before that counter existed still load. Changing any of that changes what a
/// `.map` file *means*, which is not a refactor: a file written by the previous
/// version of this crate would stop loading, or would load as something else.
///
/// The duplication is therefore guarded rather than trusted.
/// `grid.rs::the_wire_form_still_has_every_field_the_struct_has` round-trips a
/// real `GridMaps` through `serde_json` and fails if a field is added to one
/// side and not the other, and `the_wire_form_keeps_the_documented_field_names`
/// pins the seven names and their order. Unknown fields are still ignored,
/// which is what the derive did and what a file carrying a comment field from
/// a newer version needs.
#[derive(Deserialize)]
struct GridMapsWire {
    /// Lower corner of the grid, which equals [`GridBox::min`].
    min: Vec3,
    /// Number of grid points along each axis.
    dims: [usize; 3],
    /// Spacing between grid points, in Ångström.
    spacing: [f64; 3],
    /// Flat map storage. Its length is the whole point of this type.
    data: Vec<f32>,
    /// The box this grid covers.
    box_: GridBox,
    /// Heavy-atom coordinates of the receptor these maps were built from.
    receptor_heavy: Vec<Vec3>,
    /// See [`GridMaps::unknown_atom_types`]. Absent in files predating it.
    #[serde(default)]
    unknown_atom_types: usize,
}

impl TryFrom<GridMapsWire> for GridMaps {
    type Error = DockError;

    /// The only rejection here is the payload length.
    ///
    /// The two addressability rules -- the [`MAX_GRID_POINTS`] ceiling and the
    /// two-points-per-axis shape rule -- deliberately stay at
    /// [`GridMaps::check_gpu_index_range`], which is the gate `score()` goes
    /// through. Enforcing them here as well would make a map file unreadable
    /// for a reason the caller cannot act on without tabulating it, and would
    /// move two published refusals away from the place they are documented.
    /// This one is different: no arrangement of `dims` makes a wrong `data`
    /// length correct, so there is nothing to defer.
    fn try_from(w: GridMapsWire) -> Result<Self> {
        check_gpu_data_len(w.dims, w.data.len())?;
        Ok(GridMaps {
            min: w.min,
            dims: w.dims,
            spacing: w.spacing,
            data: w.data,
            box_: w.box_,
            receptor_heavy: w.receptor_heavy,
            unknown_atom_types: w.unknown_atom_types,
        })
    }
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

    /// Refuse a grid the GPU kernel's `u32` flat index cannot address.
    ///
    /// [`Self::precalculate`] enforces [`MAX_GRID_POINTS`] before it allocates,
    /// so a grid that came from it cannot fail this. `dims` is a public field
    /// and the type is `Deserialize`, so a hand-built or deserialised grid can
    /// carry any `dims` at all -- and `energy.wgsl` computes `idx * STRIDE` in
    /// `u32` (checked against the shader text by
    /// `the_shader_flat_index_is_u32_so_this_limit_is_the_right_one`, not by a
    /// line number in a comment), so a grid above the limit does not fault.
    /// It wraps, the host's `nx`/`ny`/`nz` are narrowed in the same expression,
    /// and the kernel returns a plausible number assembled from the wrong part
    /// of the buffer. The error therefore names the number that was exceeded
    /// rather than reporting an overflow, because there is no overflow to
    /// report: from inside the kernel everything looks fine.
    ///
    /// # The second rule, and why it is here rather than argued there
    ///
    /// This also refuses an axis holding **fewer than two points**, which is a
    /// different failure from an over-large one and used to be un-refused.
    ///
    /// The kernel clamps its cell with `f32(params.nx - 2u)`, and WGSL's `u32`
    /// subtraction *wraps*: on a one-point axis that bound is `4294967295`, so
    /// the clamp stops clamping and `cell + 1` is free to read anywhere. There
    /// is no legal cell to choose on such an axis at all, so the bound cannot
    /// be made safe in the shader -- the grid has to be refused.
    ///
    /// It was safe in practice, and by luck: `fractional` returns `None` on
    /// every point of a one-point axis, so the host's in-box flag was `0.0` and
    /// the kernel never reached the clamp. That is an invariant of a *different*
    /// function, in a different file, across a `pub(crate)` boundary, and the
    /// kernel's safety was resting on it silently. Checking it here turns that
    /// into a precondition the caller is told about.
    ///
    /// This is a behaviour change: such a grid used to score (as penalties on
    /// every atom, which is what a refusal and a charge agree on) and is now an
    /// error. Nothing in this crate can build one -- `estimate_dims` floors
    /// every axis at 2 -- so only a hand-built or deserialised grid is affected.
    ///
    /// # The third rule, and why it is a rule and not a note
    ///
    /// This also requires `data` to be exactly `points * map_stride()` values
    /// long. That is a third constraint, and it is independent of the two
    /// above: `check_gpu_dims` is about the `u32` index and never looks at the
    /// payload, and the shape rule is about whether an interpolable cell
    /// exists and does not either. A grid can pass both and still be short.
    /// One did -- 160 floats beside `dims` for 107,374,182 points -- and
    /// reached `score()`, which uploaded the 160 and let the shader read
    /// 4,294,967,120 values past the end of them.
    ///
    /// The deserialiser refuses that document as well, at load, which is
    /// earlier and cheaper for a caller. The check is here too because this is
    /// the gate the kernel actually goes through, and a precondition that
    /// lives only in the parser is a precondition about one construction
    /// route: `data` is a private field, so any future in-crate path that
    /// fills it wrongly is not covered by it.
    ///
    /// This is the second behaviour change on this method. A map file whose
    /// `data` is truncated or over-long used to load, used to pass every
    /// check in the crate, and used to be uploaded; it is now an error at
    /// load. Nothing this crate writes is affected -- `precalculate` builds
    /// `data` at the length `dims` implies.
    pub fn check_gpu_index_range(&self) -> Result<()> {
        check_gpu_dims(self.dims)?;
        // Rule 1 of 3, above: the point count is one the kernel's u32 flat
        // index can address. Rule 2 is the shape, below. Rule 3 is the
        // payload, at the foot. They are three separate questions, and a
        // grid can answer two of them yes and still be one value short.
        if let Some(axis @ k) = (0..3).find(|&k| self.dims[k] < 2) {
            return Err(DockError::param(
                "grid dims",
                format!("{:?}", self.dims),
                format!(
                    "axis {k} holds {} point(s). energy.wgsl interpolates trilinearly, \
                     so it reads two points along every axis and clamps its cell to \
                     `n - 2`; in WGSL that subtraction is u32 and wraps, so on a \
                     one-point axis the clamp's bound is 4294967295 and stops holding \
                     the corner reads inside the grid. A grid with fewer than two \
                     points on any axis has no interpolable cell at all",
                    self.dims[axis]
                ),
            ));
        }
        // Third and last, and the only rule about the payload rather than
        // the shape. Running it last is deliberate: a grid that is both
        // degenerate and inconsistent is reported as degenerate, because
        // "this grid has no interpolable cell" is the more fundamental claim,
        // and it is the one a caller can act on without first having to know
        // how long the file claims to be.
        check_gpu_data_len(self.dims, self.data.len())?;
        Ok(())
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
    ///
    /// # Refused here, and why
    ///
    /// `box_holds_a_receptor_atom` runs before any tabulation: a box with no
    /// receptor atom in it is all zeros, and an all-zero map set is a docking
    /// run that reports success for work it did not do. It runs *after* the
    /// spacing and grid-size checks so that a request which is wrong in two
    /// ways is still reported in terms of the one the caller can fix without
    /// moving the box.
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

        // See `box_holds_a_receptor_atom` for why this is a refusal and not a
        // warning. Both tabulations go through it, so a diagnostic can never
        // disagree with the energy it is a breakdown of.
        box_holds_a_receptor_atom(receptor, box_)?;

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
            unknown_atom_types: receptor
                .atoms
                .iter()
                .filter(|a| a.atom_type == crate::types::AtomType::Unknown)
                .count(),
        })
    }

    /// Receptor atoms whose PDBQT type this engine does not recognise.
    ///
    /// Zero for maps built by hand rather than by
    /// [`GridMaps::precalculate`], and for maps deserialised from a file written
    /// before this count existed — neither of those is a claim that every atom
    /// was recognised. Read it as "at least this many are unrecognised", which
    /// is the only reading that stays true in all three cases.
    pub fn unknown_atom_types(&self) -> usize {
        self.unknown_atom_types
    }

    /// Heavy-atom coordinates of the receptor these maps were precalculated
    /// from. Empty only for maps built by hand rather than by
    /// [`GridMaps::precalculate`].
    pub fn receptor_atoms(&self) -> &[[f64; 3]] {
        &self.receptor_heavy
    }

    /// Fractional cell coordinate of `p`, or `None` if it is outside the grid.
    ///
    /// # Why this stays `f64`
    ///
    /// `energy.wgsl` **used to** evaluate the same expression in `f32` and
    /// branch on the result, and the function is **discontinuous** at the face,
    /// so the two could take different branches on a point they should agree
    /// about. Measured on an RTX 3050:
    /// a 24x12x12 A box at the shipped 0.375 A spacing, receptor mass 0.5 A
    /// from the +x face, a carbon probe at `min + (n-1)*spacing - 9.5e-7 A`
    /// scores **15.540158** here and **0.000000** on the GPU, while placements
    /// that agree on the branch differ by at most 4.8e-4. So the disagreement
    /// is ~3.3e4 times the rounding regime around it.
    ///
    /// Evaluating *this* in `f32` instead does make the two expressions
    /// identical, and it was tried. It is reverted, and the reason is worth
    /// keeping: `u` near 12 quantises to ~9.5e-7 in `f32`, which is 3.6e-7 A of
    /// `p`, so the energy as a function of position becomes a staircase with
    /// 3.6e-7 A treads. Any central difference over a step smaller than a tread
    /// is then measuring the staircase rather than the surface -- with the
    /// suite's own `h = 1e-6 A` the numeric gradient came out 5.510610 against
    /// an analytic 6.163513, an 11% error in a test that had been exact, and a
    /// seeded docking run moved off its pinned pose. The analytic gradient
    /// would have been right and the *diagnostic* would have been lying, which
    /// is the worse outcome: it removes the instrument that catches the next
    /// error.
    ///
    /// So the CPU keeps `f64` and the host asks this function for the decision
    /// instead of taking it: `gpu::GpuContext::score` calls *this* method on the
    /// un-narrowed coordinate and uploads the answer alongside the narrowed
    /// one, and the kernel branches on that flag rather than re-deriving it.
    /// One evaluation, two consumers -- so the two backends cannot disagree
    /// about whether a point is in the box, and what is left to round is the
    /// interpolation arithmetic, which is worth `O(eps)` rather than `O(|g|)`.
    /// See `gpu::Batch` and the header of `energy.wgsl`.
    ///
    /// It is `pub(crate)` for exactly that one caller. It is not `pub`: the
    /// `Some` arm returns the cell and the fractions, which are an internal
    /// representation, and a public signature would freeze them into the API.
    ///
    /// # Why the comparisons are negated
    ///
    /// `!(i >= 0.0)` rather than `i < 0.0`, for the same reason
    /// [`GridBox::new`] writes `!(max > min)`: every comparison with `NaN` is
    /// false, so the positive form lets a `NaN` coordinate through as if it
    /// were inside, and the caller then interpolates with `NaN` weights and
    /// accumulates a `NaN` energy. Here it returns `None`, which is a refusal,
    /// and [`out_of_box_violation_per_axis`] is where the charge for that
    /// refusal is decided.
    ///
    /// # Why `dims[k] - 1` is written in `f64`
    ///
    /// It used to be `self.dims[0] - 1` in `usize`, which underflows for a
    /// deserialised grid with a zero-length axis: a subtraction panic in debug
    /// and a wrapped `usize::MAX` in release, so one corrupt input panicked two
    /// different ways. `self.dims[0] as f64 - 1.0` is `-1.0` for a zero axis and
    /// no coordinate is below it, so the answer is `None`. Nothing in this
    /// crate can build such a grid -- `estimate_dims` floors every axis at 2 --
    /// so this refuses a hand-written or corrupt payload rather than a
    /// reachable request.
    ///
    /// Whether `p` is inside the tabulated volume, on the CPU's terms.
    ///
    /// `gpu::GpuContext::score` uses this as the in-box flag it uploads beside
    /// each narrowed coordinate, so that the kernel's branch *is* this decision
    /// rather than a second evaluation of the same expression in a different
    /// precision. See the note on the comparisons below for why that matters,
    /// and `gpu::Batch` for the history.
    #[inline]
    pub(crate) fn fractional(&self, p: Vec3) -> Option<([f64; 3], [usize; 3])> {
        let mut frac = [0.0f64; 3];
        let mut cell = [0usize; 3];
        for k in 0..3 {
            let u = (p[k] - self.min[k]) / self.spacing[k];
            let i = u.floor();
            let n = self.dims[k] as f64 - 1.0;
            // `contains` is false for NaN, so a NaN cell index is rejected
            // here exactly as the two negated comparisons it replaced rejected
            // it. Do not rewrite this as `i < 0.0 || i >= n`: both comparisons
            // are false for NaN, so that form would let a NaN through.
            if !(0.0..n).contains(&i) {
                return None;
            }
            cell[k] = i as usize;
            frac[k] = u - i;
        }
        Some((frac, cell))
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
/// How many receptor atoms fall inside `box_`, and how far the nearest one is.
///
/// # Why an empty box is refused rather than reported
///
/// Every map in this module is a sum over receptor atoms, and an atom
/// contributes nothing beyond `kernels.cutoff + radius` of any grid point. A
/// box holding no receptor atom is therefore tabulated as *exactly zero
/// everywhere*, and the search then runs over that quite normally: a full
/// Monte-Carlo walk, a ranked list of poses, and a best affinity. Measured on
/// the shipped 30-atom receptor with an 18 Å box displaced 400 Å along every
/// axis, all three reported poses came back with an intermolecular energy of
/// precisely `0.0` and a best affinity of `-0.0122` kcal/mol -- ranked,
/// timed, and printed by `summary()` with no warning of any kind.
///
/// Nothing downstream can catch it. The poses are mutually consistent, they are
/// clash-free by construction because there is nothing to clash with,
/// `rejected_pose_count` is `0` (which reads as good news and is not), and
/// `unknown_atom_types` is `0`. The search genuinely ran; it ran in a vacuum,
/// and the number it reports is the ligand's own internal energy rather than
/// any interaction with the receptor.
///
/// This is a refusal and not a flag, unlike an unrecognised atom type. An
/// unrecognised type is a *graded* loss -- the atom still contributes its shape
/// term, and a count of them lets a caller weigh what was lost. Zero receptor
/// atoms in the box admits no such reading: every intermolecular term in the
/// result is identically zero, and there is nothing to weigh. A box that does
/// overlap the receptor but is far larger than the site is a different question
/// and is deliberately left alone -- a whole-protein redock is a legitimate
/// configuration, and `rejected_pose_count` keeps reporting the
/// buried-in-protein poses it produces.
///
/// The message quotes the distance to the nearest atom because "no receptor
/// atom in the box" alone does not distinguish a box that misses by a fraction
/// of an ångström from one that misses by a hundred, and those two want very
/// different corrections.
fn box_holds_a_receptor_atom(receptor: &Molecule, box_: &GridBox) -> Result<()> {
    let mut in_box = 0usize;
    let mut nearest = (f64::INFINITY, [0.0f64; 3]);
    for a in &receptor.atoms {
        if box_.contains(a.coord) {
            in_box += 1;
        }
        // `out_of_box_violation`, not a second spelling of "how far outside".
        // The first version of this used the per-axis form without its
        // trailing `.max(0.0)`, so an axis the atom was *inside* contributed a
        // negative gap, got squared, and the reported distance was larger than
        // the truth -- 35.0 Å for an atom 34 Å away, on a box offset in x only.
        // This module already states one definition of that quantity and says
        // why a second copy is how a diagnostic ends up disagreeing with the
        // number it explains.
        let d = out_of_box_violation(a.coord, box_);
        if d < nearest.0 {
            nearest = (d, a.coord);
        }
    }
    if in_box == 0 {
        return Err(DockError::Grid(format!(
            "the search box {:?}..{:?} contains none of the {} receptor atoms \
             (nearest is {:.1} Å away, at {:?}): every map over this box is \
             identically zero, so a search here ranks poses by the ligand's own \
             internal energy and reports them as a binding mode. Move the box \
             onto the site, or check that the receptor and the box were not \
             given in different coordinate frames.",
            box_.min,
            box_.max,
            receptor.atoms.len(),
            nearest.0,
            nearest.1,
        )));
    }
    Ok(())
}

fn slot_tag(slot: MapSlot) -> &'static str {
    match slot {
        MapSlot::Shape => "e",
        MapSlot::HbFromDonor => "hbdon",
        MapSlot::HbFromAcceptor => "hbacc",
        MapSlot::Hydrophobic => "hyd",
    }
}

/// How far `p` lies outside `b` on each of the three axes, in angstroms.
///
/// Zero on an axis the atom is inside; measured against the *faces*, not from
/// the centre, so an atom outside on +x is one axis wrong and nothing more.
/// An atom outside a corner is wrong on both axes, and both are charged: it
/// has to be walked back in through both.
///
/// This is the single definition of "outside by how much", and both places
/// that charge [`crate::search::OUT_OF_BOX_PENALTY`] go through it -- the
/// search, which needs the per-axis values to build the ramp's gradient, and
/// [`TermMaps::conformation_terms`], which reports the charge as a field. Two
/// copies of this is how a diagnostic ends up disagreeing with the energy it
/// is a breakdown of: the penalty used to be a flat step here while the search
/// charged a distance, and the two numbers were never compared.
///
/// # A non-finite coordinate is an infinite violation, on purpose
///
/// Three answers were available to a `NaN` or `inf` coordinate: return a `Some`
/// carrying the `NaN`, return `None` and charge nothing, or return `None` and
/// charge something unmistakable. The first is what the callers used to get and
/// it is the worst of the three: `f64 as usize` saturates, so `NaN` became cell
/// `0`; every comparison with `NaN` is false, so the bounds check passed; and
/// the caller interpolated with `NaN` weights and accumulated a `NaN` energy.
/// A `NaN` energy compares false against every number, so it neither wins nor
/// loses a ranking -- it makes the ranking meaningless while looking like a
/// result.
///
/// The second is subtler than it looks, and is why this needed a decision rather
/// than a comment. `f64::max` returns the non-`NaN` operand, so
/// `(NaN - max).max(min - NaN).max(0.0)` is `0.0`, not `NaN`: a `None` alone
/// charges **nothing** for a coordinate that is not a number, and the atom
/// silently contributes zero. That is a plausible-looking answer, which is the
/// problem.
///
/// So the charge is [`f64::INFINITY`] on every axis. It is not a distance -- no
/// finite distance is right for a coordinate that is not a number -- and it is
/// chosen for what it does downstream: `OUT_OF_BOX_PENALTY * inf` is `inf`,
/// every comparison against `inf` is well defined, and the conformation sorts
/// last instead of vanishing. A loud, well-ordered wrong answer beats a quiet
/// right-looking one.
///
/// The GPU never reaches this: `search/mod.rs` declines the whole batch when a
/// coordinate is non-finite, so there is no `f32` counterpart to agree with.
#[must_use]
pub fn out_of_box_violation_per_axis(p: Vec3, b: &GridBox) -> [f64; 3] {
    let mut v = [0.0f64; 3];
    for k in 0..3 {
        v[k] = if p[k].is_finite() {
            (p[k] - b.max[k]).max(b.min[k] - p[k]).max(0.0)
        } else {
            f64::INFINITY
        };
    }
    v
}

/// How far `p` lies outside `b` in total, in angstroms: the three axes of
/// [`out_of_box_violation_per_axis`] added together.
#[must_use]
pub fn out_of_box_violation(p: Vec3, b: &GridBox) -> f64 {
    out_of_box_violation_per_axis(p, b).iter().sum()
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
#[serde(try_from = "TermMapsWire")]
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

/// The on-the-wire form of [`TermMaps`], and the same closing `GridMaps` got.
///
/// # This is a landmine defused, not an exploit fixed
///
/// `TermMaps` is the production `GridMaps` shape exactly: a private `data`, a
/// public `dims`, a derived `Deserialize`, and a stride that is implied by
/// `dims` and stored nowhere. When `GridMaps` was found to deserialise a
/// document whose `data` length disagreed with its `dims` -- and to hand that
/// to the GPU, where the shader addresses up to index 4,294,967,279 in a
/// 640-byte buffer -- `TermMaps` was the same hole with the same shape, one
/// file over, and it was left open.
///
/// **Nothing reaches a device through a `TermMaps` today.** `TermMaps::BACKEND`
/// is `TermBackend::Cpu`, the WGSL entry point takes a `&GridMaps` and there is
/// no overload, no generic and no trait that would let a `TermMaps` be passed
/// instead, and `grid.rs::under_the_gpu_feature_the_terms_are_still_cpu_only`
/// asserts all three of those on every `gpu` build. So the blast radius today
/// is not a device read: `terms_at` indexes `self.data` with a plain `[]`, and
/// Rust bounds-checks that, so a short `data` is an **index-out-of-bounds
/// panic** in the CPU interpolator, not a silent read of whatever follows the
/// allocation. No information disclosure, no memory corruption.
///
/// What it is instead is a *landmine*: a hole that is inert only because no
/// code path reaches a device with this type, laid in the same place as the
/// live one and one refactor away from being live. A future per-term GPU path
/// -- which `grid.rs::the_per_term_decomposition_is_cpu_only_and_says_so` is
/// already watching for -- would inherit it exactly as `GridMaps` did. The
/// hole is closed now, on the same terms as `GridMaps`, through the same shared
/// rule ([`check_payload_len`]), with the same refusal text shape, and the
/// both-directions proof in
/// `tests/term_data_len.rs` is the same one `tests/map_data_len.rs` makes.
///
/// # The field contract
///
/// Every field here has the same name, type and position as the corresponding
/// field of [`TermMaps`], in the same order, and there is no `#[serde(default)]`
/// on any of them -- `TermMaps` has never carried a field that predates a
/// format, so there is nothing to default and inventing one would let a
/// document that should be refused load instead. The duplication is guarded
/// rather than trusted: `tests/term_data_len.rs::the_term_wire_form_keeps_the_
/// field_names_the_format_uses` pins the names and the order, and
/// `the_term_wire_form_still_has_every_field_the_struct_has` round-trips a real
/// `TermMaps` and fails if a field is added to one side and not the other.
#[derive(Deserialize)]
struct TermMapsWire {
    /// Lower corner of the grid, equal to [`GridBox::min`].
    min: Vec3,
    /// Number of grid points along each axis.
    dims: [usize; 3],
    /// Spacing between grid points, in Ångström.
    spacing: [f64; 3],
    /// Flat storage. Its length is the whole point of this type.
    data: Vec<f32>,
    /// The box this grid covers.
    box_: GridBox,
}

impl TryFrom<TermMapsWire> for TermMaps {
    type Error = DockError;

    /// The only rejection here is the payload length, for the same reason
    /// `GridMaps`' is: the two addressability rules stay at the gate the
    /// reader goes through, and no arrangement of `dims` makes a wrong `data`
    /// length correct, so there is nothing to defer.
    fn try_from(w: TermMapsWire) -> Result<Self> {
        check_term_data_len(w.dims, w.data.len())?;
        Ok(TermMaps {
            min: w.min,
            dims: w.dims,
            spacing: w.spacing,
            data: w.data,
            box_: w.box_,
        })
    }
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

        // The same refusal [`GridMaps::precalculate`] makes, at the same point
        // in the sequence. A diagnostic that could tabulate a box the production
        // path refuses would be able to show a caller an all-zero breakdown of a
        // box that never gets searched, which is the disagreement this module
        // already treats as the thing worth preventing.
        box_holds_a_receptor_atom(receptor, box_)?;

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
    /// `OUT_OF_BOX_PENALTY x (how far out it is)` to
    /// [`crate::scoring::TermBreakdown::out_of_box_penalty`] rather than to any
    /// term, so the penalty cannot be mistaken for a score. The charge is the
    /// one the search charges, through [`out_of_box_violation`]: this is a
    /// decomposition of the engine's intermolecular energy, so a different
    /// number here is not a different opinion, it is a wrong one.
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
                None => {
                    acc.out_of_box_penalty += crate::search::OUT_OF_BOX_PENALTY
                        * out_of_box_violation(coords[i], &self.grid_box());
                }
            }
        }
        acc
    }

    /// Fractional cell coordinate of `p`, or `None` if it is outside the grid.
    ///
    /// The same expression and the same `NaN` and zero-axis handling as
    /// [`GridMaps::fractional`], and it is transcribed rather than shared
    /// because the two types have genuinely different obligations.
    /// `GridMaps` has to agree with `energy.wgsl` on a discontinuous
    /// predicate, so the host asks [`GridMaps::fractional`] for that decision
    /// and uploads the answer ([`crate::gpu::Batch`]); this type is CPU-only
    /// and structurally so -- [`Self::BACKEND`] is
    /// [`crate::scoring::TermBackend::Cpu`] and the GPU entry point takes a
    /// `&GridMaps`, so no path carries these fields to a shader and there is
    /// nothing here to agree with. The interpolation stays `f64`.
    #[inline]
    fn fractional(&self, p: Vec3) -> Option<([f64; 3], [usize; 3])> {
        let u0 = (p[0] - self.min[0]) / self.spacing[0];
        let u1 = (p[1] - self.min[1]) / self.spacing[1];
        let u2 = (p[2] - self.min[2]) / self.spacing[2];
        let i0 = u0.floor();
        let i1 = u1.floor();
        let i2 = u2.floor();
        let n0 = self.dims[0] as f64 - 1.0;
        let n1 = self.dims[1] as f64 - 1.0;
        let n2 = self.dims[2] as f64 - 1.0;
        // Same NaN reasoning as the crate-level `fractional` above: a range
        // `contains` is false for NaN, so a NaN cell index is rejected. A
        // `i < 0.0 || i >= n` rewrite would not be equivalent.
        if !(0.0..n0).contains(&i0) || !(0.0..n1).contains(&i1) || !(0.0..n2).contains(&i2) {
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

    // ---------------------------------------------------------------------
    // A box holding no receptor atom. See `box_holds_a_receptor_atom` for why
    // this is a refusal; the measurement that made it one is below.

    #[test]
    fn a_box_with_no_receptor_atom_in_it_is_refused() {
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0), carbon(2.0, 0.0, 0.0)]).unwrap();
        // Same receptor, box moved 40 Å along x. Every map over it is zero, and
        // before the refusal a search here returned three ranked poses with an
        // intermolecular energy of exactly 0.0 and a best affinity of
        // -0.0122 kcal/mol -- a binding mode in a vacuum.
        let away = GridBox::new([34.0, -6.0, -6.0], [46.0, 6.0, 6.0]).unwrap();
        let err = GridMaps::precalculate(&rec, &away, &scoring(), 0.5, 1)
            .expect_err("a box containing no receptor atom must be refused");
        let msg = err.to_string();
        assert!(msg.contains("none of the 2 receptor atoms"), "got: {msg}");
        // The distance is the part that makes the message actionable: a box
        // that misses by 32 Å and one that misses by 0.2 Å need different
        // corrections, and "no receptor atom in the box" says neither. The
        // nearest of the two atoms is the one at x = 2.0 and the box starts at
        // x = 34, so the violation is exactly 32.0 Å -- asserted rather than
        // recomputed, so a wrong distance formula cannot pass by being
        // self-consistent with itself.
        assert!(msg.contains("32.0 Å away"), "got: {msg}");
        assert!(msg.contains("[2.0, 0.0, 0.0]"), "got: {msg}");
    }

    #[test]
    fn the_diagnostic_tabulation_refuses_the_same_box() {
        // `TermMaps` exists to explain a `GridMaps` result. If it could
        // tabulate a box the production path refuses, it would happily show a
        // caller an all-zero breakdown of geometry that never gets searched.
        let rec = Molecule::from_atoms(vec![carbon(0.0, 0.0, 0.0)]).unwrap();
        let away = GridBox::new([94.0, -6.0, -6.0], [106.0, 6.0, 6.0]).unwrap();
        assert!(TermMaps::precalculate(&rec, &away, &scoring(), 0.5, 1).is_err());
        assert!(GridMaps::precalculate(&rec, &away, &scoring(), 0.5, 1).is_err());
    }

    #[test]
    fn a_box_containing_one_receptor_atom_is_not_refused() {
        // The other direction, and the boundary that matters: the check counts
        // atoms *in* the box, so a box that clips a single atom of a large
        // receptor is accepted. That is a legitimate -- if poor -- pocket
        // choice, and refusing it would remove a capability rather than fix a
        // silent success.
        let rec = Molecule::from_atoms(vec![
            carbon(0.0, 0.0, 0.0),
            carbon(2.0, 0.0, 0.0),
            carbon(0.0, 2.0, 0.0),
            carbon(2.0, 2.0, 0.0),
        ])
        .unwrap();
        // Clips only the atom at the origin.
        let corner = GridBox::new([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]).unwrap();
        assert!(GridMaps::precalculate(&rec, &corner, &scoring(), 0.5, 1).is_ok());
    }

    #[test]
    fn a_box_larger_than_the_site_is_still_accepted() {
        // A whole-protein redock is a legitimate configuration and must keep
        // working: the refusal is for a box that *misses* the receptor, not for
        // one that swallows it. `rejected_pose_count` is what reports the
        // buried-in-protein poses a big box produces.
        let rec = Molecule::from_atoms(vec![
            carbon(0.0, 0.0, 0.0),
            carbon(4.0, 0.0, 0.0),
            carbon(0.0, 4.0, 0.0),
        ])
        .unwrap();
        let huge = GridBox::new([-60.0, -60.0, -60.0], [60.0, 60.0, 60.0]).unwrap();
        assert!(GridMaps::precalculate(&rec, &huge, &scoring(), 1.0, 1).is_ok());
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
        // 100 A out on +x of a box that ends at 6, so 94 A of violation and
        // 94 penalties. The charge used to be one penalty at every distance:
        // this breakdown is a decomposition of the engine's intermolecular
        // energy, so a number that is not the energy is a wrong one.
        let far =
            crate::search::OUT_OF_BOX_PENALTY * out_of_box_violation([100.0, 0.0, 0.0], &box_);
        assert!(
            (b.out_of_box_penalty - far).abs() < 1e-9,
            "an atom 94 A outside must cost {far}, got {}",
            b.out_of_box_penalty
        );
        assert!((b.total() - far).abs() < 1e-9);
        // The shape, not just the existence: one angstrom further out costs
        // exactly one penalty more, which a flat charge cannot show.
        let nearer = terms.conformation_terms(&[ti], &[mask], &[[99.0, 0.0, 0.0]]);
        assert!(
            (b.out_of_box_penalty - nearer.out_of_box_penalty - crate::search::OUT_OF_BOX_PENALTY)
                .abs()
                < 1e-9,
            "one A further out costs {} more, against one penalty",
            b.out_of_box_penalty - nearer.out_of_box_penalty
        );
        // And an atom outside two faces is charged for both, not for one atom
        // twice: this is the per-axis sum the gradient is the derivative of.
        let corner = terms.conformation_terms(&[ti], &[mask], &[[100.0, -100.0, 0.0]]);
        assert!(
            (corner.out_of_box_penalty - 2.0 * far).abs() < 1e-9,
            "a corner escape is charged {}, against {far} for one face",
            corner.out_of_box_penalty
        );
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
    /// `docs/SCORING.md` §5.3, as it reads now.
    ///
    /// This used to quote `cutoff + 2*R`, which §5.3 now explicitly calls the
    /// obsolete 0.4 Å-era form, so the quote described a document that no
    /// longer existed. It is now checked rather than trusted:
    /// [`every_pinned_documentation_line_still_exists`] fails if the sentence
    /// below stops appearing in `SCORING.md`.
    const REACH_DOC: &str = "SCORING.md §5.3: `SpatialKernels::cutoff = 8.0` 是表面距离的上限，而内循环手里是真实距离，\
                              所以外层预筛要多出两个半径——**最宽的那一块**的半径，加上受体原子自己的:\n\
                              cutoff + MAX_GRID_TYPE_RADIUS + a.radius";

    pub(super) fn assert_pinned<T: PartialEq + std::fmt::Debug>(
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

    /// The WGSL kernel's source text, in CPU-only builds as well as `gpu` ones.
    ///
    /// `gpu::ENERGY_WGSL` is the same file read by the same `include_str!`, so
    /// this is not a second copy of the shader that could drift from the one
    /// the pipeline is built from. The check below has to run in **both**
    /// feature configurations because the `u32` claim is about *this crate's
    /// advertised ceiling*, and a build with the `gpu` feature off makes the
    /// same claim in the same doc comment.
    const ENERGY_WGSL_SOURCE: &str = include_str!("gpu/energy.wgsl");

    /// Whether `energy.wgsl` still computes its flat grid index in `u32`.
    ///
    /// [`MAX_GRID_POINTS`] is `(u32::MAX) / STRIDE`, and the reason is a
    /// property of the shader rather than of this file. If the kernel ever
    /// widens that index, the published ceiling becomes far too *small* rather
    /// than too large -- a 16 GiB grid the kernel could address refused as
    /// un-buildable -- and nothing else in the crate would notice, because
    /// every other statement about the limit is a restatement of this one.
    ///
    /// So the claim is read out of the shader's own text. It is deliberately a
    /// check *of* the shader and not a re-derivation *of* it: it looks for the
    /// names the host and this crate already bind together elsewhere (`nx`,
    /// `ny`, `nz` and `STRIDE`, all declared `u32`) and for the multiply that
    /// turns a point index into a flat value. Transcribing the arithmetic here
    /// would let the check pass by being wrong the same way twice; naming the
    /// identifiers cannot. The price is that renaming them in the shader turns
    /// it red, which is the intended behaviour -- a rename is a request to
    /// re-read the shader, not a defect in it, and the test that calls this
    /// says so in its message.
    fn shader_flat_index_is_u32() -> bool {
        ENERGY_WGSL_SOURCE.contains("const STRIDE: u32")
            && ENERGY_WGSL_SOURCE.contains("nx: u32")
            && ENERGY_WGSL_SOURCE.contains("ny: u32")
            && ENERGY_WGSL_SOURCE.contains("nz: u32")
            && ENERGY_WGSL_SOURCE.contains("* STRIDE")
    }

    /// The shader no longer derives the out-of-box violation from its own
    /// coordinate.
    ///
    /// `params.bmax` is still in the uniform block -- it is part of the 80-byte
    /// layout that `gpu::grid_params_matches_the_wgsl_uniform_layout` pins and
    /// that `docs/SCORING.md` §9.2 documents -- but `sample()` must not read it.
    ///
    /// It used to, to measure the violation from the narrowed `f32` position
    /// while the CPU measured it from the `f64`. Both sides took the same
    /// branch and charged different amounts. The field staying in the struct
    /// with a comment that says it drives the penalty would recreate exactly
    /// that disagreement the next time somebody trusted the comment, so the
    /// absence of the read is asserted rather than described.
    ///
    /// The same shape of argument would not hold for the *branch*: the branch is
    /// the host's flag and there is no second evaluation of it to forbid. This
    /// is the one quantity that must not be computed in the kernel at all, and
    /// the fact that it is a *different* quantity from the one above is the
    /// whole distinction.
    fn the_shader_does_not_measure_the_out_of_box_violation() -> bool {
        !ENERGY_WGSL_SOURCE.contains("params.bmax")
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

    /// Every fragment this crate quotes from `docs/SCORING.md` still exists in
    /// `docs/SCORING.md`.
    ///
    /// [`assert_pinned`] compares a *number in the code* with a *number typed
    /// into a test*, and prints the documented line when they differ. That
    /// catches the code moving. It cannot catch the **document** moving: a quote
    /// in a source file is a string literal, and a string literal never notices
    /// that the paragraph it was copied from has been rewritten, deleted, or
    /// reversed. That is not hypothetical. Three comments in `dock-core`
    /// outlived the text they described, each asserting something `SCORING.md`
    /// had already stopped saying, and all three were found by a reader rather
    /// than by a test.
    ///
    /// So this is the missing half: read the document, and fail if a fragment
    /// this crate still cites has stopped being in it. It is a *fragment* match
    /// rather than a whole-line one because the quotes above are re-wrapped for
    /// source layout and their whitespace cannot be compared literally, while
    /// the distinctive part of each sentence can be.
    ///
    /// What this does **not** do is check that a surviving fragment still says
    /// what it used to say. A reworded sentence that keeps the identifier is
    /// invisible here, and that is a real limit rather than a solved problem.
    ///
    /// **That limit was measured, not assumed, and it is worse than "a
    /// reworded sentence".** Mutating the §5.3 needle to the *superseded* form
    /// `cutoff + 2.0 * a.radius` — the exact line this comment used to assert
    /// the document said, and the one whose replacement is the whole point of
    /// the fix above — leaves this test **green**. It is green because §5.3
    /// still contains that string: it names the 0.4 Å-era form in order to
    /// call it obsolete. So for a section that discusses both the old and the
    /// new form, substring presence cannot tell a citation of the live one from
    /// a citation of the dead one, and no amount of fragment tuning will make
    /// it.
    ///
    /// What this test does establish is that a cited section still **exists**
    /// and that its distinctive formula is still written down somewhere in it:
    /// it caught three real cosmetic mismatches on its first run, all of them
    /// from whitespace and a Unicode minus, and all three of which would
    /// otherwise have been reported as drift. A guard that over-reports is a
    /// guard that gets deleted, so the cosmetic normalising is not a nicety.
    /// What it cannot establish is which of two coexisting forms is the live
    /// one, and claiming otherwise here would be the third stale comment this
    /// round was about.
    #[test]
    fn every_pinned_documentation_line_still_exists() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/SCORING.md");
        let doc = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!(
                "could not read {path}: {e}. This test is the only thing that notices \
                 when a quoted specification stops existing, so it cannot also be the \
                 thing that silently stops running"
            )
        });

        // (the name this crate cites it under, the fragment that must still be there)
        //
        // The needles are written against a *normalised* copy of the document:
        // runs of whitespace collapsed to one space, and the typographic minus
        // U+2212 folded onto ASCII `-`. Both differences are cosmetic and both
        // would otherwise make this test report drift where none exists — the
        // first run of it did exactly that, failing on three sections because
        // `SCORING.md` aligns `hb(d)` in a code block with U+2212 while the
        // quote in this file used a hyphen. A guard that cries wolf on
        // whitespace is a guard that gets deleted.
        let norm = |s: &str| {
            let mut out = String::with_capacity(s.len());
            let mut space = false;
            for c in s.chars() {
                // Folded to the ASCII spelling, not to a single character, so
                // that a needle written the way a reader would type the
                // inequality still matches.
                if c == '\u{2264}' {
                    out.push_str("<=");
                    space = false;
                    continue;
                }
                if c == '\u{2265}' {
                    out.push_str(">=");
                    space = false;
                    continue;
                }
                let c = if c == '\u{2212}' { '-' } else { c };
                if c.is_whitespace() {
                    space = true;
                    continue;
                }
                if space {
                    out.push(' ');
                    space = false;
                }
                out.push(c);
            }
            // And the space immediately before a folded comparison operator goes
            // too. `SCORING.md` §2.3 writes `当 x≤ a` in one place and
            // `当 x <= a` in another, because the line is aligned in a code
            // block; a needle that had to guess which would be a needle that
            // reports drift where the document has not drifted. Two rules, both
            // cosmetic, both learned from a first run of this test that failed
            // on exactly this.
            let mut tidy = String::with_capacity(out.len());
            for c in out.chars() {
                if (c == '<' || c == '>') && tidy.ends_with(' ') {
                    tidy.pop();
                }
                tidy.push(c);
            }
            tidy
        };
        let doc = norm(&doc);
        let cited: &[(&str, &str)] = &[
            ("§2.3 the smoothstep", "S(a, b, x) = 0 当 x <= a"),
            ("§2.3 the hbond window", "hb(d) = 1 - S(-0.5, 0.0, d)"),
            (
                "§2.3 the hydrophobic window",
                "hyd(d) = 1 - S( 0.5, 1.5, d)",
            ),
            ("§4.1 the intramolecular minimum", "MIN_INTRA_BOND_DISTANCE"),
            (
                "§4.2 the out-of-box formula",
                "E_penalty = OUT_OF_BOX_PENALTY × violation",
            ),
            ("§4.2 the constant itself", "OUT_OF_BOX_PENALTY = 1000.0"),
            (
                "§4.2 the per-axis violation",
                "v[k] = (p[k] − b.max[k]) .max (b.min[k] − p[k]) .max 0",
            ),
            ("§5.1 the four slots", "`HbFromDonor`"),
            (
                "§5.2 the map stride",
                "STRIDE = GRID_TYPE_COUNT · MAPS_PER_TYPE",
            ),
            (
                "§5.3 the precalculation reach",
                "cutoff + MAX_GRID_TYPE_RADIUS + a.radius",
            ),
            ("§5.3 the surface distance", "let d = r - (rp + a.radius);"),
            ("§6.4 the default spacing", "0.375 Å"),
        ];

        let missing: Vec<&str> = cited
            .iter()
            .filter(|(_, needle)| !doc.contains(&norm(needle)))
            .map(|(label, _)| *label)
            .collect();
        let n = missing.len();
        assert!(
            missing.is_empty(),
            "dock-core cites {n} section(s) of docs/SCORING.md that are no longer \
             there: {missing:?}.\n\n\
             A quote in a source file is a string literal: it does not notice the \
             paragraph it was copied from being rewritten, deleted or reversed. Three \
             comments in this crate outlived the text they described before this test \
             existed, and every one of them was found by a reader rather than by a \
             test.\n\n\
             Either the document changed and the comment quoting it is now false, in \
             which case fix the comment -- and if the code is the side that is wrong, \
             fix that instead -- or the fragment below is too specific to survive a \
             legitimate reformat, in which case relax the fragment rather than drop \
             the citation. A finding here is the honest outcome. A gate that cannot \
             fail is worse than no gate."
        );
    }

    /// The real-distance reach of the precalculation loop, which is the half of
    /// §5.3 that a constant in `scoring.rs` cannot see.
    ///
    /// **A documented-line drift, found by writing this assertion — and since
    /// closed on both sides.** This test was written when `SCORING.md` §5.3
    /// stated the reach as `kernels.cutoff + 2.0 * a.radius` while the code said
    /// `kernels.cutoff + MAX_GRID_TYPE_RADIUS + a.radius`: the outer filter has
    /// to bound the **widest** of the ten per-probe-type blocks, and the probe's
    /// radius is not the receptor atom's, so the two forms are equal only when
    /// `MAX_GRID_TYPE_RADIUS == a.radius` — true for no element, it being the
    /// iodine radius, 2.350 Å. The documented form was therefore *narrower*
    /// than the code by `2.350 - R_atom`, and stale in the conservative
    /// direction: it could not compute the table wrong, only explain the code
    /// wrongly.
    ///
    /// §5.3 has since been corrected to the code's form and says so explicitly,
    /// and `REACH_DOC` above now quotes that sentence rather than the obsolete
    /// one. So the drift this test documents is closed — the two sides agree —
    /// and the test remains as the thing that would notice if they stopped.
    /// [`every_pinned_documentation_line_still_exists`] is what makes that
    /// claim true rather than hopeful: it fails the moment the quoted sentence
    /// stops appearing in the document.
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
            if narrowed == 0.0 {
                "exactly zero, and"
            } else {
                "normal, so"
            }
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
            band.push((
                d,
                v,
                if v.is_subnormal() {
                    "subnormal"
                } else {
                    "normal"
                },
            ));
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
            .next_back()
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
    ///
    /// The pin is the **derivation**, not the number. It used to pin `1 << 28`,
    /// which is how a memory number came to be documented as an indexing one:
    /// the constant and its justification were the same claim, so nothing
    /// noticed when the justification stopped being true. Pinning
    /// `u32::MAX / map_stride()` means the number cannot move without the
    /// kernel arithmetic moving with it, and the two arithmetic claims below are
    /// then checked against the kernel rather than restated.
    #[test]
    fn the_grid_capacity_ceiling_is_what_its_own_comment_claims() {
        const DOC: &str = "grid.rs: 'This is an *indexing* limit, not a memory limit' \
--- and SCORING.md has no line for MAX_GRID_POINTS.";
        assert_pinned(
            "MAX_GRID_POINTS",
            MAX_GRID_POINTS,
            (u32::MAX as u64) / (map_stride() as u64),
            DOC,
        );
        // The claim the comment makes is checkable, so check it. `sample()`'s
        // last read is `points * STRIDE - 1`, in a u32, and it fits.
        let last_read = MAX_GRID_POINTS * (map_stride() as u64) - 1;
        assert!(
            last_read <= u32::MAX as u64,
            "at the limit energy.wgsl reads index {last_read}, past the u32::MAX of {}; \
             the limit is above what the kernel's `idx * STRIDE` can address",
            u32::MAX
        );
        // And one more point does not fit, which is the other half of the
        // derivation: a limit that were merely safe would be needlessly tight.
        assert!(
            (MAX_GRID_POINTS + 1) * (map_stride() as u64) > u32::MAX as u64,
            "one more point still fits in a u32, so MAX_GRID_POINTS is below the \
             largest addressable point count"
        );
        let bytes = (MAX_GRID_POINTS as f64) * (map_stride() as f64) * 4.0;
        let gib = bytes / (1024.0 * 1024.0 * 1024.0);
        assert!(
            (gib - 16.0).abs() < 0.5,
            "the limit is {gib:.1} GiB of f32 at the current stride, not the 40 GiB the \
             old 2^28 comment claimed; the limit moved because the kernel's u32 index \
             moved, not because the memory budget did\n\
             documented: {DOC}"
        );
    }

    /// The reason the ceiling is `u32::MAX / STRIDE`, checked against the
    /// shader rather than cited by line.
    ///
    /// The test above proves the *constant* matches its own derivation. It
    /// cannot prove the derivation matches the kernel, because the derivation
    /// is a sentence in this file and a sentence does not fail. This one reads
    /// `energy.wgsl` and checks the property the sentence depends on: the
    /// flat index is built from `u32` dimensions and multiplied by a `u32`
    /// stride.
    ///
    /// It replaces three `energy.wgsl:123` pointers and a `// :123` comment,
    /// all of which went on naming a line that no longer held the code after
    /// the shader was edited. A hand-maintained line number is a second thing
    /// to forget, and this file had already forgotten it once.
    ///
    /// The failure message distinguishes the two ways this can go red, because
    /// they need different responses: a widened index type is a defect in the
    /// ceiling's derivation, and a renamed identifier means the check has to be
    /// re-pointed at the new name after someone re-reads the shader.
    #[test]
    fn the_shader_flat_index_is_u32_so_this_limit_is_the_right_one() {
        let src = ENERGY_WGSL_SOURCE;
        assert!(
            !src.is_empty(),
            "energy.wgsl came back empty; every claim this crate makes about the \
             kernel is being checked against nothing"
        );
        assert!(
            shader_flat_index_is_u32(),
            "energy.wgsl no longer declares nx/ny/nz and STRIDE as u32, or no longer \
             multiplies the point index by STRIDE, so MAX_GRID_POINTS = {MAX_GRID_POINTS} \
             (u32::MAX / {}) is no longer the largest grid the kernel can read. If the \
             index was widened on purpose, that is a behaviour change to a published \
             ceiling and this number has to be re-derived; if the shader's identifiers \
             were merely renamed, re-read the flat-index expression and re-point \
             `shader_flat_index_is_u32` at the new names. The file as read:\n{src}",
            map_stride()
        );
        eprintln!(
            "MEASURED: energy.wgsl still computes its flat index in u32 (STRIDE = {}, \
             nx/ny/nz declared u32, point index multiplied by STRIDE), so \
             MAX_GRID_POINTS = {MAX_GRID_POINTS} is the ceiling this crate advertises; \
             {} bytes of shader checked",
            map_stride(),
            src.len()
        );
        assert!(
            the_shader_does_not_measure_the_out_of_box_violation(),
            "energy.wgsl reads params.bmax again, so sample() is measuring the \
             out-of-box violation from the narrowed f32 coordinate again while the \
             CPU measures it from the f64. Both sides would take the same branch \
             and charge different amounts -- measured at x = 3.000000119209 as \
             1000 x 1.19e-7 against 0 -- and the band in \
             `band_for` no longer covers it. Either the host uploads the \
             violation (what it does now) or the band has to be opened back up, \
             and the second is the defect the first exists to remove.\n{src}"
        );
        eprintln!(
            "MEASURED: energy.wgsl no longer reads params.bmax, so the out-of-box \
             violation has exactly one evaluation, on the host, in f64"
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

/// The clamp in `energy.wgsl` is only safe because of something `fractional`
/// guarantees, and the guarantee used to be an argument in a comment.
///
/// `base_cell = clamp(floor(u), 0, n - 2)` is what keeps the eight corner
/// reads inside the grid, because `cell + 1` has to be at most `n - 1`. That
/// upper bound is written as `f32(params.nx - 2u)`, and **WGSL's `u32`
/// subtraction wraps**: on a one-point axis `1u - 2u` is `4294967295`, so the
/// clamp's upper bound becomes `4.29e9` and it stops clamping anything. There
/// is no in-bounds cell to pick on such an axis at all, which means the whole
/// safety of the clamp rests on the flag never being `1.0` when an axis holds
/// fewer than two points -- a fact about `fractional`, in another file, across
/// a `pub(crate)` boundary.
///
/// The argument was: "this line is reached only when the flag is 1.0, and
/// `fractional` cannot say inside on an axis with fewer than two points." That
/// is true, and it was never tested. These tests are the test.
#[cfg(test)]
mod degenerate_axes {
    use super::pinned_grid_constants::assert_pinned;
    use super::*;

    /// The document text for a map file, with the field names `GridMaps`
    /// actually has.
    ///
    /// Split out from [`maps_from_json`] so that a test can assert on the
    /// *refusal* as well as on the value, and so the fixture is written in
    /// exactly one place.
    #[allow(clippy::too_many_arguments)]
    fn json_for(
        dims: [usize; 3],
        min: [f64; 3],
        spacing: [f64; 3],
        max: [f64; 3],
        data_len: usize,
    ) -> String {
        let data: Vec<String> = (0..data_len).map(|i| (i % 7).to_string()).collect();
        format!(
            "{{\"min\":{min:?},\"dims\":{dims:?},\"spacing\":{spacing:?},\
             \"data\":[{}],\"box_\":{{\"min\":{min:?},\"max\":{max:?}}},\
             \"receptor_heavy\":[]}}",
            data.join(",")
        )
    }

    /// [`json_for`] at a fixed origin, which is all a test that is only about
    /// the payload length needs to vary.
    fn map_doc(dims: [usize; 3], data_len: usize) -> String {
        json_for(dims, [0.0; 3], [1.0; 3], [10.0; 3], data_len)
    }

    /// A `GridMaps` built the way a map file builds one: through `serde_json`
    /// into the derived `Deserialize`, using the field names the struct
    /// actually has.
    ///
    /// `data_len` is a parameter so that a *legal* grid far too large to
    /// tabulate can still be described, but it must now agree with `dims`:
    /// the deserialiser refuses a payload that does not, and that refusal is
    /// the behaviour under test rather than a limitation of the helper. A
    /// caller that needs a grid whose payload disagrees with its `dims` uses
    /// [`maps_with`], which is what an in-crate caller can still do.
    #[allow(clippy::too_many_arguments)]
    fn maps_from_json(
        dims: [usize; 3],
        min: [f64; 3],
        spacing: [f64; 3],
        max: [f64; 3],
        data_len: usize,
    ) -> GridMaps {
        let json = json_for(dims, min, spacing, max, data_len);
        serde_json::from_str(&json).unwrap_or_else(|e| {
            panic!("a GridMaps with dims {dims:?} and {data_len} floats did not deserialise: {e}")
        })
    }

    /// A `GridMaps` built directly, with a `data` length chosen by the caller.
    ///
    /// Exists because the deserialiser no longer accepts a payload that
    /// disagrees with `dims`, and several things still have to be reachable
    /// that such a payload makes reachable -- the gate's own refusal, and
    /// `fractional`'s behaviour on a grid the gate would turn away. `data` is
    /// private, so this is only available to a child module of `grid`, which
    /// is the same boundary that keeps an external caller from doing it.
    fn maps_with(dims: [usize; 3], data_len: usize) -> GridMaps {
        GridMaps {
            min: [0.0, 0.0, 0.0],
            dims,
            spacing: [1.0, 1.0, 1.0],
            data: vec![0.0f32; data_len],
            box_: GridBox {
                min: [0.0, 0.0, 0.0],
                max: [10.0, 10.0, 10.0],
            },
            receptor_heavy: Vec::new(),
            unknown_atom_types: 0,
        }
    }

    /// A NaN coordinate is rejected by **both** `fractional` paths, and the
    /// reason the bounds test is a negated `contains` rather than the plainer
    /// `i < 0.0 || i >= n` is exactly this: both of those comparisons are false
    /// for NaN, so the un-negated form would let a NaN through to the
    /// `i as usize` cast below the guard, where a NaN becomes 0 and the point
    /// is silently read off the wrong corner of the grid.
    ///
    /// This was written after that guard was rewritten to silence a clippy
    /// lint, which is the worst possible order: the rewrite touched the only
    /// thing standing between a NaN and a wrong-but-plausible index, and
    /// **nothing tested it**. A rewrite of untested behaviour is a rewrite
    /// whose correctness is a claim rather than a measurement.
    #[test]
    fn a_nan_coordinate_is_rejected_by_both_fractional_paths() {
        let dims = [4usize; 3];
        let maps = maps_with(dims, dims[0] * dims[1] * dims[2]);

        // `data` is empty on purpose: `fractional` reads only `min`, `spacing`
        // and `dims`, so a payload-length fixture here would be a second thing
        // to keep in step with `dims` for no gain. The tests that are about the
        // payload are in `tests/map_data_len.rs` and `tests/term_data_len.rs`.
        let terms = TermMaps {
            min: [0.0, 0.0, 0.0],
            dims,
            spacing: [1.0, 1.0, 1.0],
            data: Vec::new(),
            box_: GridBox {
                min: [0.0, 0.0, 0.0],
                max: [10.0, 10.0, 10.0],
            },
        };

        // A finite point well inside the grid is accepted by both, so a
        // rejection below is the NaN and not a fixture that refuses everything.
        let inside = [1.5, 1.5, 1.5];
        assert!(
            maps.fractional(inside).is_some(),
            "GridMaps::fractional refused a finite interior point {inside:?}, so a refusal \
             below would prove nothing"
        );
        assert!(
            terms.fractional(inside).is_some(),
            "TermMaps::fractional refused a finite interior point {inside:?}, so a refusal \
             below would prove nothing"
        );

        // One NaN axis at a time, so the message names which axis got through
        // rather than reporting a single opaque refusal.
        for axis in 0..3 {
            let mut p = inside;
            p[axis] = f64::NAN;
            assert!(
                maps.fractional(p).is_none(),
                "GridMaps::fractional accepted a NaN on axis {axis}: {p:?} -- the bounds \
                 guard let a NaN through to `i as usize`, which turns it into 0 and \
                 reads the wrong corner of the grid"
            );
            assert!(
                terms.fractional(p).is_none(),
                "TermMaps::fractional accepted a NaN on axis {axis}: {p:?} -- the bounds \
                 guard let a NaN through to `i as usize`, which turns it into 0 and \
                 reads the wrong corner of the grid"
            );
        }

        // All three at once, which is what a caller computing a position from
        // an uninitialised accumulator actually produces.
        let all_nan = [f64::NAN, f64::NAN, f64::NAN];
        assert!(maps.fractional(all_nan).is_none());
        assert!(terms.fractional(all_nan).is_none());

        // The two ends of the range, so a future "simplification" to a plain
        // half-open comparison cannot quietly move either edge. -0.0 is the
        // interesting one: it is accepted, and an `i <= 0` guard would not be.
        //
        // The high end is `dims - 1` and it is **exclusive**: the result is the
        // base cell of a trilinear interpolation, so the neighbour at `i + 1`
        // has to exist and the last stored point cannot be a base. That is why
        // [3, 3, 3] on a 4-point axis is refused while [2.5, 1, 1] is not --
        // writing the edge table the other way round was the mistake that
        // caught this test's first draft, which is why the case is written out
        // rather than left as a bare boundary.
        let edge_lo = [0.0, 0.0, 0.0];
        let last_base = [(dims[0] - 2) as f64 + 0.5, 1.0, 1.0];
        let first_refused = [dims[0] as f64 - 1.0, 1.0, 1.0];
        let well_past = [dims[0] as f64, 1.0, 1.0];
        for (label, p, want) in [
            ("at the low edge", edge_lo, true),
            ("at -0.0", [-0.0, 1.0, 1.0], true),
            ("at the last usable base", last_base, true),
            ("at the last stored point", first_refused, false),
            ("one axis past the grid", well_past, false),
        ] {
            assert_eq!(
                maps.fractional(p).is_some(),
                want,
                "GridMaps::fractional {label} {p:?} disagreed with the bound it is supposed \
                 to enforce"
            );
        }
    }

    /// The payload length `dims` implies, for a grid small enough to build.
    fn consistent_len(dims: [usize; 3]) -> usize {
        map_stride() * dims.iter().product::<usize>()
    }

    /// Points to ask `fractional` about on one axis of `n` points: both exact
    /// faces, the last representable `f64` below the upper face and the first
    /// above it, and everything that is not a number.
    fn axis_probes(lo: f64, n: usize, spacing: f64) -> Vec<f64> {
        let hi = lo + (n as f64 - 1.0) * spacing;
        vec![
            lo, // the lower face: inclusive
            lo - spacing,
            lo - 1.0,
            0.0,
            -0.0,
            lo + spacing * 0.5,
            lo + (n as f64 - 1.0) * spacing * 0.5,
            f64::from_bits(hi.to_bits().wrapping_sub(1)), // last f64 below the face
            hi,                                           // the upper face: exclusive
            f64::from_bits(hi.to_bits().wrapping_add(1)), // first f64 above the face
            hi + spacing,
            hi + 1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
        ]
    }

    /// `energy.wgsl`'s `base_cell` and `f`, transcribed so the contract the
    /// clamp has to keep can be checked rather than assumed.
    ///
    /// `wrapping_sub` on the `n - 2` is deliberate and is the point of the
    /// test: it is what the shader's `params.nx - 2u` does on a one-point axis,
    /// and reproducing it is what makes the failure mode observable from Rust.
    fn kernel_cell_and_fraction(maps: &GridMaps, p: [f64; 3]) -> ([usize; 3], [f64; 3]) {
        let d = maps.dims();
        let mut cell = [0usize; 3];
        let mut f = [0.0f64; 3];
        for k in 0..3 {
            // The shader's own expression, in its own precision.
            let u = ((p[k] as f32) - (maps.min[k] as f32)) / (maps.spacing[k] as f32);
            let last = (d[k] as u32).wrapping_sub(2) as f32;
            let base = u.floor().clamp(0.0, last);
            cell[k] = base as u32 as usize;
            f[k] = (u - base).clamp(0.0, 1.0) as f64;
        }
        (cell, f)
    }

    /// The property that makes the clamp safe, on grids that came out of the
    /// deserialiser.
    ///
    /// `fractional` says inside only when `0 <= floor(u) < dims[k] - 1` on
    /// every axis, and `cell` is that `floor(u)`. So `cell[k] + 1 <= dims[k]-1`
    /// always holds on the `Some` arm, and `dims[k] - 2 >= 0` is implied --
    /// which is exactly what makes `n - 2` a legal clamp bound. The first test
    /// below is that last step, which is the one that was argued.
    #[test]
    fn the_in_box_decision_is_never_taken_on_an_axis_with_fewer_than_two_points() {
        let degenerate: [[usize; 3]; 8] = [
            [1, 4, 4],
            [4, 1, 4],
            [4, 4, 1],
            [0, 4, 4],
            [4, 0, 4],
            [4, 4, 0],
            [1, 1, 1],
            [0, 0, 0],
        ];
        let configs: [([f64; 3], [f64; 3]); 2] = [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([-6.0, -6.0, -6.0], [0.375, 0.375, 0.375]),
        ];

        let mut asked = 0usize;
        for dims in degenerate {
            for (min, spacing) in configs {
                let max = [min[0] + 3.0, min[1] + 3.0, min[2] + 3.0];
                let maps = maps_from_json(dims, min, spacing, max, consistent_len(dims));
                assert_eq!(
                    maps.dims(),
                    dims,
                    "the deserialiser did not preserve dims, so this grid is not \
                     the one under test"
                );
                let probes: Vec<Vec<f64>> = dims
                    .iter()
                    .enumerate()
                    .map(|(k, &n)| axis_probes(min[k], n, spacing[k]))
                    .collect();
                for &x in &probes[0] {
                    for &y in &probes[1] {
                        for &z in &probes[2] {
                            let p = [x, y, z];
                            assert!(
                                maps.fractional(p).is_none(),
                                "fractional said inside at {p:?} on a grid whose dims \
                                 are {dims:?}. energy.wgsl would then run \
                                 `f32(params.n{{x,y,z}} - 2u)`, which for an axis of \
                                 {} point(s) wraps to {} instead of clamping, and the \
                                 clamp would stop holding the eight corner reads \
                                 inside the grid",
                                dims.iter().copied().min().unwrap_or(0),
                                (dims.iter().copied().min().unwrap_or(0) as u32).wrapping_sub(2)
                            );
                            asked += 1;
                        }
                    }
                }
            }
        }
        assert!(asked > 50_000, "only {asked} points were asked");

        // Not vacuous: the same sweep on a legal grid does say inside, so the
        // test above is refusing points rather than refusing everything.
        let dims = [4usize; 3];
        let min = [0.0f64; 3];
        let spacing = [1.0f64; 3];
        let maps = maps_from_json(dims, min, spacing, [3.0; 3], map_stride() * 64);
        let mut insides = 0usize;
        for (k, &n) in dims.iter().enumerate() {
            for &v in &axis_probes(min[k], n, spacing[k]) {
                if maps.fractional([v, 0.5, 0.5]).is_some() {
                    insides += 1;
                }
            }
        }
        assert!(
            insides > 0,
            "a legal {dims:?} grid was reported as refusing every probe, so the \
             degenerate cases above are not being distinguished from a working one"
        );

        // And the invariant the clamp depends on, stated directly on the legal
        // grid: the corner read `cell + 1` is inside the axis.
        let mut checked = 0usize;
        for ix in 0..8 {
            for iy in 0..8 {
                for iz in 0..8 {
                    let p = [
                        0.125 + ix as f64 * 0.4,
                        0.125 + iy as f64 * 0.4,
                        0.125 + iz as f64 * 0.4,
                    ];
                    let Some((frac, cell)) = maps.fractional(p) else {
                        continue;
                    };
                    for k in 0..3 {
                        assert!(
                            cell[k] < dims[k],
                            "fractional returned cell {cell:?} for {p:?} on dims \
                             {dims:?}; cell[{k}] + 1 = {} is outside the axis, so \
                             clamp(floor(u), 0, n - 2) is the only thing standing \
                             between the kernel and a read past the grid",
                            cell[k] + 1
                        );
                        assert!(
                            (0.0..1.0).contains(&frac[k]),
                            "fraction {frac:?} outside [0, 1) for {p:?}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 400,
            "only {checked} interior points carried the invariant"
        );
        eprintln!(
            "MEASURED: {asked} probe points over 8 degenerate dims x 2 origins, every \
             one refused; the same sweep on a legal [4,4,4] grid is answered \
             {insides} times; cell + 1 <= n - 1 held on {checked} interior points. \
             WGSL `u32` subtraction wraps, so a 1-point axis would have made the \
             clamp's bound 4294967295 rather than refusing"
        );
    }

    /// The clamp must not trade an out-of-bounds index for a wrong weight.
    ///
    /// A clamp that pins an index into range while silently zeroing (or
    /// saturating) the interpolation fraction is a different bug, not a fixed
    /// one, so this checks both halves of the contract at once: `cell + 1` is
    /// inside the axis, *and* `f` is the fraction the CPU computed. Swept at
    /// `f32` resolution, because that is the only resolution the kernel ever
    /// sees, across both faces and the interior, on the smallest legal grid
    /// (`n = 2`, where the clamp's bound is exactly `0`) and a larger one.
    #[test]
    fn the_clamp_keeps_the_corner_read_in_bounds_and_the_weight_the_cpu_computed() {
        // Two roundings' worth of the coordinate expression: the subtraction
        // loses about `eps32 * |min|` in p, which is `eps32 * |min| / spacing`
        // in u, and the division adds one more. For |min| = 9 A at 0.375 A
        // spacing that is `eps32 * 24` = 2.9e-6, and the worst measured over the
        // sweep below is 1.9e-6, so 1e-5 is a few times the worst the
        // derivation allows and still two orders below the 0.5 that would mean
        // "the clamp moved me to a face I am not on".
        const TOL: f64 = 1.0e-5;

        let cases: [([usize; 3], [f64; 3], [f64; 3]); 4] = [
            ([2, 2, 2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([2, 2, 2], [-1.0, -1.0, -1.0], [0.375, 0.375, 0.375]),
            ([4, 5, 3], [-6.0, -6.0, -6.0], [0.375, 0.375, 0.375]),
            ([24, 12, 12], [-9.0, -5.0, -4.0], [0.375, 0.375, 0.375]),
        ];
        let mut worst_f = 0.0f64;
        let mut worst_at = [0.0f64; 3];
        let mut worst_axis = 0usize;
        let mut at_lower = 0usize;
        let mut at_upper = 0usize;
        let mut inside_points = 0usize;

        for (dims, min, spacing) in cases {
            let max = [
                min[0] + (dims[0] - 1) as f64 * spacing[0],
                min[1] + (dims[1] - 1) as f64 * spacing[1],
                min[2] + (dims[2] - 1) as f64 * spacing[2],
            ];
            let maps = maps_from_json(
                dims,
                min,
                spacing,
                max,
                map_stride() * dims[0] * dims[1] * dims[2],
            );
            // At `f32` resolution, because that is the only resolution the
            // kernel ever sees -- but a bounded window, because the number of
            // `f32` values along a 8.6 A axis is 1.8e7 and walking all of them
            // is a measurement of nothing. Every face is covered by
            // `WINDOW` representable values on each side, which is a span of
            // ~1e-3 A and comfortably wider than the rounding step the two
            // sides can disagree over, and the interior is sampled on a grid.
            const WINDOW: u32 = 2048;
            const INTERIOR: usize = 64;
            for k in 0..3 {
                let mut bits: Vec<u32> = Vec::new();
                // A window of representable values on *both* sides of each
                // face. Bit patterns are not monotone in value for negative
                // floats, so this steps the pattern rather than the value and
                // keeps whichever side it lands on.
                for face in [min[k], max[k]] {
                    let edge = (face as f32).to_bits();
                    for d in 0..=WINDOW {
                        for b in [edge.wrapping_sub(d), edge.wrapping_add(d)] {
                            if f32::from_bits(b).is_finite() {
                                bits.push(b);
                            }
                        }
                    }
                }
                // And the interior, on a coarse grid.
                for i in 0..=INTERIOR {
                    let v = min[k] + (max[k] - min[k]) * i as f64 / INTERIOR as f64;
                    bits.push((v as f32).to_bits());
                }
                assert!(
                    bits.len() < 20_000,
                    "the sample is meant to be a window at each face plus a coarse \
                     interior, not a walk of the whole axis ({} points here)",
                    bits.len()
                );
                for b in bits {
                    let v = f32::from_bits(b) as f64;
                    let mut p = [
                        min[0] + spacing[0] * 0.5,
                        min[1] + spacing[1] * 0.5,
                        min[2] + spacing[2] * 0.5,
                    ];
                    p[k] = v;
                    let Some((frac, _)) = maps.fractional(p) else {
                        continue;
                    };
                    inside_points += 1;
                    let (cell, f) = kernel_cell_and_fraction(&maps, p);
                    for j in 0..3 {
                        assert!(
                            cell[j] < dims[j],
                            "on dims {dims:?} at {p:?} the kernel would read axis {j} \
                             at {} and its neighbour at {}, but that axis holds {} \
                             points. clamp(floor(u), 0, n - 2) did not hold",
                            cell[j],
                            cell[j] + 1,
                            dims[j]
                        );
                        assert!(
                            (0.0..=1.0).contains(&f[j]),
                            "clamped fraction {f:?} outside [0, 1] at {p:?} on {dims:?}"
                        );
                    }
                    let d = (f[k] - frac[k]).abs();
                    if d > worst_f {
                        worst_f = d;
                        worst_at = p;
                        worst_axis = k;
                    }
                    if f[k] == 0.0 && frac[k] < 0.5 {
                        at_lower += 1;
                    }
                    if f[k] == 1.0 && frac[k] > 0.5 {
                        at_upper += 1;
                    }
                }
            }
        }
        assert!(
            worst_f <= TOL,
            "the clamp moved the interpolation fraction by {worst_f:e} on axis \
             {worst_axis} at {worst_at:?}, against a tolerance of {TOL:e} and a \
             fraction the CPU computed as something else. A clamp that makes the \
             index safe by making the weight wrong is still a defect: the energy \
             would then be a plausible number interpolated at the wrong place"
        );
        assert!(
            at_lower + at_upper < inside_points / 4,
            "the clamp fired on {at_lower} interior fractions (to 0) and {at_upper} \
             (to 1) out of {inside_points} points the CPU accepted, so it is not \
             acting only at a face"
        );
        eprintln!(
            "MEASURED clamp: {inside_points} f32-resolution placements the CPU accepted; \
             cell + 1 stayed inside every axis throughout; worst |f_kernel - f_cpu| = \
             {worst_f:e} on axis {worst_axis} at {worst_at:?} against a tolerance of \
             {TOL:e}; the clamp reached a limit on {at_lower} + {at_upper} of them"
        );
    }

    /// `MAX_GRID_POINTS` at the boundary -- and what a payload that disagrees
    /// with `dims` now does.
    ///
    /// This test used to finish by measuring the hole: it deserialised a grid
    /// sitting exactly on the published ceiling beside 160 floats, showed that
    /// both `check_gpu_dims` and `check_gpu_index_range` accepted it, and
    /// printed the 16 GiB shortfall as a `MEASURED` line. That document is
    /// refused now, so the arithmetic is reached through [`maps_with`] -- a
    /// direct construction, which is what an in-crate caller can still do --
    /// and the refusal of the document is asserted in its place.
    ///
    /// The boundary claim itself is unchanged and is still the subject: the
    /// published ceiling is where the kernel's `u32` index says it is.
    #[test]
    fn the_published_ceiling_is_where_the_u32_index_says_it_is_on_a_deserialised_grid() {
        assert_pinned(
            "MAX_GRID_POINTS",
            MAX_GRID_POINTS,
            107_374_182,
            "grid.rs MAX_GRID_POINTS = u32::MAX / map_stride(); docs/VERIFICATION.md \
             and docs/LIMITATIONS.md both publish 107,374,182 points = 16.0 GiB, so \
             this number is a published one and cannot be moved without re-issuing \
             both documents",
        );

        // Right at the ceiling, on one axis, with the other two at one point.
        let at_limit = [MAX_GRID_POINTS as usize, 1, 1];
        check_gpu_dims(at_limit).expect("a grid at the ceiling is addressable");
        let over = [MAX_GRID_POINTS as usize + 1, 1, 1];
        assert!(
            check_gpu_dims(over).is_err(),
            "one point above the ceiling was accepted, so the limit is not tight"
        );

        // The document that used to be the finding. It is refused now, by the
        // deserialiser, and the refusal has to name both numbers -- a rule that
        // said only "bad length" would leave the caller to recompute the
        // expectation this crate just declined to compute for it.
        let doc = map_doc(at_limit, map_stride() * 4);
        let err = serde_json::from_str::<GridMaps>(&doc)
            .expect_err("a document whose payload disagrees with its dims must not deserialise");
        let text = err.to_string();
        assert!(
            text.contains(&(MAX_GRID_POINTS * map_stride() as u64).to_string()),
            "the refusal must name the length dims {:?} imply; got: {text}",
            at_limit
        );
        assert!(
            text.contains(&(map_stride() * 4).to_string()),
            "the refusal must name the length that was found; got: {text}"
        );

        // And the refusal reaches the *right* shape of grid: a large point
        // count, not a degenerate one, so what is caught is the payload rather
        // than the axis rule firing first.
        assert!(
            serde_json::from_str::<GridMaps>(&map_doc([2, 3, 17_895_697], 160)).is_err(),
            "a well-shaped grid at the ceiling with 160 floats must not deserialise"
        );

        // In-crate construction can still build the inconsistent grid, because
        // `data` is a private field and nothing stops a future path from
        // getting it wrong. So the *gate* refuses it too, independently of
        // where the grid came from -- which is why the check is in
        // `check_gpu_index_range` and not only in the deserialiser.
        let maps = maps_with(at_limit, map_stride() * 4);
        assert_eq!(maps.dims(), at_limit);
        assert_ne!(
            maps.data_len(),
            map_stride() * maps.point_count(),
            "maps_with is supposed to build a grid whose payload disagrees with its \
             dims, and it no longer does"
        );
        check_gpu_dims(maps.dims()).expect("still arithmetic-legal");
        // Two of its three axes hold one point, so the *shape* rule is the one
        // that fires here -- and it is the one that fired before this change
        // too, which is why this grid was never the kernel-boundary case.
        let refusal = maps
            .check_gpu_index_range()
            .expect_err("a grid with two one-point axes reached check_gpu_index_range");
        assert!(
            refusal.to_string().contains("one-point")
                || refusal.to_string().contains("fewer than two"),
            "the grid was refused for the wrong reason: {refusal}"
        );

        // The grid that *was* the kernel-boundary case: exactly on the ceiling,
        // every axis holding at least two points, 160 floats. It passed both
        // rules and reached the upload. It is refused now, by the payload
        // rule -- and it has to be refused, because the only way to satisfy
        // both the ceiling and the length rule at this size is to hold
        // 4,294,967,280 floats, which is 16.0 GiB. The refusal is the correct
        // answer, not a proxy for one.
        let big = maps_with([2, 3, 17_895_697], map_stride() * 4);
        assert_eq!(big.point_count() as u64, MAX_GRID_POINTS);
        check_gpu_dims(big.dims()).expect("a well-shaped grid at the ceiling is addressable");
        let refusal = big
            .check_gpu_index_range()
            .expect_err("a ceiling grid carrying 160 floats reached the gate");
        let text = refusal.to_string();
        assert!(
            text.contains("4294967280"),
            "the refusal must name the length dims {:?} imply; got: {text}",
            big.dims()
        );
        assert!(
            text.contains("160"),
            "the refusal must name the length that was found; got: {text}"
        );

        // The three rules are separate, and this is the line that says so on a
        // grid small enough to build. Each is refused for its own reason, with
        // its own numbers, and the smallest legal grid is accepted -- so no one
        // of the three is standing in for another.
        let stride = map_stride();
        let legal = maps_with([2, 2, 2], stride * 8);
        legal
            .check_gpu_index_range()
            .expect("the smallest legal grid with the right payload is accepted");
        // Rule 1, the ceiling.
        let too_big = maps_with([MAX_GRID_POINTS as usize, 2, 2], stride * 8);
        let e = too_big
            .check_gpu_index_range()
            .expect_err("past the ceiling must be refused");
        assert!(
            e.to_string().contains(&MAX_GRID_POINTS.to_string()),
            "the ceiling rule must name the ceiling; got: {e}"
        );
        // Rule 2, the shape. Payload is correct, so only the shape can refuse.
        let one_point_axis = maps_with([2, 1, 2], stride * 4);
        let e = one_point_axis
            .check_gpu_index_range()
            .expect_err("a one-point axis must be refused");
        assert!(
            e.to_string().contains("one-point") || e.to_string().contains("fewer than two"),
            "the shape rule must name the axis; got: {e}"
        );
        // Rule 3, the payload. Shape is legal and the size is legal.
        let wrong_len = maps_with([2, 2, 2], stride * 8 + 1);
        let e = wrong_len
            .check_gpu_index_range()
            .expect_err("a payload one value long must be refused");
        assert!(
            e.to_string().contains("320"),
            "the payload rule must name the expected length 320; got: {e}"
        );
        assert!(
            e.to_string().contains("321"),
            "the payload rule must name the found length 321; got: {e}"
        );
        assert!(
            !e.to_string().contains("one-point") && !e.to_string().contains("fewer than two"),
            "the payload rule is firing for the shape rule's reason: {e}"
        );

        // The flag is still decidable on a huge *well-shaped* axis, and the
        // cell is still in range. `fractional` reads `dims` and `spacing`, not
        // `data`, so the payload refusal does not reach it -- and the clamp's
        // invariant has to hold for a grid that the gate would refuse, because
        // the gate is not the only reader of `dims`.
        let p = [0.5, 1.0, 100.0];
        let (frac, cell) = big.fractional(p).expect("well inside a huge axis");
        assert_eq!(frac, [0.5, 0.0, 0.0]);
        assert_eq!(cell, [0, 1, 100]);
        assert!(
            cell[2] < big.dims()[2],
            "a huge axis broke the cell + 1 <= n - 1 invariant the clamp needs"
        );
        assert!(
            big.fractional([0.0, 1.0, 100.0]).is_some()
                && big.fractional([0.0, 2.0, 100.0]).is_none(),
            "the lower face of a two-point axis is inclusive and the upper is \
             exclusive, which is what makes `n - 2` the right clamp bound"
        );
        eprintln!(
            "MEASURED ceiling: MAX_GRID_POINTS = {MAX_GRID_POINTS}; dims {at_limit:?} \
             accepted and {over:?} refused by check_gpu_dims; a document with dims \
             {at_limit:?} and {} floats is now refused at deserialisation, naming \
             {} expected; a well-shaped ceiling grid built in-crate is refused by \
             the gate for the payload; the smallest legal grid is accepted and each \
             of the three rules refuses for its own reason",
            map_stride() * 4,
            MAX_GRID_POINTS * map_stride() as u64
        );
    }

    /// The wire form has to keep carrying every field the struct has.
    ///
    /// `GridMaps` deserialises through `GridMapsWire` so a wrong payload length
    /// is caught at load, and the price of that is a second declaration of the
    /// same seven fields. A field added to one and not the other would be
    /// silently dropped on the way in, which is the exact class of bug the
    /// duplicated declaration is supposed to make impossible -- so it is
    /// checked instead of trusted.
    ///
    /// The round trip is the check: serialising a real `GridMaps` emits every
    /// field, and if the wire form does not read one back the values differ.
    #[test]
    fn the_wire_form_still_has_every_field_the_struct_has() {
        let maps = maps_with([2, 3, 4], map_stride() * 24);
        let value = serde_json::to_value(&maps).expect("a GridMaps serialises");
        let obj = value.as_object().expect("a GridMaps is a map");
        let mut names: Vec<&str> = obj.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "box_",
                "data",
                "dims",
                "min",
                "receptor_heavy",
                "spacing",
                "unknown_atom_types",
            ],
            "the serialised field set moved. If a field was added to GridMaps and \
             not to GridMapsWire, it is now silently discarded on the way in, and \
             this list is where that has to be noticed"
        );
        let back: GridMaps =
            serde_json::from_value(value).expect("a GridMaps round-trips through its wire form");
        assert_eq!(back, maps, "a round trip changed the value");
    }

    /// The field *names* are the on-disk contract, so they are pinned by name.
    ///
    /// Renaming `box_` or reordering the declarations would make a map file
    /// written by any other version of this crate mean something else, or stop
    /// meaning anything, and a self-describing format would not complain --
    /// `serde_json` would just fill the default and the grid would be wrong.
    #[test]
    fn the_wire_form_keeps_the_documented_field_names() {
        // A document written by an older build: no `unknown_atom_types` at all,
        // because the field did not exist. It has to keep loading, and load as
        // 0, because "this map predates the counter" is a weaker claim than
        // "every atom was recognised" and is not the same claim as an error.
        let doc = map_doc([2, 2, 2], map_stride() * 8);
        let maps: GridMaps = serde_json::from_str(&doc).expect("a pre-counter map file loads");
        assert_eq!(maps.unknown_atom_types, 0);
        assert_eq!(maps.dims(), [2, 2, 2]);

        // `box_` is the awkward one: a bare `box` is not a reserved word in
        // Rust, so a rename here is a plausible-looking edit that would
        // silently stop reading the field.
        let renamed = doc.replace("\"box_\"", "\"box\"");
        assert!(
            serde_json::from_str::<GridMaps>(&renamed).is_err(),
            "a document spelling the field `box` deserialised, so `box_` is not \
             load-bearing in the wire form and the pinned name is not pinned"
        );

        // Unknown fields are still ignored, exactly as the derive ignored them.
        // A file carrying a field from a newer version must not become
        // unreadable to an older build.
        let extended = doc.replace("\"min\":", "\"future_field\":7,\"min\":");
        let maps: GridMaps = serde_json::from_str(&extended).expect("unknown fields are ignored");
        assert_eq!(maps.dims(), [2, 2, 2]);
    }
}
