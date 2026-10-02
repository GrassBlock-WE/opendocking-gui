// Open Docking — batched grid interpolation and energy summation.
//
// One workgroup scores one conformation: `workgroup_id.x` picks the
// conformation and the local invocation index picks the atom, so a whole
// conformation is reduced in shared memory and written once. That avoids
// atomics entirely — WGSL has no `atomicAdd` on `f32` — and keeps the write
// pattern to exactly one store per workgroup.
//
// Layouts must match dock_core:
//   conformations: (n_conf * n_atoms) Coord, atom-major within a
//                  conformation -- 32 bytes each, two `vec4<f32>`.
//                  `pos_flag.xyz` is the `f32` position and `pos_flag.w` is
//                  the host's in-box decision, exactly 1.0 or 0.0.
//                  `violation.x` is the host's out-of-box violation in
//                  angstroms, exactly 0.0 when the point is in the box.
//                  Both decisions travel with the point rather than in a
//                  buffer of their own because each is per *conformation*
//                  (the same atom is in the box in one pose and outside it in
//                  the next) and wgpu's downlevel limit is four storage
//                  buffers, already spent.
//   atom_data:     (n_atoms * 8) u32 — [type, w0, w1, w2, w3, pad, pad, pad],
//                  with the weights stored as bit-cast f32
//   grid:          point-major, STRIDE = 40 f32 per point
//
// The per-atom weights are packed alongside the element index rather than
// uploaded as their own storage buffer: wgpu's downlevel default allows only
// four storage buffers per shader stage, and this kernel needs five if the
// types and the weights are separate. One buffer also halves the number of
// binds. The same four-buffer limit is why the two host decisions widen the
// per-point element from 16 to 32 bytes rather than becoming two more
// bindings: `array<vec4<f32>>` cannot hold a position, a flag and a distance.

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    n_atoms: u32,
    n_conf: u32,
    // WGSL uniform layout gives a vec3 a full 16-byte slot, so pad both.
    min: vec4<f32>,
    spacing: vec4<f32>,
    // The search box's maximum corner. **No longer read by `sample`.**
    //
    // It used to be, to measure the out-of-box violation here. It does not
    // any more: the host measures that in `f64` and uploads it (see the note
    // on the out-of-box charge below), and a violation recomputed here from
    // the narrowed coordinate is a different number from the one the CPU
    // charged. The field is kept because it is part of the 80-byte uniform
    // layout that `grid_params_matches_the_wgsl_uniform_layout` pins and that
    // `docs/SCORING.md` §9.2 documents; removing it would be a 16-byte change
    // to a published layout for no gain. It is kept *and* labelled, because a
    // field that looks like it drives the penalty and does not is the same
    // failure as a comment that stopped being true.
    bmax: vec4<f32>,
}

/// One point, and the two decisions the host took about it.
struct Coord {
    /// `.xyz` is the position narrowed to `f32`; `.w` is the in-box flag.
    pos_flag: vec4<f32>,
    /// `.x` is the out-of-box violation in angstroms, measured by the host in
    /// `f64`. The rest is padding to the 32-byte stride.
    violation: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> energies: array<f32>;

@group(0) @binding(1)
var<storage, read> coords: array<Coord>;

@group(0) @binding(2)
var<storage, read> atom_data: array<u32>;

@group(0) @binding(3)
var<storage, read> grid: array<f32>;

@group(0) @binding(4)
var<uniform> params: GridParams;

const MAPS_PER_TYPE: u32 = 4u;
const STRIDE: u32 = 40u; // 10 element types × 4 slots
const ATOM_STRIDE: u32 = 8u; // 1 type + 4 weights + 3 pad

// Kcal/mol charged per angstrom, per axis, for leaving the box. This is the
// same number and the same function as `search::OUT_OF_BOX_PENALTY` applied to
// the host's sum of `grid::out_of_box_violation_per_axis`, and
// `shader_and_rust_agree_on_the_out_of_box_penalty` binds the two together:
// they are two transcriptions of one constant, and nothing in either language
// would notice if they drifted.
const OUT_OF_BOX_PENALTY: f32 = 1000.0;

var<workgroup> partial: array<f32, 64>;

/// Trilinear value of one atom at one position, or the out-of-box charge.
///
/// # The branch is not derived here
///
/// `in_box` arrives from the host: computed in `f64`, from the *un-narrowed*
/// coordinates, by `GridMaps::fractional` — the function
/// `ScoringContext::evaluate_full` itself calls, and the first statement of
/// `interpolate_with_gradient`. It used to be derived here, from
/// `(p - min) / spacing` in `f32`, against `f32(n - 1)`.
///
/// That derivation was a **decision taken from a narrowed input**, and the
/// function it decides is *discontinuous* at the face. So the CPU's `f64`
/// evaluation of the same expression and this `f32` one could take opposite
/// branches on a point they should agree about — and opposite branches are not
/// two approximations of one number. One interpolates the grid; the other
/// charges the ramp below. Those differ by the map value at the face, which is
/// 15.540 kcal/mol measured where placements that agree on the branch differ by
/// at most ~1e-3. The old `Batch` docs called the resulting error "~15.5
/// against ~4.8e-4"; that ratio is the reason the decision moved, and the
/// reason no tolerance wide enough to admit the one can be narrow enough to
/// police the other.
///
/// The previous repair was to keep the derivation and *reconcile* it: the host
/// walked the uploaded `f32` in whole ULPs until a Rust **transcription** of
/// this file's arithmetic produced the same bit pattern. That transcription was
/// found to disagree with this kernel by two ULPs of `u` — `u = 17.999998093`
/// against `n - 1 = 18.0` on a 6.4 Å box — consistent with the driver
/// contracting the division into a reciprocal multiply. A fix that depends on
/// reproducing the device's own rounding is a smaller version of the bug it
/// was written to remove, so it is gone: the decision is made once, on one
/// side, in one precision, and travels here as a bit.
///
/// # Why `u` is still computed, and why that is not the same problem
///
/// `u` is still evaluated, and still used, for the eight interpolation weights.
/// That is **arithmetic, not a decision**: the cell and the fractions feed a
/// weighted sum, and if the two sides round `u` differently the blend moves by
/// `O(eps32)` of the map value — the ordinary rounding band, not a change of
/// which function is being evaluated. The branch is treated differently from
/// the weights *precisely because* a disagreement about the branch is
/// `O(|g|)` while a disagreement about the weights is `O(eps)`. So the one
/// quantity where the roundings must not be allowed to differ is the one
/// quantity that is no longer computed twice.
///
/// That is also why `u` is clamped before it indexes anything. The host said
/// "inside" from `p` in `f64`; this `u` comes from `f32(p)`, and the two can
/// straddle `n - 1` by a rounding step. Left alone, `floor(u)` would be `n - 1`
/// and the upper corner `cell + 1` would read one point past the end of the
/// grid — the neighbouring row in x, and off the end of the buffer at the far
/// corner. The clamp pins the cell to the last cell that still has an upper
/// neighbour and the fraction to `[0, 1]`, so a point a rounding step beyond
/// the last node evaluates *to* that node, which is the same limit the
/// trilinear interpolant has there. For any point strictly inside, `floor(u)`
/// is already in range and the fraction already in `(0, 1)`, so the clamp does
/// not fire: it acts only in the ~1e-6 Å neighbourhood of a face, which is the
/// only place the two sides can differ at all. `n - 2` cannot underflow here
/// either: this line is reached only when the host said inside, and
/// `fractional` cannot say inside on an axis holding fewer than two points.
///
/// # The out-of-box charge: the *magnitude* is the host's too
///
/// The branch and the size of the charge are one decision, and the host takes
/// both. It uploads `out_of_box_violation_per_axis(p).sum()`, in `f64`, next to
/// the flag, and this kernel multiplies it by the same
/// `OUT_OF_BOX_PENALTY` the CPU multiplies its own sum by.
///
/// It used to recompute the violation here, as
/// `max(p - bmax, min - p).max(0)`, from the **narrowed** `f32` position. That
/// expression is right and it is still the right expression -- it is
/// `out_of_box_violation_per_axis` -- and it was still a defect, because the
/// CPU measures the same thing from the `f64` it scores. At
/// `x = 3.000000119209 A` the CPU charged `1000 x 1.19e-7` and the kernel
/// charged `0`: `f32(3.000000119209)` rounds to exactly `3.0`, so the narrowed
/// coordinate sat *on* the face and measured no violation at all. Both sides
/// had taken the out-of-box branch -- they agreed on the function -- and
/// disagreed on its argument by up to `eps32 x |p|`, amplified by the penalty
/// to `1000 x 1.19e-7 x 3.0 = 3.6e-4` kcal/mol.
///
/// That is small, and small is the problem: a band wide enough to hold it
/// cannot police the ~2e-4 the two backends legitimately differ by *inside*
/// the box, so the honest tolerance for the whole kernel had to be opened up
/// to accommodate a term that only exists out there. The residual is
/// pre-existing -- it is byte-identical before and after the in-box flag, so
/// it was never a regression from it -- and it is a live ranking input, since
/// a batch scorer's job is to order poses and this term orders them wrongly
/// whenever two poses differ by less than it.
///
/// Uploading the magnitude closes it without reintroducing anything. The
/// branch is still the host's flag, taken once from the un-narrowed `f64` by
/// the crate's own `fractional`; what moved is only the *size* of the term,
/// and the host is the side that knows it, in the precision the CPU charged
/// it in. One evaluation, two consumers -- the same arrangement as the flag,
/// extended one term further.
///
/// The layout can carry it because `array<vec4<f32>>` had one free slot of
/// three: a position, a flag and a distance do not fit in 16 bytes, so the
/// per-point element became a pair of `vec4`s and the binding's
/// `min_binding_size` went to 32. A fifth storage buffer is not available --
/// wgpu's downlevel default allows four and this kernel already has four --
/// which is why the element widened instead of the binding list growing.
///
/// What is left on this path is `f32` rounding of a number that is right:
/// `f32(violation) x 1000.0` against `violation x 1000.0`, a *relative*
/// `eps32` on the penalty, the same class of disagreement the in-box path
/// spends on the grid values. It is now bounded by the penalty each side is
/// summing rather than by the coordinate, so it shrinks in relative terms as
/// the penalty grows -- the opposite of the defect.
///
/// The flat per-atom `1000.0` this replaced is a different, older story and
/// is recorded in `gpu::Batch`: it was a sentinel standing in for a function
/// rather than a coarse version of one, and it inverted the ranking of poses
/// outside the box outright.
fn sample(p: vec3<f32>, in_box: f32, violation: f32, atom: u32) -> f32 {
    // The same expression the CPU evaluates in `f64` in
    // `GridMaps::fractional`, in the precision this kernel has. It no longer
    // decides anything: see the note on the branch below.
    let u = (p - params.min.xyz) / params.spacing.xyz;

    // The host writes exactly 1.0 or 0.0, so the comparison is exact either
    // way; it is written in the negated positive form the rest of the crate
    // uses for every range test (`!(max > min)`, `!(i >= 0.0)` in `grid.rs`)
    // so that a `NaN` flag lands on the refusal branch, which is the one that
    // cannot read out of bounds.
    if !(in_box > 0.5) {
        // `violation` is the host's own sum of `out_of_box_violation_per_axis`,
        // in the same per-axis form the CPU uses -- a distance to the box
        // centre would charge an atom for axes it is lawfully inside, and an
        // atom outside through a corner is charged on both faces. The charge
        // is therefore the gradient of the energy it adds, on both backends.
        return OUT_OF_BOX_PENALTY * violation;
    }

    // See "Why `u` is still computed" above: clamped, not trusted, because the
    // host's `f64` decision and this `f32` evaluation of `u` can straddle a
    // face by a rounding step, and the eight corner reads must stay in bounds.
    let last_cell = vec3<f32>(
        f32(params.nx - 2u),
        f32(params.ny - 2u),
        f32(params.nz - 2u),
    );
    let base_cell = clamp(floor(u), vec3<f32>(0.0), last_cell);
    let cell = vec3<u32>(base_cell);
    let f = clamp(u - base_cell, vec3<f32>(0.0), vec3<f32>(1.0));

    let a0 = atom * ATOM_STRIDE;
    let type_base = atom_data[a0] * MAPS_PER_TYPE;
    let w0 = bitcast<f32>(atom_data[a0 + 1u]);
    let w1 = bitcast<f32>(atom_data[a0 + 2u]);
    let w2 = bitcast<f32>(atom_data[a0 + 3u]);
    let w3 = bitcast<f32>(atom_data[a0 + 4u]);

    var value = 0.0;
    for (var c = 0u; c < 8u; c = c + 1u) {
        let cx = f32(c & 1u);
        let cy = f32((c >> 1u) & 1u);
        let cz = f32((c >> 2u) & 1u);
        let w = select(1.0 - f.x, f.x, cx > 0.5)
            * select(1.0 - f.y, f.y, cy > 0.5)
            * select(1.0 - f.z, f.z, cz > 0.5);

        let idx = (cell.x + u32(cx))
            + params.nx * ((cell.y + u32(cy)) + params.ny * (cell.z + u32(cz)));
        let g = idx * STRIDE + type_base;
        let v = grid[g] * w0 + grid[g + 1u] * w1 + grid[g + 2u] * w2 + grid[g + 3u] * w3;
        value = value + w * v;
    }
    return value;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let conf = wid.x;
    if conf >= params.n_conf {
        return;
    }
    let atom = lid;
    if atom >= params.n_atoms {
        partial[lid] = 0.0;
    } else {
        let c = coords[conf * params.n_atoms + atom];
        partial[lid] = sample(c.pos_flag.xyz, c.pos_flag.w, c.violation.x, atom);
    }
    workgroupBarrier();

    // Binary tree reduction over the workgroup.
    var stride = 32u;
    loop {
        if stride == 0u {
            break;
        }
        if lid < stride {
            partial[lid] = partial[lid] + partial[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if lid == 0u {
        energies[conf] = partial[0];
    }
}
