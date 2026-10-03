//! The `u32` flat index in `energy.wgsl`, and the limit it implies.
//!
//! The expression below is quoted without line numbers on purpose. This file
//! used to carry `energy.wgsl:123` next to it, the shader moved, and the
//! pointer went on asserting a position the code no longer occupied -- which is
//! the same failure as a stale doc anchor, just with a test file next to it.
//! `grid.rs::the_shader_flat_index_is_u32_so_this_limit_is_the_right_one` now
//! checks the property against the shader text instead, so a rename or a
//! re-type turns a test red rather than a comment false.
//!
//! `MAX_GRID_POINTS` used to be `1 << 28`, which is a *memory* number: 40 GiB of
//! `f32` at the current stride. The kernel cannot address that many points, so
//! the advertised limit was 2.5x above what the GPU would actually read, and a
//! grid in between passed every check in the crate and then wrapped its index
//! arithmetic on the device -- silently, because a `u32` wrap in WGSL is not an
//! error and the host's `nx`/`ny`/`nz` are narrowed in the same expression.
//!
//! The bound is `(u32::MAX) / (GRID_TYPE_COUNT * MAPS_PER_TYPE)`, because the
//! largest index `sample()` reads is `points * STRIDE - 1`:
//!
//! ```text
//! let idx = (cell.x + cx) + nx * ((cell.y + cy) + ny * (cell.z + cz));
//! let g = idx * STRIDE + type_base;
//! let v = grid[g] * w0 + grid[g+1] * w1 + grid[g+2] * w2 + grid[g+3] * w3;
//! ```
//!
//! `cell` is at most `n - 2` because `sample()` clamps it there outright:
//! `base_cell = clamp(floor(u), 0, n - 2)`, so `cell + corner` is at most
//! `n - 1` by construction rather than by the caller having got a guard right.
//! The clamp is load-bearing now that the in-box branch has moved to the host
//! -- the flag says "inside" from the `f64` coordinate, while `u` is recomputed
//! in `f32`, and the two can straddle a face by a rounding step -- so an
//! unclamped `floor(u)` could be `n - 1` and the upper corner would read one
//! point past the end of the grid. And `type_base + 3` is the last of the
//! `STRIDE` values belonging to that point.

use dock_core::grid::{
    check_gpu_dims, estimate_dims, GridBox, GridMaps, MAPS_PER_TYPE, MAX_GRID_POINTS,
};
use dock_core::scoring::VinaScoring;
use dock_core::types::GRID_TYPE_COUNT;
use dock_core::types::{Atom, AtomType, Element, Molecule};

/// The bound, derived here from the same two published facts `grid.rs` uses.
///
/// Written out rather than imported so that a change to `MAX_GRID_POINTS` has
/// to be made in two places to keep this test green, which is the point: the
/// constant is supposed to be *derived*, and a test that reads the derivation
/// back out of the constant it is checking cannot fail.
fn derived_bound() -> u64 {
    u32::MAX as u64 / (GRID_TYPE_COUNT * MAPS_PER_TYPE) as u64
}

fn old_limit() -> u64 {
    1 << 28
}

fn test_receptor() -> Molecule {
    let atoms: Vec<Atom> = vec![
        Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(2, [2.0, 0.0, 0.0], Element::O, AtomType::OA),
    ];
    Molecule::from_atoms(atoms).expect("receptor")
}

fn box_of(size: f64) -> GridBox {
    GridBox::new([0.0, 0.0, 0.0], [size, size, size]).expect("box")
}

fn points_of(size: f64, spacing: f64) -> u64 {
    let d = estimate_dims(&box_of(size), spacing);
    d[0] as u64 * d[1] as u64 * d[2] as u64
}

/// The constant is the kernel's index arithmetic, not a memory budget.
#[test]
fn max_grid_points_is_the_kernel_bound() {
    assert_eq!(
        MAX_GRID_POINTS,
        derived_bound(),
        "MAX_GRID_POINTS is {} but u32::MAX / (GRID_TYPE_COUNT * MAPS_PER_TYPE) is {}; \
         the constant has to be the size of the kernel's u32 index, or the advertised \
         limit is a promise the GPU cannot keep",
        MAX_GRID_POINTS,
        derived_bound()
    );
    assert_eq!(
        MAX_GRID_POINTS, 107_374_182,
        "the derivation changed: (u32::MAX) / ({} * {}) is no longer 107374182",
        GRID_TYPE_COUNT, MAPS_PER_TYPE
    );
}

/// The limit is the *largest* addressable point count, not merely a safe one.
///
/// A constant derived as a rounded-down memory number would satisfy the first
/// test and fail this one, and the failure mode is the quiet kind: the engine
/// would refuse grids the device can read perfectly well.
#[test]
fn the_limit_is_exactly_the_largest_addressable_point_count() {
    let stride = (GRID_TYPE_COUNT * MAPS_PER_TYPE) as u64;
    assert!(
        MAX_GRID_POINTS * stride <= u32::MAX as u64,
        "a grid at the limit needs {} flat values, past the u32::MAX of {}",
        MAX_GRID_POINTS * stride,
        u32::MAX
    );
    assert!(
        (MAX_GRID_POINTS + 1) * stride > u32::MAX as u64,
        "one more point would still fit in a u32, so the limit is needlessly tight"
    );
    // And the last index the kernel reads really is `points * STRIDE - 1`, and
    // it is the last one that fits.
    let last_read = MAX_GRID_POINTS * stride - 1;
    assert!(
        last_read <= u32::MAX as u64,
        "the last index at the limit is {last_read}, past u32::MAX"
    );
    assert_eq!(
        last_read,
        u32::MAX as u64 - (u32::MAX as u64) % stride - 1,
        "the last readable index is u32::MAX rounded down to a whole stride, minus \
         one; the derivation and that index have drifted apart"
    );
}

/// The advertised limit is below the old one, by the factor the finding claimed.
#[test]
fn the_old_limit_was_above_what_the_kernel_can_address() {
    assert_eq!(old_limit(), 268_435_456);
    assert_eq!(old_limit() / MAX_GRID_POINTS, 2, "2.5x, truncated");
    assert!(
        old_limit() * (GRID_TYPE_COUNT * MAPS_PER_TYPE) as u64 > u32::MAX as u64,
        "the old limit is claimed to be unreachable and is not"
    );
}

/// A grid the kernel cannot address is refused, and the message names the number.
///
/// The number is the whole point. From inside the kernel there is no overflow to
/// report -- the `u32` wraps and the read succeeds -- so a message about an
/// overflow would be describing something that did not happen.
#[test]
fn a_grid_the_kernel_cannot_address_is_refused_by_name() {
    // One point over the limit, laid out as a real grid would be.
    let dims = [MAX_GRID_POINTS as usize + 1, 1, 1];
    let err = check_gpu_dims(dims).expect_err("one point over the limit must be refused");
    let text = err.to_string();
    assert!(
        text.contains(&MAX_GRID_POINTS.to_string()),
        "the refusal must name the limit {MAX_GRID_POINTS}; got: {text}"
    );
    assert!(
        text.contains(&(MAX_GRID_POINTS + 1).to_string()),
        "the refusal must name the point count that was exceeded; got: {text}"
    );
    assert!(
        !text.to_lowercase().contains("overflow"),
        "the kernel does not overflow, it wraps and reads the wrong address, so an \
         overflow message describes something that did not happen; got: {text}"
    );

    // The old limit, which is what a grid built before this change would carry.
    let err = check_gpu_dims([old_limit() as usize, 1, 1])
        .expect_err("a grid at the old limit must now be refused");
    assert!(
        err.to_string().contains(&MAX_GRID_POINTS.to_string()),
        "got: {err}"
    );
}

/// Exactly at the limit is allowed. The guard is `>`, not `>=`.
#[test]
fn a_grid_exactly_at_the_limit_is_accepted() {
    check_gpu_dims([MAX_GRID_POINTS as usize, 1, 1])
        .expect("a grid exactly at the limit is addressable and must be accepted");
    check_gpu_dims([0, 0, 0]).expect("a zero point count is under the limit");
}

/// A `dims` whose product overflows is over the limit, not a second failure.
#[test]
fn an_overflowing_point_count_is_reported_as_over_the_limit() {
    let err = check_gpu_dims([usize::MAX, usize::MAX, usize::MAX])
        .expect_err("an overflowing point count is over the limit");
    assert!(
        err.to_string().contains(&MAX_GRID_POINTS.to_string()),
        "an overflow must be reported against the same limit rather than as its own \
         kind of failure; got: {err}"
    );
}

/// What the narrowing actually costs, stated as a fact rather than a reassurance.
///
/// The brief for this change said "a caller gets a *smaller* grid than before
/// and nothing that used to fit now fails". That is true of the grid *count*
/// and false of the *acceptance*: `MAX_GRID_POINTS` fell from 268,435,456 to
/// 107,374,182, and any box whose point count lands between those two numbers
/// used to be accepted and is now refused. Those are boxes of 16-40 GiB, so on
/// a machine that could not allocate one they failed at the allocation rather
/// than at the check -- but on a machine with 64 GiB they used to run, and they
/// do not now.
///
/// This test pins the arithmetic of that change rather than the sentiment.
#[test]
fn the_narrowing_refuses_the_band_the_old_limit_admitted() {
    let stride = (GRID_TYPE_COUNT * MAPS_PER_TYPE) as u64;
    assert!(
        old_limit() > MAX_GRID_POINTS,
        "if the limit did not move, this test is asserting nothing"
    );
    // A grid at the old limit and one just under it: both were accepted before.
    for points in [MAX_GRID_POINTS + 1, old_limit()] {
        assert!(
            points > MAX_GRID_POINTS,
            "{points} should now be over the limit"
        );
        assert!(
            points * stride > u32::MAX as u64,
            "{points} points is past what a u32 flat index can address, which is the \
             reason it is refused rather than merely discouraged"
        );
    }
}

/// Boxes at or below the new limit are still accepted, and still build.
///
/// Only modest boxes are actually constructed -- a grid at the limit is 16 GiB
/// and building one to prove arithmetic is not a trade this suite should make.
/// The acceptance *decision* for the large ones is checked through
/// [`check_gpu_dims`], which is the same code path `precalculate` refuses with.
#[test]
fn boxes_under_the_new_limit_are_still_built() {
    let spacing = 0.375;
    // 24 A at the shipped spacing: dims 65 per axis, 274625 points.
    let dims = estimate_dims(&box_of(24.0), spacing);
    assert_eq!(dims, [65, 65, 65]);
    assert!(dims[0] as u64 * dims[1] as u64 * dims[2] as u64 <= MAX_GRID_POINTS);
    check_gpu_dims(dims).expect("a 24 A box is far under the limit");

    let maps = GridMaps::precalculate(
        &test_receptor(),
        &box_of(24.0),
        &VinaScoring::new(),
        spacing,
        0,
    )
    .expect("a 24 A box at the shipped spacing must still build");
    assert_eq!(maps.dims(), dims);
    maps.check_gpu_index_range()
        .expect("a grid precalculate just built is under the limit by construction");
}

/// `precalculate` refuses an over-limit grid *before* allocating, and says which
/// number it exceeded.
///
/// This is the cheap end of the boundary: the refusal happens in the size loop
/// above `vec![]`, so this exercises the real code path at 40 GiB scale without
/// ever reserving the memory.
#[test]
fn precalculate_refuses_an_over_limit_box_before_allocating() {
    // 700 A cube at 0.375 A: 1867^3 points, far past the limit.
    let err = GridMaps::precalculate(
        &test_receptor(),
        &box_of(700.0),
        &VinaScoring::new(),
        0.375,
        0,
    )
    .expect_err("a 700 A box cannot be tabulated and must be refused");
    let text = err.to_string();
    assert!(
        text.contains(&MAX_GRID_POINTS.to_string()),
        "precalculate's refusal must name the new limit; got: {text}"
    );
    // The message must not still be quoting the old power of two.
    let stale = old_limit();
    assert!(
        !text.contains(&stale.to_string()),
        "precalculate is still quoting the old limit of {stale}; got: {text}"
    );
}

/// The boundary in box-size terms, so the constant is tied to something a caller
/// actually types.
///
/// At 0.375 A -- the shipped `DEFAULT_SPACING` -- a box of `3k` A on a side has
/// an integer `size / spacing`, so `estimate_dims` returns `8k + 1` points per
/// axis and the tabulated volume ends exactly on the box face.
#[test]
fn the_limit_in_terms_of_a_box_the_caller_can_type() {
    let spacing = dock_core::grid::DEFAULT_SPACING;
    assert_eq!(spacing, 0.375);
    // A 240 A cube at 0.375 A: 641 per axis, 263,374,721 points -- over the new
    // limit, under the old one. This is the band the narrowing refuses.
    let over = estimate_dims(&box_of(240.0), spacing);
    let over_points = over[0] as u64 * over[1] as u64 * over[2] as u64;
    assert!(
        over_points > MAX_GRID_POINTS && over_points <= old_limit(),
        "240 A should sit in the band the narrowing refuses, got {over_points}"
    );
    // A 180 A cube: 481 per axis, 111,284,641 -- just over the new limit too.
    let mid = estimate_dims(&box_of(180.0), spacing);
    let mid_points = mid[0] as u64 * mid[1] as u64 * mid[2] as u64;
    assert!(
        mid_points > MAX_GRID_POINTS,
        "180 A should be refused at the new limit, got {mid_points}"
    );
    // A 170 A cube: 455 per axis, 94,196,375 -- under it.
    let under = estimate_dims(&box_of(170.0), spacing);
    let under_points = under[0] as u64 * under[1] as u64 * under[2] as u64;
    assert!(
        under_points <= MAX_GRID_POINTS,
        "170 A should still be accepted, got {under_points}"
    );
    check_gpu_dims(under).expect("170 A is addressable");
    check_gpu_dims(mid).expect_err("180 A is not");
    assert!(
        points_of(170.0, spacing) == under_points,
        "the helper and estimate_dims disagree"
    );
}
