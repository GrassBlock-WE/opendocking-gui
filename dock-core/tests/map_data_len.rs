//! A map file's `data` length has to agree with its `dims`.
//!
//! The finding this file tests: `GridMaps` is `Deserialize`, `dims` is a public
//! field, and the derived `Deserialize` checked neither the point ceiling, the
//! axis shape, nor the length of `data`. Two of those three gaps were closed at
//! the gate ([`check_gpu_dims`], and every axis holding two points). This is
//! the third, and it is the one that was a hole all the way to the device:
//!
//! ```text
//! {"dims":[2,3,17895697], "data":[ ... 160 floats ... ], ...}
//! ```
//!
//! deserialised, passed `check_gpu_dims`, passed `check_gpu_index_range`, and
//! was handed to `score()`, which uploaded 640 bytes for a grid `energy.wgsl`
//! addresses up to index 4,294,967,279. Measured before the fix and printed by
//! the test that used to assert the hole: a shortfall of 4,294,967,120 values,
//! 16.0 GiB of read past the end of the buffer.
//!
//! The rule now enforced is exact equality, with no lenient mode. A truncated
//! or over-long `data` has no correct reading, so there is nothing to pad,
//! truncate, or tolerate; and the refusal names the expected length and the
//! found one, because a message that said only "bad length" would leave the
//! caller to recompute the expectation the crate just declined to compute.
//!
//! The three constraints are independent, and this file checks that they are
//! not confused with one another: a grid can satisfy the ceiling and the shape
//! and still be short, and a grid can satisfy the length and still be refused
//! for the other two.

use dock_core::grid::{
    check_gpu_data_len, check_gpu_dims, map_stride, GridBox, GridMaps, MAPS_PER_TYPE,
    MAX_GRID_POINTS,
};
use dock_core::scoring::VinaScoring;
use dock_core::types::{Atom, AtomType, Element, Molecule};

/// A map document written with the field names `GridMaps` actually has.
fn map_doc(dims: [usize; 3], data_len: usize) -> String {
    let data: Vec<String> = (0..data_len).map(|i| (i % 7).to_string()).collect();
    format!(
        "{{\"min\":[0.0,0.0,0.0],\"dims\":{dims:?},\"spacing\":[1.0,1.0,1.0],\
          \"data\":[{}],\"box_\":{{\"min\":[0.0,0.0,0.0],\"max\":[10.0,10.0,10.0]}},\
          \"receptor_heavy\":[]}}",
        data.join(",")
    )
}

fn test_receptor() -> Molecule {
    let atoms: Vec<Atom> = vec![
        Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(2, [2.0, 0.0, 0.0], Element::O, AtomType::OA),
    ];
    Molecule::from_atoms(atoms).expect("receptor")
}

/// The five malformed shapes, as `(label, dims, data_len)`.
fn malformed() -> Vec<(&'static str, [usize; 3], usize)> {
    let s = map_stride();
    let cube = [4usize, 4, 4];
    let exact = s * 4 * 4 * 4; // 2560
    vec![
        ("short by a lot", cube, 8),
        ("long by a lot", cube, 4096),
        ("short by exactly one", cube, exact - 1),
        ("long by exactly one", cube, exact + 1),
        // dims legal, product overflows when multiplied by the stride.
        ("dims product overflows", [1 << 22, 1 << 22, 1 << 22], 160),
    ]
}

/// Each malformed document is refused, and the refusal names both numbers.
///
/// Asserting the *numbers* rather than a keyword is the point: a guard that
/// refused for the wrong reason -- the shape rule, or the ceiling -- would
/// pass a message check and leave the payload rule untested.
#[test]
fn a_payload_that_disagrees_with_dims_is_refused_by_name() {
    let s = map_stride();
    let mut checked = 0usize;
    for (label, dims, data_len) in malformed() {
        let doc = map_doc(dims, data_len);
        let err = serde_json::from_str::<GridMaps>(&doc)
            .expect_err("a payload that disagrees with dims must not deserialise");
        let text = err.to_string();

        // The field responsible is named, not just "invalid map".
        assert!(
            text.contains("grid data length"),
            "{label}: the refusal does not name the field; got: {text}"
        );
        // What was found is named.
        assert!(
            text.contains(&format!("`data` holds {data_len} value(s)")),
            "{label}: the refusal does not name the found length {data_len}; got: {text}"
        );
        // What was expected is named, and it is `points * map_stride()`.
        let expected = check_expected(dims);
        if let Some(e) = expected {
            assert!(
                text.contains(&format!("must hold exactly {e} (")),
                "{label}: the refusal does not name the expected length {e}; got: {text}"
            );
        }
        // And it is refused as a length problem, never as a shape or size one:
        // all five of these have three well-formed axes, so a message about a
        // one-point axis or about the ceiling would be describing a fault
        // these documents do not have.
        assert!(
            !text.contains("one-point") && !text.contains("fewer than two"),
            "{label}: refused for the shape rule's reason, which this grid does not \
             trip: {text}"
        );
        checked += 1;
    }
    assert_eq!(checked, 5, "the malformed set changed shape");
    eprintln!(
        "MEASURED: {} malformed documents refused at deserialisation, each naming the \
         field, the found length and the expected length; map_stride() = {s}",
        checked
    );
}

/// The length a set of `dims` implies, or `None` when it cannot be represented.
fn check_expected(dims: [usize; 3]) -> Option<u128> {
    let mut points: u128 = 1;
    for d in dims {
        points = points.checked_mul(d as u128)?;
    }
    points.checked_mul(map_stride() as u128)
}

/// A grid that satisfies the ceiling and the shape can still be short.
///
/// This is the shape of the original finding, kept as a regression: a grid on
/// the published ceiling, every axis holding at least two points, carrying 160
/// floats. It is the case where the other two rules are *satisfied*, so before
/// the fix it reached `score()`.
#[test]
fn a_grid_on_the_ceiling_with_a_short_payload_is_still_refused() {
    // 107,374,182 = 2 * 3 * 17,895,697 exactly, so this sits on the published
    // ceiling with every axis legal -- both addressability rules pass.
    let dims = [2usize, 3, 17_895_697];
    let expected = check_expected(dims).expect("representable");
    assert_eq!(expected, 4_294_967_280);
    check_gpu_dims(dims).expect("this dims is arithmetically legal: the point ceiling holds");
    for d in dims {
        assert!(d >= 2, "axis {d} is degenerate, so this would not be the case under test");
    }

    let doc = map_doc(dims, 160);
    let err = serde_json::from_str::<GridMaps>(&doc)
        .expect_err("the ceiling dims and a short payload must not deserialise");
    let text = err.to_string();
    assert!(
        text.contains("4294967280"),
        "the refusal must name the expected length; got: {text}"
    );
    assert!(
        text.contains("`data` holds 160 value(s)"),
        "the refusal must name the found length; got: {text}"
    );
    // The rule is checked, not the ceiling: the dims are legal.
    assert!(
        !text.contains(&MAX_GRID_POINTS.to_string()) || text.contains("independent"),
        "the refusal looks like it is about the point ceiling rather than the \
         payload: {text}"
    );
    eprintln!(
        "MEASURED: dims {dims:?} = {MAX_GRID_POINTS} points passes check_gpu_dims and \
         every axis holds >= 2 points, yet 160 floats is refused; expected length \
         {expected}, kernel's last read index {}",
        expected - 1
    );
}

/// Every legal document in the tree still parses.
///
/// The set is the honest one: there are no `.map`/`.json` fixtures on disk, so
/// the only `GridMaps` documents anywhere in this workspace are the ones these
/// tests write. What has to keep working is therefore (a) a real grid from the
/// only producer in the crate, round-tripped, and (b) the deliberately odd but
/// legal `dims` that `estimate_dims` floors at 2 and that the degenerate-axis
/// sweep in `grid.rs` still deserialises.
#[test]
fn every_legal_fixture_still_parses() {
    // (a) The real producer, round-tripped. `precalculate` is the only place in
    // the crate that fills `data`, so if it ever disagreed with `dims` the
    // crate would refuse its own output.
    let maps = GridMaps::precalculate(
        &test_receptor(),
        &GridBox::new([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]).expect("box"),
        &VinaScoring::new(),
        0.5,
        0,
    )
    .expect("precalculate");
    assert_eq!(
        maps.data_len(),
        map_stride() * maps.point_count(),
        "precalculate built a grid whose payload disagrees with its dims; the \
         deserialiser would now refuse the crate's own output"
    );
    let json = serde_json::to_string(&maps).expect("a GridMaps serialises");
    let back: GridMaps = serde_json::from_str(&json).expect("a real map round-trips");
    assert_eq!(back, maps, "the round trip changed the grid");
    eprintln!(
        "MEASURED: a precalculated {}x{}x{} grid ({} points, {} values, {:.1} MiB of \
         text) round-trips byte-for-byte through the wire form",
        maps.dims()[0],
        maps.dims()[1],
        maps.dims()[2],
        maps.point_count(),
        maps.data_len(),
        json.len() as f64 / (1024.0 * 1024.0)
    );

    // (b) Odd but legal shapes, including the ones with fewer than two points on
    // an axis and the empty grid. `check_gpu_index_range` refuses the
    // degenerate ones at the *gate* -- that is unchanged and correct -- but a
    // legal document still has to load, or the length rule would be silently
    // refusing shapes as a side effect.
    let s = map_stride();
    let legal: Vec<([usize; 3], &str)> = vec![
        ([0, 0, 0], "empty grid: zero points, zero values"),
        ([1, 1, 1], "single point on every axis"),
        ([2, 2, 2], "the smallest interpolable grid"),
        ([1, 4, 4], "one point on axis 0"),
        ([4, 1, 4], "one point on axis 1"),
        ([4, 4, 1], "one point on axis 2"),
        ([0, 4, 4], "zero points on axis 0"),
        ([4, 0, 4], "zero points on axis 1"),
        ([4, 4, 0], "zero points on axis 2"),
        ([4, 4, 4], "a plain cube"),
        ([3, 1, 7], "non-cubic, degenerate on one axis"),
        ([2, 3, 5], "non-cubic and non-monotone"),
        ([1, 1, 4096], "long and thin"),
    ];
    for (dims, note) in &legal {
        let want = s * dims.iter().product::<usize>();
        let doc = map_doc(*dims, want);
        let got: GridMaps = serde_json::from_str(&doc)
            .unwrap_or_else(|e| panic!("{note}: dims {dims:?} with {want} values must load: {e}"));
        assert_eq!(got.dims(), *dims, "{note}: dims were not preserved");
        assert_eq!(got.data_len(), want, "{note}: data was not preserved");
        // Where the length is right, the length rule is satisfied even when
        // the shape rule is not -- the two are separate questions.
        check_gpu_data_len(*dims, want)
            .unwrap_or_else(|e| panic!("{note}: dims {dims:?} with {want} values: {e}"));
    }
    eprintln!(
        "MEASURED: {} legal documents parsed, from an empty grid to a \
         1x1x4096 sliver, including all six one-point-or-zero axis shapes",
        legal.len()
    );
}

/// The three rules refuse for three different reasons.
///
/// Each grid below is legal for the other two rules, so whichever message comes
/// back is the one that fired. If two of these ever produced the same message
/// the rules would have been collapsed into one, and the test would go red
/// rather than quietly agreeing.
#[test]
fn the_three_rules_are_not_the_same_rule() {
    let s = map_stride();

    // 1. The point ceiling. Shape is legal, and the payload is deliberately
    //    wrong -- it cannot be right, 16 GiB is not allocatable -- so only the
    //    ceiling can be what refuses it.
    let too_big = [MAX_GRID_POINTS as usize + 1, 2, 2];
    let e = check_gpu_dims(too_big).expect_err("past the ceiling");
    assert!(
        e.to_string().contains(&MAX_GRID_POINTS.to_string()),
        "the ceiling rule must name the ceiling; got: {e}"
    );
    check_gpu_data_len(too_big, s * 8).expect_err("the payload is also wrong here");

    // 2. The shape. Every axis must hold two points. Payload is correct.
    for dims in [[1usize, 4, 4], [4, 0, 4], [4, 4, 1]] {
        let want = s * dims.iter().product::<usize>();
        check_gpu_data_len(dims, want)
            .unwrap_or_else(|e| panic!("dims {dims:?} with {want} values is a length-legal grid: {e}"));
        assert!(
            check_gpu_dims(dims).is_ok(),
            "dims {dims:?} is under the ceiling, so only the shape rule can refuse it"
        );
    }

    // 3. The payload. Shape is legal and the point count is legal.
    for delta in [1usize, 2, s - 1, s, s + 1, 8 * s] {
        let want = s * 8;
        let dims = [2usize, 2, 2];
        check_gpu_dims(dims).expect("a 2x2x2 grid is under the ceiling");
        let e = check_gpu_data_len(dims, want + delta)
            .expect_err("a payload that disagrees with dims must be refused");
        let t = e.to_string();
        assert!(
            t.contains(&format!("must hold exactly {want} (")),
            "delta {delta}: the refusal must name the expected length {want}; got: {t}"
        );
        assert!(
            t.contains(&format!("`data` holds {} value(s)", want + delta)),
            "delta {delta}: the refusal must name the found length {}; got: {t}",
            want + delta
        );
    }
    eprintln!(
        "MEASURED: 3 rules exercised independently -- the ceiling ({MAX_GRID_POINTS} \
         points), 3 degenerate shapes with a correct payload, and 6 payload \
         mismatches on a legal 2x2x2 grid; every refusal named its own numbers"
    );
}

/// The wire form has to keep carrying exactly the fields the struct has.
///
/// `GridMaps` deserialises through `GridMapsWire` so the payload rule can run
/// at load. The price is a second declaration of the same seven fields, and a
/// field added to one and not the other would be silently dropped on the way
/// in. So the field set is pinned by name here, and the compatibility
/// behaviour of the old derive is pinned too, because "unchanged on disk" is
/// the claim that has to be true and not merely intended.
#[test]
fn the_wire_form_keeps_the_field_names_the_format_uses() {
    let s = map_stride();
    let dims = [2usize, 2, 2];
    let doc = map_doc(dims, s * 8);

    // The serialised field set, pinned. A field added to `GridMaps` and not to
    // `GridMapsWire` would show up here.
    let maps: GridMaps = serde_json::from_str(&doc).expect("the legal document loads");
    let value = serde_json::to_value(&maps).expect("serialises");
    let mut names: Vec<&str> = value
        .as_object()
        .expect("a GridMaps is a map")
        .keys()
        .map(String::as_str)
        .collect();
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
        "the serialised field set moved; a field added to one declaration and \
         not the other is now silently discarded on the way in"
    );

    // `box_` keeps its trailing underscore. A rename to `box` is a
    // plausible-looking edit, `box` is not a Rust keyword, and serde would not
    // complain -- it would just leave `box_` at its default and hand back a
    // grid covering nothing.
    let renamed = doc.replace("\"box_\"", "\"box\"");
    assert!(
        serde_json::from_str::<GridMaps>(&renamed).is_err(),
        "a document spelling the field `box` deserialised, so `box_` is not \
         load-bearing and the pinned name is not pinned"
    );

    // A file written before `unknown_atom_types` existed has to keep loading,
    // as 0. "This map predates the counter" is a weaker claim than "every atom
    // was recognised", and it is not the same claim as an error.
    let maps: GridMaps = serde_json::from_str(&doc).expect("a pre-counter map file loads");
    let round: GridMaps = serde_json::from_value(value)
        .expect("a GridMaps round-trips through its wire form");
    assert_eq!(
        round.data_len(),
        maps.data_len(),
        "the round trip changed the payload"
    );

    // Unknown fields are still ignored, exactly as the derive ignored them: a
    // file carrying a field from a newer build must not become unreadable.
    let extended = doc.replace("\"min\":", "\"future_field\":7,\"min\":");
    let maps: GridMaps = serde_json::from_str(&extended).expect("unknown fields are ignored");
    assert_eq!(maps.dims(), dims);

    eprintln!(
        "MEASURED: 7 field names pinned on the wire form; `box` rejected, \
         `unknown_atom_types` defaulted, unknown field ignored"
    );
}

/// `map_stride()` is the multiplier, and the expectation moves with it.
///
/// If `MAPS_PER_TYPE` or `GRID_TYPE_COUNT` changes, the expected length in
/// every message above changes with it. This is the assertion that the tests
/// are tracking the crate's own stride rather than a hard-coded 40.
#[test]
fn the_expected_length_follows_map_stride() {
    let s = map_stride();
    let dims = [3usize, 5, 7];
    let want = s * 3 * 5 * 7;
    assert_eq!(want, s * 105);
    check_gpu_data_len(dims, want).expect("the implied length is accepted");
    let e = check_gpu_data_len(dims, want - 1).expect_err("one value short is refused");
    assert!(
        e.to_string().contains(&format!("must hold exactly {want} (")),
        "the expected length must track map_stride() = {s}; got: {e}"
    );
    assert_eq!(
        MAPS_PER_TYPE, 4,
        "MAPS_PER_TYPE moved; energy.wgsl's STRIDE is asserted against map_stride() \
         elsewhere, and this file's expected lengths follow it"
    );
    eprintln!("MEASURED: dims {dims:?} imply {want} values at map_stride() = {s}");
}


/// Both `u128` overflow arms are reachable, and each says which one fired.
///
/// # Why this is a test and not a comment saying "unreachable"
///
/// `check_gpu_data_len` multiplies in `u128` with `checked_mul`, and has two
/// `None` arms: one for the product of the three `dims` axes overflowing, one
/// for that product times the stride overflowing. The arithmetic is exercised
/// everywhere else in this file, but *those two arms* were not constructed
/// anywhere, and the previous note about them said they were unreachable in
/// practice. That was the wrong word, and a reader could not tell the
/// difference between "unreachable" and "untested":
///
/// * **The point arm** needs the three axes alone to exceed `u128::MAX = 2^128
///   - 1`. On a 64-bit target `usize::MAX = 2^64 - 1`, and `usize::MAX`
///   squared is `2^128 - 2^65 + 1`, which still fits -- so the **third** axis is
///   what overflows, and `dims = [usize::MAX; 3]` reaches it.
/// * **The value arm** needs `points` to fit while `points * map_stride()`
///   does not, so `points > u128::MAX / 40 ~ 8.5e36`. `2^126` clears that and
///   `2^125` does not, so `dims = [1 << 42; 3]` reaches it. This one is
///   reachable on any target wide enough for `1 << 42` to be a `usize`, which
///   is why it is not `cfg`-gated.
///
/// Neither is reachable from a *real* map: `precalculate` refuses an over-large
/// box before allocating, and a `data` of the implied length cannot be built on
/// a machine the implied length overflows. But "unreachable in practice" and
/// "unreachable" are different claims, and only the second one is a reason not
/// to test. `dims` is the function's only input and these are values of it.
///
/// Each arm is also required to *name itself*, because the two produce
/// structurally identical refusals otherwise and a test that only checked
/// "it errored" would pass whichever fired.
#[test]
fn the_u128_overflow_arms_are_reached_and_named() {
    let s = map_stride();

    // Arm 1: the product of the axes overflows. 64-bit only -- on a 32-bit
    // target `usize::MAX^3` is about 2^96, which u128 holds, and the arm
    // genuinely cannot be reached. That is stated here rather than hidden
    // behind a `cfg` that would make the test silently disappear.
    if usize::BITS >= 64 {
        let dims = [usize::MAX, usize::MAX, usize::MAX];
        // The precondition, proved rather than assumed: the first two axes fit.
        let two = (usize::MAX as u128) * (usize::MAX as u128);
        assert!(
            two <= u128::MAX,
            "the point arm needs the first two axes to fit, so the third is what \
             overflows; two = {two} vs u128::MAX = {}",
            u128::MAX
        );
        assert!(
            two.checked_mul(usize::MAX as u128).is_none(),
            "the third axis must overflow for this fixture to reach the arm"
        );
        let e = check_gpu_data_len(dims, 0).expect_err("[usize::MAX; 3] must be refused");
        let text = e.to_string();
        assert!(
            text.contains("the point count overflows u128"),
            "the point arm must say so; got: {text}"
        );
        assert!(
            !text.contains("the value count overflows"),
            "the two arms must not produce the same sentence; got: {text}"
        );
        assert!(
            text.contains("grid data length"),
            "the refusal must still name the field; got: {text}"
        );
        eprintln!(
            "MEASURED: dims {dims:?} reaches the *point* overflow arm (u128::MAX = {})",
            u128::MAX
        );
    } else {
        eprintln!(
            "NOT MEASURED: usize::BITS = {}, so the point overflow arm cannot be \
             reached on this target -- usize::MAX^3 fits in u128. Stated, not \
             skipped silently.",
            usize::BITS
        );
    }

    // Arm 2: the axes fit, the product times the stride does not.
    let dims = [1usize << 42, 1 << 42, 1 << 42];
    let points = (1u128 << 42) * (1u128 << 42) * (1u128 << 42);
    assert_eq!(points, 1u128 << 126);
    assert!(points <= u128::MAX, "the point arm must not fire for this fixture");
    assert!(
        points.checked_mul(s as u128).is_none(),
        "the value arm needs points * {s} to overflow; {} * {s} = {:?}",
        points,
        points.checked_mul(s as u128)
    );
    let e = check_gpu_data_len(dims, 0).expect_err("[1<<42; 3] must be refused");
    let text = e.to_string();
    assert!(
        text.contains("the value count overflows u128"),
        "the value arm must say so; got: {text}"
    );
    assert!(
        !text.contains("the point count overflows"),
        "the two arms must not produce the same sentence; got: {text}"
    );
    // The point count is still named, because it is a real number here and the
    // caller is entitled to it.
    assert!(
        text.contains(&points.to_string()),
        "the refusal must name the point count, which fits: got: {text}"
    );
    eprintln!(
        "MEASURED: dims {dims:?} = {points} points reaches the *value* overflow arm \
         ({points} * {s} > u128::MAX = {})",
        u128::MAX
    );

    // And the boundary between the two, so the arms are not the same test twice.
    // `1 << 41` per axis is `2^123` points, times 40 is ~2^128.4 -- just over.
    // `1 << 40` is `2^120` points, times 40 is ~2^125.3 -- comfortably under, and
    // so it takes the ordinary mismatch arm instead.
    let over = [1usize << 41, 1 << 41, 1 << 41];
    let under = [1usize << 40, 1 << 40, 1 << 40];
    let over_text = check_gpu_data_len(over, 0).expect_err("over the line").to_string();
    let under_text = check_gpu_data_len(under, 0).expect_err("under the line").to_string();
    assert!(
        over_text.contains("the value count overflows u128"),
        "[1<<41; 3] = 2^123 points is past u128::MAX / 40; got: {over_text}"
    );
    assert!(
        under_text.contains("must hold exactly"),
        "[1<<40; 3] = 2^120 points is representable, so it must reach the ordinary \
         mismatch arm and name a length; got: {under_text}"
    );
    eprintln!(
        "MEASURED: [1<<41; 3] = 2^123 overflows the value arm while [1<<40; 3] = \
         2^120 does not, so the arm's threshold is exercised from both sides"
    );
}
