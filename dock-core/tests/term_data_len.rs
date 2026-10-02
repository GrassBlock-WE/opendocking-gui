//! A term-map file's `data` length has to agree with its `dims`, and this is
//! the same file as `map_data_len.rs` one type over.
//!
//! # What was found
//!
//! `GridMaps` deserialised a document whose `data` length disagreed with its
//! `dims`, passed both addressability rules, and reached `score()` — which
//! uploaded 640 bytes for a grid `energy.wgsl` addresses up to index
//! 4,294,967,279. `tests/map_data_len.rs` closed that at the wire, through
//! `GridMapsWire` and `check_gpu_data_len`.
//!
//! `TermMaps` was **the same hole with the same shape**: a private `data`, a
//! public `dims`, a derived `Deserialize`, and a stride implied by `dims` and
//! stored nowhere. Same file, same rule, same refusal text shape, same
//! both-directions proof — and it was left open.
//!
//! # This is a landmine defused, not an exploit fixed
//!
//! Nothing reaches a device through a `TermMaps` today, and the claim is
//! carried on the type rather than left to inference:
//!
//! * `TermMaps::BACKEND` is `TermBackend::Cpu`;
//! * the WGSL entry point takes a `&GridMaps`, and there is no overload, no
//!   generic and no trait that would let a `TermMaps` be passed instead;
//! * `grid.rs::under_the_gpu_feature_the_terms_are_still_cpu_only` asserts all
//!   three of those on every `gpu` build.
//!
//! So the blast radius today is not a device read. `terms_at` indexes
//! `self.data` with a plain `[]`, and Rust bounds-checks that, so a short
//! `data` is an **index-out-of-bounds panic** in the CPU interpolator — not a
//! silent read past the allocation, not information disclosure, not memory
//! corruption. Nothing here was exploitable, and this file does not claim to
//! have fixed an exploit.
//!
//! What it was is a *landmine*: a hole inert only because no path reaches a
//! device with this type, laid in the same place as the live one and one
//! refactor away from being live. A per-term GPU path — which
//! `grid.rs::the_per_term_decomposition_is_cpu_only_and_says_so` already
//! watches for, and turns red on — would have inherited it exactly as
//! `GridMaps` did. It is closed now, on the same terms, through the same shared
//! rule (`grid::check_payload_len`), and the tests below are the same
//! both-directions proof `map_data_len.rs` makes.
//!
//! The two types' refusals say different things about the consequence, and
//! that is deliberate: `energy.wgsl` reads the production payload through a
//! storage buffer the driver bounds, while `terms_at` indexes a `Vec` that
//! Rust checks. A refusal that blurred those would be a refusal that could be
//! true of neither.

use dock_core::grid::{check_term_data_len, term_stride, GridBox, TermMaps};
use dock_core::scoring::{VinaScoring, TERM_FIELDS};
use dock_core::types::{Atom, AtomType, Element, Molecule};

/// A term-map document written with the field names `TermMaps` actually has.
fn term_doc(dims: [usize; 3], data_len: usize) -> String {
    let data: Vec<String> = (0..data_len).map(|i| (i % 7).to_string()).collect();
    let joined = data.join(",");
    format!(
        "{{\"min\":[0.0,0.0,0.0],\"dims\":{dims:?},\"spacing\":[1.0,1.0,1.0],\
          \"data\":[{joined}],\"box_\":{{\"min\":[0.0,0.0,0.0],\"max\":[1.0,1.0,1.0]}}}}"
    )
}

/// A receptor with an apolar and a polar atom, built through the public
/// constructor because `Molecule`'s fields are private to the crate.
fn test_receptor() -> Molecule {
    Molecule::from_atoms(vec![
        Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(2, [2.0, 0.0, 0.0], Element::O, AtomType::OA),
    ])
    .expect("two atoms form a valid molecule")
}

/// The malformed shapes, as `(label, dims, data_len)`. The same five
/// `map_data_len.rs` uses, at this stride.
fn malformed() -> Vec<(&'static str, [usize; 3], usize)> {
    let s = term_stride();
    let cube = [4usize, 4, 4];
    let exact = s * 4 * 4 * 4;
    vec![
        ("short by a lot", cube, 8),
        ("long by a lot", cube, 4096),
        ("short by exactly one", cube, exact - 1),
        ("long by exactly one", cube, exact + 1),
        // dims legal, product overflows when multiplied by the stride.
        ("dims product overflows", [1 << 22, 1 << 22, 1 << 22], 160),
    ]
}

/// The length a set of `dims` implies, or `None` when it cannot be held.
fn check_expected(dims: [usize; 3]) -> Option<u128> {
    let mut points: u128 = 1;
    for d in dims {
        points = points.checked_mul(d as u128)?;
    }
    points.checked_mul(term_stride() as u128)
}

/// Each malformed term-map document is refused, and the refusal names both
/// numbers.
///
/// The *numbers* rather than a keyword, for the reason `map_data_len.rs` gives:
/// a guard that refused for the wrong reason would pass a message check and
/// leave the payload rule untested. And the field name, which must be
/// `term data length` and **not** `grid data length` — sharing the
/// implementation must not mean sharing the error, or a caller cannot tell
/// which of its two map files was rejected.
#[test]
fn a_term_payload_that_disagrees_with_dims_is_refused_by_name() {
    let s = term_stride();
    let mut checked = 0usize;
    for (label, dims, data_len) in malformed() {
        let doc = term_doc(dims, data_len);
        let err = serde_json::from_str::<TermMaps>(&doc)
            .expect_err("a payload that disagrees with dims must not deserialise");
        let text = err.to_string();

        assert!(
            text.contains("term data length"),
            "{label}: the refusal does not name the term field; got: {text}"
        );
        assert!(
            !text.contains("grid data length"),
            "{label}: refused as a *grid* payload; the two map types share the \
             rule but not the message, or a caller cannot tell which file was \
             rejected. Got: {text}"
        );
        assert!(
            text.contains(&format!("`data` holds {data_len} value(s)")),
            "{label}: the refusal does not name the found length {data_len}; got: {text}"
        );
        if let Some(e) = check_expected(dims) {
            assert!(
                text.contains(&format!("must hold exactly {e} (")),
                "{label}: the refusal does not name the expected length {e}; got: {text}"
            );
            // And the expected length is computed at *this* stride, so a
            // shared implementation cannot have quietly kept map_stride().
            assert!(
                text.contains(&format!("term_stride() = {s}")),
                "{label}: the refusal does not name term_stride() = {s}; got: {text}"
            );
        }
        // Refused as a length problem: all five of these have three
        // well-formed axes, so a message about a one-point axis would be
        // describing a fault these documents do not have.
        assert!(
            !text.contains("one-point") && !text.contains("fewer than two"),
            "{label}: refused for the shape rule's reason, which this document does \
             not trip: {text}"
        );
        checked += 1;
    }
    assert_eq!(checked, 5, "the malformed set changed shape");
    eprintln!(
        "MEASURED: {} malformed term documents refused at deserialisation, each \
         naming the term field, the found length, the expected length and \
         term_stride() = {s}",
        checked
    );
}

/// The consequence text is the *CPU's*, and says so.
///
/// This is the one place the two refusals must differ in substance rather than
/// in wording. `energy.wgsl` reads its payload through a storage buffer the
/// driver bounds, so a short one is a device-side read past an allocation.
/// `terms_at` indexes a `Vec` that Rust checks, so a short one is a panic. A
/// `TermMaps` refusal that talked about `energy.wgsl` would be describing a
/// read that cannot happen, and one that talked about "reading garbage" would
/// be describing a failure mode Rust does not have.
#[test]
fn the_term_refusal_names_the_cpu_panic_and_not_a_device_read() {
    let dims = [2usize, 2, 2];
    let s = term_stride();
    let doc = term_doc(dims, 8);
    let text = serde_json::from_str::<TermMaps>(&doc)
        .expect_err("8 values for 8 points at stride 60 is short")
        .to_string();

    assert!(
        text.contains("index-out-of-bounds panic"),
        "the refusal must state the consequence Rust actually has; got: {text}"
    );
    assert!(
        text.contains("terms_at"),
        "the refusal must name the reader that would fault; got: {text}"
    );
    // The *consequence* clause must not attribute the fault to a shader. The
    // message does mention `energy.wgsl` -- once, in the closing sentence,
    // explaining why the GPU point ceiling is deliberately not cited as a rule
    // this one is independent of. So the check is on the clause that states
    // the consequence, not on the whole string: a blanket "does not contain
    // energy.wgsl" would forbid the sentence that keeps the message honest.
    let consequence = text
        .split("index-out-of-bounds panic")
        .nth(1)
        .expect("the panic clause is present, asserted above")
        .split("A truncated or over-long")
        .next()
        .expect("the clause ends where the next sentence begins");
    assert!(
        !consequence.contains("energy.wgsl"),
        "the consequence clause must attribute the fault to the CPU reader, not \
         to a shader: {consequence}"
    );
    assert!(
        !consequence.contains("storage buffer"),
        "the consequence must not describe a device-side buffer read, which is \
         the GridMaps failure mode and not this one: {consequence}"
    );
    // And the point ceiling is named as explicitly *not* a rule here.
    assert!(
        text.contains("says nothing about a CPU-only vector"),
        "the closing must say why the GPU ceiling is not cited for a TermMaps; \
         got: {text}"
    );
    // The byte count is still named, because the caller is entitled to it.
    assert!(
        text.contains("index 479"),
        "expected last-read index not named: {text}"
    );
    eprintln!(
        "MEASURED: a TermMaps with dims {dims:?} and 8 values is refused with the \
         CPU index-out-of-bounds consequence, naming terms_at and index 479 \
         (8 points x {s}), and its consequence clause names no shader"
    );
}

/// Every legal term-map document still parses, and the crate's own output is one.
///
/// Same both-directions obligation `map_data_len.rs` discharges for
/// `GridMaps`: a guard that refuses too much is as broken as one that refuses
/// too little, and the only honest way to show it does not is to feed it real
/// documents.
#[test]
fn every_legal_term_fixture_still_parses() {
    // (a) The real producer, round-tripped. `precalculate` is the only place
    // in the crate that fills a `TermMaps::data`, so if it ever disagreed with
    // its own `dims` the crate would refuse its own output.
    let terms = TermMaps::precalculate(
        &test_receptor(),
        &GridBox::new([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]).expect("box"),
        &VinaScoring::new(),
        0.5,
        1,
    )
    .expect("terms");
    assert_eq!(
        terms.data_len(),
        term_stride() * terms.dims()[0] * terms.dims()[1] * terms.dims()[2],
        "precalculate built a term map whose payload disagrees with its dims; the \
         deserialiser would now refuse the crate's own output"
    );
    let json = serde_json::to_string(&terms).expect("a TermMaps serialises");
    let back: TermMaps = serde_json::from_str(&json).expect("a real term map round-trips");
    assert_eq!(back, terms, "the round trip changed a real TermMaps");

    // (b) The odd-but-legal `dims` that `estimate_dims` floors at 2.
    for dims in [[2usize, 2, 2], [2, 3, 5], [3, 5, 7]] {
        let doc = term_doc(dims, term_stride() * dims[0] * dims[1] * dims[2]);
        let parsed: TermMaps = serde_json::from_str(&doc).expect("an exactly-sized document loads");
        assert_eq!(parsed.dims(), dims);
        assert_eq!(
            parsed.data_len(),
            term_stride() * dims[0] * dims[1] * dims[2]
        );
    }
    // And the rule itself, called directly, agrees with the deserialiser.
    let dims = [3usize, 5, 7];
    let want = term_stride() * 3 * 5 * 7;
    check_term_data_len(dims, want).expect("the implied length is accepted");
    let e = check_term_data_len(dims, want - 1).expect_err("one value short is refused");
    assert!(
        e.to_string()
            .contains(&format!("must hold exactly {want} (")),
        "the expected length must track term_stride(); got: {e}"
    );
    eprintln!(
        "MEASURED: a real TermMaps round-trips, three odd-but-legal dims load, and \
         the rule accepts {want} at term_stride() = {} and refuses {want_minus}",
        term_stride(),
        want_minus = want - 1
    );
}

/// The wire form has to keep carrying exactly the fields the struct has.
///
/// The price of catching a bad `data` length at the wire is a second
/// declaration of the same five fields. A field added to `TermMaps` and not to
/// `TermMapsWire` would be silently dropped on the way in — and for a *term*
/// map that means a dropped field is a dropped Vina term, which is a wrong
/// energy rather than a wrong count.
#[test]
fn the_term_wire_form_keeps_the_field_names_the_format_uses() {
    let dims = [2usize, 2, 2];
    let doc = term_doc(dims, term_stride() * 8);

    let terms: TermMaps = serde_json::from_str(&doc).expect("the legal document loads");
    let value = serde_json::to_value(&terms).expect("serialises");
    let mut names: Vec<&str> = value
        .as_object()
        .expect("a TermMaps is a map")
        .keys()
        .map(String::as_str)
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec!["box_", "data", "dims", "min", "spacing"],
        "the serialised field set moved; a field added to one declaration and not \
         the other is now silently discarded on the way in"
    );

    // `box_` keeps its trailing underscore, for the reason
    // `map_data_len.rs` gives: serde would not complain about a rename, it
    // would just hand back a tabulation covering nothing.
    let renamed = doc.replace("\"box_\"", "\"box\"");
    assert!(
        serde_json::from_str::<TermMaps>(&renamed).is_err(),
        "a document spelling the field `box` deserialised, so `box_` is not \
         load-bearing and the pinned name is not pinned"
    );

    // Unknown fields are still ignored, exactly as the derive ignored them: a
    // file carrying a field from a newer build must not become unreadable.
    let extended = doc.replace("\"min\":", "\"future_field\":7,\"min\":");
    let parsed: TermMaps = serde_json::from_str(&extended).expect("unknown fields are ignored");
    assert_eq!(parsed.dims(), dims);

    // And there is no `#[serde(default)]` on any field of `TermMapsWire`, which
    // is the point at which a document that *should* be refused would start
    // loading. Pinned by asserting an absent field is an error, not a default.
    let missing = doc.replace("\"spacing\":[1.0,1.0,1.0],", "");
    assert!(
        serde_json::from_str::<TermMaps>(&missing).is_err(),
        "a TermMaps document with no `spacing` deserialised; `TermMapsWire` is \
         meant to have no defaulted field, so a truncated document cannot load"
    );
    eprintln!(
        "MEASURED: 5 field names pinned on the term wire form; `box` rejected, a \
         missing `spacing` rejected (no serde default), unknown field ignored"
    );
}

/// The expectation follows `term_stride()`, and this is not `map_stride()`.
///
/// The whole reason `TermMaps` needed its own rule rather than a reuse of the
/// grid one is that the two strides differ — 60 against 40. If the shared
/// implementation ever stopped being parameterised by the stride, every
/// expected length in this file would silently become the *grid* length, and
/// the malformed cases above would start failing for the wrong reason. This
/// is the assertion that they are still failing for the right one.
#[test]
fn the_expected_length_follows_term_stride_and_not_map_stride() {
    let s = term_stride();
    assert_eq!(
        s, 60,
        "term_stride() = GRID_TYPE_COUNT * TERM_FIELDS = 10 * 6 = 60 f32 per point"
    );
    assert_eq!(
        s,
        10 * TERM_FIELDS,
        "the stride must follow the crate's own constants"
    );
    assert_ne!(
        s,
        dock_core::grid::map_stride(),
        "the two strides must differ"
    );

    let dims = [3usize, 5, 7];
    let want = s * 3 * 5 * 7;
    let grid_want = dock_core::grid::map_stride() * 3 * 5 * 7;
    assert_ne!(
        want, grid_want,
        "the two expectations must differ for this to test anything"
    );

    // The grid's length is refused here, which is the load-bearing direction:
    // accepting it would mean the term rule had become the grid rule.
    let e = check_term_data_len(dims, grid_want)
        .expect_err("a grid-strided payload must not satisfy the term rule");
    let text = e.to_string();
    assert!(
        text.contains(&format!("must hold exactly {want} (")),
        "the term rule must demand {want}, not the grid's {grid_want}; got: {text}"
    );
    // The grid's number does appear in the message -- as the *found* length,
    // because that is what was passed in -- so the check is on the expectation,
    // not on the whole string. What must not happen is the grid's number being
    // named as the required one.
    let expected_clause = text
        .split("must hold exactly ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .expect("the expectation clause is present, asserted above");
    assert_eq!(
        expected_clause,
        want.to_string(),
        "the expected length must be the term one, {want}, not the grid's {grid_want}"
    );
    assert!(
        text.contains("term_stride() = 60"),
        "the expected length must be attributed to term_stride(); got: {text}"
    );
    // And the term length is accepted, so the rule is not simply refusing
    // everything.
    check_term_data_len(dims, want).expect("the term-strided payload is accepted");
    eprintln!(
        "MEASURED: term_stride() = {s} and map_stride() = {} differ, so for dims \
         {dims:?} the term rule demands {want} and refuses the grid's {grid_want}",
        dock_core::grid::map_stride()
    );
}
