//! The JSON door into [`Molecule`], exercised from outside the crate.
//!
//! `Molecule` has private fields and no `Default`, so a struct literal and
//! `Molecule::default()` are both closed — and until the `Deserialize` impl was
//! hand-written, `serde_json::from_str::<Molecule>(..)` was a third way to hold
//! a molecule that no constructor had ever checked, which is the same bypass
//! through a file rather than a brace. These tests are the evidence that the
//! door is shut, from the outside, using only `pub` items.
//!
//! What is asserted here, in the order a reader should want it:
//!
//! 1. **The unchanged case.** A molecule this crate produced still
//!    deserializes back to itself, through both constructors and including a
//!    ring, so the fix is not paid for with a broken round trip.
//! 2. **The door.** Every refusal a constructor makes is made by the JSON path
//!    too, with the constructor's own message — the same invariant that
//!    `non_finite_coordinate_refusal` is one definition of, applied to a
//!    second entry point.
//! 3. **What a caller sees.** The error's type, its `classify()`, its
//!    position, and whether it names the offending atom and axis. Measured, not
//!    assumed: `measurement_of_the_json_error_surface` prints the whole surface
//!    and the assertions below pin what it found.
//! 4. **That the inputs are reachable.** A refusal that cannot be reached is
//!    not a refusal. `which_of_these_refusals_are_reachable_through_a_real_document`
//!    measures which of the four constructor refusals a JSON document can
//!    actually reach, and the answer is measured rather than assumed: **neither**
//!    non-finite route gets past `serde_json`'s own parser, so the non-finite
//!    half of the shared rule does its work on the constructors. What the JSON
//!    door *does* close is the bond table and the cached adjacency, which is
//!    where the derived `Deserialize` was dangerous.
//! 5. **The blast radius.** [`ParsedStructure`] derives `Deserialize` and holds
//!    a `Molecule`, so its deserialization changed too.
//!
//! # These tests were mutation-checked, and the mutation is what they are for
//!
//! With `Molecule`'s `deserialize` replaced by one that reads the shape and
//! returns it unvalidated — the derived behaviour this work replaced — **five**
//! of the tests below fail:
//!
//! ```text
//! a_cached_adjacency_in_the_file_is_recomputed_rather_than_believed  FAILED
//! a_molecule_from_json_is_refused_exactly_where_a_constructor_refuses FAILED
//! a_parsed_structure_from_json_is_validated_too                       FAILED
//! an_out_of_range_bond_can_never_reach_the_engine                     FAILED
//! measurement_of_the_json_surface                                      FAILED
//! test result: FAILED. 4 passed; 5 failed
//! ```
//!
//! and the same run measured what the unchecked door actually permitted: the
//! document
//! `{"atoms":[…2…],"bonds":[{"i":0,"j":9,…}],"neighbors":[[9],[0]]}` loaded with
//! no error, `Ligand::from_molecule` on it **panicked** — `index out of bounds:
//! the len is 2 but the index is 9`, at `base[j]` in
//! `Molecule::assign_vina_atom_kinds` — and `Receptor::from_molecule` on the same
//! value *accepted* it, silently discarding a bond that names an atom the
//! structure does not contain.

use dock_core::ligand::Ligand;
use dock_core::pdbqt::ParsedStructure;
use dock_core::types::{Atom, AtomType, Element, Molecule};

/// One atom record, in the shape `Serialize` writes.
fn atom(serial: u32, coord: &str, element: &str, atom_type: &str, kind: &str) -> String {
    format!(
        r#"{{"serial":{serial},"name":"A{serial}","resname":"UNL","resid":1,"coord":{coord},"charge":0.0,"element":"{element}","atom_type":"{atom_type}","kind":"{kind}"}}"#
    )
}

/// Two carbons 1.54 Å apart, one bond, typed as this crate would type them.
fn ethane_atoms() -> String {
    format!(
        "{},{}",
        atom(1, "[0.0,0.0,0.0]", "C", "CH", "Hydrophobic"),
        atom(2, "[1.54,0.0,0.0]", "C", "CH", "Hydrophobic")
    )
}

/// The matching bond record and adjacency, exactly as `from_atoms` stores them.
fn ethane_graph() -> (&'static str, &'static str) {
    (
        r#"[{"i":0,"j":1,"rotatable":false,"in_ring":false}]"#,
        "[[1],[0]]",
    )
}

/// Wrap atoms and a graph into a whole document.
fn document(atoms: &str, bonds: &str, neighbors: &str) -> String {
    format!(r#"{{"atoms":[{atoms}],"bonds":{bonds},"neighbors":{neighbors}}}"#)
}

/// The interaction class of every atom, as an external caller reads it.
fn kinds(mol: &Molecule) -> Vec<dock_core::types::AtomKind> {
    mol.atoms().iter().map(|a| a.kind).collect()
}

/// A benzene ring, for a molecule that is not a chain: its `in_ring` flags are
/// `true` and a chain's are not, so a round trip that loses them is caught.
///
/// The coordinates are hand-written decimals, and that used to be a workaround.
/// `serde_json` is built here **with** its `float_roundtrip` feature (see the
/// workspace `Cargo.toml`), so a 17-digit literal now parses back to the same
/// bits — measured, in `a_full_precision_coordinate_round_trips_bit_exactly` —
/// and these decimals are no longer avoiding a limitation. They are left alone
/// because they are readable and a ring's geometry is obvious by eye, and
/// `a_full_precision_coordinate_round_trips_bit_exactly` is where the precise
/// claim is made, with coordinates that were not chosen to be nice.
fn benzene() -> Molecule {
    let coords = [
        [1.39, 0.0, 0.0],
        [0.695, 1.2035, 0.0],
        [-0.695, 1.2035, 0.0],
        [-1.39, 0.0, 0.0],
        [-0.695, -1.2035, 0.0],
        [0.695, -1.2035, 0.0],
    ];
    let atoms: Vec<Atom> = coords
        .iter()
        .enumerate()
        .map(|(k, c)| Atom::new((k + 1) as u32, *c, Element::C, AtomType::CP))
        .collect();
    let bonds: Vec<(usize, usize)> = (0..6).map(|k| (k, (k + 1) % 6)).collect();
    Molecule::from_bonds(atoms, &bonds).expect("a benzene ring is a valid molecule")
}

/// The atom list of a serialized molecule, re-joined for embedding elsewhere.
fn atom_list(value: &serde_json::Value) -> String {
    value["atoms"]
        .as_array()
        .expect("atoms is an array")
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn a_molecule_written_by_this_crate_deserializes_back_identically() {
    // The unchanged case, and the one that would go red if the checked
    // `Deserialize` recomputed something it should have carried through, or
    // refused a molecule its own writer had produced.
    let ethane = Molecule::from_atoms(vec![
        Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(2, [1.54, 0.0, 0.0], Element::C, AtomType::CH),
    ])
    .expect("ethane perceives one bond");

    // And the caller-graph path, where the bonds were supplied rather than
    // perceived: two carbons far enough apart that the distances would imply
    // no bond at all, so the round trip proves the graph is carried, not
    // re-perceived.
    let sparse = Molecule::from_bonds(
        vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [9.0, 0.0, 0.0], Element::C, AtomType::CH),
        ],
        &[(0, 1)],
    )
    .expect("a caller-supplied bond is not re-judged by distance");

    for original in [ethane, sparse, benzene()] {
        let text = serde_json::to_string(&original).expect("a molecule serializes");
        let back: Molecule = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("a molecule this crate wrote must load: {e}\n{text}"));
        assert_eq!(back, original, "round trip changed the molecule");
        assert_eq!(kinds(&back), kinds(&original));
    }

    // The strong form of the same claim, stated where it matters: a molecule
    // loaded from JSON is one the constructors would have produced.
    let ring = benzene();
    let text = serde_json::to_string(&ring).expect("serializes");
    let from_json: Molecule = serde_json::from_str(&text).expect("loads");
    let rebuilt = Molecule::from_bonds(
        ring.atoms().to_vec(),
        &ring.bonds().iter().map(|b| (b.i, b.j)).collect::<Vec<_>>(),
    )
    .expect("the same molecule from the constructor");
    assert_eq!(from_json, rebuilt);
    assert!(
        ring.bonds().iter().all(|b| b.in_ring),
        "the fixture must actually exercise a ring, or this test proves nothing \
         about `in_ring` surviving"
    );
}

#[test]
fn the_accepted_shape_is_unchanged() {
    // The derive accepted a document with exactly `atoms`, `bonds` and
    // `neighbors`. It still does: nothing was given a default, so a document
    // that omitted a field was refused then and is refused now.
    let text = serde_json::to_string(&benzene()).expect("serializes");
    let value: serde_json::Value = serde_json::from_str(&text).expect("valid json");
    let mut keys: Vec<&str> = value
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["atoms", "bonds", "neighbors"]);

    // Omitting the cached adjacency was an error before this change. Making it
    // optional would widen the accepted language without any gain — the value is
    // recomputed anyway — so the test pins that it is still an error.
    let (bonds, neighbors) = {
        let obj = value.as_object().expect("an object");
        (obj["bonds"].to_string(), obj["neighbors"].to_string())
    };
    let atoms = atom_list(&value);
    let missing = format!(r#"{{"atoms":[{atoms}],"bonds":{bonds}}}"#);
    let err = serde_json::from_str::<Molecule>(&missing)
        .expect_err("omitting `neighbors` was refused before this change and must stay refused");
    assert!(
        err.to_string().contains("neighbors"),
        "the refusal must name the missing field, got: {err}"
    );
    // ...and the complete document still loads.
    assert_eq!(
        serde_json::from_str::<Molecule>(&document(&atoms, &bonds, &neighbors))
            .expect("the complete document loads"),
        benzene()
    );
}

#[test]
fn a_molecule_from_json_is_refused_exactly_where_a_constructor_refuses() {
    // The door. Each case is the JSON spelling of an input a constructor
    // already refuses, and each is compared against the constructor's own
    // message — so the rule is one definition with two callers rather than two
    // rules that happen to agree today.
    let cases: Vec<(&str, String, Vec<Atom>)> = vec![
        ("no atoms", document("", "[]", "[]"), Vec::new()),
        (
            "a bond naming an atom that is not there",
            document(
                &ethane_atoms(),
                r#"[{"i":0,"j":9,"rotatable":false,"in_ring":false}]"#,
                "[[9],[0]]",
            ),
            vec![
                Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
                Atom::new(2, [1.54, 0.0, 0.0], Element::C, AtomType::CH),
            ],
        ),
        (
            "a self-bond",
            document(
                &ethane_atoms(),
                r#"[{"i":1,"j":1,"rotatable":false,"in_ring":false}]"#,
                "[[1],[1]]",
            ),
            vec![
                Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
                Atom::new(2, [1.54, 0.0, 0.0], Element::C, AtomType::CH),
            ],
        ),
    ];

    for (what, json, atoms) in cases {
        let err = serde_json::from_str::<Molecule>(&json)
            .err()
            .unwrap_or_else(|| {
                panic!("{what}: a molecule no constructor would build must be refused")
            });
        let from_constructor = match what {
            "no atoms" => Molecule::from_atoms(atoms.clone()),
            "a bond naming an atom that is not there" => Molecule::from_bonds(atoms, &[(0, 9)]),
            _ => Molecule::from_bonds(atoms, &[(1, 1)]),
        }
        .expect_err("the equivalent in-Rust input is refused");
        assert_eq!(
            err.to_string()
                .split(" at line ")
                .next()
                .unwrap_or(&err.to_string()),
            from_constructor.to_string(),
            "{what}: the JSON door and the constructor must refuse with the same \
             text, so there is one rule and not two that agree today"
        );
    }

    // The non-finite refusal is the *other* half of the shared rule, and it is
    // not reachable through `serde_json` — see
    // `which_of_these_refusals_are_reachable`. What it would do if it ever
    // became reachable is asserted here on the validator itself, so the
    // guarantee is about the rule rather than about one way of feeding it.
    let inf_atoms = vec![Atom::new(
        1,
        [f64::INFINITY, 0.0, 0.0],
        Element::C,
        AtomType::CH,
    )];
    let from_constructor =
        Molecule::from_atoms(inf_atoms).expect_err("an infinite coordinate is refused");
    assert!(
        from_constructor.to_string().contains("non-finite"),
        "and it is the shared refusal: {}",
        from_constructor
    );

    // The duplicate bond stays legal, exactly as `from_bonds` allows: it names
    // one covalent bond twice, so it is redundant rather than wrong.
    let duplicated = document(
        &ethane_atoms(),
        r#"[{"i":0,"j":1,"rotatable":false,"in_ring":false},{"i":1,"j":0,"rotatable":false,"in_ring":false}]"#,
        "[[1,1],[0,0]]",
    );
    let mol = serde_json::from_str::<Molecule>(&duplicated)
        .expect("a bond listed twice is redundant, not wrong");
    assert_eq!(mol.bond_count(), 1, "and it must be stored once");
}

#[test]
fn a_cached_adjacency_in_the_file_is_recomputed_rather_than_believed() {
    // The specific hole the derived impl had: `neighbors`, `in_ring` and
    // `Atom::kind` are *derived*, and a file that states them wrongly is not
    // describing a molecule — it is describing a molecule whose derived state
    // disagrees with the graph that produces it. Nothing downstream reads
    // `neighbors` and then re-derives it: `perceive_rotatable_bonds` walks it
    // to count heavy neighbours, so an empty adjacency on a two-carbon file
    // yields a rigid ligand, a `TORSDOF 1` run over zero degrees of freedom,
    // and a ranked result with nothing in it to say so.
    let (bonds, _) = ethane_graph();
    let lying = document(
        &ethane_atoms(),
        bonds,
        "[[],[]]", // the claim: two unbonded atoms
    );
    let mol: Molecule =
        serde_json::from_str(&lying).expect("the atoms and the bond table are valid");

    // The file's claim is gone.
    assert_eq!(
        mol.neighbors(),
        &[[1], [0]],
        "the adjacency came from the bonds"
    );
    assert_eq!(mol.bond_count(), 1);
    let ligand = Ligand::from_molecule(mol).expect("a consistent molecule prepares");
    assert_eq!(
        ligand.num_torsions(),
        0,
        "ethane has no rotatable bond by this crate's rules, so 0 is the \
         expected derived count here -- what matters is that it was derived"
    );

    // The same molecule with the *file's* adjacency, for contrast — which is
    // what the derived `Deserialize` used to hand the search. It cannot be
    // written here: `neighbors` is a private field, so this file has no way to
    // build that value at all. That is the privacy change doing its job, and it
    // is also why the contrast is asserted from inside the crate instead.
    let mut honest = Molecule::from_bonds(
        vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [1.54, 0.0, 0.0], Element::C, AtomType::CH),
        ],
        &[(0, 1)],
    )
    .expect("valid");
    assert_eq!(honest.neighbors(), &[[1], [0]]);
    honest.assign_ring_membership().expect("no ring to find");
    assert_eq!(honest.bond_count(), 1);

    // A lying `kind` is corrected the same way. `Ligand::from_molecule_with`
    // re-derives kinds itself, so this is about the value a caller holds rather
    // than about the ligand it prepares.
    let lying_kind = document(
        &format!(
            "{},{}",
            atom(1, "[0.0,0.0,0.0]", "O", "OA", "Hydrophobic"),
            atom(2, "[1.43,0.0,0.0]", "C", "CH", "Hydrophobic")
        ),
        r#"[{"i":0,"j":1,"rotatable":false,"in_ring":false}]"#,
        "[[1],[0]]",
    );
    let reloaded: Molecule = serde_json::from_str(&lying_kind).expect("valid atoms and bonds");
    assert_eq!(
        kinds(&reloaded),
        vec![
            dock_core::types::AtomKind::Acceptor,
            dock_core::types::AtomKind::Hydrophobic,
        ],
        "a carbonyl oxygen typed as `Hydrophobic` in the file must come back \
         typed as what it is"
    );
}

#[test]
fn an_out_of_range_bond_can_never_reach_the_engine() {
    // The reason the door mattered, stated as a test: with the derived
    // `Deserialize`, this document produced a molecule whose bond table named
    // atom 9 of a 2-atom structure. `perceive_rotatable_bonds` then reads
    // `neighbors[i]` to count heavy neighbours, so the search walked off the
    // end of the atom table — a panic inside the engine, on a document that had
    // loaded successfully and reported no error at any point.
    let json = document(
        &ethane_atoms(),
        r#"[{"i":0,"j":9,"rotatable":false,"in_ring":false}]"#,
        "[[9],[0]]",
    );
    let err = serde_json::from_str::<Molecule>(&json)
        .expect_err("a bond naming atom 9 of a 2-atom molecule must be refused");
    let message = err.to_string();
    assert!(
        message.contains("atom 9") && message.contains("2 atoms"),
        "the refusal must name the offending index and the atom count, got: {message}"
    );
    // Nothing downstream can be handed this value: the door is the only place
    // the bond table is read, and it has already refused. There is no second
    // entry point to check, which is the point.
}

#[test]
fn a_parsed_structure_from_json_is_validated_too() {
    // The blast radius, named rather than discovered later: `ParsedStructure`
    // derives `Deserialize` and holds a `Molecule`, so its deserialization now
    // inherits the check. That is a behaviour change to a second public type.
    let parsed = serde_json::to_string(&ParsedStructure {
        molecule: benzene(),
        torsions: Vec::new(),
        torsdof: Some(0),
        resolved_torsions: Vec::new(),
        remarks: Vec::new(),
        vina_results: Vec::new(),
        multi_model: false,
        branch_depth: Vec::new(),
    })
    .expect("a parsed structure serializes");
    let back: ParsedStructure = serde_json::from_str(&parsed).expect("and loads");
    assert_eq!(back.molecule, benzene());
    assert_eq!(back.torsdof, Some(0));

    // The same struct with a molecule no constructor would build.
    let broken_molecule = document(
        &ethane_atoms(),
        r#"[{"i":0,"j":9,"rotatable":false,"in_ring":false}]"#,
        "[[9],[0]]",
    );
    let broken = format!(
        r#"{{"molecule":{broken_molecule},"torsions":[],"torsdof":null,"resolved_torsions":[],"remarks":[],"vina_results":[],"multi_model":false,"branch_depth":[]}}"#
    );
    let err = serde_json::from_str::<ParsedStructure>(&broken)
        .expect_err("a bond naming atom 9 of a 2-atom molecule must be refused");
    assert!(
        err.to_string().contains("atom 9"),
        "the refusal must come from the molecule validator, got: {err}"
    );
}

#[test]
fn which_of_these_refusals_are_reachable_through_a_real_document() {
    // A refusal that cannot be reached is not a refusal, and the answer here is
    // not the one the code suggests. Measured on `serde_json` 1.0.151 as this
    // workspace builds it — **with** `float_roundtrip`:
    //
    // * `NaN` as a literal — **not** valid JSON. Refused by serde's own
    //   parser, before the molecule validator is consulted.
    // * `1e400` — **is** valid JSON syntax, and is refused by serde's own
    //   parser as "number out of range".
    //
    // The second of those is the one that was expected to move. Enabling
    // `float_roundtrip` makes an in-range float exact, and it was a reasonable
    // guess that it would also let an overflowing literal through as an
    // infinity — which would have made this refusal reachable for the first
    // time. It does not: the feature changes the precision of the parse, not
    // the range check in front of it. Measured, not assumed, and the guess was
    // wrong in the safe direction — the tripwire below stayed armed instead of
    // having to be rewritten into a new expectation.
    //
    // So a non-finite coordinate still cannot reach this validator through a
    // JSON document at all, and the non-finite half of the shared rule is doing
    // its work on the four constructors rather than here. What *is* reachable —
    // and what the door actually closes — is the bond table and the cached
    // adjacency, which is where the derived `Deserialize` was dangerous: a bond
    // naming atom 9 of a 2-atom molecule panicked the search.
    //
    // These assertions are written to hold in both worlds: if a future
    // `serde_json` does let a non-finite coordinate through, this test panics
    // and says so, because the fixtures and the claims here would then be wrong.
    let overflowing = document(
        &format!(
            "{},{}",
            atom(1, "[1e400,0.0,0.0]", "C", "CH", "Hydrophobic"),
            atom(2, "[1.54,0.0,0.0]", "C", "CH", "Hydrophobic")
        ),
        r#"[{"i":0,"j":1,"rotatable":false,"in_ring":false}]"#,
        "[[1],[0]]",
    );
    let nan_literal = document(
        &format!(
            "{},{}",
            atom(1, "[NaN,0.0,0.0]", "C", "CH", "Hydrophobic"),
            atom(2, "[1.54,0.0,0.0]", "C", "CH", "Hydrophobic")
        ),
        r#"[{"i":0,"j":1,"rotatable":false,"in_ring":false}]"#,
        "[[1],[0]]",
    );
    for (label, json) in [("1e400", &overflowing), ("NaN", &nan_literal)] {
        match serde_json::from_str::<Molecule>(json) {
            Ok(mol) => panic!(
                "measured change: `{label}` now deserializes into {mol:?}. The \
                 non-finite refusal has become reachable through JSON, so the \
                 door tests must add it and this comment is out of date."
            ),
            Err(e) => {
                println!("{label:<6} -> refused by serde_json: {e}");
                assert!(
                    !e.to_string().contains("non-finite"),
                    "measured: `{label}` is refused by serde's own parser, not by \
                     the molecule validator, got: {e}"
                );
            }
        }
    }

    // The two that *are* reachable, asserted with a molecule that prepares, so
    // the contrast is between "loads and docks" and "never loads".
    let good = document(
        &ethane_atoms(),
        r#"[{"i":0,"j":1,"rotatable":false,"in_ring":false}]"#,
        "[[1],[0]]",
    );
    let ligand = Ligand::from_molecule(
        serde_json::from_str::<Molecule>(&good).expect("a consistent molecule loads"),
    )
    .expect("and prepares");
    assert_eq!(
        ligand.num_torsions(),
        0,
        "ethane is rigid by this crate's rules"
    );
}

#[test]
fn measurement_of_the_json_surface() {
    // Prints the whole surface a caller can see, so the figures quoted in the
    // report are measured rather than remembered. Asserts only the two facts
    // that must not drift — the error type and the category — because those are
    // the ones a caller's `match` is written against; the rest is printed so a
    // change in `serde_json`'s formatting shows up as a diff here.
    let json = document(
        &ethane_atoms(),
        r#"[{"i":0,"j":9,"rotatable":false,"in_ring":false}]"#,
        "[[9],[0]]",
    );
    let err = serde_json::from_str::<Molecule>(&json).expect_err("must be refused");
    println!("--- a molecule the constructors would refuse ---");
    println!(
        "type              : {}",
        std::any::type_name::<serde_json::Error>()
    );
    println!("display           : {err}");
    println!("classify          : {:?}", err.classify());
    println!("line / column     : {} / {}", err.line(), err.column());
    println!(
        "downcast DockError: {}",
        <dyn std::any::Any>::downcast_ref::<dock_core::DockError>(&err).is_some()
    );
    // The two facts a caller can branch on.
    assert!(matches!(err.classify(), serde_json::error::Category::Data));
    assert!(
        <dyn std::any::Any>::downcast_ref::<dock_core::DockError>(&err).is_none(),
        "the error is serde's, so a caller cannot recover a `DockError` variant \
         from it -- the message carries the refusal, not the type"
    );

    let shape = document(&ethane_atoms(), "[]", "\"nope\"");
    let shape_err = serde_json::from_str::<Molecule>(&shape).expect_err("must be refused");
    println!("--- a field of the wrong type (a shape error) ---");
    println!("display           : {shape_err}");
    println!("classify          : {:?}", shape_err.classify());
    println!(
        "line / column     : {} / {}",
        shape_err.line(),
        shape_err.column()
    );

    let missing_err = serde_json::from_str::<Molecule>(r#"{"atoms":[],"bonds":[]}"#)
        .expect_err("must be refused");
    println!("--- a missing field (unchanged by this work) ---");
    println!("display           : {missing_err}");
    println!("classify          : {:?}", missing_err.classify());
    println!(
        "line / column     : {} / {}",
        missing_err.line(),
        missing_err.column()
    );

    // The float-parsing limitation the fixtures in this file were written
    // around, and no longer have to be written around.
    let long = -0.974_526_872_786_577_1_f64;
    let text = format!("{long}");
    let back: f64 = serde_json::from_str(&text).unwrap_or(f64::NAN);
    println!("--- float round trip (serde_json 1.0.151, float_roundtrip ON) ---");
    println!("literal           : {text}");
    println!(
        "parsed back       : {back:?}  (bit-equal: {})",
        back == long
    );
    // The one that used to land one ULP away, named so the regression is
    // visible if the feature is ever dropped: the printed pair above is the
    // measurement, and this is the assertion it would fail.
    assert_eq!(
        back, long,
        "float_roundtrip is off: the literal parsed back one ULP away. Either the \
         feature was dropped from the workspace Cargo.toml or serde_json changed \
         its default. A molecule file written by this crate would no longer read \
         back as the same bits."
    );
    println!(
        "overflow literal  : {:?}",
        serde_json::from_str::<f64>("1e400").err()
    );
    println!(
        "NaN literal       : {:?}",
        serde_json::from_str::<f64>("NaN").err()
    );
}

/// What enabling `float_roundtrip` bought, stated as a claim rather than a
/// footnote — and the test that would have been impossible to write before it.
///
/// The whole reason the rest of this file uses hand-rounded decimal literals is
/// that `serde_json`'s default float parser is allowed to land one ULP away from
/// a 17-digit literal. That made this crate's own round trip lossy in a format
/// this crate *owns*: `to_string` then `from_str` did not return the `f64` that
/// went in, and the honest response at the time was a comment apologising for
/// it. `float_roundtrip` is an empty feature in `serde_json` 1.0.151 — no new
/// dependency, no `Cargo.lock` churn — so the loss is gone rather than
/// documented.
///
/// The coordinates below are **not** chosen to be nice. They are what
/// `(k * 60°).to_radians().cos()` actually produces, and they are the case the
/// old comment said could not be asserted on: 17 significant digits, where a
/// 1-ULP error is largest in absolute terms.
#[test]
fn a_full_precision_coordinate_round_trips_bit_exactly() {
    let exact = |k: usize| {
        let theta = (k as f64) * std::f64::consts::PI / 3.0;
        (theta.cos(), theta.sin())
    };

    // First: the literal the old comment named, asserted bit-exactly. This is
    // the specific regression the feature was enabled to prevent.
    let regression = -0.974_526_872_786_577_1_f64;
    let parsed: f64 = serde_json::from_str(&format!("{regression}")).expect("a number");
    assert_eq!(
        parsed.to_bits(),
        regression.to_bits(),
        "the literal the old comment named came back as {parsed:?}, one ULP away"
    );

    // Then: a whole molecule, through the crate's own writer and its checked
    // reader, with coordinates nobody rounded. Bit equality on the *values*,
    // not just on the `PartialEq` the derived one would give.
    let coords: Vec<[f64; 3]> = (0..6)
        .map(|k| {
            let (c, s) = exact(k);
            [c, s, exact(k + 11).0 * 0.35 - 0.2]
        })
        .collect();
    let original = Molecule::from_atoms(
        coords
            .iter()
            .enumerate()
            .map(|(k, c)| Atom::new((k + 1) as u32, *c, Element::C, AtomType::CP))
            .collect(),
    )
    .expect("a ring of full-precision carbons is a valid molecule");

    let text = serde_json::to_string(&original).expect("a molecule serializes");
    let back: Molecule = serde_json::from_str(&text).expect("and loads");
    for (i, (a, b)) in original.atoms().iter().zip(back.atoms()).enumerate() {
        for axis in 0..3 {
            assert_eq!(
                a.coord[axis].to_bits(),
                b.coord[axis].to_bits(),
                "atom {i} axis {axis}: {} -> {}, a ULP apart",
                a.coord[axis],
                b.coord[axis]
            );
        }
    }
    println!(
        "full-precision round trip: 6 atoms x 3 axes, all bit-equal; sample {}",
        text.split("\"coord\":")
            .nth(1)
            .unwrap_or("?")
            .split(']')
            .next()
            .unwrap_or("?")
    );
}
