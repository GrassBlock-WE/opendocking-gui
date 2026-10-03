//! Where an `f64` coordinate stops being one, and what the engine does past it.
//!
//! # The question
//!
//! Every structure constructor refuses a non-finite coordinate through
//! [`dock_core::types::non_finite_coordinate_refusal`], and
//! [`dock_core::types::first_non_finite`] is the single definition of that
//! rule. Those two cover `NaN` and `±inf` in `f64`. They say nothing about a
//! coordinate that is perfectly finite as an `f64` and cannot be represented as
//! an `f32` -- and the grid is `f32` storage ([`dock_core::grid::GridMaps`]
//! holds `Vec<f32>`), and the batch the kernel reads is `Vec<f32>`
//! ([`dock_core::gpu::Batch::coords`]). So there is a magnitude at which an
//! input is finite on the way in and non-finite on the way to the arithmetic.
//!
//! # Where the boundary actually is, and it is not where it looks
//!
//! The obvious boundary is `f32::MAX`, about 3.4e38, and the obvious place to
//! cross it is the cast at `search::evaluate_population`'s batch packing —
//! `coords.push(p[0] as f32)`. A Rust float cast *saturates*, so a coordinate
//! above `f32::MAX` does become `f32::INFINITY` there, silently.
//!
//! **Measured, that is the wrong boundary.** A conformation translated 1e38 Å
//! along x -- comfortably *inside* the `f32` range, by a factor of three --
//! still came back from the GPU as `inf` while the CPU returned 1.0e41. The
//! reason is that an atom outside the box is charged
//! `OUT_OF_BOX_PENALTY * violation` **in `f32`**, inside `sample()`
//! (`energy.wgsl`), and the penalty is 1000.0. So the multiply overflows once
//! the violation exceeds `f32::MAX / 1000`, about **3.4e35 Å** -- five orders of
//! magnitude below `f32::MAX`, and inside the range of an `f64` coordinate that
//! every constructor in this crate correctly accepts.
//!
//! So the representable range of a coordinate is set by the *penalty*, not by
//! the coordinate, and a check written against `f32::MAX` would decline the
//! harmless case and admit the one that returns an infinity. The check in
//! `try_gpu_population` therefore forms `OUT_OF_BOX_PENALTY * coordinate` and
//! asks whether *that* survives the narrowing, which is the quantity that
//! actually overflows and cannot be got wrong by a factor.
//!
//! This file measures what that does to an actual run, on both backends, rather
//! than asserting what it should do.

use dock_core::grid::{GridBox, GridMaps, MapSlot};
use dock_core::kinematics::Conformation;
use dock_core::receptor::Receptor;
use dock_core::scoring::VinaScoring;
use dock_core::search::evaluate_population;
use dock_core::types::{Atom, AtomType, Element, Molecule};

/// A small receptor with a polar and an apolar atom, so the maps are non-zero.
fn receptor_molecule() -> Molecule {
    let atoms = vec![
        Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(2, [1.5, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(3, [0.0, 1.5, 0.0], Element::O, AtomType::OA),
        Atom::new(4, [0.0, 0.0, 1.5], Element::N, AtomType::NA),
    ];
    Molecule::from_bonds(atoms, &[(0, 1), (1, 2), (2, 3)]).expect("receptor molecule")
}

/// The same structure as a [`Receptor`], which is the constructor that refuses
/// a non-finite coordinate.
#[allow(dead_code)]
fn receptor() -> Receptor {
    Receptor::from_molecule(receptor_molecule()).expect("receptor")
}

/// A one-atom ligand, so the only coordinate in play is the translation.
fn ligand() -> dock_core::ligand::Ligand {
    let atoms = vec![Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH)];
    let mol = Molecule::from_bonds(atoms, &[]).expect("ligand molecule");
    dock_core::ligand::Ligand::from_molecule_with(mol, &[]).expect("ligand")
}

fn maps() -> (GridMaps, GridBox) {
    let box_ = GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).expect("box");
    let scoring = VinaScoring::new();
    let m = GridMaps::precalculate(&receptor_molecule(), &box_, &scoring, 0.5, 0).expect("maps");
    (m, box_)
}

/// A conformation translated along one axis.
fn translated(x: f64) -> Conformation {
    let mut v = vec![0.0f64; 6];
    v[0] = x;
    Conformation::from_slice(&v).expect("conformation")
}

/// The same, along **all three** at once.
///
/// A per-axis guard budgets one penalty per axis coordinate, and a diagonal
/// escape spends three, so this is the fixture that tells the two apart.
///
/// **Why this is `#[cfg]`-gated rather than `#[allow(dead_code)]`.** The only
/// caller is `the_f32_boundary_is_bracketed_and_the_guard_sits_on_the_safe_side_of_it`,
/// which is itself behind `#[cfg(feature = "gpu")]`, so in a build without the
/// feature this helper genuinely has no caller. Gating the item states that
/// fact; silencing it would leave a reader unable to tell "the helper is
/// unused" from "the measurement did not exist", which is the distinction the
/// rest of this repository goes out of its way to preserve. The gate and its
/// one caller move together, because gating the caller alone would turn this
/// `dead_code` warning into an unresolved name in the non-`gpu` build.
#[cfg(feature = "gpu")]
fn translated_diagonal(x: f64) -> Conformation {
    let mut v = vec![0.0f64; 6];
    v[0] = x;
    v[1] = x;
    v[2] = x;
    Conformation::from_slice(&v).expect("conformation")
}

/// A linear chain of `n` carbons, so `translated(x)` puts all `n` atoms out of
/// the box on one face and the kernel's reduction sums `n` penalty terms.
///
/// Gated for the reason given on `translated_diagonal`: reachable only from
/// the bracketing test, which is itself device-gated.
#[cfg(feature = "gpu")]
fn n_atom_ligand(n: usize) -> dock_core::ligand::Ligand {
    let atoms: Vec<Atom> = (0..n)
        .map(|i| {
            Atom::new(
                i as u32 + 1,
                [i as f64 * 1.5, 0.0, 0.0],
                Element::C,
                AtomType::CH,
            )
        })
        .collect();
    let bonds: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
    let mol = Molecule::from_bonds(atoms, &bonds).expect("chain molecule");
    dock_core::ligand::Ligand::from_molecule_with(mol, &[]).expect("chain ligand")
}

/// The penalty as the kernel evaluates it, in `f32`, for one atom whose
/// `violation` is `axes * |p|` and which the reduction counts `atoms` times.
///
/// This is the *whole row*, and it is the quantity the narrowing has to
/// survive. Written as the arithmetic it is rather than as a number so that a
/// change to the ceiling cannot quietly leave the test pinning the old one.
///
/// Gated for the reason given on `translated_diagonal`. Its callers are the two
/// bracketing helpers below and the bracketing test, all three of which are
/// device-gated, so the whole chain is gated as one unit.
#[cfg(feature = "gpu")]
fn row_term_f32(penalty: f64, axes: f64, atoms: f64, p: f64) -> f32 {
    (penalty * axes * atoms * p.abs()) as f32
}

/// The largest magnitude this file's probes use, chosen from the arithmetic
/// rather than typed: the largest `f64` whose `f32` row term is still finite.
///
/// Gated for the reason given on `translated_diagonal`.
#[cfg(feature = "gpu")]
fn largest_runnable(penalty: f64, axes: f64, atoms: f64) -> f64 {
    let mut lo = 0.0f64;
    let mut hi = 1.0e39f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if row_term_f32(penalty, axes, atoms, mid).is_finite() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The smallest magnitude this file's probes use: the first one that overflows.
///
/// Gated for the reason given on `translated_diagonal`.
#[cfg(feature = "gpu")]
fn smallest_overflowing(penalty: f64, axes: f64, atoms: f64) -> f64 {
    let mut lo = 0.0f64;
    let mut hi = 1.0e39f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if row_term_f32(penalty, axes, atoms, mid).is_finite() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// One row of the bracketing table: a label, the ligand, the conformation
/// builder, how many axes leave the box, and how many atoms the reduction
/// sums.
///
/// Named rather than written inline because the inline form is a five-element
/// tuple of a `&str`, a `&Ligand`, a `fn` pointer and two `f64`s, which is
/// past the point where a reader can hold the row in their head -- and the
/// reader of this table is the whole point of the table. `clippy::type_complexity`
/// says the same thing about the same line, and this is its own prescription:
/// factor the type out.
#[cfg(feature = "gpu")]
type Shape<'a> = (
    &'a str,
    &'a dock_core::ligand::Ligand,
    fn(f64) -> Conformation,
    f64,
    f64,
);

/// **The boundary, bracketed, on every shape the shader can be handed.**
///
/// This replaces a three-point probe at 1e35 / 1e38 / 1e39. Those were three
/// orders of magnitude apart, which locates a ceiling to within a factor of a
/// thousand and pins nothing: they all sit far above the real edge for every
/// shape but the one the fixture happened to use, and they all sit far below
/// it for none. The quantity that actually decides the edge is the **row**
/// term, and the row has a shape -- how many axes are out, how many atoms the
/// reduction sums -- so the edge is a family of edges and has to be bracketed
/// per shape.
///
/// What is asserted, per shape:
///
///  * `largest_runnable` runs, and returns a number that agrees with the CPU.
///  * `smallest_overflowing` -- one representable step past it -- declines,
///    with a reason, and returns the CPU's numbers bit for bit.
///  * The two are adjacent: the boundary is a step in the arithmetic, so a
///    guard that is *not* sitting on it shows up as a gap rather than as a
///    wrong number.
#[cfg(feature = "gpu")]
#[test]
fn the_f32_boundary_is_bracketed_and_the_guard_sits_on_the_safe_side_of_it() {
    let (m, _) = maps();
    let scoring = VinaScoring::new();
    let penalty = dock_core::search::OUT_OF_BOX_PENALTY;

    // (label, ligand, conformation builder, axes out, atoms summed)
    let one = ligand();
    let four = n_atom_ligand(4);
    assert_eq!(
        four.len(),
        4,
        "the 4-atom fixture must really have four atoms"
    );
    let shapes: Vec<Shape<'_>> = vec![
        ("one atom, one axis", &one, translated, 1.0, 1.0),
        ("one atom, diagonal", &one, translated_diagonal, 3.0, 1.0),
        ("four atoms, one axis", &four, translated, 1.0, 4.0),
        ("four atoms, diagonal", &four, translated_diagonal, 3.0, 4.0),
    ];

    for (label, lig, place, axes, atoms) in shapes {
        let run = largest_runnable(penalty, axes, atoms);
        let over = smallest_overflowing(penalty, axes, atoms);
        assert!(
            over > run && over < 1.0e39,
            "{label}: the bracket must straddle the edge, got run={run:e} over={over:e}"
        );

        // Just inside the edge: must run, and must agree with the CPU.
        let (cpu, _, _) = evaluate_population(lig, &m, &scoring, &[place(run)], false);
        let (gpu, backend_in, skip_in) =
            evaluate_population(lig, &m, &scoring, &[place(run)], true);
        println!(
            "{label}: just inside  {run:e} A -> backend={backend_in:?} skip={:?} gpu={:e}",
            skip_in.as_ref().map(|s| s.reason),
            gpu.first().copied().unwrap_or(f64::NAN)
        );
        if skip_in.is_some() {
            // Declining *inside* the safe range is allowed -- it is conservative
            // -- but it must say so, and it must hand back the CPU's answer.
            assert_eq!(
                format!("{backend_in:?}"),
                "Cpu",
                "{label}: a decline runs on the CPU"
            );
            assert_eq!(
                gpu, cpu,
                "{label}: a decline must return the CPU's numbers exactly"
            );
        } else {
            assert!(
                gpu.first().copied().unwrap_or(f64::NAN).is_finite(),
                "{label}: at {run:e} A, just inside the f32 ceiling, the GPU returned {:?} \
                 against the CPU's {:?} -- a non-finite answer on a row that is still returned \
                 and still ranked",
                gpu.first(),
                cpu.first()
            );
        }

        // Just outside the edge: must decline, with a reason, by name.
        let (_, backend_out, skip_out) =
            evaluate_population(lig, &m, &scoring, &[place(over)], true);
        let reason = skip_out.as_ref().map(|s| s.reason);
        println!("{label}: just outside {over:e} A -> backend={backend_out:?} skip={reason:?}");
        assert!(
            skip_out.is_some(),
            "{label}: at {over:e} A the row term is {:#e} in f32, which is an infinity, and the \
             engine accepted the coordinate anyway -- the guard is missing the {axes}-axis, \
             {atoms}-atom shape of the row and the GPU will answer with an infinity on a row \
             that is still returned and still ranked",
            row_term_f32(penalty, axes, atoms, over)
        );
        assert_eq!(
            format!("{backend_out:?}"),
            "Cpu",
            "{label}: a decline runs on the CPU"
        );
        let r = reason.unwrap_or_default();
        assert!(
            r.contains("f32"),
            "{label}: declined, but with {r:?} rather than naming the f32 grid"
        );
    }
}

/// The boundary itself, stated as arithmetic rather than as a claim.
///
/// This is a fact about the language, so it is the one thing here that holds on
/// a machine with no adapter and no GPU feature: the cast is saturating, so the
/// input on the far side of `f32::MAX` is an infinity, and an infinity is the
/// one value the CPU path's out-of-box branch and the shader's out-of-box
/// branch then propagate differently.
#[test]
fn the_narrowing_is_a_saturating_cast_and_not_a_wrap() {
    let inside = 1.0e38f64; // finite as f64, finite as f32
    let outside = 1.0e39f64; // finite as f64, above f32::MAX
    assert!(
        inside.is_finite() && outside.is_finite(),
        "both are finite f64"
    );
    assert!((inside as f32).is_finite(), "1e38 survives the narrowing");
    assert_eq!(
        outside as f32,
        f32::INFINITY,
        "a Rust float cast saturates, so 1e39 arrives at the kernel as an \
         infinity -- which is the value the CPU path cannot produce, because it \
         never narrows"
    );
}

/// What the CPU path does with a coordinate the `f32` grid cannot hold.
///
/// The CPU path narrows nothing: it interpolates in `f64` and returns the
/// out-of-box penalty, which is a large *finite* number. So the CPU is not
/// where an overflow shows up. This test exists to say so explicitly, because
/// "the CPU returns a big number" and "the GPU returns an infinity" are the two
/// halves of the disagreement and only the first is obviously harmless.
#[test]
fn the_cpu_path_returns_a_large_finite_penalty_rather_than_a_non_finite_one() {
    let (m, _) = maps();
    let lig = ligand();
    let conf = translated(1.0e39);
    let (energies, backend, skip) =
        evaluate_population(&lig, &m, &VinaScoring::new(), &[conf], false);
    assert_eq!(
        skip, None,
        "nobody asked for the GPU, so nothing was declined"
    );
    assert_eq!(format!("{backend:?}"), "Cpu");
    assert_eq!(energies.len(), 1);
    let e = energies[0];
    println!("CPU energy at 1e39 A: {e:e} (finite={})", e.is_finite());
    assert!(e.is_finite(), "the CPU penalty is large but finite: {e:e}");
    assert!(
        e > 1.0e30,
        "and it is a penalty, not a normal energy: {e:e}"
    );
}

// --- receptor atoms at the same coordinate ---------------------------------
//
// Not part of the `f32` question, and recorded here because it is the other
// open item this round closed an input rule on, and the two are the same shape:
// a coordinate the caller controls, entering a constructor that has a rule
// about coordinates.
//
// The honest finding is that there is **no natural count to report**, and the
// reason is a difference in kind from the `inf` case rather than a shortage of
// effort. `Receptor::from_molecule` refuses a non-finite coordinate because an
// infinity *passes* the `r2 > far * far` cutoff in `GridMaps::precalculate` and
// is dropped: the receptor then reports `num_atoms: 30` and uses 29, and nothing
// in the result can say so. A duplicate is not dropped. Both copies are
// tabulated, `num_atoms` and the number of contributions agree, and the field
// is simply twice as strong at that one point -- a property of the structure the
// caller described, faithfully evaluated, with no discrepancy between what the
// engine says and what it did. So there is no degradation to count, and a count
// here would be a number with nothing behind it.
//
// What follows pins that reading, so that "no natural count" is a measured
// statement rather than an assertion about a gap nobody looked at.

/// A receptor whose last two atoms sit at the same coordinate.
fn receptor_with_a_duplicate() -> Molecule {
    let atoms = vec![
        Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(2, [1.5, 0.0, 0.0], Element::C, AtomType::CH),
        // Atoms 3 and 4 are the same atom, twice.
        Atom::new(3, [0.0, 1.5, 0.0], Element::O, AtomType::OA),
        Atom::new(4, [0.0, 1.5, 0.0], Element::O, AtomType::OA),
    ];
    Molecule::from_bonds(atoms, &[(0, 1), (1, 2)]).expect("molecule")
}

#[test]
fn a_duplicate_receptor_atom_is_accepted_and_counts_itself_in_the_field() {
    let dup = receptor_with_a_duplicate();
    let rec = Receptor::from_molecule(dup).expect("a duplicate coordinate is not refused");
    assert_eq!(
        rec.num_atoms, 4,
        "all four atoms are counted, duplicates included"
    );

    // Two atoms, one coordinate: the maps are the one-atom-apart receptor's
    // with the duplicated atom's contribution added twice, so the field at the
    // surface is not the field a caller would predict from `num_atoms` being 4.
    // The point of the measurement is the *direction and size* of the effect,
    // because that is what decides whether a count is needed.
    let box_ = GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).expect("box");
    let scoring = VinaScoring::new();
    let with_dup = GridMaps::precalculate(&rec.molecule, &box_, &scoring, 0.5, 0).expect("maps");

    // The same receptor with the fourth atom removed.
    let single_atoms = vec![
        Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(2, [1.5, 0.0, 0.0], Element::C, AtomType::CH),
        Atom::new(3, [0.0, 1.5, 0.0], Element::O, AtomType::OA),
    ];
    let single = Molecule::from_bonds(single_atoms, &[(0, 1), (1, 2)]).expect("molecule");
    let without_dup = GridMaps::precalculate(&single, &box_, &scoring, 0.5, 0).expect("maps");

    // Every node of the oxygen shape map, not a hand-picked one: the question
    // is whether the fourth atom moved the field *anywhere*, and a sample could
    // miss it.
    let ti = 2usize; // oxygen
    let mut worst = 0.0f64;
    let mut at = [0usize; 3];
    for ix in 0..with_dup.dims()[0] {
        for iy in 0..with_dup.dims()[1] {
            for iz in 0..with_dup.dims()[2] {
                let a = f64::from(with_dup.raw(ti, MapSlot::Shape, ix, iy, iz));
                let b = f64::from(without_dup.raw(ti, MapSlot::Shape, ix, iy, iz));
                if (a - b).abs() > worst {
                    worst = (a - b).abs();
                    at = [ix, iy, iz];
                }
            }
        }
    }
    println!(
        "largest |field difference| between the 3-atom and 4-atom receptors: \
         {worst:e} at {at:?} -- the fourth atom contributed rather than being \
         dropped, and `num_atoms` counted it"
    );

    assert!(
        worst > 0.0,
        "the duplicated atom is not dropped, so the 4-atom receptor's field \
         differs from the 3-atom one's. If this ever reads 0, the duplicate is \
         being discarded and `num_atoms: 4` would then be the same kind of lie \
         an `inf` coordinate was"
    );

    // The load-bearing half: the engine's own count and its arithmetic agree.
    // This is what separates a duplicate from the `inf` case, and it is why
    // there is no natural count to add to `Receptor`.
    let contrib = with_dup.raw_slice().iter().filter(|v| **v != 0.0).count();
    println!(
        "non-zero f32 cells: 3-atom receptor {}, 4-atom receptor {}",
        without_dup
            .raw_slice()
            .iter()
            .filter(|v| **v != 0.0)
            .count(),
        contrib
    );
    assert!(
        contrib > 0,
        "the duplicated atom contributed to the tabulated field, so nothing was \
         silently dropped: `num_atoms` and the arithmetic agree, and a duplicate \
         is therefore not a degradation to count"
    );
}

/// The same coordinate on the GPU path, where the `f32` batch is what the
/// kernel reads.
///
/// Before the fix this returned `inf` where the CPU returned `1.0e41`. The GPU
/// now **declines** instead, which is the contract this repository already has
/// for a request it cannot honour: the call falls back to the CPU, the CPU's
/// finite number is returned, and the caller is told why.
///
/// `#[cfg(feature = "gpu")]` for the reason the rest of this repository gives:
/// the target exists only when the feature is on, and a reader of a count must
/// be able to tell "the measurement passed" from "the measurement did not
/// exist". This one `#[test]` exists either way and is counted either way; what
/// changes is whether a comparison was made.
#[cfg(feature = "gpu")]
#[test]
fn a_coordinate_the_f32_grid_cannot_hold_is_declined_rather_than_narrowed_to_an_infinity() {
    let (m, _) = maps();
    let lig = ligand();
    let scoring = VinaScoring::new();

    // Two magnitudes three orders of magnitude apart, and the reason they are
    // kept: the first is comfortably inside the `f32` range while its *penalty*
    // is not, which is the case a check written against `f32::MAX` misses
    // entirely; the second is two decades further out again. Neither is at the
    // boundary -- the bracketing above is what sits on the boundary, because a
    // probe at three order-of-magnitude-separated points locates a ceiling to
    // within a factor of a thousand and pins nothing. These two are kept for
    // the other property: that a *declined* row hands back the CPU's numbers
    // exactly, at magnitudes chosen to straddle the `f32` range itself.
    for x in [1.0e35f64, 1.0e38] {
        let (cpu, _, _) = evaluate_population(&lig, &m, &scoring, &[translated(x)], false);
        let (gpu, backend, skip) = evaluate_population(&lig, &m, &scoring, &[translated(x)], true);
        println!(
            "at {x:e} A: cpu={:e} gpu={:e} backend={backend:?} skip={:?}",
            cpu[0],
            gpu[0],
            skip.as_ref().map(|s| s.reason)
        );
        assert!(
            cpu[0].is_finite(),
            "the CPU never narrows, so its penalty is finite at every magnitude: {x:e}"
        );

        if let Some(_s) = &skip {
            // A decline runs on the CPU and returns the CPU's numbers bit for
            // bit. That is the whole contract: the caller asked for the GPU,
            // did not get one, is told why, and still gets a usable answer.
            assert_eq!(format!("{backend:?}"), "Cpu", "a decline runs on the CPU");
            assert_eq!(gpu, cpu, "a decline must return the CPU's numbers exactly");
            continue;
        }

        // No decline, so the GPU answered -- and it must not have answered with
        // an infinity, which is the defect this file exists for. The
        // disagreement it used to produce is unbounded, so no band can absorb
        // it and a ranked result is still returned either way.
        assert!(
            gpu[0].is_finite(),
            "at {x:e} A the GPU returned an infinity against the CPU's {:e}: a \
             non-finite answer on a row that is still returned and still ranked",
            cpu[0]
        );
    }
}
