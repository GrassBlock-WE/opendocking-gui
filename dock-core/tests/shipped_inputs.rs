//! Every structure file this repository ships, run against the engine's refusals.
//!
//! # Why this file exists
//!
//! The declared-`TORSDOF` guard in [`Ligand::from_parsed`] was written, and its
//! unit tests in `ligand.rs` prove it fires on a synthetic three-carbon file.
//! What no test proved was the other direction: that the guard does not fire on
//! a file the project *ships*. Those are different claims, and only the second
//! one is about whether the repository's own examples work.
//!
//! A fixture written **for** the guard is expected to be refused — that is what
//! it is for, and `ligand.rs` owns those. A fixture written to be **docked** is
//! not: a shipped input the engine refuses is a shipping defect, because the
//! documented commands over `examples/` stop working with no version of the
//! file's contents to blame. So this file draws the line where the crate's own
//! comments draw it, and asserts the second half.
//!
//! # The control that makes the measurement mean something
//!
//! "Zero refusals" is the answer this file is expected to produce, and a zero
//! from a gate that never fired is not an answer. So
//! `the_torsdof_guard_is_reachable_through_a_shipped_ligand` takes a shipped
//! ligand, asserts it is accepted, changes **one number in the parsed value** —
//! no fixture file, no edited `examples/`, and the identical code path — and
//! asserts the same call is then refused with both numbers named. Without that
//! test, a guard deleted outright would leave this file green.
//!
//! # What is run, and against which entry point
//!
//! * [`Ligand::from_parsed`] — the one entry point that owns the guard. A file
//!   that declares no `TORSDOF` cannot disagree with itself and is therefore
//!   outside the guard by construction; the table records which files that is,
//!   rather than letting "accepted" imply the guard fired.
//! * [`Receptor::from_molecule`] — the other shipped-input entry point, so a
//!   fixture that is a receptor is exercised the way a caller uses it rather
//!   than being reported as a ligand.
//!
//! Files that the engine has no reader for are enumerated too, and counted as
//! such. `examples/*.sdf` is RDKit input for the Python front-end and no
//! `.pdbqt` reader is supposed to accept it; a silent skip would leave the
//! enumeration looking complete while covering less than it appears to.
//!
//! Files a *gate generates* are a third thing, and are listed in
//! [`GENERATED_UNDER_EXAMPLES`] rather than swept. "A gate reads this file" and
//! "the repository ships this file" are different claims: the first is true of
//! a scratch file a gate writes and then loads, and it does not make the file an
//! input. That distinction is stated here, at the sweep, because getting it
//! backwards is not a wrong count — it is a rule that protects a scratch file
//! from cleanup and a reader who concludes the file is a legitimate input.

use std::path::{Path, PathBuf};

use dock_core::ligand::Ligand;
use dock_core::pdbqt::read_pdbqt;
use dock_core::receptor::Receptor;

/// The repository's `examples/` directory, relative to this crate's manifest.
///
/// `CARGO_MANIFEST_DIR` is `dock-core/`, and CI checks the whole repository out
/// before running `cargo test --workspace` from the root, so the parent
/// directory is the shipped tree rather than a build artifact.
fn examples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("examples")
        .canonicalize()
        .unwrap_or_else(|e| panic!("examples/ must exist and be readable: {e}"))
}

/// Every file under `dir`, recursively, sorted, so the enumeration is stable.
fn all_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| {
            e.unwrap_or_else(|err| panic!("a directory entry of {}: {err}", dir.display()))
                .path()
        })
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            all_files(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// The extension, lowercased, or `""`.
fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Path relative to `examples/`, with `/` separators so the failure text is the
/// same on every platform.
fn relative(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// One shipped structure file, and what the engine did with it.
struct Verdict {
    rel: String,
    atoms: usize,
    torsdof: Option<usize>,
    torsion_records: usize,
    ligand: Result<usize, String>,
    receptor: Result<usize, String>,
}

/// Structure files that live under `examples/` but are **generated**, as
/// `(path relative to examples/, the gate that writes it)`.
///
/// The distinction this table exists to make is the one
/// `scripts/release_tree_parity_check.py` had backwards. That file asserted
/// that `examples/_shifted_receptor.pdbqt` was a shipped *input* because
/// `scripts/workbench_interaction_check.py` reads it — but that gate is the only
/// thing that ever wrote it, from `examples/rec_prep.pdbqt` with every x shifted
/// by -40, and it regenerates the file byte for byte. "A gate reads this file"
/// and "the repository ships this file" are different claims, and only the
/// second one makes a file a member of this sweep.
///
/// The generator now writes to `target/workbench_interaction/`, which is
/// outside `examples/` and already declared non-source. This entry is therefore
/// a **tolerance for a copy left behind by a run from before that change**, and
/// it is deliberately not a licence for the sweep to grow: the path is named
/// here, a copy that is present is named in the report below, and the file is
/// never parsed. Nothing consults this table to reach a count — the 17 below is
/// `paths.len()`, and a classified file never enters `paths` — so the table
/// cannot be widened to make a number pass. Delete the leftover copy and this
/// table goes with it.
const GENERATED_UNDER_EXAMPLES: &[(&str, &str)] = &[
    (
        "_shifted_receptor.pdbqt",
        "scripts/workbench_interaction_check.py section 9, derived from rec_prep.pdbqt",
    ),
];

/// Read every shipped structure file and record what the engine did with it.
fn sweep() -> (PathBuf, Vec<Verdict>, Vec<(String, String)>, Vec<(String, String)>) {
    let root = examples_dir();
    let mut files = Vec::new();
    all_files(&root, &mut files);

    let mut paths: Vec<PathBuf> = Vec::new();
    let mut unreadable_by_design: Vec<(String, String)> = Vec::new();
    let mut generated: Vec<(String, String)> = Vec::new();
    for file in &files {
        let ext = extension(file);
        let rel = relative(file, &root);
        if let Some((_, why)) = GENERATED_UNDER_EXAMPLES
            .iter()
            .find(|(p, _)| *p == rel.as_str())
        {
            // Classified, not swept: a derived artefact says nothing about
            // whether the crate's readers accept the tree, so it is recorded
            // and skipped rather than read and counted.
            generated.push((rel.clone(), (*why).to_string()));
            continue;
        }
        match ext.as_str() {
            "pdbqt" | "pdb" => paths.push(file.clone()),
            "sdf" => unreadable_by_design.push((rel, "sdf: RDKit front-end input".to_string())),
            "py" => unreadable_by_design.push((rel, "py: a diagnostic script".to_string())),
            _other => unreadable_by_design.push((rel, format!("{ext:?}: no reader in this crate"))),
        }
    }

    let mut verdicts = Vec::new();
    for file in &paths {
        let rel = relative(file, &root);
        let parsed = match read_pdbqt(file) {
            Ok(p) => p,
            Err(e) => {
                verdicts.push(Verdict {
                    rel: rel.clone(),
                    atoms: 0,
                    torsdof: None,
                    torsion_records: 0,
                    ligand: Err(format!("read refused: {e}")),
                    receptor: Err(format!("read refused: {e}")),
                });
                continue;
            }
        };
        let atoms = parsed.molecule.len();
        let torsdof = parsed.torsdof;
        let torsion_records = parsed.torsions.len();
        let ligand = Ligand::from_parsed(parsed.clone())
            .map(|l| l.num_torsions())
            .map_err(|e| e.to_string());
        let receptor = Receptor::from_molecule(parsed.molecule.clone())
            .map(|r| r.num_atoms)
            .map_err(|e| e.to_string());
        verdicts.push(Verdict {
            rel,
            atoms,
            torsdof,
            torsion_records,
            ligand,
            receptor,
        });
    }
    (root, verdicts, unreadable_by_design, generated)
}

/// The first line of a message, trimmed for a table cell.
fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.chars().count() > 104 {
        let mut out: String = line.chars().take(101).collect();
        out.push_str("...");
        out
    } else {
        line.to_string()
    }
}

#[test]
fn every_shipped_structure_file_is_read_and_accepted() {
    let (root, verdicts, unreadable_by_design, generated) = sweep();

    // The enumeration is asserted before any verdict, because a sweep that
    // silently found nothing would pass every check below — and this file's
    // expected answer is "nothing was refused".
    assert_eq!(
        verdicts.len(),
        17,
        "the shipped structure-file count changed; the table below is the \
         enumeration this run measured, and a new or moved fixture has to be \
         classified here rather than joining the sweep silently"
    );
    assert_eq!(
        unreadable_by_design.len(),
        12,
        "the count of shipped files with no reader in this crate changed: {:?}",
        unreadable_by_design
    );

    let mut table = Vec::new();
    let mut declared: Vec<&Verdict> = Vec::new();
    let mut refused: Vec<&Verdict> = Vec::new();
    for v in &verdicts {
        if v.torsdof.is_some() {
            declared.push(v);
        }
        if v.ligand.is_err() || v.receptor.is_err() {
            refused.push(v);
        }
        let kind = if v.torsdof.is_some() {
            "declares TORSDOF"
        } else {
            "no flexibility declared"
        };
        let ligand_text = match &v.ligand {
            Ok(n) => format!("ligand accepted, {n} torsion(s)"),
            Err(e) => format!("ligand REFUSED: {}", first_line(e)),
        };
        let receptor_text = match &v.receptor {
            Ok(n) => format!("receptor accepted ({n} atoms)"),
            Err(e) => format!("receptor REFUSED: {}", first_line(e)),
        };
        table.push(format!(
            "{:<28} atoms={:<5} torsdof={:<9} torsion_records={:<3} [{kind:<21}] \
             {ligand_text} | {receptor_text}",
            v.rel,
            v.atoms,
            format!("{:?}", v.torsdof),
            v.torsion_records,
        ));
    }

    // The claim, stated as an assertion rather than read out of a table: every
    // file that declares `TORSDOF n` is accepted, so the engine's derived
    // flexibility equals the declared flexibility for every shipped input.
    assert!(
        declared.len() >= 12,
        "the sweep must actually reach the guard: {} of {} shipped files declare \
         TORSDOF, and a guard that fires on nothing is a guard that cannot be \
         distinguished from an absent one",
        declared.len(),
        verdicts.len()
    );
    for v in &declared {
        let n = v.torsdof.expect("filtered above");
        assert_eq!(
            v.ligand.as_ref().copied().ok(),
            Some(n),
            "{}: the file declares TORSDOF {n} and the engine derived {} -- this \
             is a shipped fixture written to be docked, and a refusal here is a \
             shipping defect, not a guard fixture doing its job",
            v.rel,
            v.ligand
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_else(|e| e.clone())
        );
    }

    // And the weaker claim, over every shipped structure file: nothing is
    // refused at all, by either entry point.
    assert!(
        refused.is_empty(),
        "shipped inputs refused by the engine:\n{}\n--- full table ---\n{}",
        refused
            .iter()
            .map(|v| format!("{}: {:?} / {:?}", v.rel, v.ligand, v.receptor))
            .collect::<Vec<_>>()
            .join("\n"),
        table.join("\n")
    );

    let report = format!(
        "\nexamples/: {root:?}\nshipped structure files: {}\nshipped files with no reader in this \
         crate: {}\nshipped files declaring TORSDOF: {}\nrefused by the engine: {}\ngenerated files \
         found under examples/ and classified rather than swept: {}\n\n{}\n\nnot read here: \
         {}\ngenerated, if any: {}\n",
        verdicts.len(),
        unreadable_by_design.len(),
        declared.len(),
        refused.len(),
        generated.len(),
        table.join("\n"),
        unreadable_by_design
            .iter()
            .map(|(p, why)| format!("{p} ({why})"))
            .collect::<Vec<_>>()
            .join(", "),
        if generated.is_empty() {
            "none — the classification is empty and can be deleted".to_string()
        } else {
            generated
                .iter()
                .map(|(p, why)| format!("{p} ({why})"))
                .collect::<Vec<_>>()
                .join(", ")
        },
    );
    // Written to a file as well as stdout: this table is the evidence, and a
    // `--nocapture` run is the only way to see it.
    let _ = std::fs::write(
        std::env::temp_dir().join("shipped_inputs_sweep.txt"),
        &report,
    );
    println!("{report}");
}

#[test]
fn the_torsdof_guard_is_reachable_through_a_shipped_ligand() {
    // The control for the sweep above, and the reason its zero is a result.
    //
    // Same file, same code path, one number changed in the *parsed value*
    // rather than on disk: `examples/` is a shipped tree and this test has no
    // business writing to it, but the guard reads `torsdof` off the parsed
    // structure and nothing about the refusal depends on where that value came
    // from. If the guard were deleted, or its comparison inverted, or it
    // returned early, this test would go red.
    let path = examples_dir().join("ligands").join("ibuprofen.pdbqt");
    let mut parsed = read_pdbqt(&path).expect("a shipped ligand must be readable");
    let declared = parsed
        .torsdof
        .expect("a shipped Meeko ligand declares TORSDOF");

    let honest = Ligand::from_parsed(parsed.clone())
        .unwrap_or_else(|e| panic!("{} must be accepted as shipped: {e}", path.display()));
    assert_eq!(
        honest.num_torsions(),
        declared,
        "the shipped file declares {declared} and the engine must derive the same \
         number, or the acceptance above is not an acceptance"
    );

    // One more than declared: the shape of a mis-prepped ligand.
    parsed.torsdof = Some(declared + 1);
    let err = Ligand::from_parsed(parsed.clone())
        .expect_err("a declared count the engine cannot reproduce must be refused");
    let message = err.to_string();
    assert!(
        message.contains(&format!("declares {}", declared + 1)),
        "the refusal must name the declared count, got: {message}"
    );
    assert!(
        message.contains(&format!("{declared} rotatable bond(s)")),
        "the refusal must name the derived count, got: {message}"
    );

    // And one fewer, because a guard that only catches over-declaration is
    // half a guard: this is the direction where the engine would silently drop
    // a degree of freedom the file asked for.
    parsed.torsdof = Some(declared - 1);
    let err = Ligand::from_parsed(parsed)
        .expect_err("a declared count the engine cannot reproduce must be refused");
    assert!(
        err.to_string().contains("TORSDOF"),
        "the refusal must name the record it is about, got: {err}"
    );
}
