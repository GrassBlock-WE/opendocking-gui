//! PDBQT reading and writing.
//!
//! PDBQT is the atom-typed coordinate format used across the AutoDock family.
//! This module implements a tolerant reader that accepts the dialects written
//! by AutoDock 4.2 (`obabel`-style `TORSION` records), Meeko
//! (`ROOT`/`BRANCH` trees with `TORSDOF`) and plain RDKit/prepared files, and a
//! writer that emits the `MODEL`/`REMARK VINA RESULT` dialect understood by
//! AutoDockTools-free viewers.
//!
//! Reference for the column layout:
//! * AutoDock 4.2, Morris et al., *J. Comput. Chem.* **29**, 2789 (2008) — GPL-2.0.
//! * Meeko, Morris et al., *PLoS ONE* **17**, e0163573 (2022) — LGPL-2.1.
//!
//! Both are open source; this is an independent implementation and no
//! AutoDockTools (MGLTools) code was used.

use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{DockError, Result};
use crate::types::{Atom, AtomType, Bond, Element, Molecule, Vec3};

/// A `TORSION` record from an AutoDock 4 ligand file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TorsionRecord {
    /// Serial number of the atom on the *proximal* side of the rotatable bond.
    pub atom_serial_a: u32,
    /// Serial number of the atom on the *distal* side of the rotatable bond.
    pub atom_serial_b: u32,
    /// Zero-based index of the torsion in the ligand's torsion list.
    pub index: usize,
    /// `true` for `'A'` (active) torsions, `false` for `'I'` (inactive).
    pub active: bool,
}

/// The result of parsing a structure file.
///
/// # Why this is not `Default`
///
/// It derives it no longer, because it holds a [`Molecule`] and `Molecule` has
/// no `Default` — an empty molecule is the absence of a structure, and
/// `ParsedStructure::default()` was a second way to hold one. `parse_pdbqt`
/// needs a value to fill in, and builds it as an explicit literal; every field
/// is written during the parse, so the literal is scaffolding rather than a
/// default a caller could rely on. This was forced, not chosen: keeping the
/// derive would have meant re-adding `Molecule: Default`, which is the door
/// this change exists to close.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedStructure {
    /// The molecule with its covalent graph and Vina atom kinds.
    pub molecule: Molecule,
    /// Serial numbers of explicit torsions, in file order, if any were declared.
    pub torsions: Vec<TorsionRecord>,
    /// `TORSDOF` value if the file declared one.
    pub torsdof: Option<usize>,
    /// Declared torsions resolved to `(proximal, distal, active, index)` atom indices.
    pub resolved_torsions: Vec<(usize, usize, bool, usize)>,
    /// Free-text `REMARK` lines, retained for provenance.
    pub remarks: Vec<String>,
    /// Free-text `VINA RESULT` lines, one per `MODEL`.
    pub vina_results: Vec<String>,
    /// `True` if the file contained at least one `MODEL` record.
    pub multi_model: bool,
    /// Branch-tree nesting depth per atom, for layout round-tripping.
    pub branch_depth: Vec<usize>,
}

/// A single docked pose read back from a multi-model output file.
#[derive(Debug, Clone)]
pub struct DockedPose {
    /// Coordinates for every atom, in the same order as the input molecule.
    pub coords: Vec<Vec3>,
    /// The `REMARK VINA RESULT:` line, verbatim.
    pub vina_result: String,
}

/// Parse a PDBQT structure from a string.
///
/// This parses **one** structure. A pose file holds several `MODEL` blocks and
/// they are several *poses of the same molecule*, not several molecules, so only
/// the first `MODEL`'s atoms become the molecule; the later ones are skipped and
/// `ParsedStructure::multi_model` records that the file held more.
///
/// **Why this matters.** The atoms used to accumulate across every block, so the
/// nine poses of `poses.pdbqt` parsed as one 144-atom molecule in which atoms 25
/// and 107 -- from two different poses -- sit 0.06 A apart. The duplicate-atom
/// check is right to reject that, and it did: `Ligand.from_pdbqt` refused the
/// file this crate's own `write_pdbqt` had just written. Reading every pose is
/// [`read_docked_poses`]'s job; this is the job of building one molecule to dock.
pub fn parse_pdbqt(text: &str) -> Result<ParsedStructure> {
    // Scaffolding, not a default: every field below is written during the
    // parse, and `molecule` is replaced unconditionally before this value is
    // returned (an empty atom table returns early with `?`, dropping `out`).
    let mut out = ParsedStructure {
        molecule: Molecule {
            atoms: Vec::new(),
            bonds: Vec::new(),
            neighbors: Vec::new(),
        },
        torsions: Vec::new(),
        torsdof: None,
        resolved_torsions: Vec::new(),
        remarks: Vec::new(),
        vina_results: Vec::new(),
        multi_model: false,
        branch_depth: Vec::new(),
    };
    let mut atoms: Vec<Atom> = Vec::new();
    let mut serial_to_index: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut branch_depth: Vec<usize> = Vec::new();
    let mut depth: usize = 0;
    let mut next_serial: u32 = 1;
    // Set once a second `MODEL` opens, so the atoms of the later poses are
    // skipped rather than appended to the first pose's molecule.
    let mut past_first_model = false;

    for (lineno, raw) in text.lines().enumerate() {
        let line_no = lineno + 1;
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        // Match on the first whitespace-delimited token. Reading a fixed
        // 6-character record field is not enough: `TORSDOF` is seven
        // characters and `MODEL` is only five, so a column-based test would
        // silently drop records. The first token is what every PDB-family
        // writer agrees on.
        let record = line.split_whitespace().next().unwrap_or("");

        match record {
            "ATOM" | "HETATM" => {
                if past_first_model {
                    continue;
                }
                let atom = parse_atom_line(line, line_no, next_serial)?;
                next_serial = atom.serial + 1;
                serial_to_index.insert(atom.serial, atoms.len());
                atoms.push(atom);
                branch_depth.push(depth);
            }
            "REMARK" => {
                let body = line.get(6..).unwrap_or("").trim();
                if body.contains("VINA RESULT") {
                    out.vina_results.push(body.to_string());
                } else {
                    out.remarks.push(body.to_string());
                }
                if let Some(rest) = body.strip_prefix("VINA RESULT:") {
                    let _ = rest; // captured above
                }
            }
            "TORSION" => {
                // `TORSION    1  A between atoms: C_1  and  C_2`
                if let Some(t) = parse_torsion_record(line, line_no)? {
                    out.torsions.push(t);
                }
            }
            "TORSDOF" => {
                // `TORSDOF` is seven characters, so the value must be taken
                // after the record token rather than at a fixed column.
                //
                // A value that is not a non-negative integer used to be dropped
                // here, which is the one malformed input on this record that
                // *hid* rather than surfaced: `TORSDOF -1`, `TORSDOF abc` and
                // `TORSDOF 2.5` all failed `parse::<usize>()` and left
                // `torsdof` as `None`, and a `None` is precisely the state
                // `Ligand::from_parsed` treats as "the file said nothing, so
                // there is nothing to disagree with". The file's declaration
                // was therefore discarded and the ligand docked with whatever
                // torsion count the engine derived, while a file saying
                // `TORSDOF 999999` on the same structure was refused. The one
                // value that could not be read was the only one trusted.
                //
                // Refusing it costs nothing: a writer that emits `TORSDOF`
                // emits a count, and the shipped corpus does -- 56 records
                // across 24 files, every one a plain non-negative integer
                // (0, 4 or 5). The alternative -- ignoring the value and
                // docking anyway -- is the failure this whole record exists to
                // prevent.
                match line.split_whitespace().nth(1) {
                    None => {
                        return Err(DockError::parse(
                            line_no,
                            "TORSDOF record with no value: this file declares a \
                             flexibility and will not say what it is",
                        ));
                    }
                    Some(v) => match v.parse::<usize>() {
                        Ok(n) => out.torsdof = Some(n),
                        Err(_) => {
                            return Err(DockError::parse(
                                line_no,
                                format!(
                                    "TORSDOF value {v:?} is not a non-negative integer. \
                                     A file that declares its flexibility has to declare \
                                     it in a form that can be read; dropping the value \
                                     would dock this ligand over a torsion count the \
                                     file never asked for, and nothing in the result \
                                     would show it"
                                ),
                            ));
                        }
                    },
                }
            }
            "BRANCH" => {
                depth += 1;
            }
            "ENDBRANCH" => {
                depth = depth.saturating_sub(1);
            }
            "MODEL" => {
                // A `MODEL` after the first one opens another pose, not more
                // atoms. `multi_model` is set on every one of them so the flag
                // says "this file held more than one pose", not "this is pose 2".
                if out.multi_model {
                    past_first_model = true;
                }
                out.multi_model = true;
            }
            "ENDMDL" | "TER" | "ENDROOT" | "TORTOF" => {}
            _ => {
                // `BEGIN_RES`, `UNION`, `WARNING`, `USER` and vendor banners are
                // ignored on purpose: they carry no coordinates.
            }
        }
    }

    if atoms.is_empty() {
        return Err(DockError::molecule(
            "no ATOM/HETATM records found — is this a PDBQT file?".to_string(),
        ));
    }

    // Build the molecule. `branch_depth` may be shorter if a file interleaves
    // records oddly; pad defensively rather than panicking.
    branch_depth.resize(atoms.len(), 0);

    // Put the atoms in **serial** order -- the order `write_pose` numbers them,
    // and the order a caller indexing `atoms[i]` against the file's serial
    // column expects. A file written with the branch tree lists the rigid root
    // first, so the records arrive as `5,6,7,8,9,10,4,...` and appending them in
    // file order hands back a molecule whose n-th atom is not the n-th serial.
    // The bonds would still come out right, because they are resolved through
    // `serial_to_index`, so such a molecule would score correctly and still be
    // mis-indexed: the harder kind of wrong to notice. Reindexed only when
    // [`serial_order`] accepts the numbering, same as the pose reader.
    if let Some(order) = serial_order(atoms.len(), |i| atoms[i].serial as usize) {
        let mut slots: Vec<Option<Atom>> = atoms.into_iter().map(Some).collect();
        let mut depths: Vec<Option<usize>> = branch_depth.into_iter().map(Some).collect();
        let new_atoms: Vec<Atom> = order.iter().map(|&i| slots[i].take().unwrap()).collect();
        let new_depths: Vec<usize> = order.iter().map(|&i| depths[i].take().unwrap()).collect();
        // Rebuilt rather than permuted: every key here is the new index.
        let mut new_serial_to_index: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::with_capacity(new_atoms.len());
        for (i, a) in new_atoms.iter().enumerate() {
            new_serial_to_index.insert(a.serial, i);
        }
        atoms = new_atoms;
        branch_depth = new_depths;
        serial_to_index = new_serial_to_index;
    }
    out.branch_depth = branch_depth;

    let mol = Molecule::from_atoms(atoms)?;
    out.molecule = mol;

    // Resolve declared torsions to atom indices. Records referencing atoms that
    // are not in the file are dropped rather than treated as fatal, so a
    // hand-edited ligand still docks.
    for t in &out.torsions {
        if let (Some(&ia), Some(&ib)) = (
            serial_to_index.get(&t.atom_serial_a),
            serial_to_index.get(&t.atom_serial_b),
        ) {
            out.resolved_torsions.push((ia, ib, t.active, t.index));
        }
    }
    // Restore the declared order so the torsion tree is deterministic.
    out.resolved_torsions.sort_by_key(|(_, _, _, idx)| *idx);

    Ok(out)
}

impl ParsedStructure {
    /// Serial pairs `(proximal, distal)` of the file's declared active torsions.
    pub fn active_torsion_pairs(&self) -> Vec<(usize, usize)> {
        self.resolved_torsions
            .iter()
            .filter(|(_, _, active, _)| *active)
            .map(|(a, b, _, _)| (*a, *b))
            .collect()
    }
}

fn parse_torsion_record(line: &str, line_no: usize) -> Result<Option<TorsionRecord>> {
    // Drop the record name. `TORSION` is seven characters, so slicing at a
    // fixed column would leave a stray "N" as the first field.
    let mut toks = line.split_whitespace();
    let _record = toks.next();
    let rest = toks.collect::<Vec<_>>().join(" ");
    if rest.is_empty() {
        return Ok(None);
    }
    let index_tok = rest.split_whitespace().next().unwrap_or("");
    let index: usize = index_tok
        .parse()
        .map_err(|_| DockError::parse(line_no, format!("bad torsion index {index_tok:?}")))?;
    // Accept both `A between atoms: C_1  and  C_2` (AD4) and a bare
    // `A 1 2` (some third-party writers).
    let active = rest
        .split_whitespace()
        .nth(1)
        .map(|t| t.starts_with('A'))
        .unwrap_or(true);

    let serials = if let Some(p1) = rest.find("between atoms:") {
        extract_pdb_atom_serials(&rest[p1 + "between atoms:".len()..])
    } else {
        extract_pdb_atom_serials(&rest)
    };
    if serials.len() < 2 {
        return Ok(None);
    }

    Ok(Some(TorsionRecord {
        atom_serial_a: serials[0],
        atom_serial_b: serials[1],
        index,
        active,
    }))
}

/// Pull integers out of strings like `"C_1"` or `"12"`.
fn extract_pdb_atom_serials(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(v) = cur.parse::<u32>() {
                out.push(v);
            }
            cur.clear();
        }
    }
    out
}

fn parse_atom_line(line: &str, line_no: usize, fallback_serial: u32) -> Result<Atom> {
    // Fixed-column first: coordinates are far more reliably located by column
    // than by token, because several writers emit negative coordinates and
    // unaligned fields.
    let coord = fixed_coord(line);
    let (x, y, z) = match coord {
        Some(c) => c,
        None => {
            // Whitespace fallback for hand-written / non-standard files. The
            // canonical token order is
            //   ATOM serial name resName resSeq x y z occ b-factor charge type
            // so the coordinates are tokens 5, 6 and 7.
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() < 8 {
                return Err(DockError::parse(line_no, "too few fields in ATOM record"));
            }
            let parse3 = |s: &str| -> Result<f64> {
                s.parse::<f64>()
                    .map_err(|_| DockError::parse(line_no, format!("bad coordinate {s:?}")))
            };
            (parse3(toks[5])?, parse3(toks[6])?, parse3(toks[7])?)
        }
    };

    // # Why a non-finite coordinate is refused here and not downstream
    //
    // `"NaN"`, `"inf"` and `"-inf"` all parse as `f64` without complaint, so
    // neither path above rejects them: `fixed_coord` declines to *return* a
    // non-finite triple and the whitespace fallback happily hands one back. The
    // atom then enters the molecule, and what happens next depends entirely on
    // which of several unrelated guards happens to be in the way:
    //
    // * a non-finite coordinate in a **ligand** breaks distance-based bond
    //   perception, so the derived torsion count drops. If the file declares
    //   `TORSDOF` that surfaces as a torsdof mismatch and names the wrong
    //   problem; on a terminal hydrogen, where no bond is lost, nothing
    //   notices and the run ends at `cluster_poses` with *"the search produced
    //   no valid pose -- is the ligand able to fit in the box?"*, which blames
    //   the box for a corrupt atom.
    // * a non-finite coordinate in a **receptor** is worse, and the two
    //   non-finite values diverge. `NaN` fails every comparison, so the
    //   `r2 > far * far` cutoff in [`crate::grid::GridMaps::precalculate`]
    //   cannot discard the atom; it poisons every grid point and the run dies
    //   with the same box-shaped message. An **infinity** passes that cutoff
    //   cleanly -- the squared distance to an atom at infinity is infinite, so
    //   the atom is skipped as if it were too far away to matter. It is a real
    //   atom with a real element and a real charge, and it is simply not in the
    //   maps. A 30-atom receptor with one coordinate overwritten to `inf` docks
    //   and reports -5.52 kcal/mol against a clean reference of -5.66: a 2.5%
    //   error in a perfectly normal-looking result, with nothing in
    //   `DockingResult` able to say the receptor was short one atom.
    //
    // So the honest place to refuse is the record, where the line number still
    // exists and the file is still the thing being talked about. Every later
    // guard is incidental: it fires for a reason of its own, and the message
    // names a knob the user does not have to turn.
    //
    // This is a refusal rather than a counted degradation, unlike an
    // unrecognised *atom type*. There is no honest weaker claim to make about
    // an atom with no position: it cannot be placed, so it cannot be scored,
    // and the number of such atoms says nothing about which poses survived.
    //
    // # Why this is the *second* place the rule is written, not the only one
    //
    // The test itself is [`crate::types::first_non_finite`], shared with
    // [`crate::types::Molecule::from_atoms`],
    // [`crate::ligand::Ligand::from_molecule_with`],
    // [`crate::receptor::Receptor::from_molecule`] and
    // [`crate::kinematics::Conformation::from_slice`]. The message stays here,
    // because a line number is worth having and none of the other four sites has
    // one; the *decision* does not stay here, because a decision that lives only
    // on the path that happens to be tested is a decision the in-memory
    // constructors do not make.
    if let Some(axis) = crate::types::first_non_finite(&[x, y, z]) {
        let names = ["x", "y", "z"];
        return Err(DockError::parse(
            line_no,
            format!(
                "non-finite coordinate ({x}, {y}, {z}): the {0} component is not a \
                 finite number, and a NaN or infinite position is not a location -- as \
                 a receptor atom it is silently dropped from the maps and as a ligand \
                 atom it makes every pose unusable, while the run still reports a \
                 result",
                names[axis],
            ),
        ));
    }

    let serial = slice_parse::<u32>(line, 6, 11)
        .or_else(|| whitespace_serial(line))
        .unwrap_or(fallback_serial);

    let name = col(line, 12, 16)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}_{}", Element::C.symbol(), serial));

    let resname = col(line, 17, 20)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "UNL".to_string());

    let resid = slice_parse::<i32>(line, 22, 26).unwrap_or(1);

    let charge = slice_parse::<f32>(line, 70, 76)
        .or_else(|| trailing_charge(line))
        .unwrap_or(0.0);

    let token = col(line, 77, 79)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| last_token(line))
        .unwrap_or_default();

    let atom_type = AtomType::from_token(&token);
    let element = if atom_type == AtomType::Unknown {
        infer_element_from_name(&name)
    } else {
        atom_type.element()
    };

    let atom = Atom {
        serial,
        name,
        resname,
        resid,
        coord: [x, y, z],
        charge,
        element,
        atom_type,
        kind: crate::types::AtomKind::Other,
    };
    Ok(atom)
}

fn fixed_coord(line: &str) -> Option<(f64, f64, f64)> {
    let c = fixed_coord_parsed(line)?;
    if c.0.is_finite() && c.1.is_finite() && c.2.is_finite() {
        Some(c)
    } else {
        None
    }
}

/// The three fixed coordinate columns *as parsed*, without the finiteness test
/// [`fixed_coord`] applies.
///
/// Split out so a caller can tell "these columns do not hold a coordinate" from
/// "these columns hold a coordinate that is not a number in the reals". Only
/// the second is worth refusing, and refusing it needs the distinction.
fn fixed_coord_parsed(line: &str) -> Option<(f64, f64, f64)> {
    let x: f64 = slice_parse(line, 30, 38)?;
    let y: f64 = slice_parse(line, 38, 46)?;
    let z: f64 = slice_parse(line, 46, 54)?;
    Some((x, y, z))
}

/// The trimmed text of a fixed-width column, or `""` when the line is short.
fn col(line: &str, start: usize, end: usize) -> Option<&str> {
    line.get(start..end).map(str::trim)
}

fn slice_parse<T: std::str::FromStr>(line: &str, start: usize, end: usize) -> Option<T> {
    line.get(start..end)?.trim().parse::<T>().ok()
}

fn whitespace_serial(line: &str) -> Option<u32> {
    line.split_whitespace().nth(1)?.parse::<u32>().ok()
}

/// The last token of a line, used when the atom type is not column-aligned.
fn last_token(line: &str) -> Option<String> {
    line.split_whitespace().last().map(|s| s.to_string())
}

/// Recover a charge that a writer placed in a non-standard column.
fn trailing_charge(line: &str) -> Option<f32> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 2 {
        return None;
    }
    // Walk backwards past the atom type, then try to read a float.
    let n = toks.len();
    for i in (0..n.saturating_sub(1)).rev() {
        if let Ok(v) = toks[i].parse::<f32>() {
            if toks[i].parse::<i32>().is_err() {
                return Some(v);
            }
            break;
        }
    }
    None
}

fn infer_element_from_name(name: &str) -> Element {
    let n = name.trim();
    if n.is_empty() {
        return Element::C;
    }
    // PDB atom names right-justify the element symbol: ` CA ` is a carbon
    // alpha, `CA  ` is a calcium.
    if n.len() >= 2 {
        let two = &n[0..2];
        if two.eq_ignore_ascii_case("Cl") || two.eq_ignore_ascii_case("CL") {
            return Element::Cl;
        }
        if two.eq_ignore_ascii_case("Br") || two.eq_ignore_ascii_case("BR") {
            return Element::Br;
        }
        if two.eq_ignore_ascii_case("Mg") || two.eq_ignore_ascii_case("MG") {
            return Element::Met;
        }
        if two.eq_ignore_ascii_case("Zn") || two.eq_ignore_ascii_case("ZN") {
            return Element::Met;
        }
        if two.eq_ignore_ascii_case("Fe") || two.eq_ignore_ascii_case("FE") {
            return Element::Met;
        }
    }
    match n.chars().next().unwrap().to_ascii_uppercase() {
        'H' => Element::H,
        'C' => Element::C,
        'N' => Element::N,
        'O' => Element::O,
        'S' => Element::S,
        'P' => Element::P,
        'F' => Element::F,
        'I' => Element::I,
        _ => Element::C,
    }
}

/// Read and parse a PDBQT file.
pub fn read_pdbqt(path: impl AsRef<Path>) -> Result<ParsedStructure> {
    let file = fs::File::open(path.as_ref()).map_err(|e| {
        DockError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.as_ref().display()),
        ))
    })?;
    let mut text = String::new();
    let mut reader = BufReader::new(file);
    let mut buf = String::new();
    while reader.read_line(&mut buf).map_err(DockError::Io)? > 0 {
        text.push_str(&buf);
        buf.clear();
    }
    parse_pdbqt(&text)
}

/// The serial in columns 7-11, or the 1-based position of the record when the
/// field is not a number.
///
/// The fallback is the *file* order, which is what a reader that ignored the
/// serial used to do unconditionally. It is only correct for a file that
/// happens to be written in serial order, and a file this module writes is not.
fn serial_of(line: &str, position: usize) -> usize {
    line.get(6..11)
        .and_then(|f| f.trim().parse::<usize>().ok())
        .unwrap_or(position + 1)
}

/// The permutation that puts `n` records into serial order, or `None` when the
/// serials are not exactly `1..=n`, each once.
///
/// Both readers in this module go through this one predicate, so the rule for
/// what counts as a usable numbering is stated once: absent serials, duplicates,
/// or a numbering that starts elsewhere are all "not reindexable", and the file
/// is then passed through in the order it was written.
fn serial_order(n: usize, serial_at: impl Fn(usize) -> usize) -> Option<Vec<usize>> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| serial_at(i));
    // `order[rank]` is the record holding the (rank + 1)-th smallest serial, so
    // this one clause is the whole completeness test.
    order
        .iter()
        .enumerate()
        .all(|(rank, &i)| serial_at(i) == rank + 1)
        .then_some(order)
}

/// Put a model's coordinates into serial order, the order the rest of the crate
/// and the Python surface both use (`DockingResult::pose_coords` is indexed by
/// atom index, not by the order the lines were written in).
///
/// A model is reindexed only when [`serial_order`] accepts its numbering.
/// Anything else is left in file order, because sorting by a numbering that is
/// not the atom index would substitute one wrong answer for another. That case
/// is worth knowing about, so it is a documented branch rather than a silent
/// one: a file with a complete 1..n run is reindexed, and any other file is
/// passed through with the order it was written in.
fn reindex_by_serial(model: Vec<(usize, Vec3)>) -> Vec<Vec3> {
    match serial_order(model.len(), |i| model[i].0) {
        Some(order) => order.iter().map(|&i| model[i].1).collect(),
        None => model.into_iter().map(|(_, c)| c).collect(),
    }
}

/// Read every `MODEL` of a multi-model PDBQT file, returning one pose per model.
///
/// The coordinates come back in **serial** order, not in the order the `ATOM`
/// records appear. Those are different: a file written by [`write_pose`] with
/// `write_branch_tree` lists the rigid root first, so its serials run
/// `5,6,7,8,9,10,4,11,12,1,2,3,...` on the molecule measured here. Reindexing
/// by serial is what makes a write/read round trip reproduce the pose
/// coordinates the API reported -- measured, the same file read in file order
/// is 3.99 A RMSD away from them, and read in serial order is 0.0005 A away,
/// which is the 3-decimal write precision.
pub fn read_docked_poses(path: impl AsRef<Path>) -> Result<Vec<DockedPose>> {
    let file = fs::File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut poses: Vec<DockedPose> = Vec::new();
    let mut cur: Vec<(usize, Vec3)> = Vec::new();
    let mut last_result = String::new();
    let mut pending_result: Option<String> = None;

    for line in reader.lines() {
        let line = line.map_err(DockError::Io)?;
        let record = line.split_whitespace().next().unwrap_or("");
        match record {
            "MODEL" => {
                cur.clear();
            }
            "REMARK" => {
                let body = line.get(6..).unwrap_or("").trim();
                if body.contains("VINA RESULT") {
                    pending_result = Some(body.to_string());
                }
            }
            "ATOM" | "HETATM" => {
                // `fixed_coord` answers `None` for three different reasons --
                // no coordinate in these columns, text that is not a number, and
                // a number that is not finite -- and this reader treats the
                // first two as "skip the line" but used to treat the third the
                // same way. That last one is not a line to skip: it is a line
                // that would have contributed one atom to the pose, and skipping
                // it returns a pose with fewer atoms than the file holds, with
                // nothing to say so. So the third case is separated out here and
                // refused. `parse_pdbqt` refuses the same input, for the same
                // reason, and the two readers are the only ways a coordinate
                // enters the crate.
                if let Some((x, y, z)) = fixed_coord_parsed(&line) {
                    if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                        return Err(DockError::parse(
                            0,
                            format!("non-finite coordinate ({x}, {y}, {z}) in a pose record"),
                        ));
                    }
                }
                if let Some(c) = fixed_coord(&line) {
                    // The serial comes with the coordinate because the writer
                    // does not emit `ATOM` records in serial order -- it emits
                    // them in branch-tree order, root first -- and the serial
                    // is the only thing in the file that says which atom a line
                    // is. Dropping it here is what turned this reader into a
                    // 3.99 A permutation; see [`reindex_by_serial`].
                    cur.push((serial_of(&line, cur.len()), [c.0, c.1, c.2]));
                }
            }
            "ENDMDL" => {
                poses.push(DockedPose {
                    coords: reindex_by_serial(std::mem::take(&mut cur)),
                    vina_result: pending_result.take().unwrap_or_else(|| last_result.clone()),
                });
            }
            _ => {}
        }
        if record == "REMARK" {
            // keep the last result for files that put it outside MODEL blocks
            if let Some(b) = line.get(6..) {
                let b = b.trim();
                if b.contains("VINA RESULT") {
                    last_result = b.to_string();
                }
            }
        }
    }
    Ok(poses)
}

/// Options controlling PDBQT output.
#[derive(Debug, Clone)]
pub struct PdbqtWriteOptions {
    /// Emit `MODEL`/`ENDMDL` wrappers (required for multi-pose files).
    pub multi_model: bool,
    /// Emit `ROOT`/`BRANCH`/`ENDBRANCH` blocks instead of a flat atom list.
    pub write_branch_tree: bool,
    /// Number of decimal places for coordinates.
    pub coord_precision: usize,
    /// Number of decimal places for the reported energy.
    pub energy_precision: usize,
}

impl Default for PdbqtWriteOptions {
    fn default() -> Self {
        PdbqtWriteOptions {
            multi_model: false,
            write_branch_tree: true,
            coord_precision: 3,
            energy_precision: 1,
        }
    }
}

/// A pose ready for serialisation: atoms plus their (possibly moved) coordinates.
pub struct PoseOutput<'a> {
    /// Reference to the atoms (metadata is reused verbatim).
    pub atoms: &'a [Atom],
    /// Coordinates, one per atom, in the same order as `atoms`.
    pub coords: &'a [Vec3],
    /// Energy in kcal/mol.
    pub energy: f64,
    /// Intermolecular (receptor–ligand) part of the energy.
    pub intermolecular: f64,
    /// Intramolecular (ligand internal) part of the energy.
    pub intramolecular: f64,
    /// Cluster rank of the pose, from most negative (best) upward.
    pub rank: usize,
    /// PDBQT run identifier echoed from the input.
    pub run: usize,
    /// Per-atom RMSD from the reference conformation, if computed.
    pub rmsd_lb: Option<f64>,
    /// Per-atom RMSD from the best pose, if computed.
    pub rmsd_ub: Option<f64>,
}

/// Write a single pose in PDBQT format.
pub fn write_pose<W: Write>(
    out: &mut W,
    pose: &PoseOutput<'_>,
    mol: &Molecule,
    torsions: &[(usize, usize)],
    opts: &PdbqtWriteOptions,
) -> Result<()> {
    if pose.coords.len() != pose.atoms.len() {
        return Err(DockError::molecule(format!(
            "pose has {} coordinates for {} atoms",
            pose.coords.len(),
            pose.atoms.len()
        )));
    }
    let mut buf = String::new();

    // The `REMARK` records go **inside** the `MODEL` block, not before it.
    // That is the AutoDock/Vina convention (and what this module's own read
    // fixtures show), and it is the only layout a per-model splitter can use:
    // anything written before the first `MODEL` is unattributed, so a reader
    // that walks `MODEL`..`ENDMDL` finds no energy and every pose looks like
    // 0.00. The workbench pose browser was showing exactly that.
    if opts.multi_model {
        buf.push_str("MODEL\n");
    }

    let energy = format!("{:.*}  0.000  0.000", opts.energy_precision, pose.energy);
    buf.push_str(&format!("REMARK VINA RESULT:    {energy}\n"));
    buf.push_str(&format!(
        "REMARK INTER + INTRA:         {:.*}\n",
        opts.energy_precision, pose.energy
    ));
    if let (Some(lb), Some(ub)) = (pose.rmsd_lb, pose.rmsd_ub) {
        buf.push_str(&format!("REMARK RMSD_LB, RMSD_UB: {lb:.2}, {ub:.2}\n"));
    }
    buf.push_str(&format!("REMARK Run: {}\n", pose.run));
    // The order the `ATOM` records below appear in, declared. A file that does
    // not say is relying on the reader knowing a convention this writer is the
    // only one to break: the root cluster is written first, so the serials are
    // not in ascending order. A reader that takes the n-th record for the n-th
    // atom gets a permuted molecule, and nothing in the file tells it so. See
    // [`ATOM_ORDER_REMARK`].
    buf.push_str(&format!(
        "{ATOM_ORDER_REMARK} BRANCH TREE; SERIAL = ATOM INDEX; REINDEX BY SERIAL\n"
    ));
    write_connectivity(&mut buf, pose, mol);

    if opts.write_branch_tree && !torsions.is_empty() {
        write_branch_tree(&mut buf, pose, mol, torsions, opts);
    } else {
        for (i, atom) in pose.atoms.iter().enumerate() {
            write_atom_line(&mut buf, atom, pose.coords[i], i + 1, opts);
        }
    }

    if opts.multi_model {
        buf.push_str("ENDMDL\n");
    }
    out.write_all(buf.as_bytes())?;
    Ok(())
}

/// Write one `ATOM` record at the exact PDBQT column positions.
///
/// The layout is fixed-width, so every field is padded explicitly rather than
/// relying on `format!` to guess separators:
///
/// ```text
///  1–6  "ATOM  "        7–11 serial     13–16 name     18–20 resName
/// 23–26  resSeq        31–38 x          39–46 y        47–54 z
/// 55–60  occupancy     61–66 tempFactor 71–76 charge   78–79 atom type
/// ```
fn write_atom_line(
    buf: &mut String,
    atom: &Atom,
    coord: Vec3,
    serial: usize,
    opts: &PdbqtWriteOptions,
) {
    let w = opts.coord_precision;
    let name = format_atom_name(atom);
    let mut line = String::with_capacity(80);
    line.push_str("ATOM  "); // 1–6
    line.push_str(&format!("{serial:>5}")); // 7–11
    line.push(' '); // 12  altLoc spacer
    line.push_str(&fit(&name, 4, true)); // 13–16
    line.push(' '); // 17  altLoc
    line.push_str(&fit(&atom.resname, 3, true)); // 18–20
    line.push(' '); // 21  chainID
    line.push(' '); // 22
    line.push_str(&format!("{:>4}", atom.resid)); // 23–26 resSeq
    line.push(' '); // 27  iCode
    line.push_str("   "); // 28–30
    line.push_str(&format!("{:>8.*}", w, coord[0])); // 31–38
    line.push_str(&format!("{:>8.*}", w, coord[1])); // 39–46
    line.push_str(&format!("{:>8.*}", w, coord[2])); // 47–54
    line.push_str(&format!("{:>6.2}", 1.0)); // 55–60 occupancy
    line.push_str(&format!("{:>6.2}", 0.0)); // 61–66 tempFactor
    line.push_str("    "); // 67–70
    line.push_str(&format!("{:>6.3}", atom.charge)); // 71–76 partial charge
    line.push(' '); // 77
    line.push_str(&fit(atom.atom_type.token(), 2, false)); // 78–79
    line.push('\n');
    buf.push_str(&line);
}

/// Pad or truncate `s` to exactly `width` characters.
fn fit(s: &str, width: usize, right_align: bool) -> String {
    let t = if s.chars().count() > width {
        s.chars().take(width).collect::<String>()
    } else {
        s.to_string()
    };
    let pad = width - t.chars().count();
    if right_align {
        format!("{}{}", " ".repeat(pad), t)
    } else {
        format!("{}{}", t, " ".repeat(pad))
    }
}

/// PDB atom names are right-justified in a 4-character field for one-letter
/// elements and left-justified for two-letter ones, which is what downstream
/// viewers expect.
fn format_atom_name(atom: &Atom) -> String {
    let n = atom.name.trim();
    if n.is_empty() {
        return format!(" {}", atom.element.symbol());
    }
    if n.len() >= 2
        && (n.starts_with(char::is_uppercase) && n.chars().nth(1).is_some_and(|c| c.is_lowercase()))
    {
        // Already a two-letter element (e.g. "Cl"): left-justify.
        n.to_string()
    } else if n.len() == 1 {
        format!(" {n}")
    } else {
        n.to_string()
    }
}

/// Record name carrying the declared covalent connectivity of a pose.
///
/// It is a `REMARK` rather than a bare `CONECT` for three reasons, all of them
/// about not breaking a reader that has never heard of it: every tool in this
/// format already skips `REMARK` bodies it does not recognise, the bond count
/// is stated on its own record so a truncated file is detectable rather than
/// silently short, and the record carries no columns a fixed-width reader
/// could mis-slice. The pairs are **atom serials**, never file positions.
///
/// The two record names are deliberately **not** prefixes of one another. An
/// earlier draft called the count `OD_BONDCOUNT` and the bond `OD_BOND`, and
/// `OD_BOND` is a prefix of `OD_BONDCOUNT`, so the obvious way to find the bond
/// records -- filter the lines for the bond record name -- also matched the
/// count line and then read one serial where two were expected. Caught by
/// `a_written_pose_declares_its_bonds_keyed_by_serial` below, which is what a
/// unit test is for. `OD_NBONDS` and `OD_BOND` share no prefix, and filtering
/// for `REMARK OD_BOND ` now yields exactly the bond records.
const CONNECTIVITY_COUNT_REMARK: &str = "REMARK OD_NBONDS";

/// The `REMARK` that declares how the `ATOM` records of a `MODEL` are ordered.
///
/// **Why a `REMARK` and not the ordering itself.** A `REMARK` is inert to Vina,
/// meeko and every general viewer, so a file carrying one is still a valid pose
/// file for a tool that knows nothing about it -- the same property
/// [`CONNECTIVITY_COUNT_REMARK`] and `REMARK VINA RESULT` already rely on. The
/// alternative, writing the records in serial order, would change the bytes of
/// every pose file the project has ever written and lose the branch tree, which
/// is what makes the file re-dockable. So the order is kept and **declared**.
///
/// The declaration is not what [`read_docked_poses`] acts on. That reader
/// reindexes by the serial column **unconditionally**, through
/// [`reindex_by_serial`], and would behave identically on a file carrying no
/// `REMARK OD_ATOM_ORDER` at all. The 3.99 Å-vs-0.0005 Å figures in that
/// function's own comment are serial-order against file-order, not
/// remark-against-no-remark. The `REMARK` earns its place by being the
/// *declaration* -- it tells a reader that this file's atom order is not its
/// serial order, which is the fact a third-party consumer needs and cannot
/// infer from a column it is expected to understand.
const ATOM_ORDER_REMARK: &str = "REMARK OD_ATOM_ORDER";
/// Continuation records; one bond per line, as two serials.
const CONNECTIVITY_BOND_REMARK: &str = "REMARK OD_BOND";

/// Write the pose's covalent bonds, keyed by atom serial.
///
/// **Why this exists.** `ROOT`/`BRANCH` already declare the torsion tree, but
/// the tree is indexed in *file order* while a reader that builds its view in
/// *serial order* has to re-index every bond, and `poses.pdbqt` is genuinely
/// not in serial order -- measured on the shipped file, the serials run
/// `5,6,7,8,9,10,4,11,12,1,2,3`. A reader using the tree without that
/// re-indexing step mis-indexes every bond in the molecule. The tree also says
/// nothing about the bonds *inside* the rigid root cluster, which is where a
/// ring lives. The fix is to declare the connectivity outright and key it by
/// serial, so file order stops mattering.
///
/// **Why `REMARK` and not `CONECT`.** A `CONECT` record is a real part of the
/// PDB vocabulary and other tools will parse it, on their terms: Vina and meeko
/// both ignore it, but a general viewer will honour it and draw bonds that may
/// not be the ones the engine scored. A `REMARK` is inert to all of them, so
/// the record is additive by construction: a file that carries it is still a
/// valid pose file for a tool that knows nothing about it, which is the same
/// property the existing `REMARK VINA RESULT` line relies on.
///
/// The bond list comes from [`Molecule::bonds`] -- the graph the engine scored
/// -- and not from a distance test at write time, so the record cannot disagree
/// with the energy in the same `MODEL` block.
fn write_connectivity(buf: &mut String, pose: &PoseOutput<'_>, mol: &Molecule) {
    // `pose.atoms` is a prefix-compatible view of `mol.atoms` in every caller,
    // and the atom lines number it by `i + 1`, so serial = index + 1. A bond
    // whose indices fall outside the pose is dropped rather than written with a
    // serial that no `ATOM` record carries, which would be a bond to nothing.
    let n = pose.atoms.len().min(mol.atoms.len());
    let mut bonds: Vec<(usize, usize)> = Vec::new();
    for b in &mol.bonds {
        if b.i >= n || b.j >= n || b.i == b.j {
            continue;
        }
        bonds.push((b.i, b.j));
    }
    // A deterministic order: a file written twice from the same molecule has to
    // be byte-identical, or "diff two runs" stops being evidence of anything.
    bonds.sort_unstable();
    bonds.dedup();
    buf.push_str(&format!("{CONNECTIVITY_COUNT_REMARK} {}\n", bonds.len()));
    for (i, j) in bonds {
        buf.push_str(&format!(
            "{CONNECTIVITY_BOND_REMARK} {:>5} {:>5}\n",
            i + 1,
            j + 1
        ));
    }
}

fn write_branch_tree(
    buf: &mut String,
    pose: &PoseOutput<'_>,
    mol: &Molecule,
    torsions: &[(usize, usize)],
    opts: &PdbqtWriteOptions,
) {
    // The root cluster is everything not distal to a torsion.
    let mut is_distal = vec![false; pose.atoms.len()];
    for &(a, b) in torsions {
        mark_distal(mol, a, b, &mut is_distal);
    }
    let root: Vec<usize> = (0..pose.atoms.len()).filter(|&i| !is_distal[i]).collect();

    buf.push_str("ROOT\n");
    for &i in &root {
        write_atom_line(buf, &pose.atoms[i], pose.coords[i], i + 1, opts);
    }
    buf.push_str("ENDROOT\n");

    for &(a, b) in torsions {
        buf.push_str(&format!("BRANCH {:>3} {:>3}\n", a + 1, b + 1));
        // A branch holds the atoms of *one* rigid cluster: the distal atom `b`
        // plus everything reachable from it without crossing a – b or any other
        // rotatable bond. Writing the whole subtree instead would repeat every
        // descendant atom in its own branch as well, and the reader would then
        // see a molecule with duplicated atoms.
        let mut in_branch = vec![false; pose.atoms.len()];
        collect_cluster(mol, torsions, a, b, &mut in_branch);
        for i in 0..pose.atoms.len() {
            if in_branch[i] {
                write_atom_line(buf, &pose.atoms[i], pose.coords[i], i + 1, opts);
            }
        }
        buf.push_str("ENDBRANCH\n");
    }
    buf.push_str("TORSDOF ");
    // count = torsions + 3 rigid-body degrees of freedom is *not* what
    // TORSDOF means: it counts only the active rotatable bonds.
    buf.push_str(&format!("{}\n", torsions.len()));
}

/// Mark every atom distal to the rotatable bond `a -> b`.
fn mark_distal(mol: &Molecule, a: usize, b: usize, out: &mut [bool]) {
    let mut stack = vec![b];
    out[b] = true;
    while let Some(u) = stack.pop() {
        for &v in &mol.neighbors[u] {
            if v != a && !out[v] {
                out[v] = true;
                stack.push(v);
            }
        }
    }
}

/// Collect the atoms of the single rigid cluster distal to the bond `a – b`.
///
/// The walk starts at `b` and never crosses `a – b` or any *other* rotatable
/// bond, so it stops exactly at the boundary of the cluster. Because the
/// torsion list is the full set of rotatable bonds, no explicit parent/child
/// bookkeeping is needed: a deeper branch's cluster simply lies behind its own
/// torsion bond.
fn collect_cluster(
    mol: &Molecule,
    torsions: &[(usize, usize)],
    a: usize,
    b: usize,
    out: &mut [bool],
) {
    let is_torsion = |u: usize, v: usize| {
        let key = (u.min(v), u.max(v));
        torsions
            .iter()
            .any(|&(ka, kb)| (ka.min(kb), ka.max(kb)) == key)
    };
    let mut stack = vec![b];
    out[b] = true;
    while let Some(u) = stack.pop() {
        for &v in &mol.neighbors[u] {
            if out[v] || v == a || is_torsion(u, v) {
                continue;
            }
            out[v] = true;
            stack.push(v);
        }
    }
}

/// Convenience: write a set of poses to a file.
pub fn write_poses(
    path: impl AsRef<Path>,
    poses: &[PoseOutput<'_>],
    mol: &Molecule,
    torsions: &[(usize, usize)],
    opts: &PdbqtWriteOptions,
) -> Result<()> {
    let file = fs::File::create(path.as_ref()).map_err(|e| {
        DockError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.as_ref().display()),
        ))
    })?;
    let mut w = BufWriter::new(file);
    for pose in poses {
        write_pose(&mut w, pose, mol, torsions, opts)?;
    }
    w.flush()?;
    Ok(())
}

/// Convenience: write one pose to a file.
pub fn write_single_pose(
    path: impl AsRef<Path>,
    pose: &PoseOutput<'_>,
    mol: &Molecule,
    torsions: &[(usize, usize)],
    opts: &PdbqtWriteOptions,
) -> Result<()> {
    write_poses(path, std::slice::from_ref(pose), mol, torsions, opts)
}

/// Re-exported so callers can construct bonds without importing `types`.
pub type MoleculeBond = Bond;

#[cfg(test)]
mod tests {
    use super::*;

    const ETHANOL: &str = concat!(
        "REMARK  test\n",
        "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.090 C \n",
        "ATOM      2  C2  UNL     1       1.500   0.000   0.000  1.00  0.00     0.090 C \n",
        "ATOM      3  O1  UNL     1       2.100   1.200   0.000  1.00  0.00    -0.400 OA \n",
        "ATOM      4  H1  UNL     1       2.100   2.100   0.500  1.00  0.00     0.210 HD \n",
    );

    #[test]
    fn parses_atoms_and_coordinates() {
        let s = parse_pdbqt(ETHANOL).unwrap();
        assert_eq!(s.molecule.len(), 4);
        assert!((s.molecule.atoms[1].coord[0] - 1.5).abs() < 1e-9);
        assert!((s.molecule.atoms[3].coord[2] - 0.5).abs() < 1e-9);
        assert_eq!(s.molecule.atoms[2].atom_type, AtomType::OA);
        assert_eq!(s.molecule.atoms[3].atom_type, AtomType::HD);
        assert!((s.molecule.atoms[2].charge + 0.4).abs() < 1e-6);
    }

    #[test]
    fn perceives_two_bonds() {
        let s = parse_pdbqt(ETHANOL).unwrap();
        // C1-C2, C2-O1, O1-H1
        assert_eq!(s.molecule.bond_count(), 3);
    }

    #[test]
    fn reads_torsdof_and_torsion_records() {
        // AutoDock 4 writes the *PDB serial numbers* in the TORSION record, so
        // the names must agree with the serials assigned in the file above:
        // atom 1 = C1, 2 = C2, 3 = O1, 4 = H1.
        let text = format!(
            "{ETHANOL}TORSDOF 2\nTORSION    1  A between atoms: C_1  and  C_2\nTORSION    2  A between atoms: C_2  and  O_3\n"
        );
        let s = parse_pdbqt(&text).unwrap();
        assert_eq!(s.torsdof, Some(2));
        assert_eq!(s.resolved_torsions.len(), 2);
        // Atom indices (0-based) for serials 1,2 and 2,3.
        assert_eq!(s.active_torsion_pairs(), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn torsion_records_with_unknown_atoms_are_dropped_not_fatal() {
        // The *parser* stays tolerant, and this test is what pins that. A file
        // with one good atom line and one unresolvable TORSION still parses,
        // because refusing to read a file at all is a different failure from
        // reading it imperfectly, and a reader that dies on one bad record
        // cannot report anything about the rest of the file.
        //
        // The tolerance used to end here, which is the defect: the parse
        // succeeded, `TORSDOF 1` was right there in the file, and
        // `Ligand::from_pdbqt` went on to build a zero-torsion ligand and dock
        // it. `Ligand::from_parsed` is now where the disagreement is caught --
        // see `ligand::tests::a_declared_torsion_the_engine_cannot_build_is_refused`.
        // This test covers the parse; that one covers the consequence.
        let text = format!("{ETHANOL}TORSDOF 1\nTORSION    1  A between atoms: C_1  and  C_99\n");
        let s = parse_pdbqt(&text).unwrap();
        assert_eq!(s.molecule.len(), 4);
        assert!(s.active_torsion_pairs().is_empty());
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(parse_pdbqt("REMARK nothing here\n").is_err());
    }

    // ---------------------------------------------------------------------
    // Non-finite coordinates. The measured failures this pins, on the shipped
    // `biotin_prep.pdbqt` (19 atoms) and `rec_prep.pdbqt` (30 atoms):
    //
    //   * receptor, one coordinate overwritten to `inf`:
    //       `Receptor.from_pdbqt` accepted it, `precalculate` succeeded, and
    //       `dock` returned `best = -5.5199` against a clean reference of
    //       `-5.6606` -- a 2.5% error, ranked and timed, with nothing in the
    //       result able to say the receptor was one atom short.
    //   * ligand, one coordinate overwritten to `inf`:
    //       accepted, and `dock` refused with *"axis x is 18.0 Å but the
    //       ligand needs at least inf Å"* -- a complaint about the box.
    //   * ligand, `NaN` on a terminal hydrogen:
    //       accepted, `num_torsions` unchanged at 5 so no torsdof check fires,
    //       and `dock` refused with *"the search produced no valid pose -- is
    //       the ligand able to fit in the box?"* -- again about the box.
    //
    // All three are the same defect: the file is refused somewhere other than
    // where the fault is, by a guard that exists for a different reason, with a
    // message that sends the user to a knob they do not need to turn.

    /// Replace the x coordinate of the `which`-th `ATOM` record, asserting the
    /// edit landed -- a probe that silently edits nothing measures the clean
    /// file and looks like a pass.
    ///
    /// `ETHANOL` has one `REMARK` line ahead of its four atoms, so atom `which`
    /// is on line `which + 2`.
    fn with_x(source: &str, which: usize, token: &str) -> String {
        let mut seen = 0usize;
        let mut out: Vec<String> = Vec::new();
        for line in source.lines() {
            let mut line = line.to_string();
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                if seen == which {
                    assert!(token.len() <= 8, "token must fit the 8-wide column");
                    let f: Vec<char> = line.chars().collect();
                    assert!(f.len() >= 38, "fixture line too short: {line}");
                    let mut nf: String = f[..30].iter().collect();
                    nf.push_str(&format!("{token:>8}"));
                    nf.extend(f[38..].iter());
                    line = nf;
                }
                seen += 1;
            }
            out.push(line);
        }
        assert!(seen > which, "no atom at index {which} (file has {seen})");
        out.push(String::new());
        out.join("\n")
    }

    #[test]
    fn a_non_finite_coordinate_is_refused_at_the_record_that_carries_it() {
        for token in ["NaN", "inf", "-inf", "Infinity", "INFINITY"] {
            for which in [0usize, 3] {
                // `which = 3` is ethanol's polar hydrogen: a terminal atom, so
                // no covalent bond is lost to the bad coordinate and the
                // derived torsion count is unchanged. That is the case with no
                // other guard in the way.
                let text = with_x(ETHANOL, which, token);
                let err = parse_pdbqt(&text)
                    .expect_err(&format!("x={token} on atom {} must be refused", which + 1));
                let msg = err.to_string();
                assert!(
                    msg.contains("non-finite coordinate"),
                    "x={token} on atom {}: message must name the fault, got: {msg}",
                    which + 1
                );
                // The line number is the whole reason to refuse here rather than
                // three frames later, so it has to be in the message -- and it
                // has to be the *right* line.
                assert!(
                    msg.contains(&format!("line {}", which + 2)),
                    "x={token} on atom {}: the offending line must be identified \
                     as line {}, got: {msg}",
                    which + 1,
                    which + 2
                );
            }
        }
    }

    #[test]
    fn a_non_finite_coordinate_is_refused_on_the_receptor_path_too() {
        // The receptor reaches the same `parse_atom_line`, and it is the case
        // that produced a *successful* run: an infinite coordinate is discarded
        // by the `r2 > far * far` cutoff in the tabulation loop, so the atom is
        // simply not in the maps. Refusing the record is what stops that.
        let text = with_x(ETHANOL, 0, "inf");
        let err = crate::receptor::Receptor::from_pdbqt_str(&text)
            .expect_err("a receptor atom at infinity must be refused, not tabulated around");
        assert!(err.to_string().contains("non-finite coordinate"));
    }

    #[test]
    fn a_finite_coordinate_that_merely_looks_odd_is_still_read() {
        // The refusal is for `NaN`/`inf`, not for being unusual: a large but
        // finite coordinate is a real position, and `GridBox::new` already
        // refuses a non-finite *box*, so the two ends agree about what a
        // coordinate may be.
        let text = with_x(ETHANOL, 0, "9999.5");
        let s = parse_pdbqt(&text).expect("a large finite coordinate is a location");
        assert_eq!(s.molecule.len(), 4);
        assert!((s.molecule.atoms[0].coord[0] - 9999.5).abs() < 1e-9);
    }

    // ---------------------------------------------------------------------
    // A `TORSDOF` value that cannot be read.

    #[test]
    fn a_torsdof_value_that_cannot_be_read_is_refused_not_dropped() {
        // Measured on the shipped `biotin_prep.pdbqt`: `TORSDOF -1`,
        // `TORSDOF abc` and `TORSDOF 2.5` all failed `parse::<usize>()`, left
        // `torsdof` as `None`, and the ligand docked normally with the engine's
        // own derived count -- while `TORSDOF 999999` on the same file was
        // refused. The one value the engine could not read was the only one it
        // trusted.
        for bad in ["-1", "abc", "2.5", "1e3", "+-2", "3.0"] {
            let text = format!("{ETHANOL}TORSDOF {bad}\n");
            let err = parse_pdbqt(&text)
                .expect_err(&format!("`TORSDOF {bad}` must be refused, not ignored"));
            let msg = err.to_string();
            assert!(
                msg.contains("TORSDOF"),
                "`TORSDOF {bad}`: message must name the record, got: {msg}"
            );
        }
    }

    #[test]
    fn a_torsdof_record_with_no_value_at_all_is_refused() {
        let err =
            parse_pdbqt(&format!("{ETHANOL}TORSDOF\n")).expect_err("a bare TORSDOF says nothing");
        assert!(err.to_string().contains("TORSDOF"));
    }

    #[test]
    fn a_torsdof_value_that_parses_is_unchanged() {
        // The other direction: the refusal must not swallow a well-formed
        // declaration. `TORSDOF 0` on a rigid fragment and `TORSDOF 2` on
        // ethanol both have to survive, and `TORSDOF 2` has to be *read*, not
        // merely tolerated -- otherwise this test would also pass against the
        // old drop-the-value behaviour.
        for n in [0usize, 1, 2, 999] {
            let s = parse_pdbqt(&format!("{ETHANOL}TORSDOF {n}\n")).unwrap();
            assert_eq!(
                s.torsdof,
                Some(n),
                "`TORSDOF {n}` must be read as a declaration"
            );
        }
    }

    /// n-Butane: two torsions, so ROOT + 2 BRANCH sections.
    const BUTANE_POSE: &str = concat!(
        "REMARK VINA RESULT:    -3.100  0.000  0.000\n",
        "MODEL\n",
        "ROOT\n",
        "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n",
        "ATOM      2  C2  UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C \n",
        "ENDROOT\n",
        "BRANCH   2   3\n",
        "ATOM      3  C3  UNL     1       2.150   1.300   0.000  1.00  0.00     0.000 C \n",
        "ENDBRANCH\n",
        "BRANCH   3   4\n",
        "ATOM      4  C4  UNL     1       3.650   1.300   0.000  1.00  0.00     0.000 C \n",
        "ENDBRANCH\n",
        "TORSDOF 1\n",
        "ENDMDL\n",
    );

    #[test]
    fn a_written_pose_lists_every_atom_exactly_once() {
        // Regression test. `write_branch_tree` used to emit each branch's whole
        // subtree, so every atom below a torsion was written twice — once in
        // its own branch and once in the parent branch. The file still parsed,
        // which is exactly why it went unnoticed: the reader simply saw a
        // molecule with duplicated atoms.
        let s = parse_pdbqt(BUTANE_POSE).unwrap();
        let opts = PdbqtWriteOptions::default();
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &s.molecule.atoms.iter().map(|a| a.coord).collect::<Vec<_>>(),
            energy: -3.1,
            intermolecular: -3.1,
            intramolecular: 0.0,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let mut out: Vec<u8> = Vec::new();
        write_pose(&mut out, &pose, &s.molecule, &[(1, 2)], &opts).unwrap();
        let text = String::from_utf8(out).unwrap();

        let serials: Vec<String> = text
            .lines()
            .filter(|l| l.starts_with("ATOM"))
            .map(|l| l[6..11].trim().to_string())
            .collect();
        assert_eq!(serials.len(), 4, "expected 4 atom lines, got {serials:?}");
        let mut sorted = serials.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "atom serials were repeated: {serials:?}");

        // And the whole file must read back as the same molecule.
        let back = parse_pdbqt(&text).unwrap();
        assert_eq!(back.molecule.len(), 4);
    }

    #[test]
    fn negative_coordinates_parse() {
        let text =
            "ATOM      1  C1  UNL     1      -1.234   0.000   0.000  1.00  0.00     0.000 C \n";
        let s = parse_pdbqt(text).unwrap();
        assert!((s.molecule.atoms[0].coord[0] + 1.234).abs() < 1e-9);
    }

    #[test]
    fn multi_model_readback() {
        let text = concat!(
            "MODEL\n",
            "REMARK VINA RESULT:    -7.5  0.000  0.000\n",
            "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.090 C \n",
            "ENDMDL\n",
            "MODEL\n",
            "REMARK VINA RESULT:    -6.1  0.000  0.000\n",
            "ATOM      1  C1  UNL     1       9.000   0.000   0.000  1.00  0.00     0.090 C \n",
            "ENDMDL\n",
        );
        let s = parse_pdbqt(text).unwrap();
        assert!(s.multi_model);
        assert_eq!(s.vina_results.len(), 2);
        assert!(s.vina_results[0].contains("-7.5"));
    }

    /// A pose file is several *poses of one molecule*, and this function builds
    /// one molecule. It used to append every `MODEL`'s atoms to the same list,
    /// so the nine poses of `poses.pdbqt` parsed as one 144-atom molecule in
    /// which atoms from two different poses sat 0.06 A apart. The
    /// duplicate-atom check is right to reject that and did, which is how
    /// `Ligand.from_pdbqt` came to refuse the file this crate's own
    /// `write_pdbqt` had just written.
    #[test]
    fn a_pose_file_parses_as_its_first_pose_not_as_one_merged_molecule() {
        let s = parse_pdbqt(SIDE_PIECES).unwrap();
        let n = s.molecule.atoms.len();
        let pairs = s.active_torsion_pairs();
        let opts = PdbqtWriteOptions {
            multi_model: true,
            write_branch_tree: true,
            ..Default::default()
        };
        // Two poses of the same molecule into one file, displaced 5 A so that
        // merging them would be a plainly visible 2n-atom molecule.
        let mut buf: Vec<u8> = Vec::new();
        for (rank, shift) in [0.0_f64, 5.0].into_iter().enumerate() {
            let coords: Vec<Vec3> = s
                .molecule
                .atoms
                .iter()
                .map(|a| [a.coord[0] + shift, a.coord[1], a.coord[2]])
                .collect();
            let pose = PoseOutput {
                atoms: &s.molecule.atoms,
                coords: &coords,
                energy: -5.5 + rank as f64,
                intermolecular: -4.75,
                intramolecular: -0.75,
                rank: rank + 1,
                run: 1,
                rmsd_lb: None,
                rmsd_ub: None,
            };
            write_pose(&mut buf, &pose, &s.molecule, &pairs, &opts).unwrap();
        }
        let text = String::from_utf8(buf).unwrap();
        assert_eq!(
            text.lines().filter(|l| l.starts_with("MODEL")).count(),
            2,
            "the fixture must really be two MODEL blocks, or nothing below is tested:\n{text}"
        );

        let back = parse_pdbqt(&text).unwrap();
        assert!(
            back.multi_model,
            "the file did hold two poses, so the flag has to say so"
        );
        assert_eq!(
            back.molecule.atoms.len(),
            n,
            "two poses of a {n}-atom molecule parsed as {} atoms: the poses are being \
             merged into one molecule:\n{text}",
            back.molecule.atoms.len()
        );
        // The atoms that survive are the *first* pose's, not an interleaving of
        // both -- a count alone would not notice which pose came back.
        for (i, atom) in back.molecule.atoms.iter().enumerate() {
            assert!(
                (atom.coord[0] - s.molecule.atoms[i].coord[0]).abs() < 1e-3,
                "atom {i} came back at {:?}; the first pose has it at {:?}",
                atom.coord,
                s.molecule.atoms[i].coord
            );
        }
    }

    /// A per-model splitter collects only what lies between `MODEL` and
    /// `ENDMDL`, so an energy written *before* the first `MODEL` belongs to no
    /// pose and every pose reads back as 0.00. The workbench pose browser did
    /// exactly that: all nine rows showed "0.00 kcal/mol", and the "best" pose
    /// was whichever happened to be listed first.
    #[test]
    fn the_energy_remark_is_inside_the_model_block() {
        let s = parse_pdbqt(ETHANOL).unwrap();
        let coords: Vec<Vec3> = s.molecule.atoms.iter().map(|a| a.coord).collect();
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &coords,
            energy: -7.25,
            intermolecular: -6.0,
            intramolecular: -1.25,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let opts = PdbqtWriteOptions {
            multi_model: true,
            write_branch_tree: false,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_pose(&mut buf, &pose, &s.molecule, &[], &opts).unwrap();
        let text = String::from_utf8(buf).unwrap();

        let records: Vec<&str> = text.lines().collect();
        let model = records
            .iter()
            .position(|l| l.starts_with("MODEL"))
            .expect("MODEL record");
        let end = records
            .iter()
            .position(|l| l.starts_with("ENDMDL"))
            .expect("ENDMDL record");
        let remark = records
            .iter()
            .position(|l| l.contains("VINA RESULT"))
            .unwrap_or_else(|| panic!("no VINA RESULT in:\n{text}"));
        assert!(
            model < remark && remark < end,
            "the VINA RESULT remark must sit between MODEL and ENDMDL, got \
             MODEL at {model}, remark at {remark}, ENDMDL at {end}:\n{text}"
        );

        // And it must survive the round trip: two models in, two energies out.
        // With the remark written before the first MODEL only one result comes
        // back, because the leading block belongs to no model.
        let two = text.clone() + &text;
        let back = parse_pdbqt(&two).unwrap();
        assert_eq!(
            back.vina_results.len(),
            2,
            "both models must carry their own energy, got {:?}",
            back.vina_results
        );
        for r in &back.vina_results {
            assert!(
                r.contains("-7.2"),
                "the energy did not survive the round trip: {r:?}"
            );
        }
    }

    #[test]
    fn written_pose_round_trips() {
        let s = parse_pdbqt(ETHANOL).unwrap();
        let coords: Vec<Vec3> = s.molecule.atoms.iter().map(|a| a.coord).collect();
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &coords,
            energy: -7.25,
            intermolecular: -6.0,
            intramolecular: -1.25,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let opts = PdbqtWriteOptions {
            multi_model: true,
            write_branch_tree: false,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_pose(&mut buf, &pose, &s.molecule, &[], &opts).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("VINA RESULT"));
        assert!(text.contains("-7.2"));
        assert!(text.contains("MODEL"));
        let back = parse_pdbqt(&text).unwrap();
        assert_eq!(back.molecule.len(), 4);
        for (a, b) in s.molecule.atoms.iter().zip(back.molecule.atoms.iter()) {
            assert!((a.coord[0] - b.coord[0]).abs() < 1e-3);
        }
    }

    /// Six carbons in a chain with a 2.5 A gap between atoms 2 and 3, so the
    /// molecule is two rigid pieces joined at nothing the parser calls a bond,
    /// and the two `TORSION` records sit on the two *outer* bonds. The root
    /// cluster is then atoms 1, 3, 4, 5 -- not a prefix -- so writing the tree
    /// emits serials out of order and the file cannot be read in file order.
    ///
    /// The existing `written_pose_round_trips` cannot see any of this: it writes
    /// with `write_branch_tree: false` and an empty torsion list, so its records
    /// happen to be in serial order and the round trip is trivially correct.
    const SIDE_PIECES: &str = concat!(
        "ROOT\n",
        "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n",
        "ATOM      2  C2  UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C \n",
        "ENDROOT\n",
        "BRANCH   2   3\n",
        "ATOM      3  C3  UNL     1       4.000   0.000   0.000  1.00  0.00     0.000 C \n",
        "ATOM      4  C4  UNL     1       5.500   0.000   0.000  1.00  0.00     0.000 C \n",
        "ATOM      5  C5  UNL     1       7.000   0.000   0.000  1.00  0.00     0.000 C \n",
        "ENDBRANCH\n",
        "BRANCH   5   6\n",
        "ATOM      6  C6  UNL     1       8.500   0.000   0.000  1.00  0.00     0.000 C \n",
        "ENDBRANCH\n",
        "TORSDOF 2\n",
        "TORSION    1  A between atoms: C_1  and  C_2\n",
        "TORSION    2  A between atoms: C_5  and  C_6\n",
    );

    #[test]
    fn a_pose_file_declares_its_atom_order_and_is_read_in_that_order() {
        let s = parse_pdbqt(SIDE_PIECES).unwrap();
        let pairs = s.active_torsion_pairs();
        assert_eq!(
            pairs.len(),
            2,
            "the fixture must have two rotatable bonds or the tree is flat and the \
             whole test is vacuous"
        );
        let coords: Vec<Vec3> = s.molecule.atoms.iter().map(|a| a.coord).collect();
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &coords,
            energy: -5.5,
            intermolecular: -4.75,
            intramolecular: -0.75,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let opts = PdbqtWriteOptions {
            multi_model: true,
            write_branch_tree: true,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_pose(&mut buf, &pose, &s.molecule, &pairs, &opts).unwrap();
        let text = String::from_utf8(buf).unwrap();

        // **The claim, not a number: the two orders differ and the file says
        // which one to use.** Without this the round trip below would pass on
        // any file, including a writer that had been changed to emit serial
        // order -- at which point the declaration would be describing a
        // distinction that no longer exists.
        let serials: Vec<usize> = text
            .lines()
            .filter(|l| l.starts_with("ATOM"))
            .map(|l| l[6..11].trim().parse::<usize>().unwrap())
            .collect();
        assert_eq!(serials.len(), coords.len());
        assert_ne!(
            serials,
            (1..=coords.len()).collect::<Vec<usize>>(),
            "the fixture is supposed to produce a permuted file order, and it did \
             not: {serials:?}. Either the fixture stopped being a permutation or \
             the writer started emitting serial order, and this test has to notice \
             which"
        );
        assert!(
            text.contains(ATOM_ORDER_REMARK),
            "the file does not declare its atom order, which is the whole reason a \
             reader can be misled by it. Missing {ATOM_ORDER_REMARK:?} from:\n{text}"
        );

        // Now the round trip, through the reader that consumes this file. The
        // tolerance is the write precision, 3 decimal places.
        let path = std::env::temp_dir().join("opendock-atom-order-roundtrip.pdbqt");
        std::fs::write(&path, &text).unwrap();
        let back = read_docked_poses(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.len(), 1, "one MODEL in, one pose out");
        let read_back = &back[0].coords;
        assert_eq!(read_back.len(), coords.len());
        for (i, (a, b)) in coords.iter().zip(read_back.iter()).enumerate() {
            for k in 0..3 {
                assert!(
                    (a[k] - b[k]).abs() < 1e-3,
                    "atom {i} component {k}: wrote {:+.3}, read back {:+.3}. The \
                     reader has to reindex by the serial column, because the records \
                     are written in branch-tree order",
                    a[k],
                    b[k]
                );
            }
        }

        // The negative control, so the check above cannot be satisfied by a
        // reader that happens to agree with the writer by accident: reading the
        // same records in file order is a *different molecule*, and by how much
        // is the size of the hazard the declaration exists to remove.
        let in_file_order: Vec<Vec3> = serials.iter().map(|&serial| coords[serial - 1]).collect();
        let worst = in_file_order
            .iter()
            .zip(read_back.iter())
            .map(|(a, b)| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            worst > 0.5,
            "file order and serial order differ by at most {worst:.4} A on this \
             fixture, so the declaration is describing a distinction too small to \
             matter. The serials were {serials:?}"
        );
    }

    #[test]
    fn a_reader_that_gets_no_usable_serials_keeps_the_file_order() {
        // The documented branch of `reindex_by_serial`: a model whose serials
        // are not exactly 1..n is passed through in the order it was written,
        // because sorting by a numbering that is not the atom index would
        // substitute one wrong answer for another.
        let plain: Vec<(usize, Vec3)> = vec![
            (1, [0.0, 0.0, 0.0]),
            (7, [1.0, 0.0, 0.0]),
            (3, [2.0, 0.0, 0.0]),
        ];
        let out = reindex_by_serial(plain);
        assert_eq!(
            out,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "a numbering that is not 1..n must be left alone, not partially sorted"
        );
        // A complete 1..n run is reindexed even when it arrives shuffled, and
        // the coordinate that comes back at slot k is the one whose serial was
        // k -- which is the whole point, so the fixture puts a *distinct*
        // coordinate on every serial and shuffles the arrival.
        let complete: Vec<(usize, Vec3)> = vec![
            (3, [2.0, 0.0, 0.0]),
            (1, [0.0, 0.0, 0.0]),
            (2, [1.0, 0.0, 0.0]),
        ];
        assert_eq!(
            reindex_by_serial(complete),
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "a complete 1..n run is reindexed even when it arrives shuffled"
        );
    }

    #[test]
    fn mismatched_coordinate_count_is_rejected() {
        let s = parse_pdbqt(ETHANOL).unwrap();
        let coords: Vec<Vec3> = vec![[0.0; 3]; 2];
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &coords,
            energy: -1.0,
            intermolecular: -1.0,
            intramolecular: 0.0,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let opts = PdbqtWriteOptions::default();
        let mut buf: Vec<u8> = Vec::new();
        assert!(write_pose(&mut buf, &pose, &s.molecule, &[], &opts).is_err());
    }

    #[test]
    fn whitespace_fallback_line_parses() {
        // Deliberately unaligned, 9+ whitespace separated fields.
        let text = "ATOM 1 C1 UNL 1 1.0 2.0 3.0 1.00 0.00 0.05 C\n";
        let s = parse_pdbqt(text).unwrap();
        assert!((s.molecule.atoms[0].coord[1] - 2.0).abs() < 1e-9);
        assert_eq!(s.molecule.atoms[0].atom_type, AtomType::CH);
    }

    /// A pose file must say which atoms are bonded, and it must say it in a way
    /// that survives the file not being in serial order.
    #[test]
    fn a_written_pose_declares_its_bonds_keyed_by_serial() {
        let s = parse_pdbqt(ETHANOL).unwrap();
        let coords: Vec<Vec3> = s.molecule.atoms.iter().map(|a| a.coord).collect();
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &coords,
            energy: -7.25,
            intermolecular: -6.0,
            intramolecular: -1.25,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let opts = PdbqtWriteOptions {
            multi_model: true,
            write_branch_tree: false,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_pose(&mut buf, &pose, &s.molecule, &[], &opts).unwrap();
        let text = String::from_utf8(buf).unwrap();

        // C1-C2, C2-O1, O1-H1: three bonds over four atoms.
        let declared: Vec<(usize, usize)> = text
            .lines()
            .filter(|l| l.starts_with(CONNECTIVITY_BOND_REMARK))
            .map(|l| {
                let mut it = l.split_whitespace().skip(2);
                let a: usize = it.next().unwrap().parse().unwrap();
                let b: usize = it.next().unwrap().parse().unwrap();
                (a, b)
            })
            .collect();
        assert_eq!(
            declared,
            vec![(1, 2), (2, 3), (3, 4)],
            "the record must be the molecule's own bonds, as serials:\n{text}"
        );
        assert!(
            text.contains("REMARK OD_NBONDS 3"),
            "the bond count must be stated:\n{text}"
        );

        // Serials, not positions. The atoms are written in the same order here,
        // so the two readings coincide -- which is exactly why the *file*
        // round trip in `scripts/torsion_bond_record_check.py` is the one that
        // can tell them apart, on a pose whose order is 5,6,7,8,9,10,4,...
        let serials: Vec<usize> = text
            .lines()
            .filter(|l| l.starts_with("ATOM"))
            .map(|l| l[6..11].trim().parse().unwrap())
            .collect();
        assert_eq!(serials, vec![1, 2, 3, 4]);
        for (a, b) in &declared {
            assert!(serials.contains(a) && serials.contains(b));
        }

        // And it must not disturb the reader: the file is still one molecule
        // with the same atoms, because the record is a REMARK and carries no
        // coordinates.
        let back = parse_pdbqt(&text).unwrap();
        assert_eq!(back.molecule.len(), 4);
    }

    /// The record has to be inside the `MODEL` block, for the same reason the
    /// energy remark is: a per-model splitter collects only what lies between
    /// `MODEL` and `ENDMDL`, so a bond record written outside belongs to no
    /// pose and every pose reads back with no connectivity at all.
    #[test]
    fn the_bond_record_is_inside_the_model_block() {
        let s = parse_pdbqt(ETHANOL).unwrap();
        let coords: Vec<Vec3> = s.molecule.atoms.iter().map(|a| a.coord).collect();
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &coords,
            energy: -7.25,
            intermolecular: -6.0,
            intramolecular: -1.25,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let opts = PdbqtWriteOptions {
            multi_model: true,
            write_branch_tree: false,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_pose(&mut buf, &pose, &s.molecule, &[], &opts).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let model = lines
            .iter()
            .position(|l| l.starts_with("MODEL"))
            .expect("MODEL record");
        let end = lines
            .iter()
            .position(|l| l.starts_with("ENDMDL"))
            .expect("ENDMDL record");
        let record = lines
            .iter()
            .position(|l| l.starts_with(CONNECTIVITY_COUNT_REMARK))
            .unwrap_or_else(|| panic!("no connectivity record in:\n{text}"));
        assert!(
            model < record && record < end,
            "the connectivity record must sit between MODEL and ENDMDL, got \
             MODEL at {model}, record at {record}, ENDMDL at {end}:\n{text}"
        );
    }

    /// A molecule with no bonds still says so, rather than writing nothing: a
    /// reader has to be able to tell "this writer ran and there are no bonds"
    /// from "this file predates the record", and only the count line does that.
    #[test]
    fn a_bondless_molecule_still_states_a_count() {
        let text =
            "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n";
        let s = parse_pdbqt(text).unwrap();
        let coords: Vec<Vec3> = s.molecule.atoms.iter().map(|a| a.coord).collect();
        let pose = PoseOutput {
            atoms: &s.molecule.atoms,
            coords: &coords,
            energy: -1.0,
            intermolecular: -1.0,
            intramolecular: 0.0,
            rank: 1,
            run: 1,
            rmsd_lb: None,
            rmsd_ub: None,
        };
        let opts = PdbqtWriteOptions::default();
        let mut buf: Vec<u8> = Vec::new();
        write_pose(&mut buf, &pose, &s.molecule, &[], &opts).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(
            text.contains("REMARK OD_NBONDS 0"),
            "a bondless molecule must still carry a count:\n{text}"
        );
        assert!(
            !text
                .lines()
                .any(|l| l.starts_with(CONNECTIVITY_BOND_REMARK)),
            "and no bond records behind it:\n{text}"
        );
    }
}
