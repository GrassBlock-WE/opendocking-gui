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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
pub fn parse_pdbqt(text: &str) -> Result<ParsedStructure> {
    let mut out = ParsedStructure::default();
    let mut atoms: Vec<Atom> = Vec::new();
    let mut serial_to_index: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut branch_depth: Vec<usize> = Vec::new();
    let mut depth: usize = 0;
    let mut next_serial: u32 = 1;

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
                if let Some(v) = line.split_whitespace().nth(1) {
                    if let Ok(v) = v.parse::<usize>() {
                        out.torsdof = Some(v);
                    }
                }
            }
            "BRANCH" => {
                depth += 1;
            }
            "ENDBRANCH" => {
                depth = depth.saturating_sub(1);
            }
            "MODEL" => {
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
    out.branch_depth = branch_depth;

    let mut mol = Molecule::from_atoms(atoms)?;
    out.molecule = std::mem::take(&mut mol);

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
    let x: f64 = slice_parse(line, 30, 38)?;
    let y: f64 = slice_parse(line, 38, 46)?;
    let z: f64 = slice_parse(line, 46, 54)?;
    if x.is_finite() && y.is_finite() && z.is_finite() {
        Some((x, y, z))
    } else {
        None
    }
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

/// Read every `MODEL` of a multi-model PDBQT file, returning one pose per model.
pub fn read_docked_poses(path: impl AsRef<Path>) -> Result<Vec<DockedPose>> {
    let file = fs::File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut poses: Vec<DockedPose> = Vec::new();
    let mut cur: Vec<Vec3> = Vec::new();
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
                if let Some(c) = fixed_coord(&line) {
                    cur.push([c.0, c.1, c.2]);
                }
            }
            "ENDMDL" => {
                poses.push(DockedPose {
                    coords: std::mem::take(&mut cur),
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
        let text = format!("{ETHANOL}TORSDOF 1\nTORSION    1  A between atoms: C_1  and  C_99\n");
        let s = parse_pdbqt(&text).unwrap();
        assert_eq!(s.molecule.len(), 4);
        assert!(s.active_torsion_pairs().is_empty());
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(parse_pdbqt("REMARK nothing here\n").is_err());
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
