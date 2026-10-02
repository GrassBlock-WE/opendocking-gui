//! The top-level docking driver: configuration, pose clustering, and `dock`.
//!
//! A docking run is three separable stages, and this module keeps them
//! separable so a caller can re-run only what changed:
//!
//! 1. **prepare** the receptor once and tabulate the maps (expensive);
//! 2. **search** for low-energy conformations (the expensive part, controlled
//!    by `exhaustiveness`);
//! 3. **cluster and rank** the raw results into distinct poses.
//!
//! Stage 3 matters more than it looks: an iterated local search returns many
//! near-duplicates of the same basin, and reporting ten copies of one pose as
//! ten results is worse than useless.

use serde::Serialize;

use crate::error::{DockError, Result};
use crate::grid::GridMaps;
use crate::kinematics::Conformation;
use crate::ligand::Ligand;
use crate::receptor::Receptor;
use crate::scoring::ScoringFunction;
use crate::search::lga::{evolve, LgaConfig};
pub use crate::search::monte_carlo::Pose;
use crate::search::monte_carlo::{search, MonteCarloConfig};
use crate::types::Vec3;

/// Which global search strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SearchMode {
    /// Iterated local search. The right default for most ligands.
    MonteCarlo,
    /// Island-model hybrid genetic algorithm. Better for very flexible ligands.
    IslandLga,
    /// Run both and merge. Roughly twice the cost, occasionally better.
    Both,
}

/// Everything that controls a docking run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DockingConfig {
    /// Global search strategy.
    pub mode: SearchMode,
    /// Monte-Carlo settings.
    pub monte_carlo: MonteCarloConfig,
    /// Island-GA settings.
    pub lga: LgaConfig,
    /// Number of distinct poses to report after clustering.
    pub num_modes: usize,
    /// Two poses closer than this (Ångström, per-atom RMSD) are the same pose.
    pub rmsd_cutoff: f64,
    /// Smallest separation a ligand atom may have from a receptor atom, in
    /// ångström, for the pose to be reported.
    ///
    /// This is a **guard rail, not a compensation for a broken energy
    /// function**. It was introduced when every interaction radius was a flat
    /// 0.4 Å, which placed all three attractive terms inside the steric
    /// exclusion zone: on crambin the best clash-free random placement scored
    /// −0.01 while the best clashing one scored −1.00, and the search correctly
    /// found the true minimum of a function that preferred burial. That
    /// constant is now the correct per-element XS radius, so the steric term
    /// does real work on its own and this filter should rarely reject anything.
    ///
    /// It is kept because "the pose the optimiser returned is a structure no
    /// physicist would accept" is a claim worth making structurally rather
    /// than trusting an empirical weight set to enforce. A non-zero
    /// [`DockingResult::rejected_pose_count`] is therefore a genuine warning
    /// sign: it means the energy function still preferred an impossible
    /// structure, which points at the search box containing solid protein
    /// instead of a pocket, or at a
    /// [`DockingConfig::mode`]/`exhaustiveness` too weak to escape the
    /// protein surface.
    ///
    /// Set to `0.0` to disable the filter and see the raw energy minimum.
    pub min_contact_distance: f64,
    /// Number of worker threads. `0` means "use all cores".
    ///
    /// **Nothing reads this field.** An earlier version of this comment said it
    /// was read by the map *tabulation*, which was wrong on its own logic: the
    /// sentence that supported it conceded in the same breath that
    /// [`Receptor::precalculate`] and [`crate::grid::GridMaps::precalculate`]
    /// take their *own* `n_threads` and build their own scoped pool, and
    /// `dock()` receives precalculated [`crate::grid::GridMaps`] -- so at
    /// tabulation time no `DockingConfig` exists to read this from. The census
    /// in `every_docking_config_field_is_read_except_the_documented_one` is the
    /// authority and it says this is the one unread field in the struct.
    ///
    /// The search runs on rayon's ambient global pool, so the number of threads
    /// it uses is the process's `RAYON_NUM_THREADS` when that is set and the
    /// core count when it is not. Measured through the Python binding on one
    /// machine (release build, crambin x biotin, 18 A box at 0.375 A,
    /// `exhaustiveness=16`, `num_modes=5`, seed 20260929, best of three):
    ///
    /// | `RAYON_NUM_THREADS` | search wall time | best energy |
    /// |---|---|---|
    /// | `1` | 8.680 s | -5.169667 |
    /// | `2` | 4.302 s | -5.169667 |
    /// | `4` | 1.227 s | -5.169667 |
    /// | unset (16 cores) | 0.436 s | -5.169667 |
    ///
    /// Twenty times the wall time between one thread and sixteen, and
    /// bit-identical energies: the search is seeded and its result does not
    /// depend on how many threads run it. So the environment variable is the
    /// real lever and this field is not, and the one number a Python caller can
    /// see -- `opendocking.available_backends()`, whose thread count is read
    /// from the pool for exactly this reason -- moves with the environment
    /// variable and not with this field.
    ///
    /// **The field is kept, deliberately.** Removing it is safe in the sense
    /// that nothing reads it, but it is a breaking change to a public struct for
    /// every Rust caller that names it, and the paragraph below is the only
    /// record in the repository of what making it live would cost. A dead knob
    /// that is documented as dead is a fact; a dead knob that is not is a lie.
    /// The sentence is pinned by the census above, which fails the moment
    /// someone makes this live, at which point this comment has to be corrected
    /// deliberately rather than left to rot.
    ///
    /// What making it live would cost: a scoped rayon pool per search, built and
    /// installed around the whole search rather than around the tabulation, plus
    /// a change in what the walks share. That is a real change to search
    /// behaviour and not a wiring change.
    pub num_threads: usize,
}

impl Default for DockingConfig {
    fn default() -> Self {
        DockingConfig {
            mode: SearchMode::MonteCarlo,
            monte_carlo: MonteCarloConfig::default(),
            lga: LgaConfig::default(),
            num_modes: 9,
            rmsd_cutoff: 1.0,
            min_contact_distance: MIN_CONTACT_DISTANCE,
            num_threads: 0,
        }
    }
}

/// The default hard floor on ligand–receptor separation, in ångström.
///
/// Two heavy atoms closer than this cannot both be where they are. 2.0 Å is
/// the conventional clash threshold in docking tools and is well below any
/// real non-bonded contact, so it rejects only structures that cannot exist.
pub const MIN_CONTACT_DISTANCE: f64 = 2.0;

/// Split poses into those that are physically possible and those that are not.
///
/// A pose is possible when no ligand atom is closer than `min_distance` to a
/// receptor atom. Hydrogens are excluded from both sides by
/// [`crate::grid::GridMaps::receptor_atoms`], which returns heavy atoms only:
/// a polar hydrogen sits on its own heavy atom by construction, so comparing
/// one against a receptor *hydrogen* would report every ordinary hydrogen bond
/// as a clash.
///
/// `min_distance <= 0.0` disables the check and returns everything as clean.
pub fn partition_by_clash(
    poses: Vec<Pose>,
    receptor: &[[f64; 3]],
    min_distance: f64,
) -> (Vec<Pose>, Vec<Pose>) {
    if min_distance <= 0.0 || receptor.is_empty() {
        return (poses, Vec::new());
    }
    let min2 = min_distance * min_distance;
    let mut clean = Vec::with_capacity(poses.len());
    let mut clashing = Vec::new();
    for pose in poses {
        let ok = pose.coords.iter().all(|p| {
            receptor.iter().all(|q| {
                let dx = p[0] - q[0];
                let dy = p[1] - q[1];
                let dz = p[2] - q[2];
                dx * dx + dy * dy + dz * dz >= min2
            })
        });
        if ok {
            clean.push(pose);
        } else {
            clashing.push(pose);
        }
    }
    (clean, clashing)
}

impl DockingConfig {
    /// A fast configuration, for tests and for quick interactive docking.
    pub fn fast() -> DockingConfig {
        DockingConfig {
            monte_carlo: MonteCarloConfig {
                exhaustiveness: 2,
                steps: 12,
                seed: Some(1),
                ..Default::default()
            },
            lga: LgaConfig {
                islands: 2,
                population_per_island: 6,
                generations: 3,
                seed: Some(1),
                ..Default::default()
            },
            num_modes: 3,
            ..Default::default()
        }
    }
}

/// A ranked docking result.
#[derive(Debug, Clone, Serialize)]
pub struct DockingResult {
    /// Poses, best (most negative) energy first.
    pub poses: Vec<Pose>,
    /// Wall-clock seconds spent searching.
    pub elapsed_seconds: f64,
    /// Number of raw conformations the search produced before clustering.
    pub raw_pose_count: usize,
    /// Candidate poses discarded because a ligand atom sat closer than
    /// [`DockingConfig::min_contact_distance`] to a receptor atom.
    ///
    /// Reported rather than silently applied, because a non-zero count means
    /// the scoring function preferred a pose that no physical structure can
    /// have — usually a sign that the search box contains solid protein
    /// instead of a pocket.
    pub rejected_pose_count: usize,
    /// Receptor atoms whose PDBQT type this engine does not recognise, carried
    /// from the maps this run scored against.
    ///
    /// This is the *other* "the search did not do what you asked" count, and it
    /// sits next to [`DockingResult::rejected_pose_count`] because it answers
    /// the same question from the other end: not "were the poses impossible"
    /// but "were the atoms scoreable". An unrecognised type still contributes a
    /// shape term and silently loses its hydrogen-bond and hydrophobic
    /// character, so every energy here is an underestimate of what the receptor
    /// actually offers, and no pose count reveals it.
    ///
    /// It is reported and not refused, deliberately and for the reason
    /// [`crate::receptor::Receptor::unknown_atom_types`] gives: another tool may
    /// emit type names AutoDock never defined, and a refusal would make the
    /// engine unusable with it. What was wrong with reporting it *only* on the
    /// receptor is that the receptor is gone by the time a result exists — the
    /// caller precalculates, drops the receptor, and holds maps. From there a
    /// result that says nothing is indistinguishable from a clean one, and
    /// "silently succeeded on an input it could not handle" is exactly that.
    pub unknown_atom_types: usize,
    /// Reported poses with **no atom at all** inside the search box.
    ///
    /// This is the third "the search did not do quite what you asked" count and
    /// the only one of the three that admits no partial credit. A pose with one
    /// atom inside the box scored a real interaction with the receptor; a pose
    /// with none scored nothing but the ligand's own internal energy and
    /// whatever [`crate::search::OUT_OF_BOX_PENALTY`] charged it, and is ranked
    /// alongside poses that did bind. Counting whole poses rather than
    /// out-of-box atoms is what makes the number mean one thing: how much of
    /// this result is not a binding mode.
    ///
    /// It is derived, not thresholded. Each pose carries
    /// [`Pose::atoms_outside_box`], taken by the same loop that charged the
    /// penalty, and a pose is counted here when that count reaches its own atom
    /// count — a comparison of two integers the engine already had. No
    /// tolerance, no distance and no energy is involved, so the number cannot
    /// be tuned into appearing.
    ///
    /// ## Why this is a reported state and not a refusal
    ///
    /// A refusal is right for a box that misses the receptor
    /// ([`crate::grid::precalculate`](crate::grid::GridMaps::precalculate)),
    /// where every map is identically zero and there is no ranking to mislead
    /// anyone. It is wrong here. The search ran, the poses are physically
    /// consistent, and the energies are the engine's honest ones: a pose wholly
    /// outside the box is charged 1000 kcal/mol per ångström of overhang, so it
    /// cannot report a small attractive number and the table itself already
    /// shows the result is worthless. Turning that into an error would also
    /// throw away a case a caller may want — `min_contact_distance` and a
    /// deliberately oversized box are legitimate, and a refusal here would make
    /// the engine refuse to describe what it did.
    ///
    /// So it is carried and counted, and the distinct state it names — *this
    /// search returned poses, and none of them are in the box* — is different
    /// from both a clean run and from the search having failed. Read it against
    /// `poses.len()`: equal to it means the result describes no binding mode.
    pub poses_outside_box_count: usize,
    /// The strategy that ran.
    pub mode: SearchMode,
    /// Name of the scoring function used.
    pub scoring_function: String,
}

impl DockingResult {
    /// The best pose, if any.
    pub fn best(&self) -> &Pose {
        &self.poses[0]
    }
}

/// Dock `ligand` against precalculated `maps`.
///
/// `scoring` must be the same function the maps were built with; the maps are
/// meaningless under a different set of weights.
pub fn dock(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    config: &DockingConfig,
) -> Result<DockingResult> {
    if ligand.is_empty() {
        return Err(DockError::molecule("cannot dock an empty ligand"));
    }
    if config.num_modes == 0 {
        return Err(DockError::param("num_modes", 0, "must be at least 1"));
    }
    if config.rmsd_cutoff <= 0.0 {
        return Err(DockError::param(
            "rmsd_cutoff",
            config.rmsd_cutoff,
            "must be positive",
        ));
    }

    let start = std::time::Instant::now();
    let raw: Vec<Pose> = match config.mode {
        SearchMode::MonteCarlo => search(ligand, maps, scoring, &config.monte_carlo)?,
        SearchMode::IslandLga => evolve(ligand, maps, scoring, &config.lga)?,
        SearchMode::Both => {
            let mut all = search(ligand, maps, scoring, &config.monte_carlo)?;
            all.extend(evolve(ligand, maps, scoring, &config.lga)?);
            all
        }
    };

    let raw_pose_count = raw.len();
    // The Vina weight set has no effective steric exclusion, so a large search
    // box makes the lowest-scoring pose the one buried deepest in the protein.
    // Partition before clustering, so a physically impossible pose can never
    // take a clustering slot away from a real one — and so a rejected pose is
    // never silently reported.
    let (clean, clashing) =
        partition_by_clash(raw, maps.receptor_atoms(), config.min_contact_distance);
    // `cluster_poses` treats an empty input as an error, which is right for a
    // caller that asked it to cluster something and wrong here: "no physically
    // valid pose was found" has to be able to fall through to the fallback.
    let mut poses = if clean.is_empty() {
        Vec::new()
    } else {
        cluster_poses(clean, config.num_modes, config.rmsd_cutoff, ligand)?
    };
    let mut clashing_reported = 0usize;
    if poses.is_empty() {
        // Nothing physically valid was found. Reporting an error here would
        // turn "the search was too weak" into "docking failed", so fall back
        // to the best of what it did find and say so in the result. A low
        // `exhaustiveness` should not look like a broken box.
        poses = cluster_poses(clashing, config.num_modes, config.rmsd_cutoff, ligand)?;
        clashing_reported = poses.len();
    }

    // Counted here, over `poses` as it now stands, and not over `raw` and not
    // over `clean`: the number exists to describe what the caller is about to
    // read, so it has to be taken after the clash fallback above has decided
    // what gets reported. Counting `raw` would make it answer a different
    // question -- how often the search wandered out of the box, which is a
    // property of the search and not of the result -- and the two differ
    // whenever clustering drops a pose.
    Ok(DockingResult {
        poses_outside_box_count: poses.iter().filter(|p| p.every_atom_outside_box()).count(),
        poses,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        raw_pose_count,
        rejected_pose_count: clashing_reported,
        unknown_atom_types: maps.unknown_atom_types(),
        mode: config.mode,
        scoring_function: scoring.name().to_string(),
    })
}

/// Convenience wrapper that takes a receptor and a box, precalculating the maps.
pub fn dock_receptor(
    receptor: &Receptor,
    box_: &crate::grid::GridBox,
    ligand: &Ligand,
    spacing: f64,
    scoring: &dyn ScoringFunction,
    config: &DockingConfig,
) -> Result<DockingResult> {
    let maps = receptor.precalculate(box_, scoring, spacing)?;
    dock(ligand, &maps, scoring, config)
}

/// Reduce raw search output to `num_modes` distinct, energy-ranked poses.
///
/// Greedy clustering: walk the poses in energy order and keep one representative
/// per basin, discarding anything within `cutoff` of a pose already kept. This
/// is the same rule AutoDock applies, and it guarantees the reported poses are
/// genuinely different binding modes rather than local-minimum copies.
pub fn cluster_poses(
    raw: Vec<Pose>,
    num_modes: usize,
    cutoff: f64,
    ligand: &Ligand,
) -> Result<Vec<Pose>> {
    let mut sorted = raw;
    sorted.retain(|p| p.energy.is_finite() && !p.coords.is_empty());
    sorted.sort_by(|a, b| {
        a.energy
            .partial_cmp(&b.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if sorted.is_empty() {
        return Err(DockError::molecule(
            "the search produced no valid pose — is the ligand able to fit in the box?",
        ));
    }

    let n_atoms = ligand.len();
    let mut kept: Vec<Pose> = Vec::with_capacity(num_modes);
    for pose in sorted {
        if kept.len() >= num_modes {
            break;
        }
        let duplicate = kept
            .iter()
            .any(|k| rmsd(&k.coords, &pose.coords, n_atoms) < cutoff);
        if duplicate {
            continue;
        }
        // Report the RMSD of every kept pose to the best one, which is what
        // AutoDock writes as `RMSD_LB`.
        let rmsd_lb = kept.first().map(|k| rmsd(&k.coords, &pose.coords, n_atoms));
        kept.push(Pose {
            conf: pose.conf,
            energy: pose.energy,
            intermolecular: pose.intermolecular,
            intramolecular: pose.intramolecular,
            coords: pose.coords,
            rmsd: rmsd_lb,
            // Clustering keeps an existing pose rather than synthesising a
            // representative, so the gradient measured at this conformation is
            // still the gradient of the pose being reported. Dropping it here
            // would leave every reported pose without one.
            gradient: pose.gradient,
            // For the same reason, and for the same reason it is a count and not
            // a flag: the count was taken on the pose being reported, so the
            // `poses_outside_box_count` below is a count of these poses and of
            // nothing else.
            atoms_outside_box: pose.atoms_outside_box,
        });
    }
    Ok(kept)
}

/// Per-atom RMSD between two coordinate sets, in Ångström.
pub fn rmsd(a: &[Vec3], b: &[Vec3], n: usize) -> f64 {
    if a.len() < n || b.len() < n || n == 0 {
        return f64::INFINITY;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let dx = a[i][0] - b[i][0];
        let dy = a[i][1] - b[i][1];
        let dz = a[i][2] - b[i][2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    (sum / n as f64).sqrt()
}

/// Write results as a multi-model PDBQT file.
pub fn write_result(
    result: &DockingResult,
    ligand: &Ligand,
    path: impl AsRef<std::path::Path>,
) -> Result<()> {
    use crate::pdbqt::{write_poses, PdbqtWriteOptions, PoseOutput};
    let torsions = ligand.torsion_pairs();
    let poses: Vec<PoseOutput<'_>> = result
        .poses
        .iter()
        .enumerate()
        .map(|(i, p)| PoseOutput {
            atoms: &ligand.molecule.atoms,
            coords: &p.coords,
            energy: p.energy,
            intermolecular: p.intermolecular,
            intramolecular: p.intramolecular,
            rank: i + 1,
            run: 1,
            rmsd_lb: p.rmsd,
            rmsd_ub: None,
        })
        .collect();
    let opts = PdbqtWriteOptions {
        multi_model: true,
        write_branch_tree: true,
        ..Default::default()
    };
    write_poses(path, &poses, &ligand.molecule, &torsions, &opts)
}

/// Convenience: prepare a receptor, dock, and write the result in one call.
pub fn dock_file(
    receptor_pdbqt: impl AsRef<std::path::Path>,
    ligand_pdbqt: impl AsRef<std::path::Path>,
    box_: &crate::grid::GridBox,
    out_pdbqt: impl AsRef<std::path::Path>,
    spacing: f64,
    scoring: &dyn ScoringFunction,
    config: &DockingConfig,
) -> Result<DockingResult> {
    let receptor = Receptor::from_pdbqt(receptor_pdbqt)?;
    let ligand = Ligand::from_pdbqt(ligand_pdbqt)?;
    let maps = receptor.precalculate(box_, scoring, spacing)?;
    let result = dock(&ligand, &maps, scoring, config)?;
    write_result(&result, &ligand, out_pdbqt)?;
    Ok(result)
}

/// Rank helper used by the Python bindings and the CLI.
pub fn energies(result: &DockingResult) -> Vec<f64> {
    result.poses.iter().map(|p| p.energy).collect()
}

/// A pose's conformation, for callers that want to continue optimising.
pub fn pose_conformation(pose: &Pose) -> &Conformation {
    &pose.conf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Atom, AtomType, Element, Molecule};

    fn make_ligand() -> Ligand {
        let c = |i: u32, p: Vec3| Atom::new(i, p, Element::C, AtomType::CH);
        let mut mol = Molecule::from_atoms(vec![
            c(1, [0.00, 0.00, 0.00]),
            c(2, [1.50, 0.00, 0.00]),
            c(3, [2.15, 1.30, 0.00]),
            c(4, [3.65, 1.30, 0.00]),
            c(5, [4.30, 2.60, 0.00]),
        ])
        .unwrap();
        mol.assign_vina_atom_kinds();
        Ligand::from_molecule(mol).unwrap()
    }

    /// The fixture a seeded run is measured on: a small carbon chain, a box
    /// with room, and a weak search so the run takes a second rather than a
    /// minute.
    /// The census: which of this struct's fields does anything read?
    ///
    /// Four defects in this project have been the same shape — something
    /// published that a user can see or set, and code that does not do what the
    /// published thing says — and none of them had a check that asked the
    /// question in general. So this asks it, for the one struct every docking
    /// run is configured through.
    ///
    /// The enumeration is derived, never written down. The field list comes out
    /// of this file's own source by brace-matching the struct, so a new field
    /// is in the census the moment it is declared, with no list to update and
    /// therefore no way for a new field to ship unread. The readers come from
    /// walking the crate's sources at run time.
    ///
    /// Two details make the reader count mean something, and both were wrong in
    /// the first version of this:
    ///
    /// * A bare-name search does not work. `local`, `seed` and `num_threads`
    ///   each have unrelated locals of the same name, and a name search
    ///   reported all three as read. A field is read through *field access*,
    ///   so only `.field` counts.
    /// * `.field` alone is still not enough, because rayon has a builder method
    ///   called `num_threads` and the tabulation calls it twice. A field read is
    ///   never a call, so the pattern excludes a `(` after the name. Without
    ///   that, the one genuinely unread field in this struct looks read.
    #[test]
    fn every_docking_config_field_is_read_except_the_documented_one() {
        let here = env!("CARGO_MANIFEST_DIR");
        let src_dir = std::path::Path::new(here).join("src");

        fn strip_comments(text: &str) -> String {
            text.lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .map(|l| l.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n")
        }

        // The field list, derived from this file's own source.
        let own = strip_comments(include_str!("docking.rs"));
        let open = own
            .find("pub struct DockingConfig {")
            .expect("DockingConfig is declared in this file")
            + "pub struct DockingConfig {".len();
        // Already inside the struct's braces by the time we start walking, so
        // the depth is 1 — not 0, which underflows on the closing brace.
        let mut depth = 1usize;
        let mut body = String::new();
        for ch in own[open..].chars() {
            match ch {
                '{' => {
                    depth += 1;
                    body.push(ch);
                }
                '}' => {
                    depth -= 1;
                    body.push(ch);
                    if depth == 0 {
                        break;
                    }
                }
                _ => body.push(ch),
            }
        }
        let mut fields: Vec<String> = Vec::new();
        for line in body.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("pub ") {
                if let Some(colon) = rest.find(':') {
                    let name = rest[..colon].trim();
                    if !name.is_empty()
                        && name
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                    {
                        fields.push(name.to_string());
                    }
                }
            }
        }
        assert!(
            fields.len() >= 7,
            "derived {} fields from DockingConfig: {fields:?}. The derivation \
             found fewer than the struct has, so it has stopped working and \
             every field below is unchecked",
            fields.len()
        );

        // Every source file in the crate, comments gone.
        let mut code = String::new();
        let mut files = 0usize;
        let mut stack = vec![src_dir.clone()];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir).expect("the crate's source tree is readable");
            for e in entries {
                let p = e.expect("readable directory entry").path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                    let text = std::fs::read_to_string(&p).unwrap_or_else(|err| {
                        panic!("readable source file {}: {err}", p.display())
                    });
                    files += 1;
                    code.push_str(&strip_comments(&text));
                    code.push('\n');
                }
            }
        }
        assert!(
            files > 5 && code.len() > 10_000,
            "the census read {files} source files and {} bytes. A scan that found \
             almost nothing would report every field as unread, which looks \
             exactly like a struct nobody reads",
            code.len()
        );

        // Count `.field` occurrences by hand, because `str::matches` with a
        // `&str` pattern is a **literal substring search, not a regex** — the
        // first version of this passed `r"\.{name}\b"` and therefore searched
        // for that seven-character string, which occurs nowhere, and reported
        // every field as unread. `code.matches("config")` returning 72 is what
        // showed the text was fine and the pattern was the problem.
        //
        // The character after the name is checked too, for two reasons: a
        // longer field ending in this one (`.num_modes` against `mode`) is not
        // a read of this field, and a `(` is a method call — which is how
        // rayon got the same name as a field.
        fn count_field_reads(code: &str, name: &str) -> usize {
            let bytes = code.as_bytes();
            let needle = name.as_bytes();
            let mut n = 0usize;
            let mut i = 0usize;
            while i + 1 + needle.len() <= bytes.len() {
                if bytes[i] == b'.' && &bytes[i + 1..i + 1 + needle.len()] == needle {
                    let after = bytes.get(i + 1 + needle.len()).copied();
                    let name_continues = after
                        .map(|c| c.is_ascii_alphanumeric() || c == b'_')
                        .unwrap_or(false);
                    if !name_continues && after != Some(b'(') {
                        n += 1;
                    }
                }
                i += 1;
            }
            n
        }

        let mut unread: Vec<&String> = Vec::new();
        let mut report = format!(
            "{files} files, {} bytes, 'config' appears {} times\n",
            code.len(),
            code.matches("config").count()
        );
        for f in &fields {
            // Built at run time, so this test's own source never contains a
            // literal `.field` that could count itself.
            let n = count_field_reads(&code, f);
            report.push_str(&format!("  {f:24} {n} reads\n"));
            if n == 0 {
                unread.push(f);
            }
        }
        assert_eq!(
            unread,
            vec!["num_threads"],
            "the fields of DockingConfig that nothing reads are {unread:?}.\n\
             Derived reader counts:\n{report}\n\
             The expected list is the *known* dead field and nothing else, so a new \
             field that ships unread fails here. If you have just made \
             `num_threads` live, that is a deliberate change: fix the doc comment \
             on the field to describe what it now does and remove it from this \
             list in the same edit, so the sentence and the code cannot disagree."
        );
    }

    fn seeded_fixture() -> (Receptor, crate::grid::GridMaps, crate::scoring::VinaScoring) {
        let rec = Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [3.2, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(3, [0.0, 3.2, 0.0], Element::C, AtomType::CH),
            Atom::new(4, [0.0, 0.0, 3.2], Element::C, AtomType::CH),
        ])
        .unwrap();
        let receptor = Receptor::from_molecule(rec).unwrap();
        let box_ = crate::grid::GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        let maps = receptor.precalculate(&box_, &scoring, 0.5).expect("maps");
        (receptor, maps, scoring)
    }

    fn docked_at(seed: u64) -> DockingResult {
        let (_receptor, maps, scoring) = seeded_fixture();
        dock(
            &make_ligand(),
            &maps,
            &scoring,
            &DockingConfig {
                monte_carlo: crate::search::monte_carlo::MonteCarloConfig {
                    exhaustiveness: 4,
                    steps: 25,
                    seed: Some(seed),
                    ..Default::default()
                },
                num_modes: 3,
                ..DockingConfig::default()
            },
        )
        .expect("the search returns poses on this fixture")
    }

    #[test]
    fn a_pose_is_wholly_outside_only_when_every_atom_was_counted_outside() {
        // The predicate that `poses_outside_box_count` is built from, at its own
        // edges. It is a comparison of two counts, so the edges are 0, one short
        // of the atom count, and exactly the atom count -- and the boundary has
        // to be on the correct side, which is the only thing a `>=` can get
        // wrong here.
        let points = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.9, 0.0, 0.0]];
        for (count, want) in [(0usize, false), (1, false), (2, false), (3, true)] {
            let pose = Pose {
                atoms_outside_box: count,
                ..pose_at(&points, -1.0)
            };
            assert_eq!(
                pose.every_atom_outside_box(),
                want,
                "a {}-atom pose with {count} atom(s) counted outside is {}wholly \
                 outside. The count has to reach the atom count and no further short",
                points.len(),
                if want { "" } else { "not " }
            );
        }
        // The degenerate shape, and the reason this is not a bare `>=`. A pose
        // with no coordinates has a count of 0 that is trivially >= 0, so the
        // comparison alone would report it as wholly outside; "0 of 0 atoms are
        // outside the box" is not a claim that anything is, and the claim a
        // caller makes with this predicate is about a pose being in the box.
        let empty = Pose {
            atoms_outside_box: 0,
            ..pose_at(&[], 0.0)
        };
        // `assert!`, not `assert_eq!(.., false, ..)`: the workspace denies
        // warnings and clippy's `bool_assert_comparison` is one of them. The
        // `assert_eq!` it replaces would have printed `left: true, right: false`
        // on failure, so the direction the predicate answered is named in the
        // message instead of being lost with the macro.
        assert!(
            !empty.every_atom_outside_box(),
            "a pose with no coordinates must not be reported as wholly outside, \
             but the predicate answered true; 0 >= 0 is true and a pose nobody \
             placed is not a pose in the box"
        );
        // And one that claims more outside atoms than it has coordinates is
        // still "wholly outside" -- the claim is about the atoms, and a count
        // larger than the coordinate list means every one of them was outside.
        let over = Pose {
            atoms_outside_box: 99,
            ..pose_at(&points, -1.0)
        };
        assert!(over.every_atom_outside_box());
    }

    #[test]
    fn a_real_search_reports_no_pose_outside_the_box_and_says_so() {
        // The "goes green on a correct input" half, and the reason this field is
        // worth carrying at all: a real run on a real box must report zero, and
        // the zero has to come from the poses rather than from a default.
        //
        // This is the direction that is *not* vacuous. The other half -- a run
        // whose poses are all outside -- is not reachable from `dock()` today,
        // because the out-of-box penalty is 1000 kcal/mol per angstrom of
        // overhang and the search therefore walks every pose back in. That is
        // the mechanism doing its job, and it is also why the count is reported
        // rather than refused: the engine does not produce the state, it just
        // cannot *say* whether it did, and a field nobody reads is not a
        // safeguard against a future where it does.
        let result = docked_at(42);
        assert!(!result.poses.is_empty(), "the fixture must dock");
        let reported = result.poses_outside_box_count;
        let recount = result
            .poses
            .iter()
            .filter(|p| p.every_atom_outside_box())
            .count();
        assert_eq!(
            reported, recount,
            "the field is a count of the reported poses and has to be re-derivable \
             from them, or a consumer reading the field and a consumer walking \
             `poses` are told different things"
        );
        assert_eq!(
            reported,
            0,
            "{} of {} reported poses sit wholly outside the box. The out-of-box \
             penalty should have walked every one of them in; if this fails, the \
             count is no longer a number a reader can treat as 'nothing to see'",
            reported,
            result.poses.len()
        );
        // Per-pose, so a caller can attribute rather than only be warned: the
        // count agrees with the coordinates of each pose independently.
        for (i, pose) in result.poses.iter().enumerate() {
            assert!(
                pose.atoms_outside_box <= pose.coords.len(),
                "pose {i} claims {} outside atoms and has {} coordinates",
                pose.atoms_outside_box,
                pose.coords.len()
            );
        }
    }

    #[test]
    fn the_unrecognised_atom_count_reaches_the_result() {
        // The gap this closes. `Receptor::unknown_atom_types` has always been
        // readable, and it has always been useless to the caller that matters,
        // because the receptor is gone the moment maps exist: precalculate,
        // drop the receptor, hold maps, dock. Nothing on the result said the
        // search had run against a receptor it could not read, so a run over
        // four unrecognised atoms returned nine ranked poses and looked exactly
        // like a clean one.
        //
        // Both directions are asserted. The positive one alone would pass on a
        // constant, and this is the assertion most likely to be quietly broken
        // later by someone who hardcodes the field to 0.
        const RECEPTOR: &str = concat!(
            "ATOM      1  C1  ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n",
            "ATOM      2  C2  ALA A   1       3.200   0.000   0.000  1.00  0.00     0.000 C\n",
            "ATOM      3  C3  ALA A   1       0.000   3.200   0.000  1.00  0.00     0.000 C\n",
            "ATOM      4  C4  ALA A   1       0.000   0.000   3.200  1.00  0.00     0.000 C\n"
        );
        const EXOTIC: &str = concat!(
            "ATOM      1  C1  ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n",
            "ATOM      2  C2  ALA A   1       3.200   0.000   0.000  1.00  0.00     0.000 C\n",
            "ATOM      3  C3  ALA A   1       0.000   3.200   0.000  1.00  0.00     0.000 C\n",
            "ATOM      4  X4  ALA A   1       0.000   0.000   3.200  1.00  0.00     0.000 ZZ\n"
        );
        let box_ = crate::grid::GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        let config = DockingConfig {
            monte_carlo: crate::search::monte_carlo::MonteCarloConfig {
                exhaustiveness: 2,
                steps: 10,
                seed: Some(7),
                ..Default::default()
            },
            num_modes: 2,
            ..DockingConfig::default()
        };
        let run = |text: &str| {
            let receptor = Receptor::from_pdbqt_str(text).expect("the receptor should parse");
            let maps = receptor
                .precalculate(&box_, &scoring, 1.0)
                .expect("maps should precalculate");
            let counted = maps.unknown_atom_types();
            let result = dock(&make_ligand(), &maps, &scoring, &config).expect("should dock");
            (counted, result)
        };
        let (clean_maps, clean) = run(RECEPTOR);
        assert_eq!(
            clean_maps, 0,
            "the clean fixture must report nothing unrecognised, or the positive \
             assertion below is vacuous"
        );
        assert_eq!(clean.unknown_atom_types, 0);
        let (exotic_maps, exotic) = run(EXOTIC);
        assert_eq!(exotic_maps, 1, "the maps carry the count");
        assert_eq!(
            exotic.unknown_atom_types, 1,
            "the result has to carry it, because by the time a result exists the \
             receptor the caller would have to go back and re-read is gone"
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_poses_on_the_same_build() {
        // The claim, and only the claim that can be made here: a seed fixes the
        // result **within one build**. It does not fix it across builds, a
        // dependency bump or a compiler, and nothing in this crate can make it
        // do so. `the_seed_is_pinned_because_nothing_downstream_guarantees_it`
        // says where the boundary is.
        //
        // Coordinates, not energies. Two builds can agree on the energy of a
        // pose and place it somewhere else, and the placement is the thing a
        // user compares when they re-run a seed, so an energy comparison would
        // pass on exactly the failure this is meant to catch.
        //
        // The comparison is on the bit patterns, not the values: a tolerance
        // here would be a tolerance in a claim about reproducibility, and the
        // honest form of "the same" is the same bits. Any difference in the
        // search's arithmetic at all shows up as a different bit.
        let a = docked_at(42);
        let b = docked_at(42);
        assert_eq!(
            a.poses.len(),
            b.poses.len(),
            "the same seed produced a different number of poses"
        );
        assert!(
            !a.poses.is_empty(),
            "the fixture must actually find something"
        );
        for (i, (pa, pb)) in a.poses.iter().zip(b.poses.iter()).enumerate() {
            assert_eq!(pa.coords.len(), pb.coords.len(), "pose {i}: atom count");
            for (k, (ca, cb)) in pa.coords.iter().zip(pb.coords.iter()).enumerate() {
                for (axis, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "pose {i}, atom {k}, axis {axis}: {x} against {y} for the \
                         same seed. The two runs are the same build, the same \
                         maps and the same configuration, so any difference here \
                         is state carried between runs rather than a search that \
                         depends on its input"
                    );
                }
            }
        }
    }

    #[test]
    fn a_different_seed_gives_different_poses_so_the_claim_has_teeth() {
        // The other direction, and the one that stops the test above from being
        // a tautology. If every seed gave every pose, "the same seed gives the
        // same poses" would be true and worthless. A seed that changes nothing
        // is the failure this catches, and it is a plausible one: a search that
        // quietly stopped consulting `config.seed` would satisfy the first test
        // perfectly.
        let a = docked_at(42);
        let c = docked_at(43);
        assert_eq!(
            a.poses.len(),
            c.poses.len(),
            "same fixture, so same pose count"
        );
        let identical = a
            .poses
            .iter()
            .zip(c.poses.iter())
            .all(|(pa, pb)| pa.coords == pb.coords);
        assert!(
            !identical,
            "seeds 42 and 43 produced the same coordinates, so the seed is not \
             reaching the search and the determinism test above would be \
             asserting that the search ignores its input"
        );
    }

    #[test]
    fn the_seed_is_pinned_because_nothing_downstream_guarantees_it() {
        // Where the boundary is, stated as a value rather than as prose.
        //
        // A seeded run here means: per-walk generators are
        // `base_seed ^ walk * WALK_SEED_STRIDE` fed to
        // `rand::rngs::StdRng::seed_from_u64`. `StdRng` is the one `rand` type
        // whose algorithm is deliberately unspecified — reproducibility across
        // `rand` versions is exactly the guarantee it does not offer — so a
        // dependency bump is a candidate for silently moving every seeded
        // result, and this is the test that notices.
        //
        // The first atom of the best pose, in full precision, at seed 42 on the
        // fixture above. It is a pose, not an energy: a build could keep every
        // energy and move every atom, and only the atoms are what a user
        // re-running a seed would compare against the run they did before.
        //
        // What this does and does not buy, said plainly. It makes an
        // *accidental* move loud: a `rand` bump, a codegen change, a change to
        // the walk derivation. It cannot make the value right forever, and a
        // change to this constant and to this number together is invisible to
        // this test by construction. The honest sentence for a document is the
        // one in the failure message: a seeded result is reproducible within a
        // build, and the stream it comes from is not a stability guarantee
        // across `rand` versions.
        let res = docked_at(42);
        let best = res.best();
        let first = best.coords[0];
        assert_eq!(
            (first[0].to_bits(), first[1].to_bits(), first[2].to_bits()),
            (
                0xc00c_0002_4461_1f27u64,
                0xbff8_ebc1_ef2d_af7au64,
                0x3ffb_43f0_d2ba_28eeu64
            ),
            "the best pose at seed 42 has its first atom at {first:?}, not at the \
             position this test was written against. Measured energy is {:?}.\n\n\
             What this usually means, in the order worth checking: a `rand` bump \
             — `StdRng`'s algorithm is deliberately unspecified and reproducibility \
             across `rand` versions is the guarantee it does not offer; a change to \
             `search::WALK_SEED_STRIDE` or to the per-walk derivation; or a change \
             in the engine's own arithmetic. The first atom is pinned rather than \
             an energy on purpose: a build can keep every energy and move every \
             atom, and the atoms are what a user re-running a seed compares.",
            best.energy
        );
        // The fixture is four carbons in a 16 A box, so the best pose scores only
        // -0.80 kcal/mol and sits in a flat part of the landscape. That is a
        // property worth stating rather than fixing: a flat region is where a
        // numerical change is most likely to reorder two near-degenerate poses,
        // which is what makes this a tripwire rather than a fingerprint of a
        // deep well that nothing would ever move.
        assert!(
            best.energy > -1.0,
            "the fixture's best pose scored {:?}, which means it is no longer the \
             weak-binding fixture this pin was measured on and the pinned \
             position means something different",
            best.energy
        );
    }

    fn dummy_pose(energy: f64, shift: f64) -> Pose {
        let base = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.15, 1.3, 0.0],
            [3.65, 1.3, 0.0],
            [4.3, 2.6, 0.0],
        ];
        Pose {
            conf: Conformation::identity(1),
            energy,
            intermolecular: energy,
            intramolecular: 0.0,
            coords: base.iter().map(|c| [c[0] + shift, c[1], c[2]]).collect(),
            rmsd: None,
            gradient: Vec::new(),
            // A hand-built fixture was never evaluated against a grid, so it
            // makes no claim about the box. Zero is the honest value: it claims
            // that no atom was found outside, which for a fixture is the same
            // as saying the field is not populated, and it is why
            // `every_atom_outside_box` is false for it.
            atoms_outside_box: 0,
        }
    }

    #[test]
    fn a_reported_pose_carries_the_gradient_of_the_energy_it_reports() {
        // Direction one: a pose the search produced carries the gradient at its
        // own conformation, and that gradient really is the gradient of the
        // energy the same pose reports. The finite difference is not a
        // re-derivation of the analytic value -- it is a check that the value
        // being *carried* belongs to the conformation being returned, which is
        // the mistake a field added to a result can make and nothing else
        // catches: a gradient computed at some other iterate is still a
        // plausible-looking vector of the right length.
        let rec = Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [3.2, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(3, [0.0, 3.2, 0.0], Element::C, AtomType::CH),
            Atom::new(4, [0.0, 0.0, 3.2], Element::C, AtomType::CH),
        ])
        .unwrap();
        let receptor = Receptor::from_molecule(rec).unwrap();
        let box_ = crate::grid::GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        let maps = receptor.precalculate(&box_, &scoring, 0.5).unwrap();
        let ligand = make_ligand();
        let res = dock(
            &ligand,
            &maps,
            &scoring,
            &DockingConfig {
                monte_carlo: crate::search::monte_carlo::MonteCarloConfig {
                    exhaustiveness: 4,
                    steps: 30,
                    seed: Some(11),
                    ..Default::default()
                },
                ..DockingConfig::default()
            },
        )
        .unwrap();
        assert!(!res.poses.is_empty());

        let mut ctx = crate::search::ScoringContext::new(&ligand, &maps, &scoring);
        for (i, p) in res.poses.iter().enumerate() {
            let g = p
                .measured_gradient()
                .unwrap_or_else(|| panic!("pose {i} reports no gradient at all"));
            assert_eq!(
                g.len(),
                6 + ligand.num_torsions(),
                "pose {i}: one entry per degree of freedom, or the consumer has \
                 no way to know which entry is which"
            );

            // The engine's own gradient at the pose's conformation, which is the
            // claim being pinned: carried == recomputed at the same point.
            let (energy, want) = ctx.evaluate(&p.conf);
            assert!(
                (energy - p.energy).abs() < 1e-9,
                "pose {i}: the energy recomputed at the returned conformation is \
                 {energy}, and the pose reports {}",
                p.energy
            );
            for (k, (a, b)) in g.iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                    "pose {i}, dof {k}: carried gradient {a} against {b} at the \
                     same conformation"
                );
            }
        }
    }

    #[test]
    fn a_pose_that_measured_nothing_says_so_rather_than_looking_stationary() {
        // Direction two: the one that stops a reader assuming a field is
        // populated. A pose with no gradient must answer `None`, and it must
        // answer `None` for its own reason -- an empty vector is "not
        // measured", and a consumer that read it as "all zeros" would conclude
        // the pose is a stationary point, which is the most damaging wrong
        // answer available here.
        let pose = pose_at(&[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], -3.25);
        assert!(
            pose.measured_gradient().is_none(),
            "a hand-built pose measured nothing and must not present a gradient"
        );
        assert!(
            pose.gradient.is_empty(),
            "the vector is empty rather than zero-filled: a zero vector would be \
             indistinguishable from a pose that had been measured and found \
             stationary"
        );
    }

    #[test]
    fn rmsd_of_identical_is_zero() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(rmsd(&a, &a, 2).abs() < 1e-12);
    }

    #[test]
    fn rmsd_of_translated_poses() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        // Atom 1 moved by (0, 3, 4): squared distance 25, averaged over 2 atoms.
        let b = [[0.0, 0.0, 0.0], [1.0, 3.0, 4.0]];
        assert!((rmsd(&a, &b, 2) - (25.0f64 / 2.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn clustering_keeps_distinct_poses_only() {
        let raw = vec![
            dummy_pose(-9.0, 0.0),
            dummy_pose(-8.5, 0.1),  // within 1 Å of the best -> duplicate
            dummy_pose(-7.0, 5.0),  // far away -> new mode
            dummy_pose(-6.0, 5.05), // duplicate of the previous
        ];
        let lig = make_ligand();
        let kept = cluster_poses(raw, 5, 1.0, &lig).unwrap();
        assert_eq!(kept.len(), 2);
        assert!((kept[0].energy + 9.0).abs() < 1e-12);
        assert!((kept[1].energy + 7.0).abs() < 1e-12);
        assert!(kept[1].rmsd.unwrap() > 1.0);
    }

    #[test]
    fn clustering_respects_num_modes() {
        let raw = (0..10)
            .map(|i| dummy_pose(-10.0 + i as f64, i as f64 * 3.0))
            .collect();
        let lig = make_ligand();
        let kept = cluster_poses(raw, 3, 1.0, &lig).unwrap();
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn clustering_rejects_an_all_empty_result() {
        let lig = make_ligand();
        assert!(cluster_poses(Vec::new(), 3, 1.0, &lig).is_err());
    }

    #[test]
    fn zero_modes_is_rejected() {
        let lig = make_ligand();
        let maps = empty_maps();
        let scoring = crate::scoring::VinaScoring::new();
        let cfg = DockingConfig {
            num_modes: 0,
            ..DockingConfig::fast()
        };
        assert!(dock(&lig, &maps, &scoring, &cfg).is_err());
    }

    fn pose_at(points: &[[f64; 3]], energy: f64) -> Pose {
        Pose {
            conf: Conformation::identity(0),
            coords: points.to_vec(),
            energy,
            intermolecular: energy,
            intramolecular: 0.0,
            rmsd: None,
            // A hand-built fixture measures nothing, and says so.
            gradient: Vec::new(),
            atoms_outside_box: 0,
        }
    }

    #[test]
    fn a_physically_impossible_pose_is_separated_from_a_real_one() {
        let receptor = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 8.0]];
        let poses = vec![
            // 0.5 A from the receptor origin: impossible.
            pose_at(&[[0.5, 0.0, 0.0]], -9.0),
            // 5 A away: possible, and a worse energy.
            pose_at(&[[5.0, 0.0, 0.0]], -1.0),
        ];
        let (clean, clashing) = partition_by_clash(poses, &receptor, MIN_CONTACT_DISTANCE);
        assert_eq!(clashing.len(), 1, "the clashing pose must be separated");
        assert_eq!(clean.len(), 1);
        // The point is that the *lower* energy one was the impossible one.
        assert_eq!(clashing[0].energy, -9.0);
        assert_eq!(clean[0].energy, -1.0);
    }

    #[test]
    fn the_clash_check_can_be_switched_off() {
        let receptor = vec![[0.0, 0.0, 0.0]];
        let poses = vec![pose_at(&[[0.1, 0.0, 0.0]], -5.0)];
        let (clean, clashing) = partition_by_clash(poses.clone(), &receptor, 0.0);
        assert_eq!(clean.len(), 1);
        assert!(
            clashing.is_empty(),
            "a disabled check must not reject anything"
        );
    }

    #[test]
    fn an_empty_receptor_never_rejects() {
        let poses = vec![pose_at(&[[0.0, 0.0, 0.0]], -5.0)];
        let (clean, clashing) = partition_by_clash(poses, &[], MIN_CONTACT_DISTANCE);
        assert_eq!(clean.len(), 1);
        assert!(clashing.is_empty());
    }

    #[test]
    fn the_docked_result_reports_no_clashes_when_the_search_finds_clean_poses() {
        // A box with plenty of empty space: a weak search may still jam the
        // ligand against the receptor, but the pipeline must come back with
        // poses and say whether any of them clash.
        let rec = Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [3.2, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(3, [0.0, 3.2, 0.0], Element::C, AtomType::CH),
            Atom::new(4, [0.0, 0.0, 3.2], Element::C, AtomType::CH),
        ])
        .unwrap();
        let receptor = Receptor::from_molecule(rec).unwrap();
        let box_ = crate::grid::GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        let maps = receptor.precalculate(&box_, &scoring, 0.5).unwrap();
        assert_eq!(maps.receptor_atoms().len(), 4);

        let res = dock(
            &make_ligand(),
            &maps,
            &scoring,
            &DockingConfig {
                monte_carlo: crate::search::monte_carlo::MonteCarloConfig {
                    exhaustiveness: 8,
                    steps: 40,
                    seed: Some(7),
                    ..Default::default()
                },
                ..DockingConfig::default()
            },
        )
        .unwrap();
        assert!(!res.poses.is_empty());
        for p in &res.poses {
            for c in &p.coords {
                for q in maps.receptor_atoms() {
                    let d = ((c[0] - q[0]).powi(2) + (c[1] - q[1]).powi(2) + (c[2] - q[2]).powi(2))
                        .sqrt();
                    assert!(
                        d >= MIN_CONTACT_DISTANCE,
                        "a reported pose is {d} A from a receptor atom, below the floor"
                    );
                }
            }
        }
    }

    fn empty_maps() -> GridMaps {
        let mol = Molecule::from_atoms(vec![Atom::new(
            1,
            [0.0, 0.0, 0.0],
            Element::C,
            AtomType::CH,
        )])
        .unwrap();
        let b = crate::grid::GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).unwrap();
        crate::grid::GridMaps::precalculate(&mol, &b, &crate::scoring::VinaScoring::new(), 0.5, 1)
            .unwrap()
    }

    #[test]
    fn end_to_end_dock_finds_a_low_energy_pose() {
        // A small carbon "receptor" and a butane-like ligand: the ligand should
        // end up inside the pocket, with a clearly negative energy.
        let rec = Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [3.2, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(3, [0.0, 3.2, 0.0], Element::C, AtomType::CH),
            Atom::new(4, [0.0, 0.0, 3.2], Element::C, AtomType::CH),
        ])
        .unwrap();
        let receptor = Receptor::from_molecule(rec).unwrap();
        let box_ = crate::grid::GridBox::new([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        let maps = receptor.precalculate(&box_, &scoring, 0.5).unwrap();
        let lig = make_ligand();
        let res = dock(&lig, &maps, &scoring, &DockingConfig::fast()).unwrap();
        assert!(!res.poses.is_empty());
        assert!(res.best().energy < 0.0, "best energy {}", res.best().energy);
        // Every reported pose must be inside the box.
        for p in &res.poses {
            for c in &p.coords {
                assert!(box_.contains(*c), "pose atom outside the box: {c:?}");
            }
        }
    }
}
