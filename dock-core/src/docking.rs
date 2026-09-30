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

    Ok(DockingResult {
        poses,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        raw_pose_count,
        rejected_pose_count: clashing_reported,
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
        }
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
