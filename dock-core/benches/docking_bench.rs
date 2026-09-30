//! Micro-benchmarks for the Open Docking scoring and grid hot paths.
//!
//! Run with `cargo bench -p dock-core`. These are deliberately small and
//! self-contained so they work without external data files.
//!
//! The harness is disabled (`harness = false` in `Cargo.toml`): this is a plain
//! binary that prints timings, not a `#[bench]`-annotated criterion suite. That
//! keeps the benchmark dependency-free and lets it run identically under
//! `cargo bench` and `cargo run --release --bench docking_bench`.

use std::time::Instant;

use dock_core::docking::{dock, DockingConfig};
use dock_core::grid::{GridBox, GridMaps};
use dock_core::kinematics::KinematicTree;
use dock_core::ligand::Ligand;
use dock_core::pdbqt::parse_pdbqt;
use dock_core::scoring::{ScoringFunction, VinaScoring};
use dock_core::search::monte_carlo::MonteCarloConfig;
use dock_core::types::{Atom, AtomKind, AtomType, Element, Molecule};

/// A small but realistic test ligand: 12 atoms, several rotatable bonds.
const LIGAND: &str = concat!(
    "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.020 C \n",
    "ATOM      2  C2  UNL     1       1.500   0.000   0.000  1.00  0.00     0.010 C \n",
    "ATOM      3  C3  UNL     1       2.150   1.300   0.000  1.00  0.00     0.015 C \n",
    "ATOM      4  C4  UNL     1       3.650   1.300   0.000  1.00  0.00     0.010 C \n",
    "ATOM      5  C5  UNL     1       4.300   2.600   0.000  1.00  0.00     0.030 C \n",
    "ATOM      6  N1  UNL     1      -0.700   1.100   0.000  1.00  0.00    -0.350 NA \n",
    "ATOM      7  O1  UNL     1      -1.900   1.300   0.000  1.00  0.00    -0.400 OA \n",
    "ATOM      8  C6  UNL     1       1.900  -1.400   0.000  1.00  0.00     0.000 C \n",
    "ATOM      9  C7  UNL     1       1.400  -2.100   1.200  1.00  0.00     0.000 C \n",
    "ATOM     10  O2  UNL     1       2.300  -3.300   1.500  1.00  0.00    -0.400 OA \n",
    "ATOM     11  C8  UNL     1       0.400  -1.700   0.900  1.00  0.00     0.000 C \n",
    "ATOM     12  C9  UNL     1       0.000  -1.000   0.000  1.00  0.00     0.000 C \n",
);

/// A coarse "receptor" of carbon atoms scattered through a small cavity.
fn synthetic_receptor(n: usize) -> Molecule {
    let mut atoms = Vec::with_capacity(n);
    let mut s = 1u32;
    // A tiny deterministic LCG so the benchmark is reproducible.
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f64 / 16_777_216.0) - 0.5
    };
    for i in 0..n {
        atoms.push(Atom {
            serial: i as u32 + 1,
            name: "C".to_string(),
            resname: "ALA".to_string(),
            resid: (i % 4 + 1) as i32,
            coord: [next() * 18.0, next() * 18.0, next() * 18.0],
            charge: 0.0,
            element: Element::C,
            atom_type: AtomType::CH,
            kind: AtomKind::Hydrophobic,
        });
    }
    // A scattered lattice has no bonds, so the neighbour table is empty.
    let count = atoms.len();
    Molecule {
        atoms,
        bonds: Vec::new(),
        neighbors: vec![Vec::new(); count],
    }
}

fn main() {
    println!("opendocking dock-core benchmarks");
    println!("  rayon threads: {}\n", rayon::current_num_threads());

    let scoring = VinaScoring::new();
    let box_ = GridBox::new([-9.0, -9.0, -9.0], [9.0, 9.0, 9.0]).expect("box");

    // --- 1. Grid precalculation -------------------------------------------
    let receptor = synthetic_receptor(600);
    let t0 = Instant::now();
    let maps = GridMaps::precalculate(&receptor, &box_, &scoring, 0.375, 0).expect("grid");
    let dt = t0.elapsed();
    let [nx, ny, nz] = maps.dims();
    println!(
        "grid precalculate: {dt:>12.3?}  {} atoms -> {nx} x {ny} x {nz} x {} maps ({:.1} MiB)",
        receptor.len(),
        maps.map_count(),
        maps.raw_slice().len() * 4 / (1024 * 1024) as usize,
    );

    // --- 2. Trilinear interpolation (energy only) -------------------------
    let weights = scoring.atom_weights(&receptor.atoms[0]);
    let probes: Vec<[f64; 3]> = (0..20_000)
        .map(|i| {
            let t = i as f64 * 0.001_037;
            [
                -8.0 + 16.0 * ((t * 1.7).sin() * 0.5 + 0.5),
                -8.0 + 16.0 * ((t * 2.3).cos() * 0.5 + 0.5),
                -8.0 + 16.0 * ((t * 0.9).sin() * 0.5 + 0.5),
            ]
        })
        .collect();
    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for p in &probes {
        acc += maps.interpolate(0, &weights, *p).unwrap_or(0.0);
    }
    let dt = t0.elapsed();
    println!(
        "trilinear sample:  {:>12.3?}  {} queries ({:.1} ns/query, acc {acc:.3})",
        dt,
        probes.len(),
        dt.as_nanos() as f64 / probes.len() as f64,
    );

    // --- 3. Trilinear interpolation with gradient -------------------------
    let t0 = Instant::now();
    let mut gacc = 0.0f64;
    for p in &probes {
        if let Some((e, g)) = maps.interpolate_with_gradient(0, &weights, *p) {
            gacc += e + g[0] + g[1] + g[2];
        }
    }
    let dt = t0.elapsed();
    println!(
        "trilinear + grad:  {dt:>12.3?}  {} queries ({:.1} ns/query, acc {gacc:.3})",
        probes.len(),
        dt.as_nanos() as f64 / probes.len() as f64,
    );

    // --- 4. PDBQT parsing ---------------------------------------------------
    let text = LIGAND.repeat(200);
    let reps = 50;
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = parse_pdbqt(&text).expect("parse");
    }
    let dt = t0.elapsed() / reps;
    println!(
        "pdbqt parse:       {dt:>12.3?}  per {}-atom file",
        LIGAND.lines().count() * 200,
    );

    // --- 5. Kinematic tree construction ------------------------------------
    let parsed = parse_pdbqt(LIGAND).expect("ligand");
    let t0 = Instant::now();
    let tree = KinematicTree::from_molecule(&parsed.molecule, &[]).expect("tree");
    let dt = t0.elapsed();
    println!(
        "kinematic tree:    {dt:>12.3?}  {} torsions, {} rigid clusters, {} dof",
        tree.num_torsions(),
        tree.clusters.len(),
        tree.ndof(),
    );

    // --- 6. Full docking run -----------------------------------------------
    let config = DockingConfig {
        monte_carlo: MonteCarloConfig {
            exhaustiveness: 2,
            ..Default::default()
        },
        num_modes: 3,
        ..DockingConfig::default()
    };
    let ligand = Ligand::from_molecule(parsed.molecule.clone()).expect("ligand");
    let t0 = Instant::now();
    let result = dock(&ligand, &maps, &scoring, &config).expect("dock");
    let dt = t0.elapsed();
    println!(
        "dock (exh 2):      {dt:>12.3?}  best {:.3} kcal/mol, {} poses from {} raw",
        result.best().energy,
        result.poses.len(),
        result.raw_pose_count,
    );

    // --- 7. Vinardo maps (no Gaussian term) --------------------------------
    let vinardo = VinaScoring::vinardo();
    let t0 = Instant::now();
    let maps_v = GridMaps::precalculate(&receptor, &box_, &vinardo, 0.375, 0).expect("vinardo");
    let dt = t0.elapsed();
    println!(
        "vinardo grid:      {dt:>12.3?}  {} maps",
        maps_v.map_count()
    );
}
