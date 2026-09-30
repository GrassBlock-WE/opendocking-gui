//! Energy evaluation and the local/global search strategies.
//!
//! * [`ScoringContext`] evaluates the full docking energy and its analytic
//!   gradient with respect to the conformational degrees of freedom.
//! * [`Lbfgs`] is a bounded-memory quasi-Newton local optimiser with a
//!   gradient-descent safeguard, which the discontinuous trilinear grid
//!   gradient makes necessary.
//! * [`monte_carlo`] is the iterated local search that explores the box.
//! * [`lga`] is an island-model hybrid genetic algorithm for difficult,
//!   very flexible ligands.
//! * [`evaluate_population`] scores a whole batch of conformations at once and
//!   moves the per-atom grid interpolation to the GPU when the `gpu` feature is
//!   compiled in and an adapter is available.

pub mod lbfgs;
pub mod lga;
pub mod monte_carlo;

use nalgebra::Vector3;

use crate::grid::GridMaps;
use crate::kinematics::Conformation;
use crate::ligand::Ligand;
use crate::scoring::ScoringFunction;
use crate::types::Vec3;

/// Penalty applied when a ligand atom leaves the search box.
///
/// The box is a hard constraint in AutoDock; rather than trusting the sampler
/// never to violate it, a large linear penalty is added so a violating pose is
/// never the lowest-energy one.
pub const OUT_OF_BOX_PENALTY: f64 = 1000.0;

/// The two halves of a docking energy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyBreakdown {
    /// Receptor–ligand interaction, in kcal/mol.
    pub intermolecular: f64,
    /// Ligand-internal (1-4 and beyond) interaction, in kcal/mol.
    pub intramolecular: f64,
}

impl EnergyBreakdown {
    /// Total docking energy: intermolecular plus the scaled intramolecular term.
    pub fn total(&self, intra_scale: f64) -> f64 {
        self.intermolecular + self.intramolecular * intra_scale
    }
}

/// Evaluates the docking energy and gradient for one ligand against one map set.
///
/// Holds all reusable buffers, so a long search performs no allocation in its
/// inner loop.
pub struct ScoringContext<'a> {
    ligand: &'a Ligand,
    maps: &'a GridMaps,
    scoring: &'a dyn ScoringFunction,
    /// Scratch: global coordinates for the current conformation.
    coords: Vec<Vec3>,
    /// Scratch: gradient with respect to each atom position.
    atom_grad: Vec<Vector3<f64>>,
    /// True if any atom of the last evaluation fell outside the grid.
    last_outside: bool,
}

impl<'a> ScoringContext<'a> {
    /// Build a context. `ligand` and `maps` must outlive the context.
    pub fn new(ligand: &'a Ligand, maps: &'a GridMaps, scoring: &'a dyn ScoringFunction) -> Self {
        let n = ligand.len();
        ScoringContext {
            ligand,
            maps,
            scoring,
            coords: vec![[0.0; 3]; n],
            atom_grad: vec![Vector3::zeros(); n],
            last_outside: false,
        }
    }

    /// The ligand being scored.
    pub fn ligand(&self) -> &Ligand {
        self.ligand
    }

    /// The receptor maps in use.
    pub fn maps(&self) -> &GridMaps {
        self.maps
    }

    /// True if the most recent evaluation placed an atom outside the grid.
    pub fn last_outside(&self) -> bool {
        self.last_outside
    }

    /// Coordinates produced by the most recent evaluation, in atom order.
    pub fn current_coords(&self) -> &[Vec3] {
        &self.coords
    }

    /// Evaluate the energy only, returning the full breakdown total.
    pub fn energy(&mut self, conf: &Conformation) -> f64 {
        let (b, _) = self.evaluate_full(conf);
        b.total(self.scoring.intramolecular_scale())
    }

    /// Evaluate the energy and the gradient with respect to the degrees of
    /// freedom of `conf`.
    ///
    /// The returned gradient has `6 + num_torsions` entries. When a ligand
    /// atom leaves the grid, a linear penalty is added along the direction
    /// back to the box centre so the optimiser is pushed back inside.
    pub fn evaluate(&mut self, conf: &Conformation) -> (f64, Vec<f64>) {
        let (b, g) = self.evaluate_full(conf);
        (b.total(self.scoring.intramolecular_scale()), g)
    }

    /// Run forward kinematics for `conf`, leaving the global coordinates in
    /// `self.coords` and returning nothing.
    ///
    /// This is the CPU half of the batched path: the torsion recursion cannot be
    /// moved to the GPU, so every conformation is expanded here regardless of
    /// where the *scoring* happens. See [`evaluate_population`].
    pub fn apply(&mut self, conf: &Conformation) {
        self.ligand.tree.apply(conf, &mut self.coords);
    }

    /// Intramolecular energy evaluated at the coordinates currently stored by
    /// [`ScoringContext::apply`].
    ///
    /// Split out so that the batched GPU path can take the intermolecular half
    /// from the compute kernel and still produce exactly the total the CPU path
    /// would have produced. The two halves are independent, so recombining them
    /// is not an approximation.
    pub fn intramolecular_energy_at_current_coords(&self) -> f64 {
        let mut intra = 0.0f64;
        for &(i, j) in self.ligand.intramolecular_pairs() {
            let ai = &self.ligand.molecule.atoms[i];
            let aj = &self.ligand.molecule.atoms[j];
            let (ri, rj) = (self.coords[i], self.coords[j]);
            let dx = ri[0] - rj[0];
            let dy = ri[1] - rj[1];
            let dz = ri[2] - rj[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            if r < 1e-9 {
                continue;
            }
            let d = r - (ai.element.interaction_radius() + aj.element.interaction_radius());
            if d > crate::scoring::VINA_CUTOFF {
                continue;
            }
            intra += self.scoring.pair_energy(ai, aj, d);
        }
        intra
    }
}

impl<'a> ScoringContext<'a> {
    /// Full evaluation returning the energy breakdown as well as the gradient.
    pub fn evaluate_full(&mut self, conf: &Conformation) -> (EnergyBreakdown, Vec<f64>) {
        self.ligand.tree.apply(conf, &mut self.coords);
        for g in self.atom_grad.iter_mut() {
            g.fill(0.0);
        }
        self.last_outside = false;

        let scale = self.scoring.intramolecular_scale();
        let mut inter_grad = vec![Vector3::zeros(); self.ligand.len()];
        let mut inter = 0.0f64;

        for i in 0..self.ligand.len() {
            let p = self.coords[i];
            let ti = self.ligand.type_index[i];
            let w = &self.ligand.weights[i];
            match self.maps.interpolate_with_gradient(ti, w, p) {
                Some((e, g)) => {
                    inter += e;
                    inter_grad[i] += Vector3::new(g[0], g[1], g[2]);
                }
                None => {
                    self.last_outside = true;
                    let c = self.maps.grid_box().center();
                    let mut dir = [0.0f64; 3];
                    let mut norm = 0.0;
                    for k in 0..3 {
                        dir[k] = c[k] - p[k];
                        norm += dir[k] * dir[k];
                    }
                    let norm = norm.sqrt();
                    if norm > 1e-9 {
                        for k in 0..3 {
                            dir[k] /= norm;
                            inter_grad[i][k] += OUT_OF_BOX_PENALTY * dir[k];
                        }
                        inter += OUT_OF_BOX_PENALTY;
                    }
                }
            }
        }

        let mut intra = 0.0f64;
        let mut intra_grad = vec![Vector3::zeros(); self.ligand.len()];
        for &(i, j) in self.ligand.intramolecular_pairs() {
            let ai = &self.ligand.molecule.atoms[i];
            let aj = &self.ligand.molecule.atoms[j];
            let (ri, rj) = (self.coords[i], self.coords[j]);
            let dx = ri[0] - rj[0];
            let dy = ri[1] - rj[1];
            let dz = ri[2] - rj[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            if r < 1e-9 {
                continue;
            }
            let d = r - (ai.element.interaction_radius() + aj.element.interaction_radius());
            if d > crate::scoring::VINA_CUTOFF {
                continue;
            }
            intra += self.scoring.pair_energy(ai, aj, d);
            let g = self.scoring.pair_gradient(ai, aj, d) / r;
            let contribution = Vector3::new(dx, dy, dz) * g;
            intra_grad[i] += contribution;
            intra_grad[j] -= contribution;
        }

        // Combine per-atom gradients, applying the intramolecular scale only to
        // its own half.
        let mut combined: Vec<Vector3<f64>> = (0..self.ligand.len())
            .map(|i| inter_grad[i] + intra_grad[i] * scale)
            .collect();
        core::mem::swap(&mut combined, &mut self.atom_grad);
        let grad = self.ligand.tree.conf_gradient(conf, &self.atom_grad);

        (
            EnergyBreakdown {
                intermolecular: inter,
                intramolecular: intra,
            },
            grad,
        )
    }
}

/// Which backend actually evaluated a population, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PopulationBackend {
    /// The CPU path ran.
    Cpu,
    /// The compute kernel produced the intermolecular half.
    Gpu {
        /// The adapter the kernel ran on, for diagnostics.
        adapter: String,
    },
}

impl PopulationBackend {
    /// True when the GPU was used.
    pub fn is_gpu(&self) -> bool {
        matches!(self, PopulationBackend::Gpu { .. })
    }
}

/// Why the GPU path was not taken.
///
/// This is returned rather than swallowed because "silently ran on the CPU at
/// one twentieth of the speed" is exactly the kind of thing that otherwise shows
/// up as an unexplained performance regression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSkip {
    /// A short, stable description of what made the GPU path decline.
    pub reason: &'static str,
}

impl std::fmt::Display for GpuSkip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.reason)
    }
}

/// Score a whole population of conformations.
///
/// Returns the **total** energy for each conformation — the same quantity
/// [`ScoringContext::energy`] returns — so the two paths are directly
/// comparable, and that is what a test asserts.
///
/// # Where the GPU helps
///
/// The torsion recursion stays on the CPU: torsions form a tree, so each
/// conformation's coordinates are produced by a sequential walk. What moves to
/// the device is the per-atom grid interpolation, which is embarrassingly
/// parallel and bandwidth-bound. For a big population of small ligands that
/// split is where the win is; for a single conformation the upload dominates
/// and the CPU is competitive or better.
///
/// # The one workgroup limit
///
/// The kernel reduces one conformation per workgroup and the workgroup size is
/// fixed at 64, so a ligand with more than 64 atoms cannot be scored this way.
/// Such ligands fall back to the CPU rather than being split up, because
/// splitting a reduction across workgroups is a much bigger change than the
/// benefit is worth here.
pub fn evaluate_population(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    conformations: &[Conformation],
    prefer_gpu: bool,
) -> (Vec<f64>, PopulationBackend, Option<GpuSkip>) {
    #[cfg(feature = "gpu")]
    {
        if prefer_gpu {
            match try_gpu_population(ligand, maps, scoring, conformations) {
                Ok((energies, backend)) => return (energies, backend, None),
                Err(skip) => {
                    // Fall back, but say why: a silent CPU fallback reads as a
                    // mysterious slowdown.
                    let (energies, _, _) = cpu_population(ligand, maps, scoring, conformations);
                    return (energies, PopulationBackend::Cpu, Some(skip));
                }
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    let _ = prefer_gpu;

    cpu_population(ligand, maps, scoring, conformations)
}

fn cpu_population(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    conformations: &[Conformation],
) -> (Vec<f64>, PopulationBackend, Option<GpuSkip>) {
    let mut ctx = ScoringContext::new(ligand, maps, scoring);
    let energies = conformations.iter().map(|c| ctx.energy(c)).collect();
    (energies, PopulationBackend::Cpu, None)
}

#[cfg(feature = "gpu")]
fn try_gpu_population(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    conformations: &[Conformation],
) -> Result<(Vec<f64>, PopulationBackend), GpuSkip> {
    use crate::gpu::{Batch, WORKGROUP_SIZE};

    let n_atoms = ligand.len();
    if n_atoms > WORKGROUP_SIZE as usize {
        return Err(GpuSkip {
            reason: "ligand has more atoms than the workgroup size",
        });
    }
    if conformations.is_empty() {
        return Err(GpuSkip {
            reason: "empty population",
        });
    }

    let mut ctx = ScoringContext::new(ligand, maps, scoring);
    let mut coords = Vec::with_capacity(conformations.len() * n_atoms * 3);
    let mut intra = Vec::with_capacity(conformations.len());
    for conf in conformations {
        ctx.apply(conf);
        for p in ctx.current_coords() {
            coords.push(p[0] as f32);
            coords.push(p[1] as f32);
            coords.push(p[2] as f32);
        }
        intra.push(ctx.intramolecular_energy_at_current_coords());
    }

    let mut gpu = crate::gpu::GpuContext::new().map_err(|_| GpuSkip {
        reason: "no usable GPU adapter",
    })?;
    let adapter = gpu.adapter_name().to_string();
    let batch = Batch::from_coords(coords, conformations.len(), ligand).map_err(|_| GpuSkip {
        reason: "could not pack the batch",
    })?;
    let inter = gpu.score(&batch, maps).map_err(|_| GpuSkip {
        reason: "the compute pass failed",
    })?;

    let scale = scoring.intramolecular_scale();
    let energies = inter
        .iter()
        .zip(intra.iter())
        .map(|(g, i)| *g as f64 + i * scale)
        .collect();
    Ok((
        energies,
        PopulationBackend::Gpu {
            adapter: adapter.to_string(),
        },
    ))
}

/// Pack a conformation into a flat degree-of-freedom vector:
/// `[tx, ty, tz, θx, θy, θz, τ₀, …]`.
pub fn conf_to_vec(conf: &Conformation) -> Vec<f64> {
    let mut v = Vec::with_capacity(6 + conf.torsions.len());
    v.extend_from_slice(&conf.position);
    v.extend_from_slice(&conf.orientation);
    v.extend_from_slice(&conf.torsions);
    v
}

/// Rebuild a conformation from a packed degree-of-freedom vector.
pub fn vec_to_conf(x: &[f64], num_torsions: usize) -> Conformation {
    Conformation {
        position: [x[0], x[1], x[2]],
        orientation: [x[3], x[4], x[5]],
        torsions: x[6..6 + num_torsions].to_vec(),
    }
}

/// A quadratic penalty that keeps torsions on the circle.
///
/// A torsion angle is periodic, so a quasi-Newton step that overshoots past ±π
/// must wrap rather than pay an artificial cost — this keeps the landscape
/// smooth for the optimiser.
#[inline]
pub fn wrap_torsion(mut angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    angle %= tau;
    if angle > std::f64::consts::PI {
        angle -= tau;
    } else if angle < -std::f64::consts::PI {
        angle += tau;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridBox;
    use crate::receptor::Receptor;
    use crate::scoring::VinaScoring;
    use crate::types::Molecule;

    /// n-Hexane as a rigid root plus four rotatable branches: a ligand with
    /// **several** coupled torsions, which is where a pull-back that only ever
    /// sees one torsion can go wrong without any existing test noticing.
    const HEXANE: &str = concat!(
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
        "BRANCH   4   5\n",
        "ATOM      5  C5  UNL     1       4.300   2.600   0.000  1.00  0.00     0.000 C \n",
        "ENDBRANCH\n",
        "BRANCH   5   6\n",
        "ATOM      6  C6  UNL     1       5.800   2.600   0.000  1.00  0.00     0.000 C \n",
        "ENDBRANCH\n",
        "TORSDOF 4\n",
    );

    fn parse_flexible(text: &str) -> (Ligand, GridMaps) {
        let parsed = crate::pdbqt::parse_pdbqt(text).expect("the fixture should parse");
        let torsions: Vec<(usize, usize)> = parsed
            .resolved_torsions
            .iter()
            .map(|(prox, dist, _, _)| (*prox, *dist))
            .collect();
        let ligand =
            Ligand::from_molecule_with(parsed.molecule.clone(), &torsions).expect("a ligand");
        let receptor = Receptor::from_molecule(parsed.molecule).expect("a receptor");
        let box_ = GridBox::centered([0.0, 0.0, 0.0], [16.0, 16.0, 16.0]).expect("valid box");
        let maps = receptor
            .precalculate(&box_, &VinaScoring::new(), 0.375)
            .expect("precalculation should succeed");
        (ligand, maps)
    }

    /// n-Butane as two rigid segments joined by rotatable bonds: the smallest
    /// ligand that exercises the torsion machinery at all.
    const BUTANE: &str = concat!(
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
    );

    fn parse_butane() -> (Molecule, Vec<(usize, usize)>) {
        let parsed = crate::pdbqt::parse_pdbqt(BUTANE).expect("the fixture should parse");
        let torsions = parsed
            .resolved_torsions
            .iter()
            .map(|(prox, dist, _, _)| (*prox, *dist))
            .collect();
        (parsed.molecule, torsions)
    }

    fn setup() -> (Ligand, GridMaps) {
        let (mol, torsions) = parse_butane();
        let ligand =
            Ligand::from_molecule_with(mol.clone(), &torsions).expect("butane is a ligand");
        let receptor = Receptor::from_molecule(mol).expect("butane is a receptor");
        let box_ = GridBox::centered([0.0, 0.0, 0.0], [12.0, 12.0, 12.0]).expect("valid box");
        let maps = receptor
            .precalculate(&box_, &VinaScoring::new(), 0.375)
            .expect("precalculation should succeed");
        (ligand, maps)
    }

    /// A deterministic spread of conformations.
    ///
    /// xorshift keeps this dependency-free and reproducible across platforms;
    /// a fixture built from `rand` would not be.
    fn population(n: usize, seed: u64, num_torsions: usize) -> Vec<Conformation> {
        let mut state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        (0..n)
            .map(|_| Conformation {
                position: [next() * 1.5, next() * 1.5, next() * 1.5],
                orientation: [next(), next(), next()],
                torsions: (0..num_torsions)
                    .map(|_| next() * std::f64::consts::PI)
                    .collect(),
            })
            .collect()
    }

    /// The analytic degree-of-freedom gradient must be the gradient *of the
    /// energy this module reports*.
    ///
    /// `kinematics::conf_gradient` has its own finite-difference test, but that
    /// one feeds it a hand-made linear functional of the coordinates, so it
    /// only proves the pull-back algebra. Nothing covered the other half: that
    /// the per-atom gradient assembled here — trilinear weight derivatives
    /// from the maps, plus the scaled intramolecular pair gradient — is itself
    /// the gradient of `evaluate_full().total()`. A sign error or a missing
    /// term anywhere along that chain would leave every existing test green
    /// while the optimiser walked uphill.
    #[test]
    fn the_analytic_gradient_is_the_gradient_of_the_reported_energy() {
        for (name, text) in [("butane", BUTANE), ("hexane", HEXANE)] {
            let (ligand, maps) = parse_flexible(text);
            let sc = VinaScoring::new();
            let mut ctx = ScoringContext::new(&ligand, &maps, &sc);
            assert!(
                ligand.num_torsions() > 0,
                "{name} must exercise the torsion half of the gradient"
            );

            for (seed, conf) in population(6, 0xC0FFEE01, ligand.num_torsions())
                .into_iter()
                .enumerate()
            {
                let (e0, g) = ctx.evaluate(&conf);
                assert!(!ctx.last_outside(), "{name} case {seed} left the grid");

                // 1e-5 is far below the grid spacing, so a central difference
                // stays inside one trilinear cell and the field is smooth there.
                const H: f64 = 1e-5;
                for k in 0..conf.ndof() {
                    let mut plus = conf.clone();
                    let mut minus = conf.clone();
                    set_dof(&mut plus, k, get_dof(&conf, k) + H);
                    set_dof(&mut minus, k, get_dof(&conf, k) - H);
                    let (ep, _) = ctx.evaluate(&plus);
                    let (em, _) = ctx.evaluate(&minus);
                    let numeric = (ep - em) / (2.0 * H);
                    let kind = if k < 3 {
                        "translation"
                    } else if k < 6 {
                        "rotation"
                    } else {
                        "torsion"
                    };
                    assert!(
                        (numeric - g[k]).abs() < 1e-3,
                        "{name} case {seed}, {kind} dof {k}: analytic {:+.6} vs \
                         numeric {:+.6} (E = {e0:.6})",
                        g[k],
                        numeric
                    );
                }
            }
        }
    }

    fn get_dof(c: &Conformation, k: usize) -> f64 {
        if k < 3 {
            c.position[k]
        } else if k < 6 {
            c.orientation[k - 3]
        } else {
            c.torsions[k - 6]
        }
    }

    fn set_dof(c: &mut Conformation, k: usize, v: f64) {
        if k < 3 {
            c.position[k] = v;
        } else if k < 6 {
            c.orientation[k - 3] = v;
        } else {
            c.torsions[k - 6] = v;
        }
    }

    #[test]
    fn the_cpu_population_matches_the_single_conformation_path() {
        let (ligand, maps) = setup();
        let sc = VinaScoring::new();
        let confs = population(8, 0x5EED_0001, ligand.num_torsions());
        let (energies, backend, skip) = evaluate_population(&ligand, &maps, &sc, &confs, false);
        assert_eq!(backend, PopulationBackend::Cpu);
        assert!(skip.is_none());
        assert_eq!(energies.len(), confs.len());

        let mut ctx = ScoringContext::new(&ligand, &maps, &sc);
        for (conf, got) in confs.iter().zip(energies.iter()) {
            assert!(
                (ctx.energy(conf) - got).abs() < 1e-12,
                "batched and single paths must agree exactly"
            );
        }
    }

    #[test]
    fn asking_for_the_cpu_never_reports_a_gpu_fallback() {
        let (ligand, maps) = setup();
        let sc = VinaScoring::new();
        let confs = population(4, 7, ligand.num_torsions());
        let (_, backend, skip) = evaluate_population(&ligand, &maps, &sc, &confs, false);
        assert_eq!(backend, PopulationBackend::Cpu);
        assert!(
            skip.is_none(),
            "no fallback should be reported when none happened"
        );
    }

    /// The whole point of the GPU path is that it returns the same numbers.
    /// A backend that silently returns something else is worse than no backend.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_gpu_population_agrees_with_the_cpu_population() {
        if !crate::gpu::is_available() {
            eprintln!("skipping: no GPU adapter on this machine");
            return;
        }
        let (ligand, maps) = setup();
        let sc = VinaScoring::new();
        let confs = population(32, 0xBEEF_0042, ligand.num_torsions());

        let (cpu, _, _) = evaluate_population(&ligand, &maps, &sc, &confs, false);
        let (gpu, gpu_backend, skip) = evaluate_population(&ligand, &maps, &sc, &confs, true);

        if let Some(skip) = skip {
            eprintln!("GPU path declined ({skip}); CPU result is still authoritative");
            assert_eq!(gpu_backend, PopulationBackend::Cpu);
            return;
        }

        assert!(
            gpu_backend.is_gpu(),
            "the GPU was available and asked for, so it must have run"
        );
        assert_eq!(gpu.len(), cpu.len());
        for (i, (a, b)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "conformation {i}: GPU {a} vs CPU {b}");
        }
    }

    /// An empty population must not reach the GPU and must not panic.
    #[cfg(feature = "gpu")]
    #[test]
    fn an_empty_population_is_handled() {
        let (ligand, maps) = setup();
        let sc = VinaScoring::new();
        let (energies, backend, _) = evaluate_population(&ligand, &maps, &sc, &[], true);
        assert!(energies.is_empty());
        assert_eq!(backend, PopulationBackend::Cpu);
    }

    /// A ligand larger than one workgroup cannot be batched, and the fallback
    /// has to be silent-but-correct rather than wrong.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_large_ligand_falls_back_to_the_cpu() {
        // Replicate the whole molecule into one oversized blob, so the ligand
        // is genuinely larger than a single workgroup.
        //
        // The bond graph has to be rebuilt after appending atoms: the neighbour
        // table is sized from the bonds, so appending atoms without
        // re-perceiving them leaves a molecule whose neighbour table is shorter
        // than its atom list — which is an out-of-bounds index, not a graceful
        // error.
        fn blob() -> Molecule {
            let (parsed, _) = parse_butane();
            let template = parsed.atoms;
            // 40 copies on a 4x4x3 lattice. The spacing matters: butane spans
            // 3.65 A in x and 1.3 A in y, so a 3.0 A pitch leaves at least
            // 1.7 A between any two copies. Packing the copies 0.1 A apart --
            // which is what this used to do -- put atoms on *identical*
            // coordinates (copy k's C1 against copy k+15's C2), so the blob
            // was a physically impossible structure. Nothing complained,
            // because the engine scored it anyway and returned finite
            // numbers. A fixture that cannot be docked is not a useful test.
            //
            // Every atom comes from the lattice: keeping the parsed template
            // as atoms 0..4 and *also* placing a copy at lattice origin would
            // put the same atom in the molecule twice.
            let mut atoms: Vec<crate::types::Atom> = Vec::with_capacity(40 * template.len());
            for k in 0..40 {
                let offset = [
                    3.0 * (k % 4) as f64,
                    3.0 * ((k / 4) % 4) as f64,
                    3.0 * (k / 16) as f64,
                ];
                for a in &template {
                    let mut a = a.clone();
                    for axis in 0..3 {
                        a.coord[axis] += offset[axis];
                    }
                    atoms.push(a);
                }
            }
            // Centre it, so the 14 Å box around the origin actually contains it.
            let n = atoms.len() as f64;
            for axis in 0..3 {
                let mean: f64 = atoms.iter().map(|a| a.coord[axis]).sum::<f64>() / n;
                for a in &mut atoms {
                    a.coord[axis] -= mean;
                }
            }
            let mut mol = Molecule::from_atoms(atoms).expect("a lattice of butane is valid");
            mol.perceive_bonds()
                .expect("bond perception should succeed");
            mol.assign_ring_membership()
                .expect("ring perception should succeed");
            mol
        }

        fn rotatable_pairs(mol: &Molecule) -> Vec<(usize, usize)> {
            mol.bonds
                .iter()
                .filter(|b| b.rotatable)
                .map(|b| (b.i, b.j))
                .collect()
        }

        let receptor = Receptor::from_molecule(blob()).expect("receptor");
        let box_ = GridBox::centered([0.0, 0.0, 0.0], [14.0, 14.0, 14.0]).expect("box");
        let sc = VinaScoring::new();
        let maps = receptor.precalculate(&box_, &sc, 0.375).expect("maps");

        let big = blob();
        let big_torsions = rotatable_pairs(&big);
        let ligand = Ligand::from_molecule_with(big, &big_torsions).expect("ligand");
        assert!(
            ligand.len() > crate::gpu::WORKGROUP_SIZE as usize,
            "the test ligand must actually exceed a workgroup, got {}",
            ligand.len()
        );
        let confs = population(2, 99, ligand.num_torsions());
        let (gpu, backend, skip) = evaluate_population(&ligand, &maps, &sc, &confs, true);
        assert_eq!(backend, PopulationBackend::Cpu);
        assert!(skip.is_some());
        let (cpu, _, _) = evaluate_population(&ligand, &maps, &sc, &confs, false);
        assert_eq!(gpu, cpu, "a fallback must still produce the CPU answer");
    }

    #[test]
    fn the_intramolecular_split_is_the_sum_of_its_parts() {
        let (ligand, maps) = setup();
        let sc = VinaScoring::new();
        let confs = population(4, 0xABCD_0000, ligand.num_torsions());
        let mut ctx = ScoringContext::new(&ligand, &maps, &sc);
        for conf in &confs {
            ctx.apply(conf);
            let intra = ctx.intramolecular_energy_at_current_coords();
            let full = ctx.energy(conf);
            let (breakdown, _) = ctx.evaluate_full(conf);
            assert!((intra - breakdown.intramolecular).abs() < 1e-12);
            assert!(
                (full - breakdown.total(sc.intramolecular_scale())).abs() < 1e-12,
                "the split must recombine into the same total"
            );
        }
    }

    #[test]
    fn the_test_ligand_really_has_torsions() {
        // Guards the population fixture: a ligand with no torsions would make
        // every conformation in these tests differ only in rigid placement, and
        // the torsion half of the gradient would never be exercised.
        let (ligand, _) = setup();
        assert!(
            ligand.num_torsions() > 0,
            "the fixture ligand must have at least one torsion"
        );
    }
}
