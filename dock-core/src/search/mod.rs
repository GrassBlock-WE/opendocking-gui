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

/// Odd multiplier that turns one user seed into a spread of per-walk seeds.
///
/// **This constant is a reproducibility contract, not a tuning knob.** Every
/// seeded run in this project derives its per-walk and per-island generators as
/// `base_seed ^ index * WALK_SEED_STRIDE`, so changing this value changes the
/// conformations every seeded run explores and therefore changes every
/// published number that came from a seeded run — energies, RMSDs, contact
/// counts, the distributions in `tests/convergence_budget.rs`, and anything a
/// user re-runs from the workbench's seed box. Nothing about the *search* gets
/// better or worse; every number just moves.
///
/// It is declared once, here, and both search modules name it rather than
/// repeating the literal. That is not tidiness. The literal used to appear in
/// five places, and a digit changed in one of them would have been invisible:
/// the other four would still be right, every test would still be green, and
/// the two strategies would have quietly stopped deriving the same streams.
/// A dropped digit *separator*, by contrast, is harmless — it is the same
/// value — which is why the mutation worth proving is a changed digit and not
/// a reformatted one.
///
/// `the_walk_seed_stride_is_the_value_the_published_numbers_were_produced_with`
/// pins the value, and
/// `no_search_module_carries_a_second_copy_of_the_seed_stride` keeps the copies
/// from coming back.
pub const WALK_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Odd multiplier that separates the LGA islands' generators from each other.
///
/// A second constant rather than a different use of [`WALK_SEED_STRIDE`]: the
/// two are unrelated numbers that happen to be spelled in the same shape, and
/// tying them together would assert a relationship that does not exist. Same
/// contract, though — changing it moves every seeded LGA population.
pub const ISLAND_SEED_STRIDE: u64 = 0x1234_5678_9ABC_DEF0;

/// Penalty charged, per atom, per ångström, for leaving the search box.
///
/// The box is a hard constraint in AutoDock, and rather than trusting the
/// sampler never to violate it, [`ScoringContext::evaluate_full`] charges this
/// much for every atom whose position the maps cannot interpolate. The charge
/// is **this much per ångström of violation**, so the energy of an out-of-box
/// atom is `OUT_OF_BOX_PENALTY x (distance outside)` and the gradient is the
/// derivative of that — which is the whole reason for the per-ångström
/// phrasing. What the term claims, and what it used to get wrong:
///
/// * **The energy is a ramp, and the gradient is its derivative.** An atom 1 Å
///   outside is charged [`OUT_OF_BOX_PENALTY`]; one 5 Å outside is charged five
///   times that. This is new: the charge used to be a **step**, the full amount
///   at the boundary and the same amount 2000 Å out, which made the penalty
///   invisible to the line search — every trial step outside cost exactly what
///   the start cost, no step size satisfied Armijo, and a pose with a
///   full-size gradient was stranded exactly where it stood. `docs/VERIFICATION.md`
///   defect 38 records the step behaviour from the test-fixture side, and the
///   number it quotes is a property of the old form.
/// * **The gradient points further out, so descent comes back.** For an atom
///   outside on the +x face the reported `dE/dx` is **positive**: the energy
///   rises as the atom moves out, steepest descent follows the negative
///   gradient, and the atom moves back toward the box. The sign was the other
///   way round for the life of the term — `dE/dx` measured −1000 at x = 7 in a
///   6 Å box — so descent was *pointed* out of the box. Corrected, and measured
///   in
///   `an_out_of_box_atom_is_pulled_back_in_by_its_descent_direction`.
/// * **The violation is measured per axis, against the faces, not from the
///   centre.** An atom outside only on +x is corrected on +x. The old direction
///   was the unit vector to the box *centre*, so an atom outside through a
///   corner was pushed diagonally through the interior of the box, along axes
///   where it was not out of bounds at all.
/// * **The magnitude is a plateau rather than a dial.** `line_search` caps the
///   first trial at `max_step_norm` (4 Å) and a penalty-sized gradient saturates
///   that cap immediately, so any penalty above roughly 8 produces the identical
///   first trial. What 1000.0 still decides is the *energy* a caller reads off
///   an out-of-box pose, and that energy now scales with how far out it is.
///
/// The value is documented at `docs/API.md:204`, and that line states the
/// constant *and* the per-ångström per-axis shape of the term it multiplies --
/// so a reader there learns the magnitude and the mechanism together. The pin
/// that ties the two together is `OUT_OF_BOX_DOC` in this file's tests.
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
    /// How many atoms of the last evaluation fell outside the grid.
    ///
    /// This used to be a `bool` set on the same branch that charges
    /// [`OUT_OF_BOX_PENALTY`], and the distinction it lost is the whole point
    /// of carrying it. "One atom poked through a face by 0.2 Å" and "not one
    /// atom of this pose is in the box at all" are the same `true`, and only the
    /// second of those means the reported number describes nothing the caller
    /// asked for. It was a count all along: the loop that sets it visits every
    /// atom and branches once per atom, so nothing is computed to obtain it.
    atoms_outside: usize,
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
            atoms_outside: 0,
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
        self.atoms_outside > 0
    }

    /// How many atoms the most recent evaluation placed outside the grid.
    ///
    /// Read this, not [`ScoringContext::last_outside`], when the question is
    /// whether a pose is *entirely* outside: that is
    /// `atoms_outside() == ligand.len()`, a count against a count, and it needs
    /// no threshold to express. The same branch of the same loop increments
    /// this and adds [`OUT_OF_BOX_PENALTY`], so the count and the charge can
    /// never describe different poses.
    pub fn atoms_outside(&self) -> usize {
        self.atoms_outside
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
        self.atoms_outside = 0;

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
                    self.atoms_outside += 1;
                    // How far outside the box this atom is, per axis, and the
                    // sum. Measured against the *faces* rather than from the
                    // centre: an atom outside on +x is one axis wrong, and a
                    // penalty that pulled it along the diagonal towards the
                    // centre would shove it through the interior of the box to
                    // get there.
                    let b = self.maps.grid_box();
                    let per_axis = crate::grid::out_of_box_violation_per_axis(p, &b);
                    for k in 0..3 {
                        if per_axis[k] > 0.0 {
                            // The gradient of that ramp, pointing further out on
                            // the axis this atom left by, so descent is inward.
                            let outward = if p[k] > b.max[k] { 1.0 } else { -1.0 };
                            inter_grad[i][k] += OUT_OF_BOX_PENALTY * outward;
                        }
                    }
                    let violation: f64 = per_axis.iter().sum();
                    if violation > 0.0 {
                        inter += OUT_OF_BOX_PENALTY * violation;
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
///
/// # What a decline reports
///
/// A **decline** is a call that asked for the GPU (`prefer_gpu`) and did not get
/// one. The contract is:
///
/// - `prefer_gpu == true` — exactly one of these holds: the GPU ran
///   ([`PopulationBackend::Gpu`] and `skip` is `None`), or it did not
///   ([`PopulationBackend::Cpu`] and `skip` is `Some`) and the [`GpuSkip`] says
///   why. Never both, never neither.
/// - `prefer_gpu == false` — `skip` is `None`, always. Nobody asked, so nothing
///   was declined, and a reason here would report a fallback that did not
///   happen.
///
/// `backend` alone already answers *what ran*; [`GpuSkip`] answers *why the
/// request was not honoured*, which is the half a caller needs in order to stop
/// asking. The two are not the same claim: a build compiled without the `gpu`
/// feature declines with "this build was compiled without the gpu feature",
/// while a build that has it and cannot open an adapter declines with "no usable
/// GPU adapter" — opposite repairs for the same missing speedup.
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
    {
        // A build without the feature cannot reach the kernel at all, so a
        // caller who asked for the GPU and got the CPU has been declined, and
        // a decline is reported rather than swallowed. This used to be
        // `let _ = prefer_gpu;`, which dropped the request on the floor: the
        // report said `backend: "cpu"` and `gpu_skip_reason: None`, which is
        // indistinguishable from having run on purpose, so the only thing a
        // caller could conclude was that their request had been ignored --
        // an unexplained slowdown with no cause attached.
        if prefer_gpu {
            let (energies, _, _) = cpu_population(ligand, maps, scoring, conformations);
            return (
                energies,
                PopulationBackend::Cpu,
                Some(GpuSkip {
                    reason: "this build was compiled without the gpu feature",
                }),
            );
        }
    }

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
            // **The narrowing, and the one input it declines.** `GridMaps`
            // stores `f32` and `Batch::coords` is `Vec<f32>`, so a coordinate
            // has to survive the trip through `f32` to reach the kernel.
            //
            // Two things can go wrong on that trip, and the second is the one
            // that bites first. A Rust float cast saturates, so a coordinate
            // above `f32::MAX` (about 3.4e38 A) becomes `f32::INFINITY` on the
            // way in. But well before that, an atom *outside* the box is
            // charged `OUT_OF_BOX_PENALTY * violation` in `f32` inside
            // `sample()`, and with the penalty at 1000.0 that multiply
            // overflows once the violation exceeds `f32::MAX / 1000`, about
            // **3.4e35 A**. So the representable range is set by the penalty,
            // not by the coordinate, and a test against `f32::MAX` alone would
            // decline the coordinate while letting through the one that
            // actually returns an infinity.
            //
            // Both used to pass, and the two backends then disagreed by an
            // unbounded amount on a row that was still returned and still
            // ranked. Measured on an RTX 3050, a conformation translated
            // 1e38 A along x scores 1.0e41 on the CPU -- the out-of-box
            // penalty, large but finite, because the CPU narrows nothing and
            // computes in f64 -- and `inf` on the GPU.
            //
            // It is declined rather than refused because the coordinate itself
            // is not wrong: 1e38 A is a perfectly finite `f64`, so
            // `first_non_finite` and `non_finite_coordinate_refusal` correctly
            // let it through, and a CPU-only caller gets a usable number. What
            // the GPU cannot do is *represent* the penalty for it. So the GPU
            // declines, the CPU answers, and the caller is told why -- the same
            // contract as every other decline here, and the reason the check is
            // here rather than left to the cast.
            // The narrowing itself moved into `Batch::from_coords`, which takes
            // `f64` and does the `as f32` itself. It has to live there rather
            // than here: the in-box test is discontinuous, so a coordinate one
            // ULP apart can send the two backends down different branches, and
            // the host needs the un-narrowed value to take that decision on --
            // it uploads the answer as a per-atom flag rather than letting the
            // kernel re-derive it from the narrowed value. See the `Batch`
            // docs.
            let narrow = [p[0], p[1], p[2]];
            // **What has to survive the narrowing is the whole row, not one
            // coordinate of it.** The first version of this check formed
            // `OUT_OF_BOX_PENALTY * coordinate` per axis, which is exact for a
            // one-atom ligand translated along one axis and **too permissive
            // for every other shape**, for two reasons that are both in the
            // shader rather than here:
            //
            //  * `sample()` sums the per-axis excesses -- `dot(max(per_axis,
            //    vec3(0)), vec3(1))`, `energy.wgsl` -- and then multiplies by
            //    the penalty, so a coordinate equal on all three axes costs
            //    **three** times what the per-axis check budgets for it. At the
            //    magnitude the per-axis check admits, a diagonal escape
            //    returned `inf` from the GPU against a finite CPU number.
            //  * `main()` reduces `partial[lid]` across the workgroup, one
            //    entry per atom, so a ligand of `n` atoms costs `n` times as
            //    much again.
            //
            // Measured on an RTX 3050 from a `--features gpu` build, the
            // per-axis check admitted `3.402823e35` A, against safe ceilings of
            // `1.134274e35` A for a diagonal one-atom escape (3x), `5.671e34` A
            // for two atoms (6x) and `1.772e33` A for a 64-atom ligand (192x).
            // So the divisor below is the row's worst case and not the
            // coordinate's.
            const AXES: f64 = 3.0;
            let row = OUT_OF_BOX_PENALTY * AXES * n_atoms as f64;
            // Formed in `f64` and only then narrowed, because that is the order
            // the shader does it in: the penalty, the axes and the atom count
            // are all `f32` arithmetic once the kernel has them.
            let overflows = p
                .iter()
                .any(|c| !c.is_finite() || !((row * c.abs()) as f32).is_finite());
            if overflows {
                return Err(GpuSkip {
                    reason: "a ligand coordinate is too large for the f32 grid",
                });
            }
            coords.extend_from_slice(&narrow);
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
    use crate::search::lbfgs::LbfgsConfig;
    use crate::types::{Atom, AtomType, Element, Molecule};

    /// Assert that a constant still equals the value the documentation states
    /// for it, and quote that line when it does not.
    ///
    /// A copy of the helper in `scoring::tests` rather than a shared one: it is
    /// five lines, and a shared test helper would mean a `pub(crate)` item in
    /// production code for the sake of a test.
    fn assert_pinned<T: PartialEq + std::fmt::Debug>(
        constant: &str,
        found: T,
        documented: T,
        doc: &str,
    ) {
        if found != documented {
            panic!(
                "{constant} is {found:?}, but the specification documents \
                 {documented:?}.\n\
                 documented: {doc}\n\
                 The constant and the line that states it have to change together. \
                 A number nobody re-derived is worse than a number nobody wrote down."
            );
        }
    }

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

    /// The reporting contract, pinned here so it is enforced by `cargo test` on
    /// a machine that never runs `scripts/gpu_cpu_parity_check.py`.
    ///
    /// This is the assertion that gate makes, in the only place CI actually
    /// runs. The `#[cfg(not(feature = "gpu"))]` arm used to be
    /// `let _ = prefer_gpu;`, so on a CPU-only build an explicit `use_gpu=True`
    /// came back as `backend: "cpu"` with `gpu_skip_reason: None` — a report
    /// that named what ran and said nothing about the request being dropped.
    /// Nothing in `cargo test` noticed, because the only test that asked for the
    /// GPU and expected a decline was itself `#[cfg(feature = "gpu")]`, so it
    /// never compiled on the configuration it was about.
    ///
    /// Written to hold in **both** worlds: with the feature and an adapter this
    /// run genuinely reaches the kernel and `skip` is `None`, and without it
    /// `skip` is `Some`. The one thing asserted is the XOR, which is the part
    /// that was false.
    #[test]
    fn a_decline_is_always_self_explaining() {
        let (ligand, maps) = setup();
        let sc = VinaScoring::new();
        let confs = population(4, 0xDEC1_11E5, ligand.num_torsions());

        let (asked, asked_backend, asked_skip) =
            evaluate_population(&ligand, &maps, &sc, &confs, true);
        let ran = asked_backend.is_gpu();
        let said_why = asked_skip
            .as_ref()
            .is_some_and(|s| !s.reason.trim().is_empty());
        assert_eq!(
            ran,
            !said_why,
            "'the gpu ran' XOR 'a non-empty reason why it did not' is the contract \
             that makes a decline audible: backend={asked_backend:?}, \
             gpu_skip_reason={:?}",
            asked_skip.map(|s| s.reason)
        );

        // The other half: not asking must not manufacture a decline. This is a
        // different claim from "asked and was refused", and one field carries
        // both correctly because the caller knows which one they made.
        let (_, quiet_backend, quiet_skip) =
            evaluate_population(&ligand, &maps, &sc, &confs, false);
        assert_eq!(quiet_backend, PopulationBackend::Cpu);
        assert!(
            quiet_skip.is_none(),
            "a call that never asked for the gpu declined nothing, so a reason \
             here would report a fallback that did not happen: {:?}",
            quiet_skip
        );

        // A declined call must still return the CPU's numbers: the reason is
        // added information, never a licence to answer differently.
        if let Some(skip) = &asked_skip {
            let (unasked, _, _) = evaluate_population(&ligand, &maps, &sc, &confs, false);
            assert_eq!(
                asked, unasked,
                "a call declined for {skip:?} must return the CPU answer exactly"
            );
        }

        // And the reason must name a cause the caller can act on, because the
        // two reasons mean opposite things: a missing feature is fixed by
        // getting a different wheel, a missing adapter by fixing the driver.
        #[cfg(not(feature = "gpu"))]
        {
            let skip = asked_skip.expect("a CPU-only build must decline with a reason");
            assert!(
                skip.reason.contains("gpu feature"),
                "the reason must name the missing feature rather than the missing \
                 adapter, because the two call for opposite repairs; got {skip:?}"
            );
        }
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
    /// has to be correct **and say why** rather than be wrong or be silent.
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
    fn the_walk_seed_stride_is_the_value_the_published_numbers_were_produced_with() {
        // The pin that makes a changed digit visible. The literal appears here
        // and in the declaration and nowhere else, so this is the second of two
        // places and the review is the defence: a change to *both* cannot be
        // caught by a test that one of them is written in, and pretending
        // otherwise would be a pin that certifies itself. What this does buy is
        // that the ordinary accident — someone tuning the literal and not
        // noticing that every published seeded number moved with it — is red.
        assert_eq!(
            WALK_SEED_STRIDE, 0x9E37_79B9_7F4A_7C15,
            "WALK_SEED_STRIDE is {WALK_SEED_STRIDE:#018x}, not the value every \
             published seeded number in this project was produced with. Changing \
             it moves every seeded result: the workbench's poses for a given seed, \
             the benchmark timings, and the distributions measured by \
             `tests/convergence_budget.rs`. If the change is intended, it is a \
             release-note item, not a code edit"
        );
        assert_eq!(
            ISLAND_SEED_STRIDE, 0x1234_5678_9ABC_DEF0,
            "ISLAND_SEED_STRIDE is {ISLAND_SEED_STRIDE:#018x}, not the value the \
             seeded LGA populations were produced with. Same contract as \
             WALK_SEED_STRIDE, and the same reason: it moves numbers that are \
             already published"
        );
    }

    #[test]
    fn no_search_module_carries_a_second_copy_of_the_seed_stride() {
        // The guard that makes the pin mean something. Five copies of the
        // literal used to sit in this crate, so a digit changed in one of them
        // left four correct copies and no red anywhere; the fix is one
        // declaration, and this is what stops the copies creeping back in.
        //
        // It reads the sources as text rather than trying to inspect the
        // compiled program, because the failure is textual: a re-inlined
        // literal is a perfectly good value to the compiler and the whole
        // problem is that it is a second source of truth. `include_str!` keeps
        // it honest in the other direction too — delete a search module and
        // this stops compiling rather than quietly checking nothing.
        //
        // `search/mod.rs` itself is the one file not checked, because it is
        // where the declaration and this pin live and both spell the literal
        // on purpose.
        for (name, source) in [
            ("monte_carlo.rs", include_str!("monte_carlo.rs")),
            ("lga.rs", include_str!("lga.rs")),
        ] {
            for (const_name, literal) in [
                ("WALK_SEED_STRIDE", "0x9E37_79B9_7F4A_7C15"),
                ("ISLAND_SEED_STRIDE", "0x1234_5678_9ABC_DEF0"),
            ] {
                assert!(
                    !source.contains(literal),
                    "{name} spells {literal} out again instead of naming \
                     `super::{const_name}`. A second copy is a second source of \
                     truth: changing one of the two leaves the other right, every \
                     test green, and the strategies quietly deriving different \
                     streams from the same user seed"
                );
            }
            assert!(
                source.contains("WALK_SEED_STRIDE") || source.contains("ISLAND_SEED_STRIDE"),
                "{name} derives no seeds from either stride any more. If that is \
                 intended, the constant is no longer what it says it is"
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

    // --- `OUT_OF_BOX_PENALTY` --------------------------------------------
    //
    // The one scoring constant that never reaches `SCORING.md`, because it is
    // not a scoring-function weight: it is a constraint on the sampler, applied
    // where the maps cannot be read. `docs/API.md:204` states the value, and
    // that line is what the value assertion travels with.
    // The docs state the *mechanism* as well as the value now; they did not
    // when these tests were written, which is why the tests exist. The only written-down description of the shape was
    // `docs/VERIFICATION.md` defect 38 ("the penalty is a step, not a ramp"),
    // and that line described the defect rather than an intent -- it is now
    // stale against the code below and wants an owner. The shape these tests
    // pin is the one the original `ScoringContext::evaluate` docstring asked
    // for: a *linear* penalty that leaves the optimiser pointing back inside.
    //
    // The value `docs/API.md` documents.
    const OUT_OF_BOX_DOC: &str = "API.md:204: `OUT_OF_BOX_PENALTY = 1000.0`. -- this is the \
         documentation of the constant, and it states the value together with the \
         per-angstrom, per-axis shape of the term it multiplies, which is what the \
         three assertions below pin";

    /// A one-atom ligand, so `conf.position` places its only atom exactly there
    /// and a penalty can be counted per atom without reasoning about where a
    /// kinematic tree puts the rest of a real ligand.
    fn single_carbon() -> Ligand {
        let mol = Molecule::from_atoms(vec![Atom::new(
            1,
            [0.0, 0.0, 0.0],
            Element::C,
            AtomType::CH,
        )])
        .expect("a single carbon is a valid molecule");
        assert_eq!(mol.len(), 1, "the fixture must really be one atom");
        Ligand::from_molecule(mol).expect("a single carbon is a ligand")
    }

    fn at(position: [f64; 3], num_torsions: usize) -> Conformation {
        Conformation {
            position,
            orientation: [0.0; 3],
            torsions: vec![0.0; num_torsions],
        }
    }

    /// A one-atom ligand, placed outside the box at three distances, is charged
    /// the penalty **per ångström of violation** — and the intramolecular half is
    /// untouched, because the penalty is added to `intermolecular` before the
    /// scale is applied and `intramolecular_scale` is 0.006 (a six-hundred-fold
    /// difference if it were ever added to the wrong half).
    ///
    /// This used to assert the opposite shape: one flat charge at every distance,
    /// which is what made the term invisible to the line search. The distance
    /// scaling is not a cosmetic change to a number — it is the difference
    /// between a penalty the optimiser can descend and one it cannot.
    #[test]
    fn the_out_of_box_penalty_is_per_angstrom_of_violation() {
        assert_pinned(
            "OUT_OF_BOX_PENALTY",
            OUT_OF_BOX_PENALTY,
            1000.0,
            OUT_OF_BOX_DOC,
        );
        let (_, maps) = setup();
        let ligand = single_carbon();
        let sc = VinaScoring::new();
        let mut ctx = ScoringContext::new(&ligand, &maps, &sc);
        let half = maps.grid_box().size()[0] / 2.0;

        for x in [7.0, 20.0, 2000.0] {
            let (b, _) = ctx.evaluate_full(&at([x, 0.0, 0.0], 0));
            assert!(
                ctx.last_outside(),
                "the atom at x = {x} has to actually be outside the {half} A half-box, \
                 or this is testing the interior"
            );
            let violation = x - half;
            assert!(
                (b.intermolecular - OUT_OF_BOX_PENALTY * violation).abs() < 1e-6,
                "at x = {x}: {violation} A outside must cost \
                 {} kcal/mol, got {}. The charge is per angstrom, so a pose that \
                 overhangs the box costs what it overhangs by",
                OUT_OF_BOX_PENALTY * violation,
                b.intermolecular
            );
            assert_eq!(
                b.intramolecular, 0.0,
                "a one-atom ligand has no intramolecular pairs, so this also pins \
                 that the penalty is not smuggled into the scaled half"
            );
            assert!(
                (b.total(sc.intramolecular_scale()) - OUT_OF_BOX_PENALTY * violation).abs() < 1e-6,
                "at x = {x}: the total must be the penalty and nothing else, or a \
                 caller cannot recognise an out-of-box pose from its energy"
            );
        }

        // Per atom, and additive. Butane is used here rather than the single
        // carbon because the count of outside atoms is then a real property of
        // the placement, measured rather than assumed.
        let (butane, maps) = setup();
        let sc = VinaScoring::new();
        let mut ctx = ScoringContext::new(&butane, &maps, &sc);
        let conf = at([40.0, 0.0, 0.0], butane.num_torsions());
        let (b, _) = ctx.evaluate_full(&conf);
        // `evaluate_full` leaves the expanded coordinates behind, so the count
        // is read from the conformation that produced the energy above.
        let half = maps.grid_box().size()[0] / 2.0;
        let outside = ctx
            .current_coords()
            .iter()
            .filter(|p| p[0].abs() > half || p[1].abs() > half || p[2].abs() > half)
            .count();
        assert_eq!(
            outside,
            butane.len(),
            "the fixture must place every butane atom outside, or the per-atom \
             arithmetic below is not being exercised"
        );
        assert!(
            ctx.last_outside(),
            "the clash flag has to agree with the coordinate count"
        );
        // Per atom, additive, and each atom charged for its own overhang, per
        // axis. The chain is a butane, so the atoms are not all the same
        // distance out and the sum is a real quantity rather than
        // `outside x penalty`. Violations are summed over axes too, so an atom
        // outside two faces pays for both and is pulled in on both at once.
        // The coordinates are read here, before the next evaluation overwrites
        // them: `evaluate_full` leaves behind the expansion of the conformer it
        // just scored, and this sum has to be the one that produced `b`.
        //
        // The clamp is `max(0, max(p - max, min - p))` per axis, taken from the
        // box rather than from a half-extent: an axis the atom is *inside* has
        // to contribute nothing, and getting that wrong charges a pose for
        // every axis it is lawfully in.
        let gbox = maps.grid_box();
        let sum_violation: f64 = ctx
            .current_coords()
            .iter()
            .map(|p| {
                (0..3)
                    .map(|k| (p[k] - gbox.max[k]).max(gbox.min[k] - p[k]).max(0.0))
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (b.intermolecular - OUT_OF_BOX_PENALTY * sum_violation).abs() < 1e-6,
            "{outside} atoms outside with a summed violation of {sum_violation:.4} A \
             must cost {}, got {}. Each atom pays for its own overhang and its own \
             axes, so a chain poking out sideways pays more than one reaching \
             straight out",
            OUT_OF_BOX_PENALTY * sum_violation,
            b.intermolecular
        );
        // And the charge rises with the overhang: the same ligand, 20 A further
        // out. A charge that does not move with the overhang is a step again.
        let (b_farther, _) = ctx.evaluate_full(&at([60.0, 0.0, 0.0], butane.num_torsions()));
        assert!(
            b_farther.intermolecular > b.intermolecular,
            "the same ligand 20 A further out costs {}, against {} nearer",
            b_farther.intermolecular,
            b.intermolecular
        );
    }

    /// `atoms_outside` is a count of atoms, not the boolean it replaced, and it
    /// has to be the *same* count an independent walk over the coordinates
    /// produces. Both directions, because either alone is satisfiable by a
    /// constant: a field hard-wired to `1` passes "the flag is set" and a field
    /// hard-wired to `0` passes "nothing is outside".
    #[test]
    fn the_out_of_box_count_agrees_with_the_coordinates_it_was_taken_from() {
        let (butane, maps) = setup();
        let sc = VinaScoring::new();
        let mut ctx = ScoringContext::new(&butane, &maps, &sc);
        let gbox = maps.grid_box();
        let n = butane.len();
        assert!(
            n > 1,
            "the fixture needs several atoms for a count to mean anything"
        );

        // Expected values, counted from the expanded coordinates rather than
        // from the field under test. `at(x, ..)` walks the whole chain out along
        // +x, so x picks how many atoms have cleared the face.
        let cases = [(-100.0, n), (-4.0, 0)];
        for (x, want) in cases {
            let conf = at([x, 0.0, 0.0], butane.num_torsions());
            ctx.evaluate_full(&conf);
            let coords = ctx.current_coords().to_vec();
            let counted = coords
                .iter()
                .filter(|p| crate::grid::out_of_box_violation(**p, &gbox) > 0.0)
                .count();
            assert_eq!(
                counted, want,
                "fixture: at x = {x}, {want} of {n} butane atoms are outside. If this \
                 assertion fails the fixture moved and every number below is about \
                 a different placement"
            );
            assert_eq!(
                ctx.atoms_outside(),
                counted,
                "at x = {x}: the engine counted {} atoms outside and the coordinates \
                 say {counted} are. These are the same loop, so a disagreement means \
                 the count and the penalty it accompanies are describing different \
                 poses",
                ctx.atoms_outside()
            );
            // The boolean is now a derived view of the count, not a second
            // piece of state that can disagree with it.
            assert_eq!(
                ctx.last_outside(),
                counted > 0,
                "at x = {x}: the flag and the count are the same fact and must not \
                 be able to answer differently"
            );
        }

        // And the count is per evaluation, not cumulative: a placement wholly
        // inside after a wholly-outside one must not inherit the earlier count.
        let _ = ctx.evaluate_full(&at([-100.0, 0.0, 0.0], butane.num_torsions()));
        assert_eq!(
            ctx.atoms_outside(),
            n,
            "the far placement puts every atom outside"
        );
        let _ = ctx.evaluate_full(&at([0.0, 0.0, 0.0], butane.num_torsions()));
        assert_eq!(
            ctx.atoms_outside(),
            0,
            "a fresh evaluation wholly inside the box must report 0 outside atoms. \
             A counter that only ever climbs would satisfy both assertions above \
             once the fixture happened to be visited in that order"
        );
    }

    /// The gradient the optimiser is handed **is** the gradient of the energy it
    /// adds, and it points along the axis the atom actually left by, away from
    /// the box — so steepest descent brings it back.
    ///
    /// This is the inverse of what it used to assert. It pinned the term being
    /// broken: the gradient was the full penalty on a unit vector towards the
    /// box *centre* at every distance, while the energy was flat, so the two
    /// were not the gradient of one another and the optimiser was told something
    /// the energy did not agree with. A penalty that fails that test is not a
    /// soft constraint; it is a noise source with a large number in it.
    #[test]
    fn the_out_of_box_penalty_is_the_gradient_of_the_energy_it_adds() {
        let (_, maps) = setup();
        let ligand = single_carbon();
        let sc = VinaScoring::new();
        let mut ctx = ScoringContext::new(&ligand, &maps, &sc);
        let half = maps.grid_box().size()[0] / 2.0;

        // Escaped through a corner: the correction is on the worst axis alone,
        // so the two axes this atom is *not* out of bounds on get nothing. The
        // old code pulled it diagonally towards the centre, through the interior
        // of the box, along axes where it was perfectly legal.
        let corner = [half + 14.0, half + 14.0, 0.0];
        let (_, g) = ctx.evaluate(&at(corner, 0));
        assert!(ctx.last_outside());
        assert!(
            (g[0] - OUT_OF_BOX_PENALTY).abs() < 1e-9 && (g[1] - OUT_OF_BOX_PENALTY).abs() < 1e-9,
            "a corner escape corrects along the axes it left by: gradient is \
             ({:+.9}, {:+.9}, {:+.9}), and both violated axes read +{OUT_OF_BOX_PENALTY}",
            g[0],
            g[1],
            g[2]
        );
        assert!(
            g[2].abs() < 1e-12,
            "axis z is inside the box, so it must contribute nothing: got {:+.9}. \
             A pull towards the box centre gave a large component here, which is \
             how an atom leaves through one face and gets shoved further out of \
             another",
            g[2]
        );
        // Outward, on the axis, so descent is inward.
        assert!(
            g[0] > 0.0 && g[1] > 0.0,
            "the gradient points further out, so `-grad` points back in: ({:+.9}, {:+.9})",
            g[0],
            g[1]
        );

        // The other half of the claim, and the number that used to be the tell:
        // one ångström further out costs exactly one penalty more. A step cannot
        // show that difference; a ramp can, and this is the difference.
        let (far, _) = ctx.evaluate(&at([half + 15.0, 0.0, 0.0], 0));
        let (near, _) = ctx.evaluate(&at([half + 14.0, 0.0, 0.0], 0));
        assert!(
            (far - near - OUT_OF_BOX_PENALTY).abs() < 1e-6,
            "1 A further out costs {far}, 1 A nearer costs {near}: the difference is \
             {diff:.6} and must be one penalty. This is the assertion that says the \
             energy and the gradient agree; the flat version of this test asserted \
             they did not",
            diff = far - near
        );
        // And the corner's *energy*, which the gradient above cannot reach: an
        // atom 14 A out on both x and y is 28 A out, not 14. Taking the largest
        // axis instead of the sum halves this, and no guard that only reads the
        // gradient would see it -- the gradient is per-axis either way.
        let (corner_e, _) = ctx.evaluate(&at(corner, 0));
        let (one_face, _) = ctx.evaluate(&at([half + 14.0, 0.0, 0.0], 0));
        assert!(
            (corner_e - one_face - 14.0 * OUT_OF_BOX_PENALTY).abs() < 1e-6,
            "a corner 14 A out on x and y costs {corner_e} against {one_face} for one \
             face: the difference must be 14 penalties, the second face. Charging the \
             larger of the two axes instead of their sum would make it 0"
        );
    }

    /// What the penalty actually does to a pose that has left the box, and what
    /// that costs.
    ///
    /// This is the consequence the other two tests imply, measured on the
    /// optimiser rather than on the energy function: the direction has to put a
    /// pose back inside, and it has to be able to.
    ///
    /// * **A descent step reduces the violation.** One atom 1 A outside a 6 A
    ///   face: `dE/dx` is **positive**, so `-grad` is inward, and `minimize`
    ///   brings it back inside. Before the correction this returned x = +7.0000
    ///   after 33 evaluations -- directed outward, and frozen there, because a
    ///   flat energy left Armijo no acceptable step at all.
    /// * **The energy difference is what lets the step happen.** Two points 1 A
    ///   apart, both outside, differ by exactly one penalty. That difference is
    ///   why a step can be accepted; without it the optimiser is told to move
    ///   somewhere that costs exactly what it already costs.
    /// * **The magnitude is still a plateau for the step length.**
    ///   `line_search` caps the first trial at `max_step_norm`, so what 1000.0
    ///   decides is the *energy* of an out-of-box pose, which now scales with
    ///   the overhang rather than counting atoms.
    /// * **Still not reachable for a pose the search returns.** `dock`
    ///   partitions out-of-box poses out before clustering and
    ///   `monte_carlo::search` rejects a box the ligand cannot fit in. So this is
    ///   a property of the term as written rather than something a user sees in
    ///   `poses.pdbqt` -- which is the one thing that made correcting the sign a
    ///   decision rather than an obvious bug fix.
    #[test]
    fn an_out_of_box_atom_is_pulled_back_in_by_its_descent_direction() {
        let (_, maps) = setup();
        let ligand = single_carbon();
        let sc = VinaScoring::new();
        let cfg = LbfgsConfig::default();
        let cap = cfg.max_step_norm;

        let mut ctx = ScoringContext::new(&ligand, &maps, &sc);

        // --- the direction, which a magnitude test cannot see --------------
        let (e_at_7, g) = ctx.evaluate(&at([7.0, 0.0, 0.0], 0));
        assert!(ctx.last_outside(), "7.0 is outside the 6.0 face");
        assert!(
            (e_at_7 - OUT_OF_BOX_PENALTY).abs() < 1e-6,
            "1 A outside costs one penalty, got {e_at_7}"
        );
        assert!(
            g[0] > 0.0,
            "dE/dx is {:+.6} at x = 7: the energy RISES as the atom moves out, so \
             steepest descent (-grad) is inward. A negative dE/dx here points the \
             search further out, which is what this term did for its whole life: \
             `dE/dx == -1000` is as wrong as `dE/dx == 0`, and an assertion on the \
             magnitude alone would have passed straight over the bug. The direction \
             is the correction; the size was never the question.",
            g[0]
        );
        // The consequence, exactly: the trial the optimiser is directed to take
        // costs precisely what the start costs, because the penalty is a step.
        let (e_at_11, _) = ctx.evaluate(&at([11.0, 0.0, 0.0], 0));
        assert!(
            (e_at_11 - e_at_7 - 4.0 * OUT_OF_BOX_PENALTY).abs() < 1e-6,
            "x = 11 is 4 A further out than x = 7 and costs {e_at_11} against \
             {e_at_7}: the difference must be four penalties. While the charge was \
             a step the two cost the same, no step size satisfied Armijo, and the \
             pose could not move at all"
        );
        let start = at([7.0, 0.0, 0.0], 0);
        let (out, outcome) = {
            let mut eval = |c: &Conformation| ctx.evaluate(c);
            crate::search::lbfgs::minimize(&start, &cfg, &mut eval)
        };
        let (e_out, _) = ctx.evaluate(&out);
        let e_out_1a = e_at_7;
        eprintln!(
            "OUT_OF_BOX: 1 atom at x=7, dE/dx {:+.6}; first trial would be {cap} A; \
             minimise returned E {e_out:+.4} at x {:+.4} after {} iterations and \
             {} evaluations, still outside: {}",
            g[0],
            out.position[0],
            outcome.iterations,
            outcome.evaluations,
            ctx.last_outside()
        );
        assert!(
            out.position[0] < 7.0 - 1e-9,
            "the descent step moved the atom from x = 7 to {:+.6} after {} \
             iterations: a descent step must REDUCE the violation, and this \
             assertion is unsatisfiable while the sign is inverted",
            out.position[0],
            outcome.iterations
        );
        assert!(
            !ctx.last_outside(),
            "the atom ended at x {:+.6}, inside the 6.0 face, so the penalty no \
             longer applies and the pose is scored on its interactions rather \
             than on a wall",
            out.position[0]
        );
        assert!(
            e_out < e_out_1a,
            "and the energy it comes back with is {e_out:+.4} against the \
             {e_out_1a:+.4} it started on: the penalty is a cost the pose can \
             climb down, which is the property a penalty exists for"
        );

        // --- the magnitude does not set the step length --------------------
        let pnorm = g[0..3].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (pnorm - OUT_OF_BOX_PENALTY).abs() < 1e-6,
            "one atom outside is pulled with the penalty itself, |g| = {pnorm:.6}"
        );
        let first_trial = cfg.initial_step.min(cap / pnorm) * pnorm;
        assert!(
            (first_trial - cap).abs() < 1e-9,
            "the first trial is {first_trial:.6} A, not the {cap} A cap: a \
             penalty-sized gradient saturates max_step_norm, so tuning the \
             penalty would not change one step of the trajectory"
        );
        for smaller in [2.0 * cap, 10.0 * cap, OUT_OF_BOX_PENALTY] {
            let p = cfg.initial_step.min(cap / smaller) * smaller;
            assert!(
                (p - cap).abs() < 1e-9,
                "a penalty of {smaller} should give the same capped first trial, \
                 gave {p:.6} A"
            );
        }
    }
}
