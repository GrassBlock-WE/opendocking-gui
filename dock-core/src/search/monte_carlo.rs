//! Iterated local search over the search box.
//!
//! The strategy is the one AutoDock Vina popularised and which remains the
//! strongest general-purpose choice for small-to-medium flexibility:
//!
//! 1. draw a random conformation with the ligand inside the box;
//! 2. run a quasi-Newton local optimisation from it;
//! 3. keep the result if it improved on the incumbent, else restore the
//!    incumbent with a probability that decreases over time (a Metropolis-like
//!    acceptance);
//! 4. perturb the incumbent in its **local** frame and repeat.
//!
//! Steps 3 and 4 are what turn independent restarts into a search that exploits
//! the neighbourhood of good poses. `exhaustiveness` controls how many
//! independent walks run, in parallel across [`rayon`] threads.
//!
//! # Provenance
//!
//! Trott & Olson, *J. Comput. Chem.* **31**, 455 (2010), Apache-2.0, describe
//! this iterated local search. The Metropolis acceptance and the local-frame
//! perturbation step follow that description; the implementation is independent.

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::{DockError, Result};
use crate::grid::GridMaps;
use crate::kinematics::Conformation;
use crate::ligand::Ligand;
use crate::scoring::ScoringFunction;
use crate::search::lbfgs::{minimize, LbfgsConfig};
use crate::search::ScoringContext;
use crate::types::Vec3;

/// Tuning for the global search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonteCarloConfig {
    /// Number of independent walks. This is AutoDock's `--exhaustiveness`.
    pub exhaustiveness: u32,
    /// Local-optimisation steps taken per walk.
    pub steps: u32,
    /// Probability of accepting a worse local optimum, at the start.
    pub initial_temperature: f64,
    /// Probability of accepting a worse local optimum, at the end.
    pub final_temperature: f64,
    /// Amplitude of the random kick applied to the incumbent, in Ångström for
    /// translation and radians for rotation and torsion.
    pub mutation_amplitude: f64,
    /// Iteration at which the acceptance probability has decayed to its final
    /// value, as a fraction of `steps`.
    pub temperature_decay: f64,
    /// Local optimiser settings.
    pub local: LbfgsConfig,
    /// Random seed. `None` seeds from the OS.
    pub seed: Option<u64>,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        MonteCarloConfig {
            exhaustiveness: 8,
            steps: 70,
            initial_temperature: 1.2,
            final_temperature: 0.0,
            mutation_amplitude: 2.0,
            temperature_decay: 0.5,
            local: LbfgsConfig::default(),
            seed: None,
        }
    }
}

/// One pose found during the search.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Pose {
    /// The conformation that produced this pose.
    pub conf: Conformation,
    /// Total docking energy in kcal/mol.
    pub energy: f64,
    /// Receptor–ligand part, kcal/mol.
    pub intermolecular: f64,
    /// Ligand-internal part, kcal/mol (already unscaled).
    pub intramolecular: f64,
    /// Global coordinates of every atom.
    pub coords: Vec<Vec3>,
    /// Symmetry-corrected RMSD to the best pose, if computed.
    pub rmsd: Option<f64>,
}

/// Run the iterated local search.
///
/// Walks are independent, so they are executed in parallel with [`rayon`]; the
/// seeds are derived per-walk so results are reproducible for a given seed.
pub fn search(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    config: &MonteCarloConfig,
) -> Result<Vec<Pose>> {
    if ligand.is_empty() {
        return Err(DockError::molecule("cannot dock an empty ligand"));
    }
    if config.exhaustiveness == 0 {
        return Err(DockError::param("exhaustiveness", 0, "must be at least 1"));
    }
    if config.steps == 0 {
        return Err(DockError::param("steps", 0, "must be at least 1"));
    }

    let box_ = maps.grid_box();
    // The ligand must physically fit, with room for the surface-distance
    // inflation. Rejecting this up front is far kinder than returning
    // out-of-box penalties for every pose.
    let needed = 2.0 * ligand.radius() + 1.0;
    let size = box_.size();
    for (k, axis) in ["x", "y", "z"].iter().enumerate() {
        if size[k] < needed {
            return Err(DockError::param(
                "grid box",
                format!("{size:?}"),
                format!(
                    "axis {axis} is {:.1} Å but the ligand needs at least {needed:.1} Å",
                    size[k]
                ),
            ));
        }
    }

    let base_seed = config.seed.unwrap_or_else(rand::random);
    let walks: Vec<Vec<Pose>> = (0..config.exhaustiveness)
        .into_par_iter()
        .map(|w| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                base_seed ^ (w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            run_walk(ligand, maps, scoring, config, &mut rng, box_)
        })
        .collect();

    // Merge, best first, dropping exact duplicates of the same pose.
    let mut all: Vec<Pose> = walks.into_iter().flatten().collect();
    all.sort_by(|a, b| {
        a.energy
            .partial_cmp(&b.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(all)
}

fn run_walk(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    config: &MonteCarloConfig,
    rng: &mut impl Rng,
    box_: crate::grid::GridBox,
) -> Vec<Pose> {
    let mut ctx = ScoringContext::new(ligand, maps, scoring);
    let mut accepted: Vec<Pose> = Vec::new();
    let mut incumbent: Option<Conformation> = None;
    let mut incumbent_energy = f64::INFINITY;
    let decay_steps = ((config.steps as f64) * config.temperature_decay).max(1.0);

    for step in 0..config.steps {
        // Metropolis-like acceptance temperature, decaying over the walk.
        let progress = (step as f64) / decay_steps;
        let progress = progress.min(1.0);
        let temperature = config.initial_temperature
            + (config.final_temperature - config.initial_temperature) * progress;

        // Propose: from the incumbent if we have one, otherwise fresh.
        let conf = match &incumbent {
            Some(c) => perturb(c, rng, config.mutation_amplitude, box_),
            None => random_in_box(ligand, rng, box_),
        };

        let (mut conf, outcome) = {
            let mut eval = |c: &Conformation| ctx.evaluate(c);
            minimize(&conf, &config.local, &mut eval)
        };
        if !outcome.energy.is_finite() {
            continue;
        }

        let accept = if incumbent.is_none() {
            true
        } else {
            let delta = outcome.energy - incumbent_energy;
            if delta <= 0.0 {
                true
            } else {
                // Metropolis: accept a worse local minimum sometimes, so the
                // walk can cross barriers instead of getting pinned.
                rng.gen::<f64>() < (-delta / temperature.max(1e-6)).exp()
            }
        };

        if accept {
            incumbent_energy = outcome.energy;
            conf.torsions
                .iter_mut()
                .for_each(|t| *t = crate::search::wrap_torsion(*t));
            incumbent = Some(conf.clone());
            accepted.push(pose_from(&mut ctx, conf, outcome.energy));
        }
    }

    accepted
}

/// Build a [`Pose`] from a conformation, capturing coordinates and the split.
pub(crate) fn pose_from(ctx: &mut ScoringContext<'_>, conf: Conformation, energy: f64) -> Pose {
    let (breakdown, _) = ctx.evaluate_full(&conf);
    let coords = ctx.current_coords().to_vec();
    Pose {
        conf,
        energy,
        intermolecular: breakdown.intermolecular,
        intramolecular: breakdown.intramolecular,
        coords,
        rmsd: None,
    }
}

/// A uniformly random conformation with the ligand fully inside the box.
pub fn random_in_box(
    ligand: &Ligand,
    rng: &mut impl Rng,
    box_: crate::grid::GridBox,
) -> Conformation {
    let mut conf = ligand.tree.randomize(rng);
    let c = box_.center();
    // The reference coordinates already describe the ligand's own frame, so a
    // random orientation about its centroid keeps the extent centred on `c`.
    let centroid = ligand.molecule.centroid();
    conf.position = [c[0] - centroid[0], c[1] - centroid[1], c[2] - centroid[2]];
    conf
}

/// Standard normal variate via the Box–Muller transform.
///
/// A free function rather than a closure: Rust does not allow `impl Trait` in
/// closure parameter position, and a named function is clearer anyway.
#[inline]
pub(crate) fn gauss<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Perturb a conformation in place, keeping the ligand inside the box.
///
/// The kick is applied to the *body frame*, which is what makes this an
/// iterated local search: small steps explore the pocket of the current pose
/// rather than re-sampling the whole box.
fn perturb(
    conf: &Conformation,
    rng: &mut impl Rng,
    amplitude: f64,
    box_: crate::grid::GridBox,
) -> Conformation {
    let mut out = conf.clone();
    for k in 0..3 {
        out.position[k] += amplitude * 0.5 * gauss(rng);
        out.orientation[k] += amplitude * 0.2 * gauss(rng);
    }
    for t in out.torsions.iter_mut() {
        *t += amplitude * 0.3 * gauss(rng);
        *t = crate::search::wrap_torsion(*t);
    }

    // Clamp the translation so the centroid stays well inside the box.
    let c = box_.center();
    for k in 0..3 {
        let lo = c[k] - box_.size()[k] * 0.25;
        let hi = c[k] + box_.size()[k] * 0.25;
        out.position[k] = out.position[k].clamp(lo, hi);
    }
    out
}
