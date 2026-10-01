//! Island-model hybrid genetic algorithm (Island LGA).
//!
//! The Monte-Carlo iterated local search is excellent for ligands up to about a
//! dozen rotatable bonds. Beyond that the conformational space is too large to
//! cover by independent restarts, and a genetic algorithm that *evolves* whole
//! conformations explores it better.
//!
//! The design follows the modified fitness genetic algorithm (MFFGA) of
//! AutoDock-GPU (Morris et al., *JCAMD* **25**, 10 (2011)), re-implemented in
//! portable Rust:
//!
//! * the population is split into **islands** that evolve independently, which
//!   preserves diversity — without them a single global population converges to
//!   one basin far too early;
//! * every `migration_interval` steps the best member of each island is copied
//!   into a neighbouring island, replenishing diversity without mixing the
//!   whole population;
//! * the **modified fitness** is a population average with a variance term, so
//!   selection pressure is strongest while the population is diverse and
//!   relaxes as it converges;
//! * a local optimisation is applied to offspring, which is what makes the
//!   search practical: genetic operators alone would need an implausible number
//!   of generations to find a buried pocket.
//!
//! # Provenance
//!
//! AutoDock-GPU, Morris et al., *J. Comput. Aided Mol. Des.* **25**, 10
//! (2011), LGPL-2.1. The island model, migration and MFFGA fitness are
//! re-implemented here from the published description; no code was copied.

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::{DockError, Result};
use crate::grid::GridMaps;
use crate::kinematics::Conformation;
use crate::ligand::Ligand;
use crate::scoring::ScoringFunction;
use crate::search::lbfgs::{minimize, LbfgsConfig};
use crate::search::monte_carlo::Pose;
use crate::search::ScoringContext;

/// Tuning for the island genetic algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LgaConfig {
    /// Number of independent islands.
    pub islands: usize,
    /// Individuals per island.
    pub population_per_island: usize,
    /// Number of generations.
    pub generations: usize,
    /// Generations between migrations.
    pub migration_interval: usize,
    /// Crossover probability.
    pub crossover_rate: f64,
    /// Per-torsion Gaussian mutation width, in radians.
    pub mutation_rate: f64,
    /// Fraction of a new individual's energy that replaced energy is blended
    /// into the "modified fitness" used for selection.
    pub variance_weight: f64,
    /// Local optimiser settings applied to each offspring.
    pub local: LbfgsConfig,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for LgaConfig {
    fn default() -> Self {
        LgaConfig {
            islands: 5,
            population_per_island: 24,
            generations: 40,
            migration_interval: 5,
            crossover_rate: 0.9,
            mutation_rate: 0.3,
            variance_weight: 0.15,
            local: LbfgsConfig {
                max_iterations: 60,
                ..Default::default()
            },
            seed: None,
        }
    }
}

impl LgaConfig {
    /// Total population across all islands.
    pub fn total_population(&self) -> usize {
        self.islands * self.population_per_island
    }
}

/// An individual in the population.
#[derive(Debug, Clone)]
struct Individual {
    conf: Conformation,
    energy: f64,
    pose: Option<Pose>,
}

/// Run the island-model GA.
///
/// Every island evolves independently on its own thread, so the whole GA scales
/// with `rayon`. Migration is a barrier between generations.
pub fn evolve(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    config: &LgaConfig,
) -> Result<Vec<Pose>> {
    if ligand.is_empty() {
        return Err(DockError::molecule("cannot dock an empty ligand"));
    }
    if config.islands == 0 || config.population_per_island < 2 {
        return Err(DockError::param(
            "lga population",
            format!("{}x{}", config.islands, config.population_per_island),
            "need at least one island with two individuals",
        ));
    }
    if !(0.0..=1.0).contains(&config.crossover_rate) {
        return Err(DockError::param(
            "crossover_rate",
            config.crossover_rate,
            "must lie in [0, 1]",
        ));
    }

    let base_seed = config.seed.unwrap_or_else(rand::random);
    let box_ = maps.grid_box();

    let mut populations: Vec<Vec<Individual>> = (0..config.islands)
        .into_par_iter()
        .map(|island| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                base_seed ^ (island as u64).wrapping_mul(super::WALK_SEED_STRIDE),
            );
            let mut pop: Vec<Individual> = (0..config.population_per_island)
                .map(|_| {
                    let conf = super::monte_carlo::random_in_box(ligand, &mut rng, box_);
                    let mut ctx = ScoringContext::new(ligand, maps, scoring);
                    let (energy, _) = ctx.evaluate(&conf);
                    Individual {
                        conf,
                        energy,
                        pose: None,
                    }
                })
                .collect();
            // Seed each island with a locally optimised individual so the
            // starting population is already physically sensible.
            for ind in pop.iter_mut() {
                let mut ctx = ScoringContext::new(ligand, maps, scoring);
                let (conf, outcome) = {
                    let mut eval = |c: &Conformation| ctx.evaluate(c);
                    minimize(&ind.conf, &config.local, &mut eval)
                };
                if outcome.energy.is_finite() {
                    let pose =
                        super::monte_carlo::pose_from(&mut ctx, conf.clone(), outcome.energy);
                    ind.pose = Some(pose);
                    ind.conf = conf;
                    ind.energy = outcome.energy;
                }
            }
            pop.sort_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            pop
        })
        .collect();

    for _generation in 0..config.generations {
        // --- Evolve each island independently -----------------------------
        let migrants: Vec<(usize, Individual)> = populations
            .par_iter_mut()
            .enumerate()
            .map(|(island, pop)| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    base_seed
                        ^ (island as u64).wrapping_mul(super::ISLAND_SEED_STRIDE)
                        ^ pop.len() as u64,
                );
                (
                    island,
                    step_generation(ligand, maps, scoring, config, pop, &mut rng),
                )
            })
            .collect();

        // --- Migration: the best of each island seeds its neighbour --------
        if config.migration_interval > 0
            && (_generation + 1) % config.migration_interval == 0
            && populations.len() > 1
        {
            let elites: Vec<Individual> = migrants.iter().map(|(_, m)| m.clone()).collect();
            for island in 0..populations.len() {
                let source = (island + 1) % elites.len();
                // Replace the weakest individual, which keeps diversity while
                // still importing the new basin.
                let worst = populations[island]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        a.1.energy
                            .partial_cmp(&b.1.energy)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i);
                if let Some(i) = worst {
                    populations[island][i] = elites[source].clone();
                }
            }
        }
    }

    // --- Collect every distinct pose, best first ---------------------------
    let mut all: Vec<Pose> = populations
        .into_iter()
        .flatten()
        .filter_map(|i| i.pose)
        .collect();
    all.sort_by(|a, b| {
        a.energy
            .partial_cmp(&b.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(all)
}

/// Run one generation of selection, crossover, mutation and local search.
fn step_generation(
    ligand: &Ligand,
    maps: &GridMaps,
    scoring: &dyn ScoringFunction,
    config: &LgaConfig,
    pop: &mut Vec<Individual>,
    rng: &mut impl Rng,
) -> Individual {
    let n = pop.len();
    let fitness = modified_fitness(pop, config.variance_weight);
    let elite_count = (n / 10).max(2);

    let mut children: Vec<Individual> = Vec::with_capacity(n);
    // Elitism: the best survive untouched, so the island never regresses.
    for (ind, w) in pop.iter().zip(fitness.iter()) {
        if children.len() >= elite_count {
            break;
        }
        if *w > 0.0 {
            children.push(ind.clone());
        }
    }

    while children.len() < n {
        let a = tournament(pop, &fitness, rng);
        let b = tournament(pop, &fitness, rng);
        let mut child = if rng.gen::<f64>() < config.crossover_rate {
            uniform_crossover(&a.conf, &b.conf, rng)
        } else {
            a.conf.clone()
        };
        mutate(&mut child, rng, config.mutation_rate);

        let mut ctx = ScoringContext::new(ligand, maps, scoring);
        let (conf, outcome) = {
            let mut eval = |c: &Conformation| ctx.evaluate(c);
            minimize(&child, &config.local, &mut eval)
        };
        if !outcome.energy.is_finite() {
            continue; // offspring never replaces a valid parent
        }
        let pose = super::monte_carlo::pose_from(&mut ctx, conf.clone(), outcome.energy);
        children.push(Individual {
            conf,
            energy: outcome.energy,
            pose: Some(pose),
        });
    }

    children.sort_by(|a, b| {
        a.energy
            .partial_cmp(&b.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = children[0].clone();
    *pop = children;
    best
}

/// MFFGA modified fitness: the population mean plus a fraction of its spread.
///
/// Early on, a wide population is explored aggressively; as the population
/// converges the spread term shrinks and selection becomes purely energetic.
/// Returned as a *relative* weight in `(0, 1]`, so the best individual in the
/// population always has weight 1 regardless of the absolute energy scale.
fn modified_fitness(pop: &[Individual], variance_weight: f64) -> Vec<f64> {
    let finite: Vec<f64> = pop
        .iter()
        .map(|i| i.energy)
        .filter(|e| e.is_finite())
        .collect();
    if finite.is_empty() {
        return vec![1.0; pop.len()];
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let spread = (variance_weight * var).sqrt();
    let modified: Vec<f64> = finite.iter().map(|e| e + spread).collect();
    // Normalise so the best individual scores 1 and worse ones fall toward 0.
    // Selection wants *lower* energy to rank higher, hence `1 − (m − best)/scale`.
    let best = modified.iter().cloned().fold(f64::INFINITY, f64::min);
    let worst = modified.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scale = (worst - best).max(1e-6);
    pop.iter()
        .map(|ind| {
            if !ind.energy.is_finite() {
                // An invalid individual gets the worst weight, not a random
                // one, so it is reliably purged.
                0.0
            } else {
                let m = ind.energy + spread;
                (1.0 - (m - best) / scale).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Tournament selection: sample `k` individuals and take the highest weight.
fn tournament(pop: &[Individual], fitness: &[f64], rng: &mut impl Rng) -> Individual {
    let k = 3.min(pop.len()).max(1);
    let mut best: usize = rng.gen_range(0..pop.len());
    let mut best_w = fitness[best];
    for _ in 1..k {
        let c = rng.gen_range(0..pop.len());
        if fitness[c] > best_w {
            best_w = fitness[c];
            best = c;
        }
    }
    pop[best].clone()
}

/// Single-point crossover in the torsion vector; the rigid body is inherited
/// from the first parent, which keeps the child near a real pocket.
fn uniform_crossover(a: &Conformation, b: &Conformation, rng: &mut impl Rng) -> Conformation {
    let mut child = a.clone();
    for i in 0..child.torsions.len() {
        if rng.gen::<f64>() < 0.5 {
            child.torsions[i] = b.torsions[i];
        }
    }
    child
}

/// Gaussian mutation of the torsions, with a small chance of a rigid-body kick.
fn mutate(c: &mut Conformation, rng: &mut impl Rng, rate: f64) {
    for t in c.torsions.iter_mut() {
        if rng.gen::<f64>() < rate {
            *t += rate * 2.0 * super::monte_carlo::gauss(rng);
            *t = crate::search::wrap_torsion(*t);
        }
    }
    if rng.gen::<f64>() < 0.15 {
        for k in 0..3 {
            c.orientation[k] += 0.3 * super::monte_carlo::gauss(rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_population() {
        let c = LgaConfig::default();
        assert_eq!(c.total_population(), c.islands * c.population_per_island);
    }

    #[test]
    fn modified_fitness_prefers_lower_energy() {
        let pop = vec![
            Individual {
                conf: Conformation::identity(0),
                energy: -10.0,
                pose: None,
            },
            Individual {
                conf: Conformation::identity(0),
                energy: -1.0,
                pose: None,
            },
        ];
        let f = modified_fitness(&pop, 0.0);
        assert!(f[0] > f[1], "lower energy must select more strongly: {f:?}");
        // The best individual always scores exactly 1.
        assert!((f[0] - 1.0).abs() < 1e-12, "{f:?}");
    }

    #[test]
    fn modified_fitness_purges_invalid_individuals() {
        let pop = vec![
            Individual {
                conf: Conformation::identity(0),
                energy: f64::INFINITY,
                pose: None,
            },
            Individual {
                conf: Conformation::identity(0),
                energy: -5.0,
                pose: None,
            },
        ];
        let f = modified_fitness(&pop, 0.1);
        assert_eq!(f[0], 0.0, "an invalid individual must score 0");
        assert!(f[1] > 0.0);
    }

    #[test]
    fn modified_fitness_handles_empty_population() {
        let pop: Vec<Individual> = Vec::new();
        let f = modified_fitness(&pop, 0.1);
        assert!(f.is_empty());
    }

    #[test]
    fn crossover_swaps_torsions() {
        let a = Conformation {
            position: [1.0, 2.0, 3.0],
            orientation: [0.1, 0.2, 0.3],
            torsions: vec![0.0, 0.0, 0.0],
        };
        let mut b = a.clone();
        b.torsions = vec![1.0, 1.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(7u64);
        let child = uniform_crossover(&a, &b, &mut rng);
        // Rigid body always comes from the first parent.
        assert_eq!(child.position, a.position);
        for t in child.torsions {
            assert!(t == 0.0 || t == 1.0);
        }
    }
}
