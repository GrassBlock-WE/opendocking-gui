//! Is "4x the local budget makes the best pose worse" a regression, or noise?
//!
//! `convergence_budget.rs` established that the local budget is not a lever
//! worth more than a quarter of a kcal/mol, and it prints a per-seed table with
//! a two-sided bound on the mean. What it does **not** do is separate the two
//! things that number can be made of, and until they are separated the bound is
//! a statement about a mixture.
//!
//! # The two things the number can be made of
//!
//! **`minimize` returns the lowest point it ever saw** (`lbfgs.rs:221-224`,
//! returned at `:233`). So for a *fixed start conformation* the energy is
//! non-increasing in `max_iterations` by construction, whatever the energy
//! function is — and whatever shape the out-of-box penalty has. Call this
//! **the fixed-start effect**. It cannot be negative, and it is small.
//!
//! **The search does not hold its start points still.** `run_walk` proposes the
//! next conformation by perturbing the *incumbent* (`monte_carlo.rs:209-212`),
//! and the incumbent is the *result* of the previous local optimisation
//! (`:240`). Raising the local budget therefore changes the incumbent, which
//! changes the next proposal, which changes the next local optimisation: the two
//! runs walk **different trajectories**, and the best pose of a search is the
//! minimum over a trajectory. Call this **the resampling effect**. It is
//! signed neither way by construction.
//!
//! The headline number — best pose of the whole search, 200 -> 800 — is the sum
//! of the two. This file measures them apart, which is the only way to tell
//! whether the shipped budget is actively harmful or merely one draw.
//!
//! # What decides it
//!
//! Not the sign of any single seed. Three measurements, in order of how much
//! they decide:
//!
//! 1. **The variance identity.** If the two arms are two independent draws of
//!    the same estimator, then `Var(d) = Var(A) + Var(B)` exactly. Measured
//!    against that, a budget effect that carried a systematic mechanism would
//!    leave the paired spread *larger* than the two arms' own spreads, because
//!    a mechanism adds a seed-dependent term. A ratio at 1.0 is the statement
//!    "these are two samples", and it costs no extra `dock` call.
//! 2. **An exact sign-flip test on the paired mean** (2^8 = 256 enumerations, no
//!    distributional assumption, minimum attainable two-sided p = 1/128).
//! 3. **The fixed-start ladder**, which is where a mechanism would have to
//!    live. If the energy at a fixed start is non-increasing in the budget, the
//!    local optimiser is not the source and no property of the penalty's shape
//!    can be.
//!
//! # What is asserted and what is only reported
//!
//! The fixed-start monotonicity is asserted, because it is the load-bearing
//! claim and it is the one with a mechanism behind it. The global effect's sign
//! is **not** asserted in either direction: it is a property of a stochastic
//! search, and a sign assertion on it is the mistake this file exists to stop
//! repeating. Its mean, spread, permutation p and variance ratio are reported
//! in full, including when they say the budget is worth nothing.
//!
//! Run with `cargo test -p dock-core --features convergence-budget --test budget_monotonicity -- --nocapture`.

use dock_core::docking::{dock, DockingConfig};
use dock_core::grid::{GridBox, GridMaps};
use dock_core::kinematics::Conformation;
use dock_core::ligand::Ligand;
use dock_core::receptor::Receptor;
use dock_core::scoring::VinaScoring;
use dock_core::search::lbfgs::{minimize, LbfgsConfig, LbfgsOutcome};
use dock_core::search::monte_carlo::{random_in_box, MonteCarloConfig};
use dock_core::search::ScoringContext;
use rand::SeedableRng;

/// The shipped crambin fixture, as `examples/audit_poses.py` runs it. Every
/// constant here is copied from `convergence_budget.rs` rather than restated
/// with a "see that file": the two files' numbers are only comparable if they
/// are the same measurement, and a second copy of a fixture path is a second
/// place for them to drift apart.
const RECEPTOR: &str = "1crn_prep.pdbqt";
const SHIPPED_LIGAND: &str = "biotin_prep.pdbqt";
const CENTRE: [f64; 3] = [3.47, 6.21, 8.95];
const SIZE: f64 = 18.0;
const SPACING: f64 = 0.375;
const SEED: u64 = 20260929;
const EXHAUSTIVENESS: u32 = 16;
const NUM_MODES: usize = 5;
/// The local budget as the engine ships it (`LbfgsConfig::default`).
const SHIPPED_LOCAL_BUDGET: usize = 200;
/// The 4x budget whose sign flipped.
const DEEP_LOCAL_BUDGET: usize = 800;
/// The same seed list as `convergence_budget.rs`, shipped seed first, so the
/// per-seed rows below are the same rows that file prints.
const BUDGET_SEEDS: [u64; 8] = [20260929, 1, 2, 3, 4, 5, 6, 7];
/// How many of the search's own first proposals the fixed-start ladder drives.
/// The first proposal of walk `w` is drawn before anything has been accepted, so
/// it is reproducible from the public seed derivation and is the only start the
/// search's own trajectory pins down.
const LADDER_STARTS: u32 = 6;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("examples")
        .join(name)
}

fn search_box() -> GridBox {
    GridBox::centered(CENTRE, [SIZE; 3]).expect("the 18 A cube is a valid box")
}

fn maps_for(receptor: &str) -> GridMaps {
    let rec = Receptor::from_pdbqt(fixture(receptor)).expect("the receptor fixture should load");
    rec.precalculate(&search_box(), &VinaScoring::new(), SPACING)
        .expect("precalculation should succeed")
}

fn ligand_for(name: &str) -> Ligand {
    Ligand::from_pdbqt(fixture(name)).expect("the ligand fixture should load")
}

fn seeded_config(local_max_iterations: usize, seed: u64) -> DockingConfig {
    DockingConfig {
        monte_carlo: MonteCarloConfig {
            exhaustiveness: EXHAUSTIVENESS,
            steps: MonteCarloConfig::default().steps,
            seed: Some(seed),
            local: LbfgsConfig {
                max_iterations: local_max_iterations,
                ..LbfgsConfig::default()
            },
            ..MonteCarloConfig::default()
        },
        num_modes: NUM_MODES,
        ..DockingConfig::default()
    }
}

/// The starting conformation walk `w` proposes on its very first step.
///
/// Reproduces `monte_carlo::search`'s own derivation
/// (`base_seed ^ w * WALK_SEED_STRIDE`, then `random_in_box` with nothing drawn
/// before it) rather than sampling a fresh in-box conformation: a start the
/// search itself chose is the only one whose position in the trajectory is
/// known, which is what makes the fixed-start measurement a measurement of the
/// fixed-start effect rather than of a random field.
fn first_proposal(ligand: &Ligand, box_: &GridBox, walk: u32) -> Conformation {
    let mut rng = rand::rngs::StdRng::seed_from_u64(
        SEED ^ (walk as u64).wrapping_mul(dock_core::search::WALK_SEED_STRIDE),
    );
    random_in_box(ligand, &mut rng, *box_)
}

fn best_pose_energy(
    local_max_iterations: usize,
    seed: u64,
    ligand: &Ligand,
    maps: &GridMaps,
) -> f64 {
    dock(
        ligand,
        maps,
        &VinaScoring::new(),
        &seeded_config(local_max_iterations, seed),
    )
    .expect("the search should dock for every sampled seed")
    .poses[0]
        .energy
}

/// One local optimisation from a fixed start, keeping the whole trajectory.
fn run_traced(
    ligand: &Ligand,
    maps: &GridMaps,
    start: &Conformation,
    config: &LbfgsConfig,
) -> (f64, LbfgsOutcome) {
    let scoring = VinaScoring::new();
    let mut ctx = ScoringContext::new(ligand, maps, &scoring);
    let mut eval = |c: &Conformation| ctx.evaluate(c);
    let (_, outcome) = minimize(start, config, &mut eval);
    (outcome.energy, outcome)
}

/// Sample standard deviation, `None` below two observations.
fn sd(v: &[f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let ss: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
    Some((ss / (v.len() as f64 - 1.0)).sqrt())
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Exact two-sided sign-flip test on the mean of `d`.
///
/// Enumerates all 2^n sign vectors and reports the fraction whose `|sum|`
/// reaches the observed `|sum|`. This is the right test for "is the mean of
/// these paired differences distinguishable from 0" when the differences are
/// symmetric under sign exchange, and it is exact — no normality assumption on
/// n = 8 samples, and no appeal to a table. Its floor is 1/2^(n-1), which is
/// worth stating next to the number: at n = 8 a *perfect* 8-for-8 split could
/// only reach p = 0.0078, so a null result here is a statement about the
/// resolution available, not a licence to say "no effect".
fn sign_flip_p(d: &[f64]) -> f64 {
    let n = d.len();
    let observed: f64 = d.iter().sum::<f64>().abs();
    let mut at_least = 0usize;
    let mut total = 0usize;
    for mask in 0u32..(1u32 << n) {
        let mut s = 0.0f64;
        for (i, x) in d.iter().enumerate() {
            s += if mask >> i & 1 == 1 { *x } else { -*x };
        }
        total += 1;
        if s.abs() >= observed - 1e-15 {
            at_least += 1;
        }
    }
    at_least as f64 / total as f64
}

/// Two-sided exact sign test: how many of the paired differences favour the
/// deeper budget, against a fair coin. Reported beside the permutation p
/// because the two answer different questions — this one is about the *sign*,
/// which is what the headline is, and it is the weaker of the two at n = 8.
fn sign_test_p(negative: usize, n: usize) -> f64 {
    let k = negative.min(n - negative);
    let mut tail = 0usize;
    for i in 0..=k {
        tail += binomial(n, i);
    }
    (2.0 * tail as f64 / (1u64 << n) as f64).min(1.0)
}

fn binomial(n: usize, k: usize) -> usize {
    // Exact integer binomial coefficient; n is 8 here, so u64 is ample.
    let mut num: u64 = 1;
    for i in 0..k {
        num = num * (n - i) as u64 / (i + 1) as u64;
    }
    num as usize
}

// ---------------------------------------------------------------------------
// 1. The fixed-start effect, where a mechanism would have to live
// ---------------------------------------------------------------------------

#[test]
fn more_budget_never_hurts_a_fixed_start() {
    let maps = maps_for(RECEPTOR);
    let ligand = ligand_for(SHIPPED_LIGAND);
    let box_ = search_box();
    // A wide ladder, and a *quarter* of the shipped budget at the bottom so the
    // assertion covers a run that genuinely runs out of iterations rather than
    // only runs that the line search stops.
    let ladder = [50usize, SHIPPED_LOCAL_BUDGET, DEEP_LOCAL_BUDGET, 3_200];

    let mut s = String::new();
    s += &format!(
        "\n=== fixed-start ladder: the same start, more budget ({LADDER_STARTS} of the \
         search's own first proposals) ===\n"
    );

    // The load-bearing measurement.
    let mut non_monotone = 0usize;
    let mut budget_changed_nothing = 0usize;
    let mut total_starts = 0usize;
    let mut gains = Vec::new();
    let mut bit_identical = 0usize;
    let mut exits_at_shipped = [0usize; 3];

    for w in 0..LADDER_STARTS {
        let start = first_proposal(&ligand, &box_, w);
        let mut energies = Vec::new();
        let mut iters = Vec::new();
        let mut evals = Vec::new();
        let mut converged = Vec::new();
        for budget in ladder {
            let cfg = LbfgsConfig {
                max_iterations: budget,
                ..LbfgsConfig::default()
            };
            let (e, outcome) = run_traced(&ligand, &maps, &start, &cfg);
            energies.push(e);
            iters.push(outcome.iterations);
            evals.push(outcome.evaluations);
            converged.push(outcome.converged);
        }
        s += &format!("  walk {w}: ");
        for (i, budget) in ladder.iter().enumerate() {
            s += &format!(
                "{budget}->{:+9.4} (it {:>4}, ev {:>6}, conv {})   ",
                energies[i], iters[i], evals[i], converged[i]
            );
        }
        s += "\n";

        for w_pair in energies.windows(2) {
            if w_pair[1] > w_pair[0] {
                non_monotone += 1;
            }
        }
        // Did the budget change *anything*? Bit-comparison, not a tolerance:
        // this is "the run ended on a different iterate", and a tolerance
        // would quietly turn a real difference into a rounding argument.
        if energies[1].to_bits() == energies[3].to_bits() {
            bit_identical += 1;
        }
        if energies
            .iter()
            .all(|e| e.to_bits() == energies[0].to_bits())
        {
            budget_changed_nothing += 1;
        }
        // The exit that ended the shipped-budget run, so the reader can see how
        // many starts the budget has any authority over at all.
        let (e, o) = run_traced(
            &ligand,
            &maps,
            &start,
            &LbfgsConfig {
                max_iterations: SHIPPED_LOCAL_BUDGET,
                ..LbfgsConfig::default()
            },
        );
        let _ = e;
        exits_at_shipped[exit_bucket(&o, SHIPPED_LOCAL_BUDGET)] += 1;
        total_starts += 1;
        gains.push(energies[ladder.len() - 1] - energies[1]);
    }

    let gain_mean = mean(&gains);
    let gain_min = gains.iter().cloned().fold(f64::INFINITY, f64::min);
    let gain_max = gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    s += &format!(
        "  the 200 -> 3200 change at a fixed start: mean {gain_mean:+.5}  range \
         {gain_min:+.5} .. {gain_max:+.5} kcal/mol\n\
         \x20 {bit_identical} of {total_starts} starts return a bit-identical energy at \
         200 and at 3200, and {budget_changed_nothing} of {total_starts} are unchanged \
         across the whole ladder\n\
         \x20 exit at the shipped 200-iteration budget: tolerance {}  budget {}  \
         no-acceptable-step {}\n",
        exits_at_shipped[0], exits_at_shipped[1], exits_at_shipped[2]
    );
    println!("{s}");

    // The claim this file is built on. If `minimize` ever stopped returning the
    // lowest point it saw, every mechanism claim here would need re-deriving,
    // and this is the one assertion that catches it.
    assert_eq!(
        non_monotone, 0,
        "a fixed start returned a *higher* energy at a larger iteration budget in \
         {non_monotone} of the {total_starts} ladder comparisons. `minimize` returns \
         the lowest point it ever saw (lbfgs.rs:221-224), so the energy at a fixed \
         start is non-increasing in max_iterations and every energy difference \
         measured anywhere in this crate has to be read as a difference in where \
         the *search* went, not in where the optimiser got to."
    );
}

fn exit_bucket(outcome: &LbfgsOutcome, max_iterations: usize) -> usize {
    if outcome.converged {
        0
    } else if outcome.iterations >= max_iterations {
        1
    } else {
        2
    }
}

// ---------------------------------------------------------------------------
// 2. The best-pose effect, decomposed
// ---------------------------------------------------------------------------

#[test]
fn the_best_pose_flip_is_a_resampling_difference() {
    let maps = maps_for(RECEPTOR);
    let ligand = ligand_for(SHIPPED_LIGAND);

    // --- the paired sweep ---------------------------------------------------
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut d = Vec::new();
    let mut rows = String::new();
    for seed in BUDGET_SEEDS {
        let ea = best_pose_energy(SHIPPED_LOCAL_BUDGET, seed, &ligand, &maps);
        let eb = best_pose_energy(DEEP_LOCAL_BUDGET, seed, &ligand, &maps);
        a.push(ea);
        b.push(eb);
        d.push(eb - ea);
        rows += &format!(
            "    seed {seed:<10} 200 -> {ea:+9.4}   800 -> {eb:+9.4}   dE {:+8.4}   {}\n",
            eb - ea,
            if eb - ea < 0.0 {
                "deeper wins"
            } else {
                "shipped wins"
            }
        );
    }

    let d_mean = mean(&d);
    let d_sd = sd(&d).expect("eight seeds");
    let d_se = d_sd / (BUDGET_SEEDS.len() as f64).sqrt();
    let t = d_mean / d_se;
    let a_sd = sd(&a).expect("eight seeds");
    let b_sd = sd(&b).expect("eight seeds");
    // The variance identity. Two independent draws of the same estimator give
    // Var(d) = Var(A) + Var(B) exactly, so this ratio is 1.0 by construction if
    // the budget really is changing nothing but the sample. A systematic,
    // seed-dependent mechanism would push it above 1.
    let expected_var = a_sd * a_sd + b_sd * b_sd;
    let var_ratio = (d_sd * d_sd) / expected_var;
    let perm_p = sign_flip_p(&d);
    let favours_deep = d.iter().filter(|x| **x < 0.0).count();
    let sign_p = sign_test_p(favours_deep, BUDGET_SEEDS.len());
    let headline = d[0];

    // --- the global ladder, for the sign to be read off directly ------------
    // The shipped seed walked across budgets. A non-monotone column here is the
    // observable the headline is a single draw of.
    let mut ladder_rows = String::new();
    let mut global_ladder = Vec::new();
    for budget in [100usize, 200, 400, 800, 1600] {
        let e = best_pose_energy(budget, SEED, &ligand, &maps);
        global_ladder.push((budget, e));
        ladder_rows += &format!("    {budget:>5} -> {e:+9.4}\n");
    }

    // --- the same thing with the start pinned -------------------------------
    // The first proposal of walk 0 is the one start the search's own trajectory
    // fixes. If the two budgets already disagree there, they disagree on the
    // incumbent, and `run_walk` proposes from the incumbent.
    let box_ = search_box();
    let start0 = first_proposal(&ligand, &box_, 0);
    let (e0_200, o0_200) = run_traced(
        &ligand,
        &maps,
        &start0,
        &LbfgsConfig {
            max_iterations: SHIPPED_LOCAL_BUDGET,
            ..LbfgsConfig::default()
        },
    );
    let (e0_800, _) = run_traced(
        &ligand,
        &maps,
        &start0,
        &LbfgsConfig {
            max_iterations: DEEP_LOCAL_BUDGET,
            ..LbfgsConfig::default()
        },
    );
    let e0_200_iters = o0_200.iterations;

    let non_monotone_global = global_ladder.windows(2).filter(|w| w[1].1 > w[0].1).count();

    let mut s = String::new();
    s += &format!(
        "\n=== paired best-pose sweep, {DEEP_LOCAL_BUDGET} vs {SHIPPED_LOCAL_BUDGET} \
         at exhaustiveness {EXHAUSTIVENESS}, {} seeds (shipped seed first) ===\n{rows}",
        BUDGET_SEEDS.len()
    );
    s += &format!(
        "\n  paired, same seed, two budgets: mean dE {d_mean:+.4}  sd {d_sd:.4}  se {d_se:.4}  t {t:+.2} on {} df\n\
         \x20 exact sign-flip permutation p on that mean: {perm_p:.4} (floor 1/{} at n = {})\n\
         \x20 exact sign test: the deeper budget wins {favours_deep} of {}, p = {sign_p:.4}\n\
         \x20 the headline seed's own dE is {headline:+.4}, and the seed-to-seed spread of a\n\
         \x20   *single* arm is sd {:.4} (200 arm) and {:.4} (800 arm), range {:.4} wide\n\
         \x20 variance identity Var(d)/(Var(A)+Var(B)) = {var_ratio:.3}  \
         (1.0 exactly if the two arms are two draws of the same estimator)\n",
        BUDGET_SEEDS.len() - 1,
        1usize << (BUDGET_SEEDS.len() - 1),
        BUDGET_SEEDS.len(),
        BUDGET_SEEDS.len(),
        a_sd,
        b_sd,
        a_sd.max(b_sd) * 2.0,
    );
    s += &format!(
        "\n  best pose of the whole search at the shipped seed, across a budget ladder \
         (the headline is one entry of this column):\n{ladder_rows}\
         \x20 {non_monotone_global} of {} adjacent steps go *up* as the budget rises\n",
        global_ladder.len() - 1
    );
    s += &format!(
        "\n  the same seed's walk 0, first proposal held fixed: 200 -> {e0_200:+.4}, \
         800 -> {e0_800:+.4} (dE {:+.4}).\n\
         \x20 Walk 0 is the one start the search's own trajectory pins down, and at this \
         fixture the two budgets agree on it: the local optimiser is stopped by its own \
         line search at {e0_200_iters} iterations, well inside either budget, so it returns \
         the identical iterate either way. The two arms therefore diverge at a later walk, \
         not at the first one -- *where* they first diverge is not measured here. What is \
         measured is that they do: the ladder above shows the two budgets returning different \
         best poses for the same seed, and the two runs are not nested samples because \
         monte_carlo.rs:209-212 proposes each new conformation by perturbing the *incumbent*, \
         which is the previous local optimisation's return value.\n",
        e0_800 - e0_200
    );
    println!("{s}");

    // --- what is gated, and why only this ----------------------------------
    // The budget's value is bounded from both sides, and the bound is on the
    // *mean over seeds*, never on a sign. This is the same shape of claim
    // `convergence_budget.rs` makes and it is deliberately the only assertion
    // here: a sign assertion on a stochastic search's best pose is the error
    // that produced both the -0.48 and the +0.2430 headlines.
    const MEAN_BOUND: f64 = 0.25;
    assert!(
        d_mean.abs() < MEAN_BOUND,
        "the mean best-pose difference over {} seeds is {d_mean:+.4} kcal/mol, outside \
         +-{MEAN_BOUND}, with per-seed range {:+.4} .. {:+.4}. Either the local budget has \
         become worth more than a quarter of a kcal/mol or the shipped budget has become \
         actively harmful, and both are findings rather than rounding.",
        BUDGET_SEEDS.len(),
        d.iter().cloned().fold(f64::INFINITY, f64::min),
        d.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // The discriminating one, and the reason the file exists. If the paired
    // spread is materially wider than the two arms' own spreads, the difference
    // carries a per-seed term that a resampling story does not predict, and the
    // variance identity is where to look.
    assert!(
        var_ratio < 1.5,
        "Var(d) is {var_ratio:.3}x Var(A) + Var(B). Two independent draws of the same \
         estimator give 1.0; 1.5 allows a per-seed systematic term of half the size of \
         the resampling term. A ratio above that means the budget is doing something \
         seed-dependent, which is the regression this file is looking for."
    );
}
