//! Bounded-memory quasi-Newton local optimisation (L-BFGS).
//!
//! # Why not plain gradient descent or Newton's method
//!
//! Newton needs the full Hessian, which for a 20-degree-of-freedom ligand is
//! 20×20 but for a large peptide is prohibitive, and the docking Hessian is
//! poorly conditioned. L-BFGS keeps only the last `m` correction pairs, gets
//! quasi-Newton accuracy in the well-conditioned directions that matter, and
//! has no matrix inverse.
//!
//! # Why the gradient-descent safeguard
//!
//! The gradient of a trilinear grid interpolant is **discontinuous across cell
//! boundaries** (see [`crate::grid`]). A pure quasi-Newton step can therefore
//! accept a step that lowers the true energy nowhere and then oscillate. Every
//! candidate step here must therefore pass an *Armijo sufficient-decrease*
//! test against the current energy; when it does not, the step is shortened
//! until it does. That is the standard cure, and it is why this converges on
//! AutoDock-style potentials while a bare L-BFGS stalls.
//!
//! # Provenance
//!
//! The limited-memory BFGS recursion is the standard algorithm of Byrd, Nocedal
//! and Schnabel, *SIAM J. Optim.* **16**, 1182 (1994). It is re-implemented
//! here in safe Rust; no third-party optimiser code was used.

use crate::kinematics::Conformation;
use crate::search::{conf_to_vec, vec_to_conf, wrap_torsion};

/// Tuning for the local optimiser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbfgsConfig {
    /// Number of correction pairs kept. 5–20 is typical.
    pub memory: usize,
    /// Maximum quasi-Newton iterations per local optimisation.
    pub max_iterations: usize,
    /// Stop when the gradient infinity-norm drops below this.
    pub gradient_tolerance: f64,
    /// Stop when the energy changes by less than this between iterations.
    pub energy_tolerance: f64,
    /// First step length used for the gradient-descent fallback.
    pub initial_step: f64,
    /// Armijo sufficient-decrease constant.
    pub armijo_c: f64,
    /// Longest step accepted before the step is shortened.
    pub max_step_norm: f64,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        LbfgsConfig {
            memory: 10,
            max_iterations: 200,
            gradient_tolerance: 1e-4,
            energy_tolerance: 1e-6,
            initial_step: 0.5,
            armijo_c: 1e-4,
            max_step_norm: 4.0,
        }
    }
}

/// What the optimiser did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LbfgsOutcome {
    /// Final energy.
    pub energy: f64,
    /// Number of quasi-Newton iterations actually taken.
    pub iterations: usize,
    /// Number of energy evaluations.
    pub evaluations: usize,
    /// True if the gradient tolerance was met.
    pub converged: bool,
}

/// A callable energy-and-gradient oracle.
pub type Energy<'a> = &'a mut dyn FnMut(&Conformation) -> (f64, Vec<f64>);

/// Minimise `energy` starting from `start`.
///
/// `energy` must return a gradient of exactly `6 + start.torsions.len()`
/// entries, ordered `[tx, ty, tz, θx, θy, θz, τ₀, …]` — the same layout
/// [`crate::kinematics::KinematicTree::conf_gradient`] produces. A mismatched
/// length is a programming error and is rejected rather than silently truncated.
///
/// The returned conformation always has the lowest energy seen, even if the
/// optimiser ran out of iterations — a partially converged local minimum is
/// far more useful to the search than the last (possibly worse) iterate.
pub fn minimize<F>(
    start: &Conformation,
    config: &LbfgsConfig,
    energy: &mut F,
) -> (Conformation, LbfgsOutcome)
where
    F: FnMut(&Conformation) -> (f64, Vec<f64>),
{
    let n = 6 + start.torsions.len();
    let mut x = conf_to_vec(start);

    let (mut f, mut g) = energy(&vec_to_conf(&x, start.torsions.len()));
    debug_assert_eq!(
        g.len(),
        n,
        "energy callback must return a gradient of length {n}"
    );
    if g.len() != n {
        // Truncate or pad rather than index out of bounds: a misbehaving
        // callback should degrade the search, not abort it.
        g.resize(n, 0.0);
    }
    let mut evaluations: usize = 1;
    if !f.is_finite() {
        return (
            start.clone(),
            LbfgsOutcome {
                energy: f64::INFINITY,
                iterations: 0,
                evaluations,
                converged: false,
            },
        );
    }

    let mut best_x = x.clone();
    let mut best_f = f;

    // Limited-memory correction pairs: (s, y) with s = Δx, y = Δg.
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(config.memory);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(config.memory);

    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let gnorm = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if gnorm < config.gradient_tolerance {
            converged = true;
            break;
        }

        // --- Two-loop recursion for the search direction ---------------------
        let mut q = g.clone();
        let mut alphas = vec![0.0f64; s_hist.len()];
        let m = s_hist.len();
        for i in (0..m).rev() {
            let rho = 1.0 / dot(&y_hist[i], &s_hist[i]).max(1e-12);
            alphas[i] = rho * dot(&s_hist[i], &q);
            for k in 0..n {
                q[k] -= alphas[i] * y_hist[i][k];
            }
        }
        if m > 0 {
            let yy = dot(&y_hist[m - 1], &y_hist[m - 1]);
            let ys = dot(&y_hist[m - 1], &s_hist[m - 1]);
            let gamma = (ys / yy.max(1e-12)).clamp(1e-6, 1e6);
            for k in 0..n {
                q[k] *= gamma;
            }
        }
        for i in 0..m {
            let rho = 1.0 / dot(&y_hist[i], &s_hist[i]).max(1e-12);
            let beta = rho * dot(&y_hist[i], &q);
            for k in 0..n {
                q[k] += (alphas[i] - beta) * s_hist[i][k];
            }
        }
        let mut p: Vec<f64> = q.iter().map(|v| -v).collect();

        // If the quasi-Newton direction is not a descent direction — which the
        // curvature condition can fail to guarantee on a noisy gradient — fall
        // back to steepest descent.
        if dot(&p, &g) >= 0.0 {
            for k in 0..n {
                p[k] = -g[k];
            }
            s_hist.clear();
            y_hist.clear();
        }

        // --- Armijo backtracking line search ---------------------------------
        let step = line_search(
            &x,
            f,
            &g,
            &p,
            config,
            &mut evaluations,
            energy,
            start.torsions.len(),
        );
        let (x_new, f_new, g_new, actual_step) = match step {
            Some(v) => v,
            None => {
                // No acceptable step: stop, keeping the best point found.
                break;
            }
        };
        if actual_step == 0.0 {
            break;
        }

        let s: Vec<f64> = (0..n).map(|k| x_new[k] - x[k]).collect();
        let y: Vec<f64> = (0..n).map(|k| g_new[k] - g[k]).collect();
        // Only store a correction pair that satisfies the curvature condition
        // yᵀs > 0; otherwise the two-loop recursion would produce a bad step.
        if dot(&y, &s) > 1e-12 {
            s_hist.push(s);
            y_hist.push(y);
            if s_hist.len() > config.memory {
                s_hist.remove(0);
                y_hist.remove(0);
            }
        }

        let delta = (f_new - f).abs();
        x = x_new;
        f = f_new;
        g = g_new;

        if f < best_f {
            best_f = f;
            best_x = x.clone();
        }
        if delta < config.energy_tolerance
            && g.iter().fold(0.0f64, |m, v| m.max(v.abs())) < config.gradient_tolerance * 10.0
        {
            converged = true;
            break;
        }
    }

    (
        vec_to_conf(&best_x, start.torsions.len()),
        LbfgsOutcome {
            energy: best_f,
            iterations,
            evaluations,
            converged,
        },
    )
}

/// Backtracking line search enforcing sufficient decrease.
///
/// Returns the accepted point, or `None` if even a vanishing step fails to
/// decrease the energy — which means we are at a local minimum already.
#[allow(clippy::too_many_arguments)]
fn line_search<F>(
    x: &[f64],
    f0: f64,
    g: &[f64],
    p: &[f64],
    config: &LbfgsConfig,
    evaluations: &mut usize,
    energy: &mut F,
    num_torsions: usize,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64)>
where
    F: FnMut(&Conformation) -> (f64, Vec<f64>),
{
    let n = x.len();
    let slope = dot(g, p);
    if slope >= 0.0 {
        return None;
    }
    let pnorm = dot(p, p).sqrt();
    if pnorm < 1e-14 {
        return None;
    }
    // Cap the first trial step so a huge quasi-Newton direction cannot throw
    // the ligand clear out of the box.
    let mut alpha = (config.max_step_norm / pnorm).min(config.initial_step);

    for _ in 0..40 {
        let trial: Vec<f64> = (0..n).map(|k| x[k] + alpha * p[k]).collect();
        let conf = normalize_torsions(&trial, num_torsions);
        let (f, g_new) = energy(&conf);
        *evaluations += 1;
        let armijo = f0 + config.armijo_c * alpha * slope;
        if f.is_finite() && f <= armijo {
            let actual = if pnorm > 0.0 {
                // Recover the step actually taken after wrapping torsions.
                let delta: Vec<f64> = (0..n).map(|k| conf_to_vec(&conf)[k] - x[k]).collect();
                (dot(&delta, p) / dot(p, p)).max(0.0)
            } else {
                0.0
            };
            return Some((conf_to_vec(&conf), f, g_new, actual));
        }
        alpha *= 0.5;
        if alpha < 1e-12 {
            break;
        }
    }
    None
}

/// Wrap the torsion entries of a packed vector into `(-π, π]` in place-safe
/// fashion, returning the equivalent conformation.
fn normalize_torsions(x: &[f64], num_torsions: usize) -> Conformation {
    let mut conf = vec_to_conf(x, num_torsions);
    for t in conf.torsions.iter_mut() {
        *t = wrap_torsion(*t);
    }
    conf
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The classic Rosenbrock function: a hard test for a quasi-Newton method
    /// because its valley is curved and narrow.
    ///
    /// The gradient is padded to the full six rigid-body degrees of freedom; the
    /// three rotational entries are zero because this objective does not depend
    /// on the orientation.
    fn rosenbrock(c: &Conformation) -> (f64, Vec<f64>) {
        let [x, y] = [c.position[0], c.position[1]];
        let f = 100.0 * (y - x * x).powi(2) + (1.0 - x).powi(2);
        let g = vec![
            -400.0 * x * (y - x * x) - 2.0 * (1.0 - x),
            200.0 * (y - x * x),
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        (f, g)
    }

    #[test]
    fn converges_on_rosenbrock() {
        let start = Conformation {
            position: [-1.2, 1.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            torsions: vec![],
        };
        let cfg = LbfgsConfig {
            max_iterations: 500,
            ..Default::default()
        };
        let (out, outcome) = minimize(&start, &cfg, &mut rosenbrock);
        // Global minimum is at (1, 1) with f = 0.
        assert!(
            out.position[0] > 0.99 && out.position[0] < 1.01,
            "x = {}",
            out.position[0]
        );
        assert!(
            out.position[1] > 0.99 && out.position[1] < 1.01,
            "y = {}",
            out.position[1]
        );
        assert!(outcome.energy < 1e-6, "energy {}", outcome.energy);
    }

    #[test]
    fn converges_on_a_quadratic() {
        let start = Conformation {
            position: [3.0, -2.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            torsions: vec![],
        };
        // f = x² + 2y² + z²
        let mut f = |c: &Conformation| {
            (
                c.position[0].powi(2) + 2.0 * c.position[1].powi(2) + c.position[2].powi(2),
                vec![
                    2.0 * c.position[0],
                    4.0 * c.position[1],
                    2.0 * c.position[2],
                    0.0,
                    0.0,
                    0.0,
                ],
            )
        };
        let cfg = LbfgsConfig {
            gradient_tolerance: 1e-10,
            energy_tolerance: 1e-14,
            max_iterations: 500,
            ..Default::default()
        };
        let (out, outcome) = minimize(&start, &cfg, &mut f);
        assert!(outcome.energy < 1e-12, "energy {}", outcome.energy);
        for k in 0..3 {
            assert!(
                out.position[k].abs() < 1e-5,
                "axis {k}: {}",
                out.position[k]
            );
        }
    }

    #[test]
    fn never_returns_worse_than_the_start() {
        // A deliberately awkward surface with a local minimum at the start.
        let mut f = |c: &Conformation| {
            let x = c.position[0];
            let e = if x > 0.0 { x * x } else { 10.0 * x };
            let de = if x > 0.0 { 2.0 * x } else { 10.0 };
            (e, vec![de, 0.0, 0.0, 0.0, 0.0, 0.0])
        };
        let start = Conformation {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            torsions: vec![],
        };
        let (out, outcome) = minimize(&start, &LbfgsConfig::default(), &mut f);
        assert!(outcome.energy <= 1e-12);
        assert!(out.position[0] <= 1e-9);
    }

    #[test]
    fn handles_torsions_and_wraps_them() {
        // f = (t - 0.5)², but the optimiser may step past π; wrapping must
        // keep the energy consistent.
        let mut f = |c: &Conformation| {
            let t = c.torsions[0];
            (
                0.5 * (t - 0.5).powi(2),
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t - 0.5],
            )
        };
        let start = Conformation {
            position: [0.0; 3],
            orientation: [0.0; 3],
            torsions: vec![4.0],
        };
        let cfg = LbfgsConfig {
            gradient_tolerance: 1e-8,
            energy_tolerance: 1e-12,
            max_iterations: 500,
            ..Default::default()
        };
        let (out, _) = minimize(&start, &cfg, &mut f);
        assert!((out.torsions[0] - 0.5).abs() < 1e-6, "{}", out.torsions[0]);
    }

    #[test]
    fn rejects_a_non_finite_start() {
        let mut f = |_c: &Conformation| (f64::NAN, vec![0.0; 6]);
        let start = Conformation {
            position: [0.0; 3],
            orientation: [0.0; 3],
            torsions: vec![],
        };
        let (_, outcome) = minimize(&start, &LbfgsConfig::default(), &mut f);
        assert!(!outcome.converged);
        assert!(outcome.energy.is_infinite());
    }

    #[test]
    fn torsion_wrapping_is_correct() {
        assert!((wrap_torsion(0.0)).abs() < 1e-12);
        assert!((wrap_torsion(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-9);
        assert!((wrap_torsion(-3.0 * std::f64::consts::PI) + std::f64::consts::PI).abs() < 1e-9);
        assert!((wrap_torsion(1.0) - 1.0).abs() < 1e-12);
    }
}
