//! Empirical scoring functions with first-order analytic derivatives.
//!
//! # The Vina functional form
//!
//! Every pair of atoms separated by a **surface distance**
//!
//! ```text
//! d = ‖r_i − r_j‖ − (R_i + R_j)
//! ```
//!
//! — with per-element XS radii (carbon 1.9 Å, nitrogen 1.75 Å, oxygen 1.6 Å,
//! …; see [`crate::types::Element::interaction_radius`]) and `0` for polar
//! hydrogens — is scored as
//!
//! ```text
//! E(d) = c_g1·g1(d) + c_g2·g2(d) + c_rep·rep(d)
//!        + [donor–acceptor] · c_hb · hb(d)
//!        + [apolar–apolar]     · c_hyd · hyd(d)
//! ```
//!
//! with
//!
//! | term     | definition                                          | Vina weight |
//! |----------|-----------------------------------------------------|-------------|
//! | `g1`     | `exp(−((d − 0.5)/0.5)²)`                            | −0.035579   |
//! | `g2`     | `exp(−(d/0.5)²)`                                    | −0.005156   |
//! | `rep`    | `d²` for `d < 0`, else 0                            | +0.840245   |
//! | `hb`     | 1 for `d ≤ −0.5`, smoothly → 0 at `d = 0`          | −0.587439   |
//! | `hyd`    | 1 for `d ≤ 0.5`, smoothly → 0 at `d = 1.5`          | −0.035069   |
//!
//! The intramolecular contribution is the same pair sum over 1-4 (and beyond)
//! atom pairs, scaled by `0.006` — AutoDock Vina's `slope` — which stops a
//! flexible ligand from folding back onto itself.
//!
//! # Deviations from AutoDock Vina, and why
//!
//! 0. **Interaction radii.** The separation axis is set by per-element XS radii
//!    rather than a single additive constant, because the terms below are only
//!    meaningful at real contact distances: `hb` is maximal at `d ≤ −0.7`,
//!    which for an O···O pair is a 2.5 Å heavy-atom separation. The radii used
//!    are the reference per-element values listed in
//!    [`crate::types::Element::interaction_radius`]; they have **not** been
//!    cross-checked against an AutoDock Vina binary, so absolute energies here
//!    should not be compared with published Vina numbers.
//! 1. **Smooth tails.** Vina truncates `hyd` with a hard cut at `d = 1.5` and
//!    switches `hb` at exactly `d = −0.5`. Both are discontinuous, which makes
//!    the gradient jump and stalls quasi-Newton search near a plateau. Here
//!    they are C¹ cubic smoothsteps, which costs nothing and converges better.
//! 2. **Exact donor/acceptor maps.** Vina keeps a single `e_hb` map written by
//!    receptor donors *and* acceptors and read by ligand donors *and*
//!    acceptors, so a donor–donor pair picks up a spurious hydrogen bond.
//!    [`crate::grid::MapSlot`] keeps donor and acceptor contributions in
//!    separate maps, which makes the decomposition exact.
//! 3. **Exact intramolecular sum.** Vina tabulates a per-conformation ligand
//!    grid; this implementation sums the pair terms directly over a
//!    precomputed neighbour list, which removes the interpolation error.
//!
//! # Provenance
//!
//! Functional form and weights: AutoDock Vina 1.2 — Trott & Olson, *J. Comput.
//! Chem.* **31**, 455 (2010), Apache-2.0. Vinardo weights: Quiroga & Villarreal,
//! *PLoS ONE* **11**, e0163579 (2016), CC-BY. Both are open source; the code
//! here is an independent implementation from the published equations.

use serde::{Deserialize, Serialize};

use crate::grid::{MapSlot, MAPS_PER_TYPE};
use crate::types::{Atom, AtomKind, Element, Vec3};

/// Below this separation a pair is purely steric and no other term applies.
pub const STERIC_LIMIT: f64 = 0.0;

/// Largest separation at which any term is non-zero, in Ångström.
pub const VINA_CUTOFF: f64 = 8.0;

// ---------------------------------------------------------------------------
// Spatial shape functions and their derivatives
// ---------------------------------------------------------------------------

/// Cubic smoothstep rising from 0 at `a` to 1 at `b`, with zero derivative at
/// both ends.
#[inline]
fn smoothstep(a: f64, b: f64, x: f64) -> (f64, f64) {
    if x <= a {
        return (0.0, 0.0);
    }
    if x >= b {
        return (1.0, 0.0);
    }
    let t = (x - a) / (b - a);
    let s = t * t * (3.0 - 2.0 * t);
    let ds = 6.0 * t * (1.0 - t) / (b - a);
    (s, ds)
}

/// Directionless hydrogen-bond term: 1 when the surfaces overlap, fading to 0
/// as they separate to the point of no contact.
///
/// The window is `-0.5 ≤ d ≤ 0`, which is what puts the maximum at a real
/// heavy-atom hydrogen-bond distance. With the XS radii in
/// [`crate::types::Element::interaction_radius`], `d = -0.5` is an O···O
/// separation of 2.7 Å and an N···O separation of 2.85 Å — the crystallographic
/// range for a strong hydrogen bond, whose `D···A` distance is 2.6–2.9 Å. At
/// `d = 0` the surfaces are merely touching and the term is spent.
///
/// This was `smoothstep(-0.7, -0.5, d)`, i.e. the same shape shifted 0.5 Å
/// too close: its maximum sat at 2.5 Å for an O···O pair, at the far edge of
/// any real hydrogen bond and well inside the clash region.
#[inline]
pub fn hbond_term(d: f64) -> (f64, f64) {
    let (s, ds) = smoothstep(-0.5, 0.0, d);
    (1.0 - s, -ds)
}

/// Apolar (hydrophobic) contact term: 1 for close apolar surfaces, fading out
/// by 1.5 Å.
#[inline]
pub fn hydrophobic_term(d: f64) -> (f64, f64) {
    let (s, ds) = smoothstep(0.5, 1.5, d);
    (1.0 - s, -ds)
}

/// Short-range steric repulsion: quadratic below contact, zero above.
#[inline]
pub fn repulsion_term(d: f64) -> (f64, f64) {
    if d < STERIC_LIMIT {
        (d * d, 2.0 * d)
    } else {
        (0.0, 0.0)
    }
}

/// `exp(−((d − c)/w)²)` and its derivative.
#[inline]
pub fn gaussian_term(d: f64, c: f64, w: f64) -> (f64, f64) {
    let t = (d - c) / w;
    let e = (-t * t).exp();
    let de = e * (-2.0 * t / w);
    (e, de)
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Coefficients of the Vina empirical scoring function.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VinaWeights {
    /// Weight of `g1`, centred at `d = 0.5` with width 0.5.
    pub gauss1: f64,
    /// Weight of `g2`, centred at `d = 0` with width 0.5.
    pub gauss2: f64,
    /// Weight of the short-range steric term.
    pub repulsion: f64,
    /// Weight of the hydrogen-bond term.
    pub hbond: f64,
    /// Weight of the hydrophobic term.
    pub hydrophobic: f64,
    /// Scale applied to the intramolecular (ligand-internal) energy.
    pub intramolecular_scale: f64,
}

impl Default for VinaWeights {
    fn default() -> Self {
        VinaWeights {
            gauss1: -0.035_579,
            gauss2: -0.005_156,
            repulsion: 0.840_245,
            hbond: -0.587_439,
            hydrophobic: -0.035_069,
            intramolecular_scale: 0.006,
        }
    }
}

impl VinaWeights {
    /// Weights of the Vinardo re-parameterisation.
    pub fn vinardo() -> VinaWeights {
        VinaWeights {
            gauss1: 0.0,
            gauss2: 0.0,
            repulsion: -0.045,
            hbond: -0.030,
            hydrophobic: -0.015,
            intramolecular_scale: 0.0075,
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial kernels used for grid precalculation
// ---------------------------------------------------------------------------

/// The three independent components of a pair interaction, each with its
/// derivative. Fixed size so the precalculation inner loop never allocates.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PairComponents {
    /// Shape complementarity plus steric repulsion — applies to every pair.
    pub shape: (f64, f64),
    /// Hydrogen bond — applies to donor–acceptor pairs only.
    pub hbond: (f64, f64),
    /// Hydrophobic contact — applies to apolar–apolar pairs only.
    pub hydrophobic: (f64, f64),
}

/// The tabulated spatial functions for one scoring function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialKernels {
    /// Distance beyond which everything is zero.
    pub cutoff: f64,
    /// Offset and width of the first Gaussian.
    pub gauss1: (f64, f64),
    /// Offset and width of the second Gaussian.
    pub gauss2: (f64, f64),
    /// Weight of the first Gaussian.
    pub w1: f64,
    /// Weight of the second Gaussian.
    pub w2: f64,
    /// Weight of the steric term.
    pub w_rep: f64,
    /// Weight of the hydrogen-bond term.
    pub w_hb: f64,
    /// Weight of the hydrophobic term.
    pub w_hyd: f64,
}

impl SpatialKernels {
    /// Build kernels for the given Vina weights.
    pub fn from_weights(w: &VinaWeights) -> SpatialKernels {
        SpatialKernels {
            cutoff: VINA_CUTOFF,
            gauss1: (0.5, 0.5),
            gauss2: (0.0, 0.5),
            w1: w.gauss1,
            w2: w.gauss2,
            w_rep: w.repulsion,
            w_hb: w.hbond,
            w_hyd: w.hydrophobic,
        }
    }

    /// Evaluate every component and its derivative at surface distance `d`.
    #[inline]
    pub fn eval(&self, d: f64) -> PairComponents {
        let (g1, dg1) = gaussian_term(d, self.gauss1.0, self.gauss1.1);
        let (g2, dg2) = gaussian_term(d, self.gauss2.0, self.gauss2.1);
        let (rep, drep) = repulsion_term(d);
        let (hb, dhb) = hbond_term(d);
        let (hyd, dhyd) = hydrophobic_term(d);
        PairComponents {
            shape: (
                self.w1 * g1 + self.w2 * g2 + self.w_rep * rep,
                self.w1 * dg1 + self.w2 * dg2 + self.w_rep * drep,
            ),
            hbond: (self.w_hb * hb, self.w_hb * dhb),
            hydrophobic: (self.w_hyd * hyd, self.w_hyd * dhyd),
        }
    }
}

// ---------------------------------------------------------------------------
// The scoring function trait
// ---------------------------------------------------------------------------

/// A uniform interface over empirical scoring functions.
///
/// Implementors supply the map weights, the precalculation kernels, and the
/// direct pair term used for the intramolecular energy.
pub trait ScoringFunction: Send + Sync {
    /// Short identifier, e.g. `"vina"`.
    fn name(&self) -> &'static str;

    /// Per-map multipliers for one atom: `[shape, hb_from_donor,
    /// hb_from_acceptor, hydrophobic]`.
    ///
    /// These encode the pair classification, so the precalculation loop never
    /// has to re-derive it.
    fn atom_weights(&self, atom: &Atom) -> [f32; MAPS_PER_TYPE];

    /// Kernels used to tabulate the receptor maps.
    fn spatial_kernels(&self) -> SpatialKernels;

    /// Energy of a single pair at surface distance `d`.
    fn pair_energy(&self, a: &Atom, b: &Atom, d: f64) -> f64;

    /// `d(pair_energy)/dd` at surface distance `d`.
    fn pair_gradient(&self, a: &Atom, b: &Atom, d: f64) -> f64;

    /// Scale applied to the intramolecular energy.
    fn intramolecular_scale(&self) -> f64;

    /// Human-readable one-line description.
    fn description(&self) -> String {
        format!("{} ({})", self.name(), self.name())
    }
}

/// The default scoring function: the AutoDock Vina functional form.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VinaScoring {
    /// The coefficients in use.
    pub weights: VinaWeights,
}

impl VinaScoring {
    /// Vina with the published weights.
    pub fn new() -> VinaScoring {
        VinaScoring::default()
    }

    /// Vinardo, which drops the Gaussians and re-weights the rest.
    pub fn vinardo() -> VinaScoring {
        VinaScoring {
            weights: VinaWeights::vinardo(),
        }
    }
}

impl ScoringFunction for VinaScoring {
    fn name(&self) -> &'static str {
        if self.weights.gauss1 == 0.0 {
            "vinardo"
        } else {
            "vina"
        }
    }

    fn atom_weights(&self, atom: &Atom) -> [f32; MAPS_PER_TYPE] {
        weights_for_kind(atom.kind)
    }

    fn spatial_kernels(&self) -> SpatialKernels {
        SpatialKernels::from_weights(&self.weights)
    }

    fn pair_energy(&self, a: &Atom, b: &Atom, d: f64) -> f64 {
        let k = self.spatial_kernels();
        let c = k.eval(d);
        let mut e = c.shape.0;
        if (a.can_donate() && b.can_accept()) || (a.can_accept() && b.can_donate()) {
            e += c.hbond.0;
        }
        if a.is_apolar() && b.is_apolar() {
            e += c.hydrophobic.0;
        }
        e
    }

    fn pair_gradient(&self, a: &Atom, b: &Atom, d: f64) -> f64 {
        let k = self.spatial_kernels();
        let c = k.eval(d);
        let mut e = c.shape.1;
        if (a.can_donate() && b.can_accept()) || (a.can_accept() && b.can_donate()) {
            e += c.hbond.1;
        }
        if a.is_apolar() && b.is_apolar() {
            e += c.hydrophobic.1;
        }
        e
    }

    fn intramolecular_scale(&self) -> f64 {
        self.weights.intramolecular_scale
    }

    fn description(&self) -> String {
        format!(
            "{}: g1={:.6} g2={:.6} rep={:.6} hbond={:.6} hyd={:.6}",
            self.name(),
            self.weights.gauss1,
            self.weights.gauss2,
            self.weights.repulsion,
            self.weights.hbond,
            self.weights.hydrophobic
        )
    }
}

/// Map weights implied by an atom's interaction class.
///
/// A ligand acceptor reads the map tabulated from receptor **donors** and vice
/// versa, which is what makes the decomposition exact.
pub fn weights_for_kind(kind: AtomKind) -> [f32; MAPS_PER_TYPE] {
    let mut w = [0.0f32; MAPS_PER_TYPE];
    w[MapSlot::Shape.index()] = 1.0;
    match kind {
        AtomKind::Hydrophobic => w[MapSlot::Hydrophobic.index()] = 1.0,
        AtomKind::Donor => w[MapSlot::HbFromAcceptor.index()] = 1.0,
        AtomKind::Acceptor => w[MapSlot::HbFromDonor.index()] = 1.0,
        AtomKind::DonorAcceptor => {
            w[MapSlot::HbFromDonor.index()] = 1.0;
            w[MapSlot::HbFromAcceptor.index()] = 1.0;
        }
        AtomKind::Other => {}
    }
    w
}

/// Which map slots a *receptor* atom contributes to.
#[inline]
pub fn receptor_slots(atom: &Atom) -> (bool, bool, bool) {
    (atom.can_donate(), atom.can_accept(), atom.is_apolar())
}

/// Surface distance between two atoms.
#[inline]
pub fn surface_distance(a: &Atom, b: &Atom, ra: Vec3, rb: Vec3) -> f64 {
    let dx = ra[0] - rb[0];
    let dy = ra[1] - rb[1];
    let dz = ra[2] - rb[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
        - (a.element.interaction_radius() + b.element.interaction_radius())
}

/// Element index helper re-exported for the grid precalculation.
#[inline]
pub fn interaction_radius(e: Element) -> f64 {
    e.interaction_radius()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AtomType;

    fn atom(element: Element, t: AtomType, kind: AtomKind) -> Atom {
        let mut a = Atom::new(1, [0.0, 0.0, 0.0], element, t);
        a.kind = kind;
        a
    }

    #[test]
    fn smoothstep_endpoints_and_monotonicity() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), (0.0, 0.0));
        assert_eq!(smoothstep(0.0, 1.0, 2.0), (1.0, 0.0));
        let mut prev = -1.0;
        for i in 0..=100 {
            let x = i as f64 / 100.0;
            let (v, _) = smoothstep(0.0, 1.0, x);
            assert!(v >= prev - 1e-12, "smoothstep must be non-decreasing");
            prev = v;
        }
    }

    /// Every spatial function's analytic derivative must match central
    /// differences — the analytic-gradient machinery depends on it.
    #[test]
    fn spatial_derivatives_match_finite_differences() {
        let k = SpatialKernels::from_weights(&VinaWeights::default());
        // Sample strictly between the smoothstep knots. At a knot the function
        // is C¹ but not C², so a central difference straddles a curvature jump
        // and reports a small spurious slope; `smoothstep_is_c1` checks the
        // knot behaviour directly instead.
        //
        // The knots are `hb` at −0.5 and 0.0, and `hyd` at 0.5 and 1.5. Keep
        // this list clear of them — if a term's window is ever retuned, these
        // are the values that move.
        for &d in &[
            -1.2, -0.68, -0.6, -0.52, -0.4, -0.1, 0.3, 0.52, 0.9, 1.4, 2.0, 3.0,
        ] {
            let c = k.eval(d);
            let h = 1e-6;
            let cp = k.eval(d + h);
            let cm = k.eval(d - h);
            for (name, (val, der), (hp, hm)) in [
                ("shape", c.shape, (cp.shape.0, cm.shape.0)),
                ("hbond", c.hbond, (cp.hbond.0, cm.hbond.0)),
                ("hyd", c.hydrophobic, (cp.hydrophobic.0, cm.hydrophobic.0)),
            ] {
                let numeric = (hp - hm) / (2.0 * h);
                assert!(
                    (numeric - der).abs() < 1e-6,
                    "{name} at d={d}: numeric {numeric:.9} vs analytic {der:.9}"
                );
                let _ = val;
            }
        }
    }

    /// The smoothsteps must have matching one-sided slopes at every knot.
    /// Otherwise the assembled energy would have a gradient discontinuity and
    /// quasi-Newton search would stall whenever a ligand atom crossed a knot.
    ///
    /// The windows below are the ones the terms actually use. That used to say
    /// `(-0.7, -0.5)` for `hb`, which the term stopped using when the window
    /// moved to `(-0.5, 0.0)`, so for a while this test checked a shape nobody
    /// was drawing. A C¹ property is a property of `smoothstep` and not of any
    /// one window, so the test could not notice -- and it still cannot, which is
    /// why the window itself is pinned by
    /// `the_hbond_window_lands_on_the_documented_separations` instead.
    #[test]
    fn smoothstep_is_c1() {
        for &(a, b) in &[(-0.5, 0.0), (0.5, 1.5)] {
            let h = 1e-7;
            let left = smoothstep(a, b, a + h).1;
            let right = smoothstep(a, b, b - h).1;
            assert!(left < 1e-3, "{a}..{b}: left slope {left} should vanish");
            assert!(right < 1e-3, "{a}..{b}: right slope {right} should vanish");
            // And the value must reach the endpoints exactly.
            assert_eq!(smoothstep(a, b, a).0, 0.0);
            assert_eq!(smoothstep(a, b, b).0, 1.0);
        }
    }

    #[test]
    fn hbond_and_hydrophobic_ranges() {
        for d in [-2.0, -1.0, -0.7] {
            let (v, _) = hbond_term(d);
            assert!(
                (v - 1.0).abs() < 1e-9,
                "hbond should saturate at 1, got {v}"
            );
        }
        let (v, _) = hbond_term(0.0);
        assert!(v.abs() < 1e-12);
        let (v, _) = hydrophobic_term(0.0);
        assert!((v - 1.0).abs() < 1e-9);
        let (v, _) = hydrophobic_term(2.0);
        assert!(v.abs() < 1e-12);
    }

    /// The hydrogen-bond window, restated in ångström of real interatomic
    /// distance, is part of the scoring function's contract -- and this is the
    /// assertion that keeps the docstrings that quote it honest.
    ///
    /// `d = ‖rᵢ − rⱼ‖ − (Rᵢ + Rⱼ)`, so a real separation `s` is sampled at
    /// `d = s − (Rᵢ + Rⱼ)`. Reading the window back into ångström needs the
    /// radius table too, which is why `hbond_term` and
    /// `Element::interaction_radius` cannot honestly be checked one at a time:
    /// this test moves all three claims at once, or none.
    ///
    /// The discriminating sample is `d = −0.5`. The retired window
    /// `(−0.7, −0.5)` reads **0.0** there and the current `(−0.5, 0.0)` reads
    /// **1.0**, so a revert turns this red.
    ///
    /// A revert is *not* invisible to the suite — reverting the window was
    /// measured to fail `types::tests::the_interaction_radii_place_the_terms_at_real_contact_distances`
    /// as well, which samples 2.8 Å and demands more than half the term. What
    /// the suite never checked was a *number*, and that is the gap this closes:
    /// the range test above samples `−2.0, −1.0, −0.7` and `0.0`, where both
    /// windows answer identically, and `smoothstep_is_c1` cannot see windows at
    /// all because C¹ holds for every one of them. So 2.70 Å and 2.85 Å were
    /// quoted in two docstrings and asserted by nothing — which is how
    /// `types.rs` came to say the hydrogen-bond term peaks at the 3.2 Å contact
    /// where it is in fact exactly zero.
    #[test]
    fn the_hbond_window_lands_on_the_documented_separations() {
        let d_at = |s: f64, ri: f64, rj: f64| s - (ri + rj);
        let ro = crate::types::Element::O.interaction_radius();
        let rn = crate::types::Element::N.interaction_radius();

        // Saturated: a 2.70 Å O···O or 2.85 Å N···O contact -- the
        // crystallographic range for a strong hydrogen bond -- is worth the
        // full term.
        assert_eq!(
            hbond_term(d_at(2.70, ro, ro)).0,
            1.0,
            "O...O at 2.70 A must still be fully hydrogen bonded"
        );
        assert_eq!(
            hbond_term(d_at(2.85, rn, ro)).0,
            1.0,
            "N...O at 2.85 A must still be fully hydrogen bonded"
        );

        // Strictly inside the window, which is what separates `a = −0.5` from
        // the retired `a = −0.7`: 2.80 Å is `d = −0.40`, mid-decay at 0.896
        // under the current window and hard 0.0 under the old one.
        let (v, _) = hbond_term(d_at(2.80, ro, ro));
        assert!(
            v > 0.0 && v < 1.0,
            "O...O at 2.80 A should be mid-window, got {v}"
        );

        // Spent: at `d = 0` the surfaces are merely touching, and past it there
        // is no contact at all.
        assert_eq!(hbond_term(0.0).0, 0.0, "the term is spent at contact");
        assert_eq!(hbond_term(0.5).0, 0.0, "and stays spent past it");
    }

    #[test]
    fn repulsion_is_always_non_negative() {
        for d in [-2.0, -0.5, -0.001, 0.0, 1.0] {
            let (v, der) = repulsion_term(d);
            assert!(v >= 0.0);
            if d < 0.0 {
                assert!(
                    der < 0.0,
                    "repulsion must be attractive-gradient below contact"
                );
            }
        }
    }

    #[test]
    fn weights_reflect_pair_classification() {
        let hyd = weights_for_kind(AtomKind::Hydrophobic);
        assert_eq!(hyd[MapSlot::Shape.index()], 1.0);
        assert_eq!(hyd[MapSlot::Hydrophobic.index()], 1.0);
        assert_eq!(hyd[MapSlot::HbFromDonor.index()], 0.0);

        let donor = weights_for_kind(AtomKind::Donor);
        assert_eq!(donor[MapSlot::HbFromAcceptor.index()], 1.0);
        assert_eq!(donor[MapSlot::HbFromDonor.index()], 0.0);

        let acceptor = weights_for_kind(AtomKind::Acceptor);
        assert_eq!(acceptor[MapSlot::HbFromDonor.index()], 1.0);

        let both = weights_for_kind(AtomKind::DonorAcceptor);
        assert_eq!(both[MapSlot::HbFromDonor.index()], 1.0);
        assert_eq!(both[MapSlot::HbFromAcceptor.index()], 1.0);
    }

    #[test]
    fn pair_energy_and_gradient_agree() {
        let s = VinaScoring::new();
        let c = atom(Element::C, AtomType::CH, AtomKind::Hydrophobic);
        let o = atom(Element::O, AtomType::OA, AtomKind::Acceptor);
        for &d in &[-1.0, -0.6, 0.0, 0.4, 1.0, 2.0] {
            let e = s.pair_energy(&c, &o, d);
            let h = 1e-6;
            let numeric = (s.pair_energy(&c, &o, d + h) - s.pair_energy(&c, &o, d - h)) / (2.0 * h);
            let analytic = s.pair_gradient(&c, &o, d);
            assert!(
                (numeric - analytic).abs() < 1e-6,
                "d={d}: numeric {numeric:.9} vs analytic {analytic:.9}"
            );
            let _ = e;
        }
    }

    #[test]
    fn hbond_term_only_for_opposite_polarity() {
        let s = VinaScoring::new();
        let donor = atom(Element::N, AtomType::NP, AtomKind::Donor);
        let acceptor = atom(Element::O, AtomType::OA, AtomKind::Acceptor);
        let other_donor = atom(Element::N, AtomType::NP, AtomKind::Donor);
        // Donor–acceptor at d = -0.8 is well inside the hydrogen-bond well.
        let hb = s.pair_energy(&donor, &acceptor, -0.8);
        let dd = s.pair_energy(&donor, &other_donor, -0.8);
        assert!(hb < dd - 0.1, "donor-donor should not gain a hydrogen bond");
    }

    #[test]
    fn hydrophobic_term_only_for_apolar_pairs() {
        let s = VinaScoring::new();
        let c1 = atom(Element::C, AtomType::CH, AtomKind::Hydrophobic);
        let c2 = atom(Element::C, AtomType::CH, AtomKind::Hydrophobic);
        let o = atom(Element::O, AtomType::OA, AtomKind::Acceptor);
        let cc = s.pair_energy(&c1, &c2, 0.0);
        let co = s.pair_energy(&c1, &o, 0.0);
        assert!(cc < co, "apolar contact should be better than polar-apolar");
    }

    #[test]
    fn vinardo_has_no_gaussian_terms() {
        let v = VinaScoring::vinardo();
        assert_eq!(v.name(), "vinardo");
        assert_eq!(SpatialKernels::from_weights(&v.weights).w1, 0.0);
        let c = atom(Element::C, AtomType::CH, AtomKind::Hydrophobic);
        let c2 = atom(Element::C, AtomType::CH, AtomKind::Hydrophobic);
        // At large separation both functions must be exactly zero.
        assert!(v.pair_energy(&c, &c2, 9.0).abs() < 1e-12);
    }
}
