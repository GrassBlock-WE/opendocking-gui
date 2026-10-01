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
        let (v, dv) = self.eval_terms(d);
        PairComponents {
            shape: (Self::shape_of(&v), Self::shape_of(&dv)),
            hbond: (
                v[TermField::HbFromDonor.index()],
                dv[TermField::HbFromDonor.index()],
            ),
            hydrophobic: (
                v[TermField::Hydrophobic.index()],
                dv[TermField::Hydrophobic.index()],
            ),
        }
    }

    /// The three shape terms added together, in the order the spec writes them.
    ///
    /// The hydrogen-bond slot takes only the donor half because
    /// [`PairComponents::hbond`] is a single slot that the *interpolation*
    /// weight later decides is read or not; the two halves are separated at
    /// write time and recombined by weight, so this sum is a shape-only sum.
    #[inline]
    fn shape_of(v: &TermValues) -> f64 {
        v[TermField::Gauss1.index()]
            + v[TermField::Gauss2.index()]
            + v[TermField::Repulsion.index()]
    }

    /// Every term's weighted value and its derivative at surface distance `d`.
    ///
    /// **This is the only place a Vina weight is ever multiplied in.** Both the
    /// four-slot [`Self::eval`] and the six-field per-term path call it, so a
    /// weight cannot be applied twice in one and once in the other — the
    /// failure mode that would make the per-term numbers disagree with the total
    /// for a reason that has nothing to do with the terms themselves.
    ///
    /// `HbFromDonor` and `HbFromAcceptor` carry the *same* weighted value
    /// (`w_hb · hb(d)`); they are distinct fields only because a probe may read
    /// one and not the other, which is decided at interpolation by
    /// [`probe_term_mask`], not here.
    #[inline]
    pub fn eval_terms(&self, d: f64) -> (TermValues, TermValues) {
        let (g1, dg1) = gaussian_term(d, self.gauss1.0, self.gauss1.1);
        let (g2, dg2) = gaussian_term(d, self.gauss2.0, self.gauss2.1);
        let (rep, drep) = repulsion_term(d);
        let (hb, dhb) = hbond_term(d);
        let (hyd, dhyd) = hydrophobic_term(d);
        (
            [
                self.w1 * g1,
                self.w2 * g2,
                self.w_rep * rep,
                self.w_hb * hb,
                self.w_hb * hb,
                self.w_hyd * hyd,
            ],
            [
                self.w1 * dg1,
                self.w2 * dg2,
                self.w_rep * drep,
                self.w_hb * dhb,
                self.w_hb * dhb,
                self.w_hyd * dhyd,
            ],
        )
    }
}

// ---------------------------------------------------------------------------
// Per-term decomposition
// ---------------------------------------------------------------------------
//
// # Why the terms are not readable off the map
//
// The production grid stores **four** slots per (point, probe type), and slot
// `Shape` is `w1·g1 + w2·g2 + w_rep·rep` fused into a *single* `f32` before it
// is ever written. Three of the five terms are therefore not recoverable from
// a [`crate::grid::GridMaps`] by any amount of reading: they were summed
// together at precalculation time and the sum is all that survives. Nor is the
// split an argument about precision — the f32 field holds the *rounded* sum, so
// the three pieces are gone, not merely blurred.
//
// The fix is a second tabulation over the same geometry, with one field per
// term. It is deliberately *not* a change to `MAPS_PER_TYPE`, because the WGSL
// kernel hard-codes the same stride and the `gpu` feature's test asserts the two
// agree; splitting `Shape` in place would move the GPU's memory layout without
// moving its kernel. [`TermMaps`] therefore costs its own memory and is built on
// demand, and the docking path never touches it.
//
// # Where the weights are applied — the point this module was vague about
//
// **The term weights are applied when the map is tabulated, not when a
// conformation is scored.** `SpatialKernels::eval_terms` is the only place any
// weight is multiplied in, it runs inside the precalculation loop, and what it
// returns is stored as a map field. By the time `score_conformation` runs there
// is no weight left to apply: the only per-atom multiplier still alive is the
// 0/1 **class mask** from [`weights_for_kind`], which selects *which* slots a
// probe reads and is not a weight in the Vina sense (its entries are 0.0 or 1.0
// and it carries no coefficient).
//
// The consequence for a caller: every term reported by [`TermMaps`] is already
// **weighted**, in kcal/mol, on the same scale as the total. There is no
// unweighted variant, and multiplying by the weight again would double-count it.

/// Number of tabulated fields per grid point needed to rebuild the five terms.
///
/// Six, not five. The hydrogen bond is one *term* but two *fields*, because
/// which of the two a probe may read is decided by the **receptor** atom's
/// class while the map is being written and by the **probe** atom's class when
/// it is read; merging them into one field earlier would put a receptor
/// donor's hydrogen bond where a donor probe can see it, which is precisely
/// the spurious donor–donor bond the two-slot split exists to prevent. The
/// merge into a single `hb` term therefore happens at *read* time, in
/// [`probe_term_mask`], and nowhere earlier.
pub const TERM_FIELDS: usize = 6;

/// A tabulated per-term value (or its derivative), in kcal/mol.
pub type TermValues = [f64; TERM_FIELDS];

/// One of the six tabulated fields that together rebuild the five Vina terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermField {
    /// `g1`, centred at `d = 0.5` with width 0.5.
    Gauss1 = 0,
    /// `g2`, centred at `d = 0.0` with width 0.5.
    Gauss2 = 1,
    /// `rep`, the quadratic wall below contact.
    Repulsion = 2,
    /// Hydrogen bond written by receptor **donors**, read by probe acceptors.
    HbFromDonor = 3,
    /// Hydrogen bond written by receptor **acceptors**, read by probe donors.
    HbFromAcceptor = 4,
    /// Apolar contact.
    Hydrophobic = 5,
}

impl TermField {
    /// All six fields, in storage order.
    pub const ALL: [TermField; TERM_FIELDS] = [
        TermField::Gauss1,
        TermField::Gauss2,
        TermField::Repulsion,
        TermField::HbFromDonor,
        TermField::HbFromAcceptor,
        TermField::Hydrophobic,
    ];

    /// Dense index into a [`TermValues`].
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Which fields a **receptor** atom writes while the map is tabulated.
#[inline]
pub fn receptor_term_mask(atom: &Atom) -> [bool; TERM_FIELDS] {
    [
        true,
        true,
        true,
        atom.can_donate(),
        atom.can_accept(),
        atom.is_apolar(),
    ]
}

/// Which fields a **probe** atom reads when the map is interpolated.
///
/// The hydrogen-bond entries are crossed against the receptor's: a probe
/// *donor* reads `HbFromAcceptor` and a probe *acceptor* reads `HbFromDonor`,
/// which is the same like-with-like withholding that
/// [`crate::grid::MapSlot`] documents. Getting the crossing backwards would
/// give a receptor acceptor its hydrogen bond back to a receptor acceptor.
#[inline]
pub fn probe_term_mask(atom: &Atom) -> [bool; TERM_FIELDS] {
    [
        true,
        true,
        true,
        atom.can_accept(),
        atom.can_donate(),
        atom.is_apolar(),
    ]
}

/// Which compute backend can produce a [`TermBreakdown`].
///
/// # Why this is a type and not a sentence
///
/// The per-term decomposition is **CPU-only**, and that has to be visible at
/// the point of call rather than in a document — a caller holding a breakdown
/// must be able to ask, not infer.
///
/// The alternative would be a build where a `gpu` feature silently changed
/// where the five terms come from, or where a `gpu` build returned *fewer* of
/// them. Either would be worse than the bug this decomposition was written to
/// find, because it would be invisible on whichever build the user happened to
/// install. So the fact is carried in two places a caller actually touches:
///
/// * [`TermBreakdown::backend`], on the value itself, so an answer cannot be
///   separated from the statement of what produced it; and
/// * [`crate::grid::TermMaps::BACKEND`], for a caller deciding whether to
///   build the tabulation at all.
///
/// There is exactly one variant, and adding a second is a deliberate act: it
/// would mean a shader had learned the 60-float stride, at which point the
/// two backends would have to be proven equal term by term before this enum
/// could grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TermBackend {
    /// The CPU. [`crate::grid::TermMaps`] is tabulated on the host with rayon
    /// and read back on the host.
    ///
    /// The WGSL kernel takes a [`crate::grid::GridMaps`] and nothing else — its
    /// signature is `score(&Batch, &GridMaps)` — so a `TermMaps` cannot reach a
    /// shader even by accident. `TermMaps` is 60 floats per grid point against
    /// the production map's 40, and the kernel hard-codes the 40.
    #[default]
    Cpu,
}

/// The per-term decomposition of one conformation's intermolecular energy.
///
/// Every field is in **kcal/mol** and is **already weighted** — see the module
/// section on the weight-application point. The five terms are `gauss1`,
/// `gauss2`, `repulsion`, `hbond` and `hydrophobic`; the hydrogen bond is also
/// reported split by receptor polarity, because that split is the property that
/// decides whether a probe can see the bond at all.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TermBreakdown {
    /// Which backend produced this breakdown. Always [`TermBackend::Cpu`].
    ///
    /// Read this to answer "can the GPU do this?" without leaving the value.
    /// There is no variant in which this is anything else, which is the point:
    /// the question has an answer a caller can type-check, and the only answer
    /// is ever "no, the CPU did this".
    pub backend: TermBackend,

    /// `g1 = −0.035579 · exp(−((d − 0.5)/0.5)²)`, summed over every pair.
    pub gauss1: f64,
    /// `g2 = −0.005156 · exp(−(d/0.5)²)`, summed over every pair.
    pub gauss2: f64,
    /// `rep = 0.840245 · d²` for `d < 0`, summed over every pair.
    pub repulsion: f64,
    /// The hydrogen bond: `hbond_from_donor + hbond_from_acceptor`.
    pub hbond: f64,
    /// The apolar contact term.
    pub hydrophobic: f64,
    /// The donor-polarity half of [`TermBreakdown::hbond`].
    pub hbond_from_donor: f64,
    /// The acceptor-polarity half of [`TermBreakdown::hbond`].
    pub hbond_from_acceptor: f64,
    /// Out-of-box penalty, in kcal/mol; zero unless an atom left the grid.
    ///
    /// Reported separately because it is **not** a Vina term: the search adds a
    /// flat linear penalty to pull a violating pose back, and folding it into
    /// the nearest term would make a term's value depend on whether the pose
    /// happened to be inside the box.
    pub out_of_box_penalty: f64,
}

impl TermBreakdown {
    /// The five Vina terms added together, excluding the out-of-box penalty.
    ///
    /// This is the intermolecular energy the five terms are claimed to account
    /// for, and it is the quantity the cross-check holds the engine to.
    #[must_use]
    pub fn terms_total(&self) -> f64 {
        self.gauss1 + self.gauss2 + self.repulsion + self.hbond + self.hydrophobic
    }

    /// The five terms plus the out-of-box penalty.
    ///
    /// This is the number a caller should compare against the engine's
    /// intermolecular energy, which includes the penalty for the same reason the
    /// search does.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.terms_total() + self.out_of_box_penalty
    }

    /// The `Shape` slot's value: the three terms the production map fuses.
    ///
    /// The production grid stores exactly this sum in one `f32`, so this is the
    /// number a per-slot reading would report for slot 0.
    #[must_use]
    pub fn shape(&self) -> f64 {
        self.gauss1 + self.gauss2 + self.repulsion
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

/// Every number the scoring path reads, pinned against the line of
/// `docs/SCORING.md` that states it.
///
/// # Why the failure message carries the documentation
///
/// The measurement that motivated this module: changing the hydrophobic window
/// from `1.5` to `1.45` leaves the engine scoring a **different term** and
/// passed **117 of 117** tests that were already in the suite. The suite pinned
/// the hydrogen-bond window and no other number, so a term nobody had pinned
/// could be retuned freely.
///
/// A test asserting `hydrophobic_term(1.45) == 0.00725` would have gone red,
/// and a test merely saying "the hydrophobic window is 1.5" would have gone red
/// for the same reason — but would have told its reader nothing about *why* 1.5.
/// So every assertion below names the documented line. A constant and the
/// sentence that states it then have to change together, and a reader who trips
/// one is sent to the specification rather than left to guess which side is
/// stale.
///
/// That is the failure §5.1's measurement table fell into: the numbers were
/// measured, written down, and then the code was fixed around them while the
/// table kept quoting the old ones. A number with no sentence above it is a
/// number nobody re-derived.
///
/// # What is deliberately not here
///
/// `OUT_OF_BOX_PENALTY`, `MAX_GRID_POINTS` and `TERM_FIELDS` have **no line in
/// `SCORING.md` at all**. They are pinned in `grid.rs` with failure messages
/// that say so, because a scoring constant with no specification is a finding
/// the reader should be shown, not a blank to fill in here.
#[cfg(test)]
mod pinned_constants {
    use super::*;
    use crate::ligand::MIN_INTRA_BOND_DISTANCE;
    use crate::scoring::TermField as F;

    /// `docs/SCORING.md` §2.3, the smoothstep definitions.
    const SMOOTHSTEP_DOC: &str = "SCORING.md §2.3: S(a, b, x) = 0 当 x <= a / = t^2(3-2t) 当 a < x < b, t = (x-a)/(b-a) / = 1 当 x >= b";
    /// `docs/SCORING.md` §2.3, the two windowed terms.
    const WINDOW_DOC: &str = "SCORING.md §2.3: hb(d) = 1 - S(-0.5,  0.0, d)   深度重叠时为 1，到 0 A 归零\n\
                              SCORING.md §2.3: hyd(d) = 1 - S( 0.5,  1.5, d)  近程 apolar 接触为 1，到 1.5 A 归零";
    /// `docs/SCORING.md` §2.1, the Gaussian table.
    const GAUSS_DOC: &str = "SCORING.md §2.1: | `g1` | `0.5` | `0.5` | `-0.035579` | `0` |  and  | `g2` | `0.0` | `0.5` | `-0.005156` | `0` |\n\
                             (columns: centre c, width w, vina weight, vinardo weight)";
    /// `docs/SCORING.md` §3, the Vina weight block.
    const VINA_WEIGHT_DOC: &str = "SCORING.md §3:\n\
        VinaWeights {\n\
        \x20   gauss1:   -0.035579,\n\
        \x20   gauss2:   -0.005156,\n\
        \x20   repulsion:  0.840245,\n\
        \x20   hbond:    -0.587439,\n\
        \x20   hydrophobic: -0.035069,\n\
        \x20   intramolecular_scale: 0.006,\n\
        }";
    /// `docs/SCORING.md` §3, the Vinardo weight block.
    const VINARDO_WEIGHT_DOC: &str =
        "SCORING.md §3 (Vinardo, Quiroga & Villarreal, PLoS ONE 11, e0163579, 2016):\n\
        gauss1 = gauss2 = 0        // 丢掉两个高斯\n\
        repulsion   = -0.045       // 注意是负的\n\
        hbond       = -0.030\n\
        hydrophobic = -0.015\n\
        intramolecular_scale = 0.0075";
    /// `docs/SCORING.md` §2.2, the repulsion branch.
    const REPULSION_DOC: &str = "SCORING.md §2.2: rep(d) = d^2  当 d < 0  /  = 0  当 d >= 0;  drep/dd = 2d 当 d < 0, 否则 0";
    /// `docs/SCORING.md` §5.3, the surface-distance cutoff.
    const CUTOFF_DOC: &str = "SCORING.md §5.3: `SpatialKernels::cutoff = 8.0` 是表面距离的上限，而内循环手里是真实距离，所以真实距离的截断应当是 `cutoff + 2*R`";
    /// `docs/SCORING.md` §4.1, the intramolecular slope and graph distance.
    const INTRA_DOC: &str = "SCORING.md §4.1: E_total = E_inter + slope * E_intra,  slope = 0.006（vina）/ 0.0075（vinardo）\n\
                             SCORING.md §4.1: `ligand.rs::MIN_INTRA_BOND_DISTANCE = 4`，即 1-4 及更远";

    /// Assert that a scoring constant still equals the value `SCORING.md`
    /// documents for it, and name the documented line if it does not.
    ///
    /// The equality is the easy half. What this adds over `assert_eq!` is that
    /// the specification line travels with the assertion, so a reader who trips
    /// it learns which document to read — and a maintainer changing the
    /// constant is told, in the failure they will see, that the document is
    /// now wrong too.
    fn assert_pinned<T: PartialEq + std::fmt::Debug>(
        constant: &str,
        found: T,
        documented: T,
        doc: &str,
    ) {
        if found != documented {
            panic!(
                "{constant} is {found:?}, but the specification documents {documented:?}.\n\
                 documented: {doc}\n\
                 The constant and the line that states it have to change together. \
                 A number nobody re-derived is worse than a number nobody wrote down."
            );
        }
    }

    /// `smoothstep` is the one function both windowed terms are built from, and
    /// the specification writes its definition out. Pinned to that definition
    /// rather than to a property: the C¹ behaviour is a property of *every*
    /// window, which is exactly why `smoothstep_is_c1` could not notice the
    /// hydrogen-bond window moving (see its doc comment).
    ///
    /// A retune of the *shape* — as opposed to the window — would move these
    /// numbers while leaving both windows intact, and would be a silent change
    /// to every distance-dependent term in the function.
    #[test]
    fn the_smoothstep_is_the_documented_cubic() {
        // The three branches, as written: 0 at and below a, 1 at and above b.
        assert_pinned("S at x=a", smoothstep(0.5, 1.5, 0.5).0, 0.0, SMOOTHSTEP_DOC);
        assert_pinned("S at x<a", smoothstep(0.5, 1.5, 0.4).0, 0.0, SMOOTHSTEP_DOC);
        assert_pinned("S at x=b", smoothstep(0.5, 1.5, 1.5).0, 1.0, SMOOTHSTEP_DOC);
        assert_pinned("S at x>b", smoothstep(0.5, 1.5, 1.6).0, 1.0, SMOOTHSTEP_DOC);
        // The interior branch, t^2(3 - 2t). At t = 1/2 that is exactly 0.5,
        // which is also why both window midpoints read exactly 0.5 above.
        assert_pinned(
            "S at t=0.5",
            smoothstep(0.5, 1.5, 1.0).0,
            0.5,
            SMOOTHSTEP_DOC,
        );
        // ... and at t = 1/4, t^2(3-2t) = (1/16)(5/2) = 0.15625 exactly.
        let t = 0.25;
        assert_pinned(
            "S at t=0.25 against t^2(3-2t)",
            smoothstep(0.5, 1.5, 0.5 + t).0,
            t * t * (3.0 - 2.0 * t),
            SMOOTHSTEP_DOC,
        );
        // The derivative branch, 6t(1-t)/(b-a), which is what the analytic
        // gradient depends on. At the midpoint that is 6(0.25)/1 = 1.5.
        assert_pinned(
            "dS/dx at t=0.5 over a unit window",
            smoothstep(0.5, 1.5, 1.0).1,
            1.5,
            SMOOTHSTEP_DOC,
        );
        // The two window widths are both 1.0, so the peak slope is 1.5 for
        // each. If a window were ever widened, this is the number that moves.
        assert_pinned(
            "peak dS/dx for the hyd window",
            smoothstep(0.5, 1.5, 1.0).1,
            6.0 / 1.0 * 0.25,
            SMOOTHSTEP_DOC,
        );
        assert_pinned(
            "peak dS/dx for the hb window",
            smoothstep(-0.5, 0.0, -0.25).1,
            6.0 / 0.5 * 0.25,
            SMOOTHSTEP_DOC,
        );
    }

    /// `hydrophobic_term` writes `smoothstep(0.5, 1.5, d)` inline, so there is no
    /// constant to read — the knots have to be recovered from the values the
    /// function returns.
    ///
    /// Every sample below is **discriminating**: it is a place where a retune of
    /// the window changes the answer. The last two are what turn the measured
    /// `1.5 -> 1.45` mutation red. Under a 1.5 window `hydrophobic_term(1.45)`
    /// is 0.00725 — a term that is 99.3% spent but not spent; under 1.45 the
    /// window ends *at* 1.45, so the function returns exactly 0.0 there and
    /// retires the tail 0.05 Å early. In ångström of a real C···C contact
    /// (`2 * 1.90 = 3.80 Å`) that is 5.28 Å, and §10's table says the term is
    /// `= 0` only from `d >= 1.5`, i.e. from **5.30 Å**.
    #[test]
    fn the_hydrophobic_window_is_the_documented_one() {
        // Saturated at and below the lower knot.
        assert_pinned("hyd at d=0.4", hydrophobic_term(0.4).0, 1.0, WINDOW_DOC);
        assert_pinned(
            "hyd at d=0.5 (lower knot)",
            hydrophobic_term(0.5).0,
            1.0,
            WINDOW_DOC,
        );
        // A cubic smoothstep is symmetric about the middle of its window, so
        // the midpoint value is exactly 0.5 and pins the window's *centre* as
        // well as its two ends. Retuned to 1.45 this reads 0.4605.
        assert_pinned(
            "hyd at d=1.0 (window midpoint)",
            hydrophobic_term(1.0).0,
            0.5,
            WINDOW_DOC,
        );
        // Spent at and past the upper knot.
        assert_pinned(
            "hyd at d=1.5 (upper knot)",
            hydrophobic_term(1.5).0,
            0.0,
            WINDOW_DOC,
        );
        assert_pinned("hyd at d=2.0", hydrophobic_term(2.0).0, 0.0, WINDOW_DOC);
        // 0.05 Å inside the upper knot, the term is still alive. This is the
        // single assertion that makes the 1.5 -> 1.45 mutation red: under a
        // 1.45 window the function ends *at* 1.45 and returns exactly 0.0
        // there, retiring the tail 0.05 Å early.
        //
        // `!= 0.0` rather than equality against a decimal, because the residue
        // is 1 - 0.95²·(3 - 2·0.95) = 0.00725 in exact arithmetic and
        // 0.007249999999999979 in f64; pinning the decimal would be pinning a
        // rounding rather than a window.
        assert_pinned(
            "hyd 0.05 A inside its upper knot, d=1.45: the term is not spent yet",
            hydrophobic_term(1.45).0 != 0.0,
            true,
            WINDOW_DOC,
        );
        // ... and the residue must be the small one the definition gives, not a
        // substantial one. Measured on this build: 0.007249999999999979.
        assert!(
            (hydrophobic_term(1.45).0 - 0.00725).abs() < 1e-15,
            "hyd at d=1.45 is {}, which is not the exact-arithmetic residue \
             1 - 0.95^2*(3 - 2*0.95) = 0.00725 within f64 rounding.\n\
             documented: {WINDOW_DOC}",
            hydrophobic_term(1.45).0
        );
        // The same fact read in the units §10 states it in: a C···C pair
        // (R + R = 3.80 Å) is apolar contact at 5.28 Å and spent at 5.30 Å.
        let cc = 2.0 * Element::C.interaction_radius();
        assert_pinned(
            "hyd for C...C at 5.28 A: still alive, per §10 '= 0 | d >= 1.5 | >= 5.30 A'",
            hydrophobic_term(5.28 - cc).0 != 0.0,
            true,
            WINDOW_DOC,
        );
        assert_pinned(
            "hyd for C...C at 5.30 A: spent, per §10 '= 0 | d >= 1.5 | >= 5.30 A'",
            hydrophobic_term(5.30 - cc).0,
            0.0,
            WINDOW_DOC,
        );
    }

    /// The hydrogen-bond window, pinned the same way.
    ///
    /// The Ångström reading is already covered by
    /// `the_hbond_window_lands_on_the_documented_separations`; this adds the
    /// knot-level claim *with the documented line attached*, so a retune here
    /// says which sentence of the specification is now false.
    #[test]
    fn the_hydrogen_bond_window_is_the_documented_one() {
        // `d = -0.6` is the discriminating sample against the retired
        // `smoothstep(-0.7, -0.5, d)`: the current window answers 1.0 (it is at
        // or past its lower knot), the retired one answers 0.5.
        assert_pinned("hb at d=-0.6", hbond_term(-0.6).0, 1.0, WINDOW_DOC);
        assert_pinned(
            "hb at d=-0.5 (lower knot)",
            hbond_term(-0.5).0,
            1.0,
            WINDOW_DOC,
        );
        // Symmetric about the middle of its window, so exactly 0.5 at d=-0.25.
        // Retuned to the old window this reads 0.0.
        assert_pinned(
            "hb at d=-0.25 (window midpoint)",
            hbond_term(-0.25).0,
            0.5,
            WINDOW_DOC,
        );
        assert_pinned(
            "hb at d=0.0 (upper knot)",
            hbond_term(0.0).0,
            0.0,
            WINDOW_DOC,
        );
        assert_pinned("hb at d=0.5", hbond_term(0.5).0, 0.0, WINDOW_DOC);
    }

    /// The Gaussian centres and widths. These are the separation axis of three
    /// of the five terms, and a retune of either moves where the shape
    /// complementarity well sits without changing any total by more than the
    /// sum-level checks can see.
    #[test]
    fn the_gaussian_centres_and_widths_are_the_documented_ones() {
        let k = SpatialKernels::from_weights(&VinaWeights::default());
        assert_pinned("g1 centre c", k.gauss1.0, 0.5, GAUSS_DOC);
        assert_pinned("g1 width w", k.gauss1.1, 0.5, GAUSS_DOC);
        assert_pinned("g2 centre c", k.gauss2.0, 0.0, GAUSS_DOC);
        assert_pinned("g2 width w", k.gauss2.1, 0.5, GAUSS_DOC);

        // Behaviourally, because a `SpatialKernels` field and the function that
        // reads it could disagree. A Gaussian of width `w` decays by `exp(-1)`
        // at one width from its centre, and is symmetric about it.
        let e1 = (-1.0f64).exp();
        let e4 = (-0.25f64).exp();
        let at = |k: &SpatialKernels, d: f64| k.eval_terms(d).0;
        assert_pinned(
            "w1 * g1 at d=0.5 (its own centre)",
            at(&k, 0.5)[F::Gauss1.index()],
            k.w1,
            GAUSS_DOC,
        );
        assert_pinned(
            "w1 * g1 at d=0.5 - one width",
            at(&k, 0.0)[F::Gauss1.index()],
            k.w1 * e1,
            GAUSS_DOC,
        );
        assert_pinned(
            "w1 * g1 at d=0.5 + one width",
            at(&k, 1.0)[F::Gauss1.index()],
            k.w1 * e1,
            GAUSS_DOC,
        );
        assert_pinned(
            "w1 * g1 at d=0.5 - half a width",
            at(&k, 0.25)[F::Gauss1.index()],
            k.w1 * e4,
            GAUSS_DOC,
        );
        assert_pinned(
            "w2 * g2 at d=0.0 (its own centre)",
            at(&k, 0.0)[F::Gauss2.index()],
            k.w2,
            GAUSS_DOC,
        );
        assert_pinned(
            "w2 * g2 at d=0.5 (one width out)",
            at(&k, 0.5)[F::Gauss2.index()],
            k.w2 * e1,
            GAUSS_DOC,
        );
        assert_pinned(
            "w2 * g2 at d=-0.5 (one width out)",
            at(&k, -0.5)[F::Gauss2.index()],
            k.w2 * e1,
            GAUSS_DOC,
        );
        // The two centres are 0.5 A apart, so `g1` at 0.5 is `g2` at 0.0 scaled
        // only by their weights. If either centre moved apart from the other
        // these two stop being equal and the assertion goes red.
        let r1 = at(&k, 0.5)[F::Gauss1.index()] / k.w1;
        let r2 = at(&k, 0.0)[F::Gauss2.index()] / k.w2;
        assert_pinned(
            "g1(c1) / w1 against g2(c2) / w2, which the table makes equal",
            r1,
            r2,
            GAUSS_DOC,
        );
    }

    /// The Vina weights, quoted from §3.
    ///
    /// The first three matter most: they are the terms that are always present,
    /// so a change to any of them moves every single score in the program and
    /// would be obvious. `hbond` and `hydrophobic` are the two that are
    /// conditional on the pair class, and `hydrophobic` is the one the
    /// `1.5 -> 1.45` window mutation moved without moving any total.
    #[test]
    fn the_vina_weights_are_the_documented_ones() {
        let w = VinaWeights::default();
        assert_pinned("VinaWeights::gauss1", w.gauss1, -0.035_579, VINA_WEIGHT_DOC);
        assert_pinned("VinaWeights::gauss2", w.gauss2, -0.005_156, VINA_WEIGHT_DOC);
        assert_pinned(
            "VinaWeights::repulsion",
            w.repulsion,
            0.840_245,
            VINA_WEIGHT_DOC,
        );
        assert_pinned("VinaWeights::hbond", w.hbond, -0.587_439, VINA_WEIGHT_DOC);
        assert_pinned(
            "VinaWeights::hydrophobic",
            w.hydrophobic,
            -0.035_069,
            VINA_WEIGHT_DOC,
        );
        assert_pinned(
            "VinaWeights::intramolecular_scale",
            w.intramolecular_scale,
            0.006,
            VINA_WEIGHT_DOC,
        );
    }

    /// The Vinardo weights, quoted from §3. The `gpu` feature does not touch
    /// these, so this assertion is identical in both feature configurations.
    #[test]
    fn the_vinardo_weights_are_the_documented_ones() {
        let w = VinaWeights::vinardo();
        assert_pinned("Vinardo gauss1", w.gauss1, 0.0, VINARDO_WEIGHT_DOC);
        assert_pinned("Vinardo gauss2", w.gauss2, 0.0, VINARDO_WEIGHT_DOC);
        assert_pinned("Vinardo repulsion", w.repulsion, -0.045, VINARDO_WEIGHT_DOC);
        assert_pinned("Vinardo hbond", w.hbond, -0.030, VINARDO_WEIGHT_DOC);
        assert_pinned(
            "Vinardo hydrophobic",
            w.hydrophobic,
            -0.015,
            VINARDO_WEIGHT_DOC,
        );
        assert_pinned(
            "Vinardo intramolecular_scale",
            w.intramolecular_scale,
            0.0075,
            VINARDO_WEIGHT_DOC,
        );
    }

    /// `STERIC_LIMIT` — the `d < 0` branch of §2.2. Pinned as a constant *and*
    /// behaviourally, because the two can drift: the constant can be 0.0 while
    /// the comparison underneath it is `d <= 0.0`, which changes nothing at
    /// contact but is not the documented statement.
    #[test]
    fn the_steric_limit_is_contact() {
        assert_pinned("STERIC_LIMIT", STERIC_LIMIT, 0.0, REPULSION_DOC);
        assert_pinned(
            "rep at d=0 (the contact itself)",
            repulsion_term(0.0).0,
            0.0,
            REPULSION_DOC,
        );
        assert_pinned(
            "rep at d=0.001 (just outside)",
            repulsion_term(0.001).0,
            0.0,
            REPULSION_DOC,
        );
        assert_pinned(
            "rep at d=-0.001 (just inside)",
            repulsion_term(-0.001).0,
            1e-6,
            REPULSION_DOC,
        );
        // The derivative branch has the same boundary, and the sign convention
        // matters: below contact the gradient is *negative*, which is what pulls
        // an overlapping pose apart.
        assert_pinned(
            "drep/dd at d=-0.001",
            repulsion_term(-0.001).1,
            -0.002,
            REPULSION_DOC,
        );
        assert_pinned(
            "drep/dd at d=0.0",
            repulsion_term(0.0).1,
            0.0,
            REPULSION_DOC,
        );
    }

    /// `VINA_CUTOFF`, the surface-distance limit, quoted from §5.3.
    ///
    /// The reach this implies is checked at the grid level in `grid.rs`, where
    /// the real distance is the thing the inner loop actually holds — that is
    /// the half of §5.3 that a constant here cannot see.
    #[test]
    fn the_cutoff_is_eight_angstroms_of_surface_distance() {
        assert_pinned("VINA_CUTOFF", VINA_CUTOFF, 8.0, CUTOFF_DOC);
        assert_pinned(
            "SpatialKernels::cutoff",
            SpatialKernels::from_weights(&VinaWeights::default()).cutoff,
            8.0,
            CUTOFF_DOC,
        );
        assert_pinned(
            "SpatialKernels::cutoff (vinardo)",
            SpatialKernels::from_weights(&VinaWeights::vinardo()).cutoff,
            8.0,
            CUTOFF_DOC,
        );
        // The documented consequence: every term is spent well before the
        // cutoff, so truncating *at* the cutoff rather than a little beyond it
        // costs nothing. This is what makes 8.0 a bound rather than a tuned
        // number, and it is why a change to it is not silently harmless.
        let k = SpatialKernels::from_weights(&VinaWeights::default());
        let (v, _) = k.eval_terms(VINA_CUTOFF);
        let total: f64 = v.iter().sum();
        assert!(
            total.abs() < 1e-90,
            "every term must already be spent at the cutoff, or the cutoff is \
             truncating something; the sum at d = {VINA_CUTOFF} is {total:e}.\n\
             documented: {CUTOFF_DOC}"
        );
    }

    /// The intramolecular slope and the graph distance it is applied over,
    /// quoted from §4.1. `MIN_INTRA_BOND_DISTANCE` lives in `ligand.rs`; it is
    /// asserted from here because it is a scoring constant — it decides which
    /// pairs the *whole scoring function* is summed over — and because a test
    /// that pins it has to sit somewhere that will be run.
    #[test]
    fn the_intramolecular_slope_and_graph_distance_are_the_documented_ones() {
        assert_pinned(
            "Vina intramolecular_scale",
            VinaScoring::new().intramolecular_scale(),
            0.006,
            INTRA_DOC,
        );
        assert_pinned(
            "Vinardo intramolecular_scale",
            VinaScoring::vinardo().intramolecular_scale(),
            0.0075,
            INTRA_DOC,
        );
        assert_pinned(
            "MIN_INTRA_BOND_DISTANCE (ligand.rs)",
            MIN_INTRA_BOND_DISTANCE,
            4,
            INTRA_DOC,
        );
    }

    /// A weight is multiplied in **exactly one place**, and it is the map
    /// tabulation — not the scoring call.
    ///
    /// This is the correction to an earlier claim in this file that weights are
    /// applied at score time. Pinned as numbers, because the correction is a
    /// statement about where a multiplication happens and such a statement is
    /// only worth anything if a second multiplication turns the suite red.
    ///
    /// `eval_terms` is the single place, and both consumers — the four-slot
    /// `eval` and the six-field per-term path — read from it. A caller who
    /// multiplied by `VinaWeights::hbond` again would double-count it, and
    /// since the term is reported *already weighted*, that error is invisible
    /// in the total it is supposed to explain.
    #[test]
    fn a_weight_is_applied_once_and_at_tabulation_time() {
        let k = SpatialKernels::from_weights(&VinaWeights::default());
        for &d in &[-0.8, -0.4, -0.1, 0.2, 0.6, 1.0, 1.4, 2.0, 3.0] {
            let (v, dv) = k.eval_terms(d);
            let c = k.eval(d);
            assert_pinned(
                &format!("g1 at d={d}: eval_terms against eval().shape"),
                v[F::Gauss1.index()] + v[F::Gauss2.index()] + v[F::Repulsion.index()],
                c.shape.0,
                VINA_WEIGHT_DOC,
            );
            // ... and the two conditional ones. `eval` takes only the donor
            // half of the hydrogen bond, which is a slot decision made at
            // interpolation, not a second weight.
            assert_pinned(
                &format!("hb at d={d}: eval_terms donor half against eval().hbond"),
                v[F::HbFromDonor.index()],
                c.hbond.0,
                VINA_WEIGHT_DOC,
            );
            assert_pinned(
                &format!("hb at d={d}: eval_terms acceptor half"),
                v[F::HbFromDonor.index()],
                v[F::HbFromAcceptor.index()],
                WINDOW_DOC,
            );
            assert_pinned(
                &format!("hyd at d={d}: eval_terms against eval().hydrophobic"),
                v[F::Hydrophobic.index()],
                c.hydrophobic.0,
                VINA_WEIGHT_DOC,
            );
            // The derivatives, which the search's analytic gradient depends on.
            // Same call, second tuple: `eval` is built from `eval_terms`, so if
            // the two ever diverged the value assertions above would already
            // have gone red and these are the derivative counterpart.
            assert_pinned(
                &format!("shape derivative at d={d}"),
                dv[F::Gauss1.index()] + dv[F::Gauss2.index()] + dv[F::Repulsion.index()],
                c.shape.1,
                VINA_WEIGHT_DOC,
            );
            assert_pinned(
                &format!("hyd derivative at d={d}"),
                dv[F::Hydrophobic.index()],
                c.hydrophobic.1,
                VINA_WEIGHT_DOC,
            );
        }
    }

    /// `TERM_FIELDS` is six, and the six are the five terms plus one split.
    ///
    /// **This constant has no line in `SCORING.md`.** The specification was
    /// written before the per-term tabulation existed and documents four map
    /// slots in §5.1, which is a different decomposition. So the pin below is
    /// against the module's own documented claim, and the failure message says
    /// so — a scoring constant with no specification is a finding, and the
    /// reader should be handed it rather than a plausible-looking number.
    #[test]
    fn the_term_field_count_is_six_and_named() {
        const DOC: &str =
            "scoring.rs: 'Number of tabulated fields per grid point needed to rebuild \
the five terms. Six, not five. The hydrogen bond is one term but two fields' \
-- and SCORING.md §5.1 documents FOUR map slots, which is the per-slot \
decomposition and not this one. SCORING.md has no line for TERM_FIELDS.";
        assert_pinned("TERM_FIELDS", TERM_FIELDS, 6, DOC);
        assert_pinned("TermField::ALL length", F::ALL.len(), TERM_FIELDS, DOC);
        // Six distinct dense indices, in storage order, with the hydrogen bond's
        // two halves adjacent. The order is the storage layout of every
        // `TermValues` in the crate, so it is a constant in its own right.
        let idx: Vec<usize> = F::ALL.iter().map(|f| f.index()).collect();
        assert_pinned("TermField dense indices", idx, vec![0, 1, 2, 3, 4, 5], DOC);
        assert_pinned("HbFromDonor index", F::HbFromDonor.index(), 3, DOC);
        assert_pinned("HbFromAcceptor index", F::HbFromAcceptor.index(), 4, DOC);
        assert_pinned("Hydrophobic index", F::Hydrophobic.index(), 5, DOC);
    }

    /// The class masks are a **role** partition, and neither direction is a
    /// bijection: four map slots are not four terms, and five terms are not
    /// five slots.
    ///
    /// Pinned because the earlier framing — "the four slots correspond to the
    /// terms" — is wrong in both directions and reads convincingly. Slot 0 is
    /// three fused terms; `hb` is one term across two slots.
    #[test]
    fn the_masks_are_a_role_partition_not_a_term_correspondence() {
        use crate::types::{Atom, AtomKind, AtomType};
        let mk = |kind| {
            let mut a = Atom::new(1, [0.0, 0.0, 0.0], Element::O, AtomType::OA);
            a.kind = kind;
            a
        };
        // A hydrophobic probe reads one of the six fields: the apolar one.
        let hyd = probe_term_mask(&mk(AtomKind::Hydrophobic));
        assert_eq!(
            hyd.iter().filter(|b| **b).count(),
            4,
            "shape terms + apolar"
        );
        assert!(hyd[F::Hydrophobic.index()]);
        // A donor-acceptor reads five of the six: both hydrogen-bond halves,
        // because both were written by some receptor atom, and the three shape
        // terms, which every atom writes. It does **not** read the apolar field,
        // because `is_apolar` is false for an oxygen. Five is not six — an
        // atom's class is a role, and a polar atom has no apolar role.
        let both = probe_term_mask(&mk(AtomKind::DonorAcceptor));
        assert_eq!(
            both.iter().filter(|b| **b).count(),
            5,
            "not the apolar field"
        );
        assert!(!both[F::Hydrophobic.index()]);
        assert!(both[F::HbFromDonor.index()] && both[F::HbFromAcceptor.index()]);
        // A donor reads the acceptor-written half and *not* the donor-written
        // one. This crossing is the whole of its selectivity; backwards would
        // hand a receptor acceptor its hydrogen bond back to a receptor acceptor.
        let d = probe_term_mask(&mk(AtomKind::Donor));
        assert!(d[F::HbFromAcceptor.index()]);
        assert!(!d[F::HbFromDonor.index()]);
        // The receptor side is the mirror image, and the two are crossed
        // against each other rather than aligned.
        let rd = receptor_term_mask(&mk(AtomKind::Donor));
        assert!(rd[F::HbFromDonor.index()]);
        assert!(!rd[F::HbFromAcceptor.index()]);
        // Four roles, and no one-to-one: the four map slots carry
        // three-plus-one-plus-one-plus-one, and the six fields carry three
        // shape terms plus a split hydrogen bond plus one apolar term.
        assert_eq!(crate::grid::MAPS_PER_TYPE, 4);
        assert_ne!(TERM_FIELDS, crate::grid::MAPS_PER_TYPE);
    }
}
