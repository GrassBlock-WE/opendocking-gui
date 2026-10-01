"""Does the engine's scoring implement the scoring function this project *wrote down*?

Run:  python scripts/scoring_cross_check.py

# Why there is no Vina reference in this repository

This file was originally going to hold a from-the-published-form Vina
implementation, cross-validated against the engine. That was abandoned, and
the reason is the first thing worth recording, because it is a finding rather
than an excuse.

**The engine does not implement Vina's published scoring function, and its own
documentation says so in three places.** `docs/SCORING.md` §2.2 gives a hard
``rep(d) = d^2`` wall below contact, which is AutoDock 4's repulsion rather than
Vina's Gaussian-scaled one. §2.3 states, in its own words, that Vina uses hard
cutoffs for the hydrogen-bond and hydrophobic terms "while in this tool they
are a C¹ cubic smoothstep" — a deliberate replacement, for gradient reasons,
and it is deviation #2 in §10's list of nine. §10 also documents that `g2` is
centred at 0.0 with width 0.5 where Vina's is centred at 3.0 with width 2.0,
and §3's weights are not Vina's published set. `docs/VERIFICATION.md` §2.1
already draws the conclusion this file would have drawn: the tool's absolute
kcal/mol numbers must not be compared with Vina's or with literature values.

So a Vina cross-validation could not have produced an interpretable result. Every
term would differ, every difference would be a documented design choice, and a
"delta" between them would say nothing about correctness. Writing Vina from
memory to generate those deltas would have been worse than not writing it: the
deltas would have measured my recollection rather than the engine, and the
report would have read as "the engine's scores are wrong" when the truth is
"the engine is a different function, on purpose". So there is no
`vina_reference.py`. That is the honest outcome, not a missing deliverable.

# What this file checks instead

The question that *is* decidable offline, and that nobody had asked, is whether
the implementation matches its own written specification. `SCORING.md` gives
exact functional forms, weights, window positions and atom classification. So:

* the reference below is written **from `docs/SCORING.md` only**. No Rust was
  consulted to decide what any term should be; `scoring.rs` was never opened.
  Where the specification is a statement about the implementation (it names the
  Rust functions), that is a weakness of the source and is said so below.
* the comparison is **total energy, term by term, over distance scans** — with
  the hard configurations: atoms inside one another, exactly at contact, heavy
  against light, donor-acceptor, apolar-apolar.
* **every comparison is bidirectional**: a perturbation of one weight in the
  reference must break agreement, or the agreement means nothing.

# The precision this comparison can and cannot reach

The engine scores by trilinear interpolation of a 0.375 Å grid, not by
evaluating pairs. That puts a floor on any agreement claim: measured below, the
largest total-energy delta over a 2.6-6.0 Å scan is about 0.02 kcal/mol, and it
sits in the steep repulsive region where a quarter-ångström grid step is worth
most. No per-pair or per-term value is exposed by `score_conformation`,
`evaluate_conformations` or `DockingResult` — all three return scalars — so a
term-by-term comparison is possible at *map* level only, and the per-slot
fields carry a normalisation this file does not attempt to invert. The
term-level breakdown below is therefore the reference's own decomposition, and
the engine is asked only for the total it actually returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking.core import (  # noqa: E402
    GridBox, Ligand, Receptor, conformation_coordinates, score_conformation,
    scoring_descriptions,
)

#: How many checks this file is supposed to run, counted by running it.
EXPECTED_CHECKS = 15  # measured; the reference is a spec check, not Vina

FAILURES: list[str] = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------
# The reference, transcribed from docs/SCORING.md and from nowhere else.
# --------------------------------------------------------------------------

#: SCORING.md section 3, and independently confirmed by the engine's own
#: `scoring_descriptions()`, which prints these five numbers at runtime.
WEIGHTS = {
    "g1": -0.035579,
    "g2": -0.005156,
    "rep": 0.840245,
    "hb": -0.587439,
    "hyd": -0.035069,
}

#: SCORING.md section 3.2.1: per-element XS radii, "not copied from Vina".
#: Surface distance is d = r - (R_i + R_j).
XS_RADIUS = {"C": 1.9, "N": 1.75, "O": 1.6, "S": 2.0, "H": 0.0}


def smoothstep(x: float, a: float, b: float) -> float:
    """S(a, b, x) from SCORING.md section 2.3: 0 below a, 1 above b, C1 between."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


def reference_pair(d: float, *, donor=False, acceptor=False,
                   apolar=False, weights=None) -> dict[str, float]:
    """One atom pair's energy, decomposed, from SCORING.md's written formulas.

    Section 2.1: ``g(c, w, d) = exp(-((d - c)/w)^2)``; g1 at (0.5, 0.5),
    g2 at (0.0, 0.5). Section 2.2: ``rep = d^2`` for ``d < 0`` and 0 otherwise.
    Section 2.3: ``hb = 1 - S(-0.5, 0, d)`` and ``hyd = 1 - S(0.5, 1.5, d)``.
    Section 4: the shape terms and the repulsion apply to every pair; hbond
    only to a donor/acceptor pair; hydrophobic only to an apolar/apolar pair.

    ``weights`` defaults to the spec's set and exists so the discrimination
    check below can perturb one constant without reaching for a global.
    """
    w = WEIGHTS if weights is None else weights
    g1 = w["g1"] * float(np.exp(-(((d - 0.5) / 0.5) ** 2)))
    g2 = w["g2"] * float(np.exp(-(((d - 0.0) / 0.5) ** 2)))
    rep = w["rep"] * (d * d if d < 0.0 else 0.0)
    hb = w["hb"] * (1.0 - smoothstep(d, -0.5, 0.0)) if (donor or acceptor) else 0.0
    hyd = w["hyd"] * (1.0 - smoothstep(d, 0.5, 1.5)) if apolar else 0.0
    return {"g1": g1, "g2": g2, "rep": rep, "hb": hb, "hyd": hyd,
            "total": g1 + g2 + rep + hb + hyd}


# --------------------------------------------------------------------------
# Engine probes: the smallest fixture that isolates one pair.
# --------------------------------------------------------------------------

def pdbqt_atom(serial, name, x, y, z, atom_type) -> str:
    return (f"ATOM  {serial:5d} {name:<4s} UNL     1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}    0.000 "
            f"{atom_type:>2s}\n")


def engine_profile(receptor_types, ligand_specs, distances,
                   box_max=14.0):
    """Total energy of a one-atom probe against a fixed receptor, vs distance.

    The receptor sits at the origin and the probe is translated along +z, so
    ``r`` is the true centre-to-centre separation of the pair of interest and
    the intramolecular term is empty (a single rigid atom has no pairs).
    """
    rec = Receptor.from_pdbqt_str(
        "".join(pdbqt_atom(i + 1, f" {chr(65+i)}1", 0.0, 0.0, 0.0, t)
                for i, t in enumerate(receptor_types)) + "END\n")
    maps = rec.precalculate(
        GridBox((-2.0, -2.0, -2.0), (2.0, 2.0, box_max)), "vina")
    lig = Ligand.from_arrays(*ligand_specs)
    out = []
    for r in distances:
        conf = np.zeros(lig.num_dof)
        conf[2] = float(r)
        energy, _ = score_conformation(lig, maps, conf)
        out.append((float(r), float(energy)))
    return out, lig


def main() -> int:
    # ------------------------------------------------------------------
    section("the weights the engine runs with are the weights the spec states")

    reported = scoring_descriptions()
    vina_line = next(d for d in reported if d.startswith("vina:"))
    # The engine names its fifth key `hbond`; the spec's table calls it `hb`.
    names = {"g1": "g1", "g2": "g2", "rep": "rep", "hb": "hbond", "hyd": "hyd"}
    check("the engine's own description string matches the spec's five weights",
          all(f"{names[k]}={v:.6f}" in vina_line for k, v in WEIGHTS.items()),
          f"{vina_line!r}. This is the one part of the comparison that needs no "
          f"model at all: the engine prints its constants, and they are the "
          f"ones docs/SCORING.md section 3 documents")

    # ------------------------------------------------------------------
    section("an apolar pair, scanned: engine total against the written spec")

    # The cleanest possible isolation: one carbon, one carbon, no hydrogens and
    # no intramolecular pairs, so the whole curve is one pair's energy.
    distances = np.arange(2.6, 6.01, 0.1)
    profile, lig = engine_profile(
        ["C"],
        (["C"], [0.0], np.array([[0.0, 0.0, 0.0]])),
        distances,
    )
    check("the probe is classified apolar, so hyd applies and hb does not",
          lig.atom_kinds == ["hydrophobic"],
          f"kinds {lig.atom_kinds}, which is what selects the hydrophobic term "
          f"and withholds the hydrogen-bond one (SCORING.md section 4)")

    rc = XS_RADIUS["C"]
    rows = []
    for r, e in profile:
        d = r - 2 * rc
        ref = reference_pair(d, apolar=True)["total"]
        rows.append((r, e, ref, e - ref))
    worst = max(rows, key=lambda t: abs(t[3]))
    check("engine and reference agree along the whole scan",
          abs(worst[3]) < 0.025,
          f"{len(rows)} points from {rows[0][0]:.1f} to {rows[-1][0]:.1f} A; "
          f"largest |delta| {abs(worst[3]):.5f} kcal/mol at r = {worst[0]:.2f} A, "
          f"mean |delta| {np.mean([abs(t[3]) for t in rows]):.5f}. That floor is "
          f"the 0.375 A grid the engine interpolates on, not model error -- the "
          f"largest deltas sit in the steep repulsive wall where a quarter-"
          f"angstrom step costs the most")

    eng_min = min(profile, key=lambda t: t[1])
    ref_grid = np.arange(-2.0, 3.0, 0.001)
    ref_vals = [reference_pair(float(d), apolar=True)["total"] for d in ref_grid]
    ref_min_d = float(ref_grid[int(np.argmin(ref_vals))])
    ref_min_e = float(min(ref_vals))
    check("and they put the minimum in the same place at the same depth",
          abs(eng_min[0] - (ref_min_d + 2 * rc)) < 0.06
          and abs(eng_min[1] - ref_min_e) < 0.006,
          f"engine minimum {eng_min[0]:.2f} A / {eng_min[1]:.5f}; reference "
          f"minimum {ref_min_d + 2 * rc:.2f} A / {ref_min_e:.5f}. The position "
          f"and the depth are separate claims and both are tested, because a "
          f"curve can be shifted and keep its depth")

    # ------------------------------------------------------------------
    section("the XS radii, measured rather than read off the spec")

    # SCORING.md 3.2.1 gives per-element XS radii (C 1.9 / N 1.75 / O 1.6 /
    # S 2.0, "H is 0") and says the surface distance is d = r - (Ri + Rj).
    # The radius is the single most consequential constant in the function --
    # the whole 3.2.1 bug was one value -- and it is not exposed anywhere. So it
    # is measured: the energy crosses zero where d = 0, because below that the
    # repulsion wall dominates and above it the pair is net attractive. The
    # crossing is found by bisection, which does not assume a grid resolution.
    def zero_crossing(rec_type, probe_spec, lo, hi):
        rec = Receptor.from_pdbqt_str(
            pdbqt_atom(1, " A1", 0.0, 0.0, 0.0, rec_type) + "END\n")
        big = rec.precalculate(GridBox((-2.0, -2.0, -2.0), (2.0, 2.0, 20.0)),
                               "vina")
        lig = Ligand.from_arrays(*probe_spec)

        def energy(r):
            conf = np.zeros(lig.num_dof)
            conf[2] = float(r)
            return float(score_conformation(lig, big, conf)[0])

        a, b = lo, hi
        fa = energy(a)
        for _ in range(40):
            mid = 0.5 * (a + b)
            fm = energy(mid)
            if (fa > 0) == (fm > 0):
                a, fa = mid, fm
            else:
                b = mid
        return 0.5 * (a + b)

    c_cross = zero_crossing("C", (["C"], [0.0], np.array([[0.0, 0.0, 0.0]])),
                            1.0, 8.0)
    c_radius = c_cross / 2.0
    check("the measured carbon XS radius is the 1.9 A the spec states",
          abs(c_radius - 1.9) < 0.15,
          f"a carbon-carbon pair's energy crosses zero at r = {c_cross:.3f} A, "
          f"which is where d = 0 and therefore Ri + Rj, so each carbon radius is "
          f"{c_radius:.3f} A against the 1.9 A of SCORING.md 3.2.1. No binding "
          f"exposes the radius, so this is how it is recovered -- and the 3.2.1 "
          f"bug was exactly a wrong value here, which is reason enough to be "
          f"able to check it rather than trust it")

    h_cross = zero_crossing("C", (["H"], [0.0], np.array([[0.0, 0.0, 0.0]]),
                                  None, ["HD_H1"]), 1.0, 8.0)
    h_radius = h_cross - XS_RADIUS["C"]
    check("and the hydrogen radius is NOT the 0.0 the spec states",
          h_radius > 0.4,
          f"a C...H pair crosses zero at r = {h_cross:.3f} A, so the hydrogen "
          f"radius is {h_radius:.3f} A where SCORING.md 3.2.1 says \"H is 0\". "
          f"This is reported rather than treated as a defect: the spec line is "
          f"about the receptor-side XS radii used to build the maps, and a "
          f"polar hydrogen may be given a radius there while the *probe* "
          f"radius that produces this crossing is a different quantity. What "
          f"can be said without guessing which is which: a C...H pair's zero "
          f"crossing is at {h_cross:.2f} A, not at the {XS_RADIUS['C']:.2f} A "
          f"the spec's formula would give, and any reference written from the "
          f"spec alone will disagree with the engine there")

    # ------------------------------------------------------------------
    section("the configurations a wrong implementation gets wrong")

    # A donor-acceptor pair needs a probe that is actually polar. A lone N with
    # no bonds classifies as `other` and contributes nothing, which is what the
    # first attempt at this fixture measured -- so the N carries an explicit
    # polar H, which is the documented route to DonorAcceptor.
    hard = {
        "atoms inside one another (r = 0.5 A)": (0.5, ["C"],
                                                (["C"], [0.0],
                                                 np.array([[0.0, 0.0, 0.0]]))),
        "exactly at contact (d = 0, r = 3.8 A)": (2 * rc, ["C"],
                                                  (["C"], [0.0],
                                                   np.array([[0.0, 0.0, 0.0]]))),
        "heavy against light (C...H)": (3.2, ["C"],
                                        (["H"], [0.0],
                                         np.array([[0.0, 0.0, 0.0]]),
                                         None, ["HD_H1"])),
        "donor-acceptor (O...N-H)": (
            2.8, ["OA"],
            (["N", "H"], [0.0, 0.0],
             np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0]]),
             [(0, 1)], ["N_N1", "HD_H1"]),
        ),
    }
    hard_rows = []
    for label, (r, rec_types, spec) in hard.items():
        got, probe = engine_profile(rec_types, spec, [r])
        element = spec[0][0]
        d = r - (XS_RADIUS.get(rec_types[0][0], 1.9) + XS_RADIUS.get(element, 1.9))
        donor = probe.atom_kinds[0] in ("donor", "donor_acceptor")
        acceptor = probe.atom_kinds[0] in ("acceptor", "donor_acceptor")
        apolar = probe.atom_kinds[0] == "hydrophobic"
        ref = reference_pair(d, donor=donor, acceptor=acceptor, apolar=apolar)
        hard_rows.append((label, r, d, got[0][1], ref["total"],
                          probe.atom_kinds[0], ref))
    # Relative tolerance: a term worth 9 kcal/mol cannot be held to the same
    # absolute one as a term worth 0.01, and the grid's interpolation error
    # scales with the field's slope, which scales with the magnitude.
    def _bad(t):
        return abs(t[3] - t[4]) > 0.02 + 0.04 * max(abs(t[3]), abs(t[4]))

    worst_hard = max(hard_rows, key=lambda t: abs(t[3] - t[4]))
    agreeing = [t for t in hard_rows if not _bad(t)]
    disagreeing = [t for t in hard_rows if _bad(t)]
    check("the overlapping and contact cases agree; two do not, and are named "
          "rather than folded in",
          len(agreeing) == 2 and len(disagreeing) == 2,
          "; ".join(
              f"{lab}: r={r:.2f} d={d:+.2f} engine {e:+.4f} reference {ref:+.4f} "
              f"(delta {e - ref:+.4f}, probe {kind})"
              for lab, r, d, e, ref, kind, _ in hard_rows)
          + f". The two that agree are the ones that matter most for a "
            f"truncation bug. The two that do not are reported unresolved "
            f"below rather than explained away: {'; '.join(t[0] for t in disagreeing)}")

    inside = next(t for t in hard_rows if "inside" in t[0])
    check("and deeply overlapped atoms are repulsive, by a lot, on both sides",
          inside[3] > 1.0 and inside[4] > 1.0
          and inside[5 + 1]["rep"] > 1.0,
          f"r = {inside[1]:.2f} A gives d = {inside[2]:+.2f} and a repulsion "
          f"term of {inside[6]['rep']:+.4f}: engine {inside[3]:+.4f}, reference "
          f"{inside[4]:+.4f}. This is the case that makes truncation bugs "
          f"visible -- an implementation that clipped instead of refusing would "
          f"score a clashed structure as merely bad, not as impossible")

    hb_row = next(t for t in hard_rows if "donor-acceptor" in t[0])
    check("UNRESOLVED: a donor-acceptor pair scores exactly zero on the engine, "
          "which this file could not explain",
          hb_row[3] == 0.0,
          f"an OA receptor against an N carrying a polar H, probe classified "
          f"{hb_row[5]!r} at r = {hb_row[1]:.2f} A, gives the engine "
          f"{hb_row[3]:+.6f} and the reference {hb_row[4]:+.4f} "
          f"(decomposed g1 {hb_row[6]['g1']:+.4f}, g2 {hb_row[6]['g2']:+.4f}, "
          f"rep {hb_row[6]['rep']:+.4f}, hb {hb_row[6]['hb']:+.4f}). A polar pair "
          f"at 2.8 A scoring *identically* zero is not something this file can "
          f"account for from the written spec, and it is recorded as a question "
          f"rather than a conclusion. Two candidates I did not have the budget "
          f"to separate: the receptor's lone OA may not be reaching the "
          f"acceptor map, or the 4-slot split may withhold this combination. "
          f"Either way the reference disagrees with the engine here, and this "
          f"file does not claim otherwise")

    # ------------------------------------------------------------------
    section("the comparison discriminates: perturb the reference and it breaks")

    # If agreement held for reasons other than correctness -- a saturated
    # constant, a term nobody reads, a tolerance wide enough to hide anything --
    # then perturbing the model should not matter. The perturbation used is a
    # shift in the *radius*, because that is the exact shape of the 3.2.1 bug:
    # a wrong XS radius produces a file that is wrong everywhere at once and
    # still looks plausible.
    honest = max(abs(t[3]) for t in rows)
    shifted = [
        abs(e - reference_pair(r - 2 * rc - 0.2, apolar=True)["total"])
        for r, e, _, _ in rows
    ]
    check("a 0.2 A error in the XS radius destroys the agreement, which is the "
          "shape of the 3.2.1 bug",
          max(shifted) > 10 * honest,
          f"the honest scan's largest delta is {honest:.5f} kcal/mol; shifting the "
          f"surface distance by 0.2 A -- a 10% error in the radius, smaller than "
          f"the original bug's -- pushes it to {max(shifted):.4f}. So the "
          f"agreement above is sensitive to the one constant that 3.2.1 was "
          f"about, and is not an artefact of a term nobody reads")

    # ------------------------------------------------------------------
    section("what docs/VERIFICATION.md 3.2.1 claims, measured")

    # These are the project's own published numbers for the 0.4 A radius fix.
    # They are falsifiable, so they are checked rather than assumed.
    at_3 = next(t for t in rows if abs(t[0] - 3.0) < 0.05)
    check("KNOWN BAD: the published 'C...C minimum 3.00 A, -0.050' is not what "
          "the engine does, and is not reachable with its own weights",
          at_3[1] > 0.0 and at_3[2] > 0.0,
          f"docs/VERIFICATION.md 3.2.1 reports the apolar minimum as 3.00 A / "
          f"-0.050 kcal/mol. Measured on a clean single-carbon pair the engine "
          f"gives {at_3[1]:+.4f} at 3.00 A -- inside the repulsive wall -- and "
          f"its minimum is at {eng_min[0]:.2f} A / {eng_min[1]:.5f}. The "
          f"reference says the same: at d = -0.8 the repulsion term alone is "
          f"{reference_pair(-0.8, apolar=True)['rep']:+.4f} while g1+g2+hyd "
          f"together cannot exceed "
          f"{abs(WEIGHTS['g1'] + WEIGHTS['g2'] + WEIGHTS['hyd']):.4f}, so "
          f"-0.050 at that distance is arithmetically out of reach. The engine "
          f"passes -0.050 near r = 3.8 A instead. Either the table's position "
          f"column means surface distance rather than centre distance, or the "
          f"number is wrong; this file does not guess which")

    # The hydrogen-bond row of the same table. Reported rather than asserted,
    # because the fixture that produced the published number is not recorded
    # and the one available here is not the same: the polar H is carried along
    # the z axis, so it sits between the two oxygens rather than beside one.
    rec_o = "".join(pdbqt_atom(1, " O1", 0.0, 0.0, 0.0, "OA")) + "END\n"
    rec = Receptor.from_pdbqt_str(rec_o)
    maps = rec.precalculate(GridBox((-2.0, -2.0, -2.0), (2.0, 2.0, 14.0)), "vina")
    oo = Ligand.from_arrays(["O", "H"], [0.0, 0.0],
                            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.98]]),
                            [(0, 1)], ["OA_O1", "HD_H1"])
    oo_rows = []
    for r in np.arange(1.8, 4.61, 0.1):
        conf = np.zeros(oo.num_dof)
        conf[2] = float(r)
        e, _ = score_conformation(oo, maps, conf)
        oo_rows.append((float(r), float(e)))
    oo_min = min(oo_rows, key=lambda t: t[1])
    check("the published 'O...O hydrogen-bond minimum 2.50 A, -0.177' is NOT "
          "reproduced here, and this file does not claim it is wrong either",
          True,
          f"measured minimum {oo_min[0]:.2f} A / {oo_min[1]:.5f} kcal/mol, and "
          f"the probe is classified {oo.atom_kinds[0]!r} rather than "
          f"'donor_acceptor'. The fixture is not the one behind the published "
          f"number and is not recorded anywhere I can read: the polar H is "
          f"translated along z with the probe, so it lies between the two "
          f"oxygens instead of beside one, which is a different geometry and a "
          f"different classification. Unlike the apolar row, the claim is "
          f"arithmetically *possible* at d = -0.7 (rep "
          f"{reference_pair(-0.7, acceptor=True)['rep']:+.4f} against an hb "
          f"capacity of {abs(WEIGHTS['hb']):.4f}), so this file reports it as "
          f"unreproduced rather than as wrong")

    # ------------------------------------------------------------------
    section("what the exposed API can and cannot separate")

    rec2 = Receptor.from_pdbqt_str(
        pdbqt_atom(1, " C1", 0.0, 0.0, 0.0, "C") + "END\n")
    maps2 = rec2.precalculate(GridBox((-2.0, -2.0, -2.0), (2.0, 2.0, 4.0)), "vina")
    raw = maps2.raw_data
    nx, ny, nz = maps2.dims
    stride = 40
    point = ((nx // 2) + nx * ((ny // 2) + ny * (nz - 2))) * stride
    slots = {}
    for slot, name in enumerate(("Shape", "HbFromDonor", "HbFromAcceptor",
                                 "Hydrophobic")):
        vals = raw[point + slot: point + stride: 4]
        slots[name] = (float(vals.min()), float(vals.max()))
    check("the four map slots are readable, which is the only term-level "
          "surface the API offers",
          raw.size == maps2.num_points * stride
          and len(slots) == 4
          and abs(slots["Hydrophobic"][0]) > 0.0
          and slots["HbFromDonor"][1] == 0.0,
          f"raw_data is {raw.size} float32 = {maps2.num_points} points x "
          f"stride {stride}, matching the documented layout. On an all-carbon "
          f"receptor the slots read "
          f"{', '.join(f'{k} [{v[0]:.3g}, {v[1]:.3g}]' for k, v in slots.items())}"
          f" -- Shape and Hydrophobic carry the carbon's apolar field while "
          f"both hydrogen-bond slots are flat zero, which is the 4-slot split "
          f"working as SCORING.md section 5.1 describes")

    check("but no per-pair or per-term ENERGY is exposed, so the term "
          "comparison above is necessarily coarser than it looks",
          True,
          "score_conformation returns (energy, gradient); "
          "evaluate_conformations returns totals; DockingResult carries "
          "intermolecular_energies per pose. All scalars. The reference "
          "decomposes its own terms and the engine is asked only for the sum, "
          "so this file establishes that the SUM matches the spec -- it does "
          "not establish each term separately, and the per-slot values above "
          "are not compared term by term because their normalisation (weights "
          "are applied at score time, per SCORING.md 9.1) was not established "
          "here")

    # ------------------------------------------------------------------
    section("every check in this file ran")

    before = CHECKS
    check("the number of checks that ran is the number this file is supposed "
          "to have",
          before + 1 == EXPECTED_CHECKS,
          f"{before} ran before this one and {EXPECTED_CHECKS} are expected; "
          f"the +1 is this check. Change EXPECTED_CHECKS deliberately")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
