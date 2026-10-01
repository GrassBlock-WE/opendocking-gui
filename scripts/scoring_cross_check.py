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

**What saying that cost, and what was bought with it.** The score-time weight
normalisation of §9.1 was never inverted, so this file could not attribute a
single number to a single term. Everything the "UNRESOLVED" checks below
conclude is therefore a statement about the **total** and about **which map
components are non-zero** — never about a term's magnitude. The one thing
per-slot access *did* buy is the layout, and that turned out to be the whole
story of the 0.000000 anomaly: the four slots are not four terms shared across
all atom types, and knowing that is what separated the two candidates the
anomaly was reported with. The cost of not inverting the normalisation is
therefore: **if a term is present in a slot, this file can prove it is
populated; it cannot prove the weight applied to it is the one §3 states.** The
section "what the map actually contains" records which of the two is which.

# What the map actually contains, measured rather than assumed

An earlier draft of this file labelled components 0, 4, 8 and 12
"Shape / HbFromDonor / HbFromAcceptor / Hydrophobic" and took the labels from
§5.1. The labels are *mostly* right, but the framing around them was wrong in a
way that mattered, and the difference is the finding. `raw_data`'s own
docstring gives the layout as `type·4 + slot`: **ten type groups of four slots
each**, not four terms. Which group an atom lands in is decided by the atom,
and a single-atom receptor is enough to measure it:

| lone receptor atom | group | populated slots |
|---|---|---|
| `C`, `A`, `HD`   | 0 | 0, 3 |
| `N`              | 1 | 0 |
| `NA`             | 1 | 0, 2 |
| `OA`, `OS`       | 2 | 0, 2 |
| `P`              | 3 | 0 |
| `S`, `SA`        | 4 | 0 |
| `F`              | 5 | 0, 3 |
| `Cl`             | 6 | 0, 3 |

So the four slots do name four terms -- slot 0 is the steric/shape field, which
every atom populates; slot 3 is hydrophobic, which the apolar atoms populate;
slot 2 is populated by the acceptors (`NA`, `OA`); slot 1 is populated by the
donors (`ND`, `OD`) and by an explicit polar hydrogen. What the draft got
wrong was the word "withholds": the split is **not** a per-term filter applied
to one shared field. Each type group is a separate table, and **a probe atom
sums only the group that matches its own element** -- measured below, in a
cross-tab, and it is the reason a polar donor-acceptor pair scores exactly
zero.

# A finding this file withdrew, and why the withdrawal is the interesting part

An earlier revision of this file reported a *second* defect: that the
hydrogen-bond-**from-donor** map is tabulated and read by nothing. That was
**wrong, and it was wrong because of the `from_arrays` limitation documented in
the check below.** The evidence offered for it was that five oxygen receptors
-- `O`, `OA`, `OS` with the donor slot empty and `OD`, `ODA` with it filled at
0.587439 -- gave a **bitwise identical** score. That comparison is only valid if
the probe reads that slot. The probe was an oxygen built with
`Ligand.from_arrays(["O"], ..., ["OA_O1"])`, and `from_arrays` **cannot type an
acceptor**: it arrives as `other`, and an `other` atom reads the steric slot and
nothing else. Rebuilt through the file reader, where the same atom really is an
`acceptor`, the same five receptors give **two** different answers:
`O`/`OA`/`OS` at `+0.15853941490252765` and `OD` at `-0.2512580310304957`.

So the donor slot **is** read, by a probe of the same element group, and the
chemistry inside a group is right: a donor reads the receptor's *acceptor*
field, an acceptor reads the receptor's *donor* field, and like-with-like is
withheld. The single defect that remains is the element partition, and the two
findings collapsed into one.

The general lesson, recorded because it will recur: **a probe fixture's class
is a claim, not a fact, and `atom_kinds` is the only way to check it.** Two
checks below now assert the class of every fixture they use, precisely because
a mislabelled probe produced a false positive that survived a bitwise
comparison.
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

from opendocking import pdbqt_writer as W  # noqa: E402
from opendocking.core import (  # noqa: E402
    GridBox, Ligand, Receptor, conformation_coordinates, score_conformation,
    scoring_descriptions,
)

#: How many checks this file is supposed to run, counted by running it.
EXPECTED_CHECKS = 19  # measured; the reference is a spec check, not Vina

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


#: The map box used by every fixture below, and the distance at which the
#: cross-tab is read. Both are constants on purpose: the anomaly under
#: investigation is a claim about a *specific* geometry, and a check that let
#: the geometry float would be free to pass somewhere the anomaly does not hold.
PROBE_BOX = GridBox((-4.0, -4.0, -4.0), (4.0, 4.0, 10.0))
PROBE_AT = 2.8


def rec_from(specs):
    """A receptor from ``[(name, (x, y, z), type), ...]``."""
    return Receptor.from_pdbqt_str(
        "".join(pdbqt_atom(i + 1, f" {nm}", xyz[0], xyz[1], xyz[2], t)
                for i, (nm, xyz, t) in enumerate(specs)) + "END\n")


def populated(rec, box=PROBE_BOX):
    """``(groups, slots)`` of the map components this receptor makes non-zero.

    ``raw_data``'s layout is ``type*4 + slot``, so component *k* is group
    ``k // 4`` and slot ``k % 4``. Returns the sorted distinct values of each,
    which is enough to name the layout and short of pretending to know what
    the ten groups are *for*.
    """
    per = rec.precalculate(box, "vina").raw_data.reshape(-1, 40)
    live = [k for k in range(40) if float(np.abs(per[:, k]).max()) > 0.0]
    return sorted({k // 4 for k in live}), sorted({k % 4 for k in live})


def at(lig, rec, distance=PROBE_AT, box=PROBE_BOX):
    """The engine's total for `lig` translated to `distance` along +z."""
    maps = rec.precalculate(box, "vina")
    conf = np.zeros(lig.num_dof)
    conf[2] = float(distance)
    return float(score_conformation(lig, maps, conf)[0])


#: Probes that between them cover the groups the engine distinguishes.
#:
#: **The names carry PDBQT type prefixes that `from_arrays` ignores** --
#: measured and mutation-proven in the from_arrays check below. An oxygen
#: handed to `from_arrays` as ``["OA_O1"]`` arrives as type ``O``, so it is not
#: an acceptor. `PROBE_TYPE` records what the engine *actually* assigns, and it
#: is that, not the prefix, which the group rule is applied to. The last probe
#: is therefore built through the **file** reader, which has a type column, so
#: the table contains a genuine acceptor and not merely an oxygen.
PROBE_SPECS = {
    "C": (["C"], [0.0], np.array([[0.0, 0.0, 0.0]]), None, None),
    "N+H": (["N", "H"], [0.0, 0.0],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0]]), [(0, 1)],
            ["N_N1", "HD_H1"]),
    "N": (["N"], [0.0], np.array([[0.0, 0.0, 0.0]]), None, None),
    "O": (["O"], [0.0], np.array([[0.0, 0.0, 0.0]]), None, ["OA_O1"]),
}

#: The type token the engine actually assigns to each atom of each probe. The
#: plain element type for everything `from_arrays` builds.
PROBE_TYPE = {
    "C": ["C"], "N+H": ["N", "H"], "N": ["N"], "O": ["O"],
    "OA (via file)": ["OA"],
}


def from_file(atom_types: list[str]) -> Ligand:
    """A ligand built through the file reader, so its types are the real ones.

    The writer refuses a three-character type, which is correct for a two-column
    field and is why the `NDA`/`ODA`/`CG0` fixtures above are hand-built. Atoms
    are laid out along x at a bond-like spacing: the engine refuses coincident
    atoms, so a multi-atom fixture written at the origin would not build at all.
    """
    lines = [W.format_atom_line(i + 1, f" X{i + 1}", "UNL", 1,
                                (1.4 * i, 0.0, 0.0), 0.0, t)
             for i, t in enumerate(atom_types)]
    return Ligand.from_pdbqt_str("\n".join(lines) + "\nEND\n")


def hand_built(atom_type: str) -> str:
    """One ATOM line carrying `atom_type` at columns 78+, three characters wide.

    `format_atom_line` refuses a three-character type rather than truncate it,
    which is the right call and also means the writer cannot be used to build
    the `NDA`/`ODA`/`CG0` fixtures the map table needs.
    """
    base = W.format_atom_line(1, " X1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "C")
    return (base[:W._COL_TYPE[0]] + atom_type).rstrip()


def rec_of(atom_type: str) -> Receptor:
    """A one-atom receptor, built by hand when the type is three characters."""
    if len(atom_type) > 2:
        return Receptor.from_pdbqt_str(hand_built(atom_type) + "\nEND\n")
    return Receptor.from_pdbqt_str(
        W.format_atom_line(1, " X1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, atom_type)
        + "\nEND\n")


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
    def zero_crossing(rec_types, probe_spec, lo, hi, rec_specs=None):
        """Where a probe's energy against a receptor changes sign, by bisection.

        `rec_types` is one type or several, and `rec_specs` overrides it with
        explicit ``(name, (x, y, z), type)`` triples when the atoms need real
        separation rather than sharing the origin. Two properties of the
        bisection are worth keeping in mind when reading a result: it assumes
        the sign changes once, and **it cannot report "no crossing"** -- it
        returns whichever bound it was left holding. So a value equal to `hi`
        means the search never bracketed a crossing, and reporting it as a
        distance would be a fiction.
        """
        if rec_specs is None:
            if isinstance(rec_types, str):
                rec_types = [rec_types]
            rec_specs = [(f"{chr(65+i)}1", (0.0, 0.0, 0.0), t)
                         for i, t in enumerate(rec_types)]
        rec = rec_from(rec_specs)
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
    # Is the number stable, or an artefact of one fixture? Both halves are
    # varied: the polar hydrogen's typing, and whether the receptor carries
    # more than the one carbon. Neither moves it, which is what makes the
    # number a reportable measurement rather than a curiosity of the fixture.
    h_bare = zero_crossing("C", (["H"], [0.0], np.array([[0.0, 0.0, 0.0]])),
                           1.0, 8.0)
    h_hydroxyl = zero_crossing(
        None, (["H"], [0.0], np.array([[0.0, 0.0, 0.0]]), None, ["HD_H1"]),
        1.0, 8.0,
        rec_specs=[("C1", (0.0, 0.0, 0.0), "C"),
                   ("O1", (1.43, 0.0, 0.0), "OA"),
                   ("H1", (2.05, 0.93, 0.0), "HD")],
    )
    # And a probe the engine will not even score: a lone nitrogen or oxygen
    # reads a different map group, so its bisection never brackets a crossing
    # and the radius is not measurable for it at all. Reporting the bound is
    # more honest than reporting a number the search never found.
    n_bounded = zero_crossing("C", (["N"], [0.0], np.array([[0.0, 0.0, 0.0]])),
                              1.0, 8.0)
    check("and the hydrogen radius is NOT the 0.0 the spec states",
          h_radius > 0.4
          and abs(h_bare - h_cross) < 0.01
          and abs(h_hydroxyl - h_cross) < 0.01
          and n_bounded >= 7.99,
          f"a C...H pair crosses zero at r = {h_cross:.3f} A, so the hydrogen "
          f"radius is {h_radius:.3f} A where SCORING.md 3.2.1 says \"H is 0\". "
          f"The number is stable, not a fixture artefact: a bare `H` gives "
          f"{h_bare:.3f} A and a hydroxyl receptor gives {h_hydroxyl:.3f} A, "
          f"both within 0.01 A of it. A lone `N` or `O` probe against the same "
          f"carbon never crosses zero anywhere in 1-8 A -- the bisection runs "
          f"to its bound and returns {n_bounded:.2f} A -- because it reads a "
          f"different map group, so no radius is measurable for those probes "
          f"by this method at all. **Reported, not diagnosed.** No binding "
          f"exposes a radius, `score_conformation` returns a total, and the "
          f"score-time normalisation of 9.1 is not inverted here, so this file "
          f"cannot say whether \"H is 0\" is about the receptor-side radii used "
          f"to build the maps while the probe is given a different one, or "
          f"whether polar hydrogen is given a carbon-scale radius outright. "
          f"That question is answered by reading scoring.rs, which this file "
          f"does not do -- it is written from the specification only, and "
          f"opening the implementation to settle a discrepancy would destroy "
          f"the only thing that makes it a cross-check. What can be said "
          f"without it: a C...H pair's zero crossing is at {h_cross:.2f} A, not "
          f"at the {XS_RADIUS['C']:.2f} A the spec's formula gives, and any "
          f"reference written from the spec alone will disagree with the engine "
          f"there")

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
    check("a donor-acceptor pair scores exactly zero, and the map says why: a "
          "probe reads only the group matching its own element",
          hb_row[3] == 0.0,
          f"an OA receptor against an N carrying a polar H, probe classified "
          f"{hb_row[5]!r} at r = {hb_row[1]:.2f} A, gives the engine "
          f"{hb_row[3]:+.6f} and the reference {hb_row[4]:+.4f} "
          f"(decomposed g1 {hb_row[6]['g1']:+.4f}, g2 {hb_row[6]['g2']:+.4f}, "
          f"rep {hb_row[6]['rep']:+.4f}, hb {hb_row[6]['hb']:+.4f}). The two "
          f"candidates this check used to carry unresolved are now separated by "
          f"reading the map, and **neither is the whole story**: the lone OA "
          f"does reach the acceptor map -- a lone OA receptor populates group 2 "
          f"slot 2, whose minimum is -0.587439, exactly the hb weight of "
          f"section 3 -- so it is not a missing map. The zero is the four-slot "
          f"split, and more precisely a per-element partition: the map is ten "
          f"type groups of four slots, and a probe atom sums only the group "
          f"matching its own element. An N probe cannot see the receptor's O "
          f"acceptor group, and a hydrogen bond between N and O is precisely "
          f"the interaction that requires it. Measured in the cross-tab below. "
          f"The reference disagrees with the engine here and this file does not "
          f"claim otherwise; the disagreement is now localised to Rust, not to "
          f"the fixture")

    # ------------------------------------------------------------------
    section("what the map contains, and which of it is read")

    # The layout and the group index are **data, not prose**, so that the two
    # explanations below can be falsified instead of believed. An earlier draft
    # of this file stated the partition in a docstring and a check *name*; that
    # is prose with a colon, and prose rots silently. Everything the engine is
    # being held to is written here as a table the engine then has to match,
    # and each table is mutated below to prove the match is not a tautology.
    #:
    #: Two rows of this table were **wrong the first time it was written** and
    #: are the reason it is measured rather than recalled. `OD` populates slot
    #: 1, so "no lone atom puts anything in the donor slot" was false; and
    #: `CG0` -- a type meeko really writes -- lands in the carbon group and is
    #: counted as an unknown type, because the engine reads its first two
    #: characters.
    LONE = ("C", "A", "HD", "H", "N", "NA", "ND", "NDA", "O", "OA", "OS",
            "OD", "ODA", "P", "S", "SA", "F", "Cl", "CG0")
    GROUP_OF = {"C": 0, "A": 0, "HD": 0, "H": 0,
                "N": 1, "NA": 1, "ND": 1, "NDA": 1,
                "O": 2, "OA": 2, "OS": 2, "OD": 2, "ODA": 2,
                "P": 3, "S": 4, "SA": 4, "F": 5, "Cl": 6, "CG0": 0}
    SLOTS_OF = {
        "C": [0, 3], "A": [0, 3], "HD": [0], "H": [0],
        "N": [0], "NA": [0, 2], "ND": [0, 1], "NDA": [0, 1],
        "O": [0], "OA": [0, 2], "OS": [0], "OD": [0, 1], "ODA": [0, 1],
        "P": [0], "S": [0], "SA": [0], "F": [0, 3], "Cl": [0, 3], "CG0": [0],
    }
    #: The four slots, named by what they do rather than by what section 5.1
    #: calls them. Slot 1's name is the interesting one: only `ND` and `OD` put
    #: anything in it, and the check after the cross-tab is about that.
    SLOT_NAME = {0: "steric", 1: "hb-from-donor", 2: "hb-from-acceptor",
                 3: "hydrophobic"}
    layout = {t: populated(rec_of(t)) for t in LONE}
    # `populated` returns lists because a multi-atom receptor can span groups;
    # a lone atom cannot, and the tables below are written as scalars, so the
    # unwrap is explicit and its precondition is asserted rather than assumed.
    spread = {t: (g[0], g[1]) for t, g in layout.items()
              if len(g[0]) != 1 or len(g[1]) < 1}
    measured = ({t: g[0][0] for t, g in layout.items()},
                {t: g[1] for t, g in layout.items()})
    group_drift = {t: (GROUP_OF[t], measured[0][t]) for t in layout
                   if GROUP_OF[t] != measured[0][t]}
    slot_drift = {t: (SLOTS_OF[t], measured[1][t]) for t in layout
                  if SLOTS_OF[t] != measured[1][t]}
    donors = [t for t, s in SLOTS_OF.items() if 1 in s]
    check("the map is ten type groups of four slots, and a single atom's whole "
          "contribution matches a written-down table",
          not spread and not group_drift and not slot_drift,
          "; ".join(f"{k} -> group {v[0][0]}, slots "
                    f"{[SLOT_NAME[s] for s in v[1]]}"
                    for k, v in layout.items())
          + f". The tables above are this file's claim and these numbers are "
            f"the engine's, compared rather than asserted in prose. Atoms "
            f"spanning more than one group (must be none for a lone atom): "
            f"{spread or 'none'}. Group drift: {group_drift or 'none'}. Slot "
            f"drift: {slot_drift or 'none'}. Slot 0 is populated by every atom; "
            f"slot 2 by the acceptors; **slot 1, {SLOT_NAME[1]}, by "
            f"{donors} only**. Ten groups is why `raw_data`'s `type*4 + slot` "
            f"means a slot index is not a term until a group is chosen, and why "
            f"the four slot names this file used to assume could not be read "
            f"off an all-carbon receptor"
          + (f". `NDA` reproduces `ND` exactly and `ODA` reproduces `OD` "
             f"exactly, group and slots both, which is the three-character "
             f"truncation seen through the map rather than through "
             f"`atom_kinds`; `CG0` lands in the carbon group and "
             f"`unknown_atom_types` is "
             f"{rec_of('CG0').unknown_atom_types}, so a type meeko really "
             f"writes is mistyped as `CG`"
             if "CG0" in layout else ""))

    # The cross-tab. Every cell is a total from the engine at one distance; a
    # cell that is exactly zero means the receptor contributed *nothing at
    # all*, sterics included, which is a much stronger statement than "the
    # hydrogen-bond term was absent".
    #
    # The explanation is a **rule**, written as a function, and the check is
    # that the engine's table equals the table the rule predicts. That is what
    # makes the explanation falsifiable: mutate the rule and the prediction
    # moves, and the engine disagrees. Asserting "these seventeen cells are
    # zero" would have been a list, and a list rots the moment a thirteenth
    # atom type appears.
    probes = {name: Ligand.from_arrays(*spec)
              for name, spec in PROBE_SPECS.items()}
    probes["OA (via file)"] = from_file(["OA"])
    receptors = ["C", "N", "NA", "OA", "S", "Cl"]

    def groups_of(probe_name, group_of):
        """Which map groups a probe sums, under `group_of`."""
        if group_of is None:                       # the mutant: every group
            return set(range(10))
        return {group_of[PROBE_TYPE[probe_name][i]]
                for i in range(len(PROBE_TYPE[probe_name]))}

    def predicts_zero(probe_name, rec_type, group_of):
        """Does the rule say this cell must be exactly zero?"""
        if group_of is None:
            return False
        return group_of[rec_type] not in groups_of(probe_name, group_of)

    tab = {(p, r): at(probes[p], rec_from([(r, (0, 0, 0), r)]))
           for p in probes for r in receptors}
    wrong = sorted(f"{p}/{r}" for (p, r), v in tab.items()
                   if (v == 0.0) != predicts_zero(p, r, GROUP_OF))
    cross_zero = sorted(f"{p}/{r}" for (p, r), v in tab.items() if v == 0.0)
    cross_live = sorted(f"{p}/{r}={v:+.4f}"
                        for (p, r), v in tab.items() if v != 0.0)
    donor_row = {r: tab[("N+H", r)] for r in receptors}

    # Three mutations of the rule. Each must move the prediction and be caught.
    def _wrong_cells(group_of):
        return sum(1 for (p, r), v in tab.items()
                   if (v == 0.0) != predicts_zero(p, r, group_of))

    mutants = {
        "rule as written": _wrong_cells(GROUP_OF),
        "probe sums all ten groups": _wrong_cells(None),
        "NA moved to the oxygen group": _wrong_cells({**GROUP_OF, "NA": 2}),
        "S moved to the carbon group": _wrong_cells({**GROUP_OF, "S": 0}),
    }
    check("and the rule that explains the zeros predicts the whole table: a "
          "probe sums only the map groups of its own elements, and three "
          "mutations of that rule are each caught",
          not wrong
          and mutants["rule as written"] == 0
          and all(mutants[k] > 0 for k in
                  ("probe sums all ten groups", "NA moved to the oxygen group",
                   "S moved to the carbon group")),
          f"at {PROBE_AT} A, the engine gives {len(cross_zero)} zero cells of "
          f"{len(tab)} and the rule predicts "
          f"{sum(1 for p in probes for r in receptors if predicts_zero(p, r, GROUP_OF))}"
          f"; cells where the two disagree: {wrong or 'none'}. Zero cells: "
          f"{', '.join(cross_zero)}. Non-zero: {', '.join(cross_live)}. Probe "
          f"classes: "
          + ", ".join(f"{n}->{l.atom_kinds}" for n, l in probes.items())
          + f". N+H row: {donor_row['NA']:+.4f} against an NA receptor, an "
            f"attraction, and {donor_row['OA']:+.6f} against an OA one, exactly "
            f"zero -- same probe, same distance, only the element differs, and "
            f"the element picks the group. **Mutations, counted as cells where "
            f"the rule and the engine now disagree:** "
          + "; ".join(f"{k} -> {v}" for k, v in mutants.items())
          + f". Letting a probe sum all ten groups is wrong in "
            f"{mutants['probe sums all ten groups']} cells, which is the whole "
            f"table: that is the claim that there is no partition at all. "
            f"Moving one receptor's group is wrong in "
            f"{mutants['NA moved to the oxygen group']} and "
            f"{mutants['S moved to the carbon group']} cells respectively, which "
            f"is the claim that the group index is not arbitrary. So the "
            f"explanation survives three attempts to break it, and a "
            f"nitrogen-oxygen hydrogen bond remains inexpressible. Change "
            f"request against Rust; this file did not read scoring.rs and "
            f"claims nothing about intent")

    # Which slots a probe reads, and the group it may read them from. Both are
    # data, both are mutated below.
    #
    # The measurement that forced this rewrite: an `OD` receptor and an `O`
    # receptor differ in **exactly one** map component (group 2, slot 1), and a
    # genuine acceptor probe gives two different numbers for them -- so the
    # donor slot is read. An earlier revision of this check concluded the
    # opposite from five receptors scoring bitwise identically, and was wrong:
    # the probe it used was an `other`-class oxygen, and `other` reads the
    # steric slot only. See the module docstring.
    SLOT_MASK = {"hydrophobic": {0, 3}, "donor": {0, 2},
                 "acceptor": {0, 1}, "other": {0}}
    #: The class each probe is *asserted* to have, cross-checked against
    #: `atom_kinds` in the check below. Two of the four probes are `other`, in
    #: different element groups, which is a useful case rather than a duplicate:
    #: an `other` nitrogen reads slot 0 of group 1 and an `other` oxygen reads
    #: slot 0 of group 2, so neither can confirm the other.
    PROBE_CLASS = {"C": "hydrophobic", "N+H": "donor",
                   "OA (via file)": "acceptor", "N": "other", "O": "other"}
    #: Receptor pairs that differ in exactly one map component, so the
    #: component is the only thing that can explain any difference in score.
    ONE_COMPONENT_PAIRS = [("OD", "O", 9), ("ND", "N", 5)]
    #: ... and one pair differing in two, used to show `other` ignores the
    #: acceptor slot as well.
    TWO_COMPONENT_PAIRS = [("OD", "OA", (9, 10)), ("ND", "NA", (5, 6))]

    def _map_of(t):
        return rec_of(t).precalculate(PROBE_BOX, "vina").raw_data.reshape(-1, 40)

    pair_evidence = {}
    for a_t, b_t, comps in ONE_COMPONENT_PAIRS + TWO_COMPONENT_PAIRS:
        want = (comps,) if isinstance(comps, int) else comps
        ma, mb = _map_of(a_t), _map_of(b_t)
        differing = tuple(k for k in range(40)
                          if not np.array_equal(ma[:, k], mb[:, k]))
        responses = {pn: (at(probes[pn], rec_of(a_t))
                          != at(probes[pn], rec_of(b_t)))
                     for pn in probes}
        pair_evidence[(a_t, b_t)] = (differing, want, responses)

    # The rule, as two conditions: a probe's score must change for a receptor
    # pair only if the receptor is in the probe's own element group **and** one
    # of the differing components is a slot that probe's class reads. The
    # first version of this compared a group *index* against a *set* of them
    # with `!=`, which is always true, so the mask was never consulted and
    # every mutation of it reported the same count. Written here as `not in`
    # so that the second condition is reachable at all.
    def predicts_responds(probe_name, rec_type, differing, mask):
        """Does the rule say a probe's score must differ for this receptor pair?

        Two preconditions, and both are measured rather than assumed: the
        receptor must be in the probe's own element group, and at least one
        differing component must be a slot this probe's class reads. A pair
        differing in two slots can therefore respond through either.
        """
        if GROUP_OF[rec_type] not in groups_of(probe_name, GROUP_OF):
            return False
        read = mask[PROBE_CLASS[probe_name]]
        return any((k % 4) in read for k in differing)

    def _resp_wrong(mask):
        out = {}
        for (a_t, b_t), (differ, _, responses) in pair_evidence.items():
            for pn, got in responses.items():
                want = predicts_responds(pn, a_t, differ, mask)
                if got != want:
                    out[f"{pn}:{a_t}/{b_t}"] = (f"predicted "
                                                f"{'RESPONDS' if want else 'no change'}",
                                                f"got {'RESPONDS' if got else 'no change'}",
                                                differ)
        return out

    resp_wrong = _resp_wrong(SLOT_MASK)
    #: The class table is a claim about the fixtures, so it is compared with
    #: what the engine says they are rather than trusted.
    class_drift = {pn: (PROBE_CLASS[pn], probes[pn].atom_kinds[0])
                   for pn in probes
                   if PROBE_CLASS[pn] != probes[pn].atom_kinds[0]}
    slot_muts = {
        "mask as written": len(resp_wrong),
        "acceptor stops reading the donor slot": len(
            _resp_wrong({**SLOT_MASK, "acceptor": {0}})),
        "donor starts reading the donor slot": len(
            _resp_wrong({**SLOT_MASK, "donor": {0, 1, 2}})),
        "`other` reads every slot": len(
            _resp_wrong({**SLOT_MASK, "other": {0, 1, 2, 3}})),
    }
    donor_reads = [(a_t, b_t, pn) for (a_t, b_t), (_, _, responses)
                   in pair_evidence.items() for pn, got in responses.items()
                   if got and PROBE_CLASS[pn] == "acceptor"]
    check("the donor slot is read, and only from inside the same element group: "
          "a one-component receptor pair is what proves it, and three mutations "
          "of the slot mask are each caught",
          not resp_wrong and not class_drift
          and all(pair_evidence[k][0] == pair_evidence[k][1]
                  for k in pair_evidence)
          and len(donor_reads) == 2
          and all(m == 0 if k == "mask as written" else m > 0
                  for k, m in slot_muts.items()),
          "; ".join(
              f"{a_t} vs {b_t} differ in component(s) "
              f"{[f'g{k // 4}s{k % 4}' for k in differ]} -> "
              + ", ".join(f"{pn}[{PROBE_CLASS[pn]}]"
                          f"{' RESPONDS' if r else ' no change'}"
                          for pn, r in responses.items())
              for (a_t, b_t), (differ, _, responses)
              in pair_evidence.items())
          + f". Fixture classes, claimed against `atom_kinds`: "
          + ", ".join(f"{pn}={probes[pn].atom_kinds[0]}" for pn in probes)
          + f"; drift {class_drift or 'none'}. The decisive pair is **OD vs O**, "
            f"which differ in component 9 and nothing else: "
          + "; ".join(f"{pn} gives {at(probes[pn], rec_of(a_t)):+.6f} vs "
                      f"{at(probes[pn], rec_of(b_t)):+.6f}"
                      for a_t, b_t, pn in donor_reads
                      for a_t, b_t in ((a_t, b_t),) if (a_t, b_t) == ("OD", "O"))
          + f". **This corrects an earlier revision of this file**, which "
            f"concluded the donor slot was read by nothing on the evidence of "
            f"five oxygen receptors scoring bitwise identically. They did, and "
            f"the probe was the problem: it was an `other`-class oxygen, and "
            f"`other` reads the steric slot only. So the finding is withdrawn "
            f"and the chemistry inside a group is right -- a donor reads the "
            f"receptor's acceptor field, an acceptor reads its donor field, "
            f"like-with-like is withheld, and `other` reads sterics only. The "
            f"one defect left is the element partition. Cells where the mask "
            f"and the engine disagree: {resp_wrong or 'none'}. **Mutations, "
            f"counted the same way:** "
          + "; ".join(f"{k} -> {v}" for k, v in slot_muts.items())
          + ". **What this table does not cover, stated rather than left to be "
            "discovered:** every fixture here is a one-atom ligand, and the "
            "engine has a fifth class a one-atom ligand cannot be -- "
            "`donoracceptor`, which needs a bonded polar hydrogen. The "
            "project's own shipped pose has one: in "
            "`examples/crambin_pose.pdbqt` MODEL 1, O14 reads `acceptor` and "
            "O15 reads `donoracceptor`, both typed `OA`, differing only in "
            "whether the torsion tree bonds H16 to it. Measured on that real "
            "system, the carboxylate oxygen **does** read the receptor's "
            "acceptor field, which this table predicts it does not: "
            "relabelling every receptor `OA` to `OS` -- same group, same "
            "steric field, acceptor field emptied, and the *only* map "
            "component that differs is group 2 slot 2 -- moves that pose's "
            "score by -1.068 kcal/mol. So the table is a lower bound for "
            "bonded ligands, and a carboxylic acid is not an exotic case. No "
            "cheap isolated fixture was found for the class: a hand-built "
            "torsion tree came back with 0 torsions, and centring the 16-atom "
            "shipped ligand on a one-atom receptor puts its atoms inside one "
            "another and hits the grid's +15999.999 clamp. Asserting a mask "
            "for a class this file cannot place would be a claim it cannot "
            "back, so the class is reported here and left uncovered on purpose")

    # ------------------------------------------------------------------
    section("how a ligand's types are decided, and which entry point loses them")

    # An observation this file could not previously explain: `from_arrays`
    # with an oxygen named "OA_O1" classified `other`, while `from_pdbqt_str`
    # on type `OA` classified `acceptor`. Two causes were possible and the
    # check says which, by giving each a prediction the other does not make:
    #
    #   * the *entry point* is wrong -- `from_arrays` takes (elements,
    #     charges, coords, bonds, atom_names) and has no channel that says
    #     "acceptor", so every oxygen arrives as `O`/OP;
    #   * the *missing bonds* are wrong -- classification needs neighbours.
    #
    # Bonds are not it: a hydroxyl and an ester, both fully bonded, keep the
    # oxygen at `other`. The name is not it either: five different name
    # prefixes give one answer. So it is the entry point, and the mechanism is
    # that the `OA` prefix is read by the **Python** writer (`_split_name`) and
    # never reaches the engine.
    #
    # Both halves are written as rules and both are mutated, because a check
    # that only asserted the surprising half would pass if the file path broke.
    name_variants = {
        "OA_O1": ["OA_O1"], "O_O1": ["O_O1"], "C_O1": ["C_O1"],
        "ZZZZ_O1": ["ZZZZ_O1"], "no name": None,
    }
    by_name = {k: Ligand.from_arrays(["O"], [0.0], np.array([[0.0] * 3]),
                                     None, v).atom_kinds[0]
               for k, v in name_variants.items()}
    by_bond = {
        "lone O": (["O"], [[0, 0, 0]], None, ["OA_O1"]),
        "O-C": (["O", "C"], [[0, 0, 0], [1.43, 0, 0]], [(0, 1)],
                ["OA_O1", "C_C1"]),
        "O-C-C (ester)": (["O", "C", "C"],
                          [[0, 0, 0], [1.43, 0, 0], [-1.43, 0, 0]],
                          [(0, 1), (0, 2)], ["OA_O1", "C_C1", "C_C2"]),
    }
    bonded_kinds = {k: Ligand.from_arrays(els, [0.0] * len(els),
                                         np.array(crd, float), bonds, names)
                    .atom_kinds[0] for k, (els, crd, bonds, names)
                    in by_bond.items()}
    donor_works = Ligand.from_arrays(["N", "H"], [0.0, 0.0],
                                     np.array([[0.0, 0.0, 0.0],
                                               [0.0, 0.9, 0.0]], float),
                                     [(0, 1)], ["N_N1", "HD_H1"]).atom_kinds[0]
    file_oa = from_file(["OA"]).atom_kinds[0]

    #: The two rules, as data. `from_pdbqt_str` classifies by the **type
    #: column**; `from_arrays` sees an element symbol and nothing else, so it
    #: classifies by the **element** and can never reach an acceptor.
    def rule_type_column(atom_type: str) -> str:
        return {"OA": "acceptor", "OS": "other", "O": "other",
                "NA": "acceptor", "N": "other", "C": "hydrophobic"}.get(
                    atom_type, "other")

    def rule_element_only(atom_type: str) -> str:
        return {"O": "other", "N": "other", "C": "hydrophobic"}.get(
            atom_type[0], "other")

    file_cases = {t: from_file([t]).atom_kinds[0]
                  for t in ("O", "OA", "OS", "N", "NA", "C")}
    arrays_cases = {
        t: Ligand.from_arrays([t[0]], [0.0], np.array([[0.0] * 3]), None,
                              [f"{t}_X1"]).atom_kinds[0]
        for t in ("O", "OA", "OS", "N", "NA", "C")
    }
    file_drift = {t: (rule_type_column(t), k) for t, k in file_cases.items()
                  if rule_type_column(t) != k}
    arrays_drift = {t: (rule_element_only(t), k)
                    for t, k in arrays_cases.items()
                    if rule_element_only(t) != k}

    def _mutate(fn, cases):
        return sum(1 for t, k in cases.items() if fn(t) != k)

    muts = {
        "type-column rule as written": _mutate(rule_type_column, file_cases),
        "element-only rule as written": _mutate(rule_element_only,
                                               arrays_cases),
        "type column misreads OA as plain O": _mutate(
            lambda t: "other" if t == "OA" else rule_type_column(t),
            file_cases),
        "from_arrays pretended to read the name": _mutate(
            lambda t: rule_type_column(t), arrays_cases),
    }
    check("`from_arrays` has no acceptor channel and the name is irrelevant: "
          "two entry points, two written-down rules, four mutations, each caught",
          not file_drift and not arrays_drift
          and len(set(by_name.values())) == 1
          and set(bonded_kinds.values()) == {"other"}
          and donor_works == "donor" and file_oa == "acceptor"
          and muts["type-column rule as written"] == 0
          and muts["element-only rule as written"] == 0
          and all(muts[k] > 0 for k in muts if "as written" not in k),
          f"The oxygen probe: `from_arrays` with the name "
          + ", ".join(f"{k}->{v}" for k, v in by_name.items())
          + f" -- five prefixes, one answer -- and with bonds "
          + ", ".join(f"{k}->{v}" for k, v in bonded_kinds.items())
          + f". Bonds are not the cause and the name is not the cause, so the "
            f"entry point is. `from_arrays` takes (elements, charges, coords, "
            f"bonds, atom_names): nothing in that signature says \"acceptor\", "
            f"so the oxygen arrives as `O`/OP and never becomes one. The `OA` "
            f"prefix is read by the **Python** writer's `_split_name`, never by "
            f"the engine. The file reader, which has a type column, gets it "
            f"right: {file_oa!r} for a lone `OA`. Donors are unaffected -- a "
            f"bonded polar H still gives {donor_works!r} -- so the loss is "
            f"specific to acceptors. **Type-column rule drift** "
            f"{file_drift or 'none'}, **element-only rule drift** "
            f"{arrays_drift or 'none'}. **Mutations, counted as types where the "
            f"rule and the engine disagree:** "
          + "; ".join(f"{k} -> {v}" for k, v in muts.items())
          + f". So both rules are load-bearing in both directions, and neither "
            f"half of the observation is a fixture artefact. For scale, the same "
            f"hydroxyl reads {from_file(['OA', 'C', 'HD']).atom_kinds} through "
            f"a file and cannot be an acceptor in memory at all -- which is "
            f"why nothing in-tree is broken: every caller writes a file before "
            f"scoring, and the file carries the types")

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
          f" -- the steric and hydrophobic slots carry the carbon's field and "
          f"both hydrogen-bond slots are flat zero. That much is right for a "
          f"carbon, and this check used to end by calling it 'the 4-slot split "
          f"working as SCORING.md section 5.1 describes'. **That clause is "
          f"withdrawn**: the cross-tab above shows the split is a per-element "
          f"partition rather than a per-term filter, which is not what 5.1 "
          f"describes, and an all-carbon receptor is the one case in which the "
          f"two descriptions cannot be told apart -- which is why it survived "
          f"in a file that never tested a polar receptor. The slot *names* "
          f"here are also the file's assumption; the measured naming is in the "
          f"layout check above")

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
