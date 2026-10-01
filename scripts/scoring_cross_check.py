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

| lone receptor atom | groups populated | slots populated |
|---|---|---|
| every one of `C A H HD N NA ND NDA O OA OS OD ODA P S SA F Cl CG0` | **all ten** | see the table in `main` |

So the four slots still name four terms -- slot 0 is the steric/shape field,
which every atom populates; slot 3 is hydrophobic, which the apolar atoms
populate; slot 2 is populated by the acceptors (`NA`, `OA`); slot 1 is
populated by the donors (`ND`, `OD`) and by an explicit polar hydrogen. What
this file used to describe as a **per-element partition** is gone: a lone atom
used to be written into one group of ten and read only out of the group
matching the probe's own element, which is why a polar donor-acceptor pair
scored exactly `0.000000`. Every atom is now written into all ten groups and
every probe reads all ten, so the element partition no longer exists and a
nitrogen can see an oxygen's acceptor field.

# What the fix did *not* do, and the measurement that says so

The ten groups are not copies of one another, so replicating a field into ten
of them could have rescaled every term. It did not, and the reason is
measurable rather than asserted: relabelling every `OA` in the shipped pose's
receptor to `OS` -- same element, same steric field, and the *only* map
component that changes is the acceptor slot -- moves the pose by
**1.068 kcal/mol**, the same magnitude the pre-fix build produced, even though
pre-fix that field was written once and now it is written ten times. The
check below also attributes that number to a **single atom**: of the pose's 16
atoms, only the one `donoracceptor` oxygen reads the acceptor slot, and
deleting exactly that atom takes the delta to `0.000000` while the other 15
leave it untouched.

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

So the donor slot **is** read: a donor reads the receptor's *acceptor* field,
an acceptor reads the receptor's *donor* field, and like-with-like is
withheld. At the time this collapsed the two findings into one -- the element
partition -- and the element partition has since been removed, so the
like-with-like withholding is now the whole of it: it is a property of the
**slot**, and nothing about the elements is involved. The rule this file
finally holds the engine to, measured in 28 probe/receptor-pair cases with no
element condition anywhere in it, is one line: *a probe's score changes for a
receptor pair if and only if one of the differing slots is a slot that probe's
class reads.*

The general lesson, recorded because it will recur: **a probe fixture's class
is a claim, not a fact, and `atom_kinds` is the only way to check it.** Two
checks below now assert the class of every fixture they use, precisely because
a mislabelled probe produced a false positive that survived a bitwise
comparison.
"""

from __future__ import annotations

import collections
import re
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
    GridBox, Ligand, Receptor, conformation_coordinates, load_ligand,
    score_conformation, score_conformation_terms, scoring_descriptions,
    typed_ligand_from_tables,
)
# `prep` imports RDKit lazily inside its functions, so importing the module
# here does not make this file depend on RDKit being installed. That matters:
# the checks below that use it are about the prepared path, and a suite that
# could not be imported without an optional dependency would be skipped
# exactly where the typing logic lives.
from opendocking.prep import prepare_ligand  # noqa: E402


def _atom_lines(text: str) -> list[str]:
    """The ATOM/HETATM records of a PDBQT document, in order."""
    return [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))]


def _round_trip_text(*tables) -> str:
    """The PDBQT text `typed_ligand_from_tables` carries the types through.

    Returned alongside the ligand so a caller can assert on the *carrier* and
    the *engine's reading of it* separately -- the first is a string
    comparison needing neither the engine nor a map, which is the only guard
    that cannot be satisfied by both sides drifting together.
    """
    return typed_ligand_from_tables(*tables)[1]


# --- three ways the type carrier could silently do its job wrong ----------
# Applied to the PDBQT text rather than to Rust, because the guard under test
# is "the prepared path and the file path produce the same classes" and a
# broken carrier is exactly what would break it. Each is a mistake the round
# trip could make and never notice: no type written, an acceptor flattened to
# a plain oxygen, and a polar hydrogen dropped from the branch that is what
# makes a donor visible.
def _mut_blank_type(text: str) -> str:
    return "\n".join(
        l[:W._COL_TYPE[0]] + "  " if l.startswith(("ATOM", "HETATM")) else l
        for l in text.splitlines()) + "\n"


def _mut_flatten_acceptor(text: str) -> str:
    out = []
    for l in text.splitlines():
        if (l.startswith(("ATOM", "HETATM"))
                and l[W._COL_TYPE[0]:].strip() in ("OA", "NA")):
            l = l[:W._COL_TYPE[0]] + " O"
        out.append(l)
    return "\n".join(out) + "\n"


def _mut_drop_polar_h(text: str) -> str:
    return "\n".join(
        l for l in text.splitlines()
        if not (l.startswith(("ATOM", "HETATM"))
                and l[W._COL_TYPE[0]:].strip() == "HD")) + "\n"

#: How many checks this file is supposed to run, counted by running it; the
#: reference is a spec check, not Vina.
#:
#: 21 -> 27, and the six that moved are the per-term section. Four of the six
#: are new (`each of the five terms`, `the five terms sum`, `a term is a
#: property of the PAIR`, `per-slot is a different question`) and two are
#: replacements: the old `no per-pair or per-term ENERGY is exposed` check is
#: still called, so it still counts, but its *claim* is now false and it is
#: annotated as superseded rather than deleted. The count is a census, not a
#: score -- the derivation above is what a reader has to be able to check.
EXPECTED_CHECKS = 27

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


def smoothstep(x, a, b):
    """S(a, b, x) from SCORING.md section 2.3: 0 below a, 1 above b, C1 between.

    Written with `clip` rather than three branches so the same function serves
    the scalar callers and the vectorised group-radius fit below. For a scalar
    the result is identical to the branch form: below `a` the clipped `t` is 0
    and `0*(3-0)` is 0.0 exactly, above `b` it is 1 and `1*(3-2)` is 1.0
    exactly, and in between it is the same polynomial.
    """
    t = np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)
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
    d = np.asarray(d, dtype=float)
    g1 = w["g1"] * np.exp(-(((d - 0.5) / 0.5) ** 2))
    g2 = w["g2"] * np.exp(-(((d - 0.0) / 0.5) ** 2))
    rep = w["rep"] * np.where(d < 0.0, d * d, 0.0)
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


#: The map box the group radii are recovered in. Larger than `PROBE_BOX` so the
#: annulus holds enough points at every radius to fit against.
FIT_HALF = 7.0
FIT_BOX = GridBox((-FIT_HALF, -FIT_HALF, -FIT_HALF), (FIT_HALF, FIT_HALF,
                                                     FIT_HALF))
# --------------------------------------------------------------------------
# Where the hydrogen actually went, and why the spec's "H is 0" is not a
# disagreement with the engine
# --------------------------------------------------------------------------
#
# **Measured, and it is the reason the C...H crossing cannot be read as a
# radius.** The ten groups are ten *element* radii -- the smallest is 1.212 A
# and nothing in the ladder is near zero -- so there is no hydrogen group to
# have a radius at all. A C...H zero crossing is therefore built out of the ten
# element radii, dominated by the carbon group's 1.910 A, and the
# `crossing - 2 * R_C` residual this file prints is an arithmetic difference,
# not a distance. Printing it as a radius was the first draft of this check's
# wording and it was wrong.
#
# **Read from the implementation, not measured, and labelled as such** because
# this file is otherwise written from `docs/SCORING.md` alone and opening the
# Rust to settle a discrepancy would destroy the only thing that makes it a
# cross-check. `dock-core/src/types.rs:202-204` gives the engine's reason for
# keeping hydrogen at zero: "Hydrogens keep a radius of 0: the only hydrogens a
# PDBQT retains are polar ones, and they sit *on* their own heavy-atom donor, so
# giving them a radius would make every ordinary hydrogen bond register as an
# overlap." So the spec's "H is 0" and the measured ladder do not conflict: the
# spec is describing a radius the engine never tabulates, because the hydrogen
# is scored through the heavy atom it is attached to, not as a body of its own.
# What is measurable -- and measured above -- is only that no group is near
# zero. **The two claims are different in kind and are kept apart here on
# purpose.**
#: A second, deliberately different annulus. Two fits that differ only in which
#: grid points they use give this file its own noise floor: a value that moves
#: by more than the tolerance is not being measured, and a tolerance that is not
#: comfortably above the noise floor is not testing anything. The tolerance is
#: asserted against this number rather than trusted, so loosening it is a
#: failure the check can see.
FIT_ANNULUS = (0.6, 5.0)
FIT_ANNULUS_ALT = (1.0, 4.2)
#: The element whose spec radius anchors the fit's absolute scale. The fit
#: measures `R_group + R_receptor`, so this is the one subtraction that turns a
#: shift into a radius, and it is named rather than written inline so that the
#: places that depend on it are visible.
C_GRID_TYPE_ANCHOR = "C"


def group_radii(atom_type: str = "C",
                annulus: tuple[float, float] = FIT_ANNULUS) -> list[float]:
    """Recover the XS radius each of the ten map groups is built with.

    A group's field is the spec's pair function evaluated at
    ``d = r - (R_group + R_atom)``, so the radius is a *shift* of a known
    curve and comes out of a least-squares fit against that curve. The carbon
    receptor is used because it is `hydrophobic`, which makes the spec's
    ``apolar`` decomposition the right one to fit against, and because the
    probe that reads a group does not depend on the receptor: every group is
    read by every probe now, so the same ten radii serve all of them.

    The subtraction at the end is what turns a shift into a radius, and it is
    the only place this function depends on the spec: the fit measures
    ``R_group + R_atom``, and ``R_atom`` is taken from `XS_RADIUS`. So the
    spec has to state a radius for the receiver's element, and when it does not
    this raises rather than guessing -- a default anchor would shift all ten
    numbers by the same amount and look like ten correct answers. It raises
    rather than returning `None` because a silent `None` here becomes a
    plausible-looking list of radii downstream.

    This is the mechanism behind most of what follows, and it is measured
    rather than read out of the implementation for the reason the rest of this
    file is written the way it is: the spec is the reference, and a constant
    the spec states is a prediction to be falsified, not a value to be copied.
    """
    receiver = atom_type[0]
    if receiver not in XS_RADIUS:
        raise ValueError(
            f"cannot anchor the fit for a {receiver!r} receptor: the fit measures "
            f"R_group + R_receptor, and docs/SCORING.md 3.2.1 states no XS "
            f"radius for {receiver!r}, so there is no anchor. It states "
            f"{sorted(XS_RADIUS)}; pass a receiver whose element is in that list, "
            f"or add a value you have measured -- do not let this fall back to a "
            f"default, because a wrong anchor moves all ten radii by the same "
            f"amount and reads as ten correct answers"
        )
    maps = rec_of(atom_type).precalculate(FIT_BOX, "vina")
    nx, ny, nz = maps.dims
    step = maps.spacing
    axes = [(np.arange(n) - (n - 1) / 2.0) * step for n in (nx, ny, nz)]
    radius = np.sqrt(sum(np.square(np.meshgrid(*axes, indexing="ij")))).reshape(-1)
    per = maps.raw_data.reshape(-1, 40)
    # The grid is clamped near the nucleus and is identically zero past the
    # map's cutoff; neither is part of the pair function, so both are excluded
    # rather than fitted around.
    annulus_mask = (radius > annulus[0]) & (radius < annulus[1])
    rs = radius[annulus_mask]
    shifts = np.arange(1.5, 4.6, 0.0025)
    out = []
    for g in range(10):
        vals = per[annulus_mask, 4 * g]
        unclamped = np.abs(vals) < 100.0
        d, v = rs[unclamped], vals[unclamped]
        rms = [float(np.sqrt(np.mean(
            (reference_pair(d - s, apolar=True)["total"] - v) ** 2)))
            for s in shifts]
        out.append(float(shifts[int(np.argmin(rms))]) - XS_RADIUS[receiver])
    return out


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

#: Which of the four map slots each class reads. This is the whole of the
#: selectivity the engine has left: no element, no group, nothing but the
#: slot. It is mutated below, and every mutation is caught, so it is a claim
#: being tested rather than a description.
SLOT_MASK = {"hydrophobic": {0, 3}, "donor": {0, 2},
             "acceptor": {0, 1}, "other": {0},
             # The fifth class, which a one-atom ligand cannot be: it needs a
             # bonded polar hydrogen. It is covered on the project's own shipped
             # pose rather than left as an assumption -- see the slot-mask check.
             "donoracceptor": {0, 1, 2}}


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
    # Is the number stable, or an artefact of one fixture? Both halves that
    # *should* be neutral are varied -- the probe's own typing, and whether the
    # receptor carries more than one atom. Neither moves it, and both move it
    # **bitwise** not just to within a tolerance, which is what makes this a
    # measurement of a map rather than of a rounding error.
    h_bare = zero_crossing("C", (["H"], [0.0], np.array([[0.0, 0.0, 0.0]])),
                           1.0, 8.0)
    h_two_c = zero_crossing(["C", "C"], (["H"], [0.0],
                                         np.array([[0.0, 0.0, 0.0]]),
                                         None, ["HD_H1"]), 1.0, 8.0)
    # And the half that used to be neutral and no longer is. While a probe read
    # only the map group of its own element, a lone `N` or `O` probe could not
    # see a lone carbon at all, so its bisection never bracketed a crossing and
    # ran to its bound. All three of these now cross zero, and they cross in
    # the order the spec's XS radii predict: the smaller the receptor's element
    # radius (C 1.9 > N 1.75 > O 1.6), the further in the crossing sits. The
    # ordering is the check; the three numbers being finite is not.
    n_cross = zero_crossing("C", (["N"], [0.0], np.array([[0.0, 0.0, 0.0]])),
                            1.0, 8.0)
    o_cross = zero_crossing("C", (["O"], [0.0], np.array([[0.0, 0.0, 0.0]]),
                                  None, ["OA_O1"]), 1.0, 8.0)
    # The mechanism behind the sentence below, measured rather than asserted:
    # fit each group's field against the spec's own pair function and read the
    # shift off as a radius. All ten are pinned, not the first three.
    #
    # Two independent numbers come out of that fit and they are worth keeping
    # apart, because conflating them is how a good measurement becomes a bad
    # one. The ten *absolute* radii are checked against a written-down table,
    # and the ten *differences* are checked against the spec's own radii, taken
    # from group 0 so that anything common to all ten cancels. The second is
    # not a restatement of the first: it is a prediction made from
    # `docs/SCORING.md` alone, and it would survive every entry in `LADDER`
    # being wrong in the same direction by the same amount.
    #
    # There is a third property, and it is the one that catches a single group
    # changing without any literal being updated: the ten agree *with each
    # other* to a few thousandths of an angstrom, far below the tolerance. A
    # per-group change to one radius shows up as a non-uniform offset even when
    # the changed value would still sit inside the tolerance.
    radii = group_radii("C")
    radii_alt = group_radii("C", FIT_ANNULUS_ALT)
    #: This file's own noise floor, measured by refitting against a different
    #: annulus. A tolerance that does not sit above it is not testing anything,
    #: so the check asserts the relationship rather than assuming it.
    radii_noise = max(abs(a - b) for a, b in zip(radii, radii_alt))
    #: The ten radii as measured, in group order. **This is the claim**: a
    #: written-down table the engine then has to match, mutated below to prove
    #: the match is not a tautology.
    LADDER = (1.910, 1.760, 1.610, 2.110, 2.010,
              1.555, 1.957, 2.230, 2.360, 1.212)
    #: How far a measured radius may sit from `LADDER` before it counts as
    #: different. Above the noise floor by an order of magnitude and below the
    #: smallest gap between any two entries, so it discriminates every pair.
    LADDER_TOLERANCE = 0.05
    ladder_drift = {g: (round(r, 3), LADDER[g], round(r - LADDER[g], 4))
                    for g, r in enumerate(radii)
                    if abs(r - LADDER[g]) > LADDER_TOLERANCE}
    #: The offset between measurement and table, measured rather than assumed:
    #: the spread of `measured - written` across all ten groups. The ten agree
    #: with each other far more closely than the tolerance, which is what makes
    #: this a structural guard rather than a restatement of the table: a change
    #: to one group's radius breaks the uniformity even if the value it changed
    #: to happened to stay inside the tolerance.
    bias = [round(r - LADDER[g], 4) for g, r in enumerate(radii)]
    bias_spread = max(bias) - min(bias)
    #: The spec-anchored half, which is a *different* prediction from the same
    #: data and does not need any of the ten literals. SCORING.md 3.2.1 names
    #: five elements; hydrogen is not one of the ten groups, so four of the ten
    #: have a spec value to be compared against. Differences from group 0, so
    #: the fit bias cancels on both sides.
    spec_drift = {e: (round(radii[g] - radii[0], 3),
                      round(XS_RADIUS[e] - XS_RADIUS["C"], 3))
                  for g, e in enumerate(("C", "N", "O", "P", "S", "F"))
                  if e in XS_RADIUS
                  and abs((radii[g] - radii[0])
                          - (XS_RADIUS[e] - XS_RADIUS["C"])) > 0.02}
    check("the C...H zero crossing is far from the 1.9 A the spec's formula "
          "gives, and all ten group radii are pinned, not the first three",
          h_radius > 0.4
          and h_bare == h_cross and h_two_c == h_cross
          and h_cross > n_cross > o_cross
          and max(h_cross, n_cross, o_cross) < 7.99
          and not ladder_drift and not spec_drift
          and radii_noise < LADDER_TOLERANCE
          and LADDER_TOLERANCE < 0.5
          and bias_spread < 0.02
          and len(LADDER) == 10,
          f"a C...H pair crosses zero at r = {h_cross:.3f} A, which is "
          f"{h_radius:.3f} A past the {2 * XS_RADIUS['C']:.2f} A that "
          f"SCORING.md 3.2.1's formula gives for an H radius of 0. **What that "
          f"residual is not, stated before it is used:** a hydrogen radius. "
          f"Fitting each of the ten groups against the spec's own pair function "
          f"gives "
          + ", ".join(f"g{g} {r:.3f}" for g, r in enumerate(radii))
          + f" A. **The smallest of the ten is {min(radii):.3f} A, and there "
            f"is no group near zero** -- which is the measurement behind the "
            f"next paragraph. Drift from the written table: "
            f"{ladder_drift or 'none'}, at a tolerance of "
            f"{LADDER_TOLERANCE} A. **The tolerance is itself checked, and the "
            f"noise floor it has to clear is measured here rather than "
            f"assumed:** refitting against a different annulus "
            f"({FIT_ANNULUS_ALT} instead of {FIT_ANNULUS}) moves the ten values "
            f"by at most {radii_noise:.4f} A, and the check requires that to be "
            f"below the tolerance *and* the tolerance to stay under 0.5 A. A "
            f"tolerance wide enough to swallow the numbers is not a check, and "
            f"loosening this one is now a failure it can see. **The bias is a "
            f"single constant the ten agree on, which is a second guard the "
            f"table does not give:** measured minus written is {bias}, a spread "
            f"of {bias_spread:.4f} A across all ten, so the ten agree with each "
            f"other far more closely than the tolerance, and a change to any one "
            f"change to any one group's radius would break the uniformity even "
            f"if the value it changed to stayed inside the tolerance. "
            f"**And the spec's own four values, as differences from group 0:** "
            f"the radius differences cancel any offset common to all ten: "
            f"{', '.join(f'{e} {radii[g] - radii[0]:+.3f} vs {XS_RADIUS[e] - XS_RADIUS[C_GRID_TYPE_ANCHOR]:+.3f}' for g, e in enumerate(('C', 'N', 'O', 'P', 'S')) if e in XS_RADIUS)}"
            f" (drift {spec_drift or 'none'}). Two things that should not matter "
            f"do not, bitwise: a bare `H` gives {h_bare!r} and a two-carbon "
            f"receptor gives {h_two_c!r}, both exactly {h_cross!r}. **And the "
            f"third thing that used to not matter now does:** against the same "
            f"lone carbon an `N` probe crosses at {n_cross:.3f} A and an `O` "
            f"probe at {o_cross:.3f} A, where before the element partition was "
            f"removed neither crossed anywhere in 1-8 A and the bisection "
            f"returned its 8.00 A bound. They cross in the order the spec's XS "
            f"radii predict -- C 1.9 > N 1.75 > O 1.6, so the smaller the "
            f"receptor's radius the further in the probe's own repulsion takes "
            f"over -- and that ordering is the cross-element coupling that was "
            f"missing. **Reported, not diagnosed.** No binding exposes a radius "
            f"and `score_conformation` returns a total, so this file cannot "
            f"invert the ten-group mixture and does not claim the residual "
            f"after the repulsion wall is the spec's g1/g2/hyd terms rather than "
            f"a normalisation; the claim is an ordering and a distance, not a "
            f"decomposition"
    )

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

    # The 0.000000 anomaly this check was written around, measured the only way
    # that isolates it. Two receptor atoms of the *same element* whose atom
    # types put things in different slots: `OA` fills the acceptor slot, `OS`
    # does not, and neither touches any other slot, so their steric fields are
    # the same field. Same probe, same distance, one number's difference.
    nh = Ligand.from_arrays(*hard["donor-acceptor (O...N-H)"][2])
    hb_acceptor = at(nh, rec_of("OA"))
    hb_plain = at(nh, rec_of("OS"))
    hb_row = next(t for t in hard_rows if "donor-acceptor" in t[0])
    # The mutation: if the donor class did not read the acceptor slot, these two
    # would be the same number and the whole claim would be vacuous. That is
    # checked rather than assumed, by asking whether the mask actually has the
    # slot in it -- a rule that cannot fail is not a rule.
    hb_mask_has_acceptor = 2 in SLOT_MASK["donor"]
    check("a donor-acceptor pair no longer scores exactly zero, and the "
          "receptor's acceptor field is what makes it attractive",
          hb_row[3] != 0.0 and hb_row[3] < 0.0
          and hb_acceptor < hb_plain - 0.4
          and hb_mask_has_acceptor,
          f"an OA receptor against an N carrying a polar H, probe classified "
          f"{hb_row[5]!r} at r = {hb_row[1]:.2f} A, gives the engine "
          f"{hb_row[3]:+.6f} where this check used to read exactly 0.000000, "
          f"and the reference {hb_row[4]:+.4f} (decomposed g1 "
          f"{hb_row[6]['g1']:+.4f}, g2 {hb_row[6]['g2']:+.4f}, rep "
          f"{hb_row[6]['rep']:+.4f}, hb {hb_row[6]['hb']:+.4f}). The claim is "
          f"not merely that it is non-zero. **The isolating pair:** the same "
          f"probe at the same distance gives {hb_acceptor:+.6f} against `OA` "
          f"and {hb_plain:+.6f} against `OS` -- same element, so the same "
          f"steric field, and the only map component that differs between them "
          f"is the acceptor slot -- a spread of "
          f"{hb_plain - hb_acceptor:+.4f} kcal/mol that is therefore the "
          f"hydrogen bond, and the only one in the cross-tab below is negative. "
          f"What replaced the zero is not a fifth term: it is the donor class "
          f"reading the acceptor slot, which is a property of the *slot* and "
          f"never was a property of the elements. The reference still "
          f"disagrees with the engine here and this file does not claim "
          f"otherwise; that residual is reported in the same-row comparison "
          f"above rather than explained away")

    # ------------------------------------------------------------------
    section("what the map contains, and which of it is read")

    # The layout and the group index are **data, not prose**, so that the two
    # explanations below can be falsified instead of believed. An earlier draft
    # of this file stated the partition in a docstring and a check *name*; that
    # is prose with a colon, and prose rots silently. Everything the engine is
    # being held to is written here as a table the engine then has to match,
    # and each table is mutated below to prove the match is not a tautology.
    #:
    #: Two rows of the *slot* table were **wrong the first time it was written**
    #: and are the reason it is measured rather than recalled. `OD` populates
    #: slot 1, so "no lone atom puts anything in the donor slot" was false; and
    #: `CG0` -- a type meeko really writes -- is counted as an unknown type,
    #: because the engine reads its first two characters.
    #:
    #: There is deliberately **no group column any more**. This table used to
    #: carry one, asserting that a lone atom lands in group 0, 1, 2 ... by its
    #: element, and that assertion was the file's explanation of the
    #: `0.000000` donor-acceptor cell. It is kept below as `PARTITION_OF`, in
    #: its historical role, purely so the cross-tab check can quote how far it
    #: now misses -- not because it is believed.
    LONE = ("C", "A", "HD", "H", "N", "NA", "ND", "NDA", "O", "OA", "OS",
            "OD", "ODA", "P", "S", "SA", "F", "Cl", "CG0")
    #: **Superseded, kept as the refuted rule.** A lone atom used to be written
    #: into exactly one of the ten groups, this one.
    PARTITION_OF = {"C": 0, "A": 0, "HD": 0, "H": 0,
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
    ALL_GROUPS = list(range(10))
    layout = {t: populated(rec_of(t)) for t in LONE}
    # `populated` returns lists because a receptor can span groups. A lone atom
    # now spans **all ten**, which is the point of this check and the opposite
    #: of what the superseded table above claimed, so the expectation is
    #: written as "every group" and the mutation below is "one group per atom".
    not_all_ten = {t: v[0] for t, v in layout.items() if v[0] != ALL_GROUPS}
    # ... and the ten are not copies. If they were, the peak magnitude of a
    #: lone receptor's steric field would be the same in every group; it is not,
    #: and the ordering follows the ten radii recovered above.
    peaks = {t: [round(float(np.abs(
        rec_of(t).precalculate(PROBE_BOX, "vina").raw_data.reshape(-1, 40)[:, 4 * g]
    ).max()), 4) for g in ALL_GROUPS] for t in ("C", "OA")}
    interchangeable = {t: p for t, p in peaks.items() if len(set(p)) == 1}
    slot_drift = {t: (SLOTS_OF[t], v[1]) for t, v in layout.items()
                  if SLOTS_OF[t] != v[1]}
    # And the element alone fixes the steric field: `N` and `NA` are the same
    # element in different classes, and their maps must be bitwise identical
    # outside the acceptor slot. This is the sharpest available statement that
    # the group index is no longer doing any work.
    na_vs_n = [k for k in range(40) if not np.array_equal(
        rec_of("NA").precalculate(PROBE_BOX, "vina").raw_data.reshape(-1, 40)[:, k],
        rec_of("N").precalculate(PROBE_BOX, "vina").raw_data.reshape(-1, 40)[:, k])]
    na_slots = sorted({k % 4 for k in na_vs_n})
    donors = [t for t, s in SLOTS_OF.items() if 1 in s]
    check("every receptor atom is written into all ten groups, the ten are not "
          "interchangeable, and the slot table is unchanged",
          not not_all_ten and not interchangeable and not slot_drift
          and na_slots == [2] and len(na_vs_n) == 10,
          "; ".join(f"{k} -> groups {len(v[0])}, slots "
                    f"{[SLOT_NAME[s] for s in v[1]]}"
                    for k, v in layout.items())
          + f". **Atoms not spanning all ten groups (must be none): "
            f"{not_all_ten or 'none'}.** That is the regression guard: the ten "
            f"groups used to be an element partition, one group per atom, and "
            f"the `0.000000` donor-acceptor cell was its consequence. **The ten "
            f"are not copies** -- a lone carbon's peak steric magnitude per "
            f"group is "
          + ", ".join(f"{p:.2f}" for p in peaks["C"])
          + f" across the ten (all identical in "
            f"{interchangeable or 'no group'}, which would mean the groups "
            f"carry the same radius and the ten-radius mixture is not real), and "
            f"the ordering is the radius ladder recovered in the previous "
            f"section. **Slot drift against the table: "
            f"{slot_drift or 'none'}** -- slot 0 is populated by every atom, "
            f"slot 2 by the acceptors, slot 1 ({SLOT_NAME[1]}) by {donors} "
            f"only, and none of that moved, which is the point: the fix changed "
            f"which *groups* an atom reaches, not what a slot means. **And the "
            f"element alone now fixes the steric field:** `NA` and `N` differ "
            f"in {len(na_vs_n)} components, all in slot {na_slots}, which is "
            f"one slot in each of the ten groups -- so two receptor atoms of "
            f"the same element have bitwise identical maps outside the one slot "
            f"their classes differ in. `NDA` reproduces `ND` exactly and `ODA` "
            f"reproduces `OD` exactly, slots both, which is the "
            f"three-character truncation seen through the map rather than "
            f"through `atom_kinds`; `CG0` is counted as an unknown type "
            f"({rec_of('CG0').unknown_atom_types}), so a type meeko really "
            f"writes is mistyped as `CG`")

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
    # `O` and `OS` join the list so the table contains a **same-element family
    # of three receptor types** (`O`, `OA`, `OS`) as well as the nitrogen pair.
    # That is what makes the replacement rule below testable: with only one
    # type per element there is nothing for "the class does not matter" to be
    # wrong about.
    receptors = ["C", "N", "NA", "O", "OA", "OS", "S", "Cl"]
    #: Receptor families of one element, split by whether the type fills a
    #: class slot. The claim is that the two halves of each family are
    #: bitwise indistinguishable to a probe that does not read that slot.
    ELEMENT_FAMILIES = {"N": ("N", "NA"), "O": ("O", "OA", "OS")}

    def groups_of(probe_name, group_of):
        """Which map groups a probe sums, under `group_of`."""
        if group_of is None:                       # every group: the true rule
            return set(range(10))
        return {group_of[PROBE_TYPE[probe_name][i]]
                for i in range(len(PROBE_TYPE[probe_name]))}

    def predicts_zero(probe_name, rec_type, group_of):
        """Does `group_of` say this cell must be exactly zero?

        Kept, in its historical role, so the table can be held to the rule
        that used to explain it. That rule is **superseded**: the section
        above measures every lone atom in all ten groups.
        """
        if group_of is None:
            return False
        return group_of[rec_type] not in groups_of(probe_name, group_of)

    tab = {(p, r): at(probes[p], rec_from([(r, (0, 0, 0), r)]))
           for p in probes for r in receptors}
    cross_zero = sorted(f"{p}/{r}" for (p, r), v in tab.items() if v == 0.0)
    cross_live = sorted(f"{p}/{r}={v:+.4f}"
                        for (p, r), v in tab.items() if v != 0.0)
    donor_row = {r: tab[("N+H", r)] for r in receptors}
    negative = sorted(f"{p}/{r}={v:+.6f}"
                      for (p, r), v in tab.items() if v < 0.0)

    # --- the superseded rule, and how far it now misses -------------------
    def _wrong_cells(group_of):
        return sum(1 for (p, r), v in tab.items()
                   if (v == 0.0) != predicts_zero(p, r, group_of))

    refuted = {
        "superseded: probe sums its own element's group": _wrong_cells(
            PARTITION_OF),
        "a probe sums all ten groups": _wrong_cells(None),
    }

    # --- the rule that replaced it ----------------------------------------
    # The element is no longer a gate. What predicts whether two receptor types
    # of the same element are distinguishable is the **slot** their classes
    # differ in, and whether a probe can see it is which slots its class reads.
    #: The one slot each family's members differ in: `NA` fills the acceptor
    #: slot and `N` does not, `OA` fills it and `O`/`OS` do not.
    FAMILY_SLOT = {"N": 2, "O": 2}

    def predicts_distinguishable(probe_name, family, mask):
        """Must this probe tell the two members of `family` apart?"""
        return FAMILY_SLOT[family] in mask[PROBE_CLASS[probe_name]]

    #: The class each probe is *asserted* to have, cross-checked against
    #: `atom_kinds` in the next check. Two of the five are `other`, in different
    #: element groups, which is a useful case rather than a duplicate.
    PROBE_CLASS = {"C": "hydrophobic", "N+H": "donor",
                   "OA (via file)": "acceptor", "N": "other", "O": "other"}
    fam_drift = {pn: (PROBE_CLASS[pn], probes[pn].atom_kinds[0])
                 for pn in probes if PROBE_CLASS[pn] != probes[pn].atom_kinds[0]}
    fam_cells = {}
    for fam, members in ELEMENT_FAMILIES.items():
        for pn in probes:
            vals = [tab[(pn, m)] for m in members]
            fam_cells[(pn, fam)] = (len(set(vals)) == 1, vals)
    fam_wrong = sorted(
        f"{pn}/{fam}" for (pn, fam), (identical, _) in fam_cells.items()
        if identical == predicts_distinguishable(pn, fam, SLOT_MASK))

    def _fam_wrong(mask):
        return sum(1 for (pn, fam), (identical, _) in fam_cells.items()
                   if identical == predicts_distinguishable(pn, fam, mask))

    fam_muts = {
        "mask as written": _fam_wrong(SLOT_MASK),
        "donor stops reading the acceptor slot": _fam_wrong(
            {**SLOT_MASK, "donor": {0}}),
        "`other` starts reading every slot": _fam_wrong(
            {**SLOT_MASK, "other": {0, 1, 2, 3}}),
        "acceptor starts reading the acceptor slot": _fam_wrong(
            {**SLOT_MASK, "acceptor": {0, 1, 2}}),
    }
    check("no cell in the table is zero any more, the element partition that "
          "explained them is refuted in every cell it claimed, and what "
          "replaces it is the slot rather than the element",
          not cross_zero and not fam_wrong and not fam_drift
          and refuted["a probe sums all ten groups"] == 0
          and refuted["superseded: probe sums its own element's group"] > 0
          and all(m > 0 for k, m in fam_muts.items() if "as written" not in k),
          f"at {PROBE_AT} A the engine gives **{len(cross_zero)} zero cells of "
          f"{len(tab)}**, where the rule this file was built around predicted "
          f"{sum(1 for p in probes for r in receptors if predicts_zero(p, r, PARTITION_OF))}"
          f". Non-zero: {', '.join(cross_live)}. Probe classes: "
          + ", ".join(f"{n}->{l.atom_kinds}" for n, l in probes.items())
          + f". **How far the old rule now misses, counted as cells where it "
            f"and the engine disagree:** "
          + "; ".join(f"{k} -> {v}" for k, v in refuted.items())
          + f". It is not merely unused, it is wrong in every cell it once got "
            f"right, which is what a refuted rule should look like. **The "
            f"replacement, stated so it can fail:** two receptor types of the "
            f"same element are bitwise indistinguishable to a probe whose class "
            f"does not read the one slot their classes differ in, and "
            f"distinguishable to a probe that does. Measured across "
          + "; ".join(
              f"{fam} ({'/'.join(ELEMENT_FAMILIES[fam])}) via {pn} "
              f"[{PROBE_CLASS[pn]}] -> "
              f"{'INDISTINGUISHABLE' if ident else 'distinguished'}"
              for (pn, fam), (ident, _) in sorted(fam_cells.items()))
          + f"; cells where that and the engine disagree: {fam_wrong or 'none'}"
            f"; fixture-class drift: {fam_drift or 'none'}. **Mutations, same "
            f"counting:** "
          + "; ".join(f"{k} -> {v}" for k, v in fam_muts.items())
          + f". The N+H row is where the one survives: "
          + ", ".join(f"{r} {donor_row[r]:+.4f}" for r in receptors)
          + f", and the only negative cell in the entire table is "
          + (", ".join(negative) or "none")
          + f". A nitrogen-oxygen hydrogen bond is now expressible, which is "
            f"the whole of what the `0.000000` was saying was not")

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
    #: Receptor pairs that differ in exactly one **slot** -- and now that is the
    #: only thing that can be held constant, because a lone atom is written
    #: into all ten groups, so a one-*component* pair no longer exists: the
    #: same slot in each of the ten. Five pairs, covering both single-slot
    #: families and both directions of the rule.
    ONE_SLOT_PAIRS = [("NA", "N", 2), ("OA", "O", 2), ("OA", "OS", 2),
                      ("ND", "N", 1), ("OD", "OS", 1)]
    #: ... and two pairs differing in two slots, used to show a probe can
    #: respond through either of them.
    TWO_SLOT_PAIRS = [("OD", "OA", (1, 2)), ("ND", "NA", (1, 2))]

    def _map_of(t):
        return rec_of(t).precalculate(PROBE_BOX, "vina").raw_data.reshape(-1, 40)

    pair_evidence = {}
    pair_groups = {}
    for a_t, b_t, slots in ONE_SLOT_PAIRS + TWO_SLOT_PAIRS:
        want = (slots,) if isinstance(slots, int) else slots
        ma, mb = _map_of(a_t), _map_of(b_t)
        differing = tuple(k for k in range(40)
                          if not np.array_equal(ma[:, k], mb[:, k]))
        responses = {pn: (at(probes[pn], rec_of(a_t))
                          != at(probes[pn], rec_of(b_t)))
                     for pn in probes}
        pair_evidence[(a_t, b_t)] = (tuple(sorted({k % 4 for k in differing})),
                                     want, responses)
        pair_groups[(a_t, b_t)] = len({k // 4 for k in differing})

    # The rule, in one condition: a probe's score changes for a receptor pair if
    # and only if one of the differing **slots** is a slot that probe's class
    # reads. There is no element term and no group term, and that is the
    # finding rather than a simplification -- the group condition used to be
    # here, and it is what the engine stopped obeying. The first version of
    # this compared a group *index* against a *set* of them with `!=`, which is
    # always true, so the mask was never consulted and every mutation of it
    # reported the same count. That is why the condition is written as a
    # membership test and why the mask mutations below are expected to differ
    # from one another rather than merely to be non-zero.
    def predicts_responds(probe_name, differing_slots, mask):
        """Does the rule say a probe's score must differ for this receptor pair?"""
        return any(s in mask[PROBE_CLASS[probe_name]] for s in differing_slots)

    def _resp_wrong(mask):
        out = {}
        for (a_t, b_t), (differ, _, responses) in pair_evidence.items():
            for pn, got in responses.items():
                want = predicts_responds(pn, differ, mask)
                if got != want:
                    out[f"{pn}:{a_t}/{b_t}"] = (
                        f"predicted {'RESPONDS' if want else 'no change'}",
                        f"got {'RESPONDS' if got else 'no change'}", differ)
        return out

    resp_wrong = _resp_wrong(SLOT_MASK)
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
    #: The donor slot is read by *something* -- the headline of this check, and
    #: the one thing here that is not already implied by the rule, since the
    #: rule decides only *which* probes respond, not whether any do. The count
    #: is deliberately not asserted: the rule plus a zero disagreement count
    #: already fixes it, so pinning a number would be a second copy of the same
    #: fact that could go stale on its own.
    donor_slot_read_by_someone = [p for p in donor_reads
                                  if 1 in pair_evidence[(p[0], p[1])][0]]

    # --- the fifth class, on the project's own shipped pose ---------------
    # `donoracceptor` needs a bonded polar hydrogen, so no one-atom ligand can
    # be it, and an earlier revision of this file declared the class uncovered
    # on purpose rather than guess a mask for it. It does not have to be a
    # guess: the repository ships a 16-atom pose with exactly one such atom, and
    # the receptor can be retyped so that the *only* map component that changes
    # is the acceptor slot. The pair of measurements below is the whole
    # argument -- the delta is real, and removing the one atom that should own
    # it takes the delta to exactly zero, which no other atom in the pose
    # could have done if the mask were wrong.
    _pose_box_half = 12.0

    def _pose_parts(drop: int | None = None):
        rec_body = [l for l in (ROOT / "examples" / "1crn_prep.pdbqt").read_text(
            encoding="utf-8").splitlines() if l.startswith(("ATOM", "HETATM"))]
        lig_body = [l for l in (ROOT / "examples" / "crambin_pose.pdbqt").read_text(
            encoding="utf-8").splitlines() if l.startswith(("ATOM", "HETATM"))]
        # MODEL 1 of the pose is the first sixteen ATOM lines; the rest belong
        # to the other models and are not this molecule.
        lig_body = lig_body[:16]
        if drop is not None:
            lig_body = [l for i, l in enumerate(lig_body) if i != drop]
        xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                        for l in lig_body], float)
        centre = xyz.mean(axis=0)
        box = GridBox(centre - _pose_box_half, centre + _pose_box_half)
        lg = Ligand.from_pdbqt_str("\n".join(lig_body) + "\nEND\n")
        return rec_body, lg, box

    def _retype(lines, old, new):
        return [l[:77] + new.rjust(2)
                if l.startswith(("ATOM", "HETATM")) and l[77:79].strip() == old
                else l for l in lines]

    def _pose_score(rec_lines, drop=None):
        _, lg, box = _pose_parts(drop)
        rec = Receptor.from_pdbqt_str("\n".join(rec_lines) + "\nEND\n")
        maps = rec.precalculate(box, "vina")
        return float(score_conformation(lg, maps, np.zeros(lg.num_dof))[0])

    _rec_body, _pose_lig, _ = _pose_parts()
    _pose_kinds = _pose_lig.atom_kinds
    _da_atoms = [i for i, k in enumerate(_pose_kinds) if k == "donoracceptor"]
    _pose_base = _pose_score(_rec_body)
    #: Emptying the receptor's acceptor slot: `OA` -> `OS` is the same element
    #: and the same steric field, so this isolates the acceptor slot exactly.
    _acceptor_delta = (_pose_score(_retype(_rec_body, "OA", "OS")) - _pose_base)
    #: The control: `N` -> `NS` empties no slot, so it must move nothing. A
    #: delta here would mean the "isolating" retype was not isolating.
    _null_delta = _pose_score(_retype(_rec_body, "N", "NS")) - _pose_base
    #: And the attribution: with the pose's single `donoracceptor` atom
    #: removed, no atom is left that reads the acceptor slot, so the delta must
    #: be exactly zero.
    _acceptor_delta_no_da = (_pose_score(
        _retype(_rec_body, "OA", "OS"), drop=_da_atoms[0]) - _pose_score(
            _rec_body, drop=_da_atoms[0]))
    _da_ok = (len(_da_atoms) == 1
              and abs(_acceptor_delta) > 0.5
              and _acceptor_delta_no_da == 0.0
              and _null_delta == 0.0
              and _pose_kinds[_da_atoms[0]] == "donoracceptor")
    check("the whole rule is one line and it has no element in it: a probe "
          "responds to a receptor pair if and only if one of the differing "
          "slots is a slot its class reads -- and the fifth class is now "
          "measured on the shipped pose rather than left uncovered",
          not resp_wrong and not class_drift and _da_ok
          and all(pair_evidence[k][0] == pair_evidence[k][1]
                  for k in pair_evidence)
          and set(pair_groups.values()) == {10}
          and len(donor_slot_read_by_someone) > 0
          and all(m == 0 if k == "mask as written" else m > 0
                  for k, m in slot_muts.items()),
          "; ".join(
              f"{a_t} vs {b_t} differ in slot {list(differ)} "
              f"(in {pair_groups[(a_t, b_t)]} of 10 groups) -> "
              + ", ".join(f"{pn}[{PROBE_CLASS[pn]}]"
                          f"{' RESPONDS' if r else ' no change'}"
                          for pn, r in responses.items())
              for (a_t, b_t), (differ, _, responses)
              in pair_evidence.items())
          + f". Fixture classes, claimed against `atom_kinds`: "
          + ", ".join(f"{pn}={probes[pn].atom_kinds[0]}" for pn in probes)
          + f"; drift {class_drift or 'none'}. The decisive pairs are the ones "
            f"that isolate a **single** slot -- an acceptor probe against "
            f"`OS` vs `O` and against `OD`, a donor probe against `NA` vs `N`: "
          + "; ".join(f"{a_t}/{b_t} {at(probes[pn], rec_of(a_t)):+.6f} vs "
                      f"{at(probes[pn], rec_of(b_t)):+.6f}"
                      for a_t, b_t, pn in (("OA", "O", "OA (via file)"),
                                           ("OD", "OS", "OA (via file)"),
                                           ("NA", "N", "N+H"),
                                           ("OA", "OS", "N+H")))
          + f". **This corrects an earlier revision of this file**, which "
            f"concluded the donor slot was read by nothing on the evidence of "
            f"five oxygen receptors scoring bitwise identically. They did, and "
            f"the probe was the problem: it was an `other`-class oxygen, and "
            f"`other` reads the steric slot only. So that finding is withdrawn. "
            f"**And the element condition is gone too:** this check used to "
            f"carry a second precondition -- the receptor must be in the "
            f"probe's own element group -- and it was that precondition, not "
            f"the mask, that made a nitrogen-oxygen hydrogen bond inexpressible. "
            f"Cells where the mask and the engine disagree: "
            f"{resp_wrong or 'none'}. **Mutations, counted the same way:** "
          + "; ".join(f"{k} -> {v}" for k, v in slot_muts.items())
          + f". **The fifth class, which an earlier revision of this file "
            f"declared uncovered on purpose, is now covered.** Every fixture "
            f"above is a one-atom ligand, so none of them can be "
            f"`donoracceptor`, which needs a bonded polar hydrogen -- but "
            f"`examples/crambin_pose.pdbqt` MODEL 1 has exactly one: O15, "
            f"typed `OA` like O14, differing only in whether the torsion tree "
            f"bonds H16 to it, and the pose's classes are "
            f"{collections.Counter(_pose_kinds)}. Relabelling every receptor "
            f"`OA` to `OS` -- same element, same steric field, and the only map "
            f"component that changes is the acceptor slot -- moves the pose "
            f"from {_pose_base:+.6f} to "
            f"{_pose_base + _acceptor_delta:+.6f}, a delta of "
            f"{_acceptor_delta:+.6f} kcal/mol. The two controls are what make "
            f"that a measurement of the class rather than of the pose: "
            f"retyping every receptor `N` to `NS`, which empties no slot, moves "
            f"it by {_null_delta:+.6f}; and deleting the one `donoracceptor` "
            f"atom leaves the other 15, of which 13 are `hydrophobic`, 1 "
            f"`acceptor` and 1 `other`, and the delta becomes exactly "
            f"{_acceptor_delta_no_da:+.6f}. So the mask row "
            f"{sorted(SLOT_MASK['donoracceptor'])} for that class is measured, "
            f"and the carboxylate oxygen reads the receptor's acceptor field, "
            f"which the one-atom table above could neither show nor deny. "
            f"**What the ten-group replication did to it: nothing.** The "
            f"pre-fix engine wrote that acceptor field once and read it once; "
            f"it is now written into all ten groups and the delta is the same "
            f"magnitude, so no term was rescaled by being replicated. **Still "
            f"not covered, and named rather than hidden:** what the *weight* "
            f"applied to a populated slot is. The score-time normalisation of "
            f"SCORING.md 9.1 is not inverted anywhere in this file, so every "
            f"number above is a total"
          )

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
            f"a file and cannot be an acceptor in memory at all. **An earlier "
            f"revision of this check ended there, on the claim that nothing in "
            f"tree is broken because every caller writes a file before scoring. "
            f"That was wrong twice over. The caller did not write a file, and "
            f"the next two checks are why; and when it was finally made to, the "
            f"in-memory path was changed to match rather than the file path "
            f"changed to match the in-memory one.**")

    # --- is the type-losing entry point actually reachable? ----------------
    #
    # Measured, not inferred, and the answer changed the previous paragraph.
    # `cli.py` docks whatever `load_ligand` returns, and `load_ligand` routes
    # every non-PDBQT format through `from_arrays`. So a `.sdf` reaches the
    # engine as a ligand with no acceptors at all, and the file round trip that
    # "saves" it is only taken by a *different* command.
    #
    # `bonds=None` is used deliberately rather than a reconstructed bond list:
    # the check above already established that bonds do not restore an acceptor,
    # so leaving them out makes this claim hold under weaker conditions, and it
    # keeps the check free of an RDKit dependency.
    #
    # The elements come from a **type-to-element** map, not from the first
    # character of the type column. Taking the first character feeds `from_arrays`
    # an element of "A" for every aromatic carbon, which it rejects outright --
    # and `prepare_ligand`, which is what the live path actually calls, hands it
    # "C". The first draft of this check raised `ValueError: unsupported element
    # "A"`, which is a fact worth recording but not the one under test.
    _TYPE_ELEMENT = {"A": "C", "OA": "O", "OS": "O", "NA": "N", "NS": "N",
                     "HD": "H", "SA": "S", "CL": "Cl", "BR": "Br", "MG": "Mg",
                     "ZN": "Zn", "FE": "Fe", "MN": "Mn", "CA": "Ca",
                     "CU": "Cu", "NI": "Ni", "CO": "Co", "CD": "Cd",
                     "HG": "Hg", "SI": "Si"}
    _pdbqt_lines = (ROOT / "examples" / "ibuprofen_prep.pdbqt").read_text(
        encoding="utf-8").splitlines()
    _el, _cr, _nm = [], [], []
    for _l in _pdbqt_lines:
        if _l.startswith(("ATOM", "HETATM")):
            _tok = _l[77:79].strip()
            _el.append(_TYPE_ELEMENT.get(_tok.upper(), _tok[0] if _tok else "C"))
            _cr.append([float(_l[30:38]), float(_l[38:46]), float(_l[46:54])])
            _nm.append(_l[12:16].strip())
    _via_file = Ligand.from_pdbqt_str(
        (ROOT / "examples" / "ibuprofen_prep.pdbqt").read_text(encoding="utf-8"))
    _via_arrays = Ligand.from_arrays(
        _el, [0.0] * len(_el), np.array(_cr, dtype=float), None, _nm)
    _file_classes = collections.Counter(_via_file.atom_kinds)
    _array_classes = collections.Counter(_via_arrays.atom_kinds)
    _n_acc = {k: v for k, v in
              (("file", sum(_file_classes[k] for k in
                            ("acceptor", "donoracceptor") if k in _file_classes)),
               ("from_arrays", sum(_array_classes[k] for k in
                                   ("acceptor", "donoracceptor")
                                   if k in _array_classes)))}
    # What the missing channel is worth, in kcal/mol, on a real receptor. The
    # two ligands are the same 16 atoms at the same coordinates and are scored
    # in the same field, so the difference between them is the acceptor channel
    # and nothing else -- no control is needed, because there is no second
    # variable.
    #
    # The pose used is the shipped crambin one rather than the ibuprofen file
    # above, and that is a fixture requirement rather than a preference: the
    # ibuprofen coordinates are not a crambin pose, so dropping them into
    # crambin's field puts the ligand inside the repulsion wall where every
    # term is swamped and **both** variants score identically to six decimals.
    # A number that cannot distinguish the thing it is measuring is not a
    # measurement of it, and the first attempt at this reported +0.000000 for
    # exactly that reason.
    _rec_lines = [l for l in (ROOT / "examples" / "1crn_prep.pdbqt").read_text(
        encoding="utf-8").splitlines() if l.startswith(("ATOM", "HETATM"))]
    _pose_lines = [l for l in (ROOT / "examples" / "crambin_pose.pdbqt").read_text(
        encoding="utf-8").splitlines() if l.startswith(("ATOM", "HETATM"))][:16]
    _p_el, _p_cr, _p_nm = [], [], []
    for _l in _pose_lines:
        _tok = _l[77:79].strip()
        _p_el.append(_TYPE_ELEMENT.get(_tok.upper(), _tok[0] if _tok else "C"))
        _p_cr.append([float(_l[30:38]), float(_l[38:46]), float(_l[46:54])])
        _p_nm.append(_l[12:16].strip())
    _pose_typed = Ligand.from_pdbqt_str("\n".join(_pose_lines) + "\nEND\n")
    _pose_untyped = Ligand.from_arrays(_p_el, [0.0] * len(_p_el),
                                       np.array(_p_cr, dtype=float), None, _p_nm)
    _pm = Receptor.from_pdbqt_str(
        "\n".join(_rec_lines) + "\nEND\n").precalculate(
            GridBox(np.array(_p_cr, dtype=float).mean(axis=0) - 12.0,
                    np.array(_p_cr, dtype=float).mean(axis=0) + 12.0), "vina")
    _s_file = float(score_conformation(
        _pose_typed, _pm, np.zeros(_pose_typed.num_dof))[0])
    _s_arrays = float(score_conformation(
        _pose_untyped, _pm, np.zeros(_pose_untyped.num_dof))[0])
    _acceptor_cost = _s_file - _s_arrays
    check(f"the limitation is `from_arrays`' own and still costs "
          f"{_acceptor_cost:+.3f} kcal/mol: the same molecule has no acceptor "
          f"through it and {_n_acc['file']} through the file reader",
          _n_acc["file"] > 0 and _n_acc["from_arrays"] == 0
          and _file_classes != _array_classes
          and len(_el) == _via_file.num_atoms == _via_arrays.num_atoms
          and len(_p_el) == _pose_typed.num_atoms == _pose_untyped.num_atoms
          and _s_file == _pose_base
          and abs(_acceptor_cost) > 0.1,
          f"ibuprofen, {len(_el)} atoms, via `from_arrays` with "
          f"bonds=None: {dict(sorted(_array_classes.items()))}; via the file "
          f"reader: {dict(sorted(_file_classes.items()))}. Acceptors "
          f"{_n_acc['from_arrays']} -> {_n_acc['file']}. The two carboxyl "
          f"oxygens are `other` through from_arrays, and an `other` atom reads "
          f"the steric slot and nothing else, so this is not a milder "
          f"classification but a different one. **What it costs:** the shipped "
          f"crambin pose, {len(_p_el)} atoms at identical "
          f"coordinates, in the field of `examples/1crn_prep.pdbqt` in a 24 A "
          f"box centred on the ligand, scores {_s_file:+.6f} through the file "
          f"reader and {_s_arrays:+.6f} through `from_arrays` -- a difference "
          f"of {_acceptor_cost:+.6f} kcal/mol, "
          f"{abs(_acceptor_cost) / abs(_s_file) * 100:.0f}% of the score, for "
          f"one carboxylate oxygen that is an `acceptor` in one case and `other` "
          f"in the other. The typed score is bitwise the same number the "
          f"previous section measured for the same pose and the same box, which "
          f"is asserted rather than hoped for. **What has changed, and what has "
          f"not:** the number is unchanged, because `from_arrays` is unchanged "
          f"-- it still cannot type an atom. What changed is reachability. This "
          f"check used to be titled \"the type-losing entry point is "
          f"reachable\", and it was: `rec-grid dock something.sdf` did reach the "
          f"engine through `from_arrays` with no acceptors, for exactly this "
          f"cost. `load_ligand` and `prep-ligand` no longer ask it to -- the "
          f"next check is that equality -- so this is now the price of calling "
          f"`from_arrays` **directly**, a statement about the API rather than "
          f"about what docking a `.sdf` does. **The shape of the "
          f"error also changed when the engine's element partition was removed:** "
          f"before that fix a cross-element hydrogen bond was worth exactly "
          f"0.000000 for *everyone*, so this difference was 0.000000 too and "
          f"the defect was harmless in tree. It is the full price of the missing "
          f"acceptor channel")

    _cli_src = (ROOT / "dock-py" / "python" / "opendocking" / "cli.py").read_text(
        encoding="utf-8")
    _core_src = (ROOT / "dock-py" / "python" / "opendocking" / "core.py").read_text(
        encoding="utf-8")
    _dock_fn = _cli_src[_cli_src.index("def _cmd_dock"):]
    _dock_fn = _dock_fn[:_dock_fn.index("\ndef ")] if "\ndef " in _dock_fn \
        else _dock_fn
    #: `_cmd_prep_ligand`'s **body**, cut the same way, because a regex that runs
    #: to the end of the file cannot tell a function that uses the round trip
    #: from one that merely imports it. The first version of the link below was
    #: `re.search(r"def _cmd_prep_ligand.*?typed_ligand_from_tables", src, re.S)`,
    #: and a mutation that replaced the call but left the import passed it: the
    #: name still appeared, just on the import line. Found by mutating the fix
    #: rather than by reading it, which is the only reason it is now bounded to
    #: the body and requires a call rather than a mention.
    _prep_fn = _cli_src[_cli_src.index("def _cmd_prep_ligand"):]
    _prep_fn = _prep_fn[:_prep_fn.index("\ndef ")] if "\ndef " in _prep_fn \
        else _prep_fn
    # --- the fix, and the guards that hold it in place --------------------
    #
    # `from_arrays` still cannot type an atom, and that is now a property of
    # the API rather than a defect in the tool: `load_ligand` and
    # `prep-ligand` carry the types across the boundary through the type
    # column, which is what `load_receptor` has always done for receptors
    # (`core.py:895`, fifteen lines above the function that did not).
    #
    # The guard is an **equality**, not a threshold, because that is what the
    # fix actually is: for the same molecule, the prepared path and the file
    # path must produce the same interaction classes. Everything below exists
    # to make that equality hard to satisfy by accident.
    _links = {
        "cli._cmd_dock calls load_ligand":
            bool(re.search(r"ligand\s*=\s*load_ligand\(", _dock_fn)),
        "cli._cmd_dock passes that ligand to dock()":
            bool(re.search(r"dock\(\s*\n?\s*ligand\s*,", _dock_fn)),
        "core.load_ligand no longer hands prepare_ligand to from_arrays":
            not re.search(r"def load_ligand.*?from_arrays\(\s*\*\s*prepare_ligand",
                          _core_src, re.S),
        "cli._cmd_prep_ligand *calls* the shared round trip, not just imports it":
            bool(re.search(r"typed_ligand_from_tables\s*\(", _prep_fn))
            and "from_arrays(" not in _prep_fn,
        "core.typed_ligand_from_tables is the one implementation both reach":
            bool(re.search(
                r"def typed_ligand_from_tables.*?ligand_to_pdbqt.*?"
                r"Ligand\.from_pdbqt_str", _core_src, re.S)),
        "prep-ligand still prints the classes it built":
            "atom classes" in _prep_fn,
    }

    #: Molecules with a shipped `_prep.pdbqt` beside them. The shipped file is
    #: the ground truth because it is what the project's own `prep-ligand
    #: --output` produced, so a difference is a difference from the artifact
    #: the repository already ships. `polar` says whether the molecule is
    #: *supposed* to have acceptor-class atoms; the two that are not are the
    #: negative control, and they are what stops the equality below from being
    #: satisfiable by typing everything as an acceptor.
    _PAIRS = {"ibuprofen": True, "phenol": True, "biotin": True,
              "benzene": False, "toluene": False}
    _equality, _byte_lines = {}, {}
    for name, polar in _PAIRS.items():
        _sdf = ROOT / "examples" / f"{name}.sdf"
        _ref = ROOT / "examples" / f"{name}_prep.pdbqt"
        _via_sdf = load_ligand(_sdf)
        _via_file = load_ligand(_ref)
        _acc = sum(1 for k in _via_sdf.atom_kinds
                   if k in ("acceptor", "donoracceptor"))
        _equality[name] = (
            _via_sdf.atom_kinds == _via_file.atom_kinds,
            dict(sorted(collections.Counter(_via_sdf.atom_kinds).items())),
            _acc, polar,
            _via_sdf.num_torsions == _via_file.num_torsions)
        # The engine-independent half: the ATOM lines the round trip writes are
        # the same text the shipped file contains. This compares strings and
        # needs no ligand, no map and no score, so it cannot be satisfied by an
        # engine change that happens to make both sides agree.
        _byte_lines[name] = (
            _atom_lines(_round_trip_text(*prepare_ligand(_sdf)))
            == _atom_lines(_ref.read_text(encoding="utf-8")))

    _wrong = {n: v[1] for n, v in _equality.items() if not v[0]}
    _tree_wrong = {n: v[1] for n, v in _equality.items() if not v[4]}
    #: The control, stated as a prediction the engine has to agree with rather
    #: than as an absence: a polar molecule must have acceptor-class atoms, a
    #: hydrocarbon must have none. A fix that typed every oxygen as an acceptor
    #: satisfies the first half and fails the second.
    _control_wrong = {n: v[1] for n, v in _equality.items()
                      if (v[2] > 0) != v[3]}

    # --- the three mutations, counted -------------------------------------
    # The mutators are module-level, next to the carrier helper they break, so
    # that the byte-identity guard below and the mutation counts here are held
    # to the same three definitions rather than to two copies of them.
    def _classes_of(text):
        try:
            return Ligand.from_pdbqt_str(text).atom_kinds
        except Exception as exc:                     # noqa: BLE001
            return f"raised {type(exc).__name__}"

    _mutations = {}
    for label, mutate in (("type column blanked", _mut_blank_type),
                          ("acceptor typed as plain O", _mut_flatten_acceptor),
                          ("polar hydrogen dropped", _mut_drop_polar_h)):
        caught = []
        for name in _PAIRS:
            ref = Ligand.from_pdbqt_str(
                (ROOT / "examples" / f"{name}_prep.pdbqt").read_text(
                    encoding="utf-8"))
            caught.append(_classes_of(mutate(_round_trip_text(
                *prepare_ligand(ROOT / "examples" / f"{name}.sdf"))))
                != ref.atom_kinds)
        _mutations[label] = f"{sum(caught)}/{len(_PAIRS)}"

    check("the prepared path and the file path now agree exactly, three ways of "
          "getting there wrong are each caught, and a hydrocarbon is left alone",
          all(_links.values())
          and not _wrong and not _tree_wrong and not _control_wrong
          and all(v for v in _byte_lines.values())
          and all(m != f"0/{len(_PAIRS)}" for m in _mutations.values()),
          "; ".join(f"{k} = {v}" for k, v in _links.items())
          + ". **The equality, which is the whole of the fix:** for the same "
            "molecule, `load_ligand(\"x.sdf\")` and "
            "`load_ligand(\"x_prep.pdbqt\")` return the same `atom_kinds` -- "
          + "; ".join(
              f"{n} {'==' if v[0] else '!='} ({v[2]} acceptor-class, "
              f"{'polar' if v[3] else 'HYDROCARBON'}, torsions "
              f"{'same' if v[4] else 'CHANGED'}) {v[1]}"
              for n, v in _equality.items())
          + f". Molecules where they disagree: {_wrong or 'none'}; molecules "
            f"whose rotatable-bond count the round trip changed: "
            f"{_tree_wrong or 'none'}. **The guard cannot be satisfied by "
            f"typing everything as an acceptor:** the three polar molecules "
            f"have acceptor-class atoms and the two hydrocarbons have none, so "
            f"both halves have to hold; a fix that typed every oxygen as an "
            f"acceptor fails the hydrocarbon half "
            f"({_control_wrong or 'control clean'}). **Engine-independent "
            f"half:** the ATOM lines the round trip writes are the same text "
            f"the shipped `_prep.pdbqt` contains, per molecule "
          + ", ".join(f"{n}={'same' if v else 'DIFFER'}" for n, v
                      in _byte_lines.items())
          + " -- a string comparison that no engine change can make pass for "
            "both sides at once. **Mutations, counted as molecules whose "
            "classes stop matching once the carrier is broken:** "
          + "; ".join(f"{k} -> {v}" for k, v in _mutations.items())
          + ". **The limitation of this route, recorded in "
            "`core.typed_ligand_from_tables` rather than only here:** the "
            "writer's type field is two columns wide, so a three-character "
            "type cannot pass. That is unobservable today -- `NDA` and `ND` are "
            "bitwise identical ligands, as are `ODA` and `OD` -- and if the "
            "project ever needs one, the alternative is an optional "
            "`atom_types` on `from_arrays` in the PyO3 binding and a rebuilt "
            "extension. **`from_arrays` itself is unchanged and still cannot "
            "type an atom**; what changed is that the docking path no longer "
            "asks it to"
    )


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
          "SUPERSEDED BY THE SECTION BELOW, and kept as the record of what used "
          "to be true. score_conformation returns (energy, gradient); "
          "evaluate_conformations returns totals; DockingResult carries "
          "intermolecular_energies per pose. All scalars. The reference "
          "decomposes its own terms and the engine is asked only for the sum, "
          "so this file established that the SUM matched the spec -- it did not "
          "establish each term separately. **That is no longer the case**: "
          "`score_conformation_terms` now asks the engine for the five terms "
          "directly, and the section below compares each one separately. The "
          "claim this check made is now false, and it is left visible rather "
          "than deleted because the withdrawal below had to be earned against "
          "it")

    # ------------------------------------------------------------------
    section("each term, separately, against the reference that wrote it down")

    # # Why this section exists, and what it is allowed to assume
    #
    # Everything above compares a SUM. A sum can be right for the wrong reasons:
    # one term too attractive and another too repulsive cancel exactly and the
    # whole suite stays green. This is the same shape as the grid defect fixed
    # earlier in this file, where every receptor atom was written into one of
    # ten per-element blocks and the error was invisible because the whole field
    # was wrong consistently.
    #
    # Two properties of the *question* decide how tightly the answer can be
    # checked, and both are chosen here rather than inherited:
    #
    # 1. **Both atoms sit on grid nodes.** The engine scores by trilinear
    #    interpolation of a 0.375 Å grid, so an atom placed between nodes
    #    blends four field values and carries interpolation error of order 1e-2
    #    -- the same floor the total comparison above lives on. Placing both
    #    atoms *on* nodes makes the interpolation exact (one corner has weight
    #    1), and what is left is the `f32` rounding of the tabulated field. That
    #    is what turns a per-term tolerance of 1e-2 into 1e-6, and without it
    #    no single-term perturbation small enough to hide inside the sum-level
    #    tolerance could ever be caught here.
    # 2. **The reference classifies the PAIR, not the probe.** `reference_pair`
    #    above takes `apolar=`/`donor=` from the probe alone, which is wrong in
    #    general: SCORING.md section 4 gates `hyd` on an *apolar-apolar* pair and
    #    `hb` on *opposite* polarity. It was never caught because every apolar
    #    fixture in this file also used a carbon receptor. Measuring per term
    #    makes the mistake visible, and it is why the reference below is a
    #    different function and not a call to the one above.

    #: The five terms, in the order SCORING.md section 3 lists them.
    TERM_NAMES = ("g1", "g2", "rep", "hb", "hyd")

    def reference_terms(d, *, probe_cls, rec_cls, weights=None):
        """The five weighted terms for one pair, classified by BOTH atoms.

        `probe_cls` and `rec_cls` are dicts with `donor`/`acceptor`/`apolar`
        booleans, both **measured** from the map rather than asserted from the
        type string -- the lesson the `from_arrays` check above records, applied
        to the receptor this time. Section 4's rules are then read literally:
        `hyd` needs an apolar-apolar pair, and `hb` needs one atom that can
        donate and one that can accept.
        """
        w = WEIGHTS if weights is None else weights
        d = np.asarray(d, dtype=float)
        polar = ((probe_cls["donor"] and rec_cls["acceptor"])
                 or (probe_cls["acceptor"] and rec_cls["donor"]))
        return {
            "g1": w["g1"] * np.exp(-(((d - 0.5) / 0.5) ** 2)),
            "g2": w["g2"] * np.exp(-(((d - 0.0) / 0.5) ** 2)),
            "rep": w["rep"] * np.where(d < 0.0, d * d, 0.0),
            "hb": w["hb"] * (1.0 - smoothstep(d, -0.5, 0.0)) if polar else 0.0,
            "hyd": (w["hyd"] * (1.0 - smoothstep(d, 0.5, 1.5))
                    if (probe_cls["apolar"] and rec_cls["apolar"]) else 0.0),
        }

    def classes_of(atom_type: str) -> dict:
        """A receptor atom's classes, measured from the slots it populates.

        Slot 1 is written by donors, slot 2 by acceptors, slot 3 by apolar
        atoms -- the same measured layout the map section above pins. Reading
        the class off the map rather than off the type string means a type whose
        table row is wrong shows up as a disagreement instead of cancelling.
        """
        per = rec_of(atom_type).precalculate(PROBE_BOX, "vina").raw_data.reshape(-1, 40)
        live = {k % 4 for k in range(40) if float(np.abs(per[:, k]).max()) > 0.0}
        return {"donor": 1 in live, "acceptor": 2 in live, "apolar": 3 in live}

    # Grid nodes, so both atoms can be placed on one. `0.375` is the engine's
    # default spacing and `PROBE_BOX`'s corner is -2.0 on every axis.
    TERM_BOX = GridBox((-2.0, -2.0, -2.0), (2.0, 2.0, 10.0))

    def on_node(axis_index: int) -> float:
        return -2.0 + 0.375 * axis_index

    #: Where the receptor atom sits: a node, so the *only* pairs in the problem
    #: are receptor-probe ones and the probe's only atom is on a node too.
    REC_NODE = (on_node(8), on_node(2), on_node(4))

    def rec_on_node(atom_type: str) -> Receptor:
        return Receptor.from_pdbqt_str(
            W.format_atom_line(1, " X1", "UNL", 1, REC_NODE, 0.0, atom_type)
            + "\nEND\n")

    def probe_atom(atom_type: str) -> Ligand:
        """A one-atom ligand, so its intramolecular term is empty."""
        return Ligand.from_pdbqt_str(
            W.format_atom_line(1, " X1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, atom_type)
            + "\nEND\n")

    #: (receptor type, probe type, why this pair is here)
    TERM_CASES = [
        ("C", "C", "apolar-apolar: the only way `hyd` may be non-zero"),
        ("ND", "OA", "donor-acceptor: the only way `hb` may be non-zero"),
        ("OD", "C", "polar receptor with an apolar probe: `hyd` must be ZERO"),
        ("OA", "OA", "acceptor-acceptor: like-with-like is withheld"),
    ]

    def probe_only_reference(rec_type: str, probe_type: str, d: float) -> dict:
        """What `reference_pair` above would predict, for the same pair.

        Kept as a **named refuted rule**, not deleted: it gates `hyd` and `hb` on
        the *probe* alone, which section 4 does not say. It agreed with the
        engine everywhere this file used before, because every apolar fixture
        also used a carbon receptor -- so the two errors cancelled and the rule
        looked right. The check below calls it precisely so that a reader can
        see what the pair-aware reference bought.
        """
        lig = probe_atom(probe_type)
        return reference_pair(
            d,
            donor=lig.atom_kinds[0] in ("donor", "donor_acceptor"),
            acceptor=lig.atom_kinds[0] in ("acceptor", "donor_acceptor"),
            apolar=lig.atom_kinds[0] == "hydrophobic",
        )

    #: The single d at which the refuted probe-only rule is compared. Any
    #: d works for the claim -- the two rules disagree at *every* separation,
    #: because one of them gates on the receptor and the other does not -- so
    #: this is a fixed point rather than a scan, and the scan below reports the
    #: agreement across all of them.
    REFUTED_AT = 0.0
    #: The hydrophobic term the probe-only rule predicts for a pair the engine
    #: and the pair-aware reference both score at exactly zero. It is the
    #: whole of the disagreement, so it is computed once and quoted by name.
    refuted_hyd = abs(probe_only_reference("OD", "C", REFUTED_AT)["hyd"])

    def scan(rec_type: str, probe_type: str, k_range=range(5, 22)):
        """Per-term engine/reference comparison for one pair, both on nodes.

        Returns `(rows, kind)` where each row is
        `(d, engine_terms, reference_terms)` and `kind` is the probe's measured
        class -- returned rather than looked up, so a caller can assert the
        fixture is what it claims to be.
        """
        rec = rec_on_node(rec_type)
        maps = rec.precalculate(TERM_BOX, "vina")
        terms = rec.precalculate_terms(TERM_BOX, "vina")
        lig = probe_atom(probe_type)
        rcls = classes_of(rec_type)
        pcls = {"donor": lig.atom_kinds[0] in ("donor", "donor_acceptor"),
                "acceptor": lig.atom_kinds[0] in ("acceptor", "donor_acceptor"),
                "apolar": lig.atom_kinds[0] == "hydrophobic"}
        r_rad, p_rad = XS_RADIUS[rec_type[0]], XS_RADIUS[probe_type[0]]
        rows = []
        for k in k_range:
            pos = [REC_NODE[0], REC_NODE[1], on_node(k)]
            conf = np.zeros(lig.num_dof)
            conf[0:3] = pos
            got = score_conformation_terms(lig, maps, terms, conf)
            d = float(np.linalg.norm(np.array(pos) - np.array(REC_NODE))) - r_rad - p_rad
            rows.append((d, got, reference_terms(d, probe_cls=pcls, rec_cls=rcls)))
        return rows, pcls, rcls

    # --- the measurements -------------------------------------------------
    per_term_worst = {t: 0.0 for t in TERM_NAMES}
    per_term_at = {t: None for t in TERM_NAMES}
    per_term_ref = {t: 0.0 for t in TERM_NAMES}
    per_term_n = {t: 0 for t in TERM_NAMES}
    #: `|terms_total - intermolecular|`: the identity the decomposition exists
    #: to support, measured rather than assumed.
    sum_worst, sum_at = 0.0, None
    kind_rows, withdrawn = [], []
    #: The terms at a geometry where the hydrophobic term is *saturated*, used
    #: to pin the weight-application point to a number rather than a docstring.
    saturated = None
    for rec_type, probe_type, why in TERM_CASES:
        rows, pcls, rcls = scan(rec_type, probe_type)
        for d, got, ref in rows:
            for t in TERM_NAMES:
                if ref[t] != 0.0:
                    per_term_n[t] += 1
                    delta = abs(got[t] - ref[t])
                    if delta > per_term_worst[t]:
                        per_term_worst[t], per_term_at[t] = delta, (rec_type, probe_type, d)
                        per_term_ref[t] = abs(float(ref[t]))
            if abs(got["terms_total"] - got["intermolecular"]) > abs(sum_worst):
                sum_worst = got["terms_total"] - got["intermolecular"]
                sum_at = (rec_type, probe_type, d)
            if pcls["apolar"] and not rcls["apolar"]:
                key = (rec_type, probe_type)
                if key not in {w[:2] for w in withdrawn}:
                    withdrawn.append((rec_type, probe_type, got["hyd"], ref["hyd"]))
            if (saturated is None and rec_type == "C" and probe_type == "C"
                    and d <= 0.5 and d > -1.0):
                saturated = got

    #: The noise floor, measured by re-running the whole comparison with the
    #: receptor on a *different* node. Two fixtures that differ only in which
    #: grid point they use give this file its own floor for the per-term
    #: agreement; a tolerance that does not sit above it is not testing
    #: anything, so the relationship is asserted rather than trusted.
    alt_worst = 0.0
    for rec_type, probe_type, _ in TERM_CASES:
        rec = rec_on_node(rec_type)
        maps = rec.precalculate(TERM_BOX, "vina")
        terms = rec.precalculate_terms(TERM_BOX, "vina")
        lig = probe_atom(probe_type)
        pcls = {"donor": lig.atom_kinds[0] in ("donor", "donor_acceptor"),
                "acceptor": lig.atom_kinds[0] in ("acceptor", "donor_acceptor"),
                "apolar": lig.atom_kinds[0] == "hydrophobic"}
        rcls = classes_of(rec_type)
        for k in range(5, 22):
            pos = [REC_NODE[0] + 0.375, REC_NODE[1], on_node(k)]
            conf = np.zeros(lig.num_dof)
            conf[0:3] = pos
            got = score_conformation_terms(lig, maps, terms, conf)
            d = (float(np.linalg.norm(np.array(pos) - np.array(REC_NODE)))
                 - XS_RADIUS[rec_type[0]] - XS_RADIUS[probe_type[0]])
            ref = reference_terms(d, probe_cls=pcls, rec_cls=rcls)
            for t in TERM_NAMES:
                if ref[t] != 0.0:
                    alt_worst = max(alt_worst, abs(got[t] - ref[t]))

    #: Per-term tolerance, **relative** to the term's own magnitude: `rep`
    #: reaches ~10 kcal/mol in the deepest overlap sampled here and `g2` never
    #: leaves 1e-3, so one absolute bound cannot be tight for both. 1e-6 is
    #: four orders of magnitude above the measured floor (`alt_worst`) and two
    #: orders below the single-term perturbation the discrimination check
    #: injects, so it discriminates that perturbation and nothing finer.
    TERM_TOL = 1e-6
    #: The bound on the sum identity. Larger than any per-term bound because it
    #: sums five, and because the production `Shape` field rounds `g1 + g2 +
    #: rep` into one `f32` where the term maps round three -- the one structural
    #: reason the two can differ at all.
    SUM_TOL = 1e-5
    #: The size of the single-term perturbation used below to prove the
    #: per-term checks can see something the sum cannot.
    PERTURB = 1e-2

    def term_bad(t: str) -> bool:
        """Relative to the term's own magnitude at its worst point."""
        return per_term_worst[t] > TERM_TOL * max(1.0, per_term_ref[t])

    first_terms = saturated or {}

    check("the engine can be asked for its five terms, reports the five the "
          "spec names, and they are already weighted",
          set(TERM_NAMES) <= set(first_terms)
          and all(np.isfinite(first_terms[t]) for t in TERM_NAMES)
          and abs(first_terms["terms_total"]
                  - sum(first_terms[t] for t in TERM_NAMES)) < 1e-12
          and abs(first_terms["hyd"] - WEIGHTS["hyd"]) < 1e-7
          and abs(first_terms["shape"]
                  - (first_terms["g1"] + first_terms["g2"] + first_terms["rep"])) < 1e-12,
          f"`score_conformation_terms` returns "
          f"{sorted(k for k in first_terms if k in TERM_NAMES)} in kcal/mol. "
          f"**Already weighted, and that is pinned to a number rather than "
          f"asserted in prose**: for a carbon-carbon pair at d = "
          f"{per_term_at['hyd'][2]:+.2f} A the hydrophobic term is saturated, so "
          f"the engine must report exactly the spec's {WEIGHTS['hyd']:.6f} and it "
          f"reports {first_terms['hyd']:.9f}. The Vina weights are applied while "
          f"the map is TABULATED -- `SpatialKernels::eval_terms` is the only "
          f"place one is multiplied in, and the four-slot path calls it too, so "
          f"the two cannot disagree about where. The only multiplier still "
          f"applied when a conformation is scored is the 0/1 per-slot class "
          f"mask, which selects which fields a probe reads and is not a weight. "
          f"Multiplying by a weight again would double-count it")

    check("each of the five terms matches the reference separately, and every "
          "one of them was actually exercised",
          all(per_term_n[t] > 0 for t in TERM_NAMES)
          and not any(term_bad(t) for t in TERM_NAMES)
          and TERM_TOL > alt_worst
          and PERTURB > TERM_TOL
          and TERM_TOL > 1e-9,
          "; ".join(
              f"{t}: max |engine - reference| {per_term_worst[t]:.2e} over "
              f"{per_term_n[t]} points"
              + (f" (worst at {per_term_at[t][0]}...{per_term_at[t][1]}, "
                 f"d = {per_term_at[t][2]:+.2f} A)" if per_term_at[t] else "")
              for t in TERM_NAMES)
          + f". **The geometry is what makes this tight**: both atoms sit on "
            f"0.375 A grid nodes, so the trilinear interpolation is exact (one "
            f"corner carries weight 1) and the only error left is the `f32` "
            f"rounding of the tabulated field. Off-node the same comparison sits "
            f"at ~1e-2, which is the floor the total checks above live on and "
            f"which no single-term perturbation could hide under. **The "
            f"tolerance is checked, not trusted:** the noise floor, measured by "
            f"re-running everything with the receptor moved one node along x, is "
            f"{alt_worst:.2e}, and the check requires {TERM_TOL:g} > that floor "
            f"and {PERTURB:g} > {TERM_TOL:g} so the perturbation below is inside "
            f"what this can see and outside what it forgives")

    check("the five terms sum to the intermolecular energy the engine already "
          "returned",
          abs(sum_worst) < SUM_TOL
          and SUM_TOL > alt_worst
          and abs(sum_worst) < PERTURB,
          f"worst |terms_total - intermolecular| over the whole scan is "
          f"{abs(sum_worst):.2e} kcal/mol, at {sum_at[0]}...{sum_at[1]}, "
          f"d = {sum_at[2]:+.2f} A. **This is not exactly zero and the reason is "
          f"structural, not a bug**: the production map rounds "
          f"`w1*g1 + w2*g2 + w_rep*rep` into a single `f32` field, while the "
          f"term maps round each of the three separately, so the two differ by "
          f"the storage granularity of a `f32` and by nothing else. "
          f"`f32` epsilon is 1.19e-07, and the measured value is "
          f"{abs(sum_worst) / 1.1920929e-07:.2f} of that -- the bound "
          f"{SUM_TOL:g} is above the measured value and far below the "
          f"{PERTURB:g} perturbation, so it is pinned with its reason rather "
          f"than loosened to fit. Asserting exact equality here would be "
          f"asserting a property the data layout does not have")

    check("a term is a property of the PAIR, not of the probe, and a polar "
          "receptor takes the hydrophobic term away from an apolar probe",
          len(withdrawn) == 1
          and all(abs(g) < 1e-12 and abs(r) < 1e-12 for _, _, g, r in withdrawn)
          and refuted_hyd > TERM_TOL,
          "; ".join(
              f"{r}...{p}: engine hyd {g:+.6f}, pair-aware reference {rf:+.6f}"
              for r, p, g, rf in withdrawn)
          + f". **This is a defect the old reference could not see.** "
            f"`reference_pair` above gates `hyd` on the *probe* being apolar, so "
            f"at d = {REFUTED_AT:+.2f} A against this polar receptor it predicts "
            f"{probe_only_reference('OD', 'C', REFUTED_AT)['hyd']:+.4f} where the "
            f"engine correctly gives {withdrawn[0][2]:+.6f} -- a "
            f"{refuted_hyd:.4f} kcal/mol disagreement that is the *reference's* "
            f"error, not the engine's. SCORING.md section 4 gates `hyd` on an "
            f"apolar-APOLAR pair, and the engine enforces it. Every apolar "
            f"fixture in this file used a carbon receptor, so the mistake "
            f"cancelled; the per-term surface is what made it visible, which is "
            f"the argument for having one")

    # --- the demonstration this whole section exists for -------------------
    # A per-term check nobody has watched fail is decoration. The perturbation
    # is applied to the REFERENCE, not the engine, because the reference is the
    # thing this file owns and the point is to show the *comparison* separates.
    # It is the cancellation case, and it is the one that matters: two terms are
    # moved by equal and opposite amounts, so the SUM is untouched and the only
    # thing that can notice is a per-term comparison. This is the grid defect's
    # shape -- a whole field wrong consistently, invisible from outside.
    def _perturbed(ref: dict) -> dict:
        """Two terms moved by `+-PERTURB`, so the total is unchanged."""
        out = dict(ref)
        out["g1"] = ref["g1"] + PERTURB
        out["hyd"] = ref["hyd"] - PERTURB
        return out

    disc = {t: 0.0 for t in TERM_NAMES}
    #: How far the *perturbed* reference's own total sits from the engine's
    #: intermolecular energy -- i.e. what a sum-level comparison would report.
    disc_sum = 0.0
    #: How far the perturbation moved the reference's total at all. This is the
    #: cancellation claim, and it is measured rather than assumed: if the two
    #: edits did not cancel, this would be ~PERTURB and the check would be
    #: claiming a cancellation it does not have.
    cancel_residual = 0.0
    for rec_type, probe_type, _ in TERM_CASES:
        rows, pcls, rcls = scan(rec_type, probe_type)
        for d, got, ref in rows:
            alt = _perturbed(ref)
            cancel_residual = max(cancel_residual,
                                  abs(sum(alt.values()) - sum(ref.values())))
            for t in TERM_NAMES:
                if ref[t] != 0.0:
                    disc[t] = max(disc[t], abs(alt[t] - got[t]))
            disc_sum = max(disc_sum, abs(sum(alt.values()) - got["intermolecular"]))

    check("perturbing ONE term is caught per-term and invisible in the sum, "
          "and perturbing two in opposite directions is caught with the sum "
          "provably unchanged",
          cancel_residual < 1e-12
          and disc["g1"] > TERM_TOL
          and disc["hyd"] > TERM_TOL
          and all(disc[t] <= TERM_TOL * max(1.0, per_term_ref[t])
                  for t in ("g2", "rep", "hb"))
          and disc_sum < SUM_TOL
          and disc["g1"] > disc_sum,
          f"injecting `{PERTURB:g}` into the reference's `g1` and taking it out "
          f"of `hyd` -- equal and opposite -- moves the reference's own total by "
          f"{cancel_residual:.1e} kcal/mol, so the cancellation is exact rather "
          f"than approximate. **A sum-level comparison would then report "
          f"{disc_sum:.2e}**, inside the {SUM_TOL:g} the sum identity is held "
          f"to, i.e. indistinguishable from the unperturbed run. **A per-term "
          f"comparison reports g1 {disc['g1']:.2e} and hyd "
          f"{disc['hyd']:.2e}**, both {PERTURB / TERM_TOL:.0f}x the "
          f"{TERM_TOL:g} per-term tolerance, while g2, rep and hb stay at "
          f"{', '.join(f'{t} {disc[t]:.1e}' for t in ('g2', 'rep', 'hb'))}. "
          f"**This is the deliverable and it is measured, not asserted**: "
          f"`disc['g1'] > disc_sum` is the whole claim -- one number a total "
          f"cannot see and a per-term reading can. The engine-side version of "
          f"the same mutation, a weight perturbed in `scoring.rs` and rebuilt, "
          f"is reported separately because only that one exercises the real "
          f"arithmetic")

    # ------------------------------------------------------------------
    section("per-slot is a different question from per-term, and the engine "
            "answers both")

    # The requirement-4 question, answered rather than left implicit. The four
    # map slots are what the grid physically stores; the five terms are what the
    # scoring function is written in. They do not correspond in either
    # direction, and the check above's withdrawal rested on not knowing which.
    # Now both are measurable, so the relationship is asserted as a rule.
    slot_rows = scan("C", "C")[0]
    slot_case = next((g for _, g, _ in slot_rows), {})
    slot_sum = sum(slot_case.get("slots", []))
    hb_case = scan("ND", "OA")[0]
    hb_row = next((g for _, g, _ in hb_case), {})

    check("per-slot contributions are exposed and sum to the total, and four "
          "slots are demonstrably not five terms",
          len(slot_case.get("slots", [])) == 4
          and abs(slot_sum - slot_case["total"]) < 1e-9
          # Relative, because `shape` reaches ~10 kcal/mol in the deep overlap
          # sampled here and the two sides differ only by `f32` rounding of a
          # value that size. An absolute bound tight enough for `g2` (never
          # above 1e-3) would be violated by rounding alone at this magnitude,
          # and loosening it to fit would be the wrong fix -- the right one is
          # to say what the comparison is actually sensitive to.
          and abs(slot_case["slots"][0] - slot_case["shape"])
          <= 1e-6 * max(1.0, abs(slot_case["shape"]))
          and abs(hb_row["hb"] - (hb_row["hb_from_donor"]
                                  + hb_row["hb_from_acceptor"])) < 1e-12
          # The acceptor probe reads the receptor DONOR's field and not the
          # acceptor's -- the like-with-like withholding, asserted on the slots
          # rather than only on the terms.
          and abs(hb_row["slots"][2]) < 1e-12
          and abs(hb_row["slots"][1]) > 0.01
          and abs(hb_row["hb_from_acceptor"]) < 1e-12
          and abs(hb_row["hb"]) > 0.01,
          f"the per-slot reading is a **map-role** decomposition, not a term "
          f"one: slots are "
          f"{slot_case.get('slot_names')} and they sum to "
          f"{slot_sum:.9f} against a total of {slot_case['total']:.9f}. **Four "
          f"slots are not four terms in either direction**, and both directions "
          f"are now measured rather than argued: slot 0 is the three shape "
          f"terms already fused into one f32 ({slot_case['slots'][0]:.6f} = "
          f"g1 {slot_case['g1']:.6f} + g2 {slot_case['g2']:.6f} + rep "
          f"{slot_case['rep']:.6f} = {slot_case['shape']:.6f}), so no per-term "
          f"value is recoverable from it; and the hydrogen bond is **one** term "
          f"spread over **two** slots ({hb_row['hb_from_donor']:+.6f} from "
          f"donors + {hb_row['hb_from_acceptor']:+.6f} from acceptors = "
          f"{hb_row['hb']:+.6f}), so counting slots over-counts the terms. The "
          f"acceptor probe also reads slot {hb_row['slot_names'][1]} at "
          f"{hb_row['slots'][1]:+.6f} and slot {hb_row['slot_names'][2]} at "
          f"{hb_row['slots'][2]:+.6f} -- the like-with-like withholding, now "
          f"visible on the slots and not only on the terms. "
          f"**So the earlier withdrawal was right about the consequence and "
          f"wrong about the reason**: the split is not a per-element partition "
          f"(that defect is gone) -- it is a per-ROLE partition, and the role "
          f"and the term only sometimes line up. A caller wanting 'how much of "
          f"this score is the hydrogen bond' must use the term surface, and one "
          f"wanting 'which fields did this probe draw on' can use the slots")

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
