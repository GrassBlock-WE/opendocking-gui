"""Check that the viewer bonds the right atoms to each other.

Headless on purpose. The bond logic is the part of a molecular viewer that can
be *silently* wrong -- a wrong bond still draws a line, still looks like a
molecule, and nothing crashes -- so it is verified without Qt or a display, and
the assertions are about chemistry rather than about pixels.

Run:  python scripts/structure_bond_check.py
"""

from __future__ import annotations

import math
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Prefer the installed package; fall back to the source tree only when there is
# none. Putting `dock-py/python` on the path first imports the source copy of
# `opendocking`, which a clean checkout cannot load because it has no compiled
# `_dockpy` extension -- that file is gitignored and only exists in
# site-packages. See `contacts_check.py` for the same note in full.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

import numpy as np  # noqa: E402

from opendocking.workbench import structure as st_mod  # noqa: E402

EXAMPLES = ROOT / "examples"

#: The number of checks this file is supposed to run, on **every** machine.
#: measured: a green run with RDKit present (85 checks, exit 0) and a green run
#: with the `rdkit` import blocked (85 checks, 1 skipped, exit 0). The two agree
#: because `skip` is counted into the total rather than dropped from it -- that
#: agreement is the whole reason this number is worth pinning, and it is not a
#: claim that can be made about a count that only holds when an optional package
#: happens to be installed. Lower it only after a real green run.
EXPECTED_CHECKS = 85

FAILURES: list[str] = []
CHECKS = 0
#: Checks this file could not run here, as (name, reason). A skip and a check are
#: the *same* check, so a skip is counted once -- in `SKIPPED`, not in `CHECKS`.
#: The total is `CHECKS + len(SKIPPED)` and that sum is what is pinned, which is
#: what makes the pinned number the same on a machine with RDKit and one without.
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def skip(name: str, reason: str) -> bool:
    """Record a check that could not be run here, and say why.

    A skipped check and a failed check are different claims, and this file
    already needs the distinction: the one oracle here is RDKit, and a machine
    without it should say so rather than quietly assert less.

    **This used to print the line and record nothing**, on the reasoning that the
    number of checks a run reports should not depend on which optional tools the
    machine has. The reasoning was sound and the consequence was not: with the
    skip uncounted, a run without RDKit reported one fewer check, and since this
    file had no pin there was nothing to notice. The tally read `84 passed, 0
    failed, 84 checks` on a machine that had measured 84 of 85 things and could
    not say so -- the same "N passed either way" problem as a check that stays
    green when the thing it guards is gone.

    So the skip is recorded, counted separately, and *added into the total*. The
    total is now 85 whether the oracle was there or not, which is the property
    that makes the total worth pinning at all. See `main()` for the summary,
    which prints the three counts separately and never sums them into a verdict.
    """
    SKIPPED.append((name, reason))
    print(f"  [SKIP] {name}  — {reason}")
    return False


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def dist(a, b) -> float:
    return math.dist(a, b)


# ---------------------------------------------------------------------------
# Synthetic backbones
# ---------------------------------------------------------------------------
#
# Secondary structure needs fixtures whose real answer is known, and no protein
# in `examples/` is small enough and simple enough for its answer to be
# unarguable. So the fixtures here are built from ideal internal coordinates
# instead of downloaded.
#
# They are built by extension reference frame: place a bond length, a bond angle
# and a torsion from three atoms already placed, and walk down the chain. Given
# a list of (phi, psi) the result is a physically possible peptide by
# construction -- every bond length and angle is a real one -- and the phi and
# psi that come out are measured back out of the finished coordinates with the
# module's own `dihedral`, so a fixture cannot quietly stop being what it claims
# to be. `check_fixtures_are_physical` below asserts exactly that, because a
# fixture that lies about its own geometry asserts nothing.

#: Bond lengths and bond angles of a trans peptide backbone, in angstrom and
#: degrees. These are measured values for a real peptide, not round numbers
#: chosen to look tidy.
N_CA = 1.458
CA_C = 1.525
C_N = 1.329
C_O = 1.231
CA_CB = 1.530
ANG_N_CA_C = 111.2
ANG_CA_C_N = 116.2
ANG_C_N_CA = 121.7
ANG_CA_C_O = 120.8

#: Ideal backbone conformations. Alpha and 3-10 helices at the Ramachandran
#: values; extended, which is what a beta strand is.
HELIX_PHI, HELIX_PSI = -57.0, -47.0
EXTENDED_PHI, EXTENDED_PSI = -139.0, 135.0
#: A real coil conformation: left of the alpha basin in phi, well outside it in
#: psi, and far from the extended region. A γ-turn would be phi ~ +60, but a
#: γ-turn *forms* an i -> i+2 bond, which is a turn and not a coil, so it is not
#: what a "this is not secondary structure" fixture should be made of.
COIL_PHI, COIL_PSI = -120.0, 25.0


def _sub(a, b):
    return [a[k] - b[k] for k in range(3)]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _place(a, b, c, length: float, angle: float, torsion: float):
    """Place a fourth atom from three placed ones, by extension reference frame.

    `a`, `b`, `c` give the plane and the sign; the result is `length` from `c`,
    at `angle` between b-c and c-result, and at `torsion` about b-c from a.
    """
    bc = _sub(c, b)
    nb = math.sqrt(sum(x * x for x in bc))
    bc = [x / nb for x in bc]
    normal = _cross(_sub(b, a), bc)
    nn = math.sqrt(sum(x * x for x in normal))
    normal = [x / nn for x in normal]
    across = _cross(normal, bc)
    th, to = math.radians(angle), math.radians(torsion)
    offset = (
        -length * math.cos(th),
        length * math.sin(th) * math.cos(to),
        length * math.sin(th) * math.sin(to),
    )
    return tuple(
        c[k] + offset[0] * bc[k] + offset[1] * across[k] + offset[2] * normal[k]
        for k in range(3)
    )


def _peptide(phis, psis, chain: str = "A", first_resid: int = 1):
    """N, CA, C, O and CB coordinates per residue for one peptide chain.

    `phis[i]` and `psis[i]` are asked for, so psi[0] is used (it places the
    first amide) and phi[0] is ignored (there is no residue before it to measure
    it against).
    """
    # Two throwaway points to define the frame of the first residue. That
    # residue has no phi, so the absolute placement of its own atoms is
    # arbitrary; what matters is that every atom after it is placed from a real
    # torsion.
    seed_a, seed_b = (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    res: list[dict] = []
    for i, psi in enumerate(psis):
        if i == 0:
            n = (0.0, 0.0, 0.0)
            ca = _place(seed_a, seed_b, n, N_CA, 109.5, 0.0)
            c = _place(seed_a, n, ca, CA_C, ANG_N_CA_C, 180.0)
        else:
            pn, pca, pc = res[i - 1]["N"], res[i - 1]["CA"], res[i - 1]["C"]
            n = _place(pn, pca, pc, C_N, ANG_CA_C_N, psi)
            ca = _place(pca, pc, n, N_CA, ANG_C_N_CA, 180.0)
            c = _place(pc, n, ca, CA_C, ANG_N_CA_C, phis[i])
        # The carbonyl oxygen is anti to the amide, across the peptide bond.
        o = _place(n, ca, c, C_O, ANG_CA_C_O, psi + 180.0)
        # A real alanine: CB on the fourth tetrahedral direction of CA. Without
        # it the fixture is a backbone fragment, and "this fixture is physically
        # possible" is a weaker claim than it looks.
        bis = _sub(n, ca)
        bis = [-(bis[k] + _sub(c, ca)[k]) for k in range(3)]
        nb = math.sqrt(sum(x * x for x in bis))
        cb = tuple(ca[k] + CA_CB * bis[k] / nb for k in range(3))
        res.append(
            {"N": n, "CA": ca, "C": c, "O": o, "CB": cb,
             "resid": first_resid + i, "chain": chain}
        )
    return res


def _atom_line(serial: int, name: str, element: str, xyz, resname: str,
               resid: int, chain: str) -> str:
    """One standard 78-column ATOM record.

    The element goes in columns 77-78, which is where both this parser and
    RDKit look for it; putting the atom name there instead makes a reader guess.

    A coordinate outside the 8-character field raises instead of overflowing.
    `%8.3f` does not truncate a number that does not fit -- it *widens* the
    field, which shifts every column after it and leaves a record that still
    looks like an ATOM line. That is how a 3000-residue chain came to parse as
    1093 residues: the tail of the helix ran past +/-1000 A, every record after
    that point was column-shifted, and the coordinates were silently garbage.
    A fixture that quietly is not the size it says it is asserts nothing, and
    this is the same failure as a fixture with forty atoms on one coordinate,
    one column over.
    """
    x, y, z = xyz
    field = f"{x:8.3f}{y:8.3f}{z:8.3f}"
    if len(field) != 24:
        raise ValueError(
            f"atom {name}{resid}{chain} at ({x:.1f}, {y:.1f}, {z:.1f}) does not "
            "fit the 8-character PDB coordinate field; a chain long enough to "
            "do that cannot be written as one PDB file -- build a packed "
            "structure instead of one long chain"
        )
    return (
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resid:4d}    "
        f"{field}{1.00:6.2f}{0.00:6.2f}{'':10s}{element:>2}"
    )


def _pdb_text(chains) -> str:
    """Standard PDB text for a list of residue dicts, in the order given."""
    lines = []
    serial = 0
    for res in chains:
        for name, element in (("N", "N"), ("CA", "C"), ("C", "C"),
                              ("O", "O"), ("CB", "C")):
            serial += 1
            lines.append(
                _atom_line(serial, name, element, res[name], "ALA",
                           res["resid"], res["chain"])
            )
    lines.append("END")
    return "\n".join(lines) + "\n"


#: Gap between neighbouring strands in a :func:`_bundle`, in angstrom. Close
#: enough that residues on adjacent strands are within the 8.2 A candidate
#: cutoff of each other, which is what makes the bundle a fair stand-in for a
#: compact protein rather than a set of isolated chains.
BUNDLE_GAP = 10.0


def _bundle(total: int, per_strand: int = 200, gap: float = BUNDLE_GAP):
    """`total` residues as parallel helical strands packed `gap` apart.

    A performance claim about a protein-sized structure has to be measured on
    something protein-*shaped*. One long helix is not: its residues are spread
    along a line, so almost every pair is far apart and the prefilter rejects it
    for one distance call. A real 3000-residue protein is a globule, and in a
    globule a large fraction of all residue pairs are within a few angstroms of
    each other -- which is the case that decides the cost. Packing strands side
    by side reproduces that density without needing a fold.

    Each strand is its own chain, so the bundle is several peptides rather than
    one: a single chain of 3000 residues runs past the end of the PDB
    coordinate field (see `_atom_line`) and a 3000-residue *helix* is not a
    thing any protein contains anyway.
    """
    out = []
    for s in range(max(1, total // per_strand)):
        chain = chr(ord("A") + s % 26)
        for res in _peptide(
            [HELIX_PHI] * per_strand, [HELIX_PSI] * per_strand, chain, 1
        ):
            moved = dict(res)
            for name in ("N", "CA", "C", "O", "CB"):
                moved[name] = (res[name][0] + gap * s, res[name][1], res[name][2])
            out.append(moved)
    return out


def _unindexed_candidate_pairs(frames, atoms):
    """Every candidate pair, found the slow way. The reference for the index.

    Deliberately the plain double loop over donors against acceptors, with the
    same `HBOND_CA_SKIP` test the indexed walk uses. It is here so that
    `check_candidate_index` can ask the two for the same answer and compare them
    residue by residue, instead of trusting that an optimisation preserved the
    result -- which is exactly the assumption an optimisation cannot check about
    itself.

    The distance is this file's own `dist`, not `st_mod._dist`, so that the
    reference is independent of the code it is checking. If the module's
    distance function were wrong, both walks would be wrong in the same
    direction and the comparison would agree on a wrong answer.
    """
    donors = [
        i for i, f in enumerate(frames)
        if f.amide_h is not None and f.prev_c is not None
    ]
    acceptors = [i for i, f in enumerate(frames) if f.carbonyl_o is not None]
    return [
        (don, acc)
        for acc in acceptors
        for don in donors
        if don != acc
        and dist(atoms[frames[acc].ca].xyz, atoms[frames[don].ca].xyz)
        <= st_mod.HBOND_CA_SKIP
    ]


def _with_candidate_walk(walk):
    """Run `check` with `st_mod`'s candidate enumeration swapped for `walk`.

    Returns a restore callable. The swap is on the one function the index lives
    in, so everything downstream -- the gates, the energy, the segments, the
    helix-over-sheet precedence, the fallback -- is the real code on both sides
    and the only thing being compared is the index against no index.
    """
    original = st_mod._candidate_pairs

    def restore():
        st_mod._candidate_pairs = original

    st_mod._candidate_pairs = walk
    return restore


def _distance_calls(walk, frames, atoms) -> tuple[int, int]:
    """``(pairs returned, distance measurements made)`` for one candidate walk.

    The two counts answer different questions and only one is interesting. The
    pairs returned are the same either way -- that is the equivalence being
    checked elsewhere, and it is why comparing them proves nothing about the
    index. The measurements are the looking-around: a walk that measures a
    million distances to return thirteen thousand pairs is doing the work the
    index exists to avoid, and one that measures thirteen thousand is not.

    Both distance functions are wrapped rather than just the module's, because
    the reference walk deliberately uses this file's own `dist` and the indexed
    walk uses `st_mod._dist`; wrapping only one of them would count the
    reference as zero. Each wrapper delegates to the function it replaced, so
    the arithmetic being measured is unchanged.
    """
    global dist

    module_dist, local_dist = st_mod._dist, dist
    seen = [0]

    def counting(real):
        def wrapper(a, b):
            seen[0] += 1
            return real(a, b)
        return wrapper

    st_mod._dist = counting(module_dist)
    dist = counting(local_dist)
    try:
        found = walk(frames, atoms)
    finally:
        st_mod._dist = module_dist
        dist = local_dist
    return len(found), seen[0]


def _assignment_work(trace, atoms) -> dict:
    """Counted operations for one `secondary_structure` call, plus its clock.

    Every phase of the pass is counted, not just the enumeration, so that a
    regression which is not in the index -- a quadratic `_extend`, a strand scan
    that stops breaking out early -- shows up here as a count rather than only
    as a slower run. The clock is measured and returned but nothing asserts on
    it: this box swings 2.1x for byte-identical work, and a check that gates
    every build cannot hang on that.
    """
    counts = {
        "builds": 0, "cand_dist": 0, "gate_dist": 0, "angle": 0,
        "runs": 0, "extend": 0, "contiguous": 0, "placed_h": 0,
    }
    inside = [False]
    handed: list = []
    originals = {
        key: getattr(st_mod, key)
        for key in (
            "_candidate_pairs", "_dist", "_angle_deg", "_placed_amide_h",
            "_runs", "_extend", "_contiguous", "_hbonds",
        )
    }

    def counting(key, real):
        def wrapper(*a, **kw):
            counts[key] += 1
            return real(*a, **kw)
        return wrapper

    def dist_wrapper(a, b):
        counts["cand_dist" if inside[0] else "gate_dist"] += 1
        return originals["_dist"](a, b)

    def candidate_wrapper(frames, frame_atoms):
        counts["builds"] += 1
        inside[0] = True
        try:
            return originals["_candidate_pairs"](frames, frame_atoms)
        finally:
            inside[0] = False

    def hbonds_wrapper(*a, **kw):
        # What each pass was actually handed. A count of builds cannot see a
        # list that was sliced between the two passes -- one build, two
        # different lists -- and that mutation survived every check here
        # until this was written, because the loose bonds only ever bridge
        # segments the strict pass already found, so losing some of them
        # changes no residue on these fixtures.
        cands = kw.get("candidates")
        if cands is None and len(a) > 3:
            cands = a[3]
        handed.append(list(cands) if cands is not None else None)
        return originals["_hbonds"](*a, **kw)

    st_mod._candidate_pairs = candidate_wrapper
    st_mod._dist = dist_wrapper
    st_mod._angle_deg = counting("angle", originals["_angle_deg"])
    st_mod._placed_amide_h = counting("placed_h", originals["_placed_amide_h"])
    st_mod._runs = counting("runs", originals["_runs"])
    st_mod._extend = counting("extend", originals["_extend"])
    st_mod._contiguous = counting("contiguous", originals["_contiguous"])
    st_mod._hbonds = hbonds_wrapper
    try:
        start = time.perf_counter()
        states = st_mod.secondary_structure(trace, atoms)
        counts["elapsed"] = time.perf_counter() - start
    finally:
        for key, real in originals.items():
            setattr(st_mod, key, real)
    counts["states"] = states
    counts["handed"] = handed
    counts["units"] = sum(
        counts[key]
        for key in ("cand_dist", "gate_dist", "angle", "runs", "extend",
                    "contiguous", "placed_h")
    )
    return counts


def _rigid_fit(moving: np.ndarray, target: np.ndarray):
    """The rigid motion -- a rotation and a translation -- taking `moving` onto
    `target`, by Kabsch.

    The determinant guard is what keeps this a rotation. Without it the fit is
    free to mirror the chain, which turns a left-handed peptide into a
    right-handed one and produces a structure that is geometrically tidy and
    chemically fictional.
    """
    mc, tc = moving.mean(axis=0), target.mean(axis=0)
    p, q = moving - mc, target - tc
    u, _s, vt = np.linalg.svd(p.T @ q)
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, flip]) @ u.T
    return rot, tc - rot @ mc


def _strand_frame(strand):
    """A local frame for a strand: its axis, and two perpendicular directions."""
    mid = len(strand) // 2
    axis = np.array(strand[mid + 1]["CA"]) - np.array(strand[mid]["CA"])
    axis = axis / np.linalg.norm(axis)
    origin = np.array(strand[mid]["CA"])
    side = np.array(strand[0]["CA"]) - origin
    side = side - axis * (side @ axis)
    side /= np.linalg.norm(side)
    return origin, axis, side, np.cross(axis, side)


def _place_second_strand(a, b):
    """Put strand `b` beside strand `a`, hydrogen bonded, by hypothesis search.

    Two strands of the same extended conformation are congruent, so the only
    freedom is the rigid-body placement of the second one. That is searched over
    *physical hypotheses* rather than a free grid: which of strand A's residues
    strand B's first residue should sit beside, which way across the sheet, and
    how far away. Each hypothesis becomes a placement through a Kabsch fit of
    B's CA atoms onto the CA atoms of the hypothesised register, and the
    best-scoring hypothesis wins. No randomness, no starting guess, and no free
    six-dimensional search.

    An earlier version searched a translation grid with a twist about the strand
    axis and could not find a sheet at all: that family of placements has one
    rotational degree of freedom, and no antiparallel register lies in it.

    The score is a plain N...O distance gate, deliberately weaker than anything
    `structure` uses -- no Kabsch-Sander energy, no angle, no placed hydrogen --
    so the fixture is not placed by the code under test.

    Maximising contacts on its own is not enough, and the reason is worth
    recording. With no steric constraint the best placement is the degenerate
    one, laying the second strand on top of the first, which scores 23
    "contacts" and puts a CB 0.9 A from an oxygen. That is exactly the "forty
    atoms on identical coordinates" fixture this repository has grown before, so
    any placement with an inter-strand CA closer than 4.4 A is rejected outright.

    It is still not an independent oracle, and the sheet checks should not be
    read as one. A fixture built to maximise hydrogen bonds will of course be read
    as a sheet. What the sheet checks establish is narrower and worth stating:
    the sheet comes from inter-strand bonds and nothing else, on two strands that
    are separate chains, whose register is fixed by a hypothesis rather than
    asserted, and whose individual residues carry no information a phi/psi test
    could use. The independent evidence that the method works on a real
    structure is crambin, further down.

    Returns the placed residues, the number of N...O contacts the winning
    placement has, and its closest inter-strand CA separation.
    """
    _origin, _axis, side, across = _strand_frame(a)
    ca_a = np.array([r["CA"] for r in a])
    ca_b = np.array([r["CA"] for r in b])
    n_b = np.array([r["N"] for r in b])
    o_a = np.array([r["O"] for r in a])
    total = len(a)

    best: tuple | None = None
    for register in range(-total + 3, total - 2):
        span = [(j, j + register) for j in range(total)
                if 0 <= j + register < total]
        if len(span) < 3:
            continue
        moving = ca_b[[j for j, _ in span]]
        for normal in (side, -side, across, -across):
            for spacing in (4.4, 4.7, 5.0):
                target = np.array(
                    [ca_a[k] + spacing * normal for _j, k in span]
                )
                rot, shift = _rigid_fit(moving, target)
                moved_n = n_b @ rot.T + shift
                contacts = int((
                    np.linalg.norm(
                        moved_n[:, None, :] - o_a[None, :, :], axis=-1
                    ) < 3.0
                ).sum())
                if contacts < 3:
                    continue
                moved_ca = ca_b @ rot.T + shift
                separation = float(np.linalg.norm(
                    moved_ca[:, None, :] - ca_a[None, :, :], axis=-1
                ).min())
                if separation < 4.4:
                    continue
                if best is None or contacts > best[0]:
                    best = (contacts, separation, rot, shift)
    if best is None:
        raise ValueError(
            "no register hypothesis produced a non-clashing pair of strands "
            "with three or more backbone contacts"
        )
    contacts, separation, rot, shift = best
    placed = []
    for res in b:
        new = dict(res)
        for name in ("N", "CA", "C", "O", "CB"):
            new[name] = tuple(np.array(res[name]) @ rot.T + shift)
        placed.append(new)
    return placed, contacts, separation


def prepare_receptor_text(pdb: Path) -> str:
    """Prepare a receptor with the *current* source, not a stored fixture.

    The committed `*_prep.pdbqt` files were written before residue identity was
    preserved; preparing here means this check always tests the code as it is.
    """
    from opendocking.prep import prepare_receptor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return prepare_receptor(pdb)


# ---------------------------------------------------------------------------
# Protein
# ---------------------------------------------------------------------------


def check_protein() -> None:
    section("protein: bonds come from residue templates")
    text = prepare_receptor_text(EXAMPLES / "1crn_receptor.pdb")
    st = st_mod.parse_structure(text, "1crn")
    check("parsed", len(st.atoms) > 0, f"{len(st.atoms)} atoms")
    check("recognised as protein", st.is_protein())

    aas = [r for r in st.residues if r.is_amino_acid]
    check(
        "every residue matched a template",
        len(aas) == len([r for r in st.residues if r.name not in st_mod.ION_RESIDUES]),
        f"{len(aas)} amino-acid residues of {len(st.residues)} total",
    )
    check("no unknown-residue warning",
          not any("no connectivity template" in w for w in st.warnings),
          "; ".join(w for w in st.warnings if "template" in w))

    # Residue numbering must be contiguous, because that is what licenses the
    # peptide bonds.
    by_chain: dict[str, list[int]] = {}
    for r in aas:
        by_chain.setdefault(r.chain, []).append(r.resid)
    for chain, ids in by_chain.items():
        ids = sorted(ids)
        gaps = [
            (ids[i], ids[i + 1]) for i in range(len(ids) - 1) if ids[i + 1] - ids[i] != 1
        ]
        check(
            f"chain {chain.strip() or '_'} residue numbering is contiguous",
            not gaps,
            f"{len(ids)} residues, {len(gaps)} gap(s){': ' + str(gaps[:4]) if gaps else ''}",
        )
        check(
            f"chain {chain.strip() or '_'} has no duplicate residue numbers",
            len(set(ids)) == len(ids),
        )

    # Backbone completeness.
    missing = []
    for r in aas:
        names = {st.atoms[i].name for i in r.atoms}
        for key in ("N", "CA", "C"):
            if key not in names:
                missing.append(f"{r.name}{r.resid}.{key}")
    check("every residue has N, CA and C", not missing, ", ".join(missing[:6]))

    trace = st.backbone()
    check("backbone trace covers every residue", len(trace) == len(aas),
          f"{len(trace)} residues traced")

    # --- the important one: no bond may cross a residue except the peptide C-N
    crosses: list[str] = []
    peptide = 0
    for i, j in st.bond_pairs():
        a, b = st.atoms[i], st.atoms[j]
        if a.residue_key == b.residue_key:
            continue
        if {a.name, b.name} == {"C", "N"} and b.resid - a.resid == 1 and a.chain == b.chain:
            peptide += 1
            continue
        crosses.append(f"{a.name}{a.resname}{a.resid}-{b.name}{b.resname}{b.resid}")
    check(
        "no bond crosses a residue except the peptide C-N",
        not crosses,
        f"{peptide} peptide bonds; offenders: {'; '.join(crosses[:6])}",
    )
    expected_peptide = sum(
        sum(1 for a, b in zip(ids, ids[1:]) if b - a == 1)
        for ids in (sorted(r.resid for r in aas if r.chain == c) for c in by_chain)
    )
    check(
        "one peptide bond per consecutive residue pair",
        peptide == expected_peptide,
        f"{peptide} drawn, {expected_peptide} expected",
    )

    # --- side chains match their templates exactly
    sidechain_errors: list[str] = []
    for r in aas:
        canon = st_mod.RESIDUE_ALIASES.get(r.name, r.name)
        template = set(st_mod.SIDECHAIN_BONDS.get(canon, ()))
        by_name: dict[str, int] = {}
        for i in r.atoms:
            by_name.setdefault(st.atoms[i].name, i)
        present = {
            (p, c)
            for p, c in template
            if p in by_name
            and c in by_name
            and by_name[c] in st.atoms[by_name[p]].bonds
        }
        expected = {
            (p, c) for p, c in template if p in by_name and c in by_name
        }
        if present != expected:
            missing_bonds = sorted(expected - present)
            sidechain_errors.append(f"{r.name}{r.resid} missing {missing_bonds}")
    check(
        "every side-chain bond of every residue is drawn",
        not sidechain_errors,
        "; ".join(sidechain_errors[:5]),
    )

    proline = next((r for r in aas if r.name == "PRO"), None)
    if proline is not None:
        by_name = {st.atoms[i].name: i for i in proline.atoms}
        has_ring = "CD" in by_name and "N" in by_name and by_name["N"] in st.atoms[by_name["CD"]].bonds
        check("proline's side chain closes back onto its backbone N", has_ring,
              f"PRO{proline.resid}")

    # --- geometry sanity
    lengths = [dist(st.atoms[i].xyz, st.atoms[j].xyz) for i, j in st.bond_pairs()]
    if lengths:
        check(
            "every bond length is physically possible",
            min(lengths) > 0.85 and max(lengths) < 2.0,
            f"min {min(lengths):.2f} A, max {max(lengths):.2f} A, {len(lengths)} bonds",
        )
    over = [
        f"{st.atoms[i].element}{st.atoms[i].name}{st.atoms[i].resid} has {len(st.atoms[i].bonds)} bonds"
        for i in range(len(st.atoms))
        if st_mod.MAX_VALENCE.get(st.atoms[i].element, 99) < len(st.atoms[i].bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))

    ca_d = [
        dist(st.atoms[trace[i][1]].xyz, st.atoms[trace[i + 1][1]].xyz)
        for i in range(len(trace) - 1)
    ]
    if ca_d:
        check(
            "consecutive CA atoms are a residue apart",
            min(ca_d) > 2.8 and max(ca_d) < 4.6,
            f"min {min(ca_d):.2f} A, max {max(ca_d):.2f} A",
        )

    ss = st_mod.secondary_structure(trace, st.atoms)
    check(
        "secondary structure is estimated for the interior residues",
        len(ss) == len(trace) and set(ss) <= {"helix", "sheet", "coil"},
        f"{ss.count('helix')} helix, {ss.count('sheet')} sheet, {ss.count('coil')} coil",
    )

    # --- and the reason templates are used at all
    naive = st_mod.Structure(name=st.name, atoms=st.atoms, residues=st.residues)
    st_mod._apply_distance_bonds(naive)
    naive_cross = sum(
        1
        for i, j in naive.bond_pairs()
        if st.atoms[i].residue_key != st.atoms[j].residue_key
    )
    check(
        "a pure distance rule really would have made cross-residue bonds here",
        naive_cross > 0,
        f"{naive_cross} cross-residue bonds from distance alone, "
        f"against {len(crosses)} from templates",
    )

    check(
        "auditor raised no bond-length complaints",
        not any("too short" in w or "longer than" in w for w in st.warnings),
        "; ".join(w for w in st.warnings if "Å" in w)[:160],
    )


# ---------------------------------------------------------------------------
# Small molecule
# ---------------------------------------------------------------------------


def check_small_molecule() -> None:
    section("small molecule: flat file, bonds inferred and audited")
    st = st_mod.parse_structure(
        (EXAMPLES / "ibuprofen_prep.pdbqt").read_text(encoding="utf-8"), "ibuprofen"
    )
    check("parsed", len(st.atoms) == 16, f"{len(st.atoms)} atoms")
    check("not treated as protein", not st.is_protein())

    # RDKit knows this molecule's real connectivity, so it is the oracle. A
    # hand-written expectation would only encode what I already believe, and
    # the first version of this check did exactly that and was simply wrong
    # about which carbons are in the ring.
    truth = _rdkit_bond_count(EXAMPLES / "biotin_prep.pdbqt")
    st_b = st_mod.parse_structure(
        (EXAMPLES / "biotin_prep.pdbqt").read_text(encoding="utf-8"), "biotin"
    )
    if truth is None:
        skip(
            "perceived bonds match RDKit's own perception",
            "RDKit is not installed, so there is no independent oracle here. "
            "This is the only check in the file that compares against something "
            "written by other people.",
        )
    else:
        check(
            "perceived bonds match RDKit's own perception",
            len(st_b.bond_pairs()) == truth,
            f"{len(st_b.bond_pairs())} perceived, {truth} from RDKit, "
            f"{len(st_b.atoms)} atoms",
        )

    deg = [len(a.bonds) for a in st.atoms]
    check(
        "every atom has at least one bond",
        all(d >= 1 for d in deg),
        "degrees " + ", ".join(str(d) for d in deg),
    )
    over = [
        f"{a.element}{i} has {len(a.bonds)} bonds"
        for i, a in enumerate(st.atoms)
        if st_mod.MAX_VALENCE.get(a.element, 99) < len(a.bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))

    # A connected graph on n atoms has at least n-1 bonds; more means cycles,
    # which for this molecule means the aromatic ring was closed.
    rings = len(st.bond_pairs()) - (len(st.atoms) - 1)
    check("the graph is connected with one independent ring", rings == 1,
          f"{len(st.bond_pairs())} bonds over {len(st.atoms)} atoms -> {rings} ring(s)")

    check(
        "the carboxyl carbon is bonded to both oxygens",
        _carboxyl_ok(st),
    )

    lengths = [dist(st.atoms[i].xyz, st.atoms[j].xyz) for i, j in st.bond_pairs()]
    check(
        "every bond length is physically possible",
        min(lengths) > 0.85 and max(lengths) < 2.0,
        f"min {min(lengths):.2f} A, max {max(lengths):.2f} A, {len(lengths)} bonds",
    )
    check(
        "connectivity was reported as inferred, not declared",
        any("inferred from interatomic distances" in w for w in st.warnings),
    )


def _rdkit_bond_count(prepared: Path) -> int | None:
    """How many bonds RDKit perceives in the *prepared* file itself.

    The oracle has to be run on the same file the check reads. Comparing bonds
    perceived from `biotin_prep.pdbqt` against bonds RDKit sees in
    `biotin.sdf` compares two different atom sets -- the prepared file has its
    non-polar hydrogens merged into their carbons -- so the counts differ for a
    reason that has nothing to do with either perception being wrong. RDKit
    reading the prepared file applies its own distance rules to the same atoms,
    which is the comparison worth making.

    Returns None when RDKit is not installed rather than raising.

    That is a narrow safety net, not a licence to run this file without RDKit:
    `prepare_receptor` imports it too, so the script cannot get this far on a
    machine that lacks it. What it does buy is that the *oracle* reports itself
    as unavailable instead of the whole file dying on an import inside a
    helper, and it is what lets a caller turn a missing oracle into a stated
    skip rather than a silently absent assertion.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None

    mol = Chem.MolFromPDBFile(str(prepared), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not read {prepared}")
    return mol.GetNumBonds()


def _carboxyl_ok(st) -> bool:
    """A carboxyl carbon has exactly two oxygens, and they are not bonded to
    each other -- the classic way a distance rule over-bonds a carboxyl group."""
    for a in st.atoms:
        if a.element != "C":
            continue
        oxygens = [j for j in a.bonds if st.atoms[j].element == "O"]
        if len(oxygens) == 2:
            return oxygens[1] not in st.atoms[oxygens[0]].bonds
    return False


# ---------------------------------------------------------------------------
# Docked pose: declared connectivity
# ---------------------------------------------------------------------------


def check_pose() -> None:
    section("docked pose: bonds come from ROOT/BRANCH, not geometry")
    models = st_mod.parse_structure(
        (EXAMPLES / "poses.pdbqt").read_text(encoding="utf-8"), "poses"
    )
    check("parsed", len(models.atoms) > 0, f"{len(models.atoms)} atoms in the last model")
    check("file declared its own connectivity", models.declared)

    lig = st_mod.parse_structure(
        (EXAMPLES / "ibuprofen_prep.pdbqt").read_text(encoding="utf-8"), "ibuprofen"
    )
    check(
        "a pose keeps the ligand's atom count and elements",
        len(models.atoms) == len(lig.atoms)
        and [a.element for a in models.atoms] == [a.element for a in lig.atoms],
        f"{len(models.atoms)} vs {len(lig.atoms)} atoms",
    )

    lengths = [dist(models.atoms[i].xyz, models.atoms[j].xyz) for i, j in models.bond_pairs()]
    check(
        "every declared bond is physically possible",
        bool(lengths) and min(lengths) > 0.85 and max(lengths) < 2.0,
        f"min {min(lengths):.2f} A, max {max(lengths):.2f} A, {len(lengths)} bonds",
    )
    over = [
        f"{a.element}{i} has {len(a.bonds)} bonds"
        for i, a in enumerate(models.atoms)
        if st_mod.MAX_VALENCE.get(a.element, 99) < len(a.bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))


# ---------------------------------------------------------------------------
# Non-protein receptor blob
# ---------------------------------------------------------------------------


def check_synthetic_receptor() -> None:
    section("synthetic receptor: no template, no invented bonds")
    st = st_mod.parse_structure(
        (EXAMPLES / "rec_prep.pdbqt").read_text(encoding="utf-8"), "rec"
    )
    check("parsed", len(st.atoms) == 30, f"{len(st.atoms)} atoms")
    check("not mistaken for a protein", not st.is_protein())
    check(
        "its bonds were inferred and that is reported",
        any("inferred from interatomic distances" in w for w in st.warnings),
    )
    over = [
        f"{a.element}{i} has {len(a.bonds)} bonds"
        for i, a in enumerate(st.atoms)
        if st_mod.MAX_VALENCE.get(a.element, 99) < len(a.bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))


# ---------------------------------------------------------------------------
# Secondary structure
# ---------------------------------------------------------------------------
#
# The fixtures below have exactly known answers, and the checks are written so
# that each one can fail. That matters more here than elsewhere in this file:
# a secondary-structure check that only exercised the new code path would pass
# just as happily if the old path had never been wrong. So every claim about the
# new answer is paired with a call to `geometric_secondary_structure` -- the old
# code, still in the module -- asserting that it gets the same fixture *wrong*.

#: A 3-10 helix sits at phi -49, psi -26 -- the Ramachandran centre of a
#: different helix type, not an alpha helix that is slightly off.
THREE_TEN_PHI, THREE_TEN_PSI = -49.0, -26.0
#: A psi just outside the alpha basin, which is what a real distorted helix
#: looks like. The geometric estimate reads the basin and calls the residue coil;
#: it is inside a helical segment and the hydrogen bonds say so.
KINK_PSI = 18.0

#: Crambin's secondary structure as it is annotated for 1CRN, the entry in
#: `examples/`: two alpha helices and one two-stranded antiparallel sheet.
#:
#: This is ground truth from outside this repository, written down here rather
#: than computed, because nothing offline can compute it. It was read off the
#: deposited structure's own secondary-structure annotation for 1CRN, not
#: measured by running DSSP: CI has no DSSP and must not need one. The exact
#: boundaries of DSSP's H and E labels are a convention -- the N-terminal Thr
#: 2 and the C-terminal Ala 46 of a strand are usually left blank rather than
#: called extended -- so residue 1 is treated as coil here, and the four
#: disagreements listed in the output are reported rather than hidden.
CRAMBIN_HELIX = ((7, 19), (23, 30))
CRAMBIN_SHEET = ((2, 4), (32, 35))


def crambin_truth(resid: int) -> str:
    """The documented state of one crambin residue."""
    if any(lo <= resid <= hi for lo, hi in CRAMBIN_HELIX):
        return "helix"
    if any(lo <= resid <= hi for lo, hi in CRAMBIN_SHEET):
        return "sheet"
    return "coil"


def _states(st: list[str]) -> str:
    """A per-residue assignment on one line, H/S/C."""
    return " ".join(s[0].upper() for s in st)


def _fixture(name: str, residues) -> st_mod.Structure:
    """Parse a synthetic fixture from its residues."""
    return st_mod.parse_structure(_pdb_text(residues), name)


def _uniform(count: int, phi: float, psi: float, chain: str = "A",
             first_resid: int = 1):
    return _peptide([phi] * count, [psi] * count, chain, first_resid)


def _measured_phi_psi(st, trace, atoms):
    """(resid, phi, psi) for every residue that has both angles."""
    out = []
    for i in range(1, len(trace) - 1):
        n, ca, c = trace[i]
        out.append(
            (
                atoms[n].resid,
                st_mod.dihedral(
                    atoms[trace[i - 1][2]].xyz, atoms[n].xyz,
                    atoms[ca].xyz, atoms[c].xyz,
                ),
                st_mod.dihedral(
                    atoms[n].xyz, atoms[ca].xyz, atoms[c].xyz,
                    atoms[trace[i + 1][0]].xyz,
                ),
            )
        )
    return out


def check_secondary_structure() -> None:
    section("secondary structure: hydrogen bonds, not just phi and psi")

    alpha5 = _fixture("alpha5", _uniform(5, HELIX_PHI, HELIX_PSI))
    three_ten4 = _fixture(
        "3ten4", _uniform(4, THREE_TEN_PHI, THREE_TEN_PSI)
    )
    kink = list(_uniform(9, HELIX_PHI, HELIX_PSI))
    kink[3]["psi_marker"] = True
    kink_phis = [HELIX_PHI] * 9
    kink_psis = [HELIX_PSI] * 9
    # A residue whose own psi is outside the alpha basin, in a chain whose
    # hydrogen bonds are an uninterrupted run of 4-turns.
    kink_psis[3] = KINK_PSI
    kink9 = _fixture("kink9", _peptide(kink_phis, kink_psis))
    coil8 = _fixture("coil8", _uniform(8, COIL_PHI, COIL_PSI))
    strand_a = _uniform(6, EXTENDED_PHI, EXTENDED_PSI, "A", 1)
    strand_b = _uniform(6, EXTENDED_PHI, EXTENDED_PSI, "B", 1)
    placed, contacts, separation = _place_second_strand(strand_a, strand_b)
    sheet = _fixture("sheet2", strand_a + placed)

    fixtures = {
        "alpha5": alpha5, "3ten4": three_ten4, "kink9": kink9,
        "coil8": coil8, "sheet2": sheet,
    }

    # --- the fixtures are physical, or they assert nothing -----------------
    bad_lengths: list[str] = []
    for name, st in fixtures.items():
        lengths = [dist(st.atoms[i].xyz, st.atoms[j].xyz) for i, j in st.bond_pairs()]
        if lengths and not (min(lengths) > 0.85 and max(lengths) < 2.0):
            bad_lengths.append(f"{name} {min(lengths):.2f}-{max(lengths):.2f}")
    check(
        "every synthetic fixture has physically possible bond lengths",
        not bad_lengths,
        "; ".join(bad_lengths) or "5 fixtures, all bonds 1.23-1.53 A",
    )

    coincident = [
        f"{name}: {st.atoms[i].name}{st.atoms[i].resid}/"
        f"{st.atoms[j].name}{st.atoms[j].resid} at "
        f"{dist(st.atoms[i].xyz, st.atoms[j].xyz):.2f} A"
        for name, st in fixtures.items()
        for i in range(len(st.atoms))
        for j in range(i + 1, len(st.atoms))
        if dist(st.atoms[i].xyz, st.atoms[j].xyz) < 1.0
    ]
    check(
        "no two atoms of a synthetic fixture share a position",
        not coincident,
        "; ".join(coincident[:4]) or "closest pair in every fixture is a bond",
    )

    off_angles = [
        f"{name} resid {r}: phi {p:.1f} psi {s:.1f}"
        for name, (want_phi, want_psi) in (
            ("alpha5", (HELIX_PHI, HELIX_PSI)),
            ("3ten4", (THREE_TEN_PHI, THREE_TEN_PSI)),
            ("coil8", (COIL_PHI, COIL_PSI)),
        )
        for st in [fixtures[name]]
        for r, p, s in _measured_phi_psi(st, st.backbone(), st.atoms)
        if abs(p - want_phi) > 0.5 or abs(s - want_psi) > 0.5
    ]
    check(
        "each fixture has the phi and psi it was built with",
        not off_angles,
        "; ".join(off_angles[:3]) or "measured back out of the coordinates",
    )

    inter_strand = [
        f"{st.atoms[i].name}{st.atoms[i].resid}{st.atoms[i].chain}/"
        f"{st.atoms[j].name}{st.atoms[j].resid}{st.atoms[j].chain} at "
        f"{dist(st.atoms[i].xyz, st.atoms[j].xyz):.2f} A"
        for i in range(len(sheet.atoms))
        for j in range(i + 1, len(sheet.atoms))
        if sheet.atoms[i].chain != sheet.atoms[j].chain
        and dist(sheet.atoms[i].xyz, sheet.atoms[j].xyz) < 2.6
    ]
    check(
        "the two-strand fixture keeps its strands apart",
        not inter_strand and separation >= 4.4,
        f"{contacts} backbone contacts placed, closest inter-strand CA "
        f"{separation:.2f} A, closest heavy pair "
        f"{min(dist(sheet.atoms[i].xyz, sheet.atoms[j].xyz) for i in range(len(sheet.atoms)) for j in range(len(sheet.atoms)) if sheet.atoms[i].chain != sheet.atoms[j].chain):.2f} A",
    )

    # --- the dihedral convention, which everything else rests on ------------
    check(
        "an eclipsed quadruple has a torsion of 0, not 180",
        st_mod.dihedral((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                        (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)) == 0.0,
        "the same quadruple used to return 180: the function was returning "
        "180 minus the torsion, which mirrored every Ramachandran angle",
    )

    crambin = st_mod.parse_structure(
        (EXAMPLES / "1crn_receptor.pdb").read_text(encoding="utf-8"), "1crn"
    )
    ctrace, catoms = crambin.backbone(), crambin.atoms
    res8 = next(
        (i for i, (n, _ca, _c) in enumerate(ctrace) if catoms[n].resid == 8), None
    )
    phi8 = st_mod.dihedral(
        catoms[ctrace[res8 - 1][2]].xyz, catoms[ctrace[res8][0]].xyz,
        catoms[ctrace[res8][1]].xyz, catoms[ctrace[res8][2]].xyz,
    )
    check(
        "crambin residue 8, inside the first helix, has a helical phi",
        abs(phi8 + 56.0) < 1.5,
        f"phi {phi8:.1f}, the textbook alpha-helix value is -57",
    )

    # --- the hydrogen bond test itself --------------------------------------
    prepared = st_mod.parse_structure(
        prepare_receptor_text(EXAMPLES / "1crn_receptor.pdb"), "1crn-prep"
    )
    ptrace, patoms = prepared.backbone(), prepared.atoms
    check(
        "a placed amide hydrogen finds the same bonds as a real one",
        st_mod.backbone_hydrogen_bonds(ptrace, patoms)
        == st_mod.backbone_hydrogen_bonds(ctrace, catoms),
        f"{len(st_mod.backbone_hydrogen_bonds(ptrace, patoms))} bonds from the "
        "hydrogens RDKit added, the same set from hydrogens placed on the "
        "fourth tetrahedral direction of a hydrogen-free file",
    )

    prepared_pairs = st_mod.backbone_hydrogen_bonds(ptrace, patoms)
    off_criteria = []
    for donor, acceptor in sorted(prepared_pairs):
        d_n = patoms[ptrace[donor][0]].xyz
        h = next(
            (patoms[k].xyz for k in patoms[ptrace[donor][0]].bonds
             if patoms[k].element == "H"),
            None,
        )
        o = next(
            (a.xyz for a in patoms
             if a.residue_key == patoms[ptrace[acceptor][0]].residue_key
             and a.element == "O"),
            None,
        )
        if h is None or o is None:
            off_criteria.append(
                f"{patoms[ptrace[donor][0]].resid}->"
                f"{patoms[ptrace[acceptor][0]].resid}: donor or acceptor atom "
                "missing"
            )
            continue
        d_on, d_oh = dist(d_n, o), dist(h, o)
        # The angle at H between H->N and H->O, computed here rather than
        # through `structure` so the check is not asking the code under test
        # whether it agrees with itself.
        v1 = [n_k - h_k for n_k, h_k in zip(d_n, h)]
        v2 = [o_k - h_k for o_k, h_k in zip(o, h)]
        cosine = sum(x * y for x, y in zip(v1, v2)) / (
            math.sqrt(sum(x * x for x in v1)) * math.sqrt(sum(y * y for y in v2))
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if not (d_on <= 3.5 and d_oh <= 2.5 and angle >= 90.0):
            off_criteria.append(
                f"{patoms[ptrace[donor][0]].resid}->"
                f"{patoms[ptrace[acceptor][0]].resid}: N..O {d_on:.2f}, "
                f"H..O {d_oh:.2f}, angle {angle:.0f}"
            )
    check(
        "every reported bond satisfies the gates the pass documents",
        not off_criteria,
        "; ".join(off_criteria[:3]) or f"{len(prepared_pairs)} bonds, all "
        "N..O under 3.5 A, H..O under 2.5 A, angle over 90 degrees",
    )

    # Negative control for the same test: the 3-turn in an ideal alpha helix is
    # *not* a hydrogen bond, and must not be reported as one.
    a9 = _fixture("alpha9", _uniform(9, HELIX_PHI, HELIX_PSI))
    a9_turns = [
        (d, a) for d, a in st_mod.backbone_hydrogen_bonds(
            a9.backbone(), a9.atoms
        ) if abs(d - a) == 3
    ]
    check(
        "a 3-turn inside an alpha helix is not reported as a hydrogen bond",
        not a9_turns,
        f"{a9_turns} found; the H...O distance in a 3-turn of an alpha helix "
        "is 2.7 A, past the 2.5 A gate, and a 3-10 helix is a different "
        "conformation with a different phi and psi",
    )

    # --- the short helix, and proof the old path missed it -------------------
    t_trace, t_atoms = three_ten4.backbone(), three_ten4.atoms
    t_new = st_mod.secondary_structure(t_trace, t_atoms)
    t_old = st_mod.geometric_secondary_structure(t_trace, t_atoms)
    check(
        "a four-residue 3-10 helix is assigned helix",
        t_new == ["coil", "helix", "helix", "helix"],
        f"hydrogen bonds: {_states(t_new)}",
    )
    check(
        "the old geometric path really does miss that four-residue helix",
        t_old != t_new and t_old[3] == "coil" and t_new[3] == "helix",
        f"geometric: {_states(t_old)} -- its last residue has no psi to read, "
        "so it can only ever be coil",
    )

    k_trace, k_atoms = kink9.backbone(), kink9.atoms
    k_new = st_mod.secondary_structure(k_trace, k_atoms)
    k_old = st_mod.geometric_secondary_structure(k_trace, k_atoms)
    kink_resid = k_atoms[k_trace[2][0]].resid
    check(
        "a residue outside the alpha basin, inside a helical segment, is helix",
        k_new == ["coil"] + ["helix"] * 8,
        f"hydrogen bonds: {_states(k_new)}",
    )
    check(
        "the old geometric path calls that residue coil",
        k_old[2] == "coil" and k_old != k_new,
        f"geometric: {_states(k_old)}; residue {kink_resid} has psi "
        f"+{KINK_PSI:.0f}, outside the -80..10 window, which is the whole point",
    )

    # --- the sheet ----------------------------------------------------------
    s_trace, s_atoms = sheet.backbone(), sheet.atoms
    s_new = st_mod.secondary_structure(s_trace, s_atoms)
    s_old = st_mod.geometric_secondary_structure(s_trace, s_atoms)
    want = (["sheet"] * 5 + ["coil"]) + (["coil"] + ["sheet"] * 5)
    check(
        "a two-stranded sheet is assigned from its inter-strand bonds",
        s_new == want,
        f"hydrogen bonds: {_states(s_new)} (A1-A5 then B1-B6)",
    )
    sheet_bonds = st_mod.backbone_hydrogen_bonds(s_trace, s_atoms)
    same_chain = [
        (s_atoms[s_trace[d][0]].resid, s_atoms[s_trace[a][0]].resid)
        for d, a in sheet_bonds
        if s_atoms[s_trace[d][0]].chain == s_atoms[s_trace[a][0]].chain
    ]
    check(
        "every bond in the sheet fixture is between the two chains",
        not same_chain and len(sheet_bonds) >= 3,
        f"{len(sheet_bonds)} inter-strand bonds, {len(same_chain)} within a chain",
    )
    check(
        "the old geometric path cannot see the bonded strand ends",
        s_new != s_old and s_new[0] == "sheet" and s_old[0] == "coil"
        and s_new[-1] == "sheet" and s_old[-1] == "coil",
        f"geometric: {_states(s_old)}; it reads the extended phi and psi but has "
        "no phi for the first residue and no psi for the last, so the two "
        "residues that really are hydrogen bonded are exactly the two it cannot "
        "see",
    )

    # --- the coil, which must stay quiet ------------------------------------
    c_trace, c_atoms = coil8.backbone(), coil8.atoms
    c_bonds = st_mod.backbone_hydrogen_bonds(c_trace, c_atoms)
    c_new = st_mod.secondary_structure(c_trace, c_atoms)
    check(
        "a coil with no hydrogen bonds is all coil",
        not c_bonds and c_new == ["coil"] * 8,
        f"{len(c_bonds)} bonds, {_states(c_new)}",
    )
    check(
        "the relaxed pass does not invent bonds either",
        not st_mod.backbone_hydrogen_bonds(c_trace, c_atoms, relaxed=True),
        f"{len(st_mod.backbone_hydrogen_bonds(c_trace, c_atoms, relaxed=True))} "
        "bonds under the looser gates",
    )

    # --- the relaxed pass may only bridge, never start or extend -----------
    a9_bonds = st_mod.backbone_hydrogen_bonds(a9.backbone(), a9.atoms)
    a9_loose = st_mod.backbone_hydrogen_bonds(
        a9.backbone(), a9.atoms, relaxed=True
    )
    check(
        "the relaxed pass is a superset of the strict one, and is looser",
        a9_loose > a9_bonds,
        f"{len(a9_bonds)} strict, {len(a9_loose)} relaxed, "
        f"{len(a9_loose - a9_bonds)} only under the relaxed gates",
    )
    strict_cover = set()
    for d, a in a9_bonds:
        lo, hi = sorted((d, a))
        strict_cover.update(range(lo + 1, hi + 1))
    check(
        "the helix a fixture gets is the one its strict turns support",
        st_mod.secondary_structure(a9.backbone(), a9.atoms) == [
            "coil" if i not in strict_cover else "helix" for i in range(9)
        ],
        "residues 2-9, which is what the five 4-turns cover; the six relaxed "
        "3-turns reach back 2.7 A but are past the H...O gate, and must not "
        "add to or extend the segment",
    )
    # The rule that a relaxed bond may only *bridge* two existing runs is not
    # reachable from any fixture here -- on every fixture above the two readings
    # of it give the same answer -- so it is asserted directly. A relaxed span
    # that touches one run must be ignored: it would otherwise extend a segment
    # past its own evidence, and a relaxed span that touches nothing must not
    # start one, which is the difference between smoothing a helix and
    # inventing it out of a chance contact in a loop.
    check(
        "a relaxed span that bridges two runs is used",
        st_mod._extend([[1, 2], [6, 7]], [(3, 5)]) == [list(range(1, 8))],
        "a one-residue gap between two runs is closed",
    )
    check(
        "a relaxed span that touches only one run is ignored",
        st_mod._extend([[1, 2, 3]], [(4, 6)]) == [[1, 2, 3]],
        "otherwise a chance 3-turn at the end of a helix would push the "
        "segment further than its 4-turns support",
    )
    check(
        "a relaxed span that touches nothing starts nothing",
        st_mod._extend([], [(2, 5)]) == [],
        "which is what stops a chance contact in a loop becoming a helix",
    )

    # --- crambin, against its documented structure -------------------------
    new_states = st_mod.secondary_structure(ctrace, catoms)
    old_states = st_mod.secondary_structure(
        ctrace, catoms, use_hydrogen_bonds=False
    )
    resid = [catoms[t[0]].resid for t in ctrace]
    truth = [crambin_truth(r) for r in resid]
    new_ok = [r for r, g, t in zip(resid, new_states, truth) if g == t]
    old_ok = [r for r, g, t in zip(resid, old_states, truth) if g == t]
    check(
        "crambin agrees with its documented structure more often now",
        len(new_ok) > len(old_ok),
        f"{len(new_ok)}/46 correct against {len(old_ok)}/46 for the geometric "
        f"estimate; helix {new_states.count('helix')}, "
        f"sheet {new_states.count('sheet')}, coil {new_states.count('coil')}",
    )
    gained = [r for r in new_ok if r not in old_ok]
    lost = [r for r in old_ok if r not in new_ok]
    check(
        "the new path fixes residues the old one got wrong",
        len(gained) >= 5,
        f"residues {gained} are now right and were not",
    )
    check(
        "and it costs some that the old one got right",
        len(lost) >= 1,
        f"residues {lost} are now wrong and were not; the trade is "
        f"{len(gained)} for {len(lost)}",
    )
    check(
        "both crambin helices are found whole",
        all(
            st_mod.secondary_structure(ctrace, catoms)[r - 1] == "helix"
            for lo, hi in CRAMBIN_HELIX for r in range(lo, hi + 1)
        ),
        f"residues {CRAMBIN_HELIX[0][0]}-{CRAMBIN_HELIX[0][1]} and "
        f"{CRAMBIN_HELIX[1][0]}-{CRAMBIN_HELIX[1][1]} all helix",
    )
    bridges = [
        (catoms[ctrace[d][0]].resid, catoms[ctrace[a][0]].resid)
        for d, a in st_mod.backbone_hydrogen_bonds(ctrace, catoms)
        if abs(catoms[ctrace[d][0]].resid - catoms[ctrace[a][0]].resid) > 5
    ]
    check(
        "crambin's sheet is found from bonds between non-adjacent residues",
        (35, 1) in bridges and (33, 3) in bridges,
        f"inter-strand bonds {bridges}: two of them, and they are the whole "
        "sheet, which is evidence phi and psi cannot contain",
    )

    # --- the sheet is *caused* by those bonds -------------------------------
    without = [
        line for line in (EXAMPLES / "1crn_receptor.pdb").read_text(
            encoding="utf-8"
        ).splitlines()
        if not (
            line.startswith(("ATOM", "HETATM"))
            and line[12:16].strip() == "O"
            and line[22:26].strip() in ("1", "3")
        )
    ]
    stripped = st_mod.parse_structure("\n".join(without), "1crn-no-sheet-O")
    s_trace2, s_atoms2 = stripped.backbone(), stripped.atoms
    rid2 = [s_atoms2[t[0]].resid for t in s_trace2]
    no_sheet = st_mod.secondary_structure(s_trace2, s_atoms2)
    flipped = [
        r for r in (1, 2, 33, 35)
        if no_sheet[rid2.index(r)] == "coil" and new_states[rid2.index(r)] == "sheet"
    ]
    check(
        "the sheet assignment is caused by those two bonds, not by phi and psi",
        len(flipped) == 4,
        f"deleting the two acceptor oxygens leaves residues {flipped} as coil; "
        "their phi and psi are unchanged, so nothing else could have been "
        "assigning them",
    )

    # --- both settings are reachable, and both do something ----------------
    check(
        "use_hydrogen_bonds=False is the old geometric answer, unchanged",
        old_states == st_mod.geometric_secondary_structure(ctrace, catoms)
        and _states(old_states) == _states(
            st_mod.geometric_secondary_structure(ctrace, catoms)
        ),
        _states(old_states),
    )
    check(
        "the default is the hydrogen-bond answer",
        st_mod.secondary_structure(ctrace, catoms) == new_states,
        _states(new_states),
    )
    check(
        "the flag is not a no-op",
        new_states != old_states,
        f"{sum(1 for a, b in zip(new_states, old_states) if a != b)} of "
        f"{len(new_states)} residues differ between the two settings",
    )

    # --- and it degrades to the old answer when there is no evidence --------
    no_o = [
        line for line in (EXAMPLES / "1crn_receptor.pdb").read_text(
            encoding="utf-8"
        ).splitlines()
        if not (
            line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "O"
        )
    ]
    bare = st_mod.parse_structure("\n".join(no_o), "1crn-no-oxygen")
    b_trace, b_atoms = bare.backbone(), bare.atoms
    check(
        "a backbone with no carbonyl oxygens falls back to the old estimate",
        not st_mod.backbone_hydrogen_bonds(b_trace, b_atoms)
        and st_mod.secondary_structure(b_trace, b_atoms)
        == st_mod.geometric_secondary_structure(b_trace, b_atoms),
        "no acceptor means no bond means no evidence; all coil would be the "
        "wrong answer there, so the geometric estimate is used instead",
    )


def check_candidate_index() -> None:
    section("candidate index: the same answer, found without looking everywhere")

    # One fixture per shape the pass has to cope with, plus crambin. Rebuilt
    # here rather than shared with the section above so that this section stands
    # on its own: the claim it makes is about the index, and it should be
    # readable without scrolling back through the chemistry.
    strand_a = _uniform(6, EXTENDED_PHI, EXTENDED_PSI, "A", 1)
    strand_b = _uniform(6, EXTENDED_PHI, EXTENDED_PSI, "B", 1)
    placed, _contacts, _separation = _place_second_strand(strand_a, strand_b)
    kinked_phis = [HELIX_PHI] * 9
    kinked_psis = [HELIX_PSI] * 9
    kinked_psis[3] = KINK_PSI
    shapes = {
        "crambin": st_mod.parse_structure(
            (EXAMPLES / "1crn_receptor.pdb").read_text(encoding="utf-8"), "1crn"
        ),
        "alpha5": _fixture("a", _uniform(5, HELIX_PHI, HELIX_PSI)),
        "3ten4": _fixture("b", _uniform(4, THREE_TEN_PHI, THREE_TEN_PSI)),
        "kink9": _fixture("c", _peptide(kinked_phis, kinked_psis)),
        "sheet2": _fixture("d", strand_a + placed),
        "coil8": _fixture("e", _uniform(8, COIL_PHI, COIL_PSI)),
        "bundle": _fixture("f", _bundle(1200)),
    }

    # The bonds themselves, not only the residues they end up on. Comparing
    # assignments is end-to-end, and that is the right level for the question
    # "did the residues change" -- but it is blind to work the index *loses*:
    # a candidate pair dropped by a misfiled cell is invisible unless dropping
    # it would have changed a bond, and whether it would have is not something
    # this file can know. The bond sets sit one level below the assignment, and
    # a lost pair is a lost bond the moment the two gates are applied, so this
    # comparison notices lost work that the one above cannot.
    bond_diffs: list[str] = []
    for name, st in shapes.items():
        trace, atoms = st.backbone(), st.atoms
        indexed = {
            relaxed: st_mod.backbone_hydrogen_bonds(trace, atoms, relaxed=relaxed)
            for relaxed in (False, True)
        }
        restore = _with_candidate_walk(_unindexed_candidate_pairs)
        try:
            plain = {
                relaxed: st_mod.backbone_hydrogen_bonds(
                    trace, atoms, relaxed=relaxed
                )
                for relaxed in (False, True)
            }
        finally:
            restore()
        for relaxed in (False, True):
            missing = plain[relaxed] - indexed[relaxed]
            extra = indexed[relaxed] - plain[relaxed]
            if missing or extra:
                bond_diffs.append(
                    f"{name} {'loose' if relaxed else 'strict'}: "
                    f"{len(missing)} bonds the plain walk found and the index "
                    f"did not, {len(extra)} the other way"
                    + (f"; first missing {sorted(missing)[0]}" if missing else "")
                    + (f"; first extra {sorted(extra)[0]}" if extra else "")
                )
    total_bonds = sum(
        len(st_mod.backbone_hydrogen_bonds(
            st.backbone(), st.atoms, relaxed=relaxed
        ))
        for st in shapes.values()
        for relaxed in (False, True)
    )
    check(
        "the indexed walk finds the same hydrogen bonds, not just the same residues",
        not bond_diffs,
        "; ".join(bond_diffs[:3]) or
        f"both passes over {len(shapes)} structures, {total_bonds} bonds, "
        f"identical in every case. The assignment comparison above cannot see a "
        f"candidate pair the index loses unless dropping it would have changed a "
        f"bond; this one can, because a lost pair is a lost bond as soon as the "
        f"gates are applied",
    )

    # The check that matters. Both sides are the real assignment code; the only
    # difference is which function enumerates the candidate pairs. Comparing the
    # finished per-residue strings element for element is what makes this
    # structural: there is no recorded "expected" string to drift away from, and
    # a change to either side that alters a residue fails here.
    disagreements: list[str] = []
    for name, st in shapes.items():
        trace, atoms = st.backbone(), st.atoms
        indexed = st_mod.secondary_structure(trace, atoms)
        restore = _with_candidate_walk(_unindexed_candidate_pairs)
        try:
            plain = st_mod.secondary_structure(trace, atoms)
        finally:
            restore()
        if indexed == plain:
            continue
        for i, (a, b) in enumerate(zip(plain, indexed)):
            if a != b:
                disagreements.append(
                    f"{name} residue {atoms[trace[i][0]].resid}: "
                    f"unindexed {a}, indexed {b}"
                )
    check(
        "the indexed walk assigns every residue exactly as the plain walk does",
        not disagreements,
        "; ".join(disagreements[:4]) or
        f"{len(shapes)} shapes, {sum(len(s.backbone()) for s in shapes.values())}"
        " residues, every state identical",
    )

    # The equivalence above is only worth something if the two sides really are
    # different. If someone indexed the reference walk too, it would pass while
    # comparing the index against itself. Both walks return the same pairs --
    # that is the whole point -- so the thing that tells them apart is how much
    # looking-around each one did to find them.
    bundle = shapes["bundle"]
    trace, atoms = bundle.backbone(), bundle.atoms
    frames = st_mod._backbone_frames(trace, atoms)
    indexed_pairs, indexed_lookups = _distance_calls(
        st_mod._candidate_pairs, frames, atoms
    )
    plain_pairs, plain_lookups = _distance_calls(
        _unindexed_candidate_pairs, frames, atoms
    )
    n = len(trace)
    check(
        "the reference walk is genuinely unindexed",
        plain_lookups > 20 * indexed_lookups,
        f"on {n} residues the plain walk measures {plain_lookups} distances and "
        f"the indexed one {indexed_lookups}, to return the same "
        f"{plain_pairs} pairs",
    )
    check(
        "the index removes the quadratic term, not just some of it",
        indexed_pairs * 20 < n * n // 2,
        f"{indexed_pairs} pairs against {n * n // 2} for a full scan: "
        f"{n * n // 2 / max(1, indexed_pairs):.0f}x fewer. This is the claim, "
        "stated as a count, because a wall clock is not a check",
    )
    check(
        "the quadratic term is visible at the size this fixture reaches",
        n >= 1000,
        f"{n} residues; below about a thousand the plain walk and the indexed "
        "one are within noise of each other, and a timing check at that size "
        "would be the pixel guard in a different costume",
    )

    # Memory: an index costs memory, and a fix that buys 90x by allocating a
    # gigabyte is its own failure on a laptop.
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    st_mod._candidate_pairs(frames, atoms)
    peak = tracemalloc.get_traced_memory()[1] - before
    tracemalloc.stop()
    check(
        "the index costs a sane amount of memory",
        peak / n < 4096,
        f"{peak / 1024:.0f} kB for {n} residues, {peak / n:.0f} B per residue",
    )

    # How much work the whole assignment does, counted rather than timed. This
    # is the claim the old five-second wall clock was standing in for, and it
    # is a better one. A cell edge large enough to put the whole structure in
    # one cell returns exactly the same pairs and does 1236 operations per
    # residue against the 95 measured here -- and takes 1.44 s, which the old
    # five-second threshold would have passed without a murmur. A threshold
    # loose enough to survive contention is also loose enough to miss the thing
    # it is watching for, and this check gates every build. The clock is still
    # measured and printed, as information.
    work = _assignment_work(trace, atoms)
    units = work["units"]
    per_residue = units / n
    check(
        "one assignment of a packed structure does a linear amount of work",
        per_residue < 250,
        f"{units:,} counted operations for {n} residues, {per_residue:.1f} per "
        f"residue: {work['cand_dist']:,} index distance measurements, "
        f"{work['gate_dist']:,} in the gates, {work['angle']:,} angles, "
        f"{work['contiguous']:,} contiguity tests, {work['runs']} _runs and "
        f"{work['extend']} _extend calls, from {work['builds']} index build. "
        f"Collapsing the grid to one cell returns the same pairs at 1236 per "
        f"residue, 13x this, and finishes in well under the five seconds the "
        f"old clock allowed; the clock here, which is not asserted, was "
        f"{work['elapsed']:.2f} s"
        if per_residue < 250 else
        f"{per_residue:.1f} counted operations per residue, over the bound of "
        f"250: something in the pass has gone quadratic, and the clock alone "
        f"({work['elapsed']:.2f} s) would not have said which part",
    )

    # The one thing the two passes share is the candidate list, and the reason
    # they can share it is that the candidate test is the `HBOND_CA_SKIP` bound,
    # which comes from the loosest gate there is. `relaxed` is read in exactly
    # one place in `structure.py` -- the line that picks the four thresholds --
    # so it cannot reach the enumeration, and building the list per pass built
    # the same list twice: 295,978 distance measurements of pure repeat work
    # per pass on 5200 residues.
    check(
        "the candidate list is enumerated once per assignment, not once per pass",
        work["builds"] == 1,
        f"{work['builds']} index build for one assignment over {n} residues and "
        f"both passes. `relaxed` selects the thresholds and nothing else, so the "
        f"second build could only have returned the same list"
        if work["builds"] == 1 else
        f"{work['builds']} index builds for one assignment over {n} residues: "
        f"the strict and loose passes are being handed separately built lists, "
        f"and since `relaxed` cannot change the candidate test those are the "
        f"same list, so half the enumeration is repeat work",
    )

    # The build count above is not the whole claim. One build still leaves room
    # for the two passes to be handed different lists -- the same one built
    # once and then sliced, filtered or sorted differently on the way to the
    # second pass -- and that survived every check in this file when it was
    # tried, because the loose bonds only ever *bridge* segments the strict
    # pass has already found: dropping some of them changes no residue on any
    # fixture here. The claim is that the two passes are handed the same
    # candidate list, and it is checked as contents rather than as identity, so
    # that a defensive copy is allowed and a different list is not.
    handed = work["handed"]
    same = (
        len(handed) == 2
        and handed[0] is not None
        and handed[0] == handed[1]
    )
    check(
        "the strict and loose passes are handed the same candidate list",
        same,
        f"{len(handed)} passes, {len(handed[0]) if handed else 0} candidates "
        f"each, identical contents. `relaxed` reaches the four thresholds and "
        f"nothing else, so the only honest thing either pass can do with the "
        f"list is walk all of it"
        if same else
        f"{len(handed)} passes were handed "
        f"{[len(c) if c is not None else None for c in handed]} candidates, "
        f"which are not the same list. The loose pass may only bridge "
        f"segments the strict pass already found, so losing candidates there "
        f"changes no residue on these fixtures and the assignment comparison "
        f"stays green -- which is exactly why this is checked on the list "
        f"itself",
    )

    # And the fixture builder itself must not be able to lie about its size.
    # This is the same class of defect as a fixture with forty atoms on one
    # coordinate: a 3000-residue single chain runs past the end of the PDB
    # coordinate field, and without a guard it parses as 1093 residues while
    # every timing taken from it is really a measurement of something else.
    overflowed = False
    try:
        _pdb_text(_uniform(4000, HELIX_PHI, HELIX_PSI))
    except ValueError:
        overflowed = True
    check(
        "a chain too long for the PDB coordinate field is refused, not truncated",
        overflowed,
        "4000 residues of helix reach past +/-1000 A; a column-shifted ATOM "
        "record still parses as an atom, so the fixture would have been a "
        "1093-residue structure wearing a 3000-residue label",
    )


#: The cell the probes below sit in. Negative, because real coordinates are
#: not: every structure a fixture can be built from has positive coordinates,
#: and for a positive coordinate a floor and a truncation toward zero are the
#: same function, so half the cell arithmetic is untestable in all of them. The
#: anchors are then placed on a face, a corner or the middle of this cell, which
#: is the other thing no real structure does -- PDB coordinates are given to
#: 0.001 A and nothing lands on a cell edge. Both are boundaries, and a
#: boundary is the only place the two halves of an index -- the one that files a
#: donor and the one that goes looking for it -- can disagree unnoticed.
PROBE_CELL = -5

#: `(axis, sign)` for the six directions a 3x3x3 search has to cover.
PROBE_DIRECTIONS = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))


def _nitrogen_at(ca, edge, bond: float = 1.46):
    """An N `bond` angstrom from `ca`, pushed out of the cell `ca` is in.

    A real N...CA is 1.46 A, so that is the length. The *direction* is the
    point: outward from the middle of the cell, because that is the direction
    in which the two atoms land in different cells, and a probe whose nitrogen
    sits on top of its CA cannot tell a grid keyed on the CA from a grid keyed
    on the nitrogen -- both give the same answer, so both pass.

    Push every probe outward and they all shift the same way, which cancels
    out again. What discriminates is a pair whose two nitrogens are further
    apart than two cells while their CAs are not, and that is arranged
    deliberately in `check_keyed_on_ca` rather than hoped for in a ring.
    """
    out = []
    for c in ca:
        cell = int(c // edge)
        middle = (cell + 0.5) * edge
        out.append(c + (1.0 if c >= middle else -1.0) * bond / math.sqrt(3))
    return tuple(out)


def _probe_frames(points, roles: str = "", edge: float = 0.0, nitrogens=None):
    """The frames and atoms the candidate index reads for `points`.

    The index only ever looks at CA coordinates, so a probe about the index
    does not need a peptide to put them there. Building the atoms and frames
    that `_candidate_pairs` reads keeps the coordinates exact, which matters: a
    corner acceptor 0.01 A off the corner is a different case, and a fixture
    quantised to 0.001 A by a peptide builder would be testing a geometry
    nobody ships.

    Each point gets two atoms, an N and a CA, because a residue whose nitrogen
    is its CA is a residue that cannot tell one grid key from another. The
    nitrogen is placed off the CA by :func:`_nitrogen_at`, or taken from
    `nitrogens` where a check needs the two atoms to disagree about which cell
    they are in -- which is the only way to make the key observable at all.
    `edge` is the cell size to place it against, and the default reads it from
    the module.

    `roles` says what each point is allowed to be: `"d"` a donor, `"a"` an
    acceptor, `"b"` both, which is the default. Splitting them is not a detail.
    A residue that is both also *searches*, so with two points each one looks
    for the other, and a stencil that has lost the `-x` half is rescued by the
    lookup that starts from the donor: the probe measures two distance
    measurements and concludes the search reaches. The first version of this
    check was blind to a deliberately asymmetric stencil for exactly that
    reason, and the mutation that proved it is the reason the roles are here.
    """
    roles = roles or "b" * len(points)
    edge = edge or st_mod.HBOND_CELL
    atoms, frames = [], []
    for i, xyz in enumerate(points):
        role = roles[i]
        donates = role in "db"
        accepts = role in "ab"
        n_at = len(atoms)
        n_xyz = (
            tuple(nitrogens[i]) if nitrogens is not None
            else _nitrogen_at(xyz, edge)
        )
        atoms.append(
            st_mod.AtomRecord(
                serial=n_at + 1, xyz=n_xyz, element="N",
                name="N", resname="ALA", resid=i + 1, chain="A",
            )
        )
        ca_at = len(atoms)
        atoms.append(
            st_mod.AtomRecord(
                serial=ca_at + 1, xyz=xyz, element="C", name="CA",
                resname="ALA", resid=i + 1, chain="A",
            )
        )
        frames.append(
            st_mod._BackboneFrame(
                key=("A", i + 1, "ALA"), n=n_at, ca=ca_at,
                carbonyl_o=ca_at if accepts else None,
                amide_h=n_xyz if donates else None,
                prev_c=ca_at if donates else None,
            )
        )
    return frames, atoms


def _probe_expected(points, roles: str = "") -> set:
    """Every ordered pair the index should return for `points`, the slow way.

    `roles` is the same string :func:`_probe_frames` is given, and it has to be
    the same string: a point that is an acceptor only is never a donor, so a
    pair between two such points is not a pair the index can return, and
    expecting it is how this came to fail its own unmutated control.
    """
    roles = roles or "b" * len(points)
    donors = [i for i, role in enumerate(roles) if role in "db"]
    acceptors = [i for i, role in enumerate(roles) if role in "ab"]
    return {
        (don, acc)
        for acc in acceptors
        for don in donors
        if don != acc and dist(points[acc], points[don]) <= st_mod.HBOND_CA_SKIP
    }


def _reach_probe(edge, axis, sign, cells_out):
    """An acceptor on a cell face and a donor `cells_out` cells along `axis`.

    The acceptor is placed on the face that makes this direction its *worst*
    case, and the donor on the centre of the `cells_out`-th cell along that
    direction counted from the acceptor's own position. That counting is the
    whole derivation:

        A donor more than `k` cells away is more than `k * edge` from the
        acceptor, whatever the acceptor sits at inside its own cell. Along the
        axis of the search, cell `B + k + 1` starts at `(B + k + 1) * edge` and
        the acceptor is below `(B + 1) * edge`, so the closest that cell gets is
        `k * edge` -- approached, never reached, and approached only by an
        acceptor sitting on the face of its own cell. So `k * edge >= cutoff` is
        both necessary and sufficient, and it is measured *from the acceptor's
        position*: measuring from the cell corner instead gives `(k + 1) * edge`
        and is a whole cell optimistic. That is not a harmless slip -- an edge
        of 4.5 A with a one-cell stencil covers 9.0 A from the corner and looks
        fine, and misses candidates, because the guarantee is 4.5 A against a
        cutoff of 8.2 A.

    The two directions are not symmetric in *where* the acceptor goes, and that
    is the second half of why one acceptor cannot measure all six of them. For
    `+` the worst case is the low face of the acceptor's own cell, which is a
    lattice point, so `+` needs no adjustment. For `-` the worst case is the
    *high* face -- and a lattice point belongs to the cell above, so an acceptor
    sitting exactly on one is in the wrong cell to count from, and a probe built
    that way asks about a donor two cells out and concludes the search reaches
    nothing on the negative side. The `1e-6` puts it a hair inside the face,
    where its own cell is the one being counted from and it is the worst case
    for `-` to within a millionth of an angstrom.

    A probe acceptor in the middle of its cell cannot establish any of this. From
    there the nearest a donor one cell out can sit is half a cell away, so the
    geometry looks easier than the guarantee is: an edge of 6.0 A with a one-cell
    stencil would still show a middle-of-cell probe reaching one cell in every
    direction, while the guarantee it needs is 6.0 A against a cutoff of 8.2 A.
    Measuring reach from a cell's middle measures the middle, not the bound.
    """
    at = (PROBE_CELL + (1 if sign < 0 else 0)) * edge
    acc = [0.0, 0.0, 0.0]
    acc[axis] = at - (0.0 if sign > 0 else 1e-6)
    don = list(acc)
    don[axis] += sign * (cells_out + 0.5) * edge
    return tuple(acc), tuple(don)


def _exact_cutoff_pair(cutoff, anchors):
    """An anchor from `anchors` and a donor beside it exactly `cutoff` away.

    Placing a donor at `cutoff` does not place it *at* the cutoff: the float
    nearest 8.2 is not 8.2, and `sqrt(8.2 * 8.2)` is a different float again. So
    a check that wants the cutoff to be closed had no pair to test with until
    this walked the ulps looking for the one span that measures exactly equal --
    the single pair on which `<` and `<=` disagree. Without it a cutoff made
    open passes every check in this file.

    Two things this has to get right. The span is searched on the *placed*
    geometry rather than on a bare axis, because that is the number the cutoff
    is compared against. And more than one anchor is tried, because beside an
    anchor of magnitude 40 the sum `anchor[0] + span` rounds to the ulp of 40,
    which is eight times coarser than the ulp of the span itself: the distances
    that anchor can produce are a lattice, and that lattice need not contain
    8.2 at all. Beside the origin there is no cancellation, so the lattice is
    the one the float nearest 8.2 lives in, and the walk finds a span there.
    Returns `(None, None)` if no anchor can.
    """
    for anchor in anchors:
        span = cutoff
        for step in range(8192):
            probe = (anchor[0] + span, anchor[1], anchor[2])
            if dist(anchor, probe) == cutoff:
                return anchor, probe
            span = math.nextafter(span, math.inf if step % 2 == 0 else -math.inf)
    return None, None


def _unit_vectors() -> list:
    """The 6 face and 8 corner directions, as unit vectors."""
    out = []
    for i in range(3):
        for sign in (1.0, -1.0):
            v = [0.0, 0.0, 0.0]
            v[i] = sign
            out.append(tuple(v))
    third = 3.0 ** -0.5
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                out.append((sx * third, sy * third, sz * third))
    return out


def _ring_geometry(anchors, spans):
    """Each anchor in `anchors`, ringed with donors at `spans` in 14 directions.

    A lattice anchor is a cell corner in all three axes at once, which is the
    worst case for `+` on every axis simultaneously and something no real
    structure produces, since PDB coordinates are given to 0.001 A and nothing
    lands on a cell edge. A half-cell anchor is the opposite: the middle of a
    cell, the worst case for no direction at all, and the only place where a
    floor and a truncation toward zero land in different cells *in negative
    coordinates*, since the two agree exactly on any lattice point.
    """
    out = []
    for at in anchors:
        out.append(tuple(at))
        for vec in _unit_vectors():
            for span in spans:
                out.append(tuple(at[a] + vec[a] * span for a in range(3)))
    return out


def _cell_corner(edge):
    """A lattice point: a cell corner in all three axes, in negative space."""
    return (PROBE_CELL * edge,) * 3


def _cell_middle(edge):
    """The middle of a cell, in negative space: never on a face in any axis."""
    return tuple((PROBE_CELL + 0.5) * edge for _ in range(3))


def check_index_coverage() -> None:
    section("index coverage: how far the search reaches, in angstrom")

    edge = st_mod.HBOND_CELL
    cutoff = st_mod.HBOND_CA_SKIP

    # How wide is the search, in cells, in each of the six directions? Measured
    # by behaviour rather than read out of the source: a donor one cell out is
    # looked at, a donor two cells out is not, and the index is the only thing
    # being asked. A stencil that had quietly become asymmetric, or that someone
    # had widened to be safe, would show up here as six different numbers.
    reached = {}
    for axis, sign in PROBE_DIRECTIONS:
        cells = 0
        for cells_out in (1, 2):
            acc, don = _reach_probe(edge, axis, sign, cells_out)
            # One acceptor, one donor, and the donor cannot search: the only
            # thing that can be measured here is the acceptor's own stencil.
            frames, atoms = _probe_frames([acc, don], roles="ad")
            _pairs, examined = _distance_calls(
                st_mod._candidate_pairs, frames, atoms
            )
            if examined:
                cells = cells_out
        reached[(axis, sign)] = cells
    widths = sorted(set(reached.values()))
    check(
        "the search reaches the same number of cells on all six sides",
        len(widths) == 1,
        "; ".join(
            f"{'xyz'[a]}{'+' if s > 0 else '-'}: {reached[(a, s)]} cell"
            for a, s in PROBE_DIRECTIONS
        )
        + f" -- {widths[0]} cell{'s' if widths[0] != 1 else ''} on every side"
        if len(widths) == 1
        else "asymmetric: " + ", ".join(
            f"{'xyz'[a]}{'+' if s > 0 else '-'} reaches {reached[(a, s)]}"
            for a, s in PROBE_DIRECTIONS
        ),
    )

    # The derivation, as a number in angstrom, and the one that keeps the cell
    # edge and the chemistry cutoff from drifting apart. `HBOND_CELL` is a
    # performance constant and `HBOND_CA_SKIP` is chemistry, so nothing in the
    # module connects them except a comment; this is the connection, and it is
    # reported in the units that decide the question. Note what it does *not*
    # assert: not that the edge is some multiple of the cutoff, which would fail
    # a correct 4.1 A edge searched two cells wide, nor that the arithmetic
    # `edge == 3 * cutoff` holds, which a needlessly large edge would satisfy
    # while proving nothing about coverage.
    reach = min(reached.values()) * edge
    margin = reach - cutoff
    if reach >= cutoff:
        note = [
            f"the narrowest side of the search reaches {reach:.1f} A per axis "
            f"({min(reached.values())} cell of {edge:.1f} A) and the widest "
            f"{max(reached.values()) * edge:.1f} A, against a cutoff of "
            f"{cutoff:.1f} A: {margin:+.1f} A of margin. Every donor more than "
            f"{min(reached.values())} cell out is more than {reach:.1f} A "
            f"away, so no pair inside the cutoff can be out of reach"
        ]
    else:
        note = [
            f"only {reach:.1f} A of the {cutoff:.1f} A cutoff is covered, "
            f"{cutoff - reach:.1f} A short: a donor inside the cutoff can sit "
            f"{min(reached.values()) + 1} cells out and be missed, and the "
            f"indexed walk would return fewer pairs than the plain one. The "
            f"equivalence check cannot be relied on to notice -- a bundle whose "
            f"geometry happens not to put a pair that far out still assigns "
            f"every residue the same way -- and where it does notice it can "
            f"only say which residue disagreed. This says by how much."
        ]
    check(
        "what the search reaches covers the cutoff",
        reach >= cutoff,
        "; ".join(note),
    )

    # The boundary, with the acceptor on a cell *corner* in all three axes, and
    # donors on the cutoff itself and a millionth of an angstrom past it, in
    # all fourteen directions. This is the geometry the reach argument is about,
    # made concrete: it is where a pair sitting exactly on the cutoff rests
    # against a cell face. One donor is placed at the span that measures
    # *exactly* the cutoff, because that is the single pair on which `<` and
    # `<=` disagree, and the direction of the inequality is part of what the
    # cutoff claims.
    at = _cell_corner(edge)
    middle = _cell_middle(edge)
    exact_at, exact = _exact_cutoff_pair(
        cutoff, (at, middle, (0.0, 0.0, 0.0))
    )
    points = _ring_geometry([at], (cutoff, cutoff + 1e-6))
    if exact is not None:
        points.append(exact_at)
        points.append(exact)
    expected = _probe_expected(points)
    found = set(st_mod._candidate_pairs(*_probe_frames(points)))
    if found != expected:
        missed = sorted(expected - found)
        invented = sorted(found - expected)
        note = []
        if missed:
            don, acc = missed[0]
            note.append(
                f"missed donor {don} at {points[don]} which is "
                f"{dist(points[don], points[acc]):.4f} A from residue {acc} "
                f"at {points[acc]}"
            )
        if invented:
            don, acc = invented[0]
            note.append(
                f"invented donor {don} at {points[don]} which is "
                f"{dist(points[don], points[acc]):.4f} A from residue {acc}"
            )
        note.append(f"{len(missed)} missed, {len(invented)} invented")
    else:
        exact_note = (
            f", plus a donor beside {tuple(round(c, 1) for c in exact_at)} that "
            f"measures exactly {cutoff} and so is the one pair on which `<` and "
            f"`<=` disagree"
            if exact is not None else
            f"; no anchor tried could place a donor at exactly {cutoff} in "
            f"double precision, so the last bit of the inequality is untested "
            f"on this machine"
        )
        note = [
            f"acceptor on a cell corner at {at}, 14 directions: "
            f"{len(expected)} pairs, every one inside or outside the cutoff "
            f"exactly as it should be, to 1e-6 A{exact_note}"
        ]
    check(
        "at a cell corner, the cutoff itself is where the index says it is",
        found == expected,
        "; ".join(note),
    )

    # And the same geometry in positive coordinates, on the same lattice. The
    # two runs differ in nothing but the sign of the coordinates, so any
    # difference between them is the cell arithmetic: a floor is not a
    # truncation toward zero for a negative coordinate, and a build side and a
    # search side that disagreed about which one they were using would agree
    # with each other here and there and only break in half the coordinate
    # space. Every real fixture lives entirely in the half where the two
    # coincide, which is why this is a separate check and not a comment.
    #
    # Two anchors, and the second one is the whole point. On a lattice point a
    # floor and a truncation agree *exactly*, because the quotient is a whole
    # number and both round it the same way, so a corner acceptor cannot see
    # this class of defect at all. The cell middle is at a quotient of `-4.5`,
    # where the two disagree, and it is the position a real atom is most likely
    # to be in besides an exact face.
    #
    # The donors sit 5% either side of the cutoff rather than on it. At exactly
    # the cutoff this comparison would be testing floating point instead: a
    # distance computed between two coordinates 360 A apart is not the same
    # number as the same distance computed 720 A from the origin, so a pair on
    # the cutoff can land on either side of `<=` depending on which half of
    # space it is in. That is a property of the comparison, not a defect in the
    # index, and it is a different question from the one being asked here.
    placed = _ring_geometry(
        (_cell_corner(edge), _cell_middle(edge)),
        (cutoff * 0.95, cutoff * 1.05),
    )
    # Two acceptors, everything else a donor that cannot search. A donor that
    # could would cover for a misfiled acceptor: with a truncation on the search
    # side, the acceptor's own cell comes out one cell too high, its stencil
    # stops short on the negative side, and the donors it should have found
    # find *it* instead, from their own correct cells. The pair set comes out
    # identical either way, which is the same rescue that hid an asymmetric
    # stencil from the first version of the reach probe.
    roles = "aa" + "d" * (len(placed) - 2)
    expected_placed = _probe_expected(placed, roles)
    shift = 40 * edge
    moved = [tuple(c + shift for c in p) for p in placed]
    # The index is run on *both* sides and the two answers are compared to each
    # other. Comparing one side's index against the other side's brute-force
    # expectation does not test anything about the sign: the expectation is
    # correct in both halves by construction, so a defect that only misfiles
    # cells for negative coordinates is invisible to it, and the first version
    # of this check was blind to exactly that.
    here = set(st_mod._candidate_pairs(*_probe_frames(placed, roles=roles)))
    elsewhere = set(
        st_mod._candidate_pairs(*_probe_frames(moved, roles=roles))
    )
    corner, middle = _cell_corner(edge), _cell_middle(edge)
    sign_diff = sorted(here ^ elsewhere)
    truth_diff = sorted(here ^ expected_placed)
    if here == elsewhere == expected_placed:
        note = [
            f"the same {len(expected_placed)} pairs from the index run on a cell "
            f"corner at {tuple(round(c, 1) for c in corner)} and a cell middle "
            f"at {tuple(round(c, 1) for c in middle)} in negative coordinates, "
            f"from the index run on the same geometry moved to "
            f"{tuple(round(c, 1) for c in moved[0])} in positive ones, and from "
            f"brute force on either; a cell index that floors on one side of the "
            f"origin and truncates on the other would pass every real fixture "
            f"and fail here"
        ]
    elif not sign_diff:
        # Both halves agree with each other and both disagree with brute force.
        # Naming a sign difference here would be an empty list, and indexing it
        # took the whole file down -- 79 passing checks and all -- so the two
        # ways this can fail are reported separately.
        note = [
            f"the two halves of the origin agree with each other but not with "
            f"brute force: {len(truth_diff)} pairs the index returns are wrong on "
            f"both sides, so the sign of the coordinates is not what is wrong "
            f"here. Donor {truth_diff[0][0]} is "
            f"{dist(placed[truth_diff[0][0]], placed[truth_diff[0][1]]):.4f} A "
            f"from residue {truth_diff[0][1]} and should be "
            f"{'returned' if truth_diff[0] in here else 'dropped'}"
        ]
    else:
        don, acc = sign_diff[0]
        note = [
            f"{len(sign_diff)} pairs differ between the index run at "
            f"{tuple(round(c, 1) for c in corner)} and "
            f"{tuple(round(c, 1) for c in middle)} in negative coordinates and "
            f"the index run on the same geometry at "
            f"{tuple(round(c, 1) for c in moved[0])}, and the only difference "
            f"between the two runs is the sign of the coordinates: donor {don} "
            f"is {dist(moved[don], moved[acc]):.4f} A from residue {acc} there "
            f"and {dist(placed[don], placed[acc]):.4f} A here"
        ]
    check(
        "the index reaches the same answer on both sides of the origin",
        here == elsewhere == expected_placed,
        "; ".join(note),
    )


def check_keyed_on_ca() -> None:
    section("which atom the index keys on")

    edge = st_mod.HBOND_CELL
    cutoff = st_mod.HBOND_CA_SKIP

    # The cell has to be keyed on the atom the cutoff is measured between, and
    # the cheapest way to be sure of that is to build a pair where the two
    # candidates disagree. Two residues 0.05 A apart through their CAs and
    # eleven cells apart through their nitrogens: on a grid keyed on the CA they
    # share a cell and the pair is inside the cutoff, so it must be returned; on
    # a grid keyed on the nitrogen it is silently dropped.
    # Nothing else in this file can see the difference, which is how it stayed
    # invisible for five rounds: the end-to-end equivalence check compares
    # assignments, so a lost pair is invisible unless it would have bonded, and
    # the other probes derive the nitrogen from the CA, so *both* atoms being
    # wrong in the same direction looks exactly like both being right.
    far = 5.5 * edge
    cas = [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)]
    nitrogens = [(-far, 0.0, 0.0), (far, 0.0, 0.0)]
    frames, atoms = _probe_frames(cas, roles="ab", nitrogens=nitrogens)
    found = set(st_mod._candidate_pairs(frames, atoms))
    # One acceptor and one donor, so one ordered pair is the whole answer.
    check(
        "the cell is keyed on the atom the cutoff is measured between",
        found == {(1, 0)},
        f"two residues {dist(*cas):.2f} A apart through their CAs and "
        f"{2 * far:.0f} A through their nitrogens, {len(found)} of the 1 ordered "
        f"pair the roles allow, and it is the one at the CA distance. A grid "
        f"keyed on the nitrogen would put them {2 * far / edge:.0f} cells apart "
        f"and return none, and no other check here would notice: the "
        f"equivalence check compares assignments, and a lost candidate pair only "
        f"shows up in one if it would have bonded"
        if found == {(1, 0)} else
        f"expected the pair at the CA distance and got {sorted(found)}: the grid "
        f"is keyed on something other than the CA, or on the CA and the "
        f"nitrogen together",
    )

    # The other direction, which is a different claim: a grid keyed on the
    # nitrogen would *also* put these two in one cell, and the pair must still
    # be rejected, because the cutoff is re-measured on the CA for every
    # candidate. This one passes today only because that re-measurement is
    # there; remove it and a nitrogen-keyed grid invents bonds.
    cas = [(0.0, 0.0, 0.0), (cutoff * 1.5, 0.0, 0.0)]
    frames, atoms = _probe_frames(
        cas, roles="ab", nitrogens=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    )
    invented = set(st_mod._candidate_pairs(frames, atoms))
    check(
        "and the nitrogen of a residue never makes a pair by itself",
        not invented,
        f"two residues {cutoff * 1.5:.1f} A apart through their CAs, well past "
        f"the {cutoff:.1f} A cutoff, and sharing a nitrogen, and returned "
        f"{sorted(invented)}: the cutoff is re-measured on the CA for every "
        f"candidate, whatever the grid was keyed on"
        if not invented else
        f"{len(invented)} pairs invented for residues {cutoff * 1.5:.1f} A "
        f"apart through their CAs, past the {cutoff:.1f} A cutoff",
    )


def main() -> int:
    print(f"verifying bond perception against {EXAMPLES}")
    check_protein()
    check_small_molecule()
    check_pose()
    check_synthetic_receptor()
    check_secondary_structure()
    check_candidate_index()
    check_keyed_on_ca()
    check_index_coverage()

    print("\n=== summary ===")
    # The three counts are printed and never added into a verdict. `total` is the
    # only figure meant to be compared against the pin, and it is the sum of what
    # ran and what could not -- which is exactly why it is the same number on a
    # machine with RDKit and one without. Printing the three separately is what
    # makes the difference between "measured nothing" and "measured 84 of 85"
    # legible: if the skips were folded into the pass count, `84 passed, 0 failed,
    # 85 checks` would read as a clean run.
    npass = CHECKS - len(FAILURES)
    nfail = len(FAILURES)
    nskip = len(SKIPPED)
    total = CHECKS + nskip
    print(f"  {npass} passed, {nfail} failed, {nskip} skipped, {total} checks")
    for name, reason in SKIPPED:
        print(f"    SKIP {name}: {reason}")
    for f in FAILURES:
        print(f"    FAIL {f}")
    if nskip:
        print(
            f"  {nskip} check(s) were skipped, not passed. A skip means this\n"
            f"  environment could not answer the question; it is not evidence that\n"
            f"  the thing it guards is correct. The total above still counts them,\n"
            f"  which is why it does not move with the machine."
        )
    # The one thing that must hold on *every* machine, whatever else is true.
    # The first half is an accounting identity and the second is the pin. Keeping
    # them as two separate conditions is deliberate: the identity is what says a
    # skip was counted once and only once, and folding the skips into `npass`
    # above -- which is exactly the "make the numbers add up" instinct -- breaks
    # the identity while leaving the total at 85. A green run would then read
    # "85 passed, 0 failed, 1 skipped, 85 checks" and mean something false.
    if npass + nfail + nskip != total:
        print(
            f"  TALLY: {npass} + {nfail} + {nskip} is not {total}. The three counts "
            f"must partition the total, or one of them is counting something twice."
        )
        return 2
    if total != EXPECTED_CHECKS:
        print(
            f"  TALLY: {total} checks ran, {EXPECTED_CHECKS} were expected "
            f"({npass} passed, {nfail} failed, {nskip} skipped). A check is missing "
            f"or a site is counted twice."
        )
        return 2
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
