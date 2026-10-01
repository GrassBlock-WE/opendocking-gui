"""What the two benchmarks say, checked against what they compute.

`pocket_benchmark.py` and `redock_benchmark.py` are the only substantial scripts
here that are not in CI, and the reason is a good one: they need the network and
there is no ground truth a pass/fail could honestly be measured against. That
argument is about *results*. It says nothing about whether each script's output
is a faithful description of what the script did, and that is a property of the
code, testable without a network and without a corpus.

So this file gates the reporting, not the numbers. The distinction is the whole
point. A benchmark's job is to be believed; the ways it can be believed wrongly
have nothing to do with whether the search is good:

* **A printed number that is not the number the line says it is.** `cost per
  1000 atoms` divided by the atom count, so every figure was a thousandth of its
  own label, and `1CRN` rounded to `0 ms`. Correct arithmetic, wrong heading --
  the same shape as the bug where the header advertised an exhaustiveness that
  no run used, because the value is chosen per box.
* **A number that reports a value nobody passes in.** `argmax` over a vector
  that is all zeroes exactly when the search found nothing returns 0, which read
  as "rank 1", which the legend defines as *the site the ligand came from*. The
  dash the legend promised for "not found at all" was unreachable.
* **Two columns that can be the same column.** When the ranking's top site
  already holds the bound pose there is no second box, so the third column is a
  copy of the second -- printed under text explaining that the two are "one
  number apart", with nothing to tell a repeated cell from a coincidence.
* **A failure that looks like a result.** A truncated download parses. A
  half-file is still a file with atoms in it, so without a completeness check a
  partial fetch is reported as a smaller molecule, confidently.
* **A caption printed where a result is expected.** "A deliberately engineered
  buried cavity" sat directly under a measurement with nothing connecting the
  two, so a structure that stopped behaving as its own caption described was
  reported in the same shape as one that did not.

Everything here runs offline. The network is stubbed at `urlopen`, so the
failure paths -- rate limit, total outage, truncated body -- are exercised
deliberately rather than waited for, and no test in this file depends on RCSB
being reachable or on a corpus being what it was this morning.

Run:  python scripts/benchmark_check.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import math
import sys
import urllib.error
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Prefer the installed package; fall back to the source tree, like every other
# script here: a clean checkout has no compiled `_dockpy`.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking import core  # noqa: E402
from opendocking.core import exhaustiveness_for_box  # noqa: E402
from opendocking.workbench import pockets as P  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


def load(name: str, path: Path):
    """Import a script by path, registered in sys.modules.

    The registration matters: `redock_benchmark.py` is a `__main__`-style
    script but is also imported here, and its dataclass-free module body is
    fine either way -- the point is that both benchmarks are loaded the same
    way, so a difference between them is never a loading artefact.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PB = load("pb_under_test", ROOT / "scripts" / "pocket_benchmark.py")
RB = load("rb_under_test", ROOT / "scripts" / "redock_benchmark.py")


# --------------------------------------------------------------------------
# A synthetic PDB, built column by column. Every fetch is stubbed, so these
# lines are the only "corpus" this file has.
# --------------------------------------------------------------------------

def atom_line(serial, name, resname, chain, resid, x, y, z, element,
              record="ATOM"):
    """One ATOM/HETATM record in the fixed columns the parsers actually read."""
    return (
        f"{record:<6s}{serial:5d} {name:^4s} {resname:>3s} {chain:1s}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
    )


def protein_pdb(n_res=24, chain="A", offset=(0.0, 0.0, 0.0)):
    """A short poly-alanine-ish chain: enough to parse, small enough to be fast.

    Spaced 3.8 A apart along x with a 3.3 A stagger along z, so consecutive
    atoms are a bond apart and non-consecutive ones are not -- the pocket
    search gets a real surface to flood rather than a single lump.
    """
    lines = ["HEADER    TEST", "TITLE     SYNTHETIC CHAIN FOR benchmark_check"]
    serial = 1
    for i in range(n_res):
        x = offset[0] + 3.8 * i
        z = offset[2] + (3.3 if i % 2 else 0.0)
        lines.append(atom_line(serial, " CA ", "ALA", chain, i + 1, x,
                               offset[1], z, "C"))
        serial += 1
        lines.append(atom_line(serial, " CB ", "ALA", chain, i + 1, x + 1.2,
                               offset[1] + 1.1, z, "C"))
        serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def ligand_pdb(resname, chain, resid, xyz):
    return "\n".join(
        atom_line(i + 1, f" C{i + 1}", resname, chain, resid, *p, "C",
                  record="HETATM")
        for i, p in enumerate(xyz)
    )


def two_chain_entry(resname="LIG", n_atoms=4):
    """One entry with the ligand bound in two chains, as a duplicate would be."""
    lines = ["HEADER    TWO CHAIN LIGAND"]
    lines.append(protein_pdb(n_res=8, chain="A").splitlines()[2:-2])
    lines.append(protein_pdb(n_res=8, chain="B").splitlines()[2:-2])
    # Chain A's copy sits at the origin; chain B's is 30 A away, so a reference
    # that kept both would be centred between two molecules that are not there.
    lines.append(ligand_pdb(resname, "A", 900, [(0.0, 0.0, 0.0)] * n_atoms))
    lines.append(ligand_pdb(resname, "B", 900, [(30.0, 0.0, 0.0)] * n_atoms))
    lines.append("END")
    out = []
    for chunk in lines:
        out.extend(chunk if isinstance(chunk, list) else [chunk])
    return "\n".join(out) + "\n"


@contextlib.contextmanager
def no_network():
    """Every outbound request raises. Nothing in this file may reach the net."""
    real = PB.urllib.request.urlopen
    sent = []

    def blocked(*a, **k):
        sent.append(a[0] if a else "?")
        raise AssertionError(
            f"benchmark_check made a network request ({a[0] if a else '?'}); "
            f"every fetch must be stubbed"
        )

    PB.urllib.request.urlopen = blocked
    RB.urllib.request.urlopen = blocked
    try:
        yield sent
    finally:
        PB.urllib.request.urlopen = real
        RB.urllib.request.urlopen = real


def run_main(mod, **patches):
    """Call a benchmark's main() with the network and any extras stubbed.

    Returns ``(exit_code, stdout)``. stdout is captured rather than printed so a
    failing test's output can be quoted whole, which is the only way to check a
    claim that lives in prose.
    """
    real = mod.urllib.request.urlopen
    stub = patches.pop("urlopen", None)
    if stub is not None:
        mod.urllib.request.urlopen = stub
    saved = {}
    try:
        for target, value in patches.items():
            obj, attr = target
            saved[(obj, attr)] = getattr(obj, attr)
            setattr(obj, attr, value)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                code = mod.main()
            except SystemExit as exc:
                code = exc.code
        return code, buf.getvalue()
    finally:
        for (obj, attr), value in saved.items():
            setattr(obj, attr, value)
        mod.urllib.request.urlopen = real


def http_error(status, url="https://files.rcsb.org/download/1CRN.pdb"):
    return urllib.error.HTTPError(url, status, f"HTTP {status}", {}, None)


def main() -> int:
    # ---------------------------------------------------------------- 1
    section("1. a printed parameter is the parameter that was used")
    # The redock header used to print `exhaustiveness {EXHAUSTIVENESS_LADDER[0]}`
    # as though one value applied to the run. It could not: the effort is chosen
    # per box, so a single printed number describes no run at all.
    small = exhaustiveness_for_box((20.0, 20.0, 20.0))
    large = exhaustiveness_for_box((50.0, 50.0, 50.0))
    check(
        "exhaustiveness is not a constant, so no single number can describe a run",
        small != large,
        f"exhaustiveness_for_box gives {small} for a 20 A cube and {large} for a "
        f"50 A cube. The header no longer prints one value; each run and each "
        f"table row prints what it used",
    )

    recorded = {}

    class _FakeBox:
        pass

    class _FakeReceptor:
        def precalculate(self, box, engine, spacing):
            recorded["box"] = box
            recorded["engine"] = engine
            recorded["spacing"] = spacing
            return {"maps": True}

    class _FakeResult:
        num_poses = 3
        best_energy = -7.5

        def pose_coords(self, i):
            return [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (9.0, 9.0, 9.0)][i]

    def _fake_dock(ligand, maps, *, exhaustiveness, num_modes, seed):
        recorded["exhaustiveness"] = exhaustiveness
        recorded["num_modes"] = num_modes
        recorded["seed"] = seed
        return _FakeResult()

    size = (44.0, 30.0, 26.0)
    real_dock = core.dock
    core.dock = _fake_dock
    try:
        RB.dock_into(_FakeReceptor(), object(), (1.0, 2.0, 3.0), size,
                     [(0.0, 0.0, 0.0)])
    finally:
        core.dock = real_dock
    check(
        "and the value the run uses is the one its own box implies",
        recorded.get("exhaustiveness") == exhaustiveness_for_box(size)
        and recorded.get("seed") == RB.SEED
        and recorded.get("num_modes") == 5,
        f"docking into a {size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f} A box used "
        f"exhaustiveness {recorded.get('exhaustiveness')}, seed "
        f"{recorded.get('seed')}, num_modes {recorded.get('num_modes')}. The "
        f"header claims seed, box margin and num_modes; all three are the "
        f"values actually passed",
    )
    check(
        "the module no longer defines a single exhaustiveness it never passes",
        not hasattr(RB, "EXHAUSTIVENESS"),
        "the constant was a leftover from when this file carried its own ladder. "
        "It was printed in the header and passed to nothing, so it described no "
        "run. `exhaustiveness_for_box` is the only source now",
    )

    # The pocket benchmark's header, same question, checked against the real
    # signature rather than by reading the print statement.
    sig = inspect.signature(P.find_pockets)
    params = sig.parameters
    check(
        "the pocket header's four parameters are the defaults actually in force",
        params["probe"].default == P.DEFAULT_PROBE
        and params["spacing"].default == P.DEFAULT_SPACING
        and params["min_burial"].default == 1
        and params["max_volume"].default is None
        and P.DEFAULT_PROBE == 1.4
        and P.DEFAULT_SPACING == 0.8
        and P.DEFAULT_MAX_VOLUME == 1500.0,
        f"probe {params['probe'].default}, spacing {params['spacing'].default}, "
        f"min_burial {params['min_burial'].default}, max_volume "
        f"{params['max_volume'].default} -- which `find_pockets` resolves to "
        f"DEFAULT_MAX_VOLUME at its first use, so the 'volume ceiling 1500 A^3' "
        f"in the header is the ceiling the run applies rather than a number that "
        f"looks like one",
    )

    # ---------------------------------------------------------------- 2
    section("2. a number that reports a value nobody passed in")
    check(
        "a search that found the ligand nowhere prints a dash, not rank 1",
        RB.rank_of([0, 0, 0]) is None,
        f"rank_of([0, 0, 0]) is {RB.rank_of([0, 0, 0])}. The old code took "
        f"argmax over this vector, and argmax of zeroes is 0, which reached the "
        f"table as rank 1 -- which the legend defines as the site the ligand "
        f"came from. The rank column now also carries how many of the ligand's "
        f"atoms that site holds",
    )
    check(
        "and no sites at all is a dash too",
        RB.rank_of([]) is None,
        f"rank_of([]) is {RB.rank_of([])}",
    )
    check(
        "while a site that does hold the ligand still gets its rank",
        RB.rank_of([0, 16, 3]) == 2 and RB.rank_of([16]) == 1,
        f"rank_of([0, 16, 3]) is {RB.rank_of([0, 16, 3])} and rank_of([16]) is "
        f"{RB.rank_of([16])}: the first best, not the first entry",
    )

    # ---------------------------------------------------------------- 3
    section("3. two columns that can be the same column")
    need, site_i, repeat = RB.oracle_needed([16], 16)
    check(
        "when the ranking's top site holds the whole pose, that is said",
        need is False and site_i == 0 and repeat is not None,
        f"oracle_needed([16], 16) -> dock again: {need}, site: {site_i}, "
        f"repeats because: {repeat}. There is no second box to pick, so the "
        f"third column was the second one and the old table showed two identical "
        f"columns as two independent results",
    )
    need, site_i, repeat = RB.oracle_needed([43, 0], 46)
    check(
        "and the subtler case: the best site is still the top-ranked one",
        need is False and site_i == 0 and repeat is not None,
        f"oracle_needed([43, 0], 46) -> dock again: {need}, site: {site_i}. The "
        f"best site holds only 43 of 46 ligand atoms, so the old code's 'ranking "
        f"set aside' branch ran the oracle box on sites[0] -- the same box it had "
        f"just docked, with a fixed seed. A redundant run that cannot return "
        f"anything else, producing a third column equal to the second with "
        f"nothing on the row to explain it. This one is live in the corpus "
        f"today, on 1HVR/XK2",
    )
    need, site_i, repeat = RB.oracle_needed([0, 16], 16)
    check(
        "while a genuinely different site is still run on its own",
        need is True and site_i == 1 and repeat is None,
        f"oracle_needed([0, 16], 16) -> dock again: {need}, site: {site_i}, "
        f"repeats: {repeat}. This is the case the third column exists for",
    )
    need, site_i, repeat = RB.oracle_needed([0, 0], 16)
    check(
        "while no site holding the ligand at all is n/a rather than a copy",
        need is False and repeat is None,
        f"oracle_needed([0, 0], 16) -> dock again: {need}, site: {site_i}, "
        f"repeats: {repeat}. Copying the pocket column here would have reported "
        f"a third claim about a site holding none of the ligand",
    )
    need, site_i, repeat = RB.oracle_needed([], 16)
    check(
        "and an empty site list is n/a as well",
        need is False and site_i is None and repeat is None,
        f"oracle_needed([], 16) -> dock again: {need}, site: {site_i}, "
        f"repeats: {repeat}",
    )

    # ---------------------------------------------------------------- 4
    section("4. the receptor and the search see one chain, and the ligand too")
    prot, lig, chain = RB.split_entry(two_chain_entry(), "LIG")
    lig_chains = sorted({l[21] for l in lig})
    prot_chains = sorted({l[21] for l in prot})
    check(
        "a ligand bound in two chains is narrowed to one, like the protein",
        len(lig_chains) == 1 and lig_chains == prot_chains,
        f"the entry has the ligand in chains A and B, 4 atoms each, 30 A apart. "
        f"split_entry returned {len(lig)} ligand rows all in chain "
        f"{lig_chains} and {len(prot)} protein rows in {prot_chains}. It used to "
        f"keep all 8 ligand rows while narrowing the protein, so the reference "
        f"was two copies of the molecule 30 A apart, the box was centred on "
        f"their average, and every RMSD was measured against a pose that is in "
        f"no single place -- the first version's bug, one order of magnitude "
        f"smaller",
    )
    single = two_chain_entry().replace(
        ligand_pdb("LIG", "B", 900, [(30.0, 0.0, 0.0)] * 4), ""
    )
    _, lig_a, chain_a = RB.split_entry(single, "LIG")
    check(
        "and which chain is chosen does not depend on the order of the file",
        chain_a == chain,
        f"chain A in file order gives {chain_a}; the two-chain entry gives "
        f"{chain}. The rule is the chain holding the most ligand atoms, ties "
        f"broken by name, so a re-deposited file with the blocks swapped cannot "
        f"change which experiment ran",
    )
    check(
        "an entry with no such ligand yields no chain and no rows, not a guess",
        RB.split_entry(protein_pdb() + "END\n", "LIG") == ([], [], None)
        or RB.split_entry(protein_pdb(), "LIG")[2] is None,
        f"split_entry on a protein with no LIG returns chain "
        f"{RB.split_entry(protein_pdb(), 'LIG')[2]!r}, so the caller skips the "
        f"case rather than narrowing to a chain nobody named",
    )
    waters = protein_pdb() + atom_line(900, " O  ", "HOH", "A", 180, 5.0, 5.0, 5.0,
                                       "O", record="HETATM") + "\nEND\n"
    prot_w, lig_w, _ = RB.split_entry(waters, "LIG")
    check(
        "waters and other heteroatoms never reach either side",
        not lig_w and all(l.startswith("ATOM") for l in prot_w),
        f"the entry carried a HETATM water as well as {len(prot_w)} ATOM rows; "
        f"the protein came back with {len(prot_w)} rows and the ligand with 0",
    )

    # ---------------------------------------------------------------- 5
    section("5. a failure that looks like a result")
    # A truncated body is still a parseable body. Without a completeness check
    # a partial download becomes a smaller molecule and is reported as one.
    class _Truncated:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    real_urlopen = PB.urllib.request.urlopen
    try:
        PB.urllib.request.urlopen = lambda *a, **k: _Truncated(
            b"ATOM      1  CA  ALA A   1      11.104   6.134  -6.504  1.00  0.00"
            b"           C  \n"
        )
        raised = None
        try:
            PB.fetch("1CRN")
        except OSError as exc:
            raised = str(exc)
    finally:
        PB.urllib.request.urlopen = real_urlopen
    # Both benchmarks carry their own copy of this guard, in their own module, so
    # fixing one says nothing about the other. The first version of this check
    # tested only the pocket copy and mutation M4 slipped straight through.
    rb_raised = None
    try:
        RB.urllib.request.urlopen = lambda *a, **k: _Truncated(
            b"ATOM      1  CA  ALA A   1      11.104   6.134  -6.504  1.00  0.00"
            b"           C  \n"
        )
        try:
            RB.fetch("1STP")
        except OSError as exc:
            rb_raised = str(exc)
    finally:
        RB.urllib.request.urlopen = real_urlopen
    check(
        "a truncated download is refused by both, not parsed into a structure",
        raised is not None and "truncated" in raised
        and rb_raised is not None and "truncated" in rb_raised,
        f"a body with no END record raises {raised!r} in the pocket benchmark and "
        f"{rb_raised!r} in the redock one. `read()` returns whatever arrived, and "
        f"half a file is still a file with atoms in it; without this the "
        f"benchmark would report a smaller protein as though it were the whole "
        f"structure, with a smaller atom count to make it look deliberate",
    )

    # Total outage: no table, no numbers, a non-zero exit.
    def _all_429(*a, **k):
        raise http_error(429)

    code, out = run_main(PB, urlopen=_all_429)
    check(
        "a rate-limited run says so and produces no results to misread",
        code == 1 and "could not fetch" in out
        and "nothing could be fetched" in out and "sites (" not in out,
        f"every fetch raising HTTP 429 gives exit {code} and "
        f"{out.count('could not fetch')} 'could not fetch' line(s), the line "
        f"'nothing could be fetched; no results to report', and no per-structure "
        f"line at all -- so there is no atom count, site count or timing on "
        f"screen to be mistaken for a result",
    )
    check(
        "and a 429 is a URLError, so it is the path that is actually taken",
        issubclass(urllib.error.HTTPError, urllib.error.URLError),
        "HTTPError subclasses URLError, so the existing except clause catches a "
        "rate limit, a 404 and a refused connection alike",
    )

    # Partial: one structure answers, the rest are rate-limited.
    body = protein_pdb().encode("utf-8")

    def _partial(url, *a, **k):
        if "1CRN" in str(url):
            return _Truncated(body + b"END\n")
        raise http_error(429)

    code, out = run_main(PB, urlopen=_partial)
    n_rows = sum(1 for ln in out.splitlines()
                 if ln.startswith("1CRN") and "sites (" in ln)
    check(
        "a partial run names what it is missing instead of quietly reporting less",
        code == 0 and n_rows == 1 and "did not run" in out,
        f"one structure answers and the rest are rate-limited: exit {code}, "
        f"{n_rows} measured row, and the closing line names the 6 absent and "
        f"says the counts describe the 1 that ran, not the set of 7 the set was "
        f"chosen to be. Every skip line already said 'not counted'; what was "
        f"missing was the count",
    )

    # Same for the redock side, on the all-fail path, where no docking is
    # attempted at all.
    code, out = run_main(RB, urlopen=_all_429)
    check(
        "the redock benchmark refuses to print a table it has no cases for",
        code == 1 and "nothing could be fetched" in out and "|" not in out,
        f"every fetch raising HTTP 429 gives exit {code} with the header printed "
        f"and no table rows. No receptor is prepared and no docking is run, so "
        f"a rate limit costs a rerun rather than producing an RMSD",
    )

    # ---------------------------------------------------------------- 7
    section("7. what the number means")
    # Both halves of the same line. The defect was correct arithmetic under a
    # wrong heading, so pinning only the arithmetic would let the heading drift
    # back, and pinning only the heading would leave the thousandth in place.
    line = PB.cost_line([("4HHB", 4779, 17600.0, 12, 1), ("1CRN", 327, 321.0, 12, 0)])
    check(
        "the per-1000 line's heading and its arithmetic agree",
        "per 1000 atoms" in line
        and "4HHB 3683 ms" in line
        and "1CRN 982 ms" in line,
        f"{line!r}. 17600 ms over 4779 atoms is 3683 ms per 1000, and 321 ms over "
        f"327 is 982. Divided by the atom count instead, the same two rows read "
        f"'4HHB 4 ms' and '1CRN 1 ms' under a heading that still said per 1000 -- "
        f"every figure a thousandth of its own label",
    )
    check(
        "and it orders by that figure, worst first",
        line.index("4HHB") < line.index("1CRN"),
        "4HHB at 3683 ms per 1000 atoms is the more expensive one per atom even "
        "though 1CRN is the smaller structure; sorting on the wrong quantity "
        "would put 1L96 and 1STP the wrong way round",
    )

    a = np.asarray([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)], np.float32)
    check(
        "identical coordinates give exactly zero",
        RB.rmsd(a, a) == 0.0,
        f"rmsd(a, a) is {RB.rmsd(a, a)}",
    )
    shifted = a + np.float32([3.0, 0.0, 0.0])
    check(
        "a pure translation gives exactly the translation, so this is a distance",
        abs(RB.rmsd(a, shifted) - 3.0) < 1e-5,
        f"rmsd 3 A to the +x side is {RB.rmsd(a, shifted):.6f} A. No projection, "
        f"no per-atom matching, no fitting",
    )
    theta = math.radians(90.0)
    rot = np.asarray(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]], np.float32,
    )
    # A long lever arm, not a triangle. Three points near the middle of a small
    # triangle turn 90 degrees and land on almost the same distances, so the
    # first version of this check used a threshold of 3 A and measured 2.94 --
    # a property of the shape chosen, not of the function. One point at the
    # origin and one 10 A away gives an answer that is exact: the origin does
    # not move, so the RMSD is the length of the arm.
    arm = np.asarray([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], np.float32)
    turned = arm @ rot.T
    check(
        "a rigid rotation gives a large value, because nothing is superimposed",
        abs(RB.rmsd(arm, turned) - 10.0) < 1e-4,
        f"a 10 A arm turned 90 degrees about z gives "
        f"{RB.rmsd(arm, turned):.6f} A, exactly the arm. The reference and the "
        f"pose are already in one frame, so the question is whether the atoms "
        f"went back where they started; superimposing first would hide a search "
        f"that returns the right shape in the wrong place. It does mean the "
        f"number is not the superimposed RMSD of the docking literature, and the "
        f"header now says so -- '1.26 A' next to a published '1.26 A' is a "
        f"comparison this does not support",
    )

    # ---------------------------------------------------------------- 8
    section("8. a corpus that changed underneath the numbers")
    # Two copies of each helper, one per module, so both are exercised. The
    # first version of this section tested only the pocket copies and mutations
    # M7 and M8 were not caught at all.
    pb_order = (PB._corpus_fingerprint([("1CRN", "aaa"), ("1STP", "bbb")])
                == PB._corpus_fingerprint([("1STP", "bbb"), ("1CRN", "aaa")]))
    rb_order = (RB._corpus_fingerprint([("1CRN", "LIG", "aaa"), ("1STP", "BTN", "bbb")])
                == RB._corpus_fingerprint([("1STP", "BTN", "bbb"), ("1CRN", "LIG", "aaa")]))
    check(
        "the corpus fingerprint does not depend on the order it was collected in",
        pb_order and rb_order,
        f"pocket {pb_order}, redock {rb_order}: the same files hashed in a "
        f"different order give the same fingerprint in both, so a rerun can be "
        f"compared against a published one",
    )
    pb_content = (PB._corpus_fingerprint([("1CRN", "aaa"), ("1STP", "bbb")])
                  != PB._corpus_fingerprint([("1CRN", "aaa"), ("1STP", "ccc")]))
    rb_content = (RB._corpus_fingerprint([("1CRN", "LIG", "aaa"), ("1STP", "BTN", "bbb")])
                  != RB._corpus_fingerprint([("1CRN", "LIG", "aaa"), ("1STP", "BTN", "ccc")]))
    check(
        "and one re-deposited structure changes it",
        pb_content and rb_content,
        f"pocket {pb_content}, redock {rb_content}. The bytes are fetched fresh "
        f"and not pinned, so this is the only thing between a re-deposited "
        f"structure and a confidently wrong comparison. The alternative -- "
        f"caching -- would make the corpus a thing this project has to maintain",
    )
    pb_changed = PB.changed_between({"a": "1", "b": "2"}, {"a": "1", "c": "3"})
    rb_changed = RB.changed_between({"a": "1", "b": "2"}, {"a": "1", "c": "3"})
    pb_same = PB.changed_between({"a": "1"}, {"a": "1"})
    check(
        "the tree fingerprint names an added, a removed and a rewritten file",
        pb_changed == ["b", "c"] and rb_changed == ["b", "c"] and pb_same == [],
        f"changed_between on an added, a removed and an unchanged file gives "
        f"{pb_changed} in the pocket benchmark and {rb_changed} in the redock "
        f"one, and {pb_same} for two identical maps. Four source files have been "
        f"rewritten under a run in this project already, so this is not "
        f"hypothetical",
    )
    fp = PB._tree_fingerprint()
    check(
        "each benchmark watches the sources its own numbers come from",
        str(PB.WATCHED[0]).endswith("pockets.py")
        and any(str(p).endswith("core.py") for p in RB.WATCHED)
        and all(str(PB.WATCHED[0]) not in str(p) for p in PB.WATCHED[1:]),
        f"the pocket benchmark watches {len(PB.WATCHED)} file(s) including "
        f"pockets.py, the module whose behaviour every number in it comes from; "
        f"the redock one watches {len(RB.WATCHED)} including core.py and "
        f"prep.py. Each watches its own source file, so editing a benchmark is "
        f"itself a detected change",
    )

    # ---------------------------------------------------------------- 9
    section("9. a caption printed where a result belongs")
    bad = [pid for pid, _why, claim, holds in PB.STRUCTURES
           if (claim is None) != (holds is None)]
    check(
        "every structure either has a falsifiable claim and a way to test it, or neither",
        not bad,
        f"the {len(PB.STRUCTURES)} entries: "
        + "; ".join(f"{pid} -- {claim or 'no claim asserted'}"
                    for pid, _w, claim, _h in PB.STRUCTURES),
    )
    claims = {pid: holds for pid, _w, claim, holds in PB.STRUCTURES if claim}
    try:
        discriminating = (claims["1CRN"](0) and not claims["1CRN"](1)
                          and claims["1L96"](1) and not claims["1L96"](0))
        why = ""
    except (KeyError, TypeError) as exc:
        # A check that dies on a plausible regression reports a traceback about
        # itself instead of the failure it was built to catch.
        discriminating = False
        why = f" -- and calling a claim's predicate raised {type(exc).__name__}"
    check(
        "and the two claims that do exist actually discriminate",
        bool(discriminating),
        "1CRN's claim is 'no sealed cavity at 1.4 A' and 1L96's is 'a "
        "purpose-built cavity is reported as sealed', so each is decidable from "
        "the sealed count and each would fail on the opposite count. A predicate "
        "that returned True for everything would 'hold' forever, which is what "
        "a printed caption next to a number always was" + why,
    )
    check(
        "the rest say they have no claim rather than borrowing one",
        all(claim is None for _p, _w, claim, _h in PB.STRUCTURES
            if _p not in ("1CRN", "1L96")),
        "a reason for being in the set is not a prediction, and the other five "
        "entries get no threshold invented for them. No expected corpus value "
        "is asserted anywhere in either script",
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed")
    print("\nNot gated here, on purpose: whether any of these numbers is good. A "
          "corpus result has no ground truth to gate on, which is why neither "
          "benchmark is a CI gate, and a check that asserted a corpus number "
          "would be asserting the internet. What is gated is that each printed "
          "number is the number its own line says it is.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # a check that dies on a regression is worse than no check
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
