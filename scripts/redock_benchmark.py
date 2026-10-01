"""Redocking benchmark: can the search find a pose that is already known?

This is the check the project has been missing, and it is deliberately not a
pass/fail. The three things it could be confused with are all different:

* **the search box** -- docked twice, once in a box derived from the bound
  ligand (which measures the *engine*) and once in the box the pocket search
  picks (which measures the *search* as well). Reporting only the first would
  make the pocket feature look validated when it was not exercised at all.
* **the scoring function** -- a pose can be found and scored wrongly.
* **the input** -- the bound ligand is taken from a crystal structure, and this
  reads its bonds from interatomic distances because a PDB ligand carries none.
  That gets connectivity right for a saturated chain and gets **aromaticity and
  formal charge wrong**, which changes both the rotatable-bond set and the
  partial charges. So these numbers are a floor on what the engine can do, not
  a measurement of it in isolation.

# The receptor and the search must see the same structure

An earlier version of this file searched the raw PDB and docked against
`prepare_receptor`'s output, and quietly measured two different molecules. On
1HVR -- HIV protease, a homodimer -- that meant searching 1826 atoms of both
chains while docking against 621 of one, and the 29 A "failure" that came out
was mostly this. `prepare_receptor` keeps the largest fragment and drops the
rest without being asked to.

So the receptor is narrowed to the chain the ligand is in, prepared once, and
the pocket search then reads **that same prepared file**. The workbench does
the same thing: it searches the receptor it is going to dock against, not the
file the user happened to open. Every step that could change which atoms are
present prints how many are left.

Nothing here is a CI gate: it needs a network, and each case runs a real
docking search. `scripts/benchmark_check.py` is the gate, and it needs neither --
it checks that the numbers below are the numbers the lines say they are, which
is a property of this file rather than of the corpus.

# Three columns, and the ways two of them can be the same column

The third column is meant to be a *different* box from the pocket's top-ranked
one, so that comparing the two separates a bad ranking from a bad box. It cannot
be different in two cases, and both used to reach the table looking like two
independent results:

* the top-ranked site already holds the whole bound pose -- there is no second
  box, so the numbers were copied from the pocket column and printed again;
* the top-ranked site holds only *part* of it -- then the best site still *is*
  the top-ranked one, so the oracle run repeated the pocket run on the same box.
  The seed is fixed, so it could not return anything else, and it cost a real
  docking run to say so.

Both are now labelled on the row, by `oracle_needed`. `rank` had the same
shape of problem: it was `argmax` over a per-site count of held ligand atoms,
and `argmax` of an all-zero vector -- which is exactly the vector a failed
search produces -- is 0, so "found nowhere" printed as rank 1, which the legend
defines as *the site the ligand came from*. The dash the legend promised was
unreachable.

Output is deliberately ASCII only. A source file with non-ASCII literals is
one PowerShell 5.1 `Set-Content -Encoding UTF8` away from being silently
mangled, and a benchmark that crashes on an encoding artefact measures
nothing.

Run:  python scripts/redock_benchmark.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking import core, prep  # noqa: E402
from opendocking.core import (  # noqa: E402
    EXHAUSTIVENESS_LADDER,
    GridBox,
    Receptor,
    exhaustiveness_for_box,
)
from opendocking.workbench import MoleculeView  # noqa: E402
from opendocking.workbench import pockets as P  # noqa: E402

#: ``(PDB id, ligand residue name, what the case is)``. Each is a complex a
#: small molecule is genuinely bound in, which is the only kind worth
#: redocking: a ligand that is not in a pocket to begin with has no "bound
#: pose" to recover.
CASES = [
    ("1STP", "BTN", "streptavidin + biotin, the textbook biotin case"),
    ("3PTB", "BEN", "trypsin + benzamidine, a small deep specificity pocket"),
    ("2NNQ", "T4B", "a mid-sized ligand in a mid-sized pocket"),
    ("1HVR", "XK2", "HIV protease + an inhibitor, flexible and larger"),
]

#: Extra angstrom of box around the bound ligand, beyond its own extent.
BOX_MARGIN = 6.0
SEED = 20260930

# The effort a search gets is not a constant here either. Exhaustiveness is a
# count of Monte Carlo walks and walks are spread through the box they search,
# so a fixed value quietly under-samples every large box and makes the box look
# like the problem when the search was. Measured on the 3PTB site box,
# 39 x 26 x 41 A: 12.88 A RMSD at 16, 1.26 A at 64, 1.26 A at 128 -- two seconds
# either way.
#
# The rule lives in `opendocking.core.exhaustiveness_for_box` because the
# workbench and `odcli` need exactly the same rule, and three copies of one
# rule is how they start disagreeing. This file used to have its own, and it
# had already drifted: its ladder started at 16 where the engine's starts at 8.
#
# It also used to print `exhaustiveness {EXHAUSTIVENESS_LADDER[0]}` in the
# header, as though one value applied to the whole run. None did: every run
# below takes its effort from its own box's volume, so that number described no
# run at all -- a correct-looking parameter in the output that was not a
# parameter. The header now states the rule, and each run and each table row
# reports the value it actually used.
EXHAUSTIVENESS_LADDER_USED = EXHAUSTIVENESS_LADDER

RCSB = "https://files.rcsb.org/download/{pid}.pdb"

#: Source files a number in this report can depend on, watched for the same
#: reason `pockets.py`'s are: a file edited under the run makes the tally
#: describe a tree that no longer exists.
WATCHED = [
    ROOT / "dock-py" / "python" / "opendocking" / "core.py",
    ROOT / "dock-py" / "python" / "opendocking" / "prep.py",
    ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "pockets.py",
    ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "structure.py",
    ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "__init__.py",
    Path(__file__).resolve(),
]


def _tree_fingerprint() -> dict[str, str]:
    """A hash of every file this run's numbers can depend on."""
    out: dict[str, str] = {}
    for path in WATCHED:
        try:
            out[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()).hexdigest()[:16]
        except OSError as exc:  # a file that vanished mid-run is itself a change
            out[str(path)] = f"unreadable: {exc.strerror}"
    return out


def changed_between(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Files that were added, removed or rewritten between two fingerprints."""
    return sorted(
        set(before) ^ set(after)
        | {k for k in set(before) & set(after) if before[k] != after[k]}
    )


def _corpus_fingerprint(parts: list[tuple[str, str]]) -> str:
    """One hash over everything that was downloaded, for comparing two runs."""
    blob = "\n".join(f"{pid}/{res} {digest}" for pid, res, digest in sorted(parts))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fetch(pid: str) -> str:
    with urllib.request.urlopen(RCSB.format(pid=pid), timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    # A truncated body parses. `read()` returns what arrived and a half-file is
    # still a file with atoms in it, so a partial download reaches the rest of
    # this script as a smaller molecule and is reported as if it were the whole
    # one. Refused here, where the bytes are still identifiable.
    if "END" not in text[-4096:]:
        raise OSError(
            f"{pid}: body has no END record near the end ({len(text)} bytes); "
            f"this is a truncated download, not a structure"
        )
    return text


def split_entry(text: str, ligand_res: str):
    """``(protein rows, ligand rows, ligand's chain)``.

    Waters and every other heteroatom are dropped. The protein is narrowed to
    the ligand's own chain so that the receptor cannot silently lose the chain
    the ligand is sitting in -- 1HVR is a homodimer, and preparing the whole
    entry keeps one chain and throws the other away, which is how the first
    version of this file came to search a different molecule than it docked.

    The ligand rows are narrowed to that chain as well. A ligand bound in more
    than one copy of a multimer -- which is what a crystallographic duplicate
    looks like -- used to be kept whole while the protein was narrowed, so the
    reference coordinates were several copies of the molecule in several places
    at once: the box was centred on their average and every RMSD was measured
    against a pose that exists nowhere. That is the first version's bug again,
    one order of magnitude smaller, and it is fixed by narrowing both sides to
    the same chain. The chain is chosen as the one holding the most ligand
    atoms, ties broken by name, so the choice does not depend on file order.
    """
    prot, lig = [], []
    for line in text.splitlines():
        if line.startswith("ATOM"):
            prot.append(line)
        elif line.startswith("HETATM") and line[17:20].strip() == ligand_res:
            lig.append(line)
    if not lig:
        return prot, [], None
    by_chain: dict[str, int] = {}
    for l in lig:
        by_chain[l[21]] = by_chain.get(l[21], 0) + 1
    chain = sorted(by_chain.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    lig = [l for l in lig if l[21] == chain]
    prot = [l for l in prot if l[21] == chain]
    return prot, lig, chain


def rmsd(a, b) -> float:
    """Coordinate-frame RMSD. No superposition, deliberately.

    The reference and the pose are already in the same frame -- the reference is
    read out of the same coordinates the search ran in -- so the question is
    "were these atoms put back where they started", not "is this the same
    molecule in some other frame". Superposing first would hide a search that
    returns the right shape in the wrong place, which is the thing a redocking
    benchmark exists to notice.

    It does mean the number is *not* the RMSD of the docking literature, which
    is almost always after superposition. The header says so, because "1.26 A"
    read next to a published "1.26 A" is a comparison this number does not
    support.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def rank_of(held) -> int | None:
    """The 1-based rank of the site holding the most of the bound pose.

    ``None`` when no site holds any of it. The old code took
    ``argmax`` unconditionally, and ``argmax`` of an all-zero vector is 0 -- so
    a search that found the ligand nowhere printed rank 1, which the legend
    defines as "the top-ranked site is the one the ligand came from". The dash
    the legend promises for "not found at all" was unreachable: rank was either
    ``None`` for no sites at all or ``best_i + 1``, never a dash.
    """
    held = list(held)
    if not held:
        return None
    best = max(held)
    if best <= 0:
        return None
    return held.index(best) + 1


def oracle_needed(held, n_ref: int):
    """``(dock a second time, site index or None, why the third column repeats)``.

    The third column is meant to be a *different* box from the pocket's
    top-ranked one, so that comparing the two separates a bad ranking from a
    bad box. It cannot be different in two cases, and both used to reach the
    table looking like two independent results:

    * **the top-ranked site already holds the whole bound pose** -- there is no
      second box to pick, so the numbers were copied from the pocket column and
      printed again. No second run happened; the column was a duplicate of the
      numbers next to it.
    * **the top-ranked site holds only part of the pose** -- then the best site
      *is* the top-ranked site, so the "oracle" run repeats the pocket run on
      the same box. With a fixed seed the same box returns the same numbers, so
      this column equals its neighbour too, and it also cost a full docking run
      to say so.

    The second one is easy to miss: the row looks like an ordinary third
    measurement, and the equality between the two columns reads as two boxes
    happening to agree. ``held`` is per-site ligand-atom counts, so both cases
    are decided from what was already computed.
    """
    held = list(held)
    if not held or max(held) <= 0:
        return False, None, None
    best_i = held.index(max(held))
    if best_i != 0:
        return True, best_i, None
    if max(held) >= n_ref:
        return False, best_i, (
            "no second box to pick: the top-ranked site already holds the whole "
            "bound pose, so no run was done and the column repeats its neighbour"
        )
    return False, best_i, (
        f"the best site is still the top-ranked one ({max(held)} of {n_ref} "
        f"ligand atoms), so the only box left to try is the one already tried; "
        f"a fixed seed makes it reproduce the column beside it"
    )


def best_of(result, ref_coords, n: int = 5):
    """``(top_pose_rmsd, best_pose_rmsd, n)`` over the reported poses.

    Both numbers, because they answer different questions. The top pose is what
    a user gets if they take the first line of the file; the best pose is
    whether the mode was in the output at all. A benchmark that reports only
    the second flatters the engine, and one that reports only the first hides
    cases where the right answer was found and then ranked below something
    else.
    """
    n = min(n, result.num_poses)
    if n == 0:
        return None, None, 0
    got = [rmsd(result.pose_coords(i), ref_coords) for i in range(n)]
    return got[0], min(got), n


def dock_into(receptor, ligand, centre, size, ref_coords):
    """Dock once and summarise. Returns ``(top, best, n, energy, ex, secs)``.

    The effort comes from the box's volume. Reporting it matters: a 1.2 A
    answer found in a 40,000 A^3 box at exhaustiveness 64 and a 1.2 A answer
    found in a 3,000 A^3 box at 16 are not the same claim, and a table that
    printed both as "1.2 A" would be hiding the difference.
    """
    started = time.perf_counter()
    ex = exhaustiveness_for_box(size)
    maps = receptor.precalculate(
        GridBox.from_center_size(tuple(centre), tuple(size)), "vina", 0.375
    )
    result = core.dock(
        ligand, maps, exhaustiveness=ex, num_modes=5, seed=SEED
    )
    top, best, n = best_of(result, ref_coords)
    return top, best, n, result.best_energy, ex, time.perf_counter() - started


def main() -> int:
    before = _tree_fingerprint()
    print("redocking benchmark - docking a ligand that is already bound")
    print(f"seed {SEED}, box margin {BOX_MARGIN} A, num_modes 5")
    print(f"exhaustiveness is NOT one value: it is chosen per box by "
          f"exhaustiveness_for_box from the ladder "
          f"{EXHAUSTIVENESS_LADDER_USED}, so a wider box gets a larger search. "
          f"Each run and each table row prints what it used.")
    print("RMSD in angstrom, to the pose in the crystal structure, in that same")
    print("frame and with NO superposition. It is not the superimposed RMSD the")
    print("docking literature usually quotes, and the two are not comparable.")
    print("The pocket search and the engine are given the same receptor file, and")
    print("each case prints both atom counts so the claim can be checked.")
    print(f"tree: {len(before)} source files fingerprinted before the run\n")

    rows = []
    corpus: list[tuple[str, str, str]] = []
    for pid, res, why in CASES:
        try:
            text = fetch(pid)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"{pid}: could not fetch ({exc}) - skipped, not counted")
            continue
        corpus.append((pid, res, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]))
        prot, lig, chain = split_entry(text, res)
        if not prot or not lig:
            print(f"{pid}/{res}: receptor {len(prot)}, ligand {len(lig)} "
                  f"- skipped")
            continue
        print(f"{pid}/{res}  chain {chain}, {len(prot)} protein atoms, "
              f"{len(lig)} ligand atoms")
        print(f"      why it is in the set: {why}")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rec_pdb = td / "rec.pdb"
            lig_pdbqt = td / "lig.pdbqt"
            rec_pdb.write_text("\n".join(prot) + "\nEND\n", encoding="utf-8")
            lig_pdbqt.write_text("\n".join(lig) + "\nEND\n", encoding="utf-8")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # `prepare_receptor` returns PDBQT text. The split above
                # already removed every HETATM line, so `keep_heterogens` has
                # nothing to keep -- worth saying, because leaving the bound
                # ligand in the receptor would dock it against itself and the
                # numbers would be meaningless in a way that looks like a good
                # result.
                rec_pdbqt = td / "rec_prep.pdbqt"
                rec_pdbqt.write_text(
                    prep.prepare_receptor(rec_pdb, keep_heterogens=False),
                    encoding="utf-8",
                )
            prepared = [
                l for l in rec_pdbqt.read_text(encoding="utf-8").splitlines()
                if l.startswith(("ATOM", "HETATM"))
            ]
            if len(prepared) < len(prot):
                # Say so. A receptor that lost atoms has lost a chain or a
                # fragment, and every number below is then about something
                # other than what the line above claims it is about.
                print(f"      WARNING: preparation went from {len(prot)} to "
                      f"{len(prepared)} atoms -- a fragment was dropped, so "
                      f"the search and the engine are both on the smaller one")

            # One file, read twice. The engine docks against this, and so does
            # the pocket search.
            receptor = Receptor.from_pdbqt(rec_pdbqt)
            view = MoleculeView.from_pdbqt(
                rec_pdbqt, pid, (0.62, 0.66, 0.72), 0.30, role="receptor"
            )
            ligand = core.load_ligand(lig_pdbqt)
            ref = np.asarray(ligand.reference_coords, np.float32)
            floor = 2.0 * ligand.radius + 1.0
            # "The same receptor" is a claim, and a claim is only worth making
            # with the number behind it. The two readers are separate parsers,
            # so they are counted separately: one number cannot back a claim
            # about two objects.
            n_view, n_engine = len(view.coords), receptor.num_atoms
            same = "the same count" if n_view == n_engine else "DIFFERENT COUNTS"
            print(f"      receptor file for both: {n_view} atoms in the view, "
                  f"{n_engine} in the engine -- {same}")
            if n_view != n_engine:
                print("      WARNING: the pocket search and the engine are not "
                      "looking at the same number of atoms, so the two boxes "
                      "below are not the same experiment")
            unknown = receptor.unknown_atom_types
            if unknown:
                print(f"      WARNING: {unknown} atom(s) carry a PDBQT type "
                      f"this engine does not know. Each keeps its shape term "
                      f"and silently loses its hydrogen-bond and hydrophobic "
                      f"character, so every RMSD below is measured on a "
                      f"degraded receptor")

            # --- the box the ligand itself implies ------------------------
            centre = ref.mean(axis=0)
            size = np.maximum(
                ref.max(axis=0) - ref.min(axis=0) + np.float32(2.0 * BOX_MARGIN),
                np.float32(floor),
            )
            tr, br, n_ref, e_ref, x_ref, s_ref = dock_into(
                receptor, ligand, centre, size, ref
            )
            print(f"      bound-ligand box: top {tr:5.2f} A, best {br:5.2f} A "
                  f"of {n_ref}, {e_ref:6.2f} kcal/mol, exhaustiveness {x_ref}, "
                  f"{s_ref:.1f} s")

            # --- the box the pocket search picks --------------------------
            t0 = time.perf_counter()
            sites = P.find_pockets(
                view.coords, view.elements, residues=view.residue_labels()
            )
            t_search = time.perf_counter() - t0
            ta = ba = ea = sa = None
            rank = None
            oa = ob = None
            xo = None
            n_held = 0
            repeat = None
            if not sites:
                print("      pocket box: no site found - no docking run")
            else:
                # Which of them, if any, is the one the ligand came from? The
                # box has to hold the whole bound pose for the search to have
                # found the right place, and the rank says how much credit the
                # ranking deserves. `rank_of` returns None when no site holds
                # any of it, which is the dash the legend promises.
                held = [
                    int(np.all(np.abs(ref - np.asarray(s.center, np.float32))
                               <= np.asarray(s.size, np.float32) / 2.0
                                  + P.DEFAULT_PADDING, axis=1).sum())
                    for s in sites
                ]
                rank = rank_of(held)
                n_held = max(held) if held else 0
                print(f"      pocket search: {len(sites)} sites in "
                      f"{t_search:.1f} s; the best of them holds {n_held} of "
                      f"{len(ref)} ligand atoms"
                      + (f" and is rank {rank}" if rank else
                         ", so none of them holds the bound pose and the rank "
                         "is a dash"))
                c, s = sites[0].box_center_and_size()
                s = tuple(max(float(v), floor) for v in s)
                ta, ba, _, ea, xa, sa = dock_into(receptor, ligand, c, s, ref)
                near = float(np.linalg.norm(np.asarray(sites[0].center) - centre))
                print(f"      pocket box:     top {ta:5.2f} A, best {ba:5.2f} A, "
                      f"{ea:6.2f} kcal/mol, exhaustiveness {xa}, {sa:.1f} s")
                print(f"                      site 1 is {near:.1f} A from the bound "
                      f"ligand, box {s[0]:.0f}x{s[1]:.0f}x{s[2]:.0f} A")

                # The oracle box: the site that *does* hold the bound pose,
                # judged by a human reading the table rather than by the
                # ranking. It costs one more docking run and it splits the
                # question in two, because a failure here cannot be blamed on
                # the ranking and a failure in the column above cannot be
                # blamed on the search. `oracle_needed` also reports the two
                # ways that split is impossible, so a repeated column is
                # labelled rather than left to be noticed.
                need_run, site_i, repeat = oracle_needed(held, len(ref))
                if rank is None:
                    print("      pose site's box: n/a - no site holds the bound "
                          "pose, so there is no box to blame the ranking for")
                elif need_run:
                    oc, os_ = sites[site_i].box_center_and_size()
                    os_ = tuple(max(float(v), floor) for v in os_)
                    oa, ob, _, _, xo, _ = dock_into(
                        receptor, ligand, oc, os_, ref
                    )
                    print(f"      pose site's box: top {oa:5.2f} A, best "
                          f"{ob:5.2f} A, box {os_[0]:.0f}x{os_[1]:.0f}x"
                          f"{os_[2]:.0f} A at exhaustiveness {xo}  "
                          f"(ranking set aside)")
                else:
                    # Not a third measurement. Say so, instead of printing the
                    # pocket column's numbers again and leaving a reader to
                    # work out whether two boxes agreed.
                    oa, ob = ta, ba
                    print("      pose site's box: not a separate run - " + repeat)
            flags = []
            if repeat:
                flags.append("3rd column repeats the 2nd")
            if e_ref is not None and e_ref >= 0.0:
                # A bound ligand scores around -4 to -9 here. A non-negative
                # best energy means the engine returned no favourable pose, so
                # the RMSD beside it is what a failed run produced -- it is in
                # the table as a bare number otherwise, and reads like a
                # measurement.
                flags.append(f"bound-box energy {e_ref:.0f} is not favourable")
            if len(prepared) < len(prot):
                flags.append(f"receptor lost {len(prot) - len(prepared)} atoms")
            # Appended outside the branch on purpose: a case whose pocket search
            # returned nothing is still a case, and dropping it here would take
            # it out of the denominator of the "under 2 A" tally below, which is
            # the one place a search failure has to count against the search.
            rows.append((pid, res, len(lig), tr, br, ta, ba, oa, ob, rank,
                         x_ref, xa, n_held, len(ref), repeat, flags))

    if not rows:
        print("\nnothing could be fetched; no results to report")
        print(f"tree: {'CHANGED' if changed_between(before, _tree_fingerprint()) else 'unchanged'}"
              f" across the run")
        return 1

    print("\n" + "=" * 100)
    print(f"{'case':11s} {'atoms':>5s} | {'bound box':>15s} | {'pocket box':>15s} "
          f"| {'pose site':>15s} | rank | holds   | ex here/top | !")
    print(f"{'':11s} {'':5s} | {'top':>6s} {'best':>8s} | {'top':>6s} {'best':>8s} "
          f"| {'top':>6s} {'best':>8s} | of pose |           |   ")
    print("-" * 100)
    for (pid, res, n, tr, br, ta, ba, oa, ob, rank, x_ref, xa, n_held, n_ref,
         repeat, flags) in rows:
        auto = (f"{ta:6.2f} {ba:8.2f}" if ta is not None
                else f"{'n/a':>6s} {'n/a':>8s}")
        orc = (f"{oa:6.2f} {ob:8.2f}" if oa is not None
               else f"{'n/a':>6s} {'n/a':>8s}")
        # A repeated cell is marked rather than left to be noticed. Without the
        # mark, the pose site's own box agreeing exactly with the pocket column
        # is indistinguishable from a second run that happened to land on the
        # same number, and the legend's "one number apart" is simply false in
        # the cases where there was nothing to set apart.
        if repeat:
            orc += "*"
        print(f"{pid}/{res:5s} {n:5d} | {tr:6.2f} {br:8.2f} | {auto} | {orc} "
              f"| {rank if rank else '-':>7} | {n_held:3d}/{n_ref:<3d} | "
              f"{x_ref}/{xa} | {len(flags)}")
    flagged = [(r[0] + "/" + r[1], r[15]) for r in rows if r[15]]
    if flagged:
        print("!  rows carrying caveats, which the numbers above do not show:")
        for case, fl in flagged:
            for item in fl:
                print(f"     {case}: {item}")
    repeats = [r for r in rows if r[14]]
    if repeats:
        print("*  the pose site's box is the pocket box -- that column is not a "
              "separate measurement:")
        for r in repeats:
            print(f"     {r[0]}/{r[1]}: {r[14]}")

    good_ref = [r for r in rows if r[4] is not None and r[4] < 2.0]
    good_auto = [r for r in rows if r[6] is not None and r[6] < 2.0]
    good_or = [r for r in rows if r[8] is not None and r[8] < 2.0]
    print("-" * 100)
    print(f"best pose under 2 A: {len(good_ref)}/{len(rows)} bound-ligand box, "
          f"{len(good_auto)}/{len(rows)} pocket's top-ranked box, "
          f"{len(good_or)}/{len(rows)} the pose site's own box")
    print()
    print("Read the three columns as different claims. The bound-ligand box is")
    print("centred on the answer, so it measures the search and the scoring and")
    print("nothing else. The pocket's top-ranked box is chosen without ever")
    print("seeing the ligand, so it measures the whole chain. The third column")
    print("is the site a person would have picked after reading the table -- the")
    print("one that actually holds the bound pose -- so comparing it with the")
    print("second column separates a bad *ranking* from a bad *box*: the same")
    print("search, the same site, one number apart.")
    print()
    print("...except where it is starred. When the ranking's top site already")
    print("holds the whole bound pose there is no second box to pick, no second")
    print("run, and nothing to compare: the third column is a copy of the")
    print("second. A starred row says the ranking was right, and 'one number")
    print("apart' does not apply to it.")
    print()
    print("The rank column is the search's opinion, judged against the crystal")
    print("structure: 1 means the top-ranked site is the one the ligand came")
    print("from. A high rank with a good RMSD would mean the site was found and")
    print("the ranking did not agree; a dash would mean it was not found at all.")
    print("Read it with the holds column beside it, which says how much of the")
    print("bound pose that site actually covers. The dash is printed when no site")
    print("holds a single ligand atom; it used to be unreachable, because the")
    print("rank was argmax over a vector that was all zeroes exactly when the")
    print("search had failed, and argmax of zeroes is 0, which reads as rank 1.")
    print()
    print("Caveat that limits all of it: the bound ligand's bonds are read from")
    print("interatomic distances, because a PDB ligand carries none. That gets")
    print("connectivity right and aromaticity and formal charge wrong, which")
    print("changes the rotatable-bond set and the partial charges. These are")
    print("numbers for a specific, imperfect input, not for the engine alone.")
    print()
    print(f"corpus fingerprint {_corpus_fingerprint(corpus)} over "
          f"{len(corpus)} downloaded file(s). The bytes are not pinned, so a "
          f"re-run that disagrees with the published numbers should be checked "
          f"against this value first.")
    moved = changed_between(before, _tree_fingerprint())
    print(f"tree: {'CHANGED -- ' + ', '.join(moved) if moved else 'unchanged'} "
          f"across the run")
    if moved:
        print("  A source file these numbers depend on was edited while this "
              "was running, so the figures above describe code that no longer "
              "exists. Re-run before reading them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
