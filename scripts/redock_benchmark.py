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
docking search.

Output is deliberately ASCII only. A source file with non-ASCII literals is
one PowerShell 5.1 `Set-Content -Encoding UTF8` away from being silently
mangled, and a benchmark that crashes on an encoding artefact measures
nothing.

Run:  python scripts/redock_benchmark.py
"""

from __future__ import annotations

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
EXHAUSTIVENESS = EXHAUSTIVENESS_LADDER[0]

RCSB = "https://files.rcsb.org/download/{pid}.pdb"


def fetch(pid: str) -> str:
    with urllib.request.urlopen(RCSB.format(pid=pid), timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def split_entry(text: str, ligand_res: str):
    """``(protein rows, ligand rows, ligand's chain)``.

    Waters and every other heteroatom are dropped. The protein is narrowed to
    the ligand's own chain so that the receptor cannot silently lose the chain
    the ligand is sitting in -- 1HVR is a homodimer, and preparing the whole
    entry keeps one chain and throws the other away, which is how the first
    version of this file came to search a different molecule than it docked.
    """
    prot, lig = [], []
    chain = None
    for line in text.splitlines():
        if line.startswith("ATOM"):
            prot.append(line)
        elif line.startswith("HETATM") and line[17:20].strip() == ligand_res:
            lig.append(line)
            chain = line[21]
    if chain is not None:
        prot = [l for l in prot if l[21] == chain]
    return prot, lig, chain


def rmsd(a, b) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


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
    print("redocking benchmark - docking a ligand that is already bound")
    print(f"exhaustiveness {EXHAUSTIVENESS}, seed {SEED}, "
          f"box margin {BOX_MARGIN} A, num_modes 5")
    print("RMSD in angstrom, to the pose in the crystal structure.")
    print("The pocket search and the engine are given the same receptor.\n")

    rows = []
    for pid, res, why in CASES:
        try:
            text = fetch(pid)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"{pid}: could not fetch ({exc}) - skipped, not counted")
            continue
        prot, lig, chain = split_entry(text, res)
        if not prot or not lig:
            print(f"{pid}/{res}: receptor {len(prot)}, ligand {len(lig)} "
                  f"- skipped")
            continue
        print(f"{pid}/{res}  chain {chain}, {len(prot)} protein atoms, "
              f"{len(lig)} ligand atoms")
        print(f"      {why}")

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
            print(f"      receptor for both: {len(view.coords)} atoms")

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
            if not sites:
                print("      pocket box: no site found - no docking run")
            else:
                # Which of them, if any, is the one the ligand came from? The
                # box has to hold the whole bound pose for the search to have
                # found the right place, and the rank says how much credit the
                # ranking deserves.
                held = [
                    int(np.all(np.abs(ref - np.asarray(s.center, np.float32))
                               <= np.asarray(s.size, np.float32) / 2.0
                                  + P.DEFAULT_PADDING, axis=1).sum())
                    for s in sites
                ]
                best_i = int(np.argmax(held))
                rank = best_i + 1
                print(f"      pocket search: {len(sites)} sites in "
                      f"{t_search:.1f} s; the one holding the bound pose is "
                      f"rank {rank}, holding {held[best_i]}/{len(ref)} atoms")
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
                # blamed on the search.
                oa = ob = None
                if best_i != 0 or held[best_i] < len(ref):
                    oc, os_ = sites[best_i].box_center_and_size()
                    os_ = tuple(max(float(v), floor) for v in os_)
                    oa, ob, _, _, xo, _ = dock_into(
                        receptor, ligand, oc, os_, ref
                    )
                    print(f"      pose site's box: top {oa:5.2f} A, best "
                          f"{ob:5.2f} A, box {os_[0]:.0f}x{os_[1]:.0f}x"
                          f"{os_[2]:.0f} A at exhaustiveness {xo}  "
                          f"(ranking set aside)")
                else:
                    oa, ob = ta, ba
            rows.append((pid, res, len(lig), tr, br, ta, ba, oa, ob, rank, x_ref, xa))

    if not rows:
        print("\nnothing could be fetched; no results to report")
        return 1

    print("\n" + "=" * 100)
    print(f"{'case':11s} {'atoms':>5s} | {'bound box':>15s} | {'pocket box':>15s} "
          f"| {'pose site':>15s} | rank")
    print(f"{'':11s} {'':5s} | {'top':>6s} {'best':>8s} | {'top':>6s} {'best':>8s} "
          f"| {'top':>6s} {'best':>8s} | of pose | ex here/top")
    print("-" * 100)
    for pid, res, n, tr, br, ta, ba, oa, ob, rank, x_ref, xa in rows:
        auto = (f"{ta:6.2f} {ba:8.2f}" if ta is not None
                else f"{'n/a':>6s} {'n/a':>8s}")
        orc = (f"{oa:6.2f} {ob:8.2f}" if oa is not None
               else f"{'n/a':>6s} {'n/a':>8s}")
        print(f"{pid}/{res:5s} {n:5d} | {tr:6.2f} {br:8.2f} | {auto} | {orc} "
              f"| {rank if rank else 'n/a':>7} | {x_ref}/{xa}")

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
    print("The rank column is the search's opinion, judged against the crystal")
    print("structure: 1 means the top-ranked site is the one the ligand came")
    print("from. A high rank with a good RMSD would mean the site was found and")
    print("the ranking did not agree; a dash would mean it was not found at all.")
    print()
    print("Caveat that limits all of it: the bound ligand's bonds are read from")
    print("interatomic distances, because a PDB ligand carries none. That gets")
    print("connectivity right and aromaticity and formal charge wrong, which")
    print("changes the rotatable-bond set and the partial charges. These are")
    print("numbers for a specific, imperfect input, not for the engine alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
