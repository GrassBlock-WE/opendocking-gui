"""How the pocket search behaves across a range of real proteins.

Not a pass/fail check. There is no ground truth here that a pass/fail could
be measured against, and writing one would be dishonest: whether a protein has
a *sealed* cavity is not something this tool decides, and whether a particular
site is a *binding site* for a particular ligand it cannot know.

So this fetches a spread of structures, runs the search, and prints what came
out — including the parts that are unflattering. The numbers in
`docs/VERIFICATION.md` come from this script, and re-running it is how those
numbers are checked. It needs a network, so it is deliberately **not** a CI
gate: `scripts/pockets_check.py` is the gate, and it needs none.

Why the set is what it is:

* **1CRN** crambin, already the fixture everywhere else, and known to have no
  internal cavity at a 1.4 A probe.
* **1L96** T4 lysozyme L99A: a bulky residue replaced by alanine, leaving a
  buried cavity. It is here to be able to *falsify* the detector. If a
  purpose-built cavity is not reported as sealed, that is worth knowing and is
  printed.
* **3PTB** trypsin: a deep, well-defined binding pocket, the kind a site
  finder ought to find.
* **1STP** streptavidin: a binding *groove*, and a crystal structure with
  plenty of ordered waters in it.
* **1MBN** myoglobin: the haem pocket, where a cofactor is present in the file
  and must be handled rather than silently treated as protein.
* **4HHB** haemoglobin, 4779 atoms in four chains: the largest here, and the
  one that sets the worst case for how long a search can take.
* **1TIM** a metalloproteinase with a large internal cavity and four chains.

Run:  python scripts/pocket_benchmark.py
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Prefer the installed package; fall back to the source tree. Same reason as
# every other script here: a clean checkout has no compiled `_dockpy` in its
# source tree.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking.workbench import MoleculeView  # noqa: E402
from opendocking.workbench import pockets as P  # noqa: E402

#: ``(PDB id, one line on why it is in the set)``
STRUCTURES = [
    ("1CRN", "crambin — the fixture used everywhere else here"),
    ("1L96", "T4 lysozyme L99A — a deliberately engineered buried cavity"),
    ("3PTB", "trypsin — a deep, well-defined binding pocket"),
    ("1STP", "streptavidin — a binding groove, and full of ordered waters"),
    ("1MBN", "myoglobin — a cofactor in the file, not part of the protein"),
    ("4HHB", "haemoglobin — 4779 atoms in four chains, the largest here"),
    ("1TIM", "a metalloproteinase with a large internal cavity"),
]

RCSB = "https://files.rcsb.org/download/{pid}.pdb"
PROBES = (0.8, 1.4, 2.0)


def fetch(pid: str) -> str:
    with urllib.request.urlopen(RCSB.format(pid=pid), timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def main() -> int:
    print("pocket search across real structures")
    print(f"probe default {P.DEFAULT_PROBE} A, spacing {P.DEFAULT_SPACING} A, "
          f"burial >= 1, volume ceiling {P.DEFAULT_MAX_VOLUME:.0f} A^3\n")

    rows = []
    for pid, why in STRUCTURES:
        try:
            text = fetch(pid)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"{pid}: could not fetch ({exc}) — skipped, not counted")
            continue

        view = MoleculeView.from_text(text, pid, (0.62, 0.66, 0.72), 0.30, role="receptor")
        if len(view.coords) == 0:
            print(f"{pid}: no atoms parsed — skipped, not counted")
            continue
        span = np.asarray(view.coords).max(axis=0) - np.asarray(view.coords).min(axis=0)

        t0 = time.perf_counter()
        sites = P.find_pockets(
            view.coords, view.elements, residues=view.residue_labels()
        )
        ms = (time.perf_counter() - t0) * 1000.0
        sealed = [p for p in sites if p.kind == "cavity"]

        print(f"{pid}  {len(view.coords):5d} atoms  "
              f"span {span[0]:.0f}x{span[1]:.0f}x{span[2]:.0f} A  "
              f"{ms:6.0f} ms  {len(sites)} sites ({len(sealed)} sealed)")
        print(f"      {why}")
        for i, p in enumerate(sites[:4]):
            lining = ", ".join(r for r, _ in p.lining[:4]) or "-"
            print(f"      #{i + 1} {P.kind_label(p.kind):6s} "
                  f"score {p.rank_score:5.2f}  "
                  f"{p.size[0]:4.1f} x {p.size[1]:4.1f} x {p.size[2]:4.1f} A  "
                  f"{p.voxels:4d} pts  buried {p.burial:.2f}/3  [{lining}]")
        probe_line = []
        for probe in PROBES:
            got = P.find_pockets(view.coords, view.elements, probe=probe)
            n_sealed = sum(1 for p in got if p.kind == "cavity")
            probe_line.append(f"{probe}: {len(got)} ({n_sealed} sealed)")
        print(f"      probe sweep — " + "; ".join(probe_line))
        rows.append((pid, len(view.coords), ms, len(sites), len(sealed)))

    if not rows:
        print("\nnothing could be fetched; no results to report")
        return 1

    print("\n" + "=" * 78)
    print(f"{'PDB':6s} {'atoms':>7s} {'ms':>8s} {'sites':>6s} {'sealed':>7s}")
    for pid, n, ms, ns, nsealed in rows:
        print(f"{pid:6s} {n:7d} {ms:8.0f} {ns:6d} {nsealed:7d}")

    slowest = max(rows, key=lambda r: r[2])
    biggest = max(rows, key=lambda r: r[1])
    per_atom = [(r[2] / max(1, r[1]), r[0]) for r in rows]
    per_atom.sort(reverse=True)
    print(f"\nslowest: {slowest[0]} at {slowest[2]:.0f} ms "
          f"({slowest[1]} atoms)")
    print(f"largest: {biggest[0]} at {biggest[1]} atoms, {biggest[2]:.0f} ms")
    print("cost per 1000 atoms, worst first: "
          + ", ".join(f"{pid} {t:.0f} ms" for t, pid in per_atom))
    sealed_total = sum(r[4] for r in rows)
    print(f"\nsealed cavities found across {len(rows)} structures: {sealed_total}")
    print("A sealed cavity is a claim about geometry, not about binding: a")
    print("protein with none may still have an obvious groove, and 1L96 — the")
    print("one here with a purpose-built cavity — is reported in the table above")
    print("either way. Read the rows, not the total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
