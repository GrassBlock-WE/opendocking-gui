"""How the pocket search behaves across a range of real proteins.

Not a pass/fail check. There is no ground truth here that a pass/fail could
be measured against, and writing one would be dishonest: whether a protein has
a *sealed* cavity is not something this tool decides, and whether a particular
site is a *binding site* for a particular ligand it cannot know.

So this fetches a spread of structures, runs the search, and prints what came
out -- including the parts that are unflattering. The numbers in
`docs/VERIFICATION.md` come from this script, and re-running it is how those
numbers are checked. It needs a network, so it is deliberately **not** a CI
gate: `scripts/pockets_check.py` is the gate, and it needs none.

Every structure is fetched fresh, so the corpus is the internet and can change
between two runs that both report success. Two fingerprints are printed to make
that visible rather than silent:

* a **corpus fingerprint**, one hash over every downloaded file, so a re-run
  can be compared against the run that produced the published numbers. The
  bytes are not cached and not pinned, so this is the only thing standing
  between a re-deposited structure and a confidently wrong comparison.
* a **tree fingerprint** over the source files these numbers depend on, taken
  before the first fetch and again after the last, because a file edited while
  the benchmark is running makes the tally describe a tree that no longer
  exists. Four files have been rewritten under a run in this project already.

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

import hashlib
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

#: The source files a number in this report can depend on. Watched, because a
#: tree that moves under the run makes every figure below describe code that no
#: longer exists -- and the corpus is the internet, so the one thing that is
#: *not* watched here is exactly the thing that moves.
WATCHED = [
    ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "pockets.py",
    ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "structure.py",
    ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "__init__.py",
    ROOT / "scripts" / "pockets_check.py",
    Path(__file__).resolve(),
]

#: ``(PDB id, why it is in the set, falsifiable claim or None)``.
#:
#: The third field is the one number the *docstring* asserts about that
#: structure, and it is checked against the measurement rather than printed
#: beside it. Two entries have one; the rest do not, and the report says so
#: rather than inventing a threshold. This is the difference between a comment
#: that states an expectation and a comment that states a result: the first
#: above was printed directly under the measurement with nothing connecting
#: the two, so a structure that stopped behaving the way its own caption said
#: was reported in the same shape as one that did not.
STRUCTURES = [
    ("1CRN", "crambin - the fixture used everywhere else here",
     "no sealed cavity at a 1.4 A probe", lambda sealed: sealed == 0),
    ("1L96", "T4 lysozyme L99A - a deliberately engineered buried cavity",
     "a purpose-built cavity is reported as sealed", lambda sealed: sealed >= 1),
    ("3PTB", "trypsin - a deep, well-defined binding pocket", None, None),
    ("1STP", "streptavidin - a binding groove, and full of ordered waters", None, None),
    ("1MBN", "myoglobin - a cofactor in the file, not part of the protein", None, None),
    ("4HHB", "haemoglobin - the largest here, and the worst case for timing", None, None),
    ("1TIM", "a metalloproteinase with a large internal cavity", None, None),
]

RCSB = "https://files.rcsb.org/download/{pid}.pdb"
PROBES = (0.8, 1.4, 2.0)


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
    blob = "\n".join(f"{pid} {digest}" for pid, digest in sorted(parts))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fetch(pid: str) -> str:
    with urllib.request.urlopen(RCSB.format(pid=pid), timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    # A truncated body parses. `read()` returns what arrived, and a half-file is
    # still a file with atoms in it, so the only protection is to say the size
    # and let a reader compare it against what the entry should weigh. `replace`
    # in the decode above is the same hazard in a quieter form: a mangled byte
    # becomes U+FFFD and the structure silently loses atoms.
    if "END" not in text[-4096:] and "END\n" not in text:
        raise OSError(
            f"{pid}: body has no END record near the end ({len(text)} bytes); "
            f"this is a truncated download, not a structure"
        )
    return text


def cost_line(rows) -> str:
    """The "search time per 1000 atoms" line, label and numbers built together.

    One function, because the defect this replaces was a label and an
    arithmetic that disagreed: `r[2] / r[1]` is ms *per atom*, printed under a
    heading that said per 1000, so a 17.6 s search over 4779 atoms read
    "4HHB 4 ms" and 1CRN rounded to "0 ms". Every figure was a thousandth of
    what it claimed. Building the label here rather than at the call site means
    the two cannot drift apart again, and both halves are checked:
    `scripts/benchmark_check.py` asserts the heading says per 1000 *and* that the
    numbers are ms x 1000 / atoms, so reverting either one is caught.
    """
    vals = sorted(
        ((r[2] * 1000.0 / max(1, r[1]), r[0]) for r in rows), reverse=True
    )
    return ("search time per 1000 atoms, worst first: "
            + ", ".join(f"{pid} {t:.0f} ms" for t, pid in vals))


def main() -> int:
    before = _tree_fingerprint()
    print("pocket search across real structures")
    print(f"probe default {P.DEFAULT_PROBE} A, spacing {P.DEFAULT_SPACING} A, "
          f"burial >= 1, volume ceiling {P.DEFAULT_MAX_VOLUME:.0f} A^3")
    print(f"tree: {len(before)} source files fingerprinted before the run\n")

    rows = []
    corpus: list[tuple[str, str]] = []
    skipped: list[str] = []
    claims: list[tuple[str, bool]] = []
    for pid, why, claim, holds in STRUCTURES:
        try:
            text = fetch(pid)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"{pid}: could not fetch ({exc}) - skipped, not counted")
            skipped.append(pid)
            continue
        corpus.append((pid, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]))

        view = MoleculeView.from_text(text, pid, (0.62, 0.66, 0.72), 0.30, role="receptor")
        if len(view.coords) == 0:
            print(f"{pid}: no atoms parsed - skipped, not counted")
            skipped.append(pid)
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
        print(f"      why it is in the set: {why}")
        if claim is not None:
            ok = bool(holds(len(sealed)))
            claims.append((pid, ok))
            print(f"      claimed here: {claim} -- "
                  + ("HELD" if ok else "NOT HELD") + f" ({len(sealed)} sealed)")
        else:
            print("      claimed here: nothing falsifiable; the reason above is "
                  "why the structure is in the set, not a prediction this run "
                  "can test")
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
        print("      probe sweep - " + "; ".join(probe_line))
        rows.append((pid, len(view.coords), ms, len(sites), len(sealed)))

    if not rows:
        print("\nnothing could be fetched; no results to report")
        print(f"tree: {'unchanged' if not changed_between(before, _tree_fingerprint()) else 'CHANGED'}"
              f" across the run")
        return 1

    print("\n" + "=" * 78)
    print(f"{'PDB':6s} {'atoms':>7s} {'ms':>8s} {'sites':>6s} {'sealed':>7s}")
    for pid, n, ms, ns, nsealed in rows:
        print(f"{pid:6s} {n:7d} {ms:8.0f} {ns:6d} {nsealed:7d}")

    slowest = max(rows, key=lambda r: r[2])
    biggest = max(rows, key=lambda r: r[1])
    print(f"\nslowest: {slowest[0]} at {slowest[2]:.0f} ms "
          f"({slowest[1]} atoms)")
    print(f"largest: {biggest[0]} at {biggest[1]} atoms, {biggest[2]:.0f} ms")
    print(cost_line(rows))
    sealed_total = sum(r[4] for r in rows)
    print(f"\nsealed cavities found across {len(rows)} structures: {sealed_total}")
    print("A sealed cavity is a claim about geometry, not about binding: a")
    print("protein with none may still have an obvious groove, and 1L96 - the")
    print("one here with a purpose-built cavity - is reported in the table above")
    print("either way. Read the rows, not the total.")
    if skipped:
        print(f"\n{len(skipped)} of {len(STRUCTURES)} structures did not run and "
              f"are absent from every figure above: {', '.join(skipped)}. The "
              f"counts and the total describe the {len(rows)} that did, not the "
              f"set of {len(STRUCTURES)} the set was chosen to be.")
    if claims:
        print()
        for pid, ok in claims:
            print(f"{pid}: the claim made about it in this file was "
                  + ("HELD" if ok else "NOT HELD") + ".")
    print(f"\ncorpus fingerprint {_corpus_fingerprint(corpus)} over "
          f"{len(corpus)} downloaded file(s). The bytes are not pinned, so a "
          f"re-run that disagrees with the published numbers should be checked "
          f"against this value first: a different fingerprint is a different "
          f"corpus, not a regression.")
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
