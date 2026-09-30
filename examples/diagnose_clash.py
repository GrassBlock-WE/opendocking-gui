"""Is the clash a scoring-function preference or a search failure?

Docking into a real protein returned poses with every ligand atom inside the
receptor's van der Waals volume, at minimum separations down to 0.22 A. Two
very different problems produce that symptom:

* the *scoring function* genuinely prefers buried, clashing poses, because
  the attractive terms outweigh the soft repulsion; or
* clash-free poses score just as well, but the *search* never finds them.

They need different fixes, and the symptom alone does not distinguish them. So
sample the landscape directly: draw many random placements, score each, and
compare the best clash-free placement with the best placement overall.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import opendocking  # noqa: E402

CLASH = 2.0  # A; two heavy atoms closer than this cannot both be where they are


def read_heavy(path: str) -> np.ndarray:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            t = line[77:79].strip()
            if t in ("HD", "H"):
                continue
            rows.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(rows)


def main(receptor_path: str, ligand_path: str, centre, size: float, n: int) -> int:
    centre = np.asarray(centre, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = opendocking.Receptor.from_pdbqt(receptor_path)
    lig = opendocking.Ligand.from_pdbqt(ligand_path)
    box = opendocking.GridBox.from_center_size(tuple(centre), (size, size, size))
    maps = rec.precalculate(box, scoring="vina", spacing=0.375)
    rec_heavy = read_heavy(receptor_path)

    rng = np.random.default_rng(4242)
    ndof = lig.num_dof
    pop = np.empty((n, ndof))
    pop[:, :3] = centre + rng.uniform(-size / 2, size / 2, (n, 3))
    pop[:, 3:6] = rng.uniform(-np.pi, np.pi, (n, 3))
    pop[:, 6:] = rng.uniform(-np.pi, np.pi, (n, lig.num_torsions))

    energies = opendocking.evaluate_conformations(lig, maps, pop, "vina", use_gpu=True)

    # Re-derive coordinates for each sampled placement through the engine, so
    # the clash test uses exactly the geometry that was scored.
    rows = []
    for i in range(n):
        e, _ = opendocking.score_conformation(lig, maps, pop[i], "vina")
        xyz = opendocking.conformation_coordinates(lig, pop[i])
        d = np.linalg.norm(xyz[:, None, :] - rec_heavy[None, :, :], axis=-1).min()
        rows.append((e, float(d)))
    rows.sort(key=lambda r: r[0])

    ok = [(e, d) for e, d in rows if d >= CLASH]
    bad = [(e, d) for e, d in rows if d < CLASH]
    print(f"random placements scored : {n}")
    print(f"  clash-free (>= {CLASH} A) : {len(ok)}")
    print(f"  clashing                : {len(bad)}")
    if rows:
        e, d = rows[0]
        print(f"\nbest placement overall   : E = {e:8.3f}   closest receptor atom {d:5.2f} A")
    if ok:
        e, d = ok[0]
        print(f"best clash-free placement: E = {e:8.3f}   closest receptor atom {d:5.2f} A")
        gap = rows[0][0] - e
        if gap > 0.01:
            print(
                f"  the scoring function prefers a clashing pose by {gap:+.3f} kcal/mol"
                "  <-- the energy function, not the search, is the problem"
            )
        elif gap < -0.01:
            print(
                f"  the scoring function prefers the clean pose by {-gap:+.3f} kcal/mol"
                "  (the search is leaving kcal/mol on the table)"
            )
        else:
            print(
                "  the clean placement *is* the global optimum: the scoring function"
            )
            print("  does not reward burying the ligand in the receptor.")
    if bad:
        print(f"worst clashing placement : E = {bad[0][0]:8.3f}   closest {bad[0][1]:5.2f} A")
    return 0


def _coords_for(lig, conf):
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else "1crn_prep.pdbqt",
            sys.argv[2] if len(sys.argv) > 2 else "biotin_prep.pdbqt",
            [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "9.24,9.71,6.91").split(",")],
            float(sys.argv[4]) if len(sys.argv) > 4 else 26.0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 400,
        )
    )
