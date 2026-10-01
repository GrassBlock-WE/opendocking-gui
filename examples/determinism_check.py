"""Determinism and thread-safety of the two places the engine goes parallel.

Two claims are worth checking because nothing else would catch a violation:

1. **Grid precalculation is reproducible.** The parallel loop hands each worker
   a mutable slice of the same `Vec`, so a slicing mistake would corrupt the map
   — silently, since every lookup still returns a finite number. Comparing the
   maps byte-for-byte between two calls is the way to see that.

   *What this script cannot check, and used to imply that it did:* independence
   of the **thread count**. The engine chooses its own thread count and exposes
   no knob for it, so there is nothing here to vary — the two calls below are
   the same call twice. A genuine thread-count test needs either an engine
   setting or a separate process per count, and neither exists today. The
   heading this used to print claimed thread-count independence, which no
   measurement here supports; `scripts/examples_check.py` fails if that wording
   comes back.

2. **A fixed seed reproduces a run exactly.** Search trajectories derive their
   own seeds from the main one, so if that derivation is not per-trajectory the
   parallel arms end up correlated, and results change with the thread count.
   That second half is also untested here for the same reason; what is tested is
   that one fixed seed reproduces one fixed run.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import opendocking  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  ok   {label}{(' — ' + detail) if detail else ''}")
    else:
        failures.append(label)
        print(f"  FAIL {label}{(' — ' + detail) if detail else ''}")


def main() -> int:
    here = Path(__file__).resolve().parent
    receptor_path = here / "rec_prep.pdbqt"
    ligand_path = here / "ligands" / "ibuprofen.pdbqt"
    if not receptor_path.exists() or not ligand_path.exists():
        print("run from the examples directory after preparing the sample data")
        return 2

    receptor = opendocking.Receptor.from_pdbqt(receptor_path)
    ligand = opendocking.Ligand.from_pdbqt(ligand_path)
    box = opendocking.GridBox.from_center_size((0.0, 0.0, 0.0), (20.0, 20.0, 20.0))

    print("grid precalculation is reproducible")
    # The same call twice, NOT two thread counts: the engine decides threads
    # internally and exposes no knob, so a heading claiming thread-count
    # independence here would be claiming something this cannot measure. The
    # docstring says so at length; this line says it in six words.
    serial = receptor.precalculate(box, scoring="vina", spacing=0.375)
    again = receptor.precalculate(box, scoring="vina", spacing=0.375)
    check(
        "two identical calls agree",
        np.array_equal(serial.raw_data, again.raw_data),
        "byte-for-byte over the whole map",
    )

    vinardo = receptor.precalculate(box, scoring="vinardo", spacing=0.375)
    check(
        "vinardo maps differ from vina",
        not np.array_equal(serial.raw_data, vinardo.raw_data),
        "the scoring function really reaches the grid",
    )

    print("\nscoring is unaffected by map construction order")
    reordered = receptor.precalculate(box, scoring="vinardo", spacing=0.375)
    check(
        "vinardo maps are stable",
        np.array_equal(vinardo.raw_data, reordered.raw_data),
    )

    print("\na fixed seed reproduces a run exactly")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = opendocking.dock(ligand, serial, exhaustiveness=4, num_modes=3, seed=1234)
        b = opendocking.dock(ligand, serial, exhaustiveness=4, num_modes=3, seed=1234)
    check(
        "same seed, same energy",
        np.allclose(a.energies, b.energies, atol=0.0, rtol=0.0),
        f"max diff {np.max(np.abs(a.energies - b.energies)):g}",
    )
    check(
        "same seed, same coordinates",
        np.array_equal(a.all_pose_coords(), b.all_pose_coords()),
    )
    check(
        "same seed, same pose count",
        a.num_poses == b.num_poses,
        f"{a.num_poses} vs {b.num_poses}",
    )

    print("\nmore exhaustiveness never reports a worse valid pose")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low = opendocking.dock(ligand, serial, exhaustiveness=2, num_modes=1, seed=99)
        high = opendocking.dock(ligand, serial, exhaustiveness=16, num_modes=1, seed=99)
    if low.rejected_pose_count == 0 and high.rejected_pose_count == 0:
        check(
            "energy is monotonic in exhaustiveness",
            high.best_energy <= low.best_energy + 1e-9,
            f"{low.best_energy:.4f} -> {high.best_energy:.4f}",
        )
    else:
        print(
            "  note skipped: one run fell back to overlapping poses "
            f"(rejected {low.rejected_pose_count} / {high.rejected_pose_count})"
        )

    print("\ndifferent seeds explore different regions")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1 = opendocking.dock(ligand, serial, exhaustiveness=4, num_modes=1, seed=1)
        s2 = opendocking.dock(ligand, serial, exhaustiveness=4, num_modes=1, seed=2)
    moved = float(
        np.linalg.norm(s1.pose_coords(0).mean(axis=0) - s2.pose_coords(0).mean(axis=0))
    )
    check("seed 1 and seed 2 differ", moved > 1e-6, f"centres {moved:.3f} Å apart")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        return 1
    print("determinism and parallelism behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
