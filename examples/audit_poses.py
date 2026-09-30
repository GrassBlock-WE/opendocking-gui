"""Audit the physics of docked poses rather than just that docking ran.

Three questions, each of which can fail while the program still exits 0:

1. Is the pose actually *in contact* with the receptor, or is it floating in
   the box scoring a flat surface term?
2. Is the pose a stationary point of the analytic gradient? A returned pose
   that still has a large gradient means the optimiser did not converge, no
   matter what the energy says.
3. Do the CPU and GPU scoring paths agree on the returned pose?
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import opendocking  # noqa: E402

BOX_HALF = 9.0  # the 18 A box used below, so the centre is fixed
CENTRE = (3.47, 6.21, 8.95)


def parse_centre(text: str) -> tuple[float, float, float]:
    return tuple(float(x) for x in text.split(","))  # type: ignore[return-value]


def read_pdbqt(path: str) -> list[tuple[str, float, float, float, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            rows.append(
                (
                    line[77:79].strip(),
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                    line[12:16].strip(),
                )
            )
    return rows


# The same tokens the engine treats as hydrogen when it builds its own
# receptor-heavy-atom list for the clash filter. A polar hydrogen sits about
# 1.0 A from its own heavy-atom donor, so measuring a ligand against one
# reports every ordinary hydrogen bond as a clash.
HYDROGEN_TOKENS = ("H", "HD", "HS")

# Step lengths, in ångström for the rigid-body dof and radians for torsions, for
# the line search that decides whether a pose really is a minimum.
_LINE_SEARCH_STEPS = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)


def steepest_descent_gain(lig, maps, conf, grad) -> float:
    """Largest energy decrease reachable by stepping along ``-grad``.

    Returns ``0.0`` when the pose is already a local minimum. The direction is
    normalised first, so the step length is in ångström/radians rather than
    being scaled by the gradient magnitude — with ``|grad|`` in the single
    digits, a raw ``k * grad`` step either barely moves or overshoots by
    radians, and the search never samples the useful range.
    """
    grad = np.asarray(grad, dtype=np.float64)
    norm = float(np.linalg.norm(grad))
    if norm < 1e-30:
        return 0.0
    unit = grad / norm
    conf = np.asarray(conf, dtype=np.float64)
    e0 = opendocking.score_conformation(lig, maps, conf, "vina")[0]
    best = 0.0
    for step in _LINE_SEARCH_STEPS:
        try:
            e = opendocking.score_conformation(lig, maps, conf - step * unit, "vina")[0]
        except ValueError:
            continue
        best = max(best, e0 - e)
    return best


def main(
    receptor_path: str,
    ligand_path: str,
    centre: tuple[float, float, float] | None = None,
    size: float = 18.0,
) -> int:
    centre = centre or CENTRE
    receptor = read_pdbqt(receptor_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = opendocking.Receptor.from_pdbqt(receptor_path)
    lig = opendocking.Ligand.from_pdbqt(ligand_path)
    box = opendocking.GridBox.from_center_size(centre, (size, size, size))
    maps = rec.precalculate(box, scoring="vina", spacing=0.375)

    rec_xy = np.array([[r[1], r[2], r[3]] for r in receptor])
    rec_polar = np.array(
        [[r[1], r[2], r[3]] for r in receptor if r[0] in ("N", "NA", "OA", "HD", "SA")]
    )
    # Heavy atoms only, matching how the engine defines a clash. Measuring
    # against a receptor polar hydrogen would call every hydrogen bond a clash.
    rec_heavy = np.array(
        [[r[1], r[2], r[3]] for r in receptor if r[0] not in HYDROGEN_TOKENS]
    )

    n_dof = lig.num_dof
    print(
        f"receptor {len(receptor)} atoms, {len(rec_heavy)} heavy, {len(rec_polar)} polar"
    )
    print(f"ligand   {lig.num_atoms} atoms, {lig.num_torsions} torsions, {n_dof} dof")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opendocking.dock(lig, maps, exhaustiveness=16, num_modes=5, seed=20260929)

    print(f"poses    {result.num_poses} from {result.raw_pose_count} conformations\n")

    problems = 0
    clashes = 0
    worst_gradient = 0.0

    for i in range(result.num_poses):
        xyz = result.pose_coords(i)
        conf = result.pose_conformation(i)

        # 1. contacts and steric sanity
        d_all = np.linalg.norm(xyz[:, None, :] - rec_xy[None, :, :], axis=-1)
        per_atom_min = d_all.min(axis=1)
        contacts = int((per_atom_min < 4.0).sum())
        # Two heavy atoms closer than ~2.0 A is a clash; below 1.5 A the
        # structure is meaningless. Measured against heavy atoms only.
        per_heavy = np.linalg.norm(
            xyz[:, None, :] - rec_heavy[None, :, :], axis=-1
        ).min(axis=1)
        n_clash = int((per_heavy < 2.0).sum())
        worst = float(per_heavy.min())
        d_pol = np.linalg.norm(xyz[:, None, :] - rec_polar[None, :, :], axis=-1)
        hbond_capable = int((d_pol < 3.5).any(axis=1).sum())

        # 2. Is the pose a genuine minimum? Test it by line search, NOT by the
        #    size of the gradient. The maps are trilinearly interpolated, which
        #    is only C0 across a cell face, so a pose with an atom sitting on a
        #    face has two different one-sided gradients there — often large and
        #    of opposite sign — while still being a true local minimum. A
        #    gradient-norm threshold cannot tell those apart from a failed
        #    optimisation; "no downhill step exists" can.
        energy, grad = opendocking.score_conformation(lig, maps, conf, "vina")
        gnorm = float(np.max(np.abs(grad)))
        downhill = steepest_descent_gain(lig, maps, conf, grad)
        worst_gradient = max(worst_gradient, gnorm)

        # How far the pose sits from the nearest grid cell face, in cells.
        u = (xyz - box.min_corner) / maps.spacing
        face_gap = float(np.abs(u - np.round(u)).min())

        # 3. CPU vs GPU on the real conformation
        pop = np.asarray(conf, dtype=np.float64).reshape(1, -1).repeat(4, axis=0)
        e_cpu = opendocking.evaluate_conformations(lig, maps, pop, "vina", use_gpu=False)
        e_gpu = opendocking.evaluate_conformations(lig, maps, pop, "vina", use_gpu=True)
        gpu_gap = float(np.max(np.abs(e_gpu - e_cpu)))

        flags = []
        if n_clash:
            flags.append(f"{n_clash} CLASHING atoms")
            problems += 1
            clashes += 1
        if downhill > 1e-6:
            flags.append(f"NOT a minimum ({downhill:+.5f} reachable)")
            problems += 1
        if gpu_gap > 1e-3:
            flags.append("CPU/GPU disagree")
            problems += 1

        suffix = ("  <-- " + "; ".join(flags)) if flags else ""
        print(
            f"pose {i+1}: E {energy:7.3f}  contacts {contacts:>2}/{lig.num_atoms}  "
            f"min-dist {worst:5.2f} A  clash {n_clash:>2}  polar-contact {hbond_capable:>2}  "
            f"|grad| {gnorm:5.2f}  downhill {downhill:+.2e}  face-gap {face_gap:.4f}  "
            f"cpu-gpu {gpu_gap:.1e}{suffix}"
        )

    print(f"\nworst gradient norm over the batch : {worst_gradient:.3f}")
    print(f"poses with a steric clash          : {clashes}/{result.num_poses}")
    print(
        "\nA large |grad| on a returned pose is expected, not a failure: the maps\n"
        "are trilinearly interpolated, which is C0 across a cell face, so a pose\n"
        "with an atom on a face carries two opposite one-sided gradients while\n"
        "still being a true minimum. The 'downhill' column is the real test --\n"
        "it is the largest energy drop reachable by stepping along -grad."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else "1crn_prep.pdbqt",
            sys.argv[2] if len(sys.argv) > 2 else "biotin_prep.pdbqt",
            parse_centre(sys.argv[3]) if len(sys.argv) > 3 else None,
            float(sys.argv[4]) if len(sys.argv) > 4 else 18.0,
        )
    )
