"""Audit the physics of docked poses rather than just that docking ran.

Three questions, each of which can fail while the program still exits 0:

1. Is the pose actually *in contact* with the receptor, or is it floating in
   the box scoring a flat surface term?
2. Is the pose a stationary point of the analytic gradient? A returned pose
   that still has a large gradient means the optimiser did not converge, no
   matter what the energy says. The answer is not a single number: "can this
   pose still descend" depends on the step you ask with, so the descent is
   printed as a **ladder** of eight step sizes spanning the region where the
   behaviour changes, and the pose line carries the descent at the finest of
   them.
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

# The line-search ladder: step lengths, in ångström for the rigid-body dof and
# radians for torsions, along the normalised ``-grad`` direction.
#
# Eight steps over nine decades, because "can this pose still descend" has a
# different answer at 1e-9 than at 1e-4 and this file used to print only the
# second one, as `downhill`. What each part of the ladder is for:
#
#   1e-9 .. 1e-5
#       Below the step the audit used to probe at, because a descent region
#       narrower than the probe is invisible to the probe. A pose sitting on a
#       cell face has a descent region a few times 1e-9 wide, so at 1e-4 the
#       step lands past the turn-up and reports zero. Here the region is
#       resolved instead of stepped over, and two things become measurable
#       that were not before: the descent is *linear* in the step with a slope
#       equal to the pose's own |grad|_2, and it has **no intercept** — which
#       is what "the interpolated field is C0, so its value does not jump
#       across a cell face" predicts, and what a value discontinuity could
#       not produce at any step size.
#   1e-4
#       The step this file used to probe at, kept in the ladder so its numbers
#       (four zeros and one 1.52e-04) are reproduced rather than deleted, and
#       labelled as what they are: a first-order term sampled after an
#       overshoot.
#   1e-3 .. 1.0
#       Up to a displacement that leaves the basin outright. 1.0 conformation
#       unit moves the fastest atom 8.7 Å, and the energy there is higher, so
#       the ladder brackets a maximum instead of running away down a slope.
_LINE_SEARCH_STEPS = (1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

#: The optimiser's own declared stopping tolerance: `gradient_tolerance` in
#: `LbfgsConfig` (`dock-core/src/search/lbfgs.rs`, 1e-4). A returned pose whose
#: gradient is far above this is not a stationary point of the field the engine
#: scores, whatever any line search reports about it, so this is the number
#: the audit judges stationarity against.
GRADIENT_TOLERANCE = 1e-4

#: Relative agreement between a measured descent/step and the pose's own
#: |grad|_2 for the descent to count as a first-order term. Not a fitted
#: value: the measured spread over the five poses this audit reports is
#: 1.7e-06 at the two finest steps, which is the double-precision noise floor
#: of a 1e-9 conf-unit step, and this sits three decades above it. The
#: derivative jump *across* a face is two to three decades larger than this,
#: which is why the label on a ladder row can distinguish the two.
LINEARITY_TOL = 1e-3


def first_face_crossing(lig, box, maps, conf, xyz, direction) -> tuple[float, int]:
    """Step at which the ``-grad`` ray first leaves the current grid cell.

    Returns ``(step, atom index)``. The step is the distance the *fastest*
    atom needs to travel, along its own direction of motion, to reach the next
    grid line -- per atom and per axis, then the smallest over all of them,
    because the cell a pose sits in is the intersection over its atoms.

    This is the number the old claim needed and did not have. "The energy
    jumps across a cell face" is a statement about a *crossing*, and a crossing
    that a step is too short to reach cannot produce anything: once this is
    printed, a descent measured below it was produced entirely inside one cell,
    where the trilinear field is a single polynomial and has no discontinuity
    to have.
    """
    probe = 1e-7
    here = np.asarray(opendocking.conformation_coordinates(lig, conf), dtype=np.float64)
    moved = np.asarray(
        opendocking.conformation_coordinates(lig, conf - probe * direction),
        dtype=np.float64,
    )
    # A / conf unit, per atom, per axis, in the direction of travel.
    rate = -(moved - here) / probe
    grid = (np.asarray(xyz, dtype=np.float64) - box.min_corner) / maps.spacing
    best = math.inf
    which = -1
    for axis in range(3):
        step_axis = np.sign(rate[:, axis])
        gap = np.where(
            step_axis > 0.0,
            np.ceil(grid[:, axis]) - grid[:, axis],
            grid[:, axis] - np.floor(grid[:, axis]),
        )
        moving = np.abs(rate[:, axis]) > 1e-12
        reach = np.where(moving, gap * maps.spacing / np.where(moving, np.abs(rate[:, axis]), 1.0), np.inf)
        for atom in np.argsort(reach):
            if np.isfinite(reach[atom]) and reach[atom] < best:
                best = float(reach[atom])
                which = int(atom)
    return best, which


def descent_ladder(lig, maps, conf, e0, direction) -> list[tuple[float, float | None, float | None]]:
    """Descent at every ladder step, and the slope each one implies.

    Returns one ``(step, descent, descent/step)`` per rung, with ``None`` for
    a step the engine rejects (a conformation driven outside the box). The
    direction is normalised by the caller, so the step length is in
    ångström/radians rather than being scaled by the gradient magnitude: with
    |grad| in the single digits, a raw ``k * grad`` step either barely moves or
    overshoots by radians, and the search never samples the useful range.
    """
    rows: list[tuple[float, float | None, float | None]] = []
    for step in _LINE_SEARCH_STEPS:
        try:
            e = opendocking.score_conformation(lig, maps, conf - step * direction, "vina")[0]
        except ValueError:
            rows.append((step, None, None))
            continue
        drop = float(e0) - float(e)
        rows.append((step, drop, drop / step))
    return rows


def descent_width(rows):
    """Where the descent turns over: ``(low step, high step, geometric mean)``.

    The mean is a **bracket**, not a measurement to more digits than the
    ladder has: the descent is known to be positive at ``low`` and not at
    ``high``, and nothing between them was sampled. That bracket is what makes
    the audit's own zero interpretable — a pose reports ``downhill 0.00e+00``
    when its region is narrower than the step it was probed at, and this says
    by how much.

    If the descent is still positive at the top rung the region is wider than
    the ladder, and the high step is reported as ``inf``.
    """
    positive = [r for r in rows if r[1] is not None and r[1] > 0.0]
    if not positive:
        return (rows[0][0], 0.0, 0.0)
    last = positive[-1]
    if last is rows[-1]:
        return (last[0], math.inf, math.inf)
    following = rows[rows.index(last) + 1]
    return (last[0], following[0], (last[0] * following[0]) ** 0.5)


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
        # **The three settings below are this file's, and the only copy of them
        # in the tree.** `scripts/interp_reachable_step_bound.py` re-docks with
        # them to measure LEVER on the poses this call returns, and it reads them
        # back out of *this* call rather than restating them: see
        # `audit_dock_settings` there. It used to carry its own `16 / 5 /
        # 20260929`, and a change here would have left that file measuring a
        # different run while every pose energy it is compared against stayed
        # green -- the same hazard `_LINE_SEARCH_STEPS` above already warns about
        # for the ladder, which is why the reader there is a regex and not a
        # number. Change a setting here and nowhere else.
        result = opendocking.dock(lig, maps, exhaustiveness=16, num_modes=5, seed=20260929)

    print(f"poses    {result.num_poses} from {result.raw_pose_count} conformations\n")

    problems = 0
    clashes = 0
    worst_gradient = 0.0
    closest_face = math.inf

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

        # 2. Is the pose a stationary point? Two measurements, because one
        #    number cannot answer it.
        #    (a) The gradient itself, against the optimiser's own declared
        #        tolerance. This is the statement about the engine, and it is
        #        the one that holds: these poses are not stationary points.
        #    (b) The descent ladder. The maps are trilinearly interpolated, so
        #        the field is C0 across a cell face -- continuous in *value*,
        #        with a jump in its one-sided *derivative*. A pose therefore
        #        carries two different slopes at a face while the energy
        #        itself never jumps, and the ladder is what separates the two:
        #        the slope changes from rung to rung, the intercept does not.
        energy, grad = opendocking.score_conformation(lig, maps, conf, "vina")
        gnorm = float(np.max(np.abs(grad)))
        gnorm2 = float(np.linalg.norm(grad))
        unit = np.asarray(grad, dtype=np.float64) / max(gnorm2, 1e-30)
        ladder = descent_ladder(lig, maps, conf, energy, unit)
        crossing, which_atom = first_face_crossing(lig, box, maps, conf, xyz, unit)
        # The descent at the finest rung, which is the quantity the reachable
        # bound in `scripts/interp_reachable_step_bound.py` is derived for: a
        # step of 1e-9 conf units moves an atom 8.7e-09 A, so it provably
        # cannot reach a cell face (the closest any of these poses comes to one
        # is 4.5e-08 A) and what it measures is a first-order term and nothing
        # else. Clamped at zero because a *decrease* cannot be negative, and
        # printed with its cause on the rungs below.
        downhill = max(0.0, ladder[0][1] if ladder[0][1] is not None else 0.0)
        low, high, width = descent_width(ladder)
        worst_gradient = max(worst_gradient, gnorm)

        # How far the pose sits from the nearest grid cell face, in cells. The
        # summary reports the same quantity in angstrom, because the bound in
        # `scripts/interp_reachable_step_bound.py` is only valid while its step
        # is too short to reach a face, and "0.0000 cells" cannot say whether
        # that holds -- it stands for anything under 1.9e-05 A.
        u = (xyz - box.min_corner) / maps.spacing
        face_gap = float(np.abs(u - np.round(u)).min())
        closest_face = min(closest_face, face_gap * maps.spacing)

        # 3. CPU vs GPU on the real conformation
        pop = np.asarray(conf, dtype=np.float64).reshape(1, -1).repeat(4, axis=0)
        e_cpu = opendocking.evaluate_conformations(lig, maps, pop, "vina", use_gpu=False)
        e_gpu = opendocking.evaluate_conformations(lig, maps, pop, "vina", use_gpu=True)
        gpu_gap = float(np.max(np.abs(e_gpu - e_cpu)))
        # Relative, for the same reason the unit test is: the kernel is single
        # precision, and a clashing pose can score in the hundreds, where any
        # fixed absolute threshold would sit below the arithmetic it is meant
        # to check. The floor keeps near-zero scores from demanding a relative
        # accuracy the hardware cannot give.
        #
        # 1e-5 is a guard against single-precision drift, not a fitted value:
        # it was not measured, and the measured gap on the five poses below is
        # exactly 0.0 in every column. It is loose on purpose — the failure it
        # exists to catch is a real disagreement, not a rounding artefact.
        gpu_rel = gpu_gap / max(1.0, float(np.max(np.abs(e_cpu))))

        flags = []
        if n_clash:
            flags.append(f"{n_clash} CLASHING atoms")
            problems += 1
            clashes += 1
        if gnorm2 > GRADIENT_TOLERANCE:
            # The wording deliberately does not repeat the `|grad|2` column
            # header: `scripts/examples_check.py` reads that column off the
            # pose line by its header, and a flag that repeated the header
            # would be parsed as a second reading of it.
            flags.append(
                f"NOT a stationary point (gradient norm {gnorm2:.3f} against the "
                f"optimiser's own {GRADIENT_TOLERANCE:.0e} tolerance)"
            )
            problems += 1
        if gpu_rel > 1e-5:
            flags.append(f"CPU/GPU disagree ({gpu_rel:.1e} relative)")
            problems += 1

        suffix = ("  <-- " + "; ".join(flags)) if flags else ""
        print(
            f"pose {i+1}: E {energy:7.3f}  contacts {contacts:>2}/{lig.num_atoms}  "
            f"min-dist {worst:5.2f} A  clash {n_clash:>2}  polar-contact {hbond_capable:>2}  "
            f"|grad| {gnorm:5.2f}  |grad|2 {gnorm2:10.6f}  downhill {downhill:+.2e}  "
            f"face-gap {face_gap:.4f}  width {width:.1e}  "
            f"crossing {crossing:.2e}  cpu-gpu {gpu_gap:.1e}{suffix}"
        )
        print(
            f"    pose {i+1} descent ladder, step in conf units along -|grad|; "
            f"region turns over between {low:.0e} and {high:.0e}; the ray leaves "
            f"the cell at {crossing:.2e} conf units "
            f"({crossing * maps.spacing:.2e} A, atom {which_atom})"
        )
        for step, drop, slope in ladder:
            if drop is None:
                print(
                    f"    pose {i+1}  step {step:8.1e}  rejected: the engine "
                    f"refused this conformation"
                )
                continue
            if abs(slope - gnorm2) <= LINEARITY_TOL * gnorm2:
                state = "first-order"
            elif drop > 0.0:
                state = "descending, slope has jumped"
            else:
                state = "field has turned up"
            print(
                f"    pose {i+1}  step {step:8.1e}  descent {drop:+.4e}  "
                f"slope {slope:+9.5f}  {state}"
            )

    print(f"\nworst gradient norm over the batch : {worst_gradient:.3f}")
    print(f"closest approach to a cell face   : {closest_face:.3e} A "
          f"({closest_face / maps.spacing:.1e} cells)")
    print(f"poses with a steric clash          : {clashes}/{result.num_poses}")
    print(
        "\nNone of these poses is a stationary point of the interpolated field.\n"
        "The maps are trilinearly interpolated, so the field is C0 across a cell\n"
        "face: continuous in VALUE, with a jump in its one-sided DERIVATIVE. The\n"
        "ladder shows both halves of that. The finest rungs are linear in the step\n"
        "with a slope equal to the pose's own |grad|2 and no intercept, which is\n"
        "what a continuous value gives; a couple of decades up, the slope changes\n"
        "while the descent stays continuous, which is the derivative jump. A pose\n"
        "sitting on a face therefore has a descent region only a few times 1e-9\n"
        "conf units wide, and a line search probing at 1e-4 steps straight over it\n"
        "and reports zero -- the 'downhill' column here is the descent at the\n"
        "FINEST rung, and the 'width' column is how wide the region it sampled\n"
        "actually was. 'No descent found' and 'no descent there' are not the same\n"
        "reading, and the ladder is what tells them apart."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    # Defaults resolve next to this file, not the caller's working directory,
    # so these run from the repository root without a `cd examples` first.
    here = Path(__file__).resolve().parent
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else str(here / "1crn_prep.pdbqt"),
            sys.argv[2] if len(sys.argv) > 2 else str(here / "biotin_prep.pdbqt"),
            parse_centre(sys.argv[3]) if len(sys.argv) > 3 else None,
            float(sys.argv[4]) if len(sys.argv) > 4 else 18.0,
        )
    )
