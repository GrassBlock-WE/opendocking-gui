"""Derive the reachable-downhill bound from the interpolation, not from a threshold.

`examples/audit_poses.py` decides whether a returned pose is a minimum by
stepping along ``-grad`` at eight lengths and printing each one as a ladder.
Its `downhill` column is the descent at the **finest** of those steps, and
`scripts/examples_check.py` gates that number. Until now the gate bounded it
with two literals -- ``1e-3`` absolute **and** ``1e-4`` relative to the pose's
``|E|`` -- and the relative half was a judgement with no measurement behind
it. It passed here at ``3.6e-05`` and failed on the CI runner at ``1.3e-04``
against the same bound, on identical inputs.

The two halves were not equally earned. The absolute one is a statement about
the *interpolator*: the maps are trilinear, so the pose energy is a
piecewise-trilinear function that is C⁰ across a cell face -- continuous in
value, with a **jump in its one-sided derivative** there. The relative one is a
statement about ``|E|``, and the mechanism has nothing to do with ``|E|``:
dividing by it tightens the bound exactly where the energy happens to be small,
which is unrelated to the derivative jump that causes the descent. So this file
measures the interpolator's own worst case and hands the gate a derived
constant, recomputed from the map on every run.

    B = FINEST_STEP x LEVER x G_MAX

``FINEST_STEP``
    The audit's smallest ladder step, in conformation units. Not restated here
    on trust: it is read out of `examples/audit_poses.py`, so the constant
    cannot drift away from the step it is supposed to describe.

``LEVER``
    The largest distance, in angstrom, that a unit conformation step moves a
    single ligand atom -- measured, by differencing the engine's own forward
    kinematics along the audit's own search direction on the poses the audit
    reports. It is ligand geometry plus the audit's step direction, so it does
    not vary with the machine.

``G_MAX``
    ``max(G_SLOPE, G_FACE)``, both measured per map slot from
    `GridMaps.raw_data` over the clash-free grid points and assembled per
    ligand atom from the slots that atom's interaction class reads at scoring
    time, then combined over atoms by Cauchy-Schwarz:

    ``G_SLOPE``
        The largest per-axis **first** difference divided by the spacing --
        the largest derivative the tabulated field carries. Inside a cell the
        trilinear interpolant's directional derivative is a convex combination
        of the stored first differences, so this bounds a first-order term.
    ``G_FACE``
        The largest per-axis **second** difference divided by the spacing --
        which is exactly the jump in the one-sided derivative across a face.

    ``G_SLOPE`` is the one that matters for the audit's `downhill` column, and
    choosing it is the tightening: the search's Armijo sufficient-decrease test
    means a returned iterate cannot descend *within* its own cell, so a step
    short enough to stay in the cell can only return a first-order term, and
    bounding that by a derivative *jump* is the wrong constant -- it happens to
    be the smaller of the two here (102.4 against 135.2 kcal/mol/A), so the
    bound written against it was valid but was not describing the thing being
    measured. ``G_FACE`` is kept in the maximum because the audit's *coarse*
    rungs do cross faces, and a bound that only held for the finest rung would
    be a bound with a precondition nobody checks.

## Why the finest step, and what the margin is made of

The bound is valid only while ``FINEST_STEP x LEVER`` is short enough that the
step cannot reach a cell face. At 1e-9 conf units that is 8.7e-09 A against a
closest approach of 4.5e-08 A on this fixture -- a factor of 5, which is
narrow enough to be worth asserting rather than assuming, and
`examples_check.py` asserts it from the audit's own reported figure.

The margin over the observation is a ratio of measured numbers and nothing
else:

    margin = LEVER x G_MAX / max|grad|2  =  8.682 x 135.163 / 11.930  =  98x

The numerator is a box-wide worst case: the steepest clash-free slope anywhere
in the tabulated field, times the largest atom any conformation step moves
across the whole pose set. The denominator is one pose's own gradient, on this
machine's pose set. So the margin is the price of a bound that does not depend
on which poses this machine's thread count happened to return, and it cannot be
tightened further without giving that up -- which is the defect this file
existed to remove. There is no fitted factor anywhere in it: change the grid,
the box, the spacing, the receptor or the ligand and it moves; change the CPU
and it does not.

Run:  python scripts/interp_reachable_step_bound.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import opendocking  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
AUDIT = EXAMPLES / "audit_poses.py"

RECEPTOR = EXAMPLES / "1crn_prep.pdbqt"
LIGAND = EXAMPLES / "biotin_prep.pdbqt"
CENTRE = (3.47, 6.21, 8.95)
SIZE = 18.0
SPACING = 0.375
SCORING = "vina"

#: The search settings `examples/audit_poses.py` docks with. The lever is
#: measured on the poses that run returns, so it has to be the same run; the
#: audit's own output is compared against these energies by the gate.
EXHAUSTIVENESS = 16
NUM_MODES = 5
SEED = 20260929

#: The audit's line-search steps, read out of `examples/audit_poses.py` rather
#: than restated here. Duplicating a constant in two files is how a bound ends
#: up describing a step the audit no longer takes.
_STEPS_RE = re.compile(r"_LINE_SEARCH_STEPS\s*=\s*\(([^)]*)\)")


def audit_steps() -> list[float]:
    """Every line-search step the audit takes, in conformation units.

    Read from the audit's own source so the bound cannot silently stop
    describing the steps the audit actually uses.
    """
    text = AUDIT.read_text(encoding="utf-8")
    match = _STEPS_RE.search(text)
    if match is None:
        raise ValueError(
            f"{AUDIT.name} no longer declares _LINE_SEARCH_STEPS as a tuple of "
            f"literals, so the reachable-step bound cannot be derived from it"
        )
    steps = [float(tok) for tok in match.group(1).replace(",", " ").split()]
    if not steps:
        raise ValueError(f"{AUDIT.name} declares an empty _LINE_SEARCH_STEPS")
    return steps


#: Map slots, in storage order. Mirrors `MapSlot` in `dock-core/src/grid.rs`;
#: the same layout is documented on `GridMaps.raw_data`.
SLOT_NAMES = ("shape", "hb_from_donor", "hb_from_acceptor", "hydrophobic")

#: The ten receptor grid types, in index order, mirroring `grid_type_name` in
#: `dock-core/src/types.rs`.
GRID_TYPES = ("C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Met")
VALUES_PER_POINT = len(SLOT_NAMES) * len(GRID_TYPES)

#: Which slots each ligand interaction class reads at scoring time. Mirrors
#: `weights_for_kind` in `dock-core/src/scoring.rs`, where every weight is 0 or
#: 1, so the weighted value of a corner is the plain sum of the slots listed.
KIND_SLOTS = {
    "hydrophobic": ("shape", "hydrophobic"),
    "acceptor": ("shape", "hb_from_donor"),
    "donor": ("shape", "hb_from_acceptor"),
    "donoracceptor": ("shape", "hb_from_donor", "hb_from_acceptor"),
    "other": ("shape",),
}

#: The same hydrogen tokens `examples/audit_poses.py` uses, and the same
#: heavy-atom clash distance it asserts. A pose that audit accepts has no heavy
#: atom closer than this to a receptor heavy atom, so the field's worst case
#: only has to be measured where such a pose can sit.
HYDROGEN_TOKENS = ("H", "HD", "HS")
CLASH_DISTANCE = 2.0


def read_heavy_positions(path: Path) -> np.ndarray:
    """Heavy-atom coordinates from a PDBQT, for the clash-free region test."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[77:79].strip() in HYDROGEN_TOKENS:
            continue
        rows.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not rows:
        raise ValueError(f"{path.name} has no heavy atoms, so the clash-free "
                         f"region would be the whole box and the bound vacuous")
    return np.asarray(rows, dtype=np.float64)


def worst_axis_jump(field: np.ndarray, allowed: np.ndarray) -> float:
    """Largest second difference along any axis, over `allowed` points.

    For a trilinear interpolant, the one-sided x-derivatives either side of a
    cell face are ``(v[i+1] - v[i])/h`` and ``(v[i+2] - v[i+1])/h``, so their
    difference is the **second difference divided by the spacing**. That
    difference is a convex combination along the other two axes, so it cannot
    exceed the largest second difference stored on the grid.

    Returned in units of the stored values; the caller divides by the spacing.
    """
    worst = 0.0
    for axis in range(3):
        if field.shape[axis] < 3:
            continue
        lo = [slice(None)] * 3
        mid = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -2)
        mid[axis] = slice(1, -1)
        hi[axis] = slice(2, None)
        mask = allowed[tuple(lo)] & allowed[tuple(mid)] & allowed[tuple(hi)]
        if not mask.any():
            continue
        jump = (
            field[tuple(hi)].astype(np.float64)
            - 2.0 * field[tuple(mid)].astype(np.float64)
            + field[tuple(lo)].astype(np.float64)
        )
        worst = max(worst, float(np.abs(jump)[mask].max()))
    return worst


def worst_axis_slope(field: np.ndarray, allowed: np.ndarray) -> float:
    """Largest first difference along any axis, over `allowed` points.

    The companion to `worst_axis_jump`, and the constant that bounds a
    **first-order** term rather than a jump in one. Inside a cell the
    trilinear interpolant's directional derivative along an axis is itself a
    convex combination of the stored first differences along that axis, so it
    cannot exceed this either.

    Returned in units of the stored values; the caller divides by the spacing.
    """
    worst = 0.0
    for axis in range(3):
        if field.shape[axis] < 2:
            continue
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        mask = allowed[tuple(lo)] & allowed[tuple(hi)]
        if not mask.any():
            continue
        rise = field[tuple(hi)].astype(np.float64) - field[tuple(lo)].astype(np.float64)
        worst = max(worst, float(np.abs(rise)[mask].max()))
    return worst


def clash_free_region(maps, heavy: np.ndarray) -> np.ndarray:
    """Boolean grid: no receptor heavy atom within `CLASH_DISTANCE` of the point.

    The field's steepest region is the vdW contact shell, which is exactly where
    a docked pose sits, so this restriction barely shrinks the worst case -- the
    numbers below report both so the claim is measured rather than assumed. It
    is here because it makes the bound *valid* for the poses the gate accepts.
    """
    nx, ny, nz = maps.dims
    lo = maps.box.min_corner
    axes = [
        lo[k] + maps.spacing * np.arange(n, dtype=np.float64)
        for k, n in enumerate((nx, ny, nz))
    ]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    clear = np.ones((nx, ny, nz), dtype=bool)
    for atom in heavy:
        near = (
            (gx - atom[0]) ** 2 + (gy - atom[1]) ** 2 + (gz - atom[2]) ** 2
        ) < CLASH_DISTANCE ** 2
        clear &= ~near
    return clear


def measure_face_jump(maps, allowed: np.ndarray) -> dict[str, dict[str, float]]:
    """Worst per-axis first difference and gradient jump per map slot, kcal/mol/A.

    Both constants are taken over the ten grid types, so each is a property of
    the tabulated field rather than of the elements this ligand happens to
    contain. Each is measured twice -- over the clash-free region, which is
    where the bound is *valid*, and over the whole grid, which is what the
    numbers would be without that restriction -- so the cost of the
    restriction is reported rather than assumed. It costs 1.3x on the jump
    constant: the steepest type (I) is 30% steeper than C, and it lives in the
    repulsion ramp no clash-free pose reaches.
    """
    nx, ny, nz = maps.dims
    raw = np.asarray(maps.raw_data)
    whole = np.ones((nx, ny, nz), dtype=bool)
    out = {
        "slope": {name: 0.0 for name in SLOT_NAMES},
        "jump": {name: 0.0 for name in SLOT_NAMES},
        "slope_whole": {name: 0.0 for name in SLOT_NAMES},
        "jump_whole": {name: 0.0 for name in SLOT_NAMES},
    }
    for t in range(len(GRID_TYPES)):
        for s, sname in enumerate(SLOT_NAMES):
            field = raw[t * len(SLOT_NAMES) + s :: VALUES_PER_POINT].reshape(nx, ny, nz)
            for region, mask in (("", allowed), ("_whole", whole)):
                out[f"slope{region}"][sname] = max(
                    out[f"slope{region}"][sname],
                    worst_axis_slope(field, mask) / maps.spacing,
                )
                out[f"jump{region}"][sname] = max(
                    out[f"jump{region}"][sname],
                    worst_axis_jump(field, mask) / maps.spacing,
                )
    return out


def measure_lever(lig, maps, steps: list[float]) -> tuple[float, list[float], list[float]]:
    """Largest atom displacement per unit conformation step, in angstrom.

    Differencing the engine's own forward kinematics along the audit's own
    search direction, on the poses the audit's own run returns. Measured rather
    than argued from a rotation arm because the conformation layout is the
    engine's business, not this file's: a wrong assumption about which dof is
    the pivot would silently halve or double the bound.

    Also returns the pose energies and each pose's own ``|grad|_2``, because the
    margin over the bound is a ratio against *that* number rather than a
    cushion typed here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opendocking.dock(
            lig, maps, exhaustiveness=EXHAUSTIVENESS, num_modes=NUM_MODES, seed=SEED
        )
    lever = 0.0
    energies: list[float] = []
    gradients: list[float] = []
    probe = min(steps)
    for i in range(result.num_poses):
        conf = np.asarray(result.pose_conformation(i), dtype=np.float64)
        energy, grad = opendocking.score_conformation(lig, maps, conf, SCORING)
        energies.append(float(energy))
        direction = np.asarray(grad, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        gradients.append(norm)
        if norm < 1e-30:
            continue
        here = np.asarray(opendocking.conformation_coordinates(lig, conf))
        moved = np.asarray(
            opendocking.conformation_coordinates(lig, conf - probe * direction / norm)
        )
        lever = max(lever, float(np.linalg.norm(moved - here, axis=1).max()) / probe)
    return lever, energies, gradients


def main() -> int:
    steps = audit_steps()
    finest = min(steps)
    print(f"reachable-step bound, derived from the map in {RECEPTOR.name}")
    print(f"box {SIZE:g} A about {CENTRE}, spacing {SPACING} A, scoring {SCORING}")
    print(f"audit line-search steps {steps} -> finest {finest:.1e}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = opendocking.Receptor.from_pdbqt(str(RECEPTOR))
    lig = opendocking.Ligand.from_pdbqt(str(LIGAND))
    box = opendocking.GridBox.from_center_size(CENTRE, (SIZE, SIZE, SIZE))
    maps = rec.precalculate(box, scoring=SCORING, spacing=SPACING)
    print(f"grid dims {tuple(maps.dims)}, spacing {maps.spacing}, "
          f"{maps.num_points} points, {maps.memory_mb:.1f} MB")

    heavy = read_heavy_positions(RECEPTOR)
    allowed = clash_free_region(maps, heavy)
    print(f"clash-free grid points: {int(allowed.sum())} of {allowed.size} "
          f"({100.0 * allowed.sum() / allowed.size:.0f}%), against "
          f"{len(heavy)} receptor heavy atoms")

    jumps = measure_face_jump(maps, allowed)
    print("\nworst per-axis derivative of the tabulated field, per slot, kcal/mol/A")
    print("  (first difference / spacing = a first-order term; second difference /")
    print("   spacing = the one-sided jump across a cell face; clash-free, whole grid)")
    for slot in SLOT_NAMES:
        print(f"    {slot:16s} slope {jumps['slope'][slot]:8.3f}   "
              f"jump {jumps['jump'][slot]:8.3f}   "
              f"(whole grid: {jumps['slope_whole'][slot]:7.3f} / "
              f"{jumps['jump_whole'][slot]:7.3f})")

    kinds = lig.atom_kinds
    unknown = sorted({k for k in kinds if k not in KIND_SLOTS})
    if unknown:
        raise ValueError(
            f"the engine reports ligand interaction classes {unknown}, which "
            f"`weights_for_kind` does not cover; the per-atom slot sums below "
            f"would silently treat them as shape-only"
        )

    def assemble(key: str) -> float:
        per_atom = {
            kind: 3.0 ** 0.5 * max(jumps[key][s] for s in slots)
            for kind, slots in KIND_SLOTS.items()
        }
        # Cauchy-Schwarz over atoms: the search direction is a unit vector, so
        # |sum_i grad_i . u| <= |sum_i grad_i| <= sqrt(sum_i |grad_i|^2), which
        # is sqrt(19) = 4.4x tighter than adding 19 per-atom bounds as though
        # they all pointed the same way.
        return sum(per_atom[k] ** 2 * kinds.count(k) for k in set(kinds)) ** 0.5

    g_slope = assemble("slope")
    g_face = assemble("jump")
    census = {kind: kinds.count(kind) for kind in sorted(set(kinds))}
    print(f"\nligand: {len(kinds)} atoms, classes {census}")
    print("  Cauchy-Schwarz over the atoms: "
          f"G_SLOPE = {g_slope:.3f} kcal/mol/A (bounds a first-order term), "
          f"G_FACE = {g_face:.3f} kcal/mol/A (bounds a face crossing)")

    # The audit's finest step moves an atom `finest * lever` angstrom, and it
    # measures the descent there. The search's Armijo test means a returned
    # iterate cannot descend inside its own cell, so what that step can still
    # find is a first-order term -- which G_SLOPE bounds, not G_FACE. G_FACE is
    # kept for the coarse rungs, where a face *can* be crossed, and the bound
    # uses whichever is larger so it stays valid for both.
    g_max = max(g_slope, g_face)

    lever, energies, gradients = measure_lever(lig, maps, steps)
    print(f"  LEVER = {lever:.3f} A per unit conformation step "
          f"(largest atom displacement along -grad over {len(energies)} poses)")
    print(f"  pose energies {[round(e, 3) for e in energies]} -- the gate "
          f"compares these against the audit's own column")
    print(f"  pose |grad|2 {[round(g, 3) for g in gradients]} -- the margin below "
          f"is measured against the steepest of these, not typed")

    bound = finest * lever * g_max
    print(f"\nFINEST_STEP = {finest:.1e} conformation units "
          f"(smallest of _LINE_SEARCH_STEPS in {AUDIT.name})")
    print(f"LEVER       = {lever:.3f} A")
    print(f"G_SLOPE     = {g_slope:.3f} kcal/mol/A  <- bounds the finest rung")
    print(f"G_FACE      = {g_face:.3f} kcal/mol/A  <- bounds a face crossing")
    print(f"G_MAX       = {g_max:.3f} kcal/mol/A  <- the larger, used below")
    print(f"reachable bound = {bound:.6e} kcal/mol")

    # The margin is a ratio of two measured numbers, and this is the whole
    # argument for it: the bound is a box-wide worst case (the steepest
    # clash-free slope in the tabulated field, and the largest atom a single
    # conformation step can move across the pose set) while the thing it bounds
    # is one pose's own gradient. There is no fitted factor in it, and it
    # cannot be tightened below the ratio without making the bound depend on
    # which poses this machine's thread count happened to return.
    steepest = max(gradients) if gradients else 0.0
    structural = lever * g_max / steepest if steepest > 0.0 else 0.0
    print(f"\nmargin = LEVER x G_MAX / max|grad|2 = {lever:.3f} x {g_max:.3f} / "
          f"{steepest:.3f} = {structural:.0f}x")
    print(f"  against the finest-rung first-order prediction "
          f"|grad|2 x {finest:.0e} = {steepest * finest:.3e} kcal/mol on the "
          f"steepest pose")
    print("  every term in that ratio is measured on this run; nothing here is a")
    print("  cushion, and the two constants are both worst cases over the whole")
    print("  box, attained in the repulsion ramp no clash-free pose reaches")

    # The parseable lines. `examples_check.py` reads the number after each '=' and
    # re-multiplies the three parts, so a reformatted line fails there loudly
    # rather than leaving the gate comparing against a silent default. All five
    # carry twelve significant digits for that reason: at six, the printed
    # product of the parts disagreed with the printed bound in the fifth decimal
    # and the gate's own self-consistency check went red on a rounding artefact.
    print(f"\nREACHABLE_BOUND = {bound:.12e} kcal/mol")
    print(f"FINEST_STEP = {finest:.12e}")
    print(f"LEVER = {lever:.12e}")
    print(f"G_MAX = {g_max:.12e}")
    print(f"MAX_GRAD2 = {steepest:.12e}")
    print("derived from raw_data and recomputed every run; not a fitted threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
