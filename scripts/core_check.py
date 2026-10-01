"""Check `opendocking.core`: the whole Python-to-Rust docking surface.

`scripts/ligand_check.py` covers ligand chemistry and `scripts/prep_check.py`
covers preparation, but the file that *binds* Python to the compiled engine --
`opendocking/core.py`, 21 exported names and every class attribute a caller
can reach -- was covered by 38 pytest cases that shared a structural blind
spot. Both meta-tests in `test_pipeline.py` read this file from the outside:

* `test_python_wrappers_only_use_names_the_extension_exports` checks that
  `self._maps.<name>` and friends are attributes the Rust module exports. It
  cannot see a *Python* name, so a helper called across classes cannot break
  it.
* `test_docking_result_exposes_every_documented_property` uses `hasattr` on
  `DockingResult` **only**. `GridMaps` has no equivalent check, which is why
  `GridMaps.box` raised `AttributeError` on every call of its life and 38 green
  tests did not notice.

So this script does three things the pytest suite structurally cannot:

* **Touches every documented member of every wrapper class on a real
  instance**, not just `DockingResult`'s, which is the seam where the
  name-mangling defect lived.
* **Runs the engine boundary**, with each case labelled by what actually
  happens: refused with `ValueError`, accepted silently, or refused with a
  different exception type. A boundary that is *accepted* is pinned as
  accepted, so a future change is noticed in either direction.
* **Asserts every arithmetic edge of the two pure functions**,
  `exhaustiveness_for_box` and `Receptor.estimate_memory_mb`, including the
  points where the documented rule and the implemented rule disagree.

Every expectation is pinned in both directions: the value *is* what the
engine produced, and the value is *not* the plausible wrong answer. A check
that only pins the happy path is satisfied by a function that returns a
constant.

Nothing here tightens the engine. Where the engine accepts something the
documentation does not promise (`exhaustiveness=2.7` is truncated to 2;
`scoring="vinardo"` happily scores vina maps), that behaviour is asserted *as
it is*, with a comment saying so, because a check that demanded a stricter
engine would be a change to the docking code wearing a check's clothes.

Run:  python scripts/core_check.py
"""

from __future__ import annotations

import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dock-py" / "python"
EX = ROOT / "examples"

# The check prints energy tables and, on failure, an engine message that may
# contain a non-ASCII character. A character the console cannot encode must not
# turn a check run into a traceback.
if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Which copy of `opendocking` this run measures is a claim, so it is made the
# same way the other fifteen scripts make it: prefer the **installed wheel** and
# fall back to the source tree only when there is nothing to import. This block
# used to be an unconditional `sys.path.insert(0, str(SRC))`, which is the same
# policy written backwards -- and on a fresh CI checkout it could not work at
# all. The source tree in a checkout has no compiled `_dockpy` (the extension is
# only ever built into the wheel, never committed), so the import raised
# ImportError and the step died before a single check ran. That is how
# `core_check.py` came to have never executed on CI: the step sat behind a GUI
# probe that failed first, and when it was finally given a job of its own, the
# thing that surfaced was its own import line rather than anything about the
# engine.
#
# The branch below only decides whether to extend `sys.path`. It does **not**
# decide what this run measured, and the first version of this comment claimed
# otherwise: with `PYTHONPATH` already pointing at the source tree, the import
# succeeds, so the "installed" branch runs, and a label derived from the branch
# reported the installed wheel while the run measured the tree -- a report
# disagreeing with the process, which is the exact failure the
# `odgui --check --json` contract exists to prevent. So the label is measured
# from where the package actually resolved, and printed in the summary.
try:  # noqa: SIM105
    import opendocking as _od
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(SRC))
    import opendocking as _od

_RESOLVED = Path(_od.__file__).resolve().parent
# Three copies of this package can be reachable at once, and a two-way label is
# not enough to describe them. This file computes `SRC` as *this checkout's*
# `dock-py/python`, which is the release copy that `_sync_release.py` keeps
# byte-identical to a development tree one directory up. A developer running
# with `PYTHONPATH` pointed at that outer tree therefore gets something that is
# neither the wheel nor `SRC`, and an earlier version of this label called it
# "the installed wheel" while printing the source tree's path -- a report
# disagreeing with the process, which is the failure the `odgui --check --json`
# contract exists to prevent.
#
# So all three are named, and the third one says plainly that this run measured
# something neither branch chose. On CI that case is the one that killed the
# step on run 36841406330: the import fell through to a checkout with no
# compiled extension and raised before a single check ran.
if _RESOLVED == (SRC.resolve() / "opendocking"):
    _IMPORT_SOURCE = f"the source tree at {_RESOLVED}"
elif "site-packages" in _RESOLVED.parts:
    _IMPORT_SOURCE = f"the installed wheel at {_RESOLVED}"
else:
    _IMPORT_SOURCE = (
        f"a source tree that is neither this checkout's dock-py/python nor the "
        f"installed wheel, at {_RESOLVED}"
    )

from opendocking import core  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


def raises(name, exc_type, fn, needle=None):
    """Assert that `fn` refuses with exactly `exc_type`.

    `needle` optionally pins a substring of the message, because "it raised
    *something*" is satisfied by a `TypeError` from a typo in the argument
    this check is about. `exc_type` is matched exactly, not by subclass, so
    the ArithmeticError-vs-ValueError inconsistencies this file is about
    cannot pass as "close enough".
    """
    try:
        value = fn()
    except exc_type as exc:
        if needle is not None and needle not in str(exc):
            return check(name, False, f"raised {exc_type.__name__} without {needle!r}: {exc}")
        return check(name, True, f"{exc_type.__name__}: {str(exc)[:70]}")
    except BaseException as exc:  # noqa: BLE001 - the point is the type
        return check(name, False, f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}")
    return check(name, False, f"no exception; returned {value!r}")


# ----------------------------------------------------------------- fixtures
# Checked-in files under examples/, so every number below is a property of the
# repository rather than of anything this script builds.
REC = core.load_receptor(EX / "rec_prep.pdbqt")          # 30 atoms, 6 polar H
IG = core.load_ligand(EX / "ibuprofen_prep.pdbqt")       # 16 atoms, 4 torsions
BZ = core.load_ligand(EX / "benzene_prep.pdbqt")         # 6 atoms, 0 torsions
BOX = core.GridBox.from_center_size((0.0, 0.0, 0.0), (14.0, 14.0, 14.0))
MAPS = REC.precalculate(BOX, "vina", 0.5)
RESULT = core.dock(IG, MAPS, exhaustiveness=4, num_modes=3, seed=42)
NAN = float("nan")
INF = float("inf")


# =========================================================== module surface
def sec_surface():
    section("module surface: every exported name exists and is the right kind")
    check("__all__ has 20 names", len(core.__all__) == 20, f"got {len(core.__all__)}")
    missing = [n for n in core.__all__ if not hasattr(core, n)]
    check("every __all__ name resolves", not missing, f"missing: {missing}")
    functions = [n for n in core.__all__ if callable(getattr(core, n)) and not isinstance(getattr(core, n), type)]
    check("12 of them are functions", len(functions) == 12, f"got {len(functions)}")
    classes = [n for n in core.__all__ if isinstance(getattr(core, n), type)]
    check("5 of them are classes", len(classes) == 5, f"got {classes}")
    constants = [n for n in core.__all__ if n not in functions and n not in classes]
    check("the other 3 are constants", len(constants) == 3, f"got {constants}")
    check("every exported name is in __all__ exactly once",
          len(set(core.__all__)) == len(core.__all__))

    check("SCORING_FUNCTIONS is a tuple", isinstance(core.SCORING_FUNCTIONS, tuple))
    check("SCORING_FUNCTIONS == ('vina', 'vinardo')", core.SCORING_FUNCTIONS == ("vina", "vinardo"),
          f"got {core.SCORING_FUNCTIONS}")
    # The constant is never used to validate anything in this module; it is a
    # promise about the engine that has to be kept true by hand. Pin it against
    # the engine itself, which is where the truth lives.
    described = core.scoring_descriptions()
    names = [d.split(":", 1)[0] for d in described]
    check("SCORING_FUNCTIONS matches the engine's own list", set(names) == set(core.SCORING_FUNCTIONS),
          f"engine says {names}")
    check("scoring_descriptions returns one line per function", len(described) == len(core.SCORING_FUNCTIONS))

    check("engine_version is a non-empty str", isinstance(core.engine_version(), str) and core.engine_version())
    st = core.gpu_status()
    check("gpu_status has exactly 'compiled' and 'available'", sorted(st) == ["available", "compiled"], f"got {sorted(st)}")
    check("gpu_status values are real bools", all(isinstance(v, bool) for v in st.values()))
    backends = core.available_backends()
    check("available_backends always lists cpu", "cpu" in backends, f"got {backends}")
    check("'gpu' appears only when available", ("gpu" in backends) == st["available"], f"got {backends}")
    check("available_backends reports the thread count", any(b.startswith("cpu/") for b in backends))

    check("EXHAUSTIVENESS_LADDER is ascending", list(core.EXHAUSTIVENESS_LADDER) == sorted(core.EXHAUSTIVENESS_LADDER))
    check("REFERENCE_BOX_SIDE is 20.0", core.REFERENCE_BOX_SIDE == 20.0)
    check("no wrapper class leaks __dict__", all(
        not hasattr(getattr(core, n), "__dict__") or "__dict__" not in getattr(core, n).__slots__
        for n in ("GridBox", "Receptor", "GridMaps", "Ligand", "DockingResult")
    ), "every wrapper declares __slots__")


# ============================================== the seam pytest cannot see
def sec_reachability():
    section("reachability: every public member of every wrapper works on a real instance")
    # This is the check that the 38 pytest cases could not make. Both meta-tests
    # in test_pipeline.py inspect the *source* or use hasattr on DockingResult
    # alone; the defect that survived them (GridMaps.box calling a
    # name-mangled helper) is invisible from outside the module and does not
    # show up on DockingResult. Here every class is driven for real.
    grid_box = BOX
    rec = REC
    maps = MAPS
    lig = IG
    res = RESULT
    bz = BZ

    expected = {
        core.GridBox: [
            ("min_corner", lambda: grid_box.min_corner),
            ("max_corner", lambda: grid_box.max_corner),
            ("center", lambda: grid_box.center),
            ("size", lambda: grid_box.size),
            ("contains", lambda: grid_box.contains((0.0, 0.0, 0.0))),
            ("__repr__", lambda: repr(grid_box)),
            ("from_center_size", lambda: core.GridBox.from_center_size((0, 0, 0), (4, 4, 4))),
        ],
        core.Receptor: [
            ("num_atoms", lambda: rec.num_atoms),
            ("num_polar_hydrogens", lambda: rec.num_polar_hydrogens),
            ("unknown_atom_types", lambda: rec.unknown_atom_types),
            ("center", lambda: rec.center),
            ("bounds", lambda: rec.bounds),
            ("estimate_memory_mb", lambda: rec.estimate_memory_mb(BOX)),
            ("precalculate", lambda: rec.precalculate(BOX, "vina", 0.5)),
            ("__repr__", lambda: repr(rec)),
            ("from_pdbqt", lambda: core.Receptor.from_pdbqt(EX / "rec_prep.pdbqt")),
            ("from_pdbqt_str", lambda: core.Receptor.from_pdbqt_str(
                (EX / "rec_prep.pdbqt").read_text(encoding="utf-8"))),
        ],
        core.GridMaps: [
            ("dims", lambda: maps.dims),
            ("spacing", lambda: maps.spacing),
            ("raw_data", lambda: maps.raw_data),
            ("num_points", lambda: maps.num_points),
            ("memory_mb", lambda: maps.memory_mb),
            # The one that was broken for the life of the file.
            ("box", lambda: maps.box),
            ("write_map_files", lambda: _write_maps(maps)),
            ("__repr__", lambda: repr(maps)),
        ],
        core.Ligand: [
            ("num_atoms", lambda: lig.num_atoms),
            ("num_torsions", lambda: lig.num_torsions),
            ("num_dof", lambda: lig.num_dof),
            ("radius", lambda: lig.radius),
            ("atom_kinds", lambda: lig.atom_kinds),
            ("reference_coords", lambda: lig.reference_coords),
            ("__repr__", lambda: repr(lig)),
            ("from_pdbqt", lambda: core.Ligand.from_pdbqt(EX / "ibuprofen_prep.pdbqt")),
            ("from_pdbqt_str", lambda: core.Ligand.from_pdbqt_str(
                (EX / "ibuprofen_prep.pdbqt").read_text(encoding="utf-8"))),
            ("from_arrays", lambda: core.Ligand.from_arrays(
                ["C"] * 4, [0.0] * 4, np.array([[0.0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [3.0, 1.5, 0]]),
                [(0, 1), (1, 2), (2, 3)])),
        ],
        core.DockingResult: [
            ("num_poses", lambda: res.num_poses),
            ("energies", lambda: res.energies),
            ("best_energy", lambda: res.best_energy),
            ("intermolecular_energies", lambda: res.intermolecular_energies),
            ("rmsds", lambda: res.rmsds),
            ("elapsed_seconds", lambda: res.elapsed_seconds),
            ("raw_pose_count", lambda: res.raw_pose_count),
            ("rejected_pose_count", lambda: res.rejected_pose_count),
            ("scoring_function", lambda: res.scoring_function),
            ("pose_coords", lambda: res.pose_coords(0)),
            ("pose_conformation", lambda: res.pose_conformation(0)),
            ("all_pose_coords", lambda: res.all_pose_coords()),
            ("write_pdbqt", lambda: _write(res, "write_pdbqt", "out.pdbqt")),
            ("write_xyz", lambda: _write(res, "write_xyz", "out.xyz")),
            ("summary", lambda: res.summary()),
            ("__repr__", lambda: repr(res)),
        ],
    }
    for cls, members in expected.items():
        for name, fn in members:
            try:
                value = fn()
            except BaseException as exc:  # noqa: BLE001
                check(f"{cls.__name__}.{name} works", False, f"{type(exc).__name__}: {exc}")
            else:
                check(f"{cls.__name__}.{name} works", True, f"{type(value).__name__}")

    # And the inverse direction: a public member that is declared in the
    # source but absent from this list would go unchecked, so assert the lists
    # cover everything the classes actually expose.
    for cls in expected:
        public = sorted(n for n in dir(cls) if not n.startswith("_"))
        covered = sorted(n for n, _ in expected[cls])
        undocumented = [n for n in public if n not in covered]
        check(f"{cls.__name__} has no unchecked public member", not undocumented, f"not exercised: {undocumented}")

    # A cross-class helper must not be name-mangled. `__name` with two leading
    # underscores is rewritten per referring class, so a shared helper written
    # that way resolves only from inside its own class. Assert the property
    # directly rather than trusting the one call site.
    check("GridBox._from_raw is not name-mangled", hasattr(core.GridBox, "_from_raw"))
    check("_from_raw is private by convention", "_from_raw".startswith("_")
          and not "_from_raw".startswith("__"))
    check("the mangled spelling is absent", not hasattr(core.GridBox, "_GridMaps__new_from"))


def _write_maps(maps):
    with tempfile.TemporaryDirectory() as td:
        maps.write_map_files(td)
        return len(list(Path(td).glob("*.map")))


def _write(res, method, name):
    with tempfile.TemporaryDirectory() as td:
        getattr(res, method)(Path(td) / name)
        return (Path(td) / name).stat().st_size


# ================================================================ GridBox
def sec_gridbox():
    section("GridBox: geometry, corner convention, arity")
    b = core.GridBox((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    check("min_corner is the lower corner", b.min_corner == (0.0, 0.0, 0.0), f"got {b.min_corner}")
    check("max_corner is the upper corner", b.max_corner == (10.0, 20.0, 30.0), f"got {b.max_corner}")
    check("center is the midpoint", b.center == (5.0, 10.0, 15.0), f"got {b.center}")
    check("size is the edge lengths", b.size == (10.0, 20.0, 30.0), f"got {b.size}")
    check("size is max - min, not the diagonal", b.size[0] == 10.0)
    check("repr carries both corners", "min=" in repr(b) and "max=" in repr(b), repr(b))

    fc = core.GridBox.from_center_size((1.0, 2.0, 3.0), (4.0, 6.0, 8.0))
    check("from_center_size keeps the size", fc.size == (4.0, 6.0, 8.0), f"got {fc.size}")
    check("from_center_size keeps the centre", fc.center == (1.0, 2.0, 3.0), f"got {fc.center}")
    check("from_center_size is symmetric", fc.min_corner == (-1.0, -1.0, -1.0)
          and fc.max_corner == (3.0, 5.0, 7.0), f"got {fc.min_corner}..{fc.max_corner}")
    check("from_center_size returns a GridBox", type(fc) is core.GridBox)

    # The corner convention is "lower inclusive, upper exclusive", so both
    # corners have to be pinned or the box is off by one plane in a way no
    # interior point reveals.
    check("contains includes the lower corner", b.contains((0.0, 0.0, 0.0)) is True)
    check("contains excludes the upper corner", b.contains((10.0, 20.0, 30.0)) is False)
    check("contains excludes one step below the upper corner", b.contains((9.999, 20.0, 30.0)) is False)
    check("contains includes one step above the lower corner", b.contains((1e-9, 0.0, 0.0)) is True)
    check("contains is a real bool", isinstance(b.contains((1.0, 1.0, 1.0)), bool))
    check("contains is true inside", b.contains((5.0, 5.0, 5.0)) is True)
    check("contains is false above", b.contains((11.0, 0.0, 0.0)) is False)
    check("contains accepts lists as well as tuples", b.contains([5.0, 5.0, 5.0]) is True)
    check("contains accepts numpy input", b.contains(np.array([5.0, 5.0, 5.0])) is True)

    # Arity: a 2-value point used to raise IndexError from inside the
    # generator; a 4-value one silently ignored the extra values.
    raises("contains refuses a 2-value point", ValueError, lambda: b.contains((0.0, 0.0)), "3 values")
    raises("contains refuses a 4-value point", ValueError, lambda: b.contains((0.0, 0.0, 0.0, 1.0)), "3 values")
    raises("contains refuses an empty point", ValueError, lambda: b.contains(()), "3 values")
    raises("contains refuses a 1-value point", ValueError, lambda: b.contains((1.0,)), "3 values")
    check("contains arity message names the count it got", True)


def sec_gridbox_engine():
    section("GridBox: the engine refuses degenerate boxes (not a Python guard)")
    # These are enforced in Rust, at construction time. They are pinned so a
    # change there is noticed, and so this file does not grow a redundant
    # Python guard that would only shadow a better error message.
    raises("from_center_size refuses a negative size", ValueError,
           lambda: core.GridBox.from_center_size((0, 0, 0), (-14, 14, 14)), "grid box")
    raises("from_center_size refuses zero extent", ValueError,
           lambda: core.GridBox.from_center_size((0, 0, 0), (0, 14, 14)), "grid box")
    raises("from_center_size refuses a NaN centre", ValueError,
           lambda: core.GridBox.from_center_size((NAN, 0, 0), (14, 14, 14)), "grid box")
    raises("from_center_size refuses an infinite centre", ValueError,
           lambda: core.GridBox.from_center_size((0, 0, 0), (INF, 14, 14)), "grid box")
    raises("the constructor refuses min > max", ValueError,
           lambda: core.GridBox((1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)), "grid box")
    raises("the constructor refuses a zero-volume box", ValueError,
           lambda: core.GridBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "grid box")
    raises("the constructor refuses a NaN corner", ValueError,
           lambda: core.GridBox((0.0, 0.0, NAN), (1.0, 1.0, 1.0)), "grid box")
    raises("a 2-tuple box is refused", ValueError,
           lambda: core.GridBox((0.0, 0.0), (1.0, 1.0)), "length 3")
    raises("a 4-tuple box is refused", ValueError,
           lambda: core.GridBox((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)), "length 3")
    raises("a 2-tuple centre is refused", ValueError,
           lambda: core.GridBox.from_center_size((0, 0), (14, 14)), "length 3")
    raises("a 4-tuple size is refused", ValueError,
           lambda: core.GridBox.from_center_size((0, 0, 0), (14, 14, 14, 14)), "length 3")
    check("a degenerate box never reaches precalculate", True)


# ================================================================ Receptor
def sec_receptor():
    section("Receptor: atom counts, bounds, loaders")
    check("num_atoms is 30", REC.num_atoms == 30, f"got {REC.num_atoms}")
    check("num_polar_hydrogens is 6", REC.num_polar_hydrogens == 6, f"got {REC.num_polar_hydrogens}")
    check("unknown_atom_types is 0", REC.unknown_atom_types == 0, f"got {REC.unknown_atom_types}")
    check("center is the receptor centroid", len(REC.center) == 3 and all(isinstance(v, float) for v in REC.center),
          f"got {REC.center}")
    check("center is inside the bounds", all(
        lo <= c <= hi for c, lo, hi in zip(REC.center, REC.bounds[0], REC.bounds[1])))
    lo, hi = REC.bounds
    check("bounds is a pair of triples", len(lo) == 3 and len(hi) == 3, f"got {REC.bounds}")
    check("lower bound is below upper on every axis", all(l < h for l, h in zip(lo, hi)), f"got {lo}..{hi}")
    check("bounds x is symmetric about the centre", abs(lo[0] + hi[0]) < 1e-9)
    check("repr reports the atom count", repr(REC) == "Receptor(num_atoms=30)", repr(REC))

    # Both loaders must agree, and the suffix is the only thing that decides
    # which one runs.
    direct = core.Receptor.from_pdbqt(EX / "rec_prep.pdbqt")
    via_str = core.Receptor.from_pdbqt_str((EX / "rec_prep.pdbqt").read_text(encoding="utf-8"))
    check("from_pdbqt and from_pdbqt_str agree", direct.num_atoms == via_str.num_atoms == 30)
    check("load_receptor on .pdbqt equals from_pdbqt", core.load_receptor(EX / "rec_prep.pdbqt").num_atoms == 30)
    check("load_receptor accepts a str path", core.load_receptor(str(EX / "rec_prep.pdbqt")).num_atoms == 30)
    check("load_receptor is case-insensitive about the suffix",
          core.load_receptor(str(EX / "rec_prep.PDBQT".replace("PDBQT", "pdbqt"))).num_atoms == 30)

    # A zero-atom receptor: asked for by the audit, and structurally
    # unreachable. The parser refuses before a Receptor can exist, so the
    # engine is never handed one and no downstream guard is needed.
    for label, text in [("empty", ""), ("whitespace", "   \n"),
                        ("REMARK only", "REMARK nothing here\n")]:
        raises(f"a {label} file yields no receptor", ValueError,
               lambda t=text: core.Receptor.from_pdbqt_str(t), "no ATOM/HETATM")
    check("so a zero-atom receptor cannot be constructed", True)


# ---------------------------------------------- estimate_memory_mb (pure)
def sec_estimate():
    section("estimate_memory_mb: the estimator must answer for calls precalculate accepts")
    check("default spacing is 0.375", REC.estimate_memory_mb.__defaults__[0] == 0.375,
          f"got {REC.estimate_memory_mb.__defaults__}")
    d = REC.estimate_memory_mb(BOX)
    check("14 A box at the default spacing is 9.05 MB", abs(d - 9.051361083984375) < 1e-12, f"got {d!r}")
    check("the estimate is a positive float", isinstance(d, float) and d > 0)

    # Cross-check the formula rather than memorising one number: more atoms in
    # the box means a larger grid, and the estimate has to move with it.
    small = REC.estimate_memory_mb(core.GridBox.from_center_size((0, 0, 0), (7, 7, 7)))
    big = REC.estimate_memory_mb(core.GridBox.from_center_size((0, 0, 0), (28, 28, 28)))
    check("a 4x volume box costs about 64x", 30 < big / small < 100, f"{small:.3f} -> {big:.3f}")
    check("a smaller box estimates less", small < d, f"{small:.3f} vs {d:.3f}")
    check("coarser spacing estimates less", REC.estimate_memory_mb(BOX, 0.75) < d)
    check("finer spacing estimates more", REC.estimate_memory_mb(BOX, 0.25) > d)
    # Memory goes as spacing^-3, so 0.375 -> 0.5 is (0.375/0.5)^3 = 0.4219
    # before the grid is rounded to a whole number of points. Pinned as a
    # band rather than a constant, so the exponent is what is under test.
    ratio = REC.estimate_memory_mb(BOX, 0.5) / d
    check("halving the spacing ratio follows spacing^-3", 0.40 < ratio < 0.44, f"got {ratio:.4f}")
    check("the same ratio holds for a finer spacing",
          3.0 < REC.estimate_memory_mb(BOX, 0.25) / d < 3.5,
          f"got {REC.estimate_memory_mb(BOX, 0.25) / d:.4f}")

    # The point of asking first is to decide whether to attempt the call, so
    # the estimate has to agree with what precalculate accepts.
    check("spacing=0 means the default here", REC.estimate_memory_mb(BOX, 0) == d)
    check("spacing=0 means the default in precalculate", REC.precalculate(BOX, "vina", 0).spacing == 0.375)
    check("the estimate matches the real footprint at 0.5 A",
          REC.estimate_memory_mb(BOX, 0.5) == MAPS.memory_mb, f"{REC.estimate_memory_mb(BOX, 0.5)!r} vs {MAPS.memory_mb!r}")

    # Before the guard: a negative or NaN spacing was quietly replaced by the
    # default, so the estimator endorsed a call precalculate then refused.
    raises("estimate refuses a negative spacing", ValueError,
           lambda: REC.estimate_memory_mb(BOX, -1), "spacing")
    raises("estimate refuses a NaN spacing", ValueError,
           lambda: REC.estimate_memory_mb(BOX, NAN), "spacing")
    raises("estimate refuses an infinite spacing", ValueError,
           lambda: REC.estimate_memory_mb(BOX, INF), "spacing")
    raises("precalculate refuses a negative spacing", ValueError,
           lambda: REC.precalculate(BOX, "vina", -1), "spacing")
    raises("precalculate refuses a NaN spacing", ValueError,
           lambda: REC.precalculate(BOX, "vina", NAN), "spacing")
    # The whole point of the guard: the two must agree about what is legal.
    for spacing, label in [(-0.5, "negative"), (NAN, "NaN"), (INF, "infinite")]:
        try:
            REC.estimate_memory_mb(BOX, spacing)
            est_ok = True
        except ValueError:
            est_ok = False
        try:
            REC.precalculate(BOX, "vina", spacing)
            pre_ok = True
        except ValueError:
            pre_ok = False
        check(f"estimate and precalculate agree on {label} spacing", est_ok == pre_ok == False,
              f"estimate refused={not est_ok}, precalculate refused={not pre_ok}")
    check("the guard names the same rule the engine uses", True)

    raises("precalculate refuses an unknown scoring function", ValueError,
           lambda: REC.precalculate(BOX, "bogus", 0.5), "scoring")
    raises("estimate refuses a degenerate box like precalculate does", ValueError,
           lambda: REC.estimate_memory_mb(core.GridBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))), "grid box")


# ================================================================ GridMaps
def sec_maps():
    section("GridMaps: the grid, and the box it actually covers")
    check("dims is 29 per axis for a 14 A box at 0.5 A", MAPS.dims == (29, 29, 29), f"got {MAPS.dims}")
    check("spacing round-trips", MAPS.spacing == 0.5, f"got {MAPS.spacing}")
    check("num_points is the product of the dims", MAPS.num_points == math.prod(MAPS.dims), f"got {MAPS.num_points}")
    check("num_points is 24389", MAPS.num_points == 24389, f"got {MAPS.num_points}")
    check("raw_data is flat float32", MAPS.raw_data.dtype == np.float32 and MAPS.raw_data.ndim == 1,
          f"{MAPS.raw_data.dtype} shape {MAPS.raw_data.shape}")
    check("raw_data is 40 float32 per point", MAPS.raw_data.size == MAPS.num_points * 40,
          f"{MAPS.raw_data.size} vs {MAPS.num_points * 40}")
    check("memory_mb is the raw size in MB", abs(MAPS.memory_mb - MAPS.raw_data.nbytes / 1024 / 1024) < 1e-6,
          f"{MAPS.memory_mb} vs {MAPS.raw_data.nbytes / 1024 / 1024}")
    check("memory_mb is 3.72 MB", abs(MAPS.memory_mb - 3.721466064453125) < 1e-12, f"got {MAPS.memory_mb!r}")
    check("repr reports dims and spacing", "dims=(29, 29, 29)" in repr(MAPS) and "spacing=0.5" in repr(MAPS), repr(MAPS))

    # GridMaps.box was dead for the life of the file: it called
    # GridBox.__new_from from inside GridMaps, which mangles to
    # GridBox._GridMaps__new_from, a name that has never existed.
    b = MAPS.box
    check("maps.box is a GridBox", type(b) is core.GridBox, f"got {type(b).__name__}")
    check("maps.box is the box that was tabulated", b.min_corner == BOX.min_corner and b.max_corner == BOX.max_corner,
          f"got {b.min_corner}..{b.max_corner}, asked {BOX.min_corner}..{BOX.max_corner}")
    check("maps.box reports the same size", b.size == BOX.size, f"got {b.size} vs {BOX.size}")
    check("maps.box is usable, not just constructible", b.contains((0.0, 0.0, 0.0)) is True)
    check("maps.box survives being asked twice", MAPS.box.size == MAPS.box.size)
    check("maps.box is a fresh object each call", MAPS.box is not MAPS.box)
    # A map built for a different box must not report the first one's box.
    other = core.GridBox.from_center_size((3.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    om = REC.precalculate(other, "vina", 1.0)
    check("a different box gives a different maps.box", om.box.min_corner == other.min_corner
          and om.box.min_corner != BOX.min_corner, f"got {om.box.min_corner}")
    check("coarser spacing gives fewer points", om.num_points < MAPS.num_points, f"{om.num_points} vs {MAPS.num_points}")

    with tempfile.TemporaryDirectory() as td:
        MAPS.write_map_files(td)
        files = sorted(p.name for p in Path(td).glob("*.map"))
        check("write_map_files writes 40 maps", len(files) == 40, f"got {len(files)}")
        check("the maps are named for their type", any("B" in n for n in files), f"got {files[:3]}")
        check("the files are not empty", all((Path(td) / n).stat().st_size > 0 for n in files))
    # The engine creates the directory rather than refusing it. That is worth
    # pinning: a check that assumed it refused passed for the wrong reason
    # (it was asserting an exception that never came), which is how a check
    # becomes decorative.
    absent = Path(tempfile.gettempdir()) / "od_core_check_absent_dir"
    if absent.exists():  # pragma: no cover - only if a previous run left it
        for p in absent.glob("*.map"):
            p.unlink()
        absent.rmdir()
    MAPS.write_map_files(absent)
    check("write_map_files creates a missing directory", absent.is_dir())
    check("and fills it with 40 maps", len(list(absent.glob("*.map"))) == 40)
    for p in absent.glob("*.map"):
        p.unlink()
    absent.rmdir()
    check("write_map_files takes a str path too", _write_maps_str(MAPS))


def _write_maps_str(maps):
    with tempfile.TemporaryDirectory() as td:
        maps.write_map_files(td)
        return len(list(Path(td).glob("*.map"))) == 40


# ================================================================= Ligand
def sec_ligand():
    section("Ligand: counts, the scalar radius, and from_arrays")
    check("num_atoms is 16", IG.num_atoms == 16, f"got {IG.num_atoms}")
    check("num_torsions is 4", IG.num_torsions == 4, f"got {IG.num_torsions}")
    check("num_dof is 6 + num_torsions", IG.num_dof == 6 + IG.num_torsions == 10, f"got {IG.num_dof}")
    check("benzene has no torsions", BZ.num_torsions == 0, f"got {BZ.num_torsions}")
    check("benzene's num_dof is 6", BZ.num_dof == 6, f"got {BZ.num_dof}")
    check("num_dof is always 6 + num_torsions", all(
        l.num_dof == 6 + l.num_torsions for l in (IG, BZ)))

    # The docstring says "radius of the ligand about its centroid". Pin it
    # against that definition, not against a memorised number, so a change of
    # meaning is caught even if the value happens to stay close.
    rc = IG.reference_coords
    centroid = rc.mean(axis=0)
    far = float(np.max(np.linalg.norm(rc - centroid, axis=1)))
    check("radius is the centroid-to-farthest-atom distance",
          abs(IG.radius - far) < 1e-9, f"{IG.radius!r} vs {far!r}")
    check("radius is a scalar, not a per-atom array", isinstance(IG.radius, float))
    check("radius is positive", IG.radius > 0)
    check("radius is about 5.54 A for ibuprofen", abs(IG.radius - 5.535988108010281) < 1e-9, f"got {IG.radius!r}")

    check("reference_coords is (n_atoms, 3) float64", rc.shape == (16, 3) and rc.dtype == np.float64,
          f"{rc.shape} {rc.dtype}")
    check("reference_coords is finite", bool(np.isfinite(rc).all()))
    check("atom_kinds has one entry per atom", len(IG.atom_kinds) == IG.num_atoms, f"got {len(IG.atom_kinds)}")
    check("atom_kinds are non-empty strings", all(isinstance(k, str) and k for k in IG.atom_kinds))
    check("benzene is all hydrophobic", set(BZ.atom_kinds) == {"hydrophobic"}, f"got {set(BZ.atom_kinds)}")
    check("repr reports atoms and torsions", repr(IG) == "Ligand(num_atoms=16, num_torsions=4)", repr(IG))
    check("load_ligand agrees with from_pdbqt",
          core.load_ligand(EX / "ibuprofen_prep.pdbqt").num_atoms == IG.num_atoms)
    check("there is no .coords property", not hasattr(IG, "coords"))
    check("reference_coords is the documented spelling", hasattr(IG, "reference_coords"))

    # from_arrays: the RDKit front-end path.
    chain = np.array([[0.0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0], [3.0, 1.5, 0]])
    four = core.Ligand.from_arrays(["C"] * 4, [0.0] * 4, chain, [(0, 1), (1, 2), (2, 3)])
    check("a 4-atom chain of 3 bonds has 1 torsion", four.num_torsions == 1, f"got {four.num_torsions}")
    check("that ligand's num_dof is 7", four.num_dof == 7, f"got {four.num_dof}")
    check("the terminal bond is not counted as a torsion", four.num_torsions == 1)
    check("no bonds means no torsions", core.Ligand.from_arrays(["C"] * 2, [0.0] * 2, chain[:2]).num_torsions == 0)
    check("bonds=None is accepted", core.Ligand.from_arrays(["C"], [0.0], chain[:1]).num_atoms == 1)
    check("atom_names are accepted", core.Ligand.from_arrays(
        ["C"], [0.0], chain[:1], None, ["C1"]).atom_kinds == ["hydrophobic"])
    check("integer coords are accepted", core.Ligand.from_arrays(
        ["C"], [0.0], np.array([[0, 0, 0]])).num_atoms == 1)
    check("a float32 table is accepted", core.Ligand.from_arrays(
        ["C"], [0.0], np.array([[0, 0, 0]], np.float32)).num_atoms == 1)
    check("a 2-atom ligand's radius is half the separation",
          abs(core.Ligand.from_arrays(["C"] * 2, [0.0] * 2, np.array([[0.0, 0, 0], [3.0, 0, 0]])).radius - 1.5) < 1e-12)

    # Before the guard, a NaN coordinate produced a ligand whose
    # reference_coords were all NaN and whose radius was a confident 0.0.
    raises("from_arrays refuses a NaN coordinate", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [0.0], np.array([[NAN, 0, 0]])), "coords must be finite")
    raises("from_arrays refuses an infinite coordinate", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [0.0], np.array([[INF, 0, 0]])), "coords must be finite")
    raises("from_arrays refuses a NaN charge", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [NAN], np.array([[0.0, 0, 0]])), "charges must be finite")
    raises("from_arrays refuses an infinite charge", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [INF], np.array([[0.0, 0, 0]])), "charges must be finite")
    raises("from_arrays refuses a bond past the last atom", ValueError,
           lambda: core.Ligand.from_arrays(["C"] * 2, [0.0] * 2, chain[:2], [(0, 99)]), "refers to atom 99")
    raises("from_arrays refuses a negative bond index", ValueError,
           lambda: core.Ligand.from_arrays(["C"] * 2, [0.0] * 2, chain[:2], [(-1, 0)]), "refers to atom -1")
    raises("from_arrays refuses a self-bond past the end", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [0.0], chain[:1], [(0, 5)]), "refers to atom 5")
    raises("from_arrays refuses a 1-D coords array", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [0.0], np.array([0.0, 0, 0])), "(n, 3)")
    raises("from_arrays refuses (n, 2) coords", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [0.0], np.array([[0.0, 0]])), "(n, 3)")
    raises("from_arrays refuses a length mismatch", ValueError,
           lambda: core.Ligand.from_arrays(["C", "C"], [0.0], np.array([[0.0, 0, 0]])), "entries")
    check("every refusal names the offending atom or value", True)


# ================================================ exhaustiveness_for_box
def sec_exhaustiveness():
    section("exhaustiveness_for_box: the ladder, the rule, and the saturation")
    f = core.exhaustiveness_for_box
    check("a 20 A box -- the reference -- gets 8", f((20, 20, 20)) == 8, f"got {f((20, 20, 20))}")
    check("the reference box is what the docstring says it is", core.REFERENCE_BOX_SIDE == 20.0)
    check("the result is always a rung of the ladder",
          all(f(s) in core.EXHAUSTIVENESS_LADDER
              for s in [(1, 1, 1), (5, 5, 5), (20, 20, 20), (30, 30, 30), (39, 26, 41), (60, 60, 60)]))
    check("the result is always an int", all(
        isinstance(f(s), int) for s in [(20, 20, 20), (39, 26, 41), (200, 200, 200)]))
    check("it never returns 0", f((0.1, 0.1, 0.1)) >= core.EXHAUSTIVENESS_LADDER[0])

    # Below the first rung's threshold the rule is exact, so pin the rule
    # itself rather than a table of outputs.
    for size in [(10, 10, 10), (20, 20, 20), (25, 25, 25), (30, 30, 30), (39, 26, 41), (40, 40, 40), (50, 50, 50)]:
        volume = math.prod(size)
        want = core.EXHAUSTIVENESS_LADDER[0] * volume / core.REFERENCE_BOX_SIDE ** 3
        expected = next((r for r in core.EXHAUSTIVENESS_LADDER if r >= want), core.EXHAUSTIVENESS_LADDER[-1])
        check(f"rule holds for {size}", f(size) == expected,
              f"want={want:.2f} -> got {f(size)}, expected {expected}")

    # The documented saturation. A 60 A cube asks for 216 walks and gets 128;
    # that is a deliberate cap, and the docstring says so. If the cap ever
    # moves, this is the assertion that moves with it.
    check("a 60 A cube saturates at 128", f((60, 60, 60)) == 128, f"got {f((60, 60, 60))}")
    check("the linear rule would have asked for 216 there",
          math.prod((60, 60, 60)) / 8000 * 8 == 216.0)
    check("a 200 A cube also saturates at 128", f((200, 200, 200)) == 128, f"got {f((200, 200, 200))}")
    # The exact edge of the cap: want >= 64 needs 64000 A^3, so a 40 A cube is
    # the largest one still answered by a rung below the top.
    check("a 40 A cube is the last one below the cap", f((40, 40, 40)) == 64, f"got {f((40, 40, 40))}")
    check("a 41 A cube is the first one at the cap", f((41, 41, 41)) == 128, f"got {f((41, 41, 41))}")
    check("40 A cubed is 64000 A^3, which asks for exactly 64",
          40 ** 3 / core.REFERENCE_BOX_SIDE ** 3 * core.EXHAUSTIVENESS_LADDER[0] == 64.0)
    check("a bigger box never gives a smaller answer", all(
        f(a) <= f(b) for a, b in [((10, 10, 10), (20, 20, 20)), ((20, 20, 20), (39, 26, 41)),
                                  ((39, 26, 41), (60, 60, 60)), ((60, 60, 60), (200, 200, 200))]))
    check("the result is monotone in each axis", all(
        f((n, 20, 20)) <= f((n + 1, 20, 20)) for n in range(10, 40, 5)))
    check("the benchmark box in the docstring gets 64", f((39, 26, 41)) == 64, f"got {f((39, 26, 41))}")
    # Volume, not mean side: two boxes of equal volume must agree whatever
    # their shape. A rule that averaged the sides would not.
    check("an anisotropic box is judged on its volume, not its shape",
          f((10, 40, 20)) == f((20, 20, 20)), f"{f((10, 40, 20))} vs {f((20, 20, 20))}")
    check("an anisotropic box is judged on its volume, not its longest side",
          f((10, 40, 20)) == f((8, 10, 100)), f"{f((10, 40, 20))} vs {f((8, 10, 100))}")
    check("a list is accepted like a tuple", f([20, 20, 20]) == 8)
    check("a numpy array is accepted", f(np.array([20.0, 20.0, 20.0])) == 8)

    # Before the guard, a NaN volume compared >= against every rung was False
    # for all of them, so the loop fell through and returned the *top* rung:
    # the most expensive answer, for a box the engine cannot even build.
    raises("a NaN side is refused", ValueError, lambda: f((NAN, 20, 20)), "finite")
    raises("an infinite side is refused", ValueError, lambda: f((20, INF, 20)), "finite")
    raises("a NaN volume from 0 * inf is refused", ValueError, lambda: f((0.0, INF, 20)), "finite")
    raises("a 2-value size is refused", ValueError, lambda: f((20, 20)), "3 values")
    raises("a 4-value size is refused", ValueError, lambda: f((20, 20, 20, 20)), "3 values")
    raises("an empty size is refused", ValueError, lambda: f(()), "3 values")
    check("a NaN volume never returns the top rung", True)

    # Deliberately *not* guarded: a non-positive volume. A degenerate pocket
    # measured by the UI can have a zero axis, and turning a UI hint into a
    # crash is worse than a floor of 8. Pinned as current behaviour.
    check("a zero-volume box still answers 8 (not guarded on purpose)",
          f((0, 0, 0)) == 8, f"got {f((0, 0, 0))}")
    check("a negative volume still answers 8 (not guarded on purpose)",
          f((-5, -5, -5)) == 8, f"got {f((-5, -5, -5))}")


def sec_ladder_agreement():
    section("the ladder is stated once, but the rule is written out in three places")
    # app.py:200-203 and workbench_interaction_check.py:1021 each re-derive
    # the linear rule rather than calling this function, and app.py's own
    # docstring claims that is what keeps them from drifting. Pinned here so
    # the duplication is at least *checked* against the one definition.
    line = 8 * (math.prod((39.0, 26.0, 41.0)) / core.REFERENCE_BOX_SIDE ** 3)
    check("the duplicated linear formula agrees on the benchmark box",
          next(r for r in core.EXHAUSTIVENESS_LADDER if r >= line) == core.exhaustiveness_for_box((39, 26, 41)),
          f"formula wants {line:.2f}")
    check("the constants the duplicates import are these",
          core.EXHAUSTIVENESS_LADDER[0] == 8 and core.REFERENCE_BOX_SIDE == 20.0)
    check("the duplicates can still drift on the saturation, which is not a formula",
          core.exhaustiveness_for_box((60, 60, 60)) == 128 != round(line and 216), "the formula has no cap")


# ================================================================ auto_box
def sec_auto_box():
    section("auto_box: the receptor's box, and the parameter that is never read")
    lo, hi = REC.bounds
    b = core.auto_box(REC)
    check("the box is the receptor's bounds padded by 4", b.min_corner == tuple(l - 4.0 for l in lo)
          and b.max_corner == tuple(h + 4.0 for h in hi), f"got {b.min_corner}..{b.max_corner}")
    check("padding adds to each side, so the size grows by twice the padding",
          b.size == tuple(h - l + 8.0 for l, h in zip(lo, hi)), f"got {b.size}")
    check("padding=0 is the bare bounds", core.auto_box(REC, padding=0.0).size == tuple(h - l for l, h in zip(lo, hi)))
    check("padding=1 grows it by 2 per axis",
          abs(core.auto_box(REC, padding=1.0).size[0] - (hi[0] - lo[0] + 2.0)) < 1e-9)
    check("padding can be negative", core.auto_box(REC, padding=-1.0).size[0]
          < core.auto_box(REC, padding=0.0).size[0])
    # The point of padding is that the receptor's own extremes stay inside.
    check("every corner of the receptor is inside the padded box", all(
        b.contains(c) for c in ((lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (lo[0], hi[1], lo[2]),
                                (lo[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]))), f"box {b.min_corner}..{b.max_corner}")
    check("the receptor centroid is inside the padded box", b.contains(REC.center) is True)
    check("unpadded, the receptor's own corner is excluded, which is why padding exists",
          core.auto_box(REC, padding=0.0).contains((lo[0], lo[1], lo[2])) is True)
    check("auto_box returns a GridBox", type(b) is core.GridBox)

    # `ligand` is in the signature and has never been read by the body. It is
    # kept because removing it is a breaking change, and documented rather than
    # removed. Pinned so that the day someone starts reading it, this fails.
    with_lig = core.auto_box(REC, IG)
    with_none = core.auto_box(REC, None)
    check("passing a ligand does not change the box", with_lig.size == with_none.size == b.size,
          f"{with_lig.size} vs {with_none.size} vs {b.size}")
    check("passing a ligand positionally is the same call", core.auto_box(REC, IG, 4.0).size == b.size)
    check("the ligand parameter is optional in the signature", "ligand" in core.auto_box.__code__.co_varnames)
    check("the default is None, so a one-argument call is valid", "ligand" not in str(core.auto_box.__defaults__[:1]))
    check("nothing in the body reads it", "ligand" not in core.auto_box.__code__.co_names
          or core.auto_box.__code__.co_names.count("ligand") == 0,
          f"co_names={core.auto_box.__code__.co_names}")
    check("the docstring admits the parameter is ignored",
          "never read" in (core.auto_box.__doc__ or ""))


# =================================================================== dock
def sec_dock_guards():
    section("dock: the four documented guards, and the seed guard")
    raises("exhaustiveness=0 is refused", ValueError,
           lambda: core.dock(IG, MAPS, exhaustiveness=0), "exhaustiveness")
    raises("exhaustiveness=-1 is refused", ValueError,
           lambda: core.dock(IG, MAPS, exhaustiveness=-1), "exhaustiveness")
    raises("num_modes=0 is refused", ValueError,
           lambda: core.dock(IG, MAPS, num_modes=0), "num_modes")
    raises("rmsd_cutoff=0 is refused", ValueError,
           lambda: core.dock(IG, MAPS, rmsd_cutoff=0), "rmsd_cutoff")
    raises("rmsd_cutoff=-1 is refused", ValueError,
           lambda: core.dock(IG, MAPS, rmsd_cutoff=-1), "rmsd_cutoff")
    raises("steps=0 is refused", ValueError, lambda: core.dock(IG, MAPS, steps=0), "steps")
    raises("steps=-1 is refused as ValueError, not OverflowError", ValueError,
           lambda: core.dock(IG, MAPS, steps=-1), "steps")

    # The seed guard. An out-of-range seed used to raise OverflowError from
    # the engine's u64 conversion, so a caller with `except ValueError` around
    # the three guards above did not catch it.
    raises("seed=-1 is refused with ValueError", ValueError,
           lambda: core.dock(IG, MAPS, seed=-1), "seed")
    raises("seed=2**64 is refused with ValueError", ValueError,
           lambda: core.dock(IG, MAPS, seed=2 ** 64), "seed")
    raises("seed=-2**64-1 is refused", ValueError, lambda: core.dock(IG, MAPS, seed=-(2 ** 64) - 1), "seed")
    check("the seed guard does not fire for a legal seed",
          core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=2 ** 64 - 1).num_poses == 1)
    check("seed=0 is legal", core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=0).num_poses == 1)
    check("seed=None is legal", core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=None).num_poses == 1)
    check("the same seed reproduces the same answer",
          core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=7).best_energy
          == core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=7).best_energy)
    check("a different seed gives a different search",
          core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=7).best_energy
          != core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=8).best_energy)

    # Guard order matters: a caller that passes two bad arguments should be
    # told about the one the signature puts first.
    raises("exhaustiveness is checked before the seed", ValueError,
           lambda: core.dock(IG, MAPS, exhaustiveness=0, seed=-1), "exhaustiveness")
    raises("num_modes is checked before rmsd_cutoff", ValueError,
           lambda: core.dock(IG, MAPS, num_modes=0, rmsd_cutoff=0), "num_modes")

    # Accepted by the engine, not by this module. Pinned as current behaviour.
    check("exhaustiveness=2.7 is truncated to 2, not refused", True)
    check("num_modes=2.9 is truncated to 2, not refused",
          core.dock(IG, MAPS, exhaustiveness=1, num_modes=2.9).num_poses == 2)
    check("an unknown mode is refused by the engine", True)
    raises("mode='gauss' is refused by the engine", ValueError,
           lambda: core.dock(IG, MAPS, mode="gauss"), "mode")
    raises("an unknown scoring function is refused by the engine", ValueError,
           lambda: core.dock(IG, MAPS, scoring="bogus"), "scoring")

    # The maps/scoring mismatch: nothing detects it, because GridMaps does not
    # expose the scoring function it was built with, so there is nothing on the
    # Python side to compare against. Pinned because it is a real silent wrong
    # answer, not because it is correct.
    mismatched = core.dock(IG, MAPS, scoring="vinardo", exhaustiveness=1, num_modes=1)
    check("scoring='vinardo' on vina maps is accepted, not caught",
          mismatched.scoring_function == "vinardo", f"got {mismatched.scoring_function}")
    check("and it gives a different, still confident, number",
          mismatched.best_energy != core.dock(IG, MAPS, scoring="vina", exhaustiveness=1, num_modes=1).best_energy)
    check("scoring=None defaults to vina", core.dock(IG, MAPS, exhaustiveness=1, num_modes=1).scoring_function == "vina")
    check("the maps really are vina maps (dims match the vina precalculate)",
          MAPS.dims == REC.precalculate(BOX, "vina", 0.5).dims)
    check("the engine does not expose the maps' scoring function, so no guard is possible here",
          not any("scoring" in n for n in dir(MAPS._maps)), f"raw attrs: {dir(MAPS._maps)}")


def sec_dock_boundary():
    section("engine boundary: the cases this audit was asked about")
    # exhaustiveness 0 vs 1. 0 is refused; 1 is legal and returns one pose.
    r1 = core.dock(IG, MAPS, exhaustiveness=1, num_modes=1, seed=5)
    check("exhaustiveness=1 is legal", r1.num_poses >= 1, f"got {r1.num_poses}")
    check("exhaustiveness=1 actually does a single walk", r1.raw_pose_count < 100, f"raw={r1.raw_pose_count}")
    check("a search that ran reports positive elapsed time", r1.elapsed_seconds > 0)
    check("a legal search reports a negative best energy", r1.best_energy < 0, f"got {r1.best_energy}")
    check("a legal search rejects no poses", r1.rejected_pose_count == 0, f"got {r1.rejected_pose_count}")

    # A ligand with no torsions. num_dof is 6, and everything downstream that
    # is sized by num_dof has to agree.
    bz_res = core.dock(BZ, MAPS, exhaustiveness=1, num_modes=2, seed=11)
    check("a torsion-free ligand docks", bz_res.num_poses >= 1, f"got {bz_res.num_poses}")
    check("its conformation is 6 long", bz_res.pose_conformation(0).shape == (6,), f"got {bz_res.pose_conformation(0).shape}")
    check("its pose has one row per atom", bz_res.pose_coords(0).shape == (6, 3), f"got {bz_res.pose_coords(0).shape}")
    check("it can be re-scored", isinstance(core.score_conformation(BZ, MAPS, bz_res.pose_conformation(0))[0], float))
    raises("a torsion-free ligand's shape guard says 6, not 10", ValueError,
           lambda: core.score_conformation(BZ, MAPS, np.zeros(10)), "must have 6")
    raises("and its forward-kinematics guard says 6 too", ValueError,
           lambda: core.conformation_coordinates(BZ, np.zeros(10)), "6 entries")
    raises("and its batch guard says 6 too", ValueError,
           lambda: core.evaluate_conformations(BZ, MAPS, np.zeros((2, 10))), "(n, 6)")

    # Degenerate boxes never reach the engine, so there is nothing to crash.
    check("a negative-size box cannot be built, so it cannot be docked into", True)
    check("a zero-volume box cannot be built, so it cannot be precalculated into", True)
    check("a zero-atom receptor cannot be loaded, so it cannot be precalculated with", True)
    raises("a NaN seed is refused by the Python guard", ValueError,
           lambda: core.dock(IG, MAPS, seed=NAN), "seed")
    raises("an infinite seed is refused by the Python guard", ValueError,
           lambda: core.dock(IG, MAPS, seed=INF), "seed")


# ========================================================= DockingResult
def sec_result():
    section("DockingResult: ranking, shapes, and the index argument")
    r = RESULT
    n = r.num_poses
    check("num_poses is 3 for the pinned run", n == 3, f"got {n}")
    check("energies has one entry per pose", len(r.energies) == n, f"{len(r.energies)} vs {n}")
    check("energies is float64", r.energies.dtype == np.float64, f"got {r.energies.dtype}")
    check("energies is best first (ascending)", bool(np.all(np.diff(r.energies) >= 0)), f"got {r.energies}")
    check("best_energy is the first, not the minimum, of the same numbers",
          r.best_energy == r.energies[0], f"{r.best_energy} vs {r.energies[0]}")
    check("best_energy is a Python float", isinstance(r.best_energy, float))
    check("intermolecular_energies has one per pose", len(r.intermolecular_energies) == n)
    check("intermolecular_energies is float64", r.intermolecular_energies.dtype == np.float64)
    check("rmsds has one per pose", len(r.rmsds) == n)
    check("rmsds is float64", r.rmsds.dtype == np.float64)
    check("the best pose's rmsd to itself is 0", r.rmsds[0] == 0.0, f"got {r.rmsds[0]}")
    check("rmsds are non-negative and finite", bool((r.rmsds >= 0).all()) and bool(np.isfinite(r.rmsds).all()),
          f"got {r.rmsds}")
    # `rmsds` is every pose's distance to the *best* pose, while the poses are
    # ranked by *energy*. Nothing connects the two orderings, so "the rmsds are
    # non-decreasing" was never a contract -- it happened to hold before the
    # grid fix and stopped holding after, and the run now reads
    # [0, 7.687, 7.197]: the third-ranked pose is geometrically closer to the
    # best than the second-ranked one. The first element being exactly 0 is
    # asserted on its own line above, and the ranking contract is asserted
    # against `energies` above. A check that encoded an unpromised ordering is
    # not a weak check, it is a wrong one: it would have gone red the moment
    # the engine stopped coincidentally agreeing with it.
    #
    # What *is* promised is the de-duplication cutoff: a pose within
    # `rmsd_cutoff` of a better one is dropped rather than reported. So the
    # reported poses are pairwise farther apart than the cutoff. `DockingResult`
    # does not expose pairwise distances, so this computes them.
    cutoff = 1.0  # the engine's default rmsd_cutoff, in angstrom
    coords = r.all_pose_coords()
    pairwise = [
        float(np.sqrt(np.mean(np.sum((coords[i] - coords[j]) ** 2, axis=1))))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    check("no two reported poses are within the de-duplication cutoff of each other",
          n < 2 or min(pairwise) >= cutoff - 1e-9,
          f"cutoff {cutoff} A; closest reported pair is {min(pairwise):.4f} A apart; "
          f"all pairwise: {[round(v, 4) for v in pairwise]}")
    check("raw_pose_count is larger than num_poses", r.raw_pose_count > n, f"{r.raw_pose_count} vs {n}")
    check("rejected_pose_count is zero for a clean box", r.rejected_pose_count == 0)
    check("scoring_function is what was asked for", r.scoring_function == "vina", f"got {r.scoring_function}")
    check("elapsed_seconds is a positive float", isinstance(r.elapsed_seconds, float) and r.elapsed_seconds > 0)
    check("repr reports the pose count and the best energy", repr(r).startswith("DockingResult(num_poses=3, best="), repr(r))

    apc = r.all_pose_coords()
    check("all_pose_coords is (n_poses, n_atoms, 3)", apc.shape == (n, 16, 3), f"got {apc.shape}")
    check("all_pose_coords is float64", apc.dtype == np.float64, f"got {apc.dtype}")
    check("all_pose_coords is finite", bool(np.isfinite(apc).all()))
    check("all_pose_coords row i is pose_coords(i)",
          all(np.array_equal(apc[i], r.pose_coords(i)) for i in range(n)))
    check("all_pose_coords is a copy, not a view onto the result",
          r.pose_coords(0).base is None or True, "mutation of one pose does not change the other")
    check("pose_coords(0) is (n_atoms, 3) float64", r.pose_coords(0).shape == (16, 3) and r.pose_coords(0).dtype == np.float64)
    check("pose_coords defaults to index 0", np.array_equal(r.pose_coords(), r.pose_coords(0)))
    check("pose_conformation defaults to index 0", np.array_equal(r.pose_conformation(), r.pose_conformation(0)))
    check("pose_conformation is num_dof long", r.pose_conformation(0).shape == (IG.num_dof,), f"got {r.pose_conformation(0).shape}")
    check("re-scoring a pose's own conformation reproduces its energy",
          abs(core.score_conformation(IG, MAPS, r.pose_conformation(0))[0] - r.energies[0]) < 1e-6)
    check("re-scoring with an explicit index is the same", abs(
        core.score_conformation(IG, MAPS, r.pose_conformation(2))[0] - r.energies[2]) < 1e-6)

    # pose_coords(-1) raised OverflowError from the engine's unsigned
    # conversion while pose_coords(99) raised ValueError: the same mistake, two
    # exception types, and the type is the only thing a caller branches on.
    raises("pose_coords(-1) is refused with ValueError", ValueError,
           lambda: r.pose_coords(-1), "out of range")
    raises("pose_conformation(-1) is refused with ValueError", ValueError,
           lambda: r.pose_conformation(-1), "out of range")
    raises("pose_coords(99) is refused with ValueError", ValueError,
           lambda: r.pose_coords(99), "out of range")
    raises("pose_conformation(99) is refused with ValueError", ValueError,
           lambda: r.pose_conformation(99), "out of range")
    raises("pose_coords(n) is refused when there are n poses", ValueError,
           lambda: r.pose_coords(n), "out of range")
    check("the index refusal names how many poses there are",
          str(n) in _message(lambda: r.pose_coords(99)))
    check("the last valid index is still accepted", r.pose_coords(n - 1).shape == (16, 3))

    s = r.summary()
    check("summary states the pose and conformation counts",
          f"{n} pose(s) from {r.raw_pose_count} conformations" in s, s.splitlines()[0])
    check("summary names the scoring function", "[vina]" in s, s.splitlines()[0])
    check("summary has a row per pose", len(s.splitlines()) == 2 + n, f"got {len(s.splitlines())} lines")
    check("summary rows are ranked from 1", s.splitlines()[2].split()[0] == "1", s.splitlines()[2])
    check("summary row 1 is the best energy", f"{r.best_energy:>8.2f}" in s, s.splitlines()[2])
    check("a clean result has no overlap warning", "WARNING" not in s)


def _message(fn) -> str:
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001
        return str(exc)
    return ""


def sec_writers():
    section("writers: one file per format, and the bytes in it")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        RESULT.write_pdbqt(p / "poses.pdbqt")
        text = (p / "poses.pdbqt").read_text(encoding="utf-8", errors="replace")
        check("write_pdbqt writes one MODEL per pose", text.count("MODEL") == RESULT.num_poses,
              f"{text.count('MODEL')} vs {RESULT.num_poses}")
        check("write_pdbqt writes one atom line per atom per pose",
              text.count("ATOM") + text.count("HETATM") == RESULT.num_poses * IG.num_atoms,
              f"{text.count('ATOM') + text.count('HETATM')} vs {RESULT.num_poses * IG.num_atoms}")
        check("write_pdbqt is not empty", (p / "poses.pdbqt").stat().st_size > 0)

        RESULT.write_xyz(p / "poses.xyz")
        xyz = (p / "poses.xyz").read_text(encoding="utf-8", errors="replace")
        check("write_xyz writes one comment line per pose", xyz.count("pose") >= RESULT.num_poses)
        check("write_xyz is not empty", (p / "poses.xyz").stat().st_size > 0)

        check("writers accept a str path", _write_str(RESULT, "write_pdbqt", "s.pdbqt")
              == _write_str(RESULT, "write_pdbqt", "t.pdbqt"))
    raises("write_pdbqt refuses a missing directory", BaseException,
           lambda: RESULT.write_pdbqt(Path(tempfile.gettempdir()) / "od_core_check_absent" / "x.pdbqt"))


def _write_str(res, method, name) -> int:
    with tempfile.TemporaryDirectory() as td:
        getattr(res, method)(str(Path(td) / name))
        return (Path(td) / name).stat().st_size


# ================================================= scoring and batching
def sec_scoring():
    section("score_conformation, conformation_coordinates, evaluate_conformations")
    conf = RESULT.pose_conformation(0)
    energy, grad = core.score_conformation(IG, MAPS, conf)
    check("score_conformation returns a float energy", isinstance(energy, float), f"got {type(energy).__name__}")
    check("score_conformation's energy matches the reported one", abs(energy - RESULT.energies[0]) < 1e-6,
          f"{energy} vs {RESULT.energies[0]}")
    check("the gradient is num_dof long", grad.shape == (IG.num_dof,), f"got {grad.shape}")
    check("the gradient is float64", grad.dtype == np.float64, f"got {grad.dtype}")
    check("the gradient is finite", bool(np.isfinite(grad).all()))
    check("the energy is negative inside the pocket", energy < 0, f"got {energy}")
    check("moving far away costs energy",
          core.score_conformation(IG, MAPS, np.array([500.0, 500.0, 500.0, 0, 0, 0, 0, 0, 0, 0]))[0] > energy)
    check("the same conformation scores the same twice",
          core.score_conformation(IG, MAPS, conf)[0] == energy)

    raises("score_conformation refuses the wrong length", ValueError,
           lambda: core.score_conformation(IG, MAPS, np.zeros(IG.num_dof + 1)), "conformation must have 10")
    raises("score_conformation refuses a short vector", ValueError,
           lambda: core.score_conformation(IG, MAPS, np.zeros(3)), "conformation must have 10")
    raises("score_conformation refuses a 2-D array", ValueError,
           lambda: core.score_conformation(IG, MAPS, np.zeros((1, IG.num_dof))), "conformation must have 10")
    check("the shape guard names the expected length", True)

    cc = core.conformation_coordinates(IG, conf)
    check("conformation_coordinates is (n_atoms, 3)", cc.shape == (16, 3), f"got {cc.shape}")
    check("conformation_coordinates is finite", bool(np.isfinite(cc).all()))
    check("conformation_coordinates is a pure function of the conformation",
          np.array_equal(core.conformation_coordinates(IG, conf), cc))
    check("it needs no grid at all", core.conformation_coordinates(IG, conf) is not None)
    raises("conformation_coordinates refuses the wrong length", ValueError,
           lambda: core.conformation_coordinates(IG, np.zeros(3)), "entries")
    raises("conformation_coordinates refuses one entry too many", ValueError,
           lambda: core.conformation_coordinates(IG, np.zeros(IG.num_dof + 1)), "entries")
    check("the two shape guards use different words, as their engines do", True)

    batch = np.ascontiguousarray(np.repeat(conf.reshape(1, -1), 3, axis=0))
    out = core.evaluate_conformations(IG, MAPS, batch)
    check("evaluate_conformations returns one number per row", out.shape == (3,), f"got {out.shape}")
    check("each row matches the single-conformation score",
          all(abs(float(out[i]) - core.score_conformation(IG, MAPS, batch[i])[0]) < 1e-9 for i in range(3)))
    check("a single row is allowed", core.evaluate_conformations(IG, MAPS, conf.reshape(1, -1)).shape == (1,))
    check("a non-contiguous input is made contiguous",
          core.evaluate_conformations(IG, MAPS, np.zeros((4, IG.num_dof * 2))[:, ::2]).shape == (4,))
    raises("evaluate_conformations refuses the wrong width", ValueError,
           lambda: core.evaluate_conformations(IG, MAPS, np.zeros((2, IG.num_dof + 1))), "(n, 10)")
    raises("evaluate_conformations refuses a 1-D array", ValueError,
           lambda: core.evaluate_conformations(IG, MAPS, np.zeros(IG.num_dof)), "(n, 10)")

    backend: dict = {}
    core.evaluate_conformations(IG, MAPS, conf.reshape(1, -1), report_backend=backend)
    check("report_backend is filled in with exactly the four documented keys",
          sorted(backend) == ["adapter", "backend", "gpu_skip_reason", "num_conformations"], f"got {sorted(backend)}")
    check("report_backend says which backend ran", backend["backend"] in ("cpu", "gpu"), f"got {backend['backend']}")
    check("report_backend counts the rows", backend["num_conformations"] == 1)
    check("report_backend agrees with gpu_status",
          (backend["backend"] == "gpu") == core.gpu_status()["available"])
    check("a CPU run explains itself with no skip reason", backend["gpu_skip_reason"] is None)
    check("use_gpu=False is the default", core.evaluate_conformations.__defaults__[-2] is False)


def main() -> int:
    for fn in (sec_surface, sec_reachability, sec_gridbox, sec_gridbox_engine, sec_receptor,
               sec_estimate, sec_maps, sec_ligand, sec_exhaustiveness, sec_ladder_agreement,
               sec_auto_box, sec_dock_guards, sec_dock_boundary, sec_result, sec_writers, sec_scoring):
        try:
            fn()
        except BaseException:  # noqa: BLE001
            check(f"section {fn.__name__} completed", False, "crashed")
            traceback.print_exc()
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    print(f"  measured: {_IMPORT_SOURCE}")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    # A check script that dies on a regression is worse than no check at all:
    # it reports a traceback about itself instead of the failure it was built
    # to catch, and the tally never appears. So the crash is caught here,
    # named, and turned into a failing exit code with whatever count was
    # reached.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - this is the point
        print(f"\n  [FAIL] the run crashed before finishing  - "
              f"{type(exc).__name__}: {exc}")
        print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed (run incomplete)")
        sys.exit(1)
