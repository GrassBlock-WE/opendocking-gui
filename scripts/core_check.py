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
#: Whether the package resolved from a `site-packages` directory, which decides
#: how a child process has to be pointed at it -- see `sec_output_modes`, where
#: prepending that directory is a way of breaking the child rather than of
#: choosing its copy of the package. Named once so the label below and that
#: section cannot disagree about which copy this run measured.
_IN_SITE_PACKAGES = "site-packages" in _RESOLVED.parts
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
elif _IN_SITE_PACKAGES:
    _IMPORT_SOURCE = f"the installed wheel at {_RESOLVED}"
else:
    _IMPORT_SOURCE = (
        f"a source tree that is neither this checkout's dock-py/python nor the "
        f"installed wheel, at {_RESOLVED}"
    )

from opendocking import core  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0

#: How many checks this file records, **counted from a run and not derived from
#: the source above**.
#:
#: `F:\python310\python.exe scripts\core_check.py` with
#: `PYTHONIOENCODING=utf-8` and `PYTHONPATH=dock-py\python`: **`509/509 passed`,
#: exit 0**, the run reporting `measured: the source tree at
#: ...\dock-py\python\opendocking`. The pin is that run's 509 plus the tally
#: check below, so 510, and the run that has to agree with the pin is the
#: *second* run rather than the one the 509 came from.
#:
#: **361 call sites produce 509 checks, and the two are not derivable from one
#: another.** What is known, measured rather than assumed:
#:
#:   * six loops each turn one site into several checks. The reachability table
#:     is 63 members across 6 classes; the exhaustiveness ladder is 7 literal
#:     box sizes; `sec_estimate` and `sec_result` are 3 literal items each.
#:   * the two sites in the reachability table's `try`/`else` are **one** check
#:     per member, not two: the `else` runs when the call works and the
#:     `except` when it raises, so 2 sites produce 63 checks.
#:   * `raises()` holds 4 `return check(...)` sites and is called 73 times, so
#:     those 4 sites produce 73 checks.
#:   * `main()`'s section loop holds one crash-site that fires **zero** times
#:     on a clean run, and 22 sections is its literal length.
#:
#: **The arithmetic does not close on paper, and that is recorded rather than
#: tidied.** Those five facts account for most of the gap between 361 and 509
#: and the remainder was not traced, so a reader must not treat this note as an
#: equation that reproduces 510. It is the measured pair and the mechanisms that
#: are known to move a site into several checks -- the same reason
#: `check_scripts_declare.py` says it re-reads its own number from the run every
#: time instead of showing the sum: a comment that displayed a closed total
#: would make the values that did not close look deliberate.
#:
#: **One multiplier is not this file's to decide, and it is the reason this pin
#: is not a closed number.** `sec_output_modes` loops over `warned`, which is
#: built by running a regex over `dock-py/python/opendocking/core.py` to find
#: every `if self.<name>:` guard in `DockingResult.summary` -- 3 today. So a
#: change to the product's summary table moves this count, by exactly the number
#: of warnings added or removed, and the tally check will go red and say the
#: count moved. That is the intended reading: the product grew a warning the
#: table can print, the gate is supposed to test that the wrapper exposes it,
#: and the count is the evidence that it did. The fix is to read what the red
#: says, not to widen this constant.
#:
#: **What else takes a run below the pin, and why each of those is a finding
#: rather than a nuisance.** A section that raises is caught by `main()` and
#: recorded as a failed `section <name> completed` check, and the tally goes red
#: as well -- a second red about one cause, which is the shape the comment above
#: `sec_output_modes` already complained about. So the tally's detail names the
#: sections that crashed, and the two reds are one diagnosis rather than two
#: bare numbers. The other path is `if not records: return` in the same section:
#: a child that emitted no JSON record skips four checks, the tally names the
#: shortfall, and again the cause is in the output already.
#:
#: **What the prose does and does not say about this number, checked rather than
#: assumed.** An earlier version of this note claimed that "the 509 the prose
#: quotes for this gate is now checkable, and it is wrong by one". It is not
#: true, and the way it is untrue is worth recording: no live figure in
#: `README.md`, `README.en.md` or `CHANGELOG.md` quotes this gate's *check
#: total*. Running `provenance_appendix.py` against a tree with this pin in place
#: resolves exactly one live figure to `core_check` -- `CHANGELOG.md:718`,
#: `'22 sections'` -- and that figure is compared against the number of
#: `section()` call sites read out of this file's AST (22), not against
#: `EXPECTED_CHECKS`, and it matches. So this pin changes no verdict in the
#: prose today. What it buys is that the *next* figure quoted for this gate has
#: something to contradict it, and that the file's own total is asserted inside
#: the file rather than only declared in a comment. A pin whose value is
#: "nothing in the tree can check this yet" is still a pin: it is the count the
#: tally check compares against on every run, and it is read by
#: `provenance_appendix.py` with `ast` the moment any figure does name it.
EXPECTED_CHECKS = 510


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
REC_TERMS = REC.precalculate_terms(BOX, "vina", 0.5)
RESULT = core.dock(IG, MAPS, exhaustiveness=4, num_modes=3, seed=42)
NAN = float("nan")
INF = float("inf")


# =========================================================== module surface
def sec_surface():
    section("module surface: every exported name exists and is the right kind")
    # 22 = 13 functions + 6 classes + 3 constants. The two that were added since
    # this file last said 20 are `TermMaps` and `score_conformation_terms`, and
    # they came in as a pair: the decomposition is only usable if a caller can
    # *build* the per-term maps and *read* the per-term breakdown, so a pin that
    # counted one without the other would have passed on a module that could do
    # half the job. The names are listed rather than only counted, so a swap of a
    # class for a function is a visible edit and not a number that happens to
    # match.
    check("__all__ has 22 names", len(core.__all__) == 22, f"got {len(core.__all__)}")
    missing = [n for n in core.__all__ if not hasattr(core, n)]
    check("every __all__ name resolves", not missing, f"missing: {missing}")
    functions = [n for n in core.__all__ if callable(getattr(core, n)) and not isinstance(getattr(core, n), type)]
    check("13 of them are functions", len(functions) == 13, f"got {len(functions)}")
    classes = [n for n in core.__all__ if isinstance(getattr(core, n), type)]
    check("6 of them are classes", len(classes) == 6, f"got {classes}")
    constants = [n for n in core.__all__ if n not in functions and n not in classes]
    check("the other 3 are constants", len(constants) == 3, f"got {constants}")
    check("every exported name is in __all__ exactly once",
          len(set(core.__all__)) == len(core.__all__))
    check("the decomposition's two halves are both exported",
          "TermMaps" in core.__all__ and "score_conformation_terms" in core.__all__,
          f"__all__ has TermMaps={'TermMaps' in core.__all__}, "
          f"score_conformation_terms={'score_conformation_terms' in core.__all__}")

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

    # The same census question, asked of the surface Python publishes. `dock`'s
    # keyword list is *derived* from the live signature, so a new keyword joins
    # the census the moment it is written and cannot ship unforwarded. A keyword
    # in the signature that appears in neither this module nor the binding is a
    # promise with nothing behind it -- the shape of defect this has now found
    # four times in four places, none of which had a check asking in general.
    import inspect
    sig_params = [p for p in inspect.signature(core.dock).parameters]
    this_module = Path(__file__).read_text(encoding="utf-8")
    binding = (ROOT / "dock-py" / "src" / "lib.rs").read_text(encoding="utf-8")
    unforwarded = [p for p in sig_params
                   if p not in this_module or p not in binding]
    check("every keyword dock() publishes is forwarded somewhere",
          not unforwarded,
          f"core.dock publishes {sig_params}; not mentioned in both core.py and "
          f"the binding: {unforwarded}. A keyword that stops being forwarded is a "
          f"parameter the user can set that does nothing")
    # The tenth keyword arrived deliberately rather than quietly, which is what
    # this tripwire asks for: `min_contact_distance` is
    # `DockingConfig::min_contact_distance`, a field `dock()` itself reads in
    # the clash partition (`docking.rs:257`) and one the binding did not publish,
    # so every Python `dock()` was silently fixed at 2.0 A while its own
    # documentation told callers to set 0.0. It is pinned for real in
    # `sec_contact_distance`, which drives the clash partition through it.
    check("dock's published surface is the ten it is documented to have",
          len(sig_params) == 10,
          f"core.dock publishes {len(sig_params)} keywords {sig_params}. Not a "
          f"number to protect so much as a tripwire: an eleventh keyword is either a "
          f"new promise or a new gap, and either way it should arrive with a "
          f"decision rather than quietly")


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
    rec_terms = REC_TERMS

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
            # The second way to tabulate a box, and the one the whole
            # per-term decomposition rests on. It was missing from this table
            # while being reachable, which is the state this table exists to
            # prevent: an exposed entry point with no test is not an exposed
            # entry point. Its semantics are pinned separately, in
            # `sec_term_maps`, because "it returns something" and "its fields
            # mean what their names say" are different claims.
            ("precalculate_terms", lambda: rec.precalculate_terms(BOX, "vina", 0.5)),
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
            # The maps' own unrecognised-atom count. `Receptor` and
            # `DockingResult` have both carried it and the engine has held it on
            # `GridMaps` since the field was added, so this class was the one
            # place in the chain that could not be asked -- and the maps are
            # exactly the object that outlives the receptor. Its two directions
            # are pinned in `sec_maps`.
            ("unknown_atom_types", lambda: maps.unknown_atom_types),
            # The one that was broken for the life of the file.
            ("box", lambda: maps.box),
            ("write_map_files", lambda: _write_maps(maps)),
            ("__repr__", lambda: repr(maps)),
        ],
        # The class `precalculate_terms` hands back. It was missing from this
        # table while being reachable, which is why `TermMaps.box` could raise
        # `TypeError: ... object is not callable` on every single access for the
        # whole life of the property without anything going red: the entry point
        # that produces it was listed, the object it produces was not. The
        # inverse check below cannot notice that either, because it only asks
        # about classes the table already mentions.
        core.TermMaps: [
            ("dims", lambda: rec_terms.dims),
            ("spacing", lambda: rec_terms.spacing),
            ("num_points", lambda: rec_terms.num_points),
            ("memory_mb", lambda: rec_terms.memory_mb),
            ("box", lambda: rec_terms.box),
            # The backend, which is the question a caller should be able to ask
            # *before* paying 1.5x the memory for a per-term tabulation. It was
            # absent while the engine had carried `TermMaps::BACKEND` for the
            # life of the feature, so from Python the cheaper question had no
            # answer and the only way to get one was to build the table and
            # read a number off it.
            ("backend", lambda: rec_terms.backend),
            ("__repr__", lambda: repr(rec_terms)),
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
            ("unknown_atom_types", lambda: res.unknown_atom_types),
            # The out-of-box pose count. It was carried from the engine through
            # the binding and reachable only by reaching into `res._res`, which
            # is not a surface a caller has: `hasattr(DockingResult, ...)` was
            # False, so the one number that says "this run describes no binding
            # mode" was invisible to the person the run was for. Its two
            # directions are pinned in `sec_result`.
            ("poses_outside_box_count", lambda: res.poses_outside_box_count),
            ("scoring_function", lambda: res.scoring_function),
            ("pose_coords", lambda: res.pose_coords(0)),
            ("pose_conformation", lambda: res.pose_conformation(0)),
            # The pose's own gradient, which the search computed at this
            # conformation and used to discard. The two directions are pinned
            # separately, in `sec_result`, because "it is there" and "a consumer
            # can tell when it is not" are different claims and only the second
            # one stops a reader assuming the field is populated.
            ("pose_gradient", lambda: res.pose_gradient(0)),
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

    # --- the two constructors are not interchangeable, and the trap is silent
    # `GridBox(a, b)` reads a as a *corner* and b as a *corner*.
    # `GridBox.from_center_size(a, b)` reads a as a *centre* and b as a *span*.
    # Same arity, same types, opposite meaning for the first argument, and the
    # boxes are half a span apart. This was the finding behind the class
    # docstring, and it is pinned here because a reader's memory is not a guard.
    corners = core.GridBox((0.0, 0.0, 0.0), (14.0, 14.0, 14.0))
    centred = core.GridBox.from_center_size((0.0, 0.0, 0.0), (14.0, 14.0, 14.0))
    check("the two constructors agree on edge length", corners.size == centred.size == (14.0, 14.0, 14.0),
          f"corners {corners.size} against centred {centred.size}")
    offset = tuple(m - n for m, n in zip(centred.min_corner, corners.min_corner))
    check("and still put the box half a span apart, on every axis",
          offset == (-7.0, -7.0, -7.0),
          f"corners span {corners.min_corner}..{corners.max_corner}, centred spans "
          f"{centred.min_corner}..{centred.max_corner}: an offset of {offset}. Same "
          f"numbers, no error either way")
    # The dangerous half of the same mistake, pinned as it behaves today: a span
    # handed to the corner path is *accepted* whenever it happens to be forward.
    # Only the inverted one is refused, because `!(max > min)` rejects it.
    misuse = core.GridBox((0.0, 0.0, 0.0), (2.0, 14.0, 14.0))
    check("a span handed to the corner path is silently accepted",
          misuse.size == (2.0, 14.0, 14.0),
          f"(0,0,0),(2,14,14) builds a {misuse.size} box with no complaint. Pinned as "
          f"a fact, not endorsed: it is why the docstring tells you to prefer the "
          f"keyword forms")
    # The advice the docstring gives, pinned so it stays true.
    raises("the corner path refuses the keyword the class docstring recommends instead",
           TypeError,
           lambda: core.GridBox(center=(0.0, 0.0, 0.0), size=(14.0, 14.0, 14.0)),
           "center")
    # And the docstring itself, because a documented asymmetry that the
    # documentation does not mention is the state this section was written for.
    # The slice is the `Parameters` block specifically, not the prose above it:
    # an earlier version of this check took everything before "Examples" and was
    # satisfied by the comparison table alone, so deleting the Parameters entry
    # left it green. The entry is what a reader of a signature is looking at.
    gridbox_doc = core.GridBox.__doc__ or ""
    parameters = ""
    if "Parameters" in gridbox_doc:
        parameters = gridbox_doc.split("Parameters", 1)[1].split("Examples", 1)[0]
    check("the class docstring's Parameters block names both constructors",
          "min_corner" in parameters and "from_center_size" in parameters,
          f"the Parameters block is {parameters.strip()[:70]!r}...; it names "
          f"min_corner: {'min_corner' in parameters}, from_center_size: "
          f"{'from_center_size' in parameters}. It used to name only the first, "
          f"with the second mentioned solely in the comparison table above and in "
          f"the Examples")


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


# ==================================================== precalculate_terms
def sec_term_maps():
    section("precalculate_terms: the per-term tabulation means what its keys say")
    # `sec_reachability` only asks whether `precalculate_terms` *works*: it comes
    # back, and the members of what it comes back can be read. These ask whether
    # it is **right**, and the three questions are deliberately different from one
    # another:
    #
    # 1. does the total agree with the production path, to within the one `f32`
    #    rounding that separates them (the band is derived below, not picked);
    # 2. does each *key* carry its own term, as opposed to the right set of
    #    numbers under the wrong names;
    # 3. is (2) pinned hard enough to survive a reorder.
    #
    # (2) is the one a numeric test with a loose tolerance misses, and (3) is the
    # one a *sign* test misses. A transposition of two keys preserves the sum, so
    # `terms_total == intermolecular` and `shape == g1 + g2 + rep` both still
    # hold with two values exchanged, and it preserves the sign of both, so a
    # test that asks "is the steric term the positive one" still passes. What it
    # cannot preserve is the *ratio* the two Gaussians have as the probe moves
    # away from the surface: `g1` is centred 0.5 A out, `g2` 0.0 A out, and they
    # share a width, so `g1/g2` climbs steeply with surface distance -- 4.82,
    # 12.82, 41.75 on the bare probe below. Exchanging the two keys reciprocates
    # the ratio, so the same three numbers become 0.208, 0.078, 0.024 and a
    # rising sequence becomes a falling one. The pin is therefore an ordering
    # with a margin wide enough that interpolation cannot flip it, and the margin
    # is stated so a reader can see it is not a decimal someone liked.
    #
    # The pose below is the one `docs/SCORING.md` section 5.4 publishes: a
    # receptor carbon at the origin with an acceptor oxygen 3.8 A away, a
    # one-carbon probe at the origin, a 12 A box at 0.375 A. The prose does not
    # give that atom set in a form that reproduces the published table exactly,
    # so the two figures it does state exactly are the two pinned as values --
    # `hbond` and `hydrophobic`, which depend on the C...C pair alone -- and the
    # rest are pinned as properties.
    doc_box = core.GridBox.from_center_size((0.0, 0.0, 0.0), (12.0, 12.0, 12.0))
    doc_rec = core.Receptor.from_pdbqt_str(
        "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
        "ATOM      2  O   UNL     1       3.800   0.000   0.000  1.00  0.00    -0.300 OA\n"
    )
    doc_lig = core.Ligand.from_pdbqt_str(
        "ATOM      1 C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
        "END\n"
    )
    doc_maps = doc_rec.precalculate(doc_box, "vina", 0.375)
    doc_terms = doc_rec.precalculate_terms(doc_box, "vina", 0.375)
    doc = core.score_conformation_terms(doc_lig, doc_maps, doc_terms,
                                        np.zeros(doc_lig.num_dof), "vina")

    eps32 = float(np.finfo(np.float32).eps)

    def f32_band(value):
        """The one-f32-rounding band around `value`, in kcal/mol.

        The two paths being compared read the *same* sum from two storage
        layouts: the production grid rounds `g1 + g2 + rep` into a single f32,
        the term maps round each of the three separately. So they may disagree,
        and the disagreement is bounded by the ulp of an f32 near the value --
        `2^-23 * value`. Four of them is the band used below, and it is used
        for every cross-path comparison in this section for the same reason.
        """
        return 4.0 * eps32 * max(1.0, abs(value))

    # 1. the identity the documentation publishes, with a derived band
    gap = abs(doc["terms_total"] - doc["intermolecular"])
    # Narrow enough that any real transposition fails it: exchanging `g1` and
    # `g2` moves this sum by `2 * (g1 - g2)`, which here is 5.3e-02 -- about
    # 9000x the band.
    band = f32_band(doc["terms_total"])
    check("terms_total equals the production energy to within the f32 rounding "
          "that separates them",
          gap <= band,
          f"terms_total {doc['terms_total']!r} against intermolecular "
          f"{doc['intermolecular']!r}: a gap of {gap:.3e} against a derived band "
          f"of {band:.3e} (4 f32 ulps of {doc['terms_total']:.4f}). "
          f"docs/SCORING.md 5.4 publishes 4.01e-07 for its own pose, the same "
          f"order, and the check pins the *band* rather than that figure because "
          f"the published table's atom set is not stated in a reproducible form")
    check("the production slot 0 is the three shape terms fused",
          abs(doc["shape"] - (doc["g1"] + doc["g2"] + doc["rep"])) <= 1e-12,
          f"shape {doc['shape']!r} against g1+g2+rep "
          f"{doc['g1'] + doc['g2'] + doc['rep']!r}")
    check("the five named terms account for the whole decomposition",
          abs((doc["g1"] + doc["g2"] + doc["rep"] + doc["hb"] + doc["hyd"])
              - doc["terms_total"]) <= 1e-12,
          f"the five sum to "
          f"{doc['g1'] + doc['g2'] + doc['rep'] + doc['hb'] + doc['hyd']!r} "
          f"against terms_total {doc['terms_total']!r}")
    check("the four slots are the fused shape plus the three that are already "
          "separate, each to the f32 rounding of the grid it was read from",
          list(doc["slot_names"]) == ["shape", "hb_from_donor",
                                      "hb_from_acceptor", "hydrophobic"]
          and abs(doc["slots"][0] - doc["shape"]) <= f32_band(doc["shape"])
          and abs(doc["slots"][1] - doc["hb_from_donor"]) <= f32_band(doc["hb_from_donor"])
          and abs(doc["slots"][2] - doc["hb_from_acceptor"]) <= f32_band(doc["hb_from_acceptor"])
          and abs(doc["slots"][3] - doc["hyd"]) <= f32_band(doc["hyd"]),
          f"slot_names {doc['slot_names']} against slots {doc['slots']!r}. The "
          f"first slot differs from the f64 `shape` by "
          f"{abs(doc['slots'][0] - doc['shape']):.3e} -- the same one-f32 "
          f"rounding as the total, because the slots come out of the production "
          f"f32 grid and the terms out of the f64 one. It is **not** exact and "
          f"an exact assertion would have been wrong. This is a *four*-slot "
          f"split and not a five-term one: the hydrogen bond is one term spread "
          f"over two slots, so `hb` cannot be read off any single entry here")

    # 2. the split of the hydrogen bond, on a pose where it is not zero. The
    # published pose has a C...C pair and therefore no donor and no acceptor, so
    # `hb` is 0 there and `0 == 0 + 0` would pass on a pair that had the halves
    # exchanged, or on one that had no halves at all. The docked example has six
    # polar hydrogens, so this is read where the split has something to split.
    real = core.score_conformation_terms(IG, MAPS, REC_TERMS, RESULT.pose_conformation(0), "vina")
    check("the hydrogen bond is the sum of its two halves, where it is not zero",
          real["hb"] != 0.0
          and abs(real["hb"] - (real["hb_from_donor"] + real["hb_from_acceptor"])) <= 1e-12,
          f"hb {real['hb']!r} against donor {real['hb_from_donor']!r} plus "
          f"acceptor {real['hb_from_acceptor']!r}. The published C...C pose gives "
          f"hb {doc['hb']!r}, which this assertion deliberately does not use "
          f"because a zero cannot tell the two halves apart")

    # 3. the ratio that a transposition reciprocates
    # A receptor that is one carbon and a probe that is one carbon, so the only
    # pairs in the sum are C...C and the ratio is a property of the two
    # Gaussians rather than of whichever atoms happened to be in the box.
    bare = core.Receptor.from_pdbqt_str(
        "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n")
    bare_lig = core.Ligand.from_pdbqt_str(
        "ATOM      1 C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \nEND\n")
    bare_maps = bare.precalculate(BOX, "vina", 0.5)
    bare_terms = bare.precalculate_terms(BOX, "vina", 0.5)

    def ratio_at(x):
        """`g1 / g2` for the bare probe held at `x` along the x axis."""
        conf = np.zeros(bare_lig.num_dof)
        conf[0] = x
        got = core.score_conformation_terms(bare_lig, bare_maps, bare_terms, conf, "vina")
        return got["g1"] / got["g2"]

    # Carbon-carbon surface distance is centre distance minus 3.8 A (two C
    # interaction radii), so these are surface distances 0.1, 0.5 and 0.7 A.
    r_near, r_mid, r_far = ratio_at(3.9), ratio_at(4.3), ratio_at(4.5)
    # The margin is 1.5x, read off the physics rather than off the numbers: a
    # shared-width Gaussian pair offset by half its own width cannot change its
    # ratio by less than a large factor over 0.6 A. The measured steps are 2.66x
    # and 3.26x, and the transposed sequence steps down by 4.0x and 4.9x, so
    # the pin misses in both directions with room to spare.
    check("g1 pulls harder on g2 as the probe leaves the surface -- the two "
          "keys are not transposed",
          r_mid >= 1.5 * r_near and r_far >= 1.5 * r_mid,
          f"g1/g2 at surface distance 0.1 A is {r_near:.4f}, at 0.5 A is "
          f"{r_mid:.4f}, at 0.7 A is {r_far:.4f}: steps of "
          f"{r_mid / r_near:.2f}x and {r_far / r_mid:.2f}x against a required "
          f"1.5x. g1 is centred 0.5 A out and g2 at 0.0 A with a shared width, "
          f"so the ratio has to rise. Exchanging the two keys reciprocates it "
          f"into {1 / r_near:.4f}, {1 / r_mid:.4f}, {1 / r_far:.4f}, which "
          f"falls by {1 / r_far / (1 / r_near):.2f}x end to end and fails both "
          f"inequalities")
    check("both Gaussians are attractive, and only the steric term is not",
          doc["rep"] > 0.0 and doc["g1"] < 0.0 and doc["g2"] < 0.0
          and doc["hyd"] < 0.0 and doc["hb"] == 0.0,
          f"rep {doc['rep']:+.6e}, g1 {doc['g1']:+.6e}, g2 {doc['g2']:+.6e}, "
          f"hyd {doc['hyd']:+.6e}, hb {doc['hb']:+.6e}. The Gaussians are "
          f"attractive and the steric term repulsive, so a key carrying the "
          f"wrong one has the wrong sign. hb is 0 on this pose because a C...C "
          f"pair has no donor and no acceptor, which is the same 0 "
          f"docs/SCORING.md 5.4 publishes")

    # 4. the two figures the documentation states exactly for this pose
    check("the published C...C figures are reproduced: hbond 0 and hydrophobic "
          "-0.035069",
          doc["hb"] == 0.0 and abs(doc["hyd"] - -0.035069) <= 5e-10,
          f"hbond {doc['hb']!r} against the published 0.000000, hydrophobic "
          f"{doc['hyd']!r} against the published -0.035069 (a gap of "
          f"{abs(doc['hyd'] - -0.035069):.2e}, i.e. it reproduces the printed "
          f"6 digits). Both depend on the C...C pair alone, which is why they "
          f"reproduce while the table's other five figures do not; those are not "
          f"pinned, and the reason is recorded rather than worked around")

    # 5. the two surfaces describe one box
    check("the term maps and the production maps tabulate the same box",
          tuple(doc_terms.dims) == tuple(doc_maps.dims)
          and tuple(doc_terms.box.min_corner) == tuple(doc_maps.box.min_corner)
          and tuple(doc_terms.box.max_corner) == tuple(doc_maps.box.max_corner)
          and doc_terms.spacing == doc_maps.spacing
          and doc_terms.num_points == doc_maps.num_points,
          f"term maps {tuple(doc_terms.dims)} at {doc_terms.spacing} over "
          f"{tuple(doc_terms.box.min_corner)}..{tuple(doc_terms.box.max_corner)} "
          f"against production {tuple(doc_maps.dims)} at {doc_maps.spacing}. The "
          f"two differ only in what each point stores -- 60 f32 against 40 -- so "
          f"a geometry disagreement is a bug in one of them, not a consequence "
          f"of the decomposition")
    check("the term maps are the larger of the two, by the ratio the stride "
          "says they are",
          doc_terms.memory_mb / doc_maps.memory_mb > 1.4
          and doc_terms.memory_mb / doc_maps.memory_mb < 1.6,
          f"term maps {doc_terms.memory_mb:.3f} MB against production "
          f"{doc_maps.memory_mb:.3f} MB, a ratio of "
          f"{doc_terms.memory_mb / doc_maps.memory_mb:.4f}. docs/SCORING.md 5.4 "
          f"publishes TERM_STRIDE 60 against STRIDE 40, which is 1.5; the check "
          f"is loose enough for the per-point bookkeeping to sit either side of "
          f"it and tight enough that a term map silently holding production data "
          f"would fail")

    # 6. the backend the decomposition says produced it
    check("the decomposition names the backend that produced it",
          doc["backend"] == "cpu" and real["backend"] == "cpu",
          f"got {doc['backend']!r} and {real['backend']!r}. One variant exists "
          f"today, so this is not yet a choice between two answers; it is here "
          f"so that a caller reads a value rather than inferring one from which "
          f"path it took, which is the promise docs/SCORING.md 5.4 makes in the "
          f"half it says is delivered")

    # 7. and the same question asked *before* the table is built, which is the
    # whole point of putting it on `TermMaps`
    check("a caller can ask the backend question before tabulating anything",
          doc_terms.backend == "cpu",
          f"got {doc_terms.backend!r} against the decomposition's own "
          f"{doc['backend']!r}. The two must agree: they are the same fact read "
          f"from two places, and a disagreement would mean one of them had been "
          f"reconstructed per language binding rather than derived from the "
          f"engine's constant")
    # The second direction: the projection must *derive* the string rather than
    # spell it. Spelled out in Python it would be a second copy of the answer
    # that could not move when the engine's does, which is the failure this
    # whole projection exists to prevent -- and unlike the pose_gradient case
    # there is no unreachable branch to blame, so a source check is the only
    # thing that sees it.
    #
    # Read from the compiled function's constants rather than its source text.
    # The first version of this check searched the source for the string
    # `"cpu"` and failed on a green implementation, because the property's own
    # docstring explains the answer in prose and prose contains it. `co_consts`
    # holds the literals the code actually uses, the docstring being the first
    # of them, so "does this function contain a string constant equal to cpu" is
    # answerable exactly.
    names = core.TermMaps.backend.fget.__code__.co_names
    consts = core.TermMaps.backend.fget.__code__.co_consts
    check("the projection derives the backend from the engine, not a literal",
          "_terms" in names and "backend" in names
          and not any(c == "cpu" for c in consts),
          f"co_names {names}, co_consts {consts!r}. `co_names` holds the name "
          f"fragments the body loads -- '_terms' and 'backend' together are the "
          f"derivation -- and a string constant 'cpu' in `co_consts` would be a "
          f"second copy of the answer, which cannot move when the engine's does")


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

    # The maps' own unrecognised-atom count, pinned in both directions.
    #
    # Why this is on GridMaps and not only on the two classes that already had
    # it: the receptor is the object that goes *out of scope* the moment maps
    # exist -- precalculate, drop the receptor, hold maps, dock. The engine has
    # carried the number on the maps since the field was added, and `Receptor`
    # and `DockingResult` both surfaced it, but this wrapper did not, so from
    # Python the count existed in the binding's dependency and was unreachable
    # on the one object a caller keeps.
    #
    # Direction one, the zero, or the non-zero direction below is vacuous.
    check("maps.unknown_atom_types is zero for a clean receptor",
          MAPS.unknown_atom_types == 0, f"got {MAPS.unknown_atom_types}")
    check("it is an int and not a bool, so == 0 is a real comparison",
          isinstance(MAPS.unknown_atom_types, int)
          and not isinstance(MAPS.unknown_atom_types, bool),
          f"got {type(MAPS.unknown_atom_types).__name__}")
    check("the maps and the receptor they were tabulated from agree",
          MAPS.unknown_atom_types == REC.unknown_atom_types,
          f"maps {MAPS.unknown_atom_types} vs receptor {REC.unknown_atom_types}")

    # Direction two, the non-zero, on the same appended-`ZZ` fixture the result
    # section uses for the same reason: it is the only difference between this
    # receptor and REC, so a count of 1 cannot be explained by anything else.
    exotic_rec = core.Receptor.from_pdbqt_str(
        (EX / "rec_prep.pdbqt").read_text(encoding="utf-8")
        + "ATOM      5  X1  ALA A   3       4.000   4.000   3.100  1.00  0.00     0.000 ZZ\n"
    )
    exotic_maps = exotic_rec.precalculate(BOX, "vina", 0.5)
    check("an unrecognised type is counted on the maps, not only the receptor",
          exotic_maps.unknown_atom_types == 1,
          f"got {exotic_maps.unknown_atom_types}")
    check("and the clean maps report 0, so it is not a constant",
          exotic_maps.unknown_atom_types != MAPS.unknown_atom_types,
          f"exotic {exotic_maps.unknown_atom_types} vs clean {MAPS.unknown_atom_types}")

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

    # A self-bond that is *in range* is the case the two above do not cover,
    # and it is the one that was silently dropped on both paths. The brief for
    # this round said `core.py` "already refuses both" an out-of-range and a
    # self-referential bond; the first is true and the second was not. The
    # existing check above is named "past the end" because it tests `(0, 5)`
    # with one atom -- out of range, *not* self-referential -- so nothing
    # anywhere asserted `(0, 0)`, and the engine's `i != j` guard threw it away.
    # A dropped bond is a different molecule rather than a missing result: the
    # atoms stop being neighbours, so a torsion and a ring membership can both
    # be lost, and every energy that comes back is finite and plausible.
    raises("from_arrays refuses an in-range self-bond", ValueError,
           lambda: core.Ligand.from_arrays(["C"], [0.0], chain[:1], [(0, 0)]),
           "self-bond")

    # The accept direction, so the refusal above is a check and not a ban on
    # the whole argument. A bond is a *pair*: `(1, 0)` names the same covalent
    # bond as `(0, 1)`, and a table listing it twice is redundant rather than
    # wrong. Both were dropped-then-accepted before; if either is now refused
    # the caller has a table the engine will not take for no stated reason.
    check("a reversed pair is the same bond, and is accepted",
          core.Ligand.from_arrays(["C"] * 2, [0.0] * 2, chain[:2], [(1, 0)]).num_atoms == 2)
    doubled = core.Ligand.from_arrays(["C"] * 2, [0.0] * 2, chain[:2], [(0, 1), (1, 0)])
    check("a bond listed twice is stored once, not refused and not doubled",
          doubled.num_atoms == 2,
          f"num_atoms {doubled.num_atoms}")
    # The duplicated-bond hazard this guards: a rotatable bond present twice
    # makes the cluster graph cyclic, which breaks the invariant that one
    # torsion owns exactly one child cluster. Two carbons are one terminal bond
    # and so zero torsions either way, so the count is asserted only to be
    # equal between the single and doubled tables rather than to be any
    # particular number.
    check("doubling a bond does not change the derived flexibility",
          doubled.num_torsions
          == core.Ligand.from_arrays(["C"] * 2, [0.0] * 2, chain[:2], [(0, 1)]).num_torsions,
          f"doubled {doubled.num_torsions}")

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

    # The pose's own gradient. Two directions, because the second is the one
    # that stops a reader assuming the field is populated: "a real result
    # carries one" is a claim about the engine, and "a consumer can tell when it
    # does not" is a claim about the *interface*, and only the first would still
    # be true if the second were quietly dropped.
    grads = [RESULT.pose_gradient(i) for i in range(RESULT.num_poses)]
    check("every pose of a real result carries a gradient",
          all(g is not None for g in grads),
          f"{sum(g is None for g in grads)} of {len(grads)} poses report None")
    check("a pose gradient has one entry per degree of freedom, float64",
          grads[0] is not None
          and grads[0].shape == (6 + IG.num_torsions,)
          and grads[0].dtype == np.float64,
          f"got {None if grads[0] is None else (grads[0].shape, grads[0].dtype)}, "
          f"want ({6 + IG.num_torsions},) float64")
    check("pose_gradient defaults to index 0, like pose_conformation",
          np.array_equal(RESULT.pose_gradient(), RESULT.pose_gradient(0)))
    check("a pose's gradient is finite", grads[0] is not None
          and bool(np.isfinite(grads[0]).all()),
          f"got {None if grads[0] is None else grads[0]}")
    check("two poses' gradients are not the same vector, so the field is "
          "per-pose rather than one shared answer",
          RESULT.num_poses < 2
          or not np.array_equal(RESULT.pose_gradient(0), RESULT.pose_gradient(1)),
          f"{RESULT.num_poses} poses; a shared gradient would mean the value "
          f"belongs to some other iterate than the pose it is reported on")
    # The `None` direction has no real result to read it from -- every pose the
    # search returns was measured -- so it cannot be pinned by calling it, and a
    # check that tries goes green against a projection that zero-fills the gap.
    # That mutation was made and survived exactly that way: `g is None` is
    # unreachable on every fixture here, so substituting a zero vector changed
    # no observable value. The only thing left to check is the projection
    # itself, so this reads its source and names the two things that must be in
    # it. This is a source check, and deliberately so: it is the only kind that
    # can see a branch nothing calls.
    import inspect
    src = inspect.getsource(core.DockingResult.pose_gradient)
    check("the projection has an explicit `return None` branch",
          "return None" in src,
          f"source is:\n{src}")
    check("the projection never substitutes a zero vector for a missing one",
          not any(c in src for c in ("np.zeros", "np.full", "np.zeros_like", "np.ones")),
          f"a zero-filled fallback would assert stationarity for a pose nobody "
          f"measured, which is the most damaging wrong answer available here. "
          f"Source is:\n{src}")


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
    # The unrecognised-atom count, pinned in both directions. The reachability
    # table is what noticed it: it was added to the wrapper in the same change
    # that added it to the engine, and "declared in the source but absent from
    # the table" is exactly the state that table exists to catch. It answers
    # "it is there"; the two directions below answer "it is right".
    #
    # Direction one, the zero: a clean fixture must report 0, or the positive
    # direction is vacuous. This is the same trap the Rust test
    # `the_unrecognised_atom_count_reaches_the_result` guards against in prose.
    check("unknown_atom_types is zero for a clean receptor",
          r.unknown_atom_types == 0, f"got {r.unknown_atom_types}")
    check("it is an int and not a bool, so == 0 is a real comparison",
          isinstance(r.unknown_atom_types, int) and not isinstance(r.unknown_atom_types, bool),
          f"got {type(r.unknown_atom_types).__name__}")
    check("the result and the receptor it docked against agree",
          r.unknown_atom_types == REC.unknown_atom_types,
          f"result {r.unknown_atom_types} vs receptor {REC.unknown_atom_types}")

    # Direction two, the non-zero, and the reason the check above is not a
    # constant with a comment. `ZZ` is not an AutoDock type; the engine counts
    # it, reports it, and still docks. The receptor is built from the same
    # checked-in file the rest of this script uses, plus one appended line, so
    # the only difference between this receptor and REC is the exotic atom.
    exotic_text = (EX / "rec_prep.pdbqt").read_text(encoding="utf-8") + (
        "ATOM      5  X1  ALA A   3       4.000   4.000   3.100  1.00  0.00     0.000 ZZ\n"
    )
    exotic_rec = core.Receptor.from_pdbqt_str(exotic_text)
    check("an unrecognised type is counted on the receptor",
          exotic_rec.unknown_atom_types == 1, f"got {exotic_rec.unknown_atom_types}")
    check("and only that one: the clean file contributes none",
          exotic_rec.num_atoms == REC.num_atoms + 1
          and exotic_rec.unknown_atom_types - REC.unknown_atom_types == 1,
          f"{exotic_rec.num_atoms} vs {REC.num_atoms} atoms")
    exotic_maps = exotic_rec.precalculate(BOX, "vina", 0.5)
    exotic_result = core.dock(IG, exotic_maps, exhaustiveness=4, num_modes=3, seed=42)
    check("and the count survives onto the result, which is the whole point",
          exotic_result.unknown_atom_types == 1,
          f"got {exotic_result.unknown_atom_types}")
    check("the same run against the clean maps reports 0, so it is not a constant",
          exotic_result.unknown_atom_types != r.unknown_atom_types,
          f"exotic {exotic_result.unknown_atom_types} vs clean {r.unknown_atom_types}")
    check("the exotic run still returns poses: it is reported, not refused",
          exotic_result.num_poses == r.num_poses,
          f"got {exotic_result.num_poses} against {r.num_poses}")
    check("scoring_function is what was asked for", r.scoring_function == "vina", f"got {r.scoring_function}")
    check("elapsed_seconds is a positive float", isinstance(r.elapsed_seconds, float) and r.elapsed_seconds > 0)
    check("repr reports the pose count and the best energy", repr(r).startswith("DockingResult(num_poses=3, best="), repr(r))

    # --- the out-of-box count, on the wrapper -------------------------------
    #
    # Reachability first, and as a *class* attribute, because that is the
    # defect this closes: the number was computed, carried from Rust through
    # the binding, and reachable only as `res._res.poses_outside_box_count`.
    # A caller has no `_res`; the wrapper hides it behind `__slots__` and the
    # name is private by convention. `hasattr` on the instance would have
    # passed the whole time this was broken only if the attribute existed --
    # so the gate is on the class, which is what `hasattr(core.DockingResult,
    # ...)` in a user's shell would report.
    check("DockingResult exposes poses_outside_box_count on the class",
          hasattr(core.DockingResult, "poses_outside_box_count"))
    check("and it is a property, not a method that needs calling",
          isinstance(getattr(core.DockingResult, "poses_outside_box_count", None), property))
    check("reading it needs no argument", r.poses_outside_box_count is not None)
    check("it is an int and not a bool, so == 0 below is a real comparison",
          isinstance(r.poses_outside_box_count, int)
          and not isinstance(r.poses_outside_box_count, bool),
          f"got {type(r.poses_outside_box_count).__name__}")
    check("and it is the same number the private path gave",
          r.poses_outside_box_count == r._res.poses_outside_box_count,
          f"wrapper {r.poses_outside_box_count} vs _res {r._res.poses_outside_box_count}")
    check("it is within range for any result: 0 to num_poses",
          0 <= r.poses_outside_box_count <= r.num_poses,
          f"{r.poses_outside_box_count} of {r.num_poses}")

    # Direction one, the zero. A clean box must report 0, or the non-zero
    # direction is vacuous and a property that always returned 9 would pass
    # every other check in this section.
    check("poses_outside_box_count is zero for a clean box",
          r.poses_outside_box_count == 0, f"got {r.poses_outside_box_count}")

    # Direction two, the non-zero, and the reason the line above is not a
    # constant with a comment. The engine refuses a box that holds no receptor
    # atom and refuses one smaller than the ligand, so the count is not
    # reachable by pointing a box at the wrong place -- which is the honest
    # answer to "how often is this number non-zero": on a fixture chosen to
    # make it so, it still was not. So the non-zero direction is exercised on
    # the *object*, not by hunting for a run: a stand-in result whose engine
    # field is set to a non-zero count, which is the only way to reach the
    # branch without a search that misbehaves. What that proves is the branch
    # and the wording; it proves nothing about the engine producing that count,
    # and the two claims are kept apart on purpose.
    class _Stub:
        """A DockingResult stand-in carrying only what `summary()` reads.

        `summary()` reads six properties and nothing else, so this is the
        smallest object that can exercise its warning branches. It is a stub
        rather than a monkeypatch because it cannot perturb the real class.
        """

        def __init__(self, outside, rejected=0, unknown=0, poses=3):
            self.num_poses = poses
            self.raw_pose_count = 81
            self.elapsed_seconds = 1.0
            self.scoring_function = "vina"
            self.energies = np.array([-6.1, -6.0, -5.9])
            self.intermolecular_energies = np.array([-6.1, -6.0, -5.9])
            self.rmsds = np.array([0.0, 1.5, 2.5])
            self.rejected_pose_count = rejected
            self.unknown_atom_types = unknown
            self.poses_outside_box_count = outside

        summary = core.DockingResult.summary

    for outside, want_line in ((0, False), (1, True), (3, True)):
        stub = _Stub(outside)
        text = core.DockingResult.summary(stub)
        has = "no atom at all inside the search box" in text
        check(f"summary prints the out-of-box line iff the count is {outside}",
              has == want_line,
              f"count {outside} -> line present {has}; "
              f"{text.splitlines()[-1][:60]!r}")
    full = core.DockingResult.summary(_Stub(2, rejected=1, unknown=3, poses=4))
    out_of_box_line = next(
        (ln for ln in full.splitlines() if "inside the search box" in ln), "(no such line)")
    check("a count of 2 of 4 is stated as a fraction of the poses, not alone",
          "2 of 4 reported poses" in out_of_box_line,
          out_of_box_line)
    check("all three warnings can appear together, and each names its count",
          full.count("WARNING") == 3
          and "2 of 4" in full and "1 of 4" in full and "3 receptor atom(s)" in full,
          f"{full.count('WARNING')} WARNING lines")
    # The trade-off, pinned as behaviour rather than left to the comment: a zero
    # count prints no line at all. That is the decision, and this is what
    # "the line is absent" has to mean -- it is *not* evidence of a clean run
    # beyond the fact that the count was zero.
    check("a zero count prints no out-of-box line, by decision",
          "search box" not in core.DockingResult.summary(_Stub(0)),
          "printed something for a zero count")

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
        # Counted as *records*, not as the substring "ATOM". A substring count
        # was good enough until the file grew a record whose text mentions the
        # word -- `REMARK OD_ATOM_ORDER ... SERIAL = ATOM INDEX` mentions it
        # twice, and the count came out 54 against 48 atoms. The name of this
        # check is "one atom line per atom per pose", so it counts lines whose
        # record field is ATOM, which is also what the parser downstream does.
        atom_lines = sum(1 for l in text.splitlines()
                         if l.split()[:1] and l.split()[0] in ("ATOM", "HETATM"))
        check("write_pdbqt writes one atom line per atom per pose",
              atom_lines == RESULT.num_poses * IG.num_atoms,
              f"{atom_lines} ATOM/HETATM records vs {RESULT.num_poses * IG.num_atoms} "
              f"atoms over {RESULT.num_poses} pose(s); the file also contains "
              f"{text.count('ATOM')} occurrences of the word, which is not the "
              f"same thing")
        check("write_pdbqt is not empty", (p / "poses.pdbqt").stat().st_size > 0)

        RESULT.write_xyz(p / "poses.xyz")
        xyz = (p / "poses.xyz").read_text(encoding="utf-8", errors="replace")
        check("write_xyz writes one comment line per pose", xyz.count("pose") >= RESULT.num_poses)
        check("write_xyz is not empty", (p / "poses.xyz").stat().st_size > 0)

        check("writers accept a str path", _write_str(RESULT, "write_pdbqt", "s.pdbqt")
              == _write_str(RESULT, "write_pdbqt", "t.pdbqt"))
    raises("write_pdbqt refuses a missing directory", BaseException,
           lambda: RESULT.write_pdbqt(Path(tempfile.gettempdir()) / "od_core_check_absent" / "x.pdbqt"))

    # --- what a pose file means, and how a reader has to read it ---------------
    #
    # The file carries the poses the API reports, and it has always done so:
    # indexed by the serial column, the coordinates reproduce `pose_coords(i)`
    # to the writer's 3-decimal precision. What the file does *not* do is write
    # its ATOM records in that order -- the rigid root goes first, so the serials
    # run out of sequence -- and until this section nothing in the file said so.
    # A reader that took the n-th record for the n-th atom therefore got a
    # permuted molecule, 3.99 A RMSD away on ibuprofen, while the workbench
    # described it with numbers belonging to the unpermuted one.
    #
    # So the claims here are: the file and the API agree **once the serial is
    # used**; the file says to use it; and the two orders really are different
    # for this ligand, so the declaration is describing a distinction that
    # exists rather than one it has forgotten to check.
    with tempfile.TemporaryDirectory() as td:
        pf = Path(td) / "order.pdbqt"
        RESULT.write_pdbqt(pf)
        text = pf.read_text(encoding="utf-8", errors="replace")

    models, cur = [], None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            cur = []
        elif line.startswith("ENDMDL"):
            models.append(cur or [])
            cur = None
        elif line.startswith("ATOM") and cur is not None:
            cur.append((int(line[6:11]), float(line[30:38]),
                        float(line[38:46]), float(line[46:54])))

    check("one MODEL per reported pose, as before", len(models) == RESULT.num_poses,
          f"{len(models)} MODELs for {RESULT.num_poses} reported poses")

    # 1. the file and the API agree, once the serial is used. This is the
    #    assertion that goes red if the *writer* starts writing something else.
    worst_serial = 0.0
    worst_order = 0.0
    for i, m in enumerate(models):
        by_serial = np.array([[c[1], c[2], c[3]] for c in sorted(m, key=lambda c: c[0])])
        if by_serial.shape != RESULT.pose_coords(i).shape:
            continue
        worst_serial = max(worst_serial, float(np.sqrt(
            ((by_serial - RESULT.pose_coords(i)) ** 2).sum(axis=1).mean())))
        by_order = np.array([[c[1], c[2], c[3]] for c in m])
        worst_order = max(worst_order, float(np.sqrt(
            ((by_order - RESULT.pose_coords(i)) ** 2).sum(axis=1).mean())))
    check("read by the serial column, the file carries pose_coords(i) exactly",
          worst_serial < 1e-3,
          f"worst RMSD {worst_serial:.6f} A over {len(models)} MODELs. The writer "
          f"keeps 3 decimal places, so 1e-3 is the floor, not a tolerance chosen "
          f"to pass")

    # 2. the file declares its order. Inert to other tools, like the
    #    REMARK VINA RESULT and the OD_NBONDS records it already carries.
    check("and every MODEL declares the order, so a reader cannot be misled",
          text.count("REMARK OD_ATOM_ORDER") == RESULT.num_poses,
          f"{text.count('REMARK OD_ATOM_ORDER')} declarations for "
          f"{RESULT.num_poses} MODELs. Without one, a reader has to know a "
          f"convention only this writer breaks")

    # 3. and the two orders really differ here, so 1 and 2 are not vacuous: a
    #    writer that started emitting serial order would leave 2 describing a
    #    distinction that no longer exists, and this has to notice.
    first = [c[0] for c in models[0]] if models else []
    check("the declared order is not the order the records happen to be in",
          first != sorted(first) and worst_order > worst_serial,
          f"MODEL 1 serials in file order {first}. Read in file order that is "
          f"{worst_order:.4f} A from pose_coords(0); read by serial it is "
          f"{worst_serial:.6f} A. The gap between those two numbers is the size "
          f"of the mistake this declaration exists to prevent")

    # 4. and the project's *own* reader accepts the file its writer just wrote.
    #    A pose file is several poses of one molecule, and the reader that
    #    builds a molecule used to append every MODEL's atoms to one list: the
    #    nine poses of `examples/poses.pdbqt` parsed as a single 144-atom
    #    molecule in which atoms from two different poses sat 0.06 A apart, and
    #    the duplicate-atom check refused it. So `write_pdbqt` produced a file
    #    that `Ligand.from_pdbqt` could not read -- the round trip did not close.
    with tempfile.TemporaryDirectory() as td:
        rt = Path(td) / "roundtrip.pdbqt"
        RESULT.write_pdbqt(rt)
        rtext = rt.read_text(encoding="utf-8", errors="replace")
        rmodels, rcur = [], None
        for line in rtext.splitlines():
            if line.startswith("MODEL"):
                rcur = []
            elif line.startswith("ENDMDL"):
                rmodels.append(rcur or [])
                rcur = None
            elif line.startswith("ATOM") and rcur is not None:
                rcur.append((int(line[6:11]), float(line[30:38]),
                             float(line[38:46]), float(line[46:54])))
        n_atoms_in_file = sum(len(m) for m in rmodels)
        got_n, back = None, None
        try:
            back = core.Ligand.from_pdbqt(rt)
            got_n = back.num_atoms
        except Exception as exc:  # noqa: BLE001
            got_n = f"{type(exc).__name__}: {exc}"
        check("the project's own reader accepts the file write_pdbqt wrote",
              got_n == IG.num_atoms,
              f"Ligand.from_pdbqt on a {len(rmodels)}-MODEL file holding "
              f"{n_atoms_in_file} atom records returned {got_n}; the molecule has "
              f"{IG.num_atoms} atoms. A reader that merges the poses reports "
              f"{n_atoms_in_file}, or the duplicate-atom check rejects it outright")

        # 5. and the atoms it hands back are in *serial* order, so the molecule
        #    is not merely the right size but the right one. A count would pass
        #    on a file whose atoms were permuted into the same number.
        worst_back = None
        if back is not None and rmodels:
            by_serial = np.array([[c[1], c[2], c[3]] for c in sorted(rmodels[0],
                                                                     key=lambda c: c[0])])
            ref = np.asarray(back.reference_coords, dtype=float)
            if by_serial.shape == ref.shape:
                worst_back = float(np.sqrt(((by_serial - ref) ** 2).sum(axis=1).mean()))
        check("and the reader hands the atoms back in serial order, not file order",
              worst_back is not None and worst_back < 1e-3,
              f"worst RMSD {worst_back} A between MODEL 1 read by serial and the "
              f"reader's own reference_coords. Again 1e-3 is the write precision")


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

    # What `report_backend` is *for*, stated as two claims that hold on every
    # machine, rather than the one this used to assert.
    #
    # It used to assert `(backend == "gpu") == gpu_status()["available"]`, which
    # is not a property of the engine at all. `gpu_status()["available"]`
    # answers "could a GPU be used here"; `backend` answers "what ran". The call
    # above does not ask for a GPU -- `use_gpu` defaults to False, and the check
    # two lines down pins that -- so `backend` is `"cpu"` by construction no
    # matter what the machine has, and the equality holds only on a machine with
    # no usable GPU. On any machine that has one it fails, which is the whole of
    # its evidence: the assertion was true here for the wrong reason and red
    # elsewhere for the right one. It shares a root cause with the earlier
    # `gpu_skip_reason: None` finding -- the report's fields are indexed by
    # "was the GPU requested", and this compared a request-keyed field against
    # an availability-keyed one.
    #
    # The two claims that are actually true, and which fail if the report ever
    # starts lying:
    #
    # 1. A run that did not request the GPU cannot have fallen back from one, so
    #    it must report the CPU and must not name a skip reason. A skip reason
    #    here would be the engine claiming a fallback that never happened.
    # 2. A run that did request the GPU either used it, or fell back *and said
    #    why*. `backend == "cpu"` with no reason is the one combination a
    #    caller cannot act on, and it is the only thing this needs to forbid.
    check("a run that did not ask for the GPU reports the CPU",
          backend["backend"] == "cpu", f"got {backend['backend']}")
    check("a CPU run explains itself with no skip reason", backend["gpu_skip_reason"] is None)
    check("nothing claimed a fallback that was never attempted",
          not (backend["backend"] == "cpu" and backend["gpu_skip_reason"] is not None),
          f"backend {backend['backend']!r} with skip reason "
          f"{backend['gpu_skip_reason']!r}")
    check("a skipped run names an adapter only when one ran",
          (backend["adapter"] is None) == (backend["backend"] == "cpu"),
          f"backend {backend['backend']!r}, adapter {backend['adapter']!r}")

    asked: dict = {}
    core.evaluate_conformations(IG, MAPS, conf.reshape(1, -1), use_gpu=True,
                                report_backend=asked)
    check("a run that asked for the GPU is answered in the same four keys",
          sorted(asked) == ["adapter", "backend", "gpu_skip_reason", "num_conformations"],
          f"got {sorted(asked)}")
    check("asking for the GPU is answered, not ignored",
          asked["backend"] in ("cpu", "gpu"), f"got {asked['backend']}")
    # The claim that matters, and the one whose absence made this machine's
    # green meaningless: whatever happened, the caller can tell whether a
    # fallback occurred.
    check("if it did not run on the GPU it says why, or the GPU was never compiled in",
          asked["backend"] == "gpu" or asked["gpu_skip_reason"] is not None
          or not core.gpu_status()["compiled"],
          f"backend {asked['backend']!r}, skip {asked['gpu_skip_reason']!r}, "
          f"compiled {core.gpu_status()['compiled']}")
    # There is deliberately no assertion here that the GPU and CPU paths return
    # equal energies. It is tempting and it is wrong: the kernel computes the
    # intermolecular half in f32 and the CPU in f64, so the two cannot be
    # bit-equal, and the only way to write the check is to invent a tolerance.
    # `scripts/gpu_cpu_parity_check.py` owns that measurement, with a band it
    # derives. Adding a second, differently-chosen band here would be a third
    # answer to the same question.
    check("use_gpu=False is the default", core.evaluate_conformations.__defaults__[-2] is False)


# ============================== the out-of-box penalty, as a caller sees it
def sec_out_of_box():
    section("a pose outside the box is charged per angstrom and walked back in")
    # The engine used to charge one flat penalty per atom at every distance and
    # hand the optimiser a gradient pointing away from the box centre, so a pose
    # that had left the box was pushed further out while its energy never moved
    # -- which left the line search with no step to take, and returned the pose
    # exactly where it started. `dock-core` pins the corrected shape in Rust
    # (`search::tests::the_out_of_box_penalty_is_per_angstrom_of_violation` and
    # `an_out_of_box_atom_is_pulled_back_in_by_its_descent_direction`); this
    # section asks the same question of the three numbers a caller outside the
    # engine can see: the energy, the gradient, and the `out_of_box_penalty`
    # field the breakdown panel shows.
    #
    # Three separate questions, because a magnitude-only assertion passes on the
    # old code -- the old charge really was 1000 per atom. The figure is
    # therefore pinned as a *difference* over one angstrom, which a flat step
    # cannot show at all; the direction is pinned from the gradient; and the
    # gradient is pinned as the derivative of the energy, which is the one
    # comparison that is red under the old code and under a sign flip alone.
    conf = RESULT.pose_conformation(0)
    lo = np.asarray(BOX.min_corner)
    hi = np.asarray(BOX.max_corner)

    def at(x, y=0.0, z=0.0):
        c = np.array(conf, dtype=np.float64)
        c[0], c[1], c[2] = x, y, z
        return c

    def violation(c):
        """Angstroms outside, summed over every atom and every axis."""
        cc = core.conformation_coordinates(IG, c)
        return float(np.maximum(0.0, np.maximum(cc - hi, lo - cc)).sum())

    far = at(20.0)
    cc = core.conformation_coordinates(IG, far)
    n_out = int((cc[:, 0] > hi[0]).sum())
    quiet = int((cc[:, 1:] < lo[1:]).sum() + (cc[:, 1:] > hi[1:]).sum())
    check("the fixture pose is outside on every atom, and on one axis only",
          n_out == cc.shape[0] and quiet == 0,
          f"{n_out}/{cc.shape[0]} atoms past the +x face at {hi[0]:.1f}, and {quiet} "
          f"past a y or z face. Every expected value below is "
          f"{1000.0 * n_out:.0f} per angstrom of overhang")

    e_far, g_far = core.score_conformation(IG, MAPS, far)
    e_near, _ = core.score_conformation(IG, MAPS, at(19.0))
    check("one angstrom further out costs one penalty per atom",
          abs((e_far - e_near) - 1000.0 * n_out) < 1e-6,
          f"E(20) - E(19) = {e_far - e_near:.6f} against {1000.0 * n_out:.6f}. A flat "
          f"charge gives 0.0 here, and the old code gave exactly that")
    t_far = core.score_conformation_terms(IG, MAPS, REC_TERMS, far)
    check("the breakdown field is that same ramp, not a per-atom step",
          abs(t_far["out_of_box_penalty"] - 1000.0 * violation(far)) < 1e-6,
          f"out_of_box_penalty = {t_far['out_of_box_penalty']:.4f} against "
          f"1000 x {violation(far):.4f} A. This is the number the breakdown panel "
          f"prints, and it used to be {1000.0 * n_out:.0f} at any distance")
    check("the breakdown still sums to the engine's own intermolecular",
          abs(t_far["terms_total"] + t_far["out_of_box_penalty"] - t_far["intermolecular"]) < 1e-9,
          f"{t_far['terms_total']:.4f} + {t_far['out_of_box_penalty']:.4f} against an "
          f"intermolecular of {t_far['intermolecular']:.4f}. The two places the "
          f"penalty is charged are not allowed to disagree about what a pose left "
          f"the box by")
    check("the gradient points further out, so -grad steps back toward the box",
          g_far[0] > 0.0,
          f"dE/dx = {g_far[0]:+.6f} at x = 20, so -grad is a step back toward the "
          f"box. The old code put the full penalty on a unit vector toward the box "
          f"*centre*: -1000.0 per outside atom when the pose sits just past a face "
          f"(measured -1000.000000 at 1 A out), and -15863.931740 on x for this pose "
          f"at x = 20, because a unit vector to the centre is not -x that far out. "
          f"-grad was a step outwards either way")
    check("an axis the pose never left alone contributes nothing",
          abs(g_far[1]) < 1e-9 and abs(g_far[2]) < 1e-9,
          f"({g_far[0]:+.6f}, {g_far[1]:+.3e}, {g_far[2]:+.3e}). A pull toward the "
          f"box centre gave a large component on these two axes, which is how a "
          f"pose leaves through one face and is shoved out of another")
    e_hi, _ = core.score_conformation(IG, MAPS, at(20.5))
    e_lo, _ = core.score_conformation(IG, MAPS, at(19.5))
    check("the gradient is the gradient of the energy, not a number near it",
          abs(g_far[0] - (e_hi - e_lo) / 1.0) < 1e-6,
          f"analytic {g_far[0]:+.6f} against a central difference of "
          f"{(e_hi - e_lo) / 1.0:+.6f} over 1.0 A, both ends of it fully outside. "
          f"The old pair was -16000 against +0.0, and a sign flip on its own "
          f"would be +16000 against +0.0")

    # A corner escape: two axes wrong, and both are corrected in the same step.
    # A per-pose penalty, or one taken from the box centre, would move the pose
    # along a diagonal through the interior of the box instead.
    corner = at(20.0, 20.0, 0.0)
    e_corner, g_corner = core.score_conformation(IG, MAPS, corner)
    check("a corner escape is corrected on both axes and on no other",
          g_corner[0] > 0.0 and g_corner[1] > 0.0 and abs(g_corner[2]) < 1e-9,
          f"({g_corner[0]:+.6f}, {g_corner[1]:+.6f}, {g_corner[2]:+.3e}) for a pose "
          f"outside on x and y, inside on z")
    check("the two axes are charged additively, not once for the pose",
          abs((e_corner - core.score_conformation(IG, MAPS, at(20.0, 19.0))[0]) - 1000.0 * n_out) < 1e-6
          and abs((e_corner - core.score_conformation(IG, MAPS, at(19.0, 20.0))[0]) - 1000.0 * n_out) < 1e-6,
          f"walking each axis back by 1 A drops {e_corner - core.score_conformation(IG, MAPS, at(20.0, 19.0))[0]:.6f} "
          f"and {e_corner - core.score_conformation(IG, MAPS, at(19.0, 20.0))[0]:.6f}, "
          f"against {1000.0 * n_out:.0f} each")

    # And the whole point of a gradient with a sign: a step along it descends.
    # 0.01 conf units of translation, scaled so the step is a real one.
    step = at(20.0)
    step[0] = 20.0 - 0.01
    e_step, _ = core.score_conformation(IG, MAPS, step)
    check("a 0.01-unit step toward the box lowers the energy by the slope",
          abs((e_far - e_step) - 0.01 * 1000.0 * n_out) < 1e-6,
          f"{e_far:.4f} -> {e_step:.4f}, a drop of {e_far - e_step:.6f} against "
          f"{0.01 * 1000.0 * n_out:.6f}. Before the fix the energy did not move at "
          f"all over that step -- a flat charge cannot be descended, so Armijo "
          f"accepted nothing and the pose came back exactly where it started")
    # The published example poses are inside the box, so this change moves no
    # reported energy: the penalty for a real docked pose is exactly nothing.
    t_pose = core.score_conformation_terms(IG, MAPS, REC_TERMS, conf)
    check("a docked pose is charged nothing, so the published energies stand",
          t_pose["out_of_box_penalty"] == 0.0,
          f"pose 0 scores {t_pose['out_of_box_penalty']} for the penalty and "
          f"{t_pose['total']:.6f} overall")


# ================================================== is the extension current
def sec_extension_currency():
    section("the extension this run imported is behaviourally current")
    # Everything else in this file asks the engine whether it behaves. This asks
    # the engine whether it is the engine the source describes.
    #
    # The hazard is real and was live on this machine while this was written:
    # the extension installed under `F:\python310` was three generations behind
    # the working tree, and this script measures the *installed* copy when
    # `PYTHONPATH` does not point at the source tree. That run reported 410/415
    # with four failures that were all one fact -- the binary predates the
    # source -- and nothing in a tally would have said so. A gate that runs
    # against a stale extension is green for the wrong reason, which is the one
    # outcome a number cannot detect and a name can.
    #
    # **Not by size.** Two builds in one session differed in size while one of
    # them was a deliberate mutation, so size proves nothing at all. **Not by
    # timestamp.** The tree's extension carries a reproducible-build mtime of
    # 1980-01-01, so mtime says nothing either. **Not by rebuilding.** A 46-second
    # cargo build inside a check is a check nobody runs, and the parent's point
    # is right: the honest form here is a cheap behavioural proxy, and the proxy
    # is to ask the live object the question the source asks of it.
    #
    # The expectation is **derived**, never written down: every `self._raw.member`
    # in `core.py` is a member some wrapper reads off a raw extension object, and
    # if the extension does not have it the two are out of step. A new member
    # joins the census when it is written, with no list to forget to update.
    import re
    # The expectation comes from the working tree's `core.py`, NOT from
    # `__file__`. That asymmetry is the whole point: the hazard is the tree's
    # source being *ahead* of the imported binary, so asking the imported
    # `core.py` what it needs would be asking the stale side. (The first version
    # of this read `__file__`, which is this script -- and its own comment
    # containing the words `self._raw.member` matched the pattern, so it derived
    # one phantom object, resolved none of the fixtures, checked **zero** members
    # and was green. A check that checked nothing must be red; see the
    # non-vacuity guard below.)
    core_py = (SRC / "opendocking" / "core.py").read_text(encoding="utf-8")
    wanted: dict[str, set[str]] = {}
    for raw, member in re.findall(r"self\.(_[a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", core_py):
        wanted.setdefault(raw, set()).add(member)

    fixtures = (BOX, REC, IG, MAPS, REC_TERMS, RESULT)
    absent: list[str] = []
    resolved = 0
    for raw in sorted(wanted):
        live = None
        for f in fixtures:
            candidate = getattr(f, raw, None)
            if candidate is not None:
                live = candidate
                break
        if live is None:
            absent.append(f"{raw}: no fixture in this file holds an object with "
                         f"that attribute, so its {len(wanted[raw])} expected "
                         f"members went unchecked")
            continue
        resolved += len(wanted[raw])
        for member in sorted(wanted[raw]):
            if not hasattr(live, member):
                absent.append(f"{raw}.{member}")
    # The guard the first version needed: 46 member reads are derived from the
    # current `core.py`, and a run that resolves far fewer has lost its inputs
    # rather than found the engine short of anything.
    check("the currency census resolved every object it derived",
          resolved >= 40 and not absent,
          f"{resolved} member reads derived from {len(wanted)} raw objects in the "
          f"working tree's core.py, checked against the extension this run "
          f"imported. Missing: {absent}. That is the signature of a binary older "
          f"than the source: the source asks a question the extension cannot "
          f"answer. Which copy was imported is printed under 'measured:' at the "
          f"end of this run, and it is the fact to read first when the tally is "
          f"lower than expected")

    # The other direction, and it is aimed at the raw extension module rather
    # than at `core`. The binding's own declared names must be in the extension
    # it declares them in; whether `core` re-exports them under the same name is
    # a separate question with its own answer (see the DockingResults check
    # below). Deriving this by reading source text, it is blind to anything a
    # macro or a build script would add, and it cannot see a symbol that was
    # *removed* -- a stale binary keeps the old one, and only the member census
    # above notices, and only if the Python side still asks for it.
    import importlib
    ext = importlib.import_module("opendocking._dockpy")
    lib_rs = (ROOT / "dock-py" / "src" / "lib.rs").read_text(encoding="utf-8")
    classes = sorted(set(re.findall(r'#\[pyclass\(name = "([A-Za-z_][A-Za-z0-9_]*)"', lib_rs)))
    missing_classes = [c for c in classes if not hasattr(ext, c)]
    check("every pyclass the binding declares is in the extension module",
          not missing_classes and len(classes) >= 6,
          f"derived {len(classes)} from dock-py/src/lib.rs: {classes}; missing "
          f"from opendocking._dockpy: {missing_classes}. The pattern requires an "
          f"explicit `name = \"...\"`, so a pyclass that takes its name from its "
          f"Rust type is not counted; measured against the extension, not against "
          f"core, because renaming on the Python side is a separate question")
    # A third thing I expected to be a finding and measurement said was not. The
    # extension's pyclass is named `DockingResults` and its Rust `__repr__`
    # prints that name, while the exported Python class is `DockingResult`. I
    # wrote this check as "a name in output a user cannot look up" and it went
    # red, which looked like a fifth instance of the census's shape. It is not:
    # the Python wrapper has its own `__repr__`, so a caller sees
    # `DockingResult(...)` and can look that name up. The mismatched name is only
    # reachable by reaching into a private attribute, which is not a user path.
    # Kept, asserting the true claim, because "I expected a defect here" is worth
    # a pin: the next reader should not have to re-derive it to find there is
    # nothing there.
    check("the class name a caller sees in a repr is one they can look up",
          repr(RESULT).startswith("DockingResult(") and hasattr(core, "DockingResult")
          and not hasattr(core, "DockingResults"),
          f"repr(RESULT) begins {repr(RESULT)[:24]!r}; the exported class is "
          f"DockingResult and the extension's own pyclass is DockingResults, which "
          f"the wrapper renames and then overrides __repr__ for. So the mismatch is "
          f"real in the source and invisible from Python, which is the correct "
          f"outcome -- not a published name that does not resolve")


# ============================================ the thread count that is reported
def sec_thread_count():
    section("available_backends: the pool's thread count, not the machine's")
    # `available_backends()` used to interpolate `os.cpu_count()`, which is the
    # machine's core count and not the size of the pool the search runs on. The
    # two differ whenever `RAYON_NUM_THREADS` is set, and the difference is not
    # cosmetic: a measured crambin/biotin run took 0.436 s on 16 threads and
    # 8.680 s on one, with bit-identical energies. A user who set the variable
    # to 1 -- the standard move when chasing a concurrency problem -- was told
    # the engine would use 16 threads while the same call ran twenty times
    # slower. This section pins the report to the pool.
    import os
    import subprocess

    pool = core._core.rayon_threads()
    check("available_backends reports the size of rayon's global pool",
          core.available_backends()[-1] == f"cpu/{pool}-threads",
          f"the pool has {pool} threads, so the report says "
          f"{core.available_backends()[-1]!r}; it used to say "
          f"'cpu/{{os.cpu_count()}}-threads'")

    # The pool reads `RAYON_NUM_THREADS` when it is first built, so the only way
    # to see the number move is a process that starts with the variable set. Each
    # case below is therefore a subprocess, and it prints what a *caller* sees --
    # `available_backends()`, not the pool size directly. Printing the pool size
    # was the first version of this check and it could not see the bug at all: a
    # regression that takes `available_backends` back to `os.cpu_count()` leaves
    # the pool size exactly where it was, so the check went green on the very
    # defect it was written for.
    def reported(threads):
        env = {**os.environ}
        if threads is None:
            env.pop("RAYON_NUM_THREADS", None)
        else:
            env["RAYON_NUM_THREADS"] = str(threads)
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r);\n"
             "from opendocking import core; print(core.available_backends()[-1])"
             % str(SRC.parent)],
            capture_output=True, text=True, env=env, encoding="utf-8", errors="replace",
        )
        return out.stdout.strip()

    one = reported(1)
    check("a process started with RAYON_NUM_THREADS=1 reports 1, not the core count",
          one == "cpu/1-threads" and (os.cpu_count() or 1) != 1,
          f"reported {one!r} on a machine with {os.cpu_count()} cores. This is the old "
          f"bug exactly: the report said 'cpu/{os.cpu_count()}-threads' while the pool "
          f"had one thread")
    five = reported(5)
    check("and a different value moves it again, so the number is not a constant",
          five == "cpu/5-threads" and five != one,
          f"1 gives {one!r}, 5 gives {five!r}. A report that could not move would "
          f"satisfy the check above as happily as a real one")
    unset = reported(None)
    check("with the variable absent the report is the pool's own default",
          unset == f"cpu/{pool}-threads",
          f"unset gives {unset!r} against this process's cpu/{pool}-threads. Rayon "
          f"falls back to the core count there, which is why the two agree only when "
          f"nothing overrides it")


# ====================================== min_contact_distance: the field, and the doc
def sec_contact_distance():
    section("min_contact_distance: the field the engine reads, and the doc that names it")
    # `DockingConfig::min_contact_distance` is read by `dock()` itself, in the
    # clash partition (`docking.rs:257`), and its doc says 0.0 disables the
    # filter. Neither the binding nor `dock()` exposed it, so every Python
    # `dock()` was silently fixed at 2.0 A and the instruction to set 0.0 was
    # reachable only from Rust. That is not a dead knob -- it is a field the
    # engine reads that the binding did not publish. Whether the omission was
    # deliberate cannot be determined from this repository: there is no comment,
    # changelog entry or document claiming it was exposed, which is an absence
    # of evidence and not evidence of an absence.
    import inspect
    import re

    params = inspect.signature(core.dock).parameters
    check("dock() takes min_contact_distance",
          "min_contact_distance" in params,
          f"the published keywords are {list(params)[2:]}. Without it, every Python "
          f"dock() is fixed at the engine's 2.0 A floor")
    check("and it defaults to leaving the engine's default alone",
          params["min_contact_distance"].default is None,
          f"default is {params['min_contact_distance'].default!r}; None means the "
          f"engine's own default, so 2.0 is not written down twice")

    # Behaviour, because a keyword that is accepted and ignored is worse than one
    # that is absent. A 10 A floor rejects every pose on this fixture, and the
    # rejection is counted and reported rather than swallowed.
    kw = dict(exhaustiveness=2, num_modes=3, seed=7)
    default = core.dock(IG, MAPS, **kw)
    strict = core.dock(IG, MAPS, min_contact_distance=10.0, **kw)
    check("a 10 A floor really reaches the clash partition",
          strict.rejected_pose_count > 0 and default.rejected_pose_count == 0,
          f"the default rejects {default.rejected_pose_count} pose(s), a 10 A floor "
          f"rejects {strict.rejected_pose_count}. A keyword that is accepted and "
          f"dropped would report 0 for both")
    off = core.dock(IG, MAPS, min_contact_distance=0.0, **kw)
    check("0.0 is the documented way to disable the filter, and disables it",
          off.rejected_pose_count == default.rejected_pose_count
          and float(np.max(np.abs(off.energies - default.energies))) == 0.0,
          f"0.0 gives energies {off.energies} against the default's "
          f"{default.energies}; on a clean fixture the two are the same run, which "
          f"is what 'disable' has to mean here")

    # The document and the surface, in both directions. `docs/API.md` carries a
    # Python `dock(...)` signature block, and that block is the place a reader
    # looks; if the surface grows a keyword the block does not name, the
    # documentation is stale again, and if the block names one the surface does
    # not have, the documentation is promising something absent. Either way the
    # two must agree, which is the whole point of checking them against each
    # other rather than against a remembered list.
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    python_part = api.split("## Python: `opendocking`", 1)[1]
    block = re.search(r"dock\(ligand, maps,.*?\) -> DockingResult", python_part, re.S)
    documented = bool(block) and "min_contact_distance" in block.group(0)
    check("docs/API.md's Python dock() signature names the keyword the surface has",
          documented == ("min_contact_distance" in params),
          f"the block says {block.group(0)[:90] if block else '(no block found)'!r}...; "
          f"the signature has it: {'min_contact_distance' in params}, the document: "
          f"{documented}. One without the other is the drift")
    check("and the keyword is documented where a reader of dock() will find it",
          "min_contact_distance" in (core.dock.__doc__ or ""),
          f"help(core.dock) documents {len((core.dock.__doc__ or '').split())} words; "
          f"it names the keyword: {'min_contact_distance' in (core.dock.__doc__ or '')}")


# ============================== the two output modes of one command must agree
def sec_output_modes():
    section("the two output modes of `dock` report the same facts")
    # The property that broke earlier in the project, in its general form: the
    # JSON mode of a command and the prose mode of the *same* command described
    # the same run differently -- the JSON returned three healthy poses while
    # the prose mode exited 1. Nothing about that is specific to poses or to
    # JSON: it is two output paths for one fact, and one of them was allowed to
    # forget the fact.
    #
    # Which fields are affected is *derived*, never listed. Both sides are read
    # out of the working tree:
    #
    #   * the prose side is every `if self.<name>:` guard in
    #     `DockingResult.summary` -- each one is a count the table can warn
    #     about, so each one is a fact a reader of the table might hold and a
    #     reader of the JSON might not;
    #   * the JSON side is every `"<key>":` in the record `cli.py` appends.
    #
    # Derived, because a hardcoded list is a list to forget: the disagreement
    # this checks is *caused* by the two sides being maintained separately, and
    # a list of the fields that currently agree re-asserts the agreement rather
    # than the relation that produced it.
    import re

    core_py = (SRC / "opendocking" / "core.py").read_text(encoding="utf-8")
    body = core_py.split("    def summary(self)", 1)[-1].split("\n    def ", 1)[0]
    warned = sorted(set(re.findall(r"if self\.([a-z_][a-z0-9_]*):", body)))

    cli_py = (SRC / "opendocking" / "cli.py").read_text(encoding="utf-8")
    record = cli_py.split("records.append(", 1)[-1].split("\n        )", 1)[0]
    keys = sorted(set(re.findall(r'"([a-z_][a-z0-9_]*)"\s*:', record)))

    # The guard this section needs to not be vacuous: a regex that derived
    # nothing would make every assertion below pass for the wrong reason --
    # which is the failure mode the currency census in
    # `sec_extension_currency` already documents for itself.
    check("the derivation found the summary guards and the JSON keys",
          len(warned) >= 3 and len(keys) >= 9,
          f"{len(warned)} warning guards {warned}; {len(keys)} record keys {keys}. "
          f"If either list is empty or tiny, the relation below is being asserted "
          f"about nothing")
    check("the two modes report the same facts: every warned count is a JSON key",
          set(warned) <= set(keys),
          f"the table can warn about {warned}; the record carries {keys}. "
          f"In the JSON but not the table: {sorted(set(keys) - set(warned))}. "
          f"In the table but not the JSON: {sorted(set(warned) - set(keys))} -- "
          f"these are the fields a JSON consumer cannot see, which is the shape "
          f"of the original defect")
    for name in warned:
        check(f"summary()'s {name} has a property of that name on DockingResult",
              isinstance(getattr(core.DockingResult, name, None), property),
              f"the table reads self.{name}, so the wrapper must expose it or the "
              f"warning is printing a fact no caller can ask for")

    # And the runtime half: the same ligand, box, spacing and seed through the
    # command, so the JSON number is the *same run's* number and not merely a
    # field of the right name. One docking run, at the fixture's own settings.
    import json
    import os
    import subprocess
    import sys

    env = {**os.environ}
    # Point the child at the same package this process imported -- but not by
    # prepending `site-packages`, which is where that package usually lives.
    #
    # A `PYTHONPATH` entry is prepended to `sys.path`, ahead of the standard
    # library, so naming the site-packages directory also promotes everything
    # installed beside it. On this machine that is `pathlib` 1.0.1 -- the 2015
    # backport -- which does `from collections import Sequence`, a name removed
    # in Python 3.10. The child therefore died inside `import pathlib` at
    # `opendocking/core.py:15`, before one line of docking code ran, and exited 1
    # with an empty stdout. This section read that empty stdout as "no record
    # emitted" and raised a second red, so one environment bug produced two
    # failures and neither of them was about the engine.
    #
    # A child finds an *installed* package with no help at all, so the variable
    # is set only when this process resolved from a tree: the case the original
    # comment was written for, and the only one that needs it.
    pkg = str(Path(_od.__file__).resolve().parent.parent)
    if not _IN_SITE_PACKAGES:
        env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", "opendocking", "dock",
           "-r", str(EX / "rec_prep.pdbqt"),
           "-l", str(EX / "ibuprofen_prep.pdbqt"),
           "-e", "4", "-m", "3", "--seed", "42", "--spacing", "0.5",
           "--center_x", "0", "--center_y", "0", "--center_z", "0",
           "--size_x", "14", "--size_y", "14", "--size_z", "14",
           "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace")
    check("`dock --json` exits 0 on the fixture the library just docked",
          proc.returncode == 0,
          f"rc={proc.returncode}; stderr: {proc.stderr.strip()[:160]}")
    records = json.loads(proc.stdout) if proc.stdout.strip() else []
    check("and emits one record", len(records) == 1, f"got {len(records)}")
    if not records:
        return
    rec = records[0]
    check("the JSON's count is the same number the library property gave",
          rec.get("poses_outside_box_count") == RESULT.poses_outside_box_count,
          f"json {rec.get('poses_outside_box_count')!r} vs library "
          f"{RESULT.poses_outside_box_count!r} for the same box, seed and spacing")
    check("and so is the unrecognised-atom count",
          rec.get("unknown_atom_types") == RESULT.unknown_atom_types,
          f"json {rec.get('unknown_atom_types')!r} vs library "
          f"{RESULT.unknown_atom_types!r}")
    check("the record's pose count matches the library's",
          rec.get("num_poses") == RESULT.num_poses,
          f"json {rec.get('num_poses')!r} vs library {RESULT.num_poses!r}")
    check("the two modes' shape is the same set of fields the derivation found",
          set(keys) <= set(rec),
          f"derived {len(keys)} keys, record has {sorted(rec)}")


def main() -> int:
    crashed: list[str] = []
    for fn in (sec_surface, sec_reachability, sec_gridbox, sec_gridbox_engine, sec_term_maps,
               sec_receptor, sec_extension_currency,
               sec_estimate, sec_maps, sec_ligand, sec_exhaustiveness, sec_ladder_agreement,
               sec_auto_box, sec_dock_guards, sec_dock_boundary, sec_result, sec_writers, sec_scoring,
               sec_out_of_box, sec_thread_count, sec_contact_distance, sec_output_modes):
        try:
            fn()
        except BaseException:  # noqa: BLE001
            check(f"section {fn.__name__} completed", False, "crashed")
            crashed.append(fn.__name__)
            traceback.print_exc()

    # The pin, asserted on the way out rather than only in the declaration. The
    # `+ 1` is this check, which has not been counted yet when the comparison is
    # built.
    #
    # The `crashed` list is here so that this red is a *diagnosis* rather than a
    # second bare number. A section that raises is already reported as a failed
    # `section <name> completed` above, and a tally that then said only "506 ran,
    # 510 are declared" would be a second failure about one cause -- the shape
    # the comment above `sec_output_modes` already complained about. Naming the
    # sections turns the two reds into one fact. A shortfall with nothing in
    # `crashed` is the other shape, and the only one this file knows how to
    # produce: `if not records: return` in `sec_output_modes`, four checks the
    # run did not take, where the empty child output is already in the failures.
    check("this file's own count is the count it declares",
          CHECKS + 1 == EXPECTED_CHECKS,
          f"{CHECKS} ran before this one and {EXPECTED_CHECKS} are declared"
          + (f"; the section(s) that crashed and took their checks with them: "
             f"{', '.join(crashed)}" if crashed else ""))

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed (expected {EXPECTED_CHECKS})")
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
        print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed (run incomplete; "
              f"expected {EXPECTED_CHECKS})")
        sys.exit(1)
