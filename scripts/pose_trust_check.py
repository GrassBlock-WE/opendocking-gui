"""Gate for the pose verdict panel.

The panel answers "can *this* pose's numbers be trusted?", and the claim it can
most easily get wrong is the one that looks fine: a panel describing the
*previous* pose under the new pose's name is a complete, plausible-looking
picture. So this gate is built around asking two things independently and
comparing them:

1. **The module and the product, asked separately, about the same pose.** The
   gate builds its own values dict from the `DockingResult` the window holds --
   its own `|grad|_2` loop, its own formatting, its own tolerance -- asks
   `result_trust.verdict_pose` about it, and only then reads what the panel
   drew. Comparing the panel against the module *through the panel* would
   prove the panel is self-consistent, which is a different and much weaker
   claim.
2. **Three mutations, each rejected by the same predicate that accepts the
   shipped rule.** Wrong pose first, because it is the one a reader cannot see;
   then a failing contract that has lost its remedy; then `unknown` rendered in
   the pass treatment. Every mutation is bidirectional: the shipped code is
   accepted by the predicate before and after, so a predicate that rejects
   everything is not mistaken for a predicate that works.

A third property is checked without a window at all, because it is the one that
decides whether the panel is safe to trust: **the keys are not typed.** Every
key the panel binds is checked against `result_trust.schema_for("pose")`, and
each bound key is then shown to be the one its contract actually reads -- by
withholding it and watching exactly the named contract go `unmeasured`. A gate
that fed the module dicts whose keys the gate itself wrote would be a test
written in the same vocabulary as the code it tests, which proves the two agree
with each other and not that either agrees with the engine.

`GRADIENT_TOLERANCE` is a transcription of `LbfgsConfig::gradient_tolerance`,
because the engine exposes no Python binding for it. Section 1 reads
`dock-core/src/search/lbfgs.rs` and asserts the number is the engine's own, so
an edit to the optimiser's stopping test goes red here rather than quietly
changing every verdict the panel prints.

Run:  set PYTHONPATH to dock-py/python and run this file by path.
"""
from __future__ import annotations

import dataclasses
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dock-py" / "python"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt6 import QtCore, QtWidgets  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

import result_trust as rt  # noqa: E402  (the module under test, imported by name)

import opendocking.workbench.app as appmod  # noqa: E402
from opendocking.workbench.app import MainWindow  # noqa: E402
from opendocking.workbench import pose_trust as pt  # noqa: E402

EXAMPLES = ROOT / "examples"
OUT = ROOT / "dist" / "pose_trust"
OUT.mkdir(parents=True, exist_ok=True)

LIGAND = "biotin_prep.pdbqt"
RECEPTOR = "1crn_prep.pdbqt"
BOX_CENTER = (18.0, 12.0, 20.0)
BOX_SIZE = (22.0, 22.0, 22.0)
SEED = 20260901
TOLERANCE_SOURCE = ROOT / "dock-core" / "src" / "search" / "lbfgs.rs"

#: How many checks this file is supposed to run, in every environment.
#:
#: A `skip` counts toward the total, on purpose: a total that moves with the
#: machine is not a total, and a gate that reaches its number by declining to
#: check anything is the failure this repository keeps meeting. Sections 3 and
#: 5 need a window and a QApplication; on a machine that has neither they
#: record a skip per check, and the pin below is the same number either way.
#:
#: The derivation, so a future change to it is deliberate:
#:
#:   7  section 1: the binding reads the objects, and the keys are not typed
#: + 6  section 2: three states, and the remedy rides with a failure
#: + 9  section 3: the panel describes the pose that is selected, and the
#:      wrong-pose mutation is rejected by the same predicate
#: + 1  section 4: a failing contract that lost its remedy is rejected
#: + 5  section 5: a pose from a file is an honest `unknown`, and `unknown`
#:      is never rendered in the pass treatment
#: + 2  the two meta-checks, which count themselves
#:
#: 7 + 6 + 9 + 1 + 5 + 2 = 30. Section 1 is seven rather than six because it
#: opens with the fixture check -- the run has to have produced poses with
#: gradients before any of the rest of the file means anything, and a gate that
#: quietly judged nothing would still reach its number.
EXPECTED_CHECKS = 30

EXPECTED_SECTIONS = (
    "1. the binding reads the objects, and the keys are not typed",
    "2. three states, and the remedy rides with a failure",
    "3. the panel describes the pose that is selected",
    "4. a failing contract that lost its remedy is not a warning light",
    "5. a pose from a file is an honest unknown, never a pass",
    "6. this file ran all of it",
)

#: GATE-DECLARE 1
#: sites: 16 unconditional + 14 guarded
#: guards: sha256:3570a3d256d621b9ce700bc7980a19f3b4a41a125061bfea24b95b925c3b1904
#:
#: 16 + 14 = 30, and it matches `EXPECTED_CHECKS` because every site here runs
#: exactly once: there is no `check()` inside a loop, and a site that ran twice
#: would be a count the declaration could not see.
#:
#: The fourteen guarded sites are sections 3 and 5, and the guard is the same
#: condition for both: whether a QApplication can be created at all. That is not
#: laziness, it is the point of section 5. A real `unknown` has to be **on
#: screen** to be checked for the pass treatment, because the mutation being
#: rejected there is a styling mutation, and a styling mutation cannot be seen
#: from a dataclass -- `PoseVerdict` carries the state and the rows, and only
#: the panel carries the colours. A gate that judged the third state from the
#: model alone would pass on a panel that painted `unknown` green, which is the
#: exact failure it was written to catch.
#:
#: So sections 1, 2 and 4 -- sixteen sites -- run on a machine with no display:
#: the binding, the key provenance, the three states, the precedence rule, the
#: remedy, and the remedy-loss mutation are all answerable without a window.
RESULTS: list[tuple[str, str, str]] = []
FAILURES: list[str] = []
_sections: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return bool(ok)


def skip(name: str, reason: str):
    """Record a check this machine could not answer, and say why.

    Counts toward `EXPECTED_CHECKS`, so the total is the same number on a
    machine with no display and on one with a GL context.
    """
    RESULTS.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  - {reason}")
    return False


def section(title: str) -> None:
    print(f"\n=== {title} ===")
    _sections.append(title)


def settle(app, ms: int = 300) -> None:
    """Let the window finish what it started, on the clock.

    The camera move is a wall-clock `QTimer` and `processEvents()` does not
    advance one, so a wait here is a wait on time rather than on iterations.
    """
    QTest.qWait(ms)
    for _ in range(6):
        app.processEvents()


# --------------------------------------------------------------------------
# What this gate computes for itself, from the objects, with its own code
# --------------------------------------------------------------------------
def grad_l2_of(result, index: int) -> float:
    """The pose's own gradient norm, computed here rather than imported.

    Deliberately not `pt.describe_pose`'s arithmetic: this is the independent
    half of the comparison, and a shared helper would make the two sides agree
    by construction instead of by measurement.
    """
    gradient = result.pose_gradient(int(index))
    if gradient is None:
        return float("nan")
    total = 0.0
    for value in gradient:
        v = float(value)
        total += v * v
    return total ** 0.5


def gate_values(result, index: int, penalty=None) -> dict:
    """This gate's own values dict for one pose, spelled from the schema."""
    schema = set(rt.schema_for("pose"))
    values = {"label": f"pose {int(index) + 1} of {int(result.num_poses)}"}
    assert "grad_l2" in schema and "gradient_tolerance" in schema, schema
    values["grad_l2"] = grad_l2_of(result, index)
    values["gradient_tolerance"] = pt.GRADIENT_TOLERANCE
    if penalty is not None:
        assert "out_of_box_penalty" in schema, schema
        values["out_of_box_penalty"] = float(penalty)
    return values


def fmt(value) -> str:
    """The gate's own formatting, matching the panel's by measurement.

    Asserted equal in section 2 rather than shared: if the two spellings ever
    differ, the reader sees one and the gate compares against the other.
    """
    return "not measured" if value is None else f"{float(value):.6g}"


def panel_rows(win) -> dict:
    """What the panel is showing, read off the widgets rather than off state."""
    rows = {}
    for r in range(win.tbl_verdict.rowCount()):
        name = win.tbl_verdict.item(r, 0)
        state = win.tbl_verdict.item(r, 1)
        pair = win.tbl_verdict.item(r, 2)
        if None not in (name, state, pair):
            rows[name.text()] = (state.text(), pair.text())
    return rows


def panel_head(win) -> str:
    return win.lbl_verdict_head.text()


def panel_colour(item) -> str:
    brush = item.foreground()
    return brush.color().name() if brush.style() else ""


# -- the three predicates the mutations are asked through --------------------
def describes_selection(win, index: int) -> tuple[bool, str]:
    """Is the panel on screen about pose ``index``, the selected one?

    The header and the window's own record, and both of them rather than
    either: a header that names the right pose while the record says otherwise
    is a panel that was refreshed twice, which is how a late answer from a
    retired build ends up on top of a newer selection.
    """
    head = panel_head(win)
    named = f"pose {int(index) + 1} of " in head
    recorded = win._verdict_pose_index == int(index)
    current = win._current_pose == int(index)
    why = (
        f"header {head[:110]!r}; _verdict_pose_index {win._verdict_pose_index}, "
        f"_current_pose {win._current_pose}, selected {int(index)}"
    )
    return (named and recorded and current), why


def failing_rows_have_remedies(verdict, text: str) -> tuple[bool, str]:
    """Every failing contract shows a remedy, with its kind, in ``text``.

    ``text`` is the string the reader is given -- the panel's own label on
    screen, or `PoseVerdict.explanation()` for a headless call. A remedy that
    exists in the object and never reaches the reader is the same defect as one
    that was never computed.
    """
    failing = [r for r in verdict.rows if r.state == "fails"]
    if not failing:
        return False, "no contract failed, so there is no remedy to show"
    missing = [r.name for r in failing if not r.remedy.strip()]
    untyped = [r.name for r in failing if not r.remedy_kind.strip()]
    unshown = [r.name for r in failing if r.remedy and r.remedy not in text]
    ok = not (missing or untyped or unshown)
    why = (
        f"{len(failing)} failing contract(s) {[r.name for r in failing]}; "
        f"remedy missing: {missing or 'none'}; remedy kind missing: "
        f"{untyped or 'none'}; remedy not on screen: {unshown or 'none'}. "
        f"A warning light with no dashboard is not actionable"
    )
    return ok, why


def unknown_is_not_a_pass(win) -> tuple[bool, str]:
    """Is the panel's `unknown` wearing anything but the pass treatment?

    Read off the widgets: the header's stylesheet and the colour of the word in
    every row that says `unmeasured`. Both channels, because the mutation being
    rejected here could be either one, and a panel that agreed with itself in
    one channel while a reader reads the other is exactly the failure.
    """
    head = win.lbl_verdict_head.text()
    unknown_rows = [
        win.tbl_verdict.item(r, 1)
        for r in range(win.tbl_verdict.rowCount())
        if win.tbl_verdict.item(r, 0) is not None
        and win.tbl_verdict.item(r, 1) is not None
        and win.tbl_verdict.item(r, 1).text() == "unmeasured"
    ]
    if "UNKNOWN" not in head.upper() or not unknown_rows:
        return False, (
            f"header {head[:110]!r} and {len(unknown_rows)} row(s) reading "
            f"'unmeasured'; there is no unknown state on screen to check"
        )
    sheet = win.lbl_verdict_head.styleSheet()
    pass_sheet = pt.stylesheet_for("yes")
    same_sheet = pass_sheet in sheet
    pass_colour = pt.STATE_COLOURS["yes"].lower()
    row_colours = [panel_colour(item).lower() for item in unknown_rows]
    same_colour = [c for c in row_colours if c == pass_colour]
    ok = not same_sheet and not same_colour
    why = (
        f"header stylesheet {sheet!r} against the pass {pass_sheet!r} "
        f"({'MATCHES' if same_sheet else 'differs'}); the {len(unknown_rows)} "
        f"'unmeasured' row words are painted {sorted(set(row_colours))} and the "
        f"pass colour is {pass_colour} "
        f"({'A MATCH' if same_colour else 'not among them'})"
    )
    return ok, why


def wait_for_terms(win, app, budget_ms: int = 15000) -> bool:
    """Process events until the breakdown for the selected pose has landed.

    The first selection for a box pays for the term maps on a worker thread,
    and the verdict panel's out-of-box penalty comes from that breakdown, so a
    check that read the verdict immediately would read it before the one
    contract that depends on the maps could be anything but `unmeasured`.
    Returns whether it arrived, so a build that never delivers is a different
    answer from a slow one.
    """
    waited = 0
    while waited < budget_ms:
        if win._terms_pose_index == win._current_pose and win._current_pose >= 0:
            return True
        QTest.qWait(20)
        app.processEvents()
        waited += 20
    return win._terms_pose_index == win._current_pose and win._current_pose >= 0


def main() -> int:
    schema = sorted(rt.schema_for("pose"))

    # ======================================================================
    section(EXPECTED_SECTIONS[0])
    from opendocking.core import GridBox, Receptor, dock, load_ligand

    box = GridBox.from_center_size(BOX_CENTER, BOX_SIZE)
    receptor = Receptor.from_pdbqt(EXAMPLES / RECEPTOR)
    maps = receptor.precalculate(box, "vina", 0.375)
    ligand = load_ligand(EXAMPLES / LIGAND)
    result = dock(ligand, maps, exhaustiveness=2, num_modes=9, rmsd_cutoff=1.0,
                  seed=SEED, scoring="vina")
    gradients = [grad_l2_of(result, i) for i in range(int(result.num_poses))]
    check(
        "the run reported poses with gradients, so there is something to judge",
        result.num_poses >= 2
        and all(math.isfinite(g) and g > 0.0 for g in gradients),
        f"{result.num_poses} poses; |grad|2 = "
        + ", ".join(f"{g:.4e}" for g in gradients)
        + ". All nine carry a gradient, so 'unknown' is not reachable from an "
          "in-memory pose on this fixture -- it is reached from a pose file, "
          "which section 5 shows",
    )
    bound = set(pt._READERS)
    check(
        "every key the panel binds is one the verdict module actually reads",
        bound <= set(schema),
        f"the panel binds {sorted(bound)}; schema_for('pose') is {schema}; "
        f"keys the module would report as misspelled: "
        f"{sorted(bound - set(schema)) or 'none'}",
    )
    # The next three ask, one key at a time, *which contract* that key decides.
    # A schema is a list of names and a name is not a binding: a panel can pass
    # a key the module reads and still be handing it the wrong quantity, and
    # the only way to see that is to withhold the key and watch which contract
    # goes `unmeasured`.
    base = gate_values(result, 0)
    base_unmeasured = {
        c.name for c in rt.verdict_pose(base).contracts if c.state == "unmeasured"
    }
    without_grad = dict(base)
    without_grad.pop("grad_l2")
    flipped = {c.name for c in rt.verdict_pose(without_grad).contracts
               if c.state == "unmeasured"}
    check(
        "'grad_l2' is the key stationarity reads",
        flipped - base_unmeasured == {"stationarity"},
        f"withholding grad_l2 makes newly unmeasured: "
        f"{sorted(flipped - base_unmeasured)}; already unmeasured before the "
        f"withholding: {sorted(base_unmeasured)}. Exactly one contract changed, "
        f"which is what 'this key is that contract's input' means",
    )
    without_tol = dict(base)
    without_tol.pop("gradient_tolerance")
    flipped = {c.name for c in rt.verdict_pose(without_tol).contracts
               if c.state == "unmeasured"}
    check(
        "'gradient_tolerance' is the other key stationarity reads",
        flipped - base_unmeasured == {"stationarity"},
        f"withholding gradient_tolerance makes newly unmeasured: "
        f"{sorted(flipped - base_unmeasured)}. A measured gradient with nothing "
        f"to compare against is also unmeasured, which is why this one cannot "
        f"be defaulted to zero or to the gradient itself",
    )
    with_penalty = gate_values(result, 0, 0.0)
    states = {c.name: c.state for c in rt.verdict_pose(with_penalty).contracts}
    check(
        "'out_of_box_penalty' is the key the in-box contract reads",
        states.get("inside the box") == rt.HOLDS
        and states.get("stationarity") == rt.FAILS,
        f"with out_of_box_penalty = 0.0 the in-box contract leaves unmeasured "
        f"and holds: {states}. The three bound keys are three different "
        f"contracts' inputs rather than three spellings of one",
    )
    # A caller that misspells a key gets a loud answer, not a confident
    # `unknown`. The panel refuses such a key itself, and this says what would
    # happen if it did not: the module still refuses to let the typo read as a
    # measurement gap.
    typo = dict(base)
    typo["grad_norm"] = float(base["grad_l2"])
    misspelled = rt.verdict_pose(typo)
    check(
        "a misspelled key is reported rather than read as a missing measurement",
        bool(misspelled.misspelled)
        and misspelled.trust != rt.YES
        and "grad_norm" in misspelled.summary,
        f"misspelled {misspelled.misspelled}, trust {misspelled.trust!r}, "
        f"summary names the key: {'grad_norm' in misspelled.summary}",
    )
    # The one transcribed number, checked against the engine's own source.
    lbfgs = TOLERANCE_SOURCE.read_text(encoding="utf-8")
    engine_tolerance = None
    for line in lbfgs.splitlines():
        stripped = line.strip()
        if stripped.startswith("gradient_tolerance:"):
            engine_tolerance = float(stripped.split(":", 1)[1].strip().rstrip(","))
            break
    check(
        "the tolerance the panel prints is the engine's own, read from its "
        "source rather than remembered",
        engine_tolerance is not None
        and float(pt.GRADIENT_TOLERANCE) == engine_tolerance,
        f"{TOLERANCE_SOURCE.name} declares gradient_tolerance "
        f"{engine_tolerance}, the panel uses {pt.GRADIENT_TOLERANCE}. The engine "
        f"exposes no Python binding for it, so this transcription is the only "
        f"way to have the number -- which is why it is checked here",
    )

    # ======================================================================
    section(EXPECTED_SECTIONS[1])
    shown = pt.describe_pose(result, 0, out_of_box_penalty=0.0, penalty_pose_index=0)
    check(
        "the panel's three treatments are three different treatments",
        len({pt.stylesheet_for(s) for s in ("yes", "no", "unknown")}) == 3
        and len({pt.STATE_COLOURS[s] for s in ("yes", "no", "unknown")}) == 3,
        f"yes {pt.stylesheet_for('yes')!r}; no {pt.stylesheet_for('no')!r}; "
        f"unknown {pt.stylesheet_for('unknown')!r}. An unrecognised state "
        f"renders as unknown ({pt.stylesheet_for('bogus')!r}), never as a pass",
    )
    check(
        "and the unmeasured treatment is not the pass treatment in either "
        "channel",
        pt.stylesheet_for("unknown") != pt.stylesheet_for("yes")
        and "bold" not in pt.stylesheet_for("unknown")
        and pt.STATE_COLOURS["unknown"] != pt.STATE_COLOURS["yes"],
        f"unknown is {pt.stylesheet_for('unknown')!r} and yes is "
        f"{pt.stylesheet_for('yes')!r}; the words are also different "
        f"({pt.STATE_WORDS['unknown']!r} against {pt.STATE_WORDS['yes']!r}), so "
        f"the verdict reads with the colour removed",
    )
    unmeasured_names = [r.name for r in shown.rows if r.state == "unmeasured"]
    check(
        "a contract nothing measured is shown as unmeasured, and not renamed",
        unmeasured_names == ["line-search resolution", "field support"]
        and all(
            r.word == "unmeasured" for r in shown.rows
            if r.state == "unmeasured"
        ),
        f"pose 1 of {shown.num_poses}: unmeasured {unmeasured_names}, and the "
        f"engine does not expose a descent width or a support edge, so those "
        f"two stay unmeasured on every pose rather than being passed",
    )
    check(
        "a known failure outranks 'unmeasured', and does not soften into it",
        shown.trust == rt.NO
        and bool(shown.failing_rows())
        and bool(shown.unmeasured_rows())
        and rt.verdict_pose(base).trust == rt.NO
        and rt.verdict_pose(without_grad).trust == rt.UNKNOWN,
        f"with the gradient the verdict is {shown.trust!r} (failing "
        f"{[r.name for r in shown.failing_rows()]}, unmeasured "
        f"{[r.name for r in shown.unmeasured_rows()]}); withholding it makes it "
        f"{rt.verdict_pose(without_grad).trust!r}. The failure is known, so the "
        f"answer is 'no' and not the polite word",
    )
    check(
        "a failing contract carries a remedy, its kind, and the remedy is on "
        "screen",
        failing_rows_have_remedies(shown, shown.explanation())[0],
        failing_rows_have_remedies(shown, shown.explanation())[1],
    )
    check(
        "and a contract that merely holds carries no remedy at all",
        all(not r.remedy for r in shown.rows if r.state != "fails"),
        f"remedies present on: "
        f"{[r.name for r in shown.rows if r.remedy] or 'none'}. A remedy on a "
        f"passing contract is noise, and a reader learns to ignore the field",
    )

    # ======================================================================
    section(EXPECTED_SECTIONS[2])
    app = None
    no_window = ""
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception as exc:  # noqa: BLE001 - no display, or no platform plugin
        no_window = f"no QApplication on this machine: {type(exc).__name__}: {exc}"
    if app is None:
        for name in (
            "the window's own run produced poses to judge",
            "the panel names the pose that is selected",
            "and the window recorded that same pose as the one on show",
            "the panel's numbers are a second, independent read of that pose's "
            "own gradient",
            "and the module, asked separately about the same pose, agrees",
            "the numbers change with the pose, so this is not one pose's "
            "verdict wearing two names",
            "the remedy for the failing contract is on screen, not only in the "
            "object",
            "the screenshots exist: a 'no', and the claim of which pose",
            "the wrong-pose panel is rejected by the same predicate that "
            "accepts the shipped one",
        ):            skip(name, no_window)
        win = None
    else:
        win = MainWindow(receptor=EXAMPLES / RECEPTOR, ligand=EXAMPLES / LIGAND)
        win.resize(1180, 980)
        win.show()
        for _ in range(20):
            app.processEvents()
        for spin, value in zip(win.center_spins, BOX_CENTER):
            spin.setValue(value)
        for spin, value in zip(win.size_spins, BOX_SIZE):
            spin.setValue(value)
        win.sp_exhaust.setValue(2)
        win.sp_seed.setValue(SEED)
        app.processEvents()
        win.btn_dock.click()
        for _ in range(1200):
            app.processEvents()
            if win.btn_dock.isEnabled():
                break
            QTest.qWait(10)
        settle(app, 800)
        check(
            "the window's own run produced poses to judge",
            win.pose_table.rowCount() >= 2,
            f"{win.pose_table.rowCount()} rows in the pose table, "
            f"_dock_result {win._dock_result is not None}",
        )

        def select_by_energy(n: int) -> int:
            """Select the n-th pose by energy, lowest first, and return its index."""
            win.pose_table.sortItems(1, QtCore.Qt.SortOrder.AscendingOrder)
            app.processEvents()
            row = win.pose_table.rowCount() - 1 - n
            win.pose_table.setCurrentCell(row, 0)
            wait_for_terms(win, app)
            return int(
                win.pose_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            )

        idx_a = select_by_energy(0)
        head_a = panel_head(win)
        rows_a = panel_rows(win)
        win.verdict_box.grab().save(str(OUT / "pose_verdict_no.png"))
        ok, why = describes_selection(win, idx_a)
        check("the panel names the pose that is selected", ok, why)
        check(
            "and the window recorded that same pose as the one on show",
            win._verdict_pose_index == idx_a == win._current_pose,
            f"_verdict_pose_index {win._verdict_pose_index}, _current_pose "
            f"{win._current_pose}, selected {idx_a}; header {head_a[:90]!r}",
        )
        # The independent half: this gate's own |grad|_2, its own formatting,
        # its own verdict call. Nothing below reads the panel's own arithmetic.
        l2_a = grad_l2_of(win._dock_result, idx_a)
        expected_pair = f"{fmt(l2_a)} / {fmt(pt.GRADIENT_TOLERANCE)}"
        check(
            "the panel's numbers are a second, independent read of that pose's "
            "own gradient",
            rows_a.get("stationarity", ("", ""))[1] == expected_pair
            and l2_a > pt.GRADIENT_TOLERANCE,
            f"the gate read |grad|2 = {l2_a:.6e} from "
            f"DockingResult.pose_gradient({idx_a}) and formatted "
            f"{expected_pair!r}; the panel shows "
            f"{rows_a.get('stationarity')!r}. The gradient is "
            f"{l2_a / pt.GRADIENT_TOLERANCE:.0f}x the optimiser's declared "
            f"tolerance, so stationarity must fail",
        )
        mine = rt.verdict_pose(gate_values(win._dock_result, idx_a, 0.0))
        check(
            "and the module, asked separately about the same pose, agrees",
            mine.trust == rt.NO
            and mine.failed() == ("stationarity",)
            and dict(
                (c.name, c.state) for c in mine.contracts
            ) == dict((n, s) for n, (s, _p) in rows_a.items()),
            f"the gate's own call: trust {mine.trust!r}, failed {mine.failed()}, "
            f"unmeasured {mine.unknown()}; the panel's rows say {rows_a}",
        )
        check(
            "the remedy for the failing contract is on screen, not only in the "
            "object",
            failing_rows_have_remedies(shown, win.lbl_verdict_why.text())[0],
            f"lbl_verdict_why: {win.lbl_verdict_why.text()[:220]!r}",
        )
        idx_b = select_by_energy(2)
        head_b = panel_head(win)
        rows_b = panel_rows(win)
        l2_b = grad_l2_of(win._dock_result, idx_b)
        win.lbl_verdict_head.grab().save(str(OUT / "pose_verdict_which_pose.png"))
        check(
            "the numbers change with the pose, so this is not one pose's "
            "verdict wearing two names",
            idx_b != idx_a
            and rows_a.get("stationarity") != rows_b.get("stationarity")
            and rows_b.get("stationarity", ("", ""))[1]
            == f"{fmt(l2_b)} / {fmt(pt.GRADIENT_TOLERANCE)}",
            f"pose {idx_a + 1}: {rows_a.get('stationarity')!r} against the "
            f"gate's own |grad|2 {l2_a:.6e}; pose {idx_b + 1}: "
            f"{rows_b.get('stationarity')!r} against the gate's own "
            f"{l2_b:.6e}. Header for pose {idx_b + 1}: {head_b[:80]!r}",
        )
        check(
            "the screenshots exist: a 'no', and the claim of which pose",
            all((OUT / name).is_file() for name in (
                "pose_verdict_no.png", "pose_verdict_which_pose.png")),
            f"under {OUT}: the panel with pose {idx_a + 1} and the header on its "
            f"own, which is the only way 'which pose is it describing' is a "
            f"question about a picture",
        )
        # -- mutation 1: the panel shows the wrong pose ---------------------
        # The historical failure, and the one a reader cannot see. The panel's
        # own refresh is wrapped so it does everything the shipped one does and
        # then describes the *previous* pose -- which is the shape of the bug, a
        # stale index reaching the panel. The wrap is on
        # `_refresh_pose_verdict` rather than on `_on_pose_selected` for a
        # reason that cost a run to find: the selection handler is connected to
        # `currentCellChanged` as a bound method at construction time, so
        # replacing the class attribute afterwards leaves the already-connected
        # slot pointing at the original function and the mutation silently does
        # nothing. A mutation that is not applied is a green that means nothing.
        live_refresh = appmod.MainWindow._refresh_pose_verdict

        def off_by_one(self, index):
            live_refresh(self, int(index) - 1)

        # A pose that is not the first, so the mutation shows a real other
        # pose rather than falling into the "no pose selected" branch.
        row_c = next(
            r for r in range(win.pose_table.rowCount())
            if 1 <= int(win.pose_table.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole))
            not in (idx_a, idx_b)
        )
        appmod.MainWindow._refresh_pose_verdict = off_by_one
        win.pose_table.setCurrentCell(row_c, 0)
        wait_for_terms(win, app)
        idx_c = int(win.pose_table.item(row_c, 0).data(QtCore.Qt.ItemDataRole.UserRole))
        mut_ok, mut_why = describes_selection(win, idx_c)
        appmod.MainWindow._refresh_pose_verdict = live_refresh
        # The selection is cleared before it is made again, because setting the
        # current cell to the cell it is already on emits nothing and the panel
        # would keep the mutated content -- which is correct behaviour for the
        # product and a trap for the check. Found by this check: the first
        # version of it re-selected the same row and read the mutation back
        # after the restore, so the restored half of the bidirectional proof
        # was really a second reading of the mutation.
        win.pose_table.setCurrentCell(-1, -1)
        for _ in range(6):
            app.processEvents()
        win.pose_table.setCurrentCell(row_c, 0)
        wait_for_terms(win, app)
        back_ok, back_why = describes_selection(win, idx_c)
        check(
            "the wrong-pose panel is rejected by the same predicate that "
            "accepts the shipped one",
            not mut_ok and back_ok,
            f"with the panel refreshed one pose back, pose {idx_c} selected: "
            f"{mut_why}. Restored: {back_why}",
        )

    # ======================================================================
    section(EXPECTED_SECTIONS[3])
    # Mutation 2: a failing contract loses its remedy. Applied to the module's
    # own row builder, which is the one place the remedy is attached, and asked
    # through the same predicate that accepts the shipped rows.
    live_rows = pt._rows_from_verdict
    dropped = pt.describe_pose(result, 1, out_of_box_penalty=0.0, penalty_pose_index=1)
    control_ok, control_why = failing_rows_have_remedies(
        dropped, dropped.explanation())

    def rows_without_remedies(verdict):
        return tuple(
            dataclasses.replace(row, remedy="", remedy_kind="")
            if row.state == "fails" else row
            for row in live_rows(verdict)
        )

    pt._rows_from_verdict = rows_without_remedies
    try:
        mutated = pt.describe_pose(result, 1, out_of_box_penalty=0.0,
                                   penalty_pose_index=1)
        mut_ok, mut_why = failing_rows_have_remedies(mutated, mutated.explanation())
        shown_text = mutated.explanation()
    finally:
        pt._rows_from_verdict = live_rows
    restored = pt.describe_pose(result, 1, out_of_box_penalty=0.0, penalty_pose_index=1)
    back_ok, back_why = failing_rows_have_remedies(restored, restored.explanation())
    check(
        "a failing contract that lost its remedy is rejected by the same "
        "predicate that accepts the shipped rows",
        control_ok and not mut_ok and back_ok,
        f"shipped: {control_why}. With the remedy dropped from every failing "
        f"contract: {mut_why}, and the reader is left with "
        f"{shown_text[:160]!r}. Restored: {back_why}",
    )

    # ======================================================================
    section(EXPECTED_SECTIONS[4])
    if app is None:
        for name in (
            "a window holding a pose file says 'unknown' and nothing else",
            "and 'unknown' is not wearing the pass treatment",
            "the panel rendered with the pass styling is rejected by the same "
            "predicate",
            "the panel names the action that would change the answer",
            "and the refusal is a verdict, not an exception",
        ):
            skip(name, no_window)
    else:
        file_verdict = pt.describe_file_pose(0, 9)
        fw = MainWindow(receptor=EXAMPLES / RECEPTOR, ligand=EXAMPLES / LIGAND,
                        poses=EXAMPLES / "poses.pdbqt")
        fw.resize(1180, 980)
        fw.show()
        for _ in range(20):
            app.processEvents()
        settle(app, 400)
        fw.pose_table.setCurrentCell(0, 0)
        settle(app, 600)
        file_rows = panel_rows(fw)
        fw.verdict_box.grab().save(str(OUT / "pose_verdict_unknown.png"))
        states = {name: state for name, (state, _pair) in file_rows.items()}
        check(
            "a window holding a pose file says 'unknown' and nothing else",
            "UNKNOWN" in panel_head(fw).upper()
            and set(states.values()) == {"unmeasured"}
            and len(file_rows) == 4,
            f"header {panel_head(fw)[:150]!r}; {len(file_rows)} rows, states "
            f"{states}. A pose file carries coordinates, the contracts read a "
            f"run's own numbers, and nothing here was measured",
        )
        ok, why = unknown_is_not_a_pass(fw)
        check("and 'unknown' is not wearing the pass treatment", ok, why)
        # -- mutation 3: render unknown in the pass treatment --------------
        live_sheet = pt.stylesheet_for
        live_colours = pt.STATE_COLOURS

        def pass_for_unknown(state, _live=live_sheet):
            return _live("yes") if state == "unknown" else _live(state)

        pt.stylesheet_for = pass_for_unknown
        pt.STATE_COLOURS = dict(live_colours, unknown=live_colours["yes"])
        try:
            fw._refresh_pose_verdict(fw._current_pose)
            app.processEvents()
            mut_ok, mut_why = unknown_is_not_a_pass(fw)
            fw.verdict_box.grab().save(str(OUT / "pose_verdict_unknown_mutated.png"))
        finally:
            pt.stylesheet_for = live_sheet
            pt.STATE_COLOURS = live_colours
            fw._refresh_pose_verdict(fw._current_pose)
            app.processEvents()
        back_ok, back_why = unknown_is_not_a_pass(fw)
        check(
            "the panel rendered with the pass styling is rejected by the same "
            "predicate",
            not mut_ok and back_ok,
            f"with unknown painted and weighted as a pass: {mut_why}. Restored: "
            f"{back_why}. The mutated panel is saved beside the real one as "
            f"{OUT / 'pose_verdict_unknown_mutated.png'} so the difference is a "
            f"picture and not a claim",
        )
        why_text = fw.lbl_verdict_why.text() + fw.lbl_verdict_note.text()
        check(
            "the panel names the action that would change the answer",
            "Re-dock this ligand here" in why_text
            and "conformation" in panel_head(fw).lower() + why_text.lower(),
            f"the note points at the button in the panel above; the header says "
            f"why: {panel_head(fw)[:120]!r}. This is a way out, not a dead end "
            f"dressed as a feature -- and the action is the one the energy "
            f"breakdown already offers, so there is no second button to press",
        )
        check(
            "and the refusal is a verdict, not an exception",
            file_verdict.trust == rt.UNKNOWN
            and len(file_verdict.rows) == 4
            and not file_verdict.supplied
            and "from a file" in file_verdict.summary,
            f"describe_file_pose returns trust {file_verdict.trust!r} with "
            f"{len(file_verdict.rows)} unmeasured rows, {file_verdict.supplied} "
            f"supplied and {len(file_verdict.absent)} absent. An exception here "
            f"would be a second, different answer to the same question",
        )
        fw.close()
        del fw
        for _ in range(3):
            app.processEvents()

    # ======================================================================
    section(EXPECTED_SECTIONS[5])
    before = len(RESULTS)
    check(
        "the number of checks that ran is the number this file declares",
        before + 2 == EXPECTED_CHECKS,
        f"{before} ran before these two and {EXPECTED_CHECKS} are expected; the "
        f"+2 is these two. Change EXPECTED_CHECKS deliberately",
    )
    check(
        "every section this file is supposed to run was reached",
        tuple(_sections) == EXPECTED_SECTIONS,
        f"{len(_sections)} of {len(EXPECTED_SECTIONS)} sections reached",
    )
    passed = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    failed = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    skipped = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print("\n=== summary ===")
    print(f"  {len(RESULTS)} checks: {passed} passed, {failed} failed, "
          f"{skipped} skipped")
    print(f"  screenshots: {OUT}")
    if passed + failed + skipped != len(RESULTS):
        print("  the three counts do not partition the ledger")
    for line in FAILURES:
        print(f"  FAILED: {line}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
