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
2. **Five mutations, each rejected by the same predicate that accepts the
   shipped rule.** Wrong pose first, because it is the one a reader cannot see;
   then a failing contract that has lost its remedy; then `unknown` rendered in
   the pass treatment; then the header naming its own pose a second time; and
   then the header naming a *different* pose. Every mutation is bidirectional:
   the shipped code is accepted by the predicate before and after, so a
   predicate that rejects everything is not mistaken for a predicate that
   works.

A fourth property is a **count** rather than a shape, and it is the one this
file grew for. The panel's header opens with the verdict's state and then the
verdict module's own sentence, and that sentence names the pose -- once, by the
module's own design. The panel used to add the name in front of it and again in
front of its own summary, so `dist/pose_trust/pose_verdict_unknown.png` carried
one pose named three times in two spellings. So the header is judged by
counting `pose N of M` in it: the selected pose exactly once, and no other
pose's number at all, on the path for a run and the path for a pose file. Both
directions are mutated, because "names it twice" and "names the wrong one" fail
the same sentence in ways a search for the right string cannot tell apart.

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

**The exit code is a function of the verdict, and there are three verdicts.**
`0` PASS, `1` DEFECT, `2` ALL_WINDOW_SKIPPED. The third is the one this file did
not have and the reason it needed one: twenty of the forty checks are the ones
that can only be answered against a live window, `skip()` records a result
rather than a failure, and `return 0 if failed == 0 else 1` therefore read a run
that judged **half of itself** as a clean one -- on any machine where
`QApplication([])` raises. It is a distinct code rather than a `1` because "the
panel is wrong" and "this machine has no window" are different findings with
different remedies, and a step that cannot tell them apart cannot act on either.
`npass == 0` is *not* the condition, and deliberately: at twenty of forty it
would never fire. `verdict_of` below is the whole rule, pure, and takes every
count as an argument so a run that never happened can be handed to it.

Run:  set PYTHONPATH to dock-py/python and run this file by path.
"""
from __future__ import annotations

import dataclasses
import math
import re
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

# The module under test, taken through the *panel's* resolution rather than by
# a name typed here. When the module was packaged the name changed from
# `result_trust` to `opendocking.result_trust`, and this line took the gate down
# with a ModuleNotFoundError before a single check ran -- which is the failure
# mode of a check that hard-codes the thing it is checking. Going through
# `pose_trust.verdict_module()` also means the gate and the product cannot
# disagree about where the module lives.
from opendocking.workbench import pose_trust as _pt_for_import  # noqa: E402

rt = _pt_for_import.verdict_module()

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
#: + 13 section 3: the panel describes the pose that is selected, the
#:      wrong-pose mutation is rejected by the same predicate, and the header's
#:      pose numbers are counted -- once for the shipped panel, and once per
#:      direction the count can be wrong (a reason quoted in it, another pose's
#:      number in it) -- with the reasons' single home under the table
#: + 1  section 4: a failing contract that lost its remedy is rejected
#: + 10 section 5: a pose from a file is an honest `unknown`, its header names
#:      that pose once and quotes no reason, `unknown` is never rendered in the
#:      pass treatment, a panel that could not reach the module at all does not
#:      look like a pass, and the four checks on where that module is looked for
#: + 2  the two meta-checks, which count themselves
#:
#: 7 + 6 + 13 + 1 + 10 + 2 = 40. Section 1 is seven rather than six because it
#: opens with the fixture check -- the run has to have produced poses with
#: gradients before any of the rest of the file means anything, and a gate that
#: quietly judged nothing would still reach its number.
#: **31 -> 34, and the three are where the verdict module is looked for.** One
#: that it resolves with the packaged location first and the checkout's
#: `scripts/` second; one that a cold cache asks for it by name before falling
#: back; and one that the failure message names both places it tried and says
#: that nothing passes. They live in section 5 because that is where the panel's
#: other "there is no verdict" branch is, and a module that is not found and a
#: pose with no numbers are the same kind of event for a reader: nothing was
#: measured.
#: **34 -> 39, and four of the five are the header's pose numbers.** A panel
#: that names the same pose three times in two spellings, and a panel that
#: names a different pose once, are the two ways the one sentence whose job is
#: "which pose is this about" goes wrong; both are counts, and a count that is
#: only ever tested against the shipped panel cannot tell a working predicate
#: from a strict one. The fifth is the `sys.path` one, added with the removal
#: of the `scripts/` fallback: "found a module" and "found a module and a path
#: entry" are the same sentence until something measures the difference.
#:
#: **39 -> 40, and the one is the header's reasons.** The header now carries the
#: state, the pose, a count of what the state is made of, and a pointer to the
#: table -- and the reasons are printed once, under it. The new site is the
#: claim that needs both places to be true at once: a check that only read the
#: header would pass on a panel that had simply dropped the reasons, and one
#: that only read the block would pass on a panel that printed them twice.
#:
#: **Still 40 after the all-skipped guard, and that is deliberate.** The guard is
#: a *verdict* over the ledger, not a new `check()`, so it adds no site and the
#: declared total above still equals the sites below. A gate that reached 41 by
#: testing its own exit code would be testing the same thing twice on the same
#: run, and the table that tests `verdict_of` belongs with the other pure
#: functions -- where a mutation of the guard is a one-line diff rather than a
#: failed run.
EXPECTED_CHECKS = 40

EXPECTED_SECTIONS = (
    "1. the binding reads the objects, and the keys are not typed",
    "2. three states, and the remedy rides with a failure",
    "3. the panel describes the pose that is selected",
    "4. a failing contract that lost its remedy is not a warning light",
    "5. a pose from a file is an honest unknown, never a pass",
    "6. this file ran all of it",
)

# --------------------------------------------------------------------------
# The checks that can only be answered against a live window, as ONE list.
#
# **These are the population the exit code is computed over, and they are
# hoisted out of `main()` so the number cannot be counted one way and skipped
# another.** They used to be two inline tuples inside the two `app is None:`
# branches, and the census that found this gate had no all-skipped guard needed
# to read those branches and count them by eye: 13 in section 3 and 7 in
# section 5, twenty of this file's forty. A population that exists only as
# literal text inside a branch is a population nobody can ask a question about,
# and the question here is the one the exit code is now built on: *did the
# window-dependent half of this file judge anything at all?*
#
# They are also the checks that are precise **only** when a window is real: the
# pass-treatment mutation and the header-count mutations cannot be seen from a
# dataclass, so a skipped one has not been weakly confirmed, it has not been
# confirmed. Which is why the answer to "twenty of forty skipped" cannot be 0.
# --------------------------------------------------------------------------
SECTION3_WINDOW_CHECKS = (
    "the window's own run produced poses to judge",
    "the panel names the pose that is selected",
    "and the window recorded that same pose as the one on show",
    "the panel's numbers are a second, independent read of that pose's own "
    "gradient",
    "and the module, asked separately about the same pose, agrees",
    "the numbers change with the pose, so this is not one pose's verdict "
    "wearing two names",
    "the remedy for the failing contract is on screen, not only in the object",
    "the screenshots exist: a 'no', and the claim of which pose",
    "the wrong-pose panel is rejected by the same predicate that accepts the "
    "shipped one",
    "the header names the selected pose, once, and carries no other pose's "
    "number",
    "the reasons are printed once, under the table, and the header is not a "
    "second copy",
    "a header that quotes one of the reasons is rejected by the same predicate",
    "a header carrying another pose's number is rejected by the same predicate",
)

SECTION5_WINDOW_CHECKS = (
    "a window holding a pose file says 'unknown' and nothing else",
    "and 'unknown' is not wearing the pass treatment",
    "the header of a pose from a file names that pose once, quotes no reason, "
    "and points at the table",
    "the panel rendered with the pass styling is rejected by the same "
    "predicate",
    "the panel names the action that would change the answer",
    "and the refusal is a verdict, not an exception",
    "a panel that cannot reach the verdict module does not look like a pass",
)

WINDOW_DEPENDENT_CHECKS = SECTION3_WINDOW_CHECKS + SECTION5_WINDOW_CHECKS

#: How many of this file's checks cannot be answered without a window. 13 + 7.
#:
#: It is derived, not typed, and it is the same number on every machine --
#: which is the property `EXPECTED_CHECKS` above already depends on, and the
#: reason this one is a count of *code* rather than of *this run*: a run in
#: which all twenty skipped has still run forty checks.
WINDOW_DEPENDENT_COUNT = len(WINDOW_DEPENDENT_CHECKS)

#: The three exit codes. **The code is the whole contract** -- CI reads the
#: integer and nothing else, so a verdict that does not change one of these is
#: invisible to every reader of this step's result.
#:
#: ``0`` PASS, ``1`` DEFECT, ``2`` ALL_WINDOW_SKIPPED.
#:
#: The third is the one this file did not have, and it is the reason the
#: summary used to print `40 checks: 20 passed, 0 failed, 20 skipped` and
#: return 0. Twenty of those forty are the checks that judge the panel against
#: a live window; on a machine with no `QApplication` they cannot run, and
#: `skip()` records a result rather than a failure, so `return 0 if failed == 0
#: else 1` read a run that judged **half of itself** as a clean one. The count
#: was pinned at 40 precisely so that the total could not hide it, and then the
#: exit code hid it anyway.
#:
#: It has to be a *distinct* code rather than 1. A failing check is a statement
#: about the product -- something is wrong with the panel -- and "this machine
#: could not put a window on screen" is a statement about the machine, with a
#: different remedy, and a CI step that cannot tell the two apart cannot act on
#: either. Eight of the twelve skip-capable gates in `scripts/` already draw
#: that distinction; this one did not.
PASS = 0
DEFECT = 1
ALL_WINDOW_SKIPPED = 2

VERDICT_LABELS = {
    PASS: "PASS",
    DEFECT: "DEFECT",
    ALL_WINDOW_SKIPPED: "ALL_WINDOW_SKIPPED",
}


def verdict_of(*, nfail: int, nskip: int, ntotal: int,
               window_declared: int, window_skipped: int) -> tuple[int, str]:
    """The exit code, as a pure function of what this run observed.

    Pure so the mapping can be table-tested from outside, and so the mutation
    that makes the guard lie is a one-line change with a number to watch. The
    order of the two guards is the argument, and it is the order
    ``qt_gl_probe.verdict_of`` already uses: **whether the product is wrong is
    asked before whether this machine answered.** A run that failed a check and
    also could not open a window has found a real defect, and reporting it as
    "inconclusive" would discard the only finding in it.

    ``window_declared`` is the number of checks in this file that need a window
    (a fact about the code, 20). ``window_skipped`` is how many of them this run
    could not answer. So ``window_declared - window_skipped`` is how many of
    them judged something, and that is the number this file's verdict rests on.

    Every count is an argument and none is read off a global, so a caller -- or
    a table test -- can hand it a run that never happened.
    """
    if nfail:
        return DEFECT, (f"{nfail} check(s) did not hold; the run finished and the "
                        f"panel is wrong about something")
    if nskip != window_skipped:
        # Every skip in this file comes from one of the two `app is None:`
        # branches, so the count of skips and the count of skipped window checks
        # are the same number. If they are not, a skip has been added somewhere
        # this file does not know about, and the population the verdict is
        # computed over is no longer the set of checks that ran -- which is a
        # fact about *this file*, and so a defect rather than a shrug.
        return DEFECT, (f"{nskip} check(s) skipped but only {window_skipped} of "
                        f"them are in the declared window-dependent population of "
                        f"{window_declared}, so the exit code is being computed "
                        f"over a set that is not the set that ran")
    ran = window_declared - window_skipped
    if window_declared and not ran:
        return ALL_WINDOW_SKIPPED, (
            f"all {window_declared} of this file's window-dependent checks "
            f"skipped, so the panel was never judged against a live window. "
            f"Nothing here is wrong and nothing here was verified: a run that "
            f"checked {ntotal} sites and could answer none of the "
            f"window-dependent ones is not a pass"
        )
    return PASS, (f"{ran} of {window_declared} window-dependent check(s) judged "
                  f"the panel against a live window")

#: GATE-DECLARE 1
#: sites: 20 unconditional + 20 guarded
#: guards: sha256:0c89a06ce47570ba4bd1307251429d0b28427c506afd51a0e0328087c1e17764
#:
#: 20 + 20 = 40, and it matches `EXPECTED_CHECKS` because every site here runs
#: exactly once: there is no `check()` inside a loop, and a site that ran twice
#: would be a count the declaration could not see.
#:
#: The twenty guarded sites are sections 3 and 5, and every one of them
#: carries
#: the same guard string, `if app is None:` -- whether a QApplication can be
#: created at all. That is not laziness, it is the point of section 5. A real
#: `unknown` has to be **on screen** to be checked for the pass treatment,
#: because the mutation being rejected there is a styling mutation, and a
#: styling mutation cannot be seen from a dataclass -- `PoseVerdict` carries the
#: state and the rows, and only the panel carries the colours. A gate that
#: judged the third state from the model alone would pass on a panel that
#: painted `unknown` green, which is the exact failure it was written to catch.
#: The same argument covers the last site, which is about a panel that could not
#: draw anything at all.
#:
#: So sections 1, 2 and 4, plus the four checks on where the verdict module is
#: looked for -- twenty sites -- run on a machine with no display: the
#: binding, the key provenance, the three states, the precedence rule, the
#: remedy, the remedy-loss mutation, and the module's own import order, its
#: effect on `sys.path` and its refusal are all answerable without a window.
#:
#: **16 + 22 -> 20 + 20, and the three that moved are the import checks.** They
#: were inside section 5's `else:` branch while never needing a window, and
#: they were not in the skip list either, so on a machine with no display they
#: simply did not run: the declared total moved by three between the two kinds
#: of machine, which is the failure the `EXPECTED_CHECKS` docstring above says
#: this total exists to prevent. They are above the branch now, where what they
#: read -- a candidate name, `sys.path`, a refusal message -- needs no window at
#: all.
#:
#: The digest counts the guard string once per site, so adding a site changed
#: it even though the *string* is identical to the ones beside it -- the checks
#: on where the verdict module is looked for used to be guarded by the same
#: `if app is None:` as the rest of section 5. A digest over distinct guard
#: strings would have been blind to both edits, and the census would have been
#: the only thing standing between them and an undeclared change.
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


#: Every `pose N of M` in a header, whatever else is in it.
#:
#: A pattern rather than a search for one string, because the failure that
#: matters is a number belonging to a *different* pose, and that is exactly the
#: one a search for the right string cannot see: `find` on the expected label
#: succeeds whether or not somebody else's number is also in the sentence.
POSE_IDENTITY = re.compile(r"pose (\d+) of (\d+)")

#: The three words a verdict's header may open with. `PoseVerdict.trust` is one
#: of the module's three states and nothing else -- a fourth state would have to
#: be added here deliberately, which is the point.
STATE_HEADS = ("YES", "NO", "UNKNOWN")


def names_only_the_selected_pose(head: str, verdict) -> tuple[bool, str]:
    """Is every pose number in ``head`` the selected pose, and is it named once?

    Asked as a count rather than as a shape, because the defect it replaces was
    a shape nobody looked at. `dist/pose_trust/pose_verdict_unknown.png` as it
    was captured before the fix opened

        pose 1 of 9 -- UNKNOWN: pose 1 of 9 (from a pose file): ...

    and carried the same pose's name three times in two spellings, in the one
    sentence whose entire job is to say which pose is being described. A reader
    cannot tell that from the picture; a count can.

    Five things, and all five are load-bearing:

    * ``verdict.described_pose`` appears **exactly once**. A second copy is the
      old defect, whether the copy is spelled the same way or not;
    * every `pose N of M` in the string is *this* pose of *this* run. A header
      carrying another pose's number is the failure a reader cannot see, and it
      is the one this predicate is named for;
    * the header opens with the state word, so a reader who stops after one
      word still stops on the state;
    * **no reason text appears in it.** The reasons live once, under the table,
      and a header quoting one of them is the duplication this was asked to
      remove. `NOT_MEASURED` is excluded from the comparison: it is this
      module's sentinel for *there being no reason text*, and the shipped
      header says "4 not measured", so a predicate that matched the sentinel
      would reject the very header it is meant to accept;
    * the pointer to where the reasons are **is** in it. A header with no
      reason in it and nothing saying where the reason went has not been
      shortened, it has been truncated.

    The verdict is passed whole rather than its index, count and label
    separately, because two clauses need the rows and one needs
    `described_pose` -- and three arguments kept in step at four call sites is
    three more places for the predicate and the panel to disagree.
    """
    index, num_poses = int(verdict.pose_index), int(verdict.num_poses)
    label = verdict.described_pose
    found = POSE_IDENTITY.findall(head)
    mine = (index + 1, num_poses)
    others = [(int(n), int(m)) for n, m in found if (int(n), int(m)) != mine]
    times = head.count(label)
    state = head.split(":", 1)[0].strip()
    reasons = [r.because for r in verdict.rows
               if r.because and r.because != pt.NOT_MEASURED]
    if verdict.why:
        reasons.append(verdict.why)
    leaked = [text[:60] for text in reasons if text in head]
    pointed = verdict.WHERE_REASONS in head
    ok = (times == 1 and not others and state in STATE_HEADS
          and not leaked and pointed)
    why = (
        f"the header names {label!r} {times} time(s); every 'pose N of M' in it "
        f"is {found}; the ones that are not pose {mine[0]} of {mine[1]} are "
        f"{others or 'none'}; it opens with {state!r}, one of "
        f"{list(STATE_HEADS)}; reason text quoted in it: {leaked or 'none'}; "
        f"and it points at the table: {pointed}. Header: {head[:150]!r}"
    )
    return ok, why


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


# --------------------------------------------------------------------------
# Screenshots: an instrument that has to be able to fail
# --------------------------------------------------------------------------
#: Every glyph in a capture taken under the offscreen platform is an empty box,
#: because that platform has **no font families at all** -- measured on this
#: machine as 266 families with the default platform and 0 under offscreen. The
#: geometry is identical, the colours are identical, and the words are gone, so
#: a capture like that is not weak evidence, it is not evidence: this file
#: shipped four of them once and they were read as proof of nothing.
#:
#: So the precondition is checked before anything is written: zero font families
#: means the words cannot be rendered, and the file is **not** written. Leaving
#: an earlier good capture in place beats replacing it with an unreadable one,
#: and the reason is printed either way.
#:
#: What this deliberately does not do is measure ink. Measured on this machine,
#: tofu lays down *fewer* ink pixels than real glyphs (4496 against 5387) while
#: having roughly twice as many disconnected strokes, so an "is there enough
#: font ink here" test scores an unreadable capture as the better rendering. A
#: guard that gets that backwards is worse than no guard, so this one asks the
#: platform instead of the picture.
MIN_FONT_FAMILIES = 1


def can_render_text() -> tuple[bool, str]:
    """Whether this platform can put a word on screen, and what it has."""
    from PyQt6 import QtGui

    families = QtGui.QFontDatabase.families()
    return len(families) >= MIN_FONT_FAMILIES, (
        f"platform {QtWidgets.QApplication.instance().platformName()!r} with "
        f"{len(families)} font families"
        + ("" if families else " -- this platform has no fonts, so every glyph "
                              "would be an empty box and the capture would be "
                              "worthless as evidence")
    )


def save_png(widget, path: Path) -> str:
    """Grab a widget, but only if the platform can render its words.

    Returns a line for the transcript saying what happened, either way. A caller
    that ignores the return value can still write nothing to disk without
    knowing it, which is the failure this exists to prevent.
    """
    ok, why = can_render_text()
    if not ok:
        return f"NOT WRITTEN {path.name}: {why}"
    widget.grab().save(str(path))
    return f"wrote {path.name} on a platform that can render text ({why})"


def main() -> int:
    schema = sorted(rt.schema_for("pose"))
    available_ok, _available_why = pt.verdict_is_available()

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
    shots: list[str] = []
    if app is None:
        for name in SECTION3_WINDOW_CHECKS:
            skip(name, no_window)
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
        shots.append(save_png(win.verdict_box, OUT / "pose_verdict_no.png"))
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
        shots.append(save_png(win.lbl_verdict_head,
                             OUT / "pose_verdict_which_pose.png"))
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
            "the screenshots exist, and this platform could have written them",
            all((OUT / name).is_file() for name in (
                "pose_verdict_no.png", "pose_verdict_which_pose.png"))
            and can_render_text()[0],
            "; ".join(shots) + f". Under {OUT}: the panel with pose "
            f"{idx_a + 1} and the header on its own, which is the only way "
            f"'which pose is it describing' is a question about a picture",
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

        # -- the header names one pose, once, and quotes no reason ----------
        # The header is judged as a *count* of pose numbers and a *search* for
        # reason text, because the two defects it replaces were both invisible
        # in a picture: a name repeated three times, and a paragraph of reasons
        # printed in the one element meant to be read in a glance. Every
        # mutation below goes through `_verdict_fill`, the panel's own writer,
        # so what is judged is the string on the label and not a string
        # assembled here -- the same trap as mutation 1 above, for the same
        # reason: judging a string this gate built would prove nothing about
        # the string a reader gets.
        idx_s = int(win._current_pose)
        n_s = int(win._dock_result.num_poses)
        # The panel's **own** verdict, not a fresh call. The panel builds it with
        # the out-of-box penalty out of the selected pose's breakdown, so a
        # second `describe_pose` without that penalty is a *different* verdict:
        # `inside the box` reads `unmeasured` instead of `holds` and the header
        # counts three unmeasured contracts where the label counts two. The
        # first version of this site made exactly that mistake and compared two
        # different verdicts, which is why `live.headline() == shipped_head` --
        # the clause that proves the label is this verdict's header, and not
        # merely a header of the same shape -- failed while every count in the
        # predicate passed.
        penalty = None
        if (win._terms_breakdown is not None
                and int(win._terms_breakdown.pose_index) == idx_s):
            penalty = win._terms_breakdown.out_of_box_penalty
        live = pt.describe_pose(
            win._dock_result, idx_s,
            out_of_box_penalty=penalty,
            penalty_pose_index=idx_s if penalty is not None else None,
        )
        shipped_head = win.lbl_verdict_head.text()
        ok_s, why_s = names_only_the_selected_pose(shipped_head, live)
        check(
            "the header names the selected pose, once, and carries no other "
            "pose's number",
            ok_s and live.headline() == shipped_head,
            f"{why_s}. The panel's writer puts `PoseVerdict.headline()` on this "
            f"label verbatim, and the gate reads the label rather than the "
            f"object, so the string counted is the one on screen",
        )
        # The other half of the same claim, and it is about the two places
        # rather than about the header: the reasons have to be *somewhere*, and
        # they are under the table. A check that only looked at the header
        # would pass on a panel that simply dropped them.
        #
        # "Somewhere" is not one place. A contract that **fails or is
        # unmeasured** prints its reason in the block, because that is what the
        # reader has to act on; a contract that **holds** prints it in its own
        # row's tooltip, because there is nothing to act on and the block is
        # not a list of things that are fine. The first version of this check
        # demanded all four reasons in the block and went red on the holding
        # one -- which is the check being wrong about the design rather than the
        # panel being wrong.
        block = win.lbl_verdict_why.text()
        actionable = [r for r in live.rows
                      if r.state in ("fails", "unmeasured") and r.because]
        settled = [(i, r) for i, r in enumerate(live.rows)
                   if r.state == "holds" and r.because]
        in_block = [r.name for r in actionable if r.because in block]
        in_tip = [
            f"{r.name}@row {i}"
            for i, r in settled
            if r.because in (win.tbl_verdict.item(i, 0).toolTip() or "")
        ]
        in_head = [r.name for r in live.rows
                   if r.because and r.because in shipped_head]
        check(
            "the reasons are printed once, under the table, and the header is "
            "not a second copy",
            (len(in_block) == len(actionable)
             and len(in_tip) == len(settled)
             and not in_head
             and live.explanation()[:120] in block),
            f"{len(actionable)} contract(s) fail or are unmeasured and their "
            f"reasons are in the block: {in_block}; {len(settled)} hold(s) and "
            f"theirs are in their own row's tooltip: {in_tip or 'none to check'}; "
            f"reasons quoted in the header: {in_head or 'none'}. The block is "
            f"`explanation()` verbatim: {live.explanation()[:100] in block}, and "
            f"begins {block[:90]!r}. The header is {shipped_head[:110]!r} -- a "
            f"state, a pose, a count and a pointer, and no reason at all",
        )

        # -- mutation 4: a reason put back into the header -------------------
        # The defect as the ruling described it: the header quoting something
        # the block below already prints in full. Applied to the class method
        # rather than to a field, because the header is no longer built from
        # `summary` -- a `dataclasses.replace` on that field would have changed
        # nothing on screen, which is the no-op mutation this file has been
        # caught by twice.
        shipped_method = pt.PoseVerdict.headline

        def with_reason(self):
            first = next((r.because for r in self.rows if r.because), "")
            return (f"{self.trust.upper()}: {self.described_pose} -- "
                    f"{self._clause()}.  {first}")

        pt.PoseVerdict.headline = with_reason
        try:
            win._verdict_fill(live, n_s)
            app.processEvents()
            reason_head = win.lbl_verdict_head.text()
            reason_ok, reason_why = names_only_the_selected_pose(reason_head, live)
        finally:
            pt.PoseVerdict.headline = shipped_method
            win._verdict_fill(live, n_s)
            app.processEvents()
        back_ok, back_why = names_only_the_selected_pose(
            win.lbl_verdict_head.text(), live)
        check(
            "a header that quotes one of the reasons is rejected by the same "
            "predicate",
            (ok_s and reason_head != shipped_head and not reason_ok and back_ok),
            f"the patched method did change the label: "
            f"{reason_head != shipped_head} -- {reason_head[:130]!r} against the "
            f"shipped {shipped_head[:110]!r}. With a reason in it: {reason_why}. "
            f"Restored: {back_why}",
        )

        # -- mutation 5: somebody else's pose number ------------------------
        # The other direction, and the one that is not visible at all: a name
        # that is well formed, correctly spelled, and about a different pose --
        # a number a reader has no way to check from the header alone.
        def with_other_pose(self):
            other = 0 if int(self.pose_index) != 0 else 1
            return (f"{self.trust.upper()}: pose {other + 1} of {self.num_poses}"
                    f" -- {self._clause()}.  {self.WHERE_REASONS}")

        pt.PoseVerdict.headline = with_other_pose
        try:
            win._verdict_fill(live, n_s)
            app.processEvents()
            other_head = win.lbl_verdict_head.text()
            wrong_ok, wrong_why = names_only_the_selected_pose(other_head, live)
        finally:
            pt.PoseVerdict.headline = shipped_method
            win._verdict_fill(live, n_s)
            app.processEvents()
        back_ok, back_why = names_only_the_selected_pose(
            win.lbl_verdict_head.text(), live)
        check(
            "a header carrying another pose's number is rejected by the same "
            "predicate",
            (ok_s and other_head != shipped_head and not wrong_ok and back_ok),
            f"with pose {idx_s + 1} selected and a header naming pose "
            f"{0 if idx_s != 0 else 1} of the same {n_s}-pose run, and the "
            f"patched method confirmed to have changed the label: "
            f"{other_head != shipped_head} -- {other_head[:130]!r}. "
            f"{wrong_why}. Restored: {back_why}",
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
    # -- where the verdict module is looked for, and what the refusal says ---
    # **Unguarded, and that is a correction rather than a style choice.** These
    # three were inside `else:` of the window branch below while never having
    # needed a window, so on a machine with no display they did not run *and*
    # were not in the skip list: the total this file declares moved by four
    # between a machine with a GL context and one without, which is the failure
    # the `EXPECTED_CHECKS` docstring says it exists to prevent. They are here,
    # above the branch, because what they read is a name and a message.
    #
    # The `scripts/` directory branch is gone, so what is asserted now is the
    # absence: the declared candidates are two names and nothing else, the
    # module carries no attribute naming a directory, and resolving it leaves
    # `sys.path` exactly as it found it -- which is the observable difference
    # between "found a module" and "found a module and a path entry".
    check(
        "the verdict module resolves, and nothing but a name is put on the "
        "path to find it",
        (available_ok
         and pt.IMPORT_CANDIDATES
         == ("opendocking.result_trust", "result_trust")
         and not hasattr(pt, "_SCRIPTS_FALLBACK")),
        f"available {available_ok}; the declared candidates are "
        f"{list(pt.IMPORT_CANDIDATES)}, and the module names no directory to "
        f"fall back to ({'still there' if hasattr(pt, '_SCRIPTS_FALLBACK') else 'no _SCRIPTS_FALLBACK attribute'}). The first entry is a dotted name on purpose: the module was packaged as `opendocking/result_trust.py`, so `import result_trust` stopped answering, and a consumer written against the file's old location rather than its name breaks on the day the move lands. A third candidate -- this checkout's `scripts/`, which the packaging emptied -- was removed rather than kept as a convenience: see the comment on `IMPORT_CANDIDATES`",
    )
    path_before = list(sys.path)
    pt._reset_module_cache()
    resolved_from_package = pt.verdict_module()
    path_unchanged = list(sys.path) == path_before
    check(
        "resolving the verdict module leaves `sys.path` as it found it",
        path_unchanged
        and Path(resolved_from_package.__file__).parent.name == "opendocking",
        f"before: {len(path_before)} entries; after: {len(sys.path)}; "
        f"unchanged: {path_unchanged}; resolved to "
        f"{resolved_from_package.__file__}. This was the observable difference "
        f"between the two designs: the removed branch was the only code in the "
        f"panel that wrote to `sys.path`, so a panel's import behaviour used to "
        f"depend on the shape of the checkout it ran from -- and a wheel has no "
        f"`parents[4]` to walk",
    )
    live_import = pt.importlib.import_module
    attempts = []

    def recording_import(name, *a, **kw):
        attempts.append(name)
        return live_import(name, *a, **kw)

    pt.importlib.import_module = recording_import
    pt._reset_module_cache()
    try:
        pt.verdict_module()
        resolved_after = True
    except pt.PoseVerdictUnavailable:
        resolved_after = False
    finally:
        pt.importlib.import_module = live_import
        pt._reset_module_cache()
        pt.verdict_is_available()  # put the real module back
    check(
        "a fresh resolution asks for the packaged name first and stops when it "
        "answers",
        resolved_after
        and attempts[:1] == [pt.IMPORT_CANDIDATES[0]]
        and set(attempts) <= set(pt.IMPORT_CANDIDATES),
        f"the attempts on a cold cache were {attempts}, against the declared "
        f"order {list(pt.IMPORT_CANDIDATES)}. The first one answered, so the "
        f"second was not needed either -- which is the point of the order rather "
        f"than an accident of it",
    )
    pt.importlib.import_module = lambda name, *a, **kw: (_ for _ in ()).throw(
        ImportError(f"no module named {name!r} (injected by this gate)")
    )
    pt._reset_module_cache()
    try:
        failure = pt.verdict_is_available()[1]
    finally:
        pt.importlib.import_module = live_import
        pt._reset_module_cache()
        pt.verdict_is_available()
    says_both = ("the installed package" in failure
                 and "a top-level module on the path" in failure)
    # The mutation for this one is the message itself: a refusal that still
    # offers a directory is describing a branch that cannot run, and one that
    # keeps the old claim about packaging is stating a fact that stopped being
    # true the day the module was installed.
    old_style = ("the verdict module could not be imported. It is not part "
                 "of the installed package.")
    says_not_a_pass = "nothing passes" in failure
    offers_directory = "this checkout's scripts directory" in failure
    check(
        "when it cannot be found, the panel names every candidate it tried and "
        "stops there",
        says_both and says_not_a_pass
        and not offers_directory
        and old_style not in failure,
        f"the message names every location it tried: {says_both}; it no longer "
        f"offers a directory: {not offers_directory}; it says nothing passes: "
        f"{says_not_a_pass}. It reads: "
        f"{failure[:300]}. Two claims this replaces are both gone: that the "
        f"module is not part of the installed package, which was true when it "
        f"was written, and that a checkout's `scripts/` is somewhere it would "
        f"look, which stopped being true when the module was packaged",
    )

    if app is None:
        for name in SECTION5_WINDOW_CHECKS:
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
        shots.append(save_png(fw.verdict_box, OUT / "pose_verdict_unknown.png"))
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
        # The same five clauses, on the path the defect was captured on. A
        # separate site rather than another clause in the check above, because
        # this is a different verdict from a different function: a pose read
        # from a file has no numbers at all, its header is built by
        # `describe_file_pose` rather than `describe_pose`, and it is the one
        # carrying a `why` sentence -- so a predicate that passed above says
        # nothing about it.
        n_f = len(fw._pose_models)
        idx_f = int(fw._current_pose)
        file_verdict_on_screen = pt.describe_file_pose(idx_f, n_f)
        ok_f, why_f = names_only_the_selected_pose(
            panel_head(fw), file_verdict_on_screen)
        check(
            "the header of a pose from a file names that pose once, quotes no "
            "reason, and points at the table",
            ok_f and file_verdict_on_screen.headline() == panel_head(fw),
            f"{why_f}. This is the header that "
            f"{OUT / 'pose_verdict_unknown.png'} captured before the fix, when "
            f"one pose was named three times in two spellings and the whole "
            f"paragraph of reasons was printed twice",
        )
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
            shots.append(save_png(fw.verdict_box,
                             OUT / "pose_verdict_unknown_mutated.png"))
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
            and "conformation" in why_text.lower()
            and "conformation" not in panel_head(fw).lower(),
            f"the note points at the button in the panel above. The sentence "
            f"saying *why* a file has no verdict -- 'a pose file carries "
            f"coordinates rather than a conformation' -- is under the table, in "
            f"{'conformation' in why_text.lower()}, and is **not** in the "
            f"header: {'conformation' not in panel_head(fw).lower()}. The "
            f"header now reads {panel_head(fw)[:120]!r}. This is a way out, not "
            f"a dead end dressed as a feature -- and the action is the one the "
            f"energy breakdown already offers, so there is no second button to "
            f"press",
        )
        check(
            "and the refusal is a verdict, not an exception",
            (file_verdict.trust == rt.UNKNOWN
             and len(file_verdict.rows) == 4
             and not file_verdict.supplied
             and file_verdict.from_file
             and "from a file" in file_verdict.why),
            f"describe_file_pose returns trust {file_verdict.trust!r} with "
            f"{len(file_verdict.rows)} unmeasured rows, {file_verdict.supplied} "
            f"supplied and {len(file_verdict.absent)} absent, and carries its "
            f"one-sentence reason on `why` "
            f"({'yes' if 'from a file' in file_verdict.why else 'NO'}). An "
            f"exception here would be a second, different answer to the same "
            f"question. The reason is on `why` and not in `summary` because "
            f"`summary` is the verdict module's own roll-up and the export "
            f"carries it verbatim; a second sentence of ours inside it is how "
            f"the header came to print the same paragraph twice",
        )
        # The other branch that can draw nothing, and the one that is easiest to
        # leave looking fine: the verdict module is not importable, so there are
        # no rows at all. An empty table next to a filled-in breakdown reads as
        # "nothing to say", which is how a missing module becomes a silent pass.
        live_available = pt.verdict_is_available
        reason = (
            "the verdict module could not be imported (injected by this gate). "
            "Nothing here was measured."
        )
        pt.verdict_is_available = lambda: (False, reason)
        try:
            fw._refresh_pose_verdict(fw._current_pose)
            app.processEvents()
            head = fw.lbl_verdict_head.text()
            sheet = fw.lbl_verdict_head.styleSheet()
            rows_drawn = fw.tbl_verdict.rowCount()
            why_shown = fw.lbl_verdict_why.text()
            note = fw.lbl_verdict_note.text()
        finally:
            pt.verdict_is_available = live_available
            fw._refresh_pose_verdict(fw._current_pose)
            app.processEvents()
        check(
            "a panel that cannot reach the verdict module does not look like a "
            "pass",
            (sheet == pt.stylesheet_for("unknown")
             and sheet != pt.stylesheet_for("yes")
             and "no verdict available" in head
             and "This is not a pass." in note
             and reason in why_shown
             and rows_drawn == 0),
            f"with the import failed the panel drew {rows_drawn} rows and said "
            f"{head!r} in the {sheet!r} treatment -- the unmeasured one, not the "
            f"pass one -- and its note ends with 'This is not a pass.'. An empty "
            f"table here would read as a clean run",
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
    window_names = set(WINDOW_DEPENDENT_CHECKS)
    window_skipped = sum(1 for t, n, _d in RESULTS
                         if t == "SKIP" and n in window_names)
    print("\n=== summary ===")
    print(f"  {len(RESULTS)} checks: {passed} passed, {failed} failed, "
          f"{skipped} skipped")
    print(f"  of the {WINDOW_DEPENDENT_COUNT} that need a live window, "
          f"{WINDOW_DEPENDENT_COUNT - window_skipped} judged it and "
          f"{window_skipped} could not be run here")
    print(f"  screenshots: {OUT}")
    if passed + failed + skipped != len(RESULTS):
        print("  the three counts do not partition the ledger")
    for line in FAILURES:
        print(f"  FAILED: {line}")

    # The two numbers in the line above are deliberately not added. A pass is
    # "nothing failed", and on a machine with no window that sentence is also
    # true of a run that judged half of itself -- so the verdict is not read off
    # `failed` alone. See `verdict_of`, and the note on
    # `ALL_WINDOW_SKIPPED` for why the third code exists at all.
    code, why = verdict_of(nfail=failed, nskip=skipped, ntotal=len(RESULTS),
                           window_declared=WINDOW_DEPENDENT_COUNT,
                           window_skipped=window_skipped)
    print(f"\nRESULT: {VERDICT_LABELS[code]} (exit {code}) -- {why}")
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
