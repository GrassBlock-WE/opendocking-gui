"""Does a bare Qt `QOpenGLWidget` get a context here, by a route CI does not walk?

**What this file is, and what it is not.**

It asks a *mechanism* question: can a plain `QOpenGLWidget` obtain a GL context
in this environment at all, and are there requestable configurations beyond the
ones the product already tries. It is not the product's capability answer, and it
does not try to be -- that is `odgui --check`, and the routes CI applies to it
are enumerated in one place only, `gl_route_matrix.py`.

That separation is the point of this rewrite. The file used to walk six
strategies of its own -- default, offscreen, `QT_OPENGL=software`,
`QT_OPENGL=angle`, `QT_OPENGL=desktop`, and a bare `QOpenGLContext` -- and three
of those six were routes the CI capability ladder also walks. Two enumerations
of the same question is a defect waiting to happen: they drift, someone reads
one and wires the other, and the answer to "which route works" becomes a matter
of which file you opened. So the three overlapping widget probes were removed
rather than reconciled, because the ladder already covers them with strictly
better evidence -- a real process exit code and a `--check --json` payload,
rather than a substring match on `GL_OK`.

What is left is deliberately the residue: two `QT_OPENGL` values the ladder does
not name, and one probe that does not involve a widget at all. Those are
questions only this file asks, which is the only reason it still exists.

**The overlap is checked, not just documented.**

`_overlap_report()` imports the matrix's declared routes and reports every widget
probe here that is configured by a subset of a route the ladder walks -- which is
exactly how the duplication came back in the first time, since every removed
probe was a strict subset of some rung. A comment saying "do not re-add these"
rots silently; this prints instead, and since this rewrite it is a verdict rather
than a note.

**The exit code is a function of the verdict, and there are four verdicts.**

The previous version returned 0 from `main()` and converted every escaping
exception into `SystemExit(0)` at the bottom of the file. That made the step
unable to fail by construction: the failure this file exists to detect -- a GL
context that will not come up -- was reported as success, and a child process
that died without printing a verdict was recorded as "no context" rather than as
"no answer". A step that cannot fail is worse than no step at all, because it
looks like a gate that is running.

A run now ends in exactly one of four codes:

    0  PASS          measured, and at least one residue route obtained a context
    2  NO_CONTEXT    measured, every route answered, none obtained a context
    3  DEFECT        a fact about this file: a route overlaps the CI ladder, this
                     run did not record the number of checks the pin says it
                     records, or the checks below did not pass
    4  INCONCLUSIVE  could not measure: no PyQt6, the ladder unreadable, or a
                     child that produced no verdict line

`2` and `4` are the two answers the old file collapsed into `0`. `NO_CONTEXT` is
not a defect -- on a session with no display driver that is a true and expected
observation, and this file says so rather than treating it as a failure -- but
it is also not the claim this step is trusted to make, which is "at least one
route works here", and a CI step cannot tell `0` from `2` unless the two differ.
`INCONCLUSIVE` matters more than either: an unreadable ladder means the overlap
guard did not run, and reporting that as "no overlap" is exactly how a guard
that never fired came to look like a guard that passed.

**The cancel problem is real, and it is not solved by lying about the code.**
This step runs inside the workbench job, where a failing step cancels every step
after it -- that is what happened when the GUI stack check exited 5 and thirteen
audits were cancelled. The codes above are therefore only safe because the step
needs `continue-on-error: true` in `.github/workflows/ci.yml`: the verdict is
then recorded, visible and assertable, without cancelling the steps behind it.
That line is not in the workflow yet. Until it is, a `2` or a `4` on the CI
runner cancels the audits after it, and the alternative -- a step that always
exits 0 -- is the defect this rewrite exists to remove.

**Has this step ever been able to fail? No.** Until this version every path
through the file returned 0, including the old `except ImportError` and
`except Exception` branches, and no verdict was ever a function of anything
measured. Every green this step has ever produced was therefore evidence of
nothing: read those runs as "the file ran", never as "the answer was yes".
"""

from __future__ import annotations

import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


#: The four verdicts, as exit codes. The code is the whole contract: CI reads
#: nothing else, so a verdict that does not change an integer here is invisible
#: to every reader of this step's result.
PASS = 0
NO_CONTEXT = 2
DEFECT = 3
INCONCLUSIVE = 4

VERDICT_LABELS = {
    PASS: "PASS",
    NO_CONTEXT: "NO_CONTEXT",
    DEFECT: "DEFECT",
    INCONCLUSIVE: "INCONCLUSIVE",
}

#: What one route can answer. `UNKNOWN` is the outcome the old file did not have,
#: and it is the whole point of this rewrite: it is the difference between "this
#: route said no" and "this route said nothing", and the old `probe()` returned
#: "no" for both by testing `"GL_OK" in out`.
GOT_CONTEXT = "got-context"
NO_ANSWER = "no-context"
UNKNOWN = "unknown"

OUTCOMES = (GOT_CONTEXT, NO_ANSWER, UNKNOWN)

#: The three states the overlap guard can be in. `OVERLAP_UNCHECKED` is a
#: verdict of its own and is not the same as `OVERLAP_CLEAR`: the old version
#: returned an empty clash list for both, so a guard that could not read the
#: ladder printed exactly what a guard that found nothing printed.
OVERLAP_CLEAR = "clear"
OVERLAP_FOUND = "found"
OVERLAP_UNCHECKED = "unchecked"

CHILD_TIMEOUT_SECONDS = 90

EXPECTED_CHECKS = 6

RESULTS: list[tuple[str, str, str]] = []


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def section(title: str) -> None:
    print()
    print(f"-- {title}")


def outcome_of(stdout: str, stderr: str, returncode: int) -> tuple[str, str]:
    """What one child's output actually said. Pure, and table-tested below.

    The child prints its verdict as its own last line, and both verdict words
    are line-initial, so the parse is a prefix test on the last line that
    carries one. Anything else -- a traceback, a timeout, an empty run, a child
    that succeeded and said nothing -- is `UNKNOWN`, and `UNKNOWN` is not a
    denial. The old `"GL_OK" in out` could not tell those apart, and neither
    could a reader of its printed table.
    """
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith("GL_OK"):
            return GOT_CONTEXT, line
        if line.startswith(NO_ANSWER):
            return NO_ANSWER, line
    err_lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    tail = (lines or err_lines or ["(no output)"])[-1]
    if returncode != 0:
        return UNKNOWN, f"child exit {returncode} and printed no verdict: {tail}"
    return UNKNOWN, f"child exit 0 and printed no verdict: {tail}"


def probe(name: str, body) -> tuple[str, str]:
    """Run one strategy in a child process and record what it actually said."""
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(body)
        path = fh.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True,
                              timeout=CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        Path(path).unlink(missing_ok=True)
        return UNKNOWN, f"timed out after {CHILD_TIMEOUT_SECONDS}s with no verdict"
    Path(path).unlink(missing_ok=True)
    return outcome_of(proc.stdout or "", proc.stderr or "", proc.returncode)


TEMPLATE = '''
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
{setup}
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

def widget_result():
    class W(QOpenGLWidget):
        def __init__(self):
            super().__init__()
            self.seen = False
            self.err = ""
        def initializeGL(self):
            self.seen = True
        def contextCreationFailed(self, why):
            self.err = why
    w = W()
    w.resize(64, 64)
    w.show()
    app.processEvents()
    for _ in range(20):
        app.processEvents()
    if w.seen:
        print("GL_OK initializeGL was called")
    else:
        print("no-context widget: " + (w.err or "(no reason given)"))

def raw_context():
    fmt = QtGui.QSurfaceFormat()
    fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 3)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    ctx = QtGui.QOpenGLContext()
    surf = QtGui.QOffscreenSurface()
    surf.setFormat(fmt)
    surf.create()
    if not surf.isValid():
        print("no-context offscreen surface invalid")
        return
    ok = ctx.create()
    made = ctx.makeCurrent(surf) if ok else False
    if made:
        f = ctx.format()
        print("GL_OK raw context %d.%d %s" % (f.majorVersion(), f.minorVersion(), f.renderableType().name))
    else:
        print("no-context raw makeCurrent failed")

{body}
'''


#: The routes this file walks, as ``(label, env overrides, body)``.
#:
#: Spelled as a declared list rather than inline in ``main()`` so that
#: `_overlap_report` has something to check. Every one of these is deliberately
#: absent from the CI capability ladder in `gl_route_matrix.ROUTES`; the three
#: that used not to be absent were removed, and the guard below is what stops
#: them coming back.
ROUTES = (
    (
        "QT_OPENGL=angle, widget",
        {"QT_OPENGL": "angle"},
        "widget_result()",
    ),
    (
        "QT_OPENGL=desktop, widget",
        {"QT_OPENGL": "desktop"},
        "widget_result()",
    ),
    (
        "raw QOpenGLContext on offscreen surface",
        {},
        "raw_context()",
    ),
)


def _setup_for(env: dict) -> str:
    """The child's env assignments, from the declared route's overrides.

    Generated from the dict rather than hand-written per route, because a
    hand-written `setup=` string is a third spelling of each route and is the
    shape that let this file and the matrix drift in the first place.
    """
    return "\n".join(f"os.environ[{key!r}] = {value!r}" for key, value in sorted(env.items()))


def _overlap_report() -> tuple[str, list]:
    """Widget routes here that the CI ladder already walks, and whether we know.

    Returns ``(state, clashes)`` where state is one of `OVERLAP_CLEAR`,
    `OVERLAP_FOUND` or `OVERLAP_UNCHECKED`. Comparing against
    `gl_route_matrix.ROUTES` rather than against a name list copied here,
    because a copy is the thing that goes stale. A route counts as overlapping
    when its overrides are a **subset** of a ladder route's: every probe removed
    from this file was a strict subset of some rung, so a subset test catches the
    real shape of the duplication and not just an exact repeat.

    Only widget probes are compared. The bare `QOpenGLContext` probe shares the
    empty environment with the matrix's `default` baseline but asks a different
    question -- it never builds a widget at all -- so comparing it would report a
    duplication that does not exist.
    """
    try:
        from gl_route_matrix import ROUTES as LADDER
    except Exception as exc:  # noqa: BLE001 - an informer must not die on this
        print(f"  note: could not read the CI ladder from gl_route_matrix "
              f"({type(exc).__name__}: {exc}), so route overlap is UNCHECKED here")
        return OVERLAP_UNCHECKED, []

    clashes = []
    for label, env, body in ROUTES:
        if body != "widget_result()":
            continue
        for rung in LADDER:
            if env.items() <= rung.env.items():
                clashes.append((label, rung.name, rung.ladder_index))
    return (OVERLAP_FOUND if clashes else OVERLAP_CLEAR), clashes


def verdict_of(outcomes, overlap_state, clashes=()) -> tuple[int, str]:
    """The verdict, as a pure function of what this run observed.

    Pure so the mapping can be table-tested, and so the mutation that makes the
    failure direction lie is a one-line change with a number to watch. The
    order of the three guards is the argument: whether *this file* is sound is
    asked before whether *this machine* answered, because a run that could not
    check the file's own guard is not a pass whatever the routes reported.
    """
    if overlap_state == OVERLAP_UNCHECKED:
        return INCONCLUSIVE, ("the CI ladder could not be read, so the overlap "
                              "guard did not run; unchecked is not clear")
    if overlap_state == OVERLAP_FOUND and clashes:
        return DEFECT, (f"{len(clashes)} route(s) here are a subset of a CI ladder "
                        f"rung and are walked twice")
    silent = sum(1 for o in outcomes if o == UNKNOWN)
    if silent:
        return INCONCLUSIVE, (f"{silent} of {len(outcomes)} route(s) produced no "
                              f"verdict line, so the answer is not known")
    if any(o == GOT_CONTEXT for o in outcomes):
        won = sum(1 for o in outcomes if o == GOT_CONTEXT)
        return PASS, f"{won} of {len(outcomes)} route(s) obtained a context"
    return NO_CONTEXT, (f"all {len(outcomes)} route(s) answered and none obtained "
                        f"a context")


def run_guarded(entry) -> int:
    """Run `entry`, and turn anything that escapes into a verdict, not a pass.

    The old bottom-of-file handler did the opposite: both branches ended in
    `SystemExit(0)`, which made "PyQt6 will not import" and "the measurement
    died" indistinguishable from a green run, and neither from an answer of yes.
    """
    try:
        return entry()
    except ImportError as exc:
        print(f"\nPyQt6 unavailable: {exc}")
        print(f"RESULT: {VERDICT_LABELS[INCONCLUSIVE]} (exit {INCONCLUSIVE}) -- there is "
              f"no question to ask here, which is not an answer of yes")
        return INCONCLUSIVE
    except Exception:
        traceback.print_exc()
        print(f"\nRESULT: {VERDICT_LABELS[INCONCLUSIVE]} (exit {INCONCLUSIVE}) -- the "
              f"measurement did not finish, which is not an answer of no")
        return INCONCLUSIVE


def check_the_routes_were_all_walked(outcomes) -> None:
    if len(outcomes) != len(ROUTES):
        bad("every declared route was walked and recorded an outcome",
            f"ROUTES declares {len(ROUTES)} route(s) and this run recorded "
            f"{len(outcomes)} outcome(s); a route that is declared but not walked "
            f"is a question this file stopped asking without saying so")
        return
    ok("every declared route was walked and recorded an outcome",
       f"all {len(ROUTES)} declared routes ran: "
       f"{', '.join(f'{o}' for o in outcomes)}")


def check_outcome_of_reads_a_child_honestly() -> None:
    """Table test: silence and denial are different answers."""
    shapes = [
        ("GL_OK raw context 3.3 OpenGL", "", 0, GOT_CONTEXT),
        ("no-context widget: no reason given", "", 0, NO_ANSWER),
        ("", "Traceback (most recent call last):\n  ...\nImportError: boom", 1, UNKNOWN),
        ("", "", 0, UNKNOWN),
        ("warming up\nGL_OK initializeGL was called", "", 0, GOT_CONTEXT),
    ]
    got = [outcome_of(o, e, r)[0] for o, e, r, _want in shapes]
    want = [w for _o, _e, _r, w in shapes]
    if got != want:
        bad("a child that said nothing is recorded as unknown, not as a denial",
            f"expected {want} and got {got}. Collapsing a dead child into "
            f"'no context' is how a crash was reported as a clean answer of no")
        return
    ok("a child that said nothing is recorded as unknown, not as a denial",
       f"{len(shapes)} child shapes parse to {got}, and the two that produced no "
       f"verdict line are {UNKNOWN} rather than {NO_ANSWER}")


def check_the_overlap_guard_reported_a_state(state, clashes) -> None:
    if state not in (OVERLAP_CLEAR, OVERLAP_FOUND, OVERLAP_UNCHECKED):
        bad("the overlap guard reported one of its three states",
            f"got {state!r}, which is not {OVERLAP_CLEAR!r}, {OVERLAP_FOUND!r} or "
            f"{OVERLAP_UNCHECKED!r}; an unrecognised state is a guard that cannot "
            f"be read")
        return
    if state == OVERLAP_UNCHECKED:
        ok("the overlap guard reported one of its three states",
           f"state {state!r}: the ladder could not be read, so this run is "
           f"INCONCLUSIVE and NOT a pass -- unchecked is not clear")
        return
    ok("the overlap guard reported one of its states",
       f"state {state!r} with {len(clashes)} clash(es) against "
       f"gl_route_matrix.ROUTES")


def check_verdict_of_maps_every_state() -> None:
    table = [
        ("all routes answered, none got a context",
         [NO_ANSWER, NO_ANSWER], OVERLAP_CLEAR, (), NO_CONTEXT),
        ("all routes answered, one got a context",
         [GOT_CONTEXT, NO_ANSWER], OVERLAP_CLEAR, (), PASS),
        ("a route overlapped the CI ladder",
         [GOT_CONTEXT, GOT_CONTEXT], OVERLAP_FOUND, (("l", "r", 0),), DEFECT),
        ("a route overlapped the ladder and a context was obtained",
         [GOT_CONTEXT], OVERLAP_FOUND, (("l", "r", 0),), DEFECT),
    ]
    got = [verdict_of(o, s, c)[0] for _n, o, s, c, _w in table]
    want = [w for _n, _o, _s, _c, w in table]
    if got != want:
        bad("every verdict state maps to its declared exit code",
            f"expected {want} and got {got} over the {len(table)} states in the "
            f"table. The exit code is the only thing CI reads")
        return
    # The table above compares `verdict_of` against the same constants it is
    # written with, so on its own it cannot notice a *wrong* code -- it would
    # notice a wrong mapping, which is the smaller defect. What it cannot see is
    # the one that matters most here: two verdicts sharing a code. `NO_CONTEXT` =
    # `PASS` is a one-character edit that satisfies the whole table and restores
    # exactly the defect this rewrite removes, because a CI step reading an
    # integer cannot tell the two apart. So the four codes are asserted pairwise
    # distinct, and that is the assertion with teeth.
    codes = [PASS, NO_CONTEXT, DEFECT, INCONCLUSIVE]
    collisions = sorted({c for c in codes if codes.count(c) > 1})
    if collisions:
        bad("every verdict state maps to its declared exit code",
            f"{collisions} is carried by more than one of "
            f"{dict(zip(['PASS', 'NO_CONTEXT', 'DEFECT', 'INCONCLUSIVE'], codes))}. "
            f"Two verdicts sharing an exit code is indistinguishable from one "
            f"verdict, and indistinguishable is what the old file was")
        return
    ok("every verdict state maps to its declared exit code",
       f"{dict(zip([t[0] for t in table], got))}, and the four codes "
       f"{codes} are pairwise distinct, so a step reading the integer can tell "
       f"the four verdicts apart")


def check_an_escape_is_inconclusive_not_a_pass() -> None:
    import contextlib
    import io

    def _raises(exc):
        raise exc

    # stderr is redirected too, and not decorously: `traceback.print_exc` writes
    # to stderr, so a stdout-only capture let this table test print a traceback on
    # every green run. A probe that shows a traceback while reporting PASS looks
    # broken, which is the same "looks like a failure it is not" shape this file
    # exists to stop -- one level down.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        got = [
            run_guarded(lambda: _raises(ImportError("no PyQt6 here"))),
            run_guarded(lambda: _raises(ZeroDivisionError("a bug in the template"))),
            run_guarded(lambda: PASS),
            run_guarded(lambda: NO_CONTEXT),
        ]
    want = [INCONCLUSIVE, INCONCLUSIVE, PASS, NO_CONTEXT]
    if got != want:
        bad("an exception escaping the measurement is inconclusive, not a pass",
            f"expected {want} and got {got}. The old tail returned "
            f"SystemExit(0) from both except branches, which made an "
            f"unmeasurable run indistinguishable from a green one")
        return
    ok("an exception escaping the measurement is inconclusive, not a pass",
       f"ImportError -> {INCONCLUSIVE}, any other exception -> {INCONCLUSIVE}, and "
       f"a real code passes through unchanged ({got[2]}, {got[3]})")


def check_nothing_measured_is_not_a_pass() -> None:
    table = [
        ("no route produced a verdict",
         [UNKNOWN, UNKNOWN], OVERLAP_CLEAR, INCONCLUSIVE),
        ("the ladder was unreadable",
         [GOT_CONTEXT, GOT_CONTEXT], OVERLAP_UNCHECKED, INCONCLUSIVE),
    ]
    got = [verdict_of(o, s)[0] for _n, o, s, _w in table]
    want = [w for _n, _o, _s, w in table]
    if got != want:
        bad("a run that measured nothing is never a pass",
            f"expected {want} and got {got}. 'I could not measure it' and 'I "
            f"measured it and it is fine' are different answers and the exit "
            f"code is the only place the difference can live")
        return
    ok("a run that measured nothing is never a pass",
       f"both unmeasurable states return {INCONCLUSIVE} rather than {PASS}, so a "
       f"step that learned nothing cannot be read as a green step")


def main() -> int:
    from PyQt6 import QtGui  # noqa: F401  (import check only)

    print("Qt OpenGL widget availability\n")
    print("product-level routes are enumerated in gl_route_matrix.py, which "
          "walks the CI capability ladder; this file asks only what that "
          "ladder does not\n")

    overlap_state, clashes = _overlap_report()
    for label, rung, index in clashes:
        print(f"  ::error::ROUTE OVERLAP: {label!r} is a subset of CI ladder "
              f"rung {index} ({rung!r}). It is walked twice; remove it from "
              f"ROUTES here.")

    outcomes = []
    for label, env, body in ROUTES:
        outcome, detail = probe(label, TEMPLATE.format(setup=_setup_for(env), body=body))
        outcomes.append(outcome)
        print(f"  {label:<44} {outcome:<13} {detail[:60]}")

    section("the machinery this run is reporting through")
    check_the_routes_were_all_walked(outcomes)
    check_outcome_of_reads_a_child_honestly()
    check_the_overlap_guard_reported_a_state(overlap_state, clashes)
    check_verdict_of_maps_every_state()
    check_an_escape_is_inconclusive_not_a_pass()
    check_nothing_measured_is_not_a_pass()

    code, why = verdict_of(outcomes, overlap_state, clashes)
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")

    print()
    print(f"--- {npass} passed, {nfail} failed, {nskip} skipped, {len(RESULTS)} "
          f"checks (expected {EXPECTED_CHECKS})")
    # The pin is a decision, not a rendering. Until this line it was read in
    # exactly two places, both f-strings inside `print()` above, so a run that
    # recorded three results instead of six printed "3 checks (expected 6)" and
    # returned the verdict its measurement produced, having lost three checks in
    # silence. The number was never wrong -- six `check_*` blocks, each called
    # once from `main()` -- which is exactly why nothing noticed: a correct
    # number that cannot be contradicted is decoration.
    #
    # The shape is `probe_painted_frame.py`'s, deliberately, and the tally is
    # left alone: recording the mismatch as a seventh result would make the
    # summary line say "7 checks (expected 6)" and put the lie in the output
    # rather than the exit code. DEFECT is the right verdict class -- it is
    # already "a fact about this file", not a fact about the machine.
    if len(RESULTS) != EXPECTED_CHECKS:
        print(f"RESULT: {VERDICT_LABELS[DEFECT]} (exit {DEFECT}) -- this run "
              f"recorded {len(RESULTS)} result(s) and the pin says "
              f"{EXPECTED_CHECKS}; a check block was lost or added, so the "
              f"verdict is not about a program that ran as declared")
        return DEFECT
    if nfail:
        print(f"RESULT: {VERDICT_LABELS[DEFECT]} (exit {DEFECT}) -- {nfail} of "
              f"{EXPECTED_CHECKS} checks failed, so the machinery this verdict "
              f"came from cannot be trusted and the measurement is not reported")
        return DEFECT
    if npass == 0:
        print("RESULT: DID NOT FINISH -- this run asserted nothing")
        return INCONCLUSIVE
    print(f"RESULT: {VERDICT_LABELS[code]} (exit {code}) -- {why}")
    return code


if __name__ == "__main__":
    raise SystemExit(run_guarded(main))
