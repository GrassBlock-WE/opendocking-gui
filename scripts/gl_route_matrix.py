"""Report what `odgui --check` answers under every route the CI ladder walks.

**The problem this closes.**

The capability step in the workbench job walks four Qt/OpenGL configurations and
stops at the first that returns 0, then asserts the answer is in the documented
exit vocabulary and that the `--check --json` payload agrees with the process
exit code. That is the right shape for a gate, and it is lossy in one specific
way: it stops at the first success, so a run that succeeds on rung 1 reports
nothing about rungs 2-4, and a run that fails on all four reports four bare exit
codes with no stage breakdown. The gate answers "which route, if any"; this file
answers "what did every route say, stage by stage", in one machine-readable
document, and never fails.

**This is not a second opinion on the route list.**

The declared `ROUTES` are the same four configurations the CI ladder walks, in
the same order, spelled with the same environment, plus an unconfigured
`default` baseline that marks itself `ladder_index = None` because CI has no
equivalent. An earlier version of this file declared eight routes of its own,
four of which nothing in CI ran -- which is the shape of defect this project has
already been bitten by twice, so they were dropped rather than kept as extra
coverage. Rungs that are worth adding but that CI does not walk are recorded in
`LADDER_CANDIDATES`, which is a recommendation list and is **not** run: a
recommendation recorded as a route would recreate the problem this file is
rewritten to avoid.

`summary.first_working_rung` is the gate's answer, precomputed. It is the lowest
`ladder_index` whose route got a working widget, which is the rule CI applies by
scanning the ladder in order -- computed rather than left to the caller so that
"first" cannot quietly mean "whichever happened to be listed first".

**Where the sibling probe fits.**

`qt_gl_probe.py` answers a different question and is kept separate on purpose:
it asks whether a bare `QOpenGLWidget` can get a context at all, plus two
`QT_OPENGL` values and a bare `QOpenGLContext` that are deliberately *not* in the
ladder, because they test the mechanism rather than the product's answer. The two
files therefore enumerate disjoint route sets, and `qt_gl_probe.py` asserts that
they have stayed disjoint instead of trusting a comment to keep it so.

**Why a matrix rather than a smarter probe.**

The useful answer is a set, not a value. Exit 5 says OpenGL is fine and the
*widget* is not, and whether any given remedy helps depends on the driver, the
platform plugin and the window system in ways that are not knowable from inside
one process. And the probe cannot try them itself: once `show()` has aborted the
process, whatever the environment was for that attempt is gone. Each route
therefore needs its own process, which is why this is a loop over subprocesses.

**Output shape: one JSON document on stdout, with a ``routes`` array.**

The document form was chosen over one-object-per-route (JSON Lines) because a CI
gate wants to ask "which rung worked" and `jq '.summary.first_working_rung'` is
one expression, whereas a per-line format makes the caller do the folding. The
human summary and the per-route progress lines go to **stderr**, so
``python scripts/gl_route_matrix.py > matrix.json`` leaves a file that parses
with ``json.load`` and nothing else. Both are stated here because mixing a log
into stdout is the mistake that makes a JSON file unparseable.

**Exit codes.** ``0`` every declared route was attempted -- including when none
of them produced a working widget, because that is the informative case and this
file is not the gate. ``2`` at least one declared route was **not attempted**,
which is the whole reason this file can be trusted: a matrix that silently shrank
is indistinguishable from a matrix that found nothing, and the first is a bug in
this file while the second is a finding. The two are separated structurally
rather than by a flag, below.

**How "not attempted" is detected, and why a counter was not enough.**

The naive version increments a counter inside the runner, so a route that was
never started is noticed. That version is defeated by the most likely bug there
is: a ``continue`` in the run loop, or a filter that drops a route. No counter
inside the loop moves, the count looks healthy, and the matrix reports a verdict
for four routes while five were declared -- silently. So the check compares
the declared names against the names that actually produced a record, at the top
level, after the loop. That is a structural cross-check, and it is the only
version that survives the mutation the acceptance criteria ask for.

A route that started and then hung is **not** "not attempted": the matrix did its
job, the probe is what hung. That is recorded as ``timed_out`` with no exit code,
counted separately in the summary, and does not change the exit code. Folding a
hang into "not attempted" would make a slow GPU driver look like a bug in here.

**Why this does not call ``odgui_check()`` from ``_gui_check.py``.**

That helper is the one place that knows how to ask the launcher, and reusing it
would normally be right. It cannot be here: it runs ``odgui`` from ``PATH`` in
*this* process's environment, so it cannot be given a per-route environment, and
its ``Check`` carries no stdout, stderr or wall duration, all three of which a
route matrix exists to report. What is reused instead is the part that must not
be duplicated -- the exit-code and verdict vocabulary -- imported by name from
that module, so a new exit code is one edit rather than two. The JSON read below
is a field read, not the first-line-of-prose heuristic that helper was written to
delete, so this is not a fifth copy of it.

**Platform note.** Every rung but `offscreen` asks for the `xcb` platform, which
exists on Linux and not on Windows. Run on a Windows dev machine, those rungs
report `3` (no GUI stack) because Qt cannot load a plugin that was not built for
the platform -- which is the correct answer for the configuration asked for, and
is why `default` is kept: it is the only rung here whose question the local
machine can answer. Only CI can measure rungs 1-4.

Run it:

    python scripts/gl_route_matrix.py
    python scripts/gl_route_matrix.py --timeout 180
    python scripts/gl_route_matrix.py --probe-cmd python --probe-arg stub_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

#: Reuse the launcher's own exit-code and verdict vocabulary by importing it, so
#: a new code is one edit in `_gui_check.py` and not a second table here that can
#: disagree with the first. A code this file has never heard of is reported as
#: such rather than folded into the nearest verdict: an unrecognised answer must
#: not be reported as a working one.
from _gui_check import (  # noqa: E402
    EXIT_NO_CONTEXT,
    EXIT_NO_GUI_STACK,
    EXIT_NO_WIDGET,
    EXIT_OK,
    VERDICT_NO_CONTEXT,
    VERDICT_NO_GUI_STACK,
    VERDICT_NO_WIDGET,
    VERDICT_OK,
)

EXIT_OK_MATRIX = 0
#: 2, not 1: the house meaning of 2 across these scripts is "did not finish", and
#: a declared route that never ran is exactly that. 1 is reserved for a route that
#: ran and disagreed with its expectation, which this file does not judge.
EXIT_INCOMPLETE = 2

#: The probe, by default. `--json` is not optional here: stage verdicts are the
#: point, and the prose report only has a first line to read.
DEFAULT_PROBE = ("odgui", "--check", "--json")

#: Generous, and deliberately the same number `_gui_check.py` uses, for the same
#: measured reason: the honest answer costs real time on a machine that cannot
#: supply a context, because Qt blocks inside `show()` for seconds before it
#: gives up. Eight routes at 120 s is a slow CI step, and that is the price of
#: not declaring a route un-attempted because it was slow.
DEFAULT_TIMEOUT = 120

#: How much of stdout/stderr to keep per route. Enough for the probe's report and
#: the payload, bounded so a route that prints a megabyte cannot produce a
#: megabyte-per-route JSON document. Truncation is marked, never silent.
MAX_STREAM_CHARS = 8000


@dataclass(frozen=True)
class Route:
    """One candidate configuration, and why it is in the list.

    `env` is applied to a **copy** of the parent environment, never to
    ``os.environ``: a route that mutated the parent would make every later route
    inherit it, and the matrix would report a clean sweep of routes that differ
    only in the order they ran.

    `ladder_index` is the route's 1-based position in the capability ladder CI
    walks, or None when it is not a rung. It is carried into the output because
    the gate is "the first rung that returns 0", and a caller that had to match
    on a route *name* to work that out would be matching on a string CI owns.
    """

    name: str
    env: dict = field(default_factory=dict)
    #: Why this candidate is here. Kept in the output so a reader can tell a
    #: deliberate sweep from an arbitrary list, and so a route that fails is a
    #: finding about a named hypothesis rather than a mystery.
    why: str = ""
    ladder_index: int | None = None

    def label(self) -> str:
        if not self.env:
            return "no overrides"
        return " ".join(f"{k}={v}" for k, v in sorted(self.env.items()))


#: The declared list -- and it is the **same list the CI capability ladder
#: walks**, in the same order, spelled with the same environment. That is the
#: whole reason it is written out absolutely rather than relative to whatever
#: the calling shell happens to have exported.
#:
#: An earlier version of this file declared eight routes of its own, four of
#: which CI never ran. That is the shape of thing this project keeps paying
#: for: a second enumeration of the same question, drifting from the first, with
#: nobody reading both. So the extra four were dropped rather than kept as
#: "extra coverage", and the two that were genuinely additive -- the
#: *combinations* of ladder rungs 2/3 and 2/4 -- are recorded below as
#: recommendations for the ladder rather than as routes this file walks.
#:
#: `default` is the one entry that is not a rung. CI always exports
#: `LIBGL_ALWAYS_SOFTWARE` and `QT_QPA_PLATFORM` at the job level, so it has no
#: unconfigured baseline; a dev machine does, and without it there is nothing to
#: compare the rungs against. It is marked `ladder_index = None` so the two
#: things cannot be confused by a caller that filters on the index.
ROUTES = (
    Route(
        "default",
        {},
        "the unconfigured baseline: what this machine answers with nothing "
        "overridden. Not a CI ladder rung -- CI always sets the two variables "
        "every rung builds on, so it has no equivalent",
        None,
    ),
    Route(
        "xcb_mesa_software",
        {"QT_QPA_PLATFORM": "xcb", "LIBGL_ALWAYS_SOFTWARE": "1"},
        "CI ladder rung 1: the xcb platform on Mesa's software rasteriser. The "
        "configuration CI already runs, so it is the baseline the other rungs "
        "are differences from",
        1,
    ),
    Route(
        "xcb_mesa_software_qt_software",
        {
            "QT_QPA_PLATFORM": "xcb",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "QT_OPENGL": "software",
        },
        "CI ladder rung 2: additionally route Qt's own OpenGL through Mesa. "
        "This is the advice the probe prints when the raw context works and the "
        "widget does not, and it is the remedy most likely to be the one",
        2,
    ),
    Route(
        "xcb_mesa_software_gl_integration_none",
        {
            "QT_QPA_PLATFORM": "xcb",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "QT_XCB_GL_INTEGRATION": "none",
        },
        "CI ladder rung 3: additionally stop the xcb plugin loading GLX at all, "
        "so a half-working GLX is not what the widget is built on",
        3,
    ),
    Route(
        "offscreen_mesa_software",
        {"QT_QPA_PLATFORM": "offscreen", "LIBGL_ALWAYS_SOFTWARE": "1"},
        "CI ladder rung 4: no window-system connection at all, so nothing in X11 "
        "or the display driver can be what breaks the widget",
        4,
    ),
)

#: Rungs CI does not walk, with the argument for adding each. Deliberately not
#: in `ROUTES`: this file reports on routes somebody runs, and a recommendation
#: recorded as a route would be exactly the second source of truth the declared
#: list exists to prevent. Promote one of these by editing both this tuple and
#: the ladder in `ci.yml`, never this one alone.
LADDER_CANDIDATES = (
    (
        "xcb_mesa_software_qt_software_gl_integration_none",
        {
            "QT_QPA_PLATFORM": "xcb",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "QT_OPENGL": "software",
            "QT_XCB_GL_INTEGRATION": "none",
        },
        "rungs 2 and 3 applied together. The two are different failure modes -- "
        "one changes which library draws, the other removes the GLX integration "
        "the widget would otherwise use -- so their combination is a third thing "
        "and neither rung failing says anything about it",
    ),
    (
        "offscreen_mesa_software_qt_software",
        {
            "QT_QPA_PLATFORM": "offscreen",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "QT_OPENGL": "software",
        },
        "rungs 2 and 4 applied together, for the same reason. Worth having "
        "because the offscreen rung is the only one that removes X11 from the "
        "picture, and the only thing it does not also change is which library "
        "Qt draws with",
    ),
)


#: Every key any route overrides, so each record can report what the *parent*
#: environment already had for it. Without this, a route called `default` run
#: from a shell that exported `QT_OPENGL=software` reports itself as a clean
#: baseline when it is not one, and the comparison across routes is meaningless.
_OVERRIDE_KEYS = tuple(sorted({k for r in ROUTES for k in r.env}))

#: Exit code -> verdict, keyed on the constants imported above. An exit code not
#: in here is reported as `unknown-exit-N`, never as the nearest neighbour.
_EXIT_TO_VERDICT = {
    EXIT_OK: VERDICT_OK,
    EXIT_NO_GUI_STACK: VERDICT_NO_GUI_STACK,
    EXIT_NO_CONTEXT: VERDICT_NO_CONTEXT,
    EXIT_NO_WIDGET: VERDICT_NO_WIDGET,
}


def _clip(text) -> str:
    """stdout/stderr, bounded and marked when clipped."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= MAX_STREAM_CHARS:
        return text
    return text[:MAX_STREAM_CHARS] + (
        f"\n... [{len(text) - MAX_STREAM_CHARS} more characters clipped]"
    )


def _parse_payload(stdout: str):
    """The probe's JSON object, or None with the reason printed text did not have one."""
    text = (stdout or "").strip()
    if not text.startswith("{"):
        return None
    try:
        loaded = json.loads(text)
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def raw_verdict(payload) -> dict:
    """What the cheap raw-context stage did, read from the payload's `raw` member.

    Named the same way the launcher's own advice is: the raw context is the stage
    that works, and the widget is the one that does not, so keeping them in
    separate fields is what lets a reader see the fault is in the second stage.
    """
    if not isinstance(payload, dict):
        return {"state": "absent",
                "detail": "the probe printed no JSON object, so it has no stage "
                          "verdicts to report"}
    raw = payload.get("raw")
    if not isinstance(raw, dict):
        return {"state": "absent", "detail": "the payload has no 'raw' member"}
    if raw.get("gl"):
        return {
            "state": "ok",
            "gl": raw.get("gl"),
            "functions": bool(raw.get("functions")),
            "platform": raw.get("platform"),
            "samples": raw.get("samples"),
            "waited_s": raw.get("waited"),
            "detail": f"raw context GL {raw.get('gl')}, function table "
                      f"{'usable' if raw.get('functions') else 'UNUSABLE'}",
        }
    return {"state": "failed",
            "detail": raw.get("problem") or "the raw stage reported no GL version"}


def widget_verdict(payload) -> dict:
    """What the viewport-widget stage did, from the payload's `widget` member.

    `skipped`, `ok` and `died` are three different faults and are kept apart: a
    skipped widget probe after a working raw context is the launcher's own bug
    (the payload guard in `odgui_launch_check.py` fails on it), whereas `died`
    is what a machine that cannot realise an OpenGL widget looks like.
    """
    if not isinstance(payload, dict):
        return {"state": "absent",
                "detail": "the probe printed no JSON object, so it has no stage "
                          "verdicts to report"}
    widget = payload.get("widget")
    if not isinstance(widget, dict):
        return {"state": "absent", "detail": "the payload has no 'widget' member"}
    if widget.get("gl"):
        return {
            "state": "ok",
            "gl": widget.get("gl"),
            "functions": bool(widget.get("functions")),
            "platform": widget.get("platform"),
            "samples": widget.get("samples"),
            "child_exit": widget.get("child_exit"),
            "waited_s": widget.get("waited"),
            "detail": f"viewport widget GL {widget.get('gl')}, "
                      f"{widget.get('samples')}x MSAA, function table "
                      f"{'usable' if widget.get('functions') else 'UNUSABLE'}",
        }
    if widget.get("skipped"):
        return {"state": "skipped", "detail": str(widget["skipped"])}
    if "child_exit" in widget:
        return {
            "state": "died",
            "child_exit": widget.get("child_exit"),
            "waited_s": widget.get("waited"),
            "detail": "the widget probe's child ended without answering, so Qt "
                      "could not realise an OpenGL widget even though the raw "
                      "context worked",
        }
    return {"state": "failed",
            "detail": "the payload reports a widget stage with no GL version"}


def _verdict_from_code(code):
    """The launcher's verdict for an exit code, or an honest 'unknown'."""
    if code is None:
        return "unavailable"
    if code in _EXIT_TO_VERDICT:
        return _EXIT_TO_VERDICT[code]
    return f"unknown-exit-{code}"


def run_route(route: Route, probe: list, timeout: float, base_env: dict) -> dict:
    """Run the probe once under this route's overrides and record what came back.

    Never raises. Every terminal state produces a record, because a record is
    what the caller counts: a route that raised here would leave no record at
    all, and a missing record is indistinguishable from a route that was never
    in the list.
    """
    env = dict(base_env)
    env.update(route.env)
    inherited = {k: base_env.get(k) for k in _OVERRIDE_KEYS}
    record = {
        "route": route.name,
        "why": route.why,
        "ladder_index": route.ladder_index,
        "in_ci_ladder": route.ladder_index is not None,
        "env": dict(route.env),
        "env_label": route.label(),
        "inherited_env": inherited,
        "command": list(probe),
        "exit_code": None,
        "duration_s": None,
        "attempted": False,
        "timed_out": False,
        "verdict": "unavailable",
        "problem": None,
        "raw": {"state": "absent", "detail": "the probe was never started"},
        "widget": {"state": "absent", "detail": "the probe was never started"},
        "payload": None,
        "payload_parsed": False,
        "stdout": "",
        "stderr": "",
    }

    started = time.monotonic()
    try:
        proc = subprocess.run(
            probe,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # A hang is the probe's answer, not this file's failure: the route was
        # attempted, and saying so keeps a slow driver from looking like a bug.
        record["timed_out"] = True
        record["attempted"] = True
        record["verdict"] = "timeout"
        record["raw"] = {"state": "unknown",
                         "detail": f"the probe did not finish within {timeout:g} s"}
        record["widget"] = {"state": "unknown",
                            "detail": f"the probe did not finish within {timeout:g} s"}
        record["stdout"] = _clip(exc.stdout)
        record["stderr"] = _clip(exc.stderr)
        return _finish(record, started)
    except (OSError, subprocess.SubprocessError) as exc:
        # The one honest "not attempted": the process could not be started at
        # all. Reported as a record, and counted as unattempted below, because
        # the matrix did not get to ask.
        record["attempted"] = False
        record["problem"] = f"{type(exc).__name__}: {exc}"
        record["raw"] = {"state": "absent", "detail": record["problem"]}
        record["widget"] = {"state": "absent", "detail": record["problem"]}
        return _finish(record, started)

    record["attempted"] = True
    record["exit_code"] = proc.returncode
    record["stdout"] = _clip(proc.stdout)
    record["stderr"] = _clip(proc.stderr)
    payload = _parse_payload(proc.stdout)
    record["payload"] = payload
    record["payload_parsed"] = payload is not None
    record["raw"] = raw_verdict(payload)
    record["widget"] = widget_verdict(payload)
    if isinstance(payload, dict):
        record["verdict"] = str(payload.get("verdict") or _verdict_from_code(proc.returncode))
        record["problem"] = payload.get("problem")
    else:
        # No JSON: fall back to the documented exit code, and say in the record
        # that the verdict came from the code alone, because a reader deciding
        # which route to wire into CI is entitled to know how much was measured.
        record["verdict"] = _verdict_from_code(proc.returncode)
        record["problem"] = (
            "the probe printed no JSON object, so this verdict comes from the "
            "exit code alone; run it by hand to read the prose report"
        )
    return _finish(record, started)


def _finish(record: dict, started: float) -> dict:
    record["duration_s"] = round(time.monotonic() - started, 3)
    # The single line CI should branch on, carried in the record so a caller does
    # not have to re-derive the launcher's own exit-code contract.
    record["working_widget"] = (
        record["exit_code"] == EXIT_OK and record["widget"].get("state") == "ok"
    )
    return record


def _missing(declared, ran) -> list:
    """Declared route names with no record. Structural, not a counter.

    A counter incremented inside the runner is not enough: a `continue` in the
    run loop, or a filter that drops a route, leaves it untouched while the
    matrix reports a verdict for fewer routes than it declared. Comparing the
    two name lists after the loop is the only version that notices, and it is
    the mutation the acceptance criteria ask to be proven.
    """
    return [name for name in declared if name not in ran]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Try every declared Qt/OpenGL route and report one verdict each."
    )
    ap.add_argument(
        "--probe-cmd",
        default=DEFAULT_PROBE[0],
        help="the program to probe (default: %(default)s)",
    )
    ap.add_argument(
        "--probe-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="an argument for the probe; repeat for more than one "
             f"(default: {' '.join(DEFAULT_PROBE[1:])})",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="per-route timeout in seconds (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    probe = [args.probe_cmd] + (args.probe_arg if args.probe_arg is not None
                                else list(DEFAULT_PROBE[1:]))

    base_env = dict(os.environ)
    declared = [r.name for r in ROUTES]

    if shutil.which(probe[0]) is None and not os.path.exists(probe[0]):
        print(f"matrix: the probe program {probe[0]!r} is not on PATH and is not "
              f"a file, so no route can be attempted", file=sys.stderr)

    print(f"matrix: probing {' '.join(probe)} under {len(ROUTES)} route(s), "
          f"{args.timeout:g} s each", file=sys.stderr)

    records = []
    for route in ROUTES:
        record = run_route(route, probe, args.timeout, base_env)
        records.append(record)
        print(
            f"  {record['route']:<20} exit {str(record['exit_code']):>4}  "
            f"verdict {record['verdict']:<12}  raw: {record['raw']['state']:<7}  "
            f"widget: {record['widget']['state']:<7}  "
            f"{record['duration_s']:>7.2f}s  {record['env_label']}",
            file=sys.stderr,
        )

    ran = [r["route"] for r in records]
    missing = _missing(declared, ran)
    working = [r["route"] for r in records if r["working_widget"]]
    timed_out = [r["route"] for r in records if r["timed_out"]]
    not_started = [r["route"] for r in records if not r["attempted"]]

    # The gate is "the first rung that returns 0", so the answer is computed
    # here rather than left to a caller: the rungs are ordered, and picking the
    # lowest `ladder_index` among the winners is the rule CI applies. A caller
    # that scanned `routes` in order and took the first hit would get the same
    # answer, but only by accident of list order -- and list order is not the
    # thing that makes it right.
    rungs = sorted(
        (r for r in records if r["ladder_index"] is not None),
        key=lambda r: r["ladder_index"],
    )
    winning_rungs = [r for r in rungs if r["working_widget"]]
    first_working_rung = winning_rungs[0] if winning_rungs else None

    document = {
        "schema": "gl_route_matrix/1",
        "probe": probe,
        "timeout_s": args.timeout,
        "routes": records,
        "ladder_candidates_not_walked": [
            {"route": name, "env": env, "why": why}
            for name, env, why in LADDER_CANDIDATES
        ],
        "summary": {
            "declared": len(declared),
            "attempted": sum(1 for r in records if r["attempted"]),
            "not_started": not_started,
            "not_attempted": missing,
            "timed_out": timed_out,
            "working_widget": working,
            "attempted_count": len(records) - len(not_started),
            "working_widget_count": len(working),
            "rungs_walked": [r["route"] for r in rungs],
            "working_rungs": [r["route"] for r in winning_rungs],
            "first_working_rung": first_working_rung,
        },
    }

    # The JSON document goes to stdout and nothing else does, so
    # `> matrix.json` yields a file that `json.load` accepts. The log above is
    # on stderr for that reason.
    print(json.dumps(document, indent=2, sort_keys=True, default=str))

    print(
        f"matrix: {len(records)} of {len(declared)} route(s) attempted, "
        f"{len(working)} produced a working widget, {len(timed_out)} timed out"
        + (f"; working: {', '.join(working)}" if working else ""),
        file=sys.stderr,
    )

    if missing:
        # The only non-zero exit. Not a gate verdict: the run produced its
        # verdicts, and this says the *list* was not fully walked, which is a
        # defect in this file rather than a fact about the machine.
        print(
            f"matrix: INCOMPLETE -- declared route(s) never attempted: "
            f"{', '.join(missing)}. Every declared route must be attempted; "
            f"a matrix that shrank reports nothing about the routes it skipped.",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE
    return EXIT_OK_MATRIX


if __name__ == "__main__":
    raise SystemExit(main())
