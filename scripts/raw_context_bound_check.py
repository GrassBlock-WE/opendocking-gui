"""The raw-context stage's deadline, asked of a driver that never answers.

Run:  python scripts/raw_context_bound_check.py

# What is being guarded

`launcher._probe_raw_context` asks Qt for an OpenGL context on one daemon thread
and joins that thread against `_RAW_CONTEXT_SECONDS`. The bound is the whole
reason the command is not a hang: a driver that blocks inside
`QOffscreenSurface.create()` or `QOpenGLContext.makeCurrent()` cannot be
interrupted, so without a join timeout the wait belongs to the driver rather
than to the launcher. This gate establishes that the bound is **applied**, by
asking the question of a driver that never comes back and reading the clock.

It is behavioural because the other candidate was measured and is not a guard.
`check_counted_constants.py` reads the shipped `launcher.py` and goes green on
it -- and it also went green on a copy with `worker.join(_RAW_CONTEXT_SECONDS)`
replaced by `worker.join()`. A check that reads a constant verifies that the
constant is spelled the way the check spells it. This one reads the deadline
only to size its budget; every assertion below is about an observed elapsed
time.

# The three outcomes, and which of them is a pass

The failure to avoid is a machine-dependent branch that reports "I could not
find out" as "yes" -- a permanent green carrying no information. So the
outcomes are kept apart by name, counted apart, and printed apart:

* **the bound did its job** -- a `PASS`. The call came back at the bound and the
  record said the bound lapsed. Establishable on any machine, because the fake
  surface never reaches `QOpenGLContext`.
* **no context available here** -- a `SKIP`, for the one claim that needs a
  real `QOpenGLContext.create()` to return True. Counted toward
  `EXPECTED_CHECKS`, excluded from the passed count, and printed with the
  platform and the probe's own answer attached.
* **the bound is gone** -- a `FAIL`, and the thing this file exists to catch.

**Why the skip is not `continue-on-error` in disguise.** The claim that carries
the guard -- *the deadline is applied* -- rests only on directions 1 and 2,
which install a fake `QOffscreenSurface` whose `create()` blocks before
`QOpenGLContext` is ever constructed. They are unconditional: they run
identically on a machine that cannot make a context, and they are the reason
this file is not a gate with a permanent green in it. Direction 3 runs a real
surface and is where the machine can differ; on this box it establishes three
of its four claims and skips the fourth, naming it. A `continue-on-error` step
discards the result of the whole run; here the result is computed, checked
against a constant total, and turns the step red when the deadline is deleted.
The skip removes one assertion and says so in the summary rather than in the
pass count.

# The directions, and what each would cost to fake

    1  shipped launcher, driver wedged     returns at the bound, and says so
    2  deadline deleted, driver wedged     does NOT return inside the budget
    3  shipped launcher, real driver       returns promptly, and does not
                                          claim a lapsed bound

Direction 2 is the control, and it is what makes direction 1 a measurement
rather than an assertion: same harness, same fake, same budget, and the only
difference is one line of a **sandbox copy** under `target/`. The live
`launcher.py` is hashed before and after and the run fails if it moved, so this
file cannot prove anything by editing the thing it measures.

Direction 3 is the half a wedged-only gate never runs, and on this machine it
is the one that carries the most information: a real offscreen surface
**answers** -- it fails to yield a context rather than blocking -- so the probe
comes back in well under a second and says "no context" instead of "the bound
lapsed". That is the distinction this file is built around, and it is
observable here.

# What it costs

Three sequential subprocesses, about `2 x (_RAW_CONTEXT_SECONDS + margin)` plus
the capability probe, and **one GL process at a time**: nothing else in this
repository should be running a GL route alongside them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "dock-py/python"
LAUNCHER = PACKAGE / "opendocking/workbench/launcher.py"
#: Build scratch. `target/` is pruned by `release_tree_rule.py` and is in
#: `.gitignore`, so a mutated copy of a live module here is read by no gate and
#: shipped by nothing -- which is the whole requirement on where it may live.
SCRATCH = ROOT / "target" / "raw_context_bound_check"

#: The line whose deletion is the pre-fix behaviour. Matched exactly, and its
#: uniqueness is asserted before the copy is written, so a reformat of the
#: shipped file fails loudly here instead of silently mutating nothing.
DEADLINE = "    worker.join(_RAW_CONTEXT_SECONDS)\n"
UNBOUNDED = "    worker.join()\n"

#: How long the fake driver sleeps, and how far past the deadline this gate is
#: willing to wait before calling the call unbounded. The margin is this file's
#: and the bound is the module's; neither is typed twice.
FAKE_WEDGE_S = 600.0
MARGIN_S = 20.0
CAPS_TIMEOUT_S = 120.0

#: 1  section 0, the context question was answered rather than assumed
#: + 6  section 1, the shipped launcher against a driver that never answers
#: + 1  section 2, the same question with the deadline deleted
#: + 4  section 3, a driver that answers
#: + 3  the meta-checks in the summary, which count themselves
#: = 15
#:
#: **Every one of the fifteen is unconditional, and that is the property the
#: file is designed around.** The fake surface in sections 1 and 2 blocks
#: before `QOpenGLContext` is constructed, so no site here needs a GL context;
#: section 3's fourth site is a recorded skip rather than an absent check, so
#: the total is the same on a machine with a context and on one without. A site
#: that vanished with the machine would make a smaller run look like a smaller
#: scope, which is the distinction the whole ledger exists to keep.
#:
#: The summary carries three rather than one, and the split was not planned. The
#: first version of this file asserted "the three counts partition the ledger"
#: and "the total is the declared one" in a single expression that added one to
#: both sides, and reported a correct run as 13 records against a declared 14.
#: Two claims were welded into one and the weld was the bug, so they are two
#: checks now: the partition, and the total this run will report.
EXPECTED_CHECKS = 15  # measured from a green run of this file on this tree, counted by running it

#: The call-site census, declared rather than transcribed into a table some
#: other gate owns. Derived by `census()` at the bottom of this file, from this
#: file's own syntax tree and by the same walk the auditor uses, for the reason
#: the auditor's own comments give: a hand-copied guard digest is the largest
#: uncheckable fact a repository can hold. `census()` re-derives all three
#: numbers, so the block below is a claim about this file rather than a
#: transcription of it.
#:
#: **The five guarded sites are the ones whose running depends on the machine or
#: on which branch a child took**, and the census is what makes that visible
#: without reading the file: `if can_context` is section 3's last site, and the
#: other four are section 2's control, which takes one of three branches. The
#: guard *strings* are in the digest, so moving a site between those branches
#: without changing the total is still a change -- which is the case a bare
#: count cannot see. `EXPECTED_CHECKS` above is unaffected by any of this: it
#: counts what a run reports, and the four-way split in section 2 is four
#: mutually exclusive reports of one question.
#: GATE-DECLARE 1
#: sites: 13 unconditional + 5 guarded
#: guards: sha256:4c662a9f649edd96e01ee8ec1e598e47f3b387c8918d0ec758d3b167fec5821f

RESULTS: list[tuple[str, str, str]] = []
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def skip(name: str, reason: str) -> None:
    """A question this machine cannot answer, recorded as its own outcome.

    Counts toward `EXPECTED_CHECKS` so the total is the same everywhere, and is
    **not** in the passed count: a skip is not a pass and does not make the step
    green on its own. The reason is required, so a skip cannot be a quiet way of
    dropping a site.
    """
    RESULTS.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  - {reason}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def shipped_bound() -> float:
    """`_RAW_CONTEXT_SECONDS`, read out of the shipped file rather than repeated.

    Read for the budget only. The assertions are about observed elapsed time,
    so a file that *declares* a bound it does not apply fails in section 1
    rather than passing on the strength of the declaration.
    """
    for line in LAUNCHER.read_text(encoding="utf-8").splitlines():
        if line.startswith("_RAW_CONTEXT_SECONDS = "):
            return float(line.split("=", 1)[1])
    raise AssertionError("no _RAW_CONTEXT_SECONDS in the shipped launcher")


#: The two children, as source. Written out rather than imported so that the
#: thing being measured is loaded and called exactly as the launcher calls it,
#: in a process this gate can kill. A fake installed into this process's own
#: namespace is a fake this file could get wrong.
_WEDGED_CHILD = '''
import json, os, sys, time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtGui


class _WedgedSurface:
    """A surface whose `create()` never returns, as a wedged driver's would not.

    `QOpenGLContext` is left real and is never reached, which is what lets
    sections 1 and 2 run on a machine with no GL context at all.
    """

    def __init__(self):
        self._fmt = None

    def setFormat(self, fmt):
        self._fmt = fmt

    def create(self):
        time.sleep(float(os.environ.get("OD_FAKE_WEDGE_S", "600")))

    def isValid(self):
        return True

    def destroy(self):
        pass


QtGui.QOffscreenSurface = _WedgedSurface

from opendocking.workbench import launcher

began = time.perf_counter()
found, problem = launcher._probe_raw_context()
elapsed = time.perf_counter() - began

print(json.dumps({
    "elapsed_s": round(elapsed, 3),
    "declared_bound": launcher._RAW_CONTEXT_SECONDS,
    "problem_is_none": problem is None,
    "problem": problem,
    "record": {k: v for k, v in sorted(found.items())},
}, sort_keys=True, default=str))
sys.stdout.flush()
'''

_REAL_CHILD = '''
import json, os, sys, time

# No fake. The real QOffscreenSurface and the real QOpenGLContext, so what comes
# back is what this machine does when nothing is standing in the way.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from opendocking.workbench import launcher

began = time.perf_counter()
found, problem = launcher._probe_raw_context()
elapsed = time.perf_counter() - began

print(json.dumps({
    "elapsed_s": round(elapsed, 3),
    "declared_bound": launcher._RAW_CONTEXT_SECONDS,
    "problem_is_none": problem is None,
    "problem": problem,
    "record": {k: v for k, v in sorted(found.items())},
}, sort_keys=True, default=str))
sys.stdout.flush()
'''

_CAPS_CHILD = '''
import json, os, sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtGui, QtWidgets

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
out = {"platform": QtGui.QGuiApplication.platformName(), "can_create": False}
surface = None
context = None
try:
    fmt = QtGui.QSurfaceFormat()
    fmt.setSamples(0)
    fmt.setDepthBufferSize(24)
    surface = QtGui.QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    out["surface_valid"] = bool(surface.isValid())
    if out["surface_valid"]:
        context = QtGui.QOpenGLContext()
        out["can_create"] = bool(context.create())
        if out["can_create"]:
            out["can_make_current"] = bool(context.makeCurrent(surface))
            if out.get("can_make_current"):
                now = context.format()
                out["gl"] = "%d.%d" % (now.majorVersion(), now.minorVersion())
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
finally:
    if context is not None:
        try:
            context.doneCurrent()
        except Exception:
            pass
    for obj in (context, surface):
        if obj is not None:
            try:
                obj.destroy()
            except Exception:
                pass

print(json.dumps(out, sort_keys=True, default=str))
sys.stdout.flush()
'''


def _force_remove(func, path, _exc) -> None:
    """Last resort for a `.pyc` this process will not let go of."""
    try:
        os.chmod(path, 0o700)
    except OSError:
        return
    try:
        func(path)
    except OSError:
        pass


def run_child(source: str, package_root: Path, budget: float, tag: str):
    """Run one child against one copy of the package.

    Returns `(state, payload)`, where `state` is `"answered"`, `"unbounded"` or
    `"broken"`. The distinction is the gate: a child still inside the driver at
    the budget is the pre-fix behaviour, while a child that could not start is
    a different failure that says so rather than being folded into it.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"child_{tag}.py"
    path.write_text(source, encoding="utf-8", newline="\n")
    env = dict(
        os.environ,
        PYTHONPATH=str(package_root),
        QT_QPA_PLATFORM="offscreen",
        OD_FAKE_WEDGE_S=str(FAKE_WEDGE_S),
    )
    # The child writes UTF-8 to a pipe this gate decodes explicitly, so a
    # console encoding cannot decide whether this gate completes.
    env.pop("PYTHONIOENCODING", None)
    began = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, cwd=str(ROOT), timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return "unbounded", {
            "why": f"still inside the driver at {budget:.0f} s "
                   f"({time.perf_counter() - began:.1f} s wall) and was killed",
        }
    if proc.returncode != 0:
        return "broken", {
            "why": f"exit {proc.returncode}",
            "stderr": proc.stderr.strip()[-500:],
        }
    try:
        return "answered", json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return "broken", {"why": "unparseable stdout", "stdout": proc.stdout[-300:]}


def main() -> int:
    print("the raw-context deadline, asked of a driver that never answers")
    print(f"repository {ROOT}")
    print(f"live launcher.py {sha(LAUNCHER)}\n")

    before = sha(LAUNCHER)
    bound = shipped_bound()
    budget = bound + MARGIN_S
    print(f"_RAW_CONTEXT_SECONDS = {bound:g} s, margin {MARGIN_S:g} s, so a "
          f"bounded call must return by {budget:g} s")
    print(f"the fake driver sleeps {FAKE_WEDGE_S:g} s, so an unbounded call is "
          f"still inside it when the budget expires\n")

    # -- section 0: can this machine supply a context at all? -------------
    # Asked first, in its own process, with no fake installed, because its
    # answer decides which outcome section 3's last site is looking at, and it
    # is the one question here this file is not entitled to assume.
    section("0. whether this machine can supply a context at all")
    cap_state, caps = run_child(_CAPS_CHILD, PACKAGE, CAPS_TIMEOUT_S, "caps")
    if cap_state == "broken":
        can_context = False
        cap_why = f"the capability probe did not run: {caps.get('why')}"
    else:
        can_context = bool(caps.get("can_create"))
        cap_why = (
            f"platform {caps.get('platform')!r}, surface valid "
            f"{caps.get('surface_valid')!r}, QOpenGLContext.create() -> "
            f"{caps.get('can_create')!r}"
        )
        if can_context:
            cap_why += (f", GL {caps.get('gl')!r}, makeCurrent -> "
                        f"{caps.get('can_make_current')!r}")
        if caps.get("error"):
            cap_why += f", {caps['error']}"
    print(f"  {cap_why}")
    check(
        "the context question was answered, so the run below knows which of "
        "its two outcomes it is looking at",
        cap_state == "answered",
        cap_why + ("" if cap_state == "answered" else
                   " -- without this, section 3's last site would have nothing "
                   "to compare a context against"),
    )

    # -- section 1: the shipped launcher, a wedged driver -----------------
    section("1. the launcher as shipped, against a driver that never answers")
    state, one = run_child(_WEDGED_CHILD, PACKAGE, budget, "shipped")
    if state == "unbounded":
        detail = (f"{one['why']} -- the wait belongs to the driver and not to "
                  f"the launcher, which is the defect this file exists to catch")
    elif state == "broken":
        detail = f"{one.get('why')}: {one.get('stderr', '')}"
    else:
        detail = f"came back in {one['elapsed_s']:.2f} s of a {budget:g} s budget"
    print(f"  {detail}")
    rec = one.get("record", {}) if state == "answered" else {}
    check(
        "a wedged driver costs this stage a lapsed bound, not the command",
        state == "answered" and one["elapsed_s"] < budget,
        detail,
    )
    check(
        "and it came back *at* the bound rather than before it",
        state == "answered" and one["elapsed_s"] >= bound * 0.9,
        f"{one['elapsed_s']:.2f} s against a declared {bound:g} s"
        if state == "answered" else
        f"not measured: {detail}. A call that returned instantly did not "
        f"observe a bound, it skipped one",
    )
    check(
        "the record says the bound lapsed, and did not claim a context",
        state == "answered" and rec.get("bound_expired") is True
        and rec.get("initialised") is False and one["problem_is_none"] is False,
        f"bound_expired={rec.get('bound_expired')!r}, "
        f"initialised={rec.get('initialised')!r}, problem is None="
        f"{one['problem_is_none']!r}" if state == "answered" else
        f"not measured: {detail}",
    )
    check(
        "the facts the worker did establish before the driver took it survived "
        "the timeout",
        state == "answered" and bool(rec.get("platform"))
        and rec.get("samples") is not None,
        f"platform={rec.get('platform')!r}, samples={rec.get('samples')!r}, "
        f"surface_valid={rec.get('surface_valid')!r}" if state == "answered" else
        f"not measured: {detail}. A bounded wait that discards what it learned "
        f"is a different defect from an unbounded one, and the record is where "
        f"the two are told apart",
    )
    check(
        "the reason does not claim the machine cannot do OpenGL",
        state == "answered" and "NOT the claim that it cannot" in (one["problem"] or ""),
        f"{(one['problem'] or '')[:110]}" if state == "answered" else
        f"not measured: {detail}",
    )
    check(
        "and the bound is in the record as a number, not only in prose",
        state == "answered" and rec.get("bound_seconds") == bound
        and one["declared_bound"] == bound,
        f"raw.bound_seconds={rec.get('bound_seconds')!r}, module constant="
        f"{one['declared_bound']!r}, read from the shipped file={bound!r}"
        if state == "answered" else f"not measured: {detail}",
    )

    # -- section 2: the same question, deadline deleted -------------------
    section("2. a sandbox copy with the deadline removed: same driver, same budget")
    text = LAUNCHER.read_text(encoding="utf-8")
    count = text.count(DEADLINE)
    if count != 1:
        check(
            "with the deadline gone the same call does not come back at all",
            False,
            f"the shipped launcher holds {count} occurrences of "
            f"{DEADLINE.strip()!r}, so this gate's control cannot be built: the "
            f"deadline has been reformatted or removed and direction 1 is an "
            f"assertion with nothing to compare it against",
        )
    else:
        # The **whole package**, copied, with one line of one file changed.
        # A partial tree was tried first and failed: a sandbox holding only
        # `launcher.py` puts a regular package named `opendocking` on
        # `sys.path` whose `__path__` is the sandbox alone, so the shipped
        # `app.py` beside it is invisible and the child dies on
        # `from opendocking.workbench.app import MSAA_SAMPLES` before it ever
        # reaches the line under test. Copying the package keeps the two runs
        # differing in exactly one line, which is the property that makes this
        # a control.
        copy_root = SCRATCH / "tree_no_deadline"
        # `rmtree` before `copytree`, because a previous run's tree holds
        # `__pycache__` directories whose `.pyc` files the last child process
        # may still have mapped. On Windows that is a `PermissionError` from
        # deep inside the copy, and the first version of this line surfaced it
        # as an escaped exception: the gate reported "DID NOT FINISH" on a
        # second run of a gate whose first run was green, which is the worst
        # of the three outcomes -- a gate that only works once.
        if copy_root.exists():
            shutil.rmtree(copy_root, ignore_errors=True)
            if copy_root.exists():
                shutil.rmtree(copy_root, onerror=_force_remove)
        shutil.copytree(PACKAGE / "opendocking", copy_root / "opendocking",
                        ignore=shutil.ignore_patterns("__pycache__"))
        copy = copy_root / "opendocking/workbench/launcher.py"
        copy.write_text(
            text.replace(DEADLINE, UNBOUNDED), encoding="utf-8", newline="")
        print(f"  sandbox launcher.py {sha(copy)}  (live file untouched)")
        print(f"  the two runs differ in exactly one line: "
              f"{DEADLINE.strip()!r} -> {UNBOUNDED.strip()!r}")
        print(f"  measured against the sandbox's own _dockpy.pyd, which is this "
              f"tree's copy of it\n")
        state2, two = run_child(_WEDGED_CHILD, copy_root, budget, "no_deadline")
        if state2 == "unbounded":
            check(
                "with the deadline gone the same call does not come back at all",
                True,
                f"{two['why']}. This is the defect the bound removes, measured "
                f"rather than argued: same driver, same harness, same budget -- "
                f"the shipped launcher answers inside it and this copy does not "
                f"answer at all",
            )
        elif state2 == "broken":
            check(
                "with the deadline gone the same call does not come back at all",
                False,
                f"the mutated copy failed to run ({two.get('why')}), so the "
                f"control never ran: {two.get('stderr', '')}",
            )
        else:
            check(
                "with the deadline gone the same call does not come back at all",
                False,
                f"the mutated copy returned in {two['elapsed_s']:.2f} s, so the "
                f"deadline is not what bounded the shipped run "
                f"(bound_expired={two['record'].get('bound_expired')!r}). Either "
                f"this is not the line that bounds the call or something else "
                f"does, and this gate cannot tell which",
            )

    # -- section 3: a driver that answers ---------------------------------
    # The half a wedged-only gate never runs. A real offscreen surface on this
    # machine **answers** -- it fails to yield a context rather than blocking --
    # so the probe comes back in well under a second and says "no context"
    # instead of "the bound lapsed". That distinction is the file's subject.
    section("3. a real driver, so the probe is not only a timer")
    state3, three = run_child(_REAL_CHILD, PACKAGE, budget, "healthy")
    r3 = three.get("record", {}) if state3 == "answered" else {}
    ran = state3 == "answered"
    print(f"  {three['elapsed_s']:.2f} s of a {budget:g} s budget, "
          f"bound_expired={r3.get('bound_expired')!r}, "
          f"initialised={r3.get('initialised')!r}"
          if ran else f"  did not answer: {three.get('why')}")
    check(
        "a real driver answers inside the budget instead of hanging",
        ran and three["elapsed_s"] < budget,
        f"{three['elapsed_s']:.2f} s, surface_valid={r3.get('surface_valid')!r}, "
        f"context_created={r3.get('context_created')!r}" if ran else
        f"not measured: {three.get('why')}",
    )
    check(
        "and the probe does not blame its deadline for that answer",
        ran and r3.get("bound_expired") is not True,
        f"bound_expired={r3.get('bound_expired')!r} (the key is **absent** on the "
        f"non-expired path, not False -- `_probe_raw_context` sets it only under "
        f"`if expired:`, so a reader must test it against True and not against "
        f"False or this check fails on a healthy driver), waited="
        f"{r3.get('waited')!r} s, problem={(three.get('problem') or '')[:70]!r}. A "
        f"probe reporting a lapsed bound here would be reporting a wedged driver "
        f"on a machine that simply cannot make a context, which is the confusion "
        f"this file exists to prevent"
        if ran else f"not measured: {three.get('why')}",
    )
    check(
        "the no-context outcome is told apart from the lapsed-bound one",
        ran and one.get("record", {}).get("bound_expired") is True
        and r3.get("bound_expired") is not True,
        # Reported from whichever side actually answered. The first version of
        # this line interpolated the wedged record unconditionally, so when
        # that run hung it printed "bound_expired=True in None s" -- a
        # confident sentence about a record that does not exist, which is the
        # failure mode this whole file is about.
        (f"wedged: bound_expired=True in {one['elapsed_s']} s; real: "
         f"bound_expired={r3.get('bound_expired')!r} in {three['elapsed_s']} s "
         f"-- {one['elapsed_s'] - three['elapsed_s']:.2f} s apart, and the two "
         f"records disagree about the bound in the only way that matters")
        if ran and one.get("record", {}).get("bound_expired") is True else
        (f"the real driver returned in {three.get('elapsed_s')} s with "
         f"bound_expired={r3.get('bound_expired')!r}, and the wedged run did not "
         f"answer at all ({one.get('why', one.get('elapsed_s'))}), so there is "
         f"no pair of records to compare. The two are told apart by the wedged "
         f"run reporting a lapsed bound and this one not, and the first half of "
         f"that is missing"
         if ran else f"not measured: {three.get('why')}"),
    )
    if can_context:
        check(
            "and on a machine that can supply one, the probe returns a context "
            "rather than a reason",
            ran and r3.get("initialised") is True and three["problem_is_none"] is True,
            f"initialised={r3.get('initialised')!r}, gl={r3.get('gl')!r}, "
            f"functions={r3.get('functions')!r}, problem is None="
            f"{three['problem_is_none']!r}" if ran else
            f"not measured: {three.get('why')}",
        )
    else:
        skip(
            "and on a machine that can supply one, the probe returns a context "
            "rather than a reason",
            f"this machine cannot supply one, and it is not a guess: the probe "
            f"above measured {cap_why}. The bound's own directions ran anyway -- "
            f"the fake surface blocks before QOpenGLContext is constructed -- so "
            f"this removes one assertion and not the guard. The three checks "
            f"above it did run, and the one above them is the distinction that "
            f"matters here: a driver that cannot answer is recorded as a lapsed "
            f"bound and a driver that answers without a context is not. Run this "
            f"on a machine where a context is created to answer the fourth",
        )

    # -- the accounting ---------------------------------------------------
    section("summary")
    after = sha(LAUNCHER)
    check(
        "the live launcher.py is byte-identical before and after this run",
        after == before,
        f"{before} -> {after}. This gate mutates a copy under target/ and "
        f"nothing else; a live file that moved is a gate that cannot be trusted "
        f"to have measured anything",
    )
    passed = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    failed = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    skipped = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    # Two separate claims, and the first run of this file asserted them as one
    # and reported "13 records against a declared 14" on a run that was correct
    # in every particular -- the counts partition the ledger *before* this check
    # registers, and the total it will report is that plus this check. Asserting
    # `counts == len(RESULTS) + 1` asks the partition to cover a record that does
    # not exist yet, which is a red that means nothing.
    check(
        "every result is one of the three named outcomes, and the three counts "
        "partition the ledger",
        passed + failed + skipped == len(RESULTS)
        and {t for t, _n, _d in RESULTS} <= {"PASS", "FAIL", "SKIP"},
        f"before this check: {passed} pass + {failed} fail + {skipped} skip = "
        f"{len(RESULTS)} records, and {passed + failed + skipped} accounted for. "
        f"A fourth outcome would break the summary line above, which is where "
        f"the skip stops being a silent pass",
    )
    check(
        "and this file's own count is the count it declares",
        len(RESULTS) + 1 == EXPECTED_CHECKS,
        f"{len(RESULTS)} ran before these two and {EXPECTED_CHECKS} are "
        f"declared, so the total a run reports is the declared one on this "
        f"machine. The skip counted toward it, which is the point: a machine "
        f"that cannot answer a question has a smaller scope and not a smaller "
        f"number, and a run that lost a site to a machine-dependent branch "
        f"would land here",
    )
    print(f"\n  {len(RESULTS)} checks: {passed} passed, {failed} failed, "
          f"{skipped} could not be measured here")
    if FAILURES:
        print("RESULT: FAIL")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    if passed == 0:
        print("RESULT: DID NOT FINISH - nothing was verified")
        return 2
    print("RESULT: PASS" + (f" ({skipped} could not be measured here)" if skipped else ""))
    if skipped:
        print("  the pass count above excludes the skip: this run did not "
              "measure everything it declares, and the summary names which")
    return 0


def _short(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:56]
    except Exception:  # noqa: BLE001 - a guard string is a fingerprint, not code
        return "?"


def _sites_of(src: str) -> tuple[int, list[str]]:
    """`(unconditional, sorted guard strings)` for this file's check call sites.

    A reimplementation of the walk `check_scripts_declare.py` performs, matching
    it deliberately rather than approximately, because the whole value of a
    self-declared census is that it is comparable to the number the auditor
    derives. The semantics that matter, and that a first attempt at this got
    wrong: a site counts only when the call *is* the whole statement; the walk
    climbs from the call rather than from its parent, so the statement's own
    enclosing `if` is the first guard it sees; only conditional constructs
    count as guards and function frames do not; and `if __name__` is not a
    guard, because it is how a script is run rather than a condition on whether
    a check happens.
    """
    tree = ast.parse(src)
    parent_of: dict[ast.AST, ast.AST] = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent_of[c] = n

    def branch(node: ast.AST, t: ast.Try) -> str:
        def has(body: list) -> bool:
            return any(node is d for st in body for d in ast.walk(st))
        for h in t.handlers:
            if any(node is d for d in ast.walk(h)):
                return "except"
        if t.orelse and has(t.orelse):
            return "else"
        if t.finalbody and has(t.finalbody):
            return "finally"
        return "body"

    uncond, cond = 0, []
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.Expr) or not isinstance(child.value, ast.Call):
                continue
            f = child.value.func
            if getattr(f, "id", None) not in ("check", "ok", "bad", "expect"):
                continue
            guards: list[str] = []
            cur: ast.AST = child
            while cur in parent_of:
                p = parent_of[cur]
                if isinstance(p, ast.If):
                    guards.append("if " + _short(p.test))
                elif isinstance(p, (ast.For, ast.AsyncFor)):
                    guards.append("for " + _short(p.target))
                elif isinstance(p, ast.While):
                    guards.append("while " + _short(p.test))
                elif isinstance(p, ast.Try):
                    guards.append("try:" + branch(cur, p))
                if isinstance(p, ast.Module):
                    break
                cur = p
            guards = [g for g in guards if "__name__" not in g and g != "if True"]
            if guards:
                cond.append(" | ".join(guards))
            else:
                uncond += 1
    return uncond, sorted(cond)


def census() -> tuple[int, int, str]:
    """`(unconditional, guarded, guards digest)` for this file.

    `skip` is absent from the call-name set on purpose, matching the auditor:
    its census counts sites that *assert*, and a skip is the recorded absence of
    an assertion. Counting it would make this file's census disagree with the
    auditor's on the one construct where the two are supposed to agree.
    """
    uncond, guards = _sites_of(Path(__file__).read_text(encoding="utf-8"))
    digest = hashlib.sha256("\n".join(guards).encode("utf-8")).hexdigest()
    return uncond, len(guards), digest


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - the point is to never be silent
        import traceback

        traceback.print_exc()
        print(f"\nDID NOT FINISH - an exception escaped before every section was "
              f"reached: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
