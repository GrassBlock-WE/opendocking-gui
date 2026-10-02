"""Measure how much narrower `odgui --check`'s stage 2 is than the real viewer.

**The defect this measures.**

`odgui --check` exits 5 -- "OpenGL works on this machine, but Qt cannot realise
an OpenGL widget, so the viewer window would open empty" -- on a machine where
the viewer's own ``Viewport`` gets a context, runs ``initializeGL`` to
completion, wraps Qt's context in moderngl and paints five readable frames. The
probe's conclusion and the product's capability are both facts about the same
machine, and they disagree. A diagnostic that reports a falsehood about the thing
it measures is this project's most expensive defect class, so the difference is
measured here rather than argued about.

**Why a matrix and not a second verdict.**

A single "does it work" run cannot tell you *which* construction difference
matters, and guessing is what produced the bug. The two candidates are
construction-time settings on the widget or its parent, so they are varied one at
a time against a fixed common core (same format, same pump, same reporting), each
in its own process, because a process that dies must not take the matrix with it.

**What is held constant.** The surface format is imported from
``opendocking.workbench.app`` rather than restated, the event loop is pumped with
a deadline rather than entered, and the reporting block is byte-identical across
variants. Only the construction differs. A variant that differs in two settings
would be a guess; each row below changes exactly one.

**What is checked rather than held constant.** The minimum size the stand-in
applies is read from the same product module, so no widget here is configured
from a hand-typed number, *and* it is compared against a declared pair. The
comparison is the point: a product constant that moves turns the two variants
that mirror the viewport red, by name, instead of leaving behind a probe that
measures a construction the product no longer has.

**The product is in the matrix, not a stand-in.** Two rows instantiate the real
``opendocking.workbench.app.Viewport`` and the real ``MainWindow``-shaped
arrangement, because a stand-in that gets a context where the product does not is
the very thing being ruled out. The ``Viewport`` rows additionally record
``moderngl`` -- whether ``self._ctx`` was built -- so "the widget got a context"
cannot be confused with "the renderer could use it".

**Output shape.** One JSON object on stdout, human progress on stderr, so
``python scripts/gl_route_probe_width.py > width.json`` leaves a file that
``json.load`` accepts. Exit ``0`` when every declared variant produced a record
*and at least one of them reached a context*, ``2`` when a declared variant
produced no record at all (a matrix that silently shrank is a bug in here), and
``3`` when every declared variant produced a record and **none** of them reached
one. Mirrors ``gl_route_matrix.py`` on purpose, so the two files are read the
same way.

**Why ``0`` is not "the loop finished."** The third state was missing, and its
absence is the same defect class this file exists to catch. A child that dies
inside ``show()`` has already printed ``before-show``, so it counts as *ran*, so
``missing`` is empty, so the old code returned 0 -- while ``working`` and
``product_working`` were both empty and the summary line below printed that the
baseline construction behaved like the product. Sixteen children killed by a
driver fault were a green run. Producing a record and measuring something are
different facts, and the exit code is what tells them apart.

So a matrix that measured nothing is a **counted result, not a pass**: it is
tallied separately and the reason names the environment property, which is the
convention ``binary_source_parity_check.py`` sets by recording a skip as a
second number it never adds to the passes. It is deliberately not a permanent
red either: this file never fails because the *viewer* is broken, and a code
that is red on every machine without a GL adapter is a code people switch off.

**This file is a measurement, not a gate.** It never fails because the viewer is
broken. The gate is ``odgui_launch_check.py``, which asserts that the probe and
the product agree; this file is the evidence that assertion is built on.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

#: Where the development source tree is, so a variant that imports the product
#: imports *this* checkout rather than whatever happens to be installed.
_DEV_TREE = Path(__file__).resolve().parent.parent / "dock-py" / "python"

#: How long the child pumps after showing, in seconds, and how long the parent
#: waits for the whole child. Generous: a driver that cannot supply a context
#: blocks inside ``show()`` for seconds, and that wait belongs to Qt.
_PUMP_SECONDS = 10.0
_CHILD_TIMEOUT = 60.0

#: Windows reports a process killed by a fault as an NTSTATUS in the top half of
#: the 32-bit range, so a child's ``returncode`` at or above this floor is a
#: fault and not an exit status Python raised. The distinction matters because
#: the two look identical to ``!= 0`` and mean opposite things here: a child that
#: exited 4 said it finished without a context, and a child killed at this floor
#: never got to say anything at all -- its stderr is empty, so the exit code is
#: the only thing it left behind.
_FAULT_FLOOR = 0xC0000000

#: The two fault codes this file has actually seen here, by name. An unnamed one
#: still prints its hex value, so the table is a convenience and not a filter:
#: a code missing from it is reported, not swallowed.
_FAULT_NAMES = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}


#: One child, one variant, dispatched by name. The construction table below is
#: the entire experiment: everything after ``build()`` is identical, so any
#: difference in the answer is attributable to construction.
_CHILD = r'''
import json, sys, time

from PyQt6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets

variant = sys.argv[1]
started = time.perf_counter()
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

from opendocking.workbench.app import (
    MSAA_SAMPLES,
    VIEWPORT_MIN_H,
    VIEWPORT_MIN_W,
    Viewport,
)


#: The minimum size this file claims the product's viewport applies, declared
#: here rather than imported, on purpose.
#:
#: `MSAA_SAMPLES` above is imported because the *value* is the variable under
#: test. This pair is a different kind of fact. The experiment isolates the
#: **call**: every variant is resized to 1280x820, and 240 and 640 are both far
#: below that, so the numbers do not decide any answer. What has to stay honest
#: is the claim -- that the stand-in is built the way the product is built.
#:
#: So the value the stand-in applies is still read from the product (nothing
#: here configures a widget from a hand-typed number) and this declared pair is
#: **checked** against it, which is what makes a change to the product's floor
#: red instead of leaving behind a probe that measures a viewport the product no
#: longer has. It is a tripwire, not a mirror: it configures nothing, and the
#: only way to satisfy it is to re-read the two variants against the product.
#:
#: **Do not "fix" a red here by deleting the comparison and importing instead.**
#: That makes the two values provably equal and the guard unfailable, and this
#: file's whole argument is that a guard which cannot go red is not a guard.
#:
#: The height moved from 480 to 240 as well as the width, and for a reason that is
#: in the tree rather than in taste: with the floor muted the window's minimum
#: height is 134 (view 0 + dock 81 + menu bar 33 + status bar 20), so 480 was
#: making a 533 px window that did not fit a 408 px or 326 px screen. The
#: measured fit threshold is a floor of 273; 240 sits 33 px below it, one
#: menu-bar line being the largest chrome a font change could grow.
_DECLARED_STANDIN_MINIMUM = (240, 240)


def _apply_product_minimum(widget):
    """The one construction-time call the product makes and the stand-in does not.

    The pair comes from the product, so the stand-in cannot drift away from the
    viewport it claims to mirror; the declared pair above is what says this file
    has been reconciled against it. A mismatch raises before the widget is ever
    shown, which the parent records as a harness failure and reports by name --
    so a product change is one line of output rather than a silent difference in
    what the two constructions are.
    """
    applied = (VIEWPORT_MIN_W, VIEWPORT_MIN_H)
    if applied != _DECLARED_STANDIN_MINIMUM:
        raise AssertionError(
            "this file declares that its stand-in mirrors the product's "
            f"setMinimumSize{_DECLARED_STANDIN_MINIMUM}, but this checkout's "
            f"opendocking.workbench.app now says (VIEWPORT_MIN_W, "
            f"VIEWPORT_MIN_H) = {applied}. The stand-in would measure a "
            "construction the product no longer has. Re-read bare_minsize_only "
            "and bare_both against the product, update "
            "_DECLARED_STANDIN_MINIMUM and their descriptions, and re-run."
        )
    widget.setMinimumSize(*applied)


def _format():
    # The product's own format, imported rather than restated, so no variant
    # can accidentally ask for a different surface than the viewer does.
    fmt = QtGui.QSurfaceFormat()
    fmt.setSamples(MSAA_SAMPLES)
    fmt.setDepthBufferSize(24)
    return fmt


def _emit(stage):
    """A progress marker, so a child that dies can say *where* it died.

    This is the difference between "the widget could not get a context" and
    "the widget got a context and the probe's own function-table call killed
    the process" -- two claims that need opposite remedies, and which are
    indistinguishable if the only marker is the one printed before show().
    """
    sys.stdout.write(json.dumps({"variant": variant, "stage": stage}) + "\n")
    sys.stdout.flush()


class _Bare(QtOpenGLWidgets.QOpenGLWidget):
    """A stand-in for the product's widget, identical in shape to the probe's."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialised = False
        self.functions = False
        self.version = None
        self.setFormat(_format())

    def initializeGL(self):
        _emit("initializeGL-entered")
        ctx = QtGui.QOpenGLContext.currentContext()
        if ctx is not None:
            f = ctx.format()
            self.version = "%d.%d" % (f.majorVersion(), f.minorVersion())
            _emit("initializeGL-context")
        self.functions = bool(self.initializeOpenGLFunctions())
        _emit("initializeGL-after-functions")
        self.initialised = self.functions and self.version is not None

    def paintGL(self):
        pass


class _NoFunctions(_Bare):
    """The stand-in without the Qt function-table call.

    `initializeOpenGLFunctions` is the one thing the launcher's probe requires
    and the product never performs -- the product wraps Qt's context in
    moderngl instead. This variant exists to find out whether that call is what
    ends the process, and it is the control for the marker sequence above.
    """

    def initializeGL(self):
        _emit("initializeGL-entered")
        ctx = QtGui.QOpenGLContext.currentContext()
        if ctx is not None:
            f = ctx.format()
            self.version = "%d.%d" % (f.majorVersion(), f.minorVersion())
            _emit("initializeGL-context")
        self.functions = None
        _emit("initializeGL-after-functions")
        self.initialised = self.version is not None


class _NoFormat(_Bare):
    """The stand-in with its one `setFormat` call removed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(QtGui.QSurfaceFormat())


def _build_current_probe():
    # Byte-for-byte the construction in launcher._PROBE_CHILD: a bare widget
    # added to a layout on a holder carrying WA_DontShowOnScreen.
    holder = QtWidgets.QWidget()
    holder.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget = _Bare()
    QtWidgets.QVBoxLayout(holder).addWidget(widget)
    holder.resize(320, 240)
    return holder, widget


def _build_current_probe_no_dontshow():
    holder, widget = _build_current_probe()
    holder.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, False)
    return holder, widget


def _build_current_probe_parented():
    holder = QtWidgets.QWidget()
    holder.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget = _Bare(holder)
    holder.resize(320, 240)
    return holder, widget


def _build_current_probe_large():
    holder, widget = _build_current_probe()
    holder.resize(1280, 820)
    return holder, widget


def _build_bare_real_show():
    # No holder, no attribute: the widget is the window.
    widget = _Bare()
    return widget, widget


def _build_viewport_window():
    # The product, in the shape the app puts it in: a real window, resized like
    # MainWindow, with the real Viewport inside it.
    holder = QtWidgets.QWidget()
    holder.setWindowTitle("width probe")
    holder.resize(1280, 820)
    widget = Viewport()
    QtWidgets.QVBoxLayout(holder).addWidget(widget)
    return holder, widget


def _build_viewport_bare():
    widget = Viewport()
    widget.resize(1280, 820)
    return widget, widget


def _without(method_name):
    """Build the real Viewport with exactly one __init__ call removed.

    A copy of Viewport would be a second implementation that could drift; a
    monkeypatch of the one call under test cannot. Each helper below therefore
    differs from `viewport_bare` by precisely one statement.
    """
    original = getattr(Viewport, method_name)

    def build():
        def muted(self, *args, **kwargs):
            return None

        setattr(Viewport, method_name, muted)
        try:
            widget = Viewport()
        finally:
            setattr(Viewport, method_name, original)
        widget.resize(1280, 820)
        return widget, widget

    return build


def _build_bare_focus_only():
    widget = _Bare()
    widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    widget.resize(1280, 820)
    return widget, widget


def _build_bare_minsize_only():
    widget = _Bare()
    _apply_product_minimum(widget)
    widget.resize(1280, 820)
    return widget, widget


def _build_bare_both():
    widget = _Bare()
    _apply_product_minimum(widget)
    widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    widget.resize(1280, 820)
    return widget, widget


def _build_bare_no_format():
    # The control: identical to the stand-in except that it never calls
    # setFormat, so Qt's default surface is requested instead of MSAA 4 with a
    # 24-bit depth buffer. Still overrides initializeGL, so "no context" and
    # "context but not asked for in the product's format" stay distinguishable.
    widget = _NoFormat()
    widget.resize(1280, 820)
    return widget, widget


def _build_bare_no_functions():
    widget = _NoFunctions()
    widget.resize(1280, 820)
    return widget, widget


def _build_bare_no_functions_holder():
    # The same control, but keeping the launcher's holder and
    # WA_DontShowOnScreen, so "the function table is the variable" is not
    # confounded with "the holder is also different here".
    holder = QtWidgets.QWidget()
    holder.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget = _NoFunctions()
    QtWidgets.QVBoxLayout(holder).addWidget(widget)
    holder.resize(320, 240)
    return holder, widget


VARIANTS = {
    "probe_current": _build_current_probe,
    "probe_no_dontshow": _build_current_probe_no_dontshow,
    "probe_parented": _build_current_probe_parented,
    "probe_1280x820": _build_current_probe_large,
    "bare_real_show": _build_bare_real_show,
    "viewport_in_window": _build_viewport_window,
    "viewport_bare": _build_viewport_bare,
    "bare_focus_only": _build_bare_focus_only,
    "bare_minsize_only": _build_bare_minsize_only,
    "bare_both": _build_bare_both,
    "bare_no_format": _build_bare_no_format,
    "viewport_no_focus": _without("setFocusPolicy"),
    "viewport_no_minsize": _without("setMinimumSize"),
    "viewport_no_setformat": _without("setFormat"),
    "bare_no_functions": _build_bare_no_functions,
    "bare_no_functions_holder": _build_bare_no_functions_holder,
}

# The product is instrumented, never altered: its own initializeGL is wrapped so
# the same markers appear, and the original is restored immediately. Without
# this the marker sequence would exist only for the stand-in, and the two could
# not be compared.
_original_initialize_gl = Viewport.initializeGL


def _instrumented_initialize_gl(self):
    _emit("initializeGL-entered")
    _original_initialize_gl(self)
    _emit("initializeGL-after-product")


Viewport.initializeGL = _instrumented_initialize_gl

holder, widget = VARIANTS[variant]()

found = {
    "variant": variant,
    "platform": QtGui.QGuiApplication.platformName(),
    "is_product": isinstance(widget, Viewport),
    "waited": time.perf_counter() - started,
}
try:
    found["screens"] = len(QtGui.QGuiApplication.screens())
except Exception:
    found["screens"] = "?"

# Printed before show(): on a machine that cannot supply a context, show() is
# where the process ends, and "?" for every field is a worse answer than the
# facts that could be established.
found["stage"] = "before-show"
sys.stdout.write(json.dumps(found) + "\n")
sys.stdout.flush()

holder.show()

# A widget that is its own window has to be shown on itself, and holder.show()
# already did that; the loop is shared so no variant gets a different event
# loop from any other.
deadline = time.perf_counter() + float(sys.argv[2])
while time.perf_counter() < deadline:
    app.processEvents()
    if getattr(widget, "initialised", False) or widget.isValid() and getattr(
        widget, "_ctx", None
    ) is not None:
        # Keep pumping a little past the first success so paintGL runs at least
        # once, which is what "the viewer really would open" means.
        break
    time.sleep(0.005)

frames = 0
if getattr(widget, "_ctx", None) is not None or getattr(widget, "initialised", False):
    end = time.perf_counter() + 0.4
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)
        frames += 1

found["show_seconds"] = time.perf_counter() - started
found["initialised"] = bool(getattr(widget, "initialised", False))
found["gl"] = (
    widget.version
    if getattr(widget, "version", None)
    else (lambda f: "%d.%d" % (f.majorVersion(), f.minorVersion()))(
        widget.format()
    )
    if widget.isValid()
    else None
)
found["functions"] = bool(getattr(widget, "functions", False))
found["widget_valid"] = bool(widget.isValid())
found["moderngl"] = getattr(widget, "_ctx", None) is not None
found["pump_iterations"] = frames
found["waited"] = time.perf_counter() - started
found["stage"] = "done"
sys.stdout.write(json.dumps(found) + "\n")
sys.stdout.flush()
sys.exit(0 if found["initialised"] or found["moderngl"] else 4)
'''


#: The declared variants, in the order they are run. ``changes`` names the one
#: thing each differs from ``probe_current`` by, which is the claim the matrix
#: is able to falsify.
VARIANTS: tuple[tuple[str, str, str], ...] = (
    (
        "probe_current",
        "baseline",
        "exactly the construction in launcher._PROBE_CHILD: holder, "
        "WA_DontShowOnScreen, QVBoxLayout, 320x240, widget built unparented",
    ),
    (
        "probe_no_dontshow",
        "WA_DontShowOnScreen",
        "the only change is clearing WA_DontShowOnScreen on the holder",
    ),
    (
        "probe_parented",
        "parent at construction",
        "the only change is passing the holder to the widget's constructor "
        "instead of adding it to a layout afterwards",
    ),
    (
        "probe_1280x820",
        "holder size",
        "the only change is resizing the holder to MainWindow's 1280x820",
    ),
    (
        "bare_real_show",
        "holder removed",
        "the widget is its own top-level window, shown for real, with no "
        "holder and no layout",
    ),
    (
        "viewport_in_window",
        "the product",
        "the real app.Viewport in a real shown window, the arrangement the "
        "workbench itself uses",
    ),
    (
        "viewport_bare",
        "the product, top level",
        "the real app.Viewport as a top-level window, to separate 'is a parent "
        "window needed' from 'is it the class that matters'",
    ),
    (
        "bare_focus_only",
        "focus policy",
        "the stand-in plus only setFocusPolicy(StrongFocus), the one "
        "construction-time call the product makes and the stand-in does not",
    ),
    (
        "bare_minsize_only",
        "minimum size",
        "the stand-in plus only the product's own minimum size, read from "
        "opendocking.workbench.app rather than restated here and checked "
        "against _DECLARED_STANDIN_MINIMUM so a change to the product's floor "
        "is red; the other construction-time call the product makes and the "
        "stand-in does not",
    ),
    (
        "bare_both",
        "both calls",
        "the stand-in plus both calls, to confirm the two are not each "
        "sufficient on their own",
    ),
    (
        "bare_no_format",
        "surface format (control)",
        "the stand-in with setFormat removed, so Qt's default surface is "
        "requested: the control that says the format is not the variable",
    ),
    (
        "viewport_no_focus",
        "product minus setFocusPolicy",
        "the real Viewport with setFocusPolicy muted -- one statement removed "
        "from the product's own __init__",
    ),
    (
        "viewport_no_minsize",
        "product minus setMinimumSize",
        "the real Viewport with setMinimumSize muted",
    ),
    (
        "viewport_no_setformat",
        "product minus setFormat",
        "the real Viewport with setFormat muted, the last of the three "
        "construction-time calls that differ",
    ),
    (
        "bare_no_functions",
        "the Qt function-table call",
        "the stand-in shown as a real window with initializeOpenGLFunctions "
        "removed -- the one thing the launcher's probe requires and the "
        "product never does",
    ),
    (
        "bare_no_functions_holder",
        "function table, launcher's holder",
        "the same, but keeping the holder and WA_DontShowOnScreen, so the "
        "function-table result is not confounded with a different holder",
    ),
)


def _first_failure_line(stderr: str) -> str:
    """The child's own statement of what went wrong.

    Taking `stderr.splitlines()[-1]` -- which is what this file did, and is kept
    in `stderr` -- reports the same thing for every failure the file could ever
    have: the product installs an excepthook whose last line is its own
    "ending with exit code 70" notice, so a child that stopped because a product
    constant moved and a child that stopped because a driver crashed are
    indistinguishable from the last line alone. That is the difference between
    "this variant produced no record" and "this variant produced no record
    *because* the stand-in no longer matches the product".

    The exception is stated *first*, in both shapes this produces: inside the
    excepthook's one-line report when Qt got there first, and as the unindented
    final line of a plain traceback when nothing did. So the first non-indented,
    non-blank line that is not the traceback header is the reason either way,
    and the rule does not have to know what the excepthook's wording is.
    """
    for line in stderr.splitlines():
        text = line.strip()
        if not text or line[:1].isspace():
            continue
        if text.startswith("Traceback (most recent call last):"):
            continue
        return text
    return stderr.splitlines()[-1]


def _describe_exit(code: object) -> str:
    """A child's `returncode` in words, so a fault is not read as an exit.

    A fault is a statement about the machine -- the driver, the platform plugin,
    the adapter -- and printing it as a bare integer is what let eight (then
    sixteen) identical fault codes sit in a green run unnoticed. `exit 4` and
    `0xc0000005` are the whole difference between "the child finished and said
    it had no context" and "the child was killed where it stood".
    """
    if not isinstance(code, int) or code < _FAULT_FLOOR:
        return f"exit {code}"
    name = _FAULT_NAMES.get(code, "unnamed Windows fault")
    return f"Windows fault {code:#010x} ({name})"


def _measurement_verdict(records: list[dict]) -> tuple[str, list[str]]:
    """Whether this run measured anything, and if not, what stopped it.

    Returns ``("measured", [])`` or ``("nothing_measured", [reasons])``. A
    matrix in which no arm obtained a context measured nothing -- whether the
    children were killed inside `show()` or finished cleanly without one -- and
    that is a different fact from the one a matrix that did measure reports, so
    it gets its own verdict and its own exit code rather than sharing either.

    The reasons name the environment property rather than only counting the
    arms, because "this machine cannot supply a GL context" and "this checkout
    is broken" call for opposite responses and a reader cannot tell them apart
    from a count. Two facts are read off the records rather than guessed:
    whether the fault was a Windows fault at all, and whether any arm ever
    emitted `initializeGL-entered` -- which separates a process faulted while
    *showing* the widget from one faulted inside `initializeGL`, and so says
    whether the GL bindings are implicated at all.
    """
    working = [
        r["variant"] for r in records
        if r.get("outcome") in ("moderngl", "widget")
    ]
    if working:
        return "measured", []

    ran = [r for r in records if r.get("outcome") != "harness"]
    faults: dict[int, int] = {}
    for record in ran:
        code = record.get("child_exit")
        if isinstance(code, int) and code >= _FAULT_FLOOR:
            faults[code] = faults.get(code, 0) + 1

    reasons: list[str] = []
    if faults:
        for code, count in sorted(faults.items()):
            reasons.append(
                f"{count} of {len(ran)} arm(s) that reached the experiment were "
                f"killed by {_describe_exit(code)}, with no message on stderr"
            )
    elif ran:
        reasons.append(
            f"{len(ran)} arm(s) reached the experiment and finished cleanly "
            "with no context (outcome no_context), so the platform gave a "
            "window and no GL"
        )
    else:
        reasons.append("no arm reached the experiment at all; see missing below")

    stages = {s for r in records for s in r.get("stages") or ()}
    if ran and "initializeGL-entered" not in stages:
        reasons.append(
            "no arm emitted initializeGL-entered, so nothing was faulted inside "
            "the GL bindings: the process ended while the widget was being shown"
        )
    return "nothing_measured", reasons


def _run_variant(name: str) -> dict:
    """One variant, one process, one record -- whatever it managed to print."""
    # The environment is inherited and *extended*, never replaced. An earlier
    # version of this file passed a hand-built env with an empty PATH and
    # SYSTEMROOT, and every variant then died with exit 1 before it ran a line
    # of Python -- which this file was perfectly willing to report as "the
    # driver killed the child", because exit 1 and 0xC0000409 both look like
    # "died" to a naive reader. So the structural check below exists: a variant
    # that never reached its `before-show` line is a *harness* failure and is
    # named as one, never folded into the findings.
    import os

    real_env = dict(os.environ)
    real_env["PYTHONPATH"] = str(_DEV_TREE)
    real_env["PYTHONIOENCODING"] = "utf-8"
    real_env["PYTHONDONTWRITEBYTECODE"] = "1"
    # The two variables a caller may have set to steer Qt. The matrix reports
    # one environment, so they are removed rather than inherited silently --
    # an unconfigured default is a fact this file is willing to state.
    real_env.pop("QT_OPENGL", None)
    real_env.pop("QT_QPA_PLATFORM", None)
    record: dict = {"variant": name}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, name, str(_PUMP_SECONDS)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_CHILD_TIMEOUT,
            env=real_env,
        )
    except subprocess.TimeoutExpired:
        record["outcome"] = "timed_out"
        return record
    merged: dict = {}
    stages: list[str] = []
    for line in (proc.stdout or "").splitlines():
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            merged.update(loaded)
            if loaded.get("stage"):
                stages.append(loaded["stage"])
    merged.pop("stage", None)
    record.update(merged)
    record["child_exit"] = proc.returncode
    # The full marker sequence, so a child that died says how far it got. This
    # is the field that separates "no context" from "a context, then the
    # probe's own call", which is the distinction the whole matrix exists for.
    record["stages"] = stages
    # The structural check. `before-show` is printed before anything that can
    # end the process, so its absence means the child never got as far as the
    # experiment -- an import failure, a bad interpreter, a broken harness.
    # Reporting that as "died" is how a matrix of seven crashes can be mistaken
    # for a measurement, so it is refused here.
    record["reached_experiment"] = "before-show" in stages
    if not record["reached_experiment"]:
        record["outcome"] = "harness"
    elif merged.get("moderngl"):
        record["outcome"] = "moderngl"
    elif merged.get("initialised"):
        record["outcome"] = "widget"
    elif proc.returncode != 0:
        record["outcome"] = "died"
    else:
        record["outcome"] = "no_context"
    stderr = (proc.stderr or "").strip()
    if stderr:
        record["stderr"] = stderr.splitlines()[-1]
        record["stderr_reason"] = _first_failure_line(stderr)
    return record


def main() -> int:
    declared = [name for name, _, _ in VARIANTS]
    records: list[dict] = []
    print("probe width matrix -- one process per construction")
    for name, changes, description in VARIANTS:
        print(f"  {name:<20} ({changes})", flush=True)
        print(f"  {'':<20} {description}", file=sys.stderr, flush=True)
        records.append(_run_variant(name))

    got = [r["variant"] for r in records if r.get("outcome") != "harness"]
    missing = [n for n in declared if n not in got]
    verdict, why = _measurement_verdict(records)
    working = [r["variant"] for r in records if r.get("outcome") in ("moderngl", "widget")]
    product = [r for r in records if r.get("is_product")]
    product_ok = [r["variant"] for r in product if r.get("outcome") in ("moderngl", "widget")]
    baseline = next((r for r in records if r["variant"] == "probe_current"), {})
    baseline_ran = baseline.get("outcome") != "harness"

    summary = {
        "declared": len(declared),
        "attempted": len(got),
        # The two numbers a reader needs, kept apart on purpose: `measured` is
        # the count of arms that reached a context and `unmeasured` is every
        # other arm, and the second is never added to the first. A run in which
        # all sixteen arms produced a record and none of them measured is 0 + 16,
        # not 16.
        "measured": len(working),
        "unmeasured": len(records) - len(working),
        "verdict": verdict,
        "unmeasured_reason": why,
        "missing": missing,
        "baseline_outcome": baseline.get("outcome"),
        "baseline_ran": baseline_ran,
        "working": working,
        "product_working": product_ok,
        "divergence": bool(
            baseline_ran
            and baseline.get("outcome") not in ("moderngl", "widget")
            and product_ok
        ),
    }
    payload = {
        "pump_seconds": _PUMP_SECONDS,
        "declared": [
            {"variant": n, "changes": c, "description": d} for n, c, d in VARIANTS
        ],
        "variants": records,
        "summary": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    print()
    print("outcomes")
    for record in records:
        extra = " ".join(
            f"{k}={record[k]}"
            for k in ("gl", "moderngl", "child_exit")
            if k in record
        )
        note = f"  <- {_describe_exit(record['child_exit'])}" if (
            "child_exit" in record
            and not record.get("outcome") in ("moderngl", "widget")
        ) else ""
        print(
            f"  {record['variant']:<20} {record.get('outcome', '?'):<10} "
            f"{extra}{note}"
        )
    # The tally, in the shape `binary_source_parity_check.py` prints: the
    # unmeasured count is on the line and is not folded into the measured one.
    print()
    print(
        f"--- {len(working)} measured, {len(records) - len(working)} unmeasured, "
        f"of {len(records)} arms ({len(declared)} declared)"
    )
    print()
    if summary["divergence"]:
        print(
            "  DIVERGENT: the baseline construction failed while the product's own "
            "widget worked, so the probe is narrower than the product."
        )
    elif not baseline_ran:
        print(
            "  NO CONCLUSION: the baseline variant never reached the experiment, "
            "so this run is a harness failure and not a measurement."
        )
    elif verdict == "nothing_measured":
        # The claim this file exists to falsify is a claim about agreement, and
        # there is no agreement to report when not one arm obtained a context.
        # Saying the baseline "behaved like the product" here would have been the
        # defect, not the finding: a hard fault and a working construction are
        # not the same observation, and only the second one is evidence.
        print(
            "  NO CONCLUSION: no arm reached a context, so this run measured "
            "nothing and is not evidence either way about the width."
        )
        for line in why:
            print(f"    {line}")
        print(
            "  This is a counted result, not a pass and not a failure of the "
            "viewer: on a machine that cannot supply a context the matrix has "
            "no answer to give, and reporting that as agreement is the error."
        )
    else:
        print("  the baseline construction behaved like the product")
    if missing:
        print(f"  {len(missing)} declared variant(s) produced no record: {', '.join(missing)}")
        # Naming the absence is not the same as naming the cause. A variant that
        # never reached `before-show` failed in the harness rather than in the
        # experiment, and the cause is on the child's stderr -- see
        # `_first_failure_line` for why it is not simply the last line. For a
        # product constant this file has not been reconciled against, that cause
        # is the whole finding; without this the reader is told which variants are
        # missing and nothing about why.
        for record in records:
            if record["variant"] in missing:
                cause = record.get("stderr_reason") or record.get("stderr")
                if cause:
                    print(f"    {record['variant']}: {cause}")
        # A harness failure is a bug in here, so it is reported ahead of the
        # verdict: "no arm measured" is the honest description of a run in which
        # nothing ran, and it would bury the more specific statement.
        return 2
    if verdict == "nothing_measured":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
