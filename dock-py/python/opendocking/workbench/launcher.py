"""``odgui`` -- the standalone entry point for the 3-D workbench.

Separate from the ``odcli`` command-line interface on purpose. The package ships
exactly two console scripts, ``odgui`` for the viewer and ``odcli`` for the
full CLI, so a user who reaches for one by name gets exactly that and nothing
else. Neither depends on the other being present.

Nothing here imports Qt or moderngl at module scope. Importing this module has
to stay cheap and must work on a machine with no display stack, so that
``odgui --help`` can explain how to install the GUI extras rather than dying
with a traceback.

``--check`` answers in two stages, because "no OpenGL" is two different faults
and a user needs to be told which one they have.

**Stage 1** asks for a context the cheap way, in this process:
``QOffscreenSurface`` plus a bare ``QOpenGLContext``, with the viewport's own
surface format. It needs no child process and costs about a second.

**Stage 2** runs only if stage 1 succeeded, and is the probe that judges the
product: it builds the **real** ``opendocking.workbench.app.Viewport``, the class
the workbench puts on screen, inside a ``WA_DontShowOnScreen`` holder, in a child
process. The child stays, and has to stay, because on a machine that cannot
realise the widget the driver ends the *process* with ``0xC0000409`` rather than
raising -- a ``try`` in this process cannot see that.

"Ready" is the product's own condition and not a stand-in's: ``Viewport`` has
built its moderngl context (``self._ctx is not None``). That is the point at
which the app can draw, so a verdict of 5 now means the viewer really would open
with nothing in it.

**This stage used to be narrower than the thing it judged, and said so wrongly.**
It built a bare ``QOpenGLWidget`` subclass and required that subclass's
``initializeOpenGLFunctions()`` to return true. Measured on the machine that
prompted the original two-stage split, in 16 isolated runs
(``scripts/gl_route_probe_width.py``): the bare widget *did* get a context, with
GL 4.6, every time -- and then the process ended with ``0xC0000409`` on the very
next statement, the function-table call. With that one call removed and nothing
else changed, the same widget survived and reported a usable context. The
product never calls ``initializeOpenGLFunctions`` at all; it wraps Qt's context
in moderngl. So the old stage 2 measured a capability the viewer does not need
and does not have, and reported the difference as "your viewer would open empty"
on a machine where the viewer renders. Exit 5 was earned by the probe's own
requirement, not by the product's behaviour.

The consequence for this file: **stage 2 is never skipped to make the answer
look better.** It is skipped only when stage 1 has already failed, where it
could not change the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]

#: What ``odgui`` needs beyond the engine itself. Kept in one place so the
#: error message and the packaging metadata cannot drift apart.
_GUI_REQUIREMENTS = "PyQt6>=6.4 moderngl>=5.8 numpy-stl>=2.0"

#: How long the probe pumps the event loop after showing the widget, in seconds.
#:
#: Small on purpose, because it is not where the time goes. Qt blocks *inside*
#: `show()` while it tries to create the context and then gives up on its own
#: schedule: measured at 4.15-4.6 s on a machine whose driver cannot supply
#: one. That wait belongs to Qt and this code cannot shorten or skip it, which
#: is why a negative `--check` costs seconds rather than milliseconds. What is
#: left for the loop is the case where `initializeGL` arrives *after* `show()`
#: returns, on a round or two of `processEvents`. The healthy-case cost has not
#: been measured -- no machine with a working context was available -- so this
#: is sized as a margin, not as an observation.
_GL_PUMP_SECONDS = 0.5

_BANNER = "Open Docking Workbench"

#: ``--check`` exit codes. Written down because a diagnostic that always exits
#: 0 cannot be scripted against, and because the next reader will assume 0 is
#: the only sensible answer without checking.
#:
#: ``0``  the GUI stack imported *and* Qt gave us an OpenGL context.
#: ``3``  the GUI stack is not importable. Deliberately the same code ``odgui``
#:        and ``odcli workbench`` already return for that, so ``--check`` is a
#:        faithful preflight: its exit code is the code the launch would give,
#:        and a script can test it and then run the viewer without re-deriving
#:        what the number means.
#: ``4``  the stack imported, but **no OpenGL context at all** -- even a bare
#:        one on an offscreen surface. A separate code because it is a different
#:        problem with a different fix -- a driver, a GPU, or a remote display
#:        rather than a missing package -- and a script that installs PyQt6
#:        should not treat it as a success.
#: ``5``  the stack imported and OpenGL works, but the **product's own
#:        viewport** (`app.Viewport`) did not come up, so the viewer window
#:        would still open with nothing rendered in it. Its own code because
#:        collapsing it into 4 is what made this undiagnosable: the remedy for
#:        4 ("get a machine with a GPU") is not the remedy for 5 (the driver is
#:        fine; the viewport's own path is not), and telling someone with code 4
#:        to go and buy a GPU when their OpenGL is working is the kind of
#:        confidently wrong advice that costs a day.
#:
#:        The wording of that code was itself wrong for a long time, and is the
#:        reason this paragraph is long. It used to mean "Qt cannot realise an
#:        OpenGL *widget*", which was a claim about a bare stand-in widget and
#:        not about the product -- and on this machine the stand-in died while
#:        the product rendered. It now means what the probe actually tested, so
#:        the sentence printed for it can be true.
_EXIT_OK = 0
_EXIT_NO_GUI_STACK = 3
_EXIT_NO_CONTEXT = 4
_EXIT_NO_WIDGET = 5

#: ``--check`` verdicts, the ``--json`` spelling of the codes above. One string
#: per outcome so a script branches on a name rather than on a number that a
#: future exit code could renumber.
_VERDICT_OK = "ok"
_VERDICT_NO_GUI_STACK = "no-gui-stack"
_VERDICT_NO_CONTEXT = "no-context"
_VERDICT_NO_WIDGET = "no-widget"


# --------------------------------------------------------------------------
# Which copy of the workbench answered.
#
# Three copies of this package can be loaded by the same command on the same
# machine: a development tree, the one-directional mirror beside it, and the
# installed wheel under `site-packages`. Which one wins is decided by
# `sys.path`, so a caller **cannot** infer it from *which* launcher it invoked.
# `odgui` is a console script, and a `PYTHONPATH` pointing at a source tree wins
# inside it. Measured on the machine this was written on, same command and same
# executable, differing only in the environment: with `PYTHONPATH` set to
# `dock-py/python` the launcher resolved to
# `dock-py/python/opendocking/workbench/launcher.py`, and with it cleared the
# identical command resolved to
# `site-packages/opendocking/workbench/launcher.py`.
#
# So the paths below are read from the modules this process actually imported.
# A label derived from which branch ran would have called the second one "the
# installed wheel" while measuring the first -- a report disagreeing with the
# process, which is the exact failure the `--json` contract exists to prevent,
# and the mistake `core_check.py` made and corrected the same day.
#
# The digest is here so that "the copy you are editing" becomes a comparison
# rather than an opinion. Two paths can be different strings and the same file,
# and `scripts/_gui_check.py` turns these fields into a named divergence, which
# is the only mechanical way to tell a stale install from a product defect.
# --------------------------------------------------------------------------
def _path_facts(path, module_name: str, measured_in: str) -> dict:
    """One module's resolved file, measured.

    `path` is whatever the running process itself knows -- this module's
    `__file__`, or a module's entry in `sys.modules` -- and is never assembled
    from an assumed directory layout. Nothing here imports anything: on the
    machine where PyQt6 is absent, an import attempted only to attach a label
    would fail and take the check's one real answer down with it.
    """
    import hashlib

    facts: dict = {"module": module_name, "measured_in": measured_in}
    if not path:
        facts["path"] = None
        facts["problem"] = (
            "this process never loaded the module, so there is no path to name"
        )
        return facts
    facts["path"] = str(Path(path).resolve())
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        facts["problem"] = f"the file could not be read ({exc})"
        return facts
    facts["bytes"] = len(data)
    facts["sha256"] = hashlib.sha256(data).hexdigest()
    return facts


def _provenance(subject: dict | None = None) -> dict:
    """The `provenance` block of a `--check` payload.

    `subject` is what the **child** measured about `opendocking.workbench.app`,
    the class stage 2 builds. It has to come from the child: that is the process
    that imported the class, and the parent never does, so a path assembled in
    the parent would be a guess wearing the clothes of a measurement. Stage 1
    does import `app`, but for `MSAA_SAMPLES` and on a different machine state,
    so its reading is a fallback and is labelled as one -- which is the whole
    point: "measured here" and "inferred there" must not look alike.
    """
    prov: dict = {
        "launcher": _path_facts(
            __file__, "opendocking.workbench.launcher", "this process"
        )
    }
    package = sys.modules.get("opendocking")
    init = getattr(package, "__file__", None)
    prov["package"] = str(Path(init).resolve().parent) if init else None

    if subject:
        prov["app"] = dict(subject)
    else:
        prov["app"] = _path_facts(
            getattr(
                sys.modules.get("opendocking.workbench.app"),
                "__file__",
                None,
            ),
            "opendocking.workbench.app",
            "this process (stage 1 imports app for MSAA_SAMPLES; the stage-2 "
            "child did not answer)",
        )
    return prov


def _describe_provenance(payload: dict) -> str:
    """The copy lines of the prose report, built from the same dict as `--json`.

    One line per file with the digest cut to 12 characters: the full one belongs
    in the JSON, where a script can compare it, and a reader at a terminal wants
    to know *which* file rather than to re-type 64 hex digits. Both spellings
    are formatted from `payload["provenance"]`, so the prose cannot name a
    different copy than the object does.
    """
    prov = payload.get("provenance") or {}
    rows: list[tuple[str, str]] = []
    for key, label in (("launcher", "launcher"), ("app", "viewport ")):
        facts = prov.get(key) or {}
        if not facts.get("path"):
            rows.append((label, facts.get("problem") or "the path is unknown"))
            continue
        rows.append(
            (
                label,
                f"{facts['path']}  ({facts.get('bytes', '?')} bytes, "
                f"sha256 {str(facts.get('sha256') or '?')[:12]})",
            )
        )
    if prov.get("package"):
        rows.append(("package ", str(prov["package"])))
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  copy   {label.ljust(width)}  {text}" for label, text in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odgui",
        description="Launch the Open Docking 3-D workbench.",
        epilog=(
            "With no arguments the window opens empty: load a receptor and a "
            "ligand from the File menu, or pass them below. Docked poses can "
            "be passed with --poses and are then browsable by energy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r", "--receptor", type=Path, help="receptor .pdbqt to open on start-up"
    )
    parser.add_argument(
        "-l", "--ligand", type=Path, help="ligand .pdbqt to open on start-up"
    )
    parser.add_argument(
        "-p", "--poses", type=Path, help="docked poses .pdbqt to browse on start-up"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "report whether the viewer can start, then exit without showing a "
            "window. Exits 0 if it can, 3 if PyQt6 or moderngl cannot be "
            "imported, 4 if they import but the machine cannot give Qt any "
            "OpenGL context at all, and 5 if OpenGL works but Qt cannot realise "
            "an OpenGL widget -- 4 and 5 are both graphics problems, but they "
            "are not the same one and only one of them is fixed by changing "
            "machine or driver"
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help=(
            "with --check, print the verdict as one JSON object on stdout "
            "instead of prose. The exit code is unchanged, so this is a way to "
            "read the *facts* (which GL version, whether the function table is "
            "usable) rather than a fourth answer. The object carries a 'raw' "
            "and a 'widget' object, so a script can assert that the cheap stage "
            "succeeded rather than only that nothing went red"
        ),
    )
    return parser


def _root_cause(exc: BaseException) -> BaseException:
    """The original ``ImportError``, seen through `_require_gui`'s re-raise.

    `_require_gui` catches the engine's own error and raises a friendlier one
    with `from exc`, which is right for a library caller reading a message. It
    is wrong for *this* module: the friendly text says "needs PyQt6 and
    moderngl" and nothing about which of them, and the original is the one that
    knows. So the chain is walked to its bottom before the message is built.
    """
    seen: set[int] = set()
    while exc.__cause__ is not None and id(exc) not in seen:
        seen.add(id(exc))
        exc = exc.__cause__
    return exc


def _describe_missing(exc: BaseException) -> str:
    """Turn an ImportError into the one line that actually helps.

    The single user-facing sentence for "the viewer cannot be installed here",
    shared by ``odgui``, ``odgui --check`` and ``odcli workbench`` so the three
    cannot word the same event three ways. It lives here rather than in
    ``launch`` because ``_GUI_REQUIREMENTS` lives here, and a message that
    quotes one module's pin while the install line comes from another is the
    kind of drift nobody notices until a user follows the wrong advice.
    """
    cause = _root_cause(exc)
    name = getattr(cause, "name", None) or str(cause)
    return (
        f"error: the workbench needs PyQt6 and moderngl, and `{name}` could not "
        f"be imported ({cause}).\n"
        f"       install them with:\n"
        f"           pip install {_GUI_REQUIREMENTS}"
    )


def _describe_no_context(found: dict) -> str:
    """The sentence for "it is installed, and the window still will not open".

    A different sentence on purpose. The missing-package message is answered by
    one command and then works; this one is not, and telling a user to
    `pip install` a package they already have is how a two-minute question turns
    into a bug report. So it names what was actually established, what was not,
    and the two things that are worth trying, in that order of likelihood.
    """
    # `show_seconds` is absent when the probe process ended inside `show()`,
    # which is the normal outcome on a machine like this one. Printing "0.0 s"
    # there would be quoting a default in the shape of a measurement: it reads
    # as "show() returned instantly" when in fact the process never came back.
    # Say which of the two happened instead.
    waited = found.get("waited")
    if "show_seconds" in found:
        timing = (
            f"{waited:.1f} s, of which {found['show_seconds']:.1f} s "
            f"was Qt inside show()"
        )
    else:
        timing = (
            f"{waited:.1f} s, ending inside show() -- the probe process did "
            f"not come back, which is what a driver with no context does"
        )
    return (
        "error: PyQt6 and moderngl are installed, but Qt could not create the\n"
        "       OpenGL context the viewport needs, so the window would open\n"
        "       empty. This is a graphics-driver problem, not a missing package.\n"
        f"         Qt platform   {found.get('platform', '?')}\n"
        f"         screens       {found.get('screens', '?')}\n"
        f"         samples asked {found.get('samples', '?')} (granted: unknown -- Qt "
        "only answers that once a context exists)\n"
        f"         waited        {timing}\n"
        "       Worth trying, in order:\n"
        "         1. a machine with a GPU and a current graphics driver -- a\n"
        "            remote desktop or a VM without 3-D acceleration cannot\n"
        "            give Qt a context no matter what is installed;\n"
        "         2. `odgui` on the machine with the screen, if this one is a\n"
        "            container or a headless server.\n"
        "       The docking engine and `odcli` do not need OpenGL and are\n"
        "       unaffected."
    )


def _seconds(value) -> str:
    """A duration as something a person reads, or ``?`` if it is not one.

    The probe dictionaries carry raw `perf_counter` deltas, and interpolating one
    straight into a sentence prints `2.312246200000118 s`. Rounding happens here,
    once, rather than at each of the call sites -- which is also the only way it
    cannot end up applied in some places and forgotten in others.
    """
    if isinstance(value, (int, float)):
        return f"{value:.1f} s"
    return "?"


def _describe_no_widget(payload: dict) -> str:
    """The sentence for "your OpenGL works, and the viewport still will not open".

    Deliberately *not* the sentence above. That one tells a user their machine
    cannot do OpenGL, and the natural response -- get a machine with a GPU -- is
    wrong here, because this machine demonstrably can: stage 1 created a context
    and resolved its function table moments earlier. Sending someone to buy a GPU
    when their driver is fine is the kind of confidently wrong advice that costs
    a day, and it is what a single exit code for "no context" produced here.

    So this leads with the two facts that differ, and only then says what is
    worth trying.

    The remedy list is also narrower than it used to be, and deliberately so. It
    used to lead with "a different Qt rendering path", because the old probe
    failed on a bare stand-in widget and a different widget path was therefore
    the obvious thing to try. That advice was chasing a fault in the probe: the
    stand-in died on its own ``initializeOpenGLFunctions()`` call, which
    ``QT_OPENGL=software`` does nothing about and the product never makes. With
    the product's own ``Viewport`` as the subject, the things still worth trying
    are the ones that can remain the cause.
    """
    raw = payload.get("raw") or {}
    widget = payload.get("widget") or {}
    waited = widget.get("waited")
    if "show_seconds" in widget:
        timing = (
            f"{_seconds(waited)}, of which {_seconds(widget['show_seconds'])} "
            f"was Qt inside show()"
        )
    else:
        timing = (
            f"{_seconds(waited)}, ending inside show() -- the probe process did "
            f"not come back (exit {widget.get('child_exit', '?')})"
        )
    return (
        "error: OpenGL on this machine works, but the workbench's own viewport\n"
        "       did not come up, so the viewer window would open with nothing\n"
        "       rendered in it.\n"
        f"         raw context   GL {raw.get('gl', '?')}, function table "
        f"{'usable' if raw.get('functions') else 'UNUSABLE'} "
        f"({_seconds(raw.get('waited'))})  <- OpenGL is fine\n"
        f"         viewport      GL {widget.get('gl') or 'none'}, "
        f"{widget.get('subject', 'the product viewport')} {timing}\n"
        "       The subject of that stage is opendocking.workbench.app.Viewport,\n"
        "       the class the workbench actually shows, so this is a fact about\n"
        "       the viewer and not about a stand-in. It is not a missing package\n"
        "       and not a machine without 3-D acceleration, so installing PyQt6\n"
        "       or moving to a GPU machine is unlikely to help. Worth trying, in\n"
        "       order:\n"
        "         1. an up-to-date graphics driver, or a session on the machine\n"
        "            itself rather than over remote desktop or a VM -- a virtual\n"
        "            display adapter is the usual cause of a context that works\n"
        "            offscreen and a widget that will not;\n"
        "         2. a different Qt rendering path -- run with\n"
        "            QT_OPENGL=software, which routes Qt's own OpenGL through\n"
        "            Mesa rather than the display driver.\n"
        "       `odgui --check --json` reports both stages as 'raw' and 'widget'.\n"
        "       The docking engine and `odcli` do not need OpenGL and are\n"
        "       unaffected."
    )


def _preflight() -> tuple[object | None, str | None]:
    """Import the GUI stack, reporting the first missing piece.

    Returns ``(launch, None)`` on success or ``(None, message)`` on failure.
    Checked in one place so the window can never open with a half-initialised
    renderer and a blank viewport.

    This used to ``from . import launch`` and treat the import as the test, which
    is a test that cannot fail: `launch` needs no Qt -- it defers the GUI import
    to a call inside itself -- so the import always succeeded, the ``except
    ImportError`` below was unreachable, and ``odgui --check`` reported a
    working GUI stack on machines where it could not possibly start one. It now
    calls `_require_gui`, which is the call `launch` makes, in this process,
    under these conditions. Restoring the old body after the fix changes nothing
    at all, which is the proof that it was dead.
    """
    from . import _require_gui, launch

    try:
        _require_gui()
    except ImportError as exc:
        return None, _describe_missing(exc)
    return launch, None


#: How long stage 1 waits for a context on an offscreen surface, in seconds.
#: Generous, because it is not the time that costs anything: measured at 0.8-1.5 s
#: for a success and 1.2 s for a clean failure on the machine that prompted the
#: two-stage split. The bound exists so a wedged driver cannot hang the command
#: here, where the child process cannot be used to absorb it.
_RAW_CONTEXT_SECONDS = 10.0


def _probe_raw_context() -> tuple[dict, str | None]:
    """Stage 1: ask for a context the cheap way, in this process.

    A ``QOffscreenSurface`` and a bare ``QOpenGLContext``, with the viewport's
    own surface format -- the same ``MSAA_SAMPLES`` and depth the widget probe
    asks for, imported from ``app`` rather than restated, so the two stages
    cannot drift into asking for different things.

    Why this stage exists at all: it is the one question that can be asked
    without a widget, and on the machine that prompted it the two answers
    disagreed completely. ``QOpenGLWidget`` was killed by the driver 40 times in
    40 while this same code, in the same process, obtained a context 40 times in
    40. That difference is the whole diagnosis, and it is invisible to a probe
    that only ever asks the widget question.

    No child process, and that is the point rather than a convenience. Measured
    here: it never dies, so there is nothing to survive.

    Returns ``(found, problem)``; ``problem is None`` means a context was created
    *and* made current *and* its function table resolved.
    """
    import time

    from PyQt6 import QtGui, QtWidgets

    from opendocking.workbench.app import MSAA_SAMPLES

    started = time.perf_counter()
    found: dict = {"stage": "raw"}
    problem: str | None = None
    surface = None
    context = None
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            sys.argv[:1]
        )
        found["platform"] = QtGui.QGuiApplication.platformName()
        try:
            found["screens"] = len(QtGui.QGuiApplication.screens())
        except Exception:
            found["screens"] = "?"

        fmt = QtGui.QSurfaceFormat()
        fmt.setSamples(MSAA_SAMPLES)
        fmt.setDepthBufferSize(24)
        found["samples"] = fmt.samples()

        surface = QtGui.QOffscreenSurface()
        surface.setFormat(fmt)
        surface.create()
        found["surface_valid"] = bool(surface.isValid())

        context = QtGui.QOpenGLContext()
        created = bool(context.create()) if found["surface_valid"] else False
        found["context_created"] = created
        current = bool(context.makeCurrent(surface)) if created else False
        found["context_current"] = current

        if current:
            fmt_now = context.format()
            found["gl"] = "%d.%d" % (fmt_now.majorVersion(), fmt_now.minorVersion())
            # The function table, tested for real. This PyQt6 does not wrap
            # QOpenGLFunctions, but it does wrap the per-profile subclasses, and
            # their initializeOpenGLFunctions() answers the same question against
            # the current context. Recorded because a context whose function
            # table will not resolve is not a usable viewport, and reporting
            # only the context would repeat the mistake this stage exists to fix.
            try:
                from PyQt6.QtOpenGL import QOpenGLFunctions_2_0

                found["functions"] = bool(
                    QOpenGLFunctions_2_0().initializeOpenGLFunctions()
                )
            except Exception as exc:
                found["functions"] = False
                found["functions_problem"] = f"{type(exc).__name__}: {exc}"
        else:
            found["gl"] = None
            found["functions"] = False
    except Exception as exc:  # noqa: BLE001 - any failure is "no context"
        found.setdefault("gl", None)
        found.setdefault("functions", False)
        problem = f"the raw context probe failed ({type(exc).__name__}: {exc})"
    finally:
        # Tearing down in this order matters: a context cannot be destroyed
        # while it is current, and an offscreen surface outlives its context.
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
        del app

    found["initialised"] = bool(
        found.get("context_current") and found.get("gl") and found.get("functions")
    )
    if not found["initialised"] and problem is None:
        problem = (
            "no OpenGL context on an offscreen surface"
            f" (surface valid={found.get('surface_valid')},"
            f" context created={found.get('context_created')},"
            f" made current={found.get('context_current')},"
            f" function table={found.get('functions')})"
        )
    found["problem"] = None if found["initialised"] else problem
    found["waited"] = time.perf_counter() - started
    return found, (None if found["initialised"] else problem)


#: The probe runs in a **child process**, and this is the one that matters most in
#: the whole file. Asking Qt for a context is not a Python operation that can
#: fail: on a machine that cannot supply one, `show()` blocks and then Windows
#: terminates the process with `0xC0000409` -- a `__fastfail`, not an exception,
#: so no `try` in this process can see it. That was measured here: the same
#: command returned exit 4 with a full report minutes earlier and then died
#: natively three times out of three, with no change to this file in between.
#:
#: A diagnostic that kills the process is useless on exactly the machines that
#: need it -- a remote desktop, a VM without 3-D acceleration, a virtual display
#: adapter. So the child is allowed to die, and the parent reads the exit code.
#:
#: The child is therefore the only way to ask this question, and it still has to
#: die alone. What it asks, though, is the product's question and not a proxy
#: for it: it constructs `app.Viewport` and waits for *that* to build its
#: moderngl context. The earlier version of this child built a bare
#: `QOpenGLWidget` subclass and demanded `initializeOpenGLFunctions()`, which
#: ends the process with `0xC0000409` on a machine where the viewport itself
#: renders perfectly -- see `scripts/gl_route_probe_width.py`, which is where
#: that was measured rather than believed.
_PROBE_CHILD = r"""
import json, sys, time
from PyQt6 import QtCore, QtGui, QtWidgets
from opendocking.workbench.app import MSAA_SAMPLES, Viewport

started = time.perf_counter()
found = {}
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
found["platform"] = QtGui.QGuiApplication.platformName()
try:
    found["screens"] = len(QtGui.QGuiApplication.screens())
except Exception:
    found["screens"] = "?"

# The product's own widget, not a stand-in for it. `Viewport.__init__` sets the
# minimum size, the focus policy and the surface format, so the format this
# stage asks for is the product's by construction and cannot drift from it.
# `MSAA_SAMPLES` is still reported because consumers read it from the payload.
found["samples"] = MSAA_SAMPLES
found["subject"] = "opendocking.workbench.app.Viewport"

# Which copy of `app` *this* process imported. It has to be measured here and
# not in the parent: this is the process that constructs the Viewport, and the
# parent never imports it. The helper is the launcher's own, so there is one
# implementation of "name the file" rather than two that can disagree. Failure
# here costs the provenance and nothing else -- a report that cannot say which
# copy answered must still say what the copy did.
try:
    from opendocking.workbench import launcher as _launcher
    found["subject_copy"] = _launcher._path_facts(
        sys.modules["opendocking.workbench.app"].__file__,
        "opendocking.workbench.app",
        "the stage-2 child process",
    )
except Exception as _exc:  # noqa: BLE001 - provenance must never cost the verdict
    found["subject_copy"] = {
        "module": "opendocking.workbench.app",
        "path": getattr(
            sys.modules.get("opendocking.workbench.app"), "__file__", None
        ),
        "problem": (
            "the launcher's provenance helper was unavailable "
            f"({type(_exc).__name__}: {_exc})"
        ),
    }
viewport = Viewport()

holder = QtWidgets.QWidget()
holder.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
QtWidgets.QVBoxLayout(holder).addWidget(viewport)
holder.resize(320, 240)

# Printed *before* show(), because on a machine that cannot bring the viewport
# up, show() is where the process ends, and a report of "?" for every field is a
# worse answer than the same report with the facts it could establish. The
# parent merges every line it can parse, so a second, fuller line supersedes
# this one when the child survives.
found["stage"] = "before-show"
found["waited"] = time.perf_counter() - started
sys.stdout.write(json.dumps(found) + "\n")
sys.stdout.flush()

holder.show()
found["show_seconds"] = time.perf_counter() - started

# "Ready" is the product's own condition. `Viewport.initializeGL` sets `_ctx` to
# a live moderngl context and builds the GPU programs, and `paintGL` returns
# immediately when `_ctx` is None -- so a viewport without `_ctx` is a viewport
# that opens with nothing drawn in it. Requiring `_ctx` is what makes exit 5 a
# statement about the viewer rather than about this file.
deadline = time.perf_counter() + 15.0
while viewport._ctx is None and time.perf_counter() < deadline:
    app.processEvents()
    time.sleep(0.005)

# Pump a little past readiness, so `paintGL` has actually run at least once
# before the child reports. "The viewer would open with content in it" is a
# claim about a painted frame, not about a context that merely exists.
if viewport._ctx is not None:
    settle = time.perf_counter() + 0.5
    while time.perf_counter() < settle:
        app.processEvents()
        time.sleep(0.005)

ctx = QtGui.QOpenGLContext.currentContext()
if ctx is not None:
    fmt = ctx.format()
    found["gl"] = "%d.%d" % (fmt.majorVersion(), fmt.minorVersion())
else:
    found["gl"] = None
found["functions"] = viewport._ctx is not None
found["moderngl"] = viewport._ctx is not None
found["widget_valid"] = bool(viewport.isValid())
found["initialised"] = viewport._ctx is not None
found["waited"] = time.perf_counter() - started
found["problem"] = None if viewport._ctx is not None else (
    "the product's own Viewport did not build a moderngl context, so the "
    "viewer window would open with nothing rendered in it"
)
found["stage"] = "done"
sys.stdout.write(json.dumps(found) + "\n")
sys.stdout.flush()
sys.exit(0 if viewport._ctx is not None else 4)
"""


def _probe_opengl(timeout: float = _GL_PUMP_SECONDS) -> tuple[dict, str | None]:
    """Ask Qt for the OpenGL context the viewport needs, and report the answer.

    Returns ``(found, problem)``. `found` is whatever could be established, so
    the failure message can quote it rather than shrug.

    The probe builds **the product's own `app.Viewport`**, in a holder carrying
    ``WA_DontShowOnScreen`` so the context is realised without a window
    appearing, and pumps the event loop with a deadline rather than entering it,
    so this cannot hang. Readiness is the product's own condition -- the moderngl
    context `Viewport.initializeGL` builds -- so a positive answer means the
    viewer would draw, and a negative one means it would open empty.

    It uses the product's class rather than a stand-in, and the reason is
    measured rather than preferred: on this machine a bare ``QOpenGLWidget``
    stand-in is killed by the driver with ``0xC0000409`` on its own
    ``initializeOpenGLFunctions()`` call, every time, in 16 runs -- while the
    real ``Viewport`` in the same process, at the same time, gets GL 4.6 and
    builds its moderngl context. See ``scripts/gl_route_probe_width.py``. A
    stand-in that fails where the product succeeds makes the probe narrower than
    the thing it judges, and a diagnostic in that state reports a falsehood
    about the product, so there is no stand-in here any more.

    It runs in a **child process**. That is not tidiness: on a machine that
    cannot supply a context, Qt terminates the process with `0xC0000409`
    instead of raising, so an in-process probe can only report the machines
    that work and die on the machines that do not -- which is backwards. The
    parent reads the child's exit status, so a dead child is a *result*.

    Which verdict that result earns depends on what `_probe_raw_context` found
    first: a dead child after a working raw context is ``_EXIT_NO_WIDGET`` (5),
    and on its own -- had there been no cheap first stage -- it would have been
    ``_EXIT_NO_CONTEXT`` (4). That is the whole reason the two stages exist.

    `moderngl.create_standalone_context` is deliberately *not* used. It fails on
    machines where the viewer works perfectly well, because `app` wraps Qt's
    context instead of making a second one -- two live contexts in one process
    on Windows cannot share textures, so a standalone one would render into
    memory Qt could never see. It is the wrong question. The moderngl context
    this stage does check is the product's own, taken from Qt's context by
    `Viewport` itself rather than made here.
    """
    import json
    import subprocess
    import time

    started = time.perf_counter()
    budget = max(float(timeout), 1.0) + 15.0
    found: dict = {}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_CHILD],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return (
            {"waited": time.perf_counter() - started},
            f"the OpenGL probe did not answer within {budget:.0f} s",
        )
    except OSError as exc:  # pragma: no cover - no interpreter to spawn
        return {"waited": time.perf_counter() - started}, f"the OpenGL probe could not be run ({exc})"

    raw = (proc.stdout or "").strip()
    if raw:
        # The child prints a first line before it attempts the context and a
        # fuller one after, and on a machine that cannot supply a context it
        # never gets to the second. Merge everything, so the facts that *were*
        # established survive the failure. Later lines win.
        for line in raw.splitlines():
            try:
                loaded = json.loads(line)
            except ValueError:
                continue
            if isinstance(loaded, dict):
                found.update(loaded)
    if not isinstance(found.get("waited"), (int, float)):
        found["waited"] = time.perf_counter() - started
    found.pop("stage", None)
    # The child's own status, recorded whether it answered or not. A dead child
    # is a result, and the number it died with is the most specific fact
    # available about it -- `0xC0000409` says the driver ended the process,
    # which no amount of guessing at the failure could tell us.
    found["child_exit"] = proc.returncode

    if proc.returncode == 0 and found.get("initialised"):
        found.pop("initialised", None)
        found.pop("problem", None)
        return found, None
    if proc.returncode == 0 and found:
        found.pop("initialised", None)
        return found, found.get("problem") or "no OpenGL context"
    # No JSON, or JSON without an answer: the child died. On this machine that is
    # what a driver that cannot supply a context does, and it is the case the
    # command exists for, so it is reported rather than treated as a crash of
    # the diagnostic itself.
    return found, (
        f"Qt could not create the context and the probe ended without "
        f"answering (exit {proc.returncode})"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    launch, problem = _preflight()
    if launch is None:
        # The copy is named on this path too, and it is the path where it is
        # worth most: "PyQt6 is missing" is what a user sees when their install
        # is wrong, and the first question after that is *which* install. The
        # helper imports nothing, so it cannot fail the way the thing it is
        # describing just did.
        verdict = {
            "verdict": _VERDICT_NO_GUI_STACK,
            "exit": _EXIT_NO_GUI_STACK,
            "problem": problem,
            "provenance": _provenance(),
        }
        if args.as_json:
            # `--json` promises a parseable answer, and the promise has to hold
            # on the failure that a shell script is most likely to hit -- the
            # one where PyQt6 is absent, which is also the one where a bare
            # `import` would have succeeded.
            _print_verdict(verdict)
        else:
            print(problem, file=sys.stderr)
            print(_describe_provenance(verdict), file=sys.stderr)
        return _EXIT_NO_GUI_STACK

    # `--check` answers a question about the *product*, not about the arguments,
    # so it runs before the file checks rather than after them. `odgui --check -r
    # typo.pdbqt` used to exit 2 complaining about the typo and never mention
    # the viewer, which is the opposite of what was asked for.
    if args.check:
        return _report_check(as_json=args.as_json)

    if args.as_json:
        # Silently ignoring it would leave a script parsing prose it cannot
        # read, with a 0 that looks like a successful check.
        print("error: --json is only meaningful together with --check", file=sys.stderr)
        return 2

    for label, path in (
        ("receptor", args.receptor),
        ("ligand", args.ligand),
        ("poses", args.poses),
    ):
        if path is not None and not path.is_file():
            print(f"error: {label} file not found: {path}", file=sys.stderr)
            return 2

    if not any((args.receptor, args.ligand, args.poses)):
        print(
            f"{_BANNER}\n"
            "No structure given; the window opens empty.\n"
            "  receptor/ligand : File -> Open structure, or -r / -l\n"
            "  docked poses    : odgui -p poses.pdbqt\n"
            "Run 'odgui --help' for all options.",
            file=sys.stderr,
        )

    return launch(args.receptor, args.ligand, args.poses)


def _print_verdict(payload: dict) -> None:
    """Emit a ``--check`` verdict as one JSON object on stdout.

    Keys are sorted and values that json cannot represent are stringified
    rather than dropped, so a script can assert on the object and a probe field
    added later shows up in the output instead of silently vanishing.
    """
    import json

    print(json.dumps(payload, sort_keys=True, default=str))


def _report_check(as_json: bool = False) -> int:
    """The two stages behind ``--check``, and the code each one earns.

    Stage 1 has already run by the time this is called: `_preflight` imported
    the GUI stack, or returned a message and a code. Stage 2 is the one that
    used not to exist at all, which is why `--check` claimed a working viewer on
    a machine that could not start one.

    The success report goes to stdout because it *is* the result, and the two
    failures go to stderr because they are diagnostics. That split is the same
    one every other failure in this project now obeys.

    ``as_json`` replaces the prose with a single object **built from the same
    dictionaries the prose is formatted from**, so the two cannot drift apart.
    The object carries a ``raw`` and a ``widget`` member, so a script can assert
    that the cheap stage succeeded rather than only that nothing went red. The
    exit code is identical either way.
    """
    raw, raw_problem = _probe_raw_context()

    if raw_problem is None:
        widget, widget_problem = _probe_opengl()
    else:
        widget, widget_problem = None, None
        widget = {
            "skipped": "the raw context stage failed, so the widget stage was "
            "not attempted; it could not have changed the answer"
        }

    payload: dict = {
        "raw": raw,
        "widget": widget,
        # The child measured `app`; the fallback in `_provenance` names which
        # of the two readings this is, so a reader is never left assuming the
        # parent knew something it could not have known.
        "provenance": _provenance((widget or {}).get("subject_copy")),
    }
    # The convenience keys, at the top level, so a consumer written against the
    # pre-two-stage object keeps working unchanged.
    payload["samples"] = raw.get("samples")
    payload["platform"] = raw.get("platform")
    payload["screens"] = raw.get("screens")

    if raw_problem is not None:
        payload["verdict"] = _VERDICT_NO_CONTEXT
        payload["exit"] = _EXIT_NO_CONTEXT
        payload["problem"] = raw_problem
        if as_json:
            _print_verdict(payload)
        else:
            print(_describe_no_context(raw), file=sys.stderr)
            print(_describe_provenance(payload), file=sys.stderr)
        return _EXIT_NO_CONTEXT

    if widget_problem is not None:
        payload["verdict"] = _VERDICT_NO_WIDGET
        payload["exit"] = _EXIT_NO_WIDGET
        payload["problem"] = widget_problem
        payload["raw_ok"] = True
        if as_json:
            _print_verdict(payload)
        else:
            print(_describe_no_widget(payload), file=sys.stderr)
            print(_describe_provenance(payload), file=sys.stderr)
        return _EXIT_NO_WIDGET

    import opendocking

    payload["verdict"] = _VERDICT_OK
    payload["exit"] = _EXIT_OK
    payload["engine"] = opendocking.engine_version()
    payload["gpu"] = opendocking.gpu_status()
    if as_json:
        _print_verdict(payload)
        return _EXIT_OK

    print(f"{_BANNER}: GUI stack OK")
    print(f"  engine  {payload['engine']}")
    gpu = payload["gpu"]
    print(
        "  gpu     "
        + (
            "compiled in, adapter available"
            if gpu.get("available")
            else ("compiled in, no adapter" if gpu.get("compiled") else "not compiled in")
        )
    )
    print(f"  qt      {raw.get('platform')} platform, "
          f"{raw.get('screens')} screen(s)")
    print(f"  opengl  raw context GL {raw.get('gl')}, functions "
          f"{'usable' if raw.get('functions') else 'UNUSABLE'} "
          f"({_seconds(raw.get('waited'))})")
    # "app.Viewport came up" rather than "widget GL ..." on purpose. The stage
    # that produced this line now runs the product's own widget class, and a
    # success report that still called it a bare "widget" would leave a reader
    # with the old, wrong impression that the check never touched the viewer.
    print(f"          {widget.get('subject', 'the product viewport')} came up: "
          f"GL {widget.get('gl')}, {widget.get('samples')}x MSAA, depth 24, "
          f"moderngl context "
          f"{'built' if widget.get('moderngl') else 'NOT built'}")
    # Last, and not folded into a line above, because it answers a different
    # question: the lines above say what the machine did, and this says *which
    # code* answered. On a machine where the installed copy lags the tree, every
    # line above is true and none of them is about the copy being edited.
    print(_describe_provenance(payload))
    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
