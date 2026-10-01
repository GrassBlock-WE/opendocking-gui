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

`_overlap_report()` imports the matrix's declared routes and refuses to stay
quiet if a widget probe here is configured by a subset of a route the ladder
walks -- which is exactly how the duplication came back in the first time, since
every removed probe was a strict subset of some rung. A comment saying "do not
re-add these" rots silently; this prints instead.

**It still never fails, and that is now load-bearing rather than incidental.**

This step runs inside the workbench job, where a failing step cancels every step
after it. That is exactly what happened when the GUI stack check exited 5 and
thirteen audits were cancelled. So a detected overlap is reported loudly and the
process still exits 0: the finding is for a human, and the alternative is
re-creating the failure mode the job was restructured to remove.

The one exception is unchanged from before: if PyQt6 will not import, there is
no question to ask, so the file prints that and exits 0.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def probe(name: str, body) -> bool:
    """Run one strategy in a child process and report whether GL came up."""
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(body)
        path = fh.name
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=180)
    Path(path).unlink(missing_ok=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    verdict = "GL_OK" if "GL_OK" in out else "no GL"
    detail = out.splitlines()[-1] if out else (err.splitlines()[-1] if err else "(no output)")
    print(f"  {name:<44} {verdict:<7} exit {proc.returncode}  {detail[:60]}")
    return "GL_OK" in out


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


def _overlap_report() -> list:
    """Widget routes here that the CI ladder already walks. Empty when healthy.

    Compares against `gl_route_matrix.ROUTES` rather than against a name list
    copied here, because a copy is the thing that goes stale. A route counts as
    overlapping when its overrides are a **subset** of a ladder route's: every
    probe removed from this file was a strict subset of some rung, so a subset
    test catches the real shape of the duplication and not just an exact repeat.

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
        return []

    clashes = []
    for label, env, body in ROUTES:
        if body != "widget_result()":
            continue
        for rung in LADDER:
            if env.items() <= rung.env.items():
                clashes.append((label, rung.name, rung.ladder_index))
    return clashes


def main() -> int:
    from PyQt6 import QtGui  # noqa: F401  (import check only)

    print("Qt OpenGL widget availability\n")
    print("product-level routes are enumerated in gl_route_matrix.py, which "
          "walks the CI capability ladder; this file asks only what that "
          "ladder does not\n")

    clashes = _overlap_report()
    for label, rung, index in clashes:
        print(f"  ::error::ROUTE OVERLAP: {label!r} is a subset of CI ladder "
              f"rung {index} ({rung!r}). It is walked twice; remove it from "
              f"ROUTES here. Reporting and continuing, because this step must "
              f"not cancel the steps after it.")

    wins = []
    for label, env, body in ROUTES:
        wins.append(probe(label, TEMPLATE.format(setup=_setup_for(env), body=body)))

    print()
    if any(wins):
        print("at least one route obtained a GL context")
    else:
        print(
            "no route obtained a GL context in this environment. That is "
            "consistent with a headless session with no display driver: the "
            "renderer itself is verified separately against a standalone "
            "moderngl context."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print(f"PyQt6 unavailable: {e}")
        raise SystemExit(0)
    except Exception:
        traceback.print_exc()
        raise SystemExit(0)
