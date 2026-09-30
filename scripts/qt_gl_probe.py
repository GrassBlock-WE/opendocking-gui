"""Try harder to get a real OpenGL context inside a Qt `QOpenGLWidget`.

The workbench renderer is already verified against a standalone moderngl
context (see `workbench_smoke.py`). What has *not* been verified is whether a
Qt widget can obtain a GL context at all in this environment, which is what
`initializeGL` needs.

A "no context here" message is not automatically a bug: a headless Windows
session with no display driver cannot create one, and `offscreen` in
particular is documented not to support OpenGL. The question worth answering
is *which* of the available routes actually works, so that a user on a real
desktop gets a working window and so that any future failure can be attributed
correctly.

Routes tried:
  1. default platform
  2. offscreen platform
  3. software rendering (`QT_OPENGL=software`)
  4. ANGLE / native / software requested via QSurfaceFormat
  5. a bare `QOpenGLContext` on an offscreen surface, bypassing the widget
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
    verdict = "GL OK" if "GL_OK" in out else "no GL"
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


def main() -> int:
    from PyQt6 import QtGui  # noqa: F401  (import check only)

    print("Qt OpenGL widget availability\n")
    wins = []

    wins.append(
        probe(
            "default platform, widget",
            TEMPLATE.format(setup="", body="widget_result()"),
        )
    )
    wins.append(
        probe(
            "offscreen platform, widget",
            TEMPLATE.format(
                setup='os.environ["QT_QPA_PLATFORM"] = "offscreen"', body="widget_result()"
            ),
        )
    )
    wins.append(
        probe(
            "QT_OPENGL=software, widget",
            TEMPLATE.format(
                setup='os.environ["QT_OPENGL"] = "software"', body="widget_result()"
            ),
        )
    )
    wins.append(
        probe(
            "QT_OPENGL=angle, widget",
            TEMPLATE.format(
                setup='os.environ["QT_OPENGL"] = "angle"', body="widget_result()"
            ),
        )
    )
    wins.append(
        probe(
            "QT_OPENGL=desktop, widget",
            TEMPLATE.format(
                setup='os.environ["QT_OPENGL"] = "desktop"', body="widget_result()"
            ),
        )
    )
    wins.append(
        probe(
            "raw QOpenGLContext on offscreen surface",
            TEMPLATE.format(setup="", body="raw_context()"),
        )
    )

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
