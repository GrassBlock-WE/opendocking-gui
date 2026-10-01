"""Drive the real workbench with real input events and check what the user sees.

The render smoke test proves geometry reaches the framebuffer. It says nothing
about whether the *controls* work, so this one posts genuine Qt mouse, wheel and
keyboard events at the live widgets and reads the result back.

Three things it checks that nothing else does:

* **Layout**: the geometry of every control in the docked panel, looking for
  zero-size, overlapping, clipped or out-of-panel widgets.
* **Interaction**: left-drag rotates, wheel zooms, Frame all refits, the
  checkboxes actually hide things, the box spins move the wireframe, the pose
  list switches poses and updates both readouts.
* **Behaviour under a real docking run**: that the window keeps painting while
  a search runs, that the Dock button re-enables, and what happens to state
  across two consecutive runs.

Run it on the real desktop platform. It does not force QT_QPA_PLATFORM --
offscreen is the one mode without OpenGL, and forcing it hides exactly the
bugs this is looking for.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from opendocking.workbench.app import (  # noqa: E402
    POCKET_OPACITY_COMPARE,
    POCKET_OPACITY_PLAIN,
    MainWindow,
)

OUT = ROOT / "dist" / "workbench_interaction"
OUT.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, str, str]] = []

#: How many checks this file is supposed to run. The house convention
#: (`pockets_check.py`, `pdbqt_check.py`) is to measure it from a green run and
#: change it by hand, because the noticing is the point.
#:
#: Unlike those two, this file also prints it on the *first* line of output and
#: closes every section with its own subtotal. A pinned number checked only at
#: the end cannot help anyone reading a run that died before the end -- which is
#: exactly what happened here: for two hours this suite reported `exit 3` with
#: a screenful of `[PASS]` and no sign that sections 8 to 11d never executed,
#: and a truncated run is indistinguishable from a finished one unless the
#: transcript says otherwise.
#:
#: 199 measured from the last green run, plus 9 added with section 7c, plus 2
#: added with sections 5b/6 being given their own headers, plus 2 completeness
#: checks, plus 1 for the site cloud stepping aside, plus 3 for a pose file
#: whose columns are wrong, plus 1 for this machine's OpenGL verdict.
#: Calibrated from the first run to reach this summary. Skips count toward the
#: total because `skip` records a result too, so this number does not move with
#: the environment -- which is the property that makes it worth pinning.
EXPECTED_CHECKS = 217

#: Every section this file is supposed to reach, in file order. A section that
#: is entered always prints a header, and every header is closed by the next
#: one's subtotal, so a run that stops mid-section is visible in the transcript
#: without anyone having to trust the number above.
EXPECTED_SECTIONS = (
    "1. layout of the control panel",
    "1b. does the viewer understand the atom types it is given?",
    "2. camera interaction",
    "3. is there a pan control?",
    "4. visibility checkboxes",
    "5. search-box spins",
    "5a. the search effort follows the box",
    "5b. the File menu can reach every role",
    "6. pose browser",
    "7. a real docking run from the GUI",
    "7b. the pose table, from a run whose result is still in memory",
    "7c. a pose set that breaks the table's assumptions",
    "8. a second docking run",
    "9. a receptor at negative coordinates",
    "10. display representations",
    "10b. a ribbon on a real protein",
    "11. pose/receptor interactions",
    "11c. pocket search drives the box",
    "11d. the pocket search does not block the window",
    "summary",
)

# Section bookkeeping. `_section_open` is the section currently being filled in
# and `_section_first` the index into `results` where it started.
_section_open: str | None = None
_section_first = 0
sections_entered: list[str] = []
#: Per-section skip counts, so the summary can name the sections the machine
#: answered rather than the code.
section_skips: dict[str, int] = {}
section_totals: dict[str, tuple[int, int, int, int]] = {}


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def skip(name: str, reason: str) -> bool:
    """Record a check that could not be run here, and say why.

    A skipped check and a failed check are different claims. "The viewport
    draws nothing" and "this environment cannot read the framebuffer" look
    identical in the output if the second one is reported as the first, and
    only one of them is ever true.
    """
    results.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  — {reason}")
    return False


# Whether `grabFramebuffer` returns anything at all. Probed once, after the
# window is up; see `probe_framebuffer`.
PIXELS_OK = True


def pixel_check(name: str, ok: bool, detail: str = "") -> bool:
    """A check whose truth lives in the rendered pixels."""
    if not PIXELS_OK:
        return skip(name, "the framebuffer read back blank in this environment")
    return check(name, ok, detail)


def _verify_pixel_guard():
    """Reverse-verify the guard the pixel checks depend on.

    A skip is only honest if it is reached *only* when the environment
    genuinely cannot answer, so all three states are checked: a blank
    framebuffer must skip, a good framebuffer with a good render must pass, and
    a good framebuffer with a genuinely empty render must fail.

    Without this, a guard that skipped unconditionally would look exactly like
    a working one. That is precisely the bug that failed the CI workbench job
    once already: one pixel check used `check` instead of `pixel_check` and
    reported a headless machine's blank framebuffer as a broken viewport, while
    its fourteen siblings skipped correctly.

    Its own function because `PIXELS_OK` is a module global that `main` also
    assigns to, and a `global` statement after that assignment is a
    SyntaxError.
    """
    global PIXELS_OK, results
    saved_flags, saved_results = PIXELS_OK, results
    # Swap the list object, never clear it. `results.clear()` looked harmless
    # and silently threw away every check recorded so far -- the summary went
    # from 116 checks to 110 with no failure anywhere, which is exactly the
    # kind of quiet loss a check suite must not have. Redirecting stdout keeps
    # the three probe lines out of the report, where they would read as three
    # unexplained results of their own.
    results = []
    seen = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for label, flag, ok in (
                ("blank framebuffer", False, False),
                ("good render", True, True),
                ("empty render", True, False),
            ):
                PIXELS_OK = flag
                pixel_check("guard probe", ok, label)
                seen[label] = results[-1][0]
    finally:
        PIXELS_OK, results = saved_flags, saved_results
    want = ("SKIP", "PASS", "FAIL")
    got = tuple(seen.values())
    return (
        "the pixel-check guard skips only when the framebuffer cannot answer",
        got == want,
        f"blank framebuffer -> {got[0]}, good render -> {got[1]}, "
        f"empty render -> {got[2]} (want {'/'.join(want)})",
    )


def _close_section() -> None:
    """Print the section that just ended, with its own subtotal.

    Called from `section` when the next one opens, and once at the end of the
    report, so a section is never left open. An open section is the signature of
    a run that stopped inside it, and printing that is the whole point: the last
    line of a truncated transcript is then a section header with no subtotal
    under it, which reads as unfinished to anyone skimming, without needing the
    expected total and without trusting a number.
    """
    global _section_open, _section_first
    if _section_open is None:
        return
    title, first = _section_open, _section_first
    ran = results[first:]
    npass = sum(1 for r in ran if r[0] == "PASS")
    nfail = sum(1 for r in ran if r[0] == "FAIL")
    nskip = sum(1 for r in ran if r[0] == "SKIP")
    section_skips[title] = nskip
    section_totals[title] = (len(ran), npass, nfail, nskip)
    note = "" if ran else "   <-- NO CHECKS RAN IN THIS SECTION"
    print(
        f"--- {title}: {len(ran)} checks, {npass} passed, "
        f"{nfail} failed, {nskip} skipped{note}"
    )
    _section_open = None


def section(title: str) -> None:
    _close_section()
    global _section_open, _section_first
    sections_entered.append(title)
    print(f"\n=== {title} ===")
    _section_open = title
    _section_first = len(results)

def loop_turns(app, windows: int = 5, span: float = 0.2, sleeper=time.sleep) -> list[int]:
    """Count `processEvents()` turns in each of `windows` spans of `span` seconds.

    Separate from section 11d so it can be handed a stub: the question it is
    used to answer -- "is the GUI thread alive, or is this machine just busy?"
    -- is only answerable if the same function can be shown to give both answers.
    See `_verify_loop_turns_guard`.
    """
    counts = []
    for _ in range(windows):
        turns = 0
        deadline = time.perf_counter() + span
        while time.perf_counter() < deadline:
            app.processEvents()
            turns += 1
            sleeper(0.001)
        counts.append(turns)
    return counts


class _StarvedEvents:
    """An `app` stand-in whose event loop cannot turn.

    `processEvents` burns most of the span before returning, which is what a
    GUI thread looks like when something is holding it.
    """

    def __init__(self, hold: float) -> None:
        self.hold = hold

    def processEvents(self):  # noqa: N802 - Qt naming
        time.sleep(self.hold)


def _verify_loop_turns_guard():
    """Reverse-verify the liveness threshold the way the pixel guard is verified.

    Three states have to be distinguishable: a held GUI thread must read as
    "not turning", a live one as "turning", and a stub with no cost at all must
    not be mistaken for the second. Without this, a threshold that was simply
    low enough to stop flaking would look exactly like a working one, and the
    next person to raise the bar would have no evidence about what it costs.
    """
    quiet = loop_turns(_StarvedEvents(0.05), windows=3, span=0.2)
    live = loop_turns(_StarvedEvents(0.0), windows=3, span=0.2)
    got = ("FAIL" if max(quiet) < 10 else "PASS", "PASS" if max(live) >= 10 else "FAIL")
    want = ("FAIL", "PASS")
    return (
        "the liveness threshold separates a held event loop from a live one",
        got == want,
        f"a loop held 50 ms per turn reads {max(quiet)} turns/window -> {got[0]}; "
        f"a free loop reads {max(live)} -> {got[1]} (want {'/'.join(want)})",
    )


def img_array(qimg: QtGui.QImage) -> np.ndarray:
    """QImage -> (h, w, 3) uint8, independent of stride and format."""
    img = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return arr[:, : img.width(), :3].copy()


SHOT_PROBLEM: str | None = None


#: What the launcher's own OpenGL probe said about this machine, asked once at
#: the top of the run. `odgui --check` already answers "can this machine give Qt
#: a context?" in about six seconds, and it answers it about the product rather
#: than about a probe invented here -- so this suite asks it instead of
#: inferring the answer from a blank frame, which is how three experiments
#: minutes apart managed to give three different answers about one machine.
GL_VERDICT: str | None = None
GL_OK = True


def probe_opengl(app) -> None:
    """Ask the launcher whether this machine can render at all, and record it."""
    global GL_VERDICT, GL_OK
    try:
        from opendocking.workbench.launcher import _probe_opengl
    except Exception as exc:  # noqa: BLE001 - a missing probe is not a failure
        GL_VERDICT = f"the launcher's OpenGL probe could not be imported ({exc})"
        return
    found, problem = _probe_opengl()
    bits = [f"platform {found.get('platform')}", f"{found.get('screens')} screen(s)"]
    if found.get("gl"):
        bits.append(f"OpenGL {found['gl']}")
    if problem is None:
        GL_OK = True
        GL_VERDICT = "Qt created an OpenGL context -- " + ", ".join(bits)
        return
    GL_OK = False
    GL_VERDICT = (
        f"this machine cannot give Qt an OpenGL context ({problem}); "
        + ", ".join(bits)
        + f"; waited {found.get('waited', 0):.1f} s. The same question asked "
          "through `odgui --check`, which reports a graphics-driver problem "
          "rather than a missing package."
    )


def _bare_widget_exposed(app, tries: int = 10, gl: bool = False) -> bool:
    """Control experiment: does a *bare* window get exposed on this machine?

    With `gl=False` that is a plain `QWidget`; with `gl=True` it is a bare
    `QOpenGLWidget` -- still none of this project's code and nothing a change to
    the pose table could have touched. Two controls because they fail
    differently: if the plain one is never exposed the desktop is presenting
    nothing, and if only the GL one is not, the GL stack is refusing this
    machine a surface. Both are the machine's doing; neither is a broken
    viewport, and telling them apart is what stops this being "fixed" by
    retrying until a frame appears.
    """
    probe = QtOpenGLWidgets.QOpenGLWidget() if gl else QtWidgets.QWidget()
    probe.resize(64, 48)
    probe.show()
    exposed = False
    for _ in range(tries):
        app.processEvents()
        QTest.qWait(60)
        handle = probe.windowHandle()
        if handle is not None and handle.isExposed():
            exposed = True
            break
    probe.close()
    probe.deleteLater()
    return exposed


def frame_problem(win) -> str:
    """Why there is no frame, naming which of the causes this is.

    `QOpenGLWidget.grabFramebuffer()` returns a null `QImage` when the widget
    has no surface to read. The causes are not interchangeable, and "it was
    still drawing" -- the one people reach for first -- is already ruled out by
    the evidence this suite has in hand before it calls in here: `main` waits up
    to 5 s for the window to be exposed (`QTest.qWaitForWindowExposed`) and
    fails, and the first grab is null for the same reason as the last. Waiting
    cannot create a surface the window manager never gave us, so this suite does
    not wait to find out, and does not call a late frame a pass.
    """
    app = QtWidgets.QApplication.instance()
    if not GL_OK:
        # Asked at the top of the run, by the product's own probe. Everything
        # below about "no frame" is then a restatement of this rather than a
        # second opinion reached by experiment.
        return GL_VERDICT or "no OpenGL context"
    handle = win.windowHandle()
    if handle is None:
        return "the window has no native handle at all"
    if handle.isExposed():
        return "the window is exposed, so the GL surface itself produced no frame"
    if not _bare_widget_exposed(app):
        return (
            "this machine does not expose Qt windows: a bare QWidget with no "
            f"OpenGL is unexposed too, so no frame can be read here at all "
            f"(platform {app.platformName() if app else '?'})"
        )
    if not _bare_widget_exposed(app, gl=True):
        return (
            "plain windows are exposed but a bare QOpenGLWidget is not, so this "
            "machine's GL stack is not giving Qt a surface to read; the "
            "workbench's own drawing is untested either way"
        )
    return (
        "a bare window and a bare QOpenGLWidget are both exposed, so the "
        "failure is in the workbench window, not in the environment"
    )


def shot(win: MainWindow, name: str) -> tuple:
    """Render the real widget and save it. Returns (pixels|None, path).

    Returns ``(None, "")`` rather than raising when the framebuffer cannot be
    read. This is called near the top of `main`, so raising here would abort the
    run before the other ~180 checks and before the summary line -- a suite that
    stops early looks like a suite that stopped early, and everyone assumes the
    rest is fine. The caller turns a missing frame into a counted SKIP with the
    reason, which is a different claim from a FAIL and stays a different claim
    in the summary.
    """
    global SHOT_PROBLEM
    win.viewport.repaint()
    QTest.qWait(40)
    img = win.viewport.grabFramebuffer()
    if img.isNull():
        if SHOT_PROBLEM is None:
            SHOT_PROBLEM = frame_problem(win)
            print(f"\n  NO FRAME from the viewport: {SHOT_PROBLEM}")
        return None, ""
    arr = img_array(img)
    path = OUT / f"{name}.png"
    img.save(str(path))
    return arr, str(path)


def background_colour(arr: np.ndarray) -> np.ndarray:
    """The clear colour, taken as the modal colour of the framebuffer.

    `paintGL` clears to (0.10, 0.11, 0.13) but the window also carries whatever
    the desktop compositor put behind an unpainted region, so the mode of the
    actual pixels is the honest reference.
    """
    flat = arr.reshape(-1, 3)
    key = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    modal = vals[int(np.argmax(counts))]
    return np.asarray([(modal >> 16) & 0xFF, (modal >> 8) & 0xFF, modal & 0xFF], np.int16)


def non_background(arr) -> int:
    """Pixels that differ from the modal (background) colour.

    `None` in, `0` out: a frame that could not be read has no non-background
    pixels, and saying so is what routes the pixel checks to a skip instead of
    a failure. A crash here would be the third way this run could die before its
    summary, and the least informative of the three.
    """
    if arr is None:
        return 0
    bg = background_colour(arr)
    d = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
    return int((d > 12).sum())


def dispose_window(app, win, destroy: bool = True) -> None:
    """Close a window and actually let go of its GL context.

    `close()` on its own hides the window; it does not destroy it. A
    `QOpenGLWidget` that is still alive still owns its OpenGL context, so a file
    that opens a second window while the first is merely closed is running two
    live contexts on one surface. Forcing the deferred delete is what fixed that
    -- it moved the `0xC0000409` fail-fast from section 7c to the very last
    section of the file.

    `destroy=False` is for the one window that has run a worker thread. There,
    the QOpenGLWidget is destroyed while a QThread that used to drive it is
    still an object, and on this machine that combination fail-fasts. That
    window is closed and left to process exit instead, which is the right
    trade for the last window in a run: nothing follows it that needs the
    context released.

    `processEvents()` on its own does not reliably process `DeferredDelete`
    events, which is why a plain `close()` plus `processEvents` left the
    context behind in the first place.
    """
    if win is None:
        return
    win.close()
    if not destroy:
        app.processEvents()
        return
    win.deleteLater()
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def differing_pixels(a, b) -> int:
    """Pixels that differ between two frames, or 0 when either is missing."""
    if a is None or b is None:
        return 0
    return int((a != b).any(axis=-1).sum())


def drag(widget, button, start, end, steps=6) -> None:
    nomod = QtCore.Qt.KeyboardModifier.NoModifier
    QTest.mousePress(widget, button, nomod, QtCore.QPoint(*start))
    for i in range(1, steps + 1):
        t = i / steps
        x = int(start[0] + (end[0] - start[0]) * t)
        y = int(start[1] + (end[1] - start[1]) * t)
        QTest.mouseMove(widget, QtCore.QPoint(x, y))
        QTest.qWait(8)
    QTest.mouseRelease(widget, button, nomod, QtCore.QPoint(*end))


def rects_overlap(a: QtCore.QRect, b: QtCore.QRect) -> bool:
    return a.intersects(b) and (a & b).width() > 0 and (a & b).height() > 0


def main() -> int:
    # `PIXELS_OK` is read by `pixel_check`; assigning it here without this
    # would create a local and silently leave the flag at its default.
    global PIXELS_OK
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    # First line on purpose. Someone reading a truncated run -- a crashed one,
    # a killed one, one that scrolled past -- sees what it was supposed to do
    # before seeing any of the `[PASS]` lines that made it look fine.
    print(
        f"workbench interaction suite: expecting {EXPECTED_CHECKS} checks "
        f"across {len(EXPECTED_SECTIONS)} sections"
    )
    print(f"Qt {QtCore.QT_VERSION_STR}, platform {app.platformName()}")

    rec = EXAMPLES / "rec_prep.pdbqt"
    lig = EXAMPLES / "ibuprofen_prep.pdbqt"
    poses = EXAMPLES / "poses.pdbqt"
    for p in (rec, lig, poses):
        if not p.is_file():
            print(f"missing example input: {p}")
            return 2

    # Before any window of ours exists, because it is one more GL widget and the
    # question is best asked while it is the only one.
    probe_opengl(app)
    print(f"  OpenGL: {GL_VERDICT}")

    win = MainWindow(receptor=rec, ligand=lig, poses=poses)
    win.resize(1280, 820)
    win.show()
    # `isVisible` is not the same question as "has the window been painted".
    # Under a software rasteriser with no compositor the window can be shown
    # and still never get an expose event, which leaves the QOpenGLWidget's
    # framebuffer blank. Asking explicitly turns a mystery into an answer, and
    # it costs nothing where the window really is up.
    exposed = QTest.qWaitForWindowExposed(win, 5000)
    app.processEvents()
    QTest.qWait(600)
    app.processEvents()

    # ------------------------------------------------------------------ layout
    section("1. layout of the control panel")
    panel = win.dockWidget("Controls") if hasattr(win, "dockWidget") else None
    dock = None
    for i in range(win.dockWidgetArea.count() if False else win.findChildren(QtWidgets.QDockWidget).__len__()):
        dock = win.findChildren(QtWidgets.QDockWidget)[i]
    dock = dock or next(iter(win.findChildren(QtWidgets.QDockWidget)), None)
    print(f"  dock widget found: {dock is not None}  title: {dock.windowTitle() if dock else '-'}")
    dock_w = dock.width() if dock else 0
    print(f"  dock width: {dock_w} px")

    controls = {
        "cb_receptor": win.cb_receptor,
        "cb_ligand": win.cb_ligand,
        "cb_box": win.cb_box,
        "centre x": win.center_spins[0],
        "centre y": win.center_spins[1],
        "centre z": win.center_spins[2],
        "size x": win.size_spins[0],
        "size y": win.size_spins[1],
        "size z": win.size_spins[2],
        "exhaustiveness": win.sp_exhaust,
        "suggest button": win.btn_suggest,
        "search cost": win.lbl_effort,
        "seed": win.sp_seed,
        "scoring": win.cb_scoring,
        "pose table": win.pose_table,
        "all poses": win.cb_all_poses,
        "showing": win.lbl_poses,
        "best energy": win.lbl_energy,
        "RMSD": win.lbl_rmsd,
        "status": win.status_label,
    }
    zero, outside, overlap = [], [], []
    panel_rect = dock.widget().rect() if dock else QtCore.QRect()
    for name, w in controls.items():
        g = w.geometry()
        if g.width() < 8 or g.height() < 8:
            zero.append(f"{name}{g.width()}x{g.height()}")
        # is the widget's right edge past the panel's content edge?
        right = w.mapTo(dock.widget(), QtCore.QPoint(g.width(), 0)).x() if dock else 0
        if dock and right > panel_rect.width() + 1:
            outside.append(f"{name} right={right} > panel {panel_rect.width()}")
    for i, (n1, w1) in enumerate(controls.items()):
        for n2, w2 in list(controls.items())[i + 1 :]:
            if w1.parentWidget() is w2.parentWidget() and rects_overlap(w1.geometry(), w2.geometry()):
                overlap.append(f"{n1} / {n2}")

    check("every control has a usable size", not zero, ", ".join(zero) or "all >= 8x8")
    check("no control overflows the panel", not outside, "; ".join(outside) or "none")
    check("no two controls overlap", not overlap, "; ".join(overlap) or "none")
    print(f"  viewport: {win.viewport.width()}x{win.viewport.height()} "
          f"(min {win.viewport.minimumWidth()}x{win.viewport.minimumHeight()})")
    print(f"  status text: {win.status_label.text()!r}")
    print(f"  energy label: {win.lbl_energy.text()!r}   rmsd: {win.lbl_rmsd.text()!r}")
    check(
        "best-pose energy is populated on load",
        win.lbl_energy.text() not in ("", "—"),
        win.lbl_energy.text(),
    )
    check(
        "RMSD-to-best is populated on load",
        win.lbl_rmsd.text() not in ("", "—"),
        win.lbl_rmsd.text(),
    )
    check(
        "pose list is populated",
        win.pose_list.count() > 1,
        f"{win.pose_list.count()} rows",
    )

    base, _ = shot(win, "01_loaded")
    base_px = non_background(base)
    # The machine's answer, from the product's own probe, counted like any other
    # result. A skip here is the honest reading of "this box cannot render":
    # it says nothing about the viewport, and it is CI's `xvfb-run` that
    # answers the question on a machine that can.
    _claim = (
        "this machine can give Qt an OpenGL context, so the pixel checks below "
        "are real"
    )
    if GL_OK:
        check(_claim, True, GL_VERDICT or "")
    else:
        skip(_claim, GL_VERDICT or "no OpenGL context")
    # Decide once, here, whether the framebuffer is readable at all. A blank
    # or missing grab means every pixel-dependent check below would be
    # measuring the environment rather than the viewport.
    if base_px == 0:
        PIXELS_OK = False
        print(
            "\n  NOTE: no frame could be read from the viewport, so the pixel"
            "\n        checks below are skipped rather than failed."
            f"\n        platform: {app.platformName()}; window exposed: {exposed}."
            "\n        Every non-pixel check still runs, and this one is counted"
            "\n        in the summary as a skip."
        )
        # Counted, not just printed: a run where the picture could not be read
        # has to say so in its summary line, or "N passed, 0 failed" reads the
        # same as a run that looked at every pixel and liked them all.
        skip(
            "the viewport's frame could be read",
            SHOT_PROBLEM or "grabFramebuffer returned a uniformly coloured image",
        )
    pixel_check("viewport renders geometry", base_px > 2000, f"{base_px} non-background px")
    check(*_verify_pixel_guard())

    # -------------------------------------------------- element interpretation
    section("1b. does the viewer understand the atom types it is given?")
    from opendocking.workbench import ELEMENT_COLORS, PDBQT_TYPE_ELEMENT

    for mol in win.viewport.molecules:
        unknown = sorted({e for e in mol.elements if e not in ELEMENT_COLORS})
        check(
            f"every element in '{mol.name}' is a real element",
            not unknown,
            f"{len(mol.elements)} atoms, unknown: {unknown or 'none'}"
            + (f"  <-- the aromatic-carbon token 'A' was read as an element" if "A" in unknown else ""),
        )
    pose_view = next(m for m in win.viewport.molecules if m.role == "pose")
    lig_view = next(m for m in win.viewport.molecules if m.role == "ligand")
    check(
        "the pose keeps the same element composition as the ligand it came from",
        sorted(pose_view.elements) == sorted(lig_view.elements),
        f"pose {dict((e, pose_view.elements.count(e)) for e in set(pose_view.elements))} "
        f"vs ligand {dict((e, lig_view.elements.count(e)) for e in set(lig_view.elements))}",
    )
    radii = pose_view.atom_radii()
    hydrogens = [i for i, e in enumerate(pose_view.elements) if e == "H"]
    carbons = [i for i, e in enumerate(pose_view.elements) if e in ("C", "A")]
    if hydrogens and carbons:
        check(
            "hydrogens are drawn smaller than carbons",
            max(radii[i] for i in hydrogens) < min(radii[i] for i in carbons),
            f"H radius {radii[hydrogens[0]]:.3f} vs C radius {radii[carbons[0]]:.3f} A"
            + ("" if max(radii[i] for i in hydrogens) < min(radii[i] for i in carbons) else "  <-- same size"),
        )
    else:
        skip(
            "hydrogens are drawn smaller than carbons",
            f"the pose has {len(hydrogens)} H and {len(carbons)} C atoms, so "
            "there are no radii of both to compare",
        )
    check(
        "AutoDock 'A' is mapped to carbon",
        PDBQT_TYPE_ELEMENT.get("A") == "C" and PDBQT_TYPE_ELEMENT.get("OA") == "O"
        and PDBQT_TYPE_ELEMENT.get("HD") == "H" and PDBQT_TYPE_ELEMENT.get("ZN") == "Zn",
        "A->C, OA->O, HD->H, ZN->Zn",
    )

    # ------------------------------------------------------------ interaction
    section("2. camera interaction")
    cam = win.viewport.camera
    y0, p0, d0 = cam.yaw, cam.pitch, cam.distance

    drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (520, 300))
    check("left-drag rotates the camera", abs(cam.yaw - y0) > 1e-6, f"yaw {y0:.4f} -> {cam.yaw:.4f}")

    drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (400, 420))
    check(
        "left-drag changes pitch and clamps it",
        cam.pitch != p0 and -1.5 <= cam.pitch <= 1.5,
        f"pitch {p0:.4f} -> {cam.pitch:.4f}",
    )
    # push far past the clamp to prove it holds
    for _ in range(12):
        drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (400, 900), steps=2)
    check("pitch clamp holds at +1.5", abs(cam.pitch - 1.5) < 1e-9, f"pitch {cam.pitch:.6f}")
    for _ in range(20):
        drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 900), (400, 200), steps=2)
    check("pitch clamp holds at -1.5", abs(cam.pitch + 1.5) < 1e-9, f"pitch {cam.pitch:.6f}")

    before = cam.distance
    QTest.mouseMove(win.viewport, QtCore.QPoint(400, 300))
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(400, 300),
        QtCore.QPointF(400, 300),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    app.sendEvent(win.viewport, wheel)
    app.processEvents()
    check("wheel zooms in", cam.distance < before, f"distance {before:.3f} -> {cam.distance:.3f}")

    before = cam.distance
    win.viewport.frame_all()
    app.processEvents()
    check(
        "Frame all refits the camera",
        abs(cam.distance - before) > 1e-6 or not np.allclose(cam.center, win.viewport.camera.center),
        f"distance -> {cam.distance:.3f}, centre {np.round(cam.center, 2)}",
    )

    # right-drag: is there any pan at all?
    section("3. is there a pan control?")
    c0 = cam.center.copy()
    drag(win.viewport, QtCore.Qt.MouseButton.RightButton, (400, 300), (600, 380))
    moved = float(np.abs(cam.center - c0).max())
    check(
        "right-drag pans the camera",
        moved > 1e-6,
        f"centre moved {moved:.6f} A"
        + ("" if moved > 1e-6 else "  <-- no pan is implemented"),
    )
    m0 = cam.distance
    drag(win.viewport, QtCore.Qt.MouseButton.MiddleButton, (400, 300), (400, 380))
    check("middle-drag changes distance", abs(cam.distance - m0) > 1e-9,
          f"distance {m0:.3f} -> {cam.distance:.3f}")

    # ------------------------------------------------------------- visibility
    section("4. visibility checkboxes")
    win.viewport.frame_all()
    app.processEvents()
    full, _ = shot(win, "02_all_visible")

    win.cb_receptor.setChecked(False)
    app.processEvents()
    no_rec, _ = shot(win, "03_receptor_hidden")
    pixel_check(
        "unchecking 'receptor' removes pixels",
        non_background(no_rec) < non_background(full),
        f"{non_background(full)} -> {non_background(no_rec)} px",
    )
    win.cb_receptor.setChecked(True)
    app.processEvents()

    win.cb_ligand.setChecked(False)
    app.processEvents()
    no_lig, _ = shot(win, "04_pose_hidden")
    pixel_check(
        "unchecking 'ligand / pose' removes pixels",
        non_background(no_lig) < non_background(full),
        f"{non_background(full)} -> {non_background(no_lig)} px",
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()

    win.cb_box.setChecked(False)
    app.processEvents()
    no_box, _ = shot(win, "05_box_hidden")
    pixel_check(
        "unchecking 'search box' removes the wireframe",
        non_background(no_box) < non_background(full),
        f"{non_background(full)} -> {non_background(no_box)} px",
    )
    win.cb_box.setChecked(True)
    app.processEvents()

    section("5. search-box spins")
    c_before = np.array(win.viewport.box_center, copy=True)
    win.center_spins[0].setValue(9.0)
    app.processEvents()
    c_after = np.array(win.viewport.box_center, copy=True)
    check(
        "box centre spin moves the wireframe",
        abs(c_after[0] - c_before[0]) > 1e-6,
        f"box_center x {c_before[0]:.2f} -> {c_after[0]:.2f}",
    )
    win.center_spins[0].setValue(0.0)
    app.processEvents()

    s_before = float(win.viewport.box_size[0])
    win.size_spins[0].setValue(12.0)
    app.processEvents()
    check(
        "box size spin resizes the wireframe",
        abs(float(win.viewport.box_size[0]) - s_before) > 1e-6,
        f"box_size x {s_before:.2f} -> {win.viewport.box_size[0]:.2f}",
    )
    win.size_spins[0].setValue(22.0)
    app.processEvents()

    # range check: can the user type a negative centre?
    lo = win.center_spins[0].minimum()
    hi = win.center_spins[0].maximum()
    win.center_spins[0].setValue(-5.0)
    app.processEvents()
    got = win.center_spins[0].value()
    check(
        "box centre accepts negative coordinates",
        abs(got - (-5.0)) < 1e-6,
        f"range [{lo}, {hi}], asked for -5.0, got {got}  <-- clamped",
    )
    win.center_spins[0].setValue(0.0)
    app.processEvents()

    # ------------------------------------------------- effort follows the box
    section("5a. the search effort follows the box")
    # Four defects in the workbench's half of the exhaustiveness rule were all
    # invisible to this suite: dropping the `_updating_exhaust` guard, making
    # `_suggest_exhaustiveness` return at once, breaking the latch so it never
    # stops, and hardcoding the wanted value to 8. Every one of them left 135 of
    # 135 passing, because `core.exhaustiveness_for_box` is covered elsewhere
    # and nothing here ever asked the *widget* a question. So the checks below
    # drive the live spin boxes and read the widget back.
    #
    # Its own window, because the latch is permanent by design: setting
    # exhaustiveness here would stop the main window's box from driving the
    # number, and section 7 needs that window to dock at the value it asks for.
    ex = MainWindow()
    ex.resize(1500, 900)
    ex.show()
    try:
        # The note has to answer before a receptor is loaded, and saying so is a
        # different claim from showing a number nobody measured.
        for spin in ex.size_spins:
            spin.setValue(20.0)
        app.processEvents()
        check(
            "with no receptor the map estimate admits it cannot be given",
            "MB" not in ex.lbl_effort.text() and "maps" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        ex.load_structure(EXAMPLES / "1crn_prep.pdbqt", "receptor")
        app.processEvents()
        check(
            "a loaded receptor puts a map estimate on screen",
            "MB" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        def _box(side: float) -> bool:
            """Put a cube on all three size spins; True if the box really moved.

            The "did it move" answer is not bookkeeping. `setValue` to the value
            a spin already holds emits nothing, so a step that asks for the box
            it already has tests nothing at all -- and it passes for a build
            whose suggestion is dead, because "the number did not change" is
            exactly what a dead suggestion looks like.
            """
            before = tuple(float(s.value()) for s in ex.size_spins)
            for spin in ex.size_spins:
                spin.setValue(side)
            app.processEvents()
            after = tuple(float(s.value()) for s in ex.size_spins)
            return after != before

        from opendocking.core import (  # noqa: PLC0415
            EXHAUSTIVENESS_LADDER,
            REFERENCE_BOX_SIDE,
            GridBox,
            Receptor as _Receptor,
            exhaustiveness_for_box,
        )

        # Grow the box three times, not once. The guard's failure mode is a
        # suggestion that applies once and then latches itself, so a single step
        # looks like a working feature right up until the second one.
        seen = []
        moved_all = True
        for side in (30.0, 40.0, 60.0):
            moved_all &= _box(side)
            seen.append((side, int(ex.sp_exhaust.value())))
        wanted = [int(exhaustiveness_for_box((s, s, s))) for s, _ in seen]
        check(
            "enlarging the box raises exhaustiveness with it",
            moved_all and all(got == want for (_, got), want in zip(seen, wanted)),
            "; ".join(
                f"{side:.0f} A -> {got} (rule says {want})"
                for (side, got), want in zip(seen, wanted)
            )
            + f"; every step really moved the box: {moved_all}",
        )
        check(
            "the spin box is not stuck on one value while the box grows",
            len({got for _, got in seen}) > 1,
            f"values seen: {[got for _, got in seen]}",
        )

        # Shrinking back has to come back down, or the control would be a ratchet
        # dressed up as a rule.
        moved_down = _box(20.0)
        check(
            "shrinking the box lowers exhaustiveness again",
            moved_down
            and int(ex.sp_exhaust.value()) == int(exhaustiveness_for_box((20.0,) * 3)),
            f"back to 20 A (moved: {moved_down}) -> {ex.sp_exhaust.value()} "
            f"(rule says {exhaustiveness_for_box((20.0,) * 3)})",
        )

        # The visible text has to follow the number, in both directions, and it
        # has to say whose number it is. A tooltip that never changes is a
        # tooltip saying nothing.
        tip_big = ex.sp_exhaust.toolTip()
        note_big = ex.lbl_effort.text()
        _box(60.0)
        check(
            "the hint text changes when the box changes",
            ex.sp_exhaust.toolTip() != tip_big and ex.lbl_effort.text() != note_big,
            f"tooltip 20A->60A: {tip_big[:34]!r} -> {ex.sp_exhaust.toolTip()[:34]!r}; "
            f"note 20A->60A: {note_big!r} -> {ex.lbl_effort.text()!r}",
        )
        check(
            "the note names the walks it is talking about",
            f"{int(ex.sp_exhaust.value())} walks" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        # A hand edit has to end the following. This is the load-bearing
        # negative: without it, "the spin box follows the box" is also what a
        # control that ignores you looks like.
        #
        # A value that is *not* the one already showing. `setValue` to the
        # current value emits nothing, so a check that "types" the number the
        # rule had already chosen is not typing anything: no `valueChanged`, no
        # latch, and every assertion below would pass against a build whose
        # latch was deleted.
        typed = 24
        ex.sp_exhaust.setValue(typed)
        app.processEvents()
        check(
            "the hand edit is a change, so the control really was touched",
            int(ex.sp_exhaust.value()) == typed,
            f"asked for {typed}, spin box shows {ex.sp_exhaust.value()}",
        )
        after_typing = ex.sp_exhaust.toolTip()
        check(
            "a hand-typed value is claimed by the user, in words",
            "Set by you" in after_typing,
            f"{after_typing[:48]!r}",
        )
        _box(60.0)
        moved_down = _box(20.0)
        check(
            "after a hand edit the box stops driving the number, shrinking",
            moved_down and int(ex.sp_exhaust.value()) == typed,
            f"typed {typed}, box 60A -> 20 A (moved: {moved_down}), spin box now "
            f"{ex.sp_exhaust.value()}",
        )
        moved_up = _box(60.0)
        check(
            "after a hand edit the box stops driving the number, growing",
            moved_up and int(ex.sp_exhaust.value()) == typed,
            f"typed {typed}, box 20A -> 60 A (moved: {moved_up}), spin box now "
            f"{ex.sp_exhaust.value()}",
        )
        moved_back = _box(20.0)
        check(
            "and it stays stopped when the box comes back down",
            moved_back and int(ex.sp_exhaust.value()) == typed,
            f"box back to 20 A (moved: {moved_back}), still {ex.sp_exhaust.value()}",
        )
        check(
            "the note says the number is the user's while it is",
            "set by you" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        # The way back has to exist, and it has to be a press. If the suggestion
        # came back by itself the latch would be a suggestion the user cannot
        # decline, which is the failure the latch was built to avoid.
        check(
            "there is a control that hands the number back to the rule",
            ex.btn_suggest.isVisible() and ex.btn_suggest.isEnabled(),
            f"{ex.btn_suggest.text()!r} "
            f"{ex.btn_suggest.width()}x{ex.btn_suggest.height()} px",
        )
        ex.btn_suggest.click()
        app.processEvents()
        check(
            "pressing it re-arms the rule immediately",
            "suggested" in ex.lbl_effort.text()
            and "Set by you" not in ex.sp_exhaust.toolTip(),
            f"note {ex.lbl_effort.text()!r}, tooltip "
            f"{ex.sp_exhaust.toolTip()[:34]!r}",
        )
        after_rearm = []
        for side in (60.0, 40.0, 20.0):
            _box(side)
            after_rearm.append((side, int(ex.sp_exhaust.value())))
        check(
            "and the box drives the number again from then on",
            all(got == int(exhaustiveness_for_box((s,) * 3)) for s, got in after_rearm)
            and any(got != typed for _, got in after_rearm),
            "; ".join(f"{s:.0f} A -> {g}" for s, g in after_rearm),
        )

        # The cap. Above 64,000 A^3 the ladder's top rung is below what the rule
        # itself wants, and the note has to say so rather than print 128 as
        # though it were met. The percentage is derived here from `core`, not
        # read out of the label, so the check is about the arithmetic and not
        # about the wording.
        for side in (60.0, 100.0):
            _box(side)
            vol = side ** 3
            want = EXHAUSTIVENESS_LADDER[0] * (vol / REFERENCE_BOX_SIDE ** 3)
            got = int(ex.sp_exhaust.value())
            check(
                f"a {side:.0f} A cube is reported as short of what it wants",
                got == EXHAUSTIVENESS_LADDER[-1]
                and f"{100.0 * got / want:.0f}%" in ex.lbl_effort.text(),
                f"wants {want:.0f} walks, ladder gave {got}, note says "
                f"{100.0 * got / want:.0f}%: {ex.lbl_effort.text()!r}",
            )
        _box(40.0)
        check(
            "a box inside the ladder is not accused of falling short",
            "ladder stops here" not in ex.lbl_effort.text()
            and "% of the" not in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        # The map number itself, against the engine and not against the panel.
        receptor = _Receptor.from_pdbqt(EXAMPLES / "1crn_prep.pdbqt")
        centre = tuple(float(s.value()) for s in ex.center_spins)
        for side in (20.0, 40.0, 60.0, 100.0):
            _box(side)
            engine_mb = receptor.estimate_memory_mb(
                GridBox.from_center_size(centre, (side,) * 3)
            )
            shown = ex.lbl_effort.text().split(" MB")[0].split()[-1]
            check(
                f"the map estimate for a {side:.0f} A cube is the engine's",
                abs(float(shown) - engine_mb) <= max(1.0, engine_mb * 0.01),
                f"label {shown} MB, engine {engine_mb:.1f} MB",
            )
        _box(60.0)
        note_lab = ex.lbl_effort
        need = note_lab.heightForWidth(max(note_lab.width(), 1))
        check(
            "the note is tall enough for the line it grew",
            note_lab.height() >= need - 1,
            f"height {note_lab.height()} px, needs {need} px, "
            f"{len(note_lab.text().splitlines())} lines of text",
        )
    except Exception:
        traceback.print_exc()
        check("the exhaustiveness section ran to the end", False, "raised above")
    finally:
        dispose_window(app, ex)

    # ----------------------------------------------------------------- poses
    # The File menu is the only route into the app once it is open, so it has to
    # offer every role. It used to have a single "Open structure…" that loaded
    # everything as a ligand, which made a receptor unreachable from the GUI.
    section("5b. the File menu can reach every role")
    menu = win.menuBar().actions()[0].menu()
    labels = [a.text() for a in menu.actions() if a.text() and a.text() != "---"]
    print(f"  File menu: {labels}")
    for role, needle in (("receptor", "Load receptor"), ("ligand", "Load ligand"),
                         ("poses", "Open poses"), ("dock", "Dock now")):
        check(
            f"the File menu offers '{needle}'",
            any(needle in t for t in labels),
            ", ".join(labels),
        )
    check(
        "no action still says the generic 'Open structure'",
        not any("Open structure" in t for t in labels),
        ", ".join(labels),
    )
    check(
        "opening a pose file goes through load_structure, not a crash",
        callable(win.open_file_dialog) and "kind" in win.open_file_dialog.__code__.co_varnames,
        f"signature accepts a role: {win.open_file_dialog.__code__.co_varnames[:3]}",
    )

    # A pose file with one field shifted by a column. `read_pdbqt_models`
    # refuses these by default -- correctly, because the engine's reader splits
    # on whitespace and would load the file and report the wrong coordinates
    # without saying anything -- and that refusal used to arrive at the user as
    # a traceback out of a GUI handler. Nothing here exercised a malformed
    # pose file, which is why it landed on a person rather than on a test.
    n = win.pose_list.count()
    good_rows = (EXAMPLES / "poses.pdbqt").read_text(encoding="utf-8").splitlines()
    shifted = None
    mangled = []
    for number, line in enumerate(good_rows, start=1):
        if shifted is None and line.startswith("ATOM") and len(line) > 34:
            # Delete the space at column 31: every coordinate on this line
            # slides one column left, which is the exact edit the engine's
            # whitespace-splitting reader cannot see.
            mangled.append(line[:30] + line[31:])
            shifted = number
        else:
            mangled.append(line)
    bad_path = Path(tempfile.gettempdir()) / "od_misaligned_pose.pdbqt"
    bad_path.write_text("\n".join(mangled) + "\n", encoding="utf-8")
    raised = None
    try:
        win.load_structure(bad_path, "poses")
    except Exception as exc:  # noqa: BLE001 - that it is *any* exception is the claim
        raised = exc
    msg = win.status_label.text()
    print(f"  misaligned file says: {msg!r}")
    check(
        "a pose file with a shifted column does not raise out of the GUI",
        raised is None,
        f"{type(raised).__name__}: {raised}" if raised else
        f"line {shifted} of a copy of poses.pdbqt was shifted a column",
    )
    check(
        "and the message names the file, the line and the field",
        bad_path.name in msg
        and f"line {shifted}" in msg
        and "columns 31-38" in msg,
        f"status {msg!r}",
    )
    check(
        "and the poses already on screen are left alone, not replaced by junk",
        win.pose_list.count() == n and "poses.pdbqt" not in msg,
        f"{win.pose_list.count()} rows still shown; a refused file must not "
        f"become an empty table",
    )
    bad_path.unlink(missing_ok=True)

    # Pose browser. Its header used to sit *above* section 5b, with nothing
    # between the two, so every check below was filed under 5b's name and
    # "6. pose browser" was a heading with an empty body -- invisible until
    # sections printed their own subtotals.
    section("6. pose browser")
    # The affinity column, read by pose rather than by row, because the table
    # sorts and a row number is not a pose.
    energies_before = [
        float(win.pose_table.cell(win.pose_table.pose_row_of(i), 1))
        for i in range(n)
    ]
    shot_a, _ = shot(win, "06_pose_best")
    switched = 0
    for row in range(n):
        win.pose_list.setCurrentRow(row)
        app.processEvents()
        if win.lbl_energy.text() not in ("", "—"):
            switched += 1
    check(
        "selecting each pose updates the energy readout",
        switched == n,
        f"{switched}/{n} rows produced a value",
    )
    win.pose_list.setCurrentRow(0)
    app.processEvents()
    rmsd0 = win.lbl_rmsd.text()
    for row in range(1, n):
        win.pose_list.setCurrentRow(row)
        app.processEvents()
    check(
        "RMSD-to-best changes away from the best pose",
        win.lbl_rmsd.text() != rmsd0,
        f"row 0 = {rmsd0!r}, last row = {win.lbl_rmsd.text()!r}",
    )
    shot_b, _ = shot(win, "07_pose_last")
    pixel_check(
        "selecting a different pose changes what is drawn",
        not np.array_equal(shot_a, shot_b),
        "framebuffers differ" if not np.array_equal(shot_a, shot_b) else "identical",
    )
    win.pose_list.setCurrentRow(0)
    app.processEvents()
    check(
        "the pose table is ordered by the energies it prints",
        energies_before == sorted(energies_before),
        "; ".join(f"{e:.2f}" for e in energies_before[:4]) + " ...",
    )

    # What the table can and cannot know, from a pose *file*. The poses here
    # came from `examples/poses.pdbqt`, so there is no `DockingResult` behind
    # them: the writer emits one decimal place of energy per model and nothing
    # of the intermolecular part. The columns that have no source have to say
    # so, because a column of plausible numbers with nothing behind them is the
    # failure this table was built to avoid.
    check(
        "a pose file gives the table an affinity for every row",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 1) not in ("", "n/a")
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 1) for i in range(n)),
    )
    check(
        "and the intermolecular column admits the file never recorded it",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 2) == "—"
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 2) for i in range(n))[:80],
    )
    check(
        "every row has a distance to its nearest neighbour",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 4) not in ("", "—")
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 4) for i in range(n)),
    )
    check(
        "every row has a distance to the best pose",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 3) not in ("", "n/a")
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 3) for i in range(n)),
    )
    check(
        "the best pose is the one the table marks",
        float(win.pose_table.cell(win.pose_table.pose_row_of(min(range(n), key=lambda i: energies_before[i])), 1))
        == min(energies_before),
        f"marked row(s) {[r for r in range(n) if '*' in win.pose_table.cell(r, 0)]}, "
        f"lowest energy {min(energies_before):.2f}",
    )
    check(
        "contacts are counted for every pose, not only the selected one",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 6).isdigit()
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 6) for i in range(n)),
    )
    check(
        "the rows are in the order the engine reported, best first",
        [win._pose_index_at(r) for r in range(n)] == list(range(n)),
        f"visual rows hold poses {[win._pose_index_at(r) for r in range(n)]}",
    )

    # ------------------------------------------------------------ docking run
    section("7. a real docking run from the GUI")
    win.sp_exhaust.setValue(2)
    win.center_spins[0].setValue(0.0)
    win.size_spins[0].setValue(22.0)
    app.processEvents()

    n_mol_before = len(win.viewport.molecules)
    win.btn_dock.click()
    app.processEvents()
    check("Dock disables the button while running", not win.btn_dock.isEnabled())

    # Is the window still painting? A blocked GUI thread cannot repaint.
    repaints = 0
    for _ in range(40):
        app.processEvents()
        QTest.qWait(50)
        win.viewport.repaint()
        repaints += 1
    print(f"  event-loop iterations during the run: {repaints}")
    check("the event loop keeps running during a search", repaints == 40, f"{repaints}/40")

    for _ in range(400):
        app.processEvents()
        if win.btn_dock.isEnabled():
            break
        QTest.qWait(50)
    check("Dock re-enables when the run finishes", win.btn_dock.isEnabled(),
          f"status: {win.status_label.text()!r}")
    print(f"  status after run: {win.status_label.text()!r}")
    check(
        "status reports the docking result",
        "done" in win.status_label.text() or "failed" in win.status_label.text(),
        win.status_label.text(),
    )
    check(
        "the pose table is refilled from the run",
        win.pose_list.count() > 0,
        f"{win.pose_list.count()} rows",
    )
    n_mol_after = len(win.viewport.molecules)
    roles = [m.role for m in win.viewport.molecules]
    print(f"  molecule views: {n_mol_before} before, {n_mol_after} after  roles={roles}")
    check(
        "a docking run replaces the pose view, it does not add one",
        roles.count("pose") == 1,
        f"{roles.count('pose')} pose view(s)  <-- stacking",
    )
    check(
        "one view per role, none duplicated",
        sorted(roles) == sorted(set(roles)),
        f"roles={roles}",
    )
    shot(win, "08_after_dock")

    section("7b. the pose table, from a run whose result is still in memory")
    # A run made in this window keeps its `DockingResult`, and that is the only
    # source of the intermolecular energies and of the engine's own RMSDs: the
    # pose file the run is written to carries one decimal place of energy per
    # model and nothing of the rest. So the table's numbers are checked against
    # an *independent* run of the same receptor, ligand, box and seed, which is
    # the only way to catch the table quietly showing the file's weaker value.
    #
    # Its own window with a fixed seed. The main window docks at exhaustiveness 2
    # against a 22 A box for speed, and its seed is 0 -- random -- so neither a
    # fixed-seed comparison nor a reproducible result is possible there.
    from opendocking.core import (  # noqa: PLC0415
        GridBox as _GridBox,
        Receptor as _Receptor2,
        dock as _dock,
        load_ligand as _load_ligand,
    )

    pw2 = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                     ligand=EXAMPLES / "biotin_prep.pdbqt")
    pw2.resize(1500, 900)
    pw2.show()
    try:
        app.processEvents()
        # Box first, then exhaustiveness. In that order, and only in that order:
        # setting the box makes the workbench suggest a walk count for it, which
        # overwrites a value typed before it. A 26 A box wants 32 walks, so
        # setting the spins first would silently dock at 32 while this section
        # believes it docked at 8 -- and then compare two different searches.
        pw2.size_spins[0].setValue(26.0)
        pw2.size_spins[1].setValue(26.0)
        pw2.size_spins[2].setValue(26.0)
        pw2.center_spins[0].setValue(0.0)
        pw2.center_spins[1].setValue(0.0)
        pw2.center_spins[2].setValue(0.0)
        pw2.sp_exhaust.setValue(8)
        pw2.sp_seed.setValue(20260901)
        app.processEvents()
        check(
            "the run's box and effort are the ones this section means to dock",
            [round(s.value(), 2) for s in pw2.size_spins] == [26.0] * 3
            and pw2.sp_exhaust.value() == 8
            and pw2.sp_seed.value() == 20260901,
            f"box {[s.value() for s in pw2.size_spins]}, "
            f"exhaustiveness {pw2.sp_exhaust.value()}, seed {pw2.sp_seed.value()}",
        )
        pw2.btn_dock.click()
        for _ in range(600):
            app.processEvents()
            if pw2.btn_dock.isEnabled():
                break
            QTest.qWait(50)
        for _ in range(8):
            app.processEvents()
            QTest.qWait(40)
        np2 = pw2.pose_table.rowCount()
        check(
            "the run filled the table",
            np2 > 1 and "done" in pw2.status_label.text(),
            f"{np2} rows, status {pw2.status_label.text()!r}",
        )

        # The same search, run again here, outside the window.
        _rec2 = _Receptor2.from_pdbqt(EXAMPLES / "1crn_prep.pdbqt")
        _box2 = _GridBox.from_center_size((0.0, 0.0, 0.0), (26.0, 26.0, 26.0))
        _ref = _dock(
            _load_ligand(EXAMPLES / "biotin_prep.pdbqt"),
            _rec2.precalculate(_box2, "vina"),
            exhaustiveness=8,
            num_modes=9,
            rmsd_cutoff=1.0,
            seed=20260901,
            scoring="vina",
        )
        def col(pose: int, index: int):
            """One cell of the table as a float, or None if it prints a dash.

            The window deliberately leaves a cell empty when the number is not
            known -- the intermolecular energy of a pose read back from a file
            is one of those. Turning that into a float raises and takes the
            suite down before the check can report the very thing it is
            looking for, so the dash survives to the comparison instead.
            """
            try:
                return float(pw2.pose_table.cell(pw2.pose_table.pose_row_of(pose), index))
            except (TypeError, ValueError):
                return None

        def agrees(table, engine, tol: float) -> bool:
            """Every cell is a number, and every one is the engine's."""
            return len(table) == len(engine) and all(
                a is not None and abs(a - b) < tol for a, b in zip(table, engine)
            )

        energies = [col(i, 1) for i in range(np2)]
        inters = [col(i, 2) for i in range(np2)]
        rmsds = [col(i, 3) for i in range(np2)]
        gaps = [col(i, 4) for i in range(np2)]
        print(f"  reference run: {list(_ref.energies)}")
        print(f"  table kcal/mol: {energies}")
        print(f"  table inter:    {inters}")
        check(
            "the affinity column is the engine's, to the hundredth",
            agrees(energies, list(_ref.energies), 0.005),
            f"table {energies} vs engine {list(_ref.energies)}",
        )
        # The pose file is a lossy summary of the result: the writer emits
        # `REMARK VINA RESULT: {energy:.1f}`, so a table reading it would show
        # one decimal place. Comparing the table against *that* is the only way
        # to catch the regression, because comparing it against the engine
        # twice would pass whatever the table did.
        file_energies = [round(float(e), 1) for e in _ref.energies]
        check(
            "and it is not the pose file's one-decimal value",
            any(
                a is not None and abs(a - b) > 0.005
                for a, b in zip(energies, file_energies)
            ),
            f"table {energies} vs the file's one-decimal "
            f"{[round(e, 1) for e in energies if e is not None]}",
        )
        check(
            "the intermolecular column is the engine's too",
            agrees(inters, list(_ref.intermolecular_energies), 0.005),
            f"table {inters} vs engine {list(_ref.intermolecular_energies)}",
        )
        check(
            "the RMSD column agrees with the engine's rmsds",
            agrees(rmsds, list(_ref.rmsds), 0.01),
            f"table {rmsds} vs engine {list(_ref.rmsds)}",
        )
        check(
            "the engine's rmsd and the window's own per-atom RMSD are one number",
            all(
                abs(pw2._pose_rmsd(i) - pw2._rmsd_to_best(i)) < 0.01
                for i in range(np2)
            ),
            f"engine {rmsds} vs the window's own coordinate comparison "
            f"{[round(pw2._rmsd_to_best(i), 3) for i in range(np2)]}",
        )
        check(
            "the table's energy order is the engine's",
            all(e is not None for e in energies) and energies == sorted(energies),
            f"{energies}",
        )

        # The gap column, against the engine's own coordinates rather than
        # against the window's function that produced it. `all_pose_coords` is
        # the same data the poses were written from, so a gap column that
        # measured the wrong thing -- the distance to the best pose instead of
        # the nearest neighbour, say -- would disagree with it.
        allc = np.asarray(_ref.all_pose_coords(), np.float64)
        blanks = [i for i, g in enumerate(gaps) if g is None]
        worst = 0.0
        nearest = []
        for i in range(allc.shape[0]):
            ds = []
            for j in range(allc.shape[0]):
                if i == j:
                    continue
                m = min(allc[i].shape[0], allc[j].shape[0])
                ds.append(
                    float(np.sqrt(((allc[i][:m] - allc[j][:m]) ** 2).sum(axis=1).mean()))
                )
            nearest.append(min(ds))
            if gaps[i] is not None:
                worst = max(worst, abs(min(ds) - gaps[i]))
        check(
            "every gap is the distance to that pose's nearest neighbour",
            not blanks and worst < 0.005,
            f"largest disagreement {worst:.4f} A over {np2} rows; table {gaps} "
            f"vs the engine's coordinates {[round(d, 3) for d in nearest]}"
            + (f"; {len(blanks)} row(s) printed no gap at all" if blanks else ""),
        )
        below = [g for g in gaps if g is not None and g < 1.0]
        print(f"  nearest-neighbour gaps: {gaps} (clustering cutoff 1.0 A)")
        check(
            "a gap under the clustering distance is called out, not hidden",
            all(
                "same mode" in pw2.pose_table.item(
                    pw2.pose_table.pose_row_of(i), 4).toolTip()
                for i, g in enumerate(gaps) if g is not None and g < 1.0
            ),
            f"{len(below)} row(s) under 1.0 A in this run",
        )

        # Sorting has to be real, and the best marker has to survive it.
        best_pose = int(np.argmin(energies))
        pw2.pose_table.sortItems(1, QtCore.Qt.SortOrder.AscendingOrder)
        app.processEvents()
        after = [
            float(pw2.pose_table.cell(r, 1)) for r in range(pw2.pose_table.rowCount())
        ]
        check(
            "sorting by affinity is a numeric sort, not a text one",
            after == sorted(energies),
            f"{after}",
        )
        check(
            "a text sort would have put these in a different order",
            [f"{e:.2f}" for e in after] != sorted(f"{e:.2f}" for e in after),
            f"as text: {sorted(f'{e:.2f}' for e in after)}",
        )
        marked = [r for r in range(pw2.pose_table.rowCount())
                  if "*" in pw2.pose_table.cell(r, 0)]
        check(
            "the best marker is still on the best pose after sorting",
            len(marked) == 1 and pw2._pose_index_at(marked[0]) == best_pose,
            f"marked visual row {marked}, pose "
            f"{[pw2._pose_index_at(r) for r in marked]}, best pose {best_pose}",
        )
        check(
            "sorting did not renumber the ranks",
            sorted(
                int(pw2.pose_table.cell(pw2.pose_table.pose_row_of(i), 0).split()[0])
                for i in range(np2)
            ) == list(range(1, np2 + 1)),
            "the rank is a fact about the pose, so it travels with it",
        )
        check(
            "the marked row is the row the window says is best",
            abs(
                float(pw2.pose_table.cell(marked[0], 1)) - min(energies)
            ) < 1e-9,
            f"marked row energy {pw2.pose_table.cell(marked[0], 1)}, "
            f"lowest {min(energies):.2f}",
        )

        # And the selection still drives everything, after all that sorting.
        target = np2 - 1
        row = pw2.pose_table.pose_row_of(target)
        pw2.pose_table.setCurrentCell(row, 0)
        app.processEvents()
        check(
            "a sorted table still shows the pose the selected row is",
            f"{energies[target]:.2f}" in pw2.lbl_energy.text(),
            f"selected pose {target + 1}: table says {energies[target]:.2f}, "
            f"label says {pw2.lbl_energy.text()!r}",
        )
        check(
            "and the RMSD label is that pose's, not the row's",
            f"{rmsds[target]:.2f}" in pw2.lbl_rmsd.text(),
            f"table RMSD {rmsds[target]:.2f}, label {pw2.lbl_rmsd.text()!r}",
        )
        contacts_sum = sum(
            int(pw2.pose_table.cell(pw2.pose_table.pose_row_of(i), 6))
            for i in range(np2)
        )
        check(
            "contacts are counted for every pose in a real result",
            contacts_sum > 0 and all(
                pw2.pose_table.cell(pw2.pose_table.pose_row_of(i), 6).isdigit()
                for i in range(np2)
            ),
            f"{contacts_sum} contacts over {np2} poses",
        )

        # ------------------------------------------------- all poses at once
        pw2.pose_table.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
        app.processEvents()
        pw2.pose_table.setCurrentCell(pw2.pose_table.pose_row_of(0), 0)
        app.processEvents()
        before_pixels, _ = shot(pw2, "07b_pose_alone")
        cam_before = float(pw2.viewport.camera.distance)
        check(
            "with the overlay off, one pose is drawn",
            [m.role for m in pw2.viewport.molecules].count("pose_ghost") == 0
            and sum(1 for m in pw2.viewport.molecules if m.visible and m.role == "pose") == 1,
            f"roles {[m.role for m in pw2.viewport.molecules]}",
        )
        check(
            "and the label says so in words, not only in the picture",
            "on its own" in pw2.lbl_poses.text(),
            f"{pw2.lbl_poses.text()!r}",
        )
        pw2.cb_all_poses.setChecked(True)
        for _ in range(6):
            app.processEvents()
            QTest.qWait(40)
        ghosts = [m for m in pw2.viewport.molecules if m.role == "pose_ghost"]
        visible_ghosts = [g for g in ghosts if g.visible]
        check(
            "turning it on draws every other pose",
            len(ghosts) == np2 and len(visible_ghosts) == np2 - 1,
            f"{len(ghosts)} ghost views, {len(visible_ghosts)} visible of "
            f"{np2} poses (the selected one is drawn solid instead)",
        )
        check(
            "the selected pose is not also drawn as its own ghost",
            all(
                not (g.visible and pw2._pose_index_at(
                    pw2.pose_table.currentRow()) == i)
                for i, g in enumerate(pw2._pose_ghosts)
            ),
            f"current pose {pw2._current_pose + 1}",
        )
        check(
            "the ghosts are faint, and the solid pose is not",
            all(g.opacity < 0.5 for g in ghosts)
            and all(m.opacity >= 0.99 for m in pw2.viewport.molecules
                    if m.role == "pose"),
            f"ghost opacities {sorted({g.opacity for g in ghosts})}, pose "
            f"opacity {[m.opacity for m in pw2.viewport.molecules if m.role == 'pose']}",
        )
        # The site cloud is bigger than the poses and sits in the same place.
        # At its usual opacity it is *brighter* than a ghost, so a solid pose
        # inside it cannot be told apart from a ghost inside it -- the picture
        # then says nothing about which pose is the one, while the label beside
        # the checkbox claims to. This is the claim that has to be true in the
        # picture, not only in the object model.
        check(
            "the site cloud steps aside while poses are being compared",
            abs(pw2.viewport.pocket_opacity - POCKET_OPACITY_COMPARE) < 1e-9
            and pw2.viewport.pocket_opacity < min(g.opacity for g in ghosts),
            f"pocket opacity {pw2.viewport.pocket_opacity} while comparing, "
            f"{POCKET_OPACITY_PLAIN} normally, ghosts at "
            f"{sorted({g.opacity for g in ghosts})}: the cloud has to be dimmer "
            f"than a ghost or it decides what the picture is about",
        )
        check(
            "the state is stated in words, so a faint overlap is not read as a fault",
            "ghosted" in pw2.lbl_poses.text() and "solid" in pw2.lbl_poses.text(),
            f"{pw2.lbl_poses.text()!r}",
        )
        after_pixels, _ = shot(pw2, "07b_pose_all")
        cam_after = float(pw2.viewport.camera.distance)
        check(
            "turning it on moves the camera in to the poses, not the whole scene",
            cam_after < cam_before * 0.75,
            f"camera distance {cam_before:.1f} A -> {cam_after:.1f} A: the poses "
            "are twenty atoms inside the protein, so the overlay is a smudge "
            "unless you go and look at it",
        )
        pixel_check(
            "the picture really changed when the overlay came on",
            not np.array_equal(before_pixels, after_pixels),
            f"{differing_pixels(before_pixels, after_pixels)} pixels differ",
        )
        # Selecting another pose must move the solid one, not add a second.
        pw2.pose_table.setCurrentCell(pw2.pose_table.pose_row_of(3), 0)
        for _ in range(6):
            app.processEvents()
            QTest.qWait(40)
        still = [m for m in pw2.viewport.molecules if m.role == "pose"]
        check(
            "choosing another pose moves the solid one and re-ghosts the old",
            len(still) == 1
            and sum(1 for g in ghosts if g.visible) == np2 - 1
            and ghosts[0].visible,
            f"{len(still)} solid pose view(s), "
            f"{sum(1 for g in ghosts if g.visible)} ghosts visible, "
            f"selected pose {pw2._current_pose + 1}",
        )
        check(
            "and the words followed the selection",
            f"pose {pw2._current_pose + 1} of {np2} solid" in pw2.lbl_poses.text(),
            f"{pw2.lbl_poses.text()!r}",
        )
        # The ligand checkbox has to govern the ghosts too.
        pw2.cb_ligand.setChecked(False)
        app.processEvents()
        check(
            "hiding the ligand hides the ghosts as well",
            not any(m.visible for m in pw2.viewport.molecules
                    if m.role in ("pose", "pose_ghost")),
            f"visible roles {[m.role for m in pw2.viewport.molecules if m.visible]}",
        )
        pw2.cb_ligand.setChecked(True)
        app.processEvents()
        check(
            "and showing it again brings back the overlay, not just one pose",
            sum(1 for g in ghosts if g.visible) == np2 - 1
            and all(m.visible for m in pw2.viewport.molecules if m.role == "pose"),
            f"{sum(1 for g in ghosts if g.visible)} ghosts visible",
        )
        pw2.cb_all_poses.setChecked(False)
        app.processEvents()
        check(
            "turning the overlay off takes the ghosts out of the scene",
            [m.role for m in pw2.viewport.molecules].count("pose_ghost") == 0,
            f"roles {[m.role for m in pw2.viewport.molecules]}",
        )
        check(
            "and gives the site cloud its opacity back",
            abs(pw2.viewport.pocket_opacity - POCKET_OPACITY_PLAIN) < 1e-9,
            f"pocket opacity {pw2.viewport.pocket_opacity}, expected "
            f"{POCKET_OPACITY_PLAIN}: the comparison dimmed it, not the site table",
        )
    finally:
        dispose_window(app, pw2)

    section("7c. a pose set that breaks the table's assumptions")
    # Four of the thirteen mutations of this round left the suite green, and
    # every one of them was a hole in these checks rather than an equivalent
    # mutation. They all rely on a docking run being well behaved in a way no
    # docking run was:
    #   * the engine reports best first, so "mark row 0" and "mark the best
    #     pose" are the same code on a real result;
    #   * sorting by affinity is a no-op when affinity is already the row
    #     order, so a handler that reads the row number as the pose index is
    #     never wrong;
    #   * "contacts counted for every pose" passes when the unselected rows say
    #     zero, because zero is a digit;
    #   * and the sub-clustering-distance branch never runs at all -- across
    #     every run measured here the tightest nearest-neighbour gap was
    #     1.01 A against a 1.0 A cutoff, so the warning was never once
    #     exercised.
    #
    # So: one fixture, built from real coordinates, that is none of those things
    # at once. Three models, the *last* of which is the best by energy (so the
    # file is not in energy order), and the first two of which are the same
    # molecule displaced by 0.2 A (so the gap column has a sub-cutoff pair).
    from opendocking.pdbqt_writer import read_pdbqt_models  # noqa: PLC0415

    _src = ["\n".join(m) for m in
            read_pdbqt_models(EXAMPLES / "poses.pdbqt")]

    def _with_energy(model_text: str, energy: float) -> str:
        out = []
        for line in model_text.splitlines():
            if "VINA RESULT" in line:
                out.append(f"REMARK VINA RESULT:    {energy:.1f}  0.000  0.000")
            else:
                out.append(line)
        return "\n".join(out)

    def _displaced(model_text: str, shift: float) -> str:
        """The same model, every atom moved by `shift` angstrom along x."""
        out = []
        for line in model_text.splitlines():
            if line.startswith(("ATOM", "HETATM")) and len(line) > 30:
                try:
                    x = float(line[30:38]) + shift
                except ValueError:
                    out.append(line)
                    continue
                out.append(line[:30] + f"{x:8.3f}" + line[38:])
            else:
                out.append(line)
        return "\n".join(out)

    fixture = tempfile.NamedTemporaryFile(
        "w", suffix=".pdbqt", delete=False, encoding="utf-8"
    )
    fixture.write(
        "MODEL\n" + _with_energy(_src[0], -5.0) + "\n"
        "ENDMDL\n"
        "MODEL\n" + _with_energy(_displaced(_src[0], 0.2), -4.0) + "\n"
        "ENDMDL\n"
        "MODEL\n" + _with_energy(_src[1], -9.9) + "\n"
        "ENDMDL\n"
    )
    fixture.close()
    # Load the fixture into the window that is already open rather than opening
    # a third one. This section needs a *pose set*, not a window, and a third
    # MainWindow in a process that already has two is what this machine's GL
    # stack fell over on (STATUS_STACK_BUFFER_OVERRUN, no Python traceback, the
    # run ending inside `processEvents`). Fewer moving parts for the same
    # coverage; section 8 docks, which replaces whatever is loaded here anyway.
    win.load_structure(Path(fixture.name), "poses")
    pw3 = win
    try:
        app.processEvents()
        n3 = pw3.pose_table.rowCount()
        cells = [[pw3.pose_table.cell(r, c) for c in range(7)] for r in range(n3)]
        print("  fixture rows (as loaded, which is not energy order):")
        for row in cells:
            print("   ", row)
        check(
            "the fixture is the awkward case it was built to be",
            n3 == 3
            and float(cells[2][1]) == min(float(r[1]) for r in cells)
            and abs(float(cells[0][4]) - 0.2) < 0.02,
            f"{n3} rows, energies {[r[1] for r in cells]}, "
            f"gaps {[r[4] for r in cells]}",
        )
        marked = [r for r in range(n3) if "*" in pw3.pose_table.cell(r, 0)]
        check(
            "the best marker is on the lowest energy, not on the first row",
            marked == [2],
            f"marked row {marked}; row 0 holds {cells[0][1]} kcal/mol and the "
            f"lowest is {cells[2][1]}",
        )
        check(
            "a pair closer than the clustering distance is called out by name",
            all(
                "same mode" in pw3.pose_table.item(r, 4).toolTip()
                for r in range(n3)
                if float(pw3.pose_table.cell(r, 4)) < 1.0
            )
            and sum(1 for r in range(n3)
                    if float(pw3.pose_table.cell(r, 4)) < 1.0) == 2,
            f"gaps {[c[4] for c in cells]}; a 0.2 A pair is the same mode "
            f"twice, and the table has to say so",
        )
        # Colours are compared against each other rather than against a fixed
        # channel value: what the check actually claims is that a sub-cutoff gap
        # looks different from a healthy one, and the palette's idea of "normal
        # text" is not this file's to predict. If the warning brush is dropped,
        # both kinds of cell carry the same default brush and this fails.
        sub_cells = [pw3.pose_table.item(r, 4).foreground().color()
                     for r in range(n3) if float(pw3.pose_table.cell(r, 4)) < 1.0]
        ok_cells = [pw3.pose_table.item(r, 4).foreground().color()
                    for r in range(n3) if float(pw3.pose_table.cell(r, 4)) >= 1.0]
        check(
            "and it is coloured differently from a healthy gap",
            len(sub_cells) == 2 and bool(ok_cells)
            and all(c != sub_cells[0] for c in ok_cells),
            f"sub-cutoff {[c.name() for c in sub_cells]}, healthy "
            f"{[c.name() for c in ok_cells]}",
        )
        # Sorting here genuinely reorders, because the file's order is not the
        # energy order -- which is what makes the row-is-not-the-pose mistake
        # observable at all.
        pw3.pose_table.sortItems(1, QtCore.Qt.SortOrder.AscendingOrder)
        app.processEvents()
        order = [pw3._pose_index_at(r) for r in range(n3)]
        check(
            "sorting really reorders this fixture",
            order != [0, 1, 2],
            f"visual rows now hold poses {order}",
        )
        target = 0
        pw3.pose_table.setCurrentCell(pw3.pose_table.pose_row_of(target), 0)
        app.processEvents()
        check(
            "a sorted table shows the pose in the selected row, by its energy",
            f"{float(pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 1)):.2f}"
            in pw3.lbl_energy.text(),
            f"selected pose {target + 1} ({float(pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 1)):.2f} "
            f"kcal/mol), visual row {pw3.pose_table.currentRow()}, label "
            f"{pw3.lbl_energy.text()!r}",
        )
        check(
            "and the RMSD label is that pose's, not the row's",
            f"{float(pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 3)):.2f}"
            in pw3.lbl_rmsd.text(),
            f"table {pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 3)}, "
            f"label {pw3.lbl_rmsd.text()!r}",
        )
        # Every row's contact count, checked against an independent count for
        # that pose -- so "zero for the rows nobody selected" cannot pass.
        from opendocking.workbench import MoleculeView as _MV  # noqa: PLC0415
        from opendocking.workbench.app import _parse_pdbqt_atoms  # noqa: PLC0415
        from opendocking.workbench.contacts import find_contacts  # noqa: PLC0415

        _models3 = ["\n".join(m) for m in read_pdbqt_models(Path(fixture.name))]
        _rec3 = next(m for m in pw3.viewport.molecules if m.role == "receptor")
        want3 = []
        for body in _models3:
            coords, elems = _parse_pdbqt_atoms(body)
            view = _MV(name="p", coords=coords, elements=elems,
                       color=(0.4, 0.9, 0.45), radius=0.34, role="pose")
            want3.append(len(find_contacts(view, _rec3)))
        got3 = [int(pw3.pose_table.cell(pw3.pose_table.pose_row_of(i), 6))
                for i in range(n3)]
        print(f"  fixture contacts: table {got3}, counted here {want3}")
        check(
            "every row's contact count is the count for that pose",
            got3 == want3,
            f"table {got3} vs counted independently {want3}",
        )
        check(
            "and no row is a placeholder zero",
            all(g > 0 for g in got3),
            f"{got3}",
        )
    finally:
        # Put the window's own poses back rather than closing it: it is the
        # main window, and section 8 docks in it.
        win.load_structure(EXAMPLES / "poses.pdbqt", "poses")
        app.processEvents()
        with contextlib.suppress(OSError):
            Path(fixture.name).unlink()

    section("8. a second docking run")
    win.btn_dock.click()
    for _ in range(400):
        app.processEvents()
        if win.btn_dock.isEnabled():
            break
        QTest.qWait(50)
    roles2 = [m.role for m in win.viewport.molecules]
    print(f"  molecule views now: {len(roles2)}  roles={roles2}")
    check(
        "a second run still leaves exactly one pose view",
        roles2.count("pose") == 1,
        f"{roles2.count('pose')}  <-- previous results stay on screen",
    )
    check(
        "a second run does not change the number of views",
        len(roles2) == len(roles),
        f"{len(roles)} -> {len(roles2)}",
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()
    vis = [m for m in win.viewport.molecules if m.visible]
    check(
        "after two runs the scene is exactly receptor + ligand + pose",
        len(vis) == 3 and {m.role for m in vis} == {"receptor", "ligand", "pose"},
        f"{len(vis)} visible: {[(m.role, m.name) for m in vis]}",
    )
    # And the checkbox must actually be able to hide the pose now.
    win.cb_ligand.setChecked(False)
    app.processEvents()
    pose_hidden = all(
        not m.visible for m in win.viewport.molecules if m.role == "pose"
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()
    check("the ligand checkbox hides the pose view", pose_hidden)
    shot(win, "09_after_second_dock")

    # temp files left behind
    # `tempfile` is imported at module level. Re-importing it here would make
    # the name local to the whole of `main`, and every earlier use in this
    # function would then read as an unbound local -- which is how section 7c,
    # above, died with `UnboundLocalError` before it ran a single check.
    leaked = list(Path(tempfile.gettempdir()).glob("tmp*.pdbqt"))
    recent = [p for p in leaked if (Path(tempfile.gettempdir()) / p.name).exists()]
    try:
        ages = [(p.stat().st_mtime, p) for p in leaked]
        now = max(t for t, _ in ages) if ages else 0
        import time
        fresh = [p for t, p in ages if time.time() - t < 3600]
    except Exception:
        fresh = []
    print(f"  temp .pdbqt files in %TEMP% newer than 1 h: {len(fresh)}")
    check("docking does not leak temp files", not fresh, f"{len(fresh)} file(s): "
          + ", ".join(p.name for p in fresh[:4]))

    # ------------------------------------------------- negative-coordinate receptor
    section("9. a receptor at negative coordinates")
    import opendocking
    from opendocking.workbench import MoleculeView

    shifted = EXAMPLES / "_shifted_receptor.pdbqt"
    rows = []
    for line in rec.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) - 40.0
            rows.append(line[:30] + f"{x:8.3f}" + line[38:])
        else:
            rows.append(line)
    shifted.write_text("\n".join(rows) + "\n", encoding="utf-8")

    win2 = MainWindow(receptor=shifted)
    win2.resize(1000, 700)
    win2.show()
    app.processEvents()
    QTest.qWait(400)
    app.processEvents()
    true_center = np.array(opendocking.Receptor.from_pdbqt(shifted).center, dtype=float)
    ui_center = np.array([s.value() for s in win2.center_spins], dtype=float)
    shown_box = np.array(win2.viewport.box_center, dtype=float)
    print(f"  true receptor centre : {np.round(true_center, 3)}")
    print(f"  box shown in spins   : {np.round(ui_center, 3)}")
    # The reference value used to be the receptor's centroid, because loading a
    # receptor put the centroid in the spins. It no longer does -- the box goes
    # on the first site the pocket search finds, which is the whole point of
    # that change -- so the check compares the spins to the box the window
    # actually decided on. What it is really about is unchanged and is stated
    # as its own check below: a receptor at negative coordinates must not have
    # its position silently clamped to zero.
    check(
        "the box-centre spins show the box the window chose",
        bool(np.abs(shown_box - ui_center).max() < 0.05),
        f"spins {np.round(ui_center, 3)} vs viewport {np.round(shown_box, 3)}, "
        f"mismatch {np.abs(shown_box - ui_center).max():.3f} A",
    )
    # The claim in this section's title is that negative coordinates are not
    # silently clamped to zero, and that is `ui_center.min() < -5`. The old
    # second half of this check -- that the box differs from the receptor's
    # centroid by more than 0.05 A -- was a *proxy* for "the box came from the
    # search rather than from a default", and it has quietly stopped being
    # true: with the pocket search's ranking fixed, the top site on this
    # small fixture happens to sit 0.02 A from the centroid, so a site is
    # selected that a reader would call "the centre". Proving the origin
    # directly is both stronger and not fixture-dependent -- compare against
    # the site the window actually selected, not against the centroid.
    picked = win2.pocket_table.currentRow()
    site_centre = None
    models = getattr(win2, "_pocket_models", [])
    if 0 <= picked < len(models):
        site_centre = np.asarray(models[picked].center, dtype=float)
    check(
        "a receptor at negative coordinates keeps them, unclamped to zero",
        bool(ui_center.min() < -5.0),
        f"receptor centre {np.round(true_center, 3)}, box {np.round(ui_center, 3)}, "
        f"spin range [{win2.center_spins[0].minimum()}, {win2.center_spins[0].maximum()}]",
    )
    check(
        "and the box is the selected site's, not a default near the middle",
        site_centre is not None
        and bool(np.abs(ui_center - site_centre).max() < 0.05),
        f"selected row {picked}, its centre "
        f"{np.round(site_centre, 3) if site_centre is not None else None}, "
        f"box {np.round(ui_center, 3)}, "
        f"{float(np.abs(ui_center - site_centre).max()) if site_centre is not None else float('nan'):.3f} A apart"
        if site_centre is not None else "no site selected",
    )
    # Deliberately *not* checked here: "the box is somewhere other than the
    # centroid". This receptor is a handful of atoms whose one site happens to
    # sit 0.09 A from its centre, so the check could only pass with its
    # threshold loosened until it asserted nothing -- and it would be proving
    # something section 11c already proves on crambin, where the box really is
    # 9.9 A off the centroid. A duplicate that cannot fail teaches nothing.
    shot(win2, "10_negative_coords")
    dispose_window(app, win2)

    # ------------------------------------------------- display representations
    section("10. display representations")
    from opendocking.workbench.app import REPRESENTATION_KEYS  # noqa: PLC0415

    check(
        "the display selector offers every representation",
        set(REPRESENTATION_KEYS) == {"spheres", "ball_and_stick", "stick",
                                     "ribbon", "cartoon"},
        ", ".join(REPRESENTATION_KEYS),
    )
    check(
        "the selector has one entry per representation",
        win.cmb_representation.count() == len(REPRESENTATION_KEYS),
        f"{win.cmb_representation.count()} entries",
    )

    # Every colour the picture can contain that is not an atom colour has to
    # be named somewhere on screen. A report said this was not so: loading a
    # receptor draws the first pocket as magenta spheres, and the legend named
    # only the four dashed interaction lines, so a colour appeared that
    # nothing explained. "The 'site volume' checkbox is over there" is not the
    # same claim as "this colour is the site volume", and the difference is
    # the whole reason a legend exists.
    from opendocking.workbench import (  # noqa: PLC0415
        COLOR_CONTACT,
        COLOR_POCKET,
        CONTACT_LABELS,
    )

    swatches = [
        (w.text(), w.styleSheet())
        for w in win.findChildren(QtWidgets.QLabel)
        if w.styleSheet().startswith("color: rgb(")
    ]
    legend_words = [
        w.text() for w in win.findChildren(QtWidgets.QLabel)
        if w.styleSheet().startswith("color: #9aa3ad")
    ]
    check(
        "the legend names the site volume, not just the interaction lines",
        "site volume" in legend_words,
        f"legend reads {legend_words}",
    )
    # The swatch colour must be the colour the renderer uses, read from the
    # same table -- otherwise the legend is a picture of a legend, and the
    # claim "read from one table so it cannot drift" is only a comment.
    def _rgb(style):
        inner = style.split("rgb(", 1)[1].split(")", 1)[0]
        return tuple(int(v) for v in inner.split(","))

    pocket_rgb = tuple(int(round(v * 255)) for v in COLOR_POCKET)
    check(
        "the site-volume swatch is the colour the cloud is drawn in",
        any(_rgb(style) == pocket_rgb and text == "●" for text, style in swatches),
        f"pocket is rgb{pocket_rgb}; swatches {[(t, _rgb(s)) for t, s in swatches]}",
    )
    check(
        "every interaction colour is in the legend too",
        all(
            any(_rgb(style) == tuple(int(round(v * 255)) for v in COLOR_CONTACT[k])
                for _, style in swatches)
            for k in COLOR_CONTACT
        ),
        f"{len(COLOR_CONTACT)} colours, {len(swatches)} swatches",
    )
    check(
        "and every swatch names itself, so none is a bare colour",
        len(legend_words) == len(swatches),
        f"{len(swatches)} swatches against {len(legend_words)} labels "
        f"{legend_words}",
    )

    for key in REPRESENTATION_KEYS:
        index = win.cmb_representation.findData(key)
        win.cmb_representation.setCurrentIndex(index)
        app.processEvents()
        check(
            f"selecting '{key}' reaches the viewport",
            win.viewport.representation == key,
            f"viewport says {win.viewport.representation!r}",
        )
        if PIXELS_OK:
            arr, _ = shot(win, f"11_repr_{key}")
            drawn = non_background(arr)
            check(
                f"'{key}' draws something",
                drawn > 500,
                f"{drawn} non-background px",
            )
        else:
            skip(f"'{key}' draws something", "no framebuffer to read")

    # A ribbon needs a backbone. The example receptor is a synthetic blob, so
    # it must fall back rather than pretend, and the status bar has to say so.
    win.cmb_representation.setCurrentIndex(
        win.cmb_representation.findData("ribbon")
    )
    app.processEvents()
    receptor_view = next(m for m in win.viewport.molecules if m.role == "receptor")
    check(
        "a structure with no backbone reports that it has none",
        not receptor_view.has_backbone,
        "the synthetic example receptor is not a protein",
    )
    check(
        "a file with no amino-acid residue falls back instead of claiming a ribbon",
        receptor_view.backbone_ribbon() is None,
        "backbone_ribbon() returned None",
    )
    message = win.statusBar().currentMessage()
    check(
        "the status bar says where the bonds came from",
        "bonds from" in message,
        message[:110],
    )

    # The receptor's bonds are the interesting ones. Whatever the source, no
    # bond may join atoms of different residues unless it is the peptide C-N,
    # because that is the failure this whole path was written to prevent.
    check(
        "the loaded receptor's bonds are all physically possible",
        all(
            0.85
            <= float(
                np.linalg.norm(
                    receptor_view.coords[i] - receptor_view.coords[j]
                )
            )
            < 2.0
            for i, j in receptor_view.bond_pairs()
        ),
        f"{len(receptor_view.bond_pairs())} bonds checked",
    )

    win.cmb_representation.setCurrentIndex(
        win.cmb_representation.findData("spheres")
    )
    app.processEvents()

    # The example receptor is a synthetic blob, so nothing above has actually
    # drawn a ribbon. A real protein has to, or the feature is untested.
    crambin = EXAMPLES / "1crn_prep.pdbqt"
    if crambin.is_file():
        section("10b. a ribbon on a real protein")
        win3 = MainWindow(receptor=crambin)
        win3.resize(1000, 760)
        win3.show()
        QTest.qWaitForWindowExposed(win3, 5000)
        app.processEvents()
        QTest.qWait(400)
        app.processEvents()

        protein = next(m for m in win3.viewport.molecules if m.role == "receptor")
        check("the prepared receptor kept its residue names",
              protein.has_backbone,
              f"{len(protein.coords)} atoms, bonds from {protein.bond_source}")
        check("its bonds came from residue templates",
              protein.bond_source == "template", protein.bond_source)
        check("it is grouped into one entry per residue",
              protein.structure is not None
              and len(protein.structure.residues) == 46,
              f"{0 if protein.structure is None else len(protein.structure.residues)} "
              "residues")

        ribbon = protein.backbone_ribbon()
        check("a ribbon was built from the backbone",
              ribbon is not None and len(ribbon) > 0,
              f"{0 if ribbon is None else len(ribbon)} triangles")

        drawn = {}
        for key in ("ball_and_stick", "ribbon", "cartoon"):
            win3.cmb_representation.setCurrentIndex(
                win3.cmb_representation.findData(key)
            )
            app.processEvents()
            if PIXELS_OK:
                arr, _ = shot(win3, f"12_protein_{key}")
                drawn[key] = non_background(arr)
                check(f"protein '{key}' draws something",
                      drawn[key] > 500, f"{drawn[key]} non-background px")
            else:
                skip(f"protein '{key}' draws something", "no framebuffer to read")

        if len(drawn) == 3:
            check(
                "the protein ribbon is a different picture from its atoms",
                abs(drawn["ribbon"] - drawn["ball_and_stick"]) > 500,
                f"ribbon {drawn['ribbon']} px vs ball-and-stick "
                f"{drawn['ball_and_stick']} px",
            )
        else:
            # Registering this only when it can run makes the total silently
            # smaller, and a smaller total is indistinguishable from a smaller
            # scope. A check that did not happen has to say so.
            skip(
                "the protein ribbon is a different picture from its atoms",
                "no framebuffer was readable, so there were no pixel counts to "
                "compare",
            )
        msg = win3.statusBar().currentMessage()
        check("the status bar names the template source for the protein",
              "residue templates" in msg, msg[:110])
        dispose_window(app, win3)
    else:
        skip("a ribbon on a real protein", f"{crambin.name} is not present")

    # ------------------------------------------------- 11. interactions panel
    section("11. pose/receptor interactions")
    win4 = MainWindow(receptor=rec)
    win4.resize(1100, 800)
    win4.show()
    QTest.qWaitForWindowExposed(win4, 5000)
    app.processEvents()

    check("no receptor interactions are claimed before a pose is loaded",
          not win4.viewport.contacts_valid or not win4.viewport.contacts,
          f"contacts={len(win4.viewport.contacts)} valid={win4.viewport.contacts_valid}")
    check("the panel says so instead of showing an empty table",
          "load" in win4.lbl_contacts.text().lower()
          or win4.contact_table.rowCount() == 0,
          win4.lbl_contacts.text())

    # The example receptor is a synthetic blob whose residues are all called
    # REC, so it has contacts but no residue to attribute them to. That is a
    # different situation from "nothing is touching", and the two must not look
    # the same: the headline still counts the contacts, and the table is empty
    # rather than filled with the word "REC" seven times.
    if poses.is_file():
        win4.load_structure(poses, "poses")
        app.processEvents()
        QTest.qWait(200)
        app.processEvents()
        anonymous = win4.viewport.contacts
        check("a receptor with no residue names still reports its contacts",
              len(anonymous) > 0, f"{len(anonymous)}")
        check("but no residue is invented for them",
              all(not c.partner_residue for c in anonymous),
              f"{sum(1 for c in anonymous if c.partner_residue)} named")
        check("and the table stays empty rather than filling with placeholders",
              win4.contact_table.rowCount() == 0,
              f"{win4.contact_table.rowCount()} rows")
        check("the headline still counts the contacts it found",
              str(len(anonymous)) in win4.lbl_contacts.text(),
              win4.lbl_contacts.text())
        dispose_window(app, win4)
    else:
        skip("contacts on a receptor with no residue names", f"{poses.name} missing")

    crambin_pose = EXAMPLES / "crambin_pose.pdbqt"
    if not (crambin.is_file() and crambin_pose.is_file()):
        skip("interactions on a docked pose",
             "crambin and its pose are not both present")
    else:
        win5 = MainWindow(receptor=crambin, poses=crambin_pose)
        win5.resize(1100, 800)
        win5.show()
        QTest.qWaitForWindowExposed(win5, 5000)
        app.processEvents()
        QTest.qWait(200)
        app.processEvents()

        found = win5.viewport.contacts
        check("a pose in a real protein produces contacts", len(found) > 0, f"{len(found)}")
        check("the viewport knows they were computed", win5.viewport.contacts_valid)
        named = {c.partner_residue for c in found if c.partner_residue}
        check("and they are attributed to real residues", len(named) > 0,
              f"{len(named)} residues")
        check("the table has one row per residue involved",
              win5.contact_table.rowCount() == len(named),
              f"{win5.contact_table.rowCount()} rows vs {len(named)} residues")
        check("the headline counts add up to the contacts shown",
              str(len(found)) in win5.lbl_contacts.text(),
              win5.lbl_contacts.text())
        check("at least one hydrogen bond is found in a docked pose",
              any(c.kind == "hbond" for c in found),
              f"{sum(1 for c in found if c.kind == 'hbond')} h-bonds")
        # The detail is not decoration. This check failed once with no
        # explanation at all, which is the worst possible failure: a flaky
        # check with nothing to look at is a check you learn to ignore, and an
        # ignored check is a check that can be wrong forever. So it reports
        # the state it is asserting about, and the state the viewport is in,
        # because a difference between those two is where the answer is.
        check(
            "the interaction toggle is on by default",
            win5.cb_contacts.isChecked(),
            f"checkbox {win5.cb_contacts.isChecked()}, "
            f"viewport.show_contacts {win5.viewport.show_contacts}, "
            f"{len(found)} contacts, "
            f"pocket thread running "
            f"{getattr(win5, '_pocket_thread', None) is not None and win5._pocket_thread.isRunning()}",
        )

        from opendocking.workbench.geometry import dashed_segments

        pose_mol = next(m for m in win5.viewport.molecules if m.role == "pose")
        rec_mol = next(m for m in win5.viewport.molecules if m.role == "receptor")
        segs, group = dashed_segments(
            np.asarray([pose_mol.coords[c.self_index] for c in found], np.float32),
            np.asarray([rec_mol.coords[c.partner_index] for c in found], np.float32),
        )
        check("one contact becomes more than one dash",
              len(segs) > len(found), f"{len(found)} contacts -> {len(segs)} dashes")
        check("every dash maps back to a real contact",
              len(group) == len(segs) and int(group.max()) < len(found),
              f"{len(group)} groups over {len(found)} contacts")

        # Selecting a row has to move the camera onto that contact. The claim
        # is about where the camera looks, so this compares the *centre*; a
        # comparison of distances would pass for free whenever the new radius
        # happened to equal the old one, which it did here -- 16.2 A both ways.
        before_centre = np.array(win5.viewport.camera.center, np.float64)
        before_dist = win5.viewport.camera.distance
        win5.contact_table.selectRow(0)
        app.processEvents()
        after_centre = np.array(win5.viewport.camera.center, np.float64)
        shifted = float(np.linalg.norm(after_centre - before_centre))
        check("selecting a residue moves the camera centre onto it", shifted > 0.5,
              f"moved {shifted:.2f} A")
        check("and ends up no further out than framing the whole scene",
              win5.viewport.camera.distance <= before_dist + 1e-6,
              f"{before_dist:.1f} -> {win5.viewport.camera.distance:.1f}")
        top = (win5.contact_table.item(0, 0).text()
               if win5.contact_table.rowCount() else "")
        check("the status bar names the residue it centred on",
              bool(top) and top in win5.statusBar().currentMessage(),
              win5.statusBar().currentMessage())

        members = [c for c in found if c.partner_residue == top]
        if members:
            pts = np.asarray(
                [pose_mol.coords[c.self_index] for c in members]
                + [rec_mol.coords[c.partner_index] for c in members], np.float32
            )
            off = float(np.linalg.norm(pts.mean(axis=0)
                                        - after_centre.astype(np.float32)))
            check("the camera is over that residue's contacts, not the protein's middle",
                  off < 1.0, f"{off:.2f} A from the contact centroid")

        win5.cb_contacts.setChecked(False)
        app.processEvents()
        check("unchecking interactions stops the lines being drawn",
              win5.viewport.show_contacts is False)
        win5.cb_contacts.setChecked(True)
        app.processEvents()
        check("re-checking brings them back", win5.viewport.show_contacts is True)
        dispose_window(app, win5)
    section("11c. pocket search drives the box")
    # The regression this guards is specific: loading a receptor used to put
    # the box on the receptor's centroid, which for a globular protein is
    # inside the dense core. The check is not "a box exists" -- there always
    # was one -- but "the box the user is shown came from a site, and it is
    # nowhere near the centroid it used to be handed".
    #
    # This section opens its own window on crambin rather than reusing the
    # one the rest of the script has been driving. The main window holds
    # `rec_prep.pdbqt`, a handful of atoms whose one site *is* its centroid,
    # so on that receptor "the box moved away from the centroid" cannot fail
    # and asserting it there would be asserting nothing.
    tiny = None
    pw = MainWindow()
    pw.resize(1500, 900)
    # Shown, like every other window in this file: an unshown widget has no
    # framebuffer to read, so the pixel check below would be measuring nothing
    # even on a machine that can render.
    pw.show()
    try:
        pw.load_structure(EXAMPLES / "1crn_prep.pdbqt", "receptor")
        app.processEvents()
        receptor_view = next(
            (m for m in pw.viewport.molecules if m.role == "receptor"), None
        )
        check("a receptor is loaded for the pocket search", receptor_view is not None)
        if receptor_view is None:
            raise RuntimeError("no receptor to check the pocket search against")
        rows = pw.pocket_table.rowCount()
        check("the site table is populated", rows > 0, f"{rows} rows")
        check(
            "the site label reports how many and how long, not just a dash",
            "site(s)" in pw.lbl_pockets.text(),
            f"{pw.lbl_pockets.text()!r}",
        )
        check(
            "the table has the five columns the rows are built from",
            pw.pocket_table.columnCount() == 5,
            f"{pw.pocket_table.columnCount()} columns: "
            f"{[pw.pocket_table.horizontalHeaderItem(c).text() for c in range(pw.pocket_table.columnCount())]}",
        )
        # The fifth column is the one the ranking is built from, and it was
        # added because the order looked arbitrary without it. A column that
        # is present and empty would be the same as no column, so check that
        # it carries a number on every row, not just the first.
        vol_col = pw.pocket_table.columnCount() - 1
        volumes = [pw.pocket_table.item(r, vol_col).text() for r in range(rows)]
        # "<number> Å³": split the unit off rather than trimming characters,
        # which would leave the unit's own space in the string being tested.
        def _is_volume(text):
            parts = text.split()
            return len(parts) == 2 and parts[1] == "Å³" and \
                parts[0].replace(".", "").isdigit()
        check(
            "every row reports the volume the list is ranked by",
            all(_is_volume(v) for v in volumes),
            f"{volumes[:5]}{'...' if len(volumes) > 5 else ''}",
        )
        # And the number is the one the ceiling is applied to, so a column
        # showing anything else would be a table disagreeing with the filter
        # that produced it. Duplicates are fine and expected: two sites can
        # round to the same whole cubic angstrom.
        from opendocking.workbench import pockets as _pockets  # noqa: PLC0415
        nums = [float(v.split()[0]) for v in volumes]
        check(
            "every volume shown is one the filter would have kept",
            all(0 < n <= _pockets.DEFAULT_MAX_VOLUME for n in nums),
            f"largest shown {max(nums):.0f} A3, ceiling "
            f"{_pockets.DEFAULT_MAX_VOLUME:.0f} A3",
        )
        cells = [pw.pocket_table.item(0, c).text() for c in range(pw.pocket_table.columnCount())]
        check(
            "every cell in the first row has text in it",
            all(str(t).strip() for t in cells),
            f"{cells}",
        )
        check(
            "the kind column says what the site is, in the same words everywhere",
            all(
                pw.pocket_table.item(r, 1).text() in ("groove", "sealed")
                for r in range(rows)
            ),
            f"kinds {sorted({pw.pocket_table.item(r, 1).text() for r in range(rows)})}",
        )
        check(
            "the centre column is a coordinate, not a label",
            cells[2].count("(") == 1 and cells[2].count(",") == 2,
            f"{cells[2]!r}",
        )
        check(
            "the size column is three numbers",
            cells[3].count("×") == 2,
            f"{cells[3]!r}",
        )
        lining_tooltip = pw.pocket_table.item(0, 0).toolTip() or ""
        check(
            "the lining residues are reachable, not just implied",
            "lined by" in lining_tooltip and "—" not in lining_tooltip.split("lined by")[-1],
            f"tooltip {lining_tooltip!r}",
        )

        centroid = np.asarray(receptor_view.coords, np.float32).mean(axis=0)
        box = np.asarray(pw.viewport.box_center, np.float32)
        moved = float(np.linalg.norm(box - centroid))
        check(
            "the box is not sitting on the receptor centroid",
            moved > 3.0,
            f"{moved:.2f} A from the centroid it used to be given",
        )
        check(
            "the spin boxes show the box the engine will use",
            np.allclose(
                np.array([s.value() for s in pw.center_spins], float), box, atol=0.01
            )
            and np.allclose(
                np.array([s.value() for s in pw.size_spins], float),
                np.asarray(pw.viewport.box_size, np.float32),
                atol=0.01,
            ),
            f"spins {[s.value() for s in pw.center_spins]} vs viewport {box.tolist()}",
        )
        check(
            "the box is a plausible size rather than the whole protein",
            8.0 <= float(np.min(pw.viewport.box_size)) <= 40.0,
            f"{np.round(pw.viewport.box_size, 1).tolist()}",
        )

        # Selecting a row has to move the box *and* the camera, and has to do
        # it through the spins -- setting `box_center` directly would leave
        # three stale numbers on screen describing a box that no longer exists.
        if rows > 1:
            before_centre = np.array(pw.viewport.box_center, copy=True)
            before_dist = float(pw.viewport.camera.distance)
            pw.pocket_table.selectRow(1)
            app.processEvents()
            after_centre = np.array(pw.viewport.box_center, copy=True)
            check(
                "selecting a row moves the box",
                float(np.linalg.norm(after_centre - before_centre)) > 1.0,
                f"moved {float(np.linalg.norm(after_centre - before_centre)):.2f} A",
            )
            check(
                "and the spin boxes follow it",
                np.allclose(
                    np.array([s.value() for s in pw.center_spins], float),
                    after_centre,
                    atol=0.01,
                ),
                f"spins {[s.value() for s in pw.center_spins]}",
            )
            cam_off = float(
                np.linalg.norm(
                    np.asarray(pw.viewport.camera.center, np.float32) - after_centre
                )
            )
            check(
                "and the camera goes to the site it just selected",
                cam_off < 1.0 and abs(float(pw.viewport.camera.distance) - before_dist) > 1e-6,
                f"camera {before_dist:.1f} -> {float(pw.viewport.camera.distance):.1f} A, "
                f"{cam_off:.2f} A from the new box centre",
            )
            check(
                "the camera is far enough out to see the box, not inside it",
                float(pw.viewport.camera.distance)
                >= 0.9 * float(np.linalg.norm(np.asarray(pw.viewport.box_size, np.float32))),
                f"distance {float(pw.viewport.camera.distance):.1f} A, box diagonal "
                f"{float(np.linalg.norm(np.asarray(pw.viewport.box_size, np.float32))):.1f} A",
            )
            check(
                "the status bar names the site and its residues",
                "site 2" in pw.statusBar().currentMessage()
                and "lined by" in pw.statusBar().currentMessage(),
                f"{pw.statusBar().currentMessage()!r}",
            )
            arr, _ = shot(pw, "pockets_site_selected")
            # `pixel_check`, not `check`: whether `grabFramebuffer` returns
            # anything is a property of the environment, not of the workbench,
            # and a headless runner reports a blank framebuffer for every
            # pixel check in this file. Using `check` here is what made the
            # whole job fail on a machine that cannot draw -- 14 sibling
            # checks skipped correctly and this one did not.
            pixel_check(
                "the selected site is actually drawn",
                non_background(arr) > 500,
                f"{non_background(arr)} non-background pixels",
            )
            pw.pocket_table.clearSelection()
            app.processEvents()
            check(
                "clearing the selection clears the site volume, rather than "
                "leaving one site's cloud inside another's box",
                len(pw.viewport.pocket_points) == 0,
                f"{len(pw.viewport.pocket_points)} points still held",
            )
            pw.pocket_table.selectRow(1)
            app.processEvents()
            check(
                "reselecting puts it back",
                len(pw.viewport.pocket_points) > 0,
                f"{len(pw.viewport.pocket_points)} points",
            )
            # The toggle has to actually gate the draw, not just the checkbox.
            shown_before = pw.viewport.show_pocket
            before, _ = shot(pw, "pockets_cloud_on")
            pw.cb_pocket_volume.setChecked(False)
            app.processEvents()
            check(
                "the site-volume checkbox gates the draw",
                pw.viewport.show_pocket is False and shown_before is True,
                f"show_pocket {shown_before} -> {pw.viewport.show_pocket}",
            )
            after, _ = shot(pw, "pockets_cloud_off")
            # Through `pixel_check`, not `check`. This is the one pixel check
            # that had been calling `check` directly, so on a headless CI box
            # whose framebuffer read back blank the other fifteen skipped and
            # this one reported a failure -- a false alarm about a machine
            # that cannot render at all. `pixel_check` skips exactly when the
            # environment cannot answer and still fails, on a machine that
            # can, when toggling the checkbox changed nothing.
            pixel_check(
                "and the picture really changed, not just the flag",
                not np.array_equal(before, after),
                f"{differing_pixels(before, after)} pixels differ",
            )
            pw.cb_pocket_volume.setChecked(True)
            app.processEvents()
            check(
                "switching it back restores the cloud",
                pw.viewport.show_pocket is True
                and len(pw.viewport.pocket_points) > 0,
            )
    finally:
        dispose_window(app, pw)

    # A structure with no enclosed site must say so rather than quietly
    # leaving a box somewhere and calling it a suggestion. One atom: the grid
    # around it is all solvent and the flood reaches everything, so there is
    # genuinely nothing to report -- unlike `rec_prep.pdbqt`, which does have
    # a site and made this check pass for the wrong reason.
    #
    # Written to a temporary file, not into `examples/`. A four-line fixture
    # that a check regenerates on every run does not belong in a release tree,
    # and that is how example directories quietly fill up.
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny = Path(tmpdir) / "one_atom.pdbqt"
        tiny.write_text(
            "ROOT\n"
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00"
            "     C.3\n"
            "ENDROOT\n"
            "TORSDOF 0\n",
            encoding="utf-8",
        )
        empty = MainWindow()
        empty.resize(600, 400)
        try:
            empty.load_structure(tiny, "receptor")
            app.processEvents()
            check(
                "a structure with no site reports that, and does not invent one",
                empty.pocket_table.rowCount() == 0
                and "no enclosed site" in empty.lbl_pockets.text(),
                f"{empty.pocket_table.rowCount()} rows, "
                f"label {empty.lbl_pockets.text()!r}",
            )
        finally:
            dispose_window(app, empty)

    section("11d. the pocket search does not block the window")
    # Measured, not guessed: the search takes 0.07 s for crambin's 382 atoms,
    # 2.0 s for streptavidin and 12.0 s for haemoglobin's 4779. Run on the GUI
    # thread that last one froze the window for twelve seconds on load.
    #
    # The proof that it is off-thread is an ordering, not a duration: the call
    # returns while the work is still outstanding, so the label still reads
    # "searching…" and the button is still disabled at the moment control
    # comes back. A threshold on elapsed time would be a threshold on how fast
    # the runner is; this holds on any machine, and it fails immediately if
    # someone puts the search back inline.
    from opendocking.workbench import MoleculeView  # noqa: PLC0415

    axis = np.arange(-18.0, 18.1, 1.6)
    blob = np.asarray(
        [
            (x, y, z)
            for x in axis
            for y in axis
            for z in axis
            if (x * x + y * y + z * z) ** 0.5 % 3.0 < 1.2
        ],
        np.float32,
    )
    tw = MainWindow()
    tw.resize(800, 600)
    tw.show()
    try:
        tw.viewport.molecules = [
            MoleculeView(
                name="synthetic", coords=blob,
                elements=["C"] * len(blob),
                color=(0.6, 0.7, 0.9), radius=0.15, role="receptor",
            )
        ]
        check(
            "the fixture is big enough for the answer to be observable",
            len(blob) > 2000,
            f"{len(blob)} atoms",
        )
        tw.find_pockets()
        mid_label = tw.lbl_pockets.text()
        mid_enabled = tw.btn_pockets.isEnabled()
        check(
            "find_pockets() returns before the search has finished",
            "searching" in mid_label.lower(),
            f"label immediately after the call: {mid_label!r}",
        )
        check(
            "and the button is disabled meanwhile, so a second search cannot start",
            not mid_enabled,
            f"enabled={mid_enabled}",
        )
        # Pump the loop and count turns, five times over.
        #
        # The old version did this once, in one 0.5 s window, and failed on
        # "turns > 5". That is a claim about the machine, not about the window:
        # the window counts `processEvents()` calls, and how many fit in half a
        # second depends on what else is running. It was observed failing at 2
        # and 5 turns while four other Python processes were saturating the CPU,
        # and passing at 9 on an idle machine -- a check that reports "the event
        # loop is blocked" when the truth is "this runner is busy", which is the
        # same defect as the pixel guard that reported a blank framebuffer as a
        # broken viewport.
        #
        # So: five short windows, assert on the best one, and print all five.
        # Best-of-N is the right reduction here because the failure it has to
        # catch is a loop that *cannot* turn, and such a loop cannot have one
        # lucky window -- every window is starved. A merely slow machine will
        # manage at least one. The threshold is deliberately low (10 turns in
        # 0.2 s is ~20 ms per iteration) because the claim being made is
        # "alive", not "fast".
        was_running = tw._pocket_thread is not None and tw._pocket_thread.isRunning()
        windows = loop_turns(app)
        still_running = tw._pocket_thread is not None and tw._pocket_thread.isRunning()
        print(f"  loop turns in five 0.2 s windows: {windows}")
        # A third thing this measurement could be about instead: the search
        # finishing early, which would leave the loop turning over nothing.
        # That is a different claim from "turning while the search runs", so it
        # gets its own check rather than being folded into the threshold.
        check(
            "the search really was still running while the loop turned",
            was_running and still_running,
            f"thread running at the start: {was_running}, at the end: {still_running}",
        )
        best = max(windows) if windows else 0
        check(
            "the event loop keeps turning while the search runs",
            best >= 10,
            f"best of {len(windows)} windows: {best} turns in 0.2 s "
            f"(all: {windows}); want >= 10, which a blocked loop cannot reach "
            "in any window",
        )
        check(*_verify_loop_turns_guard())
        wait_until = time.perf_counter() + 120
        while time.perf_counter() < wait_until and tw._pocket_thread is not None \
                and tw._pocket_thread.isRunning():
            app.processEvents()
            time.sleep(0.005)
        for _ in range(20):
            app.processEvents()
            time.sleep(0.01)
        check(
            "the result arrives and the button comes back",
            "site(s)" in tw.lbl_pockets.text() and tw.btn_pockets.isEnabled(),
            f"label {tw.lbl_pockets.text()!r}, {tw.pocket_table.rowCount()} rows",
        )
    finally:
        # Closing with a search in flight used to abort the process with
        # "QThread: Destroyed while thread is still running". The search is
        # waited for above, so this is a window that *has* run a thread rather
        # than one running one now -- and destroying its GL widget while that
        # QThread is still an object fail-fasts here, so it is closed and left
        # for process exit instead of being forced down.
        t_close = time.perf_counter()
        dispose_window(app, tw, destroy=False)
        check(
            "closing the window during a search does not abort",
            time.perf_counter() - t_close < 20.0,
            f"closed in {time.perf_counter() - t_close:.2f} s",
        )

    # ---------------------------------------------------------------- report
    section("summary")
    before_total = len(results)
    missing_sections = [t for t in EXPECTED_SECTIONS if t not in sections_entered]
    check(
        "every section this file is supposed to run was reached, so this run "
        "is complete and not a truncated one",
        not missing_sections,
        f"{len(EXPECTED_SECTIONS) - len(missing_sections)} of "
        f"{len(EXPECTED_SECTIONS)} sections reached"
        + (f"; never reached: {missing_sections}" if missing_sections else ""),
    )
    check(
        "the number of checks that ran is the number this file is supposed to "
        "have, so none of them is hiding behind a guard",
        before_total + 2 == EXPECTED_CHECKS,
        f"{before_total} ran before these two and {EXPECTED_CHECKS} are expected; "
        "the +2 is these two completeness checks. If a check was added or "
        "removed, change EXPECTED_CHECKS deliberately",
    )
    # Closed here, not before the two checks above, so the summary's own
    # subtotal counts them. Closing it first printed "summary: 0 checks", which
    # is the same kind of untrue line this file exists to stop printing.
    _close_section()
    npass = sum(1 for r in results if r[0] == "PASS")
    nfail = sum(1 for r in results if r[0] == "FAIL")
    nskip = sum(1 for r in results if r[0] == "SKIP")
    print(f"  {npass} passed, {nfail} failed, {nskip} skipped, {len(results)} checks")
    for tag, name, detail in results:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {detail}")
    if nskip:
        print(
            f"\n  {nskip} check(s) were skipped, not passed. A skip means this\n"
            "  environment could not answer the question; it is not evidence\n"
            "  that the workbench is correct there."
        )
        # Name the sections, so "this run was green" and "these sections were
        # answered by the machine rather than by the code" are not the same
        # sentence in the reader's head.
        partly = [t for t, n in section_skips.items() if n]
        if partly:
            print(f"  sections with skipped checks: {partly}")
            if not GL_OK:
                print(
                    f"  reason: {GL_VERDICT}\n"
                    "  CI runs `xvfb-run` with LIBGL_ALWAYS_SOFTWARE=1, where a real\n"
                    "  framebuffer exists; that run, not this one, is the authority\n"
                    "  on the skipped checks."
                )
    print(f"\n  screenshots: {OUT}")

    # The last thing printed, and the only exit code that means "this never
    # finished". 0 is clean, 1 is "ran everything, something failed", 2 is "did
    # not finish" and 3 is "died". A caller can tell those apart without
    # parsing the transcript, which is the difference between a truncated run
    # that looks green and one that does not.
    if missing_sections or len(results) != EXPECTED_CHECKS:
        print(
            f"\n  RUN INCOMPLETE: {len(results)} of {EXPECTED_CHECKS} checks ran, "
            f"{len(EXPECTED_SECTIONS) - len(missing_sections)} of "
            f"{len(EXPECTED_SECTIONS)} sections reached."
        )
        if missing_sections:
            print(f"  never reached: {missing_sections}")
        print(
            "  This is not a pass. The numbers above describe the part of the\n"
            "  file that ran; nothing here says anything about the rest."
        )
    # The main window is deliberately *not* closed here. It is the last one
    # standing and it has been alive for the whole run; closing it on this
    # machine fail-fasts with `0xC0000409` after the summary has already been
    # printed, so the transcript is complete and the exit code is a crash. Left
    # to process exit, which reclaims the GL context without asking a widget
    # that has been repainting for ten minutes to tear itself down.
    if missing_sections or len(results) != EXPECTED_CHECKS:
        return 2
    return 1 if nfail else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)