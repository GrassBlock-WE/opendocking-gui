"""Look at the workbench with a real pose in it, and save what it draws.

Not a pass/fail check on the chemistry -- a way of *seeing* the interface lines
and the fog before believing any assertion about them. Geometry that is correct
and a picture that is wrong are different states, and only the second one
matters to a user. The hydrophobic contacts that were invisible against grey
protein were found by looking at this script's output, not by an assertion.

**What it is therefore allowed to fail about, and what it is not.**

It does not assert any number: contacts, energies and row counts are printed
for a human to read, and a change in them is a finding rather than a red build.
It *does* assert the two things that make the picture worth looking at, because
without them the script is a diagnostic that cannot fail and rots silently:

* every frame it says it wrote was actually written. `QImage.save` returns a
  success flag, and that flag used to be discarded, so a frame that could not be
  encoded still printed "wrote ...".
* every frame it wrote is not a single flat colour. A blank window is not a
  screenshot of anything, and the old script would have printed "done" either
  way. This is the one assertion added, and it is the one that makes "nobody ran
  it" distinguishable from "it ran and produced nothing".

A machine that cannot give the viewport an OpenGL context is a **SKIP with a
reason**, taken from `odgui --check`, not a failure: there is no picture to look
at and pretending otherwise would be a lie in the other direction.

Exit codes: ``0`` every frame this run expected was written and is not flat,
``1`` a frame was missing or blank, ``2`` did not finish.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Prefer the installed `opendocking`; only fall back to the source tree when the
# installed one is missing. Putting `dock-py/python` on the path unconditionally
# shadows the working package with a copy that cannot import, because a clean
# checkout has no compiled `_dockpy` there. Same note as `contacts_criteria_check.py`.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

import numpy as np  # noqa: E402
from PyQt6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets  # noqa: E402

#: The one place in the tree that knows how to ask `odgui --check` what this
#: machine can do; it was a private copy in each of four check scripts.
#: `python scripts/_gui_check.py` audits that those four still use it.
from _gui_check import odgui_check  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402

EX = ROOT / "examples"
OUT = ROOT / "dist"
OUT.mkdir(parents=True, exist_ok=True)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

#: Flat-colour threshold for a frame that shows something. The same shape of
#: test `workbench_smoke.py` applies to the widget framebuffer: a window that
#: painted nothing has one colour in it.
MIN_COLOURS = 5

RESULTS: list[tuple[str, str, str]] = []

#: The frames this run actually wrote, so the summary can say how many there
#: were rather than counting checks and calling them frames.
FRAMES: list[str] = []


def record(tag: str, name: str, reason: str = "") -> bool:
    RESULTS.append((tag, name, reason))
    print(f"  [{tag}] {name}" + (f"  — {reason}" if reason else ""))
    return tag == "PASS"


def ok(name: str, reason: str = "") -> bool:
    return record("PASS", name, reason)


def bad(name: str, reason: str) -> bool:
    return record("FAIL", name, reason)


def skip(name: str, reason: str) -> bool:
    return record("SKIP", name, reason)


def _no_context_reason(viewport) -> str:
    """Why there is no picture to look at, or "" if there is one."""
    if viewport._ctx is not None:
        return ""
    answer = odgui_check()
    tip = viewport.toolTip()
    if answer.can_make_context:
        return (
            "odgui --check reports this machine CAN create a context "
            f"({answer.describe()}), so a viewport without one is a defect: "
            f"{tip or '(Qt gave no reason)'}"
        )
    return f"{tip or 'Qt never called initializeGL'}; {answer.describe()}"


def _pump(app, rounds: int = 4, ms: int = 80) -> None:
    """Let Qt settle between states, alternating with a real sleep.

    The sleeps are here because a repaint needs a turn of the event loop *and*
    time to run; they are not a way of making a check pass, and nothing in this
    file retries a failed measurement.
    """
    for _ in range(rounds):
        app.processEvents()
        QtCore.QThread.msleep(ms)
        app.processEvents()


def _colours(pix) -> int:
    """How many distinct colours are in a grabbed frame.

    Sampled on a 4-pixel grid: this is a "did anything get drawn" test, and a
    quarter of a million pixels per frame buys no extra confidence for it.
    """
    img = pix.toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
    bits = img.constBits()
    bits.setsize(img.sizeInBytes())
    a = np.frombuffer(bits, np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4
    )[:, : img.width(), :3]
    a = a[::4, ::4].astype(np.int32)
    key = (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]
    return int(len(np.unique(key)))


def save_frame(win, app, path: Path, label: str) -> None:
    """Grab the whole window, write it, and say whether that worked.

    Three things can each go wrong and all three used to print "wrote ...":
    the grab returning nothing, the encoder refusing, and the encoder writing a
    flat image. They are checked separately so the summary says which happened.
    """
    pix = win.grab()
    if pix.isNull():
        bad(f"{label}: frame written", "the window grab returned nothing")
        return
    if not pix.save(str(path)):
        bad(f"{label}: frame written", f"the encoder refused to write {path}")
        return
    if not path.is_file() or path.stat().st_size == 0:
        bad(f"{label}: frame written", f"{path} is missing or empty on disk")
        return
    colours = _colours(pix)
    size = path.stat().st_size
    FRAMES.append(path.name)
    print(f"wrote {path}  ({size} bytes, {colours} distinct colours)")
    if colours < MIN_COLOURS:
        bad(
            f"{label}: frame shows something",
            f"only {colours} distinct colour(s) — the window painted nothing, so "
            f"there is nothing here to look at",
        )
    else:
        ok(f"{label}: frame written and not flat",
           f"{size} bytes, {colours} distinct colours")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow(receptor=EX / "1crn_prep.pdbqt", poses=EX / "crambin_pose.pdbqt")
    win.resize(1500, 900)
    win.show()
    win.viewport.update()
    for _ in range(6):
        app.processEvents()
        QtCore.QThread.msleep(120)
        app.processEvents()

    vp = win.viewport
    print(f"molecules: {[(m.role, m.name, len(m.coords)) for m in vp.molecules]}")
    print(f"contacts: {len(vp.contacts)}  valid={vp.contacts_valid}")
    print(f"label: {win.lbl_contacts.text()}")
    print(f"rows in table: {win.contact_table.rowCount()}")
    for r in range(win.contact_table.rowCount()):
        cells = [win.contact_table.item(r, c).text() for c in range(4)]
        print("   ", "  ".join(f"{c:>9}" for c in cells))
    print(f"samples granted: {vp.samples_granted}")
    print(f"rep: {win.cmb_representation.currentData()}")

    # This script's whole subject is the contacts. A pose with none of them is
    # not a picture of anything this script exists to show, and it used to be
    # indistinguishable from a pose whose contacts are merely hidden.
    if vp.contacts:
        ok("the pose has contacts to look at",
           f"{len(vp.contacts)} over {win.contact_table.rowCount()} residues")
    else:
        bad("the pose has contacts to look at",
            "no contacts were found, so the overview cannot show what this "
            "script exists to show")

    reason = _no_context_reason(vp)
    if reason:
        print(f"\n  no OpenGL context, so no frame can be produced: {reason}")
        skip("overview frame", reason)
        skip("zoom frame", reason)
        win.close()
        return summarise()

    # Whole scene first.
    win.viewport.frame_all()
    _pump(app)
    save_frame(win, app, OUT / "shot_contacts_overview.png", "overview")

    # Then zoom onto the top-ranked residue, which is the interaction that matters.
    if win.contact_table.rowCount():
        win.contact_table.selectRow(0)
        top = win.contact_table.item(0, 0).text()
        print(f"focused on {top}: camera dist {vp.camera.distance:.1f}")
        _pump(app)
        save_frame(win, app, OUT / "shot_contacts_zoom.png", "zoom")
    else:
        # Named rather than passed over: the old script simply did not write the
        # second file and still printed "done".
        skip("zoom frame", "the contact table has no rows to select")

    win.close()
    app.processEvents()
    return summarise()


def summarise(incomplete: str = "") -> int:
    """Counts, and the verdict, on every path including a crash."""
    npass = sum(1 for tag, _, _ in RESULTS if tag == "PASS")
    nfail = sum(1 for tag, _, _ in RESULTS if tag == "FAIL")
    nskip = sum(1 for tag, _, _ in RESULTS if tag == "SKIP")
    print()
    print(f"--- summary: {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks")
    for tag, name, reason in RESULTS:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {reason}")
    if incomplete:
        print(f"RESULT: DID NOT FINISH — {incomplete}")
        return EXIT_INCOMPLETE
    if nfail:
        print("RESULT: FAIL — the run finished and a frame is missing or blank")
        return EXIT_FAILED
    if not FRAMES:
        # Reached when the machine could not give the viewport a context, and
        # deliberately *not* a pass. This script exists to produce pictures; a
        # run that produced none has not done its job, and reporting it as a
        # pass is the one outcome worse than the original "done", because it
        # would be a green build that produced no evidence at all.
        print("RESULT: DID NOT FINISH — no frame was produced, so there is "
              "nothing here to look at")
        return EXIT_INCOMPLETE
    print(f"RESULT: PASS — {len(FRAMES)} frame(s) written and not flat: "
          + ", ".join(FRAMES))
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to never be silent
        import traceback

        traceback.print_exc()
        print()
        print(f"DID NOT FINISH — an exception escaped before the frames above "
              f"were all recorded: {type(_exc).__name__}: {_exc}")
        raise SystemExit(summarise(f"an exception escaped: {type(_exc).__name__}: {_exc}"))
