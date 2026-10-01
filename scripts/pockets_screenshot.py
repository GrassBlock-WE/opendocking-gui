"""Look at the workbench with a receptor loaded, and see where the box went.

Not a pass/fail check on the geometry -- a way of *seeing* the pocket list and
the box it places. The claim being looked at is that the search box no longer
sits on the receptor's centroid, and a number in a spin box is not evidence of
that: only the picture of the box on a real site is. The camera-framing bug that
made every earlier screenshot a close-up of a few carbons was invisible in the
numbers, which were all correct.

**What it is therefore allowed to fail about, and what it is not.**

It does not assert any pocket geometry: site count, box size and burial are
printed for a human to read. It *does* assert that the frames it says it wrote
were written and are not flat, because otherwise the script is a diagnostic that
cannot fail:

* `QImage.save` returns a success flag, and it used to be discarded, so a frame
  that could not be encoded still printed "wrote ...".
* This script writes **three** frames, two of them only when the pocket table
  has a second row. That condition used to be an `if` with no `else`, so a
  search that found one site printed "done" having written one picture out of
  three and nothing said so. Each frame is now recorded, and a frame that was
  not taken is named.

A machine that cannot give the viewport an OpenGL context is a **SKIP with a
reason** taken from `odgui --check`, and produces no frame — which is reported
as producing no frame, not as success.

Exit codes: ``0`` every frame this run expected was written and is not flat,
``1`` a frame was missing or blank, ``2`` did not finish or produced nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Prefer the installed `opendocking`; only fall back to the source tree when the
# installed one is missing. Putting `dock-py/python` on the path unconditionally
# shadows the working package with a copy that cannot import, because a clean
# checkout has no compiled `_dockpy` there. Same note as `contacts_check.py`.
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

#: How many frames this script produces when the pocket table has a second row
#: to select. Counted here rather than written into the summary text, so the
#: number in the summary and the number of `save_frame` calls cannot drift apart.
TOTAL_FRAMES = 3

#: The frames this run wrote, so the summary can say how many there were rather
#: than counting checks and calling them frames.
FRAMES: list[str] = []

RESULTS: list[tuple[str, str, str]] = []


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


def save_frame(win, path: Path, label: str) -> None:
    """Grab the whole window, write it, and say whether that worked."""
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
    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    _pump(app)

    print("=== before any receptor is loaded ===")
    print(f"box centre: {list(win.viewport.box_center)}  size: {list(win.viewport.box_size)}")
    print(f"site label: {win.lbl_pockets.text()}")
    print(f"pocket rows: {win.pocket_table.rowCount()}")

    win.load_structure(EX / "1crn_prep.pdbqt", "receptor")
    _pump(app)

    vp = win.viewport
    receptor = next(m for m in vp.molecules if m.role == "receptor")
    centroid = np.asarray(receptor.coords, np.float32).mean(axis=0)
    box = np.asarray(vp.box_center, np.float32)
    print("\n=== after loading crambin ===")
    print(f"site label: {win.lbl_pockets.text()}")
    print(f"pocket rows: {win.pocket_table.rowCount()}")
    for r in range(win.pocket_table.rowCount()):
        cells = [win.pocket_table.item(r, c).text() for c in range(4)]
        print("   ", "  ".join(f"{c:>10}" for c in cells))
    print(f"box centre: {np.round(box, 2)}  size: {np.round(vp.box_size, 2)}")
    print(f"receptor centroid: {np.round(centroid, 2)}")
    print(f"distance from the centroid it used to use: {float(np.linalg.norm(box - centroid)):.2f} A")
    print(f"spin boxes agree with the viewport: "
          f"{[s.value() for s in win.center_spins] == [round(float(v), 3) for v in box]}")

    # The box being somewhere is the claim; there having to be *a* box is the
    # precondition for a picture of one. Asked before row 0's tooltip is read,
    # because that read is an index into a table that may be empty -- which is
    # how the old version turned "the search found nothing" into a traceback
    # rather than a sentence.
    n_sites = win.pocket_table.rowCount()
    if n_sites:
        ok("the pocket search found sites to look at", f"{n_sites} site(s)")
    else:
        bad("the pocket search found sites to look at",
            "no sites were found, so there is no box to look at")
    if n_sites:
        print(f"tooltip on row 0:\n{win.pocket_table.item(0, 0).toolTip()}")

    # The box must actually be around protein, not floating in the solvent.
    corner = box + np.asarray(vp.box_size, np.float32) / 2.0
    d = np.linalg.norm(np.asarray(receptor.coords, np.float32) - box, axis=1)
    print(f"nearest receptor atom to the box centre: {float(d.min()):.2f} A")
    print(f"receptor atoms within the box: "
          f"{int((d < float(np.max(vp.box_size)) / 2).sum())}/{len(receptor.coords)}")
    del corner

    reason = _no_context_reason(vp)
    if reason:
        print(f"\n  no OpenGL context, so no frame can be produced: {reason}")
        for label in ("overview", "site", "no-cloud"):
            skip(f"{label} frame", reason)
        win.close()
        return summarise()

    win.viewport.frame_all()
    _pump(app)
    save_frame(win, OUT / "shot_pockets_overview.png", "overview")

    # Select the site the docked ligand actually occupies and look at it close up.
    if win.pocket_table.rowCount() > 1:
        win.pocket_table.selectRow(1)
        _pump(app)
        print(f"\nafter selecting row 1: camera dist {vp.camera.distance:.1f}, "
              f"centre {np.round(vp.camera.center, 2)}")
        print(f"site cloud: {len(vp.pocket_points)} points, "
              f"radius {vp.pocket_point_radius}, opacity {vp.pocket_opacity}, "
              f"shown={vp.show_pocket}")
        save_frame(win, OUT / "shot_pockets_site.png", "site")

        # The same picture with the cloud switched off. The question the cloud has
        # to answer is "does this volume sit in a crevice, between those atoms",
        # and a screenshot that always includes it cannot answer that -- there is
        # nothing to compare it against.
        win.cb_pocket_volume.setChecked(False)
        _pump(app)
        print(f"cloud hidden: shown={vp.show_pocket}, "
              f"points still held {len(vp.pocket_points)}")
        save_frame(win, OUT / "shot_pockets_nocloud.png", "no-cloud")
        win.cb_pocket_volume.setChecked(True)
        _pump(app, rounds=3, ms=60)
    else:
        # Named rather than passed over in silence: one site means three of the
        # four frames this script exists to produce do not exist, and the old
        # run said "done".
        why = (f"the pocket table has {win.pocket_table.rowCount()} row(s); this "
               "script needs a second one to select a site")
        for label in ("site", "no-cloud"):
            skip(f"{label} frame", why)

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
    if nskip:
        print(f"    {len(FRAMES)} of {TOTAL_FRAMES} frames were written; the rest "
              "are named above")
    if incomplete:
        print(f"RESULT: DID NOT FINISH — {incomplete}")
        return EXIT_INCOMPLETE
    if nfail:
        print("RESULT: FAIL — the run finished and a frame is missing or blank")
        return EXIT_FAILED
    if not FRAMES:
        # A screenshot script that produced no picture has not done its job, and
        # reporting that as a pass would be a green run with no evidence in it.
        print("RESULT: DID NOT FINISH — no frame was produced, so there is "
              "nothing here to look at")
        return EXIT_INCOMPLETE
    print(f"RESULT: PASS — {len(FRAMES)} of {TOTAL_FRAMES} frame(s) written and "
          "not flat: " + ", ".join(FRAMES))
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
