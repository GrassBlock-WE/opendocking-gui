"""Render the before-and-after of a pose selection, so a human can look at it.

Run it directly to write the PNGs:

    F:\\python310\\python.exe scripts\\framing_selection_screens.py

**Written in the same turn as it is read.** The numbers say the pose goes from
0.15% of the frame to 2.46% and the interaction overlay from 1 042 px to 4 268.
Numbers are what the guard is written against, but whether a 16-atom ligand at
2.46% of a frame is *lookable* is a claim about a picture, and only a person can
close that gap. So this writes the two frames the change is about:

* `pose_before_whole_receptor.png` -- the shipped framing: everything in shot,
  the pose a sixteenth of a percent of it, which is what a selection used to
  leave you looking at
* `pose_after_selection.png`      -- the same pose, same orientation, same
  colours, after selecting it: the pose and its four contacting residues framed
  with the protein as context, and the dashed interaction lines drawn across it
* `pose_after_no_overlay.png`     -- the same frame with the interaction lines
  switched off, because "you can see the lines" is only a claim if there is a
  picture without them to compare against

The third image exists for that reason alone. A screenshot of a frame that
happens to contain some yellow pixels is not evidence that the overlay is
readable; a pair differing *only* in the overlay is.

The orientation is deliberately **not** changed between the two. A
before-and-after that also orbits the camera would let a reader attribute the
difference to the zoom, or to the rotation, or to neither -- and the whole
claim under test is the framing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dock-py" / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from PyQt6 import QtGui, QtWidgets  # noqa: E402

from opendocking.workbench.app import (  # noqa: E402
    MainWindow,
    POCKET_OPACITY_PLAIN,
)

from secondary_render import save_png  # noqa: E402

OUT = ROOT / "dist" / "framing"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1500, 950


def grab(win):
    """The viewport as an RGB array, background included."""
    win.viewport.repaint()
    QtWidgets.QApplication.processEvents()
    img = win.viewport.grabFramebuffer().convertToFormat(
        QtGui.QImage.Format.Format_RGB32
    )
    if img.isNull():
        return None
    p = img.constBits()
    p.setsize(img.sizeInBytes())
    a = np.frombuffer(p, dtype=np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4
    )
    return a[:, : img.width(), :3].copy()


def settle(win, app, limit: int = 200) -> None:
    """Run the camera move to completion, and *wait* for it.

    The waiting is the point, and the first version of this function was wrong
    in a way that quietly corrupted every number measured after it. It looped
    `app.processEvents()` and gave up after `limit` iterations, but the move is
    driven by a `QTimer` on **wall-clock** time: a tight `processEvents` loop
    burns its whole iteration budget in a few milliseconds and returns with the
    camera still in flight.

    Measured, on this machine: `settle()` returned with the move unfinished on
    3 of 4 poses, and two grabs taken back to back with *nothing changed* then
    differed by **47%, 64% and 75% of the frame** -- because they were of two
    different camera positions. Every "own pixels" figure measured by
    differencing two grabs is only meaningful once the camera has stopped.

    So this waits on the clock, and says so if the move will not finish.
    """
    from PyQt6.QtTest import QTest

    waited = 0
    while win.viewport.moving and waited < 4000:
        QTest.qWait(20)
        waited += 20
    for _ in range(10):
        app.processEvents()
    if win.viewport.moving:
        print(f"  WARNING: the camera move had not finished after {waited} ms")


def shares(win, base=None):
    """Pose/receptor/overlay pixel shares, by differencing the frames."""
    if base is None:
        base = grab(win)
    pose = next(m for m in win.viewport.molecules if m.role == "pose")
    rec = next(m for m in win.viewport.molecules if m.role == "receptor")
    frames = {}
    for role, name in (("pose", "pose"), ("receptor", "receptor")):
        for m in win.viewport.molecules:
            if m.role == role:
                m.visible = False
        win.viewport.update()
        for _ in range(6):
            QtWidgets.QApplication.processEvents()
        frames[name] = grab(win)
        for m in win.viewport.molecules:
            if m.role == role:
                m.visible = True
        win.viewport.update()
        for _ in range(6):
            QtWidgets.QApplication.processEvents()
    win.cb_contacts.setChecked(False)
    for _ in range(6):
        QtWidgets.QApplication.processEvents()
    no_ovl = grab(win)
    win.cb_contacts.setChecked(True)
    for _ in range(6):
        QtWidgets.QApplication.processEvents()

    total = base.shape[0] * base.shape[1]

    def diff(a, b):
        if a is None or b is None:
            return None
        return np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2) > 12

    pose_m = diff(base, frames["pose"])
    rec_m = diff(base, frames["receptor"])
    ovl_m = diff(base, no_ovl)
    return {
        "size": (base.shape[1], base.shape[0]),
        "pose_px": int(pose_m.sum()),
        "pose_share": float(pose_m.sum()) / total,
        "receptor_px": int(rec_m.sum()),
        "receptor_share": float(rec_m.sum()) / total,
        "overlay_px": int(ovl_m.sum()),
        "overlay_share": float(ovl_m.sum()) / total,
    }


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(
        receptor=ROOT / "examples" / "1crn_prep.pdbqt",
        ligand=ROOT / "examples" / "crambin_pose.pdbqt",
        poses=ROOT / "examples" / "poses.pdbqt",
    )
    win.resize(W, H)
    win.show()
    for _ in range(40):
        app.processEvents()

    pose = next(m for m in win.viewport.molecules if m.role == "pose")
    residues = sorted({c.partner_residue for c in win.viewport.contacts})
    print(f"pose {len(pose.coords)} atoms, {len(win.viewport.contacts)} contacts "
          f"over {len(residues)} residues: {residues}")

    # -- the "before": the framing a selection used to leave behind ---------
    # Settle first. Loading the poses selects a pose, which starts a camera
    # move, and calling `frame_all()` while that move is in flight measures two
    # different cameras rather than a framing: the first run of this file did,
    # and reported a 41% "overlay" on a frame where the camera had not stopped.
    #
    # The cloud is put back to `POCKET_OPACITY_PLAIN` for this shot, because
    # that is the state a selection used to leave behind: it moved neither the
    # camera nor the cloud. Selecting a pose now takes the cloud away, so leaving
    # it off here would quietly compare a new view against a new view and the
    # two numbers would flatter the change.
    settle(win, app)
    win.viewport.frame_all()
    win.viewport.pocket_opacity = POCKET_OPACITY_PLAIN
    settle(win, app)
    before = grab(win)
    sb = shares(win, before)
    print(f"\nbefore (whole receptor framed): {sb['size'][0]}x{sb['size'][1]}, "
          f"distance {win.viewport.camera.distance:.2f} A")
    print(f"  pose     {sb['pose_px']:7d} px  {sb['pose_share']:6.2%}")
    print(f"  receptor {sb['receptor_px']:7d} px  {sb['receptor_share']:6.2%}")
    print(f"  overlay  {sb['overlay_px']:7d} px  {sb['overlay_share']:6.2%}")

    # -- the "after": select a pose, and let the camera get there -----------
    # A *different* row from the one loaded, so this is a real selection rather
    # than a re-render of the framing the load already performed.
    rows = win.pose_table.rowCount()
    target = next((r for r in range(rows) if r != win.pose_table.currentRow()), 0)
    win.pose_table.selectRow(target)
    settle(win, app)
    after = grab(win)
    sa = shares(win, after)
    plan = win._last_selection_plan
    print(f"\nselected row {target} "
          f"({win.pose_table.item(target, 0).text() if win.pose_table.item(target, 0) else '?'}), "
          f"moved {plan.start_distance:.2f} -> {plan.target.distance:.2f} A "
          f"in {plan.steps} steps")
    print(f"after (selection framed): distance {win.viewport.camera.distance:.2f} A")
    print(f"  pose     {sa['pose_px']:7d} px  {sa['pose_share']:6.2%}  "
          f"({sa['pose_share'] / sb['pose_share']:.1f}x before)")
    print(f"  receptor {sa['receptor_px']:7d} px  {sa['receptor_share']:6.2%}  "
          f"({sa['receptor_share'] / sb['receptor_share']:.1f}x before)")
    print(f"  overlay  {sa['overlay_px']:7d} px  {sa['overlay_share']:6.2%}  "
          f"({sa['overlay_px'] / max(sb['overlay_px'], 1):.1f}x before)")

    # -- the same frame without the overlay, for the pair -----------------
    win.cb_contacts.setChecked(False)
    for _ in range(8):
        app.processEvents()
    bare = grab(win)
    win.cb_contacts.setChecked(True)
    for _ in range(8):
        app.processEvents()

    written = []
    for name, frame in (
        ("pose_before_whole_receptor", before),
        ("pose_after_selection", after),
        ("pose_after_no_overlay", bare),
    ):
        if frame is None:
            print(f"  {name}: no frame to write")
            continue
        written.append(save_png(OUT / f"{name}.png", frame))
        print(f"  wrote {written[-1]}")

    # -- the colour change alone, camera held fixed -----------------------
    # The pair that isolates item 1. Everything is identical between the two
    # frames -- same camera, same pose, same representation, same overlay
    # state -- and the only difference is the rule in `_draw_colours_for` that
    # decides what colour a pose is drawn in.
    #
    # "Before" is produced by calling the *previous* rule rather than by
    # checking out an old tree: it is the same renderer, the same frame and the
    # same camera, with one function replaced. A before/after that also changed
    # the camera would let a reader attribute the difference to the zoom.
    settle(win, app)
    win.cb_contacts.setChecked(False)
    for _ in range(8):
        app.processEvents()
    camera_pinned = (
        np.asarray(win.viewport.camera.center, np.float64).copy(),
        float(win.viewport.camera.distance),
    )

    def with_element_colours():
        """The rule this change replaced: element colours unless comparing."""
        def rule(mol, comparing=False):
            if mol.role == "pose_ghost" or (comparing and mol.role == "pose"):
                return np.tile(np.asarray(mol.color, np.float32),
                               (len(mol.coords), 1))
            return mol.atom_colors()
        return rule

    import opendocking.workbench.app as appmod

    live_rule = appmod._draw_colours_for
    appmod._draw_colours_for = with_element_colours()
    win.viewport.representation = win.viewport.representation  # force a redraw
    win.viewport.update()
    for _ in range(10):
        app.processEvents()
    colour_before = grab(win)
    appmod._draw_colours_for = live_rule
    win.viewport.update()
    for _ in range(10):
        app.processEvents()
    colour_after = grab(win)
    moved = max(
        float(np.abs(np.asarray(win.viewport.camera.center, np.float64)
                     - camera_pinned[0]).max()),
        abs(float(win.viewport.camera.distance) - camera_pinned[1]),
    )
    win.cb_contacts.setChecked(True)
    for _ in range(8):
        app.processEvents()
    print(f"\ncolour-only pair: interaction overlay OFF, camera held to within "
          f"{moved:.1e} A across both frames; the only difference is the rule "
          f"in _draw_colours_for")
    written = []
    for name, frame in (("pose_colour_before_element", colour_before),
                        ("pose_colour_after_identity", colour_after)):
        if frame is None:
            print(f"  {name}: no frame to write")
            continue
        written.append(save_png(OUT / f"{name}.png", frame))
        print(f"  wrote {written[-1]}")

    missing = [p for p in written if not Path(p).is_file()]
    print(f"\n{len(written)} file(s) written, {len(missing)} missing after write")
    for p in missing:
        print("  MISSING:", p)
    win.close()
    return 1 if missing or before is None or after is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
