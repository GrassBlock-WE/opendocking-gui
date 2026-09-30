"""Look at the workbench with a receptor loaded, and see where the box went.

Not a pass/fail check -- a way of *seeing* the pocket list and the box it
places. The claim being looked at is that the search box no longer sits on the
receptor's centroid, and a number in a spin box is not evidence of that: only
the picture of the box on a real site is.
"""

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

from PyQt6 import QtCore, QtOpenGLWidgets, QtWidgets  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402

EX = ROOT / "examples"
OUT = ROOT / "dist"
OUT.mkdir(parents=True, exist_ok=True)

app = QtWidgets.QApplication(sys.argv[:1])
win = MainWindow()
win.resize(1500, 900)
win.show()
for _ in range(4):
    app.processEvents()
    QtCore.QThread.msleep(80)
    app.processEvents()

print("=== before any receptor is loaded ===")
print(f"box centre: {list(win.viewport.box_center)}  size: {list(win.viewport.box_size)}")
print(f"site label: {win.lbl_pockets.text()}")
print(f"pocket rows: {win.pocket_table.rowCount()}")

win.load_structure(EX / "1crn_prep.pdbqt", "receptor")
for _ in range(4):
    app.processEvents()
    QtCore.QThread.msleep(80)
    app.processEvents()

import numpy as np  # noqa: E402

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
print(f"tooltip on row 0:\n{win.pocket_table.item(0, 0).toolTip()}")

# The box must actually be around protein, not floating in the solvent.
corner = box + np.asarray(vp.box_size, np.float32) / 2.0
d = np.linalg.norm(np.asarray(receptor.coords, np.float32) - box, axis=1)
print(f"nearest receptor atom to the box centre: {float(d.min()):.2f} A")
print(f"receptor atoms within the box: "
      f"{int((d < float(np.max(vp.box_size)) / 2).sum())}/{len(receptor.coords)}")
del corner

win.viewport.frame_all()
for _ in range(4):
    app.processEvents()
    QtCore.QThread.msleep(80)
    app.processEvents()
win.grab().save(str(OUT / "shot_pockets_overview.png"))
print(f"wrote {OUT / 'shot_pockets_overview.png'}")

# Select the site the docked ligand actually occupies and look at it close up.
if win.pocket_table.rowCount() > 1:
    win.pocket_table.selectRow(1)
    for _ in range(4):
        app.processEvents()
        QtCore.QThread.msleep(80)
        app.processEvents()
    print(f"\nafter selecting row 1: camera dist {vp.camera.distance:.1f}, "
          f"centre {np.round(vp.camera.center, 2)}")
    win.grab().save(str(OUT / "shot_pockets_site.png"))
    print(f"wrote {OUT / 'shot_pockets_site.png'}")

win.close()
app.processEvents()
print("done")
