"""Look at the workbench with a real pose in it, and save what it draws.

Not a pass/fail check -- a way of *seeing* the interface lines and the fog
before believing any assertion about them. Geometry that is correct and a
picture that is wrong are different states, and only the second one matters
to a user.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from PyQt6 import QtCore, QtOpenGLWidgets, QtWidgets  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402

EX = ROOT / "examples"
OUT = ROOT / "dist"
OUT.mkdir(parents=True, exist_ok=True)

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

# Whole scene first.
win.viewport.frame_all()
for _ in range(4):
    app.processEvents()
    QtCore.QThread.msleep(80)
    app.processEvents()
win.grab().save(str(OUT / "shot_contacts_overview.png"))
print(f"wrote {OUT / 'shot_contacts_overview.png'}")

# Then zoom onto the top-ranked residue, which is the interaction that matters.
if win.contact_table.rowCount():
    win.contact_table.selectRow(0)
    top = win.contact_table.item(0, 0).text()
    print(f"focused on {top}: camera dist {vp.camera.distance:.1f}")
    for _ in range(4):
        app.processEvents()
        QtCore.QThread.msleep(80)
        app.processEvents()
    win.grab().save(str(OUT / "shot_contacts_zoom.png"))
    print(f"wrote {OUT / 'shot_contacts_zoom.png'}")

win.close()
app.processEvents()
print("done")
