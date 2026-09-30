"""Is the structure actually centred in the viewport?

Framing bugs are easy to miss because "the molecule is visible" looks fine. This
measures where the drawn pixels actually are relative to the viewport centre,
and compares a frame taken at construction time with one taken after the
window has settled to its real size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402


def pixels(win):
    win.viewport.repaint()
    QtWidgets.QApplication.processEvents()
    img = win.viewport.grabFramebuffer().convertToFormat(
        QtGui.QImage.Format.Format_RGB32
    )
    p = img.constBits()
    p.setsize(img.sizeInBytes())
    a = np.frombuffer(p, dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    a = a[:, : img.width(), :3].astype(np.int16)
    flat = a.reshape(-1, 3)
    key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    modal = vals[int(np.argmax(counts))]
    bg = np.asarray([(modal >> 16) & 0xFF, (modal >> 8) & 0xFF, modal & 0xFF])
    mask = np.abs(a - bg).sum(axis=2) > 12
    # Compare against the *grabbed* size, not the widget's logical size:
    # `grabFramebuffer()` is in device pixels, and this machine runs a device
    # pixel ratio of 1.25, so the two differ by a quarter. Using the logical
    # size reports a 25% phantom offset that does not exist.
    return mask, img.width(), img.height()


def report(label, mask, w, h):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        print(f"  {label}: nothing drawn")
        return
    cx, cy = xs.mean(), ys.mean()
    ex, ey = w / 2.0, h / 2.0
    dx, dy = cx - ex, cy - ey
    print(
        f"  {label}: image {w}x{h}  drawn centroid ({cx:6.1f}, {cy:6.1f})  "
        f"centre ({ex:6.1f}, {ey:6.1f})  offset ({dx:+6.1f}, {dy:+6.1f}) px "
        f"= ({dx / ex:+.1%}, {dy / ey:+.1%})"
    )
    print(
        f"        drawn spans x [{xs.min()}, {xs.max()}]  y [{ys.min()}, {ys.max()}]"
    )
    clipped = xs.max() >= w - 2 or xs.min() <= 1 or ys.max() >= h - 2 or ys.min() <= 1
    print(f"        touching an edge: {'YES' if clipped else 'no'}")


app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
win = MainWindow(
    receptor=EX / "rec_prep.pdbqt",
    ligand=EX / "ibuprofen_prep.pdbqt",
    poses=EX / "poses.pdbqt",
)
win.show()
for _ in range(30):
    app.processEvents()

failures: list[str] = []


def expect(label, mask, w, h, tol=0.02):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        failures.append(f"{label}: nothing was drawn at all")
        print(f"  [FAIL] {label}: nothing drawn")
        return
    ex, ey = w / 2.0, h / 2.0
    dx, dy = (xs.mean() - ex) / ex, (ys.mean() - ey) / ey
    ok = abs(dx) <= tol and abs(dy) <= tol
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {label}: offset ({dx:+.1%}, {dy:+.1%}) "
        f"of the {w}x{h} image (tolerance {tol:.0%})"
    )
    if not ok:
        failures.append(f"{label}: drawn content is off-centre by ({dx:+.1%}, {dy:+.1%})")
    # The 22 A search box is wider than the molecule, so content reaching the
    # edges is expected. What must never happen is content spilling *outside*
    # the image, which would mean the viewport was set from the wrong size.
    if xs.max() > w - 1 or xs.min() < 0 or ys.max() > h - 1 or ys.min() < 0:
        failures.append(f"{label}: drawn outside the framebuffer")


print("\n=== as the window first appears (frame taken during __init__) ===")
expect("initial", *pixels(win))

print("\n=== after the window settles to 1280x820 ===")
win.resize(1280, 820)
for _ in range(30):
    app.processEvents()
print(f"  logical viewport {win.viewport.width()}x{win.viewport.height()}, "
      f"devicePixelRatio {win.viewport.devicePixelRatioF()}")
expect("before reframe", *pixels(win))
win.viewport.frame_all()
for _ in range(10):
    app.processEvents()
expect("after reframe", *pixels(win))

win.close()

print(f"\n  {len(failures)} failure(s)")
for f in failures:
    print(f"    {f}")
sys.exit(1 if failures else 0)
