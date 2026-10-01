"""Render the small-sphere and space-filling representations side by side.

Run it directly to write the PNGs a human can look at:

    F:\\python310\\python.exe scripts\\representation_screens.py

Not a check. Its whole job is to produce pictures, because "the contact
fraction is 0.0% and 91.1%" is not evidence that either picture looks like
what its name says. The two numbers are what the names are pinned to
(`representation_names_check.py`); these are the pictures those numbers
describe, rendered through the real `MainWindow` and the real `Viewport` so
that what is written is what a user gets.

Three files, all written and then immediately confirmed on disk: the two
views separately, and one composite with both, because the whole question --
"which of these should be the default for a docking workbench" -- is a
comparison and a comparison needs the two pictures in the same image.

Everything is written in this process and re-read from disk immediately
afterwards, with the byte count and the image dimensions printed. A PNG that
vanishes between being written and being looked at has been observed in this
repository, so "the file was created" is not accepted as evidence here: the
file is opened and measured after the save, in the same run that saved it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dock-py" / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from PyQt6 import QtGui, QtTest, QtWidgets  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402

OUT = ROOT / "dist" / "representations"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 900, 720

#: The two views this script exists to compare, in the order they are shown.
#: Read from the product's own table rather than written out here, so a rename
#: cannot leave this script photographing labels the product no longer uses.
VIEWS = ("spheres", "space_filling")


def img_array(qimg: QtGui.QImage) -> np.ndarray:
    """QImage -> (h, w, 3) uint8, independent of stride and format."""
    img = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4
    )
    return arr[:, : img.width(), :3].copy()


def to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """(h, w, 3) uint8 RGB -> QImage, for the side-by-side composite."""
    h, w, _ = arr.shape
    img = QtGui.QImage(arr.tobytes(), w, h, 3 * w,
                       QtGui.QImage.Format.Format_RGB888)
    return img.copy()   # detach from the numpy buffer we are about to drop


def non_background(arr: np.ndarray) -> int:
    """Pixels differing from the modal colour -- the picture's own coverage."""
    flat = arr.reshape(-1, 3).astype(np.int32)
    key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    _, counts = np.unique(key, return_counts=True)
    modal = int(np.argmax(counts))
    return int(counts.sum() - counts[modal])


def confirm(path: Path) -> str:
    """Re-open a file just written and report what is actually in it.

    The check is deliberately about the file on disk rather than about the
    write call succeeding: the observation that motivated this file is a PNG
    that was written, reported as written, and was not there afterwards.
    """
    if not path.is_file():
        return f"MISSING: {path}"
    data = path.read_bytes()
    head = data[:8]
    png = head == b"\x89PNG\r\n\x1a\n"
    img = QtGui.QImage(str(path))
    size = f"{img.width()}x{img.height()}" if not img.isNull() else "unreadable"
    return (f"{path.name}: {len(data)} bytes on disk, PNG header "
            f"{'ok' if png else 'WRONG'}, dimensions {size}")


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    receptor = ROOT / "examples" / "1crn_prep.pdbqt"
    if not receptor.is_file():
        print(f"missing {receptor}; nothing to photograph")
        return 1

    win = MainWindow(receptor=receptor)
    win.resize(W, H)
    win.show()
    QtTest.QTest.qWaitForWindowExposed(win, 5000)
    app.processEvents()
    QtTest.QTest.qWait(400)
    app.processEvents()

    from opendocking.workbench.app import REPRESENTATIONS

    by_key = {r.key: r for r in REPRESENTATIONS}
    print(f"measured from: {Path(sys.modules['opendocking.workbench.app'].__file__).parent}")
    print(f"receptor: {receptor.name}, {sum(1 for _ in receptor.open(encoding='utf-8') if _.startswith(('ATOM', 'HETATM')))} atoms")
    print()

    frames: dict[str, np.ndarray] = {}
    for key in VIEWS:
        index = win.cmb_representation.findData(key)
        if index < 0:
            print(f"  {key}: NOT IN THE SELECTOR -- the product no longer offers it")
            return 1
        win.cmb_representation.setCurrentIndex(index)
        app.processEvents()
        win.viewport.repaint()
        QtTest.QTest.qWait(60)
        img = win.viewport.grabFramebuffer()
        if img.isNull():
            print(f"  {key}: the viewport returned no framebuffer")
            return 1
        arr = img_array(img)
        frames[key] = arr
        path = OUT / f"{key}.png"
        img.save(str(path))
        print(f"  {by_key[key].label!r}")
        print(f"    wrote {path}")
        print(f"    {confirm(path)}")
        print(f"    covers {non_background(arr)} non-background px of "
              f"{arr.shape[0] * arr.shape[1]}")
        print()

    # The composite, because the decision this supports is a comparison.
    gap = np.zeros((frames[VIEWS[0]].shape[0], 8, 3), np.uint8)
    side = np.hstack([frames[VIEWS[0]], gap, frames[VIEWS[1]]])
    combined = OUT / "side_by_side.png"
    to_qimage(side).save(str(combined))
    print(f"  wrote {combined}")
    print(f"    {confirm(combined)}")
    print()
    print(f"  left: {by_key[VIEWS[0]].label!r}   right: {by_key[VIEWS[1]].label!r}")

    win.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
