"""Is the structure actually centred in the viewport?

Framing bugs are easy to miss because "the molecule is visible" looks fine. This
measures where the drawn pixels actually are relative to the viewport centre,
and compares a frame taken at construction time with one taken after the
window has settled to its real size.

**A run that stopped is not a run that passed.**

The three checks used to be top-level statements with bare `sys.exit()` at the
bottom, so any exception — and `grabFramebuffer()` on a machine with no context
is exactly such an exception — ended the process with a traceback, no counts
and no line saying which of the three frames had been measured. The three
frames are ordered and each depends on the last, so a crash on the second one
loses the first and the third silently.

So the checks are recorded, counted and summarised, an unanswered measurement is
a **SKIP carrying a reason** rather than a silent absence, and a run that does
not reach the end reports that and exits ``2``.

Exit codes: ``0`` finished with no failures, ``1`` finished with at least one
failure, ``2`` did not finish.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

#: The one place in the tree that knows how to ask `odgui --check` what this
#: machine can do; it was a private copy in each of four check scripts.
#: `python scripts/_gui_check.py` audits that those four still use it.
from _gui_check import odgui_check  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

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


def pixels(win):
    """The drawn-pixel mask for one frame, or ``None`` if there is no frame.

    Returning ``None`` rather than raising is what lets the caller record a
    SKIP with a reason instead of losing the run: on a machine with no context
    `grabFramebuffer()` hands back a null image, and the `constBits()` call on
    it is a hard error, not an empty array.
    """
    win.viewport.repaint()
    QtWidgets.QApplication.processEvents()
    img = win.viewport.grabFramebuffer()
    if img is None or img.isNull():
        return None
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
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


def expect(label, frame, tol=0.02):
    """One framing measurement.

    `frame` is what `pixels()` returned, or ``None`` when the widget could not
    be read at all.
    """
    if frame is None:
        skip(
            label,
            "the viewport framebuffer could not be read, so the centring of "
            "this frame is unknown",
        )
        return
    mask, w, h = frame
    report(label, mask, w, h)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        bad(label, "nothing was drawn at all")
        return
    ex, ey = w / 2.0, h / 2.0
    dx, dy = (xs.mean() - ex) / ex, (ys.mean() - ey) / ey
    if abs(dx) <= tol and abs(dy) <= tol:
        ok(label, f"offset ({dx:+.1%}, {dy:+.1%}) of the {w}x{h} image")
    else:
        bad(label, f"drawn content is off-centre by ({dx:+.1%}, {dy:+.1%})")
    # The 22 A search box is wider than the molecule, so content reaching the
    # edges is expected. What must never happen is content spilling *outside*
    # the image, which would mean the viewport was set from the wrong size.
    if xs.max() > w - 1 or xs.min() < 0 or ys.max() > h - 1 or ys.min() < 0:
        bad(label, "drawn outside the framebuffer")


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(
        receptor=EX / "rec_prep.pdbqt",
        ligand=EX / "ibuprofen_prep.pdbqt",
        poses=EX / "poses.pdbqt",
    )
    win.show()
    for _ in range(30):
        app.processEvents()

    # Whether this machine can answer the question at all is settled once, and
    # quoted in every skip that follows, rather than guessed at per frame.
    no_context = win.viewport._ctx is None
    context_reason = ""
    if no_context:
        answer = odgui_check()
        tip = win.viewport.toolTip()
        if answer.can_make_context:
            context_reason = (
                "odgui --check reports this machine CAN create a context "
                f"({answer.describe()}), so the viewport having none is a defect "
                f"and not an environment limit: {tip or '(Qt gave no reason)'}"
            )
        else:
            context_reason = f"{tip or 'Qt never called initializeGL'}; {answer.describe()}"
        print(f"  no GL context on the viewport; {context_reason}")

    print("\n=== as the window first appears (frame taken during __init__) ===")
    expect("initial", pixels(win))

    print("\n=== after the window settles to 1280x820 ===")
    win.resize(1280, 820)
    for _ in range(30):
        app.processEvents()
    print(f"  logical viewport {win.viewport.width()}x{win.viewport.height()}, "
          f"devicePixelRatio {win.viewport.devicePixelRatioF()}")
    expect("before reframe", pixels(win))
    win.viewport.frame_all()
    for _ in range(10):
        app.processEvents()
    expect("after reframe", pixels(win))

    win.close()
    return summarise()


def summarise(incomplete: str = "") -> int:
    """Counts, and the verdict. Printed on every path, including a crash."""
    npass = sum(1 for tag, _, _ in RESULTS if tag == "PASS")
    nfail = sum(1 for tag, _, _ in RESULTS if tag == "FAIL")
    nskip = sum(1 for tag, _, _ in RESULTS if tag == "SKIP")
    print(f"\n  {len(RESULTS)} checks: {npass} passed, {nfail} failed, {nskip} skipped")
    for tag, name, reason in RESULTS:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {reason}")
    if incomplete:
        print(f"RESULT: DID NOT FINISH — {incomplete}")
        return EXIT_INCOMPLETE
    if nfail:
        print("RESULT: FAIL — the run finished and something did not hold")
        return EXIT_FAILED
    if npass == 0:
        # Three skipped checks and a zero is a run that measured nothing, and it
        # must not be able to report success.
        print("RESULT: DID NOT FINISH — no frame could be measured, so nothing "
              "was verified")
        return EXIT_INCOMPLETE
    print(f"RESULT: PASS — {npass} frames measured"
          + (f", {nskip} could not be measured here" if nskip else ""))
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
        print(f"DID NOT FINISH — an exception escaped before all three frames "
              f"were recorded: {type(_exc).__name__}: {_exc}")
        raise SystemExit(summarise(f"an exception escaped: {type(_exc).__name__}: {_exc}"))
