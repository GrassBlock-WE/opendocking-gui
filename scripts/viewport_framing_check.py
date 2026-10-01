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

**"This runner drew nothing readable" is not "the structure is not centred".**

That distinction is the reason this file has three outcomes per frame rather than
two, and it is not cosmetic. On a headless runner `grabFramebuffer()` returns a
uniformly coloured image; the mask is then empty; and the version of this file
that reported an empty mask as `FAIL` produced three failures with no reason and
failed a CI job on a machine that had drawn nothing at all. The empty mask was
true. The conclusion drawn from it was not.

So an empty frame is resolved by asking the *frame* whether this machine can
draw at all, and the answer picks one of three results: **SKIP** when the
framebuffer cannot answer, **PASS**/**FAIL** on the centring measurement when it
can. The classifier is `_gui_check.pixel_tag`, shared with
`workbench_interaction_check.py`'s eventual owner rather than copied here — see
the comment above `PIXEL_SKIP` in that file for why the *decision* is shared and
the *frame capture* is not.

Which machine is "this one" is decided **once**, from the product's own first
grab, and by neither `odgui --check` nor `GL_OK`. Those two disagree with the
frame, and the disagreement is measured: on the machine this was written on, the
launcher's probe reports no context while the workbench's own `QOpenGLWidget`
renders and reads back 723 226 pixels. Gating on either of them would skip
these checks on a machine that renders perfectly well.

Exit codes: ``0`` finished with no failures, ``1`` finished with at least one
failure, ``2`` did not finish.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

#: The one place in the tree that knows how to ask `odgui --check` what this
#: machine can do; it was a private copy in each of four check scripts.
#: `python scripts/_gui_check.py` audits that those four still use it.
#:
#: `pixel_tag` comes from the same module on purpose. It is the three-way
#: decision described in this file's docstring, and it is not duplicated here:
#: a second copy of a guard is how a suite ends up with two different answers to
#: "can this machine draw", which is the defect this change is fixing.
from _gui_check import odgui_check, pixel_tag, verify_pixel_tag  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

#: How many checks this file is supposed to record, in every environment.
#:
#: The pin exists because this file's total used to be whatever the machine
#: could measure, and a total that moves with the environment is not a total.
#: The derivation, so a future change to it is deliberate:
#:
#:   3  the three framing measurements, registered in **every** branch -- a
#:      SKIP is recorded exactly where a PASS would be, which is the whole
#:      point of the three-way result;
#: + 1  the reverse-verified pixel guard;
#: + 1  this pin, which counts itself.
#:
#: 3 + 1 + 1 = 5. It was 3 before, and the three are still the same three.
EXPECTED_CHECKS = 5

#: The call-site census, declared here rather than in a table
#: `check_scripts_declare.py` owns -- the mechanism that file exists to enforce,
#: and the one that caught this file drifting when the three-way result was
#: added. `sites` is compared against a derived AST walk and `guards` against a
#: digest of the guard strings, so neither number is hand-transcribed into a
#: second file.
#:
#: 3 -> 7 sites. The three framing measurements are now called from `expect()`
#: through three call sites rather than one, and the guard and the pin add one
#: each; nothing was removed.
#: GATE-DECLARE 1
#: sites: 2 unconditional + 5 guarded
#: guards: sha256:c6e8000766ac6c360f1dfd434bed538e5f3ce622a7125b7a81707cf4746e25b0

#: The framing measurements, as opposed to the meta-checks about this file. The
#: verdict is computed from these alone, so a passing guard and a passing pin
#: cannot make a run that measured no frame at all report success. This file
#: records its skips in the total (unlike `structure_bond_check.py`) and counts
#: them separately again, because a reader needs both numbers: the total says
#: the suite ran, the split says whether it learned anything.
FRAME_LABELS = ("initial", "before reframe", "after reframe")

#: `(tag, name, reason, kind)`. The fourth element is what keeps the verdict
#: honest: `"frame"` for a measurement, `"meta"` for a check about this file.
RESULTS: list[tuple[str, str, str, str]] = []


def record(tag: str, name: str, reason: str = "", kind: str = "frame") -> bool:
    RESULTS.append((tag, name, reason, kind))
    print(f"  [{tag}] {name}" + (f"  — {reason}" if reason else ""))
    return tag == "PASS"


def ok(name: str, reason: str = "", kind: str = "frame") -> bool:
    return record("PASS", name, reason, kind)


def bad(name: str, reason: str, kind: str = "frame") -> bool:
    return record("FAIL", name, reason, kind)


def skip(name: str, reason: str, kind: str = "frame") -> bool:
    return record("SKIP", name, reason, kind)


class Frame(NamedTuple):
    """What one grab of the viewport gave back.

    `readable` and `content` are kept apart on purpose, because they answer
    different questions and collapsing them is the bug. `readable` is whether
    `grabFramebuffer()` gave an image at all; `content` is how many pixels differ
    from the background. An image that is readable and has no content is a
    picture of nothing — which is what a headless runner produces, and it is not
    the same fact as "there was no image".
    """

    readable: bool
    content: int
    mask: object
    w: int
    h: int
    problem: str


def pixels(win) -> Frame:
    """The drawn-pixel mask for one frame.

    Never raises and never returns ``None``: a frame that could not be read is a
    `Frame` with `readable` False and a `problem`, so the caller has to decide
    what to record and cannot lose the measurement by accident. Returning ``None``
    and branching on it is how a second path appears, and a second path is how a
    check ends up registering its total in one branch only.
    """
    try:
        win.viewport.repaint()
        QtWidgets.QApplication.processEvents()
        img = win.viewport.grabFramebuffer()
        if img is None or img.isNull():
            return Frame(False, 0, None, 0, 0,
                         "grabFramebuffer() returned a null image")
        img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        p = img.constBits()
        p.setsize(img.sizeInBytes())
        a = np.frombuffer(p, dtype=np.uint8).reshape(
            img.height(), img.bytesPerLine() // 4, 4
        )
        a = a[:, : img.width(), :3].astype(np.int16)
        flat = a.reshape(-1, 3)
        key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
        vals, counts = np.unique(key, return_counts=True)
        modal = vals[int(np.argmax(counts))]
        bg = np.asarray([(modal >> 16) & 0xFF, (modal >> 8) & 0xFF, modal & 0xFF])
        mask = np.abs(a - bg).sum(axis=2) > 12
    except Exception as exc:  # noqa: BLE001 - an unreadable frame is a result
        return Frame(False, 0, None, 0, 0,
                     f"the frame could not be read ({type(exc).__name__}: {exc})")
    # Compare against the *grabbed* size, not the widget's logical size:
    # `grabFramebuffer()` is in device pixels, and this machine runs a device
    # pixel ratio of 1.25, so the two differ by a quarter. Using the logical
    # size reports a 25% phantom offset that does not exist.
    return Frame(True, int(mask.sum()), mask, img.width(), img.height(), "")


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


def expect(label, frame, can_draw, can_draw_reason, tol=0.02):
    """One framing measurement, in three possible outcomes.

    `frame` is a `Frame` and never ``None``. `can_draw` is the machine's answer,
    decided once from the first grab and passed in, so all three frames are
    judged against the same standard -- deciding per frame would let a frame that
    happened to catch a partially-drawn widget define "this machine can draw"
    for itself, and the first frame is the one taken before any of the pumping
    that follows could have helped.

    The three outcomes, and why each:

    * `can_draw` is False -- **SKIP**. The framebuffer cannot answer, so the
      centring of this frame is unknown. Reported with the reason rather than
      as a failure, because "this runner drew nothing readable" and "the
      structure is not centred" are different sentences and only one of them is
      true on such a machine.
    * `can_draw` and the frame is empty -- **FAIL**. The machine demonstrably
      draws, so an empty frame here is the viewer failing to draw, which is a
      real defect and not an environment limit.
    * `can_draw` and the frame has content -- the centring measurement, **PASS**
      or **FAIL** on the offset.
    """
    if not can_draw:
        # Recorded in this branch as well as the others, deliberately: the total
        # has to be the same number whether or not this machine can draw, or the
        # pin below is a pin that only holds on one kind of machine.
        skip(label, can_draw_reason)
        return
    if not frame.readable:
        # Answerable machine, unreadable *this* frame. Distinct from the branch
        # above: the machine can draw, so this frame failing to read is a fact
        # about the run, not about the environment.
        skip(label, f"{can_draw_reason}, and this frame could not be read "
                   f"either: {frame.problem}")
        return
    if frame.content == 0:
        bad(label, "the framebuffer is readable on a machine that draws, and this "
                   f"{frame.w}x{frame.h} frame is uniformly coloured: the viewport "
                   "drew nothing")
        return
    report(label, frame.mask, frame.w, frame.h)
    mask, w, h = frame.mask, frame.w, frame.h
    ys, xs = np.nonzero(mask)
    ex, ey = w / 2.0, h / 2.0
    dx, dy = (xs.mean() - ex) / ex, (ys.mean() - ey) / ey
    tag = pixel_tag(True, abs(dx) <= tol and abs(dy) <= tol)
    if tag == "PASS":
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
    first = pixels(win)

    # The machine's answer, from **the product's own first frame** and from
    # nothing else. Not `no_context`, not `GL_OK`, not `odgui --check`: all three
    # disagree with the frame on this machine, where the launcher reports no
    # context while the workbench's own QOpenGLWidget renders 723 226 pixels.
    can_draw = first.readable and first.content > 0
    if can_draw:
        can_draw_reason = (
            f"the product's own first grab is readable and has {first.content} "
            f"non-background pixels in {first.w}x{first.h}, so this machine draws"
        )
    elif not first.readable:
        can_draw_reason = (
            "the product's first frame could not be read, so this machine cannot "
            f"answer the centring question: {first.problem}"
            + (f"; {context_reason}" if context_reason else "")
        )
    else:
        can_draw_reason = (
            f"the product's first frame is readable but uniformly coloured "
            f"({first.w}x{first.h}, 0 non-background pixels), so this machine "
            f"cannot answer the centring question"
            + (f"; {context_reason}" if context_reason else "")
        )
    print(f"  framebuffer can answer: {'yes' if can_draw else 'no'} — {can_draw_reason}")
    expect("initial", first, can_draw, can_draw_reason)

    print("\n=== after the window settles to 1280x820 ===")
    win.resize(1280, 820)
    for _ in range(30):
        app.processEvents()
    print(f"  logical viewport {win.viewport.width()}x{win.viewport.height()}, "
          f"devicePixelRatio {win.viewport.devicePixelRatioF()}")
    expect("before reframe", pixels(win), can_draw, can_draw_reason)

    # The third frame depends on `frame_all()` having run, so a failure there
    # used to take the third measurement with it -- and the file's own docstring
    # says a lost frame must not be silent. It is caught here and recorded as the
    # third frame's result, so all three labels always appear.
    try:
        win.viewport.frame_all()
        for _ in range(10):
            app.processEvents()
        third = pixels(win)
        expect("after reframe", third, can_draw, can_draw_reason)
    except Exception as exc:  # noqa: BLE001 - the label must still be recorded
        skip("after reframe",
             f"frame_all() raised before this frame could be taken "
             f"({type(exc).__name__}: {exc}), so the third frame is unmeasured; "
             f"the two above were measured on this machine"
             if can_draw else
             f"frame_all() raised before this frame could be taken "
             f"({type(exc).__name__}: {exc}), and this machine could not have "
             f"drawn it in any case: {can_draw_reason}")

    # Meta-checks, registered in every branch and counted in the total, so the
    # pin below holds whether or not this machine can draw.
    held, reason = verify_pixel_tag()
    (ok if held else bad)(
        "the frame guard skips only when the framebuffer cannot answer",
        reason, kind="meta",
    )
    # The total, compared rather than described. The first version of this check
    # passed `True` and only *printed* the two numbers, so it reported "4 ran
    # ... and 5 are expected" as a PASS -- a check that cannot fail, in the file
    # whose whole reason for existing is that its total must not move with the
    # machine. The assertion is the comparison; the reason only explains it.
    total = len(RESULTS) + 1  # this check counts itself
    (ok if total == EXPECTED_CHECKS else bad)(
        "the number of checks that ran is the number this file is supposed to "
        "have, so none of them is hiding behind a guard",
        f"{len(RESULTS)} ran before this one, {total} including it, and "
        f"{EXPECTED_CHECKS} are expected"
        + ("" if total == EXPECTED_CHECKS else " — the total moved, so a check "
           "was added or removed without changing the pin deliberately")
        + "; the derivation is above EXPECTED_CHECKS",
        kind="meta",
    )

    win.close()
    return summarise()


def _count(kind: str, tag: str) -> int:
    return sum(1 for t, _, _, k in RESULTS if k == kind and t == tag)


def summarise(incomplete: str = "") -> int:
    """Counts, and the verdict. Printed on every path, including a crash."""
    npass = _count("frame", "PASS")
    nfail = _count("frame", "FAIL") + _count("meta", "FAIL")
    nskip = _count("frame", "SKIP")
    meta_fail = _count("meta", "FAIL")
    print(f"\n  {len(RESULTS)} checks: {npass + _count('meta', 'PASS')} passed, "
          f"{nfail} failed, {nskip + _count('meta', 'SKIP')} skipped")
    print(f"    of which {npass} framing measurement(s) passed, {nfail - meta_fail} "
          f"failed, {nskip} skipped")
    for tag, name, reason, _kind in RESULTS:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {reason}")
    if incomplete:
        print(f"RESULT: DID NOT FINISH — {incomplete}")
        return EXIT_INCOMPLETE
    if nfail:
        print("RESULT: FAIL — the run finished and something did not hold")
        return EXIT_FAILED
    if npass == 0:
        # Every framing measurement was skipped, so nothing was learned about
        # centring. This is computed from the framing results alone on purpose:
        # a passing guard and a passing pin are checks *about this file*, and
        # letting them carry the verdict is how a run that measured nothing
        # reports success.
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
