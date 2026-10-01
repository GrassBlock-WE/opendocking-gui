"""Does the product's own viewport actually *draw*, and can we prove it drew?

**The hole this closes.**

`odgui --check` has two stages. Stage 1 asks for a bare `QOpenGLContext`; stage 2
constructs the product's own `app.Viewport` and waits for the moderngl context
`Viewport.initializeGL` builds. Stage 2 is a real improvement over the stand-in
it replaced -- a bare `QOpenGLWidget` dies on its own
`initializeOpenGLFunctions()` on this machine (`0xC0000409`, after a perfectly
usable GL 4.6 context), so a probe built from one asserted a falsehood about a
viewer that renders; see `gl_route_probe_width.py` for the 16-variant study
that measured it.

But stage 2's readiness condition is `viewport._ctx is not None`, and
`Viewport.paintGL` *returns immediately* when `_ctx is None`. So a viewport that
holds a context and never yields a drawable framebuffer is reported by
`odgui --check` as a working viewer. That is the original defect's shape one
level down: a probe asserting a fact about the product that the product does not
corroborate. It is measured here, not argued about -- see the mutation below,
where `_ctx` is still `True` and the frame is a single colour.

So this file adds a third stage that does what a viewer does: it lays the
viewport out, forces a paint, reads the framebuffer back and reports **what is
in the frame** -- its dimensions, the number of distinct colours, and the number
of non-background pixels. The verdict is then written from a measurement of the
frame rather than from a context handle.

**What the numbers mean, stated once so they can be quoted.**

* **distinct colours** -- unique RGB triples in the read-back frame. This is the
  primary signal and the only one the verdict uses. A frame that `paintGL`
  skipped is *uniform*: `QOpenGLWidget`'s FBO is untouched, so it reads back as
  one colour (measured here: pure black, 1 distinct colour, 0 non-background
  pixels). A frame `paintGL` drew is not: the backdrop alone is a smooth
  gradient, measured at 1 111 distinct colours. The threshold is therefore
  `> 1`, which is structural rather than fitted -- `target.clear()` writes a
  constant to every pixel, so *any* draw, however small, makes the count exceed
  1, and a clear-only frame that scored 2 would be scoring a rendering artefact
  rather than content.
* **non-background pixels** -- pixels that are not the frame's single most
  common RGB triple. Reported for the human, **not** used by the verdict, and it
  needs its definition attached whenever it is quoted: the backdrop is a
  gradient, so in a healthy frame this is ~96% of the pixels (1 593 503 of
  1 661 550 here) and it is not a measure of "how much content is visible". A
  number quoted without the frame size and without this rule is not a fact about
  the frame, it is a fact about whoever wrote the rule.
* **frame dimensions** -- the read-back QImage's width and height in *device*
  pixels, which is not `Viewport.width()`: this machine's scale factor is 1.25,
  so a 1 294x858 holder yields a 1 590x1 045 frame. Quoting the logical size
  next to a device-pixel count is how a reader ends up convinced a frame is
  larger than the surface it came from.

**The mutation, run in both directions, every time.**

A guard that can only go green is not a guard. So this file runs the real
viewport *and* a deliberately neutered one, in separate child processes, and
fails unless the second is caught:

* `--mutate-no-draw` replaces `Viewport.paintGL` with a no-op, which is the
  code path that exists today, and the file requires the verdict to become
  `drew-nothing`. The mutation is applied to the **product's class** in the
  child, not to a stand-in: a guard that compared the probe against another
  stand-in would inherit the very bias it exists to catch.
* The unmutated run must stay green. Both directions are required -- a guard
  that flagged everything, and a guard that flagged nothing, are the same defect
  seen from opposite ends.

**Why this is a separate file and not a fourth `odgui --check` exit code.**

`odgui --check` answers a question about the *machine*: can this machine give Qt
an OpenGL context. Its whole vocabulary (0/3/4/5) is machine-capability facts,
and it is pinned in **three** places in `ci.yml` (line 163's `-ne 3`, the `case`
at 582-587, and the `want` dict at 613). "The context came up and the frame is
blank" is not a machine-capability fact -- it is identical on every machine and
is caused by the product's own `paintGL` (an FBO mismatch, a program that fails
to link, a `ctx.detect_framebuffer()` binding the wrong target). Folding it into
exit 5 would give one code two different owners, and folding it into 4 would
send a user to buy a GPU for a bug in this repository. Exit 5 is therefore left
alone -- it keeps meaning "the viewport never came up" -- and this state gets a
name and a code of its own, here.

**Exit codes** (this file's own vocabulary, deliberately not `odgui --check`'s):

* ``0`` -- every registered check passed or skipped, with each skip's reason
  printed. A skip is *not* a pass and says so: on a runner whose framebuffer
  cannot be read this file proves nothing, and a green tick must never be read
  as "the viewer was verified".
* ``1`` -- a check failed: the real viewport obtained a context and drew
  nothing, or the mutation was not caught.
* ``2`` -- did not finish. The sibling scripts in this directory use the same
  three, so a reader does not have to learn a new one.

**Totals stay constant.** `EXPECTED_CHECKS` is 5 in every environment: a check
that cannot run is recorded as a SKIP carrying its reason, never dropped. The
count is derived, not measured, for the reason `check_scripts_declare.py`
exists -- a total that moves with the machine is not a total.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: The one place in the tree that knows how to ask `odgui --check` what this
#: machine can do, and that owns the three-way pixel verdict. The decision is
#: shared; the frame *capture* is not, which is the split `_gui_check` documents
#: above `PIXEL_SKIP`.
from _gui_check import pixel_tag, verify_pixel_tag  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

#: How many checks this file records, in every environment.
#:
#:   2  the two frame measurements -- the real viewport, and the neutered one.
#:      Registered in **every** branch, because a SKIP is recorded exactly where
#:      a PASS would be: on a runner that cannot read a framebuffer neither of
#:      them can run, and a total that quietly lost two checks is a total that
#:      has stopped describing the file;
#: + 1  the static check that the child still asks the right questions, which
#:      needs no display and so never skips;
#: + 1  the reverse-verified three-way guard, which also needs no display.
#:
#: 2 + 1 + 1 = 4. It was 5 for one run, before the arithmetic was counted
#: against the file rather than against the intent: the three-way guard is one
#: of the two display-independent checks, not a third one on top of them.
EXPECTED_CHECKS = 4

#: More than one colour in the frame. See the module docstring: a clear-only
#: frame is exactly one colour, so this is the structural minimum for "something
#: was drawn" and not a number fitted to make a healthy frame pass.
MIN_DISTINCT_COLOURS = 2

#: How long the child waits for the product's context, and the parent's ceiling
#: on the whole child. Generous, because a driver that cannot supply a context
#: blocks inside `show()` for seconds and that wait belongs to Qt.
_PUMP_SECONDS = 20.0
_CHILD_TIMEOUT = 90.0

#: The holder size handed to the product's viewport. Fixed, so two runs on two
#: machines produce comparable frames; the device-pixel size that comes back is
#: reported next to it rather than substituted for it, because the scale factor
#: is a property of the machine (1.25 here, 1.0 on most CI runners) and is the
#: single easiest number to quote without its context.
#:
#: **1294x858 is the logical size and is not the frame.** On this machine the
#: frame read back from a holder of exactly this size is **1590x1045** -- 1 661
#: 550 pixels rather than 1 110 652. The two describe the same picture at two
#: scalings, and that is the whole hazard: two agents measuring "the frame" of
#: the same viewport published 1 593 503 non-background pixels against 1 110 652
#: and it took a while to see that neither was wrong, only that each had quoted a
#: different thing. Any count here is a count of *device* pixels; the width and
#: height in the payload are the frame's, and this tuple is not.
HOLDER_SIZE = (1294, 858)

RESULTS: list[tuple[str, str, str]] = []


def record(tag: str, name: str, reason: str = "") -> bool:
    RESULTS.append((tag, name, reason))
    print(f"  [{tag}] {name}" + (f"  -- {reason}" if reason else ""))
    return tag == "PASS"


def ok(name: str, reason: str = "") -> bool:
    return record("PASS", name, reason)


def bad(name: str, reason: str) -> bool:
    return record("FAIL", name, reason)


def skip(name: str, reason: str) -> bool:
    return record("SKIP", name, reason)


#: The child. A module-level string, exec'd by `subprocess`, for the reason
#: `launcher._PROBE_CHILD` is: asking Qt for a context is not a Python operation
#: that can fail -- on a machine that cannot supply one, `show()` blocks and the
#: driver ends the process with `0xC0000409` rather than raising -- so the
#: probe has to be able to die alone and be read from its exit status.
#:
#: `sys.argv[1]` selects the mutation, because the two directions must run in
#: separate processes: the mutation replaces a method on the product's class,
#: and one process cannot both mutate it and measure the unmutated viewport.
_CHILD = r'''
import json, sys, time

# Parsed as tokens, never as fixed indices. `python -c SRC a b` gives
# sys.argv == ['-c', 'a', 'b'], so the mutated run has one more argument than
# the unmutated one and every positional index shifts by one between them --
# which is how the unmutated run spent a run reading '1294x858' as a float.
# Scanning for the flags is immune to the count.
MUTATE = "mutate-no-draw" in sys.argv
PUMP = 20.0
# A fallback for a child run by hand. The real run is passed `size=...` by the
# parent from its HOLDER_SIZE, so this literal is never what a measured frame
# used -- and it is the *logical* size, not the frame: a holder of 1294x858 at
# this machine's 1.25 scale factor reads back as 1590x1045. See HOLDER_SIZE in
# the parent for why that difference has to be stated rather than discovered.
SIZE = "1294x858"
for tok in sys.argv[1:]:
    if tok.startswith("pump="):
        PUMP = float(tok.split("=", 1)[1])
    elif tok.startswith("size="):
        SIZE = tok.split("=", 1)[1]

started = time.perf_counter()
out = {"mutated": MUTATE}
W, H = (int(v) for v in SIZE.split("x"))

def emit(**kw):
    out.update(kw)
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()

try:
    import numpy as np
    from PyQt6 import QtCore, QtGui, QtWidgets
except Exception as exc:
    emit(harness=True, problem="the GL/PyQt6 stack did not import: %s" % exc)
    sys.exit(3)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[1:2])

try:
    import opendocking.workbench.app as appmod
    from opendocking.workbench.app import Viewport
except Exception as exc:
    emit(harness=True, problem="app.Viewport could not be imported: %s" % exc)
    sys.exit(3)

# Named in the output so every number printed by the parent says which tree it
# came from. The dev checkout and the installed wheel are several files apart on
# this machine, and a number without this line cannot be traced.
emit(tree=appmod.__file__, subject=Viewport.__module__ + "." + Viewport.__name__)

# The mutation: the product's own paintGL, made to return without drawing. This
# is the code path that exists today when a context came up but nothing was
# rendered, and it is the one this file has to be able to catch.
if MUTATE:
    Viewport.paintGL = lambda self: None
    emit(mutation="Viewport.paintGL replaced with a no-op")

holder = QtWidgets.QWidget()
holder.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
viewport = Viewport()
QtWidgets.QVBoxLayout(holder).addWidget(viewport)
holder.resize(W, H)
emit(platform=QtGui.QGuiApplication.platformName(), holder=[W, H])
holder.show()

# Readiness is the product's own condition: the moderngl context
# `Viewport.initializeGL` builds. Waiting on it is stage 2's job, and this file
# reports it rather than re-deciding it -- the point is what happens next.
deadline = time.perf_counter() + PUMP
while viewport._ctx is None and time.perf_counter() < deadline:
    app.processEvents()
    time.sleep(0.005)

context = viewport._ctx is not None
emit(context=context, settled=time.perf_counter() - started)

if not context:
    # Not a claim about the product: this machine never gave Qt the context, so
    # the frame is the only thing that could have answered, and it cannot.
    emit(frame="unreadable", reason="no context on this machine")
    sys.exit(0)

# Let the layout settle, then force a paint rather than waiting to be asked for
# one. `repaint()` is immediate and synchronous for a QOpenGLWidget, and
# `grabFramebuffer()` renders into the widget's own FBO and hands it back, so
# between them the frame below is a frame the product drew on request.
lay = holder.layout()
if lay is not None:
    lay.activate()
for _ in range(4):
    app.processEvents()
    time.sleep(0.15)
viewport.repaint()
for _ in range(2):
    app.processEvents()
    time.sleep(0.05)

image = viewport.grabFramebuffer()
if image is None or image.isNull():
    # The framebuffer could not answer. A null image is what a headless runner
    # hands back, and treating it as "drew nothing" is the mistake this tree's
    # three-way pixel verdict exists to prevent.
    emit(frame="null", reason="grabFramebuffer() returned a null image")
    sys.exit(0)

w, h = image.width(), image.height()
buf = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint8)
# ARGB32 at four bytes per pixel; bytesPerLine may be padded, so the row stride
# is taken from the image rather than assumed to be w*4.
rows = buf.reshape(h, image.bytesPerLine() // 4, 4)[:, :w, :3]
flat = rows.reshape(-1, 3)
codes, counts = np.unique(flat, axis=0, return_counts=True)
distinct = int(len(codes))
total = int(w * h)
modal = int(counts.max())
# "Non-background" = not the frame's single most common colour. Defined here
# rather than at the call site so the number is never quoted without its rule;
# see the module docstring for why the verdict does not use it.
emit(
    frame="read",
    width=w,
    height=h,
    total_px=total,
    distinct_colours=distinct,
    non_background_px=total - modal,
    modal_rgb=[int(v) for v in codes[int(counts.argmax())]],
    scale=round(float(viewport.devicePixelRatioF()), 4),
    elapsed=time.perf_counter() - started,
)
sys.exit(0)
'''


def _measure(mutate: bool) -> dict:
    """Run the child once and return its report, or `{}` if it never spoke.

    One child per measurement, always: the mutation replaces a method on the
    product's class, so a process that had already measured the real viewport
    would be measuring its own edit by the time it got to the second run.
    """
    argv = [sys.executable, "-c", _CHILD]
    if mutate:
        argv.append("mutate-no-draw")
    # Named flags, not positional: the child scans for them precisely so that
    # adding the mutation cannot shift the meaning of the ones after it.
    argv.append(f"pump={_PUMP_SECONDS}")
    argv.append("size=%dx%d" % HOLDER_SIZE)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_CHILD_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"harness": True, "problem": f"the child could not be run ({exc})"}

    report: dict = {}
    for line in (proc.stdout or "").splitlines():
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            report.update(loaded)
    if not report:
        return {
            "harness": True,
            "problem": (
                f"the child reported nothing (exit {proc.returncode}); "
                f"stderr tail: {(proc.stderr or '').strip().splitlines()[-1:] or ['(empty)']}"
            ),
        }
    report["exit"] = proc.returncode
    return report


def _can_answer(report: dict) -> bool:
    """Whether this machine's framebuffer produced a frame at all.

    Deliberately *not* `odgui --check` and not a context handle: this file
    measures what the disagreement between them costs, so asking either of them
    would assume the answer. The frame's own answer is the only admissible one,
    and `_gui_check.pixel_tag` records the three-way decision that follows.
    """
    return report.get("frame") == "read"


def _drew_something(report: dict) -> bool:
    """The one place the threshold is applied. Both frame checks use only this.

    Deliberately a single function. The two frame measurements ask the same
    question of two frames -- one that painted, one that did not -- and a guard
    that asked it twice could have one call site hardcoded to `True` and still
    look healthy, because the other call site would carry the whole burden of
    being wrong. Sharing the predicate means a verdict that is wrong in one
    direction is wrong in the other too, and
    `check_the_frame_verdict_has_three_answers` reverse-verifies *this* function
    in both directions on every run, so a call site cannot quietly stop using
    it either.
    """
    return int(report["distinct_colours"]) > MIN_DISTINCT_COLOURS


def check_the_viewport_draws() -> None:
    """The real product's viewport: a context is necessary, a frame is the point.

    This is the check the whole file exists for. Before it, nothing in the tree
    could distinguish "the viewer paints" from "the viewer holds a context": a
    viewport with a live context and a `paintGL` that returns immediately is
    what `odgui --check` calls exit 0.
    """
    report = _measure(mutate=False)
    if report.get("harness"):
        skip(
            "the product's viewport draws something",
            f"the probe could not run: {report.get('problem')}",
        )
        return
    if not _can_answer(report):
        skip(
            "the product's viewport draws something",
            f"this machine's framebuffer could not answer "
            f"(frame={report.get('frame')!r}: {report.get('reason') or 'no reason given'}), "
            f"so nothing was measured and nothing is claimed",
        )
        return

    drawn = _drew_something(report)
    # `pixel_tag` decides between PASS and FAIL only now that the frame has
    # answered; the context is reported as context, never as a stand-in for a
    # frame, which is the mistake this file exists to stop repeating.
    tag = pixel_tag(framebuffer_can_answer=True, condition=drawn)
    if tag == "PASS":
        ok(
            "the product's viewport draws something",
            f"{report['width']}x{report['height']} device px "
            f"(holder {report['holder'][0]}x{report['holder'][1]}, scale {report['scale']}), "
            f"{report['distinct_colours']} distinct colours, "
            f"{report['non_background_px']} of {report['total_px']} pixels not the "
            f"modal colour rgb{tuple(report['modal_rgb'])}, in "
            f"{report['elapsed']:.1f} s -- tree {report.get('tree', '?')}",
        )
        return
    bad(
        "the product's viewport draws something",
        f"the viewport built a moderngl context ({report.get('context')}) and the "
        f"frame is {report['width']}x{report['height']} with only "
        f"{report['distinct_colours']} distinct colour(s) and "
        f"{report['non_background_px']} non-background pixels. A context that "
        f"yields no drawable frame is the viewer opening empty: the remedy is "
        f"in `Viewport.paintGL`, not in the machine's OpenGL.",
    )


def check_a_blank_viewport_is_caught() -> None:
    """A viewport that draws nothing must go red. Both directions, every run.

    The mutation is the code path that exists today: `paintGL` returning while
    `_ctx` is live. If this check cannot fail, the first check above is measuring
    a number and not a fact, and nothing in the tree would notice.
    """
    report = _measure(mutate=True)
    if report.get("harness"):
        skip(
            "a viewport that draws nothing is detected",
            f"the mutated child could not run: {report.get('problem')}",
        )
        return
    if not _can_answer(report):
        skip(
            "a viewport that draws nothing is detected",
            f"this machine's framebuffer could not answer "
            f"(frame={report.get('frame')!r}), so the mutation could not be "
            f"judged here and no claim is made either way",
        )
        return

    distinct = int(report["distinct_colours"])
    non_bg = int(report["non_background_px"])
    # The same predicate as the healthy path, negated, plus the corroborating
    # zero. Requiring both means a frame that came back with a stray stray
    # colour but no content still counts as caught -- the pixel count is
    # reported either way, and neither number alone is the verdict.
    caught = (not _drew_something(report)) and non_bg == 0
    if caught:
        ok(
            "a viewport that draws nothing is detected",
            f"with paintGL neutered the viewport still held a context "
            f"({report.get('context')}) and the frame came back "
            f"{report['width']}x{report['height']} with {distinct} distinct "
            f"colour(s) and {non_bg} non-background pixels -- detected, which is "
            f"a state no check in this tree could see before",
        )
        return
    bad(
        "a viewport that draws nothing is detected",
        f"with paintGL replaced by a no-op the frame still read back "
        f"{distinct} distinct colours and {non_bg} non-background pixels, so a "
        f"blank viewer is indistinguishable from a working one. The guard this "
        f"check implements is not measuring the frame.",
    )


def check_the_probe_asks_the_right_questions() -> None:
    """The child must build the product's viewport, paint it and read it back.

    The static half of a guard whose live half can only go red on a machine
    nobody has once the bug is fixed. Reading the child's source is not a proxy
    for running it and is not claimed to be: the point is to notice a *regression*
    in what the probe asks, cheaply, on any machine. A probe narrowed back to a
    `QOpenGLWidget` stand-in, or one that stops forcing a paint, or one that
    infers its verdict from `_ctx`, is the defect this file was written to end --
    and the first of those is exactly how it started.

    **The child's code is parsed, not its text searched.** The first version of
    this check was `"repaint()" not in _CHILD`, and a mutation that replaced the
    call with a comment *mentioning* it satisfied the substring -- the guard
    reported a probe that no longer forces a paint as healthy. That is the same
    defect `_gui_check._helper_calls_are_live` was written for, in a new place,
    and the mutation that found it is the reason this parses. A call inside a
    literal-false branch is also rejected, so a call cannot be neutered in place
    the way that file documents.
    """
    import ast

    try:
        tree = ast.parse(_CHILD)
    except SyntaxError as exc:
        bad(
            "the probe paints the product's viewport and reads it back",
            f"the child does not parse ({exc}), so what the probe would do cannot "
            f"be inspected at all",
        )
        return

    called: set[str] = set()
    reported: set[str] = set()
    names: set[str] = set()
    dead_depth = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
                if func.id == "emit":
                    for kw in node.keywords:
                        if kw.arg:
                            reported.add(kw.arg)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)

    problems = []
    if "Viewport" not in called:
        problems.append(
            "the child never constructs app.Viewport, so it judges a stand-in "
            "rather than the widget the workbench shows"
        )
    if "initializeOpenGLFunctions" in names:
        problems.append(
            "the child requires initializeOpenGLFunctions(), which is the "
            "measured cause of the original false 'opens empty' verdict: it "
            "ends the process with 0xC0000409 on a machine where the product "
            "renders (gl_route_probe_width.py)"
        )
    if "repaint" not in called:
        problems.append(
            "the child never forces a paint, so it would report whatever the "
            "widget happened to hold rather than a frame the product drew"
        )
    if "grabFramebuffer" not in called:
        problems.append(
            "the child never reads the framebuffer back, so there is no "
            "measurement to write the verdict from"
        )
    for needed in ("distinct_colours", "non_background_px", "width", "height"):
        if needed not in reported:
            problems.append(f"the child does not report {needed} in its payload")
    if problems:
        bad("the probe paints the product's viewport and reads it back", "; ".join(problems))
        return
    ok(
        "the probe paints the product's viewport and reads it back",
        "parsed from the child's code (not its text): it constructs "
        "app.Viewport, calls repaint() and grabFramebuffer(), and reports "
        "width, height, distinct_colours and non_background_px",
    )


def check_the_frame_verdict_has_three_answers() -> None:
    """`pixel_tag` still separates "cannot answer" from "answered no".

    Reverse-verified on every run, because a guard that skipped unconditionally
    would be indistinguishable from a working one. This check is what stops this
    file from becoming `viewport_framing_check.py`'s defect in a new place: a
    suite that reports "nothing was drawn" for a runner that drew nothing
    readable has told the reader their viewer is broken.
    """
    held, reason = verify_pixel_tag()
    if not held:
        bad("a frame that cannot answer is never reported as a blank viewer", reason)
        return
    # The shared predicate, reverse-verified in *both* directions on the real
    # numbers rather than on a hand-written table. A predicate hardcoded to one
    # answer is the one mutation that would make both frame checks agree with
    # each other and be wrong together, and it is invisible to either of them
    # individually -- which is precisely why it is checked here instead.
    drawn_frame = {"distinct_colours": MIN_DISTINCT_COLOURS + 1}
    blank_frame = {"distinct_colours": MIN_DISTINCT_COLOURS}
    got = (_drew_something(drawn_frame), _drew_something(blank_frame))
    if got != (True, False):
        bad(
            "a frame that cannot answer is never reported as a blank viewer",
            f"{reason}; and the frame predicate answered {got} for a drawn and a "
            f"blank frame, where it must answer (True, False) around the "
            f"threshold of {MIN_DISTINCT_COLOURS}",
        )
        return
    ok(
        "a frame that cannot answer is never reported as a blank viewer",
        f"{reason}; and the frame predicate answers (True, False) either side of "
        f"the {MIN_DISTINCT_COLOURS}-colour threshold",
    )


def main() -> int:
    print("Does the product's own viewport draw? (stage 3 -- measured, not inferred)\n")
    print(f"threshold: > {MIN_DISTINCT_COLOURS} distinct colour in the read-back frame")
    print("non-background = pixels that are not the frame's most common RGB triple;")
    print("  it is reported for the human and is NOT what the verdict uses\n")

    check_the_probe_asks_the_right_questions()
    check_the_frame_verdict_has_three_answers()
    check_the_viewport_draws()
    check_a_blank_viewport_is_caught()

    npass = sum(1 for tag, _, _ in RESULTS if tag == "PASS")
    nfail = sum(1 for tag, _, _ in RESULTS if tag == "FAIL")
    nskip = sum(1 for tag, _, _ in RESULTS if tag == "SKIP")
    print()
    print(f"--- summary: {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")

    if len(RESULTS) != EXPECTED_CHECKS:
        print(f"    ::error::recorded {len(RESULTS)} checks but EXPECTED_CHECKS is "
              f"{EXPECTED_CHECKS}; the total must be the same on every machine, so a "
              f"skipped check is a bug here rather than a smaller run")
        return EXIT_FAILED
    if nfail:
        return EXIT_FAILED
    if nskip:
        print("    NOTE: a skip is not a pass. This run did NOT verify that the "
              "viewer paints; the framebuffer could not answer here.")
        return EXIT_OK
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("\nRESULT: DID NOT FINISH -- see the traceback above")
        raise SystemExit(EXIT_INCOMPLETE)
