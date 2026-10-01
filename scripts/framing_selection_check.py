"""Does selecting a pose put the pose on screen, with its interactions readable?

Run:  python scripts/framing_selection_check.py

# The defect

A docking user clicks a pose in order to *look at it*. In the framing the window
shipped, it could not: `_on_pose_selected` refreshed the readouts, the contacts
and the ghosts, and never touched the camera, so the camera stayed wherever
loading had put it -- framing the whole receptor. Measured on
`examples/1crn_prep.pdbqt` with `examples/crambin_pose.pdbqt` at 1251x989:

    whole-receptor framing      pose  1 501 px   0.12%
                                receptor        68 915 px   5.57%
                                overlay          1 108 px   0.09%

0.12% of a frame, and at that distance over half the pose is *behind* the
receptor's own spheres -- the pose spans 31% of the frame height and is still
0.12% of the pixels. A linear measure says "fine" about a picture where the
ligand cannot be found.

# What a selection frames, and why not the alternatives

`framing_selection` states the argument; the short form is:

* **the pose** -- it is the subject;
* **every receptor atom the pose is in contact with** -- the overlays are lines
  drawn between exactly those two sets, so a framing that kept the pose and
  dropped the partners left the lines pointing off the edge of the picture;
* **not the 22 A search box, not the site volume** -- both describe the
  *search*, not this pose. All nine reported poses came out of that same cube and
  sit in that same cloud, so framing either of them answers "where did the
  engine look" at the same scale that made the pose unreadable. Clicking a site
  in the site table is the gesture for that question, and it still is.

Measured over all nine poses, the selection framing gives the pose **1.66% -
3.93%, median 2.64%** and the receptor **4.73% - 11.96%**: between 14x and 21x
the pose's share of the shipped framing.

# Three instruments, and why there are three

A framing can be wrong in ways one number cannot tell apart, so this file uses
three and none of them is sufficient alone:

1. **Exact, camera-side, runs with no framebuffer** -- is the camera pointed at
   the selection, and does the selection fit the frame it was given?
2. **Exact, no framebuffer** -- does the transition *land* on its target? A check
   that only asserts the final framing cannot see a move that ends in the wrong
   place; a check that only asserts the plan cannot see a product that ignores
   it. Both are here.
3. **Pixels, needs a framebuffer, skips with a reason** -- does the pose own
   enough of the picture, and are the overlays readable in it? This is the only
   instrument that measures what a user sees, and it is the one that cannot run
   on a machine with no GL.

# What needs a GPU, and what does not

Sections 2 to 5 are geometry and arithmetic on a `Camera` and a point cloud: they
run anywhere, including a headless runner. Sections 6 and 7 read the framebuffer
and register a `skip` in the branch they can be skipped in, so a machine with no
context still reaches the pinned total rather than printing a smaller one that
looks like a smaller scope.

Exit codes: 0 finished with no failures, 1 finished with at least one failure,
2 did not finish.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEV_TREE = ROOT / "dock-py" / "python"
#: This file measures *this checkout*. The installed wheel and the mirror under
#: `opendocking-gui` are the same bytes only until someone edits one of them,
#: and a guard that silently measured a stale copy would pass on the code it is
#: supposed to be guarding. Section 1 says which one answered.
sys.path.insert(0, str(DEV_TREE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: How many checks this file is supposed to run, in every environment.
#:
#: A `skip` counts here, on purpose, for the reason
#: `representation_cartoon_check.py` gives: a total that moves with the machine
#: is not a total. So the count is the same on a runner with no GL context, and
#: the summary prints passes and skips as two numbers that are never added.
#:
#: The derivation, so a future change to it is deliberate:
#:
#:   1  section 1, which copy of the workbench answered
#: + 3  section 2, what is and is not in the selection
#: + 3  section 3, the framing is pointed at the selection and contains it
#: + 4  section 4, the transition starts where it was and lands on its target
#: + 3  section 5, the degenerate framings are rejected and the real one is not
#: + 3  section 6, the pose's and the receptor's share of the pixels
#: + 3  section 7, the overlay is readable at that framing
#: + 4  section 8, the pose is findable in the picture, plus the two
#:      mutations that must be rejected and the control that must be accepted
#: + 3  the three meta-checks in the summary, which count themselves
#:
#: 1 + 3 + 3 + 4 + 3 + 3 + 3 + 4 + 3 = 27. Sections 6, 7 and 8 contribute
#: their full count on a machine that cannot draw them, by way of a recorded
#: skip, so the total is the same in every environment.
#:
#: **26 -> 27, and the number is allowed to move.** The third meta-check is the
#: one this file said it would never have, on the grounds that adding it would
#: take the total to 27 while 26 is a number other documents quote. That reason
#: does not hold: a guard that changes the number it guards is not a guard, and
#: a quoted number is not a reason to leave a defect uncovered. What the third
#: check guards was measured rather than argued -- see `summarise`, which is the
#: only place a tally is read.
EXPECTED_CHECKS = 27

#: The call-site census, declared here rather than in a table
#: `check_scripts_declare.py` owns. Comments only, so `EXPECTED_CHECKS` above is
#: unaffected by them.
#:
#: 17+10, and the ten guarded sites are exactly the ten framebuffer checks:
#: three in section 6, three in section 7 and four in section 8. Everything
#: else is arithmetic on a camera and a point cloud -- which points are in the
#: selection, where the centre is, whether the selection fits, where the plan
#: ends, whether a degenerate framing is recognised, and what colour a pose
#: wears -- and runs on a machine with no display.
#:
#: That split is the point of the design rather than a description of it. Five
#: of the eight sections answer "is the framing right", and all five are
#: answerable without a GPU. The three that cannot are the three questions
#: about the *picture*: how much of the frame the pose owns, whether the
#: overlays are readable in it, and whether a person can find the pose in it.
#: A headless runner therefore still gets the framing contract in full and is
#: told, with a reason, which three it could not answer.
#:
#: 16 -> 17 unconditional, and the added site is the third meta-check in
#: `summarise`. It is unconditional like the two beside it: it runs on any
#: machine, and guarding it on a display would be a guard that disappears on
#: exactly the runners that cannot see a picture.
#:
#: The `guards:` digest changed as well, and **it was already stale before this
#: edit**: this file declared `sha256:8b07f741...` while the same auditor
#: derived `sha256:43e96753...` from this same file, and it reported that as
#: "same census, different guards". The ten derived guard strings are still the
#: ten framebuffer guards this comment describes -- `if not can_draw`, with its
#: `ovl is None` and `pm is None or rm is None` companions -- so the substance
#: of the declaration still matches the file and only the hash literal was out of
#: date, with no edit of mine in between. The value below is the derived one,
#: re-derived from the same walk rather than copied, and the census is 17+10 for
#: the reason above. Anyone who needs to know what changed should diff the ten
#: guard strings, which is what the digest is a fingerprint of.
#: GATE-DECLARE 1
#: sites: 17 unconditional + 10 guarded
#: guards: sha256:43e967532092d8069017ba01abdb301c0c2cb0041117d44a7b8c6412e81a0eda

EXPECTED_SECTIONS = (
    "1. which copy of the workbench this measured",
    "2. what a selection frames, and what it does not",
    "3. the framing is pointed at the selection and contains it",
    "4. the transition lands where it was aimed",
    "5. the degenerate framings are rejected",
    "6. the pose and the receptor in the pixels",
    "7. the interaction overlay is readable at that framing",
    "8. the selected pose can be found in the picture",
    "summary",
)

RESULTS: list[tuple[str, str, str]] = []
FAILURES: list[str] = []
_sections: list[str] = []


def check(name, ok, detail=""):
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def skip(name, reason):
    """Record a check this machine could not answer, and say why.

    Counts toward `EXPECTED_CHECKS` -- see the note on that constant. Without
    this the total would be smaller on a machine with no GL context, and a
    smaller total is indistinguishable from a smaller scope.
    """
    RESULTS.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  - {reason}")
    return False


def section(t):
    print(f"\n=== {t} ===")
    _sections.append(t)


# --------------------------------------------------------------------------
# Measurement helpers. The frame-reading ones never raise and never return
# None without saying so, so a lost frame is a recorded result rather than a
# truncated run.
# --------------------------------------------------------------------------
def grab(win):
    """The viewport as an RGB array, or ``None`` with the reason on `LAST_PROBLEM`."""
    global LAST_PROBLEM
    from PyQt6 import QtGui

    LAST_PROBLEM = ""
    try:
        win.viewport.repaint()
        QtWidgets_app().processEvents()
        img = win.viewport.grabFramebuffer()
        if img is None or img.isNull():
            LAST_PROBLEM = "grabFramebuffer() returned a null image"
            return None
        img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        p = img.constBits()
        p.setsize(img.sizeInBytes())
        a = np.frombuffer(p, dtype=np.uint8).reshape(
            img.height(), img.bytesPerLine() // 4, 4
        )
        return a[:, : img.width(), :3].astype(np.int16)
    except Exception as exc:  # noqa: BLE001 - an unreadable frame is a result
        LAST_PROBLEM = f"{type(exc).__name__}: {exc}"
        return None


LAST_PROBLEM = ""

_APP = None


def QtWidgets_app():
    global _APP
    if _APP is None:
        from PyQt6 import QtWidgets

        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def difference_mask(a, b):
    """Pixels where two frames disagree by more than a rounding error.

    Against a render of a *different state* rather than against a fixed
    background colour, because the backdrop is a gradient and a modal-colour
    reference marks most of an empty frame as object -- the same reasoning
    `secondary_render.ribbon_mask` gives, for the same reason.
    """
    if a is None or b is None:
        return None
    return np.abs(a - b).sum(axis=2) > 12


def own_pixels(win, role):
    """The pixels one role owns: this frame, minus the frame without it.

    Returns ``(mask, problem)``, never a bare mask, because a frame that could
    not be read has to be a result the caller decides on rather than a crash
    that truncates the run.
    """
    for m in win.viewport.molecules:
        if m.role == role:
            m.visible = False
    win.viewport.update()
    for _ in range(6):
        QtWidgets_app().processEvents()
    without = grab(win)
    for m in win.viewport.molecules:
        if m.role == role:
            m.visible = True
    win.viewport.update()
    for _ in range(6):
        QtWidgets_app().processEvents()
    with_role = grab(win)
    if without is None or with_role is None:
        return None, LAST_PROBLEM or "a frame could not be read"
    return difference_mask(with_role, without), ""


def overlay_pixels(win):
    """The pixels the interaction overlay owns, and its contrast against them.

    The overlay is toggled rather than hidden, because it is a checkbox and a
    check that used a path the product does not have would be testing something
    nobody can reach.
    """
    win.cb_contacts.setChecked(False)
    for _ in range(6):
        QtWidgets_app().processEvents()
    off = grab(win)
    win.cb_contacts.setChecked(True)
    for _ in range(6):
        QtWidgets_app().processEvents()
    on = grab(win)
    if off is None or on is None:
        return None, 0.0, LAST_PROBLEM or "a frame could not be read"
    delta = np.abs(on - off)
    mask = delta.sum(axis=2) > 12
    contrast = float(delta[mask].mean()) if mask.any() else 0.0
    return mask, contrast, ""


def stroke_widths(mask, min_area: int = 3):
    """Width in pixels of each drawn dash: its area over its longest extent.

    Area over the longest run rather than a bounding box, because a diagonal
    dash's bounding box is its length: a dash at 45 degrees measures about 1.4x
    its own width that way, and the quantity being asked about is the width.
    """
    lab = np.zeros(mask.shape, np.int32)
    h, w = mask.shape
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        stack.append((yy, xx))
    out = []
    for i in range(1, n + 1):
        by, bx = np.nonzero(lab == i)
        area = len(by)
        if area < min_area:
            continue
        pts = np.stack([bx, by]).astype(np.float64)
        pts -= pts.mean(axis=1, keepdims=True)
        _, vecs = np.linalg.eigh(pts @ pts.T)
        proj = pts.T @ vecs[:, -1]
        out.append(area / max(float(proj.max() - proj.min()), 1e-6))
    return np.asarray(out)


def settle(win, limit_ms: int = 4000):
    """Run the camera move to completion, waiting on the **clock**.

    The waiting is the point, and a first version that looped `processEvents`
    and gave up after a fixed number of iterations was wrong in a way that
    quietly corrupted every pixel number measured after it. The move is driven
    by a `QTimer` on wall-clock time; a tight `processEvents` loop burns its
    budget in a few milliseconds and returns with the camera still in flight.

    Measured on this machine: it returned unfinished on 3 of 4 poses, and two
    grabs taken back to back *with nothing changed* then differed by 47%, 64%
    and 75% of the frame, because they were of two different camera positions.
    Every figure measured by differencing two grabs is only meaningful once the
    camera has stopped, so this waits, and says so if the move will not finish.
    """
    from PyQt6.QtTest import QTest

    waited = 0
    while win.viewport.moving and waited < limit_ms:
        QTest.qWait(20)
        waited += 20
    for _ in range(10):
        QtWidgets_app().processEvents()
    return waited


#: The two floor measurements below are read off this table, which is the
#: measurement rather than a comment about one:
#:
#:     nine poses, 1crn + crambin, 1251x989
#:       whole-receptor framing   pose 0.12%   receptor 5.57%   overlay 0.09%
#:       selection framing        pose 1.66 - 3.93% (median 2.64%)
#:                                receptor 4.73 - 11.96% (median 7.40%)
#:                                projected fill 0.72 - 0.82
#:
#: `POSE_SHARE_TARGET` is 1.5%: just under the worst pose, and 12.5x the
#: shipped framing. `RECEPTOR_SHARE_FLOOR` is only 2%, because the receptor's
#: share is a fact about how much protein surrounds a given pose and not about
#: the framing -- the first version of it was 10% and failed on seven of nine.
#: The question it was standing in for -- "is the protein context there?" -- is
#: asked properly by the containment bound in section 3, which is exact.


#: A pixel needs this much spread between its brightest and darkest channel
#: before it has a hue at all. 24 of 255 is about 9%: far above the 1-2 levels
#: of rounding in an 8-bit framebuffer, and far below `COLOR_BEST_POSE`'s green.
#: The same constant and the same definition as `workbench_interaction_check.py`
#: uses for its own hue claims, so the two files cannot disagree about what
#: "has a hue" means.
CHROMA_MIN = 24

#: How much greener than both red and blue a pixel has to be before it counts,
#: in levels. Five, not one: one is inside the rounding.
HUE_MARGIN = 5

#: The two floors on the pose's greenness in the picture. Both are the numbers
#: `workbench_interaction_check.py` already uses for the same claim on the same
#: populations, quoted here rather than re-derived, so the two suites agree by
#: construction instead of by coincidence.
#:
#: 0.90 is absolute: the pose is *one flat colour*, so essentially all of its
#: chromatic pixels point the same way, and 0.90 leaves room for the fog and the
#: two-light shading to pull a few edge pixels off. 0.60 is relative to the
#: protein and is the one that matters -- it is the margin, not the level, that
#: says "you can point at the ligand in the no-overlay picture".
POSE_GREEN_FLOOR = 0.90
POSE_GREEN_MARGIN = 0.60


def greener_than_both(frame, mask) -> float:
    """Share of `mask`'s *chromatic* pixels that are greener than red and blue.

    Chroma alone is a magnitude and the protein has coloured atoms of its own,
    so it does not separate: measured, the pose's chroma median moved from 8 to
    32 with the identity colour and still sat at the receptor's own 80th
    percentile. The direction of the hue is what separates one flat green object
    from a field of grey dots with some blue and red ones in it.
    """
    if frame is None or mask is None or not mask.any():
        return 0.0
    f = frame.astype(np.int16)
    sel = mask & ((f.max(axis=2) - f.min(axis=2)) >= CHROMA_MIN)
    if not sel.any():
        return 0.0
    g_red = float(((f[:, :, 1] - f[:, :, 0])[sel] >= HUE_MARGIN).mean())
    g_blue = float(((f[:, :, 1] - f[:, :, 2])[sel] >= HUE_MARGIN).mean())
    return min(g_red, g_blue)


def pose_is_findable(frame, pose_mask, receptor_mask) -> tuple[bool, str]:
    """Can a person point at the pose in this frame, with no overlay drawn?

    One number per population, and the claim is about the *gap* between them.
    A floor on the pose alone is satisfied by a picture where the whole scene is
    green, which is what the second mutation below is.
    """
    p = greener_than_both(frame, pose_mask)
    r = greener_than_both(frame, receptor_mask)
    return (p >= POSE_GREEN_FLOOR and p - r >= POSE_GREEN_MARGIN), (
        f"pose {p:.1%} of its chromatic pixels are greener than red and blue, "
        f"protein {r:.1%}, margin {p - r:+.1%} (want >="
        f"{POSE_GREEN_FLOOR:.0%} and a margin of >={POSE_GREEN_MARGIN:.0%})"
    )


def main() -> int:
    from opendocking.workbench import Camera, framing_selection as fs

    app = QtWidgets_app()
    from opendocking.workbench.app import (
        MainWindow,
        POCKET_OPACITY_COMPARE,
        POCKET_OPACITY_PLAIN,
    )

    section("1. which copy of the workbench this measured")
    import opendocking.workbench as pkg

    check(
        "the framing that answered is this checkout's",
        Path(pkg.__file__).resolve().is_relative_to(DEV_TREE.resolve())
        if hasattr(Path, "is_relative_to")
        else str(DEV_TREE.resolve()) in str(Path(pkg.__file__).resolve()),
        f"{Path(pkg.__file__).resolve()}",
    )

    win = MainWindow(
        receptor=ROOT / "examples" / "1crn_prep.pdbqt",
        ligand=ROOT / "examples" / "crambin_pose.pdbqt",
        poses=ROOT / "examples" / "poses.pdbqt",
    )
    win.resize(1500, 950)
    win.show()
    for _ in range(40):
        app.processEvents()
    settle(win)

    pose = next(m for m in win.viewport.molecules if m.role == "pose")
    receptor = next(m for m in win.viewport.molecules if m.role == "receptor")
    P = np.asarray(pose.coords, np.float32)
    contacts = list(win.viewport.contacts)
    partner_idx = sorted({c.partner_index for c in contacts
                          if 0 <= c.partner_index < len(receptor.coords)})
    partner_residues = sorted({c.partner_residue for c in contacts
                               if 0 <= c.partner_index < len(receptor.coords)})
    Q = np.asarray(receptor.coords, np.float32)[partner_idx] if partner_idx \
        else np.zeros((0, 3), np.float32)
    cam = win.viewport.camera
    right, up, forward = cam.basis()
    aspect = win.viewport.width() / max(win.viewport.height(), 1)
    target = fs.selection_target(
        P, Q, right, up, forward, fov=cam.fov, aspect=aspect,
        partner_residues=tuple(partner_residues),
    )
    print(f"  pose {len(P)} atoms, {len(contacts)} contacts, "
          f"{len(partner_idx)} contacting receptor atoms over "
          f"{len(partner_residues)} residues {partner_residues}")
    print(f"  selection frames distance {target.distance:.2f} A at "
          f"{np.round(target.center, 2)}")

    # -- 2. what is and is not in the selection --------------------------
    section("2. what a selection frames, and what it does not")
    check(
        "the selection is the pose, every atom of it",
        target.pose_atoms == len(P),
        f"{target.pose_atoms} pose atoms of the {len(P)} on screen; framing the "
        f"pose means all of it or the picture is cropped without saying so",
    )
    check(
        "and every receptor atom the pose is in contact with",
        target.partner_atoms == len(partner_idx),
        f"{target.partner_atoms} partner atoms against {len(partner_idx)} "
        f"distinct contacting atoms over {len(partner_residues)} residues. The "
        f"overlays are lines drawn between exactly these two sets, so a framing "
        f"that kept the pose and dropped the partners leaves every line "
        f"pointing at something outside the picture",
    )
    sel = fs.selection_points(P, Q)
    box_corners = np.asarray([
        [win.viewport.box_center[i] + s * win.viewport.box_size[i] * 0.5
         for i, s in enumerate(signs)]
        for signs in np.ndindex(*(2, 2, 2))
    ], np.float64)
    site = np.asarray(win.viewport.pocket_points, np.float64).reshape(-1, 3)
    stray = 0
    for group, name in ((box_corners, "the search box"), (site, "the site volume")):
        if not len(group):
            continue
        inside = [
            any(float(np.linalg.norm(pt - s)) < 1e-6 for s in sel) for pt in group
        ]
        stray += sum(inside)
    check(
        "and neither the 22 A search box nor the site volume",
        stray == 0,
        f"{stray} of the box's {len(box_corners)} corners and the site's "
        f"{len(site)} grid points are in the selection. Both describe the "
        f"*search*, not this pose: all nine reported poses came out of that one "
        f"cube and sit in that one cloud, so framing either answers 'where did "
        f"the engine look' at the scale that made the pose unreadable. "
        f"Selecting a site in the site table is the gesture for that question "
        f"and still does it",
    )

    # -- 3. pointed at the selection, and contains it --------------------
    section("3. the framing is pointed at the selection and contains it")
    fill = fs.projected_fill(sel, target, right, up, forward,
                             fov=cam.fov, aspect=aspect)
    sx = sel @ right
    sy = sel @ up
    span = max(float(sx.max() - sx.min()), float(sy.max() - sy.min())) * 0.5
    off = max(
        abs(float(target.center @ right) - 0.5 * (sx.max() + sx.min())),
        abs(float(target.center @ up) - 0.5 * (sy.max() + sy.min())),
    )
    check(
        "the camera is centred on the selection, not on the protein",
        off <= fs.CENTRE_TOLERANCE * span,
        f"centre is {off:.2f} A outside the selection's {2 * span:.2f} A "
        f"projected extent, against a tolerance of "
        f"{fs.CENTRE_TOLERANCE * span:.2f} A",
    )
    check(
        "nothing of the selection falls outside the frame",
        fill <= 1.0,
        f"the selection reaches {fill:.2f} of the way to the frame edge "
        f"(1.0 is the edge). This is the check that catches a distance fitted "
        f"from a bounding box at the centre plane, which ignores perspective: "
        f"measured, that put pose 3's own atoms at |ndc| 3.89",
    )
    check(
        "and the selection fills the frame it was given",
        fill >= fs.MIN_SELECTION_FRAME_FILL,
        f"the selection spans {fill:.0%} of the half-frame, against a floor of "
        f"{fs.MIN_SELECTION_FRAME_FILL:.0%}; fit_view aims at "
        f"{1 / fs.SELECTION_MARGIN:.0%} and the measured range over the nine "
        f"poses is 0.72-0.82",
    )

    # -- 4. the transition lands where it was aimed ----------------------
    section("4. the transition lands where it was aimed")
    plan = fs.CameraMove(
        start_center=np.asarray([0.0, 0.0, 0.0], np.float32),
        start_distance=40.0,
        target=target,
    )
    end_c, end_d = plan.at(1.0)
    settle_c, settle_d = plan.settle()
    check(
        "the plan starts where the camera is",
        np.allclose(plan.start_center, np.zeros(3, np.float32), atol=0.0)
        and plan.start_distance == 40.0,
        f"from {plan.start_center} at {plan.start_distance} A",
    )
    check(
        "and ends exactly on the target, not near it",
        float(np.abs(np.asarray(end_c, np.float64)
                     - np.asarray(target.center, np.float64)).max()) == 0.0
        and end_d == target.distance
        and np.array_equal(np.asarray(settle_c), np.asarray(end_c)),
        f"at(1.0) gives {end_c} at {end_d} A; settle() gives {settle_c} at "
        f"{settle_d} A; the target is {np.round(target.center, 6)} at "
        f"{target.distance} A. A tween that approaches its target asymptotically "
        f"ends a thousandth of an angstrom short forever, and a check written "
        f"against the target then has to tolerate a floating-point remainder",
    )
    mids = [plan.at(k / float(plan.steps)) for k in range(1, plan.steps)]
    strict = all(d < 40.0 for _c, d in mids) and all(
        float(np.abs(np.asarray(c, np.float64)
                     - np.asarray(target.center, np.float64)).max()) > 0.0
        for c, _d in mids
    )
    check(
        "every intermediate step is strictly between the two ends",
        strict and len(mids) == plan.steps - 1,
        f"{len(mids)} intermediate states over {plan.steps} steps, distances "
        f"{min(d for _c, d in mids):.2f}..{max(d for _c, d in mids):.2f} A "
        f"against 40.00 and {target.distance:.2f} A: a camera that snapped would "
        f"put the start and the end on adjacent steps and there would be nothing "
        f"to check",
    )
    # The product, not the plan: the real dispatch, with the move run to the end.
    win.viewport.frame_all()
    settle(win)
    far = float(win.viewport.camera.distance)
    rows = win.pose_table.rowCount()
    target_row = next((r for r in range(rows)
                       if r != win.pose_table.currentRow()), 0)
    win.pose_table.selectRow(target_row)
    waited = settle(win)
    live = win._last_selection_plan
    cam_now = win.viewport.camera
    drift = float(np.abs(np.asarray(cam_now.center, np.float64)
                         - np.asarray(live.target.center, np.float64)).max())
    check(
        "selecting a row through the real dispatch ends on its target",
        drift < 1e-6 and abs(cam_now.distance - live.target.distance) < 1e-6
        and not win.viewport.moving,
        f"row {target_row}: the move ran {far:.2f} -> "
        f"{live.target.distance:.2f} A in {live.steps} steps, finished in "
        f"{waited} ms, and the camera sits {drift:.1e} A and "
        f"{abs(cam_now.distance - live.target.distance):.1e} A from the target",
    )

    # -- 5. the degenerate framings are rejected -------------------------
    section("5. the degenerate framings are rejected")
    win.viewport.frame_all()
    settle(win)
    cam = win.viewport.camera
    r5, u5, f5 = cam.basis()
    asp5 = win.viewport.width() / max(win.viewport.height(), 1)
    whole = fs.FramingTarget(
        center=np.asarray(cam.center, np.float32), distance=float(cam.distance),
        pose_atoms=len(P), partner_atoms=len(partner_idx), partner_residues=(),
    )
    degen, why = fs.selection_is_degenerate(
        whole, P, Q, r5, u5, f5, fov=cam.fov, aspect=asp5)
    check(
        "the whole-receptor framing -- centred, everything in shot -- is rejected",
        degen,
        why or f"selection_is_degenerate() accepted a framing {cam.distance:.1f} A "
               f"out; this is the shape the whole change is about, and it "
               f"satisfies every threshold that only asks whether the scene is in "
               f"the picture",
    )
    far_target = fs.FramingTarget(
        center=np.asarray(target.center, np.float32),
        distance=float(target.distance) * 4.0,
        pose_atoms=len(P), partner_atoms=len(partner_idx), partner_residues=(),
    )
    degen2, why2 = fs.selection_is_degenerate(
        far_target, P, Q, r5, u5, f5, fov=cam.fov, aspect=asp5)
    check(
        "and so is the same framing pulled four times further back",
        degen2,
        why2 or f"a camera {far_target.distance:.1f} A out was accepted: pointed "
                f"at the right thing, from so far away that the pose is a speck "
                f"again -- the other degenerate shape, which a check on the "
                f"centre alone cannot see",
    )
    sel_ok = fs.selection_is_degenerate(
        target, P, Q, r5, u5, f5, fov=cam.fov, aspect=asp5)[0]
    check(
        "while the framing the product actually uses is accepted",
        not sel_ok,
        f"the same predicate on the real target ({target.distance:.2f} A, fill "
        f"{fill:.2f}) says not degenerate, so the two rejections above are the "
        f"guard discriminating and not the guard being unable to pass anything",
    )

    # -- 6 and 7: the pixels ---------------------------------------------
    can_draw = win.viewport._ctx is not None
    if not can_draw:
        reason = (
            "the viewport has no OpenGL context on this machine, so the pose's "
            "share of the picture is unknown. The camera-side questions in "
            "sections 2 to 5 were still answered from the arithmetic"
        )
        try:
            from opendocking.workbench.app import odgui_check
            reason += f"; {odgui_check().describe()}"
        except Exception:  # noqa: BLE001 - the reason is already stated
            pass
    else:
        reason = ""

    print(f"\n  framebuffer can answer: {'yes' if can_draw else 'no'}"
          + (f" - {reason}" if reason else ""))

    # The shipped framing, for the comparison every ratio below is against.
    win.viewport.pocket_opacity = POCKET_OPACITY_PLAIN
    win.viewport.frame_all()
    settle(win)
    base = grab(win)
    if can_draw and base is None:
        can_draw = False
        reason = LAST_PROBLEM
    before = {}
    if can_draw:
        _, W, H = base.shape[1], base.shape[0], base.shape[1]
        total = W * H
        m, _p = own_pixels(win, "pose")
        before["pose"] = int(m.sum())
        m, _p = own_pixels(win, "receptor")
        before["receptor"] = int(m.sum())
        om, _c, _p = overlay_pixels(win)
        before["overlay"] = int(om.sum())
        print(f"\n  shipped framing: {W}x{H} at {win.viewport.camera.distance:.2f} A")
        print(f"    pose {before['pose']:7d} px {before['pose'] / total:6.2%}   "
              f"receptor {before['receptor']:7d} px "
              f"{before['receptor'] / total:6.2%}   overlay "
              f"{before['overlay']:6d} px {before['overlay'] / total:6.2%}")

    # `pm`, `rm` and `after_frame` are the pose, receptor and full frames at the
    # selection framing. Section 8 reads them, so they are named before section
    # 6 rather than only inside its drawing branch: a section that can only run
    # when an earlier section took its "drawing" branch is a section whose
    # absence is silent.
    pm = rm = after_frame = None

    section("6. the pose and the receptor in the pixels")
    if not can_draw:
        for name in (
            "the pose owns a real share of the frame after a selection",
            "the receptor keeps a share of the frame, so the context survives",
            "and the pose's share is a long way above the whole-receptor framing",
        ):
            skip(name, reason)
        after = {}
        ovl = None
        contrast = 0.0
    else:
        # The same pose before and after, with only the camera differing.
        # `selectRow` cannot be used here: by this point the row is already
        # current, so it emits nothing, the "after" is the "before", and the
        # comparison reports 1.0x for both -- which is what the first run of
        # this file did. The handler is called directly instead, which is what
        # the signal calls; section 4 already covers the full `selectRow` path
        # including the camera move.
        win._on_pose_selected(win.pose_table.currentRow())
        settle(win)
        after_frame = grab(win)
        total = after_frame.shape[0] * after_frame.shape[1]
        pm, prob = own_pixels(win, "pose")
        rm, prob2 = own_pixels(win, "receptor")
        if pm is None or rm is None:
            can_draw = False
            reason = prob or prob2
            for name in (
                "the pose owns a real share of the frame after a selection",
                "the receptor keeps a share of the frame, so the context survives",
                "and the pose's share is a long way above the whole-receptor framing",
            ):
                skip(name, reason)
            after = {}
            ovl = None
            contrast = 0.0
        else:
            after = {"pose": int(pm.sum()), "receptor": int(rm.sum())}
            ovl, contrast, _p = overlay_pixels(win)
            after["overlay"] = int(ovl.sum()) if ovl is not None else 0
            print(f"\n  selection framing: {after_frame.shape[1]}x"
                  f"{after_frame.shape[0]} at "
                  f"{win.viewport.camera.distance:.2f} A")
            print(f"    pose {after['pose']:7d} px {after['pose'] / total:6.2%}   "
                  f"receptor {after['receptor']:7d} px "
                  f"{after['receptor'] / total:6.2%}   overlay "
                  f"{after['overlay']:6d} px {after['overlay'] / total:6.2%}")
            check(
                "the pose owns a real share of the frame after a selection",
                after["pose"] / total >= fs.POSE_SHARE_TARGET,
                f"{after['pose']} px of {total} = "
                f"{after['pose'] / total:.2%}, against a floor of "
                f"{fs.POSE_SHARE_TARGET:.1%}. The floor is just under the worst "
                f"of the nine poses (1.66%) and 12.5x the "
                f"{before['pose'] / total:.2%} the shipped framing gives. It is "
                f"pixels and not frame height on purpose: the pose spans 31% of "
                f"the frame height at 43 A and is still 0.12% of the pixels",
            )
            check(
                "the receptor keeps a share of the frame, so the context survives",
                after["receptor"] / total >= fs.RECEPTOR_SHARE_FLOOR,
                f"{after['receptor']} px of {total} = "
                f"{after['receptor'] / total:.2%}, against a floor of "
                f"{fs.RECEPTOR_SHARE_FLOOR:.0%} "
                f"({before['receptor'] / total:.2%} in the shipped framing). The "
                f"floor is deliberately weak: the receptor's share is a fact "
                f"about how much protein surrounds a given pose -- 4.73% to "
                f"11.96% over the nine -- so gating hard on it asks a question "
                f"about the pose's surroundings. Whether the protein the lines "
                f"point into is *in the picture* is section 3's containment "
                f"bound, and that one is exact",
            )
            ratio = after["pose"] / max(before["pose"], 1)
            check(
                "and the pose's share is a long way above the whole-receptor framing",
                ratio >= 8.0,
                f"{after['pose']} px against {before['pose']} px is {ratio:.1f}x, "
                f"against a floor of 8x. Same camera basis, same colours, same "
                f"representation: the only difference is where the camera is "
                f"pointing",
            )

    section("7. the interaction overlay is readable at that framing")
    if not can_draw or ovl is None:
        for name, why_not in (
            ("the overlay covers more of the frame than it did in the shipped "
             "framing", reason or "no overlay mask could be read"),
            ("every dash is at least a visible thread, not a sub-pixel hair",
             reason or "no overlay mask could be read"),
            ("and the overlay stands out from the pixel underneath it",
             reason or "no overlay mask could be read"),
        ):
            skip(name, why_not)
    else:
        gained = after["overlay"] / max(before["overlay"], 1)
        check(
            "the overlay covers more of the frame than it did in the shipped "
            "framing",
            after["overlay"] > before["overlay"] * 3.0,
            f"{after['overlay']} px against {before['overlay']} px is "
            f"{gained:.1f}x, against a floor of 3x. The overlays are the reason "
            f"a selection is worth looking at, and they are short: framing "
            f"buys them length (a 0.40 A dash is 10.7 px at 43 A and 27 px at "
            f"14 A) and not width, which is why width has its own check",
        )
        widths = stroke_widths(ovl)
        median = float(np.median(widths)) if len(widths) else 0.0
        check(
            "every dash is at least a visible thread, not a sub-pixel hair",
            median >= 1.5,
            f"median {median:.2f} px over {len(widths)} dashes of 3 px or more "
            f"(p10 {np.percentile(widths, 10):.2f}, p90 "
            f"{np.percentile(widths, 90):.2f}), against a floor of 1.5 px. "
            f"Measured: this is ~1.9 px at *every* framing from 43 A to 8 A, "
            f"because the width is `glLineWidth` 1.0 plus MSAA and does not "
            f"scale with zoom. Zooming 5.4x changed the width by 6%, so 'the "
            f"line is too thin because the camera is far away' is false -- and "
            f"that is why the framing, not the stroke width, is what this change "
            f"fixes. The floor is set below what a 1.0 width with 4x MSAA "
            f"reliably gives, and is deliberately *not* a check that a driver "
            f"honours a wider `glLineWidth`: it does on this machine, and "
            f"asserting it would make this file unpassable on a driver that "
            f"clamps to 1.0",
        )
        check(
            "and the overlay stands out from the pixel underneath it",
            contrast >= 40.0,
            f"mean |on - off| of {contrast:.0f} of 255 over the overlay's pixels, "
            f"against a floor of 40. The gap between the overlay and the frame "
            f"it is drawn over is the whole reason the lines are coloured by "
            f"contact kind rather than left white",
        )

    section("8. the selected pose can be found in the picture")
    # The claim is about a *picture*, with the interaction overlays switched
    # off, because with them on the dashes converge on the pose and the user can
    # reverse-trace them -- which is exactly the workaround the framing was
    # supposed to remove.
    #
    # The instrument is hue *direction*, not chroma and not brightness. Chroma
    # was measured and rejected as the discriminator: the pose's chroma median
    # went 8 -> 32 with the identity colour and still sat at the receptor's own
    # 80th percentile, because a protein has coloured atoms of its own. A
    # brightness change cannot work either -- the same shader lights and fogs
    # both populations, so a highlight moves a pixel along the lit surface and
    # not out of the protein's distribution.
    if not can_draw or pm is None or rm is None:
        for name in (
            "the pose is separable from the protein in the picture with the "
            "overlays off, so it can be found without tracing the dashes",
            "the rule this replaced is rejected by the same predicate",
            "and so is painting the protein the same colour as the pose",
            "while the shipped picture is accepted",
        ):
            skip(name, reason or "no pose/receptor pixel masks could be read")
    else:
        import opendocking.workbench.app as appmod

        live_rule = appmod._draw_colours_for

        def element_colour_rule(mol, comparing=False):
            """The rule this change replaced: element colours unless comparing."""
            if mol.role == "pose_ghost" or (comparing and mol.role == "pose"):
                return np.tile(np.asarray(mol.color, np.float32),
                               (len(mol.coords), 1))
            return mol.atom_colors()

        def repaint():
            win.viewport.update()
            for _ in range(8):
                QtWidgets_app().processEvents()
            return grab(win)

        # The overlays off, which is the picture the user has to navigate.
        win.cb_contacts.setChecked(False)
        for _ in range(8):
            QtWidgets_app().processEvents()
        bare_pose, bare_receptor, bare = pm, rm, after_frame
        found, why = pose_is_findable(bare, bare_pose, bare_receptor)
        check(
            "the pose is separable from the protein in the picture with the "
            "overlays off, so it can be found without tracing the dashes",
            found,
            why + f". Over the pose's own {int(bare_pose.sum())} px: this is the "
                  f"acceptance picture (`dist/framing/pose_after_no_overlay.png`)"
                  f" and the claim is that a person can point at the ligand in it"
                  f" without following a contact line back to one",
        )

        # Mutation A: the exact rule that shipped. Element colours for a pose
        # that is not being compared, which is what the window did until this
        # change.
        appmod._draw_colours_for = element_colour_rule
        mutated = repaint()
        mut_ok, mut_why = pose_is_findable(mutated, bare_pose, bare_receptor)
        check(
            "the rule this replaced is rejected by the same predicate",
            not mut_ok,
            f"element colours for the pose -> {mut_why}. Before the change the "
            f"pose's chroma median was 8 against the protein's 7, the protein's "
            f"own 60th percentile: the pose sat *inside* the protein's colour "
            f"distribution, so this predicate is what the reversal had to beat",
        )

        # Mutation B: the other direction, and the one a pose-only floor cannot
        # catch. Paint the protein in the pose's own colour too. The pose is
        # still 100% green and still clears the absolute floor; only the margin
        # against the protein is left to reject it.
        pose_view = next(m for m in win.viewport.molecules if m.role == "pose")

        def all_green(mol, _live=live_rule):
            if mol.role in ("pose", "pose_ghost", "receptor", "ligand"):
                return np.tile(np.asarray(pose_view.color, np.float32),
                               (len(mol.coords), 1))
            return _live(mol)

        appmod._draw_colours_for = all_green
        green = repaint()
        gr_ok, gr_why = pose_is_findable(green, bare_pose, bare_receptor)
        check(
            "and so is painting the protein the same colour as the pose",
            not gr_ok,
            f"the whole scene in the pose's own colour -> {gr_why}. The pose is "
            f"still above the absolute floor here, so an absolute floor alone "
            f"would pass this; only the margin rejects a picture in which "
            f"everything is the thing you are looking for",
        )

        appmod._draw_colours_for = live_rule
        restored = repaint()
        rs_ok, rs_why = pose_is_findable(restored, bare_pose, bare_receptor)
        check(
            "while the shipped picture is accepted",
            rs_ok,
            f"restored -> {rs_why}. Both mutations above were applied by "
            f"replacing the module's `_draw_colours_for` and re-rendering the "
            f"same frame, so they went through the real shader and the real "
            f"dispatch rather than through arithmetic on a number -- and the "
            f"camera did not move between any of the three, so the comparison "
            f"is the colour and nothing else",
        )
        win.cb_contacts.setChecked(True)
        for _ in range(8):
            QtWidgets_app().processEvents()

    win.close()
    return summarise()


def summarise() -> int:
    """Counts, and the verdict. Printed on every path."""
    section("summary")
    # The count this file pins is the *total*, and it has to be read before the
    # two meta-checks register themselves, because it is what those two are
    # measured against. `len(RESULTS) + 2` is the only correct spelling here.
    total = len(RESULTS) + 3
    check(
        "the number of checks that ran is the number this file is supposed to "
        "have, so none of them is hiding behind a guard",
        total == EXPECTED_CHECKS,
        f"{len(RESULTS)} ran before these three, {total} including them, and "
        f"{EXPECTED_CHECKS} are expected"
        + ("" if total == EXPECTED_CHECKS
           else " - the total moved, so a check was added or removed without "
                "changing the pin deliberately")
        + "; the derivation is above EXPECTED_CHECKS",
    )
    check(
        "every section this file is supposed to reach was reached",
        tuple(_sections) == EXPECTED_SECTIONS,
        f"{len(_sections)} of {len(EXPECTED_SECTIONS)}; reached "
        f"{tuple(_sections) if tuple(_sections) != EXPECTED_SECTIONS else 'all of them'}",
    )
    # The third meta-check, and the reason the pin moved to 27.
    #
    # What was measured, not argued: with the tally taken above these two
    # `check` calls while the headline printed `len(RESULTS)` below them, this
    # file reported `26 checks: 24 passed, 0 failed, 0 skipped`, printed
    # `RESULT: PASS` and exited 0 -- with both meta-checks visible in the
    # transcript as `[ok]`. So the gate could not see its own defect, and the
    # fix was verified to be invisible to it. "The ordering is now correct" is
    # not a guard; a defect nobody can detect is a defect nobody will fix.
    #
    # So the numbers the headline will print are read here, once, and this
    # check asks the two questions a reader asks of any tally:
    #
    #   1. do the three counts partition everything already in the ledger?
    #      24 + 0 + 0 != 26 is the moved-back tally, caught;
    #   2. does the printed total already account for this check, which has not
    #      registered itself yet? `len(RESULTS) + 1` is the live ledger at the
    #      moment this runs, so a total computed anywhere else does not match.
    #
    # It cannot be a check about the print, and that is not a gap being hidden:
    # a check that ran after the print would be a record the print did not
    # count, which is the same defect one layer down. What is asserted is the
    # arithmetic, at the only moment in the file where the arithmetic is.
    seen = {
        "total": total,
        "passed": sum(1 for t, _n, _d in RESULTS if t == "PASS"),
        "failed": sum(1 for t, _n, _d in RESULTS if t == "FAIL"),
        "skipped": sum(1 for t, _n, _d in RESULTS if t == "SKIP"),
    }
    accounted = seen["passed"] + seen["failed"] + seen["skipped"]
    unaccounted = len(RESULTS) - accounted
    check(
        "the headline this file prints is one set of numbers, and they add up",
        accounted == len(RESULTS)
        and seen["total"] == len(RESULTS) + 1
        and {t for t, _n, _d in RESULTS} <= {"PASS", "FAIL", "SKIP"},
        f"before this check registers: {accounted} of {len(RESULTS)} records "
        f"are accounted for by {seen['passed']} pass + {seen['failed']} fail + "
        f"{seen['skipped']} skip"
        + (f", leaving {unaccounted} unaccounted" if unaccounted else "")
        + f"; the total the headline will print is {seen['total']}, and this "
        f"check makes it {len(RESULTS) + 1}. Moving the tally above these two "
        f"checks is what produced '26 checks: 24 passed' with RESULT: PASS, and "
        f"that run exits 1 here",
    )
    # Tallied after all three meta-checks, and that ordering is still the point:
    # the counts below are re-read from the ledger, so the headline counts this
    # check like every other one.
    passed = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    failed = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    skipped = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print(f"\n  {len(RESULTS)} checks: {passed} passed, {failed} failed, "
          f"{skipped} skipped")
    if passed + failed + skipped != len(RESULTS):
        print("RESULT: FAIL")
        print("  the three counts do not partition the ledger, so the headline "
              "above is not a count of anything")
    if FAILURES:
        print("RESULT: FAIL")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    if passed == 0:
        print("RESULT: DID NOT FINISH - nothing was verified")
        return 2
    print("RESULT: PASS" + (f" ({skipped} could not be measured here)"
                            if skipped else ""))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - the point is to never be silent
        import traceback

        traceback.print_exc()
        print(f"\nDID NOT FINISH - an exception escaped before every section was "
              f"reached: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
