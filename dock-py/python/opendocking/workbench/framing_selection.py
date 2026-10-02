"""What a pose selection frames, and how far away the camera stops.

# The problem this exists to answer

Selecting a pose is the moment a docking user is looking at the *pose*. In the
framing the window ships, the pose is not the subject: it is 16 atoms drawn
inside a 382-atom receptor, and measured on `examples/1crn_prep.pdbqt` with
`examples/crambin_pose.pdbqt` the pose's own pixels are **1 420 of 935 984,
0.15%** -- the camera is 43 A out because that is what frames the whole protein.
`_on_pose_selected` did not move it at all; only the "all poses" checkbox
reframes, and what it frames is the poses, not this one.

Nobody selects a pose to confirm it is somewhere in the protein.

# What belongs in the framing, and what does not

**In: every pose atom, and every receptor atom the pose is actually in contact
with.** The pose is the subject. The contacting atoms are in because the
question a selection answers is *what is this touching*, and the interaction
overlays are lines drawn between exactly those two sets of atoms -- a line whose
far endpoint is outside the frame is a line that cannot be read. Framing the
union of the two also puts the *entire* interface on screen, not the part of it
that happens to be near the pose's centroid.

**Out: the search box (22 A) and the site volume.** Both describe the *search*,
not this pose. Every one of the nine reported poses came out of that same cube
and sits inside that same cloud, so framing either of them answers "where did
the engine look", which is a different question, and answers it at the same
1.7%-of-a-box scale that made the pose unreadable in the first place. A user
who wants the box clicks a site in the site table, which already moves both the
box and the camera. Selecting a pose is not that gesture.

**Out: the rest of the receptor -- as a framing requirement, not as geometry.**
The receptor is still drawn, and it is what makes "touching" answerable, but
requiring all of it to fit is the degenerate framing this module's
:func:`selection_is_degenerate` exists to reject.

# Why the distance is a view-space fit and not `radius * 2.6`

`Viewport.frame_all` frames a bounding *sphere*: `distance = max(5, radius *
2.6)`. A molecule is not a sphere, so a spherical fit spends frame on empty
space in every direction the molecule does not reach. Worse for a selection: the
pose is long and flat, and the contacts sit on one face of it, so a spherical
fit on the union is loosest along the one axis that matters most.

:func:`fit_view` fits the selection's extents **along the camera's own right and
up axes** and takes the distance that makes the larger of them fill `1/margin`
of the corresponding half-frame. The centre is the midpoint of those two
extents, not the mean of the points, so the selection lands centred in the
picture rather than centred in its own scatter.

# The two numbers this is held to

:data:`POSE_SHARE_TARGET` and :data:`RECEPTOR_SHARE_FLOOR` are *floors on the
achieved framing*, chosen before the implementation was measured and deliberately
set **below** what the implementation reaches, so they are a contract rather than
a fit to the number. The reasoning, and the measurements behind it, are in
`scripts/framing_selection_check.py`; the short version is in each constant's
comment.

They are deliberately two numbers and not one. A framing that satisfies a
pose-share floor by pushing the protein off screen has answered neither question:
"where did the ligand go" is a statement about the ligand, but "what is it
touching" is a statement about the ligand *and the protein around it*, and the
receptor floor is what keeps the second question askable.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

__all__ = [
    "FramingTarget",
    "CameraMove",
    "POSE_SHARE_TARGET",
    "RECEPTOR_SHARE_FLOOR",
    "SELECTION_MARGIN",
    "MIN_SELECTION_DISTANCE",
    "CENTRE_TOLERANCE",
    "MIN_SELECTION_FRAME_FILL",
    "PAIR_CONTEXT_SHELL",
    "MAX_DRAWN_ANGLE_FILL",
    "PairFraming",
    "projected_fill",
    "fit_view",
    "fit_drawn",
    "drawn_fill",
    "drawn_envelope_slack",
    "focus_target",
    "selection_points",
    "selection_target",
    "selection_is_degenerate",
    "framed_radius",
]


#: Minimum share of the frame the pose's own visible pixels must reach.
#:
#: Measured, not guessed. Over all nine poses of `examples/poses.pdbqt` on
#: `examples/1crn_prep.pdbqt` at 1251x989, the selection framing gives the pose
#: **1.66% - 3.93%, median 2.64%**; the whole-receptor framing the window
#: shipped gives **0.12%** (1 501 px of 1 237 239). So this floor is set just
#: under the worst measured pose, 1.5%, which is 12.5x the shipped framing and
#: leaves 10% headroom below the worst case rather than sitting on it.
#:
#: It is a share of *pixels* and not of frame height, and that is the whole
#: measurement: the pose already spans 31% of the frame height at 43 A and is
#: still 0.12% of the pixels, because at that distance over half of it is hidden
#: behind the receptor's own spheres. A linear measure says "fine" about a
#: picture where the ligand cannot be found.
POSE_SHARE_TARGET = 0.015

#: Coarse floor on the receptor's own share, meaning "the protein is still on
#: screen" and nothing stronger.
#:
#: The first version of this was 10%, chosen to be *more* than the whole-receptor
#: framing's 5.57%, and it failed on seven of the nine poses. That is the
#: measurement saying the quantity is the wrong one: the receptor's share of the
#: frame depends on how much protein happens to surround a given pose, and over
#: the nine poses it runs **4.73% - 11.96%, median 7.40%**, with no floor above
#: 4% reachable by a framing that is honest about also framing the pose. A pose
#: in a surface groove has almost no protein around it; one in a tight pocket
#: has a great deal. Gating on the share asks a question about the *pose's
#: surroundings* and calls the answer a property of the framing.
#:
#: So the floor is dropped to 2%, where it only means what it says -- the protein
#: did not go away -- and the question it was standing in for is asked properly
#: by the containment bound in :func:`projected_fill`: every contacting receptor
#: atom is inside the frame, which is exact, pose-independent, and the actual
#: requirement behind "what is it touching". The share is still *measured* and
#: reported by `scripts/framing_selection_check.py`; it is just not a gate.
RECEPTOR_SHARE_FLOOR = 0.02

#: How much of the frame's half-extent the selection is allowed to fill along
#: its larger view-space axis; 1/this is what it fills.
#:
#: 1.0 would touch the edges on both sides, which crops the selection and makes
#: "is that all of it?" unanswerable. 1.2 leaves about a sixth of each half-frame
#: as the protein context the overlays are read against, and it is still enough
#: for the *drawn* extent rather than the atom centres: the outermost atom
#: centre sits at 83% of the half-frame and the 0.30 A drawing radius adds 5.3%
#: of it, so 1.1 would put a sphere edge at 94% and 1.0 would clip it.
#:
#: Measured across margins on the shipped example, the pose's own pixels run
#: 2.79% (1.10) / 2.46% (1.20) / 2.03% (1.30) / 1.37% (1.60). 1.20 is the
#: loosest margin that still keeps a sixth of the frame for context. The
#: perspective term in :func:`fit_view` then moved the achieved numbers again --
#: to 1.66%-3.93% over the nine poses -- because the camera has to stand far
#: enough back for the point nearest the eye not to be magnified off the frame.
SELECTION_MARGIN = 1.2

#: Floor on the camera distance, in angstrom, so a tight selection cannot put
#: the eye inside the spheres it is supposed to be looking at.
#:
#: **It is not pinned to the depth clip, and the sentence that used to say so
#: was false.** It said `Viewport` derives its near and far planes from the
#: camera distance. It does not. `Camera.projection` takes `znear=0.1,
#: zfar=2000.0` as **constants**, and recovering them out of the matrix gives
#: `znear = 0.100000` at camera distances 0.3, 0.5, 1.0, 6.0 and 40.0 A -- one
#: distinct whole matrix across all five, while the same call at 60 deg of fov
#: does differ, so the recovery is sensitive and the constancy is real. No call
#: site in the tree overrides either argument: both `projection()` calls in
#: `app.py` pass `aspect` and nothing else. Clipping therefore starts at a
#: *fixed* 0.1 A in front of the eye, and 6.0 A is not where it starts. Of the
#: five framings the old sentence offered as evidence, the selection's nearest
#: point sits 0.41, 2.27, 3.27, 4.27 and 4.53 A from the eye -- all of them
#: well in front of the plane -- and each would have had to be drawn at 0.19,
#: 0.83, 0.83, 0.83 and 1.56 A, that is 3.9x to 31x closer than this floor,
#: before a single atom was clipped. **None of the five was a picture with its
#: subject clipped out of it.**
#:
#: Nor is it pinned to the fog, which is the one thing that does scale with the
#: distance: `span = max(distance * 0.55, 8.0)`, near plane at `span * 0.55` and
#: far at `span * 2.1` (`app.py`, `paintGL`). The shader term is
#: `f = clamp((v_depth - near) / (far - near), 0, 1)`, mixed by `f * f`, so it
#: is **zero below** the near plane. Measured on the subject itself, fog is
#: 0.0000 out to 4.4 A, 0.0166 at 6 A, 0.0843 at 8 A, and 0.6694 at 20 A and at
#: 40 A: haze that *grows* with distance and never hides a near subject.
#:
#: **What the floor is for, measured.** It is the bound that keeps the eye
#: outside the subject's own drawn spheres, which is
#: :data:`MAX_DRAWN_ANGLE_FILL`'s requirement read in angstrom: at 45 deg,
#: `0.30 / sin(0.25 * 22.5 deg)` = **3.061 A** of clearance per 0.30 A of drawn
#: radius. Left to itself :func:`fit_view` breaks exactly that, and for exactly
#: the selections this floor exists for, because their unfloored distance goes
#: to nearly nothing. Sweeping single atoms to 500 atoms within 8 A on the live
#: `Camera()` basis, a 2-atom pair within 0.3 A is framed at **0.180 A with the
#: eye 2.974 A inside its own drawn surface**, and 19 atoms within 1.0 A at
#: 2.567 A with the eye 1.224 A inside. Bisecting the floor against that bound
#: over the same sweep puts the boundary at **3.7901 A** -- the largest value at
#: which some swept selection still has the eye inside -- so 6.0 A is 1.58x it
#: and holds 2.210 A of clearance on the worst case. The boundary does not move
#: with the frame's aspect ratio (3.7901 A at 1.0, 1.21, 1.5 and 2.0), because
#: what sets it is the selection's own view-space depth, not the frame's width.
#: That is this constant's origin, and it is a number the product would have to
#: be wrong about to invalidate.
#:
#: **What it costs, measured.** The camera stands further back than the frame
#: needs, so a selection too small to spend the distance loses its share of the
#: picture: 19 atoms within 1.0 A fill 0.792 of the half-frame at their
#: unfloored 2.567 A and **0.337** at 6.0 A, which is under
#: :data:`MIN_SELECTION_FRAME_FILL`. The two constants disagree by
#: construction rather than by accident: at 6.0 A a selection's view-space
#: half-extent has to be 0.994 A to reach a 0.40 fill, so everything tighter is
#: a speck *because* it was floored. On the shipped fixture the trade costs
#: nothing -- over the nine poses of `examples/poses.pdbqt` the unfloored fit
#: runs **10.725-15.340 A**, so the floor binds on 0 of 9 and the tightest,
#: pose 1, clears it by 4.725 A. It is insurance that has never been spent.
#:
#: **How it is enforced, and by what.** By the `max()` in :func:`fit_view` and
#: by the `max()` in :func:`focus_target`, on the **achieved** distance rather
#: than the caller's request, so `min_distance` can raise it and cannot lower
#: it: over 441 `fit_view` calls spanning 1 to 500 atoms, radii 0.0001 to 20 A,
#: offsets of 0, 3 and 50 A and `min_distance` of 0.0 and of -5.0, **zero**
#: results came back below the floor. :func:`selection_is_degenerate`'s
#: distance clause is not part of that and does not claim to be -- it has no
#: product caller and no product path can reach it, so it is the gate's
#: instrument over those two `max()` calls and over the sweep below.
#:
#: **The sweep, which is where the number came from.** Sweeping those same
#: shapes over distances from 0.05 to 5.99 A, **four** framings below this floor
#: are accepted by the other two instruments and refused by the distance clause
#: alone: nineteen atoms within 1.0 A at 3.0, 4.0 and 5.0 A (fills 0.678, 0.509
#: and 0.407) and fifty within 2.0 A at 5.99 A (0.805), every one of them
#: centred and inside the frame, which is why neither other instrument could see
#: them. Four rather than five, and the fifth is the correction: a 2-atom pair
#: within 0.3 A at 0.5 A fills only **0.275** on the live basis, so the fill
#: clause refuses it too and the distance clause is not the only instrument
#: that sees it. 5.99 A is the furthest distance the sweep reaches, and this
#: constant is the next value above it, so the floor is a measured origin and
#: not a round number for being round.
MIN_SELECTION_DISTANCE = 6.0

#: How far outside the selection's own view-space extent the camera centre may
#: sit, as a fraction of that extent's larger half-axis.
#:
#: A view-space statement and not a world-space one, because that is the
#: question. "Is the camera pointed at the selection" is a question about where
#: the selection lands in the *picture*, and :func:`fit_view` answers it in the
#: camera's own right/up axes. The first version of this check compared the
#: centre against the selection's world-space bounding-box midpoint and
#: **rejected the correct framing**: a selection that is flat and tilted has its
#: world bbox centre displaced along the view direction from the midpoint of
#: its projected extents, by 1.37 A on the shipped example, and the tolerance
#: was 0.25 x 5.38 A. The check was measuring a disagreement between two
#: definitions of "middle" rather than a camera pointing somewhere else.
CENTRE_TOLERANCE = 0.15

#: How much of the half-frame the selection must reach, after projection, or
#: the framing is degenerate.
#:
#: The two bounds around it matter and they fail differently. Below this floor
#: the camera is too far out and the pose is a speck; above 1.0 the selection is
#: outside the picture, which is the bug :func:`fit_view`'s perspective term
#: exists to prevent. :func:`fit_view` produces `1 / SELECTION_MARGIN` = 0.83 of
#: the frame along the binding axis, and the measured worst cases over the nine
#: poses of `examples/poses.pdbqt` are 0.49 (pose 3, whose selection is the
#: deepest along the view axis and therefore the most magnified) to 0.86.
#:
#: The first version of this constant asked whether the pose's extent *along
#: the view axis* was a share of the selection's, and that question has no
#: answer to give: the extent along the view axis does not depend on how far
#: back the camera is, so a camera 43 A out and one 17 A out score identically.
#: A guard whose two arguments are independent of the quantity it exists to
#: bound is not a guard.
MIN_SELECTION_FRAME_FILL = 0.40

# Radius, in angstrom, of the receptor context the pair framing is obliged to
# hold in the frame, measured on atom centres from the pair's midpoint.
#
# **This constant is what bounds the pair's size on screen, and it is the only
# thing that does.** The first version of this rule carried a second term, a
# legibility floor of `MIN_PAIR_FRAME_FILL` on the pair's own drawn extent, and
# that term could not bind: the context contains the subject, and the two fits
# ask for different fills (0.83 of the half-frame for the context, 0.40 for the
# subject), so `d_context >= need / 0.83 > need / 0.40 = d_subject` for every
# input. A constraint that can never bind is not a constraint, and its comment
# claimed a guarantee it did not deliver -- that "a pair may not be a smaller
# thing on screen than a selected pose is allowed to be", which the measurement
# then refused at 0.060-0.178 of the half-frame against a stated floor of 0.40.
# The gate caught the dead term, not the product. It is gone.
#
# So the shell is the whole of the trade-off, and the sweep is the evidence for
# where it sits. Measured on the shipped fixture, the distance this shell
# produces and the pair's own share of the half-frame at it:
#
#     shell     atoms   distance      pair fill
#       6.0        20   14.1-20.6 A    0.109-0.281
#       8.0        36   24.2-30.6 A    0.060-0.178
#      10.0        61   29.1-35.7 A    0.050-0.150
#      12.0        80   35.4-41.9 A    0.040-0.126
#      14.0       109   41.8-48.2 A    0.034-0.108
#
# 6.0 A is rejected on its own numbers: at that distance `ribbon` and `cartoon`
# fail the eye-outside-the-drawn-surface bound by 2.35 A, and in space-filling a
# single 0.69 A oxygen is 32 px across with its partner 60 px away, so the two
# discs the reader is being asked to tell apart overlap. 12.0 A and 14.0 A are
# rejected from the other side: the pair is 0.040-0.126 of the half-frame, a
# 40-125 px object on a 989 px frame, which is the "a shape rather than a
# speck" end of the same trade-off. 8.0 A is where the eye is outside the
# surface in all six with 5.3-16.1 A to spare and the pair is still 0.060-0.178
# of the frame; the frames for 6.0 and 12.0 were looked at to draw that line.
#
# The shell is in *centres* and deliberately not in drawn radii, so switching
# representation changes how far the camera stands off and never *what* is in
# the frame. That separation is what makes the six representations comparable.
PAIR_CONTEXT_SHELL = 8.0

# No single drawn object may subtend more than this fraction of the frame's
# half-angle. The derived form of "there is a picture here and not a wall".
#
# An object of drawn radius `r` at depth `t` from the eye subtends an angular
# half-width of `asin(r / t)`, and the frame subtends `fov / 2`. So the bound is
# `t >= r / sin(fill * fov / 2)`, and it is a *closed form*: there is no
# threshold here to choose, only a fraction of the field of view to say how much
# of it one object is allowed to be. A quarter of the half-angle is a quarter of
# the frame's width as a bound on one drawn object, which leaves the frame room
# for the rest of the context; at 1.0 the bound degenerates to "nothing may be
# nearer than the frame is wide", and below 0.1 the rule stops being about
# legibility and starts being about arithmetic.
#
# This is the bound the 8.125 A framing fails, and not narrowly: the nearest
# drawn surface there is 10.0-10.7 A *behind* the eye, so the slack is negative
# by more than the frame's own half-width in every one of the six
# representations.
MAX_DRAWN_ANGLE_FILL = 0.25



class FramingTarget(NamedTuple):
    """Where the camera should end up, and what it was aimed at.

    `pose_atoms` and `partner_atoms` are carried rather than recomputed by the
    caller so that a check can state what was framed, and a mismatch between
    the two is a finding rather than a silent difference.
    """

    center: np.ndarray
    distance: float
    pose_atoms: int
    partner_atoms: int
    partner_residues: tuple


class CameraMove(NamedTuple):
    """A camera transition, as a plan rather than as a side effect.

    The window drives this with a timer, but the arithmetic lives here so that a
    check can ask the only question worth asking about a transition: *where does
    it put the camera?* Asking it needs no window, no timer and no event loop,
    which is what makes "a transition that lands off-target" a checkable defect
    rather than a thing you have to watch happen and judge.

    `at(1.0)` returns the target exactly, not approximately: the eased curve is
    evaluated and then overridden at the end. A tween that asymptotically
    approaches its target leaves a camera that is 0.001 A off forever, and a
    check comparing floats has to either tolerate that or accidentally become a
    test of floating-point luck. The endpoint is a fact about the plan, so it is
    made exact.
    """

    start_center: np.ndarray
    start_distance: float
    target: FramingTarget
    steps: int = 14

    def at(self, t: float) -> tuple[np.ndarray, float]:
        """Camera state `steps * t` of the way through the move.

        Smootherstep rather than linear: a linear move starts and stops with a
        visible jerk, and on a 40 A swing that reads as the window flinching.
        """
        t = float(min(max(t, 0.0), 1.0))
        if t >= 1.0:
            return np.asarray(self.target.center, np.float32), float(self.target.distance)
        if t <= 0.0:
            return np.asarray(self.start_center, np.float32), float(self.start_distance)
        s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        a = np.asarray(self.start_center, np.float64)
        b = np.asarray(self.target.center, np.float64)
        centre = (a + (b - a) * s).astype(np.float32)
        distance = float(self.start_distance
                         + (float(self.target.distance) - float(self.start_distance)) * s)
        return centre, distance

    def settle(self) -> tuple[np.ndarray, float]:
        """The exact end state. The tween's last step, not its limit."""
        return np.asarray(self.target.center, np.float32), float(self.target.distance)


def _as_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(arr):
        raise ValueError("selection_points() got no points to frame")
    return arr


def fit_view(
    points,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    *,
    fov: float = 45.0,
    aspect: float = 1.0,
    margin: float = SELECTION_MARGIN,
    min_distance: float = MIN_SELECTION_DISTANCE,
) -> tuple[np.ndarray, float]:
    """Centre and distance that fit `points` into the frame with `margin` to spare.

    The fit is along the camera's own `right` and `up`, so "fits the frame" is
    measured in the axes the picture is actually drawn in. `margin` is how much
    of the frame's half-extent the larger of the two is allowed to fill: 1.0
    touches both edges.

    **The distance solves a per-point constraint, not a bounding-box one**, and
    that is not a refinement. A point offset `x` from the centre *and `z` toward
    the camera* projects to `x / ((d - z) * tan * aspect)`, so the distance that
    fits it is `z + x / (tan * aspect)` and the binding point is the one nearest
    the eye, not the one furthest out. Fitting the bounding box measured at the
    centre plane instead -- which is what the first version did, and what
    `Viewport.frame_all`'s `radius * 2.6` does -- assumes every point sits at
    the centre's depth.

    Measured on the shipped poses, that assumption is false for a third of them:
    for poses 3, 4 and 6 of `examples/poses.pdbqt` the selection is long along
    the view axis, and the box fit put the selection's own points up to
    **|ndc| = 3.89** -- four times outside the frame, in a framing whose whole
    purpose was to put the selection in it. Pose 3's union projected to
    |x|max 3.89 and |y|max 3.56 where the box fit predicted 0.83 and 0.53.

    Returns `(centre, distance)`. The centre is the midpoint of the two
    projected extents, which is inside the selection by construction; using the
    mean of the points instead would put the camera at the selection's centre of
    mass and push whichever face carries the contacts toward an edge.
    """
    pts = _as_points(points)
    r = np.asarray(right, np.float64)
    u = np.asarray(up, np.float64)
    f = np.asarray(forward, np.float64)
    x = pts @ r
    y = pts @ u
    z = pts @ f          # positive is further along the view direction
    mid = np.array([(x.max() + x.min()) * 0.5, (y.max() + y.min()) * 0.5])
    # Half the world extent the frame shows along each axis, per angstrom of
    # camera distance.
    tan_half = math.tan(math.radians(fov) * 0.5)
    # A point at depth `d + z` projects to `x / ((d + z) * tan * aspect)`, so
    # fitting it means `d >= x / (tan * aspect) - z`. `forward` is the view
    # *direction*, so a larger `z` is further from the eye and the term is
    # **subtracted**. Writing it as `+ z` -- which the first version of this did
    # -- asks for a distance that shrinks as the selection grows toward the
    # camera, and it made the one pose that needed the most room the pose that
    # overflowed worst: row 3's |ndc| went from 3.89 to 10.26.
    need = max(
        float((np.abs(x - mid[0]) / max(tan_half * max(aspect, 1e-3), 1e-6) - z).max()),
        float((np.abs(y - mid[1]) / max(tan_half, 1e-6) - z).max()),
    )
    # The floor is applied to the *achieved* distance, not only to the caller's
    # request, so `min_distance` can raise it and cannot lower it. Measured: with
    # the constant read only as the default, `fit_view` on a single atom with
    # `min_distance=0.0` returned 0.000 -- a camera at its own subject -- and
    # nothing in this module or in `scripts/` noticed. No caller passes
    # `min_distance` today, so for every existing call this is the same number it
    # was; it is a floor now rather than a default.
    distance = max(float(min_distance), need * float(margin), MIN_SELECTION_DISTANCE)
    # Un-project the midpoint of the two extents: the point of the selection's
    # own plane that the camera has to look at for it to sit centred.
    centre = (r * mid[0] + u * mid[1]).astype(np.float32)
    return centre, distance


def _drawn_inputs(points, radii) -> tuple[np.ndarray, np.ndarray]:
    """Points and their per-atom drawn half-widths, checked to line up.

    The check is not ceremony. A frame computed from a single averaged radius is
    a frame computed from a number that belongs to neither atom, and it fails in
    exactly the way that is hard to see: the picture still looks like something.
    """
    pts = _as_points(points)
    r = np.asarray(radii, np.float64).reshape(-1)
    if r.size != len(pts):
        raise ValueError(
            f"fit_drawn got {r.size} radii for {len(pts)} points; a per-atom "
            f"radius is the point, and a frame drawn from a single averaged "
            f"radius is a frame drawn from a number nobody chose")
    if np.any(r < 0.0):
        raise ValueError(f"a drawn radius cannot be negative: {r.min()!r}")
    return pts, r


def fit_drawn(
    points,
    radii,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    *,
    center=None,
    fov: float = 45.0,
    aspect: float = 1.0,
    fill: float = 1.0,
) -> float:
    """Distance at which the *drawn* discs reach `fill` of the half-frame.

    :func:`fit_view` fits atom centres. This fits what the program actually
    paints, which is a different distance, and the difference is the whole point:
    a space-filling oxygen and a small-sphere oxygen are the same atom and differ
    by 5.7x in what they cover on screen.

    The constraint is the per-point one :func:`fit_view` solves, with the drawn
    radius added to the lateral extent and subtracted from the depth, because the
    part of a drawn object that reaches furthest across the picture is its
    silhouette and the part that reaches furthest toward the eye is the same
    silhouette `r` closer:

        (|x - mid_x| + r) / (tan * aspect * fill)  -  (z - r)

    **The depth term is measured from the centre the camera is centred on**, and
    `fill` divides the lateral term only. Both are differences from
    :func:`fit_view`, which uses the *absolute* depth of each point and applies
    its margin to the finished `need`, so it scales the depth term by the margin
    as well. Measured on a 3.25 A pair translated 50 A along the view axis,
    `fit_view` returns a distance of **0.000** where the correct one is 11.9 A,
    and on a pair placed obliquely it returns 7.108 A where the framing actually
    achieved is 0.768 of the half-frame rather than the 0.833 asked for. It is
    exact only when the selection's bounding plane passes through the origin,
    which is why :data:`MIN_SELECTION_FRAME_FILL` is a floor well under 0.83 and
    not 0.83 itself. A rule that states an achieved fill has to be exact about
    it, so this one measures its depth from the centre and divides the term it
    means to divide. `fit_view` is left alone: it is pinned by measurements and a
    floor of its own, and its error is bounded and documented there, which is not
    the same as being right to fix silently.

    `center` fixes the point the extents are measured from. It defaults to the
    midpoint of the selection's own view-space extents, and the pair framing
    passes the *subject's* midpoint instead, so that a context which is lopsided
    about its pair moves the camera back rather than moving the pair sideways in
    its own picture.
    """
    pts, r = _drawn_inputs(points, radii)
    u_r = np.asarray(right, np.float64)
    u_u = np.asarray(up, np.float64)
    u_f = np.asarray(forward, np.float64)
    # Everything below is measured *from the point the camera is centred on*, so
    # that translating the scene cannot change the answer. `right`, `up` and
    # `forward` are orthonormal, so the default centre -- the point in the
    # selection's own plane at the midpoint of its projected extents -- has no
    # component along `forward` and the two cases are the same expression.
    if center is None:
        x0, y0, z0 = pts @ u_r, pts @ u_u, pts @ u_f
        c = (u_r * ((x0.max() + x0.min()) * 0.5)
             + u_u * ((y0.max() + y0.min()) * 0.5)
             + u_f * ((z0.max() + z0.min()) * 0.5))
    else:
        c = np.asarray(center, np.float64).reshape(3)
    rel = pts - c
    x, y, z = rel @ u_r, rel @ u_u, rel @ u_f
    tan_half = math.tan(math.radians(fov) * 0.5)
    f = float(fill)
    if f <= 0.0:
        raise ValueError(f"fit_drawn got fill={f!r}; a frame cannot be filled "
                         f"to zero of itself")
    return max(
        float((((np.abs(x) + r)
                / max(tan_half * max(aspect, 1e-3), 1e-6)) / f - (z - r)).max()),
        float((((np.abs(y) + r)
                / max(tan_half, 1e-6)) / f - (z - r)).max()),
    )


def drawn_fill(
    points,
    radii,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    distance: float,
    *,
    center=None,
    fov: float = 45.0,
    aspect: float = 1.0,
) -> float:
    """Share of the half-frame the drawn discs actually reach, at `distance`.

    Computed from the projection rather than as a ratio of two distances, so it
    is a non-negative share with a unit both answers can be read against, and so
    it stays right when the camera is closer than the unprojected fit would put
    it -- which is what happens when the eye is inside the drawn surface, and
    which is the defect this whole line of work is about.
    """
    pts, r = _drawn_inputs(points, radii)
    c = np.asarray(center, np.float64).reshape(3) if center is not None else (
        pts.mean(axis=0))
    rel = pts - c
    z = rel @ np.asarray(forward, np.float64)
    x = rel @ np.asarray(right, np.float64)
    y = rel @ np.asarray(up, np.float64)
    tan_half = math.tan(math.radians(fov) * 0.5)
    depth = float(distance) + z - r
    depth = np.maximum(depth, 1e-6)
    return max(
        float(((np.abs(x) + r) / (depth * tan_half * max(aspect, 1e-3))).max()),
        float(((np.abs(y) + r) / (depth * tan_half)).max()),
    )


def drawn_envelope_slack(
    points,
    radii,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    distance: float,
    *,
    center=None,
    fov: float = 45.0,
    fill: float = MAX_DRAWN_ANGLE_FILL,
) -> float:
    """How far the eye is outside every drawn object, in angstrom.

    The bound is `t >= r / sin(fill * fov / 2)` per object, from
    ``asin(r / t) <= fill * fov / 2``: the smallest depth at which an object of
    drawn radius `r` can be no wider than that share of the frame. Positive means
    the frame cannot be one object; negative means the eye is inside the drawn
    surface, and the picture is the inside of a sphere with the subject drawn on
    top of it.

    Reported in angstrom so the failure can be read as a distance rather than as
    a ratio: at 8.125 A the nearest drawn surface is 10.0 A behind the eye, so
    this returns about -10 A, which is larger in magnitude than the frame's own
    half-width at that distance.
    """
    pts = _as_points(points)
    r = np.asarray(radii, np.float64).reshape(-1)
    c = np.asarray(center, np.float64).reshape(3) if center is not None else (
        pts.mean(axis=0))
    z = (pts - c) @ np.asarray(forward, np.float64)
    t = float(distance) + z
    # sin of a quarter of the half-angle; the floor keeps a zero or negative
    # `fill` from turning the bound into a division by zero.
    limit = math.sin(max(min(float(fill), 0.999), 1e-3)
                      * math.radians(float(fov)) * 0.5)
    return float((t - r / max(limit, 1e-6)).min())


class PairFraming(NamedTuple):
    """Where a pair framing puts the camera, and the numbers behind it.

    Carried rather than recomputed by callers so a check can read the rule's own
    terms -- the subject's fill of the frame, the context's fill, how far the eye
    is outside the drawn envelope -- instead of re-deriving the formula and
    thereby agreeing with the implementation by construction.
    """

    center: np.ndarray
    distance: float
    subject_fill: float
    context_fill: float
    context_atoms: int
    shell: float
    envelope_slack: float


def focus_target(
    subject_coords,
    subject_radii,
    context_coords,
    context_radii,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    *,
    fov: float = 45.0,
    aspect: float = 1.0,
    context_fill: float = 1.0 / SELECTION_MARGIN,
    shell: float = PAIR_CONTEXT_SHELL,
    min_distance: float = MIN_SELECTION_DISTANCE,
) -> PairFraming:
    """The camera that frames `subject` and holds `context` at arm's length.

    The subject is the thing the gesture is about -- two atoms, or a residue's
    contacts -- and the camera is **centred on it**. The context is the
    neighbourhood that makes it identifiable, and it sets the **distance** and
    nothing else. Splitting them that way is the point: a rule that centres on
    the context instead puts the subject wherever the context's asymmetry leaves
    it, which is the same complaint as a subject sitting at the edge of its own
    picture, arrived at from the other direction.

    The distance is **one** fit: the subject and its context together, at
    `context_fill`, with the camera centred on the subject. There is a second
    term a reader might expect here -- a floor on how large the pair itself
    reaches -- and there is deliberately not one, because it cannot bind: the
    context contains the subject and the two fits ask for different fills, so
    `d_context > d_subject` for every input. See :data:`PAIR_CONTEXT_SHELL`, which
    is the constant that actually bounds the pair's size, and what it costs.

    The fit is of *drawn* radii, so the answer depends on the representation.
    That is the property the old `radius * 5` lacked: it returned 8.125 A for all
    six modes on a fixture where the six modes differ by 5.7x in what they draw
    at one atom.
    """
    subject, s_r = _drawn_inputs(subject_coords, subject_radii)
    context, c_r = _drawn_inputs(context_coords, context_radii)
    if not len(context):
        context, c_r = subject, s_r

    centre = subject.mean(axis=0)
    all_pts = np.vstack([subject, context])
    all_r = np.concatenate([s_r, c_r])
    d_context = fit_drawn(all_pts, all_r, right, up, forward, center=centre,
                          fov=fov, aspect=aspect, fill=context_fill)
    # The achieved distance, floored as in `fit_view`: the same constant, for
    # the same reason, and the same guarantee that a caller passing a smaller
    # `min_distance` gets the floor rather than the request.
    distance = max(float(min_distance), d_context, MIN_SELECTION_DISTANCE)
    # What the fills came out as, measured off the projection at the distance
    # actually chosen rather than inferred from the fits.
    at_one = drawn_fill(subject, s_r, right, up, forward, distance, center=centre,
                        fov=fov, aspect=aspect)
    at_one_ctx = drawn_fill(all_pts, all_r, right, up, forward, distance,
                            center=centre, fov=fov, aspect=aspect)
    slack = drawn_envelope_slack(all_pts, all_r, right, up, forward, distance,
                                 center=centre, fov=fov)
    return PairFraming(
        center=np.asarray(centre, np.float32),
        distance=float(distance),
        subject_fill=float(at_one),
        context_fill=float(at_one_ctx),
        context_atoms=int(len(context)),
        shell=float(shell),
        envelope_slack=float(slack),
    )


def selection_points(pose_coords, partner_coords) -> np.ndarray:
    """The points a selection frames: the pose, and the atoms it touches."""
    pose = _as_points(pose_coords)
    partners = np.asarray(partner_coords, np.float64).reshape(-1, 3)
    if not len(partners):
        return pose
    return np.vstack([pose, partners])


def selection_target(
    pose_coords,
    partner_coords,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    *,
    fov: float = 45.0,
    aspect: float = 1.0,
    margin: float = SELECTION_MARGIN,
    min_distance: float = MIN_SELECTION_DISTANCE,
    partner_residues: tuple = (),
) -> FramingTarget:
    """The camera a pose selection should move to. See the module docstring."""
    pose = _as_points(pose_coords)
    partners = np.asarray(partner_coords, np.float64).reshape(-1, 3)
    centre, distance = fit_view(
        selection_points(pose, partners),
        right,
        up,
        forward,
        fov=fov,
        aspect=aspect,
        margin=margin,
        min_distance=min_distance,
    )
    return FramingTarget(
        center=centre,
        distance=distance,
        pose_atoms=int(len(pose)),
        partner_atoms=int(len(partners)),
        partner_residues=tuple(partner_residues),
    )


def framed_radius(points, center, right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> float:
    """How large a sphere the camera is actually framing, in angstrom.

    Measured against all three view axes, not one, because a framing can be
    correct across the screen and still be pulled a long way back along the view
    direction -- which is what happens when something other than the selection
    was fitted.
    """
    pts = _as_points(points)
    c = np.asarray(center, np.float64)
    d = pts - c
    axes = (np.asarray(right, np.float64), np.asarray(up, np.float64),
            np.asarray(forward, np.float64))
    return float(max(np.abs(d @ a).max() for a in axes))


def selection_is_degenerate(
    target: FramingTarget,
    pose_coords,
    partner_coords,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    *,
    fov: float = 45.0,
    aspect: float = 1.0,
) -> tuple[bool, str]:
    """True when a "framing" centres the picture without framing the selection.

    A framing can satisfy every naive requirement -- the scene is centred, it is
    all in the picture, nothing is cut off -- and be useless, by pointing the
    camera at the *receptor* and calling the selection framed because the
    selection is inside the picture. That is the shape of the defect this whole
    change is about, so it is named here as a predicate rather than left to a
    threshold that would have to be re-guessed.

    Three independent facts have to hold, and each is a different instrument:

    * **the eye is outside the selection's drawn spheres** -- the camera is at
      least :data:`MIN_SELECTION_DISTANCE` out. Checked first, and first because
      the other two are ratios taken from a camera position, so every one of them
      is measured from an eye that may be inside the thing it is measuring.
      **This clause is unreachable from the product**, and the reason is
      structural rather than incidental: both :func:`fit_view` and
      :func:`focus_target` floor the achieved distance, so nothing this module
      produces can arrive here below the floor. The floor's enforcement is those
      two `max()` calls. Measured over 441 `fit_view` calls spanning 1 to 500
      atoms, radii 0.0001 to 20 A, offsets of 0, 3 and 50 A and `min_distance`
      of 0.0 and of -5.0, zero results came back below it.
    * **the centre is on the selection** -- within
      :data:`CENTRE_TOLERANCE` of the selection's own extent, measured along the
      camera's right and up axes. Cheap, exact, and answerable with no
      framebuffer. This is what catches a framing pointed at the receptor.
    * **the selection fills the frame it was given** -- it spans at least
      :data:`MIN_SELECTION_FRAME_FILL` of the frame's own half-extent along its
      binding axis, and no more than the whole of it. Also camera-side, and this
      is what catches a framing pointed at the selection from so far away that
      the selection is a speck, or from so near that it overflows. **Unlike the
      first clause this one does fire on the product's own output**, because the
      distance floor pushes a tight selection back and costs it its share of the
      frame: 19 atoms within 1.0 A are accepted at their unfloored 2.567 A
      with a fill of 0.792, and rejected at the floor's 6.0 A with 0.337.

    The split matters and is not a detail of wording. The predicate as a whole is
    reachable from the product and the distance clause inside it is not, so
    "this checks the framing" is true of the fill and centre clauses and false
    of the distance one. A caller wiring this into a camera-move path should
    know it is the fill floor it is really enforcing -- and that the fill floor
    and the distance floor disagree for any selection with a view-space
    half-extent under 0.994 A, where being floored *is* what makes the fill too
    small.

    They are separate because they fail separately, and a single combined
    threshold cannot say which. The first version of the second check compared
    the pose's half-extent **along the view axis** against the selection's
    half-extent **maximised over all three axes**, which made the correct
    framing fail at every margin (4.14 A against a 5.62 A floor) -- two
    quantities on different axes, compared as if they were one. And it could
    not have caught a zoomed-out framing at all: the extent along the view axis
    does not depend on how far back the camera is, so a camera 43 A out scored
    exactly the same as one 17 A out.

    The pixel share in `scripts/framing_selection_check.py` is the third
    instrument and the only one that measures what the user sees; it needs a
    framebuffer and skips without one. No two of the three are sufficient alone:
    the centre and fill checks would accept a framing that put the selection
    dead centre and microscopic, the distance check would accept one from across
    the receptor pointed at nothing in particular, and the pixel floor would
    accept one that owns the frame with a protein and leaves the ligand at 0.15%.

    Returns `(degenerate, reason)`.
    """
    pose = _as_points(pose_coords)
    partners = np.asarray(partner_coords, np.float64).reshape(-1, 3)
    pts = selection_points(pose, partners)
    r = np.asarray(right, np.float64)
    u = np.asarray(up, np.float64)
    c = np.asarray(target.center, np.float64)

    # The first instrument, and the only one that is not a ratio. The other two
    # measure the selection against the frame from wherever the camera is, so a
    # camera inside its own subject makes both of them read comfortably: the
    # selection can be dead centre and entirely inside the picture while the eye
    # is 0.5 A from a 0.3 A pair. **The reason it is still wrong is the drawn
    # surface, not the depth clip** -- `Camera.projection` takes `znear=0.1` as a
    # constant, so nothing is clipped at 0.5 A. What is wrong at 0.5 A is that
    # 0.30 / sin(0.25 * 22.5 deg) = 3.061 A of clearance is owed and there is
    # 0.5 A of it. Measured: of the selections swept from one atom to 500 atoms
    # within 8 A, four framings below this floor are accepted by the two
    # instruments below -- nineteen within 1.0 A at 3.0, 4.0 and 5.0 A (fills
    # 0.678, 0.509, 0.407) and fifty within 2.0 A at 5.99 A (0.805) -- every
    # one of them centred and in frame, so this clause is the only one that
    # sees them. Four and not five: a 2-atom pair within 0.3 A at 0.5 A fills
    # only 0.275 on the live basis, and the fill clause refuses it too.
    #
    # **This clause is unreachable from the product, and says so rather than
    # implying otherwise.** `fit_view` and `focus_target` both apply the floor
    # to the achieved distance, and over 441 `fit_view` calls -- including
    # `min_distance` of 0.0 and of -5.0 -- nothing came back below it.
    # `focus_target` cannot even be an argument here: it returns a
    # `PairFraming`, not a `FramingTarget`. So a `True` from this branch means
    # the distance was set by hand, and the floor's real enforcement is the
    # `max()` in those two functions. See :data:`MIN_SELECTION_DISTANCE`.
    if float(target.distance) < MIN_SELECTION_DISTANCE:
        return True, (
            f"the camera sits {float(target.distance):.2f} A from a selection it "
            f"is supposed to be framing, against a floor of "
            f"{MIN_SELECTION_DISTANCE:g} A: the eye is inside its own subject's "
            f"drawn spheres, which owe it 0.30 / sin(0.25 * 22.5 deg) = 3.061 A "
            f"of clearance. **Not** the depth clip, which is a constant "
            f"`znear=0.1` and does not move with this number. Both of the other "
            f"instruments pass at this distance -- the framing can be perfectly "
            f"centred and entirely inside the frame -- which is why this one is "
            f"checked on its own and not folded into them. Reaching here means "
            f"the distance was set by hand rather than by `fit_view` or "
            f"`focus_target`, both of which apply the floor, and no product path "
            f"can do that"
        )

    # The selection's extent in the axes the picture is drawn in. This is the
    # frame the camera has to point into for the selection to be the subject.
    sx = pts @ r
    sy = pts @ u
    half = np.array([(sx.max() - sx.min()) * 0.5, (sy.max() - sy.min()) * 0.5])
    mid = np.array([(sx.max() + sx.min()) * 0.5, (sy.max() + sy.min()) * 0.5])
    span = max(float(half.max()), 1e-6)
    off = np.array([float(c @ r) - mid[0], float(c @ u) - mid[1]])
    outside = float(np.abs(off).max())
    if outside > CENTRE_TOLERANCE * span:
        return True, (
            f"the camera centre sits {outside:.2f} A outside the selection's own "
            f"{span:.2f} A projected extent, against a tolerance of "
            f"{CENTRE_TOLERANCE * span:.2f} A ({CENTRE_TOLERANCE:.0%} of it): the "
            f"picture is centred on something else. A centred, fully-visible "
            f"frame that is not pointed at the selection is the degenerate "
            f"case, and it satisfies every threshold that only asks whether the "
            f"scene is in shot"
        )

    # The second instrument, and the one that answers both remaining questions
    # at once: how much of the frame the selection actually spans, *after
    # projection*. Not the world-space half-extent over the frame's world
    # half-extent -- that is an orthographic ratio, and it is blind to
    # perspective, which is exactly how a selection long toward the camera ends
    # up outside the frame while this ratio still reads comfortable. `fit_view`
    # had that bug and it put pose 3's own atoms at |ndc| 3.89.
    fill = projected_fill(pts, target, right, up, forward, fov=fov, aspect=aspect)
    if fill > 1.0:
        return True, (
            f"the selection reaches {fill:.2f} of the way to the frame edge, so "
            f"part of it is outside the picture: a framing that crops the very "
            f"thing it was asked to frame. `fit_view` solves for this per point "
            f"and at its own depths; getting here means the distance was set by "
            f"something other than `fit_view`"
        )
    if fill < MIN_SELECTION_FRAME_FILL:
        return True, (
            f"the selection spans {fill:.0%} of the half-frame it was given, "
            f"against a floor of {MIN_SELECTION_FRAME_FILL:.0%}: the camera is "
            f"{target.distance:.1f} A out and the selection is a speck in the "
            f"middle of it. This is the other degenerate framing -- pointed at "
            f"the right thing, from far enough away that the pose is again a "
            f"fraction of a percent, and nothing about it is cut off"
        )
    return False, ""


def projected_fill(
    points,
    target: FramingTarget,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    *,
    fov: float = 45.0,
    aspect: float = 1.0,
) -> float:
    """How much of the half-frame `points` reaches, after projection.

    1.0 is the frame edge. This is the *projected* span rather than a world-space
    ratio because perspective is part of the answer: a point nearer the camera
    than the centre is magnified, and a selection that is deep along the view
    direction can leave the picture while a world-space extent ratio says it
    comfortably fits.
    """
    pts = _as_points(points)
    c = np.asarray(target.center, np.float64)
    f = np.asarray(forward, np.float64)
    d = pts - c
    depth = float(target.distance) + d @ f
    if float(depth.min()) <= 1e-6:
        # A point at or behind the eye plane has no projection. Reported as a
        # span of infinity rather than divided by, so a check fails on it
        # instead of on a number that happens to be small.
        return float("inf")
    tan_half = math.tan(math.radians(float(fov)) * 0.5)
    nx = np.abs(d @ np.asarray(right, np.float64)) / (
        depth * tan_half * max(float(aspect), 1e-3)
    )
    ny = np.abs(d @ np.asarray(up, np.float64)) / (depth * tan_half)
    return float(max(nx.max(), ny.max()))
