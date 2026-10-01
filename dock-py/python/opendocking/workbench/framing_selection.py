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
    "projected_fill",
    "fit_view",
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

#: Floor on the distance, in angstrom, so a one-atom selection cannot put the
#: eye inside it. The near plane is `distance * 0.55 * 0.55`, so below about
#: 3 A the pose itself would be clipped by its own depth range.
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
    distance = max(float(min_distance), need * float(margin))
    # Un-project the midpoint of the two extents: the point of the selection's
    # own plane that the camera has to look at for it to sit centred.
    centre = (r * mid[0] + u * mid[1]).astype(np.float32)
    return centre, distance


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

    Two independent facts have to hold, and each is a different instrument:

    * **the centre is on the selection** -- within
      :data:`CENTRE_TOLERANCE` of the selection's own extent, measured along the
      camera's right and up axes. Cheap, exact, and answerable with no
      framebuffer. This is what catches a framing pointed at the receptor.
    * **the selection fills the frame it was given** -- it spans at least
      :data:`MIN_SELECTION_OCCUPANCY` of the frame's own half-extent along its
      binding axis. Also camera-side, and this is what catches a framing
      pointed at the selection from so far away that the selection is a speck.

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
    framebuffer and skips without one. Neither pair is sufficient alone: the
    camera-side pair would accept a framing that put the selection dead centre
    and microscopic, and the pixel floor would accept one that owns the frame
    with a protein and leaves the ligand at 0.15%.

    Returns `(degenerate, reason)`.
    """
    pose = _as_points(pose_coords)
    partners = np.asarray(partner_coords, np.float64).reshape(-1, 3)
    pts = selection_points(pose, partners)
    r = np.asarray(right, np.float64)
    u = np.asarray(up, np.float64)
    c = np.asarray(target.center, np.float64)

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
