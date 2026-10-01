"""Cartoon geometry: a ribbon that twists for a helix, an arrow for a strand.

# What this file is for

The viewport had two modes, `ribbon` and `cartoon`, and they called the same
function with the same arguments. What a cartoon is, and what a ribbon is not,
is *secondary structure in the geometry*: a helical ribbon that turns about the
CA axis, and a flat arrow whose head points from N to C. The classifier that
knows which residue is which already existed and agreed with the published 1CRN
annotation on 43 of 46 residues. The geometry simply was not reading it.

So this module reads it, in the one way a viewer can see without a legend: the
shape of the thing, not its colour.

    helix   a wide flat ribbon whose cross-section rotates about the path by a
            phase-locked angle, so consecutive turns line up
    sheet   a flat ribbon of constant width that flares into an arrowhead over
            the last residues of the strand, the head at the C-terminal end
    coil    a thin square-section tube -- what `geometry.ribbon` already drew

# The frame, and why the obvious one is wrong

Every ribbon needs a frame: at each point along the path, two unit vectors
perpendicular to the tangent, saying which way is across the ribbon and which
way is through its thickness. The obvious construction takes a fixed world axis
-- usually +Z -- and crosses it with the tangent at every point. That is wrong,
and wrong in a way that raises nothing.

A world-axis frame is *discontinuous wherever the tangent runs parallel to that
axis*. There `cross(tangent, up)` passes through zero, and the direction it
emerges on the far side is the negative of the one it had before. One residue
pair at the top of the domain flips the whole cross-section through 180 degrees:
the ribbon's face turns inside out for one interval and then continues in the
opposite handedness. That is a seam, not a twist.

The construction used here is **parallel transport** (a rotation-minimising
frame): the side vector is carried from one point to the next by re-projecting
it into the new tangent's plane, which is the smallest rotation satisfying the
new tangent. A frame built that way has no seam anywhere and introduces no
twist of its own -- whatever rotation the path contains is the only rotation in
it. That is what makes it the right base to *impose* a known twist on.

# The helix twist, and what "phase-locked" has to mean

The cross-section is rotated about the tangent by

    theta(x) = 360 degrees * (x - x_run) / RESIDUES_PER_TURN

where `x` is the **continuous residue coordinate** along the trace -- the dense
samples are spread evenly from residue 0 to residue N-1, so x is just that
spacing -- and `x_run` is the coordinate of the first residue of the current
helix run. `RESIDUES_PER_TURN` is 3.6, an alpha helix.

Two consequences, and the second is the one that matters:

* the rate is constant at 100 degrees per residue, so the picture reads as a
  helix rather than as a wavy ribbon;
* because theta is a continuous function of position along the trace and the
  rate is exact, x and x+3.6 give orientations differing by exactly 360
  degrees. **That is what phase-locked means here**, and it is why the argument
  is a continuous coordinate rather than a residue index: stepping theta once
  per residue would give the same average rate but a faceted cross-section with
  a 12.5 degree jump at every residue boundary, and the ribbon would read as a
  stack of flat plates rather than as a smooth spiral.

**A correction that was measured rather than assumed.** This was first written
as a twist proportional to *arc length*, at one turn per `HELIX_RISE` = 5.4 A.
5.4 A is the helix's **axial rise per turn**, not the length of the path through
one turn: an alpha helix's CA trace runs 3.6 x 3.8 = 13.7 A of arc per turn.
Measuring the delivered geometry read 265 degrees per residue instead of 100,
which is 13.7/5.4 = 2.54 times too much -- the ratio the two lengths predict.
The number to twist by is a fact about residues per turn, so it is now written
as one.

The twist is verified from the mesh, not from this code's arithmetic --
`cartoon_cross_section_axes` reads the orientation back out of the vertex
positions the renderer will draw, and `scripts/representation_cartoon_check.py`
asks for that orientation to return after one turn.

# The arrowhead

A strand is a flat ribbon of constant width flaring into a head over its last
`HEAD_RESIDUES` residues, so the widest cross-section on a strand is at its
C-terminal end and the flare points N to C along the chain. That direction is a
checkable fact rather than a drawing convention: `head_widths` measures the
width at both ends of every strand run, and the guard requires the C end to be
the wide one on *every* run with a margin, so a head at the wrong end is
rejected rather than merely looking wrong in one screenshot.

# What this file deliberately does not do

It does not change `geometry.ribbon`. That is the `ribbon` representation's own
geometry and other gates measure it; the cartoon is a second builder beside it,
not an edit of it. The two share the curve and the frame helpers, so there is
one implementation of each rather than two.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# `_perpendicular` and `_tangent_at` are the two helpers that carry the
# parallel-transport reasoning, and they are imported under their existing names
# rather than re-implemented: a second copy of the degenerate-case handling in
# `_perpendicular` is a second chance to get the zero cross-section wrong, which
# is a bug this repository has already shipped once (see its docstring).
from .geometry import MeshData, _perpendicular, _tangent_at, catmull_rom

__all__ = [
    "HELIX_RISE",
    "HELIX_TWIST_PER_RESIDUE",
    "HEAD_RESIDUES",
    "RESIDUES_PER_TURN",
    "SHEET_HEAD_FACTOR",
    "WIDTHS",
    "THICKNESSES",
    "COLORS",
    "cartoon",
    "cartoon_cross_section_axes",
    "half_widths",
    "head_widths",
    "orientation_rate",
    "parallel_transport_frame",
    "ring_count",
    "ring_states",
    "runs",
]

#: Residues per turn of an alpha helix. The cross-section returns to the same
#: orientation after this many residues, which is the phase lock. A 3-10 helix
#: (3.0 residues/turn) would want 3.0 and a pi helix (4.0) 4.0; this file does
#: not distinguish them, because the classifier calls all three `helix` and
#: choosing a rate per run would need a per-residue helix subtype it does not
#: have. The consequence is bounded and stated rather than hidden -- a 3-10 run
#: is drawn with a 3.6-residue period and so reads as slightly under-twisted.
RESIDUES_PER_TURN = 3.6

#: The helix's *axial* rise per turn, 1.5 A x 3.6 residues. **Not** used to
#: compute the twist -- it is the projection of one turn onto the helix axis,
#: whereas the twist rate is a fact about residues per turn. It is kept because
#: it is the number a reader reaches for first, and a module that silently
#: contradicted it would be the harder thing to debug.
HELIX_RISE = 5.4

#: 360 / 3.6, the degrees of cross-section rotation per residue of an alpha
#: helix. Kept as its own constant because it is the number a helix is compared
#: against; deriving it means there is one definition rather than two literals
#: that can drift apart.
HELIX_TWIST_PER_RESIDUE = 360.0 / RESIDUES_PER_TURN

#: Residues from the C-terminal end of a strand that the arrowhead occupies.
#: Two is the shortest that gives the head visible length: ~3 A of flare at 1.5 A
#: per CA step, resolved by ~16 rings at the default 8 samples per residue.
HEAD_RESIDUES = 2

#: A sheet arrowhead is wider than the strand it ends. 2.6x sits inside the range
#: the usual cartoon renderers use, and it is also the factor the guard turns
#: into a margin: the C end must measure at least twice the N end.
SHEET_HEAD_FACTOR = 2.6

#: Half-widths in Angstrom. Coil is deliberately the pair `geometry.ribbon`
#: already used, so the coil is the same thin square tube as before and the only
#: thing that changed about it is which builder draws it.
WIDTHS = {"helix": 1.70, "sheet": 1.50, "coil": 0.35}
THICKNESSES = {"helix": 0.28, "sheet": 0.30, "coil": 0.35}

COLORS = {
    "helix": (0.85, 0.25, 0.30),
    "sheet": (0.95, 0.80, 0.25),
    "coil": (0.62, 0.66, 0.72),
}

#: Vertices per ring. The mesh is a swept quad strip, one ring of four corners
#: per dense sample, and this is the contract that lets a checker read a ring's
#: cross-section back out of a flat vertex array.
RING = 4


def ring_count(mesh: MeshData) -> int:
    """How many cross-section rings a swept-quad cartoon mesh carries."""
    return len(mesh.positions) // RING


def parallel_transport_frame(
    points: np.ndarray, seed: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-minimising ``(tangent, side, across)`` along ``points``.

    Three ``(n, 3)`` arrays of unit vectors. ``seed`` orients the first sample
    and nothing else, because parallel transport only needs a starting
    direction; every later frame is carried from its predecessor.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        z = np.tile(np.array([0.0, 0.0, 1.0]), (max(n, 0), 1))
        return z.copy(), z.copy(), z.copy()

    tangents = np.empty((n, 3), np.float64)
    sides = np.empty((n, 3), np.float64)
    acrosses = np.empty((n, 3), np.float64)

    tangents[0] = _tangent_at(pts, 0, n)
    first = np.array([0.0, 0.0, 1.0]) if seed is None else np.asarray(seed, float)
    sides[0] = _perpendicular(tangents[0], first)
    acrosses[0] = np.cross(tangents[0], sides[0])
    for i in range(1, n):
        tangents[i] = _tangent_at(pts, i, n)
        # The carried side, not the world axis. This line is the whole
        # difference between a continuous frame and one with a 180-degree seam
        # wherever the path runs parallel to up.
        sides[i] = _perpendicular(tangents[i], sides[i - 1])
        acrosses[i] = np.cross(tangents[i], sides[i])
    return tangents, sides, acrosses


def _arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along ``points``, starting at 0.

    Not used by the builder any more -- the twist is a function of the residue
    coordinate, see the module docstring -- and kept because a guard that wants
    to ask "how many Angstrom of path has this helix run" needs it, and because
    deleting a helper the moment it stops being called by one function is how
    the next caller re-implements it.
    """
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def runs(states: Sequence[str]) -> list[tuple[int, int, str]]:
    """Maximal runs of equal state as ``(start, end_inclusive, state)``."""
    out: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append((start, i - 1, states[start]))
            start = i
    return out


def cartoon(
    trace: np.ndarray,
    side_vectors: np.ndarray,
    states: Sequence[str],
    *,
    per_segment: int = 8,
    widths: dict[str, float] | None = None,
    thicknesses: dict[str, float] | None = None,
    colors: dict[str, tuple[float, float, float]] | None = None,
    twist_degrees: float = HELIX_TWIST_PER_RESIDUE,
    head_residues: int = HEAD_RESIDUES,
) -> MeshData:
    """Build the cartoon mesh for one trace and one per-residue state list.

    ``trace`` are the CA positions, ``side_vectors`` per-residue orientations
    that only seed the first frame, and ``states`` the per-residue class. The
    result is a :class:`~.geometry.MeshData` swept quad strip of four vertices
    per dense sample, in the order ``+u+v, -u+v, -u-v, +u-v`` where ``u`` is the
    (possibly twisted) half-width direction and ``v`` the half-thickness one.

    A trace shorter than two CA atoms has no ribbon to sweep and returns an
    empty mesh, the same contract `geometry.ribbon` keeps.
    """
    trace = np.asarray(trace, dtype=np.float64).reshape(-1, 3)
    if len(trace) < 2:
        return MeshData(
            *(np.zeros((0, 3), np.float32) for _ in range(3)), np.zeros(0, np.uint32)
        )

    states = [s if s in WIDTHS else "coil" for s in list(states)]
    if len(states) < len(trace):
        states += ["coil"] * (len(trace) - len(states))

    side_vectors = np.asarray(side_vectors, dtype=np.float64).reshape(-1, 3)
    if len(side_vectors) != len(trace):
        side_vectors = np.tile(np.array([0.0, 0.0, 1.0]), (len(trace), 1))

    widths = {**WIDTHS, **(widths or {})}
    thicknesses = {**THICKNESSES, **(thicknesses or {})}
    colors = {**COLORS, **(colors or {})}

    dense = catmull_rom(trace, per_segment)
    n_res = len(trace)
    idx = np.clip(
        np.round(np.linspace(0, n_res - 1, len(dense))).astype(int), 0, n_res - 1
    )
    # The *continuous* residue coordinate of each dense sample. `idx` above is
    # the same spacing rounded to an integer residue, which is what decides
    # which class a ring belongs to; the twist needs the unrounded value,
    # because a theta stepped once per residue puts a 12.5 degree jump at every
    # residue boundary and the ribbon reads as a stack of plates.
    x_res = np.linspace(0.0, float(n_res - 1), len(dense))
    dense_state = [states[i] for i in idx]

    tangents, sides, acrosses = parallel_transport_frame(dense, side_vectors[idx[0]])

    # Per-residue half-widths. The arrowhead is defined on residues and the
    # twist on the continuous coordinate; computing the widths here keeps the
    # two definitions from being applied to the same number by accident.
    res_width = np.empty(n_res, np.float64)
    for lo, hi, state in runs(states):
        body = widths[state]
        for j in range(lo, hi + 1):
            if state == "sheet" and (hi - j) < head_residues:
                frac = (j - (hi - head_residues)) / float(max(head_residues, 1))
                res_width[j] = body * (1.0 + frac * (SHEET_HEAD_FACTOR - 1.0))
            else:
                res_width[j] = body

    # Residue coordinate at which each helix run starts, so theta is measured
    # from the start of its own run and the phase lock holds run by run rather
    # than only across the whole chain.
    helix_origin = np.full(len(dense), np.nan)
    for lo, _hi, state in runs(states):
        if state != "helix":
            continue
        at_start = np.where(idx == lo)[0]
        if len(at_start):
            helix_origin[at_start[0] :] = float(x_res[at_start[0]])

    positions, normals, cols, indices = [], [], [], []
    n = len(dense)
    for i in range(n):
        res = int(idx[i])
        state = dense_state[i]
        w = res_width[res]
        t = thicknesses[state]
        col = colors[state]

        theta = 0.0
        if state == "helix":
            base = helix_origin[i]
            x0 = 0.0 if np.isnan(base) else base
            # 360 degrees per RESIDUES_PER_TURN of residue coordinate. The
            # per-residue form is converted through the constant rather than
            # used directly, so the two cannot disagree.
            per_residue = twist_degrees / HELIX_TWIST_PER_RESIDUE
            theta = np.radians(per_residue * 360.0 / RESIDUES_PER_TURN * (x_res[i] - x0))
        ct, st = float(np.cos(theta)), float(np.sin(theta))
        u = sides[i] * ct + acrosses[i] * st
        v = -sides[i] * st + acrosses[i] * ct

        positions.extend(
            [
                dense[i] + u * w + v * t,
                dense[i] - u * w + v * t,
                dense[i] - u * w - v * t,
                dense[i] + u * w - v * t,
            ]
        )
        normals.extend([v, -v, -u, u])
        cols.extend([col] * 4)

        if i > 0:
            base_idx = (i - 1) * RING
            for k in range(RING):
                a = base_idx + k
                b = base_idx + (k + 1) % RING
                indices.extend([(a, b, a + RING), (b, b + RING, a + RING)])

    return MeshData(
        np.asarray(positions),
        np.asarray(normals),
        np.asarray(cols),
        np.asarray(indices, dtype=np.uint32),
    )


def cartoon_cross_section_axes(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """Read each ring's width and thickness axes back out of the mesh.

    Returns ``(u, v)``, two ``(rings, 3)`` unit arrays. Ring ``i`` occupies
    vertices ``4i .. 4i+3`` in the order the builder emits them, so

        u_i = normalize(p[4i] - p[4i+1])   half-width direction
        v_i = normalize(p[4i] - p[4i+3])   half-thickness direction

    Deliberately a *measurement* rather than a re-run of the builder's
    arithmetic. A guard that asked the builder what twist it applied would be
    answered by the very mutation it exists to catch; the vertex positions the
    renderer will draw cannot be, because the mutation changes those too.
    """
    pos = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, RING, 3)
    axes = []
    for a, b in ((0, 1), (0, 3)):
        d = pos[:, a, :] - pos[:, b, :]
        norm = np.linalg.norm(d, axis=1)
        safe = norm > 1e-9
        unit = np.zeros_like(d)
        unit[safe] = d[safe] / norm[safe, None]
        axes.append(unit)
    return axes[0], axes[1]


def half_widths(mesh: MeshData) -> np.ndarray:
    """Per-ring half-width, read off the mesh rather than assumed."""
    pos = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, RING, 3)
    return np.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1) / 2.0


def ring_states(mesh: MeshData, states: Sequence[str]) -> list[str]:
    """The per-residue state each ring came from.

    Resampling is the builder's rule -- round the even spacing of rings onto
    residue indices -- derived rather than stored, because a mesh is positions
    and indices and the state list is the only other thing a caller holds. The
    check that this agrees with the builder is in
    `scripts/representation_cartoon_check.py`, because a resampling rule that
    drifts turns every per-class measurement into a measurement of the wrong
    residues.
    """
    rings = ring_count(mesh)
    n_res = len(states)
    idx = np.clip(np.round(np.linspace(0, n_res - 1, rings)).astype(int), 0, n_res - 1)
    return [states[i] for i in idx]


def orientation_rate(mesh: MeshData, states: Sequence[str]) -> dict:
    """How fast each ring's cross-section turns about the path, in deg/residue.

    The measurement the "cartoon is a recoloured ribbon" guard stands on. It
    reads the cross-section axes back out of the mesh (see
    `cartoon_cross_section_axes`), takes the tangent from the ring centres, and
    measures the *signed* angle from one ring's half-width axis to the next
    about that tangent:

        d_i = atan2( (u_i x u_{i+1}) . t_i , u_i . u_{i+1} )

    Normalising by the residue step gives degrees per residue. Sign matters and
    is why this is not a variance: a ribbon whose frame is merely *noisy* has a
    high standard deviation and a mean near zero, and noise is what a
    parallel-transport frame actually contributes. Measured on prepared 1CRN, the
    shipped ribbon reads mean -0.11 deg/residue with a standard deviation of
    7.02, while the cartoon reads +100.58 with 7.07 -- **the same noise, a
    hundred-fold different mean**. A variance-based discriminator was tried first
    and is the wrong instrument here: it barely moves between the two, because
    what separates them is a bias, not a spread.

    Returns a dict with ``rate`` (one value per ring-gap), ``ring_state``,
    ``x`` (the continuous residue coordinate of each ring), and the per-class
    mean/standard-deviation summaries keyed by state.
    """
    u, _v = cartoon_cross_section_axes(mesh)
    pos = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, RING, 3)
    centre = pos.mean(axis=1)
    rings = len(centre)

    tangent = np.zeros_like(centre)
    tangent[:-1] = centre[1:] - centre[:-1]
    tangent[-1] = centre[-1] - centre[-2]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.where(norm > 1e-12, tangent / np.where(norm > 1e-12, norm, 1.0), 0.0)

    sin = np.einsum("ij,ij->i", np.cross(u[:-1], u[1:]), tangent[:-1])
    cos = np.einsum("ij,ij->i", u[:-1], u[1:])
    delta = np.degrees(np.arctan2(sin, cos))

    x = np.linspace(0.0, max(len(states) - 1.0, 1.0), rings)
    # Degrees per residue: a gap covers `1 / residues_per_ring` of a residue, so
    # divide by that rather than multiplying by a literal 8. The factor is
    # derived from this mesh's own ring count, so a change to `per_segment`
    # cannot silently rescale every number a guard reads.
    residues_per_ring = max(len(states) - 1.0, 1.0) / max(rings - 1, 1)
    rate = delta / residues_per_ring

    rs = ring_states(mesh, states)
    both = [
        (rs[i], rs[i + 1]) for i in range(rings - 1)
    ]
    summary = {}
    for state in sorted(set(rs)):
        m = np.array([a == state and b == state for a, b in both], bool)
        if m.any():
            summary[state] = (float(rate[m].mean()), float(rate[m].std()), int(m.sum()))
    return {
        "rate": rate,
        "delta": delta,
        "ring_state": rs,
        "x": x,
        "residues_per_ring": residues_per_ring,
        "summary": summary,
        "u": u,
    }


def head_widths(
    mesh: MeshData, states: Sequence[str]
) -> list[tuple[int, int, float, float, str]]:
    """Half-width at both *ends* of every run of one state, measured off a mesh.

    Returns ``(res_lo, res_hi, width_at_lo, width_at_hi, state)`` per run, with
    each width averaged over only the rings that came from the run's first and
    last residue. The arrowhead's direction is then a fact about this list rather
    than a belief: a head at the wrong end appears as ``width_at_lo >
    width_at_hi`` on every run, which is what the guard asks about.

    The first version of this averaged over the whole run, which measured a
    1.00 ratio for a sheet whose head was 2.6x its body -- the head is two
    residues of a three-residue strand, so the run mean washes it out almost
    exactly. Endpoints are the quantity the claim is about.
    """
    ring_state = ring_states(mesh, states)
    widths = half_widths(mesh)
    rings = len(widths)
    span = max(rings - 1, 1)
    res_span = max(len(states) - 1, 1)
    out = []
    for lo, hi, state in runs(list(states)):
        r_lo = int(round(lo * span / res_span))
        r_hi = int(round(hi * span / res_span))
        if r_hi <= r_lo:
            continue
        # Only the rings of *this* run's own endpoints, not the run's middle:
        # an average over the whole run is what hid the head the first time.
        at_lo = [float(w) for w, s in zip(widths[r_lo : r_lo + 2], ring_state[r_lo : r_lo + 2])
                 if s == state]
        at_hi = [float(w) for w, s in zip(widths[r_hi - 1 : r_hi + 1], ring_state[r_hi - 1 : r_hi + 1])
                 if s == state]
        if not at_lo or not at_hi:
            continue
        out.append((lo, hi, float(np.mean(at_lo)), float(np.mean(at_hi)), state))
    return out
