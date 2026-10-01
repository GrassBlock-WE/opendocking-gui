"""Triangle geometry for the representations the workbench can draw.

Kept apart from the Qt and moderngl code, and apart from the chemistry in
:mod:`opendocking.workbench.structure`, because geometry is arithmetic on
coordinates: it has no idea what a residue is, and it can be checked without a
display. The three things it builds are

* **spheres**, for the space-filling view;
* **cylinders** along bonds, for ball-and-stick and for the skeletal view. Each
  bond is drawn as two half-cylinders, one per atom, so the two halves can carry
  the two atoms' colours -- the split at the midpoint is what makes a
  carbon-oxygen bond readable at a glance;
* **a swept ribbon** along a backbone trace, for proteins.

Every routine returns interleaved-free parallel arrays: positions, normals,
colours and an index buffer. One vertex layout for all three, so the renderer
needs a single draw function.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "MeshData",
    "unit_cylinder",
    "spheres",
    "bonds",
    "ribbon",
    "catmull_rom",
    "dashed_segments",
]


class MeshData:
    """Positions, per-vertex normals, per-vertex colours and triangle indices."""

    __slots__ = ("positions", "normals", "colors", "indices")

    def __init__(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)
        self.colors = np.ascontiguousarray(colors, dtype=np.float32)
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32)

    @property
    def empty(self) -> bool:
        return self.positions.size == 0 or self.indices.size == 0

    def __len__(self) -> int:
        return 0 if self.empty else int(self.indices.size // 3)


def unit_cylinder(segments: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit cylinder: radius 1, length 1, running from the origin along +z.

    Returns ``(verts, normals, faces)``. There are no end caps, because every
    use here butts a half-cylinder against another one and an open end inside a
    bond is never visible; adding them would double the triangle count for
    nothing.
    """
    verts = np.zeros((segments + 1, 3), dtype=np.float64)
    normals = np.zeros((segments + 1, 3), dtype=np.float64)
    for s in range(segments):
        a = 2.0 * math.pi * s / segments
        nx, ny = math.cos(a), math.sin(a)
        normals[s] = (nx, ny, 0.0)
        verts[s] = (nx, ny, 0.0)
    verts[segments] = (0.0, 0.0, 1.0)
    normals[segments] = (0.0, 0.0, 0.0)

    faces = []
    for s in range(segments):
        a = s
        b = (s + 1) % segments
        faces.append((a, b, segments))
    return verts, normals, np.asarray(faces, dtype=np.uint32)


def _frames(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``direction``.

    The reference axis is chosen as the world axis least aligned with the bond
    direction. Picking a fixed axis instead degenerates: a bond pointing along
    that axis gets a zero-length cross product and the basis silently becomes
    NaN, which draws as nothing at all rather than as an error.
    """
    d = direction / max(float(np.linalg.norm(direction)), 1e-9)
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(d)))] = 1.0
    u = np.cross(d, helper)
    n = np.linalg.norm(u)
    if n < 1e-9:  # pragma: no cover - guarded by the axis choice above
        helper = np.array([1.0, 0.0, 0.0])
        u = np.cross(d, helper)
        n = np.linalg.norm(u)
    u = u / n
    w = np.cross(d, u)
    return u, w


def spheres(
    centers: np.ndarray,
    radii: np.ndarray | float,
    colors: np.ndarray,
) -> MeshData:
    """One sphere per atom. ``centers`` is ``(n, 3)``."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n = len(centers)
    if n == 0:
        return MeshData(*(np.zeros((0, 3), np.float32) for _ in range(3)), np.zeros(0, np.uint32))
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (n,))
    colors = np.asarray(colors, dtype=np.float64).reshape(n, 3)

    # A UV sphere rather than the icosphere the viewer used before: here the
    # mesh is expanded once per atom on the CPU either way, and this form makes
    # the pole handling obvious.
    segments, rings = 12, 8
    verts, normals = [], []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            nrm = np.array(
                [
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi),
                ]
            )
            verts.append(nrm)
            normals.append(nrm)
    verts_a = np.asarray(verts)
    normals_a = np.asarray(normals)

    faces = []
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + s
            d = (r + 1) * segments + (s + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    faces_a = np.asarray(faces, dtype=np.uint32)

    v = len(verts_a)
    positions = (centers[:, None, :] + radii[:, None, None] * verts_a[None, :, :]).reshape(-1, 3)
    normals = np.tile(normals_a, (n, 1))
    colors = np.repeat(colors, v, axis=0)
    shifts = (np.arange(n, dtype=np.uint32) * np.uint32(v))[:, None, None]
    indices = (faces_a[None, :, :] + shifts).reshape(-1)
    return MeshData(positions, normals, colors, indices)


def bonds(
    coords: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    radius: float,
    colors: np.ndarray,
    *,
    segments: int = 10,
    two_tone: bool = True,
) -> MeshData:
    """Cylinders along ``pairs``.

    With ``two_tone`` each bond becomes two half-cylinders meeting at the
    midpoint, the first taking the colour of the first atom and the second the
    colour of the other. That is what makes ball-and-stick read correctly: a
    uniform cylinder hides which half of it belongs to which atom, and the
    element boundary is the whole point of the model.
    """
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    pairs = [tuple(p) for p in pairs if p[0] < len(coords) and p[1] < len(coords)]
    if not pairs:
        return MeshData(*(np.zeros((0, 3), np.float32) for _ in range(3)), np.zeros(0, np.uint32))

    unit_v, unit_n, unit_f = unit_cylinder(segments)
    positions, normals, cols = [], [], []
    index_blocks = []
    vertex_offset = 0

    for i, j in pairs:
        a, b = coords[i], coords[j]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            # Two atoms on the same point have no direction; there is nothing to
            # draw and no way to draw it that would not be a spike through the
            # origin.
            continue
        u, w = _frames(d)
        if two_tone:
            halves = [(a, a + d * 0.5, colors[i]), (a + d * 0.5, b, colors[j])]
        else:
            halves = [(a, b, colors[i])]
        for start, end, col in halves:
            seg = end - start
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            axis = seg / seg_len
            u, w = _frames(axis)
            # Local (x, y, z) -> world: the cylinder's z runs start -> end, its
            # radius is `radius` in the plane spanned by u and w. Written as one
            # expression so there is no half-applied transform to be left over.
            world = (
                u[None, :] * (unit_v[:, 0] * radius)[:, None]
                + w[None, :] * (unit_v[:, 1] * radius)[:, None]
                + axis[None, :] * (unit_v[:, 2] * seg_len)[:, None]
                + start[None, :]
            )
            norm_world = (
                u[None, :] * unit_n[:, 0:1]
                + w[None, :] * unit_n[:, 1:2]
                + axis[None, :] * unit_n[:, 2:3]
            )
            positions.append(world)
            normals.append(norm_world)
            cols.append(np.tile(col, (len(world), 1)))
            index_blocks.append(unit_f + np.uint32(vertex_offset))
            vertex_offset += len(world)

    if not positions:
        return MeshData(*(np.zeros((0, 3), np.float32) for _ in range(3)), np.zeros(0, np.uint32))
    return MeshData(
        np.vstack(positions),
        np.vstack(normals),
        np.vstack(cols),
        np.concatenate(index_blocks),
    )


def catmull_rom(points: np.ndarray, per_segment: int = 8) -> np.ndarray:
    """A smooth curve through ``points``.

    Endpoints are duplicated so the curve starts and ends exactly on the first
    and last residue rather than falling short of them -- a ribbon that stops
    half a residue early looks like a rendering fault, and it would be one.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        return points
    padded = np.vstack([points[0], points, points[-1]])
    out = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for t in np.linspace(0.0, 1.0, per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            )
    out.append(points[-1])
    return np.asarray(out)


def _perpendicular(tangent: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``tangent``, preferring ``seed``.

    The seed is projected into the tangent's plane and used when that
    projection is a real vector. When it is not -- the seed runs parallel or
    antiparallel to the tangent, so the projection is exactly zero -- a short
    list of world axes is tried in turn and the first that survives the
    projection is used instead. The result is therefore *always* a genuine unit
    vector perpendicular to the tangent, which is the property the ribbon's
    parallel transport depends on.

    This exists because the code it replaced got that wrong in a way that
    produced no error at all. It projected the seed, and when the projection
    came back zero it projected a ``(0, 0, 1)`` fallback **onto the same
    tangent** -- which is zero again for a z-aligned trace -- and then
    normalised the zero vector by ``max(norm, 1e-9)``. The cross-section
    collapsed, every triangle had zero area, and the ribbon simply drew
    nothing: measured, 0 of 160 triangles with any area and 44 of 84 vertices
    carrying a zero normal. A viewer renders that as an absent ribbon, which
    reads as "no secondary structure here" rather than as a bug.
    """
    for candidate in (seed, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
        vec = np.asarray(candidate, dtype=np.float64)
        projected = vec - tangent * float(vec @ tangent)
        norm = float(np.linalg.norm(projected))
        if norm > 1e-6:
            return projected / norm
    # Unreachable for a unit tangent: at most one of the three world axes can
    # be parallel to it. Kept so the function returns rather than raises.
    return np.array([0.0, 0.0, 1.0])  # pragma: no cover


def ribbon(
    trace: np.ndarray,
    side_vectors: np.ndarray,
    states: Sequence[str],
    *,
    per_segment: int = 8,
    width: dict[str, float] | None = None,
    thickness: dict[str, float] | None = None,
    colors: dict[str, tuple[float, float, float]] | None = None,
) -> MeshData:
    """A swept ribbon following a backbone trace.

    ``trace`` are the guide points (one per residue, normally the CA atoms),
    ``side_vectors`` a per-residue direction used to orient the flat face so the
    ribbon does not spin about its own axis, and ``states`` the per-residue
    secondary structure. A helix and a strand are drawn as wide flat ribbons and
    a coil as a narrow tube, all through the same swept-quad code, so there is
    one piece of geometry to get right rather than three.

    The cross-section is a rectangle in the frame ``(side, tangent x side)``.
    A coil wants a square section and gets one from the same code by asking for
    equal width and thickness.
    """
    trace = np.asarray(trace, dtype=np.float64).reshape(-1, 3)
    if len(trace) < 2:
        return MeshData(*(np.zeros((0, 3), np.float32) for _ in range(3)), np.zeros(0, np.uint32))
    side_vectors = np.asarray(side_vectors, dtype=np.float64).reshape(-1, 3)
    if len(side_vectors) != len(trace):
        side_vectors = np.tile(np.array([0.0, 0.0, 1.0]), (len(trace), 1))

    width = width or {"helix": 1.6, "sheet": 1.5, "coil": 0.35}
    thickness = thickness or {"helix": 0.22, "sheet": 0.22, "coil": 0.35}
    colors = colors or {
        "helix": (0.85, 0.25, 0.30),
        "sheet": (0.95, 0.80, 0.25),
        "coil": (0.62, 0.66, 0.72),
    }

    dense = catmull_rom(trace, per_segment)
    # Per-sample state and side vector, resampled alongside the curve.
    idx = np.clip(
        np.round(np.linspace(0, len(trace) - 1, len(dense))).astype(int),
        0,
        len(trace) - 1,
    )
    dense_state = [states[i] if i < len(states) else "coil" for i in idx]
    seed_side = side_vectors[idx]

    positions, normals, cols, indices = [], [], [], []
    n = len(dense)

    # Orientation is carried along the curve rather than re-derived per sample.
    # Taking the side vector fresh at every residue looks reasonable and is not:
    # the CA-to-CB direction swings by tens of degrees between neighbouring
    # residues, so the cross-section rotates at each one and the ribbon comes
    # out visibly beaded, with its face turning inside out along the chain.
    # Parallel transport keeps the frame continuous -- the previous side vector
    # is re-projected onto the new tangent's plane, which is the smallest
    # rotation that satisfies the new tangent, so no twist accumulates.
    tangent0 = _tangent_at(dense, 0, n)
    side = _perpendicular(tangent0, seed_side[0])

    for i in range(n):
        tangent = _tangent_at(dense, i, n)
        if i > 0:
            side = _perpendicular(tangent, side)
        across = np.cross(tangent, side)

        state = dense_state[i]
        w = width.get(state, width["coil"])
        t = thickness.get(state, thickness["coil"])
        col = colors.get(state, colors["coil"])

        # Four corners: +-w across `side`, +-t along `across`.
        corners = [
            dense[i] + side * w + across * t,
            dense[i] - side * w + across * t,
            dense[i] - side * w - across * t,
            dense[i] + side * w - across * t,
        ]
        face_normals = [across, -across, -side, side]
        positions.extend(corners)
        normals.extend(face_normals)
        cols.extend([col] * 4)

        if i > 0:
            base = (i - 1) * 4
            for k in range(4):
                a = base + k
                b = base + (k + 1) % 4
                c = a + 4
                d = b + 4
                indices.extend([(a, b, c), (b, d, c)])

    return MeshData(
        np.asarray(positions), np.asarray(normals), np.asarray(cols),
        np.asarray(indices, dtype=np.uint32),
    )


def _tangent_at(points: np.ndarray, i: int, n: int) -> np.ndarray:
    """Unit tangent at sample ``i`` by central difference."""
    prev = points[max(i - 1, 0)]
    nxt = points[min(i + 1, n - 1)]
    tangent = nxt - prev
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return tangent / norm


def dashed_segments(
    starts: np.ndarray,
    ends: np.ndarray,
    dash: float = 0.40,
    gap: float = 0.26,
) -> tuple[np.ndarray, np.ndarray]:
    """Split straight segments into dashes, for the interaction lines.

    A hydrogen bond drawn as a solid line is read as "these atoms are bonded",
    which is precisely the claim being doubted — the point of the filter is
    that a human has not confirmed it. A dash says the same thing a dashed line
    has always said in a drawing: this is a proposed connection, not a fact.

    Returns ``(segments, group)``: an ``(n, 2, 3)`` array in the same layout
    `bonds` uses, and the ``(n,)`` index of the source segment each dash came
    from. Returning the grouping rather than only the geometry is what lets the
    caller colour each dash from its own contact without re-deriving how long
    that line was, which is a second copy of the same arithmetic and a second
    chance to disagree with it.

    Segments shorter than one dash stay solid rather than vanishing: a very
    short contact is a real contact, and clipping it to nothing would make the
    densest part of the interface the sparsest part of the picture.
    """
    starts = np.asarray(starts, np.float32).reshape(-1, 3)
    ends = np.asarray(ends, np.float32).reshape(-1, 3)
    if len(starts) == 0:
        return np.zeros((0, 2, 3), np.float32), np.zeros(0, np.int32)

    period = dash + gap
    out: list[np.ndarray] = []
    group: list[int] = []
    for index, (a, b) in enumerate(zip(starts, ends)):
        length = float(np.linalg.norm(b - a))
        if length <= 1e-6:
            out.append(np.stack([a, b]))
            group.append(index)
            continue
        n = int(length // period) + 1
        for k in range(n):
            t0 = (k * period) / length
            t1 = min(1.0, (k * period + dash) / length)
            if t1 <= t0:
                break
            out.append(np.stack([a + (b - a) * t0, a + (b - a) * t1]))
            group.append(index)
    if not out:
        return np.zeros((0, 2, 3), np.float32), np.zeros(0, np.int32)
    return np.asarray(out, np.float32), np.asarray(group, np.int32)