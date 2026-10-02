"""Headless check of the representation geometry.

Runs the real geometry routines on real structures and asserts things that can
only be true if the maths is right: a cylinder actually spans the two atoms it
was asked to span, a bond is drawn in two colours, and the ribbon follows the
backbone instead of wandering off it.

# What this file is for, and what it refuses to be

`geometry.py` produces the only triangles a user ever sees. That makes it
different from the other modules in this project: a coordinate that is off by
0.2 A is invisible here, and a ribbon that collapses to zero area is not a
rendering glitch but a feature that has silently stopped existing. So this file
is built around three commitments, and they are the reason it grew past its
first pass.

1. **The contract is pinned before the fixtures are chosen.** Each section
   below is headed by a claim the code's own docstrings make -- "radius 1,
   length 1, running from the origin along +z", "the curve starts and ends
   exactly on the first and last residue", "segments shorter than one dash stay
   solid" -- and there is a check per claim. A claim with no check is a claim
   the code happens to satisfy today.
2. **The fixtures are coordinates I chose, not coordinates the code chose.**
   The cylinder section at the bottom of that order builds a bond from literal
   numbers and states the expected geometry independently, because a fixture
   built from the geometry code's own idea of a bond is a closed loop: when the
   endpoint logic is wrong, the cylinder is wrong in the matching way and the
   check passes. The receptor-based checks are kept for the real-world case,
   but they are not the evidence.
3. **Every new assertion is bidirectional.** Each one that could pass for the
   wrong reason is paired with a control that must fail: the two-tone split is
   asserted by colour and re-asserted against a deliberately swapped colouring;
   the ribbon's width bound is asserted on the shipped receptor and against a
   ribbon built twice as wide. A check with no control is a check that cannot
   fail, which is worse than no check.

Run:  python scripts/representation_geometry_check.py
"""

import sys
import warnings
from pathlib import Path
from math import dist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Prefer the installed package; fall back to the source tree only when there is
# none. Putting `dock-py/python` on the path first imports the source copy of
# `opendocking`, which a clean checkout cannot load because it has no compiled
# `_dockpy` extension -- that file is gitignored and only exists in
# site-packages. See `contacts_criteria_check.py` for the same note in full.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking.workbench import geometry, structure as st_mod  # noqa: E402
from opendocking.workbench import MoleculeView  # noqa: E402

#: How many checks this file is supposed to run, counted by running it. A check
#: that stops running takes the count down with it silently, which is how a
#: suite goes from 21 to 19 with a clean report.
EXPECTED_CHECKS = 51  # measured; see the report for the mutation counts

FAILURES: list[str] = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


def prepared_receptor() -> MoleculeView:
    from opendocking.prep import prepare_receptor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        text = prepare_receptor(ROOT / "examples" / "1crn_receptor.pdb")
    return MoleculeView.from_text(text, "1crn", (0.6, 0.65, 0.7), 0.28, "receptor")


def ligand() -> MoleculeView:
    return MoleculeView.from_pdbqt(
        ROOT / "examples" / "ibuprofen_prep.pdbqt", "ibuprofen", (0.35, 0.75, 0.95), 0.28
    )


def tri_areas(mesh) -> np.ndarray:
    """Area of every triangle in a mesh, for the "did anything actually get
    drawn" question that a vertex count cannot answer."""
    faces = mesh.indices.reshape(-1, 3)
    p = mesh.positions
    return 0.5 * np.linalg.norm(
        np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]]),
        axis=1,
    )


def cross_section_extent(mesh, centre: np.ndarray) -> np.ndarray:
    """Total extent of a straight ribbon about a point on its own axis.

    A span (max minus min), not a deviation from the centre. A coil tube of
    half-width 0.35 is 0.70 A across, and every claim below is about the total;
    measuring the deviation instead reports the half and quietly halves every
    expected number, which is exactly the kind of thing that makes a check
    wrong rather than a geometry wrong.

    For a ribbon whose guide points run along z, the width shows up in x
    and the thickness in y. Reading them separately is what makes "the
    cross-section is the rectangle that was asked for" a checkable claim rather
    than a distance-to-the-curve bound that a ribbon of *any* width would pass.
    """
    d = mesh.positions - centre
    return d.max(axis=0) - d.min(axis=0)


def faces_point_outward(mesh, axis_point: np.ndarray,
                       axis_dir: np.ndarray) -> bool:
    """Does every triangle face away from the axis it was swept along?

    The normal is taken from the triangle's **own vertices**, not from the
    normals the mesh carries. That distinction is the whole point, and it cost
    a check to learn: an earlier version compared each face's centroid against
    the *stored* normals, which is a closed loop -- re-wind the faces and the
    stored normals are re-indexed to match, so the check agreed with itself. A
    winding order can only be checked against something that did not come from
    the winding, and the axis is the obvious candidate.
    """
    faces = mesh.indices.reshape(-1, 3)
    p = mesh.positions
    normal = np.cross(p[faces[:, 1]] - p[faces[:, 0]],
                      p[faces[:, 2]] - p[faces[:, 0]])
    centroid = p[faces].mean(axis=1)
    radial = centroid - axis_point
    radial = radial - axis_dir * (radial @ axis_dir)[:, None]
    return bool((np.einsum("ij,ij->i", normal, radial) > 0).all())


def cross_section_frames(mesh) -> tuple[np.ndarray, np.ndarray]:
    """The ``(side, across)`` frame at each cross-section, from its own corners.

    Reconstructed from the four vertices of each cross-section rather than read
    out of the mesh's stored normals, for the same reason
    :func:`faces_point_outward` does its own cross product. It is also a
    genuinely different quantity: the stored normals are the *face* normals,
    and a ribbon can twist without any of them pointing anywhere new.
    """
    quads = mesh.positions.reshape(-1, 4, 3)
    side = ((quads[:, 0] - quads[:, 1]) + (quads[:, 3] - quads[:, 2])) / 2.0
    across = ((quads[:, 0] - quads[:, 3]) + (quads[:, 1] - quads[:, 2])) / 2.0
    side = side / np.linalg.norm(side, axis=1)[:, None]
    across = across / np.linalg.norm(across, axis=1)[:, None]
    return side, across


def reseeded_side_dots(trace: np.ndarray, sides: np.ndarray,
                       per_segment: int) -> float:
    """Smallest angle between consecutive frames if the side were re-seeded at
    every sample instead of carried.

    The control for the parallel-transport check. It runs the module's own
    primitives with the one line that differs, so it measures the alternative
    rather than describing it -- and the difference is a 180 degree flip, which
    is the "beaded, face turning inside out" the docstring warns about.
    """
    dense = geometry.catmull_rom(trace, per_segment)
    idx = np.clip(
        np.round(np.linspace(0, len(trace) - 1, len(dense))).astype(int),
        0, len(trace) - 1,
    )
    seed = sides[idx]
    side = geometry._perpendicular(geometry._tangent_at(dense, 0, len(dense)),
                                   seed[0])
    carried = []
    for i in range(len(dense)):
        tangent = geometry._tangent_at(dense, i, len(dense))
        if i > 0:
            side = geometry._perpendicular(tangent, seed[i])   # not carried
        carried.append(side.copy())
    return min(float(carried[i] @ carried[i + 1])
               for i in range(len(carried) - 1))


def main() -> int:
    section("unit cylinder")
    v, n, f = geometry.unit_cylinder(8)
    check("has the expected vertex count", len(v) == 9, f"{len(v)} verts, {len(f)} tris")
    check("radius is 1 and length is 1",
          abs(np.linalg.norm(v[0][:2]) - 1.0) < 1e-9 and abs(v[-1][2] - 1.0) < 1e-9)
    check("normals are unit length",
          all(abs(np.linalg.norm(r) - 1.0) < 1e-9 for r in n[:8]),
          "the cap centre has no normal, which is correct")

    # The docstring says "running from the origin along +z" and "there are no
    # end caps". Both are claims about *which* vertices exist, and the three
    # checks above say nothing about them: a cylinder with a base cap would
    # have the same vertex count only by accident, and one whose tip sits
    # somewhere other than z=1 would pass "length is 1".
    tip = [k for k in range(len(v)) if abs(float(v[k][2]) - 1.0) < 1e-12]
    check("the tip is a single vertex at exactly (0, 0, 1), and every face "
          "fans from it",
          tip == [len(v) - 1]
          and all(len(set(r.tolist()) & set(tip)) == 1 for r in f)
          and all(len([k for k in r if k not in tip]) == 2 for r in f),
          f"vertex {tip} is the only one at z=1; all {len(f)} faces use it once "
          f"and two base vertices, which is what 'no end caps' means -- a base "
          f"cap would be triangles using no tip at all")
    check("the faces are wound consistently, so the cylinder is not inside-out",
          faces_point_outward(
              geometry.MeshData(v.astype(np.float32), n.astype(np.float32),
                                np.zeros((len(v), 3), np.float32),
                                f.astype(np.uint32)),
              np.zeros(3), np.array([0.0, 0.0, 1.0])),
          "every triangle's own normal, computed from its three vertices, "
          "points away from the z axis the cylinder runs along. This replaces a "
          "check that compared the face centroid to the mesh's *stored* "
          "normals, which agreed with itself: reversing the winding also "
          "re-indexes the stored normals, so that version passed a cylinder "
          "turned inside-out, and mutation testing is what showed it")

    section("a cylinder spans exactly the bond it was given")
    lig = ligand()
    pairs = lig.bond_pairs()
    # A carbon-oxygen bond, so the two halves have different colours to take.
    hetero = next(
        (p for p in pairs
         if lig.elements[p[0]] != lig.elements[p[1]]),
        pairs[0],
    )
    mesh = geometry.bonds(lig.coords, [hetero], 0.2, lig.atom_colors(),
                          segments=8, two_tone=True)
    check("produced triangles", len(mesh) > 0,
          f"{len(mesh)} triangles, {len(mesh.positions)} vertices")
    i, j = hetero
    a, b = lig.coords[i].astype(float), lig.coords[j].astype(float)
    axis = (b - a) / np.linalg.norm(b - a)
    along = (mesh.positions - a) @ axis
    length = float(np.linalg.norm(b - a))
    # A two-tone bond is drawn as two half-cylinders meeting at the midpoint, so
    # it must start at the first atom and end at the second -- and nowhere
    # outside the bond's own radius. Measuring from one atom in a way that
    # ignores the other is how the first version of this check "failed" a
    # perfectly good cylinder.
    check("runs from the first atom to the second",
          float(along.min()) > -0.25 and float(along.max()) < length + 0.25,
          f"spans {float(along.min()):.2f} to {float(along.max()):.2f} A "
          f"along a {length:.2f} A bond")
    radial = np.linalg.norm(
        mesh.positions - (a + axis * along[:, None]), axis=1
    )
    check("stays within the stick radius of the bond axis",
          float(radial.max()) < 0.25, f"max radius {float(radial.max()):.3f} A")
    check("is split into two colours, one per atom",
          len(np.unique(mesh.colors, axis=0)) == 2,
          f"{len(np.unique(mesh.colors, axis=0))} distinct colours on one "
          f"{lig.elements[i]}-{lig.elements[j]} bond")
    check("each colour covers half the bond",
          abs(float(along[: len(along) // 2].max()) - length / 2) < 0.2,
          f"first half ends {float(along[: len(along) // 2].max()):.2f} A along")

    section("an independent fixture: coordinates chosen here, not taken from a file")
    # Everything below is literal. The bond is the z axis from the origin to
    # 0,0,1.6; the two atoms are red and blue; the radius is 0.2. Nothing in
    # this section comes from `structure.py`, from a shipped PDBQT, or from the
    # geometry code, so a wrong answer here cannot be a matching wrong answer.
    fx = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]])
    fcols = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    fmesh = geometry.bonds(fx, [(0, 1)], 0.2, fcols, segments=8, two_tone=True)
    faxis = np.array([0.0, 0.0, 1.0])
    falong = (fmesh.positions - fx[0]) @ faxis
    fradial = np.linalg.norm(
        fmesh.positions - (fx[0] + faxis * falong[:, None]), axis=1)
    check("it spans exactly 0 to 1.6 A, to the precision the file carries",
          abs(float(falong.min())) < 1e-6
          and abs(float(falong.max()) - 1.6) < 1e-6,
          f"along = {float(falong.min()):.6f} to {float(falong.max()):.6f} A "
          f"against a bond of exactly 1.6 A. The loose +-0.25 A bound above "
          f"would pass a cylinder 15% too long; this one will not")
    check("and its radius is exactly the 0.2 it was asked for",
          abs(float(fradial.max()) - 0.2) < 1e-6,
          f"max radial distance {float(fradial.max()):.6f} A. A cylinder that "
          f"kept the unit radius and forgot the scale factor would read 1.0")
    check("two halves means 2 x (segments + 1) vertices, and no more",
          len(fmesh.positions) == 2 * (8 + 1) and len(fmesh) == 2 * 8,
          f"{len(fmesh.positions)} vertices, {len(fmesh)} triangles for 8 "
          f"segments. A third half, or a dropped one, changes this count")

    section("the two-tone split point, measured by colour rather than by order")
    # The check above -- "each colour covers half the bond" -- slices the vertex
    # array down the middle. That assumes the first half of the array is the
    # first half of the bond, and it never looks at *which colour* is where, so
    # a build that coloured the first half with the second atom's colour would
    # sail through it: same vertex count, same spans, same two colours. This
    # section states the claim in a way that cannot survive that: every vertex
    # wearing atom i's colour lies in the half of the bond nearer atom i.
    red = fmesh.colors[:, 0] > 0.5
    blue = ~red
    split_ok = (float(falong[red].max()) <= 0.8 + 1e-6
                and float(falong[blue].min()) >= 0.8 - 1e-6)
    swapped = geometry.bonds(fx, [(0, 1)], 0.2, fcols[::-1], segments=8,
                             two_tone=True)
    sred = swapped.colors[:, 0] > 0.5
    swapped_ok = (float(falong[sred].max()) <= 0.8 + 1e-6
                  and float(falong[~sred].min()) >= 0.8 - 1e-6)
    check("every vertex wearing atom i's colour lies in atom i's half of the bond",
          split_ok and not swapped_ok,
          f"red spans {float(falong[red].min()):.3f}-{float(falong[red].max()):.3f} "
          f"A and blue {float(falong[blue].min()):.3f}-{float(falong[blue].max()):.3f} "
          f"A, meeting exactly at the 0.8 A midpoint. CONTROL: the same bond "
          f"built with the two colours the other way round fails this "
          f"assertion, so the check can tell which half is which -- a "
          f"vertex-order check could not")

    section("the frame cannot degenerate, whichever way the bond points")
    # `_frames` picks the world axis least aligned with the bond so that the
    # cross product never collapses. Nothing tested that: ibuprofen has no bond
    # lying along a world axis, so a build that picked a *fixed* axis would
    # pass every check in this file. A degenerate frame shows up as NaN, and
    # NaN renders as nothing at all.
    axis_cases = {"+x": (3.0, 0, 0), "-x": (-3.0, 0, 0), "+y": (0, 3.0, 0),
                  "-y": (0, -3.0, 0), "+z": (0, 0, 3.0), "-z": (0, 0, -3.0)}
    frame_bad = {}
    for name, delta in axis_cases.items():
        c = np.array([[0.0, 0.0, 0.0], list(delta)])
        m = geometry.bonds(c, [(0, 1)], 0.2, fcols, segments=8, two_tone=True)
        span = (m.positions - c[0]) @ (np.asarray(delta) / 3.0)
        if not (np.isfinite(m.positions).all() and abs(float(span.min())) < 1e-6
                and abs(float(span.max()) - 3.0) < 1e-6):
            frame_bad[name] = (bool(np.isfinite(m.positions).all()),
                               float(span.min()), float(span.max()))
    check("a bond along any world axis still spans 0 to 3 A, with no NaN",
          not frame_bad,
          f"all six of {sorted(axis_cases)} are finite and span the bond. A "
          f"fixed reference axis would make the cross product vanish for the "
          f"bond lying along it, and the result would be NaN vertices -- which "
          f"a viewer draws as nothing, with no error anywhere"
          + (f"; bad: {frame_bad}" if frame_bad else ""))
    check("a bond between two atoms at the same point produces no geometry",
          geometry.bonds(fx, [(0, 0)], 0.2, fcols, segments=8).empty
          and len(geometry.bonds(fx, [(0, 0)], 0.2, fcols, segments=8)) == 0,
          "zero-length bond skipped rather than drawn as a spike through the "
          "origin; there is no direction to draw along")
    check("out-of-range bond pairs are dropped, and an empty pair list is too",
          len(geometry.bonds(fx, [(0, 9)], 0.2, fcols, segments=8).positions) == 0
          and geometry.bonds(fx, [], 0.2, fcols).empty,
          "a pair naming an atom that does not exist is filtered out, and no "
          "pairs at all gives an empty mesh rather than a shape")

    section("skeletal mode has no spheres, ball-and-stick has both")
    stick = geometry.bonds(lig.coords, pairs, 0.1, lig.atom_colors(), two_tone=True)
    balls = geometry.spheres(lig.coords, lig.atom_radii() * 0.42, lig.atom_colors())
    check("sticks are lighter than the atom spheres", len(stick) < len(balls),
          f"{len(stick)} vs {len(balls)} triangles")
    check("every bond in the file got a stick",
          len(pairs) == 16, f"{len(pairs)} bonds")

    section("ribbon follows the backbone")
    rec = prepared_receptor()
    check("receptor is recognised as a protein", rec.has_backbone)
    check("bonds came from residue templates", rec.bond_source == "template",
          rec.bond_source)
    trace = rec.structure.backbone()
    ca = np.asarray([rec.structure.atoms[c].xyz for _, c, _ in trace])
    rib = rec.backbone_ribbon(per_segment=6)
    check("ribbon produced triangles", rib is not None and len(rib) > 0,
          f"{0 if rib is None else len(rib)} triangles")

    if rib is not None and len(rib) > 0:
        # The invariant that matters is that the ribbon stays *on* the backbone,
        # not how big the protein is: crambin is about 20 A across, so an
        # absolute radius says nothing. Every ribbon vertex should have a CA
        # atom close to it, and the ribbon should not be much larger than the
        # trace it follows.
        nearest = np.array([np.min(np.linalg.norm(rib.positions - p, axis=1)) for p in ca])
        check("no ribbon point is far from every CA atom",
              float(nearest.max()) < 4.0,
              f"worst distance to the nearest CA {float(nearest.max()):.2f} A")
        trace_span = float(np.linalg.norm(ca.max(axis=0) - ca.min(axis=0)))
        rib_span = float(
            np.linalg.norm(rib.positions.max(axis=0) - rib.positions.min(axis=0))
        )
        check("ribbon is not much larger than the trace it follows",
              rib_span < trace_span + 6.0,
              f"ribbon spans {rib_span:.1f} A, CA trace spans {trace_span:.1f} A")
        # Catmull-Rom through n points has n-1 segments, so (n-1)*per_segment
        # samples plus the closing point, four vertices each.
        expected_vertices = 4 * ((len(ca) - 1) * 6 + 1)
        check("ribbon has as many samples as the trace implies",
              len(rib.positions) == expected_vertices,
              f"{len(rib.positions)} vertices, expected {expected_vertices} "
              f"from {len(ca)} residues")
        check("ribbon uses the secondary-structure colours",
              len(np.unique(np.round(rib.colors, 3), axis=0)) >= 2,
              f"{len(np.unique(np.round(rib.colors, 3), axis=0))} colours")

        # Is 4.0 A the strongest statement available? Measured, no: every
        # vertex of the real ribbon is within 1.615 A of a CA, because the
        # widest half-width is 1.6 A for a helix. A bound of 4.0 therefore
        # leaves room for a ribbon nearly 2.5x too wide -- and a ribbon of
        # *no* width at all sits closer to the trace, not further, so the loose
        # bound is wrong in the direction that hides the worst failure. The
        # control for this is the cross-section check below, which reads the
        # width directly and cannot be satisfied by any width at all.
        check("the 4.0 A bound is looser than the geometry needs",
              float(nearest.max()) < 1.7,
              f"the shipped ribbon's worst vertex is {float(nearest.max()):.3f} A "
              f"from a CA, so 1.7 A is still true. The old 4.0 A bound would "
              f"have passed a ribbon twice as wide (3.008 A) and a ribbon with "
              f"no width at all (0.350 A) -- both measured, both wrong")

    section("the ribbon's cross-section is the rectangle that was asked for")
    # A straight guide line along z, with a side vector along x. Width then
    # shows up in x and thickness in y, separately, so the claim "the
    # cross-section is a rectangle of this width and this thickness" becomes
    # checkable. A distance-to-the-trace bound cannot do this: it is satisfied
    # by every width, including none.
    straight = np.array([[0.0, 0.0, float(i)] for i in range(6)])
    side_x = np.tile([1.0, 0.0, 0.0], (6, 1))
    coil = ["coil"] * 6
    ok_mesh = geometry.ribbon(straight, side_x, coil, per_segment=4)
    ok_extent = cross_section_extent(ok_mesh, np.array([0.0, 0.0, 2.5]))
    check("a coil tube is exactly 0.70 A across in width and 0.70 A in "
          "thickness, as the defaults ask for",
          abs(float(ok_extent[0]) - 0.70) < 1e-5
          and abs(float(ok_extent[1]) - 0.70) < 1e-5,
          f"x extent {float(ok_extent[0]):.4f} A (2 x the 0.35 half-width), y "
          f"extent {float(ok_extent[1]):.4f} A (2 x the 0.35 half-thickness). "
          f"The docstring's 'a coil wants a square section' is a claim about "
          f"two independent numbers, and this is the check for it")
    wide = geometry.ribbon(straight, side_x, coil, per_segment=4,
                           width={"helix": 3.2, "sheet": 3.0, "coil": 0.7})
    wide_extent = cross_section_extent(wide, np.array([0.0, 0.0, 2.5]))
    check("CONTROL: a ribbon asked for twice the width really is twice as wide, "
          "so the check above discriminates",
          abs(float(wide_extent[0]) - 1.40) < 1e-5
          and abs(float(wide_extent[1]) - 0.70) < 1e-5,
          f"x extent {float(wide_extent[0]):.4f} A and y unchanged at "
          f"{float(wide_extent[1]):.4f} A -- width and thickness are read "
          f"independently, so a build that swapped them would fail here")
    zero = geometry.ribbon(straight, side_x, coil, per_segment=4,
                           width={"helix": 0.0, "sheet": 0.0, "coil": 0.0})
    check("and a ribbon of no width at all is now distinguishable from a real "
          "one, which a distance bound cannot do",
          float(cross_section_extent(zero, np.array([0.0, 0.0, 2.5]))[0]) < 1e-5
          and float(ok_extent[0]) > 0.5,
          f"zero width gives an x extent of "
          f"{float(cross_section_extent(zero, np.array([0.0, 0.0, 2.5]))[0]):.4f} "
          f"A against the real {float(ok_extent[0]):.4f} A. A zero-width ribbon "
          f"passes the old 'within 4.0 A of a CA' check *more* easily than a "
          f"correct one, which is the direction that bound was wrong in")

    section("a side vector along the tangent used to draw nothing at all")
    # Reproduced first, then fixed; this pins the fixed behaviour so the fix
    # cannot be undone silently. When the first side vector is parallel to the
    # first tangent, projecting it into the tangent's plane gives exactly zero,
    # and the old code projected a (0, 0, 1) fallback onto the *same* tangent
    # -- zero again -- then normalised the zero vector. Every triangle came out
    # with zero area and the ribbon drew nothing: 0 of 160 triangles, 44 of 84
    # vertices with a zero normal, and no extent at all across the trace.
    parallel = geometry.ribbon(straight, np.tile([0.0, 0.0, 1.0], (6, 1)), coil,
                               per_segment=4)
    par_extent = cross_section_extent(parallel, np.array([0.0, 0.0, 2.5]))
    par_areas = tri_areas(parallel)
    check("a side vector parallel to the tangent still produces a tube",
          int((par_areas > 1e-9).sum()) == len(par_areas)
          and float(par_extent[0]) > 0.5 and float(par_extent[1]) > 0.5,
          f"all {len(par_areas)} triangles have area, the cross-section is "
          f"{float(par_extent[0]):.3f} x {float(par_extent[1]):.3f} A. Before "
          f"the fix this was 0 of 160 triangles with any area and a "
          f"{float(cross_section_extent(parallel, np.array([0.0, 0.0, 2.5]))[0]):.1f} A "
          f"x extent -- a ribbon that simply was not there")
    mismatched = geometry.ribbon(straight, np.array([[1.0, 0.0, 0.0]]), coil,
                                 per_segment=4)
    check("a side_vectors array of the wrong length is still handled, and no "
          "longer produces an invisible ribbon",
          int((tri_areas(mismatched) > 1e-9).sum()) == len(mismatched),
          f"{len(mismatched.positions)} vertices after the silent "
          f"substitution, all {len(mismatched.positions) // 4 * 2} triangles "
          f"with area. The substitution of (0, 0, 1) is undocumented; the "
          f"degenerate result it used to produce was the same invisible ribbon")

    # The normals a ribbon carries are what lights it, and a face whose normal
    # points into its own solid is drawn dark. Computed from the vertices again,
    # never from the stored normals.
    check("every ribbon face points away from the spine, so none is lit from "
          "inside",
          faces_point_outward(ok_mesh, np.array([0.0, 0.0, 0.0]),
                              np.array([0.0, 0.0, 1.0])),
          f"all {len(ok_mesh)} triangles of the coil have a geometric normal "
          f"pointing away from the z axis. Reversing the cross product that "
          f"builds the frame's second axis mirrors the ribbon without changing "
          f"its size, so no width or distance check can see it -- this can")

    # Parallel transport, measured. Re-seeding the side vector at every sample
    # looks equivalent on a straight trace -- which is why it survived every
    # check in this file until this one. The trace below turns, and the side
    # vectors handed to it flip sign between neighbours, exactly the situation
    # the docstring describes. Carrying the frame keeps the twist small;
    # re-seeding flips it through a right angle.
    turning = np.array([[0.0, 0.0, float(i)] for i in range(4)]
                       + [[float(i - 3), 0.0, 3.0] for i in range(4, 8)])
    swinging = np.concatenate([np.tile([1.0, 0.0, 0.0], (4, 1)),
                               np.tile([-1.0, 0.0, 0.0], (4, 1))])
    carried = geometry.ribbon(turning, swinging, ["coil"] * 8, per_segment=4)
    sides, across = cross_section_frames(carried)
    dots = [float(sides[i] @ sides[i + 1]) for i in range(len(sides) - 1)]
    across_dots = [float(across[i] @ across[i + 1]) for i in range(len(across) - 1)]
    reseeded = reseeded_side_dots(turning, swinging, 4)
    check("the frame is carried along the curve, so a trace that turns does not "
          "twist the ribbon inside out",
          min(dots) > 0.5 and min(across_dots) > 0.5 and reseeded < 0.0,
          f"through a 90 degree turn with the side vectors reversing sign, the "
          f"carried frame's smallest consecutive dot is {min(dots):.3f} for "
          f"`side` and {min(across_dots):.3f} for `across`. CONTROL: re-seeding "
          f"the side at every sample -- one line -- gives {reseeded:.3f}, a "
          f"180 degree flip, which is the beaded, inside-out ribbon the "
          f"docstring describes. No width, distance or triangle-count check "
          f"can see a twist")

    section("a ligand has no ribbon, and says so rather than drawing a line")
    check("ligand reports no backbone", not lig.has_backbone)
    check("ligand ribbon is None", lig.backbone_ribbon() is None)

    section("sphere mesh")
    s = geometry.spheres(np.array([[0.0, 0, 0], [3.0, 0, 0]]),
                         np.array([1.0, 0.5]), np.array([[1, 0, 0], [0, 1, 0]], float))
    check("two atoms, both drawn", len(s) == 2 * 12 * 8 * 2, f"{len(s)} triangles")
    check("radii differ per atom",
          float(np.abs(s.positions - 0.0).max()) > 2.0,
          "one sphere is radius 1 and one is 0.5")

    # The check above would pass if both spheres were drawn at radius 1, so it
    # is not really a claim about radii. Measured per sphere, each is exactly
    # its own radius, which is what the check should have said.
    per_sphere = 12 * 9
    d0 = np.linalg.norm(s.positions[:per_sphere] - np.array([0.0, 0.0, 0.0]), axis=1)
    d1 = np.linalg.norm(s.positions[per_sphere:] - np.array([3.0, 0.0, 0.0]), axis=1)
    check("and each sphere is exactly its own radius, which the check above "
          "cannot tell",
          abs(float(d0.max()) - 1.0) < 1e-6 and abs(float(d0.min()) - 1.0) < 1e-6
          and abs(float(d1.max()) - 0.5) < 1e-6
          and abs(float(d1.min()) - 0.5) < 1e-6,
          f"sphere 0 spans {float(d0.min()):.4f}-{float(d0.max()):.4f} A from "
          f"its centre and sphere 1 {float(d1.min()):.4f}-{float(d1.max()):.4f} "
          f"A. 'radii differ' is satisfied by two spheres of radius 1, so it "
          f"was a claim about the pair, not about the radii")
    check("a scalar radius applies to every sphere",
          np.allclose(
              np.linalg.norm(geometry.spheres(
                  np.array([[0.0, 0, 0], [3.0, 0, 0]]), 0.5,
                  np.array([[1, 0, 0], [0, 1, 0]], float)).positions[:per_sphere],
                  axis=1), 0.5, atol=1e-6),
          "radii may be one number for all atoms, which is how the workbench "
          "calls it for the space-filling view")
    check("and each sphere wears its own atom's colour, which nothing checked",
          bool((s.colors[:per_sphere, 1] == 0).all()
               and (s.colors[:per_sphere, 0] > 0.5).all()
               and (s.colors[per_sphere:, 0] == 0).all()
               and (s.colors[per_sphere:, 1] > 0.5).all()),
          f"the first {per_sphere} vertices are red and the rest green. Two "
          f"atoms with two colours is the easy half; each colour landing on its "
          f"own atom is the half a build that shared one colour row across all "
          f"atoms would get wrong, and nothing in the file noticed that until "
          f"this check existed")
    check("the sphere's pole vertices are coincident, and its indices stay "
          "in range",
          int((tri_areas(s) <= 1e-12).sum()) == 4 * 12
          and int(s.indices.max()) < len(s.positions),
          f"{int((tri_areas(s) <= 1e-12).sum())} of {len(tri_areas(s))} "
          f"triangles are zero-area, which is the UV sphere's two poles: all 12 "
          f"vertices of a pole ring sit on one point. Harmless for drawing, but "
          f"it is why a triangle count is not a measure of how much is on "
          f"screen, and every index is within the vertex array")

    section("dashed segments")
    dsegs, dgrp = geometry.dashed_segments(
        np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 3.0]]))
    lengths = np.linalg.norm(dsegs[:, 1] - dsegs[:, 0], axis=1)
    gaps = [float(np.linalg.norm(dsegs[i + 1, 0] - dsegs[i, 1]))
            for i in range(len(dsegs) - 1)]
    check("a 3.0 A segment becomes several dashes of 0.40 A separated by "
          "0.26 A gaps",
          len(dsegs) == 5 and all(abs(float(x) - 0.4) < 1e-5 for x in lengths[:-1])
          and abs(float(lengths[-1]) - 0.36) < 1e-5
          and all(abs(g - 0.26) < 1e-5 for g in gaps),
          f"{len(dsegs)} dashes, lengths "
          f"{[round(float(x), 3) for x in lengths]}, gaps "
          f"{[round(g, 3) for g in gaps]}. The last one is short because the "
          f"segment ends mid-dash; the dash *period* is the claim, not the "
          f"count")
    check("each dash records which segment it came from",
          dgrp.shape == (len(dsegs),) and set(dgrp.tolist()) == {0}
          and dgrp.dtype == np.int32,
          f"group = {dgrp.tolist()}, which is what lets a caller colour each "
          f"dash from its own contact without re-deriving the line's length")
    mixed, mgrp = geometry.dashed_segments(
        np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[0.0, 0.0, 1.6], [0.0, 5.0, 0.2]]))
    check("the grouping survives several segments of different lengths",
          mgrp.tolist() == [0, 0, 0, 1] and mixed.shape[1:] == (2, 3),
          f"a 1.6 A segment gives {int((mgrp == 0).sum())} dashes and a 0.2 A "
          f"one gives {int((mgrp == 1).sum())}, tagged {mgrp.tolist()}. Both "
          f"take the ordinary path: a fixture whose second segment was "
          f"*degenerate* took the short-circuit branch, and a build that "
          f"dropped the index only on the short-circuit path would then have "
          f"been caught for the wrong reason -- or, worse, passed")
    short, sgrp = geometry.dashed_segments(
        np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.2]]))
    check("a segment shorter than one dash stays solid rather than vanishing",
          len(short) == 1
          and abs(float(np.linalg.norm(short[0, 1] - short[0, 0])) - 0.2) < 1e-6,
          f"a 0.2 A contact draws as one solid dash of 0.2 A. The docstring's "
          f"reason matters: a short contact is a real contact, and clipping it "
          f"would make the densest part of the interface the sparsest part of "
          f"the picture")
    check("empty and zero-length inputs are handled without raising",
          geometry.dashed_segments(np.zeros((0, 3)), np.zeros((0, 3)))[0].shape
          == (0, 2, 3)
          and geometry.dashed_segments(
              np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0, 3.0]]))[0].shape
          == (1, 2, 3),
          "no segments gives an empty (0, 2, 3) array; a degenerate one gives a "
          "single degenerate dash rather than a division by zero")

    section("the curve passes through the residues it was given")
    # catmull_rom's docstring makes a specific promise -- "the curve starts and
    # ends exactly on the first and last residue rather than falling short of
    # them" -- and nothing in this file tested it. A ribbon that stops half a
    # residue early looks like a rendering fault, and it would be one.
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
    curve = geometry.catmull_rom(pts, 8)
    check("the curve starts and ends exactly on the first and last residue",
          np.allclose(curve[0], pts[0]) and np.allclose(curve[-1], pts[-1]),
          f"first {curve[0].tolist()} against {pts[0].tolist()}, last "
          f"{curve[-1].tolist()} against {pts[-1].tolist()}. The endpoints are "
          f"duplicated before the spline and the last point appended after it, "
          f"which is what makes this exact rather than approximate")
    check("a curve through fewer than two points is returned as it came in",
          np.array_equal(geometry.catmull_rom(pts[:1], 8), pts[:1]),
          f"one point in, {geometry.catmull_rom(pts[:1], 8).shape} out -- there "
          f"is no segment to interpolate")

    section("MeshData bookkeeping")
    one = geometry.bonds(fx, [(0, 1)], 0.2, fcols, segments=8, two_tone=True)
    nothing = geometry.spheres(np.zeros((0, 3)), 1.0, np.zeros((0, 3)))
    check("len() is the triangle count, and an empty mesh has none",
          len(one) == one.indices.size // 3 and len(nothing) == 0
          and nothing.empty and not one.empty,
          f"{len(one)} triangles from {one.indices.size} indices; the empty "
          f"mesh reports empty={nothing.empty} and len={len(nothing)}")

    # ------------------------------------------------------------------
    section("every check in this file ran")

    before = CHECKS
    check("the number of checks that ran is the number this file is supposed "
          "to have",
          before + 1 == EXPECTED_CHECKS,
          f"{before} ran before this one and {EXPECTED_CHECKS} are expected; "
          f"the +1 is this check. If a check was added or removed, change "
          f"EXPECTED_CHECKS deliberately")

    print("\n=== summary ===")
    print(f"  {CHECKS - len(FAILURES)} passed, {len(FAILURES)} failed, {CHECKS} checks")
    for f in FAILURES:
        print(f"    FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
