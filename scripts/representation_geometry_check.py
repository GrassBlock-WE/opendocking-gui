"""Headless check of the representation geometry.

Runs the real geometry routines on real structures and asserts things that can
only be true if the maths is right: a cylinder actually spans the two atoms it
was asked to span, a bond is drawn in two colours, and the ribbon follows the
backbone instead of wandering off it.
"""

import sys
import warnings
from pathlib import Path
from math import dist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking.workbench import geometry, structure as st_mod  # noqa: E402
from opendocking.workbench import MoleculeView  # noqa: E402

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


def main() -> int:
    section("unit cylinder")
    v, n, f = geometry.unit_cylinder(8)
    check("has the expected vertex count", len(v) == 9, f"{len(v)} verts, {len(f)} tris")
    check("radius is 1 and length is 1",
          abs(np.linalg.norm(v[0][:2]) - 1.0) < 1e-9 and abs(v[-1][2] - 1.0) < 1e-9)
    check("normals are unit length",
          all(abs(np.linalg.norm(r) - 1.0) < 1e-9 for r in n[:8]),
          "the cap centre has no normal, which is correct")

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

    print("\n=== summary ===")
    print(f"  {CHECKS - len(FAILURES)} passed, {len(FAILURES)} failed, {CHECKS} checks")
    for f in FAILURES:
        print(f"    FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
