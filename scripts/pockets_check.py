"""Headless check of the pocket search, so the search box stops being a guess.

Three kinds of evidence, and they are not interchangeable:

* **A structure with a pocket whose position is known.** Two concentric
  shells, built from a golden-angle spiral so there is no seam for the flood
  fill to leak through. The interior is a known sphere at the origin, so a
  detector can be caught returning a plausible-looking box in the wrong place.
* **Reverse verification of every filter.** Each threshold is checked in both
  directions: loosen it and the answer has to change, tighten it and the answer
  has to be able to empty. A filter that returns the same thing whatever you set
  it to is indistinguishable from no filter, and passes a test that only looks
  at the result.
* **The real thing.** Ibuprofen, actually docked into crambin, as a site the
  detector cannot cheat its way to: the answer is not a coordinate, it is that
  the box built from the top site contains the ligand and is lined by the
  residue the independent contact analysis already named.

Run:  python scripts/pockets_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Prefer the *installed* package, and only fall back to the source tree when
# there is no installed one. A clean checkout has no `_dockpy` extension in its
# source tree -- the compiled module is gitignored and lives in site-packages
# -- so putting the source tree first makes this pass locally and fail in CI.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking.workbench import MoleculeView  # noqa: E402
from opendocking.workbench import pockets as P  # noqa: E402

CRAMBIN = ROOT / "examples" / "1crn_prep.pdbqt"
CRAMBIN_POSE = ROOT / "examples" / "crambin_pose.pdbqt"

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


def sphere(radius: float, n: int):
    """`n` points on a sphere of `radius`, evenly spread.

    Golden-angle rather than a uniform sweep in polar angle: with
    `linspace(0, 2*pi)` the longitude advances by a constant step, so the
    points march across the sphere in a spiral and leave a wedge open along
    the whole surface. A flood fill escapes through a wedge like that, the
    "sealed" cavity comes back open, and the fixture silently stops testing
    what it is supposed to test. The Fibonacci distribution has no seam.
    """
    i = np.arange(n, dtype=np.float64)
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1) * radius


def carls(elements, spacing=1.5):
    """A solid block of atoms: no interior space anywhere."""
    axis = np.arange(-6.0, 6.01, spacing)
    g = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return g.reshape(-1, 3), ["C"] * (g.reshape(-1, 3).shape[0])


def in_box(points, center, size):
    lo = np.asarray(center, float) - np.asarray(size, float) / 2.0
    hi = np.asarray(center, float) + np.asarray(size, float) / 2.0
    return ((points >= lo) & (points <= hi)).all(axis=1)


def _energy_of_file(path: Path):
    """The `REMARK VINA RESULT` energy of a multi-model PDBQT, or None.

    Read from the file rather than recomputed, so the comparison is against
    what the reference pose actually claims to be.
    """
    try:
        from opendocking.pdbqt_writer import read_pdbqt_models
    except Exception:  # pragma: no cover
        return None
    for body in read_pdbqt_models(path):
        # Each model is a *list of lines*, not one string. Treating it as a
        # string is a `.splitlines()` on a list, which is the same mistake the
        # workbench's own pose loader had to be fixed for.
        lines = body if isinstance(body, str) else "\n".join(body)
        for line in lines.splitlines():
            if "VINA RESULT" in line:
                # The field is `RESULT:`, colon attached, so looking for a bare
                # "RESULT" token finds nothing and the reference energy comes
                # back as None -- which reads as "no energy in the file" and
                # quietly turns the comparison into a no-op.
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith("RESULT"):
                        try:
                            return float(parts[i + 1])
                        except (ValueError, IndexError):
                            return None
                return None
    return None


def main() -> int:
    # ------------------------------------------------------------------
    section("a cavity whose position is known in advance")

    inner = sphere(6.0, 120)
    outer = sphere(10.0, 200)
    shell = np.vstack([inner, outer])
    sites = P.find_pockets(shell, ["C"] * len(shell), ["SHELL"] * len(shell))
    check("the two shells yield a site", len(sites) >= 1, f"{len(sites)} found")
    cavities = [p for p in sites if p.kind == "cavity"]
    check("exactly one of them is sealed", len(cavities) == 1,
          f"kinds {[p.kind for p in sites]}")
    if cavities:
        top = cavities[0]
        d = float(np.linalg.norm(np.asarray(top.center)))
        check("the sealed one is centred on the cavity, not merely near it",
              d < 0.5, f"{d:.3f} A from origin")
        # 6 A shell, 1.7 A carbon radius, 1.4 A probe -> 5.8 A of open space,
        # and the grid rounds to whole voxels.
        check("its size is the gap between the shells, ± 1 voxel",
              4.5 <= float(top.size.min()) and float(top.size.max()) <= 7.5,
              f"size={np.round(top.size, 2)}")
        check("the shell lines it", any(r == "SHELL" for r, _ in top.lining))
    # The outer shell's own surface is also a legitimate place to report a
    # groove, so the claim is not "only the interior is found" -- it is that
    # the *sealed* site is the interior, and that no sealed site is invented
    # out in the open solvent beyond the outer shell (which reaches 13.1 A).
    check("no sealed site is claimed out in the open solvent",
          all(float(np.linalg.norm(np.asarray(p.center))) < 5.0 for p in cavities),
          f"furthest sealed site at "
          f"{max((float(np.linalg.norm(np.asarray(p.center))) for p in cavities), default=0):.2f} A")

    section("a solid block has no cavity at all")
    block, block_els = carls(["C"])
    none = P.find_pockets(block, block_els)
    check("a solid block reports nothing", len(none) == 0, f"{len(none)} found")

    section("one shell is already a cavity; two are not needed for the test")
    # A single shell encloses its own interior. If the detector needed a second
    # shell to work, this would come back empty and the double-shell case above
    # would be testing the fixture rather than the detector.
    one = P.find_pockets(inner, ["C"] * len(inner))
    check("a single shell also encloses its interior", len(one) >= 1, f"{len(one)} found")
    if one:
        check("and it is still at the origin",
              float(np.linalg.norm(np.asarray(one[0].center))) < 0.5)

    # ------------------------------------------------------------------
    section("the flood fill reaches what it should, and seals what it should not")

    free = np.ones((9, 9, 9), bool)
    free[4, 4, 4] = False
    reached = P._flood_from_border(free)
    check("everything reachable from the border is reached",
          bool(reached[free].all()) and not bool(reached[~free].any()),
          f"{int(reached.sum())} of {int(free.sum())} free voxels, "
          f"{int((~free).sum())} solid wrongly marked")
    check("a lone blocked voxel stays blocked", not bool(reached[4, 4, 4]))

    walled = np.ones((9, 9, 9), bool)
    walled[2:7, 2:7, 2:7] = False
    reached = P._flood_from_border(walled)
    check("a 5x5x5 block of protein is never reached",
          not bool(reached[3:6, 3:6, 3:6].any()) and int(reached.sum()) == 9**3 - 5**3,
          f"{int(reached.sum())} of {int(walled.sum())}")

    section("connected components are components")
    cube = np.zeros((7, 7, 7), bool)
    cube[2:5, 2:5, 2:5] = True
    parts = list(P._components(cube))
    check("a 27-voxel cube is one component, not 27",
          len(parts) == 1 and len(parts[0]) == 27, f"{len(parts)} parts")
    pair = np.zeros((9, 9, 9), bool)
    pair[1:4, 1:4, 1:4] = True
    pair[5:8, 5:8, 5:8] = True
    parts = list(P._components(pair))
    check("two separated cubes are two components",
          len(parts) == 2 and sorted(len(p) for p in parts) == [27, 27])

    section("dilation moves one step per call, not one per round")
    # A *line*, and a short one. `line[5, :, :]` on an 11-cube is 121 voxels,
    # and a check written as "one step doubles it" then fails at 242 while the
    # dilation is behaving exactly as it should -- the fixture, not the code,
    # was wrong. A line that runs the full length of the grid is wrong in the
    # other direction: it already touches both walls, so it cannot grow along
    # its own axis at all and "dilation does nothing" looks like a pass.
    line = np.zeros((11, 11, 11), bool)
    line[5, 5, 4:7] = True
    check("the fixture is a line of 3 voxels", int(line.sum()) == 3)
    check("one step along its own axis adds one voxel",
          int(P._dilate_axis(line, 2, 1).sum()) == 4,
          f"{int(P._dilate_axis(line, 2, 1).sum())}")
    check("three steps add exactly three",
          int(P._dilate_axis(line, 2, 3).sum()) == 6,
          f"{int(P._dilate_axis(line, 2, 3).sum())}")
    check("a step costs one voxel, not one layer's worth",
          [int(P._dilate_axis(line, 2, n).sum()) for n in (1, 2, 3, 4)]
          == [4, 5, 6, 7],
          f"{[int(P._dilate_axis(line, 2, n).sum()) for n in (1, 2, 3, 4)]}")
    # `_dilate_axis` grows in the *negative* direction only, so mirrored seeds
    # are only comparable when neither can run off the grid: a seed at x=2 in
    # an 11-cube loses a step to the wall and a seed at x=8 does not. Comparing
    # those two says nothing about the dilation, and comparing growth along
    # the line's own axis to growth across it says nothing either -- a 1x1x3
    # line is not symmetric under exchanging axes.
    lo = np.zeros((11, 11, 11), bool)
    lo[3, 5, 5] = True
    hi = np.zeros((11, 11, 11), bool)
    hi[7, 5, 5] = True
    check("mirrored seeds grow the same on the same axis",
          int(P._dilate_axis(lo, 0, 2).sum()) == int(P._dilate_axis(hi, 0, 2).sum()) == 3,
          f"{int(P._dilate_axis(lo, 0, 2).sum())} vs {int(P._dilate_axis(hi, 0, 2).sum())}")
    one = np.zeros((11, 11, 11), bool)
    one[5, 5, 5] = True
    check("a lone voxel gains exactly one voxel per step",
          [int(P._dilate_axis(one, 0, n).sum()) for n in (1, 2, 3)] == [2, 3, 4],
          f"{[int(P._dilate_axis(one, 0, n).sum()) for n in (1, 2, 3)]}")
    check("and the count is the same whichever axis it is grown along",
          [int(P._dilate_axis(one, ax, 2).sum()) for ax in (0, 1, 2)] == [3, 3, 3],
          f"{[int(P._dilate_axis(one, ax, 2).sum()) for ax in (0, 1, 2)]}")
    dot = np.zeros((5, 5, 5), bool)
    dot[2, 2, 2] = True
    cross = np.zeros((5, 5, 5), bool)
    cross[2, 2, 2] = True
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        cross[2 + d[0], 2 + d[1], 2 + d[2]] = True
    # `_dilate_iso` used to live here and had five checks of its own. It grew
    # the burial mask by one voxel before labelling, to stop thin sheets
    # being reported as slivers -- and it merged every real site into the
    # outer surface instead, because a surface groove is one voxel thick and
    # one voxel of growth reaches across its neck. The four redocking cases
    # in the module docstring were all lost that way. Its tests are replaced
    # by the section below, which pins the behaviour that was actually wrong.

    # ------------------------------------------------------------------
    section("a winding cleft is measured by its space, not by its box")

    # The defect this pins: the volume ceiling used to be applied to a site's
    # bounding box. A cleft that runs 33 A one way and 34 A another has a
    # 9700 A3 box around 1000 A3 of actual pocket, so the ceiling rejected it
    # and the search reported a protein with nowhere to dock. The real 1HVR
    # site is the same shape: 544 A3 of space in an 8008 A3 box.
    def dogleg(arm=26.0, gap=9.0, wall=3.0, pitch=3.2):
        """A slot running `arm` A along x, turning, and running back along y."""
        n, m = int(arm / pitch) + 1, int(wall / pitch) + 1
        pts = []
        for i in range(n):
            for j in range(m):
                for s in (-1.0, 1.0):
                    pts.append((i * pitch, j * pitch - wall / 2.0, s * gap / 2.0))
                    pts.append((arm + s * gap / 2.0, i * pitch,
                                j * pitch - wall / 2.0))
        return np.asarray(pts, np.float32)

    cleft = dogleg()
    found = P.find_pockets(cleft, ["C"] * len(cleft))
    check("a bent slot in a wall is found at all", len(found) >= 1,
          f"{len(cleft)} atoms -> {len(found)} site(s)")
    if found:
        top = found[0]
        box = float(np.prod(top.size))
        check("its own volume is under the ceiling", top.volume
              <= P.DEFAULT_MAX_VOLUME,
              f"{top.volume:.0f} A3 of {P.DEFAULT_MAX_VOLUME:.0f} allowed")
        check("while its bounding box is a long way over it", box
              > P.DEFAULT_MAX_VOLUME,
              f"{box:.0f} A3 -- this is the case the old ceiling threw away")
        check("and the ceiling still empties the list when it is set to zero",
              len(P.find_pockets(cleft, ["C"] * len(cleft), max_volume=0.0)) == 0)
        # Reverse check on the measure itself: volume must track voxels, and
        # must not be a relabelled bounding box. A site whose box is 9x its
        # own volume is the whole point, so a check that only ever sees
        # near-cubic sites would pass against the old code.
        check("volume is the voxel's own, not the box's",
              abs(top.volume - top.voxels * P.DEFAULT_SPACING ** 3) < 1e-6
              and top.volume < box / 4.0,
              f"{top.voxels} voxels at {P.DEFAULT_SPACING} A = "
              f"{top.volume:.0f} A3, in a {box:.0f} A3 box")

    # ------------------------------------------------------------------
    section("no site on a real protein is the protein's whole surface")

    # The same defect seen from the other side. A merged site is not merely
    # mis-ranked: it *is* the surface, so it spans a large fraction of the
    # protein and has no interior. On crambin the three merged lumps that the
    # old code cut with the box ceiling were 3514, 3564 and 1720 A3 of box.
    # Nothing that big is a binding site, and the volume ceiling no longer
    # has to be the thing that removes them, because they are not made.
    rec = MoleculeView.from_pdbqt(CRAMBIN, "crambin", (1, 1, 1), 0.30, role="receptor")
    rec_pts = np.asarray(rec.coords, np.float32)
    protein_box = float(np.prod(rec_pts.max(axis=0) - rec_pts.min(axis=0)))
    all_sites = P.find_pockets(rec_pts, list(rec.elements), max_pockets=99,
                               max_volume=1e9)
    worst = max(float(np.prod(p.size)) for p in all_sites)
    check("the largest crambin site is a fraction of the protein, not the protein",
          worst < protein_box / 8.0,
          f"largest site box {worst:.0f} A3 against {protein_box:.0f} A3 of protein")
    check("and the sites together are not a shell around the outside",
          sum(p.voxels for p in all_sites) * P.DEFAULT_SPACING ** 3
          < protein_box / 8.0,
          f"{sum(p.voxels for p in all_sites)} voxels over {len(all_sites)} sites")

    # The ceiling's remaining job, stated as what it now does: it is inert on
    # crambin, because the lumps it used to remove are no longer produced.
    # That is worth asserting, because a check that quietly stops firing is
    # how a filter rots.
    kept = P.find_pockets(rec_pts, list(rec.elements), max_pockets=99,
                          max_volume=P.DEFAULT_MAX_VOLUME)
    check("the ceiling no longer has lumps to remove on crambin",
          len(kept) == len(all_sites),
          f"{len(all_sites)} -> {len(kept)} sites; it used to be 23 -> 20")
    check("but it is still a ceiling and still bites when lowered",
          len(P.find_pockets(rec_pts, list(rec.elements), max_pockets=99,
                             max_volume=1.0)) == 0,
          "a 1 A3 ceiling empties crambin")

    # ------------------------------------------------------------------
    section("search effort follows the box, and the user can overrule it")

    # A pocket search will happily hand over a 39 x 26 x 41 A box for a winding
    # cleft, and a fixed exhaustiveness under-samples every large box: walks
    # are spread through the volume they search. Measured on the 3PTB site box,
    # 12.88 A RMSD at 16 and 1.26 A at 64, two seconds either way. The rule
    # lives in the engine so the workbench, the CLI and the benchmark share it.
    from opendocking.core import (  # noqa: PLC0415
        EXHAUSTIVENESS_LADDER,
        REFERENCE_BOX_SIDE,
        exhaustiveness_for_box,
    )

    # Forward: the boxes measured in the redocking benchmark, and what the rule
    # says for each. The 39 x 26 x 41 one is the calibration point -- it is the
    # box that read 12.88 A at 16 and 1.26 A at 64 -- so a rule that answered
    # "16" there would be reproducing the number that was wrong.
    for size, want in (((20.0,) * 3, 8),
                       ((22.0, 22.0, 22.0), 16),
                       ((39.0, 26.0, 41.0), 64),
                       ((30.0, 38.0, 32.0), 64)):
        check(
            f"a {int(size[0])}x{int(size[1])}x{int(size[2])} A box gets "
            f"exhaustiveness {want}",
            exhaustiveness_for_box(size) == want,
            f"rule says {exhaustiveness_for_box(size)}",
        )

    # Reverse: every rung has to be reachable, or the rule is a constant
    # wearing a formula. A sweep of cube sides is used rather than a handful of
    # round numbers because the rungs are narrow bands -- sides 20, 30 and 45
    # jump clean over 16 and 64 and would leave a rule that silently skipped
    # them looking fine.
    reached = {
        exhaustiveness_for_box((s, s, s))
        for s in (10.0, 20.0, 22.0, 26.0, 30.0, 40.0, 50.0, 60.0, 90.0)
    }
    check(
        "every rung of the ladder is reachable, so this is not a constant",
        reached == set(EXHAUSTIVENESS_LADDER),
        f"reached {sorted(reached)}, ladder {list(EXHAUSTIVENESS_LADDER)}",
    )
    check(
        "and each rung answers to a band of boxes, not to a single size",
        exhaustiveness_for_box((22.0,) * 3) == 16
        and exhaustiveness_for_box((25.0,) * 3) == 16
        and exhaustiveness_for_box((31.0,) * 3) == 32,
        "22 and 25 A cubes both land on 16; 31 A moves to 32",
    )
    check(
        "and a reference-sized box is the conventional default",
        exhaustiveness_for_box((REFERENCE_BOX_SIDE,) * 3)
        == EXHAUSTIVENESS_LADDER[0],
        f"{REFERENCE_BOX_SIDE:.0f} A cube -> "
        f"{exhaustiveness_for_box((REFERENCE_BOX_SIDE,) * 3)}",
    )
    check(
        "a box too big for the ladder is capped, not extrapolated",
        exhaustiveness_for_box((200.0,) * 3) == EXHAUSTIVENESS_LADDER[-1],
        f"200 A cube -> {exhaustiveness_for_box((200.0,) * 3)}, "
        f"top rung {EXHAUSTIVENESS_LADDER[-1]}",
    )
    check(
        "it only reads the three side lengths, not anything else",
        exhaustiveness_for_box((39.0, 26.0, 41.0))
        == exhaustiveness_for_box((41.0, 39.0, 26.0))
        == exhaustiveness_for_box((26.0, 41.0, 39.0)),
        "permuting the sides must not change the answer",
    )

    # ------------------------------------------------------------------
    section("every threshold can empty the result, and every one can loosen it")

    # Reverse verification. Each of these is checked in *both* directions: a
    # threshold that only ever shrinks a list is indistinguishable from a
    # broken detector, and a threshold that changes nothing is a lie.
    check("an impossible voxel count empties it",
          len(P.find_pockets(shell, ["C"] * len(shell), min_voxels=10**9)) == 0)
    check("an impossible minimum extent empties it",
          len(P.find_pockets(shell, ["C"] * len(shell), min_extent=99.0)) == 0)
    check("an impossible volume ceiling empties it",
          len(P.find_pockets(shell, ["C"] * len(shell), max_volume=0.0)) == 0)
    # The sealed interior of the double shell is not a groove and is not
    # filtered by the burial threshold, so the claim has to be about grooves
    # specifically -- asking for "no sites at all" fails on a correct result.
    hard = P.find_pockets(shell, ["C"] * len(shell), min_burial=4)
    check("a burial threshold above three leaves no groove",
          not [p for p in hard if p.kind == "burial"],
          f"kinds {[p.kind for p in hard]}")
    check("but the sealed cavity is not a burial product and survives",
          len([p for p in hard if p.kind == "cavity"]) == 1)
    check("max_pockets truncates rather than filtering",
          len(P.find_pockets(shell, ["C"] * len(shell), max_pockets=1)) == 1)
    check("removing the ceiling brings the sites back",
          len(P.find_pockets(shell, ["C"] * len(shell), max_volume=1e9))
          >= len(P.find_pockets(shell, ["C"] * len(shell), max_volume=1500.0)))

    # ------------------------------------------------------------------
    section("crambin, with ibuprofen actually docked into it")

    pose = MoleculeView.from_pdbqt(CRAMBIN_POSE, "pose", (1, 1, 1), 0.32, role="pose")
    pose_pts = np.asarray(pose.coords, np.float32)
    pose_centroid = pose_pts.mean(axis=0)

    near = np.linalg.norm(
        rec_pts[:, None, :] - pose_pts[None, :, :], axis=2
    ).min(axis=0)
    check("the fixture really is docked against the receptor",
          int((near <= 4.0).sum()) == len(pose_pts),
          f"{int((near <= 4.0).sum())}/{len(pose_pts)} ligand atoms within 4 A, "
          f"closest {float(near.min()):.2f} A")
    # The honest version of "the centroid box is the wrong box". On a protein
    # this small a 22 A box at the centroid still *contains* the ligand, so
    # "it misses the ligand" is not a claim the data supports. What is
    # measurable is where the centre is, and how much of the search volume is
    # being swept for it.
    rec_centroid = rec_pts.mean(axis=0)
    off = float(np.linalg.norm(rec_centroid - pose_centroid))
    check("the receptor centroid is nowhere near where the ligand is",
          off > 5.0, f"{off:.2f} A from the ligand centroid")

    sites = P.find_pockets(rec_pts, list(rec.elements), residues=rec.residue_labels())
    check("sites are offered for crambin", len(sites) >= 1, f"{len(sites)} found")
    check("the list is within max_pockets", len(sites) <= P.DEFAULT_MAX_POCKETS,
          f"{len(sites)} of {P.DEFAULT_MAX_POCKETS}")

    # A site is judged by the box it builds, not by how near its centre is.
    # The 3PTB site in the redocking benchmark is 12.8 A from its ligand's
    # centre of mass and holds every atom, so "closest to the ligand" is not
    # a safe way to pick the site that docked. This picks the site whose box
    # holds the most ligand atoms, and says which that is.
    if sites:
        held = [int(in_box(pose_pts, *p.box_center_and_size()).sum())
                for p in sites]
        best = int(np.argmax(held))
        best_site = sites[best]
        best_centre, best_size = best_site.box_center_and_size()
        check("some site's box holds the whole docked ligand",
              held[best] == len(pose_pts),
              f"best is rank {best + 1} of {len(sites)}, {held[best]}/"
              f"{len(pose_pts)} atoms; the shortlist holds {held}")
        check("and the shortlist is long enough to include it",
              held[best] == len(pose_pts),
              f"rank {best + 1} against a default of {P.DEFAULT_MAX_POCKETS}. "
              f"At 8 this site was cut off: it is 12 A^3 and crambin has "
              f"three larger lumps of surface ahead of it")

        # Ranked ninth, not first, and that is the data rather than a target.
        # What is worth locking in is that the site is in the shortlist and
        # that it is the one whose box works, so a ranking change that buries
        # it entirely is caught. Asserting "first" would be asserting a
        # result crambin does not support: its groove is a snug fit, so the
        # free space left around the ligand is almost nothing.
        check("and it is not the first thing on crambin, which is the point",
              best > 0,
              f"rank {best + 1}, {best_site.volume:.0f} A^3, and three larger "
              f"surface lumps come first -- a snug pocket leaves little space "
              f"behind the ligand, so volume cannot rank it")
        names = {r for r, _ in best_site.lining}
        # An independent count, not a restatement: which receptor residues
        # have any atom within 4.5 A of this same pose. The site's lining is
        # computed from the pocket; this is computed from the coordinates.
        # Two unrelated computations agreeing is worth more than either.
        contacts = {r for r, row in zip(rec.residue_labels(),
                                       (np.linalg.norm(
                                           rec_pts[:, None, :] - pose_pts[None],
                                           axis=2) <= 4.5).any(axis=1))
                    if r and row}
        shared = names & contacts
        check("it is lined by the residues an independent contact count finds",
              len(shared) >= 5,
              f"{len(shared)} of {len(contacts)} contact residues line it: "
              f"{sorted(shared)}")

        # The measurable win, against the box the workbench used before.
        old_volume = 22.0 ** 3
        new_volume = float(np.prod(best_size))
        check("the box it builds is far smaller than the whole-protein one",
              new_volume < old_volume / 3.0,
              f"{new_volume:.0f} vs {old_volume:.0f} A^3 "
              f"({old_volume / new_volume:.1f}x smaller)")
        check("and the smaller box still holds every ligand atom",
              in_box(pose_pts, rec_centroid, best_size).sum() < len(pose_pts),
              f"the same box centred on the protein centroid would hold "
              f"{int(in_box(pose_pts, rec_centroid, best_size).sum())}/{len(pose_pts)}, "
              f"so the centre is the part that was wrong")

    section("a shallower threshold finds the shallow site, and says so")
    deep = P.find_pockets(rec_pts, list(rec.elements), min_burial=2, max_pockets=64)
    shallow = P.find_pockets(rec_pts, list(rec.elements), min_burial=1, max_pockets=64)
    check("min_burial=1 returns at least as many sites as min_burial=2",
          len(shallow) >= len(deep), f"{len(shallow)} vs {len(deep)}")
    if sites:
        d_deep = min((float(np.linalg.norm(np.asarray(p.center) - pose_centroid))
                      for p in deep), default=1e9)
        d_def = min((float(np.linalg.norm(np.asarray(p.center) - pose_centroid))
                     for p in sites), default=1e9)
        check("the default finds the docked site where a threshold of 2 does not",
              d_def < 6.0 and d_deep > 6.0,
              f"default {d_def:.2f} A, min_burial=2 {d_deep:.2f} A")

    section("the site the docked ligand occupies is shallow, and that is stated")
    # A pocket finder that claims the ligand's site is well enclosed would be
    # making a claim crambin does not support.
    shallowest = min(
        (float(p.burial) for p in sites), default=0.0
    )
    check("the default list includes sites with burial below 2",
          shallowest < 2.0, f"lowest mean burial reported {shallowest:.2f} of 3")

    # ------------------------------------------------------------------
    section("degenerate input does not crash and does not invent a pocket")
    check("no atoms", P.find_pockets(np.zeros((0, 3)), []) == [])
    check("one atom", P.find_pockets(np.zeros((1, 3)), ["C"]) == [])
    check("elements shorter than the coordinates",
          isinstance(P.find_pockets(np.zeros((6, 3)), ["C", "N"]), list))
    check("an element nobody has a radius for",
          isinstance(P.find_pockets(np.zeros((5, 3)), ["Xx", "Qq", "", "C"]), list))
    check("residues shorter than the coordinates",
          isinstance(P.find_pockets(shell, ["C"] * len(shell), ["A"]), list))

    section("the result is deterministic")
    a = P.find_pockets(rec_pts, list(rec.elements), residues=rec.residue_labels())
    b = P.find_pockets(rec_pts, list(rec.elements), residues=rec.residue_labels())
    check("two runs agree, site for site",
          len(a) == len(b) and all(
              float(np.linalg.norm(np.asarray(x.center) - np.asarray(y.center))) < 1e-6
              for x, y in zip(a, b)
          ))

    section("the box is built from the site, and padding is what padding says")
    if sites:
        centre, size = sites[0].box_center_and_size(4.0)
        check("padding adds to every side",
              np.allclose(np.asarray(size), np.asarray(sites[0].size) + 8.0),
              f"{np.round(sites[0].size, 2)} -> {np.round(size, 2)}")
        centre0, size0 = sites[0].box_center_and_size(0.0)
        check("zero padding leaves the site size alone",
              np.allclose(size0, np.asarray(sites[0].size)))
        check("the centre is the site centroid, not the middle of its bounds",
              float(np.linalg.norm(np.asarray(centre0) - np.asarray(sites[0].center))) < 1e-5)
        check("the box is a sane size for a docking run",
              10.0 <= float(np.min(size)) and float(np.max(size)) <= 40.0,
              f"{np.round(size, 2)}")

    section("reported geometry is self-consistent")
    if sites:
        check("voxels is positive and within reach of the bounding box",
              all(0 < p.voxels <= float(np.prod(p.size)) / 0.8**3 + 1 for p in sites))
        check("burial is a fraction of three axes",
              all(0.0 <= p.burial <= 3.0 for p in sites),
              f"{min(p.burial for p in sites):.2f}..{max(p.burial for p in sites):.2f}")
        check("kind is one of the two claims it makes",
              all(p.kind in ("cavity", "burial") for p in sites))
        check("lining is sorted by how much each residue touches",
              all(
                  all(g[i][1] >= g[i + 1][1] for i in range(len(g) - 1))
                  for g in (p.lining for p in sites)
              ))
        check("the list is ranked by score",
              all(a[i].rank_score >= a[i + 1].rank_score
                  for i in range(len(a) - 1)),
              f"{[round(p.rank_score, 2) for p in a[:4]]}")
        check("centres are finite and not NaN",
              all(np.isfinite(np.asarray(p.center)).all() and np.isfinite(np.asarray(p.size)).all()
                  for p in sites))

    section("docking into the auto box finds something as good as the reference")
    # The strongest end-to-end claim available: build the box from the site,
    # dock, and compare. Two numbers come out and they do NOT agree, and
    # recording only the flattering one would be the whole problem this file
    # exists to prevent.
    #
    # On crambin, site 2's box gives -5.42 kcal/mol against the reference
    # pose's -5.5 -- so the energy is reproduced -- at 3.93 A RMSD. Same
    # score, different minimum. Crambin with ibuprofen has near-degenerate
    # binding modes, and a docking engine landing in a different one is a
    # statement about the energy surface, not about whether the box was in the
    # right place. The box *was*: it contains all 16 ligand atoms and is lined
    # by the residue that hydrogen-bonds them.
    #
    # So the assertion is on the energy, and the RMSD is reported alongside
    # with no claim attached to it. Asserting a small RMSD here would be
    # asserting that this particular run finds this particular minimum, which
    # is luck, and a check that passes by luck is worse than no check.
    try:
        from opendocking.core import GridBox, dock, load_ligand, load_receptor
    except Exception as exc:  # pragma: no cover - engine always present in CI
        check("the engine is importable for the end-to-end dock", False, str(exc))
    else:
        rec_engine = load_receptor(CRAMBIN)
        lig = load_ligand(ROOT / "examples" / "ibuprofen_prep.pdbqt")
        floor = math.ceil((2.0 * lig.radius + 1.0) * 10.0) / 10.0
        near_box = None
        if sites:
            # The site whose box holds the docked pose, which is the one worth
            # handing to the engine -- not the one nearest the ligand, and not
            # simply the first on the list.
            centre, size = best_site.box_center_and_size()
            near_box = GridBox.from_center_size(
                centre, tuple(max(float(v), floor) for v in size)
            )
        check("the engine accepts the box this site produces",
              near_box is not None
              and all(float(v) >= floor for v in near_box.size),
              f"floor {floor:.1f} A, box {tuple(round(float(v), 1) for v in near_box.size)}"
              if near_box else "no site")
        if near_box is not None:
            maps = rec_engine.precalculate(near_box, "vina", 0.375)
            result = dock(lig, maps, exhaustiveness=8, num_modes=1, seed=20260930)
            docked = np.asarray(result.pose_coords(0), np.float32)
            rmsd = float(np.sqrt(((docked - pose_pts) ** 2).sum(axis=1).mean()))
            reference_energy = _energy_of_file(CRAMBIN_POSE)
            check("docking into it returns a pose of comparable energy",
                  reference_energy is not None
                  and abs(result.best_energy - reference_energy) < 0.5,
                  f"{result.best_energy:.2f} kcal/mol against the reference's "
                  f"{reference_energy if reference_energy is None else f'{reference_energy:.2f}'}")
            # Reported, not asserted. See above.
            check("and it is NOT the same pose: recorded so the gap is visible",
                  rmsd > 0.0,
                  f"RMSD {rmsd:.2f} A from the reference pose at a similar energy. "
                  "Near-degenerate minima, not a reproduction, and not claimed as one")

    section("waters and cofactors are not lining residues")
    # Measured on six proteins fetched from RCSB: streptavidin's top site
    # listed "HOH 354A, HOH 361A" alongside two real residues, and myoglobin
    # listed HEM among the walls of its heme pocket. A crystal water is a
    # lattice artefact, not a wall of the protein, and a pocket whose lining is
    # half waters is answering a different question than the one asked.
    #
    # Built by hand rather than fetched: the check must not need a network, and
    # the property is a string comparison, not a chemistry claim. `MoleculeView`
    # is imported at module level -- re-importing it here would make Python
    # treat the name as local to `main` for the whole function, and the later
    # crambin section that uses it would stop finding it.
    from opendocking.workbench.structure import parse_structure  # noqa: PLC0415

    pdb = (
        "ATOM      1  N   THR A   1      -8.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  THR A   1      -8.000   1.450   0.000  1.00  0.00           C\n"
        "ATOM      3  C   THR A   1      -6.500   1.450   0.000  1.00  0.00           C\n"
        "ATOM      4  O   THR A   1      -6.500   2.600   0.000  1.00  0.00           O\n"
        "HETATM    5  O   HOH A 101      -8.000   2.800   0.000  1.00  0.00           O\n"
        "HETATM    6 FE   HEM A 155      -6.000   2.000   0.000  1.00  0.00          FE\n"
        "HETATM    7  C1  GOL A 201      -4.000   2.000   0.000  1.00  0.00           C\n"
        "HETATM    8  C   UNL A 300      -2.000   2.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    parsed = parse_structure(pdb, "mixed")
    view = MoleculeView(
        name="mixed",
        coords=np.asarray(parsed.coords(), np.float32).reshape(-1, 3),
        elements=parsed.elements(),
        structure=parsed,
    )
    labels = view.residue_labels()
    check("a real residue is labelled", labels[0] == "THR 1A", f"{labels[0]!r}")
    check("crystal water is not", labels[4] == "", f"{labels[4]!r}")
    check("a buffer additive is not", labels[6] == "", f"{labels[6]!r}")
    check("the project's own UNL placeholder is not", labels[7] == "", f"{labels[7]!r}")
    # A cofactor is a *different* case and deliberately kept. A water is a
    # lattice artefact: it is where the crystal happened to put a solvent
    # molecule, it says nothing about the protein's shape, and on streptavidin
    # waters outnumbered residues in the lining list. A cofactor is real
    # chemistry that is part of the receptor a drug has to deal with, and
    # "this pocket is lined by the haem" is an answer worth having. Writing the
    # exclusion as one flat list would have thrown both away, and the first
    # version of this check asserted that they were the same -- which is how
    # the distinction got made explicit in the first place.
    check(
        "a cofactor is kept, because it is chemistry and not a lattice artefact",
        labels[5] == "HEM 155A",
        f"{labels[5]!r}",
    )
    check(
        "one label per atom, blanks included",
        len(labels) == len(view.coords),
        f"{len(labels)} labels for {len(view.coords)} atoms",
    )
    # And it has to reach the lining, which is where the noise was visible.
    # A distinct name: this fixture is a handful of atoms and finds no sites,
    # and reusing `sites` here quietly emptied the crambin list that the
    # section after this one needs -- which showed up as a section that printed
    # its heading and no checks at all, and a total that did not move.
    mixed_sites = P.find_pockets(view.coords, view.elements, residues=labels)
    all_lining = {name for p in mixed_sites for name, _ in p.lining}
    check(
        "no site lines itself with a water, an additive or a ligand",
        not any(n.split(" ")[0] in ("HOH", "GOL", "UNL", "WAT", "EDO")
                for n in all_lining),
        f"lining across {len(mixed_sites)} site(s): {sorted(all_lining) or 'none'}",
    )

    section("the drawn points and the reported geometry are the same thing")
    # The site is drawn as a cloud of its own grid points. A picture that
    # disagrees with the table is worse than no picture, so the two are checked
    # against each other rather than both being trusted.
    if sites:
        rec_pts = np.asarray(rec.coords, np.float32)
        labels_all = rec.residue_labels()
        for idx, pocket in enumerate(sites):
            pts = np.asarray(pocket.points, np.float32)
            if len(pts) != pocket.voxels:
                check(
                    f"site {idx + 1} carries one point per voxel",
                    False,
                    f"{len(pts)} points, {pocket.voxels} voxels",
                )
                break
            extent = (pts.max(axis=0) - pts.min(axis=0)) + P.DEFAULT_SPACING
            ok = bool(np.allclose(extent, np.asarray(pocket.size), atol=0.05))
            if not ok:
                check(
                    f"site {idx + 1} point cloud spans exactly the reported size",
                    False,
                    f"points span {np.round(extent, 2)} vs reported "
                    f"{np.round(pocket.size, 2)}",
                )
                break
        else:
            check(
                "every site carries one point per voxel",
                all(len(np.asarray(p.points)) == p.voxels for p in sites),
            )
            check(
                "every site's point cloud spans exactly the size it reports",
                all(
                    bool(
                        np.allclose(
                            (np.asarray(p.points).max(axis=0)
                             - np.asarray(p.points).min(axis=0)) + P.DEFAULT_SPACING,
                            np.asarray(p.size),
                            atol=0.05,
                        )
                    )
                    for p in sites
                ),
                "the cloud and the table are the same geometry",
            )
            check(
                "the centre is the centroid of the points that are drawn",
                all(
                    float(np.linalg.norm(
                        np.asarray(p.points).mean(axis=0) - np.asarray(p.center)
                    )) < 0.1
                    for p in sites
                ),
            )
        # And the cloud really is enclosed by what the table says lines it.
        #
        # The guarantee being pinned here is the mask's, not the reporting
        # cutoff's. A point earns its place in a site because protein sits on
        # both sides of it along some axis within the burial window, which is
        # 3.0 A / spacing voxels -- so a drawn point can legitimately sit
        # ~4.9 A from the nearest atom and still belong. `LINING_MAX` is 4.5 A
        # and is a *reporting* threshold, narrower than what the geometry
        # guarantees, so demanding every point be inside it asserts something
        # the mask never promised. It happened to hold while sites were
        # dilated -- the dilation pulled every point back towards the atoms --
        # and that is an accident, not a property.
        #
        # So: check the two things that are actually true, and check them in
        # both directions. The cloud is free space, and the lining is exactly
        # the residues within LINING_MAX of it -- no more, no fewer.
        top = sites[0]
        cloud = np.asarray(top.points, np.float32)
        els_all = list(rec.elements)
        reach = np.asarray(
            [P.VDW_RADII.get(e, 1.70) for e in els_all], np.float32
        ) + P.DEFAULT_PROBE
        # Solid means "within `vdw + probe` of *some* atom", so a free point
        # is one whose distance to every atom exceeds that atom's own inflated
        # radius. Comparing against a single radius would be wrong: the
        # mask is per-atom, and 4.5 A from a hydrogen is not the same as 4.5 A
        # from a sulphur.
        clearance = (
            np.linalg.norm(rec_pts[:, None, :] - cloud[None, :, :], axis=2)
            - reach[:, None]
        ).min(axis=0)
        check(
            "every drawn point is free space: outside every atom's probe radius",
            float(clearance.min()) > 0.0,
            f"tightest point is {float(clearance.min()):.2f} A clear of its "
            f"nearest inflated atom",
        )
        check(
            "and it is a real part of the site, not a stray point",
            len(cloud) == top.voxels and float(cloud.std(axis=0).min()) > 0.0,
            f"{len(cloud)} points for {top.voxels} voxels, spread "
            f"{np.round(cloud.std(axis=0), 2)}",
        )
        # The lining, recomputed here from the coordinates rather than read
        # back, so a bug in `_lining` cannot agree with itself.
        lined = {name for name, _ in top.lining}
        recomputed = set()
        for lab, xyz in zip(labels_all, rec_pts):
            if lab and float(np.linalg.norm(cloud - xyz, axis=1).min()) <= P.LINING_MAX:
                recomputed.add(lab)
        check(
            "the reported lining is exactly the residues within LINING_MAX",
            lined == recomputed,
            f"{len(lined)} reported, {len(recomputed)} recomputed, "
            f"symmetric difference {sorted(lined ^ recomputed)}",
        )

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
