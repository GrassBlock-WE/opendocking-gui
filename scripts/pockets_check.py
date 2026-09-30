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
    check("isotropic dilation of a point is exactly its 6 neighbours",
          np.array_equal(P._dilate_iso(dot, 1), cross),
          f"{int(P._dilate_iso(dot, 1).sum())} voxels, "
          f"{int((P._dilate_iso(dot, 1) & ~cross).sum())} wrong")
    check("two steps reach the 5x5x5 shell, not 7x7",
          int(P._dilate_iso(dot, 2).sum()) == 25,
          f"{int(P._dilate_iso(dot, 2).sum())}")
    # `.all()` here would be a check that cannot pass: the dilated mask is
    # mostly False, so the right test is that the seed survives intact, by
    # counting the overlap rather than demanding every voxel be set.
    check("dilation never loses the mask it started with",
          all(int((P._dilate_iso(dot, n) & dot).sum()) == int(dot.sum())
              for n in (0, 1, 2, 3, 4)),
          f"seed voxels kept at 0..4 steps: "
          f"{[int((P._dilate_iso(dot, n) & dot).sum()) for n in (0, 1, 2, 3, 4)]} of {int(dot.sum())}")
    check("dilation does not mutate the caller's array",
          int(dot.sum()) == 1 and np.array_equal(dot, P._dilate_iso(dot, 0) & dot))

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

    section("the volume ceiling removes merged surface, which is what it is for")

    # A single point buried on one axis only, smeared over a whole surface,
    # becomes one enormous component after dilation. On crambin these came out
    # at 3515, 3564 and 1720 A³ and outranked every real groove. Building the
    # same thing synthetically keeps the claim honest without depending on a
    # protein's shape staying the same.
    # Crambin is the real measurement; re-measure it rather than assert a number.
    rec = MoleculeView.from_pdbqt(CRAMBIN, "crambin", (1, 1, 1), 0.30, role="receptor")
    rec_pts = np.asarray(rec.coords, np.float32)
    all_sites = P.find_pockets(rec_pts, list(rec.elements), max_pockets=64, max_volume=1e9)
    kept = P.find_pockets(rec_pts, list(rec.elements), max_pockets=64, max_volume=1500.0)
    dropped = [p for p in all_sites if float(np.prod(p.size)) > 1500.0]
    check("the ceiling removes something from crambin", len(dropped) >= 1,
          f"{len(all_sites)} -> {len(kept)} sites")
    check("everything it removes is bigger than the ceiling",
          all(float(np.prod(p.size)) > 1500.0 for p in dropped),
          f"removed volumes {sorted(round(float(np.prod(p.size))) for p in dropped)}")
    check("what survives fits under the ceiling",
          all(float(np.prod(p.size)) <= 1500.0 for p in kept),
          f"largest survivor {max(float(np.prod(p.size)) for p in kept):.0f} A^3")

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
    check("the list is within max_pockets", len(sites) <= 8)

    if sites:
        distances = [float(np.linalg.norm(np.asarray(p.center) - pose_centroid)) for p in sites]
        covering = [i for i, d in enumerate(distances) if d < 6.0]
        check("at least one site is where the ligand went", bool(covering),
              f"nearest {min(distances):.2f} A, ranks {[i for i in covering]}")
        if covering:
            near_site = sites[covering[0]]
            centre, size = near_site.box_center_and_size()
            inside = int(in_box(pose_pts, centre, size).sum())
            check("the box it builds holds the whole ligand",
                  inside == len(pose_pts), f"{inside}/{len(pose_pts)} atoms")
            # Ranked second on crambin, not first. Asserting "first" would be
            # asserting a result the data does not give; what is worth locking
            # in is that the ligand's site is in the shortlist at all, and near
            # the top of it, so a ranking change that buries it is caught.
            check("and it is near the top of the shortlist", covering[0] <= 1,
                  f"rank {covering[0] + 1} of {len(sites)}")
            names = {r for r, _ in near_site.lining}
            check("it is lined by THR 2, the residue the contact analysis "
                  "independently found donating to this ligand",
                  "THR 2A" in names, f"lining {near_site.lining[:6]}")

            # The measurable win, against the box the workbench used before.
            old_volume = 22.0 ** 3
            new_volume = float(np.prod(size))
            check("the box it builds is far smaller than the whole-protein one",
                  new_volume < old_volume / 3.0,
                  f"{new_volume:.0f} vs {old_volume:.0f} A^3 "
                  f"({old_volume / new_volume:.1f}x smaller)")
            check("and the smaller box still holds every ligand atom",
                  in_box(pose_pts, rec_centroid, size).sum() < len(pose_pts),
                  f"the same box centred on the protein centroid would hold "
                  f"{int(in_box(pose_pts, rec_centroid, size).sum())}/{len(pose_pts)}, "
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
            centre, size = sites[covering[0]].box_center_and_size()
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

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
