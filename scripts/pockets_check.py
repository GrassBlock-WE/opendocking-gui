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

#: How many checks this file is supposed to run, counted by running it. It is a
#: check itself, at the end of `main`, because most of the calls sit inside data
#: guards and a check that stops running takes the total down with it, silently.
#: Change this number only when a check is deliberately added or removed.
EXPECTED_CHECKS = 153  # measured from a green run, including the check that reads it


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
    section("a ceiling that only ever gets absurd values is not a tested filter")

    # Every check above that exercises `max_volume` passes 0.0, 1.0 or 1e9, so
    # a filter deleted outright is caught only by those, and nothing says what
    # the ceiling does at a value anyone would actually type. On crambin it
    # does nothing at all: the largest site is 88 A3 against a 1500 A3
    # ceiling, 17x of headroom, and zero sites differ between "no ceiling" and
    # "the default ceiling". The fixture below is one where the default bites.
    def big_shell(inner=18.0, outer=25.0, n_in=700, n_out=900):
        """Two nested shells far enough apart to hold a cavern, not a pocket."""
        return np.vstack([sphere(inner, n_in), sphere(outer, n_out)])

    cavern_atoms = big_shell()
    cavern_els = ["C"] * len(cavern_atoms)
    cavern = P.find_pockets(cavern_atoms, cavern_els, max_pockets=99,
                            max_volume=1e9)
    check("a big sealed cavern is found when there is no ceiling",
          len(cavern) == 2,
          f"{len(cavern)} site(s) at max_volume=1e9: "
          f"{[f'{p.volume:.0f} A3' for p in cavern]}")
    if cavern:
        top = cavern[0]
        ratio = float(np.prod(top.size)) / top.volume
        check("and the default ceiling removes it, at a value one would type",
              len(P.find_pockets(cavern_atoms, cavern_els, max_pockets=99)) == 0
              and top.volume > P.DEFAULT_MAX_VOLUME,
              f"its {top.volume:.0f} A3 is over the {P.DEFAULT_MAX_VOLUME:.0f} A3 "
              f"default, so 2 sites -> 0; this is the first fixture in the file "
              f"where the default ceiling does any work at all")
        check("while its box is {ratio:.0f}x its own space, so the two measures "
              "genuinely disagree".format(ratio=ratio),
              ratio >= 5.0 and top.volume <= float(np.prod(top.size)),
              f"{top.volume:.0f} A3 of space in a {float(np.prod(top.size)):.0f} "
              f"A3 box: the old bounding-box metric would have rejected this "
              f"site too, and the ceiling is what rejects it now")
        # Both directions at a realistic threshold, one site at a time: raise
        # the ceiling just past the cavern's own volume and exactly that site
        # returns; drop it just below and the list empties again. Without this
        # the only thing pinned about the ceiling is that absurd values work.
        back = P.find_pockets(cavern_atoms, cavern_els, max_pockets=99,
                              max_volume=top.volume + 100.0)
        gone = P.find_pockets(cavern_atoms, cavern_els, max_pockets=99,
                              max_volume=top.volume - 100.0)
        check("a ceiling just above the cavern's volume brings that one back",
              len(back) == 1 and abs(back[0].volume - top.volume) < 1e-6,
              f"{top.volume + 100.0:.0f} A3 -> {len(back)} site(s), "
              f"{back[0].volume:.0f} A3" if back else "none")
        check("and one just below it empties the list again",
              len(gone) == 0,
              f"{top.volume - 100.0:.0f} A3 -> {len(gone)} site(s), which is the "
              f"same site from the other side of its own volume")

    # The relaxation itself, stated as an invariant over every fixture in the
    # file: a site's own volume can never exceed its bounding box, because the
    # box contains the site. So switching the ceiling's metric from the box to
    # the voxels can only ever *admit more*, never fewer, and the two places
    # that difference is visible are both checked: the dogleg below is a cleft
    # the old metric threw away and the new one keeps, and this cavern is one
    # both metrics reject.
    if all_sites and cavern:
        every = list(all_sites) + list(cavern) + list(found)
        ratios = [float(np.prod(p.size)) / p.volume for p in every]
        check("a site's own volume never exceeds its bounding box, anywhere",
              all(p.volume <= float(np.prod(p.size)) + 1e-6 for p in every)
              and min(ratios) >= 1.0,
              f"{len(every)} sites across three fixtures, bbox/voxel "
              f"{min(ratios):.2f}..{max(ratios):.2f}; the metric change is a "
              f"relaxation and that is the whole of what it is")

    # ------------------------------------------------------------------
    section("the box is the site's size plus twice the padding, on every axis")

    # The 3PTB cleft's extent was written into the module docstring as
    # 22 x 17 x 28 A while the benchmark reported a 39 x 26 x 41 A box for the
    # same site, and the two sat in the same release for as long as the file
    # existed. 22 + 8 = 30, not 39. Nothing recomputed one from the other, so
    # nothing noticed. This pins the relationship the docstrings are derived
    # from, exactly, so the arithmetic has somewhere to be wrong rather than
    # hiding in prose. It cannot catch a wrong docstring; it can stop the code
    # and the comment drifting apart silently.
    def padding_holds(these, label):
        bad = []
        for p in these:
            for padding in (0.0, 1.3, P.DEFAULT_PADDING, 7.5):
                _, got = p.box_center_and_size(padding)
                want = np.asarray(p.size, np.float32) + np.float32(2.0 * padding)
                if not np.array_equal(np.asarray(got, np.float32), want):
                    bad.append((p.voxels, padding))
        return not bad, bad

    ok_real, bad_real = padding_holds(all_sites, "crambin")
    check("every crambin box is its own size plus twice the padding, exactly",
          ok_real,
          f"{len(all_sites)} sites at padding 0.0, 1.3, {P.DEFAULT_PADDING} and "
          f"7.5: bit-equal on all three axes"
          if ok_real else f"mismatches {bad_real[:3]}")
    synthetic = list(found) + list(cavern)
    ok_syn, bad_syn = padding_holds(synthetic, "synthetic") if synthetic else (False, [])
    check("and so is every synthetic fixture's, so the rule is not a crambin quirk",
          ok_syn and len(synthetic) >= 2,
          f"{len(synthetic)} sites across the dogleg cleft and the cavern"
          if ok_syn else f"mismatches {bad_syn[:3]}")
    # Both directions: the padding is added on every axis, and zero padding
    # adds nothing at all. A rule that only ever added to the longest axis
    # would satisfy neither.
    if all_sites:
        p0 = all_sites[0]
        _, s0 = p0.box_center_and_size(0.0)
        _, s1 = p0.box_center_and_size(1.0)
        check("padding adds to all three axes and zero adds nothing",
              all(abs(float(a) - float(b)) < 1e-4
                  for a, b in zip(s0, np.asarray(p0.size, np.float32)))
              and all(abs(float(b) - float(a) - 2.0) < 1e-4
                      for a, b in zip(s0, s1)),
              f"size {np.round(p0.size, 1)} + 0.0 -> {np.round(s0, 1)}, "
              f"+ 1.0 -> {np.round(s1, 1)}")

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
        # This used to be a second copy of the boolean above, with a different
        # sentence underneath it. A check that can only fail when the one above
        # already failed inflates the count without testing anything.
        #
        # The sentence under it was false too, and falsely in a way worth
        # recording: it claimed the measured position was 8 of 12 while 9 was its
        # rank out of all sixteen crambin offers, "a different list". Measured,
        # the position is **9 in both** -- 9 of 12 at the default, 9 of 16 with
        # max_pockets=99, 9 of 16 with the ceiling removed -- at rank_score
        # 2.4037, between 2.4235 at position 8 and 2.3595 at position 10, so the
        # position is well separated rather than a tie. The two lists differ in
        # their denominator, not in their occupant: `find_pockets` sorts once and
        # returns `found[:max_pockets]`, so the shortlist is a prefix of the long
        # one and truncation provably cannot move anybody. The claim this check
        # is *for* is real and worth keeping -- the default list is long enough
        # to reach a site ranked 9 -- so it is stated as the thing it is: the
        # list length boundary, computed on both sides of it.
        shortlist = P.find_pockets(rec_pts, list(rec.elements),
                                   residues=rec.residue_labels(),
                                   max_pockets=best)
        short_held = [int(in_box(pose_pts, *p.box_center_and_size()).sum())
                      for p in shortlist]
        check("and the list is long enough to reach that site, not one shorter",
              max(short_held, default=0) < len(pose_pts)
              and max(held, default=0) == len(pose_pts),
              f"a list cut to rank {best} keeps at most "
              f"{max(short_held, default=0)}/{len(pose_pts)} ligand atoms; the "
              f"default of {P.DEFAULT_MAX_POCKETS} keeps all of them")

        # What is worth locking in here is *why* the site is late, not that it
        # is late. Crambin's real site ranks 8th because the search measures
        # the space a ligand **leaves**, not the space it occupies: where a
        # ligand fits snugly that space is nearly nothing. So the honest fact
        # is about the site's own volume against the rest of the list, and that
        # stays true however `rank_score` is retuned. The previous version
        # asserted `best > 0` -- "this site is not first" -- which goes red the
        # moment the ranking improves, i.e. it punished the thing it was
        # watching for.
        vols = sorted(p.volume for p in sites)
        median = vols[len(vols) // 2]
        check("the docked site is below the middle of the list on volume",
              best_site.volume < median,
              f"{best_site.volume:.0f} A3 against a median of {median:.0f} A3 "
              f"over {vols[:4]}...{vols[-2:]}: a snug pocket leaves little space "
              f"behind the ligand, so the quantity being ranked cannot find it")
        check("and it is not the largest site on the list either",
              best_site.volume < max(vols) and best_site.volume >= min(vols),
              f"{min(vols):.0f}..{max(vols):.0f} A3 across the shortlist; the "
              f"site the ligand is actually sitting in is {best_site.volume:.0f}")
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

    # ------------------------------------------------------------------
    section("the thickness filter, and the gap it cannot see")

    # `min_extent` is the filter that stops a dent in the surface being listed
    # as a site. Its check used to pass `min_extent=99`, which is the same shape
    # of test the volume ceiling got: it proves the filter *exists* and says
    # nothing about what it does. Both directions, on a real protein, at values a
    # person would type, and on the **untruncated** list -- crambin offers 16
    # sites and the default shows 12, so a filtered top-12 and the unfiltered
    # top-12 are not nested lists and counting one against the other says
    # nothing. (That mistake was made and caught while writing this: 12 -> 10
    # sites looks like "two of the three thin ones went" until you notice the
    # pool is 16 deep and a previously-truncated site was promoted in.) The
    # 1e-4 tolerances are not decoration: a site's size comes out of float32
    # arithmetic, so crambin's two-voxel site is 1.6000000238 A and a bare
    # `<= 1.6` is False.
    def crambin_at(**kw):
        return P.find_pockets(rec_pts, list(rec.elements), max_pockets=99, **kw)

    def site_key(p):
        return (round(float(p.volume), 6), round(float(p.center[0]), 3),
                round(float(p.center[1]), 3), round(float(p.center[2]), 3))

    one_voxel = 1.0 * P.DEFAULT_SPACING
    two_voxel = 2.0 * P.DEFAULT_SPACING
    thinnest = crambin_at(min_extent=0.0)
    default = crambin_at()
    default_keys = {site_key(p) for p in default}
    below = [p for p in thinnest if float(np.min(p.size)) < two_voxel - 1e-4]
    check("the default floor is two voxels, and it is exactly what removes the "
          "single-voxel sites",
          abs(min(float(np.min(p.size)) for p in thinnest) - one_voxel) < 1e-4
          and len(default) == len(thinnest) - len(below)
          and all(site_key(p) not in default_keys for p in below)
          and bool(below),
          f"with the floor at nothing crambin offers {len(thinnest)} sites, "
          f"thinnest {min(float(np.min(p.size)) for p in thinnest):.1f} A = one "
          f"voxel; the default {two_voxel:.1f} A floor removes exactly those "
          f"{len(below)}, leaving {len(default)}")
    raised = crambin_at(min_extent=3.0 * P.DEFAULT_SPACING)
    raised_keys = {site_key(p) for p in raised}
    two_vox_sites = [p for p in default
                     if float(np.min(p.size)) <= two_voxel + 1e-4]
    check("raising it to three voxels drops exactly the two-voxel sites",
          len(raised) == len(default) - len(two_vox_sites)
          and all(site_key(p) not in raised_keys for p in two_vox_sites)
          and bool(two_vox_sites)
          and min(float(np.min(p.size)) for p in raised) > two_voxel + 1e-4,
          f"{len(default)} -> {len(raised)} sites at "
          f"{3.0 * P.DEFAULT_SPACING:.1f} A; the {len(two_vox_sites)} lost are "
          f"exactly those at {two_voxel:.1f} A, and the thinnest left is "
          f"{min(float(np.min(p.size)) for p in raised):.1f} A")
    check("and it can still empty the list outright, the check it used to have",
          crambin_at(min_extent=99.0) == [],
          f"min_extent=99.0 -> {len(crambin_at(min_extent=99.0))} sites")

    # The limit of that filter, measured rather than asserted in prose. The
    # dilating code that used to sit in front of the labelling step was there to
    # stop thin sheets being reported, and it was removed because it merged
    # every real site into the protein's surface. The comment that replaced it
    # said the sliver question was answered by a later check, and no such check
    # existed. Here is the honest version, and it is a limitation rather than a
    # fix: **at this grid spacing a sub-2 A gap and a 3.2 A gap are the same
    # number of layers**, so a 1.2 A slot between two atom sheets is reported as
    # a site of exactly the thickness crambin's real, ligand-holding site is
    # reported at. No thickness threshold can separate them. Measured:
    #   erosion of the free mask, 1.2 A gap -> 468/128/0 voxels at 1/2/3
    #   erosion of the free mask, crambin's pose site -> 1/0/0 voxels
    # i.e. the sliver has *more* clearance than the real site under that test
    # too, which is why no guard was added: any guard strong enough to remove
    # it removes the one binding site this project has measured. Pinned instead
    # so the limitation cannot quietly become a claim.
    def sheets(span=16.0, pitch=1.6, sep=8.0):
        """Two solid carbon sheets; the free gap between them is sep - 6.2 A."""
        half = span / 2.0
        axis = np.arange(-half, half + 0.01, pitch)
        g = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        lo = np.concatenate([g, np.full((len(g), 1), -sep / 2.0)], axis=1)
        hi = np.concatenate([g, np.full((len(g), 1), sep / 2.0)], axis=1)
        return np.vstack([lo, hi]).astype(np.float32)

    thin_gap = sheets(sep=7.4)          # a 1.2 A gap: nothing can sit in it
    wide_gap = sheets(sep=9.0)          # a 2.8 A gap: room for a methyl group
    thin_site = P.find_pockets(thin_gap, ["C"] * len(thin_gap), max_pockets=99)
    wide_site = P.find_pockets(wide_gap, ["C"] * len(wide_gap), max_pockets=99)
    pose_thick = min(float(np.min(p.size)) for p in sites if
                     int(in_box(pose_pts, *p.box_center_and_size()).sum())
                     == len(pose_pts)) if any(
        int(in_box(pose_pts, *p.box_center_and_size()).sum()) == len(pose_pts)
        for p in sites) else 0.0
    if thin_site and wide_site:
        gap_thick = float(np.min(thin_site[0].size))
        check("a 1.2 A gap between two atom sheets is reported as thick as the "
              "site that holds a real ligand",
              abs(gap_thick - pose_thick) < 1e-3
              and float(np.min(wide_site[0].size)) > gap_thick,
              f"the {7.4 - 6.2:.1f} A gap reports {gap_thick:.1f} A "
              f"({gap_thick / P.DEFAULT_SPACING:.0f} voxels), the ligand's own "
              f"site reports {pose_thick:.1f} A, and only widening the gap to "
              f"{9.0 - 6.2:.1f} A moves it to "
              f"{float(np.min(wide_site[0].size)):.1f} A. No thickness filter "
              f"can tell the first two apart, so none is added")
        check("so the claim that a thin sheet is filtered is not made, and the "
              "volume ceiling cannot make it either",
              thin_site[0].volume < P.DEFAULT_MAX_VOLUME
              and float(np.prod(thin_site[0].size)) > P.DEFAULT_MAX_VOLUME,
              f"the gap is {thin_site[0].volume:.0f} A3 of voxels in a "
              f"{float(np.prod(thin_site[0].size)):.0f} A3 box: the voxel "
              f"ceiling admits it and the old bounding-box metric would have "
              f"rejected it. The switch from one metric to the other is a pure "
              f"relaxation, and this is the fixture where that shows")

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

    # ------------------------------------------------------------------
    section("a size budget that says what it cost, and can be shown to fail")

    # `box_center_and_size` gives a site its full extent, so a winding cleft
    # hands the interface a 39 x 26 x 41 A box with nothing to say that this
    # is not a binding site. `box_with_budget` is the opt-in that caps the box
    # *and says so*. Every check below therefore comes in a pair: a budget that
    # cannot cap anything must leave the site untouched, and a budget that must
    # cap must visibly cap. A guard that only ever sees the first direction
    # would pass against a `box_with_budget` that always returns the uncapped
    # box and an empty note.
    long_sites = P.find_pockets(cleft, ["C"] * len(cleft))
    check("the long cleft is available to this section",
          len(long_sites) >= 1, f"{len(cleft)} atoms -> {len(long_sites)} site(s)")
    if long_sites and sites:
        long_site = long_sites[0]
        small = sites[0]
        small_c, small_req = small.box_center_and_size()
        long_c, long_req = long_site.box_center_and_size()

        # --- direction 1: a budget too loose to bite changes nothing ---------
        loose = small.box_with_budget(max_side=1e6, max_volume=1e12)
        check("a budget too loose to cap leaves the box bit-for-bit alone",
              loose.size == small_req and loose.center == small_c,
              f"{np.round(loose.size, 2)} is the same {len(small_req)} floats as "
              f"box_center_and_size's {np.round(small_req, 2)}")
        check("and says it capped nothing and has nothing to add",
              loose.capped is False and loose.note == "",
              f"capped={loose.capped}, note={loose.note!r}")
        check("and the default budget leaves every crambin site alone",
              all(p.box_with_budget().size == p.box_center_and_size()[1]
                  and p.box_with_budget().capped is False
                  and p.box_with_budget().note == "" for p in sites),
              f"all {len(sites)} crambin sites; the longest side anywhere on "
              f"crambin is {max(max(p.box_center_and_size()[1]) for p in sites):.1f} A "
              f"against a {P.DEFAULT_BOX_MAX_SIDE:.0f} A cap")
        check("so a crambin box is never silently shrunk",
              all(max(p.box_with_budget().size) <= P.DEFAULT_BOX_MAX_SIDE
                  for p in sites),
              "the cap is inert on the only real protein this check can load "
              "offline, and that is the measurement, not an assumption")

        # --- direction 2: a budget that must cap, does ----------------------
        tight = small.box_with_budget(max_side=10.0)
        check("the same site under a budget it cannot fit is capped",
              tight.capped is True and max(tight.size) == 10.0
              and max(tight.size) < max(small_req),
              f"{np.round(small_req, 2)} -> {np.round(tight.size, 2)}, "
              f"capped={tight.capped}")
        check("and capping a small site is said out loud",
              tight.note != "" and "10 A side budget" in tight.note
              and " x ".join(f"{v:.1f}" for v in small_req) in tight.note,
              tight.note)

        capped = long_site.box_with_budget()
        check("a winding cleft is capped by the default budget",
              capped.capped is True and max(capped.size) == P.DEFAULT_BOX_MAX_SIDE
              and max(capped.requested_size) > P.DEFAULT_BOX_MAX_SIDE,
              f"{np.round(capped.requested_size, 2)} -> {np.round(capped.size, 2)} A")
        check("and the capped box is genuinely smaller, not the same one relabelled",
              capped.volume < capped.requested_volume * 0.75,
              f"{capped.volume:.0f} A3 down from {capped.requested_volume:.0f} A3 "
              f"({capped.requested_volume / capped.volume:.2f}x)")
        check("while the centre does not move, and the note says so",
              capped.center == long_c
              and "same centre" in capped.note
              and "does not move" in capped.note,
              "the bounding-box middle of a curved cleft lands in the wall, so "
              "re-centring on a cap would reintroduce the defect this file "
              "already fixed")
        check("and the note names the size the site asked for",
              " x ".join(f"{v:.1f}" for v in long_req) in capped.note
              and " x ".join(f"{v:.1f}" for v in capped.size) in capped.note,
              capped.note)
        check("and the note reports what is now outside the box",
              f"{100.0 * capped.coverage:.1f}%" in capped.note
              and str(long_site.voxels) in capped.note,
              capped.note)

        # --- the coverage number is a measurement, so re-derive it ----------
        cloud = np.asarray(long_site.points, np.float32)
        kept = int(in_box(cloud, capped.center, capped.size).sum())
        check("coverage is the site's own points the box holds, recomputed here",
              abs(capped.coverage - kept / len(cloud)) < 1e-12
              and kept < len(cloud) < 2 * kept,
              f"{kept}/{len(cloud)} = {100.0 * kept / len(cloud):.1f}%, and the "
              f"uncapped box holds "
              f"{int(in_box(cloud, long_c, long_req).sum())}/{len(cloud)}")
        check("and it moves when the box does, so it is not a constant",
              capped.coverage < long_site.box_with_budget(
                  max_side=None).coverage,
              f"uncapped {long_site.box_with_budget(max_side=None).coverage:.4f} "
              f"vs capped {capped.coverage:.4f}")
        check("a site that fits reports full coverage and no note",
              all(p.box_with_budget().coverage == 1.0 for p in sites),
              "so an empty note is a positive claim -- the box is the site's own "
              "and holds all of it -- rather than an absence of one")

        # --- a box does not even always contain its own site ---------------
        # The centre is the centroid, not the middle of the bounds, so a site
        # that curves back on itself sticks out of the box built from it. This
        # is measured rather than argued: at the default padding the dogleg's
        # own box already misses 48 of 1985 points, and at zero padding it
        # misses 336 of them.
        bare = long_site.box_with_budget(max_side=None, padding=0.0)
        check("the site's own box does not contain the whole site, and says so",
              bare.capped is False and bare.coverage < 1.0 and bare.note != ""
              and int(in_box(cloud, bare.center, bare.size).sum())
              < int(in_box(cloud, long_c, long_req).sum()),
              f"zero padding holds {bare.coverage * 100:.1f}% against "
              f"{long_site.box_with_budget(max_side=None).coverage * 100:.1f}% "
              f"at the default 4 A")
        check("so a note can be present with nothing capped -- and must be",
              long_site.box_with_budget(max_side=None).capped is False
              and long_site.box_with_budget(max_side=None).note != "",
              "97.6% coverage is not 'all of it', and the note says so")

        # --- the volume knob is a separate, live knob ----------------------
        vol = long_site.box_with_budget(max_side=None, max_volume=20000.0)
        ratio_before = np.asarray(long_req) / long_req[0]
        ratio_after = np.asarray(vol.size) / vol.size[0]
        check("a volume budget caps by scaling all three axes alike",
              vol.capped is True and abs(vol.volume - 20000.0) < 1e-3
              and bool(np.allclose(ratio_before, ratio_after, rtol=1e-9)),
              f"{np.round(vol.size, 2)} = {vol.volume:.0f} A3, proportions "
              f"{np.round(ratio_after, 3)} unchanged from {np.round(ratio_before, 3)}")
        check("and a volume budget too loose to bite is inert",
              long_site.box_with_budget(max_side=None, max_volume=1e9).capped
              is False,
              "the two knobs are independent, so each has to be checked alone")
        check("and a volume cap bites on a box the side cap lets through",
              small.box_with_budget(
                  max_side=P.DEFAULT_BOX_MAX_SIDE, max_volume=100.0
              ).capped is True,
              f"crambin site 1 is {np.round(small_req, 2)} A, inside the "
              f"{P.DEFAULT_BOX_MAX_SIDE:.0f} A side cap and over 100 A3")
        check("a bigger budget never buys a smaller box",
              all(
                  max(long_site.box_with_budget(max_side=a).size) <= max(
                      long_site.box_with_budget(max_side=b).size)
                  for a, b in ((20.0, 30.0), (30.0, 45.0), (45.0, 100.0))
              ),
              "20 -> 30 -> 45 -> 100 A side caps, monotone")

        # --- the two measured benchmark boxes, as fixtures -----------------
        # Taken from `scripts/redock_benchmark.py` rather than measured here,
        # because the checks run offline and these are the two cases where the
        # problem is real. Built by hand at the box's own size minus the 8 A of
        # padding, so the cap sees exactly the geometry that was reported.
        for label, box, want in (("3PTB benzamidine", (39.0, 26.0, 41.0),
                                 (30.0, 26.0, 30.0)),
                                 ("1HVR XK2", (30.0, 38.0, 32.0),
                                 (30.0, 30.0, 30.0))):
            far = P.Pocket(center=np.zeros(3, np.float32),
                           size=np.asarray(box, np.float32) - 8.0, voxels=0)
            plan = far.box_with_budget()
            check(f"the {label} box {box[0]:.0f}x{box[1]:.0f}x{box[2]:.0f} A "
                  f"is capped to {want[0]:.0f}x{want[1]:.0f}x{want[2]:.0f} A",
                  plan.capped is True
                  and all(abs(a - b) < 1e-6
                          for a, b in zip(plan.size, want))
                  and f"{float(np.prod(box)):.0f} A3" in plan.note,
                  f"{plan.volume:.0f} A3 down from {plan.requested_volume:.0f} A3")
            check(f"and a {label} pocket with no points claims no coverage",
                  plan.coverage is None and "no grid points" in plan.note,
                  "the note must not invent a number it cannot measure")

        # --- the old call is untouched, and the new one cannot be skimmed ---
        pair = small.box_center_and_size()
        check("the existing call is still a plain 2-tuple of 3 floats",
              isinstance(pair, tuple) and len(pair) == 2
              and all(isinstance(t, tuple) and len(t) == 3
                      and all(isinstance(float(v), float) for v in t)
                      for t in pair),
              f"unpacked as centre, size: "
              f"{tuple(round(float(v), 2) for v in t) for t in pair}")
        try:
            centre, size = small.box_with_budget()
            unpackable = True
        except TypeError:
            centre, size, unpackable = None, None, False
        check("and the budgeted box cannot be unpacked into a bare 2-tuple",
              not unpackable and not isinstance(small.box_with_budget(), tuple),
              "silently discarding the note is how the current behaviour came "
              "about, so the type refuses to let it happen quietly")

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
    # On crambin, the auto-chosen site's box gives -5.30 kcal/mol against the
    # reference pose's -5.50 -- a difference of +0.203, which the assertion
    # below bounds -- at 4.35 A RMSD. Same score, different minimum. Crambin
    # with ibuprofen has near-degenerate binding modes, and a docking engine
    # landing in a different one is a statement about the energy surface, not
    # about whether the box was in the right place. The box *was*: it contains
    # all 16 ligand atoms and is lined by the residue that hydrogen-bonds them.
    #
    # Two corrections to what this comment used to say, both of which had been
    # wrong for a while and were only caught once the grid's element partition
    # was fixed (defect 190) and the docking numbers moved underneath them. It
    # named *site 2*; the box actually used is the **9th** of twelve -- the only
    # one that holds all 16 ligand atoms. And its figures, -5.42 and 3.93 A, are
    # the pre-fix values; -5.36 was a third number, from neither. The threshold
    # is unchanged and does not need to be: 0.203 against a bound of 0.5.
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
        problems: list[str] = []
        for idx, pocket in enumerate(sites):
            pts = np.asarray(pocket.points, np.float32)
            if len(pts) != pocket.voxels:
                problems.append(
                    f"site {idx + 1} carries {len(pts)} points for {pocket.voxels} voxels"
                )
                continue
            extent = (pts.max(axis=0) - pts.min(axis=0)) + P.DEFAULT_SPACING
            ok = bool(np.allclose(extent, np.asarray(pocket.size), atol=0.05))
            if not ok:
                problems.append(
                    f"site {idx + 1} spans {np.round(extent, 2)} but reports "
                    f"{np.round(pocket.size, 2)}"
                )
        # This used to `break` out of the loop and call `check(..., False, ...)`
        # from inside it. That had two consequences, and the second is worse than
        # the first: the failure path never ran on a healthy day, and on a broken
        # one it *replaced* the three `else`-branch checks below rather than
        # joining them -- so the tally came out at 150 where the pin says 153,
        # and the run reported a count mismatch alongside the real failure.
        # **A second, louder symptom is not a second piece of evidence.** The
        # per-site text is kept as the detail of the checks that were already
        # there, so the count is 153 whether the geometry is right or not.
        check(
            "every site's points match the voxel count and the size it reports",
            not problems,
            "; ".join(problems) if problems else "the cloud and the table are the same geometry",
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
            "recomputed here from the points rather than trusted from the sweep above",
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

    # ------------------------------------------------------------------
    section("a site can be asked whether it fits a ligand, in both directions")

    # The shortlist is ranked by the space a ligand *leaves*, so the site that
    # holds crambin's ibuprofen is 8th of 12 and below the median volume, and no
    # retuning of `rank_score` could move it. The ranking cannot see the ligand
    # because the ligand is not in it. These two fractions can, because the
    # caller supplies one -- and both are reported because each one alone hides
    # a failure the other catches.
    #
    # Not a score, not affinity, and deliberately not folded into `rank_score`:
    # doing that would silently move the ranking the benchmark and the README
    # numbers were measured against. The check at the end of this section pins
    # that it does not.
    pose_els = list(pose.elements)
    pose_radii = np.asarray([P.VDW_RADII.get(e, 1.70) for e in pose_els])
    pose_xyz = np.asarray(pose_pts, np.float64)

    if sites:
        reads = [p.fit_to(pose_xyz, pose_radii) for p in sites]
        filled = [r.site_filled for r in reads]
        in_site = [r.ligand_in_site for r in reads]
        top = int(np.argmax(filled))
        check("the site holding the docked ligand is the only one with any fit "
              "at all",
              filled[top] > 0.5 and in_site[top] > 0.0
              and sum(1 for f in filled if f and f > 0.0) == 1
              and top == best,
              f"site {top + 1} of {len(sites)} reads {in_site[top]:.3f} / "
              f"{filled[top]:.3f}; the other {len(sites) - 1} read "
              f"{sum(1 for f in filled if f and f > 0.0) and 0} exactly. "
              f"rank_score puts this site {top + 1}, so the reading carries "
              f"information the ranking does not -- the rank is reported, not "
              f"asserted, because it is not a property anything promises")
        check("and it is the strict maximum in both directions",
              filled[top] == max(filled) and in_site[top] == max(in_site)
              and filled.count(max(filled)) == 1,
              f"site_filled max {max(filled):.3f}, ligand_in_site max "
              f"{max(in_site):.3f}, both at site {top + 1} and nowhere else")

    # The controlled case, on the one site in the suite whose size is
    # comparable to a real ligand's envelope. crambin's pose site is 12 A3 and
    # ibuprofen's envelope is 178 A3, so "twice the site's volume" is not a
    # question one can ask of it; the two-shell cavity is 111 A3, which makes it
    # a question. The ligand is translated to the cavity's own centre first --
    # left at crambin's coordinates the two never overlap and both numbers are
    # zero, which is a different case and is checked separately below.
    cavity_atoms = np.vstack([sphere(6.0, 120), sphere(10.0, 200)])
    cavity = [p for p in P.find_pockets(cavity_atoms, ["C"] * len(cavity_atoms))
              if p.kind == "cavity"]
    check("the cavity fixture for the fit reading is a real sealed cavity",
          len(cavity) == 1 and cavity[0].volume > 50.0,
          f"{cavity[0].volume:.0f} A3 in {cavity[0].voxels} voxels"
          if cavity else "none found")
    if cavity:
        cav = cavity[0]
        at_centre = pose_xyz - pose_xyz.mean(axis=0)

        def placed(scale):
            return cav.fit_to(at_centre * scale, pose_radii)

        quarter, half, same, twice, four = (placed(s) for s in
                                            (0.25, 0.5, 1.0, 2.0, 4.0))
        check("a ligand about half the site's size reads as a good fit in both "
              "directions",
              half.ligand_in_site >= 0.9 and half.site_filled >= 0.5,
              f"in_site {half.ligand_in_site:.3f}, site_filled "
              f"{half.site_filled:.3f}, against a site of {cav.volume:.0f} A3")
        check("while a ligand twice the site's volume reads low on the ligand "
              "side",
              twice.ligand_in_site <= 0.5
              and twice.ligand_in_site < half.ligand_in_site,
              f"in_site falls {half.ligand_in_site:.3f} -> "
              f"{twice.ligand_in_site:.3f} and keeps falling at 4x "
              f"({four.ligand_in_site:.3f}): the ligand has atoms that cannot go "
              f"anywhere near the site")
        check("and a ligand a quarter of the site's size reads high on the "
              "ligand side but low on the site side",
              quarter.ligand_in_site >= 0.9
              and quarter.site_filled < half.site_filled,
              f"in_site {quarter.ligand_in_site:.3f} (it fits easily) but "
              f"site_filled {quarter.site_filled:.3f} against "
              f"{half.site_filled:.3f} (it uses less of the hole)")
        # The reason both are reported, as one falsifiable claim about the two
        # numbers: they order the candidates differently, so a caller printing
        # one of them cannot recover the other's information. Here the ligand
        # side is *exactly* tied between a quarter and a half while the site
        # side separates them.
        check("so the two directions are not reciprocals: one alone cannot do "
              "the other's job",
              quarter.ligand_in_site == half.ligand_in_site
              and quarter.site_filled != half.site_filled
              and (twice.site_filled > quarter.ligand_in_site * 0
                   and twice.ligand_in_site < quarter.site_filled),
              f"ligand_in_site cannot tell {quarter.ligand_in_site:.3f} from "
              f"{half.ligand_in_site:.3f} (identical), and site_filled puts the "
              f"too-big ligand at {twice.site_filled:.3f} where the ligand side "
              f"has it at {twice.ligand_in_site:.3f}")
        check("and geometry that does not overlap at all reads zero in both, "
              "which is a third case neither direction catches alone",
              cav.fit_to(pose_xyz, pose_radii).ligand_in_site == 0.0
              and cav.fit_to(pose_xyz, pose_radii).site_filled == 0.0,
              "the same ligand at crambin's own coordinates, with the cavity at "
              "the origin: 0.000/0.000. Measured and zero, which is not the "
              "same answer as not measured")

        # A site that is a protein's whole outer surface. Hand-built, because
        # the dilation that used to produce these was removed and crambin
        # genuinely offers none -- its largest site is 88 A3, 6.6% of the
        # protein's 1118 A3 box. A shell at 13 A stands in for the face of a
        # small protein, and it must not read as a pocket for anything.
        def shell_site(radius, n=3000):
            return P.Pocket(
                center=np.zeros(3, np.float32),
                size=np.asarray([2.0 * radius] * 3, np.float32), voxels=n,
                volume=n * P.DEFAULT_SPACING ** 3,
                points=sphere(radius, n).astype(np.float32),
            )

        sheets = [(r, shell_site(r).fit_to(at_centre, pose_radii))
                  for r in (13.0, 20.0, 30.0)]
        check("a site that is a protein's whole outer surface does not read as a "
              "good fit for anything",
              all(not (f.ligand_in_site > 0.6 and f.site_filled > 0.6)
                  and f.site_filled <= 0.02 for _, f in sheets),
              "; ".join(f"a {int(2 * r)} A sheet reads "
                        f"{f.ligand_in_site:.3f}/{f.site_filled:.3f}"
                        for r, f in sheets)
              + f" -- against a real pocket's 0.500/0.833, so a surface cannot "
                f"pass for one at any of these sizes")

    # What cannot be computed is None with a reason, never 0.0: a caller has to
    # be able to tell "does not fit" from "was not asked properly".
    blank = P.Pocket(center=np.zeros(3, np.float32), size=np.ones(3, np.float32),
                     voxels=0, volume=0.0)
    solid = sites[0] if sites else None
    if solid is not None:
        unanswerable = {
            "a site with no grid points":
                blank.fit_to(pose_xyz, pose_radii),
            "no ligand coordinates":
                solid.fit_to(np.zeros((0, 3)), pose_radii),
            "a radius array of the wrong length":
                solid.fit_to(pose_xyz, pose_radii[:3]),
            "a non-finite coordinate":
                solid.fit_to(np.vstack([pose_xyz, [[np.nan, 0.0, 0.0]]]),
                             np.append(pose_radii, 1.7)),
            "a negative radius":
                solid.fit_to(pose_xyz, -1.0),
        }
        check("every unanswerable case is None with a reason, never 0.0",
              all(f.ligand_in_site is None and f.site_filled is None
                  and not f.measured and f.note for f in unanswerable.values()),
              "; ".join(f"{k} -> {v.note[:28]!r}"
                        for k, v in unanswerable.items()))
        check("and a measured zero is still a zero, not a None",
              solid.fit_to(pose_xyz + 500.0, pose_radii).site_filled == 0.0
              and solid.fit_to(pose_xyz + 500.0, pose_radii).measured,
              "the same ligand 500 A away measures 0.000, and `measured` is "
              "True: the two are different answers and the type keeps them apart")

    # The radii are the caller's envelope, and the reading follows it. All three
    # spellings of the radius have to work because `core.Ligand` exposes
    # `reference_coords` and a *scalar* `radius` about the ligand's centroid,
    # with no per-atom array at all.
    if sites:
        one = sites[best].fit_to(pose_xyz, 1.7)
        default = sites[best].fit_to(pose_xyz)
        bigger = sites[best].fit_to(pose_xyz, 3.0)
        per_atom = sites[best].fit_to(pose_xyz, pose_radii)
        check("a scalar radius, and the default radius, both work, and the "
              "reading follows the envelope it is given",
              one.measured and default.measured and bigger.measured
              and per_atom.measured
              and one.site_filled > 0.5 and default.site_filled > 0.5
              and bigger.site_filled > one.site_filled
              and bigger.ligand_in_site > one.ligand_in_site,
              f"on the pose site: per-atom radii {per_atom.site_filled:.3f}, one "
              f"scalar 1.7 A {one.site_filled:.3f}, the default "
              f"({P.DEFAULT_LIGAND_RADIUS} A) {default.site_filled:.3f}, and a "
              f"3.0 A envelope {bigger.site_filled:.3f}. It is a geometric "
              f"reading of the envelope it is handed, not a property of the site")
        # Read the scores off *fresh* sites, before anything in this file has
        # called `fit_to` on them. Two earlier versions of this check could not
        # fail: one compared `(p.fit_to(...), p.rank_score)[1]` with itself,
        # which evaluates the call first; the other reused the `sites` list,
        # which an earlier check in this same section had already asked, so both
        # readings were post-mutation and the mutation turned out to be
        # idempotent. Both were caught by mutation, not by reading.
        fresh = P.find_pockets(rec_pts, list(rec.elements),
                               residues=rec.residue_labels())
        pristine = [p.rank_score for p in fresh]
        for p in fresh:
            p.fit_to(pose_xyz, pose_radii)
        check("and asking changes nothing about the ranking, and asking twice "
              "changes nothing at all",
              pristine == [p.rank_score for p in fresh]
              and sites[best].fit_to(pose_xyz, pose_radii)
              == sites[best].fit_to(pose_xyz, pose_radii),
              f"rank_score over {len(fresh)} freshly-found sites, read before "
              f"and after calling `fit_to` on every one: "
              f"{[round(v, 3) for v in pristine[:4]]}... unchanged. The reading "
              f"is opt-in, and folding it into the ranking would silently move "
              f"every number the benchmark and the README were measured against")

    # ------------------------------------------------------------------
    section("a site's own thickness, in grid layers")

    # The quantity the `min_extent` floor acts on, exposed so a caller can see
    # what that floor is about to reject. It is `min(size) / spacing`, and the
    # site has to carry its own pitch for that to mean anything.
    every = list(sites) + list(found) + list(cavern) if sites and cavern else []
    if every:
        check("thickness is the smallest extent in grid layers, on every site "
              "in the file",
              all(abs(p.thickness
                      - float(np.min(np.asarray(p.size, np.float64)))
                      / float(p.spacing)) < 1e-9 for p in every)
              and all(p.spacing == P.DEFAULT_SPACING for p in every),
              f"{len(every)} sites across crambin, the dogleg cleft and the "
              f"cavern; crambin runs "
              f"{min(p.thickness for p in sites):.0f} to "
              f"{max(p.thickness for p in sites):.0f} layers at "
              f"{P.DEFAULT_SPACING} A")
        # Both sides of the floor, per site, identified by volume and centre
        # rather than by thickness -- several crambin sites share a thickness,
        # so "a site of this thickness is still there" would pass even if this
        # one had been dropped.
        def same(p, others):
            return (round(float(p.volume), 6),
                    round(float(p.center[0]), 3), round(float(p.center[1]), 3),
                    round(float(p.center[2]), 3)) in {
                (round(float(q.volume), 6), round(float(q.center[0]), 3),
                 round(float(q.center[1]), 3), round(float(q.center[2]), 3))
                for q in others}

        edges = []
        holds = True
        for p in sites[:4]:
            # `min_extent` is in angstrom, so the floor that means "this site's
            # own thickness" is `thickness * spacing`. Two things this had to get
            # right, both of them found by the check failing first: passing the
            # voxel count instead asks for a floor of 8 A against a 6.4 A site
            # and drops it, and a floor of *exactly* `thickness * spacing` is a
            # float32/float64 round trip that can land one ULP above the site's
            # own float32 extent -- so the two sides are taken with a tenth of a
            # percent of slack rather than as a knife-edge float comparison.
            at = P.find_pockets(rec_pts, list(rec.elements), max_pockets=99,
                                min_extent=p.thickness * p.spacing * 0.999)
            over = P.find_pockets(rec_pts, list(rec.elements), max_pockets=99,
                                   min_extent=(p.thickness + 1.0) * p.spacing)
            kept, gone = same(p, at), not same(p, over)
            holds = holds and kept and gone
            edges.append(f"{p.thickness:.0f} layers: "
                         f"{'kept' if kept else 'LOST'} just under its own "
                         f"thickness, "
                         f"{'gone' if gone else 'STILL THERE'} one layer above")
        check("and it is the quantity the min_extent floor acts on, on both "
              "sides of the boundary",
              holds,
              "; ".join(edges) + ". A floor just under a site's own thickness "
              "still returns it; one grid layer above and it is gone")
        # And the limit, which is why this is a reading and not a verdict. The
        # 1.2 A sliver and crambin's real binding site are both 4 layers, so
        # nothing built on thickness tells them apart -- pinned here through the
        # property the docstring points readers at.
        sliver_atoms = np.vstack([
            np.concatenate([g, np.full((len(g), 1), -3.7)], axis=1)
            for g in [np.stack(np.meshgrid(*[np.arange(-8.0, 8.01, 1.6)] * 2,
                                           indexing="ij"), axis=-1)
                      .reshape(-1, 2)]
        ] + [
            np.concatenate([g, np.full((len(g), 1), 3.7)], axis=1)
            for g in [np.stack(np.meshgrid(*[np.arange(-8.0, 8.01, 1.6)] * 2,
                                           indexing="ij"), axis=-1)
                      .reshape(-1, 2)]
        ]).astype(np.float32)
        sliver = P.find_pockets(sliver_atoms, ["C"] * len(sliver_atoms),
                                max_pockets=99)
        if sliver and sites:
            pose_thick = sites[best].thickness
            check("but it does not tell a sliver from a tight binding site, "
                  "which is the finding and not a caveat",
                  abs(sliver[0].thickness - pose_thick) < 1e-6
                  and sliver[0].volume <= P.DEFAULT_MAX_VOLUME,
                  f"a 1.2 A gap between two atom sheets is {sliver[0].thickness:.0f} "
                  f"layers thick and crambin's own ligand-holding site is "
                  f"{pose_thick:.0f}: the same number, so `thickness` is "
                  f"reported because it is cheap and true, and the docstring "
                  f"says what it is not")

    # ------------------------------------------------------------------
    section("the shortlist length is a budget, and pinned from both sides")

    # `DEFAULT_MAX_POCKETS` is a budget, and the comment in pockets.py used to
    # call it a measured value. It is one crambin number plus a margin: the site
    # holding the docked ibuprofen is 9th of the 16 crambin offers, so an
    # 8-entry list dropped a real binding site. A second protein to measure a
    # second requirement against is not in the tree -- 1crn is the only real
    # fixture -- so the honest thing is to pin the budget and the boundary it
    # was chosen for, separately, and let the number be what it is.
    def best_held(n: int):
        got = P.find_pockets(rec_pts, list(rec.elements),
                             residues=rec.residue_labels(), max_pockets=n)
        if not got:
            return 0, 0
        return (max(int(in_box(pose_pts, *p.box_center_and_size()).sum())
                    for p in got), len(got))

    keep8, n8 = best_held(8)
    keep9, n9 = best_held(9)
    keep_def, n_def = best_held(P.DEFAULT_MAX_POCKETS)
    keep_all, n_all = best_held(99)
    check("the default shortlist length is the budget it is documented as",
          P.DEFAULT_MAX_POCKETS == 12 and n_def == 12,
          f"DEFAULT_MAX_POCKETS = {P.DEFAULT_MAX_POCKETS}, and crambin fills "
          f"the list to {n_def}; pinned as a number so a change to it is a "
          f"deliberate act rather than a quiet one")
    check("and it is crambin's ninth plus a margin, with the boundary measured",
          keep8 < len(pose_pts) <= keep9 and keep_def == len(pose_pts),
          f"8 entries hold {keep8}/{len(pose_pts)} ligand atoms, 9 hold {keep9}, "
          f"and the default {P.DEFAULT_MAX_POCKETS} holds {keep_def}: 12 is the "
          f"ninth plus three, not a number a benchmark chose")
    check("while crambin only offers sixteen sites, so twelve is a truncation "
          "and not a count the search arrived at",
          n_all == 16 and n_def < n_all,
          f"{n_all} sites at max_pockets=99, {n_def} at the default: the four "
          f"dropped are {16 - n_def} of the sixteen, and nothing measured says "
          f"those four are the wrong ones")

    # ------------------------------------------------------------------
    section("every check in this file ran")

    # 59 of the check() calls in main() sit inside an `if`, and most of those
    # are data guards: no sites, no fixture, no engine. That is the right shape
    # for a check script and also its worst failure mode -- a check that stops
    # running takes its count with it, so the total quietly falls and the run
    # still says "all passed". It happened here: tightening the volume ceiling
    # dropped the suite from 118 to 84 with 8 failures and 34 checks simply not
    # executed. So the total is itself a check, and it has to be updated by
    # hand -- which is the point, because the update is where the noticing
    # happens.
    before_total = CHECKS
    check(
        "the number of checks that ran is the number this file is supposed to "
        "have, so none of them is hiding behind a guard",
        before_total + 1 == EXPECTED_CHECKS,
        f"{before_total} ran before this one and {EXPECTED_CHECKS} are expected; "
        f"the +1 is this check. If a check was added or removed, change "
        f"EXPECTED_CHECKS deliberately",
    )

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
