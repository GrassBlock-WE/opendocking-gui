"""Where the cavities are, so the search box does not have to be guessed.

The workbench filled its box centre with the receptor's centroid. That is
almost never a binding site: for a globular protein the centroid sits in the
middle of a dense core, so the box lands on solid protein, and a search there
either finds nothing or returns poses that the engine itself reports as
implausible. `odcli` refuses to guess at all for exactly this reason -- its own
help text calls a whole-protein box "useless as a search region".

This finds the buried cavities instead, and offers them as a list.

# What a pocket is here

Lattice-style pocket detection, in the family of LIGSITE:

1. lay a grid over the receptor, inflated by a margin;
2. mark every grid point within ``vdw + probe`` of any atom as solid;
3. for every remaining point, count the axes that have protein on *both*
   sides within a few ångström -- LIGSITE's "protein-solvent-protein" events;
4. flood-fill from the grid boundary through the remaining points -- anything
   the fill reaches is open to solvent from outside, and what it cannot reach
   is sealed.

The probe radius is a water molecule by default (1.4 Å). It is the one number
that decides what counts as a hole, and it is a parameter because a different
one finds different holes: 1.4 misses the narrow clefts a ligand can enter, and
1.0 turns surface grooves into pockets that are really just dimples.

# Measured, by putting a known ligand back where it came from

The only measurement worth anything here is one with a known answer, and the
answer is "where did this ligand actually sit in the crystal structure". The
protein with the ligand's residue removed, the search run without ever being
told where to look, and each site's box tested for whether it holds the whole
bound pose:

| complex              | ligand       | atoms | rank of that site | of | its volume | atoms its box holds |
|----------------------|--------------|-------|-------------------|---|------------|--------------------|
| 1STP streptavidin    | biotin       | 16    | **1**             | 40 | 161 A³     | 16 / 16           |
| 3PTB trypsin         | benzamidine  | 9     | **1**             | 39 | 357 A³     | 9 / 9             |
| 2NNQ                 | T4B          | 36    | **1**             | 41 | 359 A³, sealed | 36 / 36        |
| 1HVR HIV protease   | XK2          | 46    | **1**             | 47 | 544 A³     | 43 / 46           |
| 1CRN crambin        | ibuprofen    | 16    | 9                 | 16 | 12 A³      | 16 / 16           |

Four first and one ninth, five out of five found. Four of the five boxes hold
**every** atom of the bound pose; the fifth holds 43 of 46, and the site is a
cleft that runs past where the inhibitor's flexible tail ends.

That is what the search is for and it is worth saying plainly, because the
version of this file that got there had a one-voxel dilation in front of the
labelling step. It merged the three open sites -- 1STP, 3PTB and 1HVR, the ones
this module calls grooves -- into the protein's outer surface: they were not
mis-ranked, they were **not in the list at all**. (2NNQ's site is a *sealed*
cavity and was never dilated, which is why it survived and the others did not.)
See the comment on `interior` in `find_pockets` for the before-and-after voxel
counts.

What it still does not do, measured rather than assumed:

* **crambin is the weak case and it is weak for a reason worth knowing.**
  Its ibuprofen site is 12 A³ and ranks ninth. The search measures the space
  a ligand **leaves**, not the space it occupies: where a ligand fits snugly
  that space is nearly nothing, and where the pocket is roomier than the
  ligand -- the usual case, and all four complexes above -- there is plenty.
  An earlier version of this file appeared to rank it second, and that was
  luck: it ranked second only because an unrelated ceiling happened to delete
  three larger lumps of surface first. The ceiling is still here. It is
  simply no longer doing that job by accident.
* a site is judged by the box it builds, not by how near its centre is. The
  trypsin site in 3PTB is 12.8 Å from the benzamidine's centre of mass and
  still holds all nine of its atoms, because the site is a 31 x 18 x 33 Å
  cleft and its centroid is nowhere near where the ligand sits. Ranking or
  filtering on centre distance would throw that one away.
* crambin has **no sealed cavity at all** at a 1.4 Å probe; every site it
  offers is an open groove. A cavity-only detector would report a protein
  with an obvious binding site as having nowhere to dock.
* two independent computations agreeing on one residue is worth more than
  either alone. The crambin site that holds the docked ibuprofen is lined by
  **ARG 17, THR 2, PHE 13, ARG 10, ASN 14 and GLU 23**, and an independent
  count of every receptor atom within 4.5 Å of that same pose names seven
  residues, six of which are exactly those.

# When the site is bigger than a good search box

`box_center_and_size` reports a site's full extent, so a site that is a long
winding cleft gets the large box it genuinely is. Measured on the two
complexes where that is the problem, from `scripts/redock_benchmark.py`:

| site                        | site extent, so its box at 4 A padding | RMSD at its effort |
|-----------------------------|-------------------------------------|--------------------|
| 3PTB benzamidine, rank 1 of 39 | 31 x 18 x 33 A -> **39 x 26 x 41 A**, 41,574 A3 | 1.26 A at 64, 12.88 A at 16 |
| 1HVR HIV protease XK2, rank 1 of 47 | 22 x 30 x 24 A -> **30 x 38 x 32 A**, 36,480 A3 | 21.24 A at 64, 10.53 A at 128 |

The 3PTB cleft's extent was written here as 22 x 17 x 28 A for as long as this
file existed, and it was **wrong**. At 4 A padding a 22 x 17 x 28 A site produces
a 30 x 25 x 36 A box, not the 39 x 26 x 41 A the benchmark reports -- and no
uniform padding reconciles them, since the three axes would need 8.5, 4.5 and
6.5 A, so it was not a padding-convention difference either. The real extent is
**31 x 18 x 33 A**, and it follows from the box by the one rule in the table
above: the box is the site's size plus twice the padding. The benchmark calls
`box_center_and_size` with no padding argument, so it prints the raw padded
box, and an independent rerun reproduced 39 x 26 x 41 A, rank 1 of 39, 9 of 9
bound-ligand atoms inside, 12.88 A RMSD at exhaustiveness 16 and 1.26 A at 64.

Worth saying why that survived so long, because the cause is not carelessness.
**Nothing in the project recomputes one of these numbers from the other.** The
benchmark prints a box, the docstring quotes an extent, and the two are related
by a rule that only a reader knows, so a wrong number in prose can sit in a
release next to a right number in a table and nothing notices -- the check
script can only assert what the code computes, never what a comment claims. So
the arithmetic now has somewhere to be wrong *instead of* hiding in prose:
`pockets_check.py` asserts `box == size + 2 * DEFAULT_PADDING` on every axis,
for the real sites and for the synthetic fixtures. That cannot catch a wrong
docstring. It pins the relationship the docstrings are derived from, which is
the part that was actually load-bearing.

41,574 A3 is not a binding site, it is most of a small protein, and the
interface puts it in three spin boxes with nothing to say that anything is
unusual. So `box_with_budget` is the other door into the same
geometry: ask for a ceiling and get the box, whether or not the ceiling bit,
plus a note that names what was capped and what the box now leaves out. A
caller that does not ask keeps the uncapped 2-tuple, unchanged, because an
unasked-for cap would be the same silent substitution this exists to stop.

Three things that measurement settles, and that a cleverer formula would not:

* **A cap is not a fix.** 3PTB's own box does find the pose, but only at
  exhaustiveness 64, and 1HVR's does not find it at 64 or 128. There is no
  single box that both holds a nine-atom ligand somewhere along a winding
  28 A cleft and stays small. That is a limit of one box per site, every
  pocket finder has it, and the useful thing to do about it is say so.
* **A cap does not have to move the centre.** The box stays on the site's own
  centroid. The bounding-box middle of a curved cleft lands in the wall, and
  that is the defect the `box_center_and_size` docstring already records as
  fixed; re-centring on a cap would bring it back.
* **A box does not even always contain its own site.** Because that centre is
  the centroid and not the middle of the bounds, a site that curves back on
  itself sticks out. On the synthetic dogleg cleft in the checks the default
  4 A padded box holds 1937 of 1985 of the site's own points, 97.6%, and at
  zero padding 1649 of them, 83.1%. So `BudgetedBox.coverage` counts points
  rather than trusting the size, and a note is produced for a box that leaks
  even when no budget was applied -- an empty note means the box is the site's
  own *and* holds all of it.

# Asking a site whether it fits a particular ligand

The shortlist is ranked by the space a ligand **leaves**, so a site that fits
its ligand snugly is a site with almost no free space, and it lands late. On
crambin with ibuprofen docked into it, the site that holds all sixteen ligand
atoms is **ninth of 12** and **below the median volume** of the list. That is
the same position the table at the top of this file calls ninth of sixteen, and
it has to be: `find_pockets` sorts once and returns `found[:max_pockets]`, so a
truncated list is a *prefix* of the long one and truncation cannot move an
occupant. Measured on the shipped fixture, the site is the only one of the
sixteen that holds all sixteen ligand atoms, and it sits at position **9 of 12**,
at **9 of 16** untruncated, and at **9 of 16** with the volume ceiling removed --
`rank_score` 2.4037, between 2.4235 at position 8 and 2.3595 at position 10, and
the first twelve `rank_score`s of the short and long lists are identical. (An
earlier version of this paragraph said *8th of 12*, which no truncation of this
list can produce; the position has always been ninth.) That is not
a bug that a better coefficient would fix: the quantity being ranked does not
mention the ligand. Retuning was tried and could not move it -- three mutations
of `rank_score`, two parameter changes, a 42-cell sweep of `max_volume` x
`min_burial` x `min_voxels`, and fifteen alternative ranking formulas all left
that site between 8th and 11th of 12. **The rank is stable because the ranking
cannot see the ligand.**

So the caller supplies one. `Pocket.fit_to(coords, radii)` reports two
fractions, and the point of reporting two is that **they are not reciprocals**:

| on crambin's pose site (12 A3, 24 voxels), ibuprofen = 16 atoms, 178 A3 of envelope | `ligand_in_site` | `site_filled` |
|---|---|---|
| the real ligand | **0.50** | **0.83** |
| 2x the volume | 0.31 | 0.79 |
| 8x the volume | 0.19 | 0.58 |
| half the size | 0.63 | 0.71 |
| a tenth the size | 0.88 | 0.21 |

Too big for the site and the left column falls while the right one holds.
Dwarfed by the site and the right column falls while the left one holds. A real
fit has both up. Print either alone and one of those two failures disappears:
a ligand twice the site's volume still fills the site completely, and a
ligand a tenth the size still sits entirely inside it.

On the same protein the reading is close to binary in practice -- the pose site
reads 0.50/0.83 and **all eleven other sites read exactly 0.00/0.00** -- so it
is a way of *rejecting* the rows that obviously cannot hold the ligand, not a
way of ranking the ones that might.

**And 0.50 is not "half a fit".** The pose site is 24 voxels, 12 A³ of them,
inside a 41 A³ box: the site is a sparse sample of the free space the ligand
will actually use, so an atom has to be within its own radius of one of just 24
points to count, and half of a 16-atom ligand is not. Both numbers are for
**comparing sites against each other or ligands against each other**, not an
absolute verdict -- a ligand fifteen times the site's volume reads 0.50 here,
and that is the sample talking, not the fit.

It is geometric compatibility: no field, no desolvation, no torsions. It is not
a score, and it is deliberately **not** folded into `rank_score`, because that
would silently move the ranking the benchmark and the README numbers were
measured against. A ligand can fit a hole's shape perfectly and still be
undockable there: 1HVR's XK2 sits in its own site and that site still reads
21.24 A RMSD.

`Pocket.thickness` is the other new reading: the site's own thickness in grid
layers, the quantity the `min_extent` floor acts on, so a caller can see what
that floor is about to reject. It runs 2 to 8 layers on crambin. It **does not
distinguish a sliver from a tight binding site**, and that is measured rather
than assumed: a 1.2 Å gap between two atom sheets is reported 4 layers thick and
so is crambin's real binding site, because at 0.8 Å spacing those are the same
four layers.

# What this is not

It does not find the binding site. It finds **enclosed space**, ranked, with
the residues lining each part. A ligand can sit on a flat surface that scores
zero and this will never mention it. The largest site is not always the one a
particular ligand wants -- a small, complementary pocket can be worth more than
a big solvent-filled one. It is a shortlist with geometry attached, not an
answer, and the caller is expected to let the user pick.

Every pocket also reports its **lining residues**: receptor residues with an
atom within ``lining_max`` of any pocket point. That is what turns "there is a
cavity near (12, 4, -3)" into something a person can check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Pocket",
    "find_pockets",
    "VDW_RADII",
    "DEFAULT_PROBE",
    "DEFAULT_SPACING",
    "DEFAULT_PADDING",
    "DEFAULT_MAX_VOLUME",
    "DEFAULT_MAX_POCKETS",
    "KIND_LABELS",
    "kind_label",
    "cavity_sensitivity",
    "DEFAULT_PROBE_SWEEP",
    "BudgetedBox",
    "SiteFit",
    "DEFAULT_BOX_MAX_SIDE",
    "DEFAULT_LIGAND_RADIUS",
]

#: Van der Waals radii, ångström. Pockets are about where a solvent can and
#: cannot reach, which is a van der Waals question; the covalent radii used for
#: bond auditing are roughly half these and would make every protein a solid
#: block with no interior at all.
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Mg": 1.73, "Zn": 1.39, "Ca": 2.31, "Fe": 2.04, "Mn": 2.05,
    "Du": 2.00,
}

#: Water's radius. A point the solvent can occupy is a point a ligand can too.
DEFAULT_PROBE = 1.4
#: Grid pitch, Å. 0.8 keeps a 3000-atom protein under half a million points
#: while still resolving a 3 Å pocket; much finer and the cost grows cubically
#: for detail the pocket ranking does not use.
DEFAULT_SPACING = 0.8
#: Å of space added around a cavity so a ligand is not confined to its walls.
DEFAULT_PADDING = 4.0
#: A cavity smaller than this many grid points is noise between two atoms.
MIN_VOXELS = 8
#: Distance from a pocket point to a lining residue's atom, Å.
LINING_MAX = 4.5

#: How each kind is named in the interface. The table used to say "groove"
#: while the status bar said "burial" for the same site, which is two words for
#: one thing and reads as two different things. One table, both callers.
KIND_LABELS = {"cavity": "sealed", "burial": "groove"}


def kind_label(kind: str) -> str:
    """`"sealed"` or `"groove"`, for anything that shows a site to a user."""
    return KIND_LABELS.get(kind, kind)


#: Probe radii, Å, that :func:`cavity_sensitivity` tries.
DEFAULT_PROBE_SWEEP = (1.4, 1.1, 0.9, 0.7, 0.5)


def cavity_sensitivity(
    coords,
    elements,
    probes=DEFAULT_PROBE_SWEEP,
    **kwargs,
):
    """``[(probe, n_sealed), ...]`` — how many sealed cavities each probe finds.

    This exists because "no sealed cavity" is not a useful thing to say on its
    own, and for a good reason. T4 lysozyme L99A (1L96) has a cavity
    *deliberately engineered* into it by replacing a bulky residue with
    alanine, and at the default 1.4 A probe this search reports **zero** sealed
    cavities in it. The cavity is roughly 100 A^3, and a 1.4 A probe inflates
    every atom by that much, so the inflated surface simply fills it in.

    The cavity is not missed because the search is bad at small cavities. It is
    missed because the probe is blunt, and a blunt probe is the right default:
    a 0.5 A probe closes up surface grooves into dozens of "sealed" pockets,
    which is a worse answer than none. The honest response to a null result is
    not to lower the default quietly, but to say what the answer would be at a
    probe that sees it — so the user can choose to look.

    Only the flood fill runs per probe; the grid is built once at the smallest
    radius asked for, because shrinking an already-built solid mask is not the
    same operation as building it and would give a different answer.
    """
    pts = np.asarray(coords, np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return [(float(p), 0) for p in probes]
    els = list(elements) if elements is not None else ["C"] * len(pts)
    els = els + ["C"] * max(0, len(pts) - len(els))
    radii = np.asarray([VDW_RADII.get(e, 1.70) for e in els], np.float32)
    spacing = kwargs.get("spacing", DEFAULT_SPACING)
    margin = kwargs.get("margin", 6.0)
    lo, shape = _grid_for(pts, spacing, margin)
    window = max(int(round(3.0 / spacing)), 1)

    out: list[tuple[float, int]] = []
    for probe in probes:
        solid = _splat_solid(pts, radii + np.float32(probe), lo, shape, spacing)
        free = ~solid
        sealed = free & ~_flood_from_border(free)
        if not sealed.any():
            out.append((float(probe), 0))
            continue
        count = 0
        for comp in _components(sealed):
            extent = (comp.max(axis=0) - comp.min(axis=0) + 1) * spacing
            if len(comp) >= kwargs.get("min_voxels", MIN_VOXELS) and \
                    float(extent.min()) >= kwargs.get("min_extent", 2.0 * spacing):
                count += 1
        out.append((float(probe), count))
    return out


#: Ceiling on a site's volume, Å³. A drug binding pocket runs a few hundred
#: cubic ångström; anything much past that is a merged surface rather than a
#: site. A heuristic, and labelled as one -- see `max_volume`.
DEFAULT_MAX_VOLUME = 1500.0

#: How many sites a search returns. **A budget, not a measurement** -- and the
#: comment this replaces said "for a measured reason", which was one number
#: wearing a general claim.
#:
#: The one fact behind it: on crambin with ibuprofen docked into it, the site
#: the ligand actually occupies ranks **ninth of sixteen** and the box it
#: builds holds all sixteen ligand atoms, so an eight-entry shortlist cut off
#: a real binding site. Twelve is that ninth plus a margin of three. That is
#: arithmetic on a single data point, and it is stated as such here so nobody
#: later cites twelve as though a benchmark chose it.
#:
#: What stops it being arbitrary is that it is pinned from both sides in
#: `pockets_check.py`: the default is exactly twelve, and the crambin site
#: that holds the ligand really is inside that default. A second local
#: structure to measure a second requirement against is **not available** --
#: `examples/1crn_prep.pdbqt` is the only real protein fixture in the tree, and
#: the other four measured complexes (1STP, 3PTB, 2NNQ, 1HVR) all put their
#: own ligand's site first, so they raise no lower bound. Widening the list is
#: cheap and mostly noise; the crambin case is the only one in hand that shows
#: a real cost for being too short.
DEFAULT_MAX_POCKETS = 12

#: Radius used for a ligand atom when the caller supplies coordinates but no
#: radii, Å. Carbon's, because carbon is most of a drug-like ligand and because
#: it is the same number :data:`VDW_RADII` uses for ``"C"``. A caller with
#: elements can build the per-atom array and do better.
DEFAULT_LIGAND_RADIUS = 1.7

#: Longest side, Å, of a search box :meth:`Pocket.box_with_budget` will hand
#: out unless it is told otherwise. A parameter, picked from the two boxes
#: that were actually measured rather than from taste.
#:
#: The 3PTB benzamidine site's box is **39 x 26 x 41 A** -- 41,574 A3, most of
#: a small protein -- and the 1HVR XK2 site's is **30 x 38 x 32 A**, 36,480 A3.
#: Those are the two longest sides in the measured set. The largest box on
#: crambin, the one real protein the checks here can load without a network, is
#: **14.4 x 18.4 x 24.8 A**, so a 30 A cap leaves every crambin site exactly as
#: it was: 24.8 A is the longest side anywhere on that protein.
#:
#: What the cap buys is *not* a successful redock, and is not claimed to be
#: one. Measured by `scripts/redock_benchmark.py`: the 3PTB box finds the pose
#: at **1.26 A RMSD**, but only at exhaustiveness 64 -- 12.88 A at 16, two
#: seconds either way -- and the 1HVR box does not find it at any effort
#: tested, **21.24 A at 64** and **10.53 A at 128**. There is no single box
#: that both holds a nine-atom ligand somewhere along a winding 28 A cleft and
#: stays small. That is a limit of one box per site and every pocket finder
#: shares it, so the honest response is to say so instead of picking a cleverer
#: formula that appears to solve it. What the cap does buy is a search region
#: whose size the caller chose rather than one they inherited, and a note
#: saying what it cost.
DEFAULT_BOX_MAX_SIDE = 30.0


@dataclass
class Pocket:
    """One candidate binding site, with enough geometry to place a box on it."""

    center: np.ndarray
    #: Bounding extent of the site itself, before padding.
    size: np.ndarray
    #: Grid points in the site; a volume proxy, not a physical volume.
    voxels: int
    #: ``"cavity"`` when it is sealed, ``"burial"`` when it is an open groove.
    kind: str = "burial"
    #: Volume of the site, Å³: ``voxels * spacing ** 3``.
    #:
    #: Carried because the *bounding box* is not a usable stand-in and using
    #: one was a real defect. A winding cleft is a long thin tube whose box
    #: is enormous and whose interior is not: the HIV protease site in 1HVR
    #: has a bounding box of 8008 A³ and 544 A³ of actual space, so a
    #: ceiling applied to the box throws away a real binding pocket, while
    #: a large flat patch of surface has a modest box and a lot of voxels.
    #: The two measures disagree in opposite directions and only the voxel
    #: count is the thing being described.
    volume: float = 0.0
    #: Mean burial score over the site, 0-3. Three means protein on all sides.
    burial: float = 0.0
    #: ``(residue, n_atoms_within_lining_max)`` pairs, most-contacted first.
    lining: list[tuple[str, int]] = field(default_factory=list)
    #: The grid points themselves, in world coordinates, ``(voxels, 3)``.
    #:
    #: Carried so the site can be *looked at* rather than only read about. A
    #: table saying "groove at (12.8, 7.2, 0.1), 5.6 x 7.2 x 5.6 A" is a
    #: claim; a translucent cloud sitting in the crevice between the atoms the
    #: table names is the reader's own check on it, and there is no way to get
    #: that check from the numbers alone. Bounded by `max_pockets` and
    #: `min_voxels`, so a few thousand points at most.
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    #: Grid pitch this site was found on, Å. Carried because two of the readings
    #: below are only meaningful against the grid that produced them -- `volume`
    #: is `voxels * spacing ** 3` and `thickness` is a count of grid layers -- and
    #: a caller who searched at a finer pitch than the default must not be told
    #: about it in the wrong units.
    spacing: float = DEFAULT_SPACING

    @property
    def rank_score(self) -> float:
        """Ranking key: size, tempered by how enclosed the site is.

        Volume alone ranks a flooded cavern above a tight, well-formed pocket.
        Taking the cube root keeps an 8x larger cavity from outscoring a pocket
        whose *shape* fits better, and multiplying by the burial score stops a
        shallow surface scrape from outranking a real cavity of the same size.

        The cube root of the volume, so it is a length in ångström: site 1 is
        ranked above site 2 because a ligand-shaped cavity beats a
        cavern-shaped one of the same depth, and the number itself is then
        comparable with the size of the box drawn on it.
        """
        return self.volume ** (1.0 / 3.0) * (0.5 + 0.5 * self.burial)

    def box_center_and_size(self, padding: float = DEFAULT_PADDING):
        """``(center, size)`` for a `GridBox` around this site.

        The centre is the site's own centroid, not its bounding-box middle: for
        a curved cleft those differ by several ångström, and the bounding box's
        middle can land in the wall.

        Deliberately uncapped. `size` is the site's full extent, so a winding
        cleft gets the large box it genuinely has -- 39 x 26 x 41 A for the 3PTB
        site -- and that is a fact about the site rather than a defect, so
        nothing here changes it and the return type stays a plain 2-tuple for
        every existing caller. `box_with_budget` is the same box with a size
        budget applied and an honest note about what the budget cost; it is a
        separate call so a caller cannot ask for a capped box and quietly lose
        the note.
        """
        size = np.asarray(self.size, np.float32) + np.float32(2.0 * padding)
        return tuple(float(v) for v in self.center), tuple(float(v) for v in size)

    def box_with_budget(
        self,
        padding: float = DEFAULT_PADDING,
        *,
        max_side: float | None = DEFAULT_BOX_MAX_SIDE,
        max_volume: float | None = None,
    ) -> BudgetedBox:
        """The site's box, capped to a size budget, with what that cost said.

        `box_center_and_size` cannot answer this. A site that is bigger than a
        sensible search box has no small box, only a large one or a wrong one,
        and a caller who is handed the large one has no way to tell it apart
        from the large box of a genuinely large pocket. So the budget is asked
        for explicitly, and the answer carries the bill:

        * ``size`` is the box to use, no longer than `max_side` on any axis and
          no larger than `max_volume` in total;
        * ``capped`` says whether either ceiling actually bit;
        * ``coverage`` is the fraction of the site's **own grid points** the
          resulting box still contains -- the honest version of "does this box
          still contain the site", since the site is a set of points and not a
          box;
        * ``note`` is a sentence for a person naming what was capped, what the
          site asked for, where the box still is, and what is now outside it.

        Not a tuple, and not iterable. `centre, size = pocket.box_with_budget()`
        is meant to be a mistake: the note is the whole point of the call and
        unpacking past it discards the only part that knows the box is
        compromised.

        **What the budget means.** `max_side` is a ceiling on the longest side,
        applied per axis. That is the knob a person can check by hand, because
        the interface shows three side lengths and nothing else. `max_volume` is
        a ceiling on the box's volume, applied **after** the side cap and as a
        **uniform scale** on all three axes, so the box keeps the site's
        proportions instead of being squeezed into a different shape that looks
        like the site and is not. Neither subsumes the other: a 30 x 30 x 30 A
        box is within the side cap and 27,000 A3 over a 20,000 A3 volume
        budget, while an 8 x 8 x 70 A box is 4480 A3 and still 70 A long. Both
        shapes occur in the measured set -- the 3PTB box is caught by either
        ceiling on its own -- so both exist, and the side cap is the default
        because a volume cap that fires on a legal 30 A cube is a rule a caller
        cannot predict.

        **The centre never moves.** The box stays on the site's own centroid
        even when the budget shrinks it, because the bounding-box middle of a
        curved cleft can land in the wall and that defect is already fixed
        above; re-centring on a cap would reintroduce it. A cap that moved the
        centre would have to say so, and this one does not have to, so it says
        instead that the centre is where it was.

        **What a budget does not do.** It does not make docking work. For 3PTB
        the site's own box does find the pose (1.26 A RMSD) but only at
        exhaustiveness 64; for 1HVR it does not (21.24 A at 64, 10.53 A at
        128). Capping makes the search cheaper and the box defensible. It does
        not make a winding 28 A cleft containable, and `note` says exactly how
        much of the site is now out of reach rather than implying the rest is
        safe.
        """
        centre, requested = self.box_center_and_size(padding)
        # float64 so the cap arithmetic does not inherit the float32 grid's
        # rounding. It is exact: the float32 values widen without changing
        # value, so an uncapped box is the *same* floats, not nearly the same.
        size = np.asarray(requested, np.float64).copy()
        spent: list[str] = []

        if max_side is not None and float(max_side) > 0.0:
            if bool((size > float(max_side)).any()):
                size = np.minimum(size, float(max_side))
                spent.append(f"a {float(max_side):g} A side budget")
        if max_volume is not None and float(max_volume) > 0.0:
            now = float(np.prod(size))
            if now > float(max_volume):
                size = size * (float(max_volume) / now) ** (1.0 / 3.0)
                spent.append(f"a {float(max_volume):.0f} A3 volume budget")

        kept, total = self._points_inside(centre, size)
        return BudgetedBox(
            center=centre,
            size=tuple(float(v) for v in size),
            requested_size=requested,
            capped=bool(spent),
            coverage=(kept / total) if total else None,
            note=_budget_note(self.size, requested, size, spent, kept, total),
        )

    @property
    def thickness(self) -> float:
        """How many grid layers thick this site is at its thinnest, in voxels.

        The smallest of the three extents divided by the grid pitch -- the same
        quantity the `min_extent` floor acts on, so a caller can see what that
        floor is about to reject instead of guessing. On crambin the sites run
        from 2 layers to 8.

        **It does not tell a sliver from a tight binding site, and that is a
        measured result rather than a caveat.** Two atom sheets with a 1.2 Å
        gap between them are reported as a site **4 voxels** thick, and so is
        crambin's own site that holds the docked ibuprofen: at 0.8 Å spacing a
        1.2 Å gap and a 3.2 Å gap are the same four layers, and nothing built on
        thickness can separate them. Eroding the free space does not help
        either, and points the wrong way -- the 1.2 Å gap survives two erosions
        with 128 voxels, the real site with **1**. A number that cannot tell
        those two apart is reported here because it is cheap and true, and the
        docstring says what it is not rather than letting a user read it as a
        verdict on whether the site is real.
        """
        spacing = float(self.spacing) or DEFAULT_SPACING
        return float(np.min(np.asarray(self.size, np.float64))) / spacing

    def fit_to(self, coords, radii=None) -> SiteFit:
        """How well this site's free space fits a ligand's envelope.

        **Geometric compatibility only.** Not affinity, not a score, not a
        ranking: it says nothing about electrostatics, desolvation or torsions,
        and `rank_score` is deliberately left alone, because folding this into
        it would silently move the ranking the benchmark and the README numbers
        were measured against. It is an additional, opt-in reading of one site,
        like `BudgetedBox.coverage`.

        The motivation is measured. The search ranks by ``volume ** (1/3) *
        burial`` -- the space a ligand **leaves** -- so a site that fits its
        ligand snugly is a site with almost no free space and lands late: on
        crambin the site holding the docked ibuprofen is **ninth of 12** and
        below the median volume -- the same ninth the module docstring reports
        as ninth of sixteen, since the shortlist is a prefix of the full list
        rather than a re-ranking of it, and measured as 9 of 12, 9 of 16 and 9
        of 16 with the ceiling removed on the shipped fixture -- and no
        retuning of the formula could move it (3
        ranking mutations, 2 parameter changes, a 42-cell parameter sweep and 15
        alternative formulas all left it between 8th and 11th). The rank is
        stable because the ranking **cannot see the ligand**. These two
        fractions can, because the caller supplies one.

        **Two fractions, and the point is that they are not reciprocals.** Each
        one hides a failure that the other catches:

        * ``ligand_in_site`` -- fraction of the ligand's atoms whose own body
          reaches this site's free space. Low means **the ligand is too big for
          the site**: it has atoms that cannot go anywhere near it.
        * ``site_filled`` -- fraction of the site's voxels inside the ligand's
          envelope. Low means **the site is bigger than the ligand**: the hole is
          mostly space the ligand will never use.

        Measured on crambin's pose site (12 Å³ of voxels, 24 of them) with
        ibuprofen, a 16-atom ligand whose envelope is 178 Å³:

        | ligand          | envelope | `ligand_in_site` | `site_filled` |
        |-----------------|----------|------------------|---------------|
        | the real one    | 178 A³   | **0.50**         | **0.83**      |
        | 2x the volume   | 215 A³   | 0.31             | 0.79          |
        | 8x the volume   | 288 A³   | 0.19             | 0.58          |
        | half the size   | 89 A³    | 0.63             | 0.71          |
        | a tenth         | 30 A³    | 0.88             | 0.21          |

        Too big and the left column falls while the right one holds; dwarfed and
        the right column falls while the left one holds; a real fit has both up.
        Printing either one alone is how a failure would be hidden.

        Two limits, both measured. The reading is **close to binary in
        practice**: on crambin the pose site reads 0.50/0.83 and all eleven
        other sites read exactly 0.00/0.00, so this rejects the rows that
        obviously cannot hold the ligand rather than ranking the ones that might.
        And **0.50 is not "half a fit"** -- the pose site is 24 voxels, 12 A³ of
        them, inside a 41 A³ box, so the site is a sparse sample of the free
        space the ligand will actually use, and an atom has to be within its own
        radius of one of just 24 points to count. A ligand fifteen times the
        site's volume reads 0.50 here. Both numbers are for comparing sites
        against each other or ligands against each other, not an absolute
        verdict.

        `coords` is ``(n, 3)``. `radii` is a per-atom radius, or a single number
        for all atoms, or ``None`` for :data:`DEFAULT_LIGAND_RADIUS`. Note that
        `opendocking.core.Ligand` exposes ``reference_coords`` and a **scalar**
        ``radius`` about the ligand's centroid -- there is no per-atom radius
        there -- so a caller with a `Ligand` can pass both, and a caller with
        elements can do better by building the per-atom array out of
        :data:`VDW_RADII`. That is why `core` is not imported here.

        Returns ``None`` in either fraction when the question cannot be answered
        -- no ligand, no site voxels, a radius array of the wrong length, a
        negative radius, a non-finite coordinate -- rather than ``0.0``, so a
        caller can tell "does not fit" from "was not measured".
        """
        pts = np.asarray(self.points, np.float32).reshape(-1, 3)
        lig = np.asarray(coords, np.float64).reshape(-1, 3)
        if len(pts) == 0:
            return SiteFit(None, None,
                           "this site carries no grid points, so its free space "
                           "cannot be compared with a ligand")
        if len(lig) == 0:
            return SiteFit(None, None, "no ligand coordinates were supplied")
        if radii is None:
            rad = np.full(len(lig), DEFAULT_LIGAND_RADIUS, np.float64)
        else:
            rad = np.asarray(radii, np.float64).reshape(-1)
            if rad.size == 1:
                rad = np.full(len(lig), float(rad[0]), np.float64)
        if len(rad) != len(lig):
            return SiteFit(None, None,
                           f"{len(rad)} radii for {len(lig)} atoms, so the "
                           f"ligand's envelope is not defined")
        if not (np.isfinite(lig).all() and np.isfinite(rad).all()):
            return SiteFit(None, None,
                           "the ligand has a non-finite coordinate or radius")
        if bool((rad < 0.0).any()):
            return SiteFit(None, None, "a negative radius is not an envelope")

        # (atoms, voxels): one distance per pair, used in both directions below.
        d = np.linalg.norm(lig[:, None, :] - pts[None, :, :].astype(np.float64),
                           axis=2)
        return SiteFit(
            ligand_in_site=float((d.min(axis=1) <= rad).mean()),
            site_filled=float((d <= rad[:, None]).any(axis=0).mean()),
            note="",
        )

    def _points_inside(self, center, size) -> tuple[int, int]:
        """``(points inside the box, points in the site)`` for this site.

        Counts the site's own grid points rather than testing the site's
        ``size``, because a size is a claim about a box and a point set is the
        thing itself. This is also the check that makes the number worth
        having: the box is centred on the site's *centroid*, not on the middle
        of its bounds, so for a site that curves back on itself the box does
        not necessarily contain the site. Measured on the synthetic dogleg
        cleft below, the default 4 A padded box holds 1937 of 1985 points --
        97.6% -- and at zero padding only 1649 of them, 83.1%. The gap is the
        part of a winding cleft that sticks out past its own box, and no
        bounding box derived from a centroid is going to contain it.
        """
        pts = np.asarray(self.points, np.float32).reshape(-1, 3)
        if len(pts) == 0:
            return 0, 0
        half = np.asarray(size, np.float64) / 2.0
        mid = np.asarray(center, np.float64)
        inside = ((pts >= (mid - half)) & (pts <= (mid + half))).all(axis=1)
        return int(inside.sum()), int(len(pts))


@dataclass(frozen=True)
class BudgetedBox:
    """A search box for a `Pocket`, and what the size budget cost.

    Frozen and non-iterable on purpose. The fields are the answer; turning
    the object into a bare ``(center, size)`` would make the note unreachable
    and put the current silent behaviour back with a new name.
    """

    #: Where to centre the box. The site's own centroid; a budget never moves it.
    center: tuple[float, ...]
    #: The box to use, in Å. Within `max_side` per axis and `max_volume` in total.
    size: tuple[float, ...]
    #: What the site asked for: `size` before the budget, padding included.
    requested_size: tuple[float, ...]
    #: Whether either ceiling actually bit. `False` means `size == requested_size`.
    capped: bool
    #: Fraction of the site's own grid points that `size` still contains, or
    #: ``None`` for a site that carries no points and so cannot be measured.
    coverage: float | None
    #: A sentence for a person. Empty only when the box is the site's own and
    #: holds every point of it, so an empty note is a positive claim.
    note: str

    @property
    def volume(self) -> float:
        """Volume of the box handed back, Å³."""
        return float(np.prod(np.asarray(self.size, np.float64)))

    @property
    def requested_volume(self) -> float:
        """Volume of the box the site asked for, Å³."""
        return float(np.prod(np.asarray(self.requested_size, np.float64)))


@dataclass(frozen=True)
class SiteFit:
    """How much of a ligand this site can hold, in each direction.

    Two fractions that are **not** reciprocals, and the reason both are here is
    that either one alone hides a failure: a ligand too big for the site reads
    well on `site_filled` and badly on `ligand_in_site`, and a ligand dwarfed by
    the site is the other way round. See :meth:`Pocket.fit_to` for the
    measured numbers.

    This is geometric compatibility, not a score and not a ranking. It is not
    folded into `Pocket.rank_score`.
    """

    #: Fraction of the ligand's atoms whose own body reaches the site's free
    #: space. Low means the ligand is too big. ``None`` when not measured.
    ligand_in_site: float | None
    #: Fraction of the site's voxels inside the ligand's envelope. Low means the
    #: site is bigger than the ligand. ``None`` when not measured.
    site_filled: float | None
    #: Why a fraction is ``None``, or ``""`` when both were measured.
    note: str

    @property
    def measured(self) -> bool:
        """Whether both fractions could be computed."""
        return self.ligand_in_site is not None and self.site_filled is not None


def _fmt_box(size) -> str:
    """A size as a person reads it: ``"39.0 x 26.0 x 41.0"``."""
    return " x ".join(f"{float(v):.1f}" for v in np.asarray(size).reshape(3))


def _budget_note(site_size, requested, size, spent, kept, total) -> str:
    """The sentence `box_with_budget` attaches, or ``""`` when there is none.

    Empty is a claim, not an absence: it is returned only when nothing was
    capped **and** the box still holds every point of the site. A site that
    leaks out of its own box therefore gets a note too, because that is exactly
    the situation where a number in a spin box is misleading on its own.
    """
    if total and kept == total and not spent:
        return ""
    pct = f"{100.0 * kept / total:.1f}%" if total else "an unknown share"
    lost = f"; the other {total - kept} are outside the search region, so a " \
           f"ligand sitting there is not reachable from this box" if total \
           and kept < total else ""
    if not total:
        held = "this site carries no grid points, so how much of it the box " \
               "holds cannot be measured here"
    elif kept == total:
        held = f"it still holds all {total} of the site's points"
    else:
        held = (f"it still holds {kept} of the site's {total} points ({pct})"
                f"{lost}")
    if spent:
        return (
            f"the site spans {_fmt_box(site_size)} A, which pads to a "
            f"{_fmt_box(requested)} A box ({float(np.prod(requested)):.0f} A3); "
            f"{' and '.join(spent)} gives {_fmt_box(size)} A "
            f"({float(np.prod(size)):.0f} A3) on the same centre -- the site's "
            f"own centroid, which a cap does not move -- and {held}"
        )
    return (
        f"no budget was applied, so this is the site's own padded box, "
        f"{_fmt_box(requested)} A ({float(np.prod(requested)):.0f} A3), and "
        f"{held} -- the box is centred on the site's centroid rather than on "
        f"the middle of its bounds, so a site that curves back on itself sticks "
        f"out of it. Add padding, or place the box by hand."
    )



def _grid_for(coords: np.ndarray, spacing: float, margin: float):
    """Grid lower corner and shape, padded so the boundary is all solvent."""
    lo = coords.min(axis=0) - margin
    hi = coords.max(axis=0) + margin
    shape = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 2)
    return lo, tuple(int(s) for s in shape)


def _splat_solid(
    coords: np.ndarray,
    reach: np.ndarray,
    lo: np.ndarray,
    shape,
    spacing: float,
):
    """Grid points within ``reach`` of each atom, marked solid."""
    solid = np.zeros(shape, dtype=bool)
    nx, ny, nz = shape
    for xyz, r in zip(coords, reach):
        i0 = int(np.floor((xyz[0] - r - lo[0]) / spacing))
        i1 = int(np.ceil((xyz[0] + r - lo[0]) / spacing))
        j0 = int(np.floor((xyz[1] - r - lo[1]) / spacing))
        j1 = int(np.ceil((xyz[1] + r - lo[1]) / spacing))
        k0 = int(np.floor((xyz[2] - r - lo[2]) / spacing))
        k1 = int(np.ceil((xyz[2] + r - lo[2]) / spacing))
        i0, i1 = max(i0, 0), min(i1, nx - 1)
        j0, j1 = max(j0, 0), min(j1, ny - 1)
        k0, k1 = max(k0, 0), min(k1, nz - 1)
        if i0 > i1 or j0 > j1 or k0 > k1:
            continue
        xs = (lo[0] + np.arange(i0, i1 + 1) * spacing) - xyz[0]
        ys = (lo[1] + np.arange(j0, j1 + 1) * spacing) - xyz[1]
        zs = (lo[2] + np.arange(k0, k1 + 1) * spacing) - xyz[2]
        d2 = xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2
        block = solid[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        np.logical_or(block, d2 <= r * r, out=block)
    return solid


def _flood_from_border(free: np.ndarray, seed_slabs: int = 1) -> np.ndarray:
    """Grow a set through free space until it stops changing.

    Iterative 6-neighbour dilation rather than a queue: the number of rounds is
    bounded by the grid's largest dimension, and each round is six array shifts
    instead of a Python-level visit per point, which is the difference between
    a fraction of a second and minutes on a large grid.
    """
    outside = np.zeros_like(free)
    # Explicit `:` on every axis: `outside[:, :n] |= free[:, :, :n]` would be a
    # (x, 1, z) against a (x, y, 1) broadcast, which numpy rejects rather than
    # doing the obvious thing.
    outside[:seed_slabs, :, :] |= free[:seed_slabs, :, :]
    outside[-seed_slabs:, :, :] |= free[-seed_slabs:, :, :]
    outside[:, :seed_slabs, :] |= free[:, :seed_slabs, :]
    outside[:, -seed_slabs:, :] |= free[:, -seed_slabs:, :]
    outside[:, :, :seed_slabs] |= free[:, :, :seed_slabs]
    outside[:, :, -seed_slabs:] |= free[:, :, -seed_slabs:]

    while True:
        grown = outside.copy()
        grown[1:, :, :] |= outside[:-1, :, :]
        grown[:-1, :, :] |= outside[1:, :, :]
        grown[:, 1:, :] |= outside[:, :-1, :]
        grown[:, :-1, :] |= outside[:, 1:, :]
        grown[:, :, 1:] |= outside[:, :, :-1]
        grown[:, :, :-1] |= outside[:, :, 1:]
        grown &= free
        if np.array_equal(grown, outside):
            return outside
        outside = grown


def _dilate_axis(solid: np.ndarray, axis: int, steps: int) -> np.ndarray:
    """`solid` grown `steps` voxels in the **negative** direction of `axis`.

    The destination is sliced from a fresh array each round rather than updated
    in place: numpy makes no promise about the order of overlapping in-place
    writes, and `s[1:] |= s[:-1]` on the same buffer is exactly the overlap that
    would silently propagate one step per round instead of one.
    """
    cur = solid
    dst = [slice(None)] * solid.ndim
    src = [slice(None)] * solid.ndim
    dst[axis] = slice(1, None)
    src[axis] = slice(0, -1)
    dst, src = tuple(dst), tuple(src)
    for _ in range(steps):
        nxt = cur.copy()
        nxt[dst] |= cur[src]
        cur = nxt
    return cur


def _burial_score(free: np.ndarray, solid: np.ndarray, window: int) -> np.ndarray:
    """How many of the three axes have protein on *both* sides of each point.

    A point with protein above and below sits in a groove or a cleft; with it
    on all three sides it sits in a pocket. Counting axes rather than asking
    about enclosure is what lets a **surface groove** score as well as a sealed
    cavity -- and surface grooves are where a great many ligands bind, crambin
    among them, which has no internal cavity at all and so would otherwise be
    reported as having nowhere to dock.
    """
    score = np.zeros(free.shape, np.int8)
    for axis in range(3):
        # Dilating along a flipped copy gives the window in the other
        # direction, so both halves are the same routine run twice.
        back = _shift(_dilate_axis(solid, axis, window), axis, 1)
        fwd = _flip_axis(_shift(
            _dilate_axis(_flip_axis(solid, axis), axis, window), axis, 1
        ), axis)
        score += (back & fwd).astype(np.int8)
    return score * free


def _flip_axis(a: np.ndarray, axis: int) -> np.ndarray:
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(None, None, -1)
    return a[tuple(idx)]


def _shift(a: np.ndarray, axis: int, n: int) -> np.ndarray:
    """Move content by `n` along `axis`, filling the gap with False."""
    out = np.zeros_like(a)
    dst = [slice(None)] * a.ndim
    src = [slice(None)] * a.ndim
    if n > 0:
        dst[axis] = slice(n, None)
        src[axis] = slice(0, -n)
    else:
        dst[axis] = slice(0, n)
        src[axis] = slice(-n, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _components(mask: np.ndarray):
    """6-connected components, yielding index arrays.

    Dilation only ever adds voxels (`frontier` is a subset of what comes back),
    so the converged set *is* the whole component and needs no second
    accumulator. Keeping one anyway -- a `comp` that is seeded and then never
    updated while the frontier grows -- is the way this returned one voxel per
    component and quietly reported every pocket as noise.
    """
    remaining = mask.copy()
    while remaining.any():
        seed = np.argwhere(remaining)[0]
        frontier = np.zeros_like(remaining)
        frontier[tuple(seed)] = True
        while True:
            grown = frontier.copy()
            grown[1:, :, :] |= frontier[:-1, :, :]
            grown[:-1, :, :] |= frontier[1:, :, :]
            grown[:, 1:, :] |= frontier[:, :-1, :]
            grown[:, :-1, :] |= frontier[:, 1:, :]
            grown[:, :, 1:] |= frontier[:, :, :-1]
            grown[:, :, :-1] |= frontier[:, :, 1:]
            grown &= remaining
            if np.array_equal(grown, frontier):
                break
            frontier = grown
        remaining &= ~frontier
        yield np.argwhere(frontier)


def _lining(coords, residues, points, spacing, lo, max_dist):
    """Residues with an atom within `max_dist` of any pocket point."""
    if not residues or len(points) == 0:
        return []
    counts: dict[str, int] = {}
    pts = points.astype(np.float32) * np.float32(spacing) + lo.astype(np.float32)
    for xyz, res in zip(coords, residues):
        if not res:
            continue
        d = np.linalg.norm(pts - xyz, axis=1).min()
        if d <= max_dist:
            counts[res] = counts.get(res, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def find_pockets(
    coords,
    elements,
    residues=None,
    *,
    probe: float = DEFAULT_PROBE,
    spacing: float = DEFAULT_SPACING,
    padding: float = DEFAULT_PADDING,
    margin: float = 6.0,
    min_voxels: int = MIN_VOXELS,
    max_pockets: int = DEFAULT_MAX_POCKETS,
    min_burial: int = 1,
    min_extent: float | None = None,
    max_volume: float | None = None,
) -> list[Pocket]:
    """Candidate binding sites in a structure, best first.

    Two kinds are found and told apart, because they are different claims:

    * **cavity** -- sealed from the solvent, found by flooding from the grid
      boundary and keeping what the flood cannot reach. Unambiguous.
    * **burial** -- not sealed, but enclosed on at least `min_burial` of the
      three axes within a few ångström. This is what a surface groove looks
      like, and a great many ligands bind in grooves.

    `min_burial` defaults to **1** rather than 2 for a measured reason. On
    crambin, with ibuprofen actually docked into it, the site the ligand
    occupies scores 0 at the probe centroid, 1 over most of its own volume and
    2 in only two grid cells. At `min_burial=2` it is absent from the list
    entirely and the list instead offers deeper grooves on the far side of the
    protein. At 1 it is found, at rank 9 of 16 -- see the module docstring for
    the full table. The price is a longer list, which is what `max_pockets`
    is for.

    `coords` is ``(n, 3)`` and `elements` a list of element symbols. `residues`
    is an optional list of residue labels, one per atom; when it is given each
    site reports which residues line it.

    Returns an empty list when there is nothing. That is a real answer, and the
    caller must still offer a hand-placed box rather than treating the result
    as the only option.
    """
    pts = np.asarray(coords, np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return []
    els = list(elements) if elements is not None else ["C"] * len(pts)
    if len(els) < len(pts):
        els = els + ["C"] * (len(pts) - len(els))
    res = list(residues) if residues is not None else [""] * len(pts)
    if len(res) < len(pts):
        res = res + [""] * (len(pts) - len(res))

    radii = np.asarray([VDW_RADII.get(e, 1.70) for e in els], np.float32)
    if min_extent is None:
        min_extent = 2.0 * spacing
    if max_volume is None:
        max_volume = DEFAULT_MAX_VOLUME
    lo, shape = _grid_for(pts, spacing, margin)

    solid = _splat_solid(pts, radii + np.float32(probe), lo, shape, spacing)
    free = ~solid
    sealed = free & ~_flood_from_border(free)

    window = max(int(round(3.0 / spacing)), 1)
    score = _burial_score(free, solid, window)
    # NOT dilated, and this is the single most consequential line in the file.
    #
    # A one-voxel isotropic growth of this mask was here to stop thin sheets
    # being reported as slivers, and it did -- by merging every site into the
    # protein's outer surface, because a surface groove is one voxel thick and
    # a growth of one voxel reaches across its neck. The merged lump then
    # failed the volume ceiling and took the real pocket with it. Measured, on
    # four complexes whose ligand is bound in the crystal structure:
    #
    #     case          the ligand's own site, undilated   after one dilation
    #     1STP biotin   314 voxels, 1305 A3 box  -> kept   4905 voxels, 71909 A3
    #     3PTB benz.    698 voxels, 10161 A3 box -> kept   4114 voxels, 77893 A3
    #     1HVR XK2     1063 voxels,  8008 A3 box -> kept   6303 voxels,109965 A3
    #
    # In all three the site is a sensible size before the growth and a lump
    # of the whole protein after it. Growth now happens per component, after
    # labelling, where it can no longer merge anything -- and on all four
    # complexes the bound ligand's own site comes out **first** in the
    # shortlist, where before this change it was not in the shortlist at all.
    interior = free & (score >= min_burial) & ~sealed

    found: list[Pocket] = []
    for mask, kind in ((sealed, "cavity"), (interior, "burial")):
        if not mask.any():
            continue
        for comp in _components(mask):
            if len(comp) < min_voxels:
                continue
            extent = (comp.max(axis=0) - comp.min(axis=0) + 1) * spacing
            if float(extent.min()) < min_extent:
                # A site one grid cell across in any direction has no interior
                # for a ligand to occupy; it is a dent in the surface, and
                # listing a dozen of them buries the two real candidates.
                continue
            volume = len(comp) * spacing ** 3
            if volume > max_volume:
                # Not a site. At `min_burial=1` the single-voxel
                # protein-solvent-protein events scattered over a protein's
                # whole surface come out as a handful of lumps, and on
                # crambin the three largest hold 193, 192 and 126 A3 -- more
                # free space than any real groove there, so a volume ranking
                # puts them first and the list becomes worse than useless.
                #
                # The ceiling is on the site's own volume and never on its
                # bounding box. The box measure is wrong in both directions
                # and using it discarded real binding pockets: a winding
                # cleft has a huge box and little space in it (1HVR: 8008 A3
                # of box, 544 A3 of pocket), while a broad shallow patch of
                # surface has a small box and many voxels in it.
                #
                # The ceiling itself is a heuristic, and a loose one: drug
                # binding pockets run a few hundred A3, and 1500 leaves room
                # for a generous one. Nothing here derives it, so it is a
                # parameter rather than a constant baked into the scoring.
                continue
            world = (
                comp.astype(np.float32) * np.float32(spacing) + lo.astype(np.float32)
            )
            found.append(
                Pocket(
                    center=world.mean(axis=0).astype(np.float32),
                    size=extent.astype(np.float32),
                    voxels=int(len(comp)),
                    volume=float(volume),
                    kind=kind,
                    burial=float(score[comp[:, 0], comp[:, 1], comp[:, 2]].mean()),
                    lining=_lining(pts, res, comp, spacing, lo, LINING_MAX),
                    points=world.astype(np.float32),
                    spacing=float(spacing),
                )
            )

    found.sort(key=lambda p: -p.rank_score)
    for p in found:
        p.lining = p.lining[:12]
    return found[:max_pockets]
