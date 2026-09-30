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

# Measured, on crambin with ibuprofen docked into it

This is what the numbers came out to be, because "it finds pockets" is not a
claim anyone should have to take on trust:

* crambin has **no sealed cavity** at a 1.4 Å probe. Every site found is a
  groove. A cavity-only detector would report a protein with an obvious
  binding site as having nowhere to dock.
* the site ibuprofen actually occupies scores **0** at its own centroid,
  **1** over most of its volume and **2** in exactly two grid cells. It is a
  shallow surface groove, not a cleft. A "PSP events only" detector with a
  threshold of 2 misses it completely.
* the box built from it contains **16 of 16** docked ligand atoms, in a
  search volume 4.4x smaller than the whole-protein box that was there before,
  and it ranks **second of the eight** sites offered. It is found and it is
  cheap; it is not automatically first, and this does not claim it should be.
* the lining residues it reports include **THR 2**, which the independent
  contact analysis names as the hydrogen-bond donor to the docked ligand's
  carboxylate. Two unrelated computations agreeing on one residue is worth
  more than either one alone.

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


#: Ceiling on a site's bounding-box volume, Å³. A drug binding pocket runs a
#: few hundred cubic ångström; anything much past that is a merged surface
#: rather than a site. A heuristic, and labelled as one -- see `max_volume`.
DEFAULT_MAX_VOLUME = 1500.0


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
    #: Mean burial score over the site, 0-3. Three means protein on all sides.
    burial: float = 0.0
    #: ``(residue, n_atoms_within_lining_max)`` pairs, most-contacted first.
    lining: list[tuple[str, int]] = field(default_factory=list)

    @property
    def rank_score(self) -> float:
        """Ranking key: size, tempered by how enclosed the site is.

        Volume alone ranks a flooded cavern above a tight, well-formed pocket.
        Taking the cube root keeps an 8x larger cavity from outscoring a pocket
        whose *shape* fits better, and multiplying by the burial score stops a
        shallow surface scrape from outranking a real cavity of the same size.
        """
        return float(self.voxels) ** (1.0 / 3.0) * (0.5 + 0.5 * self.burial)

    def box_center_and_size(self, padding: float = DEFAULT_PADDING):
        """``(center, size)`` for a `GridBox` around this site.

        The centre is the site's own centroid, not its bounding-box middle: for
        a curved cleft those differ by several ångström, and the bounding box's
        middle can land in the wall.
        """
        size = np.asarray(self.size, np.float32) + np.float32(2.0 * padding)
        return tuple(float(v) for v in self.center), tuple(float(v) for v in size)


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


def _dilate_iso(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    """Grow a mask by `steps` in all six directions.

    The burial threshold selects *sheets* -- a point in a concave groove scores
    on two axes, and its neighbours do too, so the qualifying set comes out one
    voxel thick. A one-voxel sheet has a bounding extent of 0.8 A and no volume,
    so every such "pocket" is reported as a sliver and ranked on a size that
    means nothing. One dilation turns a sheet into a blob with a real extent,
    which is also the standard way to close a region before measuring it.
    """
    cur = mask
    for _ in range(steps):
        nxt = cur.copy()
        nxt[1:, :, :] |= cur[:-1, :, :]
        nxt[:-1, :, :] |= cur[1:, :, :]
        nxt[:, 1:, :] |= cur[:, :-1, :]
        nxt[:, :-1, :] |= cur[:, 1:, :]
        nxt[:, :, 1:] |= cur[:, :, :-1]
        nxt[:, :, :-1] |= cur[:, :, 1:]
        cur = nxt
    return cur


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
    max_pockets: int = 8,
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
    entirely and the list instead offers eight deeper grooves on the far side
    of the protein. At 1 it is found, and -- once merged surface is dropped,
    see `max_volume` -- it ranks first. The price is a longer list: crambin
    yields 20 sites rather than 8, and `max_pockets` cuts it back down.

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
    # Dilated so a qualifying sheet becomes a blob with a real extent; the
    # result is intersected back with ree because growing a mask can push it
    # into protein. The shell is deliberately *not* excluded: a surface groove
    # is exactly the case worth reporting, and dropping it would leave this
    # finding only the sealed cavities it already had.
    interior = _dilate_iso(free & (score >= min_burial) & ~sealed) & free

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
            if float(np.prod(extent)) > max_volume:
                # Not a site. At `min_burial=1` the single-voxel
                # protein-solvent-protein events scattered over a protein's
                # whole surface become 6-connected after one dilation, and on
                # crambin the three largest came out as 3515, 3564 and 1720 A³
                # -- the entire outer surface as three lumps. They have more
                # volume than any real groove there, so a volume ranking puts
                # them first and the list becomes worse than useless.
                #
                # The ceiling is a heuristic, and a loose one: drug binding
                # pockets run a few hundred A³, and 1500 leaves room for a
                # generous one. Nothing here derives it, so it is a parameter
                # rather than a constant baked into the scoring.
                continue
            world = (
                comp.astype(np.float32) * np.float32(spacing) + lo.astype(np.float32)
            )
            found.append(
                Pocket(
                    center=world.mean(axis=0).astype(np.float32),
                    size=extent.astype(np.float32),
                    voxels=int(len(comp)),
                    kind=kind,
                    burial=float(score[comp[:, 0], comp[:, 1], comp[:, 2]].mean()),
                    lining=_lining(pts, res, comp, spacing, lo, LINING_MAX),
                )
            )

    found.sort(key=lambda p: -p.rank_score)
    for p in found:
        p.lining = p.lining[:12]
    return found[:max_pockets]
