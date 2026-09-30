"""What the pose is touching, and where.

A docking result is a list of numbers. This module turns the interesting ones
into a picture and a sentence: which residues the ligand sits against, which of
those contacts are hydrogen bonds, and how far away everything is.

The workbench had none of this. It could show a pose inside a receptor, but
nothing distinguished a pose buried in a hydrophobic groove from one clinging
to the outside of the protein, because *from the outside those look the same*.
The scoring function is what tells them apart, and this is the display half of
that same question.

# What "hydrogen bond" means here

A geometric filter, not a quantum calculation. A contact counts as a hydrogen
bond when the H···acceptor distance is short **and** the donor–H···acceptor
angle is wide enough:

* ``HBOND_MAX`` (2.6 Å) on the H···acceptor distance;
* ``HBOND_MIN_ANGLE`` (120°) on the donor–H···acceptor angle.

Both directions are checked, because in a docked pose the donor is usually the
*receptor* — a backbone NH or a serine OH reaching into a ligand carbonyl —
and a filter that only looked at the ligand would miss most of them.

This does not verify that the acceptor's lone pair points back at the hydrogen,
does not model water, and assigns no energies. It is the same first pass a
drawing program makes, and it is reported as a filter rather than as chemistry.
The angles and distances are returned on every contact so a caller can tighten
the cut-offs instead of trusting them.

# Why it needs the polar hydrogens

Both criteria are stated on the hydrogen, so a receptor prepared without polar
hydrogens cannot produce hydrogen bonds here at all — it would silently report
close contacts only. `prepare_receptor` adds them, and its writer drops any
whose position is not 0.7–1.35 Å from the parent, so a hydrogen listed by this
module is one that was actually placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Contact",
    "find_contacts",
    "residue_summary",
    "HBOND_MAX",
    "HBOND_MIN_ANGLE",
    "CLOSE_MAX",
]

#: H···acceptor distance, Å.
HBOND_MAX = 2.6
#: donor–H···acceptor angle, degrees.
HBOND_MIN_ANGLE = 120.0
#: Heavy-atom contact distance, Å. 4.0 is the usual "these are touching" cut.
CLOSE_MAX = 4.0

#: Atoms that can donate or accept.
POLAR = frozenset({"N", "O"})
#: Atoms treated as apolar for classification purposes.
APOLAR = frozenset({"C", "S", "P", "F", "Cl", "Br", "I"})

_KIND_ORDER = {"hbond": 0, "polar": 1, "hydrophobic": 2, "close": 3}


@dataclass(frozen=True)
class Contact:
    """One interaction between the pose and the receptor.

    ``distance`` is always between the two atoms named by ``self_name`` and
    ``partner_name`` -- for a hydrogen bond that is the *hydrogen* and the
    acceptor, not the donor and the acceptor, because that is the distance the
    2.6 Å criterion is stated on.
    """

    kind: str
    self_index: int
    partner_index: int
    distance: float
    angle: float = 0.0
    self_name: str = ""
    partner_name: str = ""
    self_element: str = ""
    partner_element: str = ""
    partner_residue: str = ""
    #: D-H...A angle's donor, named for hydrogen bonds.
    donor_name: str = ""

    @property
    def residue(self) -> str:
        """Residue label of the receptor atom, or ``""`` if it has none."""
        return self.partner_residue

    def label(self) -> str:
        """One line for the contacts table.

        A hydrogen bond names the *donor's* heavy atom rather than its
        hydrogen. `THR 2 OG1` is the atom a chemist would look up; `THR 2 H`
        is not, and the line is about as long either way.
        """
        where = self.partner_residue or f"atom {self.partner_name or self.partner_index}"
        if self.kind == "hbond":
            partner = self.donor_name or self.partner_name or self.partner_element
        else:
            partner = self.partner_name or self.partner_element
        return (
            f"{self.kind:<11} {self.self_name or self.self_element:>4}"
            f"  ->  {partner:<4} {where:<10}"
            f" {self.distance:5.2f} A"
            + (f"  {self.angle:5.1f} deg" if self.kind == "hbond" else "")
        )


@dataclass
class _Side:
    """The bits of a molecule that the search needs, computed once."""

    coords: np.ndarray
    elements: list[str]
    names: list[str]
    residues: list[str]
    #: heavy-atom indices whose element can donate or accept
    polar: np.ndarray
    #: hydrogen indices
    hydrogens: np.ndarray
    #: for every hydrogen, the index of the heavy atom it hangs off
    donor_of: dict[int, int] = field(default_factory=dict)
    #: H coordinate -> donor heavy-atom coordinate, for the angle test
    donor_vec: dict[int, np.ndarray] = field(default_factory=dict)


def _side(mol) -> _Side:
    """Index one molecule for contact search, with residue names resolved."""
    coords = np.asarray(getattr(mol, "coords", ()), np.float32).reshape(-1, 3)
    elements = list(getattr(mol, "elements", []) or [])
    n = len(coords)
    if len(elements) < n:
        elements = elements + ["C"] * (n - len(elements))

    structure = getattr(mol, "structure", None)
    names: list[str] = []
    residues: list[str] = []
    atoms = getattr(structure, "atoms", None) if structure is not None else None
    for i in range(n):
        rec = atoms[i] if atoms is not None and i < len(atoms) else None
        names.append(getattr(rec, "name", "") or "")
        if rec is None:
            residues.append("")
            continue
        resname = (getattr(rec, "resname", "") or "").strip()
        if not resname or resname in ("UNL", "REC"):
            residues.append("")
        else:
            chain = (getattr(rec, "chain", "") or "").strip()
            residues.append(f"{resname} {getattr(rec, 'resid', 0)}{chain}")

    polar = np.asarray(
        [i for i in range(n) if elements[i] in POLAR], dtype=np.int64
    )
    hydrogens = np.asarray(
        [i for i in range(n) if elements[i] == "H"], dtype=np.int64
    )

    # The hydrogen's parent comes from the bond table when there is one -- the
    # structure layer built it from templates or from the file, and it knows
    # which N this H belongs to. Falling back to "the nearest polar atom within
    # a bond length" is only a guess, so it is tried second and only for atoms
    # the bonds did not resolve.
    donor_of: dict[int, int] = {}
    pairs = np.asarray(getattr(mol, "bonds", np.zeros((0, 2), np.int32))).reshape(-1, 2)
    for a, b in pairs:
        a, b = int(a), int(b)
        if not (0 <= a < n and 0 <= b < n):
            continue
        if elements[b] == "H" and elements[a] in POLAR:
            donor_of.setdefault(b, a)
        elif elements[a] == "H" and elements[b] in POLAR:
            donor_of.setdefault(a, b)
    for h in hydrogens.tolist():
        if h in donor_of:
            continue
        if len(polar) == 0:
            continue
        d = np.linalg.norm(coords[polar] - coords[h], axis=1)
        j = int(np.argmin(d))
        if d[j] <= 1.35:
            donor_of[h] = int(polar[j])

    donor_vec = {
        h: coords[donor_of[h]] - coords[h] for h in donor_of
    }

    return _Side(
        coords=coords,
        elements=elements,
        names=names,
        residues=residues,
        polar=polar,
        hydrogens=hydrogens,
        donor_of=donor_of,
        donor_vec=donor_vec,
    )


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors, degrees, clamped against round-off."""
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))


def _classify(e1: str, e2: str) -> str:
    p1, p2 = e1 in POLAR, e2 in POLAR
    if p1 and p2:
        return "polar"
    if p1 or p2:
        return "polar"
    if e1 in APOLAR and e2 in APOLAR:
        return "hydrophobic"
    return "close"


def _hbonds(
    donor_side: _Side,
    acceptor_side: _Side,
    reverse: bool,
    hbond_max: float,
    hbond_min_angle: float,
) -> list[Contact]:
    """Hydrogen bonds with the donor on one side and the acceptor on the other.

    The thresholds are passed in rather than read from the module. They used to
    be read from `HBOND_MAX` / `HBOND_MIN_ANGLE` directly, which meant
    `find_contacts(..., hbond_max=...)` was accepted, documented, and then
    ignored -- a control that looks real and does nothing is worse than one that
    is absent, because a caller tightening the cut-offs to check a borderline
    pose would be told it had worked.
    """
    out: list[Contact] = []
    for h in donor_side.hydrogens.tolist():
        donor = donor_side.donor_of.get(h)
        if donor is None:
            # A hydrogen with no resolved parent cannot have its angle judged,
            # and a bond drawn from an angle nobody computed is exactly the kind
            # of confident-looking guess this project keeps refusing to draw.
            continue
        hpos = donor_side.coords[h]
        back = donor_side.donor_vec[h]
        if len(acceptor_side.polar) == 0:
            continue
        d = np.linalg.norm(acceptor_side.coords[acceptor_side.polar] - hpos, axis=1)
        for k in np.nonzero(d <= hbond_max)[0].tolist():
            a = int(acceptor_side.polar[k])
            ang = _angle(back, acceptor_side.coords[a] - hpos)
            if ang < hbond_min_angle:
                continue
            if reverse:
                out.append(
                    Contact(
                        kind="hbond",
                        self_index=a,
                        partner_index=h,
                        distance=float(d[k]),
                        angle=ang,
                        self_name=acceptor_side.names[a] or acceptor_side.elements[a],
                        partner_name=donor_side.names[h] or "H",
                        self_element=acceptor_side.elements[a],
                        partner_element="H",
                        partner_residue=donor_side.residues[h],
                        donor_name=donor_side.names[donor] or donor_side.elements[donor],
                    )
                )
            else:
                out.append(
                    Contact(
                        kind="hbond",
                        self_index=h,
                        partner_index=a,
                        distance=float(d[k]),
                        angle=ang,
                        self_name=donor_side.names[h] or "H",
                        partner_name=acceptor_side.names[a] or acceptor_side.elements[a],
                        self_element="H",
                        partner_element=acceptor_side.elements[a],
                        partner_residue=acceptor_side.residues[a],
                        donor_name=donor_side.names[donor] or donor_side.elements[donor],
                    )
                )
    return out


def find_contacts(
    pose,
    receptor,
    *,
    hbond_max: float = HBOND_MAX,
    hbond_min_angle: float = HBOND_MIN_ANGLE,
    close_max: float = CLOSE_MAX,
) -> list[Contact]:
    """Every interaction between ``pose`` and ``receptor``.

    The pose is always the ``self`` side, so ``residue_summary`` and the
    contacts table always read from the protein's point of view. Hydrogen bonds
    are reported once, from whichever side the donor is on; a pair that is a
    hydrogen bond is not repeated as a plain close contact.
    """
    a = _side(pose)
    b = _side(receptor)
    if len(a.coords) == 0 or len(b.coords) == 0:
        return []

    contacts: list[Contact] = _hbonds(a, b, False, hbond_max, hbond_min_angle)
    contacts += _hbonds(b, a, True, hbond_max, hbond_min_angle)
    hb_sites = {(c.self_index, c.partner_index) for c in contacts}

    # Close contacts, skipping hydrogens: an H sits 1 A from its own heavy atom
    # and would otherwise report every C-H as a contact with whatever is near.
    heavy_a = np.asarray(
        [i for i in range(len(a.coords)) if a.elements[i] != "H"], dtype=np.int64
    )
    heavy_b = np.asarray(
        [i for i in range(len(b.coords)) if b.elements[i] != "H"], dtype=np.int64
    )
    if len(heavy_a) and len(heavy_b):
        # Chunked over the receptor so a 100k-atom protein does not need a
        # (n_pose x 100k) distance matrix resident at once.
        chunk = max(1, min(len(heavy_b), 20000))
        for start in range(0, len(heavy_b), chunk):
            block = heavy_b[start : start + chunk]
            d = np.linalg.norm(
                a.coords[heavy_a][:, None, :] - b.coords[block][None, :, :], axis=2
            )
            for i_rel, j_rel in zip(*np.nonzero(d <= close_max)):
                i, j = int(heavy_a[i_rel]), int(block[j_rel])
                if (i, j) in hb_sites:
                    continue
                contacts.append(
                    Contact(
                        kind=_classify(a.elements[i], b.elements[j]),
                        self_index=i,
                        partner_index=j,
                        distance=float(d[i_rel, j_rel]),
                        self_name=a.names[i] or a.elements[i],
                        partner_name=b.names[j] or b.elements[j],
                        self_element=a.elements[i],
                        partner_element=b.elements[j],
                        partner_residue=b.residues[j],
                    )
                )

    contacts.sort(
        key=lambda c: (_KIND_ORDER.get(c.kind, 9), c.distance, c.partner_residue)
    )
    return contacts


def residue_summary(contacts: list[Contact]) -> list[tuple[str, int, int, float]]:
    """``(residue, n_hbonds, n_contacts, closest)`` per residue, best first.

    Residues are ranked by hydrogen bonds first and proximity second, because
    the residue a pose is hydrogen-bonded to is the one a chemist wants to see
    first, and "closest" on its own would rank a grazing contact above a
    hydrogen bond.
    """
    rows: dict[str, list[Contact]] = {}
    for c in contacts:
        if not c.partner_residue:
            continue
        rows.setdefault(c.partner_residue, []).append(c)

    out = []
    for residue, group in rows.items():
        hb = sum(1 for c in group if c.kind == "hbond")
        out.append((residue, hb, len(group), min(c.distance for c in group)))
    out.sort(key=lambda r: (-r[1], -r[2], r[3]))
    return out
