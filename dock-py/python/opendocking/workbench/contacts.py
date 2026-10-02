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

# The residue table and the pair table

Two tables, one list. `residue_summary` answers "which residues is this pose
against", which is the question a pose table's contact column asks. `pair_rows`
answers "which two atoms, and how far", which is the question a chemist asks
when a residue name is not enough -- a glutamine can hydrogen-bond through its
backbone amide or through its side-chain nitrogen, and those are two different
interactions with two different consequences.

They are built from the *same* `find_contacts` call, and `Pair.contact_index` is
the position in that list, so a row in the pair table and a dashed line in the
viewport are one object seen twice. Two calls would be two searches, and the
second could disagree with the first about what is touching; a table whose rows
and whose lines name different atoms is worse than no table.

# The term column, and why most of it is empty

`TERM_BY_KIND` maps only the two kinds whose *stated subject* is the test that
found the pair, and `PAIR_TERM_NOTE` is the sentence the panel prints under the
table. The reasoning is on the table itself. The short version: the engine sums
`g1`, `g2` and `rep` over every atom pair in a pose and reports one number each,
so there is no per-pair share to print, and a cell with a number in it would be a
number nobody computed.

# A limit, recorded because it was measured rather than assumed

**The residue table does not keep its selected row across a pose change, and the
first version of this paragraph said it did.** It was wrong, and it was wrong in
a way worth keeping: the reasoning underneath it was sound and the premise was
not. `QTableWidget.setRowCount` really does only drop a selection that falls
outside the new row count -- measured on Qt 6.11, row 0 selected: 7->9 rows
keeps it, 9->4 keeps it, 9->0 drops it. But `_refresh_contacts` does not go from
one count to another. It goes to **zero** first, because it empties both tables
and refills them from one `find_contacts` call, and the empty pass is where the
residue selection dies. Row 0 of the new pose is not left selected; nothing is.

So the hazard this file was asked to record is not a live defect, and saying
otherwise would have been a comment of the kind this module exists to argue
against. What is real is narrower and is stated as a constraint on the next
change rather than as a reassurance about this one:

* **The selection is dropped by the zeroing, not by a rule.** Nothing in the
  residue path clears it, clears the current cell, or otherwise reasons about
  it. A refactor that stopped zeroing the residue table -- refilling in place,
  say, to keep a scroll position -- would silently start retaining rows, and it
  would look like it worked.
* **That is the day a retained row becomes a lie.** Today a selected residue
  only flies the camera and claims nothing about the picture, so there is no
  second thing for a stale row to contradict. The moment the residue selection
  grows a highlighting behaviour the way the pair selection did -- a line, a
  marker, a dimming of everything else -- a retained row stops being a table
  detail and becomes a claim about the picture, and the claim is about the
  previous pose. The pair table is already safe: `_refresh_contacts` clears it
  explicitly, and one call above the residue table, so the fix for the residue
  table is to do what that one does.
* **The order is load-bearing.** `setRowCount(0)` before the refill is what
  makes both tables end up consistent with the pose on screen. Restoring the
  selection afterwards would undo the point, and the check below is what would
  say so.

It is pinned rather than left to this paragraph: section 14 of
`scripts/workbench_interaction_check.py` measures the selection across a pose
change on every run, reports which way it went, and asserts the half that is
load-bearing -- **the residue path touches no highlight state at all**, so there
is nothing on the screen for a row to be wrong about. If a residue highlight is
ever added, that assertion goes red and names this limit rather than leaving the
next reader to find a stale row in a picture that now claims something. The
selection half is reported and not asserted, because it is Qt's and the
zeroing's business rather than this module's promise: a Qt that preserved it
would be a correct implementation of the same code.

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
    "Pair",
    "find_contacts",
    "pair_rows",
    "residue_summary",
    "HBOND_MAX",
    "HBOND_MIN_ANGLE",
    "CLOSE_MAX",
    "TERM_BY_KIND",
    "PAIR_TERM_NOTE",
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


#: The engine's term keys, and which contact kinds can be attributed to one
#: without a guess.
#:
#: `energy_terms.TERM_ROWS` is the authority for what each term *measures*, and
#: these two entries are the cases where the term's stated subject is the same
#: test that produced the row: `hb` is "donor-acceptor attraction", and a
#: `hbond` row exists only because the pair passed a donor-H-acceptor distance
#: and angle; `hyd` is "attraction between two non-polar atoms", and a
#: `hydrophobic` row exists only because both atoms were apolar. For those two
#: the mapping is not an inference, it is a restatement.
#:
#: The other two kinds are the opposite case and are deliberately absent. The
#: remaining terms -- `g1` for a contact of about the right size, `g2` for one
#: that is merely not clashing, `rep` for the wall -- are summed over every atom
#: pair in the pose, and `score_conformation_terms` returns the sum. Nothing
#: Python can see carries a per-pair share of them: the engine reports no
#: per-pair surface distance either (see `NEVER_SUPPLIED` in
#: `workbench/pose_trust.py`, which is the same fact about the same module).
#: Which of the three a given close contact belongs to depends on the engine's
#: own distance ramps, which are not exposed, so the row names no term at all
#: rather than the most likely one.
TERM_BY_KIND = {"hbond": "hb", "hydrophobic": "hyd"}

#: What the panel prints under the pair table, so the empty cells are a fact
#: rather than a gap.
PAIR_TERM_NOTE = (
    "`hb` and `hyd` are the engine's own term keys for the two kinds whose "
    "subject is the test that found the row. The other contacts belong to `g1`, "
    "`g2` and `rep`, which the engine sums over every atom pair in the pose and "
    "reports as one number each -- it exposes no per-pair share, so this column "
    "names a term only where the mapping is the test itself and leaves the cell "
    "empty rather than dividing a pose's total by its pairs."
)


@dataclass(frozen=True)
class Pair:
    """One interaction, with both sides named and its term attributed.

    `contact_index` is the row's position in the list `find_contacts` returned,
    and it is what ties this row to one dashed line in the viewport: the line is
    drawn from `contacts[contact_index]`, so selecting a row cannot highlight a
    pair the picture is not showing.

    The two sides are always **pose** (`self_*`) and **receptor** (`partner_*`),
    including for a hydrogen bond -- `find_contacts` guarantees the pose is the
    `self` side, and for a bond one of the two is the hydrogen the 2.6 A
    criterion is stated on. `pose_atom` and `receptor_atom` print the donor's
    heavy atom in place of that hydrogen, because `GLY 12 N` is the atom a
    reader looks up and `GLY 12 H` is not; `measured_to_hydrogen` says so, and
    `tooltip` says it with the angle.
    """

    contact_index: int
    kind: str
    self_residue: str
    self_name: str
    self_element: str
    partner_residue: str
    partner_name: str
    partner_element: str
    donor_name: str
    distance: float
    angle: float
    term: str

    @property
    def measured_to_hydrogen(self) -> bool:
        """Whether ``distance`` is the H···acceptor distance of a bond."""
        return self.self_element == "H" or self.partner_element == "H"

    @property
    def pose_atom(self) -> str:
        return self._atom(self.self_residue, self.self_name, self.self_element)

    @property
    def receptor_atom(self) -> str:
        return self._atom(self.partner_residue, self.partner_name,
                          self.partner_element)

    def _atom(self, residue: str, name: str, element: str) -> str:
        if element == "H" and self.donor_name:
            name = self.donor_name
        name = name or element
        return f"{residue} {name}" if residue else name

    @property
    def distance_text(self) -> str:
        text = f"{self.distance:.2f} A"
        if self.kind == "hbond":
            text += f"  {self.angle:.0f} deg"
        return text

    def tooltip(self) -> str:
        """Everything the row asserts, including what it does not."""
        parts = [f"{self.kind}: {self.pose_atom} -> {self.receptor_atom}"]
        if self.kind == "hbond":
            parts.append(
                f"the distance is the H···acceptor one, at "
                f"{self.angle:.1f} deg donor-H-acceptor"
            )
            if self.donor_name:
                parts.append(f"named through its donor {self.donor_name}")
        parts.append(
            f"this is contact {self.contact_index + 1} of the list the dashed "
            f"lines are drawn from"
        )
        parts.append(
            f"term: {self.term}" if self.term
            else "no engine term is named for this row: the contact terms are "
                 "summed over the whole pose and the engine reports no per-pair "
                 "share"
        )
        return ". ".join(parts)


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
    atoms = getattr(structure, "atoms", None) if structure is not None else None
    names: list[str] = [
        (getattr(atoms[i], "name", "") or "") if atoms is not None and i < len(atoms) else ""
        for i in range(n)
    ]
    # Shared with the pocket lining so both name a residue identically. The
    # fallback covers a duck-typed molecule with no `residue_labels`, which is
    # what the synthetic fixtures hand in.
    labeler = getattr(mol, "residue_labels", None)
    residues: list[str] = (
        list(labeler())
        if callable(labeler)
        else [""] * n
    )
    if len(residues) < n:
        residues = residues + [""] * (n - len(residues))

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
    # The heavy-atom pair each hydrogen bond *implies*: its donor and its
    # acceptor. That is the pair the close-contact search below finds on its
    # own, and suppressing it is what `find_contacts` promises.
    #
    # It used to record the *hydrogen* and the acceptor instead. The
    # close-contact search skips hydrogens on both sides, so a key containing
    # one could never match, the suppression was a no-op in both directions, and
    # every hydrogen bond came back twice -- once as the bond and again as a
    # close contact between the same two heavy atoms, 1.0 A further out. None of
    # the 39 checks in `scripts/contacts_check.py` noticed, because the two rows
    # are different index pairs and every one of them is individually true; the
    # visible cost was that `residue_summary` counted each hydrogen bond twice,
    # in the number a user reads.
    hb_sites: set[tuple[int, int]] = set()
    for c in contacts:
        if c.kind != "hbond":
            continue
        if c.self_element == "H" and c.self_index in a.donor_of:
            hb_sites.add((a.donor_of[c.self_index], c.partner_index))
        elif c.partner_element == "H" and c.partner_index in b.donor_of:
            hb_sites.add((c.self_index, b.donor_of[c.partner_index]))

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


def pair_rows(contacts: list[Contact],
              self_residues: "list[str] | tuple[str, ...] | None" = None
              ) -> tuple[Pair, ...]:
    """One `Pair` per interaction, in the order `find_contacts` returned them.

    ``self_residues`` is the **pose's** per-atom residue labels, indexed by the
    same integer `Contact.self_index` is. It is passed in rather than read from
    the molecule because `find_contacts` has already resolved them and thrown
    that away: asking the molecule again means a second pass over it on every
    refresh, and a table whose residues came from a different pass than its
    distances is a table that can disagree with itself.

    The rows are **not** re-sorted. A pair table is a list to be read against
    the picture, and the order the dashes are drawn in is the order the contacts
    came back in; a table that reorders them would make row 7 mean something
    different from the seventh line. `Contact.kind` is already the strongest
    class first, because `find_contacts` sorts by it.
    """
    labels = list(self_residues or ())
    out: list[Pair] = []
    for i, c in enumerate(contacts):
        residue = (labels[c.self_index]
                   if 0 <= c.self_index < len(labels) else "")
        out.append(
            Pair(
                contact_index=i,
                kind=c.kind,
                self_residue=residue,
                self_name=c.self_name,
                self_element=c.self_element,
                partner_residue=c.partner_residue,
                partner_name=c.partner_name,
                partner_element=c.partner_element,
                donor_name=c.donor_name,
                distance=float(c.distance),
                angle=float(c.angle),
                term=TERM_BY_KIND.get(c.kind, ""),
            )
        )
    return tuple(out)
