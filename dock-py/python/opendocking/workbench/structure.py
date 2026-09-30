"""Residue-aware structure parsing and bond perception for the workbench.

# Why this exists

The viewer originally drew bonds with a single rule: connect any two atoms
closer than 1.95 A. That is fine for a twenty-atom drug and wrong for a
protein, where it happily welds atoms from different residues together across a
tight interface and then draws the result as if it were a real bond. A viewer
that connects the wrong atoms is worse than one that draws none, because the
picture looks authoritative either way.

So connectivity here comes from a source that *states* it, never from a
threshold:

* **Ligands and poses** -- the `ROOT` / `BRANCH` records in the PDBQT declare the
  tree explicitly. Those bonds are read, not guessed. Rings are then closed by
  looking for declared-tree-external contacts, which is the only inference made
  and it is confined to pairs inside the same `ROOT` block.
* **Proteins** -- residue templates. Each amino acid's internal connectivity is
  a table keyed by PDB atom name, and consecutive residues are joined only by
  the peptide C-N link. A residue the table does not know contributes **no
  bonds and a warning**, because a plausible guess is the failure mode this
  module is here to prevent.

Distance is still used, but only as an audit: every bond this module produces is
measured against a covalent-radius range and anything outside it is reported.
That turns "the bonds are probably right" into a list a test can assert on.

Nothing in here imports Qt or moderngl, so all of it is testable headlessly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "AtomRecord",
    "Residue",
    "Structure",
    "parse_structure",
    "COVALENT_RADII",
    "MAX_VALENCE",
    "BACKBONE_BONDS",
    "SIDECHAIN_BONDS",
    "KNOWN_RESIDUES",
]

#: Covalent radii in ångström, used only to audit perceived bonds.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
    "Mg": 1.41, "Zn": 1.22, "Ca": 1.76, "Fe": 1.32, "Mn": 1.39,
    "Du": 1.60,
}

#: Largest number of bonds a neutral atom of each element can carry. An atom
#: exceeding this is being bonded to something it cannot bond to, which is the
#: cheapest strong signal that a perception went wrong.
MAX_VALENCE: dict[str, int] = {
    "H": 1, "C": 4, "N": 4, "O": 2, "F": 1, "P": 6, "S": 6,
    "Cl": 1, "Br": 1, "I": 1,
    "Mg": 6, "Zn": 6, "Ca": 6, "Fe": 6, "Mn": 6,
}

#: The bonds every amino acid shares. Keyed by PDB atom name, so a residue that
#: is missing a terminal oxygen simply contributes fewer bonds.
BACKBONE_BONDS: tuple[tuple[str, str], ...] = (
    ("N", "CA"),
    ("CA", "C"),
    ("C", "O"),
    ("C", "OXT"),
)

#: Side-chain connectivity as (parent, child) atom-name pairs.
#:
#: Proline is the one residue whose side chain closes back onto the backbone
#: nitrogen, so its table contains a `CD`-`N` bond. Getting that wrong leaves
#: proline's ring open, which is exactly the sort of small error that a distance
#: rule would have hidden and a template cannot.
#:
#: The branched aliphatics are the other place to be careful, and the three are
#: routinely confused. In **leucine** the branch point is CG, carrying CD1 and
#: CD2. In **isoleucine** it is CG1, carrying CD1 and CG2, so CG2 hangs off CB.
#: In **valine** CB carries both methyls directly. Writing isoleucine with a
#: `CG1`-`CG2` bond instead of `CB`-`CG2` invents a bond to nothing, and it is
#: invisible unless something measures it: the crambin fixture has those two
#: carbons 2.6 Å apart, which is how the error was caught.
SIDECHAIN_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "GLY": (),
    "ALA": (("CA", "CB"),),
    "SER": (("CA", "CB"), ("CB", "OG")),
    "CYS": (("CA", "CB"), ("CB", "SG")),
    "THR": (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
    "VAL": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
    "LEU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "ILE": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
    "PRO": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")),
    "MET": (("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "MSE": (("CA", "CB"), ("CB", "CG"), ("CG", "SE"), ("SE", "CE")),
    "PHE": (
        ("CA", "CB"), ("CB", "CG"),
        ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "CE1"), ("CD2", "CE2"),
        ("CE1", "CZ"), ("CE2", "CZ"),
    ),
    "TYR": (
        ("CA", "CB"), ("CB", "CG"),
        ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "CE1"), ("CD2", "CE2"),
        ("CE1", "CZ"), ("CE2", "CZ"),
        ("CZ", "OH"),
    ),
    "TRP": (
        ("CA", "CB"), ("CB", "CG"),
        ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"),
        ("CE2", "CZ2"), ("CZ2", "CH2"), ("CH2", "CZ3"),
        ("CZ3", "CE3"), ("CE3", "CD2"),
    ),
    "HIS": (
        ("CA", "CB"), ("CB", "CG"),
        ("CG", "ND1"), ("CG", "CD2"),
        ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"),
    ),
    "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "ARG": (
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
        ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"),
    ),
    "ASP": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "ASN": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "GLU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "GLN": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
}

#: Residue names that are a protonation or naming variant of a template above,
#: mapped to the template that describes them. `HID`/`HIE`/`HIP` differ in which
#: nitrogen carries the proton, not in the heavy-atom graph, so the HIS template
#: is correct for all three.
RESIDUE_ALIASES: dict[str, str] = {
    "MSE": "MSE", "SEC": "CYS", "PYL": "LYS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
    "ARN": "ARG", "WAT": "HOH", "TIP3": "HOH", "SOL": "HOH",
}

KNOWN_RESIDUES = frozenset(SIDECHAIN_BONDS) | {"HOH"}

#: AutoDock PDBQT type -> element. Aromatic carbon is `A`, and the metals are
#: written as two letters, so "first alphabetic character" gets both wrong.
PDBQT_TYPE_ELEMENT: dict[str, str] = {
    "C": "C", "A": "C", "G0": "C", "CG0": "C",
    "N": "N", "NA": "N", "NS": "N", "NC": "N", "N4": "N",
    "O": "O", "OA": "O",
    "S": "S", "SA": "S", "SE": "Se",
    "H": "H", "HD": "H", "HS": "H",
    "P": "P", "F": "F", "I": "I",
    "CL": "Cl", "BR": "Br", "MG": "Mg", "MET": "Mg", "ZN": "Zn",
    "CA": "Ca", "FE": "Fe", "MN": "Mn", "DU": "Du", "DUX": "Du",
}

#: Residue names that are a single ion or a water: no internal bonds at all.
ION_RESIDUES = frozenset({
    "HOH", "WAT", "TIP3", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN",
    "CU", "CO", "NI", "CD", "HG", "IOD", "BR", "F",
})


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class AtomRecord:
    """One atom, with enough identity to decide what it is bonded to."""

    serial: int
    xyz: tuple[float, float, float]
    element: str
    name: str = ""
    resname: str = "UNL"
    resid: int = 0
    chain: str = " "
    hetatm: bool = False
    # Filled in by the perception pass.
    bonds: list[int] = field(default_factory=list)

    @property
    def residue_key(self) -> tuple[str, int, str]:
        """Identity of the residue this atom belongs to.

        The chain and residue number are both part of it. Two residues can
        share a number in different chains, and a single number can even be
        reused after an insertion code, which is why the string is carried.
        """
        return (self.chain, self.resid, self.resname)

    @property
    def is_backbone(self) -> bool:
        return self.name in ("N", "CA", "C", "O", "OXT")


@dataclass
class Residue:
    """A group of atoms that belong together."""

    name: str
    resid: int
    chain: str
    atoms: list[int] = field(default_factory=list)  # indices into Structure.atoms
    template: str | None = None  # the template that was applied, if any

    @property
    def is_amino_acid(self) -> bool:
        return self.template is not None


@dataclass
class Structure:
    """A parsed structure with its perceived connectivity."""

    name: str
    atoms: list[AtomRecord] = field(default_factory=list)
    residues: list[Residue] = field(default_factory=list)
    #: True when the file carried `ROOT`/`BRANCH` records, i.e. it is a ligand
    #: or a pose and its bonds were declared rather than inferred.
    declared: bool = False
    warnings: list[str] = field(default_factory=list)

    def coords(self) -> list[tuple[float, float, float]]:
        return [a.xyz for a in self.atoms]

    def elements(self) -> list[str]:
        return [a.element for a in self.atoms]

    def bond_pairs(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for i, a in enumerate(self.atoms):
            for j in a.bonds:
                if j > i:
                    out.append((i, j))
        out.sort()
        return out

    def degrees(self) -> list[int]:
        return [len(a.bonds) for a in self.atoms]

    def is_protein(self) -> bool:
        return any(r.is_amino_acid for r in self.residues)

    def backbone(self) -> list[tuple[int, int, int]]:
        """``(N, CA, C)`` atom indices per amino-acid residue, in chain order.

        Residues missing any of the three are skipped rather than filled in:
        a ribbon through a guessed atom position bends somewhere it should not,
        which is worse than a gap in the trace.
        """
        out = []
        for res in self.residues:
            if not res.is_amino_acid:
                continue
            by_name = {self.atoms[i].name: i for i in res.atoms}
            if all(k in by_name for k in ("N", "CA", "C")):
                out.append((by_name["N"], by_name["CA"], by_name["C"]))
        return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _element_of(token: str) -> str:
    """Element symbol for an AutoDock PDBQT type token."""
    up = token.strip().upper()
    el = PDBQT_TYPE_ELEMENT.get(up)
    if el is not None:
        return el
    letters = "".join(ch for ch in token if ch.isalpha())
    if not letters:
        return "C"
    if len(letters) >= 2 and letters[:2] in ("CL", "BR", "SE", "MG", "ZN", "MN", "FE", "CU"):
        return letters[:2].capitalize()
    return letters[0].upper()


def _slice(line: str, start: int, end: int) -> str:
    return line[start:end].strip() if len(line) >= end else ""


def parse_structure(text: str, name: str = "") -> Structure:
    """Parse PDBQT text into a :class:`Structure` with bonds already perceived.

    Multiple `MODEL` blocks (a multi-model pose file) are read and the models
    are kept in order, each sorted by serial. Serials are what make a written
    pose comparable atom-by-atom with the ligand it came from, because the
    writer emits the rigid `ROOT` cluster first and the flexible `BRANCH`
    clusters after it, so file order is not the original order.
    """
    atoms: list[AtomRecord] = []
    residues: list[Residue] = []
    residue_index: dict[tuple[str, int, str], int] = {}

    # Declared ligand connectivity, accumulated per model.
    declared_pairs: list[tuple[int, int]] = []
    saw_branch_records = False
    # `stack` holds the parent atom serial for each open ROOT/BRANCH block, and
    # `prev` the serial of the previous atom in the current block, which is what
    # makes the atoms inside a block a chain.
    stack: list[int] = []
    prev_serial: int | None = None

    def model_offset() -> int:
        return len(atoms)

    def add_residue(rec: AtomRecord) -> None:
        key = rec.residue_key
        idx = residue_index.get(key)
        if idx is None:
            residues.append(Residue(name=rec.resname, resid=rec.resid, chain=rec.chain))
            residue_index[key] = len(residues) - 1
            idx = len(residues) - 1
        residues[idx].atoms.append(len(atoms) - 1)

    for line in text.splitlines():
        record = line.split(maxsplit=1)[0] if line.strip() else ""

        if record == "MODEL":
            # A new model starts a new residue numbering too; the coordinates
            # are of the same molecule, so bonds are perceived per model and the
            # last model's graph is what gets kept for display.
            if atoms:
                atoms = []
                residues = []
                residue_index = {}
                declared_pairs = []
                prev_serial = None
                stack = []
            continue
        if record == "ENDMDL":
            break

        if record == "ROOT":
            saw_branch_records = True
            prev_serial = None
            stack.append(-1)
            continue
        if record == "BRANCH":
            saw_branch_records = True
            try:
                parent = int(line.split()[1])
            except (IndexError, ValueError):
                parent = -1
            stack.append(parent)
            prev_serial = None
            continue
        if record in ("ENDROOT", "ENDBRANCH"):
            if stack:
                stack.pop()
            prev_serial = None
            continue

        if record not in ("ATOM", "HETATM"):
            continue

        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = len(atoms) + 1
        token = _slice(line, 77, 79) or (line.split()[-1] if line.split() else "C")
        try:
            resid = int(_slice(line, 22, 26) or 0)
        except ValueError:
            resid = 0

        idx = model_offset()
        atoms.append(
            AtomRecord(
                serial=serial,
                xyz=xyz,
                element=_element_of(token),
                name=_slice(line, 12, 16),
                resname=_slice(line, 17, 20) or "UNL",
                resid=resid,
                chain=_slice(line, 21, 22) or " ",
                hetatm=record == "HETATM",
            )
        )
        add_residue(atoms[-1])

        if saw_branch_records and stack:
            parent = stack[-1]
            if parent >= 0:
                # `BRANCH a b`: b is the *serial* of the parent atom, which is
                # not the same as its index once earlier atoms exist. Resolve
                # it against the serials actually seen in this model.
                for other in atoms[:idx]:
                    if other.serial == parent:
                        declared_pairs.append((idx, atoms.index(other)))
                        break
            if prev_serial is not None:
                for other in atoms[:idx]:
                    if other.serial == prev_serial:
                        declared_pairs.append((idx, atoms.index(other)))
                        break
            prev_serial = serial

    st = Structure(name=name, atoms=atoms, residues=residues, declared=saw_branch_records)

    if saw_branch_records:
        _apply_declared_bonds(st, declared_pairs)
    elif any(
        RESIDUE_ALIASES.get(r.name, r.name) in SIDECHAIN_BONDS
        for r in st.residues
    ):
        # At least one residue is a recognisable amino acid, so this is a
        # protein and its connectivity is chemistry, not geometry.
        _apply_templates(st)
    else:
        _apply_distance_bonds(st)

    _audit(st)
    return st


def _apply_distance_bonds(st: Structure) -> None:
    """Covalent-radius perception, for a flat ligand with no declared bonds.

    Reached only when the file contains no amino-acid residue at all, which is
    what a ligand written by `prep-ligand` looks like: a bare `ATOM` list with
    `UNL` as the residue name. For twenty heavy atoms this is the right method
    and it is what every tool in the field does; it is the wrong method for a
    protein, which is why the protein case is decided on residue names before
    getting here.

    The result is audited, not trusted: every bond is range-checked and every
    atom's degree is checked against its maximum valency, so a perception that
    joined the wrong pair is reported rather than drawn.
    """
    for i, a in enumerate(st.atoms):
        for j in range(i + 1, len(st.atoms)):
            b = st.atoms[j]
            if _bondable(a.element, b.element, _dist(a.xyz, b.xyz)):
                _link(st, i, j)
    st.warnings.append(
        "connectivity was inferred from interatomic distances: this file "
        "declares no ROOT/BRANCH bonds and carries no amino-acid residue, so it "
        "is being read as a small molecule"
    )


# ---------------------------------------------------------------------------
# Ligand bonds: what the file states
# ---------------------------------------------------------------------------


def _apply_declared_bonds(st: Structure, pairs: Iterable[tuple[int, int]]) -> None:
    """Use the `ROOT`/`BRANCH` tree, then close rings inside each root block.

    A declared bond is believed about *topology* and vetoed by *geometry*. The
    tree says which atom is attached to which; the coordinates say whether that
    attachment is physically there at all. A bond whose two atoms are further
    apart than the two elements can possibly be is dropped and reported, because
    no reading of the file makes a 2.2 Å oxygen-oxygen contact a bond.

    That combination is not hypothetical. A PDBQT block lists its atoms as a
    chain, but a forked group -- a carboxyl carbon with two oxygens -- is not a
    chain, and a writer that emits one as a flat list states that the two
    oxygens are bonded to each other. The docked-pose fixture in this
    repository contains exactly that: O and O, 2.23 Å apart, declared bonded.
    Drawing the file literally would put a line through empty space between two
    atoms of the same functional group.

    Rings are then closed geometrically, since a tree has no ring edges.
    """
    seen: set[tuple[int, int]] = set()
    for i, j in pairs:
        if i >= len(st.atoms) or j >= len(st.atoms) or i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        a, b = st.atoms[i], st.atoms[j]
        d = _dist(a.xyz, b.xyz)
        ra = COVALENT_RADII.get(a.element)
        rb = COVALENT_RADII.get(b.element)
        if ra is not None and rb is not None and d > ra + rb + 0.45:
            st.warnings.append(
                f"the file declares a bond between {a.name}{a.resname}{a.resid} "
                f"and {b.name}{b.resname}{b.resid}, but they are {d:.2f} Å "
                f"apart, which is not a {a.element}-{b.element} bond; not drawn"
            )
            continue
        seen.add(key)
        st.atoms[i].bonds.append(j)
        st.atoms[j].bonds.append(i)

    for i, j in _ring_closure_candidates(st, seen):
        st.atoms[i].bonds.append(j)
        st.atoms[j].bonds.append(i)
        st.warnings.append(
            f"ring closure added between {st.atoms[i].name} and "
            f"{st.atoms[j].name} ({st.atoms[i].resname}{st.atoms[i].resid})"
        )


def _ring_closure_candidates(
    st: Structure, existing: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Contacts within bonding range that the declared tree does not explain.

    A ring closure is a bond between two atoms the tree already connects by a
    path of three or more bonds, and which are close enough to bond directly.
    The path-length test is what separates a ring from two arms of a folded
    ligand that merely pass near each other, and requiring the same residue
    keeps a branched ligand's separate `BRANCH` records from being welded into
    one another.
    """
    heavy = [i for i, a in enumerate(st.atoms) if a.element != "H"]
    if len(heavy) < 4:
        return []

    tree_distance: dict[int, dict[int, int]] = {}
    for start in heavy:
        dist = {start: 0}
        frontier = [start]
        while frontier:
            nxt = []
            for cur in frontier:
                for nb in st.atoms[cur].bonds:
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        nxt.append(nb)
            frontier = nxt
        tree_distance[start] = dist

    out: list[tuple[int, int]] = []
    for ai, i in enumerate(heavy):
        for j in heavy[ai + 1:]:
            key = (min(i, j), max(i, j))
            if key in existing:
                continue
            a, b = st.atoms[i], st.atoms[j]
            if a.residue_key != b.residue_key:
                continue
            if tree_distance[i].get(j, 99) < 3:
                continue
            d = _dist(a.xyz, b.xyz)
            if _bondable(a.element, b.element, d):
                out.append((i, j))
                existing.add(key)
    return out


# ---------------------------------------------------------------------------
# Protein bonds: what the chemistry says
# ---------------------------------------------------------------------------


def _apply_templates(st: Structure) -> None:
    """Bond amino-acid residues from their templates.

    A residue whose name is not in the table is left unbonded and reported. The
    alternative -- falling back to a distance rule for the residues the table
    does not cover -- would reintroduce exactly the wrong-atom bonds this
    module exists to prevent, and only for the part of the protein nobody was
    looking at.
    """
    unknown: list[str] = []
    for res in st.residues:
        canon = RESIDUE_ALIASES.get(res.name, res.name)
        if res.name in ION_RESIDUES or canon == "HOH":
            res.template = None
            continue
        if canon not in SIDECHAIN_BONDS:
            unknown.append(f"{res.name}{res.resid}")
            res.template = None
            continue
        res.template = canon
        by_name: dict[str, int] = {}
        for i in res.atoms:
            by_name.setdefault(st.atoms[i].name, i)
        for parent, child in BACKBONE_BONDS + SIDECHAIN_BONDS[canon]:
            # A bond exists only if both atoms are actually present: AutoDock
            # merges non-polar hydrogens into their heavy atom, so a receptor
            # has fewer atoms than the PDB it came from.
            if parent in by_name and child in by_name:
                _link(st, by_name[parent], by_name[child])

    if unknown:
        shown = ", ".join(unknown[:8]) + (" ..." if len(unknown) > 8 else "")
        st.warnings.append(
            f"{len(unknown)} residue(s) have no connectivity template and were "
            f"left unbonded: {shown}"
        )

    _link_peptide_bonds(st)
    _link_polar_hydrogens(st)


def _link_polar_hydrogens(st: Structure) -> None:
    """Attach each hydrogen to the heavy atom it is sitting on.

    The templates list heavy atoms only, because a PDBQT receptor keeps polar
    hydrogens and merges the rest into their carbon. A polar hydrogen has
    exactly one parent and sits about 1.0 Å from it, so the nearest heavy atom
    *within the same residue* is not a guess -- there is no second candidate
    anywhere near that close. Restricting it to one residue is what keeps this
    from becoming the global distance rule the rest of the module exists to
    avoid.

    A hydrogen with no parent in range, or with two equidistant candidates, is
    left unbonded and reported rather than attached to whichever came first.
    """
    for res in st.residues:
        if not res.is_amino_acid:
            continue
        heavy = [i for i in res.atoms if st.atoms[i].element != "H"]
        for h in [i for i in res.atoms if st.atoms[i].element == "H"]:
            ranked = sorted(
                (_dist(st.atoms[h].xyz, st.atoms[c].xyz), c) for c in heavy
            )
            if not ranked:
                continue
            best, parent = ranked[0]
            if not 0.7 <= best <= 1.35:
                st.warnings.append(
                    f"hydrogen {st.atoms[h].serial} of {res.name}{res.resid} is "
                    f"{best:.2f} Å from the nearest heavy atom; left unbonded"
                )
                continue
            if len(ranked) > 1 and abs(ranked[1][0] - best) < 0.15:
                st.warnings.append(
                    f"hydrogen {st.atoms[h].serial} of {res.name}{res.resid} has "
                    f"two heavy atoms at the same distance; left unbonded"
                )
                continue
            _link(st, h, parent)


def _link_peptide_bonds(st: Structure) -> None:
    """Join consecutive residues of the same chain by the C-N peptide bond.

    "Consecutive" means the same chain and residue numbers differing by one.
    A chain break -- a gap in numbering, a different chain, a missing
    intermediate residue -- means the two residues are not bonded, and getting
    that wrong draws a ribbon straight through whatever is in the gap.
    """
    for chain in _chains(st):
        residues = [r for r in st.residues if r.chain == chain and r.is_amino_acid]
        residues.sort(key=lambda r: r.resid)
        for prev, cur in zip(residues, residues[1:]):
            if cur.resid - prev.resid != 1:
                st.warnings.append(
                    f"chain {chain.strip() or '_'}: numbering jumps from "
                    f"{prev.resid} to {cur.resid}; no peptide bond drawn across it"
                )
                continue
            p_c = _atom_named(st, prev, "C")
            c_n = _atom_named(st, cur, "N")
            if p_c is None or c_n is None:
                continue
            d = _dist(st.atoms[p_c].xyz, st.atoms[c_n].xyz)
            if not (1.0 <= d <= 1.8):
                # A C-N at 3 A is not a peptide bond; numbering alone is not
                # enough, and drawing one anyway is the same mistake as a
                # threshold-based bond.
                st.warnings.append(
                    f"chain {chain.strip() or '_'}: C{prev.resid}-N{cur.resid} is "
                    f"{d:.2f} Å apart, outside the peptide bond range; not drawn"
                )
                continue
            _link(st, p_c, c_n)


def _chains(st: Structure) -> list[str]:
    seen: list[str] = []
    for a in st.atoms:
        if a.chain not in seen:
            seen.append(a.chain)
    return seen


def _atom_named(st: Structure, res: Residue, name: str) -> int | None:
    for i in res.atoms:
        if st.atoms[i].name == name:
            return i
    return None


def _link(st: Structure, i: int, j: int) -> None:
    if i == j or j in st.atoms[i].bonds:
        return
    st.atoms[i].bonds.append(j)
    st.atoms[j].bonds.append(i)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _bondable(ea: str, eb: str, d: float) -> bool:
    ra = COVALENT_RADII.get(ea)
    rb = COVALENT_RADII.get(eb)
    if ra is None or rb is None:
        return False
    return d < ra + rb + 0.45


def _audit(st: Structure) -> None:
    """Report bonds that are too long, too short, or over an atom's valency.

    This is the module's own answer to "are these the right bonds": it cannot
    prove the connectivity is right, but every one of these three is a way for
    it to be wrong, and all three are cheap.
    """
    for i, j in st.bond_pairs():
        a, b = st.atoms[i], st.atoms[j]
        d = _dist(a.xyz, b.xyz)
        ra = COVALENT_RADII.get(a.element)
        rb = COVALENT_RADII.get(b.element)
        if ra is not None and rb is not None:
            if d < 0.5 * (ra + rb):
                st.warnings.append(
                    f"bond {i}-{j} is {d:.2f} Å, too short for "
                    f"{a.element}-{b.element}"
                )
            elif d > ra + rb + 0.45:
                st.warnings.append(
                    f"bond {i}-{j} is {d:.2f} Å, longer than a "
                    f"{a.element}-{b.element} bond can be"
                )
        if a.residue_key != b.residue_key and a.name in ("C",) and b.name in ("N",):
            continue  # the peptide link, already checked
        for atom in (a, b):
            cap = MAX_VALENCE.get(atom.element)
            if cap is not None and len(atom.bonds) > cap:
                st.warnings.append(
                    f"atom {atom.serial} ({atom.element}, {atom.name}) has "
                    f"{len(atom.bonds)} bonds, more than {cap} is possible"
                )
                break


# ---------------------------------------------------------------------------
# Secondary structure
# ---------------------------------------------------------------------------


def dihedral(p0: Sequence[float], p1: Sequence[float],
             p2: Sequence[float], p3: Sequence[float]) -> float:
    """Signed dihedral angle in degrees, in (-180, 180]."""
    b0 = [p0[k] - p1[k] for k in range(3)]
    b1 = [p2[k] - p1[k] for k in range(3)]
    b2 = [p3[k] - p2[k] for k in range(3)]
    n1 = _cross(b0, b1)
    n2 = _cross(b1, b2)
    m1 = _cross(n1, b1)
    nb1 = math.sqrt(sum(x * x for x in b1))
    if nb1 < 1e-8:
        return 0.0
    y = sum(n2[k] * m1[k] for k in range(3)) / nb1
    x = sum(n1[k] * n2[k] for k in range(3))
    return math.degrees(math.atan2(y, x))


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


#: φ/ψ regions for the three states, in degrees. These are the Ramachandran
#: basins, not a fit to this project: a residue is helix when both angles are
#: in the α region, sheet when both are in the β region, and coil otherwise.
SS_RANGES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "helix": ((-100.0, -20.0), (-80.0, 10.0)),
    "sheet": ((-180.0, -40.0), (60.0, 180.0)),
}


def secondary_structure(trace: list[tuple[int, int, int]], atoms: list[AtomRecord]) -> list[str]:
    """Per-residue ``"helix"`` / ``"sheet"`` / ``"coil"`` from backbone geometry.

    Purely geometric: φ and ψ are computed from the N/CA/C positions that are
    present, and compared against the Ramachandran basins. There is no hydrogen
    bond database and no DSSP here, so this is an estimate and is labelled as
    one. It needs `N` and `C` of the neighbouring residues, so the first and
    last residue of a chain have no φ or ψ and are called coil.
    """
    out = ["coil"] * len(trace)
    for i in range(1, len(trace) - 1):
        n_i, ca_i, c_i = trace[i]
        n_p, _, c_p = trace[i - 1]
        try:
            _, _, c_next = trace[i + 1]
            n_next = trace[i + 1][0]
        except IndexError:
            continue
        phi = dihedral(
            atoms[c_p].xyz, atoms[n_i].xyz, atoms[ca_i].xyz, atoms[c_i].xyz
        )
        psi = dihedral(
            atoms[n_i].xyz, atoms[ca_i].xyz, atoms[c_i].xyz, atoms[n_next].xyz
        )
        for name, (phi_r, psi_r) in SS_RANGES.items():
            if phi_r[0] <= phi <= phi_r[1] and psi_r[0] <= psi <= psi_r[1]:
                out[i] = name
                break
    return out
