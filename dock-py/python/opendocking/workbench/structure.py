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
    # Secondary structure. Public because the viewer draws its ribbon from
    # these and a caller has to be able to ask the same question the drawing
    # asked: `MoleculeView.backbone_ribbon` calls `secondary_structure`, so an
    # unexported one is a product behaviour with no supported way to inspect it.
    "secondary_structure",
    "geometric_secondary_structure",
    "backbone_hydrogen_bonds",
    "dihedral",
    "SS_RANGES",
    "SS_MIN_SHEET",
    "HELIX_TURNS",
    "HBOND_STRICT",
    "HBOND_LOOSE",
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
    """Signed dihedral angle in degrees, in (-180, 180].

    The sign is the convention every torsion angle in the field uses: an
    alpha helix comes out with phi and psi both negative, and a fully eclipsed
    quadruple comes out at 0. The `x` term below is negated to get there.

    It was not, and the effect was not subtle. This returned `180 - torsion`,
    which is its own mirror, so a right-handed alpha helix was reported as a
    left-handed one: crambin residue 8, which is in the first helix of a
    structure whose coordinates put its O(i)...N(i+4) at 2.89 A, was reported as
    phi = -124, psi = -135 -- the beta region, not the alpha one. Every residue
    of both of crambin's helices therefore fell outside `SS_RANGES` and the
    purely geometric estimate called almost the whole protein coil. Two
    geometries fix that: the eclipsed quadruple
    ``(0,0,0), (1,0,0), (1,1,0), (0,1,0)`` has a torsion of 0 and used to
    return 180.
    """
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
    x = -sum(n1[k] * n2[k] for k in range(3))
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


def geometric_secondary_structure(
    trace: list[tuple[int, int, int]], atoms: list[AtomRecord]
) -> list[str]:
    """Per-residue state from φ and ψ alone -- the answer this module had before.

    Kept, named and reachable on purpose. It is the floor the hydrogen-bond pass
    in :func:`secondary_structure` claims to beat, and a claim like that is only
    worth anything if the worse answer is still available to be compared with.
    Pass ``use_hydrogen_bonds=False`` to :func:`secondary_structure` to get
    exactly this.

    What it cannot do, and why it is kept rather than deleted:

    * It scores each residue *independently*, from its own φ and ψ. Real
      secondary structure is a property of a **segment**: a four-residue helix
      whose two end residues happen to fall outside the α basin comes out
      ``coil helix helix coil``, and a two-residue helix comes out as nothing at
      all. There is no way to fix that from φ/ψ, because the information that
      makes those residues part of one helix is not in their own dihedrals.
    * It cannot see a β-sheet at all. A sheet is a property of *two* strands and
      the hydrogen bonds between them, and no per-residue angle contains it.

    It needs `N` and `C` of the neighbouring residues, so the first and last
    residue of a chain have no φ or ψ and are called coil.
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


# ---------------------------------------------------------------------------
# Hydrogen bonds: the evidence secondary structure is actually made of
# ---------------------------------------------------------------------------

#: Length of an amide N-H, used when the hydrogen has to be placed rather than
#: read. DSSP's own amide H sits 1.0 A from the nitrogen.
AMIDE_H_LENGTH = 1.0

#: Kabsch & Sander's electrostatic energy of a backbone hydrogen bond, as a
#: constant pair-charge product in kcal/mol/A^2. The energy is
#: `HBOND_Q * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)`, i.e. kcal/mol when the four
#: distances are in angstrom. This is DSSP's own measure, kept because the two
#: cancelling short terms (C-H attraction against O-H repulsion) are what
#: separate a hydrogen bond from two heavy atoms that merely happen to be close;
#: a bare distance test cannot reproduce them.
HBOND_Q = 0.084 * 332.0

#: Acceptance criteria, as `(energy_max, no_max, ho_max, angle_min)`: the
#: Kabsch-Sander energy, the N...O distance, the H...O distance, and the
#: N-H...O angle. `HBOND_STRICT` is DSSP's own set (energy below -0.5 kcal/mol,
#: N...O under 3.5 A, H...O under 2.5 A, angle above 90 degrees).
HBOND_STRICT: tuple[float, float, float, float] = (-0.5, 3.5, 2.5, 90.0)
#: The relaxed pass is **not** DSSP and is not presented as such. It exists for
#: one job: a real helix is a *run* of turns, and a single turn that misses
#: because one bond is 3.6 A instead of 3.4 A, or because the input has no
#: carbonyl oxygen on one residue, would otherwise cut the run in two. A loose
#: bond is therefore allowed to *bridge* two segments that are already there,
#: but never to start one -- otherwise the same loose test would invent helices
#: out of chance contacts in a loop.
HBOND_LOOSE: tuple[float, float, float, float] = (0.0, 3.9, 2.8, 80.0)

#: CA-CA separation beyond which an N-H...O=C pair is impossible, so the pair
#: never has to be looked at. The bound is the triangle inequality, not a fit:
#: CA...CA <= CA...N (1.46) + N...O (3.9, the loosest gate there is) + O...C
#: (1.23) + C...CA (1.53) = 8.12 A.
HBOND_CA_SKIP = 8.2

#: Edge length of one cell in the candidate index, in angstrom.
#:
#: It only has to be *at least* `HBOND_CA_SKIP`: a pair within the cutoff then
#: cannot be more than one cell apart along any axis, so the 27 cells around a
#: residue are guaranteed to contain every candidate. 9.0 rather than 8.2 so that
#: the boundary case -- a pair exactly 8.2 A apart straddling a cell face -- is
#: not one rounding error away from being missed.
#:
#: Making this 8.2 A of scanning work O(N) instead of O(N^2) is not a
#: micro-optimisation. Measured on a 15-strand bundle of 200-residue helices
#: packed 10 A apart (1400 residues, a compact stand-in): the unindexed scan took
#: **9.5 s**, of which 97.5% was `_hbonds` and 3.95 M calls to `_dist`. A single
#: 3000-residue helix was 4.9 s only because an extended chain has few near
#: pairs; the same 3000 residues packed into a bundle took **47 s**. Which
#: number matters depends on the shape of the protein, so the packed one is the
#: one to hold a fix to.
HBOND_CELL = 9.0

#: Smallest run of residues accepted as a beta strand. DSSP's own minimum is
#: two, which is also the smallest that a ribbon can draw as a strand: a lone
#: bridged residue is one contact, not a strand, and is left coil.
SS_MIN_SHEET = 2

#: Residue separations that make a helix: 3, 4 and 5 residues apart are DSSP's
#: 3-10, alpha and pi helices. A bond from residue `i` to residue `i+n` makes the
#: residues of that turn helical, which is the whole point -- a helix is a
#: segment, not one independent vote per residue. Only the separations are used
#: here; the three-letter names are the caller's business, and this function
#: returns "helix" for all three.
HELIX_TURNS: tuple[int, ...] = (3, 4, 5)


def _norm(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-9:
        return [0.0, 0.0, 0.0]
    return [x / n for x in v]


def _angle_deg(
    a: Sequence[float], b: Sequence[float], c: Sequence[float]
) -> float:
    """Angle a-b-c in degrees."""
    u = _norm([a[k] - b[k] for k in range(3)])
    v = _norm([c[k] - b[k] for k in range(3)])
    dot = max(-1.0, min(1.0, sum(u[k] * v[k] for k in range(3))))
    return math.degrees(math.acos(dot))


@dataclass
class _BackboneFrame:
    """What the hydrogen-bond pass needs to know about one traced residue.

    `key` carries the chain and residue number, which is how a chain break is
    told apart from a residue that simply happens to be the last one before it:
    `trace` is a flat list, so position `i - 1` is not necessarily the residue
    the peptide bond to `i` comes from.
    """

    key: tuple[str, int, str]
    n: int
    ca: int
    #: The carbonyl oxygen this residue donates from, if the file has one.
    carbonyl_o: int | None = None
    #: Where the amide hydrogen is. Read from the file when there is a hydrogen
    #: actually bonded to this N (a prepared receptor keeps polar hydrogens),
    #: otherwise placed. `None` when there is no preceding carbonyl to place it
    #: against -- a chain's first residue, and any residue after a break.
    amide_h: tuple[float, float, float] | None = None
    #: The C=O carbon of the residue before this one, same chain. Needed both to
    #: place the hydrogen and for the C-H term of the energy.
    prev_c: int | None = None


def _backbone_frames(
    trace: list[tuple[int, int, int]], atoms: list[AtomRecord]
) -> list[_BackboneFrame]:
    """One :class:`_BackboneFrame` per traced residue, in `trace` order.

    Assumes a trans peptide, because that is what the placed hydrogen assumes.
    For a trans peptide the amide hydrogen sits anti to the preceding C=O
    across the peptide bond, which is the fourth tetrahedral direction at the
    nitrogen; :func:`_placed_amide_h` puts it there. A cis peptide would place it
    on the wrong side of the nitrogen. Nothing here detects one, and
    `prep-receptor` does not produce them.

    `prev_c` is only set for a residue whose preceding carbonyl is reachable: a
    chain's first residue, and any residue after a chain break or a peptide bond
    that parsing refused, get `None` and so cannot donate.
    """
    oxy: dict[tuple, dict[str, int]] = {}
    h_atoms: dict[tuple, list[int]] = {}
    for idx, a in enumerate(atoms):
        if a.element == "O" and a.name in ("O", "OXT"):
            oxy.setdefault(a.residue_key, {}).setdefault(a.name, idx)
        elif a.element == "H" and a.name in ("H", "HN", "H1", "H2", "H3"):
            h_atoms.setdefault(a.residue_key, []).append(idx)

    frames: list[_BackboneFrame] = []
    for n_idx, ca_idx, _c_idx in trace:
        key = atoms[n_idx].residue_key
        table = oxy.get(key, {})
        frame = _BackboneFrame(
            key=key, n=n_idx, ca=ca_idx,
            carbonyl_o=table.get("O", table.get("OXT")),
        )
        # A hydrogen only counts if it is actually bonded to this nitrogen: a
        # side-chain amine hydrogen on the same residue would otherwise be
        # mistaken for the amide one.
        for h_idx in h_atoms.get(key, ()):
            if h_idx in atoms[n_idx].bonds:
                frame.amide_h = atoms[h_idx].xyz
                break
        frames.append(frame)

    for i, frame in enumerate(frames):
        if i == 0:
            continue
        prev = frames[i - 1]
        if prev.key[0] != frame.key[0] or frame.key[1] - prev.key[1] != 1:
            continue  # a chain break: no preceding carbonyl to work from
        c_prev = trace[i - 1][2]
        n = atoms[frame.n].xyz
        if not 1.0 <= _dist(atoms[c_prev].xyz, n) <= 1.8:
            continue  # parsing refused this peptide bond; do not invent a link
        frame.prev_c = c_prev
        if frame.amide_h is None:
            frame.amide_h = _placed_amide_h(atoms, frame, c_prev)
    return frames


def _placed_amide_h(
    atoms: list[AtomRecord], frame: _BackboneFrame, c_prev: int
) -> tuple[float, float, float]:
    """Where the amide hydrogen of `frame` is, when the file does not say.

    The nitrogen has three substituents and two of them are known here: the
    preceding carbonyl carbon and this residue's own CA. The hydrogen takes the
    fourth, approximately tetrahedral direction -- the negative bisector of the
    two known bonds -- and for the planar amide of a *trans* peptide that
    direction is anti to the preceding C=O, which is the position DSSP's
    hydrogen occupies and the one an alpha helix is built on.

    The obvious alternative, extending C(i-1) -> N(i) by 1 A, is wrong and looks
    right: it puts the hydrogen *on* the peptide-bond axis, where the N-H...O=C
    angle can never be favourable, and it silently destroys every hydrogen bond
    in the chain. On crambin, read without its hydrogens, that construction
    finds 7 backbone bonds where this one finds 24.
    """
    n = atoms[frame.n].xyz
    to_prev = _norm([atoms[c_prev].xyz[k] - n[k] for k in range(3)])
    to_ca = _norm([atoms[frame.ca].xyz[k] - n[k] for k in range(3)])
    direction = _norm([-(to_prev[k] + to_ca[k]) for k in range(3)])
    return tuple(n[k] + AMIDE_H_LENGTH * direction[k] for k in range(3))


def backbone_hydrogen_bonds(
    trace: list[tuple[int, int, int]],
    atoms: list[AtomRecord],
    *,
    relaxed: bool = False,
) -> set[tuple[int, int]]:
    """Backbone N-H...O=C hydrogen bonds, as ``(donor, acceptor)`` trace positions.

    The donor is the residue whose amide N-H points at the acceptor residue's
    carbonyl, so an alpha helix shows up as ``(i + 4, i)`` and a strand of a
    sheet as two bonds pointing in opposite directions.

    The hydrogen is read from the file when the file has one, and placed
    geometrically when it does not -- see :func:`_placed_amide_h`. A prepared
    receptor keeps its polar hydrogens; a raw PDB or a hand-written backbone
    usually has none. On crambin the two routes give the *same* 24 bonds, which
    is the check that the placement is sound: the hydrogens in the prepared file
    were positioned by RDKit's optimiser and this module never sees them.

    `relaxed=True` swaps in the looser criteria in :data:`HBOND_LOOSE`. Those
    are this module's own, not DSSP's, and are meant only for bridging two
    already-accepted segments -- see the note on :data:`HBOND_LOOSE`.
    """
    frames = _backbone_frames(trace, atoms)
    return _hbonds(
        frames, atoms, relaxed, _candidate_pairs(frames, atoms)
    )


def _candidate_pairs(
    frames: list[_BackboneFrame], atoms: list[AtomRecord]
) -> list[tuple[int, int]]:
    """``(donor, acceptor)`` trace positions close enough to be worth testing.

    The cutoff lives here and nowhere else, so that the indexed walk and the
    plain double loop filter on exactly the same predicate and differ only in
    how they find the pairs. That is what makes the two interchangeable: the
    index decides *which pairs to look at*, never *which pairs pass*.

    The index is a cell list over the donor CA atoms with an edge of
    :data:`HBOND_CELL`, searched from each acceptor through the 27 cells around
    it. Every candidate is then still measured, because a cell is a box and the
    cutoff is a sphere: cells 9 A apart hold pairs up to 12.7 A apart, and those
    have to be rejected on distance rather than assumed away.
    """
    donors = [
        i for i, f in enumerate(frames)
        if f.amide_h is not None and f.prev_c is not None
    ]
    acceptors = [i for i, f in enumerate(frames) if f.carbonyl_o is not None]
    if not donors or not acceptors:
        return []

    cells: dict[tuple[int, int, int], list[int]] = {}
    donor_ca: dict[int, tuple] = {}
    for don in donors:
        ca = atoms[frames[don].ca].xyz
        donor_ca[don] = ca
        key = (
            int(ca[0] // HBOND_CELL),
            int(ca[1] // HBOND_CELL),
            int(ca[2] // HBOND_CELL),
        )
        bucket = cells.get(key)
        if bucket is None:
            cells[key] = [don]
        else:
            bucket.append(don)

    out: list[tuple[int, int]] = []
    edge = HBOND_CELL
    for acc in acceptors:
        ca_a = atoms[frames[acc].ca].xyz
        bx = int(ca_a[0] // edge)
        by = int(ca_a[1] // edge)
        bz = int(ca_a[2] // edge)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for don in cells.get((bx + dx, by + dy, bz + dz), ()):
                        if don == acc:
                            continue
                        if _dist(ca_a, donor_ca[don]) > HBOND_CA_SKIP:
                            continue
                        out.append((don, acc))
    return out


def _hbonds(
    frames: list[_BackboneFrame],
    atoms: list[AtomRecord],
    relaxed: bool,
    candidates: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """The bonds one pass accepts, from a candidate list somebody else built.

    `candidates` comes from :func:`_candidate_pairs` and is passed in rather
    than computed here because `relaxed` does not reach it: the candidate test
    is the `HBOND_CA_SKIP` bound, which is derived from the *loosest* gate
    there is (see :data:`HBOND_CA_SKIP`), so one list of pairs is valid for both
    passes and the two passes then differ only in the four thresholds on the
    first line. `relaxed` is read in exactly one place in this module, and it
    is that line.

    The list is a required argument rather than an optional one with a default
    that builds it, so that a caller cannot quietly get a fresh enumeration per
    pass -- which is what this used to do, and it built the same list twice:
    295,978 distance measurements per pass on a 5200-residue bundle, half of
    them identical to the other half.
    """
    e_max, no_max, ho_max, angle_min = HBOND_LOOSE if relaxed else HBOND_STRICT
    out: set[tuple[int, int]] = set()
    for don, acc in candidates:
        df = frames[don]
        fr = frames[acc]
        o = atoms[fr.carbonyl_o].xyz
        n = atoms[df.n].xyz
        h = df.amide_h
        d_on = _dist(o, n)
        if d_on > no_max:
            continue
        d_oh = _dist(o, h)
        if d_oh > ho_max:
            continue
        if _angle_deg(n, h, o) < angle_min:
            continue
        c_prev = atoms[df.prev_c].xyz
        d_cn = _dist(c_prev, n)
        d_ch = _dist(c_prev, h)
        if min(d_on, d_oh, d_cn, d_ch) < 0.2:
            continue  # coincident atoms: the energy would be meaningless
        energy = HBOND_Q * (1.0 / d_on + 1.0 / d_ch - 1.0 / d_oh - 1.0 / d_cn)
        if energy <= e_max:
            out.add((don, acc))
    return out


def _peptide_neighbours(a: _BackboneFrame, b: _BackboneFrame) -> bool:
    """True when `b` is the residue the peptide bond from `a` leads to."""
    return a.key[0] == b.key[0] and b.key[1] - a.key[1] == 1


def _contiguous(frames: list[_BackboneFrame], lo: int, hi: int) -> bool:
    """True when `frames[lo] .. frames[hi]` is one unbroken stretch of peptide."""
    return all(
        _peptide_neighbours(frames[i - 1], frames[i]) for i in range(lo + 1, hi + 1)
    )


def _runs(indices: set[int]) -> list[list[int]]:
    """Sorted indices as maximal runs of consecutive values."""
    out: list[list[int]] = []
    for i in sorted(indices):
        if out and i == out[-1][-1] + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _extend(runs: list[list[int]], spans: list[tuple[int, int]]) -> list[list[int]]:
    """Join existing runs with the `spans` that bridge two of them.

    A span is used only when it touches **two or more** existing runs. That is
    the whole difference between a relaxed bond smoothing a segment and a
    relaxed bond inventing one:

    * a span touching nothing starts nothing, so a chance contact in a loop can
      never become a helix;
    * a span touching exactly one run would *extend* that run past its own
      evidence, which is how a nine-residue ideal helix came out with its
      ninth residue helical: the 3-turn from residue 9 reaches back to residue 6
      at a real 3.2 A, misses the strict H...O gate by 0.2 A, and would have
      pushed the segment one residue further than its 4-turns support;
    * a span bridging two runs is the case the relaxed pass exists for -- one
      turn missed, or slightly too long, in the middle of a segment that is
      otherwise there on both sides.
    """
    covered = {i for run in runs for i in run}
    changed = True
    while changed:
        changed = False
        for lo, hi in spans:
            touching = [
                run for run in runs if lo <= run[-1] + 1 and hi >= run[0] - 1
            ]
            if len(touching) < 2:
                continue
            new = set(range(lo, hi + 1)) | covered
            if new != covered:
                covered = new
                runs = _runs(covered)
                changed = True
    return runs


def _helix_segments(
    frames: list[_BackboneFrame],
    strict: set[tuple[int, int]],
    loose: set[tuple[int, int]],
) -> list[list[int]]:
    """Residue indices that belong to a helix, as a list of segments.

    A bond from residue `i` to residue `i+n` is a *turn*, and a turn makes the
    residues of that turn helical. That is DSSP's rule -- its 3-turn, 4-turn and
    5-turn are the 3-10, alpha and pi helices -- and it is what fixes the
    short-helix failure: the evidence for a short helix is one turn bond, and it
    covers the whole turn, not just the residues whose own phi and psi happen to
    sit in the alpha basin.

    The turn covers the *second and following* residue, which is DSSP's wording
    and is worth keeping: the CO of residue `i` is the first residue of the turn
    and is left to whatever comes before it. Marking it would extend every
    helix by one residue at each end, and on crambin the extra residues are
    6 and 22, which are not helical.
    """
    def turn_spans(bonds: set[tuple[int, int]]) -> list[tuple[int, int]]:
        spans = []
        for donor, acceptor in bonds:
            lo, hi = sorted((donor, acceptor))
            size = hi - lo
            if not any(size == n for n in HELIX_TURNS):
                continue
            if not _contiguous(frames, lo, hi):
                continue  # a chain break is not a turn
            spans.append((lo + 1, hi))
        return spans

    runs = _runs({i for lo, hi in turn_spans(strict) for i in range(lo, hi + 1)})
    return _extend(runs, turn_spans(loose))


def _sheet_segments(
    frames: list[_BackboneFrame],
    strict: set[tuple[int, int]],
    loose: set[tuple[int, int]],
    helix: list[list[int]],
) -> list[list[int]]:
    """Residue indices that belong to a beta strand, as a list of segments.

    This is the part phi and psi genuinely cannot do, so it is the part worth
    measuring. A bond is sheet evidence only when its two residues are *not*
    adjacent in sequence -- same chain and within two residues -- because a bond
    between neighbours is a turn, not a bridge; *and* not a turn apart either
    (3, 4 or 5), because a turn has already been read as helix and the two
    readings cannot both be right. A bond whose two residues are both inside one
    already-accepted helix segment is likewise helix evidence that has already
    been spent, and is not counted twice.

    Two simplifications against DSSP, both of them deliberate:

    * DSSP requires a *bridge*, which is two consecutive contacts; a lone
      contact is DSSP's `B` and this counts it too. Crambin is why: its
      two-stranded sheet has exactly two inter-strand hydrogen bonds, 35 -> 1
      and 33 -> 3, and they are two residues apart on each strand, so they are
      neither consecutive bridges nor a ladder under DSSP's own definition.
      Requiring consecutiveness finds no sheet at all in a real structure.
    * DSSP's `E` label covers a strand residue that is not itself bridged. There
      is no bond to find for those, so they are recovered structurally instead:
      a residue with a bridged residue on either side of it, in the same chain,
      is in the same strand. That is also why nothing is extended past the
      outermost bridge -- on crambin the residue after residue 35 is a loop
      residue, not a strand.
    """
    helix_of = {i: k for k, run in enumerate(helix) for i in run}

    def bridges(bonds: set[tuple[int, int]]) -> set[tuple[int, int]]:
        out = set()
        for a, b in bonds:
            lo, hi = sorted((a, b))
            size = hi - lo
            if lo == hi:
                continue
            fa, fb = frames[lo], frames[hi]
            if fa.key[0] == fb.key[0] and fb.key[1] - fa.key[1] <= 2:
                continue  # adjacent in sequence: a turn, not a bridge
            if any(size == n for n in HELIX_TURNS):
                # A bond 3, 4 or 5 residues apart is a *turn*, and a turn is
                # helix evidence: it has already put both its residues inside a
                # helix segment, and helix wins. Re-reading it as a bridge can
                # only contradict that. It is the rule that stops a permissive
                # 4-turn in a loop from being read as a strand.
                continue
            if lo in helix_of and helix_of.get(hi) == helix_of[lo]:
                continue  # already spent as a helix turn
            out.add((lo, hi))
        return out

    def strands(bonds: set[tuple[int, int]]) -> set[int]:
        bridged: set[int] = set()
        for lo, hi in bridges(bonds):
            bridged.update((lo, hi))
        # Residues interior to a bridged stretch of one strand.
        interior = set()
        for i in range(len(frames)):
            if i in bridged or i == 0 or i + 1 >= len(frames):
                continue
            if not _peptide_neighbours(frames[i - 1], frames[i]):
                continue
            if not _peptide_neighbours(frames[i], frames[i + 1]):
                continue
            if (i - 1) in bridged and (i + 1) in bridged:
                interior.add(i)
        return bridged | interior

    runs = [run for run in _runs(strands(strict)) if len(run) >= SS_MIN_SHEET]
    # A sheet is not a single event either, so a loose bridge may join two
    # strands that are already there, but it may not begin one. Only whole
    # bridges are offered: extending a strand residue by residue lets one
    # permissive contact walk a run across a loop.
    loose_bridges = bridges(loose) - bridges(strict)
    runs = _extend(runs, sorted(loose_bridges))
    return [run for run in runs if len(run) >= SS_MIN_SHEET]


def secondary_structure(
    trace: list[tuple[int, int, int]],
    atoms: list[AtomRecord],
    use_hydrogen_bonds: bool = True,
) -> list[str]:
    """Per-residue ``"helix"`` / ``"sheet"`` / ``"coil"``, from hydrogen bonds.

    The evidence is the same evidence DSSP uses, and the parts of the method
    that are DSSP's are these:

    * the amide hydrogen is placed 1.0 A from N on the fourth tetrahedral
      direction when the file has none, which for a trans peptide is where
      DSSP's hydrogen is;
    * a bond is accepted on the Kabsch-Sander energy (< -0.5 kcal/mol) together
      with the N...O, H...O and N-H...O angle gates;
    * a helix is a run of turns -- 3, 4 or 5 residues apart, the 3-10, alpha and
      pi helices -- and every residue of a turn is helical, not just the ones
      whose own phi and psi are in the alpha basin;
    * a sheet needs a hydrogen bond between residues that are not adjacent in
      sequence, which is a property of two strands and no per-residue angle can
      see it.

    What it is **not**: this is not DSSP and does not try to be. There is no
    residue competition (DSSP gives a residue to the higher-priority hydrogen
    bond, so a residue at a helix edge beside a strand is not counted twice),
    no electrostatic field beyond the four distances above, no side-chain or
    water bridges, and no turn-angle test. It reads heavy atoms and, when the
    file has no hydrogens, a *constructed* hydrogen. On a structure with
    hydrogens it should be close to DSSP; on one without, it is an estimate and
    is labelled as one. The two simplifications that most affect the answer --
    a lone inter-strand contact counting as a sheet, and the `E` strand residues
    being recovered structurally rather than from bonds -- are documented on
    :func:`_sheet_segments`.

    Where the hydrogen-bond evidence says nothing, the residue is coil. It is
    **not** given back to the geometric estimate: that is what puts spurious
    helices and strands all over a loop, and on crambin the geometric answer
    alone disagrees with the documented structure on 13 of 46 residues, most of
    them in the loop. The one exception is a structure where the pass found no
    bond at all -- a backbone with no carbonyl oxygens, say -- which degrades to
    the geometric estimate rather than to "everything is coil". Helix wins over
    sheet where both apply, because a 3-10 helix also has bonds that would
    otherwise read as a short strand.

    Pass ``use_hydrogen_bonds=False`` for the old purely geometric estimate.
    """
    if not use_hydrogen_bonds:
        return geometric_secondary_structure(trace, atoms)

    frames = _backbone_frames(trace, atoms)
    states = ["coil"] * len(trace)
    # One enumeration, two passes. The candidate list depends on the geometry
    # and on nothing else -- not on `relaxed`, not on the thresholds -- so the
    # strict pass and the loose pass are handed the same one and differ only in
    # what they then accept. Enumerating it per pass doubled the cost of the
    # most expensive part of the whole assignment for no difference in the
    # result; the counts are in `scripts/structure_bond_check.py`, which asserts
    # that the two passes still see the same list.
    candidates = _candidate_pairs(frames, atoms)
    strict = _hbonds(frames, atoms, relaxed=False, candidates=candidates)
    loose = _hbonds(frames, atoms, relaxed=True, candidates=candidates)
    if not strict:
        return geometric_secondary_structure(trace, atoms)

    helix = _helix_segments(frames, strict, loose)
    for run in helix:
        for i in run:
            states[i] = "helix"
    for run in _sheet_segments(frames, strict, loose, helix):
        for i in run:
            if states[i] == "coil":
                states[i] = "sheet"
    return states
