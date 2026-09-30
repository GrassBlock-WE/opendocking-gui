"""Chemistry-aware structure preparation using RDKit.

The docking engine deliberately knows nothing about chemistry beyond PDBQT atom
typing: it reads tables of elements, charges and coordinates. Everything
*chemical* — aromaticity perception, protonation state, Gasteiger charges,
non-polar hydrogen merging, which nitrogens are hydrogen-bond acceptors —
happens here, where RDKit can do it properly.

This split is deliberate. Re-deriving aromaticity from coordinates inside the
engine would mean guessing, and the AutoDock conventions for typing a hydroxyl
oxygen or an amidic nitrogen are conventions, not chemistry.

# Ligand preparation pipeline

1. read the structure (SDF / MOL2 / MOL / SMILES / PDB);
2. sanitise it, and if that fails try a partial sanitisation that keeps the
   geometry;
3. **merge non-polar hydrogens** into their heavy atoms — AutoDock's scoring
   function has no term for them and keeping them triples the atom count;
4. add back **polar hydrogens** on genuine donors, which is what lets the
   engine see hydrogen bonds at all;
5. assign PDBQT atom types from the RDKit perception, not from the element
   alone, so a tertiary amine is typed ``NA`` rather than ``N``;
6. compute Gasteiger charges.

# Receptor preparation pipeline

Receptors need a different treatment: they must be **rigid**, so the engine
receives only polar hydrogens, and they are usually too large for RDKit's
perception to be worth running on the whole structure. Waters, ions and
disconnected fragments are reported rather than silently kept.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = [
    "prepare_ligand",
    "prepare_receptor",
    "pdbqt_atom_type",
    "load_molecule",
    "SUPPORTED_LIGAND_FORMATS",
    "SUPPORTED_RECEPTOR_FORMATS",
]

#: Formats ``prepare_ligand`` can read.
SUPPORTED_LIGAND_FORMATS = (".sdf", ".mol", ".mol2", ".smi", ".smiles", ".pdb", ".xyz")

#: Formats ``prepare_receptor`` can read.
SUPPORTED_RECEPTOR_FORMATS = (".pdb", ".pdbqt", ".ent")

# Elements RDKit should keep as explicit atoms after hydrogen merging.
_PT_METALS = {"Mg", "Zn", "Fe", "Ca", "Mn", "Cu", "Ni", "Co", "Cd", "Hg", "Na", "K"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_molecule(path: str | Path, *, sanitize: bool = True):
    """Read a molecule from disk with RDKit.

    Parameters
    ----------
    path:
        A ``.sdf``, ``.mol``, ``.mol2``, ``.pdb`` or ``.smi`` file. The
        ``.smi`` reader takes the first molecule in the file.
    sanitize:
        Run RDKit's full sanitisation. Turn this off (the code below then
        retries with a partial sanitisation) for files that are geometrically
        fine but chemically odd — a truncated PDB ligand, say.
    """
    from rdkit import Chem, RDLogger

    # RDKit is chatty about recoverable problems; we report them ourselves.
    RDLogger.DisableLog("rdApp.*")

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_LIGAND_FORMATS:
        raise ValueError(
            f"unsupported format {suffix!r}; expected one of "
            f"{', '.join(SUPPORTED_LIGAND_FORMATS)}"
        )

    mol = None
    if suffix in (".smi", ".smiles"):
        with open(path, "r", encoding="utf-8") as handle:
            line = handle.readline().strip()
        if not line:
            raise ValueError(f"{path} is empty")
        mol = Chem.MolFromSmiles(line.split()[0], sanitize=sanitize)
    else:
        if suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), sanitize=sanitize, removeHs=False)
        elif suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=False)
            if len(supplier) == 0:
                raise ValueError(f"no molecules found in {path}")
            mol = supplier[0]
        else:
            mol = Chem.MolFromPDBFile(str(path), sanitize=sanitize, removeHs=False)

    if mol is None:
        # Second chance: keep the geometry, accept a partial sanitisation.
        mol = _load_unsanitised(path, suffix)
    if mol is None:
        raise ValueError(f"RDKit could not read a molecule from {path}")
    return mol


def _load_unsanitised(path: Path, suffix: str):
    """Retry a load without sanitisation, then sanitise what can be sanitised."""
    from rdkit import Chem

    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        mol = supplier[0] if len(supplier) else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    elif suffix in (".pdb",):
        mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    else:
        return None
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            catchErrors=True,
        )
    except Exception:  # pragma: no cover - RDKit raises many types
        pass
    return mol


# ---------------------------------------------------------------------------
# PDBQT atom typing
# ---------------------------------------------------------------------------


def pdbqt_atom_type(atom, mol) -> str:
    """Return the AutoDock atom type for one RDKit atom.

    The mapping follows Meeko and the PDBQT specification:

    ==================  =====  =============================================
    situation           type   note
    ==================  =====  =============================================
    polar hydrogen      ``HD`` carries the donor information for its partner
    acceptor nitrogen   ``NA`` includes pyridine N, tertiary amines, nitro N
    donor+acceptor N    ``NDA`` imidazole, pyrazole
    other nitrogen      ``N``   amide, pyrrole, quaternary
    acceptor oxygen     ``OA`` carbonyl, ether, alcohol
    other oxygen        ``O``   protonated
    aromatic carbon     ``A``
    aliphatic carbon    ``C``
    ==================  =====  =============================================

    Nitrogen typing is the part that matters most. Meeko and the classic
    AutoDock pipeline type a *tertiary amine* as ``N``, which the Vina scoring
    function then treats as apolar — so a ligand's key basic nitrogen would
    score like a methylene. RDKit already knows the difference, so we use its
    perception.
    """
    symbol = atom.GetSymbol()
    if symbol == "H":
        # Only polar hydrogens survive into a prepared ligand.
        return "HD"

    if symbol == "C":
        return "A" if atom.GetIsAromatic() else "C"

    if symbol == "N":
        if atom.GetFormalCharge() > 0:
            # Quaternary or protonated: no lone pair left to share.
            return "N"
        # The amide test has to come *before* the hydrogen test. An amide
        # nitrogen almost always carries a hydrogen, so the old ordering sent
        # every secondary amide down the amine branch and typed it as an
        # acceptor — which it is not.
        if _is_amide_nitrogen(atom):
            return "N"
        if atom.GetIsAromatic():
            # Pyridine-type (no hydrogen) is an acceptor. Pyrrole- and
            # imidazole-type N–H also returns `NA`, and the engine promotes it
            # to donor+acceptor because of the explicit polar hydrogen. For
            # imidazole that is exactly right; for indole and pyrrole it is
            # mildly generous, since those nitrogens are donors only. Telling
            # them apart needs a fused-ring analysis this module does not do,
            # and over-accepting one ring nitrogen is a much smaller error than
            # under-accepting every histidine.
            return "NA"
        # A neutral amine. With hydrogens it donates through them; without a
        # carbonyl to delocalise into, its lone pair still accepts. `NA` plus
        # the polar hydrogen is therefore donor *and* acceptor.
        if atom.GetTotalNumHs() > 0 or atom.GetTotalDegree() == 3:
            return "NA"
        return "N"

    if symbol == "O":
        if atom.GetFormalCharge() > 0:
            return "OA"
        return "OA"

    if symbol == "S":
        return "S"
    if symbol == "P":
        return "P"
    if symbol == "F":
        return "F"
    if symbol == "Cl":
        return "Cl"
    if symbol == "Br":
        return "Br"
    if symbol == "I":
        return "I"
    if symbol in _PT_METALS:
        return "Zn"
    return "Xx"


def _is_amide_nitrogen(atom) -> bool:
    """True for a nitrogen whose lone pair is delocalised into a carbonyl.

    A C(=O)/C(=S) neighbour with no hydrogens and a real double bond is the
    amide/thioamide discriminator. An amide nitrogen is *not* a hydrogen-bond
    acceptor, so `pdbqt_atom_type` must not label it ``NA``.

    The double bond has to be read off the **bond**, not off the oxygen atom.
    Asking an ``Atom`` for its bond type is a call that older RDKit builds
    tolerated and current ones reject with ``AttributeError``; on RDKit
    2026.03 this function raised on every molecule that reached it.
    """
    mol = atom.GetOwningMol()
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "C":
            continue
        for c in nbr.GetNeighbors():
            if c.GetIdx() == atom.GetIdx():
                continue
            if c.GetSymbol() in ("O", "S") and c.GetTotalNumHs() == 0:
                bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), c.GetIdx())
                if bond is not None and bond.GetBondType() == _DOUBLE_BOND():
                    return True
    return False


def _DOUBLE_BOND():
    from rdkit.Chem import BondType

    return BondType.DOUBLE


# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------


def prepare_ligand(
    path: str | Path,
    *,
    keep_hydrogens: bool = False,
    ph: float | None = None,
) -> tuple[list[str], list[float], np.ndarray, list[tuple[int, int]], list[str]]:
    """Prepare a ligand for docking.

    Returns
    -------
    elements, charges, coords, bonds, names
        The five parallel tables the engine consumes. ``coords`` is a
        C-contiguous ``(n, 3)`` float64 array.

    Parameters
    ----------
    path:
        Any format RDKit can read; see :data:`SUPPORTED_LIGAND_FORMATS`.
    keep_hydrogens:
        Keep *all* hydrogens. The default merges non-polar ones, which is what
        AutoDock expects and what makes the search tractable.
    ph:
        Reserved for protonation-state enumeration. The default is to use the
        protonation state already present in the input file, which is the
        predictable choice; a pH sweep is better done by preparing several
        structures explicitly.
    """
    from rdkit.Chem import AllChem, rdPartialCharges

    mol = load_molecule(path, sanitize=True)
    if mol is None:
        raise ValueError(f"could not read {path}")

    # --- 1. Protonation ---------------------------------------------------
    if ph is not None:
        warnings.warn(
            "ph= is accepted for API compatibility but protonation is taken "
            "from the input structure; prepare separate files for a pH sweep.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- 2. Merge non-polar hydrogens -------------------------------------
    if not keep_hydrogens:
        mol = Chem_remove_nonpolar_hydrogens(mol)

    # --- 3. Ensure polar hydrogens exist on donors ------------------------
    try:
        mol = _add_missing_polar_hydrogens(mol)
    except Exception as exc:  # pragma: no cover - RDKit edge cases
        warnings.warn(
            f"could not add polar hydrogens to {Path(path).name}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- 4. 3-D coordinates ------------------------------------------------
    if mol.GetNumConformers() == 0:
        # No geometry: an all-atom 3-D embedding is a reasonable last resort,
        # but the result is a guess, so say so.
        warnings.warn(
            f"{Path(path).name} has no 3-D coordinates; an embedding was "
            "generated, so the starting pose is arbitrary.",
            RuntimeWarning,
            stacklevel=2,
        )
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:  # pragma: no cover
            pass

    # --- 5. Gasteiger charges ---------------------------------------------
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol, throwOnParamFailure=False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Gasteiger charges failed ({exc}); using zero charges, which the "
            "Vina-family scoring functions ignore anyway.",
            RuntimeWarning,
            stacklevel=2,
        )

    conf = mol.GetConformer()
    elements: list[str] = []
    charges: list[float] = []
    names: list[str] = []
    coords = np.empty((mol.GetNumAtoms(), 3), dtype=np.float64)

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        pdbqt_type = pdbqt_atom_type(atom, mol)
        # Explicit hydrogens are the polar ones; the rest were merged above.
        if symbol == "H":
            elements.append("H")
        elif symbol in _PT_METALS:
            elements.append(symbol)
        else:
            elements.append(symbol)

        charge = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if not np.isfinite(charge):
            charge = 0.0
        charges.append(float(charge))

        pos = conf.GetAtomPosition(i)
        coords[i] = (pos.x, pos.y, pos.z)
        names.append(f"{pdbqt_type}_{atom.GetSymbol()}{i + 1}")

    # --- 6. Bonds ----------------------------------------------------------
    bonds: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # An O–H or N–H bond is what marks a donor; keep it so the engine can
        # see it. Every other bond to hydrogen was already merged away.
        keep = (
            mol.GetAtomWithIdx(a).GetSymbol() != "H"
            and mol.GetAtomWithIdx(b).GetSymbol() != "H"
        ) or _is_polar_h_bond(mol, a, b)
        if keep:
            bonds.append((int(min(a, b)), int(max(a, b))))

    return elements, charges, np.ascontiguousarray(coords), bonds, names


def _is_polar_h_bond(mol, a: int, b: int) -> bool:
    """True if the a–b bond joins a heavy atom to a polar hydrogen."""
    for i, j in ((a, b), (b, a)):
        if mol.GetAtomWithIdx(i).GetSymbol() == "H":
            heavy = mol.GetAtomWithIdx(j)
            return heavy.GetSymbol() in ("N", "O", "S")
    return False


def Chem_remove_nonpolar_hydrogens(mol):
    """Merge non-polar hydrogens into their heavy atoms (RDKit ``RemoveHs``).

    A thin wrapper so the intent is obvious at the call site: the *non-polar*
    qualifier matters, because polar hydrogens are what make hydrogen bonds
    visible to the scoring function.

    The implicit-hydrogen recomputation at the end is not optional. RDKit
    drops the implicit valence when ``sanitize=False`` is used, so every
    hydroxyl oxygen comes back reporting ``GetTotalNumHs() == 0`` — and a
    donor that reports no hydrogen is never recognised as one, which
    silently deletes the whole donor half of the hydrogen-bond scoring.
    """
    from rdkit import Chem

    out = Chem.RemoveHs(mol, sanitize=False)
    out.UpdatePropertyCache(strict=False)
    return out


def _add_missing_polar_hydrogens(mol):
    """Add hydrogens on heteroatoms that need them, leaving carbon alone.

    AutoDock's scoring function sees a hydrogen bond only through the ``HD``
    atom, so a hydroxyl written as a bare ``OA`` is an acceptor and nothing
    more. These hydrogens are the *only* reason a donor is visible, so failing
    to add them is not cosmetic.

    ``AllChem.AddHs(onlyOnAtoms=...)`` wants a sequence of **atom indices**, not
    a query molecule, so the SMARTS is matched first and its hits collected.
    """
    from rdkit.Chem import AllChem

    # SMARTS for the donors the engine must be able to see.
    donor = AllChem.MolFromSmarts(
        "[N&!H0&v3,N&!H0&+1,O&!H0,S&!H0,n&!H0&+0]"
    )
    if donor is None:  # pragma: no cover - the pattern is a constant
        return mol
    indices = sorted({i for match in mol.GetSubstructMatches(donor) for i in match})
    if not indices:
        return mol
    return AllChem.AddHs(mol, addCoords=True, onlyOnAtoms=indices)


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------


def prepare_receptor(
    path: str | Path,
    *,
    keep_waters: bool = False,
    keep_heterogens: bool = True,
    ph: float | None = 7.0,
) -> str:
    """Prepare a rigid receptor and return it as PDBQT text.

    Parameters
    ----------
    path:
        A ``.pdb`` or ``.ent`` file.
    keep_waters:
        Keep crystallographic waters. They are usually removed: they are
        disordered, and a wrongly-placed water blocks a real ligand.
    keep_heterogens:
        Keep ions and cofactors. Ions are often genuine binding partners, so
        they are kept by default; cofactors usually are not.
    ph:
        Recorded for provenance. Protonation states are taken from the file,
        which is the reproducible choice — see the module docstring.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_RECEPTOR_FORMATS:
        raise ValueError(
            f"unsupported receptor format {path.suffix!r}; expected .pdb, .ent or .pdbqt"
        )

    mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not read a structure from {path}")

    # --- Fragment selection ------------------------------------------------
    # Keep the largest protein fragment: a PDB file often carries a handful of
    # waters and a ligand alongside the protein, and docking against those
    # would put the search box in the wrong place.
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    if len(frags) > 1:
        largest = max(frags, key=len)
        keep = set(largest)
        dropped = len(mol.GetAtoms()) - len(keep)
        if dropped:
            warnings.warn(
                f"kept the largest of {len(frags)} fragments "
                f"({len(keep)} atoms, dropped {dropped}); use separate files "
                "to dock against multiple chains or a cofactor.",
                RuntimeWarning,
                stacklevel=2,
            )
        mol = Chem.PathToSubmol(mol, list(largest))

    # --- Water / ion filtering --------------------------------------------
    atoms = list(mol.GetAtoms())
    to_remove: list[int] = []
    n_water = 0
    n_ion = 0
    for atom in atoms:
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        resname = (info.GetResidueName() or "").strip().upper()
        name = (info.GetName() or "").strip().upper()
        if resname in ("HOH", "WAT", "DOD", "H2O") or name in ("O", "OW"):
            if resname in ("HOH", "WAT", "DOD", "H2O") and not keep_waters:
                to_remove.append(atom.GetIdx())
                n_water += 1
        elif _is_ion(resname, name):
            if keep_heterogens:
                continue
            to_remove.append(atom.GetIdx())
            n_ion += 1

    if to_remove:
        mol.RemoveAtom(*sorted(to_remove, reverse=True))
        warnings.warn(
            f"removed {n_water} water molecule(s)"
            + (f" and {n_ion} ion(s)" if n_ion else "")
            + f" from {path.name}",
            RuntimeWarning,
            stacklevel=2,
        )

    if ph is not None:
        # Protonation is taken from the file; record why.
        pass

    mol, n_donors = _add_receptor_polar_hydrogens(mol)
    if n_donors:
        warnings.warn(
            f"added {n_donors} polar hydrogen(s) to the receptor; the input "
            "file modelled none, so it would otherwise have had no "
            "hydrogen-bond donors",
            RuntimeWarning,
            stacklevel=2,
        )

    return _molecule_to_pdbqt(mol, resname="REC")


# Heavy atoms that carry a polar hydrogen in a rigid receptor, by residue.
#
# PDB files carry residue and atom names even when they carry no hydrogens, so
# the donor sites can be recovered from the file itself without guessing a
# protonation state. This list is the standard set used by every receptor
# preparation tool; the one genuine judgement call is histidine, noted below.
_RECEPTOR_DONOR_SITES: dict[str, tuple[str, ...]] = {
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "CYS": ("SG",),
    "LYS": ("NZ",),
    "ARG": ("NH1", "NH2"),
    "TRP": ("NE1",),
    # Histidine has two equally populated ring tautomers and an unsanitised PDB
    # file records neither, because there are no hydrogens to read. NE2 is the
    # HID (delta-protonated) form, the more common of the two by roughly two to
    # one. When the input file *does* carry the hydrogens, the tautomer in the
    # file is used instead — this fallback only applies when it does not.
    "HIS": ("NE2",),
}


def _add_receptor_polar_hydrogens(mol):
    """Add the polar hydrogens a receptor needs to donate hydrogen bonds.

    Returns ``(mol, n_donors)``. `AllChem.AddHs` builds a new molecule rather
    than editing in place, so the caller has to take the return value.

    **Why this exists.** An experimental PDB file normally contains no
    hydrogens at all. Without this step a real protein receptor comes out with
    zero donors: every serine, threonine and tyrosine side chain is typed as a
    bare acceptor oxygen, and every backbone amide nitrogen is inert. The
    receptor then scores hydrogen bonds that are not there and misses the ones
    that are — a silent error, because everything still runs and still returns
    plausible-looking energies.

    **Rigidness.** `AddHs` places the new hydrogens and leaves every heavy atom
    exactly where it was, so the receptor stays rigid. Nothing is optimised,
    which is what a receptor requires.

    Atoms that already carry a hydrogen neighbour are left alone, so a file
    that already models hydrogens is not given a second set.
    """
    from rdkit.Chem import AllChem

    # A PDB read with `sanitize=False` has no implicit-hydrogen counts at all,
    # so `AddHs` under-protonates — the N-terminal nitrogen came out with one
    # hydrogen instead of two. Recomputing the property cache first fixes that
    # and, just as importantly, keeps a disulfide cysteine correct: its sulfur
    # already has two bonds, so it correctly gets none.
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(False)
    mol.UpdatePropertyCache(strict=False)

    donors: set[int] = set()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        symbol = atom.GetSymbol()
        if symbol not in ("N", "O", "S"):
            continue
        if any(nbr.GetSymbol() == "H" for nbr in atom.GetNeighbors()):
            continue  # already protonated by the input file
        resname = (info.GetResidueName() or "").strip().upper()
        name = (info.GetName() or "").strip().upper()
        # The backbone amide nitrogen donates in every residue except proline,
        # whose nitrogen carries no hydrogen.
        if name == "N" and resname != "PRO":
            donors.add(atom.GetIdx())
            continue
        if name in _RECEPTOR_DONOR_SITES.get(resname, ()):
            donors.add(atom.GetIdx())

    if not donors:
        return mol, 0
    indices = sorted(donors)
    try:
        out = AllChem.AddHs(mol, addCoords=True, onlyOnAtoms=indices)
    except Exception:  # pragma: no cover - RDKit version dependent
        # A receptor without hydrogens is wrong but still usable; a hard
        # failure here would make a large structure un-dockable.
        warnings.warn(
            "could not add polar hydrogens to the receptor; it will have no "
            "hydrogen-bond donors",
            RuntimeWarning,
            stacklevel=2,
        )
        return mol, 0
    return out, len(indices)


def _is_ion(resname: str, name: str) -> bool:
    """True for a monoatomic ion residue."""
    if resname in (
        "NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE", "CU", "NI", "CO", "CD", "HG", "SO4",
    ):
        return True
    return False


def _residue_identity(atom, _seen: set | None = None) -> tuple[str, str, str, int]:
    """``(pdb_name, resname, chain, resseq)`` for one atom.

    RDKit keeps the original PDB record on each atom, so the residue an atom
    came from is available -- it was simply not being written out. Losing it
    means every atom of a 300-residue protein lands in one anonymous `REC`
    residue numbered 1, and nothing downstream can tell a backbone nitrogen
    from a side-chain one. That is enough to make a viewer unable to group
    atoms into residues, infer bonds, or draw a backbone, so it is carried
    through here.

    The polar hydrogens added by :func:`_add_receptor_polar_hydrogens` have no
    record of their own -- RDKit invents them -- so they inherit the identity of
    the heavy atom they were attached to. Without that, every added hydrogen
    would land in a residue of its own and the receptor would be shredded again,
    which is presumably why the identity used to be dropped wholesale.

    `_seen` guards the neighbour walk. Two atoms that both lack a record and
    are bonded to each other would otherwise bounce between them forever.
    """
    if _seen is None:
        _seen = set()
    if atom.GetIdx() in _seen:
        return atom.GetSymbol(), "UNL", " ", 1
    _seen.add(atom.GetIdx())

    info = atom.GetPDBResidueInfo()
    if info is not None and (info.GetResidueName() or "").strip():
        name = (info.GetName() or "").strip() or atom.GetSymbol()
        resname = (info.GetResidueName() or "").strip().upper()
        chain = (info.GetChainId() or " ").strip() or " "
        try:
            resseq = int(info.GetResidueNumber())
        except (TypeError, ValueError):
            resseq = 1
        return name, resname, chain, resseq

    for nbr in atom.GetNeighbors():
        inherited = _residue_identity(nbr, _seen)
        if inherited[1] != "UNL":
            return inherited
    return atom.GetSymbol(), "UNL", " ", 1


def _atom_name(atom) -> str:
    """The PDB atom name to write, for an atom that may be a new hydrogen.

    RDKit names an added hydrogen by copying the heavy atom's name, so writing
    `info.GetName()` straight through produces a residue with three atoms called
    `N`: the backbone nitrogen and the two hydrogens on it. That is not a
    cosmetic problem -- residue-level bond perception keys off atom names, and
    a duplicate name silently drops the bonds to everything after the first
    match. A hydrogen is therefore always written as `H`.
    """
    if atom.GetSymbol() == "H":
        return "H"
    info = atom.GetPDBResidueInfo()
    if info is not None and (info.GetName() or "").strip():
        return (info.GetName() or "").strip()
    return atom.GetSymbol()


def _molecule_to_pdbqt(mol, resname: str = "UNL") -> str:
    """Serialise an RDKit molecule to PDBQT text.

    Only polar hydrogens are written, and heavy-atom coordinates are taken as
    they are: a receptor must be rigid, so nothing here may be optimised.

    Residue names, atom names, chain identifiers and residue numbers come from
    each atom's own PDB record (see :func:`_residue_identity`). `resname` is
    only the fallback for a molecule that has no residue information at all,
    such as one read from an SDF.

    The line layout comes from :func:`opendocking.pdbqt_writer.format_atom_line`.
    There is deliberately only one formatter in the package: two hand-rolled
    versions of a fixed-column format will eventually disagree by one space,
    and a PDBQT file that is one column out still loads — it just puts the
    charge in the wrong place.
    """
    from .pdbqt_writer import format_atom_line

    conf = mol.GetConformer()
    lines: list[str] = ["REMARK  Generated by Open Docking (opendocking) preparation"]
    serial = 0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        pos = conf.GetAtomPosition(atom.GetIdx())
        if symbol == "H":
            # Keep only hydrogens bonded to N, O or S.
            nbrs = atom.GetNeighbors()
            if not nbrs or nbrs[0].GetSymbol() not in ("N", "O", "S"):
                continue
            pdbqt_type = "HD"
            # ...and only if it actually landed next to that atom. `AddHs` is
            # asked for coordinates, but it does not always manage them, and a
            # hydrogen with no position is written as (0, 0, 0) -- an atom at
            # the coordinate origin, thousands of angstroms from its own
            # parent, which then docks as if it were real. Three of them came
            # out that way on the crambin fixture. A hydrogen 0.7-1.35 A from
            # the heavy atom it was added to is a real one; anything else is
            # not, and the residue is better off without it than with a
            # phantom.
            parent = conf.GetAtomPosition(nbrs[0].GetIdx())
            separation = pos.Distance(parent)
            if not (0.7 <= separation <= 1.35):
                warnings.warn(
                    f"dropped a polar hydrogen of {nbrs[0].GetSymbol()} at "
                    f"{separation:.2f} Å from the atom it was added to: "
                    "RDKit did not give it a usable position",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
        else:
            pdbqt_type = pdbqt_atom_type(atom, mol)
        charge = (
            atom.GetDoubleProp("_GasteigerCharge")
            if atom.HasProp("_GasteigerCharge")
            else 0.0
        )
        if not np.isfinite(charge):
            charge = 0.0
        serial += 1
        name, res, chain, resseq = _residue_identity(atom)
        lines.append(
            format_atom_line(
                serial,
                _atom_name(atom),
                res or resname,
                resseq,
                (pos.x, pos.y, pos.z),
                charge,
                pdbqt_type,
                chain=chain,
            )
        )
    lines.append("TER")
    return "\n".join(lines) + "\n"
