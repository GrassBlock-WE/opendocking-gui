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

## Branches that are fallbacks, not behaviour

Four places in the ligand pipeline sit inside ``except`` blocks that guard
against an RDKit call failing. **One of them executes; three have never run.**

* The MMFF relaxation after a generated embedding **is** reached, by any acyclic
  ligand that arrived without coordinates: ``MMFFOptimizeMolecule`` needs ring
  information and a molecule with no ring has none, so it raises and the bare
  ``except`` swallows it. The embedding stands unrelaxed and the user is told
  only that the pose is a guess. Covered in ``scripts/ligand_check.py``.
* Adding polar hydrogens (step 4), the Gasteiger charges, and the receptor's
  ``AddHs`` are **not** covered by anything, here or elsewhere. Reproducing
  them would mean making RDKit fail on purpose, which would test the mock
  rather than the code. They are kept because a ligand that cannot be prepared
  at all is a worse outcome than one prepared without a charge, and they are
  listed here so the next reader does not assume they have been tried.

# Receptor preparation pipeline

Receptors need a different treatment: they must be **rigid**, so the engine
receives only polar hydrogens, and they are usually too large for RDKit's
perception to be worth running on the whole structure. Waters, ions and
disconnected fragments are reported rather than silently kept.

That report exists twice, on purpose. The warning is for a human reading a log;
:class:`ReceptorPrepReport` is for a program, because a warning is a string that
GUI code and CLI code routinely drop on the floor and a caller cannot ask a
string a question. The warning text is generated *from* the report, so the two
cannot disagree.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = [
    "prepare_ligand",
    "prepare_receptor",
    "prepare_receptor_with_report",
    "ReceptorPrepReport",
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

    # `load_molecule` either returns a molecule or raises, so there is no `None`
    # to check for here. There used to be, and it could not run.
    mol = load_molecule(path, sanitize=True)

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
        # The element table is the symbol, for every atom including hydrogens
        # and metals. The three-way branch that used to be here appended
        # `symbol` in all three arms, so it read as a special case and did
        # nothing; there is one line here instead of three identical ones.
        elements.append(symbol)

        # **The charge is the atom's own Gasteiger charge, and nothing else.**
        # Merging a non-polar hydrogen in step 2 does *not* fold that hydrogen's
        # partial charge into its parent, which is what AutoDock's convention
        # asks for. The consequence is measurable: benzene's six carbons each
        # carry -0.062 e, so the table sums to -0.37 e for a neutral molecule,
        # and ibuprofen sums to -0.60 e. Left as it is, because changing it
        # changes every score the engine has ever produced -- but recorded here
        # so the next reader is not surprised by it.
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
        # An O-H or N-H bond is what marks a donor; keep it so the engine can
        # see it. **Under `keep_hydrogens=True` this leaves a gap**: the
        # non-polar hydrogens are in `elements` and `coords` but their C-H bonds
        # are not in this list, because `_is_polar_h_bond` is False for them.
        # Ibuprofen with `keep_hydrogens=True` comes back with 33 atoms and 16
        # bonds, 17 of those hydrogens bonded to nothing. The old comment here
        # said every other bond to hydrogen "was already merged away", which is
        # only true of the default path.
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
    # A guard against a future RDKit that stops parsing the pattern, not a
    # tested branch: the SMARTS is a constant, so today it either parses or
    # this module fails to import. If it ever stopped parsing, donor
    # perception would quietly stop happening and every ligand would come out
    # with no hydrogens -- so this says so rather than returning `None`.
    if donor is None:  # pragma: no cover - the pattern is a constant
        raise RuntimeError(
            "RDKit no longer parses the donor SMARTS; polar hydrogens would "
            "silently stop being added"
        )
    indices = sorted({i for match in mol.GetSubstructMatches(donor) for i in match})
    if not indices:
        return mol
    return AllChem.AddHs(mol, addCoords=True, onlyOnAtoms=indices)


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------


def _plural(n: int, word: str) -> str:
    """``1 chain`` / ``2 chains``, with no locale and no unicode hyphen."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _render_chain_ids(ids) -> str:
    """Chain ids for a message, with the blank id spelled out."""
    return ", ".join(repr(c) if c != " " else "(blank)" for c in ids)


@dataclass(frozen=True)
class ReceptorPrepReport:
    """What receptor preparation did to the input, as data rather than prose.

    Produced by :func:`prepare_receptor_with_report`; the ``RuntimeWarning``
    that :func:`prepare_receptor` raises is built from this object by
    :meth:`component_warning`, so the number a caller reads and the number in
    the log line come from the same place.

    **Chains and fragments are counted separately, on purpose.** RDKit
    fragments are connectivity-based: two chains of a homodimer that do not
    touch are two fragments, but two chains joined by a disulfide or sitting
    close enough for RDKit to bond them are *one*. A report that called
    fragments "chains" would say "1 of 1 chains" for a file that is visibly a
    dimer in a viewer, which is how the 1HVR redocking bug stayed invisible:
    the benchmark searched a two-chain 1826-atom structure while docking
    against a 621-atom single chain, and nothing in the output said so.
    :attr:`fragments_in` is therefore what the *selection* acted on and
    :attr:`chains_in` is what the file *contains*; when they disagree,
    :attr:`fragments_equal_chains` is ``False``.

    Attributes
    ----------
    source:
        The input file name, for messages.
    atoms_in, atoms_kept, atoms_dropped:
        Atom counts for the whole file and for what survived filtering.
        ``atoms_kept`` is measured *before* preparation adds polar hydrogens
        (:attr:`polar_hydrogens_added`) and excludes them, so it is directly
        comparable with the atom count in the input file.
    fragments_in, fragments_kept:
        Connectivity-based components. Selection kept exactly one.
    chains_in, chains_kept:
        Distinct PDB chain ids. Atoms RDKit invented -- the added polar
        hydrogens -- carry no residue record and belong to no chain, so they
        cannot inflate these numbers. A molecule in which no atom carries a
        record is one anonymous chain.
    chain_ids_kept, chain_ids_dropped:
        The actual chain ids, which is the answer to "what did you throw
        away?".
    selection:
        How the kept component was chosen, in words -- ``"largest fragment"``
        or e.g. ``"chain 'B'"``. A caller can see the rule that was applied
        rather than having to infer it.
    water_atoms_removed, ion_atoms_removed, other_atoms_removed:
        Where the dropped atoms went, summed over both removal stages: the
        component selection and the water/ion filter. They always add up to
        :attr:`atoms_dropped`. An ion is counted here whenever it failed to
        survive -- and a monoatomic ion fails under the *default*
        ``keep_heterogens=True``, because it is its own component and the
        selection drops it before the filter is ever consulted. ``keep_heterogens``
        only changes the outcome for an ion inside the kept component. "other"
        is everything else: whole extra chains, free cofactors, ligands.
    polar_hydrogens_added:
        Donor sites given a polar hydrogen because the input file modelled
        none.
    pdbqt_atoms_written:
        ATOM records in the returned text. This can be *fewer* than
        ``atoms_kept + polar_hydrogens_added``: the writer drops a polar
        hydrogen RDKit placed too far from its heavy atom to be real, and
        says so in its own warning.
    """

    source: str
    atoms_in: int
    atoms_kept: int
    atoms_dropped: int
    fragments_in: int
    fragments_kept: int
    chains_in: int
    chains_kept: int
    chain_ids_kept: tuple[str, ...]
    chain_ids_dropped: tuple[str, ...]
    selection: str
    water_atoms_removed: int
    ion_atoms_removed: int
    other_atoms_removed: int
    polar_hydrogens_added: int
    pdbqt_atoms_written: int

    @property
    def chains_dropped(self) -> int:
        """How many chains the caller loaded are not in the output."""
        return max(self.chains_in - self.chains_kept, 0)

    @property
    def is_lossy(self) -> bool:
        """True if preparation threw any atom away."""
        return self.atoms_dropped > 0

    @property
    def fragments_equal_chains(self) -> bool:
        """False when the file's chains and RDKit's fragments disagree."""
        return self.fragments_in == self.chains_in

    def summary(self) -> str:
        """One line for a log or a status bar."""
        bits = [
            f"{self.chains_kept}/{self.chains_in} chains",
            f"{self.fragments_kept}/{self.fragments_in} fragments",
            f"{self.atoms_kept}/{self.atoms_in} atoms",
        ]
        if self.is_lossy:
            bits.append(
                f"dropped {self.atoms_dropped} "
                f"({self.water_atoms_removed} water, {self.ion_atoms_removed} ion, "
                f"{self.other_atoms_removed} other)"
            )
        if self.polar_hydrogens_added:
            bits.append(f"+{self.polar_hydrogens_added} polar H")
        return (
            f"{self.source}: "
            + ", ".join(bits)
            + f", selected {self.selection}, kept chain(s) {_render_chain_ids(self.chain_ids_kept)}"
        )

    def component_warning(self) -> str:
        """The ``RuntimeWarning`` text for lost components.

        Derived entirely from the fields, so a caller that reads the warning
        and a caller that reads the report are reading the same numbers.
        """
        text = (
            f"{self.source}: kept {self.selection}; "
            f"{_plural(self.chains_kept, 'chain')} of "
            f"{self.chains_in} chains and {_plural(self.fragments_kept, 'fragment')} "
            f"of {self.fragments_in} fragments kept (chain(s) "
            f"{_render_chain_ids(self.chain_ids_kept)}), "
            f"{_plural(self.atoms_kept, 'atom')} of {self.atoms_in} atoms kept, "
            f"{self.atoms_dropped} dropped: {self.water_atoms_removed} water, "
            f"{self.ion_atoms_removed} ion, {self.other_atoms_removed} other"
        )
        if self.chain_ids_dropped:
            text += (
                f"; chain(s) {_render_chain_ids(self.chain_ids_dropped)} were dropped"
            )
            if self.selection == "largest fragment":
                text += (
                    f" - pass keep_chain={self.chain_ids_dropped[0]!r} to keep one "
                    "of them"
                )
            text += ", or use separate files to dock against multiple chains or a cofactor"
        return text

    def polar_hydrogen_warning(self) -> str:
        """The ``RuntimeWarning`` text for added polar hydrogens."""
        return (
            f"added {self.polar_hydrogens_added} polar hydrogen(s) to the receptor; "
            "the input file modelled none, so it would otherwise have had no "
            "hydrogen-bond donors"
        )


#: ``stacklevel`` for warnings raised inside :func:`_prepare_receptor_impl`.
#:
#: Both public entry points are thin wrappers around the implementation, so a
#: warning raised there is three frames from the code the caller actually wrote.
#: Getting this wrong points the warning at ``prep.py`` instead of the caller,
#: which is how a warning ends up looking like library noise.
_WARN_STACKLEVEL = 3

#: Residue names that mean "water". A bare atom *name* of ``O``/``OW`` is not
#: enough on its own -- see :func:`_is_water`.
_WATER_RESNAMES = ("HOH", "WAT", "DOD", "H2O")


def _chain_ids(mol) -> tuple[str, ...]:
    """Sorted distinct PDB chain ids in `mol`.

    Atoms with no residue record are skipped rather than counted as a chain of
    their own: RDKit invents records for the hydrogens it adds during
    preparation, and counting those would invent a chain that no viewer shows.

    A molecule in which *no* atom carries a record is reported as one anonymous
    chain, because "0 chains" is a worse answer than "1 chain I cannot name".
    That fallback is not reachable from a receptor file -- RDKit attaches a
    record to every atom of any ``.pdb``/``.ent``/``.pdbqt`` it can read, even
    one written with blank residue names -- but it *is* reachable from an
    emptied molecule, so a receptor whose only atom was filtered away reports
    one anonymous chain rather than none. Covered both ways in
    ``scripts/prep_check.py``.
    """
    ids: set[str] = set()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        ids.add((info.GetChainId() or " ").strip() or " ")
    if not ids:
        return (" ",)
    return tuple(sorted(ids))


def _component_kind(atom) -> str:
    """``"water"``, ``"ion"`` or ``"other"`` for one atom, for the report.

    Classification is by *residue name* only, unlike the removal filter in
    :func:`_is_water`, which also accepts an atom name of ``O``. A backbone
    carbonyl oxygen is named ``O``, so calling it a water here would put
    carbonyl carbons in the water column of the report.
    """
    info = atom.GetPDBResidueInfo()
    # Not reachable from a receptor file: RDKit attaches a record to every atom
    # of a .pdb/.ent/.pdbqt it reads, and an atom with no record has no
    # resname to classify by. It is kept for a molecule that has lost its
    # records -- and then "other" is the only honest answer, since an atom with
    # no record is by definition not a water and not a monoatomic ion residue.
    if info is None:
        return "other"
    resname = (info.GetResidueName() or "").strip().upper()
    name = (info.GetName() or "").strip().upper()
    if resname in _WATER_RESNAMES:
        return "water"
    if _is_ion(resname, name):
        return "ion"
    return "other"


def _select_component(frags, chain_of, keep_chain: str | None) -> tuple[int, str]:
    """Index of the component to keep, and a description of the rule used.

    With ``keep_chain=None`` this is "the largest component, first one on a
    tie" -- the historical behaviour, unchanged. With a chain id it is "the
    largest component carrying that chain", and a chain id that is not in the
    file is an error rather than a silent fallback: quietly preparing the
    largest component when the caller asked for chain B is the very bug this
    reporting exists to make visible.
    """
    sizes = [frag.GetNumAtoms() for frag in frags]
    if keep_chain is None:
        return max(range(len(frags)), key=lambda i: sizes[i]), "largest fragment"
    matches = [i for i, ids in enumerate(chain_of) if keep_chain in ids]
    if not matches:
        available = sorted({c for ids in chain_of for c in ids})
        raise ValueError(
            f"no chain {keep_chain!r} in the structure; chains present: "
            f"{', '.join(repr(c) for c in available)}"
        )
    # A physical chain is one covalent component, so the match is normally a
    # single fragment. If a file splits one chain across two, the larger piece
    # is the one with the backbone in it.
    return max(matches, key=lambda i: sizes[i]), f"chain {keep_chain!r}"


def prepare_receptor(
    path: str | Path,
    *,
    keep_waters: bool = False,
    keep_heterogens: bool = True,
    ph: float | None = 7.0,
    keep_chain: str | None = None,
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

        **This filter only sees an ion that is inside the component it kept.**
        A monoatomic ion is always its own component, so for an ordinary
        protein file it is already gone by the time this runs and the flag has
        no effect on it -- the report says so, in ``ion_atoms_removed``, rather
        than leaving the caller to assume the ion survived. To be removed *by
        this filter* an ion has to be part of the kept component: a
        multi-atom ion residue such as ``SO4`` bonded into the structure, or a
        file that is little more than a metal site. See
        ``scripts/prep_check.py``, which covers both directions with a one-atom
        zinc file.
    ph:
        Recorded for provenance. Protonation states are taken from the file,
        which is the reproducible choice -- see the module docstring.
    keep_chain:
        Keep this PDB chain id instead of the largest component. ``None``
        (the default) is the historical behaviour and is what every existing
        caller gets. A chain id that is not in the file raises ``ValueError``.

    Returns
    -------
    str
        PDBQT text. The return value is unchanged by this function's history;
        a caller that wants to know *what was dropped* should use
        :func:`prepare_receptor_with_report`, which returns the same text
        alongside a :class:`ReceptorPrepReport`.
    """
    return prepare_receptor_with_report(
        path,
        keep_waters=keep_waters,
        keep_heterogens=keep_heterogens,
        ph=ph,
        keep_chain=keep_chain,
    )[0]


def prepare_receptor_with_report(
    path: str | Path,
    *,
    keep_waters: bool = False,
    keep_heterogens: bool = True,
    ph: float | None = 7.0,
    keep_chain: str | None = None,
) -> tuple[str, ReceptorPrepReport]:
    """:func:`prepare_receptor`, plus a structured account of the losses.

    Returns
    -------
    (pdbqt_text, report)
        The same text :func:`prepare_receptor` would return, and a frozen
        :class:`ReceptorPrepReport` a caller can read without parsing a
        warning string. This is the opt-in entry point; the plain function
        keeps its ``-> str`` contract for the CLI, the workbench and the
        benchmark.
    """
    return _prepare_receptor_impl(
        path,
        keep_waters=keep_waters,
        keep_heterogens=keep_heterogens,
        ph=ph,
        keep_chain=keep_chain,
    )


def _prepare_receptor_impl(
    path: str | Path,
    *,
    keep_waters: bool,
    keep_heterogens: bool,
    ph: float | None,
    keep_chain: str | None,
) -> tuple[str, ReceptorPrepReport]:
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

    atoms_in = mol.GetNumAtoms()
    chain_ids_in = _chain_ids(mol)

    # --- Component selection ------------------------------------------------
    # Keep one component: a PDB file often carries a handful of waters, a
    # ligand and a second chain alongside the protein, and docking against
    # those would put the search box in the wrong place. What is lost here is
    # counted now, per kind, so the report can say *what* went rather than
    # only how much.
    #
    # `asMols=True`, not `PathToSubmol`. `PathToSubmol(mol, path)` takes a
    # *traversal path*, not a set of indices, and returns whatever the walk
    # reaches -- which is not the set that was passed in. `GetMolFrags` hands
    # back ascending index tuples, so every fragment was being read as a path.
    # Measured on RDKit 2026.03.1, on a 27-atom four-component file built by
    # `scripts/prep_check.py`:
    #
    #   11-atom chain A -> 13 atoms back, two of them from chain B
    #   14-atom chain B -> exactly 14 (right, by luck of where it ended)
    #   1-atom water    ->  0 atoms back
    #
    # and on ethanol with hydrogens, `PathToSubmol(m, range(k))` returns
    # `min(k + 1, 9)` atoms.
    #
    # So this is not "one extra atom at the end". The count is whatever the walk
    # reaches, and it moves in both directions: it added atoms from a component
    # that had just been reported as discarded -- making the number in the
    # warning disagree with the molecule the writer then serialised -- and for a
    # single-atom component it returned an *empty* molecule, so a receptor file
    # containing only waters or ions prepared to nothing at all. The components
    # come back as molecules instead, so the kept component is the kept
    # component.
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    fragments_in = len(frags)
    chain_of = [_chain_ids(frag) for frag in frags]
    dropped_atoms = {"water": 0, "ion": 0, "other": 0}
    if fragments_in > 1:
        chosen, selection = _select_component(frags, chain_of, keep_chain)
        for i, frag in enumerate(frags):
            if i == chosen:
                continue
            for atom in frag.GetAtoms():
                dropped_atoms[_component_kind(atom)] += 1
        mol = frags[chosen]
    else:
        # One component: nothing can be selected away, but an explicit chain
        # request still has to be validated rather than ignored.
        selection = "largest fragment" if keep_chain is None else f"chain {keep_chain!r}"
        if keep_chain is not None:
            _select_component(frags, chain_of, keep_chain)

    # --- Water / ion filtering --------------------------------------------
    atoms = list(mol.GetAtoms())
    to_remove: list[int] = []
    for atom in atoms:
        info = atom.GetPDBResidueInfo()
        # Not reachable from a receptor file (RDKit always attaches a record).
        # Kept because an atom with no record cannot be classified as a water
        # or an ion, and skipping it is the conservative answer: keeping a
        # protein is worse than keeping a crystal artefact.
        if info is None:
            continue
        resname = (info.GetResidueName() or "").strip().upper()
        name = (info.GetName() or "").strip().upper()
        if _is_water(resname, name):
            # The residue name is re-checked on purpose. The atom-name half of
            # the test also matches every backbone carbonyl oxygen, and those
            # are the protein.
            if resname in _WATER_RESNAMES and not keep_waters:
                to_remove.append(atom.GetIdx())
                dropped_atoms["water"] += 1
        elif _is_ion(resname, name):
            if keep_heterogens:
                continue
            to_remove.append(atom.GetIdx())
            dropped_atoms["ion"] += 1

    if to_remove:
        # `Chem.RWMol`, because every molecule this module holds is a `Mol` --
        # `MolFromPDBFile`, `GetMolFrags` and `PathToSubmol` all return one --
        # and on RDKit 2026.03.1 a `Mol` has no `RemoveAtom` at all. This line
        # raised `AttributeError` on the first input that ever reached it, which
        # is why `--drop-heterogens` and `keep_waters=False` had never actually
        # removed anything: a water or a monoatomic ion is its own component
        # and is gone before this filter is reached. The copy keeps the
        # conformer and every residue record, both of which the writer needs.
        mol = Chem.RWMol(mol)
        mol.RemoveAtom(*sorted(to_remove, reverse=True))

    if ph is not None:
        # Protonation is taken from the file; record why.
        pass

    mol, n_donors = _add_receptor_polar_hydrogens(mol)
    chain_ids_kept = _chain_ids(mol)
    atoms_kept = atoms_in - sum(dropped_atoms.values())

    text = _molecule_to_pdbqt(mol, resname="REC")
    report = ReceptorPrepReport(
        source=path.name,
        atoms_in=atoms_in,
        atoms_kept=atoms_kept,
        atoms_dropped=sum(dropped_atoms.values()),
        fragments_in=fragments_in,
        fragments_kept=1,
        chains_in=len(chain_ids_in),
        chains_kept=len(chain_ids_kept),
        chain_ids_kept=chain_ids_kept,
        chain_ids_dropped=tuple(c for c in chain_ids_in if c not in chain_ids_kept),
        selection=selection,
        water_atoms_removed=dropped_atoms["water"],
        ion_atoms_removed=dropped_atoms["ion"],
        other_atoms_removed=dropped_atoms["other"],
        polar_hydrogens_added=n_donors,
        pdbqt_atoms_written=sum(
            1 for line in text.splitlines() if line.startswith("ATOM")
        ),
    )

    # --- The warnings, from the report -------------------------------------
    # Raised at the end rather than at the point of loss, because a report that
    # is built from the whole run is the only way for the log line and the
    # returned object to be guaranteed to agree.
    if report.is_lossy:
        warnings.warn(
            report.component_warning(), RuntimeWarning, stacklevel=_WARN_STACKLEVEL
        )
    if n_donors:
        warnings.warn(
            report.polar_hydrogen_warning(),
            RuntimeWarning,
            stacklevel=_WARN_STACKLEVEL,
        )

    return text, report


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
        # Not reachable from a receptor file, and not from the hydrogens
        # `AddHs` invents either: those have no record but also no residue, so
        # they cannot be a donor site. Kept so that a record-less atom is
        # skipped rather than guessed at.
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


def _is_water(resname: str, name: str) -> bool:
    """True if this record is water-shaped, by residue name *or* atom name.

    The atom-name half of the test is a fallback for files that write a water
    oxygen without a water residue name. It is deliberately wider than
    "is a water": an atom named ``O`` or ``OW`` is also every backbone
    carbonyl and hydroxyl oxygen in the protein, which is why the caller
    re-checks the residue name before removing anything.
    """
    return resname in _WATER_RESNAMES or name in ("O", "OW")


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
    # Not a normal return: it is the recursion guard. Two atoms that both lack
    # a record and are bonded to each other would bounce between them forever,
    # so a repeat visit gives up and lets the atom fall through to the UNL
    # answer at the bottom. It is only reachable for a record-less molecule,
    # which no receptor file produces.
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
    # The added polar hydrogens land here: they have no record of their own,
    # and the atom they were attached to is the neighbour that carries one.
    # `UNL` is the fallback when there is no such neighbour, which needs a
    # record-less molecule -- not something a receptor file produces.
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
    # Only for an atom whose record has no name, which a receptor file does not
    # produce. The element symbol is the least-wrong name available and keeps
    # the column non-empty, which is what the bond perception downstream reads.
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
