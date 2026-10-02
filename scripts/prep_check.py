"""Headless check of receptor preparation, so silent data loss stops being silent.

`prepare_receptor` keeps one component of the input and throws the rest away.
That is a good default -- a PDB file often carries a handful of waters, a
ligand and a second chain alongside the protein, and docking against those
would put the search box in the wrong place -- but it is a *loss*, and a loss
reported only as a `RuntimeWarning` is a loss that a GUI, or a benchmark with
warnings suppressed, never sees. It was: on 1HVR, a homodimer, the benchmark
searched a two-chain 1826-atom structure while docking against a 621-atom
single chain, and every RMSD it produced was meaningless. The warning existed.
Nobody saw it, and it never actually said that one of two chains was gone.

So this script checks three separate things, and the distinction matters:

* **The fixtures are physically possible.** Coordinates come from RDKit's own
  distance-geometry embedding rather than a hand-placed grid, and the geometry
  is audited afterwards: no two atoms closer than a covalent bond, no bond
  longer than one, chains tens of angstroms apart. This repository has shipped
  a fixture that put 40 atoms on identical coordinates, which tests nothing
  except that the code does not crash.
* **The structured report tells the truth in both directions.** A two-chain
  file must report exactly one chain kept *and name it*; a single-chain file
  must report nothing dropped. A report that always claims a loss, or that
  always claims none, fails one of the two.
* **The warning is generated from the report.** The emitted `RuntimeWarning`
  text must equal `report.component_warning()` for the same input. The bug that
  cost the benchmark its numbers lived in the gap between what preparation did
  and what the log said it did.

Every guard is shown failing at least once. A check that cannot fail teaches
nothing, and this repository has a rewritten check script in its history for
exactly that reason.

Run:  python scripts/prep_check.py
"""

from __future__ import annotations

import re
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Prefer the *installed* package, and only fall back to the source tree when
# there is no installed one -- the same rule the other check scripts use.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking import prep  # noqa: E402

CRAMBIN = ROOT / "examples" / "1crn_receptor.pdb"

#: The number of checks this file runs, asserted on the way out. This file had
#: no pin, which meant its total was printed and never compared -- the exact
#: failure every other gate in the tree holds itself to. `skip()` is not
#: defined here, so the count is the number of `check()` calls that run.
#:
#: **102 is what a green run of this file prints, and it cannot be read off the
#: source.** The file has 96 `check()` call sites -- 69 of them unconditional
#: and 27 behind a condition, which is the split the block below declares -- so
#: the number that matters is the number of results a run produces, and counting
#: call sites is a different number (off by 6 today) that no reading of the file
#: can correct. That is what makes this a *measured* constant rather than a typed
#: one, and it is why the derivation is the run: add a check or remove one and
#: this is a red until someone re-runs the file and takes the number it reports.
#:
#: 99 -> 102 is this round's row-225 pin (see `OOB_CLAIM_RE` above for why that
#: row can be pinned and rows 207/222 cannot).
#: GATE-DECLARE 1
#: sites: 69 unconditional + 27 guarded
#: guards: sha256:757db9bd10cd656af4d902c5a55658fb89fc5894077419cbe32c39b69f9dad5a
#:
#: The two columns above are measured by `check_scripts_declare.py` walking
#: this file's own syntax tree, and they are the count of *call sites*, which
#: is a different quantity from the pin on the line below: sites do not
#: multiply by loop iteration and guards do not stop being sites. Both are
#: declared because a reader who has one of them and assumes the other is wrong
#: in a direction the file cannot check.
EXPECTED_CHECKS = 102

# A Windows console is frequently GBK and a warning this script merely *counts*
# may contain a non-ASCII character. A character that cannot be encoded must
# not turn a check run into a traceback.
if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(errors="backslashreplace")

FAILURES: list[str] = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


def finish() -> int:
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


# ---------------------------------------------------------------------------
# Row 225 of docs/VERIFICATION.md is re-measured here.
#
# A number in the ledger that no gate re-derives drifts silently, and this one
# already drifted: the out-of-box charge used to be a flat 1000.0 step and is
# now a per-angstrom ramp, so every figure derived from a docking run moved,
# and three rows kept quoting the old ones long after the engine stopped
# producing them. Re-staling one of them turns nothing red.
#
# Row 225 is pin-able and rows 207/222 are not, and the difference is the
# search. These readings come from `score_conformation` on a hand-placed
# conformation -- one atom, no Monte Carlo, no seed, no thread count. Every
# figure in 207 and 222 comes out of a search, and row 207 itself records that
# the same fixture read 1.3e-04 on the CI runner against 3.6e-05 here because
# the walk differs with the thread count. Pinning those would be a red on a
# correct tree, which is worse than no pin because it teaches people to ignore
# red. So the pose-derived numbers stay unpinned and this comment is the
# record of why, rather than the absence of one.
# ---------------------------------------------------------------------------

#: A one-carbon ligand, so "one atom is N A outside the box" is literally true
#: and the reading is the out-of-box charge rather than a sum over a ligand
#: that also spilled.
ONE_CARBON_PDBQT = (
    "REMARK  Prepared by Open Docking\n"
    "REMARK  1 atom, 0 rotatable bonds\n"
    "TORSDOF 0\n"
    "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00"
    "     0.000 C \n"
)

#: The box `examples/audit_poses.py` docks into, so the face these readings are
#: taken against is the same box the audit's own figures came from.
OOB_CENTRE = (3.47, 6.21, 8.95)
OOB_SIZE = 18.0
OOB_SPACING = 0.375

#: The three overhangs row 225 names, in the order it names them.
OOB_OVERHANGS = (20.0, 200.0, 2000.0)

#: Row 225's figures, read out of the ledger rather than restated here. A gate
#: that carried its own copy of them would be a second place to forget to
#: update them, which is the failure this is here to close.
OOB_CLAIM_RE = re.compile(
    r"20/200/2000 Å 读数是 \*\*(\d+) / (\d+) / (\d+)\*\*"
)
OOB_SLOPE_RE = re.compile(r"dE/dx 一律 \*\*\+(\d+(?:\.\d+)?)\*\*")


def out_of_box_charges(lig, maps, face_x, overhangs):
    """``(energy, dE/dx)`` for one carbon parked at ``face_x + d``, per ``d``.

    Scored through `score_conformation` -- the same call the audit's descent
    ladder makes -- because that is the path the search charges the penalty on.
    """
    import numpy as np
    from opendocking import score_conformation

    out = []
    for over in overhangs:
        conf = np.zeros(lig.num_dof, dtype=np.float64)
        conf[0] = face_x + over
        energy, grad = score_conformation(lig, maps, conf, "vina")
        out.append((float(energy), float(grad[0])))
    return out


# ---------------------------------------------------------------------------
# Fixtures. Written inline, deterministic, offline.
# ---------------------------------------------------------------------------

#: Residue name for a synthetic chain. Deliberately not a residue RDKit knows:
#: an unknown residue is bonded by proximity, so the fixture's connectivity is
#: decided by its real coordinates rather than by a template's opinion about
#: which atom names belong together.
CHAIN_RESNAME = "CHN"

#: How far apart the two chains of the homodimer fixture are placed. Large on
#: purpose -- a gap of a fraction of an angstrom would make "two components" an
#: accident of a bonding tolerance rather than a fact about the file.
CHAIN_SEPARATION = 30.0

#: A covalent bond is 0.74 A (H2) to about 2.1 A (an S-S disulfide is 2.05 A).
#: These bound the geometry the audit below insists on, so a fixture that
#: collapses atoms or stretches a bond fails instead of quietly changing what is
#: being tested.
MIN_BOND_LENGTH = 0.70
MAX_BOND_LENGTH = 2.20


def build_geometry(smiles: str, seed: int, offset=(0.0, 0.0, 0.0)):
    """Real 3-D coordinates for a small molecule, as ``(element, x, y, z, root)``.

    The embedding is RDKit's own ETKDG followed by an MMFF relaxation, with a
    fixed seed: the geometry is physical (real bond lengths and angles) *and*
    identical on every run, which is what makes an assertion about an exact
    atom count possible. `root` is the heavy atom an atom belongs to -- itself
    for a heavy atom, the atom it is bonded to for a hydrogen -- which is how
    the bonded fixture gives a hydrogen the chain of the heavy atom it hangs
    off rather than the chain of its own index.

    ``Chem.AddHs`` appends hydrogens after the heavy atoms, so the first
    `n_heavy` entries are the heavy atoms in SMILES order. The bonded fixture
    relies on that when it cuts a molecule in half, and the audit checks it
    rather than trusting it.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # pragma: no cover - the SMILES are constants
        raise ValueError(f"bad fixture SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:  # pragma: no cover
        raise ValueError(f"could not embed {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    conf = mol.GetConformer()
    dx, dy, dz = offset
    out = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        root = atom.GetIdx()
        if atom.GetSymbol() == "H":
            nbrs = atom.GetNeighbors()
            root = nbrs[0].GetIdx() if nbrs else atom.GetIdx()
        out.append((atom.GetSymbol(), pos.x + dx, pos.y + dy, pos.z + dz, root))
    return out


def n_heavy(geometry) -> int:
    """How many leading entries of `geometry` are heavy atoms."""
    count = 0
    for element, *_ in geometry:
        if element == "H":
            break
        count += 1
    return count


def heavy_first(geometry) -> bool:
    """True if every heavy atom precedes every hydrogen, as `AddHs` guarantees."""
    seen_h = False
    for element, *_ in geometry:
        if element == "H":
            seen_h = True
        elif seen_h:
            return False
    return True


def expected_pdbqt_atoms(geometry, chain_a: int) -> int:
    """How many ATOM records this chain should produce.

    Counted from the fixture rather than read back from the output, so the
    assertion is against the chemistry: the writer keeps every heavy atom and
    only the hydrogens bonded to N, O or S.
    """
    by_root = {root: element for element, _, _, _, root in geometry if element != "H"}
    total = 0
    for element, _, _, _, root in geometry:
        if element != "H" or by_root.get(root) in ("N", "O", "S"):
            total += 1
    return total


def pdb_atom_line(serial, name, resname, chain, resseq, xyz, element) -> str:
    """One standard 78-column ``ATOM`` record.

    The element goes in columns 77-78, which is where RDKit's PDB reader looks
    for it; putting the atom *name* there instead makes the reader guess.
    """
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}"
        f"{'':>10}{element:>2}"
    )


def write_pdb(path: Path, records) -> Path:
    """Write ``(name, element, resname, chain, resseq, xyz)`` records as a PDB."""
    lines = ["REMARK   1 SYNTHETIC FIXTURE - generated by scripts/prep_check.py"]
    for serial, (name, element, resname, chain, resseq, xyz) in enumerate(records, 1):
        lines.append(pdb_atom_line(serial, name, resname, chain, resseq, xyz, element))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def chain_records(geometry, chain: str, resseq: int = 1, split=None, other_chain=None):
    """Fixture records for one molecule's geometry, labelled with a chain id.

    With `split` set, heavy atoms before it keep `chain` and the rest become
    `other_chain`: a molecule cut across a real bond, which is how a file ends
    up holding two chain ids inside one covalent component. A hydrogen follows
    the chain of the heavy atom it is bonded to, so the cut is chemically
    coherent rather than a label swap on half the hydrogens.
    """
    heavy = n_heavy(geometry)
    threshold = heavy if split is None else split
    counters: dict[str, int] = {}
    out = []
    for element, x, y, z, root in geometry:
        tag = chain if root < threshold else (other_chain or chain)
        if element == "H":
            name = "H"
        else:
            counters[element] = counters.get(element, 0) + 1
            name = f"{element}{counters[element]}"
        out.append((name, element, CHAIN_RESNAME, tag, resseq, (x, y, z)))
    return out


def solvent_records(resseq: int, chain: str = "B"):
    """One water and one zinc ion, far from every chain atom.

    Both are their own covalent component, which is the point: component
    selection runs *before* the water/ion filter, so a monoatomic ion is
    dropped by the selection even under the default ``keep_heterogens=True``.
    The report has to show that rather than leave it in a log line nobody
    reads.
    """
    mid = CHAIN_SEPARATION / 2.0
    return [
        ("O", "O", "HOH", chain, resseq, (mid, 0.0, 12.0)),
        ("ZN", "Zn", "ZN", chain, resseq, (mid, 10.0, 12.0)),
    ]


def single_atom_records(resnames, chain="A", start=(12.0, 3.0, 7.0), pitch=4.0):
    """One atom per residue, laid out on a line so none of them can bond.

    These are the fixtures for inputs that are little more than a metal site or
    a puddle of water. They matter because a single-atom component is the case
    ``PathToSubmol`` got worst: ``PathToSubmol(m, [0])`` returns an *empty*
    molecule, so under the old selection code these receptors prepared to
    nothing at all.
    """
    out = []
    for i, resname in enumerate(resnames):
        element = "O" if resname in prep._WATER_RESNAMES else "Zn"
        out.append(
            (resname, element, resname, chain, i + 1,
             (start[0] + pitch * i, start[1], start[2]))
        )
    return out


def read_back(path: Path):
    """Load a fixture the way `prepare_receptor` does, for identity checks."""
    from rdkit import Chem

    return Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)


def atom_identity(atom):
    """A stable key for one atom, used to compare two molecules by content.

    Atom indices are useless for this: ``PathToSubmol`` renumbers, so comparing
    indices would have compared two different things and found them equal. The
    residue record travels with the atom, so it does not.
    """
    info = atom.GetPDBResidueInfo()
    if info is None:
        return (atom.GetSymbol(),)
    return (
        atom.GetSymbol(),
        (info.GetChainId() or " ").strip(),
        (info.GetResidueName() or "").strip(),
        (info.GetName() or "").strip(),
        int(info.GetResidueNumber()),
        int(info.GetSerialNumber()),
    )


def identities(mol):
    return {atom_identity(a) for a in mol.GetAtoms()}


def coords_of(records):
    return [r[5] for r in records]


def min_pair_distance(points) -> float:
    """Closest approach between two *different* entries of one point list."""
    best = float("inf")
    for i, p in enumerate(points):
        for q in points[i + 1:]:
            d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2) ** 0.5
            best = min(best, d)
    return best


def min_gap(a, b) -> float:
    """Closest approach between two disjoint point lists."""
    best = float("inf")
    for p in a:
        for q in b:
            d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2) ** 0.5
            best = min(best, d)
    return best


def read_with_rdkit(path: Path):
    """Load a fixture the way `prepare_receptor` loads it."""
    from rdkit import Chem

    return Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)


def atom_lines(text: str):
    return [line for line in text.splitlines() if line.startswith("ATOM")]


def chains_in_pdbqt(text: str) -> set:
    """The chain ids present in a prepared PDBQT, read from column 22."""
    return {line[21:22].strip() for line in atom_lines(text)}


def pdbqt_coords(text: str):
    """The ``(x, y, z)`` of every ATOM record, read from columns 31-54."""
    return [
        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in atom_lines(text)
    ]


def coord_key(xyz):
    """A coordinate key that both sides of a comparison format identically.

    The fixture and the PDBQT writer both round to three decimals, from the same
    double, so formatting the same way on both sides makes the keys comparable
    without a tolerance that could hide a real difference.
    """
    return tuple(f"{v:.3f}" for v in xyz)


def expected_identities(geometry, records):
    """What the writer should emit for one chain: identity keys, not just a count.

    Every heavy atom, plus only the hydrogens bonded to N, O or S. Derived from
    the geometry (which knows which heavy atom each hydrogen hangs off) zipped
    with the records (which know the atom names), so the expectation cannot
    drift away from the names the fixture was actually written with.
    """
    by_root = {root: el for el, _, _, _, root in geometry if el != "H"}
    out = set()
    for (el, _, _, _, root), rec in zip(geometry, records):
        if el == "H" and by_root.get(root) not in ("N", "O", "S"):
            continue
        name, _element, resname, chain, resseq, xyz = rec
        out.add((chain, resname, name, resseq) + coord_key(xyz))
    return out


def pdbqt_identities(text: str):
    """The same key, read back out of a prepared PDBQT."""
    out = set()
    for line in atom_lines(text):
        out.add(
            (
                line[21:22].strip(),
                line[17:20].strip(),
                line[12:16].strip(),
                int(line[22:26]),
            )
            + (line[30:38].strip(), line[38:46].strip(), line[46:54].strip())
        )
    return out


def prepare(path: Path, **kwargs):
    """Prepare a fixture, returning ``(text, report, warnings)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text, report = prep.prepare_receptor_with_report(path, **kwargs)
    return text, report, list(caught)


def hydrogen_census(path):
    """What the writer was given, what it wrote, and the difference.

    Returns ``(hd_on_disk, capture, unprotonated, multiply, dropped)``.

    The molecule is captured by wrapping the module's own writer and its own
    `AddHs` call rather than by re-running preparation here: a second pipeline
    is a second implementation, and the whole point of this is to count what
    the product actually did. The writer's keep/drop rule is then applied to
    the molecule it received, so the dropped set is the product's decision and
    not an inference from the output text.
    """
    import warnings as _warnings
    import rdkit.Chem.AllChem as _allchem

    cap = {}
    real_writer = prep._molecule_to_pdbqt
    real_addhs = _allchem.AddHs

    def _writer(mol, resname="UNL"):
        cap["mol"] = mol
        return real_writer(mol, resname=resname)

    def _addhs(mol, *a, **kw):
        cap["donors"] = list(kw.get("onlyOnAtoms") or [])
        return real_addhs(mol, *a, **kw)

    prep._molecule_to_pdbqt = _writer
    _allchem.AddHs = _addhs
    try:
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            text, _report = prep.prepare_receptor_with_report(path)
    finally:
        prep._molecule_to_pdbqt = real_writer
        _allchem.AddHs = real_addhs

    mol = cap["mol"]
    conf = mol.GetConformer()
    cap["h_added"] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
    dropped = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        nbrs = atom.GetNeighbors()
        if not nbrs or nbrs[0].GetSymbol() not in ("N", "O", "S"):
            dropped.append((atom, "not bonded to N/O/S"))
            continue
        sep = conf.GetAtomPosition(atom.GetIdx()).Distance(
            conf.GetAtomPosition(nbrs[0].GetIdx()))
        if not (0.7 <= sep <= 1.35):
            dropped.append((atom, f"{sep:.2f} A from its parent"))
    unprot, multi = [], []
    for idx in cap.get("donors", []):
        atom = mol.GetAtomWithIdx(idx)
        # `_residue_identity` returns (name, res, chain, resseq) -- the atom
        # name FIRST. Unpacked the other way round, every label below loses its
        # residue and these assertions fail for a reason that has nothing to do
        # with the chemistry.
        name, res, _chain, seq = prep._residue_identity(atom)
        tag = f"{res} {seq} {name}".strip()
        hydrogens = [n for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
        if not hydrogens:
            unprot.append(tag)
        elif len(hydrogens) > 1:
            multi.append(tag)
    hd = sum(1 for ln in text.splitlines()
             if ln.startswith(("ATOM", "HETATM")) and ln[77:79].strip() == "HD")
    return hd, cap, unprot, multi, dropped


def component_warnings(caught) -> list:
    """The component-loss warnings, ignoring the per-atom hydrogen ones."""
    return [
        w
        for w in caught
        if w.category is RuntimeWarning and "polar hydrogen" not in str(w.message)
    ]


def guarded(fn, *args, **kwargs):
    """Call `fn`, turning a crash into a value the caller can report on.

    Returns ``(value, crash)``. A script that dies on the first exception in
    the code under test reports nothing about the checks after it, so a crash
    has to be caught and *counted* -- otherwise a section ends up with a header
    and no checks, which is the failure this file's sibling once shipped.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - any crash is a result
        return None, f"{type(exc).__name__}: {exc}"


def is_frozen(report) -> bool:
    """True if a caller cannot overwrite a report field.

    Runs on a *copy*: a report that is not frozen would otherwise be mutated
    here and every later check would be reading corrupted numbers.
    """
    copy = replace(report)
    try:
        copy.atoms_dropped = 99
    except Exception:
        return True
    return False


# ---------------------------------------------------------------------------
# The two-chain expectations, as data, so they can be run twice.
# ---------------------------------------------------------------------------


def two_chain_expectations(report, kept: str, dropped: str, atoms_kept: int):
    """What the homodimer fixture has to satisfy, as ``(name, ok, detail)``.

    Returned as data rather than run immediately, because the same list is
    applied both to the real report and to a deliberately corrupted one. That
    is how the guard is shown to be capable of failing.
    """
    out = []

    def want(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    want("the two-chain file reports two chains in", report.chains_in == 2,
         f"chains_in={report.chains_in}")
    want("the two-chain file reports exactly one chain kept", report.chains_kept == 1,
         f"chains_kept={report.chains_kept}")
    want("the two-chain file reports one chain missing", report.chains_dropped == 1,
         f"chains_dropped={report.chains_dropped}")
    want("the chain kept is named, and it is the larger one",
         report.chain_ids_kept == (kept,),
         f"chain_ids_kept={report.chain_ids_kept}, expected ({kept!r},)")
    want("the chain dropped is named, and it is the other one",
         report.chain_ids_dropped == (dropped,),
         f"chain_ids_dropped={report.chain_ids_dropped}, expected ({dropped!r},)")
    want("the kept chain and the dropped chain are different",
         bool(report.chain_ids_kept) and bool(report.chain_ids_dropped)
         and set(report.chain_ids_kept).isdisjoint(report.chain_ids_dropped),
         f"kept {report.chain_ids_kept}, dropped {report.chain_ids_dropped}")
    want("the file has more than one component", report.fragments_in > 1,
         f"fragments_in={report.fragments_in}")
    want("something was dropped", report.atoms_dropped > 0,
         f"atoms_dropped={report.atoms_dropped}")
    buckets = report.water_atoms_removed + report.ion_atoms_removed + report.other_atoms_removed
    want("the loss accounts for every dropped atom", buckets == report.atoms_dropped,
         f"water {report.water_atoms_removed} + ion {report.ion_atoms_removed} + "
         f"other {report.other_atoms_removed} = {buckets}, "
         f"atoms_dropped={report.atoms_dropped}")
    want("the atoms kept match the kept chain counted outside preparation",
         report.atoms_kept == atoms_kept,
         f"atoms_kept={report.atoms_kept}, the fixture's chain {kept!r} has "
         f"{atoms_kept} atoms")
    want("atoms_in is atoms_kept plus atoms_dropped",
         report.atoms_in == report.atoms_kept + report.atoms_dropped,
         f"{report.atoms_in} != {report.atoms_kept} + {report.atoms_dropped}")
    want("the report cannot be edited by a caller", is_frozen(report),
         "a mutable report is a report two parts of a GUI can disagree about")
    return out


# ---------------------------------------------------------------------------


def main() -> int:
    global CHECKS, FAILURES
    print(f"opendocking.prep from {Path(prep.__file__).parent}")

    section("the package under test exposes the report API")
    has_api = hasattr(prep, "ReceptorPrepReport") and hasattr(
        prep, "prepare_receptor_with_report"
    )
    check("opendocking.prep exposes ReceptorPrepReport and prepare_receptor_with_report",
          has_api,
          f"{Path(prep.__file__).name} {'has' if has_api else 'has no'} the report API -- "
          f"is PYTHONPATH pointing at the source tree?")
    if not has_api:
        return finish()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # -------------------------------------------------------------------
        section("the fixtures are physically possible")

        # Chain B is the bigger one on purpose: if selection ever became "keep
        # the first chain", this fixture would catch it.
        geom_a = build_geometry("NCCO", 0xA01)
        geom_b = build_geometry("NCCCO", 0xB02, offset=(CHAIN_SEPARATION, 0.0, 0.0))
        rec_a = chain_records(geom_a, "A")
        rec_b = chain_records(geom_b, "B")
        rec_solvent = solvent_records(2)
        dimer_pdb = write_pdb(tmp / "dimer.pdb", rec_a + rec_b + rec_solvent)
        single_pdb = write_pdb(tmp / "single.pdb", rec_a)

        # One molecule cut across a real bond: two chain ids, one component.
        geom_linked = build_geometry("NCCCCO", 0xC03)
        linked_pdb = write_pdb(
            tmp / "linked.pdb",
            chain_records(geom_linked, "A", split=n_heavy(geom_linked) // 2,
                          other_chain="B"),
        )

        everything = rec_a + rec_b + rec_solvent + chain_records(geom_linked, "A")
        positions = {tuple(round(v, 3) for v in r[5]) for r in everything}
        check("no two atoms in any fixture share a position",
              len(positions) == len(everything),
              f"{len(everything)} atoms, {len(positions)} distinct positions")

        closest = min_pair_distance(coords_of(rec_a))
        check("the closest approach inside a chain is a covalent bond, not a pile-up",
              closest >= MIN_BOND_LENGTH,
              f"closest pair within a chain is {closest:.2f} A "
              f"(an O-H bond is about 0.97 A)")

        gap = min_gap(coords_of(rec_a), coords_of(rec_b))
        check("the two chains of the dimer cannot possibly be one component",
              gap >= 10.0,
              f"closest approach between chains A and B is {gap:.2f} A")

        solvent_gap = min_gap(coords_of(rec_a) + coords_of(rec_b), coords_of(rec_solvent))
        check("the water and the ion are not bonded to the protein",
              solvent_gap >= 5.0,
              f"closest solvent-to-protein approach is {solvent_gap:.2f} A")

        check("hydrogens follow their heavy atoms in the fixture builders",
              heavy_first(geom_a) and heavy_first(geom_linked),
              "AddHs appends hydrogens last, which is what the chain split relies on")

        # The fixtures only test what they are meant to test if RDKit reads
        # them the way they were built. Recomputed here rather than assumed.
        from rdkit import Chem

        rd = read_with_rdkit(dimer_pdb)
        rd_frags = Chem.GetMolFrags(rd, asMols=False, sanitizeFrags=False)
        rd_chains = {a.GetPDBResidueInfo().GetChainId().strip() for a in rd.GetAtoms()}
        check("RDKit reads every atom the fixture wrote",
              rd.GetNumAtoms() == len(rec_a) + len(rec_b) + len(rec_solvent),
              f"{rd.GetNumAtoms()} read, {len(rec_a) + len(rec_b) + len(rec_solvent)} written")
        check("RDKit reads the dimer fixture as two chains in separate components",
              len(rd_frags) == 4 and rd_chains == {"A", "B"},
              f"{len(rd_frags)} components, chains {sorted(rd_chains)}")

        rd_single = read_with_rdkit(single_pdb)
        check("RDKit reads the single-chain fixture as one component",
              len(Chem.GetMolFrags(rd_single, asMols=False, sanitizeFrags=False)) == 1,
              f"{len(Chem.GetMolFrags(rd_single, asMols=False, sanitizeFrags=False))} components")

        rd_linked = read_with_rdkit(linked_pdb)
        linked_frags = Chem.GetMolFrags(rd_linked, asMols=False, sanitizeFrags=False)
        check("RDKit reads the bonded fixture as ONE component, so it is a real "
              "chain/fragment disagreement and not a mis-built file",
              len(linked_frags) == 1,
              f"{len(linked_frags)} components")

        lengths = []
        for m in (rd, rd_single, rd_linked):
            conf = m.GetConformer()
            for bond in m.GetBonds():
                lengths.append(
                    conf.GetAtomPosition(bond.GetBeginAtomIdx()).Distance(
                        conf.GetAtomPosition(bond.GetEndAtomIdx())
                    )
                )
        check("every bond RDKit inferred is a real bond length",
              MIN_BOND_LENGTH <= min(lengths) and max(lengths) <= MAX_BOND_LENGTH,
              f"{len(lengths)} bonds, shortest {min(lengths):.2f} A, "
              f"longest {max(lengths):.2f} A")

        # -------------------------------------------------------------------
        section("a two-chain receptor: the shape that broke the benchmark")

        text, report, caught = prepare(dimer_pdb)
        for name, ok, detail in two_chain_expectations(
            report, "B", "A", len(rec_b)
        ):
            check(name, ok, detail)

        check("the prepared receptor really contains only the kept chain",
              chains_in_pdbqt(text) == {"B"},
              f"chain ids in the PDBQT: {sorted(chains_in_pdbqt(text))}")
        expected = expected_pdbqt_atoms(geom_b, 1)
        check("the prepared receptor holds the kept chain's atoms and its polar hydrogens",
              len(atom_lines(text)) == expected,
              f"{len(atom_lines(text))} ATOM records, the fixture's chain B contributes "
              f"{expected} heavy atoms and N/O/S hydrogens")

        # The kept component has to be *exactly* the kept chain, in both
        # directions. `PathToSubmol` is a path walk, not a set selector, and on
        # RDKit 2026.03.1 it both added atoms from a component that had just
        # been reported as discarded and returned an *empty* molecule for a
        # single-atom component. Comparing sets rather than counts covers both:
        # a stray atom fails the first half, a lost atom fails the second, and
        # a count adjusted to compensate would fail both.
        prepared_ids = pdbqt_identities(text)
        expected_ids = expected_identities(geom_b, rec_b)
        check("the prepared receptor is exactly the kept chain, atom for atom",
              prepared_ids == expected_ids,
              f"{len(prepared_ids - expected_ids)} atoms in the receptor that are not "
              f"in the kept chain, {len(expected_ids - prepared_ids)} atoms of the "
              f"kept chain missing from the receptor")
        check("nothing from a discarded component survives into the receptor",
              chains_in_pdbqt(text) == {"B"}
              and min_gap(pdbqt_coords(text), coords_of(rec_a)) >= 5.0,
              f"chain ids {sorted(chains_in_pdbqt(text))}, closest prepared atom to "
              f"the discarded chain is "
              f"{min_gap(pdbqt_coords(text), coords_of(rec_a)):.2f} A")

        print(f"\n  report:  {report.summary()}")
        print(f"  warning: {report.component_warning()}\n")

        # -------------------------------------------------------------------
        section("a single-chain receptor: nothing dropped")

        _, single_report, single_caught = prepare(single_pdb)
        check("the single-chain file is one component", single_report.fragments_in == 1,
              f"fragments_in={single_report.fragments_in}")
        check("the single-chain file is one chain", single_report.chains_in == 1,
              f"chains_in={single_report.chains_in}, chain_ids_kept={single_report.chain_ids_kept}")
        check("the single-chain file keeps every atom", single_report.atoms_dropped == 0,
              f"atoms_dropped={single_report.atoms_dropped} "
              f"(water {single_report.water_atoms_removed}, "
              f"ion {single_report.ion_atoms_removed}, "
              f"other {single_report.other_atoms_removed})")
        check("the single-chain report is not lossy", single_report.is_lossy is False,
              f"is_lossy={single_report.is_lossy}")
        check("no chain is reported missing",
              single_report.chains_dropped == 0 and single_report.chain_ids_dropped == (),
              f"chains_dropped={single_report.chains_dropped}, "
              f"chain_ids_dropped={single_report.chain_ids_dropped}")
        check("atoms_in equals atoms_kept",
              single_report.atoms_in == single_report.atoms_kept,
              f"{single_report.atoms_in} in, {single_report.atoms_kept} kept")
        check("no component warning is raised for a lossless receptor",
              component_warnings(single_caught) == [],
              f"{len(single_caught)} warnings in total, "
              f"{len(component_warnings(single_caught))} of them component warnings")

        print(f"\n  report:  {single_report.summary()}\n")

        # -------------------------------------------------------------------
        section("the warning is generated from the report")

        comp = component_warnings(caught)
        check("the lossy receptor raises exactly one component warning",
              len(comp) == 1,
              f"{len(caught)} warnings in total, {len(comp)} of them component warnings")
        if comp:
            message = str(comp[0].message)
            check("the warning text is exactly the report's own text",
                  message == report.component_warning(),
                  "the RuntimeWarning and the ReceptorPrepReport cannot disagree")
            check("the warning says one of two chains was lost",
                  "1 chain of 2 chains" in message
                  and "chain(s) 'A' were dropped" in message,
                  "the old warning counted fragments and atoms and never said a chain was lost")
            check("every count in the warning is the report's count",
                  all(str(v) in message
                      for v in (report.chains_in, report.chains_kept, report.atoms_in,
                                report.atoms_kept, report.atoms_dropped)),
                  f"checked {report.chains_in}/{report.chains_kept} chains, "
                  f"{report.atoms_in}/{report.atoms_kept}/{report.atoms_dropped} atoms")
            check("the warning is attributed to the calling code, not to prep.py",
                  Path(comp[0].filename).resolve() == Path(__file__).resolve(),
                  f"reported at {Path(comp[0].filename).name}:{comp[0].lineno}")

        # A hand-written warning string would pass the "same text" check above.
        # This is the one that cannot: change a number in the report and the
        # text has to change with it.
        mutated = replace(report, chains_in=7)
        check("the warning text follows the report's fields",
              mutated.component_warning() != report.component_warning()
              and "1 chain of 7 chains" in mutated.component_warning(),
              "a hardcoded string would survive this")

        # -------------------------------------------------------------------
        section("chains and fragments counted separately")

        linked_text, linked_report, _ = prepare(linked_pdb)
        check("the bonded two-chain file is one component", linked_report.fragments_in == 1,
              f"fragments_in={linked_report.fragments_in}")
        check("the bonded two-chain file is two chains", linked_report.chains_in == 2,
              f"chains_in={linked_report.chains_in}, ids {linked_report.chain_ids_kept}")
        check("a chain count copied from the component count would be wrong here",
              linked_report.fragments_equal_chains is False,
              f"{linked_report.fragments_in} components against "
              f"{linked_report.chains_in} chains")
        check("both chains of the bonded file are kept", linked_report.chains_kept == 2,
              f"chains_kept={linked_report.chains_kept}")
        check("nothing is dropped when the chains are one component",
              linked_report.atoms_dropped == 0 and linked_report.chain_ids_dropped == (),
              f"atoms_dropped={linked_report.atoms_dropped}")
        check("both chain ids appear in the prepared output",
              chains_in_pdbqt(linked_text) == {"A", "B"},
              f"chain ids in the PDBQT: {sorted(chains_in_pdbqt(linked_text))}")

        # -------------------------------------------------------------------
        section("choosing a chain instead of the largest")

        text_a, report_a, _ = prepare(dimer_pdb, keep_chain="A")
        text_b, report_b, _ = prepare(dimer_pdb, keep_chain="B")
        text_default, report_default, _ = prepare(dimer_pdb)
        check("keep_chain='A' keeps chain A and says so",
              report_a.chain_ids_kept == ("A",) and chains_in_pdbqt(text_a) == {"A"},
              f"kept {report_a.chain_ids_kept}, output chains {sorted(chains_in_pdbqt(text_a))}")
        check("keep_chain='B' keeps chain B and says so",
              report_b.chain_ids_kept == ("B",) and chains_in_pdbqt(text_b) == {"B"},
              f"kept {report_b.chain_ids_kept}, output chains {sorted(chains_in_pdbqt(text_b))}")
        check("the two chains give different receptors", text_a != text_b,
              f"{len(atom_lines(text_a))} against {len(atom_lines(text_b))} atoms")
        check("the rule that was applied is recorded in the report",
              report_a.selection == "chain 'A'"
              and report_default.selection == "largest fragment",
              f"{report_a.selection!r} against {report_default.selection!r}")
        check("the default is unchanged: it still picks the largest component",
              text_default == text_b and report_default.chain_ids_kept == report_b.chain_ids_kept,
              "every existing caller gets exactly what it got before")
        check("asking for chain A is reported as losing chain B",
              report_a.chain_ids_dropped == ("B",),
              f"chain_ids_dropped={report_a.chain_ids_dropped}")
        for fixture, label in ((dimer_pdb, "the two-chain file"),
                               (single_pdb, "even a one-component file")):
            try:
                prepare(fixture, keep_chain="Z")
                check(f"a chain that is not in {label} is an error, not a silent fallback",
                      False, "keep_chain='Z' returned a receptor")
            except ValueError as exc:
                check(f"a chain that is not in {label} is an error, not a silent fallback",
                      True, f"ValueError: {str(exc)[:64]}")

        # -------------------------------------------------------------------
        section("where the other atoms went")

        # A monoatomic ion is its own component, so component selection removes
        # it before the water/ion filter runs, even with the default
        # keep_heterogens=True. That is pre-existing behaviour; what is new is
        # that the report states it instead of leaving it to a log line nobody
        # reads.
        check("the dropped zinc is counted as an ion, not as 'other'",
              report.ion_atoms_removed == 1 and report.water_atoms_removed == 1,
              f"ion={report.ion_atoms_removed}, water={report.water_atoms_removed}")
        check("the dropped chain's atoms are counted as 'other'",
              report.other_atoms_removed == len(rec_a),
              f"other={report.other_atoms_removed}, the fixture's chain A has "
              f"{len(rec_a)} atoms")
        _, kept_waters, _ = prepare(dimer_pdb, keep_waters=True)
        check("keep_waters=True does not rescue a water that lost component selection",
              kept_waters.water_atoms_removed == 1,
              "the water is its own component and the filter runs after the selection")
        _, no_ions, _ = prepare(dimer_pdb, keep_heterogens=False)
        check("keep_heterogens=False cannot undo a component-level loss",
              no_ions.ion_atoms_removed == 1,
              "the ion was already gone before the filter was consulted")

        # -------------------------------------------------------------------
        section("the existing contract is intact")

        with warnings.catch_warnings(record=True) as plain_caught:
            warnings.simplefilter("always")
            plain = prep.prepare_receptor(dimer_pdb)
        check("prepare_receptor still returns a str", isinstance(plain, str),
              type(plain).__name__)
        check("prepare_receptor returns exactly the text of the report call",
              plain == text, f"{len(plain)} characters, identical")
        check("prepare_receptor still raises the component warning",
              len(component_warnings(plain_caught)) == 1,
              f"{len(plain_caught)} warnings in total")
        with warnings.catch_warnings(record=True) as single_plain_caught:
            warnings.simplefilter("always")
            plain_single = prep.prepare_receptor(single_pdb)
        check("a receptor that needs no component warning gets none",
              component_warnings(single_plain_caught) == [],
              f"{len(single_plain_caught)} warnings in total, none of them component warnings")
        check("a file that already models its donor hydrogens gets none added",
              single_report.polar_hydrogens_added == 0
              and not any("polar hydrogen" in str(w.message) for w in single_plain_caught),
              f"polar_hydrogens_added={single_report.polar_hydrogens_added}, and the "
              f"warning count agrees")
        check("the output still ends with TER and every record is 79 columns",
              plain.rstrip().endswith("TER")
              and all(len(line) == 79 for line in atom_lines(plain)),
              f"{len(atom_lines(plain))} ATOM records")

        # The baseline measured on the checked-in crambin example before this
        # change: the same bytes and the same four warnings.
        if CRAMBIN.exists():
            crambin_text, crambin, crambin_caught = prepare(CRAMBIN)
            check("the checked-in crambin example is unchanged: 382 atoms, 4 warnings",
                  crambin.pdbqt_atoms_written == 382 and len(crambin_caught) == 4,
                  f"{crambin.pdbqt_atoms_written} atoms, {len(crambin_caught)} warnings")
            check("pdbqt_atoms_written matches the text that came back",
                  crambin.pdbqt_atoms_written == len(atom_lines(crambin_text)),
                  f"{crambin.pdbqt_atoms_written} reported, "
                  f"{len(atom_lines(crambin_text))} in the text")
            check("crambin is one chain and loses nothing to component selection",
                  crambin.chains_in == 1 and crambin.atoms_dropped == 0
                  and crambin.fragments_in == 1,
                  f"chains_in={crambin.chains_in}, fragments_in={crambin.fragments_in}, "
                  f"atoms_dropped={crambin.atoms_dropped}")
            # A real protein, so this is the donor-hydrogen path in production:
            # an experimental PDB file carries no hydrogens, and the report has
            # to say how many were invented. The "dropped a polar hydrogen"
            # warnings are per-atom and are not what is being checked here.
            donor = [w for w in crambin_caught
                     if str(w.message).startswith("added ")
                     and "polar hydrogen" in str(w.message)]
            check("the polar-hydrogen count in the warning is the count in the report",
                  len(donor) == 1
                  and crambin.polar_hydrogens_added > 0
                  and str(donor[0].message) == crambin.polar_hydrogen_warning(),
                  f"{crambin.polar_hydrogens_added} polar hydrogens, reported and "
                  f"warned with the same text")
            # `summary()` has one clause the warning does not: the `+N polar H`
            # tally. It is the only place a caller reading a single log line
            # learns the input file modelled no hydrogens at all, which is the
            # difference between "the receptor is what I gave you" and "the
            # receptor is what I gave you plus a set of hydrogens I invented".
            # Pinned to the measured 61 rather than to the report's own field,
            # so a regression in the *count* is caught and not just a change in
            # how the count is spelled.
            #
            # **61 is a count of donor SITES, not of hydrogens.** It is
            # `len(indices)` in `_add_receptor_polar_hydrogens`, handed to
            # `AddHs(onlyOnAtoms=...)`. On crambin the molecule that reaches
            # the writer carries 58 hydrogens, 3 of which the writer drops
            # because RDKit put them 13-22 A from their own parent, so 55 `HD`
            # records reach the file. An earlier version of this comment said
            # "61 atoms I made up", which is wrong twice over: 58 were made and
            # 55 survived. The three assertions below are what make the
            # difference checkable rather than merely stated.
            check("the summary line carries the added polar hydrogens as '+N polar H'",
                  crambin.polar_hydrogens_added == 61
                  and "+61 polar H" in crambin.summary(),
                  f"summary reads {crambin.summary()!r}")
            # The other direction, so the guard above cannot be satisfied by a
            # `summary()` that always appends the clause regardless of the count.
            check("and a receptor that needed no hydrogens does not claim any",
                  single_report.polar_hydrogens_added == 0
                  and "polar H" not in single_report.summary(),
                  f"summary reads {single_report.summary()!r}")
        # -------------------------------------------------------------------
        else:  # pragma: no cover
            check("the checked-in crambin example exists", False, f"{CRAMBIN} is missing")


        # The hydrogen census: what the report promised against what is on disk
        # -------------------------------------------------------------------
        # `polar_hydrogens_added` counts donor SITES. The file counts
        # hydrogens. The two are not the same quantity, the difference is
        # non-zero, and every number derived from treating them as one is a net
        # of errors in opposite directions -- which is why the identity below is
        # asserted as arithmetic instead of being narrated in a comment.
        #
        # Nothing here needs a product change. The writer's own rule is applied
        # to the molecule it is given, and the `HD` records are read off the
        # text that was written.
        hd, _cap, _unprot, _multi, _dropped = hydrogen_census(CRAMBIN)
        written = len(atom_lines(crambin_text))
        heavy = written - hd
        gap = crambin.polar_hydrogens_added - hd
        check(
            "the polar-hydrogen count on disk is lower than the donor-site "
            "count the report publishes, and the difference is the sites that "
            "got no hydrogen",
            heavy == crambin.atoms_kept
            and hd < crambin.polar_hydrogens_added
            and gap == len(_unprot),
            f"{crambin.polar_hydrogens_added} donor site(s) published, "
            f"{hd} `HD` record(s) written beside {heavy} heavy atom(s); the "
            f"difference is {gap}, and it is the count of donor sites that "
            f"were handed to AddHs and came back with nothing",
        )
        # The six sites, enumerated rather than counted. A count plus a
        # residue-name prefix is weaker than it looks: on this file every
        # unprotonated site *is* a CYS SG, so a check that only asks "is it a
        # cysteine" cannot tell "all six thiols" from "all six cysteines,
        # whatever they are". Pinning the tuple makes the chemistry the thing
        # under test, which is what a red here should mean.
        CRAMBIN_UNPROTONATED = ("CYS 16 SG", "CYS 26 SG", "CYS 3 SG", "CYS 32 SG",
                                "CYS 4 SG", "CYS 40 SG")
        check(
            "the donor sites that got no hydrogen are the six cysteine thiols, "
            "enumerated from the molecule rather than counted",
            len(_unprot) == gap
            and tuple(sorted(_unprot)) == CRAMBIN_UNPROTONATED,
            f"{len(_unprot)} unprotonated donor site(s): "
            f"{', '.join(_unprot) or 'none'}, against the pinned set "
            f"{', '.join(CRAMBIN_UNPROTONATED)}. The tuple is pinned rather "
            f"than a count, so a chemistry change -- a seventh thiol, a "
            f"different atom -- goes red for the right reason, and the names "
            f"are printed so the red is diagnosable",
        )
        check(
            "the published count decomposes into sites that got none, sites "
            "that got two, and hydrogens the writer dropped",
            _cap["h_added"] - len(_dropped) == hd
            and len(_unprot) - len(_multi) + len(_dropped) == gap,
            f"{_cap['h_added']} hydrogen(s) in the molecule the writer received, "
            f"{len(_dropped)} dropped for want of a position, {hd} written; "
            f"and {len(_unprot)} unprotonated - {len(_multi)} over-protonated "
            f"({', '.join(_multi) or 'none'}) + {len(_dropped)} dropped = "
            f"{len(_unprot) - len(_multi) + len(_dropped)}, which is the "
            f"published difference {gap}. **The net means nothing on its own**, "
            f"and reading it as a count of lost hydrogens sends someone to fix "
            f"the wrong thing",
        )

        # -------------------------------------------------------------------
        section("a receptor that is little more than a metal site")

        # Every molecule this module holds is a `Mol`, and a `Mol` has no
        # `RemoveAtom` on RDKit 2026.03.1 -- so the water/ion filter's removal
        # raised `AttributeError` on the first input that ever reached it. A
        # water or a monoatomic ion is normally its own component and is gone
        # before the filter runs, which is what hid the crash. A file that is
        # only an ion does reach it, because then the ion *is* the kept
        # component.
        zinc_pdb = write_pdb(tmp / "zinc.pdb", single_atom_records(["ZN"]))
        made, crash = guarded(prepare, zinc_pdb)
        check("preparing a lone zinc does not crash", crash is None,
              crash or "1 component, 1 atom")
        if made is None:
            check("the rest of the zinc section ran", False,
                  f"skipped: {crash}, so nothing downstream of it could be checked")
        else:
            zinc_text, zinc, _ = made
            check("a lone zinc survives the default preparation",
                  zinc.atoms_kept == 1 and zinc.ion_atoms_removed == 0
                  and len(atom_lines(zinc_text)) == 1,
                  f"kept {zinc.atoms_kept}, ion_removed {zinc.ion_atoms_removed}, "
                  f"{len(atom_lines(zinc_text))} ATOM records")
            made2, crash2 = guarded(prepare, zinc_pdb, keep_heterogens=False)
            check("dropping heterogens on a lone zinc does not crash", crash2 is None,
                  crash2 or "this is the path that used to raise AttributeError")
            if made2 is None:
                check("the keep_heterogens=False section ran", False,
                      f"skipped: {crash2}")
            else:
                zinc_text2, zinc2, _ = made2
                check("keep_heterogens=False removes an ion that is in the kept component",
                      zinc2.ion_atoms_removed == 1 and zinc2.atoms_kept == 0
                      and zinc2.atoms_dropped == 1 and not atom_lines(zinc_text2),
                      f"ion_removed {zinc2.ion_atoms_removed}, kept {zinc2.atoms_kept}, "
                      f"{len(atom_lines(zinc_text2))} ATOM records -- the filter ran")
                check("the ion is the difference between the two flags and nothing else",
                      zinc_text2 != zinc_text
                      and (zinc.atoms_in, zinc.fragments_in)
                      == (zinc2.atoms_in, zinc2.fragments_in)
                      and (zinc.water_atoms_removed, zinc.other_atoms_removed)
                      == (zinc2.water_atoms_removed, zinc2.other_atoms_removed),
                      f"{zinc.atoms_in} atoms and {zinc.fragments_in} components in "
                      f"both; only ion_atoms_removed moves, 0 -> "
                      f"{zinc2.ion_atoms_removed}")
                check("an emptied receptor reports one anonymous chain, not zero",
                      zinc2.chain_ids_kept == (" ",) and zinc2.chains_in == 1,
                      f"chains_in {zinc2.chains_in}, chain_ids_kept "
                      f"{zinc2.chain_ids_kept}")
        check("the anonymous chain renders as (blank), not as an empty string",
              prep._render_chain_ids((" ",)) == "(blank)"
              and prep._render_chain_ids(("A", "B")) == "'A', 'B'",
              f"blank -> {prep._render_chain_ids((' ',))!r}, named -> "
              f"{prep._render_chain_ids(('A', 'B'))!r}")

        # Two single-atom components: the case `PathToSubmol` got worst, since
        # `PathToSubmol(m, [0])` returns an empty molecule.
        waters_records = single_atom_records(["HOH", "HOH"])
        waters_pdb = write_pdb(tmp / "waters.pdb", waters_records)
        check("the two waters are two components",
              len(Chem.GetMolFrags(read_with_rdkit(waters_pdb), asMols=False,
                                   sanitizeFrags=False)) == 2,
              "2 components")
        kept_waters, crash = guarded(prepare, waters_pdb, keep_waters=True)
        check("keeping waters does not crash", crash is None, crash or "2 components")
        dry_waters, crash2 = guarded(prepare, waters_pdb)
        check("dropping waters does not crash", crash2 is None, crash2 or "2 components")
        if kept_waters is None or dry_waters is None:
            check("the rest of the waters section ran", False,
                  f"skipped after {crash or crash2}")
        else:
            waters_text, waters, _ = kept_waters
            waters_text2, waters2, _ = dry_waters
            check("keeping waters keeps exactly one of them, not none",
                  waters.atoms_kept == 1 and waters.water_atoms_removed == 1
                  and len(atom_lines(waters_text)) == 1,
                  f"kept {waters.atoms_kept}, water_removed "
                  f"{waters.water_atoms_removed}, {len(atom_lines(waters_text))} "
                  f"ATOM records (PathToSubmol would have returned 0)")
            # Both components are one atom, so the tie-break decides which
            # survives. It is the first, and this pins that.
            first = waters_records[0]
            kept_water = ((first[3], first[2], first[0], first[4])
                          + coord_key(first[5]))
            check("the water that survives is the one the tie-break picks",
                  pdbqt_identities(waters_text) == {kept_water},
                  f"kept {sorted(pdbqt_identities(waters_text))}, expected "
                  f"{[kept_water]}")
            check("dropping waters drops both, one by selection and one by the filter",
                  waters2.atoms_kept == 0 and waters2.water_atoms_removed == 2
                  and waters2.atoms_dropped == 2 and not atom_lines(waters_text2),
                  f"water_removed {waters2.water_atoms_removed}, kept "
                  f"{waters2.atoms_kept}")

        # -------------------------------------------------------------------
        section("the other input formats the CLI and the docs promise")

        round_pdbqt = tmp / "round.pdbqt"
        round_pdbqt.write_text(prep.prepare_receptor(single_pdb), encoding="utf-8")
        rt_text, rt, _ = prepare(round_pdbqt)
        check("a .pdbqt is accepted and read back as one chain in one component",
              rt.atoms_in > 0 and rt.chains_in == 1 and rt.fragments_in == 1
              and rt.atoms_dropped == 0,
              f"{rt.atoms_in} atoms, {rt.chains_in} chain(s) {rt.chain_ids_kept}, "
              f"{rt.fragments_in} component(s), {rt.atoms_dropped} dropped")
        check("a .pdbqt round trip is reported honestly",
              rt.pdbqt_atoms_written == len(atom_lines(rt_text)) > 0,
              f"{rt.pdbqt_atoms_written} reported, {len(atom_lines(rt_text))} in the text")
        ent = tmp / "same.ent"
        ent.write_text(single_pdb.read_text(encoding="utf-8"), encoding="utf-8")
        _, ent_report, _ = prepare(ent)
        check("a .ent is accepted and gives the same answer as the .pdb",
              ent_report.atoms_in == single_report.atoms_in
              and ent_report.chains_kept == single_report.chains_kept
              and ent_report.atoms_dropped == single_report.atoms_dropped
              and ent_report.source.endswith(".ent"),
              f"{ent_report.atoms_in} atoms as the .pdb, chain "
              f"{ent_report.chain_ids_kept}, source {ent_report.source}")
        try:
            prepare(tmp / "wrong.sdf")
            check("an unsupported extension is rejected", False, "a .sdf was accepted")
        except ValueError as exc:
            check("an unsupported extension is rejected", True, str(exc)[:58])

        # -------------------------------------------------------------------
        section("a receptor whose chain id is blank")

        blank_pdb = write_pdb(tmp / "blank.pdb", chain_records(geom_a, " "))
        _, blank, _ = prepare(blank_pdb)
        check("a blank chain id is one chain, not zero",
              blank.chains_in == 1 and blank.chains_kept == 1
              and blank.chain_ids_kept == (" ",),
              f"chains_in {blank.chains_in}, chain_ids_kept {blank.chain_ids_kept}")
        check("a blank chain id is rendered rather than printed as nothing",
              "(blank)" in blank.summary(),
              f"summary ends {blank.summary()[-30:]!r}")
        check("a blank chain id is not confused with a named one",
              blank.chain_ids_kept != single_report.chain_ids_kept
              and "(blank)" not in single_report.summary(),
              f"blank {blank.chain_ids_kept} against named "
              f"{single_report.chain_ids_kept}")

        # The `not ids` fallback inside _chain_ids is not reachable from a
        # file: RDKit attaches a record to every atom it reads, even when the
        # file was written with blank residue names (measured -- a file of bare
        # coordinates came back with records on all three atoms). It is
        # reachable from a molecule that has lost every atom, which is the zinc
        # case above, and directly on an in-memory molecule.
        bare = Chem.MolFromSmiles("CCO")
        check("a molecule with no residue records at all is one anonymous chain",
              all(a.GetPDBResidueInfo() is None for a in bare.GetAtoms())
              and prep._chain_ids(bare) == (" ",),
              f"no record on any of {bare.GetNumAtoms()} atoms, "
              f"_chain_ids -> {prep._chain_ids(bare)}")

        # -------------------------------------------------------------------
        section("the corrected ledger rows are re-measured, not just re-typed")

        import numpy as np
        from opendocking import GridBox, Ligand, Receptor

        one_carbon = tmp / "one_carbon.pdbqt"
        one_carbon.write_text(ONE_CARBON_PDBQT, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oob_rec = Receptor.from_pdbqt(
                str(ROOT / "examples" / "1crn_prep.pdbqt")
            )
        oob_lig = Ligand.from_pdbqt(str(one_carbon))
        oob_box = GridBox.from_center_size(
            OOB_CENTRE, (OOB_SIZE, OOB_SIZE, OOB_SIZE)
        )
        oob_maps = oob_rec.precalculate(
            oob_box, scoring="vina", spacing=OOB_SPACING
        )
        measured = out_of_box_charges(
            oob_lig, oob_maps, float(oob_box.max_corner[0]), OOB_OVERHANGS
        )

        ledger = (ROOT / "docs" / "VERIFICATION.md").read_text(encoding="utf-8")
        claim = OOB_CLAIM_RE.search(ledger)
        slope = OOB_SLOPE_RE.search(ledger)
        charged = [f"{e:.0f}" for e, _ in measured]
        published = [f"{float(g):.0f}" for g in claim.groups()] if claim else []

        check("the out-of-box charge row 225 publishes is the charge the engine makes",
              bool(claim) and charged == published,
              f"the ledger publishes {published or 'nothing'} for overhangs "
              f"{list(OOB_OVERHANGS)} A; the engine charged {charged}. The charge is "
              f"1000.0 x the overhang, so the flat step this row used to describe "
              f"would show up here as three equal numbers")

        want_slope = float(slope.group(1)) if slope else None
        slopes = [g for _, g in measured]
        check("row 225's dE/dx is the one the engine reports, and it points outward",
              want_slope is not None
              and all(abs(g - want_slope) < 1e-6 for g in slopes)
              and all(g > 0.0 for g in slopes),
              (f"the ledger publishes +{want_slope}; the engine reports "
               f"{[round(g, 6) for g in slopes]}. A positive dE/dx is the whole "
               f"fix: it means -grad points back into the box, where the old "
               f"-1000.000 pointed out of it")
              if slope is not None else
              (f"the ledger publishes no dE/dx in the form this check reads, so "
               f"there is nothing to compare against the engine's "
               f"{[round(g, 6) for g in slopes]}. The old -1000.000 cannot be "
               f"published in this slot either: it is the sign that was wrong, "
               f"and a reader must not be able to put it back"))

        # A reader that quietly found nothing would make the two checks above
        # vacuously true, so it is shown refusing text with no figures in it.
        check("the reader refuses a row that publishes no figure, rather than passing",
              OOB_CLAIM_RE.search(
                  "20/200/2000 Å 读数是 **20000 / 200000 / 2000000**"
              ) is not None
              and OOB_CLAIM_RE.search(
                  "the out-of-box row no longer quotes a figure"
              ) is None,
              "the same pattern finds a real row and declines text carrying no "
              "figure, so a re-worded row 225 goes red instead of going unchecked")

        # -------------------------------------------------------------------
        section("reverse verification: these guards can fail")

        # 1. The harness itself. A check function that always returned True
        #    would make every other check in this file meaningless.
        saved_checks, saved_failures = CHECKS, list(FAILURES)
        harness_failed = False
        probe = None
        # Run the deliberately failing check against scratch counters, so the
        # self-test cannot be mistaken for a real failure -- or hide one.
        CHECKS, FAILURES = 0, []
        try:
            with redirect_stdout(StringIO()):
                returned = check("a deliberately false condition", False, "self-test")
            probe = (repr(returned), len(FAILURES), CHECKS)
            harness_failed = returned is False and len(FAILURES) == 1 and CHECKS == 1
        finally:
            CHECKS, FAILURES = saved_checks, saved_failures
        check("the harness records a failing check as a failure", harness_failed,
              f"a false condition gave {probe} - it must give (False, 1, 1)")

        # 2. The two-chain expectations, run against a report that claims the
        #    opposite. Same predicate list, so a guard that cannot reject this
        #    cannot be trusted on the real report either.
        lying = replace(report, chains_in=1, chains_kept=1, chain_ids_kept=("A",),
                        chain_ids_dropped=())
        rejected = [
            name
            for name, ok, _ in two_chain_expectations(lying, "B", "A", len(rec_b))
            if not ok
        ]
        check("the two-chain expectations reject a report claiming one chain in, one kept",
              len(rejected) >= 3,
              f"{len(rejected)} of 12 expectations failed on the tampered report")
        print("      rejected: " + "; ".join(rejected))

        # 3. The lossless direction: a report that invents a loss has to be
        #    rejected too, or "always reports something dropped" would pass.
        phantom = replace(single_report, atoms_dropped=7, water_atoms_removed=7)
        check("the single-chain expectations reject a report that invents a loss",
              phantom.atoms_dropped != 0
              and phantom.atoms_in != phantom.atoms_kept + phantom.atoms_dropped,
              f"a phantom loss of {phantom.atoms_dropped} atoms does not add up: "
              f"{phantom.atoms_in} != {phantom.atoms_kept} + {phantom.atoms_dropped}")

        # 4. The old warning text, which is the shape the 1HVR bug took.
        check("the old fragment-and-atom-only wording is gone",
              "kept the largest of 4 fragments" not in report.component_warning(),
              "it counted fragments and atoms and never mentioned a chain")

        # The pin, asserted on the way out rather than only in the
        # declaration. The `+ 1` is this check, which has not been counted yet
        # when the comparison is built.
        #
        # Only on the path that got this far: `main` returns early when the
        # package does not import, and that path has already gone red on
        # `has_api`, so a total that does not add up there is not the number
        # this pin is about.
        check("this file's own count is the count it declares",
              CHECKS + 1 == EXPECTED_CHECKS,
              f"{CHECKS} ran before this one and {EXPECTED_CHECKS} are declared")

    return finish()


if __name__ == "__main__":
    sys.exit(main())
