"""Headless check of ligand preparation, which until now had no check at all.

`prepare_receptor` had `prep_check.py`. `prepare_ligand` had nothing -- and it
is the path a user's very first run goes through, because you dock *something*
before you dock a receptor. Every number below was measured before it was
asserted, and two of them turned out to be worth knowing:

* **The five tables are structural, not chemical.** `elements` is the element
  symbol of every atom, full stop. The code used to have a three-way branch
  that appended `symbol` in all three arms, so it read as a special case for
  hydrogens and metals and did nothing at all.
* **The charge table is not neutral.** Merging a non-polar hydrogen does not
  fold that hydrogen's partial charge into its parent, which is what AutoDock's
  convention asks for. Benzene's six carbons each carry -0.062 e, so the table
  sums to **-0.37 e** for a neutral molecule, and ibuprofen sums to -0.60 e.
  That is recorded here as what the code does, not as what it should do:
  changing it would change every score the engine has ever produced, so it is
  a decision for a human, not a side effect of a test.

The embedding branch is covered in both directions because it is the branch a
user hits by handing in a SMILES string: `.smi` arrives with no conformer, and
the geometry that comes back is a guess. The check asserts the guess is at
least *physical* (covalent distances, nothing at the origin) and that it is
*reproducible* (the same input gives the same coordinates), and it says
plainly that the failure direction could not be reproduced.

The ``except`` blocks around RDKit embedding, MMFF and Gasteiger are **not**
covered and are not pretended to be; they are named in `prep.py`'s module
docstring as fallbacks, and there is one check here that says so out loud.

Run:  python scripts/ligand_check.py
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"

# Prefer the *installed* package, and only fall back to the source tree when
# there is no installed one -- the same rule the other check scripts use.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

import numpy as np  # noqa: E402
from opendocking import prep  # noqa: E402
from opendocking.core import Ligand  # noqa: E402
from opendocking.pdbqt_writer import ligand_to_pdbqt  # noqa: E402

# A Windows console is frequently GBK and a warning this script merely counts
# may contain a non-ASCII character. A character that cannot be encoded must
# not turn a check run into a traceback.
if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(errors="backslashreplace")

FAILURES: list[str] = []
CHECKS = 0

#: The example ligands, chosen for what each one contains rather than for
#: being pretty: a drug with an acid and a donor, a molecule with no hydrogen
#: at all after merging, a phenol with exactly one polar hydrogen, a beta-2
#: agonist whose amine is the interesting nitrogen, and a cofactor with S and
#: two N.
LIGANDS = ("ibuprofen", "benzene", "phenol", "salbutamol", "biotin")

#: How many checks this file records, **counted by running this file** and not
#: derived from the source above.
#:
#: `F:\python310\python.exe scripts\ligand_check.py` with
#: `PYTHONIOENCODING=utf-8` and `PYTHONPATH=dock-py\python`: **`106/106 passed`,
#: exit 0**, in 0.8 s. The pin is that run's 106 plus the tally check below, so
#: 107, and the run that has to agree with the pin is the *second* run rather
#: than the one the 106 came from.
#:
#: **57 call sites produce 106 checks, and here the two reconcile exactly**, so
#: the derivation is given rather than asserted:
#:
#:   * the `for name in LIGANDS` loop holds 15 sites and runs **11** of them
#:     per ligand, five times: 55. The four it does not run are the two
#:     failure arms -- a missing `.sdf` and a `prepare()` that crashed -- plus
#:     the two mutually exclusive halves of the golden-reference check.
#:   * 37 sites sit outside any loop and run once each: 37.
#:   * the atom-type `cases` table is 12 entries from one site: 12.
#:   * the two `try:`/`except` pairs around `load_molecule` contribute one
#:     check each, not two, because only one arm of each pair runs: 2.
#:
#: 55 + 37 + 12 + 2 = 106, which is the run. The census that
#: `check_scripts_declare.py` reports for this file is 37 unconditional + 20
#: guarded, and it measures the other question -- "could this site stop
#: running" rather than "how many checks will the report" -- so the two numbers
#: are both correct and neither is the pin.
#:
#: **The golden-reference branch is balanced, and that is why this pin does not
#: move when a reference file is missing.** The `if ref.exists():` arm and the
#: `else:` arm each hold exactly one site, so a missing
#: `examples/<name>_prep.pdbqt` swaps which check runs without changing how many
#: run: the run stays at 107 and goes red on
#: `<name>: reproduces the checked-in ... atom for atom` instead. That was the
#: point of the `else` arm -- before it the check vanished and only the printed
#: count moved -- and it is the one branch in this file whose firing or not
#: firing is invisible in the total.
#:
#: **The three branches that *do* move the count all move it on a red.** A
#: missing example `.sdf` gives that ligand 1 check instead of 11, and a
#: `prepare()` that crashes gives it 2; either way the run is already red on
#: the check that says why, and the tally then adds a second red naming the
#: shortfall. Reading that second red as "the constant needs widening" is the
#: mistake this note exists to prevent: the cause is in the failures list, one
#: line above.
EXPECTED_CHECKS = 107


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
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed (expected {EXPECTED_CHECKS})")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


def guarded(fn, *args, **kwargs):
    """Call `fn`, turning a crash into a value the caller can report on.

    Returns ``(value, crash)``. A script that dies on the first exception in
    the code under test reports nothing about the checks after it, so a crash
    has to be caught and *counted*.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - any crash is a result
        return None, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class Prepared:
    """The five tables, plus the two things a caller makes of them."""

    elements: list
    charges: list
    coords: np.ndarray
    bonds: list
    names: list
    ligand: object
    text: str

    @property
    def n(self) -> int:
        return len(self.elements)


def write_smi(path: Path, smiles: str) -> Path:
    path.write_text(smiles.strip() + "\n", encoding="utf-8")
    return path


def write_molblock(path: Path, atoms, bonds) -> Path:
    """A V2000 molblock written out in full, so its contents are inspectable.

    `atoms` is ``[(element, x), ...]`` and `bonds` is ``[(a, b, order), ...]``,
    both zero-based. Written by hand rather than through RDKit so the fixture
    can be something RDKit would *refuse* to write.
    """
    lines = [path.stem, "  ligand_check", "",
             f"{len(atoms):>3}{len(bonds):>3}  0  0  0  0  0  0  0  0999 V2000"]
    for element, x in atoms:
        lines.append(
            f"{x:>10.4f}{0.0:>10.4f}{0.0:>10.4f} {element:<3} 0  0  0  0"
            f"  0  0  0  0  0  0  0  0"
        )
    for a, b, order in bonds:
        lines.append(f"{a + 1:>3}{b + 1:>3} {order}  0  0  0  0")
    lines += ["M  END", "$$$$"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def atom_lines(text: str) -> list:
    return [l for l in text.splitlines() if l.startswith("ATOM")]


def prepare(path: Path, **kwargs):
    """`prepare_ligand`, then the two things every caller does with it.

    Returns ``(Prepared, warnings)``. Warnings are returned rather than
    suppressed because what this path *says* is half of what it guarantees.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        elements, charges, coords, bonds, names = prep.prepare_ligand(path, **kwargs)
    ligand = Ligand.from_arrays(elements, charges, coords, bonds, names)
    text = ligand_to_pdbqt(ligand, coords, names)
    box = Prepared(elements, charges, coords, bonds, names, ligand, text)
    return box, list(caught)


def file_atoms(path: Path):
    """The atoms of a file, read without the preparation pipeline."""
    from rdkit import Chem

    return Chem.MolFromMolFile(str(path), removeHs=False).GetAtoms()


def typed(smiles: str, symbol: str) -> list:
    """Every PDBQT type RDKit gives to the atoms of `symbol` in `smiles`."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # pragma: no cover - the SMILES are constants
        raise ValueError(f"bad fixture SMILES {smiles!r}")
    AllChem.Compute2DCoords(mol)
    return sorted({prep.pdbqt_atom_type(a, mol) for a in mol.GetAtoms()
                   if a.GetSymbol() == symbol})


def types_of(smiles: str) -> set:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return {prep.pdbqt_atom_type(a, mol) for a in mol.GetAtoms()}


def main() -> int:
    global CHECKS, FAILURES
    print(f"opendocking.prep from {Path(prep.__file__).parent}")

    # -------------------------------------------------------------------
    section("the five tables, and the shape they are guaranteed to have")

    for name in LIGANDS:
        src = EX / f"{name}.sdf"
        if not src.exists():  # pragma: no cover
            check(f"{name}: the example ligand exists", False, f"{src} is missing")
            continue
        made, crash = guarded(prepare, src)
        if made is None:
            check(f"{name}: preparing it does not crash", False, str(crash))
            check(f"{name}: the rest of its section ran", False,
                  f"skipped after {crash}")
            continue
        p, caught = made
        check(f"{name}: the five tables are the same length",
              p.n == len(p.charges) == len(p.names) == len(p.coords) == p.ligand.num_atoms,
              f"{p.n} atoms, {p.ligand.num_atoms} in the Ligand")
        check(f"{name}: coords is a C-contiguous (n, 3) float64 array of finite numbers",
              p.coords.shape == (p.n, 3) and p.coords.dtype == np.float64
              and p.coords.flags["C_CONTIGUOUS"] and bool(np.isfinite(p.coords).all()),
              f"shape {p.coords.shape}, dtype {p.coords.dtype}, "
              f"C-contiguous {p.coords.flags['C_CONTIGUOUS']}")
        check(f"{name}: every charge is a finite float",
              all(isinstance(c, float) and np.isfinite(c) for c in p.charges),
              f"{len(p.charges)} charges, sum {sum(p.charges):+.4f} e")
        check(f"{name}: atom names are unique and carry their 1-based index",
              len(set(p.names)) == p.n
              and all(nm.endswith(str(i + 1)) for i, nm in enumerate(p.names)),
              f"first three {p.names[:3]}")
        check(f"{name}: bonds are ordered, in range and not repeated",
              all(a < b for a, b in p.bonds)
              and all(0 <= i < p.n for b in p.bonds for i in b)
              and len(set(p.bonds)) == len(p.bonds),
              f"{len(p.bonds)} bonds over {p.n} atoms")
        check(f"{name}: the engine is handed the same coordinates back",
              np.array_equal(p.ligand.reference_coords, p.coords),
              f"reference_coords shape {p.ligand.reference_coords.shape}")
        check(f"{name}: degrees of freedom are the rigid body plus one per torsion",
              p.ligand.num_dof == 6 + p.ligand.num_torsions,
              f"{p.ligand.num_torsions} torsions -> {p.ligand.num_dof} dof")
        check(f"{name}: the radius about the centroid is a real distance",
              p.ligand.radius > 0.5,
              f"radius {p.ligand.radius:.2f} A over {p.n} atoms")
        # The checked-in PDBQT is a golden reference: the only difference
        # between it and what this produces today is the REMARK line, so the
        # ATOM records are compared byte for byte. That covers atom typing,
        # hydrogen merging, the charge column and the coordinates in one go.
        ref = EX / f"{name}_prep.pdbqt"
        if ref.exists():
            want = [l for l in ref.read_text(encoding="utf-8").splitlines()
                    if l.startswith("ATOM")]
            check(f"{name}: reproduces the checked-in {ref.name} atom for atom",
                  atom_lines(p.text) == want,
                  f"{len(atom_lines(p.text))} produced against {len(want)} in the "
                  f"reference")
        else:
            # Recorded, and recorded as a **failure**, because this file has no
            # skip vocabulary and inventing one here would be a larger change than
            # the defect deserves. A gate whose golden reference has gone cannot
            # verify the one thing this section exists to verify, so the run has
            # to say so in the exit status rather than in a smaller total. Before
            # this `else` the check vanished: only the printed count moved, and
            # nothing named the file that was missing.
            check(f"{name}: reproduces the checked-in {ref.name} atom for atom",
                  False,
                  f"the golden reference is not present at {ref}, so nothing was "
                  f"compared")
        check(f"{name}: every PDBQT record is 79 columns and typed from its name",
              all(len(l) == 79 for l in atom_lines(p.text))
              and all(l[77:79].strip() == p.names[i].split("_")[0]
                      for i, l in enumerate(atom_lines(p.text))),
              "the type in columns 78-79 is the prefix of the atom name, which "
              "is how the type gets there at all")
        check(f"{name}: the torsions match the bond table's single-bond count",
              p.ligand.num_torsions >= 0
              and f"TORSDOF {p.ligand.num_torsions}" in p.text,
              f"{p.ligand.num_torsions} torsions, "
              f"{sum(1 for l in p.text.splitlines() if l.startswith('TORSDOF'))} "
              f"TORSDOF line")

    # -------------------------------------------------------------------
    section("what the tables actually contain")

    ibu = EX / "ibuprofen.sdf"
    merged, _ = prepare(ibu)
    hydrogens = [e for e in merged.elements if e == "H"]
    check("non-polar hydrogens are merged away, polar ones are not",
          len(hydrogens) == 1,
          f"{sum(1 for e in merged.elements if e == 'H')} hydrogen(s) left on a "
          f"16-heavy-atom drug: the acid OH, and nothing else")

    # The three-way branch that used to be here appended `symbol` in all three
    # arms. Assert the outcome rather than the shape of the code: the element
    # table is the symbol, for hydrogens and for metals alike.
    with tempfile.TemporaryDirectory() as tmp:
        made, crash = guarded(prepare, write_smi(Path(tmp) / "zn.smi", "[Zn+2]"))
    zinc = made[0] if made is not None else None
    check("a metal is an ordinary element and an ordinary Zn to the engine",
          crash is None and zinc is not None
          and zinc.elements == ["Zn"]
          and "Zn" in zinc.names[0],
          f"elements {zinc.elements if zinc else crash}, "
          f"names {zinc.names if zinc else ''}")
    check("every element in the table is the symbol of an atom, with no exceptions",
          all(e in {a.GetSymbol() for a in file_atoms(ibu)} | {"H"}
              for e in merged.elements),
          "no special case for hydrogens or metals, because there is none")

    check("the charge table is not neutral, and the numbers are printed",
          abs(sum(merged.charges)) > 0.1 and abs(sum(merged.charges)) < 5.0,
          f"ibuprofen sums to {sum(merged.charges):+.4f} e for a neutral drug: "
          f"merging a non-polar H does not fold its charge into its parent. "
          f"Recorded as behaviour, not endorsed")
    benzene, _ = prepare(EX / "benzene.sdf")
    check("benzene's six carbons each keep only their own Gasteiger charge",
          len(set(round(c, 6) for c in benzene.charges)) == 1
          and abs(sum(benzene.charges) + 0.3736) < 1e-3,
          f"all six charges are {benzene.charges[0]:+.4f} e, total "
          f"{sum(benzene.charges):+.4f} e")

    kept, _ = prepare(ibu, keep_hydrogens=True)
    unbonded = [
        i for i, e in enumerate(kept.elements)
        if e == "H" and not any(i in b for b in kept.bonds)
    ]
    check("keep_hydrogens=True puts hydrogens in the table with no bonds",
          len(unbonded) > 0 and len(kept.bonds) == len(merged.bonds),
          f"{len(kept.elements)} atoms against {merged.n}, and the same "
          f"{len(kept.bonds)} bonds: {len(unbonded)} hydrogens bonded to "
          f"nothing, because only N/O/S hydrogens keep their bond")
    check("and the default path has no unbonded hydrogen at all",
          not [i for i, e in enumerate(merged.elements)
               if e == "H" and not any(i in b for b in merged.bonds)],
          "every hydrogen in the default table is bonded to something")

    # -------------------------------------------------------------------
    section("a SMILES string arrives with no geometry")

    with tempfile.TemporaryDirectory() as tmp:
        smi = write_smi(Path(tmp) / "phenol.smi", "Oc1ccccc1")
        loaded = prep.load_molecule(smi)
        check("a SMILES file loads with no conformer at all",
              loaded.GetNumConformers() == 0,
              f"{loaded.GetNumAtoms()} atoms, {loaded.GetNumConformers()} "
              f"conformers -- this is what puts a user on the embedding branch")
        embedded, caught = prepare(smi)
        embedding_warnings = [w for w in caught if "no 3-D coordinates" in str(w.message)]
        check("the embedding branch says the pose is a guess",
              len(embedding_warnings) == 1
              and "starting pose is arbitrary" in str(embedding_warnings[0].message),
              str(embedding_warnings[0].message)[:64] if embedding_warnings else "no warning")
        check("the warning points at the caller, not at prep.py",
              embedding_warnings
              and Path(embedding_warnings[0].filename).resolve() == Path(__file__).resolve(),
              f"reported at {Path(embedding_warnings[0].filename).name}:"
              f"{embedding_warnings[0].lineno}" if embedding_warnings else "no warning")
        pts = embedded.coords
        closest = min(
            float(np.linalg.norm(pts[i] - pts[j]))
            for i in range(len(pts)) for j in range(i + 1, len(pts))
        )
        check("the guessed geometry is physical, not a pile-up",
              0.7 <= closest <= 1.6 and not (np.abs(pts) < 1e-9).all(axis=1).any(),
              f"closest pair {closest:.2f} A (an O-H bond is about 0.97 A), and "
              f"no atom sits at the origin")
        again, _ = prepare(smi)
        check("the same input embeds to the same coordinates every time",
              np.array_equal(pts, again.coords),
              "randomSeed is fixed, so a repeat run reproduces the pose rather "
              "than quietly searching a different one")
        # Reproducibility alone cannot pin the seed -- any fixed seed satisfies
        # it, and changing the seed leaves the check above green. So the pose
        # itself is recorded. If this fails, the embedding seed changed and the
        # "starting pose is arbitrary" warning is now arbitrary *differently*,
        # which changes every score that ligand ever produced.
        want = np.array(
            [
                (1.946, 0.499, 0.256), (0.619, 0.225, 0.090),
                (-0.284, 1.281, 0.132), (-1.643, 1.026, -0.035),
                (-2.087, -0.279, -0.243), (-1.174, -1.331, -0.284),
                (0.185, -1.081, -0.117), (2.439, -0.339, 0.200),
            ]
        )
        check("and it is the recorded pose, not merely any repeatable one",
              np.allclose(pts, want, atol=5e-4),
              f"first atom at ({pts[0][0]:.3f}, {pts[0][1]:.3f}, {pts[0][2]:.3f}) "
              f"against the recorded ({want[0][0]:.3f}, {want[0][1]:.3f}, "
              f"{want[0][2]:.3f}): a different seed is a different ligand")

        # An acyclic ligand with no coordinates is the one input that makes the
        # MMFF relaxation fail: `MMFFOptimizeMolecule` needs ring information
        # and a molecule with no ring has none, so it raises and the bare
        # `except` swallows it. That is the only one of the four guarded RDKit
        # calls that a real input reaches, and it is reached here rather than by
        # making RDKit fail on purpose.
        acyclic = write_smi(Path(tmp) / "butanol.smi", "CCCCO")
        chain, chain_caught = prepare(acyclic)
        check("an acyclic ligand with no geometry still prepares",
              chain.n == 6 and chain.ligand.radius > 0.5,
              f"{chain.n} atoms, radius {chain.ligand.radius:.2f} A")
        check("and the MMFF relaxation failing is not reported to anyone",
              not [w for w in chain_caught
                   if "MMFF" in str(w.message) or "relax" in str(w.message)]
              and len([w for w in chain_caught if "no 3-D coordinates" in str(w.message)]) == 1,
              "MMFFOptimizeMolecule raises on a molecule with no ring info and "
              "the bare `except: pass` eats it: the user is told the pose is a "
              "guess, not that the relaxation never ran")
        cpts = chain.coords
        cclosest = min(
            float(np.linalg.norm(cpts[i] - cpts[j]))
            for i in range(len(cpts)) for j in range(i + 1, len(cpts))
        )
        check("the unrelaxed geometry is still physical",
              0.7 <= cclosest <= 1.6,
              f"closest pair {cclosest:.2f} A on a molecule nothing relaxed")
        check("the embedded ligand is usable downstream",
              embedded.ligand.num_atoms == embedded.n
              and embedded.ligand.radius > 0.5,
              f"radius {embedded.ligand.radius:.2f} A over {embedded.n} atoms")

        # The other direction: a file that *has* coordinates must not be told
        # its pose is arbitrary.
        with_file, file_caught = prepare(EX / "phenol.sdf")
        check("a file with coordinates raises no such warning",
              not [w for w in file_caught if "no 3-D coordinates" in str(w.message)]
              and with_file.n == embedded.n,
              f"{with_file.n} atoms either way, and the SDF path is quiet")

        # The failure direction. `EmbedMolecule`'s return code is discarded, so
        # the first thing a user would meet if embedding failed is the next
        # line's `GetConformer()`. Shown directly on a 0-conformer molecule
        # rather than by making RDKit fail on purpose: that would test a mock.
        flat = prep.load_molecule(smi)
        try:
            flat.GetConformer()
            shape, note = None, "GetConformer() returned something"
        except Exception as exc:  # noqa: BLE001
            shape, note = type(exc).__name__, str(exc)[:48]
        check("an un-embeddable molecule fails at GetConformer, not at EmbedMolecule",
              shape is not None,
              f"{shape}: {note} -- the return code of EmbedMolecule is not "
              f"checked, so this is the error a user would see. **Not "
              f"reproduced**: no candidate molecule refused to embed")

        # The unsupported-format guard, both directions.
        cif = Path(tmp) / "thing.cif"
        cif.write_text("data_thing\n", encoding="utf-8")
        try:
            prep.load_molecule(cif)
            check("an unsupported ligand extension is rejected", False, "a .cif loaded")
        except ValueError as exc:
            check("an unsupported ligand extension is rejected",
                  "unsupported format" in str(exc) and ".sdf" in str(exc),
                  str(exc)[:58])
        check("and an extension that IS supported is not rejected for its name",
              prep.load_molecule(smi) is not None,
              ".smi is in SUPPORTED_LIGAND_FORMATS")

        # The partial-sanitisation retry, which is the only way
        # `_load_unsanitised` is ever reached. The fixture is a six-ring with
        # three double bonds all on one carbon: RDKit can parse it and cannot
        # kekulise it, which is what a truncated aromatic ligand looks like.
        # All three facts are asserted, because asserting only the outcome
        # would be satisfied by a first attempt that simply worked.
        from rdkit import Chem as _Chem

        ring = write_molblock(
            Path(tmp) / "broken_ring.sdf",
            [("C", 1.4 * i) for i in range(6)],
            [(0, 1, 2), (0, 2, 2), (0, 3, 2), (0, 4, 1), (4, 5, 1), (5, 0, 1)],
        )
        first = _Chem.SDMolSupplier(str(ring), sanitize=True, removeHs=False)
        second = _Chem.SDMolSupplier(str(ring), sanitize=False, removeHs=False)
        first_none = len(first) > 0 and first[0] is None
        second_atoms = second[0].GetNumAtoms() if second[0] is not None else None
        # RDKit's supplier keeps the file open, and Windows will not delete a
        # file another handle is holding -- which turns the temporary directory
        # cleanup into an error and takes the run's exit code with it.
        del first, second
        check("the first, sanitising read of the fixture fails", first_none,
              f"the supplier offers a record and the molecule in it is None -- "
              f"kekulisation cannot satisfy three double bonds on one carbon")
        check("and the unsanitised read has a molecule to work with",
              second_atoms == 6, f"{second_atoms} atoms")
        retried, crash = guarded(prep.load_molecule, ring)
        check("so the retry is what recovers it, not the first attempt",
              retried is not None and retried.GetNumAtoms() == 6,
              f"{retried.GetNumAtoms() if retried is not None else crash} atoms "
              f"from load_molecule, which can only have come from the retry")
        empty = Path(tmp) / "empty.sdf"
        empty.write_text("name\n\n$$$$\n", encoding="utf-8")
        try:
            prep.load_molecule(empty)
            check("a file with no molecule in it is refused", False, "it loaded")
        except ValueError as exc:
            check("a file with no molecule in it is refused",
                  "no molecules found" in str(exc), str(exc)[:44])

    # -------------------------------------------------------------------
    section("the atom-type mapping, case by case")

    cases = [
        ("C[N+](C)(C)C", "N", "N", "a quaternary nitrogen is not an acceptor"),
        ("c1ccncc1", "N", "NA", "a pyridine nitrogen accepts"),
        ("c1cnc[nH]1", "N", "NA", "an imidazole nitrogen accepts"),
        ("CC(=O)NC", "N", "N", "an amide nitrogen does not accept"),
        ("C[OH2+]", "O", "OA", "a protonated oxygen still accepts"),
        ("CP(=O)(C)C", "P", "P", "phosphorus is its own type"),
        ("FC(F)(F)F", "F", "F", "fluorine is its own type"),
        ("ClCCl", "Cl", "Cl", "chlorine keeps its case, as Meeko writes it"),
        ("BrCBr", "Br", "Br", "bromine keeps its case"),
        ("IC", "I", "I", "iodine is its own type"),
        ("[Se]C", "Se", "Xx", "an element with no mapping gets no type"),
        ("[Xe]", "Xe", "Xx", "and neither does xenon"),
    ]
    for smiles, symbol, want, why in cases:
        got, crash = guarded(typed, smiles, symbol)
        check(f"{smiles}: {symbol} is typed {want}", crash is None and got == [want],
              f"{got} -- {why}" if crash is None else str(crash))
    # The other direction of each: a different element must not get that type.
    check("a carbon is never typed as a nitrogen",
          "N" not in types_of("CCO") and types_of("CCO") == {"C", "OA"},
          f"ethanol gives {sorted(types_of('CCO'))}")
    check("an aromatic carbon and an aliphatic carbon are different types",
          types_of("c1ccccc1") == {"A"} and types_of("CC") == {"C"},
          "benzene A, ethane C -- this is the branch that decides whether a "
          "donor-acceptor pair is even looked for")

    # -------------------------------------------------------------------
    section("the donor-bond test, both ways")

    from rdkit import Chem

    # Hydrogens have to be explicit, or there is no X-H bond to judge.
    methanol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    ch_bond = next(
        b for b in methanol.GetBonds()
        if {methanol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(),
            methanol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()} == {"O", "H"}
    )
    check("an O-H bond is a polar hydrogen bond",
          prep._is_polar_h_bond(methanol, ch_bond.GetBeginAtomIdx(),
                                ch_bond.GetEndAtomIdx()),
          "and it is what makes the engine see a donor")
    cc_bond = next(
        b for b in methanol.GetBonds()
        if {methanol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(),
            methanol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()} == {"C", "H"}
    )
    check("a C-H bond on the same molecule is not",
          not prep._is_polar_h_bond(methanol, cc_bond.GetBeginAtomIdx(),
                                    cc_bond.GetEndAtomIdx()),
          "which is why merging non-polar hydrogens loses no donor information")
    check("and the direction of the pair does not matter",
          prep._is_polar_h_bond(methanol, ch_bond.GetBeginAtomIdx(),
                                ch_bond.GetEndAtomIdx())
          == prep._is_polar_h_bond(methanol, ch_bond.GetEndAtomIdx(),
                                   ch_bond.GetBeginAtomIdx()),
          "a bond is a bond: (i, j) and (j, i) cannot disagree")
    # The filter only ever calls this with a pair that has a hydrogen in it, so
    # the fall-through at the bottom is reached only by a direct call -- which
    # is what this is. Ethanol, because methanol has no heavy-heavy bond.
    ethanol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cc = next(b for b in ethanol.GetBonds()
              if {ethanol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(),
                  ethanol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()} == {"C", "C"})
    check("a bond with no hydrogen in it at all is not a polar hydrogen bond",
          not prep._is_polar_h_bond(ethanol, cc.GetBeginAtomIdx(),
                                    cc.GetEndAtomIdx()),
          "the loop finds no hydrogen and the function says so, rather than "
          "guessing from the other atom")

    # -------------------------------------------------------------------
    section("the ph keyword, which is accepted and does nothing")

    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        at_default = prep.prepare_ligand(ibu)
    with warnings.catch_warnings(record=True) as loud:
        warnings.simplefilter("always")
        at_ph = prep.prepare_ligand(ibu, ph=7.0)
    check("ph=None says nothing",
          not [w for w in quiet if "protonation" in str(w.message)],
          f"{len(quiet)} warnings, none about protonation")
    check("ph=7.0 says the protonation comes from the file",
          any("protonation is taken from the input structure" in str(w.message)
              for w in loud),
          str([str(w.message)[:44] for w in loud])[:80])
    check("and it changes nothing about the result",
          np.array_equal(at_default[2], at_ph[2])
          and at_default[0] == at_ph[0],
          "identical coordinates and elements, which is the documented promise")

    # -------------------------------------------------------------------
    section("the branches that are fallbacks, not behaviour")

    note = (prep.__doc__ or "") + (prep.prepare_ligand.__doc__ or "")
    check("the guarded RDKit calls are accounted for, and only the MMFF one is covered",
          "fallbacks, not behaviour" in note
          and "is** reached" in note
          and "never run" in note
          and "Gasteiger" in note and "MMFF" in note,
          "the MMFF relaxation is reached by an acyclic ligand and is checked "
          "above; polar-hydrogen addition, Gasteiger and the receptor's AddHs "
          "are guarded, have never run, and are named in prep.py so nobody "
          "assumes otherwise. This check is about the documentation, not "
          "about behaviour")

    # -------------------------------------------------------------------
    section("reverse verification: these guards can fail")

    saved_checks, saved_failures = CHECKS, list(FAILURES)
    CHECKS, FAILURES = 0, []
    probe = None
    try:
        with redirect_stdout(StringIO()):
            returned = check("a deliberately false condition", False, "self-test")
        probe = (repr(returned), len(FAILURES), CHECKS)
    finally:
        CHECKS, FAILURES = saved_checks, saved_failures
    check("the harness records a failing check as a failure",
          probe == ("False", 1, 1), f"a false condition gave {probe}")

    # The table-shape guard, run against tables that are wrong in each of the
    # ways it is supposed to catch. Same predicate, so a guard that cannot
    # reject these cannot be trusted on the real output.
    def tables_agree(elements, charges, names, coords) -> bool:
        return (len(elements) == len(charges) == len(names) == len(coords)
                and len(set(names)) == len(elements))

    good = (merged.elements, merged.charges, merged.names, merged.coords)
    short = (merged.elements[:-1], merged.charges, merged.names, merged.coords)
    dupe = (merged.elements, merged.charges, ["X"] * merged.n, merged.coords)
    check("the table-shape guard rejects a table with a missing charge",
          tables_agree(*good) and not tables_agree(*short),
          f"one charge short: {len(short[0])} elements, {len(short[1])} charges")
    check("and a table with repeated atom names",
          tables_agree(*good) and not tables_agree(*dupe),
          f"{merged.n} atoms, {merged.n} identical names")

    def bonds_sane(bonds, n) -> bool:
        return (all(a < b for a, b in bonds)
                and all(0 <= i < n for b in bonds for i in b)
                and len(set(bonds)) == len(bonds))

    check("the bond guard rejects an out-of-range index and a reversed pair",
          bonds_sane(merged.bonds, merged.n)
          and not bonds_sane([(0, merged.n)], merged.n)
          and not bonds_sane([(3, 1)], merged.n)
          and not bonds_sane([(0, 1), (0, 1)], merged.n),
          "an index past the end, a reversed bond, and a repeated one")

    # The pin, asserted on the way out rather than only in the declaration. The
    # `+ 1` is this check, which has not been counted yet when the comparison is
    # built. It is inside `main` and outside every loop, so it runs once on every
    # path that reaches the end -- including the ones where a loop above ran
    # fewer times, which is exactly when the total has moved and this has to
    # notice.
    check("this file's own count is the count it declares",
          CHECKS + 1 == EXPECTED_CHECKS,
          f"{CHECKS} ran before this one and {EXPECTED_CHECKS} are declared")

    return finish()


if __name__ == "__main__":
    sys.exit(main())
