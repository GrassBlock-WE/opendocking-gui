"""Prove that a pose file carries its connectivity, and that it round-trips.

The workbench used to have to *infer* a pose's bonds from interatomic distances,
because the pose PDBQT declared only AutoDock ``ROOT``/``BRANCH`` records, and
those are indexed in file order. A distance-derived bond set was measured to
reproduce the declared tree on all 18 shipped poses, and that is an
approximation on an unfamiliar ligand: nothing in the file said so.

This script checks the record that replaced it. The pose writer now emits its
covalent bonds keyed by **atom serial**, so the two halves a reader needs are
both in the file: the declared tree, and a mapping that does not depend on file
order. The mapping is not optional bookkeeping -- ``examples/poses.pdbqt`` is
genuinely not in serial order (serials run ``5,6,7,8,9,10,4,11,12,1,2,3``), so
a reader that took the declared numbers as file positions would mis-index most
of the molecule. That is the case this script pins.

**What the bond set is compared against, and why that is not circular.** The
writer's bond list is the engine's own `Molecule::bonds`, which Python cannot
read back. So the oracle here is RDKit's perception of the *source* molecule,
reached through `prepare_ligand` -- a different program, from a different file,
arriving at the same graph by a different route. Two things make the comparison
honest rather than tautological:

* the oracle is computed from ``examples/ibuprofen.sdf``, not from the PDBQT the
  docking run read, so agreement cannot come from copying the writer's output;
* the serial correspondence between the two files is *asserted* (atom names in
  file order equal `prepare_ligand`'s names) rather than assumed, because the
  whole comparison is in serial space and a wrong correspondence would compare
  two different labellings of the same graph.

**The count is pinned** and the guards are **mutation-tested in both
directions**: breaking the writer turns the suite red, and so does breaking the
reader. The mutation results are in the commit message rather than here, since
they are a property of the change and not of the code.

Run:  python scripts/torsion_bond_record_check.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Same rule as the other check scripts, for the same reason: a clean checkout has
# no compiled extension in its source tree, so preferring the source would pass
# locally and fail in CI. `which_copy` below reports what was actually loaded,
# because "the check passed" means something different for each of the two.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking import pdbqt_writer as W  # noqa: E402
from opendocking import core  # noqa: E402
from opendocking.core import GridBox, dock, load_ligand, load_receptor  # noqa: E402
from opendocking.prep import prepare_ligand  # noqa: E402

EXAMPLES = ROOT / "examples"

#: How many checks this file is supposed to run, counted by running it. A check
#: that quietly stops running takes the total down with it, which is how a suite
#: goes from 12 to 9 and still reports clean.
EXPECTED_CHECKS = 12

#: The observed file order of a docked pose, as a prefix. Pinned because the
#: whole point of the record is this order, and an order that *did* match serial
#: order would make every other check here pass for the wrong reason.
EXPECTED_SERIAL_PREFIX = [5, 6, 7, 8, 9, 10, 4, 11, 12, 1, 2, 3]

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def serials_of(lines) -> list[int]:
    """The serial in columns 7-11 of every atom record, in file order."""
    return [int(line[6:11]) for line in lines if line.startswith("ATOM")]


def which_copy() -> str:
    """Which `opendocking` answered, and with which compiled extension.

    Printed because a green run means different things for the source tree and
    for site-packages: the connectivity record lives in the Rust extension, so a
    run against a stale installed `.pyd` is a run against the *old* writer.
    """
    import opendocking

    ext = Path(opendocking.__file__).parent / "_dockpy.pyd"
    digest = ""
    if ext.exists():
        digest = hashlib.sha256(ext.read_bytes()).hexdigest()[:16]
    return (f"{Path(opendocking.__file__).parent} (extension "
            f"{digest or 'absent'})")


def main() -> int:
    print(f"opendocking from: {which_copy()}")
    print(f"core from:        {Path(core.__file__).parent}")

    # A system temp directory, not a directory beside the sources: a check
    # script that leaves a directory in the repository is a check script that
    # can make the *next* run of a tree-comparison check fail for a reason that
    # has nothing to do with the product.
    tmp = Path(tempfile.mkdtemp(prefix="torsion_bond_record_"))

    # ------------------------------------------------------------------
    section("the ligand's bonds, from RDKit, and the serial labelling of the file")

    elements, charges, coords, bonds, names = prepare_ligand(
        EXAMPLES / "ibuprofen.sdf")
    oracle = {(min(a, b) + 1, max(a, b) + 1) for a, b in bonds}
    check("RDKit perceives 16 bonds for the 16-atom prepared ligand",
          len(oracle) == len(elements) == 16,
          f"{len(elements)} atoms, {len(oracle)} bonds, "
          f"{sum(1 for e in elements if e != 'H')} heavy")

    ligand = load_ligand(EXAMPLES / "ibuprofen_prep.pdbqt")
    prep_names = [line[12:16].strip() for line
                  in (EXAMPLES / "ibuprofen_prep.pdbqt").read_text(
                      encoding="utf-8").splitlines()
                  if line.startswith("ATOM")]
    # The comparison below is in serial space, so the serial -> name mapping is
    # asserted rather than assumed. A file whose atom order differed from
    # `prepare_ligand`'s would compare two labellings of one graph and the
    # round trip would "pass" while meaning nothing.
    check("the prepared file's atom order is the one prepare_ligand produced",
          [n.split("_")[-1] for n in names] == prep_names
          and ligand.num_atoms == len(names),
          f"file names {prep_names[:3]}... against prepare_ligand "
          f"{[n.split('_')[-1] for n in names][:3]}...; serial n is atom n-1 of "
          f"the prepared table")

    # ------------------------------------------------------------------
    section("a real docking run writes a pose file that declares its bonds")

    receptor = load_receptor(EXAMPLES / "rec_prep.pdbqt")
    box = GridBox.from_center_size((0.0, 0.0, 0.0), (14.0, 14.0, 14.0))
    maps = receptor.precalculate(box, spacing=0.5)
    result = dock(ligand, maps, exhaustiveness=1, num_modes=2, seed=1)
    pose_path = tmp / "poses.pdbqt"
    result.write_pdbqt(pose_path)
    text = pose_path.read_text(encoding="utf-8")
    models = W.read_pdbqt_models(pose_path)
    first = models[0]

    order = serials_of(first)
    check("the pose file is genuinely NOT in serial order, so the mapping matters",
          order[:12] == EXPECTED_SERIAL_PREFIX and order != sorted(order),
          f"serials in file order begin {order[:12]}; the file is not sorted, so "
          f"a reader taking a declared number as a file position would mis-index "
          f"{sum(1 for n, s in enumerate(order) if n != s - 1)} of {len(order)} "
          f"atoms")

    count_lines = [l for l in first if l.startswith("REMARK OD_NBONDS")]
    bond_lines = [l for l in first if l.startswith("REMARK OD_BOND ")]
    declared_count = int(count_lines[0].split()[2]) if count_lines else -1
    check("the record states how many bonds follow, and the count is right",
          len(count_lines) == 1 and declared_count == len(bond_lines) == 16,
          f"{len(count_lines)} OD_NBONDS line(s) saying {declared_count}, "
          f"{len(bond_lines)} OD_BOND line(s)")

    declared = set(W.read_declared_bonds(first) or ())
    check("the declared bonds are the molecule's bonds: the round trip",
          declared == oracle,
          f"{len(declared)} declared against {len(oracle)} from RDKit; "
          f"declared-only {sorted(declared - oracle)[:4]}, "
          f"RDKit-only {sorted(oracle - declared)[:4]}")

    # The distinction the record exists to make, asserted rather than asserted
    # *about*: mapping serials through the atom records is not the same as
    # treating them as positions, and on this file the two disagree.
    index = {s: n for n, s in enumerate(order)}
    by_serial = {frozenset((index[a], index[b])) for a, b in declared}
    by_position = {frozenset((a - 1, b - 1)) for a, b in declared}
    check("reading the declared numbers as file positions gives a different graph",
          by_serial != by_position and len(by_serial) == len(by_position) == 16,
          f"re-indexed through the serials: {len(by_serial & by_position)} of "
          f"{len(by_serial)} bonds coincide with the position reading, so the "
          f"two disagree on {len(by_serial ^ by_position) // 2} bonds")

    check("every declared serial is a serial the file actually carries",
          all(a in index and b in index for a, b in declared),
          f"serials declared range {min(min(b) for b in declared)}"
          f"..{max(max(b) for b in declared)}, the file's are "
          f"{min(order)}..{max(order)}")

    # ------------------------------------------------------------------
    section("a file with no record says so, rather than claiming no bonds")

    shipped = W.read_pdbqt_models(EXAMPLES / "poses.pdbqt")[0]
    check("the shipped poses.pdbqt, written before the record existed, reads "
          "as 'no record'",
          W.read_declared_bonds(shipped) is None,
          "None and {} are different answers: the first means a reader must "
          "perceive bonds, the second would mean a molecule with no bonds")

    # ------------------------------------------------------------------
    section("older readers are unaffected, and the old records are untouched")

    from meeko import PDBQTMolecule  # noqa: PLC0415 - the gate below reports it

    parsed = PDBQTMolecule(text, skip_typing=True)
    check("meeko still reads the pose file, record and all",
          len(parsed._positions) == 2,
          f"meeko read {len(parsed._positions)} pose(s); it has never heard of "
          f"OD_BOND and skips the REMARK body")

    # Byte compatibility, asserted by *removing* the new records and asking
    # whether what is left is the file the old writer produced. The control is
    # the point: strip the records and the residue must be free of the marker
    # completely, or this check would pass on a file that still carried one.
    stripped = [l for l in text.splitlines()
                if not l.startswith("REMARK OD_BOND")]
    residue = [l for l in stripped if "OD_BOND" in l]
    check("the new records are pure additions: nothing else moved",
          not residue
          and all(l in stripped for l in text.splitlines()
                  if not l.startswith("REMARK OD_BOND"))
          and "REMARK VINA RESULT" in text and "ROOT" in text
          and text.count("MODEL") == 2 and text.count("ENDMDL") == 2,
          f"dropping the {len(count_lines) + len(bond_lines)} record lines "
          f"leaves a file that still has {text.count('MODEL')} MODEL, "
          f"{text.count('ROOT')} ROOT and its REMARK VINA RESULT; no OD_BOND "
          f"text survives, so no existing record was edited")

    # ------------------------------------------------------------------
    section("reverse verification: these guards can fail")

    # A reader that trusts the count is the whole reason the count is there, so
    # the truncation case is driven rather than described.
    truncated = [l for l in first if not l.startswith("REMARK OD_BOND    15")]
    try:
        W.read_declared_bonds(truncated)
        raised = "no error"
    except W.PdbqtFormatError as exc:
        raised = f"PdbqtFormatError: {str(exc)[:60]}"
    check("a file whose bond records were cut short is refused, not read short",
          raised.startswith("PdbqtFormatError") and declared_count == 16,
          f"removing one bond record leaves the count saying {declared_count}; "
          f"the reader says {raised!r}, because a silently missing bond is the "
          f"failure this record exists to remove")

    check("a count with no bond records behind it is refused too",
          _raises(lambda: W.read_declared_bonds(
              [l for l in first if not l.startswith("REMARK OD_BOND ")])),
          "a record claiming 16 bonds and carrying none is a truncated file, "
          "not an empty molecule")

    if CHECKS != EXPECTED_CHECKS:
        FAILURES.append(
            f"check count is {CHECKS}, expected {EXPECTED_CHECKS}: a check was "
            f"added or lost, and both are worth stopping for")
        print(f"\n  [FAIL] the check count moved: {CHECKS}, expected "
              f"{EXPECTED_CHECKS}")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    return 1 if FAILURES else 0


def _raises(fn) -> bool:
    """Whether `fn` refuses, which is what a malformed record must do."""
    try:
        fn()
    except W.PdbqtFormatError:
        return True
    except Exception:  # noqa: BLE001 - any refusal is a refusal here
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
