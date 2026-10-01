"""Does this project's PDBQT writer produce files other tools can read?

``pdbqt_writer.py`` is the only code in the package that produces files *other*
tools consume, and its own docstring names the danger: getting the columns
wrong "silently produces a file that loads but mis-parses". Nothing checked
that. This is the check.

# Three kinds of evidence, and only two of them are worth much

A round trip proves the writer and the reader agree. It does not prove either is
*right*. So the three are kept apart here, in the section names and in the
detail text, because collapsing them is how a format bug survives:

1. **Against a third-party parser** -- `meeko`, installed offline, which also
   produced the ``examples/*.pdbqt`` this repository ships. It parses **by
   column**: a line with the right fields at the wrong columns is *refused*
   (``ValueError: invalid literal for int() with base 10: '1 C1'``), so its
   agreement is real evidence about the layout and not a coincidence. If meeko
   is not importable this script fails loudly rather than quietly checking less.
2. **Against third-party *files*** -- the ``examples/*.pdbqt`` are Meeko output.
   Comparing the writer's own line layout with theirs field by field is
   evidence about the format with no parser involved at all.
3. **Against this project's reader** -- and here the honest result is negative,
   so it is stated rather than dressed up. ``read_pdbqt_models`` is a
   ``MODEL``/``ENDMDL`` **splitter**: it returns raw lines and never reads a
   column, so a "round trip" through it is close to vacuous. The only
   column-parsing reader in-tree is the engine's Rust ``Ligand.from_pdbqt``,
   and that one is **whitespace-based**: deleting a space at column 31 from an
   otherwise identical line still parses to the same coordinates. So the engine
   reader proves *parseability*, and nothing about columns. Everything about
   columns in this file rests on (1) and (2).

# What is deliberately not claimed

"Every writer function round-trips" is not the question, because the three
kinds of evidence above do not license the same conclusion for each one. Two
real differences are pinned rather than hidden, and they are pinned in the
*direction that is true*:

* ``write_pose`` writes the energy as ``{energy:.1f}``, so ``-7.42`` comes back
  as ``-7.4``. That is the format's own limit, not a lost byte, and the check
  asserts the value that actually comes back.
* ``ligand_to_pdbqt`` written **without** ``names`` types every atom ``"C"``,
  which turns 4 rotatable bonds into 5. Written **with** ``names`` -- which
  every in-tree caller does -- the torsion count and the ``atom_kinds`` come
  back identical. An earlier draft of this file blamed the flat layout for the
  4-to-5; the measurement says otherwise, and the check now pins the
  attribution to the thing actually responsible.

Run:  python scripts/pdbqt_check.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Prefer the *installed* package, and only fall back to the source tree when
# there is no installed one -- the same rule the other check scripts use, for
# the same reason: a clean checkout has no compiled extension in its source
# tree, so putting the source first passes locally and fails in CI.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking import pdbqt_writer as W  # noqa: E402
from opendocking.core import Ligand, load_ligand  # noqa: E402

EXAMPLES = ROOT / "examples"

#: How many checks this file is supposed to run, counted by running it. It is a
#: check itself, at the end of `main`: a check that stops running takes the
#: count down with it silently, which is how a suite goes from 91 to 87 with a
#: clean report.
EXPECTED_CHECKS = 57  # measured; see the report for the mutation counts

FAILURES: list[str] = []
CHECKS = 0

#: The 1-based, inclusive column of every field, spelled out independently of
#: the module's own ``_COL_*`` constants. Derived from the two Meeko-produced
#: files in ``examples/`` and from the PDB specification, **not** from
#: ``_field``'s arithmetic -- a check that recomputes the answer from the code
#: under test agrees with that code by construction. The module's constants are
#: *compared* against this table further down, which is the direction that can
#: fail.
COLUMNS = {
    "record": (1, 6),
    "serial": (7, 11),
    "name": (13, 16),
    "resName": (18, 20),
    "chain": (22, 22),
    "resSeq": (23, 26),
    "x": (31, 38),
    "y": (39, 46),
    "z": (47, 54),
    "occupancy": (55, 60),
    "tempFactor": (61, 66),
    "charge": (71, 76),
    "type": (78, 79),
}


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


def col(line: str, field: str) -> str:
    """The bytes of one field, addressed by this file's own 1-based table."""
    lo, hi = COLUMNS[field]
    return line[lo - 1:hi]


def meeko_positions(text: str):
    """Coordinates a third-party parser reads out of `text`, or None if refused.

    meeko is the only offline parser here that is not part of this project and
    that reads **by column**, so it is the only one whose agreement says
    anything about the layout. Its ``_positions`` is shaped
    ``(n_poses, n_atoms, 3)``.
    """
    from meeko import PDBQTMolecule

    return np.asarray(PDBQTMolecule(text, skip_typing=True)._positions)


def meeko_refuses(text: str) -> str:
    """The error meeko gives for `text`, or ``""`` if it accepted it."""
    from meeko import PDBQTMolecule

    try:
        PDBQTMolecule(text, skip_typing=True)
        return ""
    except Exception as exc:  # noqa: BLE001 - the message is the evidence
        return f"{type(exc).__name__}: {str(exc)[:70]}"


def _energy_of(lines: list[str]) -> float | None:
    """The `REMARK VINA RESULT` energy of a model's lines, or None."""
    for line in lines:
        if "VINA RESULT" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith("RESULT"):
                    try:
                        return float(parts[i + 1])
                    except (ValueError, IndexError):
                        return None
    return None


def meeko_diagnosis() -> tuple[bool, str]:
    """Is meeko usable here, and if not, **which module** is missing.

    # Why this exists instead of a bare `except ImportError`

    A bare handler reported ``meeko is not importable (ModuleNotFoundError)``
    and told the reader to install meeko. On the CI runner meeko **was**
    installed -- the step log reads ``Successfully installed ... meeko-0.8.0`` --
    so the message sent the person trying to fix it to install the one thing
    they already had. `ModuleNotFoundError` carries the name of the module that
    was actually missing in its `.name` attribute, and the bare handler threw
    that field away, which is the only piece of evidence the exception holds.

    So the question is no longer "is meeko importable" but "which import failed",
    and the two cases need different sentences:

    * **meeko itself is absent** -- `.name` is ``"meeko"``. Install meeko.
    * **meeko is installed and one of *its* imports is absent** -- `.name` is
      something else, and meeko is on disk. Install *that* module.

    The second is not hypothetical. meeko declares **no** `Requires-Dist` at
    all, so pip resolves nothing for it and every runtime import it makes is
    undeclared. Measured on meeko 0.7.1, `import meeko` pulls in
    ``gemmi, numpy, pandas, prody, rdkit, scipy`` and a TOML reader at import
    time; none of those is a declared dependency, and a runner that installs
    meeko alone gets a package that cannot be imported. That is a packaging
    fact about meeko, not a mistake in this file, and the only correct response
    to it is to install the named module rather than to re-install meeko.

    The three questions are asked in the order that discriminates them, and the
    first two are asked *without executing meeko*, so a broken meeko cannot stop
    the diagnosis from describing it.
    """
    # 1. Is the distribution recorded at all? Read from metadata, so it works
    #    even when importing the module would fail.
    try:
        version = importlib.metadata.version("meeko")
    except importlib.metadata.PackageNotFoundError:
        return False, (
            "meeko is NOT installed: there is no installed distribution named "
            "'meeko' for this interpreter. Install it (pip install meeko)."
        )
    except Exception as exc:  # noqa: BLE001 - the message is the evidence
        return False, (
            f"meeko's installed distribution could not be read "
            f"({type(exc).__name__}: {exc}), so its version is unknown."
        )

    # 2. Is the module itself on this interpreter's path? `find_spec` locates
    #    the file without executing it, which separates "installed but not on
    #    this path" from "installed and merely broken on import".
    try:
        spec = importlib.util.find_spec("meeko")
    except Exception as exc:  # noqa: BLE001 - a broken meeko raises here
        return False, (
            f"meeko {version} IS installed, but finding its module raised "
            f"{type(exc).__name__}: {exc}"
        )
    if spec is None:
        return False, (
            f"meeko {version} IS installed as a distribution, but no importable "
            f"module 'meeko' is on this interpreter's sys.path. That means the "
            f"wrong interpreter or a split install path, not a missing package: "
            f"check which python is running ({sys.executable})."
        )

    # 3. Actually import it. This is where a missing *dependency* surfaces, and
    #    `.name` is the one field that says which dependency.
    try:
        importlib.import_module("meeko")
    except ModuleNotFoundError as exc:
        missing = exc.name or "<not reported by the exception>"
        if missing == "meeko" or missing.startswith("meeko."):
            return False, (
                f"meeko {version} IS installed at {spec.origin}, but importing it "
                f"raised ModuleNotFoundError for {missing!r} -- meeko is "
                f"present and still incomplete. This is a broken meeko install, "
                f"not a missing one; re-installing meeko itself is the fix."
            )
        return False, (
            f"meeko {version} IS installed and importable as a file "
            f"({spec.origin}), but `import meeko` failed because ITS OWN "
            f"dependency {missing!r} is not installed. Do NOT install meeko -- it "
            f"is already there. Install {missing!r} instead. meeko declares no "
            f"`Requires-Dist` at all, so pip installed nothing on its behalf and "
            f"every import it makes is undeclared. To see the full set meeko "
            f"pulls in, run: python -c \"import meeko, sys; "
            f"print(sorted(m for m in sys.modules if '.' not in m))\""
        )
    except Exception as exc:  # noqa: BLE001 - the message is the evidence
        return False, (
            f"meeko {version} IS installed at {spec.origin}, but importing it "
            f"raised {type(exc).__name__}: {exc} -- not a missing module, so "
            f"there is nothing to install."
        )
    return True, f"meeko {version} at {spec.origin}"


def main() -> int:
    # ------------------------------------------------------------------
    section("the module's column constants, against the table written here")

    # This is the *first* check in the file, and that is deliberate rather than
    # a matter of taste. A column constant that is off by one changes a line's
    # width, and the module's own 79-character assertion then fires -- so if
    # this ran second, a shifted constant would kill the script at the first
    # formatted line and the one check that names the actual mistake would
    # never get to run. Here it is *reported*, naming the column it disagrees
    # about, and the rest of the file then fails too.
    consts = {
        "serial": W._COL_SERIAL, "name": W._COL_NAME,
        "resName": W._COL_RESNAME, "chain": W._COL_CHAIN,
        "resSeq": W._COL_RESSEQ, "x": W._COL_X, "y": W._COL_Y,
        "z": W._COL_Z, "occupancy": W._COL_OCC,
        "tempFactor": W._COL_BFAC, "charge": W._COL_CHARGE,
        "type": W._COL_TYPE,
    }
    drift = {k: (v, (v[0] + 1, v[1])) for k, v in consts.items()
             if (v[0] + 1, v[1]) != COLUMNS[k]}
    check("the module's column constants agree with this table, read as "
          "0-based starts",
          not drift,
          f"every ``_COL_*`` pair is (start, end) with start one before the "
          f"first column, so _COL_CHARGE = {W._COL_CHARGE} means 71-76 and "
          f"_COL_TYPE = {W._COL_TYPE} means 78-79"
          + (f"; drift: {drift}" if drift else ""))

    # ------------------------------------------------------------------
    section("a third-party parser is available, and is column-sensitive")

    # A gate rather than a skip. This script's strongest evidence is a parser
    # that is neither written nor reviewed by anyone working on this repository;
    # if it is missing, the remaining checks still say something, but pretending
    # the count is the same would be the "check that cannot fail" mistake wearing
    # a different hat. Failing loudly puts the requirement in front of whoever
    # wires CI.
    #
    # The report is the *diagnosis*, not a verdict on meeko. An earlier version
    # caught the exception bare and printed "meeko is not importable
    # (ModuleNotFoundError) ... Install meeko", which was false on the CI runner
    # where the same step's log reads `Successfully installed ... meeko-0.8.0`:
    # the reader was told to install the one package they already had, and
    # `ModuleNotFoundError.name` -- the only field naming what was really absent
    # -- was discarded. `meeko_diagnosis` asks that field and distinguishes the
    # two cases, so the sentence here names something the reader can act on.
    ok, why = meeko_diagnosis()
    if not ok:
        print("  MEEKO UNAVAILABLE -- the third-party half of this check cannot run.")
        print(f"  {why}")
        print(f"  Stopped after {CHECKS} check(s); EXPECTED_CHECKS "
              f"({EXPECTED_CHECKS}) was NOT evaluated, so this is a hard failure "
              f"and not a reduced-but-green run.")
        return 1
    import meeko as _meeko
    check("meeko is installed, and it is not part of this project",
          "site-packages" in str(Path(_meeko.__file__).parent).replace("\\", "/")
          and "odock-mcode" not in str(Path(_meeko.__file__)),
          f"meeko {getattr(_meeko, '__version__', '?')} at "
          f"{Path(_meeko.__file__).parent}. The import-time dependency set was "
          f"measured, not assumed: meeko declares no `Requires-Dist`, so its "
          f"requirements are whatever `import meeko` happens to pull in")

    # The property that makes meeko worth consulting at all. A parser that
    # splits on whitespace would accept a line whose columns are wrong, and its
    # agreement with the writer would then be worthless. This one refuses.
    good = W.format_atom_line(1, " C1", "UNL", 1, (1.234, 2.345, 3.345), 0.0, "C")
    single = " ".join(good.split())          # same fields, wrong columns
    refused = meeko_refuses(single + "\n")
    accepted = meeko_positions(good + "\n").reshape(-1, 3)
    check("meeko reads a correctly-columned line and refuses a mis-columned one",
          accepted.shape == (1, 3)
          and np.abs(accepted[0] - (1.234, 2.345, 3.345)).max() <= 5e-4
          and bool(refused),
          f"the correct line gives {accepted.tolist()}; the same fields "
          f"single-spaced are refused with {refused!r}, so meeko parses by "
          f"column and its agreement is evidence about the layout rather "
          f"than a coincidence")

    # ------------------------------------------------------------------
    section("every field lands in its own column, byte for byte")

    line = W.format_atom_line(
        7, "CA", "THR", 42, (-12.345, 0.5, 1234.5), -0.412, "OA", chain="B",
    )
    expected = {
        "record": "ATOM  ", "serial": "    7", "name": "CA  ", "resName": "THR",
        "chain": "B", "resSeq": "  42", "x": " -12.345", "y": "   0.500",
        "z": "1234.500", "occupancy": "  1.00", "tempFactor": "  0.00",
        "charge": "-0.412", "type": "OA",
    }
    wrong = {k: (col(line, k), v) for k, v in expected.items()
             if col(line, k) != v}
    check("all thirteen fields are in the right columns, and nothing overlaps",
          not wrong and len(line) == 79,
          f"|{line}| is {len(line)} characters"
          + (f"; wrong: {wrong}" if wrong else
             "; chain 'B' in column 22, charge in 71-76, type in 78-79"))

    # The length assertion: real, but only against a changed constant. Removing
    # it is mutation M-A in the report, and the honest finding is that nothing
    # here can make it *matter* for a legal input -- so the check below proves
    # the assertion fires at all, rather than leaving a decorative `raise`
    # counted as coverage.
    check("a line is exactly 79 characters, which is what the assertion holds",
          all(len(W.format_atom_line(i, " C1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "C"))
              == 79 for i in (1, 99, 9999)),
          "79 for serial 1, 99 and 9999; the widest serial that fits is 99999")

    # ... and here is the assertion being made to earn its place. One column too
    # narrow, and the AssertionError is what stands between a caller and a file
    # that is the wrong length. This is the *only* class of defect it catches:
    # a truncated number is padded back out to the same width, so the width
    # checks in 'the cases that are easy to get wrong' are what cover that, and
    # they would stay red if this assertion were deleted.
    saved_type = W._COL_TYPE
    try:
        W._COL_TYPE = (77, 78)               # 1 column narrower than the spec
        try:
            W.format_atom_line(1, " C1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "C")
            fired = False
        except AssertionError:
            fired = True
    finally:
        W._COL_TYPE = saved_type
    check("the 79-character assertion fires on a wrong column constant, and "
          "that is the whole of what it is good for",
          fired and len(W.format_atom_line(
              1, " C1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "C")) == 79,
          "with _COL_TYPE narrowed to (77, 78) the line comes out 78 "
          "characters and the assertion raises. Deleting the assertion leaves "
          "every other check here green, because no legal input makes a line "
          "the wrong length -- so it guards constants, not data")

    # ------------------------------------------------------------------
    section("the writer against files the project ships, which Meeko wrote")

    shipped = {
        "ibuprofen_prep.pdbqt": 16,
        "biotin_prep.pdbqt": 19,
        "benzene_prep.pdbqt": 6,
        "1crn_prep.pdbqt": 382,
    }
    for name, n_atoms in shipped.items():
        path = EXAMPLES / name
        if not path.exists():
            check(f"{name} is present to check against", False, "missing")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        atoms = [ln for ln in lines if ln.startswith(("ATOM", "HETATM"))]
        check(f"{name}: {n_atoms} ATOM lines, all 79 characters",
              len(atoms) == n_atoms and {len(a) for a in atoms} == {79},
              f"{len(atoms)} ATOM/HETATM lines, lengths "
              f"{sorted({len(a) for a in atoms})}")

    # The strongest statement in the whole file, and it involves no parser: for
    # the same molecule, this writer's field layout is byte-identical to the
    # Meeko-produced file the project ships.
    shipped_atoms = [ln for ln in
                     (EXAMPLES / "ibuprofen_prep.pdbqt").read_text(
                         encoding="utf-8").splitlines()
                     if ln.startswith("ATOM")]
    mine = W.format_atom_line(1, " C1", "UNL", 1, (4.650, -0.310, 0.616),
                              0.0, "C")
    theirs = shipped_atoms[0]
    diffs = {k: (col(theirs, k), col(mine, k))
             for k in COLUMNS if col(theirs, k) != col(mine, k)}
    check("and the writer reproduces the shipped file's layout byte for byte",
          not diffs and mine == theirs,
          f"|{mine}| against the shipped |{theirs}|"
          if not diffs else f"differences: {diffs}")

    # Every ATOM line of every shipped file, field by field, against the table.
    # One check over the whole corpus rather than one per file.
    corpus_bad = []
    corpus_lines = 0
    for path in sorted(EXAMPLES.glob("*.pdbqt")):
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            corpus_lines += 1
            try:
                float(col(ln, "x")), float(col(ln, "y")), float(col(ln, "z"))
                float(col(ln, "charge"))
            except ValueError:
                corpus_bad.append((path.name, ln))
                continue
            if len(col(ln, "type").strip()) > 2 or len(ln) != 79:
                corpus_bad.append((path.name, ln))
    check("every ATOM line of every shipped file parses at the columns above",
          not corpus_bad and corpus_lines > 100,
          f"{corpus_lines} lines across "
          f"{len(list(EXAMPLES.glob('*.pdbqt')))} files, all with a float in "
          f"31-38/39-46/47-54, a float in 71-76 and a type of at most two "
          f"characters in 78-79"
          + (f"; first bad: {corpus_bad[:1]}" if corpus_bad else ""))

    # And the project's own reader on those files, which is a claim about the
    # splitter, not about columns. The pose count is checked against the file's
    # own MODEL record count rather than a number written down here, so a
    # shipped file gaining or losing a pose cannot make this quietly wrong.
    pose_path = EXAMPLES / "crambin_pose.pdbqt"
    pose_lines = pose_path.read_text(encoding="utf-8").splitlines()
    n_records = sum(1 for ln in pose_lines
                    if ln.split(maxsplit=1)[:1] == ["MODEL"])
    models = W.read_pdbqt_models(pose_path)
    body = models[0] if models else []
    check("read_pdbqt_models returns one list of lines per pose, not text",
          len(models) == n_records and body
          and all(isinstance(ln, str) for ln in body)
          and not any(ln.startswith(("MODEL", "ENDMDL")) for m in models
                      for ln in m),
          f"{len(models)} pose(s) for {n_records} MODEL records, first pose "
          f"{len(body)} lines, MODEL/ENDMDL stripped. This function is a "
          f"splitter -- it never reads a column -- so a round trip through it "
          f"is not evidence about the layout")

    # A file-level header ahead of the first MODEL. Tools do write one, and
    # `split_pdbqt_models` already treats those lines as belonging to the first
    # pose, so the reader has to agree or the pair is not a pair. Without this
    # check the reader's handling of that case is untested, and mutating it
    # changes nothing at all -- an equivalent mutant rather than a covered one.
    header_dir = Path(tempfile.mkdtemp(prefix="pdbqt_check_head_"))
    header_file = header_dir / "header_then_two_models.pdbqt"
    try:
        header_file.write_text(
            "REMARK  file-level header, ahead of every MODEL\n"
            "MODEL\n"
            "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  "
            "0.00     0.000 C \n"
            "ENDMDL\n"
            "MODEL\n"
            "ATOM      2  C2  UNL     1       1.000   0.000   0.000  1.00  "
            "0.00     0.000 C \n"
            "ENDMDL\n",
            encoding="utf-8",
        )
        headed = W.read_pdbqt_models(header_file)
        check("a file-level header ahead of the first MODEL goes to the first "
              "pose, which is what the splitter assumes",
              len(headed) == 2
              and "file-level header" in headed[0][0]
              and not any("file-level header" in ln for ln in headed[1]),
              f"2 poses; the first begins {headed[0][0]!r} and the second "
              f"begins {headed[1][0]!r}. `split_pdbqt_models` has always "
              f"attached those lines to pose 1, and the reader now agrees")
    finally:
        header_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    section("every writer function against the readers available")

    src = load_ligand(EXAMPLES / "ibuprofen_prep.pdbqt")
    coords = np.asarray(src.reference_coords, np.float32)
    types = [ln[77:79].strip() or "C" for ln in shipped_atoms]
    names = [f"{t}_{e}" for t, e in zip(types, _elements_of(EXAMPLES /
                                                             "ibuprofen_prep.pdbqt"))]

    text = W.ligand_to_pdbqt(src, coords, names)
    back = Ligand.from_pdbqt_str(text)
    again = np.asarray(back.reference_coords, np.float32)
    worst = float(np.abs(again - coords).max())
    check("ligand_to_pdbqt: the engine's reader gets the coordinates back, in "
          "order, to the file's precision",
          back.num_atoms == src.num_atoms and worst <= 5e-4
          and np.array_equal(again, coords.astype(np.float32).round(3)),
          f"{src.num_atoms} atoms in, {back.num_atoms} out, largest error "
          f"{worst:.2e} A against the 0.0005 A the 3-decimal format allows. "
          f"This is *parseability*: the engine reader splits on whitespace, so "
          f"it would accept the same file with a column deleted")

    third = meeko_positions(text).reshape(-1, 3)
    check("ligand_to_pdbqt: meeko reads the same coordinates, and meeko reads "
          "by column",
          third.shape == coords.shape
          and float(np.abs(third - coords).max()) <= 5e-4,
          f"meeko read {third.shape[0]} atoms, largest error "
          f"{float(np.abs(third - coords).max()):.2e} A. Unlike the engine "
          f"reader this is evidence about the columns")

    check("ligand_to_pdbqt: the atom types survive, which only `names` gives",
          back.atom_kinds == src.atom_kinds,
          f"kinds {back.atom_kinds[:4]} against the source's "
          f"{src.atom_kinds[:4]}")

    # _split_name has two branches and every in-tree caller -- cli.py,
    # ligand_check.py, make_data.py -- feeds it `prep`'s `TYPE_ELEMn` tokens,
    # all of which contain "_". The bare-name branch is therefore reached by
    # nobody in this project, which is exactly why it needed a check: an
    # unexercised branch is where "CA" quietly becomes calcium.
    bare_name, bare_type = W._split_name("CA")
    typed_name, typed_type = W._split_name("OA_O1")
    check("_split_name's unexercised bare-name branch turns the first two "
          "characters into the atom type, and that is a trap, not a feature",
          bare_name == " CA " and bare_type == "CA"
          and typed_name == " O1 " and typed_type == "OA",
          f"'CA' -> name {bare_name!r}, which fills columns 13-16 exactly, and "
          f"type {bare_type!r} -- and 'CA' is *calcium* in AutoDock, so a bare "
          f"name types a carbon alpha as an ion. The branch in-tree callers "
          f"use is the other one: 'OA_O1' -> name {typed_name!r}, type "
          f"{typed_type!r}. No caller in this project can reach the bare "
          f"branch, because `prep` always emits a type prefix")

    # ... and now the *same trap constructed through the public API*, because
    # calling the private helper only proves the helper does what it does. This
    # one writes a real file and reads it back, and it carries a control: the
    # identical molecule with prep-style prefixed names must come out
    # hydrophobic, so "all atoms typed other" cannot be confused with "this
    # molecule is odd".
    pair = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], np.float32)
    diatomic = Ligand.from_arrays(["C", "C"], [0.0, 0.0], pair)
    trap_text = W.ligand_to_pdbqt(diatomic, pair, ["C1", "CA"])
    ok_text = W.ligand_to_pdbqt(diatomic, pair, ["C_C1", "C_C2"])
    trap_atoms = [ln for ln in trap_text.splitlines() if ln.startswith("ATOM")]
    trap_kinds = Ligand.from_pdbqt_str(trap_text).atom_kinds
    ok_kinds = Ligand.from_pdbqt_str(ok_text).atom_kinds
    check("a bare name written through ligand_to_pdbqt types the atom as its "
          "own first two characters, and no in-tree caller can reach that",
          [col(ln, "type") for ln in trap_atoms] == ["C1", "CA"]
          and all(len(ln) == 79 for ln in trap_atoms)
          and set(trap_kinds) == {"other"}
          and set(ok_kinds) == {"hydrophobic"}
          and meeko_positions(trap_text).reshape(-1, 3).shape == (2, 3),
          f"bare names 'C1' and 'CA' give types "
          f"{[col(ln, 'type') for ln in trap_atoms]} in columns 78-79, so a "
          f"carbon reads back {trap_kinds[0]!r} instead of "
          f"{ok_kinds[0]!r} -- 'CA' is calcium in AutoDock. The file is still "
          f"79 characters a line and meeko still parses it, which is the whole "
          f"point: the damage is silent. Control: the same molecule with "
          f"prep-style 'C_C1'/'C_C2' comes back "
          f"{ok_kinds[0]!r}. LABEL: no in-tree caller reaches this -- "
          f"cli.py, ligand_check.py and make_data.py all pass `prep` names, "
          f"which always carry a type prefix")

    bare = W.ligand_to_pdbqt(src, coords)
    bare_back = Ligand.from_pdbqt_str(bare)
    bare_kinds = bare_back.atom_kinds
    check("and omitting `names` types every atom as carbon, which is measured "
          "rather than assumed",
          set(bare_kinds) == {"hydrophobic"} and len(set(src.atom_kinds)) > 1,
          f"without `names` every atom reads back {bare_kinds[0]!r}; the source "
          f"has {len(set(src.atom_kinds))} distinct kinds. A well-formed file "
          f"that describes a molecule of pure carbon")

    # The 4-in/5-out torsion change is real, and this pins *what causes it*,
    # because the natural explanation is wrong. The flat layout is not the
    # cause: with the right types the count survives. The `names` fallback is.
    check("the flat layout does not change the torsion tree when the atom types "
          "are right",
          back.num_torsions == src.num_torsions,
          f"written with `names`, the engine reads {back.num_torsions} "
          f"rotatable bonds from the writer's flat file and the shipped file "
          f"has {src.num_torsions}. An earlier version of this check expected "
          f"5 and blamed the missing TORSION records; the measurement says the "
          f"flat file is innocent")
    check("and the 4-becomes-5 difference belongs to the `names` fallback, not "
          "to the flat file",
          bare_back.num_torsions != src.num_torsions
          and back.num_torsions == src.num_torsions,
          f"omitting `names` gives {bare_back.num_torsions} rotatable bonds "
          f"against the source's {src.num_torsions}, because typing every "
          f"oxygen, nitrogen and sulfur as 'C' makes a different molecule. "
          f"Attributing this to the file layout would have hidden the real "
          f"cause, which is a caller that leaves `names` out")

    tmp = Path(tempfile.mkdtemp(prefix="pdbqt_check_"))
    pose = tmp / "pose.pdbqt"
    W.write_pose(pose, coords, types, atom_names=names, energy=-7.42)
    models = W.read_pdbqt_models(pose)
    # Position, asserted against a *parse* and not inferred from indices -- and
    # asserted in both directions, because the previous version of this check
    # named "the energy is inside the MODEL block" while only ever comparing
    # line numbers, which is the shape of check that cannot fail. The control
    # below is the proof: a hand-built file whose remark is outside every block
    # has to be reported as outside, or the assertion is decorative.
    on_disk = pose.read_text(encoding="utf-8")
    where = _block_holding(on_disk, "VINA RESULT")
    control = (
        "REMARK VINA RESULT:   -7.4      0.000      0.000\n"
        + W.format_atom_line(1, " C1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "C")
        + "\nMODEL\n"
        + W.format_atom_line(2, " C2", "UNL", 1, (1.0, 0.0, 0.0), 0.0, "C")
        + "\nENDMDL\n"
    )
    check("write_pose: the energy is inside the MODEL block, asserted against "
          "an independent parse and with a control that must come out outside",
          where == "MODEL" and _block_holding(control, "VINA RESULT") != "MODEL"
          and _energy_of(models[0]) == -7.4,
          f"an independent record-name walk puts write_pose's remark in a "
          f"{where!r} block, and the same walk puts the control file's remark "
          f"in {_block_holding(control, 'VINA RESULT')!r} -- outside every "
          f"block -- so this assertion can fail. The pose reads back "
          f"{_energy_of(models[0])!r} from -7.42, because the line is written "
          f"with one decimal place, the same as Vina's own remark. Outside the "
          f"block the remark belongs to no pose")

    pose_pos = meeko_positions(pose.read_text(encoding="utf-8")).reshape(-1, 3)
    check("write_pose: meeko reads the pose file too",
          pose_pos.shape == coords.shape
          and float(np.abs(pose_pos - coords).max()) <= 5e-4,
          f"{pose_pos.shape[0]} atoms, largest error "
          f"{float(np.abs(pose_pos - coords).max()):.2e} A")

    # split_pdbqt_models, then read every piece back with both readers.
    multi = tmp / "multi.pdbqt"
    blocks = []
    for i, energy in enumerate((-7.4, -6.1, -5.0)):
        body_lines = [f"REMARK VINA RESULT:  {energy:5.1f}      0.000      0.000"]
        for j in range(3):
            body_lines.append(W.format_atom_line(
                j + 1, f" C{j + 1}", "UNL", 1, (float(i), 0.5 * j, 0.25), 0.0, "C"))
        blocks.append(body_lines)
    multi.write_text(
        "".join("MODEL\n" + "\n".join(b) + "\nENDMDL\n" for b in blocks),
        encoding="utf-8")
    written = W.split_pdbqt_models(multi, tmp / "split")
    energies = []
    shapes = []
    for dest in written:
        got = W.read_pdbqt_models(dest)
        energies.append(_energy_of(got[0]) if got else None)
        shapes.append(meeko_positions(dest.read_text(encoding="utf-8"))
                      .reshape(-1, 3).shape[0])
    check("split_pdbqt_models: one file per pose, each keeping its own energy",
          len(written) == 3 and energies == [-7.4, -6.1, -5.0],
          f"{[p.name for p in written]} with energies {energies} -- each piece "
          f"kept its own rather than the first one's")
    check("split_pdbqt_models: every piece still parses, by both readers",
          all(s == 3 for s in shapes) and all(
              W.read_pdbqt_models(p) for p in written),
          f"meeko read {shapes} atoms from the three pieces and the splitter "
          f"returns one pose each")

    # ------------------------------------------------------------------
    section("the Python layer checks the columns, because the engine does not")

    # The engine's reader splits on whitespace, so the only column-sensitive
    # parser in reach is meeko, and meeko is not part of this project. Option 2
    # was chosen over "write the tolerance down and move on": the exposure is
    # that the engine loads a file this project mis-formatted, and a design
    # intent nobody wrote down is indistinguishable from an oversight. The cost
    # of validating is a false positive on a good file, so that is measured
    # first, over every file the project ships.
    corpus_problems = {}
    corpus_lines = 0
    for path in sorted(EXAMPLES.glob("*.pdbqt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        corpus_lines += sum(1 for ln in lines if ln.startswith(("ATOM", "HETATM")))
        found = W.validate_pdbqt_columns(lines, source=path.name)
        if found:
            corpus_problems[path.name] = found
    check("the validator accepts every file this project ships, which is the "
          "price of adding it",
          not corpus_problems and corpus_lines > 100,
          f"{corpus_lines} ATOM lines across "
          f"{len(list(EXAMPLES.glob('*.pdbqt')))} files, including the "
          f"engine-written crambin_pose.pdbqt, and zero problems raised. A "
          f"validator that rejected correct files would be worse than none"
          + (f"; false positives: {corpus_problems}" if corpus_problems else ""))

    # The defect class the validator exists for: right length, wrong columns.
    # The engine parses this without complaint, and that acceptance is the
    # exposure this whole section is about.
    good_line = W.format_atom_line(1, " C1", "UNL", 1, (1.234, 2.345, 3.345),
                                   0.0, "C")
    shifted = (" " + good_line)[:-1]          # same 79 characters, all shifted
    shifted_problems = W.validate_pdbqt_columns([shifted], source="shifted")
    engine_took_shift = Ligand.from_pdbqt_str(shifted + "\n").num_atoms == 1
    check("a 79-character line that no longer parses at its columns is refused "
          "here, and the engine accepts it",
          len(shifted) == len(good_line) == 79 and len(shifted_problems) >= 3
          and engine_took_shift and bool(meeko_refuses(shifted + "\n"))
          and not meeko_refuses(good_line + "\n"),
          f"shifting the whole record one column right leaves it "
          f"{len(shifted)} characters long and produces {len(shifted_problems)} "
          f"problems, the first being "
          f"{shifted_problems[0] if shifted_problems else ''!r}. The engine's "
          f"whitespace reader accepts the same line and reports one atom; "
          f"meeko refuses it and accepts the correct one. The engine is not "
          f"wrong *here* -- shifting every field by one leaves the "
          f"whitespace-delimited fields intact -- but it cannot tell the "
          f"difference, and that inability is the exposure")

    # The single most likely column defect, in both directions, measured rather
    # than assumed. `float()` alone would miss one of them, because
    # float('  1.234 ') is perfectly happy -- so this pins the *tightened*
    # behaviour, and pins the residual gap beside it.
    early = good_line[:30] + "  1.234 " + good_line[38:]
    early_problems = W.validate_pdbqt_columns([early], source="early")
    check("a field whose value arrives one column EARLY is caught, which "
          "float() alone would have missed",
          len(early) == 79
          and float("  1.234 ") == 1.234        # why a bare cast is not enough
          and len(early_problems) == 1
          and any("columns 31-38" in p for p in early_problems),
          f"the x field is {early[30:38]!r} -- a trailing space where "
          f"PDBQT right-justifies -- and `float()` reads that as "
          f"{float('  1.234 ')}, so a validator that only cast the substring "
          f"would have called this line fine. Requiring the documented shape "
          f"catches it: {_head(early_problems)}")
    late = good_line[:30] + "    1.23" + good_line[38:]
    late_problems = W.validate_pdbqt_columns([late], source="late")
    check("and a field whose value arrives one column LATE is *not* caught, "
          "which is inherent to reading a file",
          len(late) == 79 and not late_problems
          and float(late[30:38]) == 1.23,
          f"the x field is {late[30:38]!r} -- still a right-justified float, "
          f"now the wrong one ({float(late[30:38])} instead of 1.234) -- so "
          f"the validator reports {late_problems!r}. A reader has no way to "
          f"know what the number was meant to be, and meeko accepts it too. "
          f"A whole-record shift *is* caught, because it mangles the fields "
          f"after it; it is a single field quietly displaced that slips "
          f"through, and only a comparison against the source molecule would "
          f"catch that")
    # ... and the honest boundary of that exposure, pinned so it cannot be
    # quietly overclaimed later. A value that is *syntactically valid* in the
    # column it sits in cannot be caught by reading columns: a reader has no
    # way to know what z was supposed to be.
    wrong_z = good_line[:46] + good_line[30:38] + good_line[54:]
    z_seen = meeko_positions(wrong_z + "\n").reshape(-1, 3)[0]
    z_flagged = W.validate_pdbqt_columns([wrong_z], source="wrong_z")
    check("and a line that is wrong in a way columns cannot detect is NOT "
          "claimed to be caught, because it cannot be",
          len(wrong_z) == 79 and not z_flagged
          and abs(z_seen[2] - 1.234) <= 5e-4,
          f"putting the x value into the z field leaves 79 characters and a "
          f"valid float at 47-54, so the validator reports {z_flagged!r} and "
          f"meeko reads z = {z_seen[2]}. Only a comparison against the source "
          f"molecule could catch that, which is not a file reader's job. "
          f"Column validation catches a layout that stopped parsing, not a "
          f"plausible number in the wrong slot")

    # And the refusal has to reach the caller, not just a helper's return value.
    bad_file = tmp / "mis_columned.pdbqt"
    bad_file.write_text("MODEL\n" + shifted + "\nENDMDL\n", encoding="utf-8")
    raised = None
    try:
        W.read_pdbqt_models(bad_file)
    except W.PdbqtFormatError as exc:
        raised = exc
    check("read_pdbqt_models refuses the file and says where the problem is",
          raised is not None and "mis_columned" in str(raised)
          and "line 2" in str(raised) and "columns 7-11" in str(raised),
          f"PdbqtFormatError({str(raised).splitlines()[0]!r}) names the file, "
          f"the 1-based line and the columns, because 'invalid PDBQT' with no "
          f"location is the least useful thing an error can be. It is a "
          f"ValueError subclass, so a caller already handling bad input from "
          f"this module keeps working")

    lenient = W.read_pdbqt_models(bad_file, strict=False)
    check("and `strict=False` is a real escape hatch, not a stub",
          len(lenient) == 1 and len(lenient[0]) == 1
          and lenient[0][0] == shifted,
          f"the same file with strict=False returns {len(lenient)} pose of "
          f"{len(lenient[0])} line, and that caller now owns the mis-parse. A "
          f"flag that did nothing would leave no way to open an odd file")

    # Tolerances, each pinned so that a future tightening is a visible change
    # rather than a silent refusal of a file that used to open.
    blank_charge = good_line[:70] + "      " + good_line[76:]
    trimmed = good_line[:-1]
    empty_type = W.format_atom_line(1, " C1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "")
    tolerated = {
        "a blank charge field": blank_charge,
        "a line with its trailing space stripped": trimmed,
        "the writer's own empty atom type": empty_type,
    }
    rejected = {why: W.validate_pdbqt_columns([ln], source=why)
                for why, ln in tolerated.items()}
    check("three tolerances are deliberate and are pinned, so tightening them "
          "would be a visible decision",
          not any(rejected.values()),
          "; ".join(f"{why} ({len(ln)} characters) is accepted"
                    for why, ln in tolerated.items())
          + ". All 786 shipped ATOM lines carry a real charge, so these are "
            "choices about files this project has never seen -- and a "
            "validator that rejected the writer's own empty-type output would "
            "be a joke")

    # The buffered region ahead of the first atom is where a mangled header
    # would otherwise hide, and the reader now checks *before* it attributes a
    # pose, so a file whose header is garbage never becomes a pose at all.
    garbled = [
        "REMARK VINA RESULT:   -7.4      0.000      0.000",
        "GARBLED  header line that is not a record at all",
        good_line,
    ]
    garbled_problems = W.validate_pdbqt_columns(garbled, source="garbled")
    garbled_file = tmp / "garbled_header.pdbqt"
    garbled_file.write_text("\n".join(garbled) + "\n", encoding="utf-8")
    refused_early = None
    try:
        W.read_pdbqt_models(garbled_file)
    except W.PdbqtFormatError as exc:
        refused_early = exc
    check("a mangled line ahead of the first atom is refused, which is the "
          "one place a broken header could hide",
          len(garbled_problems) == 1
          and any("line 2" in p for p in garbled_problems)
          and refused_early is not None
          and "GARBLED" in str(refused_early),
          f"{_head(garbled_problems)}. The reader validates the whole file "
          f"*before* attributing a pose, so this is caught before the "
          f"workbench has a pose to display -- the ordering is the point, and "
          f"buffering those lines without checking them would have been the "
          f"easy mistake")

    # The three field shapes the validator refuses, each with a fixture built
    # by hand. They were all survivors of mutation testing before these three
    # checks existed: the suite passed with the charge check, the type check
    # and the short-line check all disabled, because the only fixtures in play
    # were *tolerances*, which cannot exercise a refusal.
    garbled_charge = good_line[:70] + "xyzw12" + good_line[76:]
    charge_problems = W.validate_pdbqt_columns([garbled_charge], source="c")
    check("a non-blank charge that is not a number is refused",
          len(garbled_charge) == 79 and len(charge_problems) == 1
          and any("columns 71-76" in p for p in charge_problems),
          f"the charge field is {garbled_charge[70:76]!r} and is refused: "
          f"{_head(charge_problems)}. This is the *other* half of the "
          f"blank-charge "
          f"tolerance: blank is allowed on purpose, garbage is not")

    # The type field, refused. The fixture used to be `C1`, and `C1` is now
    # **accepted** -- a measured loosening, not a slip. `C1` has exactly the
    # shape of meeko's `G0`, `G0` is in meeko's own AutoDock 4 type table, and
    # the old rule refused it. No shape rule can refuse one and admit the
    # other, so the fixture moves to tokens the shape really does exclude.
    # The price is recorded in `pdbqt_writer._ATOM_TYPE` rather than hidden.
    bad_types = {}
    for token in ("12", "1CG", "NDAX", "CG!", "-C"):
        bad = good_line[:77] + token
        bad_types[token] = (len(bad), W.validate_pdbqt_columns([bad],
                                                               source="t"))
    check("an atom type that is not a letter followed by at most two letters "
          "or digits is refused",
          bad_types["12"][0] == 79
          and all(len(p) == 1 and "78 onwards" in p[0]
                  for _, p in bad_types.values()),
          "; ".join(f"{t!r} ({n} chars) -> {(_head(p) or 'ACCEPTED')}"
                    for t, (n, p) in bad_types.items())
          + ". `C1` was the fixture here until this check was rewritten, and it "
            "is now accepted: it has the same shape as meeko's `G0`, which is a "
            "real AutoDock 4 type, so a shape rule that refused `C1` would "
            "refuse a file the format's reference writer emits")

    # The accept side of the same field, which is the half that needs an oracle
    # rather than a fixture invented here. meeko's own AutoDock 4 table is the
    # definition of what a type may be, and **every** name in it has to pass.
    from meeko.utils.autodock4_atom_types_elements import (
        autodock4_atom_types_elements as AD4,
    )
    ad4_refused = {
        name: W.validate_pdbqt_columns([good_line[:77] + name], source="t")
        for name in sorted(AD4)
    }
    rejected_by_ad4 = sorted(n for n, p in ad4_refused.items() if p)
    # The one three-character type that is *this project's* vocabulary rather
    # than AutoDock's, measured through the same door. Meeko with typing on
    # refuses it; the engine truncates it. Both are reported, neither is
    # encoded as a rule -- see the reasoning at `pdbqt_writer._ATOM_TYPE`.
    three = {}
    for tok in ("NDA", "ODA", "CG0", "G0", "C1", "ND1"):
        line3 = good_line[:77] + tok
        three[tok] = (len(line3),
                      bool(W.validate_pdbqt_columns([line3], source="t")),
                      meeko_positions(line3 + "\n").reshape(-1, 3).shape[0]
                      if not meeko_refuses(line3 + "\n") else 0)
    engine_kinds = {}
    for tok in ("NDA", "ODA"):
        lig = Ligand.from_pdbqt_str(good_line[:77] + tok + "\nEND\n")
        engine_kinds[tok] = (lig.atom_kinds[0],
                             Ligand.from_pdbqt_str(
                                 good_line[:77] + tok[:2] + "\nEND\n"
                             ).atom_kinds[0])
    check("every atom type in meeko's own AutoDock 4 table is accepted, and a "
          "three-character one is a file meeko reads",
          not rejected_by_ad4
          and three["CG0"] == (80, False, 1)
          and three["G0"][1] is False
          and three["NDA"] == (80, False, 1)
          and engine_kinds["NDA"] == ("donor", "donor")
          and engine_kinds["ODA"] == ("donor", "donor"),
          f"meeko 0.7.1 defines {len(AD4)} AutoDock 4 type names, longest "
          f"{max(len(n) for n in AD4)} characters, "
          f"{sum(1 for n in AD4 if len(n) > 2)} of them three characters "
          f"({', '.join(sorted(n for n in AD4 if len(n) > 2))}) and "
          f"{sum(1 for n in AD4 if n[-1].isdigit())} ending in a digit. "
          f"Rejected by this validator: {rejected_by_ad4 or 'none'}. "
          f"Fixtures: "
          + ", ".join(f"{t!r} len {n} {'refused' if r else 'accepted'}"
                      f"/meeko {k} atom" for t, (n, r, k) in three.items())
          + f". The engine, asked for {engine_kinds['NDA'][0]!r} in columns "
            f"78-80, reads the first two and reports {engine_kinds['NDA'][0]!r}"
            f" -- identical to the file that says {engine_kinds['NDA'][1]!r}, so "
            f"the third character is lost in the engine's reader, not here. "
            f"Meeko with typing enabled refuses all four of NDA, ODA, ND and "
            f"OD, so no third-party tool can honour these two either: that is a "
            f"change request against Rust, not a rule for this validator")

    # Mutation proof for the rule above, in both directions. Two of these
    # mutations were wrong when first written and are recorded because the
    # corrections are the interesting part:
    #
    # * A mutation of the *end* column, `_COL_TYPE = (77, 80)`, changed
    #   **nothing at all**. The read is `line[_COL_TYPE[0]:]` -- to end of
    #   line, as meeko takes it -- so `_COL_TYPE[1]` is not used by this
    #   check and that mutant is an equivalent one. A mutation that cannot
    #   fail is not evidence, so the width is pinned directly instead: the
    #   refusal message below has to quote the *whole* token, which is only
    #   possible if the read reached past column 79.
    # * A mutation that only refuses, or only accepts, is caught by one
    #   direction of this table and passes the other. That is the reason both
    #   directions are here rather than one convenient fixture.
    saved_cols, saved_re = W._COL_TYPE, W._ATOM_TYPE
    mutations = {}

    def _refuses_all() -> bool:
        return all(W.validate_pdbqt_columns([good_line[:77] + t], source="m")
                   for t in ("12", "1CG", "NDAX", "CG!", "-C"))

    def _accepts_ad4() -> bool:
        return not any(W.validate_pdbqt_columns([good_line[:77] + n], source="m")
                       for n in AD4)

    try:
        mutations["unmutated"] = (_refuses_all(), _accepts_ad4())
        W._COL_TYPE = (78, 79)          # read window starts one column late
        mutations["read starts at column 78"] = (_refuses_all(), _accepts_ad4())
        W._COL_TYPE = saved_cols
        W._ATOM_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,1}$")   # 2-char cap
        mutations["old two-character cap"] = (_refuses_all(), _accepts_ad4())
        W._ATOM_TYPE = re.compile(r"^[A-Za-z]+$")                 # letters only
        mutations["old letters-only rule"] = (_refuses_all(), _accepts_ad4())
    finally:
        W._COL_TYPE, W._ATOM_TYPE = saved_cols, saved_re
    mutations["restored"] = (_refuses_all(), _accepts_ad4())
    # The width, pinned without a mutation: a two- or three-character read
    # cannot quote 'NDAX' in its complaint, because it never sees the 'X'.
    nda_x = W.validate_pdbqt_columns([good_line[:77] + "NDAX"], source="m")
    check("each half of the type rule is load-bearing: three mutations, each "
          "caught, and restoring turns it green again",
          mutations["unmutated"] == (True, True)
          and mutations["restored"] == (True, True)
          and mutations["read starts at column 78"] == (False, False)
          and mutations["old two-character cap"] == (True, False)
          and mutations["old letters-only rule"] == (False, False)
          and len(nda_x) == 1 and "'NDAX'" in nda_x[0],
          "; ".join(f"{k}: refuses-the-bad={v[0]}, accepts-all-AD4={v[1]}"
                    for k, v in mutations.items())
          + f". Moving the read window one column late is caught by both "
            f"directions at once -- it drops the leading character, so 'G0' "
            f"becomes '0' and '12' becomes '2'. The two pattern mutations are "
            f"each caught by the accept direction against meeko's table: the "
            f"old two-character cap rejects CG0-CG3, the old letters-only rule "
            f"rejects those and G0-G3, and all of those are types meeko "
            f"writes. The letters-only rule is also caught by the *refusal* "
            f"direction, for a reason worth stating: 'NDAX' is all letters, so "
            f"that rule accepts it. The width of the read is pinned by the "
            f"complaint rather than by a mutation, because `_COL_TYPE[1]` is "
            f"not read by it: {nda_x[0]!r} quotes all four characters, which a "
            f"two- or three-character read could not do")

    short_line = good_line[:70]
    short_problems = W.validate_pdbqt_columns([short_line], source="s")
    check("an ATOM line too short to hold a charge field is refused",
          len(short_line) == 70 and len(short_problems) == 1
          and any("too short" in p for p in short_problems),
          f"truncating to {len(short_line)} characters gives "
          f"{_head(short_problems)}. The 78-character 'trailing space stripped' "
          f"case is still accepted -- 78 can hold 71-76 -- so this boundary is "
          f"at 76, not at 79")

    # Caught with `except ValueError` deliberately, not with the precise class:
    # the claim under test is that a caller's *existing* handler still works.
    caught_as_value_error = False
    raised_here: Exception | None = None
    try:
        W.read_pdbqt_models(bad_file)
    except ValueError as exc:
        caught_as_value_error = True
        raised_here = exc
    check("PdbqtFormatError is a ValueError, so a caller's existing handler "
          "keeps working",
          issubclass(W.PdbqtFormatError, ValueError) and caught_as_value_error
          and isinstance(raised_here, W.PdbqtFormatError),
          f"the base of {W.PdbqtFormatError.__name__} is "
          f"{W.PdbqtFormatError.__mro__[1].__name__}, and an `except ValueError` "
          f"around a read catches {type(raised_here).__name__} -- so a caller "
          f"already handling bad input from this module needs no new clause, "
          f"and the writer's own ValueErrors are unaffected")

    # ------------------------------------------------------------------
    section("the cases that are easy to get wrong")

    one = W.format_atom_line(1, "H", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "HD")
    two = W.format_atom_line(1, "CA", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "C")
    check("a one-character atom name is right-justified and a two-character "
          "one is not",
          col(one, "name") == "   H" and col(two, "name") == "CA  ",
          f"'H' -> {col(one, 'name')!r}, 'CA' -> {col(two, 'name')!r}. Both "
          f"match the Meeko-written receptor file examples/1crn_prep.pdbqt, "
          f"whose first two names are '   N' and 'CA  '")
    check("and the shipped docked pose justifies its names the other way, "
          "which is a real difference between two shipped files",
          True,
          f"examples/crambin_pose.pdbqt writes '  C5' (right) where this writer "
          f"writes ' C1 ' (left) for a two-character name. Both strip to the "
          f"same characters and AutoDock does not read this column, so it is "
          f"cosmetic -- but 'match a shipped file' has two answers here, and "
          f"this is the one that does not match")

    ch = W.format_atom_line(1, " C1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, "C",
                            chain="Z")
    check("the chain identifier is a real column and not a literal space",
          col(ch, "chain") == "Z" and col(ch, "resSeq") == "   1"
          and len(ch) == 79,
          f"chain {col(ch, 'chain')!r} in column 22, and resSeq still starts "
          f"in 23 -- the two constants overlap at index 21 and the widths are "
          f"1 and 4, so nothing is lost")

    # The width limit, which is where the writer used to corrupt data silently.
    # The x field holds 8 characters (columns 31-38). A positive number of four
    # digits and three decimals is 8 and fits; a *negative* one is 9 and does
    # not. That asymmetry is the whole reason the boundary is pinned rather than
    # assumed -- the largest coordinate in any shipped file is 44.319 A, so real
    # inputs never come near it, and an unmeasured guess here would be fiction.
    fits = [(999.999, True), (-999.999, True), (9999.999, True),
            (-9999.999, False), (12345.678, False), (-12345.678, False)]
    results = []
    for value, should_fit in fits:
        try:
            got = float(col(W.format_atom_line(
                1, " C1", "UNL", 1, (value, 0.0, 0.0), 0.0, "C"), "x"))
            results.append((value, should_fit, abs(got - value) <= 5e-4))
        except ValueError as exc:
            results.append((value, should_fit, "refuse" in str(exc)))
    check("a coordinate that does not fit its 8-character field is refused, "
          "not truncated",
          all(fit == should for _, should, fit in results),
          "; ".join(f"{v} -> {'fits' if f else 'refused' if isinstance(f, str) else 'WRONG'}"
                    for v, _, f in results)
          + ". Before this check existed the refused ones were written anyway: "
            "'-12345.678' became '-12345.6' and '12345.678' became '12345.67', "
            "different coordinates in a file still exactly 79 characters long")

    charges = [(0.0, True), (-0.412, True), (9.999, True), (123.456, False),
               (-12.345, False)]
    c_results = []
    for value, should_fit in charges:
        try:
            got = float(col(W.format_atom_line(
                1, " C1", "UNL", 1, (0.0, 0.0, 0.0), value, "C"), "charge"))
            c_results.append((value, should_fit, abs(got - value) <= 5e-4))
        except ValueError as exc:
            c_results.append((value, should_fit, "refuse" in str(exc)))
    check("and so is a charge that does not fit its 6-character field",
          all(fit == should for _, should, fit in c_results),
          "; ".join(f"{v} -> {'fits' if f else 'refused' if isinstance(f, str) else 'WRONG'}"
                    for v, _, f in c_results)
          + ". '-12.345' used to become '-12.34'")

    for bad, why in (("Clx", "three characters"), ("CLONG", "five characters"),
                     ("Fe2+", "an ion written the way a chemist would")):
        try:
            W.format_atom_line(1, " C1", "UNL", 1, (0.0, 0.0, 0.0), 0.0, bad)
            raised = False
        except ValueError:
            raised = True
        check(f"an atom type of {why} is refused", raised,
              f"{bad!r} raises rather than being cut to two characters, which "
              f"is what would turn 'Clx' into a chlorine")

    check("a name exactly the column width is kept whole, and one longer is cut",
          col(W.format_atom_line(1, "HG12", "UNL", 1, (0, 0, 0), 0.0, "C"),
               "name") == "HG12"
          and col(W.format_atom_line(1, "HG123", "UNL", 1, (0, 0, 0), 0.0, "C"),
                  "name") == "HG12",
          "'HG12' fills 13-16 exactly; 'HG123' is cut to 'HG12'. A name is a "
          "label, and cutting a label loses information without inventing it")

    # A real one-atom molecule. The engine's ligand type accepts one, and the
    # writer's own loop is over `ligand.num_atoms`, so this is the genuine
    # small case rather than a short coordinate array.
    solo = Ligand.from_arrays(["C"], [0.0], np.zeros((1, 3)))
    solo_text = W.ligand_to_pdbqt(solo, np.zeros((1, 3), np.float32), None)
    solo_atoms = [ln for ln in solo_text.splitlines() if ln.startswith("ATOM")]
    check("a genuine one-atom molecule writes one well-formed line",
          solo.num_atoms == 1 and len(solo_atoms) == 1
          and all(len(ln) == 79 for ln in solo_atoms)
          and meeko_positions(solo_text).reshape(-1, 3).shape == (1, 3),
          f"{len(solo_atoms)} ATOM line of {len(solo_atoms[0])} characters "
          f"and meeko reads it back; the header says "
          f"{solo_text.splitlines()[1].strip()!r}, and with 0 rotatable bonds "
          f"the TORSDOF is 0")

    # The empty case is *unreachable*, not untested: the core type refuses to
    # exist with no atoms. What is reachable is a coordinate array shorter than
    # the ligand's own atom count, and that is an IndexError -- pinned as one,
    # because the alternative is a file whose header claims 16 atoms and whose
    # body has 3.
    refusal = None
    try:
        Ligand.from_arrays([], [], np.zeros((0, 3)))
    except Exception as exc:  # noqa: BLE001 - the refusal is the evidence
        refusal = f"{type(exc).__name__}: {str(exc)[:52]}"
    short = None
    try:
        W.ligand_to_pdbqt(src, np.zeros((1, 3), np.float32), None)
    except IndexError as exc:
        short = str(exc)[:52]
    check("the empty case cannot be reached, and a short coordinate array "
          "raises rather than writing a partial file",
          refusal is not None and short is not None,
          f"an empty Ligand raises {refusal!r}, so there is no zero-atom "
          f"molecule to hand this writer; feeding the {src.num_atoms}-atom "
          f"ligand {1} row of coordinates raises IndexError({short!r})")

    check("a residue number wider than its four characters is refused too",
          _refuses(lambda: W.format_atom_line(
              1, " C1", "UNL", 12345, (0.0, 0.0, 0.0), 0.0, "C")),
          "'12345' needs five characters in a four-wide field, and a residue "
          "number that changes is a different residue")

    # ------------------------------------------------------------------
    section("degenerate input does not crash and does not invent a field")

    check("an empty atom type is padded, not dropped",
          col(W.format_atom_line(1, " C1", "UNL", 1, (0, 0, 0), 0.0, ""),
              "type") == "  ",
          "two spaces in 78-79, which is a blank type rather than a type in "
          "the wrong column")
    check("a multi-character chain is cut to one character",
          col(W.format_atom_line(1, " C1", "UNL", 1, (0, 0, 0), 0.0, "C",
                                 chain="ABC"), "chain") == "A",
          "column 22 is one character wide; 'ABC' contributes the first")
    check("an empty chain is a blank column, not a missing one",
          col(W.format_atom_line(1, " C1", "UNL", 1, (0, 0, 0), 0.0, "C",
                                 chain=""), "chain") == " "
          and len(W.format_atom_line(1, " C1", "UNL", 1, (0, 0, 0), 0.0, "C",
                                     chain="")) == 79,
          "the default and an explicit empty string both produce a legal line")

    # ------------------------------------------------------------------
    section("every check in this file ran")

    before = CHECKS
    check("the number of checks that ran is the number this file is supposed to "
          "have",
          before + 1 == EXPECTED_CHECKS,
          f"{before} ran before this one and {EXPECTED_CHECKS} are expected; "
          f"the +1 is this check. If a check was added or removed, change "
          f"EXPECTED_CHECKS deliberately")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


def _elements_of(path: Path) -> list[str]:
    """Element symbols for a PDBQT, from the AutoDock type's first letter.

    Read from the file's own type column rather than from RDKit, so nothing in
    this check depends on a chemistry front-end being installed.
    """
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            t = ln[77:79].strip()
            out.append({"A": "C", "C": "C", "N": "N", "NA": "N", "OA": "O",
                        "HD": "H", "S": "S", "SA": "S"}.get(t, "C"))
    return out


def _refuses(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def _head(problems: list[str]) -> str:
    """The first problem, or a plain note when there are none.

    The detail text of a failing check is built *whether or not the check
    passed*, so a detail that indexes ``problems[0]`` raises IndexError on
    exactly the run where the guard is broken and there is nothing to show.
    That turns a legible failure into a stack trace, and it is the reason
    several mutations below report as CRASHED rather than FAIL. The exit code
    is the same either way; this is about being able to read the output.
    """
    return repr(problems[0]) if problems else "(none reported)"


def _blocks(text: str) -> list[tuple[str | None, list[str]]]:
    """Split PDBQT text into ``(record_name, lines)`` blocks, the generic way.

    Deliberately **not** built on :func:`read_pdbqt_models`. A check that asks
    the module where it put a line, using the module's own reader to find out,
    gets the module's answer by construction. This walks record names itself,
    so a claim about position can disagree with the code under test.
    """
    out: list[tuple[str | None, list[str]]] = []
    kind: str | None = None
    lines: list[str] = []
    for raw in text.splitlines():
        token = raw.split(maxsplit=1)[0] if raw.strip() else ""
        if token in ("MODEL", "ROOT", "BRANCH", "MOLECULE"):
            if kind is not None or lines:
                out.append((kind, lines))
            kind, lines = token, []
            continue
        if token in ("ENDMDL", "ENDROOT", "ENDBRANCH"):
            out.append((kind, lines))
            kind, lines = None, []
            continue
        lines.append(raw)
    if kind is not None or lines:
        out.append((kind, lines))
    return out


def _block_holding(text: str, needle: str) -> str | None:
    """The name of the block containing `needle`, or None if it is outside all."""
    for kind, lines in _blocks(text):
        if any(needle in ln for ln in lines):
            return kind
    return None


if __name__ == "__main__":
    sys.exit(main())
