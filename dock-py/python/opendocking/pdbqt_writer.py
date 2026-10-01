"""PDBQT writing, splitting and column validation, in pure Python.

The Rust engine writes docked poses itself; this module covers the cases where
Python already holds the coordinates — a prepared ligand, or a file that needs
splitting — and keeps the format in one readable place.

# Why the engine does not validate columns, and this module does

**The engine's reader is deliberately tolerant, and that is a choice, not an
oversight.** ``Ligand.from_pdbqt`` splits on whitespace: measured on this
project's own output, deleting the space at column 31 of an otherwise correct
line still parses to identical coordinates. That tolerance is worth having --
hand-edited files, files from other toolkits and files with a stray blank column
all load instead of refusing -- but it has a consequence that has to be written
down somewhere, or it reads as a bug: **the engine will happily load a file this
project mis-formatted, and nothing inside the engine will notice.**

So strictness lives here, at the boundary this project owns, where files enter
the application. :func:`read_pdbqt_models` is that boundary -- it is what the
workbench and the CLI both read a pose file through -- and it now checks the
columns of every line it passes, using the same table the writer writes to.
:func:`validate_pdbqt_columns` is the check on its own, for a caller who would
rather be told about a malformed line than refused.

The split is deliberate and worth stating plainly: **the engine answers "can
this file be read?", and this module answers "is this file laid out the way
PDBQT says it must be?"** A file that fails the second question is refused here
rather than silently mis-parsed further in, and ``strict=False`` is there for
the caller who has a file they know is odd and wants it anyway.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "format_atom_line",
    "ligand_to_pdbqt",
    "write_pose",
    "split_pdbqt_models",
    "read_pdbqt_models",
    "validate_pdbqt_columns",
    "PdbqtFormatError",
]

# PDBQT column positions, per the AutoDock 4 specification.  Coordinates at
# 31-38/39-46/47-54, charge at 71-76, atom type at 78-79.  Getting these wrong
# silently produces a file that loads but mis-parses, so they are spelled out
# once here.
#
# **Read the constants as 0-based starts.** Each pair is ``(start, end)`` with
# ``start`` one *before* the first column, so a field's real width is
# ``end - start`` and its 1-based column range is ``start + 1`` to ``end``.
# ``_COL_CHARGE = (70, 76)`` is therefore columns **71-76**, not 70-76, and
# ``_COL_TYPE = (77, 79)`` is **78-79**, not 77-79. That is not a nitpick: an
# earlier draft of a review quoted this comment's numbers back as "the charge is
# in 70-76 and the type in 77-79" and would have written a check against the
# wrong columns. The two real, third-party-produced files this project ships
# (``examples/ibuprofen_prep.pdbqt``, ``examples/1crn_prep.pdbqt``, both written
# by Meeko) have ``' 0.000'`` in columns 71-76, a space in 77, and the type in
# 78-79, which is what the constants here produce.
_COL_SERIAL = (6, 11)
_COL_NAME = (12, 16)
_COL_RESNAME = (17, 20)
_COL_CHAIN = (21, 22)
_COL_RESSEQ = (22, 26)
_COL_X = (30, 38)
_COL_Y = (38, 46)
_COL_Z = (46, 54)
_COL_OCC = (54, 60)
_COL_BFAC = (60, 66)
_COL_CHARGE = (70, 76)
_COL_TYPE = (77, 79)


def _field(value: str, cols: tuple[int, int], right: bool = True) -> str:
    """Pad ``value`` to exactly the width of a PDB column range.

    Truncates rather than complains, which is right for a *label* -- an atom
    name one character too long is a cosmetic problem -- and wrong for a
    *value*. See :func:`_number`.
    """
    width = cols[1] - cols[0]
    text = value[:width]
    return text.rjust(width) if right else text.ljust(width)


def _number(text: str, cols: tuple[int, int], label: str) -> str:
    """A fixed-width numeric column that refuses to lose a digit.

    The rule this encodes: **truncating a label loses information, truncating a
    number invents it.** A coordinate field is 8 wide, so ``-12345.678`` is 10
    characters and ``_field`` would have cut it to ``-12345.6`` -- a *different
    coordinate*, in a file still exactly 79 characters, that loads without
    complaint and mis-parses. Measured on this writer before the check existed:
    ``-12345.678`` came out as ``-12345.6`` and ``12345.678`` as ``12345.67``,
    both silently. The 6-wide charge field was worse, because a lost digit there
    is a lost third of a decimal place: ``-12.345`` became ``-12.34`` and
    ``123.456`` became ``123.45``.

    The ``len(line) != 79`` assertion at the end of :func:`format_atom_line`
    could never have caught either, because ``_field`` pads the truncated text
    back out to the same width. That is what makes the length assertion
    decorative for this class of defect and load-bearing only against a changed
    column constant.
    """
    width = cols[1] - cols[0]
    if len(text) > width:
        raise ValueError(
            f"{label} {text!r} needs {len(text)} characters but columns "
            f"{cols[0] + 1}-{cols[1]} hold {width}; refusing to truncate a "
            f"number, because a truncated one is a different number in a file "
            f"that still loads"
        )
    return text.rjust(width)



def format_atom_line(
    serial: int,
    name: str,
    resname: str,
    resseq: int,
    coord: Sequence[float],
    charge: float,
    atom_type: str,
    *,
    chain: str = " ",
    precision: int = 3,
) -> str:
    """Format one PDBQT ``ATOM`` record at the exact column positions.

    Parameters
    ----------
    atom_type:
        The AutoDock type token, e.g. ``"C"``, ``"OA"``, ``"Cl"``, ``"HD"``.
        Meeko writes ``Cl`` and ``Br`` with mixed case; the parser accepts
        either, and matching Meeko keeps the files readable by other tools.
    chain:
        One-character chain identifier. It is a whole column, so it was left
        out of this signature for a while and everything came out as ``" "`` --
        which is a legal file that has quietly lost half its residue identity.
        Anything that reads structure rather than just coordinates (a viewer
        grouping atoms into residues, a tool inferring which atoms are bonded)
        needs it, so it is a parameter now.
    """
    if len(atom_type) > 2:
        raise ValueError(f"PDBQT atom type {atom_type!r} is longer than two characters")
    p = precision
    parts = [
        "ATOM  ",
        _field(str(serial), _COL_SERIAL),
        " ",
        _field(name, _COL_NAME, right=len(name.strip()) == 1),
        " ",
        _field(resname, _COL_RESNAME, right=False),
        " ",
        _field((chain or " ")[:1], _COL_CHAIN, right=False),
        _number(str(resseq), _COL_RESSEQ, "residue number"),
        " ",
        "   ",
        _number(f"{coord[0]:.{p}f}", _COL_X, "x coordinate"),
        _number(f"{coord[1]:.{p}f}", _COL_Y, "y coordinate"),
        _number(f"{coord[2]:.{p}f}", _COL_Z, "z coordinate"),
        _field("1.00", _COL_OCC),
        _field("0.00", _COL_BFAC),
        "    ",
        _number(f"{charge:.3f}", _COL_CHARGE, "partial charge"),
        " ",
        _field(atom_type, _COL_TYPE, right=False),
    ]
    line = "".join(parts)
    if len(line) != 79:
        raise AssertionError(
            f"PDBQT line is {len(line)} characters, expected 79: {line!r}"
        )
    return line


def ligand_to_pdbqt(
    ligand,
    coords: np.ndarray,
    names: Sequence[str] | None = None,
    *,
    resname: str = "UNL",
) -> str:
    """Serialise a prepared ligand as a PDBQT string.

    This is a flat (non-``BRANCH``-tree) file. The engine perceives rotatable
    bonds from the covalent graph when no ``TORSION`` records are present, so
    this is a valid docking input; it is mainly used for inspection and for
    handing a prepared ligand to other tools.

    **Measured on ``examples/ibuprofen_prep.pdbqt``: the flat file does *not*
    change the torsion tree.** Written with the right `names` and read back, the
    engine reports the same 4 rotatable bonds it reports for the shipped file,
    and the same ``atom_kinds``. An earlier draft of this docstring claimed the
    count came back as 5 and blamed the absence of ``TORSION`` records; that
    was wrong. The 4-comes-back-as-5 is real, but it belongs to the `names`
    fallback below and has nothing to do with the flat layout: a file whose
    oxygens and nitrogens are all typed ``"C`` is a different molecule, and the
    engine quite reasonably finds a different number of rotatable bonds in it.

    `names` is not optional in practice. **Omitting it types every atom as
    ``"C"``**, because the fallback is ``f" C{i + 1}", "C"``: the file is
    well-formed, loads, and describes a molecule in which the oxygens, nitrogens
    and sulfurs are all carbon, so every atom reads back as hydrophobic and the
    rotatable-bond count moves with it. The measured consequence on ibuprofen is
    that ``atom_kinds`` comes back all ``hydrophobic`` instead of the source
    file's mix of hydrophobic, acceptor and donor-acceptor, and that
    ``num_torsions`` comes back 5 rather than 4. Every caller inside this
    project passes `names` (``cli.py``, ``ligand_check.py``, ``make_data.py``),
    so this has never bitten in-tree; it is a trap for the first external caller
    who does not.
    """
    lines = [
        "REMARK  Prepared by Open Docking (opendocking)",
        f"REMARK  {ligand.num_atoms} atoms, {ligand.num_torsions} rotatable bonds",
        f"TORSDOF {ligand.num_torsions}",
    ]
    for i in range(ligand.num_atoms):
        if names is not None and i < len(names):
            name, atom_type = _split_name(names[i])
        else:
            name, atom_type = f" C{i + 1}", "C"
        lines.append(
            format_atom_line(
                i + 1,
                name,
                resname,
                1,
                (coords[i, 0], coords[i, 1], coords[i, 2]),
                0.0,
                atom_type,
            )
        )
    return "\n".join(lines) + "\n"


def _split_name(token: str) -> tuple[str, str]:
    """Split a ``TYPE_ELEMn`` name back into a PDB name and a PDBQT type.

    The two branches are not equally safe, and only the first one is used
    in-tree. ``prep`` hands out names like ``"OA_O1"``, which carry their own
    type prefix, and that prefix is what the writer puts in columns 78-79.

    A **bare** name takes the second branch, which takes the first two
    characters as the type. Measured: ``"CA"`` becomes type ``"CA"``, which in
    AutoDock is *calcium*, so a bare name quietly types a carbon alpha as an
    ion; and ``"C1"`` becomes type ``"C1"``, which is not a type at all. The
    engine reads both back as ``other`` rather than ``hydrophobic``. No caller
    inside this project can reach that branch -- ``cli.py``,
    ``ligand_check.py`` and ``make_data.py`` all pass ``prep``'s prefixed
    names -- so this is a trap for an external caller, and it is pinned by
    ``scripts/pdbqt_check.py`` rather than left as folklore. Guessing an
    element symbol from a PDB name is not attempted here on purpose: the guess
    would be right more often than the first two characters are.
    """
    if "_" in token:
        pdbqt_type, rest = token.split("_", 1)
        return f" {rest[:3]:<3}", pdbqt_type
    return f" {token[:3]:<3}", token[:2]


def write_pose(
    path: str | Path,
    coords: np.ndarray,
    atom_types: Sequence[str],
    atom_names: Sequence[str] | None = None,
    energy: float | None = None,
    resname: str = "UNL",
) -> None:
    """Write a single pose as PDBQT.

    The ``REMARK VINA RESULT`` line goes **inside** the ``MODEL`` block. That is
    the AutoDock convention, and a per-model splitter such as
    :func:`read_pdbqt_models` only collects what lies between ``MODEL`` and
    ``ENDMDL`` -- a remark written before the first ``MODEL`` is unattributed,
    so every pose reads back with no energy.
    """
    lines: list[str] = ["MODEL"]
    if energy is not None:
        lines.append(f"REMARK VINA RESULT:    {energy:.1f}  0.000  0.000")
    for i in range(coords.shape[0]):
        if atom_names is not None and i < len(atom_names):
            name, pdbqt_type = _split_name(atom_names[i])
        else:
            name, pdbqt_type = f" C{i + 1}", atom_types[i]
        lines.append(
            format_atom_line(
                i + 1, name, resname, 1, coords[i], 0.0, pdbqt_type
            )
        )
    lines.append("ENDMDL")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class PdbqtFormatError(ValueError):
    """A file is readable but is not laid out the way PDBQT says it must be.

    A :class:`ValueError` so that a caller already handling bad input from this
    module keeps working, and so ``except ValueError`` in a caller does not
    have to learn a new name. The message names the file, the 1-based line
    number and the offending bytes, because "invalid PDBQT" with no location is
    the least useful thing an error can be.
    """

    def __init__(self, source: str, problems: Sequence[str]) -> None:
        self.source = source
        self.problems = list(problems)
        shown = self.problems[:8]
        more = len(self.problems) - len(shown)
        super().__init__(
            f"{source}: {len(self.problems)} line(s) are not laid out as "
            f"PDBQT requires:\n  " + "\n  ".join(shown)
            + (f"\n  ... and {more} more" if more > 0 else "")
            + "\nThe engine's own reader splits on whitespace, so it would "
              "have loaded this file and reported the wrong coordinates."
        )


#: Record names a line may begin with. Anything else is flagged: a line that
#: has lost its record name, or a mangled ATOM line, is exactly what hides in
#: the region ahead of a pose's first atom. Measured over the ten PDBQT files
#: this project ships -- 981 lines including 144 written by the engine into
#: ``crambin_pose.pdbqt`` -- **zero** fall outside this set, so refusing them
#: costs no real file.
_KNOWN_RECORDS = frozenset({
    "ATOM", "HETATM", "MODEL", "ENDMDL", "REMARK", "TORSDOF", "ROOT",
    "ENDROOT", "BRANCH", "ENDBRANCH", "MOLECULE", "END", "TER", "CONECT",
    "HEADER", "TITLE", "COMPND", "CRYST1", "MASTER", "HELIX", "SHEET",
    "SITE", "ANISOU", "BEGIN_RES", "END_RES", "IF", "WEIGHT", "MOVES",
})


#: A coordinate or charge field, as this writer emits it: right-justified, so
#: leading spaces and then digits, with a decimal point. ``float()`` alone is
#: **not** enough here, and the reason is worth writing down because it was
#: measured: on a correct line, reading x at 32-39 instead of 31-38 gives
#: ``'  1.234 '``, and ``float('  1.234 ')`` is perfectly happy -- it is
#: 1.234. So a one-column right shift, which is what "someone inserted a column
#: earlier in the record" looks like, would sail straight through a validator
#: that only tried to parse the number. Requiring the field to be
#: right-justified catches that, because the shifted field has a *trailing*
#: space. Measured over the ten shipped files: 786 of 786 ATOM lines match, so
#: this costs no real file.
_COORDINATE = re.compile(r"^ *-?\d+\.\d+$")

#: The serial field, which is an integer and is never signed in practice.
_INTEGER = re.compile(r"^ *\d+$")

#: An atom type: one or two letters, which is every AutoDock type. This
#: replaced a check for "longer than two characters" that **could never fire**,
#: because the type field is ``line[77:79]`` and so is at most two characters
#: long by slicing alone -- a dead branch that looked like coverage. Caught by
#: mutation rather than by reading, which is the argument for mutation testing
#: in one sentence. Measured over the ten shipped files: the types actually
#: used are NA, C, OA, N, S, HD and A, and 786 of 786 lines match.
_ATOM_TYPE = re.compile(r"^[A-Za-z]{1,2}$")


def validate_pdbqt_columns(
    lines: Iterable[str], *, source: str = "<text>"
) -> list[str]:
    """Report the lines of a PDBQT that are not laid out as PDBQT requires.

    Returns one message per problem, in file order; an empty list means the
    layout is sound. This is the strictness the engine deliberately does not do,
    and it is deliberately narrow: it refuses only what is *provably* wrong,
    because a validator that rejects correct files is worse than none.

    What is refused, and why each one is safe:

    * An ``ATOM``/``HETATM`` line whose serial, coordinates or charge does not
      match the shape PDBQT says that field has **at the documented columns**,
      or whose atom type is not one or two letters. This is the case that
      matters and the one the engine cannot see: the line can be exactly 79
      characters long and still have the wrong thing in the wrong place.
    * An ``ATOM``/``HETATM`` line shorter than 76 characters, which cannot hold
      the charge field at all.
    * Any non-blank line that does not begin with a record name from
      :data:`_KNOWN_RECORDS`.

    What is *tolerated*, on purpose, and each is a measured decision rather than
    an oversight:

    * A **blank charge field** is accepted as zero. All 786 shipped ATOM lines
      carry a real number there, but a hand-written file that leaves it blank is
      still a file a docking program reads, and refusing it would turn a
      cosmetic omission into a dead end.
    * A **blank atom type** is accepted, because this writer's own
      ``format_atom_line`` pads an empty type to two spaces rather than failing,
      and a validator that rejected the writer's own output would be a joke.
    * **Line length is not itself checked.** A 78- or 80-character line that
      still parses at every column is left alone, because trailing whitespace
      gets stripped by every text editor ever written.

    What it **cannot** do, stated here so that it is not overclaimed later: a
    field read one column *too far left* still looks like a valid
    right-justified number -- reading x at 30-37 on a correct line gives
    ``'    1.23'``, which is a plausible float and the wrong value. A reader has
    no way to know what a number was supposed to be, so only a comparison
    against the source molecule could catch that. This catches a layout that
    stopped parsing, not a plausible number in the wrong slot.

    The coordinate columns are the same numbers the writer writes to, and they
    are restated here as 1-based ranges rather than taken from the ``_COL_*``
    pairs on purpose: a validator that read its expectations out of the same
    constants it is checking could not catch them being wrong.
    """
    problems: list[str] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        token = line.split(maxsplit=1)[0]
        if token not in _KNOWN_RECORDS:
            problems.append(
                f"line {number}: starts with {token!r}, which is not a PDBQT "
                f"record name: {line[:60]!r}"
            )
            continue
        if token not in ("ATOM", "HETATM"):
            continue
        if len(line) < 76:
            problems.append(
                f"line {number}: {token} record is {len(line)} characters, too "
                f"short to hold the charge field at columns 71-76: {line!r}"
            )
            continue
        for label, start, end, shape in (
            ("serial", 7, 11, _INTEGER),
            ("x", 31, 38, _COORDINATE),
            ("y", 39, 46, _COORDINATE),
            ("z", 47, 54, _COORDINATE),
        ):
            raw = line[start - 1:end]
            if not shape.match(raw):
                problems.append(
                    f"line {number}: {label} at columns {start}-{end} is "
                    f"{raw!r}, which is not a right-justified PDBQT "
                    f"{label} field"
                )
        charge = line[70:76]
        if charge.strip() and not _COORDINATE.match(charge):
            problems.append(
                f"line {number}: charge at columns 71-76 is {charge!r}, "
                f"which is not a right-justified PDBQT charge field"
            )
        atom_type = line[77:79].strip()
        if atom_type and not _ATOM_TYPE.match(atom_type):
            problems.append(
                f"line {number}: atom type at columns 78-79 is {atom_type!r}, "
                f"which is not one or two letters"
            )
    return problems


def read_pdbqt_models(
    path: str | Path, *, strict: bool = True
) -> list[list[str]]:
    """Split a multi-model PDBQT into one list of lines per pose.

    A file without any ``MODEL`` record yields a single pose, so this works on
    both docked output and a plain structure file.

    **Lines that arrive before a pose's first atom belong to that pose.** This
    used to drop them on the floor: the loop only opened an implicit pose when
    it saw an ``ATOM`` or ``HETATM`` line, and discarded everything before that,
    so a file with no ``MODEL`` record lost its ``REMARK VINA RESULT`` line
    entirely. That quietly contradicted :func:`split_pdbqt_models`, whose
    ``REMARK`` block is the *first* line of each piece it writes, and made a
    split pose file read back with no energy -- the workbench then displayed
    "energy not in file" for a file that plainly contained one. Leading lines
    are now buffered and handed to whichever pose opens first, so the reader and
    the splitter agree about what belongs to a pose. A file with no atoms at all
    still yields no poses, so leading comments alone do not invent one.

    **Every line is checked before any pose is attributed** (see
    :func:`validate_pdbqt_columns`, and the module docstring for why the check
    is here and not in the engine). That ordering is the point: the buffered
    pre-atom region is exactly where a malformed header would otherwise hide,
    and a validator that ran after the poses were built would have already
    handed the workbench a pose to display.

    Parameters
    ----------
    strict:
        Refuse a file whose columns are wrong, raising
        :class:`PdbqtFormatError`. ``strict=False`` returns the lines anyway
        for a caller with a file they know is odd, and then that caller owns
        the mis-parse.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw_lines = text.splitlines()
    if strict:
        problems = validate_pdbqt_columns(raw_lines, source=str(path))
        if problems:
            raise PdbqtFormatError(str(path), problems)

    models: list[list[str]] = []
    current: list[str] | None = None
    leading: list[str] = []
    for raw in raw_lines:
        record = raw.split(maxsplit=1)[0] if raw.strip() else ""
        if record == "MODEL":
            current, leading = leading, []
            continue
        if record == "ENDMDL":
            if current is not None:
                models.append(current)
            current = None
            continue
        if current is None and record in ("ATOM", "HETATM"):
            current, leading = leading, []
        if current is not None:
            current.append(raw)
        else:
            leading.append(raw)
    if current:
        models.append(current)
    return [m for m in models if m]


def split_pdbqt_models(
    path: str | Path, output_dir: str | Path
) -> list[Path]:
    """Write one PDBQT file per pose of a multi-model file.

    Returns the list of files written. Energies are carried into each file's
    ``REMARK VINA RESULT`` line when the source had one, so a split file is
    still self-describing.

    The piece is written *without* its ``MODEL``/``ENDMDL`` wrapper, which puts
    the energy remark at the very top of the file -- before the first atom.
    Reading one of these back is therefore only lossless because
    :func:`read_pdbqt_models` attributes lines that precede a pose's first atom
    to that pose. The two functions are a pair, and that is the pair the check
    exercises end to end: with the reader's pre-atom lines dropped, every piece
    here still carries its energy on disk and still reports none.
    """
    source = Path(path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = source.read_text(encoding="utf-8")
    # Split on MODEL boundaries, keeping the REMARK block that precedes each.
    blocks: list[tuple[str, list[str]]] = []
    pending: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        record = line.split(maxsplit=1)[0] if line.strip() else ""
        if record == "MODEL":
            current = []
            continue
        if record == "ENDMDL":
            if current:
                blocks.append(("\n".join(pending + current) + "\n", current))
            current = None
            continue
        if current is None and line.startswith("REMARK"):
            pending.append(line)
            continue
        if current is not None:
            current.append(line)
    if not blocks and current:  # pragma: no cover - defensive
        blocks.append(("\n".join(pending + current) + "\n", current))

    written: list[Path] = []
    for index, (_header, body) in enumerate(blocks, start=1):
        dest = out_dir / f"{source.stem}_{index:02d}.pdbqt"
        dest.write_text("".join(line + "\n" for line in body), encoding="utf-8")
        written.append(dest)
    return written
