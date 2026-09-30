"""PDBQT writing and splitting, in pure Python.

The Rust engine writes docked poses itself; this module covers the cases where
Python already holds the coordinates — a prepared ligand, or a file that needs
splitting — and keeps the format in one readable place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "format_atom_line",
    "ligand_to_pdbqt",
    "write_pose",
    "split_pdbqt_models",
    "read_pdbqt_models",
]

# PDBQT column positions (1-based, inclusive), per the AutoDock 4
# specification.  Coordinates at 31–38/39–46/47–54, charge at 71–76, atom type
# at 78–79.  Getting these wrong silently produces a file that loads but
# mis-parses, so they are spelled out once here.
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
    """Pad ``value`` to exactly the width of a PDB column range."""
    width = cols[1] - cols[0]
    text = value[:width]
    return text.rjust(width) if right else text.ljust(width)


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
        _field(str(resseq), _COL_RESSEQ),
        " ",
        "   ",
        _field(f"{coord[0]:.{p}f}", _COL_X),
        _field(f"{coord[1]:.{p}f}", _COL_Y),
        _field(f"{coord[2]:.{p}f}", _COL_Z),
        _field("1.00", _COL_OCC),
        _field("0.00", _COL_BFAC),
        "    ",
        _field(f"{charge:.3f}", _COL_CHARGE),
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
    """Split a ``TYPE_ELEMn`` name back into a PDB name and a PDBQT type."""
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


def read_pdbqt_models(path: str | Path) -> list[list[str]]:
    """Split a multi-model PDBQT into one list of lines per pose.

    A file without any ``MODEL`` record yields a single pose, so this works on
    both docked output and a plain structure file.
    """
    models: list[list[str]] = []
    current: list[str] | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        record = raw.split(maxsplit=1)[0] if raw.strip() else ""
        if record == "MODEL":
            current = []
            continue
        if record == "ENDMDL":
            if current is not None:
                models.append(current)
            current = None
            continue
        if current is None and record in ("ATOM", "HETATM"):
            current = []
        if current is not None:
            current.append(raw)
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
