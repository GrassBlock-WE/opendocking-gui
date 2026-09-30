"""Command-line interface for Open Docking.

Subcommands
-----------
``prep-receptor``  convert a ``.pdb`` receptor to PDBQT
``prep-ligand``    convert a ligand to a docked-ready table / PDBQT
``rec-grid``       precalculate and store affinity maps
``dock``           dock one or more ligands
``split``          split a multi-model PDBQT into separate files
``info``           report engine capabilities
``workbench``      launch the interactive 3-D viewer

The box must be given explicitly for ``dock`` and ``rec-grid``. Defaulting it
from the receptor would be a trap: a whole-protein box is both enormous in
memory and useless as a search region.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

__all__ = ["main", "build_parser"]


def _positive_float(value: str) -> float:
    try:
        v = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {v}")
    return v


def _add_box_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("search box (required)")
    group.add_argument("--center_x", type=float, required=True, help="box centre, Ångström")
    group.add_argument("--center_y", type=float, required=True)
    group.add_argument("--center_z", type=float, required=True)
    group.add_argument("--size_x", type=_positive_float, required=True, help="box edge, Ångström")
    group.add_argument("--size_y", type=_positive_float, required=True)
    group.add_argument("--size_z", type=_positive_float, required=True)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="odcli",
        description="Open Docking — molecular docking engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from . import __version__

    parser.add_argument(
        "--version", action="version", version=f"odcli {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prep-receptor ----------------------------------------------------
    p = sub.add_parser(
        "prep-receptor",
        help="convert a .pdb receptor to PDBQT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--receptor", required=True, type=Path, help="input .pdb/.ent")
    p.add_argument("-o", "--output", required=True, type=Path, help="output .pdbqt")
    p.add_argument("--keep-waters", action="store_true", help="keep crystallographic waters")
    p.add_argument(
        "--drop-heterogens",
        action="store_true",
        help="remove ions and cofactors (they are kept by default)",
    )
    p.set_defaults(func=_cmd_prep_receptor)

    # --- prep-ligand ------------------------------------------------------
    p = sub.add_parser(
        "prep-ligand",
        help="prepare a ligand and report its properties",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-l", "--ligand", required=True, type=Path, help="input structure")
    p.add_argument("-o", "--output", type=Path, help="optional PDBQT output")
    p.add_argument("--keep-hydrogens", action="store_true", help="keep non-polar hydrogens")
    p.set_defaults(func=_cmd_prep_ligand)

    # --- rec-grid ---------------------------------------------------------
    p = sub.add_parser(
        "rec-grid",
        help="precalculate receptor affinity maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--receptor", required=True, type=Path, help="receptor .pdbqt")
    p.add_argument("-o", "--output", required=True, type=Path, help="output directory")
    p.add_argument("--scoring", default="vina", choices=("vina", "vinardo"))
    p.add_argument("--spacing", type=_positive_float, default=0.375, help="Ångström")
    _add_box_arguments(p)
    p.set_defaults(func=_cmd_rec_grid)

    # --- dock -------------------------------------------------------------
    p = sub.add_parser(
        "dock",
        help="dock one or more ligands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--receptor", required=True, type=Path, help="receptor .pdbqt")
    p.add_argument(
        "-l",
        "--ligand",
        required=True,
        type=Path,
        help="ligand file, or a directory of them",
    )
    p.add_argument("-o", "--output", type=Path, help="output file or directory")
    p.add_argument(
        "-e", "--exhaustiveness", type=int, default=8, help="independent search walks"
    )
    p.add_argument("-m", "--num_modes", type=int, default=9, help="distinct poses to report")
    p.add_argument("--rmsd-cutoff", type=_positive_float, default=1.0, help="Ångström")
    p.add_argument("--scoring", default="vina", choices=("vina", "vinardo"))
    p.add_argument("--spacing", type=_positive_float, default=0.375, help="Ångström")
    p.add_argument(
        "--mode", default="mc", choices=("mc", "lga", "both"), help="search strategy"
    )
    p.add_argument("--seed", type=int, default=None, help="fix the random seed")
    p.add_argument("--steps", type=int, default=None, help="local-search steps per walk")
    p.add_argument("--json", action="store_true", help="emit results as JSON")
    _add_box_arguments(p)
    p.set_defaults(func=_cmd_dock)

    # --- split ------------------------------------------------------------
    p = sub.add_parser(
        "split",
        help="split a multi-model PDBQT into one file per pose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, type=Path)
    p.add_argument("-o", "--output-dir", required=True, type=Path)
    p.set_defaults(func=_cmd_split)

    # --- info -------------------------------------------------------------
    p = sub.add_parser("info", help="report engine capabilities")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_info)

    # --- workbench --------------------------------------------------------
    p = sub.add_parser("workbench", help="launch the interactive 3-D viewer")
    p.add_argument("-r", "--receptor", type=Path, help="receptor .pdbqt")
    p.add_argument("-l", "--ligand", type=Path, help="ligand .pdbqt")
    p.add_argument("-p", "--poses", type=Path, help="docked poses .pdbqt")
    p.set_defaults(func=_cmd_workbench)

    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_prep_receptor(args: argparse.Namespace) -> int:
    from .prep import prepare_receptor

    text = prepare_receptor(
        args.receptor,
        keep_waters=args.keep_waters,
        keep_heterogens=not args.drop_heterogens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    n_atoms = sum(1 for line in text.splitlines() if line.startswith(("ATOM", "HETATM")))
    print(f"wrote {args.output} ({n_atoms} atoms)")
    return 0


def _cmd_prep_ligand(args: argparse.Namespace) -> int:
    from .core import Ligand
    from .prep import prepare_ligand

    elements, charges, coords, bonds, names = prepare_ligand(
        args.ligand, keep_hydrogens=args.keep_hydrogens
    )
    ligand = Ligand.from_arrays(elements, charges, coords, bonds, names)

    print(f"{args.ligand.name}:")
    print(f"  atoms            {ligand.num_atoms}")
    print(f"  rotatable bonds  {ligand.num_torsions}")
    print(f"  degrees of freedom  {ligand.num_dof}")
    print(f"  radius           {ligand.radius:.2f} Å")
    counts: dict[str, int] = {}
    for kind in ligand.atom_kinds:
        counts[kind] = counts.get(kind, 0) + 1
    print("  atom classes     " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if args.output:
        from .pdbqt_writer import ligand_to_pdbqt

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            ligand_to_pdbqt(ligand, coords, names), encoding="utf-8"
        )
        print(f"  wrote            {args.output}")
    return 0


def _cmd_rec_grid(args: argparse.Namespace) -> int:
    from .core import GridBox, Receptor, scoring_descriptions

    box_ = GridBox.from_center_size(
        (args.center_x, args.center_y, args.center_z), (args.size_x, args.size_y, args.size_z)
    )
    receptor = Receptor.from_pdbqt(args.receptor)
    est = receptor.estimate_memory_mb(box_, args.spacing)
    print(f"receptor: {receptor.num_atoms} atoms, "
          f"{receptor.num_polar_hydrogens} polar hydrogens")
    print(f"box: centre {box_.center}, size {box_.size}")
    print(f"estimated maps: {est:.1f} MB")
    if args.scoring == "vina":
        print("scoring: " + scoring_descriptions()[0])

    start = time.time()
    maps = receptor.precalculate(box_, args.scoring, args.spacing)
    elapsed = time.time() - start
    print(f"precalculated {maps.num_points} points in {elapsed:.2f} s "
          f"({maps.memory_mb:.1f} MB actual)")

    args.output.mkdir(parents=True, exist_ok=True)
    maps.write_map_files(args.output)
    print(f"wrote AutoDock .map files to {args.output}")
    return 0


def _cmd_dock(args: argparse.Namespace) -> int:
    from .core import GridBox, Receptor, dock, load_ligand, load_receptor

    box_ = GridBox.from_center_size(
        (args.center_x, args.center_y, args.center_z), (args.size_x, args.size_y, args.size_z)
    )
    receptor = load_receptor(args.receptor)
    print(f"receptor: {receptor.num_atoms} atoms", file=sys.stderr)

    ligands = _collect_ligands(args.ligand)
    if not ligands:
        print(f"error: no ligands found at {args.ligand}", file=sys.stderr)
        return 2

    maps = receptor.precalculate(box_, args.scoring, args.spacing)
    print(
        f"maps: {maps.num_points} points, {maps.memory_mb:.1f} MB",
        file=sys.stderr,
    )

    out_dir: Path | None = None
    out_file: Path | None = None
    if args.output is not None:
        if args.ligand.is_dir() or len(ligands) > 1:
            out_dir = args.output
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_file = args.output
            out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    failed = 0
    for path in ligands:
        try:
            ligand = load_ligand(path)
        except Exception as exc:
            print(f"error: could not read {path}: {exc}", file=sys.stderr)
            failed += 1
            continue

        if out_dir is not None:
            dest = out_dir / f"{path.stem}_docked.pdbqt"
        else:
            dest = out_file  # type: ignore[assignment]

        result = dock(
            ligand,
            maps,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            rmsd_cutoff=args.rmsd_cutoff,
            seed=args.seed,
            mode=args.mode,
            scoring=args.scoring,
            steps=args.steps,
        )
        if dest is not None:
            result.write_pdbqt(dest)

        records.append(
            {
                "ligand": str(path),
                "output": str(dest) if dest else None,
                "num_poses": result.num_poses,
                "best_energy": result.best_energy,
                "energies": result.energies.tolist(),
                "rmsd": result.rmsds.tolist(),
                "elapsed_seconds": result.elapsed_seconds,
                "rejected_pose_count": result.rejected_pose_count,
                "num_torsions": ligand.num_torsions,
            }
        )
        if not args.json:
            print(f"\n{path.name}  ({ligand.num_torsions} torsions)")
            print(result.summary())
            if dest is not None:
                print(f"  -> {dest}")

    if args.json:
        print(json.dumps(records, indent=2))
    return 1 if failed and not records else 0


def _collect_ligands(path: Path) -> list[Path]:
    """Expand a file or directory into a sorted list of ligand files."""
    if path.is_file():
        return [path]
    if path.is_dir():
        out: list[Path] = []
        for pattern in ("*.pdbqt", "*.sdf", "*.mol2", "*.mol", "*.pdb"):
            out.extend(sorted(path.glob(pattern)))
        return out
    return []


def _cmd_split(args: argparse.Namespace) -> int:
    from .pdbqt_writer import split_pdbqt_models

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = split_pdbqt_models(args.input, args.output_dir)
    print(f"wrote {len(written)} pose file(s) to {args.output_dir}")
    return 0 if written else 1


def _cmd_info(args: argparse.Namespace) -> int:
    from .core import available_backends, engine_version, gpu_status, scoring_descriptions

    info = {
        "odock_version": engine_version(),
        "scoring_functions": scoring_descriptions(),
        "gpu": gpu_status(),
        "backends": available_backends(),
    }
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Open Docking engine {info['odock_version']}")
        print("scoring functions:")
        for desc in info["scoring_functions"]:
            print(f"  {desc}")
        gpu = info["gpu"]
        if not gpu["compiled"]:
            print("gpu: not compiled in (rebuild with --features gpu)")
        elif gpu["available"]:
            print("gpu: compiled in and an adapter is available")
        else:
            print("gpu: compiled in but no adapter could be opened")
        print("parallel backends: " + ", ".join(info["backends"]))
    return 0


def _cmd_workbench(args: argparse.Namespace) -> int:
    try:
        from .workbench import launch
    except ImportError as exc:
        print(
            f"error: the workbench needs PyQt6 and moderngl ({exc}).\n"
            "Install them with:  pip install PyQt6 moderngl numpy-stl",
            file=sys.stderr,
        )
        return 3
    return launch(args.receptor, args.ligand, args.poses)


def _use_utf8_console() -> None:
    """Make ``print`` survive a console that is not UTF-8.

    A Windows console defaults to a legacy code page (GBK on a Chinese
    install, cp1252 elsewhere). Python then encodes the stream with that codec,
    so a single ``Å`` in a label raises ``UnicodeEncodeError`` and takes down a
    run that was otherwise fine. Asking for UTF-8 and falling back to lossy
    replacement is strictly better than crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic streams
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``odcli`` console script."""
    _use_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
