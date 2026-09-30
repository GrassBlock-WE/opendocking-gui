"""Command-line interface for Open Docking.

Subcommands
-----------
``prep-receptor``  convert a ``.pdb`` receptor to PDBQT
``prep-ligand``    convert a ligand to a docked-ready table / PDBQT
``rec-grid``       precalculate and store affinity maps
``dock``           dock one or more ligands
``sites``          list candidate binding sites in a receptor
``split``          split a multi-model PDBQT into separate files
``info``           report engine capabilities
``workbench``      launch the interactive 3-D viewer

The box must be given for ``dock`` and ``rec-grid``, either as the six explicit
values or as ``--auto-box N`` to take the Nth site the pocket search finds.
Defaulting it from the receptor would be a trap: a whole-protein box is both
enormous in memory and useless as a search region.
"""

from __future__ import annotations

import argparse
import json
import math
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
    """Explicit box, or ``--auto-box`` to derive one from the receptor.

    The explicit box is *not* `required=True` at the argparse level any more,
    because ``--auto-box`` has to be able to stand in for it. The "you must
    give me a box one way or the other" rule did not go away though: it is
    enforced in `_resolve_box`, which is the only place that knows whether
    `--auto-box` was passed. Leaving it to argparse would have meant the error
    says "the following arguments are required" for a command where the user
    did exactly what the help text says.
    """
    group = parser.add_argument_group("search box (give the six values, or --auto-box)")
    group.add_argument("--center_x", type=float, help="box centre, Ångström")
    group.add_argument("--center_y", type=float)
    group.add_argument("--center_z", type=float)
    group.add_argument("--size_x", type=_positive_float, help="box edge, Ångström")
    group.add_argument("--size_y", type=_positive_float)
    group.add_argument("--size_z", type=_positive_float)
    group.add_argument(
        "--auto-box",
        type=int,
        metavar="N",
        help=(
            "instead of the six values, use the Nth site the pocket search "
            "finds in the receptor (1 = best). Omit to keep the six values "
            "mandatory."
        ),
    )
    group.add_argument(
        "--box-padding",
        type=_positive_float,
        default=4.0,
        help="Ångström of space added around the chosen site",
    )


def _resolve_box(args: argparse.Namespace, receptor_path: Path, min_size: float = 0.0):
    """A `GridBox` from the explicit values or from the pocket search.

    `min_size` is the engine's own floor for a ligand
    (`2 x radius + 1 A`, see `monte_carlo.rs`). A site is a *pocket*, and a
    pocket is routinely smaller than a ligand's bounding sphere: crambin's
    best site gives a 12.0 A y edge and ibuprofen needs 12.1 A, so `--auto-box 1`
    died deep in the engine with "axis y is 12.0 A but the ligand needs at
    least 12.1 A". That message is true and useless -- it does not say that the
    box came from a site, or that widening it is the fix. So the floor is
    applied here, where the reason is still known, and the widening is
    reported rather than performed silently.

    Raises `SystemExit` with a message a user can act on, rather than
    defaulting: a whole-protein box is enormous in memory and useless as a
    search region, so there is no safe fallback to invent.
    """
    from .core import GridBox

    explicit = [args.center_x, args.center_y, args.center_z, args.size_x, args.size_y, args.size_z]
    given = [v is not None for v in explicit]
    want_auto = getattr(args, "auto_box", None)

    if any(given) and not all(given):
        missing = [
            name
            for name, ok in zip(
                ("--center_x", "--center_y", "--center_z", "--size_x", "--size_y", "--size_z"),
                given,
            )
            if not ok
        ]
        raise SystemExit(
            f"error: the search box is half-specified; missing {', '.join(missing)}. "
            "Give all six, or none of them and use --auto-box N."
        )
    if all(given):
        if want_auto:
            raise SystemExit(
                "error: --auto-box and the six box values cannot both be given; "
                "they are two ways of answering the same question."
            )
        return GridBox.from_center_size(tuple(explicit[:3]), tuple(explicit[3:]))

    if want_auto is None:
        raise SystemExit(
            "error: no search box. Give --center_x..--size_z, or --auto-box N to "
            "take the Nth site the pocket search finds. Defaulting it from the "
            "receptor would be a trap: a whole-protein box is both enormous in "
            "memory and useless as a search region."
        )
    # `is None`, not `if not want_auto`: `--auto-box 0` is falsy, and testing it
    # that way sent a user who typed 0 to the "no search box" message, which
    # does not tell them that ranks start at 1.
    if want_auto < 1:
        raise SystemExit(
            f"error: --auto-box is 1-based, got {want_auto}. "
            "Run `odcli sites -r RECEPTOR` to see how many there are."
        )

    found = _find_sites(args.receptor, probe=getattr(args, "probe", None), verbose=True)
    if not found:
        raise SystemExit(
            f"error: no enclosed site found in {receptor_path.name}. "
            "Give the box by hand with --center_x..--size_z."
        )
    if want_auto > len(found):
        raise SystemExit(
            f"error: --auto-box {want_auto} but only {len(found)} site(s) were found "
            f"in {receptor_path.name}. Try --auto-box 1..{len(found)}."
        )
    centre, size = found[want_auto - 1].box_center_and_size(args.box_padding)
    if min_size > 0.0:
        too_small = [i for i in range(3) if size[i] < min_size]
        if too_small:
            size = tuple(max(size[i], min_size) for i in range(3))
            axes = ", ".join("xyz"[i] for i in too_small)
            print(
                f"widened the box on {axes} to {min_size:.1f} A: the site itself is "
                f"smaller than this ligand needs, and the engine refuses a box "
                f"narrower than 2x its radius plus 1 A",
                file=sys.stderr,
            )
    print(
        f"box from site {want_auto}: centre "
        f"({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}) "
        f"size ({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f})",
        file=sys.stderr,
    )
    return GridBox.from_center_size(centre, size)


def _find_sites(receptor_path: Path, probe=None, verbose: bool = False):
    """`Pocket` objects for a receptor file, best first.

    The padding is deliberately *not* applied here. The site and the box built
    from it are different things and the `sites` command prints both; baking the
    padding in here meant the same number came out under two headings.
    """
    from .workbench import MoleculeView
    from .workbench import pockets as P

    view = MoleculeView.from_pdbqt(
        receptor_path, receptor_path.name, (0.6, 0.7, 0.9), 0.30, role="receptor"
    )
    started = time.perf_counter()
    kwargs = {"probe": probe} if probe else {}
    found = P.find_pockets(
        view.coords, view.elements, residues=view.residue_labels(), **kwargs
    )
    if verbose:
        print(
            f"pockets: {len(found)} site(s) in "
            f"{(time.perf_counter() - started) * 1000:.0f} ms",
            file=sys.stderr,
        )
    return found


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

    # --- sites ------------------------------------------------------------
    p = sub.add_parser(
        "sites",
        help="list candidate binding sites in a receptor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--receptor", required=True, type=Path, help="receptor .pdbqt")
    p.add_argument(
        "--probe",
        type=_positive_float,
        help="solvent probe radius, Ångström (default 1.4, a water molecule)",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_sites)

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
    from .core import Receptor, scoring_descriptions

    box_ = _resolve_box(args, args.receptor)
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


def _cmd_sites(args: argparse.Namespace) -> int:
    """Print the candidate sites, so the box can be chosen before docking.

    Exists because `--auto-box 3` is otherwise a number with nothing behind it
    visible to the user. Reporting the residues lining each site is the part
    that makes the list checkable by eye.
    """
    from .workbench import pockets as P

    found = _find_sites(args.receptor, probe=args.probe)
    padding = getattr(args, "box_padding", 4.0) or 4.0
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "rank": i + 1,
                        "kind": p.kind,
                        "center": [float(v) for v in p.center],
                        "size": [float(v) for v in p.size],
                        "box_center": [float(v) for v in p.box_center_and_size(padding)[0]],
                        "box_size": [float(v) for v in p.box_center_and_size(padding)[1]],
                        "voxels": p.voxels,
                        "burial": p.burial,
                        "lining": [{"residue": r, "atoms": n} for r, n in p.lining],
                    }
                    for i, p in enumerate(found)
                ],
                indent=2,
            )
        )
        return 0

    if not found:
        print(f"no enclosed site found in {args.receptor.name}")
        return 0
    print(f"{len(found)} candidate site(s) in {args.receptor.name}\n")
    sealed = [p for p in found if p.kind == "cavity"]
    for i, p in enumerate(found):
        lining = ", ".join(f"{r} ({n})" for r, n in p.lining[:8]) or "-"
        centre, size = p.box_center_and_size(padding)
        print(f"{i + 1}. {P.kind_label(p.kind)}  score {p.rank_score:.2f}  "
              f"{p.voxels} grid points  buried on {p.burial:.1f}/3 axes")
        print(f"   site  {p.size[0]:5.1f} x {p.size[1]:5.1f} x {p.size[2]:5.1f} A at "
              f"({p.center[0]:7.2f}, {p.center[1]:7.2f}, {p.center[2]:7.2f})")
        print(f"   box   {size[0]:5.1f} x {size[1]:5.1f} x {size[2]:5.1f} A at "
              f"({centre[0]:7.2f}, {centre[1]:7.2f}, {centre[2]:7.2f})  "
              f"(+{padding:g} A padding each side)")
        print(f"   lined by {lining}\n")

    # "No sealed cavity" is not a statement about the protein, it is a
    # statement about the probe. T4 lysozyme L99A has a cavity built into it on
    # purpose and reports zero at 1.4 A, because a 1.4 A probe inflates every
    # atom enough to fill a 100 A^3 hole. Saying so here turns a null result
    # into something the user can act on instead of a dead end.
    if not sealed:
        from .workbench import MoleculeView

        view = MoleculeView.from_pdbqt(
            args.receptor, args.receptor.name, (0.6, 0.7, 0.9), 0.30, role="receptor"
        )
        sweep = P.cavity_sensitivity(view.coords, view.elements)
        got = ", ".join(f"{p:g} A -> {n}" for p, n in sweep)
        print(f"No sealed cavity at the default {P.DEFAULT_PROBE:g} A probe. "
              f"Against a smaller probe: {got}")
        if any(n for _, n in sweep):
            print("A smaller probe finds them, and also turns surface grooves")
            print("into spurious pockets, so it is not the default. Run again")
            print("with --probe if you want to look at the smaller-radius answer.")
        else:
            print("No probe in that range finds one either, so this structure")
            print("most likely has no enclosed space at all -- only grooves.")
        print()

    print("A site is enclosed space, not a binding site. Nothing here knows which")
    print("one your ligand wants; --auto-box N takes the Nth from this list.")
    return 0


def _cmd_dock(args: argparse.Namespace) -> int:
    from .core import Receptor, dock, load_ligand, load_receptor

    ligands = _collect_ligands(args.ligand)
    if not ligands:
        print(f"error: no ligands found at {args.ligand}", file=sys.stderr)
        return 2

    # The ligands are collected before the box is resolved, because the box a
    # site produces may be too small for the ligand and the floor is the
    # ligand's own. Reading the first one here is cheap and the same file is
    # read again in the loop below; a pocket search costs far more than one
    # extra parse.
    #
    # Rounded *up* to 0.1 A, because the engine and this layer do not compute
    # the ligand's radius from bit-identical inputs and the two disagree in the
    # last few digits. Clamping to the Python value exactly gave 12.071976 A
    # where the engine wanted marginally more, and the run failed with an error
    # that said the two numbers were equal. 0.1 A is under a third of the 0.375 A
    # grid spacing, so it costs less than one voxel of search volume.
    min_size = 0.0
    try:
        min_size = math.ceil((2.0 * load_ligand(ligands[0]).radius + 1.0) * 10.0) / 10.0
    except Exception as exc:
        print(
            f"warning: could not size the box from {ligands[0].name} ({exc}); "
            "an auto box will not be widened for it",
            file=sys.stderr,
        )

    box_ = _resolve_box(args, args.receptor, min_size=min_size)
    receptor = load_receptor(args.receptor)
    print(f"receptor: {receptor.num_atoms} atoms", file=sys.stderr)

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
