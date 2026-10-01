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
    p.add_argument(
        "--keep-chain",
        default=None,
        metavar="ID",
        help=(
            "keep this PDB chain instead of the largest component; a chain that "
            "is not in the file is an error, not a silent fallback"
        ),
    )
    p.add_argument(
        "--report",
        action="store_true",
        help=(
            "print the full preparation report: chains in and kept, components, "
            "atoms dropped and why. Without it, a line is printed only when "
            "something was actually dropped."
        ),
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
        "-e",
        "--exhaustiveness",
        type=int,
        default=None,
        help="independent search walks; default follows the box's volume, "
        "because the same walks spread through a bigger box find less "
        f"(see `exhaustiveness_for_box`)",
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


def _chain_list(ids) -> str:
    """``B`` or ``A, B``, with a blank chain id spelled out."""
    return ", ".join(c if c != " " else "(blank)" for c in ids)


def _fail(message: str, code: int = 1) -> int:
    """Say what went wrong on stderr, and return the exit code.

    One function rather than a `print(..., file=sys.stderr)` per call site,
    because the stream and the code are the two things a script reads and they
    had drifted apart in both directions: `odcli workbench` put its diagnostic
    on stdout (the workbench's own `launch`, not this file), and `split`
    exited 1 having written a success-shaped line to stdout and nothing at all
    to stderr. A diagnostic on stdout is indistinguishable from a result, and
    a script that captures only stderr then sees nothing whatsoever.

    The `error: ` prefix is added here rather than at the call sites so the two
    cannot be spelled differently either.
    """
    print(f"error: {message}", file=sys.stderr)
    return code


def _prep_report_line(report) -> str:
    """One line, for when something was dropped and nobody asked for detail.

    Deliberately says *which chain* went missing. The 1HVR benchmark measured a
    two-chain structure against a 621-atom single chain for a week because a
    warning that counted fragments and atoms never once said a chain was gone.
    """
    bits = [
        f"kept {report.chains_kept} of {report.chains_in} chains"
        f" ({_chain_list(report.chain_ids_kept)})",
        f"{report.atoms_kept} of {report.atoms_in} atoms",
    ]
    if report.atoms_dropped:
        bits.append(
            f"{report.atoms_dropped} dropped ({report.water_atoms_removed} water, "
            f"{report.ion_atoms_removed} ion, {report.other_atoms_removed} other)"
        )
    if report.chain_ids_dropped:
        bits.append(f"dropped chain(s) {_chain_list(report.chain_ids_dropped)}")
    return "; ".join(bits) + "; --report for the full breakdown"


def _prep_report_lines(report) -> list[str]:
    """The full breakdown, in the two-column style `prep-ligand` already uses."""
    rows = [
        ("source", report.source),
        ("components", f"{report.fragments_kept} of {report.fragments_in} kept "
                       f"({report.selection})"),
        ("chains", f"{report.chains_kept} of {report.chains_in} kept: "
                   f"{_chain_list(report.chain_ids_kept)}"),
        ("chains dropped", _chain_list(report.chain_ids_dropped) or "none"),
        ("atoms", f"{report.atoms_kept} of {report.atoms_in} kept, "
                  f"{report.atoms_dropped} dropped"),
        ("removed", f"{report.water_atoms_removed} water, "
                    f"{report.ion_atoms_removed} ion, "
                    f"{report.other_atoms_removed} other"),
        ("polar hydrogens", f"{report.polar_hydrogens_added} added"),
        ("components equal chains", "yes" if report.fragments_equal_chains else "no"),
    ]
    width = max(len(label) for label, _ in rows)
    return [f"  {label:<{width}}  {value}" for label, value in rows]


#: Matches the component-loss warning, and only that one, so `prep-receptor` can
#: print the same fact in its own format instead of both. Every other warning
#: still reaches the terminal: the "dropped a polar hydrogen" one in particular
#: is per-atom and appears in no report, and silencing it would be hiding data.
#: It is matched by prefix because `warnings` compares this regex against the
#: start of the message.
_COMPONENT_WARNING_RE = r".*: kept .*"


def _cmd_prep_receptor(args: argparse.Namespace) -> int:
    import warnings

    from .prep import prepare_receptor_with_report

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_COMPONENT_WARNING_RE)
        text, report = prepare_receptor_with_report(
            args.receptor,
            keep_waters=args.keep_waters,
            keep_heterogens=not args.drop_heterogens,
            keep_chain=args.keep_chain,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    n_atoms = sum(1 for line in text.splitlines() if line.startswith(("ATOM", "HETATM")))
    print(f"wrote {args.output} ({n_atoms} atoms)")
    # A preparation step that always speaks is a preparation step people learn
    # to skip, so the loss is reported when there is one and not otherwise.
    # `--report` is an explicit request and is always answered.
    if args.report:
        for line in _prep_report_lines(report):
            print(line)
    elif report.is_lossy:
        print(f"  {_prep_report_line(report)}")
    return 0


def _cmd_prep_ligand(args: argparse.Namespace) -> int:
    from .core import typed_ligand_from_tables
    from .prep import prepare_ligand

    elements, charges, coords, bonds, names = prepare_ligand(
        args.ligand, keep_hydrogens=args.keep_hydrogens
    )
    # The same round trip `load_ligand` performs, and for the same reason: the
    # tables carry the AutoDock types in the atom names and `from_arrays` cannot
    # read them. This used to build the ligand untyped and print what came out,
    # which is how the defect was visible in the CLI's own output -- the classes
    # printed here are now the classes the engine will score with, and the print
    # is kept precisely because it is the cheapest way for a user to see them.
    ligand, pdbqt_text = typed_ligand_from_tables(
        elements, charges, coords, bonds, names)

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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pdbqt_text, encoding="utf-8")
        print(f"  wrote            {args.output}")
    return 0


def _scoring_description(name: str) -> str:
    """The engine's own description of one scoring function, looked up by name.

    `scoring_descriptions()` is a list of ``"<name>: <terms>"`` strings in the
    engine's order, and that order is not a contract -- it is whatever
    `scoring_functions()` happens to return. Indexing it by position is how
    `rec-grid --scoring vinardo` came to print *nothing at all* rather than
    the wrong thing: the line was guarded by `if args.scoring == "vina"` and
    no other name had a line to print. So a vinardo run wrote 40 `.map` files
    with names identical to a vina run's, 5 of which differed in content and 35
    of which were byte-identical, and the output said nothing about which
    scoring function produced them.

    Raised rather than defaulted: a scoring function the engine does not
    offer is `argparse`'s business (it already restricts the choices), so
    reaching here means the two lists have gone out of step and saying so beats
    printing a plausible wrong sentence.
    """
    from .core import scoring_descriptions

    for description in scoring_descriptions():
        if description.split(":", 1)[0].strip() == name:
            return description
    raise ValueError(
        f"the engine offers no scoring function called {name!r}; "
        f"it has {', '.join(d.split(':', 1)[0].strip() for d in scoring_descriptions())}"
    )


def _cmd_rec_grid(args: argparse.Namespace) -> int:
    from .core import Receptor

    box_ = _resolve_box(args, args.receptor)
    receptor = Receptor.from_pdbqt(args.receptor)
    est = receptor.estimate_memory_mb(box_, args.spacing)
    print(f"receptor: {receptor.num_atoms} atoms, "
          f"{receptor.num_polar_hydrogens} polar hydrogens")
    print(f"box: centre {box_.center}, size {box_.size}")
    print(f"estimated maps: {est:.1f} MB")
    # Named for whichever function was asked for. This line used to be inside
    # `if args.scoring == "vina"`, so a vinardo run printed no scoring line at
    # all and a directory of .map files could not be attributed to anything.
    print(f"scoring: {_scoring_description(args.scoring)}")

    start = time.time()
    maps = receptor.precalculate(box_, args.scoring, args.spacing)
    elapsed = time.time() - start
    print(f"precalculated {maps.num_points} points in {elapsed:.2f} s "
          f"({maps.memory_mb:.1f} MB actual)")

    args.output.mkdir(parents=True, exist_ok=True)
    maps.write_map_files(args.output)
    print(f"wrote AutoDock .map files to {args.output}")
    return 0


def _box_volume(size) -> float:
    """Volume of a ``(x, y, z)`` size tuple, Å³."""
    return float(size[0]) * float(size[1]) * float(size[2])


def _budget_report(pocket, padding: float) -> dict:
    """What a size budget would cost this site's box, as a plain dict.

    The budget is not applied to the box `sites` prints -- that one is what
    `--auto-box N` will actually build, and quietly handing back a capped box
    here would make the list describe a search that never happens. It is
    reported, so that a site whose box is too big to be a sensible search is
    visible as such instead of being an ordinary-looking 39 A edge. An empty
    `note` is a positive claim, not an absence: it means nothing was capped and
    the box still holds every point of the site.

    The ceiling is passed in rather than left to the default so the report can
    name the number it used. A caller who tightens the ceiling then knows
    which one produced the coverage figure, instead of having to assume.
    """
    from .workbench.pockets import DEFAULT_BOX_MAX_SIDE

    budget = pocket.box_with_budget(padding, max_side=DEFAULT_BOX_MAX_SIDE)
    return {
        "max_side": float(DEFAULT_BOX_MAX_SIDE),
        "capped": bool(budget.capped),
        "coverage": budget.coverage,
        "volume": budget.volume,
        "requested_volume": budget.requested_volume,
        "note": budget.note,
    }


def _cmd_sites(args: argparse.Namespace) -> int:
    """Print the candidate sites, so the box can be chosen before docking.

    Exists because `--auto-box 3` is otherwise a number with nothing behind it
    visible to the user. Reporting the residues lining each site is the part
    that makes the list checkable by eye.

    A volume is reported for both the site and its box, because the two
    disagree by a lot and the difference is the whole point: `Pocket.volume` is
    `voxels * spacing ** 3`, the space actually enclosed, while the box is the
    bounding cuboid around it. The HIV protease site in 1HVR is the case the
    library documents -- an 8008 A³ box around 544 A³ of real space -- and a
    list that gave only the box number would make a winding cleft look like the
    best site in the file. `spacing` comes out with the volume because a volume
    in A³ is not interpretable without the pitch it was measured on, and a
    caller who searched finer than the default has to be told so.
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
                        "volume": float(p.volume),
                        "spacing": float(p.spacing),
                        "box_center": [float(v) for v in p.box_center_and_size(padding)[0]],
                        "box_size": [float(v) for v in p.box_center_and_size(padding)[1]],
                        "box_volume": _box_volume(p.box_center_and_size(padding)[1]),
                        "box_budget": _budget_report(p, padding),
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
              f"({p.center[0]:7.2f}, {p.center[1]:7.2f}, {p.center[2]:7.2f})  "
              f"{p.volume:.1f} A3 at {p.spacing:g} A pitch")
        budget = _budget_report(p, padding)
        cap = f", {_box_volume(size):.0f} A3"
        if budget["note"]:
            cap += f"; {budget['note']}"
        print(f"   box   {size[0]:5.1f} x {size[1]:5.1f} x {size[2]:5.1f} A at "
              f"({centre[0]:7.2f}, {centre[1]:7.2f}, {centre[2]:7.2f})  "
              f"(+{padding:g} A padding each side{cap})")
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
    from .core import (
        Receptor,
        dock,
        exhaustiveness_for_box,
        load_ligand,
        load_receptor,
    )

    ligands = _collect_ligands(args.ligand)
    if not ligands:
        return _fail(f"no ligands found at {args.ligand}", 2)

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

    # Resolved here rather than in argparse, because it needs the box and the
    # box is not known until `_resolve_box` has run. Saying so out loud matters:
    # exhaustiveness used to be a constant in the help text, and a number that
    # changes with the box is a number the reader has to be told about, or they
    # will assume a fixed 8 and wonder why two runs of "the same" search
    # disagree.
    if args.exhaustiveness is None:
        exhaustiveness = exhaustiveness_for_box(box_.size)
        sides = " x ".join(f"{float(v):.0f}" for v in box_.size)
        print(
            f"exhaustiveness: {exhaustiveness} (default for a {sides} A box; "
            f"pass -e to choose your own)",
            file=sys.stderr,
        )
    else:
        exhaustiveness = args.exhaustiveness

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
            exhaustiveness=exhaustiveness,
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
            # The table itself is `DockingResult.summary()`, which prints the
            # column name `affinity` and the number and nothing else. The unit
            # was documented on the property that produces it and appeared
            # nowhere in the output, so a reader of the table -- or of the JSON,
            # which has the same gap -- had no way to tell kcal/mol from any
            # other score without going and looking. Stated here rather than by
            # rewriting the header, so the library and the CLI cannot each grow
            # their own idea of what that column is.
            print("affinity and inter are kcal/mol, rmsd is A from the best pose")
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
    if not written:
        # Used to print "wrote 0 pose file(s) to <dir>" on stdout and return 1:
        # a non-zero exit whose only output looked like a successful run, with
        # stderr empty. A script capturing stderr saw nothing at all, and a
        # script reading stdout saw a count of zero written into a directory
        # that was indeed empty -- true, and no help whatsoever.
        return _fail(
            f"no poses found in {args.input.name}, so nothing was written to "
            f"{args.output_dir}. A pose file needs at least one "
            "'MODEL ... ENDMDL' block; a single-model ligand PDBQT has none."
        )
    print(f"wrote {len(written)} pose file(s) to {args.output_dir}")
    return 0


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
    # No preflight of its own, deliberately. There used to be an
    # `except ImportError` here, and it could never run: `opendocking.workbench`
    # imports only the standard library plus numpy at module scope and defers
    # the GUI import to a call inside `launch`, so `from .workbench import
    # launch` cannot raise. The two entry points therefore produced the *same*
    # message from the same place, and this copy of it -- the one carrying the
    # ImportError's own text and a `pip install` hint -- was dead weight that
    # read as if it were the handler. `launch` now owns the check, the message,
    # the stream and the exit code, for `odcli` and `odgui` alike.
    from .workbench import launch

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
