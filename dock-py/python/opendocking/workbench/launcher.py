"""``odgui`` -- the standalone entry point for the 3-D workbench.

Separate from the ``odcli`` command-line interface on purpose. The package ships
exactly two console scripts, ``odgui`` for the viewer and ``odcli`` for the
full CLI, so a user who reaches for one by name gets exactly that and nothing
else. Neither depends on the other being present.

Nothing here imports Qt or moderngl at module scope. Importing this module has
to stay cheap and must work on a machine with no display stack, so that
``odgui --help`` can explain how to install the GUI extras rather than dying
with a traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]

#: What ``odgui`` needs beyond the engine itself. Kept in one place so the
#: error message and the packaging metadata cannot drift apart.
_GUI_REQUIREMENTS = "PyQt6>=6.4 moderngl>=5.8 numpy-stl>=2.0"

_BANNER = "Open Docking Workbench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odgui",
        description="Launch the Open Docking 3-D workbench.",
        epilog=(
            "With no arguments the window opens empty: load a receptor and a "
            "ligand from the File menu, or pass them below. Docked poses can "
            "be passed with --poses and are then browsable by energy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r", "--receptor", type=Path, help="receptor .pdbqt to open on start-up"
    )
    parser.add_argument(
        "-l", "--ligand", type=Path, help="ligand .pdbqt to open on start-up"
    )
    parser.add_argument(
        "-p", "--poses", type=Path, help="docked poses .pdbqt to browse on start-up"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the GUI can start, then exit without opening a window",
    )
    return parser


def _describe_missing(exc: BaseException) -> str:
    """Turn an ImportError into the one line that actually helps."""
    name = getattr(exc, "name", None) or str(exc)
    return (
        f"error: the workbench needs PyQt6 and moderngl, and `{name}` could not "
        f"be imported ({exc}).\n"
        f"       install them with:\n"
        f"           pip install {_GUI_REQUIREMENTS}"
    )


def _preflight() -> tuple[object | None, str | None]:
    """Import the GUI stack, reporting the first missing piece.

    Returns ``(launch, None)`` on success or ``(None, message)`` on failure.
    Checked in one place so the window can never open with a half-initialised
    renderer and a blank viewport.
    """
    try:
        from . import launch  # noqa: F401  (import is the test)
    except ImportError as exc:
        return None, _describe_missing(exc)
    return launch, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    launch, problem = _preflight()
    if launch is None:
        print(problem, file=sys.stderr)
        return 3

    for label, path in (
        ("receptor", args.receptor),
        ("ligand", args.ligand),
        ("poses", args.poses),
    ):
        if path is not None and not path.is_file():
            print(f"error: {label} file not found: {path}", file=sys.stderr)
            return 2

    if args.check:
        import opendocking

        print(f"{_BANNER}: GUI stack OK")
        print(f"  engine  {opendocking.engine_version()}")
        gpu = opendocking.gpu_status()
        print(
            "  gpu     "
            + (
                "compiled in, adapter available"
                if gpu.get("available")
                else ("compiled in, no adapter" if gpu.get("compiled") else "not compiled in")
            )
        )
        return 0

    if not any((args.receptor, args.ligand, args.poses)):
        print(
            f"{_BANNER}\n"
            "No structure given; the window opens empty.\n"
            "  receptor/ligand : File -> Open structure, or -r / -l\n"
            "  docked poses    : odgui -p poses.pdbqt\n"
            "Run 'odgui --help' for all options.",
            file=sys.stderr,
        )

    return launch(args.receptor, args.ligand, args.poses)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
