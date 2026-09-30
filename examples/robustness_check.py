"""Malformed input must produce a clear error, never a crash.

This matters more than usual here: the release profile is `panic = "abort"`, so
a Rust panic does not raise a Python exception — it kills the host interpreter
with exit code 0xC0000409. One `unwrap()` in the wrong place therefore takes
the whole process down, and the user's program with it.

Every case below is something a real file can contain: a truncated download, a
PDBQT from a different tool, a zero-volume atom, a box smaller than the ligand.
The requirement is the same in every case — an exception the caller can catch
and print, never a dead process.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import opendocking  # noqa: E402

GOOD_RECEPTOR = None
GOOD_LIGAND = None
failures: list[str] = []


def expect_error(label: str, fn, exc=(Exception,)) -> None:
    """Run `fn` and require it to raise something we can catch and report."""
    try:
        fn()
    except exc as e:  # noqa: BLE001 - that is the point
        kind = type(e).__name__
        msg = str(e).splitlines()[0] if str(e) else "(no message)"
        print(f"  ok   {label:<34} -> {kind}: {msg[:70]}")
        return
    except BaseException as e:  # noqa: BLE001
        failures.append(label)
        print(f"  FAIL {label:<34} -> uncatchable {type(e).__name__}: {e}")
        return
    failures.append(label)
    print(f"  FAIL {label:<34} -> returned normally, should have raised")


def run_in_subprocess(label: str, code: str) -> None:
    """Run a snippet in a child interpreter.

    A Rust panic with `panic = "abort"` never reaches Python, so it can only be
    detected by watching the child die. This is the only way to tell "raised a
    ValueError" apart from "killed the process".
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        path = fh.name
    proc = subprocess.run(
        [sys.executable, path], capture_output=True, text=True, timeout=180
    )
    Path(path).unlink(missing_ok=True)
    # A child that could not even import the package exits 1, which the abort
    # check below would happily call a pass. That makes the test prove nothing,
    # so the import errors are rejected explicitly.
    if "ModuleNotFoundError" in proc.stderr or "NameError" in proc.stderr:
        failures.append(label)
        detail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(
            f"  FAIL {label:<34} -> the child could not import the package "
            f"({detail[:50]}); nothing was actually tested"
        )
        return
    # 0xC0000409 is STATUS_STACK_BUFFER_OVERRUN, what abort() looks like on
    # Windows; on other platforms it is SIGABRT.
    aborted = proc.returncode not in (0, 1)
    if aborted:
        failures.append(label)
        print(
            f"  FAIL {label:<34} -> child died with code {proc.returncode} "
            "(this is what a Rust panic looks like)"
        )
    else:
        detail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(f"  ok   {label:<34} -> exit {proc.returncode}, {detail[:60]}")


def main() -> int:
    global GOOD_RECEPTOR, GOOD_LIGAND
    here = Path(__file__).resolve().parent
    GOOD_RECEPTOR = here / "rec_prep.pdbqt"
    GOOD_LIGAND = here / "ligands" / "ibuprofen.pdbqt"
    if not GOOD_RECEPTOR.exists() or not GOOD_LIGAND.exists():
        print("run from the examples directory after preparing the sample data")
        return 2

    receptor = opendocking.Receptor.from_pdbqt(GOOD_RECEPTOR)
    ligand = opendocking.Ligand.from_pdbqt(GOOD_LIGAND)
    box = opendocking.GridBox.from_center_size((0.0, 0.0, 0.0), (20.0, 20.0, 20.0))
    maps = receptor.precalculate(box, scoring="vina", spacing=0.375)

    print("PDBQT parsing")
    expect_error("empty file", lambda: opendocking.Receptor.from_pdbqt_str(""))
    expect_error("header only", lambda: opendocking.Ligand.from_pdbqt_str("REMARK nothing\n"))
    expect_error(
        "truncated atom line",
        lambda: opendocking.Ligand.from_pdbqt_str("ROOT\nATOM      1  C1  UNL     1  \n"),
    )
    expect_error(
        "two atoms on top of each other",
        lambda: opendocking.Ligand.from_pdbqt_str(
            "ROOT\n"
            "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
            "ATOM      2  C2  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
            "ENDROOT\n"
        ),
    )
    # An unrecognised type is deliberately *accepted* — refusing the file would
    # make the engine unusable with tools that emit types AutoDock never defined. The
    # requirement is that it be visible rather than a silent downgrade.
    exotic = opendocking.Receptor.from_pdbqt_str(
        "ATOM      1  C1  REC     1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        "ATOM      2  X1  REC     1       3.000   0.000   0.000  1.00  0.00     0.000 ZZ\n"
    )
    if exotic.unknown_atom_types == 1:
        print("  ok   unrecognised atom type          -> accepted and counted")
    else:
        failures.append("unknown atom type not reported")
        print(
            f"  FAIL unrecognised atom type          -> reported "
            f"{exotic.unknown_atom_types}, expected 1"
        )
    if receptor.unknown_atom_types == 0:
        print("  ok   the real receptor has no unknown types")
    else:
        print(
            f"  note the sample receptor has {receptor.unknown_atom_types} "
            "unknown types"
        )

    print("\nbox and parameters")
    expect_error(
        "inverted box", lambda: opendocking.GridBox.from_center_size((0, 0, 0), (-5, 5, 5))
    )
    expect_error("zero-size box", lambda: opendocking.GridBox.from_center_size((0, 0, 0), (0, 5, 5)))
    expect_error(
        "nan corner", lambda: opendocking.GridBox.from_center_size((0, 0, 0), (float("nan"), 5, 5))
    )
    expect_error(
        "inf corner", lambda: opendocking.GridBox.from_center_size((0, 0, 0), (float("inf"), 5, 5))
    )
    # Zero spacing is a documented convenience: `Receptor::precalculate` falls
    # back to the default. Negative is not, and must be rejected.
    fallback = receptor.precalculate(box, scoring="vina", spacing=0.0)
    if abs(fallback.spacing - 0.375) < 1e-12:
        print("  ok   spacing 0 falls back to the default")
    else:
        failures.append("spacing 0 fallback")
        print(f"  FAIL spacing 0 -> got {fallback.spacing}, expected 0.375")
    expect_error(
        "negative spacing",
        lambda: receptor.precalculate(box, scoring="vina", spacing=-1.0),
    )
    expect_error(
        "absurd spacing",
        lambda: receptor.precalculate(box, scoring="vina", spacing=50.0),
    )
    # This one used to abort the interpreter with a capacity overflow: a
    # spacing of 1e-9 passes the (0, 1] range check but asks for a 2e10-point
    # axis, and the release profile is panic = "abort".
    expect_error(
        "catastrophic spacing (used to abort)",
        lambda: receptor.precalculate(box, scoring="vina", spacing=1e-9),
    )
    expect_error(
        "unknown scoring function",
        lambda: receptor.precalculate(box, scoring="not-a-function", spacing=0.375),
    )

    print("\nconformation shapes")
    expect_error(
        "wrong dof count",
        lambda: opendocking.score_conformation(ligand, maps, [0.0] * (ligand.num_dof + 1)),
    )
    expect_error(
        "short batch",
        lambda: opendocking.evaluate_conformations(ligand, maps, np.zeros((4, 2))),
    )
    expect_error(
        "one-dimensional batch",
        lambda: opendocking.evaluate_conformations(ligand, maps, np.zeros(ligand.num_dof)),
    )
    expect_error(
        "nan in a conformation",
        lambda: opendocking.score_conformation(
            ligand, maps, np.full(ligand.num_dof, float("nan"))
        ),
    )
    expect_error(
        "conformation_coordinates wrong length",
        lambda: opendocking.conformation_coordinates(ligand, np.zeros(3)),
    )
    expect_error(
        "pose index out of range",
        lambda: opendocking.dock(ligand, maps, exhaustiveness=1, num_modes=1, seed=1).pose_coords(99),
    )
    expect_error(
        "box too small for the ligand",
        lambda: opendocking.dock(
            ligand,
            receptor.precalculate(
                opendocking.GridBox.from_center_size((0, 0, 0), (12.0, 12.0, 12.0)), "vina", 0.5
            ),
            exhaustiveness=1,
            num_modes=1,
            seed=1,
        ),
    )

    print("\nnumeric extremes must stay finite")
    wild = np.full((32, ligand.num_dof), 1e6)
    energies = opendocking.evaluate_conformations(ligand, maps, wild)
    if np.all(np.isfinite(energies)):
        print("  ok   energies at 1e6 stay finite")
    else:
        failures.append("non-finite energies")
        print(f"  FAIL energies at 1e6: {energies[:3]}")
    e, g = opendocking.score_conformation(ligand, maps, wild[0])
    if math.isfinite(e) and np.all(np.isfinite(g)):
        print("  ok   single-conformation energy and gradient stay finite")
    else:
        failures.append("non-finite single score")
        print(f"  FAIL single score at 1e6: E={e}, |g|inf={np.max(np.abs(g))}")

    print("\nprocess death (a Rust panic would abort the interpreter)")
    run_in_subprocess(
        "panic on garbage pdbqt",
        "import opendocking; opendocking.Ligand.from_pdbqt_str('\\x00\\xff not a pdbqt at all\\n')",
    )
    run_in_subprocess(
        "panic on one-atom molecule",
        "import opendocking; opendocking.Ligand.from_pdbqt_str("
        "'ROOT\\nATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\\nENDROOT\\n')",
    )
    run_in_subprocess(
        "panic on self-intersecting torsions",
        "import opendocking; opendocking.Ligand.from_pdbqt_str("
        "'ROOT\\n"
        "ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\\n"
        "ATOM      2  C2  UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C\\n"
        "ENDROOT\\n"
        "BRANCH   2   3\\n"
        "ATOM      3  C3  UNL     1       3.000   1.300   0.000  1.00  0.00     0.000 C\\n"
        "ENDBRANCH\\n"
        "BRANCH   1   4\\n"
        "ATOM      4  C4  UNL     1       4.500   1.300   0.000  1.00  0.00     0.000 C\\n"
        "ENDBRANCH\\n')",
    )

    print()
    if failures:
        print(f"{len(failures)} case(s) did not behave correctly: {failures}")
        return 1
    print("all malformed-input cases produced a catchable error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
