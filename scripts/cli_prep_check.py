"""Check that `odcli prep-receptor` says what preparation dropped -- and only when it did.

`prepare_receptor_with_report` exists because a warning is a string that GUI
code and CLI code routinely drop on the floor. This is the CLI half of that:
the command reads the report instead of the warning, and the report is only
worth having if something actually reads it.

Three output modes, and the third is the one that is easy to get wrong:

* **A loss is reported in one line**, naming the chain that went missing. The
  1HVR benchmark searched a two-chain 1826-atom structure while docking against
  a 621-atom single chain for a week, and the warning that would have said so
  was suppressed by the benchmark. A warning nobody reads is not a report.
* **`--report` prints the whole breakdown**, on request, in the same
  two-column style `prep-ligand` already uses.
* **A preparation that lost nothing prints nothing about it.** A step that
  always speaks is a step people learn to skip, so the check for silence is as
  much a part of the contract as the check for the message.

The real console script is run as a subprocess, the way a person would type it.
That has a trap in it, and this script checks for the trap rather than trusting
it: `odcli` imports whichever `opendocking` comes first on `sys.path`, which by
default is the **installed** wheel, not the source tree being edited. So the
subprocess is given `PYTHONPATH` pointing at `dock-py/python`, and the check
proves that worked by looking for a flag that exists only in the source tree --
and proves the check is sensitive by confirming the same flag is *absent*
without it.

Fixtures are imported from `scripts/prep_check.py` rather than rebuilt here.
Two copies of the geometry code would drift, and a drifted fixture is a check
that quietly stopped testing anything.

Run:  python scripts/cli_prep_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dock-py" / "python"
CRAMBIN = ROOT / "examples" / "1crn_receptor.pdb"

# The check itself prints report text, which can contain a non-ASCII character
# on a console that is not UTF-8. A character that cannot be encoded must not
# turn a check run into a traceback.
if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, str(ROOT / "scripts"))
try:  # noqa: SIM105
    import prep_check as F  # noqa: E402
except ImportError:  # pragma: no cover
    F = None

from opendocking.prep import prepare_receptor_with_report  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


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
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


#: The console script, and the module fallback for a checkout without one
#: installed. The console script is preferred because it is what a user types.
ODCLI = shutil.which("odcli")


def odcli_argv(args) -> list[str]:
    if ODCLI:
        return [ODCLI, *args]
    return [sys.executable, "-m", "opendocking.cli", *args]


def run(args, *, source_tree: bool = True) -> subprocess.CompletedProcess:
    """Run the real entry point in a subprocess.

    `source_tree` puts `dock-py/python` first on the child's path. It has to be
    done per child rather than assumed from the parent, because `odcli` is an
    installed script whose imports resolve in *its* process.
    """
    env = dict(os.environ)
    if source_tree:
        env["PYTHONPATH"] = str(SRC)
    else:
        env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        odcli_argv(args), capture_output=True, cwd=str(ROOT), env=env,
        # Explicitly UTF-8, not the inherited locale. `odcli` reconfigures its
        # own streams to UTF-8, but `text=True` on its own decodes with the
        # *parent's* code page, which on a GBK console turns every `Å` in a
        # report into a different character. Nothing below matches on one
        # today; the day something does, it would fail for a reason that has
        # nothing to do with the code under test.
        encoding="utf-8", errors="replace",
    )


def lines(result) -> list[str]:
    return [l for l in result.stdout.splitlines() if l.strip()]


def report_for(path: Path, **kwargs):
    """The same preparation, through the library, for the numbers to compare with."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare_receptor_with_report(path, **kwargs)


def prep(path: Path, out: Path, *flags, source_tree: bool = True):
    """Run `odcli prep-receptor` and hand back the result and the text it wrote."""
    result = run(["prep-receptor", "-r", str(path), "-o", str(out), *flags],
                 source_tree=source_tree)
    text = out.read_text(encoding="utf-8") if out.exists() else ""
    return result, text


def main() -> int:
    global CHECKS, FAILURES
    print(f"odcli: {ODCLI or '(not installed; falling back to python -m opendocking.cli)'}")
    if F is None:
        check("scripts/prep_check.py is importable for its fixtures", False,
              "the fixture builders live there and are not duplicated here")
        return finish()
    check("scripts/prep_check.py is importable for its fixtures", True,
          "reusing its geometry builders rather than copying them")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        geom_a = F.build_geometry("NCCO", 0xA01)
        geom_b = F.build_geometry("NCCCO", 0xB02, offset=(F.CHAIN_SEPARATION, 0.0, 0.0))
        rec_a = F.chain_records(geom_a, "A")
        rec_b = F.chain_records(geom_b, "B")

        # -------------------------------------------------------------------
        section("the subprocess is running the code under test")

        with_tree = run(["prep-receptor", "--help"])
        without_tree = run(["prep-receptor", "--help"], source_tree=False)
        check("odcli is a real console script and answers --help",
              with_tree.returncode == 0 and "usage: odcli" in with_tree.stdout,
              f"rc={with_tree.returncode}")
        check("the child process is importing the source tree, not the wheel",
              "--report" in with_tree.stdout and "--keep-chain" in with_tree.stdout,
              "flags that exist only in the edited source are visible to the child")
        # The intent is to prove the child really imports the source tree. It
        # used to do that by *contrast*: the installed copy predated the edited
        # flags, so `--report` was absent without PYTHONPATH and its presence
        # with it was evidence rather than decoration. That contrast is gone --
        # `pip install --force-reinstall` aligned the wheel with the tree -- and
        # a control that can no longer distinguish the two cases is a control
        # nobody is reading. So the check now demands that whichever case holds
        # is *explained*, and it verifies the aligned case directly by comparing
        # the two `cli.py` files byte for byte rather than taking the identical
        # help text as evidence that they are the same file.
        inst = subprocess.run(
            [sys.executable, "-c",
             "import opendocking, pathlib;"
             "print(pathlib.Path(opendocking.__file__).parent)"],
            capture_output=True, text=True, encoding="utf-8",
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        ).stdout.strip()
        inst_cli = Path(inst) / "cli.py" if inst else None
        aligned = bool(
            inst_cli and inst_cli.is_file()
            and inst_cli.read_bytes() == (SRC / "opendocking" / "cli.py").read_bytes()
        )
        check("and the control either contrasts, or is explained by two identical copies",
              ("--report" not in without_tree.stdout) or aligned,
              f"without PYTHONPATH the flag is "
              f"{'absent' if '--report' not in without_tree.stdout else 'PRESENT'}; "
              f"the installed cli.py is {inst_cli}, and it is "
              f"{'byte-identical to' if aligned else 'DIFFERENT from'} the source "
              f"tree's, so the two cases are "
              f"{'distinguishable' if not aligned else 'the same file and the contrast is redundant'}")

        # -------------------------------------------------------------------
        section("a lossy multi-chain receptor is reported in one line")

        dimer = F.write_pdb(tmp / "dimer.pdb", rec_a + rec_b + F.solvent_records(2))
        lib_text, rep = report_for(dimer)
        out = tmp / "dimer.pdbqt"
        result, text = prep(dimer, out)
        printed = lines(result)
        check("the command succeeds and writes the file",
              result.returncode == 0 and out.exists()
              and text.count("ATOM") == rep.pdbqt_atoms_written,
              f"rc={result.returncode}, {text.count('ATOM')} ATOM records, "
              f"report says {rep.pdbqt_atoms_written}")
        check("the file written is the same text the library produces",
              text == lib_text, f"{len(text)} characters, identical")
        detail = printed[1] if len(printed) > 1 else ""
        second = printed[1][:44] + "..." if len(printed) > 1 else "(nothing)"
        check("it prints the write line and one line about the loss",
              len(printed) == 2 and printed[0].startswith("wrote ")
              and printed[1].startswith("  "),
              f"{len(printed)} lines: {printed[0][:40]!r} then {second!r}")
        detail = printed[1] if len(printed) > 1 else ""
        check("that line names the chain kept and the chain dropped",
              f"kept {rep.chains_kept} of {rep.chains_in} chains" in detail
              and f"({rep.chain_ids_kept[0]})" in detail
              and f"dropped chain(s) {rep.chain_ids_dropped[0]}" in detail,
              detail)
        check("that line carries the report's own numbers",
              f"{rep.atoms_kept} of {rep.atoms_in} atoms" in detail
              and f"{rep.atoms_dropped} dropped" in detail
              and f"{rep.water_atoms_removed} water" in detail
              and f"{rep.ion_atoms_removed} ion" in detail
              and f"{rep.other_atoms_removed} other" in detail,
              "every count in the line comes from the report object, not a copy of it")
        check("the raw component warning does not also reach the terminal",
              "kept the largest" not in result.stderr
              and "RuntimeWarning" not in result.stderr,
              "the command replaced it with its own line; two languages for one "
              "fact is how a message gets ignored")
        # The check above proves the source tree is on the child's path by what
        # it can parse. This one proved it by what it *does*: the same input
        # through the installed wheel used to let the raw warning through, so the
        # absence above was this build's doing and not the wheel's. Once the
        # wheel was reinstalled from this build the two behave identically and
        # the contrast is gone -- so the check now demands the difference *or*
        # the explanation, using the byte comparison made a few checks earlier
        # rather than assuming the identical output proves anything.
        wheel = prep(dimer, tmp / "dimer_wheel.pdbqt", source_tree=False)[0]
        check("the behaviour difference is real, or the two copies are the same file",
              ("kept the largest" in wheel.stderr) or aligned,
              f"installed copy: raw warning "
              f"{'present' if 'kept the largest' in wheel.stderr else 'absent'}"
              f"; source tree: "
              f"{'present' if 'kept the largest' in result.stderr else 'absent'}"
              + ("" if aligned else
                 " -- and the two cli.py differ, so this is unexplained"))
        check("it points at the flag that shows more",
              "--report" in detail,
              "an unexplained line is one people stop reading")

        # -------------------------------------------------------------------
        section("a clean receptor says nothing about losing nothing")

        single = F.write_pdb(tmp / "single.pdb", rec_a)
        _, clean = report_for(single)
        out = tmp / "single.pdbqt"
        result, text = prep(single, out)
        printed = lines(result)
        check("a lossless preparation prints exactly one line",
              clean.is_lossy is False and len(printed) == 1
              and printed[0].startswith("wrote "),
              f"{len(printed)} lines, and the report agrees nothing was lost "
              f"(atoms_dropped={clean.atoms_dropped})")
        check("and that line is not about the report at all",
              not any(t in printed[0] for t in ("chain", "dropped", "component")),
              printed[0][:60])
        check("the file is still written, in full",
              text.count("ATOM") == clean.pdbqt_atoms_written > 0,
              f"{text.count('ATOM')} ATOM records")
        check("the reverse direction: a lossy input does print a second line",
              len(printed) == 1 and len(lines(prep(dimer, tmp / "d.pdbqt")[0])) == 2,
              "one line for a clean input, two for a lossy one -- so 'one line' is "
              "not what a CLI that prints nothing would also produce")

        # -------------------------------------------------------------------
        section("a partial loss is reported without inventing a lost chain")

        partial = F.write_pdb(
            tmp / "partial.pdb", rec_a + F.solvent_records(2, chain="A")
        )
        _, part = report_for(partial)
        out = tmp / "partial.pdbqt"
        result, text = prep(partial, out)
        printed = lines(result)
        detail = printed[1] if len(printed) > 1 else ""
        check("atoms lost are still reported when no chain was lost",
              len(printed) == 2 and f"{part.atoms_dropped} dropped" in detail,
              detail)
        check("and no chain is claimed to have gone missing",
              part.chains_dropped == 0 and "dropped chain(s)" not in detail,
              f"chains_dropped={part.chains_dropped}, and the line does not say "
              f"otherwise")

        # -------------------------------------------------------------------
        section("--report prints the breakdown, and answers a clean input too")

        out = tmp / "dimer_report.pdbqt"
        result, _ = prep(dimer, out, "--report")
        printed = lines(result)
        for token, label in (
            (f"{rep.chains_kept} of {rep.chains_in} kept", "chains in/kept"),
            (rep.chain_ids_dropped[0], "the dropped chain"),
            (f"{rep.atoms_kept} of {rep.atoms_in} kept", "atoms in/kept"),
            (f"{rep.atoms_dropped} dropped", "atoms dropped"),
            (f"{rep.fragments_kept} of {rep.fragments_in} kept", "components"),
            (rep.selection, "the selection rule"),
        ):
            check(f"--report states {label}", any(token in l for l in printed),
                  f"looking for {token!r}")
        check("--report is a table in the style prep-ligand already uses",
              all(l.startswith("  ") for l in printed[1:]),
              f"{len(printed) - 1} indented rows after the write line")
        out = tmp / "single_report.pdbqt"
        result, _ = prep(single, out, "--report")
        printed = lines(result)
        check("an explicit request is answered even when nothing was lost",
              len(printed) > 1 and any("0 dropped" in l for l in printed),
              f"{len(printed)} lines, report says atoms_dropped="
              f"{clean.atoms_dropped}")
        check("--report replaces the one-liner rather than adding to it",
              not any("for the full breakdown" in l for l in printed),
              "no 'see --report' hint when --report was already given")

        # -------------------------------------------------------------------
        section("choosing a chain from the command line")

        out = tmp / "keep_b.pdbqt"
        _, rep_b = report_for(dimer, keep_chain="B")
        result, text_b = prep(dimer, out, "--keep-chain", "B")
        check("--keep-chain B keeps chain B and says so",
              result.returncode == 0
              and F.chains_in_pdbqt(text_b) == {"B"}
              and any(f"kept 1 of 2 chains" in l for l in lines(result)),
              f"chains in the file: {sorted(F.chains_in_pdbqt(text_b))}")
        out = tmp / "keep_a.pdbqt"
        _, rep_a = report_for(dimer, keep_chain="A")
        result, text_a = prep(dimer, out, "--keep-chain", "A")
        check("--keep-chain A gives a different receptor from --keep-chain B",
              result.returncode == 0 and text_a != text_b
              and F.chains_in_pdbqt(text_a) == {"A"},
              f"{text_a.count('ATOM')} atoms against {text_b.count('ATOM')}")
        check("the two runs report the opposite chain as dropped",
              rep_a.chain_ids_dropped == ("B",) and rep_b.chain_ids_dropped == ("A",),
              f"A run drops {rep_a.chain_ids_dropped}, B run drops "
              f"{rep_b.chain_ids_dropped}")
        check("--keep-chain with the default is the same receptor as without it",
              prep(dimer, tmp / "keep_default.pdbqt", "--keep-chain", "B")[1]
              == prep(dimer, tmp / "plain.pdbqt")[1],
              "asking for the chain that would have been chosen anyway changes "
              "nothing")
        out = tmp / "keep_z.pdbqt"
        result, _ = prep(dimer, out, "--keep-chain", "Z")
        check("a chain that is not in the file fails loudly",
              result.returncode != 0 and "no chain 'Z'" in result.stderr,
              f"rc={result.returncode}, stderr {result.stderr.strip()[:60]!r}")
        check("and it does not leave a receptor behind",
              not out.exists(),
              "a failed run must not leave a file that looks like a result")

        # -------------------------------------------------------------------
        section("the other warnings still reach the terminal")

        out = tmp / "crambin.pdbqt"
        result, _ = prep(CRAMBIN, out)
        check("a per-atom warning the report does not carry is not silenced",
              "polar hydrogen" in result.stderr,
              "the 'dropped a polar hydrogen' warnings are per atom and appear in "
              "no report, so silencing them would be hiding data")
        check("a lossless receptor produces no component warning to suppress",
              "kept the largest" not in result.stderr,
              f"crambin atoms_dropped={report_for(CRAMBIN)[1].atoms_dropped}")

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

        # The silence guard, run against a CLI that always speaks. The
        # predicate is the one used above, so a guard that cannot reject this
        # cannot be trusted on the real output either.
        def too_chatty(stdout: str, report) -> bool:
            got = [l for l in stdout.splitlines() if l.strip()]
            return not (report.is_lossy is False and len(got) == 1
                        and got[0].startswith("wrote "))

        chatty = "wrote out.pdbqt (7 atoms)\n  kept 1 of 1 chains (A); 11 of 11 atoms\n"
        check("the silence guard rejects a CLI that always reports",
              too_chatty(chatty, clean) and not too_chatty(result.stdout, clean),
              "two lines for a lossless input is rejected, one line is not")

    return finish()


if __name__ == "__main__":
    sys.exit(main())
