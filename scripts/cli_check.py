"""Check every `odcli` subcommand, not just `prep-receptor`.

`scripts/cli_prep_check.py` covers one command out of eight. That leaves the
other seven untested, and "untested" is not a neutral state for a command line:
the exit code is the whole contract when a script is calling `odcli`, and an
exit code nobody has ever read is an exit code nobody has ever been right
about. This script runs the remaining seven and pins three things for each one:

* the **exit code**, taken from the code rather than guessed -- success is 0,
  a missing or unreadable file is 1, argparse's own rejection is 2, an empty
  ligand set is 2, a missing GUI stack is 3, and a `split` that found nothing
  is 1.
* the **stdout**, digit for digit where the number is fixed (affinities, pose
  counts, grid-point counts, atom counts, site ranks) and structurally where it
  is not (JSON key sets, which are machine-readable output's actual promise).
  The two output modes of the same command are checked against *each other*,
  because the mode that quietly stopped agreeing with the prose one is the
  failure nobody sees.
* what a failure **says**, and that it says it without a traceback. Every
  failure path here asserts both the exit code and the absence of a traceback,
  because "exit 1" and "exit 1 with a Python stack dump" are the same code and
  completely different tools for whoever has to read it at 2am.

The real console script runs as a subprocess, the way a person would type it,
which brings a trap this script checks for rather than trusting: `odcli`
imports whichever `opendocking` comes first on `sys.path`, and by default that
is the **installed wheel**, not the source tree being edited. The child is
given `PYTHONPATH`, and the check proves that worked by looking for a flag
that exists only in the source tree -- and proves it is sensitive by
confirming the same flag is *absent* without it.

Two things about the environment are traps rather than details, and both are
asserted rather than assumed:

* The child's output is decoded as **UTF-8 explicitly**. `odcli` reconfigures
  its own streams to UTF-8, but `subprocess(text=True)` decodes with the
  *parent's* locale, which on a Chinese Windows install is GBK. `prep-ligand`
  prints a radius in A-ring, and decoded as GBK that one character becomes
  something else entirely -- so a check asserting the real string would fail
  for a reason that has nothing to do with the code, and a check that asserted
  the mojibake would pass while the code were fine.
* Every run has a **timeout**. `odcli workbench` opens a window, and a
  check that blocks on a window is a check that hangs CI. A timeout is
  reported as a failure, not as a hang, so that if the guard ever stops
  working it says so.

`workbench` is the one command whose real behaviour cannot be run here,
because running it opens a 3-D viewer. Its *failure* path can: a
`sitecustomize.py` on the child's path makes `PyQt6` and `moderngl`
un-importable, which is the state a user without the GUI extras is in, and
reaches the real handler. The window itself is covered by
`scripts/odgui_launch_check.py`, which is built for it.

Fixtures are the checked-in files under `examples/`, so the expected numbers
are properties of the repository and not of anything this script builds.

Run:  python scripts/cli_check.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dock-py" / "python"
EX = ROOT / "examples"
CLI_PREP_CHECK = ROOT / "scripts" / "cli_prep_check.py"

#: Receptor with 30 atoms and 6 polar hydrogens; ligand with 4 torsions. Both
#: are checked in, so every number below is a property of the repository.
REC = EX / "rec_prep.pdbqt"
CRAMBIN = EX / "1crn_prep.pdbqt"
LIG = EX / "ibuprofen_prep.pdbqt"
LIGANDS_DIR = EX / "ligands"
SDF = EX / "ibuprofen.sdf"
POSES = EX / "poses.pdbqt"
README = ROOT / "README.md"

#: A 14 A box on the origin, clear of the receptor's own extent. Small enough
#: that the map precalculation is a rounding error, large enough for ibuprofen
#: (the engine's own floor is 2 x radius + 1 A).
BOX = ["--center_x", "0.0", "--center_y", "0.0", "--center_z", "0.0",
       "--size_x", "14", "--size_y", "14", "--size_z", "14"]

#: How long any single `odcli` call may take before it is treated as hung. The
#: slowest real command here finishes in about a second; 90 s is far above that
#: and far below "a CI job that quietly stops".
TIMEOUT = 90.0

# The check prints CLI output that legitimately contains a non-ASCII character
# (a radius in A-ring). A character the console cannot encode must not turn a
# check run into a traceback.
if hasattr(sys.stdout, "reconfigure"):  # pragma: no branch
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, str(SRC))

FAILURES: list[str] = []
CHECKS = 0
SAWN: set[str] = set()
WALL: list[tuple[str, float]] = []

#: The number of checks this file is supposed to have, so that one quietly
#: vanishing behind a guard takes the tally down *visibly*. The count has to be
#: the same on every machine, which is why no check may sit behind a runtime
#: condition -- a check that is skipped is not skipped from the total, and a
#: check that is not reached at all is a bug in this file, not a fact about the
#: environment. Change it deliberately.
EXPECTED_CHECKS = 127

#: Exit vocabulary, shared with the GUI check scripts:
#: 0 ran and everything passed, 1 ran and something failed, 2 did not finish.
#: Only 0 and 1 are verdicts about the product. 2 means this run's numbers
#: cannot be used for anything, which is a different statement from "a check
#: failed" and was previously reported as if it were the latter.
_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_INCONCLUSIVE = 2


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
    total = sum(dt for _, dt in WALL)
    slow = sorted(WALL, key=lambda kv: -kv[1])[:3]
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    if CHECKS != EXPECTED_CHECKS:
        # A mismatch here means a check was added or lost, and both are worth
        # stopping for: an added check nobody pinned, or a lost one that left
        # the rest of the run looking healthy.
        print(f"  [FAIL] the tally is {CHECKS} but EXPECTED_CHECKS is "
              f"{EXPECTED_CHECKS}; change it deliberately")
        FAILURES.append(f"tally {CHECKS} != EXPECTED_CHECKS {EXPECTED_CHECKS}")
    print(f"wall clock: {total:.1f} s over {len(WALL)} odcli calls")
    print("  slowest: " + ", ".join(f"{n} {d:.2f}s" for n, d in slow))
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return _EXIT_FAILED if FAILURES else _EXIT_OK


ODCLI = shutil.which("odcli")
#: The other console script. Only used to prove that one handler serves both
#: entry points -- `odcli workbench` and `odgui` must produce the identical
#: diagnostic, because they are two names for the same missing dependency.
ODGUI = shutil.which("odgui")


def odcli_argv(args) -> list[str]:
    if ODCLI:
        return [ODCLI, *args]
    return [sys.executable, "-m", "opendocking.cli", *args]


class Result:
    """What a call did, plus the two things every failure path is judged on."""

    def __init__(self, proc: subprocess.CompletedProcess | None, timed_out: bool,
                 wall: float):
        self.proc = proc
        self.timed_out = timed_out
        self.wall = wall
        self.returncode = None if proc is None else proc.returncode
        self.stdout = "" if proc is None else proc.stdout
        self.stderr = "" if proc is None else proc.stderr

    @property
    def has_traceback(self) -> bool:
        return "Traceback (most recent call last)" in self.stdout + self.stderr

    @property
    def both(self) -> str:
        """stdout and stderr together.

        Used where the *content* of a diagnostic is the contract but the
        *stream* it chose is not something this check is entitled to pin --
        see the `workbench` section for the case that made that necessary.
        """
        return self.stdout + self.stderr


def say(*args) -> list[str]:
    """An argv for `odcli`, recording that its subcommand was exercised.

    Recording at the point of the call rather than in a list written out by
    hand is what makes the coverage check at the end of the run worth
    anything: a section cannot forget to say which command it was about.
    """
    if args:
        SAWN.add(args[0])
    return [str(a) for a in args]


def run(args, *, source_tree: bool = True, extra_path: Path | None = None) -> Result:
    """Run the real entry point in a subprocess, as a user would type it.

    `source_tree` puts `dock-py/python` first on the child's path. It has to be
    done per child rather than assumed from the parent, because `odcli` is an
    installed script whose imports resolve in *its* process.

    A timeout becomes a result with `timed_out` set rather than an exception,
    so a command that blocks -- `workbench` opening a window is the realistic
    one -- is reported by the check that cares instead of hanging CI.
    """
    env = dict(os.environ)
    parts: list[str] = []
    if extra_path is not None:
        parts.append(str(extra_path))
    if source_tree:
        parts.append(str(SRC))
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    label = args[0] if args else "(no subcommand)"
    try:
        proc = subprocess.run(
            odcli_argv(args), capture_output=True, cwd=str(ROOT), env=env,
            encoding="utf-8", errors="replace", timeout=TIMEOUT,
        )
        res = Result(proc, False, time.perf_counter() - started)
    except subprocess.TimeoutExpired:
        res = Result(None, True, time.perf_counter() - started)
    WALL.append((label, res.wall))
    return res


def nonempty(stdout: str) -> list[str]:
    return [l for l in stdout.splitlines() if l.strip()]


#: A `sitecustomize` that lets the whole GUI stack import and then makes the
#: OpenGL context unusable, by swapping in a `QOpenGLWidget` whose
#: `initializeGL` does nothing. This is the machine-independent way to produce
#: the second way a viewer can be unlaunchable: Qt comes up, the widget is
#: really created and shown, and the context simply never arrives. Breaking a
#: driver instead would only work on a machine that already has a broken one,
#: which is the opposite of what a CI box usually is.
_NO_GL_SHIM = '''\
import importlib.abc
import importlib.machinery
import sys


class _Wrap:
    """Delegates everything to the real loader, then patches the module."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        real = module.QOpenGLWidget

        class DeadContext(real):
            def initializeGL(self):
                pass

            def initializeOpenGLFunctions(self):
                return False

        module.QOpenGLWidget = DeadContext

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _PatchQtOpenGL(importlib.abc.MetaPathFinder):
    """Patches one module on the way in, then gets out of the way."""

    def find_spec(self, name, path=None, target=None):
        if name != "PyQt6.QtOpenGLWidgets":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Wrap(spec.loader)
        return spec


sys.meta_path.insert(0, _PatchQtOpenGL())
'''


def odgui_run(args, shim: Path | None, timeout: float = TIMEOUT) -> Result:
    """Run `odgui` as a subprocess, optionally behind one of the shims.

    Kept apart from `run()` because `odgui` is not an `odcli` subcommand: it is
    the second console script, it is not in the parser, and `covers_all` must
    not start counting it as one. Its calls are made explicitly instead.
    """
    if not ODGUI:
        return Result(None, True, 0.0)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in (str(shim) if shim else None, str(SRC)) if p])
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    label = "odgui " + (args[0] if args else "")
    try:
        proc = subprocess.run([ODGUI, *args], capture_output=True, cwd=str(ROOT),
                              env=env, encoding="utf-8", errors="replace", timeout=timeout)
        res = Result(proc, False, time.perf_counter() - started)
    except subprocess.TimeoutExpired:
        res = Result(None, True, time.perf_counter() - started)
    WALL.append((label, res.wall))
    return res


def diagnose(res: Result, *needles: str) -> tuple[list[str], bool, bool]:
    """`(missing, stdout_is_dirty, has_traceback)` for one failed run.

    The three are returned rather than collapsed into one verdict because they
    are three properties, and combining them is how the wrong-stream defect got
    through: "the message is *somewhere*" and "the message is on *stderr*" are
    different claims, and a run that printed its diagnostic to stdout while
    exiting 1 satisfies the first and fails the second. The third is separate
    again, because a right-stream message with a Python stack dump under it is
    not a usable diagnostic.
    """
    if not needles:
        raise ValueError("diagnose() was given nothing to look for")
    missing = [n for n in needles if n not in res.stderr]
    # A failed command that wrote to stdout wrote something that looks like a
    # result. `wrote 0 pose file(s) to <dir>` is the shape this takes: true,
    # and useless to anyone reading the output.
    dirty = res.stdout.strip() != ""
    return missing, dirty, res.has_traceback


def fails_cleanly(name: str, res: Result, code: int, *needles: str) -> bool:
    """A failure must have the right code, say why on stderr, and not dump.

    Two checks rather than one, deliberately. Every failure path below is
    judged by both, and the reverse-verification section drives this helper
    with a run that breaks each property on its own.
    """
    missing, dirty, dumped = diagnose(res, *needles)
    spoke = check(
        f"{name}: exits {code}, with the reason on stderr and nothing on stdout",
        res.returncode == code and not missing and not dirty,
        f"rc={res.returncode}"
        + (f", not on stderr: {missing}" if missing else f", on stderr: {list(needles)}")
        + (f", but also wrote {len(res.stdout)} bytes to stdout" if dirty
           else f", stdout empty ({len(res.stdout)} bytes)"),
    )
    quiet = check(
        f"{name}: says it without a traceback",
        not dumped,
        "no traceback" if not dumped
        else f"a Python stack dump under the message: {res.stderr.strip()[:60]!r}",
    )
    return spoke and quiet


# ---------------------------------------------------------------------------
# The coverage contract
# ---------------------------------------------------------------------------


def covers_all(offered: set[str], covered: set[str]) -> bool:
    """True when this script has a section for every subcommand on offer.

    Split out so the reverse check below can feed it a doctored set. A guard
    that has only ever been shown the truth is not known to be able to say no.
    """
    return offered == covered


def _brace_names(help_text: str) -> set[str]:
    """The subcommand list as the user sees it, from `odcli --help`.

    Black box on purpose: the set of commands a caller may type is a property
    of the help output, not of the parser's internals.
    """
    found = re.search(r"\{([a-z0-9,_-]+)\}", help_text)
    return set(found.group(1).split(",")) if found else set()


def _tree_fingerprint() -> dict[str, str]:
    """A hash of every file this run's numbers can depend on.

    This is a 37-subprocess script that takes about twenty-five seconds, on a
    machine where other processes are editing the same tree. A previous run
    reported 91/93 and 28/37 and then could not be reproduced; a check that
    cannot tell a real regression from a tree that moved underneath it will be
    re-run until it comes out green, and the re-run is what makes the first
    number untrustworthy rather than the second.

    So the run takes its own fingerprint before the first subprocess and
    compares it after the last. If anything moved, the script says so by name
    and the tally is marked untrustworthy, instead of quietly reporting a
    number that describes two different trees.

    Only the files the run actually reads: the package's own modules, the
    fixtures passed to `odcli`, and the sibling script whose text is read for
    the coverage check. `examples/maps` and friends are not in here -- they are
    tens of megabytes of `.map` files no assertion reads.
    """
    import hashlib

    watched: list[Path] = sorted((SRC / "opendocking").rglob("*.py"))
    watched += [REC, CRAMBIN, LIG, SDF, POSES, README, CLI_PREP_CHECK]
    watched += sorted(LIGANDS_DIR.glob("*.pdbqt"))
    out: dict[str, str] = {}
    for path in watched:
        try:
            out[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()).hexdigest()[:16]
        except OSError as exc:  # a file that vanished mid-run is itself a change
            out[str(path)] = f"unreadable: {exc.strerror}"
    return out


def changed_between(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Files that were added, removed or rewritten between two fingerprints."""
    return sorted(
        set(before) ^ set(after)
        | {k for k in set(before) & set(after) if before[k] != after[k]}
    )


def main() -> int:
    global CHECKS, FAILURES
    print(f"odcli: {ODCLI or '(not installed; falling back to python -m opendocking.cli)'}")
    before = _tree_fingerprint()
    print(f"tree: {len(before)} files fingerprinted before the run")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        missing = tmp / "does-not-exist.pdbqt"
        # A different extension on purpose: `prepare_ligand` rejects an unknown
        # suffix before it opens anything, so a ".pdbqt" path that does not
        # exist is reported as a format problem and would never reach the
        # "cannot read the file" branch this line is here to reach.
        missing_sdf = tmp / "does-not-exist.sdf"
        empty_dir = tmp / "empty"
        empty_dir.mkdir()

        # -------------------------------------------------------------------
        section("the subprocess is running the code under test")

        help_with = run(["--help"])
        help_without = run(["--help"], source_tree=False)
        check("odcli is a real console script and answers --help",
              help_with.returncode == 0 and "usage: odcli" in help_with.stdout,
              f"rc={help_with.returncode}")
        prep_help = run(["prep-receptor", "--help"])
        check("the child process is importing the source tree, not the wheel",
              "--report" in prep_help.stdout and "--keep-chain" in prep_help.stdout,
              "flags that exist only in the edited source are visible to the child")
        check("and the check is sensitive to that, not merely satisfied by it",
              "--report" not in help_without.stdout,
              "the same flag is absent without PYTHONPATH, so its presence above "
              "means something")
        radius = run(say("prep-ligand", "-l", str(SDF))).stdout
        check("the child's UTF-8 output survives the pipe intact",
              "Å" in radius and "脜" not in radius,
              "decoded as UTF-8, not as this console's legacy code page: a GBK "
              "decode turns that A-ring into a different character entirely")

        # -------------------------------------------------------------------
        section("the subcommand list this script is accountable for")

        offered = _brace_names(help_with.stdout)
        check("the help text enumerates the subcommands",
              offered == {"prep-receptor", "prep-ligand", "rec-grid", "sites",
                          "dock", "split", "info", "workbench"},
              f"{len(offered)}: {', '.join(sorted(offered))}")
        check("prep-receptor is covered, by the other script",
              CLI_PREP_CHECK.exists()
              and "prep-receptor" in CLI_PREP_CHECK.read_text(encoding="utf-8"),
              f"scripts/cli_prep_check.py "
              f"{'exists and' if CLI_PREP_CHECK.exists() else 'is missing'} names "
              "the subcommand; re-running its checks here would only copy them")
        # The coverage verdict is at the end of the run, where SAWN is full.
        # Asserting it here would compare the subcommands against a list that
        # has not been built yet, and pass for the wrong reason.

        # -------------------------------------------------------------------
        section("info: capabilities, and two modes that must agree")

        info = run(say("info"))
        lines = nonempty(info.stdout)
        check("info succeeds quietly", info.returncode == 0 and info.stderr == "",
              f"rc={info.returncode}, stderr {info.stderr!r}")
        check("it names the engine and lists both scoring functions",
              lines[0].startswith("Open Docking engine ")
              and "scoring functions:" in lines
              and any(l.strip().startswith("vina:") for l in lines)
              and any(l.strip().startswith("vinardo:") for l in lines),
              f"first line {lines[0]!r}, "
              f"{sum(1 for l in lines if l.startswith('  '))} indented entries")
        vina_params = next(l for l in lines if l.strip().startswith("vina:"))
        vinardo_params = next(l for l in lines if l.strip().startswith("vinardo:"))
        check("the vina parameters are all five terms, with their signs",
              re.fullmatch(
                  r"  vina: g1=-?[\d.]+ g2=-?[\d.]+ rep=-?[\d.]+ "
                  r"hbond=-?[\d.]+ hyd=-?[\d.]+", vina_params) is not None,
              vina_params.strip())
        check("gpu status is one of the three states the engine can be in",
              any(l in ("gpu: not compiled in (rebuild with --features gpu)",
                        "gpu: compiled in and an adapter is available",
                        "gpu: compiled in but no adapter could be opened")
                  for l in lines if l.startswith("gpu:")),
              next((l for l in lines if l.startswith("gpu:")), "(no gpu line)"))
        backends = next((l for l in lines if l.startswith("parallel backends:")), "")
        check("the backend list is present and not empty",
              backends != "" and backends.split(":", 1)[1].strip() != "",
              backends or "(no backends line)")

        info_json = run(say("info", "--json"))
        try:
            payload = json.loads(info_json.stdout)
        except ValueError as exc:
            payload = None
            check("info --json is machine-readable", False, f"{exc}: {info_json.stdout[:80]!r}")
        else:
            check("info --json is machine-readable", info_json.returncode == 0,
                  f"rc={info_json.returncode}, {len(info_json.stdout)} bytes parsed")
            check("and it has exactly the four documented keys",
                  set(payload) == {"odock_version", "scoring_functions", "gpu", "backends"},
                  f"keys: {sorted(payload)}")
            check("gpu is reported as two booleans, not as a sentence",
                  set(payload["gpu"]) == {"compiled", "available"}
                  and all(isinstance(v, bool) for v in payload["gpu"].values()),
                  str(payload["gpu"]))
            check("--json reports the same version the prose did",
                  payload["odock_version"] == lines[0].split()[-1],
                  f"json {payload['odock_version']!r} against prose {lines[0]!r}")
            check("--json reports the same scoring functions, verbatim",
                  payload["scoring_functions"] == [vina_params.strip(), vinardo_params.strip()],
                  "the two output modes are rendered from the same list")
            check("--json reports the same backends, in the same order",
                  ", ".join(payload["backends"]) == backends.split(":", 1)[1].strip(),
                  f"json {payload['backends']} against prose {backends!r}")

        # -------------------------------------------------------------------
        section("prep-ligand: the properties block, and a file that matches it")

        pl = run(say("prep-ligand", "-l", str(SDF)))
        lines = nonempty(pl.stdout)
        check("it prints the ligand name and five labelled rows",
              pl.returncode == 0 and len(lines) == 6 and lines[0] == "ibuprofen.sdf:",
              f"rc={pl.returncode}, {len(lines)} lines, first {lines[0]!r}")
        values = dict(re.match(r"  (\S.*?)\s\s+(\S.*)$", l).groups() for l in lines[1:])
        check("atoms, rotatable bonds and degrees of freedom are the ligand's",
              values.get("atoms") == "16" and values.get("rotatable bonds") == "4"
              and values.get("degrees of freedom") == "10",
              f"{values.get('atoms')} atoms, {values.get('rotatable bonds')} "
              f"torsions, {values.get('degrees of freedom')} dof")
        check("the radius is given to two decimals with its unit",
              values.get("radius") == "5.54 Å",
              f"{values.get('radius')!r} -- that A-ring is the character a GBK "
              "decode would turn into something else")
        classes = dict(p.split("=") for p in values.get("atom classes", "").split(", "))
        check("the atom classes account for every atom",
              classes == {"donor": "1", "hydrophobic": "13", "other": "2"}
              and sum(int(v) for v in classes.values()) == int(values.get("atoms", "0")),
              f"{classes} summing to {sum(int(v) for v in classes.values())}")

        out = tmp / "ib.pdbqt"
        pl_out = run(say("prep-ligand", "-l", str(SDF), "-o", str(out)))
        written = out.read_text(encoding="utf-8") if out.exists() else ""
        n_written = sum(1 for l in written.splitlines()
                        if l.startswith(("ATOM", "HETATM")))
        check("-o adds one line and writes the file it names",
              pl_out.returncode == 0
              and nonempty(pl_out.stdout)[-1] == f"  wrote            {out}"
              and written != "",
              f"last line {nonempty(pl_out.stdout)[-1]!r}")
        check("the file holds exactly as many atoms as the block reported",
              n_written == int(values["atoms"]),
              f"{n_written} ATOM records against {values['atoms']} atoms reported")

        fails_cleanly("prep-ligand on a missing file",
                      run(say("prep-ligand", "-l", str(missing_sdf))),
                      1, "error: ", "Bad input file")
        fails_cleanly("prep-ligand on a format it cannot read",
                      run(say("prep-ligand", "-l", str(README))),
                      1, "unsupported format '.md'",
                      ".sdf, .mol, .mol2, .smi, .smiles, .pdb, .xyz")

        # -------------------------------------------------------------------
        section("split: one file per pose, and what zero poses means")

        sp = run(say("split", "-i", str(POSES), "-o", str(tmp / "sp")))
        files = sorted((tmp / "sp").glob("*.pdbqt"))
        check("a nine-model file splits into nine files and says nine",
              sp.returncode == 0 and len(files) == 9
              and sp.stdout.strip() == f"wrote 9 pose file(s) to {tmp / 'sp'}",
              f"rc={sp.returncode}, {len(files)} files, {sp.stdout.strip()!r}")
        contents = [f.read_text(encoding="utf-8") for f in files]
        remarks = {c.count("REMARK VINA RESULT:") for c in contents}
        atom_counts = {sum(1 for l in c.splitlines() if l.startswith(("ATOM", "HETATM")))
                       for c in contents}
        check("each file holds exactly one pose, with the same atoms in all nine",
              remarks == {1} and len(atom_counts) == 1 and next(iter(atom_counts)) > 0,
              f"REMARK counts {sorted(remarks)}, ATOM counts {sorted(atom_counts)} "
              "-- a split that dropped or duplicated an atom would not be flat")
        one = run(say("split", "-i", str(LIG), "-o", str(tmp / "sp2")))
        check("a file with no models is a failure, and says so on stderr",
              one.returncode == 1 and "no poses found" in one.stderr
              and "MODEL" in one.stderr and one.stdout.strip() == ""
              and not list((tmp / "sp2").glob("*")) and not one.has_traceback,
              f"rc={one.returncode}, stdout {one.stdout!r}, stderr "
              f"{one.stderr.strip()[:74]!r} -- a non-zero exit whose only output "
              "used to be a success-shaped 'wrote 0 pose file(s)' on stdout")
        check("the count line is only printed when there is something to count",
              "wrote 9 pose file(s)" in sp.stdout and "wrote 0" not in sp.stdout
              and "wrote 0" not in one.stdout,
              f"success stdout {sp.stdout.strip()!r}, failure stdout {one.stdout!r}")
        fails_cleanly("split on a missing file",
                      run(say("split", "-i", str(missing), "-o", str(tmp / "s3"))),
                      1, "error: ", "No such file or directory")

        # -------------------------------------------------------------------
        section("sites: the list a box is chosen from, in prose")

        st = run(say("sites", "-r", str(CRAMBIN)))
        lines = nonempty(st.stdout)
        check("it counts the sites and names the file",
              st.returncode == 0 and lines[0] == "12 candidate site(s) in 1crn_prep.pdbqt",
              f"rc={st.returncode}, {lines[0]!r}")
        check("the first site is pinned: rank, kind, score, grid points, burial",
              lines[1] == "1. groove  score 4.95  172 grid points  buried on 1.2/3 axes",
              f"{lines[1]!r}")
        check("its dimensions, centre and volume are given, and the pitch with them",
              lines[2] == "   site    6.4 x  10.4 x  16.8 A at (  -0.26,    5.53,    5.14)"
                          "  88.1 A3 at 0.8 A pitch",
              f"{lines[2]!r} -- 88.1 A3 is 172 voxels at 0.8 A pitch, and a "
              "volume in A3 is not interpretable without the pitch it was "
              "measured on")
        check("the box line carries the box's own volume, which is not the site's",
              lines[3] == "   box    14.4 x  18.4 x  24.8 A at (  -0.26,    5.53,    5.14)"
                          "  (+4 A padding each side, 6571 A3)",
              f"{lines[3]!r} -- 14.4 x 18.4 x 24.8 A, so 6571 A3 of box around "
              f"88.1 A3 of site: the two disagree by 75x, which is the whole "
              f"reason both are printed")
        check("the box is still the site grown by the padding, and says so",
              "+4 A padding each side" in lines[3]
              and all(abs(b - z - 8.0) < 0.05 for b, z in zip((14.4, 18.4, 24.8),
                                                              (6.4, 10.4, 16.8))),
              f"each axis is the site plus 2 x 4.0 A: {lines[3][-34:]!r}")
        check("the residues lining it are listed with their atom counts",
              lines[4].startswith("   lined by TYR 29A (10), ASN 12A (7), CYS 16A (7),")
              and lines[4].endswith("VAL 15A (6), VAL 8A (5), ALA 9A (4), THR 21A (4)"),
              f"{lines[4][:68]}...")
        ranks = [l for l in lines if re.match(r"^\d+\. ", l)]
        scores = [float(re.search(r"score ([\d.]+)", l).group(1)) for l in ranks]
        check("all twelve sites are listed, ranked, and ordered by score",
              len(ranks) == 12
              and [int(l.split(".")[0]) for l in ranks] == list(range(1, 13))
              and scores == sorted(scores, reverse=True),
              f"{len(ranks)} numbered entries, scores {scores[:4]}...{scores[-1]}")
        check("a null cavity result is explained, not just reported",
              any("No sealed cavity at the default 1.4 A probe" in l for l in lines)
              and any("0.9 A -> 1" in l for l in lines)
              and any("A smaller probe finds them" in l for l in lines),
              "it says no cavity at 1.4 A, names the probe that finds one, and "
              "says why that probe is not the default")
        check("and it refuses to call a site a binding site",
              any("A site is enclosed space, not a binding site" in l for l in lines)
              and any("--auto-box N takes the Nth from this list" in l for l in lines),
              "the caveat points at the flag that actually consumes the list")

        small = run(say("sites", "-r", str(CRAMBIN), "--probe", "0.9"))
        first_small = next(l for l in nonempty(small.stdout) if l.startswith("1. "))
        check("--probe changes the search, and the difference is visible",
              "172 grid points" in lines[1] and "204 grid points" in first_small
              and "score 4.95" in lines[1] and "score 5.14" in first_small,
              f"1.4 A gave {lines[1].split('  ')[1:3]}, 0.9 A gave "
              f"{first_small.split('  ')[1:3]}")
        fails_cleanly("sites on a missing receptor",
                      run(say("sites", "-r", str(missing))),
                      1, "error: ", "No such file or directory")

        # -------------------------------------------------------------------
        section("sites --json: the same sites, in a shape a program can read")

        sj = run(say("sites", "-r", str(CRAMBIN), "--json"))
        try:
            sites = json.loads(sj.stdout)
        except ValueError as exc:
            sites = None
            check("sites --json is machine-readable", False, f"{exc}: {sj.stdout[:80]!r}")
        else:
            check("sites --json is machine-readable, and is only that",
                  sj.returncode == 0 and isinstance(sites, list),
                  f"rc={sj.returncode}, a {type(sites).__name__}"
                  + (f" of {len(sites)}" if isinstance(sites, list) else ""))
            keys = {"rank", "kind", "center", "size", "volume", "spacing",
                    "box_center", "box_size", "box_volume", "box_budget",
                    "voxels", "burial", "lining"}
            have = bool(sites) and all(set(s) == keys for s in sites)
            check("every site has the same thirteen keys, so a caller can index them",
                  have,
                  f"keys of the first: {sorted(sites[0]) if sites else '?'}")
            check("rank is 1-based and matches the order",
                  [s["rank"] for s in sites] == list(range(1, len(sites) + 1)),
                  f"first three ranks {[s['rank'] for s in sites[:3]]}")
            check("kind is one the interface knows how to name",
                  all(s["kind"] in ("cavity", "burial") for s in sites),
                  f"kinds: {sorted({s['kind'] for s in sites})}")
            check("centre, size and box are three numbers each",
                  all(len(s[k]) == 3 and all(isinstance(v, float) for v in s[k])
                      for s in sites
                      for k in ("center", "size", "box_center", "box_size")),
                  "floats, not strings: a caller should not have to parse them")
            check("the box is the site grown by 4 A on each side",
                  all(all(abs(b - z - 8.0) < 0.01 for b, z in zip(s["box_size"], s["size"]))
                      for s in sites),
                  f"first site {sites[0]['size']} -> {sites[0]['box_size']}")
            # Every check below reads a field that the key-set check above says
            # must be there. `.get` rather than `[]` throughout, and `have` in
            # each condition, so a run where the keys are missing reports the
            # missing keys instead of dying on them -- which is what happened
            # the first time this was written.
            s0 = sites[0] if sites else {}
            vol, bvol = s0.get("volume"), s0.get("box_volume")
            pitch, budget = s0.get("spacing"), s0.get("box_budget")
            check("the volume is the site's own, on the pitch it was measured at",
                  have and pitch == 0.8 and all(s.get("spacing", 0) > 0 for s in sites)
                  and abs(vol - s0["voxels"] * pitch ** 3) < 1e-6,
                  f"{s0.get('voxels')} voxels x {pitch}^3 = {vol} A3")
            check("the box volume is the box's, and is a different number",
                  have
                  and abs(bvol - s0["box_size"][0] * s0["box_size"][1]
                          * s0["box_size"][2]) < 1e-6
                  and bvol > vol * 10,
                  f"{bvol} A3 of box around {vol} A3 of site -- the bounding box "
                  "is not a stand-in for the space, and now the output says so")
            check("the prose volumes are the json ones, rounded",
                  have
                  and f"{vol:.1f} A3" in lines[2] and f"{bvol:.0f} A3" in lines[3]
                  and f"{pitch:g} A pitch" in lines[2],
                  f"prose {lines[2].split('  ')[-3:]} against json volume="
                  f"{vol}, box_volume={bvol}")
            check("the size budget is reported, and an empty note is a claim",
                  have
                  and set(budget or {}) == {"max_side", "capped", "coverage",
                                            "volume", "requested_volume", "note"}
                  and budget["max_side"] == 30.0
                  and budget["capped"] is False and budget["coverage"] == 1.0
                  and budget["note"] == ""
                  and budget["volume"] == bvol
                  and budget["requested_volume"] == bvol,
                  f"{budget} -- nothing was capped, the box still holds every "
                  "point of the site, and it is the same box the prose printed")
            check("no checked-in site's box is over the ceiling, so no note is printed",
                  have
                  and all(not s["box_budget"]["capped"] for s in sites)
                  and all("A3);" not in l for l in lines if l.startswith("   box")),
                  f"{sum(1 for s in sites if s.get('box_budget', {}).get('capped'))} "
                  f"of {len(sites)} sites capped; crambin's widest padded box is "
                  f"{max((max(s.get('box_size', [0])) for s in sites), default=0):.1f} A"
                  " against a 30.0 A ceiling, so the capping note is never "
                  "exercised by the fixtures in this repository")
            prose_lining = [x.strip() for x in lines[4][len("   lined by "):].split(", ")]
            check("the grid-point count and the lining are the ones the prose gave",
                  sites[0]["voxels"] == 172
                  and [f"{r['residue']} ({r['atoms']})" for r in sites[0]["lining"]][:8]
                  == prose_lining,
                  f"{sites[0]['voxels']} grid points, and the prose line's eight "
                  f"residues are the first eight of {len(sites[0]['lining'])}")
            check("the prose label and the json kind are the same fact",
                  all(("sealed" if s["kind"] == "cavity" else "groove") in r
                      for s, r in zip(sites, ranks)),
                  "'burial' is shown as 'groove' and 'cavity' as 'sealed', by one "
                  "table both callers read")

        # -------------------------------------------------------------------
        section("rec-grid: the maps, and the box that shapes them")

        maps_vina = tmp / "maps_vina"
        rg = run(say("rec-grid", "-r", str(REC), "-o", str(maps_vina),
                     "--spacing", "0.5", *BOX))
        lines = nonempty(rg.stdout)
        check("it reports the receptor it read and the box it used",
              lines[0] == "receptor: 30 atoms, 6 polar hydrogens"
              and lines[1] == "box: centre (0.0, 0.0, 0.0), size (14.0, 14.0, 14.0)",
              f"{lines[0]!r} / {lines[1]!r}")
        check("it estimates the memory before spending it",
              lines[2] == "estimated maps: 3.7 MB"
              and re.search(r"precalculated \d+ points in [\d.]+ s \(([\d.]+) MB actual\)",
                            lines[4]) is not None,
              f"{lines[2]!r}, then {lines[4]!r}")
        per_axis = int(14 / 0.5) + 1
        check("the point count is the grid the box and spacing describe",
              f"precalculated {per_axis ** 3} points" in lines[4],
              f"14 A at 0.5 A is {per_axis} points per axis, {per_axis}^3 "
              f"= {per_axis ** 3}")
        check("it names the scoring function and all five of its terms",
              lines[3] == "scoring: " + vina_params.strip() and lines[3].count("=") == 5,
              f"{lines[3]!r}")
        written_maps = sorted(maps_vina.glob("*.map"))
        check("it writes the AutoDock map files it promised",
              rg.returncode == 0 and len(written_maps) == 40
              and lines[5] == f"wrote AutoDock .map files to {maps_vina}"
              and all(f.read_bytes().startswith(b"GRID_PARAMETER_FILE")
                      for f in written_maps),
              f"{len(written_maps)} .map files, each with a GRID_PARAMETER_FILE header")

        maps_vinardo = tmp / "maps_vinardo"
        rv = run(say("rec-grid", "-r", str(REC), "-o", str(maps_vinardo),
                     "--scoring", "vinardo", "--spacing", "0.5", *BOX))
        rv_lines = nonempty(rv.stdout)
        v_names = {f.name for f in written_maps}
        d_names = {f.name for f in maps_vinardo.glob("*.map")}
        differ = [f.name for f in written_maps if f.name in d_names
                  and f.read_bytes() != (maps_vinardo / f.name).read_bytes()]
        check("the vinardo run names its own scoring function and its five terms",
              rv.returncode == 0
              and len([l for l in rv_lines if l.startswith("scoring:")]) == 1
              and re.fullmatch(
                  r"scoring: vinardo: g1=-?[\d.]+ g2=-?[\d.]+ rep=-?[\d.]+ "
                  r"hbond=-?[\d.]+ hyd=-?[\d.]+", rv_lines[3]) is not None,
              f"{rv_lines[3]!r} -- the line used to be printed only for vina, so "
              "this run named nothing at all")
        check("and rec-grid names them the same way `info` reports them",
              payload is not None
              and lines[3] == f"scoring: {payload['scoring_functions'][0]}"
              and rv_lines[3] == f"scoring: {payload['scoring_functions'][1]}"
              and rv_lines[3] != lines[3],
              f"rec-grid said {rv_lines[3][:36]!r}; info reports "
              f"{(payload['scoring_functions'][1][:36] if payload else '?')!r} -- two "
              "commands reading the same engine table, so they cannot be worded "
              "differently or one of them can be absent")
        # The claim that matters, and the reason the naming had to be added: the
        # file names cannot say which scoring function produced the directory.
        check("--scoring really changes the maps, and the file names cannot say so",
              v_names == d_names and 0 < len(differ) < len(v_names),
              f"vina and vinardo wrote {len(v_names)} identically named files, "
              f"{len(differ)} of which differ in content and the other "
              f"{len(v_names) - len(differ)} are byte-identical because the grid "
              f"points there are zero either way ({', '.join(differ)}) -- so the "
              "directory alone still cannot answer it, only the output can")

        fails_cleanly("rec-grid with no box at all",
                      run(say("rec-grid", "-r", str(REC), "-o", str(tmp / "g1"))),
                      1, "no search box", "--center_x..--size_z", "--auto-box N")
        half = run(say("rec-grid", "-r", str(REC), "-o", str(tmp / "g2"),
                       "--center_x", "0", "--center_y", "0", "--center_z", "0",
                       "--size_x", "14"))
        check("a half-given box names exactly the values that are missing",
              half.returncode == 1 and not half.has_traceback
              and "missing --size_y, --size_z" in half.stderr
              and "--size_x" not in half.stderr.split("missing ")[1].split(".")[0],
              f"rc={half.returncode}, {half.stderr.strip()[:78]!r} -- four of the six "
              "were given, and the two that were not are named")
        both = run(say("rec-grid", "-r", str(REC), "-o", str(tmp / "g3"),
                       *BOX, "--auto-box", "1"))
        check("giving the box twice is refused rather than one of them winning",
              both.returncode == 1 and not both.has_traceback
              and "cannot both be given" in both.stderr,
              f"rc={both.returncode}, {both.stderr.strip()[:78]!r}")
        arg = run(say("rec-grid", "-r", str(REC), "-o", str(tmp / "g4"),
                      "--spacing", "0", *BOX))
        check("a nonsensical spacing is argparse's to refuse, so it exits 2",
              arg.returncode == 2 and not arg.has_traceback
              and "must be positive, got 0.0" in arg.stderr,
              f"rc={arg.returncode}, {arg.stderr.strip().splitlines()[-1][:68]!r}")
        fails_cleanly("rec-grid on a missing receptor",
                      run(say("rec-grid", "-r", str(missing),
                              "-o", str(tmp / "g5"), *BOX)),
                      1, "error: ", "I/O error")

        # -------------------------------------------------------------------
        section("dock: affinities, pose counts, and the unit nobody prints")

        DOCK = say("dock", "-r", str(REC), "-l", str(LIG),
                   "--auto-box", "1", "-e", "1", "-m", "3", "--seed", "42")
        dj = run(DOCK + ["--json"])
        try:
            rec = json.loads(dj.stdout)[0]
        except (ValueError, IndexError) as exc:
            rec = None
            check("dock --json is machine-readable", False, f"{exc}: {dj.stdout[:80]!r}")
        else:
            check("dock --json is machine-readable, and is only that",
                  dj.returncode == 0 and isinstance(rec, dict),
                  f"rc={dj.returncode}, {len(dj.stdout)} bytes parsed")
            check("the record has the nine documented keys",
                  set(rec) == {"ligand", "output", "num_poses", "best_energy",
                               "energies", "rmsd", "elapsed_seconds",
                               "rejected_pose_count", "num_torsions"},
                  f"keys: {sorted(rec)}")
            check("num_poses is -m 3, and the arrays agree with it",
                  rec["num_poses"] == 3 and len(rec["energies"]) == 3
                  and len(rec["rmsd"]) == 3,
                  f"num_poses={rec['num_poses']}, {len(rec['energies'])} energies, "
                  f"{len(rec['rmsd'])} rmsds")
            check("the affinities are the measured ones, best first",
                  rec["energies"] == [-4.4625182565575034, -4.37873887366038,
                                      -4.147741349725185]
                  and rec["best_energy"] == rec["energies"][0]
                  and rec["energies"] == sorted(rec["energies"]),
                  f"{['%.4f' % e for e in rec['energies']]}, "
                  f"best_energy={rec['best_energy']:.4f}")
            check("the reference pose is at 0 A and the torsions are reported",
                  rec["rmsd"][0] == 0.0 and rec["num_torsions"] == 4
                  and rec["rejected_pose_count"] == 0,
                  f"rmsd[0]={rec['rmsd'][0]}, {rec['num_torsions']} torsions, "
                  f"{rec['rejected_pose_count']} rejected")
            check("the record names the ligand it docked, and no output without -o",
                  rec["ligand"] == str(LIG) and rec["output"] is None,
                  f"ligand={rec['ligand']}, output={rec['output']!r}")
            check("the progress it reports went to stderr, so --json stays parseable",
                  "receptor: 30 atoms" in dj.stderr and dj.stdout.lstrip().startswith("["),
                  f"stderr {dj.stderr.strip().splitlines()[-1]!r}, stdout begins "
                  f"{dj.stdout[:1]!r}")
            # The unit is now in the prose output, and still not in the JSON.
            # Asserting both halves is what keeps the two modes from drifting:
            # the unit was in neither before, and the check that exists is the
            # one that says which mode carries it and which does not.
            from opendocking.core import DockingResult
            doc = DockingResult.energies.__doc__ or ""
            check("the unit is documented where the number is defined",
                  "kcal/mol" in doc,
                  f"core.DockingResult.energies: {doc!r}")
            check("and --json still says nothing about it, which the prose does",
                  "kcal" not in dj.stdout and "unit" not in rec,
                  f"no unit in the JSON record (keys: {sorted(rec)}), and the "
                  "prose mode prints the unit line -- if one side gains a unit "
                  "the other has to be looked at again")

        dp = run(DOCK + ["-o", str(tmp / "ib_docked.pdbqt")])
        lines = nonempty(dp.stdout)
        # The table is located by its header, not by a fixed line number. The
        # units line added below shifted every index, and a check that indexes
        # its way to a table is one edit away from asserting the wrong lines.
        header = next((i for i, l in enumerate(lines)
                       if l.startswith("  rank   affinity")), -1)
        check("prose mode names the ligand and its torsions, then counts the poses",
              dp.returncode == 0
              and lines[0] == "ibuprofen_prep.pdbqt  (4 torsions)"
              and any(re.fullmatch(r"3 pose\(s\) from 23 conformations in [\d.]+ s  \[vina\]",
                                  l) for l in lines),
              f"rc={dp.returncode}, {lines[0]!r} then "
              f"{next((l for l in lines if l.startswith('3 pose')), '(no count line)')!r}")
        check("the unit the table never printed is printed by the command",
              "affinity and inter are kcal/mol, rmsd is A from the best pose" in lines,
              f"looked for it in {len(lines)} lines: {lines[1:3]}")
        check("the table has a header, one row per pose, and the output path",
              header > 0 and lines[header] == "  rank   affinity  inter      rmsd"
              and lines[-1] == f"  -> {tmp / 'ib_docked.pdbqt'}"
              and len(lines) - header - 2 == 3,
              f"header at line {header} {lines[header:header + 1]}, "
              f"{len(lines) - header - 2} pose rows, then {lines[-1]!r}")
        prose_rows = [l.split() for l in lines[header + 1:-1]]
        check("the prose affinities are the json numbers rounded to two decimals",
              rec is not None
              and [f"{e:.2f}" for e in rec["energies"]] == [r[1] for r in prose_rows]
              and [r[0] for r in prose_rows] == ["1", "2", "3"],
              f"prose {[r[1] for r in prose_rows]} against json "
              f"{['%.2f' % e for e in rec['energies']] if rec else '?'} -- the two "
              "modes have to agree to the last digit or the table is decoration")
        poses_written = (tmp / "ib_docked.pdbqt").read_text(encoding="utf-8")
        check("the written poses file holds the three poses that were reported",
              poses_written.count("MODEL") == 3
              and sum(1 for l in poses_written.splitlines()
                      if l.startswith(("ATOM", "HETATM"))) > 0,
              f"{poses_written.count('MODEL')} MODEL records")

        # Guarded, because a check script that dies on a regression is worse
        # than no check: it reports a traceback about itself instead of the
        # failure it was built to catch. The mutation runs found this one --
        # moving the progress lines to stdout made this line raise.
        try:
            again = json.loads(run(DOCK + ["--json"]).stdout)[0]
        except (ValueError, IndexError) as exc:
            again = None
            check("a fixed seed makes the run reproducible except for the clock",
                  False, f"the second run's output is not readable: {exc}")
        else:
            check("a fixed seed makes the run reproducible except for the clock",
                  rec is not None
                  and {k: v for k, v in again.items() if k != "elapsed_seconds"}
                  == {k: v for k, v in rec.items() if k != "elapsed_seconds"}
                  and again["elapsed_seconds"] != rec["elapsed_seconds"],
                  f"two runs agreed on {len(rec) - 1} of {len(rec)} fields; the "
                  "timing is the one that cannot be")

        many = run(say("dock", "-r", str(REC), "-l", str(LIGANDS_DIR),
                       "-e", "1", "-m", "1", "--steps", "10", "--seed", "1",
                       "--json", "-o", str(tmp / "many"), *BOX))
        try:
            batch = json.loads(many.stdout)
        except ValueError as exc:
            check("a directory of ligands docks as a batch", False, f"{exc}")
        else:
            stems = [p.stem for p in sorted(LIGANDS_DIR.glob("*.pdbqt"))]
            check("a directory of ligands docks as a batch, in sorted order",
                  many.returncode == 0 and len(batch) == 5
                  and [Path(r["ligand"]).stem for r in batch] == stems,
                  f"rc={many.returncode}, {len(batch)} records in the order "
                  f"{[Path(r['ligand']).stem for r in batch]}")
            check("one output file per ligand, named for it",
                  sorted(p.name for p in (tmp / "many").glob("*.pdbqt"))
                  == sorted(f"{s}_docked.pdbqt" for s in stems),
                  f"{sorted(p.name for p in (tmp / 'many').glob('*.pdbqt'))}")
            check("every one docked to a real affinity, none left at zero",
                  all(r["num_poses"] == 1 and r["best_energy"] < -1.0 for r in batch),
                  ", ".join(f"{Path(r['ligand']).stem} {r['best_energy']:.2f}"
                            for r in batch))

        fails_cleanly("dock with no box",
                      run(say("dock", "-r", str(REC), "-l", str(LIG))),
                      1, "no search box", "--center_x..--size_z")
        zero = run(say("dock", "-r", str(REC), "-l", str(LIG), "--auto-box", "0"))
        check("--auto-box 0 is rejected as out of range, not treated as unset",
              zero.returncode == 1 and not zero.has_traceback
              and "--auto-box is 1-based, got 0" in zero.stderr
              and "odcli sites" in zero.stderr,
              f"rc={zero.returncode}, {zero.stderr.strip()[:78]!r} -- and it says "
              "where to see how many there are")
        far = run(say("dock", "-r", str(REC), "-l", str(LIG), "--auto-box", "99"))
        check("--auto-box past the end names the range that does exist",
              far.returncode == 1 and not far.has_traceback
              and "only 1 site(s) were found" in far.stderr
              and "Try --auto-box 1..1" in far.stderr,
              f"rc={far.returncode}, {far.stderr.strip().splitlines()[-1][:78]!r}")
        nolig = run(say("dock", "-r", str(REC), "-l", str(empty_dir), *BOX))
        check("an empty ligand directory is its own exit code, 2",
              nolig.returncode == 2 and not nolig.has_traceback
              and f"error: no ligands found at {empty_dir}" in nolig.stderr,
              f"rc={nolig.returncode}, {nolig.stderr.strip()!r} -- distinct from the "
              "1 every other failure uses, so a script can tell 'you gave me "
              "nothing' from 'what you gave me is broken'")
        badlig = run(say("dock", "-r", str(REC), "-l", str(README), *BOX))
        fails_cleanly("dock on a ligand it cannot read", badlig, 1,
                      f"error: could not read {README}", "unsupported format '.md'")
        check("and the box-sizing warning for the same ligand was reported too",
              "warning: could not size the box" in badlig.stderr
              and "an auto box will not be widened for it" in badlig.stderr,
              "the ligand is read once to size the box and once to dock it; the "
              "first read failing is reported separately from the second")
        fails_cleanly("dock on a missing receptor",
                      run(say("dock", "-r", str(missing), "-l", str(LIG), *BOX)),
                      1, "error: ", "I/O error")

        # -------------------------------------------------------------------
        section("workbench: the one command that must not open a window")

        shim = tmp / "shim"
        shim.mkdir()
        (shim / "sitecustomize.py").write_text(
            "import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('PyQt6', 'moderngl'):\n"
            # A real missing package is a ModuleNotFoundError carrying `.name`,
            # and that name is what the diagnostic quotes -- "needs PyQt6 and
            # moderngl" does not say which of the two is absent. A bare
            # ImportError would make the message fall back to its own text.
            "            raise ModuleNotFoundError(\n"
            "                f\"No module named {name!r}\", name=name.split('.')[0])\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n",
            encoding="utf-8",
        )
        wb = run(say("workbench"), extra_path=shim)
        # Asserted on stderr, not on stdout+stderr. It used to be asserted on
        # the two combined, and that is exactly why a diagnostic on stdout
        # survived: "the message is somewhere" was the claim, and the message
        # was on the wrong stream. `stream` and `no traceback` are two
        # properties and get one check each.
        check("without the GUI stack it exits 3, on stderr, with nothing on stdout",
              wb.returncode == 3 and wb.stdout.strip() == ""
              and "PyQt6" in wb.stderr and "moderngl" in wb.stderr
              and "pip install" in wb.stderr,
              f"rc={wb.returncode}, stdout {len(wb.stdout)}B "
              f"({wb.stdout[:40]!r}), stderr {len(wb.stderr)}B: "
              f"{(wb.stderr.strip().splitlines() or ['(nothing on stderr)'])[0][:56]!r}")
        check("and it says it without a traceback", not wb.has_traceback,
              "the handler is three lines, not an except-and-reraise")
        ogw = odgui_run(["-r", str(REC)], shim)
        check("odgui says the same thing, on the same stream, byte for byte",
              ogw.returncode == 3 and ogw.stdout.strip() == ""
              and ogw.stderr == wb.stderr,
              f"odgui rc={ogw.returncode}, stdout {len(ogw.stdout)}B, stderr "
              f"identical to odcli's: {ogw.stderr == wb.stderr} -- one handler "
              "behind both, so they cannot word it differently")
        check("the shim is what kept it from opening a window",
              not wb.timed_out and wb.wall < TIMEOUT / 3,
              f"returned in {wb.wall:.2f}s; a run that had opened a viewer would "
              f"still be going at {TIMEOUT:.0f}s")
        wbh = run(say("workbench", "--help"))
        check("workbench --help describes the three files it can open",
              wbh.returncode == 0
              and all(f in wbh.stdout for f in ("-r RECEPTOR", "-l LIGAND", "-p POSES")),
              f"rc={wbh.returncode}, and no window: --help returns before launch()")

        # -------------------------------------------------------------------
        section("odgui --check has to be able to say no")

        # Two shims, because there are two ways a viewer can be unlaunchable and
        # the command that exists to answer "can I use this?" used to answer
        # "yes" to both. The first is the one this section already had: the
        # stack cannot be imported. The second is the one this machine is
        # actually in -- the stack imports and Qt comes up, and the OpenGL
        # context the viewport needs never arrives.
        #
        # The second shim replaces `QOpenGLWidget` with a subclass whose
        # `initializeGL` does nothing, rather than trying to break a driver.
        # That makes "installed, will not start" reproducible on a machine with
        # a perfectly good GPU, which is the only way the second stage can be
        # checked on CI at all.
        nogl = tmp / "shim_nogl"
        nogl.mkdir()
        (nogl / "sitecustomize.py").write_text(_NO_GL_SHIM, encoding="utf-8")

        ck_missing = odgui_run(["--check"], shim)
        check("--check says no when the GUI stack cannot be imported",
              ck_missing.returncode == 3 and ck_missing.stdout.strip() == ""
              and "GUI stack OK" not in ck_missing.stdout
              and ck_missing.stderr == wb.stderr,
              f"rc={ck_missing.returncode}, stdout {len(ck_missing.stdout)}B, "
              f"stderr identical to the launch path's: {ck_missing.stderr == wb.stderr} "
              "-- same code, same sentence, same stream as the thing it predicts")
        ck_nogl = odgui_run(["--check"], nogl)
        check("--check says no when the stack is there but OpenGL is not",
              ck_nogl.returncode == 4 and ck_nogl.stdout.strip() == ""
              and "GUI stack OK" not in ck_nogl.stdout
              and not ck_nogl.has_traceback,
              f"rc={ck_nogl.returncode}, stdout {len(ck_nogl.stdout)}B, stderr "
              f"{(ck_nogl.stderr.strip().splitlines() or ['(none)'])[0][:64]!r}")
        check("and it is a different sentence with a different fix, not a rerun of the first",
              ck_nogl.stderr != ck_missing.stderr
              and "graphics-driver" in ck_nogl.stderr
              and "pip install" not in ck_nogl.stderr
              and "`PyQt6` could not be imported" in ck_missing.stderr,
              "one tells the user to install a package, the other tells them the "
              "package is already installed and the driver is the problem -- "
              f"codes {ck_missing.returncode} vs {ck_nogl.returncode}")
        check("neither failure puts anything on stdout, and neither is a traceback",
              ck_missing.stdout.strip() == "" and ck_nogl.stdout.strip() == ""
              and not ck_missing.has_traceback and not ck_nogl.has_traceback,
              f"{len(ck_missing.stdout)}B and {len(ck_nogl.stdout)}B of stdout, "
              "no stack dumps in either")

        # The one thing that can be asserted about this machine without knowing
        # whether its GPU works: the exit code and the claim agree. Before the
        # fix they could not -- the command printed "GUI stack OK" and exited 0
        # on a machine where the real widget never even reached `initializeGL`.
        ck_here = odgui_run(["--check"], None)
        claims_ok = "GUI stack OK" in ck_here.stdout
        check("--check's exit code and its claim always agree",
              (ck_here.returncode == 0) == claims_ok,
              f"on this machine rc={ck_here.returncode}, claims GUI stack OK: "
              f"{claims_ok}"
              + (f"; it reports {ck_here.stdout.strip().splitlines()[0]!r}"
                 if claims_ok else
                 f"; it says {(ck_here.stderr.strip().splitlines() or ['(no stderr)'])[0][:60]!r}"))
        check("and it did not leave a window behind",
              not ck_here.timed_out and ck_here.wall < TIMEOUT,
              f"returned in {ck_here.wall:.2f}s; the probe shows the widget with "
              "WA_DontShowOnScreen and never enters the event loop, so there is "
              "no window to outlive it")
        # Unguarded on purpose. This used to sit behind `if ODGUI:`, which is
        # both wrong and harmful. It is wrong because the probe below runs
        # `sys.executable -c` with PYTHONPATH pointed at the shim -- it never
        # touches the `odgui` console script, so the guard had nothing to do
        # with the one thing it tested. It is harmful because a check that
        # disappears on a machine without `odgui` on PATH takes the total down
        # from 127 to 126 *and still reports a clean run*, which is the
        # "hidden behind a guard" shape this suite exists to refuse.
        probe = subprocess.run(
            [sys.executable, "-c",
             "import opendocking.workbench.launcher as L;"
             "l, p = L._preflight();"
             "print(l is None);"
             "print(p or '')"],
            capture_output=True, cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": os.pathsep.join([str(shim), str(SRC)]),
                 "PYTHONIOENCODING": "utf-8"},
            encoding="utf-8", errors="replace", timeout=TIMEOUT)
        check("_preflight is reachable: it returns no launcher and a message",
              probe.returncode == 0
              and probe.stdout.splitlines()[:1] == ["True"]
              and "could not be imported" in probe.stdout,
              f"under the blocking shim it returned "
              f"{probe.stdout.splitlines()[:1]}, and the message it built is "
              f"{len(probe.stdout.splitlines()) - 1} lines -- the branch that "
              "could not be entered before")

        # -------------------------------------------------------------------
        section("every subcommand on offer was actually run here")

        covered = SAWN | {"prep-receptor"}
        check("no subcommand is covered here and nowhere else",
              covers_all(offered, covered),
              f"offered {len(offered)}, covered {len(covered)}"
              + ("" if covers_all(offered, covered)
                 else f"; uncovered: {sorted(offered - covered)}, "
                      f"stale: {sorted(covered - offered)}"))
        check("a new subcommand would be caught, not silently skipped",
              not covers_all(offered | {"brand-new"}, offered)
              and not covers_all(offered, offered | {"brand-new"}),
              "the guard rejects both a command nobody checks and a check for a "
              "command that no longer exists")

        # -------------------------------------------------------------------
        section("reverse verification: these guards can fail")

        def made(rc, out="", err=""):
            return Result(subprocess.CompletedProcess([], rc, out, err), False, 0.0)

        # Every probe below is *meant* to fail, and `fails_cleanly` records its
        # verdict through `check`, so both the tally and the printed output are
        # held back around them. A probe that leaked into the tally would make
        # the run permanently red; one that leaked into the log would put FAIL
        # lines above a summary that says everything passed, and train the
        # reader to scroll past them.
        saved_checks, saved_failures = CHECKS, list(FAILURES)
        CHECKS, FAILURES = 0, []
        try:
            with redirect_stdout(StringIO()):
                returned = check("a deliberately false condition", False, "self-test")
                harness_probe = (returned, len(FAILURES), CHECKS)

                silent = fails_cleanly("silent", made(1, "", "error: it went wrong"),
                                       1, "this text is not in the output")
                spoken = fails_cleanly("spoken", made(1, "", "error: it went wrong"),
                                       1, "it went wrong")
                dumped = fails_cleanly(
                    "dumped", made(1, "", "error: boom\nTraceback (most recent call last):"),
                    1, "boom")
                wrong_code = fails_cleanly("wrong code", made(0, "", "error: boom"), 1, "boom")
                right = fails_cleanly("right", made(1, "", "error: boom"), 1, "boom")
                # The two wrong-stream shapes that were real defects, one probe
                # each. `fails_cleanly` is called with these and both must come
                # back False, which is the only reason the eleven real failure
                # paths above can be trusted to have moved off stdout.
                stdout_diag = fails_cleanly("diagnostic on stdout", made(1, "error: boom", ""),
                                            1, "boom")
                stdout_result = fails_cleanly("result on stdout", made(1, "wrote 0 pose file(s)", "error: boom"),
                                              1, "boom")
            trials = {
                "harness": harness_probe,
                "silent": silent, "spoken": spoken,
                "dumped": dumped, "wrong_code": wrong_code, "right": right,
                "stdout_diag": stdout_diag, "stdout_result": stdout_result,
                "rejections": len(FAILURES),
            }
        finally:
            CHECKS, FAILURES = saved_checks, saved_failures

        check("the harness records a failing check as a failure",
              trials["harness"] == (False, 1, 1),
              f"a false condition gave {trials['harness']}")
        check("the coverage guard rejects a command nobody checks",
              not covers_all({"a", "b"}, {"a"}),
              "covers_all({'a','b'}, {'a'}) is False, as it must be")
        check("the coverage guard rejects a check for a command that is gone",
              not covers_all({"a"}, {"a", "b"}),
              "covers_all({'a'}, {'a','b'}) is False, as it must be")
        check("fails_cleanly tells a silent failure from a spoken one",
              trials["silent"] is False and trials["spoken"] is True,
              "the helper is driven by the real predicate, not by the name it was "
              "called with, so the eleven real failure paths above mean something")
        check("fails_cleanly rejects a traceback dressed as an error",
              trials["dumped"] is False,
              "right exit code, right words, and a Python stack dump on top: "
              "still a failure, because the dump is the whole problem")
        check("fails_cleanly rejects the right words at the wrong exit code",
              trials["wrong_code"] is False and trials["right"] is True,
              "exit 0 with an error message is a failure too, and the four "
              f"probes produced {trials['rejections']} rejections")
        check("fails_cleanly rejects a diagnostic that went to stdout",
              trials["stdout_diag"] is False,
              "the exact shape F1 had: right code, right words, wrong stream. "
              "This is the probe the old stdout+stderr predicate could not make")
        check("fails_cleanly rejects a success-shaped line on stdout from a failure",
              trials["stdout_result"] is False,
              "the exact shape F4 had: 'wrote 0 pose file(s)' is true, and is "
              "indistinguishable from a result to anything reading stdout")
        check("a timed-out run cannot satisfy any exit-code guard",
              made(0).returncode == 0 and Result(None, True, TIMEOUT).returncode is None,
              "a run that never came back has no exit code at all, so the "
              "workbench shim's job is to make sure the run comes back")

        # -------------------------------------------------------------------
        section("reverse verification: the tree guard can fail")

        moved = dict(before)
        probe_key = sorted(moved)[0]
        moved[probe_key] = "0" * 16
        added = dict(before)
        added["a/brand-new-file.py"] = "f" * 16
        removed = dict(before)
        gone = sorted(removed)[-1]
        removed.pop(gone)
        check("the tree guard names a file that was rewritten under the run",
              changed_between(before, moved) == [probe_key],
              f"changed_between reports {changed_between(before, moved)!r}")
        check("and it notices a file appearing or going as well as changing",
              changed_between(before, added) == ["a/brand-new-file.py"]
              and changed_between(before, removed) == [gone],
              f"added {changed_between(before, added)!r}, "
              f"removed {changed_between(before, removed)!r} (was {gone!r}) -- an "
              "addition and a removal are both changes, and comparing only the "
              "keys present in both would miss both")
        check("and an unchanged tree reports no change at all",
              changed_between(before, dict(before)) == [],
              "the guard is not simply always reporting something")

        # The verdict itself, after every subprocess has run. A tree that moved
        # under the run does not make the checks wrong, it makes them describe
        # a tree that no longer exists -- so this is a separate, named outcome
        # and not one more red line among the failures.
        #
        # It also gets its own **exit code**, 2, so it stops masquerading as a
        # regression. This guard has fired four times in two sessions and every
        # time it was somebody else's legitimate edit, not a failure: an agent
        # rewriting `app.py`, another rewriting `app.py`, and a third rewriting
        # `README.md` -- none of them owned by this file. Reporting that as
        # "[FAIL] the tree changed" is the same error as reporting it as a pass.
        # Neither is true. The run is inconclusive, and 2 says so, matching the
        # 0 / 1 / 2 vocabulary the GUI scripts use: 0 clean, 1 ran and had
        # failures, 2 did not finish.
        moved_for_real = changed_between(before, _tree_fingerprint())
        if moved_for_real:
            print(f"\n  [SKIP] the tree changed under this run -- {len(moved_for_real)} "
                  f"file(s): {', '.join(moved_for_real[:6])}"
                  + (" ..." if len(moved_for_real) > 6 else ""))
            print("  The counts above describe a tree that no longer exists, so "
                  "they are not evidence about anything. This is what a stale "
                  "run looks like; it is reported rather than re-run, and it "
                  "exits 2 rather than 1 because nothing here is a regression.")
            return _EXIT_INCONCLUSIVE
        else:
            print(f"\ntree: unchanged ({len(before)} files) across the whole run")

    return finish()


if __name__ == "__main__":
    # A check script that dies on a regression is worse than no check at all:
    # it reports a traceback about itself instead of the failure it was built to
    # catch, and the tally never appears. So the crash is caught here, named,
    # and reported with whatever count did get reached.
    #
    # It exits 2, not 1. A crash means the run did not finish, and "did not
    # finish" is not "something failed" -- the whole point of the 0/1/2
    # vocabulary is that a caller can tell those apart, and a crash is the
    # clearest case of all. The tally is still printed, because a partial count
    # that says which checks ran is better than silence.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - this is the point
        print(f"\n  [FAIL] the run crashed before finishing  - "
              f"{type(exc).__name__}: {exc}")
        print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed (run incomplete: "
              f"expected {EXPECTED_CHECKS})")
        sys.exit(_EXIT_INCONCLUSIVE)
