"""Is the installed copy of `opendocking` the one this tree ships?

Run:  F:\\python310\\python.exe scripts\\installed_copy_check.py

# Why this exists

A third copy of the package lives at the interpreter's `site-packages`, and it is
a *build output* of this tree rather than part of it. It was measured three
generations behind: five modules of `workbench/` absent, `app.py` 63,997 B
smaller than the tree's, and one public class (`TermsWorker`) gone from it. A user
running `odgui` from that copy gets an older workbench and no gate anywhere said
so. This is the Python-side companion to the extension-binary currency check
another gate owns; it deliberately asserts **nothing** about `_dockpy.pyd`, so
the two checks cannot contradict each other and neither duplicates the other.

# The invariant, and why it is narrower than byte-equality

**What is asserted: every module this tree's package contains exists in the
installed copy, every intra-package import this tree performs resolves there --
the import's target *and* the package initialiser the importing module is itself
reached through -- and every public name this tree's own entry points reference
is defined there.**

Byte-equality would be the wrong claim, and measurably so:

* The installed copy is a build output. CI rebuilds and reinstalls on every job,
  so a fresh install is *supposed* to differ from a working copy -- in build
  stamps, in `__pycache__`, and in the compiled extension. A byte-equality gate
  would be red on every clean run and would be switched off within a week.
* The one copy in this tree measured with a stray `~_pycache__/` directory beside
  it -- a PowerShell redirect accident, the same family as `$null` -- which is a
  byte difference that says nothing about whether the package works.
* What actually decides whether `odgui` runs is not whether bytes match but
  whether **every name it reaches for is there**. A missing module or a renamed
  class is an `ImportError` or an `AttributeError` at run time; a differing byte
  in an unchanged line is nothing at all.

So the gate asks the question whose failure is a crash, and declines to ask the
one whose failure is a diff. That is the whole trade: narrower claim, and every
red means the user is broken.

# What it does when there is no install

That is a normal machine, not a failure. An absent install records a **skip** with
the reason and still counts toward the pinned total, so the total is the same
number on a maintainer's machine, on CI, and on a laptop that has never run
`pip install`. A gate that is red on a clean checkout is a gate nobody runs.

# The self-test, and why it is inside this file

On CI this gate is a tautology, and it should be said plainly rather than
discovered again: every job rebuilds and reinstalls the wheel from the same
source in the same job, so the installed copy *is* the tree by construction and
the comparison cannot fail. It would go green six times over a question with no
wrong answer available. That is worse than a gate that skips, because it
pretends to be measuring.

It is not in CI for that reason, and the reason is written into the workflow.
But a gate that is not in CI and does not run on a maintainer's machine either
is a gate that measures nothing anywhere, so the three checks below run **in
this file, on every invocation, in both environments**. They build a
deliberately stale copy with `stale_install_fixture.py` and require this file's
own predicate to *reject* it, naming the missing module.

That is the difference between the gate being trusted and the gate being
merely present: a predicate nobody has ever seen refuse anything has not been
shown to be able to refuse anything. The measurement above still runs against
the real install; the self-test is what makes the green beside it mean
something. The builder is a deliverable, not a gate -- it is handed to
`run_script` and defines no check of its own.

**Two fixtures, because the predicate makes two claims and one fixture tests
one.** The first drops a module a sibling imports, which is what proves the
target half. The second drops a package initialiser, which is what proves the
half that was missing: this predicate used to ask only whether an import's
*target* was present, and a copy with no `tests/__init__.py` satisfied it while
`tests/make_data.py` sat in it running four relative imports that cannot resolve
without it. Check 3 was **green** on that copy, measured before the fix. One
fixture would have left the second claim untested on every run forever, which is
the failure mode a self-test exists to prevent.

# What it does when the extension binary is stale

Nothing. It never opens `_dockpy.pyd`, never compares it, and never reports on
it. Staleness there is another gate's subject, detected by mutating the binary
in place; this one stays out of the way so the two cannot disagree.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
PKG_REL = Path("dock-py", "python", "opendocking")
FIXTURE_BUILDER_NAME = "stale_install_fixture.py"
STALENESS_NAME = "STALENESS.txt"

#: How many checks this file records, in every environment. Measured from a run
#: with an install present, and unchanged without one, because a skip is a
#: result:
#:
#:   1  the installed copy was located (a skip, counted, when there is none)
#:   2  every module this tree's package contains exists in the installed copy
#:   3  every intra-package import this tree performs resolves in the install,
#:      initialiser of the importing module's own package included
#:   4  every IMPORTED name this tree references is defined in the install
#:   5  the installed copy holds no module this tree does not
#:   6  the fixture's staleness is declared, named and inspectable
#:   7  the fixture is stale in exactly the way it says it is
#:   8  this gate's own predicate refuses two built fixtures, naming the module
#:   9  the tally accounts for every result, skips included
#:
#: `6 -> 9` is the self-test, and it is three checks rather than one on purpose:
#: the claim is not "a fixture exists" but "this predicate rejects a stale
#: copy, for a reason a user would recognise". A single check would pass on a
#: fixture that was stale for the wrong reason, or one that was stale but whose
#: rejection this file could not name. Check 8 builds **two** fixtures for the
#: same reason: the predicate has two claims, and one fixture would leave the
#: second of them unmeasured on every run.
#:
#: The call-site census, declared in the file it counts, measured by walking this
#: file's own syntax tree and compared against this block. **It was NOT declared
#: anywhere until 2026-10-02**: the auditor classified this file as a gate, walked
#: it, and printed `NOT DECLARED ANYWHERE`. Nothing about this gate was in doubt
#: -- it holds a pin, it is in `INVENTORY`, and its `ok`/`bad` pairs are exactly
#: what the nine above describe. The declaration that was missing is a different
#: number, and "the file is clearly a gate" is not an answer to a question about a
#: declaration.
#:
#: 3 of the 17 sites are unconditional and 14 are guarded. The 14 are the two
#: loops -- over the tree's modules and over the install's -- so most of this
#: gate's reach is a function of what is on disk, and a site that stops being
#: reached is a site that cannot be wrong. That is why the digest is here and not
#: only the pair of counts: moving a site into a different branch leaves the
#: total alone and changes which inputs it is reachable from.
#: GATE-DECLARE 1
#: sites: 3 unconditional + 14 guarded
#: guards: sha256:e4359dd9266706bf6a861071784c603551511a69ce47f879aab66d3804fbd9fd
EXPECTED_CHECKS = 9

RESULTS: list[tuple[str, str, str]] = []


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def skip(name: str, why: str) -> None:
    RESULTS.append(("SKIP", name, why))
    print(f"[SKIP] {name}\n       {why}")


def section(t: str) -> None:
    print()
    print(f"-- {t}")


def find_installed() -> Path | None:
    """The installed copy, or None. `OD_INSTALLED_COPY` overrides, so this gate
    can be pointed at a copy that has been mutated for a test without touching
    the real `site-packages` -- which another agent is using as a fixture."""
    override = os.environ.get("OD_INSTALLED_COPY")
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    spec = importlib.util.find_spec("opendocking")
    if spec and spec.origin:
        p = Path(spec.origin).parent
        if (p / "__init__.py").is_file() and p != ROOT / PKG_REL:
            return p
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        p = Path(base) / "opendocking"
        if (p / "__init__.py").is_file() and p != ROOT / PKG_REL:
            return p
    return None


def tree_modules(pkg: Path) -> dict[str, Path]:
    return {p.relative_to(pkg).as_posix(): p
            for p in sorted(pkg.rglob("*.py"))
            if "__pycache__" not in p.parts}


def public_names(path: Path) -> set[str] | None:
    try:
        tree = ast.parse(path.read_bytes().decode("utf-8", errors="replace")
                         .lstrip("\ufeff"))
    except (OSError, SyntaxError):
        return None
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not n.name.startswith("_"):
                out.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    out.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            if not n.target.id.startswith("_"):
                out.add(n.target.id)
    return out


def referenced_names(path: Path, pkg: Path) -> set[tuple[str, str]]:
    """(module path relative to `pkg`, name) pairs this module reaches for.

    **What `n.level` is.** `ast.ImportFrom.level` is an **int**: the number of
    leading dots. `from . import x` is `level == 1`; `from ..pkg import y` is
    `level == 2`; an absolute `from opendocking.core import Grid` is `level == 0`
    and is not a relative import at all. The previous version called
    `len(n.level)` on it, which raises `TypeError` unconditionally -- this
    function could never have completed a single call on a module with a
    relative import.

    **The arithmetic.** For a module at `<pkg>/a/b/mod.py`, the package the
    dots select is `path.parents[level - 1]`:

        level 1  ->  path.parents[0]  ==  <pkg>/a/b   (the module's own package)
        level 2  ->  path.parents[1]  ==  <pkg>/a
        level 3  ->  path.parents[2]  ==  <pkg>

    and `n.module` is then resolved **under that anchor**, dotted-to-path. So
    `from .workbench import app` inside `<pkg>/workbench/app.py` anchors at
    `<pkg>/workbench` and resolves `<pkg>/workbench/workbench`, which is what
    Python actually does -- the anchor is the *package*, not the package root,
    and getting that wrong is what the previous version did: it built a bare
    `workbench.py` with no package prefix, which never matches any real module.

    Resolution is to a real file on disk (`<x>.py` or `<x>/__init__.py`), so a
    name that cannot be resolved is not reported rather than being guessed at.
    """
    try:
        tree = ast.parse(path.read_bytes().decode("utf-8", errors="replace")
                         .lstrip("\ufeff"))
    except (OSError, SyntaxError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom) or n.level < 1:
            continue
        anchor = path.parents[n.level - 1]
        base = anchor / n.module.replace(".", "/") if n.module else anchor
        target = None
        if base.is_dir() and (base / "__init__.py").is_file():
            target = base / "__init__.py"
        elif base.with_suffix(".py").is_file():
            target = base.with_suffix(".py")
        if target is None:
            continue
        try:
            rel = target.relative_to(pkg).as_posix()
        except ValueError:
            continue
        out |= {(rel, a.name) for a in n.names if a.name != "*"}
    return out


def _package_chain(pkg: Path, path: Path) -> list[str]:
    """The package initialisers `path`'s own directory is reached through.

    Every directory from the package root down to the one holding `path`, and
    only where the **tree** has an `__init__.py` there. The tree is the thing
    being mirrored, so a directory that is a PEP 420 namespace package in the
    tree is not demanded of the copy: requiring more of the copy than the tree
    has would be a check that goes red on a correct install.
    """
    out: list[str] = []
    for parent in path.parents:
        if parent == pkg.parent:
            break
        init = parent / "__init__.py"
        if init.is_file():
            out.append(init.relative_to(pkg).as_posix())
        if parent == pkg:
            break
    return out


def import_surface(pkg: Path, tmods: dict[str, Path],
                   imods: dict[str, Path]) -> tuple[dict[str, set[str]], list[str]]:
    """`(references, unresolved)`: what this tree's modules reach for inside the
    package, and which of those a given copy cannot satisfy.

    One predicate, two callers: check 3 asks the question of the real install
    and check 8 asks it of a fixture this file just built. A check of the check
    that re-implements the thing it is checking proves nothing about it.

    **`unresolved` is not only the import's target.** Asking only that was the
    hole, and it had a name: a module at `<pkg>/tests/make_data.py` runs
    `from .. import core`, which needs `<pkg>/tests/__init__.py` for `tests` to
    be a package at all -- and with that one file absent from a copy this
    function reported nothing, so check 3 passed on an install whose `tests`
    could not be imported. Measured before the fix: dropping
    `tests/__init__.py` left check 3 **green**. So every module that initiates a
    relative import now also requires the initialiser of each directory it is
    itself imported through, from its own directory up to the package root --
    which is the module that initiated the import, and the one thing the target
    test structurally cannot see.

    A module with no relative import is not asked for a chain: nothing of its
    own performs an intra-package import, and check 2 already accounts for the
    file itself.
    """
    refs: dict[str, set[str]] = {}
    unresolved: list[str] = []
    for rel, path in sorted(tmods.items()):
        want = referenced_names(path, pkg)
        if not want:
            continue
        bucket = refs.setdefault(rel, set())
        for init in _package_chain(pkg, path):
            if init not in imods:
                # The file is named package-relative, as every other message in
                # this file names one, so a reader can go and look at it and so
                # the self-test can match on the name rather than on prose.
                unresolved.append(
                    f"{init} is absent, so {rel} cannot be imported at all: the "
                    f"package it is reached through is not a package in this copy"
                )
        for target, name in want:
            bucket.add(f"{target}:{name}")
            if target not in imods:
                unresolved.append(f"{rel} imports {name} from {target} (absent)")
    return refs, unresolved


def run_script(name: str, *args: str):
    """Run a script in this directory as a subprocess, repo root as cwd.

    The one helper this repository has for handing a script to another script.
    It exists as a named call rather than an inline `subprocess.run` because
    `check_scripts_declare.py` decides that a file is a *deliverable* -- a
    measurement handed to a gate -- by finding this call in a gate's source, and
    a deliverable that launched any other way is a measurement with no oracle
    saying it was ever run.
    """
    import subprocess

    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def parse_staleness(text: str) -> dict[str, str]:
    """The `stale.*` lines of a `STALENESS.txt`, and nothing else.

    Prefixed lines only, so prose can be added above without breaking a
    consumer, and a key that is absent stays absent rather than defaulting --
    a default here would let a fixture that declares nothing look like a fixture
    that declares nothing is wrong.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("stale."):
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def choose_drop(pkg: Path, tmods: dict[str, Path]) -> str | None:
    """A module to leave out of the fixture: one a sibling actually imports.

    Derived, not hand-written. A named list would rot the moment the tree
    renamed or removed the entry, and a fixture that drops a module nothing
    imports still exercises check 2 while missing the failure a user would
    actually meet -- an `ImportError` from a live import line, which is check 3
    and the one worth proving.

    **Two exclusions, and the first one is a bug this function had.** A package
    initialiser looks referenced to `referenced_names` and is not: inside
    `<pkg>/cli.py`, `from . import __version__` resolves to `<pkg>/__init__.py`,
    because `from . import X` anchors on the *package* and the name is an
    attribute of its initialiser. So `<pkg>/__init__.py` reads as "a module a
    sibling imports", sorts first, and gets chosen -- producing a fixture whose
    missing file is the package's own front door. That is not a stale install,
    it is a broken one, and it would have made this self-test pass for a reason
    that has nothing to do with the claim. `__init__.py` and `__main__.py` are
    therefore excluded at any depth: the first is a package rather than a module
    *of* one, and the second is an entry point rather than something imported.

    **The preference for `workbench/`.** This gate's claim is about the
    workbench a user reaches by running `odgui`, which is the staleness
    `docs/VERIFICATION.md` recorded and the reason this file exists. Sorted over
    everything else the first candidate is `cli.py`, whose absence breaks `odcli`
    and not `odgui`: the same machinery, a different product. The preference
    falls back to any referenced non-initialiser module, so it cannot leave the
    fixture with nothing to do.
    """
    referenced: set[str] = set()
    for rel, path in sorted(tmods.items()):
        for target, _name in referenced_names(path, pkg):
            referenced.add(target)
    excluded = {"__init__.py", "__main__.py"}
    candidates = sorted(r for r in tmods
                        if r in referenced and Path(r).name not in excluded)
    if not candidates:
        return None
    in_workbench = [r for r in candidates if r.startswith("workbench/")]
    return (in_workbench or candidates)[0]


def choose_initialiser(pkg: Path, tmods: dict[str, Path]) -> str | None:
    """The package initialiser to leave out of the second fixture.

    **One that no import line in the tree reaches as a *target*,** so a refusal
    it provokes can only have come from the package-chain half of the
    predicate. This is the part that took three attempts: with the package
    root's `__init__.py` the two claims are not separable, because `from .
    import __version__` inside `cli.py` already reaches it as a target, so a
    self-test that accepted "the predicate named it" was satisfied by the half
    that already worked. Mutation M1 -- the chain assertion switched off --
    passed that self-test. On this tree `tests/__init__.py` is the initialiser
    no target reaches and `tests/make_data.py` is the module whose imports
    cannot resolve without it, so dropping it separates the two.

    Derived rather than named, for `choose_drop`'s reason: a hand-written entry
    rots the moment the tree renames or moves it, and a fixture that stops
    testing anything is worse than no fixture. Falls back to the package root
    when the tree has no unreached initialiser; the caller then requires the
    refusal to come from the chain anyway, so a fallback cannot quietly turn
    this into the first fixture a second time.

    `None` when the tree has no `__init__.py` at all, which is a red in the
    caller rather than a skip: an assertion nothing can exercise is a check
    that would go green without having tested anything.
    """
    inits = sorted(r for r in tmods if Path(r).name == "__init__.py")
    if not inits:
        return None
    reached = {t for path in tmods.values() for t, _n in referenced_names(path, pkg)}
    unreached = [i for i in inits if i not in reached]
    if unreached:
        return unreached[0]
    return "__init__.py" if "__init__.py" in inits else inits[0]


def build_copy(dropped: list[str]) -> tuple[Path | None, str | None]:
    """A temp copy of the tree's package without `dropped`. `(dir, problem)`.

    One builder for both fixtures, so the second cannot drift from the first in
    how it is made. `problem` is a reason string when there is nothing to test
    and `None` when there is; it is never treated as success by a caller. A
    directory that was created and then found unusable is removed here rather
    than left for a reader to trip over.
    """
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="od-stale-install-"))
    # The builder's name is spelled out here rather than read from a constant,
    # because the census looks for this exact call text to decide that the
    # builder is a deliverable a gate actually runs.
    flags = [a for d in dropped for a in ("--drop", d)]
    proc = run_script("stale_install_fixture.py", "--out", str(out), *flags)
    if proc.returncode != 0:
        shutil.rmtree(out, ignore_errors=True)
        return None, (f"{FIXTURE_BUILDER_NAME} exited {proc.returncode} building a "
                      f"copy without {', '.join(dropped)}: "
                      f"{(proc.stderr or proc.stdout).strip()[:300]}")
    if not (out / STALENESS_NAME).is_file():
        shutil.rmtree(out, ignore_errors=True)
        return None, (f"the builder reported success but wrote no "
                      f"{STALENESS_NAME} into {out}, so there is no account of "
                      f"what makes the copy old")
    return out, None


def build_self_test_fixture(pkg: Path, tmods: dict[str, Path]):
    """Build the stale copy. Returns `(dir, dropped, problem)`.

    `problem` is a reason string when there is nothing to test, and `None` when
    there is. It is never silently treated as success: the three callers turn a
    `problem` into a skip that counts, because "the builder did not run" is not
    the same claim as "the predicate is sound".
    """
    target = choose_drop(pkg, tmods)
    if target is None:
        return None, [], ("no module of this tree's package is imported by a "
                          "sibling, so there is no import a stale copy could "
                          "break and nothing to drop")
    where, problem = build_copy([target])
    if problem:
        return None, [], problem
    return where, [target], None


def check_the_fixture_declares_its_staleness(where, dropped, problem) -> None:
    name = "the fixture's staleness is declared, named and inspectable"
    if problem:
        skip(name, problem)
        return
    meta = parse_staleness((where / STALENESS_NAME).read_text(encoding="utf-8"))
    want = {"stale.kind": "stale", "stale.dropped": ",".join(dropped)}
    wrong = {k: (v, want[k]) for k, v in want.items() if meta.get(k) != want[k]}
    if wrong or not meta.get("stale.source_digest"):
        bad(name, f"{where / STALENESS_NAME} declares {meta} against the "
                  f"expected {want}"
                  + (f"; wrong: {wrong}" if wrong else "")
                  + ". A fixture whose difference from the tree is not written "
                    "down is a second copy of the package, and a reader cannot "
                    "tell a deliberate one from one that rotted")
        return
    ok(name, f"{where / STALENESS_NAME} names {meta['stale.dropped']} as absent, "
             f"kind={meta['stale.kind']}, source digest {meta['stale.source_digest'][:16]}, "
             f"built by {FIXTURE_BUILDER_NAME} from this tree")


def check_the_fixture_is_stale_in_exactly_that_way(pkg, tmods, where, dropped, problem) -> None:
    name = "the fixture is stale in exactly the way it says it is"
    if problem:
        skip(name, problem)
        return
    fmods = tree_modules(where / "opendocking")
    # `d in fmods` is the defect, not `d not in fmods`. The first version of
    # this check had it the other way round and went red on a *correct* fixture,
    # which is the whole hazard of a check whose own polarity is the claim: it
    # looks like it is testing the fixture and it is testing its own author.
    still_there = [d for d in dropped if d in fmods]
    undeclared = sorted(set(fmods) - (set(tmods) - set(dropped)))
    ghost = [d for d in dropped if d not in tmods]
    if still_there or undeclared or ghost:
        bad(name, f"the fixture does not match its own account. Named as absent "
                  f"but still here: {still_there}. Here but not named as "
                  f"differing: {undeclared}. Named as absent but not in the tree "
                  f"at all: {ghost}. A self-test that runs against a copy stale "
                  f"for a different reason would go green without ever having "
                  f"tested anything")
        return
    ok(name, f"{len(fmods)} of the tree's {len(tmods)} module(s) present, and the "
             f"{len(dropped)} declared absent really are: {dropped}. No other "
             f"difference between the fixture and the tree")


def refuses_a_copy_with_no_initialiser(pkg: Path,
                                       tmods: dict[str, Path]) -> tuple[str | None, str]:
    """The second half of check 8: this predicate against a copy with no package
    initialiser. Returns `(reason, detail)`; `reason` is `None` when it refused.

    The half that was missing, and the one a single fixture could never reach:
    every fixture `choose_drop` can build drops a module a sibling imports, and
    a missing `__init__.py` is reached by no import line at all -- it is what the
    importing module is *reached through*. A predicate that looks only at import
    targets cannot see it, and check 3 went green on such a copy before this
    existed.
    """
    target = choose_initialiser(pkg, tmods)
    if target is None:
        return ("this tree's package has no __init__.py at any depth, so the "
                "claim that an install missing one is caught has nothing to be "
                "caught here, and a check that cannot be exercised is a check "
                "that would go green without having tested anything"), ""
    where, problem = build_copy([target])
    if problem:
        return f"the second fixture could not be built: {problem}", ""
    try:
        _refs, unresolved = import_surface(pkg, tmods, tree_modules(where / "opendocking"))
    finally:
        # A temp directory, and the run is over with it. Through shutil rather
        # than deleted by hand, so a crash mid-run cannot take the tree with it.
        shutil.rmtree(where, ignore_errors=True)
    # **The refusal has to come from the chain, not merely name the file.** The
    # package root's `__init__.py` is reachable as an ordinary import target
    # (`from . import __version__` in `cli.py`), so "the predicate named it" is
    # satisfied by the half that already worked; requiring the chain-form entry
    # is what makes this a test of the missing half rather than a second reading
    # of the first one. Measured: with the chain assertion switched off, the
    # name still appeared and this check was green. That is the mutation that
    # made the requirement here.
    chain = [u for u in unresolved if "cannot be imported at all" in u]
    hit = next((u for u in chain if target in u), None)
    if hit is None:
        return (f"a copy of this tree's package with {target} removed is one "
                f"import this gate cannot see: the predicate answered "
                f"{len(unresolved)} unresolved reference(s) and named it in "
                f"none of the {len(chain)} that came from a package chain. A "
                f"refusal that does not come from the chain proves the target "
                f"half again, which is the half the first fixture already "
                f"covers"), ""
    return None, (f"and the same predicate names {target} in a copy built without "
                  f"it, from the package chain rather than from any import line "
                  f"({hit})")


def check_this_gate_refuses_it(pkg, tmods, where, dropped, problem) -> None:
    """The check of the check: this file's own predicate, on a stale copy.

    Runs the *same* `tree_modules` comparison as check 2 against the fixture,
    and requires it to name the dropped module -- and requires check 3's own
    resolution logic to see the import that a user would hit. If this passes
    because the fixture happened to break something unrelated, the detail text
    says so and the check is red.

    Two fixtures, two claims. The first is the dropped module above; the second
    is a copy with no package initialiser, which no import line reaches and so
    cannot be reached by a fixture that drops an imported module. Both go
    through `import_surface`, the function check 3 calls, because a second
    implementation of the predicate would be a second thing to be wrong.
    """
    name = "this gate's own predicate refuses the fixture, naming the module"
    if problem:
        skip(name, problem)
        return
    fmods = tree_modules(where / "opendocking")
    missing = sorted(set(tmods) - set(fmods))
    reason: str | None = None
    detail = ""
    if sorted(dropped) != missing:
        reason = (f"the predicate found {missing} where the fixture declares "
                  f"{sorted(dropped)} it dropped. A refusal that does not match "
                  f"the declared difference is a refusal for some other reason, "
                  f"and this check is asking whether *this* one works")
    else:
        _refs, unresolved = import_surface(pkg, tmods, fmods)
        broken = [u for u in unresolved if any(d in u for d in missing)]
        if not broken:
            reason = (f"{missing} is absent, but no live import line in the tree "
                      f"reaches it, so nothing a user runs would fail over it. "
                      f"The module was chosen as one a sibling imports; that is "
                      f"no longer true and the choice is derived rather than "
                      f"hard-coded, which is how this can be caught")
    if reason is None:
        why, extra = refuses_a_copy_with_no_initialiser(pkg, tmods)
        if why is not None:
            reason = why
        else:
            detail = (f"the same comparison that passes on a current copy "
                      f"returns {missing} here, and the import a user would hit "
                      f"is {broken[0]}: an ImportError in `odgui`, named by this "
                      f"gate, {extra}")
    if reason is not None:
        bad(name, reason)
    else:
        ok(name, detail)


def measure(pkg: Path, tmods: dict[str, Path], inst: Path | None) -> None:
    """Checks 1-5: the question about the real install, or five counted skips.

    Split out of `main` for one reason that is not tidiness: the self-test has
    to run whether or not an install was found, and the pinned total is the same
    number in both environments. While this body lived inside `main` next to an
    early `return`, the two paths could not share a tail, and the self-test
    would have been reachable only on a machine that happened to have an install.
    """
    if inst is None:
        why = ("no installed `opendocking` outside this working copy: an absent "
               "install is a normal machine, not a failure, and this records a "
               "skip rather than a red. Point OD_INSTALLED_COPY at a copy to "
               "ask the question anyway")
        skip("the installed copy was located", why)
        for n in ("every module this tree's package contains exists in the "
                  "installed copy",
                  "every intra-package import this tree performs resolves in "
                  "the install",
                  "every public name this tree references is defined in the "
                  "install",
                  "the installed copy holds no module this tree does not"):
            skip(n, why)
        return

    imods = tree_modules(inst)
    ok("the installed copy was located",
       f"{inst}; {len(imods)} module(s) against this tree's {len(tmods)}. The "
       f"compiled extension is not read, compared or reported on by this gate: "
       f"staleness there is another gate's subject")

    section("the import surface this tree declares")
    missing = sorted(set(tmods) - set(imods))
    if missing:
        bad("every module this tree's package contains exists in the installed copy",
            f"{len(missing)} module(s) in this tree are absent from the install: "
            + ", ".join(f"{m} ({tmods[m].stat().st_size} B)" for m in missing[:8])
            + (" ..." if len(missing) > 8 else "")
            + ". An import of any of these is an ImportError in the user's "
              "`odgui`, and nothing else in this repository would say so")
    else:
        ok("every module this tree's package contains exists in the installed copy",
           f"all {len(tmods)} module(s) present in {inst}")

    refs, unresolved = import_surface(pkg, tmods, imods)
    if unresolved:
        bad("every intra-package import this tree performs resolves in the install",
            f"{len(unresolved)} unresolved: " + "; ".join(unresolved[:6])
            + (" ..." if len(unresolved) > 6 else ""))
    else:
        ok("every intra-package import this tree performs resolves in the install",
           f"{sum(len(v) for v in refs.values())} intra-package reference(s) "
           f"from {len(refs)} module(s) all resolve against the install, "
           f"including the package initialiser each of those modules is itself "
           f"reached through -- the target test alone cannot see that one, and "
           f"it is the half that was missing")

    gone = []
    for rel, want in sorted(refs.items()):
        src = tmods[rel]
        inst_src = imods.get(rel)
        if inst_src is None:
            continue
        have = public_names(inst_src)
        if have is None:
            continue
        src_pub = public_names(src) or set()
        # a name this tree's own importer asks for, and that the install lacks
        for spec in want:
            mod, _, name = spec.rpartition(":")
            tgt = imods.get(mod)
            if tgt is None:
                continue
            tpub = public_names(tmods.get(mod, tgt)) or set()
            if name in tpub and name not in (public_names(tgt) or set()):
                gone.append(f"{rel} needs {name} from {mod}")
    if gone:
        bad("every IMPORTED name this tree references is defined in the install",
            f"{len(gone)} name(s) this tree declares and imports, that the "
            f"install does not define: " + "; ".join(sorted(set(gone))[:6])
            + ". A defined class that vanished is an AttributeError at run time "
              "and a silent absence here")
    else:
        ok("every IMPORTED name this tree references is defined in the install",
           f"{sum(len(v) for v in refs.values())} referenced name(s) checked "
           f"against the installed modules that define them")

    surplus = sorted(set(imods) - set(tmods))
    if surplus:
        bad("the installed copy holds no module this tree does not",
            f"{len(surplus)}: " + ", ".join(surplus[:8])
            + ". Drift in this direction is how a deleted module keeps working "
              "on one machine and nowhere else")
    else:
        ok("the installed copy holds no module this tree does not",
           f"none: the install's {len(imods)} module(s) are a subset of this "
           f"tree's {len(tmods)}")


def main() -> int:
    print("Is the installed copy the one this tree ships?")
    print()
    pkg = ROOT / PKG_REL
    tmods = tree_modules(pkg)
    inst = find_installed()
    section("the installed copy")
    measure(pkg, tmods, inst)

    section("the self-test: can this gate tell a stale copy from a current one?")
    where, dropped, problem = build_self_test_fixture(pkg, tmods)
    check_the_fixture_declares_its_staleness(where, dropped, problem)
    check_the_fixture_is_stale_in_exactly_that_way(pkg, tmods, where, dropped, problem)
    check_this_gate_refuses_it(pkg, tmods, where, dropped, problem)
    if where is not None:
        # The fixture is a temp directory and the run is over. Removed through
        # shutil rather than deleted by hand so the path is not left for a
        # reader to trip over, and so a crash mid-run cannot take the tree with
        # it: nothing outside the temp dir is ever written.
        shutil.rmtree(where, ignore_errors=True)

    tally()
    return finish()


def tally() -> None:
    section("the tally")
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    name = "the tally accounts for every result, skips included"
    if np_ + nf + ns == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(name, f"{np_} + {nf} + {ns} = {len(RESULTS)}, and this check is the "
                 f"{len(RESULTS) + 1}th, so the run reaches the declared "
                 f"{EXPECTED_CHECKS} -- the same with an install and without "
                 f"one, because a skip is a result")
    else:
        bad(name, f"recorded {len(RESULTS)} results ({np_}+{nf}+{ns}) before "
                  f"this check, which must be {EXPECTED_CHECKS - 1}")


def finish() -> int:
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {np_} passed, {nf} failed, {ns} skipped, {len(RESULTS)} checks "
          f"(expected {EXPECTED_CHECKS})")
    print("    three numbers that never sum to a verdict: a pass count alone "
          "cannot say an install was found")
    if nf:
        print(f"RESULT: FAIL -- {nf} of {EXPECTED_CHECKS} checks failed")
        return 1
    print(f"RESULT: OK -- {np_} passed, {ns} could not be measured here")
    return 0


def report_crash(exc: BaseException) -> int:
    """A run that did not finish says so, in the same words as a run that did.

    Added because this file crashed once, *after* printing a PASS line and a
    plausible FAIL line, and a reader could not tell a finished run from a
    stopped one without reading the exit code. The summary is now unreachable
    unless every check ran, and an unhandled failure prints `DID NOT FINISH`
    with the exception, which no finished run can print.
    """
    import traceback

    traceback.print_exc()
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- STOPPED EARLY after {len(RESULTS)} recorded result(s): "
          f"{np_} passed, {nf} failed, {ns} skipped")
    print(f"RESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}")
    print("    a finished run prints OK or FAIL above and never this line; "
          f"{len(RESULTS)} of {EXPECTED_CHECKS} checks ran")
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_exc))
