"""Does the wheel carry every module this tree's package contains?

Run:  F:\\python310\\python.exe scripts\\wheel_payload_check.py

# Why

A wheel is built from the package directory, so anything outside it does not
ship. Several modules moved into the package in the last hours, and for a while
the panel that consumes one of them imported a module that existed in neither
location. The failure mode is a panel that is dead in the installed product with
no exception and no log line, because the missing thing was never imported in
the tree -- only in the copy the tree's source happened to have.

# The invariant, and why byte-equality would be the wrong claim

**Every module the tree's package contains is present in the wheel, every
intra-package import the tree performs resolves inside it, the wheel holds no
module the tree does not, and both declared console scripts resolve to a module
and a callable the wheel actually carries.**

Byte-equality would be wrong, and measurably so. The wheel legitimately
contains artefacts the tree does not: the compiled `_dockpy.pyd`, a `RECORD`, a
`WHEEL`, `METADATA`, and a CycloneDX SBOM this build emits. Six of the twenty
nine files in the wheel are not `.py` and none of them is in the package
directory. A byte-equality gate would be red on every correct build, and a red
on every correct build is a gate nobody runs.

What actually decides whether the installed product works is not whether the two
agree byte for byte but whether **every name the package reaches for is present
in the artefact that will be installed**. A missing module is an `ImportError`;
a differing byte in an unchanged line is nothing. So the claim is narrower, and
every red it produces means the product is broken.

# No wheel built

A machine that has not built is a normal machine, not a failure. That records a
**skip** for each check, with the reason, and the skips count toward the pinned
total -- so the number is the same on a maintainer's machine, on CI, and on a
laptop that has never run `maturin build`.

# What this does not touch

It never opens `_dockpy.pyd`, never installs anything, and never writes to
`site-packages`. It reads a wheel file and prints a list.
"""

from __future__ import annotations

import ast
import io
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "dock-py" / "python" / "opendocking"

#: How many checks this file records, in every environment.
#:
#:   1  a wheel was located (a skip, counted, when none has been built)
#:   2  every module this tree's package contains is in the wheel
#:   3  every intra-package import the tree performs resolves inside the wheel
#:   4  the wheel holds no module the tree does not
#:   5  both declared console scripts resolve to a module and callable present
#:   6  the tally accounts for every result, skips included
#: **Measured from a green run, and pinned by the owner.** 6 is 1 (a wheel was
#: located) + 4 payload checks + 1 (the tally). It was certified by running a
#: copy of the real wheel through four mutations, each rejected by a different
#: check with its own message: a deleted module names the file and the imports
#: left unresolved, a phantom module names itself on the surplus check, a
#: renamed `def main` says the callable is not defined at the top level of the
#: module that is supposed to have it, and the must-stay-green cases are a
#: wheel with no compiled extension (still 6/6) and a wheel predating the tree
#: (4 skips, no red). The number is the same with a wheel, without one, and
#: with a stale one, which is the only reason a maintainer's run and CI's run
#: can be compared at all.
#:
#: The call-site census, declared in the file it counts, measured by walking this
#: file's own syntax tree and compared against this block. **It was NOT declared
#: anywhere until 2026-10-02**: the auditor classified this file as a gate, walked
#: it, and printed `NOT DECLARED ANYWHERE`. The pin above and this census answer
#: different questions -- the pin is how many results a run must record, the
#: census is how many result sites the source has -- and the gap between them is
#: where a check that was deleted without its number being touched would hide.
#:
#: 1 of the 13 sites is unconditional and 12 are guarded, and the 12 are the
#: three loops over the wheel's members, the tree's modules and the declared
#: entry points. A gate whose reach is a function of an archive is exactly the
#: gate where "the run passed" and "the run looked at the right things" come
#: apart, and the guards digest is what keeps the second one a fact about the
#: source rather than about the wheel that happened to be on disk.
#: GATE-DECLARE 1
#: sites: 1 unconditional + 12 guarded
#: guards: sha256:bf93bf582bebc21307acd988d92186bf6faaa2288c273387626a98697aafe03d
EXPECTED_CHECKS = 6

RESULTS: list[tuple[str, str, str]] = []

NO_WHEEL = ("no built wheel found: a machine that has not run `maturin build` is "
            "a normal machine, not a failure, and this records a skip rather "
            "than a red. Point OD_WHEEL at a .whl to ask the question anyway")


def ok(n, d): RESULTS.append(("PASS", n, d)); print(f"[PASS] {n}\n       {d}")
def bad(n, d): RESULTS.append(("FAIL", n, d)); print(f"[FAIL] {n}\n       {d}")
def skip(n, w): RESULTS.append(("SKIP", n, w)); print(f"[SKIP] {n}\n       {w}")


def section(t):
    print()
    print(f"-- {t}")


def find_wheel() -> Path | None:
    env = os.environ.get("OD_WHEEL")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    for d in (ROOT / "dock-py" / "target" / "wheels", ROOT / "_wheelout",
              ROOT / "dock-py" / "target" / "release"):
        if d.is_dir():
            ws = sorted(d.glob("*.whl"), key=lambda p: p.stat().st_mtime)
            if ws:
                return ws[-1]
    return None


def tree_mods():
    return {p.relative_to(PKG).as_posix()
            for p in PKG.rglob("*.py") if "__pycache__" not in p.parts}


def src_of(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8", errors="replace").lstrip("\ufeff")
    except OSError:
        return None


def relative_imports(path: Path, pkg: Path):
    """(module, name) pairs this module reaches for inside the package.

    `ast.ImportFrom.level` is an int -- the number of leading dots -- and the
    package those dots select is `path.parents[level - 1]`: level 1 is the
    module's own package, level 2 one above it. The previous version of this
    logic in a sibling file called `len(n.level)` and could not execute at all.
    """
    text = src_of(path)
    if text is None:
        return set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
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


def parse_console_scripts(text: str) -> dict[str, str]:
    """The `[console_scripts]` section of an entry_points.txt, read explicitly.

    The format is an INI file with one `name = module:attr` line per script.
    The previous version handed it to `configparser.ConfigParser`, whose
    `get()` signature is `get(section, option, ...)` -- so passing a section and
    a dict asked for an *option* named `{}`, and the dict then reached a string
    method downstream. **A parsed field was never unwrapped.**

    So this says what the format is instead of coercing until it stops
    complaining: one section header, then `key = value` lines, keys trimmed,
    values taken verbatim after the first `=`. Anything outside that shape is
    skipped rather than guessed at.
    """
    out: dict[str, str] = {}
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line[1:-1].strip() == "console_scripts"
            continue
        if not in_section or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out



def defines(text: str | None, name: str) -> bool:
    if text is None:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name == name:
                return True
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return True
    return False


def tally_and_report():
    section("the tally")
    a = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    b = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    c = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    name = "the tally accounts for every result, skips included"
    if a + b + c == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(name, f"{a} + {b} + {c} = {len(RESULTS)}, and this check is the "
                 f"{len(RESULTS) + 1}th, so the run reaches the declared "
                 f"{EXPECTED_CHECKS} -- the same with a wheel built and without "
                 f"one, because a skip is a result")
    else:
        bad(name, f"recorded {len(RESULTS)} results ({a}+{b}+{c}) before this "
                  f"check, which must be {EXPECTED_CHECKS - 1}")


def finish() -> int:
    a = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    b = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    c = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {a} passed, {b} failed, {c} skipped, {len(RESULTS)} checks "
          f"(expected {EXPECTED_CHECKS})")
    print("    three numbers that never sum to a verdict: a pass count alone "
          "cannot say a wheel was found")
    if b:
        print(f"RESULT: FAIL -- {b} of {EXPECTED_CHECKS} checks failed")
        return 1
    print(f"RESULT: OK -- {a} passed, {c} could not be measured here")
    return 0


def main() -> int:
    print("Does the wheel carry every module this tree's package contains?")
    print()
    whl = find_wheel()
    section("the wheel")
    payload_checks = ("every module this tree's package contains is in the wheel",
                      "every intra-package import the tree performs resolves "
                      "inside the wheel",
                      "the wheel holds no module the tree does not",
                      "both declared console scripts resolve inside the wheel")
    if whl is None:
        for n in ("a wheel was located",) + payload_checks:
            skip(n, NO_WHEEL)
        tally_and_report()
        return finish()
    # A wheel built before the newest source file cannot say anything about the
    # tree it is being compared with -- and it will happily agree with it, which
    # is the dangerous part. This is the same rule as the no-wheel case: an
    # unmeasurable state is a skip that counts, never a pass that lies. Found the
    # hard way: an eight-hour-old wheel answered "8 modules missing" about a tree
    # that had all eight, and nothing in the output said which wheel it had read.
    newest = max((p.stat().st_mtime for p in PKG.rglob("*.py")), default=0.0)
    stale = newest - whl.stat().st_mtime
    if stale > 0:
        why = (f"the newest source in the package is "
               f"{int(newest - newest % 1)}s, the wheel is "
               f"{int(whl.stat().st_mtime - whl.stat().st_mtime % 1)}s, so the "
               f"wheel predates the tree by {int(stale)}s: its payload is not a "
               f"statement about this tree. Rebuild and re-run")
        for n in payload_checks:
            skip(n, why)
        ok("a wheel was located",
           f"{whl} ({whl.stat().st_size} B), but it is {int(stale)}s older than "
           f"the package it is being asked about, so nothing below can be "
           f"settled from it. Read only: nothing is installed, nothing is "
           f"written to site-packages, and the compiled extension is never "
           f"opened")
        tally_and_report()
        return finish()
    ok("a wheel was located",
       f"{whl} ({whl.stat().st_size} B), built {int(-stale)}s after the newest "
       f"source file in the package, so its payload is a statement about this "
       f"tree. Read only: nothing is installed, nothing is written to "
       f"site-packages, and the compiled extension is never opened")
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
        body = {n: z.read(n) for n in names if n.endswith(".py")}
        ep = [n for n in names if n.endswith("entry_points.txt")]
        eps = z.read(ep[0]).decode("utf-8", "replace") if ep else ""
    tree = tree_mods()
    inwheel = {n[len("opendocking/"):] for n in body if n.startswith("opendocking/")}
    missing = sorted(tree - inwheel)
    section("the payload")
    if missing:
        bad("every module this tree's package contains is in the wheel",
            f"{len(missing)} module(s) are in the tree and NOT in the wheel: "
            + ", ".join(missing)
            + ". A wheel is built from the package directory, so each of these "
              "is a dead import in the installed product with no exception and "
              "no log line")
    else:
        ok("every module this tree's package contains is in the wheel",
           f"all {len(tree)} module(s) of {PKG} are present; the wheel carries "
           f"{len(inwheel)} module(s)")
    bad_refs = []
    refs = 0
    for rel in sorted(tree):
        for tgt, name in relative_imports(PKG / rel, PKG):
            refs += 1
            if tgt not in inwheel:
                bad_refs.append(f"{rel} imports {name} from {tgt} (not in the "
                                f"wheel)")
    if bad_refs:
        bad("every intra-package import the tree performs resolves inside the wheel",
            f"{len(bad_refs)} of {refs}: " + "; ".join(bad_refs[:6]))
    else:
        ok("every intra-package import the tree performs resolves inside the wheel",
           f"{refs} relative import(s) from the package, every target module "
           f"present in the wheel")
    surplus = sorted(inwheel - tree)
    if surplus:
        bad("the wheel holds no module the tree does not",
            f"{len(surplus)}: " + ", ".join(surplus[:8]))
    else:
        ok("the wheel holds no module the tree does not",
           f"none; the wheel's {len(inwheel)} module(s) are exactly this tree's "
           f"{len(tree)}")
    section("the entry points")
    scripts = parse_console_scripts(eps)
    # `name = package.module:attr`. Two things have to be true, and both are
    # read out of the wheel's OWN source rather than out of pyproject.toml: the
    # module is a module the wheel carries, and the callable is defined at that
    # module's top level. Declaring the intent and carrying the outcome are
    # different files, and a payload gate has to look at the second one.
    broken: list[str] = []
    resolved: dict[str, str] = {}
    # `inwheel` holds paths relative to the package root (the wheel's own
    # `opendocking/` prefix is stripped when it is built), so a dotted module
    # name has to lose the package component before the dots become slashes.
    # Getting this wrong makes every entry point read "no such module in the
    # wheel" against a wheel that plainly carries the module.
    pkgname = PKG.name
    for cmd, target in sorted(scripts.items()):
        mod, _, attr = target.partition(":")
        mod, attr = mod.strip(), attr.strip()
        dotted = mod[len(pkgname) + 1:] if mod.startswith(pkgname + ".") else mod
        rel = next((c for c in (f"{dotted.replace('.', '/')}.py",
                                f"{dotted.replace('.', '/')}/__init__.py")
                    if c in inwheel), None)
        if rel is None:
            broken.append(f"{cmd} -> {target}: no such module in the wheel")
        elif not attr:
            broken.append(f"{cmd} -> {target}: names no callable")
        else:
            raw = body.get(f"{pkgname}/{rel}")
            if not defines(raw.decode("utf-8", "replace") if raw else None, attr):
                broken.append(f"{cmd} -> {target}: {attr} is not defined at the "
                              f"top level of {rel}")
            else:
                resolved[cmd] = f"{rel}:{attr}"
    if not scripts:
        bad("both declared console scripts resolve inside the wheel",
            f"the wheel declares no console_scripts section: {eps!r}")
    elif broken:
        bad("both declared console scripts resolve inside the wheel",
            f"{len(broken)} of {len(scripts)}: " + "; ".join(broken))
    else:
        ok("both declared console scripts resolve inside the wheel",
           "; ".join(f"{c} -> {t} (module and callable both present, read out "
                     f"of the wheel)" for c, t in sorted(resolved.items()))
           + f". {len(scripts)} declared, all resolvable. Their absence is a "
             f"silent failure unless someone runs them, which is why this is "
             f"read out of the wheel and not out of pyproject.toml")
    tally_and_report()
    return finish()


def report_crash(exc: BaseException) -> int:
    import traceback
    traceback.print_exc()
    print()
    print(f"--- STOPPED EARLY after {len(RESULTS)} recorded result(s)")
    print(f"RESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}")
    print(f"    a finished run prints OK or FAIL above and never this line; "
          f"{len(RESULTS)} of {EXPECTED_CHECKS} checks ran")
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _e:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_e))
