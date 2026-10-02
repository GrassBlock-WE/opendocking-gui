"""Build a deliberately stale copy of the package, from this tree.

Run:  F:\\python310\\python.exe scripts\\stale_install_fixture.py --out DIR [--drop MODULE]

# What this is for

`installed_copy_check.py` answers "is the copy the user's `odgui` will import the
one this tree ships?". On CI the answer is a tautology: every job rebuilds and
reinstalls the wheel from the same source in the same job, so the copy is the
tree by construction and the gate would go green six times over a comparison
that cannot fail. The gate guards a *developer's machine*, where the copy in
`site-packages` lags the working tree by however many generations it happens to.

So the gate needs something to be wrong about. This builds it.

# The staleness is declared, not accidental

The failure mode this is written against is the obvious one: `cp -r` the package,
delete a file, and call the result a fixture. That directory is a second copy of
the package with no account of why it differs, and the next person cannot tell a
deliberate fixture from an install that rotted. So the difference is written into
the fixture as `<out>/STALENESS.txt` -- human-readable prose *and* a
`stale.*` machine-readable block, in one file, so there is exactly one place to
look and nothing to infer.

`--drop` is the only knob, it is repeatable, and it names a module *relative to
the package root* in POSIX form (`workbench/keys.py`). Dropping nothing is a
legitimate build and yields a **current** copy, which is the other half of the
pair: the same builder with one flag between them is the whole experiment.

# What it deliberately does not copy

**No compiled extension.** `_dockpy.pyd` is not read, hashed or copied, and this
script says so in the fixture it writes. Three reasons, and the first is the
decisive one: the gate that uses this fixture asserts nothing about the
extension, so carrying one would be cargo; a fixture that is stale in a second,
unrelated dimension is a worse fixture, because a red could then mean either;
and the payload is a compiled binary for one interpreter and one platform, which
has no business in a fixture that runs on every CI job. Everything copied is
`*.py`, and `__pycache__` is skipped, so no stale bytecode can answer for a
module that is not there.

# Determinism

The same tree and the same flags produce a byte-identical directory, including
`STALENESS.txt`: nothing in the written content is a timestamp, a hostname or a
path outside the repository. The only thing that varies between two builds is
the directory's own mtime. That is what makes the fixture assertable -- a
difference between two fixtures is then a difference in the tree, not in when
they were built.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_REL = Path("dock-py", "python", "opendocking")
STALENESS_NAME = "STALENESS.txt"
PKG_NAME = "opendocking"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def source_modules(pkg: Path) -> dict[str, Path]:
    """Every `.py` the tree's package contains, keyed by POSIX-relative path."""
    return {p.relative_to(pkg).as_posix(): p
            for p in sorted(pkg.rglob("*.py"))
            if "__pycache__" not in p.parts}


def digest_of(mods: dict[str, Path]) -> str:
    """A digest of the *content* of the modules copied, in path order.

    Content rather than sizes: two builds of one tree are identical, and a
    changed module in either direction moves this number, which is what a
    reader uses to ask "is this fixture still built from the tree I have?".
    """
    h = hashlib.sha256()
    for rel in sorted(mods):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(mods[rel].read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def staleness_text(source: Path, mods: dict[str, Path], dropped: list[str],
                   digest: str) -> str:
    """The one file that says what makes this copy old. Prose, then `stale.*`.

    Both halves are in the same file on purpose. A machine-readable sidecar
    would be a second place to look and a second thing to keep in step; a prose
    file alone would force a gate to parse an English sentence to learn which
    module is missing, which is how a fixture's meaning starts drifting from its
    contents.
    """
    kept = len(mods) - len(dropped)
    lines = [
        "THIS COPY IS DELIBERATELY STALE. It is not a mistake, not a leftover, and",
        "not a second copy of the package that nobody wrote down.",
        "",
        f"what makes it old: {len(dropped)} module(s) the tree has are absent here",
    ]
    if dropped:
        for rel in dropped:
            size = mods[rel].stat().st_size
            lines.append(f"    absent  {rel}   ({size} B in the tree)")
    else:
        lines.append("    nothing -- this build dropped no module, so it is CURRENT.")
        lines.append("    It exists so the gate can be run against a copy that is not")
        lines.append("    old, by the same builder and with no other difference.")
    lines += [
        "",
        f"built from    : {source}",
        f"modules       : {kept} copied of the tree's {len(mods)}",
        "extension     : none copied. `_dockpy.pyd` is not read, hashed or carried",
        "                here, and the gate that uses this fixture asserts nothing",
        "                about the compiled extension.",
        "bytecode      : `__pycache__` and `*.pyc` are skipped, so a stale .pyc can",
        "                never stand in for a module that is not here.",
        "",
        "The only difference between this package and the tree is the list at the",
        "top. Everything else is byte-identical to the source.",
        "",
        "# machine-readable, one key per line, do not reformat",
        f"stale.kind={'stale' if dropped else 'current'}",
        f"stale.dropped={','.join(dropped)}",
        f"stale.source_modules={len(mods)}",
        f"stale.copied_modules={kept}",
        f"stale.source_digest={digest}",
    ]
    return "\n".join(lines) + "\n"


def parse_staleness(text: str) -> dict[str, str]:
    """The `stale.*` block, and nothing else.

    Parsed by prefix on a line, so a reader can add prose above without
    breaking a consumer, and a missing key is a missing key rather than a
    silently-defaulted one.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("stale."):
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def build(out: Path, drops: list[str], quiet: bool = False) -> int:
    source = ROOT / PKG_REL
    mods = source_modules(source)

    unknown = [d for d in drops if d not in mods]
    if unknown:
        print(f"cannot build a stale copy: {unknown} "
              f"{'is not' if len(unknown) == 1 else 'are not'} a module of "
              f"{source.relative_to(ROOT)}", file=sys.stderr)
        avail = ", ".join(sorted(mods))
        print(f"the package has {len(mods)} modules: {avail}", file=sys.stderr)
        return 2

    target = out / PKG_NAME
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    digest = digest_of(mods)
    for rel, path in mods.items():
        if rel in drops:
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)

    (out / STALENESS_NAME).write_text(
        staleness_text(source, mods, drops, digest), encoding="utf-8", newline="\n")

    if not quiet:
        print(f"built {target}")
        print(f"  {len(mods) - len(drops)} of {len(mods)} module(s) copied")
        print(f"  dropped: {', '.join(drops) if drops else '(nothing -- this is a current copy)'}")
        print(f"  the account of what makes it old: {out / STALENESS_NAME}")
        print(f"  source content digest: {digest[:16]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a deliberately stale copy of the package from this tree.")
    ap.add_argument("--out", required=True, type=Path,
                    help="directory to build into; created if absent")
    ap.add_argument("--drop", action="append", default=[], metavar="MODULE",
                    help="a package-relative module to leave out, e.g. "
                         "workbench/keys.py. Repeatable. Omit for a current copy.")
    ap.add_argument("--print-staleness", action="store_true",
                    help="print STALENESS.txt and nothing else")
    args = ap.parse_args()

    if args.print_staleness:
        text = (args.out / STALENESS_NAME).read_text(encoding="utf-8")
        sys.stdout.write(text)
        return 0
    return build(args.out, args.drop)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - report, do not vanish
        import traceback

        traceback.print_exc()
        print(f"RESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
