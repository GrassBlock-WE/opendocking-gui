"""Pre-push / pre-release doc audit: link integrity and machine-specific leakage.

Not part of the test suite -- this is a repository hygiene check. Run it before
publishing:

    python scripts/check_repo_docs.py

Exits 0 when every relative Markdown link resolves and no absolute local path
leaked into a tracked file; 1 otherwise, with one line per problem.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Skip build output and the Rust target dir entirely. `opendocking-gui/` is a
# snapshot of this tree; scanning it would just check every file twice.
SKIP_DIRS = {
    "target", "dist", "dist-gpu", ".git", ".pytest_cache", "__pycache__",
    "opendocking-gui",
}
DOC_SUFFIXES = {".md", ".py", ".toml", ".cfg", ".ini", ".rs"}

# `[text](target)` where the target is a relative path (no scheme, no anchor).
LINK = re.compile(r"\[(?:[^\]]*)\]\(\s*([^)\s]+?)(?:\s+\"[^\"]*\")?\s*\)")
# Absolute local paths. The lookbehind matters: without it, the `h:\n` inside
# "install them with:\n" reads as a drive letter on drive H.
#
# The lookahead exists for a different reason. `H:\s*(\d+)` inside a raw
# string is a regex, not a path on drive H, and this file had already grown a
# one-off carve-out for exactly that class (`h:\n`). The general rule is that a
# regex escape is a *single* letter: if another letter or digit follows it, the
# text is a path. So `C:\Scripts\tools` is a path (`S` followed by `c`) and
# `H:\s*` is a regex (`s` followed by `*`), and the same holds for `{`, `(`,
# end of line, and every other non-word character after the escape letter.
DRIVE_ESCAPE = "sSdDwWbBAZ"
ABS_PATH = re.compile(
    r"(?<![A-Za-z0-9_\\])[A-Za-z]:[\\/]"
    rf"(?![{DRIVE_ESCAPE}][^A-Za-z0-9_])"                      # a regex escape
    r"|(?<![A-Za-z0-9_\\])\\\\[A-Za-z0-9][\w.-]*\\[\w.-]+[\\/]"  # \\host\share\
    r"|(?<![A-Za-z0-9_/])/(?:home|Users)/[A-Za-z0-9_.-]+/"       # /home/x/ or /Users/x/
)

# The pattern above is a gate, so it gets held to the same standard as the
# gates it guards: each case below must both fire and stay quiet, and the
# counting has to be able to tell a wrong answer from a missing answer. A
# leakage check that silently stops matching is worse than no check, because
# the number it reports still looks healthy.
_PATH_CASES = (
    # (line, expected number of hits, what it is)
    (r'ROOT = "C:\Users\33654\Desktop"', 1, "a drive path in a normal string"),
    (r'ROOT = r"C:\Users\33654\Desktop"', 1, "a drive path in a raw string"),
    (r'x = "d:/opt/data"  # forward slashes', 1, "a drive path with /"),
    (r"hit = r'WITHOUT polar H:\s*(\d+)'", 0, "a regex ending in \s*"),
    (r"hit = r'gap H:\d{2}'", 0, "a regex ending in \d"),
    (r"install them with:\n", 0, "prose ending a word with a colon"),
    (r'see "install them with:" below', 0, "prose quoted after a colon"),
    (r"p = r'C:\Scripts\tools'", 1, "a real path whose first letter is a regex escape name"),
    (r"ROOT = Path(r'\\host\share\dir')", 1, "a UNC path"),
    (r"cfg = '/home/33654/x'", 1, "a POSIX home path"),
    (r"cfg = '/usr/home/33654/x'", 0, "a POSIX path that is not a home directory"),
)


def _verify_path_pattern() -> list[str]:
    """Prove the leakage pattern both fires and stays quiet. Returns problems."""
    bad = []
    for line, expected, why in _PATH_CASES:
        got = len(ABS_PATH.findall(line))
        if got != expected:
            bad.append(
                f"the absolute-path pattern is wrong on {why}: expected "
                f"{expected} hit(s), got {got} in {line!r}"
            )
    return bad


def _heading_anchors(text: str) -> set[str]:
    """The anchor ids GitHub generates for the ATX headings in `text`.

    Lower-cased, punctuation dropped, spaces to hyphens -- the same rule
    GitHub uses. CJK is kept, which is why a Chinese table of contents can be
    checked at all.
    """
    anchors: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*$", line)
        if m is None:
            continue
        slug = m.group(1).strip().lower()
        slug = re.sub(r"[^\w一-鿿\- ]", "", slug)
        slug = slug.replace(" ", "-")
        anchors.add(slug)
    return anchors


def iter_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES:
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        out.append(path)
    return sorted(out)


def main() -> int:
    problems: list[str] = []
    checked_links = 0

    for path in iter_files():
        rel = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: not valid UTF-8")
            continue

        # --- absolute local paths -----------------------------------------
        # This file is skipped: it spells out the very patterns it hunts for.
        if rel.name != pathlib.Path(__file__).name:
            for number, line in enumerate(text.splitlines(), 1):
                for hit in ABS_PATH.findall(line):
                    problems.append(f"{rel}:{number}: machine-specific path {hit!r}")

        # --- relative links and in-page anchors --------------------------
        if path.suffix.lower() != ".md":
            continue
        anchors = _heading_anchors(text)
        for number, line in enumerate(text.splitlines(), 1):
            for target in LINK.findall(line):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                if target.startswith("#"):
                    # In-page anchor: it must match a heading in this file.
                    checked_links += 1
                    if target[1:] not in anchors:
                        problems.append(
                            f"{rel}:{number}: no heading matches anchor {target}"
                        )
                    continue
                if target.startswith("#") or target.startswith(":"):
                    continue
                checked_links += 1
                bare = target.split("#", 1)[0]
                if not bare:
                    continue
                if not (path.parent / bare).exists():
                    problems.append(
                        f"{rel}:{number}: broken link -> {target}"
                    )

    # --- YAML / CITATION must actually parse -----------------------------
    # A workflow that does not parse fails every run in 0 seconds with
    # "This run likely failed because of a workflow file issue", which is easy
    # to miss and impossible to debug from the Actions tab. Caught here
    # instead, before the push.
    yaml_checked = 0
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed; skipping YAML validation")
    else:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".yml", ".yaml", ".cff"}:
                continue
            if SKIP_DIRS & set(path.relative_to(ROOT).parts):
                continue
            yaml_checked += 1
            rel = path.relative_to(ROOT)
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                problems.append(f"{rel}: does not parse as YAML -- {first}")

    print(f"scanned {len(iter_files())} files, {checked_links} relative link(s), "
          f"{yaml_checked} YAML/CFF file(s), "
          f"{len(_PATH_CASES)} path-pattern cases")
    problems.extend(_verify_path_pattern())
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(f"  {line}")
        return 1
    print("no broken links, no machine-specific paths, the path pattern agrees "
          "with every case, all YAML parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
