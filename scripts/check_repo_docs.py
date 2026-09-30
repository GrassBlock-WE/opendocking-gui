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
ABS_PATH = re.compile(
    r"(?<![A-Za-z0-9_\\])[A-Za-z]:[\\/]"          # C:\  or  C:/
    r"|(?<![A-Za-z0-9_\\])\\\\[A-Za-z0-9][\w.-]*\\[\w.-]+[\\/]"  # \\host\share\
    r"|(?<![A-Za-z0-9_/])/(?:home|Users)/[A-Za-z0-9_.-]+/"       # /home/x/ or /Users/x/
)


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
          f"{yaml_checked} YAML/CFF file(s)")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(f"  {line}")
        return 1
    print("no broken links, no machine-specific paths, all YAML parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
