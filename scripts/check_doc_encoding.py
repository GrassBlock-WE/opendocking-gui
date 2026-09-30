"""Check every doc for the encoding damage PowerShell 5.1 can cause.

A single `Get-Content -Raw` / `Set-Content -Encoding UTF8` round trip on a
Chinese document reads the file as the ANSI codepage and writes the result
back as UTF-8. The result *looks* like mojibake but is not fully reversible, so
it is much cheaper to catch it than to reconstruct the file.

Run this after any doc edit that did not go through the write/edit tools.
"""

import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Build output and the release snapshot are skipped: same files, twice.
SKIP_DIRS = {
    "target", "dist", "dist-gpu", ".git", ".pytest_cache", "__pycache__",
    "opendocking-gui",
}

REPLACEMENT = "�"
CJK = re.compile("[\u4e00-\u9fff]")
# A CJK character immediately followed by '?' is the other signature: the byte
# that was lost where GBK could not represent the decoded character.
LOST = re.compile("[\u4e00-\u9fff]\?")


def documents() -> list[pathlib.Path]:
    """Every Markdown file and LICENSE, discovered rather than listed.

    A hard-coded list silently stops covering a document the moment someone
    adds it, which is exactly when the check is needed most.
    """
    found = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md"}:
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        found.append(path.relative_to(ROOT))
    licence = ROOT / "LICENSE"
    if licence.is_file():
        found.append(licence.relative_to(ROOT))
    return found


total = 0
docs = documents()
for name in docs:
    p = ROOT / name
    text = p.read_text(encoding="utf-8")
    bad = text.count(REPLACEMENT) + len(LOST.findall(text))
    total += bad
    print(f"{str(name):28s} {p.stat().st_size:6d} B  CJK {len(CJK.findall(text)):5d}  damaged {bad}")

print(f"\n{len(docs)} documents, total damaged: {total}")
sys.exit(1 if total else 0)
