"""Pre-push / pre-release doc audit: link integrity and machine-specific leakage.

Run:  python scripts/check_repo_docs.py
      OD_REPO_DOCS_ROOT=<tree> python scripts/check_repo_docs.py   # a copy, for mutations

Not part of the test suite -- this is a repository hygiene check. Run it before
publishing.

# What this file found, and what it could not say

Every finding used to be one line in a single `problems` list, and the run's only
verdict was its exit code. That list had **seven** sources -- an undecodable
file, a machine-specific path, a dead in-page anchor, a broken link, a YAML file
that does not parse, and the two self-tests that hold the leakage pattern and the
comment/docstring mask to their own cases -- and a reader who wanted to know
*which* of the seven was red, and a census that wanted to know *how many* results
this gate produces, both had to parse the output to find out.

So each of the seven is now one named result, and two more were added: a check
that the declared scope is still a scope, and the check that counts them. Every
result states what it proves **and what it does not** -- the house rule, and the
only reason a reader can tell a green from a silence. It is enforced by
signature rather than by comment: `check()` takes `proves` and `not_proves` as
required positional arguments, so a result added without the second clause is a
`TypeError` at run time rather than a quieter gate.

**No predicate was weakened, and the set of files read is unchanged.** The seven
are the same seven tests over the same files with the same patterns; what changed
is that each is separately red-able. The one thing that is new is the *pairing*:
a dead anchor and a missing target were both "a broken link" before, and they are
separate results now, because a target that is not on disk and a heading that is
not in the file are two different defects and a red that cannot say which one
happened is half a verdict. The counter `LINK_TARGETS` is still the combined
number the header printed before, so the header line is unchanged.

# The pin, and the number that is deliberately *not* pinned

`EXPECTED_CHECKS` is a constant, and the last check asserts the run reaches it.
That number is a property of this file. The number of files in scope is a
property of the working tree, and those are different claims.

**The count is not pinned, and this is measured rather than preferred.** The
briefing that asked for this work reported `scanned 149 files`. The same code, on
the same tree, minutes later with no edit to this file, printed `scanned 132
files`. Seventeen files left the scope while nothing that decides the scope
changed, which is exactly what a count of a *moving* set does. A pin of 149
would have been red before the edit that made it, and 132 would be red the next
time anybody adds a script -- so it would not be a record of anything except the
instant it was typed. `check_scripts_declare.py` calls a number nothing compares a
transcribed number; a number that compares against a working tree is worse,
because it goes red for the right reason and gets "fixed" by writing down
whatever the tree said that day.

What is pinned instead is the **scope decision**: `DOC_SUFFIXES` and `SKIP_DIRS`
are the statement of what this gate reads, and the check named "the declared scan
scope is answered" fails when a declared class has quietly emptied. It failed
when this was written, on `.cfg`: no file in the tree carries that suffix, and a
gate whose scope silently narrows to what happens to be present reports a
healthier number than it is. The count itself is still printed, from the list
that was actually walked -- the old code called `iter_files()` a second time to
get it, so the number it printed was a fresh walk rather than the one it had
scanned.

# The skip, and why it is a result

The one check that can become a skip is the YAML one, and it becomes one on an
interpreter with no PyYAML. `skip()` appends to `RESULTS` and therefore counts
toward the total, so a machine that cannot parse YAML reaches the same nine as one
that can, with the composition of the number moving and the verdict not. A gate
that reached its total by declining to look would be reaching it by not running
its checks, which is the failure `check_scripts_declare.py` exists to catch.

That is also a repair. The old code printed "PyYAML not installed; skipping YAML
validation" and then, with no problems found, printed "all YAML parses" and
exited 0 -- a claim about a measurement it had not made. The prose summary below
is now assembled from what the run actually did, so it cannot say that.

# What this file does NOT do

It does not run the other check scripts, it does not fetch `http(s)` or
`mailto:` link targets, and it does not check anything about a file whose suffix
is not in `DOC_SUFFIXES` -- a binary, a `.json`, a `.txt`. Each result above says
so again in its own words, because a reader who has only the summary line is
reading the weakest thing this file prints.
"""

from __future__ import annotations

import ast
import io
import os
import pathlib
import re
import sys
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where the run reads. `OD_REPO_DOCS_ROOT` exists so a *copy* of a tree can be
#: measured, which is the only way to prove a check can go red without writing to
#: the tree the check guards -- see `provenance_appendix.py --root` and
#: `check_text_encoding.py`'s `OD_ENCODING_SCAN_ROOT` for the same two
#: precedents. Nothing in this file ever writes to either root.
SCAN_ROOT = pathlib.Path(os.environ.get("OD_REPO_DOCS_ROOT") or ROOT).resolve()

# Skip build output and the Rust target dir entirely. `opendocking-gui/` is a
# snapshot of this tree; scanning it would just check every file twice.
SKIP_DIRS = {
    "target", "dist", "dist-gpu", ".git", ".pytest_cache", "__pycache__",
    "opendocking-gui",
}
DOC_SUFFIXES = {".md", ".py", ".toml", ".cfg", ".ini", ".rs"}

#: Declared scope with nothing in it, and the one reason each is still declared.
#:
#: A suffix whose class is empty is a silent narrowing: the run prints a smaller
#: number and the summary line is unchanged, so the only evidence is a reader
#: doing arithmetic. `.cfg` is empty on this tree and is kept, because dropping it
#: would narrow the gate's reach on the strength of a moment's evidence and a
#: `setup.cfg` appearing next month would then be unread. The reason is a value
#: rather than a comment so that it can go stale *measurably*: the scope check
#: below fails on a suffix that has a reason and has since gained a file, so
#: this dict cannot quietly become a list of excuses.
EMPTY_SUFFIX_REASONS = {
    ".cfg": "no file in this tree carries it; the suffix stays declared so a "
            "`setup.cfg` would be read the day one appears, and removing it would "
            "have narrowed the gate's reach on the strength of one measurement",
}

# --------------------------------------------------------------------------
# The declared census, and the number that is a different number.
#
# `check_scripts_declare.py` reads these three lines out of *comment tokens* and
# the `#:` prefix is load-bearing, not decoration: the head pattern is
# `^#:\s*GATE-DECLARE\s+(\d+)\s*$` against the comment string with its `#`
# intact, so a plain `# GATE-DECLARE 1` is silently not a declaration at all. The
# first version of this block was written that way, and the auditor reported
# `check_repo_docs.py: derived 8+4, declared 0+0` -- it had fallen through to
# the snapshot and was comparing my correct numbers against a `0`. A block that
# is not recognised is worse than no block, because the snapshot then answers
# for the file and the file's own declaration is what nothing reads.
#
# The numbers below were derived by running that file's own `_sites_of` and
# `_guards_digest` over this source, not by counting by hand: a hand-typed
# census is the second thing that gets forgotten, which is the whole reason the
# auditor exists.
#
# **8 unconditional and 4 guarded, against a pin of 9 results -- and the two
# numbers are not supposed to match.** The 8 are the eight content checks: seven
# `check(...)` sites that can only pass or fail, plus the eighth, the YAML one,
# which is a call site in the tree and an `if`-guarded *slot* at run time. It
# reaches the census as unconditional because it early-`return`s rather than
# living in an `else` arm, which is a choice about how the skip reads and not
# about the census. The 4 guarded are the two arms of `ok()`/`bad()` inside
# `check()` (guard string `if cond`) and the two arms of `tally()`'s `if/else`.
# Each pair is **one result, not two**: a branch has two arms and records one of
# them, so a reader who expects the site count to equal the pin is wrong in a
# way they could not otherwise check. The digest is what catches the change the
# two counts are blind to -- a site moving between guard shapes with the total
# still 12.
#
# `check_scripts_declare.py` prefers this block over its generated snapshot, and
# it fails on a gate that appears in both -- so the snapshot entry for this file
# has to be deleted by whoever owns that file. It is a generated region; deleting
# it is what `--pin` would do.
#: How many results a run records, including the one that counts them.
#:
#: **Constant by construction, and the point of the last check.** Nothing that
#: records a result is inside a loop, so the total is 9 on a green tree, 9 on a
#: red one, and 9 on an interpreter with no PyYAML -- where one result is a skip
#: rather than a pass. A gate whose total moves when its verdict moves is a gate
#: measuring the tree's mood; this one measures its own shape.
#:
#: The `scanned N files` in the header is deliberately *not* in this number. See
#: the module docstring: it is a property of the working tree, it was observed to
#: move from 149 to 132 with no edit to this file, and a pin against it would go
#: red for a reason that has nothing to do with whether this gate works.
#:
#: The declared census is the three comment lines immediately above; the two
#: numbers are different numbers and that is deliberate, not a discrepancy.
#: GATE-DECLARE 1
#: sites: 8 unconditional + 4 guarded
#: guards: sha256:e9848fd041bd9b79e7562e3c2b47e398dc276d75821b2b57d28811ce9a2bbc25
EXPECTED_CHECKS = 9  # measured from a green run on this tree: 8 passed + this one, exit 0

#: `(tag, name, proves, does-not-prove, detail)`, in the order they were
#: recorded. The tag is one of `PASS` / `FAIL` / `SKIP`, and the summary prints
#: the three as three numbers that are never added together.
RESULTS: list[tuple[str, str, str, str, str]] = []


def _emit(line: str) -> None:
    """Print one line on a console that may not be able to encode it.

    **Measured on the machine this gate runs on**: bare `sys.stdout.encoding` is
    `gbk` here. A problem message quotes a path and a link target straight out of
    a file, so a single character outside cp936 and `print` raises
    `UnicodeEncodeError` -- and it raises *while reporting the finding*, which is
    the worst moment: the traceback is not a verdict and a gate that died is a
    gate that reported nothing. Same remedy and same reasoning as
    `provenance_appendix.py`'s `_emit`: write the character as a `\\xNN` escape
    rather than take the run down, so the instrument says what it could not
    render instead of rendering less and saying nothing.
    """
    try:
        print(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="backslashreplace").decode(enc, errors="replace"))


def ok(name: str, proves: str, not_proves: str, detail: str = "") -> None:
    RESULTS.append(("PASS", name, proves, not_proves, detail))
    _emit(f"[ok  ] {name}")
    _emit(f"        proves: {proves}")
    _emit(f"        not:    {not_proves}")
    if detail:
        _emit(f"        {detail}")


def bad(name: str, proves: str, not_proves: str, detail: str = "") -> None:
    """A failure. Recorded and counted exactly like a pass.

    A failure must not move the total, or a red run would report a different
    number of results than a green one and the pin would measure the tree's mood
    rather than the shape of the gate. The detail is where the findings go; the
    `proves` line is what the run was claiming, and on a failure it is what it
    failed to establish.
    """
    RESULTS.append(("FAIL", name, proves, not_proves, detail))
    _emit(f"[FAIL] {name}")
    _emit(f"        proves: {proves}")
    _emit(f"        not:    {not_proves}")
    if detail:
        for line in detail.splitlines():
            _emit(f"        {line}")


def skip(name: str, why: str) -> None:
    """A question this run could not answer, recorded as a result.

    Appends to `RESULTS` and therefore counts toward `EXPECTED_CHECKS`, which is
    the house convention from `check_text_encoding.py` and
    `binary_source_parity_check.py`. The reason is that the alternative is worse
    than useless: a gate that reached its total by declining to measure would be
    reaching it by not running its checks, and a run that checked nothing and a
    run that checked everything would print the same number. Never added to the
    pass count -- a skip and a pass are different claims and only one is true.
    """
    RESULTS.append(("SKIP", name, "", "", why))
    _emit(f"[SKIP] {name}")
    _emit(f"        {why}")


def check(cond: bool, name: str, proves: str, not_proves: str,
          detail: str = "") -> bool:
    """Record one result. `cond` decides the tag; the two messages are required.

    `proves` and `not_proves` are positional and have no defaults on purpose.
    The house rule is that every result says what it does *not* establish, and a
    rule that is only a comment is a comment: a check added here without the
    second clause is a `TypeError` at run time, not a quieter gate. The name is
    the third argument's neighbour so a reader sees the claim and its limit on
    consecutive lines, in that order.
    """
    if cond:
        ok(name, proves, not_proves, detail)
    else:
        bad(name, proves, not_proves, detail)
    return cond


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


# What the pattern is not allowed to look at, and the one reason.
#
# **A path in a comment or a docstring cannot execute; a path in a statement
# will.** That is the whole justification, and it is deliberately a claim about
# *executability* rather than about tidiness. `Run:  F:\python310\python.exe
# scripts\x.py` in a module docstring is a usage line: it documents the machine
# a maintainer typed it on, and nothing imports it, resolves it or runs it. The
# gate was red on nine `scripts/` files for exactly that line and on nothing
# else, which is what a gate that cries wolf looks like from the inside -- a red
# a maintainer learns to ignore, produced by a rule that was right and files
# that were fine.
#
# The counterpart is not negotiable, and it is why the mask below is measured
# rather than trusted: a path in *code* is a portability defect, and this change
# must not become the reason the rule stopped seeing one.
#
# That sentence is the one that goes stale, so it is worth saying what it was
# protecting against and what has happened to it. When the mask landed,
# `scripts/docs_claims_check.py` spelled one interpreter's `site-packages` as a
# `Path` constant and this gate was red on it while green on everything else --
# noise class 11 hits, signal 1, and a fix that had taken the signal with it
# would have been worth nothing. **It is no longer red** (re-run 2026-10-02), and
# the only trace of that path left in the file is inside the comment that records
# having removed it -- which is the mask is what makes invisible. So the worked
# example this paragraph leaned on has been spent: the constant is gone, but the
# string survives in prose, and prose is exactly what this mask exists to stop
# counting. The rule is unchanged and `_PATH_CASES` is still measured every run,
# which is what has to carry the guarantee now that the example cannot. A comment
# naming only the current failure would be true today and wrong the day the next
# `Path(r"C:\...")` constant is written -- the same rot `check_scripts_declare.py`
# had to undo in its own copy of this note.
#
# Scope, stated rather than implied. The mask is built for `.py` only, from
# `tokenize` for comments and from the syntax tree for docstrings, because those
# are the two things the standard library can say are prose with certainty.
# Every other suffix this gate reads (`.md`, `.toml`, `.cfg`, `.ini`, `.rs`)
# keeps the raw rule: there is no stdlib way to know what a comment is in Rust
# or TOML, and a guess that is wrong in the permissive direction is a hole
# rather than an exclusion. Markdown has no code at all and is still scanned
# whole, because a path in a document is a path a reader is told to run.


def _code_only(text: str) -> str | None:
    """`text` with every comment and every docstring replaced by spaces.

    Spaces rather than deletion, because `ABS_PATH`'s lookbehind reads the
    character in front of a match: the mask has to keep every position it
    occupies or it would change what the pattern can see, and a mask that
    shifted the text would report line numbers that are not the file's.

    `None` when the file cannot be tokenised or parsed. The caller then scans
    the file whole rather than trust a mask it could not build -- a mask that
    failed open would be a hole wearing a fix's clothes.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        tree = ast.parse(text)
    except (SyntaxError, ValueError, tokenize.TokenError):
        return None

    original = text.splitlines(keepends=True)
    lines = list(original)

    def blank(lineno: int, start: int, end: int | None) -> None:
        if not 1 <= lineno <= len(lines):
            return
        raw = lines[lineno - 1]
        body = raw.rstrip("\r\n")
        stop = len(body) if end is None else min(end, len(body))
        if not 0 <= start < stop:
            return
        lines[lineno - 1] = body[:start] + " " * (stop - start) + raw[len(body):]

    def char_col(lineno: int, byte_col: int) -> int:
        """`ast` reports UTF-8 byte columns and `lines` holds `str`."""
        body = original[lineno - 1].rstrip("\r\n") if 1 <= lineno <= len(original) else ""
        return len(body.encode("utf-8")[:byte_col].decode("utf-8", "replace"))

    for tok in toks:
        if tok.type == tokenize.COMMENT:
            blank(tok.start[0], tok.start[1], None)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                  ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        # An f-string is not a docstring -- `__doc__` is None for one -- so it is
        # left in place, and a path inside an f-string is still executable text.
        if not (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        lit = first.value
        last = lit.end_lineno or lit.lineno
        for lineno in range(lit.lineno, last + 1):
            blank(lineno,
                  char_col(lineno, lit.col_offset) if lineno == lit.lineno else 0,
                  char_col(lineno, lit.end_col_offset) if lineno == last else None)
    return "".join(lines)


# The mask is a gate now, so it is held to the rule the pattern already is: each
# case must both hide and not hide, and the counting has to tell a wrong answer
# from a missing answer. `expected=None` is the decline case -- a file the mask
# must refuse rather than guess about.
_PROSE_CASES = (
    ('"""Run:  F:\\\\python310\\\\python.exe scripts\\\\x.py"""\n', 0,
     "a usage line in a module docstring, the shape nine scripts/ files carry"),
    ('"""Docs."""\n_spk = Path("F:/python310/lib/site-packages/opendocking")\n', 1,
     "the same path in a statement, which is the defect the rule exists for"),
    ('_spk = Path("F:/x")  # and F:/y\n', 1,
     "a path in code on a line that also carries a comment"),
    ('# measured under `F:\\python310` on this machine\n', 0,
     "a path in a comment"),
    ('#: the interpreter was F:\\python310, in a `#:` note\n', 0,
     "a path in a `#:` note, the comment style this tree uses throughout"),
    ('label = "C:/Users/33654/Desktop"\n', 1,
     "a path in a plain string, which is a statement and not a docstring"),
    ('note = "F:/a # b"\n', 1,
     "a `#` inside a string, which is not a comment"),
    ('def f():\n    """see F:\\\\x\\\\y"""\n    return 1\n', 0,
     "a path in a function docstring"),
    ('class C:\n    """see F:\\\\x\\\\y"""\n', 0,
     "a path in a class docstring"),
    ('cfg = "/home/33654/x"  # POSIX as well\n', 1,
     "a POSIX home path in code with a comment beside it"),
    ('def f(:\n    "F:/unparseable"\n', None,
     "a file that cannot be parsed, which the mask must decline rather than "
     "half-blank"),
)


def _verify_prose_mask() -> list[str]:
    """Prove the comment/docstring mask both hides and does not hide."""
    bad = []
    for source, expected, why in _PROSE_CASES:
        masked = _code_only(source)
        if expected is None:
            if masked is not None:
                bad.append(
                    f"the comment/docstring mask did not decline a file it cannot "
                    f"parse, so it would be trusted on one: {why}"
                )
            continue
        if masked is None:
            bad.append(f"the comment/docstring mask declined a file it parsed: {why}")
            continue
        got = sum(len(ABS_PATH.findall(line)) for line in masked.splitlines())
        if got != expected:
            bad.append(
                f"the comment/docstring mask is wrong on {why}: expected "
                f"{expected} hit(s), got {got}"
            )
    return bad


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


def iter_files(root: pathlib.Path) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES:
            continue
        if SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        out.append(path)
    return sorted(out)


#: Everything the walk above measured, kept so the results below are statements
#: about one list rather than about two independent walks. The old code called
#: `iter_files()` a *second* time to print the count, so the number in the
#: header was a fresh reading of the tree taken after the scan rather than the
#: number of files the scan had read -- the two differ the moment anything is
#: written while the gate runs.
FINDINGS: dict[str, list[str]] = {
    "undecodable": [],
    "leaks": [],
    "links": [],
    "anchors": [],
    "pattern": [],
    "prose": [],
    "yaml": [],
}
LINK_TARGETS = 0
ANCHOR_TARGETS = 0
YAML_FILES = 0
YAML_MODULE: object | None = None


def _scan(root: pathlib.Path) -> None:
    """Walk the scope once and fill `FINDINGS` and the counters."""
    global LINK_TARGETS, ANCHOR_TARGETS, YAML_FILES, YAML_MODULE

    for path in iter_files(root):
        rel = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            FINDINGS["undecodable"].append(f"{rel}: not valid UTF-8")
            continue

        # --- absolute local paths -----------------------------------------
        # This file is skipped: it spells out the very patterns it hunts for.
        #
        # For a `.py` the scan reads the masked text, so a path in a comment or
        # a docstring is not a hit and a path in a statement still is. The
        # fallback is the whole file: a mask that could not be built must not
        # become a line the rule stopped reading.
        if rel.name != pathlib.Path(__file__).name:
            scanned = text
            if path.suffix.lower() == ".py":
                masked = _code_only(text)
                if masked is not None:
                    scanned = masked
            for number, line in enumerate(scanned.splitlines(), 1):
                for hit in ABS_PATH.findall(line):
                    FINDINGS["leaks"].append(
                        f"{rel}:{number}: machine-specific path {hit!r}")

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
                    LINK_TARGETS += 1
                    ANCHOR_TARGETS += 1
                    if target[1:] not in anchors:
                        FINDINGS["anchors"].append(
                            f"{rel}:{number}: no heading matches anchor {target}")
                    continue
                if target.startswith("#") or target.startswith(":"):
                    continue
                LINK_TARGETS += 1
                bare = target.split("#", 1)[0]
                if not bare:
                    continue
                if not (path.parent / bare).exists():
                    FINDINGS["links"].append(
                        f"{rel}:{number}: broken link -> {target}")

    # --- YAML / CITATION must actually parse -----------------------------
    # A workflow that does not parse fails every run in 0 seconds with
    # "This run likely failed because of a workflow file issue", which is easy
    # to miss and impossible to debug from the Actions tab. Caught here
    # instead, before the push.
    try:
        import yaml
    except ImportError:
        YAML_MODULE = None
    else:
        YAML_MODULE = yaml
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".yml", ".yaml", ".cff"}:
                continue
            if SKIP_DIRS & set(path.relative_to(root).parts):
                continue
            YAML_FILES += 1
            rel = path.relative_to(root)
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                FINDINGS["yaml"].append(f"{rel}: does not parse as YAML -- {first}")

    FINDINGS["pattern"].extend(_verify_path_pattern())
    FINDINGS["prose"].extend(_verify_prose_mask())


def _report(count: int, problems: list[str], cap: int = 3) -> str:
    """`count` finding(s), the first `cap` of them, and the rest by number.

    A red that lists 400 lines is a red nobody reads to the end, and the line
    that decides the run is usually the first one. So the first few are quoted
    verbatim and the remainder is a number -- which is a real loss and is
    labelled as one, because "and more" without the count is how a reader
    concludes there were three.
    """
    if not count:
        return f"0 finding(s)"
    shown = problems[:cap]
    left = count - len(shown)
    head = "\n".join(shown)
    return head if left <= 0 else f"{head}\n... and {left} more"


# --------------------------------------------------------------------------
# The eight content results. Each is one predicate, named, and each says what it
# does not establish. The order is the order of the walk: what was read, what
# was in it, what it pointed at, then whether the instruments themselves still
# work, then whether the scope is still a scope.
# --------------------------------------------------------------------------


def check_readable(found: list[str], total: int) -> None:
    check(
        not found,
        "every file in scope decodes as UTF-8",
        "the gate can read every file its own scope claims, so a result derived "
        "from a file's text is a result about that file",
        "that the bytes are the encoding their author meant. A UTF-16 file that "
        "happens to be all ASCII decodes as UTF-8 with NULs in it, and nothing "
        "here looks for a NUL. It also says nothing about a file whose suffix is "
        "not in DOC_SUFFIXES: those are not read at all, and the scope check is "
        "what notices a *declared* class going empty, not a class nobody declared",
        _report(len(found), found) if found else
        f"{total} file(s) in scope, every one of them decoded",
    )


def check_no_leak(found: list[str], total: int) -> None:
    check(
        not found,
        "no machine-specific path in any statement",
        "no file in scope carries an absolute local path in text that can "
        "execute: a statement in a `.py`, and any line at all in a `.md`, "
        "`.toml`, `.cfg`, `.ini` or `.rs`",
        "that prose is clean. A path in a comment or a docstring is exempt by "
        "design, on the claim that it cannot execute -- so a portability defect "
        "documented in a comment is deliberately not this result's business, and "
        "this file skips itself by name for the same reason. The mask is the "
        "one narrow thing here: for a `.md` or `.rs` there is no stdlib parser "
        "that can tell a comment from code, so those are read whole and a path "
        "inside a fenced block or a `//` comment in Rust is a hit",
        _report(len(found), found, cap=3) if found else
        f"{total} file(s) walked, this one excluded by name",
    )


def check_links(found: list[str], total: int) -> None:
    check(
        not found,
        "every relative Markdown link target exists on disk",
        "every relative link target in a `.md` file in scope names a path that "
        "is there, checked against the filesystem rather than against a list",
        "that the target is the *right* file -- an existing file at the end of a "
        "renamed link passes. It does not resolve a `#fragment` on a link to "
        "another file (only same-file anchors are checked, and they are the next "
        "result), and it does not fetch `http(s):` or `mailto:` targets at all. "
        "The regex reads every line, so a link-shaped string inside a fenced "
        "code block is measured as a link",
        _report(len(found), found) if found else
        f"{LINK_TARGETS - ANCHOR_TARGETS} relative target(s) resolved",
    )


def check_anchors(found: list[str], total: int) -> None:
    check(
        not found,
        "every in-page anchor matches a heading in the file that carries it",
        "every `#fragment` link in a `.md` file in scope names a heading in that "
        "same file, under GitHub's ATX slug rule, which is the rule the link "
        "will be resolved by when a reader clicks it",
        "that this slug rule is GitHub's for every input. It lower-cases, drops "
        "punctuation and hyphenates spaces; CJK is kept, and a heading that "
        "differs from GitHub on a duplicate or on a trailing symbol is a case "
        "this cannot decide. Links into *other* files carry a fragment and are "
        "counted by the previous result as a whole-file check only",
        _report(len(found), found) if found else
        f"{ANCHOR_TARGETS} same-file anchor(s) resolved, out of the "
        f"{LINK_TARGETS} link target(s) the header counts",
    )


def check_pattern(found: list[str], total: int) -> None:
    check(
        not found,
        "the absolute-path pattern fires and stays quiet on every case it is held to",
        "`ABS_PATH` returns the declared hit count on all "
        f"{len(_PATH_CASES)} cases in `_PATH_CASES`, so a pattern that had "
        "stopped matching would report a healthy run instead of a silent one",
        "that the cases are the right cases. Eleven hand-written strings are a "
        "sample of the shapes a path can take, and nothing here would notice a "
        "twelfth shape -- a UNC path on a host with a hyphen, a `%USERPROFILE%` "
        "expansion, a path assembled from two halves. Adding a case is what "
        "covers that, and no check in this file can tell that one is missing",
        _report(len(found), found) if found else
        f"{len(_PATH_CASES)}/{len(_PATH_CASES)} cases as declared",
    )


def check_prose(found: list[str], total: int) -> None:
    check(
        not found,
        "the comment/docstring mask hides and does not hide on every case it is held to",
        "`_code_only` returns the declared hit count on all "
        f"{len(_PROSE_CASES)} cases in `_PROSE_CASES`, and returns `None` on the "
        "one case it must decline rather than guess about, so the exemption "
        "cannot become a blanket one",
        "that the mask is right on a file whose docstring is an f-string, an "
        "implicitly concatenated string, or a stub with a body but no docstring; "
        "those cases are not written down. It does not apply to any suffix but "
        "`.py` at all -- for `.md`, `.toml`, `.cfg`, `.ini` and `.rs` the whole "
        "file is read, and that is the price of there being no stdlib way to "
        "know what a comment is in Rust",
        _report(len(found), found) if found else
        f"{len(_PROSE_CASES)}/{len(_PROSE_CASES)} cases as declared, "
        f"including the one that must be declined",
    )


def check_yaml(found: list[str], total: int) -> None:
    name = "every YAML/CFF file in scope parses"
    if YAML_MODULE is None:
        skip(
            name,
            "PyYAML is not importable in this interpreter, so no YAML file could "
            "be parsed and none can be reported. The slot is filled rather than "
            "left empty: the total below is the same on a machine that can parse "
            f"YAML and one that cannot. This is a skip and not a pass -- "
            f"{YAML_FILES} file(s) were never opened, and a run that checked "
            "nothing prints the same total as a run that checked everything",
        )
        return
    check(
        not found,
        name,
        f"all {YAML_FILES} `.yml`/`.yaml`/`.cff` file(s) outside the skip "
        "directories load under `yaml.safe_load`, so a workflow that would fail "
        "every run in 0 seconds with 'This run likely failed because of a "
        "workflow file issue' is caught before the push",
        "that the parsed document is correct. Valid YAML naming a step that does "
        "not exist parses cleanly, and a `.github/workflows/ci.yml` that is "
        "syntactically fine and semantically wrong passes this result entirely. "
        "It also does not cover a YAML file under a skipped directory, which is "
        "by design and not by measurement",
        _report(len(found), found) if found else
        f"{YAML_FILES} file(s) parsed, 0 failures",
    )


def check_scope(files: list[pathlib.Path]) -> None:
    """The declared scope against the set the walk actually produced.

    Counts suffixes from the walked list rather than from a second walk, for the
    same reason the header does: a number taken from a fresh walk is a reading of
    the tree taken after the scan, not a property of the scan.
    """
    counts: dict[str, int] = {}
    for path in files:
        s = path.suffix.lower()
        counts[s] = counts.get(s, 0) + 1
    empty_now = sorted(s for s in DOC_SUFFIXES if counts.get(s, 0) == 0)
    unexplained = [s for s in empty_now if s not in EMPTY_SUFFIX_REASONS]
    stale = [s for s in EMPTY_SUFFIX_REASONS if counts.get(s, 0) > 0]
    problems: list[str] = []
    for s in unexplained:
        problems.append(
            f".{s.lstrip('.')}: declared in DOC_SUFFIXES and holds no file, with "
            "no reason recorded")
    for s in stale:
        problems.append(
            f".{s.lstrip('.')}: carries a reason saying its class is empty, and "
            f"now holds {counts[s]} file(s) -- delete the stale reason")
    check(
        not problems,
        "the declared scan scope is answered: every suffix is populated, or carries a reason that is still true",
        "every suffix in `DOC_SUFFIXES` holds at least one file in scope or has a "
        "recorded reason for holding none, and a reason that has gone stale is "
        "itself a finding -- so a scope that quietly narrows to whatever happens "
        "to be present reports red rather than a smaller number",
        "that this set of suffixes is the right set. `DOC_SUFFIXES` and "
        "`SKIP_DIRS` are a *decision* and nothing in this run can say whether "
        "the decision is correct: adding a suffix widens the gate and removing "
        "one narrows it, and neither move makes a result go red. What this "
        "catches is a class that emptied under a scope nobody changed, which is "
        "the failure that is silent; the failure that is a decision is not "
        "detectable from inside the decision",
        _report(len(problems), problems) if problems else
        f"{len(files)} file(s) over {len(counts)} of {len(DOC_SUFFIXES)} declared "
        f"suffix(es); "
        + (", ".join(f"{s} empty by decision ({len(EMPTY_SUFFIX_REASONS)} reason "
                     f"on file)" for s in sorted(EMPTY_SUFFIX_REASONS))
           if EMPTY_SUFFIX_REASONS else "no class empty"),
    )


def main() -> int:
    _emit("Is anything in this tree a broken link, a machine-specific path, or "
          "a file that does not parse?")
    files = iter_files(SCAN_ROOT)
    if SCAN_ROOT != ROOT:
        _emit(f"   (OD_REPO_DOCS_ROOT: measuring {SCAN_ROOT} instead of {ROOT})")
    _scan(SCAN_ROOT)

    _emit(f"\nscanned {len(files)} files, {LINK_TARGETS} relative link(s), "
          f"{YAML_FILES} YAML/CFF file(s), "
          f"{len(_PATH_CASES)} path-pattern cases, "
          f"{len(_PROSE_CASES)} prose-mask cases")

    _emit("\n=== what was read ===")
    check_readable(FINDINGS["undecodable"], len(files))
    check_no_leak(FINDINGS["leaks"], len(files))

    _emit("\n=== what it pointed at ===")
    check_links(FINDINGS["links"], LINK_TARGETS)
    check_anchors(FINDINGS["anchors"], LINK_TARGETS)

    _emit("\n=== whether the instruments still work ===")
    check_pattern(FINDINGS["pattern"], len(_PATH_CASES))
    check_prose(FINDINGS["prose"], len(_PROSE_CASES))
    check_yaml(FINDINGS["yaml"], YAML_FILES)

    _emit("\n=== whether the scope is still a scope ===")
    check_scope(files)

    tally()
    return finish()


def tally() -> None:
    npass = sum(1 for r in RESULTS if r[0] == "PASS")
    nfail = sum(1 for r in RESULTS if r[0] == "FAIL")
    nskip = sum(1 for r in RESULTS if r[0] == "SKIP")
    name = "the tally accounts for every result, skips included"
    if npass + nfail + nskip == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(
            name,
            "every result this run recorded is exactly one of pass, fail or "
            "skip, and this is the last of them, so the run reaches its declared "
            f"total of {EXPECTED_CHECKS}",
            "that the number is the *right* number. It says "
            f"{EXPECTED_CHECKS} results were recorded; whether those are the right "
            "results is what the eight above are for, and a run that replaced "
            "every one of them with an unconditional pass would reach this same "
            "total and this same exit code. The pin measures the shape of the "
            "gate, never the health of the tree",
            f"{npass} + {nfail} + {nskip} = {len(RESULTS)}, and this check is the "
            f"{len(RESULTS) + 1}th. The same {EXPECTED_CHECKS} on a green tree, "
            "on a red one, and on an interpreter with no PyYAML, because a skip "
            "is a result and a failure does not move the total",
        )
    else:
        bad(
            name,
            "every result this run recorded is exactly one of pass, fail or "
            f"skip, and this is the last of them, so the run reaches its declared "
            f"total of {EXPECTED_CHECKS}",
            "that the number is the right number. It says "
            f"{EXPECTED_CHECKS} results were recorded; whether those are the right "
            "results is what the eight above are for",
            f"recorded {len(RESULTS)} result(s) before this check "
            f"({npass}+{nfail}+{nskip}), which must be {EXPECTED_CHECKS - 1}. A "
            "result that stopped being recorded is invisible from the outside: "
            "the run would print a smaller total and, without this line, exit 0",
        )


def finish() -> int:
    npass = sum(1 for r in RESULTS if r[0] == "PASS")
    nfail = sum(1 for r in RESULTS if r[0] == "FAIL")
    nskip = sum(1 for r in RESULTS if r[0] == "SKIP")
    _emit("")
    _emit(f"--- {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")
    _emit("    three numbers that are never added together: a pass count alone "
          "cannot say whether there was anything to look at")
    if nfail:
        _emit(f"RESULT: FAIL -- {nfail} of {EXPECTED_CHECKS} checks failed")
        return 1
    # Assembled from what the run did, because the sentence this line used to be
    # a fixed string claimed "all YAML parses" on an interpreter with no PyYAML
    # in it -- a claim about a measurement the run had not made.
    yaml_claim = ("all YAML parses" if YAML_MODULE is not None
                  else "no YAML file could be checked, PyYAML is not importable")
    _emit("no broken links, no machine-specific path in any statement, the path "
          "pattern and the prose mask agree with every case, " + yaml_claim)
    _emit(f"RESULT: OK -- {npass} passed, {nskip} could not be measured here")
    return 0


def report_crash(exc: BaseException) -> int:
    """A run that did not finish says so, in the same words as one that did.

    A gate that dies after printing a `[ok  ]` line leaves a reader unable to
    tell a finished run from a stopped one without reading the exit code, and an
    unhandled traceback is not a verdict. The summary is unreachable unless every
    check ran, and `DID NOT FINISH` is a line no finished run can print. Exit 2
    rather than 1: 1 means a check failed and 1 result is on the record, and
    there is no result here.
    """
    import traceback

    traceback.print_exc()
    npass = sum(1 for r in RESULTS if r[0] == "PASS")
    nfail = sum(1 for r in RESULTS if r[0] == "FAIL")
    nskip = sum(1 for r in RESULTS if r[0] == "SKIP")
    _emit("")
    _emit(f"--- STOPPED EARLY after {len(RESULTS)} recorded result(s): "
          f"{npass} passed, {nfail} failed, {nskip} skipped")
    _emit(f"RESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}")
    _emit("    a finished run prints OK or FAIL above and never this line; "
          f"{len(RESULTS)} of {EXPECTED_CHECKS} checks ran")
    return 2


if __name__ == "__main__":
    # No `sys.stdout.reconfigure` here, deliberately, where
    # `check_text_encoding.py` does call it. Reconfiguring to utf-8 makes every
    # line encodable and therefore makes `_emit`'s escape path unreachable: a
    # character cp936 cannot hold would arrive as U+FFFD, silently mangled, and
    # the reader could not tell a mangled path from a real one. The console is
    # left as it is and the escape is used instead, so the output says what it
    # could not render.
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_exc))
