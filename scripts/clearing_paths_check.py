"""The clearing paths, driven by mutations instead of described in prose.

Run:  python scripts/clearing_paths_check.py

**What a clearing path is, and why this file exists.** A gate in this tree
reports a red. Somebody has to do something about it, and the things that can be
done are not all equal: read the diff and fix the gate, or *clear* the red
without fixing anything. A clearing path is any mechanism that makes a red go
away without the thing that was red being corrected. `check_scripts_declare.py
--pin` is one. `--accept-shrink` and `--accept-moved` are two more, and they are
not clearing paths on their own -- they are the words that unlock one.

**Nothing tested them.** CI runs the detector, not the clearing paths, so the
only exercise `--pin` ever got was a person running it and reading a diff. That
is a real method and it does not scale: two defects in `--pin` were found this
way in two rounds -- a non-idempotent rewrite, and then a silent success over a
gate whose guard condition had been inverted -- and both were found by reading
and by hand-building a mutation. A person-shaped test finds one defect per round
and the tree does not get safer between rounds. The durable form of those two
findings is not a third round of prose; it is this file.

**Why the mutations are derived from the AST.** Both hand-built mutations in
those two rounds were invalid for the same reason: the line was typed rather
than located, so it did not match the file. A mutation nobody has to keep in
sync with the source cannot rot, and one that is *located by predicate* cannot
quietly stop matching -- it stops finding a node and this gate goes red saying
which one it wanted. Every mutation below finds its node by predicate and
splices the source using that node's own byte positions.

**This round covers `--pin` and the writes-nothing clause of `--census`, and
says so rather than implying coverage.** The rest are listed at the bottom of
this file with what each one would need. One of them, done properly, is worth
more than seven described.

**The cost, stated.** Each slot runs the clearing path as a subprocess against
a private copy of `scripts/`, and a run of that gate is not fast. This file
takes minutes, not seconds. That is the price of testing a command rather than
describing it, and a version that ran in a second would be testing a mock.

**The rest of the clearing paths, and what each would take.** The ones below
are the ones this tree actually has, found by reading it rather than by
remembering it. For each: what has to be driven, and what the declared outcome
is. None of them needs anything this file does not already do -- they need a
`run()` that passes different arguments and a mutation located the same way.

* **`--census`, per drift class.** The read-only half. **Covered for the
  byte-identical clause, in both directions and with a control**: the slots
  near the bottom hash the whole sandbox tree, not the subject file, so a
  command that created a second file would be caught as well as one that
  rewrote the first; they take the exit-1 branch (something has drifted), the
  exit-0 branch (nothing has, reached by pasting the three declaration lines
  the command itself printed), and then a run of `--pin` that *does* write, so
  the two silences are known to be about the command. **Not covered: a
  mutation per drift class.** Both slots above run against whatever drift a
  copy of this tree happens to carry, so the claim is "the tree did not move
  while a census ran", not "the tree does not move for each of the six kinds
  of drift". Six mutations of a shrinking subject is a bigger piece of work
  than the clause is worth on its own, and the clause is the part that was
  unchecked.
* **Pasting a `GATE-DECLARE` block into a gate's own source.** Twenty-three
  gates clear their reds this way and it is a hand edit, so it cannot be driven
  from here at all. What it needs is the property `--census` already has and
  `--pin` does not: a block that is *stale* must be distinguishable from a
  block that is *absent*, and today only the second is a red.
* **`MAX_SNAPSHOT_ONLY`.** Called a clearing path in this tree's own prose, and
  honestly: raising it is indistinguishable from paying down debt, and the
  constant only ever falls, so nothing detects a raise. It needs no mutation --
  it needs the cap recorded somewhere a raise would have to match, which is a
  change to the gate rather than to a test.
* **The membership tables.** `INVENTORY`, `EXCLUSIONS`, `UNPROVABLE_BY_AST`,
  `UNJUSTIFIED_PINS`, `SKIP_CONVENTION`, `MAX_UNPINNED_SCRIPTS`. Adding a name
  to any of them clears a red, and each already carries a reverse check -- a
  name that has stopped applying to anything is reported. That is the pattern
  this file is built on, and it is the one clearing path in this tree that is
  already instrumented.
* **`check_text_encoding.py`'s `ENCODING_EXEMPT` and `SKIP_DIRS`.** Already
  instrumented, and the best of them: `check_exemptions_are_real` resolves a
  witness file for every skip on every run, so a skip whose justification was
  deleted goes red instead of quietly covering its directory. That is the shape
  to copy, and it is why the `| ` marker in a `--pin` refusal is countable
  rather than decorative.
* **`OD_ENCODING_SCAN_ROOT`.** Pointing a gate at a different tree. Declared
  outcome: the gate's own total does not move. One line of environment, no
  mutation, and it would have caught a real red that gate already had.
* **Editing an `EXPECTED_CHECKS` constant to match what a run printed.** The
  house convention, so it is the one everybody does, and the pin says where
  the number came from but nothing checks that it did. It needs the same
  witness idea: the run's own tally is the oracle, and a constant that was
  typed to a run rather than derived from one cannot be told apart from one
  that was.
* **The in-flight deferral.** A gate whose mtime is newer than the run's start
  is measured, printed, and held against nothing. That is the right default
  and it is also a clearing path, because a file saved during a run is
  uncompared that run. Capped at two, so it cannot become the general answer;
  a harness would drive it by writing a gate mid-run.
"""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

# This gate's own output is UTF-8, whatever the console's codepage is. Same
# line, same reason, as every other gate in scripts/: a gate whose output
# cannot be decoded is a gate that only works for the person who remembered.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
#: The clearing path under test, by name. `--pin` and `--accept-*` are read out
#: of `sys.argv` by that file, so they cannot be reached any other way: this
#: gate drives a real command, not a function.
SUBJECT = "check_scripts_declare.py"
#: A private copy of `scripts/`, so a run under test can only ever write here.
#: Under `target/`, which `.gitignore` carries, so the scratch is never shipped.
SANDBOX = ROOT / "target" / "clearing_paths"
#: The call names `_sites_of` counts. The predicate below has to be the same set
#: the audited gate uses, or a mutation lands on something that is not a site
#: and the run under test correctly reports no change -- which is the way a
#: mutation harness passes without testing anything.
CALLS = ("check", "ok", "bad", "expect")
#: What the subject calls the generated region. The gate list is read out of
#: this rather than typed, so a gate that grows a GATE-DECLARE block of its own
#: drops off the list by itself instead of being mutated into a no-op.
SNAPSHOT_HEAD = "DECLARED_SNAPSHOT = {"

#: How many results this file records, in every environment and on every tree.
#: The last check asserts the run reached it, and that the slot list below is
#: the same length, so the constant and the code cannot drift apart.
#:
#: **1 unconditional and 1 guarded against a pin of 16, and those are three
#: different numbers on purpose.** The fifteen slot checks are call sites inside
#: a `for`, so the guarded column is where they are; the sixteenth is the
#: tally, at the end of `main()` and under no condition at all, so it is the
#: unconditional one. The pin is results, not sites: a slot that declines still
#: records one, so the total is 16 on a tree where every mutation applied and on
#: a tree where none of them could.
#: GATE-DECLARE 1
#: sites: 1 unconditional + 1 guarded
#: guards: sha256:204dc01ad091bdd3e1f2a9a52dd868f66c1a1ab7e6327da8848f93b1f2cc27b5
EXPECTED_CHECKS = 16  # measured from a green run: 15 slot results + this one, exit 0

RESULTS: list[tuple[str, str]] = []


def record(tag: str, name: str, detail: str = "") -> bool:
    """One result. A slot that cannot run is recorded, never dropped.

    A gate that reached its total by declining to measure would be a smaller
    run wearing a pass, so a slot that cannot be filled records a skip with a
    reason and the total still has to be reached.
    """
    RESULTS.append((tag, name))
    print("[%s] %s" % (tag, name))
    for line in str(detail).splitlines():
        print("       " + line)
    return tag == "ok"


def check(ok, name: str, detail: str = "") -> bool:
    if not isinstance(ok, bool):
        # A slot that returns a tuple where a verdict belongs hands `check()`
        # something truthy, and a truthy non-verdict takes the passing branch.
        # That is a check that cannot fail, in a file whose subject is checks
        # that cannot quietly pass -- and it happened here, on the one slot
        # whose whole job is to be the witness against this run touching a live
        # file. Refused rather than coerced: the total is still reached, the
        # slot still records exactly one result, and the defect is the verdict
        # instead of being invisible inside it.
        return record("FAIL", name,
                      "this slot returned %r where a pass/fail verdict belongs, "
                      "and %r is %s, so left alone it could never have failed. "
                      "The detail it meant to report follows.\n%s"
                      % (ok, ok, "truthy" if ok else "falsy", detail))
    return record("ok" if ok else "FAIL", name, detail)


def decline(name: str, why: str) -> bool:
    return record("SKIP", name, why)


# --------------------------------------------------------------------------
# The sandbox
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_sandbox() -> Path:
    """A private copy of `scripts/`, and only of `scripts/`.

    The only thing removed here is this gate's own scratch directory from a
    previous run, which is the one path in the repository this file owns
    outright. It has to go rather than be merged into, because a file deleted
    from `scripts/` would otherwise linger in the copy and keep a gate alive
    that no longer exists -- the exact silent-shrink this file exists to catch.
    """
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    dest = SANDBOX / "scripts"
    dest.mkdir(parents=True)
    for f in sorted(SCRIPTS.glob("*.py")):
        shutil.copy2(f, dest / f.name)
    return SANDBOX


def run(tree: Path, *args: str):
    """The subject, run as a command, in the sandbox.

    `sys.executable` rather than a named interpreter: a path typed into this
    file would be a path that only resolves on the machine it was typed on, and
    this file is about a command working rather than about a command being
    remembered. Decoding is `errors="replace"` for the same reason the audit
    gates reconfigure stdout: a child that dies on a non-ASCII byte must not
    take this gate's verdict down with it.
    """
    cmd = [sys.executable, str(tree / "scripts" / SUBJECT)] + list(args)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(tree))
    return p.returncode, p.stdout, p.stderr


# --------------------------------------------------------------------------
# Mutations, each located by predicate
# --------------------------------------------------------------------------

def _line_starts(data: bytes) -> list[int]:
    out, n = [], 0
    for line in data.split(b"\n"):
        out.append(n)
        n += len(line) + 1
    out.append(n)
    return out


def _span(starts: list[int], node) -> tuple[int, int]:
    return (starts[node.lineno - 1] + node.col_offset,
            starts[node.end_lineno - 1] + node.end_col_offset)


def _whole_lines(starts: list[int], node, data: bytes):
    """The byte range of the lines a node fills, or None if it shares one.

    None is the common answer and it is deliberate: a mutation that splices out
    half of a line corrupts the file in a way the run under test then refuses
    to measure, and the harness would report a red about its own mutation.
    """
    a = starts[node.lineno - 1]
    b = starts[node.end_lineno] if node.end_lineno < len(starts) else len(data)
    sa, sb = _span(starts, node)
    if data[a:sa].strip() or data[sb:b].strip():
        return None
    return a, b


def _is_check_call(node) -> bool:
    """True for the statement shape the audited gate counts, by predicate."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    return getattr(node.value.func, "id", None) in CALLS


def _parents(tree) -> dict:
    out = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            out[c] = n
    return out


def _is_guarded(node, parents: dict) -> bool:
    """True when a loop or a branch sits between the site and the module.

    The mutations below that add or remove a site all target an *unconditional*
    one, and they have to: a duplicated guarded site moves the guarded column
    and the guard digest, and a deleted guarded site does the same, so a slot
    asserting "growth left the digest alone" would be asserting something about
    a mutation it did not make. Choosing the site by this predicate is what
    makes each mutation mean the one thing its name says.
    """
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try)):
            return True
    return False


# `ast.parse` is handed **bytes**, not `str`, and that is the whole reason the
# splices below are correct. On `str` input a node's `col_offset` is a UTF-8
# *byte* offset, so any non-ASCII character earlier in the line shifts every
# position after it and the splice lands in the middle of a different
# character. On `bytes` input the offsets are offsets into the bytes that are
# being spliced. Every mutation in this file would be a corruption rather than
# a mutation on a gate with a non-ASCII string in it.
def _parse(data: bytes):
    return ast.parse(data), _line_starts(data)


def mutation_move(data: bytes):
    """Negate the condition of an `if` that directly guards a check site.

    The same call sites under a different condition: the total does not move
    and the digest does, which is the event `--accept-moved` exists for.
    Measured on `contacts_criteria_check.py` by hand before this file existed,
    where `--pin` absorbed it silently and the census then called the tree
    clean. The replacement is parenthesised because `not a and b` is not
    `not (a and b)`, and an unparenthesised splice would be a mutation that
    changes the shape of the tree rather than the meaning of one check.
    """
    tree, starts = _parse(data)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not any(_is_check_call(st) for st in node.body):
            continue
        test = ast.unparse(node.test)
        # The two shapes `_sites_of` throws away. Negating one of them would
        # turn a site it ignores into one it counts, which is a different
        # mutation wearing this one's name.
        if "__name__" in test or test == "True":
            continue
        a, b = _span(starts, node.test)
        return data[:a] + ("not (" + test + ")").encode("utf-8") + data[b:]
    return None


def mutation_shrink(data: bytes):
    """Delete one unconditional check site. The direction that must never be quiet."""
    tree, starts = _parse(data)
    parents = _parents(tree)
    for node in ast.walk(tree):
        if not _is_check_call(node) or _is_guarded(node, parents):
            continue
        whole = _whole_lines(starts, node, data)
        if whole is None:
            continue
        return data[:whole[0]] + data[whole[1]:]
    return None


def mutation_grow(data: bytes):
    """Duplicate one unconditional check site.

    The guard set is untouched by this one, so the digest does not move. That
    is not a limitation of the mutation, it is the assertion: it is the
    direction `--pin` is supposed to absorb without a word from anybody, and
    growth that also moved a guard would be testing two things at once.
    """
    tree, starts = _parse(data)
    parents = _parents(tree)
    for node in ast.walk(tree):
        if not _is_check_call(node) or _is_guarded(node, parents):
            continue
        whole = _whole_lines(starts, node, data)
        if whole is None:
            continue
        return data[:whole[1]] + data[whole[0]:whole[1]] + data[whole[1]:]
    return None


def mutation_unreadable(data: bytes):
    """Remove one closing paren, so the file is a state that never existed.

    Located by predicate like every other mutation: the paren is the character
    immediately before a check call's end position, so this is still derived
    from the tree rather than typed. This is the input `--pin` promises to
    refuse rather than freeze, and nothing else in the tree tests that promise.
    """
    tree, starts = _parse(data)
    for node in ast.walk(tree):
        if not _is_check_call(node):
            continue
        b = _span(starts, node.value)[1]
        if data[b - 1:b] != b")":
            continue
        return data[:b - 1] + data[b:]
    return None


# --------------------------------------------------------------------------
# Reading what the run under test wrote
# --------------------------------------------------------------------------

def snapshot_names(src: str) -> list[str]:
    m = re.search(re.escape(SNAPSHOT_HEAD) + r"(.*?)\n\}", src, re.S)
    return re.findall(r'"([^"]+)":\s*\(', m.group(1)) if m else []


def snapshot_entry(tree: Path, name: str):
    src = (tree / "scripts" / SUBJECT).read_text(encoding="utf-8")
    m = re.search(r'"' + re.escape(name) + r'":\s*\((\d+), (\d+),\s*"([0-9a-f]{64})"\)',
                  src)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def guard_lines(err: str) -> list[str]:
    """The guards a refusal listed, counted by the marker it prints them behind."""
    return [l for l in err.splitlines() if l.startswith(GUARD_MARK)]


# --------------------------------------------------------------------------
# Reading `--census`, and hashing a tree
# --------------------------------------------------------------------------

#: A whole tree, as `{relative path: sha256}`, for the read-only claim.
#:
#: **Whole tree rather than the subject file, and that is the point.** The
#: claim `--census` makes is "the file is byte-identical either way", and a
#: witness that hashes only the file it names would pass a command that wrote a
#: *different* file -- a new snapshot, a cache, an `__pycache__` entry. This
#: one notices a file that appeared as well as one that changed, because the
#: paths are in the keys.
def tree_digest(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        # `__pycache__` is excluded because the interpreter writes it on
        # import and it is not part of what the command claims about. Nothing
        # else is: an unexplained file under this root is the finding.
        if "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        out[rel.as_posix()] = sha256(p)
    return out


def moved_between(before: dict, after: dict) -> list[str]:
    """Every path that was added, removed or changed. Sorted, and named."""
    out = []
    for name in sorted(set(before) | set(after)):
        if name not in before:
            out.append("+ " + name)
        elif name not in after:
            out.append("- " + name)
        elif before[name] != after[name]:
            out.append("~ " + name)
    return out


#: The three lines a declaration is made of, in the order they appear, each
#: with the pattern that identifies its slot.
#:
#: **Keyed by slot rather than collected from the diff, and that is a fix.** The
#: first version of the paste took the `+` lines of the census's diff and
#: wrote them over the block. A gate that merely GREW has a two-line hunk --
#: `#: GATE-DECLARE 1` is a context line, not a `+` -- so the paste replaced
#: three lines with two and deleted the head. The gate went from GREW to
#: UNDECLARED, which is how the bug was found: the slot above declined instead
#: of passing and named the census exit that followed. Merging by slot means a
#: hunk of two lines, three lines, or one line all land in the right place.
DECL_SLOTS = (
    ("head", re.compile(r"^#: GATE-DECLARE \d+$")),
    ("sites", re.compile(r"^#: sites: \d+ unconditional \+ \d+ guarded$")),
    ("guards", re.compile(r"^#: guards: sha256:[0-9a-f]{64}$")),
)
DECL_ORDER = [name for name, _rx in DECL_SLOTS]


def decl_slot(line: str):
    """`(slot, text)` for a declaration line, or `(None, None)`."""
    s = line.strip()
    for name, rx in DECL_SLOTS:
        if rx.match(s):
            return name, s
    return None, None


def census_paste(tree: Path, out: str) -> tuple[list, list]:
    """Apply the census's own printed diff to the sandbox copy, and say what.

    **The command's output is the only source of the three lines.** Nothing is
    re-derived here, and no number is typed: a gate whose declaration lives in
    its own source can only be re-baselined by a human pasting what `--census`
    printed, so driving that paste is the honest way to reach the branch where
    the command reports a clean tree. Re-deriving the numbers with a second
    implementation of the census inside this file would be a control written
    twice, and the second copy is the one nobody reads.

    Both shapes the census prints are handled, because they print the same
    declaration: a gate whose own block is stale, and a gate with **no**
    declaration anywhere, for which the diff is an addition and there is
    nothing to replace -- so those three lines are inserted above the pin,
    which is where the convention puts them.
    """
    done, skipped = [], []
    for section in out.split("=" * 74):
        head = section.strip().splitlines()[0] if section.strip() else ""
        m = re.match(r"^(\S+\.py)\s+\[", head)
        if not m:
            continue
        name = m.group(1)
        want = {}
        for l in section.splitlines():
            if not l.startswith("  +") or l.startswith("  +++"):
                continue
            slot, text = decl_slot(l[3:])
            if slot:
                want[slot] = text
        if not want:
            skipped.append(name)
            continue
        path = tree / "scripts" / name
        if not path.is_file():
            skipped.append(name + " (not in the sandbox)")
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        have = {}
        for i, l in enumerate(lines):
            slot, _text = decl_slot(l)
            if slot and slot not in have:
                have[slot] = i
        if have:
            new = list(lines)
            for slot, text in want.items():
                new[have[slot]] = text + "\n"
            note = "replaced " + ",".join(
                s for s in DECL_ORDER if s in want)
        else:
            # An UNDECLARED gate has nowhere to replace: the census prints the
            # three lines as an addition and says nothing about where they go,
            # because that is a judgement. The convention in this tree is that
            # they sit directly above the pin, so that is where they are put,
            # and the fact that this file chose the position is why the slot
            # asserts on the exit code and not on the placement.
            pin = next((i for i, l in enumerate(lines)
                        if re.match(r"^EXPECTED_CHECKS\s*=", l)), None)
            if pin is None:
                skipped.append(name + " (no declaration and no pin to sit above)")
                continue
            block = [want[s] + "\n" for s in DECL_ORDER if s in want]
            new = lines[:pin] + block + lines[pin:]
            note = "added " + ",".join(s for s in DECL_ORDER if s in want)
        path.write_text("".join(new), encoding="utf-8", newline="")
        done.append("%s (%s)" % (name, note))
    return done, skipped


# --------------------------------------------------------------------------
# The run under test
# --------------------------------------------------------------------------

#: The guard a refusal prints each guard behind. Read out of the subject rather
#: than assumed, so a reworded message goes red here instead of silently
#: passing a check that is looking for text that is no longer printed.
GUARD_MARK = "          | "


def census_kinds(out: str) -> dict[str, str]:
    """The drift class `--census` filed each gate under, read off its headers.

    `--census` prints one `name  [KIND ...]` header per drifted gate and a
    unified diff under it. The header is the line to read and the diff is not:
    the diff body repeats the gate's name inside the quoted snapshot row, so a
    substring search for the name matches a gate that is only *mentioned* in a
    diff for something else. The class is also the whole point -- "is the gate
    listed" cannot tell a growth from a move, and the difference between those
    two is the entire subject of this file.
    """
    kinds = {}
    for line in out.splitlines():
        m = re.match(r"^(\S+\.py)\s+\[([A-Z]+)", line)
        if m:
            kinds[m.group(1)] = m.group(2)
    return kinds


class UnderTest:
    """One sandbox, driven through the states a clearing path has to survive.

    The state is shared between slots on purpose: each failure mode is set up
    on a tree that the *previous* mode has already settled, so a red is about
    the mode under test and not about a tree that was dirty for some earlier
    reason. That costs one extra run per re-settle and buys the property that
    makes a failure readable.
    """

    def __init__(self):
        self.tree = None
        self.gate = None
        self.pristine = b""
        self.why = None
        self.refused_bytes = None
        self.was = None
        self.derby = None
        self.live_before: dict[str, str] = {}
        self.live_after: dict[str, str] = {}

    def prepare(self):
        self.tree = make_sandbox()
        self.gate = pick_target(self.tree)
        if self.gate is None:
            self.why = ("no gate in the generated snapshot has a check site under "
                        "an `if` and another on a line of its own, so there is "
                        "nothing here to move, shrink or grow")
            return False
        self.pristine = (self.tree / "scripts" / self.gate).read_bytes()
        return True

    # -- state helpers ------------------------------------------------------

    def subject(self) -> Path:
        return self.tree / "scripts" / SUBJECT

    def restore(self):
        (self.tree / "scripts" / self.gate).write_bytes(self.pristine)

    def mutate(self, fn) -> bool:
        data = (self.tree / "scripts" / self.gate).read_bytes()
        out = fn(data)
        if out is None:
            return False
        (self.tree / "scripts" / self.gate).write_bytes(out)
        return True

    def entry(self):
        return snapshot_entry(self.tree, self.gate)

    def settle(self, *flags):
        """Absorb whatever is outstanding, so the next mode starts from clean.

        Called with whatever flags the outstanding state needs. It is a run of
        the real command rather than a reset of the files, because a harness
        that tidied the sandbox itself would be testing a tree the command
        would never produce.
        """
        return run(self.tree, "--pin", *flags)

    def reset(self):
        """A fresh copy of the tree, settled, with no flag.

        **This is the fix for a bug this file shipped with on its first run.**
        The phases used to share one sandbox and undo the previous mutation by
        restoring the gate's bytes. That is not enough, because `--pin` is a
        *forward* move: once the growth slot has been absorbed, the declaration
        says 32 call sites and no command puts it back to 31. So the move slot
        was handed a tree one site short of its own declaration, the refusal it
        got was a SHRANK, and four slots downstream reported a defect in
        `--accept-moved` that was not there. The two agree that the tree moved;
        only one of them is the event under test, and the harness cannot tell
        them apart from the outside.

        So each phase starts from a copy that agrees with itself, and the
        agreeing is done by the command rather than by an edit. A fresh copy of
        this tree is never clean -- four gates grow as other work lands -- and
        growth is the one direction `--pin` absorbs without a word, so a plain
        settle is expected to succeed. When it does not, the outstanding state
        is a shrink or a move and this returns False rather than arranging the
        tree to suit the test.
        """
        self.tree = make_sandbox()
        self.gate = pick_target(self.tree)
        if self.gate is None:
            return False, ("no gate in the generated snapshot can carry all four "
                           "mutations, so there is nothing to drive")
        self.pristine = (self.tree / "scripts" / self.gate).read_bytes()
        code, _out, err = run(self.tree, "--pin")
        if code != 0:
            return False, ("a fresh copy of scripts/ could not be settled with a "
                           "plain --pin (exit %d), so it has an outstanding "
                           "shrink or move and no phase of this run can be set "
                           "up on a tree that agrees with itself. That is a "
                           "property of the tree and not of the clearing path, "
                           "and arranging the tree to suit the test is the one "
                           "thing this file must not do. The refusal was: %s"
                           % (code, err.strip().splitlines()[0][:160]
                              if err.strip() else "(silent)"))
        return True, "a fresh copy of scripts/, settled by a plain --pin"

    # -- the slots ----------------------------------------------------------

    def slot_growth(self):
        """A gate that gained a check is absorbed with no word from anybody."""
        was = self.entry()
        if not self.mutate(mutation_grow):
            return False, "no whole-line check site to duplicate in " + self.gate
        before = sha256(self.subject())
        code, _out, err = run(self.tree, "--pin")
        after = self.entry()
        ok = (code == 0 and before != sha256(self.subject())
              and after[0] == was[0] + 1 and after[1] == was[1]
              and after[2] == was[2] and "refusing" not in err)
        return ok, ("declared %d+%d, derived %d+%d; the digest is %s, and it did "
                    "not move because a duplicated site adds no guard. exit %d, "
                    "no flag asked for"
                    % (was[0], was[1], after[0], after[1],
                       "unchanged" if after[2] == was[2] else "MOVED", code))

    def slot_clean(self):
        """The overwhelmingly common case: nothing to do, and nothing typed."""
        before = sha256(self.subject())
        code, _out, err = run(self.tree, "--pin")
        same = before == sha256(self.subject())
        return (code == 0 and same and err.strip() == ""), (
            "exit %d on a tree that already agrees, the file is byte-identical "
            "afterwards (%s), and stderr was empty (%r). A clearing path that "
            "demanded ceremony here would be abandoned, so this is the case "
            "that keeps the refusals affordable"
            % (code, same, err.strip()[:60]))

    def slot_moved_refused(self):
        """A moved gate is refused by name, with both digests and every guard."""
        ready, why = self.reset()
        if not ready:
            return None, why
        if not self.mutate(mutation_move):
            return False, "no `if` guarding a check site in " + self.gate
        self.was = self.entry()
        self.refused_bytes = sha256(self.subject())
        code, _out, err = run(self.tree, "--pin")
        found = re.search(r"sha256:([0-9a-f]{12}) -> sha256:([0-9a-f]{12})", err)
        listed = guard_lines(err)
        ok = (code == 2 and self.gate in err and found is not None
              and found.group(1) == self.was[2][:12]
              and found.group(2) != self.was[2][:12]
              and len(listed) == self.was[1]
              and "--accept-moved" in err and "MOVED" in err)
        return ok, ("%s. The guard on an `if` was negated, so the declared %d "
                    "call sites are still %d and only the digest moved. exit %d. "
                    "The refusal names the gate, both digests (%s -> %s), lists "
                    "%d guard(s) against the %d the declaration records, and "
                    "asks for --accept-moved"
                    % (why, self.was[0] + self.was[1],
                       self.was[0] + self.was[1], code,
                       found.group(1) if found else "?",
                       found.group(2) if found else "?",
                       len(listed), self.was[1]))

    def slot_moved_wrote_nothing(self):
        """The refusal is a refusal: not a traceback, and not a rewrite."""
        now = self.entry()
        same = sha256(self.subject()) == self.refused_bytes
        return (now == self.was and same), (
            "the generated snapshot still reads %d+%d %s and every byte of the "
            "file is as it was, so the refusal cost nothing and changed nothing"
            % (now[0], now[1], now[2][:12]))

    def slot_census_before(self):
        """The move is real: the census files it as a move, so the refusal is
        load-bearing rather than the release being unconditional.

        Without this the release below would be unfalsifiable. A harness that
        mutated nothing would also find the gate absent afterwards, and the
        check that matters would be measuring its own failure.
        """
        code, out, _err = run(self.tree, "--census")
        kind = census_kinds(out).get(self.gate)
        return (kind == "MOVED"), (
            "--census files %s as %s, with the call-site total flat and the "
            "digest differing. So the tree really is moved, `--pin` really did "
            "refuse something, and the release below is a decision rather than "
            "an unconditional pass (census exit %d, which this slot does not "
            "assert on: a gate declaring its own GATE-DECLARE block cannot be "
            "re-baselined by --pin at all, so a copy of this tree is never "
            "globally clean and its exit code is not evidence about the gate "
            "under test)"
            % (self.gate, kind or "not listed at all", code))

    def slot_moved_accepted(self):
        """With the word typed, the move is absorbed -- and visibly."""
        code, _out, _err = run(self.tree, "--pin", "--accept-moved")
        now = self.entry()
        self.derby = now
        ok = (code == 0 and now[2] != self.was[2]
              and (now[0], now[1]) == (self.was[0], self.was[1]))
        return ok, ("exit %d, and the snapshot moved from %s to %s over the same "
                    "%d+%d call sites. Same counts, new digest: the release is "
                    "the only thing that changed, and it took the flag to do it"
                    % (code, self.was[2][:12], now[2][:12], now[0], now[1]))

    def slot_census_after(self):
        """And only now is the gate gone from the census's list."""
        code, out, _err = run(self.tree, "--census")
        left = census_kinds(out)
        return (self.gate not in left), (
            "%s is no longer listed by --census at all. %d other gate(s) still "
            "are, and this slot does not assert on them: they declare their own "
            "blocks, so --pin cannot reach them, which is exactly what --census "
            "is for (census exit %d). Read next to the slot above: same tree, "
            "same census, same run of the subject -- the only difference in the "
            "world is that the digest had been accepted out loud"
            % (self.gate, len(left), code))

    def slot_idempotent(self):
        """Three passes, byte-identical from the second.

        The first defect ever found in this command was a rewrite that was not
        idempotent, and it is a property no single run can show: it takes a
        second run to see that the first one left something behind.
        """
        seen = [sha256(self.subject())]
        codes = []
        for _ in range(2):
            code, _out, _err = run(self.tree, "--pin", "--accept-moved")
            codes.append(code)
            seen.append(sha256(self.subject()))
        return (codes == [0, 0] and seen[0] == seen[1] == seen[2]), (
            "the release above was the first pass; two more exited %s and left "
            "%s, %s, %s. Byte-identical from the second, so a second run cannot "
            "tell you to re-run it. The exit codes are asserted as well as the "
            "bytes on purpose: a refused run writes nothing and is therefore "
            "trivially byte-identical, and this slot would pass for the right "
            "reason on a command that was refusing"
            % (codes, seen[0][:12], seen[1][:12], seen[2][:12]))

    def slot_shrink_refused(self):
        """The other direction, and the one that was already there."""
        ready, why = self.reset()
        if not ready:
            return None, why
        was = self.entry()
        self.was = was
        if not self.mutate(mutation_shrink):
            return False, "no whole-line check site to delete in " + self.gate
        before = sha256(self.subject())
        code, _out, err = run(self.tree, "--pin")
        now = self.entry()
        return (code == 2 and "SHRANK" in err and "--accept-shrink" in err
                and now == was and before == sha256(self.subject())), (
            "%s. One unconditional check site was deleted from a gate declaring "
            "%d+%d, so it now has one call site fewer and the counts say so. "
            "exit %d, SHRANK named, --accept-shrink asked for, and the snapshot "
            "left untouched at %d+%d %s"
            % (why, was[0], was[1], code, now[0], now[1], now[2][:12]))

    def slot_shrink_accepted(self):
        """Still symmetric, still a word somebody has to type."""
        code, _out, _err = run(self.tree, "--pin", "--accept-shrink")
        now = self.entry()
        return (code == 0 and now[0] == self.was[0] - 1), (
            "exit %d, and the snapshot records %d+%d. The flag is a decision, "
            "not a repair: the gate still has one call site fewer than it did"
            % (code, now[0], now[1]))

    def slot_unreadable(self):
        """A gate that cannot be read, and flags that do not unlock that.

        The acceptance flags exist to release a *classification*. A file that
        does not parse has no classification, so neither flag may reach it --
        otherwise `--accept-moved` is not a word about movement, it is a switch
        that turns the guard off.
        """
        if not self.mutate(mutation_unreadable):
            return False, "no call to remove a closing paren from in " + self.gate
        # Which of the two refusals fires depends on which probe meets the
        # broken file first: the declaration probe runs `tokenize` and the
        # census walk runs `ast.parse`, and an unclosed bracket is an EOF error
        # to one of them and a syntax error to the other. Both are the same
        # refusal and the slot asserts on the property, not on the wording.
        refused_by = ("could not be read", "does not parse")
        before = sha256(self.subject())
        code, _out, err = run(self.tree, "--pin")
        plain = code == 2 and any(m in err for m in refused_by)
        code2, _o2, err2 = run(self.tree, "--pin", "--accept-shrink",
                               "--accept-moved")
        both = code2 == 2 and any(m in err2 for m in refused_by)
        said = [l.strip() for l in err.splitlines()
                if any(m in l for m in refused_by)]
        return (plain and both and before == sha256(self.subject())), (
            "one closing paren removed, so the gate is a state that never "
            "existed. Plain: exit %d and a named refusal (%s). With both "
            "acceptance flags: exit %d and still a named refusal. The file is "
            "untouched either way, so the flags release a classification and "
            "not the measuring"
            % (code, said[0][:90] if said else "no refusal line found", code2))

    # -- the census slots ---------------------------------------------------
    #
    # `--census` is the read-only half of the clearing path and its "writes
    # nothing" clause is the whole contract: exit 1 when something has drifted,
    # 0 when nothing has, and **the tree byte-identical either way**. The three
    # slots below are the two directions of that clause and the control that
    # says the witness is a measurement.

    def slot_census_dirty_read_only(self):
        """The direction a run of `--census` actually takes on a working tree.

        A copy of `scripts/` is never clean -- gates grow as other work lands,
        and a gate whose declaration lives in its own source cannot be
        re-baselined by `--pin` at all -- so this is the reachable direction
        and it is the one that matters most: a read that finds work to report
        and writes the report nowhere.
        """
        self.tree = make_sandbox()
        before = tree_digest(self.tree)
        code, out, _err = run(self.tree, "--census")
        after = tree_digest(self.tree)
        moved = moved_between(before, after)
        said = ("Nothing above was written to disk" in out
                or "every gate's declaration matches" in out)
        return (code == 1 and not moved and said), (
            "census exit %d with %d file(s) in the sandbox hashed before it and "
            "%d after, and %s. The command's own last line agrees that it wrote "
            "nothing (%s). A copy of this tree is not clean and cannot be made "
            "clean by `--pin`, so exit 1 is the expected answer here and the "
            "byte-identical clause is the part under test"
            % (code, len(before), len(after),
               "none of them moved" if not moved
               else "these moved: " + ", ".join(moved), said))

    def slot_census_clean_read_only(self):
        """The other direction: nothing has drifted, and still nothing moved.

        Reached by settling the snapshot gates with `--pin` and then pasting
        into the sandbox copies the three declaration lines `--census` itself
        printed, which is the only way a human re-baselines a gate that owns its
        block. A gate that stopped drifting must not start writing: the early
        return is a different code path from the drift report, and a witness
        that only ever saw one of them has not seen the command.

        **The settle here passes `--accept-shrink` and `--accept-moved`, and
        this slot does not use `reset()`.** `reset()` declines when a copy of
        this tree cannot be settled with a plain `--pin`, and at the time of
        writing one gate had lost call sites -- so a tree with a shrink in it
        could never reach the branch this slot exists to test. The flags are
        recorded in the detail line rather than hidden: they decide what the
        census is *asked*, not what the census *does*, and the claim under test
        is that the census wrote nothing either way. Declining here instead would
        have left the exit-0 clause untested for as long as any gate anywhere
        was mid-edit, which is most of the time.
        """
        self.tree = make_sandbox()
        settle_code, _o, settle_err = run(self.tree, "--pin", "--accept-shrink",
                                          "--accept-moved")
        if settle_code != 0:
            return None, ("a fresh copy of scripts/ could not be settled even "
                          "with both acceptance flags (exit %d), so the clean "
                          "census could not be reached and nothing is claimed "
                          "about that direction. The refusal was: %s"
                          % (settle_code, settle_err.strip().splitlines()[0][:140]
                             if settle_err.strip() else "(silent)"))
        _code, first, _e = run(self.tree, "--census")
        pasted, skipped = census_paste(self.tree, first)
        before = tree_digest(self.tree)
        code, out, _err = run(self.tree, "--census")
        after = tree_digest(self.tree)
        moved = moved_between(before, after)
        clean = code == 0 and "every gate's declaration matches" in out
        if not clean:
            return None, ("the sandbox was settled with `--pin "
                          "--accept-shrink --accept-moved` (exit %d) and %d "
                          "declaration(s) pasted, but the census still exits %d, "
                          "so the direction where nothing has drifted could not "
                          "be reached and nothing is claimed about it. Gates it "
                          "could not re-baseline: %s"
                          % (settle_code, len(pasted), code, skipped or "none"))
        return (not moved), (
            "census exit 0, printing that every gate's declaration matches the "
            "tree, with %d file(s) hashed before it and %d after and %s. The "
            "tree was settled first with `--pin --accept-shrink --accept-moved` "
            "(exit %d) and the %d declaration(s) the census then printed were "
            "pasted into the sandbox copies -- %s -- which is a person editing "
            "gates, done outside the command, and is the distinction this whole "
            "file is about"
            % (len(before), len(after),
               "none of them moved" if not moved
               else "these moved: " + ", ".join(moved),
               settle_code, len(pasted), "; ".join(sorted(pasted))))

    def slot_census_witness_can_fail(self):
        """The control: the same witness, on a run that *does* write.

        Without this the two slots above are a green print. `--pin` on a copy
        of this tree has to rewrite the generated snapshot, and this asserts
        that `tree_digest` sees it and names it -- so "nothing moved" in the two
        slots above is a measurement of the command and not a measurement of
        the witness finding nothing because it cannot find anything.
        """
        self.tree = make_sandbox()
        before = tree_digest(self.tree)
        code, _out, _err = run(self.tree, "--pin")
        after = tree_digest(self.tree)
        moved = moved_between(before, after)
        names = [m[2:] for m in moved]
        # The digest's keys are paths **relative to the sandbox root**, so the
        # name to expect carries the `scripts/` prefix. The first version of
        # this compared `moved_between`'s own output against the bare subject
        # name, so it reported a red on a run that had detected the write
        # correctly and named it -- the slot failed for a reason that had
        # nothing to do with the claim, which is the failure mode a control
        # written in a hurry always has.
        want = ["scripts/" + SUBJECT]
        return (code == 0 and names == want), (
            "--pin on the same kind of copy exited %d and moved %d file(s): %s. "
            "The witness is the same function the two slots above use, on the "
            "same kind of tree, so their silence is about `--census` and not "
            "about a witness that cannot see a write"
            % (code, len(moved), ", ".join(names) or "nothing, which is the "
               "failure this slot exists to catch"))

    def slot_live_untouched(self):
        """Nothing this run did reached outside the sandbox."""
        self.live_after = {p.name: sha256(p) for p in sorted(SCRIPTS.glob("*.py"))}
        moved = [n for n, _h in self.live_before.items()
                 if self.live_after.get(n) != self.live_before[n]]
        # `(not moved)` and **not** `(not moved, "")`. The stray comma made
        # `ok` the tuple `(False, "")`, which is truthy, so `check()` took its
        # passing branch on every run and this slot could not fail -- in the
        # one slot whose entire job is to be the witness against this run
        # touching a live file. It survived a green 13-of-13 because the slot
        # that would have caught it is the slot that was broken, and it was
        # visible in the same breath: the detail line said "these moved:
        # check_text_encoding.py" while the verdict above it said ok. A detail
        # that contradicts its own verdict is the tell, and reading the detail
        # is why it was found.
        return (not moved), (
            "%d live file(s) in scripts/ hashed before this run and after it, "
            "and %s. Every write the runs above performed went to %s, which is "
            "under `target/` and named in `.gitignore`. A red here means a file "
            "in scripts/ changed while this gate was running, which on a shared "
            "tree is somebody else and is named above rather than left as a "
            "mystery"
            % (len(self.live_before), "none of them moved" if not moved
               else "these moved: " + ", ".join(moved), SANDBOX))


def pick_target(tree: Path):
    """The gate to mutate, chosen out of the generated snapshot.

    A gate that carries a GATE-DECLARE block of its own is skipped by `--pin`
    entirely, so mutating one would exercise nothing and this harness would
    pass over a mutation that never happened. Reading the list out of the
    generated region rather than naming a gate here means a gate that grows a
    block drops off the list by itself, and the run either finds another or
    declines every slot with a reason instead of testing a no-op.
    """
    src = (tree / "scripts" / SUBJECT).read_text(encoding="utf-8")
    for name in snapshot_names(src):
        f = tree / "scripts" / name
        if not f.is_file():
            continue
        data = f.read_bytes()
        if (mutation_move(data) and mutation_shrink(data)
                and mutation_grow(data) and mutation_unreadable(data)):
            return name
    return None


#: `(phase, method, name)` in the order they run.
#:
#: The phase is load-bearing rather than decorative. Each phase after the first
#: begins by rebuilding the sandbox, and if that rebuild cannot produce a tree
#: that agrees with itself the phase **declines and every remaining slot in it
#: declines with it**. Without the grouping, one decline would leave the rest of
#: the phase running against whatever state the previous phase happened to
#: leave behind -- which is precisely the bug this file shipped with, and the
#: grouping is the fix rather than a pause between two loops.
SLOTS = [
    ("growth", "slot_growth",
     "a gate that gained a check is absorbed by a plain --pin, with no flag and "
     "no warning, which is the direction that fires when the project does "
     "something good"),
    ("growth", "slot_clean",
     "a tree that already agrees is a no-op: plain --pin exits 0 and writes "
     "nothing"),
    ("moved", "slot_moved_refused",
     "a moved gate is REFUSED by name, with the old digest, the new digest, "
     "every guard the gate carries, and the word that would release it"),
    ("moved", "slot_moved_wrote_nothing",
     "the refusal wrote nothing: a named refusal, not a traceback and not a "
     "rewrite"),
    ("moved", "slot_census_before",
     "the census sees the move before the release, so the refusal above is "
     "load-bearing rather than the release being unconditional"),
    ("moved", "slot_moved_accepted",
     "--accept-moved releases it, writing a new digest over the same call-site "
     "counts"),
    ("moved", "slot_census_after",
     "and only then does the census stop listing the gate"),
    ("moved", "slot_idempotent",
     "three --pin passes leave the file byte-identical from the second"),
    ("shrink", "slot_shrink_refused",
     "a shrunken gate is still refused without --accept-shrink, and the refusal "
     "writes nothing"),
    ("shrink", "slot_shrink_accepted",
     "--accept-shrink still releases it, and still only when somebody says so"),
    ("unreadable", "slot_unreadable",
     "a gate that cannot be read is refused by name, and neither acceptance "
     "flag reaches it"),
    ("census", "slot_census_dirty_read_only",
     "the read-only half writes nothing when it has drift to report (exit 1)"),
    ("census", "slot_census_clean_read_only",
     "and writes nothing when it has none to report (exit 0), which is a "
     "different branch of the same command"),
    ("census", "slot_census_witness_can_fail",
     "and the hash witness that says so does see a write when one happens, so "
     "the two silences above are about --census and not about the witness"),
    ("live", "slot_live_untouched",
     "nothing this run did reached outside the sandbox"),
]


def main() -> int:
    ut = UnderTest()
    # Hashed before anything else happens, including the copy, so the claim
    # below covers the whole run rather than the part after the sandbox exists.
    ut.live_before = {p.name: sha256(p) for p in sorted(SCRIPTS.glob("*.py"))}
    armed = ut.prepare()
    blocked = None
    phase = None
    if not armed:
        # Declined slot by slot rather than returning early, so the tally below
        # is reached either way and a tree with nothing to mutate produces the
        # same shape of report as one with everything.
        for _phase, _method, name in SLOTS:
            decline(name, ut.why)
    else:
        print("sandbox: %s" % SANDBOX)
        print("gate under mutation: %s (chosen out of the generated snapshot, "
              "not named here)" % ut.gate)
        print("")
        for ph, method, name in SLOTS:
            if ph != phase:
                phase = ph
                blocked = None
            if blocked is not None:
                decline(name, blocked)
                continue
            try:
                ok, detail = getattr(ut, method)()
            except Exception as exc:
                # A slot that raises is a red about the slot, never a missing
                # result. Swallowing it would make this file's total reachable
                # by crashing, which is the failure every pin in this tree
                # exists to prevent.
                ok = False
                detail = "the slot raised %s: %s" % (type(exc).__name__, exc)
            # `None` rather than False is the difference between "this is
            # broken" and "this could not be asked". A phase whose tree could
            # not be settled declines, and so does the rest of that phase:
            # arranging the tree to suit the test would make the result
            # meaningless, and a red about that would be a red about this file
            # rather than about the clearing path.
            if ok is None:
                decline(name, detail)
                blocked = detail
            else:
                check(ok, name, detail)
    npass = sum(1 for t, _n in RESULTS if t == "ok")
    nfail = sum(1 for t, _n in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n in RESULTS if t == "SKIP")
    print("")
    print("--- %d passed, %d failed, %d skipped" % (npass, nfail, nskip))
    check(
        npass + nfail + nskip == len(RESULTS) == EXPECTED_CHECKS - 1
        and len(SLOTS) == EXPECTED_CHECKS - 1,
        "the tally accounts for every result, skips included",
        "%d results were recorded before this check (%d+%d+%d), which must be "
        "%d, and the slot list above is %d long. The constant says %d, and a "
        "slot that stopped running, or a slot added without moving the "
        "constant, goes red here rather than quietly making this file a "
        "smaller run. It says the shape of the harness is what was intended; "
        "whether the twelve results above are the right twelve is what they "
        "are for"
        % (len(RESULTS), npass, nfail, nskip, EXPECTED_CHECKS - 1, len(SLOTS),
           EXPECTED_CHECKS),
    )
    if nfail:
        print("RESULT: FAIL -- %d of %d checks failed" % (nfail, EXPECTED_CHECKS))
        return 1
    print("RESULT: OK -- %d checks" % EXPECTED_CHECKS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

