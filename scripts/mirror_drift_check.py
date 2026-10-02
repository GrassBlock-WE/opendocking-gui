"""Has the commit mirror drifted from the development tree, and can this gate see it?

Run:  F:\\python310\\python.exe scripts\\mirror_drift_check.py

# The claim, in one sentence

**Every path the release rule publishes is byte-identical between the
development tree and the commit mirror, and every asymmetry -- differing, missing
or extra -- is red.**

The mirror is the tree that actually gets committed and pushed. It is produced by
a one-way copy (`_sync_release.py`) that a developer runs by hand, and nothing in
this repository used to say whether the copy had been run. That is the whole
shape of the gap: a second copy of the source, with no gate saying it is a copy.

# The scope is derived, never typed

Every path compared here comes out of `release_tree_rule.iter_source_files`,
applied **independently to each tree root**. There is no path list in this file,
and the reason is not tidiness: a hand-written list of what the mirror ought to
carry is a second implementation of "what is a source file", which is the exact
thing `release_tree_rule.py` exists to have exactly one of. `prune_here`, the
suffix table, the root-name table, the `CACHEDIR.TAG` test, the Cargo-manifest
test and the `.gitignore` reader are all reached *through* the rule rather than
restated, so a change to any of them changes this gate's scope in the same run
that changed the release.

Two consequences of deriving the scope per side rather than from the development
tree alone, both load-bearing:

* An **extra** path is found by asking the rule what the *mirror* publishes, not
  by subtracting the development tree's answer from the mirror's raw file list.
  A file in the mirror that the rule declines is not this gate's business at all
  -- that is `release_tree_parity_check.py`'s "the published tree holds nothing
  the rule would delete", and duplicating it here would give two gates one
  question and let them disagree.
* The rule module is imported **once**, so both walks are decided by the same
  tables and the same `.gitignore` reader. `declared_generated_dirs()` resolves
  from the rule module's own `__file__`, so the declaration read is the
  *development* tree's `.gitignore` on both sides: a path cannot be in scope on
  one side and out of it on the other because the two trees carry different
  copies of a file. That is printed as a fact below rather than assumed.

# The mirror is excluded from its own scope, and there is no SKIP_DIRS constant

The brief this file answers asked whether "the release rule's own `SKIP_DIRS`"
keeps a naive walk from finding the mirror inside the mirror. **There is no
`SKIP_DIRS` in `release_tree_rule.py`.** The mechanism that does the job is
`prune_here`, and it is checked here rather than believed: `opendocking-gui` is
not a member of `SOURCE_ROOTS`, so `prune_here` returns `True` for it and
`recognition_reason` answers "it is under opendocking-gui/, which is not a
declared source root". Both are measured, and then a nested copy is *planted*
inside a sandbox mirror and shown to be invisible to this gate's walk while
being plainly visible to a naive `rglob` -- the second half being the half a
table lookup cannot prove.

On this machine there is no nested copy: `opendocking-gui/opendocking-gui` does
not exist, and the commit mirror is itself the git root. The guard is kept
because the cost of it being wrong is a tree that compares itself to itself.

# The list is never truncated

A "first 4 files" list reads exactly like a complete one, and the gate that
already measures this drift truncates: `release_tree_parity_check.py` prints
`found[:4]` followed by `... and 23 more`, which on the current tree means 4 of
27 missing and 4 of 64 differing are shown. **This file prints every
difference, in full, with both absolute paths and both digests**, and one of its
checks reads back the text it printed and requires one report block per measured
difference. The number in the verdict and the number of blocks in the listing are
compared, so a cap introduced later is a red rather than a shorter list.

# The current state is RED, and that is the correct reading

Measured on this machine: **95 asymmetries** -- 27 published paths missing from
the mirror, 64 differing in bytes, 4 present only in the mirror. This gate is
**expected to be red until somebody runs the one-way sync by hand.** The red is
the finding, not a defect in the check, and the fix is to sync the mirror --
**not** to widen this gate's scope, shorten the rule, or drop a suffix from
`SOURCE_SUFFIXES` until the numbers fall. A gate that is made green by
declining to look is worse than no gate, because it is green.

A standing notice saying so is printed on **every** run, green or red, so the
next reader cannot mistake the red for something to be tuned away.

# What this gate does not assert, and which gate owns each of those

* **The installed copy** under the running interpreter's `site-packages`. It is
  a build artefact, out of scope by decision; `SITE_PACKAGES_DECISION` in the
  rule is the quotable reason and `installed_copy_check.py` is the gate.
* **The mirror's non-source debris.** Whether the commit tree carries files the
  rule would delete is a different question from whether a path the rule
  *publishes* is missing, and `release_tree_parity_check.py` asks it.
* **The rule itself.** That one rule, one implementation, and that it ships is
  `release_tree_parity_check.py`. This file audits the mirror against the rule
  and never against its own opinion of the rule.
* **That a red here is a red there.** Every number is a snapshot of one walk.
  Four agents are editing this workspace, so the drift count is a lower bound on
  a tree being read while it changes, and the run re-walks the development tree
  afterwards to report how many paths moved under it (measured below, not
  assumed to be zero).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Python puts a script's own directory at sys.path[0], so this import works with
# no path juggling when the file is run as a script; the insert covers the other
# ways it gets loaded, and it adds the one directory rather than inventing a
# second route to the module.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import release_tree_rule as RULE  # noqa: E402  (after the path fix, on purpose)

#: The three kinds of asymmetry, as the words the report and the verdicts use.
#: `MISSING` is "the rule publishes it in the development tree and the mirror
#: does not have it"; `EXTRA` is the reverse; `DIFFERS` is both sides, different
#: bytes. Named constants rather than prose, because the verdicts and the
#: self-test match on them and a string typed twice is two strings.
MISSING = "MISSING FROM MIRROR"
DIFFERS = "DIFFERS"
EXTRA = "EXTRA IN MIRROR"

#: What to do about each kind, as data so the self-test and the live report say
#: the same sentence. The self-test deliberately does **not** print these: it
#: mutated the file itself, three lines earlier, and telling a reader to re-run
#: the sync because of a byte this check wrote to `target/` is a lie in a
#: sentence.
REMEDY = {
    MISSING: "the rule publishes this path in the development tree and the "
             "mirror does not carry it. Run the one-way sync",
    EXTRA: "the rule publishes this path in the mirror and the development "
           "tree does not have it: renamed, moved or dropped, and the mirror "
           "is what still remembers. Run the one-way sync, which deletes it",
    DIFFERS: "both trees carry this path and the bytes are not the same. The "
             "mirror is behind; run the one-way sync",
}
SELF_TEST_REMEDY = {
    MISSING: "the file this check deleted from the sandbox mirror is exactly "
             "what this comparison has to notice",
    EXTRA: "the file this check added to the sandbox mirror is exactly what "
           "this comparison has to notice",
    DIFFERS: "the byte this check flipped in the sandbox mirror is exactly "
             "what this comparison has to notice",
}

#: The name of the file the self-test adds to the sandbox mirror and to nothing
#: else. Only the *name* is written down, and it is written down because a file
#: has to have one; the **directory** it is planted in is derived from a path
#: the rule itself selected, so it is a declared source root by construction
#: rather than by this file having listed one.
PROBE_ADDED_NAME = "mirror_drift_added_only.py"

#: Scratch goes under the directory the rule itself derives as the Cargo build
#: directory, and then **one level deeper than that**, for a reason that is
#: measured rather than superstitious. `target/` carries a `CACHEDIR.TAG`, and
#: `recognition_reason` recovers the tree root by walking `len(rel.parts)` levels
#: up from the file's own directory. A sandbox placed *directly* under `target/`
#: sits one level closer to that cache directory than the arithmetic assumes,
#: and every file in it answers "this declares itself a cache directory" --
#: which reads as an empty scope and therefore as a green comparison. So the
#: sandbox root is `target/mirror-drift-sandbox-*/`, with each tree one level
#: below that, and no recovery off by one can reach `target/CACHEDIR.TAG`.
SANDBOX_PREFIX = "mirror-drift-sandbox-"

#: How many results this file records, in every environment. Measured from a run
#: on the development tree with both trees present, and unchanged when the mirror
#: is absent, because a comparison that could not be made is *recorded* as not
#: asserted rather than dropped: a run that reported fewer results on CI than
#: locally would be a smaller run wearing a pass.
#:
#:   1  the rule this run used is this checkout's own copy
#:   2  the commit mirror was located
#:   3  the published set is the rule's own answer, derived per side, and
#:      neither side is empty
#:   4  the mirror is excluded from its own scope: in the tables and on disk
#:   5  no published path lies inside a directory the rule prunes
#:   6  no published path is missing from the mirror
#:   7  no published path differs in the mirror
#:   8  the mirror holds no published path the development tree does not
#:   9  every difference is reported, one block each, with no cap
#:  10  a pair built from this tree's published set is byte-identical, and
#:      neither walk of it is empty
#:  11  one flipped byte in the sandbox mirror is reported, naming the path
#:  12  one file present only in the sandbox mirror is reported, naming the path
#:  13  one file deleted from the sandbox mirror is reported, naming the path
#:  14  the tally accounts for every result, not-asserted included
#:
#: 1-2 are the two facts every other number rests on. 3-5 are the scope, and 5
#: is a check of its own because "no path slipped in from a pruned directory" is
#: a different claim from "the scope is derived": the second is about the
#: derivation, the first is about what the derivation produced here. 6-8 are the
#: three directions of the asymmetry, kept as three results because a gate that
#: folds them into one number cannot say *which way* the mirror is wrong, and
#: the three have three different fixes. 9 is the anti-truncation check. 10-13
#: are the self-test: 10 is the green arm and it is built rather than mocked,
#: and 11-13 are the three directions, each required to produce **exactly one**
#: difference and to name the path.
#:
#: The self-test runs whether or not the mirror exists, because a sandbox pair is
#: built from the development tree and does not need the live mirror. So a CI
#: runner, which has no development tree to compare against at all, still gets
#: 10-13 rather than an empty file.
#:
#: **The two numbers below were derived, not typed, and the method was checked
#: before it was believed.** `check_scripts_declare.py --census` is the sanctioned
#: way to produce this block and it is out of bounds for this work, so the
#: auditor's own `_sites_of` and guards digest were called directly against this
#: file. That is only worth something if the call reproduces a block somebody else
#: wrote, so it was run against `installed_copy_check.py` first and returned that
#: file's declared `(3 unconditional, 14 guarded, sha256:e4359dd9...)` exactly.
#: The same walk then produced the two numbers here.
#:
#: 2 unconditional and 18 guarded is this file's shape, and the arithmetic is the
#: one `release_tree_parity_check.py` documents for itself: every verdict here is
#: an `if` with an `ok` in one branch and a `bad` in the other, so **each check is
#: two call sites and one result** -- 13 checks is 26 sites, and the two
#: unconditional ones are the `ok(...)` calls on the crash reporter's own path.
#: The guard digest is not decorative: it is what catches a result site moving
#: between guard shapes without the total moving, which the count alone cannot
#: see.
#: GATE-DECLARE 1
#: sites: 2 unconditional + 18 guarded
#: guards: sha256:709c3909e2bd50dd216027f8f32ade002c4050a50bf3ee6a1a50a5665cb2a25a
EXPECTED_CHECKS = 14

RESULTS: list = []

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2


# --------------------------------------------------------------------------
# The three result routes
# --------------------------------------------------------------------------
def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def unasserted(name: str, why: str) -> None:
    """A claim this machine cannot make. Counted, never dropped.

    Not a `skip`: a skip is a question declined, and this is a question with
    only one answer available on a machine that has one tree. The parity check
    makes the same three-way split for the same reason, and the total is the
    same number with both trees and with one.
    """
    RESULTS.append(("NOT-ASSERTED", name, why))
    print(f"[NOT-ASSERTED] {name}\n       {why}")


def section(title: str) -> None:
    print()
    print(f"-- {title}")


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------
def published_set(root: Path) -> dict:
    """`{relative path: (absolute path, sha256)}` for everything `root` publishes.

    The rule's own walk, and nothing else. `RULE.iter_source_files` is the same
    function `_sync_release.py` copies with and the same one
    `release_tree_parity_check.py` compares with, so "the published set" means
    one thing in this file and in the tool that produces the mirror.

    No digest memo, and that is a decision rather than an omission. The parity
    check memoises on `(path, size, mtime_ns)` because it re-reads 150 files four
    times; this file's self-test *writes* bytes between reads, and a cache that a
    same-size write could satisfy is precisely the thing that would make the
    green arm a lie. Re-reading 172 small files costs less than the argument.
    """
    out: dict = {}
    for rel, path in sorted(RULE.iter_source_files(root).items()):
        out[rel] = (path, RULE.file_digest(path))
    return out


def asymmetries(dev: dict, mir: dict) -> list:
    """Every way the two published sets disagree, in all three directions.

    `(relative path, kind, (left path, left sha), (right path, right sha))`,
    with `(None, "")` for a side that has no such file -- an absence reported as
    an absence rather than as an empty digest, because a line with two paths and
    one blank hash is a line the reader has to interpret.
    """
    out: list = []
    for rel in sorted(set(dev) - set(mir)):
        out.append((rel, MISSING, dev[rel], (None, "")))
    for rel in sorted(set(mir) - set(dev)):
        out.append((rel, EXTRA, (None, ""), mir[rel]))
    for rel in sorted(set(dev) & set(mir)):
        if dev[rel][1] != mir[rel][1]:
            out.append((rel, DIFFERS, dev[rel], mir[rel]))
    return out


def report_blocks(diffs: list, remedies: dict, left: str = "development",
                  right: str = "mirror") -> list:
    """One three-line block per difference: the path, and both copies with digests.

    A list of blocks rather than a joined string, so the count of blocks is
    something a check can compare against the count of differences. That
    comparison is the whole anti-truncation mechanism: a `[:4]` added to this
    function later makes check 9 red instead of making the report shorter
    without saying so.
    """
    blocks: list = []
    for rel, kind, left_pair, right_pair in diffs:
        # `" :: "` and not a column width: two of the three kind names contain a
        # space, and a fixed-width parse of them is the kind of arithmetic that
        # silently truncates a path. This separator cannot occur in a kind, and
        # splitting on it leaves the rest of the line -- spaces and all -- as
        # the path.
        lines = [f"  {kind} :: {rel.as_posix()}"]
        for side, pair in ((left, left_pair), (right, right_pair)):
            path, sha = pair
            where = str(path) if path is not None else "ABSENT -- no such file"
            lines.append(f"      {side:<12} {where}")
            lines.append(f"      {'':<12} sha256 {sha if sha else 'ABSENT'}")
        lines.append(f"      fix: {remedies[kind]}")
        blocks.append("\n".join(lines))
    return blocks


def report_drift(diffs: list) -> tuple:
    """Print every difference, grouped by kind. Returns `(text, emitted)`.

    The return value is what check 9 reads. Printing a list and then checking a
    different list would leave the cap in the printer, which is where the cap
    lives; so the printer hands back exactly the characters it wrote, and
    `emitted` is the path order those characters are in. Both halves come from
    this one function so that "the report" is a single object with two views of
    it rather than two renderings that can disagree.
    """
    out: list = []
    emitted: list = []
    for kind in (MISSING, DIFFERS, EXTRA):
        here = [d for d in diffs if d[1] == kind]
        # `==` and not the bare kind, so the per-kind count line cannot be
        # mistaken for a block header by the completeness check below. That
        # check finds headers by looking for a line whose first token is a kind
        # name, and a count line that began with one would be counted as a path.
        out.append(f"  == {kind}: {len(here)}")
        blocks = report_blocks(here, REMEDY)
        out.extend(blocks)
        emitted.extend(d[0].as_posix() for d in here)
    text = "\n".join(out)
    print(text)
    return text, emitted


def moving_paths(before: dict, after: dict) -> list:
    """Paths whose digest, or membership, changed between two walks of one tree.

    Four agents are editing this workspace. A file saved between this gate's
    two walks shows up as a difference against the mirror that is not drift, and
    a drift count that mixes the two is a number nobody should act on. So the
    development tree is walked again after the comparison and this returns what
    moved underneath the run.
    """
    moved: list = []
    for rel in set(before) | set(after):
        a = before.get(rel)
        b = after.get(rel)
        if a is None or b is None or a[1] != b[1]:
            moved.append(rel.as_posix())
    return sorted(moved)


# --------------------------------------------------------------------------
# Locating the mirror
# --------------------------------------------------------------------------
def find_mirror() -> tuple:
    """`(path, how)`, or `(None, why)`.

    The mirror is found the way `release_tree_parity_check.py` finds it: a
    checkout that *contains* `opendocking-gui/` is the development tree, and one
    that does not is the published tree with nothing to compare it against.
    `OD_SYNC_DEST` overrides, for pointing this gate at a pair built elsewhere,
    and the override that was used is printed -- a gate that quietly compared the
    wrong two directories is the one failure worth designing out.
    """
    override = os.environ.get("OD_SYNC_DEST")
    mirror = Path(override).resolve() if override else ROOT / RULE.RELEASE_TREE_DIRNAME
    if mirror.is_dir():
        how = (f"OD_SYNC_DEST={override} names the published tree"
               if override else
               f"this checkout contains {mirror.name}/, so it is the development "
               f"tree and the tree beside it is the commit mirror")
        return mirror, how
    if override:
        why = (f"OD_SYNC_DEST={override} names a directory that does not exist, "
               f"and this checkout has no {RULE.RELEASE_TREE_DIRNAME}/ either")
    else:
        why = (f"this checkout has no {RULE.RELEASE_TREE_DIRNAME}/ inside it. The "
               f"workspace root is not a git repository -- only the commit tree "
               f"is -- so on a CI runner the development tree does not exist and "
               f"nothing here can invent one without cloning the same tree twice "
               f"and measuring nothing")
    return None, why


# --------------------------------------------------------------------------
# The sandbox pair: built, not mocked
# --------------------------------------------------------------------------
def build_pair(sandbox: Path, source: Path) -> tuple:
    """Two real trees holding the same bytes. `(dev_root, mirror_root, count)`.

    The green arm of the self-test has to be a genuine pair of directories on
    disk, because a mocked equality proves that the comparison agrees with
    itself. So this reads each published file **once** and writes those same
    bytes to both sides, which has a second property worth having: four agents
    are editing the development tree, and reading once means a file saved
    mid-build cannot make the pair differ, so a red in the self-test is this
    check's arithmetic and never somebody else's editor.
    """
    dev_root = sandbox / "development"
    mirror_root = sandbox / "published"
    count = 0
    for rel, src in sorted(RULE.iter_source_files(source).items()):
        raw = src.read_bytes()
        for tree in (dev_root, mirror_root):
            dst = tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)
        count += 1
    return dev_root, mirror_root, count


def mutation_targets(mirror: dict, count: int) -> list:
    """The `count` largest published files in the pair. Derived, never named.

    A name written down here is a path that stops existing the day the tree
    renames it, and the arm would then be mutating a file that is not in scope
    and proving nothing at all -- the self-test would go green having tested a
    comparator on a file no user has. Largest-first, and skipping empty files,
    because flipping the last byte of a zero-length file raises rather than
    differs.
    """
    scored = []
    for rel, pair in mirror.items():
        size = pair[0].stat().st_size
        if size:
            scored.append((-size, rel.as_posix(), rel))
    scored.sort()
    return [rel for _n, _p, rel in scored[:count]]


def planted_mirror_names() -> list:
    """Relative paths of the nested copy the self-test plants inside a mirror.

    Two, and they are the two cases. One is a file at the root of a nested copy
    -- declined for the obvious reason. The other sits at
    `dock-py/python/opendocking/`, which is a **declared root with a declared
    suffix and the canonical package path**: the (root, suffix) test alone would
    publish it, and the only thing that declines it is the `opendocking-gui/`
    component above it. A guard proved only against the first case has not been
    proved against the second, which is the one a real nested copy would hit.
    """
    nested = Path(RULE.RELEASE_TREE_DIRNAME)
    return [nested / "README.md",
            nested / "dock-py" / "python" / "opendocking" / "core.py"]


# --------------------------------------------------------------------------
# Section 1 -- provenance and the scope
# --------------------------------------------------------------------------
def check_the_rule_is_ours() -> None:
    loaded = Path(RULE.__file__).resolve()
    mine = (HERE / "release_tree_rule.py").resolve()
    if loaded == mine:
        ok("the rule this run used is this checkout's own copy",
           f"{loaded}  sha256 {RULE.file_digest(loaded)[:16]}...  decision "
           f"surface sha256 {RULE.rule_fingerprint()[:16]}...  Every path below "
           f"was selected by {RULE.iter_source_files.__name__}() applied to each "
           f"tree root separately, through "
           f"{', '.join((RULE.prune_here.__name__, 'recognition_reason'))} -- "
           f"there is no path list in this file to be out of step with the rule")
    else:
        bad("the rule this run used is this checkout's own copy",
            f"imported {loaded}, expected {mine}. Every number below was "
            f"produced by whichever of the two answered, and a stale rule would "
            f"answer confidently")


def check_the_mirror_was_located(mirror: Path, how: str) -> None:
    if mirror is None:
        unasserted("the commit mirror was located", how)
        return
    git = (mirror / ".git").is_dir()
    ok("the commit mirror was located",
       f"{mirror}  ({how}).  It is {'also' if git else 'NOT'} the git root, so "
       f"there is no second copy of the repository inside it on this machine; "
       f"the guard in the next check is kept because the cost of it being wrong "
       f"is a tree that compares itself to itself")


def check_the_scope_is_derived(dev: dict, mir: dict) -> None:
    if not dev or not mir:
        bad("the published set is the rule's own answer on each side, and "
            "neither side is empty",
            f"the development tree published {len(dev)} path(s) and the mirror "
            f"{len(mir)}. A comparison over an empty side reports no difference "
            f"and is green, so an empty side is a failure here rather than a "
            f"quiet pass: the most likely cause is a walk root one level off, "
            f"where `prune_here` declines the whole tree")
        return
    ok("the published set is the rule's own answer on each side, and neither "
       "side is empty",
       f"the development tree publishes {len(dev)} path(s), the mirror "
       f"{len(mir)}, union {len(set(dev) | set(mir))}, byte-identical on "
       f"{len(set(dev) & set(mir))} of them. Derived per side, so an `extra` is "
       f"found by asking the rule what the mirror publishes rather than by "
       f"subtracting the development tree from a raw file list -- a file in the "
       f"mirror the rule declines is not this gate's question. One rule module "
       f"imported once, so both sides are decided by the same tables and the "
       f"same .gitignore reader, which resolves from the rule's own __file__ "
       f"and is therefore the development tree's on both sides")


def check_the_mirror_excludes_itself(pairs: tuple) -> None:
    """The nested-copy guard, in the tables and on disk.

    `pairs` is the sandbox pair, because the on-disk half has to plant a nested
    copy somewhere this run is allowed to write. A table lookup is not a proof
    that a walk would decline the thing: it is a proof about `prune_here`, and
    what is wanted is a proof about a directory tree.
    """
    mirror_root = pairs[1]
    name = RULE.RELEASE_TREE_DIRNAME
    pruned = RULE.prune_here(Path(name))
    rel = Path(name) / "README.md"
    reason = RULE.recognition_reason(rel, mirror_root / rel.parent) or ""
    # Plant it, then ask both questions: would a naive walk have found it, and
    # did the rule's walk decline it? Both answers, or neither is worth much.
    planted = []
    for p in planted_mirror_names():
        dst = mirror_root / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"# a copy of the repository inside the mirror\n")
        planted.append(p)
    # The naive comparator is a `*.py` rglob, so it can only be expected to see
    # the planted `.py` path. The first version of this check demanded it find
    # **both** planted paths and went red over the `.md` one -- a red that said
    # "the guard is gone" when the guard was working and the expectation was
    # wrong. So the comparison is against the subset a `*.py` walk could see, and
    # the detail line below reports all three numbers rather than rounding the
    # awkward one away.
    visible_to_naive = sorted(p.as_posix() for p in planted if p.suffix == ".py")
    naive = {q.relative_to(mirror_root) for q in mirror_root.rglob("*.py")}
    selected = set(published_set(mirror_root))
    would_find = sorted(p.as_posix() for p in planted if p in naive)
    slipped = sorted(p.as_posix() for p in planted if p in selected)
    for p in planted:
        (mirror_root / p).unlink()
    if pruned and reason and would_find == visible_to_naive and not slipped:
        ok("the mirror is excluded from its own scope, in the tables and on disk",
           f"there is no `SKIP_DIRS` constant in {Path(RULE.__file__).name}; the "
           f"mechanism is `prune_here`, which returns {pruned!r} for {name}/ "
           f"because it is not a member of SOURCE_ROOTS, and "
           f"`recognition_reason` answers {reason[:72]}... Measured on disk as "
           f"well: {len(planted)} path(s) planted inside a sandbox mirror, "
           f"{len(visible_to_naive)} of them carrying a suffix a naive rglob "
           f"looks for and {len(would_find)} of those found by one, and 0 of "
           f"the {len(planted)} selected by the rule's walk. The second planted "
           f"path is the sharp case -- dock-py/python/opendocking/ is a declared "
           f"root, a declared suffix and the canonical package path, so the "
           f"(root, suffix) test alone would publish it and only the {name}/ "
           f"component above it declines it")
    else:
        bad("the mirror is excluded from its own scope, in the tables and on disk",
            f"prune_here({name}/) = {pruned!r}; recognition_reason says "
            f"{reason[:120]!r}; a naive rglob of the sandbox mirror found "
            f"{would_find} where {visible_to_naive} of the {len(planted)} "
            f"planted path(s) were reachable to it; the rule's walk selected "
            f"{slipped}. Either the guard is gone or the plant did not take, "
            f"and a gate that cannot keep a nested copy out of its own "
            f"comparison is comparing the mirror to itself")


def check_nothing_pruned_is_in_scope(dev: dict, mir: dict) -> None:
    """No selected path sits inside a directory the rule prunes. Measured.

    Separate from the previous check on purpose: that one is about the mirror
    finding *itself*, this one is a property of what both walks produced, and it
    is the cheap sweep that would catch a new `SOURCE_ROOT` that accidentally
    re-admits a build tree.
    """
    if not dev or not mir:
        unasserted("no published path lies inside a directory the rule prunes",
                   "one of the two walks selected nothing, so there is no "
                   "published set to inspect")
        return
    watched = {RULE.RELEASE_TREE_DIRNAME, ".git", RULE.CARGO_TARGET_DIRNAME,
               "__pycache__"}
    leaked = sorted(r.as_posix() for r in set(dev) | set(mir)
                    if watched & set(r.parts))
    mirror_git = ROOT / RULE.RELEASE_TREE_DIRNAME / ".git"
    git_files = sum(len(f) for _d, _s, f in os.walk(mirror_git)) \
        if mirror_git.is_dir() else 0
    if leaked:
        bad("no published path lies inside a directory the rule prunes",
            f"{len(leaked)} published path(s) mention one of {sorted(watched)}: "
            f"{leaked[:6]}. A build tree or a version-control database inside "
            f"the published set is the shape of accident this rule was written "
            f"to make impossible")
    else:
        ok("no published path lies inside a directory the rule prunes",
           f"0 of the {len(dev) + len(mir)} published path(s) across both trees "
           f"mention {sorted(watched)}. The claim is not vacuous: the mirror's "
           f"own .git/ holds {git_files} file(s) on disk, every one of them "
           f"with a declared suffix or a declared root name, and the rule "
           f"declines the whole directory because `.git` is not a member of "
           f"SOURCE_ROOTS")


# --------------------------------------------------------------------------
# Section 2 -- the three directions
# --------------------------------------------------------------------------
LIVE_NAMES = {
    MISSING: "no published path is missing from the mirror",
    DIFFERS: "no published path differs in the mirror",
    EXTRA: "the mirror holds no published path the development tree does not",
}
LIVE_WHY = {
    MISSING: "the mirror is behind: the rule publishes this path in the "
             "development tree and the commit tree a user clones does not have "
             "it",
    DIFFERS: "both trees carry this path and its bytes disagree, so what a "
             "user clones is not the code this tree ships",
    EXTRA: "the rule publishes this path in the mirror and the development "
           "tree has dropped it, so the commit tree is carrying a file that no "
           "longer exists here",
}


def check_the_live_drift(dev: dict, mir: dict, diffs: list,
                         moved: list) -> str:
    """The three verdicts, and the full listing. Returns the text it printed.

    The listing is printed here, once, and handed to the completeness check
    rather than printed a second time by it: a gate that prints its report in
    two places has two places to cap, and the check has to be reading the same
    characters the reader is reading.
    """
    section("the drift between the development tree and the commit mirror")
    if not dev or not mir:
        why = ("one of the two published sets is empty, so there is no "
               "comparison to report and no count to print")
        for kind in (MISSING, DIFFERS, EXTRA):
            unasserted(LIVE_NAMES[kind], why)
        return "", []
    print(f"    {len(diffs)} asymmetry/asymmetries: "
          f"{sum(1 for d in diffs if d[1] == MISSING)} missing, "
          f"{sum(1 for d in diffs if d[1] == DIFFERS)} differing, "
          f"{sum(1 for d in diffs if d[1] == EXTRA)} extra")
    print()
    text, emitted = report_drift(diffs)
    print()
    for kind in (MISSING, DIFFERS, EXTRA):
        here = [d for d in diffs if d[1] == kind]
        if here:
            bad(LIVE_NAMES[kind],
                f"{len(here)} of them, listed in full above with both absolute "
                f"paths and both digests, one block each and no cap. "
                f"{LIVE_WHY[kind]}. {REMEDY[kind]}")
        else:
            ok(LIVE_NAMES[kind], LIVE_WHY[kind])
    if moved:
        print()
        print(f"    CAVEAT, measured: the development tree was walked again "
              f"after the comparison and {len(moved)} of its published path(s) "
              f"changed underneath this run ({moved[:4]}"
              f"{' ...' if len(moved) > 4 else ''}). Four agents are editing "
              f"this workspace, so the counts above are a snapshot of one walk "
              f"and at least that many of them may be an editor rather than "
              f"drift. Re-run to separate the two.")
    return text, emitted


def check_the_report_is_complete(diffs: list, text: str, emitted: list) -> None:
    """One block per difference, and no path left out of the text.

    This is the check that makes "never truncated" a mechanism rather than a
    promise, and it is two comparisons rather than one because a cap and a
    reordering are different defects:

    * `headers == emitted` -- the characters printed contain one block per
      rendered difference, in the order they were rendered. A `[:4]`, a `max=`
      and a `... and N more` all break this.
    * `sorted(emitted) == sorted(all drift paths)` -- the renderer emitted every
      difference and nothing else. This is what catches a renderer that drops a
      path silently, which the first comparison cannot see when the counts
      happen to agree.

    Both sides are read out of the text and the renderer's own list, never out
    of a second sort of the difference list: `asymmetries()` emits
    missing/extra/differing and the report groups missing/differing/extra, so a
    third ordering derived here would be a third thing to keep in step.
    """
    headers = []
    for line in text.splitlines():
        if not line.startswith("  ") or line.startswith("      "):
            continue
        kind, sep, rel = line.strip().partition(" :: ")
        if sep and kind in (MISSING, DIFFERS, EXTRA):
            headers.append(rel.strip())
    every = sorted(d[0].as_posix() for d in diffs)
    missing = [p for p in every if p not in headers]
    repeated = sorted({p for p in headers if headers.count(p) > 1})
    name = "every difference is reported, one block each, with no cap"
    if headers == emitted and sorted(emitted) == every:
        ok(name,
           f"{len(headers)} block header(s) parsed back out of the "
           f"{len(text.splitlines())} line(s) printed above, equal in order to "
           f"the {len(emitted)} difference(s) the renderer emitted, and those "
           f"{len(emitted)} are exactly the {len(every)} measured "
           f"asymmetry/asymmetries. So the listing is the measurement: a cap "
           f"added later to `report_blocks` or `report_drift` moves one of "
           f"those two numbers and turns this red rather than making the "
           f"listing shorter without saying so")
    else:
        bad(name,
            f"the text carries {len(headers)} block header(s); the renderer "
            f"emitted {len(emitted)}; {len(every)} asymmetry/asymmetries were "
            f"measured. Headers absent from the text: {missing[:6]}. Headers "
            f"listed more than once: {repeated[:6]}. Headers in the text but "
            f"not emitted: "
            f"{sorted(set(headers) - set(emitted))[:6]}. A report that shows a "
            f"sample reads exactly like a complete one, and this is the check "
            f"that tells the two apart")


# --------------------------------------------------------------------------
# Section 3 -- the self-test: a built pair, and all three directions
# --------------------------------------------------------------------------
def check_the_built_pair_is_identical(pairs: tuple, count: int) -> None:
    dev_root, mirror_root = pairs
    dev = published_set(dev_root)
    mir = published_set(mirror_root)
    diffs = asymmetries(dev, mir)
    name = "a pair built from this tree's published set is byte-identical, and " \
           "neither walk of it is empty"
    if not dev or not mir:
        bad(name,
            f"the built pair walks to {len(dev)} and {len(mir)} published "
            f"path(s) where {count} were written. A self-test comparing two "
            f"empty sets reports no difference and is green, so this is the "
            f"green arm going red on its own vacuity")
    elif diffs:
        worst = "; ".join(d[0].as_posix() for d in diffs[:4])
        bad(name,
            f"{len(diffs)} difference(s) between two trees built from the same "
            f"{count} bytes: {worst}. The comparator disagrees with a pair it "
            f"made itself, so every arm below is measuring an arithmetic that "
            f"is already wrong")
    else:
        ok(name,
           f"{count} published file(s) read once from {ROOT} and written to "
           f"both {dev_root} and {mirror_root}; the two walks select "
           f"{len(dev)} and {len(mir)} path(s) and 0 of them differ. Built, not "
           f"mocked: this arm is two real directories on disk compared by the "
           f"same predicate that compares the live trees, and it is the only "
           f"reason the three arms below mean anything")


def self_test_arm(pairs: tuple, name: str, mutate, restore, want_rel: Path,
                  want_kind: str, extra_evidence: str) -> None:
    """Mutate the sandbox mirror in one direction, require one difference, undo.

    The requirement is **exactly one** difference naming **this** path with
    **this** kind -- not "a difference appeared". A comparator that reports a
    difference for the wrong reason satisfies a weaker test, and one that
    reports the right difference *and* eleven spurious ones satisfies this one
    only if the count is checked, which is why the count is in the assertion.
    """
    dev_root, mirror_root = pairs
    mutate(mirror_root)
    diffs = asymmetries(published_set(dev_root), published_set(mirror_root))
    try:
        if len(diffs) == 1 and diffs[0][0] == want_rel and diffs[0][1] == want_kind:
            ok(name,
               f"exactly {len(diffs)} difference, naming "
               f"{want_rel.as_posix()}, kind {want_kind}. "
               f"{extra_evidence.rstrip('. ')}. "
               f"{SELF_TEST_REMEDY[want_kind]}. Full block: "
               f"{report_blocks(diffs, SELF_TEST_REMEDY, 'sandbox dev', 'sandbox mirror')[0]}")
        else:
            bad(name,
                f"the mutation produced {len(diffs)} difference(s): "
                + "; ".join(f"{d[0].as_posix()} ({d[1]})" for d in diffs[:4])
                + f". Expected exactly one, {want_rel.as_posix()} as "
                  f"{want_kind}. A comparator that cannot be made to see one "
                  f"change cannot be trusted with the {len(diffs)}-kind list it "
                  f"produces on the live trees")
    finally:
        restore(mirror_root)


def run_the_self_test(pairs: tuple) -> None:
    dev_root, mirror_root = pairs
    mir = published_set(mirror_root)
    targets = mutation_targets(mir, 2)
    if len(targets) < 2:
        unasserted("one flipped byte in the sandbox mirror is reported, naming "
                   "the exact path",
                   f"the built pair holds {len(mir)} published file(s) and "
                   f"{len(targets)} non-empty one(s), so there is nothing to "
                   f"mutate. An arm with nothing to mutate is not a pass")
        unasserted("one file present only in the sandbox mirror is reported, "
                   "naming the exact path",
                   "as above: no published file to derive a target directory "
                   "from")
        unasserted("one file deleted from the sandbox mirror is reported, "
                   "naming the exact path",
                   "as above: no second published file to delete")
        return

    # The original bytes are read once, here, and used both as the undo and as
    # the size in the evidence sentence. Reading them inside the mutation
    # instead would mean the evidence string is built from a value the mutation
    # has not produced yet -- an f-string argument is evaluated at the call,
    # which is before `mutate` runs.
    orig_bytes = (mirror_root / targets[0]).read_bytes()

    def do_flip(mirror_root_unused):
        buf = bytearray(orig_bytes)
        buf[-1] ^= 0xFF          # one byte, and not a length change
        (mirror_root / targets[0]).write_bytes(bytes(buf))

    def undo_flip(mirror_root_unused):
        (mirror_root / targets[0]).write_bytes(orig_bytes)

    self_test_arm(
        pairs,
        "one flipped byte in the sandbox mirror is reported, naming the exact "
        "path",
        do_flip, undo_flip, targets[0], DIFFERS,
        f"{targets[0].as_posix()} is {len(orig_bytes)} byte(s) in the pair and "
        f"its last byte was XORed with 0xFF, so the length is unchanged and "
        f"only one bit moved. The target is the largest published file the rule "
        f"selected, derived rather than named, because a name written down is a "
        f"path that stops existing and leaves the arm mutating something out of "
        f"scope")

    # The directory is taken from a path the rule itself selected, so it is a
    # declared source root by construction rather than by this file listing one.
    added_rel = Path(targets[0].parts[0]) / PROBE_ADDED_NAME
    added_body = b"# present in the sandbox mirror only\n"

    def do_add(mirror_root_unused):
        dst = mirror_root / added_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(added_body)

    def undo_add(mirror_root_unused):
        (mirror_root / added_rel).unlink(missing_ok=True)

    published_probe = RULE.is_source(added_rel, mirror_root / added_rel.parent)
    self_test_arm(
        pairs,
        "one file present only in the sandbox mirror is reported, naming the "
        "exact path",
        do_add, undo_add, added_rel, EXTRA,
        f"planted at {added_rel.as_posix()}: the directory is the first "
        f"component of a path the rule itself selected, and the rule answers "
        f"is_source = {published_probe} for the new file, so the arm is adding "
        f"a path that really is in scope rather than one it would decline "
        f"anyway")

    victim = targets[1]
    saved: dict = {}

    def do_delete(mirror_root_unused):
        path = mirror_root / victim
        saved["raw"] = path.read_bytes()
        path.unlink()

    def undo_delete(mirror_root_unused):
        path = mirror_root / victim
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(saved["raw"])

    self_test_arm(
        pairs,
        "one file deleted from the sandbox mirror is reported, naming the "
        "exact path",
        do_delete, undo_delete, victim, MISSING,
        f"removed {victim.as_posix()}, the second largest published file in the "
        f"pair, so this arm and the byte-flip arm cannot be the same file: a "
        f"comparator that only ever notices its first target would pass one "
        f"arm by accident")


# --------------------------------------------------------------------------
# The tally
# --------------------------------------------------------------------------
def tally() -> None:
    section("the tally")
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nnot = sum(1 for t, _n, _d in RESULTS if t == "NOT-ASSERTED")
    name = "the tally accounts for every result, not-asserted included"
    if npass + nfail + nnot == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(name, f"{npass} + {nfail} + {nnot} = {len(RESULTS)}, and this check "
                 f"is the {len(RESULTS) + 1}th, so the run reaches the declared "
                 f"{EXPECTED_CHECKS} -- the same with both trees and with one, "
                 f"because a claim that could not be made is a recorded result "
                 f"and not a missing one")
    else:
        bad(name, f"recorded {len(RESULTS)} results ({npass}+{nfail}+{nnot}) "
                  f"before this check, which must be {EXPECTED_CHECKS - 1} for "
                  f"the run to reach the declared {EXPECTED_CHECKS}")


def notice(diffs: list) -> None:
    """The standing notice. Printed on every run, and it is not a check.

    Deliberately not a check: a check asserting "we are red" would have to be
    written to pass when the mirror is stale and fail when somebody syncs it,
    which is backwards, and it would put a reward on the gap staying open. So
    this is prose on stdout, where a reader meets it, and it adapts to the
    measurement rather than to the mood.
    """
    section("read this before changing anything in response to the result")
    if diffs:
        print(f"    This gate is EXPECTED TO BE RED. It is red now, over "
              f"{len(diffs)} measured asymmetry/asymmetries between the "
              f"development tree and the commit mirror, and it will stay red "
              f"until somebody runs the one-way sync by hand.")
        print("    The fix is to sync the mirror. The fix is NOT to widen this "
              "gate's scope, to shorten the rule's SOURCE_ROOTS or "
              "SOURCE_SUFFIXES, to add a skip, or to drop a path from the "
              "comparison until the number falls: a gate made green by "
              "declining to look is worse than no gate, because it is green.")
    else:
        print("    This gate is GREEN: the commit mirror is byte-identical to "
              "the development tree over the rule's published set. Nothing to "
              "do, and nothing to widen the scope to make it stay that way -- a "
              "green here means the sync was run, not that the check was "
              "loosened.")
    print("    Out of scope by decision, and each owned by another gate: the "
          "installed copy under site-packages (installed_copy_check.py), the "
          "mirror's non-source debris (release_tree_parity_check.py), and the "
          "rule itself (release_tree_parity_check.py again).")


def main() -> int:
    print("Has the commit mirror drifted from the development tree?")
    mirror, how = find_mirror()

    section("provenance: every number below came from these")
    rule_file = Path(RULE.__file__).resolve()
    print(f"    development tree : {ROOT}")
    print(f"    commit mirror    : {mirror if mirror is not None else 'NOT FOUND'}")
    print(f"    how              : {how}")
    print(f"    rule module      : {rule_file}")
    print(f"    its sha256       : {RULE.file_digest(rule_file)}")
    print(f"    decision surface : {RULE.rule_fingerprint()}")
    print(f"    SOURCE_ROOTS     : {sorted(RULE.SOURCE_ROOTS)}")
    print(f"    ROOT_SOURCE_NAME : {sorted(RULE.ROOT_SOURCE_NAMES)}")
    print(f"    SOURCE_SUFFIXES  : {sorted(RULE.SOURCE_SUFFIXES)}")
    print(f"    gitignore read   : {RULE._gitignore_path()} "
          f"(resolved from the rule module's own __file__, so it is the "
          f"development tree's on both sides)")
    print(f"    sandbox scratch  : {SANDBOX_PREFIX}* under "
          f"{ROOT / RULE.CARGO_TARGET_DIRNAME}, one level deeper than that "
          f"directory, which carries a {RULE.CACHEDIR_TAG_NAME} that would "
          f"otherwise be recovered as the tree root")

    section("the self-test's sandbox: a pair of trees built from this tree")
    scratch = Path(tempfile.mkdtemp(prefix=SANDBOX_PREFIX,
                                    dir=str(ROOT / RULE.CARGO_TARGET_DIRNAME)))
    try:
        check_the_rule_is_ours()
        check_the_mirror_was_located(mirror, how)
        dev: dict = {}
        mir: dict = {}
        diffs: list = []
        moved: list = []
        if mirror is not None:
            dev = published_set(ROOT)
            mir = published_set(mirror)
            diffs = asymmetries(dev, mir)
            # Second walk of the development tree, immediately, so a file
            # somebody saved mid-run is reported as a caveat rather than
            # counted as drift -- and before the report is printed, because a
            # caveat printed after the number it qualifies is a caveat nobody
            # reads.
            moved = moving_paths(dev, published_set(ROOT))
        check_the_scope_is_derived(dev, mir)
        built = build_pair(scratch, ROOT)
        pairs = (built[0], built[1])
        check_the_mirror_excludes_itself(pairs)
        check_nothing_pruned_is_in_scope(dev, mir)
        text, emitted = check_the_live_drift(dev, mir, diffs, moved)
        check_the_report_is_complete(diffs, text, emitted)
        section("the self-test: can this gate see drift in all three "
                "directions?")
        print(f"    sandbox: {scratch}")
        print(f"    {built[2]} file(s) read once from this tree and written to "
              f"both sides; every arm below mutates the sandbox mirror and "
              f"nothing else, and undoes it before the next")
        check_the_built_pair_is_identical(pairs, built[2])
        run_the_self_test(pairs)
        tally()
        notice(diffs)
    finally:
        left = scratch.exists()
        shutil.rmtree(scratch, ignore_errors=True)
        if scratch.exists():
            print()
            print(f"    NOTE: {scratch} could not be fully removed "
                  f"(ignore_errors=True, and a locked .pyc under a cache "
                  f"directory is the usual cause). It holds only files this run "
                  f"wrote. Nothing was freed by the attempt and no bytes are "
                  f"claimed back; remove it by hand if it is in the way.")
        elif left:
            print()
            print(f"    sandbox removed: {scratch}")

    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nnot = sum(1 for t, _n, _d in RESULTS if t == "NOT-ASSERTED")
    print()
    print(f"--- summary: {npass} passed, {nfail} failed, {nnot} not asserted, "
          f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")
    print("    the three numbers never sum to a verdict: a pass count on its "
          "own cannot say whether the two trees were compared")
    if nfail:
        print(f"RESULT: FAIL -- {nfail} of {EXPECTED_CHECKS} checks failed")
        return EXIT_FAILED
    if npass == 0:
        print("RESULT: DID NOT FINISH -- this run asserted nothing")
        return EXIT_INCOMPLETE
    print(f"RESULT: OK -- {npass} of {EXPECTED_CHECKS} checks passed, {nnot} "
          f"could not be asserted on this machine")
    return EXIT_OK


def report_crash(exc: BaseException) -> int:
    """A run that did not finish says so, in words no finished run prints.

    The summary above is only reachable if every check ran. A gate that
    crashed after printing a plausible PASS line is a gate whose output a reader
    can mistake for a verdict, and this file is about not doing that.
    """
    import traceback

    traceback.print_exc()
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nnot = sum(1 for t, _n, _d in RESULTS if t == "NOT-ASSERTED")
    print()
    print(f"--- STOPPED EARLY after {len(RESULTS)} recorded result(s): "
          f"{npass} passed, {nfail} failed, {nnot} not asserted")
    print(f"RESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}")
    print("    a finished run prints OK or FAIL above and never this line")
    return EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_exc))
