"""Do the two trees agree, and is the rule that says so auditable from the release?

**The promise this checks.** `_sync_release.py` copies the development tree into
the commit tree and asserts byte-equality at the end of a run a developer starts
by hand. That is a tool, not a contract: CI checks out the commit tree, never
sees the development tree, and could not have told anyone the mirror was stale
even if it had looked. Meanwhile the rule that decides *what is a source file*
lived in the tool, at the workspace root, where the tool's own rule excludes it
-- so the published artefact contained no copy of the decision that produced it.
The rule now lives in `release_tree_rule.py`, which ships, and this file is what
makes the promise falsifiable.

**What it can assert, and what it cannot, stated before the results.**

* On a maintainer's machine both trees exist, so the byte-for-byte comparison runs
  and names every file that disagrees, with two absolute paths and two digests
  per disagreement. A report that says "out of sync" is a note; this one says
  which file, which copy, and which bytes.
* On a CI runner **the development tree does not exist.** The workspace root is
  not a git repository at all -- only the commit tree is -- so there is nothing
  for a one-way copy to be compared against, and no runner-side step can invent
  one: cloning the same repository twice produces two identical trees, which
  would pass by construction and measure nothing. The three two-tree checks are
  therefore recorded as *not asserted*, with that reason, and they still count
  toward the total so the number is the same on both machine types. What runs
  everywhere instead is section C, which builds a shadow copy of the tree that
  does exist and mutates it in both directions to prove the comparison can go
  red, and section A, which audits the rule rather than the trees.
* The one direction no single-tree check can cover is *a source file that was
  never committed*. Inside the published tree, a file that is absent looks
  exactly like a file that was never written. That limitation is inherent to a
  one-way contract and is recorded here rather than papered over by a check that
  would trivially pass.

**The third copy.** Three copies of `opendocking` exist here: the development
tree, the commit tree, and the one the interpreter installed into its
`site-packages`. The third is out of scope by decision, not by oversight -- the
reason is a constant in the shipped rule, printed below, and check D1 proves the
boundary is a path property (nothing named `site-packages` lives inside either
published tree) rather than a promise in a comment. Its current state is
reported, including when it has drifted, because "deliberately out of scope" and
"nobody looks at it" have to look different.

Run it with no arguments from either tree. `OD_SYNC_DEST` (the tool's own switch)
and `OD_DEV_TREE` point it at the other tree; `OD_KEEP_SHADOW=1` keeps the
shadow copies for inspection instead of deleting them.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import site
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Python puts a script's own directory at sys.path[0], so this import works with
# no path juggling when the file is run as a script; the insert covers the other
# ways it gets loaded, and it adds the one directory rather than inventing a
# second route to the module.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import release_tree_rule as RULE  # noqa: E402  (after the path fix, on purpose)

#: How many results this file is supposed to record, counted by running it on
#: the development tree with both trees present. The pin is the same number
#: with one tree, because a check that cannot be asserted is *recorded* as
#: not-asserted rather than dropped -- a run that reported fewer results on CI
#: than locally would be a smaller run wearing a pass.
#: GATE-DECLARE 1
#: sites: 0 unconditional + 71 guarded
#: guards: sha256:7f3bd0f2423031167631f488d296fb5f395cbe380f9dcc260c1f1225a857eeb2
#:
#: 46 -> 71 sites, re-derived against the auditor rather than by hand, and the
#: reason the paragraph below is now in the past tense. The two-sites-per-check
#: shape is still what the file looks like and is still not re-verified for the
#: twenty-five added sites; what this block claims is only the site count, and
#: the results stay the pin on the line below.
#:
#: **0 unconditional is what the file's shape produces, not an omission.** Every
#: check in here is a verdict computed and then routed through `ok` or `bad`, so
#: every call site sits under an `if`; the guarded column is the whole file. The
#: sites and the results are different quantities and both are declared, and the
#: difference is now stated rather than left to be rediscovered: **each check is
#: an `if` with an `ok` in one branch and a `bad` in the other, so it is two call
#: sites and one result.** The table loop is one site running once per row.
#:
#: 34 -> 46 sites and 32 -> 41 results, and the two move by different things on
#: purpose. The twelve new sites are section E's six new checks at two sites
#: each: the rule stopped being an exclusion list, so this file no longer has
#: anything to assert about *what it leaves out*, and it now has to assert that a
#: file nobody declared is not shipped -- a different question with a different
#: shape. The nine new results are those six plus three more rows in the table
#: above, which grew from fifteen to eighteen: `odck.md` (must not ship although
#: six root `.md` files do), `scripts/release_tree_rule.pyc` (must not ship
#: although its root is declared), and `dock-py/.pytest_cache/README.md` (must
#: not ship although its root *and* its suffix are both declared -- that row is
#: the regression the inversion made, and section E's E6 is what caught it).
#:
#: **This block was not machine-verified when it was written, and that is worth
#: recording because the reason was right.** `check_scripts_declare.py` measures
#: the gates named in its `INVENTORY`, and this file was not in it -- it reported
#: three pinned gates missing from the inventory and this was one of them. So the
#: numbers above were derived by lifting the auditor's own `_sites_of` and
#: `_guards_digest` out of its source and running them against this file, rather
#: than by letting the census compare them. **The file is in `INVENTORY` now, the
#: comparison has run, and it went red: the lift was wrong by thirteen guarded
#: sites.** That is the outcome this paragraph predicted, and it is the whole
#: argument for the syntax-tree walk over lifting a function out of another
#: file: a lift is exactly as trustworthy as the care of whoever did it, and care
#: is not a mechanism. The two columns above are now measured by the thing that
#: compares them.
#:
#: The earlier `30 -> 34` movement, for the record: the table dispatched through
#: a variable (`verdict = bad, ...; verdict(...)`), which the auditor's
#: syntax-tree walk cannot see, so the census said 30 while the run reported 32.
#: Every site is now an explicit `ok`/`bad` call.
EXPECTED_CHECKS = 51

#: The classification table below, split so the expected verdict is written by
#: hand for every row. A table derived from the rule would agree with the rule
#: by construction and would survive any change to it, which is the opposite of a
#: control. Every CONTROL_CASES row is a case where the rule must say *no*, and
#: each carries the words its reason has to contain, so the reason text is
#: load-bearing too and not a decorative string.
#:
#: **No row here is `$null`.** That is the point of the inversion, and a table
#: that listed it would put the accident back into the shape the inversion
#: removed. `$null` is a 668-byte PowerShell redirect artefact at the workspace
#: root with no suffix; under `SOURCE_SUFFIXES` (which does not contain the empty
#: suffix) and `ROOT_SOURCE_NAMES` (which does not contain that name) it is not a
#: source file by construction, and section E proves it with a name nobody has
#: declared. The two control rows that *are* here are the two decisions that
#: genuinely needed a table: a root file that must not ship although six of its
#: neighbours do (`odck.md`), and a suffix under a declared root that must not
#: ship although its root is declared (`scripts/release_tree_rule.pyc`).
SOURCE_CASES = (
    "dock-py/python/opendocking/workbench/launcher.py",
    "dock-py/python/opendocking/_dockpy.pyd",
    "examples/1crn_receptor.pdb",
    "examples/benzene.sdf",
    "scripts/release_tree_rule.py",
    "scripts/release_tree_parity_check.py",
    "docs/VERIFICATION.md",
    ".github/workflows/ci.yml",
    "dock-core/Cargo.toml",
    "README.md",
)
CONTROL_CASES = (
    (".term-overlay/opendocking/_dockpy.pyd", "second copy of the package"),
    ("opendocking-gui/dock-py/python/opendocking/workbench/launcher.py",
     "second copy of the package"),
    ("vendor/site-packages/opendocking/core.py", "second copy of the package"),
    ("_sync_release.py", "root-level"),
    ("odck.md", "root-level"),
    ("target/debug/opendocking.dll", "not a declared source root"),
    ("scripts/release_tree_rule.pyc", "not a declared source suffix"),
    # NOT HERE: `dock-py/.pytest_cache/README.md`. It was a row in this table
    # until a clone of the published tree proved what a row in this table cannot
    # be: the table's verdict is asked with `abs_dir=None` unless the directory
    # exists, and the tag test needs the file. On a clean clone the directory is
    # absent, the rule had nothing to read, and the row went red saying "the
    # rule says shipped" -- for a tree on which the rule is correct. That is a
    # control that is red for the wrong reason, which is the mirror image of a
    # control that is green for the wrong one, and it is the same defect with
    # the sign flipped. It lives in section E now, where the directory is
    # planted, because a control case must be decidable from the path alone.
)

#: The minimum length for the out-of-scope declaration, so the check that it
#: exists is not satisfied by a word.
MIN_DECISION_CHARS = 120

RESULTS: list[tuple[str, str, str]] = []
UNASSERTED: list[tuple[str, str]] = []
EXIT_OK, EXIT_FAILED, EXIT_INCOMPLETE = 0, 1, 2


def section(title: str) -> None:
    print()
    print(f"-- {title}")


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def unasserted(name: str, why: str) -> None:
    """A check this machine cannot make, recorded rather than dropped.

    Deliberately not called `skip`: the suite's auditor classifies any gate that
    defines one by what its tally does, and this file's tally is simpler than
    that question. It counts the record, prints it in a third column that never
    sums into the verdict, and keeps the total constant across machines.
    """
    RESULTS.append(("NOT-ASSERTED", name, why))
    UNASSERTED.append((name, why))
    print(f"[NOT-ASSERTED] {name}\n       {why}")


# --------------------------------------------------------------------------
# Which two trees, and how this process learned
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Trees:
    development: Path | None
    published: Path
    how: str

    def describe(self) -> str:
        dev = str(self.development) if self.development else "ABSENT on this machine"
        return (f"development: {dev}\n"
                f"published  : {self.published}\n"
                f"decided by : {self.how}")


def resolve_trees() -> Trees:
    """Which tree am I in, and where is the other one?

    The commit tree is the one that *contains* a commit tree, so the question
    has an answer that is measured rather than configured: a checkout with
    `opendocking-gui/` inside it is the development tree, and a checkout without
    one is the published tree. `OD_SYNC_DEST` and `OD_DEV_TREE` override that for
    a shadow run, and the override that was used is reported, because a check that
    quietly compared the wrong pair of directories is the one failure this file
    exists to make impossible.
    """
    dest = os.environ.get("OD_SYNC_DEST")
    mirror = Path(dest).resolve() if dest else ROOT / RULE.RELEASE_TREE_DIRNAME
    if mirror.is_dir():
        return Trees(ROOT, mirror,
                     (f"OD_SYNC_DEST={dest} names the published tree"
                      if dest else
                      f"this checkout contains {mirror.name}/, so it is the "
                      f"development tree and the mirror beside it is the "
                      f"published one"))
    dev = os.environ.get("OD_DEV_TREE")
    if dev and Path(dev).is_dir():
        return Trees(Path(dev).resolve(), ROOT,
                     f"OD_DEV_TREE={dev} names the development tree; this "
                     f"checkout has no {RULE.RELEASE_TREE_DIRNAME}/, so it is "
                     f"the published one")
    if dest:
        why = (f"OD_SYNC_DEST={dest} names a directory that does not exist, and "
               f"this checkout has no {RULE.RELEASE_TREE_DIRNAME}/ either")
    else:
        why = (f"this checkout has no {RULE.RELEASE_TREE_DIRNAME}/ inside it, and "
               f"no OD_DEV_TREE was given, so there is one tree here: the "
               f"published one. The development tree is not under version "
               f"control (the workspace root is not a git repository) and CI "
               f"checks the commit tree out, so it cannot be reconstructed here "
               f"without cloning the same tree twice and measuring nothing")
    return Trees(None, ROOT, why)


# --------------------------------------------------------------------------
# The comparison, and the report it produces
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Difference:
    rel: str
    kind: str
    left: Path
    right: Path
    left_sha: str
    right_sha: str
    remedy: str
    left_name: str = "development"
    right_name: str = "published"

    def describe(self) -> str:
        """Two absolute paths and two digests, or an explicit absence.

        An absent file has no digest and is reported as `ABSENT` rather than
        given an empty hash, because a line with two paths and one empty string
        is a line a reader has to interpret. The kind says which side is missing,
        so "the mirror is stale" and "the tool would delete this" cannot be
        confused for each other.
        """
        return (
            f"{self.rel}: {self.kind}\n"
            f"        {self.left_name:<12} {self.left}\n"
            f"        {'':<12} sha256 {self.left_sha or 'ABSENT -- no such file'}\n"
            f"        {self.right_name:<12} {self.right}\n"
            f"        {'':<12} sha256 {self.right_sha or 'ABSENT -- no such file'}\n"
            f"        fix: {self.remedy}"
        )


#: ``{relative path: (absolute path, sha256)}``. A snapshot, not a live view:
#: the digest is taken once, from the bytes that were read, and the comparison
#: never re-reads a file to decide whether it changed. That is not an
#: optimisation. A live re-read races anybody editing the tree -- this workspace
#: has four agents in it, and the first run of this file reported a one-file
#: disagreement between a tree and a byte-for-byte copy of itself, because
#: `app.py` was saved between the copy and the comparison. A parity check that
#: can be made red by an unrelated editor is a check people learn to re-run.
Snapshot = dict[Path, tuple[Path, str]]


#: Digest memo, keyed on the file's own size and modification time. Not in the
#: rule module, deliberately: `file_digest` there is a function of bytes, and a
#: cache inside it would be a function of *when it was last asked*. This is a
#: property of one run of this file. A mutation changes the size or the mtime, so
#: it always misses -- the shadow comparisons below re-read 150 files four times,
#: and without this the check spends longer proving it works than the trees take
#: to disagree.
_DIGESTS: dict = {}


def _digest_of(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    got = _DIGESTS.get(key)
    if got is None:
        got = _DIGESTS[key] = RULE.file_digest(path)
    return got


def snapshot(root: Path) -> Snapshot:
    """Every source file under `root`, with the digest of the bytes read."""
    out: Snapshot = {}
    for rel, path in sorted(RULE.iter_source_files(root).items()):
        out[rel] = (path, _digest_of(path))
    return out


#: The three remedies, as data rather than as prose inside `compare`, because the
#: same three kinds of difference are reported in two contexts and the fix is not
#: the same sentence in both: against the commit tree it is "run the tool", and
#: against a shadow the check mutated itself it is "the mutation is what the
#: comparison is supposed to see". Printing the first in the second context would
#: tell a reader to re-run `_sync_release.py` because of a file the check wrote
#: to a temporary directory three lines earlier.
REMEDY = {
    "missing": "copy it across, or decide deliberately that the release should "
               "not carry it",
    # Reached whenever the rule has no objection, which is the common case: a
    # file renamed or moved in the development tree is still a perfectly good
    # source path, and what puts it in this bucket is only that the development
    # tree no longer has it. Saying so is the difference between a reader who
    # knows to look for a rename and one who goes looking for a rule change
    # that does not exist.
    "extra": "the rule still ships this path; the development tree no longer "
             "has the file, so the next run of the tool deletes it -- renamed, "
             "moved or dropped, and the published tree is what remembers",
    "differs": "the published tree is behind; re-run _sync_release.py, and if "
               "the difference is intended then the one-way promise is the thing "
               "that is wrong",
}
SHADOW_REMEDY = {
    "missing": "the file the check removed from the shadow is exactly what this "
               "comparison has to notice",
    "extra": "the file the check added to the shadow is exactly what this "
             "comparison has to notice",
    "differs": "the byte the check appended is exactly what this comparison has "
               "to notice",
}

KIND_MISSING = "present in one tree only (missing from the published tree)"
#: **Not "not a source".** That phrasing was here and it was wrong, and it was
#: wrong in the direction that costs a reader the most: it told them the rule
#: declines these paths, when the rule *ships* every one of them. The four
#: files this currently names -- `scripts/check_doc_encoding.py`,
#: `scripts/contacts_check.py`, `scripts/contacts_check2.py`,
#: `scripts/result_trust.py` -- are all `scripts/*.py`, which is a declared root
#: and a declared suffix, so `recognition_reason` returns `None` for all four,
#: meaning "this is a source". They are listed here because the *development
#: tree* no longer has them, not because the rule would exclude them: three were
#: renamed (`check_text_encoding.py`, `contacts_attribution_check.py`,
#: `contacts_criteria_check.py`) and one moved into the package
#: (`dock-py/python/opendocking/result_trust.py`, byte-identical to the stale
#: copy). A reader told "not a source" looks for a rule change and finds none,
#: because there is nothing to find.
KIND_EXTRA = "present in one tree only (in the published tree, not in the development tree)"
KIND_DIFFERS = "present in both trees, different bytes"


def compare(left_root: Path, left: Snapshot,
            right_root: Path, right: Snapshot,
            remedies: dict = None,
            left_name: str = "development",
            right_name: str = "published",
            ) -> list[Difference]:
    """Every way the two snapshots disagree, in all three directions.

    The roots are passed alongside the snapshots rather than recovered from
    them, so the "this file is missing" line can name where it *would* be. A
    report that names a path only on one side leaves the reader to guess which
    tree the other one came from, and that is the ambiguity this whole file is
    about.
    """
    remedies = REMEDY if remedies is None else remedies
    out: list[Difference] = []
    for rel in sorted(set(left) - set(right)):
        out.append(Difference(
            rel.as_posix(), KIND_MISSING,
            left[rel][0], right_root / rel,
            left[rel][1], "", remedies["missing"], left_name, right_name))
    for rel in sorted(set(right) - set(left)):
        reason = RULE.recognition_reason(rel, right[rel][0].parent)
        # `reason is None` means the rule *does* ship this path, so the file is
        # listed because the development tree dropped it, and a sentence that
        # named a recognition reason would be a lie. The tool's delete pass
        # removes it either way -- it deletes by membership in the development
        # tree, not by recognition -- which is why the text below is about the
        # tool rather than about the rule.
        if remedies is REMEDY and reason:
            remedy = (f"the rule does not ship it -- {reason} -- so the next "
                      f"run of _sync_release.py deletes it from the published "
                      f"tree")
        else:
            remedy = remedies["extra"]
        out.append(Difference(
            rel.as_posix(), KIND_EXTRA,
            left_root / rel, right[rel][0], "",
            right[rel][1], remedy, left_name, right_name))
    for rel in sorted(set(left) & set(right)):
        if left[rel][1] != right[rel][1]:
            out.append(Difference(
                rel.as_posix(), KIND_DIFFERS,
                left[rel][0], right[rel][0], left[rel][1], right[rel][1],
                remedies["differs"], left_name, right_name))
    return out


def _shadow_of(root: Path, into: Path) -> Snapshot:
    """Copy every source file of `root` into `into`, the way the tool would.

    Returns the **source** side of the comparison: each file's path in `root`,
    with the digest of the bytes that were written to the shadow. Two reasons it
    is not the destination and not a second read of the tree:

    * a report has to name two copies, and the copy that answered a question is
      the one in the tree, not the one in the temporary directory;
    * four other agents are editing this workspace, and the first run of this
      file reported a disagreement between the development tree and a
      byte-for-byte copy of itself, in `app.py`, because it was saved between the
      copy and the comparison. A parity check that an unrelated editor can turn
      red is a check people learn to re-run.
    """
    out: Snapshot = {}
    for rel, src in sorted(RULE.iter_source_files(root).items()):
        dst = into / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_bytes()
        dst.write_bytes(raw)
        out[rel] = (src, hashlib.sha256(raw).hexdigest())
    return out


def _published_contents(root: Path) -> Snapshot:
    """Every file in the published tree except `.git/`, digested.

    **Not the rule's own view, on purpose.** The right-hand side of the
    two-tree comparison has to be the set the *tool* enumerates when it deletes
    what is no longer a source, and the tool walks the commit tree raw. Taking
    the rule's pruned view here instead made the "nothing the rule would delete"
    check unable to see a single file: five PNGs under `dist/` are in the commit
    tree now and the next sync deletes them, and a pruned comparison cannot
    report a file whose whole directory is pruned out of the question. A check
    that cannot see the state it was written for is worse than no check, because
    it is green.
    """
    out: dict[Path, tuple[Path, str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(n for n in dirnames if n != ".git")
        for name in sorted(filenames):
            rel = rel_dir / name if rel_dir.parts else Path(name)
            path = Path(dirpath) / name
            out[rel] = (path, _digest_of(path))
    return out


def _unpruned_sources(root: Path) -> dict[Path, Path]:
    """The same membership decision, reached without `prune_here`."""
    out: dict[Path, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            rel = rel_dir / name if rel_dir.parts else Path(name)
            if RULE.is_source(rel, Path(dirpath)):
                out[rel] = Path(dirpath) / name
    return out



# --------------------------------------------------------------------------
# Section A -- the rule: one implementation, and it ships
# --------------------------------------------------------------------------
def check_the_rule_ships(trees: Trees) -> None:
    section("the rule is one rule, and the published tree carries it")
    loaded = Path(RULE.__file__).resolve()
    mine = (HERE / "release_tree_rule.py").resolve()
    if loaded == mine:
        ok("the rule this run used is this checkout's own copy",
           f"{loaded}  sha256 {RULE.file_digest(loaded)[:16]}...  "
           f"decision surface sha256 {RULE.rule_fingerprint()[:16]}...")
    else:
        bad("the rule this run used is this checkout's own copy",
            f"imported {loaded}, expected {mine}. Every number below was produced "
            f"by whichever of the two answered, and a stale rule would answer "
            f"confidently")

    wanted_excluded = [(p, False, "") for p in SOURCE_CASES]
    wanted_excluded += [(p, True, kw) for p, kw in CONTROL_CASES]
    for rel, should_exclude, keyword in wanted_excluded:
        # `ok` and `bad` are called by name in both branches rather than through a
        # `verdict` variable holding one of them. The suite's auditor derives this
        # file's call-site census from the syntax tree and matches the *name* of
        # the call, so a site reached through a variable is a site it cannot
        # see: the census would have read 30 while the run reported 32, and the
        # two numbers are supposed to be the same question. One site, fifteen
        # invocations, which is the granularity the census documents.
        _abs = ROOT / rel
        reason = RULE.recognition_reason(
            Path(rel), _abs.parent if _abs.parent.is_dir() else None)
        got = reason is not None
        name = (f"the rule classifies {rel} as "
                f"{'a non-source' if should_exclude else 'a source'}")
        if got != should_exclude:
            bad(name,
                f"the table says this "
                f"{'must be excluded' if should_exclude else 'must ship'}, the "
                f"rule says {'excluded' if got else 'shipped'} "
                f"({reason or 'no reason given'})")
        elif got and keyword and keyword not in reason:
            bad(name,
                f"excluded, but the reason does not mention {keyword!r}: {reason}")
        elif got:
            ok(name, f"excluded: {reason}")
        else:
            ok(name, "shipped: a source of the release")

    missing = [p for p in SOURCE_CASES if not (ROOT / p).is_file()]
    if missing:
        bad("every path the table calls a source exists in this tree",
            f"absent here: {missing}. A table of paths that are not in the tree "
            f"is a table of fiction, and the rows above would be testing nothing")
    else:
        ok("every path the table calls a source exists in this tree",
           f"all {len(SOURCE_CASES)} exist under {ROOT}")

    excluded_me = [p for p in ("release_tree_rule.py", "release_tree_parity_check.py")
                   if not RULE.is_source(Path("scripts") / p)]
    if excluded_me:
        bad("the rule ships the rule and the check that audits it",
            f"excluded: {excluded_me}. The published tree would then contain no "
            f"copy of the decision that decided what it contains, which is the "
            f"state this file was written to end")
    else:
        ok("the rule ships the rule and the check that audits it",
           "scripts/release_tree_rule.py and scripts/release_tree_parity_check.py "
           "are both sources under the rule's own tables")

    others = _second_implementations(ROOT)
    if others:
        detail = "; ".join(f"{p} defines {what}" for p, what in others)
        bad("no second implementation of the rule in this tree", detail)
    else:
        ok("no second implementation of the rule in this tree",
           f"parsed every .py under {ROOT} for the rule's "
           f"{len(RULE.RULE_FUNCTIONS)} functions and {len(RULE.RULE_TABLES)} "
           f"tables; only {RULE.__name__}.py defines them")

    copies = RULE.iter_package_copies(trees.published)
    if copies:
        detail = "; ".join(
            f"{rel} -- {why} -- {trees.published / rel}" for rel, why in copies[:6])
        bad("the published tree contains no second copy of the package",
            f"{len(copies)} file(s): {detail}"
            + ("" if len(copies) <= 6 else " ..."))
    else:
        ok("the published tree contains no second copy of the package",
           f"walked {trees.published} without pruning, because pruning skips "
           f"exactly the directories this question is about, and found none")


def _second_implementations(root: Path) -> list[tuple[str, str]]:
    """Files other than the rule module that define its functions or tables.

    Read from the syntax tree, and a file that does not parse is scanned by text
    rather than skipped: `scripts/redock_benchmark.py` begins with a UTF-8 BOM,
    which `ast.parse` rejects on every interpreter version even though Python
    imports it perfectly well. A reader that gave up on such a file would report
    a clean tree for an unreadable one, which is the shape of failure this whole
    file is about.
    """
    out: list[tuple[str, str]] = []
    rule_file = (HERE / "release_tree_rule.py").resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        # Pruned with the rule's own `prune_here`, for the same reason the walk
        # in `iter_source_files` is: a second copy of the rule inside `target/`
        # or `.git` is not something the release can ship, so looking for one
        # there costs a directory traversal and finds nothing that matters.
        dirnames[:] = sorted(
            name for name in dirnames
            if not RULE.prune_here(rel_dir / name if rel_dir.parts else Path(name)))
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            rel = rel_dir / name if rel_dir.parts else Path(name)
            if not RULE.is_source(rel, Path(dirpath)) or path.resolve() == rule_file:
                continue
            raw = path.read_bytes().decode("utf-8", errors="replace")
            try:
                tree = ast.parse(raw.lstrip("\ufeff"))
            except SyntaxError:
                for line in raw.splitlines():
                    stripped = line.strip()
                    for fname in RULE.RULE_FUNCTIONS:
                        if stripped.startswith(f"def {fname}(") \
                                or stripped.startswith(f"def {fname} ("):
                            out.append((str(path), f"def {fname}() (text scan: "
                                              f"this file does not parse)"))
                    for tname in RULE.RULE_TABLES:
                        if stripped.startswith(f"{tname} =") \
                                or stripped.startswith(f"{tname}:") \
                                or stripped.startswith(f"{tname}="):
                            out.append((str(path), f"{tname} (text scan: this "
                                              f"file does not parse)"))
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.name in RULE.RULE_FUNCTIONS:
                    out.append((str(path), f"def {node.name}()"))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) \
                                and target.id in RULE.RULE_TABLES:
                            out.append((str(path), f"{target.id} = ..."))
    return out


# --------------------------------------------------------------------------
# Section B -- the two trees
# --------------------------------------------------------------------------
def check_the_trees_agree(trees: Trees) -> None:
    section("the two trees agree, byte for byte")
    if trees.development is None:
        why = (f"{trees.published} is the only tree here, so there is nothing to "
               f"compare it against: {trees.how}")
        for name in ("no source file is missing from the published tree",
                     "the published tree holds nothing the rule would delete",
                     "every file in both trees is byte-identical"):
            unasserted(name, why)
        return
    dev_files = snapshot(trees.development)
    rel_files = _published_contents(trees.published)
    diffs = compare(trees.development, dev_files, trees.published, rel_files)
    print(f"    development tree selects {len(dev_files)} source file(s); the "
          f"published tree holds {len(rel_files)} (every file except .git, "
          f"which is the set the tool's delete pass walks)")
    by_kind: dict[str, list[Difference]] = {}
    for d in diffs:
        by_kind.setdefault(d.kind, []).append(d)
    plans = (
        ("no source file is missing from the published tree",
         KIND_MISSING,
         "the published tree is behind the development tree. Every file listed "
         "here is one the rule says belongs in the release and the release does "
         "not have it"),
        ("the published tree holds nothing the rule would delete",
         KIND_EXTRA,
         "the published tree carries files the development tree does not. Most "
         "of them are paths the rule still ships -- renamed or moved rather than "
         "declined -- and the tool's delete pass removes them by membership in "
         "the development tree, not by recognition, so each line below is "
         "published now and gone later, which is the state that reads as "
         "intentional until it is not"),
        ("every file in both trees is byte-identical",
         KIND_DIFFERS,
         "the two trees disagree about the contents of a file that is in both"),
    )
    for name, kind, why in plans:
        found = by_kind.get(kind, [])
        if found:
            shown = "; ".join(f"\n       {d.describe()}" for d in found[:4])
            more = "" if len(found) <= 4 else f"\n       ... and {len(found) - 4} more"
            bad(name, f"{len(found)} of them{shown}{more}\n       {why}")
        else:
            ok(name, why)


# --------------------------------------------------------------------------
# Section C -- the comparison can go red
# --------------------------------------------------------------------------
def check_the_comparison_can_go_red(trees: Trees, tmp: Path) -> None:
    section("the comparison can go red (shadow copies, both directions)")
    marker = Path("scripts/release_tree_parity_check.py")
    clean = tmp / "shadow-of-this-tree"

    def diffs_now(original: Snapshot) -> list[Difference]:
        """The comparison under test, against the shadow as it stands *now*.

        The right-hand set is the shadow's own walk rather than the set that was
        copied into it: handing `compare` the original set would make the
        deleted-file and added-file mutations below report a difference for the
        wrong reason -- a set that had changed rather than a file that had.
        """
        return compare(ROOT, original, clean, snapshot(clean),
                       remedies=SHADOW_REMEDY, left_name="original",
                       right_name="shadow")

    shadow = _shadow_of(ROOT, clean)
    # The left side is the bytes that were copied out of this tree, keyed to the
    # file they came from; the right side is re-read from the shadow each time.
    original = shadow
    base = original

    diffs = diffs_now(original)
    if not diffs:
        ok("an unmutated copy of this tree is byte-identical to it",
           f"{len(base)} source file(s) copied by the rule and compared back; "
           f"nothing to report, which is the only state in which the mutations "
           f"below mean anything")
    else:
        bad("an unmutated copy of this tree is byte-identical to it",
            f"{len(diffs)} difference(s) on an unmutated copy, so every mutation "
            f"below is measuring a comparator that already disagrees: "
            + "; ".join(d.describe().splitlines()[0] for d in diffs[:4]))

    target = clean / marker
    original_bytes = target.read_bytes()

    target.write_bytes(original_bytes + b"\n# appended by the parity check's mutation\n")
    hit = [d for d in diffs_now(original)
           if d.rel == marker.as_posix() and d.kind == KIND_DIFFERS]
    if hit and hit[0].left_sha and hit[0].right_sha and hit[0].left.is_absolute() \
            and hit[0].right.is_absolute():
        ok("a changed byte is reported with both copies and both digests",
           hit[0].describe().replace("\n", "\n       "))
    else:
        bad("a changed byte is reported with both copies and both digests",
            f"appending a line to {marker} in the shadow produced "
            f"{len(hit)} difference(s) naming this file as differing; a "
            f"comparator that cannot see a one-byte change cannot be trusted to "
            f"see a 12 000-byte one")
    target.write_bytes(original_bytes)

    target.unlink()
    gone = [d for d in diffs_now(original)
            if d.rel == marker.as_posix() and d.kind == KIND_MISSING]
    if gone and gone[0].left.is_absolute() and gone[0].right.is_absolute():
        ok("a file deleted from one side is reported with both copies",
           gone[0].describe().replace("\n", "\n       "))
    else:
        bad("a file deleted from one side is reported with both copies",
            f"removing {marker} from the shadow produced {len(gone)} "
            f"difference(s) naming this file as missing")
    target.write_bytes(original_bytes)

    extra_rel = Path("scripts/_shadow_only.py")
    extra = clean / extra_rel
    extra.write_text("# present in the shadow only\n", encoding="utf-8", newline="")
    surplus = [d for d in diffs_now(original) if d.rel == extra_rel.as_posix()]
    if surplus and surplus[0].right.is_absolute():
        ok("a file present in only one tree is reported with both copies",
           surplus[0].describe().replace("\n", "\n       "))
    else:
        bad("a file present in only one tree is reported with both copies",
            f"adding {extra_rel.as_posix()} to the shadow produced "
            f"{len(surplus)} difference(s) naming it")
    extra.unlink()

    # The mirror check, in the direction that proves it can bite: a copy of the
    # package planted in a shadow of the published tree must be *found*, and the
    # walk that looks for it must not be the pruned one.
    shadow_rel = clean if trees.published == ROOT else tmp / "shadow-of-published"
    if shadow_rel is not clean:
        _shadow_of(trees.published, shadow_rel)
    planted_rel = Path(".term-overlay/opendocking/_dockpy.pyd")
    planted = shadow_rel / planted_rel
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"MZ" + b"\0" * (748 * 1024 - 2))
    found = [rel for rel, _why in RULE.iter_package_copies(shadow_rel)]
    if planted_rel in found:
        ok("a planted second copy of the package is named in the published tree",
           f"{planted} found by the unpruned walk ({len(found)} copy/copies "
           f"under {shadow_rel}); this is the case a pruned walk reports as "
           f"clean, because it prunes the very directory it is looking for")
    else:
        bad("a planted second copy of the package is named in the published tree",
            f"planted {planted} and the scan of {shadow_rel} returned "
            f"{[r.as_posix() for r in found]}. A check that cannot see a second "
            f"copy of the package cannot hold up the rule that excludes one")
    planted.unlink()

    roots = [trees.published] + ([ROOT] if trees.development else [])
    mismatched = []
    for root in roots:
        fast = set(RULE.iter_source_files(root))
        slow = set(_unpruned_sources(root))
        if fast != slow:
            mismatched.append(
                f"{root}: pruned walk selects {len(fast)}, unpruned {len(slow)}, "
                f"only-pruned {sorted(fast - slow)[:3]}, only-unpruned "
                f"{sorted(slow - fast)[:3]}")
    if mismatched:
        bad("the pruned walk and an unpruned one select the same files",
            "; ".join(mismatched) + ". `prune_here` is allowed to skip a "
            "directory only when nothing under it can be a source, and that is "
            "a claim about the tables, so it is measured rather than assumed")
    else:
        ok("the pruned walk and an unpruned one select the same files",
           "checked " + ", ".join(str(r) for r in roots) +
           "; the pruning is sound, so skipping target/ changes nothing but the "
           "time taken")


# --------------------------------------------------------------------------
# Section E -- the recognition rule, in both directions
# --------------------------------------------------------------------------
def _root_files(root: Path) -> list[str]:
    """Every file sitting directly at the top of `root`, by name."""
    return sorted(p.name for p in Path(root).iterdir() if p.is_file())


def check_recognition(tmp: Path) -> None:
    """Both directions of a recognition rule, on a tree that is actually built.

    The inversion from an exclusion list to a recognition rule has one property
    worth testing and one property worth refusing, and they are opposite.

    The property worth testing is that **a name nobody declared is not shipped.**
    Under the old rule that was false by default: `$null`, a 668-byte
    PowerShell redirect artefact at the workspace root, was classified as a
    source file and would be copied into every release until somebody added it to
    a table. Under this rule it is not shipped because its suffix is empty and
    its name is not in `ROOT_SOURCE_NAMES` -- nobody thought about it. The only
    way to *show* that rather than assert it is to plant a file nobody has ever
    heard of and watch the rule decline it, which is what the synthetic tree
    below is for. `$null` itself is deliberately not named anywhere in this
    file: a table row for it would put the accident back into the shape the
    inversion removed, and E1 covers it without a word.

    The property worth refusing is a check that can only go green. So the four
    checks below are **not** four independent assertions that something is
    absent -- three of them are about absence, and an empty result would satisfy
    all three. E3 is the positive direction, it requires the walk to have
    selected the planted `.py`, and its detail prints the whole selected set.
    Read as a group they cannot pass vacuously: if the rule stopped selecting
    anything at all, E2 and E4 would stay green and E3 would go red.

    The synthetic tree is used rather than a shadow of this one because the
    question is about the rule's path logic and not about the contents of 150
    files; a shadow would answer the same thing forty times slower.
    """
    section("the recognition rule: declared in, undeclared out, both measured")

    # -- E1: the root table, from both sides, on the trees that are here ----
    roots = [("development", ROOT)] + (
        [("published", ROOT / RULE.RELEASE_TREE_DIRNAME)]
        if (ROOT / RULE.RELEASE_TREE_DIRNAME).is_dir() else [])
    problems = []
    for label, root in roots:
        present = set(_root_files(root))
        shipped = {r.name for r in RULE.iter_source_files(root)
                   if len(r.parts) == 1}
        missing = sorted(RULE.ROOT_SOURCE_NAMES - present)
        smuggled = sorted(shipped - RULE.ROOT_SOURCE_NAMES)
        not_shipped = sorted(present - shipped)
        if missing:
            problems.append(f"{label}: declared but absent {missing}")
        if smuggled:
            problems.append(f"{label}: shipped without being declared {smuggled}")
        if not shipped:
            problems.append(f"{label}: nothing at the root is shipped at all")
        print(f"    {label} tree {root}: {len(present)} file(s) at the root, "
              f"{len(shipped)} shipped, {len(not_shipped)} not: "
              f"{not_shipped}")
    if problems:
        bad("the root table is a claim about what ships, and it is checkable",
            "; ".join(problems) + ". Every declared name has to exist (a table of "
            "fiction tests nothing) and no undeclared name may ship (which is "
            "the property the inversion was for: an unlisted file at the root is "
            "declined without anyone having listed it)")
    else:
        ok("the root table is a claim about what ships, and it is checkable",
           f"{len(RULE.ROOT_SOURCE_NAMES)} declared name(s) all exist and all "
           f"ship, in {len(roots)} tree(s); and every file at the root that does "
           f"not ship is accounted for by not being declared. No name is "
           f"written down here: the walk reads the directory")

    # -- the synthetic tree, built on disk and walked by the real walk ------
    tree = tmp / "recognition"
    tag = b"Signature: 8a477f597d28d172789f06886806bc55\n# a cache directory tag\n"
    # A cache directory whose name is in no table anywhere in this repository,
    # because the rule does not read names for this. Written before the walk so
    # that the walk sees a real directory carrying a real tag.
    real_cache = tree / "dock-py" / ".pytest_cache"
    real_cache.mkdir(parents=True, exist_ok=True)
    (real_cache / RULE.CACHEDIR_TAG_NAME).write_bytes(tag)
    (real_cache / "README.md").write_bytes(b"planted\n")
    future = tree / "scripts" / ".zz_future_tool_cache"
    future.mkdir(parents=True, exist_ok=True)
    (future / RULE.CACHEDIR_TAG_NAME).write_bytes(tag)
    planted = {
        "declared suffix, declared root": tree / "scripts" / "zz_planted_source.py",
        "declared suffix, declared root, unremarkable": tree / "scripts" / "zz_ordinary.md",
        "no suffix, declared root": tree / "scripts" / "zz_planted_accident",
        "declared suffix, undeclared root": tree / "vendor" / "zz_planted_vendor.py",
        "build-artefact suffix, declared root":
            tree / "scripts" / "zz_planted_source.pyc",
        "declared suffix, declared root, generated directory":
            tree / "scripts" / "__pycache__" / "zz_planted_source.py",
        "declared suffix, declared root, self-declared cache":
            future / "README.md",
    }
    for path in planted.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"planted by the recognition mutation\n")
    # -- a planted crate, so the Cargo derivation is tested rather than argued
    crate = tree / "dock-py"
    crate.mkdir(parents=True, exist_ok=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "zz-planted"\nversion = "0.0.0"\n', encoding="utf-8",
        newline="")
    (crate / "src").mkdir(parents=True, exist_ok=True)
    (crate / "src" / "lib.rs").write_text("pub fn f() {}\n", encoding="utf-8",
                                          newline="")
    built = crate / "target" / "debug" / "zz_planted.rs"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_text("not hand-written\n", encoding="utf-8", newline="")
    # The over-broad direction: a directory with Cargo's name and no manifest.
    (tree / "scripts" / "target").mkdir(parents=True, exist_ok=True)
    (tree / "scripts" / "target" / "zz_real.py").write_text("x = 1\n",
                                                            encoding="utf-8",
                                                            newline="")
    # The boundary: what the other ecosystems leave behind, with a declared suffix.
    nm = tree / "dock-py" / "node_modules" / "zz_pkg"
    nm.mkdir(parents=True, exist_ok=True)
    (nm / "README.md").write_text("third-party\n", encoding="utf-8", newline="")
    planted["declared suffix, beside a planted Cargo.toml"] = crate / "src" / "lib.rs"
    planted["declared suffix, in a planted crate's target dir"] = built
    planted["declared suffix, in a manifest-less target dir"] = (
        tree / "scripts" / "target" / "zz_real.py")
    planted["declared suffix, in a planted node_modules"] = nm / "README.md"
    selected = {rel.as_posix() for rel in RULE.iter_source_files(tree)}
    print(f"    synthetic tree {tree}: the rule selected {len(selected)} of the "
          f"{len(planted)} planted file(s): {sorted(selected)}")

    # -- E2: no suffix, inside a root that IS declared ---------------------
    undeclared_suffix = "scripts/zz_planted_accident"
    reason = RULE.recognition_reason(Path(undeclared_suffix))
    if undeclared_suffix not in selected and reason and "suffix" in reason:
        ok("an unfamiliar suffix-less file inside a source root is not shipped",
           f"{tree / undeclared_suffix} is not in the selected set, and the rule "
           f"gives its reason without anybody having named it: {reason}")
    else:
        bad("an unfamiliar suffix-less file inside a source root is not shipped",
            f"planted {tree / undeclared_suffix} under a declared source root "
            f"with no suffix; the walk selected "
            f"{undeclared_suffix in selected} and the reason is "
            f"{reason!r}. This is the accident the inversion was for, and it has "
            f"to fail here as well as in the abstract")

    # -- E3: a declared suffix inside a declared root IS shipped -----------
    # Two files, because "declared suffix" and "unremarkable" are different
    # claims and the second one is the one a name-based rule would fail: a
    # `.md` under `scripts/` looks exactly like the file that has to be
    # declined (`dock-py/.pytest_cache/README.md`), and the only thing that
    # tells them apart is whether the directory it is in declares itself a
    # cache. This is the negative direction of E7, in the same section, so the
    # two cannot both be green by accident.
    declared = "scripts/zz_planted_source.py"
    ordinary = "scripts/zz_ordinary.md"
    if (declared in selected and ordinary in selected
            and not RULE.recognition_reason(Path(declared))
            and not RULE.recognition_reason(Path(ordinary))):
        ok("a genuine source file with a declared suffix is shipped",
           f"{tree / declared} and {tree / ordinary} are both in the selected "
           f"set with no exclusion reason -- a `.py` and a `.md`, the second "
           f"chosen because it is the same suffix as the file E7 declines. "
           f"This is the direction that keeps E2, E4, E5, E6 and E7 from "
           f"passing vacuously: a rule that selected nothing at all would "
           f"satisfy every absence check in this section")
    else:
        bad("a genuine source file with a declared suffix is shipped",
            f"planted {tree / declared} and {tree / ordinary}, both declared "
            f"suffixes under a declared root, and the walk selected "
            f"{declared in selected} and {ordinary in selected} "
            f"(selected: {sorted(selected)}). The rule has stopped "
            f"recognising its own table, and every absence check above is "
            f"meaningless while that is true")

    # -- E4: the same suffix under a root that is NOT declared -------------
    undeclared_root = "vendor/zz_planted_vendor.py"
    reason = RULE.recognition_reason(Path(undeclared_root))
    if undeclared_root not in selected and reason and "root" in reason:
        ok("a declared suffix under an undeclared root is not shipped",
           f"{tree / undeclared_root} carries the same declared suffix as the "
           f"file E3 shipped, and is not selected: {reason}. The two halves of "
           f"the recognition are both load-bearing, and this is the half that "
           f"keeps `dist/`, `target/` and the commit tree beside this one out "
           f"without naming them")
    else:
        bad("a declared suffix under an undeclared root is not shipped",
            f"planted {tree / undeclared_root} and the walk selected "
            f"{undeclared_root in selected} (reason {reason!r}). Membership is "
            f"supposed to need both halves")

    # -- E5: a build-artefact suffix under a declared root -----------------
    artefact = "scripts/zz_planted_source.pyc"
    reason = RULE.recognition_reason(Path(artefact))
    if artefact not in selected and reason and "suffix" in reason:
        ok("a build-artefact suffix under a declared root is not shipped",
           f"{tree / artefact} is not selected, and no row anywhere says "
           f"'.pyc': the artefact suffixes are gone from the rule rather than "
           f"enumerated in it, so the next artefact suffix does not need adding "
           f"either")
    else:
        bad("a build-artefact suffix under a declared root is not shipped",
            f"planted {tree / artefact} and the walk selected "
            f"{artefact in selected} (reason {reason!r}). The old rule excluded "
            f"this suffix by name; the new rule declines it by not recognising "
            f"it, and that is only true if it really is absent from "
            f"SOURCE_SUFFIXES")


    # -- E6: the same declared suffix inside a GENERATED directory ----------
    generated = "scripts/__pycache__/zz_planted_source.py"
    reason = RULE.recognition_reason(Path(generated))
    if generated not in selected and reason and "generated directory" in reason:
        ok("a declared suffix inside a generated directory is not shipped",
           f"{tree / generated} carries the same declared suffix as the file E3 "
           f"shipped, in the same declared root, and is declined: {reason}. This "
           f"is the regression the recognition rule made and this check is what "
           f"caught it: 'under a declared root' alone shipped "
           f"dock-py/.pytest_cache/README.md, because pytest writes its cache "
           f"into whatever directory it was run from and a `.md` under `dock-py/` "
           f"passes both halves of the rule")
    else:
        bad("a declared suffix inside a generated directory is not shipped",
            f"planted {tree / generated} and the walk selected "
            f"{generated in selected} (reason {reason!r}). `GENERATED_DIRS` is "
            f"matched at any depth, and this is the only check that says so")


    # -- E7: a self-declared cache, whatever the directory is called --------
    real_cached = "dock-py/.pytest_cache/README.md"
    if real_cached in selected:
        bad("a self-declared cache directory is not shipped, whatever it is called",
            f"{tree / real_cached} was selected. This is the row that used to sit "
            f"in CONTROL_CASES, where it needed the directory to exist and so was "
            f"red on a clean clone for the wrong reason")
    cached = "scripts/.zz_future_tool_cache/README.md"
    reason = RULE.recognition_reason(Path(cached), tree / "scripts" / ".zz_future_tool_cache")
    if cached not in selected and reason and "cache directory" in reason:
        ok("a self-declared cache directory is not shipped, whatever it is called",
           f"{tree / cached} is declined, and so is every other file under the "
           f"directory, because the directory carries a {RULE.CACHEDIR_TAG_NAME} "
           f"with the cachedir signature: {reason[:120]}... The name "
           f"`.zz_future_tool_cache` appears in no table in "
           f"{RULE.__name__}.py -- `GENERATED_DIRS` holds "
           f"{sorted(RULE.GENERATED_DIRS)}, and this directory is not one of "
           f"them. That is the whole of the derived half of the rule: a "
           f"directory that says out loud that a tool rewrites it is not "
           f"source, and the tool that wrote the tag is pytest, mypy, pip, "
           f"black or ruff, present or future")
    else:
        bad("a self-declared cache directory is not shipped, whatever it is called",
            f"planted {tree / cached} inside a directory carrying the cachedir "
            f"signature, and the walk selected {cached in selected} (reason "
            f"{reason!r}). This is the only check that says the rule reads a "
            f"declaration rather than a list of names")


    # -- E8: a crate's build tree, derived before anything has been built ----
    built = "dock-py/target/debug/zz_planted.rs"
    src = "dock-py/src/lib.rs"
    reason = RULE.recognition_reason(Path(built), tree / "dock-py" / "target" / "debug")
    if built not in selected and src in selected and reason and "Cargo" in reason:
        ok("a crate's build directory is derived from its manifest, not its name",
           f"{tree / built} is not selected, and {tree / src} -- a real Rust "
           f"source in the same crate, one directory up -- is: {reason[:130]}... "
           f"Nothing has been built into the planted directory, so this is the "
           f"one test here that does not depend on a run having happened first")
    else:
        bad("a crate's build directory is derived from its manifest, not its name",
            f"planted a manifest-bearing crate and selected "
            f"{built in selected} for the build tree and {src in selected} for "
            f"its source (reason {reason!r}). Both halves have to hold: a rule "
            f"that declines the crate's own source is not a fix")

    # -- E9: the over-broad direction, and a name with no manifest ----------
    bare = "scripts/target/zz_real.py"
    if bare in selected:
        ok("a directory named target with no manifest above it still ships",
           f"{tree / bare} is selected. `is_manifest_build_dir` asks for a "
           f"`{RULE.CARGO_MANIFEST_NAME}` one level up, so the popular name on "
           f"its own decides nothing -- the over-broad version of this rule "
           f"would decline it, and this is the check that says so")
    else:
        bad("a directory named target with no manifest above it still ships",
            f"planted {tree / bare} and the walk did not select it. The "
            f"derivation has widened from a manifest to a name, which would "
            f"decline any directory called `target` in any project")

    # -- E10: the boundary, pinned rather than left to be discovered --------
    node = "dock-py/node_modules/zz_pkg/README.md"
    if node in selected:
        ok("the boundary of the derivation is pinned: a non-Cargo build tree ships",
           f"{tree / node} IS selected, and that is the declared limit rather "
           f"than an oversight. The Cargo derivation is the only manifest-driven "
           f"one available here: `pyproject.toml` declares no build directory, "
           f"PEP 517's `build/`, `dist/` and `*.egg-info/` are frontend "
           f"conventions rather than manifest content, and this tree has no "
           f"`package.json` at all. If a future manifest makes an output "
           f"directory derivable, this check goes red and the pin moves with it")
    else:
        bad("the boundary of the derivation is pinned: a non-Cargo build tree ships",
            f"{tree / node} was not selected, so the rule now covers a case "
            f"this pin says it does not. Either a derivation was added (good, "
            f"update the text above and this check) or the rule became "
            f"over-broad (not good, find which)")


#: Every non-comment line in the shipped `.gitignore`, with the answer the rule
#: gives and **how that answer is reached**. This is the table that makes the
#: decline-to-read decision checkable instead of a judgement: the rule now reads
#: `.gitignore` for directory literals, so the two declarations are one subject
#: with two answers, and a line that appears in the file without appearing here
#: is a line nobody answered.
#:
#: `("declined", "<the mechanism>")` -- the rule must refuse a probe carrying it.
#: `("shipped",  "<the mechanism>")` -- the rule must publish one, because the
#: `.gitignore` line is wrong or narrower than the rule's own tables.
#: `("unimplemented", "<what it would take>")` -- the reader has no opinion, and
#: that is stated here rather than left to a silent mismatch.
#:
#: One key was **removed** rather than reclassified, and the removal is the
#: point: `examples/_shifted_receptor.pdbqt` used to be answered `"shipped"` on
#: the grounds that `workbench_interaction_check.py` reads it. That gate is also
#: the only thing that ever *wrote* it -- a receptor derived from
#: `rec_prep.pdbqt` with every x shifted by -40, regenerated byte-identically on
#: every run -- and it writes it into a scratch directory now, which
#: `/target/` already covers. So there is no file literal left to answer, and
#: inventing one to keep the key alive would be a line in `.gitignore` naming a
#: path that can no longer exist.
#:
#: Measured over the 18 lines: 10 literal directories, 2 literal files, 6
#: wildcard patterns, 0 negations -- counted by
#: `[l for l in live if ...]`, not by hand, because the three numbers have to
#: sum to the line count for the table to be describing a partition. (They did
#: not, until `examples/_shifted_receptor.pdbqt` came out: the previous text
#: said "18 lines" against a 19-line file whose parts summed to 19, and the
#: arithmetic was never the kind anybody checked because nothing consumed it.)
#: `examples/_shifted_receptor.pdbqt` is no longer listed at all, for the reason
#: the file-literal check below states. Nothing here needs a pattern language,
#: which is why there is no pattern language.
GITIGNORE_ANSWERS = {
    "target": ("declined", "the reader, and also the root test"),
    "dist": ("declined", "the reader, and also the root test"),
    "dist-gpu": ("declined", "the reader, and also the root test"),
    "__pycache__": ("declined", "the reader, and also GENERATED_DIRS by name"),
    ".pytest_cache": ("declined", "the reader, and also the CACHEDIR.TAG test"),
    ".venv": ("declined", "the reader; nothing else names it"),
    "venv": ("declined", "the reader; nothing else names it"),
    "examples/maps": ("declined", "the reader"),
    "examples/multi": ("declined", "the reader"),
    "examples/split_dir": ("declined", "the reader"),
    "dockpy.dll": ("shipped", "nothing to ship: absent, and `.dll` is an "
                             "undeclared suffix"),
    "$null": ("shipped", "declined instead, by ROOT_SOURCE_NAMES -- recorded "
                         "here as shipped-shaped because the entry is a file "
                         "literal the reader does not act on"),
    "**/*.rs.bk": ("unimplemented", "a wildcard; no `.rs.bk` exists in either "
                                    "tree and `.bk` is an undeclared suffix"),
    "*.py[cod]": ("unimplemented", "a character class; `.pyc` and `.pyo` are "
                                   "undeclared, and `.pyd` is the divergence "
                                   "recorded below"),
    "*.egg-info/": ("unimplemented", "a wildcard before a directory slash; no "
                                     "`.egg-info` exists in either tree"),
    "*_dockpy.pyd": ("shipped", "the divergence: `.pyd` IS a declared source "
                                "suffix, so `opendocking/_dockpy.pyd` ships "
                                "and the two declarations name the same file "
                                "for different reasons"),
    "*_dockpy.so": ("shipped", "nothing to ship; `.so` is an undeclared suffix"),
    "*_dockpy.dylib": ("shipped", "nothing to ship; `.dylib` is undeclared"),
}


def check_the_gitignore_answers() -> None:
    """Two claims, two checks: every line is answered, and every answer is true."""
    section("the repository's own .gitignore, and the answer for every line in it")

    raw = RULE._gitignore_path()
    try:
        lines = [ln.strip() for ln in
                 raw.read_bytes().decode("utf-8", errors="replace").splitlines()]
    except OSError as exc:
        bad("every line in the shipped .gitignore has an answer in this file",
            f"{raw} could not be read: {exc}. The rule reads this file for its "
            f"directory literals, so a rule that cannot read it is a rule "
            f"behaving as though the project declared nothing")
        return
    live = [ln for ln in lines if ln and not ln.startswith("#")]
    bare = {ln.strip("/").lstrip("/") for ln in live
            if ln.endswith("/") and not any(
                c in ln for c in RULE.GITIGNORE_UNIMPLEMENTED)}
    unanswered = [ln for ln in live
                  if ln not in GITIGNORE_ANSWERS
                  and ln.strip("/").lstrip("/") not in GITIGNORE_ANSWERS]
    if unanswered:
        bad("every line in the shipped .gitignore has an answer in this file",
            f"{len(live)} line(s) in {raw}, {len(GITIGNORE_ANSWERS)} answered "
            f"here; unanswered: {unanswered}. A new tool that writes into a "
            f"source root and adds one line cannot be published by accident "
            f"while this is red, and cannot be ignored by accident either")
    else:
        ok("every line in the shipped .gitignore has an answer in this file",
           f"{len(live)} line(s) in {raw}, all answered here: "
           f"{sum(1 for v in GITIGNORE_ANSWERS.values() if v[0] == 'declined')} "
           f"declined, {sum(1 for v in GITIGNORE_ANSWERS.values() if v[0] == 'shipped')} "
           f"shipped, {sum(1 for v in GITIGNORE_ANSWERS.values() if v[0] == 'unimplemented')} "
           f"unimplemented by the reader and stated as such")

    # -- the answers are true, proved by planting them ----------------------
    # No early return anywhere in this function. The first version had one after
    # the reader check, and it meant that a failure there suppressed the check
    # below: the run reported 47 instead of 48. A total that moves when a check
    # fails is a total describing the failure, not the file.
    reader = {entry for entry, _anchored in RULE.declared_generated_dirs()}
    if bare - reader:
        bad("the reader declines every directory the .gitignore declares",
            f"{sorted(bare - reader)} are answered 'declined' above but the "
            f"reader does not report them. Read {sorted(reader)}")
    else:
        ok("the reader declines every directory the .gitignore declares",
           f"all {len(bare)} directory literal(s) in {raw} are in the reader: "
           f"{sorted(reader)}. This asserts the reader saw them, not that each "
           f"one declines a file: the three directories that actually billed the "
           f"rule (`examples/maps`, `examples/multi`, `examples/split_dir`, 50 "
           f"files between them) are measured against the real tree by the "
           f"missing-from-published class below, which reports any of them that "
           f"is still selected")

    # Anchoring: `/target/` is root-relative in git, and this reader keeps that.
    # Both branches of this check used to call `ok`, which makes it a check that
    # is structurally incapable of going red -- the shape this repository keeps
    # refusing, found by asking the question rather than by a mutation. It now
    # asks for the verdict the anchoring implies, which M-K turns red.
    synthetic = Path("scripts/target/zz_real.py")
    if RULE.is_source(synthetic, ROOT / "scripts"):
        ok("an anchored entry does not match the same name deeper in the tree",
           f"{raw} declares `/target/`, which git reads as root-relative, and "
           f"the reader keeps that: `{synthetic}` is selected. No such "
           f"directory exists on disk, so this asks the rule rather than the "
           f"filesystem -- which is the point, because a check that reads the "
           f"directory cannot fire on a machine where the directory is absent, "
           f"and that is mutual absence: nothing present and nothing reported")
    else:
        bad("an anchored entry does not match the same name deeper in the tree",
            f"`{synthetic}` is not selected, so a directory called `target` "
            f"anywhere in a source root is being declined on the strength of a "
            f"leading `/` that git would not apply there. This is the "
            f"over-broad direction, and it is the one that deletes real files")

    # What decides a file literal, stated as the check it deserves to be.
    #
    # This branch used to assert that `examples/_shifted_receptor.pdbqt` ships
    # *because* `workbench_interaction_check.py` reads it, and failed if it did
    # not, on the grounds that "a reader that took file literals at face value
    # would have deleted a file the suite needs". The answer was right; the
    # reason was inverted, and an inverted reason is worse than a wrong answer
    # because it survives every change to the rule. `release_tree_rule.py:349-350`
    # reads `.gitignore` for **directory** literals and skips file literals on
    # purpose, so `is_source` on a file is a function of the directories above
    # it and of nothing else -- it never opens a file, so it cannot know who
    # reads one. And the file in question was written by that same gate, minutes
    # earlier in the same process, from `rec_prep.pdbqt` with every x shifted by
    # -40; it regenerates byte-identically, so it is a *derived* fixture and
    # "a gate reads it" and "the repository ships it" are not the same claim.
    # The old text made the second claim out of the first, which is what
    # convinced a reviewer the file was a legitimate shipped input.
    #
    # So the distinction is worth a named check rather than prose, for a reason
    # specific to this failure: the rule gives the same answer for a file a gate
    # reads and a file nothing reads, and that equality is the claim. Asking it
    # three ways makes the claim falsifiable -- if the reader ever starts
    # honouring file literals, the first two answers diverge from the third.
    #
    # All three probes are synthetic, so the check fires on a machine where none
    # of the files exist. A check that reads the filesystem here would be
    # satisfiable by absence, which is the mutual-absence trap the anchoring
    # check above already names.
    read_by_a_gate = Path("examples/zz_probe_a_gate_reads_this.pdbqt")
    read_by_nobody = Path("examples/zz_probe_nothing_reads_this.pdbqt")
    same_name_in_declared = Path("examples/maps/zz_probe_a_gate_reads_this.pdbqt")
    gate_side = RULE.is_source(read_by_a_gate, ROOT / "examples")
    nobody_side = RULE.is_source(read_by_nobody, ROOT / "examples")
    declared_side = RULE.is_source(same_name_in_declared, ROOT / "examples" / "maps")
    if gate_side is True and nobody_side is True and declared_side is False:
        ok("a file literal is decided by the directories above it, not by who reads it",
           f"the reader takes no notice of readers: `{read_by_a_gate.name}` -- a "
           f"name a gate is imagined to read -- and `{read_by_nobody.name}` -- a "
           f"name nothing reads -- get the same answer ({gate_side}), because "
           f"`declared_generated_dirs()` returns directory literals only. The "
           f"third probe is the control that says the answer is not a constant: "
           f"the same name one level down, inside the declared "
           f"`examples/maps/`, is {declared_side}. 'a gate reads this' and 'the "
           f"repository ships this' are different claims and only the second one "
           f"is an input to the rule")
    else:
        bad("a file literal is decided by the directories above it, not by who reads it",
            f"probe answers: read-by-a-gate={gate_side}, read-by-nobody="
            f"{nobody_side}, same-name-inside-a-declared-dir={declared_side}. "
            f"The first two must agree (the reader has no opinion about who "
            f"reads a file) and the third must differ from them (the directory "
            f"is what decides). Either the reader has started honouring file "
            f"literals from {raw}, in which case every file literal in it is now "
            f"a deletion instruction and the divergence it was scoped to avoid "
            f"is real, or it has stopped matching declared directories, which "
            f"is the direction that deletes real files")


def check_the_tree_carries_no_declared_output(trees: Trees) -> None:
    """The inverse of the reader, and the hole it leaves.

    The reader stops a declared-non-source directory being *published*. Nothing
    stopped it being *present*, so regenerating `examples/maps/` would have been
    harmless to the release and invisible to every gate while 44.6 MB sat in the
    source tree. This is the check for that, and it asks the question in the only
    way that cannot be over-broad:

        for every file inside a directory the repository declares non-source,
        would the rule publish it if the declaration were not there?

    If the answer is no, the declaration is carrying nothing and the directory is
    harmless -- which is the case for every one of them today, measured: 21 721
    files under `target/`, 224 under `dist/`, 5 under `.pytest_cache/` and every
    `.pyc` in `__pycache__` are all files no suffix or root in `SOURCE_SUFFIXES`
    / `SOURCE_ROOTS` claims. If the answer is yes, the declaration is the only
    thing standing between that file and every clone, and it is named here.

    **Why this cannot fire on the wrong thing.** A check that said "these
    directories must not exist" would be red on `target/` and `__pycache__` on
    any machine that has ever run a build or an import -- which is every machine,
    and would be a check whose red means "Python works". This one asks whether
    anything is being *hidden*, so a cache full of bytecode is silent and a
    directory of `.pdbqt` fixtures is not. That asymmetry is the whole design:
    the over-broad direction is the one that costs a machine its build, and the
    under-broad one costs a person 44.6 MB.
    """
    section("no declared-non-source directory is hiding a file the rule would ship")
    # -- the switch's own both-directions proof, first ----------------------
    # Without this the check is satisfied by `honour_declaration` being ignored:
    # ask "would it ship without the declaration" of a predicate that always
    # applies the declaration and the answer is always no, so the check reports
    # nothing hidden and goes green with a fixture sitting in examples/maps/.
    # That is the green-for-the-wrong-reason shape, found by asking what M-O
    # would do rather than by running M-O.
    probe = Path("examples/maps/zz_switch_probe.pdbqt")
    with_decl = RULE.is_source(probe, ROOT / "examples" / "maps")
    without_decl = RULE.is_source(probe, ROOT / "examples" / "maps", False)
    if with_decl is False and without_decl is True:
        ok("the declaration switch is the only thing that declines a declared directory",
           f"asked about `{probe}` both ways: the rule declines it with the "
           f"declaration and would publish it without one. This is what makes "
           f"the check below a measurement rather than a restatement -- if the "
           f"switch stopped working, both answers would be no and this check "
           f"would go green on a fixture it should have named")
    else:
        bad("the declaration switch is the only thing that declines a declared directory",
            f"asked about `{probe}` both ways: with the declaration "
            f"{with_decl}, without it {without_decl}. Expected False then True. "
            f"If the switch does not change the answer, every absence check "
            f"below is vacuous -- it would report nothing hidden because it "
            f"cannot ask the question")

    hidden: list[tuple[str, str]] = []
    declared = RULE.declared_generated_dirs()
    seen_dirs: set[str] = set()
    for label, root in (("development", ROOT),
                        ("published", trees.published)):
        if root is None or not Path(root).is_dir():
            continue
        for entry, anchored in sorted(declared):
            if anchored:
                candidates = [Path(root) / entry]
            else:
                candidates = sorted(p for p in Path(root).rglob(entry)
                                    if p.is_dir())
            for d in candidates:
                seen_dirs.add(f"{label}:{d.relative_to(root).as_posix()}")
                for f in sorted(d.rglob("*")):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(root)
                    if RULE.is_source(rel, f.parent) or not RULE.is_source(
                            rel, f.parent, False):
                        continue
                    hidden.append((rel.as_posix(),
                                   RULE.recognition_reason(rel, f.parent) or ""))
    if hidden:
        bad("no declared-non-source directory is hiding a file the rule would ship",
            f"{len(hidden)} file(s) that the repository's .gitignore is the only "
            f"thing stopping, in {len(seen_dirs)} declared director(y/ies): "
            + "; ".join(f"{r} -- {why[:80]}" for r, why in hidden[:6])
            + (" ..." if len(hidden) > 6 else "")
            + ". The release is protected while these sit here, which is exactly "
              "why they are invisible: the reader declines them, so no other "
              "class reports them. If a directory was regenerated by hand, "
              "delete it; nothing in the suite writes here.")
    else:
        ok("no declared-non-source directory is hiding a file the rule would ship",
           f"{len(seen_dirs)} declared director(y/ies) present in "
           f"{len({s.split(':')[0] for s in seen_dirs})} tree(s) "
           f"({', '.join(sorted({s.split(':', 1)[1] for s in seen_dirs}))}) and "
           f"every file in them is one the rule would decline on its own "
           f"evidence -- undeclared suffix, undeclared root, or a derived "
           f"generated-directory test. Asked with the declaration switched off, "
           f"so this is a measurement of the reader's load rather than a "
           f"restatement of its answer")


#: The sixth question the payload table should answer, kept as a limit and not
#: as a check. **"Which files in the payload cannot be reviewed by reading?"**
#:
#: Answer today: `dock-py/python/opendocking/_dockpy.pyd`, and nothing else. It
#: is a compiled extension for one interpreter build and one platform; a reader
#: of the release tree cannot tell a correct one from a corrupted or mutated one
#: by opening it, and no assertion in this repository inspects its bytes. It is
#: published on the strength of the build discipline of whoever last compiled it,
#: and the parity check's DIFFERS class is what tells a maintainer that those
#: bytes changed -- which is the right division: the rule classifies what kind of
#: file this is, and a classification has no opinion about whether a binary is
#: current.
#:
#: It is written here rather than asserted because a check for it would have to
#: be a check that the bytes are correct, and the only oracle for that is a
#: rebuild -- which is the engine owner's build, not a fact inside either tree.
UNREVIEWABLE_PAYLOAD = ("dock-py/python/opendocking/_dockpy.pyd",)

#: **The same question asked of the other artefact, which has a different answer
#: and is the one a reader of the repository actually has.** *Which files in the
#: published repository cannot be reviewed by reading?* -- **none**, and that is
#: worth stating rather than leaving implied, because the two answers used to be
#: the same file.
#:
#: The gap between them is a real structural fact about this project and the
#: first time the two artefacts have been distinguishable: `.gitignore` carries
#: `*_dockpy.pyd`, so git never tracked the compiled extension. The **sync
#: payload** is a working copy and therefore does contain it; the **published
#: repository** cannot. All three CI jobs build and install the wheel themselves,
#: and the gate scripts import the installed package, so nothing depends on a
#: committed binary.
#:
#: This is why `UNREVIEWABLE_PAYLOAD` above has exactly one entry and this table
#: has none, and why the check below asserts both rather than one: a payload
#: question answered for the repository, or the reverse, is a green that means
#: nothing -- the first version of this file made exactly that conflation.
UNREVIEWABLE_REPOSITORY: tuple = ()


def check_the_unreviewable_payload() -> None:
    section("which payload files cannot be reviewed by reading")
    listed = set(RULE.iter_source_files(ROOT))
    drift = [p for p in UNREVIEWABLE_PAYLOAD if Path(p) not in listed]
    if drift:
        bad("the payload and repository unreviewable lists are current",
            f"{drift} are listed here as unreviewable but the rule no longer "
            f"publishes them. Either they became reviewable -- say so and change "
            f"this list -- or the suffix table moved and this is now wrong")
    elif [p for p in UNREVIEWABLE_REPOSITORY if p in listed]:
        bad("the payload and repository unreviewable lists are current",
            f"{list(UNREVIEWABLE_REPOSITORY)} are listed as carried by the "
            f"published repository. The rule publishes them, but that is the "
            f"payload: `.gitignore` carries `*_dockpy.pyd`, so git never tracked "
            f"it and no published tree can contain one. A clone that appeared to "
            f"need this file would be a clone nobody has")
    else:
        size = (ROOT / UNREVIEWABLE_PAYLOAD[0]).stat().st_size
        ok("the list of payload files that cannot be reviewed by reading is current",
           f"{len(UNREVIEWABLE_PAYLOAD)} file(s), "
           f"{UNREVIEWABLE_PAYLOAD[0]} ({size:,} B), and the rule does publish "
           f"it. Recorded as a limit, not asserted: no check here can say the "
           f"bytes are correct, because the only oracle is a rebuild. This is a "
           f"limit statement and it is deliberately not a verdict -- a check "
           f"that can only ever pass would be a pin with a comment on it")


# --------------------------------------------------------------------------
# Section D -- the third copy
# --------------------------------------------------------------------------
def _site_packages() -> list[Path]:
    found: set[Path] = set()
    getter = getattr(site, "getsitepackages", None)
    if callable(getter):
        for entry in getter():
            found.add(Path(entry))
    for key in ("purelib", "platlib"):
        entry = sysconfig.get_paths().get(key)
        if entry:
            found.add(Path(entry))
    return sorted(found)


def check_the_third_copy(trees: Trees) -> None:
    section("the third copy of the package: out of scope, and says so")
    inside = []
    for sp in _site_packages():
        for name, root in (("published", trees.published),
                           ("development", trees.development)):
            if root is not None and sp.is_relative_to(root):
                inside.append(f"{sp} is inside the {name} tree ({root})")
    if inside:
        bad("no site-packages lives inside either published tree",
            "; ".join(inside) + ". An interpreter whose site-packages is inside "
            "a release tree is a reachable state (a virtualenv created inside "
            "the checkout), and it would put a compiled .pyd for one interpreter "
            "into the artefact everybody else clones")
    else:
        ok("no site-packages lives inside either published tree",
           "checked " + ", ".join(str(p) for p in _site_packages()) +
           f" against {trees.published}" +
           (f" and {trees.development}" if trees.development else "") +
           ". This is what makes the exclusion below a boundary rather than a "
           "promise")

    decision = getattr(RULE, "SITE_PACKAGES_DECISION", "")
    if len(decision) >= MIN_DECISION_CHARS:
        ok("the out-of-scope decision is recorded in the shipped rule",
           f"{len(decision)} characters in {RULE.__name__}.py, printed in full "
           f"above the results; a decision that lives only in a comment is a "
           f"decision the next reader reverses")
    else:
        bad("the out-of-scope decision is recorded in the shipped rule",
            f"{RULE.__name__}.SITE_PACKAGES_DECISION is "
            f"{len(decision)} characters, below the {MIN_DECISION_CHARS} this "
            f"check requires. Out of scope and nobody looks at it have to be "
            f"written down differently")


def report_the_third_copy(trees: Trees) -> None:
    """The measured state of the copy that is not part of the contract.

    Reported, not asserted: it is a local install artefact, it is outside both
    published trees, and a developer's answer to its drift is to reinstall. But
    it drifts silently, and a silent drift that has been *declared* is a much
    easier thing to act on than one nobody wrote down -- so the numbers are here
    on every run, with the same two-path, two-digest form as everything else.
    """
    print()
    print("-- the third copy, reported and not asserted")
    print(f"    {getattr(RULE, 'SITE_PACKAGES_DECISION', '(no decision recorded)')}")
    for sp in _site_packages():
        installed = sp / "opendocking"
        if not installed.is_dir():
            continue
        files = sum(1 for _ in installed.rglob("*") if _.is_file())
        size = sum(_.stat().st_size for _ in installed.rglob("*") if _.is_file())
        print(f"    installed copy: {installed}")
        print(f"      {files} file(s), {size / 1024:.0f} KiB -- an install "
              f"artefact, not part of either published tree")
        for rel in ("workbench/launcher.py", "core.py"):
            here, there = ROOT / "dock-py" / "python" / "opendocking" / rel, installed / rel
            if here.is_file() and there.is_file():
                a, b = RULE.file_digest(here), RULE.file_digest(there)
                state = "identical to" if a == b else "DRIFTED from"
                print(f"      {state} {here}\n"
                      f"        sha256 {a[:16]}...  vs  {there}\n"
                      f"        sha256 {b[:16]}...")
    if not any((sp / "opendocking").is_dir() for sp in _site_packages()):
        print(f"    no site-packages on this interpreter holds a copy of the "
              f"package; the out-of-scope decision does not depend on it "
              f"existing")


# --------------------------------------------------------------------------
def _tally() -> tuple[int, int, int]:
    return (sum(1 for t, _n, _d in RESULTS if t == "PASS"),
            sum(1 for t, _n, _d in RESULTS if t == "FAIL"),
            sum(1 for t, _n, _d in RESULTS if t == "NOT-ASSERTED"))


def check_the_totals() -> None:
    """The last result of the run, and the one the summary is built from.

    Placed after everything else on purpose: a summary printed before its own
    last check would state a total one short of the number the file declares,
    and a reader comparing the two would be comparing a lie with a number.
    """
    npass, nfail, nnot = _tally()
    if npass + nfail + nnot == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok("the summary accounts for every result, and the total is the "
           "declared one",
           f"{npass} + {nfail} + {nnot} = {len(RESULTS)}, and this check is the "
           f"{len(RESULTS) + 1}th, so the run reaches the declared "
           f"{EXPECTED_CHECKS} -- the same on a maintainer's machine with both "
           f"trees and on a runner with one")
    else:
        bad("the summary accounts for every result, and the total is the "
            "declared one",
            f"recorded {len(RESULTS)} results ({npass}+{nfail}+{nnot}) before "
            f"this check, which must be {EXPECTED_CHECKS - 1} for the run to "
            f"reach the declared {EXPECTED_CHECKS}. A total that changes with "
            f"the machine is how a run gets to be green by checking less")


def main() -> int:
    trees = resolve_trees()
    loaded = Path(RULE.__file__).resolve()
    print("Do the two trees agree, and is the rule auditable from the release?")
    print()
    print("-- provenance: every number below came from these")
    print(f"    rule module  : {loaded}")
    print(f"    its digest   : sha256 {RULE.file_digest(loaded)}")
    print(f"    decision set : sha256 {RULE.rule_fingerprint()}")
    print(f"    SOURCE_ROOTS : {sorted(RULE.SOURCE_ROOTS)}")
    print(f"    ROOT_SOURCE_N: {sorted(RULE.ROOT_SOURCE_NAMES)}")
    print(f"    SOURCE_SUFFIX: {sorted(RULE.SOURCE_SUFFIXES)}")
    print(f"    canonical pkg: {RULE.CANONICAL_PACKAGE_PARENT.as_posix()}")
    print(trees.describe())

    tmp = Path(tempfile.mkdtemp(prefix="odock-release-parity-"))
    try:
        check_the_rule_ships(trees)
        check_the_trees_agree(trees)
        check_the_comparison_can_go_red(trees, tmp)
        check_recognition(tmp)
        check_the_gitignore_answers()
        check_the_tree_carries_no_declared_output(trees)
        check_the_unreviewable_payload()
        check_the_third_copy(trees)
        report_the_third_copy(trees)
        check_the_totals()

        npass, nfail, nnot = _tally()
        print()
        print(f"--- summary: {npass} passed, {nfail} failed, {nnot} not asserted, "
              f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")
        for name, why in UNASSERTED:
            print(f"    not asserted here: {name} -- {why[:100]}")
        print("    the three numbers never sum to a verdict: a pass count on its "
              "own cannot say whether the trees were compared")

        if nfail:
            print(f"RESULT: FAIL -- {nfail} of {EXPECTED_CHECKS} checks failed")
            return EXIT_FAILED
        if npass == 0:
            print("RESULT: DID NOT FINISH -- this run asserted nothing")
            return EXIT_INCOMPLETE
        print(f"RESULT: OK -- {npass} of {EXPECTED_CHECKS} checks passed, "
              f"{nnot} could not be asserted on this machine")
        return EXIT_OK
    finally:
        if os.environ.get("OD_KEEP_SHADOW"):
            print(f"\nshadow trees kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("\nRESULT: DID NOT FINISH -- see the traceback above")
        raise SystemExit(EXIT_INCOMPLETE)
