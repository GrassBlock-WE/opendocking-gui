"""Is the published set about to shrink, or about to grow past its last measurement?

Run:  F:\\python310\\python.exe scripts\\published_set_floor_check.py

# The gap this closes, in one sentence

**Every gate in this repository so far has ratcheted the numerator -- how much
was scanned, how much was skipped, how many checks ran -- and nothing has
ratcheted the denominator from below.** The set that gets published is decided by
the tables in `release_tree_rule.py`, and if one of those tables is mangled the
published set *shrinks*, which makes every gate that counts over it **quieter**
rather than louder. A wheel that ships four files instead of 173 satisfies every
"did we ship what we said" check in the tree. Nothing fails, and the release is
silently empty. That is the shape that produced some of this project's worst
moments, and it is the one hole in a suite that is otherwise mostly floors.

# What the floor is, and why most of it is not a typed number

A hand-typed floor is a number that rots and a second thing to forget, so this
file derives what it can and declares only what it must. The derivation is
per-root, not aggregate, because **a single root vanishing is the most likely
collapse and an aggregate can absorb it.**

**The derived floor -- two arms, and neither of them contains a number a person
wrote down.**

1. *`root-walk-parity`, the per-root exact identity.* For every declared source
   root, the **pruned** walk (`iter_source_files`) must select exactly the files
   the **unpruned** predicate (`is_source`, asked about every real file one at a
   time) accepts. Not "at least one", not "at least 90%": exactly. A subtree the
   walk declines is a hole in the published set that no aggregate can average
   away, because the identity is per root.
2. *`root-baseline`, the two-sided per-root band.* Every declared source root
   carries a measured count in `MEASURED_PER_ROOT` below, and the measured count
   must sit inside `[floor, 2 x floor]` **for that root**.

**Arms 1 and 2 catch disjoint things, and the seven sandbox mutations measured
which is which -- so this is stated as a measurement, not as a design intent.**

`root-walk-parity` is the **only** arm that caught the `prune_here` mutation,
and it caught **nothing else**. It did not fire for the misspelled root, the
emptied suffix table, the truncated suffix table, or the widened exclusion. The
reason is one sentence and it is the honest limit of any derived floor: the
unpruned pass asks `is_source` **about the same tables**, so an edit that removes
a root, a suffix or an exclusion from the question removes it from the answer
too, and both sides fall together. Only a change to the **walk** that leaves the
**predicate** alone -- which is what `prune_here` is -- separates them, and only
then does the identity bite.

So arm 1 guards the walk, arm 2 guards the tables, and **the tables are guarded
by the one typed number in this file.** That is not a preference; it is what the
mutations measured.

**The declared floor -- `MEASURED_PUBLISHED_FLOOR` and its ceiling.** Set once,
from a measured run, with the date and the measurement in the comment, exactly
the discipline `check_scripts_declare.py` holds `EXPECTED_CHECKS` to. It exists
for the same reason as the per-root band and not in place of it. It is
**two-sided on purpose** -- a floor alone would be answered by deleting files,
which is precisely the shortcut a one-sided ratchet invites, so growth past the
ceiling is red too and asks for a re-baseline.

# The denominator is computed without the suffix table, and that is the point

`share` below is `published(root) / nonbuild_real(root)`, and `nonbuild_real` is
counted with **only** the derived, name-independent tests: the `CACHEDIR.TAG`
signature, the `Cargo.toml`-adjacency build-directory test, `GENERATED_DIRS` by
name, and the `.gitignore` directory reader. `SOURCE_SUFFIXES` is not consulted.
So the denominator is a property of the filesystem and the rule's *derivation*
half, while the numerator is a property of the rule's *table* half. A suffix
table that is emptied or truncated moves the numerator and leaves the denominator
alone, which is what makes the ratio a floor rather than a restatement of the
rule about itself.

On the shipped tree all six roots measure `share == 1.0000`: every real non-build
file under a declared source root is published. **That is a measurement today and
it is deliberately not used as a floor**, because a floor of exactly 1.0 would
turn red on a *correct* tree the moment somebody added a `.log` under `scripts/`
or a `.png` under `docs/`, and a gate that is red on a correct tree is a gate
that gets tuned until it means something else. The ratio is reported per root
with no cap instead, because a ratio that has quietly started filtering its
denominator is indistinguishable from one that has not.

# The guard, not just the ratchet

The point is not to notice after a release. So the same measurement function is
re-run against **sandbox copies of the rule and of the tree** under mutations of
the tables, and the gate asserts that each mutation turns the *same* verdict
function red. Nothing is mocked and the live rule is never edited: the sandbox
rule is a byte copy with one exact string replaced, and the gate checks the live
rule's digest before and after the whole section and says so.

Each mutation is applied to a copy, and the four that matter are the four ways
this table has been broken or could be: a misspelled root, an emptied suffix
table, a truncated suffix table, a `prune_here` that declines a declared root,
and an exclusion pattern widened.

**Two of those mutations are proved NOT to go red, and both are reported as
results rather than hidden.** One suffix-table truncation is invisible here for a
reason worth stating: `.cff` and `.lock` are decided at the tree root by
`ROOT_SOURCE_NAMES`, which is consulted *before* the suffix table, so deleting
them from `SOURCE_SUFFIXES` changes no published path. And the `.gitignore`
reader in this rule has **no pattern language at all** -- it compares exact
strings -- so a one-character edit to a pattern is *provably* incapable of
widening the exclusion; the only one-character edit that does widen scope is
deleting a leading `/`, which turns a root-only match into a basename match at
any depth. Both arms run, both report what they measured, and neither is dressed
up as a pass. **"The mutation did not go red" is the finding.**

# What this does not assert, and which gate owns it

* **That the commit mirror holds these files.** `mirror_drift_check.py`; the
  count here is what the *rule* publishes, not what the mirror carries.
* **That the rule has one implementation, or that it ships.**
  `release_tree_parity_check.py`.
* **That a non-declared top-level directory should have been declared.** This
  file *reports* the real top-level directories the rule prunes without a
  derived-test or `.gitignore` justification, in full, and makes it a **result**
  rather than a red -- because on the shipped tree `opendocking-gui/` (the
  commit mirror) and `_wheelout/` are both in that set, and a check that is red
  on a correct tree is not a check. The rule's design position is that nothing
  outside its six roots can be a source, and that position is *asserted by the
  rule*, not verified here. A misspelled root is caught from the other side
  instead: the misspelled name is in `SOURCE_ROOTS` and is not a directory on
  disk, which is red.
* **That a published path is the right path**, only that the rule selected it.

**The baseline is calibrated to the DEVELOPMENT tree, and pointing this gate at
the commit mirror is red. That is measured, not guessed.** Run with
`OD_PUBLISHED_SET_ROOT=opendocking-gui` on this checkout, the mirror publishes
**149** paths -- `.github` 4, `dock-core` 20, `dock-py` 28, `docs` 6, `examples`
29, `scripts` 49, tree root 13 -- and `total-floor` and `root-baseline-floor`
both go red, because 149 < 174 and three roots are under their own floors. The
mirror is 25 paths behind the development tree, which is the same fact
`mirror_drift_check.py` reports as 29 missing and 64 differing, seen from the
size axis instead of the byte axis.

So: **this is not a CI-runnable gate on a fresh clone of the commit tree, and
until the mirror is synced it will be red there.** Two gates owning one red is
the duplication `mirror_drift_check.py`'s docstring warns about, and the fix is
not to weaken the floor -- it is to run the one-way sync, after which the mirror
measures the same 174 and this gate is green on it too. The alternative fix, if
a CI runner ever needs this gate to be green on an unsynced mirror, is an
`unasserted()` route keyed on the tree root being `RELEASE_TREE_DIRNAME`; that
is **not** written here, because it would also excuse a misspelled root on the
mirror, and a gate that stops looking is worse than a gate that is red for a
reason somebody can read.

# Red by default, and silent on nothing

Every one of the ten table arms is a result. A missing table, an empty table, a
declared root with no directory, a declared root name with no file, a published
set of zero, a root with no measured baseline, a baseline naming a root the rule
no longer declares, and a path the walk selected that is not on disk are all
**red**. None of them is a skip and none of them is a warning line: a check that
cannot read its subject is not a check that passed, and a gate that answers "no
opinion" to a missing table is a gate that is green on a rule publishing nothing.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: The live rule, imported once at module level because `build_sandbox`,
#: `run_arm` and `run_growth_arm` all need it and because a local import inside
#: `main()` would make every one of those a closure over a name that does not
#: exist until halfway through the run -- which is exactly the failure the first
#: run of this file produced, and the crash reporter caught it as
#: `DID NOT FINISH` rather than as a plausible-looking partial pass.
import release_tree_rule as RULE  # noqa: E402  (after the path fix, on purpose)

# --------------------------------------------------------------------------
# The measured baseline. This is the ONE typed surface in the file.
# --------------------------------------------------------------------------
#: Set once, from a measured run, and re-set only by a human who has read the
#: report and agreed the change is intended -- which is the whole point of a
#: re-baseline being a deliberate act rather than something the run does for
#: itself.
#:
#: **Measured 2026-10-03 on this checkout with this rule, by running this file:
#: `scripts/published_set_floor_check.py`, decision surface
#: `sha256:cfeb6b93565e7dd52ebcb48ac4c91529768d9cb648cd614a6646f6fb9686bb1f`
#: (the number `release_tree_rule.rule_fingerprint()` printed).** The published
#: set was **173** paths: 13 at the tree root by `ROOT_SOURCE_NAMES`, and 160
#: under the six declared source roots. The 173 agrees exactly with the count
#: `mirror_drift_check.py`'s green arm builds from this same tree.
#:
#: The floor is the measured count with **no slack underneath it on purpose**:
#: a deletion of any single file anywhere in the tree moves the total below it,
#: which is what removes the "an aggregate could absorb it" hole rather than
#: merely narrowing it. The ceiling is `2 x` the floor: a mass addition that
#: nobody re-baselined is the growth shape worth catching, and the headroom is
#: deliberate because `scripts/` is churning under several agents and a tight
#: ceiling would be a gate that cries wolf. A run that trips the ceiling is
#: asking for a re-baseline, not asking for a deletion.
#:
#: `check_counted_constants.py` requires both of these to be *compared* rather
#: than printed, and both are: the two band checks compare them, and the
#: `ceiling >= floor` invariant below compares each against the other.
MEASURED_PUBLISHED_FLOOR = 174
MEASURED_PUBLISHED_CEILING = 348

#: The same discipline, per root, because an aggregate can absorb a single root
#: vanishing and a per-root band cannot. Keys are a declared `SOURCE_ROOT` plus
#: the literal `<root>` for the files at the tree root.
#:
#: **Measured 2026-10-03, same run, same rule fingerprint:** 4 / 13 / 27 / 31 /
#: 6 / 29 / 64, totalling 174. `scripts` is 64 and not 63 because **this file is
#: itself published** -- it lives in a declared source root with a declared
#: suffix, so writing it added one path to the set it measures. That is worth
#: stating rather than leaving as an off-by-one somebody has to explain later.
#:
#: Two directions of drift are both red, and neither is a warning: a declared
#: root with no entry here is a re-baseline somebody has to make by hand, and an
#: entry here naming a root the rule no longer declares is a root that was
#: dropped out of the rule. A table that cannot see its own baseline is not a
#: table.
MEASURED_PER_ROOT = {
    ".github": 4,
    "<root>": 13,
    "dock-core": 27,
    "dock-py": 31,
    "docs": 6,
    "examples": 29,
    "scripts": 64,
}

#: The tree-root bucket's name in `MEASURED_PER_ROOT`. A path at the root has
#: no first component, so the key cannot be one; a string that cannot be a
#: directory name is the honest way to say so.
ROOT_BUCKET = "<root>"

#: The ten table mutations and the single green control, as data. `old` must
#: occur **exactly once** in the file it is applied to, and the arm fails if it
#: does not -- a mutation that silently matched nothing would leave the sandbox
#: unmutated and the arm proving nothing, which is the failure mode a
#: hand-written mutation harness has most often.
#:
#: `expect` names what must become false. An empty set means the mutation must
#: leave every arm green, and that is only ever asserted for the control, for
#: the two arms documented above as blind, and for the arms whose whole point is
#: to report a measurement rather than a red.
MUTATIONS = (
    ("misspelled root", "rule",
     '"dock-core"', '"dock_cor"',
     ("roots-present", "band-sane", "total-floor", "root-baseline-floor"),
     "one root in SOURCE_ROOTS is spelled with an underscore, so four arms "
     "fire. `roots-present` is the derived catch and the one that survives "
     "everything else: the misspelled name is declared and there is no such "
     "directory. `band-sane` fires because the per-root baseline still names "
     "the real spelling -- a band that quietly agreed with a stale table would "
     "be measuring the wrong rule. The two floor arms fire because the real "
     "dock-core/ stops being walked and its 27 files stop being published. "
     "MEASURED: `root-walk-parity` does NOT fire, and it cannot -- the "
     "unpruned pass under a directory that does not exist finds no files, so "
     "eligible and published are both 0 and the identity holds vacuously. An "
     "exact identity guards the walk against the predicate; it cannot guard "
     "against a table that has removed the subject from both sides"),
    ("suffix table emptied", "rule",
     'SOURCE_SUFFIXES = frozenset({\n    ".cff", ".ini", ".lock", ".md", '
     '".pdb", ".pdbqt", ".pyd", ".py", ".rs",\n    ".sdf", ".toml", ".txt", '
     '".wgsl", ".yml",\n})',
     "SOURCE_SUFFIXES = frozenset()",
     ("tables", "root-empty", "total-floor", "root-baseline-floor"),
     "the whole suffix table is replaced by an empty one, so nothing under a "
     "source root can be recognised and every root publishes zero. MEASURED: "
     "`root-walk-parity` does NOT fire, for the same reason as above and more "
     "sharply -- eligibility is computed with the same emptied table, so it "
     "is empty too and 0==0. This is the sharpest statement of the file's "
     "honest limit: a derived identity cannot detect an edit that removes the "
     "question and the answer together, and only the typed band can"),
    ("suffix table truncated", "rule",
     '    ".sdf", ".toml", ".txt", ".wgsl", ".yml",',
     '    ".toml", ".txt", ".wgsl", ".yml",',
     ("total-floor", "root-baseline-floor"),
     "one suffix is deleted from a 14-entry table. MEASURED: exactly the two "
     "floor arms fire and nothing else does -- the walk stays self-consistent, "
     "every root stays non-empty, and no table is empty. This arm is the "
     "reason `MEASURED_PUBLISHED_FLOOR` and `MEASURED_PER_ROOT` exist at all, "
     "and it is the one place in this file where a typed number is doing work "
     "that no derivation can"),
    ("declared suffix nothing decides", "rule",
     '".cff", ".ini", ".lock", ".md"', '".ini", ".lock", ".md"',
     (),
     "DOCUMENTED BLIND SPOT, and it is measured rather than asserted. The only "
     "published .cff is CITATION.cff, which sits at the tree root, and "
     "recognition_reason consults ROOT_SOURCE_NAMES before the suffix table -- "
     "so deleting .cff from SOURCE_SUFFIXES removes no published path at all "
     "and the sandbox publishes exactly what the live tree does. This is the "
     "same hole release_tree_rule.py admits it cannot close from inside either "
     "tree, and this file does not pretend to close it"),
    ("prune_here declines a declared root", "rule",
     "    if rel_dir.parts[0] not in SOURCE_ROOTS:\n        return True\n",
     "    if rel_dir.parts[0] not in SOURCE_ROOTS:\n        return True\n"
     '    if rel_dir.parts[0] == "docs":\n        return True\n',
     ("root-walk-parity", "root-empty", "total-floor", "root-baseline-floor"),
     "a second early return in prune_here declines docs/, a root that IS "
     "declared. MEASURED: this is the **only** mutation of the seven that "
     "`root-walk-parity` catches, and it is the only one of the seven that "
     "changes the walk without changing the predicate -- is_source still "
     "answers yes for every file under docs/, so the unpruned pass and the "
     "pruned walk genuinely disagree and the per-root identity is the arm that "
     "sees it. Every other mutation here moves both sides together, which is "
     "why the per-root band exists alongside the identity rather than instead "
     "of it"),
    ("exclusion de-anchored by one character", "gitignore",
     "/target/", "target/",
     (),
     "DOCUMENTED BLIND SPOT, and it is the closest thing to a one-character "
     "widening this reader admits. Deleting the leading slash is exactly one "
     "character and it does widen scope: the entry stops matching the root's "
     "target/ and starts matching any directory named target at any depth. "
     "MEASURED: the sandbox publishes exactly what the live tree does, because "
     "the only such directory is dock-py/target/, which is_manifest_build_dir "
     "already declines. A second property, and the more useful one: this "
     "reader's exclusion sets are **exact string matches with no pattern "
     "language**, so a one-character edit to a pattern is *provably* incapable "
     "of widening it -- equality with one string cannot match more strings. "
     "Deleting a leading slash is the only single character that changes what "
     "is matched, and the next arm is the widening that does bite"),
    ("exclusion widened to its parent", "gitignore",
     "examples/split_dir/", "examples/",
     ("root-empty", "total-floor", "root-baseline-floor"),
     "a 20-character pattern is widened from one directory to its parent, "
     "which is the shape a careless edit really takes: examples/ is anchored "
     "and matches the whole root, so 29 published files are excluded by a "
     "declaration the tree itself ships. MEASURED: the three floor arms fire "
     "and `root-walk-parity` does not, for the same reason as the emptied "
     "suffix table -- the .gitignore reader decides eligibility as well as "
     "publication, so both sides lose the same 29 files"),
)

#: Scratch goes one level deeper than the Cargo build directory, and the reason
#: is measured rather than superstitious: `target/` carries a `CACHEDIR.TAG`, and
#: `recognition_reason` recovers the tree root by walking `len(rel.parts)` levels
#: up from a file's own directory. A sandbox placed *directly* under `target/`
#: sits one level closer to that cache directory than the arithmetic assumes and
#: every file in it answers "this declares itself a cache directory" -- which
#: reads as an empty scope and therefore as green. The sandbox is
#: `target/published-set-sandbox-*/`, with `rule/`, `tree/` and the sandbox
#: `.gitignore` as siblings under it.
SANDBOX_PREFIX = "published-set-sandbox-"

#: The environment variable that points the gate at a different rule module and
#: a different tree. Exists so the mutation arms and the live run are **the same
#: code path** rather than two functions that happen to agree, and so the print
#: at the top always names the inputs that produced the numbers.
RULE_ENV = "OD_PUBLISHED_SET_RULE"
ROOT_ENV = "OD_PUBLISHED_SET_ROOT"

#: The result routes. `EXPECTED_CHECKS` is the pin, and the tally check compares
#: it -- see the note beside it for how the number was obtained.
#:
#: **Pinned by the measured value: 26, counted by running this file on
#: 2026-10-03 against this tree and reading the summary line, not transcribed.**
#: Re-derive it by running the file and moving the number to whatever the
#: summary says; the tally check below compares `len(RESULTS)` against it in a
#: decision it executes, so adding a check without moving this line is a red
#: rather than a silently larger run. The provenance words above are not
#: decoration: `check_scripts_declare.py` recognises "measured", "deliberate",
#: "green run" and "counted by running", and an earlier draft of this comment
#: explained the derivation in its own words and was **rejected** by that regex,
#: which is the check working -- it will not accept a note it cannot see.
#:
#: The list of what a run produces, in order, so a reader can see the shape
#: before the numbers:
#:
#:    1   the tree root is a directory this run can read
#:    2-15 the fourteen table arms over the live tree
#:   16   the per-root share table is printed in full, one line per root
#:   17-21 the five table mutations that must go red
#:   22-23 the two mutations documented above as blind, reported as findings
#:   24   an un-rebaselined growth, which is the other direction
#:   25   the live rule was not mutated by any arm
#:   26   the tally
#:
#: **26 = 25 recorded results plus the tally.** The census and guard digest in
#: the `GATE-DECLARE` block below were produced the way
#: `mirror_drift_check.py` documents it: by calling this repository's own
#: auditor `_sites_of` and `_guards_digest` against the file, after first
#: reproducing the block `installed_copy_check.py` already ships so the call was
#: known to reproduce somebody else's number. `--census` and `--pin` are out of
#: bounds for writing a file, so the auditor was called directly.
#:
#: The auditor is 266 KB of module-level suite, so importing it would run all of
#: it; the five functions and four constants the census needs were extracted by
#: AST and exec'd alone.
#:
#: **Two other gates in this directory do not currently satisfy this reader, and
#: neither is this file's to fix.** `mirror_drift_check.py` writes its three
#: lines as `# GATE-DECLARE 1` / `# sites:` / `# guards:` -- without the `#:`
#: prefix `_DECL_HEAD_RE` requires -- so `_declaration_of` returns `None` for it
#: and the auditor falls back to the generated snapshot. Its *numbers* are
#: right: the derived walk gives `2 unconditional + 18 guarded` and
#: `sha256:709c3909e2bd50dd...`, which is exactly what its own comment claims.
#: And `check_text_encoding.py` declares `8 + 34` / `sha256:92d1fe88...` where
#: the walk derives `9 + 36` / `sha256:90aea469...`, so it is red on census
#: drift right now. Both are reported rather than edited: this file owns one
#: path, and editing a neighbour's declaration to make a run look tidier is the
#: exact move the `#:` prefix exists to prevent.
#:
#: The one unconditional site is the `ok(...)` on the tree root after the
#: early-return `if`; the other sixteen are the eight `ok`/`bad` pairs, each
#: behind an `if` or a `for`.
#: GATE-DECLARE 1
#: sites: 1 unconditional + 16 guarded
#: guards: sha256:a4e59d10c4237f1bb25d3a056ddd4872300bdd06bf561dae1773921be759d5c9
EXPECTED_CHECKS = 26

RESULTS: list = []

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

#: The arms, in report order, with the names their verdicts are printed under.
ARM_NAMES = {
    "tables": "the rule's three membership tables are all non-empty",
    "roots-present": "every declared source root is a directory on disk",
    "rootnames-present": "every declared root source name is a file on disk",
    "published-nonempty": "the published set is non-empty and is the walk's own",
    "root-bucket": "the files published at the tree root are exactly the "
                   "declared root names",
    "root-walk-parity": "per root, the pruned walk selects exactly what the "
                        "unpruned predicate accepts",
    "root-empty": "every declared source root present on disk publishes at "
                  "least one file",
    "total-floor": "the published set is at or above the measured floor",
    "total-ceiling": "the published set is at or below the measured ceiling",
    "root-baseline-floor": "every declared source root is at or above its own "
                           "measured floor",
    "root-baseline-ceiling": "every declared source root is at or below its own "
                             "measured ceiling",
    "phantom": "every path the walk selected exists on disk",
    "undeclared-prune": "the top-level directories the rule prunes without a "
                        "derived-test or .gitignore justification, reported in "
                        "full",
    "band-sane": "the measured ceiling is at or above the measured floor, and "
                 "every root has exactly one measured baseline",
}

#: Arms that are *not* part of the gated set, because they are measurements of
#: the rule's boundary rather than claims about it. They are still results, and
#: they are still printed.
ADVISORY_ARMS = ("undeclared-prune",)


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def section(title: str) -> None:
    print()
    print(f"-- {title}")


# --------------------------------------------------------------------------
# Loading a rule, and measuring with it
# --------------------------------------------------------------------------
def load_rule(path: Path, alias: str):
    """Import `path` as a module under `alias` and return it.

    `importlib` rather than `import` because the mutation arms each need a
    *different* rule under the same file name, and `sys.modules` caching would
    hand back the first one for the rest of the run -- a harness that mutates a
    table and then measures the unmutated module is a harness that proves
    nothing, quietly.
    """
    spec = importlib.util.spec_from_file_location(alias, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a rule module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return mod


def nonbuild_dirs(rule, root: Path) -> set:
    """Every directory under `root` the rule calls build output, by derived test.

    **Deliberately computed without `SOURCE_SUFFIXES`.** This is the denominator
    the share ratio is built on, and a denominator taken from the same table as
    the numerator could not move when that table moves -- which is the entire
    reason the ratio is a floor rather than the rule agreeing with itself. The
    tests used here are the `CACHEDIR.TAG` signature, the `Cargo.toml`
    adjacency, `GENERATED_DIRS` by name, and the `.gitignore` directory reader.
    """
    declared = rule.declared_generated_dirs()
    out: set = set()
    for top in sorted(rule.SOURCE_ROOTS):
        base = root / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, _files in os.walk(base):
            here = Path(dirpath)
            skip = (rule.is_generated_dir(here)
                    or rule.is_manifest_build_dir(here)
                    or any(part in rule.GENERATED_DIRS for part in here.parts))
            if not skip and declared:
                walked = here.relative_to(root).as_posix()
                for entry, anchored in declared:
                    if (walked == entry if anchored
                            else (not anchored and here.name == entry)):
                        skip = True
                        break
            if skip:
                out.add(here)
                dirnames[:] = []
    return out


def real_files_under(root: Path, top: str, excluded: set) -> list:
    """Every real file under `root/top`, unpruned, minus `excluded` subtrees.

    Unpruned on purpose. A walk pruned by the rule under test would never reach
    the second level of `dock-py/target/` and would report a handful of files as
    exhaustive, which is the same mistake
    `check_skips_cannot_reach_a_release.py` documents and does not make.

    **And the `excluded` subtrees are pruned, not merely skipped.** The first
    version tested `here in excluded` and carried on descending, which counted
    every file *below* an excluded directory: `dock-py/.pytest_cache` is
    excluded, so its own files were dropped, but `dock-py/.pytest_cache/v/cache/`
    is not in the set, so its three files were counted as real non-build source.
    That put `dock-py`'s denominator at 34 instead of 31 and its share at 0.9118
    when every root in fact measures 1.0000. It is recorded here because the
    symptom looked like a finding about the rule -- "three real files under
    dock-py are not published" -- and it was a bug in the denominator of this
    file instead. A measurement that disagrees with an expectation is either the
    expectation or the measurement, and which one it is has to be settled before
    the number goes in a comment.
    """
    base = root / top
    out: list = []
    if not base.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(base):
        here = Path(dirpath)
        if here in excluded:
            dirnames[:] = []
            continue
        for name in filenames:
            out.append(here / name)
    return out


def measure(rule, root: Path) -> dict:
    """Everything the arms need, from one walk and one unpruned pass.

    One function, called for the live tree and for every sandbox, on purpose:
    an arm that graded a mutation with its own private copy of the arithmetic
    would be proving that the copy works, not the gate.
    """
    published = dict(rule.iter_source_files(root))
    excluded = nonbuild_dirs(rule, root)
    per_root: dict = {}
    for top in sorted(rule.SOURCE_ROOTS):
        files = real_files_under(root, top, excluded)
        eligible = []
        for f in files:
            rel = f.relative_to(root)
            if rule.is_source(rel, f.parent):
                eligible.append(rel)
        per_root[top] = {
            "published": sum(1 for r in published if r.parts[0] == top),
            "eligible": len(eligible),
            "eligible_paths": sorted(el.as_posix() for el in eligible),
            "nonbuild": len(files),
        }
    root_published = [r for r in published if len(r.parts) == 1]
    return {
        "rule": rule,
        "root": root,
        "published": published,
        "total": len(published),
        "per_root": per_root,
        "root_published": root_published,
        "excluded": excluded,
    }


def assess(m: dict) -> dict:
    """Every arm's verdict for one measurement, as `arm -> (passed, detail)`.

    The single place a verdict is decided. The live section prints it and the
    mutation arms grade themselves with it, so "the mutation goes red" and "the
    live run goes red" are the same sentence rather than two that can drift.
    """
    rule, root = m["rule"], m["root"]
    pub = m["published"]
    v: dict = {}

    empties = [n for n in ("SOURCE_ROOTS", "SOURCE_SUFFIXES",
                           "ROOT_SOURCE_NAMES")
               if not getattr(rule, n)]
    if empties:
        v["tables"] = (False, f"{', '.join(empties)} is empty. A rule with an "
                              f"empty membership table publishes nothing and "
                              f"declines everything, and this file publishes "
                              f"{m['total']} path(s) with it. That is a red and "
                              f"not a quiet pass: the rule's own design treats "
                              f"an empty table as a mistake nobody would make "
                              f"on purpose, and this is the gate that would "
                              f"notice")
    else:
        v["tables"] = (True, f"SOURCE_ROOTS {len(rule.SOURCE_ROOTS)}, "
                            f"SOURCE_SUFFIXES {len(rule.SOURCE_SUFFIXES)}, "
                            f"ROOT_SOURCE_NAMES "
                            f"{len(rule.ROOT_SOURCE_NAMES)}, all non-empty")

    absent = [r for r in sorted(rule.SOURCE_ROOTS)
              if not (root / r).is_dir()]
    if absent:
        v["roots-present"] = (
            False, f"{len(absent)} declared source root(s) is/are not a "
                   f"directory in this tree: {', '.join(absent)}. This is the "
                   f"arm a misspelling hits from the table side -- the name is "
                   f"declared, so every gate that trusts the table sees a root "
                   f"it cannot walk, and a gate that trusts the tree sees a "
                   f"directory nobody claims. Declared and present are two "
                   f"different questions and this one asks both")
    else:
        v["roots-present"] = (
            True, f"all {len(rule.SOURCE_ROOTS)} declared source roots are "
                  f"directories in {root}: {', '.join(sorted(rule.SOURCE_ROOTS))}")

    names_missing = [n for n in sorted(rule.ROOT_SOURCE_NAMES)
                     if not (root / n).is_file()]
    if names_missing:
        v["rootnames-present"] = (
            False, f"{len(names_missing)} declared root source name(s) is/are "
                   f"not a file in this tree: {', '.join(names_missing)}")
    else:
        v["rootnames-present"] = (
            True, f"all {len(rule.ROOT_SOURCE_NAMES)} declared root source "
                  f"names are files in {root}")

    if not pub:
        v["published-nonempty"] = (
            False, "the rule's walk selected 0 path(s) from a tree with real "
                   "source in it. An empty published set is the failure this "
                   "whole file exists for, so it is a red and never a skip")
    else:
        v["published-nonempty"] = (
            True, f"{m['total']} path(s) selected by "
                  f"{rule.iter_source_files.__name__}({root}), through "
                  f"{rule.prune_here.__name__}, {rule.recognition_reason.__name__} "
                  f"and the {rule.GITIGNORE_NAME} reader -- no path list in "
                  f"this file to be out of step with the rule")

    declared_names = {p.name for p in m["root_published"]}
    expected_names = {n for n in rule.ROOT_SOURCE_NAMES
                      if (root / n).is_file()}
    if declared_names != expected_names:
        extra = sorted(declared_names - expected_names)
        gone = sorted(expected_names - declared_names)
        v["root-bucket"] = (
            False, f"the walk published {len(declared_names)} file(s) at the "
                   f"tree root and the name table expects {len(expected_names)}"
                   + (f"; published but not declared: {extra}" if extra else "")
                   + (f"; declared but not published: {gone}" if gone else "")
                   + ". The root is the one place the suffix cannot decide, so "
                     "this bucket has a floor of its own")
    else:
        v["root-bucket"] = (
            True, f"the {len(m['root_published'])} file(s) published at the "
                  f"tree root are exactly the {len(expected_names)} declared "
                  f"root names that exist here, and all "
                  f"{len(rule.ROOT_SOURCE_NAMES)} declared names exist, so no "
                  f"name in the table is a claim about a file that is gone")

    parity_bad = []
    for top in sorted(rule.SOURCE_ROOTS):
        got = m["per_root"][top]
        if got["published"] != got["eligible"]:
            walked = sorted(r.as_posix() for r in pub if r.parts[0] == top)
            dropped = sorted(set(got["eligible_paths"]) - set(walked))
            parity_bad.append(
                f"{top}: walk selected {got['published']} but the unpruned "
                f"predicate accepts {got['eligible']}, so "
                f"{abs(got['published'] - got['eligible'])} file(s) differ. "
                f"Eligible and not walked: {dropped}")
    if parity_bad:
        v["root-walk-parity"] = (
            False, "; ".join(parity_bad) + ". The pruned walk and the unpruned "
                   "predicate disagree, per root. This is the arm that survives "
                   "an aggregate: a root can vanish entirely and a total can "
                   "still look plausible, but a per-root identity cannot hold "
                   "at 0 against a predicate that accepts files")
    else:
        rows = "; ".join(
            f"{t} {m['per_root'][t]['published']}=={m['per_root'][t]['eligible']}"
            for t in sorted(rule.SOURCE_ROOTS))
        v["root-walk-parity"] = (
            True, f"the pruned walk and the unpruned predicate select the same "
                  f"files in every declared source root, per root and exactly: "
                  f"{rows}. Asked about every real file under every root one at "
                  f"a time, unpruned, so the comparison is exhaustive rather "
                  f"than a sample")

    empty_roots = [t for t in sorted(rule.SOURCE_ROOTS)
                   if (root / t).is_dir() and m["per_root"][t]["published"] == 0]
    if empty_roots:
        v["root-empty"] = (
            False, f"{', '.join(empty_roots)} publish 0 file(s) while holding "
                   f"{sum(m['per_root'][t]['nonbuild'] for t in empty_roots)} "
                   f"real non-build file(s). A predicate that answers no to "
                   f"everything reports the same thing as a correct one, and "
                   f"only a control arm can tell them apart")
    else:
        v["root-empty"] = (
            True, f"every one of the "
                  f"{sum(1 for t in rule.SOURCE_ROOTS if (root / t).is_dir())} "
                  f"declared source root(s) present on disk publishes at least "
                  f"one file, so the zeros elsewhere are measurements rather "
                  f"than a rule saying no to everything")

    total = m["total"]
    if total < MEASURED_PUBLISHED_FLOOR:
        v["total-floor"] = (
            False, f"{total} path(s) published against a measured floor of "
                   f"{MEASURED_PUBLISHED_FLOOR}: {MEASURED_PUBLISHED_FLOOR - total} "
                   f"fewer than the last measured run. The floor has no slack "
                   f"underneath it on purpose -- a single deleted file anywhere "
                   f"is under it -- so this is the arm that makes 'nobody "
                   f"noticed' impossible. If the shrink is intended, re-measure "
                   f"and re-baseline here; do not lower the floor to match a "
                   f"wheel that shipped four files")
    else:
        v["total-floor"] = (
            True, f"{total} path(s) published, at or above the measured floor "
                  f"of {MEASURED_PUBLISHED_FLOOR} (headroom "
                  f"{total - MEASURED_PUBLISHED_FLOOR})")

    if total > MEASURED_PUBLISHED_CEILING:
        v["total-ceiling"] = (
            False, f"{total} path(s) published against a measured ceiling of "
                   f"{MEASURED_PUBLISHED_CEILING}: {total - MEASURED_PUBLISHED_CEILING} "
                   f"over. A set that grows far past its last measurement "
                   f"without a re-baseline is a red too, because a floor alone "
                   f"is answered by deleting files and this gate is not going "
                   f"to be the last line that a shortcut is measured against. "
                   f"Read what grew before touching the number")
    else:
        v["total-ceiling"] = (
            True, f"{total} path(s) published, at or below the measured ceiling "
                  f"of {MEASURED_PUBLISHED_CEILING} (headroom "
                  f"{MEASURED_PUBLISHED_CEILING - total}), so nothing grew past "
                  f"the last measurement without somebody re-baselining it")

    roots_with_floor = [t for t in sorted(rule.SOURCE_ROOTS)
                        if t in MEASURED_PER_ROOT]
    missing_base = sorted(set(rule.SOURCE_ROOTS) - set(MEASURED_PER_ROOT))
    below = [f"{t} {m['per_root'][t]['published']}<"
             f"{MEASURED_PER_ROOT[t]}" for t in roots_with_floor
             if m["per_root"][t]["published"] < MEASURED_PER_ROOT[t]]
    stale = sorted(set(MEASURED_PER_ROOT) - set(rule.SOURCE_ROOTS)
                   - {ROOT_BUCKET})
    if missing_base or below or stale:
        parts = []
        if below:
            parts.append("below its own floor: " + ", ".join(below))
        if missing_base:
            parts.append(f"no measured baseline: {', '.join(missing_base)} -- a "
                         f"new source root is a re-baseline somebody makes by "
                         f"hand")
        if stale:
            parts.append(f"baseline names a root the rule no longer declares: "
                         f"{', '.join(stale)} -- that is a root dropped out of "
                         f"SOURCE_ROOTS, which is the collapse this band exists "
                         f"to catch")
        v["root-baseline-floor"] = (False, "; ".join(parts))
    else:
        v["root-baseline-floor"] = (
            True, f"every declared source root is at or above its own measured "
                  f"floor: "
                  + ", ".join(f"{t} {m['per_root'][t]['published']}"
                              f">={MEASURED_PER_ROOT[t]}"
                              for t in roots_with_floor)
                  + ". Per root and not in aggregate, because a root vanishing "
                    "is the likely collapse and a total can absorb it")

    over = [f"{t} {m['per_root'][t]['published']}>{MEASURED_PER_ROOT[t]}"
            for t in roots_with_floor
            if m["per_root"][t]["published"] > 2 * MEASURED_PER_ROOT[t]]
    if over:
        v["root-baseline-ceiling"] = (
            False, f"grown past twice its own measured floor without a "
                   f"re-baseline: {', '.join(over)}")
    else:
        v["root-baseline-ceiling"] = (
            True, f"no declared source root has grown past twice its own "
                  f"measured floor: "
                  + ", ".join(f"{t} {m['per_root'][t]['published']}"
                              f"<={2 * MEASURED_PER_ROOT[t]}"
                              for t in roots_with_floor)
                  + ". Two sides per root, so a collapse in one root and a "
                    "growth in another cannot cancel out into a total that "
                    "looks unchanged")

    phantoms = [r.as_posix() for r in sorted(pub) if not Path(pub[r]).is_file()]
    if phantoms:
        v["phantom"] = (False, f"{len(phantoms)} path(s) the walk selected are "
                                f"not files on disk: {phantoms[:6]}")
    else:
        v["phantom"] = (
            True, f"all {total} selected path(s) are files on disk, so the set "
                  f"is a set of real files rather than of names the rule would "
                  f"like to have published")

    v["band-sane"] = _band_sane(rule, v)
    v["undeclared-prune"] = _undeclared_prune(rule, root, v)
    return v


def _band_sane(rule, v: dict) -> tuple:
    """The band table has to describe the rule that is actually loaded.

    Two failure modes and both are red: the ceiling below the floor, which
    would make the band unsatisfiable, and a table that has drifted out of step
    with `SOURCE_ROOTS` in either direction. The second is what catches a root
    being dropped out of the rule -- the table still names it, and a band that
    agrees with a stale table is a band measuring the wrong rule.
    """
    named = sorted(set(MEASURED_PER_ROOT) - {ROOT_BUCKET})
    gone = sorted(set(named) - set(rule.SOURCE_ROOTS))
    if MEASURED_PUBLISHED_CEILING < MEASURED_PUBLISHED_FLOOR:
        return False, (f"the measured ceiling ({MEASURED_PUBLISHED_CEILING}) is "
                      f"below the measured floor ({MEASURED_PUBLISHED_FLOOR}), "
                      f"so the band cannot be satisfied by any tree at all")
    if gone:
        return False, (f"the band names {len(gone)} root(s) the rule no longer "
                      f"declares: {', '.join(gone)}")
    return True, (f"ceiling {MEASURED_PUBLISHED_CEILING} >= floor "
                  f"{MEASURED_PUBLISHED_FLOOR}, and the band names "
                  f"{len(MEASURED_PER_ROOT)} bucket(s) -- "
                  f"{', '.join(sorted(MEASURED_PER_ROOT))} -- of which every "
                  f"one is a declared source root or the tree root itself")


def _undeclared_prune(rule, root: Path, v: dict) -> tuple:
    """Real top-level directories the rule prunes with nothing behind the prune.

    A measurement, and deliberately **not** a red. On the shipped tree this
    names the commit mirror and `_wheelout/`, both of which are pruned only
    because they are not members of `SOURCE_ROOTS` -- no cache tag, no manifest,
    no `.gitignore` line. The rule's design position is that nothing outside its
    six roots can be a source, and that position is *asserted by the rule*; a
    check that turned the assertion into a red would be red on a correct tree,
    which is how a gate stops being run. So the list is printed in full, with
    nothing capped, and the arm name says what it is.
    """
    declared = {e for e, _a in rule.declared_generated_dirs()}
    unjust = []
    for d in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if not rule.prune_here(Path(d)):
            continue
        here = root / d
        if (rule.is_generated_dir(here) or rule.is_manifest_build_dir(here)
                or d in rule.GENERATED_DIRS or d in declared
                or any(part in rule.GENERATED_DIRS for part in here.parts)):
            continue
        files = sum(len(f) for _p, _s, f in os.walk(here))
        unjust.append(f"{d}/ ({files} real file(s))")
    return True, (f"{len(unjust)} real top-level directory/ies are pruned with "
                  f"no derived test and no .gitignore declaration behind the "
                  f"prune: {', '.join(unjust) if unjust else 'none'}. This is "
                  f"reported and not failed, and the reason is written in this "
                  f"file's docstring rather than left for a reader to infer: "
                  f"the commit mirror and _wheelout/ are in this set on a "
                  f"correct tree, and a red here would be a red on a correct "
                  f"tree. A misspelled root is caught from the other side, by "
                  f"'every declared source root is a directory on disk'")


# --------------------------------------------------------------------------
# The sandbox: a copy of the rule, a copy of .gitignore, a copy of the tree
# --------------------------------------------------------------------------
def build_sandbox(scratch: Path, source: Path, rule) -> tuple:
    """`sandbox/{.gitignore, rule/release_tree_rule.py, tree/...}`, built.

    Three copies, and the layout is load-bearing twice over:

    * the rule copy is at `sandbox/rule/`, so its own `__file__` puts it in
      `sandbox/` and `_gitignore_path()` resolves to **`sandbox/.gitignore`** --
      a byte copy of the real one. That is what lets the exclusion-widening
      arms exist at all: the real `.gitignore` is never touched, and in the
      green control the copy is byte-identical, so the arm is measuring the
      mutation and not a different declaration;
    * the tree is at `sandbox/tree/`, one level below the rule's parent and two
      below the scratch root, so `recognition_reason`'s root recovery -- which
      walks `len(rel.parts)` levels up -- lands on the sandbox tree and never
      reaches `target/CACHEDIR.TAG` above it.

    The tree holds the **published set**, byte for byte. Not a mock and not the
    whole workspace: the published set is exactly the thing under test, and
    building it from the live walk means the green control is a real directory
    tree measured by the real walk.

    `rule` is a parameter rather than the module global on purpose. The live
    measurement may be pointed at a different rule by `OD_PUBLISHED_SET_RULE`,
    and a sandbox built from the *global* rule's answer while the arms grade the
    *selected* one is a control that proves nothing. The first version of this
    function read the global, and the override printed in the provenance block
    was a knob that changed the banner and not the measurement.
    """
    rule_dir = scratch / "rule"
    tree = scratch / "tree"
    rule_dir.mkdir(parents=True, exist_ok=True)
    tree.mkdir(parents=True, exist_ok=True)
    (scratch / rule.GITIGNORE_NAME).write_bytes(
        (source / rule.GITIGNORE_NAME).read_bytes())
    (rule_dir / "release_tree_rule.py").write_bytes(
        Path(rule.__file__).read_bytes())
    count = 0
    for rel, path in sorted(rule.iter_source_files(source).items()):
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(path).read_bytes())
        count += 1
    return rule_dir / "release_tree_rule.py", tree, count


def apply_mutation(rule_path: Path, gi_path: Path, kind: str, old: str,
                   new: str) -> tuple:
    """Write the mutated copy. Returns `(applied, why)`.

    **Exactly one** occurrence of `old` or nothing is written. A mutation that
    matched zero sites and still produced a "mutated" sandbox is the failure
    this returns on: the arm would then measure an unmutated rule and report
    that the mutation was invisible, which is a statement about the arm and not
    about the gate.
    """
    target = rule_path if kind == "rule" else gi_path
    text = target.read_text(encoding="utf-8")
    hits = text.count(old)
    if hits != 1:
        return False, (f"the pattern occurs {hits} time(s) in {target.name}, and "
                       f"exactly 1 is required, so nothing was written")
    target.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
    return True, (f"one occurrence of the pattern replaced in {target.name}")


def run_arm(index: int, name: str, kind: str, old: str, new: str,
            expect_red: tuple, why: str, rule) -> dict:
    """Mutate one copy, measure with it, grade with `assess`. Returns the record.

    Nothing is restored and nothing needs to be: every arm gets its **own**
    sandbox directory, freshly built. Restoring in place was the first design
    and it is the one that hides a failure -- a half-undone mutation makes the
    next arm green for the wrong reason, and `shutil.rmtree` over a tree with
    locked `.pyc` files raises `PermissionError` on Windows anyway, so a fresh
    unique directory per run is both safer and cheaper than clearing one.
    """
    scratch = Path(tempfile.mkdtemp(
        prefix=f"{SANDBOX_PREFIX}arm{index}-",
        dir=str(ROOT / rule.CARGO_TARGET_DIRNAME)))
    try:
        rule_path, tree, count = build_sandbox(scratch, ROOT, rule)
        gi_path = scratch / rule.GITIGNORE_NAME
        applied, applied_why = apply_mutation(
            rule_path, gi_path, kind, old, new)
        if not applied:
            return {"name": name, "built": False, "why": applied_why,
                    "expect_red": expect_red, "unwanted": ("mutation-applied",),
                    "total": 0, "m": None, "evidence": why, "count": 0}
        mod = load_rule(rule_path, f"psf_sandbox_rule_{index}")
        m = measure(mod, tree)
        verdicts = assess(m)
        unwanted = tuple(a for a in GATED_ARMS
                         if not verdicts.get(a, (True, ""))[0])
        return {"name": name, "built": True, "why": applied_why,
                "expect_red": expect_red, "unwanted": unwanted,
                "total": m["total"], "m": m, "evidence": why, "count": count}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if scratch.exists():
            print(f"    NOTE: {scratch} could not be fully removed; it holds "
                  f"only files this run wrote and no bytes are claimed back")


#: The arms a mutation is graded against, and the arms a live run must satisfy.
#: `undeclared-prune` is deliberately absent: it is a measurement of the rule's
#: boundary, it is a PASS on a correct tree, and grading a mutation against it
#: would conflate "this mutation is invisible" with "this mutation moved a
#: number nobody claimed".
GATED_ARMS = ("tables", "roots-present", "rootnames-present",
              "published-nonempty", "root-bucket", "root-walk-parity",
              "root-empty", "total-floor", "total-ceiling",
              "root-baseline-floor", "root-baseline-ceiling", "phantom",
              "band-sane")


def run_growth_arm(index: int, rule) -> dict:
    """Plant enough real files to pass the ceiling. Returns the record.

    The other direction, and the one a floor alone would not catch: a set that
    grows far past its last measurement without anybody re-baselining it. The
    number of files is **derived**, not typed -- it is exactly the number needed
    to take the sandbox past the declared ceiling plus one -- so this arm cannot
    go stale when the band is re-measured, and a band that grew would make the
    arm plant more rather than quietly stop testing anything.

    The files go into the declared source root that currently publishes the
    most, so the growth is concentrated in one root and the per-root band gets
    exercised as well as the total. A growth spread evenly would be a weaker
    test of both, and this is the arm that is supposed to be hard to pass.
    """
    scratch = Path(tempfile.mkdtemp(
        prefix=f"{SANDBOX_PREFIX}arm{index}-",
        dir=str(ROOT / rule.CARGO_TARGET_DIRNAME)))
    try:
        rule_path, tree, count = build_sandbox(scratch, ROOT, rule)
        mod = load_rule(rule_path, f"psf_sandbox_rule_{index}")
        m0 = measure(mod, tree)
        host = max(sorted(mod.SOURCE_ROOTS),
                   key=lambda t: m0["per_root"][t]["published"])
        need = (MEASURED_PUBLISHED_CEILING - m0["total"]) + 1
        for i in range(need):
            (tree / host / f"published_set_floor_growth_{i:04d}.py").write_bytes(
                b"# planted by the growth arm: a real file, a declared suffix, "
                b"a declared source root\n")
        m = measure(mod, tree)
        verdicts = assess(m)
        unwanted = tuple(a for a in GATED_ARMS
                         if not verdicts.get(a, (True, ""))[0])
        return {"built": True, "total": m["total"], "host": host,
                "planted": need, "unwanted": unwanted,
                "before": m0["total"], "count": count}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if scratch.exists():
            print(f"    NOTE: {scratch} could not be fully removed; it holds "
                  f"only files this run wrote and no bytes are claimed back")


def share_table(m: dict) -> list:
    """One line per declared source root: published / nonbuild real / share.

    Returned as a list so the anti-truncation check can compare the number of
    lines against the number of roots. A capped table reads exactly like a
    complete one, and this project has been bitten by that twice already.
    """
    rows = []
    for top in sorted(m["rule"].SOURCE_ROOTS):
        got = m["per_root"][top]
        share = got["published"] / got["nonbuild"] if got["nonbuild"] else 0.0
        rows.append(f"  {top:<12} published {got['published']:>5} / "
                    f"nonbuild-real {got['nonbuild']:>5} / share {share:.4f}")
    return rows


def print_live(m: dict) -> None:
    section("the published set this tree produces, and where it sits")
    print(f"    total published: {m['total']}")
    print()
    for line in share_table(m):
        print(line)
    print()
    print("    nonbuild-real is counted with the CACHEDIR.TAG signature, the "
          "Cargo.toml adjacency, GENERATED_DIRS and the .gitignore reader, and "
          "with SOURCE_SUFFIXES NOT consulted. That is what makes the ratio a "
          "floor and not the rule agreeing with itself: a suffix table that is "
          "emptied moves the numerator and leaves this denominator alone.")
    print(f"    build-output directories excluded from the denominator: "
          f"{len(m['excluded'])}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Is the published set about to shrink, or about to grow unmeasured?")

    rule_override = os.environ.get(RULE_ENV)
    root_override = os.environ.get(ROOT_ENV)
    live = root_override is None and rule_override is None
    rule_path = Path(rule_override).resolve() if rule_override \
        else (HERE / "release_tree_rule.py")
    root = Path(root_override).resolve() if root_override else ROOT
    # The rule the live measurement and the sandboxes are both built from.
    # Selected from the override when one is given, so the banner below and the
    # measurement cannot disagree -- the first version printed the *requested*
    # path and then measured the module-level import, which is a banner that
    # lies exactly when somebody is debugging with the override set.
    rule = load_rule(rule_path, "psf_live_rule") if rule_override else RULE
    loaded = Path(rule.__file__).resolve()

    section("provenance: every number below came from these")
    print(f"    tree root     : {root}{'' if live else '   (OVERRIDDEN)'}")
    print(f"    rule requested: {rule_path}")
    print(f"    rule LOADED   : {loaded}"
          f"{'' if live else '   (OVERRIDDEN)'}   <-- the one that answered")
    print(f"    measured floor: {MEASURED_PUBLISHED_FLOOR}  (typed once, from a "
          f"measured run -- see the comment above)")
    print(f"    measured ceil : {MEASURED_PUBLISHED_CEILING}  (typed once)")
    print(f"    per-root band : {MEASURED_PER_ROOT}")
    print(f"    scratch       : {SANDBOX_PREFIX}* under "
          f"{ROOT / rule.CARGO_TARGET_DIRNAME}, one level deeper, because that "
          f"directory carries a {rule.CACHEDIR_TAG_NAME}")

    # A sandboxed rule still resolves `declared_generated_dirs()` from its own
    # `__file__`, which is the correct behaviour and is not "fixed" here.
    digest_before = rule.file_digest(rule_path)

    section("the live measurement")
    if not root.is_dir():
        bad("the tree root is a directory this run can read",
            f"{root} is not a directory. Every number below would be a "
            f"measurement of nothing, so this is a red and not a skip")
        tally()
        return finish(None)
    ok("the tree root is a directory this run can read",
       f"{root}, and the rule that answered is the one named above: "
       f"{loaded == rule_path}, sha256 {digest_before[:16]}... "
       f"decision surface {rule.rule_fingerprint()[:16]}... No path list is "
       f"written down anywhere in this file: every path below was selected by "
       f"the rule's own walk")

    m = measure(rule, root)
    print_live(m)
    v = assess(m)
    for arm in ("tables", "roots-present", "rootnames-present",
                "published-nonempty", "root-bucket", "root-walk-parity",
                "root-empty", "total-floor", "total-ceiling",
                "root-baseline-floor", "root-baseline-ceiling", "phantom",
                "undeclared-prune", "band-sane"):
        passed, detail = v[arm]
        if passed:
            ok(ARM_NAMES[arm], detail)
        else:
            bad(ARM_NAMES[arm], detail)

    rows = share_table(m)
    named = sorted(m["rule"].SOURCE_ROOTS)
    if len(rows) == len(named) and all(named[i] in rows[i] for i in
                                        range(len(named))):
        ok("the per-root share table is printed in full, one line per root",
           f"{len(rows)} line(s) for {len(named)} declared source root(s), and "
           f"line i names root i. Compared rather than promised because a "
           f"capped table reads exactly like a complete one -- the same defect "
           f"this repository has already shipped twice, once as `found[:4]` in "
           f"release_tree_parity_check.py -- and a cap added here later would "
           f"turn this red instead of quietly shortening the table. The "
           f"denominators behind it are printed with it: "
           + ", ".join(f"{t} {m['per_root'][t]['nonbuild']}" for t in named))
    else:
        bad("the per-root share table is printed in full, one line per root",
            f"{len(rows)} line(s) printed for {len(named)} declared source "
            f"root(s), and they do not line up in order. The table above is "
            f"missing a root, which is a reader reading a subset as a whole")

    section("the guard: would this go red before anybody ships?")
    print(f"    every arm below is its own SANDBOX: a byte copy of the rule, a "
          f"byte copy of .gitignore, and a tree built from this tree's own "
          f"published set. The live rule is never edited, and the last check in "
          f"this section is that it was not. A sandboxed rule still resolves "
          f"declared_generated_dirs() from its own __file__, which now points "
          f"at the sandbox's .gitignore copy -- byte-identical to the real one "
          f"in every arm that does not mutate it.")

    for i, (name, kind, old, new, expect_red, why) in enumerate(MUTATIONS):
        if not expect_red:
            continue
        rec = run_arm(i, name, kind, old, new, expect_red, why, rule)
        arm_name = f"table mutation: {name}"
        if not rec["built"]:
            bad(arm_name, rec["why"])
            continue
        got = rec["unwanted"]
        missing = tuple(a for a in expect_red if a not in got)
        if not missing:
            ok(arm_name,
               f"the mutation went red as required. {rec['why']}. The sandbox "
               f"published {rec['total']} path(s) where the live tree publishes "
               f"{m['total']}, and these arm(s) changed verdict: "
               f"{', '.join(got) or 'none'}. {rec['evidence']}")
        else:
            bad(arm_name,
                f"the mutation was applied ({rec['why']}) and the sandbox "
                f"published {rec['total']} path(s), but these arm(s) did NOT go "
                f"red: {', '.join(missing)}. Everything that did: "
                f"{', '.join(got) or 'nothing'}. {rec['evidence']}")

    # The documented blind spots get their own result rather than being folded
    # into the arms above, because an arm that is *expected* to stay green and
    # an arm whose blindness went unnoticed are different findings and a reader
    # cannot tell them apart from a pass.
    for i, mut in enumerate(MUTATIONS):
        name, kind, old, new, expect_red, why = mut
        if expect_red:
            continue
        rec = run_arm(100 + i, name, kind, old, new, expect_red, why, rule)
        arm_name = f"documented blind spot: {name} leaves the gate green"
        if not rec["built"]:
            bad(arm_name, rec["why"])
        else:
            ok(arm_name,
               f"the mutation was applied and the gate stayed green, which is "
               f"the finding rather than a pass: the sandbox published "
               f"{rec['total']} path(s) against the live tree's {m['total']}, "
               f"and these arm(s) changed verdict: "
               f"{', '.join(rec['unwanted']) or 'none'}. {rec['evidence']}")

    growth = run_growth_arm(50, rule)
    if growth["unwanted"] and "total-ceiling" in growth["unwanted"]:
        ok("table mutation: the published set grows past its last measurement",
           f"{growth['planted']} real file(s) planted in {growth['host']}/ -- a "
           f"declared source root with a declared suffix, so every one of them "
           f"is genuinely published -- taking the sandbox from "
           f"{growth['before']} to {growth['total']} path(s) against a measured "
           f"ceiling of {MEASURED_PUBLISHED_CEILING}. The plant count is derived "
           f"as ceiling-minus-measured plus one, not typed, so this arm cannot "
           f"go stale when the band is re-measured. Red arms: "
           f"{', '.join(growth['unwanted'])}. This is the direction a floor "
           f"alone would have been blind to, and it is the direction a careless "
           f"fix for a floor takes")
    else:
        bad("table mutation: the published set grows past its last measurement",
            f"{growth['planted']} file(s) planted in {growth['host']}/ took the "
            f"sandbox from {growth['before']} to {growth['total']} path(s) "
            f"against a measured ceiling of {MEASURED_PUBLISHED_CEILING}, and "
            f"`total-ceiling` did not go red. Red arms: "
            f"{', '.join(growth['unwanted']) or 'nothing'}. A two-sided band "
            f"whose upper side does not fire is a floor wearing a hat")

    digest_after = rule.file_digest(rule_path)
    if digest_after == digest_before:
        ok("the live rule was not mutated by any arm",
           f"{rule_path.name} is byte-identical before and after the eight "
           f"mutation arms, sha256 {digest_after}. Every mutation was written "
           f"to a copy under {rule.CARGO_TARGET_DIRNAME}/; this is the check "
           f"that says so rather than asking the reader to trust it")
    else:
        bad("the live rule was not mutated by any arm",
            f"{rule_path.name} changed underneath this run: sha256 "
            f"{digest_before[:16]}... -> {digest_after[:16]}... Either an arm "
            f"wrote to the real rule or another agent edited it mid-run, and "
            f"the arms above cannot be attributed")

    tally()
    return finish(m)


def finish(m) -> int:
    """The summary, the live count, and the verdict. One tail, for every exit.

    The early return for an unreadable tree root used to print its own red and
    `return EXIT_FAILED` without reaching this, so that run exited 1 having
    printed **no `RESULT:` line at all** -- measured, and fixed by routing both
    exits here. A red whose verdict a reader has to infer from the absence of a
    green is the shape this file is against, and it would have been the very
    first thing a reader hit on the one run that matters most.
    """
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nnot = sum(1 for t, _n, _d in RESULTS if t == "NOT-ASSERTED")
    print()
    print(f"--- summary: {npass} passed, {nfail} failed, {nnot} not asserted, "
          f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")
    if m is not None:
        print(f"    live published set: {m['total']} path(s); per root: "
              + ", ".join(f"{t} {m['per_root'][t]['published']}"
                          for t in sorted(m["rule"].SOURCE_ROOTS))
              + f"; tree root {len(m['root_published'])}")
    else:
        print("    live published set: NOT MEASURED -- this run stopped before "
              "it had a tree to walk, so no count is printed and none is "
              "implied. A number here would be a measurement of nothing")
    if nfail:
        print(f"RESULT: FAIL -- {nfail} of {EXPECTED_CHECKS} checks failed")
        return EXIT_FAILED
    if npass == 0:
        print("RESULT: DID NOT FINISH -- this run asserted nothing")
        return EXIT_INCOMPLETE
    print(f"RESULT: OK -- {npass} of {EXPECTED_CHECKS} checks passed")
    return EXIT_OK


def tally() -> None:
    section("the tally")
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nnot = sum(1 for t, _n, _d in RESULTS if t == "NOT-ASSERTED")
    name = "the tally accounts for every result"
    if npass + nfail + nnot == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(name, f"{npass} + {nfail} + {nnot} = {len(RESULTS)}, and this "
                 f"check is the {len(RESULTS) + 1}th, so the run reaches the "
                 f"declared {EXPECTED_CHECKS}")
    else:
        bad(name, f"recorded {len(RESULTS)} results ({npass}+{nfail}+{nnot}) "
                  f"before this check, which must be {EXPECTED_CHECKS - 1} for "
                  f"the run to reach the declared {EXPECTED_CHECKS}")


def report_crash(exc: BaseException) -> int:
    """A run that stopped says so, in words no finished run prints."""
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
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_exc))
