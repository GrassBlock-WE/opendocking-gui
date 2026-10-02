"""The one implementation of "what is a source file", and it ships.

**Why this file exists at all.** The rule used to be a private copy inside
`_sync_release.py`, at the workspace root. That is the worst of both worlds: the
root-level `_` prefix is what the rule itself uses to keep scratch files out of
the release, so the file carrying the rule was *excluded by the rule it
contained*. The commit tree -- the artefact a user clones -- therefore held no
copy of the decision that decided what it held. A rule that cannot be read from
the thing it governs is not auditable, and "the tool asserts byte-equality at the
end of a run somebody ran by hand" is not a contract either.

So the rule moved here, into a file that is published, and both consumers import
it: `_sync_release.py` (which copies) and `release_tree_parity_check.py` (which
compares). One implementation, two callers, and the copy that ships is the copy
that runs. Nothing here is a second opinion about the rule; if you find a
heuristic about "is this a source file" anywhere else in either tree, that is a
defect, and `release_tree_parity_check.py` fails on it by parsing the trees.

**The rule is a recognition rule, and that is a change of shape, not of taste.**

It used to be an *exclusion* rule: three tables of things not to ship
(`EXCLUDE_DIRS`, `EXCLUDE_NAMES`, `EXCLUDE_SUFFIX`), plus two predicates, and
everything not named was published. That shape has one property and one defect,
and the defect is the one that keeps costing this project money. It is closed
under "anything I did not think of": a new kind of scratch file, build output or
accident ships, silently, until a person notices and adds a row. The census that
replaced the hand-typed ledger here grew its exclusion list from two entries to
twelve in a week, and five of those twelve were files nothing in the repository
referenced at all. An exclusion list is a promise about everything you failed to
think of.

So membership is now a *claim about what belongs*:

    a file is shipped if and only if
        (its path lies under a declared SOURCE_ROOT)
      and (its suffix is a declared SOURCE_SUFFIX, or its whole name at the
           tree root is a declared ROOT_SOURCE_NAME)

Both halves are positive, so a file nobody has heard of is not shipped *by
construction*. The accident this file was written for -- `$null`, a 668-byte
file at the workspace root that PowerShell wrote when a redirect was typed
wrong -- is not shipped because its suffix is empty and its name is not
declared. Nobody put `$null` on a list. The empty suffix simply is not a suffix
this repository ships, and that was already true of every other suffix-less
accident before it.

**What the inversion costs, stated here because it is a real cost.** Every
suffix and every root in the two tables is now a *decision somebody has to keep
true*, and this failure mode is the mirror image of the one it replaced:

* A new file kind that *should* ship is not shipped until somebody adds its
  suffix. The direction of that failure is the good one -- the file simply does
  not reach the release, and `release_tree_parity_check.py` reports it as
  missing from the published tree, which is a red someone reads. An exclusion
  rule's failure is the opposite: the file *is* published, and the report is
  green until a user notices.
* The tables can rot in the other direction too: a declared suffix that nothing
  produces (`SOURCE_SUFFIXES` carries `.lock` for `Cargo.lock` and `.cff` for
  `CITATION.cff` today) is a claim about the future that nothing enforces. The
  parity check asserts every declared root name exists; it cannot assert that a
  declared suffix is still wanted, because there is no observation of "a suffix
  that should have shipped and did not" from inside either tree.
* So the honest statement is: an exclusion list is a promise about everything
  you failed to think of, and a recognition rule is a claim about what belongs
  that can be checked -- and the check is only as good as the table it checks,
  which somebody still has to keep true. The inversion buys a failure that is
  loud and local. It does not buy one that is absent.

**The two tables, and why `ROOT_SOURCE_NAME` exists at all.** `SOURCE_SUFFIXES`
is recognisable per root: `.rs` under `dock-core`, `.pdbqt` under `examples`.
At the *tree root* the suffix is not enough, and this is a finding about what
actually ships rather than a workaround for it. Six root files end in `.md` and
one of them, `odck.md`, is a scratch document that must not ship. Suffix
recognition would publish it. And four root files have no suffix at all --
`LICENSE`, `.gitignore`, `.gitattributes` are three, `$null` was the fourth --
so a rule that recognised the empty suffix would publish the accident instead.

So the root is recognised by *whole name*, in a table that is short, positive,
and whose every entry the parity check asserts exists. That is the one place a
name has to be written down, and it is a list of what ships rather than a list
of what does not: an unlisted root file is not shipped without anyone having
thought of it, which is the property the inversion was for.

**The bootstrap gap: past tense from the first sync, and a fresh gap on every
clone.** The first sync has run. It carried 149 files and deleted 16, and the
fingerprint below is now in the published tree, so "the copy that ships is the
copy that runs" is a property this project *has*, not one it intends.

It was false before that commit, and it is false again on every fresh clone.
The rule is a source file, so a clone has the rule and not the compiled
extension, and `_dockpy.pyd` is listed in `.gitignore` -- git never tracked it,
so no published tree can carry one. Every CI job builds and installs the wheel
itself. **That is the honest design and it is not a defect**, but it means the
artefact a user clones is not a runnable tree, and a reader who wants one has to
build it.

So the thing a reader has to do to re-establish the property, now that the first
commit carried it: **commit the rule, and let the next sync publish it.** There
is no shortcut and there should not be one, because the ordering cannot be
reversed -- this file has to exist in the development tree before any copy can
carry it. A gate cannot close that: a check asserting the property would be
false on a clean checkout and would have to be written to pass, which is the
trade this repository refuses.

"The copy
that ships is the copy that runs" is true of this file *after* the first sync and
false *before* it, and the ordering cannot be reversed: `_sync_release.py` imports
this module in order to decide what to copy, so this file has to exist in the
development tree before any sync can publish it, and a one-way copy can only
publish what is already there. The property is therefore unreachable on a clean
checkout of the commit tree, which is exactly where it is checked.

No gate closes that, and none should be written. A check asserting a property
that is false on the first run of a clean checkout has to be written to pass
anyway, and a check written to pass is the thing this repository keeps refusing.
So it is recorded here as what it is -- a property with a before and an after,
not a defect -- and what the first sync is *for* is to move the tree from the
"after" side of that sentence. Until it runs, `release_tree_parity_check.py`
reports this file among the sources missing from the published tree, which is
the honest reading and not a failure anybody has to explain away.

**The third copy of the package is out of scope, on purpose.** Three copies of
`opendocking` exist on a maintainer's machine: the development tree, this
project's own commit tree under it, and the one the interpreter installed into
its `site-packages`. Only the first two are what this repository publishes. The
third is a local install artefact: it holds a compiled `_dockpy.pyd` for one
specific interpreter build, it is not in the commit tree, and putting it there
would be publishing a binary nobody else's Python can load. It is therefore
*declared* out of scope rather than ignored, and the declaration is a named
constant here so that a reader can find it, plus a check in
`release_tree_parity_check.py` that proves the boundary is a path property
(no `site-packages` inside either published tree) instead of a promise in a
comment. "Deliberately out of scope" and "nobody looks at it" are different
claims and only one of them is safe to leave undocumented.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: The commit tree's directory name, relative to the workspace root. It is a
#: *sibling* of the development tree, not a child of the package, and a path that
#: mentions it is never a source of the development tree.
RELEASE_TREE_DIRNAME = "opendocking-gui"

#: Where the package legitimately lives inside the workspace. Everything else
#: named `opendocking` is a copy somebody made, not a source directory.
CANONICAL_PACKAGE_PARENT = Path("dock-py", "python")

#: The top-level directories that contain source, recognised as the *first* path
#: component and nothing else: `dock-py/python/opendocking/core.py` is under
#: `dock-py`, and there is no third spelling of that fact here.
#:
#: The table is also the walk's pruning boundary, which is why it has to be
#: monotone: nothing outside these six directories can be a source, so
#: `prune_here` skips everything else -- `target/` (~19 000 files),
#: `dist/`, `_wheelout/`, `.git/`, `opendocking-gui/` -- without deciding each
#: file individually. That it is sound is measured, not assumed:
#: `release_tree_parity_check.py` compares this walk against an unpruned one.
SOURCE_ROOTS = frozenset({
    ".github", "dock-core", "dock-py", "docs", "examples", "scripts",
})

#: Suffixes this repository ships, under any declared `SOURCE_ROOT`. Read off the
#: tree by measurement rather than by taste; each one is there because a file
#: with it is a file somebody clones:
#:
#:   .cff  CITATION.cff                .ini  a package data file
#:   .lock Cargo.lock                  .md   the documents
#:   .pdb  the example receptors       .pdbqt the example receptor and poses
#:   .pyd  the compiled extension      .py   the product and the release tooling
#:   .rs   the engine                  .sdf  the example ligands
#:   .toml Cargo.toml, pyproject       .txt  requirements.txt
#:   .wgsl the shader                  .yml  the workflows
#:
#: Note what is *not* here and used to be, as a build-artefact list:
#: `.pyc`, `.pyo`, `.map`, `.whl`, `.log`, `.exe`, `.dll`. Under an exclusion
#: rule each of those needed a row; under this one they are absent because
#: nothing declares them. `examples/` ships real `.pdb` and `.pdbqt` inputs,
#: which is exactly why a blanket "delete the suffix" rule was wrong: it deleted
#: them from the commit tree. The empty suffix is absent for the same reason and
#: is what keeps `$null` out.
SOURCE_SUFFIXES = frozenset({
    ".cff", ".ini", ".lock", ".md", ".pdb", ".pdbqt", ".pyd", ".py", ".rs",
    ".sdf", ".toml", ".txt", ".wgsl", ".yml",
})

#: Files at the tree root that ship, by whole name. The root is the one place
#: the suffix cannot decide, for two measured reasons: six root files end in
#: `.md` and one of them (`odck.md`, a scratch document) must not ship, and four
#: root files have no suffix to read.
#:
#: This is a recognition table, so it is bounded by what ships rather than by
#: what might: an entry here is a claim that the file belongs, and
#: `release_tree_parity_check.py` asserts every one of them exists in the tree.
#: A new root file is not published until it is added here, which fails in the
#: direction that shows up as a red rather than as an unwanted file in a clone.
ROOT_SOURCE_NAMES = frozenset({
    ".gitattributes", ".gitignore", "CITATION.cff", "CHANGELOG.md",
    "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "Cargo.lock", "Cargo.toml",
    "LICENSE", "README.en.md", "README.md", "SECURITY.md",
    "requirements.txt",
})

#: The file a directory writes to declare that it is a *cache* directory, and the
#: signature it writes inside it. This is the `cachedir` specification at
#: bford.info/cachedir/spec.html, and the tool that wrote our copy says so in
#: its own tag file: `dock-py/.pytest_cache/CACHEDIR.TAG` reads
#: "Signature: 8a477f597d28d172789f06886806bc55" and then links the spec.
#:
#: **This is the derived half of the rule, and it is the half the inversion was
#: missing.** A recognition rule that only recognises sources is still a list --
#: a list of the good kind, but a list, and it ships whatever nobody thought to
#: classify. So the rule also has to *recognise a not-source*, and this is the
#: test that does it without a name: a directory that carries this signature has
#: said out loud that it is regenerated by running something, and nothing inside
#: it is hand-written source. The test is a fixed 48-byte constant with no
#: pattern language in it, so there is no matcher to get subtly wrong, and it
#: covers tools nobody has run here yet: mypy, pip, black and ruff all write
#: the same tag.
#:
#: **The stronger derivation available, and why it was not taken.** The tree's
#: own `.gitignore` ships (`ROOT_SOURCE_NAMES` below), and it already declares
#: `__pycache__/`, `.pytest_cache/` and `$null` -- line 38 is the PowerShell
#: accident, with a comment explaining it. Reading it would make the rule *be*
#: the project's own declaration rather than a copy of it, which cannot drift.
#: It was not taken because a gitignore pattern language -- anchoring, `**`,
#: directory-only suffixes, `!` negation -- is a second rule of comparable
#: subtlety to the one being replaced, and a bug in it is a file shipped or a
#: file silently dropped. The cachedir signature has neither problem. If this
#: repository ever wants the stronger form, the tag test is the floor it would
#: sit on, not a replacement for it.
CACHEDIR_TAG_NAME = "CACHEDIR.TAG"
CACHEDIR_TAG_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55"

#: Cargo's manifest, and the directory Cargo writes into when nothing says
#: otherwise. A crate's build tree is not a directory with a suspicious name: it
#: is the directory the manifest beside it says Cargo writes to. So
#: `dock-py/target/` is generated **by construction** -- `dock-py/Cargo.toml` is
#: right there declaring `dock-py` a crate -- and that derivation works before
#: `cargo build` has ever run, which is the case no marker rule can reach.
#:
#: **This derivation is Cargo-only, and that is a hole shaped exactly like the
#: one it fills.** Measured over this tree: `Cargo.toml` exists at the root and
#: under both `dock-py/` and `dock-core/`, so two build trees are derivable.
#: The other ecosystems present declare no output directory at all --
#: `dock-py/pyproject.toml` has no build-dir setting, PEP 517's `build/`,
#: `dist/` and `*.egg-info/` are frontend conventions rather than manifest
#: content, and there is no `package.json` anywhere in the tree, so Node has
#: nothing to derive from even in principle. A derivation that works for one
#: ecosystem and silently does not work for three would be a lie told in code,
#: so it is declared here, and `release_tree_parity_check.py` section E pins the
#: boundary with a planted tree rather than leaving it to be discovered.
#:
#: **What it costs, and what it does not cover.** The default target directory
#: is overridable by `build.target-dir` in `.cargo/config.toml` and by the
#: `CARGO_TARGET_DIR` environment variable. Neither is checked, and the
#: environment variable cannot be: it is not in the tree. What *is* checkable is
#: the file form, and there is no `.cargo/config.toml` at the root, under
#: `dock-py/` or under `dock-core/`, so on this tree the default holds and the
#: derivation is sound. A tree that overrides it has a directory this rule will
#: mis-handle, and the honest statement is that the derivation is sound for the
#: default layout rather than sound in general.
CARGO_MANIFEST_NAME = "Cargo.toml"
CARGO_TARGET_DIRNAME = "target"

#: The repository's own declaration of what it does not track, read as a list of
#: **directory names and paths**. Not a pattern language.
#:
#: Measured over the 18 patterns this repository's `.gitignore` carries:
#: **10 literal directories, 3 literal files, 6 wildcard patterns, 0 negations.**
#: The reader below interprets the 10 and nothing else -- it strips a leading
#: `/`, strips a trailing `/`, and matches a directory whose relative path or
#: basename is the entry. There is no `*`, no `?`, no character class, no `!`,
#: and no file literal. That is not a partial parser of git's grammar; it is a
#: list of directory names read out of a file that is already in the tree and
#: already ships.
#:
#: **What it pays, and the bill it settles.** Three directories in the tree are
#: named by `.gitignore`, read by nothing, and were being shipped:
#: `examples/maps/`, `examples/multi/` and `examples/split_dir/`. A repository
#: that declares a directory non-source and then publishes it has two
#: declarations disagreeing, and the rule was the one that got to publish. Three
#: of the ten entries (`target`, `__pycache__`, `.pytest_cache`, `dist`,
#: `dist-gpu` at the root) were already answered by the root and generated-dir
#: tests; this reader agrees with them, which is the point of having two
#: declarations answer the same question.
#:
#: **Why file literals are deliberately not read.** `examples/_shifted_
#: receptor.pdbqt` is named by `.gitignore` and is an *input* that
#: `workbench_interaction_check.py` reads, so the `.gitignore` entry is wrong
#: and the rule is right. `$null` is named and is already declined by
#: `ROOT_SOURCE_NAMES`. `dockpy.dll` is named and does not exist. Reading file
#: literals would have made the rule delete a file the suite needs, on the
#: authority of a line that is wrong. So they are not read, and
#: `release_tree_parity_check.py` records each one with its answer instead --
#: which is how "the two declarations disagree about a file, and here is which
#: one is right" becomes a check rather than a comment.
#:
#: **What it costs.** A rule that reads a second file can now be wrong because
#: that file is malformed, and a reader that ignores what it cannot parse is
#: worse than one that does not exist. Both are answered by the same check that
#: audits the other tables: every non-comment line in `.gitignore` must appear
#: in that check's answer table, and any line carrying a construct this reader
#: does not implement -- a wildcard, a negation -- must be *listed there as
#: unimplemented* rather than silently skipped. A new tool that writes into a
#: source root and adds one line to `.gitignore` therefore cannot be published
#: by accident, and cannot be ignored by accident either.
GITIGNORE_NAME = ".gitignore"
#: Constructs this reader does not implement. Present so a future line using one
#: is recognisable rather than merely unmatched.
GITIGNORE_UNIMPLEMENTED = ("*", "?", "[", "!")


def _gitignore_path() -> Path:
    """The `.gitignore` beside this module, resolved from `__file__`.

    From `__file__` and not from the working directory, for the same reason
    `_sync_release.py` derives its root that way: the rule has to answer the same
    question from any cwd, and from the tree the module actually lives in -- the
    development tree or the published one.
    """
    return Path(__file__).resolve().parent.parent / GITIGNORE_NAME


def declared_generated_dirs() -> frozenset:
    """Directory names and paths the shipped `.gitignore` declares non-source.

    Read once, from the file, and returned as a set. Nothing here interprets a
    wildcard: a line carrying one is a *pattern*, this reader has no opinion
    about it, and `release_tree_parity_check.py` is the place that says so out
    loud. A missing or unreadable `.gitignore` yields an empty set rather than an
    exception -- the rule then behaves as it did before this reader existed, and
    the check is what notices the file is gone.
    """
    try:
        text = _gitignore_path().read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return frozenset()
    out: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("!") or any(c in s for c in GITIGNORE_UNIMPLEMENTED):
            continue
        if not s.endswith("/"):
            continue          # a file literal, deliberately not read
        anchored = s.startswith("/") or "/" in s.strip("/")
        out.add((s.strip("/"), anchored))
    return frozenset(out)


#: Directories a tool writes that do **not** carry the cachedir tag.
#:
#: **One name, and the derivation was tried and does not exist for it.**
#: `is_generated_dir` decides `.pytest_cache` and `.mypy_cache` without being
#: told their names, because those tools write the tag. CPython writes **no**
#: tag into `__pycache__`, so there is nothing in that directory to derive the
#: answer from: no marker, no self-declaration, no convention. That is the
#: honest boundary of the derived half of this rule, and this table is the name
#: that buys it.
#:
#: What the name costs, measured rather than argued: **it is currently
#: redundant.** `__pycache__` holds only `.pyc`, no suffix in `SOURCE_SUFFIXES`
#: claims `.pyc`, and so the suffix test already declines every file in every one
#: of them -- 32 files across four `__pycache__` directories on this machine,
#: 0 shipped with this row deleted. So the row buys nothing today and is kept
#: for the one case it would buy something for: a file with a declared suffix
#: inside `__pycache__`, which CPython cannot create and which a stale copy or a
#: hand-run `compileall` could. The alternative to keeping it -- accepting that
#: such a file ships -- is a worse trade for one word, and `prune_here` prunes
#: the subtree either way.
GENERATED_DIRS = frozenset({"__pycache__"})

#: The names a second implementation of this rule would have to define. The
#: parity check parses both trees for these, so moving a table or a predicate out
#: of this file is a check that can go red rather than an edit nobody sees.
RULE_FUNCTIONS = (
    "is_package_copy",
    "declared_generated_dirs",
    "is_generated_dir",
    "is_manifest_build_dir",
    "recognition_reason",
    "is_source",
    "prune_here",
    "iter_source_files",
    "file_digest",
)
RULE_TABLES = (
    "SOURCE_ROOTS",
    "SOURCE_SUFFIXES",
    "GENERATED_DIRS",
    "ROOT_SOURCE_NAMES",
    "CANONICAL_PACKAGE_PARENT",
)

#: Why the third copy of the package is not part of this contract, in enough
#: words to be quotable in a report. The check asserts this string exists rather
#: than trusting that the next reader will find this file; a decision that lives
#: only in a comment is a decision that gets reversed by whoever reads the
#: directory listing.
SITE_PACKAGES_DECISION = (
    "out of scope: the copy under the running interpreter's site-packages is a "
    "local install artefact, not part of what this repository publishes. It "
    "carries a compiled _dockpy.pyd built for one interpreter and one platform, "
    "it is not in the commit tree, and the copy is one-directional (development "
    "tree -> commit tree), so nothing in either published tree depends on it. It "
    "drifts, and drift there is fixed by reinstalling, not by syncing"
)


def is_package_copy(rel: Path) -> bool:
    """True when this file sits inside a *second* copy of the package.

    A hand-maintained name list is how `.term-overlay/opendocking/` reached
    the sync in the first place: an install overlay nobody wrote down got
    copied into the release tree, taking a compiled ``_dockpy.pyd`` and a
    partial duplicate of the package with it. The name of the next overlay is
    not knowable in advance, so the test is structural instead: a path that
    contains an ``opendocking`` directory component which is not this project's
    own source location is a copy, wherever it came from and whatever it is
    called.

    The point is not tidiness. The release tree is what a user clones, and a
    second copy of the package inside it is a second thing named
    ``opendocking`` -- the same ambiguity that cost this project an hour of
    "where did the file go" today, except shipped to everyone.

    Still load-bearing under the recognition rule rather than folded into it: a
    copy nested *inside* a declared source root -- `dock-py/python/opendocking/`
    with a `.venv` in it, say -- passes both halves of the (root, suffix) test,
    and this is what stops it.
    """
    parts = rel.parts
    if "opendocking" not in parts:
        return False
    index = parts.index("opendocking")
    return Path(*parts[:index]) != CANONICAL_PACKAGE_PARENT


def is_generated_dir(abs_dir: Path) -> bool:
    """True when this directory *declares itself* a cache directory.

    The declaration is the `CACHEDIR.TAG` file described at
    :data:`CACHEDIR_TAG_SIGNATURE`. It is the only test in this file that reads
    the filesystem, and that is why the two membership functions below take an
    optional absolute directory: a path-only predicate cannot answer "is this
    directory a cache", and answering it by name is the thing this rule stopped
    doing.

    Reads at most the first 256 bytes, and an unreadable or absent tag is
    ``False`` rather than an error: a directory with no tag is an ordinary
    directory until something says otherwise, and a rule that raised here would
    make the release depend on the permissions of every directory in the tree.
    """
    try:
        head = (Path(abs_dir) / CACHEDIR_TAG_NAME).read_bytes()[:256]
    except OSError:
        return False
    return CACHEDIR_TAG_SIGNATURE.encode("ascii") in head


def is_manifest_build_dir(abs_dir: Path) -> bool:
    """True when this directory is where Cargo would write, by its own manifest.

    The second derived test, and the only one that works **before the tool has
    run**: `dock-py/Cargo.toml` declares `dock-py` a crate, so `dock-py/target/`
    is a build directory whether or not anything has ever been built into it.
    No marker, no run, no name heuristic beyond Cargo's own default.

    Distance one on purpose. Cargo puts the target directory beside the *root*
    manifest -- the workspace root for a virtual workspace -- and a member crate
    does not get its own, so looking further up would be wrong for exactly the
    layout that is most common.

    It is deliberately not "a directory named `target`". That version would
    decline `scripts/target/anything.py` in a repository that merely has a
    directory with a popular name, which is the over-broad direction of the same
    mistake; the manifest is the thing that makes it a build tree, and without
    the manifest this function says no.
    """
    d = Path(abs_dir)
    if d.name != CARGO_TARGET_DIRNAME:
        return False
    return (d.parent / CARGO_MANIFEST_NAME).is_file()


def recognition_reason(rel: Path, abs_dir: Path | None = None,
                           honour_declaration: bool = True) -> str | None:
    """Why this path is *not* a source file, or ``None`` when it is one.

    Named for what it does rather than for what it replaced: under a recognition
    rule almost every path here is a non-source, so "exclusion reason" described
    a minority case and would have read as though the rule were still an
    exclusion list.

    The reason is a separate function rather than a bool because the two
    consumers need different halves of the answer and neither should have to
    re-derive it: the sync tool prints the reason when it deletes a file from the
    commit tree, and the parity check reports a file the rule would not ship as a
    sentence a reader can act on. Returning a reason instead of a flag also means
    a new non-source cannot be added without saying what it is.
    """
    if is_package_copy(rel):
        return ("a second copy of the package: the path contains an "
                "`opendocking` component that is not "
                f"{CANONICAL_PACKAGE_PARENT.as_posix()}")
    if not rel.parts:
        return "the path is the tree root, which is not a file"
    if len(rel.parts) == 1:
        if rel.name in ROOT_SOURCE_NAMES:
            return None
        return ("a root-level file whose whole name is not declared in "
                "ROOT_SOURCE_NAMES. The suffix cannot decide at the root -- six "
                "root files end in `.md` and one of them must not ship, and four "
                "have no suffix at all -- so root membership is by name. This is "
                "the shape that keeps a suffix-less accident out without anybody "
                "having listed it")
    top = rel.parts[0]
    if top not in SOURCE_ROOTS:
        return (f"it is under {top}/, which is not a declared source root "
                f"({sorted(SOURCE_ROOTS)}), so nothing under it is recognised "
                f"as source. Build output, caches, the commit tree beside this "
                f"one and untracked scratch directories all land here, and none "
                f"of them needed to be named")
    for part in rel.parts:
        if part in GENERATED_DIRS:
            return (f"{part}/ is a generated directory by name, from "
                    f"GENERATED_DIRS. That table is empty today; the branch "
                    f"stays because it is where the next measured case belongs")
    if abs_dir is not None:
        # Both derived tests ask about a *directory tree*, and a file can be
        # several levels inside the directory that matters: `dock-py/target/
        # debug/lib.rs` is two levels below `dock-py/target/`. The first version
        # of this tested only the file's own directory, so it never reached the
        # manifest, and the coverage probe reported the build tree as a source.
        # The root is recovered from the two arguments rather than passed in --
        # `abs_dir` is exactly `len(rel.parts)` levels below it -- so the callers
        # that walk a tree did not have to change.
        root = Path(abs_dir)
        for _ in rel.parts:
            root = root.parent
        node = Path(abs_dir)
        while True:
            if is_generated_dir(node):
                return (f"{node.name}/ declares itself a cache directory: it "
                        f"carries a {CACHEDIR_TAG_NAME} with the cachedir "
                        f"signature, so it is rewritten by running something and "
                        f"nothing inside it is hand-written source. The "
                        f"directory's *name* is irrelevant here -- this is the "
                        f"test that catches the cache of a tool nobody has run "
                        f"yet")
            if is_manifest_build_dir(node):
                return (f"{node.name}/ is the Cargo build directory of "
                        f"{CARGO_MANIFEST_NAME} one level up, which is where "
                        f"Cargo writes unless build.target-dir or "
                        f"CARGO_TARGET_DIR says otherwise. Derived from the "
                        f"manifest rather than from the name or from anything a "
                        f"run left behind, so it holds before the first build. "
                        f"Cargo-only: no other ecosystem in this tree declares "
                        f"its output directory in a manifest")
            if node == root or node.parent == node:
                break
            node = node.parent
        # Last, not first, and that ordering is load-bearing. The reader is the
        # broadest of the three tests, so running it first made every
        # `.pytest_cache` and every `target/` answer with "the .gitignore says
        # so" and cost the more specific mechanisms their reasons -- which is
        # not cosmetic: three checks here assert the *reason*, and a broad test
        # that answers first makes a narrow one unobservable.
        declared = declared_generated_dirs() if honour_declaration else frozenset()
        if declared:
            walked = ""
            for part in rel.parts[:-1]:
                walked = f"{walked}/{part}" if walked else part
                for entry, anchored in declared:
                    if (walked == entry if anchored
                            else (not anchored and part == entry)):
                        return (f"{walked}/ is declared non-source by the "
                                f"repository's own {GITIGNORE_NAME}, which this "
                                f"rule reads as a list of directories. One "
                                f"declaration, and the rule cannot publish what "
                                f"the project says it does not track")
    suffix = rel.suffix.lower()
    if suffix not in SOURCE_SUFFIXES:
        return (f"{suffix or 'its empty suffix'} is not a declared source "
                f"suffix ({sorted(SOURCE_SUFFIXES)}). The empty suffix is the "
                f"case that matters: a file with no extension is not source "
                f"unless its whole name is declared at the root")
    return None


def is_source(rel: Path, abs_dir: Path | None = None,
              honour_declaration: bool = True) -> bool:
    """True when this relative path is a source file of the release.

    `abs_dir` is the absolute directory the file sits in, and it is optional so
    that the table tests in `release_tree_parity_check.py` can ask about a path
    that does not exist. Passing it is what enables the `CACHEDIR.TAG` test, and
    every caller that is walking a real tree passes it.
    """
    return recognition_reason(rel, abs_dir, honour_declaration) is None


def prune_here(rel_dir: Path) -> bool:
    """True when *nothing* under this relative directory can be a source.

    A walk can skip a whole subtree, and the skip has to be derived from the
    same tables that decide membership -- a second list of "directories I do not
    bother to look in" is a second rule, and the two would drift exactly the way
    this file exists to prevent. The derivation is one line and it is monotone
    downwards, which is the only property a prune needs:

        a directory at or below the tree root is skipped exactly when its first
        component is not a declared `SOURCE_ROOT`.

    The tree root itself is never skipped, because `ROOT_SOURCE_NAME` lives
    there; `rel_dir.parts` is empty for it, so the test below is written to say
    that rather than to fall off the front of the tuple. The second clause is
    `GENERATED_DIRS`, matched at any depth rather than only at the first
    component: a root test alone ships `dock-py/.pytest_cache/README.md`, which
    is the measurement recorded at that constant. It is the same constant
    `recognition_reason` uses, so the walk and the predicate cannot disagree
    about it.

    Deliberately **not** used, because each of them would hide real sources:

    * a suffix-based prune would skip `build.exe/`, whose own contents are
      recognised by *their* suffixes;
    * a generated directory is pruned, but only because the same constant
      declines every file inside it. Those are one fact, not two, and the parity
      check compares this walk against an unpruned one to prove it.
    """
    if not rel_dir.parts:
        return False
    if rel_dir.parts[0] not in SOURCE_ROOTS:
        return True
    return any(part in GENERATED_DIRS for part in rel_dir.parts)


def iter_source_files(root: Path) -> dict[Path, Path]:
    """Every source file under `root`, as ``{relative path: absolute path}``.

    Pruned by `prune_here` so the walk does not descend into `target/` (about
    19 000 files on the maintainer's machine) to decide each one of them is not
    a source. The equivalence between this walk and an unpruned one is not
    assumed: `release_tree_parity_check.py` measures it.
    """
    root = Path(root)
    out: dict[Path, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        keep = []
        for name in sorted(dirnames):
            child = rel_dir / name if rel_dir.parts else Path(name)
            if not prune_here(child):
                keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            rel = rel_dir / name if rel_dir.parts else Path(name)
            if is_source(rel, Path(dirpath)):
                out[rel] = Path(dirpath) / name
    return out


def iter_package_copies(root: Path) -> list[tuple[Path, str]]:
    """Every second copy of the package under `root`, as ``(relative, reason)``.

    **This walk does not use `prune_here`, and that is the whole point of it.**
    A pruned walk prunes exactly the directories this question is about -- the
    one under `.term-overlay/opendocking/` is skipped *because* it is not under a
    declared source root -- so asking the pruned walk where the copies are returns
    an empty list for a tree that is full of them. That is not a subtlety: it is
    the check that looks for a second copy quietly becoming the wrong answer, and
    it would have reported a clean tree on the very commit that introduced the
    problem.

    So this traversal descends through everything and asks one question
    (`is_package_copy`). It has no skip list at all, which under the recognition
    rule is possible for the first time: the old version had to say "except the
    build-output directories", and that phrase was a second copy of a table that
    lived in another file.
    """
    root = Path(root)
    out: list[tuple[Path, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            rel = rel_dir / name if rel_dir.parts else Path(name)
            if is_package_copy(rel):
                out.append((rel, recognition_reason(rel) or ""))
    return out


def file_digest(path: Path) -> str:
    """sha256 of a file's bytes, read in chunks so a 748 KB `.pyd` or a large
    fixture does not have to be held in memory to be compared."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rule_fingerprint() -> str:
    """A digest of the decision surface: the tables, as the tool and the check
    both read them.

    One number that changes when -- and only when -- the rule changes, so a
    report can say which rule produced a file list instead of asking the reader
    to diff two modules by eye. Derived from the live objects, never written
    down: a constant beside the tables would be a third place to keep in step.
    """
    h = hashlib.sha256()
    for name in sorted(SOURCE_ROOTS):
        h.update(f"ROOT:{name}\n".encode("utf-8"))
    for name in sorted(GENERATED_DIRS):
        h.update(f"GENERATED:{name}\n".encode("utf-8"))
    for name in sorted(ROOT_SOURCE_NAMES):
        h.update(f"ROOTNAME:{name}\n".encode("utf-8"))
    for suffix in sorted(SOURCE_SUFFIXES):
        h.update(f"SUFFIX:{suffix}\n".encode("utf-8"))
    h.update(f"PKG:{CANONICAL_PACKAGE_PARENT.as_posix()}\n".encode("utf-8"))
    return h.hexdigest()
