"""Which `_dockpy.pyd` would Python load right now, and is it this tree's build?

Run:  F:\\python310\\python.exe scripts\\extension_install_parity_check.py

# The question, and why it is not "are the two files equal"

There are three copies of the compiled extension on a maintainer's machine and
they are three different objects: the development tree's build under
`dock-py/python/opendocking/`, the commit mirror's build under
`opendocking-gui/`, and the interpreter's `site-packages` install. Nothing in
this tree compared the first and the third, and the hole was open for a reason
that is structural rather than accidental: `PYTHONPATH` decides which of the
two `import` resolves, so **the same gate run exited 1 with `PYTHONPATH` set
and exited 0 without it**, on two different builds of one source. It was found
by a person reading a diff, and it was fixed by a person copying a file.

A gate that compared the two files unconditionally would have been wrong twice
over, and the first way is the one that matters. There are three legitimate
states, and only one of them has two copies in play:

* **A maintainer who has not installed the package.** No `site-packages` copy
  exists. That is the clone case, and it is a normal machine, not a defect.
* **A maintainer testing the in-tree build by setting `PYTHONPATH`.** Both
  copies are on the search path. They are *supposed* to be the same build, and
  the point of asking is to catch the case where they are not.
* **A maintainer testing the installed package.** `PYTHONPATH` is unset, the
  `site-packages` binary is what every `odgui` and `odcli` run loads, and the
  in-tree file is not in the question at all. Its being different from the
  winner is not a finding, because the user is not running it.

So the useful question is not "are these two files equal" but **"which binary
would Python load right now, and is that one the build of the Rust in this
tree?"** This file asks that, and it answers it by letting the interpreter do
the resolving rather than by deciding which file to look at.

# How the winner is established, and why it is asked twice

Two independent instruments, and a check that they agree:

1. **The interpreter's own resolver.** `importlib.util.find_spec("opendocking")`
   for the parent -- which does *not* execute `__init__.py`, because a
   top-level `find_spec` reads `sys.path` and stops -- and then
   `importlib.machinery.PathFinder.find_spec("opendocking._dockpy", ...)` for
   the child, over the parent's own search locations. That is the call CPython
   makes when the child is imported, and `spec.origin` is the file it would
   load. Importing the parent instead would execute `from .core import ...` in
   it, which loads the extension *and* the engine, so the measurement would
   change what it measures; that is why this goes through `PathFinder` and not
   `importlib.import_module`.
2. **The search path read as a list.** Every `<entry>/opendocking/_dockpy<sfx>`
   that exists, for `entry` in `sys.path` in order, using **every** suffix in
   `importlib.machinery.EXTENSION_SUFFIXES` rather than `[0]`. On CPython 3.10
   that list is `['.cp310-win_amd64.pyd', '.pyd']`; taking the first entry only
   is how the real extension gets rejected in favour of a file that does not
   exist. Within one directory the earliest matching suffix wins, which is
   `FileFinder`'s own order, so the per-directory break in the walk below is
   the loader's rule and not a simplification of it.

A `sys.path_hooks` entry, a zip import or anything else the enumeration does
not model would make the two disagree, and disagreement is a red rather than a
preference: the check is `the copy in play is the one the import order names`,
and it also requires the winner to be one of the copies this file measured,
which is what catches a fourth copy somewhere neither the tree nor the install
is.

**"In play" is defined by reachability, not by existence.** A copy is in play
when some `sys.path` entry can resolve it; everything else on the machine is
reported and not judged. That one definition is what makes all three states
above fall out of a single code path rather than needing three.

# What the winner is then held to

**The Rust in this tree.** `dock-py/src/lib.rs` beside the package is parsed by
`binary_source_parity_check.declared_surface` -- *that file's* function, not a
second parser here -- and every name it declares is looked for in the surface
the winner actually offers, measured by loading the winner in a child
interpreter. A winner that is behind its source is red with the names, which is
an `AttributeError` a user would hit rather than a digest a reader has to
interpret.

The comparison of the *copies in play to each other* is a different claim and
uses the same instrument: when two or more copies can be reached, they are the
same build by construction, so **they must export the same surface**. Their
digests are printed as corroboration and are deliberately not the verdict,
because this repository has already been bitten by byte-equality on a build:
`binary_source_parity_check.py`'s own docstring records that a rebuild of
unchanged source is not guaranteed byte-identical (the pyo3/numpy abi3 build,
the Rust toolchain, the linker, and `RUSTFLAGS: "-C target-cpu=x86-64"`, which
is `ci.yml:85` as measured on 2026-10-03 and not the `ci.yml:71` the sibling
file's docstring cites), so a byte verdict would be red on every correct rebuild
by a different toolchain. Two copies that differ in bytes and agree in surface
are reported as exactly that.

# The overlap, and what is therefore not repeated here

Read before writing, as required, and recorded here because a reader should not
have to re-derive it:

* `binary_source_parity_check.py` owns **the surface comparison**, and this
  file *imports and calls* it rather than restating it. It also owns the
  question "is this tree's binary the build of this tree's `lib.rs`", for a
  binary named by `OD_BINARY_SOURCE_PYD` / `OD_BINARY_SOURCE_ROOT`. This file
  derives no parser of its own, declares no second check count for that
  comparison, and does not re-run it as a subprocess: the two questions differ
  in *which file is the subject*, and the only thing added here is the subject.
  Where this file's winner **is** the tree's own in-tree binary, the parity
  check here and the one there are the same question asked of the same file,
  and that is said in the output rather than left for a reader to work out.
* `installed_copy_check.py` owns the *Python* surface of the installed copy
  (modules, intra-package imports, referenced names) and says in its own
  docstring that it asserts **nothing** about `_dockpy.pyd`, deliberately, so
  the two gates cannot contradict each other. Nothing here reads a `.py` file
  out of the installed copy, and nothing there reads a `.pyd`.
* `release_tree_rule.py`'s `SITE_PACKAGES_DECISION` puts the `site-packages`
  copy out of scope **for publication**: it is a local install artefact, it is
  not in the commit tree, and drift there is fixed by reinstalling rather than
  by syncing. That is a statement about what this repository ships, and it is
  not a statement about which binary `import` resolves. So this file does not
  contradict the decision, it reads and quotes it, and it asks the question the
  decision does not answer: when the `site-packages` binary is the one in play,
  is it the build of the Rust in this tree? It never publishes anything, never
  syncs anything, and never writes to any `.pyd`.
* `mirror_drift_check.py` and `release_tree_parity_check.py` own the commit
  mirror. This file **reports** the mirror's size and digest and gives it no
  verdict at all: the mirror is stale by design, a sync job owns it, and a red
  here would be a red about somebody else's work.

# The three states are one code path, and each can go red

| state | copies in play | what this file answers |
|---|---|---|
| both present, equal | 2 | the winner, why it won, agreement of the two instruments, agreement of the two surfaces, and the winner's parity with the tree's Rust |
| both present, different | 2 | as above, and **red** on the disagreement, naming both digests and the names that differ |
| in-tree only | 1 | the winner's parity with the tree's Rust, and a **counted skip** on the agreement claim, saying why one copy cannot disagree with itself |
| installed only | 1 | as above, with the winner being the installed binary and the in-tree file reported as present and out of play |

# A tree with no compiled extension is a SKIP, counted, never a pass

`.gitignore` carries `*_dockpy.pyd`, so git has never tracked one and a fresh
checkout has none; `binary_source_parity_check.py` states that rule and its
reason at length, and this file follows the precedent it sets. Nothing in play
means every measurement below records a skip **with its reason**, each counted
toward the pinned total and printed as a second number that is never added to
the pass count. A skip that is silently a pass is the worst outcome available
here, because it would be indistinguishable from "checked and current".

# What the exit code means, in each of the three states

`finish()` returns one number and `.github/workflows/ci.yml` runs this file as
a step whose outcome *is* that number, so what the number means is stated here
rather than left to be inferred from the code:

* **1** -- at least one check failed. That is the drift this file exists to
  catch, a copy that would not load, or a self-test that would not hold.
* **2** -- the run asserted nothing: no check passed, or it stopped early. It
  reads as a failure, which is what it is.
* **0** -- no check failed. **It does not say that every claim was measured.**
  A clone with no compiled extension measures what it can, records the rest as
  the counted skips the rule above requires, and exits 0. That is the same
  precedent `binary_source_parity_check.py` sets for the same state, and a
  counted skip is deliberately **not** turned into a failure here.

So the two exit-0 states are told apart by the run rather than by its reader.
A run that measured everything prints `RESULT: OK -- N passed and every one of
the M checks was measured`; a run that could not prints `RESULT: OK, NOT FULLY
MEASURED -- K of M checks could not be measured here`, which names the count in
the token itself. A CI step that reads the exit code alone therefore sees the
same 0 for both, and the line above it says which one it was: a clone runner
goes green on a verdict that states on its face that it measured part of the
question, and a reader who wants the stronger claim reads that line rather
than the tick.

# What this file does not do, stated here rather than discovered

* **It does not compare the winner's bytes to anything, and it does not assert
  that two builds of one source are byte-identical.** Two in-play copies that
  differ in bytes but agree in surface are reported as a possible same-source
  rebuild, not as a failure. See the rebuild note above.
* **It cannot see behaviour drift.** A scoring-weight change, a changed
  constant, a reordered loop: none of them move a name. Same limit, same
  price, as `binary_source_parity_check.py` states. `cargo test` owns that.
* **It does not reinstall anything, and it never writes a `.pyd` anywhere.**
  Drift in the installed copy is fixed by reinstalling; this file reports the
  condition and says so. `OD_EXT_PARITY_ROOT` points it at another tree's
  `lib.rs` and package, which is how the four states are exercised without
  moving a byte of the real ones.
* **It does not model `sys.path_hooks` or a zip import in the enumeration.**
  The first instrument is the interpreter's own `PathFinder`, so a winner
  inside one of those is still found; the second is the walk, and the check
  that the two agree is red when they do not rather than quietly preferring the
  model. The disagreement *is* the failure mode, so it is reported.
* **It does not model a package whose `__init__.py` extends `__path__`.** This
  is the one unmodelled case that is *worse* than the `sys.path_hooks` one
  above, and the difference is the whole reason it is stated rather than filed
  with it. A top-level `find_spec` does not execute `__init__.py`, so an
  initialiser that appends to `__path__` -- the `pkgutil.extend_path` idiom, or
  any assignment to it -- leaves **both** instruments reading the *unextended*
  path: the walk reads `sys.path`, and the resolver reads the parent's
  unextended `submodule_search_locations`. They would agree, and they would both
  be wrong, because the real import executes the initialiser first and then
  resolves `_dockpy` out of a directory neither of them looked at. The
  `sys.path_hooks` case fails loudly -- disagreement is what check 2 looks for
  -- while this one is a silent wrong winner, which is the outcome this whole
  file is arranged to prevent. This tree's own
  `dock-py/python/opendocking/__init__.py` does not do it: 1,803 bytes with no
  `__path__`, no `extend_path`, no `pkgutil` and no `sys.path` in it, measured
  on 2026-10-03, so nothing here is currently mis-resolved and the limitation
  is unmodelled rather than present. It is **stated, not modelled**: modelling
  it would mean either executing the initialiser -- which is the thing this
  file's first instrument deliberately avoids -- or reading a `.py` file of the
  package it measures, which this file's stated rule is that it never does.
* **It does not judge a copy that is out of play.** A skip and a red are not
  the same answer, and an out-of-play copy gets a digest and a sentence saying
  it was not the binary Python would load.
* **It writes two temp directories and no repository file.** A child-loader
  shim, and a two-directory fixture for the resolver self-test. Both are
  `tempfile.mkdtemp()` paths removed in a `finally`, because the sibling's
  `survey()` writes its shim beside the repository root and a run pointed at
  another tree has no business writing there.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIB_REL = Path("dock-py", "src", "lib.rs")
PKG_REL = Path("dock-py", "python", "opendocking")
PACKAGE = "opendocking"
PYD_STEM = "_dockpy"
PYD_NAME = PYD_STEM + ".pyd"

#: How many checks this file records, in every environment, and why a skip is a
#: result rather than an absence. Measured from a run in each of the four
#: states in the table above, not by arithmetic:
#:
#:   1  which copy Python would load right now, established from the
#:      interpreter's own search path (a skip when nothing is in play)
#:   2  the copy in play is the one the import order names, and it is one of
#:      the copies this file measured
#:   3  every copy of the extension on this machine is reported, in play or not
#:   4  every copy in play loads as a compiled extension
#:   5  the copy in play is the build of the Rust in this tree
#:   6  the parity comparison is not vacuous: an absent name is caught
#:   7  the resolution is not vacuous: a planted two-directory fixture
#:      resolves to the first candidate
#:   8  a state that could not be measured is recorded as a skip, never as a pass
#:   9  every copy that could be in play exports the same surface
#:  10  a drift is a verdict: both drift checks recorded one, each is the
#:      result its own inputs imply, and a drifted state is a FAIL
#:  11  every skip this run recorded is one the state forced, and no result
#:      that could be measured was skipped
#:  12  the tally accounts for every result, skips included
#:
#: Checks 6 to 8 and 10 to 11 exist because checks 1 to 5 and 9 are instruments
#: that have never been seen refuse anything, and a predicate nobody has
#: watched disagree is not evidence. 6 and 7 call what the real checks call --
#: 6 calls the same `parity_gap()` check 5 calls, and 7 calls the same two
#: resolvers checks 1 and 2 use -- on a synthetic input, so a gate testing a
#: copy of its own logic would pass while the real one was broken. 8 asks the
#: recorder itself, which is where the rule "a tree with no compiled extension
#: is a counted skip, never a pass" actually lives; without it that rule is a
#: sentence in a docstring. **10 is the one that was missing, and an audit named
#: it**: 6 and 7 ask whether a *predicate* can refuse, nothing asked whether a
#: *drift* is a verdict at the sites that record one, and the two checks that
#: can catch it -- 5 and 9 -- are mutually redundant, so disabling either one
#: alone still exits 1. An audit that disabled both was answered
#: `11 passed, 0 failed`, exit 0, with the stale build winning, which is this
#: file's own self-tests unable to notice losing the thing they exist for. So
#: 10 asks both drift checks three questions each -- did they record a result,
#: is that result the one their own inputs imply, and is a drifted state a FAIL
#: through the same comparison -- and a deleted, inverted or force-passed arm
#: now costs a red instead of nothing. 11 then audits this run's own skips
#: against the state it measured, in both directions: a check that skipped
#: while it still had a subject is this gate quietly degrading into the
#: unconditional two-file comparison it exists to replace, and the tally cannot
#: see that because a skip is a result. None of the five touches a real `.pyd`:
#: 6's absent name is synthetic, 7's fixture is two empty files `PathFinder`
#: names and nothing ever opens, 10's two drifted states are name sets held in
#: memory while its re-derivations read this run's own `RESULTS` and the
#: surfaces already surveyed, and 8 and 11 pass string literals and that same
#: `RESULTS`.
#:
#: The call-site census, declared in the file it counts rather than in
#: `check_scripts_declare.py`'s generated snapshot, and measured by walking this
#: file's own syntax tree with that tool's own walk. A pin is how many results a
#: run records; the census is how many result sites the source has. They are
#: different questions, and the guard digest is here so that a site moving
#: between guard shapes without changing the total stays visible.
#:
#: **0 unconditional, and that is the shape of this gate rather than an
#: accident of how it was written.** Every other gate here has sites that run
#: whatever the machine looks like; this one has none, because every claim it
#: makes is a claim *about* the state -- which copy is in play, whether it
#: loads, whether the two agree, whether the winner is the tree's build -- and
#: each of those is written as an `if`/`else` whose two arms are the two
#: verdicts. A site that could run without knowing whether there was a binary
#: to run against would be a claim this file has no business making, and the
#: count is the machine-checkable form of that sentence: the census cannot see a
#: site that is unguarded here, because there is none to see.
#:
#: The first version of this block was written with a bare `#` and the reader
#: reports such a block as `NOT DECLARED ANYWHERE`, which is true and useless.
#: The `#:` prefix is what `check_scripts_declare.py` matches on, and it was
#: caught by transcribing that file's own reader rather than by running it.
#: GATE-DECLARE 1
#: sites: 0 unconditional + 29 guarded
#: guards: sha256:64692765a0aba2b15888754558808af0f184c6b0185a1cd2ebeaf7de7dfdb955
EXPECTED_CHECKS = 12

RESULTS: list[tuple[str, str, str]] = []

#: The nine names, held here so that the two call sites of a state-dependent
#: check cannot drift apart in the string. `binary_source_parity_check.py` keeps
#: two of its names as constants for the same reason.
WINNER_CHECK = ("which copy Python would load right now, established from the "
                "interpreter's own search path")
AGREEMENT_OF_RESOLVERS_CHECK = ("the copy in play is the one the import order "
                                "names, and it is one of the copies this file "
                                "measured")
INVENTORY_CHECK = ("every copy of the extension on this machine is reported, in "
                   "play or not")
LOADED_CHECK = "every copy in play loads as a compiled extension"
PARITY_CHECK = "the copy in play is the build of the Rust in this tree"
VACUOUS_PARITY_CHECK = ("the parity comparison is not vacuous: an absent name is "
                        "caught")
VACUOUS_RESOLUTION_CHECK = ("the resolution is not vacuous: a planted "
                            "two-directory fixture resolves to the first candidate")
SKIP_DISCIPLINE_CHECK = ("a state that could not be measured is recorded as a "
                         "skip, never as a pass")
SKIP_AUDIT_CHECK = ("every skip this run recorded is one the state forced, and "
                    "no result that could be measured was skipped")
AGREEMENT_CHECK = "every copy that could be in play exports the same surface"
DRIFT_VERDICT_CHECK = ("a drift is a verdict: both drift checks recorded one, each "
                       "is the result its own inputs imply, and a drifted state "
                       "is a FAIL through the comparison that records it")


def verdict_of(ok: bool, detail: str, skip: str | None) -> tuple[str, str]:
    """`("PASS"|"FAIL"|"SKIP", text)` for one result. Pure: no side effect.

    Split out of `check()` so that the rule "a state that cannot be measured is
    a skip and never a pass" is a *function* with an oracle rather than a branch
    inside the recorder. The self-test at the end of `main` calls this three
    times with a synthetic argument and requires all three answers, including
    the one that says a measurable pass is still a pass -- a `skip` test that
    won unconditionally would silence every check in this file while passing a
    one-sided test of its own.
    """
    if skip == "":
        raise ValueError("an empty skip reason is not a reason")
    if skip is not None:
        return "SKIP", skip
    return ("PASS" if ok else "FAIL"), detail


def check(name: str, ok: bool, detail: str = "", skip: str | None = None) -> None:
    """Record one result. `skip` is the reason this run could not measure it.

    Three outcomes, and `skip` beats `ok` on purpose: a site that would have
    passed vacuously because there was nothing to measure must not be able to
    say so. An empty `skip` is refused rather than treated as absent, because a
    caller that built its reason and got an empty one has a bug this would
    otherwise hide behind a PASS.
    """
    tag, text = verdict_of(ok, detail, skip)
    RESULTS.append((tag, name, text))
    print(f"[{tag}] {name}\n       {text}")


def section(t: str) -> None:
    print()
    print(f"-- {t}")


# ----------------------------------------------------------------- siblings

def _sibling(name: str):
    """A sibling script as a module, imported by path and without `sys.path`.

    `sys.path` is the *subject* of this file, so appending to it to make an
    import work would be a measurement changing the environment it measures. A
    script run already has `scripts/` at `sys.path[0]`, so the ordinary import
    works; the guarantee should not rest on how the file was invoked. Returns
    None when the file is not beside this one, so a missing sibling is a
    counted skip rather than a traceback.
    """
    path = HERE / name
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_od_extparity_" + name[:-3], path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The gate that owns the surface comparison, and the rule module that owns what
#: this repository ships. Both are read, not restated: the first because a
#: second parser of `lib.rs` would be a second thing to be wrong about one
#: file, the second because the commit mirror's directory name is a decision
#: with a reason attached and typing it here would fork that decision.
BSP = _sibling("binary_source_parity_check.py")
RULE = _sibling("release_tree_rule.py")
MIRROR_DIRNAME = getattr(RULE, "RELEASE_TREE_DIRNAME", "opendocking-gui")
DECISION = getattr(RULE, "SITE_PACKAGES_DECISION", "")


# -------------------------------------------------------------- measurement

def measure(path: Path) -> tuple[int, str] | None:
    """`(size, sha256)` of a file, or None when it is not there.

    One reader for every number this file prints, so a digest in the inventory
    and a digest in a verdict cannot disagree: they are the same call. An absent
    copy is `None` and is reported as absent, never as a zero.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return len(data), hashlib.sha256(data).hexdigest()


def brief(path: Path) -> str:
    """`779,264 B sha256 839b50736db6347e`, or `absent`. One format everywhere."""
    got = measure(path)
    if got is None:
        return "absent"
    size, digest = got
    return f"{size:,} B sha256 {digest[:16]}"


# ---------------------------------------------------------------- the winner

def ranked_candidates(search_path: list[str]) -> list[Path]:
    """Every `_dockpy.pyd` reachable through `search_path`, in search order.

    The second instrument, and the one that can be reasoned about from the
    output. Two details are load-bearing:

    * **Every** suffix in `EXTENSION_SUFFIXES`, not `[0]`. On CPython 3.10 that
      list is `['.cp310-win_amd64.pyd', '.pyd']` and the real extension is
      `_dockpy.pyd`; the ABI-tagged entry first would report nothing in play
      and the run would announce the clone case on a machine that has a build.
    * **The first matching suffix within one directory wins, then the walk
      moves to the next entry.** That is `FileFinder`'s own order -- loaders
      outer, suffixes inner -- so two files in one directory resolving to one
      candidate is the loader's rule and not a simplification of it.

    An empty entry is the interpreter's current directory and it is skipped --
    but **not** because it cannot name a file here, which is what this sentence
    said first and which is false. With `''` at `sys.path[0]` and the working
    directory at `.../dock-py/python` it does name the in-tree binary:
    `Path("")/"opendocking"/"_dockpy.pyd"` is that file, measured on 2026-10-03
    as 779,264 B sha256 839b5073. The reason it is dropped is that the two
    instruments would then hold *different objects*: the walk would return a
    relative path and `spec.origin` an absolute one, so check 2 would go red on
    a bookkeeping difference rather than on a resolution disagreement, and the
    reported path would begin with a bare `''` that a reader cannot open. The
    drop is silent only here. **The observed behaviour is a red on check 2, not
    a silent pass**: the interpreter's own resolver still names the in-tree file
    and the walk does not, and that disagreement is precisely what check 2
    exists to report. An empty `PYTHONPATH=";"` entry is not a way to get here:
    CPython splits `PYTHONPATH` on `;` and drops the empty components, measured
    as zero empty entries in `sys.path` under `PYTHONPATH=";"`, so an empty
    entry comes from `-c`, `-m` or an interactive start rather than from the
    environment variable this file is mostly about.
    """
    out: list[Path] = []
    for entry in search_path:
        if not entry:
            continue
        base = Path(entry) / PACKAGE
        if not base.is_dir():
            continue
        for sfx in importlib.machinery.EXTENSION_SUFFIXES:
            cand = base / (PYD_STEM + sfx)
            if cand.is_file():
                out.append(cand)
                break
    return out


def interpreter_winner() -> tuple[Path | None, str]:
    """`(path, how)` for the file CPython itself resolves, from two calls.

    The parent is located with `importlib.util.find_spec("opendocking")`, which
    for a top-level name reads `sys.path` and **does not execute** the package's
    `__init__.py`; the child is then located with `PathFinder.find_spec` over
    that parent's own search locations, which is the call the import system
    makes once the parent is in hand. Importing the parent instead would
    execute `from .core import ...` -- which loads the extension and the engine
    -- so the measurement would be the thing that changed the machine.
    """
    try:
        parent = importlib.util.find_spec(PACKAGE)
    except (ImportError, ValueError) as exc:
        return None, (f"locating the {PACKAGE} package raised "
                      f"{type(exc).__name__}: {exc}")
    if parent is None:
        return None, f"no {PACKAGE} package is on sys.path"
    locations = [str(p) for p in (parent.submodule_search_locations or ())]
    child = importlib.machinery.PathFinder.find_spec(f"{PACKAGE}.{PYD_STEM}",
                                                     locations)
    if child is None or not child.origin:
        how = (f"the {PACKAGE} package resolves to "
               f"{', '.join(locations) or '(no search location)'}, and it holds "
               f"no {PYD_STEM} file in any of "
               f"{list(importlib.machinery.EXTENSION_SUFFIXES)}")
        return None, how
    return (Path(child.origin),
            f"the package is at {', '.join(locations)} and the child resolves to "
            f"{child.origin} through {type(child.loader).__name__}")


def labels_for(path: Path, ranked: list[Path], in_tree: Path,
               mirror: Path) -> str:
    """Every relation this file can prove about one path, joined.

    A path can be two of them at once -- the in-tree binary is `in play` when
    `PYTHONPATH` points at `dock-py/python` -- so the labels are joined rather
    than chosen between. `site-packages` is a **path property**, the shape
    `release_tree_rule.py` uses to prove its boundary is a path property rather
    than a promise in a comment, and not a claim about where a copy came from.
    """
    out: list[str] = []
    if path in ranked:
        out.append(f"IN PLAY, rank {ranked.index(path) + 1} of {len(ranked)}")
    if path == in_tree:
        out.append("the development tree's own build")
    if path == mirror:
        out.append("the commit mirror's build: stale by design, owned by a "
                   "sync job, reported here and given no verdict")
    if "site-packages" in path.parts:
        out.append("under site-packages, so an install artefact by the path "
                   "property release_tree_rule.py uses")
    if not out:
        out.append("a copy on sys.path that is neither this tree's nor the "
                   "mirror's")
    return "; ".join(out)


# ---------------------------------------------------------------- the surface

_CHILD = r'''
import importlib.machinery, importlib.util, json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
out = {"ok": False}
try:
    path = sys.argv[1]
    # Any suffix in EXTENSION_SUFFIXES, not [0]: on CPython 3.10 that list is
    # ['.cp310-win_amd64.pyd', '.pyd'] and the file under test is _dockpy.pyd.
    if not any(path.endswith(sfx) for sfx in importlib.machinery.EXTENSION_SUFFIXES):
        out["error"] = "not a compiled extension: %s" % path
    else:
        spec = importlib.util.spec_from_file_location("opendocking._dockpy", path)
        if spec is None or not isinstance(spec.loader,
                                          importlib.machinery.ExtensionFileLoader):
            out["error"] = ("no extension loader for %s (got %r)"
                            % (path, type(spec.loader).__name__ if spec else None))
        else:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["opendocking._dockpy"] = mod
            spec.loader.exec_module(mod)
            out["module"] = sorted(n for n in dir(mod) if not n.startswith("__"))
            classes = {}
            for n in out["module"]:
                obj = getattr(mod, n)
                if isinstance(obj, type):
                    classes[n] = sorted(a for a in dir(obj) if not a.startswith("__"))
            out["classes"] = classes
            out["ok"] = True
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print("__SURFACE__" + json.dumps(out))
'''


def survey(pyd: Path) -> dict:
    """Load one `.pyd` by path in a child and report what it offers.

    A child, for the reason `binary_source_parity_check.py` gives: a binary
    that does not match this interpreter can abort the process that loads it,
    and this file has to report that rather than become it.

    The one piece of that sibling this file does **not** call is its child
    shim, and the reason is a path rather than a disagreement:
    `binary_source_parity_check.survey()` writes its shim to
    `ROOT/.binary_source_parity_child.py`, so a run of *this* file pointed at
    another tree would put a file in *this* repository that the run has no
    business writing. The shim is therefore written to a temp directory here and
    the directory is removed in a `finally`. The loader is the same fifteen
    lines; the analysis consuming what it prints is imported from there, not
    reimplemented.
    """
    tmp = Path(tempfile.mkdtemp(prefix="od-extparity-"))
    (tmp / "child.py").write_text(_CHILD, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        proc = subprocess.run([sys.executable, str(tmp / "child.py"), str(pyd)],
                              capture_output=True, text=True, timeout=300, env=env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("__SURFACE__")), None)
    if line is None:
        return {"ok": False, "error": (
            f"the child printed no surface report and exited {proc.returncode}; "
            f"stderr: {((proc.stderr or '').strip()[-300:]) or '(empty)'}")}
    return json.loads(line[len("__SURFACE__"):])


def names_of(data: dict) -> set[str]:
    """Every Python-visible name one survey reported, module and class level."""
    out = set(data.get("module", ()))
    for members in data.get("classes", {}).values():
        out |= set(members)
    return out


def parity_gap(declared: dict[str, int], offered: set[str]) -> list[str]:
    """Names the source declares that the binary does not offer. One predicate.

    Both the parity check and its self-test call **this**, so the self-test
    exercises the code that produces the verdict rather than a restatement of
    it. A test that called `binary_source_parity_check.missing_members`
    directly while the real check called something else would pass with the
    real check broken, which is the whole failure mode a self-test exists to
    close. The comparison itself is still that file's, and is not restated
    here.
    """
    return BSP.missing_members(declared, offered)


def disagreeing_copies(copies: list[Path], loaded: dict) -> list[str]:
    """One sentence per adjacent pair of reachable copies whose surfaces differ.

    The surface comparison the agreement check records its verdict from, as a
    callable, for exactly the reason `parity_gap()` is one: the drift
    self-test has to ask the code that produces the verdict rather than a
    restatement of it, or it would pass while the check that uses it was
    broken. The verdict stays at the call site -- this says *whether* two
    reachable copies disagree and why, and the agreement check is what says
    that a disagreement is a red.
    """
    diffs: list[str] = []
    for a, b in zip(copies, copies[1:]):
        na, nb = names_of(loaded[a]), names_of(loaded[b])
        if na != nb:
            diffs.append(
                f"{a} exports {sorted(na - nb)[:8]} that {b} does not, and "
                f"{b} exports {sorted(nb - na)[:8]} that {a} does not")
    return diffs


def declared_at(name: str, funcs: dict[str, int],
                members: dict[str, tuple[str, int]]) -> str:
    """`line 1288, a member of GridMaps` for a name the source declares.

    The line number is the part a reader has to go and look at, so it has to be
    the real one: the first version of this file flattened the declared names
    into one dict with `0` for every class member, and the finding it printed
    for a missing member read `poses_outside_box_count (line 0)`. A line number
    of 0 is not a place in the file.
    """
    if name in funcs:
        return f"line {funcs[name]}, a module-level function"
    owner, line = members.get(name, ("?", 0))
    return f"line {line}, a member of {owner}"


def plant_two_copies() -> tuple[Path, Path, Path, list[str]] | None:
    """Two empty extension-named files in two temp package directories.

    A resolver self-test needs two candidates to be a test of order, and the
    cheapest honest pair is two files `PathFinder` will name and that nothing
    will ever open: `find_spec` matches the file *name* against the loader
    suffixes and does not read a byte of it. Each directory is given the
    ordinary `<dir>/opendocking/__init__.py` shape so that the function under
    test does the finding, because a directory with no initialiser is aggregated
    across every `sys.path` entry as a namespace portion -- a different
    resolution problem, which would test the fixture rather than the resolver.

    **The two directories are named `z` and `a`, in that order, and the reason
    is a mutation.** The first version of this fixture was `a` and `b`, which
    is already alphabetical, so a `ranked_candidates` that returned
    `sorted(out)` -- destroying the search order this whole file turns on --
    produced the same list and the self-test stayed green. A fixture whose
    order coincides with the alphabetical order cannot test order. Here the
    alphabetical order is the *reverse* of the search order, so a walk that
    ignores order returns the two the other way round and the check fires.

    Returns `(first, second, base, search_path)`, and the caller removes `base`:
    the fixture is a temp directory and the run is over with it.
    """
    base = Path(tempfile.mkdtemp(prefix="od-extparity-fix-"))
    made: list[Path] = []
    search: list[str] = []
    for letter in ("z", "a"):
        pkg = base / letter / PACKAGE
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="ascii", newline="\n")
        (pkg / PYD_NAME).write_bytes(b"")
        made.append(pkg / PYD_NAME)
        search.append(str(base / letter))
    return made[0], made[1], base, search


def resolve_in(search_path: list[str]) -> tuple[Path | None, str]:
    """`interpreter_winner()` against an explicit search path, for the fixture.

    The same two calls with `sys.path` swapped for the fixture's own list, and
    restored in the `finally`: the mutation this check exists for is the
    resolver ignoring the order of the list it was handed, and a test asserting
    against the real `sys.path` could not tell an ordered resolver from an
    unordered one.
    """
    saved = sys.path
    try:
        sys.path = list(search_path)
        return interpreter_winner()
    finally:
        sys.path = saved


# ----------------------------------------------------------------- the checks

def main() -> int:
    print("Which `_dockpy.pyd` would Python load right now, and is it this "
          "tree's build?")

    root = Path(os.environ.get("OD_EXT_PARITY_ROOT", ROOT)).resolve()
    lib = root / LIB_REL
    in_tree = root / PKG_REL / PYD_NAME
    mirror = root / MIRROR_DIRNAME / PKG_REL / PYD_NAME
    pythonpath = os.environ.get("PYTHONPATH", "")
    print(f"  tree        {root}")
    print(f"  source      {lib}")
    print(f"  in-tree     {brief(in_tree)}")
    print(f"  mirror      {brief(mirror)}")
    print(f"  PYTHONPATH  {pythonpath if pythonpath else 'is not set'}")
    print("  every number below was measured on this machine as it stands right "
          "now; none of it is carried over from a previous run")

    ranked = ranked_candidates(sys.path)
    winner, how = interpreter_winner()
    first = ranked[0] if ranked else None

    # ---- 1, which copy is in play and which one wins
    section("which copy is in play, and which one wins")
    order = (f"sys.path holds {len(sys.path)} entries and {len(ranked)} of them "
             f"resolve to a {PYD_NAME}: "
             + ("; ".join(f"{i + 1}) {p}" for i, p in enumerate(ranked))
                or "none")
             + f". The interpreter's own resolver says: {how}")
    if winner is None:
        check(WINNER_CHECK, False, "",
              skip=f"no compiled extension is in play: {how}. This is the clone "
                   f"case, and it is a skip rather than a pass: `.gitignore` "
                   f"carries `*_dockpy.pyd`, so no published tree has one, and "
                   f"until this tree is built there is no binary here to hold to "
                   f"the Rust beside it")
    else:
        beaten = [p for p in ranked if p != winner]
        check(WINNER_CHECK, True,
              f"Python would load {winner}. {order}"
              + (f". It beats {len(beaten)} other copy/copies that are also in "
                 f"play and shadowed by it: "
                 + "; ".join(f"{p} ({brief(p)})" for p in beaten)
                 + ". Both are in play, so both are held to the Rust below"
                 if beaten else
                 ". It is the only copy on the search path, so nothing here is "
                 "shadowed by it and no second copy can disagree with it"))

    # ---- 2, the two instruments agree, and the winner is a measured copy
    measured = list(dict.fromkeys(ranked + [in_tree, mirror]))
    if winner is None:
        check(AGREEMENT_OF_RESOLVERS_CHECK, False, "",
              skip="nothing is in play, so there is no winner for the two "
                   "instruments to agree about")
    elif first is None or winner != first:
        check(AGREEMENT_OF_RESOLVERS_CHECK, False,
              f"the interpreter's resolver names {winner}, while walking "
              f"sys.path in order found "
              + (", ".join(str(p) for p in ranked) if ranked else "no copy at all")
              + ". The two derivations disagree, so this file does not know "
                "which file is in play. That is the loud direction: a "
                "sys.path_hooks entry, a zip import or a .pth this walk does "
                "not model, rather than a preference between two right answers")
    else:
        check(AGREEMENT_OF_RESOLVERS_CHECK, True,
              f"both derivations name {winner}: the resolver's `spec.origin` "
              f"and the first copy found walking sys.path in order. It is one "
              f"of the {len(measured)} file(s) enumerated from the search path, "
              f"so it is not a fourth copy the walk could not see")

    # ---- 3, every copy on this machine is reported
    section("every copy this machine has, in play or not")
    lines = []
    for p in measured:
        lines.append(f"{brief(p):>34}  {p}  [{labels_for(p, ranked, in_tree, mirror)}]")
    reported = all(str(p) in line for p, line in zip(measured, lines))
    if len(measured) == len(lines) and reported:
        check(INVENTORY_CHECK, True,
              "\n       ".join(lines)
              + f"\n       in play under the environment as it stands: "
                f"{len(ranked)} of the {len(measured)} above. The rest are "
                f"reported with a digest and no verdict, because a copy Python "
                f"would not load is not this gate's subject. `release_tree_rule.py` puts the "
                f"site-packages copy out of scope for publication -- "
              + (DECISION.split(" It ")[0] if DECISION else
                 "that rule module is not beside this file, so the decision is "
                 "not quotable here")
              + " -- and this file asks the question that decision does not "
                "answer, which is which copy `import` reaches")
    else:
        check(INVENTORY_CHECK, False,
              f"{len(measured)} copy/copies were measured and {len(lines)} were "
              f"reported, so the inventory below is incomplete and a reader "
              f"cannot tell which copy answered")

    # ---- 4, every copy in play loads
    section("does every copy in play load")
    loaded = {p: survey(p) for p in ranked}
    broken = [(p, loaded[p].get("error")) for p in ranked if not loaded[p].get("ok")]
    if not ranked:
        check(LOADED_CHECK, False, "",
              skip="nothing is in play, so there is no binary to load")
    elif broken:
        check(LOADED_CHECK, False,
              f"{len(broken)} of {len(ranked)} copy/copies in play did not load "
              f"as a compiled extension: "
              + "; ".join(f"{p}: {err}" for p, err in broken)
              + ". A copy Python would import before any other is a runtime "
                "failure, and it is not the same finding as a stale build")
    else:
        check(LOADED_CHECK, True,
              f"all {len(loaded)} copy/copies in play loaded by path in a child "
              f"interpreter, offering "
              + "; ".join(f"{'/'.join(p.parts[-3:])} {len(names_of(d))} name(s)"
                          for p, d in loaded.items())
              + ". Loading is by path, so a copy that could never be imported is "
                "still measured rather than skipped")

    # ---- 5, the copy in play against the Rust in this tree
    section("is the copy in play the build of the Rust in this tree")
    declared: dict[str, int] = {}
    nfuncs = 0
    parsed = False
    if BSP is None:
        parity_skip = ("binary_source_parity_check.py is not beside this file and "
                       "it owns the declared-surface parser. This file calls it "
                       "rather than carrying a second parser of lib.rs, so "
                       "without it there is nothing here to compare")
    elif not lib.is_file():
        parity_skip = (f"{lib} does not exist, so there is no declared surface "
                       f"to hold the binary to. A tree that does not carry "
                       f"dock-py/src/lib.rs is not a tree this gate can judge")
    elif winner is None:
        parity_skip = ("no compiled extension is in play, so whether the binary "
                       "Python would load is the build of the Rust in this tree "
                       "cannot be asked here")
    elif not loaded.get(winner, {}).get("ok"):
        parity_skip = (f"the copy in play, {winner}, did not load; the check "
                       f"above reports that, and there is no surface here to "
                       f"compare")
    else:
        text = lib.read_text(encoding="utf-8", errors="replace")
        funcs, classes = BSP.declared_surface(text)
        nfuncs = len(funcs)
        members = {m: (cname, ln) for cname, v in classes.items()
                   for m, ln in v.items()}
        declared = {**funcs, **{m: 0 for v in classes.values() for m in v}}
        parsed = bool(declared)
        parity_skip = ("" if parsed else
                       f"{lib} parsed to an empty surface over "
                       f"{len(text.splitlines())} lines, which is a parser that "
                       f"stopped understanding the file rather than a build "
                       f"with nothing in it")
    if parity_skip:
        check(PARITY_CHECK, False, "", skip=parity_skip)
    else:
        missing = parity_gap(declared, names_of(loaded[winner]))
        if missing:
            check(PARITY_CHECK, False,
                  f"{len(missing)} name(s) {lib.name} declares and the copy in "
                  f"play does not offer: "
                  + ", ".join(f"{n} ({declared_at(n, funcs, members)})"
                              for n in missing)
                  + f". {winner} ({brief(winner)}) is behind the source beside "
                    f"it, and every call to one of those is an AttributeError at "
                    f"the call site")
        else:
            check(PARITY_CHECK, True,
                  f"{winner} ({brief(winner)}) offers every one of the "
                  f"{len(declared)} Python-visible name(s) {lib.name} declares "
                  f"-- {nfuncs} module-level function(s) and "
                  f"{len(declared) - nfuncs} class member(s) -- so the binary "
                  f"Python would load is a build of the Rust in this tree"
                  + (f". This is the question binary_source_parity_check.py asks "
                     f"of a tree's own binary, asked here of the binary the "
                     f"import system resolves; its OD_BINARY_SOURCE_PYD is how "
                     f"it is pointed at a file, and this run did not have to "
                     f"decide that"
                     if winner == in_tree else
                     f". binary_source_parity_check.py does not answer this one: "
                     f"it measures a tree's binary against the lib.rs in that "
                     f"tree, and this winner is a copy outside the tree whose "
                     f"lib.rs is {lib}"))

    # ---- 6, the self-test of the comparison
    if BSP is None:
        check(VACUOUS_PARITY_CHECK, False, "",
              skip="the comparison this check fires is the one the parity check "
                   "above calls, and that sibling is not beside this file, so "
                   "there was nothing to fire")
    else:
        sentinel = "od_extparity_deliberately_not_exported_name"
        offered = set() if winner is None else names_of(loaded.get(winner, {}))
        fired = parity_gap({**declared, sentinel: 0}, offered)
        others = [n for n in fired if n != sentinel]
        check(VACUOUS_PARITY_CHECK, sentinel in fired,
              "the same missing_members() the parity check above calls, asked "
              "about a source declaring one name no binary has, reports it"
              + (f", alongside {len(others)} genuinely absent name(s) this run "
                 f"found: {', '.join(others)} -- those are findings, not a "
                 f"failure of this check" if others else "")
              + ". The absent name is synthetic and no real .pyd is touched")

    # ---- 7, the self-test of the resolver
    planted = plant_two_copies()
    if planted is None:
        check(VACUOUS_RESOLUTION_CHECK, False, "",
              skip="the two-directory fixture could not be built, so this run "
                   "has not watched the resolver refuse anything")
    else:
        firstp, secondp, base, search = planted
        try:
            picked, _why = resolve_in(search)
            walked = ranked_candidates(search)
            # Both liveness facts are read *before* the fixture is removed. The
            # first version of this check asked `secondp.exists()` in its
            # condition, which is evaluated after the `finally` had already
            # taken the directory away -- so the check reported a fixture it
            # had itself deleted, and it went red on a resolver that had
            # answered correctly.
            planted_both = firstp.is_file() and secondp.is_file()
        finally:
            shutil.rmtree(base, ignore_errors=True)
        # **Both instruments, not just the interpreter's.** The first version
        # asked only `interpreter_winner()`, and a mutation that sorted the
        # enumeration's output -- destroying the search order the whole file
        # turns on -- left this check green while check 2 went red. The fixture
        # is the only place both can be asked about order at once.
        check(VACUOUS_RESOLUTION_CHECK,
              picked == firstp and walked == [firstp, secondp] and planted_both
              and firstp != secondp,
              f"a temp tree holding two empty {PYD_NAME} files, at {firstp} and "
              f"{secondp}, with the first directory earlier on the search path, "
              f"resolves to {picked} and the enumeration returns "
              f"{[str(p) for p in walked]}. That is the arm PYTHONPATH decides: the "
              f"first reachable copy wins and the second is shadowed, which is "
              f"why 'are the two files equal' cannot be the question. Neither "
              f"file is opened -- PathFinder matches the name and does not read "
              f"it -- and both are the fixture's, never a real one")

    # ---- 8, the self-test of the recorder, which is what makes a skip a skip
    answers = {
        "a measurable pass": verdict_of(True, "d", None),
        "a measurable failure": verdict_of(False, "d", None),
        "an unmeasurable state asked to pass": verdict_of(True, "d", "r"),
        "an unmeasurable state asked to fail": verdict_of(False, "d", "r"),
    }
    want = {
        "a measurable pass": "PASS",
        "a measurable failure": "FAIL",
        "an unmeasurable state asked to pass": "SKIP",
        "an unmeasurable state asked to fail": "SKIP",
    }
    wrong = {k: v[0] for k, v in answers.items() if v[0] != want[k]}
    if wrong:
        check(SKIP_DISCIPLINE_CHECK, False,
              f"the recorder answers {wrong} where the rule is {want}. A skip "
              f"that lost to a pass would turn a tree with no compiled "
              f"extension into a green run that had measured nothing, and a skip "
              f"that won unconditionally would silence every check in this file")
    else:
        check(SKIP_DISCIPLINE_CHECK, True,
              "asked about a state it could not measure, the recorder returns "
              "SKIP whether the site believes it passed or failed, and it still "
              "returns PASS and FAIL for the states it can measure. So a tree "
              "with no compiled extension records skips and never passes, and a "
              "check that lost its subject is not counted as a pass either")

    # ---- 9, the copies in play against each other
    # `several` is computed here rather than at the self-test and the audit
    # below because all three of them need it, and the audit must come *after*
    # this check: an audit that ran first would be auditing a run that had not
    # recorded its last result yet.
    several = [p for p in ranked if loaded.get(p, {}).get("ok")]
    section("do the copies in play agree with each other")
    if len(several) < 2:
        strays = [p for p in (in_tree, mirror) if p.is_file() and p not in ranked]
        check(AGREEMENT_CHECK, False, "",
              skip=f"only {len(several)} copy/copies of the extension can be "
                   f"reached under this environment, so there is no second copy "
                   f"that could disagree with the one in play"
                   + (f". {len(strays)} further copy/copies exist on this "
                      f"machine and are not in play "
                      f"({', '.join(str(p) for p in strays)}), and comparing one "
                      f"of those to the winner would be comparing two files of "
                      f"which one is not the binary Python would load -- which "
                      f"is the noise this gate exists to avoid" if strays else ""))
    else:
        pairs = list(zip(several, several[1:]))
        diffs = disagreeing_copies(several, loaded)
        if diffs:
            check(AGREEMENT_CHECK, False,
                  f"{len(several)} copies are reachable and they do not agree: "
                  + "; ".join(diffs)
                  + f". The two are the same build by construction, and a "
                    f"maintainer who sets PYTHONPATH is choosing between them "
                    f"deliberately. This is the drift that was found by hand, "
                    f"and it is invisible to every gate that looks at one tree")
        else:
            same_bytes = all(measure(a) == measure(b) for a, b in pairs)
            check(AGREEMENT_CHECK, True,
                  f"all {len(several)} reachable copies export the same "
                  f"{len(names_of(loaded[several[0]]))} Python-visible name(s), "
                  f"and their bytes are "
                  + ("identical, so the two files a maintainer can reach are one "
                     "build" if same_bytes else
                     "not identical ("
                     + "; ".join(f"{p.name} {brief(p)}" for p in several)
                     + "). Equal bytes are not this claim and are not required "
                       "by it: a rebuild of unchanged source is not guaranteed "
                       "byte-identical (the pyo3/numpy abi3 build, the toolchain, "
                       "the linker and RUSTFLAGS target-cpu all feed it), so a "
                       "byte verdict here would be red on every correct rebuild. "
                       "The surface is the claim")
                  )

    # ---- 10, the self-test of the drift verdicts, which is what makes 5 and 9
    # load-bearing rather than merely redundant
    # **The check the audit asked for, and the absence of it is why the drift
    # self-tests above were not enough.** 6 and 7 ask whether a *predicate* can
    # refuse and 8 asks whether the recorder can say SKIP; nothing here asked
    # whether a drift produces a *verdict* at the sites that record one. That
    # left checks 5 and 9 mutually redundant -- disabling either one alone still
    # exits 1, which reads like safety and is not -- and an audit that disabled
    # both was answered `11 passed, 0 failed`, exit 0, with the stale binary
    # winning. Three arms per drift check, and each closes a different way of
    # losing it:
    #   * **it recorded a result at all**, read out of this run's own RESULTS, so
    #     a deleted call site is a *missing name* rather than a silent one;
    #   * **what it recorded is what its own inputs imply**, re-derived through
    #     the same two functions the check calls, so an arm that is inverted or
    #     forced to PASS disagrees with itself;
    #   * **a drifted state is a FAIL** through those same functions, so an arm
    #     that cannot fail is caught even in a run that has no drift in it.
    # The re-derivation reads the surfaces this run already surveyed and the two
    # synthetic states are name sets held in memory, so no real `.pyd` is
    # touched, no file is written, and the two arms that need a drifted state do
    # not need one to exist.
    #
    # **The one shape these three cannot see, stated here rather than left to be
    # discovered.** An arm forced to PASS in a run that has no drift in it
    # records byte-for-byte what a correct arm records, and no self-test that
    # only reads a clean run can tell the two apart; pretending otherwise would
    # be the kind of claim this file does not make elsewhere. Arms 1 and 3 are
    # state-blind and fire whatever the machine looks like. Arm 2 fires
    # whenever the recorded verdict and the re-derived one differ, which is
    # every run this file was written to catch -- the one with the drift -- and
    # every run in which a drift check records a skip instead of a verdict. It
    # does not fire for a neutered arm in a clean run, and the mutations this
    # check is worth are the ones that stop a drift from being recorded at all.
    recorded_tags = {n: t for t, n, _d in RESULTS}
    faults: list[str] = []
    verdicts: list[str] = []
    if BSP is None:
        drift_skip = ("binary_source_parity_check.py is not beside this file. "
                      "The parity comparison here is that sibling's function "
                      "rather than a second parser in this file, so with it "
                      "absent there is no verdict left for these arms to ask "
                      "about -- and a self-test that skipped quietly would be "
                      "the one hole this check exists to close")
    else:
        drift_skip = ""
        # -- the parity arm, which re-derives what check 5 recorded
        tag = recorded_tags.get(PARITY_CHECK)
        if tag is None:
            faults.append(f"{PARITY_CHECK} recorded no result at all, so a "
                          f"binary behind its source is judged by nothing in "
                          f"this run")
        else:
            if parity_skip:
                want_parity = "SKIP"
            else:
                want_parity = ("FAIL" if parity_gap(
                    declared, names_of(loaded[winner])) else "PASS")
            verdicts.append(f"{PARITY_CHECK} recorded {tag} where its own "
                            f"inputs imply {want_parity}")
            if tag != want_parity:
                faults.append(f"{PARITY_CHECK} recorded {tag} where its own "
                              f"inputs imply {want_parity}: the arm that "
                              f"recorded it is not reading the comparison it "
                              f"reports")
        behind = "od_extparity_synthetic_name_behind_its_source"
        offered = (set() if winner is None
                   else names_of(loaded.get(winner, {})))
        behind_tag, _ = verdict_of(
            not parity_gap({**declared, behind: 0}, offered), "d", None)
        verdicts.append(f"a source declaring {behind} against the "
                        f"{len(offered)} name(s) the copy in play offers is "
                        f"{behind_tag} through the comparison check 5 calls")
        if behind_tag != "FAIL":
            faults.append(f"a source declaring {behind}, which no binary "
                          f"offers, is {behind_tag} through the comparison "
                          f"check 5 calls, so a stale build would be answered "
                          f"{behind_tag} rather than FAIL")
        # -- the agreement arm, which re-derives what check 9 recorded
        tag = recorded_tags.get(AGREEMENT_CHECK)
        if tag is None:
            faults.append(f"{AGREEMENT_CHECK} recorded no result at all, so "
                          f"two reachable copies of different builds are judged "
                          f"by nothing in this run")
        else:
            if len(several) < 2:
                want_agreement = "SKIP"
            else:
                want_agreement = ("FAIL" if disagreeing_copies(several, loaded)
                                  else "PASS")
            verdicts.append(f"{AGREEMENT_CHECK} recorded {tag} where its own "
                            f"inputs imply {want_agreement}")
            if tag != want_agreement:
                faults.append(f"{AGREEMENT_CHECK} recorded {tag} where its own "
                              f"inputs imply {want_agreement}: the arm that "
                              f"recorded it is not reading the comparison it "
                              f"reports")
        drifted = {in_tree: {"ok": True,
                             "module": ["od_extparity_synthetic_only_name"],
                             "classes": {}},
                   mirror: {"ok": True, "module": [], "classes": {}}}
        gap = disagreeing_copies([in_tree, mirror], drifted)
        verdicts.append(f"two reachable copies whose surfaces differ by one "
                        f"synthetic name are {len(gap)} disagreement(s) through "
                        f"the comparison check 9 calls")
        if not gap:
            faults.append("two reachable copies whose surfaces differ by one "
                          "synthetic name are not a disagreement through the "
                          "comparison check 9 calls, so two copies of different "
                          "builds would be answered PASS")
    if drift_skip:
        check(DRIFT_VERDICT_CHECK, False, "", skip=drift_skip)
    elif faults:
        check(DRIFT_VERDICT_CHECK, False,
              f"{len(faults)} way(s) a drift could pass this gate with every "
              f"self-test above green: " + "; ".join(faults)
              + ". Each is a way of losing a drift *verdict* that the predicate "
                "self-tests in 6 and 7 cannot see, because a predicate asked "
                "about a synthetic input still passes while the site that "
                "records its answer is deleted, inverted, or unable to say FAIL")
    else:
        check(DRIFT_VERDICT_CHECK, True,
              "both drift checks recorded a result, each recorded result is the "
              "one its own inputs imply, and a drifted state is a FAIL through "
              "the same two comparisons they call: " + "; ".join(verdicts)
              + ". So the two are redundant for the *verdict* of a run and are "
                "not redundant for the gate: disabling either one alone now "
                "turns this check red, which is the state the audit measured "
                "their absence in (both disabled, `11 passed, 0 failed`, "
                "exit 0, the stale build winning)")

    # ---- 11, every skip this run recorded is one it had to record
    # The audit runs *after* every result it audits, which took two attempts:
    # first it sat above the agreement check and so audited a run that had not
    # recorded its last result yet -- a skip the state had forced, not yet in
    # RESULTS, read as a result that should have been measured. The tally
    # cannot see this class of error, which is why it has its own check.
    unmeasurable: set[str] = set()
    if winner is None:
        unmeasurable |= {WINNER_CHECK, AGREEMENT_OF_RESOLVERS_CHECK}
    if not ranked:
        unmeasurable.add(LOADED_CHECK)
    if parity_skip:
        unmeasurable.add(PARITY_CHECK)
    if BSP is None:
        unmeasurable.add(VACUOUS_PARITY_CHECK)
        unmeasurable.add(DRIFT_VERDICT_CHECK)
    if planted is None:
        unmeasurable.add(VACUOUS_RESOLUTION_CHECK)
    if len(several) < 2:
        unmeasurable.add(AGREEMENT_CHECK)
    skipped = {n for t, n, _d in RESULTS if t == "SKIP"}
    # Both directions, and the second is the one that matters here. A check
    # that skips while it still had a subject is this gate degrading into the
    # unconditional two-file comparison it exists to replace: the tally would
    # not notice, because a skip is a result, and the run would print a green
    # next to a state that had something to measure.
    should_have_skipped = sorted(unmeasurable - skipped)
    should_have_measured = sorted(skipped - unmeasurable)
    if should_have_skipped or should_have_measured:
        check(SKIP_AUDIT_CHECK, False,
              f"{len(should_have_skipped)} result(s) this run had no subject for "
              f"and did not skip: {should_have_skipped}; and {len(should_have_measured)} "
              f"result(s) it could measure and skipped anyway: "
              f"{should_have_measured}. The state this run measured was: "
              f"{len(ranked)} copy/copies in play, a winner of {winner}, "
              f"{len(several)} of them loadable, and the parity comparison "
              f"{'unmeasurable' if parity_skip else 'measurable'}")
    else:
        check(SKIP_AUDIT_CHECK, True,
              f"every skip this run recorded is one the state forced -- "
              f"{len(skipped)} of {len(RESULTS) + 1} results -- and the "
              f"{len(RESULTS) + 1 - len(skipped)} that could be measured were "
              f"measured rather than skipped. So the two states this file "
              f"distinguishes are distinguishable in the output: "
              + (f"with {len(several)} loadable copy/copies in play the "
                 f"agreement was measured"
                 if len(several) > 1 else
                 f"with {len(several)} loadable copy/copies in play there was "
                 f"nothing for the agreement to compare, and it said so"))

    tally()
    return finish()


def tally() -> None:
    section("the tally")
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    name = "the tally accounts for every result, skips included"
    if npass + nfail + nskip == len(RESULTS) == EXPECTED_CHECKS - 1:
        check(name, True,
              f"{npass} + {nfail} + {nskip} = {len(RESULTS)}, and this check is "
              f"the {len(RESULTS) + 1}th, so the run reaches the declared "
              f"{EXPECTED_CHECKS} -- with two copies in play, with one, and "
              f"with none, because a skip is a result")
    else:
        check(name, False,
              f"recorded {len(RESULTS)} results ({npass}+{nfail}+{nskip}) before "
              f"this check, which must be {EXPECTED_CHECKS - 1}")


def finish() -> int:
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")
    print("    two numbers that are never added together: a pass count alone "
          "cannot say a binary was found, and a skip is not a pass. The exit "
          "code below says only that nothing FAILED, which is not the same "
          "claim as everything having been measured, so the verdict line names "
          "which of the two states this run was in")
    if nfail:
        print(f"RESULT: FAIL -- {nfail} of {EXPECTED_CHECKS} checks failed")
        return 1
    if npass == 0:
        print("RESULT: DID NOT FINISH -- this run asserted nothing")
        return 2
    if nskip:
        print(f"RESULT: OK, NOT FULLY MEASURED -- {nskip} of {len(RESULTS)} "
              f"checks could not be measured here, each with its reason above. "
              f"Exit 0 means no check failed and the {npass} that could be "
              f"measured did pass; it does NOT mean this file's whole question "
              f"was answered, and a CI step that reads the exit code alone has "
              f"proved nothing about the {nskip} it could not measure")
        return 0
    print(f"RESULT: OK -- {npass} passed and every one of the {len(RESULTS)} "
          f"checks was measured; nothing was skipped")
    return 0


def report_crash(exc: BaseException) -> int:
    """A run that did not finish says so, in the same words as a run that did.

    A summary line that printed `0 failed` after a crash would be
    indistinguishable from a clean run to anything reading the output rather
    than the exit code, and the exit code is the number a gate is judged on.
    """
    import traceback

    traceback.print_exc()
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- STOPPED EARLY after {len(RESULTS)} recorded result(s): "
          f"{npass} passed, {nfail} failed, {nskip} skipped")
    print(f"RESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}")
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_exc))
