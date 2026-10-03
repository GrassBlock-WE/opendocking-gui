"""Does every check script in `scripts/` say how many checks it runs?

Run:  python scripts/check_scripts_declare.py

# The problem this closes

Nine check scripts pin their total with `EXPECTED_CHECKS` and assert it. Twelve
do not. For those twelve, the count printed at the end is whatever happened to
run, and **nothing fails when it shrinks**.

The clearest case is not hypothetical. On a machine with no OpenGL context,
`workbench_interaction_check.py` cannot measure anything that needs a frame. It
reports the skips, separately and with reasons, and it still reaches its pinned
total -- because its `skip` records a result that counts toward the sum, so
"could not measure this" is a *result* rather than an absence. That is the
convention worth copying, and the section on skip conventions below says which
file does it which way.

The failure this file exists to prevent is the one a skip mechanism can hide: a
gate that reaches its number by not running the checks. A skip that increments
the check counter would let exactly that happen, so this file's own `skip` is
asserted not to touch `CHECKS`, and the summary prints passes and skips as two
numbers that are never added together.

# The contract every check script must satisfy

Either

    EXPECTED_CHECKS = 153  # measured from a green run, including this check

or, if a script genuinely cannot pin (its count depends on the machine in a way
that is understood and accepted), say so out loud:

    NO_EXPECTED_CHECKS = "reason, in enough words to be worth reading"

Both are module-level and both must be justified in a comment. A pin with no
provenance is a transcribed number, which is the thing this repository keeps
running into -- `scoring_cross_check.py` holds its own `WEIGHTS` literal for
exactly that reason. Requiring the comment costs nothing and makes "where did
this number come from" answerable at a glance.

# The second layer: `GATE-DECLARE`, and why the ledger had to go

The pin above is a claim about *how many results a run produces*. It is checked
by the run, in the file that owns it. The claim this file audits is a different
one: **how many `check()` call sites the source has, and which of them sit
behind a guard.** That fact is recoverable from the file by parsing it, so it
needs no ledger at all -- but it does need somewhere to record what it was
*supposed* to be, or nothing notices when it moves.

This file used to record that in two hand-typed tables and both of them drifted:

    SITE_INVENTORY = {"pockets_check.py": (65, 84, [ ... 84 guard strings ... ]), ...}
    UNPINNED_BASELINE = {"cli_prep_check.py": 37, ...}

`UNPINNED_BASELINE`'s numbers were **never compared to anything**. They were
printed in a table and believed. `cli_prep_check.py` sat there claiming 37
while the suite it describes was not even being measured -- see the hole noted
at `MAX_SNAPSHOT_ONLY`. And `SITE_INVENTORY` was a transcription of about two
hundred guard strings, which is the largest hand-maintained fact in the
repository and the reason `cli_check.py`'s pin moving 127 -> 128 -> 129 took a
round to clear: the number lived in two files and only one of them was the
file that changed.

So the declaration moved **into the thing it declares**, and the auditor stopped
keeping a table of what scripts are supposed to contain:

    #: GATE-DECLARE 1
    #: sites: 79 unconditional + 6 guarded
    #: guards: sha256:9f2c...e41b

Three facts, none of them hand-typed:

* `sites` is the call-site census, compared against `_sites_of`, which walks the
  AST. Add a check, wrap one in an `if`, delete one, and the derived number
  moves while the declared number does not.
* `guards` is a digest of the sorted guard strings, which is what catches the
  change the census alone cannot see: a site that moves between guard shapes
  without changing the total (`elif` -> `if` at the same nesting). It is one
  line instead of two hundred, and a human cannot transcribe a SHA-256.
* Both are parsed out of **comment tokens**, via `tokenize`, so a `GATE-DECLARE`
  quoted inside a docstring cannot be mistaken for a declaration. That is the
  same class of bug `pin_of` was written to avoid, and it is why the census is
  not read with a regex over the file text.

**What the declaration is actually for, and it is not the census.** The table
this file prints is already fully derived -- the numbers in it come out of
`_sites_of`, and the declared number is never printed anywhere. So a reader
loses nothing at the table if every declaration were deleted. What the
declaration buys is narrower and is a *baseline*: a frozen value to compare
against, so that a gate's contents changing is visible to somebody who did not
change them.

**It can drift, and it has, and here is the count.** The previous paragraph in
this file said "Why this cannot drift". That was wrong, and the proof is a run
against this tree: four gates are red, and the movement is a *human adding a
`check()`* -- `cli_check.py` 118 -> 120 sites, `core_check.py` 282 -> 357,
`workbench_interaction_check.py` 245 -> 297, `docs_claims_check.py` 100 -> 102.
Every one is growth, which is the normal way a gate gets better. A number
written in a comment goes stale; that is not fixable, and the claim that it was
is the kind of stated-but-untrue property this file exists to catch.

What *is* fixable is the cost of the stale direction, and that is what the
sections below do:

* a red now says **which kind** of movement it is. GREW, SHRANK, MOVED and
  UNDECLARED are four different events and were one indistinguishable string;
  a deletion and an addition produced byte-identical red text in the mutation
  proof. SHRANK is the one that must never be waved through.
* `--pin` **refuses to absorb a SHRANK** without `--accept-shrink`, and
  **refuses to absorb a MOVED** without `--accept-moved`. Growth -- the
  direction that fires when the project does something good -- is one command.
  Shrinking is a word somebody has to type, and so is the direction the two
  counts cannot see. The second flag exists because the first one was not
  enough: a MOVED gate keeps its call-site total and changes the conditions on
  them, so `--pin` rewrote the digest and exited 0 over a check that had been
  inverted, and the census then read the tree as clean. See `_pin`.
* `--census` prints what every drifted gate's declaration *should* say, as a
  diff, and **writes nothing**. `--pin` can only regenerate the table in this
  file, so for a gate whose number lives in its own source it could never do
  the job; this is the half it cannot do.
* the rewrite `--pin` performs is anchored on whole lines and asserts its own
  ordering, because the old anchors spliced a region into this file's own
  string literal and left a 226,082-byte file at 244,043 bytes that no longer
  parsed.

**A correction to the older claim, because it is load-bearing and false.** This
file said a declaration "cannot be satisfied by typing the number the auditor
wants to see, because the auditor computes it". It can: the red prints the
derived number in the very sentence that reports the disagreement. What is
actually true is narrower and is the property worth having -- the declaration is
**not independent** of the derivation, so it cannot drift *silently* and cannot
be a free-floating assertion no computation ever checks. It can absolutely be
transcribed, and four times in one session it was.

**One honest exception.** This file does not own every gate. `cli_check.py`,
`pockets_check.py` and seventeen others belong to other owners, and a
declaration block is an edit to *their* file. So for any gate whose file does
not carry a block, the declared value comes from `DECLARED_SNAPSHOT` -- which is
**generated**, never typed: `python scripts/check_scripts_declare.py --pin`
regenerates it from the very `_sites_of` walk the comparison uses. A generated
snapshot cannot drift the way a typed table did (drift is still caught; it just
takes an explicit regeneration to accept it), and it cannot be hand-edited to a
wrong number, because the comparison is against a computation.
`MAX_SNAPSHOT_ONLY` counts the gates still relying on it, and only ever falls.

#: The third class, and the reason this file can tell a measurement from a gate
#:
#: `INVENTORY` is every gate; `EXCLUSIONS` is every file whose name says "check"
#: and is not one, each with a reason. That left no home for a third kind of
#: file: a script that **measures** a constant, prints it, and is run by a gate
#: as a subprocess. `scripts/interp_reachable_step_bound.py` is one, and when it
#: was first written as `interp_mutate_check.py` this file reported it as a new
#: unpinned gate -- a deliverable being asked for a pin on a check total it does
#: not have, because the only signal available was the four letters in its name.
#:
#: So the distinction is now structural rather than a rename. A file under
#: `DELIVERABLE_PREFIXES` **is** a deliverable, and that buys it two properties
#: instead of an exemption:
#:
#:   * it defines no `check()` and carries no `EXPECTED_CHECKS` -- if it does, it
#:     is a gate wearing the wrong name, and it goes red instead of escaping the
#:     census;
#:   * some gate's source names it -- so it cannot sit in the tree unrun, which
#:     for a file whose entire output is a number is the failure mode that
#:     matters: a stale constant is indistinguishable from a correct one by
#:     reading it, because there is no verdict line and no exit code to look at.
#:
#: `examples_` is not a prefix, and the reason is a measured collision rather
#: than a preference: `examples_check.py` is a gate and begins with it.
#: The residual limitation -- a deliverable renamed to a name with neither a
#: prefix nor "check" in it is invisible here -- is recorded at the constant,
#: because the whole census is built on names that *say* check and widening it
#: is a change to other owners' files.
#
# Why this is a separate file from `docs_claims_check.py`

Three reasons, in order of weight:

1. **This one runs without the engine.** It only reads source text. It works on
   a machine where the wheel was never built, which is exactly the machine where
   you most want to know whether the suite's own bookkeeping is intact.
   `docs_claims_check.py` imports `opendocking` and cannot.
2. **Different blast radius.** A failure in the documentation audit and a failure
   in the suite inventory are different facts and should not be reported as one
   number in one place.
3. `docs_claims_check.py` is already misnamed -- it now covers the READMEs too,
   which was flagged to the document owner. Adding a fourth concern to it makes
   the name wronger, not righter.

# What this file does NOT do

It does not run the other check scripts. Seventeen of them take minutes, several
need a GPU or an OpenGL context, and a gate that times out teaches people to skip
it. This is a **static** audit of declarations; the runtime half is each
script's own pin, which is the thing that has to survive anyway.

Deliberately absent: a check that every pin is *correct*. A wrong pin is caught
by the script that owns it, which fails on the very run that discovers it. This
file only checks that the pin exists, is justified, and is actually compared
against something rather than merely declared.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import io
import re
import sys
import time
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# This gate's own output is UTF-8, whatever the console's codepage is.
#
# Added because it was not, and the failure is silent in the way that matters
# most here. Bare `sys.stdout.encoding` on a Chinese-Windows machine is `gbk`,
# and Python's `gbk` codec encodes U+2014 -- the em dash this file's prose uses
# throughout -- as the two bytes `A1 AA`. So a run of this gate on this machine
# emitted a byte sequence that is **not valid UTF-8**, and the first thing to
# break was a reader: a script that captured this output and decoded it as UTF-8
# raised `UnicodeDecodeError` on the first em dash and got no verdict at all. A
# gate whose output cannot be decoded is a gate whose result cannot be
# collected, and the failure mode is a traceback in someone else's tool rather
# than a red here.
#
# The same line is in `check_text_encoding.py` and
# `damaged_encoding_fixture.py`, and it is here for the same reason rather than
# for symmetry. `errors="replace"` is not a way of being tidy: it means a
# character this file cannot encode is dropped rather than raised, so the output
# is always decodable and the loss is visible as a `?`.
#
# **Do not remove this line as redundant.** That is the specific mistake it
# invites: it sits directly above `SCRIPTS = ...`, it re-encodes a stream that
# would be perfectly fine for anyone whose console is UTF-8, and it looks like
# defensive decoration of the kind this file elsewhere criticises. It is not.
# It is the only thing standing between this gate's verdict and a `UnicodeDecodeError`
# in a tool that did not happen to inherit an encoding setting, and the failure
# it prevents is not hypothetical: `docs_claims_check.py` died at check 19 of
# 124 on a non-ASCII print and survived only because an unrelated wrapper
# injected `PYTHONIOENCODING`. **A gate whose output cannot be decoded is a gate
# that only works for the person who remembered.** If this line is ever "cleaned
# up", the deletion is the bug, and the way to notice is to run this file with
# `PYTHONIOENCODING` unset and `chcp 936` active -- which is the default on the
# machine this was written on.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: `--census` and `--pin` are commands, not audits: each answers one question
#: and prints one thing. But this module prints its preamble at import, long
#: before either dispatch -- the classification section, the exit-reachability
#: walk, the per-pin loop -- and all of it lands in front of the answer, so
#: `--census` opened with two hundred lines of unrelated gate output and buried
#: the diff it exists to show. Swallowed here rather than papered over at the
#: dispatch, because the dispatch is downstream of the printing.
#:
#: Only `sys.stdout` is touched. `sys.stderr` is not, so a refusal from
#: `_pin` still reaches the reader, which is the whole point of putting those
#: messages on stderr in the first place.
_COMMAND_MODE = "--census" in sys.argv or "--pin" in sys.argv
_command_stdout = None
if _COMMAND_MODE:
    _command_stdout = sys.stdout
    sys.stdout = io.StringIO()


def _restore_stdout() -> None:
    """Put the real stdout back, if a command mode swallowed it."""
    global _command_stdout
    if _command_stdout is not None:
        sys.stdout = _command_stdout
        _command_stdout = None


SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"
TARGET = SCRIPTS / "check_scripts_declare.py"

CHECKS = 0
FAILURES: list[str] = []

#: Pinned by the owner or measured from a green run. Every in-scope script is
#: either in here with a pin, or declares `NO_EXPECTED_CHECKS`.
#:
#: This is a list of *names*, not of numbers, and it is here so that **deleting a
#: check script is a failure**. A directory scan cannot see a deletion -- the
#: file simply stops appearing and the scan reports a clean, smaller inventory.
#: The two halves of this file fail in opposite directions: a manifest entry
#: with no file on disk is a failure, and a file on disk with no manifest entry
#: is a different failure. Neither can pass silently, so a stale manifest here
#: cannot hide a real gap.
#: A gate pins its total by comparing what ran against a module-level
#: `EXPECTED_CHECKS`. Every file carries one, or says out loud why it cannot.
INVENTORY = [
    "benchmark_check.py",
    "binary_source_parity_check.py",
    "check_repo_docs.py",
    "check_counted_constants.py",
    "check_scripts_declare.py",
    "check_text_encoding.py",
    "check_verdict_bool.py",
    "clearing_paths_check.py",
    "cli_check.py",
    "cli_prep_check.py",
    "contacts_attribution_check.py",
    "contacts_criteria_check.py",
    "core_check.py",
    "energy_terms_check.py",
    "examples_check.py",
    "extension_install_parity_check.py",
    "extension_surface_check.py",
    "framing_selection_check.py",
    "gpu_cpu_parity_check.py",
    "gpu_feature_check.py",
    "guarded_sites_check.py",
    "installed_copy_check.py",
    "ligand_check.py",
    "mirror_drift_check.py",
    "pdbqt_check.py",
    "pockets_check.py",
    "pose_trust_check.py",
    "prep_check.py",
    "provenance_appendix.py",
    "published_set_floor_check.py",
    "raw_context_bound_check.py",
    "release_tree_parity_check.py",
    "representation_cartoon_check.py",
    "representation_geometry_check.py",
    "representation_names_check.py",
    "scoring_cross_check.py",
    "screenshot_frame_check.py",
    "skip_reasons_check.py",
    "docs_claims_check.py",
    "structure_bond_check.py",
    "torsion_bond_record_check.py",
    "unreached_product_check.py",
    "viewport_framing_check.py",
    "wheel_payload_check.py",
    "workbench_interaction_check.py",
    "x11_window_parse_check.py",
]

#: Files whose *name* says "check" but which are not gates, and why. Every such
#: file must appear here with a reason: a script that can be waved through with
#: a name change is not a classification, it is a loophole.
#:
#: **This table grew, and the growth is the finding.** It held two entries, both
#: of them helper modules -- and both of them are now classified structurally by
#: `_imported_by_a_sibling`, so the entries are redundant and kept only because
#: `role_of` consults this first. The other ten are files that are *not* gates,
#: *not* modules anything imports, and *not* handed to `run_script`: manual
#: probes and benchmarks. The old census could not see a single one of them,
#: because not one has "check" in its name.
#:
#: **Five of the ten are named by no other file in the repository** -- not
#: imported, not run, not cited: `framing_selection_screens.py`,
#: `probe_painted_frame.py`, `representation_cartoon_screens.py`,
#: `representation_screens.py` and `secondary_screens.py`. A grep for each of
#: those five names over `scripts/`, `examples/`, `dock-py/`, `dock-core/`,
#: `docs/` and `.github/` returns only the file itself. That is the honest cost
#: of making the census name-independent, and it is worth more than the census:
#: a diagnostic nobody runs is the failure mode this file exists to prevent, and
#: these five were not even on the list of things to worry about.
#:
#: **And this is where pins go to stop being verified, which is now a rule
#: rather than an observation.** Two entries below, `qt_gl_probe.py` and
#: `probe_painted_frame.py`, each carry an `EXPECTED_CHECKS`, and the per-pin loop
#: -- the thing that asserts "<name>'s pin is actually compared, not just
#: declared" -- walks `INVENTORY` and touches neither. So a table entry was
#: enough to place a promise outside the reach of every mechanism in this file,
#: and the two files are not in the same state: `probe_painted_frame.py` compares
#: its own pin at its line 642 and is verified, while `qt_gl_probe.py` renders its
#: pin in two f-strings inside `print()` and can be contradicted by nothing. The
#: check that closes it is in the section below, and the first arm of its
#: disjunction -- "or put it in INVENTORY" -- is blocked by the disjointness
#: check in the same section, because reclassifying a file is not the same
#: transaction as verifying its arithmetic. Neither file was moved, and the
#: reasons below are still the reasons: what a file *decides* and whether its own
#: numbers are checked are two separate questions and this table only answers the
#: first.
EXCLUSIONS = {
    "_gui_check.py":
        "a helper module imported by other scripts, not a gate; it defines no "
        "check() and running it does not report a pass/fail total. Also "
        "classified structurally: six siblings import it",
    "odgui_launch_check.py":
        "a launch probe, and the file x11_window_parse_check.py splices its "
        "helpers out of at import time; it is consumed, not run as a gate. Also "
        "classified structurally: three siblings import it",
    "pocket_benchmark.py":
        "a benchmark, consumed by benchmark_check.py through "
        "importlib.util.spec_from_file_location rather than by an import "
        "statement, which is why the structural detector does not see it. It "
        "needs the network, so it is not in CI; what keeps it honest is "
        "benchmark_check.py reporting drift in its numbers against a baseline",
    "redock_benchmark.py":
        "a benchmark, and the sibling of pocket_benchmark.py above: "
        "benchmark_check.py loads it the same way, as a module under test, and "
        "gates on the table it prints. It defines no check() of its own, declares "
        "no pin, and main() returns 0 unconditionally, so run directly it is a "
        "measurement that reports a number and cannot fail. It needs the network, "
        "and .github/workflows/ci.yml names both benchmarks there as not gates. "
        "This entry exists because the unpinned-gate check found it classified as a "
        "gate on the weak `raise SystemExit(main())` marker and vouched for "
        "nowhere -- which is the finding, not a formality: deleting this entry puts "
        "it back in that state and the check goes red again",
    "workbench_smoke.py":
        "a manual GUI smoke run: it needs a real OpenGL context, nothing in the "
        "tree launches it, and it imports _gui_check.py for the framebuffer "
        "test. The scripts/_gui_check.py CONSUMERS tuple names it as one of the "
        "four entry points, so the direction of the dependency is the other way "
        "round from the one the importer detector looks for",
    "contacts_screenshot.py":
        "a screenshot capture, run by hand against a live GUI; it imports "
        "_gui_check.py and is named in that file's CONSUMERS tuple. Not in CI "
        "because a CI machine has no window to photograph",
    "pockets_screenshot.py":
        "a screenshot capture, run by hand against a live GUI, for the same "
        "reason as contacts_screenshot.py and with the same _gui_check.py "
        "dependency; it differs only in which panel it photographs",
    "gl_route_probe_width.py":
        "a 16-variant GL measurement whose JSON is read by a person into the "
        "prose of workbench/launcher.py and scripts/probe_painted_frame.py. "
        "Nothing runs it, because it needs a live GL 4.6 context and its result "
        "is a study rather than a verdict; the two files that cite it are the "
        "record that it ran",
    "framing_selection_screens.py":
        "a screen-capture harness, named by no other file in the repository. It "
        "imports secondary_render.py and needs a live GUI, so there is nothing "
        "for a gate to launch and nothing for an import detector to find",
    "probe_painted_frame.py":
        "a single GL frame probe, named by no other file in the repository. It "
        "imports _gui_check.py and was written to settle one question about "
        "initializeOpenGLFunctions; scripts/gl_route_probe_width.py is the "
        "16-variant study that followed it, cited from the product's own "
        "documentation",
    "qt_gl_probe.py":
        "a GL mechanism probe, not a correctness gate: its four codes are facts "
        "about the machine, and the step needs continue-on-error: true so a "
        "machine fact cannot cancel the audits after it. It carries an "
        "EXPECTED_CHECKS because its verdict function and its child-output parser "
        "are asserted -- and that pin **is compared in a decision the run "
        "executes**, so it is no longer a promise this file cannot contradict. It "
        "used to be read in exactly two places, both f-strings inside print(): "
        "the '(expected 6)' summary and the 'N of 6 checks failed' line in the "
        "failure branch. The number was right (six check_* blocks, each called "
        "once from main()) and the total was therefore right on a correct run, "
        "which is precisely why it went unnoticed: a pin nothing can contradict "
        "cannot be wrong, and cannot be right either in any sense a reader could "
        "have checked. The comparison now returns DEFECT -- a verdict class this "
        "file already used for a fact about itself -- on a run that recorded a "
        "different number of results, and the recorded tally is left alone so the "
        "summary line cannot report a seventh result that the file did not run. "
        "Same class as probe_painted_frame.py and gl_route_probe_width.py, and it "
        "needs one workflow line rather than an inventory entry",
    "representation_cartoon_screens.py":
        "a screen-capture harness, named by no other file in the repository. It "
        "imports secondary_render.py, so it is an entry point rather than a "
        "module, and it needs a live GUI",
    "representation_screens.py":
        "a screen-capture harness, named by no other file in the repository. It "
        "imports secondary_render.py, so it is an entry point rather than a "
        "module, and it needs a live GUI",
    "secondary_screens.py":
        "a screen-capture harness, named by no other file in the repository. It "
        "imports secondary_render.py, so it is an entry point rather than a "
        "module, and it needs a live GUI",
}

#: Prefixes that mark a **measurement deliverable**: a script in this directory
#: whose job is to measure one quantity and print it, and which some *gate*
#: runs as a subprocess. It is not a gate, and the difference is structural
#: rather than a matter of what it is called.
#:
#: **Why this exists.** `interp_reachable_step_bound.py` was written with a
#: name ending in `_check.py`, and this file reported it as a new unpinned
#: gate: a deliverable that measures a constant and hands it to
#: `examples_check.py` was being asked for a pin on a check total it does not
#: have. Renaming it fixed that instance and left the hole open -- the gate
#: cannot tell a deliverable from a harness, so anything it cannot classify has
#: to be classified by a human with a reason attached, and the next measurement
#: script hits the same wall. A naming convention the auditor *enforces* is the
#: structural answer, and it is enforced from both sides:
#:
#:  * a file matching a prefix here must not define a `check()` or a pin, so it
#:    cannot be a gate wearing a deliverable's name to escape the census;
#:  * a file matching a prefix here must be launched by some gate -- handed to
#:    `run_script`, the one helper here that runs a script as a subprocess --
#:    so it cannot sit in the tree unrun, which for a file whose entire output
#:    is a number is the failure mode that matters: a stale constant is
#:    indistinguishable from a correct one by reading it, because there is no
#:    verdict line and no exit code to look at. A mention in prose is not a run,
#:    and the first version of this rule could not tell the two apart until a
#:    mutation made it.
#:
#: **One prefix, and `examples_` is not it.** It was the obvious second choice
#: and it is unusable: `examples_check.py` -- a gate, with a census and a pin --
#: begins with it, so a prefix set containing `examples_` declares the suite's
#: own examples gate to be a deliverable and the first half of this rule goes
#: red on a file that has been here for months. That is recorded here rather
#: than fixed by an exception list, because an exception list is a hand-typed
#: table and the collision is a fact about the two names.
#:
#: **The limitation, stated rather than papered over.** This is enforced for
#: the names that carry the signal. A deliverable renamed to something with
#: neither a prefix nor "check" in it (`mutation_probe.py`) is invisible here,
#: exactly as it was before this section existed, because the whole census is
#: built on names that *say* check. Closing that would mean auditing every
#: `run_script` argument in the tree, which is a change to other owners' files;
#: the honest scope of this one is the case that actually arose.
DELIVERABLE_PREFIXES = ("interp_",)


def _is_deliverable(name: str) -> bool:
    return name.startswith(DELIVERABLE_PREFIXES)


def _defines_check_call(path: Path) -> bool:
    """Does this file define a `check()` of its own? The marker of a gate.

    A file that cannot fail cannot be a gate, and a gate that cannot fail is
    not a gate by another name either. Read from the syntax tree, and an
    unparseable file counts as one: a gate that cannot be read is not thereby
    a deliverable.
    """
    try:
        tree = ast.parse(read(path))
    except SyntaxError:
        return True
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "check"
        for n in ast.walk(tree)
    )


def _run_by_a_gate(name: str) -> list[str]:
    """The gates that actually *launch* this file, not merely name it.

    `run_script` is this repository's one helper for running a script as a
    subprocess with the repository root as its working directory, and a
    deliverable is run by being handed to it. The first version of this check
    searched for the bare quoted name instead, and a mutation proved it wrong
    the only way a reader bug can be proved: replacing the `run_script` call
    with a stub left the name in a `section()` title and a `report()` argument,
    the search still found it, and the check passed on a file nothing ran. So
    the form searched for is the call, and a mention in prose is not a run.
    """
    return sorted(
        gate for gate in INVENTORY
        if (SCRIPTS / gate).exists() and f'run_script("{name}"' in read(SCRIPTS / gate)
    )


def _imported_by_a_sibling(name: str) -> list[str]:
    """The siblings that `import` this one, rather than running it.

    A module another script imports is consumed, not executed, so asking it
    for a pin or for a `run_script` call site asks the wrong question -- and
    `EXCLUSIONS` already had to carry `scripts/_gui_check.py` for exactly that
    reason, which is a human with a reason attached standing where a syntax
    tree answers the question outright. This is that answer, and it is the
    fourth role in `role_of` below.

    The form searched is the import statement, read from the tree, so a
    mention in prose is not an import: `secondary_render.py` is named in five
    files and imported by five, and those two counts agree here for no reason
    that would survive a reworded comment.

    **What this cannot see, stated because it cost an exemption.**
    `benchmark_check.py` loads `pocket_benchmark.py` through
    `importlib.util.spec_from_file_location`, which is a call and not an
    import statement, and a detector that looked for "the name appears as a
    string argument" would be back to matching prose. So `pocket_benchmark.py`
    carries an exemption saying so, and the thing that actually keeps it
    honest is `benchmark_check.py` reporting drift in its numbers.
    """
    stem = name[:-3] if name.endswith(".py") else name
    found: list[str] = []
    for other in sorted(SCRIPTS.glob("*.py")):
        if other.name == name:
            continue
        try:
            tree = ast.parse(read(other))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [other.name for a in node.names if a.name.split(".")[0] == stem]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] == stem:
                    found.append(other.name)
    return sorted(set(found))


def _can_report_failure(path: Path) -> bool:
    """Does this file have a verdict of its own that it can lose?

    The third shape of gate declaration, and the reason two files were
    invisible to the old census. `check_repo_docs.py` defines no `check()` and
    declares no `EXPECTED_CHECKS`: it counts what it found and hands the count to
    `sys.exit` as a non-literal argument -- `main()`. That *is* its declaration,
    and the `IfExp`/`Call` distinction is the whole test: a literal
    `sys.exit(0)` is a script that has already decided, and asking it for a pin
    would be asking about a number it never had.

    `check_text_encoding.py` used to be called `check_doc_encoding.py`, and it used
    to be the other such file. It scanned 17 Markdown documents; it now judges 159
    of the 161 files outside its skip directories -- the two it drops are an
    excluded suffix and a compiled binary, which is a difference in *reach*
    rather than a disagreement -- against 13 covered suffixes plus the
    extensionless rule, declares `EXPECTED_CHECKS = 19`, and carries a
    `GATE-DECLARE` block -- so it answers to the ordinary pin mechanism rather than
    to this predicate. The name was wrong for the scope and was kept, which is
    precisely how a comment outlives the fact it describes: the name was still
    wrong and the sentence about it no longer explained why.

    **This predicate on its own is not a gate marker, and the first version of
    `role_of` found that out by running it.** Every script in this directory
    ends in `raise SystemExit(main())`, including
    `scripts/interp_reachable_step_bound.py`, which is a measurement
    deliverable and not a gate at all. Put this test first and the census
    classified the one deliverable in the tree as a gate, leaving the
    `deliverable` role empty and vacuous -- a rule that cannot fire is
    indistinguishable from a rule that passes. So it is consulted *last* in
    `role_of`, only for a file that no gate launches and no sibling imports,
    which is what leaves it discriminating something.
    """
    try:
        tree = ast.parse(read(path))
    except SyntaxError:
        return True  # a gate that cannot be read is not thereby a deliverable
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in ("exit", "SystemExit") and not isinstance(node.args[0], ast.Constant):
            return True
    return False


#: The four roles a file in this directory can hold, and the order they are
#: decided in. Every one of them is structural -- a declaration in the syntax
#: tree, a call in a sibling, or an entry in `EXCLUSIONS` with a reason -- and
#: none of them is the four letters in the file's name.
#:
#: **The order is the argument, and one position in it was wrong first.**
#:
#:  * `exempt` first: a human who has written a reason outranks a structure
#:    they did not look at.
#:  * `gate` on a **strong** marker -- a pin or a `check()` of its own. Nothing
#:    else can fake either, and this is the only test that is decisive on its
#:    own.
#:  * `deliverable` next, and *before* the weak gate marker, because a gate
#:    launching this file through `run_script` is incompatible with this file
#:    being that gate. Putting it after the weak marker is what emptied the
#:    role; see `_can_report_failure`.
#:  * `imported` after that: a module a sibling imports is consumed rather than
#:    run, and cannot be launched even in principle.
#:  * `gate` on the **weak** marker last, because every script here ends in
#:    `SystemExit(main())` and the predicate only discriminates once the three
#:    stronger relations have been ruled out.
def role_of(name: str) -> str:
    path = SCRIPTS / name
    if name in EXCLUSIONS:
        return "exempt"
    if _declares_a_pin(path) or _defines_check_call(path):
        return "gate"
    if _run_by_a_gate(name):
        return "deliverable"
    if _imported_by_a_sibling(name):
        return "imported"
    if _can_report_failure(path):
        return "gate"
    return "unclassified"


def _declares_a_pin(path: Path) -> bool:
    """Does this file assign a module-level `EXPECTED_CHECKS`?

    Its own tiny reader rather than the file's `pin_of`, which is defined
    further down: this section has to run *before* that, and a pin is a pin
    here -- the only question asked of a deliverable is whether it has one.
    """
    try:
        tree = ast.parse(read(path))
    except SyntaxError:
        return True
    for node in tree.body:
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        if any(t.id == "EXPECTED_CHECKS" for t in targets):
            return True
    return False

MIN_NO_PIN_REASON = 40

#: Proxy for "this number has a note saying where it came from". Kept in one
#: place because it was previously spelled three slightly different ways at three
#: call sites, which is how a rule quietly stops being one rule. See the note at
#: the `justified =` line for what it can and cannot see.
#:
#: **`counted by running it` was added, and this time the evidence was a real
#: false red rather than a reader bug.** The standing rule in this repository is
#: "fix the reader before you loosen the rule", and it was followed twice before
#: this: `pin_of` was widened to see the `#:` block above a constant rather than
#: the vocabulary being widened to compensate. That reasoning does not apply here,
#: because the reader was not the problem. `torsion_bond_record_check.py` says
#: "How many checks this file is supposed to run, **counted by running it**" --
#: which is a provenance statement in this repository's own house voice, and
#: which `pockets_check.py` and `pdbqt_check.py` both use verbatim in their pins.
#: The regex simply did not know the phrase. Rejecting a comment that says where
#: the number came from, because it used a different synonym, is the reader bug
#: under a different name.
#:
#: The mutation that keeps this honest: narrowing the list back to the old three
#: words turns that gate red again, so the entry is load-bearing and not slack.
PROVENANCE_RE = re.compile(
    r"measured|deliberate|green run|counted by running", re.I
)

#: **Measured, not typed: there is no `IN_FLIGHT` dict any more.** A gate is
#: deferred only if its file was written *during this run* -- see `_is_in_flight`
#: for why that is a race detector rather than a mood, and for the two wrong
#: versions that were measured and discarded first.
#:
#: The fate of the two entries this used to hold, on evidence:
#:
#: * **`cli_check.py` -- removed.** It ran 129/129 green, its pin is 129, and its
#:   census (79 unconditional + 37 guarded) matches what `--pin` derived. The work
#:   the entry was parked for has landed, so there is nothing to defer.
#: * **`workbench_interaction_check.py` -- no longer needs deferring, and that is
#:   not a judgement call.** Its census moved 134 -> 148 -> 153 -> 165 and its pin
#:   243 -> 261 -> 268 during this session while its owner landed sections, so a
#:   time-based exemption was the only thing keeping it from reporting a drift
#:   every few minutes. Comparing mtimes against `RUN_START` instead covers the
#:   same race and is keyed on the save, not on the author: the file is measured
#:   whenever it is not being saved *right now*. It is separately red in its own
#:   right -- 241 passed, 1 failed, 1 skipped, 243 checks, exit 1, failing "the
#:   liveness threshold separates a held event loop from a live one", a
#:   timing-dependent check in a file this one does not own. That red is its
#:   owner's to clear; it is recorded here rather than absorbed.
#:
#: **The cap is 2, and it is now a statement about a race rather than about
#: anybody's working habits.** Two files saved inside the seconds a run takes is
#: a plausible ceiling; a run that defers more than that is not measuring a
#: race, it is measuring a broken clock or a filesystem with odd timestamp
#: behaviour, and both are worth a red.
MAX_IN_FLIGHT = 2

#: There is deliberately no `IN_FLIGHT` dict to add back. The exemption is keyed
#: on a file's mtime rather than on a name in a table, so it cannot be recreated
#: by editing a dict, and it cannot outlive the work it excuses: there is no
#: work-shaped key left to age out.

#: The auditor holds itself to the declaration it audits, like every other gate:
#: its own call-site census, declared in its own file, next to the code it
#: counts. This is the block every gate is migrating towards, and it is here
#: rather than in the generated snapshot precisely so that this file is the
#: worked example and not an exception to its own rule.
#: GATE-DECLARE 1
#: sites: 39 unconditional + 6 guarded
#: guards: sha256:16216eb9568a0e78526ed7fac68b16e8722108dfbe64ffac92c8083bdb7530d1
#:
#: **`38 -> 39`, and the digest above has not moved once.** The one added site is
#: the check that the drift triage is a total, disjoint partition -- a bare
#: `check(...)` at module level -- so the unconditional column went up by one
#: and the sorted guard strings are byte-for-byte what this block declared
#: before. `sha256:16216eb...` is not asserted to be stable, it was computed
#: from the file on disk and came out the same. Had that site been wrapped in an
#: `if`, the count would still have gone to 39 and the digest would have moved,
#: which is the change the two counts are blind to.
#:
#: **The four reds this file had are now five, and the fifth is this file.** It
#: went red on its own census the moment the triage check was added, before the
#: `sites:` line above was updated, which is the contract working rather than
#: the contract being inconvenient: the auditor is not exempt from the
#: mechanism it audits, and the run that adds a check to it reports itself in
#: the same four-way summary as everybody else.
#:
#: **`139 -> 140`, and the one is the unauditable-gate check.** It is a bare
#: `check(...)` at module level, which is why the `sites:` line above goes
#: `37 -> 38` and the guard digest does not move at all -- `sha256:16216eb...`
#: is byte-for-byte what this block declared before this round. That is
#: evidence rather than assertion: the digest is computed from the file on disk
#: and it came out the same. Had the new site been wrapped in an `if`, the
#: count would still have gone to 38 and the digest would have moved, which is
#: the change the two counts alone are blind to.
#:
#: **What the new check is, in one line:** a gate this file cannot read or
#: parse is a named red, never a `skip()` and never a `0`. The 0 is the
#: particular hazard here, because the walk's own 0 means "read it, and it
#: cannot fail" -- so folding an unreadable file into that number would accuse
#: a gate of being incapable on the strength of never having been looked at,
#: and the two are indistinguishable from the outside.
#:
#: **`28 -> 31` and the digest above does not move, which is the whole reason
#: there is a digest line as well as a count.** All three new sites are bare
#: `check(...)` statements at module level, so the unconditional column went up
#: by three and the sorted guard strings are unchanged -- `sha256:16216eb...`
#: is byte-for-byte what this block declared before this round. That is
#: evidence rather than assertion: the digest is computed from the file on
#: disk and it came out the same. Had any of the three been wrapped in an
#: `if`, the count would still have gone to 31 and the digest would have moved
#: -- which is the change the two numbers are blind to and the digest exists
#: to catch. The count alone cannot tell those two cases apart.
#:
#: **116 -> 118, and the two are the census's blind spot.** Both are
#: unconditional, which is why the `sites:` line above goes 26 -> 28 and the
#: guard digest does not move at all -- the digest exists to catch a site
#: changing guard shape without the total changing, and neither of these has
#: ever been guarded. One counts the result sites `_sites_of` cannot see and
#: holds them against a declared number per gate; the other mutates a copy of
#: `cli_check.py` to prove the first can go red. The gap they close is the
#: one this file had and did not state: a call in any position other than "the
#: whole statement" -- `r = check(...)`, `return check(...)` -- is invisible to
#: the census, and because the declared and derived columns both come from that
#: one walk, such a site moved neither column and the comparison above stayed
#: green on a gate that had genuinely grown a result.
#:
#: **115 -> 116, and the one is the check in the section "an unpinned gate can
#: still be unable to fail".** It is a single site and unconditional, which is
#: why the `sites:` line above goes 25 -> 26 and the guard digest does not
#: move at all -- the digest exists to catch a site changing guard shape
#: without the total changing, and this site has never been guarded. That is
#: also the honest description of what it checks: it is a static reading of
#: eight syntax trees, not a run, and the six mutations recorded at it are
#: string edits to a copy of a gate's source rather than eight subprocesses.
#:
#: **The gap it closes is one level below the `qt_gl_probe.py` one.** That was a
#: pin nothing could contradict. This is a gate that can contradict nothing,
#: because a run that recorded a failure and still exited 0 is indistinguishable
#: from a run that recorded nothing and exited 0 -- which is exactly
#: `redock_benchmark.py`, classified as a gate by `role_of` and vouched for by
#: nothing until the entry above it was written. The other seven mechanisms in
#: this file are all silent about that: the per-pin loop needs a pin, and the
#: census loop walks every gate in `INVENTORY` without asking what a result
#: site is *for*. So the claim is about the exit code, and the eight are walked
#: through `unpinned` rather than named, which is why nothing here has to be
#: edited when one of them gains a pin.
#:
#: 20 -> 21, and the added site is the census losing its dependence on a name.
#: The one it replaced was "no check-shaped script is unclassified", and it
#: walked `sorted(p.name for p in SCRIPTS.glob("*.py") if "check" in p.name)`:
#: 27 of the 45 files here, selected by four letters, with nothing whatever to
#: say about the other 18. The replacement walks every `*.py` and asks
#: `role_of`, which is decided from a declaration in the file's own syntax tree,
#: an import statement in a sibling, a `run_script` call in a gate, or an
#: `EXCLUSIONS` entry with a reason. The second site is the old check's actual
#: job -- a new gate cannot join the suite untracked -- keyed on the pin rather
#: than on the spelling, because INVENTORY is what the per-pin loop walks and a
#: pin that is not in it is a number nothing compares against a run.
#:
#: `62 -> 73` is the pin, and it is almost entirely the `EXCLUSIONS` loop:
#:
#:   62 + 10
#:       one site per entry in `EXCLUSIONS`, and the table went from two entries
#:       to twelve. The census itself is now derived; the *reasons* are the
#:       only typed thing left in it, which is exactly what that loop is for.
#:       Ten of the twelve are manual probes and benchmarks that declare no
#:       pin, are imported by nothing, and are launched by no gate -- and five
#:       of those ten are named by no other file in the repository at all.
#:   + 1
#:       the name-independent census above.
#:
#: The uncomfortable half of that is the one this file has been carrying since
#: the per-pin loop landed: **this file's own total is a function of how many
#: gates in the tree carry a pin**, and now also of how many files somebody has
#: had to justify by hand. Both make it a shared number rather than a private
#: one. The mechanism is working as designed -- a file that cannot be
#: classified is *measured* from that moment, which is the whole point -- but
#: the next person to land a pin or justify a probe moves this pin, and the
#: honest thing is to say so here rather than let the next red look like
#: somebody's arithmetic.
#:
#: 17 -> 18 was the check that no gate is left in DECLARED_SNAPSHOT after it
#: grew a GATE-DECLARE block. The guard digest is unchanged because the new site
#: is unconditional, which is the digest's job: a site moving between guard
#: shapes without changing the total is invisible to the two counts and visible
#: here.
#:
#: 18 -> 20, and both new sites are the deliverable convention: no file under a
#: deliverable prefix is a gate in disguise, and every deliverable is run by a
#: gate. Both are unconditional, so the digest is unchanged. They are two sites
#: rather than one because they fail in opposite directions -- the first
#: catches a gate escaping the census, the second catches a measurement nobody
#: reads -- and a single check asserting "the convention holds" would report
#: one red for two different problems.
#:
#: 60 -> 53, and the derivation is worth reading because the number moved a
#: little and the file lost a lot of checks.
#:
#: **Removed, 15 checks, all of them about a hand-typed table:**
#:
#:   * 12 x "<name> has no pin and is not in the standing-debt ledger" -- one per
#:     `UNPINNED_BASELINE` row. These asserted that twelve specific names are in
#:     a dict this file owns, which is a statement about this file, not about the
#:     repository. Replaced by the single derived ratchet, which is the same
#:     claim ("at most 12 gates are unpinned") and cannot be satisfied by listing
#:     the right twelve names.
#:   * "every ledger entry is still an unpinned script that exists" -- the
#:     staleness test for that dict. Unrepresentable now: a gate that gains a
#:     pin leaves the derived set on its own, so there is no stale row to catch.
#:   * "every script excluded from the site inventory is still a gate we track"
#:     and "every file exempt from measurement is in flight, and vice versa" --
#:     both existed to police the relationship between `SITE_INVENTORY` (17
#:     entries) and `INVENTORY` (21). Both sets are gone: every gate is measured,
#:     so there is nothing to police.
#:
#: **Added, 8 checks, all of them about the declaration mechanism or about gates
#: that appeared while this file was being rewritten:** the census comparison, the
#: one-source-per-gate rule, the snapshot ratchet and the in-flight cap (4), plus
#: three per-pin checks each for two gates that did not exist when this number
#: was last measured -- `torsion_bond_record_check.py` and
#: `representation_names_check.py` (6). Both arrived unclassified and were caught
#: by the "no check-shaped script is unclassified" check, which is the mechanism
#: working: a new gate cannot join the suite without being named here.
#:
#: **Added, 1 check:** "no gate is left in the snapshot after it grew a block of
#: its own". The detail line of the one-source-per-gate check *states* that a
#: gate may not appear in both places, and nothing tested it -- `_declared_for`
#: prefers the block, so a stale snapshot entry was invisible to every count here
#: and `--pin` would have removed it without comment. Stating a rule in prose and
#: not testing it is the same shape as the `SITE_INVENTORY` hole this file was
#: written to end, and the control that proves the new check can bite is
#: `examples_check.py`, which this same edit moved onto a block: it was in the
#: snapshot until then.
#:
#: 53 + 1 = 54, and **only** that one. This paragraph is the reason the pin was
#: left at 54 while the run measured 57: those three checks belong to gates
#: being edited by their own owners, and a pin that silently absorbs another
#: owner's in-flight work is a pin nobody can audit.
#:
#: **54 -> 62, and this time the pin does move**, because the three have landed
#: and the arithmetic is fully accounted for:
#:
#:   54 + 3
#:       one gate gained an `EXPECTED_CHECKS`, and the per-pin loop runs three
#:       sites (positive, justified, compared) for each pinned gate. This was
#:       the red the paragraph above declined to absorb, and it is now settled
#:       work rather than a race.
#:   + 3
#:       a second gate gained its pin during this session, same mechanism.
#:   + 2
#:       the two deliverable-convention sites above.
#:
#: **`73 -> 91`.** The first `3` of it is mine: `pose_trust_check.py` entered
#: `INVENTORY` above, and the per-pin loop runs three sites (positive,
#: justified, compared) for every pinned gate it walks -- the same mechanism the
#: `+3` rows above record. The remaining `+15` is not mine to attribute and I
#: will not pretend otherwise: it is other gates in the tree acquiring an
#: `EXPECTED_CHECKS` of their own while this session's workers were running, and
#: `core_check.py`, `examples_check.py` and `release_tree_parity_check.py` were
#: each mid-edit when this number was measured. So the count is a **shared**
#: quantity and the next pin to land moves it again.
#:
#: That sharedness is the uncomfortable half this file has carried since the
#: per-pin loop landed, and it is not a constant of this file the way
#: `examples_check.py`'s 29 is. The mechanism is working as designed -- a file
#: that cannot be classified is *measured* from that moment, which is the whole
#: point -- but the next person to land a pin moves this pin, and the honest
#: thing is to say so here rather than let the next red look like somebody's
#: arithmetic.
#: **`91 -> 97`, and every one of the six is accounted for by the same
#: mechanism: a gate that gained an `EXPECTED_CHECKS` of its own.** Three sites
#: each, from the per-pin loop (positive, justified, compared). The gates that
#: landed them during this session are `pose_trust_check.py`, `prep_check.py`
#: and `installed_copy_check.py`. **This file's own total is therefore a
#: function of the tree, not of this file**, and it is a shared number: the
#: next pin to land moves it again.
#:
#: Two of the six are still in flight and are recorded here as *not* yet
#: accounted: `prep_check.py`'s pin does not yet say where its number came
#: from, and seven gates have a declared call-site census that their owners are
#: mid-edit away from. Both are reds in this run, deliberately not absorbed: a
#: pin that silently swallows another owner's in-flight work is a pin nobody can
#: audit. The re-pin belongs to whoever owns those files.
#:
#: The uncomfortable half is the one this file has carried since the per-pin
#: loop landed, and it is not a constant of this file the way
#: `examples_check.py`'s 29 is. The mechanism is working as designed -- a file
#: that cannot be classified is *measured* from that moment, which is the whole
#: point -- but the next person to land a pin moves this pin, and the honest
#: thing is to say so here rather than let the next red look like somebody's
#: arithmetic.
#: **`97 -> 100`, and the three are one mechanism again: `wheel_payload_check.py`
#: gained an `EXPECTED_CHECKS` of its own**, so the per-pin loop added its three
#: sites (positive, justified, compared). That is the fifth gate to land a pin
#: during this session -- `pose_trust_check.py`, `prep_check.py`,
#: `installed_copy_check.py`, and now this one -- and the four before it were
#: only registered **after** each had been seen red by a mutation, two of them
#: by me rather than by their authors. A pin in this table is a claim that
#: something verified the number, not a claim that the number looks plausible.
#:
#: The rest of what moved here is not mine: seven gates have a declared
#: call-site census their owners are mid-edit away from, and that re-pin belongs
#: to whoever owns those files.
#: **`100 -> 101`, one more entry in `EXCLUSIONS`.** `qt_gl_probe.py` joined
#: that table with a written reason rather than entering `INVENTORY`, because it
#: is a GL mechanism probe and not a correctness gate -- its four exit codes are
#: facts about the machine. The table is walked **once per entry** by the
#: census loop, so one entry is one site, and this file's total moves with it.
#: The reason it did not simply go in `INVENTORY` is the reason it matters: as a
#: pin with no inventory entry it was red, and before that it had been passing
#: all three declaration checks purely by coincidence -- classified as a gate by
#: a weak "file ends with `raise SystemExit(main())`" marker, and invisible to
#: the untracked check because that one is keyed on *has a pin*.
#: **`101 -> 104`, and the three are the same mechanism as always:
#: `extension_surface_check.py` entered `INVENTORY`**, so the per-pin loop added
#: its three sites. It earned its place the only way a gate here can: it was
#: built against a failure this repository actually hit three times in one
#: evening -- a `TypeError` from three frames down because the compiled
#: extension was behind the source in arity -- and its mutations name which side
#: is behind rather than merely that something is.
#:
#: **It is not wired into CI, and that is a decision rather than an omission.**
#: Every job builds and installs from the same source, so the mismatch cannot
#: arise there and a step would be green by construction -- a gate with nothing
#: to be wrong about. It runs as a local preflight, which is where the stale
#: extension actually lives.
#:
#: Two gates in this table now carry a **self-test**: they build the thing they
#: exist to reject and require their own predicate to refuse it. That is the
#: difference between a green that means the subject was checked and a green
#: that means nothing was found, and it is the only mechanism here that can tell
#: those two apart on a machine where the subject is fine.
#:
#: **`104 -> 106`, and the two are the hole this file was still describing as
#: closed.** The pin-membership check is keyed on *declares a pin*, so a gate on
#: disk with no pin was classified by `role_of` and vouched for by nothing: the
#: exact shape of the `SITE_INVENTORY` failure this file exists to end, and one
#: that only became visible at the moment such a gate grew a pin, because that is
#: the only moment the check keyed on the pin can see it. A rule that reports a
#: long-standing gap only once somebody fixes it is a rule that misattributes the
#: red to whoever closed it.
#:
#:   104 + 1
#:       the new check -- a gate that declares no pin must be in INVENTORY or in
#:       EXCLUSIONS with a reason. It is unconditional, which is why the `sites:`
#:       line above goes 21 -> 22 and the guard digest does not move: the digest
#:       exists to catch a site changing guard shape without the total changing,
#:       and this site was never guarded.
#:   + 1
#:       one entry in EXCLUSIONS: `redock_benchmark.py`. The table is walked once
#:       per entry, so an entry is one executed check and **no new site** -- which
#:       is why it is `+1` here and nothing at all in the census.
#:
#: The second of those is the finding rather than the bookkeeping, and how it
#: was found is the part worth keeping: by **writing** the check, not by
#: mutating the program into a red. `role_of` had been calling
#: `redock_benchmark.py` a gate on the weak `raise SystemExit(main())` marker for
#: as long as that marker had been consulted, and the reason it was in no table
#: is that no check had ever asked. The mutations that prove the new check bites
#: are recorded at it, and both are **removals** -- a name out of INVENTORY, this
#: entry out of EXCLUSIONS -- because what is being guarded is a classification,
#: and a classification can be wrong while every other check passes exactly when
#: the name stops being there.
#:
#: **`106 -> 115`, and only 3 of the 9 are this round's work.** The total is
#: three mechanisms and a reader who cannot check the arithmetic by reading the
#: file is reading a number, not a claim, so here they are as of 2026-10-02,
#: derived rather than counted off a run:
#:
#:   106 + 3
#:       the three new checks in the section "a pin is a promise, and EXCLUSIONS
#:       does not discharge one", all unconditional. Two of the three are the rule
#:       and its cap; the third is the staleness test on the debt table, without
#:       which the table could outlive the gap it describes -- the same hole
#:       `SITE_INVENTORY` had. All three are sites and no new guard, so the
#:       digest above is unchanged: `22 -> 25` unconditional and nothing else.
#:   + 6
#:       **not this round's, and the second one is named now.**
#:       The per-pin loop runs three sites for every gate in `INVENTORY` that
#:       carries a pin, and it had 23 of them at 106. It has 25 now. Two gates
#:       landed an `EXPECTED_CHECKS` of their own in between -- 3 sites each, 6
#:       total. `gpu_cpu_parity_check.py` is one of them (its pin was derived and
#:       added in the previous round). **The other is `check_text_encoding.py`**,
#:       and the two records that name it are both outside this file:
#:
#:         * `docs/VERIFICATION.md` row 272, whose fix column says the gate now
#:           "scans 159 files / 14 types and declares `EXPECTED_CHECKS = 19`",
#:           and whose symptom column describes `check_doc_encoding.py` -- a
#:           different name, and a file that declared no total at all. A pin
#:           that is part of a rename's fix is a pin that arrived with the
#:           rename.
#:         * the gate's own `GATE-DECLARE` block, which records that "the
#:           previous version of this file had **zero** such sites" and that it
#:           sat in this file's generated snapshot as `(0, 0, ...)` "so the
#:           auditor believed a zero that described a file which did not report
#:           checks". A gate that is *in* a generated census is already
#:           inventory, and this file's `INVENTORY` did not change across the
#:           window -- so what arrived is the pin.
#:
#:       Its last write falls between the 106 reading and the 07:24 save that
#:       wrote 115, which is corroboration and **not** the proof: the 106
#:       reading left no timestamp and no set, so this is an identification from
#:       two independent records plus a window, not a derivation, and it should
#:       be read as the same class of claim as any other hand-attribution here.
#:       The reason it was unattributable at all is the reason to record the
#:       *set* next time rather than the count: a count of twenty-five files
#:       cannot say which two changed. The unpinned set below is already derived
#:       for exactly this reason.
#:
#:       `cli_check.py`, `core_check.py`, `examples_check.py`,
#:       `prep_check.py` and `release_tree_parity_check.py` have all had their
#:       call-site counts move under this run as well, and those re-pins belong
#:       to their owners and are deliberately not absorbed here.
#:
#: **The four `NOT DECLARED ANYWHERE` reds are closed, and that moved no number
#: at all**, which is the part worth noticing. `extension_surface_check.py`,
#: `gpu_cpu_parity_check.py`, `installed_copy_check.py` and
#: `wheel_payload_check.py` each carried a pin and were each in `INVENTORY` and
#: each declared no call-site census anywhere -- so the census loop walked all
#: four, compared them against nothing, and printed `NOT DECLARED ANYWHERE` for
#: each. Each now carries a `GATE-DECLARE` block derived by walking its own
#: syntax tree, and **not one of those four edits changed how many checks this
#: file runs**: a census is a comparison, and the four were already being walked.
#: That is the difference between a declaration and a pin, and it is why a pin
#: being present is not evidence that a gate is declared -- see the `INVENTORY`
#: header.
#:
#: **118 -> 121, and none of the three is a change to an existing gate.**
#: Two are a debt this file was already carrying when the round started -- the
#: run reported 120 before the total check against a declared 118, so the pin
#: was two short of what the file actually does -- and the third is a new gate,
#: `binary_source_parity_check.py`, entering `INVENTORY`. The per-pin loop is
#: keyed on that table, so one more gate is one more check. The number is the
#: measured one and nothing here is typed: a reader who finds 121 surprising
#: should run the file and read the count it prints, not diff the loops.
#:
#: **`121 -> 130`, and all nine are this round's, with three mechanisms.**
#: The run before this edit measured 127 against a declared 121, so the file
#: was six short of what it already did -- that debt was already there and is
#: absorbed here rather than left for the next reader to find. On top of it,
#: three new sites, all unconditional:
#:
#:   127 + 1
#:       "a zero here means one of two things": every gate that reads
#:       `exit_reach = 0` is either named in `UNPROVABLE_BY_AST` or is a red.
#:       Until this existed, a `0` and a "could not be read" were the same
#:       number, and the rule printed neither.
#:   + 1
#:       its staleness test, without which an entry in `UNPROVABLE_BY_AST`
#:       could outlive the hole it recorded -- the `SITE_INVENTORY` shape, and
#:       the reason that table is only mentioned in this file as a mistake.
#:   + 1
#:       "the comprehension arm earns its verdict": the arm that reads
#:       `nfail = sum(1 for ... if ...)` as a tally had no consumer that could
#:       observe it, so it could have been deleted and this file would have
#:       stayed green.
#:   + 6
#:       the root-file section, counted by running the file: two checks that the
#:       release rule's tables were readable and that the workspace root is not
#:       one of its SOURCE_ROOTs (so no root file can be produced by the wheel
#:       or the commit mirror), the census itself, and three controls that
#:       prove the census is a measurement -- two on the reader, showing it
#:       refuses a path component and drops a credit when the reference is
#:       removed, and one on the classifier, showing it can report a clean
#:       root so its red is a finding rather than a constant. All six are
#:       unconditional, so the `sites:` line above goes 31 -> 37 and the guard
#:       digest does not move.
#:
#: **`136 -> 139`, and every one of the three is the per-pin loop, not a new
#: check.** The number was re-derived by running the file rather than taken from
#: a hand-off, and it landed on the same 139 for the reason the loop's shape
#: predicts: `check_repo_docs.py` gained an `EXPECTED_CHECKS`, and a gate that
#: enters the per-pin loop costs exactly three -- justified, positive, compared.
#: That gate is the whole delta, and it is worth naming because the alternative
#: reading is that this round added three checks of its own. It did not: the
#: edits below are one `SKIP_CONVENTION` row, three deleted snapshot rows, a
#: stronger reader, a stdout `reconfigure`, and a cap lowered to a measured
#: set. **A pin arriving is a cost to this file that has nothing to do with what
#: this file was asked to do**, which is why the constant is set from the run
#: and never from the diff.
#:
#: **141 -> 147, and the cause is `INVENTORY` growing by two, not this file
#: gaining checks.** `clearing_paths_check.py` and `check_counted_constants.py`
#: were added to the inventory, and the per-pin loop below walks every name in
#: it, so each new gate arrives with its own three per-pin results. Two gates,
#: three results each, six: 141 + 6 = 147, and the first attempt at this number
#: was 146 because the run prints "146 ran before this one and 146 are expected;
#: the +1 is this check" -- the count excludes the tally, which counts itself.
#: Nothing in this file's own check count moved: the declared census above is
#: unchanged at 39 unconditional + 6 guarded, which is the column that would
#: have caught an edit here, and which did not move.
#:
#: **159 -> 165, and the cause is `INVENTORY` again, not this file.** Two gates
#: joined the list this round -- `guarded_sites_check.py` and
#: `published_set_floor_check.py` -- and the per-pin loop below walks every name
#: in it, so each new gate arrives with its own results. **Re-read from the run's
#: own "N ran before this one and M are expected" line every time**, which is why
#: this comment keeps restating the rule instead of the arithmetic: three of the
#: values this number has passed through were miscounts, and a comment that
#: showed the sum would have made all three look deliberate.
EXPECTED_CHECKS = 192

#: Gates with no `EXPECTED_CHECKS`. **Derived, not typed.**
#:
#: This used to be `UNPINNED_BASELINE`, a dict of twelve names and twelve
#: numbers, and the numbers were the worst thing in the file: they were printed
#: in a table and compared against *nothing*. `cli_prep_check.py` was recorded at
#: 37 while nothing in this file could tell you what it actually ran, and
#: `core_check.py` was recorded at 392 against a derived 259 call sites -- the
#: two numbers mean different things and the table did not say which was which.
#: A hand-typed number that is never compared is not a measurement, it is a
#: guess with a table around it, and a reader has no way to tell.
#:
#: Now the set is computed from the files themselves (`unpinned` below), so the
#: two directions of drift that used to need a human are unrepresentable:
#:
#:   * a script that **gains** a pin leaves the set by itself. The old failure --
#:     "in the ledger but has gained an EXPECTED_CHECKS", which is what turned
#:     this gate red at 58/60 -- required someone to remember to delete a row
#:     from a different file. There is no row to delete.
#:   * a script that **loses** its pin joins the set by itself, and the ratchet
#:     below is what notices.
#:
#: `MAX_UNPINNED_SCRIPTS` is the only hand-typed number left here, and it is a
#: cap rather than a record.
#:
#: **Set to 7, the measured size of the set, because the ratchet's own message
#: asks for exactly that.** It says to lower the cap "at which point ... so the
#: slack cannot be spent on a new gap", and slack is the thing a cap exists to
#: refuse. The number was counted by running the file, not taken from a
#: hand-off that put it at 11; the derived set is the seven gates the census
#: above names. A cap of 12 over a set of 7 is not a tight bound, it is a
#: budget.
#:
#: The risk is stated rather than hidden: the set is *derived*, so it moves when
#: another agent adds or removes a pin, and a pin removed for a good reason would
#: take this red until somebody lowers nothing and raises this by one. That is
#: the ratchet working, not a false positive, and it is the same trade every
#: other cap in this file makes.
MAX_UNPINNED_SCRIPTS = 7

#: The same ratchet, for the other kind of debt: a pin that exists and is
#: compared, but carries no note saying where the number came from. An
#: unjustified pin is a transcribed number, which is the failure this repository
#: keeps meeting -- `scoring_cross_check.py` holds its own `WEIGHTS` literal for
#: exactly that reason.
#:
#: **Empty, and every pin here now carries a note.** `cli_check.py` and
#: `workbench_interaction_check.py` were the last two, and both were false reds
#: caused by `pin_of` reading only the comment on the constant's own line and not
#: the `#:` block above it -- this repository's house style for saying something at
#: length. The reader was fixed; the vocabulary was not widened, because a
#: mutation narrowing it straight back stayed green and proved the widening was
#: unnecessary. The rule to keep: fix the reader before you loosen the rule.
#: The ones that lack a note would be listed here, not silently tolerated.
UNJUSTIFIED_PINS: dict[str, str] = {}
MAX_UNJUSTIFIED_PINS = 0


def check(ok: bool, name: str, detail: str = "") -> bool:
    """Record one result. **A verdict that is not a `bool` is refused.**

    `if not ok` reads a Python *truth value*, not a verdict, so a non-bool
    that happens to be truthy takes the passing branch and one that happens to
    be falsy takes the failing branch, and in neither case does the output say
    which value was wrong. The shape that actually reached this tree is a slot
    ending `return (not moved, "")`: a **tuple** is truthy whatever the bool
    inside it says, so the one slot in `clearing_paths_check.py` whose entire
    job is to be the witness against that run touching a live file could not
    fail. It survived a green 13-of-13, and it was visible only because the
    detail line said `these moved: check_text_encoding.py` directly under an
    `[ok]`.

    So the *type* is checked rather than the truth value. A refusal still
    increments the total and still records exactly one result -- the total has
    to stay reachable, or this file's own pin becomes a lie -- and it is
    recorded as a failure, because a slot that handed this function a
    non-verdict has lost its verdict and the run has to say so in words rather
    than in an exit code. The falsy half (`()`, `""`, `None`, `0`) is the half
    that matters most: it is a live false red that no output explains.

    `check_verdict_bool.py` sweeps every call site in `scripts/` for this shape
    and executes this function against a stub to prove the guard below is
    load-bearing rather than merely present.
    """
    global CHECKS
    CHECKS += 1
    if not isinstance(ok, bool):
        FAILURES.append(name)
        print(f"[FAIL] {name}")
        print(f"       this call passed {ok!r} ({type(ok).__name__}) where a "
              f"bool belongs, and it is "
              f"{'truthy' if ok else 'falsy'}, so read as a verdict it would "
              f"have said {'pass' if ok else 'fail'} no matter what it was "
              f"meant to mean. Refused rather than coerced")
        if detail:
            print(f"       {detail}")
        return False
    if not ok:
        FAILURES.append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")
    return ok


#: Checks this file could not run here, as (name, reason). See `skip` below and
#: the `SKIP_CONVENTION` table further down for why this exists at all.
SKIPPED: list[tuple[str, str]] = []


def skip(name: str, reason: str) -> bool:
    """Record a check this environment could not answer, and say why.

    A skipped check and a passed check are different claims, and only one of them
    is ever true. "No gate here needs a `skip()`" and "every gate here needed one
    and this machine could not provide it" are indistinguishable in a tally that
    only counts passes -- which is why `skip` does **not** increment `CHECKS`,
    and why the summary line prints the two numbers side by side instead of
    summing them.

    It returns False so that it can stand in for a check in a branch, exactly as
    `check` does, without pretending the branch verified anything.
    """
    SKIPPED.append((name, reason))
    print(f"[SKIP] {name}")
    print(f"       {reason}")
    return False


def section(t: str) -> None:
    print(f"\n=== {t} ===")


def read(path: Path) -> str:
    # utf-8-sig: `redock_benchmark.py` carries a BOM, and a strict decode of it
    # raises. A gate that cannot read the repository is not a gate.
    return path.read_bytes().decode("utf-8-sig")

#: When this run started, captured at import and before any file is read.
#:
#: This is what makes the in-flight deferral precise rather than a mood. It is
#: compared against each gate's mtime, so the only files that are deferred are
#: files that were **saved while this run was in progress** -- the genuine race,
#: where the auditor reads a census and the owner saves an edit a moment later,
#: and the gate reports a drift for a file that is being worked on as you read it.
RUN_START = time.time()


def _age_minutes(path: Path) -> float:
    """How long ago `path` was last written, in minutes."""
    return (time.time() - path.stat().st_mtime) / 60.0


def _is_in_flight(path: Path) -> bool:
    """Was this file written *during* this run?

    **Measured from the filesystem, not named in a dict.** This is the mechanism
    that replaced `IN_FLIGHT = {"cli_check.py": ..., "workbench_interaction_check.py": ...}`,
    and the difference between them is the whole point. The old list was keyed on
    a *name*, with a prose reason, and nothing checked whether the reason was
    still true -- a structure that cannot expire, which is why both of its
    entries had been parked across many rounds. This one is keyed on the file's
    own mtime against the start of the run, so it covers exactly one thing: a
    save that landed between the auditor starting and the auditor reading.

    Two wrong versions of this were built and measured first, and the reasons
    they were wrong are worth more than the code:

    * **A six-hour window.** Chosen so CI would not flap. Measured, it swallowed
      18 of 22 gates -- every file anyone had touched that afternoon -- and
      called the measurement "in flight". It was not deferring a race, it was
      disabling the census for most of the tree, and the cap went red for it,
      which is how the mistake was caught.
    * **A fifteen-minute window.** Better, but still wrong in the way that
      matters: it punished the *author* for working. Edit a gate, run the gate,
      and the gate reports a red about your own save -- including when the save
      is the thing you were verifying. A mechanism whose failure mode is "you
      touched a file" is a mechanism people disable.

    Comparing against `RUN_START` instead has neither defect. A file saved an
    hour ago is measured, always, no matter how recently that was; a file saved
    a second ago is measured too, because this run began before it. The deferral
    can only fire for a save that genuinely landed inside this run, which on a
    developer machine is normally zero files.

    It is also deterministic on CI, which the two window versions were not: a
    fresh checkout writes every file before the job starts, so nothing is ever
    deferred there and the census is fully measured on every runner run.

    Properties, so it cannot be mistaken for the thing it replaced:

    * **It only ever covers the census comparison.** Existence, inventory
      membership, pin provenance and the skip conventions are measured for a
      deferred file too. It is not an exemption from being tracked.
    * **It cannot grow.** A file that was not saved during the run cannot be
      deferred, so the set is bounded by what a concurrent editor can do in the
      seconds a run takes -- not by how long anyone has been working.
    * **It prints.** Every deferred file is named, with its age, in the check
      that asserts the cap.
    """
    try:
        if path.resolve() == TARGET:
            # This file never defers itself. It is the auditor; the only writer
            # during a run is whoever is editing the auditor, and that editor can
            # see the census they just changed. Deferring the auditor's own
            # census would mean the one file in the repository that cannot be
            # mid-save is the one file exempt from being checked.
            return False
        return path.stat().st_mtime >= RUN_START
    except OSError:
        return False  # unreadable mtime: measure it rather than defer it



# ==========================================================================
section("every file in scripts/ is classified, by something other than its name")

# The census used to be `sorted(p.name for p in SCRIPTS.glob("*.py") if "check"
# in p.name)`, which is a census of a naming convention. It found 27 of the 44
# files here and had no way to complain about the other 17: a gate renamed
# `mutation_probe.py` was invisible, and so was a measurement script nobody
# runs. The replacement walks every `*.py` and asks `role_of`, so the question
# is what the file *declares*, not what it is called. The cost is stated at
# `EXCLUSIONS`: ten files need a written reason, and five of those are named by
# no other file in the repository at all, which is the finding rather than an
# inconvenience.
every_script = sorted(p.name for p in SCRIPTS.glob("*.py"))
roles = {name: role_of(name) for name in every_script}
unclassified = [n for n, r in roles.items() if r == "unclassified"]
by_role: dict[str, list[str]] = {}
for name, role in roles.items():
    by_role.setdefault(role, []).append(name)
check(
    not unclassified,
    "every file in scripts/ has a role, and none of them is its own name",
    f"{len(every_script)} file(s) classified without reading a name: "
    + "; ".join(
        f"{len(by_role.get(role, []))} {role}"
        for role in ("gate", "imported", "deliverable", "exempt", "unclassified")
    )
    + f". {len(unclassified)} unaccounted for: {unclassified}. "
    f"A file lands in one of the first four by a declaration in its own syntax "
    f"tree, an import statement in a sibling, a run_script call in a gate, or "
    f"an entry in EXCLUSIONS with a reason -- so a new script cannot sit here "
    f"unclassified by choosing a name the old census could not see"
    if not unclassified
    else f"{len(unclassified)} file(s) match no role: {unclassified}. Either "
         f"it is a gate that declares nothing, a module nothing imports, a "
         f"measurement no gate runs, or it needs an EXCLUSIONS entry with a "
         f"reason. Naming it is the fix; nothing about its name is",
)

# The old census's real job -- a new gate cannot join the suite without being
# tracked -- is kept, and keyed on the pin instead of on the spelling. INVENTORY
# is what drives the per-pin loop below, so a gate that carries a pin and is
# missing from it is a gate whose pin is never verified by anything. That is
# the failure the name-based version caught, and it caught it by accident of
# spelling: `benchmark_check.py` has carried a `check()` for the whole session
# and was not in INVENTORY either, so the old rule was one rename away from
# passing on a gate nothing was checking.
pinned = sorted(
    n for n in every_script
    if n not in EXCLUSIONS
    and (SCRIPTS / n).exists()
    and _declares_a_pin(SCRIPTS / n)
    and n != TARGET.name
)
untracked = [n for n in pinned if n not in INVENTORY]
check(
    not untracked,
    "every gate that declares a pin is in the inventory, so something verifies it",
    f"{len(pinned)} pinned gate(s) other than this one, {len(INVENTORY)} listed; "
    f"not tracked: {untracked}. INVENTORY is what the per-pin loop below walks, "
    f"so a pin that is not in it is a number nothing compares against a run. "
    f"The test is the pin, not the spelling, so a gate that renames itself "
    f"stays tracked and a new one has to be added deliberately"
    if not untracked
    else f"{len(untracked)} pinned gate(s) are not in INVENTORY: {untracked}. "
         f"Add the name, or take the pin out of the file -- do not leave a pin "
         f"that only the file that wrote it can vouch for",
)

# The same hole, on the other side of the pin, and it was open for a reason that
# only looked like a strength: the check above is keyed on **declares a pin**, so
# a gate that has not grown one is not in the set it walks. The consequence is
# that a gate on disk with no `EXPECTED_CHECKS` was classified by `role_of` and
# vouched for by nobody -- the exact shape of failure this file was written to
# clear out, a file that is classified but not accounted for. And it was worse
# than a static gap, because the moment the gate grew a pin the check *above*
# would have gone red: a rule that cannot see a gap until the gap is fixed is a
# rule that reports other people's arithmetic as if it were its own.
#
# So the claim here is deliberately the **disjunction**, not "every pin is
# tracked". For any file `n` in this directory:
#
#     role_of(n) == "gate"  and  n declares no pin
#         =>  n in INVENTORY    it is a gate, and the census loop below measures it
#          or n in EXCLUSIONS   somebody looked at it and wrote down why it is
#                               not one -- which is what `role_of` consults first
#
# **The `EXCLUSIONS` arm cannot fire today, and that is structural rather than a
# defect in the rule.** `role_of` returns "exempt" for anything in that table
# before it looks at a pin, a `check()` or `sys.exit(main())`, so a file sitting
# in `EXCLUSIONS` is never in `role_of(n) == "gate"` in the first place and the
# second arm is unreachable. It is written out anyway, for two reasons that are
# stronger than reachability: the rule a reader needs is the disjunction, because
# "classified as a gate" and "vouched for by a name or by a reason" are two
# different claims and stating only the first would leave a future reordering of
# `role_of` with nothing behind it; and a file that reaches that arm gets there
# by being *exempted*, which is a stricter property than being vouched for rather
# than a looser one. A vacuous clause is only a problem when it is pretending to
# be a test; this one states the rule, and the comment says which arm is live.
#
# `redock_benchmark.py` is the live instance, and it was found by **writing** the
# check rather than by mutating the program into it: `role_of` calls it a gate on
# the weak `raise SystemExit(main())` marker, it declares no pin, it defines no
# `check()` of its own, and it was in neither table. It is now in `EXCLUSIONS`
# with the reason, because `main()` returns 0 unconditionally and
# `benchmark_check.py` loads it as a module under test and gates on its table --
# a benchmark, which is the same relation and the same limitation as
# `pocket_benchmark.py` two entries below. What keeps a reader from having to
# take that on trust is the mutation in the other direction: take the entry out
# and this goes red again, so the entry is what vouches for the file and not the
# fact that its name contains no interesting letters.
_gates = sorted(n for n in every_script if roles[n] == "gate")
_unpinned_gates = [n for n in _gates if not _declares_a_pin(SCRIPTS / n)]
_unvouched = [n for n in _unpinned_gates
              if n not in INVENTORY and n not in EXCLUSIONS]
check(
    not _unvouched,
    "every gate that declares no pin is in the inventory, or excluded with a reason",
    "; ".join(
        f"{n} -> {'INVENTORY' if n in INVENTORY else 'EXCLUSIONS'}"
        for n in _unpinned_gates
    ) + f". {len(_gates)} file(s) classified as a gate, {len(_gates) - len(_unpinned_gates)} "
    f"of them declaring a pin and {len(_unpinned_gates)} not. A gate with no pin is "
    f"invisible to the membership check above, which is keyed on the pin -- so without "
    f"this one, a gate could sit on disk, be classified by role_of, and be named by "
    f"no table, and the only thing that would have noticed is the moment somebody "
    f"gave it a pin"
    if not _unvouched else
    f"{len(_unvouched)} gate(s) declare no pin and appear in neither INVENTORY nor "
    f"EXCLUSIONS: {_unvouched}. They are classified -- role_of calls each of them a "
    f"gate -- and nothing vouches for them: the pin-membership check cannot see "
    f"them, because there is no pin, and the census loop below never walks a name "
    f"that is not in INVENTORY. Put the name in INVENTORY, or write down in "
    f"EXCLUSIONS why it is not a gate",
)


# --------------------------------------------------------------------------
# The same eight, asked a second question.
# --------------------------------------------------------------------------
#
# The check above asks whether each unpinned gate is **vouched for** -- named in a
# table with a reason. It is green, and it is green for all eight, and being green
# is not the same as being able to fail. A gate can sit on disk, be in `INVENTORY`,
# pass the unvouched check, and still return 0 whatever it found. That is
# `redock_benchmark.py`'s shape, found last round, and this file's other six
# mechanisms are all silent about it: the pin loop needs a pin, the census loop
# walks every gate in `INVENTORY` without asking what a site is *for*, and the
# ratchet counts files rather than verdicts.
#
# **So the claim is about the exit code, not about the tables.** For every gate
# that declares no pin: there is a return on the path that decides `main()`'s
# value which is not the constant 0 and which is guarded by a list the gate
# appends its results to. In words -- *a recorded failure can change what this
# process exits with.* Everything else this file asserts about those eight is
# about where their names are written down; this is about what they do.
#
# **The eight are not a list, and the check does not enumerate them.** It walks
# `unpinned` (derived, below) so a gate that gains a pin leaves the set by itself
# and a new one is caught by the same code that caught the last one.
#
# **What it does not claim, and the evidence rather than the assertion.** A
# stronger question -- "can this gate go red?" -- is not answerable statically
# here, and three attempts at it were each wrong in a way that would have
# shipped a false verdict. Counting guarded non-zero returns in every function
# reachable from `main()` reports `redock_benchmark.py` as able to fail, because
# its `dock_into()` helpers return data. Following the return *value* only misses
# the block-structured gates, which call `finish()` as a statement rather than
# returning it, and that pass produced 15 false reds on gates that demonstrably
# fail. So this check states the weaker, sound claim and stops: **the exit is
# reachable from the tally**, which is a precondition for failability and not a
# guarantee of it.
#
# **The polarity is not checked either.** `if not problems: return 1` -- a gate
# that fails when everything is fine -- satisfies this check, because the test
# still names the list. Detecting that needs to know which way round is correct
# for that gate, which is a question about the gate and not about its shape. It
# is caught by mutating the gate's own input and watching the verdict move, which
# is what the mutation table below records.
def _const_fold(node):
    """The constant `node` folds to, or None."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BoolOp):
        vals = [_const_fold(v) for v in node.values]
        want = isinstance(node.op, ast.And)
        for v in vals:                      # False and X, X or True
            if v is (not want):
                return not want
        rest = [v for v in vals if v is want]
        if not rest:
            return want
        if len(rest) == 1:
            return rest[0]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        v = _const_fold(node.operand)
        if isinstance(v, bool):
            return not v
    return None


def _counts_a_filtered_set(node) -> bool:
    """Is `node` a *count* of the members a comprehension kept?

    `sum(1 for t, _n, _d in RESULTS if t == "FAIL")` and
    `len([k for k in declared if k not in whitelisted])`, and nothing else.

    The filter is required, and it is what keeps this from being the widened
    *approver* the other direction of this rule's mutation is about. Half the
    comprehensions in `scripts/` are bulk transformations --
    `others = [n for n in missing if n != sentinel]` in
    `binary_source_parity_check.py`, `imported = [a.name for a in n.names]` in
    `extension_surface_check.py` -- and a name bound to one of those is a list
    being built, not a count of failures. Two further things are deliberately
    not accepted: a bare comprehension with no `sum`/`len` around it, and
    `sum`/`len` over anything that is not a comprehension (`nfail = len(RESULTS)`
    is a total, and treating it as a tally would approve a gate that counts
    every result rather than the bad ones).

    A bare `sum(1 for x in xs)` with no `if` is also refused, and that is a real
    limit rather than a preference: a gate that counted its results without
    filtering them would not be seen here. No gate in `scripts/` is written that
    way -- every one filters on a tag -- and widening it later is a one-line
    change that the mutation table below already covers.
    """
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        return False
    if getattr(node.func, "id", None) not in ("sum", "len"):
        return False
    arg = node.args[0]
    if not isinstance(arg, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return False
    # **No** comprehension node carries `.ifs` -- only `DictComp` does, and a
    # `DictComp` is not accepted here. List, set and generator comprehensions
    # all keep their filters on the `comprehension` nodes, so one walk reads
    # all three and a multi-clause `sum(1 for a in x if a for b in y if b)` is
    # answered the same way. Two earlier versions of this arm branched on the
    # node type and asked `.ifs` of a `ListComp` and of a `SetComp`; both
    # raised `AttributeError`, and the fixture set is what caught the second
    # one -- `scoring_cross_check.py` only proves the `SetComp` half.
    return any(c.ifs for c in arg.generators)


def _exit_reachable_from_tally(src: str, fname: str = "main",
                               depth: int = 0, seen: frozenset = frozenset(),
                               comprehension_tally: bool = True):
    """How many returns on `fname`'s exit path name a result list.

    `comprehension_tally=False` runs the identical walk with the
    comprehension arm of the tally scan switched off. It exists so that "is
    this arm doing any work?" is a *measurement* -- the same gates, the same
    tree, one flag apart -- rather than an assertion about a number somebody
    wrote down. See the check named "the comprehension arm earns its verdict".
    """
    tree = ast.parse(src)
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n
    fns = {f.name: f for f in tree.body
           if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if fname in fns:
        if fname in seen or depth > 4:
            return 0
        seen = seen | {fname}
        fn = fns[fname]
    elif depth == 0:
        #: **A gate whose entry point is not `main()` used to read as unable
        #: to fail, for a reason that had nothing to do with tallies.** The
        #: walk used to open with `if fname not in fns: return 0`, so a module
        #: that does its work in top-level statements and ends in
        #: `sys.exit(...)` -- no `main()` to find -- was scored 0 without a
        #: single line of it being read. That is a false *indictment* of the
        #: same kind the comprehension arm below fixes, and it is invisible
        #: from the outside: the 0 is indistinguishable from a gate that
        #: genuinely cannot fail. Three gates in this repository are written
        #: that way, and this file is one of them.
        #:
        #: The fallback is deliberately confined to `depth == 0`. Recursion
        #: into a helper keeps the old `return 0`: a missing *helper* means
        #: the call site's value is not this function's return value, and
        #: widening that would invent exits rather than find them.
        fn = tree
    else:
        return 0

    tallies = set()
    for n in ast.walk(tree):
        tgt = None
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "append":
            tgt = n.func.value
        elif isinstance(n, ast.AugAssign):
            tgt = n.target
        elif isinstance(n, ast.Assign):
            # the second spelling of a tally, and it was missing: a gate that
            # counts its own results with a comprehension
            # (`nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")`) is
            # recording the same thing as a gate that appends to a list, and
            # before this arm it read as having no tally at all.
            #
            #: The set this arm rescues is **measured, not listed**, by the
            #: `comprehension_tally=False` pass in the check named "the
            #: comprehension arm earns its verdict": ten of the gates
            #: `role_of` finds change number when it is switched off, and
            #: seven of those go from 0 to positive. A 0 there is a false
            #: *indictment* -- a gate that counts its own results correctly
            #: being called unable to fail.
            #:
            #: **An earlier version of this comment named the seven gates and
            #: named one of them wrong.** It listed
            #: `release_tree_parity_check.py`, whose `npass, nfail, nnot =
            #: _tally()` is a *tuple* target over a *call*: the arm wants a
            #: `sum`/`len` over a comprehension and a single `Name` target, so
            #: it declines, and the file reads 0 with the arm and 0 without it.
            #: The gate missing from that list is `energy_terms_check.py`. A
            #: name written beside a rule is a claim nobody re-measures.
            if comprehension_tally and _counts_a_filtered_set(n.value):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        tallies.add(t.id)
            continue
        if isinstance(tgt, ast.Name):
            tallies.add(tgt.id)
    # a list the function declares for itself -- `check_repo_docs.py` keeps its
    # `problems` local to main(), and it is the one gate of the eight that
    # defines no `check()` at all
    local = set(tallies)
    for n in ast.walk(fn):
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and isinstance(n.value, (ast.List, ast.Dict, ast.Set)):
            local.add(n.target.id)
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and isinstance(n.value, (ast.List, ast.Dict, ast.Set)):
                    local.add(t.id)

    def _names(n):
        return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}

    def _if_chain(node):
        cur, out = node, []
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.If):
                out.append(cur.test)
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
        return out

    got = 0
    for r in ast.walk(fn):
        if not isinstance(r, ast.Return) or r.value is None:
            continue
        v = r.value
        if isinstance(v, ast.Constant) and v.value == 0:
            continue
        if isinstance(v, ast.Call) and getattr(v.func, "id", None) in fns:
            # the helper's *value* is the return value, so its exits are ours
            # -- and `comprehension_tally` has to travel with the recursion.
            #: **It did not, and that is the second bug this flag found.** The
            #: first version of this signature added the parameter and used it
            #: here, but the recursive call left it at its default, so an
            #: "arm off" measurement was arm-on at every level below the first
            #: -- and the gates whose comprehensions live in a helper rather
            #: than in `main()` were the ones it mis-measured. Five of the
            #: seven the arm rescues are like that: `binary_source_parity_
            #: check.py` counts inside `finish()`, `extension_surface_
            #: check.py` inside `main()` but reached through one. Read as
            #: "the arm rescues two gates" it was a *smaller* claim than the
            #: truth and it would have been typed into a check as one.
            #: A measurement switch that does not reach the code it is
            #: measuring is not a measurement.
            got += _exit_reachable_from_tally(src, v.func.id, depth + 1, seen,
                                              comprehension_tally)
            continue
        if _names(v) & local:
            got += 1
            continue
        for t in _if_chain(r):
            # a test that folds to a constant is not a test: `if False and
            # problems:` still names the list, and without this fold the check
            # does not bite on the one gate of the eight that guards a local
            if _const_fold(t) is False:
                break
            if _names(t) & local:
                got += 1
                break
    return got


#: Gates whose exit reachability this run could not measure, as
#: `{name: reason}`.
#:
#: **A gate this auditor cannot read has demonstrated nothing** -- not that it
#: can fail, and not that it cannot. Those are different claims and only the
#: second is an accusation, so an unreadable gate must be neither scored as a
#: `0` nor waved through as though it had been.
#:
#: **Why a red, and not a `skip()`.** A skip that fires during a concurrent edit
#: becomes permanent the moment nobody is looking: the tally keeps reaching the
#: declared total, the file is never re-read, and the one gate that quietly
#: stopped being audited is indistinguishable from the ones that were still
#: audited. That is not a hypothetical here. An earlier gate in this repository
#: died at check 19 of 124 on a non-ASCII print and survived only because an
#: unrelated wrapper injected `PYTHONIOENCODING` -- a measurement that can
#: expire is worse than one that is loudly wrong, because the expiry is
#: invisible and the wrongness is not.
_EXIT_UNAUDITABLE: dict[str, str] = {}


def _record_unauditable(name: str, why: str) -> None:
    """Note a gate whose exit reachability this run could not measure.

    **The no-double-count property is structural rather than checked.** Four
    call sites ask for a gate's exit reachability, and a file that is being
    written right now can fail to parse at some of them and not others. A list
    would accumulate one entry per failed request; a dict keyed on the name
    keeps the first reason and drops the rest, so one unreadable file is one
    finding no matter how many times it is asked about.
    """
    _EXIT_UNAUDITABLE.setdefault(name, why)


def _is_transient(name: str) -> bool:
    """Is this gate's unreadability expected to clear on a later run?

    **The distinction is whether anything is writing the file right now**, and it
    is `_is_in_flight` again rather than a second idea in parallel with it --
    that function already answers "was this saved during this run" from the
    filesystem, which is the only form of the question that has a measurable
    answer.

    The two cases are indistinguishable from inside a failed parse, and they mean
    opposite things to whoever reads the red:

    * *in flight* -- the file was written during this run, so the bytes on disk
      are a partial save. The failure is **transient**: the editor settles, and
      re-running measures the gate. The advice is to re-run.
    * *not in flight* -- nothing has touched the file since the run began, so
      the bytes on disk *are* the file. The failure is **not transient** and
      will be reported identically on every future run until somebody repairs
      it, and "just re-run it" is the wrong advice to give.

    A reader who cannot tell these two apart cannot act on either, which is why
    the message says which one it is rather than only that it happened.
    """
    return _is_in_flight(SCRIPTS / name)


#: Every gate's source, read once.
#:
#: The four call sites below used to each read and re-parse every gate on its
#: own. That was not only wasteful -- it was a way to get **two different
#: answers for one file inside one run**: a gate saved mid-audit could parse at
#: the first call site and fail to parse at the third, and the run would print
#: both numbers without noticing that they were supposed to be the same
#: measurement. Reading once and consulting the result is what makes the
#: no-double-count property hold for the good case as well as the bad one.
_GATE_SRC: dict[str, str] = {}
for _n in _gates:
    try:
        _GATE_SRC[_n] = read(SCRIPTS / _n)
    except (OSError, UnicodeDecodeError) as _exc:
        _record_unauditable(_n, "it cannot be read: "
                            f"{type(_exc).__name__}: {_exc}")

#: `name` -> exit reachability, for the gates this run could both read and
#: parse. **An absent name is not a zero.** It is in `_EXIT_UNAUDITABLE`
#: instead, and this is the whole reason the two are kept apart: the walk's `0`
#: means "read it, and it cannot fail", so folding an unreadable file into it
#: would accuse a gate of being incapable on the strength of having never been
#: looked at. Every consumer below uses `.get()`, so a missing key degrades to
#: "not measured" rather than to `0`.
_exit_reach: dict[str, int] = {}
for _n, _s in _GATE_SRC.items():
    try:
        _exit_reach[_n] = _exit_reachable_from_tally(_s)
    except (SyntaxError, ValueError) as _exc:
        # `ast.parse` reports a null byte as `ValueError`, not `SyntaxError`, so
        # catching only one of them leaves a hole exactly the size of a file
        # saved by a tool that pads its buffer.
        _record_unauditable(_n, "it does not parse: "
                            f"{type(_exc).__name__}: {_exc}")


def _reach_text(n: str) -> str:
    """One gate's exit reachability, as the run prints it.

    A gate that was not measured is printed as such rather than as `0`, so the
    per-gate listing cannot quietly present an unreadable file as a measured
    zero -- which is the false indictment the two dicts exist to keep apart.
    """
    v = _exit_reach.get(n)
    if v is None:
        return f"{n} (NOT MEASURED -- see the unauditable-gate check below)"
    return f"{n} ({v})"


def _unauditable_detail() -> str:
    """The named, per-file red for every gate this run could not measure."""
    if not _EXIT_UNAUDITABLE:
        return (f"all {len(_gates)} gate(s) in scripts/ were read and parsed by "
                f"this run, so every number above is a measurement rather than "
                f"an absence of a complaint. This is a red and not a skip "
                f"because a gate that cannot be read has demonstrated nothing: "
                f"not that it can fail, and not that it cannot, and a skip that "
                f"fired during a concurrent save would outlive the save and "
                f"leave the tally reaching its declared total with one gate no "
                f"longer audited at all")
    parts = []
    for n in sorted(_EXIT_UNAUDITABLE):
        flying = _is_transient(n)
        parts.append(
            f"{n}: {_EXIT_UNAUDITABLE[n]} -- and this file "
            + ("WAS WRITTEN DURING THIS RUN, so the bytes on disk are a "
               "partial save: the failure is TRANSIENT and re-running the "
               "gate once the editor settles will measure it"
               if flying else
               "has not been touched since this run began, so the bytes on "
               "disk ARE the file: this is NOT TRANSIENT, it will be reported "
               "identically on every future run until somebody repairs it, and "
               "re-running the gate will not clear it"))
    return (
        f"{len(_EXIT_UNAUDITABLE)} of {len(_gates)} gate(s) could not be read "
        f"or parsed, so their exit reachability is UNKNOWN -- not zero. A 0 "
        f"would accuse a gate of being unable to fail on the strength of never "
        f"having been looked at, and the two are indistinguishable from the "
        f"outside. Each is named, with the error and whether it is transient: "
        + "; ".join(parts))


check(
    not _EXIT_UNAUDITABLE,
    "every gate in scripts/ could be read and parsed, so its exit "
    "reachability was measured rather than assumed",
    _unauditable_detail(),
)


section("an unpinned gate can still be unable to fail")

#: Which of the unpinned gates have no path from a recorded result to a
#: non-zero exit. **Derived, not typed** -- this is the whole point.
_cannot_fail = [n for n in _unpinned_gates
                if _exit_reach.get(n) == 0]
check(
    not _cannot_fail,
    "every unpinned gate can turn a recorded failure into a non-zero exit",
    f"{len(_unpinned_gates)} unpinned gate(s), "
    f"{len(_unpinned_gates) - len(_cannot_fail)} with an exit reachable from "
    f"their own result list: "
    + "; ".join(_reach_text(n)
                for n in _unpinned_gates)
    + ". The check above asks whether these eight are *named* somewhere; this one "
    "asks whether a wrong input can move their verdict, which is the question a "
    "membership list cannot answer and the one `redock_benchmark.py` failed. It "
    "does not claim they *do* go red -- only that the exit code is reachable from "
    "the tally, which is a precondition for it. Give one of them a pin and it "
    "leaves this set by itself; nothing here has to be edited"
    if not _cannot_fail else
    f"{len(_cannot_fail)} unpinned gate(s) declare no pin and cannot turn a "
    f"recorded failure into a non-zero exit: {_cannot_fail}. A gate in this "
    f"state reports success whatever it found, which is a number nobody reads. "
    f"Either give it a pin and a decision that consults it, or -- if the exit is "
    f"genuinely unreachable and always 0 -- say so in EXCLUSIONS and stop calling "
    f"it a gate",
)

# ==========================================================================
# ==========================================================================
#: **The gates whose exit-reachability this walk cannot establish, and why.**
#:
#: Typed, deliberately, and that is the one thing this file normally refuses
#: to do. The alternative -- forcing each of them to a number -- would mean
#: widening the walk until it produces an answer, and an answer the walk had
#: to be bent to produce is not evidence. A name and a reason is weaker than
#: a number and it is honest about being weaker. The reasons name the line
#: and the construct, so the next reader can check the claim rather than
#: trust it, and the staleness test below makes sure an entry cannot outlive
#: the hole it recorded.
#:
#: **All three of these gates do exit non-zero when a check fails.** They are
#: not the `redock_benchmark.py` failure and they are not excused by this
#: table; they are a limit of the walk, and the difference matters more than
#: the count. The claim being made is "`exit_reach` could not be established
#: for this file", never "this file cannot fail".
UNPROVABLE_BY_AST = {
    "pose_trust_check.py":
        "`code, why = verdict_of(...)` -- a tuple target, so the tally scan "
        "never registers `code` as a tally name, and the file's single exit "
        "is `return code`. Reading it needs tuple targets.",
    "release_tree_parity_check.py":
        "`npass, nfail, nnot = _tally()` -- a tuple target again; the "
        "comprehensions it counts live in `_tally()`'s `return`, which is "
        "not an assignment. Its three exits are the constants EXIT_FAILED / "
        "EXIT_INCOMPLETE / EXIT_OK, which name no tally. Reading it needs "
        "tuple targets and a tally returned as a value.",
    "viewport_framing_check.py":
        "`nfail = _count(...) + _count(...)` -- a `BinOp`, not a `Call`, so "
        "the comprehension arm declines it; the comprehension it is built "
        "from is a `return` inside `_count()`. Its exits are EXIT_* "
        "constants. Reading it needs arithmetic over two call results.",
}


section("a zero here means one of two things, and the run says which")

#: **A `0` from this walk was two different findings wearing one number, and
#: only one of them was a defect.** "This gate has no path from its recorded
#: results to a non-zero exit" and "this gate is written in a shape this walk
#: cannot read" are opposites in their consequences -- the first is a gate that
#: reports success whatever it found, the second is an absence of evidence --
#: and they were both reported as `0`. A reader who cannot tell them apart
#: draws the wrong conclusion from each: they either indict a sound gate, or
#: they acquit a broken one. **This is the same asymmetry the provenance
#: appendix records**, where a claim that cannot be derived is named rather
#: than counted as a fact, and it is handled the same way here.
#:
#: So a `0` is now split by name, and the two lists are printed. The split is:
#:
#:   * **proved unable to fail** -- the walk read the gate, found no exit
#:     reachable from its tally, and that is a red.
#:   * **could not be analysed** -- the gate is in `UNPROVABLE_BY_AST` below,
#:     with the reason its shape defeats the walk. Counted, printed, exit 0.
#:
#: The three entries are not a list of gates somebody gave up on. Each is a
#: specific construct, and the reason names the line:
#:
#:   * `release_tree_parity_check.py` binds `npass, nfail, nnot = _tally()`.
#:     The tally scan wants a single `Name` target, and a tuple target is not
#:     one, so the three names never become tallies and `if nfail:` reads as a
#:     test of something the walk never registered. Its three returns are the
#:     module-level constants `EXIT_FAILED` / `EXIT_INCOMPLETE` / `EXIT_OK`,
#:     which name no tally either. **It does exit non-zero on a failure** --
#:     the walk simply cannot see it, which is the whole content of the word
#:     "unprovable".
#:   * `viewport_framing_check.py` binds `nfail = _count(...) + _count(...)`.
#:     A `BinOp` is not a `Call`, so the comprehension arm declines it; the
#:     comprehension it is *built from* lives in the body of `_count()` as a
#:     `return`, which is not an `Assign`. Its returns are `EXIT_*` constants.
#:   * `pose_trust_check.py` binds `code, why = verdict_of(...)` -- the tuple
#:     case again -- and its single exit is `return code`, a `Name` bound by
#:     that same tuple unpack.
#:
#: **Reading `0` as "proved unable to fail" for any of these three would be
#: wrong, and the mutation below is what proves it rather than asserts it.**
#: Weakening `_counts_a_filtered_set` to accept arbitrary assignments pushes
#: `viewport_framing_check.py` from 0 to 4 -- not because the gate changed, but
#: because the walk started counting a `BinOp` of calls as a tally. The two
#: readings are distinguishable, and which one a given gate is in is a fact
#: this run prints.
_zero = sorted(n for n, v in _exit_reach.items() if v == 0)
_proved_cannot_fail = [n for n in _zero if n not in UNPROVABLE_BY_AST]
_unproven = [n for n in _zero if n in UNPROVABLE_BY_AST]

print("  exit-reachability, read from source by the walk above:")
print("    %d gate(s) have an exit reachable from their own tally" % (len(_gates) - len(_zero)))
print("    %d PROVED UNABLE TO FAIL (a red, named above): %s"
      % (len(_proved_cannot_fail), _proved_cannot_fail or "none"))
print("    %d COULD NOT BE ANALYSED (named, counted, not a finding): %s"
      % (len(_unproven), _unproven or "none"))
for _n in _unproven:
    print("      %-34s %s" % (_n, UNPROVABLE_BY_AST[_n]))
if not _zero:
    print("    no gate is in either class: every one this walk could read, it read")
else:
    print("    a 0 is one of the two lines above and nothing else -- it is never a "
          "bare 0, and the two are not the same finding")

check(
    not _proved_cannot_fail,
    "every gate that reads 0 is a gate this file can name as unprovable, so a "
    "0 is never an unexplained one",
    f"{len(_gates)} gate(s) walked, {len(_gates) - len(_zero)} with an exit "
    f"reachable from their own tally, {len(_unproven)} named in "
    f"UNPROVABLE_BY_AST as not analysable, and {len(_proved_cannot_fail)} "
    f"neither: {_proved_cannot_fail}. A gate in that last group has been read "
    f"and found to have no exit reachable from its results, which is the "
    f"`redock_benchmark.py` failure and a red. It is not excused by appearing "
    f"in UNPROVABLE_BY_AST, because it is not in it"
    if not _proved_cannot_fail else
    f"{len(_proved_cannot_fail)} gate(s) read as unable to fail and are not in "
    f"UNPROVABLE_BY_AST: {_proved_cannot_fail}. Either they have a real path "
    f"to a non-zero exit that this walk cannot see -- in which case the "
    f"construct that hides it belongs in UNPROVABLE_BY_AST with a reason "
    f"naming the line -- or they have none, which is a gate that reports "
    f"success whatever it found",
)

# The staleness test the debt tables in this file kept needing and did not
# have. Without it, fixing one of these gates is silently free: the entry sits
# in a dict describing a hole that is no longer there, and the next reader
# counts a gate as unprovable when it is merely unwalked. This is the
# `SITE_INVENTORY` shape -- a typed table outliving the fact it recorded --
# and it is the reason the check above is a set comparison in both directions.
check(
    set(UNPROVABLE_BY_AST) == set(_unproven),
    "every gate named as unprovable is still one this walk cannot read",
    f"UNPROVABLE_BY_AST holds {len(UNPROVABLE_BY_AST)} name(s); "
    f"{len(_unproven)} gate(s) currently read 0. "
    f"Stale: {sorted(set(UNPROVABLE_BY_AST) - set(_unproven)) or 'none'}. "
    f"Unnamed: {sorted(set(_unproven) - set(UNPROVABLE_BY_AST)) or 'none'}. "
    f"Deleting an entry is a real edit to a real gate and has to be a red "
    f"here, not a quiet improvement nobody notices"
    if set(UNPROVABLE_BY_AST) == set(_unproven) else
    f"UNPROVABLE_BY_AST says {sorted(UNPROVABLE_BY_AST)} but the walk finds "
    f"0 for {sorted(_unproven)}. An entry the walk can now read describes a "
    f"hole that has been closed; a 0 with no entry is an unexplained one. "
    f"Reconcile the two",
)

# ==========================================================================
section("the comprehension arm earns its verdict, and does not earn too much")

#: **A rule nobody measures is a rule the next person deletes.** The arm that
#: reads `nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")` as a tally
#: was added to stop seven gates being falsely indicted, and the only
#: consumer of the number it feeds is `_cannot_fail` -- which walks the eight
#: *unpinned* gates, and every one of those is >= 1 with the arm and >= 1
#: without it. So the arm could have been deleted, or inverted, or widened to
#: accept every assignment in the repository, and **nothing in this file would
#: have gone red**. That is the gap this section closes, and it is the reason
#: the walk carries a `comprehension_tally` flag at all: the claim is
#: measured by running the same walk twice over the same gates.
#:
#: The claim is deliberately **two-sided**, because the one-sided version is
#: the decoration this file has spent its length removing:
#:
#:   1. the arm must be *doing* something -- at least one gate is non-zero
#:      only because of it. Remove the arm and this is false.
#:   2. the arm must not be doing it *on nothing* -- every gate it rescues
#:      must really contain the construct it claims to recognise. Widen the
#:      arm to accept arbitrary assignments and this is false.
#:
#: Clause 2 needs an oracle that does not share code with the rule under test,
#: or it could not disagree with it. `_literal_filtered_tally` is written
#: separately and is deliberately cruder: it ignores call arity, keywords and
#: the DictComp case, so it is an *upper* bound on "this file contains the
#: shape", which is the direction a subset test needs. Measured effect of
#: each mutation, both run against this check rather than argued about:
#:
#:   arm removed              rescued 7 -> 0     clause 1 red
#:   accepts any assignment   rescued 7 -> 9     clause 2 red, on ONE gate
#:
#: **That last line is the honest limit of this check and it is stated here
#: rather than left to be discovered.** The widen direction rests on a single
#: gate -- `viewport_framing_check.py`, 0 -> 4 -- because it is the only gate
#: in the tree whose `BinOp`-of-calls tally is invisible to the oracle but
#: visible to a widened arm. `pose_trust_check.py` also moves under that
#: mutation (0 -> 6) and is *not* caught, because the oracle is right that the
#: file does contain a filtered tally, just not one on its exit path. A
#: tighter oracle would need to re-derive which function the walk entered,
#: which is the rule itself, and an oracle that re-derives the rule under test
#: cannot contradict it. One discriminating gate is a thin margin and it is
#: named as one; clause 1 carries the weight, clause 2 stops the specific
#: over-widening that actually happened while this was written.
def _literal_filtered_tally(src: str) -> bool:
    """Does this source contain `name = sum(1 for ... if ...)`?

    The oracle for clause 2 above, written independently of
    `_counts_a_filtered_set` on purpose. Deliberately an upper bound: it does
    not check arity, keywords, or that the comprehension is not a `DictComp`,
    so it can say "yes" for a construct the arm would decline. That is the
    safe direction -- a false yes weakens the subset test, a false no would
    invent a red.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        v = node.value
        if not (isinstance(v, ast.Call)
                and getattr(v.func, "id", None) in ("sum", "len")
                and len(v.args) == 1):
            continue
        arg = v.args[0]
        if not isinstance(arg, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            continue
        if any(c.ifs for c in arg.generators):
            return True
    return False


_arm_off = {n: _exit_reachable_from_tally(src,
                                         comprehension_tally=False)
            for n, src in _GATE_SRC.items() if n in _exit_reach}
_rescued_by_arm = sorted(n for n in _gates
                         if _exit_reach.get(n, 0) > 0 and _arm_off[n] == 0)
_unearned = [n for n in _rescued_by_arm
             if not _literal_filtered_tally(_GATE_SRC[n])]
#: Walked over `_exit_reach`, **not** over `_gates`. The two are not the same
#: set, and assuming they are is how this line used to raise `KeyError` on a
#: gate that was unreadable: `_arm_off` is built from the gates that could be
#: read, so a gate in `_gates` and absent from `_exit_reach` has no second
#: measurement to compare against. There is nothing to compare, which is a fact
#: the unauditable-gate check above already reports -- a second crash here
#: would only hide it behind a traceback.
_moved_by_arm = sorted(n for n in _exit_reach if _exit_reach[n] != _arm_off[n])

print("  the comprehension arm, measured by running the walk twice:")
print("    %d gate(s) change number when it is switched off: %s"
      % (len(_moved_by_arm), _moved_by_arm))
print("    %d are non-zero ONLY because of it: %s"
      % (len(_rescued_by_arm), _rescued_by_arm or "none -- the arm does nothing"))

check(
    bool(_rescued_by_arm) and not _unearned,
    "the comprehension arm earns its verdict on real gates and reaches "
    "nothing it has not earned",
    f"{len(_gates)} gate(s) walked twice, once with the arm and once without. "
    f"{len(_moved_by_arm)} change number: {_moved_by_arm}. "
    f"{len(_rescued_by_arm)} are non-zero only because of the arm: "
    f"{_rescued_by_arm}. Without this check the arm could be deleted and "
    f"nothing here would notice, because the only consumer of the number is "
    f"the unpinned check above and all eight of those are >= 1 either way"
    if bool(_rescued_by_arm) and not _unearned else
    (f"the comprehension arm rescues NO gate: every one of the "
     f"{len(_moved_by_arm)} it moves is positive without it too, so the arm is "
     f"dead weight that reads as protection. Remove it, or find the gate it "
     f"was written for" if not _rescued_by_arm else
     f"{len(_unearned)} gate(s) are non-zero only because of the arm and do "
     f"not contain the construct it claims to recognise: {_unearned}. The arm "
     f"is reaching further than its own rule allows, which approves exits it "
     f"has not read -- the widening that clause 2 of the section comment "
     f"names, and the one mutation that moves this number from 7 to 9")
)

# ==========================================================================
section("a pin is a promise, and EXCLUSIONS does not discharge one")

# The check above is a disjunction, and the second arm -- "or somebody wrote
# down why it is not a gate" -- is a *classification*. It says what a file
# decides. It says nothing about the numbers the file writes down, and that is
# where the next hole was, two entries down the same table.
#
# **The obligation, stated so that it is answerable from the tree rather than
# from a list of names.** A number written as `EXPECTED_CHECKS = n` reads as a
# promise that the file records exactly n results. For that to be more than
# decoration, something the run executes must be able to go differently because
# the number was not what the file counted. `INVENTORY` gets that for free: the
# per-pin loop below asserts "<name>'s pin is actually compared, not just
# declared" for every gate it walks, and it walks every gate in `INVENTORY`.
# **It walks nothing in `EXCLUSIONS`.** So the obligation is defined by the
# *declaration* -- a file that writes `EXPECTED_CHECKS = n` owes a comparison --
# and the walk is defined by a *table*. A file can satisfy the table and break
# the declaration, and did: `qt_gl_probe.py` and `probe_painted_frame.py` both
# sit in `EXCLUSIONS` and both carry a pin, and they are not in the same state.
#
# So the rule is, for any file `n` in `EXCLUSIONS` that declares a pin:
#
#     n in INVENTORY   the per-pin loop verified it -- and, by the check
#                      below, this is impossible, so it is not a way out
#      or the file itself compares its own pin in a decision it executes
#
# **The first arm is dead by construction, and the check below is what makes it
# dead rather than merely empty.** `role_of` consults `EXCLUSIONS` first, so a
# file in both tables is classified `exempt` while the census loop still walks it
# as inventory; the two tables would be asserting opposite things about one file
# and no existing check would have said so. Adding a name to `INVENTORY` to make
# a red below go away would therefore be a reclassification, not a repair, and it
# is the cheapest possible way to turn this check into decoration. Stating that
# in a comment and not testing it is the `SITE_INVENTORY` hole this file was
# written to end, so it is tested.


def _pin_is_compared(path: Path):
    """(compared?, where) -- is this file's own pin read by a *decision*?

    The property is deliberately not "the number appears in the file". It is:
    something the run executes could have gone differently because the number was
    not what the file counted. That is narrower than a text search and stronger
    than a search for the letters, and it is the entire difference between the
    two files above -- both print their pin in their summary line, and only one
    of them can fail because of it.

    Measured from the tree. A *read* of the constant is an `ast.Name` in load
    context, which cannot be the assignment itself, and it is a **decision** if
    any enclosing node is a `Compare`, the `test` of an `If`/`While`, or the
    `test` of an `Assert`. Everything else is a **render**: inside a `print`, an
    f-string, a message passed as a detail. `EXPECTED_CHECKS - 1` on the right of
    a `==` is a decision, and so is the whole `npass + nfail + nskip == len(RESULTS)
    == EXPECTED_CHECKS - 1` chain, because the name is a descendant of the
    `Compare` however deep the arithmetic is.

    **A render is not a weaker comparison, it is the absence of one.** That is
    the claim, and it is worth separating from the tempting alternative
    implementation -- look for the constant on a line that also contains a
    comparison operator, which is what the per-pin loop below does -- because the
    two disagree, and a mutation is what says which is right.

    **A mutation that survived, and it is mine.** Rewriting
    `probe_painted_frame.py`'s comparison as a rendered f-string --
    `f"{len(RESULTS) != EXPECTED_CHECKS and EXPECTED_CHECKS or ''}"`, with the
    `return EXIT_FAILED` deleted -- leaves this check **green**. The file still
    contains an `ast.Compare` with the constant in it, so the tree measurement
    says "decided", and the file still cannot fail: the boolean is computed and
    thrown away. Telling a *computed* comparison from a *consumed* one is a
    dataflow question and not a syntactic one, and no amount of walking parents
    answers it. Recorded as a known false negative rather than left to be
    discovered, because a check with a documented hole is a debt and a check with
    an undocumented one is a lie. The first version of this docstring claimed the
    tree version beat the text version on exactly that mutation; the mutation says
    otherwise and the sentence was wrong.

    **The mutation that does separate them.** Delete the comparison (so the file
    genuinely has no decision) *and* put an operator on the render line where it
    belongs to a different expression --
    `f"...(expected {EXPECTED_CHECKS}; differs if nfail > 0)"`. The line now
    carries the constant *and* a comparison operator, so the line-text version
    passes, and the constant is a `FormatValue` inside a `print` with no `Compare`
    above it, so this one goes red. That is the whole argument for reading the
    tree: the text version is satisfied by a line, and a line is not an
    obligation.

    **What this cannot see, stated because this file has already paid for the
    same limitation twice.** A comparison inside a function nothing calls is
    counted here, a comparison whose boolean is discarded is counted here (the
    survivor above), and so is a comparison on a branch a machine fact can skip.
    Catching any of them needs a call graph or a dataflow pass -- the same wall
    `_sites_of` documents for guards. So this is a lower bound and the word is
    doing work: it proves the pin is read by a decision *somewhere in the file*,
    not that the decision is reached, or obeyed, on every run. For the one file
    that passes here, the comparison is the third statement of `main()` after four
    unconditional block calls and its result is returned, so it is reached and
    obeyed on every path that reaches the summary at all. That was read by hand
    once and is recorded here because no tool in this file can be asked for it.
    """
    try:
        tree = ast.parse(read(path))
    except SyntaxError as exc:
        return (False, f"it does not parse, so nothing can be read of it: {exc}")
    parent_of: dict = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent_of[c] = n
    decisions: list[int] = []
    renders: list[int] = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Name) and n.id == "EXPECTED_CHECKS"
                and isinstance(n.ctx, ast.Load)):
            continue
        cur, decided = n, False
        while cur in parent_of:
            p = parent_of[cur]
            if isinstance(p, ast.Compare):
                decided = True
                break
            if isinstance(p, (ast.If, ast.While)) and cur is p.test:
                decided = True
                break
            if isinstance(p, ast.Assert) and cur is p.test:
                decided = True
                break
            cur = p
        (decisions if decided else renders).append(n.lineno)
    where = (f"read by a decision on line(s) {sorted(decisions)}"
             if decisions else
             f"only ever rendered, on line(s) {sorted(renders)} -- no decision "
             f"anywhere in the file reads it")
    return (bool(decisions), where)


#: EXCLUSIONS entries whose pin is rendered and never compared, named rather
#: than tolerated. A ratchet, like the other two in this file, and unlike
#: `UNNAMED_SKIP_TALLIES` it has a working staleness test: "unverified" is a
#: measurable property of the source rather than a relationship between a
#: summary and a counter, so an entry that has stopped describing a real gap is
#: visible rather than indistinguishable from the debt it replaced.
#:
#: **One entry, and the two files it is about are not in the same state.**
#: `probe_painted_frame.py` came out of the measurement above *verified*: its pin
#: is read by a decision at its line 642, `if len(RESULTS) != EXPECTED_CHECKS:
#: return EXIT_FAILED`, which is the shape every gate in `INVENTORY` is held to
#: by the per-pin loop. `qt_gl_probe.py` came out of it with **zero** decisions
#: and two renders, and that is the finding rather than a formality: its
#: `EXPECTED_CHECKS = 6` is *not a wrong number* -- it has six `check_*` blocks,
#: each called once from `main()`, so a correct run does record six -- it is a
#: number that nothing can contradict.
#:
#: **The mutation matrix, because a check that cannot go red is worse than no
#: check.** Eight mutations, all applied to files this one does not own, all run
#: against a *copy* of `scripts/` -- the auditor derives `ROOT` from `__file__`,
#: so a copy at `<root>/scripts/` measures itself and never the real tree. "Not
#: equivalent" below means the mutated program would behave differently if it
#: were run; "equivalent" means the mutation is a no-op for the program and only
#: the declaration changed, which is the case this check is actually about.
#:
#:   M0  control, unmutated copy .............. 0 new reds (the 8 drifting gates
#:       owned by other workers are present in the control too, and are not
#:       counted as signal)
#:   M1  `probe_painted_frame.py`: delete the pin comparison ......... **RED**
#:       (ratchet + staleness). Not equivalent: a run recording the wrong
#:       number of results now returns `EXIT_OK` instead of `EXIT_FAILED`.
#:   M2  the same, rewritten as a rendered f-string ................... **GREEN
#:       -- survived.** Equivalent for *this* measurement and not for the
#:       program: the tree still holds a `Compare`, the boolean is discarded,
#:       and the file still cannot fail. Recorded as a known false negative at
#:       `_pin_is_compared`, and it is why the first draft of that docstring --
#:       which claimed this mutation was the one the text version lost -- was
#:       wrong and was corrected.
#:   M2b M1 plus a comparison operator on the render line, belonging to a
#:       different expression ......................................... **RED**
#:       (ratchet + staleness). Equivalent: comments and a removed block, the
#:       program is unchanged. This is the separator that shows the tree
#:       measurement is stronger than the line-text one the per-pin loop uses.
#:   M3  `qt_gl_probe.py`: add the comparison it lacks ................. **RED**
#:       (staleness: "an entry that has stopped being true"). Not
#:       equivalent -- the file can now fail on a wrong total, which is the
#:       fix. The red is the check asking to be told, not a complaint.
#:       **No longer a mutation: this is the edit below, applied to the real
#:       file**, and the red it produced was exactly this one.
#:   M4  `qt_gl_probe.py`: rename `EXPECTED_CHECKS` -> `EXPECTED_TOTAL` ... **RED**
#:       (staleness). **Equivalent**: a module-private constant's name is not
#:       observable, so the output and the exit code are identical. The check
#:       moves anyway, and that is correct rather than a false positive: what it
#:       guards is the *declaration*, and a file whose pin is no longer declared
#:       is a file the ledger has stopped being able to reason about. This is the
#:       case where a mutation surviving would have been a real defect, so it is
#:       worth being explicit that the two axes differ: M2 is behaviour-changing
#:       and check-silent, M4 is behaviour-neutral and check-loud.
#:   M5  `check_scripts_declare.py`: put `qt_gl_probe.py` in `INVENTORY` as
#:       well .......................................................... **RED x6**
#:       (disjointness, "pin says where the number came from", "pin is actually
#:       compared", the census, one-source-per-gate, and this file's own total).
#:       This is the answer to "is reclassifying a way out?" and it is no, twice
#:       over: the disjointness check below catches the contradiction, and the
#:       pre-existing per-pin loop *already* caught the unverified pin the moment
#:       the file became inventory -- including by its own line-text heuristic,
#:       which does not fire on `qt_gl_probe.py` today only because it is not
#:       walked. The first arm of the disjunction is not merely blocked, it was
#:       never viable.
#:   M6  `extension_surface_check.py`: declared `sites: 2+10` -> `2+9` .... **RED**
#:       (census drift). Equivalent: a comment.
#:   M7  the same file, `guards:` digest zeroed ........................ **RED**
#:       ("same census, different guards"). Equivalent: a comment. M6 and M7
#:       together are the argument for the digest half existing.
#:   M8  `wheel_payload_check.py`: delete the `GATE-DECLARE` block ........ **RED**
#:       with `NOT DECLARED ANYWHERE` for that file -- the original four reds,
#:       returned exactly. Equivalent: comments. This is the control that says
#:       the block is what closed the hole, and not the pin it sits next to.
#: **Empty, and the cap under it is 0.**
#:
#: It held one name. `qt_gl_probe.py` carried `EXPECTED_CHECKS = 6` -- the right
#: number, six `check_*` blocks each called once from `main()` -- and read it in
#: exactly two places, both f-strings inside `print()`, so a run that recorded
#: three results would have printed `3 checks (expected 6)` and returned the
#: verdict its measurement produced. The file now compares its own pin in a
#: decision it executes and `_pin_is_compared` measures it as compared rather
#: than as rendered, so the entry stopped being true.
#:
#: **It was not deleted quietly.** The staleness check below is what noticed,
#: because a repaired file is the one case where a name in this table is
#: describing a defect that no longer exists -- a paragraph in this file about a
#: hole in a file that has been closed, which is defect 278 wearing the ledger's
#: own clothes. Deleting the entry on sight would have left the cap at 1 with
#: nothing behind it, and a cap nothing is measured against is a comment with
#: arithmetic in it. So the cap is 0, which is the claim that can still be
#: wrong: the next file to grow a pin in `EXCLUSIONS` without a decision behind
#: it goes red on the cap, and the check's own detail line says what to do.
UNVERIFIED_EXCL_PINS: dict[str, str] = {}
MAX_UNVERIFIED_EXCL_PINS = 0

_double_listed = sorted(set(INVENTORY) & set(EXCLUSIONS))
check(
    not _double_listed,
    "no file is in INVENTORY and EXCLUSIONS at once, so reclassifying cannot "
    "discharge a pin",
    f"INVENTORY has {len(INVENTORY)} name(s), EXCLUSIONS has "
    f"{len(EXCLUSIONS)}, and {_double_listed} appear in both. `role_of` consults "
    f"EXCLUSIONS first, so a name in both is classified `exempt` while the census "
    f"loop still walks it as inventory -- two tables asserting opposite things "
    f"about one file, with nothing here to notice. That matters because the "
    f"check below can be made green by moving a name rather than by fixing a "
    f"file, and a reclassification is not a repair: it changes what this file "
    f"believes about the gate and nothing about the gate. Pick one table"
    if not _double_listed else
    f"{len(_double_listed)} name(s) are in both tables: {_double_listed}. They "
    f"cannot be both a tracked gate and an explained non-gate; the reason in "
    f"EXCLUSIONS wins in `role_of` and the census entry is walked anyway",
)

_excl_pins = sorted(n for n in EXCLUSIONS
                    if (SCRIPTS / n).exists() and _declares_a_pin(SCRIPTS / n))
_pin_verdicts = {n: _pin_is_compared(SCRIPTS / n) for n in _excl_pins}
_now_unverified = sorted(n for n, (compared, _w) in _pin_verdicts.items()
                         if not compared)
check(
    len(_now_unverified) <= MAX_UNVERIFIED_EXCL_PINS,
    "an EXCLUSIONS entry may carry a pin only if something can contradict it",
    "; ".join(f"{n} -> {'compares it' if _pin_verdicts[n][0] else 'RENDERS IT ONLY'}"
              + ("" if _pin_verdicts[n][0] else f" ({_pin_verdicts[n][1]})")
              for n in _excl_pins)
    + f". {len(_excl_pins)} file(s) in EXCLUSIONS declare a pin against a cap of "
      f"{MAX_UNVERIFIED_EXCL_PINS}. INVENTORY does not get this rule by "
      f"generosity: the per-pin loop below asserts it for every gate it walks, and "
      f"it walks nothing in EXCLUSIONS, so a table entry is currently enough to "
      f"place a promise outside the reach of every mechanism this file has. The "
      f"test is a *decision* in the syntax tree and not a print: a run that "
      f"recorded the wrong number of results has to be able to come out different"
    if _excl_pins else
    f"no file in EXCLUSIONS declares an EXPECTED_CHECKS, so this rule is "
    f"vacuously satisfied today; it runs against any file that grows one",
)

check(
    set(UNVERIFIED_EXCL_PINS) == set(_now_unverified),
    "every recorded unverified EXCLUSIONS pin is still a pin that nothing "
    "compares",
    f"{len(UNVERIFIED_EXCL_PINS)} recorded: {sorted(UNVERIFIED_EXCL_PINS)}; "
    f"{len(_now_unverified)} measured: {_now_unverified}. A file that has been "
    f"fixed, renamed or had its pin taken out stops being a real gap, and an "
    f"entry left behind would be a paragraph in this file describing a defect "
    f"that no longer exists -- which is defect 278 wearing the ledger's own "
    f"clothes. Remove the entry when the file is repaired, and lower the cap"
    if set(UNVERIFIED_EXCL_PINS) == set(_now_unverified) else
    f"recorded but not measured: "
    f"{sorted(set(UNVERIFIED_EXCL_PINS) - set(_now_unverified))}; "
    f"measured but not recorded: "
    f"{sorted(set(_now_unverified) - set(UNVERIFIED_EXCL_PINS))}. The first list "
    f"is an entry that has stopped being true; the second is a gap that has "
    f"appeared since the last time anybody looked",
)

# The two halves of the deliverable convention, which is what stops the
# classification above from being a human with a reason attached. A
# measurement script is not a gate and is not asked for a pin; it is asked to
# be *run*, because a measurement nobody runs is a number that rots silently.
deliverables = sorted(p.name for p in SCRIPTS.glob("*.py") if _is_deliverable(p.name))
gate_lookalikes = [
    n for n in deliverables
    if _defines_check_call(SCRIPTS / n) or _declares_a_pin(SCRIPTS / n)
]
check(
    not gate_lookalikes,
    "no file under a deliverable prefix is a gate in disguise",
    f"{len(deliverables)} deliverable(s) {[n for n in deliverables if n not in gate_lookalikes]}; "
    f"{len(gate_lookalikes)} define a check() or carry a pin: {gate_lookalikes}. "
    f"The prefix says 'this script measures a quantity and a gate runs it', so "
    f"one that can fail is a gate and wants a census, an EXPECTED_CHECKS and a "
    f"place in INVENTORY -- rename it or classify it, do not leave the two "
    f"readings in conflict"
    if not gate_lookalikes
    else f"a gate is carrying a deliverable's name: {gate_lookalikes}",
)
unrun = [n for n in deliverables if not _run_by_a_gate(n)]
check(
    not unrun,
    "every measurement deliverable is run by a gate, so its number is read",
    "; ".join(
        f"{n} <- {_run_by_a_gate(n) or 'NO GATE NAMES IT'}" for n in deliverables
    ) + f". A deliverable prints a measured constant and nothing else: there is "
      f"no exit code a reader would look at and no verdict line, so if no gate "
      f"runs it the file is not a diagnostic nobody runs, it is a number nobody "
      f"checks -- and a stale number is indistinguishable from a correct one "
      f"by reading it"
    if deliverables
    else f"no file in scripts/ carries a deliverable prefix "
         f"({', '.join(DELIVERABLE_PREFIXES)}), so this rule is currently "
         f"vacuous; the two halves of it still run against every file that "
         f"grows one",
)

for name, reason in sorted(EXCLUSIONS.items()):
    check(
        (SCRIPTS / name).exists() and len(reason) >= MIN_NO_PIN_REASON,
        f"the exclusion of {name} is real and gives a reason",
        f"{len(reason)} characters"
        + ("" if (SCRIPTS / name).exists() else " -- BUT THE FILE DOES NOT EXIST"),
    )

# ==========================================================================
section("every file in the inventory is still on disk")

missing = [n for n in INVENTORY if not (SCRIPTS / n).exists()]
check(
    not missing,
    "no check script has been deleted from the inventory",
    f"{len(INVENTORY)} listed; missing: {missing}. Deleting a gate is a "
    f"coverage decision and should be a deliberate edit to this list, not a "
    f"silent `rm`",
)

# ==========================================================================
section("every gate declares EXPECTED_CHECKS or says why it cannot")

# `EXPECTED_CHECKS = 153  # measured ...` at module level, with the provenance
# comment belonging to *that line*.
#
# Extracted through the AST rather than a regex over the text, for two reasons
# this file was bitten by. A regex finds the `EXPECTED_CHECKS = 153` inside this
# file's own docstring before the real constant below it. And `\s` matches
# newlines, so `EXPECTED_CHECKS = 233` with no comment swallows the *next* line's
# comment and reports provenance that belongs to a different constant. A module
# scope cannot be inside a string, so asking the parser removes both.
def pin_of(path: Path):
    """(value, provenance) for a module-level EXPECTED_CHECKS, or None.

    Provenance is the trailing comment on the assignment line **plus** the
    contiguous block of comment lines directly above it. Both forms are used in
    this repository and reading only the trailing one produced two false reds:
    `cli_check.py` and `workbench_interaction_check.py` both document the number
    in a `#:` block above the constant, which is this repo's own house style for
    saying something at length, and a rule that cannot see it reports a
    documented number as a transcribed one. The upward walk stops at the first
    line that is not a comment, so it cannot reach past the constant's own
    docstring or borrow a note belonging to something else.
    """
    src = read(path)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return ("SYNTAX ERROR", str(exc))
    lines = src.splitlines()
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        for t in targets:
            if t.id != "EXPECTED_CHECKS":
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                m = re.search(r"#(.*)$", line)
                above: list[str] = []
                for i in range(node.lineno - 2, -1, -1):
                    stripped = lines[i].strip()
                    if stripped.startswith("#"):
                        above.append(stripped.lstrip("#").strip())
                        continue
                    if not stripped:
                        continue
                    break
                parts = [p for p in reversed(above) if p]
                if m:
                    parts.append(m.group(1).strip())
                return (int(value.value), " ".join(parts).strip())
            return ("NON-LITERAL", ast.dump(node.value)[:60])
    return None


NOPIN_RE = re.compile(r'^NO_EXPECTED_CHECKS\s*=\s*"""(.*?)"""', re.S | re.M)

pinned, unpinned, undeclared = [], [], []
for name in INVENTORY:
    path = SCRIPTS / name
    if not path.exists():
        continue
    src = read(path)
    pin = pin_of(path)
    nopin = NOPIN_RE.search(src)
    if pin and pin[0] != "SYNTAX ERROR" and pin[0] != "NON-LITERAL":
        pinned.append((name, pin[0], pin[1]))
    elif nopin:
        unpinned.append((name, nopin.group(1).strip()))
    else:
        undeclared.append(name)

#: **Derived, not typed.** `undeclared` is every gate in `INVENTORY` that has
#: neither an `EXPECTED_CHECKS` nor a `NO_EXPECTED_CHECKS` reason, computed
#: above. There is no ledger for it to be a member of and nothing to keep in
#: step: a gate that gains a pin leaves this list on the next run without anyone
#: editing a table in a different file.
#:
#: What replaced the per-name check that used to sit here is the ratchet below,
#: which is the stronger statement anyway -- "at most N gates are unpinned"
#: covers every one of these names at once and cannot be satisfied by listing
#: the right twelve of them.
check(
    len(undeclared) <= MAX_UNPINNED_SCRIPTS,
    f"no more than {MAX_UNPINNED_SCRIPTS} check scripts are unpinned",
    f"{len(undeclared)} derived from the files themselves: {sorted(undeclared)}. "
    f"Neither an `EXPECTED_CHECKS` nor a `NO_EXPECTED_CHECKS` reason. Adding a "
    f"13th unpinned gate fails here; giving any of these a pin drops the count, "
    f"at which point lower MAX_UNPINNED_SCRIPTS so the slack cannot be spent on "
    f"a new gap. Nothing in this file has to be edited when a script gains a "
    f"pin -- that is the drift this replaced",
)

#: The debt table is printed after the census loop, because the number it shows
#: for each gate is the derived one and the census is what derives it.

# ==========================================================================
section("a declared pin is justified, positive, and actually compared")

#: Gates whose pin-provenance check is deferred because the file was written
#: during this run. Named here rather than in a hand-typed table for the same
#: reason the census deferral is: an exemption that cannot expire is a parking
#: space.
_deferred_pins: list[str] = []

for name, value, comment in sorted(pinned):
    check(
        value > 0,
        f"{name}'s pin is a positive count",
        f"EXPECTED_CHECKS = {value}",
    )
    # A pin with no provenance is a transcribed number, which is the failure this
    # repository keeps meeting elsewhere.
    # `PROVENANCE_RE` is a **proxy**, not the requirement. The requirement is
    # "there is a note saying where this number came from"; the regex only
    # recognises some ways of writing one. It reads only its own set of words, and
    # that set was deliberately left alone during this round even though it looked
    # too narrow: two pins here are documented in a `#:` block above the constant,
    # which `pin_of` used to be blind to, and the obvious response was to widen
    # the keywords as well. A mutation narrowing it back proved that unnecessary --
    # both notes now say "measured" -- so the real defect was in `pin_of`, and
    # widening the vocabulary to paper over a reader bug would have loosened a
    # rule for a reason that turned out to be false. Named in one place because it
    # was previously spelled three slightly different ways at three call sites,
    # which is how a rule quietly stops being one rule.
    justified = bool(PROVENANCE_RE.search(comment))
    _flying_pin = _is_in_flight(SCRIPTS / name)
    if _flying_pin:
        # A pin that was written seconds ago will often not have its note yet.
        # Holding a half-finished edit to a documentation standard reports "you
        # are mid-save" as "this is wrong", and the fix is always to stop
        # running the gate. Deferred, recorded, and it expires by itself.
        _deferred_pins.append(f"{name} ({_age_minutes(SCRIPTS / name):.0f} min)")
    check(
        justified or name in UNJUSTIFIED_PINS or _flying_pin,
        f"{name}'s pin says where the number came from",
        (f"comment: {comment.strip()!r}"
         if justified else
         f"comment is empty, and this file is listed in UNJUSTIFIED_PINS as a "
         f"known gap: {UNJUSTIFIED_PINS.get(name, 'NOT LISTED')!r}"),
    )
    # A pin nobody compares is decoration: the number is written down and
    # nothing reads it, so the script can lose half its checks and stay green.
    #
    # Read by `_pin_is_compared`, which walks the tree, and no longer by
    # matching lines. **This loop was the reason the two disagreed.** It asked
    # whether a line mentioned the name and carried a comparison operator, so a
    # `#:` comment saying `len(RESULTS) != EXPECTED_CHECKS` was credited as a
    # comparison, and so was a sentence inside an f-string. Measured on this
    # file before the change: 14 credited sites, of which **2 were real**. Six
    # were comment lines, five were string literals, three were the regex and
    # the `any(t.id == "EXPECTED_CHECKS" ...)` reader comparing the *string* to
    # the name rather than the pin to anything, and the rest were prose about
    # other gates' pins. A sibling agent's own `#:` comment tripped it in
    # practice and was reworded to avoid the reader, which treats a symptom: the
    # reader was the thing that was wrong, and a check whose subject is "is this
    # promise kept" cannot credit a sentence *about* it.
    #
    # No gate's verdict changes -- all 31 pinned gates carry at least one real
    # decision -- and the message now names a line in code rather than a line of
    # text, so a reader can go and look at the decision instead of the prose.
    # The separating mutation is M2b in the matrix above: it is the one the text
    # version passes and this one refuses.
    compared, where = _pin_is_compared(SCRIPTS / name)
    check(
        compared,
        f"{name}'s pin is actually compared, not just declared",
        f"read by a decision, not by a render: {where}",
    )

check(
    len(UNJUSTIFIED_PINS) <= MAX_UNJUSTIFIED_PINS,
    f"no more than {MAX_UNJUSTIFIED_PINS} pins are unjustified",
    f"{len(UNJUSTIFIED_PINS)} in UNJUSTIFIED_PINS against a cap of "
    f"{MAX_UNJUSTIFIED_PINS}. Same ratchet as the unpinned count: adding a "
    f"provenance comment removes an entry and lowers the cap, so the slack "
    f"cannot be spent on a new bare pin",
)

_stale_pins = [
    n for n in UNJUSTIFIED_PINS
    if not (SCRIPTS / n).exists()
    or pin_of(SCRIPTS / n) is None
    or PROVENANCE_RE.search((pin_of(SCRIPTS / n) or (0, ""))[1])
]
check(
    not _stale_pins,
    "every UNJUSTIFIED_PINS entry is still an unjustified pin that exists",
    f"{len(UNJUSTIFIED_PINS)} entries; stale: {_stale_pins}. Remove the entry "
    f"once the comment is added",
)

# ==========================================================================
section("an unpinned gate gives a reason worth reading")

for name, reason in sorted(unpinned):
    check(
        len(reason) >= MIN_NO_PIN_REASON,
        f"{name}'s no-pin reason is worth reading",
        f"{len(reason)} characters: {reason[:90]!r}",
    )

if not unpinned:
    check(True, "every unpinned gate gives a reason worth reading",
          "no gate currently declares NO_EXPECTED_CHECKS")

# ==========================================================================
section('every gate declares the call sites it actually has')

#: What each gate's call-site census is *supposed* to be, for the gates whose
#: file does not carry its own `GATE-DECLARE` block yet.
#:
#: **This table is generated, not written.** `python scripts/check_scripts_declare.py
#: --pin` rewrites the region between the two markers below, straight from the
#: `_sites_of` walk that the comparison then runs. That is the whole point of
#: keeping it here instead of in each file during the transition: it is
#: impossible to type a number into it that the auditor will believe, because
#: the auditor computes the number and compares. Hand-editing it to "fix" a red
#: does not work -- it produces a different red.
#:
#: It exists at all only because this file does not own every gate. A
#: `GATE-DECLARE` block is an edit to somebody else's file, and eighteen gates
#: belong to other owners. `MAX_SNAPSHOT_ONLY` counts them and only falls.
#:
#: A gate appears here **only** if it has no block of its own; `_declared_for`
#: prefers the block, and a gate in both places is a failure rather than a
#: silent precedence rule.
#: --- BEGIN GENERATED BLIND SPOTS (python scripts/check_scripts_declare.py --pin) ---
#: Every check-like call in the inventory that the census above cannot
#: count, because the call is not the whole statement. A gate appearing
#: here for the first time is the red this table exists to produce.
#: --- a gate name is followed by (count, why it is there) ---
BLIND_SPOTS: dict[str, tuple[int, str]] = {
    "cli_check.py": (3,
        "two results whose value the caller consumes (`spoke and quiet` in fails_cleanly) plus one self-test that wants the boolean back. The census sees 120 of this file's 123 check-like calls"),
    "cli_prep_check.py": (1,
        "a self-test: `returned = check(...)` is how this file proves its own check() can record a failure, and the assignment is the point"),
    "core_check.py": (4,
        "four `return check(...)` in one helper, so an exception turns into a recorded result rather than a traceback. The census sees 357 of 361"),
    "ligand_check.py": (1,
        "a self-test, same shape as cli_prep_check.py's"),
    "prep_check.py": (1,
        "a self-test, same shape as cli_prep_check.py's"),
    "workbench_interaction_check.py": (1,
        "one `return check(...)`: the result of this call is the caller's verdict, so it cannot become a bare statement without changing what the gate decides"),
}
#: --- END GENERATED BLIND SPOTS ---
#: --- BEGIN GENERATED SNAPSHOT (python scripts/check_scripts_declare.py --pin) ---
DECLARED_SNAPSHOT = {
    "benchmark_check.py": (34, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "cli_check.py": (82, 38,
        "e2ef8b21905a914e21c598911b87d272678ccf203538382be9027750415dcc5d"),
    "cli_prep_check.py": (32, 2,
        "2013eb0ff8a729058c868eee6568b0898977584da1f8a4c8c79d917e1bc3b184"),
    "contacts_attribution_check.py": (15, 2,
        "14b3f81f282ef9120b0c2c9405f6060bf764650a4cac8c35e4aa66274992a5ee"),
    "contacts_criteria_check.py": (19, 22,
        "a548473d3772baf82275d8110f6df5dca83fd4251e3f421d66fa73ed788b3735"),
    "core_check.py": (350, 8,
        "ccd8b848d563a61b105a61011fdf90384793979b24908f4b8f4be788c5d678a6"),
    "ligand_check.py": (38, 20,
        "c67f638574f3fabdc9e55889ba12605598183454ab466f785b4e7a6752a334cd"),
    "pdbqt_check.py": (49, 4,
        "b27c4f071eca1a4757f0ae454557123488ea6da0467a8bc10a5ed69076382c87"),
    "pockets_check.py": (65, 84,
        "4bf203db36996b5950ef84c3e42982aa5379a9351a16fab1a2245fc9ec5f3ab8"),
    "representation_geometry_check.py": (46, 5,
        "1138babf87c705ad487e8c9d894dbf2ddaea822a32a5361b6c6c92a9d230aa52"),
    "scoring_cross_check.py": (27, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "torsion_bond_record_check.py": (12, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "workbench_interaction_check.py": (115, 195,
        "6cf77000b4becde05d78a255ea34542672c3e507763912114aa12b3ba804cf2a"),
}
#: --- END GENERATED SNAPSHOT ---

#: How many gates are still relying on the generated table instead of declaring
#: their own census.
#:
#: This is **not** an exemption. Every one of these gates is measured, compared
#: against its declared census, and goes red on drift exactly like the ones that
#: carry a block -- the only difference is which file the expected value is
#: written in. The old `IN_FLIGHT` list was a real exemption: those two files
#: were held out of the comparison altogether, so nothing about them could ever
#: be reported. This is the opposite of that, and the difference is the reason
#: the mechanism can shrink instead of parking.
#:
#: **36 gates are measured; 23 of them declare their own block, so 13 are on the
#: snapshot and the cap is 13.** Measured by a run on 2026-10-02, not counted off
#: a diff: the check below prints both numbers every time it fires, so a reader
#: who thinks this paragraph is wrong can run the file rather than believe it.
#:
#: The count is derived, not asserted, and this paragraph is the thing that has
#: now been wrong twice. It said "23 gates, four declare a block, so 19", which
#: described the tree at the moment it was written and nothing since; then it
#: said "33 gates, 17 declare a block, so 14", which was wrong in all three
#: numbers at once -- it overstated the measured total by three, understated the
#: block-declaring gates by six, and so left the cap a point above the set it
#: claims to bound. **The constant was 14 while the set was 13**, and because
#: the ratchet is `<=` that slack is invisible: it reads as a healthy bound and
#: is really one unit of unearned room. A cap whose own arithmetic is wrong is
#: worse than no cap, because it is read as a current measurement.
#:
#: **What the cap is for.** It is a *ratchet*, not an exemption and not a
#: record: it bounds how much of the tree can still be relying on a number
#: written in this file rather than in the gate that owns it, so the migration
#: to per-gate `GATE-DECLARE` blocks can only move one way. It is deliberately
#: a `<=` over a **derived** set, which is what makes it a ratchet at all --
#: every one of these 13 gates is measured against its snapshot row and goes red
#: on drift exactly like a block-declaring gate, so nothing is being held out of
#: the census here. The cap counts only *which file the expected number lives
#: in*, and its target is 0.
#:
#: **What happens when it is hit.** The check
#: `"no more than 13 gates rely on the generated snapshot"` goes red and prints
#: the derived set by name next to the cap, so the red is a finding about a
#: specific gate rather than a number somebody has to go and count. It goes red
#: in exactly one direction: a gate that gains a block lowers the set and stays
#: green, and a gate that *appears* without one raises it. That asymmetry is
#: deliberate -- the failure this guards against is new debt, not old debt being
#: paid down -- and it means a red here is always about something that was just
#: added, never about something that has been true for months. The clearing
#: paths are the two the message names, in this order of preference:
#:
#:   1. run `python scripts/check_scripts_declare.py --census` and paste the
#:      three-line `GATE-DECLARE` block it prints into the gate's own source.
#:      That removes the gate from this set for good and the cap should be
#:      lowered to the new measured size in the same commit;
#:   2. if the gate genuinely cannot carry a block, say so in the census and
#:      leave it -- but then the cap is the wrong instrument and the prose above
#:      it is what has to change, not the number.
#:
#: Raising the cap is also a clearing path, and it is the one with no teeth:
#: nothing in this file records what the cap was, so a raised cap is
#: indistinguishable from a migration and stays green forever. The cap is a
#: bound somebody typed, and the only thing holding it to the set it bounds is
#: the discipline of lowering it -- which is why the arithmetic is written out
#: here in full rather than left to the reader.
#:
#: **What moved, and what did not.** Four gates were migrated onto blocks of
#: their own this round: `examples_check.py` and
#: `release_tree_parity_check.py`, which already had a block and whose declared
#: columns had drifted from the tree (46+0 against 41+0, 0+71 against 0+58), and
#: `prep_check.py` and `x11_window_parse_check.py`, which had neither a block nor
#: -- for `x11` -- a pin, and were living entirely on a generated row. Every
#: number in those four blocks was measured by this file's own `_sites_of` and
#: `_guards_digest` against the file on disk; none was typed, and each is
#: asserted by the census check on the next run.
#:
#: **The cap is 13 and the target is 0, and the difference is not slack.** Every
#: one of the 13 belongs to another owner, so driving it to zero is a sequence
#: of edits to their files, not a decision available here. A gate that grows a
#: block leaves the snapshot the moment `--pin` runs and not before, so between
#: the migration and the regeneration it is counted in *neither* the cap's favour
#: nor its disfavour -- the set is read from the file on disk either way -- and
#: the separate "no gate is left in the snapshot after it grew a block" check is
#: what holds it, red, naming both, until the region is regenerated. That red is
#: the migration being visible rather than the migration being wrong.
#:
#: `examples_check.py` moved off this snapshot to carry a block of its own. The
#: prose above called a gate appearing in both places a failure, and then did
#: not check for it: `_declared_for` prefers the block, so a stale snapshot entry
#: was invisible to every count here and `--pin` would have removed it silently
#: the next time anyone ran it. It is checked now, which is the same lesson as
#: the `SITE_INVENTORY` hole recorded below: a rule stated in a comment and
#: absent from the code is a comment.
MAX_SNAPSHOT_ONLY = 13

#: The hole this table used to have, recorded because the shape of it is the
#: lesson. `SITE_INVENTORY` held 17 entries against an `INVENTORY` of 21, and
#: **no check compared the two sets**, so the four files with no entry were
#: audited by nothing at all: `cli_check.py` and `workbench_interaction_check.py`
#: (exempted by name), and -- silently, with no name anywhere --
#: `cli_prep_check.py` and this file. That is how `cli_prep_check.py` came to
#: carry a recorded "37" in a ledger while no code in the repository was capable
#: of asking what it ran.
#:
#: The census loop below now runs over **every** name in `INVENTORY`, so there
#: is no set of names that can be left out of it.


def _short(node) -> str:
    try:
        return ast.unparse(node)[:56]
    except Exception:
        return "?"


def _sites_of(src: str):
    """(unconditional count, sorted guard strings) for check-like call sites.

    Deliberately counts the *call sites*, not the invocations: a site inside a
    `for` still counts once here even though it may run many times. That is the
    right granularity for this question, which is "could this site stop
    running", not "how many checks will the run report".
    """
    tree = ast.parse(src)
    parent_of = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent_of[c] = n

    def branch(node, t):
        def has(body):
            return any(node is d for st in body for d in ast.walk(st))
        for h in t.handlers:
            if any(node is d for d in ast.walk(h)):
                return "except"
        if t.orelse and has(t.orelse):
            return "else"
        if t.finalbody and has(t.finalbody):
            return "finally"
        return "body"

    uncond, cond = 0, []
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.Expr) or not isinstance(child.value, ast.Call):
                continue
            f = child.value.func
            if getattr(f, "id", None) not in ("check", "ok", "bad", "expect"):
                continue
            # `cur` starts at the *call*, not at its parent. Starting at the
            # parent skips the statement's own enclosing block, which is
            # exactly the `if` this check exists to notice -- that bug made the
            # guarded column undercount by one nesting level, and it is what
            # let a real mutation through.
            #
            # Only *conditional* constructs count as guards. Function frames
            # are ignored on purpose: a check inside a helper that `main()`
            # always calls does run every time, and counting `def sec_writers`
            # as a guard would label most of `core_check.py` conditional and
            # make the column useless. The cost is a known limitation -- a
            # check inside a helper that is never called is NOT detected, and
            # catching that needs a call graph, not a walk.
            guards, cur = [], child
            while cur in parent_of:
                p = parent_of[cur]
                if isinstance(p, ast.If):
                    guards.append("if " + _short(p.test))
                elif isinstance(p, (ast.For, ast.AsyncFor)):
                    guards.append("for " + _short(p.target))
                elif isinstance(p, ast.While):
                    guards.append("while " + _short(p.test))
                elif isinstance(p, ast.Try):
                    guards.append("try:" + branch(cur, p))
                if isinstance(p, ast.Module):
                    break
                cur = p
            # `if __name__ == "__main__"` is the normal way a script is run,
            # not a condition on whether a check happens.
            guards = [g for g in guards
                      if "__name__" not in g and g != "if True"]
            if guards:
                cond.append(" | ".join(guards))
            else:
                uncond += 1
    return uncond, sorted(cond)


#: The census's blind spot, declared rather than left implicit.
#:
#: `_sites_of` counts a call only when the call *is* the whole statement: it
#: matches `ast.Expr` whose value is a `Call`. A call in any other position --
#: `r = check(...)`, `return check(...)` -- is a real result site that this
#: walk never sees. **That blindness is on both sides of the comparison**, which
#: is what makes it worth closing rather than noting: the derived column and
#: the declared column are produced by the same walk, so a gate that grew a
#: `r = check(...)` moves neither number and the census check below stays
#: green. The comparison is still load-bearing -- it catches every `Expr`
#: site, which is 1932 of the 1943 here -- but "the census is complete" was
#: never a claim it could support, and this table is what makes the remainder
#: a number that moves instead of a gap nobody is measuring.
#:
#: **The failure direction this buys is a red on a gate that adds a blind
#: site.** Before this table, adding `r = check(...)` to any gate was free.
#: After it, the derived blind count moves against a declared count and the
#: check goes red. That is the whole point: a blind site is not a wrong
#: number, it is an *invisible* one, and the only way to notice an invisible
#: thing is to count it on purpose.
#:
#: **Derived by `_blind_sites_of`, never typed.** `--pin` writes this region
#: from the same walk the check below compares against, exactly as it does
#: for `DECLARED_SNAPSHOT`, so a declaration here cannot record anything the
#: next run will not re-derive.
#:
#: The values are `(count, "why this is here")`. The reason is load-bearing
#: rather than decorative: every entry is a place where the census is blind,
#: and a reader deciding whether to close one needs to know whether the
#: result is consumed (`spoke = check(...)` feeds a caller) or discarded
#: (`returned = check(...)` is a self-test that wants the boolean back).
BLIND_BEGIN = "#: --- BEGIN GENERATED BLIND SPOTS"
BLIND_END = "#: --- END GENERATED BLIND SPOTS ---"


def _blind_sites_of(src: str) -> list[tuple[int, str]]:
    """check-like calls `_sites_of` cannot count, as `(lineno, form)`.

    Deliberately a *separate* walk rather than a flag on `_sites_of`. The two
    questions are different -- "could this site stop running" and "is there a
    result site here that the census cannot see" -- and folding them together
    would change the number 33 gates have already declared, which is a change
    this file does not get to make on their behalf.

    The `form` is the statement kind, so the report says *how* the call is
    positioned rather than only how many there are. `Return` and `Assign` are
    the two that matter in practice: a returned `check(...)` is a result whose
    verdict is the caller's, and an assigned one is a result captured for a
    later `and`/`or`.
    """
    tree = ast.parse(src)
    parent_of = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent_of[c] = n
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) not in ("check", "ok", "bad", "expect"):
            continue
        stmt = node
        while stmt in parent_of and not isinstance(stmt, ast.stmt):
            stmt = parent_of[stmt]
        if isinstance(stmt, ast.Expr):
            continue  # this one the census counts
        if isinstance(stmt, ast.Assign):
            form = "assigned to " + ast.unparse(stmt.targets[0])[:40]
        elif isinstance(stmt, ast.AnnAssign):
            form = "annotated assignment to " + ast.unparse(stmt.target)[:40]
        elif isinstance(stmt, ast.AugAssign):
            form = "augmented assignment to " + ast.unparse(stmt.target)[:40]
        elif isinstance(stmt, ast.Return):
            form = "returned"
        elif isinstance(stmt, ast.For):
            form = "the iterable of a for loop"
        else:
            form = "inside a " + type(stmt).__name__
        out.append((node.lineno, form))
    return sorted(out)


# --------------------------------------------------------------------------
# The `GATE-DECLARE` block: the format, and the reader for it
# --------------------------------------------------------------------------
#
# A gate declares what it is supposed to contain like this, in comments:
#
#     #: GATE-DECLARE 1
#     #: sites: 79 unconditional + 6 guarded
#     #: guards: sha256:<64 hex>
#
# Three properties, each of them load-bearing:
#
# * **It is read from comment tokens, never from the file text.** `pin_of` above
#   is extracted through the AST for the same reason and carries the scar
#   tissue: a regex over source text finds `EXPECTED_CHECKS = 153` inside this
#   file's own docstring before the real constant. `tokenize` cannot be fooled
#   that way, because a docstring line is a STRING token and never a COMMENT
#   one -- so this file may *describe* the format in its docstring, and several
#   files already do, without any of them declaring anything.
# * **The version is in the header** so that changing the format later is a
#   detectable edit rather than a silent reinterpretation of an old block.
# * **Both numbers are compared, and neither is believed.** `sites` is compared
#   to `_sites_of`; `guards` is compared to a digest of the same walk's guard
#   strings. There is no path by which the declaration can satisfy the auditor
#   on its own, which is the entire property being bought.
DECL_VERSION = "1"

_DECL_SITES_RE = re.compile(
    r"^#:\s*sites:\s*(\d+)\s+unconditional\s*\+\s*(\d+)\s+guarded\s*$"
)
_DECL_GUARDS_RE = re.compile(r"^#:\s*guards:\s*sha256:([0-9a-f]{64})\s*$")
_DECL_HEAD_RE = re.compile(r"^#:\s*GATE-DECLARE\s+(\d+)\s*$")


def _comment_lines(path: Path) -> list[str]:
    """Every real comment in `path`, in order, `#` included.

    `tokenize` is used rather than a regex over the text so that a line like
    `#: sites: 4 unconditional + 0 guarded` written *inside a docstring* is a
    string and not a comment. That is not hypothetical: this file's docstring and
    the reader below both document the format with exactly those words.

    The `#` is deliberately **kept**, so the patterns below can anchor on `^#:`.
    Stripping it first was tried and is worse in a way worth recording: this
    file documents the format in an indented example (`#     #: GATE-DECLARE 1`),
    which becomes `#: GATE-DECLARE 1` once the hash is stripped -- indistinguishable
    from a real declaration. Keeping the hash means an indented example cannot
    match, and the format the patterns accept is literally the format written in
    the file's own comments.
    """
    out = []
    with io.open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT:
                out.append(tok.string.strip())
    return out


def _declaration_of(path: Path):
    """(unconditional, guarded, guards-digest) from a `GATE-DECLARE` block.

    Returns None when the file declares no block at all, and the string
    ``"MALFORMED <why>"`` when it starts one it cannot finish -- a half-written
    block is a mistake in the file, not an absence, and it must not read as
    "no declaration" because that would silently fall back to the snapshot.
    """
    lines = _comment_lines(path)
    i = 0
    while i < len(lines):
        m = _DECL_HEAD_RE.match(lines[i])
        if not m:
            i += 1
            continue
        if m.group(1) != DECL_VERSION:
            return "MALFORMED GATE-DECLARE version %s, this file reads %s" % (
                m.group(1), DECL_VERSION)
        sites = guards = None
        for line in lines[i + 1:]:
            ms = _DECL_SITES_RE.match(line)
            mg = _DECL_GUARDS_RE.match(line)
            if ms:
                sites = (int(ms.group(1)), int(ms.group(2)))
            elif mg:
                guards = mg.group(1)
            elif _DECL_HEAD_RE.match(line):
                break
        if sites is None:
            return "MALFORMED no `sites:` line"
        if guards is None:
            return "MALFORMED no `guards: sha256:` line"
        return (sites[0], sites[1], guards)
    return _near_miss_declaration(path, lines)


#: A declaration that is *almost* one. Measured, not hypothetical: two gates in
#: this tree wrote their whole block as `# GATE-DECLARE 1` / `# sites: ...` /
#: `# guards: ...` -- the bare hash, no colon -- and each was reported as
#: `NOT DECLARED ANYWHERE`, which is true and useless. The reader cannot see a
#: block that is three lines away from valid; its author can, and the difference
#: between "you did not declare" and "you declared it with the wrong prefix" is
#: the difference between a mystery and an edit. The numbers in both blocks were
#: right, which is the worst part: the file looked finished.
_NEAR_MISS_RE = re.compile(r"^#\s*GATE-DECLARE\s+\d+\s*$")


def _near_miss_declaration(path: Path, lines: list[str]):
    """A named answer when the block is present but its prefix is wrong.

    Returns ``None`` -- genuinely no declaration -- when there is no near miss,
    so this cannot turn a clean file red. It only ever *replaces* the less
    specific "no declaration" answer with the specific one, which is the whole
    point: the two are different sentences about the author's file.
    """
    for n, line in enumerate(lines, 1):
        if _NEAR_MISS_RE.match(line):
            return (
                "MALFORMED GATE-DECLARE written with a bare `#` at %s:%d -- the "
                "three lines must read `#: GATE-DECLARE 1`, `#: sites: N "
                "unconditional + M guarded`, `#: guards: sha256:...`; the `#:` "
                "prefix is what the reader matches on, and a block without it is "
                "indistinguishable from no block at all" % (path.name, n)
            )
    return None


def _guards_digest(guards: list[str]) -> str:
    """sha256 over the newline-joined sorted guard strings.

    A digest rather than the list, for one reason: `SITE_INVENTORY` used to hold
    the guard strings verbatim, about two hundred of them, and every one was a
    chance to transcribe a `CRAMBIN.is_file() and CRAMBIN_POSE.is_file()` with
    a bracket in the wrong place. The list is still what is compared -- this
    hashes it -- so the sensitivity is unchanged and the transcription surface
    is one line.
    """
    return hashlib.sha256("\n".join(guards).encode("utf-8")).hexdigest()


def _declared_for(name: str, path: Path):
    """(uncond, guarded, digest, where_from) for a gate, block preferred."""
    block = _declaration_of(path) if path.exists() else None
    if isinstance(block, str):
        return (None, None, None, block)
    if block is not None:
        return (block[0], block[1], block[2], "its own GATE-DECLARE block")
    snap = DECLARED_SNAPSHOT.get(name)
    if snap is None:
        return (None, None, None, "NOT DECLARED ANYWHERE")
    return (snap[0], snap[1], snap[2], "the generated snapshot")


#: The header and footer `--pin` writes around each generated region. They are
#: text, not anchors: the anchors below are what the rewrite splices on, and
#: keeping the two separate is the whole fix.
PIN_BEGIN = "#: --- BEGIN GENERATED SNAPSHOT"
PIN_END = "#: --- END GENERATED SNAPSHOT ---"

#: The lines the rewrite actually splices on. Each is tried in order and
#: **exactly one of them must match, once**, or the rewrite refuses.
#:
#: **Why not `PIN_BEGIN`.** `--pin` used to search for the text of `PIN_BEGIN`
#: and splice there, and the text of `PIN_BEGIN` is inside this file's own
#: string literal on the line above. `src.find(PIN_BEGIN)` therefore matched the
#: literal rather than the region whenever the region's own opening comment was
#: missing, `end` resolved *earlier* than `start`, and the rewrite pasted a
#: region plus a duplicated tail into the middle of that literal. The result was
#: a 226,082-byte file that became 244,043 bytes and no longer parsed
#: (`SyntaxError: unterminated string literal`), on the one run whose whole
#: purpose was to leave a valid file behind. Measured on a copy under
#: `target/iso_census6/`; the live tree was never run through it.
#:
#: The fix is not a better search string, it is a different *kind* of anchor.
#: Every candidate below is a complete line that cannot be produced by the
#: constants that spell the region's name -- `PIN_BEGIN = "..."` is a line
#: beginning with that identifier, and the emitted header carries a
#: `(python ... --pin) ---` suffix the assignment cannot have. The header is
#: listed first and the bare assignment second, so a region that is missing its
#: header -- which is the state this file was actually in -- is repaired rather
#: than refused, and a region that has one is not given a second.
_PIN_SUFFIX = " (python scripts/check_scripts_declare.py --pin) ---"
SNAP_HEADS = ["\n" + PIN_BEGIN + _PIN_SUFFIX, "\nDECLARED_SNAPSHOT = {"]
BLIND_HEADS = ["\n" + BLIND_BEGIN + _PIN_SUFFIX,
               "\nBLIND_SPOTS: dict[str, tuple[int, str]] = {"]
SNAP_FOOT = "\n" + PIN_END
BLIND_FOOT = "\n" + BLIND_END


#: How far apart two opening anchors may be and still be the same region. A
#: region is a header comment, at most four lines of `#:` prose, and the
#: assignment; four kilobytes is generous for that and far too little to be two
#: unrelated regions.
_REGION_HEAD_WINDOW = 4096


def _splice_region(src: str, heads, foot: str, region: str, what: str):
    """Replace the region between two line anchors with `region`, or return None.

    Three properties, each of them a failure this file has already had:

    * **The anchors are whole lines, not substrings of the markers.** A rewrite
      can no longer find its own marker's string literal.
    * **One region, not one anchor.** A region carries *both* accepted opening
      anchors at once -- the emitted header and the assignment under it -- so
      requiring one match would refuse every healthy file. The head is the
      earliest match and the rest have to sit within `_REGION_HEAD_WINDOW` of
      it; two matches far apart mean two regions, and rewriting one would leave
      the tree describing itself twice.
    * **Ordering is asserted, not assumed.** A foot at or before the head is
      refused rather than spliced, which is the shape that turned into 17,961
      bytes of duplicated source. It returns None, the caller reports, and the
      file is left exactly as it was.
    """
    found = sorted(i for h in heads for i in _all_occurrences(src, h))
    foots = _all_occurrences(src, foot)
    if not found or len(foots) != 1:
        print(f"refusing to rewrite the {what} region: expected an opening "
              f"anchor and exactly one closing anchor, found {len(found)} and "
              f"{len(foots)} in {TARGET.name}. The file has not been modified",
              file=sys.stderr)
        return None
    start = found[0]
    if any(i - start > _REGION_HEAD_WINDOW for i in found[1:]):
        print(f"refusing to rewrite the {what} region: found {len(found)} "
              f"opening anchors more than {_REGION_HEAD_WINDOW} characters "
              f"apart, so this file appears to contain more than one "
              f"{what} region. The file has not been modified", file=sys.stderr)
        return None
    end = foots[0] + len(foot)
    if end <= start:
        print(f"refusing to rewrite the {what} region: its closing anchor is at "
              f"or before its opening one, which is the shape that used to "
              f"duplicate the tail of this file. The file has not been modified",
              file=sys.stderr)
        return None
    return src[:start] + "\n" + region + src[end:]


def _all_occurrences(src: str, needle: str) -> list[int]:
    out, i = [], src.find(needle)
    while i != -1:
        out.append(i)
        i = src.find(needle, i + 1)
    return out


def _ascii_safe(s: str) -> str:
    """`s` unchanged if a console can be relied on to print it, escaped if not.

    A guard string is `ast.unparse` output lifted straight out of another
    file's source, so a condition holding a non-ASCII literal puts a non-ASCII
    character in one. This refusal goes to `sys.stderr`, and the reconfigure at
    the top of this file deliberately does **not** touch stderr -- so on the
    machine this was written on, where a bare `sys.stdout.encoding` is `gbk`,
    printing one raw raises `UnicodeEncodeError` and turns the loud refusal into
    the traceback it exists to avoid. That is the worst of the three outcomes
    and it is one escape call away from not happening.

    Escaping happens only in the case that would otherwise crash, so on an
    all-ASCII guard this is a no-op and prints nothing different.
    """
    if s.isascii():
        return s
    return s.encode("ascii", "backslashreplace").decode("ascii")


def _pin() -> int:
    """Rewrite the generated snapshot from what the files actually contain.

    Run:  python scripts/check_scripts_declare.py --pin

    This is the only supported way to change a declared census for a gate that
    has no block of its own, and it is deliberately not a normal run: it edits
    the auditor rather than reporting on it, so it cannot be reached by accident
    and the edit lands in review as an explicit act rather than as a number that
    changed because someone ran something.

    Two rules keep it honest:

    * **A gate with its own block is skipped, not overwritten.** The block is
      the declaration; regenerating the snapshot for it would create a second
      number to disagree with the first.
    * **The values come from `_sites_of`, the same function the comparison
      uses.** So the snapshot cannot record anything other than what the next
      run will derive. If a file is half-written, `--pin` refuses rather than
      freezing a census of a state that never existed.

    Returns the process exit code.
    """
    _restore_stdout()
    rows = {}
    guard_lists: dict[str, list[str]] = {}
    refused = []
    for name in INVENTORY:
        path = SCRIPTS / name
        if not path.exists():
            refused.append(f"{name}: not on disk")
            continue
        # The declaration is probed *before* the file is parsed, because a gate
        # that declares itself has to be skipped before its numbers are
        # measured. That order is why the probe needs its own guard: reading
        # the declaration runs `tokenize` over the file, and a file being saved
        # right now can raise `TokenError` rather than `SyntaxError`. The
        # docstring below promises `--pin` "refuses rather than freezing a
        # census of a state that never existed", and until this guard it did
        # the opposite -- it died with a traceback, which freezes nothing and
        # tells the reader nothing about which gate was unreadable. Measured on
        # a copy whose `cli_prep_check.py` had one line removed from the middle
        # of a call.
        try:
            declares = _declaration_of(path)
        except (SyntaxError, tokenize.TokenError) as exc:
            refused.append(f"{name}: could not be read ({type(exc).__name__}: "
                           f"{exc})")
            continue
        if declares is not None:
            continue  # declares itself; the snapshot has no business here
        try:
            uncond, guards = _sites_of(read(path))
        except (SyntaxError, tokenize.TokenError) as exc:
            refused.append(f"{name}: does not parse ({exc})")
            continue
        rows[name] = (uncond, len(guards), _guards_digest(guards))
        # The guard strings themselves are kept for the length of this call and
        # are never written anywhere. The snapshot stores the digest precisely
        # so that a hundred and fifty transcribable strings do not have to be
        # maintained by hand; that is the right trade for a *declaration*, and
        # the wrong one for a *refusal*, because a refusal whose whole job is
        # "look at this before you accept it" cannot then print a hash. These
        # are derived from the source on this run and die with the process.
        guard_lists[name] = guards
    if refused:
        print("refusing to pin; these gates could not be measured:", file=sys.stderr)
        for r in refused:
            print(f"  {r}", file=sys.stderr)
        return 2
    # **A shrinkage is not a regeneration, it is a decision.** Growth is what
    # this tree does when it works: four of the four reds in this session were
    # a gate gaining checks, and re-baselining one is bookkeeping. A gate that
    # has *lost* call sites is the failure the whole census exists to catch --
    # it tests less than it did, and it will do so quietly forever -- so it is
    # not something a command may absorb on the way past. It takes
    # `--accept-shrink`, which is a word somebody has to type.
    #
    # This is the answer to "can the declaration be made incapable of going
    # stale". It cannot: it is a number in a comment and every number goes
    # stale. What can be made incapable of going stale *by accident* is the
    # acceptance, and only in the direction where being wrong is expensive.
    shrunk = []
    for name, (u, g, _d) in sorted(rows.items()):
        was = DECLARED_SNAPSHOT.get(name)
        if was is not None and (u + g) < (was[0] + was[1]):
            shrunk.append(f"{name}: {was[0]}+{was[1]} -> {u}+{g} "
                         f"({u + g - was[0] - was[1]:+d} call sites)")
    # **A move is the shrinkage the two counts cannot see.** The loop above
    # catches a gate that has *lost* call sites, which is the direction where
    # being wrong is loud: a number falls off the end. A MOVED gate has the
    # same total and a different digest -- the same `check(...)` under a
    # different condition -- so both columns sit exactly where the declaration
    # says they should be and nothing anywhere goes red. Turning
    # `if kinds.get('hbond'):` into `if not kinds.get('hbond'):` is the same
    # call sites before and after, and a gate that now runs its contact check
    # under the opposite condition and will report on it forever.
    #
    # `_DRIFT_KINDS` calls this "the digest's job", and `--pin` was absorbing it
    # on the way past, which delegated that job to nobody: the digest is the
    # only thing that makes the change visible, and the one command in this tree
    # that would have hidden it did. Measured on a sandbox copy of
    # `contacts_criteria_check.py` with that one condition negated: `--pin`
    # exited 0, printed no warning, and rewrote the snapshot from
    # `a548473d3772` to `611837e921cf`; the census then read the tree as
    # `0 moved, SHRANK: none, UNDECLARED: none` and the gate printed a clean
    # `18 unconditional 22 guarded`. It takes `--accept-moved`, the same shape
    # as `--accept-shrink` and for the same reason: a word somebody has to type.
    #
    # **Classified by `_drift_class`, not by a second subtraction here.** The
    # census files this same event under this same name, and two
    # implementations of one four-way decision is one more thing that can
    # drift. `_drift_class` is defined below this function; it is called at
    # dispatch time, which is the ordering `_census_report` already relies on.
    moved = []
    for name, (u, g, d) in sorted(rows.items()):
        was = DECLARED_SNAPSHOT.get(name)
        if was is None or (u, g, d) == (was[0], was[1], was[2]):
            continue  # undeclared, or the declaration already agrees exactly
        if _drift_class(u + g, was[0] + was[1], was[0], u) != "moved":
            continue  # grew or shrank; the loop above owns those two
        if (u, g) == (was[0], was[1]):
            how = (f"the same {g} guard(s) by count, and the conditions on them "
                   f"are not the ones the digest recorded")
        else:
            how = (f"{u - was[0]:+d} unconditional and {g - was[1]:+d} guarded "
                   f"with the total flat, so a call site crossed the guard and "
                   f"neither column says which one")
        moved.append((name, was, (u, g, d), how))
    want_shrink = bool(shrunk) and "--accept-shrink" not in sys.argv
    want_moved = bool(moved) and "--accept-moved" not in sys.argv
    if want_shrink or want_moved:
        print("refusing to pin: the tree and the declaration disagree in a "
              "direction the two counts cannot see, and that is a decision "
              "rather than a regeneration", file=sys.stderr)
        flags = []
        if want_shrink:
            flags.append("--accept-shrink")
            print("  SHRANK -- fewer call sites than the declaration says, and a "
                  "gate that lost call sites tests less than it did:", file=sys.stderr)
            for s in shrunk:
                print(f"    {s}", file=sys.stderr)
            print("    This is the one thing the census exists for.", file=sys.stderr)
        if want_moved:
            flags.append("--accept-moved")
            print("  MOVED -- the same number of call sites under different "
                  "conditions, so nothing in this file's own arithmetic moved:",
                  file=sys.stderr)
            for name, was, got, how in moved:
                print(f"    {name}: derived {got[0]}+{got[1]}, declared "
                      f"{was[0]}+{was[1]}, total flat at {got[0] + got[1]}; "
                      f"{how}", file=sys.stderr)
                print(f"        guards sha256:{was[2][:12]} -> sha256:{got[2][:12]}",
                      file=sys.stderr)
                print(f"        the {len(guard_lists[name])} guard(s) it carries "
                      f"right now, which is what there is to look at:", file=sys.stderr)
                for gs in guard_lists[name]:
                    print(f"          | {_ascii_safe(gs)}", file=sys.stderr)
            # Said out loud rather than implied, because the sentence above is
            # the one an operator is most entitled to believe: it is not a
            # diff. The snapshot stores the digest, not the old guard strings,
            # so the guards that *changed* are the ones the diff of the gate
            # touches, and no amount of reading this file can recover them.
            #
            # One guard per line behind a `| `, and that marker is load-bearing
            # rather than decorative: it is what makes the list countable.
            # `clearing_paths_check.py` asserts that the number of lines here
            # equals the guarded count the declaration records, so a refusal
            # that truncated the list, summarised it, or printed a placeholder
            # cannot pass for one that named all of them.
            print("    Which of those changed is in your diff of the gate, not "
                  "here: the declaration stores the digest and not the old guard "
                  "strings, and that is the trade that made a hundred and fifty "
                  "strings untranscribable in the first place.", file=sys.stderr)
        print("Read the diff, and the census, before you decide either way:",
              file=sys.stderr)
        print("  python scripts/check_scripts_declare.py --census", file=sys.stderr)
        print("and if the change is what you meant, say so out loud:",
              file=sys.stderr)
        print("  python scripts/check_scripts_declare.py --pin " + " ".join(flags),
              file=sys.stderr)
        return 2
    body = ["DECLARED_SNAPSHOT = {"]
    for name in sorted(rows):
        u, g, d = rows[name]
        body.append(f'    "{name}": ({u}, {g},')
        body.append(f'        "{d}"),')
    body.append("}")
    new_region = "\n".join([PIN_BEGIN + " (python scripts/check_scripts_declare.py --pin) ---",
                            *body, PIN_END])
    src = io.open(TARGET, encoding="utf-8").read()
    out = _splice_region(src, SNAP_HEADS, SNAP_FOOT, new_region, "snapshot")
    if out is None:
        return 2

    # The blind-spot table is regenerated in the same pass and by the same
    # command, deliberately. Two generated regions written by one `--pin` can
    # never disagree about which run produced them; two written by two
    # invocations can, and the disagreement would look like a real finding.
    #
    # A gate already listed keeps its reason: `--pin` writes counts, and a
    # reason is prose somebody wrote about a decision. Re-deriving the count
    # and keeping the note is the only combination that does not silently
    # delete an argument, so a count that moves is visible in the diff as a
    # count and not as a vanished paragraph.
    bbody = [BLIND_BEGIN + _PIN_SUFFIX,
             "#: Every check-like call in the inventory that the census above cannot",
             "#: count, because the call is not the whole statement. A gate appearing",
             "#: here for the first time is the red this table exists to produce.",
             "#: --- a gate name is followed by (count, why it is there) ---",
             "BLIND_SPOTS: dict[str, tuple[int, str]] = {"]
    blind_rows = []
    for name in sorted(INVENTORY):
        path = SCRIPTS / name
        if not path.exists():
            continue
        try:
            hits = _blind_sites_of(read(path))
        except SyntaxError:
            continue
        if not hits:
            continue
        prev = BLIND_SPOTS.get(name)
        why = (prev[1] if prev and prev[1].strip()
               else "no reason recorded yet -- say why this call is not the "
                    "whole statement")
        blind_rows.append((name, len(hits), why))
    for name, n, why in blind_rows:
        flat = " ".join(why.split())
        bbody.append(f'    "{name}": ({n},')
        bbody.append(f'        "{flat}"),')
    bbody.append("}")
    bbody.append(BLIND_END)
    out = _splice_region(out, BLIND_HEADS, BLIND_FOOT, "\n".join(bbody),
                         "blind-spot")
    if out is None:
        return 2

    # newline="" so the file's own endings survive; utf-8 explicitly, because
    # PowerShell 5.1's -Encoding UTF8 writes a BOM and would corrupt the em
    # dashes this file is full of.
    with io.open(TARGET, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"pinned {len(rows)} gate(s) into the generated snapshot:")
    for name in sorted(rows):
        u, g, d = rows[name]
        print(f"  {name:<38} {u:>4} unconditional {g:>3} guarded  sha256:{d[:12]}")
    print(f"pinned {len(blind_rows)} gate(s) into the generated blind-spot table:")
    for name, n, _why in blind_rows:
        print(f"  {name:<38} {n:>4} site(s) the census cannot see")
    skipped = [n for n in INVENTORY
               if (SCRIPTS / n).exists() and _declaration_of(SCRIPTS / n) is not None]
    print(f"left alone (declares its own block): {sorted(skipped)}")
    print("Now read the diff. A census you did not expect to move is the thing "
          "this command exists to make visible.")
    return 0


#: The four things a disagreement between a gate's declared census and the
#: derived one can be, and the single function that decides which. Both the red
#: below and `--census` call it, because two implementations of a four-way
#: classification is one more thing that can drift.
#:
#: **It lives above `_census_report`, not next to the loop it classifies.**
#: `--census` is dispatched before the census loop runs, so a classifier defined
#: beside the loop is a `NameError` in the command that exists to explain the
#: loop's output.
_DRIFT_KINDS = {
    "grew": "GREW  (the gate gained checks; re-baselining is bookkeeping)",
    "shrank": "SHRANK (the gate LOST call sites and will test less; read twice)",
    "moved": "MOVED (same total, different guard strings -- the digest's job)",
    "undeclared": "UNDECLARED (nothing in this tree says what it should hold)",
}


def _drift_class(got_n: int, want_n, d_uncond, got_uncond) -> str:
    """Which of the four kinds of drift this is.

    `want_n` is None when there is no declared number to compare against, which
    is the only route to `undeclared`. A total is enough to tell grew from
    shrank; the caller adds the opposite-columns note separately, because that
    note is about the *columns* and this is about the *total*.
    """
    if want_n is None:
        return "undeclared"
    if got_n > want_n:
        return "grew"
    if got_n < want_n:
        return "shrank"
    return "moved"


def _drift_kind(got_n: int, want_n, d_uncond, got_uncond) -> str:
    return _DRIFT_KINDS[_drift_class(got_n, want_n, d_uncond, got_uncond)]


def _census_report() -> int:
    """Print what every drifted gate's declaration *should* say. Write nothing.

    Run:  python scripts/check_scripts_declare.py --census

    **This is the honest middle, and it exists because `--pin` cannot reach
    three of the four drifted gates on this tree.** `--pin` regenerates
    `DECLARED_SNAPSHOT`, which is a table in *this* file, so it can only speak
    for the fourteen gates on it. A `GATE-DECLARE` block lives in the gate's own
    source, and a block is an edit to a file this auditor does not own --
    `docs_claims_check.py` drifted 90+10 -> 89+13 and no invocation of `--pin`
    will ever put that right, because the number is in somebody else's file.

    So this mode does the half `--pin` cannot: for every gate, block or
    snapshot, it prints the exact declaration the tree currently implies, as a
    unified diff against what is there, and **writes nothing at all**. The
    artefact on disk is unchanged until a human pastes it, which keeps the
    regeneration an explicit, reviewable act instead of a side effect.

    The distinction from `--pin` is the whole point and it is not subtle:

    * `--census` is a *read*. Exit code 1 if anything has drifted, 0 if not, and
      the file is byte-identical either way. It is safe to run at any time, on
      any tree, including one somebody is editing.
    * `--pin` is a *write* to this file, and it is the only thing that may.

    Exits 1 when something has drifted, so it composes with a pre-commit hook
    the way the normal run does.
    """
    _restore_stdout()
    moved, unreadable = [], []
    for name in INVENTORY:
        path = SCRIPTS / name
        if not path.exists():
            continue
        try:
            got_uncond, got_guards = _sites_of(read(path))
        except SyntaxError as exc:
            unreadable.append(f"{name} ({exc})")
            continue
        d_uncond, d_guarded, d_digest, where = _declared_for(name, path)
        got_n, got_digest = got_uncond + len(got_guards), _guards_digest(got_guards)
        # `have` and `want` are built in the *same* notation on purpose. An
        # earlier version of this compared a snapshot row against a
        # `GATE-DECLARE` block, which are different shapes, so all fourteen
        # snapshot gates read as drifted -- including the ten whose numbers were
        # already right. A diff that reports a disagreement it invented is
        # worse than no diff, and it is the same mistake as the digest for a
        # gate with no guards: comparing two things that are not the same kind
        # of thing.
        if d_uncond is None:
            kind = "UNDECLARED -- nothing in this tree says what it should hold"
            have = ["<no declaration: %s>" % where]
            want = ["#: GATE-DECLARE %s" % DECL_VERSION,
                    "#: sites: %d unconditional + %d guarded"
                    % (got_uncond, len(got_guards)),
                    "#: guards: sha256:%s" % got_digest]
        elif "its own GATE-DECLARE block" in where:
            kind = _drift_kind(got_n, d_uncond + d_guarded, d_uncond, got_uncond)
            # Only the three declaration lines, not the comment they live in:
            # a diff carrying a gate's prose cannot be reviewed.
            have = [ln for ln in _comment_lines(path)
                    if _DECL_HEAD_RE.match(ln) or _DECL_SITES_RE.match(ln)
                    or _DECL_GUARDS_RE.match(ln)]
            want = ["#: GATE-DECLARE %s" % DECL_VERSION,
                    "#: sites: %d unconditional + %d guarded"
                    % (got_uncond, len(got_guards)),
                    "#: guards: sha256:%s" % got_digest]
        else:
            kind = _drift_kind(got_n, d_uncond + d_guarded, d_uncond, got_uncond)
            have = ['    "%s": (%d, %d,' % (name, d_uncond, d_guarded),
                    '        "%s"),' % d_digest]
            want = ['    "%s": (%d, %d,' % (name, got_uncond, len(got_guards)),
                    '        "%s"),' % got_digest]
        if have == want:
            continue
        moved.append((name, kind, where, have, want))
    for name, kind, where, have, want in moved:
        print("=" * 74)
        print(f"{name}  [{kind}]")
        print(f"  declared in: {where}")
        for line in difflib.unified_diff(have, want, "as declared", "as derived",
                                         lineterm="", n=1):
            print("  " + line)
    if unreadable:
        print()
        print("not measured, and no diff is possible for these:")
        for u in unreadable:
            print(f"  {u}")
    print()
    if not moved and not unreadable:
        print("every gate's declaration matches the tree. Nothing to re-baseline.")
        return 0
    print(f"{len(moved)} gate(s) drifted. Nothing above was written to disk.")
    snap = [n for n, _k, w, _h, _t in moved if "snapshot" in w]
    blocks = [n for n, _k, w, _h, _t in moved if "block" in w]
    if snap:
        print(f"  on the generated snapshot ({len(snap)}): "
              f"python scripts/check_scripts_declare.py --pin")
    for n in blocks:
        print(f"  in its own block: paste into {n} the three lines above")
    return 1


if "--census" in sys.argv:
    sys.exit(_census_report())

if "--pin" in sys.argv:
    sys.exit(_pin())


_drifted = []
_unparseable = []
_measured = 0
_census: dict[str, tuple[int, int, str]] = {}
_in_block: list[str] = []
_snapshot_only: list[str] = []
_flying: list[str] = []
#: Every disagreement between a gate's declared census and the derived one,
#: filed under **which kind of event it is** rather than lumped into one list.
#:
#: **This is the finding, and it is why the four reds in one session were all
#: noise.** A run against this tree reports four drifted gates, and all four
#: are the *same shape of event*: the gate gained checks. `cli_check.py` 118 ->
#: 120, `core_check.py` 282 -> 357, `workbench_interaction_check.py` 245 -> 297,
#: and `docs_claims_check.py` 100 -> 102. Adding a check is how a gate is
#: supposed to grow, and the remedy is to re-baseline a number. The red did not
#: distinguish that from the one event that matters: a gate that has *lost* call
#: sites, and will from now on test less than it did, silently and forever.
#:
#: A deletion and an addition produced byte-identical red text in the mutation
#: proof -- `derived 78+6, declared 79+6` and `derived 80+6, declared 79+6` --
#: because the message was a subtraction and said nothing about its sign. So the
#: partition below is not decoration. `_shrank` is the class that must never be
#: waved through, `_moved` is the class that must never be waved through
#: *quietly*, `--pin` refuses to absorb the first without `--accept-shrink` and
#: the second without `--accept-moved`, and the other two carry their own
#: remedy.
#:
#: `docs_claims_check.py` is the case that shows why a total is too coarse: it
#: went 90+10 -> 89+13, so its unconditional column *fell by one* while the
#: total rose by two. One site moved into a guard and three were added. "The
#: number went up" is true and useless; "the unconditional count went down" is
#: the half a reader needs.
_grew: list[str] = []
_shrank: list[str] = []
_moved: list[str] = []
_undeclared: list[str] = []
#: The same four lists under their class names, so the loop below and
#: `_drift_class` cannot be two implementations of one decision.
_BY_CLASS = {"grew": _grew, "shrank": _shrank, "moved": _moved,
             "undeclared": _undeclared}
for name in INVENTORY:
    path = SCRIPTS / name
    if not path.exists():
        # Already a failure in its own right ("no check script has been deleted
        # from the inventory"); counting it again here would double-report one
        # fact. Declared-but-absent is covered by that check.
        continue
    try:
        got_uncond, got_guards = _sites_of(read(path))
    except SyntaxError as _exc:
        # A gate that will not parse is a real problem, but it is not a problem
        # this loop can measure a call-site count for. Counting what we could not
        # read and calling it a drift would be a guess, so it is a skip with a
        # reason -- and the summary below prints it beside the pass count so it
        # cannot be mistaken for one.
        _unparseable.append(name)
        skip(f"inventory the call sites in {name}", f"it does not parse: {_exc}")
        continue
    _measured += 1
    d_uncond, d_guarded, d_digest, where = _declared_for(name, path)
    _census[name] = (got_uncond, len(got_guards), _guards_digest(got_guards))
    # Which of the two sources this gate's declaration comes from is recorded
    # before any deferral: "where does every gate's number live" must account for
    # all measured gates, including ones whose comparison is deferred.
    if "its own GATE-DECLARE block" in where:
        _in_block.append(name)
    elif "the generated snapshot" in where:
        _snapshot_only.append(name)
    if _is_in_flight(path):
        # Deferred, not skipped: the census is still derived and still printed,
        # it is simply not held against a declaration yet. See `_is_in_flight`.
        _flying.append(f"{name} ({_age_minutes(path):.0f} min since last write)")
        continue
    if d_uncond is None:
        _undeclared.append(f"{name}: {where}")
        _drifted.append(f"{name}: {where}")
        print(f"[FAIL] {name}: {where}")
        print(f"       derived {got_uncond} unconditional / {len(got_guards)} "
              f"guarded, guards sha256:{_guards_digest(got_guards)[:12]}...")
        continue
    got_n, want_n = got_uncond + len(got_guards), d_uncond + d_guarded
    if (got_uncond, len(got_guards)) != (d_uncond, d_guarded):
        note = ""
        if (got_uncond - d_uncond) * (len(got_guards) - d_guarded) < 0:
            # The two columns moved in opposite directions: a total is the wrong
            # summary of this and the sentence has to say so.
            note = (f"; the columns moved in OPPOSITE directions "
                    f"({got_uncond - d_uncond:+d} unconditional, "
                    f"{len(got_guards) - d_guarded:+d} guarded), so the total "
                    f"alone hides a site that stopped being unconditional")
        msg = (f"{name}: derived {got_uncond}+{len(got_guards)}, declared "
               f"{d_uncond}+{d_guarded}{note}")
        _drifted.append(msg)
        _BY_CLASS[_drift_class(got_n, want_n, d_uncond, got_uncond)].append(msg)
    elif _guards_digest(got_guards) != d_digest:
        msg = (f"{name}: same census, different guards "
               f"(declared sha256:{d_digest[:12]}..., derived "
               f"sha256:{_guards_digest(got_guards)[:12]}...)")
        _drifted.append(msg)
        _BY_CLASS["moved"].append(msg)


check(
    not _drifted and not _unparseable,
    "every gate declares the call sites it actually has",
    f"{_measured} of {len(INVENTORY)} gates measured, "
    f"{sum(u + g for u, g, _d in _census.values())} call sites across them. "
    f"{len(_grew)} gate(s) GREW, {len(_shrank)} SHRANK, {len(_moved)} moved shape "
    f"with the total flat, {len(_undeclared)} undeclared. "
    f"GREW: {_grew or 'none'} -- a gate gained checks, which is how gates grow; "
    f"re-baseline it with `--pin`, or paste the block `--census` prints. "
    f"SHRANK: {_shrank or 'none'} -- a gate LOST call sites and will test less "
    f"than it did; this is the one this census exists for and `--pin` refuses to "
    f"absorb it without `--accept-shrink`. MOVED: {_moved or 'none'} -- same "
    f"total, different guard strings, which is the change the two counts cannot "
    f"see and the digest exists to catch; `--pin` refuses to absorb one of "
    f"those without `--accept-moved` either, because rewriting the digest is "
    f"how a moved gate used to become invisible. UNDECLARED: "
    f"{_undeclared or 'none'}",
)

# The partition above is the load-bearing part of that message, and a partition
# nothing checks is a comment. This asserts it is **total and disjoint**: every
# entry in `_drifted` is in exactly one of the four classes, and no gate is in
# two. Without it a future edit that files a drift under no class at all would
# still print a tidy four-way summary that silently omits the gate -- and the
# class being omitted is the one whose absence is the defect.
_partition = _grew + _shrank + _moved + _undeclared
_overlap = [g for c in (_grew, _shrank, _moved, _undeclared) for g in c
            if c.count(g) > 1]
check(
    sorted(_partition) == sorted(_drifted) and not _overlap,
    "every drifted gate is filed under exactly one kind of drift",
    f"{len(_drifted)} drifted, {len(_grew)} grew + {len(_shrank)} shrank + "
    f"{len(_moved)} moved + {len(_undeclared)} undeclared = {len(_partition)} "
    f"filed, {len(_overlap)} in more than one class ({_overlap or 'none'}). "
    f"GREW and SHRANK are the same subtraction with opposite signs and they "
    f"mean opposite things -- one is how a gate grows, the other is a gate that "
    f"has stopped testing something -- so the run reports them as two numbers "
    f"and `--pin` will not absorb the second one silently. A drift in no class "
    f"at all would print a summary that quietly omits a gate, which is the "
    f"failure mode this whole census is about",
)

# --------------------------------------------------------------------------
# The census's blind spot, counted on purpose
# --------------------------------------------------------------------------
# The check above compares two columns that `_sites_of` produced, so a call
# that walk cannot see moves neither and cannot go red. That is not a defect in
# the comparison -- it is load-bearing for the 1932 sites it does see -- but it
# does mean "the census is complete" was never true, and until now nothing
# measured the remainder. This measures it.
#
# **What would have gone wrong without it, stated as the mutation.** Take
# `cli_check.py` and add one line inside `fails_cleanly`:
#
#     extra = check("a new assertion", True)
#
# Before this check: `_sites_of` is unchanged at 120, the declared snapshot is
# unchanged at 118, the digest is unchanged, and `check_scripts_declare.py`
# exits 0. A gate has gained a result that produces output, prints a line, and
# returns a bool the caller ignores -- and the ledger that is supposed to
# account for every result site in this repository cannot see that it exists.
# After this check: the derived blind count for `cli_check.py` goes 3 -> 4
# against a declared 3, and the run goes red naming the gate.
#
# The mutation is run for real, on a copy, immediately below. A check whose
# failure direction has only been argued is a comment.
_blind: dict[str, list[tuple[int, str]]] = {}
for _name in INVENTORY:
    _p = SCRIPTS / _name
    if not _p.exists():
        continue
    try:
        _blind[_name] = _blind_sites_of(read(_p))
    except SyntaxError:
        # Already reported as a skip by the loop above; counting it here as
        # "no blind sites" would assert something about a file nobody read.
        _blind[_name] = []

_blind_drift = []
_blind_undeclared = []
for _name in sorted(_blind):
    _got = len(_blind[_name])
    _decl = BLIND_SPOTS.get(_name)
    if _decl is None:
        if _got:
            _blind_undeclared.append(f"{_name}: {_got} undeclared")
        continue
    if _got != _decl[0]:
        # One condition, both directions. A gate that closed its last blind
        # site is as wrong here as one that opened a new one: the first is a
        # table entry describing a file that no longer exists, which is the
        # stale-snapshot failure this file already has a check for elsewhere.
        _blind_drift.append(f"{_name}: derived {_got}, declared {_decl[0]}")
    elif not _decl[1].strip():
        _blind_drift.append(f"{_name}: declared with no reason")

check(
    not _blind_drift and not _blind_undeclared,
    "every result site the census cannot see is counted here",
    f"{sum(len(v) for v in _blind.values())} blind site(s) across "
    f"{sum(1 for v in _blind.values() if v)} of {len(_blind)} gates; "
    f"{len(BLIND_SPOTS)} declared. drifted: {_blind_drift}; undeclared: "
    f"{_blind_undeclared}. The census counts a call only where the call is the "
    f"whole statement, so `r = check(...)` and `return check(...)` are result "
    f"sites it cannot see -- and because both columns of the check above come "
    f"from the same walk, such a site moves neither column. This is the second "
    f"half of that: it is now a declared number per gate, so adding one goes "
    f"red instead of passing unseen. A gate that closed its last blind site "
    f"goes red too, so the table cannot keep describing a file that moved",
)

# The self-test. A copy of a real gate, mutated to add one blind site, must
# move the derived count and nothing else -- that is the property the check
# above is relying on, and it is worth more asserted than explained.
_MUT = "check(True, 'a site the census cannot see')\n"
_probe_src = read(SCRIPTS / "cli_check.py")
_probe_before_u, _probe_before_guards = _sites_of(_probe_src)
_probe_before_n = len(_probe_before_guards)
_probe_before_blind = len(_blind_sites_of(_probe_src))
_probe_after = _probe_src.replace(
    "    return spoke and quiet",
    "    _extra = " + _MUT + "    return spoke and quiet", 1)
if _probe_after == _probe_src:
    _probe_ok = False
    _probe_why = ("the anchor `return spoke and quiet` was not found in "
                  "cli_check.py, so the mutation did not land and this proves "
                  "nothing")
else:
    _au, _ag = _sites_of(_probe_after)
    _ab = len(_blind_sites_of(_probe_after))
    _probe_ok = ((_au, len(_ag)) == (_probe_before_u, _probe_before_n)
                 and _ab == _probe_before_blind + 1)
    _probe_why = (
        f"on a copy of cli_check.py with one `_extra = check(...)` added: the "
        f"census stayed {_probe_before_u}+{_probe_before_n} "
        f"(derived {_au}+{len(_ag)}) and the blind count went "
        f"{_probe_before_blind} -> {_ab}. The first half is the gap this file "
        f"had; the second is the only reason the check above can catch it"
        if _probe_ok else
        f"the mutation moved the wrong numbers: census {_probe_before_u}+"
        f"{_probe_before_n} -> {_au}+{len(_ag)}, blind {_probe_before_blind} "
        f"-> {_ab}. The check above is not measuring what it claims")

check(
    _probe_ok,
    "the census-blind-site check can go red on a gate that adds one",
    _probe_why,
)

check(
    len(_in_block) + len(_snapshot_only) == _measured,
    "every gate's declared census comes from exactly one place",
    f"{len(_in_block)} declare their own block, {len(_snapshot_only)} come from "
    f"the generated snapshot. A gate may not appear in both: the block wins and "
    f"the snapshot is regenerated by --pin, which omits any gate that has a "
    f"block. One source per gate is what makes 'which number did this run "
    f"compare against' a question with an answer",
)

# The rule the detail line above states, as a check. `_declared_for` prefers a
# block and returns without ever consulting the snapshot, so a gate left in both
# places is not a disagreement this file can see -- it compares against the
# block, passes, and the stale snapshot number sits there until the next `--pin`
# silently deletes it. Stating the rule in prose and not testing it is the same
# shape as the `SITE_INVENTORY` hole this file was written to end, so it is
# tested. `examples_check.py` is the control that proves the check can bite: it
# carried a snapshot entry until it grew a block, and this went red on it.
_in_both = sorted(set(_in_block) & set(DECLARED_SNAPSHOT))
check(
    not _in_both,
    "no gate is left in the snapshot after it grew a block of its own",
    f"{_in_both} appear in both DECLARED_SNAPSHOT and a GATE-DECLARE block. The "
    f"block is what the comparison uses, so the snapshot copy is dead weight "
    f"that --pin would delete without comment -- and if the block were ever "
    f"removed, the stale number would come back as the declaration. Delete the "
    f"snapshot entry, not the block"
    if _in_both else
    f"none of the {len(_in_block)} block-declaring gates is still listed in the "
    f"snapshot, so each declared number has exactly one home",
)

check(
    len(_snapshot_only) <= MAX_SNAPSHOT_ONLY,
    f"no more than {MAX_SNAPSHOT_ONLY} gates rely on the generated snapshot",
    f"{len(_snapshot_only)} against a cap of {MAX_SNAPSHOT_ONLY}: "
    f"{sorted(_snapshot_only)}. This is not an exemption -- all of these gates are "
    f"measured and all of them go red on drift. It counts only which file the "
    f"expected number is written in. Paste a GATE-DECLARE block into a gate's own "
    f"file (the block for its current census is printed below) and lower the cap",
)

check(
    len(_flying) <= MAX_IN_FLIGHT,
    f"no more than {MAX_IN_FLIGHT} gates are deferred as in flight",
    f"{len(_flying)} deferred: {_flying or 'none'}. Deferred means: the file was "
    f"written *after this run started*, so its census is derived and printed but "
    f"not yet compared against a declaration. Keyed on mtime against RUN_START "
    f"rather than on a name in a table, so it covers a save landing mid-read and "
    f"nothing else -- a file saved a minute ago is measured, and on a fresh CI "
    f"checkout nothing is ever deferred. It cannot be parked, and the cap is "
    f"asserted on the measured size",
)
print()
print("  call-site census -- declared against derived, and where each is declared:")
print()
for name in sorted(_census):
    u, g, dg = _census[name]
    print("    %-38s %4d unconditional  %3d guarded   %s"
          % (name, u, g, "own block" if name in _in_block else "snapshot"))
print()
print("  standing debt -- gates with no EXPECTED_CHECKS, derived from the files:")
print("  (a call-site census, not a run total; the two are different quantities")
print("   and the old hand-typed ledger used to present one as the other)")
print()
for name in sorted(undeclared):
    _c = _census.get(name)
    print("    %-34s %s" % (
        name,
        "%4d unconditional  %3d guarded" % (_c[0], _c[1]) if _c
        else "(census unavailable -- see the skip list)",
    ))
print()
print("  paste-ready GATE-DECLARE blocks, for gates still on the snapshot:")
print()
for name in sorted(_snapshot_only):
    u, g, dg = _census[name]
    print("    # ---- %s ----" % name)
    print("    #: GATE-DECLARE %s" % DECL_VERSION)
    print("    #: sites: %d unconditional + %d guarded" % (u, g))
    print("    #: guards: sha256:%s" % dg)
print()

# ==========================================================================
section("a check that did not run must say so, not count as a pass")

#: Correction to something this file claimed in an earlier round. It said "no
#: script in the repo defines `skip()`". That was wrong. `skip` exists in seven
#: scripts, and it has been there the whole time. What is actually true is worse
#: in one place and better in another, so the real audit is not "is skip
#: missing" but "what does each file's skip do to its tally".
#:
#: Three conventions are in use and they do not agree:
#:
#:   counts      the skip is recorded as a result *and* counts toward the total,
#:              so the pin is the same number on every machine
#:              (`workbench_interaction_check.py`)
#:   third_tag   the skip is recorded as its own tag and reported on a separate
#:              counter, and zero passes is escalated to "did not finish"
#:              (`viewport_framing_check.py`)
#:   uncounted   the skip prints a line and is recorded nowhere and counted
#:              nowhere, so the tally cannot tell a clean run from a run that
#:              could not measure three things
#:              (`structure_bond_check.py`, which says so in its own docstring)
#:
#: `counts` is the one to converge on. `third_tag` is defensible and its
#: zero-pass rule is worth keeping. `uncounted` is the failure this section
#: exists to make visible: the run says "N passed" either way.
#:
#: **These labels are a hand audit, not a measurement, and the difference is not
#: hidden.** An earlier version of this section tried to measure all three from
#: the source and had to be thrown away: it could not see a skip that records
#: itself by delegating to a helper (`viewport_framing_check.skip` calls
#: `record`), and it read this file's own `len(results)` -- which appears in a
#: comment -- as evidence that its skip counted toward the total. A ledger of
#: heuristics is the same failure as a transcribed number wearing a comment, so
#: what is asserted below is only what can be measured: that every gate defining
#: `skip()` is classified, that the labels are the ones on file, and that no
#: file's `skip` can increment its own check counter. The label itself is
#: asserted by a human and can be wrong; it is a record of a reading, not a proof.
SKIP_CONVENTION = {
    "binary_source_parity_check.py":
        "records a skip as its own tag and counts it toward the pinned total, so "
        "a fresh clone -- which has no `_dockpy.pyd`, because `.gitignore` line "
        "17 is `*_dockpy.pyd` and git has never tracked one -- reports the same 8 "
        "as a tree that built one, with the five binary-dependent checks skipped "
        "rather than passed. That is the point rather than an inconvenience: a "
        "clone that reported 'nothing to compare' as a pass would be "
        "indistinguishable from a tree whose binary is current, and the whole "
        "subject of this gate is a binary nobody was watching. A skip must never "
        "be able to reach the total by declining to measure",
    "check_repo_docs.py":
        "records a skip as its own tag and counts it toward the pinned total, so "
        "an interpreter with no PyYAML reports the same 9 as one that has it -- "
        "the YAML result becomes a skip and the run is still 9, with 7 passed, 0 "
        "failed, 1 skipped. This also repairs a false statement: the old code "
        "printed 'PyYAML not installed; skipping YAML validation' and then, with "
        "nothing found to skip, printed 'all YAML parses' and exited 0, so on a "
        "machine without PyYAML it claimed a measurement it had not made. The "
        "prose summary is now assembled from what the run did rather than from "
        "what it intended to do",
    "check_text_encoding.py":
        "records a skip as its own tag and counts it toward the pinned total, so "
        "a run pointed at a root holding nothing in scope reaches the same total "
        "as one that scanned the whole tree -- of everything outside `SKIP_DIRS`, "
        "the files it does not judge are exactly those excluded by suffix, "
        "refused by `looks_binary`, or both, and each of those exclusions is "
        "named in that file with a reason. Passes and skips print as two numbers "
        "that are never added. **No total and no file count is transcribed "
        "here, deliberately:** both move while other agents add files, and this "
        "file's own pin moved 20 -> 22 -> 23 in one afternoon with no change to "
        "the scope decision. The number to read is `EXPECTED_CHECKS` in that "
        "file, and the number of files is printed by its own `SCOPE` block on "
        "every run. A gate that must not scan anything must not report the same "
        "thing as one that scanned and found nothing. Measured on 2026-10-02 "
        "against an empty `OD_ENCODING_SCAN_ROOT`: 23 checks, 14 passed, 8 "
        "skipped, **1 failed** -- and the failure is a finding rather than a "
        "wart, because a root with no files in it is reported as not a tree "
        "rather than passing as a clean scan",
    "gpu_cpu_parity_check.py":
        "records a skip as its own tag and counts it toward the pinned total, so "
        "a machine with no GPU adapter reports the same 9 as one that has one -- "
        "with the parity checks among them skipped rather than passed. That is the "
        "point rather than an inconvenience: this gate is red on any machine with "
        "an adapter until the two backends agree outside the box, and green by "
        "skip on the ones that cannot check. A skip must never be able to reach "
        "the total by declining to measure",
    "gpu_feature_check.py":
        "records the skip as its own tag on RESULTS, and `block_pinned_total` "
        "asserts `npass + nfail + nskip + 1 == EXPECTED_CHECKS` at its line 662, "
        "so the skip sits inside the partition of the pin rather than beside it: "
        "a machine with no GPU adapter reaches the same 11 as one that has one. "
        "The summary prints `nskip` and `len(RESULTS)` as separate numbers at its "
        "line 805, and `len(RESULTS) != EXPECTED_CHECKS` returns "
        "EXIT_COUNT_MISMATCH rather than reporting the shortfall as success, so "
        "the three tags are closed and cannot be quietly added to. This is the "
        "`third_tag` convention and it earns the entry on the partition, not on "
        "the tag: the tag is a label, and the `+ 1` self-count at line 664 is "
        "what makes a padded pin fail",
    "check_scripts_declare.py":
        "records the skip in its own list, keeps it out of CHECKS, and prints the "
        "count beside the pass count",
    "extension_surface_check.py":
        "records the skip as a third tag on RESULTS and counts it toward the "
        "pinned total, so a machine whose compiled extension cannot be imported "
        "reports the same 6 as one that can. That matters more than usual here: "
        "this gate imports the extension but never opens it and never calls into "
        "it -- calling is the action that raises the arity TypeError it exists to "
        "report, so a check that had to call the thing under test could not "
        "survive the mismatch it is looking for",
    "framing_selection_check.py":
        "records the skip as a third tag on RESULTS, and the skip() docstring says "
        "in words that it counts toward EXPECTED_CHECKS; the meta-check added this "
        "session asserts pass+fail+skip == len(RESULTS) and that the printed total "
        "is len(RESULTS)+1, so the three partition the tally rather than sitting "
        "beside it",
    "installed_copy_check.py":
        "records the skip as a third tag on RESULTS, and the tally asserts "
        "npass + nfail + nskip == len(RESULTS) == EXPECTED_CHECKS - 1, so a skip "
        "is inside the partition and the pinned total is the same with and "
        "without an install -- an absent install is a normal machine, so it "
        "must read as 6 skips and not as a smaller total. It also separates a "
        "run that did not finish from a run that did, in the text rather than "
        "in the exit code",
    "pose_trust_check.py":
        "records the skip as a result that counts toward the pinned total, the "
        "same shape as workbench_interaction_check.py: the total is the same "
        "number whether or not the machine could answer, and a skip is printed "
        "as a result rather than folded into the pass count",
    "provenance_appendix.py":
        "takes a `counts` argument, and that is the whole classification. Its "
        "line 547 records every skip in SKIPPED and emits it, but adds it to the "
        "pinned total only when `counts=True`, which is the default. The two "
        "directions are not the same shape, so both are named here rather than "
        "filed under one of the three labels: the counted form is the "
        "`counts` convention, and it is closed by `CHECKS + COUNTED_SKIPS + 1 == "
        "EXPECTED_CHECKS` at its line 2878, so it cannot reach the pin by "
        "declining to measure. The uncounted form is the opposite of padding -- "
        "it *removes* a result from the pin rather than adding one -- and it is "
        "marked `SKIP*` at line 566 so the reader can see which is which, with "
        "line 1110 the one call site that uses it, for a prose file that is not "
        "present and so is not a slot this file owed a result in. The summary "
        "prints `COUNTED_SKIPS` and the slot total as separate numbers at lines "
        "2892-2893, so a run that skipped is not a run that passed. This file's "
        "census is being moved by another author while this entry was written; "
        "any drift it shows is theirs and not a defect in the skip body read here",
    "check_counted_constants.py":
        "records the skip as a third tag on RESULTS, counts it toward the pinned "
        "total, and prints 'N passed, M failed, K skipped' as three numbers, so "
        "the tally line carries it. **The verdict line does not**: `finish()` "
        "returns OK whenever `nf == 0` and its `RESULT: OK --` text never mentions "
        "a skip, so on a run that skipped, the last line a reader sees is a green "
        "one and the skip count is two lines above it. That is recorded here "
        "rather than fixed because this file's owner is not available in this "
        "round, and the honest reading of this entry is 'a skip here is counted "
        "and printed, and is not in the verdict'",
    "raw_context_bound_check.py":
        "records the skip as a result that counts toward the pinned total, and "
        "it is the same shape as workbench_interaction_check.py for the same "
        "reason this gate was built at all: 'this machine cannot make an OpenGL "
        "context' used to be the entire finding, and a run that printed nothing "
        "is indistinguishable from a run with nothing to say. Its summary reads "
        "`RESULT: PASS (1 could not be measured here)` with the pass count "
        "excluding it, so the skip is visible in the exit status's own text "
        "rather than inferred from a total that did not move",
    "representation_cartoon_check.py":
        "records the skip as a third tag on RESULTS, counts it toward the pinned "
        "total on purpose, and prints 'N check(s) were SKIPPED, not passed' "
        "separately from the pass/fail/skip/check summary",
    "screenshot_frame_check.py":
        "records the skip as a third tag on RESULTS and asserts inside its own run "
        "that npass + nfail + nskip == len(RESULTS) == EXPECTED_CHECKS - 1, so the "
        "tag is inside the partition and a padded pin cannot pass",
    "skip_reasons_check.py":
        "records the skip on RESULTS as a third tag and deliberately does NOT "
        "increment its check counter, so a skip cannot let this file reach its pin "
        "by declining to look -- the one failure mode the file exists to detect. "
        "Its incomplete-enumeration case does not ride on the tally at all: it is a "
        "separate summary line and exit 3, which outranks the failing exit 1",
    "structure_bond_check.py":
        "records the skip in its own list, counts it into the pinned total, and "
        "prints pass/fail/skip as three separate numbers that never sum to a "
        "verdict; the total is asserted to partition into the three",
    "unreached_product_check.py":
        "records the skip as a result that does NOT count toward the pinned "
        "total, and prints passes and skips as two numbers it never adds. The "
        "one skip is the __all__ completeness check over "
        "`workbench/__init__.py`'s `Workbench` entry, which is neither "
        "defined nor imported there: `from opendocking.workbench import "
        "Workbench` raises ImportError and `import *` raises AttributeError. "
        "It is carried as a skip rather than a failure because it is a real "
        "defect in a file this gate does not own, so a failure would make the "
        "gate permanently red over somebody else's one-line fix -- and because "
        "a skip that reached the total by declining to measure is exactly the "
        "failure this repository keeps meeting. The recorded skip is the "
        "stronger outcome, not the weaker one: it names the defect and leaves "
        "it visible",
    "viewport_framing_check.py":
        "records the skip as a third tag, counts it on a separate counter, and "
        "turns zero passes into 'did not finish' (exit 2)",
    "wheel_payload_check.py":
        "records the skip as a third tag and counts it toward the pinned total, "
        "so a machine with no wheel built and a machine with a current wheel "
        "both report 6 -- a machine that has not run `maturin build` is a "
        "normal machine. The same rule covers a wheel that exists but predates "
        "the tree: its payload is not a statement about this tree, so the four "
        "payload checks skip and the reason carries the exact age in seconds, "
        "rather than a verdict that would agree with a wheel the gate should "
        "have refused",
    "workbench_interaction_check.py":
        "records the skip as a result that counts toward the total, so the pin is "
        "the same number on every machine",
}

#: Gates that name a skip but whose summary cannot account for it, so their
#: tally reads the same whether or not the check happened. A ratchet.
#:
#: **Empty, and the emptiness is not a claim that nothing is owed.** The only
#: entry was `structure_bond_check.py`, which has been repaired: its `skip`
#: now records, counts into the pinned total, and the summary prints
#: pass/fail/skip as three numbers asserted to partition that total. Read this
#: as "the last known debt was paid", not as "the debt is zero" -- the
#: limitation note at the assertion below is the load-bearing part of this
#: entry.
#:
#: Keep the dict. An empty ledger and no ledger are different things, and only
#: one of them records that anyone looked.
UNNAMED_SKIP_TALLIES: dict[str, str] = {}
MAX_UNNAMED_SKIP_TALLIES = 0


def _defines_skip(path: Path) -> bool:
    try:
        tree = ast.parse(read(path))
    except SyntaxError:
        return False
    return any(
        isinstance(n, ast.FunctionDef) and n.name == "skip" for n in ast.walk(tree)
    )


def _skip_touches_the_counter(path: Path, counter: str) -> bool:
    """Does this file's `skip()` body increment `counter`?

    This is the one property of the skip mechanism that is worth measuring rather
    than asserting, and it is measurable exactly: a name is either written to
    inside the function body or it is not. If `skip` could increment the check
    counter, a gate could pad its own pin by not running the checks, which is the
    failure this whole file exists to catch.
    """
    try:
        tree = ast.parse(read(path))
    except SyntaxError:
        return True  # unreadable: assume the worst rather than report a clean bill
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "skip"),
        None,
    )
    if fn is None:
        return False
    for n in ast.walk(fn):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) \
                and n.target.id == counter:
            return True
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == counter:
                    return True
    return False


def _summary_prints_the_skip_count(path: Path) -> bool:
    """Does a module-level `print` actually render the number of skips?

    Measured from the syntax tree on purpose. The weaker question -- "does the
    word 'skipped' appear anywhere in this file" -- is answered yes by a comment,
    and a mutation that folded the skip count into the pass line survived it. A
    `print` whose source mentions `len(SKIPPED)` is the thing that has to exist,
    because that is the line a reader uses to tell "passed" from "did not run".
    """
    try:
        tree = ast.parse(read(path))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "print":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                    and sub.func.id == "len" \
                    and sub.args and isinstance(sub.args[0], ast.Name) \
                    and sub.args[0].id == "SKIPPED":
                return True
    return False


_found = sorted(n for n in INVENTORY if (SCRIPTS / n).exists() and _defines_skip(SCRIPTS / n))
check(
    _found == sorted(SKIP_CONVENTION),
    "every gate that defines skip() is classified by what its skip does to the tally",
    f"defines skip(): {_found}. Classified: {sorted(SKIP_CONVENTION)}. Adding or "
    f"removing a skip() in any gate lands here, which is the point: a gate that "
    f"gained a skip has changed what its tally means",
)

# The one convention that is a defect rather than a choice. A skip the summary
# cannot account for is the invisible case: the run prints the same "N passed"
# whether or not the check happened. Ratcheted, for the usual reason -- the fix is
# three lines in a file this one does not own, so the debt is recorded and printed
# rather than asserted into a red its owner has to clear first.
#
#: **LIMITATION, and it is not fixable in this file: this ratchet fails only when
#: the set GROWS.** Deleting an entry -- which is exactly what a well-meaning
#: person does once they have fixed the file -- makes this check pass, and there
#: is no staleness test here to notice. The other two ratchets in this file both
#: have one (`UNJUSTIFIED_PINS` is cross-checked against `PROVENANCE_RE`),
#: `UNJUSTIFIED_PINS` against `PROVENANCE_RE`), and it is tempting to write a
#: third. It cannot be done, and the reason is worth more than the check would
#: have been: whether a skip is counted is not a property of the source that a
#: parser can be asked for. It is a property of the *relationship* between the
#: skip and the total, and the ways to express that relationship -- an append in
#: the body, a delegate to a `record` helper, a wrapper that converts a check
#: into a skip, a counter summed into the headline -- are not distinguishable by
#: looking for one token. An earlier version of this section tried, and had to be
#: thrown away: it could not see `viewport_framing_check.skip` (which records by
#: calling `record`) and it read a `len(results)` in a *comment* as evidence that
#: a skip counted toward the total. A staleness test built on that would have
#: reported the debt as fixed and as unpaid in the same breath.
#:
#: So the honest state is this: the ratchet catches a file *gaining* an
#: uncounted skip, and it cannot catch a file quietly leaving the list. If
#: `UNNAMED_SKIP_TALLIES` is ever empty, the correct reading is "nobody has
#: looked", not "nobody owes anything". The label in `SKIP_CONVENTION` is the
#: record of the human reading, and the way to keep that honest is to re-read it
#: when a file's summary changes -- not to trust this dict.
_now_unnamed = sorted(
    n for n in _found if n in UNNAMED_SKIP_TALLIES
)
check(
    len(_now_unnamed) <= MAX_UNNAMED_SKIP_TALLIES,
    "no gate has a skip() its summary cannot account for",
    f"{len(_now_unnamed)} in UNNAMED_SKIP_TALLIES against a cap of "
    f"{MAX_UNNAMED_SKIP_TALLIES}: {_now_unnamed}. Same ratchet as the others -- "
    f"give the summary a skip count, drop the entry, lower the cap",
)

check(
    set(UNNAMED_SKIP_TALLIES) <= set(_found),
    "every recorded unnamed-skip tally is still a gate that defines skip()",
    f"{sorted(UNNAMED_SKIP_TALLIES)}",
)

# ==========================================================================
section("the probe-once wrapper, described so another file can copy it")

#: THE PATTERN. `workbench_interaction_check.py` has 134 guarded call sites and
#: gets the right answer anyway, which is the whole point of writing it down.
#: Its shape, in four parts:
#:
#:   1. Decide the environment question **once**, at the point where the answer
#:      is first available, and store it in a module-level flag:
#:
#:          PIXELS_OK = probe_framebuffer(win)   # or whatever the question is
#:
#:      The flag is assigned in one place. That is what makes it a fact about the
#:      run rather than a condition re-evaluated at 134 different moments, and it
#:      is why the answer cannot differ between two checks in the same run.
#:
#:   2. Route every affected check through a wrapper that consults the flag and
#:      otherwise delegates unchanged:
#:
#:          def pixel_check(name, ok, detail=""):
#:              if not PIXELS_OK:
#:                  return skip(name, "the framebuffer read back blank in this environment")
#:              return check(name, ok, detail)
#:
#:      Note the shape: the wrapper takes the *same* arguments as `check`, so
#:      converting a call site is a one-word edit and the verdict is still
#:      computed -- it is just not *reported* when the environment cannot support
#:      it. The alternative, guarding each call site with its own `if`, is what
#:      produces 134 separate decisions that can disagree with each other.
#:
#:   3. Make `skip` a result, not an absence. The wrapper's `return skip(...)` is
#:      why the file reaches its pinned total on a machine with no OpenGL: the
#:      skip is recorded and counted, so the sum is unchanged and only the
#:      composition of the number moves. This is the `counts` convention, and
#:      `structure_bond_check.py` now does the same with one flag it does not yet
#:      need.
#:
#:   4. Escalate the *all-or-nothing* case, and only that case. A run with 3 skips
#:      and 40 passes is a partial result and should report a verdict; a run with
#:      0 passes measured nothing and must not be able to report success. The
#:      rule `if npass == 0: exit 2` captures exactly that, and it is the form to
#:      copy: **exit 2 means "this run measured nothing", not "this run skipped
#:      something".** A file that escalates on skips alone will go red on every
#:      headless machine and be switched off within a week.
#:
#: What to copy this for: `cli_check.py`'s 32 `try:else` sites, `pockets_check.py`'s
#: `if cavern` / `if cavities`, `prep_check.py`'s `if made is None`. What not to
#: do with it: apply it to a `for` loop over test cases. A loop over a fixed set
#: of inputs is not an environment question, and turning its iterations into skips
#: would make a data set look like a broken machine.

print()
print("  skip() -- what each gate's skip does to its tally:")
for _n, _tag in sorted(SKIP_CONVENTION.items()):
    print("    %-34s %s" % (_n, _tag))
print()

# This file has a skip() of its own and has to survive the audit it runs on the
# others. The measurable half is asserted; the rest is stated as a limitation.
check(
    _defines_skip(TARGET),
    "check_scripts_declare.py defines skip(), so the convention it audits is one "
    "it also obeys",
    f"this file recorded {len(SKIPPED)} skip(s) on this run",
)
check(
    not _skip_touches_the_counter(TARGET, "CHECKS"),
    "check_scripts_declare.py's skip cannot pad the pin by not running a check",
    f"`skip` does not write to CHECKS. A skip that incremented it would let a gate "
    f"reach its own total by declining to check anything, which is the failure "
    f"this file exists to catch and this line would be committing it",
)
check(
    _summary_prints_the_skip_count(TARGET),
    "check_scripts_declare.py's summary prints its skip count, not just the word",
    f"a print at module level renders len(SKIPPED): "
    f"{_summary_prints_the_skip_count(TARGET)}. An earlier version of this check "
    f"looked for the substring 'skipped' anywhere in the file, which passed on "
    f"the word appearing in a comment -- so folding the skips into the pass count "
    f"was a mutation that survived. The count has to be *rendered*, and that is "
    f"checkable in the syntax tree",
)


# ==========================================================================
section("a root file is accounted for by a producer, not by a gate's silence")

#: `check_text_encoding.py` asks of every root file whether it is "declared
#: source, named by a gate, or plainly unclaimed", and it calls the third
#: bucket "debris by elimination". The elimination is the part that cannot
#: carry that weight. The test it applies is *no gate mentions this name*;
#: the thing the bucket has to establish is *nothing writes this file*. Those
#: are different questions, and the first one passing is not evidence about
#: the second. That is the shape of the defect this repository keeps meeting,
#: in the one place where it becomes a file on disk that nobody can blame for
#: not having been deleted.
#:
#: So this section asks the second question. Its first half is positive and
#: checkable, which is what makes the rest arithmetic rather than inference:
#: the rule ships a file iff it sits under a `SOURCE_ROOT` **or** its whole
#: name at the tree root is in `ROOT_SOURCE_NAMES`
#: (`release_tree_rule.iter_source_files`). The workspace root is not a
#: `SOURCE_ROOT` on this tree, so **no file at the workspace root is produced
#: by the wheel, the sdist, or `_sync_release.py` at all** -- and that is
#: measured off the tables, not inferred from anybody not speaking up.
#: `_sync_release.py` is the same route and not a second one: it binds
#: `iter_source_files` at line 68 and copies exactly what it returns.
RELEASE_RULE = "release_tree_rule.py"


def _load_release_rule() -> dict:
    """The release rule's two positive tables, read from the file that owns them."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "_release_tree_rule_under_audit", SCRIPTS / RELEASE_RULE
    )
    if spec is None or spec.loader is None:
        return {}
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "SOURCE_ROOTS": tuple(getattr(mod, "SOURCE_ROOTS", ())),
        "ROOT_SOURCE_NAMES": frozenset(getattr(mod, "ROOT_SOURCE_NAMES", ())),
    }


def _stated_root_names(src: str) -> set[str]:
    """Root file names this source *states*, measured rather than searched.

    A bare string is a stated root name only where somebody had to mean one:

      * an element of a tuple / list / set of names,
      * the argument of a `Path(...)` call,
      * a key of a dict.

    The two shapes that manufacture a false credit are both outside that list,
    and both are on this tree today:

      * **a path component of a `/` composition.** The expression
        `ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "app.py"`
        contains the literal `"app.py"`, and the file at the workspace root is
        not the file that expression names. A scan for literals cannot tell
        them apart, because a path built with `Path.__truediv__` is one string
        literal *per component* -- which is why a reader that takes "some gate
        says `app.py`" as "some gate opens the root `app.py`" credits six root
        files on this tree that no code path opens.
      * **a bare argument to something else.** `str(watched).endswith(
        "pockets.py")` is a claim about a string, and it is true precisely when
        a gate is running from the workspace root and the file it means is a
        copy under `dock-py/python/`.

    This walk is a *positive* rule -- three named shapes, enumerated -- so what
    it refuses is explicit rather than inferred, and the two controls below
    assert both directions on synthetic source holding all five shapes at once.
    A reader whose rule is "everything else" cannot be tested that way.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return set()

    def plain(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "/" not in node.value and "\\" not in node.value \
                and " " not in node.value:
            return node.value
        return None

    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for el in node.elts:
                name = plain(el)
                if name:
                    out.add(name)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                name = plain(key) if key is not None else None
                if name:
                    out.add(name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "Path" and node.args:
            name = plain(node.args[0])
            if name:
                out.add(name)
    return out


_RULE_TABLES = _load_release_rule()
_source_roots = _RULE_TABLES.get("SOURCE_ROOTS", ())
_root_source_names = _RULE_TABLES.get("ROOT_SOURCE_NAMES", frozenset())

_root_files = sorted(p.name for p in ROOT.iterdir() if p.is_file())
_stated: dict[str, list[str]] = {}
#: This file is excluded from the scan, and the reason is not politeness. The
#: reader's own controls below are written with two *real* root file names in
#: them -- `app.py` and `pockets.py` are the two shapes a control has to
#: refuse, and a control written with invented names would not be testing the
#: case -- and a `{"app.py", "pockets.py"}` set literal in this file is
#: indistinguishable, to this file's own walk, from a gate stating that
#: those two root files are accounted for. The first version of this section
#: did exactly that and credited both of them from its own test fixture. An
#: auditor that can vote in the census it is auditing measures its fixture.
#: Nothing is lost by the exclusion: every name this file states at the root is
#: stated by a gate as well, which the printed table below shows.
_scan = [p for p in sorted(SCRIPTS.glob("*.py")) if p.resolve() != TARGET.resolve()]
for _p in _scan:
    for _name in _stated_root_names(read(_p)):
        _stated.setdefault(_name, []).append(_p.name)

def _unaccounted_root(root_files, declared, stated) -> list[str]:
    """The root files neither table accounts for. The whole rule, in one line.

    Kept separate from the reading of the two tables so it can be controlled on
    a file set that is not the tree's. A rule that can only be exercised by
    editing the repository is a rule whose green has never been observed.
    """
    return [n for n in root_files if n not in declared and n not in stated]


_root_unaccounted = _unaccounted_root(_root_files, _root_source_names, _stated)

# The census is only a measurement if the thing it reads was read. Without this
# an unreadable rule yields empty tables, every root file lands in
# `_root_unaccounted`, and the check below goes red *for the right reason by
# accident* -- which is the least useful red there is. So the load is its own
# result, and the detail of the red names which of the two happened.
check(
    bool(_source_roots) and bool(_root_source_names),
    "the release rule this section reads was readable, so the census is not "
    "measuring an empty table",
    f"loaded {len(_source_roots)} SOURCE_ROOT(s) and {len(_root_source_names)} "
    f"ROOT_SOURCE_NAME(s) from {SCRIPTS / RELEASE_RULE}. If this is red, the "
    f"accounted/unaccounted split below is an artefact of a failed import and "
    f"not a finding about the tree",
)

# The one positive fact the rest of the section is arithmetic on. If the
# workspace root ever *becomes* a SOURCE_ROOT this goes red, and the red means
# the argument changed rather than that a file appeared: every claim the census
# below makes about a root file not being shipped would become false.
check(
    "." not in _source_roots and "" not in _source_roots,
    "the workspace root is not a SOURCE_ROOT, so the release build and the "
    "commit mirror cannot produce a root file",
    f"SOURCE_ROOTS = {list(_source_roots)}. `iter_source_files` walks these, "
    f"and the workspace root is not one of them, so a root file reaches the "
    f"wheel and the commit mirror only if its whole name is in "
    f"ROOT_SOURCE_NAMES. `_sync_release.py` binds the same function at its "
    f"line 68, so it is the same route rather than a second one",
)

print()
print("  root-level files -- what accounts for each, and by what:")
print()
for _n in _root_files:
    if _n in _root_source_names:
        _why = f"declared in ROOT_SOURCE_NAMES ({RELEASE_RULE})"
    elif _n in _stated:
        _why = "stated at the root by " + ", ".join(sorted(set(_stated[_n]))[:3])
    else:
        _why = "** nothing found **"
    print("    %-26s %s" % (_n, _why))
print()

# The red this section exists for. It is a red about the tree, not a ratchet,
# and it is deliberately not made to pass by a table: the three buckets above
# are derived, and the only way to move a file out of the third is to change
# what is on disk or to add the file to `ROOT_SOURCE_NAMES`, which is a
# decision somebody has to make and defend. A hand-maintained owner list here
# would be a second thing to forget, and its whole purpose is to stop people
# being told that a silence is a measurement.
check(
    not _root_unaccounted,
    "every root-level file is accounted for by something a run can point at",
    f"{len(_root_unaccounted)} of {len(_root_files)} root files are neither a "
    f"declared ROOT_SOURCE_NAME nor stated at the root by any of the "
    f"{len(_scan)} files in scripts/ (this file excluded -- its controls name "
    f"two real root files and would vote in their own favour): "
    f"{_root_unaccounted}. The release build and the commit mirror cannot "
    f"produce them -- the check above measures that -- and no gate names them, "
    f"so nothing here knows they exist. The bucket `check_text_encoding.py` "
    f"calls \"debris by elimination\" is the same {len(_root_unaccounted)} "
    f"files reached by a different route and a different number, and this one "
    f"is the larger of the two because it does not credit a path component as "
    f"a reference. Delete them, or give each one a declared name; do not add a "
    f"list of owners here"
    if _root_unaccounted else
    f"all {len(_root_files)} root files are accounted for: "
    f"{len(_root_source_names)} by ROOT_SOURCE_NAMES and "
    f"{len(_root_files) - len(_root_source_names)} stated at the root by a "
    f"gate. Every one of them has a named route, and the routes are read off "
    f"{RELEASE_RULE} and the scripts rather than kept in this file",
)

# --------------------------------------------------------------------------
# The reader, proved in both directions on source that cannot be edited away.
# --------------------------------------------------------------------------
#: Five shapes, one string each. The first three are the positive rule; the
#: last two are the two false credits found on this tree. A loose literal scan
#: returns all five, and this control is what would notice.
_READER_FIXTURE = '''
_A = ("README.md", "README.en.md")
_B = {"odck.md": "excluded"}
_C = Path("SECURITY.md")
_D = ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "app.py"
_E = str(watched).endswith("pockets.py")
'''
_READER_WANTS = {"README.md", "README.en.md", "odck.md", "SECURITY.md"}
_READER_REFUSES = {"app.py", "pockets.py"}

_reader_got = _stated_root_names(_READER_FIXTURE)
check(
    _reader_got == _READER_WANTS and not (_reader_got & _READER_REFUSES),
    "the root-file reader can tell a root file from a path component that "
    "shares its name",
    f"on a fixture holding all five shapes it returned {sorted(_reader_got)}; "
    f"it must return exactly {sorted(_READER_WANTS)} and none of "
    f"{sorted(_READER_REFUSES)}. The three shapes it accepts are a tuple of "
    f"names, a dict key and a Path() argument. The two it must refuse are the "
    f"`/` composition in `_D` -- a literal per component, so the root "
    f"`app.py` is not the file that expression names -- and the bare argument "
    f"in `_E`, which is a claim about a string. Both of those are real root "
    f"file names, which is why the census above excludes this file: a control "
    f"written with invented names would not be testing this case, and a "
    f"control written with real ones is a set literal this walk reads as a "
    f"statement. A reader that credited either is the reader that puts root "
    f"files in the accounted bucket on the strength of a code path that never "
    f"opens them",
)

# The other direction, because a reader that credits nothing is not strict, it
# is broken. Drop one name out of a two-name tuple and that one name has to go
# while its sibling stays -- which also pins the credit to the *reference*
# rather than to the file happening to exist at the root. Both names are on
# this list, so neither can be the one that survives by accident.
_reader_without = _stated_root_names(_READER_FIXTURE.replace(
    '_A = ("README.md", "README.en.md")', '_A = ("README.en.md",)'))
check(
    "README.md" not in _reader_without
    and "README.en.md" in _reader_without
    and _reader_without == _READER_WANTS - {"README.md"},
    "the root-file reader's credit is the reference, not the file existing",
    f"dropping README.md out of the two-name tuple dropped that credit and kept "
    f"its sibling: {sorted(_reader_without)}. Both names are declared source at "
    f"the root, so a reader that answered \"is there a README.md\" instead of "
    f"\"does something state README.md\" would have kept it -- and that is the "
    f"census this section is not allowed to be",
)


# The red above is red on the tree, so it has never been observed reporting a
# clean root -- and a rule whose only observed colour is red is a rule nobody
# can tell apart from a constant. These are the three cases in one fixture, and
# they are the whole classifier: a file with no account for it must come back,
# a file with either account must not, and a file set where every name has one
# must come back empty. The third is the direction the real tree cannot supply
# today, and it is the one that would matter the day somebody clears the red.
_CONTROL_FILES = ["LICENSE", "README.md", "stray.py"]
_CONTROL_DECLARED = {"LICENSE"}
_CONTROL_STATED = {"README.md": ["some_gate.py"]}
_control_got = _unaccounted_root(
    _CONTROL_FILES, _CONTROL_DECLARED, _CONTROL_STATED)
_control_all_accounted = _unaccounted_root(
    ["LICENSE", "README.md"], _CONTROL_DECLARED, _CONTROL_STATED)
check(
    _control_got == ["stray.py"] and _control_all_accounted == [],
    "the root-file rule can report a clean root, so its red is a finding and "
    "not a constant",
    f"on a three-name fixture it returned {_control_got} -- the one name with "
    f"neither a declared name nor a stating gate -- and on the same fixture "
    f"with `stray.py` removed it returned {_control_all_accounted}. The rule is "
    f"\"neither table accounts for it\", so both accounts have to clear a name "
    f"and one of them clearing is enough; a rule that returned everything, or "
    f"nothing, would pass the red above for the wrong reason. The tree cannot "
    f"supply this direction today, which is exactly why it is controlled rather "
    f"than assumed",
)


# ==========================================================================
section("this file holds itself to the same contract")

_self = pin_of(TARGET)
check(
    _self is not None
    and _self[0] == EXPECTED_CHECKS
    and bool(PROVENANCE_RE.search(_self[1])),
    "check_scripts_declare.py pins itself, justifies the number, and the pin "
    "matches what it actually runs",
    f"the constant says {EXPECTED_CHECKS}; the module-level pin says "
    f"{_self and _self[0]!r} with comment {_self and _self[1]!r}",
)

# The pin is compared above, but prove it with a number rather than a regex.
_before = CHECKS
check(
    _before + 1 == EXPECTED_CHECKS,
    "the number of checks that ran is the number this file is supposed to have",
    f"{_before} ran before this one and {EXPECTED_CHECKS} are expected; the +1 "
    f"is this check. Change EXPECTED_CHECKS deliberately",
)

print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
for f in FAILURES:
    print(f"  FAILED: {f}")

# The skip line is not decoration. It is the only place in this file's output
# where "passed" and "did not run" are stated as two different numbers, and it is
# printed unconditionally so that a run which skipped everything still says so.
if SKIPPED:
    print(f"\n  {len(SKIPPED)} check(s) were skipped, not passed:")
    for _n, _r in SKIPPED:
        print(f"    SKIP {_n}: {_r}")
    print("  A skip means this run could not answer the question. It is not "
          "evidence that the thing it guards is correct.")
else:
    print("\n  0 skipped: every check this file ran, it ran here.")

sys.exit(1 if FAILURES else 0)
