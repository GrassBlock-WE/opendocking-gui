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

**Why this cannot drift.** A script that gains a check changes what
`_sites_of` derives and not what it declares, so the run is red until the
declaration moves -- and the declaration is now a comment in the same file, one
screen from the code that has to change. A declaration edited to a wrong number
is red for the same reason, and cannot be satisfied by typing the number the
auditor wants to see, because the auditor computes it. There is no longer any
place a human writes down what a script should contain.

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
import hashlib
import io
import re
import sys
import time
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
    "check_doc_encoding.py",
    "check_repo_docs.py",
    "check_scripts_declare.py",
    "cli_check.py",
    "cli_prep_check.py",
    "contacts_check.py",
    "contacts_check2.py",
    "core_check.py",
    "energy_terms_check.py",
    "examples_check.py",
    "framing_selection_check.py",
    "ligand_check.py",
    "pdbqt_check.py",
    "pockets_check.py",
    "prep_check.py",
    "release_tree_parity_check.py",
    "representation_cartoon_check.py",
    "representation_geometry_check.py",
    "representation_names_check.py",
    "scoring_cross_check.py",
    "screenshot_frame_check.py",
    "docs_claims_check.py",
    "structure_bond_check.py",
    "torsion_bond_record_check.py",
    "viewport_framing_check.py",
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
    invisible to the old census. `check_doc_encoding.py` and `check_repo_docs.py`
    define no `check()` and declare no `EXPECTED_CHECKS`: they count what they
    found and hand the count to `sys.exit` as a non-literal argument --
    `1 if total else 0` and `main()` respectively. That *is* their declaration,
    and the `IfExp`/`Call` distinction is the whole test: a literal
    `sys.exit(0)` is a script that has already decided, and asking it for a pin
    would be asking about a number it never had.

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
#: sites: 21 unconditional + 6 guarded
#: guards: sha256:16216eb9568a0e78526ed7fac68b16e8722108dfbe64ffac92c8083bdb7530d1
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
#: The first two are the uncomfortable half and they are worth naming: **this
#: file's own total is a function of how many gates in the tree carry a pin**,
#: because the per-pin checks are a loop. So the number is not a constant of
#: this file the way `examples_check.py`'s 29 is -- it moves when another owner
#: lands a pin, and the next person to do that moves it again. That is the
#: mechanism working as designed (a gate that gains a pin is *measured* from
#: that moment, which is the point) but it makes this pin a shared number
#: rather than a private one, and the honest thing is to say so here rather
#: than let the next red look like somebody's arithmetic.
EXPECTED_CHECKS = 73

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
MAX_UNPINNED_SCRIPTS = 12

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
    global CHECKS
    CHECKS += 1
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
    compared = [
        line for line in read(SCRIPTS / name).splitlines()
        if "EXPECTED_CHECKS" in line
        and re.search(r"==|!=|>=|<=|>|<", line)
        and not re.match(r"^\s*EXPECTED_CHECKS\s*=", line)
    ]
    check(
        bool(compared),
        f"{name}'s pin is actually compared, not just declared",
        f"{len(compared)} comparison site(s); the first: "
        f"{compared[0].strip()[:70] if compared else 'NONE -- the pin is read by nobody'}",
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
#: --- BEGIN GENERATED SNAPSHOT (python scripts/check_scripts_declare.py --pin) ---
DECLARED_SNAPSHOT = {
    "benchmark_check.py": (33, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "check_doc_encoding.py": (0, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "check_repo_docs.py": (0, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "cli_check.py": (81, 37,
        "17c289b915fee673719e48789e700a7d347d151a0f068265f38ef32b0f2060e6"),
    "cli_prep_check.py": (31, 2,
        "2013eb0ff8a729058c868eee6568b0898977584da1f8a4c8c79d917e1bc3b184"),
    "contacts_check.py": (18, 22,
        "a548473d3772baf82275d8110f6df5dca83fd4251e3f421d66fa73ed788b3735"),
    "contacts_check2.py": (14, 2,
        "14b3f81f282ef9120b0c2c9405f6060bf764650a4cac8c35e4aa66274992a5ee"),
    "core_check.py": (276, 6,
        "896968e9c518428ec441282d479256cb20805d6eec78da9af3b4695f717fcc7d"),
    "ligand_check.py": (37, 19,
        "5fb40532e4e326a3d1ec1b71d51779e792a1616e4fbd8f2cdff78c1386a912dd"),
    "pdbqt_check.py": (49, 4,
        "b27c4f071eca1a4757f0ae454557123488ea6da0467a8bc10a5ed69076382c87"),
    "pockets_check.py": (65, 84,
        "4bf203db36996b5950ef84c3e42982aa5379a9351a16fab1a2245fc9ec5f3ab8"),
    "prep_check.py": (62, 27,
        "757db9bd10cd656af4d902c5a55658fb89fc5894077419cbe32c39b69f9dad5a"),
    "representation_geometry_check.py": (46, 5,
        "1138babf87c705ad487e8c9d894dbf2ddaea822a32a5361b6c6c92a9d230aa52"),
    "scoring_cross_check.py": (27, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "torsion_bond_record_check.py": (12, 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "workbench_interaction_check.py": (85, 160,
        "f81a42ae8442a3c73499ade27546196f181adefb00e9bfa04cfe6dd7e7ceccab"),
    "x11_window_parse_check.py": (9, 2,
        "cef6d60f6d78425fd0fc4cf3491b54593342f03020649a99f262004e1e071280"),
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
#: 23 gates; four of them declare their own block, so 19 are on the snapshot at
#: the time of writing, and all but those four belong to other owners. Lower it as
#: owners paste their own block in; the slack cannot be spent on a new gate,
#: because a new gate is measured either way.
#:
#: `examples_check.py` moved off this snapshot to carry a block of its own. The
#: prose above called a gate appearing in both places a failure, and then did
#: not check for it: `_declared_for` prefers the block, so a stale snapshot entry
#: was invisible to every count here and `--pin` would have removed it silently
#: the next time anyone ran it. It is checked now, which is the same lesson as
#: the `SITE_INVENTORY` hole recorded below: a rule stated in a comment and
#: absent from the code is a comment.
MAX_SNAPSHOT_ONLY = 19

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


PIN_BEGIN = "#: --- BEGIN GENERATED SNAPSHOT"
PIN_END = "#: --- END GENERATED SNAPSHOT ---"


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
    rows = {}
    refused = []
    for name in INVENTORY:
        path = SCRIPTS / name
        if not path.exists():
            refused.append(f"{name}: not on disk")
            continue
        if _declaration_of(path) is not None:
            continue  # declares itself; the snapshot has no business here
        try:
            uncond, guards = _sites_of(read(path))
        except SyntaxError as exc:
            refused.append(f"{name}: does not parse ({exc})")
            continue
        rows[name] = (uncond, len(guards), _guards_digest(guards))
    if refused:
        print("refusing to pin; these gates could not be measured:", file=sys.stderr)
        for r in refused:
            print(f"  {r}", file=sys.stderr)
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
    start = src.find(PIN_BEGIN)
    end = src.find(PIN_END)
    if start == -1 or end == -1:
        print(f"could not find the snapshot markers in {TARGET}", file=sys.stderr)
        return 2
    end += len(PIN_END)
    out = src[:start] + new_region + src[end:]
    # newline="" so the file's own endings survive; utf-8 explicitly, because
    # PowerShell 5.1's -Encoding UTF8 writes a BOM and would corrupt the em
    # dashes this file is full of.
    with io.open(TARGET, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"pinned {len(rows)} gate(s) into the generated snapshot:")
    for name in sorted(rows):
        u, g, d = rows[name]
        print(f"  {name:<38} {u:>4} unconditional {g:>3} guarded  sha256:{d[:12]}")
    skipped = [n for n in INVENTORY
               if (SCRIPTS / n).exists() and _declaration_of(SCRIPTS / n) is not None]
    print(f"left alone (declares its own block): {sorted(skipped)}")
    print("Now read the diff. A census you did not expect to move is the thing "
          "this command exists to make visible.")
    return 0


if "--pin" in sys.argv:
    sys.exit(_pin())


_drifted = []
_unparseable = []
_measured = 0
_census: dict[str, tuple[int, int, str]] = {}
_in_block: list[str] = []
_snapshot_only: list[str] = []
_flying: list[str] = []
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
        _drifted.append(f"{name}: {where}")
        print(f"[FAIL] {name}: {where}")
        print(f"       derived {got_uncond} unconditional / {len(got_guards)} "
              f"guarded, guards sha256:{_guards_digest(got_guards)[:12]}...")
        continue
    if (got_uncond, len(got_guards)) != (d_uncond, d_guarded):
        _drifted.append(
            f"{name}: derived {got_uncond}+{len(got_guards)}, declared "
            f"{d_uncond}+{d_guarded}"
        )
    elif _guards_digest(got_guards) != d_digest:
        _drifted.append(
            f"{name}: same census, different guards "
            f"(declared sha256:{d_digest[:12]}..., derived "
            f"sha256:{_guards_digest(got_guards)[:12]}...)"
        )


check(
    not _drifted and not _unparseable,
    "every gate declares the call sites it actually has",
    f"{_measured} of {len(INVENTORY)} gates measured, "
    f"{sum(u + g for u, g, _d in _census.values())} call sites across them; "
    f"drifted: {_drifted}. A site added, deleted, or wrapped in an `if` moves the "
    f"derived census and not the declared one. The same census with different "
    f"guards is caught too, by the digest -- that is the change the two counts "
    f"alone cannot see",
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
    "check_scripts_declare.py":
        "records the skip in its own list, keeps it out of CHECKS, and prints the "
        "count beside the pass count",
    "structure_bond_check.py":
        "records the skip in its own list, counts it into the pinned total, and "
        "prints pass/fail/skip as three separate numbers that never sum to a "
        "verdict; the total is asserted to partition into the three",
    "viewport_framing_check.py":
        "records the skip as a third tag, counts it on a separate counter, and "
        "turns zero passes into 'did not finish' (exit 2)",
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
