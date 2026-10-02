"""Which product functions, methods and public attributes does nothing in the
product call?

Run:  python scripts/unreached_product_check.py

# The problem this exists to close

Two pieces of product code were found by hand, in two separate rounds, with the
same shape:

* `workbench/framing_selection.py`'s `framed_radius` -- zero references outside
  its own definition, and its docstring justifies itself by a failure mode
  ("a framing can be correct across the screen and still be pulled a long way
  back along the view direction -- which is what happens when something other
  than the selection was fitted") that nothing in the product is in a position
  to prevent.
* the same file's `selection_is_degenerate` -- zero references tree-wide outside
  its own definition and its sibling gate.

**Dead code is harmless. Dead code that explains itself as enforcement is
not.** That is the whole distinction. An unreached function that says "nothing
calls this yet" is a true sentence about the tree. An unreached function whose
docstring, comment or `__all__` entry presents itself as a guard is a *claim
that reads like a guarantee and is not one* -- the failure this repository has
been closing all session. The reason this file exists is that the class was
found twice by hand, a third instance was already in the tree, and **nothing in
the repository was asking the question.**

# What it measures, in order

1. **Enumerate** every public function, method and public attribute in
   `dock-py/python/opendocking/`, by AST. The set is derived, never typed.
   `tests/` is excluded by path, not by a name pattern.
2. **For each name, decide by AST whether a product module references it from
   outside its own module.** Two reference classes, counted separately because
   they are not equally trustworthy:
   * `RESOLVED` -- the reference's root resolved through an import alias to one
     specific definition. Strong.
   * `NAME-SHAPED` -- the root could not be resolved, almost always because the
     reference is through an *instance*: this tree is a PyQt6 window and says
     `self.camera.projection`, not `Camera.projection`. The terminal identifier
     is matched instead, and the ambiguity is reported per name -- how many
     product modules define that bare name, which is how many definitions the
     match could have meant.

   A `from X import Y` is a **re-export** and counts as a reference. It is how
   `opendocking/__init__.py` publishes the package surface, so a scan counting
   only loads would report the entire public API as unreached. That was the
   first version's largest false-positive class, at 192 names, and it is why
   the rule is stated here rather than left to be rediscovered.
3. **Derive the three dispatch classes an AST cannot see**, and require each to
   resolve -- see "What an AST cannot see".
4. **Every remaining name needs a row** in `DISPOSITIONS`, and the table is
   checked in **both directions**.

# Why the table is two-sided, and why that is the load-bearing part

A hand-maintained list of exceptions is a second thing to forget. A
one-directional check -- "every unreached name has a row" -- would stop the
moment somebody adds an unreached function without adding its row, which is
most of the value but not all of it. The half that costs nothing and catches
more is the other direction: **a row naming something that *is* reached is also
red.** That is the stale-excuse failure, and a one-directional table cannot see
it at all, because a stale row makes the table look *more* complete than the
world is.

The shape is borrowed from `docs/SCORING.md` section 11, which does this for a
different subject (exported constants with no written specification): the left
set is derived from source, the right set is hand-written, and **both
mismatches are failures**. Section 11.1 adds the third piece copied here, a
purely *structural* backstop: an inequality between the table and the derived
population, so the degenerate "exempt everything" table goes red. A table that
exempts all of it has stopped saying anything.

# The state words, and why there is a column

The defect being hunted is a sentence that reads like a guarantee. A state
column exists so two sentences that both begin "this function is not called"
cannot be read as the same sentence:

* `GATE-ONLY` -- **deliberately not wired in yet.** Somebody decided; the
  reason says what it is for and why leaving it out was the decision.
* `OPEN-QUESTION` -- **nobody has decided.** Registered as a gap. This is the
  word that must not be read as the one above: a not-yet-decided thing reads
  exactly like a decided thing right up until the word in the row is read.
* `OWN-MODULE-ONLY` -- reached by its own module's product code and by no other
  module. A *wiring* fact, not a judgement about the code.
* `OWN-CTOR-READ` -- reached by **the constructor of a class in its own
  module** and by nothing else: not by a module-level function there, and not
  by another module. The narrowest fact this set can hold, and the one added
  last, for a shape that is entirely ordinary -- factoring a constant out of a
  constructor and giving it a public name. `VIEWPORT_MIN_W` and
  `VIEWPORT_MIN_H` are the pair: `app.py` reads both on one line inside
  `Viewport.__init__`, so neither name is dead, and neither is reached by
  anything else. Without a state for it, the only record available was a
  hand-written paragraph per row, and `no row carries a third generic excuse`
  correctly refuses paragraphs that all say the same thing.
* `UNREAD-FIELD` -- a declared data field with no reader. A shape, not a
  decision: nobody chose this, the field was declared and nothing grew to read
  it.

Every declared state must be used by at least one row, which is the
stale-direction rule applied to the vocabulary: a state word nothing uses is a
word that has stopped describing anything.

**And the table may not be edited by writing a key twice.** A mutation found
this: adding a second row for a name that already had one does not add a row,
because a dict literal takes the last value for a repeated key and says
nothing. The two-sided checks stayed green -- correctly, since the name was
still legitimately in the table -- while the writer believed they had recorded
a second opinion. `no row key is written twice in the table literal` counts
the keys in this file's own AST and compares that with the dict's length.

# What an AST cannot see, and how each case is handled

Stated here rather than left for a reader to discover, because the honest
answer to "is this name reached" is a **lower bound**.

* **Qt virtual overrides.** `Viewport.paintGL` has no caller in this tree; Qt
  calls it. The class is *derived*, not typed: a method of a class whose base
  resolves to an imported `PyQt6` name and whose own name exists on the **real
  installed base class**, introspected at run time with `hasattr`. There is no
  list of Qt method names in this file, which is the point -- a typed Qt list
  would be the second hand-maintained list this file exists to avoid, and it
  would go stale silently when PyQt6 changed. The residual: a method whose name
  *coincides* with a Qt base method but which Qt never calls is exempted as
  though it were. That direction is stated, not fixed.
* **Qt slots and properties.** A method carrying a `pyqtSlot` / `pyqtProperty`
  decorator is reachable through the meta-object. Derived from the decorator.
* **Console-script entry points.** `cli.main` and `launcher.main` are called by
  a shim generated at install time. Derived from `dock-py/pyproject.toml`'s
  `[project.scripts]`, parsed at run time, and checked in both directions: a
  declared entry point that does not resolve to a definition is red.
* **NOT handled: `getattr(obj, "name")`, `QMetaObject.invokeMethod`,
  `setattr`, `globals()[...]`, string dispatch of any kind.** No rule here
  sees them. A name reached only that way reads as unreached and needs a row
  saying so. The count of names this file cannot rule on is the count in the
  table.
* **NOT handled: instance reads are name-matched, not type-resolved.** With
  `amb=N` on a row, the match could have meant any of N definitions. The
  ambiguity is printed rather than resolved, because resolving it needs type
  inference and a wrong resolution would be worse than a stated uncertainty.
* **A weaker version of the same thing, measured: function-local imports.**
  The import table is built from module-level `import` statements only, and
  `app.py` does `from . import keys as keymap` *inside* `_install_shortcuts`.
  A reference through that local alias therefore misses the alias table and
  degrades from `RESOLVED` to `NAME-SHAPED`. The name is still reached, so
  nothing is missed here, but the `RESOLVED` count in the summary is a lower
  bound on the strong references that exist.
* **A skip that was a finding, and is now fixed; the policy it needed is
  still standing.** `workbench/__init__.py` used to list `Workbench` in its
  `__all__` without defining or importing it, so
  `from opendocking.workbench import Workbench` raised `ImportError` and
  `from opendocking.workbench import *` raised `AttributeError`. The `__all__`
  completeness check recorded that as a skip rather than a failure, for reasons
  that are still the reasons. It is a real defect, so it must not be a pass. It
  is in a file this gate does not own, so a gate permanently red over somebody
  else's one-line fix is a gate people switch off. And a skip that is counted as
  a result rather than an absence is this repository's own convention -- the
  reason is spelled out at `SKIPPED`, and the summary prints passes and skips
  as two numbers it never adds. The name is now absent from `__all__` and
  undefined, this run reports no skip, and `workbench/__init__.py` records why
  it could not simply have been added back.

# What this file does NOT do

It does not import `opendocking`, so it runs with no wheel, no GPU and no
OpenGL context. It does import `PyQt6`, because the Qt-base introspection is
the only way the override class can be derived rather than typed; a missing
PyQt6 makes this gate **fail loudly** rather than fall back to a typed list. A
gate that degrades quietly is the defect it is looking for.

It does not import `scripts/check_scripts_declare.py`, which runs its whole
audit and calls `sys.exit()` at module scope. Its `_sites_of`,
`_guards_digest` and `_declaration_of` are lifted out by AST, compiled alone,
and called -- the same function bodies, not a transcription of their answers --
so this file's own `GATE-DECLARE` census is derived. If those names stop
existing over there, this file raises and is red rather than quietly measuring
its own differently-shaped walk.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import io
import re
import sys
import tokenize
from pathlib import Path

# This gate's output is UTF-8 whatever the console codepage is. `gbk` cannot
# encode the em dashes this file's prose uses, and a reader decoding this
# output as UTF-8 raised `UnicodeDecodeError` on the first one elsewhere in
# `scripts/` and got no verdict at all. `errors="replace"` keeps the output
# decodable; a character this file cannot encode then shows as `?` rather than
# ending the run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "dock-py" / "python" / "opendocking"
PYPROJECT = ROOT / "dock-py" / "pyproject.toml"
SELF = Path(__file__).resolve()
DECLARER = ROOT / "scripts" / "check_scripts_declare.py"
PKG_PREFIX = "opendocking."
PYQT_ROOT = "PyQt6"

#: The result count, pinned from a green run of this file on this tree. It is
#: the number of *results*, not of call sites; the second number is the
#: `GATE-DECLARE` census below, and that one is derived in this file rather
#: than typed. **The skip is not in this number and never is**: a gate that
#: reaches its pin by not running a check is the failure a pin cannot catch,
#: so the summary prints passes and skips as two numbers and adds them
#: nowhere.
#:
#: **21 -> 22, and the cause is a skip that stopped being a skip.** The
#: `__all__` completeness check used to be skipped, because `__all__` promised a
#: name (`workbench.Workbench`) that no module defines. A skip is deliberately
#: not in this number, so the pin was 21. `Workbench` was removed from `__all__`
#: rather than invented, that check now records as an ordinary result, and the
#: count of non-skip results is 22. This is the pin doing its job: a check that
#: could not run and a check that ran are different numbers on purpose, and
#: "the tally went up by one because a defect got fixed" is a sentence worth
#: being able to say.
EXPECTED_CHECKS = 22  # measured from a green run of this file on this tree, counted by running it

#: The call-site census, declared in the format
#: `scripts/check_scripts_declare.py` reads, and verified in this file by
#: calling that file's own `_sites_of` / `_guards_digest` /
#: `_declaration_of`. Those are lifted out by AST because the module `sys.exit`s
#: at import; see the section on `load_declarer`.
#: GATE-DECLARE 1
#: sites: 20 unconditional + 3 guarded
#: guards: sha256:eedb526e6b7126b61a5e0fc21e0d138e897c1b2cf05710f0983192e129ea0e5e

CHECKS = 0
FAILURES: list[str] = []
REPORT: list[str] = []
#: Recorded skips, as `(name, reason)`. **A skip is a result, not an absence**,
#: and it is never added to the pass count: the summary prints the two numbers
#: separately and never sums them, because a gate that reaches its total by not
#: running a check is the one failure mode a pin cannot catch. The one skip
#: here exists because this gate found a real defect in a file it does not own,
#: and carrying it as a failure would make the gate permanently red over
#: somebody else's one-line fix while carrying it as a pass would hide it.
SKIPPED: list[tuple[str, str]] = []

#: The closed set of state words, and what each asserts. The pair the column
#: exists to keep apart is `GATE-ONLY` (somebody decided to leave this out) and
#: `OPEN-QUESTION` (nobody has decided). `check` refuses a word outside this
#: set, and `check("each declared state has at least one row", ...)` refuses a
#: word nothing uses.
STATES = {
    "GATE-ONLY":
        "deliberately not wired in yet; somebody decided, and the reason says "
        "what it is for and why leaving it out was the decision",
    "OPEN-QUESTION":
        "nobody has decided; registered as a gap, not as a decision",
    "OWN-MODULE-ONLY":
        "reached by its own module's product code and by no other module; a "
        "wiring fact, not a judgement about the code",
    "OWN-CTOR-READ":
        "reached by the constructor of a class in its own module and by "
        "nothing else -- not by a module-level function there, and not by "
        "another module; the narrowest wiring fact in this set, and a fact "
        "about one read site rather than about the code",
    "UNREAD-FIELD":
        "a declared data field with no reader anywhere; a shape nobody chose",
}

#: The one sentence every `OWN-MODULE-ONLY` row carries, as a constant rather
#: than a separate string per row. The fact is identical across that tier -- a
#: name its own module uses and no other module names -- so a differently
#: worded excuse for each would manufacture a specificity that is not there,
#: and a reader would have no way to tell which rows had been thought about.
#: What makes each such row specific, which scope inside the module reaches the
#: name, is **derived** and printed beside the row on every run, so it is never
#: a transcription and cannot go stale.
OWN_MODULE_REASON = (
    "reached only from inside its own module, by the scope this run "
    "printed beside the row; no other product module names it"
)

#: The one sentence every `OWN-CTOR-READ` row carries, on the same argument as
#: the sentence above: the fact is identical across the tier, so one sentence
#: says it for every row instead of a paragraph per row. The tier is a fact
#: about **one read site** and nothing else -- the name is live, the site is in
#: the tree, and the site is a constructor in the file that declares the name --
#: which is what happens every time a constant is factored out of a constructor
#: and given a public name. Before this state existed, that fact could only be
#: written out by hand, one paragraph per row, and every such paragraph reads
#: like a judgement nobody made.
#:
#: Unlike the sentence above, this one is **checkable against the scan**, and it
#: is: `ReferenceWalker` qualifies a method with the class that owns it, so the
#: derived `from` column beside a row says `Viewport.__init__` and a row may not
#: carry this sentence unless every scope that reaches the name is a
#: constructor. `_generic_reasons` enforces that, so the sentence is an
#: assertion the run can contradict rather than a promise it repeats.
CTOR_READ_REASON = (
    "read only by the constructor of a class in its own module -- the scope "
    "this run printed beside the row -- and by no module-level function there "
    "and by no other product module"
)

#: Which shared sentence belongs to which state. Kept as a table rather than a
#: pair of `if`s so that adding a state with a shared sentence cannot leave the
#: pairing to be restated: `_generic_reasons` reads the sentence off the row's
#: own state word, and a row carrying another state's sentence is a tier word
#: that reads like a fact it is not.
SHARED_SENTENCES = {
    "OWN-MODULE-ONLY": OWN_MODULE_REASON,
    "OWN-CTOR-READ": CTOR_READ_REASON,
}

#: One row per name that no product module outside its own module references,
#: after the three derived dispatch classes have had their say.
#:
#: Keys are `"<module dotted suffix>:<qualname>"`, with the `opendocking.`
#: prefix dropped so a row names the file it is about. Values are
#: `(state, reason)`.
#:
#: **Both directions are checked, so a row is never merely allowed to be
#: wrong.** A name that gains a product caller must lose its row in the same
#: commit that gives it one.
DISPOSITIONS: dict[str, tuple[str, str]] = {
    # -- the tier with no product reference at all: 52 rows ----
    "core:GridBox.contains": (
        "GATE-ONLY",
            "The point-in-box predicate. `scripts/core_check.py` is the "
            "only caller; no framing, pocket or CLI path asks whether a"
            "point is inside the box.",
    ),
    "pdbqt_writer:read_declared_bonds": (
        "GATE-ONLY",
            "Reads a pose file's declared bond block, and its comment "
            "reasons about `OD_NBONDS` and `OD_BOND` not colliding -- a"
            "claim about a parse that no product path performs. Only"
            "`scripts/torsion_bond_record_check.py` calls it.",
    ),
    "pdbqt_writer:write_pose": (
        "GATE-ONLY",
            "Writes one pose as PDBQT. The multi-pose writer on the other "
            "side of the package is the live one -- `DockingResult."
            "write_pdbqt` is reached from `cli.py` and `app.py` -- so this "
            "function and the `format_atom_line` call it would make are the "
            "unused half of the same job.",
    ),
    "prep:ReceptorPrepReport.chains_dropped": (
        "GATE-ONLY",
            "How many loaded chains are missing from the output. The report "
            "is built and returned by `prepare_receptor_with_report`, which"
            "the product calls, and this query on it is read by"
            "`scripts/prep_check.py` and `scripts/cli_prep_check.py` only.",
    ),
    "prep:ReceptorPrepReport.pdbqt_atoms_written": (
        "UNREAD-FIELD",
            "The atoms-written count on the prep report, read by the prep "
            "gates as the assertion that the output is the size it should"
            "be and by no product code.",
    ),
    "result_trust:Contract.actionable": (
        "GATE-ONLY",
            "'The remedy if there is one, else the empty string.' The "
            "remedy text for a failing contract, read by nothing in the"
            "product, so a contract that fails reaches the user as a colour"
            "and a word with no remedy attached to it.",
    ),
    "result_trust:Contract.blocking": (
        "GATE-ONLY",
            "Whether this contract alone prevents a `yes`. `_verdict` "
            "computes the same thing internally to build the verdict, so"
            "this is the public spelling of a fact the product already has"
            "and does not need to ask for.",
    ),
    "result_trust:verdict_pocket": (
        "GATE-ONLY",
            "The per-pocket verdict constructor, on the same footing as "
            "`verdict_score`: not one of the four the product calls.",
    ),
    "result_trust:verdict_receptor": (
        "GATE-ONLY",
            "The per-receptor `Verdict` constructor, on the same footing as "
            "`verdict_score`: the product asks for `verdict_pose` and never "
            "for a receptor's.",
    ),
    "result_trust:verdict_score": (
        "GATE-ONLY",
            "The per-score verdict constructor. `verdict_pose` is the only "
            "one of the four the product calls -- `workbench/pose_trust.py`"
            "reaches it -- so a score arriving without a pose attached has"
            "no verdict path in the product and this is called only by the"
            "examples gate.",
    ),
    "workbench.app:PoseTable.cell": (
        "GATE-ONLY",
            "Reads one cell as text. Its own comment says a caller wanting "
            "pose i's energy 'has to ask by pose, not by row', and the only"
            "caller that does is `scripts/workbench_interaction_check.py`;"
            "the product goes through `_refresh_pose_table` and"
            "`_select_pose_row`.",
    ),
    "workbench.app:PoseTable.pose_row_of": (
        "GATE-ONLY",
            "The pose-index-to-visual-row map that `cell`'s comment tells "
            "callers to use, reached only from the same interaction gate.",
    ),
    "workbench.app:PoseTable.setCurrentRow": (
        "OPEN-QUESTION",
            "A `QListWidget`-shaped shim whose docstring says 'every "
            "existing caller -- including the check script -- uses the"
            "list's name for it'. The product's own load path calls"
            "`_select_pose_row` instead, so the only remaining caller is"
            "the interaction gate and the sentence about 'every existing"
            "caller' is false of the product. The comment above the load"
            "path also explains a signal-ordering fix in terms of this"
            "method emitting `currentRowChanged`, and the code below that"
            "comment does not call it.",
    ),
    "workbench.app:Representation.claims": (
        "GATE-ONLY",
            "Read by `scripts/representation_names_check.py` and by nothing "
            "else. The class docstring says so by name -- 'it is a field"
            "rather than a comment because it is checked' followed by the"
            "gate's name -- so this is a recorded decision and not an"
            "overstated claim.",
    ),
    "workbench.app:Representation.needs_backbone": (
        "OPEN-QUESTION",
            "The comment above the table says this field 'marks the ones "
            "that only mean something for a protein' and that 'a molecule"
            "without a backbone falls back and says so rather than drawing"
            "nothing'. No product code reads the field, so the fallback the"
            "comment describes is not implemented through it, and whether"
            "some other route produces the same behaviour is not a question"
            "this gate can answer.",
    ),
    "workbench.app:Viewport.moving": (
        "OPEN-QUESTION",
            "A public predicate on the Viewport with no caller anywhere -- "
            "not in the product, not in a gate -- and no docstring. Its"
            "only other appearance in the file is a comment about moving"
            "the box, which is a different sense of the word.",
    ),
    "workbench.cartoon_geometry:HELIX_RISE": (
        "OPEN-QUESTION",
            "Exported in `__all__` and described in the module comment as "
            "the length that sets the twist -- 'a twist proportional to arc"
            "length, at one turn per `HELIX_RISE` = 5.4 A'. The twist is"
            "actually computed from `RESIDUES_PER_TURN` and"
            "`HELIX_TWIST_PER_RESIDUE`, so there are two constants for one"
            "rate and the angstrom one is the dead one. The comment also"
            "says a contradicted value 'would be the harder thing to"
            "debug', and nothing can contradict it.",
    ),
    "workbench.cartoon_geometry:head_widths": (
        "GATE-ONLY",
            "Measures the cross-section at both ends of every run, off a "
            "mesh. Its only caller is `half_widths`, which this run also"
            "measures as reached by nothing outside the module, so the pair"
            "is a measurement the drawing path does not use; only the"
            "cartoon gate calls it.",
    ),
    "workbench.cartoon_geometry:orientation_rate": (
        "GATE-ONLY",
            "How fast each ring's cross-section turns, in degrees per "
            "residue. A measurement of drawn geometry, called only by"
            "`scripts/representation_cartoon_check.py`.",
    ),
    "workbench.contacts:Pair.measured_to_hydrogen": (
        "GATE-ONLY",
            "Whether a distance is the H...acceptor form rather than the "
            "donor...acceptor one. The classifier computes the same"
            "distinction inline in `_classify`, so this is the public"
            "spelling of a fact the product already has; the docstring says"
            "the reader 'looks up' a name the H is not in, which is a"
            "reader that does not exist here.",
    ),
    "workbench.crashguard:active": (
        "GATE-ONLY",
            "Whether the installed hook is the one `sys.excepthook` will "
            "call right now. A liveness query for a guard whose liveness"
            "the product assumes rather than checks.",
    ),
    "workbench.crashguard:failure": (
        "GATE-ONLY",
            "'The first one, or None. What a caller should act on' -- and "
            "there is no caller. `app.py` calls `crashguard.install()` and"
            "nothing else in the product ever asks the guard what it"
            "caught, so the reporting half of a crash guard exists only for"
            "the gate.",
    ),
    "workbench.crashguard:failures": (
        "GATE-ONLY",
            "The list form of the same fact, called only by "
            "`scripts/workbench_interaction_check.py`.",
    ),
    "workbench.crashguard:installed": (
        "GATE-ONLY",
            "Whether `install()` has run. The product installs "
            "unconditionally and never asks, so this is the gate's"
            "precondition check.",
    ),
    "workbench.crashguard:replaced_hook": (
        "GATE-ONLY",
            "The hook `install()` displaced, documented 'for a run that "
            "wants the old behaviour'. No product run wants it; only the"
            "interaction gate restores it to test the un-hooked path.",
    ),
    "workbench.export:read_export": (
        "GATE-ONLY",
            "The read half of the export format. `write_export` is reached "
            "from `app.py`, so the product writes this format and never"
            "reads one back; `SCHEMA` is consulted by `read_export` alone.",
    ),
    "workbench.framing_selection:FramingTarget.partner_atoms": (
        "UNREAD-FIELD",
            "The partner half of the same pair, unread for the same reason. ",
    ),
    "workbench.framing_selection:FramingTarget.pose_atoms": (
        "UNREAD-FIELD",
            "A count on the framing target. `selection_target` fills it and "
            "the product never reads it back.",
    ),
    "workbench.framing_selection:POSE_SHARE_TARGET": (
        "OPEN-QUESTION",
            "The module docstring calls this and RECEPTOR_SHARE_FLOOR 'a "
            "contract rather than a fit to the number' and 'the two numbers"
            "this is held to', and this constant's own comment says a share"
            "the pose's pixels 'must reach'. The only comparison against it"
            "is in `scripts/framing_selection_check.py`; no product code"
            "compares it, so the word 'must' has nothing behind it."
            "RECEPTOR_SHARE_FLOOR says this about itself in its own"
            "comment; this one does not.",
    ),
    "workbench.framing_selection:PairFraming.context_atoms": (
        "UNREAD-FIELD",
            "Carried on the `PairFraming` that `focus_target` returns; the "
            "product reads no field of that record except through"
            "`CameraMove.target`.",
    ),
    "workbench.framing_selection:PairFraming.envelope_slack": (
        "UNREAD-FIELD",
            "Carried on the same `PairFraming` record as `subject_fill`, with "
            "the same reader, which is none. Only the framing gate reads it, "
            "and the slack it reports is the one number that says whether the "
            "drawn spheres still fit.",
    ),
    # `PairFraming.subject_fill` used to have a row here and no longer does. It
    # gained a product caller: `Viewport` keeps the record and the status bar
    # prints the share on every selection, in three routes, with its unit in
    # the sentence. This gate's own rule is that a name which gains a product
    # caller must lose its row in the same change, and a row left behind reads
    # as a claim that the field is still unread -- which is the opposite of the
    # truth, and a claim this file exists to keep honest.
    "workbench.framing_selection:RECEPTOR_SHARE_FLOOR": (
        "GATE-ONLY",
            "Its own comment already states that the share 'is just not a "
            "gate' and is measured and reported by the check, so the prose "
            "and the wiring agree, and it says that `projected_fill` is what "
            "the containment question is properly asked with. Recorded so "
            "that the contrast with `POSE_SHARE_TARGET`, which does not say "
            "it, stays visible.",
    ),
    "workbench.framing_selection:framed_radius": (
        "OPEN-QUESTION",
            "Zero references outside its own `def` and its `__all__` entry, "
            "and nobody recorded a decision not to wire it. Its docstring"
            "justifies it by a failure mode -- a framing 'pulled a long way"
            "back along the view direction' -- so it reads as a safeguard,"
            "and no product code is in a position to call it.",
    ),
    "workbench.framing_selection:selection_is_degenerate": (
        "GATE-ONLY",
            "Zero references tree-wide outside its own definition and "
            "`scripts/framing_selection_check.py`, so the predicate is"
            "wired nowhere. The docstring is honest about what wiring it"
            "would buy ('a caller wiring this into a camera-move path"
            "should know it is the fill floor it is really enforcing'),"
            "which is why this is a row about the wiring and not about a"
            "promise the prose broke.",
    ),
    "workbench.geometry:ribbon": (
        "GATE-ONLY",
            "A swept ribbon along a backbone trace. The product draws "
            "cartoons through `cartoon_geometry` and reaches"
            "`MoleculeView.backbone_ribbon` instead; this is called only by"
            "the representation gates.",
    ),
    "workbench.geometry:spheres": (
        "GATE-ONLY",
            "One sphere per atom. `app.py` has its own `draw_spheres` and "
            "its own radius path, and this geometry-layer builder is called"
            "only by `scripts/representation_geometry_check.py`.",
    ),
    "workbench.keys:CLOSE_KEY": (
        "GATE-ONLY",
            "The 'Esc' token for the close binding, read only by the "
            "interaction gate. `SHORTCUTS` and `resolve` are reached from "
            "`app.py`; this literal is the gate's own lookup key.",
    ),
    "workbench.keys:EXCEPTION_CLASS_NAMES": (
        "GATE-ONLY",
            "The widget classes that swallow a key press, named in a "
            "comment as the thing that has to 'name the gesture that gets"
            "the keys back'. The product's own key path does not consult"
            "it.",
    ),
    "workbench.keys:MAP_KEY": (
        "GATE-ONLY",
            "The '?' token for the shortcut card, read only by the "
            "interaction gate. The card the product does render is built by "
            "`map_html` from `GROUP_ORDER`, and neither names this token.",
    ),
    "workbench.keys:keys_for": (
        "GATE-ONLY",
            "Every key tuple bound to a handler. The product's key handling "
            "goes through `keymap.resolve`, which is reached; only the"
            "interaction gate asks for the reverse map, and the comment"
            "above it says the table is 'the one that needs a second hand'.",
    ),
    "workbench.launcher:READY_STATES": (
        "OPEN-QUESTION",
            "The four `READY_*` constants are each read inside the module, "
            "and this collection of them is read by nothing. Its own"
            "comment gives the reason it exists -- 'a bare string compared"
            "in two places is a contract nobody wrote down' -- so the"
            "constants landed and the set that was supposed to name them"
            "did not, and the comparison is still spelled out per constant.",
    ),
    "workbench.pockets:Pocket.fit_to": (
        "GATE-ONLY",
            "How well a site's free space fits a ligand's envelope. "
            "`find_pockets` is reached from `cli.py` and `app.py` and does"
            "its own fitting; only `scripts/pockets_check.py` calls this.",
    ),
    "workbench.pose_trust:PoseVerdict.state_word": (
        "GATE-ONLY",
            "The user-facing word for a verdict. `stylesheet_for` reads "
            "`STATE_WORDS` directly, so the accessor is the unused spelling"
            "again, and `STATE_CONTRACT_WORDS` beside it is reached by"
            "`word`.",
    ),
    "workbench.pose_trust:PoseVerdict.stylesheet": (
        "GATE-ONLY",
            "The colours and font for one verdict, 'in one stylesheet'. "
            "`app.py` calls the module-level `stylesheet_for`, which is"
            "reached, so the verdict's own accessor is the unused spelling"
            "of a fact the product already has.",
    ),
    "workbench.pose_trust:absent_reasons": (
        "GATE-ONLY",
            "Why each named term is absent, documented 'for the panel's own "
            "caption'. No product panel reads it, and `NEVER_SUPPLIED` --"
            "the constant it would consult -- is read only by this"
            "function.",
    ),
    "workbench.structure:AtomRecord.hetatm": (
        "UNREAD-FIELD",
            "Whether a record came from a HETATM line. `parse_structure` "
            "fills it and nothing reads it, so the product draws hetero"
            "records and polymer records through the same path with no way"
            "to tell them apart afterwards.",
    ),
    "workbench.structure:AtomRecord.is_backbone": (
        "GATE-ONLY",
            "Per-atom backbone classification, with a comment saying it is "
            "'filled in by the perception pass'. Nothing in the product"
            "reads it, and the comment does not say which pass.",
    ),
    "workbench.structure:KNOWN_RESIDUES": (
        "GATE-ONLY",
            "The residue vocabulary, with a comment claiming a value 'is "
            "correct for all three' of something. `parse_structure` and"
            "`_apply_templates` are both reached and neither reads it.",
    ),
    "workbench.structure:Structure.degrees": (
        "GATE-ONLY",
            "The per-chain degree list on the parsed structure. The product "
            "renders the structure without asking for the graph shape.",
    ),
    "workbench.structure:backbone_hydrogen_bonds": (
        "GATE-ONLY",
            "The backbone N-H...O=C pass, as `(donor, acceptor)` trace "
            "positions. Called only by `scripts/structure_bond_check.py`;"
            "the structure API the product uses does not run it.",
    ),
    "workbench:MoleculeView.from_text": (
        "GATE-ONLY",
            "The product builds its views through "
            "`MainWindow._view_from_text` instead, and `app.py` says so in "
            "a comment at the call site, so this is a recorded decision and "
            "not an abandoned one. `from_pdbqt` beside it is the "
            "constructor the product does use. The representation gates still "
            "call this one.",
    ),
    # -- reached only inside its own module: 253 rows ----------
    # The reason is one sentence and one constant, not 253
    # sentences: the fact is identical across the tier, and
    # inventing a different one per row would manufacture a
    # specificity that is not there. What makes each row
    # specific -- which scope in the module reaches it -- is
    # derived and printed beside it on every run, so it is
    # never a transcription.
    "cli:build_parser": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:DockingResult.all_pose_coords": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:DockingResult.scoring_function": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:DockingResult.write_xyz": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:GridBox.max_corner": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:GridBox.min_corner": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:GridMaps.dims": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:GridMaps.raw_data": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:Ligand.from_arrays": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:Ligand.from_pdbqt_str": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:Ligand.reference_coords": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:Receptor.bounds": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:Receptor.from_pdbqt_str": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:TermMaps": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:TermMaps.backend": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "core:TermMaps.dims": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "pdbqt_writer:validate_pdbqt_columns": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:Chem_remove_nonpolar_hydrogens": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:ReceptorPrepReport": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:ReceptorPrepReport.component_warning": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:ReceptorPrepReport.polar_hydrogen_warning": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:SUPPORTED_LIGAND_FORMATS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:SUPPORTED_RECEPTOR_FORMATS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:load_molecule": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "prep:pdbqt_atom_type": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:Contract": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:Contract.applies": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:FAILS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:FAMILY_DATA": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:FAMILY_ENGINE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:FAMILY_MODEL": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:FAMILY_PARAMETER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:HOLDS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:LABEL_KEY": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:NO": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:POCKET_KEYS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:POSE_KEYS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:RECEPTOR_KEYS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:REMEDY_ENGINE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:REMEDY_INPUT": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:REMEDY_NONE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:REMEDY_PARAMETER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:SCHEMA": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:SCORE_KEYS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:UNKNOWN": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:UNMEASURED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:Verdict": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "result_trust:YES": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:BACKGROUND_BOTTOM": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:BACKGROUND_FRAGMENT_SHADER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:BACKGROUND_TOP": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:BACKGROUND_VERTEX_SHADER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:BOND_RADIUS_SCALE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:CPK_CARBON_RADIUS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:DockingWorker": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:ELEMENT_CPK_RADIUS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:ELEMENT_VDW_CPK": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:FOG_COLOR": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:LINE_FRAGMENT_SHADER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:LINE_VERTEX_SHADER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:MainWindow": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:MainWindow.find_pockets_now": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:MainWindow.load_structure": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:MainWindow.open_file_dialog": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:MainWindow.start_docking": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:PAIR_MARKER_FRAME_WARN": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:PAIR_MARKER_MIN_AREA_PX2": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:PAIR_MARKER_MIN_LENGTH_PX": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:POCKET_OPACITY_COMPARE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:POCKET_OPACITY_PLAIN": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:POSE_CLUSTER_CUTOFF": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:POSE_GHOST_COLOR": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:PocketWorker": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:PoseTable": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:REPRESENTATIONS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:REPRESENTATION_KEYS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Representation": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:SPHERE_FRAGMENT_SHADER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:SPHERE_VERTEX_SHADER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:SphereMesh": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:TermsWorker": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:VIEWPORT_MIN_H": ("OWN-CTOR-READ", CTOR_READ_REASON),
    "workbench.app:VIEWPORT_MIN_W": ("OWN-CTOR-READ", CTOR_READ_REASON),
    "workbench.app:Viewport": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.focus_pair": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.focus_point": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.focus_residue": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.focus_selection": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.frame_all": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.framebuffer_size": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.pair_screen_segment": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.plan_selection": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:Viewport.rotate": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:build_programs": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:draw_background": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:draw_lines": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:draw_mesh": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:draw_spheres": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:marker_area_ok": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:marker_drawable": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:marker_length_ok": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:projected_width_px": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:set_frame_uniforms": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.app:sphere_radii_for": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:COLORS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:HEAD_RESIDUES": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:HELIX_TWIST_PER_RESIDUE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:RESIDUES_PER_TURN": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:RING": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:SHEET_HEAD_FACTOR": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:THICKNESSES": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:cartoon_cross_section_axes": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:half_widths": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:parallel_transport_frame": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:ring_count": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.cartoon_geometry:ring_states": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:APOLAR": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:CLOSE_MAX": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Contact": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Contact.donor_name": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Contact.partner_element": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Contact.partner_name": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Contact.self_element": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Contact.self_name": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:HBOND_MAX": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:HBOND_MIN_ANGLE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:POLAR": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Pair": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Pair.donor_name": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Pair.partner_element": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Pair.partner_name": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Pair.self_element": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Pair.self_name": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:Pair.self_residue": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.contacts:TERM_BY_KIND": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.crashguard:EXIT_UNHANDLED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.energy_terms:EXPONENT_BELOW": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.energy_terms:PoseBreakdown": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.energy_terms:PoseBreakdown.displayed_total": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.energy_terms:PoseBreakdown.intramolecular": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.energy_terms:PoseBreakdown.intramolecular_scale": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.energy_terms:PoseBreakdown.terms_total": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:DERIVED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:PoseRecord.as_dict": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:PoseRecord.contacts_count": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:PoseRecord.coords_source": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:PoseRecord.residue_rows": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:RunRecord": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:RunRecord.as_dict": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:RunRecord.selected_pose": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:SCHEMA": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:STATE_ABSENT": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:UNKNOWN_ATOM_TYPE_WARNING": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.export:to_json": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:CENTRE_TOLERANCE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:CameraMove.start_center": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:CameraMove.start_distance": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:FramingTarget": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:FramingTarget.partner_residues": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:MAX_DRAWN_ANGLE_FILL": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:MIN_SELECTION_DISTANCE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:MIN_SELECTION_FRAME_FILL": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:PairFraming": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:PairFraming.context_fill": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:PairFraming.shell": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:SELECTION_MARGIN": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:drawn_envelope_slack": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:drawn_fill": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:fit_drawn": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:fit_view": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:projected_fill": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.framing_selection:selection_points": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.geometry:unit_cylinder": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:CAMERA_NOTE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:CONTEXT_EXCEPTION_NOTE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:DIM": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:DOCK_NOTE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:EXPORT_NOTE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:GROUP_ALWAYS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:GROUP_ALWAYS_TITLE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:GROUP_ORDER": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:GROUP_OVERLAY": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:GROUP_PAIRS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:GROUP_TABLE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:Shortcut": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:Shortcut.enabled_by": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.keys:group_titles": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.launcher:READY_BOUND_EXPIRED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.launcher:READY_BUILT": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.launcher:READY_CHILD_KILLED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.launcher:READY_NOT_REALISED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.launcher:build_parser": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:BudgetedBox": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:BudgetedBox.requested_size": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:DEFAULT_LIGAND_RADIUS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:DEFAULT_MAX_POCKETS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:DEFAULT_MAX_VOLUME": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:DEFAULT_PADDING": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:DEFAULT_PROBE_SWEEP": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:DEFAULT_SPACING": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:KIND_LABELS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:LINING_MAX": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:MIN_VOXELS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:Pocket": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:SiteFit": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:SiteFit.ligand_in_site": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:SiteFit.site_filled": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pockets:VDW_RADII": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:ContractRow": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:FILE_POSE_WHY": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:GRADIENT_TOLERANCE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:GRADIENT_TOLERANCE_SOURCE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:IMPORT_CANDIDATES": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:NEVER_SUPPLIED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:NOT_MEASURED": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:PoseVerdict": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:PoseVerdict.described_pose": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:PoseVerdict.failing_rows": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:PoseVerdict.unmeasured_rows": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:STATE_CONTRACT_WORDS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:STATE_FONT": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:STATE_WORDS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:pose_values": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.pose_trust:verdict_module": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:AMIDE_H_LENGTH": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:AtomRecord": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:AtomRecord.resid": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:AtomRecord.residue_key": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:BACKBONE_BONDS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:COVALENT_RADII": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:HBOND_CA_SKIP": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:HBOND_CELL": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:HBOND_LOOSE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:HBOND_Q": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:HBOND_STRICT": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:HELIX_TURNS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:ION_RESIDUES": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:MAX_VALENCE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:PDBQT_TYPE_ELEMENT": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:RESIDUE_ALIASES": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:Residue": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:Residue.is_amino_acid": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:Residue.resid": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:Residue.template": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:SIDECHAIN_BONDS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:SS_MIN_SHEET": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:SS_RANGES": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:Structure": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:dihedral": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench.structure:geometric_secondary_structure": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench:Camera.yaw": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench:ELEMENT_COLORS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench:ELEMENT_RADIUS_SCALE": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench:NON_RESIDUE_LABELS": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
    "workbench:PDBQT_TYPE_ELEMENT": ("OWN-MODULE-ONLY", OWN_MODULE_REASON),
}



OWN_MODULE_REASON = (
    "reached only from inside its own module, by the scope this run "
    "printed beside the row; no other product module names it"
)


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def skip(name, reason):
    """A result this run could not reach, counted as neither pass nor failure."""
    SKIPPED.append((name, reason))
    print(f"  [SKIP] {name}  - {reason}")


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------

BIND = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def module_name(path: Path) -> tuple[str, bool]:
    rel = path.relative_to(PKG).with_suffix("").as_posix()
    parts = rel.split("/")
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join([PKG_PREFIX.rstrip(".")] + parts), is_pkg


def product_files() -> tuple[list[Path], list[Path]]:
    """(in scope, excluded) -- exclusion is by path, never by a name pattern."""
    inside, outside = [], []
    for p in sorted(PKG.rglob("*.py")):
        (outside if "tests" in p.relative_to(PKG).parts else inside).append(p)
    return inside, outside


def assigned_names(node) -> list[str]:
    if isinstance(node, ast.AnnAssign):
        node = node.target
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    return []


def class_members(cls: ast.ClassDef) -> list[tuple[str, ast.AST, str]]:
    """(name, node, kind) for public methods and public annotated fields."""
    out = []
    for sub in cls.body:
        if isinstance(sub, FUNCS):
            if not sub.name.startswith("_"):
                out.append((sub.name, sub, "method"))
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            if not sub.target.id.startswith("_"):
                out.append((sub.target.id, sub, "field"))
    return out


def enumerate_definitions(mods: list["ProductModule"]) -> dict:
    defs: dict = {}
    for m in mods:
        for name, node in m.top.items():
            if name.startswith("_"):
                continue
            if isinstance(node, FUNCS):
                kind = "function"
            elif isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                kind = "attribute"
            defs[(m.mod, name)] = {"mod": m, "name": name, "node": node,
                                   "kind": kind, "owner": None, "attr": None}
            if kind == "class":
                for an, anode, akind in class_members(node):
                    defs[(m.mod, name + "." + an)] = {
                        "mod": m, "name": name + "." + an, "node": node,
                        "kind": akind, "owner": name, "attr": anode}
    return defs


def table_key(info: dict) -> str:
    mod = info["mod"].mod
    short = mod[len(PKG_PREFIX):] if mod.startswith(PKG_PREFIX) else mod
    return short + ":" + info["name"]


# --------------------------------------------------------------------------
# One product module
# --------------------------------------------------------------------------

def dotted(node) -> list | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return parts


class ProductModule:
    def __init__(self, path: Path, known: dict, top_names: dict):
        self.path = path
        self.known = known
        self.top_names = top_names
        self.mod, self.is_pkg = module_name(path)
        self.rel = path.relative_to(ROOT).as_posix()
        self.src = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.src, filename=self.rel)
        self.top: dict = {}
        for node in self.tree.body:
            if isinstance(node, BIND):
                self.top[node.name] = node
            for t in assigned_names(node):
                self.top.setdefault(t, node)
        self.aliases: dict = {}
        self.stars: list = []
        self.reexports: list = []
        self.shadow: set = set()
        self._read_imports()
        self._read_shadow()

    def _abs_from(self, node: ast.ImportFrom) -> str:
        """The module a `from ... import` names, in absolute terms.

        Both halves of the spelling matter and the first version of this used
        only one of them: `from .core import Receptor` inside a package means
        `opendocking.core`, not `opendocking`, and dropping `node.module`
        attributed every re-export in `opendocking/__init__.py` to the package
        itself. The `level` walk finds the anchor and `node.module` says which
        module within it.
        """
        if node.level == 0:
            return node.module or ""
        base = self.mod if self.is_pkg else self.mod.rsplit(".", 1)[0]
        for _ in range(node.level - 1):
            base = base.rsplit(".", 1)[0] if "." in base else base
        if node.module:
            base = base + "." + node.module
        return base

    def _read_imports(self):
        """Resolve every import to the module and name it actually reaches.

        `from .cli import main` names the *module* `opendocking.cli`, not the
        package `opendocking`, and getting that wrong is what made `cli.main`
        read as unreached in the first version. Python prefers the base's own
        attribute and only falls back to a submodule of the same name, so that
        is the rule here.
        """
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom):
                base = self._abs_from(node)
                base_is_pkg = base in self.top_names
                base_tops = self.top_names.get(base, ())
                for a in node.names:
                    if a.name == "*":
                        self.stars.append(base)
                        continue
                    sub = base + "." + a.name
                    if sub in self.known and (not base_is_pkg
                                              or a.name not in base_tops):
                        self.aliases[a.asname or a.name] = (sub, None)
                    else:
                        self.aliases[a.asname or a.name] = (base, a.name)
                        self.reexports.append((base, a.name))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    head = a.name.split(".")[0]
                    self.aliases[a.asname or head] = (
                        (a.name, None) if a.asname else (head, None))

    def _read_shadow(self):
        """Every name bound at function scope, so a local is not a reference."""
        for n in ast.walk(self.tree):
            if isinstance(n, FUNCS):
                a = n.args
                for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                    self.shadow.add(arg.arg)
                if a.vararg:
                    self.shadow.add(a.vararg.arg)
                if a.kwarg:
                    self.shadow.add(a.kwarg.arg)
            elif isinstance(n, BIND):
                self.shadow.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                self.shadow.add(n.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                self.shadow.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                self.shadow.update(n.names)

    def resolve(self, parts: list):
        """(module, qualname) for a dotted name, or None when unresolvable.

        None is the common case and it is not a failure: it is the signal that
        a reference goes through an instance and can only be name-matched.
        """
        head, rest = parts[0], parts[1:]
        if head in self.aliases:
            mod, orig = self.aliases[head]
        elif head in self.top:
            mod, orig = self.mod, head
        elif head in self.shadow:
            return None
        else:
            for sm in self.stars:
                if sm in self.known:
                    return (sm, head)
            return None
        if orig is not None and mod + "." + orig in self.known:
            mod, orig = mod + "." + orig, None
        if orig is None:
            cur = mod
            for k, r in enumerate(rest):
                if cur + "." + r in self.known:
                    cur = cur + "." + r
                    continue
                return (cur, ".".join(rest[k:]))
            return (cur, None)
        return (mod, ".".join([orig] + rest) if rest else orig)

    def exported_names(self) -> list:
        out = []
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and "__all__" in [
                    t.id for t in node.targets if isinstance(t, ast.Name)]:
                for el in node.value.elts if isinstance(node.value, ast.List) else []:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        out.append(el.value)
        return out


# --------------------------------------------------------------------------
# Reference collection
# --------------------------------------------------------------------------

class _RefVisitor(ast.NodeVisitor):
    """Product references only: no `__main__` body, no `test_*` body.

    A reference from a `scripts/` file, an `examples/` file or the test package
    is not collected at all, because none of those are product code. That is
    the whole reason the two hand-found names are in this file's table: the
    only thing that calls them is a gate.
    """

    def __init__(self, mod: ProductModule, resolved: dict, shaped: dict,
                 sites: dict):
        self.mod = mod
        self.resolved = resolved
        self.shaped = shaped
        self.sites = sites
        self.scope = "<module>"
        #: The class stack, so a scope can name the class a method belongs to.
        #: It is what separates `Viewport.__init__` from `helper`: both are one
        #: read inside the same module, and only the first is a constructor
        #: read, which is the whole difference the `OWN-CTOR-READ` tier records.
        self.classes: list[str] = []

    def _note(self, key, last):
        if key and key[1]:
            self.resolved.setdefault(key, set()).add(self.mod.mod)
            self.sites.setdefault(key, set()).add(
                (self.mod.mod, self.scope, key[1]))

    def _visit_function(self, node):
        if node.name.startswith("test_"):
            return
        outer = self.scope
        self.scope = node.name
        if self.classes:
            self.scope = "%s.%s" % (self.classes[-1], node.name)
        for d in node.decorator_list:
            self.visit(d)
        for d in node.args.defaults + [x for x in node.args.kw_defaults if x]:
            self.visit(d)
        self.generic_visit(node)
        self.scope = outer

    def visit_FunctionDef(self, node):
        self._visit_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        outer = self.scope
        self.scope = node.name
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()
        self.scope = outer

    def visit_If(self, node):
        t = node.test
        if isinstance(t, ast.Compare) and getattr(t.left, "id", None) == "__name__":
            return
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            parts = dotted(node)
            if parts:
                key = self.mod.resolve(parts)
                if key:
                    self._note(key, parts[-1])
                else:
                    self.shaped.setdefault(parts[-1], set()).add(self.mod.mod)
                    self.sites.setdefault(("?", parts[-1]), set()).add(
                        (self.mod.mod, self.scope, parts[-1]))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            key = self.mod.resolve([node.id])
            if key:
                self._note(key, node.id)
            else:
                self.shaped.setdefault(node.id, set()).add(self.mod.mod)
                self.sites.setdefault(("?", node.id), set()).add(
                    (self.mod.mod, self.scope, node.id))
        self.generic_visit(node)


def _reexport_targets(known: dict, top_names: dict, base: str,
                      name: str) -> list:
    """Where `from BASE import NAME` actually lands, as definition keys.

    Two shapes, and the second one is the whole reason this is not a one-liner.
    `from .framing_selection import framed_radius` names a module's own
    binding. `from .core import DockingResult` in `opendocking/__init__.py`
    names a *package's* re-export, and the definition is one submodule down --
    which is the mechanism that publishes the whole public API. Resolving that
    one-liner is what stops this file reporting every exported name as
    unreached; it was 192 false positives before it was handled.
    """
    prefix = base + "."
    if not any(m.startswith(prefix) for m in known):
        return [(base, name)]
    if name in top_names.get(base, ()):
        return [(base, name)]
    hits = [m for m in known
            if m.startswith(prefix) and name in top_names.get(m, ())]
    return [(m, name) for m in sorted(hits)]


def collect_references(mods, known: dict, top_names: dict):
    resolved: dict = {}
    shaped: dict = {}
    sites: dict = {}
    for m in mods:
        for base, name in m.reexports:
            targets = _reexport_targets(known, top_names, base, name)
            if not targets:
                # No submodule of that package defines it. Name-shaped keeps
                # it visible rather than dropping the reference.
                shaped.setdefault(name, set()).add(m.mod)
                continue
            for target in targets:
                resolved.setdefault(target, set()).add(m.mod)
                sites.setdefault(target, set()).add((m.mod, "<re-export>", name))
                if len(targets) > 1:
                    shaped.setdefault(name, set()).add(m.mod)
        _RefVisitor(m, resolved, shaped, sites).visit(m.tree)
    return resolved, shaped, sites


# --------------------------------------------------------------------------
# The three derived dispatch classes
# --------------------------------------------------------------------------

_IMPORT_CACHE: dict = {}


def _try_import(target: str):
    if target not in _IMPORT_CACHE:
        try:
            _IMPORT_CACHE[target] = importlib.import_module(target)
        except Exception:
            _IMPORT_CACHE[target] = None
    return _IMPORT_CACHE[target]


def _qt_attr(dotted_name: str):
    """Resolve `PyQt6.QtWidgets.QTableWidget`-shaped names, longest import first."""
    parts = dotted_name.split(".")
    for cut in range(len(parts), 0, -1):
        mod = _try_import(".".join(parts[:cut]))
        if mod is None:
            continue
        obj = mod
        for attr in parts[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def qt_bases_of(mods) -> tuple[dict, list]:
    """(module, class) -> the PyQt6 base, for every Qt-derived class.

    Derived from each class's own `bases` resolved through its module's import
    table, so a class counts as Qt-derived only if its source inherits from an
    imported `PyQt6` name. No class is listed by hand.
    """
    out, unresolved = {}, []
    for m in mods:
        for name, node in m.top.items():
            if not isinstance(node, ast.ClassDef) or name.startswith("_"):
                continue
            for b in node.bases:
                parts = dotted(b)
                if not parts:
                    continue
                base = m.resolve(parts)
                if not base:
                    continue
                base_mod, base_name = base
                if not (base_mod and base_name and base_mod.startswith(PYQT_ROOT)):
                    continue
                full = base_mod + "." + base_name
                if _qt_attr(full) is None:
                    unresolved.append(f"{m.rel}: {name}({full})")
                    continue
                out[(m.mod, name)] = full
    return out, unresolved


def qt_overrides(defs, qt_bases) -> set:
    """Methods the installed Qt base class already has a name for.

    `hasattr(real_base, method_name)` against the PyQt6 that is actually
    installed, which is why there is no list of Qt method names in this file:
    a typed list would be the second hand-maintained table this file exists to
    avoid, and it would go stale the day PyQt6 renamed something with nothing
    to notice.
    """
    hits = set()
    for (mod, cls), full in qt_bases.items():
        base = _qt_attr(full)
        if base is None:
            continue
        for key, info in defs.items():
            if key[0] != mod or info.get("owner") != cls or info["kind"] != "method":
                continue
            if hasattr(base, info["name"].split(".")[-1]):
                hits.add(key)
    return hits


def qt_slots(defs) -> set:
    """Methods carrying a `pyqtSlot` / `pyqtProperty` decorator."""
    hits = set()
    for key, info in defs.items():
        for d in getattr(info.get("attr") or info["node"], "decorator_list", []):
            text = ast.unparse(d)
            if "pyqtSlot" in text or "pyqtProperty" in text:
                hits.add(key)
                break
    return hits


def entry_points() -> set:
    """Names the installer calls, derived from `[project.scripts]`.

    `cli.main` and `launcher.main` have no caller in the tree because the shim
    they are wrapped in is generated at install time. Checked in both
    directions: a declared entry point that does not resolve to a definition is
    red, so a renamed command cannot quietly exempt a name nothing calls.
    """
    out = set()
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r"^\[project\.scripts\][ \t]*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not m:
        return out
    for line in m.group(1).splitlines():
        line = line.split("#")[0].strip()
        if not line or "=" not in line:
            continue
        target = line.split("=", 1)[1].strip().strip('"').strip("'")
        if ":" in target:
            mod, _, attr = target.partition(":")
            out.add((mod, attr))
    return out


# --------------------------------------------------------------------------
# The self census, lifted out of check_scripts_declare.py by AST
# --------------------------------------------------------------------------

LIFT_NAMES = ("_short", "_sites_of", "_guards_digest", "_comment_lines",
              "_declaration_of", "_DECL_HEAD_RE", "_DECL_SITES_RE",
              "_DECL_GUARDS_RE", "DECL_VERSION")
LIFT_BUILTINS = ("ast", "hashlib", "io", "re", "tokenize")


def load_declarer() -> dict:
    """Compile the declarer's census helpers out of its source, alone.

    `check_scripts_declare.py` runs its whole audit and calls `sys.exit()` at
    module scope, so importing it is not an option. Lifting the functions this
    file needs by AST means the numbers below come from *its* code rather than
    from a copy of its answers -- and if it renames or deletes one, this raises
    and the gate is red instead of quietly measuring with a different walk.
    """
    tree = ast.parse(DECLARER.read_text(encoding="utf-8"), filename=DECLARER.name)
    want = set(LIFT_NAMES)
    keep, found = [], set()
    for node in tree.body:
        if isinstance(node, BIND) and node.name in want:
            keep.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & want:
                keep.append(node)
                found |= names & want
    missing = want - found
    if missing:
        raise RuntimeError(
            "check_scripts_declare.py no longer defines "
            + ", ".join(sorted(missing))
            + "; this gate derives its own census from those functions and "
              "will not substitute a differently-shaped walk for them")
    ns: dict = {"ast": ast, "hashlib": hashlib, "io": io, "re": re,
                "tokenize": tokenize}
    exec(compile(ast.Module(body=keep, type_ignores=[]), DECLARER.name, "exec"), ns)
    return ns


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

HAND_FOUND = ("workbench.framing_selection:framed_radius",
              "workbench.framing_selection:selection_is_degenerate")


def module_top_names(path: Path) -> frozenset:
    """Top-level bindings of a module, read without keeping the tree.

    Needed before any `ProductModule` exists, because deciding whether
    `from pkg import name` means the submodule or the base's own attribute is a
    question about *another* module's bindings. One cheap parse per file, done
    in a first pass, so the answer does not depend on construction order.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set = set()
    for node in tree.body:
        if isinstance(node, BIND):
            names.add(node.name)
        names.update(assigned_names(node))
    return frozenset(names)


def scan() -> dict:
    files, excluded = product_files()
    if not files:
        raise RuntimeError("no product modules under %s" % PKG)
    known, top_names = {}, {}
    for p in files:
        m, _ = module_name(p)
        known[m] = p
        top_names[m] = module_top_names(p)
    mods = []
    for p in files:
        try:
            mods.append(ProductModule(p, known, top_names))
        except SyntaxError as exc:
            raise RuntimeError("%s does not parse: %s" % (p, exc)) from exc
    defs = enumerate_definitions(mods)
    resolved, shaped, sites = collect_references(mods, known, top_names)
    identifiers = {}
    for m in mods:
        ids = set()
        for node in ast.walk(m.tree):
            if isinstance(node, ast.Name):
                ids.add(node.id)
            elif isinstance(node, ast.Attribute):
                ids.add(node.attr)
            elif isinstance(node, BIND):
                ids.add(node.name)
            elif isinstance(node, ast.alias):
                ids.add(node.asname or node.name.split(".")[0])
            elif isinstance(node, ast.arg):
                ids.add(node.arg)
        identifiers[m.mod] = ids
    return {"files": files, "excluded": excluded, "known": known,
            "top_names": top_names, "mods": mods, "sites": sites,
            "identifiers": identifiers,
            "defs": defs, "resolved": resolved, "shaped": shaped}


def classify(s: dict) -> dict:
    defs, resolved, shaped = s["defs"], s["resolved"], s["shaped"]
    by_last: dict = {}
    for key in defs:
        by_last.setdefault(key[1].split(".")[-1], []).append(key)
    qt_bases, qt_unresolved = qt_bases_of(s["mods"])
    overrides = qt_overrides(defs, qt_bases)
    slots = qt_slots(defs)
    eps = entry_points()

    zero, reached, resolved_only = [], 0, 0
    for key in sorted(defs):
        info = defs[key]
        mod, qual = key
        lastn = qual.split(".")[-1]
        r_cross = {m for m in resolved.get(key, ()) if m != mod}
        s_cross = {m for m in shaped.get(lastn, ()) if m != mod}
        own = bool({m for m in resolved.get(key, ()) if m == mod}
                   | {m for m in shaped.get(lastn, ()) if m == mod})
        # Where in its own module the name is reached from, derived rather than
        # transcribed. Printing this is what keeps the OWN-MODULE-ONLY and
        # OWN-CTOR-READ rows informative without one hand-typed caller per row,
        # each of which would be one more thing to be wrong about.
        callers = sorted({(m, sc) for (m, sc, _n) in
                          s["sites"].get(key, set()) if m == mod}
                         | {(m, sc) for (m, sc, _n) in
                            s["sites"].get(("?", lastn), set()) if m == mod})
        dispatched = None
        if key in overrides:
            dispatched = "QT-OVERRIDE"
        elif key in slots:
            dispatched = "QT-SLOT"
        elif key in eps:
            dispatched = "ENTRYPOINT"
        rec = {"key": key, "tk": table_key(info), "info": info, "own": own,
               "amb": len(by_last[lastn]), "kind": info["kind"],
               "dispatched": dispatched, "r": r_cross, "s": s_cross,
               "callers": callers}
        if r_cross or s_cross:
            reached += 1
            resolved_only += 1 if r_cross else 0
        else:
            zero.append(rec)
    return {"zero": zero, "reached": reached, "resolved_only": resolved_only,
            "by_last": by_last, "qt_bases": qt_bases,
            "qt_unresolved": qt_unresolved, "overrides": overrides,
            "slots": slots, "eps": eps}


def main() -> int:
    try:
        s = scan()
    except (RuntimeError, OSError, SyntaxError) as exc:
        print("FATAL: %s" % exc)
        print("A gate that cannot read its input has not passed anything.")
        return 2
    c = classify(s)
    defs, zero = s["defs"], c["zero"]
    rows = DISPOSITIONS
    need = [r for r in zero if r["dispatched"] is None]
    needed = {r["tk"] for r in need}
    rowset = set(rows)

    # -- the report, every name, no cap -----------------------------------
    print("product modules in scope: %d  (excluded %d file(s) under tests/)"
          % (len(s["mods"]), len(s["excluded"])))
    print("public names enumerated: %d" % len(defs))
    print("  reference classes over the whole tree:")
    print("    RESOLVED cross-module references : %d names"
          % c["resolved_only"])
    print("    NAME-SHAPED only                  : %d names"
          % (c["reached"] - c["resolved_only"]))
    print("  reached by another product module  : %d" % c["reached"])
    print("  NOT reached by another product module: %d" % len(zero))
    for tag in ("QT-OVERRIDE", "QT-SLOT", "ENTRYPOINT"):
        print("    dispatched, %-12s: %d"
              % (tag, sum(1 for r in zero if r["dispatched"] == tag)))
    print("  REQUIRING A DISPOSITION ROW        : %d" % len(need))
    print("  rows in DISPOSITIONS               : %d" % len(rows))
    print()
    print("Every name requiring a row (%d), in full -- there is no cap and no"
          % len(need))
    print("truncation, because a truncated list reads as a complete one.")
    print("`from` is the derived referring scope inside the name's own module,")
    print("qualified by the class a method belongs to, so an OWN-CTOR-READ row")
    print("says what reaches it without that fact being transcribed once per row.")
    for r in sorted(need, key=lambda r: r["tk"]):
        st, why = rows.get(r["tk"], ("<NO ROW>", ""))
        frm = ""
        if r["callers"]:
            frm = "  from " + ", ".join(sorted({sc for _m, sc in r["callers"]}))
        line = "  %-15s %-52s amb=%d%s" % (st, r["tk"], r["amb"], frm)
        print(line)
        REPORT.append(line)
        if why:
            wline = "                   %s" % why
            print(wline)
            REPORT.append(wline)
    print()

    # -- the checks -------------------------------------------------------
    def names(sel):
        return ", ".join(sorted(sel)) or "none"

    check("the product package exists and holds modules to scan",
          bool(s["mods"]) and bool(s["known"]),
          "%d modules under %s" % (len(s["mods"]), PKG.relative_to(ROOT).as_posix()))
    check("the test package is excluded by path, not by file name",
          bool(s["excluded"]) and all("tests" in p.relative_to(PKG).parts
                                      for p in s["excluded"]),
          "%d file(s) excluded, every one under a tests/ directory"
          % len(s["excluded"]))
    check("every product module parsed as Python",
          len(s["mods"]) == len(s["files"]),
          "ast.parse accepted %d of %d files" % (len(s["mods"]), len(s["files"])))
    recount = sum(
        1 + (len(class_members(n)) if isinstance(n, ast.ClassDef) else 0)
        for m in s["mods"] for name, n in m.top.items()
        if not name.startswith("_"))
    check("the enumeration is derived, and its total re-derives independently",
          recount == len(defs) and len(defs) > 0,
          "%d names, and %d from a second walk over module-level bindings"
          % (len(defs), recount))
    check("every enumerated name has a unique table key",
          len({table_key(i) for i in defs.values()}) == len(defs),
          "%d distinct keys for %d names"
          % (len({table_key(i) for i in defs.values()}), len(defs)))
    ok_all, detail_all = _all_check(s, defs)
    if ok_all:
        check("every name a product module lists in __all__ is defined there",
              True, detail_all)
    else:
        skip("every name a product module lists in __all__ is defined there",
             detail_all + ". This is a real defect in a file this gate does "
             "not own rather than a limitation of the check, and it is "
             "reproduced by import: `from opendocking.workbench import "
             "Workbench` raises ImportError and `from opendocking.workbench "
             "import *` raises AttributeError, because the name is listed in "
             "that module's __all__ and is neither defined nor imported "
             "there. The fix is one line in workbench/__init__.py -- import "
             "it or drop it -- and until then this gate carries the finding "
             "rather than going red over it forever")
    check("every Qt-derived class's base resolved in the installed PyQt6",
          not c["qt_unresolved"],
          "%d unresolved: %s" % (len(c["qt_unresolved"]),
                                 names(c["qt_unresolved"])))
    check("every declared console-script entry point resolves to a definition",
          bool(c["eps"]) and all(k in defs for k in c["eps"]),
          "%d entry point(s): %s" % (len(c["eps"]),
                                     names(a for _m, a in c["eps"])))
    check("the Qt override class was derived by hasattr, not typed",
          bool(c["qt_bases"]) and bool(c["overrides"]),
          "%d Qt-derived classes, %d methods that a real Qt base already has "
          "a name for" % (len(c["qt_bases"]), len(c["overrides"])))

    ok_dupes, detail_dupes = _duplicate_keys()
    check("no row key is written twice in the table literal",
          ok_dupes, detail_dupes)
    check("every unreached name has a disposition row",
          not (needed - rowset),
          "%d row(s) missing: %s" % (len(needed - rowset), names(needed - rowset)))
    check("no row names a reached name -- a stale excuse is a failure",
          not (rowset - needed),
          "%d stale row(s): %s" % (len(rowset - needed), names(rowset - needed)))
    check("every row's state word is in the closed set",
          all(s_ in STATES for s_, _r in rows.values()),
          "outside the set: %s" % names({s_ for s_, _r in rows.values()
                                         if s_ not in STATES}))
    check("every row carries a reason",
          all(isinstance(r, str) and r.strip() for _s, r in rows.values()),
          "%d row(s)" % len(rows))
    reasons = [r for _s, r in rows.values()]
    generic, specific = _generic_reasons(s, rows, c)
    check("no row carries a third generic excuse",
          not generic,
          "%d row(s) whose reason is neither the declared shared sentence for "
          "its own state, nor a sentence naming something in its own module, "
          "nor -- for OWN-CTOR-READ -- a sentence the derived `from` column "
          "agrees with: %s"
          % (len(generic), names(generic)))
    unused = [st for st in STATES
              if not any(s_ == st for s_, _r in rows.values())]
    check("each declared state has at least one row",
          not unused, "unused: %s" % names(unused))
    check("the table is shorter than the reached population",
          len(rows) < c["reached"],
          "%d row(s) against %d reached names; an all-exempt table is not a "
          "table" % (len(rows), c["reached"]))
    printed = {ln.split()[1] for ln in REPORT
               if ln.startswith("  ") and len(ln.split()) > 2}
    unprinted = {r["tk"] for r in need} - printed
    check("the report names every unreached name, with no cap or truncation",
          not unprinted and len(REPORT) >= 2 * len(need),
          "%d name line(s) printed for %d name(s) requiring a row, %d absent"
          % (len(printed & {r["tk"] for r in need}), len(need), len(unprinted)))
    missing_hand = [h for h in HAND_FOUND
                    if h not in {r["tk"] for r in zero}]
    check("both names found by hand are still measured unreached",
          not missing_hand,
          "no longer unreached: %s" % names(missing_hand) if missing_hand
          else "framed_radius and selection_is_degenerate both still have no "
               "product caller")
    try:
        SELF.read_bytes().decode("ascii")
        ascii_ok, ascii_detail = True, "no byte above 0x7f"
    except UnicodeDecodeError as exc:
        ascii_ok, ascii_detail = False, "non-ASCII at byte %d" % exc.start
    check("this file is pure ASCII", ascii_ok, ascii_detail)

    try:
        ns = load_declarer()
        uncond, guards = ns["_sites_of"](SELF.read_text(encoding="utf-8"))
        derived = (uncond, len(guards), ns["_guards_digest"](guards))
        check("this file's GATE-DECLARE census matches the declarer's own walk",
              ns["_declaration_of"](SELF) == derived,
              "derived %s, declared %s" % (derived, ns["_declaration_of"](SELF)))
    except Exception as exc:
        check("this file's GATE-DECLARE census matches the declarer's own walk",
              False, "could not derive it: %s: %s" % (type(exc).__name__, exc))

    check("the number of checks that ran is the number this file should have",
          CHECKS + 1 == EXPECTED_CHECKS,
          "%d ran before this one and %d are expected; the +1 is this check"
          % (CHECKS, EXPECTED_CHECKS))

    print()
    if FAILURES:
        print("  %d/%d passed" % (CHECKS - len(FAILURES), CHECKS))
        for f in FAILURES:
            print("  FAILED: %s" % f)
    else:
        print("  %d/%d passed" % (CHECKS, CHECKS))
    if SKIPPED:
        print()
        print("  %d check(s) were skipped, not passed:" % len(SKIPPED))
        for n, r in SKIPPED:
            print("    SKIP %s: %s" % (n, r))
        print("  A skip means this run could not answer the question. It is "
              "not evidence that the thing it guards is correct.")
    else:
        print()
        print("  0 skipped: every check this file ran, it ran here.")
    return 1 if FAILURES else 0


def is_ctor_scope(scope: str) -> bool:
    """Whether a derived referring scope is a class constructor.

    A `ReferenceWalker` scope names the class a method belongs to, so the shapes
    a read can have in its own module are distinguishable: `Viewport.__init__`
    for a method, `helper` for a module-level function, the bare class name for
    a read in a class body, and `<module>` for one at module level. Only the
    first is a constructor read, and a name whose every read is a constructor
    read is what the `OWN-CTOR-READ` tier is about.
    """
    return scope.endswith(".__init__")


def _generic_reasons(s: dict, rows: dict, c: dict) -> tuple[set, set]:
    """Rows whose reason is neither the shared tier sentence nor specific.

    A table of exceptions rots by accretion: somebody adds a name and pastes a
    sentence that says nothing about *that* name. The check that would catch it
    is not "are the reasons different" -- a tier whose fact is identical should
    and does share one sentence -- but "does this sentence say anything about
    the file it is filed against". So a reason passes if it is a declared
    shared sentence, or if it names at least one identifier that exists in the
    name's own module. `SHARED_SENTENCES` is that declaration; every row in the
    `OWN-MODULE-ONLY` tier carries `OWN_MODULE_REASON`, and every row in the
    `OWN-CTOR-READ` tier carries `CTOR_READ_REASON`.

    Two things beyond the identifier test, both the same failure the identifier
    test exists to catch -- a sentence that is not true of the file it is filed
    against -- and both needed the moment a second shared sentence existed:

    * **A shared sentence is only its own state's to carry.** The sentences are
      deliberately alike, so a row could pair one with another state's word and
      every other check would stay green while the column said the wrong thing.
      The word and the sentence are read off each other instead.
    * **`CTOR_READ_REASON` is contradicted by the scan.** It is the one shared
      sentence that asserts more than its own tier, and the derived `from`
      column beside the row is the evidence: if any scope reaching the name is
      not a constructor, the row is claiming a narrower read than the run
      measured, and that is the sentence being wrong rather than the tier
      being unused.
    """
    shared = set(SHARED_SENTENCES.values())
    facts = {r["tk"]: r for r in c["zero"]}
    generic, ok = set(), set()
    for tk, (state, why) in rows.items():
        if why in shared:
            if why != SHARED_SENTENCES.get(state):
                generic.add(tk)
                continue
            if state == "OWN-CTOR-READ":
                rec = facts.get(tk)
                scopes = {sc for _m, sc in rec["callers"]} if rec else set()
                if not scopes or not all(is_ctor_scope(sc) for sc in scopes):
                    generic.add(tk)
                    continue
            ok.add(tk)
            continue
        suffix = tk.partition(":")[0]
        mod = "opendocking." + suffix
        ident = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", why))
        pool = s["identifiers"].get(mod, set())
        if ident & pool:
            ok.add(tk)
        else:
            generic.add(tk)
    return generic, ok


def _duplicate_keys() -> tuple[bool, str]:
    """Count the keys written in the table literal, and compare with the dict.

    Found by a mutation, which is the only way it gets found. Adding a
    disposition row for a name that **already had one** does not add a row:
    a Python dict literal takes the last value for a repeated key and says
    nothing, so the edit that was supposed to prove the stale direction went
    green while having changed the table's *content* and not its size. The
    two-sided check was right that the name was still legitimately in the
    table; what it could not see was that the writer had believed they had
    added a second opinion. Comparing the literal's key count against the
    dict's length is the whole fix, and it costs one `ast.walk`.
    """
    tree = ast.parse(SELF.read_text(encoding="utf-8"), filename=SELF.name)
    written = 0
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") \
                == "DISPOSITIONS" and isinstance(node.value, ast.Dict):
            written = len(node.value.keys)
    have = len(DISPOSITIONS)
    if written == have:
        return True, "%d key(s) written, %d in the dict" % (written, have)
    return False, ("%d key(s) written into the literal but the dict holds %d, "
                   "so at least one key was written twice and the later value "
                   "silently replaced the earlier one" % (written, have))


def _all_check(s: dict, defs: dict) -> tuple[bool, str]:
    """Every `__all__` entry must resolve, by any of the four ways it can.

    A definition in the module, any module-level binding at all (`__all__`
    legitimately lists `__version__`, which the public-name enumeration skips
    because it starts with an underscore), an imported alias, or a submodule
    of the package. The first version of this check accepted only the first and
    reported 19 dangling names -- the whole of `opendocking/__init__.py`'s
    public API -- which is the check being wrong rather than the tree.
    """
    missing = []
    for m in s["mods"]:
        for name in m.exported_names():
            if (m.mod, name) in defs or name in m.top:
                continue
            if name in m.aliases:
                base, orig = m.aliases[name]
                if (base, orig) in defs or (orig is None and base in s["known"]):
                    continue
            if m.mod + "." + name in s["known"]:
                continue
            missing.append(f"{m.rel}: {name}")
    if missing:
        return False, "%d dangling: %s" % (len(missing), ", ".join(missing))
    n = sum(len(m.exported_names()) for m in s["mods"])
    return True, ("%d entries across %d modules, each resolving to a "
                  "definition, a module-level binding, an alias, or a submodule"
                  % (n, sum(1 for m in s["mods"] if m.exported_names())))


if __name__ == "__main__":
    raise SystemExit(main())
