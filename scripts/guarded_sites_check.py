"""Guarded call sites: what every guard does to the tally, and which guards can
be false on a headless machine with nothing else failing.

Run:  python scripts/guarded_sites_check.py

**The defect this file exists for.** A gate records a result by calling
`check(...)`, and a large share of the call sites in this tree sit behind a
guard: an `if`, a `for`, a `try`, a `with`. **A guarded site is a site that may
not run**, and a site that did not run leaves no trace in the tally. The pin
(`EXPECTED_CHECKS`) is the only thing left that notices, and the pin notices a
*smaller* total rather than a missing claim, which is the weakest possible form
of noticing. This file measures the guarded sites instead of assuming they ran,
and it asks one question of each guard: **when this guard is false, what happens
to the result this site would have recorded?**

The answer has more shapes than "it runs" and "it does not", and the shapes have
to be told apart *in effect* rather than in wording. A guard that turns the
result into a `SKIP` and a guard that deletes it are different, and a guard
whose body ends in `return` so that the statement after it records the same
claim is different again from a guard that simply has no `else`.

**Two axes, and the second is the one that matters.**

* *Axis 1, what the guard does to the tally* -- the `CATEGORIES` table. Ten
  categories plus `unclassified`, derived from the syntax tree, exhaustive by
  construction, and two-sided: a category with no live site is a red, so the
  table cannot keep a row for a shape the tree no longer has.
* *Axis 2, whether the guard reads the machine* -- `ENV_NAMES` plus the
  structural `FS_ATTRS` probe. This is the axis the brief's hard failure lives
  on: a guard that can be false on the machine CI runs on, with no other check in
  the same file failing when it is. `ENV_REMOVALS` is the declared inventory of
  the guards that answer yes, one row per `(file, guard)`, and it is two-sided
  in both directions -- an env-decoded guard with no row is a red, and a row
  naming a guard that is no longer there is a red, because a stale excuse reads
  exactly like a live one.

**The headline file.** `workbench_interaction_check.py` is the largest gate here
(347 declared checks, 351 call sites) and the one this project has been reading
for correctness all session. It is reported on by name below, per guard, and the
answer to the headless question is printed in full rather than summarised.

**The file set is derived, never typed.** The `role_of` that decides which file
is a gate is *lifted out of the source of `check_scripts_declare.py` by AST and
executed*, rather than re-implemented here. That file has no
`if __name__ == "__main__"` guard and ends in `sys.exit(...)`, so importing it
would run the whole auditor and exit; re-implementing it would leave two
classifications that could silently disagree. Lifting avoids both. The lift is
checked, not trusted: `check_role_agreement` re-derives four role assignments
from four *different* readers -- an `ast.FunctionDef` search, a `run_script`
**call** search, an `ast.Import` search, and a regex over the `EXCLUSIONS`
**text** -- and compares them to what the lifted `role_of` said, for four files
this file did not write.

**What an AST scan cannot see, stated before the counts.**

* **Dynamic dispatch.** A call reached through `getattr`, a dict of handlers, a
  plugin table, or a Qt signal is a call this walk never sees as a site. A
  decorator that wraps a call site into a deferred one changes whether the site
  runs without changing a single node this walk reads.
* **`exec` and `eval`.** A `check()` compiled from a string is not a `Call`
  node. None is known here; this file cannot tell you that.
* **Qt slots.** A method decorated `@pyqtSlot` is invoked by the event loop
  through C++, so "is this site reached" is a question about a signal
  connection this walk does not follow. `workbench_interaction_check.py` drives
  the real window through `QTest`, and most of its sites are inside `main()`, but
  a slot body is not counted as guarded even though it plainly may not run.
* **Function frames are deliberately not guards.** A `check` inside a helper
  that `main()` always calls does run every time; counting `def` as a guard
  would label most of `core_check.py` conditional and make the column useless.
  The cost is stated rather than hidden: a `check` inside a helper that **is
  never called** is invisible to this file, and catching that needs a call
  graph, not a walk.
* **Statements, not invocations.** A `for` over a 20-row fixture counts once
  here and twenty times at run time. The right granularity for "could this site
  stop running" is not "how many results will the run print".
* **Only calls that are the whole statement.** `r = check(...)` and
  `return check(...)` are real result sites this walk does not count, the same
  blindness `check_scripts_declare.py` declares at `_sites_of`. This file counts
  any statement whose expression is a call to a check-like name, so a call bound
  to a name first is outside it.

**The probe can fail, and that is the only thing that makes the rest a result.**
Two of the ten categories -- `try/reraises` and `try/silent` -- match **no site
in this tree**, and `unclassified` is red by definition. A table whose rows are
all unexercised is a table that cannot be wrong and therefore cannot be right.
`probe_categories` builds one small file per category in a private sandbox under
`target/` and asserts this file's own classifier reports that category for it;
`probe_directions` then does the four mutations the census has to catch. The
classifier under test is **this file's own code**, executed with `__name__` set
away from `"__main__"` and its `SCRIPTS` global pointed at the sandbox, so what
is exercised is the code that runs, not a second copy of the rules.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

# This gate's own output is UTF-8 whatever the console's codepage is, for the
# reason every other gate in scripts/ gives: a gate whose output cannot be
# decoded is a gate that only works for the person who remembered.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SELF = Path(__file__).name
AUDITOR = SCRIPTS / "check_scripts_declare.py"

#: The file this gate was written about, reported on by name. Not the only gate
#: here; the one whose guarded sites are worth reading out loud.
HEADLINE = "workbench_interaction_check.py"

#: The names a check-like call is spelled with across this tree.
#:
#: **The first four are a hand-typed copy of the tuple inside
#: `check_scripts_declare.py`'s `_sites_of`, and this comment used to claim the
#: opposite.** It said they "are lifted from it rather than typed, so the two
#: files cannot disagree about what a check site is". Nothing lifted them --
#: `CALLS` below is a literal -- so the guarantee was false, and it was false in
#: the one place a reader goes to decide whether two censuses of the same file
#: are counting the same sites. A stale comment is a nuisance; a false claim
#: about census provenance is a confidently wrong census. The four are still the
#: four; the guarantee is not, so what is written here is a fact a reader can
#: check rather than a mechanism that does not exist.
#:
#: **The last two are this tree's two other spellings -- a `skip` that records
#: a result and a `pixel_check` that routes to one -- and the two censuses
#: therefore have different site universes.** Measured on
#: `unreached_product_check.py`, which both files census: this walk reports
#: `22 unconditional + 4 guarded` over 26 sites, the declarer's `_sites_of`
#: reports `22 unconditional + 3 guarded` over 25. The extra site is the `skip`
#: under the same `if ok_all:` as a `check`; every site the two share is
#: classified identically, and the whole difference is whether that `skip` is a
#: site at all.
#:
#: **The two numbers are not supposed to meet, and forcing them to would mean
#: deleting a site that exists.** They answer different questions. The
#: GATE-DECLARE number is a drift contract on a declared census, read by
#: `_declaration_of` and compared against one shared walk; it is a statement
#: about a declaration, not a coverage claim. This number feeds
#: `ENV_REMOVALS` and the category table, so its question is which sites can
#: fail to record and what each guard does to the result. A `skip` records a
#: result, so a leak detector that could not see it would be blind to exactly
#: the sites it exists to find.
#:
#: **The matching unconditional column is a coincidence and is not evidence the
#: two definitions agree.** It matches only because the one extra site happens
#: to be guarded. Moving that `skip` out of its `if` and leaving everything else
#: alone was measured, not guessed: the columns then read 22 and 23. So nothing
#: enforces the agreement, it must not be cited as the two definitions
#: matching, and a future unconditional `skip` would move one column and leave
#: the other where it is with no gate going red. Two further differences are
#: latent there and neither is a bug: the declarer drops
#: `if __name__ == "__main__"` and `if True` guards, which this walk counts as
#: guards, and this walk's `with` arm in `classify_file` models a context
#: manager as a guard where the declarer does
#: not. That last one is the widest gap in the tree -- 280 sites -- and it is
#: why `cli_check.py` reads 82 unconditional in the declarer's census and 0
#: here. Fixing it means editing the declarer's guard set, which is not this
#: file's to change.
CALLS = ("check", "ok", "bad", "expect", "skip", "pixel_check")

#: A statement that cannot fall through to the next statement. The `if` category
#: `jumps-to-a-result` is built on exactly this set, so it is named once here
#: rather than spelled at the one place that uses it.
JUMPS = (ast.Return, ast.Raise, ast.Continue, ast.Break)

#: Methods whose answer is a fact about the filesystem rather than about the
#: code under test. A guard that asks one of these is asking the machine, and it
#: is detected **structurally** -- no declaration needed -- because there is no
#: reading of `path.is_file()` that is about the program.
FS_ATTRS = ("is_file", "is_dir", "exists", "is_symlink", "stat", "lstat",
            "is_absolute", "expanduser")

#: Constructs this walk models as guards. Anything else it meets on the way up
#: from a site is a guard it does not understand, and a site with one is
#: `unclassified` -- a red, not a skip.
GUARDING = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
            ast.AsyncWith)

#: Constructs this walk meets and deliberately does not count as guards. Listed
#: rather than assumed, because the difference between this list and a construct
#: that is simply not mentioned is the difference between a decision and a gap.
NON_GUARDING = ("FunctionDef", "AsyncFunctionDef", "ClassDef", "Lambda",
                "Module", "arguments", "keyword", "comprehension")

#: The classification. `(id, construct, what happens to the result)` -- and the
#: third column is the whole point, because a category whose effect is not
#: stated cannot be told apart from another one by anything but its name.
#:
#: **Distinguishable in effect, which is the requirement.** `if/registers-else`
#: and `if/jumps-to-a-result` both end with the site in the tally; they are
#: separate rows because they are separate *mechanisms*, and a rule that
#: recognises only one of them reports half the truth as all of it. `if/blank`
#: is the only `if` category that removes a result, and that is what makes it the
#: only one `ENV_REMOVALS` has to speak to.
#:
#: **`try/reraises` and `try/silent` match no site in this tree.** That is a
#: measurement, not an omission: every `try` in these files either has no
#: handler, or has one that records, and there is no third kind here. A rule
#: nobody has exercised is a rule nobody has checked, so both are built by
#: `probe_categories` on every run and a table that lost its only exercise goes
#: red.
CATEGORIES: dict[str, tuple[str, str]] = {
    "if/registers-else": (
        "if",
        "nothing is removed: the else arm records, so the result is in the "
        "tally either way, as a SKIP rather than a PASS",
    ),
    "if/jumps-to-a-result": (
        "if",
        "nothing is removed: the body ends in an unconditional jump, and the "
        "statements after it -- across a run of further jump-terminated ifs -- "
        "record, so exactly one result is registered on every path",
    ),
    "if/blank": (
        "if",
        "THE RESULT IS REMOVED: nothing records on the false path, so the tally "
        "is one smaller when the guard is false",
    ),
    "try/registers": (
        "try",
        "the un-run results are replaced by whatever the handler records, so "
        "the count changes by (the handler's count - the results after the "
        "raise) rather than by zero",
    ),
    "try/reraises": (
        "try",
        "the run ends: the exception leaves the handler, so there is no tally "
        "to be smaller",
    ),
    "try/silent": (
        "try",
        "THE RESULTS ARE REMOVED and the run continues: the handler swallows "
        "the exception and records nothing, so every result after the raise is "
        "gone with no trace but the smaller number",
    ),
    "try/unprotected": (
        "try",
        "the run ends: no handler, so an exception propagates and there is no "
        "tally to be smaller",
    ),
    "loop/fixed-iterations": (
        "for/while",
        "the count is a constant of the source: a literal iterable or range()",
    ),
    "loop/data-iterations": (
        "for/while",
        "the count is a function of the data, not of the machine",
    ),
    "with/cleanup": (
        "with",
        "the body always runs; only __exit__ can end the run, so nothing is "
        "removed and there is no tally to be smaller",
    ),
    "unclassified": (
        "anything else",
        "this walk met a construct it does not model, so it does not know what "
        "the guard does to the tally, and a guard whose effect is unknown is a "
        "red rather than a skip",
    ),
}

#: A category that removes a result. The set is *derived* from the third column
#: above by asking whether it begins with the word "THE", so widening a
#: category's effect to "removes" cannot be done by editing a set somewhere
#: else: the two would disagree and this one would be the stale copy.
REMOVING = tuple(k for k, v in CATEGORIES.items() if v[1].startswith("THE"))

#: Categories allowed to match no site in this tree, and each of them has to
#: earn it twice. **It must be unexercised** -- otherwise the name is a stale
#: excuse, the same failure `ENV_REMOVALS` is checked for and for the same
#: reason. **And it must have a fixture**, because that is the only place it is
#: ever executed: `try/reraises` and `try/silent` match nothing here, and
#: `unclassified` matches nothing *by definition*, since the whole point of that
#: category is that a site carrying it is a red. A category in neither list is a
#: red. That is what keeps the exemption from becoming a dumping ground: three
#: names, each with a buildable proof, and a fourth would have to be argued for
#: in this file rather than added quietly.
UNEXERCISED_BY_DESIGN: tuple[str, ...] = ("try/reraises", "try/silent",
                                          "unclassified")

# --------------------------------------------------------------------------
# Axis 2: guards that read the machine
# --------------------------------------------------------------------------

#: Per file, the names a guard test reads when it is asking about **the machine**
#: rather than about the code under test. The filesystem family is detected
#: structurally by `FS_ATTRS` and is deliberately absent from this table: there
#: is no reading of `path.is_file()` that is a statement about the program, so
#: declaring those names by hand would be 200 lines of transcription for a fact
#: the tree already carries in its own syntax.
#:
#: **What is in here is a claim, and it is checked from both ends.** A name
#: here that no guard test reads is a red (`check_env_names_are_read`), so the
#: table cannot accumulate entries for names that stopped mattering. And the
#: classification is only ever applied to a guard whose *false* arm removes a
#: result, so an over-broad entry here costs one extra declared row rather than
#: a wrong verdict.
#:
#: The distinction being drawn, in the two files where it is least obvious:
#: `workbench_interaction_check.py`'s `PIXELS_OK` and `GL_OK` are the machine
#: (a framebuffer read blank, an OpenGL context refused), while its `_own_n`,
#: `members` and `chroma_ratios` are not listed because they are row counts and
#: accumulators the file computed a few lines earlier -- a guard false because
#: the measurement found nothing is the measurement talking, and it is a result,
#: not an excuse.
ENV_NAMES: dict[str, tuple[str, ...]] = {
    "check_text_encoding.py": ("files", "missing", "opened", "unreadable"),
    "cli_prep_check.py": ("F",),
    "contacts_screenshot.py": ("path", "pix"),
    "extension_surface_check.py": ("data",),
    "gpu_cpu_parity_check.py": ("ligand", "maps"),
    "gpu_feature_check.py": ("ligand", "maps"),
    "installed_copy_check.py": ("inst",),
    "ligand_check.py": ("made", "ref", "src"),
    "odgui_launch_check.py": ("answer", "child", "raw", "report", "widget"),
    "pockets_screenshot.py": ("path", "pix"),
    "probe_painted_frame.py": ("report", "tag"),
    "published_set_floor_check.py": ("root",),
    "qt_gl_probe.py": ("got", "outcomes", "state"),
    "screenshot_frame_check.py": ("box", "missing", "r"),
    "viewport_framing_check.py": ("can_draw", "frame"),
    "wheel_payload_check.py": ("whl",),
    "workbench_interaction_check.py": ("GL_OK", "PIXELS_OK", "app", "renderer"),
    "workbench_smoke.py": ("qimage", "win"),
}

#: Every guard that reads the machine **and** removes the result when it is
#: false, with what compensates. One row per `(file, guard text)`, and the two
#: checks either side of this table are the reason it is a table and not a
#: paragraph: a live guard with no row is a red, and a row with no live guard is
#: a red.
#:
#: **The status word in each reason is the epistemic status of the reason, not a
#: verdict on the guard.** `read` means the surrounding code was read and the
#: compensation was seen; `text-only` means the guard expression was read and the
#: compensation was **inferred**, which is a weaker claim and is counted
#: separately in the output so a reader knows how much of the table rests on
#: more than the guard text. A table that hid that difference would be a table
#: of 6 uniform-looking rows, 5 of which nobody looked at properly.
#: The two rows this table used to carry for `check_text_encoding.py
#: | rule.is_file()` and `ligand_check.py | ref.exists()` are **gone, and that is
#: the point of the two-sided check**: both guards were found by this file, both
#: were real, both have been given a compensating `else` arm by their owners, and
#: neither removes a result from its pin any more. A row left behind would have
#: read exactly like a live one -- it would have said "this guard is a documented
#: removal" about a guard that now records what it found. **This table is shorter
#: because two of its rows were defects, not because the table was widened**, and
#: that sentence is the only honest reading of the deletion.
ENV_REMOVALS: dict[tuple[str, str], tuple[str, str]] = {
    ("gpu_cpu_parity_check.py", "maps is not None"): (        "text-only",
        "the enclosing else arm registers named skips for the same two claims, "
        "so a machine with no maps is reported as an absence rather than as a "
        "smaller total; read the else at the end of the fixture-building block, "
        "not this guard's own surroundings",
    ),
    ("gpu_feature_check.py", "maps is None or ligand is None"): (
        "text-only",
        "the true arm registers five named skips naming the missing fixtures, and "
        "the false arm is the whole rest of the run, so a machine that could not "
        "build its own fixtures reports five absences instead of a total that "
        "moved",
    ),
    ("odgui_launch_check.py", "answer.verdict == 'no-widget'"): (
        "read",
        "deliberate, and the file says why in the block itself: the no-widget "
        "evidence check used to call ok() and then fall through to the ok() "
        "below, so a no-widget machine recorded two entries and every other "
        "machine one, and the total was a function of the verdict. Folding the "
        "assertion into the check's own verdict made the count 4 on an ok "
        "machine and 5 on a no-widget one, measured both ways. The cost is that "
        "this guard removes a result with no skip, and the compensation is a "
        "measurement recorded in a comment rather than a statement in the tree",
    ),
    ("odgui_launch_check.py", "str(raw.get('gl')) not in reason"): (
        "read",
        "its body is a bad() followed by a return, and the ok() that follows the "
        "enclosing no-widget block records on every path that reaches it, so the "
        "claim survives as a SKIP-or-PASS under the other name. Only on a "
        "no-widget machine whose reason omits the raw GL does the extra bad() "
        "run, and then the return keeps it to one entry",
    ),
    ("provenance_appendix.py", "not _p.is_file()"): (
        "read",
        "the one site in this whole tree that records a result and then declines "
        "to count it: `skip(..., counts=False)`. The skip is printed, so the "
        "reader sees which prose file was absent, and the pin is unaffected on "
        "purpose, because the number of prose files is not a property of the "
        "gate. It is listed here because `counts=False` is a *declared* "
        "exclusion, not a syntax fact, and a reader of the tally alone cannot "
        "see it",
    ),
}

# --------------------------------------------------------------------------
# The four role facts, re-derived four different ways
# --------------------------------------------------------------------------

#: `(file, role, what this file derives independently, and how)`. Each `fact`
#: is computed by a reader that shares no code with the lifted `role_of`, which
#: searches source **text** for `run_script("name"`. Three of the four are
#: derived by walking a syntax tree; the fourth reads the `EXCLUSIONS` block as
#: text rather than as the dict, so a lift that picked up the wrong node would
#: disagree with it. The four files are not this file's, and they were not
#: chosen because they are easy -- they were chosen to cover four different
#: roles, and the fourth is the only one of the four where **no** structural
#: relation holds, so the exemption is the only thing that can be its role.
ROLE_FACTS: tuple[tuple[str, str, str], ...] = (
    ("benchmark_check.py", "gate",
     "it defines a check() of its own: an ast.FunctionDef named check"),
    ("interp_reachable_step_bound.py", "deliverable",
     "examples_check.py hands it to run_script as a call argument, found by "
     "walking for the Call rather than by searching source text"),
    ("secondary_render.py", "imported",
     "six siblings carry an ast.Import or ast.ImportFrom naming it, found by "
     "walking for the import statement"),
    ("pocket_benchmark.py", "exempt",
     "nothing structural holds: it defines no check(), declares no pin, no "
     "gate names it in a run_script call and no sibling imports it -- "
     "benchmark_check.py loads it through importlib, which is a call and not an "
     "import. The only thing that can make it exempt is the key, and the key is "
     "read here from the EXCLUSIONS **text**"),
)

#: GATE-DECLARE 1
#: sites: 12 unconditional + 0 guarded
#: guards: sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
#: Twelve, and none guarded, on purpose rather than by accident. Every check site
#: below is a flat statement in `main()`, and the helpers return `(verdict,
#: detail)` pairs instead of recording, so that no site in this file is one that
#: could stop running -- which is the defect this file is about, and it would be
#: absurd to commit it here. The cost is a real one: a guarded site in this file
#: would exercise the guarded path on a live tree instead of only in the probe,
#: and `_sites_of` in `check_scripts_declare.py` counts only a call that *is* the
#: whole statement, so a `check` inside a conditional would be a site its census
#: cannot see -- and the declared count it is compared against is a generated
#: region that only `--pin` may rewrite. Adding one would leave a red this file
#: has no permitted way to clear. The empty-string digest is sha256 of nothing,
#: which is what a file with no guards hashes to.
#: **The first version of this block said eight, and the census caught it.** Ten
#: was the number of `check(` calls in `main()`: the census, the category table
#: read from both ends, the environment inventory in both directions, the
#: declared machine-reading names, the headline file, the file set, the two
#: probes, and the tally. Two had been added without the number moving, which is
#: exactly the failure the pin exists to catch, caught here by the pin rather
#: than by a reader.
#:
#: **Ten -> twelve, and the two are the cross-auditor section.** Not a new axis
#: of measurement: the same walk, asked a second question. The reason they are
#: two checks and not one is that they can fail for unrelated reasons, and a
#: single check over both would report "the two auditors disagree" whether what
#: happened was a classification disagreement or a name that left a census.
#: The two are flat `check(...)` statements in `main()` like every other site
#: here, so the guarded column stays 0 and the digest does not move -- which is
#: the only reason this edit was available without `--pin`.
#: EXPECTED_CHECKS is measured from a run, not typed.
EXPECTED_CHECKS = 12

CHECKS = 0
FAILURES: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    """Record one result.

    A non-bool verdict is refused rather than coerced, matching the house
    `check()` in `check_verdict_bool.py` and `check_scripts_declare.py`: `check`
    decides with `if not ok`, which is a Python truth value, so a tuple whose
    first element is a good bool takes the passing branch whatever the bool
    inside it says. The refusal keeps the total reachable and puts the defect
    in the output.
    """
    global CHECKS
    CHECKS += 1
    if not isinstance(ok, bool):
        FAILURES.append(name)
        print("[FAIL] %s" % name)
        print("       this call passed %r (%s), which is %s, where a bool "
              "belongs. Refused rather than coerced."
              % (ok, type(ok).__name__, "truthy" if ok else "falsy"))
        if detail:
            print("       %s" % detail)
        return False
    if not ok:
        FAILURES.append(name)
    print("[%s] %s" % ("ok " if ok else "FAIL", name))
    if detail:
        print("       %s" % detail)
    return ok


def section(t: str) -> None:
    print("\n=== %s ===" % t)


def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


def unparse(node, limit: int = 74) -> str:
    """`ast.unparse`, truncated. A guard text is a key here, so it is compared
    at a fixed width and the table's keys are written at that width."""
    try:
        return ast.unparse(node)[:limit]
    except Exception:
        return "<unprintable>"


# --------------------------------------------------------------------------
# Lifting the real `role_of` out of the auditor
# --------------------------------------------------------------------------

#: The module-level names `role_of` and its five helpers close over. Each is
#: named here and its absence is a loud failure, because a lift that silently
#: dropped one would produce a `NameError` on the first file rather than a
#: wrong answer -- and `check_census` turns that into a red.
LIFT_FUNCS = ("read", "_declares_a_pin", "_defines_check_call", "_run_by_a_gate",
              "_imported_by_a_sibling", "_can_report_failure", "role_of")
LIFT_NAMES = ("ROOT", "SCRIPTS", "INVENTORY", "EXCLUSIONS")


#: The auditor's parse tree, cached on (size, mtime_ns). Cached because the
#: probes lift it fifteen times and it is 265 KB; re-read on every change,
#: because **another agent owns that file and may edit it while this one
#: runs**, and a cached tree would answer about a version of it that is no
#: longer on disk.
_AUDITOR_TREE: dict = {}


def _auditor_tree():
    st = AUDITOR.stat()
    key = (st.st_size, st.st_mtime_ns)
    if _AUDITOR_TREE.get("key") != key:
        _AUDITOR_TREE.clear()
        _AUDITOR_TREE["key"] = key
        _AUDITOR_TREE["tree"] = ast.parse(read(AUDITOR))
    return _AUDITOR_TREE["tree"]


def lift_role_of(scripts_dir: Path):
    """The auditor's own `role_of`, lifted by AST and executed.

    **Why not import it.** `check_scripts_declare.py` runs its entire audit at
    module scope and ends in `sys.exit(...)`, so an import runs a four-thousand
    line audit and raises `SystemExit` -- and a gate that catches that and then
    prints is a gate whose prints were skipped by a raise that came first.
    **Why not re-implement it.** Then this file and that one hold two
    classifications of "is this a gate", and nothing would notice them
    disagreeing. Lifting the real nodes is the third way: the functions are
    executed from the source that is on disk, unaltered, so the two files cannot
    drift, and the auditor's audit never runs.

    `scripts_dir` is rebound in the lifted namespace *after* the exec, which is
    what lets the same classifier run over a sandbox: the helpers read their
    `SCRIPTS` from their own globals, and those globals are this namespace.
    """
    tree = _auditor_tree()
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in LIFT_FUNCS:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in LIFT_NAMES for t in node.targets):
            body.append(node)
    ns = {"__name__": "guarded_sites_role_lift", "ast": ast, "Path": Path,
          "__file__": str(AUDITOR)}
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "lifted_from_check_scripts_declare", "exec"), ns)
    ns["SCRIPTS"] = scripts_dir
    return ns


# --------------------------------------------------------------------------
# Axis 1: the classifier
# --------------------------------------------------------------------------

def _records(stmts) -> bool:
    """Does any statement here call a check-like name, at any depth?"""
    for st in stmts:
        for n in ast.walk(st):
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) in CALLS:
                return True
    return False


def _ends_in_jump(stmts) -> bool:
    return bool(stmts) and isinstance(stmts[-1], JUMPS)


def _siblings(node, parent):
    for field in ("body", "orelse", "finalbody"):
        v = getattr(parent, field, None)
        if isinstance(v, list) and any(x is node for x in v):
            return v
    return None


def _if_effect(site, node, parent) -> str:
    """What the `if` does to the result of a site in its body.

    Three answers, and the third is the one that needed a rule of its own.
    `if G: bad(...); return` followed by `ok(...)` reads as a blank arm to any
    walk that only looks for an `else` -- and in `contacts_screenshot.py` and
    `pockets_screenshot.py` that reading is wrong three times over, because the
    three arms each end in `return` and the code after them records. **Exactly
    one result is registered on every path**: each true arm records and leaves,
    and the chain of arms that all fall through lands on the statement after
    the last of them, which records. The first version of this rule passed the
    `if` as its own parent and therefore never found the siblings, which made
    the category unreachable and left 47 environment-reading guards in the
    removal table that the code had already answered.
    """
    if _records(node.orelse):
        return "if/registers-else"
    if not _ends_in_jump(node.body):
        return "if/blank"
    sibs = _siblings(node, parent)
    if sibs is None:
        return "if/blank"
    i = sibs.index(node)
    j = i + 1
    while j < len(sibs) and isinstance(sibs[j], ast.If) and _ends_in_jump(sibs[j].body):
        j += 1
    return "if/jumps-to-a-result" if j < len(sibs) and _records(sibs[j:]) else "if/blank"


def _try_effect(node) -> str:
    if not node.handlers:
        return "try/unprotected"
    if any(isinstance(n, ast.Raise) for h in node.handlers for n in ast.walk(h)):
        return "try/reraises"
    if _records(node.handlers):
        return "try/registers"
    return "try/silent"


def _handler_effect(handler) -> str:
    """The effect of *this* `except` arm, which is the guard on a site inside it.

    Three answers, and they are the same three the `try` gets, because they are
    the same three effects: a handler that records leaves a result, one that
    re-raises ends the run, and one that swallows and says nothing removes the
    result and lets the run continue. The difference is that for a site in the
    `try` *body* the try's own effect is the right question, and for a site in a
    handler arm only that arm's effect is -- which is why a try with a
    recording handler and a silent one is not one category.
    """
    if any(isinstance(n, ast.Raise) for n in ast.walk(handler)):
        return "try/reraises"
    if _records([handler]):
        return "try/registers"
    return "try/silent"


def _loop_effect(node) -> str:
    if isinstance(node, ast.While):
        return "loop/data-iterations"
    it = getattr(node, "iter", None)
    if isinstance(it, (ast.Tuple, ast.List, ast.Set, ast.Constant)):
        return "loop/fixed-iterations"
    if isinstance(it, ast.Call) and getattr(it.func, "id", None) == "range":
        return "loop/fixed-iterations"
    return "loop/data-iterations"


def _with_effect() -> str:
    return "with/cleanup"


def classify_file(src: str, path_label: str):
    """(rows, problems) for one file's source. A row per check-like site.

    Every chain element carries the guard's **test node**, not just its text.
    That is what lets the environment axis be answered from the tree rather
    than from a regex over `ast.unparse`: a first version kept only the printed
    text and re-parsed it, which meant a guard longer than the 74-character key
    width could not be re-parsed at all and a filesystem probe past the cut was
    invisible. A question asked of the syntax tree is not a question about how
    wide a column is.

    A file that does not parse yields `(None, why)`. **A file that cannot be
    parsed is a red here, not a skip**: a gate whose source this file cannot
    read is a gate whose guarded sites are unmeasured, and unmeasured is the
    state this whole file exists to move out of.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return None, "%s does not parse: %s" % (path_label, exc)
    parent_of = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent_of[c] = n
    rows, problems = [], []
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.Expr) or not isinstance(child.value, ast.Call):
                continue
            if getattr(child.value.func, "id", None) not in CALLS:
                continue
            chain, cur, odd = [], child, []
            while cur in parent_of:
                p = parent_of[cur]
                if isinstance(p, ast.If):
                    chain.append((_if_effect(child, p, parent_of[p]),
                                  unparse(p.test), p.lineno, p.test))
                elif isinstance(p, (ast.For, ast.AsyncFor, ast.While)):
                    chain.append((_loop_effect(p),
                                  unparse(getattr(p, "target", None) or p.test, 52),
                                  p.lineno, None))
                elif isinstance(p, ast.Try):
                    chain.append((_try_effect(p), "try", p.lineno, None))
                elif isinstance(p, ast.ExceptHandler):
                    # A check inside `except:` runs only when the `try` raised,
                    # which makes the handler a guard in its own right and the
                    # sharpest one in this tree: 32 sites across 26 files sit in
                    # one. The effect is the *handler's*, not the try's, so it is
                    # read off this handler and filed under the same three
                    # category ids -- a category here names an effect, and two
                    # constructs producing the same effect are not two
                    # categories. Walking up past the handler to the `Try` and
                    # reading the try's effect instead would have been wrong in
                    # both directions: a `try` with a recording handler and a
                    # silent one does not have one effect, it has one per
                    # handler.
                    chain.append((_handler_effect(p),
                                  "except " + (unparse(p.type, 40) if p.type
                                               else "bare"),
                                  p.lineno, None))
                elif isinstance(p, (ast.With, ast.AsyncWith)):
                    chain.append((_with_effect(),
                                  unparse(p.items[0].context_expr, 52), p.lineno,
                                  p.items[0].context_expr))
                elif not isinstance(p, GUARDING) and type(p).__name__ not in NON_GUARDING:
                    odd.append(type(p).__name__)
                cur = p
            chain.reverse()
            rows.append({"line": child.value.lineno,
                         "call": child.value.func.id,
                         "chain": chain,
                         "odd": odd})
            if odd:
                problems.append("%s:%d sits inside %s, which this walk does not "
                                "model as a guard"
                                % (path_label, child.value.lineno, odd[0]))
    return rows, problems


def reads_the_machine(test, names) -> bool:
    """Is this guard asking about the machine?

    Two detectors, and only two. The structural one is an attribute whose name
    is in `FS_ATTRS`, which needs no declaration because `x.is_file()` has one
    meaning whatever the program under test does. The declared one is a name in
    this file's `ENV_NAMES` row, which is a claim a person made and is checked
    from both ends by `check_env_names_read`.
    """
    if test is None:
        return False
    for n in ast.walk(test):
        if isinstance(n, ast.Attribute) and n.attr in FS_ATTRS:
            return True
        if isinstance(n, ast.Name) and n.id in names:
            return True
    return False


def audit(scripts_dir: Path, categories=None, env_removals=None, env_names=None):
    """Everything the checks read, derived in one pass. No table is typed here.

    The three table arguments exist so the probe can hand this function a
    **mutated** table and watch the same code react. That is the whole reason
    they are parameters: a census whose only table is the module global cannot
    be tested in the direction that matters, because there is no way to remove a
    row from the running program except by editing the file on disk.
    """
    categories = CATEGORIES if categories is None else categories
    env_removals = ENV_REMOVALS if env_removals is None else env_removals
    env_names = ENV_NAMES if env_names is None else env_names
    try:
        ns = lift_role_of(scripts_dir)
        role_of = ns["role_of"]
    except Exception as exc:                      # noqa: BLE001 -- reported, not swallowed
        return {"lift_error": "%s: %s" % (type(exc).__name__, exc)}

    rows_by_file, roles, parse_errors, odd_sites = {}, {}, [], []
    total = unconditional = 0
    by_category = {k: 0 for k in categories}
    env_guards = {}
    read_env_names = {}
    for path in sorted(scripts_dir.glob("*.py")):
        try:
            role = role_of(path.name)
        except Exception as exc:                  # noqa: BLE001 -- a red, not a skip
            parse_errors.append("%s: role_of raised %s: %s"
                                % (path.name, type(exc).__name__, exc))
            continue
        roles[path.name] = role
        rows, problems = classify_file(read(path), path.name)
        if rows is None:
            parse_errors.append(problems)
            continue
        if problems:
            odd_sites.extend(problems)
        if not rows:
            continue
        rows_by_file[path.name] = {"role": role, "rows": rows}
        names = env_names.get(path.name, ())
        for r in rows:
            total += 1
            if r["odd"]:
                by_category["unclassified"] = by_category.get("unclassified", 0) + 1
                continue
            if not r["chain"]:
                unconditional += 1
                continue
            cat = r["chain"][0][0]
            by_category[cat] = by_category.get(cat, 0) + 1
            # The environment axis, at chain level rather than at the innermost
            # construct: a site is env-removed when *some* guard on the way to
            # it reads the machine and that guard's own effect removes the
            # result. Keyed on (file, guard text), because that is the unit a
            # reader argues about and the unit a declared row has to name.
            for cat, text, line, test in r["chain"]:
                key = "%s:%d if %s" % (path.name, line, text)
                slot = env_guards.setdefault(
                    key, {"file": path.name, "guard": text, "line": line,
                          "reads_machine": False, "removes": False, "sites": 0,
                          "declared": (path.name, text) in env_removals})
                slot["sites"] += 1
                slot["reads_machine"] = slot["reads_machine"] or \
                    reads_the_machine(test, names)
                slot["removes"] = slot["removes"] or cat in REMOVING
                for nm in names:
                    if re.search(r"\b%s\b" % re.escape(nm), text):
                        read_env_names[(path.name, nm)] = \
                            read_env_names.get((path.name, nm), 0) + 1

    # `(file, guard text)` tuples, not the printed keys: `ENV_REMOVALS` is keyed
    # on tuples, and comparing a tuple table against a list of display strings
    # finds nothing -- which is the shape of bug this line was written to fix,
    # because it made every row in the table look stale at once.
    env_removed = sorted((v["file"], v["guard"]) for v in env_guards.values()
                         if v["reads_machine"] and v["removes"])
    declared = sorted(env_removals)
    return {
        "roles": roles,
        "rows_by_file": rows_by_file,
        "total": total,
        "unconditional": unconditional,
        "guarded": total - unconditional,
        "by_category": by_category,
        "categories": categories,
        "env_guards": env_guards,
        "env_removed": env_removed,
        "env_needed": env_removed,
        "env_declared": declared,
        "env_removals": env_removals,
        "env_stale": [k for k in declared if k not in env_removed],
        "env_undeclared": [k for k in env_removed if k not in env_removals],
        "unused_categories": sorted(k for k, v in by_category.items() if not v),
        "read_env_names": read_env_names,
        "unread_env_names": sorted(
            "%s:%s" % (f, nm)
            for f, row in env_names.items() for nm in row
            if (f, nm) not in read_env_names),
        "parse_errors": parse_errors,
        "odd_sites": odd_sites,
    }


# --------------------------------------------------------------------------
# Check 1-5
# --------------------------------------------------------------------------

def _per_file_table(a) -> list:
    out = []
    for fname in sorted(a["rows_by_file"]):
        blob = a["rows_by_file"][fname]
        rows = blob["rows"]
        g = sum(1 for r in rows if r["chain"] or r["odd"])
        out.append((fname, blob["role"], len(rows), len(rows) - g, g))
    return out


def check_census(a) -> tuple:
    """Every in-scope file parses, and every site in it is classified."""
    if "lift_error" in a:
        return False, ("the file set could not be derived at all: %s. The "
                       "auditor's role_of was lifted by AST and something it "
                       "closes over was not among the lifted names, so no "
                       "census is possible and a smaller, hand-typed one would "
                       "be exactly the excuse this file was written to refuse"
                       % a["lift_error"])
    problems = list(a["parse_errors"]) + list(a["odd_sites"])
    by = a["by_category"]
    counts = "; ".join("%s %d" % (k, by.get(k, 0)) for k in a["categories"])
    files = _per_file_table(a)
    lines = ["%d file(s) carry a check-like call site, %d site(s) in all: %d "
             "unconditional, %d behind a guard. By innermost construct: %s."
             % (len(files), a["total"], a["unconditional"], a["guarded"], counts)]
    for fname, role, n, u, g in files:
        lines.append("  %-42s %-11s %4d site(s), %4d unconditional, %4d guarded"
                     % (fname, role, n, u, g))
    lines.append("Per file, every row, with no cap: the list above is the whole "
                 "census. Roles come from the lifted role_of, and a file is in "
                 "scope for holding a check-like site whatever its role -- six "
                 "of them are formally exempt and are listed with that role, "
                 "because a screenshot capture that records PASS/FAIL is "
                 "answering the same question a gate does.")
    detail = "\n       ".join(lines)
    if problems:
        return False, (detail + "\n       %d file(s) or site(s) this walk cannot "
                       "account for: %s" % (len(problems), problems))
    return True, detail


def check_categories_both_sides(a) -> tuple:
    """Every category in the table is exercised, or earns its exemption twice."""
    if "lift_error" in a:
        return False, "no census, so the category table cannot be read"
    by = a["by_category"]
    lines = []
    for k in a["categories"]:
        n = by.get(k, 0)
        mark = "" if n else ("exempt: " if k in UNEXERCISED_BY_DESIGN else "UNEXERCISED, ")
        lines.append("  %-24s %5d site(s)  %s%s"
                     % (k, n, mark, a["categories"][k][1][:92]))
    stale = [k for k in UNEXERCISED_BY_DESIGN if by.get(k, 0)]
    unbuilt = [k for k in UNEXERCISED_BY_DESIGN if k not in CATEGORY_FIXTURES]
    orphan = [k for k, n in sorted(by.items()) if not n and k not in UNEXERCISED_BY_DESIGN]
    unknown = [k for k in UNEXERCISED_BY_DESIGN if k not in a["categories"]]
    detail = ("%d categor(ies), read from both ends. The forward direction is "
              "the census above; the reverse one is this check, and it is why "
              "the table cannot rot: a category with no live site is a red, "
              "because a row describing a shape the tree no longer has is a "
              "claim about the past that reads exactly like a claim about the "
              "present. %d categor(ies) are exempt from that on two counts each "
              "-- unexercised here, and carrying a fixture that builds it -- and "
              "the fixtures are the only place any of them is ever executed.\n"
              "       %s"
              % (len(a["categories"]), len(UNEXERCISED_BY_DESIGN),
                 "\n       ".join(lines)))
    why = []
    if orphan:
        why.append("%d categor(ies) match no site and are not in "
                   "UNEXERCISED_BY_DESIGN: %s" % (len(orphan), orphan))
    if stale:
        why.append("%d name(s) in UNEXERCISED_BY_DESIGN match %d live site(s), "
                   "so the exemption is stale: %s"
                   % (len(stale), sum(by.get(k, 0) for k in stale), stale))
    if unbuilt:
        why.append("%d name(s) in UNEXERCISED_BY_DESIGN have no fixture, so "
                   "nothing executes them: %s" % (len(unbuilt), unbuilt))
    if unknown:
        why.append("%d name(s) in UNEXERCISED_BY_DESIGN are not categories at "
                   "all: %s" % (len(unknown), unknown))
    if why:
        return False, detail + "  " + "; ".join(why)
    return True, detail


def check_env_removals_declared(a) -> tuple:
    """No machine-reading guard removes a result without a declared reason."""
    if "lift_error" in a:
        return False, "no census, so the environment axis cannot be read"
    undeclared = a["env_undeclared"]
    machine = sorted(k for k, v in a["env_guards"].items() if v["reads_machine"])
    removing = sorted(k for k, v in a["env_guards"].items() if v["removes"])
    lines = ["%d guard(s) in this tree read the machine, and %d remove the "
             "result when they are false. Every guard that does both is in "
             "ENV_REMOVALS with what compensates, and the table holds %d row(s)."
             % (len(machine), len(removing), len(a["env_declared"]))]
    for k in machine:
        v = a["env_guards"][k]
        verdict = "DECLARED" if v["declared"] else "NOT DECLARED"
        lines.append("  %-6s %5d site(s)  %s" % (verdict, v["sites"], k))
    lines.append("The full set of machine-reading guards is above, not the "
                 "subset that needed a row: a guard that reads the machine and "
                 "keeps its result is the good case, and naming only the bad "
                 "ones would leave the reader unable to tell which is which.")
    detail = "\n       ".join(lines)
    if undeclared:
        return False, (detail + "\n       %d machine-reading guard(s) remove a "
                       "result and carry no row: %s. Either the guard is fine "
                       "and somebody has to say why, or it is the defect this "
                       "file exists for; silence is the one answer that is not "
                       "available" % (len(undeclared), undeclared))
    return True, detail


def check_env_removals_fresh(a) -> tuple:
    """Every declared removal still matches a guard that exists."""
    if "lift_error" in a:
        return False, "no census, so the environment table cannot be read"
    table = a["env_removals"]
    lines = []
    for k in a["env_declared"]:
        status, reason = table.get(k, ("?", "?"))
        lines.append("  %-8s %-46s %s"
                     % (status, "%s | if %s" % k, reason[:150]))
    detail = ("%d declared row(s), each read against the guard it names. The "
              "reverse direction is the point: a row whose guard no longer "
              "exists is a stale excuse, and a stale excuse is worse than no "
              "row at all, because in every output a reader sees it is "
              "indistinguishable from a live one. The `status` word is the "
              "epistemic status of the reason -- `read` means the surrounding "
              "code was read and the compensation was seen, `text-only` means "
              "the guard expression was read and the compensation was inferred, "
              "which is a weaker claim and is counted here so a reader knows "
              "how much of the table rests on more than the guard text.\n"
              "       %s" % (len(a["env_declared"]), "\n       ".join(lines)))
    stale = a["env_stale"]
    if stale:
        return False, (detail + "\n       %d row(s) name a guard that is no "
                       "longer there: %s" % (len(stale), stale))
    return True, detail


def check_env_names_read(a) -> tuple:
    """Every declared machine-reading name is read by some guard test."""
    if "lift_error" in a:
        return False, "no census, so ENV_NAMES cannot be read against the tree"
    unused = [n for n in a["unread_env_names"]]
    live = sorted(a["read_env_names"])
    detail = ("ENV_NAMES holds %d file row(s) covering %d name(s), and %d of "
              "them are read by at least one guard test on this run. The reverse "
              "direction is the check: a name here that no guard reads is a "
              "claim about a guard that has gone, and it would keep classifying "
              "a future site as machine-reading for a reason that no longer "
              "applies. The names in use: %s"
              % (len(ENV_NAMES), sum(len(v) for v in ENV_NAMES.values()),
                 len(live), ", ".join("%s:%s" % k for k in live) or "none"))
    if unused:
        return False, (detail + "\n       %d declared name(s) no guard test "
                       "reads: %s" % (len(unused), unused))
    return True, detail


def check_headline(a) -> tuple:
    """The file this gate was written about: per guard, and the answer."""
    if "lift_error" in a:
        return False, "no census, so the headline file cannot be read"
    blob = a["rows_by_file"].get(HEADLINE)
    if blob is None:
        return False, ("%s carries no check-like call site on this run, so "
                       "there is nothing to report about its guards. Named "
                       "rather than skipped: this is the file the whole census "
                       "was built to speak about" % HEADLINE)
    rows = blob["rows"]
    g = [r for r in rows if r["chain"] or r["odd"]]
    names = ENV_NAMES.get(HEADLINE, ())
    by_cat = {}
    for r in rows:
        if r["odd"]:
            by_cat["unclassified"] = by_cat.get("unclassified", 0) + 1
        elif r["chain"]:
            by_cat[r["chain"][0][0]] = by_cat.get(r["chain"][0][0], 0) + 1
    seen = {}
    for r in rows:
        for cat, text, line, _test in r["chain"]:
            k = (cat, text, line)
            seen[k] = seen.get(k, 0) + 1
    machine = {k: v for k, v in a["env_guards"].items()
               if v["file"] == HEADLINE and v["reads_machine"]}
    bad = sorted(k for k, v in machine.items() if v["removes"])
    lines = ["%s: %d site(s), %d unconditional, %d behind %d distinct guard(s). "
             "Innermost construct: %s."
             % (HEADLINE, len(rows), len(rows) - len(g), len(g), len(seen),
                "; ".join("%s %d" % kv for kv in sorted(by_cat.items())))]
    lines.append("Every guard that reads the machine, with what it does to the "
                 "result when it is false -- the headless question, per guard:")
    for k in sorted(machine):
        v = machine[k]
        lines.append("  %-7s %-46s %5d site(s)"
                     % ("KEEPS" if not v["removes"] else "REMOVES",
                        k.split(" if ", 1)[-1], v["sites"]))
    # The guards that remove a result whether or not they read the machine, so
    # the "no" above is a measurement over the whole file and not a statement
    # about the machine-reading subset only. A reader who saw twelve KEEPS and
    # no list of the four blanks would be entitled to suspect the blanks are
    # hidden rather than answered.
    blanks = sorted(k for k, v in a["env_guards"].items()
                    if v["file"] == HEADLINE and v["removes"])
    lines.append("Every guard in this file that removes a result, machine-reading "
                 "or not -- %d of them:" % len(blanks))
    for k in blanks:
        v = a["env_guards"][k]
        lines.append("  %-46s %5d site(s)  reads the machine: %s"
                     % (k.split(" if ", 1)[-1], v["sites"],
                        "yes" if v["reads_machine"] else "no"))
    lines.append("The four are decided by values the file computed a few lines "
                 "earlier -- a contact list, a chroma table keyed by "
                 "representation, a row count, and a pixel mask that came back "
                 "empty -- so a false guard is the measurement talking rather "
                 "than the machine declining, and in each of the four the block "
                 "registers a sibling result on the other path. That is the "
                 "whole basis of the verdict below, and it is a reading of the "
                 "code rather than a run: nothing here executed a window.")
    detail = "\n       ".join(lines)
    if bad:
        return False, (detail + "\n       %d machine-reading guard(s) in this "
                       "file remove a result with nothing else in the file "
                       "failing: %s" % (len(bad), bad))
    return True, detail


# --------------------------------------------------------------------------
# Check 6: the file set
# --------------------------------------------------------------------------

def _exclusions_keys_from_text() -> set:
    """The `EXCLUSIONS` keys, read from the source **text**.

    A different reader from the lift, which evaluates the dict node. That is
    deliberate: if the lift picked up the wrong assignment, or a second
    assignment shadowed it, the two disagree here and the disagreement is the
    finding.
    """
    src = read(AUDITOR)
    try:
        block = src[src.index("EXCLUSIONS = {"):]
        block = block[:block.index("\n}\n")]
    except ValueError:
        return set()
    return set(re.findall(r'^\s{4}"([^"]+)":', block, re.M))


def check_role_agreement(a) -> tuple:
    """The lifted `role_of`, against four facts derived four other ways."""
    if "lift_error" in a:
        return False, "the file set could not be derived, so nothing to agree with"
    got, why = [], []
    for fname, want, how in ROLE_FACTS:
        mine = derive_role_fact(fname, want)
        if not mine:
            why.append("%s: this file could not re-derive the fact (%s)" % (fname, how))
            continue
        have = a["roles"].get(fname)
        if have == want:
            got.append("%s = %s" % (fname, want))
        else:
            why.append("%s: role_of says %r, the independent derivation says %r "
                       "(%s)" % (fname, have, want, how))
    detail = ("The file set is derived, never typed: `role_of` is lifted out of "
              "the source of %s by AST and executed, because that file has no "
              "`if __name__` guard and ends in sys.exit(), so importing it runs "
              "its whole audit, and because re-implementing it would leave two "
              "classifications free to disagree. %d of %d role(s) agreed with a "
              "reader that shares no code with it: %s. The four cover four "
              "different roles and four files this file did not write, and the "
              "fourth is the only one where no structural relation holds at all, "
              "so the exemption is the only thing that can be its role."
              % (AUDITOR.name, len(got), len(ROLE_FACTS), ", ".join(got)))
    if why:
        return False, detail + " Disagreements: " + "; ".join(why)
    return True, detail


def derive_role_fact(fname: str, want: str) -> bool:
    """Re-derive one role from the tree, for the four files above."""
    src = read(SCRIPTS / fname)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    if want == "gate":
        return any(isinstance(n, ast.FunctionDef) and n.name == "check"
                   for n in ast.walk(tree))
    if want == "deliverable":
        for other in sorted(SCRIPTS.glob("*.py")):
            try:
                t2 = ast.parse(read(other))
            except SyntaxError:
                continue
            for n in ast.walk(t2):
                if (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run_script"
                        and any(isinstance(x, ast.Constant) and x.value == fname
                                for x in n.args)):
                    return True
        return False
    if want == "imported":
        stem = fname[:-3] if fname.endswith(".py") else fname
        for other in sorted(SCRIPTS.glob("*.py")):
            if other.name == fname:
                continue
            try:
                t2 = ast.parse(read(other))
            except SyntaxError:
                continue
            for n in ast.walk(t2):
                if isinstance(n, ast.Import) and any(
                        x.name.split(".")[0] == stem for x in n.names):
                    return True
                if isinstance(n, ast.ImportFrom) and n.module and \
                        n.module.split(".")[0] == stem:
                    return True
        return False
    if want == "exempt":
        keys = _exclusions_keys_from_text()
        if fname not in keys:
            return False
        # The exemption has to be the *only* thing that can be holding the role.
        if any(isinstance(n, ast.FunctionDef) and n.name == "check"
               for n in ast.walk(tree)):
            return False
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(x, ast.Name) and x.id == "EXPECTED_CHECKS"
                    for x in node.targets):
                return False
        for other in sorted(SCRIPTS.glob("*.py")):
            try:
                t2 = ast.parse(read(other))
            except SyntaxError:
                continue
            for n in ast.walk(t2):
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run_script" \
                        and any(isinstance(x, ast.Constant) and x.value == fname
                                for x in n.args):
                    return False
                if isinstance(n, ast.Import) and any(
                        x.name.split(".")[0] == fname[:-3] for x in n.names):
                    return False
                if isinstance(n, ast.ImportFrom) and n.module and \
                        n.module.split(".")[0] == fname[:-3]:
                    return False
        return True
    return False


# --------------------------------------------------------------------------
# Check 7: the probes
# --------------------------------------------------------------------------

#: One small gate per category, each written to exercise exactly that rule. The
#: ``unclassified`` entry is a `match` statement: a case body runs only on a
#: match, this walk does not model `match_case` as a guard, and a site inside
#: one is exactly the "a guard this walk cannot read" case. It is the shape a
#: future `match`-guarded check site would take, and nothing in this tree has
#: one yet.
CATEGORY_FIXTURES: dict[str, str] = {
    "if/registers-else": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    guard = True\n"
        "    if guard:\n"
        "        check('a', True)\n"
        "    else:\n"
        "        check('a', False)\n"
        "    return 0\n"
    ),
    "if/jumps-to-a-result": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    guard = True\n"
        "    if guard:\n"
        "        check('a', False)\n"
        "        return 0\n"
        "    check('a', True)\n"
        "    return 0\n"
    ),
    "if/blank": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    guard = True\n"
        "    if guard:\n"
        "        check('a', True)\n"
        "    return 0\n"
    ),
    "try/registers": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    try:\n"
        "        check('a', True)\n"
        "    except Exception:\n"
        "        check('a', False)\n"
        "    return 0\n"
    ),
    "try/reraises": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    try:\n"
        "        check('a', True)\n"
        "    except Exception:\n"
        "        raise\n"
        "    return 0\n"
    ),
    "try/silent": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    try:\n"
        "        check('a', True)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return 0\n"
    ),
    "try/unprotected": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    try:\n"
        "        check('a', True)\n"
        "    finally:\n"
        "        pass\n"
        "    return 0\n"
    ),
    "loop/fixed-iterations": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    for i in (1, 2, 3):\n"
        "        check('a', True)\n"
        "    return 0\n"
    ),
    "loop/data-iterations": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    rows = [1, 2]\n"
        "    for i in rows:\n"
        "        check('a', True)\n"
        "    return 0\n"
    ),
    "with/cleanup": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    import contextlib\n"
        "    with contextlib.nullcontext():\n"
        "        check('a', True)\n"
        "    return 0\n"
    ),
    "unclassified": (
        "def check(name, ok, detail=''):\n"
        "    return bool(ok)\n"
        "def main():\n"
        "    probe = 1\n"
        "    match probe:\n"
        "        case 1:\n"
        "            check('a', True)\n"
        "    return 0\n"
    ),
}

#: A file whose only interesting site is a machine-reading guard with a blank
#: arm. The both-direction mutations act on `ENV_REMOVALS` around this.
ENV_FIXTURE = (
    "from pathlib import Path\n"
    "def check(name, ok, detail=''):\n"
    "    return bool(ok)\n"
    "def main():\n"
    "    lib = Path('lib')\n"
    "    if not lib.is_file():\n"
    "        check('the library was built', True)\n"
    "    return 0\n"
)
ENV_FIXTURE_KEY = ("probe_env_guard.py", "not lib.is_file()")


#: A monotonic counter for sandbox names. It exists because the first version
#: named a sandbox `<tag>-<pid>-<nanoseconds>` and that is **not unique**: a pid
#: is reused by the OS, and a nanosecond residue is reused inside one process,
#: so a leftover directory from an earlier run with the same pid made
#: `mkdir` raise `FileExistsError` and killed the run at check 8. The counter
#: plus `exist_ok=False` and a retry loop cannot collide with anything, because
#: a collision is detected rather than assumed away.
_SANDBOX_SEQ = 0


def _sandbox(tag: str) -> Path:
    """A fresh, unique directory under `target/`, one level deeper than it.

    Three rules are folded in here. **A fresh unique directory every run**, never
    a cleared one: `shutil.rmtree` raises `PermissionError` on Windows over a
    tree with locked `.pyc` files, and the workaround that "works" is to leave
    the directory. **One level deeper than `target/`**, because
    `target/CACHEDIR.TAG` exists and root recovery reaches one level past the
    true root, so a sandbox sitting directly under `target/` reads as entirely
    non-source. **Only `.py` files are written into it and none of them is ever
    imported**, so there is nothing in it for Windows to hold a lock on.
    """
    global _SANDBOX_SEQ
    base = ROOT / "target" / "guarded_sites_probe"
    base.mkdir(parents=True, exist_ok=True)
    while True:
        _SANDBOX_SEQ += 1
        d = base / ("%s-%d-%d" % (tag, os.getpid(), _SANDBOX_SEQ))
        try:
            d.mkdir(parents=True)
        except FileExistsError:
            continue
        return d


def _drop_sandbox(d: Path) -> str:
    """Empty a sandbox this run created, and say so if it could not.

    Fifteen directories per run, left behind, is a scratch pile in a tree where
    six other agents are working; deleting only what this file created is the
    rule and the reason this exists. **`shutil.rmtree` is not used**: it is the
    call that raises `PermissionError` on a Windows tree holding a `.pyc`, and
    the sandboxes hold three or four `.py` files and nothing else, so each is
    unlinked by name and the directory removed. A failure here is **returned**,
    not swallowed, and is printed by the probe that hit it -- a sandbox that
    could not be cleaned is a fact about the run, not a detail to lose.
    """
    try:
        for f in sorted(d.iterdir()):
            f.unlink()
        d.rmdir()
        return ""
    except OSError as exc:
        return "%s could not be emptied (%s); it is still there" % (d, exc)


def _self_namespace():
    """This file's own code, executed with `main()` switched off.

    The classifier under test is **this file's code**, not a second copy of the
    rules written for the probe. A copy would be a control nobody reads, and the
    two could disagree in exactly the direction the probe exists to detect.
    """
    ns = {"__name__": "guarded_sites_probe", "ast": ast, "os": os, "re": re,
          "sys": sys, "Path": Path,
          "__file__": str(Path(__file__).resolve())}
    exec(compile(Path(__file__).read_bytes(), SELF, "exec"), ns)
    return ns


def _write_sandbox(d: Path, name: str, body: str) -> None:
    (d / name).write_bytes(body.encode("ascii"))


def probe_categories() -> tuple:
    """Every category in the table is reachable, proven by building it.

    Two of the ten match no site in this tree, and `unclassified` is red by
    definition, so without this the table would carry three rows that nothing
    has ever executed -- a rule nobody has exercised is a rule nobody has
    checked, and the failure mode is a green here over a classifier that stopped
    resolving. Each fixture is written to a private sandbox and put through
    **this file's own** `audit`.
    """
    ns = _self_namespace()
    problems, ran, left = [], [], []
    for cat, body in CATEGORY_FIXTURES.items():
        d = _sandbox("cat")
        try:
            _write_sandbox(d, "probe_cat_gate.py", body)
            got = ns["audit"](d, env_removals={}, env_names={})
            # The assertion is that **every guarded site in the fixture** fell in
            # the intended category and in no other. Not that the count is one,
            # and not that every site is guarded: three of these fixtures hold
            # two sites on purpose -- `if/registers-else` has one call per arm,
            # and `if/jumps-to-a-result` has one before the jump and one *after*
            # it, which is correctly unconditional. The first version of this
            # probe asserted one of each, so it reported three working
            # categories as broken and one working category as half-working.
            spread = {k: v for k, v in got.get("by_category", {}).items() if v}
            if "lift_error" in got:
                problems.append("%s: the audit could not run in the sandbox (%s)"
                                % (cat, got["lift_error"]))
            elif spread and list(spread) == [cat]:
                ran.append("%s %d" % (cat, spread[cat]))
            else:
                problems.append("%s: the fixture's %d site(s) came out as %s"
                                % (cat, got.get("total", -1), spread))
        finally:
            note = _drop_sandbox(d)
            if note:
                left.append(note)
    detail = ("%d fixture(s) written to a private sandbox under target/ and put "
              "through this file's own audit(), which is the code that runs and "
              "not a copy of it. %s. try/reraises and try/silent match no site "
              "in this tree and unclassified must match none, so those three "
              "rows have no live exercise anywhere else; these fixtures are the "
              "exercise, and a rule that stopped resolving would leave them "
              "reporting zero. Every sandbox is emptied again before the next "
              "one is built, and anything that could not be emptied is named "
              "below rather than left for someone else to find: %s"
              % (len(CATEGORY_FIXTURES), "; ".join(ran),
                 "; ".join(left) if left else "none left behind"))
    if problems:
        return False, detail + " Unreachable or mis-classified: " + "; ".join(problems)
    if left:
        return False, detail + " A sandbox could not be emptied, so this run left scratch in target/."
    return True, detail


def probe_directions() -> tuple:
    """The census can be made to fail, in each of the four directions.

    (a) a one-file sandbox with a guard this file classifies: green.
    (b) a new guarded site whose guard this walk cannot read: red.
    (c) a required `ENV_REMOVALS` row deleted: red.
    (d) a row for a guard that no longer exists: red.

    Each mutation is applied to the **tables**, not to the code, and the same
    `audit()` is asked again. That is why `audit` takes them as arguments: a
    table reachable only as a module global cannot have a row removed from a
    running program, and a rule that can only be tested by editing the file on
    disk is a rule that is tested once and then trusted.
    """
    ns = _self_namespace()
    audit = ns["audit"]
    env_rows = {k: v for k, v in ns["ENV_REMOVALS"].items()
                if k[0].startswith("probe_")}
    steps, problems, left = [], [], []

    # (a) green as shipped.
    d = _sandbox("dir-a")
    try:
        _write_sandbox(d, "probe_a_gate.py", CATEGORY_FIXTURES["if/registers-else"])
        a = audit(d, env_removals={}, env_names={})
        ok_a = (not a.get("parse_errors") and not a.get("odd_sites")
                and a.get("env_undeclared") == [] and a.get("env_stale") == [])
        steps.append("(a) clean sandbox: %d site(s), 0 unclassified, 0 undeclared, "
                     "0 stale -> %s"
                     % (a.get("total", -1), "green" if ok_a else "RED"))
        if not ok_a:
            problems.append("(a) a clean sandbox was not green: parse=%s odd=%s "
                            "undeclared=%s stale=%s"
                            % (a.get("parse_errors"), a.get("odd_sites"),
                               a.get("env_undeclared"), a.get("env_stale")))
    finally:
        note = _drop_sandbox(d)
        if note:
            left.append(note)

    # (b) a new guarded site whose guard this walk cannot read.
    d = _sandbox("dir-b")
    try:
        _write_sandbox(d, "probe_b_gate.py", CATEGORY_FIXTURES["unclassified"])
        b = audit(d, env_removals={}, env_names={})
        ok_b = b["by_category"].get("unclassified") == 1 and len(b.get("odd_sites", [])) == 1
        steps.append("(b) one site inside a `match` case: %d unclassified site(s), "
                     "%d file-level complaint(s) -> %s"
                     % (b["by_category"].get("unclassified", 0),
                        len(b.get("odd_sites", [])),
                        "red" if ok_b else "GREEN (no teeth)"))
        if not ok_b:
            problems.append("(b) a site inside a match case was not reported as "
                            "unclassified, so an unmodelled guard would pass here")
    finally:
        note = _drop_sandbox(d)
        if note:
            left.append(note)

    # (c) a required row deleted.
    d = _sandbox("dir-c")
    try:
        _write_sandbox(d, ENV_FIXTURE_KEY[0], ENV_FIXTURE)
        good = dict(env_rows)
        good[ENV_FIXTURE_KEY] = ("text-only", "the sandbox's own declared row")
        dropped = {k: v for k, v in good.items() if k != ENV_FIXTURE_KEY}
        c = audit(d, env_removals=dropped)
        ok_c = (ENV_FIXTURE_KEY in c["env_needed"]
                and ENV_FIXTURE_KEY in c["env_undeclared"])
        steps.append("(c) the ENV_REMOVALS row for `%s` deleted, the same sandbox "
                     "unchanged: it is reported undeclared -> %s"
                     % (ENV_FIXTURE_KEY[1], "red" if ok_c else "GREEN (no teeth)"))
        if not ok_c:
            problems.append("(c) deleting a required row did not turn the census red")

        # (d) a row for a guard that no longer exists.
        bogus = (ENV_FIXTURE_KEY[0], "not lib.is_a_directory()")
        added = dict(good)
        added[bogus] = ("text-only", "a row for a guard this file does not have")
        dres = audit(d, env_removals=added)
        ok_d = bogus in dres["env_stale"]
        steps.append("(d) a row added for `%s`, a guard the file does not have, the "
                     "sandbox otherwise untouched: it is reported stale -> %s"
                     % (bogus[1], "red" if ok_d else "GREEN (no teeth)"))
        if not ok_d:
            problems.append("(d) a row naming a guard that does not exist was not "
                            "reported stale")
    finally:
        note = _drop_sandbox(d)
        if note:
            left.append(note)

    detail = ("Four mutations, four private sandboxes, this file's own audit() "
              "asked again each time with the tables changed and the code "
              "unchanged. Sandboxes left behind: %s.\n       %s"
              % ("; ".join(left) if left else "none", "\n       ".join(steps)))
    if problems:
        return False, detail + "  " + "; ".join(problems)
    if left:
        return False, detail + " A sandbox could not be emptied, so this run left scratch in target/."
    return True, detail


# --------------------------------------------------------------------------
# Axis 3: the two auditors, cross-checked
# --------------------------------------------------------------------------

#: The functions lifted out of `check_scripts_declare.py` for the cross-check.
#: `_short` is here because `_sites_of` calls it, and a lift that left a called
#: helper behind raises `NameError` on the first guard rather than answering --
#: a loud failure, which is why it is listed rather than discovered.
CROSS_LIFT_FUNCS = ("_short", "_sites_of")

#: The two vocabularies this file has to keep apart, and what a reader is owed
#: when they disagree.
#:
#: `check_scripts_declare.py` and this file both walk every `scripts/*.py` and
#: both emit an unconditional column and a guarded column. They answer two
#: different questions and **the two numbers are not supposed to meet**: one is
#: a drift contract ("can this site stop running?"), the other feeds a leak
#: detector ("which sites might fail to record?"). What is supposed to hold is
#: narrower and is what the two checks below assert:
#:
#:   * **On the sites both of them see, the two must agree** -- or say which
#:     construct difference explains the disagreement, out loud and counted.
#:   * **The universes may differ, but only by a name set read off the code**,
#:     and never in the direction that would blind this file to a site the
#:     other one can see.
#:
#: **What is derived, and how.** Nothing in this section types a check-like
#: name, a call-site count, or a guard construct. The auditor's name set is the
#: tuple in its own filter expression, read out of its parse tree; the three
#: lines the tracer watches are the three statements where its walk makes its
#: own decisions; the constructs it treats as guards are the classes its own
#: `isinstance` tests name, told apart from the ones that guard nothing by
#: whether their body appends to `guards`. Change any of those in that file and
#: the comparison below moves, because it is running that file's code.

#: `(id, one line on what this section asserts)`, so a reader of the output can
#: find the claim rather than the number. Not a table of results.
CROSS_SECTIONS = (
    ("shared-sites", "sites both censuses see, and where they are classified "
                     "the same way"),
    ("deliberate-difference", "the names only one census counts, and the "
                              "constructs only one census models as a guard"),
)


def _call_name(node):
    """`isinstance` if that is what this is, `ast.If` if it is that."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _next_sibling(fn, node):
    """The statement after `node` in whatever body holds it."""
    for parent in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            lst = getattr(parent, field, None)
            if isinstance(lst, list):
                for i, x in enumerate(lst):
                    if x is node and i + 1 < len(lst):
                        return lst[i + 1]
    return None


def _declarer_constructs(fn):
    """The AST classes the auditor's own walk treats as a guard, read off it.

    **An `isinstance(p, X)` test counts as a guard exactly when its body
    appends to `guards`.** That is the whole rule, and it is what separates the
    four guards from the `ast.Module` test two lines below them, which only
    breaks out of the walk. A name in a table here would be a second copy of a
    decision that file has already made, and a second copy is the defect this
    whole file is about.
    """
    out: dict[str, bool] = {}
    for n in ast.walk(fn):
        t = n.test if isinstance(n, ast.If) else None
        if not (isinstance(t, ast.Call) and _call_name(t.func) == "isinstance"
                and isinstance(t.args[0], ast.Name) and t.args[0].id == "p"):
            continue
        arg = t.args[1]
        seq = arg.elts if isinstance(arg, ast.Tuple) else [arg]
        appends = any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                      and x.func.attr == "append" and isinstance(x.func.value, ast.Name)
                      and x.func.value.id == "guards"
                      for st in n.body for x in ast.walk(st))
        for a in seq:
            nm = _call_name(a)
            if nm:
                out[nm] = out.get(nm, False) or appends
    return out


def cross_lift():
    """The auditor's `_sites_of`, lifted by AST, with its own shape read off it.

    Same reasoning as `lift_role_of` and for the same reason: importing that
    file runs its whole audit and exits, and re-implementing the walk would put
    two readers of "what is a check site" in this tree, which is the thing the
    cross-check exists to catch rather than to add to.

    **Every locator here fails loudly.** A name, a guard, or a shape that is not
    found uniquely raises, and the two checks below turn that into a red --
    because the failure mode of a comparison that quietly compares nothing is a
    green gate, which is the one thing worse than a red one.
    """
    tree = _auditor_tree()
    body, fn = [], None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in CROSS_LIFT_FUNCS:
            body.append(node)
            if node.name == "_sites_of":
                fn = node
    if fn is None or len(body) != len(CROSS_LIFT_FUNCS):
        raise ValueError("the auditor no longer defines %s at module level, or "
                         "defines one of them twice" % (CROSS_LIFT_FUNCS,))
    filt, classify, raw = None, None, None
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare):
            ops, cmp_ = n.test.ops, n.test.comparators[0]
            if (len(ops) == 1 and type(ops[0]).__name__ == "NotIn"
                    and isinstance(cmp_, ast.Tuple)
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                            for e in cmp_.elts)):
                filt = (n, tuple(e.value for e in cmp_.elts))
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "guards":
            classify = n
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.ListComp)
                and any(isinstance(t, ast.Name) and t.id == "guards"
                        for t in n.targets)):
            raw = n
    if filt is None or classify is None or raw is None:
        raise ValueError("the auditor's _sites_of no longer has a name filter, "
                         "a `if guards:` decision or a guard exclusion this "
                         "could read; the three statements it watches are found "
                         "by shape, and the shape moved")
    accept = _next_sibling(fn, filt[0])
    if accept is None:
        raise ValueError("the name filter is the last statement in its body, so "
                         "there is no line after it to read the accepted site on")
    ns = {"__name__": "guarded_sites_census_lift", "ast": ast, "Path": Path,
          "__file__": str(AUDITOR)}
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "lifted_from_check_scripts_declare", "exec"), ns)
    return {"ns": ns, "calls": filt[1], "accept": accept.lineno,
            "classify": classify.lineno, "raw": raw.lineno,
            "constructs": _declarer_constructs(fn)}


def declarer_sites(src_text: str, lift) -> tuple:
    """The auditor's census of one file, one row per site, off its own loop.

    **Why a tracer at all.** `_sites_of` returns two numbers and a sorted list
    of guard strings. That is enough to compare totals and not enough to name a
    site, and "a disagreement somewhere" is not a report. So the lifted
    function is executed with a line tracer on the three lines where it makes
    its own decisions, and each row is read out of its frame at the moment it
    makes it: the call it accepted, the guards it had collected *before* it
    dropped the ones it excludes, and the list it was holding when it decided
    guarded-or-not. The name set is the auditor's own tuple, read off its own
    filter node, so nothing here spells a check-like name and a change to that
    tuple moves this comparison.

    **The `continue` on the filter's own line is why the watch is on a
    successor statement and not on "any line after the filter".** A rejected
    site fires a line event on the `continue` and one on the loop head, both of
    which sit below the filter, so watching "past the filter" collects the
    rejections too. The first version of this did that and over-counted 35
    sites in a file the walk had called zero; the self-check below is what
    turns that class of mistake into a red rather than a number.

    **The cost is real and is paid on purpose.** A line event fires for every
    statement the walk executes: about 20s over this tree against well under a
    second untraced, which is most of what this gate now costs. What it buys is
    `path:line` for both sides of every disagreement. Only the lifted frame is
    traced -- every other frame is released on arrival -- and `finally` clears
    the tracer whatever the walk does, so nothing else in the process is left
    instrumented.
    """
    want = {lift["accept"]: "site", lift["classify"]: "kept", lift["raw"]: "raw"}
    log = []
    def tracer(frame, event, arg):
        if frame.f_code.co_name != "_sites_of":
            return None
        if event == "line" and frame.f_lineno in want:
            g = frame.f_locals.get("guards")
            log.append((want[frame.f_lineno], frame.f_locals.get("child"),
                        list(g) if isinstance(g, list) else None))
        return tracer
    sys.settrace(tracer)
    try:
        uncond, guards = lift["ns"]["_sites_of"](src_text)
    finally:
        sys.settrace(None)
    rows, cur = [], None
    for what, child, g in log:
        if what == "site":
            cur = {"line": child.value.lineno, "call": child.value.func.id,
                   "raw": None, "guards": None}
            rows.append(cur)
        elif what == "raw" and cur is not None:
            cur["raw"] = g or []
        elif what == "kept" and cur is not None:
            cur["guards"] = g or []
    # **The trace is validated against the walk's own return value on every
    # file**, in all three columns. A tracer that read the wrong lines would
    # otherwise produce a confident, wrong, smaller number.
    problems = []
    if len(rows) != uncond + len(guards):
        problems.append("the trace read %d site(s) and the walk counted %d"
                        % (len(rows), uncond + len(guards)))
    if sum(1 for r in rows if r["guards"]) != len(guards):
        problems.append("the trace read %d guarded and the walk kept %d"
                        % (sum(1 for r in rows if r["guards"]), len(guards)))
    if any(r["raw"] is None for r in rows):
        problems.append("%d site(s) never reached the exclusion line, so the "
                        "pre-exclusion guard list was not read for them"
                        % sum(1 for r in rows if r["raw"] is None))
    return rows, (uncond, len(guards)), problems


def _construct_categories() -> dict:
    """Which categories each guard construct can produce, read off `CATEGORIES`.

    `CATEGORIES` names its construct in its **first** column -- "if", "try",
    "for/while", "with", "anything else", the second being the effect on the
    result -- and `GUARDING` names the AST classes this walk goes up through. A
    class is matched to the categories whose construct column contains its own
    name with a leading `Async` dropped, because `AsyncFor` is the async
    spelling of one guard and the table has no separate row for it.

    **Checked in both directions.** Every construct in `GUARDING` must be
    explained by at least one category (`check_cross_agreement` reds if one is
    not), so this map cannot quietly stop covering the walk's own model. It is
    the one mapping in this file that is written rather than lifted, and it maps
    two vocabularies rather than copying a number, which is the difference.
    Reading the wrong column of that table made all five shared constructs
    unmapped on the first run, and the check caught it as a red rather than as
    208 unexplained disagreements.
    """
    out: dict[str, tuple] = {}
    for cls in GUARDING:
        base = cls.__name__.lower()
        if base.startswith("async"):
            base = base[len("async"):]
        out[cls.__name__] = tuple(k for k, v in CATEGORIES.items()
                                  if base in re.findall(r"[a-z]+", v[0].lower()))
    return out


def cross_audit(scripts_dir: Path):
    """Both censuses of every file, side by side. No count in this is typed."""
    try:
        lift = cross_lift()
    except Exception as exc:                      # noqa: BLE001 -- a red, not a skip
        return {"lift_error": "%s: %s" % (type(exc).__name__, exc)}
    cat_of = _construct_categories()
    unmapped = sorted(c.__name__ for c in GUARDING if not cat_of[c.__name__])
    d_constructs = {k for k, v in lift["constructs"].items() if v}
    modelled = {c.__name__ for c in GUARDING}
    # Construct-only-here: this walk guards on it, the auditor does not model
    # it as a guard at all. Every classification disagreement has to be
    # explained by one of these, or it is a FAIL -- see below.
    only_here = sorted(modelled - d_constructs)
    # The other direction, and it is a red: a construct the auditor guards on
    # that this walk does not model would be a whole shape of site this file
    # cannot see, which is the defect this file exists to move out of.
    only_there = sorted(d_constructs - modelled)

    rows, tracer, unreadable = [], [], []
    seen_d, seen_g = set(), set()
    for path in sorted(scripts_dir.glob("*.py")):
        try:
            src_text = read(path)
        except Exception as exc:                  # noqa: BLE001 -- reported below
            unreadable.append("%s: %s" % (path.name, exc))
            continue
        try:
            drows, (uncond, nguard), problems = declarer_sites(src_text, lift)
        except SyntaxError as exc:
            unreadable.append("%s: does not parse (%s)" % (path.name, exc))
            continue
        if problems:
            tracer.extend("%s: %s" % (path.name, p) for p in problems)
        grows, gwhy = classify_file(src_text, path.name)
        if grows is None:
            unreadable.append(gwhy)
            continue
        dd = {(r["line"], r["call"]): r for r in drows}
        gg = {(r["line"], r["call"]): r for r in grows}
        for r in drows:
            seen_d.add(r["call"])
        for r in grows:
            seen_g.add(r["call"])
        both = sorted(set(dd) & set(gg))
        dis = []
        for key in both:
            dr, gr = dd[key], gg[key]
            d_guard = bool(dr["guards"])
            g_guard = bool(gr["chain"]) or bool(gr["odd"])
            if d_guard == g_guard:
                continue
            cats = [c for c, _, _, _ in gr["chain"]]
            # Two accountings, both read from the code rather than declared.
            # (1) the auditor had a guard and then dropped it, which is its
            #     own exclusion list doing what it says; (2) every construct
            #     this walk went up through is one the auditor does not model.
            # Anything else is a site the two readers disagree about for a
            # reason neither code path accounts for, and it is a FAIL.
            if dr["raw"] and not dr["guards"]:
                why = "the auditor dropped a guard test its own source excludes"
            elif cats and all(any(c in cat_of.get(k, ()) for k in only_here)
                             for c in cats):
                why = "guarded here on %s, not modelled there" % "/".join(
                    sorted({k for c in cats for k in only_here
                            if c in cat_of.get(k, ())}))
            else:
                why = None
            dis.append({"line": key[0], "call": key[1], "why": why,
                        "d_guards": dr["guards"] or [], "d_raw": dr["raw"] or [],
                        "g_cats": cats,
                        "g_text": " | ".join("%s %s" % (c, t) for c, t, _, _
                                             in gr["chain"]) or
                                  ("unclassified" if gr["odd"] else "-")})
        d_only = sorted(set(dd) - set(gg))
        g_only = sorted(set(gg) - set(dd))
        by_name: dict[str, int] = {}
        for key in g_only:
            by_name[gg[key]["call"]] = by_name.get(gg[key]["call"], 0) + 1
        rows.append({"file": path.name, "both": len(both), "d_only": len(d_only),
                     "g_only": len(g_only), "dis": dis, "d_uncond": uncond,
                     "d_guard": nguard,
                     "g_uncond": sum(1 for r in grows if not r["chain"] and not r["odd"]),
                     "g_guard": sum(1 for r in grows if r["chain"] or r["odd"]),
                     "g_only_by_name": by_name,
                     "d_only_sites": ["%s:%d" % (path.name, k[0]) for k in d_only]})
    return {
        "lift_error": None,
        "calls_declarer": lift["calls"], "calls_this": CALLS,
        "seen_declarer": sorted(seen_d), "seen_this": sorted(seen_g),
        "unmapped_constructs": unmapped,
        "only_here_constructs": only_here, "only_there_constructs": only_there,
        "rows": rows,
        "shared": sum(r["both"] for r in rows),
        "agreed": sum(r["both"] - len(r["dis"]) for r in rows),
        "disagreed": sum(len(r["dis"]) for r in rows),
        "unexplained": ["%s:%d" % (r["file"], d["line"])
                        for r in rows for d in r["dis"] if d["why"] is None],
        "d_only": sum(r["d_only"] for r in rows),
        "g_only": sum(r["g_only"] for r in rows),
        "g_only_by_name": {k: v for r in rows
                           for k, v in r["g_only_by_name"].items()},
        "d_only_sites": [s for r in rows for s in r["d_only_sites"]],
        "tracer_problems": tracer, "unreadable": unreadable,
    }


def check_cross_agreement(x) -> tuple:
    """The shared universe, and the equality asserted on it.

    **The claim: on every site both censuses see, the two readers classify the
    site the same way** -- unless one of the two accountings the code itself
    supplies explains the disagreement, and the explanation is printed. A
    disagreement with no accounting is a FAIL naming the site.

    **What the accounting is, and why it is not a licence.** It is narrow on
    purpose and both halves are read out of a code path rather than written
    here:

    * *the auditor dropped a guard test its own source excludes* -- measured by
      comparing the guards it had collected before its exclusion list with the
      ones it kept, both read off its own frame.
    * *guarded here on a construct the auditor does not model* -- measured
      against the set difference of the two code paths' own construct sets.

    The second half has a measured size today, and it is not small: it is every
    `with` site in the tree, a couple of hundred of them across five files,
    because the auditor's walk does not go up through a context manager. **The
    number is on the run, not here**, because a number written in a comment is
    a number that rots, and the whole reason this section exists is that two
    readers' numbers used to be able to rot apart silently. That is a real
    difference of instrument between the two files and it is **not** a
    disagreement about a site -- the two are answering two questions, and both
    files' headers say so. It is recorded as a number on every run rather than
    as a tolerance, so that it is one edit away from being visible, and so that
    if it ever grows for a reason nobody wrote down, the growth is a red rather
    than a slower gate.
    """
    if x.get("lift_error"):
        return False, ("the auditor's census could not be lifted, so nothing "
                       "was compared: %s" % x["lift_error"])
    per_file = ["       %-38s %5d shared, %d agree, %d differ, %d only here, "
                "%d only there" % (r["file"], r["both"],
                                   r["both"] - len(r["dis"]), len(r["dis"]),
                                   r["g_only"], r["d_only"])
                for r in x["rows"] if r["dis"] or r["g_only"] or r["d_only"]]
    lines = ["       %s" % ln for ln in per_file] or ["       (no file differs)"]
    lines.append("       constructs only this walk guards on: %s -- and every "
                 "disagreement below is on one of them"
                 % (x["only_here_constructs"] or "none"))
    lines.append("       constructs the auditor guards on that this walk does "
                 "not model: %s"
                 % (x["only_there_constructs"] or "none"))
    lines.append("       constructs in GUARDING no category can explain: %s"
                 % (x["unmapped_constructs"] or "none"))
    # Every disagreement is named. **One `path:line`, not two**: the two
    # censuses are talking about one site at one line, so there is one location
    # and two verdicts, and printing it twice would imply they were two sites.
    for r in x["rows"]:
        for d in r["dis"]:
            lines.append("       %s:%d %s  --  auditor: %s  |  this walk: %s  |  "
                         "because: %s"
                         % (r["file"], d["line"], d["call"],
                            " | ".join(d["d_guards"]) or "unconditional",
                            d["g_text"], d["why"]))
    detail = ("%d site(s) are in both censuses and %d of them are classified the "
              "same way by both. The two numbers this file and the auditor print "
              "are not meant to meet -- one is a drift contract on a declared "
              "census, this one feeds ENV_REMOVALS and the category table -- and "
              "both files' headers say so. What must hold is the narrower thing "
              "asserted here: on the shared universe the two readers do not "
              "contradict each other, and where they do, the accounting is "
              "printed rather than tolerated.\n       %d disagreement(s), every "
              "one named:\n%s\n       %s"
              % (x["shared"], x["agreed"], x["disagreed"], "\n".join(lines),
                 "UNEXPLAINED: %s -- a site the two readers are reading "
                 "differently, and nothing in either code path accounts for it"
                 % x["unexplained"] if x["unexplained"] else
                 "Every disagreement is accounted for by a difference the two "
                 "code paths state themselves, so none of them is a site the two "
                 "readers are reading differently."))
    ok = (not x["unexplained"] and not x["unmapped_constructs"]
          and not x["tracer_problems"] and not x["unreadable"])
    if not ok:
        detail += ("  PROBLEMS: unexplained=%s unmapped_constructs=%s "
                   "tracer=%s unreadable=%s"
                   % (x["unexplained"] or "none", x["unmapped_constructs"] or "none",
                      x["tracer_problems"] or "none", x["unreadable"] or "none"))
    return ok, detail


def check_cross_difference(x) -> tuple:
    """The deliberate difference, named, counted, and read off the code.

    **Two differences are allowed, and both are derived rather than declared.**

    * **The name sets.** The auditor counts four spellings; this file counts
      those four plus `skip` and `pixel_check`. The extra names are in *this*
      file's set and not the auditor's, and the sites they account for are
      counted from the walk rather than transcribed -- so editing either tuple
      moves this number, and a name dropped from this file's tuple turns the
      other direction red below instead of quietly shrinking the census.
    * **The construct sets.** This walk models `with` as a guard; the auditor
      does not go up through one. The difference is the set difference of the
      two files' own `GUARDING` and `isinstance` tests, not a list written here.

    **The direction is argued, and it is labelled as an argument.** The
    measurement -- how many sites the wider set sees -- is derived. The claim
    that the wider set is the right one for *this* file is a reading: this file
    asks which sites might fail to record a result, and a `skip` and a
    `pixel_check` both record one, so a census that could not see them would be
    blind to exactly the sites it exists to find. The auditor asks a different
    question -- can this site stop running, measured against a declared census
    -- and its four names are the right four for it. **A gate that printed that
    argument as a measurement would be the failure this file is about**, so it
    is printed as an argument and the measurement is printed beside it.
    """
    if x.get("lift_error"):
        return False, ("the auditor's name set could not be read, so no "
                       "difference is recorded: %s" % x["lift_error"])
    only_here = sorted(set(x["calls_this"]) - set(x["calls_declarer"]))
    only_there = sorted(set(x["calls_declarer"]) - set(x["calls_this"]))
    seen_only_here = sorted(set(x["seen_this"]) - set(x["seen_declarer"]))
    # A name in the auditor's set that this file does not count is a red: this
    # walk is a leak detector, and a whole spelling it cannot see is a hole in
    # the detector rather than a difference of opinion.
    detail = ("the auditor's filter, read out of its own parse tree, is %s; this "
              "file's CALLS is %s. Only this file counts: %s. Only the auditor "
              "counts: %s.\n       The wider set is this file's, and that is the "
              "right way round *for this file*, which asks which sites might "
              "fail to record a result -- and that sentence is a reading, not a "
              "measurement. The auditor asks whether a site can stop running, "
              "measured against a declared census, and its four names are the "
              "right four for that question. Neither number is meant to reach "
              "the other.\n       Measured: %d site(s) are in this census and not "
              "the auditor's, accounted for by %s; the names were seen at run "
              "time: %s. Each is a spelling that records a result, so a leak "
              "detector built on the narrower set would be blind to exactly the "
              "sites it exists to find.\n       Site counts, all derived: this "
              "walk %d unconditional / %d guarded over %d, the auditor %d "
              "unconditional / %d guarded over %d.\n       %d site(s) are in the "
              "auditor's census and not this file's: %s. %d construct(s) are "
              "guarded here and not there: %s; %d the other way: %s."
              % (x["calls_declarer"], x["calls_this"], only_here or "none",
                 only_there or "none", x["g_only"],
                 ", ".join("%s=%d" % (k, v)
                           for k, v in sorted(x["g_only_by_name"].items()))
                 or "no such site",
                 seen_only_here or "none",
                 sum(r["g_uncond"] for r in x["rows"]),
                 sum(r["g_guard"] for r in x["rows"]),
                 sum(r["g_uncond"] + r["g_guard"] for r in x["rows"]),
                 sum(r["d_uncond"] for r in x["rows"]),
                 sum(r["d_guard"] for r in x["rows"]),
                 sum(r["d_uncond"] + r["d_guard"] for r in x["rows"]),
                 x["d_only"], x["d_only_sites"][:20] or "none",
                 len(x["only_here_constructs"]), x["only_here_constructs"] or "none",
                 len(x["only_there_constructs"]), x["only_there_constructs"] or "none"))
    ok = not only_there and not x["d_only"] and not x["only_there_constructs"] \
        and not x["tracer_problems"] and not x["unreadable"]
    if not ok:
        detail += ("  PROBLEMS: the auditor counts %s and this file does not, "
                   "which is a hole in the detector rather than a difference of "
                   "opinion; %d site(s) it sees and this file does not (%s); it "
                   "guards on %s and this walk does not model them."
                   % (only_there or "none", x["d_only"],
                      x["d_only_sites"][:20] or "none",
                      x["only_there_constructs"] or "none"))
    return ok, detail


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    section("the census: every check-like site, and the guard on each")
    a = audit(SCRIPTS)
    # One binding per check, not a shared `ok` reused down the function. The
    # verdict-shape sweep resolves a `check(ok, ...)` site by finding the single
    # assignment its first argument came from, so a name written eleven times
    # leaves every one of those sites unreadable -- and an unreadable verdict is
    # a red against a cap of zero, not a stylistic note. The names below say
    # which check they carry, which is what made the collision findable.
    census_ok, census_detail = check_census(a)
    check(census_ok, "every in-scope file parses and every site in it is classified",
          census_detail)

    section("the category table, read from both ends")
    categories_ok, categories_detail = check_categories_both_sides(a)
    check(categories_ok,
          "every category in the table is exercised by a site that exists",
          categories_detail)

    section("the machine-reading guards, and the inventory of the ones that remove")
    removals_ok, removals_detail = check_env_removals_declared(a)
    check(removals_ok,
          "no machine-reading guard removes a result without a declared reason",
          removals_detail)
    fresh_ok, fresh_detail = check_env_removals_fresh(a)
    check(fresh_ok, "every declared removal still matches a guard that exists",
          fresh_detail)
    names_ok, names_detail = check_env_names_read(a)
    check(names_ok, "every declared machine-reading name is read by some guard",
          names_detail)

    section("%s: the headless question, per guard" % HEADLINE)
    headless_ok, headless_detail = check_headline(a)
    check(headless_ok,
          "no guard in the headline file can be false headless with nothing "
          "else failing", headless_detail)

    section("the file set")
    roles_ok, roles_detail = check_role_agreement(a)
    check(roles_ok, "the derived file set agrees with four independent derivations",
          roles_detail)

    section("the probes")
    probe_cats_ok, probe_cats_detail = probe_categories()
    check(probe_cats_ok,
          "every category in the table is reachable, proven by building it",
          probe_cats_detail)
    directions_ok, directions_detail = probe_directions()
    check(directions_ok,
          "the census can be made to fail, in all four directions",
          directions_detail)

    section("the two auditors, cross-checked (%s)"
            % ", ".join("%s: %s" % (i, w) for i, w in CROSS_SECTIONS))
    x = cross_audit(SCRIPTS)
    cross_ok, cross_detail = check_cross_agreement(x)
    check(cross_ok, "every site both censuses see is classified the same way by both",
          cross_detail)
    crossdiff_ok, crossdiff_detail = check_cross_difference(x)
    check(crossdiff_ok,
          "the two censuses differ by a name set read off the code, and "
          "only by one", crossdiff_detail)

    section("the tally")
    check(
        CHECKS + 1 == EXPECTED_CHECKS,
        "this file ran the number of checks it declares",
        "%d result(s) were recorded before this one, and this one is the %dth, "
        "against a declared %d. The count is taken before this check increments "
        "it, so the declared number is the number of checks a run produces and "
        "not the number it had got to -- a site that stopped running is a red "
        "here rather than a smaller file wearing the same pin"
        % (CHECKS, CHECKS + 1, EXPECTED_CHECKS),
    )
    print("")
    print("--- %d passed, %d failed, of %d declared"
          % (CHECKS - len(FAILURES), len(FAILURES), EXPECTED_CHECKS))
    if FAILURES:
        print("RESULT: FAIL -- %s" % "; ".join(FAILURES))
        return 1
    print("RESULT: OK -- %d checks" % CHECKS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
