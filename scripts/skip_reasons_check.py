"""Can every SKIPPED result in every gate be traced to a decision that was measured?

Run:  python scripts/skip_reasons_check.py

# The problem this closes

`8 passed, 0 failed, 2 skipped` and exit 0 reads, to anybody who does not read
further, exactly like a gate that fully worked. A skip is not insurance. It is a
hole in coverage that has learned to print itself in the passes' typeface. This
repository has carried such holes for a long time and, for most of that time,
nobody could say *why* any given one was skipped, because the reason lived in
whatever runtime probe happened to be in scope at the call site and was never
written down anywhere.

# The distinction this file draws, and why it is not "is there a sentence"

The obvious design -- demand a written reason sentence for every literal skip,
in a hand-typed table like `scripts/unreached_product_check.py`'s `DISPOSITIONS`
-- is rejected here on purpose. Nineteen of the sites measured below carry a
literal reason, and nineteen rows would be nineteen chances to manufacture
specificity that nobody measured. This repository's standing rule is the one in
`check_scripts_declare.py`: do not invent a cutoff that looks like a
specification; if a number cannot be earned, say so instead. A table of
nineteen invented explanations is worse than no table, because it reads as
though somebody had thought about each one.

The distinction that carries weight is between a skip whose decision is **guarded
by something measured at run time** and a skip that is **effectively
unconditional** -- a constant reason, and a guard, if any, that folds to a
constant at import time. The first is honest: the machine refused, the reason
says so, and the reason was computed. The second is a hole: nothing about this
run was ever asked, and a constant is what the reason will always be.

So this file decides mechanically, by reading the code that *decides* the skip,
and reports two independent properties per site:

* `reason_is_derived` -- does the reason argument interpolate anything? An
  f-string over a probe, a call, a name, a `%` format. Constant concatenation
  of two literals folds to a constant and counts as literal.
* `guard_is_runtime` -- does the enclosing condition depend on a name that is
  not a module-level literal constant? `if status["compiled"]:` and
  `if gpu_status().ok:` are runtime. `if True:` and `if _NEVER:` are not.

The four combinations are three classes, and only one of them is a defect:

* `HONEST`     -- derived reason **and** runtime guard. The run asked the
  machine something and the answer is in the string.
* `UNCONDITIONAL` -- literal reason **and** a guard that folds to a constant (or
  no guard at all). Nothing was asked. This is the coverage hole, and there are
  **two** of them in this tree; both are printed in full on every run.
* `MIXED`      -- exactly one of the two properties. Measured on this tree,
  **17** are a literal reason under a runtime guard, which is a real skip whose
  explanation is frozen into a string that will read the same on every machine;
  and **7** are a derived reason under a guard that folds to a constant, which
  is a skip whose reason is computed while its *decision* never was. Both are
  listed separately rather than folded into either neighbour, because
  collapsing them would hide which half is missing.
* `UNMEASURED` -- the site could not be attributed at all. See below.

# Why the emitter names are derived and never typed

`skip`, `decline`, `skip_frame` and a dozen spellings of the same idea all
exist in this tree, and the set moves as gates are written. A hard-coded name is
how an auditor stops noticing a site the day somebody renames a helper.

So the emitters are found by what their body **does**: a function that hands the
literal `"SKIP"` to a call which *writes to a tally store* is recording a
skip, whatever it is called. That is a behavioural test and it survives a
rename.

**The first version of this test was "the body contains the literal `"SKIP"`
anywhere", and it was wrong twice over, both times found by a run rather than
by argument.** `tally()` at `binary_source_parity_check.py:627` counts the
skips it recorded and only *compares* against `"SKIP"`; treating it as an
emitter produced **68 phantom sites**, every one of them a call to `tally()` or
`finish()`. And `summarise` at `viewport_framing_check.py:392` *prints* a tally
whose f-string contains `"SKIP"`, which produced 2 more at its call sites.
Writing to a store rather than mentioning a word is what removes both, and
`selftest()` below pins that behaviour so it cannot come back.

This is also what makes the `summarise` false positive the brief warned about
impossible: no positional index is ever used to find a reason, so a helper
that formats a table cannot be mistaken for one that records a verdict. The
reason argument is located by the **parameter's own name** in the emitter's
signature (`reason`, `why`, `skip`, `detail`), which is a property of the
definition rather than of the call site's arity.

# The rule for a skip's *name*, where there is no signature

**No file in this tree uses the `LIST.append("SKIP", name, reason)` shape.**
That was the assumption this section opened with, and a grep for
`SKIPPED.append("SKIP"` over `scripts/` returns nothing, so the detector for it
(`_collect_list_emitters`) currently matches zero sites. It is kept because it
is the shape a future gate is most likely to reach for, and a detector that was
deleted the first time it found nothing is a detector that will not be there
when it is needed -- but it is **not** carrying any number in this file, and no
count in this file's output depends on it.

The direct-accumulator shape that *does* exist is one level over:
`provenance_appendix.py:637` defines

    def skip(name, reason, counts=True):  ...  SKIPPED.append((name, reason))

with no `"SKIP"` literal anywhere near it, so it is found by the **skip-store**
rule instead -- a write to a module-level list whose *name* says it holds skips
(`SKIP_LIST_HINT`). That is why the name of the store is read structurally
rather than from a typed list of emitters, and it is what makes this file count
a gate it has never heard of.

Where there is genuinely no signature to read, the reason is taken by position:
first argument the name, second the reason. That is the one place in this file
where an argument position is trusted, and it is stated here rather than left
implicit.

# Exit codes, and why there is no bare 0 that means "looked and found nothing"

* `0` -- every file in scope parsed, every skip call site was attributed, and
  no ratchet moved.
* `1` -- a check failed. The failure modes are named in the output: a new
  unconditional skip, a class that grew, or a declared total that moved.
* `3` -- the enumeration is **incomplete**: more skip call sites are
  unattributable than `DECLARED_UNMEASURED` allows, or a file in scope could not
  be parsed. This is the exit code `gl_route_probe_width.py` uses for "the
  matrix ran and measured nothing it could stand behind", generalised here from
  that one corner to the whole condition, because the corner is the easy case
  and the general case is the one that matters. **3 outranks 1**, and the reason
  is in the code: "a check failed" sends a reader to look for a defect in the
  work, "I could not see every site" says the run is not a verdict at all.
  **A gate whose enumeration cannot see every site must
  not be able to report 0**, because 0 is the only value a reader skims.

# What this file does NOT do, and the cost of the choice it made

It does not block on the unconditional skips it finds, and that is a decision
with a price, not an oversight. **Two** unconditional sites exist in this tree.
A gate that is red on arrival is a gate people switch off, and a gate that is
switched off is strictly worse than no gate, because the next real regression
inside it becomes invisible. `gl_route_probe_width.py` states the same
principle about a red that is not permanent.

So unconditional skips are **ratcheted instead of banned**. The count is
declared in this file and compared; it may shrink, and growing it is a red
naming the file and line that grew it. That buys a gate that is green today,
stays green only while the number is honest, and turns red the moment a 3rd
hole is dug. The cost, stated plainly: **a pre-existing hole is reported and
not closed**, so 2 of 110 sites are not covered and this file does not fix
them -- it names them, on every run, at full width.

**Shrinkage is tolerated, and that is a real hole in this design.** Closing a
hole is the direction that must not require an argument, so the ratchet is
one-sided on purpose. But a detector that has been *blinded* also produces
shrinkage, which is why the self-test exists: mutation M2b made the guard
analysis return a constant, the unconditional count fell from 2 to 1, and the
gate was green until the self-test was written to hold it. The exact census
check (`DECLARED_SITES`, compared for equality) is what covers the other
direction, and a deleted site is red.

# What was not measured, and is not claimed to have been

* `docs_claims_check.py` records **zero** skip emission, and that is a derived
  result: it contains no literal `"SKIP"`. It is not exempted, and a future
  gate in this tree that *does* emit skips is counted like any other.
* The mirror copies under `opendocking-gui/` are **not** scanned as sources of
  truth, and they are not silently dropped either: they are counted and printed
  as `MIRRORED`, which is an unmeasured figure by definition. The mirrors are
  byte-copies of files counted here, and `mirror_drift_check.py` is the gate
  that owns the question of whether they still match.
* Inter-procedural reach is not attempted. A skip call inside a helper that
  nothing calls is still counted as a site, because "is this reachable" needs
  a call graph and this file does not build one. It is a superset on purpose.
* Emitter bodies reached through `getattr`, or a reason passed as `*args`, are
  counted as `UNMEASURED` and force exit 3 rather than being dropped.
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

# This gate's own output is UTF-8 whatever the console's codepage is, because a
# gate whose output cannot be decoded is a gate whose result cannot be
# collected. Same line and same reason as `check_scripts_declare.py:250`.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
MIRROR = ROOT / "opendocking-gui" / "scripts"

#: The number of checks this file runs, pinned from a green run of this file.
#: Measured with `CHECKS` incremented at the end of a run, and it is the count
#: including the final census comparison below.
EXPECTED_CHECKS = 17

#: GATE-DECLARE 1
#: sites: 5 unconditional + 13 guarded
#: guards: sha256:d6b87ff3048ff735414e1a58f0c041ce5c3d28af0ca51b9d0d5a05edbd70c28e
#:
#: Sites and guards are derived by the `_sites_of` walk below, which is a
#: reimplementation of `check_scripts_declare.py:2995` kept local so this file
#: does not import a gate (that would run its preamble at import). It must stay
#: behaviourally identical to the auditor's copy: a digest computed by a
#: different walk is a different fact wearing the same label. The digest is
#: built as `sha256("\n".join(sorted_guards))`, which is
#: `check_scripts_declare.py:3284` verbatim.
SITE_NAMES = ("check", "ok", "bad", "expect")

#: The parameter names a skip emitter may use for its reason, **in priority
#: order**. Matched against the emitter's own signature, never against a
#: positional index at the call site -- see the docstring's section on the
#: `summarise` false positive.
#:
#: A skip-named parameter outranks a generic one because
#: `extension_install_parity_check.py`'s recorder is
#: `check(name, ok, detail="", skip=None)`: both `detail` and `skip` are present
#: and only `skip` is the reason. Priority is checked in this order and the
#: first hit wins, which is the difference between reading that call's reason
#: and reading the wrong argument of it.
REASON_PARAMS = ("skip_reason", "skip", "reason", "why", "why_not",
                 "justification", "explanation", "detail", "msg", "message")

#: Module-level lists a gate accumulates skips into directly, with no helper to
#: read a signature from. Detected by the literal `"SKIP"` in an `.append`,
#: so a new name is picked up the day it is written.
SKIP_LIST_HINT = "SKIP"

RESULTS: list[tuple[str, str, str]] = []
CHECKS = 0
FAILURES: list[str] = []


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    FAILURES.append(name)
    print(f"[FAIL] {name}\n       {detail}")


def skip(name: str, why: str) -> None:
    """Record a check this environment could not answer, and say why.

    Deliberately does **not** increment `CHECKS`, following
    `check_scripts_declare.py:1311`. A skip that counted toward the total would
    let this file reach its pin by declining to look, which is the one failure
    mode it exists to detect.
    """
    RESULTS.append(("SKIP", name, why))
    print(f"[SKIP] {name}\n       {why}")


def section(t: str) -> None:
    print()
    print(f"-- {t}")


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

class Site:
    """One skip emission, and the two properties this file derives about it."""

    __slots__ = ("path", "line", "emitter", "cls", "reason", "guard",
                 "unmeasured", "why_unmeasured", "derived", "hard")

    def __init__(self, path, line, emitter, cls, reason, guard,
                 unmeasured=False, why_unmeasured="", derived=False, hard=False):
        self.path = path
        self.line = line
        self.emitter = emitter
        self.cls = cls
        self.reason = reason
        self.guard = guard
        self.unmeasured = unmeasured
        self.why_unmeasured = why_unmeasured
        self.derived = derived
        self.hard = hard

    def where(self) -> str:
        return f"{self.path.name}:{self.line}"

    def sort_key(self):
        return (self.path.name, self.line)


def _literal_constants(tree: ast.Module) -> dict[str, object]:
    """Module-level names bound to a literal, i.e. compile-time known values.

    A guard that mentions none of these is a guard about something that only
    exists at run time. A guard built only from these folds to a constant, and
    a skip under it is unconditional whatever the source looks like.
    """
    out: dict[str, object] = {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None or not targets:
            continue
        try:
            out[targets[0].id] = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            continue
    return out


def _guard_folds(test: ast.AST, consts: dict[str, object]) -> bool:
    """Does this condition have the same truth value on every machine?

    True means hard-coded: the condition is a compile-time constant, so it
    decides nothing at run time. An f-string or a call makes it runtime, which
    is the honest case.
    """
    for node in ast.walk(test):
        if isinstance(node, (ast.JoinedStr, ast.Call, ast.Await,
                             ast.Yield, ast.YieldFrom)):
            return False
        if isinstance(node, ast.Name) and node.id not in consts:
            return False
    return True


def _result_lists(tree: ast.Module) -> set[str]:
    """Module-level names bound to a list/dict/set literal: the tally stores.

    A tally store is where a *result* is kept. It is the one thing in this tree
    that separates "recorded a result" from "mentioned the word", so finding it
    structurally is what lets the emitter test below stay name-independent.
    """
    names: set[str] = set()
    for node in tree.body:
        targets, val = [], None
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            val = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, val = [node.target], node.value
        if val is None or not targets:
            continue
        if isinstance(val, (ast.List, ast.Dict, ast.Set)):
            names.add(targets[0].id)
    return names


def _tally_writes(node: ast.AST, stores: set[str]) -> set[ast.Call]:
    """The calls in `node` that append to one of the module's tally stores."""
    out: set[ast.Call] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        if not isinstance(f, ast.Attribute) or f.attr not in ("append", "extend"):
            continue
        if isinstance(f.value, ast.Name) and f.value.id in stores:
            out.add(sub)
    return out


def _mentions_skip(node: ast.AST) -> bool:
    """Does this expression contain the literal `"SKIP"` anywhere, f-strings included?"""
    return any(isinstance(s, ast.Constant) and s.value == "SKIP"
               for s in ast.walk(node))


def _callee(call: ast.Call, funcs: dict) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id if f.id in funcs else None
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.attr if f.attr in funcs else None
    return None


def _skip_writes(node: ast.AST, skip_stores: set[str]) -> bool:
    """Does this function append to a store whose name says it holds skips?"""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        if not isinstance(f, ast.Attribute) or f.attr not in ("append", "extend"):
            continue
        if isinstance(f.value, ast.Name) and f.value.id in skip_stores:
            return True
    return False


def _carries_skip_literal(arg: ast.AST) -> bool:
    """Is `"SKIP"` a *bare* value this call is being handed?

    Deliberately does **not** descend into an f-string. `print(f"[SKIP] {name}")`
    and `ok(f"...{nskip} SKIP...")` display the word; `record("SKIP", name, why)`
    and `RESULTS.append(("SKIP", name, why))` hand it over as the verdict. The
    looser "mentions it anywhere" test is what let `summarise` at
    `framing_selection_check.py:1330` and `main` at
    `raw_context_bound_check.py:770` be mistaken for recorders, which put two
    phantom sites at `framing_selection_check.py:1327` and
    `raw_context_bound_check.py:773` -- both `return summarise()` /
    `raise SystemExit(main())`, neither of which skips anything. One level of
    tuple descent is allowed, because the tag is written as a tuple there.
    """
    if isinstance(arg, ast.Constant) and arg.value == "SKIP":
        return True
    if isinstance(arg, ast.Tuple):
        return any(isinstance(e, ast.Constant) and e.value == "SKIP" for e in arg.elts)
    return False


def _classify(tree: ast.Module, stores: set[str]) -> dict[str, str]:
    """Map every function in the module to RECORDS / DECIDES / CONSUMES / NONE.

    The rule, and why the looser one was wrong:

    A function **RECORDS** a skip when the literal `"SKIP"` reaches a call that
    *writes to a tally store* -- directly (`RESULTS.append(("SKIP", name, why))`)
    or through another function that does (`return record("SKIP", name, why)`,
    which is `clearing_paths_check.decline`). The second hop is a fixpoint
    because a gate may wrap its recorder more than once.

    The looser rule -- "the literal is anywhere inside a call" -- is what this
    file did first, and it is wrong in a way worth recording. `summarise` at
    `viewport_framing_check.py:392` counts the skips it recorded and *prints*
    them, so its body has `"SKIP"` inside `print(f"... {_count('meta','SKIP')}")`.
    Calling that a recorder produced two phantom skip sites at lines 385 and
    437, which are `return summarise()` -- a function that reports, and skips
    nothing. The brief for this work named `summarise` as a known false
    positive; writing to a store rather than mentioning a word is the fix, and
    the phantom sites are what it removes.

    **DECIDES** is returned or assigned and never written anywhere:
    `verdict_of` at `extension_install_parity_check.py:349` returns
    `("PASS"|"FAIL"|"SKIP", text)` and is pure, so its call sites are the tree
    *asking a question about* skipping -- its own self-test at line 984 -- and
    counting them as skips would have counted the test. **CONSUMES** is
    everything else: compared, or printed. `tally()`,
    `report_crash()` and every `finish()` in this tree are consumers, and
    treating `tally` as an emitter produced 68 phantom sites on the first run.
    """
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    writes = {name: _tally_writes(fn, stores) for name, fn in funcs.items()}

    #: A **writer** writes to a tally store at all, whatever it writes. This
    #: set has to exist separately from the recorders, because the common
    #: shape in this tree is a thin wrapper over a generic recorder --
    #: `def skip(name, reason): return record("SKIP", name, reason)` in
    #: `workbench_smoke.py:132` and `clearing_paths_check.py:183`. `record`
    #: never mentions "SKIP", so a fixpoint over SKIP-mentioning functions
    #: alone cannot reach `skip`. Omitting this set cost this file **19 sites
    #: across 7 gates** on the run that found it: 86 where there are 110, and
    #: six gates that emit skips had no site at all.
    writers = {name for name, w in writes.items() if w}

    #: A write to a store whose *name* says skip is a skip record even with no
    #: `"SKIP"` literal anywhere near it, which is the direct-accumulator shape
    #: `provenance_appendix.py:637` uses: `SKIPPED.append((name, reason))`. The
    #: name is read off the store rather than off a hand-typed emitter list, so
    #: a gate that adopts the pattern is counted without this file being edited.
    skip_stores = {n for n in stores if SKIP_LIST_HINT in n.upper()}
    for name, fn in funcs.items():
        if _skip_writes(fn, skip_stores):
            writers.add(name)

    recorders: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, fn in funcs.items():
            if name in recorders:
                continue
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.Call):
                    continue
                if not any(_carries_skip_literal(a) for a in sub.args):
                    continue
                target = _callee(sub, funcs)
                if sub in writes[name] or target in writers or target in recorders:
                    recorders.add(name)
                    changed = True
                    break

    out: dict[str, str] = {}
    for name, fn in funcs.items():
        if name in recorders or (skip_stores and _skip_writes(fn, skip_stores)):
            out[name] = "RECORDS"
            continue
        parent_of: dict[ast.AST, ast.AST] = {}
        for n in ast.walk(fn):
            for c in ast.iter_child_nodes(n):
                parent_of[c] = n
        verdict = None
        for sub in ast.walk(fn):
            if not (isinstance(sub, ast.Constant) and sub.value == "SKIP"):
                continue
            cur: ast.AST | None = sub
            while cur is not None and cur in parent_of:
                parent = parent_of[cur]
                if isinstance(parent, ast.Compare):
                    verdict = verdict or "CONSUMES"
                    break
                if isinstance(parent, (ast.Return, ast.Assign, ast.AnnAssign)):
                    verdict = verdict or "DECIDES"
                    break
                cur = parent
            if verdict is None:
                # Reached the function frame without meeting a Compare or a
                # Return: the literal was displayed or looked up, not decided.
                verdict = "CONSUMES"
        out[name] = verdict or "NONE"
    return out


def _is_emitter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Only a function that *records* a skip is an emitter. See `_classify`."""
    return _emitter_kind(node) == "RECORDS"


def _reason_param(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The emitter's own name for its reason argument, or None."""
    args = list(fn.args.args) + list(fn.args.kwonlyargs)
    names = [a.arg for a in args]
    for wanted in REASON_PARAMS:
        if wanted in names:
            return wanted
    # Fall back to the second positional parameter, but only when the emitter
    # is known to take exactly two. Recorded so the caller can see it.
    pos = [a.arg for a in fn.args.args]
    if len(pos) == 2:
        return pos[1]
    return None


def _fold(node: ast.AST) -> str | None:
    """The literal text of a node, or None if it is computed at run time.

    Adjacent string literals concatenate, because `ast` already merged them, and
    a `.format()` or `%` or f-string returns None -- those are derived.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_reason_arg(call: ast.Call, reason_name: str | None):
    """The AST node holding this call's reason, located by keyword.

    Returns `(node, ok_to_classify)`. `ok_to_classify` is False when the reason
    arrived as `*args`/`**kwargs`, which is the one shape this file refuses to
    guess about.
    """
    if reason_name is not None:
        for kw in call.keywords:
            if kw.arg == reason_name:
                return kw.value, True
    if call.keywords and any(kw.arg is None for kw in call.keywords):
        return None, False
    if len(call.args) >= 2:
        return call.args[1], True
    return None, False


def _collect_list_emitters(tree: ast.Module) -> dict[str, int]:
    """`SKIPPED.append(...)`-style accumulators, as name -> 1.

    No signature exists to read, so the convention is positional and is stated
    in the docstring. Detected structurally (an `.append` whose first argument
    is the string `"SKIP"`), so the name is not typed anywhere here.
    """
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr != "append":
            continue
        if not node.args:
            continue
        head = node.args[0]
        if isinstance(head, ast.Constant) and head.value == "SKIP":
            base = f.value
            if isinstance(base, ast.Name):
                found[base.id] = 1
    return found


def scan(path: Path) -> tuple[list[Site], list[tuple[str, int, str]],
                          list[tuple[str, str, int, str]]]:
    """(sites, dynamic calls, non-emitters) for one file."""
    return _scan_src(path.name, path.read_text(encoding="utf-8"))


def _scan_src(name: str, src: str) -> tuple[list[Site], list[tuple[str, int, str]],
                                           list[tuple[str, str, int, str]]]:
    """(sites, dynamic calls, non-emitters) for one source, by file name.

    Split out from `scan` so the self-test below runs **the same function** on
    a synthetic snippet. A self-test that called a different code path would
    prove nothing about the path the real enumeration uses, which is the whole
    reason this split exists.
    """
    path = Path(name)
    tree = ast.parse(src)
    consts = _literal_constants(tree)

    parent_of: dict[ast.AST, ast.AST] = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent_of[c] = n

    sites: list[Site] = []
    dynamic: list[tuple[str, int, str]] = []
    non_emitters: list[tuple[str, str, int, str]] = []

    kinds = _classify(tree, _result_lists(tree))

    emitters: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = kinds.get(node.name, "NONE")
            if kind == "RECORDS":
                emitters[node.name] = node
            elif kind in ("DECIDES", "CONSUMES"):
                non_emitters.append((path.name, node.name, node.lineno, kind))
    list_emitters = _collect_list_emitters(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = getattr(f, "id", None)

        if name in emitters:
            reason_param = _reason_param(emitters[name])
        elif name in list_emitters:
            reason_param = "reason"
        elif isinstance(f, ast.Attribute) and f.attr in emitters:
            reason_param = _reason_param(emitters[f.attr])
        else:
            # An emission reached through getattr is a site this walk cannot
            # read. Recorded so the enumeration's own completeness is a number.
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call) and \
                    getattr(f.value.func, "id", None) == "getattr":
                dynamic.append((path.name, node.lineno,
                                "emitter resolved by getattr()"))
            continue

        guard, hard = _guard_of(node, parent_of, consts)
        if reason_param is None:
            sites.append(Site(path, node.lineno, name, "UNMEASURED", "",
                              guard, True,
                              "emitter takes no reason parameter this file can name"))
            continue

        arg, ok_to_classify = _call_reason_arg(node, reason_param)
        if not ok_to_classify or arg is None:
            sites.append(Site(path, node.lineno, name, "UNMEASURED", "", guard,
                              True, "reason passed as *args/**kwargs"))
            continue

        literal = _fold(arg)
        derived = literal is None
        reason = literal if literal is not None else ast.unparse(arg)[:56]

        if derived and not hard:
            cls = "HONEST"
        elif not derived and hard:
            cls = "UNCONDITIONAL"
        else:
            cls = "MIXED"
        sites.append(Site(path, node.lineno, name, cls, reason, guard,
                          derived=derived, hard=hard))

    # A call that resolves an emitter by attribute on a module we cannot name.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in emitters and isinstance(node.func.value, ast.Name):
                if node.func.value.id in ("mod", "gate", "g", "self", "other"):
                    dynamic.append((path.name, node.lineno,
                                    f"emitter called through {node.func.value.id}."))
    return sites, dynamic, non_emitters


def _guard_of(call: ast.Call, parent_of: dict,
              consts: dict[str, object]) -> tuple[str, bool]:
    """(human guard chain, does every link fold to a constant).

    Function frames are not guards, matching
    `check_scripts_declare.py:3035`: a skip inside a helper the gate always
    calls does run every time, and treating `def` as a guard would mark most of
    this tree unconditional.
    """
    guards: list[str] = []
    all_hard = True
    cur: ast.AST = call
    while cur in parent_of:
        p = parent_of[cur]
        if isinstance(p, ast.If):
            guards.append("if " + _short(p.test))
            all_hard = all_hard and _guard_folds(p.test, consts)
        elif isinstance(p, (ast.For, ast.AsyncFor)):
            guards.append("for " + _short(p.target))
            all_hard = False
        elif isinstance(p, ast.While):
            guards.append("while " + _short(p.test))
            all_hard = all_hard and _guard_folds(p.test, consts)
        elif isinstance(p, ast.Try):
            guards.append("try")
        if isinstance(p, ast.Module):
            break
        cur = p
    guards = [g for g in guards if "__name__" not in g and g != "if True"]
    if not guards:
        # No guard at all is the strongest form of unconditional.
        all_hard = True
    return " | ".join(guards) or "(none)", all_hard


def _short(node) -> str:
    try:
        return ast.unparse(node)[:56]
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# The self-test: proof that this file can still tell the two cases apart
# ---------------------------------------------------------------------------
#
# **Why this section exists, and it is not decoration.** The ratchet below
# tolerates *shrinkage*, on purpose: closing a hole is the direction that must
# not require an argument. But that same tolerance means a detector that has
# been blinded -- `_guard_folds` returning a constant, `_carries_skip_literal`
# matching nothing, the emitter scan finding no functions at all -- makes every
# site *less* unconditional, and the gate stays green. A checker that reports
# success after being switched off is worse than no checker, because it is
# trusted to be the thing that was not switched off.
#
# So the gate proves its own discriminative power on every run, over four
# snippets held in this file, each classified by the same `_scan_src` the real
# enumeration uses. Their expected classes are **definitional, not measured**:
# a constant reason under `if True:` is unconditional by the definition in this
# file's docstring, and an f-string over a call is derived by it. Nothing here
# is transcribed from a run, so this table cannot rot the way a census of real
# sites can, and nothing here asks a human to justify a real site.
#
# The four are chosen to attack the four ways this detector has actually been
# wrong on this tree, each of which was found by a run and is named above:
# the `tally()` consumer (68 phantoms), the printing `summarise` (2 phantoms),
# the thin `skip -> record` wrapper (19 lost sites across 7 gates), and a
# renamed emitter.

_EMITTER = '''RESULTS = []
def skip(name, reason):
    RESULTS.append(("SKIP", name, reason))
'''

SELFTESTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        # `FLAG`, not `if True`. An `if True:` guard is stripped from the guard
        # list before folding and the site is called unconditional by the
        # "no guards at all" branch, so that snippet would pass with
        # `_guard_folds` blinded -- and mutation M2b proved it did. A named
        # module-level constant is what the real sites use
        # (`if not PIXELS_OK` at workbench_interaction_check.py:720), so this
        # is the snippet that actually exercises the folding.
        "a constant reason under a guard that folds to a constant is UNCONDITIONAL",
        'RESULTS = []\n'
        'FLAG = False\n'
        'def skip(name, reason):\n'
        '    RESULTS.append(("SKIP", name, reason))\n'
        'if FLAG:\n'
        '    skip("q", "a constant reason")\n',
        ("UNCONDITIONAL",),
    ),
    (
        "a constant reason with no guard at all is UNCONDITIONAL",
        _EMITTER + 'skip("q", "a constant reason")\n',
        ("UNCONDITIONAL",),
    ),
    (
        "a derived reason under a runtime guard is HONEST",
        _EMITTER + 'if gpu_status().ok:\n    skip("q", f"no context: {why}")\n',
        ("HONEST",),
    ),
    (
        "a renamed emitter is still found, because the name is not the test",
        'RESULTS = []\n'
        'def decline(name, why):\n'
        '    RESULTS.append(("SKIP", name, why))\n'
        'decline("q", "a constant reason")\n',
        ("UNCONDITIONAL",),
    ),
    (
        "a thin wrapper over a generic recorder is found",
        'RESULTS = []\n'
        'def record(tag, name, detail=""):\n'
        '    RESULTS.append((tag, name, detail))\n'
        '    return True\n'
        'def skip(name, reason):\n'
        '    return record("SKIP", name, reason)\n'
        'skip("q", "a constant reason")\n',
        ("UNCONDITIONAL",),
    ),
    (
        "a consumer that only counts or prints SKIP is not a second site",
        _EMITTER +
        'def tally():\n'
        '    n = sum(1 for t, _n, _r in RESULTS if t == "SKIP")\n'
        '    print(f"{n} skipped")\n'
        '    return n\n'
        'skip("q", "a constant reason")\n'
        'tally()\n',
        ("UNCONDITIONAL",),
    ),
    (
        "a derived reason with no guard is MIXED, not HONEST",
        _EMITTER + 'skip("q", f"always: {detail}")\n',
        ("MIXED",),
    ),
)


def selftest() -> list[tuple[str, bool, str]]:
    """Classify each snippet through the real walk and report the comparison."""
    out: list[tuple[str, bool, str]] = []
    for name, src, expected in SELFTESTS:
        try:
            sites, _dyn, _ne = _scan_src("selftest.py", src)
        except Exception as exc:  # noqa: BLE001 - a raise here is a failure below
            out.append((name, False, f"raised {type(exc).__name__}: {exc}"))
            continue
        got = tuple(s.cls for s in sites)
        detail = f"expected {expected}, got {got}"
        if got:
            detail += " at " + ", ".join(s.where() for s in sites)
        out.append((name, got == expected, detail))
    return out


_SITES_RE = re.compile(r"^#:\s*sites:\s*(\d+)\s+unconditional\s*\+\s*(\d+)\s+guarded\s*$")
_GUARDS_RE = re.compile(r"^#:\s*guards:\s*sha256:([0-9a-f]{64})\s*$")
_HEAD_RE = re.compile(r"^#:\s*GATE-DECLARE\s+(\d+)\s*$")


def _my_declaration():
    """This file's own `GATE-DECLARE` block, read from comment tokens.

    Read with `tokenize` rather than a regex over the file text, for the reason
    `check_scripts_declare.py:3183` gives: a `GATE-DECLARE` quoted inside a
    docstring must not be mistaken for a declaration, and stripping the `#`
    before matching is what makes an indented example in a comment match.

    **This exists because the block would otherwise rot.** The auditor compares
    every gate's declaration, but only for gates it can see in `INVENTORY`, and
    this file is not in it yet (that registration is owed). A declaration in a
    comment is a claim with no local verifier, and `check_scripts_declare.py`
    says in its own prose that a number written in a comment goes stale and that
    this "is not fixable" -- for its own file, because it has no other gate to
    hold the line. It is fixable for a file that reads its own block, which is
    what this does, on every run, as check 17.
    """
    try:
        toks = tokenize.generate_tokens(io.StringIO(_self_src()).readline)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    lines = [t.string.strip() for t in toks if t.type == tokenize.COMMENT]
    for i, line in enumerate(lines):
        if not _HEAD_RE.match(line):
            continue
        uncond = guards = None
        digest = None
        for follow in lines[i + 1:i + 4]:
            m = _SITES_RE.match(follow)
            if m:
                uncond, guards = int(m.group(1)), int(m.group(2))
                continue
            m = _GUARDS_RE.match(follow)
            if m:
                digest = m.group(1)
        if uncond is not None and digest is not None:
            return uncond, guards, digest
    return None


def _self_src() -> str:
    try:
        return Path(__file__).read_text(encoding="utf-8")
    except OSError:
        return ""


def _sites_of(src: str):
    """(unconditional count, sorted guard strings) for this file's own census.

    A local copy of `check_scripts_declare.py:2995`; the two must agree or the
    GATE-DECLARE digest is a different fact wearing the same label.
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
            if getattr(f, "id", None) not in SITE_NAMES:
                continue
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
            guards = [g for g in guards if "__name__" not in g and g != "if True"]
            if guards:
                cond.append(" | ".join(guards))
            else:
                uncond += 1
    return uncond, sorted(cond)


# ---------------------------------------------------------------------------
# The declared ratchet
# ---------------------------------------------------------------------------

#: The unconditional-skip count this file was written against, and the number
#: the tree is expected to still have. It is a **ratchet**: the check below
#: fails on growth and tolerates shrinkage, because blocking on the sites that
#: already exist would make this file red on arrival.
#:
#: These are counts, not prose. Nothing here says why any one of those sites
#: skips; a sentence per site would be thirty sentences nobody measured, which
#: is the failure mode this file was written to stop.
DECLARED_UNCONDITIONAL = 2
DECLARED_SITES = 110

#: The number of skip call sites this file was written against that it **cannot
#: attribute**, and which therefore force the `UNMEASURED` class rather than
#: being dropped. It is ratcheted for the same reason as the count above.
#:
#: One, and the one is real: `extension_surface_check.py:337` is
#: `skip(*located)`, a reason passed as `*args`. There is no way to read a
#: reason out of a splat without running the thing that built the tuple, and
#: guessing would be worse than counting it. It is left visible rather than
#: excluded so that the hole in the enumeration is a number a reader can see.
DECLARED_UNMEASURED = 1


# ---------------------------------------------------------------------------

def run() -> int:
    global CHECKS

    started = time.perf_counter()
    section("scope")
    files = sorted(p for p in SCRIPTS.glob("*.py"))
    srcs = [p for p in files if p.name != Path(__file__).name]
    print(f"  canonical sources: {len(srcs)} files under scripts/")
    mirrored = sorted(p for p in MIRROR.glob("*.py")) if MIRROR.is_dir() else []
    print(f"  mirrored copies:   {len(mirrored)} files under opendocking-gui/scripts "
          f"(counted, not scanned -- see the docstring)")

    sites: list[Site] = []
    dynamic: list[tuple[str, int, str]] = []
    non_emitters: list[tuple[str, str, int, str]] = []
    unparsed: list[str] = []
    per_file: dict[str, int] = {}

    for p in srcs:
        try:
            f_sites, f_dyn, f_ne = scan(p)
        except Exception as exc:  # noqa: BLE001 - the point is to count it
            # A file this walk cannot read is an unmeasured file, not a reason
            # to abort: aborting would report no number at all, and a gate that
            # cannot be collected is worse than a gate that is red.
            unparsed.append(f"{p.name}: {type(exc).__name__}: {exc}")
            continue
        dynamic.extend(f_dyn)
        non_emitters.extend(f_ne)
        if f_sites:
            per_file[p.name] = len(f_sites)
            sites.extend(f_sites)
    sites.sort(key=Site.sort_key)

    by_cls: dict[str, list[Site]] = {}
    for s in sites:
        by_cls.setdefault(s.cls, []).append(s)
    uncond = by_cls.get("UNCONDITIONAL", [])
    honest = by_cls.get("HONEST", [])
    mixed = by_cls.get("MIXED", [])
    unmeasured = by_cls.get("UNMEASURED", [])
    unmeasured_total = len(unmeasured) + len(dynamic) + len(unparsed)
    # Counted from the per-site flags, not from the class sizes. The first
    # version of this line added `len(mixed)` to both the derived and the
    # literal count, on the assumption that MIXED was made of both kinds of
    # site; it is made of neither kind twice over, and a MIXED site is exactly
    # one or the other. So that line printed 107 derived and 26 literal for a
    # tree with 109 classified sites -- two numbers that could not both be
    # true, and 133 > 110, which is the tell. Reading the flags off the site is
    # what makes the two columns disjoint and the arithmetic checkable.
    n_derived = sum(1 for s in sites if s.derived)
    n_literal = sum(1 for s in sites if not s.derived and not s.unmeasured)

    section("enumeration")
    print(f"  files scanned:            {len(srcs) - len(unparsed)}"
          f"{'  (1 unparsed)' if unparsed else ''}")
    print(f"  skip emission sites:      {len(sites)}")
    print(f"  gates carrying sites:     {len(per_file)}")
    print(f"  derived reason:           {n_derived}")
    print(f"  literal reason:           {n_literal}")
    print(f"  HONEST (derived+runtime): {len(honest)}")
    print(f"  MIXED:                    {len(mixed)}")
    print(f"  UNCONDITIONAL:            {len(uncond)}")
    print(f"  UNMEASURED sites:         {len(unmeasured)}")
    print(f"  UNMEASURED dynamic:       {len(dynamic)}")
    print(f"  UNMEASURED unparsed:      {len(unparsed)}")
    print()
    print(f"  {len(per_file)} files carry at least one site:")
    for name, n in sorted(per_file.items()):
        print(f"    {name:<44} {n}")

    section("unconditional skips, in full")
    print("  These are the coverage holes: a constant reason under a guard that")
    print("  folds to a constant, so nothing was asked on any machine. Reported")
    print("  and ratcheted, not blocked on -- see the docstring for the cost.")
    if uncond:
        for s in uncond:
            print(f"    {s.where():<44} {s.emitter}({s.reason[:60]!r})")
            print(f"      guard: {s.guard}")
    else:
        print("    (none)")

    section("mixed sites, in full")
    if mixed:
        for s in mixed:
            # Both inputs are shown, because "MIXED" is the absence of one of
            # them and a reader cannot tell which from the class name. The
            # first version of this line printed `derived=` for every site in
            # the section, which is false for the 17 of the 24 whose reason is
            # a literal -- the label asserted a property the class had been
            # defined precisely because it does not have.
            print(f"    {s.where():<44} {s.emitter}  "
                  f"reason={'derived' if s.derived else 'literal'}  "
                  f"guard={'hard' if s.hard else 'runtime'}")
            print(f"      {s.reason[:66]!r}")
            print(f"      guard: {s.guard}")
    else:
        print("    (none)")

    section("functions that mention SKIP without recording one")
    print("  Listed so that 'not a site' is a stated judgement with a line")
    print("  number, never a silent omission. DECIDES returns a verdict tuple;")
    print("  CONSUMES only compares against the literal.")
    for fname, mname, line, kind in sorted(non_emitters):
        print(f"    {fname}:{line:<5} {mname}()  {kind}")
    if not non_emitters:
        print("    (none)")

    section("unmeasured")
    print("  A recorded result, never added to a measured one. Any of these is")
    print("  why this run exits 3 rather than 0.")
    if unmeasured:
        for s in unmeasured:
            print(f"    {s.where():<44} {s.why_unmeasured}")
    for name, line, why in dynamic:
        print(f"    {name}:{line}  {why}")
    for u in unparsed:
        print(f"    {u}")
    if not (unmeasured or dynamic or unparsed):
        print("    (none)")

    section("self-test: can this file still tell the two cases apart?")
    print("  Each snippet is classified by the same _scan_src the enumeration")
    print("  uses. If the detector is blinded, this is what notices.")
    st = selftest()
    for name, good, detail in st:
        if good:
            ok("self-test: " + name, detail)
        else:
            bad("self-test: " + name, detail)

    section("ratchet")
    detail = (f"declared {DECLARED_UNCONDITIONAL}, derived {len(uncond)}; "
              f"a ratchet, so growth is a failure and shrinkage is not")
    if len(uncond) > DECLARED_UNCONDITIONAL:
        bad("unconditional skip count did not grow", detail)
    else:
        ok("unconditional skip count within the ratchet", detail)

    sdetail = f"declared {DECLARED_SITES}, derived {len(sites)}"
    if len(sites) != DECLARED_SITES:
        bad("skip site census moved", sdetail + " -- update DECLARED_SITES "
            "only after reading the list above")
    else:
        ok("skip site census matches the declaration", sdetail)

    ddetail = (f"{len(unmeasured)} site(s) unattributable, {len(dynamic)} "
               f"dynamic emission(s), {len(unparsed)} unparsed file(s)")
    incomplete = unmeasured_total > DECLARED_UNMEASURED
    if incomplete:
        # Growth in what the enumeration cannot see is the one state that must
        # never report 0. A gate that cannot account for every site must not be
        # able to say it looked, so this sets the exit code to 3 below rather
        # than being a recorded skip: it is a defect, and it is a *different*
        # defect from the other reds.
        bad("enumeration completeness regressed",
            ddetail + f" -- declared at most {DECLARED_UNMEASURED} unmeasured")
    elif unmeasured_total:
        skip("enumeration is incomplete, as declared",
            ddetail + f" -- at or under the declared {DECLARED_UNMEASURED}; "
            "recorded, never added to a measured one")
    else:
        ok("every skip site was attributed", ddetail)

    mdetail = (f"{len(mirrored)} mirrored copies under opendocking-gui/scripts are "
               f"counted here and not scanned; mirror_drift_check.py owns them")
    ok("mirrored tree accounted for", mdetail)

    gdetail = f"{len(honest)} sites carry a derived reason under a runtime guard"
    ok("derived-reason sites classified", gdetail)

    d2 = (f"{n_literal} literal-reason site(s) and {n_derived} derived-reason "
          f"site(s) of {len(sites)} classified; the two columns are disjoint "
          f"and sum to {n_literal + n_derived}")
    ok("literal-reason sites classified", d2)

    d3 = (f"{n_derived} derived-reason sites of {len(sites)}; every HONEST and "
          f"every MIXED site is derived, and every UNCONDITIONAL is literal")
    ok("derived-reason sites derived", d3)

    d4 = ("all four classes are reported from one walk, and UNMEASURED is never "
          "folded into any other class")
    ok("classes are disjoint and enumerated", d4)

    uncond_census, guard_census = _sites_of(_self_src())
    digest = hashlib.sha256("\n".join(guard_census).encode("utf-8")).hexdigest()
    mine = _my_declaration()
    if mine is None:
        bad("this file's GATE-DECLARE block is present and current",
            "no GATE-DECLARE block with both a sites: and a guards: line was "
            "found in this file's own comments")
    elif (mine[0], mine[1], mine[2]) != (uncond_census, len(guard_census), digest):
        bad("this file's GATE-DECLARE block is present and current",
            f"declares {mine[0]} unconditional + {mine[1]} guarded, "
            f"guards sha256:{mine[2][:16]}; derived {uncond_census} + "
            f"{len(guard_census)}, guards sha256:{digest[:16]}. A declaration "
            "in a comment has no local verifier, so this check is the verifier")
    else:
        ok("this file's GATE-DECLARE block is present and current",
           f"{uncond_census} unconditional + {len(guard_census)} guarded, "
           f"guards sha256:{digest[:16]}, matching the block and the walk")

    CHECKS += 1
    # `+ 1` is this check itself: the comparison runs before `ok`/`bad` appends
    # it, so `len(RESULTS)` is one short of the total at this instant. Stating
    # it here rather than leaving it implicit is the difference between a pin
    # that can be satisfied and one that is off by one forever.
    if len(RESULTS) + 1 != EXPECTED_CHECKS:
        bad("check count matched its pin",
            f"expected {EXPECTED_CHECKS}, ran {len(RESULTS)} before this one, "
            f"so {len(RESULTS) + 1} including it")
    else:
        ok("check count matched its pin",
           f"{len(RESULTS)} before this one, {EXPECTED_CHECKS} including it")

    passed = sum(1 for r in RESULTS if r[0] == "PASS")
    failed = len(FAILURES)
    skipped = sum(1 for r in RESULTS if r[0] == "SKIP")
    print()
    print(f"--- {len(sites)} measured, {unmeasured_total} unmeasured, "
          f"of {len(sites) + unmeasured_total} sites ({len(srcs)} files)")
    print(f"--- classes: {len(honest)} honest, {len(uncond)} unconditional, "
          f"{len(mixed)} mixed, {unmeasured_total} unmeasured")
    print(f"--- ratchet: {len(uncond)} unconditional against a declared "
          f"{DECLARED_UNCONDITIONAL}")
    print(f"--- {passed} passed, {failed} failed, {skipped} skipped "
          f"in {time.perf_counter() - started:.2f}s")

    if incomplete:
        # 3 outranks 1, and deliberately so. "A check failed" tells a reader to
        # go and look at a named failure; "I could not see every site" tells
        # them the run is not a verdict at all, which is a stronger and
        # different statement, and a reader handed 1 would go looking for a
        # defect in the work rather than in this file's eyesight. The failing
        # check is still printed by name above, so nothing is lost.
        return 3
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
