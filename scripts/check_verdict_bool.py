"""Every verdict a check site hands to `check()` has to be a `bool`, and the
reason is a shape that has already shipped in this tree.

Run:  python scripts/check_verdict_bool.py

**The defect this file exists for.** A gate records a result by calling
`check(ok, name, detail)`, and `check` decides with `if not ok`. That is a
*Python truth value*, not a verdict, so anything truthy passes and anything
falsy fails -- including a value that was never a verdict at all. The shape
that got us was a slot that ended `return (not moved, "")`: a **tuple** whose
first element is a perfectly good bool. The tuple is truthy whatever the bool
inside it says, so `check()` took its passing branch on every run, forever, in
the one slot whose entire job was to be the witness against the run touching a
live file. It survived a green 13-of-13 and was visible only because the detail
line said `these moved: check_text_encoding.py` directly under an `[ok]`.

A truthy non-bool is the quieter half: it passes when it means pass, so nothing
is red until the day the inner value is False and the tuple is still truthy.
A **falsy** non-bool -- `()`, `""`, `None`, `0` -- is the half that is a live
false red: a gate reports FAIL over a value that was never a verdict, and
nothing in the output says which value was wrong. This file partitions the two
apart in the detail line for exactly that reason.

**What the sweep can and cannot see, stated before the results.** The subject
is the *static* shape of the argument, resolved by the rules in `_shape_of`,
and the rules are listed there because a sweep whose rules are unstated is a
sweep whose verdicts cannot be argued with. Three limits are load-bearing and
are not papered over:

* it is not a type checker. It resolves literals, operators, and calls to
  functions in the same file. A call into numpy, Qt or the built extension, a
  name bound in more than one place, and a conditional with one arm it cannot
  read are all `UNRESOLVED` unless `EXTERNAL_CALLS` records what was found, and
  every entry in that table is a claim a person made and wrote down how.
* it is not flow-sensitive. A name bound to two different shapes is
  `UNRESOLVED` even when one of the two is a bool, and that is the rule that
  finds `clearing_paths_check.py`'s own `ok`, whose only `ok = ` assignment is
  in an `except` branch. A sweep that took the first binding it found would
  have called that site a bool and been wrong.
* it counts *call sites*, not invocations, and it counts a call in any
  statement position rather than only the `Expr` form `check_scripts_declare.py`
  counts. The three `check(*_verify_..._guard())` spreads in
  `workbench_interaction_check.py` are invisible to the `Expr` walk and are
  resolved here through the returned tuple -- as is a name bound by a `for`
  target, by an `x: bool = ...`, and by an unpacking `a, b = ...`, which is
  most of what a Python gate does with a verdict before handing it over.

**The probe can fail, and that is the only thing that makes the rest a
result.** `check_sweep_has_teeth` takes a real gate out of this tree, splices
its verdict into a tuple, and asserts this sweep reports that one site and
names it. A sweep that cannot report a non-bool would report a clean tree here
too, which is the same green-with-nothing-under-it that this file is about.

**Two house verdicts are executed, not read.** `check()` is defined at module
level in a file that runs its whole audit on import, so the refusal in
`check_scripts_declare.py` cannot be imported and cannot be called from a
sibling gate. It is therefore spliced out of the source and executed here
against a stub, in both directions: a non-bool must be *refused* (recorded as a
failure, never coerced into one) and a real `False` must still be an ordinary
failure. A guard that is present in the source and does nothing at runtime is
the same defect one level up, and reading the source cannot tell the two apart.
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

# This gate's own output is UTF-8 whatever the console's codepage is, for the
# reason every other gate in scripts/ gives: a gate whose output cannot be
# decoded is a gate that only works for the person who remembered.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SELF = Path(__file__).name

#: The names a check-like call is spelled with across this tree. Read out of
#: `check_scripts_declare.py`'s own `_sites_of` rather than typed, so the two
#: files cannot disagree about what a check site is.
CALLS = ("check", "ok", "bad", "expect")

#: Parameter names that mean "the verdict" when a definition does not annotate
#: it. Fourteen files here spell it `def check(name, ok, detail="")` with no
#: annotation at all, so the annotation cannot be the only way to find it.
VERDICT_NAMES = ("ok", "cond", "condition", "verdict", "passed", "passes",
                 "good", "result", "can_draw", "holds", "fine", "valid")

#: Calls whose return type is fixed by the language and documented, so the
#: sweep can say `bool` without importing anything. `str` is handled separately
#: and is the one builtin whose return is *not* a bool, which is why the rule
#: is a list and not "every builtin".
BUILTIN_BOOLS = ("bool", "isinstance", "issubclass", "callable", "all", "any",
                 "hasattr")

#: GATE-DECLARE 1
#: sites: 6 unconditional + 0 guarded
#: guards: sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
#: Six and none guarded, and the zero is the interesting half: every check site
#: in this file is a statement in `main()` at module scope. The two that began
#: as `return check(...)` were moved out of their helpers for that reason --
#: `_sites_of` only counts a call that *is* the whole statement, and the
#: declared total it is compared against is a generated region that only
#: `--pin` may rewrite, so a site the census cannot see is a red this file
#: would leave behind with no permitted way to clear it. The empty-string
#: digest is sha256 of nothing, which is what a file with no guards hashes to.
#: EXPECTED_CHECKS is measured from a run, not typed: the sweep itself, the two
#: bounds on what it cannot see, the probe, the house verdicts, and the tally.
EXPECTED_CHECKS = 6

CHECKS = 0
FAILURES: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    """Record one result. **A non-bool is refused, not coerced.**

    The same guard the house verdict in `check_scripts_declare.py` carries, and
    for the same reason: a truthy non-verdict takes the passing branch, so a
    slot that returns a tuple where a verdict belongs hands this function
    something it can read as "pass" no matter what the tuple contains. The
    refusal keeps the total reachable -- the result is still recorded exactly
    once -- and puts the defect in the output where the run that made it can be
    read.
    """
    global CHECKS
    CHECKS += 1
    if not isinstance(ok, bool):
        FAILURES.append(name)
        print(f"[FAIL] {name}")
        print(f"       this call passed {ok!r} ({type(ok).__name__}), which is "
              f"{'truthy' if ok else 'falsy'}, where a bool belongs. Refused "
              f"rather than coerced: as a verdict it would have read as "
              f"{'a pass' if ok else 'a failure'} regardless of what it meant.")
        if detail:
            print(f"       {detail}")
        return False
    if not ok:
        FAILURES.append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")
    return ok


def section(t: str) -> None:
    print(f"\n=== {t} ===")


def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


def unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unprintable>"


# --------------------------------------------------------------------------
# The rules, in the order they are tried
# --------------------------------------------------------------------------

def _verdict_index(fn) -> int | None:
    """Which positional parameter carries the verdict, or None if none does.

    None is not a failure: `ok(name, detail)` and `bad(name, detail)` -- the
    shape in thirteen files here -- have no verdict parameter at all, because
    the decision *is* which of the two was called. Those sites cannot carry a
    non-bool verdict, which is a stronger result than saying they were checked.
    """
    ps = fn.args.posonlyargs + fn.args.args
    for i, p in enumerate(ps):
        if p.annotation is not None and unparse(p.annotation) == "bool":
            return i
    for i, p in enumerate(ps):
        if p.arg in VERDICT_NAMES:
            return i
    return None


def _returns(fn) -> list:
    """Every `return`ed expression in a function."""
    return [n.value for n in ast.walk(fn)
            if isinstance(n, ast.Return) and n.value is not None]


def _shape_of(node, info, depth=0):
    """(shape, why). shape in `bool` / `non-bool` / `unresolved`."""
    if depth > 8:
        return "unresolved", "recursion cut"
    # 1. Literals. A bool literal is the only expression whose type is decided
    #    by its own spelling.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool", "the literal %r" % (node.value,)
        return "non-bool", "the literal %r" % (node.value,)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return "non-bool", "a %s literal" % type(node).__name__.lower()
    if isinstance(node, ast.Dict):
        return "non-bool", "a dict literal"
    # 2. Operators. Every one of these three is typed `bool` by the language.
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return "bool", "a %s" % type(node).__name__.lower()
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "bool", "a not-expression"
    if isinstance(node, ast.IfExp):
        a = _shape_of(node.body, info, depth + 1)
        b = _shape_of(node.orelse, info, depth + 1)
        if a[0] == b[0] == "bool":
            return "bool", "both arms of the conditional are bool"
        if a[0] == b[0] == "non-bool":
            return "non-bool", "both arms of the conditional are non-bool"
        # Anything else is **not** evidence of a non-bool. An arm the sweep
        # cannot read is a reason to say so, and the first version of this rule
        # called a pair of unreadable arms "non-bool" and so accused
        # `framing_selection_check.py:464` of the defect this file hunts. A
        # classifier that accuses on insufficient evidence is worse than no
        # classifier: the one report it produces is the one nobody re-reads.
        return "unresolved", ("a conditional whose arms are %s and %s"
                              % (a[0], b[0]))
    # 3. `check(*helper())`: the verdict is one element of a returned tuple.
    #    `ok, detail = slot()` is the same question asked one step earlier, and
    #    `_tuple_elements` is what answers both.
    if isinstance(node, ast.Starred):
        elts = _tuple_elements(node.value, info, depth + 1)
        idx = info["want_index"]
        if elts is not None and idx is not None and all(len(e) > idx for e in elts):
            shapes = set()
            for e in elts:
                shapes.add(_shape_of(e[idx], info, depth + 1)[0])
            if len(shapes) == 1:
                only = shapes.pop()
                if only in ("bool", "non-bool"):
                    return only, ("element %d of the tuple %s returns"
                                  % (idx, unparse(node.value)[:40]))
                return "unresolved", ("element %d of the tuple %s returns"
                                      % (idx, unparse(node.value)[:40]))
            return "unresolved", ("element %d of the tuple %s returns %s"
                                  % (idx, unparse(node.value)[:40], sorted(shapes)))
        return "unresolved", ("a spread of %s" % unparse(node.value)[:44])
    # 4. Calls.
    if isinstance(node, ast.Call):
        f = node.func
        name = getattr(f, "id", None)
        if name in BUILTIN_BOOLS:
            return "bool", "%s() returns a bool" % name
        if name == "str":
            return "non-bool", "str() returns a str"
        if name and name in info["local_returns"]:
            anns = info["local_anns"].get(name, [])
            if anns and all(a == "bool" for a in anns):
                return "bool", "%s() is annotated -> bool" % name
            if anns:
                return "non-bool", "%s() is annotated -> %s" % (name, "/".join(anns))
            rets = info["local_returns"][name]
            tuples = [r for r in rets if isinstance(r, ast.Tuple)]
            if len(tuples) == len(rets) and rets:
                return "tuple", "%s() returns a %d-tuple" % (name, len(tuples[0].elts))
            if not rets:
                return "unresolved", "%s() never returns a value" % name
            # A straight-line function that ends in a `return` cannot return
            # anything else, so its returns are its type. Restricted to
            # straight-line bodies on purpose: a function with an `if` in it can
            # fall off the end and yield `None`, and a sweep that cannot see
            # that would call a `None` a bool -- which is the falsy half of
            # this file's subject, arrived at by the honest-looking route.
            if _straight_line(info["local_defs"][name]):
                shapes = set()
                for r in rets:
                    shapes.add(_shape_of(r, info, depth + 1)[0])
                if len(shapes) == 1:
                    only = shapes.pop()
                    if only in ("bool", "non-bool"):
                        return only, ("every return in the straight-line %s() "
                                      "is a %s" % (name, only))
                    return "unresolved", ("every return in the straight-line "
                                          "%s() is a %s" % (name, only))
                return "unresolved", ("the returns in the straight-line %s() "
                                      "disagree: %s" % (name, sorted(shapes)))
            return "unresolved", "%s() is unannotated and not straight-line" % name
        if isinstance(f, ast.Attribute):
            return "external", "a call to %s()" % unparse(f)[:48]
        if name:
            return "unresolved", "a call to %s(), not defined in this file" % name
        return "unresolved", "a call to %s" % unparse(f)[:40]
    # 5. A name. A parameter of the enclosing function shadows every other
    #    binding of that name, which matters because the common spelling of a
    #    verdict *is* `ok` and a file that also assigns `ok` somewhere else
    #    would otherwise have its parameter read as whichever of those won.
    if isinstance(node, ast.Name):
        shadow = info.get("scope_params", {}).get(node.id)
        if shadow is not None:
            fname, ann = shadow
            if ann == "bool":
                return "bool", "the parameter %r of %s(), annotated bool" % (
                    node.id, fname)
            if ann:
                return "non-bool", "the parameter %r of %s(), annotated %s" % (
                    node.id, fname, ann)
            return "unresolved", "the parameter %r of %s(), unannotated" % (
                node.id, fname)
        bound = info["binds"].get(node.id)
        if not bound:
            return "unresolved", "the name %s, bound nowhere in this file" % node.id
        shapes, whys = set(), []
        for value, idx in bound:
            s, w = _shape_of(value, info, depth + 1)
            if idx is not None:
                elts = _tuple_elements(value, info, depth + 1)
                if elts is not None and all(len(e) > idx for e in elts):
                    sub = set()
                    for e in elts:
                        sub.add(_shape_of(e[idx], info, depth + 1)[0])
                    if len(sub) == 1:
                        s, w = sub.pop(), "%s = %s -> element %d [%s]" % (
                            node.id, unparse(value)[:30], idx, w)
                    else:
                        s, w = "unresolved", ("%s = %s -> element %d is %s"
                                              % (node.id, unparse(value)[:30], idx,
                                                 sorted(sub)))
            shapes.add(s)
            whys.append("%s = %s [%s]" % (node.id, unparse(value)[:30], w))
        if shapes == {"bool"}:
            return "bool", whys[0]
        if shapes == {"non-bool"}:
            return "non-bool", whys[0]
        return "unresolved", "bound to %s by %d assignment(s)" % (
            "/".join(sorted(shapes)), len(bound))
    if isinstance(node, ast.Attribute):
        return "external", "the attribute %s" % unparse(node)[:48]
    if isinstance(node, ast.Subscript):
        # `helper(...)[0]` where the helper returns a tuple: the same question
        # rule 3 asks, asked with a subscript instead of a spread.
        idx = None
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, int) and not isinstance(sl.value, bool):
            idx = sl.value
        elts = _tuple_elements(node.value, info, depth + 1)
        if elts is not None and idx is not None and all(len(e) > idx for e in elts):
            shapes = set()
            for e in elts:
                shapes.add(_shape_of(e[idx], info, depth + 1)[0])
            why = "element %d of the tuple %s returns" % (
                idx, unparse(node.value)[:40])
            if shapes == {"bool"}:
                return "bool", why
            if shapes == {"non-bool"}:
                return "non-bool", why
            return "unresolved", "%s %s" % (why, sorted(shapes))
        return "unresolved", "a subscript"
    return "unresolved", "a %s" % type(node).__name__.lower()


def _own_nodes(fn):
    """The nodes of `fn` that are not inside a nested function or lambda.

    A `yield` inside a closure belongs to the closure, and counting it as the
    outer function's would let a generator with no yields of its own be read as
    one that yields a tuple -- the kind of inference that turns a hole into a
    confident wrong answer.
    """
    out = []
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                          ast.ClassDef)):
            continue
        out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _straight_line(fn) -> bool:
    """True when every path through `fn` reaches one of its `return`s.

    The condition for trusting a function's `return` statements as its type.
    A body containing an `if`, a loop or a `try` can fall off the end and
    return `None` without saying so, and `None` is the falsy half of this
    file's subject -- so a sweep that trusted those returns would resolve a
    `None` to a bool by the most convincing route available.
    """
    for n in ast.walk(fn):
        if n is fn:
            continue
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
                          ast.With, ast.Match)):
            return False
        if isinstance(n, ast.Return) and n.value is None:
            return False
    return bool(fn.body) and isinstance(fn.body[-1], ast.Return)


def _tuple_elements(node, info, depth: int):
    """The element lists of a call to a local function that returns a tuple.

    Also a **tuple or list literal**, and that second case is not a convenience.
    Without it, `ok, detail = (not moved, "")` -- a correct unpacking assignment,
    the single most common way a gate in this tree hands a verdict over -- narrowed
    to nothing, and the name then resolved through the plain literal rule to
    `non-bool`. That is the *same verdict* this file gives the defect it was
    written to find, `return (not moved, "")` handed straight to `check()`. One is
    correct code and the other is the bug, and the sweep could not tell them
    apart: it reported `unreached_product_check.py:1709` as passing a non-bool
    verdict, where the code is
    `ascii_ok, ascii_detail = True, "no byte above 0x7f"` followed by
    `check("this file is pure ASCII", ascii_ok, ascii_detail)`. A classifier that
    accuses on insufficient evidence is worse than no classifier, because the one
    report it produces is the one nobody re-reads.

    **The gate's teeth are not weakened by this.** Only the *unpacking-assignment*
    path consults `_tuple_elements`. A tuple in argument position -- `check((a, b))`
    or `check(*(a, b))` -- never reaches it: there is no `binds` entry to narrow,
    so it still resolves through the literal rule to `non-bool`, and the probe
    below splices its mutant exactly that way.

    None when the question cannot be asked, which is the honest answer for
    every other shape. A function with two `return` statements of different
    arities is not a tuple, and treating it as one is how a sweep ends up
    asserting a shape nobody wrote. A generator counts as well as a function,
    because `for name, ok, detail in helper():` is the same question.
    """
    if isinstance(node, (ast.Tuple, ast.List)):
        return [node.elts]
    if not isinstance(node, ast.Call) or isinstance(node.func, ast.Attribute):
        return None
    name = getattr(node.func, "id", None)
    if not name or name not in info["local_returns"]:
        return None
    rets = info["local_returns"][name] + info["local_yields"].get(name, [])
    if not rets or not all(isinstance(r, ast.Tuple) for r in rets):
        return None
    return [r.elts for r in rets]


def _falsiness(node) -> bool | None:
    """Whether a non-bool verdict's *value* is falsy, or None if unknowable.

    The partition the report turns on. `()` and `""` and `None` are falsy, so
    a slot that meant to pass reports a failure over a value that was never a
    verdict -- a red nothing in the output explains. `(not moved, "")` is
    truthy whatever the bool inside it says, so it passes, and passes again
    the day the bool is False. Same defect class, opposite symptoms, and only
    one of them is ever going to be noticed by reading a log.
    """
    if isinstance(node, ast.Constant):
        return not bool(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return len(node.elts) == 0
    return None


def _classify_file(path: Path):
    """(sites, note). `sites` is a list of dicts, one per check-like call."""
    src = read(path)
    tree = ast.parse(src)
    parent_of = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent_of[c] = n
    info = {
        "binds": {}, "local_returns": {}, "local_anns": {},
        "local_defs": {}, "local_yields": {}, "want_index": None,
    }
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info["local_returns"].setdefault(n.name, _returns(n))
            info["local_defs"].setdefault(n.name, n)
            if n.returns is not None:
                info["local_anns"].setdefault(n.name, []).append(unparse(n.returns))
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        ys = [n.value for n in _own_nodes(fn)
              if isinstance(n, (ast.Yield, ast.YieldFrom)) and n.value is not None]
        if ys:
            info["local_yields"].setdefault(fn.name, []).extend(ys)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Name):
                info["binds"].setdefault(t.id, []).append((n.value, None))
            elif isinstance(t, (ast.Tuple, ast.List)):
                for i, e in enumerate(t.elts):
                    if isinstance(e, ast.Name):
                        info["binds"].setdefault(e.id, []).append((n.value, i))
        # `x: bool = ...` is an AnnAssign, not an Assign, and a sweep that
        # collects only Assign reports the name as bound nowhere.
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and n.value is not None:
            info["binds"].setdefault(n.target.id, []).append((n.value, None))
        # `for name, ok, detail in helper():` binds element 1 of what the
        # helper yields, which is the same question as tuple unpacking and is
        # asked against the iterable rather than against an assignment.
        if isinstance(n, (ast.For, ast.AsyncFor)) and \
                isinstance(n.target, (ast.Tuple, ast.List)):
            for i, e in enumerate(n.target.elts):
                if isinstance(e, ast.Name):
                    info["binds"].setdefault(e.id, []).append((n.iter, i))
    defs = {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in CALLS}

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None)
        if fname not in CALLS:
            continue
        site = {"line": node.lineno, "call": fname, "text": unparse(node)[:70],
                "shape": None, "why": "", "verdict": "", "falsy": None}
        if fname not in defs:
            site["shape"] = "no-local-def"
            site["why"] = "no %s() defined in this file" % fname
            sites.append(site)
            continue
        idx = _verdict_index(defs[fname])
        if idx is None:
            site["shape"] = "no-verdict-parameter"
            site["why"] = ("%s() takes no verdict: %s"
                           % (fname, ", ".join(
                               p.arg for p in defs[fname].args.args)))
            sites.append(site)
            continue
        pname = (defs[fname].args.posonlyargs + defs[fname].args.args)[idx].arg
        site["verdict"] = pname
        # The index is what rule 3 needs to know *which* element of a spread
        # tuple is the verdict, so it is passed down rather than re-derived.
        info["want_index"] = idx
        info["scope_params"] = _scope_params(_enclosing(node, parent_of))
        if len(node.args) > idx:
            arg = node.args[idx]
        elif len(node.args) == 1 and isinstance(node.args[0], ast.Starred):
            # `check(*helper())`: there is no argument *at* the verdict
            # position, and rule 3 is exactly the rule for this. Without this
            # arm the three `check(*_verify_..._guard())` sites in
            # `workbench_interaction_check.py` read as "no argument", which is
            # a statement about the call and not about the verdict.
            arg = node.args[0]
        else:
            kw = next((k.value for k in node.keywords if k.arg == pname), None)
            if kw is None:
                site["shape"] = "unresolved"
                site["why"] = ("no argument for %r at position %d and none by "
                               "keyword" % (pname, idx))
                sites.append(site)
                continue
            arg = kw
        shape, why = _shape_of(arg, info)
        site["shape"] = shape
        site["why"] = why
        site["falsy"] = _falsiness(arg)
        sites.append(site)
    return sites, src


def _enclosing(node, parent_of):
    cur = node
    while cur in parent_of:
        cur = parent_of[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
    return None


def _scope_params(fn) -> dict:
    """{parameter name: (function name, annotation or "")} for one function.

    Set per call site rather than globally, because a parameter shadows
    file-level assignments of the same name only inside its own body. The
    imprecision this leaves is a closure: a nested function's body is walked as
    part of the outer one, so a nested `ok = ()` would still be attributed to
    the outer function. No such site exists in this tree, and the direction the
    error runs is towards `unresolved`, not towards a false `bool`.
    """
    if fn is None:
        return {}
    out = {}
    for p in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        out[p.arg] = (fn.name, unparse(p.annotation) if p.annotation else "")
    return out


def sweep() -> tuple[dict, list, list]:
    """(by_file, non_bool, unresolved). One pass over every `*.py` here."""
    by_file, bad, unknown = {}, [], []
    for p in sorted(SCRIPTS.glob("*.py")):
        try:
            sites, _src = _classify_file(p)
        except SyntaxError as exc:
            sites = [{"line": exc.lineno or 0, "call": "-", "text": "",
                      "shape": "unparseable", "why": str(exc)[:60],
                      "verdict": ""}]
        by_file[p.name] = sites
        for s in sites:
            if s["shape"] == "non-bool":
                bad.append((p.name, s))
            elif s["shape"] in ("unresolved", "external", "no-local-def",
                                "unparseable"):
                unknown.append((p.name, s))
    return by_file, bad, unknown


# --------------------------------------------------------------------------
# The blind-spot table: for each shape the sweep cannot read, what it is in
# fact and how that was found. Most entries are a call; the rest are a name
# bound in more than one place, or a conditional, which the sweep declines for
# reasons that have nothing to do with calls.
# --------------------------------------------------------------------------

#: Places a person has looked and written down why this sweep cannot see, as
#: `{head of the reason: what was found}`. **An entry here is not a claim that
#: the expression is a bool.** It is a record that somebody read the site, and
#: the `what was found` column carries its own epistemic status so a reader can
#: tell a measurement from a source somebody read, from a docstring, from an
#: admission of ignorance:
#:
#: * *measured* -- the return type was observed on this machine;
#: * *read* -- the callee is in this tree, every `return` it has was read, and
#:   each of them names the verdict. The observation is of the source and not of
#:   a run, so the entry says why it could not be run;
#: * *documented* -- the API says so and it was not observed here;
#: * *not read* -- the value comes from this project's own package, and nothing
#:   in this file can read it, so the site stays unresolved in substance and is
#:   recorded rather than cleared.
#:
#: Keyed on a prefix of the reason string rather than on `file:line`, because a
#: line number is stale the moment anybody edits above it and a stale key is a
#: silent exemption. The reverse check below is what stops that: an entry that
#: matches nothing any more is a red, so the table cannot quietly grow until it
#: covers the tree.
EXTERNAL_CALLS: dict[str, str] = {
    "np.array_equal()":
        "measured: returns a Python bool, not a numpy scalar, on the numpy "
        "2.2.6 installed here -- which is the distinction that matters, "
        "because a numpy bool would be `isinstance(x, bool)`-false and every "
        "refusal in this file keys on that test",
    "np.allclose()":
        "measured: returns a Python bool on the numpy 2.2.6 installed here",
    "is_relative_to()":
        "measured: pathlib.PurePath.is_relative_to returns a Python bool",
    "is_dir()":
        "measured: pathlib.Path.is_dir returns a Python bool",
    "startswith()":
        "measured: str.startswith returns a Python bool",
    "isEnabled()":
        "documented: PyQt6 declares QWidget.isEnabled() -> bool. Not observed "
        "here, because observing it needs a QApplication and this gate runs in "
        "the job with no display -- which is exactly why it is recorded rather "
        "than measured",
    "isChecked()":
        "documented: QAbstractButton.isChecked() -> bool, same reason as "
        "isEnabled() above",
    "is_protein()":
        "not read: a predicate on this project's own structure wrapper. Named "
        "rather than left out because the name says what it returns and only "
        "the package can be asked",
    "contacts_valid":
        "not read: an attribute of this project's own viewport object. "
        "Whatever the package declares is the answer, and nothing in this file "
        "can read it, so the site is recorded rather than cleared",
    "has_backbone":
        "not read: a property of this project's own record object, same "
        "reason as contacts_valid above",
    "models.declared":
        "not read: an attribute of this project's own model list, same "
        "reason as contacts_valid above",
    "ev.built":
        "not read: an attribute of this project's own term record, same "
        "reason as contacts_valid above",
    "_is_polar_h_bond()":
        "not read: a private predicate in this project's own preparation "
        "module. Named here rather than left out because a private helper is "
        "the one most likely to change its return type, and a name in this "
        "table is a name somebody will look at when it does",
    # -- the non-call shapes ------------------------------------------------
    # The sweep's other blind spots are not calls at all: a name it unions
    # across functions because it is not flow-sensitive, and a tuple element
    # whose helper it cannot straighten. The leading word of each value below
    # carries the status, as it does for the entries above.
    "element 1 of the tuple _half_width_fill_is_the_modules_own_term":
        "measured, and the earlier record here was wrong: the element 1 that "
        "reaches `check` is not the verdict, and the site is a live truthy "
        "non-bool. The callee returns a TWO-tuple `(same and axis, <f-string>)` "
        "and the call site is `check(*_half_width_fill_is_the_modules_own_term"
        "())` into this file's `check(name: str, ok: bool, detail: str = \"\")`, "
        "so the spread binds element 0 to `name` and element 1 to `ok`: the "
        "verdict parameter receives the 354-character detail string, a `str` "
        "that is never empty and so is truthy on every run. Executed here "
        "through the real call site: the unpatched helper returned `(True, "
        "<354-char str>)` and `check` printed `[PASS] True`; with "
        "`framing_selection.drawn_fill` forced to 0.5 the helper returned "
        "`(False, <340-char str>)` and `check` still printed `[PASS] False` -- "
        "a passing branch over a value that was never a verdict, with the bool "
        "printed as the check's own name. Element 0 is a bool, measured True / "
        "False / True in three states (`same` false, both terms below the "
        "vertical term, and unpatched); the defect is the binding, not the "
        "type. In a file this gate does not own, and it is the shape this file "
        "was written for",
    "element 1 of the tuple _verify_blocked_window_guard() returns":
        "measured: the real `_verify_blocked_window_guard()` was executed "
        "through the real call site `check(*_verify_blocked_window_guard())`. "
        "It returns a three-tuple whose element 1 read True, a `bool`, on the "
        "shipped table, and False, a `bool`, with `search_overlap_verdict` "
        "broken so the table disagreed -- and `check` recorded PASS and FAIL "
        "respectively. The reason this entry gave for reading rather than "
        "running it -- that the call needs a QApplication and a window that is "
        "deliberately blocked -- was false: the body is a table over "
        "`search_overlap_verdict` and touches no Qt object at all, so it needs "
        "neither a display nor a QApplication",
    "element 1 of the tuple _verify_loop_turns_guard() returns":
        "measured: the real `_verify_loop_turns_guard(probes)` was executed "
        "through the real call site under `QT_QPA_PLATFORM=offscreen`, with a "
        "real `QApplication` and no GL context and no window in the process. "
        "Element 1 of its three-tuple read True, a `bool`, at probes 8, 1 and "
        "0, and False, a `bool`, when the live loop was replaced by the "
        "file's own `_StarvedEvents` -- and `check` recorded PASS, PASS, PASS "
        "and FAIL. The earlier reason given for not running it, that it posts "
        "Qt events to a live QApplication and so shares the guard's excuse, "
        "was wrong about what that excuse was: an event loop is not a surface, "
        "and nothing here opens a window or an OpenGL context",
    "element 0 of the tuple failing_rows_have_remedies(":
        "measured: the real `failing_rows_have_remedies` body was executed "
        "against stub verdicts covering all four of its paths -- no failing "
        "row, every remedy already on screen, a remedy missing from the text, "
        "and a blank remedy -- and element 0 came back False, True, False, "
        "False, each `bool`. Both of its returns are annotated "
        "`tuple[bool, str]`, and the sweep cannot see the annotation narrow "
        "the element because one of the two returns spells element 0 as a "
        "name. The two sites are this one function reached once from the "
        "panel's own label and once from `PoseVerdict.explanation()`",
    "_write_maps_str() is unannotated and not straight-line":
        "measured: the real `_write_maps_str` body was executed against a stub "
        "writer that creates 40 and then 39 `.map` files, and returned True "
        "and False, both `bool`. It has exactly one `return` and its value is "
        "an `ast.Compare`, so the `with` that makes `_straight_line` decline "
        "is the whole of the blindness here: a `with` cannot be proved by that "
        "rule not to fall off the end, which is the conservative direction and "
        "not a claim that this one does",
    "bound to bool/non-bool/unresolved by 4 assignment(s)":
        "read: at the atom-type check in `pdbqt_check.py` the live binding of "
        "`raised` is the literal False in the `try` arm or the literal True in "
        "the `except ValueError` arm -- a bool literal in both. The `None` and "
        "the caught exception that make the sweep's union mixed are bound some "
        "330 lines earlier in the same function, for a different check that "
        "reads `raised` as an exception object and not as a verdict. The sweep "
        "is not flow-sensitive and cannot see that the first pair was already "
        "consumed",
    "element 1 of the tuple _panel_whole_reachable(":
        "read: `_panel_whole_reachable(...)` is annotated `-> tuple[bool, str]` "
        "and has two returns -- the literal False when there is nothing to "
        "drag, and `(ok, ...)` where `ok = reachable and not_the_floor` is a "
        "BoolOp over the Compares `narrowest is not None` and `room >= need`. "
        "The real helper was executed here against a real MainWindow on the "
        "offscreen platform, never shown and with `viewport.context()` None "
        "after every step, and element 0 came back False, a `bool`, in three "
        "states: as shipped, with `dock=None`, and with the view's floor at "
        "the window's width. The pass state was NOT reached, which is why this "
        "stays `read`: on a window that was never shown the dock does not "
        "reflow and `resizeDocks` is inert, so the h-bar never reaches 0 and "
        "`reachable` is False in every run. Showing the window is what would "
        "move the splitter, and the centre widget is a `QOpenGLWidget`, so that "
        "would take a GL context this machine may not give a second process. "
        "Note the call site supplies the name first, so here the verdict is "
        "element 0 and the key's element 1 is the detail string",
    "a conditional whose arms are external and bool":
        "measured: both arms of that conditional are bool. The arm the sweep "
        "cannot read is `Path.is_relative_to()`, executed here: "
        "`isinstance(Path('a/b').resolve().is_relative_to(Path('a').resolve()), "
        "bool)` is True on the pathlib in this interpreter. The other arm is "
        "`... in ...`, an `ast.Compare`, which this sweep already reads as a "
        "bool. `unresolved` here is one arm it cannot read beside one it can, "
        "which is the situation the `_shape_of` comment above refuses to guess "
        "in and is right to",
    "bound to bool/unresolved by 5 assignment(s)":
        "read: the harness `ok` in `clearing_paths_check.py` is element 0 of "
        "what the slot named by `getattr(ut, method)()` returned, or the "
        "literal False from the `except` arm. Every return in every slot named "
        "by `SLOTS` is a two-tuple whose element 0 is a bool literal, a "
        "Compare, a BoolOp, a `not` expression, or -- in four of them -- the "
        "literal None. Those four are exactly what the "
        "`if ok is None: decline(...)` guard immediately above the call "
        "exists to intercept, so nothing that is not a bool reaches `check`. "
        "What the sweep unions in is the same name bound inside three slot "
        "methods, which is a union across functions and not a disagreement "
        "about a value. Not executed: a slot drives a real `--pin` in a "
        "sandbox, which this file may not do",
    "bound to bool/unresolved by 6 assignment(s)":
        "read: both of these sites take element 0 of a local helper in "
        "`pose_trust_check.py` -- `describes_selection(...)` and "
        "`unknown_is_not_a_pass(fw)`. Each is annotated `-> tuple[bool, str]`, "
        "each has a single return, and in each that return's element 0 is a "
        "BoolOp over Compares. What the sweep unions instead is the "
        "file-wide name `ok`, bound in six places across four functions, so a "
        "name live in one function is read through the bindings of three "
        "others",
    "bound to bool/unresolved by 7 assignment(s)":
        "read: `ok` is element 0 of `CLAIM_TESTS[rep.claims](...)`. That table "
        "has six entries and every one of them is a `_claim_*` predicate "
        "defined in the same file; each is annotated `-> tuple[bool, str]` and "
        "each returns either the literal False or a two-tuple whose element 0 "
        "is a BoolOp or a Compare, so element 0 is a bool in all seven of "
        "their returns. The sweep reaches the call as `test(...)`, and `test` "
        "is a subscript of a dict literal rather than a name it can follow, so "
        "it declines at the call rather than at the table",
    "bound to non-bool/unresolved by 4 assignment(s)":
        "measured: all three sites were executed against a real MainWindow on "
        "the offscreen platform, never shown and with `viewport.context()` "
        "None after every step. Every element 1 of every triple came back a "
        "`bool`: four triples from `_panel_visibility_report` (False, False, "
        "True, True) and three from `_view_notice_report` (True, True, True) "
        "as shipped; the same two again with the panel notice suppressed; the "
        "same two again with the view's notice text made to claim a figure the "
        "measure does not give; and `_view_notice_report` once more with the "
        "camera dollied in until the view was spent, which is what turned its "
        "first triple's element 1 to False. Seventeen verdict elements, "
        "`bool` every time, both values seen. The values are the geometry of "
        "an unshown window and are not what a shown window would report; the "
        "types are what this entry is about. What the sweep unions in is a "
        "fourth binding of the same `_ok`, thousands of lines further down, "
        "whose value is a call to a name this file does not define",
    "bound to bool/non-bool by 6 assignment(s)":
        "measured and read: one of the two sites is "
        "`ok = bool(inside) and all((state_of[r] == want for r in inside))`, "
        "executed here against a set where every residue matched, one where one "
        "differed, and an empty one: True, False, False, each `bool`, and "
        "`bool()` and `all()` are bool by the language. The other is element 1 "
        "of `_verify_degenerate_guard(...)`, whose one return spells it "
        "`got == want`, a Compare. The mixed union is the file-wide name `ok` "
        "again, bound in six places, one of which is element 2 of a tuple of "
        "pixel fixtures",
    "bound to external by 1 assignment(s)":
        "read: `framing_selection_check.py` binds `degen` and `degen2` to "
        "element 0 of "
        "`opendocking.workbench.framing_selection.selection_is_degenerate(...)`"
        ", which the sweep cannot follow because the call is made through the "
        "module alias `fs` rather than through a bare name. All five of that "
        "function's returns are two-tuples whose element 0 is the literal True "
        "or the literal False, and it is annotated `-> tuple[bool, str]`. Read "
        "in `dock-py/python/opendocking/workbench/framing_selection.py` rather "
        "than executed: the arguments are a live camera basis and a live "
        "MainWindow",
    "bound to unresolved by 1 assignment(s)":
        "read: `ok` is element 1 of the `(name, ok, detail)` triples that "
        "`two_chain_expectations(...)` returns, and the nested `want()` inside "
        "it appends `(name, bool(ok), detail)` -- the coercion is spelled in "
        "the source, so element 1 is a Python bool whatever the expression "
        "handed to `want` evaluated to. The sweep reaches a function whose "
        "return is a name, cannot take the list apart, and so narrows to "
        "nothing",
}

#: How many sites may be neither resolved nor recorded above, without a red.
#: This is today's measurement, not a budget, and it is the count of sites the
#: sweep cannot resolve that no entry above accounts for. That count was 17,
#: each of the 17 a hole in this file rather than a pass, and every one of them
#: is now written down in the table above beside what was found and how it was
#: found. The measurement is therefore **zero**, and zero is what the same
#: derivation produces rather than a number anybody picked: it is the count,
#: not a budget, and it moves down when a case is closed and not up when a run
#: goes well.
#:
#: **Zero is the tightest this ratchet can be set, and it is derived rather
#: than chosen.** The cap existed so the hole would be a number that moves --
#: closing one case lowers it, and a new call the sweep cannot read is a red
#: rather than a gap nobody is measuring. Recording the last unrecorded site is
#: what moved it, which is the thing the comment always said would move it.
#: From here the cap can only be met by writing the new site down: raising the
#: number would put the gap back, and the reverse check above is what stops
#: this table from growing until it covers the tree instead of being re-read.
#:
#: What the fifteen recorded shapes are, and why none of them is a rule this
#: file should widen to reach:
#:
#: * eight are a name. The sweep is not flow-sensitive, so it unions every
#:   binding of a name across every function in the file; the other half of
#:   that eight are a call made through a module alias, which the sweep cannot
#:   follow at all. Resolving either needs scope analysis.
#: * five are element 0 or element 1 of a tuple a local helper returns, where
#:   the helper is unannotated, is not straight-line, or spells that element as
#:   a name the sweep then has to resolve the same way. Resolving them needs an
#:   interprocedural return-type rule.
#: * one is a conditional with one arm the sweep cannot read and one it can,
#:   which is the case `_shape_of` deliberately declines to guess in.
#: * one is a bare call to a local helper whose `with` keeps it from being
#:   straight-line, though it has a single `return` and that is a Compare.
#:
#: None of the fifteen is a site the sweep got wrong. Each is a site it is not
#: built to read, and the entry above says which of them that is and how the
#: answer was arrived at.
MAX_UNRESOLVED_SITES = 0

#: The gate used as the subject of the probe in `check_sweep_has_teeth`. Named
#: because a probe that picked its own target at random would sometimes pick a
#: file with no call site in the shape it needs, and "the probe declined" is a
#: result nobody can act on.
PROBE_GATE = "prep_check.py"


def _external_key(why: str) -> str | None:
    """The table key a site falls under, or None."""
    for head in sorted(EXTERNAL_CALLS, key=len, reverse=True):
        if head in why:
            return head
    return None


def _probe_site(tree) -> tuple[int, int] | None:
    """A `check` call whose verdict is a plain bool expression, in `tree`.

    Located by predicate, never by line: a probe that names a line is a probe
    that stops matching the day the file is edited, and then it either fails
    for the wrong reason or -- worse -- is loosened until it stops failing.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "check" or len(node.args) < 2:
            continue
        verdict = node.args[1]
        if isinstance(verdict, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
            return node.lineno, node.col_offset
    return None


def _splice_tuple(data: bytes, line: int, col: int) -> bytes:
    """Wrap the verdict starting at (line, col) in a two-element tuple."""
    starts = [0]
    for i, b in enumerate(data):
        if b == 0x0A:
            starts.append(i + 1)
    off = starts[line - 1] + col
    tree = ast.parse(data)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "check" \
                and node.lineno == line and node.col_offset == col:
            target = node
            break
    verdict = target.args[1]
    a = starts[verdict.lineno - 1] + verdict.col_offset
    b = starts[verdict.end_lineno - 1] + verdict.end_col_offset
    return data[:a] + b"(" + data[a:b] + b', "")' + data[b:]


def _sweep_in(directory: Path) -> list:
    """Run this file's own `sweep()` over a one-file tree.

    **The sweep under test is this file's own code, not a copy of the
    rules.** Re-implementing the classifier for the probe would be a control
    written twice, and the second copy is the one nobody reads. So the module
    is executed with `__name__` set away from `"__main__"` -- which is what
    keeps `main()` from running -- and its `SCRIPTS` global is pointed at the
    probe directory afterwards. The scope is one file on purpose: the only
    thing the probe can see is the file it mutated.
    """
    ns = {"__name__": "verdict_probe", "__file__": str(Path(__file__).resolve())}
    exec(compile(Path(__file__).read_bytes(), SELF, "exec"), ns)
    ns["SCRIPTS"] = directory
    _by_file, bad, _unknown = ns["sweep"]()
    return [(f, s["line"], s["why"]) for f, s in bad]


def probe_sweep() -> tuple[bool, str]:
    """Prove the sweep can report the defect it is looking for.

    **A sweep that cannot fail is the defect, one level out.** If
    `_shape_of` stopped resolving a verdict -- a renamed builtin, a widened
    rule, a typo in a predicate -- every other check in this file would go
    green on a tree that had grown the tuple shape back, and the whole file
    would be the defect it was written to catch. So the probe is a real gate
    from this tree with its verdict spliced into a tuple, in a private copy,
    and the assertion is that *this* sweep names that one site.

    Returns `(verdict, detail)` rather than recording it, so that the check
    site is an ordinary statement in `main()` and not a `return check(...)`.
    That is not a style preference: `_sites_of` in `check_scripts_declare.py`
    only sees a call that *is* the whole statement, so a `return check(...)`
    here would be a site the census cannot count -- and the count it compares
    against is a generated region that only `--pin` may rewrite. Adding a site
    the census cannot see would leave a red that only the forbidden command can
    clear, so the six check sites in this file are all statements in `main()`
    and the declaration says six.
    """
    src = SCRIPTS / PROBE_GATE
    probe_dir = ROOT / "target" / "verdict_probe"
    if not src.is_file():
        return False, ("%s is not in scripts/, so the sweep's ability to fail "
                       "is untested. Named rather than skipped: a probe that "
                       "cannot be set up is a red, because the alternative is "
                       "an unproven sweep" % PROBE_GATE)
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)
    data = src.read_bytes()
    try:
        at = _probe_site(ast.parse(data))
    except SyntaxError:
        at = None
    if at is None:
        return False, ("no `check(name, <bool expression>, ...)` call in %s "
                       "has a verdict this probe can locate by predicate, so "
                       "the sweep is untested" % PROBE_GATE)
    line, col = at
    clean = probe_dir / "clean"
    mutated_dir = probe_dir / "mutated"
    for d in (clean, mutated_dir):
        d.mkdir()
        shutil.copy2(src, d / src.name)
    (mutated_dir / src.name).write_bytes(_splice_tuple(data, line, col))

    before = _sweep_in(clean)
    after = _sweep_in(mutated_dir)
    return (not before and len(after) == 1 and after[0][0] == PROBE_GATE), (
        "against an unmutated copy of %s: %d non-bool verdict(s), which is "
        "what a clean tree has to read. After wrapping one real verdict in a "
        "two-element tuple at line %d of that same file: %d. %s. Both numbers "
        "are the result -- the second is what makes the first a measurement "
        "rather than a green print"
        % (PROBE_GATE, len(before), line, len(after),
           ("named %s:%d, %s" % after[0]) if after else
           "and it found nothing, which is the failure this check exists for"),
    )


def _load_check(path: Path):
    """A gate's `check`, lifted out of its source and executed on a stub.

    `check_scripts_declare.py` runs its whole audit at import, so its `check`
    cannot be imported by a sibling gate, and the stub is what stands in for
    the module-level state it touches: `CHECKS`, `FAILURES`, `record` and
    `print`. Returns `(function, failures)`; `failures` is the list the loaded
    `check` appends a name to, which is the observable the assertions below
    are made on rather than the return value -- a `check` that returned the
    non-bool unchanged would still be refusing, and one that returned False for
    every input would pass a return-value test.
    """
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError):
        return None, None
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if fn is None:
        return None, None
    failures, recorded = [], []
    ns = {"print": lambda *a, **k: None,
          "CHECKS": 0,
          "FAILURES": failures,
          "record": lambda tag, n, detail="": recorded.append((tag, n))
          or (tag == "ok")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path.name, "exec"), ns)
    return ns["check"], failures


def house_verdicts() -> tuple[bool, str]:
    """Both house `check()`s refuse a non-bool, in both directions.

    Six probes, and the two boolean ones are the control: a `check` that
    refused every value it was given would satisfy "a non-bool is refused" and
    be useless. So `False` must be an ordinary failure, `True` an ordinary
    pass, and `()`, `""`, `None` and `(True, "")` must each be a failure of
    their own. The first three are the falsy half -- the live false reds -- and
    the last is the shape that shipped here.
    """
    problems, ran = [], []
    for name in ("check_scripts_declare.py", SELF):
        f, failures = _load_check(SCRIPTS / name)
        if f is None:
            problems.append("%s: no check() could be loaded" % name)
            continue
        got = {}
        for value in (False, True, (), "", None, (True, "")):
            before = len(failures)
            ret = f(value, "probe")
            got[value] = (len(failures) > before, ret)
            if value is False or value is True:
                continue
        if got[False][0] is not True:
            problems.append("%s: a real False was not recorded as a failure" % name)
        if got[True][0] is not False:
            problems.append("%s: a real True was recorded as a failure" % name)
        for value in ((), "", None, (True, "")):
            if got[value][0] is not True:
                problems.append("%s: %r was accepted as a verdict (returned %r)"
                                % (name, value, got[value][1]))
        ran.append("%s: False -> %s, True -> %s, and (), \"\", None and "
                   "(True, \"\") all -> failure"
                   % (name, got[False][0], got[True][0]))
    if problems:
        return False, "; ".join(problems)
    return True, (
        "Both were executed, not read, each against a stub namespace, and each "
        "got the same six probes. %s. The observation is the name reaching the "
        "gate's own failure list rather than the return value, so a check that "
        "refused everything would fail the two boolean probes above instead of "
        "passing them" % " | ".join(ran))


def main() -> int:
    section("every check-like call site, and the shape of its verdict")
    by_file, bad, unknown = sweep()
    total = sum(len(v) for v in by_file.values())
    # The partition the whole file is about, measured rather than described:
    # a falsy non-bool is a live false red, a truthy one is a check that
    # cannot fail, and neither is visible in the other's output.
    falsy = [(f, s) for f, s in bad if s["falsy"] is True]
    truthy = [(f, s) for f, s in bad if s["falsy"] is False]
    unsorted = [(f, s) for f, s in bad if s["falsy"] is None]
    counts = {}
    for sites in by_file.values():
        for s in sites:
            counts[s["shape"]] = counts.get(s["shape"], 0) + 1

    def _where(rows):
        return "; ".join("%s:%d %s" % (f, s["line"], s["why"]) for f, s in rows) or "none"

    check(
        not bad,
        "no check site passes a verdict that is not a bool",
        "%d call site(s) across %d file(s) in scripts/, classified as %s. %d "
        "pass a verdict that is not a bool, and the partition is the result "
        "rather than the count: %d FALSY, which is a live false red -- a gate "
        "reporting a failure over a value that was never a verdict, and "
        "nothing in its output says which value was wrong (%s); %d TRUTHY, "
        "which passes when it means pass and so cannot fail (%s); %d whose "
        "falsiness this sweep cannot decide (%s)"
        % (total, len(by_file), sorted(counts.items()), len(bad),
           len(falsy), _where(falsy), len(truthy), _where(truthy),
           len(unsorted), _where(unsorted)),
    )

    keys = {}
    for fname, s in unknown:
        k = _external_key(s["why"])
        keys.setdefault(k, []).append("%s:%d" % (fname, s["line"]))
    recorded = sum(len(v) for k, v in keys.items() if k is not None)
    bare = [("%s:%d %s" % (f, s["line"], s["why"])) for f, s in unknown
            if _external_key(s["why"]) is None]
    check(
        len(bare) <= MAX_UNRESOLVED_SITES,
        "every verdict the sweep cannot read is either recorded or capped",
        "%d site(s) the sweep could not resolve. %d are recorded in "
        "EXTERNAL_CALLS, each with what was found written beside it and how "
        "that was found; %d are neither recorded nor resolved, against a cap of "
        "%d. The cap is the sweep's own blindness as a number that moves -- "
        "closing one case lowers it, and a new call the sweep cannot read and "
        "nobody has looked at is a red here rather than a gap nobody is "
        "measuring. The unrecorded %d: %s"
        % (len(unknown), recorded, len(bare), MAX_UNRESOLVED_SITES, len(bare),
           " | ".join(bare)),
    )
    stale = [h for h in EXTERNAL_CALLS if h not in keys]
    check(
        not stale,
        "every EXTERNAL_CALLS entry still matches a site it is here for",
        "%d entr(ies) in the table, covering %d site(s) this run. An entry that "
        "stopped matching is a red rather than a record nobody re-reads: a "
        "stale key is a note about a call that is no longer there, and it would "
        "otherwise sit in the table looking like coverage. Unmatched: %s"
        % (len(EXTERNAL_CALLS), recorded, stale or "none"),
    )

    section("the probe, and the house verdicts")
    ok, detail = probe_sweep()
    check(ok, "the sweep can report a verdict that is not a bool", detail)
    ok, detail = house_verdicts()
    check(ok, "the house verdicts refuse a non-bool instead of coercing it",
          detail)

    section("the tally")
    check(
        CHECKS + 1 == EXPECTED_CHECKS,
        "this file ran the number of checks it declares",
        "%d result(s) were recorded before this one, and this one is the %dth, "
        "against a declared %d. The count is taken before this check "
        "increments it, so the declared number is the number of checks a run "
        "produces and not the number it had got to -- a slot that stopped "
        "running is a red here rather than a smaller file wearing the same pin"
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
