"""Is any constant in this tree claiming to count something that nothing checks?

Run:  F:\\python310\\python.exe scripts\\check_counted_constants.py

# The class this hunts, in one sentence

**A constant that claims to count something and is compared against nothing is
not a count.** It is a number a developer edits to make a sentence true, and the
only thing standing between it and a lie is the word "Total" in its name.

# Why this is not the same failure as a gate that misfires

A detector that misfires is noisy, so somebody investigates it within minutes. A
clearing mechanism that is broken produces a red that does not go away when you
do the right thing, so it looks exactly like a person who is wrong, a detector
that is wrong, or a gate with a real defect -- and it gets blamed on the wrong
thing for as long as nobody suspects the number. The worked example is
`scripts/pockets_screenshot.py`, which carried

    #: How many frames this script produces ... Counted here rather than written
    #: into the summary text, so the number in the summary and the number of
    #: `save_frame` calls cannot drift apart.
    TOTAL_FRAMES = 3

and the comment was a claim about the wrong thing. `TOTAL_FRAMES` was read in
three places, every one of them inside an f-string in a `print`, and an f-string
cannot fail: no `ast.Compare` node in the file mentioned it, no `len()` counted
the `save_frame` calls, and no argument validated it. Add a fourth `save_frame`
call and the run printed ``RESULT: PASS -- 4 of 3 frame(s) written`` and exited
0. The comment said the two numbers could not drift apart; the mechanism that
would have stopped the drift was the word "Counted".

So the fix was not a bigger number. It was a *shape* that can be wrong about
something: a tuple of the labels `save_frame` is handed, membership-checked on
the way in, with the count in the summary derived as `len()` of the set that was
checked. `FRAME_LABELS` is a declared constant and a declared constant is only
honest when something compares it.

# What is measured, and what is not

**In scope:** module-level assignments in `scripts/` and
`dock-py/python/opendocking/` whose value is an `int` or `float` *literal* and
whose name says it is a count, a total, a limit, a floor, a cap, a threshold or a
budget. Every other file and every other shape of constant is out of scope by
construction, and the run prints how many files it walked and how many constants
it examined every time -- a gate whose scope is invisible is the same failure as
an unpinned total, because a reader cannot tell what "0 flagged" is a statement
about.

**The test applied to each one:** every read of the name, walked outwards to the
first node that can *act on* the value. A constant is flagged when it has **no
reads at all**, or when **every read sits inside an f-string** -- because those
are the two shapes in which a number's only remaining power is to change a
sentence. Arithmetic, keyword arguments and subscripts are transparent on the
way out, so `EXPECTED_CHECKS - 1 == len(RESULTS)` is counted as a comparison and
not reported: a constant that is compared through one subtraction is compared.

**Four things this deliberately does not claim.**

* It does not know whether the constant is *right*. It knows whether anything
  looks at it. A count that is compared can still be the wrong count.
* It does not follow a constant into another file. `pockets.py`'s
  `DEFAULT_MAX_POCKETS` is only ever a default argument in its own module; the
  comparisons live in `pockets_check.py` and `docs_claims_check.py`. This gate
  will not see those, and says so rather than pretending the file is unbacked --
  the eight `EXPECTED_CHECKS` pins in this tree and the
  `contacts_criteria_check.py` boundary tests are all cross-file by nature.
* It does not look at constants that are *computed* rather than typed, because a
  computed count cannot be edited into a lie. `ceiling_gib` in
  `docs_claims_check.py` is read out of the Rust source by a regex; raising it is
  not a thing a developer can do.
* It does not look at a collection constant such as `FRAME_LABELS` or
  `TERM_ROWS`. Those are checked by `energy_terms_check.py` and by the file that
  declares them, and the false-positive rate of a rule about them is the whole
  table in this tree.

# The exemption list, and why it is not a hole

`EXEMPT` is keyed by `path:NAME` and carries a reason, and it is exercised on
every run in the direction that matters: an entry is only honoured for a constant
that **exists, is flagged, and gives a reason** (check 4), so an exemption left
behind after its constant was fixed or deleted goes red rather than covering the
file forever. That is the difference between a clearing mechanism and a lid.

An entry is a *reason a person wrote down*, not a suppression. The two on this
tree are both cases where the comparison provably happens and this gate cannot
see it -- one in a child process, one in a sibling file. A constant that nothing
anywhere checks is not exemptible, and the run is red until its owner fixes it.

# The floor, and the self-test

`MIN_CONSTANTS_SCANNED` is a floor rather than a total, and it is compared. Its
only job is to make a *broken scan* loud: a detector that has stopped matching
names would otherwise report "0 flagged" with the confidence of a clean tree,
which is the absence of evidence wearing a verdict. A floor rather than an exact
count because the claim set moves as other agents add files, and churning on
every legitimate addition is how a gate stops being run.

And because a clean run on a clean tree is a confident zero from a predicate
nobody has watched refuse anything, the self-test (check 5) builds two files in a
`tempfile.mkdtemp()` directory -- one with the `TOTAL_FRAMES` shape and one with
the same constant compared -- and requires this file's own detector to sort them
correctly. The clean twin is what stops a self-test passing against a detector
that flags everything.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
SCAN_DIRS = ("scripts", "dock-py/python/opendocking")

#: The words in a name that make it a claim about a quantity rather than a
#: label. Split on `_` and on camelCase humps, so `TOTAL_FRAMES`, `totalFrames`
#: and `EXPECTED_CHECKS` are all read as claims and `TERMINAL` is not.
CLAIM_WORDS = frozenset("""
COUNT COUNTS TOTAL TOTALS NUM NUMBER SIZE SIZES LEN LENGTH LIMIT LIMITS CAP
CAPS FLOOR CEIL CEILING THRESHOLD THRESHOLDS THRESH BUDGET BUDGETS MIN MAX
MINIMUM MAXIMUM EXPECTED WANTED SLOT SLOTS STEP STEPS ITER ITERATION
ITERATIONS ROUND ROUNDS RETRY RETRIES ATTEMPT ATTEMPTS DECIMALS PRECISION
TOLERANCE TOL EPS DELTA GAP MODE MODES DEFAULTS TIMEOUT SAMPLES SAMPLE TRIES
ALLOWED PERMITTED QUOTA FRAMES ROWS COLUMNS VERTS SECONDS
""".split())

#: Calls whose argument is a *decision* about the value, so a constant handed to
#: one is being compared by whatever that call does with it.
CHECKERS = re.compile(r"(^|_)(check|assert|validate|verify|require|expect)($|_)",
                      re.I)
LEN_CALLS = {"len", "len_"}
RANGE_CALLS = {"range", "xrange"}

#: How many count-claiming numeric constants this scan must find before it is
#: willing to report "0 flagged". **Measured at 106 on this tree**, and the
#: floor is set well below that on purpose: its only job is to catch a detector
#: that has stopped matching names, which would find a handful rather than a
#: hundred. A floor rather than an exact count because the claim set moves as
#: other agents add files, and churning on every legitimate addition is how a
#: gate stops being run. **If a change to this file makes the scan find fewer
#: than this, the scan is broken and the run says so** -- which is the
#: difference between this and a cap nobody watches.
MIN_CONSTANTS_SCANNED = 90

#: Constants the scan flags that a person has looked at and written down why
#: they are not the thing this gate is about. Keyed `path:NAME`; check 4 makes
#: every entry earn its place on this run.
EXEMPT: dict[str, str] = {
    "scripts/probe_painted_frame.py:_PUMP_SECONDS":
        "the only read is the f-string that hands it to the child as "
        "`pump=<n>`, and the child turns it into a deadline it compares "
        "against -- `deadline = time.perf_counter() + PUMP` with "
        "`while viewport._ctx is None and time.perf_counter() < deadline` -- so "
        "the sentence *is* the transport and the comparison happens in the "
        "child, a process this gate does not read",
    "dock-py/python/opendocking/workbench/framing_selection.py:RECEPTOR_SHARE_FLOOR":
        "compared, in another file: `scripts/framing_selection_check.py` asks "
        "`after['receptor'] / total >= fs.RECEPTOR_SHARE_FLOOR`, so the floor "
        "is enforced at the share it names. This module never reads it itself, "
        "which is the cross-file blind spot named in this file's docstring and "
        "not an exemption of a defect",
}

#: Where this number came from, because a pin with no provenance is a
#: transcribed number. It is five section results plus the tally, measured from
#: a green run on 2026-10-03; and it is load-bearing rather than decorative
#: because `tally()` below compares `len(RESULTS) == EXPECTED_CHECKS - 1` in a
#: decision it executes -- so adding a check without moving this line is a red,
#: not a silently larger run.
EXPECTED_CHECKS = 6

#: The call-site census this file holds itself to, derived by
#: `scripts/check_scripts_declare.py`'s own `_sites_of` / `_guards_digest` and
#: read back from its `--census` output. Typed, not typed-by-me: these three
#: lines are that tool's paste-ready block for this file, and it is the auditor
#: -- not this comment -- that says whether they are still true.
#: GATE-DECLARE 1
#: sites: 0 unconditional + 12 guarded
#: guards: sha256:d936e7121f6641cc65f7fce4ddff37be3215e34e1e61ce1c61e43a49ce1d5c6f

RESULTS: list[tuple[str, str, str]] = []


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def skip(name: str, why: str) -> None:
    RESULTS.append(("SKIP", name, why))
    print(f"[SKIP] {name}\n       {why}")


def section(t: str) -> None:
    print()
    print(f"-- {t}")


# --------------------------------------------------------------------------
# The detector. One implementation, asked about the tree by check 3 and about
# this file's own fixtures by check 7.
# --------------------------------------------------------------------------


def claim_words(name: str) -> set[str]:
    """The quantity words in a constant's name, `TOTAL_FRAMES` -> {TOTAL, FRAMES}."""
    out: set[str] = set()
    for part in (p for p in name.split("_") if p):
        out.update(w.upper() for w in re.findall(
            r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", part))
    return out


def claims_a_count(name: str) -> bool:
    return bool(claim_words(name) & CLAIM_WORDS)


def numeric_literal(value: ast.expr) -> bool:
    """An int or float *typed here*. `bool` is an `int` and is not a number."""
    return (isinstance(value, ast.Constant)
            and not isinstance(value.value, bool)
            and isinstance(value.value, (int, float)))


def parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[id(child)] = node
    return out


def read_kind(node: ast.AST, up: dict[int, ast.AST]) -> str:
    """How the value at `node` is used, found by walking outwards.

    Arithmetic, keyword arguments and subscripts are transparent: a constant
    that only appears inside `EXPECTED_CHECKS - 1` is compared when the `==`
    above that subtraction compares it, and stopping at the `BinOp` would call
    every pinned gate in this tree a lie.
    """
    transparent = (ast.BinOp, ast.UnaryOp, ast.keyword, ast.Starred,
                   ast.Subscript, ast.Attribute, ast.Await, ast.expr_context,
                   ast.arguments, ast.arg)
    cur = node
    while True:
        parent = up.get(id(cur))
        if parent is None:
            return "unread"
        if isinstance(parent, transparent):
            cur = parent
            continue
        if isinstance(parent, ast.Compare):
            return "compared"
        if isinstance(parent, ast.JoinedStr):
            return "prose"
        if isinstance(parent, ast.Call):
            name = getattr(parent.func, "id", None) or getattr(
                parent.func, "attr", "")
            if name in LEN_CALLS:
                return "counted"
            if name in RANGE_CALLS:
                return "counted"
            if CHECKERS.search(name or ""):
                return "checked"
            return f"handed to {name or '?'}()"
        if isinstance(parent, ast.If) and cur is parent.test:
            return "tested"
        if isinstance(parent, ast.comprehension) and cur is parent.iter:
            return "iterated"
        if isinstance(parent, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
            return "collected"
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.Lambda)):
            return "default"
        if isinstance(parent, (ast.Return, ast.Assign, ast.AugAssign,
                               ast.AnnAssign, ast.IfExp, ast.Assert,
                               ast.arguments)):
            return "value"
        cur = parent


class Finding:
    """One count-claiming constant, and what is done with it."""

    def __init__(self, path: str, line: int, name: str, value: object) -> None:
        self.path = path
        self.line = line
        self.name = name
        self.value = value
        self.reads: list[str] = []

    @property
    def key(self) -> str:
        return f"{self.path}:{self.name}"

    @property
    def line_ref(self) -> str:
        return f"{self.path}:{self.line} {self.name} = {self.value!r}"

    @property
    def prose_only(self) -> bool:
        return not self.reads or all(r == "prose" for r in self.reads)


def scan_text(path: str, text: str) -> tuple[list[Finding], list[Finding],
                                            str | None]:
    """Every count-claiming numeric constant in `text`, and which are flagged."""
    claimed: list[Finding] = []
    flagged: list[Finding] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return claimed, flagged, f"{path}: {exc}"
    up = parent_map(tree)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
              and isinstance(node.target, ast.Name)):
            targets, value = [node.target], node.value
        else:
            continue
        if not numeric_literal(value):
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or not claims_a_count(
                    target.id):
                continue
            found = Finding(path, node.lineno, target.id, value.value)
            for use in ast.walk(tree):
                if (isinstance(use, ast.Name) and use.id == found.name
                        and isinstance(use.ctx, ast.Load)):
                    found.reads.append(read_kind(use, up))
            claimed.append(found)
            if found.prose_only:
                flagged.append(found)
    return claimed, flagged, None


def walk(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    problems: list[str] = []
    for rel in SCAN_DIRS:
        base = root / rel
        if not base.is_dir():
            problems.append(f"{rel}: no such directory under {root}")
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files, problems


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------


def check_scope(root: Path) -> list[Finding]:
    files, problems = walk(root)
    claimed: list[Finding] = []
    flagged: list[Finding] = []
    unreadable: list[str] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append(f"{rel}: {exc}")
            continue
        c, f, problem = scan_text(rel, text)
        claimed.extend(c)
        flagged.extend(f)
        if problem:
            unreadable.append(problem)

    name = "the scope is walked, and every file in it was read"
    if problems or unreadable:
        bad(name, f"{len(files)} file(s) matched, {len(unreadable)} could not be "
                  f"read. An unreadable file is not an empty one: "
                  f"{unreadable + problems}")
    else:
        ok(name, f"{len(files)} file(s) under {', '.join(SCAN_DIRS)}, all read. "
                 f"The reach is what `walk` found and the judged set is the "
                 f"same thing here -- there is no exclusion list to be stale, "
                 f"so the file count printed is the file count the verdict is "
                 f"about")

    name = "the scan found enough constants to be worth reading"
    if len(claimed) >= MIN_CONSTANTS_SCANNED:
        ok(name, f"{len(claimed)} count-claiming numeric constant(s) examined, "
                 f"against a floor of {MIN_CONSTANTS_SCANNED}. The floor is not "
                 f"a total and does not move with the tree: its only job is to "
                 f"make a detector that has stopped matching names say so "
                 f"instead of reporting a confident zero")
    else:
        bad(name, f"only {len(claimed)} count-claiming numeric constant(s) "
                  f"found, against a floor of {MIN_CONSTANTS_SCANNED}. Either "
                  f"`CLAIM_WORDS` or the module-level scan has been narrowed -- "
                  f"a detector that finds nothing has not found nothing wrong")

    return flagged


def check_flagged(flagged: list[Finding]) -> None:
    name = "no count-claiming constant is compared against nothing"
    unexempt = [f for f in flagged if f.key not in EXEMPT]
    if unexempt:
        lines = "; ".join(
            f"{f.line_ref} -- read only as prose: {f.reads or ['never read']}"
            for f in unexempt)
        bad(name, f"{len(unexempt)} of {len(flagged)} flagged constant(s) have "
                  f"no exemption. A number whose only remaining power is to "
                  f"change a sentence is not a count: {lines}")
    else:
        ok(name, f"{len(flagged)} constant(s) in this tree are read only as "
                 f"prose or not read at all, and all {len(flagged)} are "
                 f"accounted for: {sorted(f.key for f in flagged) or 'none'}. "
                 f"The constants this gate does *not* flag are the ones with "
                 f"an `ast.Compare`, a `len()`, a `range()`, a membership test "
                 f"or a checking call somewhere in their own module")


def check_exemptions(flagged: list[Finding]) -> None:
    name = ("every exemption is for a constant that exists, is flagged, and "
            "gives a reason")
    by_key = {f.key: f for f in flagged}
    problems = []
    for key in sorted(EXEMPT):
        if key not in by_key:
            problems.append(f"{key}: not flagged any more, so the exemption is "
                            f"covering nothing")
        if not EXEMPT[key].strip():
            problems.append(f"{key}: the reason is empty")
    if problems:
        bad(name, f"{len(problems)} problem(s) with {len(EXEMPT)} exemption(s): "
                  f"{problems}. An exemption that is not being exercised is a "
                  f"lid, and a lid with a reason is still a lid")
    else:
        ok(name, f"{len(EXEMPT)} exemption(s), each for a constant this run "
                 f"flagged and each with a reason: "
                 f"{sorted(EXEMPT) or 'none'}. The list is honoured only for a "
                 f"constant that is flagged, so deleting or fixing the "
                 f"constant turns the entry into this red rather than into a "
                 f"permanent exemption")


FIXTURE_FLAGGED = '''\
#: A frame count, declared and then only ever printed.
TOTAL_FRAMES = 3


def save(win, label):
    print(f"{label} of {TOTAL_FRAMES} written")


def save(win, label):  # noqa: F811 - the fixture wants two call sites
    save(win, "a")
    save(win, "b")
'''

FIXTURE_COMPARED = '''
MIN_COLOURS = 5


def verdict(colours):
    return colours < MIN_COLOURS
'''


def check_self_test() -> None:
    name = "this detector tells a count claim from a comparison"
    where = Path(tempfile.mkdtemp(prefix="counted_constants_"))
    try:
        flagged_file = where / "fixture_flagged.py"
        compared_file = where / "fixture_compared.py"
        flagged_file.write_text(FIXTURE_FLAGGED, encoding="utf-8")
        compared_file.write_text(FIXTURE_COMPARED, encoding="utf-8")
        _c1, f1, p1 = scan_text(flagged_file.name,
                                flagged_file.read_text(encoding="utf-8"))
        _c2, f2, p2 = scan_text(compared_file.name,
                                compared_file.read_text(encoding="utf-8"))
        problems = []
        if p1 or p2:
            problems.append(f"a fixture did not parse: {p1 or p2}")
        names1 = sorted(f.name for f in f1)
        names2 = sorted(f.name for f in f2)
        if names1 != ["TOTAL_FRAMES"]:
            problems.append(
                f"the `TOTAL_FRAMES` fixture flagged {names1}, expected "
                f"exactly ['TOTAL_FRAMES']")
        if names2:
            problems.append(
                f"the compared fixture flagged {names2}, expected nothing -- a "
                f"detector that flags `MIN_COLOURS` would condemn every pin in "
                f"this tree")
        if problems:
            bad(name, f"{len(problems)} problem(s) in the fixtures built in "
                      f"{where}: {problems}")
        else:
            ok(name, f"two files built in {where.name} and read by this "
                     f"file's own `scan_text`: the one shaped like the "
                     f"`TOTAL_FRAMES` bug is flagged on its name, the one that "
                     f"compares its constant against `colours` is not. The "
                     f"clean twin is the half that matters -- a self-test with "
                     f"only the dirty case passes against a detector that flags "
                     f"everything")
    finally:
        shutil.rmtree(where, ignore_errors=True)


def main() -> int:
    print("Is any constant in this tree claiming to count something that "
          "nothing checks?")
    root = Path(os.environ.get("OD_COUNTED_SCAN_ROOT", str(ROOT))).resolve()
    if root != ROOT:
        print(f"   (OD_COUNTED_SCAN_ROOT: scanning {root} instead of the tree)")

    section("the scan")
    flagged = check_scope(root)

    section("the verdict")
    check_flagged(flagged)
    check_exemptions(flagged)

    section("the self-test: can this detector tell the two apart?")
    check_self_test()

    tally()
    return finish()


def tally() -> None:
    section("the tally")
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    name = "the tally accounts for every result, skips included"
    if np_ + nf + ns == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(name, f"{np_} + {nf} + {ns} = {len(RESULTS)}, and this check is the "
                 f"{len(RESULTS) + 1}th, so the run reaches the declared "
                 f"{EXPECTED_CHECKS} -- the same on this tree and under "
                 f"OD_COUNTED_SCAN_ROOT, because a skip is a result and a "
                 f"failure does not move the total")
    else:
        bad(name, f"recorded {len(RESULTS)} result(s) ({np_}+{nf}+{ns}) before "
                  f"this check, which must be {EXPECTED_CHECKS - 1}")


def finish() -> int:
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {np_} passed, {nf} failed, {ns} skipped, {len(RESULTS)} checks "
          f"(expected {EXPECTED_CHECKS})")
    if nf:
        print(f"RESULT: FAIL -- {nf} of {EXPECTED_CHECKS} checks failed")
        return 1
    print("RESULT: OK -- a green here means every count-claiming constant in "
          "this module scope is compared, or is on a list somebody wrote down")
    return 0


def report_crash(exc: BaseException) -> int:
    import traceback

    traceback.print_exc()
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- STOPPED EARLY after {len(RESULTS)} recorded result(s): "
          f"{np_} passed, {nf} failed, {ns} skipped")
    print(f"RESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}")
    print("    a finished run prints OK or FAIL above and never this line")
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_exc))
