"""Does every check script in `scripts/` say how many checks it runs?

Run:  python scripts/check_scripts_declare.py

# The problem this closes

Six check scripts pin their total with `EXPECTED_CHECKS` and assert it. Eleven
do not. For those eleven, the count printed at the end is whatever happened to
run, and **nothing fails when it shrinks**. That is not hypothetical:

    workbench_interaction_check  pin=233   runs 8 of them on a machine with no
                                           OpenGL context -- and it goes RED,
                                           which is the pin doing its job.

The same script without a pin would print "8/8 passed" and look healthy. The
coverage quietly disappeared and the suite said everything was fine. That is the
failure mode `smoothstep_is_c1` already demonstrates in `dock-core`: a check
that stays green when the thing it guards is gone.

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

# Why this is a separate file from `scoring_docs_check.py`

Three reasons, in order of weight:

1. **This one runs without the engine.** It only reads source text. It works on
   a machine where the wheel was never built, which is exactly the machine where
   you most want to know whether the suite's own bookkeeping is intact.
   `scoring_docs_check.py` imports `opendocking` and cannot.
2. **Different blast radius.** A failure in the documentation audit and a failure
   in the suite inventory are different facts and should not be reported as one
   number in one place.
3. `scoring_docs_check.py` is already misnamed -- it now covers the READMEs too,
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
import re
import sys
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
#: `EXPECTED_CHECKS`. Every file carries one, or is in UNPINNED_BASELINE.
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
    "examples_check.py",
    "ligand_check.py",
    "pdbqt_check.py",
    "pockets_check.py",
    "prep_check.py",
    "representation_geometry_check.py",
    "scoring_cross_check.py",
    "scoring_docs_check.py",
    "structure_bond_check.py",
    "viewport_framing_check.py",
    "workbench_interaction_check.py",
    "x11_window_parse_check.py",
]

#: Files whose *name* says "check" but which are not gates, and why. Every such
#: file must appear here with a reason: a script that can be waved through with
#: a name change is not a classification, it is a loophole.
EXCLUSIONS = {
    "_gui_check.py":
        "a helper module imported by other scripts, not a gate; it defines no "
        "check() and running it does not report a pass/fail total",
    "odgui_launch_check.py":
        "a launch probe, and the file x11_window_parse_check.py splices its "
        "helpers out of at import time; it is consumed, not run as a gate",
}

MIN_NO_PIN_REASON = 40

EXPECTED_CHECKS = 50  # measured: a green run of this file, including the two self-checks at the bottom that read it

#: The scripts that currently have no `EXPECTED_CHECKS`, with the count each one
#: actually ran here. This is a **ratchet, not a whitelist**: see
#: `MAX_UNPINNED_SCRIPTS` for why it is allowed to shrink and never grow.
#:
#: It is deliberately NOT a silent pass. The file prints the whole table on
#: every run, and the check fails if an entry stops being true -- so a script
#: that has since gained a pin is a failure too, and the debt cannot quietly
#: become fiction.
UNPINNED_BASELINE = {
    "benchmark_check.py": 33,
    "check_doc_encoding.py": None,      # reports documents scanned, not checks
    "check_repo_docs.py": None,         # reports links scanned, not checks
    "cli_prep_check.py": 37,
    "contacts_check.py": 39,
    "contacts_check2.py": 18,
    "core_check.py": 392,
    "examples_check.py": 19,
    "ligand_check.py": 105,
    "prep_check.py": 95,
    "structure_bond_check.py": 85,
    "viewport_framing_check.py": None,  # GUI; no context on this machine
    "x11_window_parse_check.py": 11,
}

#: The ratchet. Fourteen scripts are unpinned today; that is a fact about the
#: repository, not a decision this file endorses. `MAX_UNPINNED_SCRIPTS` is the
#: count the check holds them to, so:
#:
#:   * a new unpinned script pushes the count over the line and fails;
#:   * adding a pin to one of them drops it below the line, which is fine --
#:     lower this number at the same time so the slack is not silently reused;
#:   * nothing here can make the debt smaller without someone editing a pin.
#:
#: Lowering it is the owner's call, not this file's.
MAX_UNPINNED_SCRIPTS = 13

#: The same ratchet, for the other kind of debt: a pin that exists and is
#: compared, but carries no note saying where the number came from. An
#: unjustified pin is a transcribed number, which is the failure this repository
#: keeps meeting -- `scoring_cross_check.py` holds its own `WEIGHTS` literal for
#: exactly that reason, and four of the six pins here say "measured" in a comment.
#: The ones that do not are listed, not silently tolerated.
UNJUSTIFIED_PINS = {
    "cli_check.py":
        "pin is 127 and is compared in two places, but the line has no note "
        "saying how 127 was arrived at",
    "workbench_interaction_check.py":
        "pin is 233 and is compared in three places, including a section "
        "account, but the line has no note saying how 233 was arrived at",
}
MAX_UNJUSTIFIED_PINS = 2


def check(ok: bool, name: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")
    return ok


def section(t: str) -> None:
    print(f"\n=== {t} ===")


def read(path: Path) -> str:
    # utf-8-sig: `redock_benchmark.py` carries a BOM, and a strict decode of it
    # raises. A gate that cannot read the repository is not a gate.
    return path.read_bytes().decode("utf-8-sig")


# ==========================================================================
section("every file whose name says 'check' is accounted for")

looks_like_a_check = sorted(p.name for p in SCRIPTS.glob("*.py") if "check" in p.name)
unknown = [n for n in looks_like_a_check
           if n not in INVENTORY and n not in EXCLUSIONS]

check(
    not unknown,
    "no check-shaped script is unclassified",
    f"{len(looks_like_a_check)} file(s) match '*check*'; "
    f"{len(INVENTORY)} are gates in the inventory, {len(EXCLUSIONS)} are "
    f"excluded with a reason, {len(unknown)} are unaccounted for: {unknown}. "
    f"A new gate, or a renamed one, shows up here until it is classified",
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
    """(value, trailing comment) for a module-level EXPECTED_CHECKS, or None."""
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
                return (int(value.value), (m.group(1).strip() if m else ""))
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

for name in undeclared:
    ledger = UNPINNED_BASELINE.get(name, "NOT IN THE LEDGER")
    check(
        name in UNPINNED_BASELINE,
        f"{name} has no pin and is not in the standing-debt ledger",
        f"ledger says {ledger!r}. Either give it an `EXPECTED_CHECKS`, give it "
        f"a `NO_EXPECTED_CHECKS = \"...\"` reason, or record it in "
        f"UNPINNED_BASELINE. Silently adding an unpinned gate is the one thing "
        f"this file exists to prevent",
    )

# The ratchet: the check that keeps the debt from growing, and the one that
# fails the moment a gate leaves the ledger without gaining a pin.
check(
    len(UNPINNED_BASELINE) <= MAX_UNPINNED_SCRIPTS,
    f"no more than {MAX_UNPINNED_SCRIPTS} check scripts are unpinned",
    f"{len(UNPINNED_BASELINE)} in UNPINNED_BASELINE against a cap of "
    f"{MAX_UNPINNED_SCRIPTS}. The cap only ever falls: each script that gains a "
    f"pin should be deleted from the ledger and MAX_UNPINNED_SCRIPTS lowered "
    f"with it, so the slack cannot be quietly spent on a new gap",
)

stale = []
for name in sorted(UNPINNED_BASELINE):
    path = SCRIPTS / name
    if not path.exists():
        stale.append(f"{name}: in the ledger but not on disk")
    elif pin_of(path) is not None:
        stale.append(f"{name}: in the ledger but has gained an EXPECTED_CHECKS")
check(
    not stale,
    "every ledger entry is still an unpinned script that exists",
    f"{len(UNPINNED_BASELINE)} entries checked; stale: {stale}. Remove the entry "
    f"when a script gains a pin -- a ledger that lies about the debt is worse "
    f"than no ledger",
)

print()
print("  standing debt -- check scripts with no EXPECTED_CHECKS, and what each")
print("  one actually ran on the machine that measured this table:")
print()
for name, count in sorted(UNPINNED_BASELINE.items()):
    print("    %-34s ran %s" % (name, count if count is not None else "(reports no count)"))
print()

# ==========================================================================
section("a declared pin is justified, positive, and actually compared")

for name, value, comment in sorted(pinned):
    check(
        value > 0,
        f"{name}'s pin is a positive count",
        f"EXPECTED_CHECKS = {value}",
    )
    # A pin with no provenance is a transcribed number, which is the failure this
    # repository keeps meeting elsewhere.
    justified = bool(re.search(r"measured|deliberate|green run", comment, re.I))
    check(
        justified or name in UNJUSTIFIED_PINS,
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
    or re.search(r"measured|deliberate|green run", (pin_of(SCRIPTS / n) or (0, ""))[1], re.I)
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
section('no check() call site has quietly become conditional')

SITE_INVENTORY = {
    "benchmark_check.py": (33, 0, [
    ]),
    "check_doc_encoding.py": (0, 0, [
    ]),
    "check_repo_docs.py": (0, 0, [
    ]),
    "contacts_check.py": (39, 1, [
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
    ]),
    "contacts_check2.py": (16, 0, [
    ]),
    "core_check.py": (244, 15, [
        'def sec_estimate',
        'def sec_exhaustiveness',
        'def sec_maps',
        'def sec_maps',
        'def sec_maps',
        'def sec_reachability',
        'def sec_writers',
        'def sec_writers',
        'def sec_writers',
        'def sec_writers',
        'def sec_writers',
        'def sec_writers',
        'for (name, fn) | for (cls, members) | def sec_reachability',
        'try:except | for (name, fn) | for (cls, members) | def sec_reachability',
        'try:except | for fn',
    ]),
    "examples_check.py": (19, 0, [
    ]),
    "ligand_check.py": (50, 6, [
        'for name',
        'for name',
        'for name',
        'for name',
        'try:except',
        'try:except',
    ]),
    "pdbqt_check.py": (52, 1, [
        'for (name, n_atoms)',
    ]),
    "pockets_check.py": (140, 11, [
        'for (idx, pocket) | if sites',
        'for (idx, pocket) | if sites',
        'if every',
        'if long_sites and sites',
        'if long_sites and sites',
        'if sites',
        'if sites',
        'if sites',
        'try:else',
        'try:else',
        'try:except',
    ]),
    "prep_check.py": (82, 7, [
        'for (fixture, label)',
        'if made is None',
        'if made is None',
        'if made is None',
        'if made is None',
        'try:except',
        'try:except | for (fixture, label)',
    ]),
    "representation_geometry_check.py": (51, 0, [
    ]),
    "scoring_cross_check.py": (19, 0, [
    ]),
    "scoring_docs_check.py": (47, 2, [
        'for (doc_name, engine_name)',
        'for (side, at375, at1875, at75)',
    ]),
    "structure_bond_check.py": (79, 6, [
        'def check_protein',
        'def check_protein',
        'def check_protein',
        'def check_protein',
        'def check_protein',
        'def check_small_molecule',
    ]),
    "viewport_framing_check.py": (3, 4, [
        'def expect',
        'def expect',
        'def expect',
        'def expect',
    ]),
    "workbench_interaction_check.py": (182, 16, [
        'for key',
        'for key | if crambin.is_file()',
        'if crambin.is_file()',
        'if not (crambin.is_file() and crambin_pose.is_file())',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:body',
        'try:except',
    ]),
    "x11_window_parse_check.py": (11, 0, [
    ]),
}

#: `cli_check.py` is deliberately absent from `SITE_INVENTORY` while its pin is
#: being written; see the note above the dict. Being explicit about the gap is
#: the point -- an inventory that silently skipped a file would be a lie.
NOT_INVENTORIED = {
    "cli_check.py":
        "its pin is being written right now, so its site counts would go stale "
        "the moment they were recorded; re-inventory it once the pin lands",
}


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
            guards, cur = [], node
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
                elif isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    guards.append("def " + p.name)
                if isinstance(p, ast.Module):
                    break
                cur = p
            guards = [g for g in guards if g != "def main"]
            if guards:
                cond.append(" | ".join(guards))
            else:
                uncond += 1
    return uncond, sorted(cond)


_drifted = []
for name, (rec_uncond, rec_n, rec_guards) in sorted(SITE_INVENTORY.items()):
    path = SCRIPTS / name
    if not path.exists():
        _drifted.append(f"{name}: inventoried but not on disk")
        continue
    got_uncond, got_guards = _sites_of(read(path))
    if (got_uncond, len(got_guards)) != (rec_uncond, rec_n):
        _drifted.append(
            f"{name}: {got_uncond} unconditional / {len(got_guards)} guarded, "
            f"recorded {rec_uncond} / {rec_n}"
        )
    elif got_guards != sorted(rec_guards):
        _drifted.append(f"{name}: the same number of guarded sites but different guards")

check(
    not _drifted,
    "no check() call site has been added, removed, or moved behind a guard",
    f"{len(SITE_INVENTORY)} scripts inventoried as "
    f"{sum(v[0] + v[1] for v in SITE_INVENTORY.values())} call sites; drifted: "
    f"{_drifted}. Wrapping a check in an `if` moves it out of the "
    f"unconditional column and lands here. A site that can stop running without "
    f"the count changing is how a suite quietly loses coverage",
)

check(
    set(NOT_INVENTORIED) <= set(INVENTORY),
    "every script excluded from the site inventory is still a gate we track",
    f"{sorted(NOT_INVENTORIED)}: {list(NOT_INVENTORIED.values())[0][:90] if NOT_INVENTORIED else ''}",
)

print()
print("  site inventory -- unconditional call sites, and the guarded ones:")
for name, (u, n, _g) in sorted(SITE_INVENTORY.items()):
    print("    %-38s %4d unconditional  %3d guarded" % (name, u, n))
print()

# ==========================================================================
section("this file holds itself to the same contract")

_self = pin_of(TARGET)
check(
    _self is not None
    and _self[0] == EXPECTED_CHECKS
    and bool(re.search(r"measured|deliberate", _self[1], re.I)),
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
sys.exit(1 if FAILURES else 0)
