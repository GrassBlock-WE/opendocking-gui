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
    "docs_claims_check.py",
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

#: Proxy for "this number has a note saying where it came from". Kept in one
#: place because it was previously spelled three slightly different ways at three
#: call sites, which is how a rule quietly stops being one rule. See the note at
#: the `justified =` line for what it can and cannot see.
PROVENANCE_RE = re.compile(
    r"measured|deliberate|green run", re.I
)

#: Files an owner is editing *at the moment this gate runs*. Measured, not
#: assumed: during this round `cli_check.py`'s pin went 127 -> 128 and
#: `workbench_interaction_check.py`'s went 233 -> 243, and both files' mtimes
#: moved inside a single minute. A static audit that reads a file mid-rewrite
#: measures a state that never existed, and a gate whose result depends on when
#: you ran it is worse than no gate. These files are still required to *exist*,
#: to be in `INVENTORY`, and to be named in `NOT_INVENTORIED` -- the exemption is
#: from measurement, never from being tracked.
#:
#: Delete an entry once its owner's edits land and the file can be inventoried
#: again. An entry that is never deleted is a hole with a comment on it, so the
#: set is printed on every run and its size is asserted.
IN_FLIGHT = {
    "cli_check.py": "pin moved 127 -> 128 during this round",
    "workbench_interaction_check.py": "pin moved 233 -> 243 during this round",
}
MAX_IN_FLIGHT = 2

EXPECTED_CHECKS = 60  # measured: a green run of this file, including the two self-checks at the bottom that read it

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
    "viewport_framing_check.py": None,  # GUI; no context on this machine
    "x11_window_parse_check.py": 11,
}

#: The ratchet. Twelve scripts are unpinned today; that is a fact about the
#: repository, not a decision this file endorses. `MAX_UNPINNED_SCRIPTS` is the
#: count the check holds them to, so:
#:
#:   * a new unpinned script pushes the count over the line and fails;
#:   * adding a pin to one of them drops it below the line, which is fine --
#:     lower this number at the same time so the slack is not silently reused;
#:   * nothing here can make the debt smaller without someone editing a pin.
#:
#: Lowering it is the owner's call, not this file's.
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

for name in undeclared:
    if name in IN_FLIGHT:
        # A file being rewritten can be caught mid-write, at which point it
        # genuinely has no parseable pin. That is a true observation about a
        # half-written file and a useless one about the repository, so it is
        # called out here and excluded from the ledger check rather than being
        # allowed to decide the exit code. See the note on IN_FLIGHT.
        print(f"[FLY ] {name} is being edited right now; not held to the ledger")
        continue
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
section('no check() call site has quietly become conditional')

SITE_INVENTORY = {
    "benchmark_check.py": (33, 0, [
    ]),
    "check_doc_encoding.py": (0, 0, [
    ]),
    "check_repo_docs.py": (0, 0, [
    ]),
    "contacts_check.py": (18, 22, [
        'if found',
        'if found',
        'if found',
        'if found',
        'if found',
        "if kinds.get('hbond') | if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())",
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
        'if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file())',
    ]),
    "contacts_check2.py": (14, 2, [
        'for (title, contacts, expect_heavy)',
        'for resname',
    ]),
    "core_check.py": (253, 6, [
        'for (spacing, label)',
        'for cls',
        'for size',
        'try:else | for (name, fn) | for (cls, members)',
        'try:except | for (name, fn) | for (cls, members)',
        'try:except | for fn',
    ]),
    "examples_check.py": (19, 0, [
    ]),
    "ligand_check.py": (37, 19, [
        'for (smiles, symbol, want, why)',
        'for name',
        'for name',
        'for name',
        'for name',
        'for name',
        'for name',
        'for name',
        'for name',
        'for name',
        'for name',
        'if made is None | for name',
        'if made is None | for name',
        'if not src.exists() | for name',
        'if ref.exists() | for name',
        'try:body',
        'try:body',
        'try:except',
        'try:except',
    ]),
    "pdbqt_check.py": (49, 4, [
        'for (bad, why)',
        'for (name, n_atoms)',
        'if not path.exists() | for (name, n_atoms)',
        'try:body',
    ]),
    "pockets_check.py": (65, 84, [
        'for (label, box, want) | if long_sites and sites',
        'for (label, box, want) | if long_sites and sites',
        'for (size, want)',
        'if all_sites',
        'if all_sites and cavern',
        'if cavern',
        'if cavern',
        'if cavern',
        'if cavern',
        'if cavities',
        'if cavities',
        'if cavities',
        'if cavity',
        'if cavity',
        'if cavity',
        'if cavity',
        'if cavity',
        'if cavity',
        'if every',
        'if every',
        'if found',
        'if found',
        'if found',
        'if found',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if long_sites and sites',
        'if near_box is not None | try:else',
        'if near_box is not None | try:else',
        'if one',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sites',
        'if sliver and sites | if every',
        'if solid is not None',
        'if solid is not None',
        'if thin_site and wide_site',
        'if thin_site and wide_site',
        'try:else',
        'try:except',
    ]),
    "prep_check.py": (62, 27, [
        'for (name, ok, detail)',
        'if CRAMBIN.exists()',
        'if CRAMBIN.exists()',
        'if CRAMBIN.exists()',
        'if CRAMBIN.exists()',
        'if CRAMBIN.exists()',
        'if CRAMBIN.exists()',
        'if CRAMBIN.exists()',
        'if comp',
        'if comp',
        'if comp',
        'if comp',
        'if kept_waters is None or dry_waters is None',
        'if kept_waters is None or dry_waters is None',
        'if kept_waters is None or dry_waters is None',
        'if kept_waters is None or dry_waters is None',
        'if made is None',
        'if made is None',
        'if made is None',
        'if made2 is None | if made is None',
        'if made2 is None | if made is None',
        'if made2 is None | if made is None',
        'if made2 is None | if made is None',
        'try:body',
        'try:body | for (fixture, label)',
        'try:except',
        'try:except | for (fixture, label)',
    ]),
    "representation_geometry_check.py": (46, 5, [
        'if rib is not None and len(rib) > 0',
        'if rib is not None and len(rib) > 0',
        'if rib is not None and len(rib) > 0',
        'if rib is not None and len(rib) > 0',
        'if rib is not None and len(rib) > 0',
    ]),
    # 19 -> 21 when the grid fix (defect 190) made the per-element partition
    # falsifiable: two new unconditional sites, the ten-group fit and the
    # donor-acceptor refutation. The owner folded the `from_arrays` magnitude
    # into an existing check rather than adding a 22nd, so this count rose by
    # two and not three -- and this file is what noticed, which is the point of
    # it: a check site appearing or vanishing silently is exactly the change
    # nobody reads a diff for.
    "scoring_cross_check.py": (21, 0, [
    ]),
    "docs_claims_check.py": (44, 7, [
        'for (doc_name, engine_name)',
        'for (doc_name, engine_name)',
        'for (element, want)',
        'for (spacing, stated) | for (side, at375, at1875, at75)',
        'if doc_name not in doc_vinardo | for (doc_name, engine_name)',
        'if mc_cite',
        'if prep_cite',
    ]),
    "structure_bond_check.py": (79, 6, [
        'for (chain, ids)',
        'for (chain, ids)',
        'if ca_d',
        'if lengths',
        'if proline is not None',
        'if truth is None',
    ]),
    "viewport_framing_check.py": (3, 4, [
        'if abs(dx) <= tol and abs(dy) <= tol',
        'if abs(dx) <= tol and abs(dy) <= tol',
        'if len(xs) == 0',
        'if xs.max() > w - 1 or xs.min() < 0 or ys.max() > h - 1 or ',
    ]),
    "x11_window_parse_check.py": (9, 2, [
        'try:body',
        'try:body',
    ]),
}

#: Nothing is inventoried yet, and the reason is worth keeping rather than
#: quietly dropping.
#:
#: Two files were inventoried during this round and then taken back out, for the
#: same reason: **they were being edited while they were being measured.**
#: `cli_check.py`'s unguarded site count went 77 -> 79 inside one pass, and
#: `workbench_interaction_check.py`'s went 64 -> 71, and both files' mtimes moved
#: within a minute of this being written. Recording a count from a file that is
#: changing under the measurement does not make this gate stricter; it makes it a
#: source of false reds, and a gate that cries wolf is a gate people stop running.
#: Re-inventory both once their owners' edits land. The exemption is here so the
#: gap is stated rather than implied by an absence -- an inventory that silently
#: skipped a file would be a lie.
NOT_INVENTORIED = {
    "cli_check.py":
        "in flight: its unguarded site count moved from 77 to 79 under a single "
        "measurement pass, so any number recorded now would be stale unread",
    "workbench_interaction_check.py":
        "in flight: its unguarded site count moved from 64 to 71 and its pin moved "
        "from 233 to 243 under a single measurement pass, and its skip/summary "
        "structure is exactly what this round is auditing -- a count taken now "
        "describes neither the old file nor the new one",
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


_drifted = []
_unparseable = []
for name, (rec_uncond, rec_n, rec_guards) in sorted(SITE_INVENTORY.items()):
    path = SCRIPTS / name
    if not path.exists():
        _drifted.append(f"{name}: inventoried but not on disk")
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
    if (got_uncond, len(got_guards)) != (rec_uncond, rec_n):
        _drifted.append(
            f"{name}: {got_uncond} unconditional / {len(got_guards)} guarded, "
            f"recorded {rec_uncond} / {rec_n}"
        )
    elif got_guards != sorted(rec_guards):
        _drifted.append(f"{name}: the same number of guarded sites but different guards")

check(
    not _drifted and not _unparseable,
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

check(
    set(IN_FLIGHT) == set(NOT_INVENTORIED),
    "every file exempt from measurement is in flight, and vice versa",
    f"in flight: {sorted(IN_FLIGHT)}; not inventoried: {sorted(NOT_INVENTORIED)}. "
    f"Asserted as equality, not containment, so a file cannot quietly lose its "
    f"exemption and start being measured mid-edit -- which is the one failure "
    f"that would make this gate's result depend on when it was run. If the two "
    f"sets should ever differ, that is a deliberate edit to this check",
)

check(
    len(IN_FLIGHT) <= MAX_IN_FLIGHT,
    "no more than 2 gates are exempt from measurement as in flight",
    f"{len(IN_FLIGHT)} against a cap of {MAX_IN_FLIGHT}: {sorted(IN_FLIGHT)}. "
    f"This is a temporary exemption for concurrent edits, not a category. A third "
    f"file here means the owners' work has not landed and this gate is measuring "
    f"against a moving target",
)

print()
print("  site inventory -- unconditional call sites, and the guarded ones:")
for name, (u, n, _g) in sorted(SITE_INVENTORY.items()):
    print("    %-38s %4d unconditional  %3d guarded" % (name, u, n))
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
#: have one (`UNPINNED_BASELINE` is cross-checked against the pins on disk;
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
