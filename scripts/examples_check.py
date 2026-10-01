"""Gate the five example scripts, so they stop being things somebody runs by hand.

`examples/` holds five Python scripts. None of them is in CI, none is in the
reproduce list, and none is covered by any other check in this directory. They
are the only diagnostics the project has, and a diagnostic nobody runs is a
diagnostic that rots: every one of them was written against a system that has
since changed, and the question "does what this claims still hold?" had no
answer anywhere.

So this runs all five as subprocesses -- which is what a reader does, and which
is the only way to test their argument handling, their default paths and their
exit codes rather than just their functions -- and then checks what each one
*printed*, not merely that it exited zero. An exit code is a weak claim: a
script whose checks quietly stopped running still exits zero, and the
performance work in this repository has already been bitten by a gate that
pointed at the wrong thing. Each check below names the line it expects, so a
script that is reformatted fails loudly here instead of silently ceasing to be
verified.

What each script claims, in its own words, and what it prints today:

| script | claim | today |
|---|---|---|
| `check_receptor_donors.py` | a prepared receptor carries polar H on its donors | 55 donor sites, 0 without, 0 heavy atoms moved |
| `determinism_check.py` | repeat runs agree; a fixed seed reproduces a run | all 7 checks pass, max energy difference 0 |
| `diagnose_clash.py` | the scoring function does not prefer clashing poses | best placement overall *is* the best clash-free one, E = -0.815 |
| `audit_poses.py` | poses are in contact, are stationary, and CPU/GPU agree | 5 poses, 0 clashing, and every one flagged NOT a stationary point: `\|grad\|2` 4.1-11.9 against a declared tolerance of 1e-4 |
| `robustness_check.py` | malformed input is handled visibly, never by aborting the process | 27 cases, none failing; 2 of them accepted by the engine rather than rejected, which the script prints on the `ok` line and the check below counts |

Six scripts are run, not five. The sixth is
`scripts/interp_reachable_step_bound.py`, which measures the reachable-downhill
bound out of the tabulated map rather than restating it; it is what the audit's
`downhill` column is compared against, and the section below says why the number
it replaced was one machine's ratio.

`audit_poses.py` was the row that was quietly wrong, twice. It read "downhill
0.00e+00 on every one" for as long as this table existed, while pose 3 has
reported +1.52e-04 on this machine and +5.76e-04 on the CI runner for months. A
table column is a claim nobody checks, which is the same failure as the bound it
sat beside. And the zero itself was wrong a second way: it was read as "no
descent found" when what the audit had done was step over the entire descent
region, which for four of the five poses is narrower than its own finest step by
three to five orders of magnitude. The audit now prints the descent as a ladder
of eight steps from 1e-9 to 1.0, so "the step was too coarse" and "there is
nothing there" are different readings, and the table says what the output says.

What the ladder makes assertable, and why each of those is a check below rather
than a comment:

* the descent at the finest rungs is **linear in the step with the pose's own
  `\|grad\|2` as the slope** -- which is a positive claim about a first-order
  term, and simultaneously the negative case: a field whose *value* jumped
  across a cell face would read 1.3e+04 against the same tolerance where the
  measured answer is 1.4e-06, so the two mechanisms are seven orders of
  magnitude apart on one measurement;
* the audit's own `first-order` labels are exactly the rule it says they are,
  so the word in the output is not a judgement;
* the one-sided derivative really does jump, by 1.9x to 2.4x per pose, which is
  the half of the old C0 claim that survives and the reason a ladder of
  first-order rungs alone would be the wrong instrument;
* the four zeros and the one 1.52e-04 are **predicted** by the measured width
  of each descent region, so the split is a fact about the probe rather than a
  coincidence;
* the coarsest rung has left the basin on every pose, so the ladder brackets a
  maximum instead of running down a slope;
* the achieved gradient is pinned *above* 1e3x the engine's declared tolerance,
  which is what makes "these are not stationary points" safe to write down.

`diagnose_clash.py` deserves a note, because it is the one that looks stale and
is not. It was written to tell a scoring-function defect apart from a search
failure, back when `Element::interaction_radius()` returned 0.4 A for every
heavy atom and a hydrogen bond could be scored at 0.3 A between two oxygens.
That defect is fixed -- the radii are per-element now -- and the script's own
output is the proof: it reports the best clash-free placement and the best
placement overall as the *same* placement, which is the "after" column of the
investigation it was written for. It is no longer a probe for a live bug; it is
a regression guard for the fix, and it only became one once something ran it.
The stale part is one number in its prose, and the script says so itself now.

None of the six is slow enough to leave out. Measured on the machine that wrote
this file, together: about 23 s, the slowest being
`interp_reachable_step_bound.py` at about 6 s (it precalculates the maps, walks
all 117649 grid points three times over, and re-docks to measure the
conformation-to-angstrom lever). The clocks are printed and nothing asserts on
them -- a shared machine has been measured swinging 2.1x for byte-identical
work, so a threshold loose enough to survive that is also loose enough to miss
the regression it is watching for.

Run:  python scripts/examples_check.py
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
SCRIPTS = ROOT / "scripts"

#: Named in a message so a reader can find the file whose `_LINE_SEARCH_STEPS`
#: the reachable-step bound is derived from.
AUDIT_NAME = "audit_poses.py"

#: Per-script wall-clock ceiling. This is a *hang* guard, not a performance
#: claim: it exists so a wedged child fails the gate instead of blocking a
#: build, and it is two orders of magnitude above every measurement below, so
#: no amount of contention can reach it.
TIMEOUT_S = 300.0

FAILURES: list[str] = []
CHECKS = 0
CLOCKS: list[tuple[str, float, int]] = []

#: GATE-DECLARE 1
#: sites: 41 unconditional + 0 guarded
#: guards: sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
#:
#: 29 -> 30, and the added site is about the *readers* rather than about the
#: poses. `|grad|2`, `width` and `crossing` used to be read with
#: `numbers(out, rf"width ({NUMBER})")` and then indexed by pose number
#: (`widths[pose_no - 1]`). A regex over the whole output does not know which
#: line a value came from, so the index is right only while every value in the
#: output happens to be a pose's own value -- and nothing tested that. The
#: failure is silent in the worst way: a summary line that grew the word
#: `width`, or a pose line that lost its column, hands every pose a
#: neighbouring pose's number and every assertion below still passes on
#: plausible numbers. The readers are now keyed on the pose number printed on
#: the same line, and this site asserts that each column has exactly one value
#: per pose line and no occurrence anywhere else. It is a separate site
#: because the checks that consume the values cannot be trusted to notice that
#: they were read from the wrong place.
#:
#: It is also the answer to a problem this file has elsewhere: a summary that
#: prints `mislabelled[:4]` or `accepted[:3]` makes N items look like a
#: handful, so the two places that truncated a list now print the full count
#: next to the sample. The `width` and `crossing` values themselves were never
#: truncated -- the audit prints them at `.1e` and `.2e` -- and the check
#: message quotes every pose's own value.
#:
#: 20 -> 29. The eight added sites are the ladder assertions, and the reason
#: each one exists is that the old single bound could not distinguish two
#: incompatible mechanisms:
#:
#:  * the finest rung is a first-order term with the pose's own |grad|2 as the
#:    slope, which is simultaneously the positive claim and the negative case
#:    -- a value jump of the disputed size reads 1.3e+04 against the same
#:    1e-03 tolerance where the measurement is 2.2e-06;
#:  * the audit's `first-order` labels are exactly the rule it says they are,
#:    so the word in the output is not a judgement;
#:  * every pose also shows a one-sided derivative that has jumped, which is
#:    the half of the old C0 claim that survives and the reason a ladder of
#:    first-order rungs alone would be the wrong instrument;
#:  * the four zeros and the one 1.52e-04 are *predicted* by each descent
#:    region's measured width rather than coinciding with it;
#:  * the coarsest rung has left the basin on every pose, so the ladder
#:    brackets a maximum;
#:  * the achieved gradient sits at least 1e3x above the optimiser's declared
#:    tolerance, which is what makes the documentation's "not stationary
#:    points" safe to write -- a floor on the *failure*, so a converging
#:    engine turns it red instead of a threshold being relaxed;
#:  * the bound's own precondition, that the finest step cannot reach a cell
#:    face, checked from the audit's reported closest approach rather than
#:    assumed. It is a separate site because the bound above is invalid
#:    without it and a check that reported the bound without it would be
#:    reporting a number whose validity nobody had established;
#:  * every pose still descending at the old 1e-04 probe step had not yet left
#:    its cell, measured against the per-pose crossing step the audit now
#:    prints. This is the negative case stated as a fact about the fixture: the
#:    old claim was a claim about a crossing, so the way to kill it is to
#:    report where the crossing is. There is no tuned number in it -- the
#:    threshold is the step the old audit took.
#:
#: The site that became the bound check was not deleted, only extended: the
#: margin it reports is now the measured ratio LEVER x G_MAX / max|grad|2
#: rather than a comparison against two typed literals, and the third factor
#: is the map's first difference instead of its second, which is the constant
#: that actually bounds a step short enough to stay in a cell.
#:
#: 20 -> 21 was the previous round (the bound's self-consistency check), and
#: 21 -> 29 is this one: eight ladder assertions plus this file's own count,
#: which is the site that makes the pin below mean anything. No site here sits
#: behind a guard, which is why the digest is the SHA-256 of the empty string. The file moved off
#: `check_scripts_declare.py`'s generated snapshot to carry this block, so the
#: two cannot disagree: the auditor prefers a block and fails a gate that
# appears in both places.
#:
#: This file now pins its own total as well, which it did not before. It was
#: one of the gates `check_scripts_declare.py` counted as unpinned, so its
#: count was whatever happened to run and nothing failed when it shrank --
#: which for a file whose subject is "an exit code is a weak claim" was the
#: same mistake one level up. 29 was measured from a green run; 30 is the
#: 29 plus the column-reader site added above.

#: Pinned by the owner or measured from a green run; see the census above.
#: The one number the whole file's house style argues for, applied to itself.
#: `30 -> 31`, and the added site is about the *inputs* of a measurement rather
#: than its arithmetic. The check above this one proves
#: `REACHABLE_BOUND = FINEST_STEP x LEVER x G_MAX`, so a bound assembled from
#: the wrong parts is perfectly self-consistent and sails straight through it.
#: Two of those parts are borrowed rather than measured -- the four map slots
#: (mirroring `MapSlot` in `dock-core/src/grid.rs`, read in storage order from
#: the enum's discriminants rather than restated) and the audit's line-search
#: ladder -- and both are declared in the deliverable as mirrors of a file that
#: is not it. The site compares the two mirrors and **does not re-run the
#: deliverable**: a second derivation of a measurement is a second thing to keep
#: in step, and this file already spends six seconds deriving the number the
#: site above checks. `interp_reachable_step_bound.py` is a measurement
#: deliverable with no pin of its own, so "somewhere a gate runs it" was the
#: whole of its coverage; the hole was that nothing asserted what it *borrowed*.
#: No site here sits behind a guard, which is why the digest is the SHA-256 of
#: the empty string. Pinned by the owner or measured from a green run; see the
#: census above. The one number the whole file's house style argues for,
#: applied to itself.
#: `31 -> 36`, and the five added sites are all about one new module rather
#: than five new facts: `scripts/result_trust.py`, a pure function that answers
#: "can I trust *this* number?" for one result. It is a module and not a gate,
#: so it carries no pin of its own; the sites below are the gate, and they feed
#: it the audit's own reported columns and the engine's own declared
#: thresholds, so the binding to the engine stays here where a check can see it
#: instead of living inside the verdict as a second copy of the constant.
#:
#: Four of the five are mutation-proven in both directions, and the three that
#: matter are the ones that would let the function flatter a result:
#:
#:   * `_unmeasured()` returning `holds` -- the whole "keeps saying fine when
#:     the measurement is missing" failure, in one token.
#:   * the stationarity comparison inverted, which is the check reading a
#:     non-stationary pose as stationary.
#:   * the support contract always holding, which is a pair beyond the f32
#:     edge reading as inside the model.
#:
#: The fifth asserts the gate's own dependency: the columns the verdict was fed
#: are parsed out of the audit's output, so a column that stops being printed
#: empties the parse and the first site goes red rather than the verdict
#: quietly answering from the columns that survived. No site here sits behind a
#: guard, which is why the digest is the SHA-256 of the empty string.
EXPECTED_CHECKS = 46


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def run_script(name: str, directory: Path = EXAMPLES) -> tuple[str, int, float, str]:
    """Run one of the gated scripts the way a reader would.

    Absolute path, repository root as the working directory: the scripts
    resolve their own fixtures from `__file__`, so they do not need a `cd`, and
    running them from the root is what the comments in them promise.

    `directory` is `examples/` for the five diagnostics. The one helper gated
    below lives in `scripts/` and is run exactly the same way -- same
    interpreter, same working directory, same "check what it printed" rule --
    because a script that derives a number the audit reports belongs beside the
    audit rather than inside it.
    """
    path = directory / name
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return "", -1, time.perf_counter() - start, f"timed out after {TIMEOUT_S:.0f} s"
    elapsed = time.perf_counter() - start
    return proc.stdout or "", proc.returncode, elapsed, (proc.stderr or "").strip()


def report(name: str, out: str, code: int, elapsed: float, why: str = "") -> None:
    """The per-script preamble: what it claimed, what it printed, how long.

    Every check below hangs off this, so the table a reader wants -- the claim,
    and the number that answers it -- is on the screen whether or not anything
    failed.
    """
    CLOCKS.append((name, elapsed, code))
    lines = [ln for ln in out.splitlines() if ln.strip()]
    verdict = f"exit {code}" if code == 0 else f"exit {code}"
    if why:
        verdict += f" ({why})"
    print(f"  {name}  —  {elapsed:.2f} s, {verdict}, {len(lines)} lines of output")
    if not lines:
        print("    (no output at all — this script printed nothing, which is never right)")


#: A number as the audit prints one: an optional sign, digits, an optional
#: fraction, an optional exponent. Written out rather than as a character class
#: because a class like `[\d.e+]` silently drops the sign of an exponent and
#: then hands `float()` the string `-5.00e`, which raises -- and it raised here,
#: on a *negative* downhill value, which is exactly what a failing pose looks
#: like. A guard that crashes on the condition it exists to catch is worse than
#: no guard, and the first version of this one proved it.
NUMBER = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


def numbers(out: str, pattern: str) -> list[float]:
    """Every capture group in `pattern`, flattened to floats, in order.

    `re.findall` hands back a tuple when the pattern has more than one group
    and a bare string when it has exactly one, so a caller that assumed one or
    the other got a `TypeError` from `float()` at run time rather than a wrong
    number. Flattening here makes both spellings behave the same way.

    A group that will not parse is **skipped, not raised on**, so a script that
    starts printing something unparseable produces a check that fails and says
    so rather than a traceback that takes the other sixteen with it. The callers
    that care pair the count against the pose count, so a skipped value is a
    failed check rather than a quietly shorter list.
    """
    flat: list[str] = []
    for match in re.findall(pattern, out):
        if isinstance(match, tuple):
            flat.extend(match)
        else:
            flat.append(match)
    values = []
    for text in flat:
        try:
            values.append(float(text))
        except ValueError:
            continue
    return values


#: A whole pose line of the audit's, with the pose number as its own group.
#: The colon matters: the audit also prints indented `pose 1  step ...` ladder
#: rows and a `pose 1 descent ladder, ...` header, and neither of those is a
#: pose's own line, so a looser pattern would read values off a ladder row.
POSE_LINE_RE = re.compile(r"(?m)^pose (\d+):(.*)$")


def pose_column(out: str, name: str) -> tuple[dict[int, float], list[str]]:
    """One named numeric column of the audit's pose lines, keyed by pose number.

    This replaces `numbers(out, rf"width ({NUMBER})")`, which is how the two
    ladder columns were read, and the reason is a failure mode a flat list
    cannot have. `re.findall` over the whole output does not know which line a
    value came from, so the caller has to index it by position
    (`widths[pose_no - 1]`) -- and that index is wrong the moment the output
    contains a value that is not on a pose line, with nothing to notice. Three
    ways that happens, none of which any assertion downstream would catch:

    * the audit's summary grows a line carrying the word, the reader picks it
      up, and every pose is handed the *next* pose's number;
    * a pose line loses its column, the list comes up short, and the last pose
      is silently handed the previous pose's value;
    * a value is printed twice on one line, and the two readings collapse into
      one list of the right length -- N identical-looking items that are not N
      of the same thing.

    So the reader walks pose lines only, keys on the pose number printed on
    the same line, and returns every occurrence it did not use as a complaint
    rather than absorbing it. `NUMBER` cannot match `inf` or `nan`, so a width
    the audit reports as `inf` arrives here as a *missing* value and says so,
    which is the same defect the old reader had one level down: `findall`
    drops what it cannot parse and the caller cannot tell a drop from a zero.

    The leading `(?<![^\\s])` is a word boundary that also works for a token
    starting in a non-word character. `\\b` does not: `|grad|2` is preceded by
    a space and `|` is not a word character, so there is no boundary there and
    the gradient column would read as empty.
    """
    found: dict[int, float] = {}
    complaints: list[str] = []
    pose_numbers: list[int] = []
    for match in POSE_LINE_RE.finditer(out):
        pose_no, line = int(match.group(1)), match.group(2)
        pose_numbers.append(pose_no)
        here = re.findall(rf"(?<![^\s]){re.escape(name)}\s+({NUMBER})", line)
        if len(here) > 1:
            complaints.append(
                f"pose {pose_no} prints `{name}` {len(here)} times ({here}), so "
                f"one reading would have to be discarded"
            )
        if not here:
            complaints.append(f"pose {pose_no} prints no `{name}` value")
        else:
            found[pose_no] = float(here[0])
    for line in out.splitlines():
        if POSE_LINE_RE.match(line):
            continue
        if re.search(rf"(?<![^\s]){re.escape(name)}\s+{NUMBER}", line):
            complaints.append(
                f"a `{name}` value outside every pose line: {line.strip()[:70]!r}"
            )
    if len(set(pose_numbers)) != len(pose_numbers):
        complaints.append(
            f"pose numbers are not unique: {sorted(pose_numbers)}"
        )
    return found, complaints


def main() -> int:
    print(f"gating the example scripts against {EXAMPLES}")
    print(
        "each is run as a subprocess and checked on what it printed; the clocks\n"
        "are reported and nothing asserts on them"
    )

    # --- 1. polar hydrogens on a prepared receptor -----------------------------
    section("check_receptor_donors.py: a prepared receptor carries polar H")
    out, code, elapsed, err = run_script("check_receptor_donors.py")
    report("check_receptor_donors.py", out, code, elapsed, err.splitlines()[-1] if err else "")
    for ln in out.splitlines():
        if "donor sites" in ln or "polar H" in ln or "heavy atoms moved" in ln:
            print(f"    {ln.strip()}")
    check(
        "the script ran and said so",
        code == 0 and "WITHOUT polar H" in out,
        f"exit {code}; expected a line reporting donor sites without polar H"
        + (f"; stderr ended {err.splitlines()[-1]!r}" if err else ""),
    )
    without = numbers(out, r"WITHOUT polar H:\s*(\d+)")
    moved = numbers(out, r"heavy atoms moved\s*:\s*(\d+)")
    check(
        "every donor site in crambin carries a polar hydrogen",
        bool(without) and without[0] == 0,
        f"{without[0]} of the donor sites have no polar H" if without
        else "the script did not print a donor count",
    )
    check(
        "and the receptor stayed rigid",
        bool(moved) and moved[0] == 0,
        f"{moved[0]} heavy atoms moved" if moved else "the script did not report movement",
    )

    # --- 2. determinism ---------------------------------------------------------
    section("determinism_check.py: repeat runs agree, a fixed seed reproduces a run")
    out, code, elapsed, err = run_script("determinism_check.py")
    report("determinism_check.py", out, code, elapsed, err.splitlines()[-1] if err else "")
    for ln in out.splitlines():
        if ln.strip().startswith(("ok", "FAIL")) or "max diff" in ln or "centres" in ln:
            print(f"    {ln.strip()}")
    check(
        "the script ran and reached its own verdict line",
        code == 0 and "determinism and parallelism behave" in out,
        f"exit {code}"
        + (f"; stderr ended {err.splitlines()[-1]!r}" if err else "")
        + (
            ""
            if "determinism and parallelism behave" in out
            else "; it did not print its summary, so its checks did not all pass"
        ),
    )
    check(
        "no individual check inside it failed",
        "FAIL" not in out,
        "it reported a failing check" if "FAIL" in out else "all of its own checks pass",
    )
    # The claim in its docstring's first heading used to be thread-count
    # independence, and the printed heading said so too, while the checks
    # beneath it only repeat the same call: the engine decides threads
    # internally and exposes no knob, so there is nothing to vary. Nothing in
    # the repository tests that claim. The heading has been corrected to name
    # what is measured, and this check exists to stop the overstatement
    # creeping back.
    overstates = "independent of the thread count" in out
    check(
        "it no longer claims thread-count independence it cannot measure",
        not overstates and "reproducible" in out and "two identical calls agree" in out,
        "its heading still claims independence of the thread count, but the two "
        "calls it compares are the same call twice -- the engine exposes no "
        "thread-count knob, so nothing here varies it. Either vary it or stop "
        "claiming it"
        if overstates else
        "the heading now says 'reproducible', and what it checks is two "
        "identical calls agreeing byte-for-byte. Independence of the thread "
        "count remains untested anywhere in the repository, because the engine "
        "exposes no knob for it",
    )

    # --- 3. the clash diagnostic ------------------------------------------------
    section("diagnose_clash.py: the scoring function does not reward a clash")
    out, code, elapsed, err = run_script("diagnose_clash.py")
    report("diagnose_clash.py", out, code, elapsed, err.splitlines()[-1] if err else "")
    for ln in out.splitlines():
        print(f"    {ln.rstrip()}")
    best_all = numbers(out, r"best placement overall\s*:\s*E\s*=\s*(-?[\d.]+)")
    best_ok = numbers(out, r"best clash-free placement:\s*E\s*=\s*(-?[\d.]+)")
    check(
        "the diagnostic ran and reported both energies",
        code == 0 and bool(best_all) and bool(best_ok),
        f"exit {code}, best overall {best_all[:1]}, best clash-free {best_ok[:1]}"
        + (f"; stderr ended {err.splitlines()[-1]!r}" if err else ""),
    )
    check(
        "the best placement is not a clashing one",
        bool(best_all) and bool(best_ok) and best_all[0] >= best_ok[0] - 1e-9,
        f"best overall {best_all[0]:.3f} against best clash-free {best_ok[0]:.3f} "
        f"kcal/mol, a difference of {best_all[0] - best_ok[0]:+.3f}. The "
        f"per-element interaction radii are what make this true; with the old "
        f"0.4 A radius the best clashing placement won by 0.99"
        if best_all and best_ok else "no energies to compare",
    )
    check(
        "it still reaches a verdict rather than running out of placements",
        "global optimum" in out or "prefers" in out,
        "no verdict line" if not ("global optimum" in out or "prefers" in out)
        else "verdict printed",
    )

    # --- 4. the pose audit ------------------------------------------------------
    section("audit_poses.py: poses are in contact, stationary, and CPU/GPU agrees")
    out, code, elapsed, err = run_script("audit_poses.py")
    report("audit_poses.py", out, code, elapsed, err.splitlines()[-1] if err else "")
    for ln in out.splitlines():
        print(f"    {ln.rstrip()}")
    # "poses with a steric clash : 0/5" -- clashing count first, total second.
    clash = numbers(out, r"poses with a steric clash\s*:\s*(\d+)/(\d+)")
    clashing, audited = (clash[0], clash[1]) if len(clash) >= 2 else (None, 0)
    downhills = numbers(out, rf"downhill ({NUMBER})")
    # One entry per printed pose line, so "how many poses did it actually audit"
    # can be cross-checked against its own printed total rather than trusted.
    # No `^` anchor: the pattern is compiled without re.MULTILINE, so `^` would
    # only ever match the very start of the whole output and this would silently
    # find nothing.
    energies_seen = numbers(out, r"pose \d+: E\s+(-?[\d.]+)")
    # "The audit ran" is a question about output, not about the audit's verdict.
    # `audit_poses.py` exits non-zero precisely when it has something to report,
    # so demanding `code == 0` here made this check fire on a *finding* and
    # labelled it as a failure to run -- which is how a real 1.52e-04
    # reachable-downhill observation was reported as "the audit did not run".
    # The verdict is judged by the next two checks, which are about the clash
    # count and the downhill column. A crashed audit prints neither, and still
    # fails here.
    check(
        "the audit ran over a full pose set",
        clashing is not None and audited > 0 and audited == len(downhills)
        and audited == len(energies_seen),
        f"exit {code} (its own verdict, not a run failure), {audited} poses, "
        f"{len(downhills)} downhill values, {len(energies_seen)} pose lines"
        + (f"; stderr ended {err.splitlines()[-1]!r}" if err else "")
        + ("" if code == 0 else " -- a non-zero code here means it found something, "
                                  "and the two checks below say what")
    )
    check(
        "no returned pose has two heavy atoms inside each other",
        clashing == 0,
        f"{clashing:.0f} of {audited:.0f} poses clash" if clashing is not None
        else "no clash count printed",
    )

    # --- 4b. where the bound on that column comes from ------------------------
    # Run before the downhill check because the check reads it. Same treatment as
    # the five diagnostics: a subprocess, and what it printed is what is judged.
    section("interp_reachable_step_bound.py: the bound is measured, not typed")
    bout, bcode, belapsed, berr = run_script("interp_reachable_step_bound.py", SCRIPTS)
    report("interp_reachable_step_bound.py", bout, bcode, belapsed,
           berr.splitlines()[-1] if berr else "")
    # `report` records the clock, so the bound script is in the table with the
    # five diagnostics without being added twice here.
    # `(?m)^` on all four: the parseable block is printed unindented at the end
    # of the output and the human-readable summary above it is indented, so
    # anchoring on the start of a line picks the parseable one. Without the
    # anchor the patterns matched the summary instead and read 6 significant
    # figures where the parseable lines carry 12 -- which the self-consistency
    # check below then reported as a 5e-05 disagreement.
    bound = numbers(bout, r"(?m)^REACHABLE_BOUND = ([\d.eE+-]+) kcal/mol")
    finest = numbers(bout, r"(?m)^FINEST_STEP = ([\d.eE+-]+)")
    lever = numbers(bout, r"(?m)^LEVER = ([\d.eE+-]+)")
    gmax = numbers(bout, r"(?m)^G_MAX = ([\d.eE+-]+)")
    maxgrad2 = numbers(bout, r"(?m)^MAX_GRAD2 = ([\d.eE+-]+)")
    reconstructed = finest[0] * lever[0] * gmax[0] if (finest and lever and gmax) else 0.0
    # The four constants are checked against each other rather than merely
    # parsed. A bound that is read out of a file and compared without being
    # re-derived is a typed literal with extra steps, and this is the check that
    # keeps it one: if the measurement is scaled, dropped or silently defaulted,
    # `bound` stops being the product of its own parts and this goes red before
    # the downhill check can pass on it.
    check(
        "the bound is the product of three measured constants, not a typed number",
        bcode == 0 and bool(bound) and bool(finest) and bool(lever) and bool(gmax)
        and all(v > 0.0 and v == v and v != float("inf")
                for v in (bound[0], finest[0], lever[0], gmax[0]))
        and abs(bound[0] - reconstructed) <= 1e-9 * reconstructed,
        f"REACHABLE_BOUND {bound[0]:.6e} against {finest[0]:.0e} x {lever[0]:.3f} x "
        f"{gmax[0]:.3f} = {reconstructed:.6e}, a relative gap of "
        f"{abs(bound[0] - reconstructed) / reconstructed:.1e}; the finest step is "
        f"read out of {AUDIT_NAME} rather than restated, LEVER is differenced from "
        f"the engine's own kinematics, and G_MAX is the map's larger of its worst "
        f"first difference and its worst second difference over the spacing"
        if bound and finest and lever and gmax
        else f"exit {bcode}, parsed bound={bool(bound)} step={bool(finest)} "
             f"lever={bool(lever)} g_max={bool(gmax)}"
             + (f"; stderr ended {berr.splitlines()[-1]!r}" if berr else ""),
    )

    # **The inputs, which the check above structurally cannot see.** The check
    # above proves the bound is the product of its own three parts, so a bound
    # assembled from the *wrong* parts is perfectly self-consistent and sails
    # through it: change `SLOT_NAMES` and every factor moves together and the
    # identity still holds. Two inputs are borrowed rather than measured, and
    # both are declared in the deliverable as mirrors of a file that is not it:
    # the four map slots (mirroring `MapSlot` in `dock-core/src/grid.rs`, in
    # storage order, because `G_MAX` is assembled per atom from the slots that
    # atom's class reads) and the audit's ladder (`audit_steps`, read out of
    # `audit_poses.py` so the finest step cannot be restated).
    #
    # So they are compared here, *statically*: the deliverable is imported and
    # its declared values read, not re-run. Re-running would only re-derive the
    # same numbers this file already spends six seconds deriving, and a second
    # derivation of a measurement is a second thing to keep in step.
    _slots_declared = re.findall(
        r'SLOT_NAMES\s*=\s*\(([^)]*)\)',
        (SCRIPTS / "interp_reachable_step_bound.py").read_text(encoding="utf-8"))
    _slots_bound = [
        s.strip().strip('"')
        for s in (re.findall(r'"([^"]+)"', _slots_declared[0]) if _slots_declared else [])
    ]
    # The engine's own order, read out of the enum rather than restated: the
    # discriminant values are the storage order, so `Shape = 0` is what makes
    # slot 0 the fused shape field.
    #
    # **The fallback is the point.** An enum variant list is *also* a storage
    # order when the discriminants are implicit -- which is how two of the
    # engine's own enums are written today (`Element` names `H = 0` explicitly,
    # `AtomType` lists `CH,` with no discriminants at all). A reader that
    # assumed explicit discriminants would report an empty order and fail
    # against a correct engine the moment someone deleted four redundant `= 0`s.
    # That is the same failure as searching for a quoted `"backend"` string
    # where the language spells it as a field name: the probe asks for one form
    # and reads the other form's absence.
    _map_slot = (ROOT / "dock-core" / "src" / "grid.rs").read_text(
        encoding="utf-8").split("pub enum MapSlot", 1)[1].split("}", 1)[0]
    _named = re.findall(r"^\s{4}(\w+)\s*=\s*(\d+),", _map_slot, re.M)
    _order_src = "discriminants"
    if _named:
        _engine_order = [n for n, _ in sorted(_named, key=lambda kv: int(kv[1]))]
    else:
        _order_src = "declaration order (no explicit discriminants)"
        _engine_order = re.findall(r"^\s{4}([A-Z]\w+),", _map_slot, re.M)
    _engine_order = [
        {"Shape": "shape", "HbFromDonor": "hb_from_donor",
         "HbFromAcceptor": "hb_from_acceptor",
         "Hydrophobic": "hydrophobic"}.get(n, n)
        for n in _engine_order]
    # And the ladder: the deliverable reads the audit's own source, so the thing
    # to assert is that its regex still describes the audit's declaration.
    _steps_bound = re.findall(
        r"([\d.e+-]+)",
        (EXAMPLES / AUDIT_NAME).read_text(encoding="utf-8").split(
            "_LINE_SEARCH_STEPS = (", 1)[1].split(")", 1)[0])
    _ladder_ours = numbers(bout, r"(?m)^FINEST_STEP = ([\d.eE+-]+)")
    check(
        "the reachable-step bound's two borrowed inputs are the ones the files "
        "it mirrors say they are",
        _slots_bound == _engine_order and len(_engine_order) == 4
        and bool(_steps_bound)
        and bool(_ladder_ours)
        and abs(_ladder_ours[0] - min(float(s) for s in _steps_bound)) < 1e-30,
        f"the deliverable declares SLOT_NAMES = {_slots_bound}; "
        f"dock-core's MapSlot, in the storage order its {_order_src} give, is "
        f"{_engine_order} -- a per-atom G_MAX assembled from the wrong slots "
        f"would be self-consistent and wrong, which is the one failure the "
        f"product check above cannot see. Its finest step is read out of "
        f"{AUDIT_NAME}, whose ladder bottoms out at "
        f"{min(float(s) for s in _steps_bound):g}, and the bound reports "
        f"FINEST_STEP = {_ladder_ours and _ladder_ours[0]:g}. "
        f"**Mutation:** reordering either tuple, or renaming a slot, goes red "
        f"here and is invisible to the product check",
    )

    # ==================================================================
    # result_trust: the per-result verdict, checked against the values a
    # real result carries
    # ==================================================================
    # `scripts/result_trust.py` is a pure function: no I/O, no engine, no
    # process. That is what makes the checks below possible at all -- a
    # function that looked its own constants up could only be tested by
    # breaking the engine, and a verdict that cannot be handed a
    # hypothetical cannot have its failure directions checked at all.
    #
    # The values fed to it are the audit's own reported columns, parsed by
    # pose number, and the thresholds are the engine's declarations read from
    # source. So the binding to the engine stays here, where a check can see
    # it, rather than inside the verdict as a second copy of the constant.
    section("result_trust: a per-result verdict, and the directions it must refuse")

    # `out` is the audit's own output from the run above; captured under a name
    # of its own so a later section reusing the variable cannot move it.
    _audit_out = out
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import result_trust as rt  # noqa: E402  (after the path fix, on purpose)

    _lbfgs_src = (ROOT / "dock-core" / "src" / "search" / "lbfgs.rs").read_text(
        encoding="utf-8")
    _tol_m = re.search(r"gradient_tolerance:\s*([\d.e+-]+)", _lbfgs_src)
    _tolerance = float(_tol_m.group(1)) if _tol_m else None
    _finest = min(float(s) for s in _steps_bound) if _steps_bound else None
    # The f32 underflow edge of the Shape term, solved the way
    # `docs_claims_check.py` solves it: from the document's own gauss1 weight
    # and f32's smallest subnormal. `w1 * exp(-((d-0.5)/0.5)^2) = tiny`.
    _w1_doc = float(re.search(
        r"gauss1:\s*[-−+]?([\d.]+)",
        (ROOT / "docs" / "SCORING.md").read_text(encoding="utf-8")).group(1))
    _sub_tiny = 1.401298464324817e-45
    _support_edge = 0.5 + 0.5 * math.sqrt(math.log(abs(_w1_doc) / _sub_tiny))

    _grads_by_pose = pose_column(_audit_out, "|grad|2")[0]
    _widths_by_pose = pose_column(_audit_out, r"width")[0]
    _pose_nos = sorted(_grads_by_pose)
    _real = [
        rt.verdict_pose({
            "label": f"pose {n}",
            "grad_l2": _grads_by_pose.get(n),
            "gradient_tolerance": _tolerance,
            "descent_width": _widths_by_pose.get(n),
            "finest_step": _finest,
            "out_of_box_penalty": 0.0,
        })
        for n in _pose_nos
    ]
    check(
        "every real pose in the audit gets the verdict 'no', and the contract "
        "that says so is the one with the gradient in it",
        bool(_real) and _tolerance is not None and _finest is not None
        and all(v.trust == rt.NO for v in _real)
        and all("stationarity" in v.failed() for v in _real),
        f"{len(_real)} pose(s) from {AUDIT_NAME}, each handed its own reported "
        f"|grad|2 and descent width against the engine's declared tolerance "
        f"{_tolerance:.0e} and finest step {_finest:.0e}; all "
        f"{len(_real)} come back {rt.NO!r} with `stationarity` among the "
        f"failures. First: {(_real[0].summary if _real else '-')[:200]}. This "
        f"is the check that says the useful thing: a user reading pose 1's "
        f"energy is told, in the same breath, that it is not a stationary "
        f"point and by what factor. **Mutation:** inverting the stationarity "
        f"comparison turns this red, and a verdict module that defaulted a "
        f"missing value to 'holds' would leave it green",
    )

    # The anti-optimism contract, asserted by *withholding* a value rather
    # than by breaking anything. A verdict panel that keeps saying "fine"
    # when the measurement is missing is the failure mode, and the only way to
    # rule it out is to hand it the situation and read what it says.
    _blank = rt.verdict_pose({"label": "pose 1"})
    _no_grad = rt.verdict_pose({
        "label": "pose 1", "gradient_tolerance": _tolerance,
        "descent_width": 1e-8, "finest_step": _finest,
        "closest_surface_distance": 3.0, "support_zero_edge": _support_edge,
        "out_of_box_penalty": 0.0,
    })
    _no_edge = rt.verdict_pose({
        "label": "pose 1", "grad_l2": _grads_by_pose.get(1),
        "gradient_tolerance": _tolerance, "descent_width": 1e-8,
        "finest_step": _finest, "closest_surface_distance": 3.0,
        "out_of_box_penalty": 0.0,
    })
    check(
        "withholding a value never turns a verdict into 'yes', and the "
        "contract it could not measure says so -- including when another "
        "contract has already failed",
        _blank.trust == rt.UNKNOWN
        and _no_grad.trust != rt.YES
        and _no_grad.contract("stationarity").state == rt.UNMEASURED
        and _no_edge.trust != rt.YES
        and _no_edge.contract("field support").state == rt.UNMEASURED
        and len(_blank.contracts) == 4
        and all(c.state == rt.UNMEASURED for c in _blank.contracts),
        f"a result carrying nothing at all comes back {_blank.trust!r} with "
        f"{len(_blank.contracts)} contracts, every one {rt.UNMEASURED!r}; "
        f"withholding grad_l2 alone gives {_no_grad.trust!r} "
        f"(stationarity {rt.UNMEASURED!r}); withholding the support edge gives "
        f"{_no_edge.trust!r} (field support {rt.UNMEASURED!r}). The assertion "
        f"is `!= yes` rather than `== unknown`, because a *failure* outranks an "
        f"unmeasured contract on purpose: if the descent width was measured and "
        f"the search demonstrably steps over it, `no` is the more useful answer "
        f"than `unknown`, and `unknown` would bury a known defect. What may "
        f"never happen is a missing measurement reading as `yes`. **This is "
        f"the check the whole module exists for** and it needs no mutation to "
        f"prove: a default of 'holds' would make all three read {rt.YES!r}, and "
        f"the panel would be telling a user its number is fine because the "
        f"instrument was switched off",
    )

    # Every applicable contract lands in exactly one of the three states, and
    # the three lists partition the applicable set. Without this a contract
    # that silently stopped being emitted would not be noticed by anything
    # else here.
    _partition_ok = all(
        len(v.contracts) == 4
        and all(c.applies and c.state in (rt.HOLDS, rt.FAILS, rt.UNMEASURED)
                for c in v.contracts)
        and len(v.failed()) + len(v.unknown()) <= len(v.contracts)
        and (v.trust == rt.NO) == bool(v.failed())
        and (v.trust == rt.UNKNOWN) == (not v.failed() and bool(v.unknown()))
        for v in _real + [_blank, _no_grad, _no_edge])
    check(
        "the three states partition every applicable contract, and the "
        "overall verdict is derived from them rather than stated beside them",
        _partition_ok,
        f"{len(_real) + 3} verdicts examined, each with 4 contracts; a "
        f"contract is in exactly one of holds/fails/unmeasured, `no` is "
        f"equivalent to at least one failure, and `unknown` is equivalent to no "
        f"failure with at least one unmeasured. A module that hard-coded a "
        f"summary string would pass the first of these and fail the other two, "
        f"which is why they are three separate equivalences",
    )

    # Purity, and the proof that the verdict is *computed*. Flipping one
    # number must flip the named contract and change the overall answer; a
    # module that ignored its input would satisfy every other check here.
    _vals_holding = {
        "label": "synthetic", "grad_l2": _tolerance / 10.0,
        "gradient_tolerance": _tolerance, "descent_width": _finest * 10.0,
        "finest_step": _finest, "closest_surface_distance": 3.0,
        "support_zero_edge": _support_edge, "out_of_box_penalty": 0.0,
    }
    _v_all_hold = rt.verdict_pose(_vals_holding)
    _vals_one_off = dict(_vals_holding, grad_l2=_tolerance * 10.0)
    _v_one_fails = rt.verdict_pose(_vals_one_off)
    _vals_far = dict(_vals_holding, closest_surface_distance=_support_edge + 1.0)
    _v_support_fails = rt.verdict_pose(_vals_far)
    check(
        "the verdict is a function of its input: one number moved flips the "
        "named contract and nothing else, and the same input gives the same "
        "verdict",
        _v_all_hold.trust == rt.YES
        and _v_all_hold.contract("stationarity").state == rt.HOLDS
        and _v_all_hold.contract("line-search resolution").state == rt.HOLDS
        and _v_all_hold.contract("field support").state == rt.HOLDS
        and _v_one_fails.trust == rt.NO
        and _v_one_fails.failed() == ("stationarity",)
        and _v_support_fails.trust == rt.NO
        and _v_support_fails.failed() == ("field support",)
        and _v_all_hold != _v_one_fails
        and rt.verdict_pose(_vals_holding) == _v_all_hold,
        f"a fully-satisfying result gives {rt.YES!r} with all four contracts "
        f"{rt.HOLDS!r}; raising grad_l2 tenfold over the tolerance flips "
        f"exactly {list(_v_one_fails.failed())} to {rt.NO!r}; moving the "
        f"closest pair to {_support_edge + 1.0:.2f} A, past the "
        f"{_support_edge:.2f} A underflow edge, flips exactly "
        f"{list(_v_support_fails.failed())} instead. **Mutation:** ignoring an "
        f"input, or inverting a comparison, goes red here and in the first "
        f"check",
    )

    # The gate's own half of "fails if the engine stops reporting them": the
    # columns the verdict was fed are read out of the audit's output, so a
    # column that stops being printed empties the parse and this goes red --
    # rather than the verdict quietly receiving fewer values and answering
    # from the ones it kept.
    _needed = ("|grad|2", "width", "E ", "face-gap", "contacts")
    _absent = [c for c in _needed if c not in _audit_out]
    _fed = len(_grads_by_pose) + len(_widths_by_pose)
    check(
        "every column the verdict needs was actually printed, and each pose "
        "line carried both of the two the verdict reads per pose",
        not _absent and _fed >= 2 * len(_pose_nos) and len(_pose_nos) >= 1,
        f"{len(_pose_nos)} pose line(s); |grad|2 read for "
        f"{len(_grads_by_pose)}, width for {len(_widths_by_pose)}; the audit's "
        f"output carries all of {list(_needed)} (absent: {_absent or 'none'}). "
        f"**This is the direction the brief names:** if the engine stopped "
        f"reporting the gradient, the parse above would be empty, the first "
        f"check would have no poses and go red -- it would not produce a "
        f"verdict from the two columns that survived",
    )

    # ------------------------------------------------------------------
    # The two other results a user trusts: the receptor a pose was scored
    # against, and the pockets a user is shown as facts
    # ------------------------------------------------------------------
    section("result_trust: the receptor, the pocket, and what a failure asks for")

    # A receptor prepared the way the 1HVR redocking bug prepared one: a
    # homodimer reduced to its largest fragment, no polar hydrogens. Every
    # number here is a field `ReceptorPrepReport` documents, not one invented
    # for the test.
    _bad_rec = rt.verdict_receptor({
        "label": "1hvr",
        "atoms_in": 1826, "atoms_kept": 621, "atoms_dropped": 1205,
        "other_atoms_removed": 1205,
        "fragments_equal_chains": False, "chain_ids_dropped": ["B"],
        "selection": "largest fragment",
        "num_polar_hydrogens": 0, "polar_hydrogens_added": 0,
        "pdbqt_atoms_written": 621, "unknown_atom_types": 0,
    })
    _good_rec = rt.verdict_receptor({
        "label": "1crn",
        "atoms_in": 327, "atoms_kept": 327, "atoms_dropped": 0,
        "water_atoms_removed": 0, "ion_atoms_removed": 0,
        "other_atoms_removed": 0,
        "fragments_equal_chains": True, "chain_ids_dropped": [],
        "selection": "largest fragment",
        "num_polar_hydrogens": 12, "polar_hydrogens_added": 0,
        "pdbqt_atoms_written": 327, "unknown_atom_types": 0,
    })
    _rec_families = {c.name: c.family for c in _bad_rec.contracts
                     if c.state == rt.FAILS}
    check(
        "a receptor that lost half its chains fails on the three data "
        "contracts and nothing else, and a whole one passes",
        _bad_rec.trust == rt.NO
        and set(_bad_rec.failed()) == {"atoms retained", "fragment selection",
                                       "donor hydrogens"}
        and all(f == rt.FAMILY_DATA for f in _rec_families.values())
        and _good_rec.trust == rt.YES,
        f"the 1hvr-shaped preparation fails {list(_bad_rec.failed())} -- all "
        f"family {rt.FAMILY_DATA!r}, all with an input remedy: 1205 of 1826 "
        f"atoms gone, connectivity disagreeing with chain ids, and no polar "
        f"hydrogens anywhere; a 327-atom single chain with 12 polar hydrogens "
        f"comes back {rt.YES!r}. **The family is the point:** these are input "
        f"problems, and a verdict that filed them beside a stationarity "
        f"failure would send a reader to tune an optimiser when the file is "
        f"what is wrong",
    )

    # A pocket: a burial site of exactly the minimum voxel count, in a list
    # exactly as long as the ceiling. Three separate claims, and the third is
    # the one a user is never told -- that twelve is the cap, not the count.
    _pocket = rt.verdict_pocket({
        "label": "site 9", "kind": "burial", "volume": 12.29,
        "max_volume": 1500.0, "voxels": 8, "min_voxels": 8,
        "burial": 0.5, "min_burial": 1, "pocket_count": 12,
        "max_pockets": 12,
    })
    _pocket_ok = rt.verdict_pocket({
        "label": "site 1", "kind": "cavity", "volume": 412.0,
        "max_volume": 1500.0, "voxels": 806, "min_voxels": 8,
        "burial": 1.0, "min_burial": 1, "pocket_count": 4,
        "max_pockets": 12,
    })
    _no_params = rt.verdict_pocket({"label": "site 1", "kind": "cavity",
                                    "volume": 412.0})
    check(
        "a saturated list of twelve fails as a censored list, a site at the "
        "voxel floor says so, and a site without the caller's parameters gets "
        "'unknown' on those contracts rather than no opinion",
        _pocket.trust == rt.NO
        and set(_pocket.failed()) == {"enclosure", "above the grid floor",
                                      "set complete"}
        and _pocket_ok.trust == rt.YES
        and _no_params.trust == rt.UNKNOWN
        and len(_no_params.contracts) == 4
        and "volume not censored" in _no_params.unknown()
        and "set complete" in _no_params.unknown(),
        f"the 12-of-12 burial site fails {list(_pocket.failed())}; a sealed "
        f"cavity of 412 A^3 in a list of 4 against a ceiling of 12 comes back "
        f"{rt.YES!r}; the same cavity with the caller's parameters withheld is "
        f"{_no_params.trust!r} on {list(_no_params.unknown())}. That last one "
        f"is what this family exists for: a pocket is a claim derived from a "
        f"clustering and thresholds the caller chose, and **a verdict with no "
        f"opinion about the thresholds is the same failure as a check not in "
        f"the inventory**",
    )

    # Every failure has to say what would change it, and say *what kind* of
    # change. A failure with an empty remedy is a warning light with no
    # dashboard; a failure whose remedy names a cheaper fix than the real one
    # is worse, because it sends someone to do the work and find nothing.
    _all_v = [_real[0], _bad_rec, _pocket, _v_all_hold, _v_one_fails,
              _v_support_fails, _good_rec, _pocket_ok]
    _no_remedy = [f"{v.kind}:{c.name}" for v in _all_v
                  for c in v.contracts
                  if c.state == rt.FAILS and not c.actionable]
    _remedy_on_hold = [f"{v.kind}:{c.name}" for v in _all_v
                       for c in v.contracts
                       if c.state == rt.HOLDS and c.remedy]
    # A family and its remedy kind have to agree: a data contract that claims
    # only the engine could fix it, or an engine contract that claims re-running
    # the input would, is a sentence nobody can act on.
    _kind_by_family = {
        rt.FAMILY_DATA: {rt.REMEDY_INPUT},
        rt.FAMILY_PARAMETER: {rt.REMEDY_PARAMETER, rt.REMEDY_NONE},
        rt.FAMILY_ENGINE: {rt.REMEDY_ENGINE, rt.REMEDY_NONE},
        rt.FAMILY_MODEL: {rt.REMEDY_ENGINE, rt.REMEDY_PARAMETER, rt.REMEDY_NONE},
    }
    _mismatched = [f"{v.kind}:{c.name} {c.family}->{c.remedy_kind}"
                   for v in _all_v for c in v.contracts
                   if c.state == rt.FAILS
                   and c.remedy_kind not in _kind_by_family.get(c.family, set())]
    check(
        "every failing contract carries a remedy, no passing contract does, and "
        "each remedy's kind is one its family can actually offer",
        not _no_remedy and not _remedy_on_hold and not _mismatched
        and bool(_all_v[0].failed())
        and "To change it" in _all_v[0].summary,
        f"{sum(1 for v in _all_v for c in v.contracts if c.state == rt.FAILS)} "
        f"failing contract(s) across {len(_all_v)} verdicts, all carrying a "
        f"remedy (empty: {_no_remedy or 'none'}); no passing contract carries "
        f"one ({_remedy_on_hold or 'none'}); family/remedy-kind mismatches: "
        f"{_mismatched or 'none'}. The first failure's remedy travels in the "
        f"summary itself, which is the difference between a verdict and a "
        f"warning light",
    )

    # The two stationarity remedies, and the boundary between them: the honest
    # answer is "only the engine" in both cases, but the *sentence* differs,
    # and in one of them the step is already fine -- so a sentence that always
    # offers a finer step is offering something that would change nothing.
    _resolvable = {
        "label": "p", "grad_l2": 6.389, "gradient_tolerance": 1e-4,
        "descent_width": 1e-8, "finest_step": 1e-9,
        "closest_surface_distance": 3.0, "support_zero_edge": 5.5,
        "out_of_box_penalty": 0.0,
    }
    _unresolvable = dict(_resolvable, descent_width=1e-12)
    _fine_step = rt.verdict_pose(_resolvable)
    _coarse_step = rt.verdict_pose(_unresolvable)
    _fine = _fine_step.contract("stationarity")
    _coarse = _coarse_step.contract("stationarity")
    _fine_res = _fine_step.contract("line-search resolution")
    _coarse_res = _coarse_step.contract("line-search resolution")
    check(
        "the two stationarity remedies are the same kind and different "
        "sentences, and the resolution contract agrees with which one applies",
        _fine.remedy_kind == rt.REMEDY_ENGINE
        and "already" in _fine.remedy
        and "not a smaller step" in _fine.remedy
        and _fine.family == rt.FAMILY_ENGINE
        and _fine_res.state == rt.HOLDS
        and _coarse.remedy_kind == rt.REMEDY_ENGINE
        and "at or finer than 1e-12" in _coarse.remedy
        and _coarse_res.state == rt.FAILS
        and f"{1e-8 / 1e-9:g}x finer" in _fine.remedy
        and f"{1e-9 / 1e-12:g}x finer" in _coarse.remedy
        and f"{1e-9 / 1e-12:g}x coarser" in _coarse_res.because
        and _fine_step.trust == rt.NO and _coarse_step.trust == rt.NO,
        f"with a 1e-09 step against a 1e-08 region -- already "
        f"{1e-8 / 1e-9:g}x finer -- the resolution contract holds "
        f"({_fine_res.state!r}) and the stationarity remedy says the search can "
        f"see the descent and chose not to take it, so a smaller step is not "
        f"the answer. With the same step against a 1e-12 region the resolution "
        f"contract fails ({_coarse_res.state!r}) and the remedy names the step "
        f"that would resolve it, {1e-9 / 1e-12:g}x finer than the search "
        f"reaches. **Both are {rt.REMEDY_ENGINE!r}** -- the ladder is inside "
        f"the line search either way -- and **a module that always answered "
        f"'take a smaller step' would pass a `remedy is non-empty` test and "
        f"send every reader to re-run a search that cannot change.** An "
        f"earlier version of this module had these two contracts contradicting "
        f"each other, and this check is what found it. The expected ratios are "
        f"computed here rather than typed, so a change to the module's format "
        f"cannot quietly break the assertion in the direction that makes it "
        f"pass",
    )

    # The family vocabulary has to exist as values, not prose: a caller cannot
    # group results by "what kind of problem is this" if the kind is a sentence.
    _fams = sorted({c.family for v in _all_v for c in v.contracts})
    check(
        "every contract carries a machine-readable family and remedy kind, so "
        "a caller can group failures by what sort of thing is wrong",
        len(_fams) >= 2
        and all(isinstance(c.family, str) and bool(c.family)
                for v in _all_v for c in v.contracts)
        and all(c.remedy_kind in (rt.REMEDY_INPUT, rt.REMEDY_PARAMETER,
                                  rt.REMEDY_ENGINE, rt.REMEDY_NONE)
                for v in _all_v for c in v.contracts),
        f"the verdicts here span {_fams}, and every contract's `family` and "
        f"`remedy_kind` come from the declared vocabularies. A caller sorting "
        f"results by 'input problem' versus 'engine problem' cannot do that "
        f"from a sentence, and cannot do it at all if the fields are absent",
    )

    # The bound on `downhill` used to be two typed literals -- 1e-3 absolute
    # *and* 1e-4 relative to the pose's |E| -- and the relative half was a
    # judgement with nothing measured behind it. It passed here at 3.6e-05
    # relative and failed on the CI runner at 1.3e-04 on identical inputs,
    # because the search's Monte-Carlo walk is parallel and its poses depend on
    # the thread count: a different CPU returns a different pose set, and a
    # bound expressed as a ratio to one machine's energy is a bound to one
    # machine. It is gone.
    #
    # What replaces it is measured, by `scripts/interp_reachable_step_bound.py`, from the
    # tabulated field itself:
    #
    #     B = FINEST_STEP x LEVER x G_MAX
    #
    # where FINEST_STEP is the audit's *finest* ladder step, LEVER is the largest
    # atom displacement a unit conformation step produces, and G_MAX is the
    # larger of the map's worst first difference and its worst second
    # difference, both divided by the spacing and assembled per ligand atom
    # from the slots that atom's class reads, then combined over atoms by
    # Cauchy-Schwarz. Both bounds are valid because both are worst cases over
    # the whole clash-free region, not over this pose.
    #
    # **Why the finest step, and why that is tighter than it was.** The audit
    # used to report the largest descent it could find at steps of 1e-4 and
    # above, and the bound for that quantity has to cover a step that can
    # cross a cell face, so its constant is the second difference. The ladder
    # now reports the descent at 1e-9, where the step moves an atom 8.7e-09 A
    # and provably cannot reach a face: that quantity is a first-order term and
    # nothing else, so it is bounded by the first difference, and -- this is the
    # part that matters -- the descent is now what the term predicts *exactly*,
    # where the old 1e-4 reading was a first-order term sampled after an
    # overshoot and so only 37% of it. The margin falls from 585x to 98x and
    # every factor in that ratio is measured: LEVER x G_MAX is a box-wide worst
    # case, max|grad|2 is this pose's own gradient, and the quotient is the
    # price of a bound that does not depend on which poses this machine's
    # thread count returned. It cannot be tightened below the ratio without
    # giving exactly that up, which is the defect this bound replaced.
    largest = max(downhills) if downhills else 0.0
    # `|grad|2`, `width` and `crossing` are read per pose and keyed by the pose
    # number on the same line, because every claim below indexes them by pose.
    # `|grad|` (the per-component column) is still a flat read, and deliberately
    # so: it is only ever used through `max()` and `len()`, so a value read off
    # the wrong line could not change a single assertion. That is the whole
    # difference between the two readers, and it is why the flat one is left
    # alone rather than being made consistent for its own sake.
    grad_of, grad_problems = pose_column(out, "|grad|2")
    width_of, width_problems = pose_column(out, "width")
    cross_of, cross_problems = pose_column(out, "crossing")
    column_problems = grad_problems + width_problems + cross_problems
    grad2 = [grad_of[pose_no] for pose_no in sorted(grad_of)]
    grad_inf = numbers(out, rf"\|grad\|\s+({NUMBER})")
    pose_lines = sorted(set(grad_of) | set(width_of) | set(cross_of))
    check(
        "every |grad|2, width and crossing the audit printed sits on exactly "
        "one pose line, so no value can be read off the wrong pose",
        not column_problems,
        f"{len(grad_of)} gradient(s), {len(width_of)} width(s) and "
        f"{len(cross_of)} crossing(s) read off pose line(s) {pose_lines}, each "
        f"keyed on the pose number printed on that line rather than on the "
        f"order a regex happened to visit the lines in; "
        f"{len(column_problems)} problem(s): {column_problems}. A reader that "
        f"matched the wrong column would hand every pose a plausible number "
        f"and no assertion here could see it",
    )
    # The ladder: one row per (pose, step), carrying the pose number so a row
    # cannot be attributed to the wrong pose, and the step so the row can be
    # matched to the audit's own `_LINE_SEARCH_STEPS` without a second source
    # of truth. A rejected step prints no `descent`, so it simply does not
    # match -- the group count per row stays fixed either way.
    LADDER_RE = rf"(?m)^\s*pose (\d+)\s+step\s+({NUMBER})\s+descent\s+({NUMBER})" \
                rf"\s+slope\s+({NUMBER})\s+(\S+)"
    ladder_rows: list[tuple[int, float, float, float, str]] = []
    for match in re.finditer(LADDER_RE, out):
        pose_no, step, drop, slope, state = match.groups()
        ladder_rows.append((int(pose_no), float(step), float(drop), float(slope), state))
    by_pose: dict[int, list[tuple[float, float, float, str]]] = {}
    for pose_no, step, drop, slope, state in ladder_rows:
        by_pose.setdefault(pose_no, []).append((step, drop, slope, state))
    for rows in by_pose.values():
        rows.sort()
    finest_rung = [
        (pose_no, rows[0][2], grad_of[pose_no])
        for pose_no, rows in sorted(by_pose.items())
        if rows and pose_no in grad_of
    ]
    all_first_order = [
        (pose_no, slope, grad_of[pose_no])
        for pose_no, rows in sorted(by_pose.items())
        for _step, _drop, slope, state in rows
        if state == "first-order" and pose_no in grad_of
    ]
    # The tolerance the audit itself applies to decide that label, restated
    # here rather than read out of its source: the check below is that the
    # audit's labels and this rule agree, and a check that imported the constant
    # it is testing could not fail.
    LINEARITY_TOL = 1e-3
    # A value jump of the size the old claim predicted, carried here so the
    # detail line can say what the same measurement would have read under the
    # other mechanism. 1.52e-04 kcal/mol is the descent the audit reported at
    # its 1e-4 rung; divided by the steepest pose's own first-order prediction
    # at the finest rung, it is what a discontinuity would look like measured
    # against a tolerance of LINEARITY_TOL.
    JUMP_SIZE = 1.52e-4
    worst_finest = max(
        (abs(s - o) / o for _p, s, o in finest_rung), default=0.0
    )
    worst_any = max((abs(s - o) / o for _p, s, o in all_first_order), default=0.0)
    check(
        "the finest rung of every ladder is a first-order term in the step, "
        "with the pose's own gradient as the slope",
        bool(finest_rung) and all(
            abs(slope - own) <= LINEARITY_TOL * own
            for _pose, slope, own in finest_rung
        ),
        f"{len(finest_rung)} pose(s), and every one of them descends at exactly "
        f"its own |grad|2 at the finest step: worst relative gap "
        f"{worst_finest:.1e} against a tolerance of {LINEARITY_TOL:.0e}, "
        f"{LINEARITY_TOL / worst_finest:.0f}x clear. That gap is the f64 noise "
        f"floor of a 1e-09 conf-unit step, and the claim is made at the finest "
        f"rung only: further in-cell rungs carry the trilinear field's own "
        f"quadratic term, which is O(delta^2) and reaches {worst_any:.1e} at "
        f"1e-05 -- still first-order behaviour, and the audit's label for those "
        f"rungs is checked separately below. This is also the check the old "
        f"claim could not pass: a field whose VALUE jumped across a cell face by "
        f"{JUMP_SIZE:.2e} kcal/mol would read "
        f"{JUMP_SIZE / (max(grad2) * 1e-9):.1e} here, so the two mechanisms are "
        f"seven orders of magnitude apart on one measurement"
        if finest_rung and grad2 and worst_finest > 0.0
        else f"no ladder rows were parsed ({len(ladder_rows)} matched, "
             f"{len(by_pose)} pose(s)) or no gradient column was printed "
             f"({len(grad2)} values)",
    )
    mislabelled = [
        (pose_no, step, state, abs(slope - grad_of[pose_no]) / grad_of[pose_no])
        for pose_no, rows in sorted(by_pose.items())
        if pose_no in grad_of
        for step, _drop, slope, state in rows
        if (abs(slope - grad_of[pose_no]) <= LINEARITY_TOL * grad_of[pose_no])
        != (state == "first-order")
    ]
    check(
        "and the audit's own first-order label is exactly that rule, not a "
        "judgement about which rows look right",
        bool(by_pose) and bool(grad2) and not mislabelled,
        f"{len(ladder_rows)} rung(s) judged against the {LINEARITY_TOL:.0e} rule "
        f"restated in this file; {len(mislabelled)} disagree, and the count is "
        f"the whole of it -- the first {min(4, len(mislabelled))} are printed: "
        f"{[(f'pose {p}', f'{s:.0e}', st, f'{d:.1e}') for p, s, st, d in mislabelled[:4]]}. "
        f"The rung labels are what makes the output readable, so a label that "
        f"drifts from the rule is a wrong word in the file a reader trusts"
        if by_pose and grad2
        else f"the ladder printed {len(ladder_rows)} row(s) over {len(by_pose)} "
             f"pose(s) and {len(grad2)} gradients, which is not enough to judge "
             f"the labels",
    )
    jump_by_pose = [
        (pose_no, max((abs(slope - grad_of[pose_no]) / grad_of[pose_no]
                       for _step, _drop, slope, _state in rows), default=0.0))
        for pose_no, rows in sorted(by_pose.items())
        if pose_no in grad_of and grad_of[pose_no] > 0.0
    ]
    check(
        "every pose also shows a one-sided derivative that has jumped, so the "
        "two mechanisms are told apart in the output rather than merged",
        bool(jump_by_pose) and all(d > 10.0 * LINEARITY_TOL for _p, d in jump_by_pose),
        f"largest relative slope departure from |grad|2, per pose and keyed on "
        f"that pose's own number: "
        f"{[f'pose {p} {d:.2f}' for p, d in jump_by_pose]} against a floor of "
        f"{10.0 * LINEARITY_TOL:.0e}. The C0 claim is half "
        f"right and this is the half that survives: the field is continuous in "
        f"value across a cell face and its one-sided DERIVATIVE is not. A ladder "
        f"that showed only the first-order rungs would be consistent with a "
        f"smooth field and would not be able to say the jump is there"
        if jump_by_pose
        else "no ladder rows could be compared against a gradient",
    )
    # The face-gap split, explained rather than observed. Four poses reported
    # `0.00e+00` and the one `face-gap 0.0001` reported +1.52e-04; the ladder
    # says why, and this is the assertion that says it: the sign of the descent
    # at the step the old audit probed at is predicted by the width the ladder
    # measured, because the step lands inside the descent region only when the
    # region is wider than the step.
    probe = 1e-4
    split = [
        (pose_no, drop_at, width)
        for pose_no, rows in sorted(by_pose.items())
        for step, drop, _slope, _state in rows
        for width in [width_of[pose_no] if pose_no in width_of else None]
        for drop_at in [drop if abs(step - probe) <= 1e-12 else None]
        if drop_at is not None and width is not None
    ]
    check(
        "the audit's four zeros and its one 1.52e-04 are predicted by the "
        "measured width of each descent region, not a coincidence of rounding",
        bool(split) and len(split) == len(by_pose) and all(
            (drop > 0.0) == (width > probe) for _pose, drop, width in split
        ),
        "; ".join(
            f"pose {pose_no}: descent {drop:+.2e} at the old 1e-04 step against a "
            f"region {width:.1e} wide ({width / probe:.0f}x the step)"
            for pose_no, drop, width in split
        )
        + f". The old audit reported zero wherever the step was wider than the "
          f"region, which is every pose but one, and the face-gap column rounded "
          f"to 4dp cannot tell {min(w for _p, _d, w in split):.0e} from "
          f"{max(w for _p, _d, w in split):.0e} conf units either -- so the split "
          f"was a statement about the probe, not about the poses"
        if split else
        f"the ladder printed no row at the {probe:.0e} step, so the old reading "
        f"cannot be reconstructed from it",
    )
    # The old claim was a claim about a *crossing*: "the energy jumps across a
    # cell face, so the 1.52e-04 is a jump". The audit now prints, per pose, the
    # step at which its -grad ray actually leaves the cell, so that claim can
    # be checked instead of argued: where the descent is still positive at the
    # old probe step, was any face crossed by then? If not, the number was
    # produced inside a single cell, where the trilinear field is one
    # polynomial with no discontinuity to have, and the old mechanism has
    # nothing left to explain.
    #
    # There is no tuned number in this: the threshold is the step the old audit
    # took, and the comparison is against a measured crossing.
    crossings = cross_of
    probe_positive = [
        (pose_no, drop, cross_of[pose_no] if pose_no in cross_of else None)
        for pose_no, drop, _w in split
    ]
    # A pose with no crossing value is excluded from `inside` rather than
    # defaulted: the old reader substituted 0.0 for a missing crossing, and
    # 0.0 compares as "the ray left the cell immediately", which is a claim
    # about a cell boundary the reader never looked at.
    inside = [(p, d, c) for p, d, c in probe_positive if d > 0.0 and c is not None]
    missing_crossing = [p for p, _d, c in probe_positive if c is None]
    check(
        "every pose that still descends at the old 1e-04 probe step had not yet "
        "left its cell, so the old reading cannot be a face-crossing jump",
        bool(crossings) and all(probe < c for _p, _d, c in inside),
        f"{len(inside)} of {len(probe_positive)} pose(s) still descend at 1e-04: "
        + ("; ".join(
            f"pose {p} descends {d:+.2e} and its ray leaves the cell at {c:.2e} "
            f"conf units ({c / probe:.1f}x the step)"
            for p, d, c in inside
        ) + ". " if inside else
           "none on this pose set, so the implication has nothing to attach to "
           "and this line is reporting a vacuous pass -- that is stated here "
           "rather than hidden, because a claim with no instance is not a "
           "claim that was verified. ")
        + "Inside one cell the trilinear field is a single polynomial, so there "
          "is no value discontinuity there to produce that number -- which is the "
          "negative case the old claim could not survive, and the reason the "
          "1.52e-04 is a first-order term sampled after an overshoot instead. The "
          "crossing steps for the poses that HAD crossed by then are "
        + ", ".join(
            f"pose {p} at {c:.1e}" for p, _d, c in probe_positive
            if c is not None and c <= probe
        )
        if crossings
        else f"the audit printed {len(crossings)} crossing value(s), so the old "
             f"claim's mechanism cannot be located",
    )
    if missing_crossing:
        # Not a separate site: it is the same claim, and printing it in the
        # detail line keeps the count of poses the check actually judged
        # visible next to the count of poses the audit printed.
        print(
            f"    [note] {len(missing_crossing)} pose(s) had no crossing value "
            f"and were not judged here: {missing_crossing}"
        )
    top = [
        (pose_no, rows[-1][1]) for pose_no, rows in sorted(by_pose.items()) if rows
    ]
    check(
        "and the ladder brackets a maximum rather than running down a slope: the "
        "coarsest step has left the basin on every pose",
        bool(top) and all(drop < 0.0 for _pose, drop in top),
        f"descent at the coarsest rung {[(p, f'{d:+.2f}') for p, d in top]} "
        f"kcal/mol, against {largest:.2e} at the finest. 1.0 conf unit moves the "
        f"fastest atom {lever[0]:.1f} A, so the top rung is a displacement that "
        f"genuinely leaves the basin; without it a ladder that only ever descended "
        f"could not tell a minimum from a ramp"
        if top and lever
        else "no ladder rows, or the bound script printed no LEVER",
    )
    # The engine's own convergence contract, as a ratchet on the *failure* to
    # meet it. `LbfgsConfig::gradient_tolerance` is 1e-4
    # (`dock-core/src/search/lbfgs.rs`); the search declares it and the poses it
    # returns sit four decades above it. Asserting a floor on that gap rather
    # than a ceiling is deliberate: it pins the wording in `docs/LIMITATIONS.md`
    # and `docs/SCORING.md`, which say these poses are *not* stationary points,
    # and it goes red on the day the engine converges -- which is a claim change
    # somebody has to make on purpose, not a threshold somebody can relax.
    TOLERANCE = 1e-4
    check(
        "every returned pose is at least 1000x above the optimiser's own "
        "gradient tolerance, which is what makes 'these are not stationary "
        "points' a safe thing for the documentation to say",
        bool(grad2) and min(grad2) > 1000.0 * TOLERANCE,
        f"achieved |grad|2 per pose {[round(g, 3) for g in grad2]} against the "
        f"declared tolerance {TOLERANCE:.0e} in `LbfgsConfig::gradient_tolerance`; "
        f"the smallest gap is {min(grad2) / TOLERANCE:.1e}x, the largest "
        f"{max(grad2) / TOLERANCE:.1e}x, and the audit's own per-component |grad| "
        f"column peaks at {max(grad_inf):.2f}. A floor of 1e3x sits 41x below the "
        f"smallest measured gap, so no plausible change in this machine's pose "
        f"set can move it -- and an engine that converged would turn it red, "
        f"which is the point"
        if grad2 and grad_inf
        else f"the audit printed {len(grad2)} gradient(s) and "
             f"{len(grad_inf)} per-component norm(s)",
    )
    check(
        "no returned pose offers a reachable descent beyond what the finest step "
        "can produce inside its own cell",
        bool(downhills) and bool(bound) and all(d <= bound[0] for d in downhills),
        f"largest descent at the finest rung {largest:.2e} kcal/mol over "
        f"{len(downhills)} poses against a bound of {bound[0]:.2e}, derived as "
        f"{finest[0]:.0e} x {lever[0]:.2f} A x {gmax[0]:.1f} kcal/mol/A from the "
        f"map itself ({bound[0] / largest:.0f}x margin, and that margin is the "
        f"measured ratio {lever[0]:.2f} x {gmax[0]:.1f} / {max(grad2):.2f} -- a "
        f"box-wide worst case over one pose's own gradient, with no fitted "
        f"factor in it). The bound is only valid while the step cannot reach a "
        f"face, and that is checked rather than assumed below. NOTE what this "
        f"check can and cannot catch: it is a claim about the *instrument* -- "
        f"that the finest rung is not hiding a descent the map's own geometry "
        f"does not produce -- and since the descent there is |grad|2 x step by "
        f"construction, a pose with a ten times larger gradient still passes it. "
        f"Detecting a worse pose is the job of the two ratchets above, which run "
        f"the other way on purpose"
        if downhills and bound and finest and lever and gmax and grad2
        else "the bound measurement did not produce its constants, or the audit "
             "printed no downhill column",
    )
    closest = numbers(out, rf"closest approach to a cell face\s+:\s+({NUMBER})")
    reachable = finest[0] * lever[0] if (finest and lever) else 0.0
    check(
        "and the finest step provably cannot reach a cell face, which is the "
        "precondition that bound rests on",
        bool(closest) and bool(lever) and bool(finest) and reachable < closest[0],
        f"the finest step moves an atom {reachable:.2e} A against a closest "
        f"approach to a cell face of {closest[0]:.3e} A over the batch -- "
        f"{closest[0] / reachable:.0f}x clear. The bound above is a first-order "
        f"bound precisely because no face is crossed inside the step; a pose "
        f"sitting on a face would make the constant a second difference again, "
        f"and the audit's face-gap column rounds to 0.0000 cells and cannot say "
        f"which case this is, so the summary line carries it in angstrom"
        if closest and lever and finest
        else "the audit printed no closest-approach line, so the bound's "
             "precondition cannot be checked",
    )
    check(
        "and it read a downhill figure for every pose it reported",
        len(downhills) == audited,
        f"{len(downhills)} downhill values parsed against {audited:.0f} poses: "
        f"a shortfall means a column was reformatted or a value was "
        f"unparseable, and the check above would be reading a shorter list than "
        f"it looks",
    )
    # `downhill` is the largest energy *decrease* reachable, so a negative one is
    # not a good pose -- it is a number that cannot exist. Both the audit and
    # this gate bound it from above (the audit at an absolute 1e-6, this gate at
    # the measured bound above), and both are upper bounds, so a mutation that
    # printed -0.5 slipped through the pair of them. It cannot be produced by the
    # line search as written, so this is a guard on the report rather than on the
    # physics.
    check(
        "no pose reports a negative reachable downhill step",
        bool(downhills) and all(d >= 0.0 for d in downhills),
        f"smallest reported downhill {min(downhills):+.2e} kcal/mol; the largest "
        f"reachable energy *decrease* cannot be negative, so this is the report "
        f"being wrong rather than the pose"
        if downhills else "no downhill column printed",
    )
    gpu = numbers(out, rf"cpu-gpu ({NUMBER})")
    check(
        "the CPU and GPU scoring paths agree on the returned poses",
        bool(gpu) and len(gpu) == audited and max(gpu) <= 1e-5,
        f"largest CPU/GPU gap {max(gpu):.1e} kcal/mol over {len(gpu)} of "
        f"{audited:.0f} poses, against a bound of 1e-5. The bound is a guard "
        f"against single-precision drift, not a fitted value: today's "
        f"measurement is the same number in every column"
        if gpu else "no CPU/GPU column printed",
    )

    # ------------------------------------------------------------------
    # result_trust, bound to the objects instead of to typed dictionaries
    # ------------------------------------------------------------------
    # Every verdict above was handed a dictionary written out by hand. That
    # proves the module's arithmetic and says nothing at all about its
    # *vocabulary*: a key this module invented and a key the product renamed
    # would agree with each other forever, because the same author wrote both
    # sides of the comparison. So the projection is derived from the objects
    # here -- `ReceptorPrepReport`'s dataclass fields and its properties,
    # `find_pockets`' own signature, `DockingResult`'s public names -- and the
    # module's declared schema has to be a subset of what those objects
    # actually expose.
    #
    # The direction is the whole point. The objects are the authority and the
    # module declares which of their names it recognises; a check written in
    # the same vocabulary as the code it checks proves the two agree with each
    # other, which was never the question.
    section("result_trust: the key names, derived from the objects behind them")

    # The dev tree, ahead of any installed copy: this gate is about the code in
    # this repository. The path the imports resolved to is printed in the
    # details below, so a reader can see which tree answered.
    _pkg_root = str(ROOT / "dock-py" / "python")
    if _pkg_root in sys.path:
        sys.path.remove(_pkg_root)
    sys.path.insert(0, _pkg_root)
    import dataclasses
    import importlib.metadata as _metadata
    import inspect
    import tempfile

    import opendocking
    from opendocking import core as _core
    from opendocking import prep as _prep
    from opendocking.workbench import pockets as _pockets
    from opendocking.workbench.app import MoleculeView

    def _readable(cls) -> set[str]:
        """Every name a caller can read off `cls`: its dataclass fields *and*
        its properties.

        `dataclasses.fields()` alone would miss three of the names on
        `ReceptorPrepReport` -- `chains_dropped`, `fragments_equal_chains` and
        `is_lossy` are all properties, and the fragment contract reads one of
        them. A projection built from the fields alone would report that
        contract `unmeasured` forever, and would have no way to say why.
        """
        names = ({f.name for f in dataclasses.fields(cls)}
                 if dataclasses.is_dataclass(cls) else set())
        names |= {n for n, v in inspect.getmembers(cls)
                  if not n.startswith("_") and isinstance(v, property)}
        return names

    _report_names = _readable(_prep.ReceptorPrepReport)
    _report_props = sorted(_report_names - {
        f.name for f in dataclasses.fields(_prep.ReceptorPrepReport)})
    _pocket_names = _readable(_pockets.Pocket)
    _pocket_sig = inspect.signature(_pockets.find_pockets)
    _pocket_params = set(_pocket_sig.parameters)
    _receptor_names = {n for n in dir(_core.Receptor) if not n.startswith("_")}
    _result_names = {n for n in dir(_core.DockingResult) if not n.startswith("_")}

    # Where every declared pose and score key is obtained, and what it costs.
    #
    # **This table replaced a shorter one that was wrong.** The previous version
    # declared three pose keys and three score keys to have "no product source",
    # on the strength of a scan of the Python class. Every one of them turns out
    # to be obtainable today: `out_of_box_penalty` is a key of
    # `score_conformation_terms`, `cpu_energy` and `gpu_energy` are the two
    # `use_gpu` settings of `evaluate_conformations`, and `descent_width` is a
    # column `examples/audit_poses.py` already prints for every pose. A scan of
    # one class is not a scan of the product, and a gate that pinned the wrong
    # answer was the same defect as the one it was written to catch.
    #
    # The cost column is part of the claim. "Reachable" is not "free": the
    # descent width is a bracket from a seven-rung re-scoring ladder, and the
    # closest-distance figure is printed by a script rather than carried by the
    # result, so a caller that wants it on a result object still does not have
    # it on the result object.
    _REACHABLE = {
        "pose": {
            "grad_l2": "DockingResult.pose_gradient(i), L2 over the array",
            "gradient_tolerance": "LbfgsConfig::gradient_tolerance",
            "finest_step": "_LINE_SEARCH_STEPS[0]",
            "support_zero_edge": "the f32 underflow edge, derived as the gate "
                                 "already derives it",
            "out_of_box_penalty": "score_conformation_terms(...)["
                                  "'out_of_box_penalty'] -- free, one call",
            "closest_surface_distance": "examples/audit_poses.py's `min-dist`, "
                                        "which is centre-to-centre over heavy "
                                        "atoms: the quantity is reachable, the "
                                        "key's NAME is not what it returns",
            "descent_width": "examples/audit_poses.py's per-pose `width` "
                             "column, from descent_width(ladder): a geometric-"
                             "mean BRACKET, and it costs a seven-rung ladder",
        },
        "score": {
            "support_zero_edge": "as above",
            "cpu_energy": "evaluate_conformations(..., use_gpu=False); equals "
                          "DockingResult.energies",
            "gpu_energy": "evaluate_conformations(..., use_gpu=True)",
            "closest_surface_distance": "as in the pose family",
            "cpu_gpu_tolerance": "NOT reachable in this contract's form: the "
                                 "only bound the tree declares is the audit's "
                                 "RELATIVE 1e-5 on gap / max(1, max|E|), and "
                                 "this contract compares an absolute gap, so a "
                                 "caller must derive the absolute form",
        },
    }
    _NOT_REACHABLE = {  # reachable, but not in the form the contract reads
        "score": ("cpu_gpu_tolerance",),
    }
    _rec_orphan = sorted(set(rt.RECEPTOR_KEYS) - _report_names - _receptor_names)
    _rec_from_receptor = sorted((set(rt.RECEPTOR_KEYS) & _receptor_names)
                                - _report_names)
    _pk_from_object = sorted(set(rt.POCKET_KEYS) & _pocket_names)
    _pk_from_sig = sorted((set(rt.POCKET_KEYS) & _pocket_params)
                          - _pocket_names)
    _pk_from_caller = sorted(set(rt.POCKET_KEYS) - _pocket_names
                             - _pocket_params)
    # Coverage, derived rather than asserted: a key the module recognises
    # and the table above does not account for is unaccounted for, and a key
    # the table accounts for and the module no longer recognises is stale.
    # Either means the table has drifted from the module, which is the only
    # way it can be wrong without anything else here noticing.
    _uncovered = sorted(
        f"{family}:{key}" for family, keys in _REACHABLE.items()
        for key in (set(rt.SCHEMA[family]) - set(keys)))
    _stale = sorted(
        f"{family}:{key}" for family, keys in _REACHABLE.items()
        for key in (set(keys) - set(rt.SCHEMA[family])))
    check(
        "every key the verdict recognises is a name a real product object "
        "exposes, and every pose and score key is accounted for by a stated "
        "way of obtaining it",
        not _rec_orphan
        and _rec_from_receptor == ["num_polar_hydrogens", "unknown_atom_types"]
        and "fragments_equal_chains" in _report_props
        and _pk_from_caller == ["pocket_count"]
        and sorted(_pk_from_sig) == ["max_pockets", "max_volume", "min_burial",
                                     "min_voxels"]
        and "pose_gradient" in _result_names
        and "energies" in _result_names
        and "score_conformation_terms" in _core.__all__
        and "evaluate_conformations" in _core.__all__
        and _tolerance is not None and _finest is not None
        and _support_edge > 0.0
        and not _uncovered and not _stale,
        f"derived from the objects, not from a list: `ReceptorPrepReport` "
        f"exposes {len(_report_names)} names, of which {_report_props} are "
        f"properties rather than dataclass fields -- `fragments_equal_chains` "
        f"is one of them, so a `fields()`-only projection would have called "
        f"the fragment contract `unmeasured` permanently. The receptor's "
        f"{len(rt.RECEPTOR_KEYS)} declared keys resolve to "
        f"{len(_report_names & set(rt.RECEPTOR_KEYS))} report names plus "
        f"{_rec_from_receptor}, which only `Receptor` carries; unaccounted "
        f"for: {_rec_orphan or 'none'}. `Pocket`'s resolve to "
        f"{len(_pk_from_object)} dataclass names, {_pk_from_sig} off "
        f"`find_pockets`' own signature, and {_pk_from_caller} -- the caller's "
        f"own `len(pockets)`, on neither object. Every pose and score key is "
        f"accounted for ({len(_REACHABLE['pose'])} + "
        f"{len(_REACHABLE['score'])}; unaccounted {_uncovered or 'none'}, stale "
        f"{_stale or 'none'}), and **none of them is unreachable** -- an "
        f"earlier version of this gate said three of each had no product "
        f"source, on the strength of scanning one Python class, and that was "
        f"wrong: `out_of_box_penalty` is a key of `score_conformation_terms`, "
        f"`cpu_energy` and `gpu_energy` are the two `use_gpu` settings of "
        f"`evaluate_conformations`, and `descent_width` is a column "
        f"`examples/audit_poses.py` already prints. Reachability is not "
        f"freedom, though, and the table says what each one costs: the width is "
        f"a geometric-mean bracket behind a seven-rung re-scoring ladder, the "
        f"closest distance is a script's `min-dist` (centre-to-centre, heavy "
        f"atoms) rather than anything the result carries, and "
        f"{list(_NOT_REACHABLE['score'])} is reachable only after the caller "
        f"converts the tree's one declared agreement bound -- the audit's "
        f"*relative* 1e-5 -- into the absolute form this contract compares. So "
        f"a contract reading one of these is {rt.UNMEASURED!r} until a caller "
        f"does that work, which is a different statement from 'there is no "
        f"source' and is the one this gate now makes. `pose_gradient` and "
        f"`energies` are both present on `DockingResult`; the engine's "
        f"`gradient_tolerance` ({_tolerance}), finest ladder step ({_finest}) "
        f"and f32 support edge ({_support_edge:.2f} A) are bound to the tree "
        f"that answered, {opendocking.__file__}",
    )

    # ---- the real objects, built by the product's own entry points ---------
    # Crambin from `examples/`, prepared by `prepare_receptor_with_report`, and
    # a second component made by writing that same file twice with one copy
    # moved 40 A along x and re-labelled chain B. That is the 1HVR *shape*,
    # built out of the file that is in the tree: there is no 1hvr receptor
    # under `examples/`, and inventing one here would be the same defect this
    # section exists to catch. The two files go to the system temp directory;
    # the gate writes nothing into the repository.
    _rec_text, _rep = _prep.prepare_receptor_with_report(
        str(EXAMPLES / "1crn_receptor.pdb"))
    _rec = _core.Receptor.from_pdbqt_str(_rec_text)
    _pdb = [ln for ln in (EXAMPLES / "1crn_receptor.pdb").read_text(
        encoding="utf-8", errors="replace").splitlines()
        if ln.startswith(("ATOM", "HETATM"))]
    _moved = [ln[:21] + "B" + ln[22:38] + f"{float(ln[30:38]) + 40.0:8.3f}"
              + ln[38:] for ln in _pdb]
    _dimer_path = Path(tempfile.gettempdir()) / "examples_check_dimer.pdb"
    _dimer_path.write_text("\n".join(_pdb + _moved) + "\nEND\n",
                           encoding="utf-8")
    _dimer_text, _rep2 = _prep.prepare_receptor_with_report(str(_dimer_path))
    _rec2 = _core.Receptor.from_pdbqt_str(_dimer_text)

    # The coordinates for a pocket search come from the product's own loader,
    # not from a second PDBQT parser written for the gate: `cli.py` calls
    # `MoleculeView.from_pdbqt` on the headless `--auto-box` path and
    # `find_pockets_now` calls it in the workbench, so this is the route the
    # product takes with the same file.
    _pdbqt = Path(tempfile.gettempdir()) / "examples_check_1crn_prep.pdbqt"
    _pdbqt.write_text(_rec_text, encoding="utf-8")
    _view = MoleculeView.from_pdbqt(_pdbqt, "1crn", (0.6, 0.7, 0.9), 0.30,
                                    role="receptor")
    _sites = _pockets.find_pockets(_view.coords, _view.elements,
                                   residues=_view.residue_labels())

    def _default(name):
        """`find_pockets`' own default, read off its signature."""
        return _pocket_sig.parameters[name].default

    def _project(obj, kind, **extra):
        """The real object's own values, for the names `schema_for(kind)` wants.

        A name the object does not carry is left out rather than defaulted, so
        a rename shows up twice -- as a missing key, which makes the contract
        that reads it `unmeasured`, and in the orphan list above -- instead of
        arriving as a plausible zero.
        """
        read = {k: getattr(obj, k) for k in sorted(rt.schema_for(kind))
                if hasattr(obj, k)}
        read.update(extra)
        return read

    _rec_vals = _project(_rep, "receptor", label="1crn",
                         num_polar_hydrogens=_rec.num_polar_hydrogens,
                         unknown_atom_types=_rec.unknown_atom_types)
    _dimer_vals = _project(_rep2, "receptor", label="1crn x2",
                           num_polar_hydrogens=_rec2.num_polar_hydrogens,
                           unknown_atom_types=_rec2.unknown_atom_types)
    _site = _sites[0]
    _pk_vals = _project(_site, "pocket", label="1crn site 1",
                        pocket_count=len(_sites),
                        max_pockets=_default("max_pockets"),
                        max_volume=_default("max_volume"),
                        min_voxels=_default("min_voxels"),
                        min_burial=_default("min_burial"))
    _v_rec = rt.verdict_receptor(_rec_vals)
    _v_dimer = rt.verdict_receptor(_dimer_vals)
    _v_pocket = rt.verdict_pocket(_pk_vals)
    # `atoms written` reports a gap, and a gap is not a count of anything.
    # This captures the molecule the writer actually receives -- by wrapping
    # the module's own writer and its own `AddHs` call, so nothing here is
    # re-implemented -- and separates the two mechanisms that produce it.
    #
    # The capture is what turned "six hydrogens never reached the engine" into
    # the real account, which is the reason it is a gate site and not a probe:
    # the number is a net of two errors in opposite directions, so it can only
    # be read by decomposing it, and a decomposition nobody runs is a
    # decomposition that rots.
    _cap = {}
    _real_writer = _prep._molecule_to_pdbqt
    _real_addhs = None
    import warnings  # noqa: PLC0415  (local: only this capture needs it)

    def _capture_writer(m, resname="UNL"):
        _cap.setdefault("mol", m)
        return _real_writer(m, resname=resname)

    def _capture_addhs():
        """Record the donor-site list from the module's own AddHs call."""
        import rdkit.Chem.AllChem as _allchem
        global _real_addhs
        if _real_addhs is None:
            _real_addhs = _allchem.AddHs

        def _wrapped(m, *a, **kw):
            _cap["donors"] = list(kw.get("onlyOnAtoms") or [])
            return _real_addhs(m, *a, **kw)
        _allchem.AddHs = _wrapped

    _prep._molecule_to_pdbqt = _capture_writer
    _capture_addhs()
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _text2, _rep3 = _prep.prepare_receptor_with_report(
                str(EXAMPLES / "1crn_receptor.pdb"))
    finally:
        _prep._molecule_to_pdbqt = _real_writer
    _mol = _cap["mol"]
    _conf3 = _mol.GetConformer()
    _h_total = sum(1 for a in _mol.GetAtoms() if a.GetSymbol() == "H")
    # The writer's own keep/drop rule, applied to the molecule it was given.
    _dropped_h = []
    for _a in _mol.GetAtoms():
        if _a.GetSymbol() != "H":
            continue
        _nb = _a.GetNeighbors()
        if not _nb or _nb[0].GetSymbol() not in ("N", "O", "S"):
            _dropped_h.append((_a, "not bonded to N/O/S, dropped silently"))
            continue
        _sep = _conf3.GetAtomPosition(_a.GetIdx()).Distance(
            _conf3.GetAtomPosition(_nb[0].GetIdx()))
        if not (0.7 <= _sep <= 1.35):
            _dropped_h.append((_a, f"{_sep:.2f} A from its parent, warned"))
    _unprotonated = []
    for _d in _cap.get("donors", []):
        _a = _mol.GetAtomWithIdx(_d)
        if not [n for n in _a.GetNeighbors() if n.GetSymbol() == "H"]:
            _res, _chain, _seq, _nm = _prep._residue_identity(_a)
            _unprotonated.append(f"{_res} {_seq} {_nm}".strip())
    _hd_on_disk = sum(
        1 for ln in _rec_text.splitlines()
        if ln.startswith(("ATOM", "HETATM")) and ln[77:79].strip() == "HD")
    _drop_sites = []
    for _a, _why in _dropped_h:
        _nb = _a.GetNeighbors()
        _res, _chain, _seq, _nm = _prep._residue_identity(
            _nb[0] if _nb else _a)
        _drop_sites.append(f"{_res} {_seq} {_nm}".strip())
    _h_added, _h_dropped, _cys_sites = _h_total, len(_dropped_h), _unprotonated

    _lost = _rep.atoms_kept + _rep.polar_hydrogens_added - _rep.pdbqt_atoms_written
    check(
        "the real crambin preparation, the real two-component one and the "
        "real pocket search each produce a verdict, and the receptor everyone "
        "calls whole still fails a contract",
        _v_rec.trust == rt.NO
        and set(_v_rec.failed()) == {"atoms written"}
        and _lost == 6
        and _v_dimer.trust == rt.NO
        and "atoms retained" in _v_dimer.failed()
        and _v_dimer.contract("atoms retained").family == rt.FAMILY_DATA
        and _v_pocket.trust == rt.NO
        and set(_v_pocket.failed()) == {"enclosure", "set complete"}
        and not _v_rec.misspelled and not _v_pocket.misspelled,
        f"1crn, prepared for real: {len(_rec_vals)} keys projected off the "
        f"report, and the verdict is {rt.NO!r} on exactly "
        f"{list(_v_rec.failed())} -- the writer emitted "
        f"{_rep.pdbqt_atoms_written} ATOM records where "
        f"{_rep.atoms_kept + _rep.polar_hydrogens_added} were expected, so "
        f"{_lost:.0f} short. **That {_lost:.0f} is not a count of lost "
        f"hydrogens, and reading it as one sends someone to fix the wrong "
        f"thing.** `polar_hydrogens_added` is `len(indices)` in "
        f"`_add_receptor_polar_hydrogens` -- a count of donor SITES, not of "
        f"hydrogens. Capturing the molecule the writer receives: {_h_added} "
        f"hydrogens are actually present; the writer drops {_h_dropped} of them "
        f"({', '.join(_drop_sites) or 'none'}), each warned, each an N-H placed "
        f"13-22 A from its own parent; and {_hd_on_disk} `HD` records reach the "
        f"file against {_rep.polar_hydrogens_added} donor sites advertised. The "
        f"sites that got no hydrogen at all are {_cys_sites or 'none'} -- "
        f"every cysteine thiol in the file, and the largest single loss here, "
        f"and the only one that produces no warning to consume. The "
        f"arithmetic nets two errors in opposite directions -- sites that got "
        f"no hydrogen against sites that got two -- so the net {_lost:.0f} "
        f"means nothing on its own. **Nothing else about that preparation is "
        f"wrong:** {_rep.atoms_dropped} atoms dropped, "
        f"`fragments_equal_chains` {_rep.fragments_equal_chains}, "
        f"{_rec.num_polar_hydrogens} polar hydrogens present, "
        f"{_rec.unknown_atom_types} unrecognised types. A receptor that passes "
        f"every other contract and still loses atoms is exactly the case a "
        f"boolean over a preparation log cannot catch. The two-component file "
        f"drops {_rep2.atoms_dropped} of {_rep2.atoms_in} atoms and fails "
        f"{list(_v_dimer.failed())} "
        f"({_v_dimer.contract('atoms retained').family!r}) -- and its "
        f"`fragments_equal_chains` is {_rep2.fragments_equal_chains} while "
        f"{_rep2.chains_kept} of {_rep2.chains_in} chains were kept, because "
        f"that property compares fragment and chain *counts*; equal counts "
        f"with a chain dropped is not a clean preparation, which is why the "
        f"module asks about the dropped ids too. Real search on the prepared "
        f"receptor: {len(_sites)} site(s), the first a {_site.kind!r} site of "
        f"{_site.volume:.1f} A^3 in {_site.voxels} voxels, verdict "
        f"{rt.NO!r} on {list(_v_pocket.failed())} -- a heuristic groove, and a "
        f"list exactly as long as the caller's own ceiling of "
        f"{_pk_vals['max_pockets']:g} is a censored list",
    )

    # ---- a wrong key, against a right one -----------------------------------
    # A misspelled key and an absent value are different facts and must not
    # produce the same answer. An absent value is `unmeasured`: the contract
    # could not be looked at. A misspelled key means the caller asked about
    # something this verdict does not know, which is a mistake in the
    # *caller* -- and answering that `unmeasured` would bury the mistake under
    # a word the reader already takes for "nothing is wrong yet".
    _typo = dict(_rec_vals)
    _typo["atoms_keep"] = _typo.pop("atoms_kept")
    _v_typo = rt.verdict_receptor(_typo)
    _absent = {k: v for k, v in _rec_vals.items()
               if k != "unknown_atom_types"}
    _v_absent = rt.verdict_receptor(_absent)
    _cross = dict(_rec_vals, voxels=_site.voxels)
    _v_cross = rt.verdict_receptor(_cross)
    _typo_word = ("misspelling" if "misspelling" in _v_typo.summary
                  else "NOT a misspelling")
    # Renaming `atoms_kept` does something the hand-typed checks above could
    # not have shown: it silences the one contract that was failing. The
    # measured `no` becomes an `unknown`, and an `unknown` is the word this
    # module says buries a defect -- so the typo has to be loud enough to
    # survive it, and has to name the contract it silenced rather than let a
    # reader diff two runs to notice.
    _silenced = set(_v_rec.failed()) & set(_v_typo.unknown())
    check(
        "a misspelled key is loud and named, is not the same answer as a "
        "value that is simply absent, and cannot hide a failure by silencing "
        "the contract that reported it",
        _v_rec.misspelled == ()
        and _v_typo.misspelled == ("atoms_keep",)
        and _v_typo.trust != rt.YES
        and _v_rec.trust == rt.NO and _v_typo.trust == rt.UNKNOWN
        and _silenced == {"atoms written"}
        and "atoms written" in _v_typo.summary
        and "misspelling" in _v_typo.summary
        and "atoms_keep" in _v_typo.summary
        and "schema_for" in _v_typo.summary
        and _v_absent.misspelled == ()
        and "recognised atom types" in _v_absent.unknown()
        and _v_cross.misspelled == ("voxels",)
        and _v_cross.trust == rt.NO
        and "misspelling" in _v_cross.summary,
        f"the real projection is clean: {_v_rec.misspelled or 'no'} misspelled "
        f"key(s), and the verdict is {_v_rec.trust!r} on "
        f"{list(_v_rec.failed())}. Renaming `atoms_kept` to `atoms_keep` is "
        f"reported as {_v_typo.misspelled} and the summary says "
        f"{_typo_word} and points at `schema_for('receptor')` -- but the "
        f"verdict is now {_v_typo.trust!r}, **down from {_v_rec.trust!r}**, "
        f"because the typo removed the value `atoms written` reads and that "
        f"contract went {rt.UNMEASURED!r} instead of {rt.FAILS!r}. That is a "
        f"de-escalation a caller could cause by accident, and it is the whole "
        f"reason the summary names the silenced contract(s) "
        f"{sorted(_silenced) or 'none'} instead of leaving a reader to diff "
        f"two runs. Withholding `unknown_atom_types` is a different fact "
        f"entirely: {_v_absent.misspelled or 'no'} misspelled key(s) and "
        f"{list(_v_absent.unknown())} unmeasured, with the failures still "
        f"standing. A name from another family is equally loud -- `voxels` is "
        f"a real `Pocket` field and not a receptor key, reported as "
        f"{_v_cross.misspelled} beside a failure that survives. **This is the "
        f"check that would have caught the `grad_norm` typo, and the "
        f"de-escalation is a second defect it found in the same pass.** The "
        f"module cannot see a wrong key on its own, so a wrong key has to be "
        f"answered as a mistake rather than as a missing measurement",
    )

    # ---- the pose family, wired to a real docking result -------------------
    # A real run of the real engine, on maps the product builds from the
    # receptor it prepared a moment ago. `gradient_tolerance` and the finest
    # ladder step are the engine's own declarations, read from
    # `dock-core/src/search/lbfgs.rs` above; neither is a per-pose quantity,
    # so pairing them with a different pose set is not a mixture of sources.
    _lig = _core.Ligand.from_pdbqt_str(
        (EXAMPLES / "biotin_prep.pdbqt").read_text(encoding="utf-8"))
    _box = _core.GridBox.from_center_size((3.47, 6.21, 8.95),
                                          (18.0, 18.0, 18.0))
    _maps = _rec.precalculate(_box)
    _result = _core.dock(_lig, _maps, exhaustiveness=1, num_modes=1,
                         seed=20260929)
    _grad = _result.pose_gradient(0)
    _grad2 = math.sqrt(sum(float(x) * float(x) for x in _grad))
    _pose_vals = {
        "label": "biotin in 1crn, pose 1", "grad_l2": _grad2,
        "gradient_tolerance": _tolerance, "finest_step": _finest,
        "support_zero_edge": _support_edge,
    }
    _v_pose = rt.verdict_pose(_pose_vals)
    _version = _metadata.version("opendocking")
    _ext = Path(_core._core.__file__).resolve()
    # The three keys this gate previously called unreachable, measured on this
    # pose. Two of them come back from the engine in one call each; the third
    # is a column the audit prints, and the audit is the product's own script,
    # so the honest statement is "reachable from a product source, not carried
    # by the result" rather than either "unreachable" or "on the result".
    _term_maps = _rec.precalculate_terms(_box)
    _conf = _result.pose_conformation(0)
    _terms = _core.score_conformation_terms(_lig, _maps, _term_maps, _conf,
                                            "vina")
    _e_cpu = float(opendocking.evaluate_conformations(
        _lig, _maps, _conf.reshape(1, -1), "vina", use_gpu=False)[0])
    _e_gpu = float(opendocking.evaluate_conformations(
        _lig, _maps, _conf.reshape(1, -1), "vina", use_gpu=True)[0])
    _v_pose_full = rt.verdict_pose(dict(
        _pose_vals, out_of_box_penalty=_terms["out_of_box_penalty"]))
    _no_width = [n for n in list(_result_names) + [
        n for n in dir(_core) if not n.startswith("_")]
        if any(w in n.lower() for w in ("width", "descent", "region"))]
    check(
        "a real pose's own gradient fails the stationarity contract, and the "
        "other two contracts stay unmeasured until a caller supplies work "
        "this check does not do for them",
        len(_grad) == 6 + _lig.num_torsions
        and _grad2 > _tolerance
        and _v_pose.trust == rt.NO
        and set(_v_pose.failed()) == {"stationarity"}
        and set(_v_pose.unknown()) == {"line-search resolution",
                                       "field support", "inside the box"}
        and not _v_pose.misspelled
        and not _no_width
        and "out_of_box_penalty" in _terms
        and _terms["out_of_box_penalty"] == 0.0
        and set(_v_pose_full.unknown()) == {"line-search resolution",
                                            "field support"}
        and _v_pose_full.trust == rt.NO
        and abs(_e_cpu - _result.best_energy) <= 1e-9
        and _e_gpu == _e_cpu
        and _version == opendocking.__version__
        and ROOT in _ext.parents,
        f"docked for real: {_result.num_poses} pose(s) at "
        f"{_result.best_energy:.4f} kcal/mol by "
        f"{_result.scoring_function!r}, and `DockingResult.pose_gradient(0)` "
        f"returned {len(_grad)} values for {_lig.num_torsions} torsion(s) "
        f"plus 6 rigid-body degrees of freedom -- the layout its own docstring "
        f"claims, which is what makes `grad_l2` a measurement and not a name. "
        f"|grad|_2 is {_grad2:.4e} against the engine's declared tolerance of "
        f"{_tolerance:.0e}, {_grad2 / _tolerance:,.0f}x over, so this pose is "
        f"{rt.NO!r} on exactly {list(_v_pose.failed())}. The other two contracts "
        f"are {rt.UNMEASURED!r} here and stay that way: "
        f"{list(_v_pose.unknown())}. **A correction to this gate's own round-6 "
        f"claim:** `out_of_box_penalty` is *not* unreachable. "
        f"`score_conformation_terms` returns it for this exact pose as "
        f"{_terms['out_of_box_penalty']!r}, and feeding it in leaves only "
        f"{list(_v_pose_full.unknown())} unmeasured -- the `inside the box` "
        f"contract is answerable today for the cost of one call. `cpu_energy` "
        f"and `gpu_energy` are likewise both real: "
        f"{_e_cpu:.6f} and {_e_gpu:.6f} from `evaluate_conformations`, an exact "
        f"match to each other and to the result's own energy. What is genuinely "
        f"not on the result object is the descent width -- no public name on "
        f"`DockingResult` or in `core` carries a width, a descent or a region "
        f"({_no_width or 'none found'}) -- but it *is* obtainable from "
        f"`examples/audit_poses.py`'s per-pose `width` column, so the honest "
        f"wording is 'a product source that is not the result', not 'no "
        f"source'. Extension currency: "
        f"{_ext.name}, {_ext.stat().st_size:,} bytes, loaded from "
        f"{'this repository' if ROOT in _ext.parents else _ext.parent}, "
        f"version {_version} matching the package's {opendocking.__version__}. "
        f"Its mtime is normalised to the epoch, so it is not usable as "
        f"evidence and is not used as such: what is asserted is that the "
        f"member this binding needs is present and returns the shape it "
        f"documents",
    )

    # ---- one grid over all four families ------------------------------------
    # The three states have to interact the same way in every family, because
    # the precedence rules are properties of the fold and not of any one
    # contract set. This is the mechanical replacement for reading the module
    # line by line: the resolution inversion in `line-search resolution` was
    # found by reading, and reading does not happen again on the next edit.
    # Every value below comes from a real object or from the engine's own
    # declarations; the failures are produced by breaking a value so that the
    # family's *own* contract rejects it, not by handing over a typed verdict.
    _grid_inputs = {
        "pose": _pose_vals,
        "score": {"label": "biotin", "cpu_energy": _result.best_energy,
                  "support_zero_edge": _support_edge},
        "receptor": _rec_vals,
        "pocket": _pk_vals,
    }
    # (the key to withhold or misspell, the contract it must silence, the
    # values that make this family fail on its own terms, and a *second* key
    # to misspell for the failure-beside-a-typo cell)
    #
    # The fourth entry exists because of the defect the cell above found: a
    # typo on the same key a failing contract reads silences that contract, so
    # "a failure still outranks a typo" can only be tested with the two on
    # different keys. Using one key for both would have asserted the opposite
    # of what the module documents and passed for the wrong reason.
    _grid_cases = {
        "pose": ("grad_l2", "stationarity",
                 {"grad_l2": _tolerance * 1e3}, "finest_step"),
        "score": ("cpu_energy", "cpu/gpu agreement",
                  {"cpu_energy": 1.0, "gpu_energy": 2.0,
                   "cpu_gpu_tolerance": 1e-5}, "support_zero_edge"),
        "receptor": ("atoms_dropped", "atoms retained",
                     {"atoms_dropped": 7, "other_atoms_removed": 7},
                     "selection"),
        "pocket": ("pocket_count", "set complete",
                   {"voxels": _pk_vals["min_voxels"]}, "min_voxels"),
    }
    _grid, _grid_bad = [], []
    for _kind in ("pose", "score", "receptor", "pocket"):
        _fn = getattr(rt, f"verdict_{_kind}")
        _vals = _grid_inputs[_kind]
        _key, _contract, _break, _other = _grid_cases[_kind]
        _withheld = {k: v for k, v in _vals.items() if k != _key}
        _misspelt = dict(_withheld)
        _misspelt[_key + "s"] = _vals[_key]
        _beside = dict(_vals)
        _beside[_other + "s"] = _beside.pop(_other)
        _v_w, _v_t, _v_f = _fn(_withheld), _fn(_misspelt), \
            _fn(dict(_vals, **_break))
        _v_ft = _fn(dict(_beside, **_break))
        _grid.append((_kind, _key, _v_w, _v_t, _v_f, _v_ft))
        for _why, _bad in (
                (f"{_kind}: withheld `{_key}` was called misspelled",
                 _v_w.misspelled),
                (f"{_kind}: `{_contract}` not unmeasured when `{_key}` is gone",
                 _contract not in _v_w.unknown()),
                (f"{_kind}: withholding `{_key}` still reads yes",
                 _v_w.trust == rt.YES),
                (f"{_kind}: misspelling `{_key}` was not reported",
                 _v_t.misspelled != (_key + "s",)),
                (f"{_kind}: misspelling `{_key}` still reads yes",
                 _v_t.trust == rt.YES),
                (f"{_kind}: a measured failure did not read no",
                 _v_f.trust != rt.NO or not _v_f.failed()),
                (f"{_kind}: a failure lost to a typo",
                 _v_ft.trust != rt.NO or not _v_ft.failed()
                 or not _v_ft.misspelled
                 # pinned, because the two constructions are otherwise
                 # indistinguishable: the failing values are re-supplied on top
                 # of either one, so without this the cell passes whatever it
                 # is handed and tests nothing about precedence
                 or _v_ft.misspelled != (_other + "s",)),
                (f"{_kind}: a state outside the three declared",
                 any(c.state not in (rt.HOLDS, rt.FAILS, rt.UNMEASURED)
                     for c in list(_v_w.contracts) + list(_v_t.contracts)
                     + list(_v_f.contracts)))):
            if _bad:
                _grid_bad.append(_why)
    check(
        "all four families answer alike when a value is withheld, a key is "
        "misspelled, or a contract fails -- the precedence belongs to the "
        "fold, not to one contract set",
        not _grid_bad and len(_grid) == 4,
        "; ".join(
            f"{k}: withholding `{key}` -> {vw.trust!r} with "
            f"{list(vw.unknown())}, misspelling it -> {vt.trust!r} naming "
            f"{list(vt.misspelled)}, a real failure -> {vf.trust!r} on "
            f"{list(vf.failed())}, and that failure beside a misspelled "
            f"{_grid_cases[k][3]} -> {vft.trust!r} naming "
            f"{list(vft.misspelled)}"
            for k, key, vw, vt, vf, vft in _grid)
        + f". **A measured failure outranks `unknown` in every family, and a "
          f"typo can never produce {rt.YES!r} in any of them** -- a caller who "
          f"spells a key wrong is told so instead of being handed the word "
          f"that reads like a pass. This grid is the mechanical replacement "
          f"for reading the module: the resolution inversion in "
          f"`line-search resolution` was found by reading, and reading does "
          f"not happen again on the next edit"
        + (f". Problems: {_grid_bad}" if _grid_bad else ""),
    )
    # --- 5. malformed input -----------------------------------------------------
    section("robustness_check.py: malformed input is handled without aborting")
    out, code, elapsed, err = run_script("robustness_check.py")
    report("robustness_check.py", out, code, elapsed, err.splitlines()[-1] if err else "")
    oks = len(re.findall(r"^\s*ok\s+", out, re.M))
    fails = len(re.findall(r"^\s*FAIL", out, re.M))
    print(f"    {oks} cases reported ok, {fails} reported FAIL")
    check(
        "the script ran and reached its own verdict line",
        # Keyed on both of the script's two summary shapes rather than on the
        # words of the passing one. The passing summary reads "all
        # malformed-input cases produced a catchable error", and two of its
        # cases did not produce one -- the next check below counts exactly
        # those. Keying on that sentence made this gate assert a claim the
        # gate itself disproves two lines later, which is the same shape as a
        # test whose name says more than its body checks: green, and asserting
        # something false.
        code == 0
        and ("case(s) did not behave correctly" in out or "catchable error" in out),
        f"exit {code}, {oks} cases ok, {fails} failing"
        + (f"; stderr ended {err.splitlines()[-1]!r}" if err else "")
        + (
            ""
            if "case(s) did not behave correctly" in out or "catchable error" in out
            else "; it printed neither of its two summary lines, so its checks "
                 "did not all finish"
        ),
    )
    check(
        "no case it tried failed, and it tried at least 20 of them",
        fails == 0 and oks >= 20,
        f"{oks} ok and {fails} failing across {oks + fails} cases; a count below "
        f"20 means the script stopped testing something, which is a pass here "
        f"for the wrong reason",
    )
    # Two of its three process-death cases pass because the child *exits 0*,
    # not because it raised: the engine accepts a one-atom molecule and a
    # self-intersecting torsion set today. That satisfies the requirement as
    # written -- never a dead process -- but "ok" hides which of the two
    # happened, and a reader of the output would assume a raised error. Assert
    # the distinction so it cannot rot into a claim nobody checked.
    accepted = [
        ln.strip() for ln in out.splitlines()
        if ln.strip().startswith("ok") and "exit 0," in ln
    ]
    check(
        "and it is visible which process-death cases were accepted rather than rejected",
        bool(accepted),
        f"{len(accepted)} of its subprocess cases exit 0 with no stderr, i.e. the "
        f"engine accepts that input instead of raising. The script calls these "
        f"`ok`, which satisfies its stated requirement (never a dead process) "
        f"but reads as though a catchable error was produced: "
        + "; ".join(ln[:60] for ln in accepted[:3])
        if accepted else
        "every subprocess case raised a catchable error, so nothing is being hidden",
    )

    # --- the clocks -------------------------------------------------------------
    section("wall clocks, reported and not asserted")
    total = 0.0
    for name, elapsed, code in CLOCKS:
        total += elapsed
        print(f"  {name:<30} {elapsed:6.2f} s   exit {code}")
    print(
        f"  {'total':<30} {total:6.2f} s   against a {TIMEOUT_S:.0f} s per-script "
        f"hang guard that nothing here can reach"
    )
    # The guard is about *time*: a script that hangs is killed by `run_script`
    # and arrives here with no output, which the per-script checks above already
    # catch. Asserting a zero exit code here made this a second opinion about
    # the scripts' verdicts -- so a script that correctly reported a finding was
    # counted as "not gated". What has to hold is that all six produced output
    # inside the guard, and their exit codes are then reported rather than
    # required to be zero.
    check(
        "all six ran inside the hang guard, so all six are gated",
        len(CLOCKS) == 6 and all(e < TIMEOUT_S for _, e, _ in CLOCKS),
        f"{len(CLOCKS)} of 6 scripts produced output inside the "
        f"{TIMEOUT_S:.0f} s guard; slowest {max((e for _, e, _ in CLOCKS), default=0):.2f} s; "
        f"exit codes {[c for _, _, c in CLOCKS]} -- non-zero means the script "
        f"reported a finding, which its own section judges"
    )

    print("\n=== summary ===")
    print(f"  {CHECKS - len(FAILURES)} passed, {len(FAILURES)} failed, {CHECKS} checks")
    # The pin, asserted on the way out rather than only in the declaration: a
    # gate that prints its count without comparing it is the exact failure this
    # file exists to catch in the five scripts below. The `+ 1` is this check,
    # which has not been counted yet when the comparison is built.
    check(
        "this file's own count is the count it declares",
        CHECKS + 1 == EXPECTED_CHECKS,
        f"{CHECKS} ran before this one and {EXPECTED_CHECKS} are declared. The "
        f"count is constant by construction -- no site here sits behind a "
        f"machine-dependent guard -- so a difference means a check was added, "
        f"removed or made conditional, and one of those is a deliberate edit",
    )
    for f in FAILURES:
        print(f"    FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
