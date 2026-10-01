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
| `audit_poses.py` | poses are in contact, are stationary, and CPU/GPU agree | 5 poses, 0 clashing, downhill 0.00e+00 on every one |
| `robustness_check.py` | malformed input raises something catchable, never aborts | 24 cases, all catchable |

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

None of the five is slow enough to leave out. Measured on the machine that
wrote this file, together: under ten seconds, the slowest being
`determinism_check.py` at about 3.4 s (it runs the search four times and
precalculates five grid maps). The clocks are printed and nothing asserts on
them -- a shared machine has been measured swinging 2.1x for byte-identical
work, so a threshold loose enough to survive that is also loose enough to miss
the regression it is watching for.

Run:  python scripts/examples_check.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

#: Per-script wall-clock ceiling. This is a *hang* guard, not a performance
#: claim: it exists so a wedged child fails the gate instead of blocking a
#: build, and it is two orders of magnitude above every measurement below, so
#: no amount of contention can reach it.
TIMEOUT_S = 300.0

FAILURES: list[str] = []
CHECKS = 0
CLOCKS: list[tuple[str, float, int]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def run_script(name: str) -> tuple[str, int, float, str]:
    """Run one of the example scripts the way a reader would.

    Absolute path, repository root as the working directory: the scripts
    resolve their own fixtures from `__file__`, so they do not need a `cd`, and
    running them from the root is what the comments in them promise.
    """
    path = EXAMPLES / name
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
    check(
        "the audit ran over a full pose set",
        code == 0 and clashing is not None and audited > 0,
        f"exit {code}, {len(downhills)} poses"
        + (f"; stderr ended {err.splitlines()[-1]!r}" if err else ""),
    )
    check(
        "no returned pose has two heavy atoms inside each other",
        clashing == 0,
        f"{clashing:.0f} of {audited:.0f} poses clash" if clashing is not None
        else "no clash count printed",
    )
    check(
        "every returned pose is a genuine minimum of the interpolated field",
        bool(downhills) and all(d <= 1e-6 for d in downhills),
        f"largest reachable downhill step {max(downhills):.2e} kcal/mol over "
        f"{len(downhills)} poses" if downhills else "no downhill column printed",
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
    # this gate treat a negative value as fine, because both test
    # `downhill > 1e-6`, and a mutation that printed -0.5 slipped through the
    # pair of them. It cannot be produced by the line search as written, so
    # this is a guard on the report rather than on the physics.
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

    # --- 5. malformed input -----------------------------------------------------
    section("robustness_check.py: malformed input raises, never aborts")
    out, code, elapsed, err = run_script("robustness_check.py")
    report("robustness_check.py", out, code, elapsed, err.splitlines()[-1] if err else "")
    oks = len(re.findall(r"^\s*ok\s+", out, re.M))
    fails = len(re.findall(r"^\s*FAIL", out, re.M))
    print(f"    {oks} cases reported ok, {fails} reported FAIL")
    check(
        "the script ran and reached its own verdict line",
        code == 0 and "catchable error" in out,
        f"exit {code}, {oks} cases ok, {fails} failing"
        + (f"; stderr ended {err.splitlines()[-1]!r}" if err else ""),
    )
    check(
        "every malformed input it tried produced a catchable error",
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
    check(
        "all five ran inside the hang guard, so all five are gated",
        len(CLOCKS) == 5 and all(c == 0 for _, _, c in CLOCKS),
        f"{len(CLOCKS)} of 5 scripts ran to completion with a zero exit code",
    )

    print("\n=== summary ===")
    print(f"  {CHECKS - len(FAILURES)} passed, {len(FAILURES)} failed, {CHECKS} checks")
    for f in FAILURES:
        print(f"    FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
