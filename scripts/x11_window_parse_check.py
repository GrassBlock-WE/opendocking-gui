"""Reverse-verify the X11 window lookup in the launch check, on any platform.

The launch check is the only one that has to find a window belonging to a
*different* process, so it is the only one whose correctness depends on parsing
another tool's output. That parsing cannot run on Windows, and on CI it ran
wrong for a long time without anyone noticing: the step was best-effort, so a
check that never once succeeded looked the same as one that did.

This checks the parsing directly, with no X server involved. Each helper tool
is stubbed with the format it really prints, and each case asserts both that
the right window id comes back and that a near-miss title does not.

Two of these cases are regressions for specific defects:

* ``wmctrl -l`` prints four fields -- ``<id> <desktop> <host> <title>`` -- and
  the title holds spaces, so it has to be the unsplit remainder. The parser
  required a five-field split, which no four-field line can produce, so that
  branch was unreachable.
* ``xwininfo -root -tree`` puts the window id on its own line and the title
  indented beneath it, so the id has to be carried down from the parent.

A third pins the close path: ``xdotool`` being installed must not be mistaken
for a way to close a window. ``xdotool windowclose`` destroys the X window, so
Qt never runs ``closeEvent`` and the process survives with nothing on screen --
a different event from the one being tested, not a slower version of it.

The last check is this file's own pin, and the one before it tries to break
that pin on purpose: a copy of this file is run twice in a throwaway
directory, once whole and once with one ``check()`` call site spliced out, and
the second run has to come out red on the pin. A pin that nothing can
contradict is a number in a comment.

Run:  python scripts/x11_window_parse_check.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "odgui_launch_check.py"

#: The title the fixtures below were written against -- *not* the title the
#: launch check is currently looking for. Those are two different facts and
#: conflating them is what this file used to do: it injected its own copy into
#: the namespace the lifted helpers run in, so every title assertion was
#: measured against a constant that lived in this file. The product's
#: `WINDOW_TITLE` sits above the extracted slice (it is a module-level
#: assignment, and the slice starts at `def _run(`), so `exec` never rebound it
#: and the injected value won. Renaming the product's window -- which decides
#: whether the launch check can find its own window at all -- left this file
#: reporting 11/11.
#:
#: It is kept only so `check()` can say out loud when the two have parted. The
#: helpers are given the *product's* value; see `load_lookup()`.
WINDOW_TITLE = "Open Docking Workbench"

FAILURES: list[str] = []
CHECKS = 0
#: The call-site census this file is supposed to have, declared in the file it
#: counts rather than in a table another file owns.
#: Measured by walking this file's own syntax tree and comparing against this
#: block, so there is no path by which the declaration satisfies the auditor on
#: its own: a `check()` site added, deleted or wrapped in an `if` moves the
#: derived census and not this one. The two guarded sites are the two inside
#: the `try:` that hands the close path a stubbed `shutil`, and they are the
#: reason the digest is not the empty string's; this file has no `if` anywhere
#: around a `check()` call, so the `try:` accounts for the whole guarded column.
#: **This file has an `EXPECTED_CHECKS` now, and the reason it did not is worth
#: keeping.** A pin is a *run's* result count, and unlike the census above it
#: cannot be derived from the source. Every `check()` here is called exactly
#: once, so the two numbers would have happened to agree at 14 -- which is
#: exactly the coincidence that should not be taken for a measurement, and a
#: number nobody ran is not one. So the number below was taken from a run
#: instead of from the argument that the two would agree.
#:
#: **11 -> 12 unconditional, and the digest has not moved once.** The added
#: site is the self-test's, and it is a bare `check()` statement in `main()`
#: like every other one here -- the recursion guard that stops two copies
#: spawning each other lives inside the helper it calls, not around the call
#: site, which is why the `if` count above is still zero. The two guarded
#: sites are untouched, so `sha256:cef6d60f...` is what the file declared
#: before this round. The whole pair was re-derived by lifting the auditor's
#: own `_sites_of` and `_guards_digest` out of its source and running them
#: here, rather than by reimplementing them and comparing two numbers this
#: file wrote twice.
#:
#: GATE-DECLARE 1
#: sites: 12 unconditional + 2 guarded
#: guards: sha256:cef6d60f6d78425fd0fc4cf3491b54593342f03020649a99f262004e1e071280

#: How many checks this file records, **counted from a run**.
#:
#: `F:\python310\python.exe scripts\x11_window_parse_check.py` with
#: `PYTHONIOENCODING=utf-8` and `PYTHONPATH=dock-py\python`, on a Windows
#: machine with no X server and none needed. The number below was read off
#: the run that printed `12/14 passed (expected 13)` with this constant still
#: at 13 -- a run that was red *on this line*, which is the only kind of run
#: that measures the count instead of restating it. It said 14, the tally
#: check printed `got 14, want 13`, and 14 is what is declared here. The run
#: that confirms the pin is a *later* run than that one, and it is the one
#: that reports `14/14 passed`, exit 0.
#:
#: **13 -> 14, and the one is the self-test.** `pin_is_load_bearing()` copies
#: this file into a throwaway directory, runs that copy, and runs it again
#: with one `check()` call site spliced out. It is a check like any other
#: here, so it is counted: a change to `check()`'s signature that added a
#: witness and left this line alone would have gone red, which is exactly
#: what the run above shows.
#:
#: **What the fourteen are, because "a run" is not a derivation.** Ten
#: unconditional case sites in `main`, the two inside the `try:` that hands the
#: close path a stubbed `shutil`, the self-test, and the tally itself. No
#: `check()` here sits in a loop, in an `if`, or in data of any kind, so 14 is
#: not a reading of this machine: it is the number this file's shape produces
#: anywhere, and the census above is the derivation that says so where it can
#: be checked.
#:
#: **The ways a run can fail to reach the tally are crashes rather than
#: shortfalls,** which is the distinction that keeps this pin from having to
#: absorb them. `load_lookup()` calls `sys.exit` if the launch check's helpers
#: move, the close path's `finally` restores `shutil` before the tally runs,
#: and the self-test's `TemporaryDirectory` is emptied on the way out. None
#: of them leaves a smaller number behind, so none is a count this constant
#: has to be widened to survive. The one shortfall it does absorb is a skipped
#: self-test: a run with `OD_X11_PIN_CHILD` set still records all fourteen and
#: checks one fewer thing, which `CHILD_FLAG` above says out loud.
EXPECTED_CHECKS = 14


def product_window_title(src: str) -> str:
    """The launch check's own `WINDOW_TITLE`, read out of its module body.

    Lifted the same way the helpers are, and for the same reason: importing the
    target would run it. Read from the AST rather than by regex because this
    file is the one that was bitten by a constant that looked right and was not
    the product's -- a regex here would be the same mistake in a new place.
    """
    import ast

    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "WINDOW_TITLE" for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                break
    sys.exit(
        f"{TARGET.name} has no module-level WINDOW_TITLE literal; this check "
        f"needs updating rather than guessing which title the launch check "
        f"looks for"
    )


def load_lookup():
    """Pull the X11 helpers out of the launch check without running it.

    The launch check launches odgui at import time, so importing it is not an
    option; the helpers are lifted from the source instead. If the names move,
    this fails loudly rather than quietly testing nothing.
    """
    src = TARGET.read_text(encoding="utf-8")
    start = src.index("def _run(")
    end = src.index("try:\n    while time.time() < deadline:")
    ns = {
        "re": re,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        # The product's constant, not this file's. The lifted slice does not
        # define it, so whichever value is put here is the one every title
        # assertion below is actually measured against.
        "WINDOW_TITLE": product_window_title(src),
    }
    exec(compile(src[start:end], str(TARGET), "exec"), ns)  # noqa: S102
    for name in ("find_window_x11", "close_window_x11"):
        if name not in ns:
            sys.exit(f"{name} is no longer in {TARGET.name}; this check needs updating")
    return ns


def check(name, ok, detail):
    """Record one check. The verdict arrives already decided, by the caller.

    The house shape, and the reason this file uses it now: the comparison
    belongs at the call site, so `EXPECTED_CHECKS` is named inside a
    `Compare` there rather than handed in as an argument. The previous
    `(name, got, want)` helper compared inside its own body, which made this
    file's pin an *argument* at a call site and left the auditor unable to
    find a decision anywhere in the file that read the constant -- a pin
    that was enforced, and invisible to the thing that checks pins.
    """
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def stub(responses):
    def _run(argv, timeout=10):
        return responses.get(argv[0])

    return _run


XDTOOL_OUT = "29360134\n29360135\n"
WMCTRL_OUT = (
    "0x03000001  0  hostname  Od Docking Workbench\n"
    "0x03000002  0  hostname  Open Docking Workbench\n"
    "0x03000003  0  hostname  odgui\n"
)
XWININFO_OUT = (
    "0x1e1f9a4:\n"
    '   "xdg-desktop-portal" (has no name field)\n'
    '   "Open Docking Workbench": ("odgui" "Workbench")  1294x858+0+0  +0+0\n'
    "   0x0 0 Roots\n"
)

#: The one call site `pin_is_load_bearing()` removes, named by the string that
#: site passes as its name. A case site rather than the pin: removing the pin
#: would remove the witness, and the claim under test is that the *rest* of
#: the file notices a site going missing.
MUTATED_SITE = "a longer title is not accepted as ours"

#: Set on a copy of this file so the self-test runs one level deep and never
#: two. Two copies each spawning the other is not a self-test. It is a way a
#: run can be made vacuous, and it is said here rather than left for a reader
#: to find: setting it in the environment turns this one check into a green
#: that measured nothing. Nothing else in the file reads it.
CHILD_FLAG = "OD_X11_PIN_CHILD"


def _tail(text: str) -> str:
    """The last non-empty line of a child's output, ASCII-safe.

    A copy of this file prints nothing but ASCII, but the child being read
    here may have died instead, and a traceback quoting a path off this
    machine must not turn into a UnicodeEncodeError inside the check that is
    trying to report it.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    last = lines[-1] if lines else "<no output>"
    return last.encode("ascii", "replace").decode("ascii")


def _drop_call_site(src: str, name: str):
    """`src` with the single `check(...)` statement named `name` removed.

    Splices that one statement's own line range out and leaves a `pass` in
    its place, so the copy still parses and still runs all the way to its
    tally: the witness has to be a site that stops running, not a file that
    dies before the pin can read anything. Returns `(None, why)` rather than
    a guess when the name does not resolve to exactly one site, so a rename
    elsewhere cannot quietly turn this into some other mutation while the
    check above it still reports a mutation.
    """
    import ast

    sites = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and getattr(n.value.func, "id", None) == "check"
             and n.value.args and isinstance(n.value.args[0], ast.Constant)
             and n.value.args[0].value == name]
    if len(sites) != 1:
        return None, (f"{len(sites)} call site(s) in this file are named "
                      f"{name!r}, so there is no single site to remove and "
                      f"the mutation would not be the one this check claims "
                      f"to make")
    stmt = sites[0]
    lines = src.splitlines(keepends=True)
    removed = f"{' ' * stmt.col_offset}pass  # removed by the self-test\n"
    return "".join(lines[:stmt.lineno - 1] + [removed] + lines[stmt.end_lineno:]), None


def _run_copy(where: Path, src: str):
    """Lay `src` down as `where`'s copy of this file and run it.

    The launch check goes in beside it because the gate reads the X11 helpers
    out of that file's source: a copy in a directory without it exits through
    `load_lookup()` before the tally, which would prove nothing about a count
    that is never reached.
    """
    (where / "x11_window_parse_check.py").write_text(src, encoding="utf-8")
    (where / "odgui_launch_check.py").write_text(
        TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    env = dict(os.environ)
    env[CHILD_FLAG] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(where / "x11_window_parse_check.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=300,
    )


def pin_is_load_bearing() -> tuple[bool, str]:
    """Break the pin on purpose, in a throwaway copy, and read the result.

    Two runs, because one is not a measurement:

    * a **control** -- an unmutated copy, which has to be green. Without it a
      red from the second run would prove nothing, because a gate that is red
      in a sandbox for any unrelated reason passes this check.
    * the **witness** -- the same copy with one `check()` call site spliced
      out, which has to come out red, has to go red *on the pin*, and has to
      be missing the removed case's name from its output.

    The control is what makes the witness mean something; the third assertion
    is what makes it about the count rather than about the case that was
    taken out.
    """
    if os.environ.get(CHILD_FLAG):
        # One level, never two. In a copy this returns without running
        # anything, which is the one thing it is allowed to do: the copy is
        # the fixture here, and the file that has to be able to fail is this
        # one.
        return True, ("not exercised: this run is a copy of this file, and two "
                      "copies that each spawn the other is not a self-test")
    src = Path(__file__).resolve().read_text(encoding="utf-8")
    short = f"got {EXPECTED_CHECKS - 1}, want {EXPECTED_CHECKS}"
    with tempfile.TemporaryDirectory(prefix="x11_pin_") as tmp:
        where = Path(tmp) / "scripts"
        where.mkdir(parents=True)
        control = _run_copy(where, src)
        if control.returncode != 0:
            return False, (
                f"the unmutated control copy exited {control.returncode}, so a "
                f"red from the mutated copy would not be about the mutation. "
                f"Its last line was {_tail(control.stdout)!r}")
        mutated, why = _drop_call_site(src, MUTATED_SITE)
        if mutated is None:
            return False, why
        witness = _run_copy(where, mutated)
        problems = []
        if witness.returncode == 0:
            problems.append(
                f"it exited 0, so deleting the {MUTATED_SITE!r} call site cost "
                f"the run nothing and this pin is not load-bearing")
        if short not in witness.stdout:
            problems.append(
                f"its output does not contain {short!r}, which is the line the "
                f"pin prints when the tally and the declaration disagree")
        if MUTATED_SITE in witness.stdout:
            problems.append(
                f"its output still names the {MUTATED_SITE!r} case, so that "
                f"site ran anyway and the red is coming from somewhere else")
        if problems:
            return False, "; ".join(problems)
        return True, (f"an unmutated copy exited 0, and a copy with the "
                      f"{MUTATED_SITE!r} call site spliced out exited "
                      f"{witness.returncode} with {short!r} as the only failure "
                      f"line")


def main() -> int:
    ns = load_lookup()
    find = ns["find_window_x11"]

    print("=== the title the lifted helpers are actually looking for ===")
    # Pinned first, before any of the cases below, because every case below is
    # measured against this one value. If the product's window title and the
    # title these fixtures were written against have parted, the cases would
    # otherwise go red one at a time with messages about parsing, when the only
    # thing that changed was the constant.
    got, want = ns["WINDOW_TITLE"], WINDOW_TITLE
    check(
        "the title these fixtures were written against is the one the launch "
        "check looks for",
        got == want,
        f"got {got!r}, want {want!r}",
    )

    print("=== xdotool present: the CI case, no window manager ===")
    ns["_run"] = stub({"xdotool": XDTOOL_OUT})
    got, want = find(), ("29360134", "xdotool search")
    check("xdotool is preferred and is named in the report",
          got == want, f"got {got!r}, want {want!r}")

    print("\n=== xdotool absent, wmctrl present ===")
    ns["_run"] = stub({"wmctrl": WMCTRL_OUT})
    got, want = find(), ("0x03000002", "wmctrl -l")
    check("the id is taken from a four-field line",
          got == want, f"got {got!r}, want {want!r}")

    print("\n=== only xwininfo present: it can find but not close ===")
    ns["_run"] = stub({"xwininfo": XWININFO_OUT})
    got, want = find(), ("0x1e1f9a4", "xwininfo -tree (cannot close)")
    check("the id is carried down from the parent line",
          got == want, f"got {got!r}, want {want!r}")
    ns["_run"] = stub({"xwininfo": '0x1e1f9b0:\n   "Od Docking Workbench" (none)\n'})
    got, want = find(), (None, None)
    check("a near-miss title is rejected", got == want,
          f"got {got!r}, want {want!r}")

    print("\n=== nothing installed ===")
    ns["_run"] = stub({})
    got, want = find(), (None, None)
    check("no tool means no window, not a wrong window", got == want,
          f"got {got!r}, want {want!r}")

    print("\n=== one tool failing does not end the search ===")
    ns["_run"] = stub({"xdotool": None, "wmctrl": WMCTRL_OUT})
    got, want = find(), ("0x03000002", "wmctrl -l")
    check("a non-zero exit from xdotool falls through to wmctrl",
          got == want, f"got {got!r}, want {want!r}")

    print("\n=== regression: titles that are not ours ===")
    ns["_run"] = stub({"wmctrl": "0x03000009  0  hostname  odgui\n"})
    got, want = find(), (None, None)
    check("a single-word title does not match", got == want,
          f"got {got!r}, want {want!r}")
    ns["_run"] = stub({"wmctrl": "0x0300000a  0  hostname  Open Docking Workbench Extra\n"})
    got, want = find(), (None, None)
    check("a longer title is not accepted as ours", got == want,
          f"got {got!r}, want {want!r}")
    ns["_run"] = stub({"xdotool": ""})
    got, want = find(), (None, None)
    check("an anchored regex that matches nothing yields no window",
          got == want, f"got {got!r}, want {want!r}")

    print("\n=== the close path refuses to claim success it cannot deliver ===")
    close = ns["close_window_x11"]
    real_shutil = ns["shutil"]

    class Only:
        def __init__(self, names):
            self.names = names

        def which(self, name):
            return f"/usr/bin/{name}" if name in self.names else None

    try:
        ns["shutil"] = Only(set())
        got, want = close("0x1"), "none"
        check("no tools at all means no close mechanism", got == want,
              f"got {got!r}, want {want!r}")

        # The defect this pins: `xdotool windowclose` destroys the X window
        # instead of asking Qt to close, so having it is not having a way to
        # close. Reporting a verified shutdown here would be reporting an
        # event that never happened.
        ns["shutil"] = Only({"xdotool", "xwininfo"})
        got, want = close("0x1"), "none"
        check("xdotool alone does not count as a close mechanism",
              got == want, f"got {got!r}, want {want!r}")
    finally:
        ns["shutil"] = real_shutil

    # The witness, before the pin it is a witness *for*. Placed here rather
    # than above the cases so that everything it copies has already run: the
    # copy it mutates is this file as it stands at this point, not as it
    # stood when the file was opened.
    ok, detail = pin_is_load_bearing()
    check("this file's own pin goes red when a call site stops running",
          ok, detail)

    # The pin, asserted on the way out rather than only in the declaration.
    #
    # **Written in the house `(name, ok, detail)` shape, which this file did
    # not use until now.** It used to be `(name, got, want)` with the
    # comparison inside the helper, so `EXPECTED_CHECKS` arrived here as an
    # *argument*: enforced, and invisible to a reader looking for a decision
    # that reads the constant. The condition is now spelled here, where the
    # other gates spell theirs, and that is the whole of the difference --
    # `CHECKS + 1 == EXPECTED_CHECKS` asks exactly what `check` used to ask
    # of two arguments. The derivation still goes in the name, which is where
    # this file puts it, so a wrong total reads `got 13, want 14`.
    check("this file's own count is the count it declares: "
          f"{CHECKS} ran before this one, and {EXPECTED_CHECKS} are declared",
          CHECKS + 1 == EXPECTED_CHECKS,
          f"got {CHECKS + 1}, want {EXPECTED_CHECKS}")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed (expected {EXPECTED_CHECKS})")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
