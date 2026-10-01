"""Launch `odgui` as a real subprocess and confirm a window actually comes up.

Running the installed console script end to end is the only way to check the
thing the user will type. This does not poke at MainWindow directly: it starts
`odgui -r ... -l ... -p ...`, waits for a window with the right title, and shuts
it down the way a person would -- by asking the window to close, which is the
`closeEvent` path where a still-running QThread would abort the process.

Finding the window and asking it to close is inherently platform-specific, and
that part is the *only* platform-specific part. The claim being checked is not.
So the two platforms get their own mechanism and the same verdict:

* **Windows** -- `FindWindowW` for the handle, `ImageGrab` for the screenshot,
  `PostMessage(WM_CLOSE)` to close.
* **X11** -- `xdotool search` to find the window, `wmctrl -c` to ask the window
  manager to close it. There is no screenshot: capturing one needs ImageMagick,
  and the interaction check already covers rendering, so the report says so
  rather than pretending it looked.

Getting this right took two corrections, both found by running it and neither
by reading it.

The first guess was `wmctrl` for everything. It asks the window manager for the
client list over EWMH, and CI runs this under `xvfb-run`, which starts a
display with **no window manager** -- so it found nothing and the loop ran out
its deadline. Finding is now `xdotool search`, which walks the X tree with
`XQueryTree` and needs no window manager.

The second guess was `xdotool windowclose`, on the theory that it delivers the
same event as `PostMessage(WM_CLOSE)`. It does not: it destroys the X window
outright, so Qt never runs `closeEvent` and the process keeps running with
nothing on screen. The log showed the window found at 0.6 s and the process
still alive at 20.6 s, which is a destroyed window rather than a slow close.
Closing therefore goes through `wmctrl -c`, which needs a window manager -- so
CI now starts a minimal one, because a real desktop has one and the check is
supposed to model a real desktop. `xdotool windowclose` is kept only to clean up
a window that could not be closed politely, and is never counted as a verified
shutdown.

If nothing can ask the window to close, the script says so and falls back to
proving only that the process stayed alive. That is a weaker claim and it is
reported as one.

**A run that stopped is not a run that passed.**

This file always ended with a `RESULT:` line, which is more than most, but the
line before it used to be the only thing standing between a crash and a silent
green: the whole body was module-level code, so a `FileNotFoundError` from a
missing `odgui`, or anything else raised, ended the process with a traceback
and no verdict at all. And three of the ways the run can end -- no window
appeared, the window ignored the close request, `odgui` exited on its own --
all printed the same bare `RESULT: FAIL`, so a reader could not tell a product
regression from a window that was never going to appear on this machine.

So every terminal state now carries its own verdict and its own reason, and a
window that never appears is separated from a product that is broken by asking
`odgui --check` whether this machine can produce a context at all.

Exit codes: ``0`` the window came up and closed cleanly, or it came up and the
report says which weaker claim was verified instead; ``1`` finished, the claim
did not hold; ``2`` did not finish.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: The one place in the tree that knows how to ask `odgui --check` what this
#: machine can do. It was a private copy in each of four check scripts, which
#: meant a fix to it had to be made four times and three of them would have been
#: missed. `python scripts/_gui_check.py` audits that these four still use it.
from _gui_check import odgui_check

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
OUT = ROOT / "dist" / "odgui_launch"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_TITLE = "Open Docking Workbench"
IS_WINDOWS = sys.platform == "win32"

#: How long to wait for the window to appear. Named so the reason quoted in a
#: verdict is the same number the loop actually used, rather than a second
#: literal that can drift from it.
WINDOW_DEADLINE = 60

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

# --- state, before the helpers on purpose ----------------------------------
#
# `x11_window_parse_check.py` execs the source between the `_run` helper and the
# watch loop below, so everything in that window has to be self-contained: it
# gets `re`, `shutil`, `subprocess`, `sys` and `WINDOW_TITLE`, and nothing else.
# Anything else in the slice fails at exec time with a `NameError` -- which is
# what happened when this block was left below the helpers, because `EX` is not
# in that namespace. So the launch arguments and the watch state live up here,
# outside the slice, exactly where they were before.
#
# The two anchors are matched as literal text, so this comment must not spell
# either of them out: a first attempt at documenting the coupling wrote the
# helper's name inline, and the guard then sliced the *comment* and died with a
# `SyntaxError`. That is the cost of a text-level dependency, and the way to
# live with it is not to quote the anchors.

args = [
    "odgui",
    "-r", str(EX / "rec_prep.pdbqt"),
    "-l", str(EX / "ibuprofen_prep.pdbqt"),
    "-p", str(EX / "poses.pdbqt"),
]
print("running:", " ".join(args), flush=True)

if shutil.which("odgui") is None:
    # Used to be a FileNotFoundError out of Popen: a traceback and no verdict,
    # for a machine that is simply missing the console script.
    print(
        "\nRESULT: DID NOT FINISH — `odgui` is not on PATH, so there was nothing "
        "to launch",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_INCOMPLETE)

proc = subprocess.Popen(args, cwd=str(ROOT))
shot = OUT / "window.png"
deadline = time.time() + WINDOW_DEADLINE
seen = False
rc = None
close_mechanism = "none"
# Did odgui die on its own while we were watching? This is the one thing that
# says something about its health, and it is the *only* thing that says
# something when no close mechanism exists: an exit code produced by our own
# terminate() below is evidence about us, not about odgui.
exited_early = False
#: Why the run ended the way it did, in a form the verdict can quote. Every way
#: out of the loop sets this; a verdict with no reason is what made the three
#: failure modes indistinguishable from one another.
outcome = ""
forced = False
#: Set when the machine, rather than the viewer, is why the window never came
#: up. The run then gets a skip verdict and its own exit code, because nothing
#: was verified either way and FAIL would be a claim it cannot make.
env_blocked = False
#: Set when the watch itself broke. Distinct from `outcome`, which describes a
#: verdict the run reached; this says there was no verdict to reach.
incomplete = ""

# ---------------------------------------------------------------------------
# The outcome ledger.
#
# This script used to end with a `RESULT:` line and nothing else, so its checks
# existed only as prose. Adding the two-stage `--check` payload guard below
# needs somewhere to record a failure, and a guard that can only print is a
# guard that cannot be counted.
#
# Placed here, before the X11 helpers, because `x11_window_parse_check.py` execs
# the source between the `_run` helper and the watch loop below: anything added
# inside that window has to stand alone with only `re`, `shutil`, `subprocess`,
# `sys` and `WINDOW_TITLE` in scope.
RESULTS: list[tuple[str, str, str]] = []


def ok(name: str, reason: str = "") -> bool:
    RESULTS.append(("PASS", name, reason))
    print(f"  [PASS] {name}" + (f"  — {reason}" if reason else ""))
    return True


def bad(name: str, reason: str) -> bool:
    RESULTS.append(("FAIL", name, reason))
    print(f"  [FAIL] {name}  — {reason}")
    return False


def skip(name: str, reason: str) -> bool:
    RESULTS.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  — {reason}")
    return False


def check_payload(answer) -> None:
    """`odgui --check` must report *both* stages, and the widget one must be real.

    This exists because of a specific mutant, and it is worth naming: if the
    launcher stopped running the widget probe once the cheap raw-context probe
    succeeded, `--check` would return `ok` and exit 0 **on a machine whose
    viewport cannot open** -- the original bug, wearing a new exit code. Nothing
    about the exit code catches that; only the payload does.

    So the assertion is about evidence rather than about the verdict:

    * a `raw` and a `widget` member must both be present;
    * if the raw stage got a context, the widget stage must have been *attempted*
      -- a `skipped` note there is a failure, not an excuse, because a working
      raw context is exactly the case where the widget probe is the only thing
      left to learn;
    * if the verdict is `ok`, the widget member must carry a GL version, because
      `ok` is a claim about the widget and not about OpenGL in general.

    When the raw stage itself failed there is nothing to say about stage 2, and
    that is a SKIP rather than a failure -- the machine could not answer.
    """
    detail = dict(getattr(answer, "detail", None) or {})
    raw = detail.get("raw")
    widget = detail.get("widget")
    if raw is None or widget is None:
        bad(
            "the --check payload reports both stages",
            f"raw present={raw is not None}, widget present={widget is not None}, "
            f"verdict={answer.verdict!r}, keys={sorted(detail)}",
        )
        return
    if not raw.get("gl"):
        skip(
            "the --check payload reports both stages",
            f"the cheap stage could not get a context either "
            f"(verdict {answer.verdict!r}), so there is nothing to say about "
            f"the widget stage",
        )
        return
    if widget.get("skipped"):
        bad(
            "the --check payload reports both stages",
            "the raw context worked, so the widget stage had to run, but it was "
            f"skipped: {widget['skipped']}",
        )
        return
    if answer.verdict == "no-widget":
        # The reason is what four other scripts print, so a reason that only
        # quotes the widget probe's own problem would tell every one of them
        # "this machine cannot give Qt an OpenGL context" -- on a machine whose
        # OpenGL demonstrably works. Requiring the raw stage's GL version to
        # appear checks the substance without depending on the wording.
        reason = getattr(answer, "reason", "") or ""
        if str(raw.get("gl")) not in reason:
            bad(
                "the no-widget reason carries the raw stage's evidence",
                f"verdict is no-widget but the reason does not mention the raw "
                f"context's GL {raw.get('gl')}: {reason!r}",
            )
            return
        ok(
            "the no-widget reason carries the raw stage's evidence",
            f"quotes raw GL {raw.get('gl')}",
        )
    if answer.verdict == "ok" and not widget.get("gl"):
        bad(
            "the --check payload reports both stages",
            f"the verdict is 'ok' but the widget stage recorded no GL version: "
            f"{widget}",
        )
        return
    ok(
        "the --check payload reports both stages",
        f"raw GL {raw.get('gl')}, widget GL {widget.get('gl') or 'none'}"
        + (f", child exit {widget['child_exit']}" if "child_exit" in widget else ""),
    )


def _nfail() -> int:
    """How many recorded checks failed. The verdict below refuses to pass if >0."""
    return sum(1 for tag, _, _ in RESULTS if tag == "FAIL")


def _summarise() -> None:
    """Counts, on every path out of this file including a broken watch."""
    npass = sum(1 for tag, _, _ in RESULTS if tag == "PASS")
    nfail = sum(1 for tag, _, _ in RESULTS if tag == "FAIL")
    nskip = sum(1 for tag, _, _ in RESULTS if tag == "SKIP")
    print()
    print(f"--- summary: {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks")
    for tag, name, reason in RESULTS:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {reason}")


# The machine's own answer to "can you start the viewer", asked once, up front.
# The window test below is the primary evidence -- it watches a real process --
# so this never decides the verdict. It supplies two things the window test
# cannot: the reason to print when no window ever appears, and the payload guard
# in `check_payload`, which is the only thing in the tree that can catch a
# `--check` that stopped running its widget stage.
#
# Placed *here*, above the X11 helpers, rather than next to the watch loop it
# feeds: the loop sits at module level because `x11_window_parse_check.py`
# slices this file from the `_run` helper to that loop, and a statement inside
# that window gets exec'd by the check. That was not a guess -- moving this
# block down put `odgui_check()` inside the slice and the guard failed with
# `NameError: name 'odgui_check' is not defined`, which is the same lesson as
# the earlier one about quoting a helper's name in a comment.
answer = odgui_check()
print(f"  --check says: {answer.describe()}")
check_payload(answer)


def _descendants(root_pid: int) -> set[int]:
    """Every PID whose ancestry reaches `root_pid`, `root_pid` included.

    Needed because the console script is two processes. `odgui` is a
    console-script shim, and the shim starts a *child* interpreter to run the
    launcher; the window belongs to the child, not to the process `Popen` handed
    back. Matching on `proc.pid` alone therefore found nothing at all --
    measured, not assumed: with an exact-PID guard the check reported "no window
    appeared within 60 s" on a run where the window was demonstrably up.

    Read from a Toolhelp32 snapshot rather than WMI, so it costs no import and no
    subprocess. A process that exits between the snapshot and the lookup simply
    is not in the set, which is the safe direction: the window is then not
    claimed as ours.
    """
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002

    class _Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.windll.kernel32
    snapshot = kernel.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == -1 or snapshot == 0xFFFFFFFF:
        return {root_pid}
    parents: dict[int, int] = {}
    try:
        entry = _Entry()
        entry.dwSize = ctypes.sizeof(_Entry)
        if kernel.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                parents[entry.th32ProcessID] = entry.th32ParentProcessID
                if not kernel.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel.CloseHandle(snapshot)

    family = {root_pid}
    # Repeat until it stops growing: a process tree is not one level deep, and
    # one pass would miss a grandchild -- which is exactly what the shim is.
    grew = True
    while grew:
        grew = False
        for child, parent in parents.items():
            if parent in family and child not in family:
                family.add(child)
                grew = True
    return family


def find_window_windows(pid: int | None = None):
    """The HWND whose title matches **and** whose owner is in `pid`'s tree.

    The title used to be the whole test, and that is not enough. A title lookup
    returns the first window carrying that title *whoever owns it*, so a second
    `odgui`, or another agent's `workbench_interaction_check.py` on the same
    desktop, is indistinguishable from the window this check started. Measured
    here, not assumed: over 42 launches, runs that matched a window within 0.3 s
    of spawning -- too fast to be the new process, whose own window appears at
    1.0-2.2 s -- failed 4 times in 14 (29%), while runs that matched their own
    window failed 1 time in 28 (3.6%).

    The failure mode is specific and nasty. `WM_CLOSE` goes to the *other*
    process's window, which closes in about 70 ms, and the process being watched
    is never asked to close at all. It then sits there until the deadline, gets
    terminated, and `TerminateProcess` reports exit code 1 -- so the check
    reports a viewer that will not close cleanly when the truth is that it was
    never asked.

    `pid=None` keeps the old title-only behaviour and exists so the mutation
    that proves this guard can turn it off; nothing on the happy path passes it.
    """
    import ctypes
    from ctypes import wintypes

    found: list[int] = []
    mine = _descendants(pid) if pid is not None else None

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _collect(hwnd, _lparam):
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            if buf.value == WINDOW_TITLE:
                owner = wintypes.DWORD()
                ctypes.windll.user32.GetWindowThreadProcessId(
                    hwnd, ctypes.byref(owner)
                )
                if mine is None or owner.value in mine:
                    found.append(hwnd)
                    return False  # stop walking the desktop
        return True

    ctypes.windll.user32.EnumWindows(_collect, 0)
    return found[0] if found else None


def _run(argv: list[str], timeout: int = 10) -> str | None:
    """stdout of a helper tool, or None if it is missing or unhappy."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def find_window_x11() -> tuple[str | None, str | None]:
    """The X window id whose title matches, and the tool that found it.

    Tries the mechanisms in the order that works without a window manager
    first, because CI has none.
    """
    # 1. xdotool walks the X tree itself (XQueryTree), so no WM is required.
    #    It applies --name as a regex, so the anchors are what stop a window
    #    called "Open Docking Workbench (crashed)" from being accepted.
    out = _run(["xdotool", "search", "--name", f"^{re.escape(WINDOW_TITLE)}$"])
    if out:
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            return ids[0], "xdotool search"

    # 2. wmctrl needs a window manager: EWMH _NET_CLIENT_LIST is the WM's to
    #    publish. This is why it alone made this step time out under xvfb.
    #
    #    `wmctrl -l` prints "<id> <desktop> <host> <title>", and the title
    #    contains spaces, so it must be the unsplit remainder: maxsplit=3
    #    leaves it whole in parts[3]. Requiring five fields here -- which is
    #    what this used to do -- can never match a four-field line, so the
    #    branch was dead even where a window manager was present.
    out = _run(["wmctrl", "-l"])
    if out:
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) == 4 and parts[3].strip() == WINDOW_TITLE:
                return parts[0], "wmctrl -l"

    # 3. xwininfo can at least show the window exists, which is half the claim.
    #    The tree puts the id on its own line and the title indented under it:
    #
    #        0x1e1f9a4:
    #           "Open Docking Workbench": ("odgui" "Workbench")  1294x858+0+0
    #
    #    so the id has to be carried down from the parent line rather than read
    #    off the title line.
    out = _run(["xwininfo", "-root", "-tree"])
    if out:
        current = None
        for line in out.splitlines():
            head = re.match(r"^\s*(0x[0-9a-fA-F]+):\s*$", line)
            if head:
                current = head.group(1)
                continue
            if current and f'"{WINDOW_TITLE}"' in line:
                return current, "xwininfo -tree (cannot close)"

    return None, None


def close_window_x11(wid: str) -> str:
    """Ask the window to close the way a person would. Returns what worked.

    Only `wmctrl -c` counts, and the reason matters. `xdotool windowclose` looks
    like the obvious alternative and is not: it destroys the X window outright
    instead of sending `WM_DELETE_WINDOW`, so Qt never runs `closeEvent` and
    the process sits there with no window. That is not a slower close, it is a
    different event entirely -- and CI proved it, holding the process alive for
    the full 20 s after the window had already gone. So it is used only to
    clean up a window we are about to abandon, and it is reported as what it is
    rather than as a shutdown that was verified.
    """
    if shutil.which("wmctrl") is not None:
        proc = subprocess.run(
            ["wmctrl", "-i", "-c", wid], capture_output=True, timeout=20
        )
        if proc.returncode == 0:
            return "wmctrl -i -c (window manager close request -> closeEvent)"
        print(f"wmctrl close failed: {proc.stderr.decode('utf-8', 'replace')[:200]}")
    return "none"


def abandon_window_x11(wid: str) -> None:
    """Destroy a window we could not close politely, so nothing is left mapped."""
    if shutil.which("xdotool") is not None:
        subprocess.run(
            ["xdotool", "windowclose", wid], capture_output=True, timeout=20,
            check=False,
        )


# --- the watch, at module level on purpose ---------------------------------
#
# `x11_window_parse_check.py` lifts the X11 helpers above out of this file by
# slicing its source between the `_run` helper and the watch loop below,
# because it cannot import this module -- importing it launches `odgui`. That
# is a real dependency rather than an accident, and it is a guard worth
# keeping, so the shape here is load-bearing: the `try:` at column 0 with the
# `while` indented under it. Refactoring this block into a `main()` function
# broke that check with a bare `ValueError: substring not found`, and the honest
# response to a guard that fires is to satisfy the guard, not to relax it.
#
# What the guard costs is the ability to wrap this block in a function-level
# `try`. The did-not-finish guarantee therefore comes from the `except` around
# the loop *body* below, which is enough: every way this loop can die is inside
# that body, and each one now records a reason and still reaches a verdict.

try:
    while time.time() < deadline:
        try:
            if proc.poll() is not None:
                rc = proc.returncode
                exited_early = True
                outcome = (
                    f"odgui exited on its own with {rc} before a window appeared"
                )
                print(f"process exited early with {rc}")
                break

            if IS_WINDOWS:
                # The PID is not a nicety: without it this matches any window on
                # the desktop carrying the same title, including another agent's.
                hwnd = find_window_windows(proc.pid)
                if hwnd:
                    import win32con
                    import win32gui
                    from PIL import ImageGrab

                    # Give the GL context and the first paint a moment.
                    time.sleep(3.0)
                    try:
                        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    except Exception as exc:
                        # The handle went stale between finding it and using it,
                        # because odgui exited during the sleep above. Measured
                        # here, not reasoned about: `GetWindowRect` raised
                        # `pywintypes.error(1400, invalid window handle)` on a run
                        # where odgui had already exited 1, and the whole check died
                        # on that line -- a traceback and no verdict, for what was
                        # really the fact that the viewer exited before it opened
                        # a window. The next pass sees `proc.poll()` and reports
                        # exactly that, or the loop runs out its deadline and
                        # reports that no window appeared; neither path can
                        # produce a PASS, so dropping the handle is not a retry
                        # that could turn a failure into a pass.
                        print(
                            f"the window handle went stale before it could be "
                            f"measured ({type(exc).__name__}); still watching"
                        )
                        hwnd = None
                    if hwnd:
                        w, h = right - left, bottom - top
                        print(f"window found: {w}x{h} at ({left},{top})")
                        img = ImageGrab.grab(bbox=(left, top, right, bottom))
                        img.save(shot)
                        print(f"screenshot: {shot}  ({img.size[0]}x{img.size[1]})")
                        colours = img.getcolors(maxcolors=1 << 22)
                        print(
                            "distinct colours in the captured window: "
                            f"{len(colours) if colours else '>16M'}"
                        )
                        seen = True
                        close_mechanism = "PostMessage(WM_CLOSE)"
                        print(
                            "posting WM_CLOSE (the closeEvent path a person would "
                            "use)…",
                            flush=True,
                        )
                        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                        try:
                            rc = proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            outcome = (
                                "the window ignored WM_CLOSE and had to be killed, "
                                "so the close path a person would use was not "
                                "exercised"
                            )
                            print(
                                "the window ignored WM_CLOSE; forcing termination"
                            )
                            forced = True
                            break
                        break
            else:
                wid, found_by = find_window_x11()
                if wid:
                    seen = True
                    print(f"window {wid} found by {found_by}")
                    if shutil.which("import") is not None:
                        # ImageMagick's `import` is the one capture tool that is
                        # usually already there. Best effort: no screenshot is not a
                        # failure, and the report says which case this was.
                        try:
                            subprocess.run(
                                ["import", "-window", wid, str(shot)],
                                check=True,
                                timeout=20,
                                capture_output=True,
                            )
                            print(f"screenshot: {shot}")
                        except (OSError, subprocess.SubprocessError) as exc:
                            print(f"screenshot unavailable ({exc}); continuing")
                    else:
                        print(
                            "no screenshot tool (imagemagick) installed; skipping "
                            "capture"
                        )
                    # Let startup finish before closing. A close request delivered
                    # while the GL context is still coming up is not a fair test of
                    # the close path.
                    time.sleep(3.0)
                    close_mechanism = close_window_x11(wid)
                    if close_mechanism == "none":
                        print(
                            "wmctrl is unavailable or failed, so nothing can ask "
                            "the window to close through its window manager",
                            file=sys.stderr,
                        )
                        abandon_window_x11(wid)
                    else:
                        print(
                            f"asking the window to close via {close_mechanism}…",
                            flush=True,
                        )
                    try:
                        rc = proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        outcome = (
                            "the window ignored the close request and had to be "
                            "killed, so the close path was not exercised"
                        )
                        print(
                            "the window ignored the close request; forcing "
                            "termination"
                        )
                        forced = True
                        break
                    break
        except BaseException as exc:  # noqa: BLE001 - the point is to never be silent
            # The one guarantee this file owes its reader: a watch that breaks is
            # reported as a watch that broke, with the reason, and never as a
            # window that was or was not found.
            import traceback

            traceback.print_exc()
            incomplete = f"{type(exc).__name__}: {exc}"
            print(
                f"\nRESULT: DID NOT FINISH — the watch broke before it could "
                f"reach a verdict: {incomplete}",
                file=sys.stderr,
            )
            break
        time.sleep(0.5)
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
    print(f"exit code: {rc}")

if incomplete:
    # The watch already printed its verdict; nothing below may overwrite it.
    _summarise()
    raise SystemExit(EXIT_INCOMPLETE)

if not seen and not exited_early and not forced:
    # The window never turned up and odgui was still running when the deadline
    # passed. That is the one outcome where the machine, rather than the viewer,
    # is the likely cause, so the machine is asked -- and if the machine cannot
    # answer, the run gets its own verdict below instead of being folded into
    # FAIL, which is what used to happen and which made "the viewer is broken"
    # indistinguishable from "this machine has no OpenGL".
    #
    # This does not weaken the check: on CI `odgui --check` is already a gate of
    # its own (see `.github/workflows/ci.yml`), so a run that skips here is a run
    # whose machine has already failed a different step, and a run that fails
    # here on a machine whose `--check` passes is a real failure.
    if answer.can_make_context:
        env_blocked = False
        outcome = (
            f"no window appeared within {WINDOW_DEADLINE:.0f} s, on a machine "
            f"where `odgui --check` says it can create a context "
            f"({answer.describe()}) — so the viewer is not starting its window"
        )
    else:
        env_blocked = True
        outcome = (
            f"no window appeared within {WINDOW_DEADLINE:.0f} s, and this machine "
            f"cannot start the viewer here ({answer.describe()}), so whether the "
            f"viewer would have opened a window is unknown"
        )

if env_blocked:
    print(f"\nRESULT: SKIP — {outcome}")
    _summarise()
    raise SystemExit(EXIT_INCOMPLETE)

if seen and close_mechanism == "none":
    # The window is up but nothing could ask it to close. Say so instead of
    # reporting a graceful shutdown that was never tested.
    print(
        "\nnothing here can ask the window to close, so the close path was not "
        "exercised; only 'the window came up and stayed up' was verified",
        file=sys.stderr,
    )
    # Deliberately not `rc is not None`: by this point odgui has usually been
    # killed by the finally block above, so that exit code is ours. What is
    # still evidence is that it was still running when we found its window.
    #
    # Named `partial_ok` rather than `ok`: a local `ok` here would shadow the
    # ledger's `ok()` function for the rest of the module, so the next `ok(...)`
    # call would rebind a bool over a function. That is the kind of shadowing
    # that only bites on the next edit, in a different place.
    partial_ok = not exited_early
    print(
        "\nRESULT:",
        "PARTIAL — odgui opened a window and stayed up; its clean shutdown was "
        "not verified"
        if partial_ok
        else f"FAIL — {outcome or 'odgui exited on its own before a window was found'}",
    )
    _summarise()
    raise SystemExit(EXIT_OK if partial_ok else EXIT_FAILED)

window_ok = seen and rc == 0
print(f"\nclosed via: {close_mechanism}")
if window_ok and _nfail() == 0:
    print("\nRESULT: PASS — odgui opens a real window and closes cleanly")
    _summarise()
    raise SystemExit(EXIT_OK)
if window_ok:
    # The window behaved, but a recorded check did not. Reporting PASS here
    # would let a red check ride out under a green verdict, which is the
    # vacuous-green failure this file exists to stop.
    print(
        f"\nRESULT: FAIL — the window opened and closed cleanly, but "
        f"{_nfail()} recorded check(s) did not hold"
    )
    _summarise()
    raise SystemExit(EXIT_FAILED)
print(
    f"\nRESULT: FAIL — {outcome or 'the window came up but odgui did not exit cleanly'}"
)
_summarise()
raise SystemExit(EXIT_FAILED)
