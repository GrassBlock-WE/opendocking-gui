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
* **X11** -- `xdotool search` to find the window, `xdotool windowclose` to send
  it a `WM_DELETE_WINDOW` client message. There is no screenshot: capturing one
  needs ImageMagick, and the interaction check already covers rendering, so the
  report says so rather than pretending it looked.

`xdotool` is the mechanism that matters, and the reason is that CI runs this
under `xvfb-run`, which starts a display with **no window manager**. `wmctrl`
asks the window manager for the client list over EWMH, so with no WM it finds
nothing and the loop above it can only spin until the deadline. `xdotool`
walks the X tree with `XQueryTree` and addresses the window directly, so it
works with or without a WM. That is also why `xdotool windowclose` is the
right analogue of `PostMessage(WM_CLOSE)`: both deliver a close request to the
window itself rather than asking someone else to deliver it, so both land on
Qt's `closeEvent`. `wmctrl` is kept as a fallback, and `xwininfo` as a way to
at least prove the window exists when neither of the others is installed.

If nothing can ask the window to close, the script says the graceful close
could not be checked and falls back to proving only that the process stayed
alive. That is a weaker claim and it is reported as one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
OUT = ROOT / "dist" / "odgui_launch"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_TITLE = "Open Docking Workbench"
IS_WINDOWS = sys.platform == "win32"

args = [
    "odgui",
    "-r", str(EX / "rec_prep.pdbqt"),
    "-l", str(EX / "ibuprofen_prep.pdbqt"),
    "-p", str(EX / "poses.pdbqt"),
]
print("running:", " ".join(args), flush=True)

proc = subprocess.Popen(args, cwd=str(ROOT))
shot = OUT / "window.png"
deadline = time.time() + 60
seen = False
rc = None
close_mechanism = "none"
# Did odgui die on its own while we were watching? This is the one thing that
# says something about its health, and it is the *only* thing that says
# something when no close mechanism exists: an exit code produced by our own
# terminate() below is evidence about us, not about odgui.
exited_early = False


def find_window_windows():
    """The HWND whose title matches, or None."""
    import ctypes

    return ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)


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

    `xdotool windowclose` sends WM_DELETE_WINDOW straight to the window, which
    is the same event `PostMessage(WM_CLOSE)` delivers on Windows and the same
    one that runs Qt's `closeEvent`.
    """
    if shutil.which("xdotool") is not None:
        proc = subprocess.run(
            ["xdotool", "windowclose", wid], capture_output=True, timeout=20
        )
        if proc.returncode == 0:
            return "xdotool windowclose (WM_DELETE_WINDOW -> closeEvent)"
        print(f"xdotool windowclose failed: {proc.stderr.decode('utf-8', 'replace')[:200]}")
    if shutil.which("wmctrl") is not None:
        subprocess.run(["wmctrl", "-i", "-c", wid], timeout=20, check=False)
        return "wmctrl -i -c (window manager close request)"
    return "none"


try:
    while time.time() < deadline:
        if proc.poll() is not None:
            rc = proc.returncode
            exited_early = True
            print(f"process exited early with {rc}")
            break

        if IS_WINDOWS:
            hwnd = find_window_windows()
            if hwnd:
                # Give the GL context and the first paint a moment.
                time.sleep(3.0)
                import win32con
                import win32gui
                from PIL import ImageGrab

                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
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
                    "posting WM_CLOSE (the closeEvent path a person would use)…",
                    flush=True,
                )
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                try:
                    rc = proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print("the window ignored WM_CLOSE; forcing termination")
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
                    print("no screenshot tool (imagemagick) installed; skipping capture")
                close_mechanism = close_window_x11(wid)
                if close_mechanism == "none":
                    print(
                        "neither xdotool nor wmctrl is available, so nothing can "
                        "ask the window to close",
                        file=sys.stderr,
                    )
                else:
                    print(f"asking the window to close via {close_mechanism}…", flush=True)
                try:
                    rc = proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print("the window ignored the close request; forcing termination")
                    break
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
    ok = not exited_early
    print(
        "\nRESULT:",
        "PARTIAL — odgui opened a window and stayed up; its clean shutdown was "
        "not verified"
        if ok
        else f"FAIL — odgui exited on its own with {rc} before a window was found",
    )
    sys.exit(0 if ok else 1)

ok = seen and rc == 0
print(f"\nclosed via: {close_mechanism}")
print("\nRESULT:", "PASS — odgui opens a real window and closes cleanly" if ok else "FAIL")
sys.exit(0 if ok else 1)
