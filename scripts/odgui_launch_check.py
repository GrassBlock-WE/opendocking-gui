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
* **X11** -- `wmctrl -l` to list windows by title, `wmctrl -c` to send the
  close request through the window manager. There is no screenshot: capturing
  one needs ImageMagick, and the interaction check already covers rendering, so
  the report says so rather than pretending it looked.

If `wmctrl` is not installed the script says the graceful close could not be
checked and falls back to proving only that the process stayed alive. That is a
weaker claim and it is reported as one -- the alternative, crashing on
`ctypes.windll`, is not a check at all.
"""

from __future__ import annotations

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


def find_window_windows():
    """The HWND whose title matches, or None."""
    import ctypes

    return ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)


def find_window_x11() -> str | None:
    """The X11 window id whose title matches, or None.

    `wmctrl -l` prints `0x<id> <desktop> <host> <title>`; the title can contain
    spaces, so only the first field is treated as structured data.
    """
    if shutil.which("wmctrl") is None:
        return None
    try:
        out = subprocess.run(
            ["wmctrl", "-l"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[4].strip() == WINDOW_TITLE:
            return parts[0]
    return None


try:
    while time.time() < deadline:
        if proc.poll() is not None:
            rc = proc.returncode
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
            wid = find_window_x11()
            if wid:
                seen = True
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
                close_mechanism = "wmctrl -c (window manager close request)"
                print(
                    "asking the window manager to close it (the closeEvent path "
                    "a person would use)…",
                    flush=True,
                )
                subprocess.run(["wmctrl", "-i", "-c", wid], timeout=20, check=False)
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
        "\nno window manager tool is available, so the close path was not "
        "exercised; only 'the window came up' was verified",
        file=sys.stderr,
    )
    ok = rc is not None
    print(
        "\nRESULT:",
        "PARTIAL — odgui opened a window; its clean shutdown was not verified"
        if ok
        else "FAIL",
    )
    sys.exit(0 if ok else 1)

ok = seen and rc == 0
print(f"\nclosed via: {close_mechanism}")
print("\nRESULT:", "PASS — odgui opens a real window and closes cleanly" if ok else "FAIL")
sys.exit(0 if ok else 1)
