"""Launch `odgui` as a real subprocess and confirm a window actually comes up.

Running the installed console script end to end is the only way to check the
thing the user will type. This does not poke at MainWindow directly: it starts
`odgui -r ... -l ... -p ...`, waits, screenshots the window, and shuts it down
the way a person would -- with Alt+F4, i.e. closeEvent, which is where a
still-running QThread would abort the process.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
OUT = ROOT / "dist" / "odgui_launch"
OUT.mkdir(parents=True, exist_ok=True)

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

try:
    import ctypes

    user32 = ctypes.windll.user32
    while time.time() < deadline:
        if proc.poll() is not None:
            rc = proc.returncode
            print(f"process exited early with {rc}")
            break
        hwnd = user32.FindWindowW(None, "Open Docking Workbench")
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
            print(f"distinct colours in the captured window: {len(colours) if colours else '>16M'}")
            seen = True
            print("posting WM_CLOSE (the closeEvent path a person would use)…", flush=True)
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            try:
                rc = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print("the window ignored WM_CLOSE; forcing termination")
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

ok = seen and rc == 0
print("\nRESULT:", "PASS — odgui opens a real window and closes cleanly" if ok else "FAIL")
sys.exit(0 if ok else 1)
