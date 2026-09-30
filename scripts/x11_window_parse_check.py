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

Run:  python scripts/x11_window_parse_check.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "scripts" / "odgui_launch_check.py"
WINDOW_TITLE = "Open Docking Workbench"

FAILURES: list[str] = []
CHECKS = 0


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
        "WINDOW_TITLE": WINDOW_TITLE,
    }
    exec(compile(src[start:end], str(TARGET), "exec"), ns)  # noqa: S102
    for name in ("find_window_x11", "close_window_x11"):
        if name not in ns:
            sys.exit(f"{name} is no longer in {TARGET.name}; this check needs updating")
    return ns


def check(name, got, want):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if got == want else 'FAIL'}] {name}")
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


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


def main() -> int:
    ns = load_lookup()
    find = ns["find_window_x11"]

    print("=== xdotool present: the CI case, no window manager ===")
    ns["_run"] = stub({"xdotool": XDTOOL_OUT})
    check("xdotool is preferred and is named in the report",
          find(), ("29360134", "xdotool search"))

    print("\n=== xdotool absent, wmctrl present ===")
    ns["_run"] = stub({"wmctrl": WMCTRL_OUT})
    check("the id is taken from a four-field line",
          find(), ("0x03000002", "wmctrl -l"))

    print("\n=== only xwininfo present: it can find but not close ===")
    ns["_run"] = stub({"xwininfo": XWININFO_OUT})
    check("the id is carried down from the parent line",
          find(), ("0x1e1f9a4", "xwininfo -tree (cannot close)"))
    ns["_run"] = stub({"xwininfo": '0x1e1f9b0:\n   "Od Docking Workbench" (none)\n'})
    check("a near-miss title is rejected", find(), (None, None))

    print("\n=== nothing installed ===")
    ns["_run"] = stub({})
    check("no tool means no window, not a wrong window", find(), (None, None))

    print("\n=== one tool failing does not end the search ===")
    ns["_run"] = stub({"xdotool": None, "wmctrl": WMCTRL_OUT})
    check("a non-zero exit from xdotool falls through to wmctrl",
          find(), ("0x03000002", "wmctrl -l"))

    print("\n=== regression: titles that are not ours ===")
    ns["_run"] = stub({"wmctrl": "0x03000009  0  hostname  odgui\n"})
    check("a single-word title does not match", find(), (None, None))
    ns["_run"] = stub({"wmctrl": "0x0300000a  0  hostname  Open Docking Workbench Extra\n"})
    check("a longer title is not accepted as ours", find(), (None, None))
    ns["_run"] = stub({"xdotool": ""})
    check("an anchored regex that matches nothing yields no window",
          find(), (None, None))

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
        check("no tools at all means no close mechanism", close("0x1"), "none")

        # The defect this pins: `xdotool windowclose` destroys the X window
        # instead of asking Qt to close, so having it is not having a way to
        # close. Reporting a verified shutdown here would be reporting an
        # event that never happened.
        ns["shutil"] = Only({"xdotool", "xwininfo"})
        check("xdotool alone does not count as a close mechanism",
              close("0x1"), "none")
    finally:
        ns["shutil"] = real_shutil

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
