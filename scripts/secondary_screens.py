"""Render the shipped receptor's ribbon in each secondary-structure class.

Run it directly to write the PNGs a human can look at:

    F:\\python310\\python.exe scripts\\secondary_screens.py

Not a check. Its whole job is to produce pictures, because "the mask changed"
is not evidence that a helix looks like a helix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dock-py" / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from opendocking.workbench import Camera  # noqa: E402
from opendocking.workbench.structure import (  # noqa: E402
    parse_structure,
    secondary_structure,
)

from secondary_render import (  # noqa: E402
    OffscreenRenderer,
    flat_states,
    ribbon_for_states,
    save_png,
)

OUT = ROOT / "dist" / "secondary_structure"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 900, 700


def camera_for(structure) -> Camera:
    """Frame the whole molecule, the way the window's Frame all does."""
    xyz = np.asarray([a.xyz for a in structure.atoms], np.float64)
    centre = xyz.mean(axis=0)
    radius = float(np.linalg.norm(xyz - centre, axis=1).max())
    cam = Camera()
    cam.center = centre.astype(np.float32)
    cam.distance = radius * 2.6
    return cam


def main() -> int:
    text = (ROOT / "examples" / "1crn_prep.pdbqt").read_text(encoding="utf-8")
    st = parse_structure(text, "1crn_prep")
    trace = st.backbone()
    states = secondary_structure(trace, st.atoms)
    cam = camera_for(st)

    r = OffscreenRenderer.open(W, H)
    if r is None:
        print("no offscreen context:", __import__("secondary_render").LAST_PROBLEM)
        return 1

    jobs = [
        ("ss_real", states),
        ("ss_helix_only", flat_states(len(trace), "helix")),
        ("ss_sheet_only", flat_states(len(trace), "sheet")),
        ("ss_coil_only", flat_states(len(trace), "coil")),
    ]
    frames = {}
    for name, sts in jobs:
        mesh = ribbon_for_states(st, sts)
        frame = r.render([(mesh, 1.0)], cam, W, H)
        save_png(OUT / f"{name}.png", frame)
        frames[name] = frame
        counts = {k: sts.count(k) for k in ("helix", "sheet", "coil")}
        print(f"{name:16s} {counts}  ->  {OUT / (name + '.png')}")
    r.close()

    real = frames["ss_real"]
    for name in ("ss_helix_only", "ss_sheet_only", "ss_coil_only"):
        print(f"differing px, real vs {name}: {int((real != frames[name]).any(axis=-1).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
