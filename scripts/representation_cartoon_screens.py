"""Render the cartoon so a human can look at it. Not a check.

Run it directly to write the PNGs:

    F:\\python310\\python.exe scripts\\representation_cartoon_screens.py

Written in the same turn as it is read. "The guard says the twist is 100.58
deg/residue" is a number; whether the picture reads as a helix is the claim the
number is standing in for, and only a person can close that gap. So this writes:

* `cartoon_whole.png`   -- the shipped receptor, cartoon, colour
* `cartoon_greyscale.png` -- the same frame with every vertex forced to one
  neutral grey, which is the black-and-white print and the colour-blind reader
  in one picture
* `ribbon_greyscale.png` -- the same frame drawn by `backbone_ribbon`, also
  grey, so the two can be put side by side
* `cartoon_closeup_*.png` -- a helix-rich stretch, a strand, and a coil, each
  framed on its own residues at a distance where the geometry is legible

The close-ups are the point of the file. Framed whole, a 46-residue protein at
900x700 is 19 px per residue and the difference between a spiral and a tube is
about four of them; the guards measure the geometry, but nobody can *see* a
twist at that scale, and a claim about legibility has to be looked at.
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

from opendocking.workbench import Camera, MoleculeView  # noqa: E402
from opendocking.workbench import cartoon_geometry as cg  # noqa: E402
from opendocking.workbench.geometry import ribbon as build_ribbon  # noqa: E402
from opendocking.workbench.structure import parse_structure, secondary_structure  # noqa: E402

from secondary_render import OffscreenRenderer, save_png  # noqa: E402

OUT = ROOT / "dist" / "cartoon"
OUT.mkdir(parents=True, exist_ok=True)

#: Every vertex this grey, so nothing in the picture can be read as a class
#: except its geometry. Matches the value the guard measures with.
GREY = {k: (0.62, 0.62, 0.62) for k in ("helix", "sheet", "coil")}

W, H = 900, 700
CLOSE_W, CLOSE_H = 760, 620


def camera_for(points: np.ndarray, *, pad: float = 2.4) -> Camera:
    """Frame a point cloud the way the window's Frame all does."""
    centre = points.mean(axis=0)
    radius = float(np.linalg.norm(points - centre, axis=1).max())
    cam = Camera()
    cam.center = centre.astype(np.float32)
    cam.distance = max(radius * pad, 4.0)
    return cam


def mesh_bounds(mesh) -> np.ndarray:
    return np.asarray(mesh.positions, np.float64)


def main() -> int:
    path = ROOT / "examples" / "1crn_prep.pdbqt"
    mol = MoleculeView.from_text(
        path.read_text(encoding="utf-8"), path.stem, (0.6, 0.65, 0.7), 0.30, "receptor"
    )
    trace = mol.structure.backbone()
    states = list(secondary_structure(trace, mol.structure.atoms))
    guide, sides, _ = mol._backbone_guide_and_sides()
    resids = [mol.structure.atoms[ca].resid for _, ca, _ in trace]

    counts = {k: states.count(k) for k in ("helix", "sheet", "coil")}
    print(f"receptor {len(mol.coords)} atoms, {len(trace)} residues, states {counts}")

    r = OffscreenRenderer.open(W, H)
    if r is None:
        print("no offscreen context:", __import__("secondary_render").LAST_PROBLEM)
        return 1

    written: list[str] = []

    # -- the whole molecule, three ways -------------------------------
    whole = cg.cartoon(guide, sides, states)
    cam = camera_for(np.asarray([a.xyz for a in mol.structure.atoms], np.float64))
    for name, mesh in (
        ("cartoon_whole", whole),
        ("cartoon_greyscale", cg.cartoon(guide, sides, states, colors=GREY)),
        ("ribbon_greyscale", build_ribbon(guide, sides, states, colors=GREY)),
    ):
        frame = r.render([(mesh, 1.0)], cam, W, H)
        written.append(save_png(OUT / f"{name}.png", frame))
        print(f"  {name:22s} {len(mesh.indices):5d} triangles -> {written[-1]}")

    # A second context at the close-up size. `OffscreenRenderer.render` reads
    # back exactly `width * height * 3` bytes from a framebuffer object sized
    # when the context was opened, so asking one context for two sizes fails
    # in a reshape rather than silently scaling.
    r.close()
    rc = OffscreenRenderer.open(CLOSE_W, CLOSE_H)
    if rc is None:
        print("no offscreen context at the close-up size:",
              __import__("secondary_render").LAST_PROBLEM)
        return 1

    # -- close-ups, one per class -------------------------------------
    # Each close-up is built from the real classification restricted to that
    # run's own residues, so the frame shows the geometry of that class on the
    # real protein rather than a synthetic straight trace. Framing the whole
    # molecule instead would put 19 px on a residue, and the difference between
    # a spiral and a tube is about four of those.
    jobs = []
    for lo, hi, state in cg.runs(states):
        if state in ("helix", "sheet"):
            jobs.append((lo, hi, state))

    coils = [(lo, hi) for lo, hi, s in cg.runs(states) if s == "coil"]
    if coils:
        jobs.append((*max(coils, key=lambda t: t[1] - t[0]), "coil"))

    for lo, hi, state in jobs:
        sel = list(range(lo, hi + 1))
        mesh = cg.cartoon(guide[sel], sides[sel], [states[i] for i in sel], colors=GREY)
        c = camera_for(mesh_bounds(mesh), pad=3.0)
        frame = rc.render([(mesh, 1.0)], c, CLOSE_W, CLOSE_H)
        name = f"closeup_{state}_{lo}_{hi}"
        written.append(save_png(OUT / f"{name}.png", frame))
        print(
            f"  {name:22s} residues "
            f"{resids[lo]}-{resids[hi]} ({state}, {hi - lo + 1} res), "
            f"{len(mesh.indices)} triangles -> {written[-1]}"
        )

    # The longest helix again, in colour, so the greyscale claim can be
    # compared against what a viewer actually sees in the window.
    helices = [(lo, hi) for lo, hi, s in cg.runs(states) if s == "helix"]
    if helices:
        lo, hi = max(helices, key=lambda t: t[1] - t[0])
        sel = list(range(lo, hi + 1))
        mesh = cg.cartoon(guide[sel], sides[sel], [states[i] for i in sel])
        c = camera_for(mesh_bounds(mesh), pad=3.0)
        frame = rc.render([(mesh, 1.0)], c, CLOSE_W, CLOSE_H)
        written.append(save_png(OUT / f"closeup_helix_{lo}_{hi}_colour.png", frame))
        print(f"  closeup_helix_{lo}_{hi}_colour (the same frame in colour) -> {written[-1]}")

    rc.close()

    missing = [p for p in written if not Path(p).is_file()]
    print(f"\n{len(written)} file(s) written, {len(missing)} missing after write")
    if missing:
        for p in missing:
            print("  MISSING:", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
