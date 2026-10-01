"""Render the workbench's own ribbon offscreen, so its pixels can be counted.

# Why this exists

The interaction suite reads its frames out of a live `QOpenGLWidget`, which is
the right way to test a *window*. On a machine whose GL stack will not hand Qt a
surface there is no frame to read, so every pixel check in that suite skips and
says so. That is honest, but it means the one question this file exists to
answer -- *does the secondary structure change the picture?* -- would go
unasked on exactly the machines that cannot run a window.

So this renders the same geometry through the same code, offscreen:

* the mesh is the one :meth:`MoleculeView.backbone_ribbon` produced, i.e. the
  real `geometry.ribbon` with the real per-residue states;
* it is drawn by the real :func:`opendocking.workbench.app.draw_mesh`, with the
  real shaders from :func:`~opendocking.workbench.app.build_programs` and the
  real :func:`~opendocking.workbench.app.draw_background` backdrop;
* it is viewed by the real :class:`opendocking.workbench.Camera`.

Only the surface differs: a moderngl framebuffer instead of a Qt one. That
makes the numbers comparable with the rest of the suite -- same clear colour,
same fog, same mask rule -- while remaining available where no window is.

Nothing here decides what secondary structure *is*. It draws whatever states it
is handed, which is what makes it usable as a control: the same trace rendered
all-helix and all-sheet isolates the class-to-pixel path from the
classification itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dock-py" / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

__all__ = [
    "OffscreenRenderer",
    "ribbon_for_states",
    "flat_states",
    "degenerate",
    "SCRIPT_SECTIONS",
]


class OffscreenRenderer:
    """A moderngl context plus one framebuffer, for reading the workbench back.

    ``None`` from :meth:`open` is a normal outcome, not an error: a machine
    without a usable OpenGL 3.3 driver cannot answer a pixel question, and the
    caller has to record that as a skip with the reason rather than as a
    failure. Every method therefore takes the renderer as ``| None``.
    """

    def __init__(self, ctx, fbo, sphere_prog, bg_prog, bg_vao):
        self.ctx = ctx
        self.fbo = fbo
        self.sphere_prog = sphere_prog
        self.bg_prog = bg_prog
        self.bg_vao = bg_vao

    @classmethod
    def open(cls, width: int = 900, height: int = 700) -> "OffscreenRenderer | None":
        """Build a context, or return ``None`` with the reason on `LAST_PROBLEM`."""
        global LAST_PROBLEM
        try:
            import moderngl

            from opendocking.workbench.app import build_programs

            ctx = moderngl.create_context(standalone=True, require=330)
            fbo = ctx.simple_framebuffer((width, height), components=4)
            sphere, _lines, background = build_programs(ctx)
            # The backdrop program takes no attributes, so it needs a bare VAO
            # before it can be drawn. Same requirement as in `paintGL`.
            bg_vao = ctx.vertex_array(background, [])
        except Exception as exc:  # noqa: BLE001 - any failure means "no answer"
            LAST_PROBLEM = f"{type(exc).__name__}: {exc}"
            return None
        return cls(ctx, fbo, sphere, background, bg_vao)

    def render(self, meshes, camera, width: int = 900, height: int = 700):
        """Draw `meshes` through the workbench's own path and read the pixels.

        Returns an ``(h, w, 3)`` uint8 array in **RGB** order. That is what a
        moderngl framebuffer read gives and it is what the channel constants
        below name -- but it is *not* the order this suite's `img_array`
        produces for a Qt grab, which is BGR (`Format_RGB32` stores B, G, R, A).

        Both are correct and mixing them silently inverts every colour
        comparison, so the order is stated here and the callers use
        :data:`SS_RED` / :data:`SS_GREEN` / :data:`SS_BLUE` and the
        `*_leads_*` helpers rather than raw indices. Measured on the shipped
        receptor: the all-helix ribbon reads R 56.4 / G 28.3 / B 34.6, the
        all-sheet one R 61.8 / G 55.8 / B 32.2, and the all-coil one
        R 47.2 / G 50.8 / B 57.9 -- which is RGB and is what makes the three
        classes separable by channel at all.
        """
        from opendocking.workbench.app import draw_background, draw_mesh, set_frame_uniforms

        ctx = self.ctx
        fbo = self.fbo
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        fbo.clear(0.10, 0.11, 0.13, 1.0)

        aspect = width / max(height, 1)
        view = camera.view_matrix()
        mvp = camera.projection(aspect) @ view
        # The same fog the viewport uses, from the same rule: the range tracks
        # the camera distance so it neither does nothing nor blacks out.
        span = max(camera.distance * 0.55, 8.0)
        set_frame_uniforms(self.sphere_prog, view, span * 0.55, span * 2.1)
        draw_background(ctx, self.bg_prog, self.bg_vao, width, height)

        for mesh, opacity in meshes:
            draw_mesh(ctx, self.sphere_prog, mesh, mvp, opacity)

        raw = fbo.read(components=3, alignment=1)
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()

    def close(self) -> None:
        for obj in (self.bg_vao, self.bg_prog, self.sphere_prog, self.fbo, self.ctx):
            try:
                obj.release()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass


#: Why the last :meth:`OffscreenRenderer.open` returned ``None``.
LAST_PROBLEM: str | None = None

#: Channel indices in the array :meth:`OffscreenRenderer.render` returns: RGB.
#: Named because a bare `frame[:, :, 2]` is the single easiest thing in this
#: file to get backwards, and getting it backwards does not raise -- it
#: measures the complement of the quantity the check is about.
SS_RED, SS_GREEN, SS_BLUE = 0, 1, 2


def ribbon_mask(frame, background) -> np.ndarray:
    """Pixels the ribbon owns: those that differ from a render of nothing.

    Taken against a frame rendered with an empty mesh list rather than against
    `workbench_interaction_check.background_colour`, and the reason is that the
    backdrop is a **gradient** (`draw_background` fills the frame from
    `BACKGROUND_TOP` to `BACKGROUND_BOTTOM`). A modal-colour reference is
    therefore the colour at one end of that gradient, and thresholding against
    it marks most of the empty frame as object: measured, 413 044 of 630 000
    pixels instead of the 93 496 the ribbon actually covers.
    """
    if frame is None or background is None:
        return np.zeros(frame.shape[:2], bool) if frame is not None else None
    d = np.abs(frame.astype(np.int16) - background.astype(np.int16)).sum(axis=2)
    return d > 12


def red_excess(frame, mask, lead: int = 20) -> float:
    """Share of `mask`'s pixels where red leads green by `lead` levels or more.

    The helix is the one class whose red leads its green by a wide margin: the
    all-helix ribbon reads R 56.4 against G 28.3, the all-sheet one R 61.8
    against G 55.8, and the all-coil one R 47.2 against G 50.8. Red on its own
    does not separate them -- sheet is *more* red overall -- so the margin
    against green is the quantity that isolates the helix.
    """
    if frame is None or mask is None or not mask.any():
        return 0.0
    a = frame[:, :, SS_RED].astype(np.int16)
    b = frame[:, :, SS_GREEN].astype(np.int16)
    return float(((a - b)[mask] >= lead).mean())


def flat_states(n: int, state: str) -> list[str]:
    """`n` residues all given the same state.

    The degenerate case, built explicitly so a check can ask for it. An
    implementation that answers "coil" for everything satisfies every
    "the ribbon was drawn" check there is, and this is how that is caught.
    """
    return [state] * int(n)


def degenerate(states) -> bool:
    """True when every residue carries the same state.

    The failure mode this exists to name: a classifier that finds no hydrogen
    bonds returns all-coil, and an all-coil ribbon is a perfectly good picture.
    It renders, it is the right size, and it says nothing. So a suite that only
    asks "is there a ribbon" is green on a classifier that never worked, and the
    guard has to be a positive claim that the classes are *not* all the same.
    """
    uniq = set(states)
    return len(uniq) <= 1


def ribbon_for_states(structure, states):
    """The ribbon the viewport would draw for an explicit list of states.

    The same `geometry.ribbon` call `MoleculeView.backbone_ribbon` makes, with
    the guide points, the side vectors and the per-residue widths, thicknesses
    and colours left exactly as the product sets them. Only the states differ
    between two calls, which is what makes two renders comparable.

    Takes a `Structure` rather than a trace and an atom list because the side
    vectors need residue membership -- which CB belongs to which CA -- and only
    the parsed structure knows that.
    """
    from opendocking.workbench.geometry import ribbon as build_ribbon

    trace = structure.backbone()
    atoms = structure.atoms
    guide = np.asarray([atoms[ca].xyz for _, ca, _ in trace], dtype=np.float64)
    sides = np.asarray(_side_vectors(trace, atoms, structure))
    return build_ribbon(guide, sides, list(states))


def _side_vectors(trace, atoms, structure) -> list:
    """CA-to-CB per residue, N-CA-C bisector for glycine.

    Deliberately a re-implementation rather than a call into the molecule view:
    this needs to work for a `Structure` that no `MoleculeView` holds, so that a
    synthetic peptide can be rendered the same way a receptor is. The rule is
    the product's, copied rather than imported, and the two are checked against
    each other in the suite.
    """
    cb_of_ca = {}
    for res in structure.residues:
        cb = next((i for i in res.atoms if atoms[i].name == "CB"), None)
        if cb is None:
            continue
        ca = next((i for i in res.atoms if atoms[i].name == "CA"), None)
        if ca is not None:
            cb_of_ca[ca] = cb
    sides = []
    for n_idx, ca_idx, c_idx in trace:
        cb = cb_of_ca.get(ca_idx)
        if cb is not None:
            sides.append(np.asarray(atoms[cb].xyz) - np.asarray(atoms[ca_idx].xyz))
            continue
        a = np.asarray(atoms[n_idx].xyz) - np.asarray(atoms[ca_idx].xyz)
        b = np.asarray(atoms[c_idx].xyz) - np.asarray(atoms[ca_idx].xyz)
        sides.append(a + b)
    return sides


def save_png(path, frame) -> str:
    """Write an RGB array out as a PNG, without needing Qt for it.

    Pillow rather than a hand-rolled zlib writer: it is already a dependency of
    nothing else in this repository, so this is the one place that can fail on
    a missing import, and it is confined to producing pictures for a human to
    look at. The checks read the frame array, not the file.
    """
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8), mode="RGB").save(str(path))
    return str(path)


#: Reference secondary structure for crambin 1CRN, the receptor this repository
#: ships in `examples/1crn_prep.pdbqt`, as half-open residue-number ranges.
#:
#: This is the published annotation of that entry, not a fit to this code: 1CRN
#: is a small plant protein, and PDBsum/DSSP both list a two-stranded
#: antiparallel beta sheet at residues 1-4 and 32-35, an alpha helix at 7-19, a
#: second alpha helix at 23-30, and a short 3-10 helix at the C-terminus.
#: Stated here as a range per segment so a check can state what it expects and
#: fail with the number it got.
SCRIPT_SECTIONS = (
    "sheet 1-4 (two-stranded antiparallel, N-terminal strand)",
    "helix 7-19 (alpha helix 1)",
    "helix 23-30 (alpha helix 2)",
    "sheet 32-35 (two-stranded antiparallel, C-terminal strand)",
)

