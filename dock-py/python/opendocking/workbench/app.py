"""The workbench application: window, OpenGL viewport, and docking thread.

Separated from :mod:`opendocking.workbench` (geometry, colours, parsing) so that
importing the package does not require PyQt6 or moderngl to be installed.

# How the OpenGL context is shared

Qt owns the window and the OpenGL context (``QOpenGLWidget``); moderngl wraps
*that* context rather than creating a second one. Two live contexts in one
process on Windows cannot share textures, so a moderngl standalone context
would render into memory Qt could never see. Wrapping Qt's context is the only
arrangement that works portably, and it costs one call inside ``paintGL``.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import Camera, MoleculeView, _icosphere, _parse_pdbqt_atoms, _require_gui
from . import crashguard

_require_gui()
from PyQt6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets  # noqa: E402
from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # noqa: E402

# An uncaught exception inside a Qt slot ends this process with `0xC0000409`
# and prints nothing, so a bug in a signal handler is indistinguishable from a
# driver fault until somebody reads a log that does not exist. This module is
# the earliest point every GUI entry point shares -- `odgui`,
# `python -m opendocking.workbench.launcher` and `odcli workbench` all reach it
# through `workbench.launch`, and the `odgui --check` stage-2 child imports
# `app` too -- and it is a module-scope statement, not one call in one launcher,
# because a hook installed in one module is not a hook. It goes after
# `_require_gui()`: a machine with no PyQt6 has no slots to raise in, and the
# diagnostic that says so is better than a hook guarding nothing.
#
# The hook prints the traceback *and* ends the process with a non-zero code. The
# second half is the one that matters; a hook that only prints leaves the
# process alive and exiting 0, which turns a crash into a silent pass. See
# `crashguard` for why the code is handed back through the event loop only where
# this package owns it.
crashguard.install()

__all__ = [
    "run",
    "MainWindow",
    "Viewport",
    "DockingWorker",
    "SphereMesh",
    "build_programs",
    "draw_spheres",
    "draw_lines",
    "draw_mesh",
    "Representation",
    "REPRESENTATIONS",
    "REPRESENTATION_KEYS",
    "ELEMENT_VDW_CPK",
    "ELEMENT_CPK_RADIUS",
    "CPK_CARBON_RADIUS",
    "sphere_radii_for",
]

SPHERE_VERTEX_SHADER = """
#version 330 core
uniform mat4 mvp;
// The view matrix is needed separately, not folded into `mvp`, because depth
// fog is a function of how far a fragment is from the eye, and `mvp` has
// already had the perspective divide baked into it. Passing the modelview
// matrix is the difference between "far away" and "somewhere odd on screen".
uniform mat4 mv;
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_color;
out vec3 v_color;
out vec3 v_normal;
out float v_depth;
void main() {
    v_color = in_color;
    v_normal = in_normal;
    vec4 eye = mv * vec4(in_position, 1.0);
    // The camera looks down -z, so distance in front of it is -eye.z.
    v_depth = -eye.z;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

SPHERE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
in vec3 v_normal;
in float v_depth;
uniform float opacity;
uniform vec3 fog_color;
uniform vec2 fog_range;   // (near, far): below near is clear, above far is fog
out vec4 out_color;
void main() {
    vec3 n = normalize(v_normal);
    // Two lights, because one light makes every sphere look like the same
    // ball: a second one from the opposite side separates the front of an atom
    // from the back and gives the model some volume without a second pass.
    vec3 key_dir  = normalize(vec3(0.45, 0.55, 0.75));
    vec3 fill_dir = normalize(vec3(-0.55, -0.25, 0.35));
    float key  = max(dot(n, key_dir), 0.0);
    float fill = max(dot(n, fill_dir), 0.0);
    // A little rim light, which is what stops a dense receptor reading as one
    // flat silhouette.
    float rim = pow(1.0 - abs(n.z), 3.0) * 0.18;
    float shade = 0.26 + 0.62 * key + 0.22 * fill + rim;
    vec3 rgb = v_color * shade;

    // Aerial perspective. A 3000-atom receptor drawn at a useful zoom is a
    // solid mass of overlapping spheres, and depth is the only cue that says
    // which ones are in front. Fog restores it, and it is cheap.
    float f = clamp((v_depth - fog_range.x) / max(fog_range.y - fog_range.x, 1e-4),
                    0.0, 1.0);
    f = f * f;                       // ease in: near geometry stays crisp
    rgb = mix(rgb, fog_color, f);
    out_color = vec4(rgb, opacity);
}
"""

LINE_VERTEX_SHADER = """
#version 330 core
uniform mat4 mvp;
uniform mat4 mv;
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;
out vec3 v_color;
out float v_depth;
void main() {
    v_color = in_color;
    v_depth = -(mv * vec4(in_position, 1.0)).z;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
in float v_depth;
uniform float opacity;
uniform vec3 fog_color;
uniform vec2 fog_range;
out vec4 out_color;
void main() {
    // Lines get a weaker fog than the spheres do. A contact line is the
    // annotation on the picture; fogging it as hard as the protein it is drawn
    // over would make the far side of the interface unreadable for the sake of
    // a consistency that helps nobody.
    float f = clamp((v_depth - fog_range.x) / max(fog_range.y - fog_range.x, 1e-4),
                    0.0, 1.0);
    f = f * f * 0.55;
    out_color = vec4(mix(v_color, fog_color, f), opacity);
}
"""

#: A fullscreen gradient behind the scene, so the background is not one flat
#: colour meeting the molecule with a hard line.
BACKGROUND_VERTEX_SHADER = """
#version 330 core
out vec2 v_uv;
void main() {
    // One oversized triangle rather than a quad: no diagonal seam, one fewer
    // vertex, and there is nothing to interpolate across it anyway.
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

BACKGROUND_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
uniform vec3 top_color;
uniform vec3 bottom_color;
out vec4 out_color;
void main() {
    out_color = vec4(mix(bottom_color, top_color, v_uv.y), 1.0);
}
"""

#: Background gradient, fog colour and MSAA sample count, in one place because
#: the fog colour has to be the colour the geometry fades *into*; a fog tinted
#: differently from the background draws a visible seam across the model.
BACKGROUND_TOP = (0.055, 0.062, 0.082)
BACKGROUND_BOTTOM = (0.125, 0.140, 0.175)
FOG_COLOR = (0.095, 0.105, 0.130)
#: Multisampling. 4x is the point where edges visibly settle; 8x costs real
#: frame time for a difference nobody can see on a shaded sphere.
MSAA_SAMPLES = 4

def _mat4(m: np.ndarray) -> np.ndarray:
    """Flatten a 4x4 matrix the way a GLSL ``mat4`` uniform wants it.

    moderngl uploads uniforms with ``glUniformMatrix4fv(..., GL_FALSE, ...)``,
    which reads the sixteen floats in **column-major** order. A C-contiguous
    ``ravel()`` is row-major, i.e. the transpose, and the scene renders as a
    sheaf of stretched triangles rather than raising anything. ``order="F"``
    is the order the uniform is declared in.
    """
    return np.asarray(m, dtype=np.float32).flatten(order="F")


def _draw_colours_for(mol) -> np.ndarray:
    """The per-atom RGB a view is actually drawn in, as an ``(n, 3)`` array.

    One function, because "what colour is this view" has two answers in this
    file -- the element table, and the flat overrides a pose wears -- and they
    have to be asked in the same place. A check that asked
    `MoleculeView.atom_colors` while the renderer used its own copy would be
    asking about a colour nothing draws, and it would pass or fail for reasons
    that have nothing to do with the picture.

    **A pose is drawn in its own flat colour, always, and not only while poses
    are being compared.** The ghosts have always been flat (`POSE_GHOST_COLOR`,
    a cool slate) and the selected pose flat in its identity green while the
    compare overlay was up; this makes the pose flat unconditionally, because
    the measurement says the element table costs more than it carries here.

    What it costs, measured on prepared 1CRN with the crambin pose at the
    selection framing, interaction overlay **off**, over the pose's own 23 092
    pixels:

        chroma (max channel - min channel, 0-255)   median   % >= 24
        pose                                                8        10.7%
        receptor                                            7        25.3%

    The pose's median chroma sits at the **receptor's 60th percentile**. That is
    the whole finding: the pose is not "a bit greyer than the protein", it is
    *inside the protein's own chroma distribution*, so no threshold on hue,
    chroma or brightness separates the two populations. A 16-atom ligand at
    0.30 A radii is 2.5% of a 1251x989 frame made of 10-30 px blobs, and the
    pose's red oxygens and blue nitrogens are 10 px dots in a field of grey
    protein dots -- the element table was buying a distinction the picture
    cannot show at the framing that shows the pose at all.

    **What this gives up**, stated plainly: element identity on the selected
    pose's atoms, in every representation, whenever a pose is selected. "That
    atom is an oxygen" is no longer readable off the picture. It is still
    readable where it actually matters for this question -- the contact table
    names each contacting residue and its element, and the residue tooltip
    says the contact kind -- and a user who wants element colours on a pose has
    them for the ligand, which is a different view (`role == "ligand"`).

    The alternative that was rejected is a *highlight on top of* the element
    colours -- a rim pass, an emissive boost, a halo. The numbers above are why:
    a brightness change moves a pixel along the lit surface, and both
    populations are lit and fogged by the same shader, so it cannot move the
    pose out of the receptor's chroma distribution. Making a 2.5%-of-frame
    object findable needs a difference in *kind*, which is what a flat hue is.

    The `comparing` argument this function used to take is gone rather than left
    reading nothing: a pose view only exists when a pose is selected, so "the
    pose is flat" and "a pose is selected" are the same condition, and a
    parameter that no longer changes the answer is a second way to be wrong
    about it.
    """
    if mol.role in ("pose", "pose_ghost"):
        return np.tile(np.asarray(mol.color, np.float32), (len(mol.coords), 1))
    return mol.atom_colors()


def _pose_bonds(coords, elements) -> np.ndarray:
    """Bond pairs for a pose, from the coordinates that are about to be drawn.

    **This is a distance-derived approximation, and the view says so.** Pose
    views are built by `_view_from_text` rather than by `MoleculeView.from_text`
    because `_parse_pdbqt_atoms` hands back atoms in *serial* order, which is
    the order the pose table's RMSD and the docking engine's atom indices both
    speak; delegating to `from_text` would draw a pose whose bond indices
    pointed at the wrong atoms on any file whose records are not already in
    serial order. That is a real reason to build the view here, and it leaves the
    view with no bonds at all -- which is why ``stick``, a representation that
    draws bonds and nothing else, showed the receptor and no pose whatsoever.
    The user picked a pose and the picture had nothing of it in it.

    So the bonds are perceived here, from the same array the renderer will draw,
    with the house rule from `structure._bondable`: two atoms are bonded when
    they are closer than the sum of their covalent radii plus 0.45 A. That is
    the same rule `parse_structure` applies to a ligand that declares no
    `ROOT`/`BRANCH` tree, and on the shipped example it returns the identical
    16 pairs for all nine poses, so this is the module's own answer and not a
    second, laxer one. It is a rule about distances, not a reading of the file:
    a pose's `ROOT`/`BRANCH` records *are* declared bonds that this does not
    consult, and `bond_source` stays ``"distance"`` so the status bar prints
    "inferred from distances" rather than "read from the file". That distinction
    is the whole reason the field exists.

    `_bondable` is private to `structure`. Reaching for it is deliberate: a
    second copy of the radius table in this file would be free to drift from the
    one the receptor was drawn with, and the two views would then disagree about
    what counts as a bond while both claiming to be right.
    """
    from .structure import _bondable

    n = len(coords)
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _bondable(
            elements[i], elements[j], float(np.linalg.norm(coords[i] - coords[j]))
        )
    ]
    return np.asarray(pairs, np.int32).reshape(-1, 2)


def _walks_wanted(size) -> tuple[float, float]:
    """The ``(requirement, ceiling)`` of the density rule for a box of this size.

    Two numbers, because the rule has two answers and only one of them is what
    a spin box may hold.

    ``core.exhaustiveness_for_box`` answers "which rung of the ladder", and
    above the top rung that answer is a ceiling rather than a measurement.
    Reporting the rung as though it were the requirement is the one thing this
    workbench must not do: at 60 A on a side the density wants 216 walks and the
    ladder can only say 128, which is 59% of what the rule itself asks for, and
    a box that size is exactly where a missed pose is most likely. So the first
    number is the **unsaturated** requirement and the second is the engine's
    own answer, and the effort note compares them.

    The ceiling is the engine's function, called -- not a second reading of the
    ladder's last rung, and not a second reading of the reference box. An
    earlier version of this file imported the two calibration constants and
    re-derived the density itself, on the claim that "the numbers cannot drift
    apart" because the constants were shared. That claim was false in exactly
    one place and true everywhere else, which is the worst shape for a claim to
    be in: the shared constants did pin the *slope* and the *reference*, and
    nothing at all pinned the cap, the rung selection, or the refusal to answer
    a box the engine cannot build. Delegating the ceiling means that if the
    ladder grows, or the capping stops being "return the last rung", this note
    says so without being edited.

    The function raises `ValueError` on a two-element or non-finite size rather
    than answering, and that refusal now propagates -- which is the right
    default and is *not* caught here. The one caller reads three
    `QDoubleSpinBox`es, and Qt clamps a NaN or an infinity to the box's range
    rather than storing it, so a box this note is about cannot be one the engine
    refuses; adding a handler for a case the widget cannot produce would be an
    untested branch dressed as a safety net. A caller that can be handed an
    arbitrary size gets the engine's refusal, which is the answer it deserves.

    The requirement itself cannot come from the function, and it is worth being
    blunt about why rather than quietly re-deriving it: the function rounds up
    to a rung, so it cannot return a figure between two rungs, and the figure
    between them is the whole point. Asking it for 216 walks on a 60 A cube gets
    128 -- the ceiling, not the requirement. So the requirement is computed from
    the same two constants the function reads, and the function is called for
    everything it alone knows.
    """
    from ..core import EXHAUSTIVENESS_LADDER, REFERENCE_BOX_SIDE, exhaustiveness_for_box

    sides = tuple(float(v) for v in size)
    # Ask the engine first, and use its answer. It refuses a two-element size
    # and a non-finite edge outright, so a box it cannot build is refused here
    # too rather than formatted into a percentage of NaN further down, and the
    # cap the note names is whatever the ladder's top rung is today.
    ceiling = float(exhaustiveness_for_box(sides))
    volume = float(np.prod(np.asarray(sides, np.float64)))
    want = float(EXHAUSTIVENESS_LADDER[0]) * (volume / float(REFERENCE_BOX_SIDE) ** 3)
    return want, ceiling


#: Per-atom RMSD below which the engine treats two poses as the same mode, in
#: angstrom. It is `core.dock`'s own default, and `DockingWorker` now passes it
#: explicitly rather than relying on that default staying put, because the pose
#: table compares two reported poses against it: a pair closer than this means
#: the clustering did not separate them, and that is a finding to be shown
#: rather than a rounding detail to be smoothed away.
POSE_CLUSTER_CUTOFF = 1.0

#: The site cloud's opacity, normally and while poses are being compared. The
#: cloud is larger than the poses and sits in the same place, so at its usual
#: strength it outranks them: a solid pose at 1.0 next to a pink cloud at 0.34
#: cannot be told apart from a ghost inside it, and the picture then says
#: nothing about which pose is the one. Comparing poses is a closer question
#: than "where is the pocket", so the cloud steps aside for it.
#:
#: **Zero, and that is a measured number rather than a taste.** Fading the cloud
#: was tried first, at 0.10, on the reasoning that a dim cloud would still be
#: there. It does not work, and the way it fails is worth writing down: opacity
#: scales how *bright* the cloud is, not how much of the frame it *occupies*.
#: Measured on a real run (1crn + biotin, 26 A box, 9 poses, same camera, each
#: contributor toggled against an otherwise identical frame), the cloud changed
#: **76,398 pixels** whether its opacity was 0.34 or 0.10 -- the same number to
#: the pixel. The selected pose changed 8,343. So the cloud owned 9.2x the
#: picture the one thing the user turned the overlay on to look at, and dimming
#: it by 3x left the ratio untouched. Fading is the wrong lever; only leaving
#: changes the ratio, and 0.0 drops the cloud's 76,398 to 0.
POCKET_OPACITY_PLAIN = 0.34
POCKET_OPACITY_COMPARE = 0.0

#: What a ghosted pose is drawn in: one flat, cool slate, deliberately *not* the
#: receptor's `(0.62, 0.66, 0.72)` and deliberately not an element colour.
#:
#: The original ghost colour was within 0.07 per channel of the receptor's, and
#: that was invisible rather than subtle: a ghost is a pose like any other, so
#: `_draw_molecule` gave it per-element colours from the same table the protein
#: uses, and this constant only reached the few atoms whose element is not in
#: that table. Eight ghosts therefore rendered as eight dimmer grey carbons,
#: inside a space-filling grey protein, at 0.30 -- and "which one is solid" had
#: no answer in the picture.
#:
#: Two poses now differ in *kind*, not only in strength: the selected pose keeps
#: its element colours (red oxygens, blue nitrogens) and the ghosts are a single
#: flat tone, so a multicoloured molecule in a field of one-colour ones is a
#: difference the eye makes before it reads a single number. It is cooler and a
#: good deal darker than the receptor so that neither of them can pass for the
#: other's protein.
POSE_GHOST_COLOR = (0.38, 0.47, 0.62)


def _pose_distance(a_text: str, b_text: str) -> float | None:
    """Per-atom RMSD between two poses, in angstrom, or ``None`` if empty.

    One implementation, because two would be two numbers for one quantity. The
    engine's `DockingResult.rmsds` and this agree to 0.001 A on a measured
    nine-pose result, so the pose table can show either without the window ever
    holding two answers to "how far is this pose from that one".

    No alignment: the poses come out of the same search in the same frame, and
    superimposing them would hide exactly the displacement the number is meant to
    report.
    """
    a = _parse_pdbqt_atoms(a_text)[0]
    b = _parse_pdbqt_atoms(b_text)[0]
    n = min(len(a), len(b))
    if n == 0:
        return None
    return float(np.sqrt(((a[:n] - b[:n]) ** 2).sum(axis=1).mean()))


class _PoseItem(QtWidgets.QTableWidgetItem):
    """A table cell that sorts by its number, not by its text.

    Sorting has to be real sorting. `-4.9` and `-4.10` are the wrong order as
    strings, and a pose table that sorts "1.9" before "1.22" is worse than one
    that refuses to sort: the user reads the order as the ranking.

    The number lives in `UserRole`; a cell with no number sorts after every cell
    that has one, so a missing value never masquerades as a small one.
    """

    def __lt__(self, other) -> bool:  # noqa: D105 - Qt protocol
        mine = self.data(QtCore.Qt.ItemDataRole.UserRole)
        theirs = (
            other.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(other, QtWidgets.QTableWidgetItem)
            else None
        )
        if mine is not None and theirs is not None:
            return mine < theirs
        if mine is None and theirs is not None:
            return False
        if mine is not None and theirs is None:
            return True
        return super().__lt__(other)


class PoseTable(QtWidgets.QTableWidget):
    """The docking result as a table, with the two list methods kept.

    `count()` and the one-argument form of `item()` exist so
    `scripts/workbench_smoke.py` -- which is not this file's to edit -- can keep
    reading "how many poses, and does any of them read 0.00 kcal/mol" off the
    control that replaced the list. With no column given, `item(row)` answers
    with the whole row joined, so that check keeps testing the energy remark
    rather than silently testing a rank cell that never contains a number.
    """

    def count(self) -> int:
        """Number of pose rows, as the list this replaced reported it."""
        return self.rowCount()

    def setCurrentRow(self, row: int) -> None:  # noqa: N802 - Qt naming
        """Select a row, as the list this replaced did.

        `QTableWidget` has `setCurrentCell` and no `setCurrentRow`, and every
        existing caller -- including the check script, which this file does not
        own the tests for -- uses the list's name for it.
        """
        if 0 <= row < self.rowCount():
            self.setCurrentCell(row, 0)

    def cell(self, row: int, column: int) -> str:
        """One cell's text, or `""` for a cell that is not there.

        Sorting moves rows, so a caller that wants *pose* ``i``'s energy has to
        ask by pose, not by row. `pose_row_of` is that question.
        """
        item = super().item(row, column)
        return "" if item is None else item.text()

    def pose_row_of(self, index: int) -> int:
        """The visual row showing pose ``index``, or -1."""
        for row in range(self.rowCount()):
            item = super().item(row, 0)
            if item is not None and item.data(QtCore.Qt.ItemDataRole.UserRole) == index:
                return row
        return -1

    def item(self, row: int, column: int | None = None):  # noqa: D102 - see class docstring
        if column is not None:
            return super().item(row, column)
        cells = []
        for col in range(self.columnCount()):
            item = super().item(row, col)
            if item is not None:
                cells.append(item.text())
        return _PoseItem("  ".join(cells))


class SphereMesh:
    """The unit-sphere mesh every atom is expanded from.

    Kept separate from the Qt widget so the drawing code can be exercised
    against a standalone moderngl context — which is the only way to test the
    shaders on a machine with no display.
    """

    __slots__ = ("verts", "faces", "n_indices")

    def __init__(self, verts: np.ndarray, faces: np.ndarray) -> None:
        self.verts = np.ascontiguousarray(verts, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.uint32)
        self.n_indices = int(self.faces.size)


def build_programs(ctx):
    """Compile the shader programs on `ctx`."""
    sphere = ctx.program(
        vertex_shader=SPHERE_VERTEX_SHADER,
        fragment_shader=SPHERE_FRAGMENT_SHADER,
    )
    lines = ctx.program(
        vertex_shader=LINE_VERTEX_SHADER,
        fragment_shader=LINE_FRAGMENT_SHADER,
    )
    background = ctx.program(
        vertex_shader=BACKGROUND_VERTEX_SHADER,
        fragment_shader=BACKGROUND_FRAGMENT_SHADER,
    )
    return sphere, lines, background


def set_frame_uniforms(prog, view: np.ndarray, near: float, far: float) -> None:
    """Push the per-frame uniforms every program shares.

    The view matrix and the fog range change when the camera moves, and they
    change for the whole frame rather than per object, so they are set once
    here instead of being threaded through every draw call.
    """
    if "mv" in prog:
        prog["mv"].write(_mat4(view))
    if "fog_color" in prog:
        prog["fog_color"].value = tuple(float(v) for v in FOG_COLOR)
    if "fog_range" in prog:
        prog["fog_range"].value = (float(near), float(far))


def draw_background(ctx, prog, vao, width: int, height: int) -> None:
    """Fill the frame with the gradient before anything else is drawn.

    Depth writes are off for the pass: this is a backdrop, and letting it
    write depth would make every atom behind it fail the depth test and
    vanish. The VAO is passed in rather than looked up, because it is what the
    program is bound to -- `prog["vao"] = ...` raises `KeyError`, since a
    program's mapping holds uniforms and nothing else.
    """
    import moderngl

    ctx.disable(moderngl.DEPTH_TEST)
    prog["top_color"].value = tuple(float(v) for v in BACKGROUND_TOP)
    prog["bottom_color"].value = tuple(float(v) for v in BACKGROUND_BOTTOM)
    vao.render(moderngl.TRIANGLES, vertices=3)
    ctx.enable(moderngl.DEPTH_TEST)


def draw_spheres(ctx, prog, mesh: SphereMesh, mol, mvp, colors=None,
                 radii=None) -> None:
    """Draw every atom of `mol` as a shaded sphere.

    The mesh is expanded on the CPU rather than drawn with hardware
    instancing. An earlier version declared the per-atom attributes with
    ``divisor=1`` and then filled them with ``np.repeat``, which are two
    mutually exclusive choices; with a few hundred atoms the expansion is well
    under a millisecond and it is the version that is obviously correct.

    `colors` is the per-atom RGB to draw, and the caller passes the answer from
    :func:`_draw_colours_for` rather than this reaching for `atom_colors()`
    itself. It used to, and that made the argument pointless: a pose ghost is
    only flat-coloured if the *spheres* are flat-coloured, and "spheres" is the
    default representation, so an override applied anywhere but here would have
    coloured the bonds and left every atom its element colour. Two places
    asking the same question is how that happens.

    `radii` is the per-atom radius in Angstrom, and the caller passes
    :func:`sphere_radii_for` rather than this working it out, because the
    space-filling mode's radii are physical and every other mode's are a
    drawing scale. Left as ``None`` it is the old behaviour -- the molecule's
    own element-scaled radius -- so a caller that has not been updated still
    gets separated small spheres rather than a surprise surface.
    """
    import moderngl

    n_atoms = len(mol.coords)
    if n_atoms == 0:
        return
    v = len(mesh.verts)
    centers = mol.coords.astype(np.float32).reshape(n_atoms, 1, 3)
    # Per-atom radius, broadcast against the shared unit mesh. A single scalar
    # here draws every element at the same size, which makes a polar hydrogen
    # look like a carbon.
    per_atom_radius = (np.asarray(mol.atom_radii(), np.float32) if radii is None
                       else np.asarray(radii, np.float32))
    radii = per_atom_radius.reshape(n_atoms, 1, 1)
    positions = (centers + radii * mesh.verts[None, :, :]).reshape(-1, 3)
    # On a unit sphere the position is the normal, so the mesh supplies both.
    normals = np.tile(mesh.verts, (n_atoms, 1))
    per_atom = mol.atom_colors() if colors is None else colors
    colors = np.repeat(np.asarray(per_atom, np.float32), v, axis=0)
    # Re-index the shared mesh once per atom.
    shifts = (np.arange(n_atoms, dtype=np.uint32) * np.uint32(v))[:, None, None]
    indices = np.ascontiguousarray((mesh.faces[None, :, :] + shifts).reshape(-1))

    # One buffer per attribute, never an interleaved buffer shared by several
    # attributes: moderngl's default stride is the size of *one* attribute, so
    # sharing a buffer silently reads the first vec3 of every vertex three
    # times over. The data goes in correct and the geometry comes out wrong,
    # with nothing raised.
    pos_buf = ctx.buffer(np.ascontiguousarray(positions, np.float32).tobytes())
    nrm_buf = ctx.buffer(np.ascontiguousarray(normals, np.float32).tobytes())
    col_buf = ctx.buffer(np.ascontiguousarray(colors, np.float32).tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(
        prog,
        [
            (pos_buf, "3f", "in_position"),
            (nrm_buf, "3f", "in_normal"),
            (col_buf, "3f", "in_color"),
        ],
        index_buffer=ibo,
    )
    prog["mvp"] = _mat4(mvp)
    prog["opacity"] = float(mol.opacity)
    vao.render(moderngl.TRIANGLES)
    vao.release()
    pos_buf.release()
    nrm_buf.release()
    col_buf.release()
    ibo.release()


def draw_lines(ctx, prog, positions, colors, mvp, opacity) -> None:
    """Draw line segments from explicit vertex positions and colours."""
    import moderngl

    pos = np.ascontiguousarray(positions, dtype=np.float32)
    col = np.ascontiguousarray(colors, dtype=np.float32)
    pos_buf = ctx.buffer(pos.tobytes())
    col_buf = ctx.buffer(col.tobytes())
    vao = ctx.vertex_array(
        prog,
        [(pos_buf, "3f", "in_position"), (col_buf, "3f", "in_color")],
    )
    prog["mvp"] = _mat4(mvp)
    prog["opacity"] = float(opacity)
    vao.render(moderngl.LINES)
    vao.release()
    pos_buf.release()
    col_buf.release()


def draw_mesh(ctx, prog, mesh, mvp, opacity) -> None:
    """Draw a :class:`~opendocking.workbench.geometry.MeshData` with the sphere program.

    One upload path for every solid representation, so spheres, bond cylinders
    and the backbone ribbon all go through the same attribute layout and the
    same shader. Adding a representation therefore means generating geometry,
    not touching the renderer.

    The buffer-per-attribute rule from :func:`draw_spheres` applies here for the
    same reason: moderngl's default stride is the size of *one* attribute, so
    sharing a buffer between `in_position` and `in_normal` reads the first vec3
    three times over and renders wrong geometry with nothing raised.
    """
    import moderngl

    if mesh is None or mesh.empty:
        return
    pos_buf = ctx.buffer(mesh.positions.tobytes())
    nrm_buf = ctx.buffer(mesh.normals.tobytes())
    col_buf = ctx.buffer(mesh.colors.tobytes())
    ibo = ctx.buffer(mesh.indices.tobytes())
    vao = ctx.vertex_array(
        prog,
        [
            (pos_buf, "3f", "in_position"),
            (nrm_buf, "3f", "in_normal"),
            (col_buf, "3f", "in_color"),
        ],
        index_buffer=ibo,
    )
    prog["mvp"] = _mat4(mvp)
    prog["opacity"] = float(opacity)
    vao.render(moderngl.TRIANGLES)
    vao.release()
    pos_buf.release()
    nrm_buf.release()
    col_buf.release()
    ibo.release()


#: CPK van der Waals radii in Angstrom. One published table, used for every
#: element, rather than a hand-picked number per element: a per-element number
#: chosen by eye is a number nobody can check, and hydrogen is where that
#: always shows up first.
ELEMENT_VDW_CPK: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Mg": 1.73, "Zn": 1.39, "Ca": 2.31, "Fe": 2.04, "Mn": 2.05,
    # A dummy atom carries no element of its own; it stands in for the atom it
    # replaced, and carbon is both the commonest of those and the anchor below.
    "Du": 1.70,
}

#: The one number the space-filling mode is anchored on, in Angstrom.
CPK_CARBON_RADIUS = 0.77

#: The same table scaled so that carbon is exactly ``CPK_CARBON_RADIUS``. So
#: ``ELEMENT_CPK_RADIUS[e] / CPK_CARBON_RADIUS`` is the published CPK ratio for
#: `e`, hydrogen included: 1.20/1.70 of carbon, like any other element.
#:
#: Measured on prepared 1CRN (382 atoms, mean nearest neighbour 1.288 A): at
#: these radii **91.1% of atoms have a neighbour inside 2r**, against **0.0%**
#: at the 0.30 A the default uses. That gap is the whole difference between a
#: surface and a cloud of separated dots, and it is why the default keeps its
#: small radius and its honest name while the solid view is a separate mode.
ELEMENT_CPK_RADIUS: dict[str, float] = {
    e: CPK_CARBON_RADIUS * v / ELEMENT_VDW_CPK["C"]
    for e, v in ELEMENT_VDW_CPK.items()
}


def sphere_radii_for(mol, key: str) -> np.ndarray:
    """Per-atom sphere radius in Angstrom, for representation `key`.

    One place answers this, because the honest answer differs per
    representation and two places answering it is exactly how a mode ends up
    disagreeing with its own name. `spheres` and `ball_and_stick` scale
    `MoleculeView.radius`, which is 0.30 A for a receptor and is a *drawing*
    scale rather than a physical one; `space_filling` is the one mode whose
    radii are physical, so it ignores that scale and reads the CPK table.

    `molecule` with no elements (a bare coordinate list) falls back to the
    carbon anchor, which is the honest answer for "an element we do not know"
    and the same fallback the element table itself uses.
    """
    n = len(mol.coords)
    if key == "space_filling":
        elements = getattr(mol, "elements", None)
        if not elements or len(elements) != n:
            return np.full(n, CPK_CARBON_RADIUS, np.float32)
        return np.asarray(
            [ELEMENT_CPK_RADIUS.get(e, CPK_CARBON_RADIUS) for e in elements],
            dtype=np.float32,
        )
    radii = np.asarray(mol.atom_radii(), np.float32)
    if key == "ball_and_stick":
        # The 0.42 is the same factor the draw path used inline, so the balls
        # are not resized by this refactor.
        return radii * 0.42
    return radii


class Representation(NamedTuple):
    """One entry in the display selector.

    `claims` is the geometric property that the user-facing `label` asserts,
    and it is a field rather than a comment because it is checked:
    `scripts/representation_names_check.py` measures the geometry each entry
    really draws and fails if it does not have the property its name claims. A
    name whose property nobody measures is a name nobody can defend, and this
    project has already shipped one of those -- a protein at 0.30 A labelled
    "Space-filling", which is the defect this field exists to make impossible.

    `label` is what the user reads; `key` is what the rest of the program and
    the check suite pass around, so renaming a label does not have to rename
    the thing being tested.
    """

    key: str
    label: str
    needs_backbone: bool
    claims: str


#: The representations the viewport offers, in the order they are listed.
#: `needs_backbone` marks the ones that only mean something for a protein; a
#: molecule without a backbone falls back and says so rather than drawing
#: nothing.
#:
#: Every label here was checked against the geometry it names, and three of them
#: did not survive it. "Space-filling" was drawing separated spheres (0.0% of
#: atoms in contact), and "Cartoon" was drawing the *same ribbon* as the mode
#: above it plus side-chain sticks -- no extra width, no extra colour, no
#: helix-and-arrow encoding, which is what a cartoon is. `ribbon` and
#: `cartoon` differ by exactly one thing and the name now says which.
#:
#: `cartoon` is now built by `MoleculeView.cartoon`, which twists the ribbon
#: through a helix and flares a strand into a C-terminal arrowhead. The label is
#: "Cartoon (twisted ribbon + arrows)" because that is the geometry, and
#: `scripts/representation_names_check.py` measures whether it is true.
REPRESENTATIONS: tuple[Representation, ...] = (
    Representation(
        "spheres", "Small spheres (separated)", False, "separated_spheres"
    ),
    Representation(
        "space_filling", "Space-filling (CPK radii)", False, "solid_surface"
    ),
    Representation(
        "ball_and_stick", "Ball and stick", False, "balls_and_sticks"
    ),
    Representation(
        "stick", "Skeletal (sticks only)", False, "sticks_only"
    ),
    Representation(
        "ribbon", "Ribbon (backbone)", True, "backbone_ribbon"
    ),
    Representation(
        "cartoon", "Cartoon (twisted ribbon + arrows)", True,
        "twisted_cartoon",
    ),
)
REPRESENTATION_KEYS = tuple(r.key for r in REPRESENTATIONS)


class Viewport(QOpenGLWidget):
    """The OpenGL rendering surface.

    Renders instanced icospheres (one instance per atom) plus bond lines and a
    wireframe search box. Instancing matters: a 3000-atom receptor becomes one
    draw call instead of 3000.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        # Multisampling is a property of the surface, so it has to be asked for
        # before the context exists. Qt answers with the count it actually
        # granted, which is not always the one that was requested, so the real
        # value is kept for the status bar rather than the wish.
        fmt = QtGui.QSurfaceFormat()
        fmt.setSamples(MSAA_SAMPLES)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.samples_granted = fmt.samples()
        self.camera = Camera()
        self.molecules: list[MoleculeView] = []
        self.box_center = np.zeros(3, np.float32)
        self.box_size = np.asarray([22.0, 22.0, 22.0], np.float32)
        self.show_box = True
        #: Grid points of the selected pocket, drawn as a cloud so the site's
        #: claim can be seen rather than only read.
        self.pocket_points = np.zeros((0, 3), np.float32)
        self.show_pocket = True
        self.pocket_point_radius = 0.42
        #: Low on purpose. The cloud sits inside protein, so an opaque one hides
        #: the atoms whose proximity is the reason the site was reported. It is
        #: taken out of the frame entirely while poses are being compared -- see
        #: `POCKET_OPACITY_COMPARE`.
        self.pocket_opacity = POCKET_OPACITY_PLAIN
        #: True while "all poses ghosted together" is on. Kept as the
        #: *rendering state* the picture and the status label share -- it is
        #: what `_on_all_poses_visibility` toggles and what the poses label
        #: words out -- but it no longer decides any colour: a pose view wears
        #: its own flat colour whether or not this is set, because the element
        #: table cannot separate a pose from the protein it sits in. See
        #: `_draw_colours_for`, which carries the measurement.
        self.pose_compare = False
        #: Pose/receptor interactions, as returned by `contacts.find_contacts`.
        #: Empty means "not computed", which the status bar says out loud
        #: rather than leaving an empty list to be read as "no interactions".
        self.contacts: list = []
        self.show_contacts = True
        #: True once `contacts` has been computed for the current pair, so the
        #: panel can distinguish "none found" from "never looked".
        self.contacts_valid = False
        #: One of ``REPRESENTATION_KEYS``. Set through :meth:`set_representation`
        #: so the status bar can be told about it.
        self.representation = "spheres"
        self._ctx = None
        self._sphere_prog = None
        self._line_prog = None
        self._bg_prog = None
        self._bg_vao = None
        self._n_mesh_indices = 0
        self._dragging: QtCore.Qt.MouseButton | None = None
        self._last_mouse = (0.0, 0.0)
        self.on_pose_changed = None
        #: The camera move in flight, and its step counter. `None` when the
        #: camera is where it was told to be, which is the only state any
        #: other code has to care about.
        self._move = None
        self._move_step = 0
        self._move_timer = None

    # -- Qt / GL lifecycle -------------------------------------------------

    def initializeGL(self) -> None:  # pragma: no cover - needs a display
        """Wrap Qt's context in moderngl and build the GPU resources."""
        try:
            import moderngl

            self.makeCurrent()
            self._ctx = moderngl.create_context(standalone=False, require=330)
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        except Exception as exc:
            # A machine without OpenGL 3.3 is unusual but not fatal: the
            # controls stay usable and the status bar explains the failure.
            self._ctx = None
            self.setToolTip(f"OpenGL unavailable: {exc}")
            return

        self._sphere_prog, self._line_prog, self._bg_prog = build_programs(self._ctx)
        # The backdrop has no attributes, so it needs a bare VAO before it can
        # be drawn: moderngl refuses to render a program with no enabled
        # vertex array, and the symptom is a black screen rather than an error.
        self._bg_vao = self._ctx.vertex_array(self._bg_prog, [])
        verts, faces = _icosphere(1)
        self._mesh = SphereMesh(verts, faces)
        self._n_mesh_indices = self._mesh.n_indices

    def paintGL(self) -> None:  # pragma: no cover - needs a display
        if self._ctx is None:
            return

        self.makeCurrent()
        ctx = self._ctx
        # `QOpenGLWidget` composites from a framebuffer object, **not** the
        # default framebuffer. Drawing into `ctx.screen` therefore draws
        # somewhere that is never shown and the viewport comes out blank.
        # `detect_framebuffer` wraps whatever FBO is bound right now, which is
        # the one Qt wants painted; the widget's own size (times the device
        # pixel ratio) is `target.size`, not `self.width()`.
        target = ctx.detect_framebuffer()
        target.use()
        target.viewport = (0, 0, target.size[0], target.size[1])
        target.clear(0.10, 0.11, 0.13, 1.0)

        aspect = target.size[0] / max(target.size[1], 1)
        view = self.camera.view_matrix()
        mvp = self.camera.projection(aspect) @ view

        # Fog is scaled to how far away the scene actually is. A fixed range
        # would either do nothing on a 30 000-atom receptor or black out a
        # ligand viewed up close, and the camera distance is the one number
        # that tracks both.
        span = max(self.camera.distance * 0.55, 8.0)
        near, far = span * 0.55, span * 2.1
        set_frame_uniforms(self._sphere_prog, view, near, far)
        set_frame_uniforms(self._line_prog, view, near, far)
        draw_background(ctx, self._bg_prog, self._bg_vao, target.size[0], target.size[1])

        for mol in self.molecules:
            if not mol.visible or len(mol.coords) == 0:
                continue
            self._draw_molecule(mol, mvp)
        if self.show_pocket and len(self.pocket_points):
            self._draw_pocket(mvp)
        if self.show_box:
            self._draw_box(mvp)
        if self.show_contacts and self.contacts:
            self._draw_contacts(mvp)

        # Leave the default framebuffer bound, as Qt expects.
        ctx.screen.use()

    def _draw_molecule(self, mol: MoleculeView, mvp) -> None:  # pragma: no cover
        from .geometry import bonds as bond_geometry
        from .geometry import spheres as sphere_geometry

        mode = self.representation
        if mode in ("ribbon", "cartoon") and not mol.has_backbone:
            # A ligand has no backbone, so a ribbon of it would be a line with
            # nothing on it. Fall back and let the status bar say so.
            mode = "ball_and_stick"

        pairs = mol.bond_pairs()
        colors = _draw_colours_for(mol)

        if mode in ("spheres", "space_filling"):
            draw_spheres(self._ctx, self._sphere_prog, self._mesh, mol, mvp,
                         colors, radii=sphere_radii_for(mol, mode))
            if mode == "space_filling":
                # No bond lines here, and that is what makes the name true: at
                # CPK radii every bond lies inside the surface of its own
                # atoms, so drawing them would cost a draw call per molecule
                # and put no pixel on screen. The small-sphere mode below is
                # the one that needs them, because its atoms are far enough
                # apart that the bonds are the only thing joining them up.
                return
            segs = mol.bond_segments()
            if len(segs):
                draw_lines(
                    self._ctx,
                    self._line_prog,
                    segs.reshape(-1, 3),
                    np.tile(np.asarray(mol.color, np.float32), (len(segs) * 2, 1)),
                    mvp,
                    float(mol.opacity),
                )
            return

        if mode == "ball_and_stick":
            draw_mesh(
                self._ctx, self._sphere_prog,
                sphere_geometry(mol.coords, sphere_radii_for(mol, mode), colors),
                mvp, mol.opacity,
            )
            draw_mesh(
                self._ctx, self._sphere_prog,
                bond_geometry(mol.coords, pairs, float(mol.radius) * 0.20, colors),
                mvp, mol.opacity,
            )
            return

        if mode == "stick":
            # The skeletal view: bonds only, which is the usual way a small
            # molecule is drawn when the question is its shape rather than its
            # surface.
            draw_mesh(
                self._ctx, self._sphere_prog,
                bond_geometry(mol.coords, pairs, float(mol.radius) * 0.16, colors),
                mvp, mol.opacity,
            )
            return

        # ribbon / cartoon
        # Two different meshes over the same trace. `ribbon` sweeps a quad
        # strip whose cross-section does not turn about the path; `cartoon`
        # twists it through a helix and flares a strand into an arrowhead. The
        # call is deliberately not shared: a cartoon drawn by `backbone_ribbon`
        # is the defect this dispatch was split to remove, and a shared call
        # would make it a one-line accident again.
        draw_mesh(
            self._ctx,
            self._sphere_prog,
            mol.cartoon() if mode == "cartoon" else mol.backbone_ribbon(),
            mvp,
            mol.opacity,
        )
        if mode == "cartoon":
            # Side chains as thin sticks so the detail is still reachable
            # without going back to a space-filling view of 30 000 atoms.
            draw_mesh(
                self._ctx, self._sphere_prog,
                bond_geometry(mol.coords, pairs, float(mol.radius) * 0.13, colors),
                mvp, mol.opacity,
            )

    def _draw_lines(self, positions, colors, mvp, opacity) -> None:  # pragma: no cover
        draw_lines(self._ctx, self._line_prog, positions, colors, mvp, opacity)

    def _draw_contacts(self, mvp) -> None:  # pragma: no cover - GUI
        """Draw the pose/receptor interaction lines.

        Solved against the molecules currently in the scene by role rather than
        against indices captured when the contacts were computed, so a structure
        that has since been reloaded cannot leave the lines pointing at
        whatever moved into that slot. A contact whose atom no longer exists is
        dropped rather than clamped to a neighbour.
        """
        from . import COLOR_CONTACT
        from .geometry import dashed_segments

        pose = next((m for m in self.molecules if m.role == "pose"), None)
        receptor = next((m for m in self.molecules if m.role == "receptor"), None)
        if pose is None or receptor is None:
            return

        starts, ends, colors = [], [], []
        for c in self.contacts:
            if not (0 <= c.self_index < len(pose.coords)):
                continue
            if not (0 <= c.partner_index < len(receptor.coords)):
                continue
            starts.append(pose.coords[c.self_index])
            ends.append(receptor.coords[c.partner_index])
            colors.append(COLOR_CONTACT.get(c.kind, COLOR_CONTACT["close"]))
        if not starts:
            return

        segs, group = dashed_segments(
            np.asarray(starts, np.float32), np.asarray(ends, np.float32)
        )
        # `group` says which contact each dash came from, so every dash of one
        # hydrogen bond is that bond's colour rather than a gradient along it.
        rgb = np.asarray(colors, np.float32)[group]
        self._draw_lines(segs, rgb, mvp, 0.95)

    def _draw_box(self, mvp) -> None:  # pragma: no cover
        from . import COLOR_BOX

        c = np.asarray(self.box_center, np.float32)
        h = np.asarray(self.box_size, np.float32) * 0.5
        corners = np.asarray(
            [
                [c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ],
            np.float32,
        )
        edges = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        ]
        pts = []
        for a, b in edges:
            pts.append(corners[a])
            pts.append(corners[b])
        data = np.asarray(pts, np.float32)
        self._draw_lines(
            data, np.tile(np.asarray(COLOR_BOX, np.float32), (len(data), 1)), mvp, 0.9
        )

    def _draw_pocket(self, mvp) -> None:  # pragma: no cover
        """The selected site's own grid points, as a translucent cloud.

        Drawn after the box and before the contacts, and deliberately small and
        translucent: it is a claim about a place, and it has to be legible
        *through* the protein rather than painted over it. A solid blob would
        hide the very atoms whose proximity is the reason the site exists.

        `sphere_geometry` is the function, not the module -- the same alias
        `_draw_molecule` uses. Writing `sphere_geometry.spheres(...)` raises an
        AttributeError *inside* `paintGL`, and an exception in a paint event
        does not propagate out of Qt: PyQt6 aborts the process with
        0xC0000409 and no traceback. That is what a first attempt at this did,
        and it looks exactly like a driver crash.

        Depth testing is off for this one draw. A site is by definition inside
        protein, so a depth-tested cloud is hidden by the very atoms that make
        it a site: in space-filling, which is the default, the screenshot
        showed about half the volume. Drawing it through the protein turns the
        cloud into an x-ray overlay, and "is this volume surrounded by atoms?"
        is the question the picture exists to answer -- which the occluded half
        could not. Depth is restored immediately after, so the box and the
        contacts still draw in the right order relative to everything else.
        """
        import moderngl

        from .geometry import spheres as sphere_geometry

        pts = np.asarray(self.pocket_points, np.float32).reshape(-1, 3)
        if len(pts) == 0:
            return
        if float(self.pocket_opacity) <= 0.0:
            # A zero opacity still uploads every point and runs a draw call to
            # produce nothing. The comparison mode sets exactly this value, so
            # the early-out is the difference between "the cloud steps aside"
            # and "the cloud is still there, only invisible" -- and an
            # assertion that reads the framebuffer cannot tell those apart from
            # the pixels alone, because both leave the cloud's footprint at
            # zero. Skipping the work is also simply correct.
            return
        from . import COLOR_POCKET

        data = sphere_geometry(
            pts,
            float(self.pocket_point_radius),
            np.tile(np.asarray(COLOR_POCKET, np.float32), (len(pts), 1)),
        )
        self._ctx.disable(moderngl.DEPTH_TEST)
        try:
            draw_mesh(self._ctx, self._sphere_prog, data, mvp, self.pocket_opacity)
        finally:
            self._ctx.enable(moderngl.DEPTH_TEST)

    # -- Interaction -------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI
        self._dragging = event.button()
        self._last_mouse = (event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:  # pragma: no cover - GUI
        if self._dragging is None:
            return
        x, y = event.position().x(), event.position().y()
        dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
        self._last_mouse = (x, y)
        if self._dragging == QtCore.Qt.MouseButton.LeftButton:
            self.camera.yaw += dx * 0.01
            self.camera.pitch = max(-1.5, min(1.5, self.camera.pitch + dy * 0.01))
        elif self._dragging == QtCore.Qt.MouseButton.MiddleButton:
            self.camera.distance = max(2.0, self.camera.distance * (1.0 + dy * 0.005))
        elif self._dragging == QtCore.Qt.MouseButton.RightButton:
            self._pan(dx, dy)
        self.update()

    def _pan(self, dx: float, dy: float) -> None:
        """Slide the look-at point across the screen plane.

        The shift is scaled by the distance and by the vertical field of view,
        so a pixel of drag moves the same *angular* amount however far away the
        molecule is. Without this there is no way to bring an off-centre
        structure into frame except by orbiting until it happens to line up.
        """
        height = max(self.height(), 1)
        width = max(self.width(), 1)
        scale = 2.0 * self.camera.distance * math.tan(math.radians(self.camera.fov) * 0.5)
        right, up, _ = self.camera.basis()
        self.camera.center = (
            self.camera.center
            - right * (dx * scale / width)
            - up * (dy * scale / height)
        ).astype(np.float32)

    def mouseReleaseEvent(self, event) -> None:  # pragma: no cover - GUI
        self._dragging = None

    def wheelEvent(self, event) -> None:  # pragma: no cover - GUI
        self.camera.distance = max(
            2.0, self.camera.distance * (0.9 ** (event.angleDelta().y() / 120.0))
        )
        self.update()

    def frame_all(self) -> None:  # pragma: no cover - GUI
        """Point the camera at everything currently visible."""
        self._cancel_move()
        pts = [m.coords for m in self.molecules if m.visible and len(m.coords)]
        if not pts:
            return
        allpts = np.vstack(pts)
        self.camera.center = allpts.mean(axis=0).astype(np.float32)
        radius = float(np.linalg.norm(allpts - self.camera.center, axis=1).max())
        self.camera.distance = max(5.0, radius * 2.6)
        self.update()

    def focus_point(self, point, radius: float) -> None:  # pragma: no cover - GUI
        """Frame a point of known extent.

        `focus_residue` cannot do this job: it only moves when the pose and
        receptor are both loaded and the residue is among the contacts, and a
        pocket has to be inspectable before any ligand is docked. The
        multiplier is generous on purpose -- the thing worth seeing is the box
        with protein around it, so a tight framing that fills the window with
        the box's own face is not informative.
        """
        self._cancel_move()
        self.camera.center = np.asarray(point, np.float32)
        self.camera.distance = max(12.0, float(radius) * 1.6)
        self.update()

    # -- the camera transition --------------------------------------------

    #: How long a camera move takes, in milliseconds, and how many steps it is
    #: taken in. Both are here rather than inline because the plan
    #: (`framing_selection.CameraMove`) is what a check samples, and a plan whose
    #: timing is buried in a `QTimer` call is a plan nobody can sample.
    MOVE_MS = 280
    MOVE_STEPS = 14

    def _begin_move(self, plan) -> None:  # pragma: no cover - GUI
        """Run `plan` to completion, a step at a time.

        A camera that snaps from 43 A to 16 A loses the user's sense of where
        they were: the frame they were reading is replaced by one they have
        never seen, with nothing connecting the two. Moving is the difference
        between a viewer and a slideshow.

        The last step is :meth:`CameraMove.settle` rather than another
        interpolation, so the camera ends *exactly* on the target. A tween that
        approaches its target asymptotically ends a thousandth of an angstrom
        short forever, and a check written against the target then has to
        tolerate a floating-point remainder to be reliable on a machine it was
        not written on.
        """
        self._move = plan
        self._move_step = 0
        if self._move_timer is None:
            self._move_timer = QtCore.QTimer(self)
            self._move_timer.timeout.connect(self._advance_move)
        self._move_timer.start(max(1, self.MOVE_MS // self.MOVE_STEPS))

    def _advance_move(self) -> None:  # pragma: no cover - GUI
        plan = self._move
        if plan is None:
            self._move_timer.stop()
            return
        self._move_step += 1
        done = self._move_step >= plan.steps
        centre, distance = plan.settle() if done else plan.at(
            self._move_step / float(plan.steps)
        )
        self.camera.center = centre
        self.camera.distance = distance
        self.update()
        if done:
            self._move_timer.stop()
            self._move = None

    def _settle_move_now(self) -> None:  # pragma: no cover - GUI
        """Jump to the end of any running move. Used when a second one starts.

        Without this a second selection clicked during the first move's 280 ms
        starts from wherever the first had got to, and the two plans interleave
        into a camera that never passes through the target of either.
        """
        if self._move is not None:
            centre, distance = self._move.settle()
            self.camera.center = centre
            self.camera.distance = distance
        if self._move_timer is not None:
            self._move_timer.stop()
        self._move = None
        self.update()

    def _cancel_move(self) -> None:  # pragma: no cover - GUI
        """Drop any running move, leaving the camera where it is.

        The counterpart to `_settle_move_now`, for the methods that place the
        camera absolutely: `frame_all`, `focus_point` and `focus_residue` are
        all "the camera is *here* now" and none of them animates. Without this
        they were stomped: a tween still in flight overwrites whatever they set
        on its next tick, so a residue clicked during the 280 ms of a selection
        move ended up centred 3.05 A from its own contacts -- measured, and the
        measurement is what turned a "the camera moved" check into a failure
        that had nothing to do with the thing it was written for.
        """
        if self._move is None:
            return
        if self._move_timer is not None:
            self._move_timer.stop()
        self._move = None
        self._move_step = 0

    @property
    def moving(self) -> bool:  # pragma: no cover - GUI
        return self._move is not None

    def focus_selection(self, pose_coords, partner_coords, **kwargs):  # pragma: no cover - GUI
        """Point the camera at a pose and the atoms it touches.

        The *what* of this -- which points are in the selection, how far back
        the camera stops, what a degenerate framing is -- is
        `framing_selection`, which has no Qt in it and can be asked all of those
        questions without a window. This method supplies the two things only the
        viewport knows: the camera's own right/up basis, and the aspect ratio of
        the surface being drawn into.

        It returns the :class:`~opendocking.workbench.framing_selection.CameraMove`
        it ran, because a caller that wants to know where the camera ended -- a
        check, or a status line -- should not have to go and measure it again.
        """
        from . import framing_selection as fs

        right, up, forward = self.camera.basis()
        aspect = max(self.width(), 1) / max(self.height(), 1)
        plan = self.plan_selection(
            pose_coords, partner_coords, right, up, forward, aspect=aspect, **kwargs
        )
        self._settle_move_now()
        self._begin_move(plan)
        return plan

    def plan_selection(self, pose_coords, partner_coords, right, up, forward, *, aspect=1.0, **kwargs):
        """The move a selection would make, from where the camera is now."""
        from . import framing_selection as fs

        target = fs.selection_target(
            pose_coords, partner_coords, right, up, forward,
            fov=self.camera.fov, aspect=aspect, **kwargs
        )
        return fs.CameraMove(
            start_center=np.asarray(self.camera.center, np.float32),
            start_distance=float(self.camera.distance),
            target=target,
            steps=self.MOVE_STEPS,
        )

    def focus_residue(self, residue: str) -> bool:  # pragma: no cover - GUI
        """Move the camera onto a residue's contacts. True if it moved.

        Zooms to the span of the atoms actually making contact rather than to
        the whole residue, which for a buried side chain is the difference
        between the interaction filling the window and being a few pixels wide
        in it. A residue with no contacts found leaves the camera alone and says
        so, rather than jumping somewhere arbitrary.
        """
        self._cancel_move()
        pose = next((m for m in self.molecules if m.role == "pose"), None)
        receptor = next((m for m in self.molecules if m.role == "receptor"), None)
        if pose is None or receptor is None:
            return False
        pts: list[np.ndarray] = []
        for c in self.contacts:
            if c.partner_residue != residue:
                continue
            if 0 <= c.self_index < len(pose.coords):
                pts.append(pose.coords[c.self_index])
            if 0 <= c.partner_index < len(receptor.coords):
                pts.append(receptor.coords[c.partner_index])
        if not pts:
            return False
        arr = np.asarray(pts, np.float32)
        self.camera.center = arr.mean(axis=0)
        radius = float(np.linalg.norm(arr - self.camera.center, axis=1).max())
        self.camera.distance = max(6.0, radius * 5.0)
        self.update()
        return True


class PocketWorker(QtCore.QObject):
    """Runs a pocket search off the GUI thread.

    The search is pure numpy, but pure numpy is still work: measured on this
    machine it takes 0.07 s for crambin's 382 atoms, 2.0 s for streptavidin and
    **12.0 s for haemoglobin's 4779**. Run inline, that last one froze the
    window for twelve seconds on load, with the status text set and the event
    loop unable to run and paint it -- so the app looked hung rather than busy.

    That is the same mistake `DockingWorker` was written to avoid, and the
    lesson had already been paid for once in this file. It is a heuristic
    search over a grid that grows with the cube of the protein's extent, so
    there is no size below which the freeze stops mattering.
    """

    finished = QtCore.pyqtSignal(object, float)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, coords, elements, residues, probe=None):
        super().__init__()
        self.coords = coords
        self.elements = elements
        self.residues = residues
        self.probe = probe

    @QtCore.pyqtSlot()
    def run(self) -> None:  # pragma: no cover - exercised interactively
        try:
            from . import pockets as P

            started = time.perf_counter()
            kwargs = {"probe": self.probe} if self.probe else {}
            found = P.find_pockets(
                self.coords, self.elements, residues=self.residues, **kwargs
            )
            self.finished.emit(found, (time.perf_counter() - started) * 1000.0)
        except Exception as exc:
            self.failed.emit(str(exc))


class DockingWorker(QtCore.QObject):
    """Runs a docking job off the GUI thread.

    Qt delivers the result through `finished`, so the window never blocks while
    a search is running — the single most important property of a usable
    docking GUI.
    """

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progressed = QtCore.pyqtSignal(str)

    def __init__(self, receptor, ligand_path, box_, exhaustiveness, seed, scoring="vina"):
        super().__init__()
        self.receptor = receptor
        self.ligand_path = ligand_path
        self.box_ = box_
        self.exhaustiveness = exhaustiveness
        self.seed = seed
        self.scoring = scoring

    @QtCore.pyqtSlot()
    def run(self) -> None:  # pragma: no cover - exercised interactively
        try:
            from ..core import dock, load_ligand

            # Map precalculation belongs here, not on the GUI thread. It used
            # to run inline in `start_docking`, which froze the window: the
            # status text was set and the event loop could not run to repaint
            # it, so the user saw nothing happen at all.
            self.progressed.emit("precalculating maps…")
            maps = self.receptor.precalculate(self.box_, self.scoring)
            self.progressed.emit(
                f"searching ({self.exhaustiveness} walks) — the UI stays responsive"
            )
            result = dock(
                load_ligand(self.ligand_path),
                maps,
                exhaustiveness=self.exhaustiveness,
                num_modes=9,
                # Passed rather than left to `core.dock`'s default, because the
                # pose table compares reported poses against this number and
                # "whatever the default happens to be" is not a threshold anyone
                # can check a claim against.
                rmsd_cutoff=POSE_CLUSTER_CUTOFF,
                seed=self.seed if self.seed else None,
                scoring=self.scoring,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class TermsWorker(QtCore.QObject):
    """Tabulates the term maps and decomposes one pose, off the GUI thread.

    The same pattern `DockingWorker` uses, for the same reason. Measured on
    1crn at spacing 0.375, the two precalculations the panel needs cost 0.82 s
    for a 22 A box and 4.09 s for a 40 A one, and a zero-delay `QTimer` armed
    immediately before the build had still not fired when the build returned --
    which is the measurement that says the event loop could not run at all
    during it. On the GUI thread that is a window which answers nothing for four
    seconds, with no explanation, the first time somebody selects a pose after
    moving the box to 40 A. So the first selection pays for the maps here, once,
    and every later one reads the cache.
    """

    ready = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, evaluator, result, pose_index: int) -> None:
        super().__init__()
        self._evaluator = evaluator
        self._result = result
        self._index = int(pose_index)

    @QtCore.pyqtSlot()
    def run(self) -> None:  # pragma: no cover - GUI
        from .energy_terms import TermsUnavailable

        try:
            breakdown = self._evaluator.breakdown(self._result, self._index)
        except TermsUnavailable as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"the term decomposition failed: {exc}")
        else:
            self.ready.emit(breakdown)


class MainWindow(QtWidgets.QMainWindow):
    """The workbench main window."""

    def __init__(self, receptor=None, ligand=None, poses=None):
        super().__init__()
        self.setWindowTitle("Open Docking Workbench")
        self.resize(1280, 820)

        self.viewport = Viewport()
        self.setCentralWidget(self.viewport)

        self._receptor_path: Path | None = None
        self._ligand_path: Path | None = None
        self._pose_view: MoleculeView | None = None
        self._pose_models: list[tuple[str, float]] = []
        # The `DockingResult` behind the poses on screen, when there is one. A
        # poses *file* carries a single energy per model and nothing else -- the
        # writer emits `REMARK VINA RESULT: -5.0 0.000 0.000`, so the round trip
        # through a file rounds the energy to one decimal and throws the
        # intermolecular part away. Keeping the result is what lets the table
        # show the engine's own numbers instead of the file's weaker ones.
        self._dock_result = None
        # Per-pose (hydrogen bonds, contacts, residues), filled for every pose
        # rather than for the selected one. See `_refresh_pose_contacts` for why.
        self._pose_contacts: list[tuple[int, int, int]] = []
        # One ghost view per reported pose, built once when the poses load.
        self._pose_ghosts: list = []
        self._pose_ghosts_on = False
        self._current_pose = -1
        #: True while the site table has a selection, i.e. while the user is
        #: looking at a *site*. One of the two states `_sync_pocket_opacity`
        #: reads, and the only one this file's own handlers set.
        self._pocket_selected = False
        #: The last move `focus_selection` ran, kept so a check or a status line
        #: can ask where the camera went without measuring it a second time.
        self._last_selection_plan = None
        self._pose_dists = None
        self._pocket_models: list = []
        self._pocket_thread: QtCore.QThread | None = None
        self._pocket_worker: PocketWorker | None = None
        self._maps = None
        self._worker = None
        self._thread = None
        # A handle on the loaded receptor, kept only so the map-size estimate
        # has something to ask. `estimate_memory_mb` is reached through a
        # `Receptor` even though the number depends on the box alone, so the
        # handle is built once per file and reused; see `_map_sizer`.
        self._sizer_path: Path | None = None
        self._sizer = None

        self._build_ui()
        for path, kind in ((receptor, "receptor"), (ligand, "ligand"), (poses, "poses")):
            if path:
                self.load_structure(Path(path), kind)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:  # pragma: no cover - GUI
        panel_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel_widget)

        self.cmb_representation = QtWidgets.QComboBox()
        for rep in REPRESENTATIONS:
            self.cmb_representation.addItem(rep.label, rep.key)
        self.cmb_representation.currentIndexChanged.connect(self._on_representation)
        form.addRow("display", self.cmb_representation)

        self.cb_receptor = QtWidgets.QCheckBox("receptor")
        self.cb_receptor.setChecked(True)
        self.cb_receptor.toggled.connect(self._on_visibility)
        form.addRow(self.cb_receptor)

        self.cb_ligand = QtWidgets.QCheckBox("ligand / pose")
        self.cb_ligand.setChecked(True)
        self.cb_ligand.toggled.connect(self._on_visibility)
        form.addRow(self.cb_ligand)

        self.cb_box = QtWidgets.QCheckBox("search box")
        self.cb_box.setChecked(True)
        self.cb_box.toggled.connect(self._on_box_visibility)
        form.addRow(self.cb_box)

        self.cb_contacts = QtWidgets.QCheckBox("interactions (dashed)")
        self.cb_contacts.setChecked(True)
        self.cb_contacts.toggled.connect(self._on_contact_visibility)
        form.addRow(self.cb_contacts)

        # A legend, because four line colours that mean nothing by themselves
        # are a worse feature than no colours. Each swatch is painted with the
        # same RGB the line renderer uses, read from one table, so the legend
        # cannot drift away from the picture.
        #
        # The site volume is in here too, and it is here because a report said
        # it was not: loading a receptor puts a box on the first pocket and
        # draws that pocket as a cloud of magenta spheres straight away, and
        # the only legend entry was the four dashed interaction lines. So a
        # colour appeared in the picture that nothing on screen named, and the
        # first reading of it was "what are those balls". A dot rather than a
        # line, because it is a volume and not a line -- the shape of the
        # swatch is part of what the legend is claiming.
        legend = QtWidgets.QWidget()
        legend_row = QtWidgets.QHBoxLayout(legend)
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.setSpacing(8)
        from . import COLOR_CONTACT, COLOR_POCKET, CONTACT_LABELS

        for kind in ("hbond", "polar", "hydrophobic", "close"):
            r, g, b = (int(round(v * 255)) for v in COLOR_CONTACT[kind])
            chip = QtWidgets.QLabel("━")
            chip.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold;")
            text = QtWidgets.QLabel(CONTACT_LABELS[kind])
            text.setStyleSheet("color: #9aa3ad;")
            legend_row.addWidget(chip)
            legend_row.addWidget(text)

        pr, pg, pb = (int(round(v * 255)) for v in COLOR_POCKET)
        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet(f"color: rgb({pr},{pg},{pb}); font-weight: bold;")
        dot.setToolTip(
            "The selected site's own free space, drawn through the protein so "
            "you can see whether it sits in a crevice. Toggle it with the "
            "'site volume' box below."
        )
        dtext = QtWidgets.QLabel("site volume")
        dtext.setStyleSheet("color: #9aa3ad;")
        legend_row.addWidget(dot)
        legend_row.addWidget(dtext)
        legend_row.addStretch(1)
        form.addRow("legend", legend)
        # Named, because a check has to be able to ask about *this* row rather
        # than about every label in the window. It used to reach for the whole
        # window and identify a legend word by its stylesheet, which meant any
        # new secondary-text label in the panel was counted as a legend entry
        # and the "every swatch names itself" check went red on a panel that was
        # telling the truth.
        self.legend_row = legend

        # Pocket candidates for the receptor, offered as a list the user picks
        # from rather than as a box that silently moved. Clicking a row places
        # the box and flies the camera to the site; the hand-set spin boxes stay
        # authoritative, so a click is a starting point and not a lock.
        #
        # The list exists because the box used to be centred on the receptor's
        # centroid, which for a globular protein is inside the dense core: a
        # search there is either empty or returns poses the engine itself calls
        # implausible. `odcli` refuses to guess for the same reason.
        self.btn_pockets = QtWidgets.QPushButton("Find pockets")
        self.btn_pockets.clicked.connect(self.find_pockets)
        form.addRow(self.btn_pockets)

        self.lbl_pockets = QtWidgets.QLabel("—")
        self.lbl_pockets.setWordWrap(True)
        form.addRow("site", self.lbl_pockets)

        # Five columns, and the fifth is the one the ranking is built from.
        # The list is ordered by `volume ** (1/3) * burial` and a user cannot
        # see either term, which made the order look arbitrary: a 6 x 10 x 17
        # A groove with 88 A3 of space in it outranked the 20 A3 pocket a
        # ligand was actually sitting in, and nothing on screen said why.
        # Showing the volume makes the order legible, and the kind column
        # already separates "sealed" from "groove", which is the other thing
        # worth knowing before clicking a row.
        self.pocket_table = QtWidgets.QTableWidget(0, 5)
        self.pocket_table.setHorizontalHeaderLabels(
            ["#", "kind", "centre", "size", "vol Å³"]
        )
        # Two columns are short by nature and three are not. Letting the
        # centre column stretch and the rest size to their contents is what
        # keeps "vol A^3" from being clipped to a "v" the way a last-section
        # stretch did: the stretch went to the widest column, which is the one
        # with the coordinates in it.
        head = self.pocket_table.horizontalHeader()
        for col in (0, 1, 3, 4):
            head.setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        head.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        head.setStretchLastSection(False)
        self.pocket_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.pocket_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        # Tall enough for about eight rows. The shortlist is twelve now --
        # crambin's real binding site is ninth -- so a table showing five made
        # the user scroll for something that used to be one click away.
        self.pocket_table.setMinimumHeight(250)
        self.pocket_table.verticalHeader().setVisible(False)
        self.pocket_table.itemSelectionChanged.connect(self._on_pocket_selected)
        form.addRow(self.pocket_table)

        self.cb_pocket_volume = QtWidgets.QCheckBox("site volume")
        self.cb_pocket_volume.setChecked(True)
        self.cb_pocket_volume.setToolTip(
            "Draw the selected site's own grid points as a translucent cloud, "
            "so the table's claim can be checked against the protein"
        )
        self.cb_pocket_volume.toggled.connect(self._on_pocket_visibility)
        form.addRow(self.cb_pocket_volume)

        # The centre must accept negative coordinates. They are ordinary, not
        # exotic: a receptor whose coordinates run from -40 to +10 has a
        # centroid well below zero, and with a (0, 999) range the spin silently
        # clamped it to 0 -- so the box was placed 40 A away from the molecule
        # and the user was shown a value that was never the receptor's.
        self.center_spins = self._spin_row(form, "box centre", -9999.0, 9999.0)
        self.size_spins = self._spin_row(form, "box size", 1.0, 500.0, value=22.0)

        # Exhaustiveness is a number of search walks, and walks are spread
        # through the box. A fixed 8 is right for a ligand-sized box and four
        # times too thin for a 40 A one -- the pocket search will happily hand
        # the user a 39 x 26 x 41 A box for a winding cleft, and a search that
        # under-samples reports a bad *pose*, which looks exactly like the box
        # being wrong. Measured on that box: 12.88 A RMSD at 16, 1.26 A at 64.
        #
        # So the spin box follows the box unless the user has set it
        # themselves, and the tooltip says why the number is not a constant.
        # `exhaustiveness_for_box` is the engine's, shared with `odcli` and the
        # redocking benchmark, so there is one rule in three places rather than
        # three rules that drift apart. It supplies the number this spin box
        # holds, and it also supplies the ceiling the note below compares that
        # number against -- see `_walks_wanted`, which asks it rather than
        # reading the ladder's last rung for itself.
        self.sp_exhaust = QtWidgets.QSpinBox()
        self.sp_exhaust.setRange(1, 256)
        self.sp_exhaust.setValue(8)
        self._exhaust_is_default = True
        # Set before the first `setValue` above can emit, and read by
        # `_suggest_exhaustiveness`'s guard. The attribute has to exist before
        # any signal can fire, so it is set here rather than at first use.
        self._updating_exhaust = False
        self.sp_exhaust.valueChanged.connect(self._on_exhaust_edited)

        # The latch was one-way with nothing in the repo able to clear it, so
        # one stray click on the spin box ended the suggestion for the rest of
        # the session -- which is the 12.88 A outcome this control exists to
        # prevent, arrived at by a single misclick. The way back is therefore a
        # button the user presses, and never a second suggestion made for them:
        # a control that undoes what you typed is worse than one that never
        # existed. It re-arms the rule; it does not itself pick a number, so
        # the value on screen only moves when the rule says it should.
        self.btn_suggest = QtWidgets.QPushButton("use suggested value")
        self.btn_suggest.setToolTip(
            "Hand exhaustiveness back to the box. Only you can do this: the "
            "window will not overwrite a number you set, and will not put the "
            "suggestion back on its own either."
        )
        self.btn_suggest.clicked.connect(self._use_suggested_exhaustiveness)
        ex_row = QtWidgets.QWidget()
        ex_layout = QtWidgets.QHBoxLayout(ex_row)
        ex_layout.setContentsMargins(0, 0, 0, 0)
        ex_layout.addWidget(self.sp_exhaust)
        ex_layout.addWidget(self.btn_suggest)
        form.addRow("exhaustiveness", ex_row)

        # What the search will cost, and whether the number above is actually
        # enough for this box. Two facts, one line each, because they are the
        # two ways a run goes wrong quietly: grids that do not fit in RAM, and
        # a box under-sampled at the top of the ladder.
        self.lbl_effort = QtWidgets.QLabel("maps: load a receptor to size them")
        self.lbl_effort.setWordWrap(True)
        self.lbl_effort.setToolTip(
            "Grid memory for the current box at the engine's default 0.375 A "
            "spacing, the same estimate `odcli rec-grid` prints. It grows with "
            "the cube of the box: 25 MB for a 20 A cube, 192 MB for 40 A, 2.9 GB "
            "for 100 A.\n\n"
            "Measured on a 60 A box: 0.62 s at 8 walks, 5.70 s at 128, a "
            "0.09 kcal/mol difference in the best energy. That is the whole "
            "argument for spending effort on a big box, and also the reason "
            "the ladder's steps are visible in the clock."
        )
        form.addRow("search cost", self.lbl_effort)

        # Both labels say something before anything is loaded, rather than
        # sitting empty until the first box change: a control with no tooltip
        # yet is a control the user hovers over and learns nothing from.
        self._describe_exhaustiveness(True)
        self._refresh_effort_note()

        self.sp_seed = QtWidgets.QSpinBox()
        self.sp_seed.setRange(0, 2**31 - 1)
        self.sp_seed.setValue(0)
        form.addRow("seed (0 = random)", self.sp_seed)

        self.cb_scoring = QtWidgets.QComboBox()
        self.cb_scoring.addItems(["vina", "vinardo"])
        form.addRow("scoring", self.cb_scoring)

        btn_frame = QtWidgets.QPushButton("Frame all")
        btn_frame.clicked.connect(self.viewport.frame_all)
        form.addRow(btn_frame)

        self.btn_dock = QtWidgets.QPushButton("Dock")
        self.btn_dock.clicked.connect(self.start_docking)
        form.addRow(self.btn_dock)

        # A table, not a list of bare numbers. Nine rows of "pose 3 -4.71
        # kcal/mol" answer none of the questions a docking result exists to
        # answer: are these nine different answers, how far is each from the
        # best one, and does the second-best one have any hydrogen bonds at all.
        # Every column below is a number the code can actually produce; there is
        # no column here whose value had to be invented to fill the space.
        self.pose_table = PoseTable(0, 7)
        self.pose_table.setHorizontalHeaderLabels(
            ["pose", "kcal/mol", "inter", "RMSD", "gap", "H", "contacts"]
        )
        # Every column stretches, so the table always fits the panel it is in.
        # `ResizeToContents` on seven numeric columns asks for more width than
        # the 490 px control panel has, and the panel's scroll area has its
        # horizontal scrollbar switched off -- so the last three columns were
        # simply not on screen, with nothing saying so. A column that is cut off
        # is worse than no column.
        head = self.pose_table.horizontalHeader()
        head.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        head.setStretchLastSection(False)
        # Below this, Qt is free to shrink a column under its own header and
        # elide the text; measured, the seven sections overflowed the field
        # column by 51 px and the last three columns were off screen entirely.
        head.setMinimumSectionSize(40)
        self.pose_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.pose_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.pose_table.setSortingEnabled(True)
        # Tall enough for the nine poses a run reports, so the whole result is
        # readable without scrolling -- the same reasoning as the site table's
        # 250 px, measured rather than guessed: nine rows at 29 px plus a 25 px
        # header and the frame is 290.
        self.pose_table.setMinimumHeight(290)
        self.pose_table.verticalHeader().setVisible(False)
        self.pose_table.currentCellChanged.connect(self._on_pose_selected)
        self.pose_table.setToolTip(
            "Every column is a number the engine or this window measured.\n\n"
            "pose: the rank the engine reported, best first. It does not "
            "renumber when you sort, because it is a fact about the pose.\n"
            "kcal/mol: total energy. inter: the receptor-ligand part of it, "
            "recorded only for a run made in this window -- a pose file "
            "carries one energy per model and drops the rest.\n"
            "RMSD: distance to the best pose. gap: distance to the nearest "
            "other reported pose, which is what makes this row a different "
            f"answer; the engine treats anything under {POSE_CLUSTER_CUTOFF} A "
            "as the same mode, so a gap below that is a pair the clustering "
            "failed to separate, shown in red."
            "H and contacts: pose-to-receptor hydrogen bonds and contacts, "
            "counted for every pose, not only the selected one."
        )
        # A full-width row, like the site table and the contact table, and for
        # the same reason: a label column in the form takes 107 px, which left
        # the field column 344 px -- not enough for seven columns whose headers
        # are `kcal/mol` (49 px) and `contacts` (48 px) wide. Full width, the
        # seven sections are 70 px each and every header is legible. The column
        # names are the table's own header row, which is what a table is for.
        form.addRow(self.pose_table)
        # `workbench_smoke.py` reads `pose_list`; it is not this file's to edit,
        # so the name it uses still resolves. See `PoseTable`.
        self.pose_list = self.pose_table

        self.cb_all_poses = QtWidgets.QCheckBox("all poses ghosted together")
        self.cb_all_poses.setToolTip(
            "Draw every reported pose at once, each one faint, with the "
            "selected pose solid on top, and move the camera in to look at "
            "them. It answers 'are poses 2 and 3 the same place reached "
            "twice, or two different answers', which the gap column answers "
            "only in numbers."
        )
        self.cb_all_poses.toggled.connect(self._on_all_poses_visibility)
        form.addRow(self.cb_all_poses)

        # The on/off state has to be readable as words, not only as a picture.
        # A translucent ligand lying across a solid one looks like a rendering
        # fault, and a checkbox the user set an hour ago is not enough to
        # explain the picture in front of them.
        self.lbl_poses = QtWidgets.QLabel("no poses")
        self.lbl_poses.setWordWrap(True)
        form.addRow("showing", self.lbl_poses)

        self.lbl_energy = QtWidgets.QLabel("—")
        self.lbl_energy.setStyleSheet("font-weight: bold; font-size: 15px;")
        form.addRow("best energy", self.lbl_energy)

        self.lbl_rmsd = QtWidgets.QLabel("—")
        form.addRow("RMSD to best", self.lbl_rmsd)

        # Which term produced this ranking. The engine decomposes a pose's
        # energy into the five weighted Vina terms, and until this panel
        # existed the only way to read them was to leave the GUI: a user could
        # see that pose 3 beat pose 1 and not why. The terms are named the way
        # `docs/SCORING.md` names them, and the engine's own keys sit in the
        # tooltips so the numbers can be traced back to `core.py`.
        #
        # Two things this widget is shaped around. It names the pose it is
        # describing in its own header, because a table of energies with no pose
        # on it is a set of numbers about an unknown molecule -- and a panel
        # quietly showing the *previous* pose is worse than no panel. And every
        # row is always present, including a term that came back as exactly
        # zero, because "this pose is ranked by four terms and not five" is
        # information and an absent row is not.
        terms_box = QtWidgets.QGroupBox("pose energy breakdown")
        terms_layout = QtWidgets.QVBoxLayout(terms_box)
        self.lbl_terms_pose = QtWidgets.QLabel("no pose selected")
        self.lbl_terms_pose.setWordWrap(True)
        self.lbl_terms_pose.setStyleSheet("font-weight: bold;")
        terms_layout.addWidget(self.lbl_terms_pose)
        self.tbl_terms = QtWidgets.QTableWidget(0, 2)
        self.tbl_terms.setHorizontalHeaderLabels(["term", "kcal/mol"])
        terms_head = self.tbl_terms.horizontalHeader()
        terms_head.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        terms_head.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.tbl_terms.verticalHeader().setVisible(False)
        self.tbl_terms.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        # Tall enough for all ten rows -- five terms and the five the engine
        # returns so the decomposition can be checked. The first version left
        # the default height, which clipped "sum of the five terms" behind a
        # scrollbar: the row a reader most wants to compare against the terms
        # above it was the one that could not be seen. 282 and 310 each still
        # clipped the last row; a header plus eleven rows measures about 350,
        # and the screenshots are what caught both. The eleventh row is the
        # engine's own sum of the five, which the column-reconciliation fix
        # added rather than replacing the displayed one.
        self.tbl_terms.setMinimumHeight(364)
        self.tbl_terms.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        terms_layout.addWidget(self.tbl_terms)
        self.lbl_terms_note = QtWidgets.QLabel("")
        self.lbl_terms_note.setWordWrap(True)
        self.lbl_terms_note.setStyleSheet("color: #9aa3ad; font-size: 11px;")
        terms_layout.addWidget(self.lbl_terms_note)
        # The next action, as a button, because "these poses came from a file
        # and a file has no conformation vector" is an explanation and not a
        # way forward. The engine's decomposition is keyed on the conformation
        # and there is no inverse -- `conformation_coordinates` goes one way and
        # nothing in the public API comes back -- so the only honest route from a
        # file pose to a decomposition is to dock the ligand here, which reports
        # poses *with* their conformations. Hidden unless that is exactly the
        # situation, so it never sits there looking available and being useless.
        self.btn_terms_dock = QtWidgets.QPushButton("Re-dock this ligand here")
        self.btn_terms_dock.setVisible(False)
        self.btn_terms_dock.setToolTip(
            "Runs a docking with the receptor and ligand already loaded. The "
            "poses it reports carry the conformations the term decomposition "
            "is keyed on, so the breakdown appears for the pose you select "
            "afterwards."
        )
        self.btn_terms_dock.clicked.connect(self._on_terms_dock_clicked)
        terms_layout.addWidget(self.btn_terms_dock)
        form.addRow(terms_box)
        # Kept so a screenshot can be of the panel itself rather than of the
        # whole window, which is what makes "which pose is it describing"
        # legible in a picture instead of a fact in a log.
        self.terms_box = terms_box
        # The pose the table is describing, and the evaluator that produced it.
        # Both are written here and nowhere else, so "which pose is this" and
        # "whose maps are these" have one answer each.
        self._terms_pose_index = -1
        #: The breakdown currently on screen, kept so the verdict panel can
        #: reuse this pose's out-of-box penalty instead of asking the engine
        #: for it a second time. Written only by `_terms_fill` and cleared only
        #: by `_refresh_energy_terms`, and the verdict panel uses it only when
        #: its `pose_index` is the selected pose -- a penalty belonging to
        #: another pose is refused rather than shown.
        self._terms_breakdown = None
        self._terms_evaluator = None
        self._terms_evaluator_key = None
        self._terms_thread = None
        self._terms_worker = None

        # Can *this* pose's numbers be trusted? The breakdown above answers
        # "which terms produced this energy"; this one answers the other
        # question, which is whether the energy is the field's own value at
        # this pose at all.
        #
        # It is here, beside the breakdown, for the same reason the breakdown
        # is where it is: both are per-pose, both name the pose they are about,
        # and the two answers have to be read together. A panel that reported a
        # `yes` next to a breakdown whose `out_of_box_penalty` was non-zero
        # would be two of this file's panels disagreeing in the reader's face.
        #
        # Three states, and the third is the point. `yes`, `no` and `unknown`
        # are three different answers, and `unknown` is **not** a softer `yes`:
        # it means a measurement was not made, and a reader who cannot tell it
        # from a pass will read it as one. So the state is a word in the header
        # and a word in every row, and the colour is a second channel rather
        # than the only one. `scripts/pose_trust_check.py` asserts the three
        # treatments differ, so the panel cannot quietly converge on the pass
        # style for a contract nothing measured.
        verdict_box = QtWidgets.QGroupBox("pose verdict")
        verdict_layout = QtWidgets.QVBoxLayout(verdict_box)
        self.lbl_verdict_head = QtWidgets.QLabel("no pose selected")
        self.lbl_verdict_head.setWordWrap(True)
        self.lbl_verdict_head.setStyleSheet("font-weight: bold;")
        verdict_layout.addWidget(self.lbl_verdict_head)
        # Three columns: what was checked, what came out, and the pair of
        # numbers that decided it. The remedy is not a column -- it is a
        # sentence, and a sentence in a table cell is a sentence nobody reads.
        self.tbl_verdict = QtWidgets.QTableWidget(0, 3)
        self.tbl_verdict.setHorizontalHeaderLabels(
            ["contract", "state", "measured / threshold"]
        )
        verdict_head = self.tbl_verdict.horizontalHeader()
        verdict_head.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        verdict_head.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        verdict_head.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.tbl_verdict.verticalHeader().setVisible(False)
        self.tbl_verdict.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.tbl_verdict.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        # Four rows and a header, measured the same way the breakdown's height
        # was: a 29 px row plus a 25 px header, and the rows are always all
        # four -- a contract that was not measured is information, and an absent
        # row is not.
        self.tbl_verdict.setMinimumHeight(150)
        verdict_layout.addWidget(self.tbl_verdict)
        self.lbl_verdict_why = QtWidgets.QLabel("")
        self.lbl_verdict_why.setWordWrap(True)
        self.lbl_verdict_why.setStyleSheet("font-size: 11px;")
        verdict_layout.addWidget(self.lbl_verdict_why)
        self.lbl_verdict_note = QtWidgets.QLabel("")
        self.lbl_verdict_note.setWordWrap(True)
        self.lbl_verdict_note.setStyleSheet("color: #9aa3ad; font-size: 11px;")
        verdict_layout.addWidget(self.lbl_verdict_note)
        # Kept for the same reason `terms_box` is: so a screenshot can be of
        # this panel alone, which is the only way "which pose is it describing"
        # and "is `unknown` distinguishable from a pass" are questions about a
        # picture rather than about a log.
        self.verdict_box = verdict_box
        # The pose this panel last described. -1 is "nothing on screen", and it
        # is the value the gate compares the selection against.
        self._verdict_pose_index = -1
        form.addRow(verdict_box)

        # Residue-level summary of the pose/receptor interface, one row per
        # residue. Selecting a row flies the camera to that contact, which is
        # the whole reason the table exists: "THR 2 is hydrogen-bonded" is a
        # fact, "THR 2 is over there" is the question you had when you clicked.
        self.lbl_contacts = QtWidgets.QLabel("—")
        self.lbl_contacts.setWordWrap(True)
        form.addRow("interactions", self.lbl_contacts)

        self.contact_table = QtWidgets.QTableWidget(0, 4)
        self.contact_table.setHorizontalHeaderLabels(
            ["residue", "H-bond", "contacts", "closest"]
        )
        self.contact_table.horizontalHeader().setStretchLastSection(True)
        self.contact_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.contact_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.contact_table.setMinimumHeight(150)
        self.contact_table.itemSelectionChanged.connect(self._on_contact_selected)
        form.addRow(self.contact_table)

        self.status_label = QtWidgets.QLabel("ready — load a receptor and a ligand")
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        dock = QtWidgets.QDockWidget("Controls", self)
        # A plain widget in a dock is clipped, not scrolled: once the panel is
        # taller than the window the Dock button and everything below it become
        # unreachable, and a control you cannot reach is not a control. The
        # contacts table alone is 150 px tall, which is what pushed it over.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel_widget)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        menu = self.menuBar().addMenu("File")
        act_rec = menu.addAction("Load receptor…")
        act_rec.setShortcut("Ctrl+R")
        act_rec.triggered.connect(lambda: self.open_file_dialog("receptor"))
        act_lig = menu.addAction("Load ligand…")
        act_lig.setShortcut("Ctrl+L")
        act_lig.triggered.connect(lambda: self.open_file_dialog("ligand"))
        act_poses = menu.addAction("Open poses…")
        act_poses.setShortcut("Ctrl+P")
        act_poses.triggered.connect(lambda: self.open_file_dialog("poses"))
        menu.addSeparator()
        act_dock = menu.addAction("Dock now")
        act_dock.setShortcut("Ctrl+D")
        act_dock.triggered.connect(self.start_docking)
        menu.addSeparator()
        act_quit = menu.addAction("Quit")
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

    def _spin_row(self, form, label, lo, hi, value=0.0):  # pragma: no cover - GUI
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        spins = []
        for _ in range(3):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setValue(value)
            spin.valueChanged.connect(self._on_box_changed)
            layout.addWidget(spin)
            spins.append(spin)
        form.addRow(label, holder)
        return spins

    # -- Data loading ------------------------------------------------------

    def load_structure(
        self, path: Path, kind: str, result=None
    ) -> None:  # pragma: no cover - GUI
        """Load a receptor, ligand, or multi-pose file into the viewport.

        ``result`` is the `DockingResult` the poses came from, when they came
        from a run in this window. It is the only way to the intermolecular
        energies and the engine's own RMSDs, because the pose file does not
        carry them. Omitted for a file the user opened, which is the honest
        case: those columns then say they do not know.
        """
        from . import COLOR_BEST_POSE, COLOR_LIGAND, COLOR_RECEPTOR

        if kind == "poses":
            from ..pdbqt_writer import PdbqtFormatError, read_pdbqt_models

            # `read_pdbqt_models` hands back one *list of lines* per pose. Join
            # once here, because everything downstream -- the energy remark, the
            # RMSD comparison and the coordinate parse -- works on text. Passing
            # the line lists straight through crashed on the first
            # `.splitlines()`, which took down the whole window: `odgui -p
            # poses.pdbqt` could not start at all.
            #
            # The reader refuses a file whose columns are wrong, by default and
            # on purpose, because the engine's own reader splits on whitespace
            # and would load it and report the wrong coordinates without ever
            # saying so. That decision is right, and it puts the obligation on
            # this side: a user who opens a hand-edited pose file must be told
            # which line is wrong, not handed a traceback out of a GUI handler.
            #
            # Only `PdbqtFormatError`. A bare `except ValueError` here would
            # also swallow a genuine bug in the reader and dress it up as "your
            # file is malformed", which sends the user off to edit a file that
            # is fine.
            try:
                models = ["\n".join(lines) for lines in read_pdbqt_models(path)]
            except PdbqtFormatError as exc:
                first = exc.problems[0] if exc.problems else "no detail given"
                extra = len(exc.problems) - 1
                self.status_label.setText(
                    f"{exc.source}: {len(exc.problems)} line(s) are not laid out "
                    f"as PDBQT requires -- {first}"
                    + (f" (and {extra} more)" if extra > 0 else "")
                )
                return
            if not models:
                self.status_label.setText(f"{path.name} contains no poses")
                return
            self._pose_models = [(body, _energy_of(body)) for body in models]
            self._pose_dists = None
            # Attach the result only if it really describes these poses. A
            # result with a different pose count is a different run, and pairing
            # the two would put one run's energy on another's coordinates --
            # which is worse than showing nothing, because it looks measured.
            self._dock_result = None
            if result is not None and getattr(result, "num_poses", 0) == len(models):
                self._dock_result = result
            self._refresh_pose_table()
            best = self._best_row()
            # Build the view *before* selecting the row. `setCurrentRow` emits
            # `currentRowChanged`, and `_on_pose_selected` bails out while
            # `self._pose_view` is still None -- so selecting first left the
            # RMSD-to-best readout stuck on "—" for the whole session.
            self._pose_view = self._view_from_text(
                self._pose_models[best][0], path.name, COLOR_BEST_POSE, 0.34
            )
            self._pose_view.role = "pose"
            # Drop any previous pose view instead of stacking a new one on top.
            # Every run writes to a differently-named temp file, so identity has
            # to come from the role, not the name. With the old code each run
            # left its result in the scene, and the "ligand" checkbox could no
            # longer hide them because `_on_visibility` only knew about the
            # current view.
            self.viewport.molecules = [
                m for m in self.viewport.molecules if m.role != "pose"
            ]
            self.viewport.molecules.append(self._pose_view)
            self.lbl_energy.setText(
                f"{self._pose_energy(best):.2f} kcal/mol"
                if self._pose_energy(best) is not None
                else "energy not in file"
            )
            self.lbl_rmsd.setText("0.00 Å")
            self._pose_ghosts = self._build_pose_ghosts(path.name)
            self._current_pose = -1
            self.pose_table.blockSignals(True)
            self.pose_table.setSortingEnabled(False)
            self._select_pose_row(best)
            self.pose_table.setSortingEnabled(True)
            self.pose_table.blockSignals(False)
            self._on_pose_selected(self.pose_table.currentRow())
            self.status_label.setText(f"loaded {len(models)} pose(s) from {path.name}")
        else:
            view = MoleculeView.from_pdbqt(
                path,
                path.name,
                COLOR_RECEPTOR if kind == "receptor" else COLOR_LIGAND,
                0.30 if kind == "receptor" else 0.32,
                role=kind,
            )
            # Reloading a receptor or a ligand replaces the previous one rather
            # than leaving a ghost of it in the scene.
            self.viewport.molecules = [
                m for m in self.viewport.molecules if m.role != kind
            ]
            self.viewport.molecules.append(view)
            if kind == "receptor":
                self._receptor_path = path
                # Parse the engine's handle now, while this branch is already
                # blocking for the pocket search below, so the map-size estimate
                # is a 0.2 us lookup on every later box change instead of a
                # receptor parse per drag. Measured: 0.4 ms for the 30-atom
                # example, 240 ms at 5000 atoms.
                self._map_sizer()
                # This used to put the box centre on the receptor's centroid.
                # For a globular protein that is inside the dense core, so the
                # search region sat in solid protein: either it came back
                # empty or the engine returned poses it then reported as
                # implausible, and the three spin boxes showed numbers that
                # looked like a deliberate answer. Nothing about a centroid
                # suggests a binding site, so the box now goes on the first
                # pocket the search offers -- visibly, with the row selected,
                # so the user can see where it came from and choose another.
                # Synchronous on purpose. Loading a receptor is the one moment
                # where the window may block: the alternative is a box sitting
                # at the old position with a search running behind it, and the
                # user starts a docking run against a box nobody chose. The
                # button and every repeat search go through the worker instead,
                # so the twelve-second haemoglobin case cannot freeze a session
                # the user is already using.
                found = self.find_pockets_now()
                if found:
                    self._auto_select_pocket_row(0)
                else:
                    self.lbl_pockets.setText(
                        "no enclosed site found — the box is still where you "
                        "left it, not a suggestion"
                    )
            else:
                self._ligand_path = path
            self.status_label.setText(
                f"loaded {path.name} ({len(view.coords)} atoms)"
            )
        self._refresh_contacts()
        self._refresh_pose_contacts()
        self._refresh_effort_note()
        self.viewport.frame_all()

    # -- the pose table ----------------------------------------------------

    def _pose_energy(self, index: int) -> float | None:
        """Pose ``index``'s total energy, in kcal/mol, from the best source.

        The engine's number when the run is still in memory, otherwise the file's
        `REMARK VINA RESULT`. Both the table and the big label come through
        here, so the two can never show different energies for the same pose --
        and the engine's is worth preferring: the writer rounds to one decimal
        (`-4.973` is written `-5.0`), which is coarse enough to make poses 1 and
        2 tie when the engine can tell them apart by 0.011 kcal/mol.
        """
        if not 0 <= index < len(self._pose_models):
            return None
        result = self._dock_result
        if result is not None:
            try:
                return float(result.energies[index])
            except (IndexError, TypeError, ValueError):
                pass
        return self._pose_models[index][1]

    def _pose_inter(self, index: int) -> float | None:
        """Pose ``index``'s receptor-ligand energy, or ``None`` if not recorded.

        Only a run in this window has one. The pose writer emits
        `REMARK VINA RESULT: <total> 0.000 0.000`, so the second and third
        numbers in the AutoDock convention are placeholders and the
        intermolecular part does not survive the file. Showing "—" is the
        honest answer; showing the total again under a second heading would look
        like two independent measurements of the same quantity.
        """
        result = self._dock_result
        if result is None or not 0 <= index < len(self._pose_models):
            return None
        try:
            return float(result.intermolecular_energies[index])
        except (IndexError, TypeError, ValueError):
            return None

    def _pose_rmsd(self, index: int) -> float | None:
        """Pose ``index``'s per-atom RMSD to the best pose, in angstrom.

        The engine's own value when the run is in memory, the same coordinate
        comparison otherwise. Both are the same quantity -- measured, they agree
        to 0.001 A -- so there is one number per pose in this window whichever
        way the poses arrived.
        """
        result = self._dock_result
        if result is not None and 0 <= index < len(self._pose_models):
            try:
                return float(result.rmsds[index])
            except (IndexError, TypeError, ValueError):
                pass
        return self._rmsd_to_best(index)

    def _pose_distance_matrix(self) -> np.ndarray:
        """Per-atom RMSD between every pair of reported poses, in angstrom.

        Parsed once, then n x n. The pose *gap* column needs one number per row
        and every comparison is against a pose this window already has in
        memory, so re-parsing per pair would be paying for the same text 81
        times to produce a matrix of 81 small differences.
        """
        n = len(self._pose_models)
        cached = getattr(self, "_pose_dists", None)
        if cached is not None and cached.shape == (n, n):
            return cached
        coords = [_parse_pdbqt_atoms(body)[0] for body, _ in self._pose_models]
        dists = np.zeros((n, n), np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                m = min(len(coords[i]), len(coords[j]))
                if m == 0:
                    continue
                d = float(
                    np.sqrt(
                        ((coords[i][:m] - coords[j][:m]) ** 2).sum(axis=1).mean()
                    )
                )
                dists[i, j] = d
                dists[j, i] = d
        self._pose_dists = dists
        return dists

    def _pose_gap(self, index: int) -> tuple[float | None, int]:
        """Pose ``index``'s distance to the nearest *other* reported pose.

        Returns ``(gap, other_index)``. One definition for every row, which is
        why it is the nearest pose rather than the nearest *better* one: a cell
        that is blank for the best pose is a blank cell, and a blank cell sorts
        to the top of a descending sort, which is how the best pose ends up
        displayed as if it were the odd one out.

        This is the column that answers "are these nine answers or one answer
        nine times". The engine clusters by RMSD before reporting, so every
        reported pose should sit at least `POSE_CLUSTER_CUTOFF` from every other
        one, and a row that does not is a pair the clustering failed to separate
        -- shown in red, with the pose it duplicates named, rather than rounded
        away. Measured on crambin + biotin at exhaustiveness 8, the tightest
        gap in a nine-pose result is 1.01 A against that 1.0 A cutoff, so the
        column is not decoration: this run really does have a pair 1% apart.
        """
        dists = self._pose_distance_matrix()
        if not 0 <= index < len(self._pose_models):
            return None, -1
        row = dists[index]
        if len(row) < 2:
            return None, -1
        nearest = int(np.argmin(np.where(np.arange(len(row)) == index, np.inf, row)))
        if nearest == index or not np.isfinite(row[nearest]):
            return None, -1
        return float(row[nearest]), nearest

    def _refresh_pose_contacts(self) -> None:  # pragma: no cover - GUI
        """Count pose-to-receptor contacts for *every* pose, once.

        Measured, not guessed: `find_contacts` costs 0.5 ms per pose against the
        30-atom example receptor, 3.6 ms against crambin's 382 atoms and 8.9 ms
        against a 4832-atom one -- so all nine poses are 5, 33 and 80 ms. That
        is affordable enough to do up front, and doing it up front is the point:
        a contact column filled in only for the row you happen to click is a
        column that says a good-energy pose has no hydrogen bonds when nobody
        has looked at it yet, which is precisely the misreading it exists to
        prevent.
        """
        from .contacts import find_contacts, residue_summary

        self._pose_contacts = []
        receptor = next(
            (m for m in self.viewport.molecules if m.role == "receptor"), None
        )
        if receptor is None or not self._pose_models:
            self._fill_pose_contact_cells()
            return
        for body, _ in self._pose_models:
            pose = self._view_from_text(body, "pose", (0.4, 0.9, 0.45), 0.34)
            found = find_contacts(pose, receptor)
            hbonds = sum(1 for c in found if c.kind == "hbond")
            self._pose_contacts.append(
                (hbonds, len(found), len(residue_summary(found)))
            )
        self._fill_pose_contact_cells()

    def _fill_pose_contact_cells(self) -> None:  # pragma: no cover - GUI
        """Put the counted contacts into the H and contacts columns."""
        table = self.pose_table
        for row in range(table.rowCount()):
            index = self._pose_index_at(row)
            if index is None or not 0 <= index < len(self._pose_contacts):
                continue
            hbonds, total, residues = self._pose_contacts[index]
            self._set_cell(row, 5, str(hbonds), hbonds)
            item = self._set_cell(row, 6, str(total), total)
            if item is not None:
                item.setToolTip(
                    f"pose {index + 1}: {total} contacts over {residues} residues"
                )

    def _pose_index_at(self, row: int) -> int | None:
        """The pose a visual row shows, which sorting may have moved."""
        item = self.pose_table.item(row, 0)
        if item is None:
            return None
        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _select_pose_row(self, index: int) -> None:  # pragma: no cover - GUI
        """Select the visual row showing pose ``index``."""
        for row in range(self.pose_table.rowCount()):
            if self._pose_index_at(row) == index:
                self.pose_table.setCurrentCell(row, 0)
                return

    def _set_cell(self, row: int, col: int, text: str, value=None) -> _PoseItem:
        item = _PoseItem(text)
        if value is not None:
            item.setData(QtCore.Qt.ItemDataRole.UserRole, float(value))
        if col > 0:
            item.setTextAlignment(
                int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            )
        # Deliberately no foreground colour here. A cell with its own colour
        # keeps it when the row is selected, and the selection then draws the
        # text in the highlight's colour over a background that is not the
        # highlight's -- which is how "1 *" came to read as "*" with a blue bar
        # where the 1 should be. The style owns the selected row; the window
        # only colours the one cell that carries a fact.
        self.pose_table.setItem(row, col, item)
        return item

    def _refresh_pose_table(self) -> None:  # pragma: no cover - GUI
        """Fill the pose table from the poses in memory.

        The rank cell carries the pose's index in `UserRole`, so every lookup
        after this goes through the row the user can see rather than through a
        visual row number, which sorting is free to rearrange. The best pose is
        marked in its own cell for the same reason: a badge that jumps to
        whichever row happens to be first after a sort is worse than no badge.
        """
        table = self.pose_table
        table.blockSignals(True)
        was_sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.setRowCount(0)
        best = self._best_row()
        n = len(self._pose_models)
        table.setRowCount(n)
        for index in range(n):
            rank = _PoseItem(f"{index + 1}")
            rank.setData(QtCore.Qt.ItemDataRole.UserRole, index)
            # Centred, not left-aligned: a selected row is framed at the cell
            # edge and a left-aligned first character ends up underneath that
            # frame. The rank cell is the only one the window colours.
            rank.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
            if index == best:
                rank.setText(f"{index + 1} *")
                rank.setToolTip(
                    "best pose: the lowest total energy of this result"
                )
                rank.setForeground(QtGui.QBrush(QtGui.QColor(0x5C, 0xE0, 0x7A)))
            table.setItem(index, 0, rank)

            energy = self._pose_energy(index)
            self._set_cell(
                index, 1, f"{energy:.2f}" if energy is not None else "n/a", energy
            )
            inter = self._pose_inter(index)
            inter_cell = self._set_cell(
                index, 2, f"{inter:.2f}" if inter is not None else "—", inter
            )
            if inter is None and inter_cell is not None:
                inter_cell.setToolTip(
                    "not recorded: a pose file carries one energy per model"
                )
            rmsd = self._pose_rmsd(index)
            self._set_cell(index, 3, f"{rmsd:.2f}" if rmsd is not None else "n/a", rmsd)
            gap, which = self._pose_gap(index)
            gap_cell = self._set_cell(
                index, 4, f"{gap:.2f}" if gap is not None else "—", gap
            )
            if gap_cell is not None:
                if gap is None:
                    gap_cell.setToolTip("only one pose: nothing to be near")
                elif gap < POSE_CLUSTER_CUTOFF:
                    gap_cell.setToolTip(
                        f"only {gap:.2f} A from pose {which + 1}, under the "
                        f"{POSE_CLUSTER_CUTOFF} A the engine clusters at: these two "
                        "are the same mode and the result is reporting it twice"
                    )
                    gap_cell.setForeground(QtGui.QBrush(QtGui.QColor(0xE0, 0x6C, 0x6C)))
                else:
                    margin = (
                        " -- only just above it"
                        if gap < POSE_CLUSTER_CUTOFF * 1.1
                        else ""
                    )
                    gap_cell.setToolTip(
                        f"{gap:.2f} A from pose {which + 1}, its nearest neighbour "
                        f"(the engine clusters at {POSE_CLUSTER_CUTOFF} A){margin}"
                    )
        self._fill_pose_contact_cells()
        # The default order is the order the engine reported, best first. Left to
        # itself a freshly enabled sorting table orders the first column
        # descending, which put pose 9 at the top and the best pose at the
        # bottom -- the ranking the engine spent the search computing, upside
        # down, on the first frame the user saw.
        table.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
        table.setSortingEnabled(was_sorting)
        table.blockSignals(False)
        self._update_poses_label()

    def _update_poses_label(self) -> None:  # pragma: no cover - GUI
        """Say in words what the picture is showing.

        A faint ligand lying across a solid one looks like a rendering fault, so
        the state of the overlay is stated next to the control that sets it
        rather than left to be inferred from the picture.
        """
        n = len(self._pose_models)
        if not n:
            self.lbl_poses.setText("no poses")
            return
        if self._current_pose < 0:
            self.lbl_poses.setText(f"{n} pose(s) loaded")
            return
        if self._pose_ghosts_on:
            self.lbl_poses.setText(
                f"pose {self._current_pose + 1} of {n} solid, "
                f"{n - 1} ghosted at once"
            )
        else:
            self.lbl_poses.setText(f"pose {self._current_pose + 1} of {n}, on its own")

    def _build_pose_ghosts(self, name: str) -> list:
        """One faint view per reported pose, built once and shown on demand.

        `POSE_GHOST_COLOR` and not the pose's own green, and thinner than it, so
        a ghost can never be mistaken for the selected pose where they overlap.
        The flat colour only reaches the framebuffer because `_draw_molecule`
        overrides the element colours for this role -- see the note there. The
        ghosts are not parsed again on every selection: only their `visible` flag
        moves. The list position *is* the pose index -- `MoleculeView` is not
        this file's to extend, so nothing is stashed on the view.
        """
        self._pose_ghosts = []
        for i, (body, _) in enumerate(self._pose_models):
            view = self._view_from_text(
                body, f"{name} pose {i + 1}", POSE_GHOST_COLOR, 0.26
            )
            view.role = "pose_ghost"
            view.opacity = 0.30
            view.visible = False
            self._pose_ghosts.append(view)
        self._sync_ghosts_in_scene()
        return self._pose_ghosts

    def _sync_ghosts_in_scene(self) -> None:  # pragma: no cover - GUI
        """Keep the ghost views in the scene exactly while the overlay is on."""
        self.viewport.molecules = [
            m for m in self.viewport.molecules if m.role != "pose_ghost"
        ]
        if self._pose_ghosts_on:
            self.viewport.molecules.extend(self._pose_ghosts)
        self._apply_pose_visibility()

    def _apply_pose_visibility(self) -> None:  # pragma: no cover - GUI
        """One place that decides which pose views are drawn.

        Both the "ligand / pose" checkbox and the overlay checkbox change what
        should be on screen, and each of them used to write `visible` on its own.
        Whichever ran last won, so hiding the ligand with the overlay on left
        the ghosts drawn -- nine faint ligands and no solid pose, which is the
        opposite of what either checkbox says.
        """
        ligands = self.cb_ligand.isChecked()
        for mol in self.viewport.molecules:
            if mol.role == "receptor":
                mol.visible = self.cb_receptor.isChecked()
            elif mol.role != "pose_ghost":
                mol.visible = ligands
        present = {id(m) for m in self.viewport.molecules}
        for index, view in enumerate(self._pose_ghosts):
            if id(view) in present:
                view.visible = (
                    ligands and self._pose_ghosts_on and index != self._current_pose
                )

    def _best_row(self) -> int:
        """Index of the lowest-energy pose, or 0 when no energy is known.

        Ties and all-``None`` both fall back to the first row rather than an
        arbitrary pick from an all-equal list. Energies come from
        `_pose_energy`, so "best" means the best number on screen rather than the
        best number the file happened to keep.
        """
        known = [self._pose_energy(i) for i in range(len(self._pose_models))]
        known = [e for e in known if e is not None]
        if not known:
            return 0
        lowest = min(known)
        return next(i for i, e in enumerate(known) if e == lowest)

    def _view_from_text(self, text, name, color, radius) -> MoleculeView:  # pragma: no cover
        coords, elements = _parse_pdbqt_atoms(text)
        return MoleculeView(
            name=name,
            coords=coords,
            elements=elements,
            bonds=_pose_bonds(coords, elements),
            color=color,
            radius=radius,
            # Stated rather than inherited: the default happens to be the same
            # string, but the default is a coincidence of a dataclass field and
            # this is a decision about provenance. See `_pose_bonds`.
            bond_source="distance",
        )

    # -- UI callbacks ------------------------------------------------------

    def _on_visibility(self) -> None:  # pragma: no cover - GUI
        self._apply_pose_visibility()
        self.viewport.update()

    def _on_all_poses_visibility(self, checked: bool) -> None:  # pragma: no cover - GUI
        """Draw every reported pose at once, each one faint.

        The picture this produces is ambiguous on its own -- a translucent
        ligand across a solid one reads as a rendering fault -- so the state is
        also stated in words next to the checkbox, in `_update_poses_label`.

        **Why the selected pose changes colour along with the ghosts.** This
        recolours the molecule the user was already looking at, which is the sort
        of thing that reads as a bug until you know what question is being asked.
        Turning the overlay on changes the question from *what is this molecule*
        to *which of the nine is this* -- and the element table answers the first
        question and is close to useless for the second. A docked ligand is
        mostly grey carbon, so nine poses in element colours are nine grey
        molecules, and grey carbon at 0.30 opacity sitting on grey carbon at 1.0
        is not a distinction the eye makes.

        So both ends of the comparison go flat: eight ghosts in one slate, the
        selected pose in the green its own table row and the status bar already
        call it. That is a difference of *kind* rather than of strength, and it
        is the one the eye makes before it reads a single number. The green is
        not a new colour invented for this: it is `COLOR_BEST_POSE`, the same
        one the `*` in the pose table marks its best row with, so the picture and
        the words name the same pose.

        With the overlay off, `comparing` is False and the pose goes back to
        element colours, because then the question is what the molecule is
        again. This is a state of the comparison and not a repaint of the
        molecule: the same view object is drawn in five atom colours with the box
        unticked and in one flat green with it ticked. `_draw_colours_for` is
        where that decision is made; the state bar reports both.
        """
        self._pose_ghosts_on = bool(checked)
        self._sync_ghosts_in_scene()
        # The site cloud steps aside -- all the way, not part way. It is bigger
        # than the poses, it sits in the same place, and its footprint measured
        # on a real run is nine times the selected pose's, so the one thing the
        # user turned the overlay on to look at was the smallest thing in the
        # picture. See `POCKET_OPACITY_COMPARE` for the measurement, and for why
        # fading it was tried first and did not work. The user asked to compare
        # poses; the pocket is still in the scene, it is just no longer answering
        # first, and it comes back at its own strength with the overlay off.
        # The value itself is `_sync_pocket_opacity`'s to decide, from the
        # overlay flag and whether a site is selected, rather than assigned here.
        self._sync_pocket_opacity()
        # The selected pose wears its identity green whether or not this is on.
        # It used to be only while comparing, and the measurement that changed
        # that is in `_draw_colours_for`: over the pose's own 23 092 pixels its
        # chroma median is 8 against the receptor's 7, which is the receptor's
        # own 60th percentile -- the pose sat *inside* the protein's colour
        # distribution, so there was nothing for the eye to find it by. This
        # assignment now records the comparison state for the label and the
        # ghosts, and no longer changes a colour.
        self.viewport.pose_compare = bool(checked)
        if checked:
            # And go and look at them. The overlay answers "are these nine
            # answers or one answer nine times", and in the default framing the
            # poses are twenty atoms inside a four-hundred-atom protein and a
            # site cloud, so the answer is a smudge in the corner. Framing the
            # poses is what makes the difference visible, and it is the same
            # courtesy the site table extends when a click moves the box.
            self._frame_poses()
        self._update_poses_label()
        self.viewport.update()

    def _sync_pocket_opacity(self) -> None:  # pragma: no cover - GUI
        """The one place that decides how present the site cloud is.

        Three handlers used to assign `viewport.pocket_opacity` each -- the
        compare toggle, the site table and the pose selection -- and nothing said
        which of them won. It was "the last one to run", and that is not a rule
        a picture measurement can rely on: which of the three ran last depends on
        which checks a suite happened to execute, so the same window rendered
        two different pictures run to run.

        Measured, and the symptom was a representation check that read 0.1%,
        3.9%, 23.0% and 2.7% on four consecutive runs and a stable 18.0% in
        isolation. The cause is this attribute: with the cloud at
        `POCKET_OPACITY_PLAIN` the translucent magenta shifts pixels across the
        suite's absolute mask threshold, and it does so by *different* amounts
        in the two sphere modes -- the space-filling surface measures 791 824 px
        with the cloud on against 837 500 with it off, so the "surface" ends up
        covering less than the separated spheres and the direction of the
        comparison inverts. One owner, derived from state, is the fix; see
        `scripts/workbench_interaction_check.py` section 10 for the other half.

        The rule, in one sentence: the cloud leaves whenever the user is looking
        at a *pose* -- comparing poses, or inspecting one -- and comes back when
        they are looking at a *site*.
        """
        hide = self._pose_ghosts_on or (
            self._current_pose >= 0 and not self._pocket_selected
        )
        self.viewport.pocket_opacity = (
            POCKET_OPACITY_COMPARE if hide else POCKET_OPACITY_PLAIN
        )

    def _frame_poses(self) -> None:  # pragma: no cover - GUI
        """Point the camera at the poses rather than at the whole scene."""
        chunks = [
            np.asarray(m.coords, np.float32)
            for m in self.viewport.molecules
            if m.role in ("pose", "pose_ghost") and len(m.coords)
        ]
        if not chunks:
            return
        points = np.concatenate(chunks, axis=0)
        centre = points.mean(axis=0)
        self.viewport.focus_point(centre, float(np.abs(points - centre).max()) * 2.3)

    def _frame_selection(self) -> None:  # pragma: no cover - GUI
        """Frame the selected pose and the receptor atoms it is in contact with.

        Not the same gesture as `_frame_poses`, and the difference is the point.
        `_frame_poses` is the "compare nine answers at once" view: it frames all
        of them together, because the question there is whether the nine are
        nine answers or one answer nine times. This is the "look at *this* one"
        view, and what makes it answerable is that the contacting receptor atoms
        are in the selection too -- the overlays are lines drawn between the two
        sets, so a framing that kept the pose and dropped the partners would
        leave the lines pointing off the edge of the picture.

        The search box and the site cloud are deliberately not in it. Both belong
        to the search rather than to this pose -- every reported pose came out of
        that cube and sits in that cloud -- and framing either of them is the
        gesture the site table already has. See `framing_selection` for why, and
        for the measurements.
        """
        pose = next((m for m in self.viewport.molecules if m.role == "pose"), None)
        receptor = next(
            (m for m in self.viewport.molecules if m.role == "receptor"), None
        )
        if pose is None or not len(pose.coords):
            return
        # The site cloud steps aside, because `_sync_pocket_opacity` is what
        # decides it now and a pose is selected. The reason is in
        # `_draw_pocket` and in `POCKET_OPACITY_COMPARE`: the cloud is depth-off
        # and nine times the pose's own footprint, so a framing that puts the
        # pose at 2.6% of the frame underneath a cloud that owns ten times that
        # is a framing of the cloud. See the measurement in
        # `scripts/framing_selection_screens.py`, which writes the picture.
        self._sync_pocket_opacity()
        pose_coords = np.asarray(pose.coords, np.float32)
        partners: list[np.ndarray] = []
        residues: list[str] = []
        if receptor is not None and self.viewport.contacts:
            wanted = sorted({
                c.partner_index for c in self.viewport.contacts
                if 0 <= c.partner_index < len(receptor.coords)
            })
            if wanted:
                partners.append(np.asarray(receptor.coords, np.float32)[wanted])
                residues = sorted({
                    c.partner_residue for c in self.viewport.contacts
                    if 0 <= c.partner_index < len(receptor.coords)
                })
        partner_coords = (
            np.concatenate(partners, axis=0) if partners else np.zeros((0, 3), np.float32)
        )
        self._last_selection_plan = self.viewport.focus_selection(
            pose_coords, partner_coords, partner_residues=tuple(residues)
        )

    def _on_box_visibility(self) -> None:  # pragma: no cover - GUI
        self.viewport.show_box = self.cb_box.isChecked()
        self.viewport.update()

    def _on_contact_visibility(self) -> None:  # pragma: no cover - GUI
        self.viewport.show_contacts = self.cb_contacts.isChecked()
        self.viewport.update()

    def _refresh_contacts(self) -> None:  # pragma: no cover - GUI
        """Recompute the pose/receptor interface and refill the table.

        Called whenever either side changes, so the numbers in the table always
        belong to the pose on screen. An empty table is only ever shown together
        with a reason: "no receptor" and "nothing touches" are different facts
        and the panel says which one it is.
        """
        from .contacts import find_contacts, residue_summary

        pose = next((m for m in self.viewport.molecules if m.role == "pose"), None)
        receptor = next(
            (m for m in self.viewport.molecules if m.role == "receptor"), None
        )

        self.contact_table.setRowCount(0)
        if pose is None or receptor is None:
            self.viewport.contacts = []
            self.viewport.contacts_valid = False
            self.lbl_contacts.setText("load a receptor and a pose")
            return

        found = find_contacts(pose, receptor)
        self.viewport.contacts = found
        self.viewport.contacts_valid = True
        summary = residue_summary(found)

        kinds: dict[str, int] = {}
        for c in found:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        parts = [f"{len(found)} contacts over {len(summary)} residues"]
        for kind in ("hbond", "polar", "hydrophobic", "close"):
            if kinds.get(kind):
                parts.append(f"{kinds[kind]} {kind}")
        if not found:
            parts.append("nothing within 4.0 A")
        self.lbl_contacts.setText("  |  ".join(parts))

        self.contact_table.setRowCount(len(summary))
        from . import COLOR_CONTACT

        for row, (residue, hbonds, total, closest) in enumerate(summary):
            # The residue cell carries the colour of its strongest contact, so
            # the table and the picture agree without a second lookup: the row
            # that is tinted for a hydrogen bond is the row whose yellow-white
            # dashes run across the middle of the model.
            members = [c for c in found if c.partner_residue == residue]
            top_kind = min(
                (c.kind for c in members),
                key=lambda k: ("hbond", "polar", "hydrophobic", "close").index(k),
            )
            r, g, b = (int(round(v * 255)) for v in COLOR_CONTACT[top_kind])
            tint = f"color: rgb({r},{g},{b});"
            for col, text in enumerate(
                [residue, str(hbonds), str(total), f"{closest:.2f}"]
            ):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, residue)
                    item.setForeground(QtGui.QBrush(QtGui.QColor(r, g, b)))
                    item.setToolTip(f"{top_kind} contact")
                else:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(0xC8, 0xCE, 0xD6)))
                self.contact_table.setItem(row, col, item)
        self.viewport.update()

    def _on_contact_selected(self) -> None:  # pragma: no cover - GUI
        rows = self.contact_table.selectionModel().selectedRows() if self.contact_table.selectionModel() else []
        if not rows:
            return
        item = self.contact_table.item(rows[0].row(), 0)
        if item is None:
            return
        residue = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.viewport.focus_residue(residue)
        self.statusBar().showMessage(f"centred on {residue}", 4000)

    def _on_representation(self) -> None:  # pragma: no cover - GUI
        """Switch how the scene is drawn, and say what that means for the bonds.

        The status line is the point of this handler as much as the redraw. A
        ball-and-stick picture makes a wrong bond as convincing as a right one,
        so the bar always reports where the connectivity came from -- read out
        of the file, out of the residue templates, or inferred from distances --
        and names any file whose bonds could not be taken at face value.
        """
        key = self.cmb_representation.currentData()
        if key not in REPRESENTATION_KEYS:
            return
        self.viewport.representation = key
        self.viewport.update()
        self._describe_representation()

    def _describe_representation(self) -> None:  # pragma: no cover - GUI
        parts: list[str] = []
        for mol in self.viewport.molecules:
            if not mol.visible:
                continue
            n = len(mol.bond_pairs())
            where = {
                "declared": "read from the file",
                "template": "residue templates",
                "distance": "inferred from distances",
            }.get(mol.bond_source, mol.bond_source)
            note = ""
            if self.viewport.representation in ("ribbon", "cartoon") and not mol.has_backbone:
                note = " (no backbone, drawn ball-and-stick)"
            parts.append(
                f"{mol.name}: {n} bonds from {where}{note}"
            )
            for w in mol.warnings[:2]:
                parts.append(f"  ! {w}")
        if not parts:
            return
        self.statusBar().showMessage("  |  ".join(parts))

    def find_pockets(self) -> list:  # pragma: no cover - GUI
        """Search the receptor for candidate sites, off the GUI thread.

        Returns the previous result, or an empty list, and fills in the table
        when the answer arrives. Callers that need the sites *now* -- which
        only means the load path selecting the first row -- use
        `find_pockets_now` instead.

        Nothing about the answer changes by being computed on a worker thread;
        what changes is that a twelve-second search no longer takes the window
        with it.
        """
        receptor = next((m for m in self.viewport.molecules if m.role == "receptor"), None)
        if receptor is None:
            self._fill_pockets([], "load a receptor first")
            return []
        if self._pocket_thread is not None and self._pocket_thread.isRunning():
            # A second search while one is in flight: drop it rather than queue.
            # The user asked twice within a few seconds, which means they want
            # the newest answer, and the older one is already out of date.
            return self._pocket_models
        self.lbl_pockets.setText("searching…")
        self.btn_pockets.setEnabled(False)
        self._pocket_thread = QtCore.QThread(self)
        self._pocket_worker = PocketWorker(
            np.asarray(receptor.coords, np.float32),
            list(receptor.elements),
            receptor.residue_labels(),
        )
        self._pocket_worker.moveToThread(self._pocket_thread)
        self._pocket_thread.started.connect(self._pocket_worker.run)
        self._pocket_worker.finished.connect(self._on_pockets_finished)
        self._pocket_worker.failed.connect(self._on_pockets_failed)
        self._pocket_worker.finished.connect(self._pocket_thread.quit)
        self._pocket_worker.failed.connect(self._pocket_thread.quit)
        self._pocket_thread.start()
        return self._pocket_models

    def _on_pockets_finished(self, found, elapsed_ms) -> None:  # pragma: no cover - GUI
        self.btn_pockets.setEnabled(True)
        self._fill_pockets(
            found,
            f"{len(found)} site(s) in {elapsed_ms:.0f} ms" if found
            else "no enclosed site found — place the box by hand",
        )
        if found and not self.pocket_table.selectionModel().selectedRows():
            # Only auto-select when nothing is selected. A search the user
            # asked for while reading the list should not move the box out from
            # under them, which is exactly what load-time selection would do.
            # And it is a default, not a click, so it does not claim to be one:
            # see `_auto_select_pocket_row`.
            self._auto_select_pocket_row(0)

    def _auto_select_pocket_row(self, row: int) -> None:  # pragma: no cover - GUI
        """Select a site row as a *default*, which is not the user choosing it.

        Both auto-selects -- the one on load and the one after a search the user
        asked for -- go through here, because they fire the same
        `itemSelectionChanged` signal a click does and there is no way to tell
        the two apart from the signal alone.

        The difference matters now that `_sync_pocket_opacity` reads
        `_pocket_selected`: a default site selection would otherwise count as
        "the user is looking at a site" for the rest of the session, keep the
        site cloud at full strength over a pose the user had selected, and
        silently undo the one thing a pose selection does to the picture. It
        did exactly that when this rule first went in as two assignments at the
        two call sites: with a pose selected the cloud still measured 0.34.

        A default selection still *shows* -- the cloud is drawn whenever no pose
        is selected, which is the receptor-only case this exists for -- it just
        does not claim to be a gesture.

        The highlight is then cleared, with signals blocked. Clearing it is not
        cosmetic: leaving row 0 highlighted while `_pocket_selected` is False
        put the table and the cloud in states that disagreed, and it created a
        state the user could not leave. `itemSelectionChanged` does not fire when
        the row that is already current is clicked, so a user clicking the
        highlighted row to say "yes, this site" produced no event at all and the
        cloud stayed away. The first version of this rule kept the highlight and
        set a flag instead, and the suite caught exactly that: a check that
        selected a site row and expected the cloud back read 0.0.
        """
        self.pocket_table.selectRow(int(row))
        self.pocket_table.blockSignals(True)
        try:
            self.pocket_table.clearSelection()
        finally:
            self.pocket_table.blockSignals(False)
        self._pocket_selected = False
        self._sync_pocket_opacity()

    def _on_pockets_failed(self, message: str) -> None:  # pragma: no cover - GUI
        self.btn_pockets.setEnabled(True)
        self._fill_pockets([], f"pocket search failed: {message}")

    def find_pockets_now(self) -> list:  # pragma: no cover - GUI
        """Synchronous search, for the load path and for headless checks.

        Blocking is only acceptable where the caller can afford it: loading a
        receptor, where the alternative is a box nobody chose, and a test that
        wants the answer before it asserts on it. The button and any repeat
        search go through `find_pockets` and do not block.
        """
        receptor = next((m for m in self.viewport.molecules if m.role == "receptor"), None)
        if receptor is None:
            self._fill_pockets([], "load a receptor first")
            return []
        from . import pockets as P

        started = time.perf_counter()
        try:
            found = P.find_pockets(
                receptor.coords, receptor.elements, residues=receptor.residue_labels()
            )
        except Exception as exc:
            self._fill_pockets([], f"pocket search failed: {exc}")
            return []
        elapsed = (time.perf_counter() - started) * 1000.0
        self._fill_pockets(
            found,
            f"{len(found)} site(s) in {elapsed:.0f} ms" if found
            else "no enclosed site found — place the box by hand",
        )
        return found

    def _fill_pockets(self, found, summary: str) -> None:  # pragma: no cover - GUI
        self._pocket_models = list(found)
        self.lbl_pockets.setText(summary)
        # No selection means no site, and no site means nothing to draw. The
        # cloud is cleared rather than left showing the previously selected
        # site's points, which would put one site's volume inside another's
        # box and look like a mistake in the search rather than in the view.
        if not self.pocket_table.selectionModel() or not self.pocket_table.selectionModel().selectedRows():
            self.viewport.pocket_points = np.zeros((0, 3), np.float32)
        self.pocket_table.blockSignals(True)
        self.pocket_table.setRowCount(len(found))
        for row, p in enumerate(found):
            from . import pockets as P

            lining = ", ".join(name for name, _ in p.lining[:4]) or "—"
            cells = (
                str(row + 1),
                P.kind_label(p.kind),
                f"({p.center[0]:.1f}, {p.center[1]:.1f}, {p.center[2]:.1f})",
                f"{p.size[0]:.0f} × {p.size[1]:.0f} × {p.size[2]:.0f}",
                f"{p.volume:.0f} Å³",
            )
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 0:
                    # The lining is too long for a column and too useful to
                    # drop, so it lives in the tooltip and the status bar.
                    item.setToolTip(
                        f"site {row + 1}: {p.voxels} grid points, "
                        f"buried on {p.burial:.1f} of 3 axes\nlined by {lining}"
                    )
                self.pocket_table.setItem(row, col, item)
        self.pocket_table.blockSignals(False)

    def _on_pocket_visibility(self, checked: bool) -> None:  # pragma: no cover - GUI
        self.viewport.show_pocket = bool(checked)
        self.viewport.update()

    def _on_pocket_selected(self) -> None:  # pragma: no cover - GUI
        """Put the box on the selected site and look at it.

        Also the single place the site volume is cleared. That reset used to
        live in `_fill_pockets`, which only runs when a *new search* fills the
        table -- so clearing the selection by hand left the previous site's
        cloud on screen, inside whatever the new selection put there, showing a
        site the table no longer listed.

        The spins are updated rather than bypassed, so the numbers on screen
        and the numbers the engine gets cannot disagree -- an earlier design
        that moved `box_center` directly would have left three stale spin
        boxes describing a box that no longer existed.
        """
        model = self.pocket_table.selectionModel()
        rows = model.selectedRows() if model else []
        index = rows[0].row() if rows else -1
        models = getattr(self, "_pocket_models", None)
        if not models or not 0 <= index < len(models):
            self.viewport.pocket_points = np.zeros((0, 3), np.float32)
            # Clearing the selection means no site is being looked at, which is
            # a state the cloud's opacity depends on. It used to be invisible
            # here: `pocket_opacity` was only ever written by the two handlers
            # that *chose* something, so clearing left the last choice standing
            # with nothing on screen to justify it.
            self._pocket_selected = False
            self._sync_pocket_opacity()
            self.viewport.update()
            return
        pocket = models[index]
        if not 0 <= index < len(self._pocket_models):
            return
        pocket = self._pocket_models[index]
        from . import pockets as P

        centre, size = pocket.box_center_and_size()
        for spin, value in zip(self.center_spins, centre):
            spin.setValue(float(value))
        for spin, value in zip(self.size_spins, size):
            spin.setValue(float(value))
        self._on_box_changed()
        self.viewport.pocket_points = np.asarray(
            getattr(pocket, "points", np.zeros((0, 3), np.float32)), np.float32
        ).reshape(-1, 3)
        # The cloud comes back at its own strength, because this is the gesture
        # that answers "show me the site" and a site you cannot see is not one.
        # `_sync_pocket_opacity` owns the decision; all this does is record that
        # a site is the thing being looked at.
        self._pocket_selected = True
        self._sync_pocket_opacity()
        # Frame the *box*, not the site, and by its diagonal rather than its
        # longest side. The site is 7 A across and the box 15 A; framing the
        # site put the camera 18 A out, which is inside the protein's own
        # surface atoms, and the picture became a close-up of a few carbons
        # with the box edges off screen. Framing on the longest side still cut
        # the box off, because what has to fit on screen is the diagonal.
        self.viewport.focus_point(centre, float(np.linalg.norm(size)))
        names = ", ".join(name for name, _ in pocket.lining[:6])
        self.statusBar().showMessage(
            f"site {index + 1} ({P.kind_label(pocket.kind)}) at "
            f"({pocket.center[0]:.1f}, {pocket.center[1]:.1f}, {pocket.center[2]:.1f})"
            + (f" — lined by {names}" if names else ""),
            8000,
        )

    def _on_box_changed(self) -> None:  # pragma: no cover - GUI
        self.viewport.box_center = np.asarray(
            [s.value() for s in self.center_spins], np.float32
        )
        self.viewport.box_size = np.asarray([s.value() for s in self.size_spins], np.float32)
        self._suggest_exhaustiveness()
        self._refresh_effort_note()
        self.viewport.update()

    def _on_exhaust_edited(self, _value: int) -> None:  # pragma: no cover - GUI
        """The user took the exhaustiveness over; stop suggesting."""
        if getattr(self, "_updating_exhaust", False):
            return
        self._exhaust_is_default = False
        self._describe_exhaustiveness(False)
        self._refresh_effort_note()

    def _use_suggested_exhaustiveness(self) -> None:  # pragma: no cover - GUI
        """Re-arm the suggestion, because the user asked for it and not otherwise.

        The latch has to be clearable. Set-only was not a safety property, it was
        a dead end: one misclick on the spin box and the box stopped driving the
        search effort for the rest of the session, which is the same
        under-sampled large box the suggestion was added to prevent.

        Deliberately not automatic. A box change that quietly re-took the number
        would undo a decision the user had made and visible evidence of it, and
        the user could not get it back without pressing this button again -- so
        it would be strictly worse than a control that never existed.

        The button does not choose a value. It hands the number to the rule,
        which then does what it would have done for a box the user never
        touched. That way the value on screen is only ever the rule's answer or
        the user's own, never a third thing nobody chose.
        """
        self._exhaust_is_default = True
        self._suggest_exhaustiveness()
        self._refresh_effort_note()

    def _describe_exhaustiveness(self, suggested: bool) -> None:  # pragma: no cover - GUI
        """Say in the spin box's tooltip whose number this is.

        One place, because the two callers are the two ends of the latch and
        they have to agree: a re-armed box whose tooltip still said "set by you"
        would be a control describing a state it is not in, which is the same
        class of wrong as the missing one.
        """
        if not suggested:
            self.sp_exhaust.setToolTip(
                "Independent search walks. Set by you, so it is yours: the "
                "window will not change it back when the box changes. "
                "'use suggested value' hands it back to the box."
            )
            return
        size = tuple(float(s.value()) for s in self.size_spins)
        self.sp_exhaust.setToolTip(
            f"suggested for a {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} A "
            f"box. Walks are spread through the box, so a bigger box needs "
            f"more of them; the same number that finds a pose in a 20 A box "
            f"can miss it in a 40 A one.\n\n"
            f"The ladder is discrete (8, 16, 32, 64, 128), so it steps rather "
            f"than tracks: 64 becomes 128 at 64,000 A^3, which for a cube is "
            f"40.0 A against 40.1 A. A tenth of an angstrom doubles the search."
        )

    def _map_sizer(self):
        """A `Receptor` handle for the map-size estimate, built once per file.

        `estimate_memory_mb` is a method on `Receptor`, but the number it
        returns depends on the box alone: measured identical for a 30-atom and
        a 46-atom receptor, 0.2 us a call, against 0.4 ms (30 atoms) to 240 ms
        (5000 atoms) for the parse that produces the handle. So the handle is
        built when the receptor is loaded -- on a path that already blocks for
        the pocket search -- and reused for every later box change. Building one
        per change would put a quarter of a second into every drag of a size
        spin box.

        None means "cannot answer": no receptor, or one the engine will not
        parse. The note then says so instead of showing a number nobody measured.
        """
        path = self._receptor_path
        if path is None:
            return None
        if self._sizer is not None and self._sizer_path == path:
            return self._sizer
        from ..core import Receptor

        try:
            self._sizer = Receptor.from_pdbqt(path)
        except Exception:
            self._sizer = None
            self._sizer_path = None
            return None
        self._sizer_path = path
        return self._sizer

    def _map_memory_mb(self, size) -> float | None:  # pragma: no cover - GUI
        """The engine's own estimate of the grid memory `size` will need, in MB."""
        receptor = self._map_sizer()
        if receptor is None:
            return None
        from ..core import GridBox

        try:
            box_ = GridBox.from_center_size(
                tuple(float(s.value()) for s in self.center_spins), tuple(size)
            )
            return float(receptor.estimate_memory_mb(box_))
        except Exception:
            return None

    def _refresh_effort_note(self) -> None:  # pragma: no cover - GUI
        """One line under the spin box: what the maps cost, and whether 128 is enough.

        The memory half is `odcli rec-grid`'s number, which the workbench did
        not have at all. It is worth having before the Dock button is pressed:
        a 100 A cube needs 2.9 GB of grids at the default 0.375 A spacing, and
        the alternative to knowing that is finding out.

        The second half is the honesty half. Above 64,000 A^3 the top rung is
        not what the density rule wants -- 216 walks for a 60 A cube against the
        128 available, 1000 against 128 for 100 A -- and the spin box was
        showing 128 as if it were the answer. It is the best number the ladder
        holds, not a met one, and the box sizes where it bites are the ones most
        likely to miss a pose in the first place. So the shortfall is printed
        rather than rounded away.
        """
        size = tuple(float(s.value()) for s in self.size_spins)
        mb = self._map_memory_mb(size)
        if mb is None:
            self.lbl_effort.setText("maps: load a receptor to size them")
            self._fit_effort_note()
            return
        walks = int(self.sp_exhaust.value())
        owner = (
            "suggested for this box"
            if getattr(self, "_exhaust_is_default", True)
            else "set by you"
        )
        text = f"maps {mb:.0f} MB | {walks} walks, {owner}"
        want, ceiling = _walks_wanted(size)
        if want > walks * 1.02:
            volume = float(np.prod(np.asarray(size, np.float64)))
            tail = (
                f"{100.0 * walks / want:.0f}% of the {want:.0f} walks a "
                f"{volume:.0f} A^3 box wants"
            )
            if walks >= ceiling:
                # A cap is a property of the ladder and has to be named as one,
                # or 128 reads as "enough" in the exact size range where it is
                # not. The cap is the engine's own ceiling for this box, not a
                # second reading of the ladder here.
                text += f"\nthe ladder stops here: {tail}"
            else:
                text += f"\nbelow what the box wants: {tail}"
        self.lbl_effort.setText(text)
        self._fit_effort_note()

    def _fit_effort_note(self) -> None:  # pragma: no cover - GUI
        """Give the note room for the line it just grew.

        Word wrap only wraps if something asks for the height. The panel sits in
        a scroll area that resizes its widget, so a two-line note was laid out
        one line tall and the second was cut off -- the cap warning, which is the
        only part of the text that changes, was the part that disappeared.
        """
        label = self.lbl_effort
        width = label.width()
        if width <= 1:
            width = label.sizeHint().width()
        height = label.heightForWidth(max(width, 1))
        if height > 0:
            label.setMinimumHeight(height)

    def _suggest_exhaustiveness(self) -> None:  # pragma: no cover - GUI
        """Keep the search effort in step with the box, until the user says otherwise.

        The only caller is `_on_box_changed`, which is the single place the box
        changes -- whether a pocket row put it there or a spin box did -- so
        there is one path by which the suggestion can be made and no second
        place to forget.

        Once the user touches the spin box the suggestion stops, until they ask
        for it back with 'use suggested value'. A control that keeps
        overwriting what you typed is worse than a control that never helped:
        you learn to distrust the one number you cannot see the effect of until
        after the run. A control that cannot be taken back is worse still,
        because the way to recover is a click you do not know exists.
        """
        from ..core import exhaustiveness_for_box

        if not getattr(self, "_exhaust_is_default", True):
            return
        size = tuple(float(s.value()) for s in self.size_spins)
        want = exhaustiveness_for_box(size)
        # Guarded, because `setValue` emits `valueChanged` and that is the very
        # signal used to decide the user has taken over. Without the guard the
        # suggestion marks itself as a user edit on its first application and
        # never applies again -- which looks exactly like the feature never
        # having been written.
        if int(self.sp_exhaust.value()) != want:
            self._updating_exhaust = True
            try:
                self.sp_exhaust.setValue(want)
            finally:
                self._updating_exhaust = False
        # Set even when the value already agrees, so a box whose suggestion was
        # handed back is labelled as ours rather than left claiming it was the
        # user's.
        self._describe_exhaustiveness(True)

    def _on_pose_selected(self, row: int, _col: int = 0) -> None:  # pragma: no cover - GUI
        """Show the pose in ``row``: picture, both readouts, contacts, ghosts.

        ``row`` is the *visual* row, so the pose it means is read out of the
        cell rather than assumed -- sorting is on, and a handler that treated
        the row number as the pose index would show pose 4 when the user clicked
        the row that says 7.
        """
        if row < 0 or not self._pose_models or self._pose_view is None:
            return
        index = self._pose_index_at(row)
        if index is None or not 0 <= index < len(self._pose_models):
            return
        self._current_pose = index
        text = self._pose_models[index][0]
        new_view = self._view_from_text(
            text, self._pose_view.name, self._pose_view.color, self._pose_view.radius
        )
        # Carry the role across. `_view_from_text` builds a fresh view, which
        # defaults to role "ligand"; losing it here made the pose stop being
        # recognisable as a pose, so the next docking run could not tell it from
        # a ligand view and stacked another result on top of it.
        new_view.role = self._pose_view.role
        new_view.visible = self._pose_view.visible
        idx = self.viewport.molecules.index(self._pose_view)
        self.viewport.molecules[idx] = new_view
        self._pose_view = new_view
        energy = self._pose_energy(index)
        self.lbl_energy.setText(
            f"{energy:.2f} kcal/mol" if energy is not None else "energy not in file"
        )
        # The label and the table read the same number, from the same call. They
        # used to read two different ones whenever a run was in memory, because
        # the label took the file's one-decimal energy and the table the
        # engine's full-precision one.
        self.lbl_rmsd.setText(
            "—" if self._pose_rmsd(index) is None else f"{self._pose_rmsd(index):.2f} Å"
        )
        # The breakdown is asked for the *same* index the label and the table
        # above were given, passed in rather than read back off `self`, so that
        # "the panel is describing the selected pose" is a fact about this call
        # and not about a field that could have been written by something else.
        self._refresh_energy_terms(index)
        # The verdict, for the same index and in the same breath. The index is
        # passed down rather than read back off `self` for the reason the
        # breakdown's is: a handler that refreshed the panel from a field
        # something else had written would put one pose's verdict under
        # another's name.
        self._refresh_pose_verdict(index)
        self._apply_pose_visibility()
        self._update_poses_label()
        self._refresh_contacts()
        # After the contacts, because the selection is the pose *and what it is
        # touching*: framing first would aim the camera at a pose and nothing
        # else, and the partner atoms would then be outside the picture the
        # overlays are drawn into.
        self._frame_selection()
        self.viewport.update()

    def _refresh_energy_terms(self, index: int) -> None:  # pragma: no cover - GUI
        """Show pose ``index``'s term decomposition, or say why there isn't one.

        ``index`` is passed in rather than read back off ``self`` so the panel
        and the two labels above it are provably describing the same pose. A
        handler that refreshed the panel from a field something else had
        written would put the previous pose's terms under the new pose's name,
        which is the one outcome this panel exists to make impossible.

        Nothing here recomputes a term. The five numbers come out of the
        engine's own `score_conformation_terms`, already weighted, and the rows
        underneath them are the engine's own consistency pair so a reader can
        see that the decomposition adds up rather than take it on trust.
        """
        from ..core import GridBox
        from .energy_terms import (
            TERM_ROWS,
            TermEvaluator,
            TermsUnavailable,
            format_kcal,
        )

        table = self.tbl_terms
        table.setRowCount(0)
        self._terms_pose_index = -1
        self._terms_breakdown = None
        self.btn_terms_dock.setVisible(False)
        if index < 0 or not self._pose_models:
            self._terms_retire()
            self.lbl_terms_pose.setText("no pose selected")
            self.lbl_terms_note.setText(
                "Select a pose to see which terms produced its energy."
            )
            return

        count = len(self._pose_models)
        self.lbl_terms_pose.setText(f"pose {index + 1} of {count}")
        if self._receptor_path is None or self._ligand_path is None:
            self._terms_retire()
            self.lbl_terms_note.setText(
                "No receptor or ligand path, so there is nothing to tabulate "
                "the maps from."
            )
            return
        centre = tuple(float(s.value()) for s in self.center_spins)
        size = tuple(float(s.value()) for s in self.size_spins)
        scoring = self.cb_scoring.currentText()
        # The maps depend on the receptor, the box, the ligand and the scoring
        # function and on nothing else, so they are built once per combination of
        # those. Moving a spin box gives a new key and a new evaluator;
        # selecting another pose reuses it.
        key = (str(self._receptor_path), str(self._ligand_path),
               centre, size, scoring)
        if self._terms_evaluator_key != key:
            self._terms_evaluator = TermEvaluator(
                self._receptor_path,
                GridBox.from_center_size(centre, size),
                self._ligand_path,
                scoring,
            )
            self._terms_evaluator_key = key
        if self._terms_evaluator.built:
            try:
                breakdown = self._terms_evaluator.breakdown(
                    self._dock_result, index
                )
            except TermsUnavailable as exc:
                self._terms_unavailable(exc, count)
                return
            self._terms_fill(breakdown, count)
            return
        # Not built yet. The maps depend on the receptor, the box, the ligand and
        # the scoring function and on nothing else, so this is paid once per
        # combination of those and then cached. It is the expensive part -- see
        # `TermsWorker` for the measurement -- so it runs off this thread, and the
        # panel says so, because four silent seconds and a hang look identical
        # from the outside.
        #
        # The two numbers in that sentence were measured here, not inherited.
        # 22 A is a 59^3 grid and 40 A a 107^3 grid, at the default 0.375 A
        # spacing, on `examples/1crn_prep.pdbqt`; three runs each, taking the
        # first `TermEvaluator.breakdown` (build plus score plus the ligand
        # load) on an otherwise idle process:
        #
        #     22 A   0.808 / 0.755 / 1.164 s   -> "about 0.8 s"
        #     40 A   4.009 / 5.006 / 5.020 s   -> "about 5 s"
        #
        # The 40 A figure was "4 s" here until this revision, and it was
        # understated: the spread above reaches 5.0 s, and 4 s describes none
        # of the three runs. The 22 A figure was already right and is unchanged.
        #
        # What the sentence does *not* claim is that the window stayed
        # comfortable. That the build is not on the GUI thread is a fact about
        # the four lines below; whether a 5 s tabulation produces a visible
        # stutter on a given machine has not been measured for this panel --
        # section 11d of `scripts/workbench_interaction_check.py` measures it
        # for the *pocket search*, which is a different thread and a different
        # duration. So the wording names where the work happens and stops
        # there. The earlier "the window stays responsive while it happens" was
        # a claim about a user-visible effect that nothing had measured.
        self._terms_retire()
        self.lbl_terms_pose.setText(f"pose {index + 1} of {count}")
        self.lbl_terms_note.setText(
            "tabulating the term maps for this box -- about 0.8 s at 22 A "
            "and 5 s at 40 A, measured on 1crn at the default 0.375 A "
            "spacing. The build runs on its own QThread, so the GUI thread "
            "is not the one doing it"
        )
        self._terms_thread = QtCore.QThread(self)
        self._terms_worker = TermsWorker(
            self._terms_evaluator, self._dock_result, index
        )
        self._terms_worker.moveToThread(self._terms_thread)
        self._terms_thread.started.connect(self._terms_worker.run)
        self._terms_worker.ready.connect(self._on_terms_ready)
        self._terms_worker.failed.connect(self._on_terms_failed)
        # The window keeps the thread object, and drops it only when the thread
        # says it is done. Holding the reference until then is the other half of
        # not blocking: a retired build keeps answering, and a *running* one is
        # never in a state where something drops its last reference.
        self._terms_thread.finished.connect(self._terms_reaped)
        self._terms_thread.start()

    def _terms_retire(self) -> None:
        """Ask a running build to stop, without blocking.

        `wait()` from inside a signal handler is a re-entrant block on the event
        loop, and it aborted the process: the first version waited in
        `_on_terms_ready`, which the dock's own row selection reaches, and the
        run died with 0xC0000409 before printing a summary. So a build is only
        ever *asked* to stop here, and the thread object is parented to the
        window and released when it reports `finished` -- which means it is never
        destroyed while running, the other way to get the same abort.
        """
        thread = self._terms_thread
        if thread is not None and thread.isRunning():
            thread.quit()

    def _terms_reaped(self) -> None:  # pragma: no cover - GUI
        """Release a thread object, but only the one that actually reported.

        `finished` arrives queued, so a thread retired by a later selection
        can deliver its signal *after* that later selection has already stored
        its own thread. Dropping the reference on every `finished` would
        therefore clear a running thread's handle -- and `closeEvent` waits on
        this attribute, so the window would then stop waiting for a build that
        is still running, and the abort would come back at the next close.
        """
        sender = self.sender()
        if sender is not None and sender is not self._terms_thread:
            return
        self._terms_thread = None
        self._terms_worker = None

    def _on_terms_ready(self, breakdown) -> None:  # pragma: no cover - GUI
        """A build finished. Use it only if it is still the pose on show.

        The selection can move while four seconds of map tabulation are in
        flight, and filling the panel with the pose that was asked for would put
        one pose's terms under another's name -- the one thing this panel must
        never do. So a late answer for a pose that is no longer selected is
        dropped, and the newer selection's own request fills the panel.
        """
        self._terms_retire()
        if breakdown.pose_index != self._current_pose:
            return
        self._terms_fill(breakdown, len(self._pose_models))

    def _on_terms_failed(self, message: str) -> None:  # pragma: no cover - GUI
        from .energy_terms import TermsUnavailable  # noqa: PLC0415

        self._terms_retire()
        self._terms_unavailable(
            TermsUnavailable(message), len(self._pose_models)
        )

    def _terms_unavailable(self, exc, count: int) -> None:  # pragma: no cover
        """Say why there are no terms, and offer the way forward if there is one."""
        self.lbl_terms_note.setText(str(exc))
        # The one case with a way forward: no run in memory, so no
        # conformation, so no decomposition -- and a receptor and a ligand
        # loaded, so a run is one click away. The button says so instead of
        # leaving the reader to work it out.
        can_dock = (
            self._dock_result is None
            and self._receptor_path is not None
            and self._ligand_path is not None
        )
        self.btn_terms_dock.setVisible(can_dock)
        if can_dock:
            self.lbl_terms_note.setText(
                f"{exc}. The button below runs a docking with the loaded "
                f"receptor and ligand; the poses it reports carry the "
                f"conformations the decomposition needs, and this panel "
                f"will describe whichever one is selected afterwards."
            )

    def _terms_fill(self, breakdown, count: int) -> None:  # pragma: no cover
        """Put one breakdown on screen. The only writer of the table's rows."""
        # Imported here rather than reusing an import in the caller: this method
        # is also reached from a signal handler, where a name that only existed
        # in some other method's local scope is a `NameError` -- and an
        # exception escaping a PyQt6 slot aborts the process with 0xC0000409,
        # which is what the first threaded run did, silently, on every build.
        from .energy_terms import TERM_ROWS, format_kcal

        self._terms_pose_index = int(breakdown.pose_index)
        self._terms_breakdown = breakdown
        self.lbl_terms_pose.setText(
            f"pose {breakdown.pose_index + 1} of {count}, {breakdown.scoring} "
            f"-- total {format_kcal(breakdown.total)} kcal/mol"
        )
        notes = {name: note for name, _key, note in TERM_ROWS}
        keys = {name: key for name, key, _note in TERM_ROWS}
        rows = [(name, value) for name, _k, value, _n in breakdown.term_rows()]
        rows += [(label, value) for label, value, _src in breakdown.check_rows()]
        table = self.tbl_terms
        table.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            first = QtWidgets.QTableWidgetItem(label)
            tip = notes.get(label)
            if tip:
                first.setToolTip(f"{tip}. Engine key: {keys[label]!r}.")
            table.setItem(row, 0, first)
            second = QtWidgets.QTableWidgetItem(value)
            second.setTextAlignment(
                int(QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter)
            )
            table.setItem(row, 1, second)
        self.lbl_terms_note.setText(
            "The first sum is the sum of the values printed above it, so the "
            "column adds up by hand; the engine's own sum is on the next row "
            "and can differ in the last printed decimal, because the five terms "
            "are printed rounded. The terms are already weighted, exactly as "
            "score_conformation_terms returns them, so do not weight them "
            "again. A term too small for four decimals is printed in exponent "
            "form and counts as zero in that first sum."
        )
        # The breakdown carries this pose's out-of-box penalty, which is one of
        # the four numbers the verdict panel needs, so the verdict is asked
        # again now that it has arrived. Until it does, that one contract reads
        # `unmeasured` -- which is the truth: the penalty is a property of a
        # scored pose, and nothing has scored this one yet.
        self._refresh_pose_verdict(int(breakdown.pose_index))

    def _refresh_pose_verdict(self, index: int) -> None:  # pragma: no cover - GUI
        """Say whether pose ``index``'s own numbers can be trusted, or why not.

        Nothing here is a measurement. The gradient is read out of the array
        the run already holds, the tolerance is the optimiser's declared one,
        the out-of-box penalty is reused from the breakdown above *for this
        pose*, and the folding into one of three states is a pure function. So
        this panel adds no engine call and no thread: 500 of these on this
        machine took 77.8 us each, against 0.808 s for the first breakdown the
        panel above has to wait for.

        The refusal is a state, not a gap. A pose read from a file has no
        gradient and no penalty -- the engine keys both on the conformation of a
        run in memory, and there is no inverse from coordinates -- so the panel
        says `unknown` on all four contracts and names the action that would
        change it. It does not show the previous pose's verdict, which is the
        one thing a panel like this must never do.
        """
        from .pose_trust import (
            NO_POSE,
            NO_POSE_NOTE,
            PoseVerdictUnavailable,
            describe_file_pose,
            describe_pose,
            stylesheet_for,
            verdict_is_available,
        )

        self.tbl_verdict.setRowCount(0)
        self._verdict_pose_index = -1
        available, why = verdict_is_available()
        if not available:
            # The module is not importable, so nothing can be measured. This
            # is drawn in the `unknown` treatment rather than left blank: a
            # blank table beside a filled-in breakdown reads as "nothing to
            # say", which is the reading that turns a missing module into a
            # silent pass.
            self.lbl_verdict_head.setText("no verdict available")
            self.lbl_verdict_head.setStyleSheet(stylesheet_for("unknown"))
            self.lbl_verdict_why.setText(why)
            self.lbl_verdict_note.setText(
                "No contract below was checked, because there was nothing to "
                "check them with. This is not a pass."
            )
            return
        if index < 0 or not self._pose_models:
            self._verdict_clear(NO_POSE, NO_POSE_NOTE, "unknown")
            return
        count = len(self._pose_models)
        try:
            if self._dock_result is None:
                verdict = describe_file_pose(index, count)
            else:
                penalty = None
                penalty_index = None
                breakdown = self._terms_breakdown
                if (breakdown is not None
                        and int(breakdown.pose_index) == int(index)):
                    penalty = float(breakdown.out_of_box_penalty)
                    penalty_index = int(breakdown.pose_index)
                verdict = describe_pose(
                    self._dock_result, index, penalty, penalty_index
                )
        except PoseVerdictUnavailable as exc:
            self._verdict_clear(
                f"pose {index + 1} of {count}: no verdict", str(exc), "unknown"
            )
            return
        self._verdict_fill(verdict, count)

    def _verdict_clear(self, head: str, why: str, state: str) -> None:
        """An empty table with a stated reason, never an empty panel."""
        from .pose_trust import stylesheet_for

        self.lbl_verdict_head.setText(head)
        self.lbl_verdict_head.setStyleSheet(stylesheet_for(state))
        self.lbl_verdict_why.setText(why)
        self.lbl_verdict_note.setText(
            "Four contracts are checked and every one of them needs a number "
            "this result does not carry, so the verdict is 'unknown' -- which "
            "is not a pass. The action that would change it is "
            "'Re-dock this ligand here' in the panel above: the poses it "
            "reports carry the conformations and the run-time numbers the "
            "contracts read."
        )

    def _verdict_fill(self, verdict, count: int) -> None:  # pragma: no cover - GUI
        """Put one verdict on screen. The only writer of these rows.

        Every row comes out of the single `describe_pose` call above, so the
        table cannot hold two contracts measured at two different moments, and
        the header names the pose those rows are about.
        """
        from .pose_trust import (
            STATE_COLOURS,
            stylesheet_for,
        )

        self._verdict_pose_index = int(verdict.pose_index)
        self.lbl_verdict_head.setText(verdict.headline())
        self.lbl_verdict_head.setStyleSheet(stylesheet_for(verdict.trust))
        table = self.tbl_verdict
        table.setRowCount(len(verdict.rows))
        for row, contract in enumerate(verdict.rows):
            first = QtWidgets.QTableWidgetItem(contract.name)
            # The contract's own `source`, which is the engine-side thing each
            # number was compared against -- `result.grad_l2 vs
            # LbfgsConfig::gradient_tolerance` and so on. It is the same
            # traceability the breakdown gives through its engine keys.
            first.setToolTip(
                f"{contract.source}. Family: {contract.family}. "
                f"{contract.because}"
            )
            table.setItem(row, 0, first)
            word = QtWidgets.QTableWidgetItem(contract.word)
            word.setTextAlignment(
                int(QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter)
            )
            # The state is coloured and also typed: a reader who cannot see the
            # colour reads the same word, which is why the word is in the cell
            # and not only in the header.
            word.setForeground(QtGui.QBrush(QtGui.QColor(
                STATE_COLOURS.get(contract.state, STATE_COLOURS["unknown"])
            )))
            if contract.state == "fails":
                font = word.font()
                font.setBold(True)
                word.setFont(font)
            elif contract.state == "unmeasured":
                font = word.font()
                font.setItalic(True)
                word.setFont(font)
            table.setItem(row, 1, word)
            third = QtWidgets.QTableWidgetItem(contract.pair)
            third.setTextAlignment(
                int(QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter)
            )
            table.setItem(row, 2, third)
        self.lbl_verdict_why.setText(verdict.explanation())
        note = (
            "Three states, and they are three different answers. 'holds' means "
            "the measurement was made and the contract is satisfied; 'fails' "
            "means it was made and is not; 'unmeasured' means the result did "
            "not carry the number the contract needs, which is NOT a pass and "
            "is printed in its own colour and its own italic. A known failure "
            "outranks 'unmeasured': a `no` names the contract that failed, and "
            "the unmeasured ones are still listed. The numbers in the third "
            "column are measured / threshold, both read from the objects named "
            "in each row's tooltip; a row with no numbers says so rather than "
            "showing a zero. The values the engine does not report are the "
            "descent width and the support edge: no object here carries them, "
            "so those contracts stay unmeasured rather than being passed."
        )
        if (self._dock_result is None and self._receptor_path is not None
                and self._ligand_path is not None):
            # Nothing above can be measured, and an explanation is not a way
            # forward. The action is named, and it is the button in the panel
            # above rather than a second one here: the same click produces the
            # poses that carry the numbers these contracts read. The first
            # version of this panel did not say it, and the gate caught that a
            # dead end had been presented as a feature.
            note += (
                " Nothing here can be measured from a pose file, and "
                "'Re-dock this ligand here' in the panel above is the action "
                "that changes it: the poses it reports carry the run's own "
                "numbers, and this panel will describe whichever of them is "
                "selected afterwards."
            )
        self.lbl_verdict_note.setText(note)

    def _on_terms_dock_clicked(self) -> None:  # pragma: no cover - GUI
        """Run a docking, so the poses carry the conformations the terms need."""
        self.start_docking()

    def _rmsd_to_best(self, row: int):  # pragma: no cover - GUI
        if not self._pose_models:
            return None
        best = self._best_row()
        if row == best:
            return 0.0
        return _pose_distance(self._pose_models[best][0], self._pose_models[row][0])

    def open_file_dialog(self, kind: str = "ligand") -> None:  # pragma: no cover - GUI
        """Load a structure, asking the user which *role* it plays.

        A single generic "Open structure…" used to load everything as a
        ligand, so a receptor could not be loaded from the menu at all — the
        only way in was ``-r`` on the command line, which is a strange
        requirement for a GUI whose status bar says "load a receptor and a
        ligand". Choosing the role up front also matters because the three
        kinds are displayed differently and tracked separately.
        """
        titles = {
            "receptor": "Load receptor",
            "ligand": "Load ligand",
            "poses": "Open docked poses",
        }
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, titles.get(kind, "Open structure"), "", "Structures (*.pdbqt *.pdb *.sdf)"
        )
        if path:
            self.load_structure(Path(path), kind)

    # -- Docking -----------------------------------------------------------

    def start_docking(self) -> None:  # pragma: no cover - GUI
        """Kick off a docking run on a worker thread."""
        from ..core import GridBox, Receptor

        if not self._receptor_path:
            self.status_label.setText("load a receptor first")
            return
        if not self._ligand_path:
            self.status_label.setText("load a ligand first")
            return

        self.btn_dock.setEnabled(False)
        self._maps = None
        try:
            box_ = GridBox.from_center_size(
                tuple(float(s.value()) for s in self.center_spins),
                tuple(float(s.value()) for s in self.size_spins),
            )
            receptor = Receptor.from_pdbqt(self._receptor_path)
        except Exception as exc:
            self.status_label.setText(f"setup failed: {exc}")
            self.btn_dock.setEnabled(True)
            return

        # Everything from here on happens on a worker thread, map
        # precalculation included. Doing it inline froze the window: the status
        # text was set and then the event loop could not run to repaint it, so
        # the user got no feedback at all while a 40 A box took seconds to tabulate.
        self._thread = QtCore.QThread(self)
        self._worker = DockingWorker(
            receptor,
            self._ligand_path,
            box_,
            self.sp_exhaust.value(),
            self.sp_seed.value(),
            self.cb_scoring.currentText(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_dock_finished)
        self._worker.failed.connect(self._on_dock_failed)
        self._worker.progressed.connect(self.status_label.setText)
        self._thread.start()

    def _on_dock_finished(self, result) -> None:  # pragma: no cover - GUI
        self.btn_dock.setEnabled(True)
        self._thread.quit()
        self._thread.wait()
        self._thread.deleteLater()
        self._thread = None
        self._worker = None

        from ..pdbqt_writer import read_pdbqt_models

        # Write the poses to a temp file, load them, then delete it. It used to
        # be created with delete=False and never removed, so every Dock click
        # left a `tmp*.pdbqt` behind in %TEMP%.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".pdbqt", delete=False
        ) as handle:
            result.write_pdbqt(handle.name)
            temp = handle.name
        try:
            # The result comes along, because the file it was just written to is
            # a lossy summary of it: one decimal place on the energy and nothing
            # at all of the intermolecular part. Handing the file over without
            # the result is how the pose table ended up unable to show the
            # engine's own numbers.
            self.load_structure(Path(temp), "poses", result=result)
        finally:
            Path(temp).unlink(missing_ok=True)
        del read_pdbqt_models
        self.status_label.setText(
            f"done: {result.num_poses} poses, best {result.best_energy:.2f} kcal/mol "
            f"in {result.elapsed_seconds:.1f} s"
        )

    def _on_dock_failed(self, message: str) -> None:  # pragma: no cover - GUI
        self.btn_dock.setEnabled(True)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
            self._worker = None
        self.status_label.setText(f"docking failed: {message}")

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI
        """Stop running searches before the window goes away.

        A `QThread` that is still running when its owner is destroyed aborts the
        process with "QThread: Destroyed while thread is still running". Closing
        the window during a search used to be exactly that.
        """
        for name in ("_thread", "_pocket_thread", "_terms_thread"):
            thread = getattr(self, name, None)
            if thread is not None and thread.isRunning():
                thread.quit()
                # The pocket search and the term-map build are bounded numpy
                # loops, not cancellable ones, so `quit()` only takes effect
                # between slots. A long wait is better than an abort: the user
                # asked to close, and the alternative used to be a crash on the
                # way out.
                thread.wait(15000 if name != "_thread" else 3000)
        super().closeEvent(event)


def _energy_of(model_text: str) -> float | None:
    """The pose energy from a ``REMARK VINA RESULT`` line, or ``None``.

    ``None`` rather than ``0.0``: a file that carries no energy remark — a
    prepared ligand, a structure from another tool — has an *unknown* energy,
    and printing 0.00 would put a fabricated number in the pose list and make
    ``argmin`` pick an arbitrary pose as "best".
    """
    for line in model_text.splitlines():
        if "VINA RESULT" in line:
            try:
                return float(line.split(":", 1)[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def run(receptor=None, ligand=None, poses=None) -> int:
    """Launch the Qt application.

    Two things are wrapped around the event loop, and both are about the exit
    code. `event_loop()` marks the one place in this package that owns the
    loop, which is what lets `crashguard` end a crashed run by asking `exec()`
    to return `EXIT_UNHANDLED` instead of killing the process -- the window then
    closes the way it closes for any other exit. `finalise()` is the backstop:
    a failure `crashguard` recorded can never be reported as 0, whatever the
    loop returned. `odcli workbench` returns this number too, so the code a
    user sees from the CLI and from `odgui` is the same code.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow(receptor, ligand, poses)
    window.show()
    with crashguard.event_loop():
        code = app.exec()
    return crashguard.finalise(code)
