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

import numpy as np

from . import Camera, MoleculeView, _icosphere, _parse_pdbqt_atoms, _require_gui

_require_gui()
from PyQt6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets  # noqa: E402
from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # noqa: E402

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
    "REPRESENTATIONS",
    "REPRESENTATION_KEYS",
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


def draw_spheres(ctx, prog, mesh: SphereMesh, mol, mvp) -> None:
    """Draw every atom of `mol` as a shaded sphere.

    The mesh is expanded on the CPU rather than drawn with hardware
    instancing. An earlier version declared the per-atom attributes with
    ``divisor=1`` and then filled them with ``np.repeat``, which are two
    mutually exclusive choices; with a few hundred atoms the expansion is well
    under a millisecond and it is the version that is obviously correct.
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
    radii = np.asarray(mol.atom_radii(), np.float32).reshape(n_atoms, 1, 1)
    positions = (centers + radii * mesh.verts[None, :, :]).reshape(-1, 3)
    # On a unit sphere the position is the normal, so the mesh supplies both.
    normals = np.tile(mesh.verts, (n_atoms, 1))
    colors = np.repeat(mol.atom_colors().astype(np.float32), v, axis=0)
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


#: The representations the viewport offers, in the order they are listed.
#: `needs_backbone` marks the ones that only mean something for a protein; a
#: molecule without a backbone falls back and says so rather than drawing
#: nothing.
REPRESENTATIONS: tuple[tuple[str, str, bool], ...] = (
    ("spheres", "Space-filling", False),
    ("ball_and_stick", "Ball and stick", False),
    ("stick", "Skeletal (sticks only)", False),
    ("ribbon", "Ribbon (backbone)", True),
    ("cartoon", "Cartoon (ribbon + side chains)", True),
)
REPRESENTATION_KEYS = tuple(key for key, _, _ in REPRESENTATIONS)


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
        #: the atoms whose proximity is the reason the site was reported.
        self.pocket_opacity = 0.34
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
        colors = mol.atom_colors()

        if mode == "spheres":
            draw_spheres(self._ctx, self._sphere_prog, self._mesh, mol, mvp)
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
                sphere_geometry(mol.coords, mol.atom_radii() * 0.42, colors),
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
        draw_mesh(
            self._ctx, self._sphere_prog, mol.backbone_ribbon(), mvp, mol.opacity
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
        self.camera.center = np.asarray(point, np.float32)
        self.camera.distance = max(12.0, float(radius) * 1.6)
        self.update()

    def focus_residue(self, residue: str) -> bool:  # pragma: no cover - GUI
        """Move the camera onto a residue's contacts. True if it moved.

        Zooms to the span of the atoms actually making contact rather than to
        the whole residue, which for a buried side chain is the difference
        between the interaction filling the window and being a few pixels wide
        in it. A residue with no contacts found leaves the camera alone and says
        so, rather than jumping somewhere arbitrary.
        """
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
                seed=self.seed if self.seed else None,
                scoring=self.scoring,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


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
        self._pocket_models: list = []
        self._pocket_thread: QtCore.QThread | None = None
        self._pocket_worker: PocketWorker | None = None
        self._maps = None
        self._worker = None
        self._thread = None

        self._build_ui()
        for path, kind in ((receptor, "receptor"), (ligand, "ligand"), (poses, "poses")):
            if path:
                self.load_structure(Path(path), kind)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:  # pragma: no cover - GUI
        panel_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel_widget)

        self.cmb_representation = QtWidgets.QComboBox()
        for key, label, _ in REPRESENTATIONS:
            self.cmb_representation.addItem(label, key)
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
        legend = QtWidgets.QWidget()
        legend_row = QtWidgets.QHBoxLayout(legend)
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.setSpacing(8)
        from . import COLOR_CONTACT, CONTACT_LABELS

        for kind in ("hbond", "polar", "hydrophobic", "close"):
            r, g, b = (int(round(v * 255)) for v in COLOR_CONTACT[kind])
            chip = QtWidgets.QLabel("━")
            chip.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold;")
            text = QtWidgets.QLabel(CONTACT_LABELS[kind])
            text.setStyleSheet("color: #9aa3ad;")
            legend_row.addWidget(chip)
            legend_row.addWidget(text)
        legend_row.addStretch(1)
        form.addRow("legend", legend)

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

        self.sp_exhaust = QtWidgets.QSpinBox()
        self.sp_exhaust.setRange(1, 256)
        self.sp_exhaust.setValue(8)
        form.addRow("exhaustiveness", self.sp_exhaust)

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

        self.pose_list = QtWidgets.QListWidget()
        self.pose_list.currentRowChanged.connect(self._on_pose_selected)
        form.addRow("poses", self.pose_list)

        self.lbl_energy = QtWidgets.QLabel("—")
        self.lbl_energy.setStyleSheet("font-weight: bold; font-size: 15px;")
        form.addRow("best energy", self.lbl_energy)

        self.lbl_rmsd = QtWidgets.QLabel("—")
        form.addRow("RMSD to best", self.lbl_rmsd)

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

    def load_structure(self, path: Path, kind: str) -> None:  # pragma: no cover - GUI
        """Load a receptor, ligand, or multi-pose file into the viewport."""
        from . import COLOR_BEST_POSE, COLOR_LIGAND, COLOR_RECEPTOR

        if kind == "poses":
            from ..pdbqt_writer import read_pdbqt_models

            # `read_pdbqt_models` hands back one *list of lines* per pose. Join
            # once here, because everything downstream -- the energy remark, the
            # RMSD comparison and the coordinate parse -- works on text. Passing
            # the line lists straight through crashed on the first
            # `.splitlines()`, which took down the whole window: `odgui -p
            # poses.pdbqt` could not start at all.
            models = ["\n".join(lines) for lines in read_pdbqt_models(path)]
            if not models:
                self.status_label.setText(f"{path.name} contains no poses")
                return
            self._pose_models = [(body, _energy_of(body)) for body in models]
            self.pose_list.blockSignals(True)
            self.pose_list.clear()
            for i, (_, energy) in enumerate(self._pose_models):
                text = (
                    f"{energy:7.2f} kcal/mol" if energy is not None else "(energy n/a)"
                )
                self.pose_list.addItem(f"pose {i + 1}   {text}")
            self.pose_list.blockSignals(False)
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
                f"{self._pose_models[best][1]:.2f} kcal/mol"
                if self._pose_models[best][1] is not None
                else "energy not in file"
            )
            self.lbl_rmsd.setText("0.00 Å")
            self.pose_list.setCurrentRow(best)
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
                    self.pocket_table.selectRow(0)
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
        self.viewport.frame_all()

    def _best_row(self) -> int:
        """Index of the lowest-energy pose, or 0 when no energy is known.

        Ties and all-``None`` both fall back to the first row rather than an
        arbitrary pick from an all-equal list.
        """
        known = [e for _, e in self._pose_models if e is not None]
        if not known:
            return 0
        lowest = min(known)
        return next(
            i for i, (_, e) in enumerate(self._pose_models) if e == lowest
        )

    def _view_from_text(self, text, name, color, radius) -> MoleculeView:  # pragma: no cover
        coords, elements = _parse_pdbqt_atoms(text)
        return MoleculeView(
            name=name, coords=coords, elements=elements, color=color, radius=radius
        )

    # -- UI callbacks ------------------------------------------------------

    def _on_visibility(self) -> None:  # pragma: no cover - GUI
        rec = self.cb_receptor.isChecked()
        lig = self.cb_ligand.isChecked()
        for mol in self.viewport.molecules:
            mol.visible = rec if mol.role == "receptor" else lig
        self.viewport.update()

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
            self.pocket_table.selectRow(0)

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
        self.viewport.update()

    def _on_pose_selected(self, row: int) -> None:  # pragma: no cover - GUI
        if row < 0 or not self._pose_models or self._pose_view is None:
            return
        text, energy = self._pose_models[row]
        new_view = self._view_from_text(
            text, self._pose_view.name, self._pose_view.color, self._pose_view.radius
        )
        # Carry the role across. `_view_from_text` builds a fresh view, which
        # defaults to role "ligand"; losing it here made the pose stop being
        # recognisable as a pose, so the next docking run could not tell it from
        # a ligand view and stacked another result on top of it.
        new_view.role = self._pose_view.role
        idx = self.viewport.molecules.index(self._pose_view)
        self.viewport.molecules[idx] = new_view
        self._pose_view = new_view
        rmsd = self._rmsd_to_best(row)
        self.lbl_energy.setText(
            f"{energy:.2f} kcal/mol" if energy is not None else "energy not in file"
        )
        self.lbl_rmsd.setText("—" if rmsd is None else f"{rmsd:.2f} Å")
        self._refresh_contacts()
        self.viewport.update()

    def _rmsd_to_best(self, row: int):  # pragma: no cover - GUI
        if not self._pose_models:
            return None
        best = self._best_row()
        if row == best:
            return 0.0
        a = _parse_pdbqt_atoms(self._pose_models[best][0])[0]
        b = _parse_pdbqt_atoms(self._pose_models[row][0])[0]
        n = min(len(a), len(b))
        if n == 0:
            return None
        return float(np.sqrt(((a[:n] - b[:n]) ** 2).sum(axis=1).mean()))

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
            self.load_structure(Path(temp), "poses")
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
        for name in ("_thread", "_pocket_thread"):
            thread = getattr(self, name, None)
            if thread is not None and thread.isRunning():
                thread.quit()
                # The pocket search is a bounded numpy loop, not a cancellable
                # one, so `quit()` only takes effect between slots. A long wait
                # is better than an abort: the user asked to close, and the
                # alternative used to be a crash on the way out.
                thread.wait(15000 if name == "_pocket_thread" else 3000)
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
    """Launch the Qt application."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow(receptor, ligand, poses)
    window.show()
    return app.exec()
