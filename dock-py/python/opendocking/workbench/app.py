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
// Explicit locations. moderngl assigns locations from the order of the
// `vertex_array` buffer list, and getting that order wrong silently feeds
// colours into positions; pinning them removes the ambiguity.
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_color;
out vec3 v_color;
out vec3 v_normal;
void main() {
    v_color = in_color;
    v_normal = in_normal;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

SPHERE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
in vec3 v_normal;
uniform float opacity;
out vec4 out_color;
void main() {
    // A single Lambert term is enough to read the curvature of a sphere and
    // costs almost nothing; a second light or shadowing would not survive a
    // 30 000-atom receptor at interactive rates.
    float lambert = 0.35 + 0.65 * max(dot(normalize(v_normal),
                                          normalize(vec3(0.4, 0.5, 0.8))), 0.0);
    out_color = vec4(v_color * lambert, opacity);
}
"""

LINE_VERTEX_SHADER = """
#version 330 core
uniform mat4 mvp;
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
uniform float opacity;
out vec4 out_color;
void main() { out_color = vec4(v_color, opacity); }
"""


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
    """Compile the two shader programs on `ctx`."""
    sphere = ctx.program(
        vertex_shader=SPHERE_VERTEX_SHADER,
        fragment_shader=SPHERE_FRAGMENT_SHADER,
    )
    lines = ctx.program(
        vertex_shader=LINE_VERTEX_SHADER,
        fragment_shader=LINE_FRAGMENT_SHADER,
    )
    return sphere, lines


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
        self.camera = Camera()
        self.molecules: list[MoleculeView] = []
        self.box_center = np.zeros(3, np.float32)
        self.box_size = np.asarray([22.0, 22.0, 22.0], np.float32)
        self.show_box = True
        #: One of ``REPRESENTATION_KEYS``. Set through :meth:`set_representation`
        #: so the status bar can be told about it.
        self.representation = "spheres"
        self._ctx = None
        self._sphere_prog = None
        self._line_prog = None
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

        self._sphere_prog, self._line_prog = build_programs(self._ctx)
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
        mvp = self.camera.projection(aspect) @ self.camera.view_matrix()

        for mol in self.molecules:
            if not mol.visible or len(mol.coords) == 0:
                continue
            self._draw_molecule(mol, mvp)
        if self.show_box:
            self._draw_box(mvp)

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

        self.status_label = QtWidgets.QLabel("ready — load a receptor and a ligand")
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setWidget(panel_widget)
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
                c = view.center()
                for spin, value in zip(self.center_spins, c):
                    spin.blockSignals(True)
                    spin.setValue(float(value))
                    spin.blockSignals(False)
                self._on_box_changed()
            else:
                self._ligand_path = path
            self.status_label.setText(
                f"loaded {path.name} ({len(view.coords)} atoms)"
            )
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
        """Stop a running search before the window goes away.

        A `QThread` that is still running when its owner is destroyed aborts the
        process with "QThread: Destroyed while thread is still running". Closing
        the window during a search used to be exactly that.
        """
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
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
