"""Headless verification of the Open Docking workbench renderer.

Two levels are checked, and the script reports which one it reached:

1. **Offscreen Qt** — build the real `MainWindow`, load an actual receptor and
   a real docked pose, and let Qt lay the interface out. This proves the widget
   tree, the controls and the pose browser are wired up.

2. **Standalone moderngl** — compile the two shader programs and run the exact
   `draw_spheres` / `draw_lines` calls the widget uses, against a standalone
   OpenGL context, then read the framebuffer back and check that something was
   actually drawn.

Level 2 is what makes this worth running: a GLSL typo, a mismatched vertex
attribute name or a wrong index buffer all raise here instead of silently
producing a black window. It still cannot judge whether the picture *looks*
right — that needs a human on a screen.
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.pop("QT_QPA_PLATFORM", None)
REPO = pathlib.Path(__file__).resolve().parents[1]

# Prefer the *installed* `opendocking`, because that is what a user actually gets
# and it is the artifact the test suite should be exercising. The source tree
# (`dock-py/python`) is only put on the path as a fallback, and only when it
# already carries a built native module — otherwise it would shadow the
# installed package with a copy that cannot import at all.
_SOURCE = REPO / "dock-py" / "python"
if (_SOURCE / "opendocking" / "_dockpy.pyd").exists() or (
    _SOURCE / "opendocking" / "_dockpy.abi3.so"
).exists():
    sys.path.insert(0, str(_SOURCE))
else:
    import opendocking as _installed  # noqa: F401

    print(f"using installed opendocking from {_installed.__file__}")

for _stream in (sys.stdout, sys.stderr):
    _reconf = getattr(_stream, "reconfigure", None)
    if _reconf is not None:
        try:
            _reconf(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic streams
            pass

import numpy as np  # noqa: E402

from opendocking.workbench import (  # noqa: E402
    Camera,
    MoleculeView,
    _icosphere,
    _parse_pdbqt_atoms,
)

EXAMPLES = REPO / "examples"


def _first_model(path: pathlib.Path):
    """Coordinates and elements of the *first* MODEL in a multi-model file."""
    text = path.read_text(encoding="utf-8")
    if "MODEL" not in text:
        return _parse_pdbqt_atoms(text)
    head = text.split("MODEL", 1)[1]
    head = head.split("ENDMDL", 1)[0]
    return _parse_pdbqt_atoms("MODEL" + head)


def _synthetic(n_atoms: int = 12, seed: int = 3) -> MoleculeView:
    """A molecule that does not depend on the example data being present."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-4.0, 4.0, (n_atoms, 3)).astype(np.float32)
    elements = ["C", "O", "N", "S"][: n_atoms % 4] * (n_atoms // 4 + 1)
    return MoleculeView(
        name="synthetic",
        coords=coords,
        elements=elements[:n_atoms],
        bonds=[(i, i + 1) for i in range(n_atoms - 1)],
        radius=0.35,
    )


def check_pose_loading(qapp, rec_p, pose_p) -> None:
    """Load a pose file through the real window and check what it read.

    Three things have to hold, and each has been broken at least once:

    * loading must not raise — it used to call ``.splitlines()`` on the line
      *lists* that ``read_pdbqt_models`` returns;
    * the energies must be real, not a column of 0.00 — which is what a
      ``REMARK VINA RESULT`` written outside the ``MODEL`` block looks like to a
      per-model reader;
    * the RMSD readout must be populated, which it was not while
      ``setCurrentRow`` ran before the pose view existed.
    """
    import opendocking.workbench.app as wb

    win = wb.MainWindow(
        receptor=rec_p,
        ligand=EXAMPLES / "ibuprofen_prep.pdbqt",
        poses=pose_p,
    )
    try:
        win.resize(900, 650)
        win.show()
        for _ in range(5):
            qapp.processEvents()
        n = win.pose_list.count()
        rows = [win.pose_list.item(i).text() for i in range(n)]
        print(
            f"  pose file: {n} pose(s), best energy {win.lbl_energy.text()!r}, "
            f"RMSD {win.lbl_rmsd.text()!r}"
        )
        assert n > 0, "the pose list is empty"
        assert win.lbl_rmsd.text() != "—", "the RMSD readout was never populated"
        assert "0.00" not in win.lbl_energy.text(), (
            f"the best pose reads 0.00 kcal/mol — the VINA RESULT remark is not "
            f"being read: rows {rows[:3]}"
        )
        # Every row at 0.00 is the signature of an unreadable energy remark.
        assert not all("0.00" in t for t in rows), f"every pose is 0.00: {rows[:3]}"
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def check_qt_window(errors: list[str]) -> bool:
    """Level 1: does the real window build, lay itself out, and *render*?

    The platform is deliberately left at Qt's default. The `offscreen`
    platform is documented not to support OpenGL, so forcing it makes
    `QOpenGLWidget.initializeGL` never fire — which reads exactly like a broken
    renderer and is not one. `scripts/qt_gl_probe.py` measures which routes
    actually yield a context.

    This level therefore does the real thing: build `MainWindow`, let Qt give
    the viewport a context, and read the widget's own framebuffer back.
    """
    from PyQt6 import QtWidgets

    import opendocking.workbench.app as wb

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = wb.MainWindow()
    win.resize(900, 650)
    win.show()
    for _ in range(5):
        qapp.processEvents()

    rec_p = EXAMPLES / "rec_prep.pdbqt"
    pose_p = EXAMPLES / "poses.pdbqt"
    if rec_p.is_file() and pose_p.is_file():
        # Exercise the pose-loading path the way the CLI does, by constructing
        # the window with the pose file. This used to build its own MoleculeView
        # objects and parse the pose by hand, so
        # `MainWindow.load_structure(kind="poses")` was never executed -- and
        # that path crashed on the first `.splitlines()`, because
        # `read_pdbqt_models` returns lines, not text. The render check passed
        # while `odgui -p poses.pdbqt` could not start at all.
        check_pose_loading(qapp, rec_p, pose_p)
    if rec_p.is_file() and pose_p.is_file():
        rec_coords, rec_elements = _parse_pdbqt_atoms(rec_p.read_text(encoding="utf-8"))
        pose_coords, pose_elements = _first_model(pose_p)
        print(f"  receptor {rec_coords.shape[0]} atoms, first pose {pose_coords.shape[0]} atoms")
        receptor = MoleculeView(
            name="receptor", coords=rec_coords, elements=rec_elements, bonds=[], radius=0.35
        )
        ligand = MoleculeView(
            name="ligand",
            coords=pose_coords,
            elements=pose_elements,
            bonds=[(i, i + 1) for i in range(max(0, len(pose_coords) - 1))],
            radius=0.30,
        )
    else:
        print("  (example data missing; using a synthetic molecule)")
        receptor, ligand = _synthetic(30), _synthetic(16, seed=9)

    win.viewport.molecules = [receptor, ligand]
    win.viewport.camera = Camera(distance=40.0, yaw=0.6, pitch=0.4)
    win.viewport.box_center = (0.0, 0.0, 1.5)
    win.viewport.box_size = (18.0, 18.0, 18.0)
    for _ in range(5):
        qapp.processEvents()
    print(f"  MainWindow built; viewport size {win.viewport.width()}x{win.viewport.height()}")

    if win.viewport._ctx is None:
        # The offscreen platform is the usual cause. Say so, because "no
        # context" otherwise looks like a broken renderer rather than an
        # environment that cannot provide one.
        print(
            "  no GL context on the viewport; the offscreen platform cannot "
            f"provide one (Qt said {win.viewport.toolTip()!r})"
        )
        errors.append("no Qt GL context")
        return False

    # The real check: the widget drew, and the drawing reached the framebuffer.
    win.viewport.repaint()
    for _ in range(5):
        qapp.processEvents()
    # QOpenGLWidget.grabFramebuffer() already returns a QImage in PyQt6; there
    # is no pixmap in between to convert from.
    qimage = win.viewport.grabFramebuffer()
    if qimage.isNull():
        errors.append("grabFramebuffer returned a null image")
        print("  grabFramebuffer returned a null image")
        return False
    w, h = qimage.width(), qimage.height()
    colours: set[int] = set()
    non_background = 0
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            rgb = qimage.pixel(x, y)
            colours.add(rgb)
            if (rgb & 0x00FFFFFF) != 0:
                non_background += 1
    print(
        f"  widget framebuffer {w}x{h}: {len(colours)} distinct colours, "
        f"{non_background} non-black samples"
    )
    if len(colours) < 5 or non_background < 50:
        errors.append("the Qt widget framebuffer looks empty")
        print("  the Qt widget framebuffer looks empty — nothing was drawn")
        return False
    out = REPO / "dist" / "workbench_qt_widget.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    qimage.save(str(out))
    print(f"  wrote {out}")
    return True


def check_renderer(errors: list[str]) -> bool:
    """Level 2: compile the shaders and actually draw, then read it back."""
    try:
        import moderngl
    except ImportError:  # pragma: no cover
        print("  moderngl is not installed")
        errors.append("moderngl missing")
        return False

    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:
        print(f"  no standalone OpenGL context: {type(exc).__name__}: {exc}")
        errors.append("no standalone GL context")
        return False

    print(f"  GL renderer: {ctx.info.get('GL_RENDERER', '?')}")
    print(f"  GL version:  {ctx.info.get('GL_VERSION', '?')}")

    import opendocking.workbench.app as wb

    try:
        sphere_prog, line_prog, bg_prog = wb.build_programs(ctx)
        print("  all three shader programs compiled")
    except Exception as exc:
        errors.append(f"shader compile failed: {type(exc).__name__}: {exc}")
        print(f"  shader compile FAILED: {exc}")
        return False

    W, H = 320, 240
    ctx.enable(moderngl.DEPTH_TEST)
    verts, faces = _icosphere(1)
    mesh = wb.SphereMesh(verts, faces)
    print(f"  mesh: {len(mesh.verts)} vertices, {mesh.n_indices} indices")

    # A standalone context has no default framebuffer on Windows, so draw into
    # an explicit one and read that back.
    fbo = ctx.simple_framebuffer((W, H), components=4)
    fbo.use()
    ctx.viewport = (0, 0, W, H)
    ctx.clear(0.05, 0.05, 0.08, 1.0)

    # Use the same camera the workbench uses, so the picture is the one a user
    # would see rather than a raw wireframe at the origin.
    camera = Camera(distance=26.0, yaw=0.7, pitch=0.45)
    view = camera.view_matrix()
    mvp = camera.projection(W / H) @ view

    # The per-frame uniforms are what the viewport sets each paint. Without
    # them the shaders still compile and still draw -- `mv` defaults to zero,
    # so the fog factor is zero and the backdrop is never drawn -- which means
    # a smoke test that skipped them would pass while the real window showed
    # something else entirely.
    span = max(camera.distance * 0.55, 8.0)
    wb.set_frame_uniforms(sphere_prog, view, span * 0.55, span * 2.1)
    wb.set_frame_uniforms(line_prog, view, span * 0.55, span * 2.1)
    bg_vao = ctx.vertex_array(bg_prog, [])
    wb.draw_background(ctx, bg_prog, bg_vao, W, H)

    rec_p = EXAMPLES / "rec_prep.pdbqt"
    pose_p = EXAMPLES / "poses.pdbqt"
    subjects: list[tuple[str, MoleculeView]] = []
    if rec_p.is_file() and pose_p.is_file():
        rc, re_ = _parse_pdbqt_atoms(rec_p.read_text(encoding="utf-8"))
        pc, pe = _first_model(pose_p)
        subjects.append(
            (
                "receptor",
                MoleculeView(
                    name="receptor", coords=rc, elements=re_, bonds=[], radius=0.35
                ),
            )
        )
        subjects.append(
            (
                "ligand",
                MoleculeView(
                    name="ligand",
                    coords=pc,
                    elements=pe,
                    bonds=[(i, i + 1) for i in range(max(0, len(pc) - 1))],
                    radius=0.30,
                ),
            )
        )
        print(f"  drawing the real example: {rc.shape[0]}-atom receptor + {pc.shape[0]}-atom pose")
    else:
        subjects.append(("synthetic", _synthetic(12)))
    try:
        for _name, mol in subjects:
            wb.draw_spheres(ctx, sphere_prog, mesh, mol, mvp)
            segs = mol.bond_segments()
            if len(segs):
                wb.draw_lines(
                    ctx,
                    line_prog,
                    segs.reshape(-1, 3),
                    np.tile(np.asarray(mol.color, np.float32), (len(segs) * 2, 1)),
                    mvp,
                    1.0,
                )
    except Exception as exc:
        errors.append(f"draw failed: {type(exc).__name__}: {exc}")
        print(f"  draw FAILED: {exc}")
        return False
    print("  draw_spheres + draw_lines completed without error")

    # `fbo.read` returns rows bottom-up, so flip for a top-down image.
    data = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1]
    unique = len(np.unique(data.reshape(-1, 3), axis=0))
    lit = int((data.max(axis=2) > 40).sum())
    print(f"  framebuffer: {unique} distinct colours, {lit} non-background pixels")
    if lit < 100:
        errors.append(f"framebuffer looks empty ({lit} lit pixels)")
    else:
        print("  framebuffer is not blank — geometry reached the screen")

    out = REPO / "dist" / "workbench_render.ppm"
    out.parent.mkdir(parents=True, exist_ok=True)
    # `fbo.read(components=3)` hands back RGB in that order, and both writers
    # below expect RGB. This used to reverse the channels, so every screenshot
    # the smoke test produced had its red and blue swapped -- which made a
    # carbon-grey protein look blue and a red oxygen look blue, and sent anyone
    # reading the image looking for a colour bug that was in the harness.
    rgb = np.ascontiguousarray(data)
    try:
        import imageio.v2 as imageio

        png = out.with_suffix(".png")
        imageio.imwrite(png, rgb)
        print(f"  wrote {png}")
    except ImportError:
        with open(out, "wb") as fh:
            fh.write(f"P6\n{W} {H}\n255\n".encode())
            fh.write(rgb.tobytes())
        print(f"  wrote {out} (imageio not installed)")
    return True


def main() -> int:
    errors: list[str] = []
    print("[1/2] the real Qt window, on the default platform")
    check_qt_window(errors)
    print("[2/2] renderer against a standalone OpenGL context")
    reached = check_renderer(errors)
    if not reached:
        print()
        print("The renderer could not be exercised in this environment.")
    for e in errors:
        print("ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
