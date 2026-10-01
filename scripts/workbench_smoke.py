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

**A run that stopped is not a run that passed.**

That is the reason for everything below. This file used to carry no per-check
markers and no summary line at all, and its checks were bare `assert`s inside
the two level functions. So a run that died on the *first* pose-loading
assertion printed the same shape of transcript as a run that passed — the same
handful of lines, then a non-zero exit with nothing saying which checks never
ran — and it did so by skipping level 2 entirely. A `[FAIL]` count of zero was
not evidence of anything, because no `[FAIL]` marker existed to count.

So:

* every check is recorded as PASS, FAIL or SKIP, and the counts go on one
  summary line printed on **every** path out of `main`, including the one where
  an exception escapes;
* a failed check no longer aborts the run, so level 2 still executes after a
  level-1 failure and the summary describes the whole run;
* a question this machine cannot answer — no OpenGL context, no framebuffer —
  is a **SKIP carrying a reason**, not a silent absence and not a failure of the
  renderer. It is counted, so the summary can never be read as a clean sweep;
* a run in which *nothing at all* was verified cannot report success. If no
  check passed, the script says so and exits ``2``.

Exit codes:

``0``  finished, no failures
``1``  finished, at least one failure
``2``  **did not finish** — an exception escaped, or no check was able to run

The reason a missing context is a skip rather than a failure is worth stating,
because the alternative is a suite that cries wolf on every headless machine and
is therefore ignored on the one that matters. `Viewport.initializeGL` sets a
tooltip when it ran and failed, and leaves that tooltip empty when Qt never
called it at all; those are two different facts and the two branches below
report them differently. `odgui --check` is what settles which one this is: it
is the same two-stage question the launcher asks a user, and this file does not
write a second probe to ask it again.
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

#: The one place in the tree that knows how to ask `odgui --check` what this
#: machine can do; it was a private copy in each of several check scripts.
#: `python scripts/_gui_check.py` audits that they still use it.
from _gui_check import odgui_check  # noqa: E402

from opendocking.workbench import (  # noqa: E402
    Camera,
    MoleculeView,
    _icosphere,
    _parse_pdbqt_atoms,
)

EXAMPLES = REPO / "examples"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INCOMPLETE = 2

#: Every check this run produced, as ``(tag, name, reason)``. A list rather
#: than counters, so the summary can name the skips instead of only counting
#: them -- a bare "1 skipped" does not tell a reader which question the machine
#: declined to answer.
RESULTS: list[tuple[str, str, str]] = []


def record(tag: str, name: str, reason: str = "") -> bool:
    """Print and remember one check. Returns True only for a PASS."""
    RESULTS.append((tag, name, reason))
    print(f"  [{tag}] {name}" + (f"  — {reason}" if reason else ""))
    return tag == "PASS"


def ok(name: str, reason: str = "") -> bool:
    return record("PASS", name, reason)


def bad(name: str, reason: str) -> bool:
    return record("FAIL", name, reason)


def skip(name: str, reason: str) -> bool:
    return record("SKIP", name, reason)


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

    These are recorded rather than asserted. An `assert` here used to end the
    whole script: level 2 never ran and no summary was printed, so the level-2
    coverage that is the point of this file was lost to a level-1 problem, and
    the transcript gave no hint that anything had been lost.
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
        if n > 0:
            ok("pose list is populated", f"{n} pose(s)")
        else:
            bad("pose list is populated", "the pose list is empty")
        if win.lbl_rmsd.text() == "—":
            bad("RMSD readout populated", "it still reads the em dash placeholder")
        else:
            ok("RMSD readout populated", win.lbl_rmsd.text())
        if "0.00" in win.lbl_energy.text():
            bad(
                "best pose energy is real",
                f"the best pose reads {win.lbl_energy.text()} — the VINA RESULT "
                f"remark is not being read: rows {rows[:3]}",
            )
        else:
            ok("best pose energy is real", win.lbl_energy.text())
        # Every row at 0.00 is the signature of an unreadable energy remark.
        if rows and all("0.00" in t for t in rows):
            bad("pose energies are not all 0.00", f"rows {rows[:3]}")
        else:
            ok("pose energies are not all 0.00")
    except Exception as exc:  # noqa: BLE001 - recorded, then the run continues
        bad("pose loading did not raise", f"{type(exc).__name__}: {exc}")
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def check_qt_window() -> bool:
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
        # Two different things leave `_ctx` as None, and they are not the same
        # report. `initializeGL` sets a tooltip when it ran and failed, so a
        # non-empty tooltip means Qt gave us a context and it was not a usable
        # one. An empty tooltip means Qt never called `initializeGL` at all.
        # `odgui --check` settles which machine this is: if it says the machine
        # can make a context, then our viewport failing to get one is a defect
        # and is reported as one.
        tip = win.viewport.toolTip()
        answer = odgui_check()
        if answer.can_make_context:
            bad(
                "Qt GL context on the viewport",
                "odgui --check reports this machine CAN create a context "
                f"({answer.describe()}), so a viewport without one is ours: "
                f"{tip or '(Qt gave no reason)'}",
            )
        else:
            skip(
                "Qt GL context on the viewport",
                f"{tip or 'Qt never called initializeGL'}; {answer.describe()}",
            )
        return False
    ok("Qt GL context on the viewport", win.viewport.toolTip() or "created")

    # The real check: the widget drew, and the drawing reached the framebuffer.
    win.viewport.repaint()
    for _ in range(5):
        qapp.processEvents()
    # QOpenGLWidget.grabFramebuffer() already returns a QImage in PyQt6; there
    # is no pixmap in between to convert from.
    qimage = win.viewport.grabFramebuffer()
    if qimage.isNull():
        bad("grabFramebuffer returned an image", "the image was null")
        return False
    ok("grabFramebuffer returned an image", f"{qimage.width()}x{qimage.height()}")
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
        bad(
            "the Qt widget framebuffer is not empty",
            f"{len(colours)} distinct colours, {non_background} non-black samples "
            f"— nothing was drawn",
        )
        return False
    ok("the Qt widget framebuffer is not empty",
       f"{len(colours)} colours, {non_background} non-black samples")
    out = REPO / "dist" / "workbench_qt_widget.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not qimage.save(str(out)):
        bad("the widget frame was written", f"{qimage.save(str(out))} from {out}")
    else:
        ok("the widget frame was written", str(out))
        print(f"  wrote {out}")
    return True


def check_renderer() -> bool:
    """Level 2: compile the shaders and actually draw, then read it back."""
    try:
        import moderngl
    except ImportError:
        skip("moderngl importable", "moderngl is not installed")
        return False

    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:
        # The machine could not give this process a context. That is a fact
        # about the machine, and the reason is reported rather than guessed at.
        skip(
            "standalone OpenGL context",
            f"{type(exc).__name__}: {exc}",
        )
        return False
    ok("standalone OpenGL context", ctx.info.get("GL_RENDERER", "?"))

    print(f"  GL renderer: {ctx.info.get('GL_RENDERER', '?')}")
    print(f"  GL version:  {ctx.info.get('GL_VERSION', '?')}")

    import opendocking.workbench.app as wb

    try:
        sphere_prog, line_prog, bg_prog = wb.build_programs(ctx)
        print("  all three shader programs compiled")
    except Exception as exc:
        bad("all three shader programs compile", f"{type(exc).__name__}: {exc}")
        print(f"  shader compile FAILED: {exc}")
        return False
    ok("all three shader programs compile")

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
        bad("draw_spheres + draw_lines complete", f"{type(exc).__name__}: {exc}")
        print(f"  draw FAILED: {exc}")
        return False
    ok("draw_spheres + draw_lines complete")
    print("  draw_spheres + draw_lines completed without error")

    # `fbo.read` returns rows bottom-up, so flip for a top-down image.
    data = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1]
    unique = len(np.unique(data.reshape(-1, 3), axis=0))
    lit = int((data.max(axis=2) > 40).sum())
    print(f"  framebuffer: {unique} distinct colours, {lit} non-background pixels")
    if lit < 100:
        bad("the renderer framebuffer is not blank", f"only {lit} lit pixels")
    else:
        ok("the renderer framebuffer is not blank", f"{lit} lit pixels")
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
        written = png
    except ImportError:
        with open(out, "wb") as fh:
            fh.write(f"P6\n{W} {H}\n255\n".encode())
            fh.write(rgb.tobytes())
        print("  imageio not installed, wrote a PPM instead")
        written = out
    # Checked rather than assumed: `QImage.save` and `imageio.imwrite` both
    # return a success flag that used to be discarded, so a frame that could
    # not be written still printed "wrote ...".
    if not written.is_file() or written.stat().st_size == 0:
        bad("the renderer frame was written", f"{written} is missing or empty")
    else:
        ok("the renderer frame was written", f"{written} ({written.stat().st_size} bytes)")
        print(f"  wrote {written}")
    return True


def summarise(incomplete: str = "") -> int:
    """The one line that says what this run actually established.

    Printed unconditionally, including after an exception has escaped `main`.

    `incomplete` is the reason the run did not reach the end, if it did not. It
    forces the exit code to `EXIT_INCOMPLETE` whatever the counts say, and that
    override is the whole point: a run that died half way through having already
    passed six checks would otherwise report `6 passed, 0 failed` and exit 0,
    which is the exact shape of a successful run. A half-run and a clean run must
    not share an exit code, because the exit code is what a CI gate reads.
    """
    npass = sum(1 for tag, _, _ in RESULTS if tag == "PASS")
    nfail = sum(1 for tag, _, _ in RESULTS if tag == "FAIL")
    nskip = sum(1 for tag, _, _ in RESULTS if tag == "SKIP")
    print()
    print(f"--- summary: {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks")
    for tag, name, reason in RESULTS:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {reason}")
    if nskip:
        print("    a skipped check is a question this machine did not answer, "
              "not one that passed")
    if incomplete:
        print(f"RESULT: DID NOT FINISH — {incomplete}")
        return EXIT_INCOMPLETE
    if nfail:
        print("RESULT: FAIL — the run finished and something did not hold")
        return EXIT_FAILED
    if npass == 0:
        # The guard that stops a vacuous green. Every check being skipped means
        # the run verified nothing, and a script that verified nothing must not
        # be able to report success: this is the "0 [FAIL] and no summary"
        # failure wearing a different hat.
        print("RESULT: DID NOT FINISH — no check was able to run, so nothing "
              "was verified")
        return EXIT_INCOMPLETE
    print(f"RESULT: PASS — {npass} checks verified"
          + (f", {nskip} could not be run here" if nskip else ""))
    return EXIT_OK


def main() -> int:
    print("[1/2] the real Qt window, on the default platform")
    check_qt_window()
    print("[2/2] renderer against a standalone OpenGL context")
    reached = check_renderer()
    if not reached:
        print()
        print("The renderer could not be exercised in this environment; the "
              "summary below says which checks that cost.")
    return summarise()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to never be silent
        # Nothing below this line may raise, or the one line that says the run
        # stopped would itself be lost. A traceback to stderr, a counted reason
        # in the summary, and an exit code that cannot be read as a pass.
        import traceback

        traceback.print_exc()
        print()
        print(f"DID NOT FINISH — an exception escaped before the checks above "
              f"were all recorded: {type(_exc).__name__}: {_exc}")
        raise SystemExit(summarise(f"an exception escaped: {type(_exc).__name__}: {_exc}"))
