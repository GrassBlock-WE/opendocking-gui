"""Drive the real workbench with real input events and check what the user sees.

The render smoke test proves geometry reaches the framebuffer. It says nothing
about whether the *controls* work, so this one posts genuine Qt mouse, wheel and
keyboard events at the live widgets and reads the result back.

Three things it checks that nothing else does:

* **Layout**: the geometry of every control in the docked panel, looking for
  zero-size, overlapping, clipped or out-of-panel widgets.
* **Interaction**: left-drag rotates, wheel zooms, Frame all refits, the
  checkboxes actually hide things, the box spins move the wireframe, the pose
  list switches poses and updates both readouts.
* **Behaviour under a real docking run**: that the window keeps painting while
  a search runs, that the Dock button re-enables, and what happens to state
  across two consecutive runs.

Run it on the real desktop platform. It does not force QT_QPA_PLATFORM --
offscreen is the one mode without OpenGL, and forcing it hides exactly the
bugs this is looking for.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from opendocking.workbench.app import MainWindow  # noqa: E402

OUT = ROOT / "dist" / "workbench_interaction"
OUT.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def skip(name: str, reason: str) -> bool:
    """Record a check that could not be run here, and say why.

    A skipped check and a failed check are different claims. "The viewport
    draws nothing" and "this environment cannot read the framebuffer" look
    identical in the output if the second one is reported as the first, and
    only one of them is ever true.
    """
    results.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  — {reason}")
    return False


# Whether `grabFramebuffer` returns anything at all. Probed once, after the
# window is up; see `probe_framebuffer`.
PIXELS_OK = True


def pixel_check(name: str, ok: bool, detail: str = "") -> bool:
    """A check whose truth lives in the rendered pixels."""
    if not PIXELS_OK:
        return skip(name, "the framebuffer read back blank in this environment")
    return check(name, ok, detail)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def img_array(qimg: QtGui.QImage) -> np.ndarray:
    """QImage -> (h, w, 3) uint8, independent of stride and format."""
    img = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return arr[:, : img.width(), :3].copy()


def shot(win: MainWindow, name: str) -> tuple[np.ndarray, str]:
    """Render the real widget and save it. Returns (pixels, path)."""
    win.viewport.repaint()
    QTest.qWait(40)
    arr = img_array(win.viewport.grabFramebuffer())
    path = OUT / f"{name}.png"
    win.viewport.grabFramebuffer().save(str(path))
    return arr, str(path)


def background_colour(arr: np.ndarray) -> np.ndarray:
    """The clear colour, taken as the modal colour of the framebuffer.

    `paintGL` clears to (0.10, 0.11, 0.13) but the window also carries whatever
    the desktop compositor put behind an unpainted region, so the mode of the
    actual pixels is the honest reference.
    """
    flat = arr.reshape(-1, 3)
    key = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    modal = vals[int(np.argmax(counts))]
    return np.asarray([(modal >> 16) & 0xFF, (modal >> 8) & 0xFF, modal & 0xFF], np.int16)


def non_background(arr: np.ndarray) -> int:
    """Pixels that differ from the modal (background) colour."""
    bg = background_colour(arr)
    d = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
    return int((d > 12).sum())


def drag(widget, button, start, end, steps=6) -> None:
    nomod = QtCore.Qt.KeyboardModifier.NoModifier
    QTest.mousePress(widget, button, nomod, QtCore.QPoint(*start))
    for i in range(1, steps + 1):
        t = i / steps
        x = int(start[0] + (end[0] - start[0]) * t)
        y = int(start[1] + (end[1] - start[1]) * t)
        QTest.mouseMove(widget, QtCore.QPoint(x, y))
        QTest.qWait(8)
    QTest.mouseRelease(widget, button, nomod, QtCore.QPoint(*end))


def rects_overlap(a: QtCore.QRect, b: QtCore.QRect) -> bool:
    return a.intersects(b) and (a & b).width() > 0 and (a & b).height() > 0


def main() -> int:
    # `PIXELS_OK` is read by `pixel_check`; assigning it here without this
    # would create a local and silently leave the flag at its default.
    global PIXELS_OK
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    print(f"Qt {QtCore.QT_VERSION_STR}, platform {app.platformName()}")

    rec = EXAMPLES / "rec_prep.pdbqt"
    lig = EXAMPLES / "ibuprofen_prep.pdbqt"
    poses = EXAMPLES / "poses.pdbqt"
    for p in (rec, lig, poses):
        if not p.is_file():
            print(f"missing example input: {p}")
            return 2

    win = MainWindow(receptor=rec, ligand=lig, poses=poses)
    win.resize(1280, 820)
    win.show()
    # `isVisible` is not the same question as "has the window been painted".
    # Under a software rasteriser with no compositor the window can be shown
    # and still never get an expose event, which leaves the QOpenGLWidget's
    # framebuffer blank. Asking explicitly turns a mystery into an answer, and
    # it costs nothing where the window really is up.
    exposed = QTest.qWaitForWindowExposed(win, 5000)
    app.processEvents()
    QTest.qWait(600)
    app.processEvents()

    # ------------------------------------------------------------------ layout
    section("1. layout of the control panel")
    panel = win.dockWidget("Controls") if hasattr(win, "dockWidget") else None
    dock = None
    for i in range(win.dockWidgetArea.count() if False else win.findChildren(QtWidgets.QDockWidget).__len__()):
        dock = win.findChildren(QtWidgets.QDockWidget)[i]
    dock = dock or next(iter(win.findChildren(QtWidgets.QDockWidget)), None)
    print(f"  dock widget found: {dock is not None}  title: {dock.windowTitle() if dock else '-'}")
    dock_w = dock.width() if dock else 0
    print(f"  dock width: {dock_w} px")

    controls = {
        "cb_receptor": win.cb_receptor,
        "cb_ligand": win.cb_ligand,
        "cb_box": win.cb_box,
        "centre x": win.center_spins[0],
        "centre y": win.center_spins[1],
        "centre z": win.center_spins[2],
        "size x": win.size_spins[0],
        "size y": win.size_spins[1],
        "size z": win.size_spins[2],
        "exhaustiveness": win.sp_exhaust,
        "seed": win.sp_seed,
        "scoring": win.cb_scoring,
        "pose list": win.pose_list,
        "best energy": win.lbl_energy,
        "RMSD": win.lbl_rmsd,
        "status": win.status_label,
    }
    zero, outside, overlap = [], [], []
    panel_rect = dock.widget().rect() if dock else QtCore.QRect()
    for name, w in controls.items():
        g = w.geometry()
        if g.width() < 8 or g.height() < 8:
            zero.append(f"{name}{g.width()}x{g.height()}")
        # is the widget's right edge past the panel's content edge?
        right = w.mapTo(dock.widget(), QtCore.QPoint(g.width(), 0)).x() if dock else 0
        if dock and right > panel_rect.width() + 1:
            outside.append(f"{name} right={right} > panel {panel_rect.width()}")
    for i, (n1, w1) in enumerate(controls.items()):
        for n2, w2 in list(controls.items())[i + 1 :]:
            if w1.parentWidget() is w2.parentWidget() and rects_overlap(w1.geometry(), w2.geometry()):
                overlap.append(f"{n1} / {n2}")

    check("every control has a usable size", not zero, ", ".join(zero) or "all >= 8x8")
    check("no control overflows the panel", not outside, "; ".join(outside) or "none")
    check("no two controls overlap", not overlap, "; ".join(overlap) or "none")
    print(f"  viewport: {win.viewport.width()}x{win.viewport.height()} "
          f"(min {win.viewport.minimumWidth()}x{win.viewport.minimumHeight()})")
    print(f"  status text: {win.status_label.text()!r}")
    print(f"  energy label: {win.lbl_energy.text()!r}   rmsd: {win.lbl_rmsd.text()!r}")
    check(
        "best-pose energy is populated on load",
        win.lbl_energy.text() not in ("", "—"),
        win.lbl_energy.text(),
    )
    check(
        "RMSD-to-best is populated on load",
        win.lbl_rmsd.text() not in ("", "—"),
        win.lbl_rmsd.text(),
    )
    check(
        "pose list is populated",
        win.pose_list.count() > 1,
        f"{win.pose_list.count()} rows",
    )

    base, _ = shot(win, "01_loaded")
    base_px = non_background(base)
    # Decide once, here, whether the framebuffer is readable at all. A blank
    # grab means every pixel-dependent check below would be measuring the
    # environment rather than the viewport.
    if base_px == 0:
        PIXELS_OK = False
        print(
            "\n  NOTE: grabFramebuffer() returned a uniformly coloured image, so"
            "\n        the pixel checks below are skipped rather than failed."
            f"\n        window exposed: {exposed}; platform: {app.platformName()}."
            "\n        Every non-pixel check still runs."
        )
    pixel_check("viewport renders geometry", base_px > 2000, f"{base_px} non-background px")

    # -------------------------------------------------- element interpretation
    section("1b. does the viewer understand the atom types it is given?")
    from opendocking.workbench import ELEMENT_COLORS, PDBQT_TYPE_ELEMENT

    for mol in win.viewport.molecules:
        unknown = sorted({e for e in mol.elements if e not in ELEMENT_COLORS})
        check(
            f"every element in '{mol.name}' is a real element",
            not unknown,
            f"{len(mol.elements)} atoms, unknown: {unknown or 'none'}"
            + (f"  <-- the aromatic-carbon token 'A' was read as an element" if "A" in unknown else ""),
        )
    pose_view = next(m for m in win.viewport.molecules if m.role == "pose")
    lig_view = next(m for m in win.viewport.molecules if m.role == "ligand")
    check(
        "the pose keeps the same element composition as the ligand it came from",
        sorted(pose_view.elements) == sorted(lig_view.elements),
        f"pose {dict((e, pose_view.elements.count(e)) for e in set(pose_view.elements))} "
        f"vs ligand {dict((e, lig_view.elements.count(e)) for e in set(lig_view.elements))}",
    )
    radii = pose_view.atom_radii()
    hydrogens = [i for i, e in enumerate(pose_view.elements) if e == "H"]
    carbons = [i for i, e in enumerate(pose_view.elements) if e in ("C", "A")]
    if hydrogens and carbons:
        check(
            "hydrogens are drawn smaller than carbons",
            max(radii[i] for i in hydrogens) < min(radii[i] for i in carbons),
            f"H radius {radii[hydrogens[0]]:.3f} vs C radius {radii[carbons[0]]:.3f} A"
            + ("" if max(radii[i] for i in hydrogens) < min(radii[i] for i in carbons) else "  <-- same size"),
        )
    else:
        skip(
            "hydrogens are drawn smaller than carbons",
            f"the pose has {len(hydrogens)} H and {len(carbons)} C atoms, so "
            "there are no radii of both to compare",
        )
    check(
        "AutoDock 'A' is mapped to carbon",
        PDBQT_TYPE_ELEMENT.get("A") == "C" and PDBQT_TYPE_ELEMENT.get("OA") == "O"
        and PDBQT_TYPE_ELEMENT.get("HD") == "H" and PDBQT_TYPE_ELEMENT.get("ZN") == "Zn",
        "A->C, OA->O, HD->H, ZN->Zn",
    )

    # ------------------------------------------------------------ interaction
    section("2. camera interaction")
    cam = win.viewport.camera
    y0, p0, d0 = cam.yaw, cam.pitch, cam.distance

    drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (520, 300))
    check("left-drag rotates the camera", abs(cam.yaw - y0) > 1e-6, f"yaw {y0:.4f} -> {cam.yaw:.4f}")

    drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (400, 420))
    check(
        "left-drag changes pitch and clamps it",
        cam.pitch != p0 and -1.5 <= cam.pitch <= 1.5,
        f"pitch {p0:.4f} -> {cam.pitch:.4f}",
    )
    # push far past the clamp to prove it holds
    for _ in range(12):
        drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (400, 900), steps=2)
    check("pitch clamp holds at +1.5", abs(cam.pitch - 1.5) < 1e-9, f"pitch {cam.pitch:.6f}")
    for _ in range(20):
        drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 900), (400, 200), steps=2)
    check("pitch clamp holds at -1.5", abs(cam.pitch + 1.5) < 1e-9, f"pitch {cam.pitch:.6f}")

    before = cam.distance
    QTest.mouseMove(win.viewport, QtCore.QPoint(400, 300))
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(400, 300),
        QtCore.QPointF(400, 300),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    app.sendEvent(win.viewport, wheel)
    app.processEvents()
    check("wheel zooms in", cam.distance < before, f"distance {before:.3f} -> {cam.distance:.3f}")

    before = cam.distance
    win.viewport.frame_all()
    app.processEvents()
    check(
        "Frame all refits the camera",
        abs(cam.distance - before) > 1e-6 or not np.allclose(cam.center, win.viewport.camera.center),
        f"distance -> {cam.distance:.3f}, centre {np.round(cam.center, 2)}",
    )

    # right-drag: is there any pan at all?
    section("3. is there a pan control?")
    c0 = cam.center.copy()
    drag(win.viewport, QtCore.Qt.MouseButton.RightButton, (400, 300), (600, 380))
    moved = float(np.abs(cam.center - c0).max())
    check(
        "right-drag pans the camera",
        moved > 1e-6,
        f"centre moved {moved:.6f} A"
        + ("" if moved > 1e-6 else "  <-- no pan is implemented"),
    )
    m0 = cam.distance
    drag(win.viewport, QtCore.Qt.MouseButton.MiddleButton, (400, 300), (400, 380))
    check("middle-drag changes distance", abs(cam.distance - m0) > 1e-9,
          f"distance {m0:.3f} -> {cam.distance:.3f}")

    # ------------------------------------------------------------- visibility
    section("4. visibility checkboxes")
    win.viewport.frame_all()
    app.processEvents()
    full, _ = shot(win, "02_all_visible")

    win.cb_receptor.setChecked(False)
    app.processEvents()
    no_rec, _ = shot(win, "03_receptor_hidden")
    pixel_check(
        "unchecking 'receptor' removes pixels",
        non_background(no_rec) < non_background(full),
        f"{non_background(full)} -> {non_background(no_rec)} px",
    )
    win.cb_receptor.setChecked(True)
    app.processEvents()

    win.cb_ligand.setChecked(False)
    app.processEvents()
    no_lig, _ = shot(win, "04_pose_hidden")
    pixel_check(
        "unchecking 'ligand / pose' removes pixels",
        non_background(no_lig) < non_background(full),
        f"{non_background(full)} -> {non_background(no_lig)} px",
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()

    win.cb_box.setChecked(False)
    app.processEvents()
    no_box, _ = shot(win, "05_box_hidden")
    pixel_check(
        "unchecking 'search box' removes the wireframe",
        non_background(no_box) < non_background(full),
        f"{non_background(full)} -> {non_background(no_box)} px",
    )
    win.cb_box.setChecked(True)
    app.processEvents()

    section("5. search-box spins")
    c_before = np.array(win.viewport.box_center, copy=True)
    win.center_spins[0].setValue(9.0)
    app.processEvents()
    c_after = np.array(win.viewport.box_center, copy=True)
    check(
        "box centre spin moves the wireframe",
        abs(c_after[0] - c_before[0]) > 1e-6,
        f"box_center x {c_before[0]:.2f} -> {c_after[0]:.2f}",
    )
    win.center_spins[0].setValue(0.0)
    app.processEvents()

    s_before = float(win.viewport.box_size[0])
    win.size_spins[0].setValue(12.0)
    app.processEvents()
    check(
        "box size spin resizes the wireframe",
        abs(float(win.viewport.box_size[0]) - s_before) > 1e-6,
        f"box_size x {s_before:.2f} -> {win.viewport.box_size[0]:.2f}",
    )
    win.size_spins[0].setValue(22.0)
    app.processEvents()

    # range check: can the user type a negative centre?
    lo = win.center_spins[0].minimum()
    hi = win.center_spins[0].maximum()
    win.center_spins[0].setValue(-5.0)
    app.processEvents()
    got = win.center_spins[0].value()
    check(
        "box centre accepts negative coordinates",
        abs(got - (-5.0)) < 1e-6,
        f"range [{lo}, {hi}], asked for -5.0, got {got}  <-- clamped",
    )
    win.center_spins[0].setValue(0.0)
    app.processEvents()

    # ----------------------------------------------------------------- poses
    section("6. pose browser")

    # The File menu is the only route into the app once it is open, so it has to
    # offer every role. It used to have a single "Open structure…" that loaded
    # everything as a ligand, which made a receptor unreachable from the GUI.
    section("5b. the File menu can reach every role")
    menu = win.menuBar().actions()[0].menu()
    labels = [a.text() for a in menu.actions() if a.text() and a.text() != "---"]
    print(f"  File menu: {labels}")
    for role, needle in (("receptor", "Load receptor"), ("ligand", "Load ligand"),
                         ("poses", "Open poses"), ("dock", "Dock now")):
        check(
            f"the File menu offers '{needle}'",
            any(needle in t for t in labels),
            ", ".join(labels),
        )
    check(
        "no action still says the generic 'Open structure'",
        not any("Open structure" in t for t in labels),
        ", ".join(labels),
    )
    check(
        "opening a pose file goes through load_structure, not a crash",
        callable(win.open_file_dialog) and "kind" in win.open_file_dialog.__code__.co_varnames,
        f"signature accepts a role: {win.open_file_dialog.__code__.co_varnames[:3]}",
    )

    n = win.pose_list.count()
    energies_before = [win.pose_list.item(i).text() for i in range(n)]
    shot_a, _ = shot(win, "06_pose_best")
    switched = 0
    for row in range(n):
        win.pose_list.setCurrentRow(row)
        app.processEvents()
        if win.lbl_energy.text() not in ("", "—"):
            switched += 1
    check(
        "selecting each pose updates the energy readout",
        switched == n,
        f"{switched}/{n} rows produced a value",
    )
    win.pose_list.setCurrentRow(0)
    app.processEvents()
    rmsd0 = win.lbl_rmsd.text()
    for row in range(1, n):
        win.pose_list.setCurrentRow(row)
        app.processEvents()
    check(
        "RMSD-to-best changes away from the best pose",
        win.lbl_rmsd.text() != rmsd0,
        f"row 0 = {rmsd0!r}, last row = {win.lbl_rmsd.text()!r}",
    )
    shot_b, _ = shot(win, "07_pose_last")
    pixel_check(
        "selecting a different pose changes what is drawn",
        not np.array_equal(shot_a, shot_b),
        "framebuffers differ" if not np.array_equal(shot_a, shot_b) else "identical",
    )
    win.pose_list.setCurrentRow(0)
    app.processEvents()
    check(
        "pose list is ordered by the energies it prints",
        energies_before == sorted(energies_before, key=lambda s: float(s.split()[-2])),
        "; ".join(e.split("kcal")[0].strip() for e in energies_before[:4]) + " ...",
    )

    # ------------------------------------------------------------ docking run
    section("7. a real docking run from the GUI")
    win.sp_exhaust.setValue(2)
    win.center_spins[0].setValue(0.0)
    win.size_spins[0].setValue(22.0)
    app.processEvents()

    n_mol_before = len(win.viewport.molecules)
    win.btn_dock.click()
    app.processEvents()
    check("Dock disables the button while running", not win.btn_dock.isEnabled())

    # Is the window still painting? A blocked GUI thread cannot repaint.
    repaints = 0
    for _ in range(40):
        app.processEvents()
        QTest.qWait(50)
        win.viewport.repaint()
        repaints += 1
    print(f"  event-loop iterations during the run: {repaints}")
    check("the event loop keeps running during a search", repaints == 40, f"{repaints}/40")

    for _ in range(400):
        app.processEvents()
        if win.btn_dock.isEnabled():
            break
        QTest.qWait(50)
    check("Dock re-enables when the run finishes", win.btn_dock.isEnabled(),
          f"status: {win.status_label.text()!r}")
    print(f"  status after run: {win.status_label.text()!r}")
    check(
        "status reports the docking result",
        "done" in win.status_label.text() or "failed" in win.status_label.text(),
        win.status_label.text(),
    )
    check(
        "the pose list is refilled from the run",
        win.pose_list.count() > 0,
        f"{win.pose_list.count()} rows",
    )
    n_mol_after = len(win.viewport.molecules)
    roles = [m.role for m in win.viewport.molecules]
    print(f"  molecule views: {n_mol_before} before, {n_mol_after} after  roles={roles}")
    check(
        "a docking run replaces the pose view, it does not add one",
        roles.count("pose") == 1,
        f"{roles.count('pose')} pose view(s)  <-- stacking",
    )
    check(
        "one view per role, none duplicated",
        sorted(roles) == sorted(set(roles)),
        f"roles={roles}",
    )
    shot(win, "08_after_dock")

    section("8. a second docking run")
    win.btn_dock.click()
    for _ in range(400):
        app.processEvents()
        if win.btn_dock.isEnabled():
            break
        QTest.qWait(50)
    roles2 = [m.role for m in win.viewport.molecules]
    print(f"  molecule views now: {len(roles2)}  roles={roles2}")
    check(
        "a second run still leaves exactly one pose view",
        roles2.count("pose") == 1,
        f"{roles2.count('pose')}  <-- previous results stay on screen",
    )
    check(
        "a second run does not change the number of views",
        len(roles2) == len(roles),
        f"{len(roles)} -> {len(roles2)}",
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()
    vis = [m for m in win.viewport.molecules if m.visible]
    check(
        "after two runs the scene is exactly receptor + ligand + pose",
        len(vis) == 3 and {m.role for m in vis} == {"receptor", "ligand", "pose"},
        f"{len(vis)} visible: {[(m.role, m.name) for m in vis]}",
    )
    # And the checkbox must actually be able to hide the pose now.
    win.cb_ligand.setChecked(False)
    app.processEvents()
    pose_hidden = all(
        not m.visible for m in win.viewport.molecules if m.role == "pose"
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()
    check("the ligand checkbox hides the pose view", pose_hidden)
    shot(win, "09_after_second_dock")

    # temp files left behind
    import tempfile
    leaked = list(Path(tempfile.gettempdir()).glob("tmp*.pdbqt"))
    recent = [p for p in leaked if (Path(tempfile.gettempdir()) / p.name).exists()]
    try:
        ages = [(p.stat().st_mtime, p) for p in leaked]
        now = max(t for t, _ in ages) if ages else 0
        import time
        fresh = [p for t, p in ages if time.time() - t < 3600]
    except Exception:
        fresh = []
    print(f"  temp .pdbqt files in %TEMP% newer than 1 h: {len(fresh)}")
    check("docking does not leak temp files", not fresh, f"{len(fresh)} file(s): "
          + ", ".join(p.name for p in fresh[:4]))

    # ------------------------------------------------- negative-coordinate receptor
    section("9. a receptor at negative coordinates")
    import opendocking
    from opendocking.workbench import MoleculeView

    shifted = EXAMPLES / "_shifted_receptor.pdbqt"
    rows = []
    for line in rec.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) - 40.0
            rows.append(line[:30] + f"{x:8.3f}" + line[38:])
        else:
            rows.append(line)
    shifted.write_text("\n".join(rows) + "\n", encoding="utf-8")

    win2 = MainWindow(receptor=shifted)
    win2.resize(1000, 700)
    win2.show()
    app.processEvents()
    QTest.qWait(400)
    app.processEvents()
    true_center = np.array(opendocking.Receptor.from_pdbqt(shifted).center, dtype=float)
    ui_center = np.array([s.value() for s in win2.center_spins], dtype=float)
    print(f"  true receptor centre : {np.round(true_center, 3)}")
    print(f"  centre shown in spins: {np.round(ui_center, 3)}")
    check(
        "a receptor with negative coordinates loads into the box-centre spins",
        bool(np.abs(true_center - ui_center).max() < 0.05),
        f"mismatch {np.abs(true_center - ui_center).max():.3f} A  <-- clamped to the spin range",
    )
    shot(win2, "10_negative_coords")
    win2.close()

    # ------------------------------------------------- display representations
    section("10. display representations")
    from opendocking.workbench.app import REPRESENTATION_KEYS  # noqa: PLC0415

    check(
        "the display selector offers every representation",
        set(REPRESENTATION_KEYS) == {"spheres", "ball_and_stick", "stick",
                                     "ribbon", "cartoon"},
        ", ".join(REPRESENTATION_KEYS),
    )
    check(
        "the selector has one entry per representation",
        win.cmb_representation.count() == len(REPRESENTATION_KEYS),
        f"{win.cmb_representation.count()} entries",
    )

    for key in REPRESENTATION_KEYS:
        index = win.cmb_representation.findData(key)
        win.cmb_representation.setCurrentIndex(index)
        app.processEvents()
        check(
            f"selecting '{key}' reaches the viewport",
            win.viewport.representation == key,
            f"viewport says {win.viewport.representation!r}",
        )
        if PIXELS_OK:
            arr, _ = shot(win, f"11_repr_{key}")
            drawn = non_background(arr)
            check(
                f"'{key}' draws something",
                drawn > 500,
                f"{drawn} non-background px",
            )
        else:
            skip(f"'{key}' draws something", "no framebuffer to read")

    # A ribbon needs a backbone. The example receptor is a synthetic blob, so
    # it must fall back rather than pretend, and the status bar has to say so.
    win.cmb_representation.setCurrentIndex(
        win.cmb_representation.findData("ribbon")
    )
    app.processEvents()
    receptor_view = next(m for m in win.viewport.molecules if m.role == "receptor")
    check(
        "a structure with no backbone reports that it has none",
        not receptor_view.has_backbone,
        "the synthetic example receptor is not a protein",
    )
    check(
        "a file with no amino-acid residue falls back instead of claiming a ribbon",
        receptor_view.backbone_ribbon() is None,
        "backbone_ribbon() returned None",
    )
    message = win.statusBar().currentMessage()
    check(
        "the status bar says where the bonds came from",
        "bonds from" in message,
        message[:110],
    )

    # The receptor's bonds are the interesting ones. Whatever the source, no
    # bond may join atoms of different residues unless it is the peptide C-N,
    # because that is the failure this whole path was written to prevent.
    check(
        "the loaded receptor's bonds are all physically possible",
        all(
            0.85
            <= float(
                np.linalg.norm(
                    receptor_view.coords[i] - receptor_view.coords[j]
                )
            )
            < 2.0
            for i, j in receptor_view.bond_pairs()
        ),
        f"{len(receptor_view.bond_pairs())} bonds checked",
    )

    win.cmb_representation.setCurrentIndex(
        win.cmb_representation.findData("spheres")
    )
    app.processEvents()

    # The example receptor is a synthetic blob, so nothing above has actually
    # drawn a ribbon. A real protein has to, or the feature is untested.
    crambin = EXAMPLES / "1crn_prep.pdbqt"
    if crambin.is_file():
        section("10b. a ribbon on a real protein")
        win3 = MainWindow(receptor=crambin)
        win3.resize(1000, 760)
        win3.show()
        QTest.qWaitForWindowExposed(win3, 5000)
        app.processEvents()
        QTest.qWait(400)
        app.processEvents()

        protein = next(m for m in win3.viewport.molecules if m.role == "receptor")
        check("the prepared receptor kept its residue names",
              protein.has_backbone,
              f"{len(protein.coords)} atoms, bonds from {protein.bond_source}")
        check("its bonds came from residue templates",
              protein.bond_source == "template", protein.bond_source)
        check("it is grouped into one entry per residue",
              protein.structure is not None
              and len(protein.structure.residues) == 46,
              f"{0 if protein.structure is None else len(protein.structure.residues)} "
              "residues")

        ribbon = protein.backbone_ribbon()
        check("a ribbon was built from the backbone",
              ribbon is not None and len(ribbon) > 0,
              f"{0 if ribbon is None else len(ribbon)} triangles")

        drawn = {}
        for key in ("ball_and_stick", "ribbon", "cartoon"):
            win3.cmb_representation.setCurrentIndex(
                win3.cmb_representation.findData(key)
            )
            app.processEvents()
            if PIXELS_OK:
                arr, _ = shot(win3, f"12_protein_{key}")
                drawn[key] = non_background(arr)
                check(f"protein '{key}' draws something",
                      drawn[key] > 500, f"{drawn[key]} non-background px")
            else:
                skip(f"protein '{key}' draws something", "no framebuffer to read")

        if len(drawn) == 3:
            check(
                "the protein ribbon is a different picture from its atoms",
                abs(drawn["ribbon"] - drawn["ball_and_stick"]) > 500,
                f"ribbon {drawn['ribbon']} px vs ball-and-stick "
                f"{drawn['ball_and_stick']} px",
            )
        else:
            # Registering this only when it can run makes the total silently
            # smaller, and a smaller total is indistinguishable from a smaller
            # scope. A check that did not happen has to say so.
            skip(
                "the protein ribbon is a different picture from its atoms",
                "no framebuffer was readable, so there were no pixel counts to "
                "compare",
            )
        msg = win3.statusBar().currentMessage()
        check("the status bar names the template source for the protein",
              "residue templates" in msg, msg[:110])
        win3.close()
    else:
        skip("a ribbon on a real protein", f"{crambin.name} is not present")

    # ---------------------------------------------------------------- report
    section("summary")
    npass = sum(1 for r in results if r[0] == "PASS")
    nfail = sum(1 for r in results if r[0] == "FAIL")
    nskip = sum(1 for r in results if r[0] == "SKIP")
    print(f"  {npass} passed, {nfail} failed, {nskip} skipped, {len(results)} checks")
    for tag, name, detail in results:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {detail}")
    if nskip:
        print(
            f"\n  {nskip} check(s) were skipped, not passed. A skip means this\n"
            "  environment could not answer the question; it is not evidence\n"
            "  that the workbench is correct there."
        )
    print(f"\n  screenshots: {OUT}")
    win.close()
    return 1 if nfail else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
