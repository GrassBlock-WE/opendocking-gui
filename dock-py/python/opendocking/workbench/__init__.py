"""Interactive 3-D docking workbench (PyQt6 + ModernGL).

A single window with a ModernGL viewport and Qt controls:

* **Receptor / ligand / pose** are drawn as instanced spheres and sticks;
* the **search box** is draggable, so the grid placement — the single most
  consequential choice in a docking run — is made by pointing at the pocket
  rather than by typing coordinates;
* a live **energy readout** scores the current pose, and a **pose browser**
  steps through the ranked results.

# Design notes

The viewport uses a *shaded* instanced-sphere renderer rather than a
ball-and-stick model: spheres read correctly at any zoom level, need no mesh
generation, and let a whole receptor (tens of thousands of atoms) render in one
draw call. Bonds are drawn as line segments on top, coloured by interaction
class, which is what makes hydrogen bonds legible.

Everything heavy happens off the GUI thread: docking runs in a `QThread` and
results are delivered through a Qt signal, so the window never freezes.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

#: The public surface of the workbench package.
#:
#: **`Workbench` was here and was removed, and the reason is not tidiness.**
#: It was listed but neither defined nor imported, so
#: `from opendocking.workbench import Workbench` raised ImportError and
#: `from opendocking.workbench import *` raised AttributeError -- a promise
#: this module made and could not keep. `scripts/unreached_product_check.py`
#: carried it as a skip for exactly that reason.
#:
#: **It could not be added either, and that is the part worth writing down.**
#: A census of the whole tree found `Workbench` as an *identifier* in three
#: places and nowhere else: this line, `docs/API.md`'s import example, and the
#: two gates that report the defect. Every other hit in the repository is the
#: window *title* "Open Docking Workbench". So there is no caller to wire it
#: to, and the name has no referent: the window class is `app.MainWindow` and
#: the entry point is `launch`, a function.
#:
#: Making the name real would mean an eager `from .app import MainWindow as
#: Workbench` here, and that breaks a property `cli.py` states in its own
#: comment: this package "imports only the standard library plus numpy at
#: module scope and defers the GUI import to a call inside `launch", so that
#: `from .workbench import launch` cannot raise. An import of `app` at module
#: scope would take PyQt6 and moderngl as hard requirements of `odcli sites`,
#: `odcli info` and `odcli workbench` -- paths that have no window, no event
#: loop and no slots. Trading a broken promise for a broken CLI is not a fix.
#:
#: `docs/API.md` still shows the name in its import line. That file is not this
#: module's, so the sentence there is left for its owner.
__all__ = ["launch", "MoleculeView", "Camera"]


def _require_gui():
    """Import the GUI stack, with a clear message when it is missing.

    The one place the GUI stack is tested, for every entry point. `odcli
    workbench` and `odgui` both reach this through `launch`, so a user who
    cannot import PyQt6 gets the same sentence whichever command they typed --
    which was not true before: `odcli` had its own, unreachable, copy of this
    message, and `workbench.launcher._preflight` tested the wrong import (it
    imported `launch`, which never needs Qt) and so never reported anything.
    """
    try:
        from PyQt6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets  # noqa: F401
        import moderngl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the workbench needs PyQt6 and moderngl. Install them with:\n"
            "    pip install PyQt6 moderngl numpy-stl"
        ) from exc


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

#: CPK-ish colours, keyed by element symbol.
ELEMENT_COLORS: dict[str, tuple[float, float, float]] = {
    "C": (0.55, 0.55, 0.58),
    "N": (0.20, 0.35, 0.95),
    "O": (0.90, 0.20, 0.20),
    "S": (0.90, 0.80, 0.25),
    "P": (0.95, 0.55, 0.20),
    "F": (0.40, 0.90, 0.45),
    "Cl": (0.35, 0.85, 0.35),
    "Br": (0.65, 0.35, 0.20),
    "I": (0.55, 0.20, 0.65),
    "H": (0.95, 0.95, 0.95),
    "Zn": (0.60, 0.60, 0.70),
    "M": (0.60, 0.60, 0.70),
}

#: AutoDock PDBQT atom type -> element symbol.
#:
#: The type is **not** always "element plus a polar suffix". `A` is aromatic
#: *carbon*, so the obvious "leading alphabetic character" rule turns it into a
#: fictitious element: it misses the colour table, falls back to the view
#: colour, and a benzene ring renders as if it were a different element from
#: the aliphatic carbons beside it. The aromatic half of ibuprofen was drawn
#: green while the rest of the same molecule was grey. Same for the metals,
#: which are written `ZN`/`Mg`/`Ca` and are not single letters.
PDBQT_TYPE_ELEMENT: dict[str, str] = {
    "C": "C", "A": "C", "G0": "C", "CG0": "C",
    "N": "N", "NA": "N", "NS": "N", "NC": "N", "N4": "N",
    "O": "O", "OA": "O",
    "S": "S", "SA": "S",
    "H": "H", "HD": "H", "HS": "H",
    "P": "P", "F": "F", "I": "I",
    "CL": "Cl", "BR": "Br",
    "MG": "Mg", "MET": "Mg", "ZN": "Zn", "CA": "Ca", "FE": "Fe", "MN": "Mn",
    "DU": "Du", "DUX": "Du",
}

#: Per-element radius as a *multiplier* of `MoleculeView.radius`, not an
#: absolute radius. Drawing every atom at the same size makes a polar hydrogen
#: look like a carbon and inflates the apparent size of the molecule; the
#: multiplier keeps the overall scale -- and therefore every framing calculation
#: -- unchanged while making the difference legible.
ELEMENT_RADIUS_SCALE: dict[str, float] = {
    "H": 0.35, "Du": 0.40,
    "C": 1.00, "N": 0.95, "O": 0.90, "S": 1.10, "P": 1.05,
    "F": 0.75, "Cl": 1.00, "Br": 1.05, "I": 1.10,
    "Mg": 1.00, "Zn": 1.00, "Ca": 1.00, "Fe": 1.00, "Mn": 1.00,
}

#: Colours used to distinguish the ligand from the receptor.
COLOR_RECEPTOR = (0.62, 0.66, 0.72)
COLOR_LIGAND = (0.35, 0.75, 0.95)
COLOR_BEST_POSE = (0.40, 0.90, 0.45)
COLOR_BOX = (0.95, 0.75, 0.20)

#: The selected pocket's own grid points, drawn as a translucent cloud. Chosen
#: to be unmistakable against both the grey protein and the yellow box: a warm
#: magenta reads as neither, and the cloud is usually *inside* the box, so a
#: colour close to the box's would vanish into it.
COLOR_POCKET = (0.85, 0.35, 0.75)

#: Residue names that are not a residue: crystal water, heavy water, the
#: placeholders the writers emit for ligands, and the handful of cryoprotectant
#: and buffer additives that turn up in crystal structures often enough to
#: matter. Excluded from residue labelling, and so from pocket lining and
#: contact attribution. See `MoleculeView.residue_labels`.
NON_RESIDUE_LABELS = frozenset({
    "HOH", "DOD", "WAT", "H2O",  # water, heavy water, some writers' spelling
    "UNL", "REC", "UNK", "LIG",  # this project's own placeholders
    "GOL", "EDO", "PEG", "PG4", "PGE", "1PE",  # ethylene glycol and PEG fragments
    "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE",
    "ACT", "ACY", "DMS", "TRS", "FMT", "IPA", "BME", "CIT", "NO3",
    "MES", "EPE", "TLA", "FLC", "GOL", "MRD",
})

#: Interaction lines, by the class `contacts.find_contacts` assigns.
#:
#: The colours are chosen against the two things these lines are drawn over: a
#: grey protein and a yellow search box. Hydrophobic contacts were grey to begin
#: with, which made 16 of 48 lines in a real pose invisible — a contact you
#: cannot see is not reported to anyone, however correctly it was computed.
#: Hydrogen bonds are near-white rather than yellow so they cannot be mistaken
#: for the box.
COLOR_CONTACT = {
    "hbond": (1.00, 0.97, 0.72),
    "polar": (0.30, 0.82, 1.00),
    "hydrophobic": (0.82, 0.45, 0.95),
    "close": (0.45, 0.52, 0.62),
}

#: The selected pair, drawn as one solid rod between the two atoms the pair
#: table's selected row names.
#:
#: A **solid** rod, where every other interaction line is dashed, and a colour
#: chosen so that it can be told from everything else on screen. Both halves are
#: load-bearing and they do different jobs: the eye reads the solidity, and the
#: separation from the palette is what lets an instrument name the marker.
#:
#: The palette it has to stay out of is every element colour plus the five
#: scene colours, and the first attempt at this failed on it -- orange-red was
#: inside nitrogen's `(0.20, 0.35, 0.95)` and phosphorus's `(0.95, 0.55, 0.20)`
#: neighbourhood, and a gate counting "orange" pixels would have been counting
#: atoms. Saturated magenta is what is left: it is the only entry in the
#: palette whose red and blue are both near full while its green is near zero,
#: and that is a band nothing else reaches. The nearest neighbour is the
#: hydrophobic contact's orchid `(0.82, 0.45, 0.95)`, whose green is 115 against
#: this one's 26 -- which is also why the rod is solid and not dashed.
#:
#: **What reaches the framebuffer is not this tuple, and that was measured
#: rather than assumed.** The rod is drawn with the *line* program, which fogs
#: like every other line in the viewer, so on 1crn/biotin at the framing `F`
#: gives -- all six representations, from the marker's own pixels and not from a
#: colour search -- the colour on screen is BGR **(163, 26, 159)**, not the
#: nominal `(242, 26, 242)`. The shader's own term predicts it: with the fog
#: range at 7.6-29.1 A and the pair at 25.2 A, the mix factor is 0.368, and
#: `mix((242, 26, 242), (25, 28, 33), 0.368)` is `(162, 27, 165)`.
#:
#: **The gap is a function of the camera distance, and that is the part worth
#: carrying.** The fog range is not a constant of the scene: `paintGL` scales it
#: by `max(camera.distance * 0.55, 8.0)`, so pulling the camera in pulls the fog
#: in with it and the same rod arrives progressively less blended. Measured on
#: 1crn/biotin, pose row 0, pair row 0, spheres -- the same marker, the same
#: tolerance of 30, the same six representations:
#:
#: ==================  ==========  ==========================================
#: camera distance     marker px   pixels within 30 of the nominal colour
#: ==================  ==========  ==========================================
#: 24.2-30.6 A (F)      271-425    **0 in all six representations**
#: 8.125 A              3 722      3 448
#: 6.0 A                7 155      6 769
#: ==================  ==========  ==========================================
#:
#: So "a purity count reads 0" is a statement about the *framing* the count was
#: taken at, not about the marker, and it is easy to over-read as the latter.
#: The count is not a constant zero: it moves by three orders of magnitude
#: across the camera distances this product actually offers, and on a second
#: fixture it reads 0 1707-2744 px in five of six representations inside a
#: *single* cell, because the five re-frame the camera to slightly different
#: distances. A gate built on it would be red or green according to the zoom.
#:
#: Two consequences, and they point opposite ways.
#:
#: * A **purity count against the nominal colour cannot see the marker
#:   reliably at any framing** -- at 24 A it reads 0 on a frame where the marker
#:   plainly covers 271-425 px, and at 6 A it reads 6 769 on a frame where the
#:   marker is the same object. The suite measures the marker as a *difference*
#:   against the same camera with the draw suppressed, and locates it by
#:   projecting the two atoms; neither depends on this tuple arriving unblended,
#:   and neither depends on where the camera is.
#: * The **separation argument survives the fog, and by a margin worth naming.**
#:   The fog colour's green is 28 against this colour's 26, so fog pulls red
#:   and blue down together and leaves green exactly where it was -- and green
#:   is the channel that separates this from the pocket cloud `(0.85, 0.35,
#:   0.75)`, at 89, and from nitrogen at 153. The neighbour distances quoted
#:   above are distances between *nominal* colours, so they are the right thing
#:   to quote for choosing a colour, but the discrimination they rest on is
#:   carried by the one channel the fog does not touch.
COLOR_PAIR = (0.95, 0.10, 0.95)

#: The same classes, for the table and the legend.
CONTACT_LABELS = {
    "hbond": "H-bond",
    "polar": "polar",
    "hydrophobic": "hydrophobic",
    "close": "other",
}


def _unit_sphere(segments: int = 16, rings: int = 12) -> np.ndarray:
    """Vertices and per-vertex normals of a unit sphere, for instancing."""
    verts: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            n = (
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            )
            verts.append(n)
            normals.append(n)
    return np.asarray(verts, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def _icosphere(subdivisions: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """A unit icosahedron, optionally subdivided, with face normals.

    Icosahedra distribute vertices far more evenly than a UV sphere, so an
    instanced sphere looks round at a fraction of the triangle count — which
    matters when the receptor is 30 000 atoms.
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.asarray(
        [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int32,
    )
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_faces = []
        # A list of new midpoints, appended to `verts` only after the level is
        # finished. Appending inside `midpoint` would need `nonlocal`/`global`
        # gymnastics with numpy's rebinding semantics, and this way the index
        # arithmetic stays obvious.
        added: list[np.ndarray] = []

        def midpoint(a: int, b: int, added=added) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (verts[a] + verts[b]) / 2.0
                m /= np.linalg.norm(m)
                added.append(m)
                cache[key] = len(verts) + len(added) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        if added:
            verts = np.vstack([verts, np.asarray(added)])
        faces = np.asarray(new_faces, dtype=np.int32)
    return verts.astype(np.float32), faces.astype(np.uint32)


def _parse_pdbqt_atoms(text: str) -> tuple[np.ndarray, list[str]]:
    """Extract coordinates and element symbols from PDBQT text.

    Atoms are returned in **serial order within each MODEL**, not in file
    order. A docked pose is written with the rigid ``ROOT`` cluster first and
    the flexible ``BRANCH`` clusters after it, so file order is not the order of
    the input ligand; the serial column is. Returning serial order is what lets
    a written pose be compared element-by-element against the ligand it came
    from.
    """
    models: list[list[tuple[int, tuple[float, float, float], str]]] = [[]]

    def current() -> list[tuple[int, tuple[float, float, float], str]]:
        return models[-1]

    for line in text.splitlines():
        record = line.split(maxsplit=1)[0] if line.strip() else ""
        if record == "MODEL" and models[-1]:
            # A new model only starts once the previous one has atoms, so a
            # stray "MODEL" line cannot produce an empty frame.
            models.append([])
            continue
        if record not in ("ATOM", "HETATM"):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = len(current())
        token = line[77:79].strip() or line.split()[-1]
        # Explicit table first: `A` is aromatic carbon, and a "leading
        # alphabetic character" rule would make it a fictitious element. Only
        # fall back to that rule for types the table has never seen.
        element = PDBQT_TYPE_ELEMENT.get(token.upper())
        if element is None:
            stripped = "".join(ch for ch in token if ch.isalpha())
            element = stripped[:1].upper() or "C"
        current().append((serial, xyz, element))

    frames = [m for m in models if m]
    if not frames:
        return np.zeros((0, 3), dtype=np.float32), []
    # Multiple models: concatenate in model order, each sorted by serial.
    rows: list[tuple[int, tuple[float, float, float], str]] = []
    for frame in frames:
        rows.extend(sorted(frame, key=lambda r: r[0]))
    return (
        np.asarray([r[1] for r in rows], dtype=np.float32),
        [r[2] for r in rows],
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MoleculeView:
    """One structure the viewport can draw."""

    name: str
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    elements: list[str] = field(default_factory=list)
    bonds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    color: tuple[float, float, float] = COLOR_LIGAND
    radius: float = 0.28
    visible: bool = True
    opacity: float = 1.0
    # What this view *is*, independent of where it was read from: "receptor",
    # "ligand" or "pose". Identity has to be explicit because the pose view is
    # built from a temp file whose name changes on every run, so matching on
    # `name` cannot tell the current pose from the previous one -- which is how
    # old results stayed on screen and the "ligand" checkbox stopped hiding them.
    role: str = "ligand"
    #: How the bonds were obtained: "declared", "template" or "distance".
    #: Shown in the status bar, because a bond order nobody can vouch for
    #: should not look the same as one that came out of the file.
    bond_source: str = "distance"
    #: Anything the structure layer wants the user to know about this file.
    warnings: list[str] = field(default_factory=list)
    #: True when the structure has an amino-acid backbone, which is what the
    #: ribbon representations need.
    has_backbone: bool = False
    #: The parsed structure, kept so the backbone can be re-read for the ribbon
    #: without parsing the file a second time.
    structure: object = None

    @classmethod
    def from_pdbqt(
        cls,
        path: str | Path,
        name: str,
        color,
        radius: float,
        role: str = "ligand",
    ) -> "MoleculeView":
        from .structure import parse_structure

        parsed = parse_structure(Path(path).read_text(encoding="utf-8"), name)
        source = "declared" if parsed.declared else "distance"
        if not parsed.declared and parsed.is_protein():
            source = "template"
        return cls(
            name=name,
            coords=np.asarray(parsed.coords(), np.float32).reshape(-1, 3),
            elements=parsed.elements(),
            bonds=np.asarray(parsed.bond_pairs(), np.int32).reshape(-1, 2),
            color=color,
            radius=radius,
            role=role,
            bond_source=source,
            warnings=list(parsed.warnings),
            has_backbone=parsed.is_protein(),
            structure=parsed,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        name: str,
        color,
        radius: float,
        role: str = "ligand",
    ) -> "MoleculeView":
        from .structure import parse_structure

        parsed = parse_structure(text, name)
        source = "declared" if parsed.declared else "distance"
        if not parsed.declared and parsed.is_protein():
            source = "template"
        return cls(
            name=name,
            coords=np.asarray(parsed.coords(), np.float32).reshape(-1, 3),
            elements=parsed.elements(),
            bonds=np.asarray(parsed.bond_pairs(), np.int32).reshape(-1, 2),
            color=color,
            radius=radius,
            role=role,
            bond_source=source,
            warnings=list(parsed.warnings),
            has_backbone=parsed.is_protein(),
            structure=parsed,
        )

    def atom_colors(self) -> np.ndarray:
        """Per-atom RGB, from the element table unless overridden."""
        if not self.elements:
            return np.tile(np.asarray(self.color, np.float32), (len(self.coords), 1))
        return np.asarray(
            [ELEMENT_COLORS.get(e, self.color) for e in self.elements], dtype=np.float32
        )

    def atom_radii(self) -> np.ndarray:
        """Per-atom draw radius in Å, `radius` scaled by element."""
        if not self.elements:
            return np.full(len(self.coords), float(self.radius), np.float32)
        return np.asarray(
            [float(self.radius) * ELEMENT_RADIUS_SCALE.get(e, 1.0) for e in self.elements],
            dtype=np.float32,
        )

    def residue_labels(self) -> list[str]:
        """Residue label per atom, ``""`` where there is nothing to call it.

        ``"THR 23"``, or ``"ASN 12A"`` with a chain. This is the one place the
        format is decided: the contact table and the pocket lining both have to
        name a residue the same way or the same site reads as two different
        ones in two panels of the same window.

        ``UNL`` and ``REC`` are the placeholders the writers use for "this is
        not a named residue", and they get ``""`` -- naming a ligand
        ``"UNL 1"`` in one panel and nothing in another is worse than a blank.

        Crystal waters and the buffer additives in
        `NON_RESIDUE_LABELS` get ``""`` as well. Measured across six
        proteins, streptavidin's top site listed ``lined by HOH 354A, HOH
        361A`` alongside two real residues: a water is a lattice artefact, it
        says nothing about the shape of the protein, and on that structure
        the waters outnumbered the residues in the lining list outright, so
        the useful part of the answer was buried under its own noise.

        **A cofactor is a different case and is kept.** ``HEM 155A`` comes
        back as a label, deliberately. A water is where the crystal happened
        to put a solvent molecule; a cofactor is real chemistry that is part
        of the receptor a drug has to deal with, and "this pocket is lined by
        the haem" is an answer worth having. Writing the exclusion as one
        flat list would have thrown both away.

        That distinction used to be documented here by using ``HEM 155A`` as
        an example of the noise being removed -- in a sentence that also
        called a cofactor "a ligand someone put there on purpose, and not a
        wall of the protein". The code never did that, ``pockets_check.py``
        asserts the opposite on purpose, and a docstring that argues against
        its own function is worse than one that says nothing: it is the part
        a reader trusts. So the example is HOH because that is what the list
        actually removes, and the cofactor case is stated as the deliberate
        exception it is.
        """
        structure = self.structure
        atoms = getattr(structure, "atoms", None) if structure is not None else None
        out: list[str] = []
        for i in range(len(self.coords)):
            rec = atoms[i] if atoms is not None and i < len(atoms) else None
            resname = (getattr(rec, "resname", "") or "").strip().upper() if rec else ""
            if not resname or resname in NON_RESIDUE_LABELS:
                out.append("")
                continue
            chain = (getattr(rec, "chain", "") or "").strip() if rec else ""
            out.append(f"{resname} {getattr(rec, 'resid', 0)}{chain}")
        return out

    def bond_pairs(self) -> np.ndarray:
        """The bonds this structure was given, as an ``(n_bonds, 2)`` array.

        These come from :mod:`opendocking.workbench.structure`, which reads them
        out of `ROOT`/`BRANCH` records, from amino-acid templates, or -- for a
        flat ligand with neither -- from a covalent-radius rule that is audited
        and reported. There is deliberately no distance fallback here any more:
        a viewer that silently re-derives bonds cannot tell the user which
        bonds it is sure about.
        """
        return np.asarray(self.bonds, dtype=np.int32).reshape(-1, 2)

    def bond_segments(self) -> np.ndarray:
        """Bond endpoints as an ``(n_bonds, 2, 3)`` array."""
        pairs = self.bond_pairs()
        if len(pairs) == 0 or len(self.coords) < 2:
            return np.zeros((0, 2, 3), np.float32)
        return self.coords[pairs]

    def center(self) -> np.ndarray:
        """Centroid, or the origin for an empty structure."""
        if len(self.coords) == 0:
            return np.zeros(3, np.float32)
        return self.coords.mean(axis=0)

    def _backbone_guide_and_sides(self):
        """``(guide points, side vectors, states)`` for a backbone sweep.

        The shared front half of `backbone_ribbon` and `cartoon`, because the
        two draw the same trace and differ only in what they sweep along it. The
        side vectors are the CA-to-CB direction, which is what stops a flat
        ribbon twisting about its own axis as the chain runs; glycine has no CB,
        so it falls back to the N-to-C bisector, which points the same way for
        every residue.
        """
        from .structure import secondary_structure

        if self.structure is None or not self.has_backbone:
            return None
        trace = self.structure.backbone()
        if len(trace) < 2:
            return None
        atoms = self.structure.atoms
        guide = np.asarray([atoms[ca].xyz for _, ca, _ in trace], dtype=np.float64)

        sides = []
        for n, ca, c in trace:
            side = None
            for cand in self.structure.residues:
                if ca in cand.atoms:
                    cb = next(
                        (i for i in cand.atoms if atoms[i].name == "CB"), None
                    )
                    if cb is not None:
                        side = np.asarray(atoms[cb].xyz) - np.asarray(atoms[ca].xyz)
                    else:
                        # Glycine: the N-CA-C bisector stands in for CB.
                        a = np.asarray(atoms[n].xyz) - np.asarray(atoms[ca].xyz)
                        b = np.asarray(atoms[c].xyz) - np.asarray(atoms[ca].xyz)
                        side = a + b
                    break
            sides.append(side if side is not None else np.array([0.0, 0.0, 1.0]))
        states = secondary_structure(trace, atoms)
        return guide, np.asarray(sides), states

    def backbone_ribbon(self, **kwargs):
        """A swept ribbon along the backbone, or ``None`` if there is none.

        The `ribbon` representation's own geometry: one swept quad strip whose
        width follows the classified secondary structure but whose cross-section
        does not turn about the path. The `cartoon` mode builds a different mesh
        from the same classification -- see `cartoon`.
        """
        from .geometry import ribbon as build_ribbon

        parts = self._backbone_guide_and_sides()
        if parts is None:
            return None
        guide, sides, states = parts
        return build_ribbon(guide, sides, states, **kwargs)

    def cartoon(self, **kwargs):
        """The cartoon mesh: a twisted ribbon for helices, arrows for strands.

        The same trace, the same side vectors and the same per-residue states as
        `backbone_ribbon`, swept into a different shape: the cross-section turns
        about the path through a helix and stays flat through a strand, and a
        strand's last residues flare into an arrowhead. `None` when there is no
        backbone, so a ligand falls back the same way it does for a ribbon.
        """
        from .cartoon_geometry import cartoon as build_cartoon

        parts = self._backbone_guide_and_sides()
        if parts is None:
            return None
        guide, sides, states = parts
        return build_cartoon(guide, sides, states, **kwargs)


@dataclass
class Camera:
    """A minimal orbit camera."""

    center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    distance: float = 40.0
    yaw: float = 0.6
    pitch: float = 0.4
    fov: float = 45.0

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal ``(right, up, forward)`` of the current view direction.

        Exposed because panning has to move the look-at point *in the view
        plane*: translating along world x would slide a molecule unpredictably
        as the camera orbits.
        """
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        forward = np.asarray([sy * cp, -cy * cp, -sp], np.float32)
        up_hint = np.asarray([0.0, 0.0, 1.0], np.float32)
        right = np.cross(forward, up_hint)
        n = np.linalg.norm(right)
        if n < 1e-6:
            # Looking straight up or down: `forward` is parallel to the hint and
            # the cross product vanishes, so pick an arbitrary but valid axis.
            right = np.asarray([1.0, 0.0, 0.0], np.float32)
        else:
            right = right / n
        up = np.cross(right, forward)
        return right.astype(np.float32), up.astype(np.float32), forward

    def right(self) -> np.ndarray:
        return self.basis()[0]

    def up(self) -> np.ndarray:
        return self.basis()[1]

    def view_matrix(self) -> np.ndarray:
        """4×4 view matrix, right-handed, looking down −z."""
        right, up, forward = self.basis()
        eye = self.center - forward * self.distance
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -forward
        m[0, 3] = -float(right @ eye)
        m[1, 3] = -float(up @ eye)
        m[2, 3] = float(forward @ eye)
        return m

    def projection(self, aspect: float, znear: float = 0.1, zfar: float = 2000.0) -> np.ndarray:
        """4×4 perspective projection."""
        f = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / max(aspect, 1e-3)
        m[1, 1] = f
        m[2, 2] = (zfar + znear) / (znear - zfar)
        m[2, 3] = 2.0 * zfar * znear / (znear - zfar)
        m[3, 2] = -1.0
        return m


def launch(
    receptor: str | Path | None = None,
    ligand: str | Path | None = None,
    poses: str | Path | None = None,
) -> int:
    """Open the workbench window.

    Returns the Qt exit code, or 3 if the GUI dependencies are missing.

    The diagnostic goes to **stderr**. It is a diagnostic, and a diagnostic on
    stdout is indistinguishable from a result: `odcli workbench > out.txt` would
    otherwise capture the failure as if it were the command's output. Every
    other `odcli` failure already went to stderr, and `odgui` reached this same
    branch, so this one line made the two entry points disagree about the same
    event.

    The *wording* comes from `launcher._describe_missing` rather than from here,
    so `odcli workbench`, `odgui` and `odgui --check` produce one sentence for
    one event instead of three. The import is inside the `except` block on
    purpose: `launcher` is the console script for this package and importing it
    at module scope would make the two modules circular.
    """
    try:
        _require_gui()
    except ImportError as exc:
        from .launcher import _describe_missing

        print(_describe_missing(exc), file=sys.stderr)
        return 3
    from .app import run

    return run(receptor, ligand, poses)
