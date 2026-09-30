"""Open Docking (``opendocking``) — a modern, open-source molecular docking toolkit.

The package is three layers:

* :mod:`opendocking.core` — thin, typed wrappers over the compiled Rust engine.
* :mod:`opendocking.prep` — chemistry-aware structure preparation with RDKit.
* :mod:`opendocking.workbench` — an interactive PyQt6 + ModernGL 3-D viewer.

Typical use::

    from opendocking import Receptor, Ligand, GridBox, dock

    receptor = Receptor.from_pdbqt("rec.pdbqt")
    box_ = GridBox.from_center_size(receptor.center, (22., 22., 22.))
    maps = receptor.precalculate(box_)
    ligand = Ligand.from_pdbqt("lig.pdbqt")
    result = dock(ligand, maps, exhaustiveness=8, num_modes=9)
    result.write_pdbqt("poses.pdbqt")

The command line offers the same operations::

    odcli rec-grid -r rec.pdbqt -o maps/
    odcli dock -r rec.pdbqt -l lig.pdbqt --center_x 0 --center_y 0 \\
               --center_z 0 --size_x 22 --size_y 22 --size_z 22 -o out.pdbqt
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "core",
    "Receptor",
    "GridMaps",
    "Ligand",
    "GridBox",
    "DockingResult",
    "dock",
    "score_conformation",
    "conformation_coordinates",
    "evaluate_conformations",
    "load_receptor",
    "load_ligand",
    "auto_box",
    "gpu_status",
    "available_backends",
    "scoring_functions",
    "engine_version",
    "SCORING_FUNCTIONS",
]

from .core import (
    SCORING_FUNCTIONS,
    DockingResult,
    GridBox,
    GridMaps,
    Ligand,
    Receptor,
    available_backends,
    auto_box,
    dock,
    engine_version,
    evaluate_conformations,
    gpu_status,
    load_ligand,
    load_receptor,
    score_conformation,
    conformation_coordinates,
    scoring_descriptions as scoring_functions,
)
