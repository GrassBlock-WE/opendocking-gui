"""Typed Python wrappers over the compiled docking engine.

Everything here is a thin layer: the heavy lifting happens in the Rust extension
module :mod:`opendocking._dockpy`. The wrappers exist to give the library a coherent
Python surface — one import location, a consistent docstring style, and objects
that are pleasant to use interactively — not to add behaviour that does not
belong in a language binding.

NumPy arrays cross into Rust **without copying**; see the ``dock-py`` module
documentation for the exact contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - exercised implicitly by every import
    from . import _dockpy as _core
except ImportError as exc:  # pragma: no cover - only on a broken build
    raise ImportError(
        "The Open Docking native extension is not available. Build it with:\n"
        "    pip install ./python\n"
        "or, for a development build:\n"
        "    maturin develop --release -m opendocking._dockpy\n"
        f"Underlying error: {exc}"
    ) from exc

__all__ = [
    "GridBox",
    "Receptor",
    "GridMaps",
    "Ligand",
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
    "scoring_descriptions",
    "engine_version",
    "SCORING_FUNCTIONS",
]

#: Scoring functions the engine understands.
SCORING_FUNCTIONS = ("vina", "vinardo")


def engine_version() -> str:
    """Return the version of the compiled engine."""
    return _core.version()


def scoring_descriptions() -> list[str]:
    """Return a human-readable description of each scoring function."""
    return _core.scoring_functions()


class GridBox:
    """An axis-aligned cuboid in which the ligand is allowed to be placed.

    Parameters
    ----------
    min_corner, max_corner:
        Inclusive lower and exclusive upper corners, in Ångström, in the
        receptor's own coordinate frame.

    Examples
    --------
    >>> box_ = GridBox.from_center_size((0.0, 0.0, 0.0), (22.0, 22.0, 22.0))
    >>> box_.size
    (22.0, 22.0, 22.0)
    """

    __slots__ = ("_box",)

    def __init__(self, min_corner: Sequence[float], max_corner: Sequence[float]) -> None:
        self._box = _core.GridBox(
            tuple(float(v) for v in min_corner),
            tuple(float(v) for v in max_corner),
        )

    @classmethod
    def from_center_size(
        cls, center: Sequence[float], size: Sequence[float]
    ) -> "GridBox":
        """Build a box of the given size centred on ``center``."""
        return cls.__new_from(  # type: ignore[attr-defined]
            _core.GridBox.from_center_size(
                tuple(float(v) for v in center), tuple(float(v) for v in size)
            )
        )

    @classmethod
    def __new_from(cls, raw: Any) -> "GridBox":
        obj = object.__new__(cls)
        obj._box = raw
        return obj

    @property
    def min_corner(self) -> tuple[float, float, float]:
        """Lower corner of the box."""
        return tuple(self._box.min)

    @property
    def max_corner(self) -> tuple[float, float, float]:
        """Upper corner of the box."""
        return tuple(self._box.max)

    @property
    def center(self) -> tuple[float, float, float]:
        """Geometric centre of the box."""
        return tuple(self._box.center)

    @property
    def size(self) -> tuple[float, float, float]:
        """Edge lengths of the box, in Ångström."""
        return tuple(self._box.size)

    def contains(self, point: Sequence[float]) -> bool:
        """Return whether ``point`` lies inside the box."""
        p = tuple(float(v) for v in point)
        lo, hi = self.min_corner, self.max_corner
        return all(lo[k] <= p[k] < hi[k] for k in range(3))

    def __repr__(self) -> str:
        return f"GridBox(min={self.min_corner}, max={self.max_corner})"


class Receptor:
    """A rigid receptor, prepared for map precalculation."""

    __slots__ = ("_rec",)

    def __init__(self, raw: Any) -> None:
        self._rec = raw

    @classmethod
    def from_pdbqt(cls, path: str | Path) -> "Receptor":
        """Read a receptor from a PDBQT file."""
        return cls(_core.Receptor.from_pdbqt(Path(path)))

    @classmethod
    def from_pdbqt_str(cls, text: str) -> "Receptor":
        """Read a receptor from PDBQT text."""
        return cls(_core.Receptor.from_pdbqt_str(text))

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the receptor."""
        return self._rec.num_atoms

    @property
    def num_polar_hydrogens(self) -> int:
        """Number of explicit polar hydrogens.

        A well-prepared receptor has polar hydrogens on every donor. Zero is a
        strong hint that the structure was prepared with non-polar hydrogens
        merged, which costs hydrogen bonds at every serine, threonine and
        tyrosine. :func:`opendocking.prep.prepare_receptor` adds them for you.
        """
        return self._rec.num_polar_hydrogens

    @property
    def unknown_atom_types(self) -> int:
        """Atoms whose PDBQT type this engine does not recognise.

        Not an error — another tool may emit type names AutoDock never defined
        — but such an atom still contributes its shape term while silently
        losing its hydrogen-bond and hydrophobic character.
        """
        return self._rec.unknown_atom_types

    @property
    def center(self) -> tuple[float, float, float]:
        """Centroid of the receptor."""
        return tuple(self._rec.center)

    @property
    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Axis-aligned bounding box of the receptor."""
        lo, hi = self._rec.bounds
        return tuple(lo), tuple(hi)

    def estimate_memory_mb(self, box_: GridBox, spacing: float = 0.375) -> float:
        """Estimate the memory the maps for ``box_`` will need, in megabytes."""
        return self._rec.estimate_memory_mb(box_._box, float(spacing))

    def precalculate(
        self, box_: GridBox, scoring: str = "vina", spacing: float = 0.375
    ) -> "GridMaps":
        """Tabulate the affinity maps for ``box_``.

        Parameters
        ----------
        box_:
            The search region. This must be the *binding site*, not the whole
            protein: a box covering a 3000-atom receptor needs tens of
            gigabytes of maps.
        scoring:
            ``"vina"`` or ``"vinardo"``. The maps are specific to this choice —
            reusing them under a different scoring function gives wrong answers.
        spacing:
            Grid resolution in Ångström. 0.375 is the AutoDock default; halving
            it quarters the interpolation discontinuity at 8× the memory.
        """
        return GridMaps(
            self._rec.precalculate(box_._box, scoring, float(spacing))
        )

    def __repr__(self) -> str:
        return f"Receptor(num_atoms={self.num_atoms})"


class GridMaps:
    """Precalculated receptor affinity maps."""

    __slots__ = ("_maps",)

    def __init__(self, raw: Any) -> None:
        self._maps = raw

    @property
    def dims(self) -> tuple[int, int, int]:
        """Number of grid points along each axis."""
        return tuple(self._maps.dims)

    @property
    def spacing(self) -> float:
        """Grid spacing in Ångström."""
        return self._maps.spacing

    @property
    def raw_data(self) -> np.ndarray:
        """The raw map storage, flat ``float32``.

        Layout is ``((ix + nx·(iy + ny·iz)) · STRIDE) + type·4 + slot`` with
        ``STRIDE = GRID_TYPE_COUNT · MAPS_PER_TYPE = 40``. Useful for writing
        the grid out, plotting a slice, or comparing two grids exactly.
        """
        return self._maps.raw_data()

    @property
    def num_points(self) -> int:
        """Total number of tabulated grid points."""
        return self._maps.num_points

    @property
    def memory_mb(self) -> float:
        """Approximate memory footprint, in megabytes."""
        return self._maps.memory_mb

    @property
    def box(self) -> GridBox:
        """The box these maps cover."""
        return GridBox.__new_from(self._maps.box_)

    def write_map_files(self, directory: str | Path) -> None:
        """Write AutoDock-compatible ``.map`` files into ``directory``."""
        self._maps.write_map_files(Path(directory))

    def __repr__(self) -> str:
        return f"GridMaps(dims={self.dims}, spacing={self.spacing})"


class Ligand:
    """A prepared docking ligand."""

    __slots__ = ("_lig",)

    def __init__(self, raw: Any) -> None:
        self._lig = raw

    @classmethod
    def from_pdbqt(cls, path: str | Path) -> "Ligand":
        """Read a ligand from a PDBQT file."""
        return cls(_core.Ligand.from_pdbqt(Path(path)))

    @classmethod
    def from_pdbqt_str(cls, text: str) -> "Ligand":
        """Read a ligand from PDBQT text."""
        return cls(_core.Ligand.from_pdbqt_str(text))

    @classmethod
    def from_arrays(
        cls,
        elements: Sequence[str],
        charges: Sequence[float],
        coords: np.ndarray,
        bonds: Iterable[tuple[int, int]] | None = None,
        atom_names: Sequence[str] | None = None,
    ) -> "Ligand":
        """Build a ligand from a plain chemistry table.

        This is the path the RDKit front-end uses: RDKit resolves aromaticity,
        protonation and Gasteiger charges, then hands over a table, so the
        engine never re-derives chemistry it would only get wrong.
        """
        arr = np.ascontiguousarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"coords must have shape (n, 3), got {arr.shape}")
        bond_list = [tuple(int(i) for i in b) for b in bonds] if bonds else None
        return cls(
            _core.Ligand.from_arrays(
                list(elements),
                [float(c) for c in charges],
                arr,
                bond_list,
                list(atom_names) if atom_names else None,
            )
        )

    @property
    def num_atoms(self) -> int:
        """Number of atoms."""
        return self._lig.num_atoms

    @property
    def num_torsions(self) -> int:
        """Number of rotatable bonds."""
        return self._lig.num_torsions

    @property
    def num_dof(self) -> int:
        """Total degrees of freedom: 6 rigid-body plus one per torsion."""
        return self._lig.num_dof

    @property
    def radius(self) -> float:
        """Radius of the ligand about its centroid, in Ångström."""
        return self._lig.radius

    @property
    def atom_kinds(self) -> list[str]:
        """Interaction class of every atom: hydrophobic, donor, acceptor, …"""
        return self._lig.atom_kinds

    @property
    def reference_coords(self) -> np.ndarray:
        """Reference coordinates, shape ``(n_atoms, 3)``, dtype float64."""
        return self._lig.reference_coords()

    def __repr__(self) -> str:
        return f"Ligand(num_atoms={self.num_atoms}, num_torsions={self.num_torsions})"


class DockingResult:
    """The ranked poses produced by a docking run."""

    __slots__ = ("_res",)

    def __init__(self, raw: Any) -> None:
        self._res = raw

    @property
    def num_poses(self) -> int:
        """Number of distinct poses."""
        return self._res.num_poses

    @property
    def energies(self) -> np.ndarray:
        """Total energies of every pose, best first, in kcal/mol."""
        return np.asarray(self._res.energies, dtype=np.float64)

    @property
    def best_energy(self) -> float:
        """Energy of the best pose, in kcal/mol."""
        return float(self._res.best_energy)

    @property
    def intermolecular_energies(self) -> np.ndarray:
        """Receptor–ligand contribution of every pose, in kcal/mol."""
        return np.asarray(self._res.intermolecular_energies, dtype=np.float64)

    @property
    def rmsds(self) -> np.ndarray:
        """RMSD of every pose to the best one, in Ångström."""
        return np.asarray(self._res.rmsds, dtype=np.float64)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds spent searching."""
        return float(self._res.elapsed_seconds)

    @property
    def raw_pose_count(self) -> int:
        """Number of raw conformations produced before clustering."""
        return int(self._res.raw_pose_count)

    @property
    def rejected_pose_count(self) -> int:
        """Reported poses that overlap the receptor.

        Non-zero means the search never found a physically possible pose and
        the engine fell back to reporting the best of what it had. That is
        almost always a sign the search box contains solid protein rather than
        a pocket, or that ``exhaustiveness`` is too low.
        """
        return int(self._res.rejected_pose_count)

    @property
    def scoring_function(self) -> str:
        """Name of the scoring function used."""
        return self._res.scoring_function

    def pose_coords(self, index: int = 0) -> np.ndarray:
        """Coordinates of pose ``index``, shape ``(n_atoms, 3)``, float64."""
        return self._res.pose_coords(index)

    def pose_conformation(self, index: int = 0) -> np.ndarray:
        """The degree-of-freedom vector of pose ``index``.

        Layout is ``[tx, ty, tz, θx, θy, θz, τ₀, …]`` — the same vector
        :func:`score_conformation` takes, so a pose can be re-scored, checked
        for a vanishing analytic gradient, or used as the start of a further
        optimisation.
        """
        return np.asarray(self._res.pose_conformation(index), dtype=np.float64)

    def all_pose_coords(self) -> np.ndarray:
        """Every pose stacked, shape ``(n_poses, n_atoms, 3)``, float64."""
        return self._res.all_pose_coords()

    def write_pdbqt(self, path: str | Path) -> None:
        """Write the poses as a multi-model PDBQT file."""
        self._res.write_pdbqt(Path(path))

    def write_xyz(self, path: str | Path) -> None:
        """Write the poses as a multi-frame XYZ file."""
        self._res.write_xyz(Path(path))

    def summary(self) -> str:
        """A short human-readable table of the results."""
        lines = [
            f"{self.num_poses} pose(s) from {self.raw_pose_count} conformations "
            f"in {self.elapsed_seconds:.2f} s  [{self.scoring_function}]",
            "  rank   affinity  inter      rmsd",
        ]
        for i, (e, inter, r) in enumerate(
            zip(self.energies, self.intermolecular_energies, self.rmsds), start=1
        ):
            lines.append(f"  {i:>4}   {e:>8.2f}  {inter:>7.2f}  {r:>7.2f}")
        if self.rejected_pose_count:
            lines.append(
                f"  WARNING: {self.rejected_pose_count} of {self.num_poses} "
                "reported poses overlap the receptor. The search never found a "
                "physically possible placement, which usually means the box "
                "contains solid protein rather than a pocket, or that "
                "exhaustiveness is too low."
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DockingResult(num_poses={self.num_poses}, best={self.best_energy:.2f})"


def dock(
    ligand: Ligand,
    maps: GridMaps,
    *,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    rmsd_cutoff: float = 1.0,
    seed: int | None = None,
    mode: str = "mc",
    scoring: str | None = None,
    steps: int | None = None,
) -> DockingResult:
    """Dock ``ligand`` against precalculated ``maps``.

    Parameters
    ----------
    ligand:
        A prepared ligand.
    maps:
        Maps built with the *same* scoring function passed here (or omitted, in
        which case ``vina`` is assumed).
    exhaustiveness:
        Number of independent search walks, AutoDock's ``--exhaustiveness``.
        Higher explores more of the box and finds better poses; the cost is
        close to linear. 8 is a good default, 32 is thorough.
    num_modes:
        How many distinct poses to report after clustering.
    rmsd_cutoff:
        Per-atom RMSD below which two poses are considered the same mode, in
        Ångström. AutoDock's default is 1.0; raise it for very flexible ligands.
    seed:
        Fixes the random seed, making a run reproducible.
    mode:
        ``"mc"`` for iterated local search, ``"lga"`` for the island genetic
        algorithm, or ``"both"``.
    steps:
        Local-search steps per walk. Leave unset unless tuning.
    """
    if exhaustiveness < 1:
        raise ValueError("exhaustiveness must be at least 1")
    if num_modes < 1:
        raise ValueError("num_modes must be at least 1")
    if rmsd_cutoff <= 0:
        raise ValueError("rmsd_cutoff must be positive")
    return DockingResult(
        _core.dock(
            ligand._lig,
            maps._maps,
            int(exhaustiveness),
            int(num_modes),
            float(rmsd_cutoff),
            None if seed is None else int(seed),
            mode,
            scoring,
            None if steps is None else int(steps),
        )
    )


def score_conformation(
    ligand: Ligand,
    maps: GridMaps,
    conformation: Sequence[float],
    scoring: str = "vina",
) -> tuple[float, np.ndarray]:
    """Evaluate one conformation and return its energy and analytic gradient.

    ``conformation`` is the packed degree-of-freedom vector
    ``[tx, ty, tz, θx, θy, θz, τ₀, …]``. The gradient has the same layout and
    can be handed straight to any optimiser.
    """
    arr = np.ascontiguousarray(conformation, dtype=np.float64)
    if arr.shape != (ligand.num_dof,):
        raise ValueError(
            f"conformation must have {ligand.num_dof} values, got {arr.shape}"
        )
    energy, grad = _core.score(
        ligand._lig,
        maps._maps,
        scoring,
        (float(arr[0]), float(arr[1]), float(arr[2])),
        (float(arr[3]), float(arr[4]), float(arr[5])),
        np.ascontiguousarray(arr[6:]),
    )
    return float(energy), np.asarray(grad, dtype=np.float64)


def conformation_coordinates(ligand: Ligand, conformation: Sequence[float]) -> np.ndarray:
    """World-space coordinates of a conformation, shape ``(n_atoms, 3)``.

    ``conformation`` is the same packed degree-of-freedom vector
    :func:`score_conformation` takes. This is pure forward kinematics — no grid
    is involved — so it works without a :class:`GridMaps`.

    Useful for drawing a conformation, measuring interatomic distances, or
    checking a candidate pose for steric clashes before trusting its score.
    """
    arr = np.ascontiguousarray(conformation, dtype=np.float64)
    return _core.conformation_coordinates(ligand._lig, arr)


def evaluate_conformations(
    ligand: Ligand,
    maps: GridMaps,
    conformations: np.ndarray,
    scoring: str = "vina",
    use_gpu: bool = False,
    report_backend: dict | None = None,
) -> np.ndarray:
    """Evaluate a batch of conformations without copying the input.

    ``conformations`` must be a C-contiguous ``(n, 6 + num_torsions)`` float64
    array. This is the zero-copy path: the array is borrowed by the engine, so
    scoring a large population is bounded by the output allocation alone.

    ``use_gpu`` moves the per-atom grid interpolation to the compute kernel.
    It is **off by default**, and that default is deliberate: the kernel
    accumulates in single precision, so it agrees with the CPU path to about
    1e-8 rather than bit-for-bit. A default that silently changes the last
    digits of a result depending on the machine would make runs irreproducible
    across hardware. Opt in with ``use_gpu=True`` when you are scoring a large
    population and the speed is worth that much precision.

    Pass a dict as ``report_backend`` and it is filled in with
    ``backend`` (``"gpu"`` or ``"cpu"``), ``adapter``, ``gpu_skip_reason``
    and ``num_conformations``. A silent CPU fallback reads as an unexplained
    slowdown, so ask for it if you care.

    The result is the *total* energy — intermolecular plus the scaled
    intramolecular term — identical to calling :func:`score_conformation` once
    per row, up to that precision difference.
    """
    arr = np.ascontiguousarray(conformations, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ligand.num_dof:
        raise ValueError(
            f"conformations must have shape (n, {ligand.num_dof}), got {arr.shape}"
        )
    return _core.evaluate_conformations(
        ligand._lig, maps._maps, scoring, arr, bool(use_gpu), report_backend
    )


def gpu_status() -> dict[str, bool]:
    """Report whether the GPU backend is compiled in and usable.

    The CPU engine is fully functional on its own, so this is purely
    informational.
    """
    return {
        "compiled": bool(_core.gpu_compiled()),
        "available": bool(_core.gpu_available()),
    }


def available_backends() -> list[str]:
    """List the parallel backends the engine will consider for a search."""
    import os

    backends = ["cpu"]
    if gpu_status()["available"]:
        backends.append("gpu")
    n = os.cpu_count() or 1
    backends.append(f"cpu/{n}-threads")
    return backends


def auto_box(
    receptor: Receptor, ligand: Ligand, padding: float = 4.0
) -> GridBox:
    """Build a box that encloses the whole receptor, padded by ``padding``.

    This is a convenience for small receptors and for tests. For a real protein
    the search box must be a *binding site* — a 3000-atom receptor in one box
    would need tens of gigabytes of maps and would make the search far less
    effective. Use the workbench's box tool, or pass ``--center_*`` and
    ``--size_*`` on the command line, to place it by hand.
    """
    (lo, hi) = receptor.bounds
    return GridBox(
        [lo[0] - padding, lo[1] - padding, lo[2] - padding],
        [hi[0] + padding, hi[1] + padding, hi[2] + padding],
    )


def load_receptor(path: str | Path) -> Receptor:
    """Read a receptor, accepting either ``.pdbqt`` or a plain ``.pdb``."""
    path = Path(path)
    if path.suffix.lower() == ".pdbqt":
        return Receptor.from_pdbqt(path)
    from .prep import prepare_receptor

    return Receptor.from_pdbqt_str(prepare_receptor(path))


def load_ligand(path: str | Path) -> Ligand:
    """Read a ligand, preparing it with RDKit for any non-PDBQT format.

    ``.sdf``, ``.mol2``, ``.mol`` and ``.smi`` all go through RDKit so that
    aromaticity, protonation and charges are handled chemically rather than
    guessed.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdbqt":
        return Ligand.from_pdbqt(path)
    from .prep import prepare_ligand

    return Ligand.from_arrays(
        *prepare_ligand(path)
    )
