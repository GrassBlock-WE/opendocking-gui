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
    "exhaustiveness_for_box",
    "EXHAUSTIVENESS_LADDER",
    "REFERENCE_BOX_SIDE",
    "gpu_status",
    "available_backends",
    "scoring_descriptions",
    "engine_version",
    "SCORING_FUNCTIONS",
]

#: Scoring functions the engine understands.
SCORING_FUNCTIONS = ("vina", "vinardo")

#: The side of the box that `exhaustiveness_for_box` treats as "no extra
#: effort needed". A ligand-sized search box is 20 A on a side; that is the
#: case the conventional default of 8 was actually chosen for.
REFERENCE_BOX_SIDE = 20.0

#: Sampling effort, as a ladder rather than a formula, so the number a run
#: used can be read off by hand and is the same on every machine.
EXHAUSTIVENESS_LADDER = (8, 16, 32, 64, 128)


def _as_bounded_int(value: Any, name: str, low: int, high: int) -> int:
    """Return ``value`` as an int, or refuse it with a ``ValueError``.

    The engine converts these arguments to unsigned integers, and that
    conversion is where the mistake used to surface: too large, negative, NaN
    and infinite all came back as ``OverflowError`` or ``ValueError`` from
    *inside* Rust, with no mention of which argument was wrong. Catching the
    conversion here means one exception type and one message that names the
    parameter, which is the only thing a caller can act on.

    The bounds are half-open, ``low <= result < high``, and are the engine's
    own: a seed is a u64, and ``steps`` must leave room for at least one.
    """
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be an integer in [{low}, {high}), got {value!r}"
        ) from exc
    if not low <= as_int < high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}), got {value!r}")
    return as_int


def exhaustiveness_for_box(size) -> int:
    """A sensible number of search walks for a box of this size.

    Exhaustiveness is a count of Monte Carlo walks, and walks are spread
    through the box they search. The same 8 that comfortably covers a
    20 A box is spread four times thinner in a 40 A one, so a fixed value
    quietly under-samples every large box -- and a search that under-samples
    reports a bad *pose*, which is indistinguishable from the box being
    wrong. This is not a theory; it is what the redocking benchmark measured
    on the 3PTB site box, 39 x 26 x 41 A:

        exhaustiveness  16 -> 12.88 A RMSD,  -3.67 kcal/mol,  1 s
        exhaustiveness  64 ->  1.26 A RMSD,  -3.88 kcal/mol,  2 s
        exhaustiveness 128 ->  1.26 A RMSD,  -3.88 kcal/mol,  3 s

    Same box, same receptor, same ligand, same seed. The box was fine; the
    search had not been given enough room to find the mode.

    Linear in the box's volume, and deliberately so. One calibration point
    says a 5.2x bigger box needs at least 8x the walks; a power law could be
    fitted to make that come out exactly, and a power law fitted to a single
    measurement is a way of pretending to know something. Linear is a
    whole number, it is on the conservative side of the one number we have,
    and "8 walks per 8000 A^3" is a thing you can hold in your head.

    **The result saturates at the top rung and the linear rule does not.** A
    60 A cube is 216,000 A^3 and the rule asks for 216 walks; this returns 128.
    A 200 A cube asks for 8000 and gets 128, a 62x under-estimate reported as
    a default. The cap is deliberate -- the ladder is a ladder, and something
    has to be at the top -- but it was nowhere in this docstring, so a caller
    reading "linear in the box's volume" and getting 128 for a 200 A box had no
    way to know which sentence was wrong. Callers that need the real figure
    should ask for it rather than read a default as a recommendation.

    A non-finite or wrongly-shaped ``size`` is refused rather than answered.
    A NaN volume compared `>=` against every rung is False for all of them, so
    the loop fell through and returned the *top* rung: 128 walks, the most
    expensive answer, for a box the engine cannot even build. A two-element
    size was read as a 2-D volume and answered without complaint.

    This is a *default*, and callers are free to ignore it. The one thing not
    free is pretending the two are the same claim: "1.2 A found in a 41,000 A^3
    box at 64" and "1.2 A found in a 3,000 A^3 box at 8" are different results,
    and a report that prints both as "1.2 A" is hiding the difference.
    """
    sides = tuple(float(v) for v in size)
    if len(sides) != 3:
        raise ValueError(
            f"size must have 3 values, got {len(sides)}; a box is "
            "three-dimensional and a two-value size is not a smaller box"
        )
    volume = float(np.prod(np.asarray(sides, np.float64)))
    if not np.isfinite(volume):
        raise ValueError(
            f"size must be finite, got {sides}; the engine rejects a box with a "
            "NaN or infinite edge too, so no number of walks can help"
        )
    reference = REFERENCE_BOX_SIDE ** 3
    want = EXHAUSTIVENESS_LADDER[0] * (volume / reference)
    for rung in EXHAUSTIVENESS_LADDER:
        if rung >= want:
            return rung
    return EXHAUSTIVENESS_LADDER[-1]


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
        return cls._from_raw(
            _core.GridBox.from_center_size(
                tuple(float(v) for v in center), tuple(float(v) for v in size)
            )
        )

    @classmethod
    def _from_raw(cls, raw: Any) -> "GridBox":
        """Wrap an engine-side box.

        Single underscore on purpose. This was `__new_from`, and a name with
        two leading underscores is mangled per *class that uses it*: written
        inside `GridBox` as `cls.__new_from` it becomes
        `cls._GridBox__new_from` and works, but written inside `GridMaps` as
        `GridBox.__new_from` it becomes `GridBox._GridMaps__new_from`, which
        does not exist. So `GridMaps.box` raised
        `AttributeError: type object 'GridBox' has no attribute
        '_GridMaps__new_from'` on every single call, for its whole life, and
        nothing noticed: no caller in the repository reads it, and the one
        meta-test that inspects this file checks the *Rust* attribute names,
        not the Python ones. A cross-class helper needs a name that is not
        mangled.
        """
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
        """Return whether ``point`` lies inside the box.

        Lower corner inclusive, upper corner exclusive, matching the corner
        convention in the class docstring. ``point`` must have three values:
        with two it raised ``IndexError`` from inside the generator, and with
        four the extra values were read past and ignored, so both answered
        about a point the caller had not described.
        """
        p = tuple(float(v) for v in point)
        if len(p) != 3:
            raise ValueError(
                f"point must have 3 values, got {len(p)}; a box is "
                "three-dimensional and a two-value point is not half a point"
            )
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
        """Estimate the memory the maps for ``box_`` will need, in megabytes.

        Answers for exactly the calls `precalculate` will accept, because the
        point of asking first is to decide whether to attempt the tabulation.
        It did not. A negative, infinite or NaN ``spacing`` was quietly
        replaced by the default, so a caller was told "9.1 MB, go ahead" for a
        call that `precalculate` then rejected with `ValueError` -- and for a
        NaN spacing it returned 0.0012 MB, a number for no grid at all.

        ``spacing=0`` means "use the default" here, which is what the engine
        does with it. That convention is the engine's and was not written down
        on either Python signature; it is now.
        """
        spacing = float(spacing)
        if not np.isfinite(spacing) or spacing < 0.0:
            # `not isfinite` rather than the `spacing != spacing` NaN test: an
            # infinite spacing got past that one and came back as 0.0012 MB,
            # "a number for no grid at all", for a call `precalculate` refuses.
            # An estimator that answers for a call which will crash is worse
            # than no estimator, because the answer is a reason to try.
            raise ValueError(
                f"spacing must be zero (meaning the default) or a positive "
                f"value, got {spacing}; precalculate rejects the same value"
            )
        return self._rec.estimate_memory_mb(box_._box, spacing)

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
            Nothing enforces it: `dock` will score these maps with the other
            function and report a plausible number. The maps do not remember
            what built them, so the obligation stays the caller's.
        spacing:
            Grid resolution in Ångström. 0.375 is the AutoDock default; halving
            it quarters the interpolation discontinuity at 8× the memory.
            ``0`` means "use the default" -- the engine's convention, and now
            written down on both signatures. Negative and non-finite values are
            rejected, and `estimate_memory_mb` rejects the same ones.
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
        """The box these maps cover.

        Equal to the box that was passed to `Receptor.precalculate`, not a
        recomputation of it: the engine stores the one it tabulated, and a
        caller comparing the two is asking whether the maps went where they
        think.
        """
        return GridBox._from_raw(self._maps.box_)

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

        ``coords``, ``charges`` and ``elements`` must agree in length and be
        finite. The engine checks the lengths and the elements; it did not
        check the numbers. A table with a NaN coordinate produced a ligand
        whose ``reference_coords`` were all NaN and whose ``radius`` was
        ``0.0`` -- a confident answer about a molecule with no position, which
        then poisons every energy computed from it.
        """
        arr = np.ascontiguousarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"coords must have shape (n, 3), got {arr.shape}")
        charge_list = [float(c) for c in charges]
        if not np.isfinite(arr).all():
            bad = int(np.argmax(~np.isfinite(arr).all(axis=1)))
            raise ValueError(
                f"coords must be finite; atom {bad} has "
                f"{arr[bad].tolist()}"
            )
        if not np.isfinite(charge_list).all():
            bad = int(np.argmax(~np.isfinite(np.asarray(charge_list))))
            raise ValueError(
                f"charges must be finite; charge {bad} is {charge_list[bad]}"
            )
        bond_list = [tuple(int(i) for i in b) for b in bonds] if bonds else None
        if bond_list is not None:
            n = arr.shape[0]
            for bond in bond_list:
                for i in bond:
                    if not 0 <= i < n:
                        raise ValueError(
                            f"bond {bond} refers to atom {i}, and this molecule "
                            f"has {n} atoms"
                        )
        return cls(
            _core.Ligand.from_arrays(
                list(elements),
                charge_list,
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
        """Coordinates of pose ``index``, shape ``(n_atoms, 3)``, float64.

        ``index`` must be a pose that exists. A negative index raised
        ``OverflowError`` from the engine's unsigned conversion while an
        out-of-range one raised ``ValueError`` -- the same mistake, two
        exception types, and the type is the only thing a caller branches on.
        """
        return self._res.pose_coords(self._checked_index(index))

    def pose_conformation(self, index: int = 0) -> np.ndarray:
        """The degree-of-freedom vector of pose ``index``.

        Layout is ``[tx, ty, tz, θx, θy, θz, τ₀, …]`` — the same vector
        :func:`score_conformation` takes, so a pose can be re-scored, checked
        for a vanishing analytic gradient, or used as the start of a further
        optimisation.
        """
        return np.asarray(self._res.pose_conformation(self._checked_index(index)), dtype=np.float64)

    def _checked_index(self, index: int) -> int:
        """Turn a pose index into one the engine will accept, or explain why not."""
        idx = int(index)
        if not 0 <= idx < self.num_poses:
            raise ValueError(
                f"pose index {idx} out of range: this result has "
                f"{self.num_poses} pose(s)"
            )
        return idx

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
        Local-search steps per walk. Leave unset unless tuning. Must be at
        least 1 if given; a negative one raised ``OverflowError`` from the
        engine's unsigned conversion while ``steps=0`` raised ``ValueError``.
    """
    if exhaustiveness < 1:
        raise ValueError("exhaustiveness must be at least 1")
    if num_modes < 1:
        raise ValueError("num_modes must be at least 1")
    if rmsd_cutoff <= 0:
        raise ValueError("rmsd_cutoff must be positive")
    if steps is not None:
        # `steps=0` is refused by the engine with a ValueError, so the bound is
        # drawn where the engine draws it; a negative value never got that far
        # and came back as OverflowError instead.
        _as_bounded_int(steps, "steps", 1, 2 ** 63)
    if seed is not None:
        # The engine takes a u64, so an out-of-range seed used to raise
        # OverflowError from the conversion -- ArithmeticError, not ValueError,
        # so a caller with `except ValueError` around the three documented
        # guards above did not catch it. Same class of mistake, different
        # exception type, and the type is the only thing a caller branches on.
        #
        # The conversion lives inside the helper because `int(float("inf"))`
        # raises OverflowError as well: writing the test as `int(seed) < 0`
        # only moved the leak rather than closing it.
        _as_bounded_int(seed, "seed", 0, 2 ** 64)
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
    single precision -- a relative error of order 1e-7, which on a score of
    order 10 is around a microcalorie per mole -- rather than bit-for-bit. A
    default that silently changes the last digits of a result depending on the
    machine would make runs irreproducible across hardware. Opt in with
    ``use_gpu=True`` when you are scoring a large population and the speed is
    worth that much precision.

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
    receptor: Receptor, ligand: Ligand | None = None, padding: float = 4.0
) -> GridBox:
    """Build a box that encloses the whole receptor, padded by ``padding``.

    This is a convenience for small receptors and for tests. For a real protein
    the search box must be a *binding site* — a 3000-atom receptor in one box
    would need tens of gigabytes of maps and would make the search far less
    effective.

    To place it on a real site rather than by hand: ``--auto-box N`` on the
    command line, or the site list in the workbench, both of which use
    :func:`opendocking.workbench.pockets.find_pockets`. This function is *not*
    that — it is the whole-receptor box this paragraph is warning about, kept
    because a test needs something trivial and honest to compare against.

    ``ligand`` is accepted and **never read**. The box comes from the
    receptor's bounds alone, so the name says "box for this receptor" and the
    signature used to say "box for this receptor and this ligand". Nothing in
    the repository calls this function at all; it is exported, so it is part of
    the public surface, and the parameter is left in place rather than removed
    because dropping it is a breaking change for whoever is calling it with a
    ligand in hand and expecting it to matter. It is now documented instead of
    merely present.
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
