# Open Docking (`odockmcode`)

[English](README.en.md) | [中文](README.md)

A molecular docking toolkit. The Rust engine handles affinity-map
precalculation, empirical scoring and conformational search; the Python front
end handles chemistry-aware preparation (RDKit aromaticity, protonation, charges,
polar hydrogens); and an interactive PyQt6 + ModernGL 3-D workbench sits on top.

The pipeline itself is classic and well established: the receptor is
precalculated into an affinity grid (trilinear interpolation), the ligand is
optimised with iterated local search plus quasi-Newton descent, and scoring uses
the form of the AutoDock Vina empirical function. On top of that it adds
**analytic first-order gradients**, **pure-safe Rust**
(`#![forbid(unsafe_code)]`) and **a GPU batch-scoring path that is actually on
the hot path**.

> **Two things you must know**
> 1. **It has never been validated by redocking** — there is no evidence that it
>    places a known ligand back into the correct pocket.
> 2. **Absolute energies must not be compared with AutoDock Vina or with
>    literature values** (no cross-validation was done). Relative ranking, pose
>    quality and geometric sanity are usable.
>
> The full list is in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Install

Needs Rust 1.75+, Python 3.9+, `maturin`. **No pre-built wheel is shipped; you
build one once.**

```bash
python -m pip install -r requirements.txt

# CPU build
python -m maturin build --release -m dock-py/Cargo.toml --out dist
python -m pip install dist/odockmcode-0.1.0-*.whl

# only for the 3-D workbench (the engine needs no graphics stack at all)
python -m pip install PyQt6 moderngl numpy-stl
```

Optional GPU build:

```bash
python -m maturin build --release -m dock-py/Cargo.toml --out dist-gpu --features gpu
python -m pip install --force-reinstall --no-deps dist-gpu/odockmcode-0.1.0-*.whl
odockmcode info        # expect: gpu: compiled in and an adapter is available
```

Only **Windows 11 x86_64 with an NVIDIA RTX 3050 Laptop** was actually run.
macOS, Linux, AMD discrete, Intel integrated and Apple Silicon are **untested**.

## Usage

### Command line

The full CLI is **`odockmcode`**. (`odock` belongs to another project on the
maintainer's machine; this project has no such command.)

| Subcommand | What it does |
|---|---|
| `info` | print engine version, scoring parameters, GPU status, parallel backends |
| `prep-receptor` | `.pdb` → receptor PDBQT |
| `prep-ligand` | any structure → ligand PDBQT + a property report |
| `rec-grid` | precalculate maps and write AutoDock `.map` files |
| `dock` | dock one ligand or a whole directory |
| `split` | split a multi-model PDBQT into one file per pose |
| `workbench` | launch the interactive 3-D viewer (same as `odgui`) |

A complete run on the bundled example data:

```bash
cd examples

# receptor PDB -> PDBQT
odockmcode prep-receptor -r receptor.pdb -o rec_prep.pdbqt

# ligand SDF -> PDBQT (also prints torsions, DOF count, atom classification)
odockmcode prep-ligand -l ibuprofen.sdf -o ibuprofen_prep.pdbqt

# precalculate + dock; the search box must be given explicitly
odockmcode dock -r rec_prep.pdbqt -l ibuprofen_prep.pdbqt \
           --center_x 0 --center_y 0 --center_z 0 \
           --size_x 20 --size_y 20 --size_z 20 \
           -e 8 -o poses.pdbqt
```

Common options:

```
-e, --exhaustiveness N     independent search trajectories (default 8, use 32 to be thorough)
-m, --num_modes N          how many de-duplicated conformations to output (default 9)
--rmsd-cutoff R            RMSD below which two poses count as the same (default 1.0 Å)
--scoring {vina,vinardo}   scoring function
--spacing S                grid spacing (default 0.375 Å)
--mode {mc,lga,both}       global search strategy
--seed N                   fix the random seed (for reproducibility)
--json                     machine-readable output
```

### Python

```python
from odockmcode import Receptor, Ligand, GridBox, dock

receptor = Receptor.from_pdbqt("rec_prep.pdbqt")
box_ = GridBox.from_center_size((0., 0., 0.), (20., 20., 20.))
maps = receptor.precalculate(box_, scoring="vina", spacing=0.375)

ligand = Ligand.from_pdbqt("ibuprofen_prep.pdbqt")
result = dock(ligand, maps, exhaustiveness=8, num_modes=9)
print(result.summary())
result.write_pdbqt("poses.pdbqt")
```

The full API is in [`docs/API.md`](docs/API.md).

### 3-D workbench

`odgui` is a standalone entry point that depends on no particular command name.

```bash
odgui                                       # empty window; load from the File menu
odgui -r rec_prep.pdbqt -l lig.pdbqt -p poses.pdbqt
odgui --check                               # report whether it can start, no window
```

Left drag rotates, right drag pans, middle drag dollies, wheel zooms, `Frame all`
re-frames. The File menu offers `Load receptor… / Load ligand… / Open poses… /
Dock now`.

## Things to know before you use the results

- **The search box must be given explicitly.** A box covering the whole protein
  costs tens of gigabytes and is equivalent to having no search region; aim it
  at the pocket.
- **Option names use underscores**: `--center_x`, not `--center-x`.
- **`use_gpu` defaults to False.** The GPU accumulates in `f32` and differs from
  the CPU by about 1e-6; a default that varies with the hardware would break
  cross-machine reproducibility. Pass `use_gpu=True` explicitly, then read the
  fallback reason out of `report_backend`.
- **Do not use `|grad|` to decide whether a pose converged.** The grid is a
  trilinear interpolant and therefore only C⁰, and in practice every returned
  pose has an atom sitting on a cell face, where `|grad|` is often 2–6 while the
  energy will not move anywhere. Use a line search.
- **If you edited anything under `dock-py/python/`, rebuild and reinstall the
  wheel**, or `pytest --pyargs` will still be testing the previous build.

## Documentation

| Document | Contents |
|---|---|
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | **read this first**: known limitations and out-of-scope uses |
| [`docs/API.md`](docs/API.md) | complete Rust and Python API reference |
| [`docs/SCORING.md`](docs/SCORING.md) | scoring derivation, item by item against the Rust |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | data flow, parallelism, memory layout, zero-copy contract |
| [`docs/VERIFICATION.md`](docs/VERIFICATION.md) | **honest verification report**: what was run, what was not verified |
| [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) | provenance, licences, clean-room statement |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | workflow, hard constraints, local verification gate |
| [`SECURITY.md`](SECURITY.md) | security policy |
| [`CHANGELOG.md`](CHANGELOG.md) | changelog |
| [`CITATION.cff`](CITATION.cff) | citation metadata |

## Licence

**GPL-3.0-or-later**, full text in [`LICENSE`](LICENSE). Because this project
distributes GPL-3.0 code, any binary distribution must provide the complete
source under GPL-3.0 as well. Provenance of the functional forms and weights is
in [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md).
