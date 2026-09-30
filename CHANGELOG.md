# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The project is pre-1.0. The `0.x` line is where the interfaces still move.

---

## [Unreleased]

### Fixed

- `Ligand.from_pdbqt_str` did not work at all. It called the file-opening
  reader with the document text, so the PDBQT was treated as a filename and
  every call failed — `os error 123` on Windows, a search for a file named
  after the whole document elsewhere. `Receptor.from_pdbqt_str` was unaffected
  and always went through the in-memory parser. Nothing caught this because the
  only callers pass deliberately malformed text, where a parser rejection and a
  failed file open look identical.
- A ligand with two atoms on the same coordinate was accepted silently. Every
  number came out finite, so docking returned plausible-looking poses built
  from geometry that does not exist. Ligands are now refused when any two
  atoms are closer than 0.5 Å, with the offending atom names in the error. The
  floor is far below the shortest real bond (H–H at 0.74 Å) and a test pins
  that it cannot reject a real structure.
- The CPU/GPU agreement test compared raw absolute errors against a fixed
  `1e-4`. The conformations it scores reach ~10³ kcal/mol, where a single
  f32 ulp is already ~1e-4, so the threshold was smaller than the arithmetic it
  was checking and passed only on the machine it was written on. It is now a
  relative criterion, reverse-verified to still reject an injected 1e-3
  relative error and to still reject a 0.01 absolute error at realistic docked
  energies.
- `docs/ARCHITECTURE.md` claimed the GPU path returns bit-for-bit the same
  numbers as the CPU path. It does not: the kernel accumulates in single
  precision and the measured relative difference is ~1e-7, varying with the
  shader compiler. Only the intramolecular term is bit-identical. The matching
  claim in the `evaluate_conformations` docstring is corrected too.
- `examples/robustness_check.py` decoded its child processes' stderr with the
  system code page. On a non-UTF-8 locale the decode failed inside the reader
  thread, stderr was left unset, and the guard raised `TypeError` instead of
  reporting a verdict.
- Three example scripts defaulted to bare filenames, so they only ran after a
  `cd examples`. They now resolve their inputs next to the script.
- The `workbench` CI job requested `libxkbcommon-x11` instead of
  `libxkbcommon-x11-0`, and did not install `libxcb-cursor0`, without which Qt
  6.5+ refuses to load its xcb platform plugin.

### Changed

- The `dock-core` test fixture for the GPU CPU-fallback path replicated butane
  40 times 0.1 Å apart, which placed atoms on identical coordinates. It now
  uses a 4×4×3 lattice at 3.0 Å pitch, so every atom pair is physically
  possible.

---

## [0.1.0] — first public release

Initial release of the engine, the Python front end, and the 3-D workbench.

### Added

**Engine (`dock-core`, GPL-3.0-or-later, `#![forbid(unsafe_code)]`)**

- AutoDock Vina and Vinardo empirical scoring, as an analytic first-order
  gradient plus per-conformation forces.
- Precalculated affinity maps over a 4-slot grid. Donor and acceptor
  contributions are split into separate maps, which removes the
  donor–donor pseudo-hydrogen-bonds a single `e_hb` map produces.
- Ligand parameterisation as `6 + n_torsion` DOF: translation, SO(3) rotation
  via the exponential map, and one torsional angle per rotatable bond. The
  torsion set is required to be a cut set of the molecular graph.
- Global search: iterative local search (Metropolis acceptance, local-frame
  perturbation) with `exhaustiveness` independent trajectories on rayon, plus
  an optional island-model LGA.
- Local optimisation: L-BFGS with an Armijo sufficient-decrease test, which is
  required because the trilinear interpolant is only C⁰ across cell faces.
- Pose de-duplication by per-atom RMSD, with a steric-clash partition *before*
  clustering so a physically impossible pose cannot consume a cluster slot.
  `rejected_pose_count` is reported when no clean pose is found at all.
- Grid map export in AutoDock `.map` format, and a configurable point cap so an
  oversized box fails cleanly instead of exhausting memory.
- Optional `gpu` feature: a WGSL compute path over the same interpolation
  kernel as the CPU, with explicit backend reporting and fallback reasons.
- Robustness: malformed `.pdbqt` input produces a catchable error rather than a
  `panic = "abort"` that would kill the calling interpreter.

**Python bindings and front end (`dock-py`, distribution `opendocking`)**

- `Receptor`, `Ligand`, `GridBox`, `GridMaps`, `DockingResult` wrappers with a
  zero-copy contract: input NumPy arrays are borrowed by Rust, never copied;
  outputs are moved once.
- RDKit-backed structure preparation: aromaticity, protonation, Gasteiger
  charges, and polar-hydrogen placement for donors and acceptors.
- A single 79-column PDBQT writer and reader pair, used by both the CLI and the
  workbench, so anything written can be read back.
- `odcli`, the command line: `prep-receptor`, `prep-ligand`, `rec-grid`,
  `dock`, `split`, `info`, `workbench`.
- `odgui`, a standalone entry point for the interactive workbench that does not
  depend on any particular command name and imports neither Qt nor moderngl at
  module scope, so `odgui --check` and `odgui --help` work on a machine with no
  graphics stack.
- Workbench: per-atom element colouring from an explicit PDBQT-type table,
  per-atom radii, pose browsing sorted by energy with RMSD-to-best, a search
  box tool, and docking on a worker thread so the window stays responsive.

**Documentation**

- `docs/SCORING.md` — every term derived from the formula down to the line of
  code, with each deliberate deviation from Vina marked.
- `docs/VERIFICATION.md` — what was actually run, what the results were, and
  an explicit list of what was not verified.
- `docs/LIMITATIONS.md`, `docs/API.md`, `docs/ARCHITECTURE.md`.

### Known limitations

Carried over from [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md); the two that
matter most:

- **No redocking validation.** There is no evidence the engine places a known
  ligand back into its crystallographic pocket.
- **No cross-validation against AutoDock Vina.** Absolute kcal/mol values must
  not be compared with Vina output or literature numbers. Relative ordering,
  pose quality and geometric sanity can be used.

Verified on Windows 11 x86_64 with an NVIDIA RTX 3050 Laptop GPU. macOS, Linux,
AMD, Intel integrated graphics and Apple Silicon are untested.

[Unreleased]: https://github.com/GrassBlock-WE/opendocking-gui/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GrassBlock-WE/opendocking-gui/releases/tag/v0.1.0
