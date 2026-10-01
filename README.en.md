# Open Docking

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
> 1. **Redocking has been tried, on a small batch, and part of it did not work.**
>    `scripts/redock_benchmark.py` docks a ligand that is *already bound* in a
>    crystal structure and asks whether it comes back: the engine gets two of
>    four under 2 Å (biotin 1.17 Å, benzamidine 1.19 Å), and the box the pocket
>    search picks gets one (3PTB, 1.26 Å). 1HVR's 46-atom flexible ligand is
>    still 21 Å and **has not been tuned until the number looked better**. The
>    numbers and how they were collected are in
>    [`docs/VERIFICATION.md`](docs/VERIFICATION.md) §3.7.
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
python -m pip install dist/opendocking-0.1.0-*.whl

# only for the 3-D workbench (the engine needs no graphics stack at all)
python -m pip install PyQt6 moderngl numpy-stl
```

Optional GPU build:

```bash
python -m maturin build --release -m dock-py/Cargo.toml --out dist-gpu --features gpu
python -m pip install --force-reinstall --no-deps dist-gpu/opendocking-0.1.0-*.whl
odcli info        # expect: gpu: compiled in and an adapter is available
```

Only **Windows 11 x86_64 with an NVIDIA RTX 3050 Laptop** was actually run.
macOS, Linux, AMD discrete, Intel integrated and Apple Silicon are **untested**.

## Usage

### Command line

The full CLI is **`odcli`**.

| Subcommand | What it does |
|---|---|
| `info` | print engine version, scoring parameters, GPU status, parallel backends |
| `prep-receptor` | `.pdb` → receptor PDBQT |
| `prep-ligand` | any structure → ligand PDBQT + a property report |
| `rec-grid` | precalculate maps and write AutoDock `.map` files |
| `dock` | dock one ligand or a whole directory |
| `sites` | list candidate binding sites in a receptor, with lining residues |
| `split` | split a multi-model PDBQT into one file per pose |
| `workbench` | launch the interactive 3-D viewer (same as `odgui`) |

A complete run on the bundled example data:

```bash
cd examples

# receptor PDB -> PDBQT
odcli prep-receptor -r receptor.pdb -o rec_prep.pdbqt

# ligand SDF -> PDBQT (also prints torsions, DOF count, atom classification)
odcli prep-ligand -l ibuprofen.sdf -o ibuprofen_prep.pdbqt

# what does this receptor have to offer?
odcli sites -r 1crn_prep.pdbqt

# precalculate + dock. The box is still required: give the six numbers, or
# take the Nth site with --auto-box
odcli dock -r 1crn_prep.pdbqt -l ibuprofen_prep.pdbqt \
           --auto-box 2 -e 8 -o poses.pdbqt
```

The search box is **still mandatory** — there are just two ways to supply it.
`odcli` has always refused to derive it from the receptor, because a
whole-protein box is enormous in memory and useless as a search region.
`--auto-box N` takes the Nth site from `odcli sites`, and `--box-padding`
sets the clearance on each side (default 4 Å). A site smaller than the ligand
is widened to the engine's floor (`2 × radius + 1 Å`) and the widening is
reported; a box you set yourself is left exactly as you gave it.

Common options:

```
-e, --exhaustiveness N     independent search walks (defaults to the box's volume — see below; use 32 to be thorough)
-m, --num_modes N          how many de-duplicated conformations to output (default 9)
--rmsd-cutoff R            RMSD below which two poses count as the same (default 1.0 Å)
--scoring {vina,vinardo}   scoring function
--spacing S                grid spacing (default 0.375 Å)
--mode {mc,lga,both}       global search strategy
--seed N                   fix the random seed (for reproducibility)
--json                     machine-readable output
--auto-box N               use the Nth site instead of the six box values
```

### Python

```python
from opendocking import Receptor, Ligand, GridBox, dock

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

**Display modes** (the `display` selector in the control panel):

| Mode | What it draws |
|---|---|
| `Space-filling` | a sphere per atom, bonds as thin lines |
| `Ball and stick` | smaller atoms plus a cylinder along each bond, each half coloured by its own atom |
| `Skeletal` | bonds only |
| `Ribbon` | a smooth swept ribbon along the backbone, coloured by an estimated secondary structure |
| `Cartoon` | the ribbon plus side chains as thin sticks |

The last two need a protein backbone; a structure without one falls back to
ball-and-stick and says so in the status bar. The status bar also states where
each structure's bonds came from — read out of the file, from residue
templates, or inferred from distances. A ball-and-stick picture makes a wrong
bond as convincing as a right one, so that line is not decoration.

**Interaction analysis** (the `interactions` toggle in the control panel):

A pose buried in a hydrophobic groove and a pose clinging to the outside of the
protein look identical from the outside. The workbench now says which is which:

- contacts are drawn as **dashed lines**, coloured by class, with a legend
  beside them (hydrogen bond near-white, polar cyan, hydrophobic violet, other
  slate);
- an `interactions` table lists every residue involved with its hydrogen-bond
  count, contact count and closest approach, **ranked hydrogen bonds first** —
  ranking by distance alone puts a grazing contact ahead of a hydrogen bond;
- **clicking any row flies the camera onto that residue's contacts**.

A hydrogen bond here is a **geometric filter**: H···acceptor under 2.6 Å and a
donor–H···acceptor angle over 120°, checked in **both** directions because in a
docked pose the donor is usually the receptor. It is not a quantum calculation:
it does not check that the acceptor's lone pair points back, does not model
water bridges, and assigns no energies. Every contact carries its distance and
angle so the cut-offs can be tightened rather than trusted.

> **Where residue names come from**: the synthetic example receptor
> (`rec_prep.pdbqt`) has every residue called `REC`, so against it the workbench
> can report *how many* contacts there are but not attribute them, and the
> table is empty. That is not a failure — it genuinely has no residue identity.
> Load a real protein to get residue numbers.

The picture itself is 4x multisampled, has depth fog (aerial perspective, which
is what makes the front of a large receptor distinguishable from the back),
two lights plus a rim term, and a gradient background rather than one flat
colour. The fog range follows the camera distance, so it stays out of the way
when a ligand fills the window and does real work when a big receptor is zoomed out.

**Candidate binding sites** (`Find pockets` / `site` in the control panel):

When you load a receptor the workbench **searches for pockets by itself**, puts
the search box on the first site it finds, and selects that row — so you can
see where the box came from, and click any other row to move it. Selecting a
row moves the box, the three coordinate spins and the camera together, and the
numbers on screen are the numbers the engine gets.

> **This used to be the receptor's centroid**, the arithmetic mean of every
> atom. For a globular protein that is inside the dense core, so the search
> region sat in solid protein while three spin boxes displayed the result as
> though it were a considered answer. When no site is found the workbench says
> so and leaves the box alone, rather than implying it found one.

What it searches for is **enclosed space, not a binding site**. Each entry
reports its centre, extent, a burial score (how many of the three axes have
protein on both sides) and the **residues lining it** — which turns "there is a
groove near (12, 7, 0)" into something a person can check. A ligand lying flat
on the protein surface scores zero and is never listed.

The selected site is also **drawn**: its own grid points, as a translucent
volume you can switch off with `site volume`. A table row reading "groove,
lined by ARG 17A / ASN 14A / THR 2A" is a *claim*; a magenta volume sitting
between those atoms is the reader's own check on it, and no number of extra
columns provides one. Crystal waters and buffer additives are not lining
residues; cofactors are, because they are real chemistry.

The search runs on a **worker thread**. Measured: 0.16 s for crambin's 327
atoms, 1.6 s for streptavidin's 1001, and **11.0 s for haemoglobin's 4779**.
A large receptor takes a while, but the window stays usable.

## Things to know before you use the results

- **The search box must be given** — either the six numbers, or `--auto-box N`.
  A box covering the whole protein costs tens of gigabytes and is equivalent to
  having no search region. An automatic box is far smaller: on crambin it is
  between 5.8x and 29.7x smaller than the whole-protein box, site by site — the
  figure depends on how the whole-protein box is taken, so no single number is
  quoted here.
- **The pocket search finds enclosed space, not a binding site.** A ligand lying
  flat on the surface scores zero and is never listed, and the largest site is
  not necessarily the one your ligand wants. It is a shortlist to choose from,
  and choosing is the part it leaves to you.
- **Exhaustiveness follows the search box.** It is a count of Monte Carlo walks,
  and walks are spread through the box they search, so a fixed 8 that covers a
  20 Å box is eight times too thin for a 40 Å one — and an under-sampled search
  returns a bad *pose*, which looks exactly like the box being wrong. Measured
  on a 39 × 26 × 41 Å box: **12.88 Å at 16, 1.26 Å at 64**, two seconds either
  way. `odcli` derives it when you do not pass `-e` and prints the value and the
  reason; the workbench follows the box until you set it yourself, and then
  leaves it alone. The rule is `opendocking.core.exhaustiveness_for_box`, shared
  by all three callers, and it scales **linearly in the box's volume** — one
  calibration point does not license a fitted power law.
- **A snug binding site is found but ranked low.** On crambin with ibuprofen, the
  site the ligand occupies is **12.3 Å³** and comes **ninth of the twelve** the
  default returns (ninth of sixteen with the pocket-count cap lifted; the rank
  does not change), behind
  three larger lumps of surface. That is not a bug: the search measures the space
  a ligand *leaves*, not the space it occupies, and a tight fit leaves almost
  none. Where the pocket is roomier than the ligand — all four redocking
  complexes above — the true site ranks **first**, every time. The table's fifth
  column carries the volume the ranking is built from, which is why the order is
  not simply "biggest first".
- **A long winding cleft gets a long box, and a long box is a hard box.** 3PTB's
  site is a 31 × 18 × 33 Å cleft and its box is 39 × 26 × 41 Å. No single box
  both contains a 9-atom ligand somewhere in a winding cleft and stays small;
  this is a real limit of "one box per site" and every pocket code has it. What
  helps is sampling effort proportional to volume: the same box reads 12.88 Å
  at exhaustiveness 16 and **1.26 Å** at 64, in under two seconds either way.
- **1HVR is the worst case and is left as it is.** A 46-atom flexible ligand in a
  30 × 38 × 32 Å box gives 21.24 Å at exhaustiveness 64 and 10.53 Å at 128 —
  improving monotonically with sampling, which says the search is under-sampled
  rather than that the site is wrong.
- **Re-docking from the automatic box reproduces crambin's reference energy but
  not its pose.** Docking ibuprofen into the site the reference pose occupies
  gives −5.36 kcal/mol against the reference's −5.50, at 4.24 Å RMSD. The energy
  is reproduced; the pose is not. Crambin with ibuprofen has near-degenerate
  binding modes, and that is a property of the energy surface, not a misplaced box.
- **The 1500 Å³ ceiling on a site is a heuristic**, not a derived quantity. It is
  now a guard against a genuinely huge cavity and nothing else. It used to have a
  second, accidental job — deleting the merged surface lumps the pre-labelling
  dilation produced — and that job disappeared with the dilation, so the ceiling
  is now inert on crambin (16 sites either way with the cap lifted, 12 either
  way at the default) and a check says so explicitly.
- **A small sealed cavity will not be found at the default probe.** T4 lysozyme
  L99A has one built in on purpose, and this search reports zero sealed
  cavities there: the cavity is about 100 Å³ and a 1.4 Å probe inflates every
  atom enough to fill it. Lowering the default would be worse — a 0.5 Å probe
  closes surface grooves into dozens of spurious pockets — so the default
  stands and `odcli sites` reports what a smaller probe would find
  (`1.4 A -> 0, 1.1 A -> 0, 0.9 A -> 1, …`). An empty result you cannot act on
  is just a sentence.
- **The search slows down quickly with protein size**: 11 seconds for
  haemoglobin's 4779 atoms. It runs on a worker thread so the window stays
  usable, but it does take a while.
- **Option names use underscores**: `--center_x`, not `--center-x`.
- **`use_gpu` defaults to False.** The GPU accumulates in `f32`; `core.py`'s own
  docstring puts the relative error at order 1e-7, which **was not re-measured
  here** (this machine reports no usable GPU); a default that varies with the
  hardware would break
  cross-machine reproducibility. Pass `use_gpu=True` explicitly, then read the
  fallback reason out of `report_backend`.
- **Do not use `|grad|` to decide whether a pose converged.** The grid is a
  trilinear interpolant and therefore only C⁰, and in practice a returned pose
  nearly always has an atom close to a cell face, where `|grad|` can reach 2–6
  while the
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
