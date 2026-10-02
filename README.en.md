# Open Docking

[English](README.en.md) | [中文](README.md)

A molecular docking toolkit. The Rust engine handles affinity-map
precalculation, empirical scoring and conformational search; the Python front
end handles chemistry-aware preparation (RDKit aromaticity, protonation, charges,
polar hydrogens); and an interactive PyQt6 + ModernGL 3D workbench sits on top
(`dock-py/python/opendocking/workbench/app.py`).

The pipeline itself is classic and well established: the receptor is
precalculated into an affinity grid (trilinear interpolation), the ligand is
optimised with iterated local search plus quasi-Newton descent, and scoring uses
the form of the AutoDock Vina empirical function. On top of that it adds
**analytic first-order gradients**, **pure-safe Rust**
(`#![forbid(unsafe_code)]`) and an **opt-in GPU batch-scoring path** — which is
**not on the search hot path**. See below for what it does and does not do.

> **Two things you must know**
> - **Redocking has been tried, on a small batch, and part of it did not work.**
>    A ligand *already bound* in a crystal structure is re-docked and asked to
>    come back. The `scripts/redock_benchmark.py` run got two of the four complexes
>    under 2 Å (biotin 1.17 Å, benzamidine 1.19 Å), and the box its pocket search
>    picks got one (3PTB, 1.26 Å), while 1HVR's 46-atom flexible ligand is still
>    21 Å and **has not been tuned until the number looked better**.
>    **Those figures carry a premise that has to be stated: they are readings
>    from an engine that has since been fixed and rebuilt.**
>    `docs/VERIFICATION.md` records what that fix was and which build was missed.
>    `scripts/redock_benchmark.py` fetches its corpus live and pins no version of
>    it, and that run left no fingerprint to compare a later one against.
>    So re-running it today is **a new experiment**, not a re-reading.
>    The numbers are still in `docs/VERIFICATION.md` §3.7, where
>    `scripts/redock_benchmark.py`'s sampling settings are set out too.
>    This file therefore does not swap them for new ones.
> - **Absolute energies must not be compared with AutoDock Vina or with
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

# only for the 3D workbench (the engine needs no graphics stack at all)
python -m pip install PyQt6 moderngl numpy-stl
```

Optional GPU build:

```bash
python -m maturin build --release -m dock-py/Cargo.toml --out dist-gpu --features gpu
python -m pip install --force-reinstall --no-deps dist-gpu/opendocking-0.1.0-*.whl
odcli info        # expect: gpu: compiled in and an adapter is available
```

Only **Windows 11 x86_64 with an NVIDIA RTX 3050 Laptop** was actually run, and
every Å, kcal/mol and second figure below was produced on that machine by
`scripts/redock_benchmark.py`, `scripts/pocket_benchmark.py` and
`scripts/gpu_cpu_parity_check.py`. Nothing in this file has been run on anything
else, and the "not verified" list in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) is the long version of that
sentence.
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
| `workbench` | launch the interactive 3D viewer (same as `odgui`) |

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

### 3D workbench

`odgui` is a standalone entry point that depends on no particular command name.

```bash
odgui                                       # empty window; load from the File menu
odgui -r rec_prep.pdbqt -l lig.pdbqt -p poses.pdbqt
odgui --check                               # report whether it can start, no window
```

Left drag rotates, right drag pans, middle drag dollies, wheel zooms, `Frame all`
re-frames. The File menu offers `Load receptor… / Load ligand… / Open poses… /
Dock now`.

**Keyboard** (`?` opens this list at any time; the window knows which keys are live
right now):

| Key | What it does |
|---|---|
| `←` `→` | previous / next pose |
| `↑` `↓` | the same on the pose table; previous / next contact pair on the pair table |
| `W` `A` `S` `D` | orbit the camera up / left / down / right |
| `F` | frame whatever is selected — pair, residue, pose, or all poses if nothing is |
| `K` | run a docking search for the box, exhaustiveness and seed in the panel |
| `C` / `V` | show or hide the contact lines / the site volume cloud |
| `M` | next display representation |
| `E` | export the run to a file |
| `?` | show or hide the shortcut map, `Esc` closes it |

Two of these are deliberate rather than convenient:

- **`↑` `↓` are not installed as window shortcuts.** `QTableWidget` uses them itself
  to move the current row, and that move emits the *same signal a click emits* and
  runs the *same slot* — so the selected contact pair is highlighted, centred and
  named in the status bar by the key exactly as it is by the mouse. Installing a
  second handler for one gesture is how the untested one becomes the broken one.
  They are still listed, because "which keys are live now" is something a reader
  needs; not so that you press them again.
- **`F` does not recompute a framing — it replays the camera move the selection
  already earned**, through the same viewport call. The previous implementation
  took the camera 8.80 Å away from a pair's midpoint and pulled back 3.8×, so a
  2.05 Å contact line fell from 32 px of highlight to 5, and
  `scripts/workbench_interaction_check.py` does not measure those five numbers.
  **They are a threshold-style record**: all five are literals typed into that
  script's own detail string, which you can find by searching for `8.80 A off`.
  `scripts/workbench_interaction_check.py` asserts today's behaviour instead, and a
  selected pair stays within 1.0 Å of the two atoms' midpoint while the camera's
  move and distance change stay under 0.5 Å.
  "I selected this, then asked
  it to frame it, and now it is tiny" is a gesture that undoes itself, and the worst
  possible place for it is a key whose own text says it frames the selection.

  Pressing `K` again while a search is still running **refuses and says why**; it
  does not queue. Two searches sharing one pair of thread slots is exactly how results
  get lost, and a queued request that never runs is a silent failure. The search runs
  on a **worker thread**, so the window does not freeze, and the panel states which
  stage it has reached.

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

A hydrogen bond is a **geometric filter**: H···acceptor under 2.6 Å and a
donor–H···acceptor angle over 120°, and those two figures are `HBOND_MAX` and
`HBOND_MIN_ANGLE` in `dock-py/python/opendocking/workbench/contacts.py`, re-read by
`scripts/docs_claims_check.py`. The check runs in **both** directions, because in a
docked pose the donor is usually the receptor. It is not a quantum calculation: it
does not check that the acceptor's lone pair points back, does not model water
bridges, and assigns no energies. Every contact carries its distance and angle so
the cut-offs can be tightened rather than trusted.

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
protein on both sides) and the **residues lining it**. That turns "there is a
groove near (12, 7, 0)" — a made-up triple `odcli sites` never printed, not a
reading — into something a person can check. A ligand lying flat on the protein
surface scores zero and is
never listed, which `odcli sites` prints and `scripts/pockets_check.py` asserts.

The selected site is also **drawn**: its own grid points, as a translucent
volume you can switch off with `site volume`. A table row reading "groove,
lined by ARG 17A / ASN 14A / THR 2A" is a *claim*; a magenta volume sitting
between those atoms is the reader's own check on it, and no number of extra
columns provides one. Crystal waters and buffer additives are not lining
residues; cofactors are, because they are real chemistry.

The search runs on a **worker thread**. Measured by `scripts/pocket_benchmark.py` at 0.16 s for crambin's 327
atoms, 1.6 s for streptavidin's 1001, and **11.0 s for haemoglobin's 4779**.
A large receptor takes a while, but the window stays usable.

**Pose trust** (the `trust` selector in the control panel):

Poses read from a **pose file** are not the same thing as a run that just finished.
A file stores coordinates, not a conformation — and the engine's gradient and its
out-of-box penalty are properties of the field *as it was scored*, during a memory
search. The panel says so first, then lists four contracts:

| Contract | What it asks |
|---|---|
| `stationarity` | is this pose a stationary point of the field the engine scored? |
| `line-search resolution` | what width could the line search have resolved? |
| `field support` | how close is the ligand to the nearest receptor atom? |
| `inside the box` | did any atom leave the grid? |

Each row has **two independent channels**: the state is written in **its own colour
and its own italic** (`holds` / `fails` / `unmeasured`), and the numbers sit in the
`measured / threshold` column, read from the object that row's tooltip names rather
than recomputed. The three states are three different answers — `unmeasured` is
**not** a pass. It means the result did not carry the number that contract needs,
and it is printed in its own colour and its own italic.

A known failure **outranks** unmeasured: a `no` names which contract failed, and the
unmeasured ones are still listed.

> The full definition of all three states, and why each is drawn the way it is, is
> in the panel's own page — that is where a reader looks first, so it should not
> live only in a source comment.

**Export** (the `E` key, or `Export run to a file` in the File menu — the same
method):

Writes every pose with its energies, terms, verdict, contacts and provenance, and
prints the path in the status bar. **With no pose loaded it says so and writes
nothing** — which is an answer too.

In the file, `0.0` and `null` are not the same thing: the first is **measured as
zero**, the second is **not measured**, and carries `state` and `because` explaining
why not, with the unmeasured entries also collected in `absent_index`.
Writing "did not measure" as 0 records "we know there is nothing there", and
those three are written by `dock-py/python/opendocking/workbench/app.py`, which
decides them field by field at export time.
**There is a coverage gap worth stating: no gate pins the exported file's
schema.** `scripts/pose_trust_check.py` never reads that export — it contains no
`absent_index` and no `json`, and its one occurrence of "export" is a word inside
a detail string, so the distinction above is held by the source rather than by a
gate.

> Exported coordinates are bit-identical to the ones on screen, and the GUI and the
> CLI produce byte-identical files. **That is not evidence the window is right** —
> only that the window is not a second opinion.

## Things to know before you use the results

- **The search box must be given** — either the six numbers, or `--auto-box N`.
  A box covering the whole protein costs tens of gigabytes and is equivalent to
  having no search region. An automatic box is far smaller, which
  `scripts/pockets_check.py` asserts on crambin, and site by site it measured
  between 5.8x and 29.7x smaller than the whole-protein box. That range is
  **recomputed by no script in this tree**: it depends on how the whole-protein
  box is taken, so it is quoted only as an order of magnitude. The same script does
  compute one ratio at run time, but in one place only and for the single site
  ranked 9th on crambin.
  Its denominator is a 22 Å cube typed into `scripts/pockets_check.py` rather
  than a box it measured.
  `scripts/pockets_check.py` prints 1505 against 10648 Å³, which is 7.1x.
  That 22 Å is **not** the `REFERENCE_BOX_SIDE` in
  `dock-py/python/opendocking/core.py` either, which is 20 Å.
  So the range has no instrument behind it at all, not even the taking of it.
- **The pocket search finds enclosed space, not a binding site.** A ligand lying
  flat on the surface scores zero and is never listed, and the largest site is
  not necessarily the one your ligand wants. It is a shortlist to choose from,
  and choosing is the part it leaves to you.
- **Exhaustiveness follows the search box.** It is a count of Monte Carlo walks,
  and walks are spread through the box they search, so a fixed 8 covering a 20 Å
  box is eight times too thin for a 40 Å one — an illustration, not a reading, and
  `scripts/docs_claims_check.py` is what keeps the rule stated as a rule. The rule
  is `exhaustiveness_for_box` in `dock-py/python/opendocking/core.py`, shared by
  all three callers, and it scales **linearly in the box's volume** — one
  calibration point does not license a fitted power law. That calibration point is
  a 39 × 26 × 41 Å box reading **12.88 Å at 16 and 1.26 Å at 64**, two seconds
  either way, and `scripts/redock_benchmark.py` does not print it.
  **Neither reading is any script's output**: they sit in a source comment in
  `scripts/redock_benchmark.py`, which `scripts/pockets_check.py` restates.
  `scripts/pockets_check.py` asserts an answer of 64 for that box.
  `scripts/redock_benchmark.py` runs one exhaustiveness per box, chosen from
  that box's volume, so it cannot print a 16-and-64 pair at all.
  `odcli` derives the
  value when you do not pass `-e` and prints it and
  the reason, and the workbench follows the box until you set it yourself.
- **A snug binding site is found but ranked low.** `scripts/pockets_check.py` puts
  the site ibuprofen occupies on crambin at **12.3 Å³**, **ninth of the twelve**
  the default returns (ninth of sixteen with the cap lifted, rank unchanged),
  behind three larger lumps of surface. Both figures are re-derived from the
  shipped `examples/1crn_prep.pdbqt` each time that gate runs.
  That is not a bug: the search measures the space a ligand *leaves*, not the
  space it occupies, and a tight fit leaves almost none. Where the pocket is
  roomier than the ligand — all four redocking complexes above — the true site
  ranks **first**, every time. The table's fifth column carries the volume the
  ranking is built from, which is why the order is not simply "biggest first".
- **A long winding cleft gets a long box, and a long box is a hard box.**
  3PTB's site is a 31 × 18 × 33 Å cleft, its box is 39 × 26 × 41 Å, and no box both
  holds the 9-atom ligand somewhere in that winding cleft and stays small — all three
  from the `scripts/redock_benchmark.py` run the opening note calls superseded.
  The box size has separate live evidence: `scripts/pockets_check.py` asserts on
  every run that the volume rule answers 64 for it, and computes its volume as
  41574 Å³. The cleft span and the atom count have no such gate. This is a real
  limit of "one box per site" and every pocket code has it.
  What helps is sampling effort proportional to volume, and the same box reads
  12.88 Å on exhaustiveness 16 and **1.26 Å** on 64 in under two seconds, but
  that is likewise only the calibration point in a `scripts/redock_benchmark.py`
  comment, and nothing recomputes it.
- **1HVR is the worst case and is left as it is.** A 46-atom flexible ligand in a
  30 × 38 × 32 Å box gives 21.24 Å at exhaustiveness 64 and 10.53 Å at 128, all
  three from the same superseded `scripts/redock_benchmark.py` run.
  `scripts/redock_benchmark.py` runs one exhaustiveness per box, so the 64-and-128
  pair was printed by no script either. The 30 × 38 × 32 Å box size is a literal
  `scripts/pockets_check.py` carries, and it asserts 64 for that size.
  The pair improves monotonically with sampling — which says that reading was
  under-sampled rather than that the site is wrong — but that is the internal
  consistency of a superseded reading, not a measurement retaken today.
- **Re-docking from the automatic box reproduces crambin's reference energy but
  not its pose.** The script that does this is `scripts/pockets_check.py`, not
  `scripts/redock_benchmark.py` — the latter's four cases are 1STP / 3PTB / 2NNQ /
  1HVR, and **crambin is not among them**. That run by
  `scripts/pockets_check.py` printed −5.50 kcal/mol against the −5.50 reference
  energy it computes itself, at 3.73 Å RMSD.
  The energy lands at the same height and the pose lands in another minimum, and
  the check says so itself — that this is not a reproduction and is not claimed
  as one. The older −5.36 kcal/mol / 4.24 Å pair is not what
  `scripts/pockets_check.py` prints today, and no recorded run produced it at
  all. The evidence is in the comment `scripts/pockets_check.py` carries at its
  lines 1083-1084, which records −5.42 kcal/mol / 3.93 Å as the pre-fix values
  and calls −5.36 "a third number, from neither".
  The −5.30 kcal/mol / 4.35 Å that same comment calls current is not what
  `scripts/pockets_check.py` prints today either, so that comment is a record
  of an earlier state rather than a measurement. The pair is therefore recorded
  here as untraceable, not as the pre-fix version — calling it a version would
  give it a provenance nothing in the tree supports.
  Crambin with ibuprofen has near-degenerate binding modes — a property of the
  energy surface, not a misplaced box.
- **The 1500 Å³ ceiling on a site is a heuristic**, not a derived quantity, and
  `scripts/docs_claims_check.py` re-reads it out of `pockets.py` on every run. It
  is now a guard against a genuinely huge cavity and nothing else.
- It used to have a second, accidental job — deleting the merged surface lumps the
  pre-labelling dilation produced — and that job disappeared with the dilation.
  The ceiling is now inert on crambin, 16 sites either way with the cap lifted and
  12 either way at the default, both counts asserted by
  `scripts/docs_claims_check.py`.
- **A small sealed cavity will not be found at the default probe.** T4 lysozyme
  L99A has one built in on purpose, and this search reports zero sealed
  cavities there: the cavity is about 100 Å³ and a 1.4 Å probe inflates every
  atom enough to fill it, which `scripts/pocket_benchmark.py` is in the corpus to
  demonstrate — 1L96 is one of the structures it walks. Lowering the default would
  be worse, since a 0.5 Å probe closes surface grooves into dozens of spurious
  pockets, so the default stands and `odcli sites` reports what a smaller probe
  would find (`1.4 A -> 0, 1.1 A -> 0, 0.9 A -> 1, …`).
  An empty result you cannot act on is just a sentence.
- **The search slows down quickly with protein size** — 11 seconds for
  haemoglobin's 4779 atoms, from the same `scripts/pocket_benchmark.py` run as the
  table above. It runs on a worker thread so the window stays usable, but it does
  take a while.
- **Option names use underscores**: `--center_x`, not `--center-x`.
- **`use_gpu` defaults to False, and the GPU is not on the search hot path.** GA,
  Monte-Carlo and L-BFGS **never** call it: its only caller is the batch entry
  point `evaluate_conformations`. Pass `use_gpu=True` explicitly. The default
  does not vary with the hardware so that runs stay comparable across machines.

  **The out-of-box half of that path used to be a different function** (Vulkan, RTX
  3050 Laptop, driver 572.83, 1crn receptor, biotin ligand, 22 Å box, 0.5 Å
  spacing, 512 conformations), and all but the driver version are the run
  `scripts/gpu_cpu_parity_check.py` records. In-box the two backends agree to the
  level of `f32` rounding (worst \|GPU−CPU\| = **3.772e-05 kcal/mol**, ranking
  identical across all 427 rows), which is what the "order 1e-7" claim in
  `dock-py/python/opendocking/core.py` predicts. Out of box
  `scripts/gpu_cpu_parity_check.py` measured **all 85 rows exceeding that, worst
  3222.32 kcal/mol, with the ordering inverted**. It cannot measure that
  again, because on the fixed kernel `scripts/gpu_cpu_parity_check.py` now reads
  1.890e-03.

  The cause was that `energy.wgsl` returned a **flat `1000.0` per protruding atom** —
  a sentinel — while the CPU accumulated a per-Ångström, per-axis ramp, so the GPU's
  term was capped at `1000 × protruding atoms` against an unbounded CPU. One
  conformation slid along x (**before** the fix):

  | shift — defect record; today `scripts/gpu_cpu_parity_check.py` | atoms out | CPU (before) | GPU (before) |
  |---|---|---|---|
  | 6.5 | 2 | 705.851 | 1999.808 |
  | 8.0 | 7 | 7882.264 | **6999.985** |

  At shift 8 the GPU scores a pose **better** that the CPU scores much worse — and
  ordering conformations is the entire reason a batch scorer exists, which is the
  part `scripts/gpu_cpu_parity_check.py` exists to keep true. The engine's own
  fixture excluded this region by construction (`dock-core/src/gpu/mod.rs` asserts
  `reach < 6.0`),
  which is why it was never measured. (`report_backend` carries a `backend` field
  throughout, so a GPU run and a CPU run were **always** distinguishable from their
  reports. A build with no `gpu` feature used to be the one case that lied:
  `use_gpu=True` fell back to the CPU and gave no reason — and "I asked and was
  refused" is not the same claim as "it ran on the CPU on purpose", so a caller
  could not tell a refusal from an ignored request. **Fixed 2026-10-02.** The
  contract is an exclusive-or over four cases: a report names the `gpu` backend
  **or** carries a non-empty `gpu_skip_reason`, never both and never neither;
  a call that asked for the GPU and did not get one therefore always carries a
  reason; and a call that passed `use_gpu=False` carries none, because it declined
  nothing. Seven sites can decline, and each names itself: `this build was
  compiled without the gpu feature`, `empty population`, `ligand has more atoms
  than the workgroup size`, `a ligand coordinate is too large for the f32 grid`,
  `no usable GPU adapter`, `could not pack the batch`, and `the compute pass
  failed`. Seven different causes need seven different responses, which is why
  `backend == "cpu"` alone is not the answer. The contract, and the test holding
  it, are in `dock-core/src/search/mod.rs`: `evaluate_population`'s doc comment and
  `a_decline_is_always_self_explaining`.)

  The kernel now uses exactly the CPU's expression — per axis, per ångström, summed
  over the three faces, measured against the **box's faces** rather than the last
  tabulated point because `estimate_dims` rounds up and the tabulated volume can be
  slightly larger than the box — and both sides go through the one definition,
  `grid.rs::out_of_box_violation_per_axis`. **But a fix you can read in the source
  is not a fix that has been demonstrated**, so the two gates have to be told apart:

  - `shader_and_rust_agree_on_the_out_of_box_penalty` is a **textual** check: it
    parses the constant and requires the kernel to contain that per-axis expression.
    It catches "reverted to a flat sentinel"; it cannot catch a flipped sign or a
    `bmax` passed wrong, because it never computes either side.
  - `the_cpu_and_gpu_paths_agree_outside_the_box_too` is the **numerical** one: it
    slides conformations out past the box face, compares the two backends point by
    point, and asserts there is no ranking inversion.

  The second **skips rather than fails on a machine with no adapter** (it prints
  `NOT MEASURED` and returns), so its being green in CI is **not** evidence that this
  source has been demonstrated. **To check, run the gate**:
  `scripts/gpu_cpu_parity_check.py` reports its own total, measures in-box and
  out-of-box separately, and skips on a machine with no adapter while counting the
  skip toward that total.
- **Do not use `|grad|` to decide whether a pose converged.** The grid is a
  trilinear interpolant and therefore only C⁰, and a returned pose nearly always
  has an atom near a cell face. In this tree's run of `examples/audit_poses.py`,
  the five poses read `|grad|` 4.0–8.3 and L2 norms 5.3–12.5, while a step along
  −grad moves the energy by only 1e-9–1e-8 kcal/mol.
  **That range follows the build and the machine rather than being a constant.**
  `docs/VERIFICATION.md` records that the parallel search's walk order follows the
  thread count, that different CPUs return different pose sets, and that two
  builds produced two different pose sets. The earlier 2–6 was rounded outward
  from a 2.3–5.9 reading in `docs/VERIFICATION.md` and is not the output of
  `examples/audit_poses.py`, which does not contain what that script reports
  today either. Use a line search.
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

**GPL-3.0-or-later**, full text in [`LICENSE`](LICENSE), and the `3.0` in that
identifier is not a measurement of this project but a licence number, which is why
`scripts/check_text_encoding.py` is what watches the file. Because this project
distributes GPL-3.0 code, any binary distribution must provide the complete
source under GPL-3.0 as well, and `Cargo.toml` and `dock-py/pyproject.toml` declare
that same string in their `license` fields. Provenance of the functional forms and
weights is in [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md).
