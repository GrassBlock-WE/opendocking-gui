# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The project is pre-1.0. The `0.x` line is where the interfaces still move.

---

## [Unreleased]

### Added

- **`scripts/redock_benchmark.py`**: docking a ligand that is *already bound* in
  a crystal structure, and asking whether the search finds it back. This is the
  check the project was missing. Three columns, because they are three
  different claims: a box derived from the bound ligand measures the engine; the
  box the pocket search picks measures the whole chain; and the box of the site
  that *actually* holds the bound pose separates a bad ranking from a bad box.
  Sampling effort is chosen from the box's volume rather than fixed — a fixed
  value quietly under-samples every large box and makes the box look like the
  problem when the search was. The receptor is narrowed to the ligand's own
  chain and prepared once, and the search and the engine are then given **the
  same file**, because measuring two different molecules produces two numbers
  that mean nothing. Not a CI gate: it needs a network, and there is no ground
  truth a pass/fail could be honestly measured against.
- **The pocket table shows why it is ordered the way it is.** The list is
  ranked by `volume ** (1/3) * burial` and the user could see neither term, so
  the order looked arbitrary — a 88 Å³ groove outranked the 20 Å³ pocket a
  ligand was sitting in and nothing on screen said why. A fifth column carries
  the site's own volume, which is the measure the ceiling is applied to and the
  one the fix below made meaningful.
- **The site is drawn, so it can be checked.** The workbench renders the
  selected site's own grid points as a translucent cloud, and the `site volume`
  toggle gates it. A table row saying "groove at (12.8, 7.2, 0.1), lined by
  ARG 17A, ASN 14A, THR 2A" is a claim; a magenta volume sitting between those
  atoms is the reader's own check on it, and no amount of table columns
  provides one. The cloud is drawn with depth testing off — a site is by
  definition inside protein, so a depth-tested one is hidden by the very atoms
  that make it a site, and in space-filling about half of it disappeared.
  `pockets_check.py` asserts the cloud and the table describe the same
  geometry: one point per voxel, the same extent, the same centre, and every
  point in free space.
- **`odcli sites` says what a null result means.** "No sealed cavity" is a
  statement about the probe, not about the protein, and the two are easy to
  confuse. T4 lysozyme L99A has a cavity *deliberately engineered* into it and
  this search reports **zero** sealed cavities there, because a 1.4 Å probe
  inflates every atom enough to fill a 100 Å³ hole. When nothing sealed is
  found, the command now reports what a smaller probe would find
  (`1.4 A -> 0, 1.1 A -> 0, 0.9 A -> 1, …`) and says plainly that a smaller
  probe also invents spurious pockets, which is why it is not the default. An
  empty result you cannot act on is just a sentence.
- **`scripts/pocket_benchmark.py`**: the search run over seven real structures
  from RCSB — crambin, T4 lysozyme L99A, trypsin, streptavidin, myoglobin,
  haemoglobin and a metalloproteinase — printing what came out, including the
  unflattering parts. Deliberately **not** a CI gate: it needs a network, and
  there is no ground truth here that a pass/fail could be honestly measured
  against. `pockets_check.py` is the gate and needs none.
- **The search no longer freezes the window.** It runs on a worker thread.
  Measured across the benchmark set: 0.07 s for crambin's 382 atoms, 2.0 s for
  streptavidin, **11.0 s for haemoglobin's 4779** — and that last one ran
  inline, so loading a large receptor froze the window for eleven seconds with
  the status text set and the event loop unable to paint it.
- **The search box stops being a guess.** Loading a receptor used to centre the
  box on the receptor's centroid — the arithmetic mean of every atom, which for
  a globular protein is inside the dense core. The search region was therefore
  sitting in solid protein while three spin boxes displayed the result as
  though it were an answer. `odcli` had always refused to guess, calling a
  whole-protein box "useless as a search region"; the workbench now does the
  same thing properly instead of guessing quietly.
  `opendocking.workbench.pockets` finds candidate sites on a grid in the
  LIGSITE family — protein-solvent-protein events, plus a flood fill from the
  grid boundary to tell a sealed cavity from an open groove — with no Qt and
  no display. Each site carries its bounding geometry, a burial score, its own
  grid points, and the residues lining it.
  - The workbench lists the sites, selects the first, places the box on it and
    flies the camera there. Selecting any other row moves the box, the three
    spins and the camera together, so the numbers on screen and the numbers
    the engine gets cannot disagree. A receptor with no site says so and
    leaves the box alone rather than implying one was found.
  - `odcli sites -r RECEPTOR` prints the list, with lining residues and both
    the site size and the box built from it. `--json` for machines.
  - `odcli dock` and `odcli rec-grid` accept `--auto-box N` to take the Nth
    site instead of the six explicit values. **The six values stay mandatory
    otherwise** — the rule moved out of argparse and into `_resolve_box`, which
    is the only place that knows whether `--auto-box` was passed, so the error
    can say what to do instead of "the following arguments are required".
  - A site can be smaller than the ligand that has to fit in it, so an
    automatic box is widened to the engine's own floor (`2 x radius + 1 A`) and
    the widening is reported. A box you set yourself is left alone and gets the
    engine's exact message.

- **Pose/receptor interaction analysis.** The workbench can now say what the
  pose is doing: a residue-level table of the interface, dashed lines between
  contacting atoms coloured by class (hydrogen bond, polar, hydrophobic,
  other), a legend keyed to the same colour table the renderer uses, and a
  click on any row that flies the camera onto that residue's contacts.
  `opendocking.workbench.contacts` does the work with no Qt and no display, and
  `scripts/contacts_check.py` verifies it headlessly (39/39). Hydrogen bonds
  are a geometric filter — H···acceptor under 2.6 Å and a donor–H···acceptor
  angle over 120° — checked in **both** directions, because in a docked pose the
  donor is usually the receptor. The distance and angle come back on every
  contact so the cut-offs can be tightened rather than trusted. A hydrogen
  whose parent cannot be resolved is never drawn as a bond.
- **Anti-aliasing, depth fog and a background gradient.** The viewport asks the
  surface format for 4x multisampling, the shaders gained a second light, a rim
  term and aerial perspective, and the backdrop is a gradient rather than one
  flat colour. Fog scales with the camera distance, so it does nothing on a
  ligand viewed up close and is doing real work on a large receptor.
- The controls panel is now scrollable. It was a plain widget in a dock, so once
  the panel was taller than the window the Dock button and everything below it
  were unreachable.
- `examples/crambin_pose.pdbqt`: ibuprofen docked into crambin with this engine
  (9 poses, best −5.5 kcal/mol, seed 20260930, box covering the protein). It is
  here so the interaction checks have a pose with real residue names to work
  on — the example receptor is a synthetic blob whose residues are all called
  `REC`, and against it every contact is correctly anonymous.
- **Five display modes in the workbench**, chosen from a `display` selector:
  space-filling, ball-and-stick, skeletal (sticks only), ribbon and cartoon.
  The last two need a protein backbone; a structure without one falls back to
  ball-and-stick and says so in the status bar rather than drawing an empty
  ribbon. The backbone ribbon is a swept surface along the CA trace, oriented by
  parallel transport so it does not twist, and coloured by a geometric estimate
  of secondary structure.
- `opendocking.workbench.structure`, a GUI-free module that reads a structure
  with its residue identity intact and decides its connectivity. `structure.py`
  and `geometry.py` are importable without Qt or a display, and
  `scripts/structure_bond_check.py` and
  `scripts/representation_geometry_check.py` verify them headlessly.
- The status bar now reports where each structure's bonds came from — read out
  of `ROOT`/`BRANCH` records, out of amino-acid templates, or inferred from
  distances — and names any bond that could not be taken at face value.

### Fixed

- **The pocket search lost every real binding site it found.** A one-voxel
  isotropic dilation of the burial mask, applied *before* connected-component
  labelling, was there to stop thin sheets being reported as slivers. It did the
  opposite: a surface groove is one voxel thick, so growing it one voxel reaches
  across its neck and merges the site into the protein's outer surface. The
  merged lump then failed the volume ceiling and took the pocket with it. On the
  component holding the bound ligand, measured before and after the growth:

  | complex | undilated | after one dilation |
  |---|---|---|
  | 1STP biotin | 314 voxels, 1305 Å³ box → kept | 4905 voxels, 71909 Å³ |
  | 3PTB benzamidine | 698 voxels, 10161 Å³ box → kept | 4114 voxels, 77893 Å³ |
  | 1HVR XK2 | 1063 voxels, 8008 Å³ box → kept | 6303 voxels, 109965 Å³ |

  A sensible pocket before the growth, a lump of the whole protein after it, and
  the list looked perfectly normal throughout — the sites were not mis-ranked,
  they were **not in the list at all**. Growth now happens per component, after
  labelling, where it can no longer merge anything. On all four redocking
  complexes the bound ligand's own site now comes out **first**; before this it
  was absent. The search is also faster, the dilation having been the main cost.
- **The volume ceiling was applied to a site's bounding box**, which is wrong in
  both directions and discarded real pockets. A winding cleft is a long thin
  tube: 1HVR's site is **544 Å³ of space in an 8008 Å³ box**, so the ceiling
  rejected it, and a synthetic dog-legged slot reproduces the shape exactly —
  1016 Å³ in a 9698 Å³ box, which the old ceiling dropped. The ceiling is now
  on the site's own volume (`voxels * spacing³`), and `Pocket` carries that
  number so it can be read rather than recomputed. A consequence worth stating:
  the ceiling is now **inert on crambin** (16 sites either way; it used to be
  23 → 20), because the lumps it was compensating for are no longer produced.
  `pockets_check.py` asserts that explicitly — a filter that quietly stops
  firing is how a filter rots.
- **`max_pockets` default 8 → 12.** On crambin with ibuprofen docked into it, the
  site the ligand actually occupies ranks **ninth of sixteen**, and its box holds
  all sixteen ligand atoms. An eight-entry shortlist cut it off, so the feature
  silently did not offer a binding site that was sitting in the ninth slot.
- The `Real launch, then close` check could never pass on CI, and had been
  failing on every run since it was introduced. CI runs it under `xvfb-run`,
  which starts a display with **no window manager**, and `wmctrl` asks the
  window manager for the client list over EWMH — with no WM it can neither find
  the window nor close it, so the loop ran out its 60 s deadline and the
  process was killed. Finding the window now uses `xdotool search`, which
  walks the X tree with `XQueryTree` and needs no window manager.
- `xdotool windowclose` is not a way to close a window, and treating it as one
  is worse than having no mechanism at all: it destroys the X window outright,
  so Qt never runs `closeEvent` and the process keeps running with nothing on
  screen. CI showed the window found at 0.6 s and the process still alive at
  20.6 s, which is a destroyed window rather than a slow close. Closing goes
  back through `wmctrl -c`, so the workbench job now starts a minimal window
  manager (`openbox`) — a real desktop has one, and the check is supposed to
  model a real desktop. `xdotool windowclose` is kept only to clean up a window
  that could not be closed politely, and is never reported as a verified
  shutdown.
- The `wmctrl -l` branch was unreachable even where a window manager was
  present. That output has four fields (`<id> <desktop> <host> <title>`) and
  the title contains spaces, so it must be the unsplit remainder; the parser
  split at most five and then required exactly five, which no four-field line
  can produce.
- When no close mechanism existed, the verdict was `rc is not None` — but by
  then `odgui` had usually been killed by the check's own cleanup, so that
  exit code was manufactured by the check and said nothing about the
  application. It now requires that the process was still running when its
  window was found.
- `odgui_launch_check.py` is no longer Windows-only: it runs on X11 and
  reports which mechanism found and closed the window.
- `scripts/x11_window_parse_check.py` (new) parses the launch check's window
  lookup against each tool's real output format, and checks that having
  `xdotool` installed is not mistaken for having a way to close a window.
  11/11, no X server needed, runs on either platform, wired into CI.
- The workbench CI step was still labelled "(50 checks)" after the interaction
  check grew to 75.
- Two interaction checks could silently stop existing. Both were guarded by an
  `if` on their own precondition, so when it did not hold the check was never
  registered — not passed, not skipped, simply absent. On a headless runner
  that made the total read 74 instead of 75, and a smaller total is
  indistinguishable from a smaller scope. Both now record a SKIP with the
  reason, and the total is 75 in every environment. Reverse-verified by forcing
  `PIXELS_OK` off to reproduce the runner's condition: 61 passed, 0 failed,
  14 skipped, 75 checks — the same 61 passes CI reports, plus the one that used
  to vanish.

- Three checks put `dock-py/python` on `sys.path` ahead of everything else, so
  they imported the **source copy** of `opendocking`. A clean checkout cannot
  load that copy: the compiled `_dockpy` extension is gitignored and only exists
  in site-packages, so every one of them died on a fresh CI runner with "The
  Open Docking native extension is not available" — while working perfectly on
  the maintainer's machine, where a development build had left the extension
  sitting in the source tree. They now prefer the installed package and only
  fall back to the source tree when there is none, which is also the more useful
  default: it is what a user gets. `workbench_smoke.py` had the right pattern
  written down already; the new scripts did not copy it.- `find_contacts` accepted `hbond_max` and `hbond_min_angle` and **ignored
  them**: the helper read the module constants directly, so only the
  close-contact cut-off was ever applied. A caller tightening the thresholds to
  check a borderline pose would have been told it had worked.
- `workbench_smoke.py` reversed the red and blue channels before writing its
  screenshots, so a grey protein rendered blue and every red oxygen rendered
  blue. Anyone reading those images was hunting a colour bug that was in the
  harness.
- The hydrophobic interaction lines were drawn in grey, over a grey protein.
  Sixteen of the forty-eight contacts in a real pose were invisible — a contact
  that cannot be seen is not reported to anyone, however correctly it was
  computed. They are violet now, and the hydrogen-bond line is near-white
  rather than the yellow the search box already uses.
- `workbench_interaction_check.py` put only `examples/` on `sys.path`, so
  `opendocking` resolved to the *installed wheel* and the check was silently
  testing the last build rather than the working tree. Same class of mistake as
  the launch check's stale count, one layer up: a check that is not looking at
  the thing it claims to look at.
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
- **`prepare-receptor` destroyed all residue identity.** Every atom of a
  receptor was written as residue `REC`, residue number 1, with no chain and no
  atom name, so a 46-residue protein came out as one anonymous blob. Nothing
  downstream could group atoms into residues, infer which atoms are bonded, or
  draw a backbone. The name, residue, chain and number now come from each
  atom's own PDB record.
- A polar hydrogen added during receptor preparation was written with its heavy
  atom's name, producing residues with three atoms called `N`. Hydrogens are now
  written as `H`.
- `AddHs` sometimes returns a hydrogen with no position, which was written as
  `(0, 0, 0)` — an atom at the coordinate origin, thousands of ångströms from
  its own parent, which then docked as though it were real. Three of them
  appeared on the crambin fixture. A hydrogen that did not land within
  0.7–1.35 Å of the atom it was added to is now dropped with a warning.
- **Bond perception in the viewer** was a flat "closer than 1.95 Å" rule, which
  welds atoms of different residues together across a protein interface. On the
  crambin fixture that rule invents 48 bonds between residues. Connectivity now
  comes from `ROOT`/`BRANCH` records where the file has them, from amino-acid
  templates where the file is a protein, and from a covalent-radius rule only
  for a flat small molecule with neither — and in every case it is audited and
  the source is reported.
- A declared bond whose two atoms are further apart than the two elements can
  possibly be is no longer drawn. The docked-pose writer emits a forked group
  such as a carboxyl as a flat chain, so its files declare a bond between two
  oxygens 2.23 Å apart; the viewer drops it and says why. The writer itself
  still needs nested `BRANCH` records and is listed under Known limitations.
- `scripts/odgui_launch_check.py` used `ctypes.windll`, which does not exist off
  Windows, so the check crashed instead of running on Linux. It now closes the
  window through the X11 window manager there, and reports a partial result if
  no window manager tool is installed rather than claiming a clean shutdown it
  never tested.

### Fixed

- **Crystal waters and buffer additives are no longer reported as the residues
  lining a pocket.** The lining list is meant to answer "what is the wall of
  this site", and a water is a lattice artefact rather than a wall: across
  seven structures, streptavidin had more waters than residues in its top
  site's lining. Cofactors are a different case and are deliberately kept —
  "this pocket is lined by the haem" is real chemistry and worth knowing.
- **Clearing the site selection no longer leaves the previous site's volume
  cloud on screen**, sitting inside whatever the new selection put there.

### Changed

- The `dock-core` test fixture for the GPU CPU-fallback path replicated butane
  40 times 0.1 Å apart, which placed atoms on identical coordinates. It now
  uses a 4×4×3 lattice at 3.0 Å pitch, so every atom pair is physically
  possible.

### Known limitations

- **The pocket search finds enclosed space, not a binding site.** It reports
  cavities and grooves with their geometry and lining residues, ranked by size
  and how enclosed they are. A ligand sitting on a flat surface scores zero and
  is never mentioned, and the largest site is not necessarily the one a given
  ligand wants. It is a shortlist to choose from, and the caller is expected to
  let the user choose.
- **A snug binding site is found but ranked low, and that is not fixable by
  ranking harder.** On crambin with ibuprofen docked into it, the site the
  ligand occupies is **12 Å³** and comes **ninth of sixteen**; three larger
  lumps of surface come first. The reason is worth knowing because it is not a
  bug: the search measures the space a ligand *leaves*, not the space it
  occupies, and where a ligand fits snugly that space is nearly nothing. On the
  four complexes whose pockets are roomier than their ligands, the true site
  ranks **first** every time. An earlier version of this file appeared to rank
  crambin's second, and that was luck — it ranked second only because an
  unrelated ceiling happened to delete three larger lumps of surface first.
- **A long cleft gets a long box, and a long box is a hard box.** 3PTB's site is
  a 22 × 17 × 28 Å cleft and its box is 39 × 26 × 41 Å. There is no single box
  that both contains a 9-atom ligand somewhere in a winding cleft and stays
  small; this is a real limit of "one box per site" and every pocket code has
  it. What helps is sampling effort proportional to volume, which is why
  `redock_benchmark.py` picks it that way and prints it.
- **1HVR is the worst redocking case and is left as it is.** A 46-atom flexible
  ligand in a 30 × 38 × 32 Å box returns 21.24 Å at exhaustiveness 64 and
   10.53 Å at 128 — improving monotonically with sampling, which says the
  search is under-sampled rather than that the site is wrong. It is not tuned
  until the number looks better.
- **Re-docking from the automatic box reproduces crambin's reference energy but
  not its pose.** Docking ibuprofen into the site crambin's reference pose
  occupies gives −5.42 kcal/mol against the reference's −5.50, at 3.93 Å RMSD.
  The energy is reproduced; the pose is not. Crambin with ibuprofen has
  near-degenerate binding modes, and landing in a different one is a property
  of the energy surface rather than evidence that the box was misplaced — the
  box does contain all 16 ligand atoms and is lined by the residue that
  hydrogen-bonds them. `scripts/pockets_check.py` asserts the energy and
  *records* the RMSD without asserting anything about it, because a small RMSD
  would be asserting that this run happened to find this minimum.
- **The 1500 Å³ volume ceiling on a site is a heuristic.** It is a guard against
  a genuinely huge cavity, and it is a parameter rather than a derived quantity:
  nothing here justifies the particular number. It used to have a second,
  accidental job — removing the lumps of merged outer surface that the
  pre-labelling dilation produced — and that job is gone now that the dilation
  is, so the ceiling is inert on crambin and the checks say so.
- **A small sealed cavity will not be found at the default probe.** T4 lysozyme
  L99A has one built in on purpose, and this search reports zero sealed
  cavities in it: the cavity is about 100 Å³ and a 1.4 Å probe inflates every
  atom enough to fill it. Lowering the default would be worse — a 0.5 Å probe
  closes surface grooves into dozens of spurious pockets — so the default
  stands and `odcli sites` reports what a smaller probe would find instead of
  leaving a null result to sit there.
- **The pocket search gets slower quickly with the size of the protein.** The
  grid grows with the cube of the extent and the flood fill iterates to a fixed
  point: 0.16 s for crambin's 327 atoms, 1.9 s for a 1436-atom protein, and
  **11.0 s for haemoglobin's 4779**. It runs on a worker thread so the window
  stays usable, but a very large receptor will still take a while.
- The docked-pose writer lists the atoms of a `BRANCH` as a chain, as the
  PDBQT format requires, but a forked group — a carboxyl carbon with two
  oxygens — is not a chain. Such groups need nested `BRANCH` records. Until
  then a pose file can declare a bond that the geometry contradicts; the viewer
  detects and reports it, but the file itself is not strictly conformant.
- Secondary structure is estimated from backbone φ and ψ against the
  Ramachandran basins. It is geometric, not DSSP, and it under-detects short
  helices.

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
