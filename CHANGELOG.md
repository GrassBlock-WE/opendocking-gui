# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The project is pre-1.0. The `0.x` line is where the interfaces still move.

---

## [Unreleased]

### Added

- **`scripts/scoring_cross_check.py` cross-checks the engine's scoring against
  the project's own written specification.** It is deliberately **not** a
  cross-validation against Vina, and the file says why at the top: the engine is
  a documented Vina *derivative* with nine intentional deviations — a hard `d²`
  repulsion wall from AutoDock 4, a locally-invented C¹ smoothstep for the `hb`
  and `hyd` windows chosen specifically to differ from Vina's hard cutoffs, and a
  `g2` centred at 0.0 where Vina's sits at 3.0. Comparing absolute kcal/mol with
  Vina would measure the difference between two different functions, on purpose.
  - The reference was written from `docs/SCORING.md` **alone**; `scoring.rs` was
    never opened to decide a term. Its weakness is stated in its own docstring:
    "independent" here means independent of the *code*, not of the *project*.
  - **Verified:** apolar heavy-atom pair energies track the written spec to a mean
    **0.0045 kcal/mol** over 35 points, minima agreeing to 0.03 Å, and the five
    weights match the engine's own runtime description. The 0.0207 kcal/mol floor
    is the 0.375 Å grid the engine interpolates on, not model error.
  - **It falsified one of our own published numbers.** `docs/VERIFICATION.md`
    §3.2.1 reports the apolar C···C minimum at 3.00 Å / −0.050. The engine gives
    **+0.514 at 3.00 Å** — inside the repulsive wall — with its minimum at
    **4.30 Å / −0.0692**. Under our own weights, repulsion alone is +0.538 at that
    distance while the three attractive terms together cannot exceed 0.076, so
    −0.050 is arithmetically out of reach. The table has been corrected and the
    discrepancy recorded rather than smoothed over.
  - **Two findings are open, not closed.** A C···H pair crosses zero at 3.74 Å,
    which implies a hydrogen interaction radius near 1.84 Å where §3.2.1 states
    H is 0. And a polar donor–acceptor pair at 2.8 Å scores **exactly zero** on
    the engine where the specification gives −0.335. Both need someone with the
    Rust open, and the report says so instead of asserting agreement.

- **The benchmarks now say what their numbers mean, and two of them were not saying
  anything true.** Neither is a CI gate and neither will be — a corpus result has
  no ground truth, and gating one would be asserting the internet. What *is*
  gateable, and is now, is that every number each prints is the number its own line
  says it is (`scripts/benchmark_check.py`, 33 checks, offline in about a second
  with `urlopen` stubbed).
  - **`pocket_benchmark.py` printed "cost per 1000 atoms" while dividing by the
    atom count** — milliseconds *per atom* under a per-1000 heading. A 17.6 s
    search over 4779 atoms read **"4HHB 4 ms"**, and 1CRN rounded to **"0 ms"**.
    The real figure is 3683 ms. The heading and the arithmetic are now produced in
    one place and both halves are checked.
  - **`redock_benchmark.py`'s header advertised an exhaustiveness that no run
    used.** Effort is chosen per box from its volume; today's eight runs used 8, 8,
    16, 16, 32, 64, 8 and 64. The header now states the rule and every run and row
    prints what it used.
  - **The rank column printed 1 when no site held a single ligand atom**, because
    `argmax` of an all-zero vector is 0 — and an all-zero vector is exactly what a
    failed search produces. The legend's dash for "not found at all" was
    unreachable. A `holds` column now sits beside it.
  - **The third column can silently equal the second, and did in three of four
    cases.** Either no second box exists — the top site already holds the whole
    bound pose, so the numbers were copied across and printed as an independent
    result — or the best site is still the top-ranked one, and the "ignore the
    ranking" branch re-docked *the same box with a fixed seed*. That second one is
    the important half: 1HVR's top site holds 43 of 46 ligand atoms, so the third
    column was a bit-identical copy of the second, and it read as independent
    confirmation. **Two runs of the same box with the same seed agreeing is the
    weakest possible evidence, presented as the strongest.** Both are labelled now
    and the redundant run is gone.
  - **A truncated download was reported as a whole structure.** `r.read()` returns
    whatever arrived and `decode("utf-8", "replace")` turns a corrupt byte into
    U+FFFD, so half a file still parses — and the smaller atom count makes it look
    deliberate. Both fetchers now refuse a body with no `END` record.
  - **A caption was printed where a result belonged.** 1L96's note says a
    purpose-built buried cavity "is here to be able to *falsify* the detector, and
    if it is not reported as sealed that is worth knowing and is printed." The
    code printed the caption and **never performed the test**. It is now checked,
    and it reports **NOT HELD** at the default probe: 0 sealed at 1.4 Å, 2 at
    0.8 Å. 1CRN's holds.
  - **1HVR's receptor lost 283 atoms** in preparation (913 → 630) and that warning
    was only on the per-case lines, not in the table — and 3PTB's bound box reports
    **+9000 kcal/mol**, a failed run, as a bare 20.02 Å. Both are in the table now.

- **Receptor preparation now says what it threw away, as data.** `prepare_receptor`
  keeps one component of its input and discards the rest. That is usually right, but
  the only record was a `RuntimeWarning` — and a warning is a string that GUI code
  and CLI code routinely drop on the floor. On 1HVR, a homodimer, the redocking
  benchmark searched a 1826-atom structure while docking against a 621-atom single
  chain, and **every RMSD it produced was meaningless**. `prepare_receptor_with_report`
  now returns a `ReceptorPrepReport`; `prepare_receptor() -> str` is unchanged, and the
  warning is emitted *from* the report so the two cannot drift apart.
  - **Chains and RDKit components are counted separately**, because they are not the
    same thing: RDKit cuts components by connectivity, so a homodimer with no
    inter-chain disulphide is two components. Reporting "1 fragment" while two chains
    went in would be true and useless.
  - `odcli prep-receptor` prints **one line** when something was dropped, the full
    table under `--report`, and **nothing at all when nothing was lost**. A
    preparation step that always speaks is one people learn to skip.

- **Ligand preparation had no check script at all.** `prepare_receptor` had one;
  `prepare_ligand` had nothing, and it is the path a user's first run goes through —
  you dock something before you dock a receptor. `scripts/ligand_check.py` closes
  that, and an `ast`+`trace` sweep of `prep.py` went from **40 never-executed
  statements to 17**. The rest are listed in the module docstring rather than papered
  over; the four that need RDKit to throw are **not** covered by making RDKit throw.

- **A pose is now measured against the ligand, not just against other pockets.**
  The pocket search ranks sites by the space a ligand *leaves*, so crambin's own
  ligand-holding site lands **8th of 12** and no amount of re-ranking fixes it: the
  ranking cannot see the ligand. `Pocket.fit_to()` reports both directions — the
  fraction of the **ligand** that lands in free space, and the fraction of the
  **site** it would fill — and the two being different is the point, because a ligand
  too big for a tight site and a ligand dwarfed by a cavern both score 1.0 in one
  direction and 0.1 in the other. On crambin the real site reads **0.500 / 0.833 and
  all eleven others read exactly 0.000 / 0.000**.
  - This is **geometric compatibility, not affinity and not a score**, it is not
    folded into the existing ranking, and the docstring says all three. Calling it a
    score is how a geometric measure becomes a binding-affinity claim by accident.

- **The search box now comes with a bill.** `Pocket.box_with_budget()` returns the
  box *and* the receipt: whether the engine's 30 Å side limit capped it, how much of
  the site still fits, and one sentence saying so. A winding cleft's box is set by
  `site size + 2 × padding` and the engine has a different idea of how big a box
  should be; the difference used to be silent. `BudgetedBox` deliberately **cannot be
  unpacked** — dropping the note is how this defect happened in the first place.
  - Crambin's twelve boxes are **bit-for-bit unchanged**; this only says what was
    already true.

- **Secondary structure is assigned from backbone hydrogen bonds.** The pure
  geometric estimate is still there and still selectable, but it missed whole 3-10
  helices. The new pass is DSSP-shaped: Kabsch–Sander energy, a strict and a relaxed
  pass, turn sizes, a minimum sheet length, and a CA–CA skip bound taken from the
  triangle inequality rather than from taste. Crambin now agrees with its own
  annotation on **39 of 46** residues, up from 33.

- **The other seven `odcli` subcommands had never been run.** Eight exist; one had
  been checked. `scripts/cli_check.py` runs the rest as real subprocesses and pins
  three things per command: the exit code (read from the code, not guessed — two
  are unintuitive, `dock` returns **2** for an empty ligand directory and `workbench`
  returns **3** for a missing GUI stack), the stdout, and what a failure says. All
  eleven failure paths assert the diagnostic is present *and* that no Python
  traceback is.
  - **Each command's two output modes are checked against each other.** The prose
    `dock` table has to agree with `dock --json` to the last decimal, and the prose
    `sites` list with `sites --json`. A machine-readable mode that quietly stopped
    matching its prose counterpart is the failure nobody sees.
  - A **coverage guard** compares the subcommand table from `odcli --help` against
    the set the script actually ran, demanding **set equality** — so adding a
    command without a section fails the build, and removing one without deleting
    its section fails it too.

- **`sites` now reports a volume, and `dock` names its unit.** `sites` printed three
  dimensions and a grid-point count in both modes and never a volume, so a caller
  wanting Å³ had to multiply for themselves — while `Pocket` has carried a `volume`
  field and `box_with_budget()` has returned a budget, neither of which the command
  line could see. `dock` printed an `affinity` column whose unit was documented only
  on `DockingResult.energies`.

- **`rec-grid --scoring vinardo` never said which scoring function it used.** The
  `scoring:` line is printed only for vina. A vinardo run writes 40 `.map` files
  whose names are identical to a vina run's — 5 differ in content, the other 35 are
  byte-identical because those grid points are zero under both — so **the output
  directory alone cannot tell you which scoring function produced it.**

- **`odcli workbench` reported its error on stdout.** 107 bytes on stdout, **0 on
  stderr**, exit 3. Every other `odcli` failure and `odgui` itself write to stderr,
  so a script capturing only stderr saw nothing at all. `cli.py`'s `except
  ImportError` around the workbench import is also unreachable — the package defers
  its GUI import to call time — so the detailed `({exc})` message never ran and
  `odcli` and `odgui` gave two different sentences for the same problem. Both entry
  points now share one preflight.

- **`split` exited 1 with no models but said so only on stdout**, stderr empty —
  unique among the `odcli` failures, and invisible to anything capturing stderr.

- **The five diagnostics in `examples/` are now a gate.** `audit_poses.py`,
  `check_receptor_donors.py`, `determinism_check.py`, `diagnose_clash.py` and
  `robustness_check.py` were in no CI job, in no reproduce list, and covered by no
  other check — which makes them the only diagnostics this project has, and a
  diagnostic nobody runs rots against a system that has since changed.
  `scripts/examples_check.py` runs all five as subprocesses in under ten seconds
  and judges **what each one printed**, not that it exited zero: a script whose
  checks quietly stopped running still exits zero.
  - `determinism_check.py` claimed its grid precalculation was "independent of
    the thread count" while comparing the same call with itself. The engine picks
    its own thread count and exposes no knob, so there was nothing to vary and
    nothing in the repository tested the claim. The heading now says
    "reproducible", which is what two identical calls agreeing byte-for-byte
    actually establishes, and the gate fails if the overstatement returns.
  - `diagnose_clash.py` is labelled rather than deleted. Its 0.22 Å minimum
    separation was measured when every heavy atom had a 0.4 Å interaction radius;
    the worst clashing placement today is **1.92 Å**, 8.7 times further out. The
    script now reports the best clash-free placement and the best placement
    overall as the *same* placement at E = −0.815 — the "after" column of the
    investigation it was written for, where the broken engine preferred a
    clashing placement by 0.99 kcal/mol. It is a regression guard for that fix,
    and only became one once something ran it.
  - `robustness_check.py` now says which of its cases were **accepted rather than
    rejected**: two of its three process-death cases pass because the child exits
    zero, since the engine accepts a one-atom molecule and a self-intersecting
    torsion set. That satisfies the requirement as written, but a column of `ok`
    reads as though a catchable error was raised.

- **`odgui --check` can now say no.** It used to print `GUI stack OK` and exit 0
  on a machine where the viewer could not possibly start, and it is the command
  you run precisely to find that out. Two causes, both fixed:
  `_preflight()` tested `from . import launch`, which needs no Qt, so its
  `except ImportError` branch was unreachable and it always reported success; and
  `--check` had no second stage at all, so an OpenGL context that never arrives
  was never looked for. It now calls `_require_gui()` — the same call `launch()`
  makes, in the same process, under the same conditions — and adds a second stage
  that asks Qt for the context the viewport needs, using the viewport's own
  surface format.
  - **It distinguishes the two ways a viewer can be unlaunchable, and gives each
    its own sentence.** "PyQt6 and moderngl are installed but Qt could not
    create an OpenGL context" is not answered by `pip install`, and telling
    someone to install a package they already have is how a two-minute question
    turns into a bug report. The message names what was established, what was
    not, and the two things worth trying in order — and says that the engine and
    `odcli` do not need OpenGL and are unaffected.
  - **Exit codes are 0, 3 and 4, written in three places** — the module constants,
    the function, and the flag's own help. 3 is deliberately the code `odgui` and
    `odcli workbench` already return for a missing stack, so `--check` is a
    faithful preflight: a script that sees 3 knows exactly what launching will
    do. 4 is separate because it is a different problem with a different fix. A
    diagnostic that always exits 0 cannot be scripted against.
  - `--check` now runs **before** the file-argument checks. `odgui --check -r
    typo.pdbqt` used to exit 2 complaining about the typo and never mention the
    viewer, which is the opposite of what was asked.
  - `moderngl.create_standalone_context` is **not** used as a health probe, and
    the reason is recorded next to the probe that replaced it: it fails on
    machines where the viewer works perfectly well, because the viewer wraps Qt's
    context rather than creating a second one. It answers the wrong question.

- **`odgui --check` no longer dies on the machines it exists for.** Asking Qt for an
  OpenGL context is not a Python operation that can fail: on a machine whose
  driver cannot supply one, `show()` blocks and then Windows ends the process
  with `0xC0000409` — a `__fastfail`, not an exception, so no `try` in the
  process can see it. Measured here with no change to the file: the same command
  returned exit 4 with a full report, and then died natively three times out of
  three. **A diagnostic that kills the process is useless on exactly the machines
  that need it** — a remote desktop, a VM without 3-D acceleration, a virtual
  display adapter. The probe now runs in a child process, so a dead child is a
  *result*: "no context", exit 4, in the parent's own wording.
  - The child prints what it established **before** it attempts the context, and
    the parent merges every line it can parse. A report of "?" for every field is
    a worse answer than the same report carrying the facts it did establish.
  - When the probe ends inside `show()`, the message says so — *"ending inside
    show(), the probe process did not come back"* — rather than quoting a default
    0.0 s in the shape of a measurement.
- **A context is not a working viewport, and the probe now says so.** The renderer
  calls GL through the function table, so `initializeOpenGLFunctions` returning
  False means the table is absent even where a context exists. This also makes
  the probe *testable*: the probe widget is a subclass that defines
  `initializeGL`, so a stand-in that makes it a no-op cannot be detected by "was
  it called" — it is always called. Requiring **both** the context and the
  function table is the honest definition of ready, and the only thing a stand-in
  can actually block.
- **`odgui` and `odcli workbench` now say the same thing, byte for byte.** The
  better-worded missing-dependency message in `workbench.launcher` was guarded by
  an `except ImportError` that could never run, so nobody had ever seen it and
  nobody had noticed that the two entry points told a user two different
  sentences for the same problem. The wording now belongs to one place and all
  three paths — `odgui`, `odgui --check` and `odcli workbench` — produce identical
  output on the same stream.

- **The cell index in the secondary-structure pass now has its coverage claim
  checked instead of asserted.** `HBOND_CELL` only has to be at least
  `HBOND_CA_SKIP`, and the relation is geometric: a donor more than `k` cells out is
  more than `k * edge` from the acceptor whatever the acceptor sits at inside its
  own cell, so a search `k` cells wide covers an `R` cutoff exactly when
  `k * edge >= R`. Measuring from the cell **corner** instead gives
  `(k + 1) * edge` and is a whole cell optimistic — which is how a 4.5 Å edge with a
  one-cell stencil looks adequate and misses candidates.
  The check measures the stencil in cells from the running code and fails with the
  shortfall in Å. The measured stencil is **1 cell on all six sides: 9.0 Å against
  a cutoff of 8.2 Å**, 0.8 Å of margin. It deliberately does **not** assert
  `HBOND_CELL == 3 * HBOND_CA_SKIP`, because a 4.1 Å edge with a two-cell stencil
  measures exactly 8.2 Å and that form would kill a correct configuration.

- **Search effort now follows the search box.** Exhaustiveness is a count of
  Monte Carlo walks, and walks are spread through the box they search, so a
  fixed 8 that comfortably covers a 20 Å box is spread four times thinner in a
  40 Å one. The pocket search will happily hand you a 39 × 26 × 41 Å box for a
  winding cleft, and a search that under-samples returns a bad *pose* — which
  looks exactly like the box being wrong. Measured on that box: **12.88 Å RMSD
  at exhaustiveness 16, 1.26 Å at 64**, two seconds either way. Same box, same
  receptor, same ligand, same seed.
  - The workbench's exhaustiveness box follows the box size, and **stops
    following the moment you set it yourself** — a control that overwrites what
    you typed is worse than one that never helped, because you learn to distrust
    the one number whose effect you cannot see until after the run.
  - `odcli` derives it when you do not pass `-e`, and **prints the value and the
    reason**. It used to be a constant in the help text, and a number that moves
    with the box has to be said out loud or you will assume 8 and wonder why two
    runs of "the same" search disagree.
  - The rule is `opendocking.core.exhaustiveness_for_box`, shared by all three
    callers. The benchmark already had a copy and **it had drifted**: its ladder
    started at 16 where the engine's starts at 8. One rule in one place, or
    three rules that disagree.
  - Scaling is **linear in the box's volume**, deliberately. One calibration
    point says a 5.2× bigger box needs at least 8× the walks; a power law could
    be fitted to match that exactly, and fitting a law to a single measurement
    is a way of pretending to know something. Linear is a whole number, is on
    the conservative side of the number we have, and "8 walks per 8000 Å³" fits
    in your head.
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

- **`GridMaps.box` raised `AttributeError` on every call.** The helper it used was
  named with two leading underscores, so calling it from a second class mangled
  it to a name that did not exist. A public property that had never once returned,
  and 38 pytest cases missed it: one meta-test checks that wrapper attributes are
  names the Rust extension exports, the other uses `hasattr` and only covers
  `DockingResult`. The defect lived in the seam between two Python classes, where
  neither can see.
  - `scripts/core_check.py` now walks **every public member of every wrapper on a
    real instance and reverse-asserts that none was left unchecked** — 392 checks
    in 16 sections, ten of the new guards mutation-proven.
- **The same mistake thrown two different ways.** `dock(seed=-1)`, `dock(steps=-1)`
  and `pose_coords(-1)` raised `OverflowError` while the guards beside them raised
  `ValueError`, so a caller guarding the documented parameters with
  `except ValueError` did not catch them. All are `ValueError` now, and the message
  names the parameter and its legal range.
- **`Receptor.estimate_memory_mb` answered for calls `precalculate` refuses.** A
  negative, NaN or infinite `spacing` was quietly replaced by the default —
  9.05 MB for the first, 0.0012 MB for the last — while `precalculate` raised on
  all three. The estimate and the thing it estimates no longer disagree about what
  is legal.
- **A ligand built from NaN coordinates reported a confident `radius` of 0.0.**
  `from_arrays` checked shapes and lengths and nothing else, so `reference_coords`
  came out all-NaN and the radius came out a plausible-looking zero. Non-finite
  charges and out-of-range bond indices are refused too, and the message names
  the offending atom.
- **`exhaustiveness_for_box` answered 128 — its most expensive answer — for a NaN
  volume**, because `nan >= rung` is false for every rung and it fell through the
  loop. It also accepted a 2- or 4-value size without complaint. Both are refused.
- **A hydrogen bond was being reported twice.** One 1.8 Å bond came back as the
  bond *and* again as a `polar` close contact between the same two heavy atoms
  1.0 Å further out — and `residue_summary` counted it **twice** in the number a
  user reads. The de-duplication key recorded the **hydrogen's** index while the
  close-contact search skips hydrogens, so it never matched: the suppression was
  inert in both directions.
  - **No threshold check could ever have caught this.** Each of the two rows is
    individually true. The duplication lived in the relationship between them, and
    a threshold check only ever looks at one value at a time. It took asking what
    the rows had to do *with each other*.
  - The code was changed to match the docstring, not the other way round: the
    docstring is the contract the user reads.
- **A ribbon whose side vector happened to be parallel to the tangent drew
  nothing at all** — 0 of 160 triangles with any area, 44 of 84 vertices with a
  zero normal, x and y span exactly 0.0 — and the caller believed it had
  succeeded. A viewer reads that as "no secondary structure here" rather than as a
  bug.
- **The ribbon check was passing the wrong ribbons.** "Every ribbon point is
  within 4.0 Å of a CA atom" was satisfied by a ribbon **twice as wide** (3.008 Å)
  and by a ribbon with **no width at all** (0.350 Å) — and the degenerate one
  passed it *more easily* than a correct ribbon. A constraint that the degenerate
  case satisfies more readily than the correct one is not a weak constraint, it is
  an inverted one, and tightening the number would never have found it. The
  cross-section extent is now measured directly.
- **The two-tone bond's colour was asserted by vertex order.** Moving the split
  point by one bond length still yields two half-cylinders of the right total
  length in the right colours, attached to the wrong atom at one end. The
  assertion now names each endpoint's colour, because which vertex comes first is
  an implementation detail.

- **The dihedral function returned `180° − torsion`, which is its own mirror.** So
  every Ramachandran comparison in the workbench was being made against inverted
  angles. Crambin residue 8, in the first helix of a structure whose coordinates put
  its O(i)···N(i+4) at 2.89 Å, was reported as φ = −124°, ψ = −135° — the beta
  region, not the alpha one — so every residue of both of crambin's helices fell
  outside the ranges and the geometric estimate called almost the whole protein coil.
  - It survived because the first "reference" implementation used for the check had
    the mirror **on the other side**: two bugs cancelled, and a symmetric test
    geometry — four coplanar points, torsion 0, answer 180 — is what finally broke
    the tie. **A control that is symmetric to the bug is not a control.**

- **The water and ion filter had never removed a single atom.** It called
  `RemoveAtom` on a `Chem.Mol`, and `Mol` has no such method — only `RWMol` does. It
  never crashed only because water and single-atom ions are always their own
  component and were gone before the filter ran. So `--drop-heterogens` and
  `keep_waters=False` were not merely untested: the code behind them was **dead and
  broken at the same time**, and no input on earth could have told you.

- **`Chem.PathToSubmol(mol, atoms)` does not take a set of atoms.** It takes a
  traversal path and returns whatever that path reaches. Measured on one molecule:
  asking for 3 atoms returns **4**; a single-atom molecule asked for its one atom
  returns **0**, so a receptor file containing nothing but waters or ions prepared
  to nothing at all. The under-returning direction is the worse one, and it is the
  one a "count the atoms" assertion would have missed.

- **A failed MMFF relaxation was swallowed silently.** An acyclic ligand that
  arrived without coordinates has no ring information, so `MMFFOptimizeMolecule`
  raises and a bare `except: pass` eats it. The user was told only that the pose was
  a guess, and nobody was told the pose had not been relaxed at all. It is covered
  now, and `prep.py`'s module docstring says which of the four guarded RDKit calls
  have actually run.

- **If a future RDKit stopped parsing the donor SMARTS, polar hydrogens would have
  stopped being added and nothing would have said so.** That branch used to return
  the molecule unchanged. It now raises, and the message says what the consequence
  is. It is a behaviour change on an unreachable path: no change today, loud instead
  of silent if it ever happens.

- **The workbench said "128 walks" as though 128 were enough.** The rule saturates
  by design: at its own density a 60 Å cube wants 216 walks and gets 128, and a
  100 Å cube wants 1000 and gets 128 — **13%**, in exactly the size range where bad
  poses are likeliest. Measured on a 60 Å box: 0.62 s at exhaustiveness 8, 5.70 s at
  128, for a 0.09 kcal/mol difference. The workbench now shows the map size and says
  **"the ladder stops here: 13% of the 1000 walks a 1000000 A^3 box wants"**. A
  capped number presented as a met one is a lie with a checkbox next to it.

- **One stray click on the spin box cost you the suggestion for the whole session.**
  `_exhaust_is_default` was set to `False` and nothing in the repository ever set it
  back. There is now a **use suggested value** button — pressed by you, never
  automatic, because a control that undoes what you typed is worse than one that
  never existed.

- **The exhaustiveness wiring was untested while the rule itself was well tested.**
  `_suggest_exhaustiveness`, `_exhaust_is_default` and `_updating_exhaust` appeared
  in no check script at all. Deleting the guard, making the function return
  immediately, breaking the latch, and hardcoding the value all left the suite at
  **135/135** — a feature that can be perfect or completely dead and looks the same.
  It now goes red on all four.

- **Some checks could not fail, and some failed when the code got better.** Two
  checks carried byte-identical boolean expressions. One asserted that a site is
  *not* rank 1, so four mutations that improved the ranking turned the gate red.
  And 34 checks sat behind an `if`: tightening the volume ceiling took the suite
  from 118 to 84 and it still reported every check as passed. `EXPECTED_CHECKS` now
  pins the total, and the pin explains itself when it fires.

- **A documented pocket extent was wrong, and two numbers in this file said so.**
  The 3PTB cleft was recorded as a **22 × 17 × 28 Å** site next to a **39 × 26 × 41
  Å** box, and those cannot both be true: at the default 4 Å padding a 22 × 17 × 28 Å
  site produces a 30 × 25 × 36 Å box, not 39 × 26 × 41. The box was re-measured and
  reproduces exactly (39 × 26 × 41 Å, rank 1 of 39 sites, 9/9 bound-ligand atoms in
  it, 12.88 Å RMSD at exhaustiveness 16 against 1.26 Å at 64), and the site extent
  follows from it: **31 × 18 × 33 Å**. Every "22 × 17 × 28" in this project was wrong,
  not merely stale, and is now 31 × 18 × 33 in the READMEs and here.
  - Recorded rather than quietly corrected because the interesting part is *how* two
    contradictory numbers sat in the same document for a release. Nothing recomputes
    one from the other, so the only thing that noticed was a person doing arithmetic
    on a docstring — and the contradiction was noticed only because a worker agent was
    told to reproduce a claim and could not.
- **Crystal waters and buffer additives are no longer reported as the residues
  lining a pocket.** The lining list is meant to answer "what is the wall of
  this site", and a water is a lattice artefact rather than a wall: across
  seven structures, streptavidin had more waters than residues in its top
  site's lining. Cofactors are a different case and are deliberately kept —
  "this pocket is lined by the haem" is real chemistry and worth knowing.
- **Clearing the site selection no longer leaves the previous site's volume
  cloud on screen**, sitting inside whatever the new selection put there.
- **A colour in the picture had no name anywhere on screen.** Loading a receptor
  puts the box on the first pocket and draws that pocket's own grid points as
  magenta spheres straight away, and the legend named only the four dashed
  interaction lines. So a report read those spheres as unexplained balls in a
  box. The legend now carries the site volume too, with a **dot** rather than a
  line — it is a volume, and the shape of the swatch is part of what the legend
  claims. "The `site volume` checkbox is over there" is not the same claim as
  "this colour is the site volume", and that difference is the whole reason a
  legend exists. Four checks now assert that every colour the viewport draws
  that is not an atom colour is named, that the site-volume swatch is the
  colour the cloud is actually drawn in, and that no swatch is left as a bare
  colour with no label.
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

### Changed

- The `dock-core` test fixture for the GPU CPU-fallback path replicated butane
  40 times 0.1 Å apart, which placed atoms on identical coordinates. It now
  uses a 4×4×3 lattice at 3.0 Å pitch, so every atom pair is physically
  possible.

### Performance

- **Secondary-structure assignment no longer scans every residue pair.** Backbone
  hydrogen-bond candidates come from a uniform cell list over donor CA atoms instead
  of a full double loop, with the 8.12 Å CA–CA cutoff still re-measured exactly for
  every candidate. On a 5200-residue structure the assignment goes from **88.95 s to
  0.98 s**, with **91× fewer distance measurements** (26,899,626 → 295,978) for the
  same 58,876 hydrogen bonds, at about 800 bytes per residue of index.
  - **The assignment is unchanged**, residue for residue, and that is the only claim
    worth making. The check computes it twice — once through the real path, once
    through a deliberately unindexed walk that uses its own distance function — and
    compares the two sequences state by state. It also asserts the reference really
    was unindexed, by counting distance measurements: counting returned bonds would
    be circular, since returning the same bonds *is* the equivalence.
  - Below about a thousand residues the two are within measurement noise. This is a
    large-structure fix, and it is reported as one.
- **The assignment no longer enumerates the candidate list twice.** The strict and
  the loose pass each built their own, with identical arguments, and
  `HBOND_CA_SKIP` does not depend on `relaxed` — so the second build rebuilt the
  list the first one had just built. `relaxed` is read in exactly one place in the
  module, the line that picks the four thresholds, and the candidate test cannot
  reach it. The list is now built once and handed to both, and it is a **required**
  argument rather than an optional one that builds its own, so a caller cannot
  quietly get one build per pass back.
  On a 5200-residue packed bundle: 295,978 distance measurements per pass instead
  of 591,956, and the counted work of a whole assignment falls from **827,598 to
  531,619**. The enumeration's share drops from 71.5% to 55.7%; the hydrogen-bond
  gates rise to 39.5% and do not overtake it. Per-residue counts stay flat across
  1200/3000/5200 — which is what a change in *how many times it runs* should look
  like, as opposed to a change in what each pass does.
- **The index now has a check on which atom keys it, and a second pair of eyes on
  the bond set itself.** A grid keyed on the nitrogen instead of the CA was caught
  by **nothing**. It had a second face that the obvious framing misses: `donor_ca`
  and the cell key are the same variable, so keying on N measures the cutoff
  *between nitrogens* and the index **invents bonds** rather than merely losing
  them. Assignment-level equivalence was not sensitive enough either — it compares
  residues, and losing a candidate pair changes no residue when that pair would not
  have bonded anyway. The bond sets are now compared directly: both passes over
  seven structures, 3621 bonds, identical in every case.
- **The pass is checked by counting its work rather than by a wall clock.** The old
  assertion was "a 1200-residue packed structure is assigned in under 5 s". Collapse
  the grid to a single cell and it returns **exactly the same pairs**, costs 1236
  counted operations per residue instead of 95, and finishes in **1.44 s** — well
  inside the five seconds the threshold allowed. The guard was not weak; it was
  pointed at the wrong thing. The count now covers every phase the pass has, not
  just the index. The clock is still measured and printed, as information.

### Documentation

- `exhaustiveness_for_box` now says that its linear rule **saturates at 128**, so a
  200 Å box is answered up to **62× below** what the rule advertises. The docstring
  said "linear in volume" and did not mention the cap.
- `auto_box` now says that its `ligand` argument is accepted and never read. It is
  documented rather than removed, because removing a public parameter is a
  breaking change and the honest documentation is the smaller one.
- `MoleculeView.residue_labels` no longer uses `HEM 155A` as an example of the
  noise it removes. The code **deliberately keeps cofactors** — "this pocket is
  lined by the haem" is real chemistry and worth knowing — and `pockets_check.py`
  asserts that on purpose, with a note that the first version of that check got it
  wrong and that is how the distinction got made. The docstring was the leftover
  of a decision that was already taken and settled; the code is right.

### Known limitations

- **A column shifted the wrong way cannot be caught by any file reader.** A field
  that arrives one column late is a perfectly well-formed, right-justified float
  with the wrong value in it — `meeko` accepts it too. Only comparing against the
  molecule the file was written from would catch it. This is pinned as a check so
  that it is not overclaimed.
- **`read_pdbqt_models` refuses a mis-columned file by default, and that is a
  decision, not an oversight.** The engine's `Ligand.from_pdbqt` splits on
  whitespace, so it cannot see a column error at all; that tolerance is
  deliberate, because a hand-edited or third-party file should load. Strictness
  therefore lives at the boundary this project owns and where files enter the
  app. Before refusing anything, the validator was measured against all 786 ATOM
  lines in the shipped examples — including the 144 the engine itself wrote — and
  produced **zero false positives**.
  - `float()` alone is not a validator: a coordinate arriving one column early
    parses as a valid number, so field *shape* is checked now, not just
    conversion.
  - `strict=False` opts out, for callers who want the engine's old tolerance.

- **The ligand charge table is not neutral, and this is measured on every run rather
  than fixed.** Merging a non-polar hydrogen does not fold that hydrogen's partial
  charge into its parent, which is what AutoDock's convention asks for. Benzene's six
  carbons each carry −0.0623 e, so the table sums to **−0.37 e** for a neutral
  molecule, and ibuprofen sums to **−0.60 e**.
  - **Left as it is deliberately.** Folding the charge in would change every score
    the engine has ever produced, which is a decision for a human and not a side
    effect of writing a test. The check now prints the number, and asserts neither
    that it is a defect nor that it is correct.
  - `keep_hydrogens=True` also puts **17 hydrogens with no bond** into ibuprofen's
    33-atom table, because only N/O/S hydrogens keep theirs. The comment that said
  otherwise was true only of the default path.

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
  a 31 × 18 × 33 Å cleft and its box is 39 × 26 × 41 Å. There is no single box
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
  occupies gives −5.36 kcal/mol against the reference's −5.50, at 4.24 Å RMSD.
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
