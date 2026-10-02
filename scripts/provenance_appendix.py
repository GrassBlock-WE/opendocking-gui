"""Do the check counts the prose quotes still hold, and can every number that
cannot be derived be traced to the instrument that produced it?

**That second clause is the goal. The credit Tier 3b actually issues is a
citation, and the difference is the whole subject of this round.** Every residue
literal is put to one question -- does the sentence's unit name something that
*exists*: a file in this tree, a gate carrying a live pin, a program this
machine's `PATH` resolves -- and the class that answers yes is
**`names-existing`**. It is named for that question and not for a production.
Running the artefact is not attempted anywhere below and could not be done from
a tree, so a green here is never evidence that a script emitted a number: it is
evidence that the sentence points at something a reader can go and run. Four
figures in this repository's own prose sat under this class while it was named
for the artefact rather than for the question, and every one of them was green
for exactly that reason. The four ways, with the code each one rests on, are
worked through in the Tier 3b section below.

Run:  python scripts/provenance_appendix.py
      python scripts/provenance_appendix.py --root <tree>    # a copy, for mutations
      python scripts/provenance_appendix.py --emit           # print the appendix block

# What this is, and what it is not

This is **Tier 2 and Tier 3** of a three-tier provenance scheme. Tier 1 -- a number
that is a literal in the source, so the marker for it is a *symbol citation* rather
than a state word -- is `docs_claims_check.py`'s existing mechanism and is
deliberately **not** reimplemented here. That file owns the census for it, and a
second implementation would be a second thing to keep in step with the first.

The decision this file acts on, and it is a decision rather than a default: **no
per-number state word is written into the prose.** A `[measured]` marker in a
sentence asserts a state that no check can hold true, because a check can only
compare two things and a marker is a third thing that is merely believed. Such a
marker decays into decoration, and a reader who has watched it decay once stops
reading it. Everything below is instead a *difference of two sets the machine
computes*, so the absence of a human-maintained list is a property of the design
rather than a promise about discipline.

# Tier 2: a check count may only be quoted with the gate that owns it

Every check-count figure in the prose of `README.md`, `README.en.md` and
`CHANGELOG.md` is resolved to the gate its sentence names, that gate's live
`EXPECTED_CHECKS` is read **out of the gate file** with `ast` -- never out of a
table in this file -- and the two are compared. Four things follow, and none of
them is a list somebody typed:

* **A count quoted with no resolvable gate name is a finding.** Not a warning and
  not a note. It is the exact defect that let the numbers rot, because a number
  that names nothing has nothing that can contradict it. The verdict for it is
  `unrefutable`, and the check goes red.
* **A count whose gate carries no pin is a finding**, and it is a *stronger* one
  than a wrong count: the number may be right, and nothing in the tree would ever
  be able to say so. The gates that carry neither `EXPECTED_CHECKS` nor
  `NO_EXPECTED_CHECKS` have a check of their own, and that check describes a
  **count** rather than a picture of one: above zero it names every such gate and
  the two module-level declarations that close one, and at zero it says the
  condition is absent and that the walk which found it is the walk that read the
  pins. No total is written in this sentence, because a total typed beside a
  measured count is a second number to keep in step with the first, and a number
  nothing can contradict is the defect this file exists to find -- including when
  the number is written by this file.
* **A stale count and a dated reading are different things, and this file can tell
  them apart without a human saying which is which.** Scope comes from the
  nearest preceding `##` heading: `## [Unreleased]` and the two READMEs (which
  carry no release heading at all) are *live* and must match the tree; a dated
  `## [x.y.z]` heading is a *released* record and is reported, not failed.
  Within one live scope, the **last** total-asserting figure quoted for a gate is
  that gate's current claim and is compared to the pin; every earlier figure for
  the same gate is a **superseded reading** and is reported with the movement,
  never failed. That ordering is positional, so `CHANGELOG.md`'s own
  "61 passed, 0 failed, 14 skipped, 75 checks" is classified as a dated reading
  *because* a later sentence in the same scope quotes the gate's present total --
  and the italic annotation a human wrote beside it, saying exactly that, is
  corroboration rather than the mechanism. Delete the annotation and the
  classification does not move, which is the test of whether the mechanism or the
  prose is doing the work.
* **A number introduced by the literal token `EXPECTED_CHECKS` is the most
  total-asserting shape there is**, because it names the pin itself, so it is
  admitted as a figure in its own right. This is what makes the rule above work on
  the one sentence that matters: `...`s `EXPECTED_CHECKS` is now 338` is what
  supersedes the `75 checks` on the line above it.

A figure is admitted when its **sentence** carries a count word and its shape
asserts a total. The unit is the sentence and not the paragraph because the
paragraph is too coarse in both directions: `volume ** (1/3) * burial` and
`9/9 bound-ligand atoms` both sit in paragraphs that discuss checks elsewhere,
and admitting them would have this file comparing an exponent and a burial
fraction against a check total. Sentence scope drops both, and the residue census
below picks them up as Tier 3, which is where they belong. Gate resolution is
**nearest by line distance within the paragraph**, because markdown wraps a
sentence across lines and `509 checks` names its gate two lines above it.

# Tier 3: the residue, and the column that is derived rather than typed

What is left -- every numeric literal in the prose that is not adjacent to a
`file::symbol` citation (Tier 1) and not part of a figure above (Tier 2) -- is
RMSDs, energies, timings, box extents and pixel areas. None of it is derivable.
For each such literal the residue census emits a row: **value, date, instrument,
tree state, command**, and enforces it from both sides, because a one-sided rule
cannot see its own absence:

* a residue literal with **no row** is red;
* a row with **no date or no instrument** is red;
* a row whose **witness is older than its date** is red.

**The tree-state column is derived, never typed, and for the same reason the
`[measured]` marker was rejected.** A row is `current` only if the artefact it
names is newer than the row's date, and `stale` otherwise. So the state is an
output of a comparison at run time and there is nowhere for a human to write it.

**Where that witness over-claims, under-claims, and is simply absent.** mtime is
a weak instrument and this file is explicit about the three ways it fails rather
than presenting a column of confident `current`s:

* **It over-claims freshness.** A build artefact under `target/`, `dist/`,
  `_wheelout/` or `__pycache__/`, or any binary suffix, is rewritten by a rebuild
  that changed no source. The witness then advances and a number that describes
  the old source is reported `current`. These rows are classed `over-claims` and
  counted, and a reader can see the class rather than the verdict.
* **It under-claims freshness.** A single file's mtime is the right witness only
  if that file is the measurement's only real dependency. For a measurement whose
  real input is a compiled extension, the witness is a build product of several
  sources and **no single mtime is the right witness at all**. Rather than guess,
  the witness's import closure is computed from the instrument's own syntax tree
  and the row reports `1 of N`: one file's timestamp is being asked to speak for
  N. Those rows are classed `under-claims`.
* **It is absent.** A row may declare `none` as its instrument, and that is a
  legitimate and publishable answer -- for a number that depends on a machine, a
  GPU, a network or a wall clock, **no witness exists for that class** and the
  honest thing is to say so rather than to invent a file. Those rows are classed
  `no-witness` and counted. A row naming a file that is *not there*, though, is
  red: that is a claim about an instrument, not an absence of one.

# Tier 3b: the measurement case, which is row-free because a row is the wrong shape

Everything above is the **record** half of Tier 3, and it is correct and
**unexercised**: no prose file carries an appendix block, so the six row checks
per file are six counted skips and the block half of this gate is green only in
the sense that it has measured one easy case. That is stated in the run's own
summary line, and the transition -- the moment someone pastes a first block --
remains the deliverable. Nothing below changes that, and nothing below *requires*
a block.

The reason the measurement case cannot be answered by a row is worth putting
plainly, because it was the reason this file nearly got it wrong twice. A row
answers "who produced this, and when". A **measurement** -- an RMSD, an energy, a
timing, a pixel area -- is not produced by a *file that contains it*. It is
produced by **running an instrument**, and so its provenance is a **run**, which
is historical by nature and cannot be re-derived from the tree as it stands. A
row keyed on `(value, line)` therefore records a run nobody can re-run, and a
date nobody typed is not a run. What *is* machine-derivable is the claim the
sentence itself makes, and the claim is small and checkable:

> **Does the sentence name an instrument that exists?**

A file in this tree, a gate that carries a pin, a named command. Each residue
literal is put to that question, and the answer is one of eight classes, all
**derived at run time**:

* **`names-existing`** -- the sentence **named** something and it is **there**:
  a file in this tree, a gate with a live pin, or a program on this machine's
  `PATH`, and named as a path or a program rather than by stem. Green, and the
  artefact is printed so a reader can go and look at it. No row, no date, no
  column, nothing to maintain, and nothing that can go stale between runs.
  **It credits a citation and nothing beyond it** -- that the sentence points
  at a real artefact, not that the artefact produced this figure. The four ways
  that distinction is lost, measured on this tree's own prose, are below.
* **`names-unpinned-gate`** -- it names a gate that exists and pins nothing. An
  RMSD from an instrument nobody can hold to account for is weaker than one from
  a pinned gate, so this is its **own class and its own check**, visible rather
  than collapsed into the green above. It is red, and the reading behind that is
  stated where the check is: the defect is not in the prose, it is the same
  unpinned gate Tier 2 already reports, reached from a second direction.
* **`names-document`** -- it names a **markdown file**. Red, its own class, its
  own check, and the reason is the one this file has refused everywhere else: a
  document is prose, so a sentence naming one is **prose agreeing with prose**.
  A document contains the number *because a human typed it there*, which makes
  "the document says 12.88" and "12.88 is right" a single proposition. It is not
  `names-nothing` either -- the sentence does name something real, and burying
  six findings in a 588-strong class would make them unactionable.
* **`stem-match`** -- the name the sentence wrote is **not a path in this tree**,
  and the resolver matched it against a stem index. Red, its own class, its own
  check, and this is the false-green generator: a sentence naming `volume` went
  green because *some* file in the tree is called `volume.py`. Where two or more
  files share the stem, which one was meant is decided by `os.walk` order.
* **`names-absent`** -- it names something that is **not** in the tree, or not on
  this machine's `PATH`. That is a claim about an instrument, so it is red and
  the absent name is printed.
* **`names-nothing`** -- the sentence names no instrument at all. Red, named,
  and **this is the case a human must look at.**
* **`unit-too-coarse`** -- the unit is wider than `MAX_UNIT_LINES`, so no
  instrument is credited. Red, and kept separate from `names-nothing` because the
  two are different findings: one sentence names nothing, the other is a bullet
  this splitter swallowed whole.

An eighth bucket, **`no-unit`**, holds a literal whose line produced no unit at
all. It is a defect in *this reader*, it is red, and it exists so that a reader
gap cannot be silently reported as "names nothing".

**THE FOUR WAYS A NAMED ARTEFACT IS THE WRONG ATTRIBUTION. All four were live
in this repository's own prose while this class was named for the artefact it
credits rather than for the question it asks, and all four were green, because
naming an artefact and being credited with producing a figure are two different
propositions and the check only asks the first.** They are written out with the
code each one rests on, because a concrete instance is what stops a reader
re-deriving the mistake, and because the class name is the only thing standing
between this list and the next revision of the same sentences:

* **The figures are in the named file, but only inside a source comment, and
  nothing prints them.** `README.en.md` credits the exhaustiveness ladder
  **12.88 A at 16** and **1.26 A at 64** to `scripts/redock_benchmark.py`. Those
  two numbers are at `scripts/redock_benchmark.py:116` and nowhere else in that
  file -- inside the comment that opens "Measured on the 3PTB site box" -- and
  the same comment says the script **structurally cannot** print the pair,
  because every run takes its effort from its own box's volume, "so that number
  described no run at all" (`:124-129`), over a `CASES` list of four complexes
  (`:101-106`). The file exists, the pair is in it, and running it prints
  neither: a *pair* of exhaustiveness values for one box is not a shape this
  script has anywhere in it.
* **The named script has no such case -- and the sentence denying that still
  credits it.** The crambin reference energy, **-5.50 kcal/mol**, comes from
  `scripts/pockets_check.py`. `scripts/redock_benchmark.py`'s `CASES`
  (`:101-106`) is 1STP / 3PTB / 2NNQ / 1HVR and holds no crambin at all.
  `README.md:353-355` says precisely that, in the same bullet that names the
  script it is ruling out, and the figure was green: the unit named a real
  file, and a sentence that *refutes* its own attribution still names the file.
* **The range is a hand-widened rounding, and the rounding inherited the
  citation.** `README.en.md:533-535` records an earlier **2-6** gradient range
  as "rounded outward from a 2.3-5.9 reading in `docs/VERIFICATION.md`" and
  says it "is not the output of `examples/audit_poses.py`". It is right, and
  `examples/audit_poses.py` bears it out: that script prints one
  `worst gradient norm over the batch` (`:368`) and a per-pose descent ladder
  of eight rungs (`:350-366`) -- a maximum and a ladder, never an interval. An
  interval widened by hand carries the citation of the measurement it was
  widened from, and no run of the named script will ever reproduce the widening.
* **The numbers are typed literals inside the instrument's own printed string.**
  Five figures describing the old `F` framing -- **8.80 A**, **3.8x**,
  **2.05 A**, **32 px** and the **5** they fell to -- are credited to
  `scripts/workbench_interaction_check.py`. They are typed constants in that
  check's own *detail* string (`:11080-11082`: "took the camera 8.80 A off and
  pulled back from 6.00 A to 22.78 A, turning a 2.05 A line from 32 px of
  highlight into 5"). What the check **measures** is the four values
  interpolated beside them -- `{_pair_before:.2f}`, `{_pair_after:.2f}`,
  `{_res_shift:.2f}`, `{_res_dist_shift:.2f}` (`:11077-11084`) -- and what it
  **asserts** is `_pair_before < 1.0 and _pair_after < 1.0 and _res_shift <
  0.5 and _res_dist_shift < 0.5` (`:11074-11076`). The literals are thresholds
  describing history; the measurements are the four numbers printed after them.
  `README.en.md:192` counts them itself -- "does not measure those five
  numbers" -- and the class credited the script anyway. To be exact about which
  number is which: the literals are a **historical record typed into a message**,
  the thresholds are the `1.0` and `0.5` in the assertion, and the measurements
  are the four values printed beside both.

**None of the four is a gap in the resolver**, which is the point worth keeping:
each sentence wrote a full path to a file that is really in this tree, so
`resolve_instrument` returned `exists` and the class could not have said
otherwise. Catching them needs a *run*, and a run is not derivable from a tree.
What the class can do -- and now says -- is decline to imply that one happened.

**The green branch is not flat, and the run says so -- with the question that
decides it printed as a column.** A number attributed to a pinned gate, to a file
the sentence named in full, and to a program on this machine's `PATH` are three
different claims, so `names-existing` is broken down by kind in every run
(`gate`, `file`, `command/in-repo`, `command/outside-repo`), and beside each
count the run prints **what would have to happen in this repository for that
green to go red**. That column is the answer to "is this check decoration", and
it is not always "yes it can": a `command/outside-repo` green is printed as
**NO** -- nothing in the tree can redden it, because the program lives on the
machine's `PATH`, and moving `PATH` reprints the path without changing a verdict.

**What this check is not, stated here so nobody has to discover it.** It
verifies that the sentence *names* a real instrument. It does not verify that the
instrument produced this number, that the number is right, or that the run ever
happened -- the first needs a run log, the second needs a pin, the third needs a
date, and none of the three is derivable from a tree. And **whether a named
command exists is a fact about the machine running this gate**: it is resolved
with `shutil.which`, memoised, and the count of greens that rested on a `PATH`
hit is printed, because that number moves when the machine's `PATH` moves.
**This paragraph is the limit, and the four cases above are the limit being
reached in practice** -- so the class above is named for the question it asks.
The name it used to carry read as a claim about production, which is the one
thing this paragraph has just said the check does not establish.

# The same number written two ways, and the fix is a reservation beside the others

`_DATE`, `_VERSION` and `_DIGEST` exist because a date, a version and a digest are
*shapes that would otherwise be read as several numbers*. A caret power is the
fourth member of that family and it was missing: `2^28` was tokenized as `2` and
`28`, while the same value written `107,374,182` was one literal. So the residue
census for a single constant moved by 100% with the **spelling** and by 0% with
the **value** -- which means every count this file prints about such a constant
was measuring the prose, not the tree. That is this file's own subject matter
happening to it, so `_POW` now reserves the shape next to the other three.

**Measured on the real tree, and the measurement is null: 0 caret shapes in all
three prose files, so the residue is 760 before and 760 after.** Nothing in this
run moved because of the fix, and that is the reason it is worth making rather
than the reason to skip it. The constant appears zero times in the prose today
and will not stay that way; a defect that only bites on a sentence nobody has
written yet is invisible exactly until the sentence exists, and a scanner that
has not been fixed stops helping at that moment and not before. The count is
printed in every run so that "zero" reads as *the tree does not spell it that
way* and never as *the reservation is not wired up*.

# The appendix is excluded from its own scan, and that exclusion is verified

`SCORING.md` section 11 is cut out of the constants check because a report that
supplies its own coverage certifies itself. The same hazard is here and worse:
this file's appendix is a table of numbers *about* numbers, so an unexcluded scan
would have the appendix supply the provenance of the appendix.

The exclusion is the region between `<!-- provenance:appendix:begin -->` and
`<!-- provenance:appendix:end -->`, and unlike a typed line span it **cannot be
widened to hide anything**, because it is not merely skipped: the block is
**regenerated from the residue outside it and compared**, and a block that has
been reordered, truncated, or had its header changed mismatches. Deleting the
markers does not hide anything either -- the block's own numbers re-enter the
residue and the unrecorded count rises by exactly what the block was holding.

**What that comparison is and is not, measured rather than asserted.** The cells
are the regeneration's *input* -- `render_block` copies `date`, `instrument` and
`command` out of the rows it is comparing -- so it verifies the block's
**structure and key set**, and it is **blind by construction** to a cell edited
in place. Each cell is covered instead by the check that owns it. An earlier
version of this paragraph claimed that "editing a number inside the block makes
the comparison mismatch"; that was true only of the two key columns, and a
figure whose name overstates what was measured is the defect this file is in the
business of catching -- including when the figure is one of its own checks.

**And deleting the markers returns the file to the "no block" state, which under
the asymmetry below is a green skip rather than a red.** That is not a hole: the
state it produces *is* the state in which nothing is refutable yet, and the
trace of the deletion is visible as a number -- the residue rises by exactly the
number of literals the deleted block was holding, and every one of them is named
as unrecorded. On the fixture that is 3 literals becoming 17.

**And the block stores five columns, not eight.** `value`, `line`, `date`,
`instrument`, `command` are stored; `witness`, `covers` and `state` are
**derived at run time and printed, never written into the file**. That is the
Tier-1 argument applied to the appendix itself: a `state` column in the
document would be a state word no check could hold true, and it would be
re-derived every time a witness's mtime moved. The block is regenerated from the
residue, so a row whose `value` or `line` does not match what the tokenizer found
is red, and so is a row that names a residue literal which is not there.

# THE VERDICT IS ASYMMETRIC, and that is the design, not an omission

**A block that does not exist is a skip. A block that exists and has a gap is a
red.** Read the two directions as separate claims, because they are:

* **No block in the file** -- that file's residue is reported as **counted, named,
  unrecorded** literals, and the six row checks become six *counted skips*
  against the same pinned total. Exit 0. The run says in words that the block
  does not exist, names how many literals are unrecorded because of it, and says
  what that means: **nothing in that file is refutable yet**, because there is no
  appendix there to hold a claim this file could contradict.
* **A block in the file, with a gap** -- a residue literal with no row, a row with
  no date, a row with no instrument, a row whose witness predates its date, a row
  for a literal that is not in the prose, a malformed row -- **red**, and named.
* **A block that is present and complete** -- green, and now Tier 2 and the row
  checks are carrying the weight, which is the only state in which their passing
  means anything.

**This asymmetry is the house pattern from `installed_copy_check.py`, and it is
load-bearing.** That file treats "no install here" as a counted skip inside a
**constant** total and exits 0, because *absence of a measurement is a missing
measurement, not a failure* -- a skip that is counted and named is a result.

**Do not "fix" this back to symmetric.** The symmetric version of this gate was
written and measured: it was **12/16 on the real tree, red before it had anything
to guard**, with 680 of 690 residue literals unrecorded, and it could not go green
because the block lives in three files this gate does not own. A gate that is red
before it has anything to guard is noise, and a noise gate gets switched off
within a release -- which is the same failure, one level up, as the four stale
check counts that sat in prose where nothing could go red. A permanently red gate
protects nothing, and the literals it was protecting rot with it ignored.

So the **appendix half** of this gate is green today and turns red the instant
anyone pastes a first appendix block, which is the moment it starts being able
to lie. That transition is the deliverable. (The gate as a whole is *not* green
on the real tree, and the docstring should not pretend otherwise: Tier 2 and
Tier 3b reds are open there, and every run names each one with its count. They
are findings about the documents, they predate this policy, and they are what
this gate is for. A gate that is green for the wrong reason is worse than one
that is red for the right one.) It is the same shape as the `marker_findable`
situation: the check is green only because it has measured one easy case -- *no
block exists* -- and the moment it measures more it goes red on real defects. A
gate that is equally green with a complete block and with a block that dropped a
row is not measuring the block at all.

**And the pinned total is the same number in both states, which is the whole
trick.** 6 row checks per prose file, 3 prose files, always: 6 that run, or 6
that skip, decided by whether the block is there. `EXPECTED_CHECKS` does not move
when a block is added, so adding one cannot be absorbed by bumping the pin --
which is the only way a total like this can be trusted to mean something. It is
also why the tally counts skips: a pin that counted only passes would have to
move the day a block appeared, and a pin that moves the moment a document
changes is a pin that can be moved to hide a defect.

**Measured, on a fixture tree, in this direction and not the other.** No block:
`12/12 checks, 18 counted skips, 30/30 slots`, exit 0. A block with one row
missing: `17/18, 12 counted skips, 30/30 slots`, exit 1, one red naming the two
unrecorded literals. That gap closed: `18/18, 12 counted skips, 30/30 slots`,
exit 0. **Same 30 before and after**, and the run went red in between on a real
defect rather than on the presence of a document.

# What this does NOT cover

Stated here rather than left for a reader to discover:

* **The tokenizer's blind spots, which are the failure mode most likely to bite,
  because a tokenizer that misses a class silently under-reports -- and an
  under-reporting scanner is worse than no scanner, because it reads as a clean
  bill of health.** So it does not merely *have* blind spots, it **counts them and
  prints the count next to the residue in every run**: CJK numerals (`一二三…`,
  which the ASCII-digits-only reader cannot see at all, in a substantially Chinese
  `README.md`), numbers split across a markdown soft wrap, literals inside inline
  code spans, literals inside fenced regions, and literals inside the appendix
  block itself. The line reads `residue N ... M invisible to this tokenizer`, and
  the per-file table gives the breakdown. **A gate that reports "690 literals" and
  one that reports "690 literals, of which at least 259 are invisible to this
  tokenizer" are not the same gate, and only the second one is honest.** A
  conservation check recomputes the total literal count by a second, whole-file
  scan and compares, so a bucket that is quietly dropped shows up as an
  arithmetic failure rather than as a smaller number. It still cannot tell a
  version (`0.1.0`) from a measurement, a date (`2026-08-14`) from a count, or a
  long hex digest from a literal, except by reserving those three shapes in order
  -- and a caret power (`2^28`) is now the fourth, because the same value written
  `107,374,182` is one literal and written `2^28` used to be two, which made the
  census move with the spelling instead of the value. It does not have to *parse*
  the invisible literals. It has to **say it is not looking at them.**
* **Line numbers are printed, so this file checks its own line numbers.** Every
  `file:line` it prints is re-resolved through a second, independent read of the
  same file -- `read_bytes().split(b"\n")`, no decode, no newline translation --
  and the literal is required to be on that line. A gate that reports a line
  number which does not resolve is worse than a gate that reports no line number,
  because a reader who trusts it is sent to the wrong line and does not find out.

  **What that check found on the real tree, and it corrects an earlier claim.**
  The hazard is real but it is not what it was first written down as. The `0x85`
  bytes in these three files are **never NEL**: every one of them is the second
  byte of a UTF-8 multi-byte sequence (the `Å` in the tree's mojibaked `Å`), 103
  in `README.md`, 33 in `README.en.md`, 91 in `CHANGELOG.md`, and **0 bare**.
  A UTF-8 reader that splits on `\n` and a byte-level reader that splits on `b"\n"`
  therefore agree exactly -- 388 / 471 / 1293 lines -- and this file's line
  numbers are right. What *does* move them is a reader that maps bytes 1:1 and
  then treats `U+0085` as a line break, which counts one extra line per `Å` and
  shifts every later line by the number of `Å` above it. A cp936 (GBK) read, which
  is what a GBK console does, is worse still: it cannot decode these files at all
  and raises rather than answering. So the shift is reported as a **named number
  per file** -- the largest a 1:1+NEL reader could introduce -- and is not this
  file's verdict, because the encoding defect belongs to the gate that owns
  encoding and a permanent red here would be the noise this file exists to avoid.
* **It does not verify that a Tier-3 number is correct.** Only that a human
  recorded which instrument produced it and when. There is no pin for 12.88 Å and
  this file does not pretend to be one.
* **The measurement case (Tier 3b) verifies that the sentence names a real
  instrument, and that is all it verifies.** It does not verify that the
  instrument produced this number, that the run happened, or that the number is
  right -- those need a run log, a date and a pin respectively, and none of the
  three is derivable from a tree. **Existence is a claim about the tree and about
  this machine's `PATH`**, not about the number: a command resolves through
  `shutil.which`, so a green that rests on a `PATH` hit is a statement about the
  machine the gate ran on. That is why each one prints the path it resolved to
  and whether it is inside this repository, and why the run prints **which of
  the remaining greens nothing in this repository is able to redden**.
* **A `file` green can be refuted on existence but not on the number.** Delete
  or rename the file and it goes red; **edit the number inside the file and it
  stays green**, because this tier asks whether the instrument was *named*, not
  whether it produced the number. That is the limitation behind the
  `command/in-repo` row of the refutation column, and it is stated there rather
  than left for a reader to infer from a green.
* **A `stem-match` is a finding about this reader's resolver as much as about
  the prose.** The sentence naming `core.py` is not wrong -- the tree really has
  a `core.py` under `dock-py/`. What is wrong is that *two* files in this tree
  carry that stem, so the resolver picked one by walk order. Rewriting the
  sentence to a full path closes it; rewriting it to a bare name does not.
* **`MAX_UNIT_LINES` is a judgement, and its whole effect is printed.** The
  measurement case refuses to credit an instrument from a unit wider than this,
  because a bullet without terminal punctuation yields a "sentence" that is the
  whole bullet. The number of units and residue literals each candidate cap would
  refuse is printed every run, so moving the constant is a visible decision with a
  stated price rather than a silent tightening.
* **The possessive widening is a fixture-only capability and is described as
  such.** Zero hits on the real tree, exercised only by fixtures.
* **The `call-sites` figure compares `N sections` against `section()` *call
  sites*, and is now named for that in its own output.** This was `sections`
  everywhere, and a figure whose name overstates what was measured is the exact
  defect this whole round is about. A call site inside a branch that never runs is
  counted and is not printed by the gate. The two agree on every gate measured
  today (measured, not assumed: the check prints both numbers and the size of any
  difference), and the check's name says `call sites`, not `sections`.

* **A figure is only as refutable as its sentence.** A count that names a gate but
  is qualified ("34 checks sat behind an `if`") is reported `unrefutable`, and
  that red means *no mechanism can contradict this number* -- not that the number
  is wrong. Rewriting a changelog to make every historical number refutable is a
  decision for the document's owner, and the count of such figures is printed so
  the size of that decision is visible before it is taken.

# Why this is a separate file from `docs_claims_check.py`

Because that file cannot run without the engine -- it imports `opendocking` -- and
the machine most in need of knowing whether the suite's own bookkeeping is intact
is the machine where the wheel was never built. This file reads source text and
nothing else, and it fails in the opposite direction from a documentation audit:
where `docs_claims_check.py` asks whether a documented number still holds, this
asks whether a quoted number still has anything that could refute it.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import re
import shutil
import sys
import time
from pathlib import Path

#: How many checks this file records, in every environment, whether or not an
#: appendix block exists. The total does not move when a block is added, because
#: the six row checks per prose file become six counted *skips* rather than
#: disappearing; that is what makes this total mean something. Measured on the
#: real tree and on a fixture tree carrying a block.
#:
#:    7  Tier 2: gates are parseable, pinned, and the quoted counts resolve
#:    2  Tier 3: the residue census is reported, and conserves against a
#:       second whole-file scan of the same bytes
#:    6  Tier 3b: every measurement names an instrument that exists, an unpinned
#:       gate is its own class rather than a green, a markdown document is not a
#:       carrier, a stem match is not a citation, every command-green records how
#:       it resolved and whether that is inside this repository, and all eight
#:       classes are conserved against the residue count
#:    1  the line numbers this file is about to print resolve through a
#:       second, independent read
#: 6x3  the appendix row checks: per prose file, either run or counted skip
#:    3  this file pins itself, holds itself to the same contract, and the
#:       checks that ran are the checks pinned
#:
#: **The census below is a different number from the pin above, and this file's
#: own auditor says so out loud rather than letting a reader assume the two
#: should match.** `EXPECTED_CHECKS` is how many *results* a run records, which
#: includes the 18 counted skips and the 6x3 appendix row slots. The census is
#: how many *call sites* the source has: 19 here, not 36. A call site does not
#: multiply by loop iteration, a skip is not a call, and a site that stops being
#: reached is not a site that stops being a site. Both are declared because a
#: reader who has one and assumes the other is wrong has no way to check.
#:
#: **These three lines were `NOT DECLARED ANYWHERE` until this block was added,
#: and the number in them was not typed.** `check_scripts_declare.py` only prints
#: paste-ready blocks for gates already on its generated snapshot, and this file
#: was on no snapshot at all -- those are the same fact, so the block it wanted
#: pasted was one it never printed. The numbers here were therefore derived by
#: calling that file's own `_sites_of` and `_guards_digest` over this file's
#: source, the same walk the auditor performs, rather than counted by hand: a
#: hand-typed census is the second thing that gets forgotten, which is the whole
#: reason the auditor exists. The digest is what catches a site changing guard
#: shape while the two counts stay the same.
#:
#: **`15 -> 18 -> 19` unconditional and the digest above has not moved once, and
#: that is the digest doing its job rather than nothing happening.** Three checks
#: were added for the two demotions and the command disclosure, then one for
#: this file's own bytes; all four sit at module level with no `if` around them,
#: so the guard set is unchanged and only the count moved. A site that had
#: migrated between guard shapes without changing the total would be invisible to
#: `sites:` alone and visible here.
#: GATE-DECLARE 1
#: sites: 19 unconditional + 1 guarded
#: guards: sha256:75170e23f43a76d26d18a8f7f58a48781235fa674949135fcb78714e8f4a21a8
EXPECTED_CHECKS = 37
#: The row checks are `APPENDIX_CHECK_NAMES` long, once per prose file, and the
#: sum of the two is the only reason the total is stable across the transition.
APPENDIX_CHECK_NAMES = (
    "the appendix is excluded from its own scan, and its structure is verified against "
    "what this file would generate",
    "every residue literal has a row: nothing is measured and unrecorded",
    "every row accounts for a literal that is actually there",
    "every row carries a date, an instrument and a command",
    "every row that names an instrument names a file that is there",
    "every row's witness is newer than the row's date",
)

CHECKS = 0
FAILURES: list[str] = []
SKIPPED: list[tuple[str, str]] = []
#: Skips that occupy a slot in `EXPECTED_CHECKS`. **A counted skip is a result**,
#: which is the house pattern from `installed_copy_check.py` and the whole reason
#: this gate is green today: the six appendix row checks per prose file are six
#: slots, and a file with no block fills them with skips instead of leaving them
#: empty, so pasting a block cannot be absorbed by bumping the pin. A skip that is
#: *not* one of a question's slots -- a prose file that is not in the tree at all
#: -- is recorded and named but does not occupy one.
COUNTED_SKIPS = 0


def _emit(line: str) -> None:
    """Print one line, on a console that may not be able to encode it.

    **Measured on the machine this gate runs on.** The first version of the
    line-number check printed `Å` in its own detail text, and the console is
    cp936, so `print` raised `UnicodeEncodeError` and the gate died *while
    reporting its own finding* -- after 25 checks had already run. A gate that
    crashes on the machine it is meant to protect is a gate that reports
    nothing at all, and the failure is invisible because the traceback is not a
    verdict.

    So every line goes out through here, and a character the console cannot
    encode is written as a `\\xNN` escape rather than taking the run down. The
    escape is visible, which is the point: the instrument says what it could not
    render instead of rendering less and saying nothing.
    """
    try:
        print(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="backslashreplace").decode(enc, errors="replace"))


def check(ok: bool, name: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(name)
    _emit(f"[{'ok ' if ok else 'FAIL'}] {name}")
    if detail:
        _emit(f"       {detail}")
    return ok


def section(t: str) -> None:
    _emit(f"\n=== {t} ===")


def skip(name: str, reason: str, counts: bool = True) -> None:
    """A question this run could not answer.

    Recorded, and counted inside `EXPECTED_CHECKS` when it stands in for a slot
    this file would otherwise have filled with a result. Never added to the pass
    total: a gate that reaches its number by declining to look is a gate that
    reaches its number by not running its checks, which is the failure this file
    exists to catch.

    The *slot* is a different thing from the *pass*, and that is the house
    pattern from `installed_copy_check.py`: a skip is a result, so it is counted
    and the run still exits 0, and the pinned total is the same number whether
    the measurement was taken or could not be. Absence of a measurement is a
    missing measurement, not a failure.
    """
    global COUNTED_SKIPS
    if counts:
        COUNTED_SKIPS += 1
    SKIPPED.append((name, reason))
    _emit(f"[{'SKIP' if counts else 'SKIP*'}] {name}: {reason}")


def read(path: Path) -> str:
    return io.open(path, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# The second read, and the check that this file's line numbers survive it
# ---------------------------------------------------------------------------
#
# This file prints `file:line` for every residue literal and every figure, and a
# line number that does not resolve sends a reader to the wrong line without
# telling them. So the line numbers are re-resolved through a read that shares no
# code with the one that produced them.
def byte_lines(path: Path) -> list[bytes]:
    """The same file as bytes, split on `LF`, with nothing decoded.

    Independent of `read` in the two ways that matter: it never runs the
    universal-newline translation, and it never produces a character that a
    decode could have invented or destroyed. `LF` is the ground truth for what a
    byte-oriented tool calls a line, and it is the definition every `file:line`
    in this report is quoting.
    """
    return path.read_bytes().split(b"\n")


def nel_split_lines(path: Path) -> list[bytes]:
    """What a reader that maps bytes 1:1 and calls `0x85` a line break sees.

    Named for the hazard rather than for a tool. **Measured on the real tree:**
    every `0x85` byte in these three files is the second byte of a UTF-8
    multi-byte sequence -- the `Å` in the tree's mojibaked `Å` -- and none is
    bare, so a UTF-8 reader and this byte reader agree exactly and this file's
    line numbers are correct. A reader that skips the decode is the one that
    gains a line per `Å` and shifts everything below it. The size of that shift
    is printed per file rather than asserted, because the encoding defect is
    another gate's to own and a permanent red here would be noise.
    """
    out, cur = [], bytearray()
    for byte in path.read_bytes():
        if byte in (0x0A, 0x0D, 0x85):
            out.append(bytes(cur))
            cur = bytearray()
        else:
            cur.append(byte)
    out.append(bytes(cur))
    return out


def nel_shift(path: Path) -> int:
    """How far a `0x85`-splitting reader could move the last line, in lines."""
    return len(nel_split_lines(path)) - len(byte_lines(path))


#: The three prose files whose numbers are the subject. `docs/` is
#: `docs_claims_check.py`'s scope and is not duplicated here.
PROSE = ("README.md", "README.en.md", "CHANGELOG.md")

APPENDIX_BEGIN = "<!-- provenance:appendix:begin -->"
APPENDIX_END = "<!-- provenance:appendix:end -->"


def appendix_span(lines: list[str]) -> tuple[int, int] | None:
    """The 0-based inclusive line span of the generated block, or `None`.

    Defined here rather than beside the residue census because the figure scan
    needs it too: the block is a table of numbers *about* numbers, and leaving
    it in scope for one tier while cutting it out of the other is how a pasted
    block ends up supplying its own provenance.
    """
    try:
        b = next(i for i, t in enumerate(lines) if t.strip() == APPENDIX_BEGIN)
        e = next(i for i, t in enumerate(lines) if t.strip() == APPENDIX_END)
    except StopIteration:
        return None
    return (b, e) if e > b else None

BLOCKED_BY = argparse.ArgumentParser(add_help=False)
BLOCKED_BY.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
BLOCKED_BY.add_argument("--emit", action="store_true")
ARGS, _ = BLOCKED_BY.parse_known_args()
ROOT = Path(ARGS.root).resolve()
SCRIPTS = ROOT / "scripts"

# ---------------------------------------------------------------------------
# The number reader, and the three shapes it reserves before it reads a number
# ---------------------------------------------------------------------------
#
#: A date, a three-part version and a long hex digest are consumed as whole
#: units and produce **no** numeric literal. A reader that did not reserve them
#: would turn `## [0.1.0]` into the number 0.1, `2026-08-14` into three numbers
#: and a truncated digest into a sixteen-digit measurement.
_DATE = r"\d{4}-\d{2}-\d{2}"
_VERSION = r"\d+\.\d+\.\d+"
_DIGEST = r"(?<![\w])[0-9a-fA-F]{12,}(?![\w])"
#: A power written with a caret, as `2^28`. **Reserved because the same value
#: spelled two ways must cost the same, and without this it does not.**
#: `2^28` is read as two literals -- `2` and `28` -- and `107,374,182` is read
#: as one, so the residue census for a single number moved by 100% with the
#: spelling and not at all with the value. That is the defect this file exists
#: to catch, and it was in the reader. The lookarounds keep the base from
#: swallowing a preceding word and keep the exponent from taking a trailing one.
_POW = r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*\^\s*[-+]?\d+(?![\w])"
#: The number itself. The two lookarounds are the whole reason this is not
#: `[\d.]+`: the leading one keeps `dist2_3` and `v2` from yielding a number, and
#: the trailing one keeps `39x26` from yielding one either.
_NUMBER = r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w)"

TOKEN_RE = re.compile(f"({_DATE})|({_VERSION})|({_DIGEST})|({_POW})|({_NUMBER})")
#: The caret shape on its own, so a run can **count what the reservation
#: consumed** rather than infer it from a total that would look the same either
#: way. `findall` on a group-less pattern yields whole matches, which is what the
#: count wants; had `_POW` carried a capture group this would count groups.
POW_RE = re.compile(_POW)
#: Which capture group is the number, now that a fifth shape precedes it. The
#: group index moved from 4 to 5, and every reader that indexed it by hand had
#: to move with it -- which is why it is **one name** here and no reader below
#: spells the index out.
_NUMBER_GROUP = 5

#: The shapes that assert a **total**, which is the only kind of number a pin can
#: refute. `ratio` is `N/M`, `total` is `N checks`/`N total`, `call-sites` is the
#: call-site census, `verb` is a run summary (`N passed`), and `pin` is a number
#: introduced by the literal token `EXPECTED_CHECKS` -- the most total-asserting
#: shape there is, since it names the pin.
#:
#: **The ratio branch refuses a hyphen.** `(?!-[\w])` on `N/M` is there because of
#: `the 127/80-conformation engine fixture, not the gate's 512` in `CHANGELOG.md`:
#: `127/80` there is a label on a noun, not a proportion, and the sentence says
#: so in as many words. Admitted, it was compared as a denominator against
#: `gpu_cpu_parity_check`'s pin of 9 and reported **stale** -- a red that means
#: "this proportion disagrees with a check total", which is a sentence about
#: nothing. A figure admitted that is not the kind of figure it is admitted as is
#: the same defect as a figure named for a measurement it is not.
#: **The third branch is the possessive, and its position in the alternation is
#: load-bearing rather than incidental.** `re` tries alternatives left to right at
#: each position, so in `the 127/80-conformation engine fixture, not the gate's
#: 512` the ratio branch is offered `127/80` first and refuses it on `(?!-[\w])`;
#: the possessive branch is then reached at the `gate`, not at the `512`, and
#: admits `gate's 512` on its own. Widening the detector therefore had to keep the
#: refusal reachable, and the proof that it is reachable is a check below, not a
#: reading of this comment.
FIGURE_RE = re.compile(
    r"(?i)(?<![\w.])[-+]?\d[\d,]*\s*/\s*[-+]?\d[\d,]*" r"(?!\d)(?!-[\w])"
    r"|(?<![\w.])[-+]?\d[\d,]*(?:\s*(?:checks?|gates?|sections?|total)\b|(?:\s*(?:passed|failed|skipped)(?![\w])))"
    r"|\b(?:gate|suite|check run|this gate)'?s\s+\d[\d,]*\b"
)
PIN_FIGURE_RE = re.compile(r"EXPECTED_CHECKS[^.\n]{0,60}?(?<![\w.])(\d+)(?!\w)")

#: A check count whose count word comes **before** the number, as in `the gate's
#: 512`. `FIGURE_RE`'s first two branches admit a figure on a count word *after*
#: the number -- `N checks`, `N passed` -- so a possessive attribution was not
#: admitted at all and the number fell through to the residue as a Tier 3
#: literal. A detector with a known hole in it is a defect, and this one was a
#: hole in the only mechanism that could have caught a wrong count, so the
#: document owner widened `FIGURE_RE` above to admit it. The earlier version of
#: this comment called the tree's own instance `CHANGELOG.md:1140`'s `but that is
#: the 127/80-conformation engine fixture, not the gate's 512` and called it
#: **live rot** because `gpu_cpu_parity_check.py` pins `EXPECTED_CHECKS = 9`.
#:
#: **That reading was wrong, and the correction is the reason the widening is not
#: the whole answer.** 512 is not a check total in that gate at all: it is
#: `N_CONF = 512` at `scripts/gpu_cpu_parity_check.py:240`, the conformation
#: count, while the pin is `EXPECTED_CHECKS = 9` at line 330. The sentence was
#: accurate and the rule was about to refute it for the wrong reason -- "this
#: check count disagrees with the gate's own pin" is a sentence about nothing
#: when one side is a conformation census. That is the same defect this file
#: exists to catch, in the reader rather than in the document, and it is why
#: `kind` for a possessive is `attributed` and **is not compared against a check
#: pin**; it is answered by whether the sentence names a gate at all. The owner's
#: decision was to widen the detector, and it is recorded here as taken; what the
#: widening exposed is that `FIGURE_RE` admits a figure on a *shape* and a
#: possessive names a shape without saying what the number counts. Routing that
#: to a per-gate constant census rather than to a check pin is a further decision
#: about what a figure is, and it is the document owner's, so this file reports
#: the count below instead of taking it.
#:
#: **MEASURED, and it is a fixture-only capability.** This reader finds **zero**
#: possessives in the three prose files on the real tree: the widening is
#: exercised only by fixtures under `target/prov/`, where a sentence is written
#: to carry one. It is therefore **not coverage**, and this file must not be
#: described as if it were -- it is a shape the detector is prepared to admit,
#: with the routing (an `attributed` figure is answered by whether a gate is named
#: and never by a check pin) pinned down before the sentence that motivated it
#: was edited away. The count is printed in every run precisely so that "zero"
#: stays visible instead of being mistaken for "not looked at".
POSSESSIVE_TOTAL_RE = re.compile(r"(?i)\b(?:gate|suite|check run|this gate)'?s\s+(\d+)")

#: A sentence must carry one of these for a figure in it to be a *check* count.
#: `suite` is in the list because it is what keeps `135/135` admitted: its own
#: sentence has no other count word, and dropping it would have moved the one
#: figure in this file that is unambiguously a gate total down into Tier 3.
COUNT_WORD = re.compile(
    r"(?i)\bchecks?\b|\bgates?\b|\bsuite\b|\bpassed\b|\bfailed\b|\bskipped\b|\btotal\b"
)
PYFILE_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\.py")
BARE_GATE_RE = re.compile(r"\b([a-z][a-z0-9_]*_check)\b")
#: Directories whose contents are rewritten by a build that changed no source,
#: and the binary suffixes whose mtime is a linker's opinion rather than a
#: source's. A witness in either class advances without the number changing.
BUILD_DIRS = frozenset({"target", "dist", "dist-gpu", "_wheelout", "__pycache__", "build"})
BINARY_SUFFIXES = frozenset({".pyd", ".dll", ".so", ".dylib", ".exe", ".whl", ".a", ".rlib"})


def numbers_in(text: str) -> list[tuple[str, int, int]]:
    """Every numeric literal as `(value, start, end)`, reserved shapes removed."""
    out = []
    for m in TOKEN_RE.finditer(text):
        g = m.group(_NUMBER_GROUP)
        if g is not None:
            out.append((g, m.start(_NUMBER_GROUP), m.end(_NUMBER_GROUP)))
    return out


def strip_inline_code(line: str) -> str:
    """Blank out inline code spans, keeping every character offset intact.

    Offsets are preserved by replacing with spaces rather than deleting, so a
    column position in the stripped line is still a column position in the
    original and a reported line number cannot drift away from its literal.

    This is also what keeps a **citation** of a number out of the figure set:
    `volume ** (1/3) * burial` is quoted from the source, and what the prose
    asserts about it is that the source says it, which is Tier 1's question.
    """
    return re.sub(r"`[^`]*`", lambda m: " " * (m.end() - m.start()), line)


# ---------------------------------------------------------------------------
# Tier 2, part 1: the gate index, read out of the gates themselves
# ---------------------------------------------------------------------------
def module_constant(path: Path, name: str):
    """A module-level literal assignment, read from the tree rather than a regex.

    `ast` rather than text because a pin quoted inside a docstring is not a
    declaration, and a declaration inside a function is not a module-level one.
    """
    try:
        tree = ast.parse(read(path))
    except (SyntaxError, OSError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if isinstance(node.value, ast.Constant):
                return node.value.value
    return None


def defines_check(path: Path) -> bool:
    """Does this gate define a `check()` of its own? A file that cannot fail is
    not a gate, and a gate that cannot fail is not a gate by another name."""
    try:
        tree = ast.parse(read(path))
    except (SyntaxError, OSError):
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "check"
        for n in ast.walk(tree)
    )


def section_census(path: Path) -> int | None:
    """`section()` **call sites**, which is not the same as sections executed."""
    try:
        tree = ast.parse(read(path))
    except (SyntaxError, OSError):
        return None
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "section"
    )


def import_closure(path: Path) -> set[Path]:
    """The files whose source can change this one, read from its own imports."""
    try:
        tree = ast.parse(read(path))
    except (SyntaxError, OSError):
        return {path}
    stems = {p.stem for p in ROOT.glob("*.py")} | {p.stem for p in SCRIPTS.glob("*.py")}
    found = {path}
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for stem in names:
            if stem in stems and stem != path.stem:
                for cand in (SCRIPTS / f"{stem}.py", ROOT / f"{stem}.py"):
                    if cand.is_file():
                        found.add(cand.resolve())
                        break
    return found


PINS: dict[str, int] = {}
SECTIONS: dict[str, int] = {}
GATE_STEMS: set[str] = set()
UNREADABLE: list[str] = []
for _p in sorted(SCRIPTS.glob("*.py")):
    GATE_STEMS.add(_p.stem)
    _pin = module_constant(_p, "EXPECTED_CHECKS")
    if isinstance(_pin, int):
        PINS[_p.stem] = _pin
    _no = module_constant(_p, "NO_EXPECTED_CHECKS")
    _sec = section_census(_p)
    if _sec is not None:
        SECTIONS[_p.stem] = _sec
    if _pin is None and _no is None and _sec is None:
        UNREADABLE.append(_p.name)

#: Every file under `scripts/` that is a gate by the strong marker and pins
#: nothing. The absence is itself a Tier 2 finding and it is *not* recoverable
#: from the prose, so it gets its own check and its own count.
UNPINNED_GATES = sorted(
    p.stem for p in SCRIPTS.glob("*.py")
    if defines_check(p)
    and p.stem not in PINS
    and module_constant(p, "NO_EXPECTED_CHECKS") is None
)


def unpinned_gate_detail(pins: int, unpinned: list[str]) -> str:
    """The unpinned-gate finding's detail, with the example chosen by the count.

    **The example is the count.** The sentence this replaced named one gate as a
    representative and was left describing nothing on the day the last unpinned
    gate was pinned, while the count beside it fell to zero and the sentence did
    not move. That is this file's own subject matter happening to it one level
    down: a fixed example attached to a measured number is a claim nothing here
    can contradict, sitting inside the finding that reports exactly that defect.
    So there is no example to keep in step, only the list the walk produced.

    **The two branches are disjoint, and zero is one of them.** A check that can
    say "I looked and there was none" has run, and a reader who cannot tell that
    apart from a check that had nothing to look at is reading a green that means
    silence. So the empty branch says what the walk covered and what would appear
    here if a gate ever dropped a pin, and the populated branch names every gate
    and the two module-level declarations that close one. Neither is a fallback
    for the other, and the count alone says which one ran.

    Both counts are passed in rather than read from a global so that the text is
    a function of two numbers a caller can hand it -- a fixture can ask what the
    sentence says at one gate and at three without editing this file to find out.
    """
    head = (
        f"{pins} gate(s) carry a live EXPECTED_CHECKS, read out of the gate files "
        f"with ast. {len(unpinned)} gate(s) define check() and pin nothing: "
        f"{unpinned}. "
    )
    if not unpinned:
        return head + (
            "Zero is a measurement and not an absence of one, and telling the two "
            "apart is the whole point: the pins counted just above were read out of "
            "the same gate files by the same walk, so an empty list here is the tree "
            "being clean rather than this check having nothing to look at. It is "
            "armed at this count, not retired by it -- the first gate that defines "
            "check() and declares neither EXPECTED_CHECKS nor NO_EXPECTED_CHECKS "
            "names itself here, by name, on the run that finds it."
        )
    return head + (
        "A count quoted for one of these has nothing that could ever contradict it, "
        "which is a stronger defect than a count that is merely wrong: the number "
        "may be right and be unverifiable forever. Each is closed by one of two "
        "module-level declarations and by nothing else -- a literal EXPECTED_CHECKS "
        "holding the number of results a run records, which this file then reads "
        "back out of the file with ast, or a literal NO_EXPECTED_CHECKS holding the "
        "reason this gate cannot have one. Named here so the work is a list and not "
        f"a search: {', '.join(unpinned)}."
    )


# ---------------------------------------------------------------------------
# Tier 2, part 2: the figures
# ---------------------------------------------------------------------------
def split_paragraphs(lines: list[str]):
    """`(start, end)` line numbers, 1-based inclusive, per blank-line-separated run."""
    buf, start = [], None
    for i, t in enumerate(lines, 1):
        if t.strip():
            if start is None:
                start = i
            buf.append(i)
        else:
            if buf:
                yield start, buf[-1]
            buf, start = [], None
    if buf:
        yield start, buf[-1]


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans as character offsets.

    A sentence, and not a paragraph, is the unit that admits a figure: a
    paragraph is too coarse in both directions, and two real figures in this
    tree -- `volume ** (1/3) * burial` and `9/9 bound-ligand atoms` -- sit in
    paragraphs that discuss checks elsewhere. A reader at paragraph scope
    compares an exponent and a burial fraction against a check total.

    **A newline is not a sentence boundary, and the first version of this
    function treated it as one.** Markdown soft-wraps, so `75 checks.` and the
    `` `EXPECTED_CHECKS` is now 338 `` that supersedes it are two *lines* of one
    sentence -- and splitting on the newline put each on its own, at which point
    the sentence holding `338` had no count word in it, was not admitted, and
    the `75` on the line above was reported as a **stale total rather than a
    dated reading**. That is the exact failure the supersession rule exists to
    prevent, reached by treating a line break as a thought break. A newline is
    whitespace here and nothing else.
    """
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?:;])\s+", text):
        if text[start:m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def split_items(lines: list[str], lo: int, hi: int) -> list[tuple[int, int]]:
    """`(start, end)` per **list item** inside a paragraph, or the paragraph itself.

    A markdown list in this tree has bullets with no blank line between them, so
    a paragraph can hold three unrelated claims about three different gates. The
    `CHANGELOG.md` paragraph holding the X11 entry, the CI-label entry and the
    reverse-verification entry is one such run, and resolving a gate across it
    attributes the workbench CI step's `(50 checks)` to the X11 gate four lines
    above it. The bullet is the unit a reader would use, and it is found by a
    pattern on the line, not by a table of where the claims are.
    """
    if not any(LIST_ITEM.match(lines[i - 1]) for i in range(lo, hi + 1)):
        return [(lo, hi)]
    out, start = [], None
    for i in range(lo, hi + 1):
        if LIST_ITEM.match(lines[i - 1]):
            if start is not None:
                out.append((start, i - 1))
            start = i
    if start is not None:
        out.append((start, hi))
    return out


def scope_of(lines: list[str], lineno: int) -> str:
    """`live` unless a **dated** release heading is open above this line.

    Derived from the heading structure, which is why it costs nothing to
    maintain: `## [Unreleased]` and a README with no release heading at all are
    claims about the tree as it stands, and `## [0.1.0] -- ...` is a record of
    what was true when it was cut.
    """
    for i in range(lineno - 1, -1, -1):
        m = re.match(r"^##\s+\[(.+?)\]", lines[i])
        if m:
            return "live" if m.group(1).strip().lower().startswith("unreleased") else "released"
    return "live"


def gates_in(lines: list[str], lo: int, hi: int) -> list[tuple[str, int]]:
    """`(stem, line)` for every `scripts/*.py` this paragraph names."""
    out = []
    for i in range(lo, hi + 1):
        for m in PYFILE_RE.finditer(lines[i - 1]):
            if m.group(1) in GATE_STEMS:
                out.append((m.group(1), i))
        for m in BARE_GATE_RE.finditer(lines[i - 1]):
            if m.group(1) in GATE_STEMS:
                out.append((m.group(1), i))
    seen, uniq = set(), []
    for stem, ln in out:
        if stem not in seen:
            seen.add(stem)
            uniq.append((stem, ln))
    return uniq


class Figure:
    """One quoted count, and everything the machine could resolve about it."""

    __slots__ = ("path", "line", "scope", "kind", "shape", "value", "gate",
                 "gate_line", "live", "verdict", "superseded_by", "note")

    def __init__(self, path, line, scope, kind, shape, value):
        self.path, self.line, self.scope, self.kind = path, line, scope, kind
        self.shape, self.value = shape, value
        self.gate: str | None = None
        self.gate_line = 0
        self.live: int | None = None
        self.verdict = ""
        self.superseded_by = ""
        self.note = ""


def figures_of(path: Path) -> list[Figure]:
    """Every check-count figure in one prose file, with the verdict it earns.

    A figure is admitted on either of two **derived** grounds, and the second
    one exists because the first is not sufficient:

    * its own sentence carries a count word -- which is what keeps `135/135`
      admitted, since `suite` is the only count word in that sentence; or
    * the list item it sits in names a gate -- which is what admits `39/39`,
      whose sentence (`... verifies it headlessly (39/39), and ...`) contains no
      count word at all, and `22 sections`, whose sentence is a fragment.

    A figure inside an inline code span is never admitted: `volume ** (1/3) *
    burial` is a formula quoted from the source, and a citation of a number is
    Tier 1's business, not a claim about a gate.

    **The appendix block is excluded here too, not only from the residue.** The
    argument in this file's own docstring -- a report that supplies its own
    coverage certifies itself -- was applied to Tier 3 and quietly not applied
    here, which meant a pasted block could put a *new figure* into Tier 2 through
    its own cells: a line number in a row that sat next to a `checks` word would
    be compared against a pin, by the block, about the block. The block is
    generated from the residue, so anything in it is already accounted for by the
    residue that produced it.
    """
    lines = read(path).split("\n")
    span = appendix_span(lines)
    out: list[Figure] = []
    for plo, phi in split_paragraphs(lines):
        if span and not (phi < span[0] or plo > span[1]):
            continue
        for lo, hi in split_items(lines, plo, phi):
            para = "\n".join(lines[lo - 1:hi])
            named = gates_in(lines, lo, hi)
            for s0, s1 in split_sentences(para):
                sent = para[s0:s1]
                if not (COUNT_WORD.search(sent) or named):
                    continue
                bare = strip_inline_code(sent)
                base = para[:s0].count("\n")
                spans: list[tuple[str, str, int, int]] = []
                for m in FIGURE_RE.finditer(bare):
                    shape = sent[m.start():m.end()].strip()
                    nums = numbers_in(shape)
                    if not nums:
                        continue
                    # In `N/M` the total a pin would refute is the **denominator**,
                    # and the two are equal in every ratio this tree quotes.
                    last = nums[-1][0]
                    kind = ("call-sites" if "section" in shape.lower()
                            else "ratio" if "/" in shape
                            else "verb" if re.search(r"(?i)passed|failed|skipped", shape)
                            else "attributed" if POSSESSIVE_TOTAL_RE.fullmatch(shape)
                            else "total")
                    spans.append((kind, shape, int(last.replace(",", "").lstrip("+-")),
                                  base + sent[:m.start()].count("\n") + 1))
                pm = PIN_FIGURE_RE.search(sent)
                if pm:
                    # Read on the **unstripped** sentence, and that asymmetry is
                    # deliberate. `EXPECTED_CHECKS` is a code-span citation by
                    # convention -- every mention of it in this tree is
                    # backticked -- so running this reader on the stripped text
                    # blanks the token out and the shape is never admitted. Which
                    # is how `75 checks` came to be reported as a *stale total*
                    # instead of a dated reading: the sentence that supersedes it
                    # was invisible, so nothing superseded it, and a position
                    # rule silently became a comparison rule. A citation of a
                    # literal is not a figure; a citation of the *pin* is the
                    # figure.
                    spans.append(("pin", f"EXPECTED_CHECKS is {pm.group(1)}",
                                  int(pm.group(1)),
                                  base + sent[:pm.start(1)].count("\n") + 1))
                for kind, shape, val, offset in spans:
                    lineno = lo + offset - 1
                    f = Figure(path.name, lineno, scope_of(lines, lineno), kind, shape, val)
                    if named:
                        # Nearest by line distance within the list item: markdown
                        # wraps, and `509 checks` names its gate two lines above it.
                        f.gate, f.gate_line = min(named, key=lambda g: abs(g[1] - lineno))
                    out.append(f)
    return out


ALL_FIGURES: list[Figure] = []
#: `(file, line, matched text)` rather than a preformatted string, because the
#: report below has to say whether each hit was *admitted*, and that is not known
#: until `figures_of` has run over the same line. Formatting it here would freeze
#: the old claim -- "not admitted" -- into the code, where the widening could not
#: correct it.
POSSESSIVE: list[tuple[str, int, str]] = []
for _name in PROSE:
    _p = ROOT / _name
    if not _p.is_file():
        skip(f"prose/{_name}", f"not present under {ROOT}", counts=False)
        continue
    ALL_FIGURES.extend(figures_of(_p))
    for _i, _line in enumerate(read(_p).split("\n"), 1):
        for _m in POSSESSIVE_TOTAL_RE.finditer(_line):
            POSSESSIVE.append((_name, _i, _m.group(0)))

#: Supersession, decided by position alone: within one scope, the **last**
#: figure quoted for a gate is that gate's current claim. Everything earlier for
#: the same gate is a dated reading, and this is what keeps a reverse-
#: verification's totals from being reported as a wrong number.
#: Supersession is keyed on `(scope, gate, quantity)`, and the third component
#: is load-bearing. Keyed on `(scope, gate)` alone, `22 sections` -- a *different*
#: quantity from `509 checks`, cited one line below it -- superseded the check
#: total, and the check total stopped being compared with anything. `ratio`,
#: `total`, `verb` and `pin` are all the same quantity (a gate's total, and the
#: run summary that reports it) and must supersede each other; `call-sites` is a
#: census of a different thing and must not.
#:
#: `attributed` is a third distinct quantity, and adding it is what stops the
#: widening from doing the thing a widened approver does. A possessive number is
#: grouped apart from `check-total` so it can neither supersede a real check
#: total nor be superseded by one, and so `the gate's 512` cannot end a run by
#: displacing the `512 checks` beside it.
def quantity_of(kind: str) -> str:
    if kind == "call-sites":
        return "call-sites"
    if kind == "attributed":
        return "attributed"
    return "check-total"


_by_gate: dict[tuple[str, str, str], list[Figure]] = {}
for _f in ALL_FIGURES:
    if _f.gate:
        _by_gate.setdefault((_f.scope, _f.gate, quantity_of(_f.kind)), []).append(_f)

for (scope, gate, _qty), group in _by_gate.items():
    group.sort(key=lambda f: (f.line, f.shape.find(str(f.value))))
    for f in group:
        f.live = PINS.get(gate)
        if f is group[-1]:
            if f.kind == "call-sites":
                # Judged against the section() census and **not** against the
                # pin: a gate's pin counts checks, and `22 sections` is not a
                # check count, so asking the pin about it is asking the wrong
                # question of the right file.
                derived = SECTIONS.get(gate)
                f.verdict = "current" if derived == f.value else "stale-call-sites"
            elif f.live is None:
                f.verdict = "no-pin"
            elif f.value == f.live:
                f.verdict = "current"
            else:
                f.verdict = "stale"
        else:
            f.verdict = "superseded"
            f.superseded_by = f"CHANGELOG.md:{group[-1].line}" if group[-1].path == "CHANGELOG.md" \
                else f"{group[-1].path}:{group[-1].line}"

for f in ALL_FIGURES:
    if f.kind == "attributed":
        # A possessive attributes a number to a gate **without saying what the
        # number counts**, so a check pin is the wrong instrument to ask: this
        # tree's own instance is `N_CONF = 512` against `EXPECTED_CHECKS = 9`, and
        # comparing them would report the sentence about the 127/80 fixture stale
        # when the sentence is accurate. The verdict is `unrefutable` and that
        # word is doing the real work -- the honest answer is that **no mechanism
        # here can refute this number**, which is the defect worth a red, and not
        # the false claim that the number is wrong.
        f.verdict = "unrefutable"
        f.superseded_by = ""
        f.note = ("possessive attribution: a check pin cannot refute it, because "
                  "the sentence does not say the number is a check total")
    elif not f.gate:
        f.verdict = "unrefutable"
        f.note = "names no gate in its paragraph"
    elif f.scope == "released" and f.verdict not in ("no-pin",):
        f.verdict = f"released-{f.verdict}"

VERDICT_ORDER = ("current", "superseded", "stale", "stale-call-sites", "no-pin",
                 "unrefutable", "released-current", "released-stale")


# ---------------------------------------------------------------------------
# Tier 3: the residue, and the appendix that records it
# ---------------------------------------------------------------------------
def fenced_lines(lines: list[str]) -> set[int]:
    out, inside = set(), False
    for i, t in enumerate(lines, 1):
        if t.lstrip().startswith("```"):
            inside = not inside
            out.add(i)
        elif inside:
            out.add(i)
    return out


TIER1_CITATION = re.compile(r"[\w./-]+\.rs::[\w:]+|[\w./-]+\.py::[\w:]+")

#: The characters a Chinese reader sees as a numeral and this one cannot. Counted,
#: **not parsed** -- parsing `统一` apart from `三十` needs a Chinese numeral
#: parser, and building one here would be a second tokenizer to keep in step. So
#: this is a character count and a run count, reported as *candidates*, and a run
#: that turns out to be an ordinary word (`统一`, `一直`, `三角`) inflates the
#: number rather than deflating it. That is the direction to be wrong in: the
#: claim being made is "this tokenizer is not looking here", and a count that is
#: too high understates its own coverage rather than overstating it.
CJK_NUMERALS = "一二三四五六七八九十百千万亿兆零〇两廿卅"
CJK_RUN_RE = re.compile(f"[{CJK_NUMERALS}]+")


def figure_spans(line: str) -> list[tuple[int, int]]:
    """The character spans of the numbers that belong to a check-count figure.

    **A line is not a figure.** The first version of this excluded Tier 2 by
    line proximity -- any literal within one line of a figure -- and the fixture
    caught it in one run: a sentence holding `100/100` beside four instrument
    readings on the *same line* lost all four to Tier 2, and the residue came
    out empty, which is the tokenizer failing silently in the direction that
    under-reports. The spans are recomputed with the same two readers the figure
    set is built from, so the two sides cannot disagree by construction.
    """
    bare = strip_inline_code(line)
    out = [(m.start(), m.end()) for m in FIGURE_RE.finditer(bare)]
    out += [(m.start(1), m.end(1)) for m in PIN_FIGURE_RE.finditer(line)]
    return out


def code_spans(line: str) -> list[tuple[int, int]]:
    """The character spans of this line's inline code spans, offsets intact.

    **The previous way of counting code-span literals was wrong and was counting
    the wrong thing.** `strip_inline_code` replaces a code span with spaces, so a
    literal inside one is not tokenized at all -- it is not in the residue, not in
    the figure set, and not in any bucket. The old reader then tried to recover
    it with `line[:cs].count("`") % 2` on the *stripped* line, where the
    backticks no longer exist, so that test is 0 for every literal and the branch
    is dead for a real code span. It was not dead for a **stray** backtick: a
    line with an odd number of them leaves one behind, and every literal after it
    was booked as `in-code` while sitting in plain prose. The tree's 10 `in-code`
    literals were 10 stray backticks, not 10 code spans.

    So the number is now a span test against the raw line, which is the same
    technique `figure_spans` already used for exactly this reason, and it books
    what it finds into a bucket instead of letting it vanish. An under-reporting
    scanner is worse than no scanner, because it reads as a clean bill of health;
    the residue total is unaffected in kind but the blind-spot number beside it
    was simply false, and a false blind-spot number is worse than no number.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"`[^`]*`", line)]


def reference_literal_count(lines: list[str]) -> int:
    """How many numeric literals the tokenizer finds in this file, unclassified.

    **A second pass over the same tokenizer, not a second tokenizer.** It exists
    to catch a *bucket* being dropped from the classification, which is the
    failure this file is guarding against, and calling it an independent
    recomputation of the tokenization would be the overclaim this round exists to
    remove. It uses `findall` where the residue pass uses `finditer`, over the
    same lines with the same pattern, so the two cannot disagree about what a
    literal is -- and a literal that fell out of every bucket makes them differ.

    It is deliberately not a whole-file scan. A whole-file scan would see a
    number split across a soft wrap as one or zero tokens where the per-line pass
    sees two, and the check would go red on a tokenizer difference that is not
    the thing it is for.

    **The group index is read from `_NUMBER_GROUP` and not written out, because
    writing it out is how the caret reservation would have silently broken this
    check.** `findall` yields tuples, so the index is one less than the numbered
    group. When `_POW` was inserted ahead of `_NUMBER` the group moved 4 -> 5 and
    a hard-coded `m[3]` would have gone on counting the *power* group and
    comparing the wrong population -- and it would still have balanced, because
    both sides would have been wrong the same way. A conservation check that
    cannot fail is not a check.

    **This check earned its place on its first run, by being wrong.** The first
    version asked `m[3] is not None`, and `findall` yields `''` -- not `None` --
    for a group that did not participate, so every match counted and the
    reference came out 15 above the classification: the 15 reserved date, version
    and digest shapes in the three files, which are consumed as whole units and
    correctly produce no literal. It was a red on the right tree for a real
    reason, which is the only way to know a conservation check is not decoration.
    """
    return sum(
        1
        for line in lines
        for m in TOKEN_RE.findall(line)
        if m[_NUMBER_GROUP - 1]
    )


def residue_of(path: Path) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """`(value, line)` literals that are neither Tier 1 nor Tier 2, and a census.

    The census is returned rather than discarded, and **every literal the
    tokenizer sees is booked into exactly one bucket**, so the buckets sum to the
    tokenizer's own count of the file and the difference between those two numbers
    is a check rather than a hope. The blind spots are buckets too: they are the
    literals this reader declined to classify, counted, so a blind spot that grows
    is a number rather than an absence.
    """
    lines = read(path).split("\n")
    span = appendix_span(lines)
    fenced = fenced_lines(lines)
    census = {"tier1": 0, "tier2": 0, "residue": 0, "code": 0, "fence": 0,
              "block": 0, "cjk_chars": 0, "cjk_runs": 0, "soft_wrap": 0,
              "residue_in_cjk": 0, "seen": reference_literal_count(lines),
              "fenced_lines": 0, "pow": 0}
    out: list[tuple[str, int]] = []
    for i, raw in enumerate(lines, 1):
        in_block = bool(span) and span[0] <= i <= span[1]
        in_fence = i in fenced
        if in_fence:
            census["fenced_lines"] += 1
        cspans = code_spans(raw)
        fspans = figure_spans(raw)
        # The Tier 1 test stays on the **stripped** line, which is where it has
        # always been: a `file::symbol` citation inside backticks is a citation
        # of a symbol rather than a measurement on the page, and moving this
        # would move the residue for a reason unrelated to the blind spots.
        stripped = strip_inline_code(raw)
        for value, cs, _ce in numbers_in(raw):
            if in_block:
                census["block"] += 1
                continue
            if in_fence:
                census["fence"] += 1
                continue
            if any(a <= cs < b for a, b in cspans):
                census["code"] += 1
                continue
            if any(a <= cs < b for a, b in fspans):
                census["tier2"] += 1
                continue
            if TIER1_CITATION.search(stripped):
                census["tier1"] += 1
                continue
            out.append((value, i))
            census["residue"] += 1
            if re.search(r"[\u4e00-\u9fff]", raw):
                census["residue_in_cjk"] += 1
        if in_block or in_fence:
            continue
        # The two classes the tokenizer cannot see **at all**, and the only reason
        # they are counted here is that a scanner which cannot see them must say
        # so. A number split across a markdown soft wrap is two literals to this
        # reader, or one and a fragment, and a CJK numeral is nothing at all.
        for m in CJK_RUN_RE.finditer(stripped):
            census["cjk_runs"] += 1
            census["cjk_chars"] += m.end() - m.start()
        # **A caret power is one number written two ways, and this reader used to
        # see two numbers.** `2^28` was tokenized as `2` and `28`, so the residue
        # census for one constant moved by 100% with the spelling and by 0% with
        # the value -- the census was measuring prose, not the tree. `_POW` now
        # reserves the shape, and this counts how many the reservation actually
        # consumed, so "zero" reads as "the tree does not spell it that way" and
        # not as "the reservation is not wired up". **Measured on the real tree:
        # 0, and the residue is 760 before and 760 after** -- so no number in that
        # run moved because of the fix. That is the point, and it is also why the
        # count cannot be left implicit: on a fixture carrying `2^28` in all
        # three files (`target/prov/fix_pow`) the same line reads **3**, and the
        # residue is 7 before the reservation and 1 after. A fix whose only
        # evidence is a zero is indistinguishable from a fix that does nothing,
        # so the zero is reported and the fixture is the other half of the
        # evidence. The constant appears zero times in the real prose today and
        # will not stay that way; a defect that only bites on a sentence nobody
        # has written yet is invisible until the sentence exists, which is
        # exactly when a scanner that has not been fixed stops helping.
        census["pow"] += len(POW_RE.findall(stripped))
        nxt = lines[i] if i < len(lines) else ""
        # A number split across a soft wrap leaves a **digit** on both sides of
        # the break, so the test is digit-then-digit and nothing else. The first
        # version of this also accepted a trailing `+`, `-` and `.`, which
        # counted every markdown bullet that ends in a hyphen: 170 phantom
        # splits in this tree, of which 1 was real. A blind-spot number that is
        # an order of magnitude too large is its own kind of dishonesty.
        if re.search(r"\d$", stripped.rstrip()) and \
                re.match(r"\d", strip_inline_code(nxt).lstrip()):
            census["soft_wrap"] += 1
    census["blind_excluded"] = census["code"] + census["fence"] + census["block"]
    census["blind_unparsed"] = census["soft_wrap"] + census["cjk_runs"]
    census["booked"] = (census["tier1"] + census["tier2"] + census["residue"]
                        + census["blind_excluded"])
    return out, census


# ---------------------------------------------------------------------------
# Tier 3b: the measurement case -- row-free, because a run is not a row
# ---------------------------------------------------------------------------
#
# A constant is produced by a file containing it, so its marker is a citation and
# Tier 1 answers it. A check count is produced by a gate, so its marker is a pin
# and Tier 2 answers it. A **measurement** is produced by *running something*, so
# its provenance is a **run** -- historical, and not re-derivable from a tree that
# has moved on since. That is the whole reason the row shape is wrong for this
# class and right for the other two, and it is why this section asks a smaller
# question instead: **does the sentence name an instrument that exists?**
#
# Nothing here is typed. The file index comes off the disk, the pin comes out of
# the gate's own AST, and whether a command exists is `shutil.which`. There is no
# table of "which instrument produced the 12.88 A" to forget, which is the same
# reason `[measured]` was rejected and the same reason the derived columns are not
# written into the appendix.

#: **A unit wider than this is not a sentence, and must not hand out a green.**
#: `split_sentences` breaks on terminal punctuation, and a markdown list item
#: that runs to the end of its bullet without one -- which this tree's
#: `CHANGELOG.md` does for thirty lines at a stretch -- yields a "sentence" that
#: is the whole bullet. That is where the coincidence came back when this was
#: first written: **69 residue literals inherited `scripts/core_check.py` from the
#: first sentence of one 30-line bullet**, which is the rejected defect one level
#: up -- a file credited for a number because it appears somewhere in the same
#: paragraph. So the unit is capped, and a literal in a wider unit gets its own
#: class rather than a green.
#:
#: **The cap is measured, not chosen, and the whole curve is printed by this
#: file** so that moving it is a visible decision with a stated price. Over the
#: 1070 units these three files contain: 249 span one line, 416 two, 255 three,
#: 95 four, and 12 span six or more. The tail begins at four, so that is where
#: the constant sits; a cap of 2 would refuse 482 residue literals, 3 refuses 260,
#: 4 refuses 125, 6 refuses 19. The number of literals each candidate cap would
#: refuse is printed in every run.
MAX_UNIT_LINES = 4

#: The instrument shapes a sentence may name. All three are **resolved against
#: the tree or the machine**, never against a list typed here.
#:
#: A filename is only an instrument if the extension is one this tree actually
#: ships, which is why the list is a list of suffixes rather than `\w+\.\w+` --
#: a bare `\w+\.\w+` would read `volume.energy` and `f.fasta` out of a formula
#: and go looking for them.
INSTRUMENT_FILE_RE = re.compile(
    r"(?<![\w/.-])([\w][\w./-]*\.(?:py|rs|md|toml|yml|yaml|json|sh|txt|cfg|ini|html|"
    r"glsl|glslv|cu|cuh|cpp|h|hpp|ps1|bat|lock))")
#: A gate by its bare name, as `core_check` rather than `scripts/core_check.py`.
#: Same reader `gates_in` already uses, so "names a gate" means one thing in
#: Tier 2 and in Tier 3b.
INSTRUMENT_GATE_RE = re.compile(r"(?<![\w.-])([a-z][a-z0-9_]*_check)\b")
#: A named command is the first word of a code span. The span matters: a
#: command in this prose is always quoted, and a bare word in running text is
#: not a command.
INSTRUMENT_COMMAND_RE = re.compile(r"`([^`\n]+)`")
#: What a program's name may look like. **This filter is the difference between
#: a named command and a piece of prose that happens to sit in backticks**, and
#: the first version of the reader had no filter at all and produced these
#: "commands": `|ndc|` (a markdown table cell whose backticks paired across the
#: pipe), `decode("utf-8",` , `os, cd, (0,` and `f32::MAX`. Two characters
#: minimum, and nothing that is not a word: that is what separates `cargo test`
#: from `(dims−1)·spacing`.
PROGRAM_NAME_RE = re.compile(r"^[A-Za-z][\w.-]{1,31}$")

#: Directories that are not instruments because a build rewrote them, and
#: dot-directories because those are the VCS and the tool cache. `target/` is in
#: `BUILD_DIRS`, and a sentence naming a file under it is naming a build product.
NO_INSTRUMENT_DIRS = BUILD_DIRS | {".git", ".github", ".pytest_cache", ".venv"}


def instrument_index() -> tuple[set[str], dict[str, list[str]]]:
    """`(tree-relative posix paths, stem -> paths)` for the whole tree.

    **Walked with `os.walk(followlinks=False)`, and that is not a style choice.**
    `Path.rglob("*")` follows directory symlinks, and this tree has eighteen of
    them -- other agents' fakeroots, which are junctions back into the tree --
    so a recursive glob on it does not come back in reasonable time. `os.walk`
    with `followlinks=False` does not descend into a link, so the walk is
    bounded, skips the build directories, and finishes in hundredths of a second.
    Nothing here is deleted and no link is followed, which is also why this file
    is safe to run beside another agent's fixtures.
    """
    paths: set[str] = set()
    by_stem: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        rel = Path(dirpath).relative_to(ROOT)
        if rel.parts and (rel.parts[0] in NO_INSTRUMENT_DIRS
                          or any(p.startswith(".") for p in rel.parts)):
            continue
        dirnames[:] = [d for d in dirnames if d not in NO_INSTRUMENT_DIRS]
        for name in filenames:
            p = (rel / name).as_posix()
            if p.split("/")[0] in NO_INSTRUMENT_DIRS:
                continue
            paths.add(p)
            by_stem.setdefault(Path(name).stem, []).append(p)
    return paths, by_stem


TREE_FILES, TREE_BY_STEM = instrument_index()

#: `shutil.which` walks every directory on `PATH` -- milliseconds a call on
#: Windows, and this file wants thousands. Memoised; the answer cannot change
#: inside one run, and the memo is also the honest record of what was asked.
_WHICH: dict[str, str | None] = {}


def which(tok: str) -> str | None:
    if tok not in _WHICH:
        _WHICH[tok] = shutil.which(tok)
    return _WHICH[tok]


def command_class(tok: str) -> tuple[str, str | None]:
    """`('in-repo'|'outside-repo'|'absent', resolved path)` for a named command.

    **The split is the finding, and it was asked for after the measurement, not
    before it.** `shutil.which("split")` has answered with two *different*
    coreutils binaries on two runs of this gate on this machine, with no line of
    this repository changing in between -- neither of them a project artefact, and
    the second not even the same installation of the program. **The specific
    paths are deliberately not written here.** They were, in the first version of
    this docstring, and `check_repo_docs.py` did not flag them because a path in
    a docstring is masked as prose -- which is exactly why they had to go
    unprompted: a stale path in a comment is a claim the run never consults, so
    it cannot go red when the machine changes, and it reads as a specification
    while being a snapshot. The run prints the resolution; the argument does not
    need the literal. A
    command that resolves *inside* the tree is a different claim from one that
    resolves on `PATH`, and on the real tree **not one of them resolves inside**.

    `in-repo` is a containment test on the resolved path and not on the name, so
    it is the same question asked the same way for every command. A `.EXE` under
    this repository's own `target/` would count as in-repo; that is a real
    limitation and the class is printed beside the path rather than hidden.
    """
    r = which(tok)
    if r is None:
        return "absent", None
    try:
        p = Path(r).resolve()
        inside = p == ROOT or ROOT in p.parents
    except OSError:
        inside = False
    return ("in-repo" if inside else "outside-repo"), r


def resolve_instrument(kind: str, tok: str) -> tuple[str, str, str, str, int]:
    """`(state, resolved, kind, how, siblings)` for one named instrument.

    `state` is `exists`, `unpinned` or `absent`. **`unpinned` is not a flavour of
    `exists`**, which is the whole point of that class: the file is there, and
    the gate that owns it says out loud by saying nothing that nobody can hold
    it to.

    `how` is `exact`, `stem` or `which`, and `siblings` is the number of files
    the stem index matched. **The stem branch is the false-green generator, and
    it is measured rather than argued.** A sentence that writes `core.py` is a
    citation; a resolver that accepts a bare `core` because *some* file in the
    tree is called `core.py` is a coincidence, and when more than one file shares
    the stem the choice among them is `os.walk` order, which is neither stable
    across machines nor meaningful. Measured on the real tree: **15 instrument
    resolutions went through the stem index and 15 of those 15 matched a stem
    shared by 2 or 4 files.** So the fallback was not a rare tie-break; on the
    sentences that used it, it was wrong about *which* file every time.
    """
    if kind == "command":
        return ("exists" if which(tok) else "absent"), tok, "command", HOW_WHICH, 0
    if kind == "gate":
        cand = f"scripts/{tok}.py"
        if cand in TREE_FILES:
            return ("unpinned" if tok in UNPINNED_GATES else "exists"), cand, "gate", HOW_EXACT, 1
        sibs = TREE_BY_STEM.get(tok, [])
        for hit in sibs:
            return (("unpinned" if Path(hit).stem in UNPINNED_GATES else "exists"),
                    hit, "gate", HOW_STEM, len(sibs))
        return "absent", tok, "gate", HOW_EXACT, 0
    for cand in (tok, f"scripts/{tok}"):
        if cand in TREE_FILES:
            stem = Path(cand).stem
            if stem in GATE_STEMS:
                return ("unpinned" if stem in UNPINNED_GATES else "exists"), cand, "gate", HOW_EXACT, 1
            return "exists", cand, ("document" if cand.endswith(".md") else "file"), HOW_EXACT, 1
    sibs = TREE_BY_STEM.get(Path(tok).stem, [])
    for hit in sibs:
        return "exists", hit, ("document" if hit.endswith(".md") else "file"), HOW_STEM, len(sibs)
    return "absent", tok, "file", HOW_EXACT, 0


def instruments_named(text: str) -> list[tuple[str, str]]:
    """`(kind, token)` for every instrument this sentence names, in reading order.

    Files and gates are read from the **raw** sentence, not the inline-code
    stripped one, because `` `scripts/core_check.py` `` *is* how this prose names
    a file and `strip_inline_code` would blank the only mention of it. That is
    the opposite asymmetry from the figure reader, and for the same underlying
    reason: a citation is not the claim, the claim is what the sentence asserts
    about the instrument.
    """
    out: list[tuple[str, str]] = []
    for m in INSTRUMENT_FILE_RE.finditer(text):
        out.append(("file", m.group(1)))
    for m in INSTRUMENT_GATE_RE.finditer(text):
        out.append(("gate", m.group(1)))
    for m in INSTRUMENT_COMMAND_RE.finditer(text):
        toks = m.group(1).split()
        if not toks:
            continue
        head = toks[0]
        if not PROGRAM_NAME_RE.match(head):
            # A single capital letter is a symbol (`R`), not a program, and
            # `shutil.which` will happily resolve one: this machine has a
            # versioned R installation on PATH, so an unfiltered reader credited
            # a sentence that merely said `R` with naming an instrument. The
            # install path is not written here on purpose -- it is a fact about
            # this machine, it would go stale silently, and nothing reads it.
            continue
        if which(head) or (len(toks) > 1 and not head.startswith("-")):
            out.append(("command", head))
    return out


#: The eight classes a measurement can land in. **The order is the reading order
#: in the table below, and it is deliberately NOT the order the sentence names
#: things in** -- `_CARRIER_RANK` is what decides, because a verdict that depends
#: on which word came first in a sentence is a verdict a reword can flip.
#:
#: **`document` and `stem` are not carriers and are not green.** The question
#: this tier asks is not "is the artefact there" -- all four shapes are there --
#: but "what would have to happen for this green to go red". A pinned gate
#: answers that (drop the pin and the class moves). A command answers it (uninstall
#: it and the class moves). A file the sentence named **in full** answers it on
#: existence. A *stem the sentence never wrote as a path* does not answer it at
#: all, and a **markdown document cannot**, because a document contains the
#: number *because a human typed it there*: "the document says 12.88" and "12.88
#: is right" are then the same proposition. Prose agreeing with prose is the
#: relation this file refused when it refused the producer search's green branch,
#: and it is the same relation an appendix row has when its `instrument` cell
#: holds a `.md`.
#:
#: Measured on the real tree, and the measurement is the argument: of 30 greens,
#: 6 rested on a document and 3 on a stem match, so 9 of 30 -- 30% -- were
#: resting on something no mechanism in this repository can contradict.
#:
#: **What the first name credits, in one line: a citation.** `names-existing` is
#: the question "did the sentence name an artefact that is there", and the
#: artefact is printed so a reader can go and look at it. It is **not** the
#: question "would running that artefact print this figure", and no ordering or
#: membership in this tuple can make it one. The name says *named and there*
#: because that is the whole of the question asked.
#:
#: **The name this class used to carry was itself the defect, and the old token is
#: deliberately not repeated anywhere in this file.** Read as English next to a
#: green in a provenance table, a class named after the *instrument* is a claim
#: about production -- this figure is credited to an instrument -- and a reader
#: is entitled to believe it. Four figures in the READMEs were green under it while
#: their own sentences said the named script did not measure them; the four
#: cases, quoted against the code each rests on, are in this file's docstring
#: under the Tier 3b heading. So the rename is not cosmetic and it is not a
#: claim that the check got better: **the check is the same check**, the class
#: table, its verdict column, the kind breakdown and the per-file census all
#: move together because they read this one tuple. A second vocabulary for one
#: bucket would be this file's own subject matter one level down, which is why
#: the superseded name is spelled nowhere above, below, or in this file's own
#: output -- a reader who greps for it and finds it is reading a plan, not a
#: class, and this file does not leave that ambiguity lying around.
MEASURE_CLASSES = ("names-existing", "names-unpinned-gate", "names-document",
                   "stem-match", "names-absent", "names-nothing",
                   "unit-too-coarse", "no-unit")
#: The classes that are a human's problem. `names-existing` is the only green;
#: the rest is the case a human must look at, and each of the four non-green
#: *existence* classes is red for a different reason and is reported separately so
#: that collapsing them into one number would destroy the distinction.
MEASURE_RED = ("names-unpinned-gate", "names-document", "stem-match",
               "names-absent", "names-nothing", "unit-too-coarse", "no-unit")

#: **How a name was turned into a path, which is a different question from
#: whether the path is there.** `exact` is the sentence naming the file; `stem`
#: is the resolver matching a bare `core.py` against a stem index and taking the
#: first hit; `which` is `shutil.which`. The middle one is the false-green
#: generator this file now refuses, and it is derived here rather than typed.
HOW_EXACT, HOW_STEM, HOW_WHICH = "exact", "stem", "which"


def measure_units(path: Path) -> list[tuple[int, int, str]]:
    """`(first, last, text)` per unit, at the granularity the figures use.

    The same `split_paragraphs` -> `split_items` -> `split_sentences` walk
    `figures_of` already performs, so a figure and a measurement in the same
    sentence are read the same way, and the appendix block is cut out here for
    the same reason it is cut out there: a report that supplies its own coverage
    certifies itself, and a pasted block naming a gate must not be able to
    answer for the prose around it.
    """
    lines = read(path).split("\n")
    span = appendix_span(lines)
    out = []
    for plo, phi in split_paragraphs(lines):
        if span and not (phi < span[0] or plo > span[1]):
            continue
        for lo, hi in split_items(lines, plo, phi):
            para = "\n".join(lines[lo - 1:hi])
            base = lo - 1
            for s0, s1 in split_sentences(para):
                out.append((base + para[:s0].count("\n") + 1,
                            base + para[:max(s0, s1 - 1)].count("\n") + 1,
                            para[s0:s1]))
    return out


#: **The order the sentence's named instruments are read in, and it is a claim
#: about strength rather than about position.** The strongest carrier named wins,
#: so a sentence that names both a pinned gate and a markdown document is green,
#: and a sentence that names only the document is red. The previous reader took
#: whichever resolved first in reading order, which made the verdict depend on
#: word order in a sentence -- so reordering a sentence could change a green into
#: a red with no fact in the tree having changed.
#:
#: **Ranks are only for the four carrier shapes; everything else is ranked by
#: `MEASURE_RED` order instead**, which is the same class of preference the
#: unpinned-gate branch has always made and prints its price for. There is no
#: `None` here on purpose: a rank table with a hole in it is a table where a
#: lookup can fail, and the one thing this file refuses to do is raise halfway
#: through a run.
_CARRIER_RANK = {
    ("gate", "exists"): 0,
    ("command", "in-repo"): 1,
    ("command", "outside-repo"): 2,
    ("file", "exists"): 3,
}


def classify_measurement(value: str, unit: tuple[int, int, str] | None) -> dict:
    """Put one residue literal in exactly one class, and say why.

    **The unit is chosen by the literal, not the other way round.** A literal is
    attached to the first unit on its line that contains its own text, and to the
    first unit on its line otherwise -- which is what happens when a number was
    reformatted by the tokenizer (`1,000` for `1000`) or when two sentences share
    a line. Two literals that land on the same unit necessarily get the same
    answer, because the question is a property of the sentence and not of the
    column the number sits in.
    """
    if unit is None:
        return {"klass": "no-unit", "instrument": "", "kind": "", "span": 0, "why": ""}
    first, last, text = unit
    span = last - first + 1
    row = {"instrument": "", "kind": "", "span": span}
    if span > MAX_UNIT_LINES:
        row.update(klass="unit-too-coarse",
                   why=f"the unit spans {span} line(s), over MAX_UNIT_LINES="
                       f"{MAX_UNIT_LINES}, so no instrument is credited from it")
        return row
    # `(state, resolved, kind, how, siblings, token)` per name, in reading order.
    resolved = [r + (t,) for k, t in instruments_named(text) for r in [resolve_instrument(k, t)]]
    unpinned = [r for r in resolved if r[0] == "unpinned"]
    absent = [r for r in resolved if r[0] == "absent"]
    #: **A carrier is a name the sentence wrote, and `how` is what decides that.**
    #: The first version of this demotion listed the kinds (`gate`, `command`,
    #: `file`) and left `how` out, and read 0 stem matches on a tree where the
    #: scratch measurement had just counted 15 -- because a stem-resolved *file*
    #: is still a `file`, so it stayed in the carrier list and went green exactly
    #: as before. The class reported zero and the false greens carried on, which
    #: is the worst shape a check can have: it looked like it had found nothing.
    #: The test is `how`, not `kind`, because `how` is the question "did the
    #: sentence name this artefact or did the resolver guess".
    carriers = [r for r in resolved
                if r[0] == "exists" and r[2] in ("gate", "command", "file")
                and r[3] != HOW_STEM]
    if unpinned:
        # Unpinned first, deliberately and still: a sentence that names an
        # unaccountable gate is weaker evidence than one that names a plain file,
        # and letting the file win would hide exactly the class the owner asked to
        # see. The price of the choice is printed by the report.
        r = unpinned[0]
        row.update(klass="names-unpinned-gate", instrument=r[1], kind=r[2],
                   why=f"{r[1]} defines check() and pins no EXPECTED_CHECKS")
    elif carriers:
        # Strongest first, and the rank is `_CARRIER_RANK`, not reading order.
        # A command is split here because one that resolves inside this repository
        # is a different claim from one this repository has never heard of.
        def strength(r):
            if r[2] == "command":
                return _CARRIER_RANK[("command", command_class(r[5])[0])]
            return _CARRIER_RANK[(r[2], "exists")]
        r = sorted(carriers, key=strength)[0]
        if r[2] == "command":
            where, resolved_path = command_class(r[5])
            row.update(klass="names-existing", instrument=resolved_path or r[5],
                       kind=f"command/{where}",
                       why=f"named {r[5]!r}, which resolved to {resolved_path!r} -- "
                           f"{where} this repository")
        else:
            row.update(klass="names-existing", instrument=r[1], kind=r[2],
                       why=f"named as {r[5]!r}, resolved to {r[1]}")
    elif absent:
        r = absent[0]
        row.update(klass="names-absent", instrument=r[1], kind=r[2],
                   why=f"{r[5]!r} is not in the tree and is not on PATH")
    else:
        # A stem match and a document are named *after* the carriers on purpose:
        # a sentence that names a real gate AND a document is green, and only a
        # sentence resting on the document alone is reported here.
        #
        #: **The stem is tested first, and that order is load-bearing.** A document
        #: can be reached by the stem index too -- `VERIFICATION.md` is not a tree
        #: path, and two files in this tree are called that -- and a resolution the
        #: resolver guessed is a worse finding than one the sentence wrote, because
        #: in that case the *artefact* is in doubt as well as the carrier. So a
        #: guessed document is a `stem-match` and says so, rather than a
        #: `names-document` that quietly implies the file was named.
        stems = [r for r in resolved if r[0] == "exists" and r[3] == HOW_STEM]
        docs = [r for r in resolved if r[0] == "exists" and r[2] == "document"]
        if stems:
            r = stems[0]
            plural = ("s" if r[4] > 1 else "")
            row.update(klass="stem-match", instrument=r[1], kind=f"stem/{r[2]}",
                       why=f"the sentence names {r[5]!r}, which is not a path in this "
                           f"tree; it was matched by STEM to {r[1]}, and {r[4]} file"
                           f"{plural} share that stem, so which one was meant is "
                           f"decided by directory-walk order and not by the sentence")
        elif docs:
            r = docs[0]
            row.update(klass="names-document", instrument=r[1], kind="document",
                       why=f"{r[1]} is markdown, and a document contains this number "
                           f"because a human wrote it there -- so 'the document says "
                           f"it' and 'it is right' are the same proposition")
        else:
            row.update(klass="names-nothing",
                       why="the sentence names no file, no gate and no command")
    return row


#: `{file: [(value, line, class, instrument, kind, unit_span, why)]}`. The
#: population loop is below, next to the residue census it reads -- it needs
#: `RESIDUE`, and for the same reason the residue is computed before the first
#: line is printed: the classes and the line numbers reported beside them come
#: out of one pass over one set of literals, not two readers that could disagree.


BLOCK_COLUMNS = ("value", "line", "date", "instrument", "command")


def parse_block(lines: list[str], span) -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    """Rows out of the appendix table, keyed by `(value, line)`, plus junk.

    Only the five **stored** columns are read. `witness`, `covers` and `state`
    are derived at run time and are never read from the file, because a state
    word in the document is the `[measured]` marker this scheme rejected.

    A row that is not five cells is returned as **junk rather than skipped**,
    because the first version of this skipped it and a mutation that blanked an
    instrument cell by accident produced six cells instead of five. The run went
    red, but it went red as "unrecorded" and "block mismatch", which reads like
    a defect in the document and is actually a defect in the row. Junk is named.
    """
    rows: dict[tuple[str, str], dict[str, str]] = {}
    junk: list[str] = []
    if not span:
        return rows, junk
    for t in lines[span[0] + 1:span[1]]:
        t = t.strip()
        if not t.startswith("|") or set(t) <= set("|-: "):
            continue
        cells = [c.strip() for c in t.strip("|").split("|")]
        if len(cells) != len(BLOCK_COLUMNS):
            junk.append(f"{len(cells)} cell(s): {t[:60]}")
            continue
        if cells[0] == "value":
            continue
        # The line is kept as the **string** the cell held, because the key
        # this row is matched on is `(value, str(line))` built from the residue.
        # An `int` here compares unequal to a correct row, and the two-sided
        # check then reported all four rows as both unrecorded and phantom at
        # once -- a failure that looks like a defect in the prose and is
        # actually a defect in the reader.
        rows[(cells[0], cells[1])] = dict(zip(BLOCK_COLUMNS, cells))
    return rows, junk


def derive_witness(instrument: str) -> tuple[str, str, str]:
    """`(witness, covers, class)` for a row's declared instrument.

    `class` is one of `direct`, `over-claims`, `under-claims`, `no-witness`,
    `missing`. It is a *class* and not a verdict on purpose: the honest question
    about an mtime is what it fails to cover, and `current` is a comparison
    rather than a property of the witness.
    """
    if instrument.strip().lower() in ("", "-", "none", "n/a"):
        return "none", "-", "no-witness"
    cand = (ROOT / instrument.strip()).resolve()
    if not cand.is_file():
        return "MISSING", "-", "missing"
    try:
        inside = ROOT in cand.parents or cand == ROOT
    except OSError:
        inside = False
    rel = cand.relative_to(ROOT).as_posix() if inside else cand.as_posix()
    closure = import_closure(cand) if cand.suffix == ".py" else {cand}
    covers = f"1 of {len(closure)}"
    # Classified on the path **relative to the tree**, never on the absolute
    # one. The absolute test was wrong in a way the real tree hid: `target` is a
    # build directory, so every witness under this repository's own `target/`
    # scratch came back `over-claims` -- and the fixture tree lives there, so
    # all four of its witnesses were mislabelled until this was fixed. A class
    # that depends on where the checkout happens to sit is not a class.
    if any(p in BUILD_DIRS for p in rel.split("/")) or cand.suffix.lower() in BINARY_SUFFIXES:
        klass = "over-claims"
    elif len(closure) > 1:
        klass = "under-claims"
    else:
        klass = "direct"
    return rel, covers, klass


def mtime_of(rel: str) -> float | None:
    if rel in ("none", "MISSING", "-"):
        return None
    try:
        return (ROOT / rel).stat().st_mtime
    except OSError:
        return None


# ---------------------------------------------------------------------------
# The residue, computed before anything is printed, so the line numbers this
# run is about to print can be verified before they are printed
# ---------------------------------------------------------------------------
RESIDUE: dict[str, list[tuple[str, int]]] = {}
CENSUS: dict[str, dict[str, int]] = {}
for _name in PROSE:
    _p = ROOT / _name
    if _p.is_file():
        RESIDUE[_name], CENSUS[_name] = residue_of(_p)

#: Tier 3b's census, built here for the reason the residue census is: the
#: classes are computed before the first line is printed, so the class counts and
#: the line numbers they are reported with come out of one pass over one set of
#: literals rather than two readers that could disagree.
MEASURED: dict[str, list[tuple]] = {}
for _name in PROSE:
    if _name not in RESIDUE:
        continue
    _units = measure_units(ROOT / _name)
    _by_line: dict[int, list[tuple[int, int, str]]] = {}
    for _u in _units:
        for _ln in range(_u[0], _u[1] + 1):
            _by_line.setdefault(_ln, []).append(_u)
    _rows = []
    for _value, _line in RESIDUE[_name]:
        _cands = _by_line.get(_line, [])
        _hit = next((u for u in _cands if _value in u[2]), _cands[0] if _cands else None)
        _c = classify_measurement(_value, _hit)
        _rows.append((_value, _line, _c["klass"], _c["instrument"], _c["kind"],
                      _c["span"], _c["why"]))
    MEASURED[_name] = _rows

#: A pin figure's line is the line of the **number**, not of the
#: `EXPECTED_CHECKS` token that introduces it. Markdown soft-wraps, and the tree
#: has ``...`s `EXPECTED_CHECKS` is now 338 and the`` on one line, so the token
#: and its number can be a line apart; pointing at the token made this file print
#: a line number for a number that was on the next line, and the line self-check
#: below caught it on its first run.
#: Every `file:line` this run prints, and whether a second read agrees that it
#: points at the literal being quoted. Built from the two populations that print
#: line numbers -- the residue literals and the Tier 2 figures -- so a new kind
#: of printed location has to be added here to be checked, which is the point.
#:
#: A figure is probed on its **value**, not on its shape. The shape of a pin
#: figure is synthesised (`EXPECTED_CHECKS is 338`) and is therefore not text in
#: the file at all -- it is not in the document with backticks, and the document
#: says `is now 338` where the shape says `is 338`. Probing a synthesised string
#: against a line is a red about the reader, not about the file, which is the
#: failure the two earlier generations of this reader had in different clothes.
LINE_PROBES: list[tuple[str, int, str]] = [
    (n, ln, v) for n, lits in RESIDUE.items() for v, ln in lits
] + [(f.path, f.line, str(f.value)) for f in ALL_FIGURES]

LINE_FAULTS: list[str] = []
NEL_SHIFT: dict[str, int] = {}
for _name in PROSE:
    _p = ROOT / _name
    if not _p.is_file():
        continue
    _n_text = len(read(_p).split("\n"))
    _n_byte = len(byte_lines(_p))
    NEL_SHIFT[_name] = nel_shift(_p)
    if _n_text != _n_byte:
        LINE_FAULTS.append(
            f"{_name}: the text reader sees {_n_text} line(s) and the byte reader "
            f"{_n_byte}; every line number for this file is in question")
for _name, _ln, _val in LINE_PROBES:
    _p = ROOT / _name
    if not _p.is_file():
        continue
    _bl = byte_lines(_p)
    if not 1 <= _ln <= len(_bl):
        LINE_FAULTS.append(
            f"{_name}:{_ln} is outside the {len(_bl)} line(s) the byte reader sees")
    elif _val.encode("utf-8") not in _bl[_ln - 1]:
        LINE_FAULTS.append(
            f"{_name}:{_ln} does not contain {_val!r} when the file is read as bytes: "
            f"that line holds {_bl[_ln - 1][:60]!r}")


print("provenance: Tier 2 (gate-observed) and Tier 3 (instrument readings)")
print(f"tree: {ROOT}")
print(f"prose: {', '.join(PROSE)}")
print(f"gates indexed: {len(GATE_STEMS)}  with a live pin: {len(PINS)}")


# ==========================================================================
section("Tier 2: every quoted check count is compared against a live pin")

check(
    not UNREADABLE,
    "every gate file is parseable, so no pin is missing because a file could not be read",
    f"{len(GATE_STEMS)} file(s) under scripts/, {len(UNREADABLE)} unreadable: {UNREADABLE}. "
    f"A pin read out of a file that will not parse is not a pin, and a gate that "
    f"cannot be read is not thereby excused",
)

check(
    not UNPINNED_GATES,
    "every gate that defines check() pins its total or says out loud why it cannot",
    unpinned_gate_detail(len(PINS), UNPINNED_GATES),
)

_by_verdict: dict[str, list[Figure]] = {}
for f in ALL_FIGURES:
    _by_verdict.setdefault(f.verdict, []).append(f)

print(f"\n  {len(ALL_FIGURES)} check-count figure(s) admitted, by verdict:")
for v in VERDICT_ORDER:
    if v in _by_verdict:
        print(f"    {v:18} {len(_by_verdict[v]):3}")
print()
#: A possessive hit and whether the widened `FIGURE_RE` actually admitted it. The
#: two lists are reported separately because they fail in **opposite** ways: a hit
#: that is not admitted is a hole in the detector, and a hit that is admitted and
#: then refuted against a check pin is a false red about a number that was never a
#: check total. Printing one combined count would have hidden the difference, and
#: the earlier version of this block printed the false half of it as fact.
_ADMITTED_LINES = {(f.path, f.line) for f in ALL_FIGURES if f.kind == "attributed"}
_unadmitted = [t for t in POSSESSIVE if (t[0], t[1]) not in _ADMITTED_LINES]
print(f"  **possessive attributions:** {len(POSSESSIVE)} total -- "
      f"{len(POSSESSIVE) - len(_unadmitted)} admitted as `attributed` and answered by "
      f"whether a gate is named, {len(_unadmitted)} still not admitted and named below")
if not POSSESSIVE:
    print("    **zero on this tree, so this is a fixture-only capability and NOT coverage.**")
    print("    The widening is exercised only by fixtures under target/prov/, where a")
    print("    sentence is written to carry one. The routing it needs -- an `attributed`")
    print("    figure is answered by whether a gate is named and never by a check pin, so")
    print("    that `the gate's 512` (N_CONF, a conformation count, against a pin of 9) is")
    print("    not reported stale -- is pinned down here whether or not a real sentence")
    print("    exercises it. 'Zero hits' must read as 'not exercised on this tree', never")
    print("    as 'not looked at'")
for _n, _i, _t in _unadmitted:
    print(f"    {_n}:{_i} \"{_t}\"")
print()
for f in sorted(ALL_FIGURES, key=lambda x: (x.path, x.line)):
    pin = "no pin" if f.live is None else str(f.live)
    print(f"  {f.path}:{f.line} [{f.scope}] {f.shape!r} -> gate={f.gate or '(none)'} "
          f"({f.gate_line or '-'}) live={pin} => {f.verdict}"
          + (f" <- superseded by {f.superseded_by}" if f.superseded_by else ""))

check(
    not _by_verdict.get("stale"),
    "no live check count disagrees with the gate's own pin",
    f"{len(_by_verdict.get('stale', []))} stale, {len(_by_verdict.get('current', []))} current, "
    f"{len(_by_verdict.get('superseded', []))} superseded. A figure is compared only if its "
    f"sentence asserts a total and its paragraph names the gate; the comparison is against "
    f"a pin read out of that gate's own file at run time",
)

check(
    not _by_verdict.get("unrefutable"),
    "no check count is quoted without naming the gate it belongs to",
    f"{len(_by_verdict.get('unrefutable', []))} unrefutable: "
    + "; ".join(f"{f.path}:{f.line} {f.shape!r}" for f in _by_verdict.get("unrefutable", []))
    + ". A count that names nothing has nothing that can contradict it, which is the "
      "defect that let these numbers rot in the first place. This red means *no mechanism "
      "can refute this number*, not that the number is wrong",
)

check(
    not _by_verdict.get("no-pin"),
    "no quoted count belongs to a gate that pins nothing",
    f"{len(_by_verdict.get('no-pin', []))} figure(s) quote a gate with no pin: "
    + "; ".join(f"{f.path}:{f.line} -> {f.gate}" for f in _by_verdict.get("no-pin", []))
    + f". The unpinned gates are {UNPINNED_GATES}",
)

check(
    not _by_verdict.get("stale-call-sites"),
    "every 'N sections' figure agrees with the section() CALL SITES in that gate",
    f"{len(_by_verdict.get('stale-call-sites', []))} disagree. Compared: "
    + ("; ".join(
        f"{f.path}:{f.line} says {f.value} sections for {f.gate}, whose AST holds "
        f"{SECTIONS.get(f.gate)} call sites"
        for f in ALL_FIGURES if f.kind == "call-sites") or "none")
    + ". **What this measures, named for what it is:** a `section()` call site, from "
      "the gate's own AST. A call site inside a branch that never runs is counted here "
      "and is not printed by the gate, so this figure is *not* a count of sections "
      "executed and it is named `call-sites` everywhere in this file rather than "
      "`sections`. Sections actually executed is measured by nothing here: it is the "
      "gate's own stdout, and this file does not run gates. The quantity, the "
      "instrument and the limitation are all three named, because a figure whose name "
      "overstates what was measured is the defect this check exists to catch",
)

_groups = {k: v for k, v in _by_gate.items() if v}
_multi_current = [k for k, g in _groups.items()
                  if sum(1 for f in g if f.verdict in
                         ("current", "stale", "stale-call-sites", "no-pin")) != 1]
_misfiled = [(f.path, f.line) for k, g in _groups.items() for f in g
             if (f is g[-1]) != (f.verdict in
                                 ("current", "stale", "stale-call-sites", "no-pin"))]
check(
    not _multi_current and not _misfiled,
    "each (scope, gate) group has exactly one current claim, and it is the last one",
    f"{len(_groups)} group(s); {_multi_current} with other than one current claim, "
    f"{len(_misfiled)} figure(s) misfiled: {_misfiled}. This is what keeps a reverse- "
    f"verification's 61/0/14/75 from being reported as a wrong number: the position "
    f"decides, so the human annotation beside it saying exactly that is corroboration "
    f"and not the mechanism -- delete the annotation and the verdict does not move",
)


# ==========================================================================
section("Tier 3: the residue, the census, and the blind spots")

_total_res = sum(len(v) for v in RESIDUE.values())
_t1 = sum(c["tier1"] for c in CENSUS.values())
_t2 = sum(c["tier2"] for c in CENSUS.values())
_seen = sum(c["seen"] for c in CENSUS.values())
_booked = sum(c["booked"] for c in CENSUS.values())
_excl = sum(c["blind_excluded"] for c in CENSUS.values())
_code = sum(c["code"] for c in CENSUS.values())
_fence = sum(c["fence"] for c in CENSUS.values())
_block = sum(c["block"] for c in CENSUS.values())
_wrap = sum(c["soft_wrap"] for c in CENSUS.values())
_cjk_c = sum(c["cjk_chars"] for c in CENSUS.values())
_cjk_r = sum(c["cjk_runs"] for c in CENSUS.values())
_pow = sum(c["pow"] for c in CENSUS.values())
_ne = len([c for n, c in CENSUS.items() if c["residue_in_cjk"]])

print(f"\n  residue: {_total_res} literal(s) across {len(RESIDUE)} file(s)")
print(f"    numeric literals the tokenizer saw in these files: {_seen}")
print(f"      classified Tier 1 (adjacent to a file::symbol citation): {_t1}")
print(f"      classified Tier 2 (part of a figure above):               {_t2}")
print(f"      residue (neither):                                    {_total_res}")
print(f"      excluded from the residue by design:                   {_excl}"
      f"  (inline code span {_code}, fenced region {_fence}, appendix block {_block})")
print(f"    **not seen at all, counted and not parsed:**")
print(f"      numbers split across a markdown soft wrap:             {_wrap}")
print(f"      CJK numeral characters:                                {_cjk_c}"
      f"  in {_cjk_r} candidate run(s) -- counted, NOT parsed")
print(f"    **seen, and consumed whole by a reserved shape:**")
print(f"      caret powers (`2^28`), which used to read as two literals:  {_pow:4}"
      f"   -> {2 * _pow} literal(s) before the reservation, 0 after")
print()
print(f"  per file, and the blind spots beside them:")
for _name in PROSE:
    if _name not in CENSUS:
        continue
    c = CENSUS[_name]
    print(f"    {_name:14} residue {c['residue']:4}  seen {c['seen']:4}  tier1 {c['tier1']:3}  "
          f"tier2 {c['tier2']:3}  code {c['code']:3}  fence {c['fence']:3}  block {c['block']:3}  "
          f"wrap {c['soft_wrap']:2}  cjk {c['cjk_chars']:4}c/{c['cjk_runs']:3}r")
print()

check(
    _total_res > 0 and all(c["residue"] == len(RESIDUE[n]) for n, c in CENSUS.items()),
    "the residue is computed, and the count of literals this tokenizer CANNOT see "
    "is printed beside it in every run",
    f"{_total_res} residue literal(s) out of {_seen} the tokenizer saw, of which "
    f"{_ne} file(s) put some on a line containing CJK. **The blind spots are a named "
    f"number, not an admission in a comment:** {_excl} literal(s) were seen and then "
    f"excluded from the residue by design ({_code} inside an inline code span, {_fence} "
    f"inside a fenced region, {_block} inside the appendix block itself), and "
    f"{_wrap} number(s) are split across a markdown soft wrap where this reader sees "
    f"two literals or one and a fragment. {_cjk_c} CJK numeral character(s) in {_cjk_r} "
    f"run(s) are invisible to an ASCII-digits-only reader -- counted, **not parsed**, so "
    f"the number of numerals they actually form is not measured and this over-counts "
    f"rather than under-counts, which is the direction to be wrong in. A gate that "
    f"reports \"{_total_res} literals\" and one that reports \"{_total_res} literals, of "
    f"which at least {_excl + _wrap} are not in the residue and {_cjk_c} more are in a "
    f"script this reader cannot count\" are not the same gate, and only the second is "
    f"honest. It does not have to parse the blind spots. It has to say it is not "
    f"looking at them",
)

check(
    all(c["booked"] == c["seen"] for c in CENSUS.values()),
    "every literal the tokenizer saw is booked into exactly one bucket, so a bucket "
    "that is quietly dropped is an arithmetic failure and not a smaller number",
    f"{_booked} booked against {_seen} seen"
    + (f"; MISMATCH: "
       f"{ {n: (c['booked'], c['seen']) for n, c in CENSUS.items() if c['booked'] != c['seen']} }"
       if _booked != _seen else "")
    + ". The reference count is a second pass over the **same** tokenizer (`findall` "
      "where the residue pass uses `finditer`), so this is not a claim that the "
      "tokenization was re-derived by someone else -- it is a claim that no literal "
      "fell out of the classification. The old reader dropped one: a literal inside an "
      "inline code span was blanked by `strip_inline_code` before the tokenizer ever "
      "saw it, and the recovery test could not see it either, so code-span literals "
      "were in no bucket at all and the number printed as `in-code` was counting stray "
      "backticks instead",
)


# ==========================================================================
section("Tier 3b: every measurement, resolved by the instrument its sentence names")

_m_by_class: dict[str, dict[str, int]] = {n: {c: 0 for c in MEASURE_CLASSES}
                                          for n in MEASURED}
_m_by_kind: dict[str, int] = {}
_m_inst: dict[str, int] = {}
for _n, _rows in MEASURED.items():
    for _v, _ln, _k, _inst, _kind, _span, _why in _rows:
        _m_by_class[_n][_k] += 1
        if _k == "names-existing":
            _m_by_kind[_kind] = _m_by_kind.get(_kind, 0) + 1
            _inst = _inst or "(unresolved)"
            _m_inst[_inst] = _m_inst.get(_inst, 0) + 1

_m_total = sum(len(v) for v in MEASURED.values())
_m_counts = {c: sum(_m_by_class[n].get(c, 0) for n in MEASURED) for c in MEASURE_CLASSES}
_m_green = _m_counts.get("names-existing", 0)
_m_unpinned = _m_counts.get("names-unpinned-gate", 0)
_m_document = _m_counts.get("names-document", 0)
_m_stem = _m_counts.get("stem-match", 0)
_m_absent = _m_counts.get("names-absent", 0)
_m_nothing = _m_counts.get("names-nothing", 0)
_m_coarse = _m_counts.get("unit-too-coarse", 0)
_m_nounit = _m_counts.get("no-unit", 0)
_m_human = sum(_m_counts.get(c, 0) for c in MEASURE_RED)
#: **The demoted total: literals that were green before this round and are not
#: now, with the reason for each.** Typed as a count only -- the evidence is
#: printed below, per class, from the same rows the checks read.
_m_demoted = _m_document + _m_stem

print(f"\n  { _m_total } residue literal(s) put to one question: does the sentence name an")
print("  instrument that exists? A file in the tree, a gate with a pin, a command on PATH.")
print("  No row, no date, no column to maintain -- and no appendix block required.")
print("  **A green below is that question answered yes. It is a citation, not a")
print("  production: nothing in this run executes the artefact it names.**\n")
#: The verdict for each class, and **the event that would make it go red**. The
#: second column is the whole argument for the class existing: a green whose
#: refuting event is a sentence rather than a fact is a green that cannot fail.
_MEASURE_VERDICT = {
    "names-existing": "green -- it is named and it is there; nothing here ran it, "
                      "see the refutation column",
    "names-unpinned-gate": "RED -- exists, and nothing can hold it to account",
    "names-document": "RED -- prose agreeing with prose is not a witness",
    "stem-match": "RED -- a stem is a coincidence, and the tree says which files matched",
    "names-absent": "RED -- a human must look at this",
    "names-nothing": "RED -- a human must look at this",
    "unit-too-coarse": "RED -- the unit is too coarse to credit an instrument",
    "no-unit": "RED -- a gap in this reader, which must never read as zero",
}
#: What would have to happen **in this repository** for a green of that kind to
#: turn red. Written out rather than computed because the honest answer for two
#: of the three kinds is "nothing here can", and a computed column that returned
#: `True` for those would be the decoration this table exists to prevent.
_MEASURE_REFUTES = {
    "gate": "yes -- drop the gate's EXPECTED_CHECKS and it becomes names-unpinned-gate",
    "file": "on existence only -- delete or rename the file and it goes red; EDIT the "
            "number inside it and it stays green, because this tier asks whether the "
            "instrument was named, not whether it produced the number",
    "command/in-repo": "yes -- the program is a file in this tree, so deleting it is red",
    "command/outside-repo": "**NO** -- nothing in this repository can make it red. The "
                            "program lives on this machine's PATH; changing PATH "
                            "reprints the path and does not redden the class, and "
                            "uninstalling the program is not a change to the tree",
}
#: **The fallback below is not decoration, it is the reason the table can be
#: indexed by a derived value.** `_MEASURE_REFUTES` is keyed on the green's
#: *kind*, and kinds are computed by `classify_measurement` -- so a tree that
#: produced a kind this version has never seen would raise `KeyError` partway
#: down the file, after some checks had already reported. A gate that dies while
#: producing its report produces no report, and the traceback is not a verdict.
#: An unknown kind therefore prints as unknown **and says it is unknown**, which
#: is the same rule as the `no-unit` bucket: a gap in this reader must never read
#: as a green.
_MEASURE_REFUTES_UNKNOWN = ("**UNKNOWN** -- this version of the table has no "
                            "refutation statement for this kind of green, which is "
                            "a gap in the reader and not a claim about the kind")
print(f"  {'class':22} {'count':>6}  verdict")
for _c in MEASURE_CLASSES:
    print(f"    {_c:20} {_m_counts.get(_c, 0):6}  {_MEASURE_VERDICT[_c]}")
print(f"    {'-- needs a human':20} {_m_human:6}  the union of every red class")
print()
print("  per file:")
for _n in PROSE:
    if _n not in _m_by_class:
        continue
    c = _m_by_class[_n]
    _hu = sum(c.get(k, 0) for k in MEASURE_RED)
    print(f"    {_n:14} " + "  ".join(f"{k}={c.get(k, 0)}" for k in MEASURE_CLASSES)
          + f"   needs-a-human={_hu}")
print()
print("  **WHAT THE GREEN CREDITS, AND THE FOUR WAYS A NAMED ARTEFACT IS THE WRONG ONE.**")
print("  The green class is a CITATION check. It establishes that the sentence's unit")
print("  named something that EXISTS -- a file in this tree, a gate with a live pin, a")
print("  program this machine's PATH resolves -- and it prints that artefact so a")
print("  reader can go and look at it. It does NOT establish that running the artefact")
print("  prints this figure, that the figure is right, or that the run ever happened.")
print("  Those need a run log, a pin and a date, and none of the three is derivable")
print("  from a tree. The class is named `names-existing` for the question it asks;")
print("  the name it used to carry read as a claim about production, and a reader")
print("  is entitled to believe a class name.")
print()
print("  A named artefact is the wrong attribution in four ways, and all four were live")
print("  in this repository's own prose under the old name. They are worked through")
print("  with the code each one rests on in this file's docstring, under the Tier 3b")
print("  heading, and they are printed here because a green that cannot be misread")
print("  should not have to be defended in a comment:")
print("    1. THE FIGURES ARE IN A COMMENT, AND NOTHING PRINTS THEM. An exhaustiveness")
print("       ladder of 12.88 A at 16 and 1.26 A at 64, credited to")
print("       scripts/redock_benchmark.py, is at redock_benchmark.py:116 -- inside a")
print("       source comment -- and that script derives ONE exhaustiveness per box")
print("       from its volume, so it structurally cannot print a pair for one box.")
print("    2. THE NAMED SCRIPT HAS NO SUCH CASE. A crambin reference energy credited")
print("       to scripts/redock_benchmark.py, whose CASES is 1STP/3PTB/2NNQ/1HVR. The")
print("       sentence that denies the attribution still NAMES the script, and naming")
print("       it is all this class asks.")
print("    3. THE RANGE IS A HAND-WIDENED ROUNDING. A 2-6 gradient range credited to")
print("       examples/audit_poses.py, which prints one worst-norm and a descent")
print("       ladder -- a maximum and eight rungs, never an interval. The widening")
print("       inherited the citation of the measurement it was widened from.")
print("    4. THE NUMBERS ARE TYPED LITERALS IN THE INSTRUMENT'S OWN DETAIL STRING.")
print("       Five framing figures inside workbench_interaction_check.py's printed")
print("       `F` detail -- 8.80 A, 3.8x, 2.05 A, 32 px, 5 -- are a historical record")
print("       typed into a message, printed beside the four values the check actually")
print("       measures and the 1.0/0.5 thresholds it asserts them against.")
print()
print(f"  **the size of the exposure, measured rather than asserted:** "
      f"{_m_by_kind.get('gate', 0) + _m_by_kind.get('file', 0)} of {_m_green} green(s) rest")
print("  on a SCRIPT this repository ships -- a gate or a plain file, reached through a")
print("  path the sentence wrote -- and a script named in a sentence is exactly what a")
print("  reader takes to have printed the number. The refutation column below is the")
print("  honest answer to how far each of those greens can be pushed. Naming an artefact")
print("  is a question this reader can answer; whether the run happened is a question it")
print("  cannot, and it says so rather than implying otherwise.")
print()
print("  **WHAT EACH GREEN RESTS ON, AND WHETHER ANYTHING HERE CAN REFUTE IT.**")
print("  The green branch is not flat, and a green that cannot go red is not a green:")
#: **The sort key below used the OUTER `_k` instead of its own parameter `k`,
#: and it was one insertion away from crashing.** `_k` is the *class* variable
#: left over from the counting loop above, so the key function read
#: `_m_by_kind['names-nothing']` -- a key that is by construction never in that
#: dict. It survived only because the value `_k` happened to be left holding
#: a real carrier kind at the moment `sorted` called it. Adding a loop between
#: the two changed which value was left over and it raised. A key
#: function that reads a variable it did not receive is a coin flip, and this one
#: was on edge for a release.
for _k in sorted(_m_by_kind, key=lambda k: -_m_by_kind[k]):
    _pct = 100 * _m_by_kind[_k] // max(_m_green, 1)
    print(f"    {_k:20} {_m_by_kind[_k]:5}  {_pct:3}% of the green")
    print(f"      {'can it go red?':20} {_MEASURE_REFUTES.get(_k, _MEASURE_REFUTES_UNKNOWN)}")
print(f"    {'-- total':20} {_m_green:5}")
_gate_greens = _m_by_kind.get("gate", 0)
print(f"  **the number that decides whether this check can bite: {_gate_greens} of "
      f"{_m_total} residue literal(s) name a GATE -- an instrument with a pin that can")
print(f"  contradict it.** A check whose green branch were carried by documents would "
      f"be decoration, and this is the count that would show it.")
print()
print(f"  distinct instruments carrying a green: {len(_m_inst)}")
for _k, _v in sorted(_m_inst.items(), key=lambda kv: -kv[1])[:12]:
    print(f"    {_v:5}  {_k}")
print()

# ---------------------------------------------------------------------------
# The two demotions, measured. Both were asked for as numbers before either was
# changed, and both are printed as numbers in every run so a reader can check
# the decision rather than take it.
# ---------------------------------------------------------------------------
_m_docs = [(n, r) for n, rows in MEASURED.items() for r in rows if r[2] == "names-document"]
_m_stems = [(n, r) for n, rows in MEASURED.items() for r in rows if r[2] == "stem-match"]
_m_doc_files = sorted({r[3] for _n, r in _m_docs})
_m_stem_amb = sum(1 for _n, r in _m_stems if "share that stem" in r[6])
print("  THE DOCUMENT DEMOTION. A markdown file is prose, and a sentence naming one is")
print("  prose agreeing with prose -- the relation this file refused when it refused the")
print("  producer search's green branch, and the relation an appendix row has when its")
print("  `instrument` cell holds a `.md`. A document contains the number *because a human")
print("  wrote it there*, so \"the document says 12.88\" and \"12.88 is right\" are one")
print("  proposition, not two. It is not `names-nothing` either: the sentence does name")
print("  something real, and burying 6 findings in a 600-strong class would make them")
print("  unactionable. So it is its own class, and its own check.")
print(f"    documents demoted out of green: {_m_document}   files: {_m_doc_files}")
for _n, _r in _m_docs:
    print(f"      {_n}:{_r[1]} {_r[0]!r} -> {_r[3]}")
print(f"    **what would change this decision:** a document that is generated by a gate")
print(f"    and cannot be hand-edited would be an OUTPUT of that gate rather than a")
print(f"    witness to it, and would be a carrier by construction. Nothing in this tree")
print(f"    is, and this file does not make an exception mechanism for a case it has not")
print(f"    seen -- an exception that is never exercised is a green that cannot go red,")
print(f"    one level down from the defect it was added to fix.")
print()
print("  THE STEM-FALLBACK DEMOTION, MEASURED. The resolver used to accept a bare name")
print("  that was not a path and match it against a stem index, taking the first hit. A")
print("  sentence writing `core.py` is a citation; a resolver accepting a bare `core`")
print("  because SOME file in the tree is called `core.py` is a coincidence, and where")
print("  two or more files share a stem the choice is `os.walk` order -- neither stable")
print("  across machines nor meaningful.")
print(f"    literals routed to a stem match: {_m_stem}"
      f"   of which the stem is shared by 2+ files: {_m_stem_amb}"
      f"   ({100 * _m_stem_amb // max(_m_stem, 1)}% of them)")
for _n, _r in _m_stems:
    print(f"      {_n}:{_r[1]} {_r[0]!r} -> {_r[3]}")
print(f"    **the rate is the argument.** It is not a rare tie-break that happens to be")
print(f"    usually right: on the sentences that used it, this fallback was wrong about")
print(f"    WHICH file every time, and it was handing out greens for it.")
print()
print("  THE COMMAND SPLIT. Not a typed allowlist -- a containment test on the path")
print("  `shutil.which` already returned. It is printed per command because a command")
print("  that resolves outside this tree is a statement about the machine the gate ran")
print("  on, and a reader is entitled to know which one they are looking at:")
_cmds = sorted({(r[3], r[4]) for rows in MEASURED.values() for r in rows
                if r[2] == "names-existing" and r[4].startswith("command/")})
_m_cmd_in = sum(1 for rows in MEASURED.values() for r in rows
                if r[2] == "names-existing" and r[4] == "command/in-repo")
_m_cmd_out = sum(1 for rows in MEASURED.values() for r in rows
                 if r[2] == "names-existing" and r[4] == "command/outside-repo")
for _path, _kind in _cmds:
    _cnt = sum(1 for rows in MEASURED.values() for r in rows
               if r[2] == "names-existing" and r[3] == _path)
    print(f"    {_kind:22} {_cnt:3}  {_path}")
print(f"    resolved INSIDE this repository: {_m_cmd_in}   outside it: {_m_cmd_out}")
print(f"    **on this machine not one command-green is a repository artefact**, and at")
print(f"    least one resolution has been observed to CHANGE between two runs of this")
print(f"    gate with no line of this repository changing in between. That is the whole")
print(f"    argument for printing the path at run time instead of trusting the class.")
print(f"    **This file names no path, and that is deliberate rather than incidental.**")
print(f"    A drive letter written into a tracked file prints identically on every")
print(f"    machine and reads as a specification, which is the exact opposite of what a")
print(f"    `PATH` hit is -- and it is a claim the run never consults, so it cannot go")
print(f"    red when the machine changes underneath it. The paths above are resolved")
print(f"    this run; the argument does not depend on which ones they were.")
print()

_cap_curve = []
for _cap in (1, 2, 3, 4, 6, 8):
    _refused = sum(1 for _rows in MEASURED.values() for r in _rows if r[5] > _cap)
    _cap_curve.append((_cap, _refused))
print(f"  the unit cap, and its price. MAX_UNIT_LINES = {MAX_UNIT_LINES}; a unit wider")
print("  than that gets no instrument credited, because a bullet with no terminal")
print("  punctuation yields a 'sentence' that is the whole bullet:")
for _cap, _ref in _cap_curve:
    print(f"    cap {_cap:>2} line(s): {_ref:4} of {_m_total} literal(s) refused a green"
          + ("   <- in force" if _cap == MAX_UNIT_LINES else ""))
print(f"  a 1:1+NEL reader aside, a unit that wide is a paragraph, and crediting a file")
print("  for a number somewhere inside a paragraph is the coincidence this replaced.")
print()

#: The red list, per class, so every finding is named rather than counted. The
#: counts above are the result; these are the evidence for it.
_m_find: dict[str, list[str]] = {c: [] for c in MEASURE_CLASSES}
for _n, _rows in MEASURED.items():
    for _v, _ln, _k, _inst, _kind, _span, _why in _rows:
        if _k in ("names-existing",):
            continue
        _m_find[_k].append(f"{_n}:{_ln} {_v} -> {_inst or '(none)'}")


def _sample(lst: list[str], head: str = "") -> str:
    if not lst:
        return "0"
    return f"{len(lst)}: " + (head + " " if head else "") + "; ".join(lst[:4])


check(
    not _m_absent and not _m_nothing and not _m_coarse and not _m_nounit,
    "of the four classes this check owns: every measurement either names an instrument "
    "that exists or names one that is not there, so a human can see it",
    f"{_m_total} residue literal(s) in {len(MEASURED)} file(s), spread over "
    f"{len(MEASURE_CLASSES)} classes. **This check owns the four existence classes "
    f"({', '.join(k for k in ('names-absent', 'names-nothing', 'unit-too-coarse', 'no-unit'))}) "
    f"and the other four have their own checks or are the green** -- "
    f"`names-unpinned-gate`, `names-document` and `stem-match` are the other three "
    f"reds, and folding them in here is exactly the "
    f"'collapsing the difference into green' the owner named, one level down. "
    f"{_m_green} name an instrument that exists (the instrument is printed beside each); "
    f"{_m_nothing} name nothing at all; {_m_absent} name something that is not in the tree "
    f"and not on this PATH; {_m_coarse} sit in a unit wider than {MAX_UNIT_LINES} line(s) "
    f"and are credited with no instrument; {_m_nounit} produced no unit at all. "
    f"**{_m_human} of {_m_total} are red and all {len(MEASURE_RED)} red classes are "
    f"counted here**, and that is the result, not a failure of the check: a number whose "
    f"sentence names nothing cannot be traced to anything runnable, which is the only "
    f"thing this gate can ask about. "
    f"This is not the appendix's `unrecorded` count re-reported -- a block would still be "
    f"required to hold a row per literal, and no block exists in any of the three files, so "
    f"the six row checks per file remain counted skips and the block half of this gate is "
    f"still unexercised. Named: {_sample(_m_find['names-nothing'])} name nothing; "
    f"{_sample(_m_find['names-absent'])} name something absent; "
    f"{_sample(_m_find['unit-too-coarse'], 'too coarse: ')} "
    f"{_sample(_m_find['no-unit'], 'no unit: ')}",
)

_m_unpinned_gates = sorted({r[3] for rows in MEASURED.values() for r in rows
                            if r[2] == "names-unpinned-gate"})
check(
    not _m_unpinned,
    "no measurement is attributed to a gate that pins nothing -- the class is named, "
    "not collapsed into the green",
    f"{_m_unpinned} residue literal(s) name a gate that defines check() and declares no "
    f"EXPECTED_CHECKS: {_m_unpinned_gates}. "
    f"This is red, and the reading behind it is the owner's to change: the defect is not "
    f"in the prose -- the sentence names a real file -- it is the same unpinned gate Tier "
    f"2 already reports by the other road, so counting it as a green here would be "
    f"'collapsing the difference into green' in the sense the owner meant, and folding it "
    f"into the red above would destroy the distinction the class exists to keep. It is "
    f"therefore its own check with its own count. The unpinned gates are {UNPINNED_GATES}, "
    f"and pinning one moves its measurements to the green branch and nothing else moves",
)

check(
    not _m_document,
    "no measurement is attributed to a markdown DOCUMENT -- a document is prose, and a "
    "sentence naming one is prose agreeing with prose",
    f"{_m_document} residue literal(s) name a document instead of an instrument: "
    f"{[f'{n}:{r[1]} {r[0]} -> {r[3]}' for n, r in _m_docs]}. The documents involved are "
    f"{_m_doc_files}. "
    f"**The decision, and its price:** a `.md` file contains this number because a human "
    f"typed it there, so \"the document says {_m_docs[0][1][0] if _m_docs else 'N'}\" and "
    f"\"{_m_docs[0][1][0] if _m_docs else 'N'} is right\" are one proposition, not two. "
    f"That is the relation this file refused when it refused the producer search's green "
    f"branch, and it is the same relation an appendix row has when its `instrument` cell "
    f"holds a `.md`. It was green here until this round, and demoting it is what moved the "
    f"green branch from 30 to {_m_green}. **What would change this decision:** a document "
    f"that is generated by a gate and cannot be hand-edited would be an OUTPUT of that "
    f"gate rather than a witness to it. No exception mechanism was added for a case that "
    f"does not exist, because an exception never exercised is a green that cannot go red "
    f"-- the same defect, one level down. This is not `names-nothing`: the sentence does "
    f"name something real, and {_m_document} findings buried in a {_m_nothing}-strong "
    f"class would be unactionable, which is a different kind of wrong",
)

check(
    not _m_stem,
    "no measurement is attributed to a file the sentence never named -- a stem match is a "
    "coincidence, and the tree says how many files it could have meant",
    f"{_m_stem} residue literal(s) were resolved through the stem index rather than "
    f"through a name the sentence wrote: "
    f"{[f'{n}:{r[1]} {r[0]} -> {r[3]}' for n, r in _m_stems]}. "
    f"**{_m_stem_amb} of {_m_stem} ({100 * _m_stem_amb // max(_m_stem, 1)}%) matched a stem "
    f"that 2 or more files in this tree share**, so which file the sentence meant was "
    f"decided by `os.walk` order. The resolver used to take the first hit and hand out a "
    f"green for it, which made this the largest single false-green generator in the file: "
    f"a sentence naming `volume` went green because *some* file in the tree is called "
    f"`volume.py`. A sentence writing `core.py` is a citation; accepting a bare `core` "
    f"because a file called `core.py` exists is not. What would change this decision: a "
    f"stem that exactly one file in the tree carries is still a coincidence, so uniqueness "
    f"is NOT the test -- the test is whether the sentence wrote a path, and only the "
    f"`exact` branch of the resolver can answer that",
)

#: **The command disclosure is checked, not merely printed.** A class with no check
#: is a class that reports forever without a verdict, which this file's own
#: `APPENDIX_CHECK_NAMES` comment calls out in the other direction. What is asserted
#: is the *disclosure*: every command-green has a resolved path on record and a
#: containment verdict derived from it. What is deliberately NOT asserted is that
#: the in-repo count is above zero -- that would be red on this machine and green on
#: a different one, which is a check whose verdict is a fact about the machine, and
#: a pin that moves with the machine is a pin that can be moved to hide a defect.
_m_cmd_greens = [(n, r) for n, rows in MEASURED.items() for r in rows
                 if r[2] == "names-existing" and r[4].startswith("command/")]
_m_cmd_undisclosed = [(n, r) for n, r in _m_cmd_greens
                      if not r[3] or command_class(r[3])[0] != r[4].split("/")[1]]
check(
    not _m_cmd_undisclosed,
    "every command-green records how it resolved and whether that resolution is inside "
    "this repository -- a `PATH` hit is a fact about the machine, and it is never silent",
    f"{len(_m_cmd_greens)} command-green(s): {_m_cmd_in} resolve inside this repository "
    f"and {_m_cmd_out} resolve outside it. {len(_m_cmd_undisclosed)} without a recorded "
    f"resolution. Paths: {[r[3] for _n, r in _m_cmd_greens]}. "
    f"**The containment test is derived, not a typed allowlist** -- it asks whether the "
    f"path `shutil.which` returned is under this repository, which is why a reader who "
    f"wanted 'a command this project ships' gets that question asked honestly instead of "
    f"answered from a list somebody typed. On this machine the answer is that not one of "
    f"them is a repository artefact, and at least one has resolved to a different binary "
    f"on a later run of this gate with no line of this tree changing in between. "
    f"**These greens cannot be made red by anything in "
    f"this repository** -- moving `PATH` reprints the path and changes no verdict. That "
    f"is why the class is printed per command and not summarised, and it is also why "
    f"**no path is written into this file**: the resolution is a fact about the machine "
    f"the gate ran on, and a machine fact frozen into a tracked source file would print "
    f"the same on every machine, look authoritative, and be a claim the run never "
    f"consults -- which is to say, a claim that cannot go red",
)

check(
    all(sum(_m_by_class[n].values()) == len(MEASURED[n]) for n in MEASURED),
    "every residue literal is booked into exactly one instrument class, so a class "
    "this reader quietly dropped is an arithmetic failure and not a smaller number",
    f"{sum(sum(c.values()) for c in _m_by_class.values())} booked against "
    f"{_m_total} residue literal(s) in {len(MEASURED)} file(s); classes are "
    f"{list(MEASURE_CLASSES)}. The last bucket, `no-unit`, exists for the failure this "
    f"check is for: a literal whose line produced no unit at all is a defect in this "
    f"reader, and it reads {sum(_m_by_class[n].get('no-unit', 0) for n in MEASURED)} times "
    f"here, so 'this reader looks at every residue literal' is a number and not a claim. "
    f"**{len(MEASURE_CLASSES)} classes where there were 6:** the two demotions added one "
    f"each, and this check is what proves neither of them swallowed a literal rather than "
    f"rehoming one",
)


# ---------------------------------------------------------------------------
# The appendix, and the proof that it cannot certify itself
# ---------------------------------------------------------------------------
def render_block(residue: list[tuple[str, int]], existing) -> list[str]:
    """The block this file would generate, from the residue outside it.

    The five stored columns come from the residue (`value`, `line`) and from the
    row that already claims that literal (`date`, `instrument`, `command`). The
    derived columns are printed, not stored, so a witness's mtime moving cannot
    make the stored text disagree with the tree.
    """
    out = [APPENDIX_BEGIN,
           "",
           "| value | line | date | instrument | command |",
           "|---|---|---|---|---|"]
    for value, lineno in residue:
        row = existing.get((value, str(lineno)), {})
        out.append("| " + " | ".join([
            value, str(lineno), row.get("date", ""), row.get("instrument", ""),
            row.get("command", ""),
        ]) + " |")
    out += ["", APPENDIX_END]
    return out


def _epoch(iso: str) -> float:
    """A date as a timestamp, or 0.0 for an absent one.

    0.0 rather than `inf`: a row with no date is already red for having no date,
    and reporting it *also* as a stale witness would be a second red for one
    defect, which makes the failure count stop meaning what it says.
    """
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})$", iso.strip())
    if not m:
        return 0.0
    y, mo, d = (int(g) for g in m.groups())
    try:
        return time.mktime(time.strptime(f"{y:04d}-{mo:02d}-{d:02d}", "%Y-%m-%d"))
    except (ValueError, OverflowError):
        return 0.0


DERIVED: dict[str, list[tuple[tuple[str, int], dict[str, str]]]] = {}
#: Every finding this file can raise about one file's appendix, so a check cannot
#: be added without somewhere for its result to land.
FINDING_KEYS = ("mismatch", "unrecorded", "phantom", "no_date", "no_instrument",
                "missing_witness", "stale_witness", "duplicate_pairs")
#: Per file, because the verdict now depends on whether *that* file has a block.
#: One file carrying an appendix must not decide the fate of another that has
#: none, and the symmetric version of this gate could not tell the two apart.
FINDINGS: dict[str, dict[str, list[str]]] = {}
BLOCK_PRESENT: dict[str, bool] = {}
class_tally: dict[str, int] = {}

for _name in PROSE:
    _p = ROOT / _name
    if not _p.is_file():
        BLOCK_PRESENT[_name] = False
        FINDINGS[_name] = {k: [] for k in FINDING_KEYS}
        continue
    _lines = read(_p).split("\n")
    _span = appendix_span(_lines)
    BLOCK_PRESENT[_name] = _span is not None
    _f = FINDINGS[_name] = {k: [] for k in FINDING_KEYS}
    _existing, _junk = parse_block(_lines, _span)
    for _j in _junk:
        # A row that is not five cells cannot be read, so it cannot be said to
        # have a date, an instrument and a command. It is named as a missing
        # one rather than dropped, which is the difference between a check that
        # reports its own absence and one that quietly stops looking.
        _f["no_date"].append(f"{_name} (malformed row, {_j})")
    _res = RESIDUE[_name]
    _keys = {(v, str(ln)) for v, ln in _res}
    # **The unrecorded count used to under-report, and the asymmetry makes that
    # matter more than it did.** The row key is `(value, line)`, so two literals
    # that share a value and a line -- `2 x 2` on one line, or `39 x 26 x 41` if
    # a value repeated -- collapse into one key, and the number named as
    # "unrecorded" was the number of *distinct pairs*, not the number of
    # literals. The new policy has to name this count exactly, because the count
    # is the argument for not going red. So both are counted and printed: the
    # literals, the distinct pairs they collapse into, and the difference. The
    # block format is the limitation, not the document: a markdown row keyed on
    # `(value, line)` cannot express two occurrences, so those literals are
    # guarded by one row and this file says so rather than quietly counting the
    # smaller number.
    _dupes = len(_res) - len(_keys)
    _f["duplicate_pairs"] = [f"{_name}: {len(_res)} literal(s) in {len(_keys)} distinct "
                             f"(value, line) pair(s); {_dupes} occurrence(s) share a key with "
                             f"another and are guarded by one row"]
    for key in sorted(_keys - set(_existing)):
        _f["unrecorded"].append(f"{_name}:{key[1]} {key[0]}")
    for key in sorted(set(_existing) - _keys):
        _f["phantom"].append(f"{_name}:{key[1]} {key[0]}")
    # **The regeneration comparison is scoped to a block whose key set already
    # agrees, so one defect is one red.** An incomplete block mismatches the
    # regeneration *because* it is incomplete, so running both checks on it
    # reported a missing row as two failures -- and this file's own `_epoch`
    # docstring says why that is wrong: "a second red for one defect makes the
    # failure count stop meaning what it says". What is left for this check is
    # the defect only it can see: a block whose keys are all correct and whose
    # *cells* are not, which is an edited date, an edited instrument, or a value
    # that was changed where the same value also appears on another line.
    if _span and not _f["unrecorded"] and not _f["phantom"] \
            and render_block(_res, _existing) != _lines[_span[0]:_span[1] + 1]:
        _f["mismatch"].append(_name)
    rows = []
    for key, row in sorted(_existing.items(), key=lambda kv: int(kv[0][1])):
        date = row.get("date", "").strip()
        inst = row.get("instrument", "").strip()
        cmd = row.get("command", "").strip()
        if not date:
            _f["no_date"].append(f"{_name}:{key[1]}")
        if not inst:
            _f["no_instrument"].append(f"{_name}:{key[1]}")
        if not cmd:
            _f["no_date"].append(f"{_name}:{key[1]} (command)")
        wit, covers, klass = derive_witness(inst)
        class_tally[klass] = class_tally.get(klass, 0) + 1
        mt = mtime_of(wit)
        state = "no-witness" if mt is None else ("current" if mt > _epoch(date) else "stale")
        if klass == "missing":
            _f["missing_witness"].append(f"{_name}:{key[1]} -> {inst}")
        if state == "stale":
            _f["stale_witness"].append(
                f"{_name}:{key[1]} witness {wit} older than {date or '(no date)'}")
        rows.append((key, {"witness": wit, "covers": covers, "class": klass, "state": state}))
    DERIVED[_name] = rows


if ARGS.emit:
    print("\n" + "=" * 72)
    print("APPENDIX BLOCKS -- paste into the file named above each one.")
    print("derived columns are printed, not stored:")
    print("=" * 72)
    for _name in PROSE:
        if _name not in RESIDUE:
            continue
        _p = ROOT / _name
        _lines = read(_p).split("\n")
        print(f"\n--- {_name} ---")
        print("  value            line     witness                       covers  class          state")
        for key, d in DERIVED[_name]:
            print(f"  {key[0]:<15} {key[1]:<8} {d['witness']:<30} {d['covers']:<7} "
                  f"{d['class']:<14} {d['state']}")
        print()
        for ln in render_block(RESIDUE[_name], parse_block(_lines, appendix_span(_lines))[0]):
            print(ln)

print(f"\n{'=' * 72}")
section("Tier 3: the appendix, and the proof it cannot certify itself")

#: **A correction to this file's own earlier claim, printed unconditionally.**
#: It used to live only in the docstring and in a `check()` detail string, which
#: is not the same thing as being visible: the detail is printed only when that
#: check runs, and every check below it is a **skip** on a tree with no block --
#: which is every tree today. So the correction was true and invisible, and a
#: reader arriving from a git blame or a cached report would have found the old
#: claim stated plainly in the docstring with nothing beside it saying it had been
#: withdrawn. It is printed here, above the section it qualifies, for the same
#: reason the residue is printed rather than summarised: a claim about what this
#: file can and cannot verify is worth exactly as much as its visibility.
print("\n  CORRECTION, to a claim an earlier version of this file made about its own")
print("  appendix check. The old claim was: \"editing a number inside the block makes")
print("  the comparison mismatch.\" That was true of the two KEY columns only --")
print("  `value` and `line` -- and false of the cells. `render_block` copies `date`,")
print("  `instrument` and `command` out of the very rows it is comparing, so a cell")
print("  edited in the document regenerates identically and the comparison is BLIND")
print("  TO IT BY CONSTRUCTION. The check verifies the block's structure and key set")
print("  -- markers, header, row count, row order -- and each cell is covered")
print("  instead by the separate check that owns it. A figure whose name overstated")
print("  what was measured is the defect this file is in the business of catching,")
print("  including when the figure is one of its own checks.")

print("\n  which prose files carry an appendix block:")
for _n in PROSE:
    _present = BLOCK_PRESENT.get(_n)
    _f = FINDINGS.get(_n, {})
    _lit = len(RESIDUE.get(_n, []))
    _unrec = len(_f.get("unrecorded", []))
    _dup = _f.get("duplicate_pairs", [])
    print(f"    {_n:14} {'block present' if _present else 'NO BLOCK':13}  "
          f"{_unrec:4} of {_lit:4} residue literal(s) unrecorded"
          f"{'' if _present else '  <- counted, named, not a failure'}")
    for _d in _dup:
        print(f"    {'':14} key limitation: {_d}")
print()
print("  witness classes, by count (a class, not a verdict):")
for k in sorted(class_tally):
    print(f"    {k:14} {class_tally[k]:4}")
print()

# ---------------------------------------------------------------------------
# The asymmetry, in code: no block -> counted skips; a block with a gap -> red
# ---------------------------------------------------------------------------
#: What each of the six row checks reports, and how it says so. Kept as one table
#: so the *order* of `APPENDIX_CHECK_NAMES` and the order of the emit loop below
#: cannot drift apart: they are the same six questions in the same order, and a
#: name that is checked but never asked is a name that reports green forever.
_APPENDIX_CLAIMS = (
    ("mismatch",
     lambda f, n: f"{len(f['mismatch'])} block(s) differ from the regeneration: "
     f"{f['mismatch']}. The exclusion is not a typed line span that could be widened to "
     f"hide a number -- the block is regenerated from the residue outside it and compared, "
     f"so the markers, the header, the row count and the row order are all verified. **What "
     f"it cannot verify is a cell's content, and this file no longer claims it can.** The "
     f"cells are the regeneration's *input*: `render_block` copies `date`, `instrument` and "
     f"`command` out of the very rows it is comparing, so a cell that was edited in the "
     f"document regenerates identically and this check is blind to it by construction. That "
     f"was the old claim here -- \"editing a number inside the block makes the comparison "
     f"mismatch\" -- and it was true only of the two key columns. The cells are covered by "
     f"the checks that own them: a blank one by the date/instrument check, a witness that is "
     f"not on disk by the witness check, a witness older than its date by the staleness "
     f"check. A key that no longer matches the tokenizer is named from both sides by the two "
     f"row checks. Measured: rows reordered with every key and every cell correct is caught "
     f"here; a cell edited in place is not"),
    ("unrecorded",
     lambda f, n: f"{len(f['unrecorded'])} unrecorded of {len(RESIDUE.get(n, []))} residue "
     f"literal(s) in this file. A Tier 3 literal with no row is a number in the prose that "
     f"nobody can say where it came from"
     + (f"; first: {f['unrecorded'][:5]}" if f["unrecorded"] else "")),
    ("phantom",
     lambda f, n: f"{len(f['phantom'])} row(s) name a value/line pair the tokenizer did not "
     f"find: {f['phantom']}. This is the other direction of the same two-sided rule, and it "
     f"is what catches a row whose value was edited in the document"),
    ("no_date",
     lambda f, n: f"{len(f['no_date'])} missing date or command, "
     f"{len(f['no_instrument'])} missing instrument. A row with a value and nothing else is a "
     f"transcription"),
    ("missing_witness",
     lambda f, n: f"{len(f['missing_witness'])} row(s) name a witness that is not on disk: "
     f"{f['missing_witness']}. A row may declare `none`, which is a legitimate answer for a "
     f"number that depends on a machine, a GPU, a network or a wall clock -- no witness "
     f"exists for that class, and saying so is more useful than a column of confident "
     f"`current`"),
    ("stale_witness",
     lambda f, n: f"{len(f['stale_witness'])} stale: {f['stale_witness']}. The state column is "
     f"a comparison at run time and is never written into the document, so it cannot be stale "
     f"-- only the witness can be, and that is what this red is for"),
)

for _name in PROSE:
    _present = BLOCK_PRESENT.get(_name, False)
    _f = FINDINGS.get(_name, {})
    _n_res = len(RESIDUE.get(_name, []))
    if not _present:
        _why = (
            f"no appendix block in {_name}: the block does not exist, so nothing in this file "
            f"is refutable yet, and its {_n_res} residue literal(s) are counted and named as "
            f"unrecorded rather than failed"
            if (ROOT / _name).is_file() else
            f"{_name} is not present under {ROOT}, so there is nothing in it to record"
        )
        print(f"  {_name}: {_why}")
        for _i, _claim in enumerate(_APPENDIX_CLAIMS):
            skip(f"{_name}: {APPENDIX_CHECK_NAMES[_i]}", _why)
        continue
    print(f"  {_name}: block present, {len(_f['unrecorded'])} unrecorded of {_n_res} "
          f"-- these six are asked, not skipped")
    for (_key, _claim), _name_of_check in zip(_APPENDIX_CLAIMS, APPENDIX_CHECK_NAMES):
        check(not _f[_key], f"{_name}: {_name_of_check}", _claim(_f, _name))


# ==========================================================================
section("this instrument reports on its own output")

check(
    not LINE_FAULTS,
    "every file:line this run printed resolves through a SECOND, independent read "
    "of the same bytes",
    f"{len(LINE_PROBES)} location(s) checked against `read_bytes().split(b'\\n')` -- a read "
    f"that shares no code with the text reader, runs no newline translation and decodes "
    f"nothing. {len(LINE_FAULTS)} disagree"
    + (f": {LINE_FAULTS[:4]}" if LINE_FAULTS else "")
    + ". This file prints a line number for every residue literal and every figure, and a "
      "line number that does not resolve sends a reader to the wrong line without telling "
      "them, so the instrument has to be able to report its own error"
    + (
        ". **What it found on the real tree, and it corrects the earlier claim.** The `0x85` "
        f"bytes are never NEL: every one is the second byte of a UTF-8 multi-byte sequence "
        f"(the U+00C5 angstrom sign, twice-encoded, in the tree's mojibaked angstroms), and "
        f"the text and byte readers agree exactly"
        f" ({', '.join(f'{n} {len(byte_lines(ROOT / n))}' for n in PROSE if (ROOT / n).is_file())} lines). "
        f"A reader that maps bytes 1:1 and treats `U+0085` as a break would shift line "
        f"numbers by up to "
        + ", ".join(f"{n} {NEL_SHIFT[n]}" for n in PROSE if n in NEL_SHIFT)
        + " lines -- a cp936 (GBK) read, which is what a GBK console does, cannot decode "
          "these files at all and raises rather than answering. That shift is reported as a "
          "named number and is not this file's verdict: the encoding defect belongs to the "
          "gate that owns encoding, and a permanent red here would be the noise this file "
          "exists to avoid"
        if NEL_SHIFT else ""),
)

print("\n  line numbers, under a reader that mistakes a UTF-8 continuation byte for NEL:")
for _n in PROSE:
    if _n in NEL_SHIFT:
        print(f"    {_n:14} max shift {NEL_SHIFT[_n]:+5} line(s)   "
              f"({len(byte_lines(ROOT / _n))} real line(s))")
print()

# ==========================================================================
section("this file holds itself to the same contract")

def declared_in_inventory() -> tuple[bool, str]:
    """Is this file listed in `check_scripts_declare.py`'s `INVENTORY`? Read, not asserted.

    **This claim used to be typed into the check below and it went stale.** The
    detail text said `provenance_appendix.py` "is not yet in INVENTORY ... so that
    gate's untracked-pin check is red on this file until the name is added". The
    name *was* added -- it is at `INVENTORY[23]`, `check_scripts_declare.py:216` --
    and the sentence kept being printed as fact on every run. That is the exact
    defect this file is in the business of catching, in this file's own output, and
    a prose claim about another file's table is the one kind of claim that can be
    checked here, so it is now read at run time.

    Read as text rather than imported, because `check_scripts_declare.py` is a
    script that exits at the end and is currently mid-edit by its owner; importing
    it would make this file's own result depend on someone else's half-finished
    edit. The `INVENTORY = [` ... `]` span is located with a non-greedy match and
    the name is looked for inside it, so a mention in a comment or a docstring
    elsewhere in that file does not count as membership.
    """
    auditor = SCRIPTS / "check_scripts_declare.py"
    try:
        text = read(auditor)
    except OSError as exc:
        return False, f"the auditor could not be read ({exc.__class__.__name__})"
    m = re.search(r"^INVENTORY\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    if not m:
        return False, "no `INVENTORY = [` block found in it"
    for i, line in enumerate(m.group(1).split("\n"), start=text[:m.start(1)].count("\n") + 1):
        if re.search(rf"""["']{re.escape(Path(__file__).name)}["']""", line):
            return True, f"{auditor.name}:{i}"
    return False, "it is not listed there"


_self_pin = module_constant(Path(__file__).resolve(), "EXPECTED_CHECKS")
_self_in_inventory, _self_in_where = declared_in_inventory()
_inv_state = (f"lists it at {_self_in_where}" if _self_in_inventory
              else f"does not list it ({_self_in_where})")
check(
    _self_pin == EXPECTED_CHECKS and isinstance(_self_pin, int),
    "provenance_appendix.py pins itself, with the pin read back out of its own tree",
    f"the module-level constant says {_self_pin!r}. Whether this file is tracked by "
    f"`check_scripts_declare.py` is **read at run time, not asserted here**: its "
    f"INVENTORY {_inv_state}. "
    f"The previous version of this sentence said the opposite as a standing fact and "
    f"was wrong -- the name had been added and the sentence kept printing, which is "
    f"this file's own subject matter happening to it. A claim about another file's "
    f"table is the one claim here that can be checked, so it is checked",
)

# ==========================================================================
# This file holds itself to the same contract
# ==========================================================================
#: The absolute-path shape, **lifted out of `check_repo_docs.py` by AST rather
#: than retyped here.** That gate already owns the definition and keeps a
#: self-test for it; a second copy in this file would be a second thing to keep
#: in step with the first, which is the same argument that keeps Tier 1 out of
#: this file entirely. The gate is not imported -- it is a script that exits at
#: the end and belongs to its owner -- so the two `Assign` nodes are compiled
#: and run here, which is the auditor pattern already used above.
def _abs_path_pattern():
    auditor = SCRIPTS / "check_repo_docs.py"
    try:
        tree = ast.parse(read(auditor))
    except (OSError, SyntaxError) as exc:
        return None, f"could not read {auditor.name} ({exc.__class__.__name__})"
    wanted = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in ("ABS_PATH", "DRIVE_ESCAPE"):
                wanted[t.id] = node
    missing = {"ABS_PATH", "DRIVE_ESCAPE"} - set(wanted)
    if missing:
        return None, f"{auditor.name} no longer defines {sorted(missing)}"
    ns = {"re": re}
    for name in ("DRIVE_ESCAPE", "ABS_PATH"):
        try:
            value = eval(compile(ast.Expression(wanted[name].value), auditor.name, "eval"), ns)
        except Exception as exc:  # a gate that will not compile must not crash this one
            return None, f"{name} did not evaluate ({exc.__class__.__name__}: {exc})"
        #: **`eval` does not bind the name it just evaluated.** The first
        #: expression is a bare string literal, so evaluating it returns the
        #: value and leaves the namespace exactly as it was -- and the second
        #: expression is an f-string that interpolates `DRIVE_ESCAPE` at
        #: runtime, so it raised `NameError` on a name that was demonstrably
        #: two lines above. The assignment back into `ns` is the whole fix, and
        #: the reason it is written out rather than done with `exec` is that
        #: `exec` would also have run any *other* statement in that node.
        ns[name] = value
    return ns["ABS_PATH"], f"lifted from {auditor.name} by AST, not retyped here"


#: The four damage signatures this file's own history has actually produced.
#: `check_text_encoding.py` is the gate that owns encoding for the whole tree and
#: runs in CI; this is not a second copy of it. It is the narrower promise that
#: **the file this run is executing has not itself been mangled on the way in**,
#: which is a different question from whether the tree is clean, and the one
#: whose blind spot bit last round: a file that decodes cleanly, carries no BOM
#: and has correct line endings can still have lost a byte inside a character.
_LOST_BYTE = re.compile("[一-鿿]\\?")


def own_source_findings() -> tuple[list[str], str]:
    """Findings against this file's own bytes, and where the pattern came from.

    **Scoped to code AND prose on purpose.** `check_repo_docs.py` masks
    docstrings and comments before it looks for a path, which is right for its
    purpose -- a gate that spells out the patterns it hunts for would trip on
    itself. It is the wrong scope for *this* question, because the three stale
    machine paths that were actually in this file were all in a docstring or a
    comment, where that gate could not see them. Silence from a gate is not a
    verdict on an input the gate never looked at.
    """
    data = Path(__file__).resolve().read_bytes()
    out: list[str] = []
    if data.startswith(b"\xef\xbb\xbf"):
        out.append("carries a UTF-8 BOM")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"does not decode as UTF-8: {exc}"], "decode step"
    #: **The replacement character is built with `chr()`, never written as a
    #: literal -- and that is not a style rule, it is this check catching its
    #: author.** The first version of this line was a literal U+FFFD sitting in
    #: this file, and the check went red on the file that contains it. A gate
    #: that hunts for a character cannot spell that character in its own source.
    repl = chr(0xfffd)
    if repl in text:
        out.append(f"carries {text.count(repl)} U+FFFD replacement character(s)")
    lost = _LOST_BYTE.findall(text)
    if lost:
        out.append(f"carries {len(lost)} CJK-then-`?` lost-byte sequence(s), "
                   f"first at offset {text.find(lost[0])}")
    pattern, origin = _abs_path_pattern()
    if pattern is None:
        out.append(f"could not obtain the absolute-path pattern: {origin}")
    else:
        hits = [f"line {n}: {m.group(0)!r}"
                for n, line in enumerate(text.split("\n"), 1)
                for m in pattern.finditer(line)]
        if hits:
            out.append(f"carries {len(hits)} machine-specific absolute path(s) "
                       f"-- {hits[:4]}")
    return out, origin


_own_findings, _own_origin = own_source_findings()
check(
    not _own_findings,
    "this file's own bytes are clean: no BOM, no U+FFFD, no lost byte, and no "
    "machine-specific path in code OR in prose",
    f"{len(_own_findings)} finding(s) in {Path(__file__).name} "
    f"({len(Path(__file__).resolve().read_bytes())} bytes). The absolute-path shape is "
    f"{_own_origin}, so this check and `check_repo_docs.py` cannot disagree about what "
    f"a machine path is. **The scope is the whole file, unmasked, and that is the "
    f"point:** the stale paths this check exists to catch were all in a docstring or a "
    f"comment, where a gate that masks prose is silent by design. A gate that cannot "
    f"see an input is not a verdict on it. `check_text_encoding.py` still owns encoding "
    f"for the whole tree and runs in CI; this is the narrower promise that the file "
    f"executing this run was not mangled on its way in, which is the failure that a "
    f"decode-clean, BOM-free, LF-only file can still hide"
    + (f". Named: {_own_findings}" if _own_findings else ""),
)

_before = CHECKS + COUNTED_SKIPS
check(
    _before + 1 == EXPECTED_CHECKS,
    "the tally accounts for every slot, counted skips included -- so the pin is the "
    "same number whether or not a block exists",    f"{CHECKS} check(s) ran and {COUNTED_SKIPS} counted skip(s) occupy the rest of the "
    f"{EXPECTED_CHECKS} pinned slots; {len(SKIPPED) - COUNTED_SKIPS} further skip(s) are "
    f"recorded and named but are not one of this file's questions. The +1 is this check. "
    f"**A pin that only counts passes cannot carry a skip policy**, because the day a "
    f"block appears the six row checks per file would have to be added to the pin to keep "
    f"the arithmetic honest -- and a pin that moves the moment a document changes is a pin "
    f"that can be moved to hide a defect",
)

print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed, {COUNTED_SKIPS} counted skip(s), "
      f"{CHECKS + COUNTED_SKIPS}/{EXPECTED_CHECKS} slots accounted for")
for f in FAILURES:
    _emit(f"  FAILED: {f}")
if SKIPPED:
    _emit(f"\n  {len(SKIPPED)} question(s) were skipped, not answered. A skip is a result: "
          f"it is named, counted inside the same pinned total, and this run still exits 0.")
    for _n, _r in SKIPPED:
        _emit(f"    SKIP {_n}: {_r}")
else:
    _emit("\n  0 skipped: every question this file asked, it asked here.")

sys.exit(1 if FAILURES else 0)
