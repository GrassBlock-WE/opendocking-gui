"""Do the two scoring backends agree, and does the product say which one answered?

Run:  python scripts/gpu_cpu_parity_check.py

**What this file is, and what it is not.**

It is a *parity* gate. The repository ships two scoring backends -- the CPU
interpolator, which is always compiled in, and a `wgpu`/WGSL compute kernel
behind the optional `gpu` feature.

**What the READMEs say about the second, and the date they stopped saying
otherwise.** They now say **not on the search hot path**: an "opt-in GPU
batch-scoring path -- which is not on the search hot path", whose "only caller
is the batch entry point `evaluate_conformations`", with `use_gpu` defaulting
to False so that runs stay comparable across machines (`README.en.md` 16-17 and
453-456, `README.md` 11-12 and 387-389). Until 2026-10-02 both said the
opposite -- the largest claim either made about the kernel was that it was "a
working GPU batch-scoring path" / "actually on the hot path" -- and that claim
had never been compared against anything. The correction is the claim *this*
file measured, and it is dated here because a gate that quotes its own
self-description has to be able to say which tense that description was in.

The reason a claim of that shape needed measuring at all is that it is not the
kind of statement a reader can check. Both READMEs also carried a
`report_backend` fallback reason, and a GPU run and a CPU run were always
distinguishable from their reports; what actually lies is a build with no
`gpu` feature, where `use_gpu=True` falls back silently and gives no reason.
So the claim was not wrong in the arithmetic, it was wrong about *where the
arithmetic happens* -- and nothing short of running both backends on the same
conformations would have said so.

The CI matrix's `--features gpu` leg (`.github/workflows/ci.yml`) compiles the
feature and runs `cargo test` on a runner with no adapter, so the engine's own
parity fixture in `dock-core/src/gpu/mod.rs` takes its no-adapter branch and
returns. The engine's fixture is the right place to *measure* the kernel and
this file does not duplicate that; what it cannot be is run by CI on the
platform that ships. This file can.

**The mirror carries the same sentence in the same tense, and that is a
reading rather than a guarantee.** `opendocking-gui/README.en.md` 16-17 reads
"an opt-in GPU batch-scoring path -- which is not on the search hot path" and
`opendocking-gui/README.md` 11-12 says the same in Chinese, so a reader who
checks the claim in the mirror now gets the answer this paragraph gives instead
of the pre-2026-10-02 wording: 296-299 there is a table header about pose
trust, not a `report_backend` instruction. This paragraph used to claim the
opposite, and it was wrong in the direction that matters, because a reader who
was told the mirror still said "actually on the hot path" would have had no
reason to open it and would have carried the stale claim forward. It is still
not a file this gate audits: the mirror is written by a one-way sync this file
does not run, so "in the same tense today" is a reading with a date on it and
not a property, and the only thing that can put the old words back is a
re-sync from a tree that still has them.

**What it establishes on a build that cannot reach the kernel, and what it
cannot.**

Seven of the nine results it records need no adapter, and all seven are about
*reporting* rather than arithmetic -- which is deliberate, because reporting is
the part that goes wrong silently. Two of them are the whole point of the GPU
question: whether the `gpu` line in `available_backends()` is a probe or a
constant, and whether a caller who asks for the GPU is ever told why they did
not get one. The last two results need the kernel, and this file keys that on
whether the adapter probe ever ran rather than on what hardware is in the case,
so a build without the `gpu` feature skips them **with the skip recorded and
counted** even on a machine that has a card in it, and a reader can tell the
difference between "the two backends agree" and "nobody looked".

**The skip convention, stated so it cannot be misread.**

Every block produces exactly one result: PASS, FAIL or SKIP. A skip is a result
-- printed, counted, and part of the pinned total -- because "could not measure
this" has to be an answer someone can give. But the summary prints passes,
failures and skips as **three numbers that are never added**, and
`EXPECTED_CHECKS` is 9 on every machine. So `7 passed, 0 failed, 2 skipped` reads
as seven things established and two not established, and never as "9 checks
passed". The check that stops the opposite mistake reaching the summary is
`the_pinned_total_counts_results_and_not_only_passes`.

**Why the numeric band is derived rather than written down.**

An `f32` GPU reduction over a grid cannot be bit-identical to an `f64` CPU sum,
so the question is not "are they equal" but "is the difference the size of
accumulated rounding, or the size of a bug". The band is therefore built from
the quantity the rounding error is actually proportional to -- the magnitude of
the `f32` grid values the kernel reads -- and every factor in it is read out of
the maps this run built or counts something structural. The one structural fact
it leans on is that a map weight is exactly `0.0` or `1.0` and at most two
slots are set per atom kind (`weights_for_kind`, `dock-core/src/scoring.rs`),
which bounds a node reading by `2 * max|grid|`. No weight table is transcribed
here: the earlier attempt at such a band was proportional to the *total*, which
the engine's own fixture rejected, because a conformation whose neighbouring
nodes nearly cancel is evaluated from nodes orders of magnitude larger than its
own total -- a total-proportional band collapses exactly where the kernel is
hardest to get right.

**One block was red on a build without the `gpu` feature, and is not any more.**

* `a batch the gpu cannot take is declined with a reason, not silently` failed on
  a build without the `gpu` feature, because `evaluate_population` dropped
  `prefer_gpu` in its `#[cfg(not(feature = "gpu"))]` arm and returned no skip
  reason, so the report carried `backend: "cpu"` and no reason at all. The
  energies were still right and the backend was still named, so it was a missing
  reason rather than a wrong number -- and the READMEs said so in those words
  before the arm was repaired (`README.md` 415-417 and `README.en.md` 487-490,
  both of which are now in the past tense and say so). A reader who followed
  that failure to the prose would have edited correct documentation; the defect
  was the `#[cfg]` arm in `dock-core/src`, and this block is the reason it is
  fixed.

  **The arm is fixed in the source and in the binary this gate imports.** The
  `#[cfg(not(feature = "gpu"))]` arm at `dock-core/src/search/mod.rs:439-459`
  returns `Some(GpuSkip { reason: "this build was compiled without the gpu
  feature" })` when `prefer_gpu` is set and `None` when it is not -- it used to
  read `let _ = prefer_gpu;` and drop the request. The binary that implemented
  the old arm is gone from this machine, so the block above is green and the
  question is no longer "is it red" but "which build said so".

  **This file no longer transcribes that build, because it was wrong within a
  day of being written.** It used to carry a table of three timestamps and one
  run's split, which is a claim about a binary typed into a file that outlives
  the binary: it named an install of 778,240 B / `066fdd0c` dated 2026-10-02
  09:59, and a `mod.rs` of 2026-10-02 13:23, and none of the three is what is on
  disk now. So every run prints the size and the digest of the extension its
  import resolved to, in the `which binary this run measured` block, and a
  recorded number in this file is tied to a build by the reader who runs it
  rather than by the reader who remembers. A number in this docstring that is
  not beside a digest is one of the dated measurements below, and those name
  their date and the shape of the build they came from instead.

  **Any number here that came from a run is a number about the build that was
  loaded when it was run**; the pins, the call-site census and the declaration
  are not, because none of them was read off a build.

A gate that cannot fail is worse than no gate, so this one is not papered over.

**The out-of-box block *was* red on every machine with an adapter, and is not any
more -- so its failure text is written against the current claim, not the old
one.**

`a row that leaves the box is scored by the same function on both backends` is
the block that found the second of those defects, and it is described here
because *how a gate's prose rots* is the subject. The kernel used to return a
flat `1000.0` for an atom outside the tabulated volume
(`dock-core/src/gpu/energy.wgsl`) while the CPU added `OUT_OF_BOX_PENALTY` per
angstrom of violation. The GPU's term was then capped at `1000 x protruding
atoms` while the CPU's was unbounded, the two differed by thousands of kcal/mol,
and the ranking of poses outside the box inverted. Nothing reported it: the
`report_backend` calls that ran all said `backend: "gpu"`.

The host now evaluates `max(p - bmax, min - p)` per axis and sums it in `f64`
(`out_of_box_violation_per_axis(...).iter().sum()`, `gpu/mod.rs`), and the
kernel multiplies that one scalar by `OUT_OF_BOX_PENALTY` -- the same
per-axis, per-angstrom function the CPU evaluates, just evaluated on the side
that has the `f64` copy of the coordinate. That move is also why the shader no
longer contains the per-axis `max`: a sentence here used to say the shader
itself "evaluates ... per axis, sums it", and that was the sentence that rotted
while the arithmetic stayed right. `shader_and_rust_agree_on_the_out_of_box_penalty`
(`dock-core/src/gpu/mod.rs`) binds the two transcriptions of the constant
together and pins the ramp's *shape*, which a constant-only assertion could not
tell apart from a flat step.

Measured 2026-10-02 on an RTX 3050 Laptop GPU, from a `--features gpu` build:
512 seeded conformations, `vina`, a 22 A box at 0.5 A spacing. **9 passed,
0 failed, 0 skipped.** Worst `|GPU-CPU|` is 3.772e-05 kcal/mol inside the box
(0.056% of the derived band, 427 rows) and 1.890e-03 kcal/mol outside it (2.79%
of the band, 85 rows). The ranking is identical over all 512 rows -- 0 of
130,816 pairs ordered differently, so nothing inverts anywhere. The slide that
used to split the backends by thousands now tracks: 8 A along x gives
7882.2637 on the CPU against 7882.2650 on the GPU. `dock-core/src` was edited
underneath that measurement, so the run was repeated against a rebuild that
picked the edits up; every number above is identical on both builds, which is
the only reason they are quoted without a revision attached.

**How far this paragraph is still true, as of its own date.** These numbers came
from a `--features gpu` build made at the time, not from the non-gpu extension
the rest of this file talks about, so that extension does not invalidate them.
The tree has moved underneath them, and this file no longer claims to know by
how much: `dock-core/` and `dock-py/` have both been edited since, and the
extension on disk has been rebuilt twice since, so the sentence this paragraph
used to close with -- naming the size, the digest and a missing export of the
binary "currently on disk" -- was a claim about a build that no longer exists,
and it named one that *does* export `DockingResult.unknown_atom_types` as the
one that does not. Read this as "this is what the two backends did on an RTX
3050 on that date", which is what a dated measurement is for. Re-running it is
the only way to turn it back into a present-tense claim, and a run now prints
the digest of whatever it loaded, so the reader can say which build they are
looking at.

**That paragraph is the hazard, and it is why the blocks are keyed on the
disagreement rather than on it.** A gate whose description of the product is out
of date eventually asserts the old behaviour, and a repair to the defect then
leaves the gate red -- which is how a whole team learns to read red as "known".
So: the eight blocks that never depended on the flat sentinel are unchanged by
its repair, and each is mutation-tested here in both directions. The ninth -- the
out-of-box block -- is the only predicate whose original failure mode has been
removed, which makes it the only one whose *new* failure mode is worth
re-proving: it still fires, on a disagreement, but a flat step and a drifted
penalty constant are now the two things to tell apart, and the slide below is
what tells them apart.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# The same receptor, ligand, box and scoring function `energy_terms_check.py`
# docks, so the two gates agree on what a real case is here rather than each
# inventing one. The spacing is 0.5 A rather than that gate's 0.375 A because
# this file is about arithmetic on the grid and a finer grid only makes it
# slower.
RECEPTOR = "1crn_prep.pdbqt"
LIGAND = "biotin_prep.pdbqt"
BOX_CENTER = (18.0, 12.0, 20.0)
BOX_SIZE = (22.0, 22.0, 22.0)
SPACING = 0.5
SCORING = "vina"

#: Fixed, so the population -- and therefore the worst-case difference -- is a
#: property of this file rather than of the run.
SEED = 20260901
N_CONF = 512

#: The kernel reduces one conformation per workgroup and the workgroup size is
#: 64, so a ligand with more atoms than this cannot be batched at all. The gate
#: provokes that limit rather than writing the number down a second time.
WORKGROUP_SIZE = 64

#: 8 corners x 2 map slots. Both are counts of something, and both are stated
#: where they are used so they cannot drift from the thing they count.
CORNERS_PER_ATOM = 8
SLOTS_PER_ATOM = 2

#: The nine, and what each one *is*. Counted from this file, not off a run:
#:
#:   6  reporting contracts that need no adapter, so none of them may depend on
#:      hardware being present -- the three `gpu_status` / `available_backends`
#:      contracts, the per-term decomposition naming its own backend, the
#:      `report_backend` XOR (a call that asked for the GPU says it ran or says
#:      why it did not, never both and never neither, and a call that did not
#:      ask must not invent a fallback that did not happen), and a decline this
#:      file provokes rather than hopes for. All six are about what the product
#:      *says*, which is the part that goes wrong silently: a wrong number is
#:      visible in a report, a wrong claim about which backend answered is not.
#:   + 2  the parity blocks, in-box and out-of-box. These two are the gate. An
#:      `f32` GPU reduction over a grid is not going to be bit-identical to an
#:      `f64` CPU sum, so the question is the size of accumulated rounding
#:      against the size of a bug, and the out-of-box block is the predicate
#:      whose original failure mode this repository repaired. They need a real
#:      adapter, and they say so when there is not one.
#:   + 1  `block_pinned_total`, which asserts the arithmetic and the partition
#:      into pass / fail / skip. It counts itself, which is why this is nine
#:      blocks and not eight.
#:
#:   6 + 2 + 1 = 9, and the seven that need no adapter are the six plus the
#: tally, which is why a GPU-less runner reaches the same number as a machine
#: with a card in it: `skip` records a result, so the two parity blocks stop
#: contributing measurements without stopping contributing results.
#:
#: **Why this is a derivation and not a transcription.** Each of the nine is a
#: `block_*` function, there are exactly nine of them, and every one reaches
#: exactly one `ok`, `bad` or `skip` on every path. So a block that could fall
#: through without recording anything would break the total rather than shrink
#: it quietly, and `block_pinned_total` is what would notice. The number is a
#: property of this file's source, which is why a run that produced nine passes
#: would be the wrong thing to put here: it would be a statement about a
#: machine, and it would keep saying it on a machine that never ran.
#:
#: **Where the reading is, and why it is not copied in.** The last time the two
#: adapter-dependent blocks were actually measured rather than skipped -- an
#: RTX 3050 Laptop adapter, built with `--features gpu`, reporting
#: `compiled=True` and `available=True` so that neither result came from the
#: no-adapter branch -- is written up in this file's docstring above, in the
#: section about the out-of-box block, together with the worst in-box and
#: out-of-box differences and the pair-reordering count. **Those figures are
#: deliberately not transcribed here.** They are a reading of one run against
#: one fixture and one engine revision; a pin that justified itself by quoting
#: them would go stale without ever looking stale, because the numbers would
#: still be sitting next to the constant. Read the docstring, or run the file.
#:
#: **What moves it, and what cannot:** a tenth block, a block split into two, or
#: a block that records two results -- each visible as a diff against the nine
#: named above, and each a deliberate edit here. Nothing else moves it: not the
#: machine, not whether the build has the `gpu` feature, and not whether the two
#: parity blocks could reach the kernel at all, which is the entire reason a
#: skip counts toward the total instead of being dropped from it.
#:
#: measured by counting the nine blocks above, then cross-checked on a machine
#: with an adapter and on one without, where the two parity blocks skip and the
#: total still reaches the same number.
#:
#: The call-site census, declared in the file it counts, measured by walking this
#: file's own syntax tree and compared against this block. **It was NOT declared
#: anywhere until 2026-10-02**, and the reason is worth writing down because it is
#: not a special case: `check_scripts_declare.py` classified this file as a gate,
#: walked it, and printed `NOT DECLARED ANYWHERE` for it. The pin above and this
#: census are two different numbers about the same file -- `EXPECTED_CHECKS` is
#: how many *blocks* the run must reach, the census is how many *result sites* the
#: source has -- and only the second can notice a block that records twice, or
#: whose `ok`/`bad` pair was deleted while the block survived. A file can be in
#: `INVENTORY`, hold a pin, and still have the only declaration of its own shape
#: written in somebody else's table.
#:
#: 1 of the 23 sites is unconditional and 22 are guarded. That is the shape of a
#: parity gate: almost every result is conditional on what the machine and the
#: build could answer, which is exactly why the guards digest matters here more
#: than the counts -- twenty-two guarded sites can be reshuffled into eleven
#: branches of two and the two counts would not notice.
#: GATE-DECLARE 1
#: sites: 1 unconditional + 22 guarded
#: guards: sha256:21badd1e4364867ae590fb3fd47fd60d8aaf1a4200f9dba21e0dc7b7587d3468
EXPECTED_CHECKS = 9

RESULTS: list[tuple[str, str, str]] = []

EXIT_OK = 0
EXIT_DEFECT = 1
EXIT_COUNT_MISMATCH = 3
EXIT_UNMEASURABLE = 4


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def skip(name: str, detail: str) -> None:
    RESULTS.append(("SKIP", name, detail))
    print(f"[SKIP] {name}\n       {detail}")


def section(title: str) -> None:
    print()
    print(f"-- {title}")


# --- fixtures --------------------------------------------------------------


def import_engine() -> tuple[object, object, str]:
    """Import the engine and say, in words, which copy this run measured.

    Three copies can be reachable at once -- the installed wheel, this
    checkout's `dock-py/python`, and whatever `PYTHONPATH` points at -- and a
    report that does not name which one is a number about an unknown binary.
    Same three-way labelling `core_check.py` does, for the same reason.
    """
    src = ROOT / "dock-py" / "python"
    try:  # noqa: SIM105
        import opendocking as od
    except ImportError:  # only on a checkout with no compiled extension
        sys.path.insert(0, str(src))
        import opendocking as od

    resolved = Path(od.__file__).resolve().parent
    if resolved == (src.resolve() / "opendocking"):
        where = f"this checkout's source tree at {resolved}"
    elif "site-packages" in resolved.parts:
        where = f"the installed wheel at {resolved}"
    else:
        where = f"a third copy, neither this checkout's nor site-packages, at {resolved}"

    from opendocking import core

    return od, core, where


def population(num_dof: int) -> np.ndarray:
    """A seeded DOF population, translated around the box and torsioned.

    Spread rather than clustered, so most atoms land on real trilinear blends
    instead of on grid nodes: a node lookup is the easiest case the kernel has,
    and a band only proved on nodes is a band proved on nothing.
    """
    rng = np.random.default_rng(SEED)
    confs = np.empty((N_CONF, num_dof), dtype=np.float64)
    for axis, centre in enumerate(BOX_CENTER):
        confs[:, axis] = centre + rng.uniform(-8.0, 8.0, N_CONF)
    confs[:, 3:6] = rng.uniform(-np.pi, np.pi, (N_CONF, 3))
    if num_dof > 6:
        confs[:, 6:] = rng.uniform(-np.pi, np.pi, (N_CONF, num_dof - 6))
    return np.ascontiguousarray(confs)


def rows_inside_box(core, box, ligand, confs: np.ndarray) -> np.ndarray:
    """One bool per row: does every atom of this conformation lie in the box.

    The criterion is the kernel's own -- `u = (p - min) / spacing`, out when
    `u < 0` or `u >= n - 1` -- and `GridBox.contains` states the same thing as
    `lo <= p < hi`, because the grid's last stored node sits exactly on the
    box's upper corner. Getting this wrong would put a row in the wrong
    population and every number below with it, so it is derived rather than
    assumed.
    """
    lo = np.asarray(box.min_corner)
    hi = np.asarray(box.max_corner)
    out = np.zeros(len(confs), dtype=bool)
    for i, conf in enumerate(confs):
        xyz = core.conformation_coordinates(ligand, conf)
        out[i] = bool(np.all((xyz >= lo) & (xyz < hi)))
    return out


def atoms_outside(core, box, ligand, conf) -> int:
    """How many atoms of one conformation lie outside the box."""
    lo = np.asarray(box.min_corner)
    hi = np.asarray(box.max_corner)
    xyz = core.conformation_coordinates(ligand, conf)
    over = np.maximum(lo - xyz, 0.0) + np.maximum(xyz - hi, 0.0)
    return int((np.linalg.norm(over, axis=1) > 0).sum())


def provoke_a_decline(n_atoms: int):
    """An n-atom linear carbon chain, one torsion per bond.

    Used to *provoke* a decline that does not depend on the hardware: the
    kernel refuses a ligand with more atoms than one workgroup, so on a machine
    that has an adapter this batch is declined for that reason, and on a machine
    that has not it is declined for the other one. Either way the caller asked
    for the GPU and the engine has to say so.
    """
    from opendocking import core

    coords = np.stack(
        [
            np.arange(n_atoms, dtype=np.float64) * 1.5,
            np.zeros(n_atoms),
            np.zeros(n_atoms),
        ],
        axis=1,
    )
    return core.Ligand.from_arrays(
        ["C"] * n_atoms,
        [0.0] * n_atoms,
        coords,
        [(i, i + 1) for i in range(n_atoms - 1)],
    )


def extension_fingerprint(od) -> str:
    """Size and digest of the extension this run's import actually resolved to.

    Printed rather than transcribed into this file's docstring, and that is the
    whole point of it: a number about a binary written into a file that outlives
    the binary is a claim that goes stale without looking stale, and this
    repository has had three of them. A rebuild in `site-packages` announces
    itself to nobody, so the run says which one it measured and a reader who
    wants to tie a recorded figure to a build runs the file and reads this.

    No verdict rides on it. The numbers below are the results; this is the
    address they were read from.
    """
    pkg = Path(od.__file__).resolve().parent
    found = sorted(p for p in pkg.glob("_dockpy*") if p.is_file())
    if not found:
        return "(no _dockpy* extension beside the package, so this run measured no binary)"
    return "; ".join(
        f"{p.name} {p.stat().st_size} B sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}"
        for p in found
    )


# --- the blocks ------------------------------------------------------------


def block_gpu_status_shape(core) -> None:
    name = "gpu_status publishes two bools and nothing else"
    status = core.gpu_status()
    keys = sorted(status)
    if keys == ["available", "compiled"] and all(
        isinstance(status[k], bool) for k in keys
    ):
        ok(name, f"gpu_status() == {status}: exactly the two documented keys, both bool")
    else:
        bad(name, f"gpu_status() == {status}: keys {keys}, expected ['available', 'compiled'] "
                  "both bool")


def block_available_implies_compiled(core) -> None:
    name = "an adapter is not reported available on a build with no GPU code"
    status = core.gpu_status()
    if not status["available"] or status["compiled"]:
        ok(name, f"compiled={status['compiled']}, available={status['available']}: a build "
                 "with no wgpu in it has nothing to open an adapter with. One of these "
                 "two numbers is a compile-time constant and the other is a probe; only "
                 "one of them is constructed, and this is the fact that tells them apart")
    else:
        bad(name, f"available={status['available']} while compiled={status['compiled']}")


def block_backends_advertise_only_a_probed_gpu(core) -> None:
    name = "available_backends advertises gpu exactly when a probe found one"
    status = core.gpu_status()
    backends = core.available_backends()
    advertises = "gpu" in backends
    if advertises == status["available"]:
        ok(name, f"available_backends() == {backends} and gpu_status()['available']="
                 f"{status['available']}: the gpu line is present exactly when "
                 "GpuContext::new() succeeded. It is a probe, not a `cfg!` constant -- "
                 "the compiled flag is reported separately and is not what this list reads")
    else:
        bad(name, f"available_backends() == {backends} advertises gpu={advertises} while "
                  f"gpu_status()['available']={status['available']}. A backend list that "
                  "is measured on the cpu side and assumed on the gpu side is half a lie")


def block_terms_name_their_backend(core, box, receptor, maps, ligand) -> None:
    name = "the term decomposition names its own backend, and it is the CPU"
    term_maps = receptor.precalculate_terms(box, SCORING, SPACING)
    conf = np.zeros(ligand.num_dof, dtype=np.float64)
    conf[:3] = BOX_CENTER
    breakdown = core.score_conformation_terms(ligand, maps, term_maps, conf, SCORING)
    # `getattr`, not attribute access: an older build genuinely lacks
    # `TermMaps.backend`, and a gate that raises on an old binary has measured
    # nothing and said nothing. Absent is a reportable fact, so it is reported.
    maps_backend = getattr(term_maps, "backend", None)
    value_backend = breakdown.get("backend") if hasattr(breakdown, "get") else None
    if maps_backend == "cpu" and value_backend == "cpu":
        ok(name, f"TermMaps.backend={maps_backend!r} and "
                 f"score_conformation_terms(...)[backend]={value_backend!r} on a build "
                 f"whose gpu feature compiled={core.gpu_status()['compiled']}. There is no "
                 "term-map kernel -- the GPU entry point takes GridMaps, not TermMaps -- "
                 "so on a gpu build the decomposition is absent rather than degraded, and "
                 "it says so on its own surface")
    elif maps_backend is None or value_backend is None:
        bad(name, f"this build exposes no backend name for the decomposition at all: "
                  f"TermMaps.backend is {maps_backend!r} "
                  f"({'the attribute does not exist' if not hasattr(term_maps, 'backend') else 'it is None'}) "
                  f"and the breakdown's is {value_backend!r}. A decomposition that cannot "
                  "name the backend that produced it cannot be checked against one")
    else:
        bad(name, f"TermMaps.backend={maps_backend!r}, "
                  f"score_conformation_terms backend={value_backend!r}: a decomposition "
                  "that does not name the CPU is one whose provenance the caller cannot "
                  "check")


def block_reports_are_self_consistent(reports, cpu_report) -> None:
    """The contract, on the reports of calls that *asked* for the GPU.

    A `use_gpu=False` call is deliberately not covered by it: nobody asked for
    the GPU, so `backend='cpu'` with no reason is the right answer and holding
    it to the other rule would be the gate being wrong rather than the product.
    That call is held to the opposite rule instead -- it must never invent a
    fallback that did not happen.
    """
    name = "every report says either the gpu ran or why it did not"
    broken = []
    for label, report in reports:
        ran = report.get("backend") == "gpu"
        reason = report.get("gpu_skip_reason")
        said_why = isinstance(reason, str) and bool(reason.strip())
        if ran == said_why:
            broken.append((label, dict(report)))
    phantom = cpu_report.get("gpu_skip_reason") is not None
    if not broken and not phantom:
        ok(name, f"{len(reports)} use_gpu=True report(s) satisfy 'gpu ran' XOR 'a "
                 "non-empty reason why it did not' -- never both, never neither -- and "
                 "the use_gpu=False report claims no fallback that did not happen")
    else:
        bad(name, f"{len(broken)} of {len(reports)} use_gpu=True reports break the XOR: "
                  f"{broken}. use_gpu=False reported a fallback anyway: {phantom} "
                  f"({cpu_report!r}). 'gpu' XOR a non-empty gpu_skip_reason is the "
                  "contract that makes a decline audible")


def block_declined_batch_says_why(core, maps) -> None:
    name = "a batch the gpu cannot take is declined with a reason, not silently"
    n = WORKGROUP_SIZE + 6
    ligand = provoke_a_decline(n)
    if ligand.num_atoms != n:
        bad(name, f"the provoked ligand has {ligand.num_atoms} atoms, not {n}; this file's "
                  "fixture is broken, not the engine")
        return
    confs = np.zeros((4, ligand.num_dof), dtype=np.float64)
    gpu_report: dict = {}
    gpu = core.evaluate_conformations(ligand, maps, confs, SCORING, True, gpu_report)
    cpu = core.evaluate_conformations(ligand, maps, confs, SCORING, False, None)
    reason = gpu_report.get("gpu_skip_reason")
    said_why = isinstance(reason, str) and bool(reason.strip())
    same = bool(np.array_equal(gpu, cpu))
    ran_on_cpu = gpu_report.get("backend") == "cpu"

    if said_why and same and ran_on_cpu:
        ok(name, f"a {n}-atom batch was declined with reason {reason!r}, reported as "
                 f"backend='cpu', and returned the CPU answer bit-for-bit. The decline was "
                 "provoked, not hoped for, so this runs on a machine that has an adapter too")
    elif same and ran_on_cpu and not said_why:
        status = core.gpu_status()
        bad(name, f"evaluate_population's #[cfg(not(feature = \"gpu\"))] arm drops "
                  f"prefer_gpu and returns no skip: use_gpu=True came back as "
                  f"backend='cpu' with gpu_skip_reason=None on a build reporting "
                  f"compiled={status['compiled']} "
                  f"available={status['available']}. The energies are right "
                  f"(bit-identical to the CPU run) and `backend` is named, so the "
                  f"caller can tell what ran -- what is missing is the *reason*. "
                  f"The fix is in `dock-core/src`, in that arm; nothing in this "
                  f"repository's prose needs changing, because README.md:415-417 "
                  f"and README.en.md:487-490 already name this exact behaviour as "
                  f"the defect -- in the past tense, which is what they have said "
                  f"since the arm was repaired, and before that in the present "
                  f"tense ('a build with no gpu feature, where use_gpu=True "
                  f"falls back silently and gives no reason')")
    else:
        bad(name, f"provoked batch came back as {gpu_report}, reason={reason!r}, "
                  f"bit-identical to cpu={same}")


def derived_band(core, maps, ligand) -> float:
    """The band, in kcal/mol, above which the difference stops being rounding.

    `ROUNDINGS` is a count of the `f32` roundings a grid value passes through on
    its way to a conformation's energy -- one per value entering the sum -- and
    the two things it counts are asserted here so it cannot drift from them.
    `max|grid|` is read from the maps this run built, and the `2` is the most
    map slots any atom kind reads (`weights_for_kind`, scoring.rs: every weight
    is exactly 0.0 or 1.0, and at most two are set).
    """
    gmax = float(np.max(np.abs(maps.raw_data)))
    roundings = float(CORNERS_PER_ATOM * SLOTS_PER_ATOM * ligand.num_atoms)
    eps32 = float(np.finfo(np.float32).eps)
    return roundings * eps32 * (SLOTS_PER_ATOM * gmax) * ligand.num_atoms


def both_backends(core, maps, ligand, confs):
    gpu_report: dict = {}
    gpu = core.evaluate_conformations(ligand, maps, confs, SCORING, True, gpu_report)
    cpu = core.evaluate_conformations(ligand, maps, confs, SCORING, False, None)
    return gpu, cpu, gpu_report


def block_parity_in_box(core, box, maps, ligand, adapter) -> None:
    name = "gpu and cpu agree, in kcal/mol, on every row that stays in the box"
    confs = population(ligand.num_dof)
    gpu, cpu, report = both_backends(core, maps, ligand, confs)
    if report.get("backend") != "gpu":
        skip(name, f"this run did not reach the kernel (backend={report.get('backend')!r}, "
                   f"gpu_skip_reason={report.get('gpu_skip_reason')!r}), so no GPU-vs-CPU "
                   "difference was computed and none is reported. This line is an absence, "
                   "not a result")
        return

    inside = rows_inside_box(core, box, ligand, confs)
    rows = np.flatnonzero(inside)
    # A vacuity guard. If the population stopped producing in-box rows this would
    # become "0 rows agree", which is not a smaller claim, it is no claim.
    if len(rows) < N_CONF // 4:
        bad(name, f"only {len(rows)} of {N_CONF} rows kept every atom inside the box, so "
                  "the parity comparison would be measuring a handful of rows and calling "
                  "it agreement. The population is meant to spread across the box")
        return

    diff = np.abs(gpu[rows] - cpu[rows])
    band = derived_band(core, maps, ligand)
    worst = int(np.argmax(diff))
    within = bool(np.all(diff <= band))
    order_same = bool(
        np.array_equal(
            np.argsort(cpu[rows], kind="stable"), np.argsort(gpu[rows], kind="stable")
        )
    )
    detail = (
        f"adapter {adapter!r}; {len(rows)} of {N_CONF} rows in box (the other "
        f"{N_CONF - len(rows)} left it and are the next block's subject). Worst "
        f"|GPU-CPU| = {diff[worst]:.3e} kcal/mol at row {rows[worst]} (cpu "
        f"{cpu[rows][worst]:.6f}, gpu {gpu[rows][worst]:.6f}), against a derived band of "
        f"{band:.3e} -- {100.0 * float(diff[worst]) / band:.3f}% of it. Ranking over those "
        f"rows identical: {order_same}. max|cpu| = {float(np.max(np.abs(cpu[rows]))):.3f}"
    )
    if within and order_same:
        ok(name, detail + ". That is the size of accumulated f32 rounding, not the size "
                         "of a bug, and it is why the two paths are not bit-identical")
    else:
        bad(name, detail)


def block_out_of_box_agrees(core, box, maps, ligand, adapter) -> None:
    name = "a row that leaves the box is scored by the same function on both backends"
    confs = population(ligand.num_dof)
    gpu, cpu, report = both_backends(core, maps, ligand, confs)
    if report.get("backend") != "gpu":
        skip(name, f"no adapter here (backend={report.get('backend')!r}), so the "
                   "out-of-box behaviour of the kernel was not exercised and no "
                   "disagreement was computed or denied")
        return

    inside = rows_inside_box(core, box, ligand, confs)
    outside = np.flatnonzero(~inside)
    band = derived_band(core, maps, ligand)
    outside_diff = np.abs(gpu[outside] - cpu[outside]) if len(outside) else np.zeros(0)
    over_band = int((outside_diff > band).sum())

    # A single controlled slide, so the disagreement is shown as a curve rather
    # than only as a worst case, and so the reader can see where it starts.
    seed_row = int(np.flatnonzero(inside)[0])
    base = confs[seed_row].copy()
    curve = []
    for shift in (0.0, 4.0, 6.0, 6.5, 7.0, 8.0, 9.0, 12.0):
        one = base.copy()
        one[0] = base[0] + shift
        one = np.ascontiguousarray(one.reshape(1, -1))
        n_out = atoms_outside(core, box, ligand, one[0])
        e_gpu, e_cpu, _r = both_backends(core, maps, ligand, one)
        curve.append((shift, n_out, float(e_cpu[0]), float(e_gpu[0])))

    if over_band == 0:
        ok(name, f"{len(outside)} rows left the box and none differed by more than "
                 f"{band:.3e} kcal/mol")
        return

    worst = int(np.argmax(outside_diff))
    slide = "  ".join(
        f"shift {s:>4}: {n} out, cpu {c:>10.3f} gpu {g:>10.3f}"
        for s, n, c, g in curve
    )
    # The widest point of the slide, named rather than written down: the old text
    # of this message hardcoded "at shift 8", which is a claim about one
    # population and this file is free to change its population.
    split = max(curve, key=lambda t: abs(t[3] - t[2]))
    bad(
        name,
        f"adapter {adapter!r}: {over_band} of {len(outside)} rows that left the box "
        f"differed from the CPU by more than the derived band of {band:.3e} kcal/mol, "
        f"worst {outside_diff[worst]:.4e} kcal/mol at row {outside[worst]} "
        f"(cpu {cpu[outside][worst]:.3f}, gpu {gpu[outside][worst]:.3f}). Both "
        f"backends are claimed to evaluate one function -- OUT_OF_BOX_PENALTY times "
        f"the sum, over protruding atoms and axes, of max(p - bmax, min - p, 0) -- so "
        f"a difference of this size against a band of {band:.3e} is not f32 rounding, "
        f"it is a different function. The curve below is what tells the two candidate "
        f"causes apart, and they are the only two that survive the shader's own test. "
        f"A flat OUT_OF_BOX_PENALTY per protruding atom rather than per angstrom caps "
        f"the GPU's term at OUT_OF_BOX_PENALTY x {ligand.num_atoms} however far out it "
        f"goes, so it peels away from the CPU's unbounded ramp as the slide continues; "
        f"a penalty constant that has drifted from search::OUT_OF_BOX_PENALTY instead "
        f"leaves the shape intact and the offset wrong everywhere, including at shift 0. "
        f"The widest split on the controlled slide is {abs(split[3] - split[2]):.4e} "
        f"kcal/mol at shift {split[0]} with {split[1]} atom(s) out. One conformation "
        f"slid along x -- {slide}. Nothing reports this: report_backend calls that run "
        f"{ {k: v for k, v in report.items() if k != 'num_conformations'} }",
    )


def block_pinned_total(npass: int, nfail: int, nskip: int) -> None:
    name = "the_pinned_total_counts_results_and_not_only_passes"
    # `+ 1` is this check itself, which is already in `RESULTS` but has not been
    # counted into the arguments yet. The pin includes it, the same way
    # `EXPECTED_CHECKS = 153  # measured from a green run, including this check`
    # reads in `check_scripts_declare.py`.
    got = npass + nfail + nskip + 1
    if got == EXPECTED_CHECKS:
        ok(name, f"{got} results against a pin of {EXPECTED_CHECKS}: {npass} passed, "
                 f"{nfail} failed, {nskip} skipped -- three numbers, printed separately and "
                 "never added together")
    else:
        bad(name, f"{got} results against a pin of {EXPECTED_CHECKS} ({npass} passed, "
                  f"{nfail} failed, {nskip} skipped). The pin is the number of blocks this "
                  "file has: fewer means a run did not finish, more means a block grew "
                  "that nobody pinned")


# --- run -------------------------------------------------------------------


def main() -> int:
    od, core, where = import_engine()
    box = core.GridBox.from_center_size(BOX_CENTER, BOX_SIZE)

    section("which binary this run measured")
    print(f"opendocking = {od.__file__}")
    print(f"engine      = {core.engine_version()} from {where}")
    print(f"extension   = {extension_fingerprint(od)}")
    print(f"gpu_status  = {core.gpu_status()}")
    print(f"backends    = {core.available_backends()}")

    block_gpu_status_shape(core)
    block_available_implies_compiled(core)
    block_backends_advertise_only_a_probed_gpu(core)

    maps = receptor = ligand = None
    section("the per-term decomposition, which is CPU-only on every build")
    # Built first, then asked, and the two failures are told apart. A block that
    # both raised and recorded a result would count twice against the pin, which
    # is how a gate ends up with a count no run can produce.
    try:
        receptor = core.Receptor.from_pdbqt(EXAMPLES / RECEPTOR)
        maps = receptor.precalculate(box, SCORING, SPACING)
        ligand = core.Ligand.from_pdbqt(EXAMPLES / LIGAND)
    except Exception as exc:  # noqa: BLE001 -- a fact about this run, not a crash
        bad("the term decomposition names its own backend, and it is the CPU",
            f"could not build the maps to ask: {type(exc).__name__}: {exc}")
    if maps is not None and ligand is not None:
        block_terms_name_their_backend(core, box, receptor, maps, ligand)

    if maps is not None:
        section("the reporting contract, which every machine can provoke")
        probe = np.ascontiguousarray(population(ligand.num_dof)[:8])
        cpu_report: dict = {}
        core.evaluate_conformations(ligand, maps, probe, SCORING, False, cpu_report)
        gpu_report: dict = {}
        core.evaluate_conformations(ligand, maps, probe, SCORING, True, gpu_report)
        print(f"       use_gpu=False -> {cpu_report}")
        print(f"       use_gpu=True  -> {gpu_report}")
        block_reports_are_self_consistent([("use_gpu=True", gpu_report)], cpu_report)
        block_declined_batch_says_why(core, maps)

        status = core.gpu_status()
        adapter = "(not reported on this run)"
        section("the two parity blocks, which need an adapter")
        if status["available"]:
            # The adapter name is only available on a report of a run that
            # actually used the kernel, so take it from the parity run itself.
            probe_report: dict = {}
            core.evaluate_conformations(ligand, maps, probe, SCORING, True, probe_report)
            adapter = probe_report.get("adapter") or "(unnamed adapter)"
            block_parity_in_box(core, box, maps, ligand, adapter)
            block_out_of_box_agrees(core, box, maps, ligand, adapter)
        else:
            for name, why in (
                ("gpu and cpu agree, in kcal/mol, on every row that stays in the box",
                 "the kernel was not reached, so no difference was computed"),
                ("a row that leaves the box is scored by the same function on both "
                 "backends",
                 "the kernel was not reached, so its out-of-box behaviour was not "
                 "exercised"),
            ):
                skip(name, f"no adapter here (compiled={status['compiled']}, "
                           f"available={status['available']}), so {why} and none is "
                           "reported. This line is an absence, not a result")

    section("the pin")
    counts = (
        sum(1 for t, _n, _d in RESULTS if t == "PASS"),
        sum(1 for t, _n, _d in RESULTS if t == "FAIL"),
        sum(1 for t, _n, _d in RESULTS if t == "SKIP"),
    )
    block_pinned_total(*counts)
    npass, nfail, nskip = (
        sum(1 for t, _n, _d in RESULTS if t == "PASS"),
        sum(1 for t, _n, _d in RESULTS if t == "FAIL"),
        sum(1 for t, _n, _d in RESULTS if t == "SKIP"),
    )
    print()
    print(f"--- {npass} passed, {nfail} failed, {nskip} skipped, {len(RESULTS)} results "
          f"(expected {EXPECTED_CHECKS})")
    if nfail:
        print(f"RESULT: DEFECT (exit {EXIT_DEFECT}) -- {nfail} of {len(RESULTS)} recorded "
              "results failed; every failure above names the file that would fix it")
        return EXIT_DEFECT
    if len(RESULTS) != EXPECTED_CHECKS:
        return EXIT_COUNT_MISMATCH
    if maps is None:
        print(f"RESULT: UNMEASURABLE (exit {EXIT_UNMEASURABLE}) -- the maps this file needs "
              "were never built, so nothing above is a measurement of the engine")
        return EXIT_UNMEASURABLE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
