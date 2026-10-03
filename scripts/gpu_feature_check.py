"""Does the `gpu` feature answer for itself, on a machine that cannot see a GPU?

Run:  python scripts/gpu_feature_check.py

**What this file is, and the machine it was written for.**

It is a *feature-contract* gate. The repository ships an optional `gpu` feature
behind which a `wgpu`/WGSL compute kernel lives, and the question this file asks
is not "do the two backends agree" -- `gpu_cpu_parity_check.py` asks that, and
it needs an adapter. The question here is the one that comes first:

    when a caller asks for the GPU, does the product say what happened?

A caller that asked for the GPU and got the CPU has been **declined**. A decline
that is not reported is a silent, unexplained slowdown, and a silent slowdown is
the failure mode that survives every other test in this repository, because the
numbers are still right. So this file is deliberately about the reporting layer
and barely about the arithmetic.

**Every CI runner is a CPU-only build, and this file is written for that.**

That is the constraint that shaped every block below. A runner that has built
`--features gpu` still has no adapter, and the `default` leg of the matrix has
neither the feature nor an adapter -- which is the machine *this* file has to be
useful on, or it is a gate that only runs on one person's laptop. So the blocks
are split by what the build can answer:

  * what needs **no adapter and no feature**: the `gpu_status` /
    `available_backends` contracts, what the `backend` field is allowed to say,
    the four-decline XOR, the bit-for-bit CPU fallback, and the `f32` boundary
    behaviour. All of these run everywhere, and all of them are the ones that go
    wrong silently.
  * what needs the **feature**: whether the `f32` narrowing guard is what
    declined the oversized coordinate, rather than a missing adapter. Without
    the feature the guard is not in the binary at all, so this row **skips with
    the absence named**.
  * what needs a **real adapter**: one row, which **skips with the absence
    named**.

**A skip is a result, and this file says so in its own output.**

`EXPECTED_CHECKS` is 11 on every machine, and `skip` writes a third tag onto
`RESULTS` so a skip reaches the total exactly the way a pass does. The summary
prints three numbers that are never added, and the total beside them is read
**after** the pin block has recorded itself, so the three add up to the total on
every run. A summary whose numbers do not add up is a reporting defect and this
file had one: it used to sample the three tags before the pin and print
`8 passed, 0 failed, 2 skipped` beside `11 results`, which a reader cannot tell
from a pin that was forgotten. A build without the `gpu` feature skips the two
rows that need the kernel and passes the other nine, so it reports
`9 passed, 0 failed, 2 skipped`; a build with the feature and an adapter
reaches the kernel, skips nothing, and reports `11 passed, 0 failed, 0 skipped`.
The check that stops the opposite mistake reaching the summary is
`the_pinned_total_counts_results_and_not_only_passes`.

**The two rows that skip name the mechanism, not the symptom.**

The interesting fact about this repository's GPU path is that on a build
without the `gpu` feature, `gpu_compiled()` is literally `cfg!(feature = "gpu")`
(`dock-py/src/lib.rs`) and `gpu_available()` returns a constant `false` in its
`#[cfg(not(feature = "gpu"))]` arm. **No adapter probe is ever constructed.**
So a build without the feature reports "no GPU" on a machine that has a GPU in
it, and the two facts are indistinguishable from the outside. That is a
build-identity coupling, not a hardware absence, and the wording of both skip
rows says which one it was looking at, because a skip that cannot name its own
cause is a hole wearing a result's clothes.

**Nothing here is transcribed that can be derived.**

The out-of-box penalty, the kernel's workgroup size, the shape of the kernel's
out-of-box term, and the census of decline reasons are all read out of the
sources this repository owns, by the functions named next to them. The shape is
the one that is read and **printed rather than asserted**, because
`dock-core/src/gpu/mod.rs` already binds the shader to the Rust in
`shader_and_rust_agree_on_the_out_of_box_penalty` and `cargo test` runs that on
a machine that never runs this file; the reading is printed here because the
`f32` bound below is computed on top of it. The decline census in particular is
a grep for `reason: "` in `dock-core/src/search/mod.rs` and is
**never written down here**, because a closed list of the ways a decline can
happen is a list that goes stale the moment a seventh is added -- and the last
one that was added to this repository, the `f32` narrowing guard, was found by
bracketing a boundary rather than by reading a list.

**The `f32` row is here because that guard is a boundary, and a boundary tested
at three order-of-magnitude-separated points is a guess.**

An `f32` out-of-box penalty is `OUT_OF_BOX_PENALTY * violation` evaluated in
`f32`, so a coordinate is representable on the GPU path only while *that
product* stays finite -- which is a much lower ceiling than `f32::MAX`. The
block below asserts the property that actually matters and holds on every
build: **no coordinate, of any magnitude, ever returns a non-finite energy from
a path that reports itself as having run.** A decline may happen; an infinity
may not.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# The same receptor, ligand, box and scoring function `gpu_cpu_parity_check.py`
# uses, so the two gates agree on what a real case is rather than each inventing
# one, and so a reader who runs both is comparing like with like.
RECEPTOR = "1crn_prep.pdbqt"
LIGAND = "biotin_prep.pdbqt"
BOX_CENTER = (18.0, 12.0, 20.0)
BOX_SIZE = (22.0, 22.0, 22.0)
SPACING = 0.5
SCORING = "vina"

SEARCH_RS = ROOT / "dock-core" / "src" / "search" / "mod.rs"
ENERGY_WGSL = ROOT / "dock-core" / "src" / "gpu" / "energy.wgsl"

#: Magnitudes fed through the `f32` row. They span both ways of running out of
#: `f32` -- the coordinate itself, and the penalty the coordinate is multiplied
#: by -- and the last one is past `f32::MAX` so the saturating cast is exercised
#: too. Every one of them is a finite `f64`, so every one of them is an input
#: the engine's own coordinate rules accept.
F32_PROBES = (1.0e35, 1.0e38, 1.0e39, 1.0e308)

#: What a report has to look like for this file to read it as a report. A dict
#: that is missing a key cannot be judged against the XOR, so it is a failure
#: rather than a silent pass.
REQUIRED_REPORT_KEYS = ("backend", "gpu_skip_reason", "adapter")

#: The eleven, and what each one *is*:
#:
#:   4  build-identity contracts that need neither an adapter nor the feature:
#:      the `gpu_status` shape, `available` not being claimed on a build with no
#:      GPU code in it, `available_backends` advertising `gpu` only when a probe
#:      found one, and what the `backend` field is allowed to say.
#:   + 2  the decline contract: a GPU request either ran on the GPU or carried
#:      a non-empty reason, across four separately provoked declines; and a
#:      declined call returned the CPU's numbers bit for bit. These are the
#:      blocks that go wrong silently, which is why they are the core of the
#:      file rather than a preliminary to it.
#:   + 1  the `f32` boundary, as a property rather than a threshold: no
#:      magnitude returns a non-finite energy from a path that says it ran.
#:   + 1  whether the `f32` guard is what declined the oversized coordinate.
#:      **Skips** on a build without the feature, where the guard is not in the
#:      binary to be exercised.
#:   + 1  that the kernel was reached and compared against the CPU. **Skips**
#:      without an adapter.
#:   + 1  this file's own predicate, run against deliberately broken reports so
#:      that a guard nobody has seen fail is known to be able to fail.
#:   + 1  the pin.
#:
#:   4 + 2 + 1 + 1 + 1 + 1 + 1 = 11, on every machine, because a skip is a
#:   result and not an absence. The partition is asserted by
#:   `the_pinned_total_counts_results_and_not_only_passes`, so a block that
#:   could fall through without recording would move the total rather than
#:   shrink it quietly.
#:
#: **What moves it.** An extra block, a block split in two, or a block that
#: records twice -- each a diff against this comment and each a deliberate edit
#: here. Nothing else moves it: not the machine, not the absence of the `gpu`
#: feature, and not the absence of an adapter.
#:
#: The census, declared in the file it counts and measured by walking this
#: file's own syntax tree: **0 of the 23 result sites is unconditional and all
#: 23 are guarded.** That is the shape of a feature-contract gate and it is the
#: reason the guards column matters more than the counts here: every single
#: result depends on what the build and the machine could answer, so 23 guarded
#: sites could be reshuffled into 11 branches of 2 and the total alone would not
#: notice. The digest is what does.
#:
#: **`INVENTORY` membership is a point-in-time claim and the authoritative
#: source is `scripts/check_scripts_declare.py`, not this comment.** It read
#: "neither `gpu_feature_check.py` nor `provenance_appendix.py` is in it" for
#: several rounds after both names had been added, because the sentence was
#: typed once and never re-read. A sibling gate found the same stale claim about
#: itself and fixed it by *reading* the membership at run time rather than
#: typing it, which is the only version of this sentence that cannot rot.
#:
#: As of 2026-10-02 both names are present, and this file's `GATE-DECLARE` head
#: above is the `GATE-DECLARE 1` block the auditor reads back — a bare `#`
#: instead of `#:` made the block invisible to a reader matching
#: `^#:\s*GATE-DECLARE`, and the file reported `NOT DECLARED ANYWHERE` while
#: looking perfectly declared. A declaration that is present but unreadable is
#: the same failure as one that is absent, wearing a disguise.
#: GATE-DECLARE 1
#: sites: 0 unconditional + 23 guarded
#: guards: sha256:f03362c20379cb9c3c6a4f38ddc04d19ec291f7495ea6e91308dc9de00aaadb4
EXPECTED_CHECKS = 11

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


# --- what the sources say, read rather than transcribed ---------------------


def read_text(path: Path) -> str:
    # `encoding="utf-8"` explicitly, and never the locale's: this repository
    # holds non-ASCII in its comments, and a mis-decoded read would make a
    # regex fail in a way that looks like a missing declaration.
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


def out_of_box_penalty() -> float:
    """`search::OUT_OF_BOX_PENALTY`, from the declaration that defines it."""
    m = re.search(r"pub const OUT_OF_BOX_PENALTY: f64 = ([0-9.]+);", read_text(SEARCH_RS))
    if not m:
        raise AssertionError(f"no OUT_OF_BOX_PENALTY declaration in {SEARCH_RS}")
    return float(m.group(1))


def decline_reasons() -> list[str]:
    """Every decline the engine can report, counted by grepping for its own site.

    Deliberately **not** a list written down here. A closed list of the ways a
    decline can happen is a list that goes stale the moment one is added, and
    the one that was added most recently -- the `f32` narrowing guard -- is
    precisely the one a hand-written list would have been missing.
    """
    return re.findall(r'reason: "([^"]+)"', read_text(SEARCH_RS))


def workgroup_size() -> int:
    """`@workgroup_size(N)` in the shader, which is also the batch ceiling."""
    m = re.search(r"@workgroup_size\((\d+)\)", read_text(ENERGY_WGSL))
    if not m:
        raise AssertionError(f"no @workgroup_size in {ENERGY_WGSL}")
    return int(m.group(1))


def shader_charges_the_hosts_summed_violation() -> bool:
    """Whether the kernel's out-of-box term is still the one the bound assumes.

    The bound is `OUT_OF_BOX_PENALTY * AXES * n_atoms * max|p|`, and it needs two
    things: a sum over the three axes somewhere, and a kernel that multiplies one
    already-summed scalar by the penalty instead of reducing a per-atom distance
    itself. Both still hold. The host sums the axes in `f64`
    (`out_of_box_violation_per_axis(...).iter().sum()`, `gpu/mod.rs`) and the
    shader takes that scalar as `violation: f32` and multiplies it once.

    What does not hold is the *previous* version of this probe, which looked for
    `max(per_axis, vec3<f32>(0.0))` in the shader. That reduction moved to the
    host, so the probe kept asking a question the shader stopped being asked and
    read `False` for a shape the shader has -- and a printed `False` reads to a
    reader as the opposite of the truth, which is the whole failure this
    repository exists to remove.

    **It is read and printed, not asserted, and that is the honest limit.** The
    first version of this docstring claimed the shape was "asserted rather than
    remembered" while the only thing that happened to it was a `print`, and a
    docstring claiming an assertion that does not exist is worse than no
    docstring. Making it a twelfth result would move `EXPECTED_CHECKS` and the
    call-site census for a shape that `dock-core/src/gpu/mod.rs` already binds
    in `shader_and_rust_agree_on_the_out_of_box_penalty`, which `cargo test` runs
    on machines that never run this file -- so the strong assertion lives there,
    and this run prints the reading beside the number that is computed on top of
    it.
    """
    wgsl = read_text(ENERGY_WGSL)
    return (
        "OUT_OF_BOX_PENALTY * violation" in wgsl
        and "violation: f32" in wgsl
    )


#: The out-of-box term is a sum over three axes and the kernel's reduction sums
#: one term per atom, so a row's `f32` total is bounded by
#: `OUT_OF_BOX_PENALTY * AXES * n_atoms * max|p|`. Both factors are counted
#: from something: `AXES` from the `vec3` the **host** reduces -- the shader
#: stopped reducing it and takes the summed scalar as `violation: f32` -- and
#: `n_atoms` from the ligand the caller passed.
AXES = 3


def f32_safe_coordinate(penalty: float, n_atoms: int) -> float:
    """The largest `|p|` whose worst-case `f32` out-of-box term is still finite.

    The guard in `search::try_gpu_population` narrows `OUT_OF_BOX_PENALTY * p`,
    which is exact for a one-atom ligand translated along one axis and **too
    permissive for every other shape**: the shader sums the per-axis excesses
    and then the kernel sums one term per atom, so the quantity that has to
    survive the narrowing is the whole row, not one coordinate of it.

    This function is the number the guard would have to use to be on the safe
    side, and the block below reports the ratio between the two. It is
    arithmetic, not a measurement, and it is labelled as such wherever printed.
    """
    return float(np.finfo(np.float32).max) / (penalty * AXES * max(1, n_atoms))


# --- fixtures ---------------------------------------------------------------


def import_engine() -> tuple[object, object, str]:
    """Import the engine and say, in words, which copy this run measured.

    Same three-way labelling `gpu_cpu_parity_check.py` and `core_check.py` do,
    for the same reason: a number about an unlabelled binary is a number about
    an unknown one.
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


def chain_ligand(n_atoms: int):
    """An n-atom linear carbon chain: a batch the kernel provably cannot take.

    Provoked rather than hoped for, so the decline exists on a machine that has
    an adapter as well as on one that does not.
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


def ask(core, ligand, maps, confs, use_gpu: bool) -> tuple[np.ndarray, dict]:
    report: dict = {}
    energies = core.evaluate_conformations(ligand, maps, confs, SCORING, use_gpu, report)
    return np.asarray(energies, dtype=np.float64), report


def reason_of(report: dict) -> str:
    """The report's decline reason, or `""` -- never `None`, never a crash.

    A missing key is `""` rather than an exception on purpose: "the report did
    not carry a reason" and "the report carried an empty one" have to be
    distinguishable downstream, and neither is a reason to stop reading.
    """
    r = report.get("gpu_skip_reason")
    return r.strip() if isinstance(r, str) else ""


def xor_holds(report: dict, asked_for_gpu: bool) -> tuple[bool, str]:
    """The contract, as a pure function of one request and its report.

    The request is an argument, not a background fact, and that is the whole
    reason this is not a one-argument predicate. "A decline carries a reason" is
    only half of it: the other half is that a caller who did **not** ask for the
    GPU must not be handed a fallback that did not happen, and a report cannot
    tell those two apart on its own. So the phantom-fallback clause is
    unreachable unless the request travels with the report -- which is why
    `block_this_predicate_can_go_red` fails against the one-argument version.

    Clauses, all three of which have to hold:

    * `backend` names something that ran, and only that.
    * Asked for the GPU and did not get it -> a decline, and a decline carries
      a non-empty reason.
    * Did not ask for the GPU -> no reason, ever, and no adapter either.

    Returned as a verdict plus the clause that broke, so a red block can say
    which one failed rather than only that one did.
    """
    for key in REQUIRED_REPORT_KEYS:
        if key not in report:
            return False, f"the report has no {key!r} key: {report!r}"
    backend = report.get("backend")
    if backend not in ("cpu", "gpu"):
        return False, f"backend={backend!r} names nothing that ran; the contract is cpu|gpu"
    reason = reason_of(report)

    if not asked_for_gpu:
        if reason:
            return False, (
                f"use_gpu=False carried gpu_skip_reason={reason!r}: a caller who did not ask for "
                "the GPU was told a fallback happened that never did, which is a decline "
                "invented out of nothing"
            )
        if report.get("adapter") is not None:
            return False, f"use_gpu=False carried adapter={report.get('adapter')!r}"
        return True, "did not ask for the GPU, ran on the CPU, no reason, as it should be"

    if backend == "gpu":
        if reason:
            return False, (f"backend='gpu' with gpu_skip_reason={reason!r}: it ran, so there is "
                           "nothing to decline")
        return True, "asked for the GPU, ran on the GPU, no reason, as it should be"
    if not reason:
        return False, (
            "use_gpu=True came back backend='cpu' with gpu_skip_reason=None: a caller who asked "
            "for the GPU and got the CPU has been declined, and a decline with no reason is an "
            "unexplained slowdown"
        )
    if report.get("adapter") is not None:
        return False, (f"backend='cpu' with adapter={report.get('adapter')!r}: nothing ran on the "
                       "GPU to have an adapter")
    return True, f"asked for the GPU, declined with reason {reason!r}"


# --- the blocks -------------------------------------------------------------


def block_gpu_status_shape(core) -> None:
    name = "gpu_status publishes two bools and nothing else"
    status = core.gpu_status()
    keys = sorted(status)
    if keys == ["available", "compiled"] and all(isinstance(status[k], bool) for k in keys):
        ok(name, f"gpu_status() == {status}: exactly the two documented keys, both bool")
    else:
        bad(name, f"gpu_status() == {status}: keys {keys}, expected ['available', 'compiled'] both bool")


def block_available_implies_compiled(core) -> None:
    name = "an adapter is not reported available on a build with no GPU code"
    status = core.gpu_status()
    if not status["available"] or status["compiled"]:
        ok(name, f"compiled={status['compiled']}, available={status['available']}: a build with no "
                 "wgpu in it has nothing to open an adapter with. One of these is a compile-time "
                 "constant and the other is a probe, and this is the fact that tells them apart")
    else:
        bad(name, f"available={status['available']} while compiled={status['compiled']}: a build "
                  "with no GPU code in it cannot have opened an adapter")


def block_backends_advertise_only_a_probed_gpu(core) -> None:
    name = "available_backends advertises gpu exactly when a probe found one"
    status = core.gpu_status()
    backends = core.available_backends()
    advertises = "gpu" in backends
    if advertises == status["available"]:
        ok(name, f"available_backends() == {backends} and available={status['available']}: the gpu "
                 "line is present exactly when GpuContext::new() succeeded. A backend list measured "
                 "on the cpu side and assumed on the gpu side is half a lie")
    else:
        bad(name, f"available_backends() == {backends} advertises gpu={advertises} while "
                  f"available={status['available']}")


def block_backend_answers_only_what_ran(core, reports) -> None:
    name = "the backend field answers only what ran, never why"
    problems = []
    for label, report in reports:
        backend = report.get("backend")
        if backend not in ("cpu", "gpu"):
            problems.append(f"{label}: backend={backend!r}")
        if backend == "cpu" and report.get("adapter") is not None:
            problems.append(f"{label}: backend='cpu' carries adapter={report.get('adapter')!r}")
        if backend == "gpu" and not report.get("adapter"):
            problems.append(f"{label}: backend='gpu' with adapter={report.get('adapter')!r}")
    shown = "; ".join(f"{label} -> {report.get('backend')!r}" for label, report in reports)
    if not problems:
        ok(name, f"{shown}. `backend` is the one field in the report whose whole job is to say "
                 "what computed the number, and `gpu_skip_reason` is the one whose job is to say "
                 "why it could not. Merging them would make a decline indistinguishable from a choice")
    else:
        bad(name, f"{shown}: {problems}")


def block_gpu_request_ran_or_said_why(core, probes) -> None:
    name = "a gpu request either ran on the gpu or said why it did not"
    lines, broken = [], []
    for label, asked, report in probes:
        held, why = xor_holds(report, asked)
        # Printed one per line and never totalled: a reason string is not a
        # quantity, and "4 reasons" is not a smaller claim than the four of them.
        lines.append(f"       {label:<34} -> {why}")
        if not held:
            broken.append(f"{label}: {why}")
    detail = "\n".join(lines)
    if not broken:
        ok(name, f"every one of the {len(probes)} provoked requests was answered in one of the two "
                 f"legal ways, and none of them was both:\n{detail}")
    else:
        bad(name, f"{len(broken)} of {len(probes)} provoked requests break the XOR:\n{detail}\n       "
                  f"the first broken one is: {broken[0]}. 'gpu' XOR a non-empty gpu_skip_reason is "
                  "the whole contract, and a caller who cannot tell a decline from a choice has "
                  "no way to know their request was ignored")


def block_a_decline_returns_the_cpu(core, cases) -> None:
    name = "a declined call returns the cpu numbers bit for bit"
    worst, checked, broken = 0.0, 0, []
    for label, energies, report, cpu in cases:
        if not reason_of(report):
            # Not a decline, so there is no fallback to compare. Counted
            # separately and named, so "nothing was checked" can never read as
            # "everything was checked".
            continue
        checked += 1
        same = bool(np.array_equal(energies, cpu))
        if not same:
            d = float(np.max(np.abs(energies - cpu))) if energies.size else float("inf")
            broken.append(f"{label}: differs from the cpu answer by {d:.3e}")
        elif energies.size:
            worst = max(worst, float(np.max(np.abs(energies))))
    if checked == 0:
        bad(name, "none of the provoked requests was reported as a decline, so the CPU fallback "
                  "was never exercised and this block measured nothing. Either every one of them "
                  "really did run on the GPU -- in which case the fixtures are wrong -- or the "
                  "build is answering a decline with no reason, which is the failure the row "
                  "above reports and which leaves the fallback untested")
    elif broken:
        bad(name, f"{len(broken)} of {checked} declines did not return the CPU's answer: {broken}")
    else:
        ok(name, f"all {checked} declines returned the CPU's energies exactly, largest magnitude "
                 f"among them {worst:.3e}. A fallback that quietly reorders the caller's "
                 "conformations is not a fallback")


def block_no_magnitude_returns_an_infinity(core, ligand, maps, scoring_constant) -> None:
    name = "no coordinate magnitude returns a non-finite energy from a path that says it ran"
    lines, offenders = [], []
    for x in F32_PROBES:
        confs = np.zeros((1, ligand.num_dof), dtype=np.float64)
        confs[0, 0] = x
        cpu, cpu_report = ask(core, ligand, maps, confs, False)
        gpu, gpu_report = ask(core, ligand, maps, confs, True)
        held, why = xor_holds(gpu_report, True)
        if not held:
            offenders.append(f"{x:.3e} A: the request itself broke the XOR: {why}")
        cpu_finite = bool(np.all(np.isfinite(cpu)))
        declined = reason_of(gpu_report) != ""
        if declined:
            if not np.array_equal(gpu, cpu):
                offenders.append(f"{x:.3e} A: declined, but the fallback is not the cpu answer")
        elif not cpu_finite:
            # The CPU's own f64 arithmetic overflowed -- 1e308 A times a penalty
            # of 1000 is past f64::MAX. That is the CPU's business and it is not
            # a GPU divergence, so it is reported and not scored. Scoring it
            # would make this row fail for a reason that has nothing to do with
            # the `f32` grid the row is about.
            lines.append(f"       {x:>9.3e} A  cpu={cpu[0]:>12.4e}  gpu={gpu[0]:>12.4e}  "
                         f"backend={gpu_report.get('backend')!r}  reason={reason_of(gpu_report) or '(none)'}"
                         "   <- the cpu's own f64 penalty overflowed; not a gpu divergence, not scored")
            continue
        else:
            # The load-bearing clause, and the one this row exists for: the CPU
            # had a finite number here, so a non-finite one from the GPU is a
            # divergence of unbounded size. It is returned, ranked, and reported
            # as a normal energy.
            if not np.all(np.isfinite(gpu)):
                offenders.append(f"{x:.3e} A: the cpu returned {cpu[0]:e} and the gpu returned "
                                 f"{gpu[0]:e} with backend={gpu_report.get('backend')!r} and no "
                                 "decline reason -- a non-finite answer where the cpu had a finite one")
        lines.append(
            f"       {x:>9.3e} A  cpu={cpu[0]:>12.4e}  gpu={gpu[0]:>12.4e}  "
            f"backend={gpu_report.get('backend')!r}  reason={reason_of(gpu_report) or '(none)'}"
        )
    detail = "\n".join(lines)
    if not offenders:
        ok(name, f"the ceiling on this path is set by the penalty, not the coordinate: "
                 f"OUT_OF_BOX_PENALTY={scoring_constant:g} is applied in f32 inside the kernel, so "
                 f"the representable range ends below f32::MAX/{scoring_constant:g} rather than at "
                 f"f32::MAX. No magnitude produced a non-finite energy where the cpu had a finite "
                 f"one; the magnitudes that the cpu itself cannot represent are named as such and "
                 f"are not scored against the GPU:\n{detail}")
    else:
        bad(name, f"{len(offenders)} clause(s) broke across {len(F32_PROBES)} magnitudes: "
                  f"{offenders}\n{detail}")


def block_the_guard_is_what_declined(core, status, big_report) -> None:
    name = "the f32 narrowing guard is what declined the oversized coordinate, not a missing adapter"
    if status["compiled"]:
        reason = reason_of(big_report)
        # The point of the row: on a build with the feature, the guard sits in
        # front of the adapter probe, so a coordinate the f32 grid cannot hold is
        # declined for *that* reason even on a machine that has a card in it.
        if reason and "f32" in reason:
            ok(name, f"compiled=True and a 1e308 A coordinate came back declined with {reason!r}: "
                     "the guard is in front of the adapter probe, so the decline names the grid "
                     "and not the hardware")
        else:
            bad(name, f"compiled=True but the oversized coordinate came back as {big_report!r}. "
                      "Either the guard did not fire -- in which case the f32 overflow is back -- "
                      "or it fired behind the adapter probe, so a caller cannot tell a coordinate "
                      "the grid cannot hold from a machine with no card in it")
    else:
        skip(name, "the extension was compiled without the `gpu` feature, so gpu_compiled() is "
                   "the compile-time constant cfg!(feature = \"gpu\") and try_gpu_population is "
                   "not in this binary at all. The narrowing guard lives inside that function, so "
                   "there is nothing in this build to exercise: no coordinate, of any magnitude, "
                   "reaches it. This is a build-identity coupling rather than a hardware absence -- "
                   "the machine may well have an adapter, and this binary could not have asked. "
                   "What this build *can* still say about the f32 path is the row above, and that "
                   "row has its own verdict printed above this line")


def block_the_kernel_was_compared(core, status, adapter) -> None:
    name = "the kernel answered and the two backends were compared on this machine"
    if status["available"]:
        ok(name, f"an adapter was opened ({adapter}) and a GPU request reached the kernel, so the "
                 "two backends were compared here rather than inferred. What they agree to is "
                 "gpu_cpu_parity_check.py's question and not this file's; this row exists to record "
                 "that the comparison was possible on this machine at all")
    else:
        skip(name, f"no adapter here (compiled={status['compiled']}, "
                   f"available={status['available']}), so no GPU request reached the kernel and no "
                   "GPU-vs-CPU difference was computed. None is reported. This line is an absence, "
                   "not a result, and it is on the total so that it cannot be mistaken for agreement")


def block_this_predicate_can_go_red() -> None:
    name = "this file's own xor predicate goes red on a report that breaks it"
    ran = {"backend": "gpu", "gpu_skip_reason": None, "adapter": "RTX 3050", "num_conformations": 1}
    declined = {"backend": "cpu", "gpu_skip_reason": "no usable GPU adapter",
                "adapter": None, "num_conformations": 1}
    # `(asked_for_gpu, report, expected)` for each case. The two legal shapes
    # and then one violation per clause, each breaking exactly one thing. A
    # predicate that cannot fail is not a predicate, and a gate built on one
    # reports a contract that was never checked -- which is why the phantom case
    # below is a `use_gpu=False` request carrying a reason, and why the request
    # is an argument rather than a background fact.
    cases = [
        (True, ran, True, "asked for the gpu and got it"),
        (True, declined, True, "asked for the gpu and was told why not"),
        (False, {"backend": "cpu", "gpu_skip_reason": None, "adapter": None,
                 "num_conformations": 1}, True, "never asked, ran on the cpu"),
        (True, {"backend": "cpu", "gpu_skip_reason": None, "adapter": None,
                "num_conformations": 1}, False,
         "a decline with no reason (the defect this repository repaired)"),
        (False, declined, False,
         "a phantom fallback on a request that never asked for the gpu"),
        (True, {"backend": "gpu", "gpu_skip_reason": "no usable GPU adapter", "adapter": "RTX 3050",
                "num_conformations": 1}, False, "both at once -- ran and declined"),
        (True, {"backend": "vulkan", "gpu_skip_reason": None, "adapter": None,
                "num_conformations": 1}, False, "a backend that names nothing"),
    ]
    missed = []
    for asked, report, want, label in cases:
        held, why = xor_holds(report, asked)
        if held != want:
            missed.append(f"{label}: expected {'pass' if want else 'fail'}, got "
                          f"{'pass' if held else 'fail'} ({why})")
    n_legal = sum(1 for _a, _r, w, _l in cases if w)
    n_broken = len(cases) - n_legal
    if missed:
        bad(name, f"{len(missed)} of {len(cases)} synthetic requests were judged wrongly, so the "
                  f"predicate this file runs its real rows with is not the one it claims: {missed}")
    else:
        ok(name, f"the same predicate the real rows use was handed {len(cases)} hand-built "
                 f"requests and got every one of them right: {n_legal} legal shapes pass and each "
                 f"of the {n_broken} single-clause violations fails. So a green XOR row means the "
                 "reports satisfied the contract, not that the check cannot notice")


def block_pinned_total(npass: int, nfail: int, nskip: int) -> None:
    name = "the_pinned_total_counts_results_and_not_only_passes"
    # `+ 1` is this check itself: already in RESULTS, not yet in the arguments.
    got = npass + nfail + nskip + 1
    if got == EXPECTED_CHECKS:
        ok(name, f"{got} results against a pin of {EXPECTED_CHECKS}: {npass} passed, {nfail} failed, "
                 f"{nskip} skipped -- three numbers, printed separately and never added together. "
                 "The same total on a machine with an adapter and on a runner without one is the "
                 "property that makes a skip a result")
    else:
        bad(name, f"{got} results against a pin of {EXPECTED_CHECKS} ({npass} passed, {nfail} "
                  f"failed, {nskip} skipped). Fewer means a block did not run; more means a block "
                  "grew that nobody pinned")


# --- run --------------------------------------------------------------------


def main() -> int:
    od, core, where = import_engine()
    status = core.gpu_status()
    penalty = out_of_box_penalty()
    reasons = decline_reasons()
    wg = workgroup_size()

    section("which binary this run measured")
    print(f"opendocking = {od.__file__}")
    print(f"engine      = {core.engine_version()} from {where}")
    print(f"gpu_status  = {status}")
    print(f"backends    = {core.available_backends()}")

    section("what the sources declare, read rather than transcribed")
    print(f"OUT_OF_BOX_PENALTY = {penalty:g}  (dock-core/src/search/mod.rs)")
    print(f"workgroup_size     = {wg}      (dock-core/src/gpu/energy.wgsl)")
    print(f"out-of-box term    = {shader_charges_the_hosts_summed_violation()}      (energy.wgsl, "
          "the shape the safe-coordinate bound below assumes; READ AND PRINTED, not asserted -- "
          "cargo test binds it in dock-core/src/gpu/mod.rs)")
    print(f"decline sites      = {len(reasons)}      (grep of `reason: \"` in search/mod.rs; not "
          "written down anywhere, because a closed list goes stale)")
    # Printed one per line and never totalled into anything.
    for r in reasons:
        print(f"    {r!r}")
    print(f"f32-safe |p| for a 1-atom ligand = {f32_safe_coordinate(penalty, 1):.6e} A "
          f"(= f32::MAX / ({penalty:g} * {AXES} * 1)); ARITHMETIC, not a measurement")

    block_gpu_status_shape(core)
    block_available_implies_compiled(core)
    block_backends_advertise_only_a_probed_gpu(core)

    box = core.GridBox.from_center_size(BOX_CENTER, BOX_SIZE)
    maps = ligand = None
    section("fixtures")
    try:
        receptor = core.Receptor.from_pdbqt(EXAMPLES / RECEPTOR)
        maps = receptor.precalculate(box, SCORING, SPACING)
        ligand = core.Ligand.from_pdbqt(EXAMPLES / LIGAND)
    except Exception as exc:  # noqa: BLE001 -- a fact about this run, not a crash
        bad("the fixtures this file needs could be built", f"{type(exc).__name__}: {exc}")

    if maps is None or ligand is None:
        for name, why in (
            ("a gpu request either ran on the gpu or said why it did not",
             "the maps and ligand were never built, so no request could be made"),
            ("a declined call returns the cpu numbers bit for bit",
             "the maps and ligand were never built, so no request could be made"),
            ("no coordinate magnitude returns a non-finite energy from a path that says it ran",
             "the maps and ligand were never built, so no coordinate could be placed"),
            ("the f32 narrowing guard is what declined the oversized coordinate, not a missing adapter",
             "the maps and ligand were never built, so no coordinate could be placed"),
            ("the kernel answered and the two backends were compared on this machine",
             "the maps and ligand were never built, so no request could be made"),
        ):
            skip(name, f"this run could not build its own fixtures, so {why}. The absence is "
                   "recorded rather than absorbed into a smaller total")
    else:
        print(f"receptor/ligand = {RECEPTOR} / {LIGAND}, {ligand.num_atoms} atoms, "
              f"{ligand.num_dof} dof, box {BOX_SIZE} at {SPACING} A")

        section("what ran, and what it said")
        confs = np.zeros((4, ligand.num_dof), dtype=np.float64)
        confs[:, 0] = 12.0
        _, cpu_report = ask(core, ligand, maps, confs, False)
        _, gpu_report = ask(core, ligand, maps, confs, True)
        print(f"       use_gpu=False -> {cpu_report}")
        print(f"       use_gpu=True  -> {gpu_report}")
        block_backend_answers_only_what_ran(
            core, [("use_gpu=False", cpu_report), ("use_gpu=True", gpu_report)])

        section("five provoked requests: four that asked for the GPU, one that did not")
        big = np.zeros((1, ligand.num_dof), dtype=np.float64)
        big[0, 0] = 1.0e308
        oversized = chain_ligand(wg + 1)
        probes, cases = [], []
        for label, asked, lig, pop in (
            ("a normal gpu request", True, ligand, confs),
            (f"a {wg + 1}-atom batch (over the workgroup)", True, oversized,
             np.zeros((1, oversized.num_dof), dtype=np.float64)),
            ("an empty population", True, ligand, np.zeros((0, ligand.num_dof), dtype=np.float64)),
            ("a coordinate the f32 grid cannot hold", True, ligand, big),
            # The other half of the XOR, and the half a one-sided check can
            # never see: a caller who did NOT ask must not be handed a fallback
            # that did not happen. Provoked with the same batch as the first row
            # so the two differ in exactly one thing -- the request.
            ("the same batch, but use_gpu=False", False, ligand, confs),
        ):
            energies, report = ask(core, lig, maps, pop, asked)
            probes.append((label, asked, report))
            if asked:
                # The baseline is **the same batch answered on the CPU**, not
                # one shared population: "a decline returns the CPU's numbers
                # bit for bit" is a claim about one request and its own CPU
                # twin, and a 65-atom batch compared against a 19-atom one
                # would be a claim about nothing.
                twin, _twin_report = ask(core, lig, maps, pop, False)
                cases.append((label, energies, report, twin))
        block_gpu_request_ran_or_said_why(core, probes)
        block_a_decline_returns_the_cpu(core, cases)

        section("the f32 boundary, as a property and not as a threshold")
        block_no_magnitude_returns_an_infinity(core, ligand, maps, penalty)

        section("the two rows that need something this build may not have")
        adapter = "(not reported on this run)"
        if status["available"]:
            probe_report: dict = {}
            core.evaluate_conformations(ligand, maps, confs, SCORING, True, probe_report)
            adapter = probe_report.get("adapter") or "(unnamed adapter)"
        # Asked for before the block that reads it: the guard row needs the
        # oversized coordinate's report by name, and on a build without the
        # feature that report is the evidence for *why* the row skipped.
        _, big_report = ask(core, ligand, maps, big, True)
        print(f"       1e308 A coordinate -> {big_report}")
        block_the_guard_is_what_declined(core, status, big_report)
        block_the_kernel_was_compared(core, status, adapter)

    section("this file against itself")
    block_this_predicate_can_go_red()

    section("the pin")
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    block_pinned_total(npass, nfail, nskip)
    # Re-read, and not reused: the three tags have to cover the pin's own result
    # as well, or the summary prints a triple that does not add up to the total
    # printed beside it. `block_pinned_total` adds its `+ 1` because its caller
    # samples the counts *before* it records; the line below samples them after.
    # Re-reading also means a pin that fails is counted as the failure it is, so
    # a count that is wrong exits 1 rather than slipping out as a count mismatch.
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {npass} passed, {nfail} failed, {nskip} skipped, {len(RESULTS)} results "
          f"(expected {EXPECTED_CHECKS})")
    if nfail:
        print(f"RESULT: DEFECT (exit {EXIT_DEFECT}) -- {nfail} of {len(RESULTS)} recorded results "
              "failed; every failure above names the clause that broke")
        return EXIT_DEFECT
    if len(RESULTS) != EXPECTED_CHECKS:
        return EXIT_COUNT_MISMATCH
    if maps is None:
        print(f"RESULT: UNMEASURABLE (exit {EXIT_UNMEASURABLE}) -- this run's own fixtures were "
              "never built, so nothing above is a measurement of the engine")
        return EXIT_UNMEASURABLE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
