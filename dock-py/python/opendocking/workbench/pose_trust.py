"""Can *this* pose's numbers be trusted? The verdict, bound to one run.

`scripts/result_trust.py` is a pure verdict module: it reads no file, imports no
engine and starts no process, so it can be called from here, from a report
script and from a gate with the same answer. It takes a **flat dict of numbers
plus thresholds**, all optional, and folds them into one of three answers --
`yes`, `no`, `unknown` -- through the engine's own published contracts. This
module is the other half of that: it is the workbench's binding of those
numbers to the objects a run really carries, and it is the only place in the
package where a verdict is produced.

Four things about the binding are the whole design, and each of them is a
defect it exists to prevent.

**The key names are not typed, they are looked up.** `result_trust` publishes
the names each family reads through `schema_for(kind)`, and this module asks.
A caller that spells `grad_l2` from memory and the module that reads `grad_norm`
would produce a panel that answers `unknown` with complete confidence on every
pose in the repository -- the polite word that buries a defect. So the names
live in one table, `_READERS`, and every one of them is checked against
`schema_for("pose")` before it is used: a rename on the other side stops this
panel with a sentence naming the new schema instead of quietly emptying it.

**A value that no object carries is left out, not defaulted.** `_READERS`
returns `None` for "this result did not measure it", and `None` is dropped from
the dict rather than written as `0.0`. That is the whole difference between
`unmeasured` and `holds`, and it is why this module has no default for
anything.

**The remedy is not optional.** Every contract that `fails` carries a sentence
saying what would change the answer and what it costs, and this panel prints it.
A warning light with no dashboard is not actionable, and the reader of a
docking is the person who has to decide what to do next.

**A pose from a file has no verdict, and that is a state, not a gap.** A pose
file carries coordinates. The engine's decomposition and its pose gradient are
keyed on the conformation vector of a run in memory, and there is no inverse
from coordinates back to one -- recovering it means solving the inverse
kinematics, which is a different program with its own error. So a file pose
gets `unknown` from all four contracts, with the reason spelled out, which is
this panel's honest answer: *I have no opinion about this pose, and here is
exactly what I would have to measure to have one.* It is not a dead end dressed
up as a feature; the action that changes it is named, and it is the button in
the panel above.

What this panel costs, measured rather than asserted: reading one pose's
gradient took **7.7 us**, folding it into a verdict took **13.7 us**, and the
whole of `describe_pose` -- binding, formatting, four rows -- took **77.8 us**
(500 to 2000 iterations each, on 1crn + biotin, 11 degrees of freedom,
`F:\\python310`, this machine). So the whole panel is a read of arrays the run
already holds: no tabulation, no engine call, no thread, and nothing here that
could block the GUI thread the way the term maps do. The one value it does not
compute itself, the out-of-box penalty, is reused from the energy breakdown
above it *for the same pose index* -- recomputing it would mean a second engine
call for a number already on screen, and a panel that disagreed with the panel
above it.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: The optimiser's own declared stopping tolerance, `LbfgsConfig::
#: gradient_tolerance` in `dock-core/src/search/lbfgs.rs`. The engine does not
#: expose it to Python -- there is no binding for it in `opendocking/core.py` --
#: so this is a transcription, and a transcription is only acceptable if
#: something checks it. `scripts/pose_trust_check.py` reads that Rust file and
#: asserts this number is the engine's own, so an edit to the optimiser's
#: stopping test turns the gate red rather than quietly changing every verdict
#: this panel prints.
#:
#: The same number is transcribed in `examples/audit_poses.py` (`GRADIENT_
#: TOLERANCE`), which is the authority this one is checked against as well.
GRADIENT_TOLERANCE = 1e-4
GRADIENT_TOLERANCE_SOURCE = "dock-core/src/search/lbfgs.rs, LbfgsConfig::gradient_tolerance"

#: The three answers, and the word each one is printed as. The word is the
#: primary channel and the colour is the second one: the verdict has to be
#: readable with all colour removed, because a reader who cannot tell `no` from
#: `unknown` will read a "I don't know" as a pass, and that is the same shape as
#: a missing measurement wearing a green light.
STATE_WORDS: dict[str, str] = {
    "yes": "holds",
    "no": "fails",
    "unknown": "unmeasured",
}

#: Two attributes per state as well as a colour, because colour alone fails for
#: a red-green deficiency and a greyscale screenshot. `no` is bold, `unknown`
#: is italic, and -- the one that matters -- `unknown` is **not** the pass
#: colour and does not borrow the pass weight. On the light panel background
#: these are all dark enough to read.
STATE_COLOURS: dict[str, str] = {
    "yes": "#1a7f37",
    "no": "#b3261e",
    "unknown": "#8a5300",
}
STATE_FONT: dict[str, str] = {
    "yes": "font-weight: normal;",
    "no": "font-weight: bold;",
    "unknown": "font-style: italic;",
}

#: What the panel says when there is no pose, and when there is no verdict to
#: show. The second is a distinct string on purpose: "no pose selected" and
#: "the verdict module is not installed" are different facts, and one widget
#: showing both would make the second look like the first.
NO_POSE = "no pose selected"
NO_POSE_NOTE = "Select a pose to see whether its own numbers can be trusted."


class PoseVerdictUnavailable(RuntimeError):
    """The verdict cannot be produced for this pose, and here is why.

    Three different refusals share this type, and each of them has to be
    distinguishable in the panel's text rather than collapsed into one message:
    the verdict module is not importable, the schema no longer has a key this
    panel binds, and the numbers offered belong to another pose.
    """


_MODULE: Any = None
_MODULE_ERROR: str = ""


def verdict_module() -> Any:
    """The `result_trust` module, imported once, or a stated reason it is absent.

    Imported by bare name because that is what it is: a pure module in
    `scripts/`, not part of the installed package. In this source tree, and in
    a checkout of the release, `scripts/` is on the path of anything that puts
    the repository root there; in an installed wheel it is not, and the panel
    then says so in words rather than showing a verdict it cannot compute.
    Whether `result_trust` should move into `opendocking/` is a packaging
    decision that belongs to the module's owner, not to this panel.
    """
    global _MODULE, _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE
    if _MODULE_ERROR:
        raise PoseVerdictUnavailable(_MODULE_ERROR)
    try:
        _MODULE = importlib.import_module("result_trust")
    except Exception as exc:  # noqa: BLE001 - any import failure is the same fact
        _MODULE_ERROR = (
            "the verdict module could not be imported "
            f"({type(exc).__name__}: {exc}). It is a pure module in "
            "`scripts/result_trust.py` and is not part of the installed "
            "package, so a wheel-installed workbench has no verdict to show. "
            "Nothing here was measured, so nothing here passes."
        )
        raise PoseVerdictUnavailable(_MODULE_ERROR) from exc
    return _MODULE


def _reset_module_cache() -> None:
    """Forget the import. For the gate's import-failure test, and nothing else."""
    global _MODULE, _MODULE_ERROR
    _MODULE = None
    _MODULE_ERROR = ""


@dataclass(frozen=True)
class ContractRow:
    """One contract, as the panel prints it.

    `measured` and `threshold` are **strings**, formatted here rather than
    handed to the widget, so that what the reader reads and what the gate parses
    are the same characters. `None` in either is the word "not measured", never
    a zero: a zero here would be indistinguishable from a measured zero, which
    is exactly the confusion `inside the box` is about.
    """

    name: str
    state: str
    measured: str
    threshold: str
    because: str
    remedy: str
    remedy_kind: str
    family: str
    source: str

    @property
    def pair(self) -> str:
        """`measured / threshold`, or the words for there being no pair.

        One word rather than "not measured (measured, not threshold)": the
        column's job is to show which numbers decided this contract, and a row
        with neither of them says that in one line. Which side is missing is
        the contract's own `source` in the row's tooltip -- `result.descent_
        width` names the absent value rather than the present one.
        """
        if self.measured == NOT_MEASURED or self.threshold == NOT_MEASURED:
            return NOT_MEASURED
        return f"{self.measured} / {self.threshold}"

    @property
    def word(self) -> str:
        return STATE_CONTRACT_WORDS.get(self.state, self.state)


#: `unmeasured` is the module's third state; a row that cannot say it is not
#: being rendered, it is being reclassified.
STATE_CONTRACT_WORDS = {"holds": "holds", "fails": "fails", "unmeasured": "unmeasured"}
NOT_MEASURED = "not measured"


def _fmt(value: Any) -> str:
    """One number as the panel prints it, or the words for its absence.

    `%.6g` rather than a fixed-point format for the same reason
    `energy_terms.format_kcal` uses exponent form below its threshold: a
    fixed-point format prints a tolerance of 1e-4 as `0.0001` and a penalty of
    1e-9 as `0.0000`, and the second is a claim that the penalty is zero.
    """
    if value is None:
        return NOT_MEASURED
    try:
        f = float(value)
    except (TypeError, ValueError):
        return NOT_MEASURED
    return f"{f:.6g}"


@dataclass(frozen=True)
class PoseVerdict:
    """One pose's verdict, and the numbers behind it."""

    trust: str
    pose_index: int
    num_poses: int
    summary: str
    rows: tuple[ContractRow, ...]
    supplied: tuple[str, ...]
    absent: tuple[str, ...]
    misspelled: tuple[str, ...] = ()

    @property
    def described_pose(self) -> str:
        """The one line that says which pose these numbers are about."""
        return f"pose {self.pose_index + 1} of {self.num_poses}"

    @property
    def state_word(self) -> str:
        return STATE_WORDS.get(self.trust, self.trust)

    def headline(self) -> str:
        """The one line a reader reads first, state first and in words.

        The state word comes before the reason so that a reader who stops
        after one word still stops on the state rather than on a number.
        """
        return f"{self.described_pose} -- {self.trust.upper()}: {self.summary}"

    def stylesheet(self) -> str:
        """The colours and the font for this verdict, in one stylesheet."""
        return stylesheet_for(self.trust)

    def failing_rows(self) -> tuple[ContractRow, ...]:
        return tuple(r for r in self.rows if r.state == "fails")

    def unmeasured_rows(self) -> tuple[ContractRow, ...]:
        return tuple(r for r in self.rows if r.state == "unmeasured")

    def explanation(self) -> str:
        """The failing contracts' reasons and remedies, and what was not measured.

        This is the actionable half of the panel and the reason a reader can do
        anything with a `no`: every failing contract prints its `because` and
        its remedy with the remedy's own kind, because "re-prepare the input"
        and "only the engine could change it" are not the same kind of news.
        Unmeasured contracts print their reason too -- an absent measurement
        that is not named reads as a row somebody forgot.
        """
        parts: list[str] = []
        for row in self.failing_rows():
            kind = f" ({row.remedy_kind})" if row.remedy_kind else ""
            parts.append(f"{row.name}: {row.because}  To change it{kind}: "
                         f"{row.remedy or 'NO REMEDY IS ATTACHED'}")
        for row in self.unmeasured_rows():
            parts.append(f"{row.name}: {row.because}")
        return "  ".join(parts)


def stylesheet_for(state: str) -> str:
    """The stylesheet for one of the three states.

    Kept here, next to the words and the colour, so that a reader can see the
    three treatments side by side in one place, and so the gate can assert on
    them without a widget. `unknown` gets its own colour and its own font and
    is given the pass weight by nothing: if this function ever returns the
    stylesheet for `yes` when asked for `unknown`, a reader cannot tell.
    """
    colour = STATE_COLOURS.get(state)
    font = STATE_FONT.get(state)
    if colour is None or font is None:
        # An unknown state renders as `unknown`, because inventing a fourth
        # treatment is how a new state ends up looking like a pass.
        colour = STATE_COLOURS["unknown"]
        font = STATE_FONT["unknown"]
    return f"color: {colour}; {font}"


# --------------------------------------------------------------------------
# The binding: which object each value is read from
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Binding:
    """One value, and the object it is read from.

    `reader` returns `None` for "the object does not carry this", which is the
    honest answer and is dropped from the dict rather than defaulted.
    """

    reader: Callable[[Any, int, float | None], float | None]
    origin: str


def _read_gradient(result: Any, _index: int, _penalty: float | None) -> float | None:
    """`|grad|_2` of pose ``index``, from the result's own stored gradient.

    The result already holds this array -- the search computed it and kept it
    -- so reading it is a lookup, not a call into the engine. `None` is
    returned for a result whose gradient was never measured, and `None` is
    **not** the same answer as a zero vector: a zero gradient would assert that
    the pose is stationary, which is the most damaging wrong answer here.
    """
    if result is None:
        return None
    gradient = result.pose_gradient(int(_index))
    if gradient is None:
        return None
    total = 0.0
    for value in gradient:
        v = float(value)
        total += v * v
    return total ** 0.5


def _read_tolerance(_result: Any, _index: int, _penalty: float | None) -> float | None:
    """The optimiser's declared stopping tolerance. See `GRADIENT_TOLERANCE`."""
    return float(GRADIENT_TOLERANCE)


def _read_penalty(_result: Any, _index: int, penalty: float | None) -> float | None:
    """The pose's out-of-box penalty, reused from the energy breakdown.

    Only the breakdown for *this* pose may be used, and `describe_pose`
    refuses a penalty that came with another pose's index rather than showing
    it: a stale penalty is a number about a different molecule, and the panel
    that shows one is worse than a panel that shows nothing.
    """
    return None if penalty is None else float(penalty)


#: The names are spelled here and **nowhere else**, and every one of them is
#: checked against `result_trust.schema_for("pose")` at bind time. So this
#: table is the one place a rename can be absorbed, and it cannot absorb one
#: silently: `describe_pose` raises and the panel says what the new schema is.
_READERS: dict[str, _Binding] = {
    "grad_l2": _Binding(_read_gradient, "DockingResult.pose_gradient(i)"),
    "gradient_tolerance": _Binding(
        _read_tolerance, f"LbfgsConfig::gradient_tolerance ({GRADIENT_TOLERANCE_SOURCE})"
    ),
    "out_of_box_penalty": _Binding(
        _read_penalty, "score_conformation_terms' out_of_box_penalty, for this pose"
    ),
}

#: What the panel can never supply, and why. Named rather than left implicit,
#: because a reader looking at a panel that says `unmeasured` on two of four
#: rows will ask what would have to be measured, and the answer should be one
#: sentence rather than an archaeology project.
#:
#: * `descent_width` -- the engine does not report the width of a pose's
#:   descent region. `examples/audit_poses.py` measures it with its own eight
#:   step ladder, which is a *bracket* between two sampled steps and not the
#:   engine's number, and it is deliberately not exposed: the descent width is a
#:   property of the line search that produced a step, not a function of the
#:   final pose.
#: * `finest_step` -- **the engine has no step ladder to read.** The Armijo
#:   search in `dock-core/src/search/lbfgs.rs` backtracks continuously
#:   (`alpha *= 0.5`, stopping at an absolute floor of 1e-12), so there is no
#:   `_LINE_SEARCH_STEPS[0]` to take. A number here would be invented, and an
#:   invented threshold is worse than a missing one.
#: * `closest_surface_distance` / `support_zero_edge` -- the engine reports
#:   neither a per-pair surface distance nor the f32 underflow edge of the Shape
#:   term, so the pair is absent on both sides.
NEVER_SUPPLIED: dict[str, str] = {
    "descent_width": (
        "the engine does not report the width of a pose's descent region, and "
        "it is a property of the line search that produced a step rather than "
        "of the final pose"
    ),
    "finest_step": (
        "the engine's Armijo line search backtracks continuously rather than "
        "stepping down a fixed ladder, so there is no finest step to read and "
        "naming one would be inventing the threshold"
    ),
    "closest_surface_distance": (
        "the engine reports no per-pair surface distance for a pose"
    ),
    "support_zero_edge": (
        "the f32 underflow edge of the Shape term is not exposed to Python"
    ),
}


def pose_values(
    result: Any,
    pose_index: int,
    out_of_box_penalty: float | None = None,
    penalty_pose_index: int | None = None,
) -> dict[str, Any]:
    """The flat dict `result_trust.verdict_pose` reads, built from ``result``.

    ``out_of_box_penalty`` may only be supplied together with the index it
    belongs to. A penalty without its index, or with an index that is not this
    pose, is refused: it is the one value this panel does not compute, and it
    is therefore the one a stale field would poison silently.

    Keys no object carried are simply absent. That is the module's
    `unmeasured`, and it is not an error: a pose read from a file has no
    gradient, and the honest answer to "is this pose trustworthy" for a file
    pose is "not measured", said in words.
    """
    module = verdict_module()
    schema = set(module.schema_for("pose"))
    index = int(pose_index)
    if index < 0:
        raise PoseVerdictUnavailable(
            f"pose index {index} is not a pose; nothing was measured"
        )
    if out_of_box_penalty is not None and penalty_pose_index is None:
        raise PoseVerdictUnavailable(
            "an out-of-box penalty was offered without saying which pose it "
            "belongs to, and this panel will not show a number it cannot name a "
            "molecule for"
        )
    if (out_of_box_penalty is not None
            and int(penalty_pose_index) != index):
        raise PoseVerdictUnavailable(
            f"the out-of-box penalty offered is pose "
            f"{int(penalty_pose_index) + 1}'s, and the panel is describing pose "
            f"{index + 1}: refusing rather than showing one pose's number under "
            f"another's name"
        )
    values: dict[str, Any] = {}
    num_poses = 0
    if result is not None:
        num_poses = int(result.num_poses)
        if not 0 <= index < num_poses:
            raise PoseVerdictUnavailable(
                f"pose {index + 1} is not one of the {num_poses} poses this run "
                f"reported"
            )
    values["label"] = f"pose {index + 1} of {num_poses}" if num_poses else (
        f"pose {index + 1} (from a pose file)"
    )
    for name in sorted(_READERS):
        if name not in schema:
            raise PoseVerdictUnavailable(
                f"the verdict module no longer reads {name!r} for a pose: "
                f"schema_for('pose') is {sorted(schema)}. The panel is not going "
                f"to guess a spelling, because a wrong one answers 'unmeasured' "
                f"for every pose in the repository"
            )
        value = _READERS[name].reader(result, index, out_of_box_penalty)
        if value is not None:
            values[name] = value
    return values


def _rows_from_verdict(verdict: Any) -> tuple[ContractRow, ...]:
    """The module's contracts, in its order, named and formatted."""
    rows = []
    for contract in verdict.contracts:
        rows.append(
            ContractRow(
                name=contract.name,
                state=contract.state,
                measured=_fmt(contract.measured),
                threshold=_fmt(contract.threshold),
                because=contract.because,
                remedy=contract.remedy,
                remedy_kind=contract.remedy_kind,
                family=contract.family,
                source=contract.source,
            )
        )
    return tuple(rows)


def describe_pose(
    result: Any,
    pose_index: int,
    out_of_box_penalty: float | None = None,
    penalty_pose_index: int | None = None,
) -> PoseVerdict:
    """The verdict for pose ``pose_index`` of ``result``, and nothing else's.

    One call to the module, and every row on screen comes out of that one
    call: a panel whose rows came from different calls is a panel whose rows can
    disagree, and a row that disagrees with the row above it is not a number a
    reader can act on.
    """
    module = verdict_module()
    values = pose_values(
        result, pose_index,
        out_of_box_penalty=out_of_box_penalty,
        penalty_pose_index=penalty_pose_index,
    )
    verdict = module.verdict_pose(values)
    schema = set(module.schema_for("pose"))
    supplied = tuple(sorted(k for k in values if k != "label"))
    absent = tuple(sorted(k for k in schema if k not in values))
    num_poses = int(result.num_poses) if result is not None else 0
    return PoseVerdict(
        trust=verdict.trust,
        pose_index=int(pose_index),
        num_poses=num_poses,
        summary=verdict.summary,
        rows=_rows_from_verdict(verdict),
        supplied=supplied,
        absent=absent,
        misspelled=tuple(verdict.misspelled),
    )


def describe_file_pose(pose_index: int, num_poses: int = 0) -> PoseVerdict:
    """A pose that came from a file: no opinion, and the reason in words.

    Not a special case of `describe_pose` with ``result=None`` on purpose. A
    caller that passes no result by accident -- a lost run, a pose table
    cleared -- is a *bug*, and this function is for the one situation that is
    not: a pose file, which structurally cannot carry what the contracts read.
    The window knows which of the two it has, and says so.
    """
    module = verdict_module()
    label = (f"pose {int(pose_index) + 1} of {int(num_poses)} (from a pose file)"
             if num_poses else f"pose {int(pose_index) + 1} (from a pose file)")
    verdict = module.verdict_pose({"label": label})
    schema = set(module.schema_for("pose"))
    return PoseVerdict(
        trust=verdict.trust,
        pose_index=int(pose_index),
        num_poses=int(num_poses),
        summary=(
            f"{label}: these poses were read from a file, and a pose file "
            f"carries coordinates rather than a conformation -- the engine's "
            f"gradient and its out-of-box penalty are properties of a run in "
            f"memory, and nothing here was measured. {verdict.summary}"
        ),
        rows=_rows_from_verdict(verdict),
        supplied=(),
        absent=tuple(sorted(schema)),
        misspelled=tuple(verdict.misspelled),
    )


def verdict_is_available() -> tuple[bool, str]:
    """Whether a verdict can be produced at all, and the reason if not.

    Asked by the panel before it draws anything, so that a missing module is a
    sentence on screen rather than an empty table that looks like a clean run.
    """
    try:
        verdict_module()
    except PoseVerdictUnavailable as exc:
        return False, str(exc)
    return True, ""


def absent_reasons(names: tuple[str, ...]) -> Mapping[str, str]:
    """Why each of ``names`` is absent, for the panel's own caption."""
    return {name: NEVER_SUPPLIED.get(name, "not reported by any object here")
            for name in names}
