"""Can I trust *this* number? A per-result verdict, as a pure function.

This is not a summary of `docs/SCORING.md` or `docs/LIMITATIONS.md`. Those
documents say what is true of the engine in general; this says what is true of
**the result in your hand**, from the values that result carries, and it says
which measurement each word came from. Every contract here has a number behind
it or it is `unmeasured` -- and `unmeasured` is never allowed to read as "fine".

# The one property that matters

A verdict is one of three states, not two:

    holds         the measurement was made and the contract is satisfied
    fails         the measurement was made and the contract is not
    unmeasured    the result did not carry the value the contract needs

and the whole verdict is `yes` **only if every applicable contract holds**. A
single `unmeasured` makes it `unknown`, which is not a pass. A panel that
cannot say "I don't know" will say "fine" the first time a measurement goes
missing, and a reader has no way to tell that from a real pass -- which is the
same shape as the defect this project keeps finding, one level up from the
code. `claim_result_trust_check` asserts this by *calling* the function with a
value withheld.

# Precedence, stated here rather than only in a test message

**A known failure outranks "I don't know".**

If one contract has `fails` and another has `unmeasured`, the verdict is `no`,
not `unknown`. The reason is that `unknown` is a polite word that buries a
defect: a caller who sees `unknown` stops asking, and the failure that *was*
measured goes with it. A caller who sees `no` still sees the one contract that
failed, and the others are still listed as unmeasured in `verdict.unknown()`.

The corollary, which is the whole design: `unknown` is never the answer when
something is actually known to be wrong. It is the answer when **nothing** has
been found wrong and something could not be looked at. The next person to add a
contract should reach for a new *state* rather than reaching for `unknown` as
the softer-sounding option.

# What a `fails` has to carry, or it is a warning light with no dashboard

A user who reads "this pose is not a stationary point" has no next step. So
every failing contract carries a `remedy`: **what would change the answer, and
what that costs**, in the same one-sentence numbers-in-it form as `because`.

The remedy also declares its `remedy_kind`, because the kinds are not
interchangeable and a flat sentence would pretend they are:

    an input would change it        the data was wrong; re-preparing fixes it
    a parameter would change it     the caller asked for a different setting
    only the engine would change it the measurement is right and the behaviour
                                    is the engine's, so no available setting helps
    nothing available would         the number is what it is; the remedy is to
                                    stop reading it as the thing you wanted

That last kind is a real answer and it is the one that must not be softened.
"Nothing short of changing the optimiser's stopping test changes this" is more
useful than a cheaper-sounding step size that would not work. The gate asserts
this: a remediation that reports a cheaper fix than the real one is a **worse**
defect than a missing remediation, because it sends someone to do the work and
find nothing.

# Why the numbers are supplied rather than looked up

Every threshold below is a parameter. The module reads no file, imports no
engine, and starts no process, so it can be called from a workbench, a report
script, a notebook, or a test with the same answer -- and it can be tested
against a *hypothetical* result, which is the only way to check the failure
directions without breaking the engine on purpose. A function that looked its
own constants up could still do that, but it could not be reasoned about
without running the engine, and every mutation of it would be a mutation of
`dock-core`.

The caller supplies the thresholds *from the engine's own declarations*
(`LbfgsConfig::gradient_tolerance`, `_LINE_SEARCH_STEPS[0]`, and the f32
support edges), so the binding to the engine stays in the gate, where it can be
checked, rather than here, where it would be a second copy.

# What it does not do

It does not decide whether a pose is *good*. It says which of the engine's own
published contracts this particular result satisfies, and it says so with the
number. Whether a non-stationary pose is acceptable for your purpose is a
judgement about your purpose, and putting it here would make this a second
opinion wearing a verdict's clothes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: The three states a contract can be in. Named constants because a bare string
#: in a dataclass is a typo waiting to become a silent pass.
HOLDS = "holds"
FAILS = "fails"
UNMEASURED = "unmeasured"

#: The three answers. `UNKNOWN` is deliberately not a soft YES.
YES = "yes"
NO = "no"
UNKNOWN = "unknown"


#: The four kinds of remedy. Named because a sentence that says "try X" when
#: the truth is "only the engine could" sends someone to do the work and find
#: nothing, and a check cannot tell that apart from a correct sentence unless
#: the kinds are values rather than prose.
REMEDY_INPUT = "an input would change it"
REMEDY_PARAMETER = "a parameter would change it"
REMEDY_ENGINE = "only the engine would change it"
REMEDY_NONE = "nothing available would change it"

#: Kinds of contract failure, by *what sort of thing* is wrong. Two failures
#: can both be `fails` and still not be the same kind of news: a receptor that
#: lost residues is an input problem, an optimiser that stopped early is an
#: engine problem, and a site clipped by a cap is a parameter problem. A verdict
#: that flattened them into one red light would be asking the reader to guess.
FAMILY_DATA = "data"
FAMILY_ENGINE = "engine"
FAMILY_PARAMETER = "parameter"
FAMILY_MODEL = "model"


@dataclass(frozen=True)
class Contract:
    """One published contract, as it applies to one result."""

    name: str
    applies: bool
    state: str
    measured: float | None
    threshold: float | None
    because: str
    source: str
    family: str = FAMILY_ENGINE
    remedy: str = ""
    remedy_kind: str = REMEDY_NONE

    @property
    def blocking(self) -> bool:
        """Whether this contract alone prevents a `yes`."""
        return self.applies and self.state in (FAILS, UNMEASURED)

    @property
    def actionable(self) -> str:
        """The remedy if there is one, else the empty string.

        Deliberately empty for a contract that merely holds: a passing contract
        with a remedy attached is noise, and a reader learns to ignore the field.
        """
        return self.remedy if (self.applies and self.state == FAILS) else ""


@dataclass(frozen=True)
class Verdict:
    kind: str
    trust: str
    contracts: tuple[Contract, ...]
    summary: str
    misspelled: tuple[str, ...] = ()

    def contract(self, name: str) -> Contract | None:
        for c in self.contracts:
            if c.name == name:
                return c
        return None

    def failed(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.contracts if c.applies and c.state == FAILS)

    def unknown(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.contracts if c.applies and c.state == UNMEASURED)


def _num(values: Mapping[str, Any], key: str) -> float | None:
    """The value, or None. `None` is the *only* way a value is absent.

    A zero is a value and must never be confused with a missing one, which is
    the bug this accessor exists to make impossible: `if not values.get(key)`
    treats a measured `0.000000` as unmeasured, and the out-of-box penalty is
    exactly a value whose *meaning* is carried by its being zero.
    """
    v = values.get(key, None)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN is not a measurement


def _unmeasured(name: str, why: str, source: str) -> Contract:
    return Contract(name, True, UNMEASURED, None, None, why, source)


def _stationarity(values: Mapping[str, Any]) -> Contract:
    """Is this pose a stationary point of the field the engine scored?

    The strongest single question a reader can ask about a reported pose, and
    the one this project has measured as *no*: the returned poses carry
    gradients four orders of magnitude above the optimiser's own declared
    tolerance. So a verdict that cannot see `grad_l2` must not answer "yes"
    here -- which is why this contract, like every other, has an `unmeasured`
    state instead of a default.
    """
    g = _num(values, "grad_l2")
    tol = _num(values, "gradient_tolerance")
    if g is None:
        return _unmeasured(
            "stationarity",
            "this result carries no gradient norm, so whether the pose is a "
            "stationary point was not measured -- not assumed either way",
            "result.grad_l2",
        )
    if tol is None:
        return _unmeasured(
            "stationarity",
            "no optimiser tolerance was supplied, so a measured gradient has "
            "nothing to be compared against",
            "LbfgsConfig::gradient_tolerance",
        )
    if g <= tol:
        return Contract(
            "stationarity", True, HOLDS, g, tol,
            f"|grad|2 = {g:.3e} is at or below the optimiser's declared "
            f"tolerance {tol:.0e}, so the pose is a stationary point of the "
            f"field the engine scored",
            "result.grad_l2 vs LbfgsConfig::gradient_tolerance",
            FAMILY_ENGINE,
        )
    # The remediation is *derived from the two numbers the result already
    # carries*, not from a guess at what the caller could do. The step that
    # would make this pose a stationary point is one the search must be willing
    # to take; whether it is willing is the engine's business, and saying so is
    # more useful than naming a setting that does not exist.
    _step = _num(values, "finest_step")
    _width = _num(values, "descent_width")
    if _step is not None and _width is not None:
        if _step <= _width:
            _remedy = (
                f"the search's finest step ({_step:.0e}) is already "
                f"{_width / _step:.0f}x finer than this pose's descent region "
                f"({_width:.0e}), so the search can see the remaining descent "
                f"and chose not to take it; closing this needs a different "
                f"stopping test, not a smaller step"
            )
            _kind = REMEDY_ENGINE
        else:
            _remedy = (
                f"a step at or finer than {_width:.0e} would resolve this "
                f"pose's descent region, {_step / _width:g}x finer than the "
                f"finest step the search takes ({_step:.0e}); the step has to "
                f"come from the line search itself, so no setting reaches it"
            )
            _kind = REMEDY_ENGINE
    elif _step is not None:
        _remedy = (
            f"the search's finest step is {_step:.0e}; with this pose's "
            f"descent region unmeasured there is no way to say whether a finer "
            f"step would resolve it, and the stopping test is the engine's"
        )
        _kind = REMEDY_ENGINE
    else:
        _remedy = (
            "no finest line-search step was supplied, so the step that would "
            "resolve this cannot be named; what is measured is the gradient, "
            "and reducing it is the optimiser's business"
        )
        _kind = REMEDY_ENGINE
    return Contract(
        "stationarity", True, FAILS, g, tol,
        f"|grad|2 = {g:.3e} is {g / tol:g}x the optimiser's declared "
        f"tolerance {tol:.0e}, so this pose is NOT a stationary point of the "
        f"field the engine scored; it is a minimum only to within the line "
        f"search's resolution",
        "result.grad_l2 vs LbfgsConfig::gradient_tolerance",
        FAMILY_ENGINE, _remedy, _kind)


def _resolution(values: Mapping[str, Any]) -> Contract:
    """Is the pose's descent region resolvable at the search's finest step?

    The qualifier on the pose contract, as a separate question: a descent
    region narrower than the finest step the line search takes cannot be seen
    by it, so the pose is a minimum *to within the resolution* rather than a
    minimum. Both numbers are needed; either alone would be a claim about
    something else.
    """
    w = _num(values, "descent_width")
    step = _num(values, "finest_step")
    if w is None:
        return _unmeasured(
            "line-search resolution",
            "this result carries no measured width for its descent region, so "
            "the resolution the line search could resolve was not measured",
            "result.descent_width",
        )
    if step is None:
        return _unmeasured(
            "line-search resolution",
            "no finest line-search step was supplied, so a measured descent "
            "width has nothing to be compared against",
            "the search's _LINE_SEARCH_STEPS[0]",
        )
    if step <= w:
        # **The step being *smaller* than the region is the good case**, and
        # getting this backwards is a mistake this module made once: a step of
        # 1e-9 against a region 1e-8 wide can land *inside* the region and
        # would find the descent, so the region is resolved and the remaining
        # gradient is the optimiser's stopping test rather than the line
        # search's resolution. The failing case is the other one -- a step
        # coarser than the region steps over the whole thing.
        return Contract(
            "line-search resolution", True, HOLDS, w, step,
            f"the descent region is {w:.1e} wide and the finest step the line "
            f"search takes is {step:.0e}, {w / step:g}x finer, so the search "
            f"can land inside the region and resolve the descent in it",
            "result.descent_width vs the search's _LINE_SEARCH_STEPS[0]",
            FAMILY_ENGINE,
        )
    return Contract(
        "line-search resolution", True, FAILS, w, step,
        f"the descent region is {w:.1e} wide but the finest step the line "
        f"search takes is {step:.0e}, {step / w:g}x coarser, so the search "
        f"steps over the whole region and reports no descent: the pose is a "
        f"minimum only to within that resolution",
        "result.descent_width vs the search's _LINE_SEARCH_STEPS[0]",
        FAMILY_ENGINE,
        f"a step at or finer than {w:.0e} would resolve this region, "
        f"{step / w:g}x finer than anything the line search takes today "
        f"({step:.0e}); the ladder is inside the optimiser, so there is no "
        f"setting that reaches it",
        REMEDY_ENGINE)


def _support(values: Mapping[str, Any]) -> Contract:
    """Does any contributing pair sit outside the tabulated field's support?

    Beyond the underflow edge the Shape term is not a small number, it is an
    unrepresentable one, and the score reads `0.000000` there -- the same value
    a genuinely absent interaction produces. So this contract is about whether
    the number was *inside the model*, and the evidence is a distance.
    """
    d = _num(values, "closest_surface_distance")
    edge = _num(values, "support_zero_edge")
    if d is None:
        return _unmeasured(
            "field support",
            "this result carries no closest receptor-ligand surface distance, "
            "so whether any contributing pair lay outside the tabulated field's "
            "support was not measured -- and a score cannot tell you, because "
            "outside support it reads 0.000000",
            "result.closest_surface_distance",
        )
    if edge is None:
        return _unmeasured(
            "field support",
            "no support edge was supplied, so a measured distance has nothing "
            "to be compared against",
            "the f32 underflow edge of the Shape term",
        )
    if d <= edge:
        return Contract(
            "field support", True, HOLDS, d, edge,
            f"the closest contributing pair is {d:.2f} A apart, inside the "
            f"{edge:.2f} A at which the tabulated term is still representable",
            "result.closest_surface_distance vs the f32 underflow edge",
            FAMILY_MODEL,
        )
    return Contract(
        "field support", True, FAILS, d, edge,
        f"the closest contributing pair is {d:.2f} A apart, beyond the "
        f"{edge:.2f} A at which the tabulated term underflows to exactly zero: "
        f"this score is reading outside the model, and the 0.000000 it reports "
        f"there is indistinguishable from a genuinely absent interaction",
        "result.closest_surface_distance vs the f32 underflow edge",
        FAMILY_MODEL,
        f"the term stops being representable in f32 at {edge:.2f} A, so no "
        f"spacing, probe or box setting reaches {d:.2f} A -- recovering a "
        f"number there means storing the field in f64, which is a change to "
        f"the engine and costs 2x the map's memory; a contact table or a "
        f"distance statistic answers the question without one",
        REMEDY_ENGINE)


def _in_box(values: Mapping[str, Any]) -> Contract:
    """Did any atom leave the grid? A non-zero penalty says it did."""
    p = _num(values, "out_of_box_penalty")
    if p is None:
        return _unmeasured(
            "inside the box",
            "this result carries no out-of-box penalty, so whether any atom "
            "left the grid was not measured",
            "result.out_of_box_penalty",
        )
    if p == 0.0:
        return Contract(
            "inside the box", True, HOLDS, 0.0, 0.0,
            "no atom left the grid, so the reported energy is the field's own "
            "value at this pose",
            "result.out_of_box_penalty",
            FAMILY_ENGINE,
        )
    return Contract(
        "inside the box", True, FAILS, p, 0.0,
        f"{p:.6f} kcal/mol of out-of-box penalty is folded into the reported "
        f"energy, so the number is not the field's value at this pose",
        "result.out_of_box_penalty",
        FAMILY_ENGINE,
        f"{p:.6f} kcal/mol of the reported energy is the penalty rather than "
        f"the field, so the pose is not a placement this box can score; the "
        f"result does not report how far outside the box any atom sits, so the "
        f"box cannot be sized from this number -- enlarging it costs memory in "
        f"proportion to its volume, and the cheap correct action is to discard "
        f"the pose",
        REMEDY_PARAMETER)


# --------------------------------------------------------------------------
# The declared schema: the key names each family reads
# --------------------------------------------------------------------------
#
# **This block is the thing a real object is checked against, and it exists
# because of a hole found in round 5.** Every assertion in the gate fed this
# module dicts whose keys I typed, so every assertion was a key I wrote: if a
# field were really named `atoms_drop` and the module read `atoms_dropped`,
# the module would answer `unmeasured` on every real receptor and the gate
# would still be green, because the gate's dict was my dict. A test written in
# the same vocabulary as the code it tests proves that the two agree with each
# other, which is not the same as proving either agrees with the product.
#
# So the names are declared here, *once*, and the gate derives the real
# objects' available names and asserts this set is a subset of them. The
# direction matters: the object's names are the authority, and this set is the
# claim being tested against them.
#
# The receptor names are `ReceptorPrepReport`'s own -- and note that one of
# them, `fragments_equal_chains`, is a **property** and not a dataclass field.
# A projection built from `dataclasses.fields()` alone would miss it, which is
# the same asks-one-spelling trap one level down: the field is real, the
# spelling the projection happened to look for is not, and the fragment
# contract would read as `unmeasured` forever.

#: Keys that are allowed on any family and read by none of them.
LABEL_KEY = "label"

POSE_KEYS = frozenset({
    "grad_l2", "gradient_tolerance", "descent_width", "finest_step",
    "closest_surface_distance", "support_zero_edge", "out_of_box_penalty",
})
SCORE_KEYS = frozenset({
    "closest_surface_distance", "support_zero_edge",
    "cpu_energy", "gpu_energy", "cpu_gpu_tolerance",
})
#: `ReceptorPrepReport`'s fields and properties, plus the two `Receptor`
#: carries that the report does not.
RECEPTOR_KEYS = frozenset({
    "source", "atoms_in", "atoms_kept", "atoms_dropped",
    "fragments_in", "fragments_kept", "fragments_equal_chains",
    "chains_in", "chains_kept", "chain_ids_kept", "chain_ids_dropped",
    "selection", "water_atoms_removed", "ion_atoms_removed",
    "other_atoms_removed", "polar_hydrogens_added", "pdbqt_atoms_written",
    "num_polar_hydrogens", "unknown_atom_types",
})
#: `Pocket`'s fields, plus the caller's own parameters -- which is a second
#: authority, and a real one: `find_pockets`' **signature** is where
#: `max_pockets`, `min_voxels`, `min_burial` and `max_volume` come from, and
#: the gate derives those from the signature rather than from a list here.
POCKET_KEYS = frozenset({
    "center", "size", "voxels", "kind", "volume", "burial", "lining",
    "points", "spacing", "pocket_count",
    "max_pockets", "max_volume", "min_voxels", "min_burial",
})

SCHEMA: dict[str, frozenset[str]] = {
    "pose": POSE_KEYS,
    "score": SCORE_KEYS,
    "receptor": RECEPTOR_KEYS,
    "pocket": POCKET_KEYS,
}


def schema_for(kind: str) -> frozenset[str]:
    """The key names a caller may project onto a `verdict_<kind>`.

    This is the discovery half of the loudness: a caller should be able to ask
    what names are wanted instead of guessing them, which is precisely how
    `grad_norm` instead of `grad_l2` happened.

    **Recognised is not the same as read.** A recognised key that no contract
    consults -- `center`, `size`, `lining`, `points` and `spacing` on a pocket
    are all in that position -- is accepted and carried, because a caller
    projecting a whole site should not be told it passed a name the panel
    happens not to draw. It buys no contract. The names that *are* checked are
    the ones each `Contract.source` names, and a caller who wants the
    difference should read that field rather than infer it from this set.
    """
    if kind not in SCHEMA:
        raise KeyError(
            f"no schema for {kind!r}; this module has {sorted(SCHEMA)}")
    return SCHEMA[kind]


def _unknown_keys(kind: str, values: Mapping[str, Any]) -> tuple[str, ...]:
    """Keys the caller supplied that this family never reads.

    Sorted, so two calls with the same mistake give the same message and a
    test can name them.
    """
    wanted = SCHEMA.get(kind)
    if wanted is None:
        return ()
    return tuple(sorted(k for k in values
                        if k not in wanted and k != LABEL_KEY))


def _verdict(kind: str, contracts: Sequence[Contract], values: Mapping[str, Any]) -> Verdict:
    """Fold the contracts into one answer. `unmeasured` outranks `yes`.

    And a key the family does not read is **not** the same thing as a value
    that is absent, which is why this function looks at the caller's keys
    before it folds anything. A misspelled key would otherwise turn every
    contract into `unmeasured` and the verdict into a confident-sounding
    `unknown`, and `unknown` is exactly the answer a reader stops asking
    about -- so a typo would be indistinguishable from a result that
    genuinely carried nothing, and the module would be wrong quietly.
    """
    misspelled = _unknown_keys(kind, values)
    applicable = [c for c in contracts if c.applies]
    # The name goes in once. A caller that labels its poses `pose 3` and a
    # template that also says "pose" produces "pose pose 3", which is the kind
    # of small sloppiness that makes a reader distrust the rest of the text.
    name = str(values.get("label") or kind)
    failed = [c for c in applicable if c.state == FAILS]
    unknown = [c for c in applicable if c.state == UNMEASURED]
    if misspelled:
        # A known failure still outranks everything, so a misspelled key cannot
        # hide one. Otherwise this is an `unknown` and the reason is named.
        trust = NO if failed else UNKNOWN
    elif failed:
        trust = NO
    elif unknown:
        trust = UNKNOWN
    else:
        trust = YES
    if misspelled:
        # A misspelled key can *silence* a contract rather than merely muddy the
        # verdict. Rename the one value a failing contract reads and that
        # contract goes `unmeasured` -- which, on a real crambin preparation,
        # turns a measured `no` into an `unknown`. `unknown` is the one word
        # this module says buries a defect, so a caller's typo must not be
        # able to produce it quietly. Two things follow, and both used to be
        # absent: the explanation is unconditional (it was dropped whenever
        # another contract had failed, which is exactly when a reader most
        # needs it), and the contracts the typo silenced are named, so the
        # de-escalation is visible instead of having to be diffed against a
        # previous run.
        also = []
        if failed:
            also.append("failing: " + "; ".join(c.because for c in failed))
        if unknown:
            also.append("unmeasured, and one of the names above may be the "
                        "key they read: " + ", ".join(c.name for c in unknown))
        summary = (
            f"{name}: this verdict cannot be trusted because the result "
            f"carries {len(misspelled)} key(s) no {kind} contract reads: "
            f"{', '.join(misspelled)} -- that is a misspelling, not a "
            f"missing measurement; ask result_trust.schema_for({kind!r}) for "
            f"the names it wants"
            + (f"  Also {' and '.join(also)}." if also else "")
        )
    elif trust == NO:
        # The consequence, in the same breath as the verdict. A user who reads
        # "this pose is not a stationary point" and gets no next step has been
        # given a warning light with no dashboard, so the first failure's
        # remedy travels with the summary -- and the *kind* of remedy is stated
        # too, because "re-prepare the input" and "only the engine could" are
        # not the same kind of news and a flat sentence would blur them.
        lead = failed[0]
        _because = lead.because if lead.because.endswith((".", "!", "?")) \
            else lead.because + "."
        summary = (
            f"do not read {name} as a converged result: " + _because
            + (f"  To change it ({lead.remedy_kind}): {lead.remedy}"
               if lead.remedy else "")
            + (f"  Also failing: {', '.join(c.name for c in failed[1:])}"
               if len(failed) > 1 else "")
        )
    elif trust == UNKNOWN:
        summary = (
            f"{name}: no contract failed, but "
            + "; ".join(c.because for c in unknown)
            + " -- so this verdict is 'unknown', not 'yes'"
        )
    else:
        summary = (
            f"{name}: every applicable contract holds ({len(applicable)} checked)"
        )
    return Verdict(kind, trust, tuple(contracts), summary, misspelled)


def verdict_pose(values: Mapping[str, Any]) -> Verdict:
    """The per-pose verdict. `values` carries the pose's own reported numbers.

    Recognised keys, all optional -- and every one of them changes the answer,
    which is the point: a caller that supplies fewer keys gets `unknown`, and a
    caller that supplies none gets a verdict that says so in words.

        label                      a name for the pose, for the message
        grad_l2                    |grad|_2 of the field at the pose
        gradient_tolerance         LbfgsConfig::gradient_tolerance
        descent_width              width of the pose's descent region
        finest_step                the search's finest _LINE_SEARCH_STEPS entry
        closest_surface_distance   receptor-ligand surface distance, A
        support_zero_edge          the f32 underflow edge, A
        out_of_box_penalty         kcal/mol
    """
    return _verdict("pose", (
        _stationarity(values),
        _resolution(values),
        _support(values),
        _in_box(values),
    ), values)


def verdict_score(values: Mapping[str, Any]) -> Verdict:
    """The per-score verdict, for a number without a pose attached.

    A score can fail the support contract on its own -- a score computed for a
    configuration whose pairs are outside the tabulated field is outside the
    model -- and that is the case where no ladder, no gradient and no pose
    exist to rescue it. The contracts that need a pose are simply not
    applicable rather than silently passing.
    """

    def _gpu_agreement(v: Mapping[str, Any]) -> Contract:
        cpu = _num(v, "cpu_energy")
        gpu = _num(v, "gpu_energy")
        tol = _num(v, "cpu_gpu_tolerance")
        if cpu is None or gpu is None:
            return _unmeasured(
                "cpu/gpu agreement",
                "this result carries no paired CPU and GPU energy, so the two "
                "paths were not compared",
                "result.cpu_energy / result.gpu_energy",
            )
        if tol is None:
            return _unmeasured(
                "cpu/gpu agreement",
                "no agreement tolerance was supplied, so a measured "
                "difference has nothing to be compared against",
                "the GPU path's declared agreement",
            )
        gap = abs(cpu - gpu)
        if gap <= tol:
            return Contract(
                "cpu/gpu agreement", True, HOLDS, gap, tol,
                f"the two paths differ by {gap:.2e} kcal/mol, within the "
                f"declared {tol:.0e}",
                "result.cpu_energy / result.gpu_energy",
            )
        return Contract(
            "cpu/gpu agreement", True, FAILS, gap, tol,
            f"the two paths differ by {gap:.2e} kcal/mol, above the declared "
            f"{tol:.0e}, so this score is not portable between them",
            "result.cpu_energy / result.gpu_energy",
        )

    return _verdict("score", (
        _support(values),
        _gpu_agreement(values),
    ), values)


# --------------------------------------------------------------------------
# The receptor: a pose's contacts are a claim about a receptor, and the
# receptor is the part a preparation can quietly damage
# --------------------------------------------------------------------------
#
# Every contract here reads a field that `ReceptorPrepReport` actually carries
# (`atoms_in` / `atoms_kept` / `atoms_dropped`, `fragments_equal_chains`,
# `chain_ids_dropped`, the `water_` / `ion_` / `other_atoms_removed` triple,
# `polar_hydrogens_added`, `pdbqt_atoms_written`) or a field `Receptor` carries
# (`num_polar_hydrogens`, `unknown_atom_types`). None of them is a number this
# module computed, and none is derived from a document.

def _atoms_dropped(values: Mapping[str, Any]) -> Contract:
    kept = _num(values, "atoms_kept")
    dropped = _num(values, "atoms_dropped")
    if kept is None or dropped is None:
        return _unmeasured(
            "atoms retained",
            "this preparation did not report how many atoms it dropped, so "
            "what survived cannot be checked against the input -- a receptor "
            "that quietly lost residues still produces a score",
            "ReceptorPrepReport.atoms_kept / atoms_dropped",
        )
    if dropped == 0:
        return Contract(
            "atoms retained", True, HOLDS, kept, 0.0,
            f"all {kept:.0f} input atoms survived preparation",
            "ReceptorPrepReport.atoms_dropped",
            FAMILY_DATA,
        )
    where = []
    for _k, _label in (("water_atoms_removed", "water"),
                       ("ion_atoms_removed", "ions"),
                       ("other_atoms_removed", "everything else")):
        _v = _num(values, _k)
        if _v:
            where.append(f"{_v:.0f} {_label}")
    return Contract(
        "atoms retained", True, FAILS, dropped, 0.0,
        f"{dropped:.0f} of {kept + dropped:.0f} input atoms were dropped "
        f"({'; '.join(where) or 'the breakdown was not reported'}), so a "
        f"result computed from this receptor describes a structure that is not "
        f"the one that was read",
        "ReceptorPrepReport.atoms_dropped",
        FAMILY_DATA,
        f"re-preparing with the selection named explicitly -- "
        f"{str(values.get('selection', 'the rule it reported'))!r} is the rule "
        f"that chose -- brings the {dropped:.0f} atoms back, and the cost is "
        f"one more preparation; the search itself does not change, only the "
        f"receptor it is asked about",
        REMEDY_INPUT)


def _fragment_selection(values: Mapping[str, Any]) -> Contract:
    """Did the component the preparation kept agree with the file's chains?"""
    if "fragments_equal_chains" not in values:
        return _unmeasured(
            "fragment selection",
            "this preparation did not report whether the component it kept "
            "matched the file's chains, and a report that calls a fragment a "
            "chain is how a homodimer is docked as a monomer without "
            "anything saying so",
            "ReceptorPrepReport.fragments_equal_chains",
        )
    if bool(values.get("fragments_equal_chains")):
        return Contract(
            "fragment selection", True, HOLDS, 1.0, 1.0,
            "the component that survived preparation was one chain",
            "ReceptorPrepReport.fragments_equal_chains",
            FAMILY_DATA,
        )
    ids = values.get("chain_ids_dropped") or ()
    return Contract(
        "fragment selection", True, FAILS, 0.0, 1.0,
        f"the kept component is not a chain: connectivity and chain ids "
        f"disagree, so {len(ids)} chain id(s) "
        f"({', '.join(str(c) for c in ids) or 'unnamed'}) went into a result "
        f"scored against one fragment",
        "ReceptorPrepReport.fragments_equal_chains",
        FAMILY_DATA,
        f"the selection rule is reported as "
        f"{str(values.get('selection', 'unknown'))!r}; naming the chain "
        f"explicitly instead of taking the largest fragment makes the kept "
        f"component the one the file calls a chain, and the cost is one more "
        f"preparation",
        REMEDY_INPUT)


def _unrecognised_atoms(values: Mapping[str, Any]) -> Contract:
    n = _num(values, "unknown_atom_types")
    if n is None:
        return _unmeasured(
            "recognised atom types",
            "this result does not report how many atoms the engine could not "
            "recognise, and an unrecognised atom still contributes its shape "
            "term while silently losing its hydrogen-bond and hydrophobic "
            "character",
            "Receptor.unknown_atom_types",
        )
    if n == 0:
        return Contract(
            "recognised atom types", True, HOLDS, 0.0, 0.0,
            "every atom's PDBQT type is one the engine knows",
            "Receptor.unknown_atom_types",
            FAMILY_DATA,
        )
    return Contract(
        "recognised atom types", True, FAILS, n, 0.0,
        f"{n:.0f} atom(s) carry a PDBQT type the engine does not recognise: "
        f"they contribute a shape term while losing their hydrogen-bond and "
        f"hydrophobic character, so contacts involving them are understated",
        "Receptor.unknown_atom_types",
        FAMILY_DATA,
        f"the type name has to be one this engine recognises -- re-exporting "
        f"the receptor with known AutoDock types fixes it, and it costs one "
        f"more preparation; there is no setting that teaches the engine a new "
        f"token at scoring time",
        REMEDY_INPUT)


def _donor_hydrogens(values: Mapping[str, Any]) -> Contract:
    """Are polar hydrogens present for the donors?

    The threshold is not mine: it is the engine's own documented hint, that
    zero explicit polar hydrogens on a prepared receptor means the structure
    was written with non-polar hydrogens merged and every serine, threonine
    and tyrosine loses its hydrogen bond. Inventing a lower bound here would
    be inventing a specification, so the contract asks the question the engine
    already answered.
    """
    present = _num(values, "num_polar_hydrogens")
    added = _num(values, "polar_hydrogens_added")
    if present is None and added is None:
        return _unmeasured(
            "donor hydrogens",
            "this result carries neither a polar-hydrogen count nor a count of "
            "hydrogens the preparation added, so donor coverage cannot be "
            "checked -- a receptor prepared with merged hydrogens scores "
            "happily and loses every serine hydrogen bond",
            "Receptor.num_polar_hydrogens / ReceptorPrepReport.polar_hydrogens_added",
        )
    total = (present or 0.0) + (added or 0.0)
    if total > 0:
        return Contract(
            "donor hydrogens", True, HOLDS, total, 0.0,
            f"{total:.0f} explicit polar hydrogen(s) are present "
            f"({present or 0:.0f} in the structure, {added or 0:.0f} added)",
            "Receptor.num_polar_hydrogens / ReceptorPrepReport.polar_hydrogens_added",
            FAMILY_DATA,
        )
    return Contract(
        "donor hydrogens", True, FAILS, 0.0, 0.0,
        "no explicit polar hydrogens anywhere: the structure was prepared with "
        "non-polar hydrogens merged, which costs a hydrogen bond at every "
        "serine, threonine and tyrosine, and hydrogen bonds are what this "
        "scoring function is most sensitive to",
        "Receptor.num_polar_hydrogens / ReceptorPrepReport.polar_hydrogens_added",
        FAMILY_DATA,
        f"adding polar hydrogens on the donor sites is what "
        f"`prepare_receptor` does and it reports how many it added; the cost "
        f"is one more preparation and a receptor {added or 0:.0f} hydrogens "
        f"larger",
        REMEDY_INPUT)


def _written_atoms(values: Mapping[str, Any]) -> Contract:
    """Did the writer keep everything the preparation kept?"""
    written = _num(values, "pdbqt_atoms_written")
    expected = _num(values, "atoms_kept")
    added = _num(values, "polar_hydrogens_added")
    if written is None or expected is None:
        return _unmeasured(
            "atoms written",
            "this preparation did not report how many ATOM records the writer "
            "emitted, and that count can be lower than what was kept -- the "
            "writer drops a polar hydrogen it could not place",
            "ReceptorPrepReport.pdbqt_atoms_written",
        )
    want = expected + (added or 0.0)
    if written >= want:
        return Contract(
            "atoms written", True, HOLDS, written, want,
            f"the writer emitted {written:.0f} ATOM records for "
            f"{expected:.0f} kept atom(s) plus {added or 0:.0f} added "
            f"hydrogen(s)",
            "ReceptorPrepReport.pdbqt_atoms_written",
            FAMILY_DATA,
        )
    return Contract(
        "atoms written", True, FAILS, written, want,
        f"the writer emitted {written:.0f} ATOM records where {want:.0f} were "
        f"expected, so {want - written:.0f} atom(s) the preparation kept did "
        f"not reach the engine",
        "ReceptorPrepReport.pdbqt_atoms_written",
        FAMILY_DATA,
        f"the {want - written:.0f} missing atom(s) are ones the writer could "
        f"not place; the writer's own warning names them, and the cost of "
        f"reading past it is that these atoms are scored as if they were not "
        f"there",
        REMEDY_INPUT)


def verdict_receptor(values: Mapping[str, Any]) -> Verdict:
    """The per-receptor verdict: can this receptor's numbers be trusted?

    A pose's contacts are a claim about a receptor, and a preparation can
    damage a receptor without raising anything. So this verdict is asked
    *before* a pose's, not instead of it: a perfect pose on a receptor that
    dropped half its chains is not a trustworthy result.

        label, selection, chain_ids_dropped
        atoms_in, atoms_kept, atoms_dropped
        water_atoms_removed, ion_atoms_removed, other_atoms_removed
        fragments_equal_chains, polar_hydrogens_added, pdbqt_atoms_written
        num_polar_hydrogens, unknown_atom_types
    """
    return _verdict("receptor", (
        _atoms_dropped(values),
        _fragment_selection(values),
        _unrecognised_atoms(values),
        _donor_hydrogens(values),
        _written_atoms(values),
    ), values)


# --------------------------------------------------------------------------
# The pocket: a claim about where something binds, derived from a clustering
# and a threshold, both of which are the caller's
# --------------------------------------------------------------------------
#
# The failure this exists for is specific: a list of twelve pockets is shown as
# twelve facts, when what the code did was take the best twelve of whatever the
# clustering found. The set is censored and the pocket is a claim, and both are
# things the numbers can be asked about.

def _enclosure(values: Mapping[str, Any]) -> Contract:
    """Is this a sealed cavity, or a claim about being enclosed?

    The workbench's own `find_pockets` docstring calls these two different
    kinds of claim, and it is right: a cavity is found by flooding from the
    boundary and keeping what the flood cannot reach, while a burial site is
    enclosed on enough axes to look like a groove. Both are legitimate and only
    one is unambiguous, so this is a contract about *what kind of claim*, and it
    fails for the heuristic kind with a remediation that says so.
    """
    kind = values.get("kind")
    if kind is None:
        return _unmeasured(
            "enclosure",
            "this site does not say whether it was found as a sealed cavity or "
            "as a burial site, and those are different claims: one is "
            "unambiguous and the other is a heuristic",
            "Pocket.kind",
        )
    if str(kind) == "cavity":
        return Contract(
            "enclosure", True, HOLDS, 1.0, 1.0,
            "sealed from the solvent: found by flooding from the grid boundary "
            "and keeping what the flood could not reach",
            "Pocket.kind",
            FAMILY_MODEL,
        )
    burial = _num(values, "burial")
    axes = _num(values, "min_burial")
    return Contract(
        "enclosure", True, FAILS, burial, axes,
        f"a burial site, not a sealed cavity: it is enclosed on enough axes to "
        f"look like a groove"
        + (f" ({burial:.2f} of the fraction buried)"
           if burial is not None else "")
        + ", which is a heuristic about where a ligand might sit rather than a "
          "measurement of a pocket",
        "Pocket.kind",
        FAMILY_MODEL,
        f"running the probe sweep"
        + (f" below {axes:.0f} axis/axes of burial, or tightening `min_burial`"
           if axes is not None else "")
        + " separates grooves from cavities; the cost is more grid points per "
          "probe, so a sweep costs the sum of all its probes rather than one",
        REMEDY_PARAMETER)


def _volume_censoring(values: Mapping[str, Any]) -> Contract:
    """Is this volume a measurement, or a number that stopped at a ceiling?"""
    volume = _num(values, "volume")
    cap = _num(values, "max_volume")
    if volume is None or cap is None:
        return _unmeasured(
            "volume not censored",
            "this site does not carry the ceiling its volume was measured "
            "under, so a volume that stopped at the ceiling cannot be told "
            "from one that did not",
            "Pocket.volume / the caller's max_volume",
        )
    if volume < cap:
        return Contract(
            "volume not censored", True, HOLDS, volume, cap,
            f"the volume {volume:.1f} A^3 is below the {cap:.0f} A^3 ceiling it "
            f"was measured under, so it is a measurement",
            "Pocket.volume / the caller's max_volume",
            FAMILY_PARAMETER,
        )
    return Contract(
        "volume not censored", True, FAILS, volume, cap,
        f"the volume reads {volume:.1f} A^3 against a ceiling of {cap:.0f} A^3, "
        f"so it is censored: the site is at least this big and the number is "
        f"not its size",
        "Pocket.volume / the caller's max_volume",
        FAMILY_PARAMETER,
        f"raising or removing `max_volume` would measure it, and the cost is a "
        f"grid over the uncapped region; if only the ordering matters the "
        f"rank is still meaningful even when the volume is not",
        REMEDY_PARAMETER)


def _resolution_floor(values: Mapping[str, Any]) -> Contract:
    """Is this site more than one voxel, or exactly the floor?"""
    voxels = _num(values, "voxels")
    floor = _num(values, "min_voxels")
    if voxels is None or floor is None:
        return _unmeasured(
            "above the grid floor",
            "this site does not carry its voxel count or the floor it was kept "
            "at, so a site one voxel from not existing cannot be told from one "
            "that comfortably exists",
            "Pocket.voxels / MIN_VOXELS",
        )
    if voxels > floor:
        return Contract(
            "above the grid floor", True, HOLDS, voxels, floor,
            f"{voxels:.0f} voxels against a floor of {floor:.0f}, so the site "
            f"is {(voxels / floor):.1f}x the smallest thing that was kept",
            "Pocket.voxels / MIN_VOXELS",
            FAMILY_MODEL,
        )
    return Contract(
        "above the grid floor", True, FAILS, voxels, floor,
        f"{voxels:.0f} voxel(s) against a floor of {floor:.0f}: this site is "
        f"exactly at the resolution the search keeps, and one voxel fewer would "
        f"not have been returned at all",
        "Pocket.voxels / MIN_VOXELS",
        FAMILY_MODEL,
        f"lowering `min_voxels` below {floor:.0f} would admit smaller sites, "
        f"and the cost is more clusters from the same grid rather than a "
        f"larger one; a finer `spacing` changes what one voxel means, which is "
        f"the difference between a smaller site and a smaller measurement",
        REMEDY_PARAMETER)


def _set_completeness(values: Mapping[str, Any], count: float, cap: float) -> Contract:
    """Is this the whole set, or the best N of it?

    This is the one that a user shown a list of twelve never gets told. The
    ceiling is the caller's `max_pockets`, and a list exactly as long as the
    ceiling is a censored list: there may be a thirteenth.
    """
    if count < cap:
        return Contract(
            "set complete", True, HOLDS, count, cap,
            f"{count:.0f} site(s) against a ceiling of {cap:.0f}, so the "
            f"clustering found fewer than the caller asked to keep",
            "len(pockets) / the caller's max_pockets",
            FAMILY_PARAMETER,
        )
    return Contract(
        "set complete", True, FAILS, count, cap,
        f"{count:.0f} sites returned against a ceiling of {cap:.0f}: this is "
        f"the best {cap:.0f}, not all of them, and a site just below the cut "
        f"exists unless the clustering provably ended",
        "len(pockets) / the caller's max_pockets",
        FAMILY_PARAMETER,
        f"raising `max_pockets` shows more of the set and the cost is one more "
        f"clustering pass, but it does not make the list complete either -- "
        f"only an unsaturated count does, and this one is saturated",
        REMEDY_PARAMETER)


def verdict_pocket(values: Mapping[str, Any]) -> Verdict:
    """The per-pocket verdict: is this a fact, a claim, or a censored number?

        label, kind, volume, voxels, burial
        max_volume, min_voxels, min_burial, pocket_count
    """
    count = _num(values, "pocket_count")
    cap = _num(values, "max_pockets")
    contracts = [
        _enclosure(values),
        _volume_censoring(values),
        _resolution_floor(values),
    ]
    if count is not None and cap is not None:
        contracts.append(_set_completeness(values, count, cap))
    else:
        contracts.append(_unmeasured(
            "set complete",
            "this result does not carry how many sites were found against the "
            "ceiling it was asked for, so a list as long as the ceiling cannot "
            "be told from the whole set",
            "len(pockets) / the caller's max_pockets",
        ))
    return _verdict("pocket", contracts, values)
