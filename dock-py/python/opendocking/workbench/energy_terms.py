"""The per-term energy decomposition of one pose, for the workbench.

The engine exposes its decomposition twice over: `Receptor.precalculate_terms`
tabulates one map field per Vina term, and `core.score_conformation_terms`
returns the five terms for one conformation. This module is the workbench's
route to it and the only route. Every number a user can see in the panel comes
out of that one call. No term is recomputed here, because a term recomputed in
Python is a second implementation of the scoring function, and a second
implementation that quietly disagrees with the engine is indistinguishable from
a bug in the engine.

Two properties of the engine's API decide what the panel can honestly show.

* **The decomposition is keyed on a conformation, not on coordinates.**
  `DockingResult` keeps the packed degree-of-freedom vector of every pose it
  reported, so a pose from a run made in this session can be decomposed. A pose
  read from a file carries coordinates and no torsion vector, and there is no
  honest way to recover one: recovering it means solving the inverse kinematics,
  which is a different program with its own error. So the panel says why it has
  nothing to show instead of showing the previous pose's numbers, which is the
  failure mode this module exists to make impossible.
* **The five terms come back already weighted.** The weights are applied while
  the map is tabulated, so multiplying by them again would double-count. The
  engine also returns `terms_total`, `intermolecular` and `total` precisely so
  that the decomposition can be *checked* rather than believed, and this module
  carries all three so the panel can show the arithmetic rather than a sum it
  computed itself.

The term names are the ones `docs/SCORING.md` uses in its weights table
(`gauss1`, `gauss2`, `repulsion`, `hbond`, `hydrophobic`); the engine's own
keys (`g1`, `g2`, `rep`, `hb`, `hyd`) are carried next to them so a reader who
goes looking in `core.py` can match what they see to what they read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: (name as `docs/SCORING.md` names it, the engine's key, what it measures).
#: Fixed tuples rather than a dict: the panel's rows have an order, and a dict
#: would make the order a property of insertion for no reason a reader could
#: check.
TERM_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "gauss1",
        "g1",
        "the close, narrow reward for a contact of about the right size",
    ),
    (
        "gauss2",
        "g2",
        "the wide, shallow counterpart: a contact that is merely not clashing",
    ),
    (
        "repulsion",
        "rep",
        "the wall. Never negative -- a pair of atoms overlapping costs energy",
    ),
    (
        "hbond",
        "hb",
        "donor-acceptor attraction, saturating at the documented separations",
    ),
    (
        "hydrophobic",
        "hyd",
        "attraction between two non-polar atoms",
    ),
)

#: Below this magnitude a term is printed in exponent form. The point is not
#: tidiness: `+0.0000` is a claim that a term is zero, and a term that is
#: really -1e-30 has been flattened into exactly that claim by four decimal
#: places. An exponent keeps the sign and the magnitude, which is the whole
#: reason to look at the term at all.
EXPONENT_BELOW = 1e-4


class TermsUnavailable(RuntimeError):
    """The decomposition cannot be produced for this pose, and here is why.

    A distinct type because the panel's two failure modes need two different
    sentences, and a bare `Exception` would collapse them into one: "this pose
    has no conformation vector" is a property of where the pose came from, and
    "the maps could not be built" is a property of the receptor and the box.
    """


def format_kcal(value: float, places: int = 4) -> str:
    """One term, with its sign, and without a small number pretending to be 0.

    `+0.0000` and `-0.0000` are the two answers a fixed-point format gives for
    a term that is not zero, and they are indistinguishable from each other and
    from a true zero. So anything under `EXPONENT_BELOW` is printed in exponent
    form, an exact zero is printed as `0`, and everything else keeps its sign at
    `places` decimals.
    """
    v = float(value)
    if v == 0.0:
        return "0"
    if abs(v) < EXPONENT_BELOW:
        return f"{v:+.3e}"
    if not math.isfinite(v):
        return f"{v}"
    return f"{v:+.{places}f}"


def _as_displayed(value: float, places: int = 4) -> float:
    """The number `format_kcal` will print, as a number.

    A term printed in exponent form has no four-decimal reading, so it
    contributes nothing to a four-decimal sum -- which is precisely what the
    printed column adds up to. `round` here and `format` there are the same
    rounding, so the two cannot disagree; the gate checks that by parsing the
    printed strings rather than by trusting this function.
    """
    v = float(value)
    if abs(v) < EXPONENT_BELOW:
        return 0.0
    return round(v, places)


@dataclass(frozen=True)
class PoseBreakdown:
    """One pose's decomposition, as the engine returned it.

    `pose_index` is here rather than left to the caller because the panel's
    whole value is that it names the pose it is describing: a breakdown with no
    pose on it is a set of numbers about an unknown molecule.
    """

    pose_index: int
    num_poses: int
    scoring: str
    total: float
    terms: tuple[tuple[str, str, float], ...]
    terms_total: float
    intermolecular: float
    intramolecular: float
    intramolecular_scale: float
    out_of_box_penalty: float

    @property
    def described_pose(self) -> str:
        """The one line that says which pose these numbers belong to."""
        return f"pose {self.pose_index + 1} of {self.num_poses}"

    def term_rows(self) -> list[tuple[str, str, str, str]]:
        """`(name, engine key, value, what it measures)` per term, in order.

        The description is looked up from `TERM_ROWS` rather than carried in
        `self.terms`, so the numbers and the prose cannot drift apart: a term
        whose name is not in the table would get an empty description instead of
        silently borrowing another term's.
        """
        notes = {name: note for name, _key, note in TERM_ROWS}
        return [
            (name, key, format_kcal(value), notes.get(name, ""))
            for name, key, value in self.terms
        ]

    @property
    def displayed_total(self) -> float:
        """The sum of the five terms **as this panel prints them**.

        This exists because of a defect a reader found by adding the column up
        by hand. The engine's `terms_total` is the exact sum of the exact terms;
        the panel prints each term rounded to four decimals, so on pose 9 of the
        biotin fixture the column read -2.2364, -0.1803, +0.3965, 0, -0.8322 --
        which is -2.8524 by hand -- while the sum row said -2.8525. Both numbers
        were true and the panel looked broken, because for a panel whose whole
        purpose is to let somebody audit a number, a column that does not add up
        to the total printed under it is the worst property it could have. On
        pose 7 the two agreed exactly, so the defect was invisible on half the
        poses, which is worse: it looked like a rounding curiosity rather than a
        rule.

        So the sum of the column is printed *as the sum of the column*, and the
        engine's own `terms_total` is printed next to it under its own name.
        Both are shown because they are different claims: one says the printed
        numbers add up, the other is the engine's cross-check against its own
        production path. Collapsing them would have cost the cross-check.
        """
        return sum(_as_displayed(value) for _n, _k, value in self.terms)

    def check_rows(self) -> list[tuple[str, str, str]]:
        """The numbers that let the decomposition be checked, not believed.

        The first row is the sum of the column above it and is therefore the sum
        of *printed* values; the second is the engine's own `terms_total`, and
        the two can differ in the last printed decimal place for exactly that
        reason. The engine's `terms_total` against its `intermolecular` is the
        consistency pair that matters: they are computed by different paths and
        agree only up to the `float32` the production shape field rounds once
        and the term maps round per term. A reader who cannot see all three
        cannot tell a rounding artefact from a real disagreement.
        """
        return [
            (
                "sum of the five terms above",
                format_kcal(self.displayed_total),
                "displayed",
            ),
            ("sum of the five terms (engine)", format_kcal(self.terms_total),
             "engine"),
            (
                "intermolecular (production path)",
                format_kcal(self.intermolecular),
                "engine",
            ),
            (
                "intramolecular x scale",
                f"{format_kcal(self.intramolecular)} x "
                f"{self.intramolecular_scale:g}",
                "engine",
            ),
            ("total (the pose's own energy)", format_kcal(self.total), "engine"),
            (
                "out-of-box penalty",
                format_kcal(self.out_of_box_penalty),
                "engine",
            ),
        ]


class TermEvaluator:
    """Builds the engine's maps once, then answers per-pose breakdowns.

    The maps are the expensive part and they depend only on the receptor, the
    box, the spacing and the scoring function, so they are built on the first
    question and kept. Constructing the evaluator is free; that is deliberate,
    because the window rebuilds it whenever the box moves and a constructor that
    tabulated a grid would make moving a spin box freeze the window.
    """

    def __init__(
        self,
        receptor_path: str | Path,
        box_: Any,
        ligand_path: str | Path,
        scoring: str = "vina",
        spacing: float = 0.375,
    ) -> None:
        self.receptor_path = Path(receptor_path)
        self.box = box_
        self.ligand_path = Path(ligand_path)
        self.scoring = str(scoring)
        self.spacing = float(spacing)
        self._maps = None
        self._term_maps = None
        self._ligand = None

    @property
    def built(self) -> bool:
        """Whether the maps have been tabulated yet."""
        return self._maps is not None

    def _build(self) -> None:
        from ..core import Receptor, load_ligand

        try:
            receptor = Receptor.from_pdbqt(self.receptor_path)
            self._maps = receptor.precalculate(
                self.box, self.scoring, self.spacing
            )
            self._term_maps = receptor.precalculate_terms(
                self.box, self.scoring, self.spacing
            )
            self._ligand = load_ligand(self.ligand_path)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self._maps = None
            self._term_maps = None
            self._ligand = None
            raise TermsUnavailable(
                f"the term maps could not be built for this receptor and box: "
                f"{exc}"
            ) from exc

    def breakdown(self, result: Any, pose_index: int) -> PoseBreakdown:
        """Pose ``pose_index`` of ``result``, decomposed by the engine.

        ``result`` is a `DockingResult`. It is required rather than optional
        because the conformation vector this needs lives on it, and a caller
        holding only a pose file has nothing to pass -- which is the honest
        answer for that case, not a reason to guess a conformation.
        """
        from ..core import score_conformation_terms

        if result is None:
            raise TermsUnavailable(
                "these poses came from a file, and a pose file carries "
                "coordinates rather than the conformation the engine's "
                "decomposition is keyed on. Run a docking here, or read the "
                "breakdown with `odcli`, and the terms are the same numbers"
            )
        if not 0 <= int(pose_index) < int(result.num_poses):
            raise TermsUnavailable(
                f"pose {int(pose_index) + 1} is not one of the "
                f"{int(result.num_poses)} poses this run reported"
            )
        if self._maps is None:
            self._build()
        conformation = np.asarray(
            result.pose_conformation(int(pose_index)), np.float64
        )
        try:
            raw = score_conformation_terms(
                self._ligand,
                self._maps,
                self._term_maps,
                conformation,
                self.scoring,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the user
            raise TermsUnavailable(
                f"the engine declined to decompose this pose: {exc}"
            ) from exc
        return _breakdown_from_raw(raw, int(pose_index), int(result.num_poses),
                                   self.scoring)


def _breakdown_from_raw(
    raw: dict, pose_index: int, num_poses: int, scoring: str
) -> PoseBreakdown:
    """The engine's dict, named and ordered. The only place keys are spelled.

    A missing key is a finding rather than a zero: the engine changing what it
    returns should stop this panel, not make it print `0` for a term it could
    not find.
    """
    try:
        terms = tuple(
            (name, key, float(raw[key])) for name, key, _note in TERM_ROWS
        )
        return PoseBreakdown(
            pose_index=pose_index,
            num_poses=num_poses,
            scoring=scoring,
            total=float(raw["total"]),
            terms=terms,
            terms_total=float(raw["terms_total"]),
            intermolecular=float(raw["intermolecular"]),
            intramolecular=float(raw["intramolecular"]),
            intramolecular_scale=float(raw["intramolecular_scale"]),
            out_of_box_penalty=float(raw["out_of_box_penalty"]),
        )
    except KeyError as exc:
        raise TermsUnavailable(
            f"the engine's decomposition has no {exc.args[0]!r} any more, so "
            f"this panel would be showing a zero it did not measure"
        ) from exc
