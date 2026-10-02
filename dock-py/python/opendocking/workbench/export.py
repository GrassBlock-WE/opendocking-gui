"""Taking a run away: one file, everything the window showed, and its provenance.

**Why this module exists.** The workbench can show a person everything about a
docking run and can hand them none of it. Nine poses, per-term breakdowns,
per-pair contacts, a verdict per pose, the box, the seed, the exhaustiveness --
all of it exists as pixels on one machine, and a number somebody saw once and
cannot get back is a number they cannot check.

**The shape of the answer is the interesting part, and the shape is dictated by
one recurring failure of this project.** A report that silently omits a number
looks exactly like a report in which that number was fine. An exported run that
left out the four `unmeasured` contracts would be indistinguishable from one
where all four held. So:

* **The unit of export is not a number, it is a record about a number.**
  `number()` returns ``{"value": ..., "state": "measured"|"unmeasured",
  "because": ...}``. A missing measurement is ``null`` **and** the state **and**
  the sentence saying why, so it cannot be read as a zero, and cannot be read
  as a field nobody filled in. This is the whole design in one function.
* **The verdict is copied, never recomputed.** `holds` / `fails` / `unmeasured`
  and each contract's `because` travel as they were decided, because a report
  that re-derives its own verdict from the numbers on the way out is a report
  that can disagree with the window it came from -- and a disagreement nobody
  looks for is a disagreement that ships.
* **What *is* recomputed is named as recomputed.** The residue and pair tables
  are built at export time from the coordinates in the same file, because the
  window only ever filled them for the pose on screen. `DERIVED` says so, in
  the module and in the file, and the gate checks the exported rows against the
  rows the *table* was showing -- a different call, at a different time.
* **Provenance is not a footer.** The backend, the seed, the box, the
  exhaustiveness, the scoring function, and the sha256 of both input files are
  in the document, because an exported number with no box beside it is not a
  result anybody can reproduce.
* **A warning the window speaks is a warning the file speaks.** The window
  prints a `WARNING:` line for a run whose receptor carried atom types the
  engine could not classify, and the exported file is the artifact somebody
  takes away -- so `warnings` is a top-level list beside `absent`, carrying the
  same kind of sentence. A file that is silent about a defect the window
  announced is the file being wrong about the run.
* **`absent` is not `unmeasured` is not `not_run`, and a fourth thing is not
  `measured`.** A count the engine build does not export at all is
  :data:`STATE_ABSENT`: there is no run-specific failure behind it, so
  `unmeasured` would blame a flaky measurement for a stale binary. All four
  states are spelled, and the one that is not a measurement is listed.

Nothing here reads a file the window did not already read, and nothing here
touches Qt: the document is built from plain values so it can be built in a
check with no display and diffed against the window by hand.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "DERIVED",
    "STATE_ABSENT",
    "UNKNOWN_ATOM_TYPE_WARNING",
    "number",
    "unrecognised_atom_types",
    "PoseRecord",
    "RunRecord",
    "run_export",
    "to_json",
    "write_export",
    "read_export",
    "ExportUnavailable",
    "default_export_path",
]

#: The one string that says what this file is. A reader that finds anything else
#: -- a different schema, no schema -- is looking at something this module did
#: not write, and `read_export` says so rather than guessing.
SCHEMA = "opendocking.run-export/1"

#: What the export computes for itself, stated in the file next to the
#: consequences. A reader who wants to check `contacts` needs to know it was
#: computed here and not carried out of the panel, and a reader who wants to
#: check `verdict` needs to know it was **not**.
DERIVED = {
    "contacts": (
        "computed at export time by contacts.find_contacts from the "
        "'coords' in this same file, once per pose"
    ),
    "terms": (
        "copied from the engine's own score_conformation_terms, one call per "
        "pose, at export time"
    ),
    "verdict": (
        "copied from the panel's PoseVerdict rows, which are the module's own "
        "verdict. NOT recomputed from the numbers in this file: a report that "
        "re-derives its own verdict is a report that can disagree with the "
        "window it came from"
    ),
    "energies_and_rmsd": "copied from the DockingResult the run produced",
}


class ExportUnavailable(Exception):
    """There is no run to export, and the message says which of the reasons.

    Raised rather than written as an empty document, because an exported file
    with `"poses": []` and no reason is the failure this module exists to stop:
    it looks like a run that found nothing.
    """


#: The third state, and the one this module was missing.
#:
#: `unmeasured` means *this run could not measure it*; `not_run` means *this run
#: never tried*. Neither is what a count looks like when the **engine build**
#: has no such field to read: there is no run-specific failure behind it, and
#: every run from that build would report the same absence. Spelling that
#: `unmeasured` makes a stale `.pyd` look like a flaky measurement, which is
#: the more confusing of the two lies -- one sends a reader to re-run, the
#: other sends them to rebuild.
#:
#: So it gets its own state, and `unrecognised_atom_types` below is its only
#: producer. The honest reading of a value in this state is **"this file cannot
#: tell you"**, which is emphatically not "nothing was wrong": the count's whole
#: purpose is to be a count of things that *were* wrong, and a reader who is
#: handed `null` with no state would be free to read it either way.
STATE_ABSENT = "absent"

#: The consequence, in the file's own words, for a receptor carrying atom types
#: the engine cannot classify.
#:
#: A count alone is not a warning. `number()` clears `because` for a value that
#: was measured, and it is right to: a measured number needs no excuse. But
#: `382.0` sitting in `provenance` is a number whose *consequence* is nowhere in
#: the document, and the consequence is the whole point -- every energy in the
#: file understates the receptor. So the sentence travels beside the number, in
#: the top-level `warnings` list, rather than being folded into the record and
#: thereby lost.
#:
#: Wording is this module's own and not `DockingResult.summary()`'s. The engine
#: says the same thing in its own words in a window; an export that copied its
#: prose would be a second place to keep it in step, and this sentence is the
#: one a reader of the *file* is owed.
UNKNOWN_ATOM_TYPE_WARNING = (
    "{n} receptor atom(s) carry a PDBQT type this engine does not recognise. "
    "They contribute a shape term while losing their hydrogen-bond and "
    "hydrophobic character, so every energy in this file understates the "
    "receptor. Re-export the receptor with known AutoDock types to fix it."
)


def number(value: Any, because: str = "", state: str = "unmeasured",
           source: str = "") -> dict:
    """One exported number, with its own claim about whether it exists.

    ``None`` is **not** how a missing measurement travels. A bare ``null`` in a
    JSON document cannot be told apart from a zero that failed to serialise,
    from a field nobody filled in, and from a field that was deliberately not
    run -- four different facts with one spelling. So the unit is a record:

    * ``{"value": 0.0, "state": "measured", "because": "", "source": "the
      engine"}`` -- a real zero, which is a result, and the reader can see
      which of the three sources it came from;
    * ``{"value": null, "state": "unmeasured", "because": "..."}`` -- a
      measurement that was not made, with the sentence that says so;
    * ``{"value": null, "state": "not_run", "because": "..."}`` -- not even
      attempted, which is a different fact again and gets its own state.

    `state` says which of those it is, and it is not free text, so a reader can
    count them. `source` matters even for a value that is present: an affinity
    the engine reported and an affinity read out of a pose file are both
    "measured" and they are not the same measurement, and a reader auditing the
    number needs to know which one they are holding.
    """
    if value is None:
        return {"value": None, "state": state, "because": because,
                "source": source}
    return {"value": float(value), "state": "measured", "because": "",
            "source": source}


def unrecognised_atom_types(result, *, source: str) -> tuple[dict, str | None]:
    """Receptor atoms whose PDBQT type the engine does not recognise.

    Returns ``(record, warning)``. `record` is a :func:`number`, so it carries
    its own state; `warning` is the sentence a reader needs when the count is
    not zero, and ``None`` when there is nothing to warn about.

    **Why the read is wrapped.** The field was added to the result *after* the
    shared `_dockpy.pyd` in the checkout was built, and a property that reads a
    method the binary does not have raises `AttributeError` rather than
    returning a default. So an export taken with an older engine would have
    died at the last step of a long run, on a field that exists to make the file
    *more* honest. The absence is therefore a **recorded state**
    (:data:`STATE_ABSENT`) and a line in the document's `absent` index, and the
    export still happens.

    That is the whole difference between this and the bug it replaces. A crash
    is loud and this would have been, so the danger is the other shape: writing
    ``0`` for a count nobody could read. ``0`` here is a claim -- *every atom
    was recognised* -- and a file making that claim from a binary that cannot
    answer the question is precisely the silent-clean-export failure this
    function exists to stop.

    `result` is ``None`` for a pose-file session. That is not a build problem
    and not a measurement problem: there is no run, so there is no receptor the
    engine ever scored, and the count is :data:`STATE_ABSENT` for that reason and
    not a second one.
    """
    if result is None:
        return (
            number(None, state=STATE_ABSENT, source=source,
                   because="these poses came from a file, so there is no run "
                           "behind them and no receptor for the engine to have "
                           "scored; nothing here is a claim about atom types"),
            None,
        )
    try:
        raw = result.unknown_atom_types
    except AttributeError as exc:
        return (
            number(None, state=STATE_ABSENT, source=source,
                   because=f"the engine build that produced this run does not "
                           f"export the count ({type(exc).__name__}: {exc}). "
                           f"That is a property of the build, not of this run, "
                           f"and it is not a claim that every atom was "
                           f"recognised"),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - the file still has to be written
        return (
            number(None, state="unmeasured", source=source,
                   because=f"the engine's own accessor raised "
                           f"{type(exc).__name__}: {exc}"),
            None,
        )
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        return (
            number(None, state="unmeasured", source=source,
                   because=f"the engine returned {raw!r}, which is not a count "
                           f"({type(exc).__name__}: {exc})"),
            None,
        )
    return (
        number(count, source=source),
        UNKNOWN_ATOM_TYPE_WARNING.format(n=count) if count else None,
    )


@dataclass(frozen=True)
class PoseRecord:
    """One pose, with every field carrying its own state.

    `terms` is ``None`` when the engine was never asked -- a pose file carries
    coordinates, not the conformation the decomposition is keyed on -- and that
    is exported as ``None`` with ``terms_because`` saying so, beside a state of
    ``"unmeasured"`` rather than a state of ``"zero terms"``.
    """

    index: int
    coords: tuple[tuple[float, float, float], ...]
    coords_source: str
    affinity: dict
    rmsd: dict
    intermolecular: dict
    terms: dict | None
    terms_state: str
    terms_because: str
    verdict: dict
    residue_rows: tuple[tuple[Any, ...], ...]
    pair_rows: tuple[tuple[Any, ...], ...]
    contacts_count: int

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "described_pose": f"pose {self.index + 1}",
            "coords": [list(c) for c in self.coords],
            "coords_source": self.coords_source,
            "atom_count": len(self.coords),
            "affinity_kcal_per_mol": self.affinity,
            "rmsd_to_best": self.rmsd,
            "intermolecular_kcal_per_mol": self.intermolecular,
            "energy_terms": self.terms,
            "energy_terms_state": self.terms_state,
            "energy_terms_because": self.terms_because,
            "verdict": self.verdict,
            "contacts": {
                "count": self.contacts_count,
                "residues": [list(r) for r in self.residue_rows],
                "pairs": [list(p) for p in self.pair_rows],
            },
        }


@dataclass(frozen=True)
class RunRecord:
    """The whole document: what produced it, what it holds, and what is missing.

    `absent` is the index of every field that is not a measurement, in one list
    at the top of the document. The per-field records already carry it, and this
    is the copy a reader can read without walking nine pose objects -- which is
    the difference between an absence that is *documented* and one that a reader
    has to go looking for. A report that silently omits a number looks exactly
    like a report in which the number was fine, and this list is what makes the
    two look different.

    `warnings` is its sibling and answers the other half of the same question.
    `absent` says what this file cannot tell you; `warnings` says what this file
    *knows* is wrong with the run. They are not the same list and must not be
    merged: an empty `absent` beside a full `warnings` is the run that
    measured everything cleanly and is still wrong, which is the case the
    unrecognised-atom-types count is about.
    """

    provenance: dict
    poses: tuple[PoseRecord, ...]
    selected_pose: int
    written: str
    absent: tuple[dict, ...] = ()
    notes: tuple[str, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "written": self.written,
            "selected_pose": self.selected_pose,
            "pose_count": len(self.poses),
            "absent": [dict(a) for a in self.absent],
            "absent_count": len(self.absent),
            "warnings": list(self.warnings),
            "warning_count": len(self.warnings),
            "provenance": self.provenance,
            "derived": dict(DERIVED),
            "notes": list(self.notes),
            "poses": [p.as_dict() for p in self.poses],
        }


def run_export(*, provenance: dict, poses: tuple[PoseRecord, ...],
               selected_pose: int = -1, absent: tuple[dict, ...] = (),
               notes: tuple[str, ...] = (),
               warnings: tuple[str, ...] = ()) -> RunRecord:
    """The document, with the checks that would make it a lie stated as errors.

    Refuses an empty pose list rather than writing one. A file saying
    ``"poses": []`` with no reason in it is a run that found nothing, and it is
    read as one; a file that says it found nothing **and why** is a different
    claim, and the one this module is willing to make.
    """
    if not poses:
        raise ExportUnavailable(
            "there is no pose to export, so nothing was written. A run with no "
            "poses and a file saying so are different things, and this file "
            "will only be one of them"
        )
    return RunRecord(
        provenance=dict(provenance),
        poses=tuple(poses),
        selected_pose=int(selected_pose),
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        absent=tuple(absent),
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


def to_json(record: RunRecord) -> str:
    """The document as text.

    ``allow_nan=False`` on purpose. `json` writes a NaN as the bare token
    ``NaN``, which is not JSON and which most readers -- including
    ``json.loads`` in another language -- reject, so a document containing one
    is a document that cannot be read back. A term the engine returned as NaN is
    exported as a **record saying so**, not as a token nobody can parse.
    """
    return json.dumps(record.as_dict(), indent=1, sort_keys=True,
                      allow_nan=False)


def write_export(record: RunRecord, path: Path | str) -> Path:
    """Write it, and say where. UTF-8, LF, so the file is the same everywhere."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(record), encoding="utf-8", newline="\n")
    return p


def read_export(path: Path | str) -> dict:
    """Read one back, checking only that it is this kind of file.

    Deliberately **not** a validator of the numbers. The gate that proves the
    round trip reads the file with a plain `json.load` and compares it against
    the live objects, because a reader that shares code with the writer cannot
    disagree with the writer and therefore cannot prove anything. This one is
    here for the CLI and for a person: it confirms the schema and hands the
    document over, and it refuses a file whose `schema` is missing or different
    rather than reading it as something it is not.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ExportUnavailable(
            f"{path}: the top level is {type(doc).__name__}, not an object, so "
            f"this is not a {SCHEMA} document"
        )
    got = doc.get("schema")
    if got != SCHEMA:
        raise ExportUnavailable(
            f"{path}: schema is {got!r}, not {SCHEMA!r}. Either it is another "
            f"kind of file or it was written by a different version, and "
            f"reading it as this one would be a guess"
        )
    return doc


def default_export_path(root: Path | str, *, scoring: str, seed: Any,
                        poses: int, source: str = "run") -> Path:
    """Where an export goes when the caller does not choose.

    Under `dist/exports/`, beside the screenshots this window already writes,
    because a file the user cannot find is a file they do not have. The name
    carries the three things that make two sessions different, so a directory of
    exports is a directory of distinguishable runs rather than of `run.json`
    files.

    `source` decides the first word. A pose-file session has no run and no
    seed, and calling its file `run-vina-seed0-9poses.json` would put a run and
    a seed into a name for a thing that has neither -- the filename is a claim,
    and this is the one place a reader looks first.
    """
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(scoring))
    if source == "run":
        return (Path(root) / "dist" / "exports"
                / f"run-{safe}-seed{seed}-{poses}poses.json")
    return (Path(root) / "dist" / "exports"
            / f"poses-{safe}-{poses}poses.json")
