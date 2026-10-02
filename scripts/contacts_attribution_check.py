"""What the contact analysis is attributed to, and what it does not say.

`contacts_criteria_check.py` checks the *criteria*: a synthetic hydrogen bond built to a
known geometry, moved one degree and one ångström at a time across each
threshold, and a pose translated 400 Å away to prove the filter can empty
itself. That is the right thing to check and it is checked well.

This file checks the question beside it. A contact table is a list of
authoritative-looking lines, and every line says *which residue* — so the ways
this module can be wrong without any threshold being wrong are the ones about
attribution and invariance rather than about cut-offs:

* **Where the scene is put must not matter.** A rotation of everything together
  is a different test from a 400 Å translation: it moves every coordinate in
  three axes at once, and any code that read an absolute position — a guess at
  a donor, a comparison against a fixed plane — would survive the translation
  and fail the rotation.
* **One interaction must come back as one row.** `find_contacts` promises that a
  hydrogen bond is not repeated as a plain close contact. It was, in both
  directions, because the suppression recorded the hydrogen's index while the
  close-contact search skips hydrogens; and no threshold check could see it,
  because each of the two rows is individually true.
* **The count a user reads must be the count they can account for.**
  `residue_summary` totals its per-residue counts, and it drops any contact
  whose partner has no residue label. That is defensible — a summary of
  residues cannot list a residue with no name — but it is *silent*, so the
  number in the panel is smaller than the number of contacts above it and
  nothing says so. This file pins the behaviour in both directions: that the
  drop happens, and that `Contact.label()` still names the atom honestly, so
  the table and the summary disagree visibly rather than quietly.
* **The word in the `kind` column is not a chemical verdict.** It is a
  two-valued summary of which elements are present, and a methyl carbon 3.9 Å
  from a backbone nitrogen is reported as `polar`. That is a choice somebody
  made, so it is pinned here rather than left to be over-read.

Nothing here needs a docking run or a fixture file: every case is synthetic
and exact. Run:  python scripts/contacts_attribution_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking.workbench import MoleculeView  # noqa: E402
from opendocking.workbench import contacts as C  # noqa: E402
from opendocking.workbench import structure as S  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0

#: How many checks this file records, **counted by running this file** and not
#: derived from the source above.
#:
#: `F:\python310\python.exe scripts\contacts_attribution_check.py` with
#: `PYTHONIOENCODING=utf-8` and `PYTHONPATH=dock-py\python`: **`18 checks
#: passed`, exit 0**, in 0.4 s. The pin is that run's 18 plus the tally check
#: below, so 19, and the run that has to agree with the pin is the *second* run
#: rather than the one the 18 came from.
#:
#: **18 is not the number of `check()` sites, and the difference is the whole
#: reason this note exists.** This file has 16 call sites and runs 18 checks,
#: because two of the sites sit in `for` loops: one over the two
#: donate-directions in section 2, one over the `("UNL", "REC")` placeholder
#: family in section 4. A pin copied from a site census would be 16 and would be
#: wrong by two, and a census-derived pin is the specific mistake this constant
#: is here to avoid -- so the two numbers are stated separately and the pin is
#: the run's.
#:
#: **Why 18 is stable rather than merely observed, which is the part that makes
#: it safe to pin.** Both loops iterate over literal tuples written above them,
#: so their multipliers are 2 and 2 no matter what the module under test
#: returns. Nothing else in this file counts anything: every `find_contacts`
#: result is consumed as a verdict inside a `check(...)` call and never as a
#: loop bound or a slice, and the file's own docstring says why that is
#: possible -- every case here is synthetic and exact, so no fixture file and no
#: docking run stands between the source and the number. The two shapes that
#: would move it are both edits: a site added or deleted, or a loop given a
#: length that comes from data.
EXPECTED_CHECKS = 19


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


class _Atoms:
    """The smallest thing `residue_labels` will accept as a structure."""

    def __init__(self, atoms):
        self.atoms = atoms


def build(rows, names=None, residues=None):
    """A MoleculeView from `(element, x, y, z, bonded_to_index_or_None)` rows.

    `names` and `residues` are what a parsed file would carry; left out, the
    view has no structure and every label is `""`, which is the state the
    synthetic fixtures in `contacts_criteria_check.py` are already in.
    """
    coords = np.asarray([(r[1], r[2], r[3]) for r in rows], np.float32).reshape(-1, 3)
    elements = [r[0] for r in rows]
    bonds = [(i, r[4]) for i, r in enumerate(rows) if r[4] is not None]
    view_kw = {}
    if names is not None or residues is not None:
        names = names or [f"X{i}" for i in range(len(rows))]
        residues = residues or [("ALA", i + 1, "A") for i in range(len(rows))]
        view_kw["structure"] = _Atoms([
            S.AtomRecord(
                serial=i + 1, xyz=(r[1], r[2], r[3]), element=r[0],
                name=names[i], resname=residues[i][0], resid=residues[i][1],
                chain=residues[i][2],
            )
            for i, r in enumerate(rows)
        ])
    return MoleculeView(
        name="synthetic",
        coords=coords,
        elements=elements,
        bonds=np.asarray(bonds, np.int32).reshape(-1, 2),
        **view_kw,
    )


def hbond_at(angle_deg=180.0, h_to_a=1.8, receptor_name="O", receptor_residue=None):
    """Donor O, its H 1.0 A along +x, and an acceptor at a chosen geometry.

    Mirrors the helper in `contacts_criteria_check.py` so the two files build the same
    geometry and a change to one is visible against the other.
    """
    phi = math.radians(angle_deg)
    dx, dy, dz = (math.cos(math.pi - phi), math.sin(math.pi - phi), 0.0)
    a = (1.0 + h_to_a * dx, h_to_a * dy, h_to_a * dz)
    pose = build([("O", 0.0, 0.0, 0.0, None), ("H", 1.0, 0.0, 0.0, 0)])
    kwargs = {"names": [receptor_name]}
    if receptor_residue is not None:
        kwargs["residues"] = [receptor_residue]
    return pose, build([("O", a[0], a[1], a[2], None)], **kwargs)


def identity(contacts):
    """The part of a contact that is geometry, with the printing left out."""
    return sorted(
        (c.kind, c.self_index, c.partner_index, round(c.distance, 4),
         round(c.angle, 4), c.partner_residue, c.self_element, c.partner_element)
        for c in contacts
    )


def rigidly_move(view, rot=(23.0, -47.0, 88.0), shift=(100.0, -200.0, 37.0)):
    """Rotate about the origin by three given angles, then translate."""
    ax, ay, az = (math.radians(v) for v in rot)
    rx = np.array([[1, 0, 0],
                   [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)],
                   [0, 1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0],
                   [math.sin(az), math.cos(az), 0],
                   [0, 0, 1]])
    rot_m = rz @ ry @ rx
    return MoleculeView(
        name=view.name,
        coords=(view.coords @ rot_m.T + np.asarray(shift)).astype(np.float32),
        elements=list(view.elements),
        bonds=np.asarray(view.bonds, np.int32).reshape(-1, 2),
        structure=view.structure,
    )


def main() -> int:
    section("1. where the scene is put must not matter")
    pose, receptor = hbond_at()
    base = C.find_contacts(pose, receptor)
    moved_pose = rigidly_move(pose)
    moved_rec = rigidly_move(receptor)
    same = identity(C.find_contacts(moved_pose, moved_rec)) == identity(base)
    check(
        "rotating and translating the whole scene changes nothing",
        same,
        f"{len(base)} contact(s) before and after a three-axis rotation by "
        f"(23, -47, 88) degrees and a shift of (100, -200, 37) A: "
        f"{[c.kind for c in base]} unchanged. A 400 A translation only moves one "
        f"axis, so any code reading an absolute position can pass it and still "
        f"be wrong; this moves all three"
        if same else
        f"the contact set changed under a rigid motion of the whole scene: "
        f"{identity(base)} became {identity(C.find_contacts(moved_pose, moved_rec))}",
    )
    # The other direction, which is the one that stops this being "fixed" by
    # making each side independently translation-blind: a pose rotated against a
    # fixed receptor is a different geometry, and a different answer is correct.
    spun = C.find_contacts(
        rigidly_move(pose, rot=(0.0, 0.0, 97.0), shift=(0.0, 0.0, 0.0)),
        receptor,
    )
    check(
        "but rotating the pose against a fixed receptor is a different geometry",
        identity(spun) != identity(base),
        f"{[c.kind for c in base]} becomes {[c.kind for c in spun]} when only "
        f"the pose is turned 97 degrees about z. Asserted so that 'invariant' "
        f"cannot be satisfied by ignoring one side's coordinates"
        if identity(spun) != identity(base) else
        "turning the pose alone left the answer identical, which no rigid "
        "motion of one side of a pair can do",
    )

    section("2. one interaction, one row, in both directions")
    # Direction 1: the pose donates. Direction 2: the receptor donates, which is
    # the common case in a docked pose and the one whose key was equally broken.
    forward = C.find_contacts(*hbond_at())
    reverse_pose, reverse_receptor = hbond_at()
    reverse = C.find_contacts(reverse_receptor, reverse_pose)
    for title, contacts, expect_heavy in (
        ("the pose donates", forward, (0, 0)),
        ("the receptor donates", reverse, (0, 0)),
    ):
        hb = [c for c in contacts if c.kind == "hbond"]
        others = [c for c in contacts if c.kind != "hbond"]
        check(
            f"one hydrogen bond and nothing else, when {title}",
            len(hb) == 1 and not others,
            f"{len(hb)} hydrogen bond(s) and {len(others)} close contact(s) "
            f"{[c.kind for c in others]}. `find_contacts` promises the bond is "
            f"not repeated as a plain close contact, and the implied pair is "
            f"{expect_heavy} at 2.80 A, inside the 4.0 A cut"
            if len(hb) == 1 and not others else
            f"expected exactly the bond; got {[c.kind for c in contacts]}",
        )
    check(
        "and the heavy-atom pair the bond implies is the one that is gone",
        all(
            not any(
                c.kind != "hbond" and {c.self_index, c.partner_index} == {0, 0}
                for c in contacts
            )
            for contacts in (forward, reverse)
        ),
        "in both directions the close contact between the two heavy atoms, "
        f"{(2.80 - forward[0].distance) if forward else 0:.2f} A further out "
        "than the hydrogen it sits on, is absent, so the row is the bond and not "
        "the bond plus its own shadow",
    )

    # The negative direction: a suppression that satisfied itself by dropping
    # every close contact would pass both checks above. Tightening the *distance*
    # criterion is the way to do it on identical coordinates: the 1.8 A
    # H...acceptor pair stops being a bond, and the 2.8 A O...O pair is still
    # inside the 4.0 A cut and must come back. (Tightening the *angle* cannot do
    # it on a 180 deg bond, which is the first version of this check and why it
    # reported that the close-contact path had broken.)
    loose = C.find_contacts(*hbond_at(180.0, 1.8), hbond_max=1.0)
    check(
        "while a pair that is not a hydrogen bond still reports its contact",
        any(c.kind != "hbond" for c in loose),
        f"with the H...acceptor cut-off pulled in to 1.0 A the 1.8 A pair is no "
        f"longer a bond, and it comes back as {[c.kind for c in loose]}. The "
        f"suppression follows the hydrogen bond, not the geometry: these are the "
        f"same coordinates as the two checks above"
        if any(c.kind != "hbond" for c in loose) else
        "pulling the hydrogen-bond cut-off in left nothing at all, so the "
        "close-contact path has stopped working rather than being filtered",
    )
    # And the other negative direction: an unpaired hydrogen is not a bond, so
    # its contact must not be suppressed either. Same code path, different
    # reason, and a suppression keyed on "there is a hydrogen nearby" would
    # break it.
    orphan = C.find_contacts(
        build([("O", 6.0, 0.0, 0.0, None), ("H", 1.0, 0.0, 0.0, None)]),
        build([("O", 2.8, 0.0, 0.0, None)]),
    )
    check(
        "and an unpaired hydrogen's contact is not suppressed either",
        any(c.kind != "hbond" for c in orphan),
        f"{[c.kind for c in orphan]}: the hydrogen 5 A from any polar atom "
        f"cannot be vouched for, so there is no bond to suppress and the O...O "
        f"contact at 2.80 A is reported"
        if any(c.kind != "hbond" for c in orphan) else
        "an unpaired hydrogen suppressed a real close contact, which means the "
        "suppression is keyed on a hydrogen being nearby rather than on a bond "
        "existing",
    )

    section("3. the count in the summary is the count you can account for")
    labelled = C.find_contacts(
        build([("C", 0.0, 0.0, 0.0, None)]),
        build([("N", 3.1, 0.0, 0.0, None)], names=["NZ"],
              residues=[("TYR", 29, "A")]),
    )
    summary = C.residue_summary(labelled)
    counted = sum(row[2] for row in summary)
    check(
        "a residue's contact count is the number of contacts attributed to it",
        counted == len(labelled),
        f"{len(labelled)} contact(s) found, {counted} accounted for in "
        f"{len(summary)} summary row(s) {summary}",
    )
    bondy = C.find_contacts(*hbond_at(receptor_residue=("THR", 2, "A")))
    rows = C.residue_summary(bondy)
    check(
        "a hydrogen bond is counted once in the summary, not twice",
        sum(row[2] for row in rows) == len(bondy)
        and sum(row[1] for row in rows) == len([c for c in bondy if c.kind == "hbond"]),
        f"{len(bondy)} contact(s) found and {sum(r[2] for r in rows)} counted "
        f"across {rows}. This is the number a user reads, and before the "
        f"implied-pair key was fixed every bond added one to it here while "
        f"adding nothing to the table above",
    )

    section("4. what has no residue name, and what that costs")
    water = C.find_contacts(
        build([("C", 0.0, 0.0, 0.0, None)]),
        build([("O", 3.1, 0.0, 0.0, None)], names=["O"],
              residues=[("HOH", 180, "A")]),
    )
    dropped = C.residue_summary(water)
    check(
        "a contact with no residue label is dropped from the summary, silently",
        len(water) == 1 and not dropped,
        f"find_contacts reports {len(water)} contact(s); residue_summary "
        f"reports {len(dropped)} row(s). A crystal water is not a residue, so "
        f"it cannot be listed as one -- but the panel above and the panel below "
        f"disagree and nothing says so"
        if len(water) == 1 and not dropped else
        f"expected one contact and no summary row, got {len(water)} and "
        f"{len(dropped)}",
    )
    check(
        "though the contact line still names the atom rather than going blank",
        bool(water) and "atom " in water[0].label(),
        f"{water[0].label()!r} -- the residue field degrades to the atom name "
        f"rather than to nothing, so the table above is honest about having no "
        f"residue and the summary below is the one that stays quiet"
        if water and "atom " in water[0].label() else
        f"expected the residue field to degrade to the atom name, got "
        f"{water[0].label()!r}" if water else "no contact to print",
    )
    # The same drop for a placeholder the writers emit, so the behaviour is
    # pinned for the whole family rather than for water alone.
    for resname in ("UNL", "REC"):
        cs = C.find_contacts(
            build([("C", 0.0, 0.0, 0.0, None)]),
            build([("O", 3.1, 0.0, 0.0, None)], names=["O"],
                  residues=[(resname, 1, "A")]),
        )
        check(
            f"and for the {resname} placeholder as well",
            len(cs) == 1 and not C.residue_summary(cs),
            f"{resname} 1 A: {len(cs)} contact(s), "
            f"{len(C.residue_summary(cs))} summary row(s), label "
            f"{build([('O', 3.1, 0.0, 0.0, None)], names=['O'], residues=[(resname, 1, 'A')]).residue_labels()[0]!r}"
            if len(cs) == 1 and not C.residue_summary(cs) else
            f"{resname}: {len(cs)} contacts, "
            f"{len(C.residue_summary(cs))} summary rows",
        )

    section("5. two chains, one residue number")
    two_chains = build(
        [("N", 3.1, 0.0, 0.0, None), ("O", 9.1, 0.0, 0.0, None)],
        names=["NZ", "OG"],
        residues=[("TYR", 29, "A"), ("TYR", 29, "B")],
    )
    labels = two_chains.residue_labels()
    chain_rows = C.residue_summary(
        C.find_contacts(build([("C", 0.0, 0.0, 0.0, None), ("C", 6.0, 0.0, 0.0, None)]),
                        two_chains)
    )
    check(
        "two residues numbered 29 in two chains stay two residues",
        labels == ["TYR 29A", "TYR 29B"] and len(chain_rows) == 2,
        f"labels {labels} and {len(chain_rows)} summary row(s) {chain_rows}. The "
        f"chain is part of the label, which is what stops one site's contacts "
        f"being merged into another's -- a silent conflation would look like a "
        f"stronger single contact"
        if labels == ["TYR 29A", "TYR 29B"] and len(chain_rows) == 2 else
        f"labels {labels}, {len(chain_rows)} summary row(s) {chain_rows}",
    )

    section("6. the word in the kind column is not a verdict")
    # Pinned as a table, because the whole content of that column is these
    # eight lines. `_classify` is private, so this is a pin on behaviour, not on
    # an implementation: any refactor that changes a word changes one of these.
    mapping = {
        ("C", "N"): "polar", ("N", "N"): "polar", ("O", "C"): "polar",
        ("C", "C"): "hydrophobic", ("C", "S"): "hydrophobic",
        ("S", "S"): "hydrophobic", ("C", "Cl"): "hydrophobic",
    }
    got = {pair: C._classify(*pair) for pair in mapping}
    if got == mapping:
        note = [
            "; ".join(f"{a}{b}->{k}" for (a, b), k in sorted(mapping.items()))
            + ". A methyl carbon 3.9 A from a backbone nitrogen reads `polar`, "
            "because the rule is 'is either atom N or O' -- the column is a "
            "two-valued summary of which elements are present, not a statement "
            "about the chemistry of the pair"
        ]
    else:
        note = [f"expected {mapping}, measured {got}"]
    check(
        "the kind word is decided by the two elements and nothing else",
        got == mapping,
        "; ".join(note),
    )
    c_n = C.find_contacts(
        build([("C", 0.0, 0.0, 0.0, None)]),
        build([("N", 3.9, 0.0, 0.0, None)], names=["NZ"],
              residues=[("ALA", 12, "A")]),
    )
    check(
        "and a methyl carbon 3.9 A from a backbone nitrogen is reported as polar",
        [c.kind for c in c_n] == ["polar"],
        f"{[c.label() for c in c_n]}. Nothing is wrong with the geometry; the "
        f"point is that the word describes the elements, and a reader who takes "
        f"it as a polar interaction is reading more than the filter measured"
        if [c.kind for c in c_n] == ["polar"] else
        f"got {[c.kind for c in c_n]}",
    )

    section("7. a donor guessed by proximity, not read from the bond table")
    # Everything above resolves each hydrogen through `bonds`, so the module's
    # other inference -- "the nearest polar atom within a bond length" -- is
    # never reached. It is the one place the module guesses, and a guess is
    # where an absolute coordinate sneaks in: measuring the H against the world
    # origin instead of against its own donor works in the coordinates the
    # fixture was written in and stops working the moment the scene is turned.
    loose_h = build([("O", 0.0, 0.0, 0.0, None), ("H", 1.0, 0.0, 0.0, None)])
    acceptor = build([("O", 2.8, 0.0, 0.0, None)], names=["OG"],
                     residues=[("THR", 2, "A")])
    guessed = C.find_contacts(loose_h, acceptor)
    spun_guess = C.find_contacts(rigidly_move(loose_h), rigidly_move(acceptor))
    check(
        "a hydrogen with no bond-table entry is still resolved to its donor",
        [c.kind for c in guessed] == ["hbond"],
        f"{[c.label() for c in guessed]}. `bonds` is empty here, so this bond "
        f"exists only because the module looked for a polar atom within 1.35 A "
        f"of the hydrogen -- the guess, not the table"
        if [c.kind for c in guessed] == ["hbond"] else
        f"expected the proximity guess to find the donor, got "
        f"{[c.kind for c in guessed]}",
    )
    check(
        "and that guess is not measured against a fixed place in space",
        identity(spun_guess) == identity(guessed),
        f"the same guess after a three-axis rotation and a shift of "
        f"(100, -200, 37) A gives {identity(spun_guess)}. Distances and angles "
        f"between the two atoms are all this path has to go on; a guess that "
        f"read an absolute coordinate would survive a one-axis translation and "
        f"fail here"
        if identity(spun_guess) == identity(guessed) else
        f"the guessed donor changed when the whole scene was moved rigidly: "
        f"{identity(guessed)} became {identity(spun_guess)}, so the inference "
        f"depends on where the scene is put",
    )

    # The pin, asserted on the way out rather than only in the declaration. The
    # `+ 1` is this check, which has not been counted yet when the comparison is
    # built, and it is placed before the failure report so a wrong tally is
    # reported as a failed check rather than only as a number in the summary.
    check("this file's own count is the count it declares",
          CHECKS + 1 == EXPECTED_CHECKS,
          f"{CHECKS} ran before this one and {EXPECTED_CHECKS} are declared")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print(f"{CHECKS} checks passed (expected {EXPECTED_CHECKS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
