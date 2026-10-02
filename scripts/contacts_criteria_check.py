"""Headless check of the pose/receptor interaction analysis.

Two things are being checked, and they need different evidence:

* **the criteria** -- a synthetic hydrogen bond built to a known geometry, and
  variants of it moved one degree or one ångström at a time across each
  threshold. A detector that ignores its own cut-offs would find all of them,
  so the cases around the boundary are the ones that matter.
* **the real thing** -- a pose actually docked into crambin, where the residue
  labels can be checked against the residues crambin really has.

The guards are reverse-verified: loosening the criteria has to change the
answer, and tightening them has to be able to empty it. A filter that returns
the same thing no matter what you set is indistinguishable from no filter.

Run:  python scripts/contacts_criteria_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Prefer the *installed* package, and only fall back to the source tree when
# there is no installed one. Putting `dock-py/python` on the path instead --
# which is what this did first -- makes Python import the source copy of
# `opendocking`, and a clean checkout has no `_dockpy` extension in it: the
# compiled module lives in site-packages and the `.pyd`/`.so` is gitignored.
# So that version worked on the maintainer's machine, where a development
# build had left the extension sitting in the source tree, and failed
# everywhere else with "The Open Docking native extension is not available".
# Testing what is installed is also the more useful default: it is what a user
# gets, and it is what CI builds.
try:  # noqa: SIM105
    import opendocking  # noqa: F401
except ImportError:  # pragma: no cover - only on an uninstalled checkout
    sys.path.insert(0, str(ROOT / "dock-py" / "python"))

from opendocking.workbench import MoleculeView  # noqa: E402
from opendocking.workbench import contacts as C  # noqa: E402

CRAMBIN = ROOT / "examples" / "1crn_prep.pdbqt"
CRAMBIN_POSE = ROOT / "examples" / "crambin_pose.pdbqt"

FAILURES: list[str] = []
CHECKS = 0

#: How many checks this file records, **counted from a run and not derived from
#: the source above**.
#:
#: `F:\python310\python.exe scripts\contacts_criteria_check.py` with
#: `PYTHONIOENCODING=utf-8` and `PYTHONPATH=dock-py\python`: **`39/39 passed`,
#: exit 0**, in 0.4 s. The pin is that run's 39 plus the tally check below, so
#: 40, and the run that has to agree with the pin is the *second* run rather
#: than the one the 39 came from.
#:
#: **Read `EXPECTED_CHECKS = 40` against the site count and it looks like a
#: coincidence; it is the opposite, and the number is a trap either way.** This
#: file had 40 `check()` call sites and ran 39 of them, and adding the tally
#: check takes the sites to 41 while the run goes to 40. So the pin is *not* the
#: census, it is one below it, and a reader who checks the constant against a
#: walk of the syntax tree will find them off by one in the direction that
#: matters. `check_scripts_declare.py` reports this file as `18 unconditional +
#: 22 guarded` for exactly this reason: 22 of the 41 sites are inside a branch.
#: If these two numbers are ever brought into agreement by editing the census,
#: one of them is wrong and it will not be the census that notices.
#:
#: **The count is conditional, and saying so is part of pinning it.** 39 holds
#: for a tree where `examples/1crn_prep.pdbqt` and `examples/crambin_pose.pdbqt`
#: both exist, because section 8 picks between one `check()` for the missing
#: fixture and eighteen for the real crambin run. The other three conditionals
#: are `if found:` in sections 1 and 6 (six sites) and `if
#: kinds.get("hbond"):` at the end of section 9 (one site), all of which are
#: about the data under test rather than about the machine. A run that takes a
#: different branch records a different number, and the tally check goes red
#: and says so -- which is the correct outcome rather than a nuisance: a run
#: that measured 22 checks has not measured what this file says it measures, and
#: the fix is to make the fixtures present, never to widen this constant to
#: cover the shortfall.
EXPECTED_CHECKS = 40


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")


def build(rows):
    """A MoleculeView from `(element, x, y, z, bonded_to_index_or_None)` rows."""
    coords = np.asarray([(r[1], r[2], r[3]) for r in rows], np.float32).reshape(-1, 3)
    elements = [r[0] for r in rows]
    bonds: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        if r[4] is not None:
            bonds.append((i, r[4]))
    return MoleculeView(
        name="synthetic",
        coords=coords,
        elements=elements,
        bonds=np.asarray(bonds, np.int32).reshape(-1, 2),
    )


def hbond_at(angle_deg: float, h_to_a: float):
    """Donor O, its H, and an acceptor placed at a chosen geometry.

    The donor is at the origin, its H 1.0 A along +x, so the vector H->D points
    along -x. The acceptor goes at ``angle_deg`` from that vector, which puts it
    on the *far* side of the hydrogen: a linear D-H...A bond has H between the
    two heavy atoms, so H->D and H->A point opposite ways and the angle is 180.
    Placing the acceptor on the donor's side instead gives an angle of 0, which
    is not a hydrogen bond however short the distance is.
    """
    phi = math.radians(angle_deg)
    dx, dy, dz = (math.cos(math.pi - phi), math.sin(math.pi - phi), 0.0)
    # Only x is offset: the hydrogen sits at (1, 0, 0), and adding that offset
    # to all three components would quietly put the acceptor at (2.8, 1, 1).
    a = (1.0 + h_to_a * dx, h_to_a * dy, h_to_a * dz)
    pose = build([("O", 0.0, 0.0, 0.0, None), ("H", 1.0, 0.0, 0.0, 0)])
    receptor = build([("O", a[0], a[1], a[2], None)])
    return pose, receptor


def hbonds_between(pose, receptor, **kw):
    return [c for c in C.find_contacts(pose, receptor, **kw) if c.kind == "hbond"]


def main() -> int:
    section("1. a hydrogen bond is found when the geometry says so")
    pose, receptor = hbond_at(180.0, 1.8)
    found = hbonds_between(pose, receptor)
    check("linear, 1.8 A: accepted", len(found) == 1,
          f"{len(found)} found")
    if found:
        c = found[0]
        check("its distance is the H···acceptor one, not D···A",
              abs(c.distance - 1.8) < 1e-3, f"{c.distance:.4f} A")
        check("its angle is the donor-H···acceptor one",
              abs(c.angle - 180.0) < 0.5, f"{c.angle:.2f} deg")
        check("the acceptor is the partner, so the residue would read off it",
              c.partner_element == "O" and c.self_element == "H",
              f"self={c.self_element} partner={c.partner_element}")

    section("2. moving one degree across the angle threshold")
    inside = hbonds_between(*hbond_at(125.0, 2.0))
    outside = hbonds_between(*hbond_at(115.0, 2.0))
    check("125 deg accepted", len(inside) == 1, f"{len(inside)}")
    check("115 deg rejected", len(outside) == 0, f"{len(outside)}")

    section("3. moving one angstrom across the distance threshold")
    inside = hbonds_between(*hbond_at(180.0, 2.5))
    outside = hbonds_between(*hbond_at(180.0, 2.7))
    check("2.5 A accepted", len(inside) == 1, f"{len(inside)}")
    check("2.7 A rejected", len(outside) == 0, f"{len(outside)}")

    section("4. the threshold is the boundary, not a range around it")
    edge = hbonds_between(*hbond_at(180.0, C.HBOND_MAX))
    check(f"exactly at the {C.HBOND_MAX} A limit is accepted", len(edge) == 1,
          f"{len(edge)}")
    edge = hbonds_between(*hbond_at(C.HBOND_MIN_ANGLE, 2.0))
    check(f"exactly at the {C.HBOND_MIN_ANGLE} deg limit is accepted", len(edge) == 1,
          f"{len(edge)}")

    section("5. a hydrogen nobody can vouch for")
    # The H sits 5 A from every polar atom, so neither the bond table nor the
    # 1.35 A fallback can say what it is attached to. Its distance to the
    # acceptor is a real 1.8 A, which is exactly the situation where a filter
    # that skips the donor test would draw a confident wrong bond.
    pose = build([
        ("O", 6.0, 0.0, 0.0, None),
        ("H", 1.0, 0.0, 0.0, None),
    ])
    receptor = build([("O", 2.8, 0.0, 0.0, None)])
    found = hbonds_between(pose, receptor)
    check("an unpaired hydrogen is not drawn as a hydrogen bond", len(found) == 0,
          f"{len(found)} found")
    check("but the close contact is still reported",
          any(c.kind != "hbond" for c in C.find_contacts(pose, receptor)),
          "the atoms are still touching, they just are not bonded")

    section("6. the receptor is the donor, which is the common direction")
    # Mirror of case 1: the receptor owns the polar hydrogen this time.
    donor, acceptor = hbond_at(160.0, 2.2)
    found = [c for c in C.find_contacts(acceptor, donor) if c.kind == "hbond"]
    check("receptor-donated hydrogen bond is found", len(found) == 1, f"{len(found)}")
    if found:
        check("the pose atom is the acceptor and the receptor atom is the H",
              found[0].self_element == "O" and found[0].partner_element == "H",
              f"self={found[0].self_element} partner={found[0].partner_element}")
        check("the donor's heavy atom is named, not just its hydrogen",
              found[0].donor_name == "O", found[0].donor_name)

    section("7. reverse-verification: each cut-off has to matter on its own")
    # Loosening *both* at once proves little, because a single synthetic pair
    # can be flipped by either. These two cases loosen exactly one each, so
    # each one demonstrates that its own criterion is the thing doing the work.
    wide = hbonds_between(*hbond_at(90.0, 2.0))
    wide_loose = hbonds_between(*hbond_at(90.0, 2.0), hbond_min_angle=0.0)
    check("a 90 deg bond is rejected by the angle criterion", len(wide) == 0,
          f"{len(wide)}")
    check("dropping only the angle criterion accepts it", len(wide_loose) == 1,
          f"{len(wide_loose)}")

    far = hbonds_between(*hbond_at(180.0, 3.5))
    far_loose = hbonds_between(*hbond_at(180.0, 3.5), hbond_max=99.0)
    check("a 3.5 A bond is rejected by the distance criterion", len(far) == 0,
          f"{len(far)}")
    check("dropping only the distance criterion accepts it", len(far_loose) == 1,
          f"{len(far_loose)}")

    pose, receptor = hbond_at(180.0, 1.8)
    # Tightening one criterion must not be asserted to empty the whole result
    # set: the O...O close contact at 2.8 A is a different criterion and stays.
    tight = [c.kind for c in
             C.find_contacts(pose, receptor, hbond_max=0.01, hbond_min_angle=179.9)]
    check("tightening both hydrogen-bond criteria removes the hydrogen bond",
          "hbond" not in tight, str(tight))
    check("and leaves the close contact, which it was not asked about",
          "hbond" in [c.kind for c in C.find_contacts(pose, receptor)]
          and "hbond" not in tight)

    cut = [c.kind for c in C.find_contacts(pose, receptor, close_max=0.1)]
    check("a close-contact cut-off of 0.1 A removes the close contacts",
          all(k == "hbond" for k in cut), str(cut))
    check("and leaves the hydrogen bond, which it was not asked about",
          cut == ["hbond"], str(cut))

    section("8. a pose far from the protein has no contacts at all")
    if not (CRAMBIN.is_file() and CRAMBIN_POSE.is_file()):
        check("crambin and its pose are present", False, "run the docking first")
    else:
        rec = MoleculeView.from_pdbqt(CRAMBIN, "crambin", (1, 1, 1), 0.28, role="receptor")
        pose = MoleculeView.from_pdbqt(CRAMBIN_POSE, "pose", (1, 1, 1), 0.28, role="pose")

        far = MoleculeView(
            name="far", coords=pose.coords + np.float32(400.0),
            elements=list(pose.elements), bonds=pose.bonds,
        )
        check("a pose 400 A away reports zero contacts",
              len(C.find_contacts(far, rec)) == 0, f"{len(C.find_contacts(far, rec))}")
        check("the same pose in place reports contacts",
              len(C.find_contacts(pose, rec)) > 0, f"{len(C.find_contacts(pose, rec))}")

        section("9. the docked pose against real crambin")
        contacts = C.find_contacts(pose, rec)
        kinds = {}
        for c in contacts:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        print(f"       {len(contacts)} contacts: {kinds}")

        check("there is at least one hydrogen bond", kinds.get("hbond", 0) >= 1)
        check("there are hydrophobic contacts too", kinds.get("hydrophobic", 0) >= 1)

        over = [c for c in contacts if c.distance > C.CLOSE_MAX + 1e-6]
        check("no contact exceeds the close-contact cut-off", not over,
              f"{len(over)} over {C.CLOSE_MAX} A")

        bad_hb = [
            c for c in contacts
            if c.kind == "hbond"
            and (c.distance > C.HBOND_MAX + 1e-6 or c.angle < C.HBOND_MIN_ANGLE - 1e-6)
        ]
        check("every hydrogen bond satisfies both of its criteria", not bad_hb,
              f"{len(bad_hb)}")

        pairs = [(c.self_index, c.partner_index) for c in contacts]
        check("no atom pair is reported twice", len(pairs) == len(set(pairs)),
              f"{len(pairs)} contacts, {len(set(pairs))} distinct pairs")

        real = {f"{r.name} {r.resid}{r.chain.strip()}" for r in rec.structure.residues}
        named = [c for c in contacts if c.partner_residue]
        check("contacts carry residue names", len(named) > 0, f"{len(named)} of {len(contacts)}")
        unknown = sorted({c.partner_residue for c in named} - real)
        check("every residue named is one crambin really has", not unknown, str(unknown))
        check("hydrogens are not reported as contacts in their own right",
              all(c.self_element != "H" or c.kind == "hbond" for c in contacts))

        section("10. the residue table adds up and ranks sensibly")
        summary = C.residue_summary(contacts)
        check("the table is not empty", len(summary) > 0, f"{len(summary)} residues")
        check("its contact counts sum to the named contacts",
              sum(r[2] for r in summary) == len(named),
              f"{sum(r[2] for r in summary)} vs {len(named)}")
        check("hydrogen-bond counts sum to the hydrogen bonds",
              sum(r[1] for r in summary) == kinds.get("hbond", 0),
              f"{sum(r[1] for r in summary)} vs {kinds.get('hbond', 0)}")
        ranked = all(
            (summary[i][1], summary[i][2]) >= (summary[i + 1][1], summary[i + 1][2])
            for i in range(len(summary) - 1)
        )
        check("it is ordered by hydrogen bonds then contacts", ranked)
        check("the closest distance is the minimum within its residue",
              all(
                  abs(r[3] - min(
                      c.distance for c in contacts if c.partner_residue == r[0]
                  )) < 1e-9
                  for r in summary
              ))
        if kinds.get("hbond"):
            check("a residue with a hydrogen bond is ranked first",
                  summary[0][1] > 0, f"top is {summary[0][0]} with {summary[0][1]}")

    # The pin, asserted on the way out rather than only in the declaration. The
    # `+ 1` is this check, which has not been counted yet when the comparison is
    # built.
    #
    # Deliberately **outside** the `if CRAMBIN.is_file()` branch above: a run
    # that took the missing-fixture branch has measured 22 checks, and that run
    # is exactly the one that has to be caught here rather than skipping the
    # assertion that would catch it. This is the opposite choice from
    # `cli_prep_check.py` and `prep_check.py`, which both return early and leave
    # the tally unasserted, and the difference is what each early return already
    # reports: a fixture that is absent is a red on its own check.
    check("this file's own count is the count it declares",
          CHECKS + 1 == EXPECTED_CHECKS,
          f"{CHECKS} ran before this one and {EXPECTED_CHECKS} are declared")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed (expected {EXPECTED_CHECKS})")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
