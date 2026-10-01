"""Do the frames a change can break still get produced, and are they pictures?

Run:  F:\\python310\\python.exe scripts\\screenshot_frame_check.py

# The problem this closes

Five scripts under `scripts/` produce pictures, and until this file existed no
gate ran any of them:

    framing_selection_screens.py      probe_painted_frame.py
    representation_cartoon_screens.py representation_screens.py
    secondary_screens.py

`check_scripts_declare.py` records the consequence as a finding rather than an
inconvenience: five of the twelve files in its `EXCLUSIONS` table are "named by
no other file in the repository -- not imported, not run, not cited". A
screenshot script that no gate invokes is a script whose frames nobody looks at
on any given change, and looking at the frames is the one verification step in
this project that cannot be automated away, because a correct pixel count is not
the same thing as a correct picture. Two agents this session produced pictures,
a person read them and found things a count could not have told them.

So this file makes the frames a checked artefact rather than a manual one, in
two separate moves, because they answer two different questions.

**1. The scripts stop being orphans.** Every driver is named here by path, and
each one is a check. Rename a driver or delete it and this file goes red and
names it. That is a different claim from "somebody remembers to run it": it is
assertable, it is reachable (a file can be renamed), and it is proven reachable
in both directions by the mutation recorded below. What it does *not* cover is a
driver deleted from both trees at once -- inside either tree, absent and
never-written are the same observation, and the oracle for that is `git
ls-files`, which is not something a gate may depend on. That limit is stated
rather than papered over, in the same words the parity check uses for its own.

**2. The frames get measured, in the two ways a rendering check passes for the
wrong reason.** Those two ways are *blank* and *indistinguishable*, and they are
separate checks with separate numbers because they are separate defects:

* A frame that renders nothing satisfies every "did it render" question there
  is. Checks 3 and 4 ask whether the ribbon and the cartoon actually own pixels,
  measured against a render of nothing rather than against a modal colour --
  `secondary_render.ribbon_mask` exists because the backdrop is a gradient, and
  thresholding a gradient against one modal colour marks 413 044 of 630 000
  pixels as object when the ribbon covers 93 496.
* A frame that renders but cannot tell one thing from another satisfies every
  "is it non-blank" question there is. `secondary_render.degenerate()` says why:
  a classifier that finds no hydrogen bonds returns all-coil, and an all-coil
  ribbon is a perfectly good picture -- it renders, it is the right size, and it
  says nothing. Checks 7 and 8 ask whether the classes are separable in pixels.

**Both are reachable product states, and the mutations are what say so.** A
camera that frames nothing is a blank frame with the variants still rendering, so
checks 3 and 5 go red while 4, 6, 7 and 8 stay green. A classifier that returns
all-coil is a classifier that never worked, so check 8 goes red while 3, 4, 5,
6 and 7 stay green. Neither mutation reaches a state a correct implementation
cannot be in, and neither threshold was moved to make a run pass: the floors are
the fractions the frames actually measure, recorded at
`MIN_RIBBON_SHARE` and `MIN_CARTOON_SHARE` with the measured values beside them.

**What stays manual, and why that is not the same as nobody looking at it.**
Three of the five drivers need a live window or a live GL context and write
their PNGs for a person to read: `representation_screens.py`,
`framing_selection_screens.py` and `probe_painted_frame.py`. Their frames are
not produced here. What keeps them honest is this file naming all five (so a
rename is red) plus the numeric coverage that already exists for what they
photograph -- `representation_names_check.py` pins the representation vocabulary,
`viewport_framing_check.py` pins the framing, and `probe_painted_frame.py` is
itself a four-check gate that runs green. "Deliberately not produced here" and
"nobody runs it" are different claims and only one of them is safe to leave
unwritten.

**The skip convention.** Every check here registers a result in every branch. A
machine with no usable OpenGL 3.3 driver records eight skips with
`OffscreenRenderer.LAST_PROBLEM` as the reason and still reaches the pinned
total, so `EXPECTED_CHECKS` is the same number everywhere. A count that depends
on the machine is not a count.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for _p in (ROOT / "dock-py" / "python", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: The five drivers this file exists to stop orphaning, with the PNGs each one
#: writes. Named here rather than discovered, because discovery is the loophole:
#: a driver that stopped writing PNGs would quietly drop out of a directory scan
#: and the suite would report a smaller, greener list.
DRIVERS = (
    ("framing_selection_screens.py", ("pose_after_selection.png",)),
    ("probe_painted_frame.py", ()),
    ("representation_cartoon_screens.py", ("cartoon_whole.png",)),
    ("representation_screens.py", ("side_by_side.png",)),
    ("secondary_screens.py", ("ss_real.png",)),
)

#: The two drivers this file renders through itself, with the directory each
#: writes into and the frames this file measures there.
OFFSCREEN = {
    "secondary_screens.py": (
        ROOT / "dist" / "secondary_structure",
        ("ss_real", "ss_helix_only", "ss_sheet_only", "ss_coil_only"),
    ),
    "representation_cartoon_screens.py": (
        ROOT / "dist" / "cartoon",
        ("cartoon_whole", "ribbon_greyscale"),
    ),
}

W, H = 900, 700

#: Floors for "the frame owns pixels". Measured on the shipped receptor at
#: 900x700: the secondary-structure ribbon covers 93 496 px of 630 000 (14.8%)
#: and the cartoon a comparable fraction, so both floors sit an order of
#: magnitude below what they read and three orders above zero. A floor at zero
#: is not a check; a floor at the measured value is a threshold that only a
#: regression can cross.
MIN_RIBBON_SHARE = 0.02
MIN_CARTOON_SHARE = 0.02

#: The margin, in pixels, that the object's bounding box must keep from the frame
#: edge. A framing bug crops the object, and cropping is measured as a box that
#: reaches an edge -- which is a different failure from a box that covers the
#: whole frame, and the two are separate checks for that reason.
FRAME_MARGIN_PX = 8

#: Two frames must differ by at least this fraction of their pixels to count as
#: separable. The four secondary-structure frames measured on the shipped
#: receptor differ by 52 437, 92 128 and 84 953 px of 630 000 -- 8.3% to 14.6%
#: -- so the floor is a third of the smallest, which tolerates a driver change
#: without tolerating a classifier that stopped discriminating.
MIN_SEPARATION = 0.02

#: How many checks this file is supposed to record, in every environment.
#: Measured from a green run on this machine (15/15, exit 0) and unchanged when
#: `OffscreenRenderer.open` returns None, because a skip records a result and
#: counts toward the total. The derivation:
#:
#:   5  the five drivers, one check each (section 1)
#: + 2  the two offscreen drivers invoked, and their PNGs re-read from disk
#: + 2  the ribbon and the cartoon own pixels          (blank)
#: + 2  the ribbon and the cartoon are inside the frame (mis-framed)
#: + 3  the three classes differ; the classified frame differs from all three
#:      degenerate ones; the greyscale cartoon differs from the greyscale ribbon
#:                                                        (indistinguishable)
#: + 1  the classifier is not all one class
#: + 1  this check
#: = 16
#: Measured: 93 496 of 630 000 px (14.84%) for the ribbon and 129 092 (20.49%)
#: for the cartoon, which is why the two blank floors are 2%. The ribbon figure
#: is the same number `secondary_render.ribbon_mask` documents independently,
#: so it is not a threshold fitted to one run.
#: GATE-DECLARE 1
#: sites: 0 unconditional + 22 guarded
#: guards: sha256:d0d353e5c46800c3ddaab21b1ab89e210ac0a8fe5b3a7d208662e4278050282d
#:
#: **0 unconditional is the file's shape.** Every check routes through `ok` or
#: `bad` under an `if`, so the guarded column is the whole file; and each check is
#: two call sites and one result, so 22 sites and 16 results are the same 16
#: checks counted two ways. The site loop over `DRIVERS` is one site running five
#: times.
#:
#: Derived by lifting the auditor's own `_sites_of` and `_guards_digest` out of
#: `check_scripts_declare.py` and running them against this file, rather than by
#: letting the census compare them: the census measures the gates in its
#: `INVENTORY` and this file is not in it yet. The moment it is, the derivation
#: stops being mine and becomes the auditor's, which is the point of declaring
#: it here in the first place.
EXPECTED_CHECKS = 16

RESULTS: list[tuple[str, str, str]] = []


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def skip(name: str, why: str) -> None:
    RESULTS.append(("SKIP", name, why))
    print(f"[SKIP] {name}\n       {why}")


def section(title: str) -> None:
    print()
    print(f"-- {title}")


# --------------------------------------------------------------------------
# Section 1 -- the drivers are not orphans
# --------------------------------------------------------------------------
def check_the_drivers_exist() -> None:
    """Every driver named above is on disk and parses.

    Presence and parseability, not success. A driver that runs and fails is
    that driver's business and is reported by whoever runs it; what this file
    answers is the question the `EXCLUSIONS` table raised -- is the script that
    makes the pictures still here -- and that question has to be answerable on a
    machine with no window, where "run it" returns nothing.
    """
    section("every screenshot driver is named by a gate, so none of them is an orphan")
    for name, _pngs in DRIVERS:
        path = HERE / name
        if not path.is_file():
            bad(f"{name} is present and parses",
                f"{path} does not exist. This file names all five drivers, so a "
                f"rename or a deletion is red here rather than a capability "
                f"quietly leaving the release. (A driver removed from both "
                f"trees at once is not catchable from inside either one: absent "
                f"and never-written are the same observation.)")
            continue
        try:
            ast.parse(path.read_bytes().decode("utf-8", errors="replace")
                      .lstrip("\ufeff"))
        except SyntaxError as exc:
            bad(f"{name} is present and parses", f"{path}: SyntaxError {exc}")
            continue
        ok(f"{name} is present and parses",
           f"{path} ({path.stat().st_size} B). Named by "
           f"screenshot_frame_check.py, so it is no longer one of the files "
           f"nothing in the repository refers to.")


# --------------------------------------------------------------------------
# Section 2 -- the offscreen drivers are invoked, and their files re-read
# --------------------------------------------------------------------------
def run_driver(name: str) -> tuple[int, str]:
    """Run a driver the way a person would: as a script, in its own process.

    A subprocess rather than an import, for the reason the file exists: a
    driver is an entry point with a `main()`, and importing one would make this
    gate depend on its module-level side effects (three of them create an output
    directory at import time) instead of on what it produces.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(HERE / name)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, timeout=600,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def check_the_offscreen_drivers_run(renderer_problem: str | None) -> None:
    section("the two offscreen drivers are run by this gate")
    for name, (out_dir, stems) in OFFSCREEN.items():
        check = f"{name} runs here and its frames are on disk"
        if renderer_problem is not None:
            skip(check, f"no offscreen OpenGL context on this machine "
                        f"({renderer_problem}), and this driver needs one. The "
                        f"check is recorded rather than dropped so the total is "
                        f"the same everywhere.")
            continue
        code, log = run_driver(name)
        missing = [f"{s}.png" for s in stems if not (out_dir / f"{s}.png").is_file()]
        if code != 0 or missing:
            tail = " | ".join(log.strip().splitlines()[-3:])
            bad(check, f"exit {code}, {len(missing)} of {len(stems)} frame(s) "
                       f"missing after the run ({missing}). Last output: {tail}")
            continue
        sizes = [f"{s}.png {(out_dir / f'{s}.png').stat().st_size} B" for s in stems]
        ok(check, f"exit 0; re-read from disk after the run: {'; '.join(sizes)}. "
                  f"The file is measured after the save rather than trusted from "
                  f"the write call, because a PNG reported as written and absent "
                  f"afterwards has been observed in this repository.")


# --------------------------------------------------------------------------
# Section 3 -- blank, mis-framed, indistinguishable
# --------------------------------------------------------------------------
def _mask_box(mask: np.ndarray):
    """(rows, cols) where the object is, or None for an empty mask."""
    if mask is None or not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def check_the_frames(renderer_problem: str | None) -> None:
    """Render the two offscreen frames and ask the three questions.

    Every check below registers a result in every branch, so this function's
    contribution to the total is the same whether the frames rendered or the
    machine had no context.
    """
    section("the frames: not blank, not cropped, not indistinguishable")
    names = {
        "blank.ribbon": "the secondary-structure ribbon owns pixels",
        "blank.cartoon": "the cartoon owns pixels",
        "framed.ribbon": "the secondary-structure ribbon is inside the frame",
        "framed.cartoon": "the cartoon is inside the frame",
        "separate.classes": "the three secondary-structure classes differ in pixels",
        "separate.degenerate": "the classified ribbon differs from all three degenerate ones",
        "separate.geometry": "the greyscale cartoon differs from the greyscale ribbon",
        "classifier": "the classifier is not all one class",
    }
    if renderer_problem is not None:
        why = (f"no offscreen OpenGL context on this machine "
               f"({renderer_problem}). Recorded rather than dropped, so the "
               f"total is the same on a machine that can render and one that "
               f"cannot.")
        for text in names.values():
            skip(text, why)
        return

    from opendocking.workbench import MoleculeView
    from opendocking.workbench import cartoon_geometry as cg
    from opendocking.workbench.structure import parse_structure, secondary_structure

    from secondary_render import (OffscreenRenderer, degenerate, flat_states,
                                  ribbon_for_states, ribbon_mask)

    # The drivers' own framing functions, imported rather than restated: this
    # gate measures what the drivers measure, so a change to either driver's
    # `camera_for` is a change to what is asserted here.
    from representation_cartoon_screens import camera_for as cartoon_camera_for
    from secondary_screens import camera_for as ribbon_camera_for

    r = OffscreenRenderer.open(W, H)
    if r is None:  # pragma: no cover - the guard above normally catches this
        import secondary_render
        for text in names.values():
            skip(text, f"the context opened during the probe and then failed: "
                       f"{secondary_render.LAST_PROBLEM}")
        return

    text = (ROOT / "examples" / "1crn_prep.pdbqt").read_text(encoding="utf-8")
    structure = parse_structure(text, "1crn_prep")
    trace = structure.backbone()
    states = list(secondary_structure(trace, structure.atoms))
    xyz = np.asarray([a.xyz for a in structure.atoms], np.float64)

    empty = r.render([], ribbon_camera_for(structure), W, H)
    ribbon = r.render([(ribbon_for_states(structure, states), 1.0)],
                      ribbon_camera_for(structure), W, H)
    variants = {
        state: r.render([(ribbon_for_states(structure,
                                            flat_states(len(trace), state)), 1.0)],
                        ribbon_camera_for(structure), W, H)
        for state in ("helix", "sheet", "coil")
    }

    mol = MoleculeView.from_text(text, "1crn_prep", (0.6, 0.65, 0.7), 0.30,
                                 "receptor")
    guide, sides, _ = mol._backbone_guide_and_sides()
    whole = cg.cartoon(guide, sides, states)
    cartoon = r.render([(whole, 1.0)], cartoon_camera_for(xyz), W, H)

    # NOT closed here: the greyscale comparison at the end of this function
    # renders through the same context, and `OffscreenRenderer.render` calls
    # `fbo.use()`, which on a released moderngl object is an AttributeError
    # rather than a failure anybody would read as "no frame".
    ribbon_m = ribbon_mask(ribbon, empty)
    cartoon_m = ribbon_mask(cartoon, r.render([], cartoon_camera_for(xyz), W, H))
    total = float(W * H)

    # -- blank --------------------------------------------------------------
    share = None if ribbon_m is None else float(ribbon_m.sum()) / total
    if share is not None and share >= MIN_RIBBON_SHARE:
        ok(names["blank.ribbon"],
           f"{int(ribbon_m.sum())} of {int(total)} px ({share:.2%}) differ from "
           f"a render of nothing, floor {MIN_RIBBON_SHARE:.0%}. Measured against "
           f"an empty frame rather than a modal colour, because the backdrop is "
           f"a gradient and one modal colour marks most of an empty frame as "
           f"object.")
    else:
        bad(names["blank.ribbon"],
            f"the ribbon owns {share} of the frame against a floor of "
            f"{MIN_RIBBON_SHARE:.0%} ({int(ribbon_m.sum()) if ribbon_m is not None else 0} "
            f"of {int(total)} px). A frame that renders nothing satisfies every "
            f"'did it render' question, which is the first of the two ways this "
            f"check could pass for the wrong reason.")

    share = None if cartoon_m is None else float(cartoon_m.sum()) / total
    if share is not None and share >= MIN_CARTOON_SHARE:
        ok(names["blank.cartoon"],
           f"{int(cartoon_m.sum())} of {int(total)} px ({share:.2%}) differ from "
           f"a render of nothing, floor {MIN_CARTOON_SHARE:.0%}, "
           f"{len(whole.indices)} triangles.")
    else:
        bad(names["blank.cartoon"],
            f"the cartoon owns {share} of the frame against a floor of "
            f"{MIN_CARTOON_SHARE:.0%} ({int(cartoon_m.sum()) if cartoon_m is not None else 0} "
            f"of {int(total)} px, {len(whole.indices)} triangles)")

    # -- mis-framed --------------------------------------------------------
    for label, mask, floor in (("framed.ribbon", ribbon_m, MIN_RIBBON_SHARE),
                               ("framed.cartoon", cartoon_m, MIN_CARTOON_SHARE)):
        box = _mask_box(mask)
        if box is None:
            # An empty mask is a false answer to this question, not an absence
            # of one: "the object is inside the frame, with a margin" is false
            # when there is no object in the frame. Recording a skip here was
            # the first version of this file and the mutation matrix caught it
            # -- a blank frame produced a SKIP on this check while the blank
            # check beside it produced a FAIL, so one defect reported as one
            # failure and one shrug. Two claims, two verdicts.
            bad(names[label],
                f"the mask is empty, so nothing is inside the frame at all. "
                f"This check asks whether the object is inside the frame with a "
                f"{FRAME_MARGIN_PX} px margin, and an empty mask is a false "
                f"answer rather than an absent one")
            continue
        r0, r1, c0, c1 = box
        gaps = (c0, W - 1 - c1, r0, H - 1 - r1)
        if min(gaps) >= FRAME_MARGIN_PX:
            ok(names[label],
               f"the object occupies rows {r0}-{r1}, cols {c0}-{c1} of {W}x{H}; "
               f"the smallest gap to an edge is {min(gaps)} px against a floor of "
               f"{FRAME_MARGIN_PX}. A framing bug crops the object, and cropping "
               f"is a box that reaches an edge -- a different failure from a box "
               f"that fills the frame.")
        else:
            bad(names[label],
                f"the object reaches the frame edge: rows {r0}-{r1}, cols "
                f"{c0}-{c1}, gaps (left {gaps[0]}, right {gaps[1]}, top {gaps[2]}, "
                f"bottom {gaps[3]}) px against a floor of {FRAME_MARGIN_PX}. "
                f"Whatever is being clipped is not a framing anybody chose.")

    # -- indistinguishable --------------------------------------------------
    def differ(a, b):
        return float((a != b).any(axis=-1).sum()) / total

    pairs = {("helix", "sheet"): differ(variants["helix"], variants["sheet"]),
             ("helix", "coil"): differ(variants["helix"], variants["coil"]),
             ("sheet", "coil"): differ(variants["sheet"], variants["coil"])}
    worst = min(pairs.items(), key=lambda kv: kv[1])
    if worst[1] >= MIN_SEPARATION:
        ok(names["separate.classes"],
           "; ".join(f"{a} vs {b} {v:.2%}" for (a, b), v in sorted(pairs.items()))
           + f" of the frame, smallest {worst[1]:.2%} against a floor of "
             f"{MIN_SEPARATION:.0%}")
    else:
        bad(names["separate.classes"],
            f"{worst[0][0]} and {worst[0][1]} differ by {worst[1]:.2%} of the "
            f"frame against a floor of {MIN_SEPARATION:.0%} "
            + "; ".join(f"{a} vs {b} {v:.2%}" for (a, b), v in sorted(pairs.items()))
            + ". Two classes that render the same picture are one class as far as "
              "a user is concerned.")

    against = {state: differ(ribbon, variants[state])
               for state in ("helix", "sheet", "coil")}
    closest = min(against.items(), key=lambda kv: kv[1])
    if closest[1] >= MIN_SEPARATION:
        ok(names["separate.degenerate"],
           "; ".join(f"real vs all-{s} {v:.2%}" for s, v in sorted(against.items()))
           + f" of the frame, closest {closest[1]:.2%} against a floor of "
             f"{MIN_SEPARATION:.0%}")
    else:
        bad(names["separate.degenerate"],
            f"the classified ribbon differs from the all-{closest[0]} ribbon by "
            f"{closest[1]:.2%} of the frame against a floor of "
            f"{MIN_SEPARATION:.0%} "
            + "; ".join(f"real vs all-{s} {v:.2%}" for s, v in sorted(against.items()))
            + ". This is the second way a rendering check passes for the wrong "
              "reason: the picture is not blank, and it is the picture of a "
              "classifier that returned one class for everything.")

    grey = {k: (0.62, 0.62, 0.62) for k in ("helix", "sheet", "coil")}
    g1 = r.render([(cg.cartoon(guide, sides, states, colors=grey), 1.0)],
                  cartoon_camera_for(xyz), W, H)
    from opendocking.workbench.geometry import ribbon as build_ribbon
    g2 = r.render([(build_ribbon(guide, sides, states, colors=grey), 1.0)],
                  cartoon_camera_for(xyz), W, H)
    d = differ(g1, g2)
    if d >= MIN_SEPARATION:
        ok(names["separate.geometry"],
           f"the greyscale cartoon and the greyscale ribbon differ by {d:.2%} of "
           f"the frame against a floor of {MIN_SEPARATION:.0%} -- the two are "
           f"different geometry, not one mesh under two names.")
    else:
        bad(names["separate.geometry"],
            f"the greyscale cartoon and the greyscale ribbon differ by {d:.2%} "
            f"against a floor of {MIN_SEPARATION:.0%}, so the cartoon is drawing "
            f"the ribbon's mesh under its own label")

    if degenerate(states):
        bad(names["classifier"],
            f"every one of the {len(states)} residues carries the same state "
            f"({states[0] if states else 'none'}). A classifier that finds no "
            f"hydrogen bonds returns all-coil, and an all-coil ribbon is a "
            f"perfectly good picture: it renders, it is the right size, and it "
            f"says nothing.")
    else:
        counts = {k: states.count(k) for k in ("helix", "sheet", "coil")}
        ok(names["classifier"],
           f"{len(states)} residues classified {counts}; not all one class, "
           f"which is the precondition for the separability checks above meaning "
           f"anything")

    r.close()


# --------------------------------------------------------------------------
def check_the_totals() -> None:
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    if npass + nfail + nskip == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok("the tally accounts for every result, skips included",
           f"{npass} + {nfail} + {nskip} = {len(RESULTS)}, and this check is the "
           f"{len(RESULTS) + 1}th, so the run reaches the declared "
           f"{EXPECTED_CHECKS} -- the same on a machine with a GL context and on "
           f"one without, because a skip records a result")
    else:
        bad("the tally accounts for every result, skips included",
            f"recorded {len(RESULTS)} results ({npass}+{nfail}+{nskip}) before "
            f"this check, which must be {EXPECTED_CHECKS - 1} for the run to "
            f"reach the declared {EXPECTED_CHECKS}")


def main() -> int:
    print("Are the frames a checked artefact, or a manual one?")
    print()
    check_the_drivers_exist()

    problem: str | None = None
    try:
        import secondary_render

        probe = secondary_render.OffscreenRenderer.open(64, 64)
        if probe is None:
            problem = secondary_render.LAST_PROBLEM or "unknown"
        else:
            probe.close()
    except Exception as exc:  # noqa: BLE001 - any failure means "no answer"
        problem = f"{type(exc).__name__}: {exc}"
    print()
    print(f"-- the offscreen OpenGL context")
    if problem is None:
        print(f"       a standalone GL 3.3 context opened at 64x64, so the "
              f"frame questions below are answerable on this machine")
    else:
        print(f"       no context: {problem}. Every frame check is recorded as a "
              f"skip with that reason and still counts toward the total.")

    check_the_offscreen_drivers_run(problem)
    check_the_frames(problem)
    check_the_totals()

    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")
    print("    the three numbers never sum to a verdict: a pass count on its "
          "own cannot say whether a frame was rendered")
    if nfail:
        print(f"RESULT: FAIL -- {nfail} of {EXPECTED_CHECKS} checks failed")
        return 1
    if npass == 0:
        print("RESULT: DID NOT FINISH -- this run asserted nothing")
        return 2
    print(f"RESULT: OK -- {npass} of {EXPECTED_CHECKS} checks passed, {nskip} "
          f"could not be measured on this machine")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("\nRESULT: DID NOT FINISH -- see the traceback above")
        raise SystemExit(2)
