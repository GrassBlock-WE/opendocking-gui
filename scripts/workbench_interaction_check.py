"""Drive the real workbench with real input events and check what the user sees.

The render smoke test proves geometry reaches the framebuffer. It says nothing
about whether the *controls* work, so this one posts genuine Qt mouse, wheel and
keyboard events at the live widgets and reads the result back.

Three things it checks that nothing else does:

* **Layout**: the geometry of every control in the docked panel, looking for
  zero-size, overlapping, clipped or out-of-panel widgets.
* **Interaction**: left-drag rotates, wheel zooms, Frame all refits, the
  checkboxes actually hide things, the box spins move the wireframe, the pose
  list switches poses and updates both readouts.
* **Behaviour under a real docking run**: that the window keeps painting while
  a search runs, that the Dock button re-enables, and what happens to state
  across two consecutive runs.

Run it on the real desktop platform. It does not force QT_QPA_PLATFORM --
offscreen is the one mode without OpenGL, and forcing it hides exactly the
bugs this is looking for.
"""

from __future__ import annotations

import contextlib
import ast
import gc
import io
import math
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))
# `secondary_render` and `secondary_reference` live beside this file. They are
# imported by name below rather than through a package, because they are
# fixtures and an offscreen renderer for this suite, not product code.
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from opendocking.workbench import (  # noqa: E402
    COLOR_BEST_POSE,
    COLOR_PAIR,
    COLOR_RECEPTOR,
    crashguard,
)
from opendocking.workbench.app import (  # noqa: E402
    POCKET_OPACITY_COMPARE,
    POCKET_OPACITY_PLAIN,
    POSE_GHOST_COLOR,
    REPRESENTATION_KEYS,
    MainWindow,
    _draw_colours_for,
    _walks_wanted,
)
from opendocking.workbench.structure import (  # noqa: E402
    parse_structure,
    secondary_structure,
)

import secondary_reference as ss_ref  # noqa: E402
import secondary_render as ss_render  # noqa: E402

OUT = ROOT / "dist" / "workbench_interaction"
OUT.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, str, str]] = []

#: How many checks this file is supposed to run. The house convention
#: (`pockets_check.py`, `pdbqt_check.py`) is to measure it from a green run and
#: change it by hand, because the noticing is the point.
#:
#: Unlike those two, this file also prints it on the *first* line of output and
#: closes every section with its own subtotal. A pinned number checked only at
#: the end cannot help anyone reading a run that died before the end -- which is
#: exactly what happened here: for two hours this suite reported `exit 3` with
#: a screenful of `[PASS]` and no sign that sections 8 to 11d never executed,
#: and a truncated run is indistinguishable from a finished one unless the
#: transcript says otherwise.
#:
#: 199 measured from the last green run, plus 9 added with section 7c, plus 2
#: added with sections 5b/6 being given their own headers, plus 2 completeness
#: checks, plus 1 for the site cloud stepping aside, plus 3 for a pose file
#: whose columns are wrong, plus 1 for this machine's OpenGL verdict, plus 11
#: for the overlay's visual hierarchy -- across every representation, not just
#: the default -- and the engine's own ceiling. Calibrated from the first run
#: to reach this summary. Skips count toward the total because `skip` records a
#: result too, so this number does not move with the environment -- which is the
#: property that makes it worth pinning.
#:
#: Plus 10 for section 7d, measured the same way: a pose view carries bonds,
#: they are the file's declared tree, they are physical distances inside one
#: atom's valency, every pose of a set agrees, every ghost has them, the status
#: bar says where they came from, and the pose is measurably on screen in three
#: representations. The `stick` skip in 7b became one of those checks, so it
#: added nothing to the total by itself: a skip was already being counted, and
#: a suite that reports "unanswerable" as a result and "231 passed" in its
#: headline has quietly told the reader that everything was fine.
#: Plus 2 for closing the two places where the total used to move with the
#: environment, which is the defect this file's pin exists to catch and which
#: CI found on a headless runner: "the viewport's frame could be read" was
#: registered only when the frame was blank (243 with a framebuffer, 244
#: without), and section 11c's eleven site-selection checks were registered only
#: when the pocket search found more than one site. Both now register in every
#: branch, the first unconditionally, the second as a named skip list, and the
#: list is itself checked against what the block recorded.
#:
#: Plus 1 for a guard that shows the fast-machine skip above cannot hide a
#: genuinely blocked window, and the reason the total is now environment-free.
#:
#: Plus 19 for section 10c, measured the same way: the secondary structure a
#: ribbon is drawn from has to be read off a real receptor and off a synthetic
#: ideal alpha helix, has to agree with the published annotation of 1CRN, has to
#: leave the loop residues alone, and has to change the rendered pixels. The
#: 19 are 4 stating what the classifier answered (the backbone was read, the
#: states are not all one class on either the prepared file or the raw PDB, and
#: the guard reverse-verifies), 5 for the five published 1CRN segments, 2
#: pinning the three residues where the answer departs from that annotation
#: (so the reference cannot be quietly widened to match the code), 1 for the
#: loops, 2 for the ideal helix (its geometry, then its classification), 1
#: saying the offscreen render draws the very mesh the viewport draws, and 4
#: measured in pixels. Each has a skip registered in the same branch it can be
#: skipped in, so a machine with no offscreen context reports the skips rather
#: than a smaller total.
#: Plus 3 for the space-filling mode, measured the same way. Adding a
#: representation gives every per-representation loop one more leg, and the
#: three are the legs that moved: 2 in section 10 (the selector reaching the
#: viewport, and the mode drawing something), and 1 in section 7b's overlay
#: hierarchy sweep. The other per-representation loops in this file are fixed
#: tuples and were not widened: 7d's three representations and 10b's three.
#:
#: Plus 3 more for the change that mode forced, none of them a re-pin of a
#: number that had merely been overtaken:
#:
#:   * 1 in 7b, the control for `POSE_CHROMA_FLOOR`. Section 7b's relative
#:     saturation clause is the one threshold the new mode invalidated, and it
#:     was re-derived rather than widened: it read 2.0 against a mode that
#:     measures 1.9, and it is now 1.0, which is the boundary of the claim it
#:     was expressing ("the pose is at least as chromatic as the ghosts"). The
#:     control exists because a floor that any pose passes is not a floor, and
#:     it measures a half-chroma pose at 0.97x, which the floor rejects.
#:   * 1 in section 10, the space-filling mode against the small-sphere mode as
#:     pictures. A mode that drew the default's frame under a new label would
#:     be the defect this change is about wearing a different hat, so the two
#:     are compared. This one started life as an *area* comparison with the
#:     floor justified by "a continuous surface fills the gaps", and both
#:     halves of that are gone now; see the "276 -> 276" note below.
#:   * 1 in section 11c, the site cloud measured in all six representations.
#:     This one is a check that came out *against* its own hypothesis: the
#:     expectation was that a van der Waals surface would bury a cloud of
#:     points that sits inside the protein, and the measurement says the cloud
#:     covers the same 23 719 px in every mode, because `_draw_pocket` runs
#:     with the depth test off. So what is pinned is the true claim -- the site
#:     stays visible in every representation -- rather than the expected one.
#:
#: **271 -> 276 is five added checks.** Four are `POSE_FRAMING_CHECKS` -- the
#: new section 11e, which asks whether selecting a pose *moves* the camera and
#: whether the move lands where it was aimed. The fifth is in section 11b, on
#: the residue selection that was here already: it asserted the camera arrived
#: *near* the contact centroid, within 1.0 A, and a move that stopped 0.9 A
#: short satisfied it. It now also asks whether the move arrived *exactly*
#: where `focus_residue` computed, which is a different claim and the one that
#: catches a transition ending off-target.
#:
#: Nothing was removed. Two thresholds moved, and both moved because the first
#: number was **unreachable by a correct implementation** rather than because a
#: measurement came out badly:
#:
#:   * that new residue drift floor was an absolute 1e-6 A on a float32
#:     camera, where one ulp at 16.64 A is already 2.0e-06. It failed at
#:     3.58e-06, which is the representation's own resolution and not a
#:     thousandth of an angstrom of product error. It is now 1e-6 *relative*,
#:     about eight ulps.
#:   * the pose-move sampling required every observed frame to be strictly
#:     between the two ends, and the first sample is the frame *before* the
#:     tween's first tick, so it reads the start. It now asserts a strictly
#:     decreasing sequence and a minimum sample count, which is what the check
#:     was for: that the camera moved rather than cut.
#: **276 -> 275 is one check fewer, and one rule reversed.** The three checks
#: that asked what colour a pose wears became two. The two that remain ask a
#: better question than the one that went: the dropped check asserted "with the
#: overlay off the pose goes back to element colours", which was a documented
#: decision and is now **reversed by measurement** -- over the pose's own 23 092
#: pixels its chroma median was 8 against the receptor's 7, which is the
#: receptor's own 60th percentile, so the pose sat *inside* the protein's colour
#: distribution and a person could not point at it in the picture at all. A pose
#: now wears its flat identity colour whether or not the compare overlay is up,
#: and the surviving check is the absolute one ("the pose wears its own colour")
#: plus the one it was really about ("the pose and the ghosts differ in kind").
#: Nothing else was removed, and the four section-11e checks from the previous
#: round are unchanged.
#: **275 -> 276 is one check split into two.** The check that asserted "turning
#: the compare overlay off gives the site cloud its opacity back" asserted a rule
#: that no longer holds and should not: the cloud's presence follows *what the
#: user is looking at*, and with a pose selected and no site chosen it stays
#: away. Under the old rule the path "select a pose, turn the overlay on, turn
#: it off" ended with the cloud back on top of the pose -- drawn with the depth
#: test off, owning nine times the pose's pixels -- which is the defect the
#: selection framing exists to remove. So the one check became two: the cloud
#: stays away when a pose is selected, *and* a site selection brings it back. A
#: rule that can only ever say "off" is not a rule about what to draw, and only
#: the first half would have caught that.
#:
#: **276 -> 276, and no number moved: one check re-derived, and the pin above
#: is not the thing that changed.** Section 10's space-filling comparison was
#: measuring the *difference of two footprint areas* against a 10% floor that
#: its own comment justified with "a continuous surface fills the gaps between
#: separated spheres, so it has to cover materially more". Two measurements
#: killed both halves of that. The direction is backwards on this fixture: with
#: the pose out of the frame the receptor alone reads 5.6% and 4.6% across two
#: runs, and in both the *separated* mode covers more than the CPK surface
#: (590 061 against 558 676, and 579 624 against 554 256), because `spheres`
#: also draws bonds and small spheres plus sticks beat a vdW surface here. And
#: the number is not a property of the two modes at all, because the main
#: window docks unseeded -- section 7b of this file says so -- so the selected
#: pose, and the camera round 1's `_frame_selection` then frames on it, differ
#: every run. Three runs of this same code read 6.8%, 24.2% and 24.9%. A floor
#: that low with a number that moves that far is a coin.
#:
#: So the check now asks a question that is a property of the modes: how many
#: pixels change when the mode does, at the background mask's own tolerance. The
#: pose is hidden for the sweep, because a check that depends on a random dock
#: result depends on a random dock result. The floor is still 10%, and it is
#: not lowered to sit under the 4.6% the old quantity measured. Three green
#: runs read 33.0%, 24.9% and 23.0%; the mutation, which makes `space_filling`
#: draw the small-sphere picture under its own label, reads 0 px and goes red.
#:
#: **276 -> 277, and the one is a wait.** Section 7b read the camera distance
#: while a pose-selection move was still running, and `processEvents()` does not
#: advance a wall-clock `QTimer`, so the read was a moment of a move rather than
#: a camera position: the same pose, out of the same nine fixed-seed poses,
#: measured 23.6 A on one run and 45.4 A on another. The distance is now taken
#: after a clock wait that ends when the camera is at rest, and that the camera
#: got there is itself a check, because "it settled" and "it settled somewhere
#: else" are different answers.
#:
#: The direction that number was compared against is gone as well, and it was
#: the wrong claim rather than a wrong number: the overlay reframes to all nine
#: poses, which on this fixture is 23.6 A -> 31.4 A, *back* rather than in,
#: because row 0 is the worst-energy pose and the most isolated one. What is
#: asserted now is the geometry a user would recognise -- every pose on screen
#: is inside the picture -- which is direction-free and which a camera that did
#: not reframe cannot satisfy. The old threshold also passed on 45.4 A, a
#: distance left over from an unrelated earlier framing.
# Plus 14 for section 12, measured the same way: an uncaught exception in a Qt
# slot ends the process with `0xC0000409` and prints nothing, so the 6 fault
# children are 1 for the same fault with the hook removed, 1 for the print-only
# hook that was refused (it ends the same fault at 0, which is why the exit code
# is the point), 1 for the shipped hook's exit code, and 1 each for the file, the
# line, the exception and the message it names, 1 that the code came back out of
# the product's own `run()` rather than out of a kill, 1 that the failure is
# recorded, and 1 for the chain under a re-raise. Then 3 more: 1 that an
# exception the slot catches itself never reaches the hook, 1 that a clean run
# of the same entry point is untouched, and 1 that `odgui --check --json` gives
# the same verdict and the same exit code with the hook in place as without it
# (two wall-clock fields excluded and named). The last 2 are in-process: the
# `finalise` backstop, and the two branches of `_stop` with the hard exit
# recorded rather than performed.
#: **291 -> 300, and the nine are section 13, the keyboard.**
#:
#: Eight of them are the claims, one is the capture. The eight: every declared
#: key is installed and none is installed twice; a key press gives the same pose
#: a click gives, compared field by field; the steppers beat the table's own
#: column navigation while the table's own Up and Down keep working; the first
#: and the last pose are handled and said out loud; the map is generated from
#: the table, so a key cannot exist undocumented; the map's claim about context
#: matches the measured swallow set; a letter key does not also move the table's
#: own cell through its type-ahead; and the camera keys and the mouse drag are
#: one method. The ninth is that the two captures were written by a platform
#: that can render words.
#:
#: Each claim that can be mutated is mutated and restored inside its own check,
#: which is why the count is nine and not fifteen: a control that is accepted, a
#: mutation that is rejected and a restore that is accepted again are one line of
#: evidence, not three checks.
#:
#: **300 -> 308, and the eight are section 14, one interaction pair at a time.**
#:
#: A residue name is not a pair, and a residue-level table cannot be walked one
#: interaction at a time -- which is the unit a chemist reasons in. The eight:
#:
#: 1. one row per contact, and each row names the two atoms *its own* line
#:    connects, checked against the contact at that row's index rather than
#:    against the row number, because a highlight that pointed at "row 7" would
#:    pass every check that never read a name;
#: 2. a selected row sets the highlight to that pair's two atom indices and the
#:    camera centre to their midpoint;
#: 3. **the pixels**: the highlight's colour counted in a framebuffer grab with
#:    the row selected, and zero in a second grab with only the selection
#:    cleared. A band, not an equality, and a band chosen to contain
#:    `COLOR_PAIR` and nothing else this file draws -- the blue ceiling is what
#:    separates it from CPK-red oxygen, which shares its green;
#: 4. **the mutation for 3**: row still selected, draw switched off by hand, and
#:    the pixels must go to zero and come back. Without this, check 3 could be
#:    counting the protein and still be green, which is the silent way a pixel
#:    check cannot fail;
#: 5. the pair table's own Up and Down move the selection -- the window installs
#:    no `QShortcut` for them and none of the four key-map rows carries a
#:    handler, so the key and the click reach the highlight by one signal;
#: 6. the selection and the highlight do not survive a pose change, and the
#:    table is the new pose's (row count against that pose's contact count);
#: 7. an empty pair table says why it is empty, and a filled one says what its
#:    term column is and why most of it is blank;
#: 8. the key map lists those two keys under their own heading, and every group
#:    in the order has one.
#:
#: **324 -> 325, and the one is the dock failure's voice; the rest is section 14
#: being repaired rather than added to.**
#:
#: Section 14's two pixel checks were taking their samples in the shared `kw`,
#: whose camera sections 7-13 had already orbited, zoomed and reframed fourteen
#: sections' worth. The count came back 0 and the check went red, and the red was
#: a measurement of the *history* rather than of the draw: in a window nobody
#: had touched the same line reads 95 px at a 30 A box's own framing, and
#: pressing `F` takes it back to 0. A measurement has to happen in a state you
#: are holding still, so both grabs and the mutation now happen in a window built
#: for the purpose, from one fixed search rather than from a pose file -- with a
#: receptor, because `find_contacts` needs something to be in contact with.
#: The gate's own window reads **10 px**, not 95, and both numbers are true
#: rather than one of them stale: its box is 22 A, which projects a shorter line,
#: and the threshold is 8. **That margin is two pixels**, and a reader should be
#: told it is thin rather than surprised by a red on another machine. Widening the
#: box would buy pixels and cost the property the window exists for, which is a
#: pose framing nothing has touched.
#: state.
#:
#: Three site-level changes, and only one executed check more. The pixel check
#: gained a branch: when the pose file produces no interaction pair there is no
#: line to count, and that is a **skip with the reason** rather than a red, since
#: it is a question the platform cannot answer rather than a property it fails.
#: The mutation gained the same shape, for the same reason. So the pixel block
#: went from 4 sites/2 executed to 6/2 -- and the +2 in the count below is sites,
#: not checks, which is the distinction the file has now had to make twice.
#:
#: What makes the repair a check rather than a re-colouring is that **the
#: mutation runs in the same window and has to be the red case**: the row stays
#: selected, nothing re-selects, no camera moves, and the only change is the
#: draw -- so a count that survived it was never counting the highlight. Run in
#: the old window it would have read 0 on both sides and "passed" the red case.
#: Both pairs of frames are also compared **as bytes**, not only by count: two
#: frames that were the same picture report 0 against 0, and a count-only
#: comparison calls that a result. Byte equality is what says the switch landed.
#:
#: The one added check is the dock failure's voice in section 16: a search that
#: does not run used to say `docking failed: <engine text>` and stop, which names
#: what happened and gives no next action. Three cases -- a build mismatch, a box
#: the ligand does not fit in, and an unclassifiable failure that must admit it
#: has no fix to name -- because a set of cases that only tested the easy ones
#: would let a guess through.
#:
#: Section 15 is the first thing in this file whose whole point is that it
#: leaves, so its checks are mostly about a *file*, and the reader for every one
#: of them is `json.load` -- not `export.read_export`, and not the writer's own
#: objects. A reader that shares code with the writer cannot disagree with it,
#: and a round trip proved that way proves nothing. The nine:
#:
#: 1. the exported affinity, rmsd and intermolecular energy of every pose are
#:    equal to the `DockingResult`'s own **to the bit**, and the coordinates are
#:    checked twice against the engine's array -- within the three decimals the
#:    pose PDBQT carries, and exactly equal to `round(engine, 3)` as float32, so
#:    the `coords_source` claim is a fact about the file rather than an
#:    assertion in it. A third comparison, against `win._pose_ghosts`, is the
#:    *same array* the export read, so it is reported and labelled as the
#:    same-object check it is rather than counted as round-trip evidence;
#: 2. the provenance in the file -- seed, exhaustiveness, scoring, box centre,
#:    box size, engine version, backends -- against the window's own spins, and
#:    the receptor's sha256 recomputed by this gate from the same bytes;
#: 3. the verdict, contract by contract: same `trust`, same states, same `because`
#:    strings as the live `PoseVerdict`, and `derived` saying "NOT recomputed".
#:    The discriminator is `inside the box`, which reads `unmeasured` for a
#:    verdict built without the breakdown's penalty -- so a re-derivation on the
#:    way out would disagree with the panel here, and the file agrees with the
#:    panel;
#: 4. **the discrimination test**: one file has to hold a measured `0.0` and an
#:    unmeasured `null` at the same time, or telling them apart proves nothing;
#:    every `absent` entry carries a reason, `absent_count` matches the list, and
#:    every unmeasured contract appears in the top-level index;
#: 5. a pose-file export, in its own freshly constructed window, naming all
#:    three states -- `measured` (the affinity, sourced to the file),
#:    `not_run` (the intermolecular part) and `unmeasured` (the terms and all
#:    four contracts) -- with the pose file's own sha256, hashed here;
#: 6. the residue and pair rows in the file are the rows the table is showing,
#:    for the pose on screen. Two `find_contacts` calls at two different times,
#:    which is the claim; comparing the export against itself would not be;
#: 7. an empty window refuses, says why, and puts no new file on disk;
#: 8. the `E` key and the File menu item reach `_export_run`, measured by what
#:    each one *writes*: the seed is put somewhere distinctive, the key is
#:    pressed as a key, the seed is moved, the menu item is triggered, and the
#:    two files have to name their own seeds. Neither route needed a Python-side
#:    call, so a dead shortcut or an unconnected action writes nothing. Two
#:    earlier versions of this check were wrong and both are written down in the
#:    code: `kw._key_export is kw._export_run` is False because the key handler
#:    is a wrapper, and wrapping `kw._export_run` with a counter also reads
#:    False for the menu item because a `QAction` keeps the bound method it was
#:    connected to, so an attribute swapped in afterwards is never called. The
#:    key's note is also required to say where the file goes and what the empty
#:    case does;
#: 9. the export's own wall clock, median of three, under 2000 ms -- a pause on
#:    the GUI thread rather than a freeze.
#:
#: The tenth is not new behaviour, and it is here because it was **wrong**. The
#: section-14 check first asserted that the residue table keeps its selected row
#: across a pose change, and it went red: the selection does not survive, because
#: `_refresh_contacts` empties the table with `setRowCount(0)` before refilling
#: it, and that is the one row-count step Qt answers by dropping the selection
#: (7->9 and 9->4 keep row 0, 9->0 drops it -- measured on the widget itself as
#: well as here). The premise under the paragraph in `workbench/contacts.py` was
#: therefore false, and both it and the matching comment in `app.py` were
#: rewritten. The check now pins the half that is load-bearing in both worlds --
#: **the residue path touches no highlight state at all**, so there is nothing on
#: screen for a row to be wrong about -- and *reports* the selection rather than
#: asserting it, because whether it survives is Qt's and the zeroing's business,
#: and a Qt that preserved it would be a correct implementation of the same code.
#:
#: The number moved by ten and not by nine-plus-one, and the reason is worth
#: writing down: 316 `check`/`pixel_check`/`skip` **call sites** were counted
#: statically and 318 checks **ran**. The static count is not the number -- some
#: sites are inside loops, and a site that runs twice is a site the file cannot
#: audit by counting. The pin is the executed number, and this is the second time
#: the two have had to be told apart.
#:
#: The two pixel checks are guarded by `PIXELS_OK` and are **skips, not
#: failures**, in the other branch -- the same rule every other pixel check in
#: this file follows. Both branches produce one result, so the total does not
#: move with the machine.
#:
#: **328 -> 331, and the three are section 7d's serial-order fixture.** The
#: reader was reading the `ATOM` serial column and then ignoring it: it appended
#: in file order, and for a pose file the writer emits the rigid `ROOT` cluster
#: before the flexible `BRANCH` clusters, so the serials of `poses.pdbqt` run
#: 5, 6, 7, 8, 9, 10, 4, 11, 12, 1, 2, 3. Atom i was therefore not the
#: ligand's atom i. The three checks are: that the fixture really is a
#: permutation and not the identity (a `write_branch_tree: false` round trip
#: writes 1..N in order and cannot see this defect at all), that the reader
#: honours the serial column, and that the index remap keeps the declared bonds
#: pointing at the atoms they declare. Each was mutation-tested: dropping the
#: remap, and dropping the sort, both go red.
#: **331 -> 333, and the two are section 15's receptor-warning pair.** The
#: engine counts
#: receptor atoms whose PDBQT type it cannot classify and says so in
#: `summary()`; the window shows the run's summary; the exported file said
#: nothing, and the file is what somebody takes away. So section 15 now asks
#: the question in two halves, because one half alone proves nothing: a file
#: that names a defect on a run the engine read perfectly is a warning system
#: that cannot stay quiet, and a reader who has been trained to skip the line
#: will skip the real one. The first check exports a run over a receptor with
#: five atoms given a type AutoDock never defined and requires the file to name
#: the count and its consequence; the second requires the gate's own clean run
#: to name nothing at all.
#:
#: Both are careful about which build they are talking to, and both say which
#: way they went in the transcript. The count reached `DockingResult` after the
#: `_dockpy.pyd` in a checkout was built, so on a stale binary the property
#: raises `AttributeError` and the honest answer is the export's `absent` state
#: -- never a zero, which would be a claim about a receptor the file never read.
#: A conditional assertion is only honest if it is visible, so each of these
#: prints the build's answer in full, and each additionally asserts through a
#: stand-in result the one direction no build can refuse: five atoms in becomes
#: a measured 5.0 with the sentence attached.
#:
#: **338 -> 339**, and the one is named. Section 14 gained the area-guard and
#: status-line check, which is +1. Section 1's "no control overflows the panel"
#: became "every control in the panel is reachable" -- a question about fit
#: replaced by a question about reachability, since a scrolling dock answers
#: the old one wrongly -- and that is a rename, not an addition, so it does not
#: count. The first run after the change went red at 339 of 338, which is the
#: behaviour this constant exists for: it is the only thing standing between a
#: gate and a check that quietly stops running.
#:
#: **339 -> 343, and the four are named, and all four are in section 14.**
#:
#: * **M9, the refusal branch is reachable by a real drag.** It had been called
#:   "latent" for a round, which is a way of saying nobody had run it. It runs
#:   now, by dragging, and the check carries a self-check that the drag moved
#:   the angle it claims to sweep -- the defect that made a scratch yaw sweep
#:   print 19 identical rows.
#: * **M10, the length floor refuses the marker the area floor waves through.**
#:   Without this the new constant is decoration: if the old guard also refused
#:   the case, either guard would do and nothing was added.
#: * **M11, a marker whose declared rectangle covering more than 1% of the frame
#:   is still drawn and the status bar says how much of the frame it is.** The
#:   2 A wall, which a reader cannot tell from a fault. **Rewritten, not
#:   re-pinned**: the bar now names *which* share it is reporting (the declared
#:   rectangle's projected area, 14.14%, not the pixels the rod paints, 8.41%),
#:   because both were once called "the share of the frame" and they differ by
#:   up to 2x. Same check, different claim, and the probe follows the new
#:   wording rather than a keyword.
#: * **M12, the pad floor is needed, sufficient and one-sided.** The floor moved
#:   0.75 -> 1.0 px because 0.75 left a stray pixel on a correct draw, and the
#:   third claim is the one that could have broken.
#:
#: **343 -> 347, and the four are named, and all four are in the new section
#: 14b.** They re-measure the two marker floors against the live framebuffer
#: rather than against a transcription of a past run, which is what makes the
#: two constants in `workbench/app.py` reproducible from the tree. They are
#: `pixel_check` and not `check`, so the count moves by four and the call-site
#: census in `check_scripts_declare.py` does not move at all -- that census's
#: declared column for this file lives in that file's generated snapshot and is
#: already behind this file, and widening a debt whose remedy is a `--pin` this
#: section may not run is not this section's to do.
#:
#: Section 14's M10 detail text was also corrected. It asserted that the lit set
#: is "two or more columns above one pixel of length, at every phase", measured
#: over 96 cameras: that instrument counted 1-px bins from the mark's own start
#: point, so it returned 1 for anything at or below one pixel by construction and
#: the guarantee could not have failed. Re-measured in framebuffer coordinates,
#: two columns at every phase needs 1.20 px at both device pixel ratios. The
#: first corrected sweep aimed by bisecting the off-axis angle over [0, 45 deg]
#: and could not resolve a short length on a wide marker -- every requested
#: length from 0.30 to 2.00 px measured 0.88-1.11 px at an 11 A distance, and
#: nothing said so. The instrument was replaced by a bracketed secant and the
#: wide-width rows re-measured; that is the version in `app.py`.
#:
#: The M8 area check was **rewritten, not added**: it now asks the combined
#: predicate against both floors, so it is the same one check and does not
#: count. Its name changed with it.
#:
#: 347 -> 351, and the number came from the run rather than from arithmetic:
#: a run with the four M13/11b checks in it reported "349 ran before these two
#: and 347 are expected", i.e. 351 in all, and every other check in that run
#: passed. The four are the framed-subject share: the empty state before any
#: pair is framed, the residue route, all 27 pair rows on this window, and the
#: unit in the wording.
#:
#: 351 -> 354, and the number came from the run rather than from arithmetic. The
#: three are the narrow-panel checks in section 1: whether the panel is shown
#: whole or the notice is up, whether the legend words are on screen, and
#: whether the notice is inside the strip it warns about. A dpr 2.5 run with
#: them in it reported "352 ran before these two and 351 are expected", i.e.
#: 354 in all, with every other check passing -- and the same three at dpr 1.25
#: reported the same count, which is the point of them: a check that only ran at
#: one ratio could not have failed at the other.
#:
#: 354 -> 355, again from the run: the one is the reverse verification of those
#: three, which is a fourth call site and not a fifth check, because it reports
#: one verdict over four states. Its number is the only way to tell a predicate
#: that can go red from one that merely has not.
#: **355 -> 357, and the number came from the run, not from arithmetic.**
#: Section 1 gained two, both on the narrow-panel work and both properties
#: rather than observations:
#:
#:   1. "the user can drag the control panel to a width where every control is
#:      on screen" -- the reachability question the 3-D view's old 640 px floor
#:      made unanswerable at dpr 2.5, asked as the drag itself. It goes red
#:      when the floor is put back, which the reverse verification below does.
#:   2. "the notice never says the controls need zero pixels" -- the notice
#:      stated its deficit as `panel.minimumSizeHint() - viewport` (491) while
#:      the panel had actually been given 495, so with the bar at 1 px the
#:      notice was up and said "0 px more". A 355-check suite missed it because
#:      every check asked whether the notice was *shown* and none asked what it
#:      said. That one was found by looking at a frame, not by a number.
#:
#: **The reverse verification grew by two cases and counts zero.** The two it
#: gained -- a floor raised to the window's own width, and a notice whose own
#: figure contradicts the bar -- are more things the shipped predicate must be
#: able to go red on, inside a function that reports one verdict over all of
#: them. It is also where the residue-framing check stopped being
#: `distance <= before_dist`: that compared two different subjects' minimum
#: distances and only held because the old floor made the frame wide enough
#: for the pose to dominate (aspect 1.333: residue 22.6 A; aspect 0.500:
#: residue 48.5 A, same window and same row).
#:
#: **This round: 366 -> 367, and the 367 is read off a run rather than
#: added up.** `_check_calls_are_wired` is the one new check, and the run
#: that had it printed `RUN INCOMPLETE: 367 of 366 checks ran` while every
#: other check in that run was green -- the completeness check below is the
#: only thing that went red, and it went red on the number, not on the
#: workbench. So the pin moved to the figure the run reported. Counting
#: `before_total + this one` by hand would have produced the same 367 and
#: proved nothing, which is the reason the number is measured here rather
#: than derived: the check it pins exists to catch a count that is not what
#: the file actually runs, and arithmetic would have assumed the very thing
#: it is there to check.
EXPECTED_CHECKS = 367

#: Which pair rows the findability assertion is measured on, in section 8b.
#:
#: **Three consecutive rows from the top, and not one.** `marker_findable` used
#: to be green on `pair_table.setCurrentCell(0, 0)` and on nothing else, and
#: that single row is the whole reason `_pair_quad` was able to offset the rod
#: along its own axis and ship.
#:
#: **Why row 0 passed, corrected.** It was recorded here as "the one row of ten
#: whose segment lies 45 degrees to the screen axes", and that is wrong: at
#: framing `F` row 0's segment runs at 151.75 deg, which is 28.25 deg off
#: horizontal, and the pre-fix offset drew it at **0.890** of its declared
#: width -- thin, and visible as such, but above the 0.5 floor. The 45 degree
#: row of the first ten is **row 4**, at -134.80 deg, and it is the only one of
#: those ten the pre-fix offset drew correctly (1.061). So the one-row gate was
#: green not because the row it measured was right but because `marker_findable`
#: only asks for half the declared width, and row 0 happened to clear that. The
#: sweep starts at 0 and so does not contain row 4 either; widening to three
#: rows is what caught the defect, not the choice of starting row.
#:
#: Measured on the product as shipped, on this fixture at the framing `F`
#: produces: 5.74, 2.50 and 2.13 px across against 6.45, 6.40 and 6.82 declared,
#: fills of 0.94, 0.41 and 0.33. After the fix all 27 rows read 1.03 to 1.09 of
#: declared.
#:
#: **The whole table, not a sample of it.** Ten of the twenty-seven rows were
#: measured in round 1 and the base rate was reported from those ten, as "five
#: of ten not findable". Over all 27 it is **11 of 27 not findable** (40.7%,
#: Wilson 95% CI 24.5-59.3%), and 11 of 42 on the second fixture, 25 of 69
#: pooled (36.2%, 25.9-48.0%). The ten-row interval contained the true rate, so
#: the sample was not wrong -- but its framing was, and in the direction that
#: matters: the defect is **graded, not binary**, and the drawn-to-declared
#: ratio is a smooth function of the segment's screen angle. 23 of 27 rows were
#: drawn measurably wrong at some degree and only 4 were right. See
#: `marker_findable` for the model and the per-row table.
#:
#: Consecutive from the top on purpose. Any other choice would be three rows
#: picked because they pass, and the point of the widening is that the rows are
#: not chosen by their result.
PAIR_ROW_SWEEP = (0, 1, 2)

#: **How many rows the sweep is required to produce, counted separately from
#: which rows it walks.**
#:
#: The first version of the "all three had to be measurable at all" clause read
#: `len(_row_marks) == len(PAIR_ROW_SWEEP)`, which is circular: the sweep is
#: what it is being compared against, so shortening the sweep shortens the
#: requirement with it. Measured, not argued -- `PAIR_ROW_SWEEP` set to `(0, 1)`
#: and the suite re-run: the detail string duly reported "**All 2** had to come
#: back findable", the check came back **PASS**, and 338 of 338 checks ran. A
#: guard that moves when the thing it guards moves is not a guard, and the
#: failure it was written to catch -- a fixture that quietly produces fewer rows
#: -- is precisely the one that would slip through it.
#:
#: So the count lives here, on its own, and the two cannot be edited together
#: without the suite going red on the mismatch. A reader changing
#: `PAIR_ROW_SWEEP` to measure more rows has to change this too, deliberately.
EXPECTED_PAIR_ROWS = 3

#: Every section this file is supposed to reach, in file order. A section that
#: is entered always prints a header, and every header is closed by the next
#: one's subtotal, so a run that stops mid-section is visible in the transcript
#: without anyone having to trust the number above.
EXPECTED_SECTIONS = (
    "1. layout of the control panel",
    "1b. does the viewer understand the atom types it is given?",
    "1c. does the 3-D view say when its width is spent?",
    "2. camera interaction",
    "3. is there a pan control?",
    "4. visibility checkboxes",
    "5. search-box spins",
    "5a. the search effort follows the box",
    "5b. the File menu can reach every role",
    "6. pose browser",
    "7. a real docking run from the GUI",
    "7b. the pose table, from a run whose result is still in memory",
    "7c. a pose set that breaks the table's assumptions",
    "7d. a pose is a molecule you can draw",
    "8. a second docking run",
    "9. a receptor at negative coordinates",
    "10. display representations",
    "10b. a ribbon on a real protein",
    "10c. the ribbon knows what shape it is drawing",
    "11. pose/receptor interactions",
    "11c. pocket search drives the box",
    "11e. selecting a pose frames it, and the move lands on target",
    "11d. the pocket search does not block the window",
    "12. an unhandled exception in a slot is reported, and exits non-zero",
    "13. the window without a mouse",
    "14. one interaction pair at a time",
    "14b. the two marker floors, measured again on this machine",
    "15. taking the run away",
    "16. docking from inside the window",
    "summary",
)

# Section bookkeeping. `_section_open` is the section currently being filled in
# and `_section_first` the index into `results` where it started.
_section_open: str | None = None
_section_first = 0
sections_entered: list[str] = []
#: Per-section skip counts, so the summary can name the sections the machine
#: answered rather than the code.
section_skips: dict[str, int] = {}
section_totals: dict[str, tuple[int, int, int, int]] = {}


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def skip(name: str, reason: str) -> bool:
    """Record a check that could not be run here, and say why.

    A skipped check and a failed check are different claims. "The viewport
    draws nothing" and "this environment cannot read the framebuffer" look
    identical in the output if the second one is reported as the first, and
    only one of them is ever true.
    """
    results.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  — {reason}")
    return False


# Whether `grabFramebuffer` returns anything at all. Probed once, after the
# window is up; see `probe_framebuffer`.
PIXELS_OK = True


def pixel_check(name: str, ok: bool, detail: str = "") -> bool:
    """A check whose truth lives in the rendered pixels."""
    if not PIXELS_OK:
        return skip(name, "the framebuffer read back blank in this environment")
    return check(name, ok, detail)


def _verify_pixel_guard():
    """Reverse-verify the guard the pixel checks depend on.

    A skip is only honest if it is reached *only* when the environment
    genuinely cannot answer, so all three states are checked: a blank
    framebuffer must skip, a good framebuffer with a good render must pass, and
    a good framebuffer with a genuinely empty render must fail.

    Without this, a guard that skipped unconditionally would look exactly like
    a working one. That is precisely the bug that failed the CI workbench job
    once already: one pixel check used `check` instead of `pixel_check` and
    reported a headless machine's blank framebuffer as a broken viewport, while
    its fourteen siblings skipped correctly.

    Its own function because `PIXELS_OK` is a module global that `main` also
    assigns to, and a `global` statement after that assignment is a
    SyntaxError.
    """
    global PIXELS_OK, results
    saved_flags, saved_results = PIXELS_OK, results
    # Swap the list object, never clear it. `results.clear()` looked harmless
    # and silently threw away every check recorded so far -- the summary went
    # from 116 checks to 110 with no failure anywhere, which is exactly the
    # kind of quiet loss a check suite must not have. Redirecting stdout keeps
    # the three probe lines out of the report, where they would read as three
    # unexplained results of their own.
    results = []
    seen = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for label, flag, ok in (
                ("blank framebuffer", False, False),
                ("good render", True, True),
                ("empty render", True, False),
            ):
                PIXELS_OK = flag
                pixel_check("guard probe", ok, label)
                seen[label] = results[-1][0]
    finally:
        PIXELS_OK, results = saved_flags, saved_results
    want = ("SKIP", "PASS", "FAIL")
    got = tuple(seen.values())
    return (
        "the pixel-check guard skips only when the framebuffer cannot answer",
        got == want,
        f"blank framebuffer -> {got[0]}, good render -> {got[1]}, "
        f"empty render -> {got[2]} (want {'/'.join(want)})",
    )


def _own_tuple_returns(node):
    """The `return (a, b, ...)`s that belong to this function, not a nested one.

    `ast.walk` over a function also visits the functions defined inside it, and
    a spread's element types read off the wrong `return` is a wrong answer that
    looks exactly like a right one.
    """
    found, stack = [], list(node.body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(cur, ast.Return) and isinstance(cur.value, ast.Tuple):
            found.append(cur.value)
        stack.extend(ast.iter_child_nodes(cur))
    return found


def _check_calls_are_wired() -> tuple[
    list[tuple[int, str]], list[tuple[int, str]], str
]:
    """Does every check in this file hand `check` a string and a bool, in the
    right two slots?

    `check` prints `PASS` when its `ok` argument is truthy, and a `str` is never
    falsy. A call that hands it a string therefore hands it a verdict that
    cannot go red, and a check that cannot fail is not a check. This was not
    hypothetical. The half-width check in section 1c got a `(bool, str)` back
    from its helper and was written `check(*helper())`, so the bool landed in
    `name` and the 354-character explanation landed in `ok`. A failing verdict
    was recorded as `[PASS] False` -- the bool printed as the check's own name,
    the detail empty -- and no input could ever have reported a failure.

    **Why the sweep could not see it, and why widening it is not the answer.**
    `check_verdict_bool.py` reads the element at the *verdict parameter's*
    index, and that is the right element to read: at that index the argument
    really is a bool, because the bool was written one slot to the left and the
    string was written over the slot it should have filled. Nothing about that
    call is a str-in-the-verdict-slot call. The mistake is one position off, and
    the only thing that carries it is the shape of the call itself -- which is
    what this reads.

    **The rule, measured over this file and not over that one call.** With the
    defect in place: 310 `check(...)` calls and 22 `pixel_check(...)` calls,
    both taking `(name, ok, detail)`, of which 5 pass a spread. A spread is
    only correct when its *first* element is a string, and the five are not the
    same case: four spread a three-element `(str, bool, str)` and are right,
    and the fifth spreads a two-element `(bool, str)` and is the bug. So two
    rules, and the second is what catches the shape written by hand:

    1. an element that resolves to a `bool` landing in the `name` slot;
    2. an element that resolves to a `str` landing in the verdict slot.

    (This file now adds one call of its own, written out as arguments rather
    than as a spread, so a count taken after this line is one higher in the
    first of those numbers and unchanged in the other two.)

    **What it will not do.** Both rules fire only on a type this can *prove*,
    from a literal, from a return annotation that names the elements, or from
    the tuple a helper literally returns. A spread whose element types are
    written down nowhere is counted and named as unread in the detail below,
    and is *not* counted as sound: a checker that guessed here would be the
    same class of bug it is looking for, and the number of sites it could not
    read is printed so its reach stays visible.
    """
    def kind(node) -> str:
        """What this expression certainly evaluates to: str, bool, or unknown."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            return "str" if isinstance(node.value, str) else "other"
        if isinstance(node, ast.JoinedStr):
            return "str"
        # A comparison, an `and`/`or`, and a `not` each yield a bool.
        if isinstance(node, (ast.Compare, ast.BoolOp)) or (
                isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
            return "bool"
        return "unknown"

    def annotated(fn):
        """Element types from a `-> tuple[str, bool, ...]` annotation."""
        ret = fn.returns
        if ret is None or not isinstance(ret, ast.Subscript):
            return None
        if getattr(ret.value, "id", None) != "tuple":
            return None
        if not isinstance(ret.slice, ast.Tuple):
            return None
        out = []
        for el in ret.slice.elts:
            nm = el.id if isinstance(el, ast.Name) else None
            out.append(nm if nm in ("str", "bool") else "unknown")
        return out

    def spread_kinds(value):
        """The element types of `*value`, or None where the code does not say."""
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            fn = defs.get(value.func.id)
        elif isinstance(value, ast.Name):
            fn = defs.get(value.id)
        else:
            return None
        if fn is None:
            return None
        # An annotation that names the elements beats reading the body: the
        # promise is the thing under test, and `-> tuple` on its own -- which
        # is all two of the three verifiers give -- says nothing at all.
        named = annotated(fn)
        if named is not None:
            return named
        # Otherwise the tuple it literally returns, and only when every one of
        # its own returns agrees: a function returning two different shapes has
        # no single shape to read.
        own = _own_tuple_returns(fn)
        if not own:
            return None
        shapes = [[kind(el) for el in tup.elts] for tup in own]
        return shapes[0] if all(s == shapes[0] for s in shapes) else None

    me = Path(__file__)
    tree = ast.parse(me.read_text(encoding="utf-8"), filename=str(me))
    defs = {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    seen = spreads = 0
    wrong: list[tuple[int, str]] = []
    unread: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) \
                or node.func.id not in ("check", "pixel_check"):
            continue
        seen += 1
        if node.keywords:
            # Written with keywords, so its arguments are somewhere this does
            # not model. Named, rather than passed over in silence.
            unread.append((node.lineno, "written with keyword arguments"))
            continue
        slots: dict[int, tuple[str, str]] = {}
        index = 0
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                spreads += 1
                kinds = spread_kinds(arg.value)
                if kinds is None:
                    unread.append((node.lineno, "a spread whose element types are "
                                               "written down nowhere"))
                    break
                for offset, k in enumerate(kinds):
                    slots[index + offset] = (k, "element %d of the spread" % offset)
                index += len(kinds)
                continue
            slots[index] = (kind(arg), ast.unparse(arg)[:44])
            index += 1
        for target, wanted, where in ((0, "str", "name"), (1, "bool", "verdict")):
            got = slots.get(target)
            if got is None or got[0] in (wanted, "unknown", "other"):
                continue
            wrong.append((node.lineno,
                          "%s is a %s and lands in the %s slot, where a %s is "
                          "required" % (got[1], got[0], where, wanted)))

    # **What it returns is what it found, and the verdict is left to the call
    # site.** The caller asks `not miswired`, so the argument that lands in
    # the verdict slot is a boolean expression rather than a name this gate
    # would have to trace through a tuple to type -- which is also how the
    # other two tuple helpers are read here (`not _floored_ok`).
    return wrong, unread, (
        f"{seen} call site(s) read, {spreads} of them a spread and {seen - spreads} "
        f"written out as arguments; a name slot wants a str and a verdict slot "
        f"wants a bool, and a spread is only right when its first element is a "
        f"str. Miswired: "
        + ("; ".join("line %d: %s" % pair for pair in wrong) if wrong else "none")
        + f". {len(unread)} site(s) could not be read, and are not counted as "
          f"sound: "
        + ("; ".join("line %d: %s" % pair for pair in unread) if unread else "none")
    )


def _verify_degenerate_guard(trace, atoms):
    """Reverse-verify the guard the secondary-structure checks depend on.

    A guard that cannot fail is not a guard. This swaps the classifier for one
    that returns a single constant for every residue -- the exact bug a naive
    implementation of this feature always has, because "no hydrogen bonds
    found" and "coil" are the same answer -- and shows the guard reporting it.

    Two directions, because either one alone is unfalsifiable:

    * with the real classifier the guard must PASS, or it is guarding nothing;
    * with a constant-returning classifier it must FAIL, or it is vacuous.

    The mutation is applied to the module attribute this suite calls through,
    and put back in a `finally`: a guard that leaves the product patched after
    a failed run would make the rest of the file's own results meaningless.
    """
    import opendocking.workbench.structure as structure_module

    real = structure_module.secondary_structure
    try:
        verdicts = {}
        for label, impl in (
            ("real", real),
            ("constant coil", lambda *a, **k: ["coil"] * len(a[0])),
        ):
            structure_module.secondary_structure = impl
            verdicts[label] = "PASS" if not ss_render.degenerate(impl(trace, atoms)) else "FAIL"
    finally:
        structure_module.secondary_structure = real

    # And back again, so the second reading is not just the first one cached.
    restored = "PASS" if not ss_render.degenerate(real(trace, atoms)) else "FAIL"
    want = ("PASS", "FAIL", "PASS")
    got = (verdicts["real"], verdicts["constant coil"], restored)
    return (
        "a classifier that returns one state for every residue fails the "
        "non-degenerate guard",
        got == want,
        f"real -> {got[0]}, constant 'coil' -> {got[1]}, restored -> {got[2]} "
        f"(want {'/'.join(want)})",
    )


def search_overlap_verdict(was_running: bool, still_running: bool) -> str:
    """What a run that observed `was_running` then `still_running` can claim.

    Three outcomes, and the middle one is the whole point of this function.

    * **FAIL** when the search was *not* running even at the start. That is a
      product defect -- `find_pockets()` is supposed to return while the work is
      outstanding, and the check before this one already asserts it. A search
      that was already finished here was run inline.
    * **SKIP** when it was running at the start and had finished by the end. The
      search was in flight, and the event loop did turn, but the two never
      overlapped *as observed*, so this run cannot witness a window blocked by a
      running search. That is the environment being unable to answer, which is a
      different claim from the answer being "no", and the suite already has a
      word for it.
    * **PASS** only when the search was still running after the loop turned, so
      the turns measured were turns made while it was in flight.

    This existed as a bare `was_running and still_running`, which is a FAIL in
    all three cases. On a fast runner the search finished inside the second of
    event pumping and the check reported a defect that does not exist -- and
    worse, it was *more* reliable the slower the machine, so the slower the
    runner the more confident a check about the window became. A check whose
    confidence rises as the machine gets worse is measuring the machine.

    Being a pure function is what makes it testable: the skip branch is
    reachable only when `still_running` is False, so a run with a *running*
    search can never take it, and a blocked loop during a running search is
    caught by the liveness threshold instead. `_verify_blocked_window_guard`
    shows all of that.
    """
    if not was_running:
        return "FAIL"
    if not still_running:
        return "SKIP"
    return "PASS"


def _verify_blocked_window_guard() -> tuple:
    """Reverse-verify that a genuinely blocked window still turns this red.

    The skip added by :func:`search_overlap_verdict` is only safe if it cannot
    hide a real block. Two things have to hold, and both are checked here rather
    than argued in a comment:

    1. the skip is reachable **only** when the search was no longer running, so
       a run that observed a live search is never excused;
    2. a live search with a held event loop still fails -- the drain count is
       what catches it, and `_verify_loop_turns_guard` already shows the drain
       count separates a loop that cannot turn from one that can.

    Every combination is listed, including the two that must never be SKIP, so
    a future edit that widens the skip turns this red instead of quietly making
    the section green on a broken window.
    """
    # A held loop delivers 0 of 8 queued events and a free one delivers all 8.
    # These are the two readings `_verify_loop_turns_guard` takes for real; the
    # table below is what the section does with them.
    held = 0
    free = 8
    want = {
        # (search running at start, at end, held loop?) -> verdict
        (True, True, False): "PASS",
        (True, True, True): "FAIL",    # a real block, with the search live
        (True, False, False): "SKIP",  # the search beat the loop: cannot answer
        (True, False, True): "SKIP",
        (False, True, False): "FAIL",  # never ran: the defect
        (False, False, False): "FAIL",
    }
    got = {}
    for (was, still, is_held), expect in want.items():
        delivered = held if is_held else free
        overlap = search_overlap_verdict(was, still)
        # The section is red if *either* the overlap claim or the drain count is
        # red, because both are claims about the same window. The liveness half
        # is now "did the queued events come out", which is an exact count
        # rather than a comparison against a rate.
        alive = "PASS" if delivered == free else "FAIL"
        got[(was, still, is_held)] = (
            "RED" if "FAIL" in (overlap, alive) else overlap
        )
    # Red whenever the overlap claim is red, *or* whenever the loop was held --
    # a held loop means the event loop was blocked, and that is worth reporting
    # whether or not a search happened to still be running. SKIP survives only
    # when the search beat the loop *and* the loop was free.
    expect_red = {k for k, v in want.items() if v == "FAIL" or k[2]}
    ok = all(got[k] == ("RED" if k in expect_red else v) for k, v in want.items())
    return (
        "a window blocked while a search is running is still reported, and the "
        "fast-machine skip cannot hide one",
        ok,
        "; ".join(
            f"search {was}->{still}, loop {'held' if h else 'free'} -> {got[(was, still, h)]}"
            for (was, still, h) in want
        )
        + " (want RED whenever the search never ran or the loop was held; SKIP "
        "only when the search finished before the loop turned and the loop was "
        f"free; PASS otherwise; got {'ok' if ok else 'MISMATCH'})",
    )


#: The checks that section 11c only runs when the pocket search found more than
#: one site for the fixture, which is the same shape of bug as the framebuffer
#: check that used to live only in its failure branch: a whole block of checks
#: that is *invisible* on a machine whose search found a single site, so the
#: total moves with the environment and the pin can no longer tell anyone
#: whether a check was removed or merely never ran.
#:
#: Listed here so the `rows <= 1` branch can register a skip for each one, and
#: verified against what the block actually recorded, so renaming a check inside
#: the block without updating this list turns a check red on a machine with more
#: than one site rather than silently shrinking the total on a machine with one.
SITE_SELECTION_CHECKS: tuple[str, ...] = (
    "selecting a row moves the box",
    "and the spin boxes follow it",
    "and the camera goes to the site it just selected",
    "the camera is far enough out to see the box, not inside it",
    "the status bar names the site and its residues",
    "the selected site is actually drawn",
    "clearing the selection clears the site volume, rather than leaving one "
    "site's cloud inside another's box",
    "reselecting puts it back",
    "the site cloud stays visible in every representation, the space-filling "
    "one included",
    "the site-volume checkbox gates the draw",
    "and the picture really changed, not just the flag",
    "switching it back restores the cloud",
)


def _close_section() -> None:
    """Print the section that just ended, with its own subtotal.

    Called from `section` when the next one opens, and once at the end of the
    report, so a section is never left open. An open section is the signature of
    a run that stopped inside it, and printing that is the whole point: the last
    line of a truncated transcript is then a section header with no subtotal
    under it, which reads as unfinished to anyone skimming, without needing the
    expected total and without trusting a number.
    """
    global _section_open, _section_first
    if _section_open is None:
        return
    title, first = _section_open, _section_first
    ran = results[first:]
    npass = sum(1 for r in ran if r[0] == "PASS")
    nfail = sum(1 for r in ran if r[0] == "FAIL")
    nskip = sum(1 for r in ran if r[0] == "SKIP")
    section_skips[title] = nskip
    section_totals[title] = (len(ran), npass, nfail, nskip)
    note = "" if ran else "   <-- NO CHECKS RAN IN THIS SECTION"
    print(
        f"--- {title}: {len(ran)} checks, {npass} passed, "
        f"{nfail} failed, {nskip} skipped{note}"
    )
    _section_open = None


def section(title: str) -> None:
    _close_section()
    global _section_open, _section_first
    sections_entered.append(title)
    print(f"\n=== {title} ===")
    _section_open = title
    _section_first = len(results)

class _ProbeEvent(QtCore.QEvent):
    """A custom event whose *delivery* is the observation.

    Nothing about it is a measurement: it either arrives at its receiver or it
    does not, and the only way it arrives is the event loop running.
    """

    def __init__(self, token: int) -> None:
        super().__init__(QtCore.QEvent.Type.User)
        self.token = token


class _ProbeSink(QtCore.QObject):
    """Receives `_ProbeEvent`s and records which tokens actually arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.delivered: list[int] = []

    def event(self, ev) -> bool:  # noqa: N802 - Qt naming
        if isinstance(ev, _ProbeEvent):
            self.delivered.append(ev.token)
        return super().event(ev)


def loop_turns(app, probes: int = 8, budget: int = 64) -> tuple[int, int]:
    """Post `probes` events and count how many the event loop actually delivered.

    **This is a drain count, not a rate.** The previous version of this function
    counted `processEvents()` calls in fixed wall-clock windows and compared the
    best window against a threshold. That was a claim about the machine: the
    loop is not free, it does real repaints on a real window, so how many turns
    fit in 0.2 s depends on this processor and on what else is running. It read
    29 turns in one run and 7 in another on the same idle machine, and the
    first of five windows in a quiet baseline run read **1** while the other four
    read 20, 21, 24 and 29. A threshold compared against that number is a
    threshold on load.

    What is asked here is a *discrete* question with a discrete answer: post N
    events at a receiver, return control to the loop, and see whether they came
    out the other end. A loop that cannot turn delivers **none** of them however
    long it is given, because an undelivered event stays queued forever. A loop
    that can turn delivers all N, and does it in one pass, because
    `processEvents()` drains the queue it is given rather than sampling it. So
    the number is a property of the code under test, not of the clock: a slower
    machine returns the same 8, just a few milliseconds later.

    `budget` is a pump cap so a genuinely blocked loop cannot hang the suite. It
    is not a threshold: any budget >= 1 gives the same answer for a live loop,
    and a starved one reads 0 at every budget. Returns (delivered, pumps).
    """
    sink = _ProbeSink()
    for i in range(probes):
        QtCore.QCoreApplication.postEvent(sink, _ProbeEvent(i))
    # The posted events do not keep the Python wrapper alive on their own, and a
    # sink collected mid-pump would stop recording without anything failing, so
    # the name is bound to a local that outlives the loop.
    keep = sink
    del sink
    pumps = 0
    while len(keep.delivered) < probes and pumps < budget:
        app.processEvents()
        pumps += 1
    return len(keep.delivered), pumps


class _StarvedEvents:
    """An `app` stand-in whose event loop cannot turn.

    `processEvents` returns without touching the queue, which is what a GUI
    thread looks like when something is holding it. Used to reverse-verify the
    drain count, which is the same role `_StarvedEvents` had for the turn count.
    """

    def processEvents(self):  # noqa: N802 - Qt naming
        return None


def _verify_loop_turns_guard(probes: int = 8):
    """Reverse-verify the liveness criterion the way the pixel guard is verified.

    **What this claims.** The criterion in section 11d is "the events queued for
    the GUI thread came out", and the only thing that can make that false is a
    loop that does not run. So the guard runs the *same function* against a loop
    that cannot turn and requires it to deliver nothing, and against the real
    loop and requires it to deliver everything. Both halves are exact counts, so
    neither can drift with load: the starved one is 0 because an undelivered
    event stays queued, not because the window was too short, and the live one
    is `probes` because one `processEvents()` drains the queue.

    This is the whole claim, and it is a smaller claim than the one it replaces.
    The old control asserted that a free loop could also *reach a rate*, which is
    a fact about this processor. It is deliberately not asserted here, and the
    old `held 50 ms per turn` arithmetic is gone with it: there is no duration
    left in the criterion to be bound.
    """
    starved = loop_turns(_StarvedEvents(), probes=probes)
    live = loop_turns(QtWidgets.QApplication.instance(), probes=probes)
    starved_caught = starved[0] == 0
    live_ok = live[0] == probes
    ok = starved_caught and live_ok
    return (
        "a loop that cannot turn delivers none of the queued events, and a loop "
        "that can turn delivers all of them: a drain count, not a rate",
        ok,
        f"{probes} events posted to each: a loop that never pumps delivered "
        f"{starved[0]} of them in {starved[1]} pumps (an undelivered event stays "
        f"queued, so this is exact rather than a matter of how long it waited), "
        f"and the real loop delivered {live[0]} of {probes} in {live[1]} "
        f"pump{'s' if live[1] != 1 else ''} -> "
        f"{'caught' if starved_caught else 'NOT CAUGHT'}. Neither number is "
        f"compared with a duration, so neither can move with the load on the "
        f"machine; the old control's free-loop rate is not asserted here at all, "
        f"because a rate is a fact about the processor and not about the window",
    )


def img_array(qimg: QtGui.QImage) -> np.ndarray:
    """QImage -> (h, w, 3) uint8, independent of stride and format."""
    img = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return arr[:, : img.width(), :3].copy()


SHOT_PROBLEM: str | None = None


#: What the launcher's own OpenGL probe said about this machine, asked once at
#: the top of the run. `odgui --check` already answers "can this machine give Qt
#: a context?" in about six seconds, and it answers it about the product rather
#: than about a probe invented here -- so this suite asks it instead of
#: inferring the answer from a blank frame, which is how three experiments
#: minutes apart managed to give three different answers about one machine.
GL_VERDICT: str | None = None
GL_OK = True


def probe_opengl(app) -> None:
    """Ask the launcher whether this machine can render at all, and record it."""
    global GL_VERDICT, GL_OK
    try:
        from opendocking.workbench.launcher import _probe_opengl
    except Exception as exc:  # noqa: BLE001 - a missing probe is not a failure
        GL_VERDICT = f"the launcher's OpenGL probe could not be imported ({exc})"
        return
    found, problem = _probe_opengl()
    bits = [f"platform {found.get('platform')}", f"{found.get('screens')} screen(s)"]
    if found.get("gl"):
        bits.append(f"OpenGL {found['gl']}")
    if problem is None:
        GL_OK = True
        GL_VERDICT = "Qt created an OpenGL context -- " + ", ".join(bits)
        return
    GL_OK = False
    GL_VERDICT = (
        f"this machine cannot give Qt an OpenGL context ({problem}); "
        + ", ".join(bits)
        + f"; waited {found.get('waited', 0):.1f} s. The same question asked "
          "through `odgui --check`, which reports a graphics-driver problem "
          "rather than a missing package."
    )


def _bare_widget_exposed(app, tries: int = 10, gl: bool = False) -> bool:
    """Control experiment: does a *bare* window get exposed on this machine?

    With `gl=False` that is a plain `QWidget`; with `gl=True` it is a bare
    `QOpenGLWidget` -- still none of this project's code and nothing a change to
    the pose table could have touched. Two controls because they fail
    differently: if the plain one is never exposed the desktop is presenting
    nothing, and if only the GL one is not, the GL stack is refusing this
    machine a surface. Both are the machine's doing; neither is a broken
    viewport, and telling them apart is what stops this being "fixed" by
    retrying until a frame appears.
    """
    probe = QtOpenGLWidgets.QOpenGLWidget() if gl else QtWidgets.QWidget()
    probe.resize(64, 48)
    probe.show()
    exposed = False
    for _ in range(tries):
        app.processEvents()
        QTest.qWait(60)
        handle = probe.windowHandle()
        if handle is not None and handle.isExposed():
            exposed = True
            break
    probe.close()
    probe.deleteLater()
    return exposed


def frame_problem(win) -> str:
    """Why there is no frame, naming which of the causes this is.

    `QOpenGLWidget.grabFramebuffer()` returns a null `QImage` when the widget
    has no surface to read. The causes are not interchangeable, and "it was
    still drawing" -- the one people reach for first -- is already ruled out by
    the evidence this suite has in hand before it calls in here: `main` waits up
    to 5 s for the window to be exposed (`QTest.qWaitForWindowExposed`) and
    fails, and the first grab is null for the same reason as the last. Waiting
    cannot create a surface the window manager never gave us, so this suite does
    not wait to find out, and does not call a late frame a pass.
    """
    app = QtWidgets.QApplication.instance()
    if not GL_OK:
        # Asked at the top of the run, by the product's own probe. Everything
        # below about "no frame" is then a restatement of this rather than a
        # second opinion reached by experiment.
        return GL_VERDICT or "no OpenGL context"
    handle = win.windowHandle()
    if handle is None:
        return "the window has no native handle at all"
    if handle.isExposed():
        return "the window is exposed, so the GL surface itself produced no frame"
    if not _bare_widget_exposed(app):
        return (
            "this machine does not expose Qt windows: a bare QWidget with no "
            f"OpenGL is unexposed too, so no frame can be read here at all "
            f"(platform {app.platformName() if app else '?'})"
        )
    if not _bare_widget_exposed(app, gl=True):
        return (
            "plain windows are exposed but a bare QOpenGLWidget is not, so this "
            "machine's GL stack is not giving Qt a surface to read; the "
            "workbench's own drawing is untested either way"
        )
    return (
        "a bare window and a bare QOpenGLWidget are both exposed, so the "
        "failure is in the workbench window, not in the environment"
    )


def shot(win: MainWindow, name: str) -> tuple:
    """Render the real widget and save it. Returns (pixels|None, path).

    Returns ``(None, "")`` rather than raising when the framebuffer cannot be
    read. This is called near the top of `main`, so raising here would abort the
    run before the other ~180 checks and before the summary line -- a suite that
    stops early looks like a suite that stopped early, and everyone assumes the
    rest is fine. The caller turns a missing frame into a counted SKIP with the
    reason, which is a different claim from a FAIL and stays a different claim
    in the summary.
    """
    global SHOT_PROBLEM
    win.viewport.repaint()
    QTest.qWait(40)
    img = win.viewport.grabFramebuffer()
    if img.isNull():
        if SHOT_PROBLEM is None:
            SHOT_PROBLEM = frame_problem(win)
            print(f"\n  NO FRAME from the viewport: {SHOT_PROBLEM}")
        return None, ""
    arr = img_array(img)
    path = OUT / f"{name}.png"
    img.save(str(path))
    return arr, str(path)


def background_colour(arr: np.ndarray) -> np.ndarray:
    """The clear colour, taken as the modal colour of the framebuffer.

    `paintGL` clears to (0.10, 0.11, 0.13) but the window also carries whatever
    the desktop compositor put behind an unpainted region, so the mode of the
    actual pixels is the honest reference.
    """
    flat = arr.reshape(-1, 3)
    key = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    modal = vals[int(np.argmax(counts))]
    return np.asarray([(modal >> 16) & 0xFF, (modal >> 8) & 0xFF, modal & 0xFF], np.int16)


def non_background(arr) -> int:
    """Pixels that differ from the modal (background) colour.

    `None` in, `0` out: a frame that could not be read has no non-background
    pixels, and saying so is what routes the pixel checks to a skip instead of
    a failure. A crash here would be the third way this run could die before its
    summary, and the least informative of the three.
    """
    if arr is None:
        return 0
    bg = background_colour(arr)
    d = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
    return int((d > 12).sum())


def dispose_window(app, win, destroy: bool = True) -> None:
    """Close a window and actually let go of its GL context.

    `close()` on its own hides the window; it does not destroy it. A
    `QOpenGLWidget` that is still alive still owns its OpenGL context, so a file
    that opens a second window while the first is merely closed is running two
    live contexts on one surface. Forcing the deferred delete is what fixed that
    -- it moved the `0xC0000409` fail-fast from section 7c to the very last
    section of the file.

    `destroy=False` is for the one window that has run a worker thread. There,
    the QOpenGLWidget is destroyed while a QThread that used to drive it is
    still an object, and on this machine that combination fail-fasts. That
    window is closed and left to process exit instead, which is the right
    trade for the last window in a run: nothing follows it that needs the
    context released.

    `processEvents()` on its own does not reliably process `DeferredDelete`
    events, which is why a plain `close()` plus `processEvents` left the
    context behind in the first place.
    """
    if win is None:
        return
    win.close()
    if not destroy:
        app.processEvents()
        return
    win.deleteLater()
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def differing_pixels(a, b) -> int:
    """Pixels that differ between two frames, or 0 when either is missing."""
    if a is None or b is None:
        return 0
    return int((a != b).any(axis=-1).sum())


def _grab(win: MainWindow):
    """Repaint and read the frame back, without writing a file.

    `shot` is the one that saves, and a measurement probe has no business
    leaving half a dozen `_footprint_before.png` files in the screenshot
    directory for a human to wonder about. Returns ``None`` rather than raising
    when the framebuffer cannot be read, the same as `shot`: the pixel guard
    routes that case to a skip, and a probe that raised would take the rest of
    the section with it.
    """
    win.viewport.repaint()
    QTest.qWait(40)
    img = win.viewport.grabFramebuffer()
    if img.isNull():
        return None
    return img_array(img)


def _own_pixels(win: MainWindow, hide):
    """``(frame, mask)`` for the pixels `hide` is responsible for.

    Same camera and same frame either side, so the mask is that one thing's own
    footprint. `hide` is a callable returning a restorer -- `_hide_role` builds
    one.
    """
    before = _grab(win)
    restore = hide()
    after = _grab(win)
    restore()
    _grab(win)
    if before is None or after is None:
        return None, None
    mask = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2) > 8
    return before, mask


def _channel_excess(frame, mask, hi: int, lo: int) -> float:
    """Share of `mask`'s pixels where channel ``hi`` leads channel ``lo``.

    **Channel order, because it is the whole answer here.** The framebuffer
    arrives as `Format_RGB32`, whose bytes are B, G, R, A -- so channel 0 is
    *blue*, 1 is green and 2 is *red*. Read it as RGB and this measures a
    sign-flipped quantity, which reports the slate ghost as redder than the
    protein and would send the next person looking for a lighting bug.

    0.02 is about five 8-bit levels: enough to shrug off shading and rounding,
    small enough that carbon grey (0.58 against 0.55, so 0.03 the other way) and
    oxygen red (0.90 against 0.20) both fall on the correct side of it.
    """
    if frame is None or mask is None or not mask.any():
        return 0.0
    a = frame[:, :, hi].astype(np.float32) / 255.0
    b = frame[:, :, lo].astype(np.float32) / 255.0
    return float(((a - b)[mask] >= 0.02).mean())


def _bluer_fraction(frame, mask) -> float:
    """Share of `mask`'s pixels that are bluer than red."""
    return _channel_excess(frame, mask, 0, 2)


def _greener_fraction(frame, mask) -> float:
    """Share of `mask`'s pixels that are greener than *both* red and blue.

    `min` of the two excesses rather than either one alone: a pose that is
    green beside a blue ghost has to beat both, and "green leads red" is
    satisfied by a blue object as easily as by a green one.
    """
    if frame is None or mask is None or not mask.any():
        return 0.0
    return min(
        _channel_excess(frame, mask, 1, 2),
        _channel_excess(frame, mask, 1, 0),
    )


# ---------------------------------------------------------------------------
# Hue, counted only where there is one
# ---------------------------------------------------------------------------
#
# `_greener_fraction` and `_bluer_fraction` above ask about every pixel in the
# mask, which is the right question for the default representation: a
# space-filling pose is 8 578 lit sphere pixels and 78% of them have a hue.
#
# It is the wrong question for the bond-only ones, and the reason is measured
# rather than guessed. A thin lit cylinder is mostly silhouette -- the normals
# on either side of it graze the light and come back nearly black -- so 34% of
# the pixels a `stick` pose owns have no hue at all. Counting a near-black pixel
# against "is this green" measures the renderer's lighting, not the colour
# hierarchy the check is about, and it fails the pose for being thin. The four
# representations that fell over this way all read 77.6-83.5% while the same
# pixels measured over the chromatic population alone read 92.1-97.7%.

#: A pixel needs this much spread between its brightest and darkest channel
#: before it has a hue. 24 of 255 is ~9%: far above the 1-2 levels of rounding in
#: an 8-bit framebuffer, far below the 61 a slate ghost spreads between its own
#: extremes, and far below `COLOR_BEST_POSE`'s green.
CHROMA_MIN = 24

#: The selected pose has to be the vivid thing: this much more chromatic than
#: the ghosts it is read against. Re-derived when the space-filling mode joined
#: the per-representation sweep, and the derivation is the point rather than the
#: number.
#:
#: The clause exists so a pose that has lost its colour cannot pass on hue
#: alone: both hue fractions are absolute, so a uniformly desaturated pose
#: could in principle be "not slate" and "not green" at the same time. What the
#: clause has to express is therefore "the pose is at least as chromatic as the
#: ghosts", and 1.0 is the exact boundary of that claim. Any number above it is
#: a preference about how vivid a pose ought to look, not a statement about
#: which failures this catches.
#:
#: Measured pose-over-ghost chroma across all six representations: 3.5x
#: (`spheres`), 4.9x (`ball_and_stick`), 5.0x (`stick`), 5.1x (`ribbon`), 5.1x
#: (`cartoon`), 1.9x (`space_filling`). The last is low because *the ghosts got
#: brighter*, not because the pose got worse: at CPK radii the receptor is a
#: solid lit surface, so the ghosts sit against that surface instead of against
#: background and 47% of their pixels pick up a hue they would not otherwise
#: have had. The pose's own chromatic fraction in that mode is 89%, the
#: highest of the six, which is the part of the claim that has to hold.
#:
#: This read `2.0`, which `space_filling` misses at 1.9. Choosing 1.5 would have
#: been the same decision with more decimals, so it is 1.0 -- and the floor is
#: checked against a control below, because a floor that any pose passes is not
#: a floor.
POSE_CHROMA_FLOOR = 1.0


def _chromatic(frame) -> np.ndarray:
    """Per-pixel channel spread -- how much colour a pixel carries.

    **`int16`, not the framebuffer's own dtype, and that is load-bearing.**
    `uint8 - uint8` wraps: a pixel 38 levels bluer than green comes back 218
    levels *greener*. A measurement written on the raw dtype therefore reports
    every population in the frame -- the slate ghosts included -- as ~100%
    green, and does so while looking perfectly plausible. The first version of
    the numbers above was measured that way and said 99.7% green for the ghosts.
    It was caught only because the ghosts being green is visibly absurd, which is
    not a property one can rely on catching twice.
    """
    f = frame.astype(np.int16)
    return f.max(axis=2) - f.min(axis=2)


def _chromatic_fraction(frame, mask) -> float:
    """Share of `mask`'s pixels that have a hue at all."""
    if frame is None or mask is None or not mask.any():
        return 0.0
    return float((_chromatic(frame)[mask] >= CHROMA_MIN).mean())


def _channel_excess_chromatic(frame, mask, hi: int, lo: int) -> float:
    """`_channel_excess`, restricted to the pixels of `mask` that have a hue.

    The 5-level margin is the same one `_channel_excess` documents, in the same
    units: `>= 0.02` of full scale is `>= 5.1` of 255, and rounding is to the
    nearer level. Using a looser 2 here would have made the new helpers quietly
    more forgiving than the old ones rather than more accurate, which is the one
    change to a threshold that is never a fix.
    """
    if frame is None or mask is None or not mask.any():
        return 0.0
    f = frame.astype(np.int16)
    sel = mask & (_chromatic(frame) >= CHROMA_MIN)
    if not sel.any():
        return 0.0
    return float(((f[:, :, hi] - f[:, :, lo])[sel] >= 5).mean())


def _bluer_fraction_chromatic(frame, mask) -> float:
    return _channel_excess_chromatic(frame, mask, 0, 2)


def _greener_fraction_chromatic(frame, mask) -> float:
    return min(
        _channel_excess_chromatic(frame, mask, 1, 2),
        _channel_excess_chromatic(frame, mask, 1, 0),
    )


def _set(vp, **attrs):
    """Set viewport attributes; returns a callable that puts them back."""
    saved = {k: getattr(vp, k) for k in attrs}
    for k, v in attrs.items():
        setattr(vp, k, v)

    def restore():
        for k, v in saved.items():
            setattr(vp, k, v)

    return restore


def _save_pose_crop(win: MainWindow, name: str, radius: float = 150.0) -> str:
    """Save a crop of the live viewport around the pose cluster.

    This is the image a person actually needs for the overlay, and the whole
    viewport is the wrong size for it: nine poses inside a 300-atom protein
    occupy a corner of a 1251 px frame.

    **The crop is taken in the framebuffer's own pixels, and that is not the
    widget's.** `grabFramebuffer` returns device pixels, so a 1001 px wide
    viewport hands back a 1251 px image on a display whose `devicePixelRatio` is
    1.25. The projection below is in widget coordinates -- it has to be, they
    are what the camera matrices are defined against -- so it is scaled by the
    ratio before it is used as an index. Computing the box in widget pixels and
    slicing the image with it gives a crop of the wrong place at 0.8 scale,
    which reads as a framing bug rather than an indexing one and sends the next
    person looking for a camera problem that is not there.
    """
    vp = win.viewport
    frame = _grab(win)
    if frame is None:
        return ""
    chunks = [
        np.asarray(m.coords, np.float64)
        for m in vp.molecules
        if m.role in ("pose", "pose_ghost") and len(m.coords)
    ]
    if not chunks:
        return ""
    points = np.concatenate(chunks, axis=0)
    aspect = vp.width() / max(vp.height(), 1)
    mvp = vp.camera.projection(aspect) @ vp.camera.view_matrix()
    homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    clip = homo @ mvp.T
    w = np.where(np.abs(clip[:, 3:4]) < 1e-9, 1e-9, clip[:, 3:4])
    ndc = clip[:, :3] / w
    dpr = vp.devicePixelRatioF()
    # Widget pixels -> device pixels, once, here.
    sx = (ndc[:, 0] * 0.5 + 0.5) * vp.width() * dpr
    sy = (0.5 - ndc[:, 1] * 0.5) * vp.height() * dpr
    cx, cy = float(sx.mean()), float(sy.mean())
    r = radius * dpr
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(frame.shape[1], cx + r)), int(min(frame.shape[0], cy + r))
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1], np.uint8)
    if crop.size == 0:
        return ""
    h, wid, _ = crop.shape
    img = QtGui.QImage(
        crop.tobytes(), wid, h, wid * 3, QtGui.QImage.Format.Format_RGB888
    )
    path = OUT / f"{name}.png"
    img.save(str(path))
    print(
        f"  crop {name}: {wid}x{h} at ({x0},{y0}), dpr {dpr}, "
        f"centre ({cx:.0f},{cy:.0f}) -> {path}"
    )
    return str(path)


def _hide_role(win: MainWindow, role: str):
    """Hide every view with this role; returns a callable that shows them again.

    `visible` is written by `_apply_pose_visibility` from the checkboxes, so a
    probe has to put back what it found rather than assume `True` -- and it has
    to put it back through the same setter, or the next checkbox click would
    read a flag the probe left behind.
    """
    was = [(m, m.visible) for m in win.viewport.molecules if m.role == role]
    for m, _ in was:
        m.visible = False

    def restore():
        for m, v in was:
            m.visible = v

    return restore


def _footprint_of(win: MainWindow, mutate) -> int:
    """How much of the frame a change accounts for, in pixels.

    `mutate` is handed the viewport and must return a callable that undoes
    whatever it did -- `_set` is the usual way to build one.

    A *differential* measure, and that is the whole point. The obvious
    alternative -- count the pixels that are not the background colour -- is
    useless here: `paintGL` draws a gradient behind everything, so "not
    background" is mostly a count of the background, and every contributor
    measures in the hundreds of thousands. Changing one thing, re-rendering, and
    differencing cancels the gradient and leaves only that thing's own share of
    the picture, which is the quantity a visual-hierarchy claim is actually
    about.

    It restores what it changed. A probe that left the scene altered would make
    every check after it meaningless, and a measurement helper is exactly the
    kind of code that gets trusted to be harmless.
    """
    vp = win.viewport
    before = _grab(win)
    restore = mutate(vp)
    after = _grab(win)
    restore()
    _grab(win)
    return differing_pixels(before, after)


def drag(widget, button, start, end, steps=6) -> None:
    nomod = QtCore.Qt.KeyboardModifier.NoModifier
    QTest.mousePress(widget, button, nomod, QtCore.QPoint(*start))
    for i in range(1, steps + 1):
        t = i / steps
        x = int(start[0] + (end[0] - start[0]) * t)
        y = int(start[1] + (end[1] - start[1]) * t)
        QTest.mouseMove(widget, QtCore.QPoint(x, y))
        QTest.qWait(8)
    QTest.mouseRelease(widget, button, nomod, QtCore.QPoint(*end))


def rects_overlap(a: QtCore.QRect, b: QtCore.QRect) -> bool:
    return a.intersects(b) and (a & b).width() > 0 and (a & b).height() > 0


# ---------------------------------------------------------------------------
# Section 12: the crash guard, and the children that drive it.
#
# The children are written to a temp directory and run by path rather than with
# `-c`, for one reason: a traceback from `python -c` names `<string>`, and a
# guard whose subject is "the file and the line are named" cannot be satisfied by
# a filename that is not a file. As files they also make the expected line
# number derivable from the source the child was given, which is what makes
# check 12's line check independent of the hook it is checking.
#
# `offscreen`, forced on the children and not on this process: the fault under
# test is a Python-level slot, which the Qt dispatch raises regardless of the
# platform, and a child that opened a window on the desktop would be a nuisance
# to run as part of a gate. The product's viewport cannot get a context there,
# which costs the child its painting and nothing else -- `run()` still returns
# the code `crashguard` asked for.
# ---------------------------------------------------------------------------
_SLOT_FAULT_CHILD = r'''
import json
import sys

from opendocking.workbench import app as appmod
from opendocking.workbench import crashguard

mode = sys.argv[1]
if mode == "nohook":
    # The product exactly as it was before `crashguard` existed: the hook
    # `install()` replaced, put back. Same code, same dispatch, same fault.
    sys.excepthook = crashguard.replaced_hook()
elif mode == "plain":
    # The shape that was refused: print the traceback and return. Kept in the
    # suite so the reason for the exit code is measured rather than asserted.
    import traceback

    def _print_only(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = _print_only

from PyQt6 import QtCore, QtWidgets

qt = QtWidgets.QApplication(sys.argv[:1])


def _raise_in_slot(self):
    raise ValueError("injected: the term map could not be keyed")


def _chain_in_slot(self):
    try:
        raise KeyError("injected: the term-map key went missing")
    except KeyError as exc:
        raise ValueError("injected: the term map could not be keyed") from exc


def _catch_in_slot(self):
    try:
        raise KeyError("injected, and caught in the slot on purpose")
    except KeyError as exc:
        print(f"slot caught it: {exc}", file=sys.stderr, flush=True)


_FAULTS = {
    "hook": _raise_in_slot,
    "nohook": _raise_in_slot,
    "plain": _raise_in_slot,
    "chained": _chain_in_slot,
    "caught": _catch_in_slot,
}
if mode in _FAULTS:
    appmod.MainWindow._on_contact_visibility = _FAULTS[mode]


def _click_contacts():
    # A real control, toggled for real: the fault is raised from a signal the
    # window itself connected, dispatched by Qt, which is the boundary.
    for widget in QtWidgets.QApplication.topLevelWidgets():
        box = getattr(widget, "cb_contacts", None)
        if box is not None:
            box.click()
            print("clicked the contacts checkbox", file=sys.stderr, flush=True)
            return
    print("no window carried a contacts checkbox", file=sys.stderr, flush=True)


QtCore.QTimer.singleShot(400, _click_contacts)
QtCore.QTimer.singleShot(2500, QtWidgets.QApplication.quit)
code = appmod.run()
print(f"run() returned {code}", file=sys.stderr, flush=True)
print(f"failures={crashguard.failures()!r}", file=sys.stderr, flush=True)
# The record as JSON, not as the repr above: a Windows path in a repr is
# backslash-escaped, so a parent matching on the text would be matching on an
# encoding rather than on the record.
print("RECORD " + json.dumps([dict(r) for r in crashguard.failures()]),
      file=sys.stderr, flush=True)
sys.exit(code)
'''

_CHECK_AB_CHILD = r'''
"""`odgui --check`, once with the hook in place and once with it removed."""
import json
import sys

from opendocking.workbench import crashguard  # noqa: F401
from opendocking.workbench import app as _app  # noqa: F401  installs the hook
from opendocking.workbench import launcher

if sys.argv[1] == "off":
    sys.excepthook = crashguard.replaced_hook()
code = launcher._report_check(as_json=True)
print("HARNESS " + json.dumps({
    "exit": code,
    "active": crashguard.active(),
    "failures": len(crashguard.failures()),
}), file=sys.stderr, flush=True)
raise SystemExit(code)
'''


def _crash_child(script: Path, args: list[str], budget: float = 180.0,
                 platform: str | None = "offscreen") -> tuple:
    """Run one section-12 child in the foreground; return `(exit, out, err)`.

    Foreground and unbuffered on purpose. A child that dies on the way out can
    leave a pipe buffer half-written, and this project's two worst "crashes"
    were both a pipeline closing early rather than a process dying -- so the
    exit code, the stdout and the stderr are collected by `subprocess` itself
    and printed by this file.

    `platform=None` leaves the platform alone, which the `--check` child needs:
    offscreen has no OpenGL, so the diagnostic under `--check` there answers "no
    context" and the comparison would be between two failures. The fault child is
    the opposite case -- its subject is a Python-level slot, which the Qt
    dispatch raises on any platform, and offscreen keeps a window off the
    desktop while a gate runs.
    """
    import os
    import subprocess

    env = dict(os.environ)
    if platform is None:
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env["QT_QPA_PLATFORM"] = platform
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # The children import the product, so they have to be able to find the copy
    # this process is testing rather than one that happens to be installed.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-u", str(script), *args],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=budget,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _record_of(stderr: str) -> list:
    """The `RECORD` line a fault child printed: `crashguard.failures()` as JSON."""
    import json

    for line in stderr.splitlines():
        if line.startswith("RECORD "):
            try:
                loaded = json.loads(line[len("RECORD "):])
            except ValueError:
                return []
            if isinstance(loaded, list):
                return loaded
    return []


def _verdict_of(stdout: str) -> dict | None:
    """The `--check` JSON object a child printed, or `None` if it printed none."""
    import json

    for line in reversed(stdout.strip().splitlines()):
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict) and "verdict" in loaded:
            return loaded
    return None


#: The `--check` payload fields that are wall-clock readings of one run and so
#: cannot be equal across two runs. Named here, and stripped by
#: `_without_timings` from the same tuple, because a list of excused fields
#: spelled somewhere other than the code that removes them is a list that will
#: drift.
TIMING_KEYS = ("waited", "show_seconds")


def _without_timings(payload: dict | None) -> dict:
    """A `--check` payload with the wall-clock fields removed, recursively.

    Two runs of the same diagnostic cannot agree on how long they took, so
    those fields are excluded and named in the check. Everything else -- the
    verdict, the exit code, the GL version, the provenance digest -- has to be
    equal, and is.
    """

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in TIMING_KEYS}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(payload) if isinstance(payload, dict) else {}


#: The payload leaves whose disagreement means the **file** changed rather than
#: the **answer**. `_path_facts` in the launcher publishes `sha256` and `bytes`
#: per resolved module, so a digest that moves between two probes of the same
#: command is the file being rewritten underneath them.
DIGEST_LEAVES = ("sha256", "bytes")


def _flatten(value, prefix: str = "") -> dict:
    """Every leaf of a payload, keyed by its dotted path."""
    out: dict = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = value
    return out


def _moving_leaves(on: dict | None, off: dict | None) -> list:
    """Which payload leaves moved between two `--check` runs, and why it matters.

    `_without_timings` answers *whether* two payloads agree. This answers
    *where* they do not, down to a leaf, because the two things that can sit
    behind one `provenance` are different problems with different owners:

    * a **digest or a byte count moved**, so the file on disk changed between
      the two probes. Both runs answered correctly about different files, and
      the disagreement is about the tree, not about the product.
    * **anything else moved** -- a `measured_in` label, a verdict, a path -- and
      that is the two runs genuinely disagreeing, which is a product or a
      launcher question.

    A detail string that names only the top-level field cannot tell those
    apart. The sentence this replaces -- "Both runs report the same provenance
    digest, so they measured the same file" -- was hardcoded into the f-string
    and never computed, so on the one run where it was false it named the wrong
    cause with confidence.
    """
    fa, fb = _flatten(on or {}), _flatten(off or {})
    moved = []
    for leaf in sorted(set(fa) | set(fb)):
        if fa.get(leaf) == fb.get(leaf):
            continue
        moved.append((leaf, fa.get(leaf), fb.get(leaf)))
    return moved


def _is_digest_leaf(leaf: str) -> bool:
    return leaf.rsplit(".", 1)[-1] in DIGEST_LEAVES


def _panel_whole_reachable(app, win, dock, scroll, hbar) -> tuple[bool, str]:
    """Can the user drag the control panel to a width where it is shown whole?

    **A property, and the reason it is asked as a drag rather than as a
    number.** The 3-D view is this window's central widget and the panel is a
    left dock, so the view's minimum width *is* the window's minimum width, and
    a floor on the view is a ceiling on the room the controls can be given.
    That floor was 640, and at dpr 2.5 a 1280x820 request is clamped by a
    768x432 screen to 770x533: the dock was left 126 px, the panel is whole at
    511 px of dock, and no gesture could reach 511. The user had no way to give
    the controls room at all. Asking "is 511 px of dock reachable" is therefore
    the question; asking "is the dock 126 px" is only a symptom, and would read
    the same on a screen where 126 was plenty.

    **Two halves, because they fail differently and a reader needs to know
    which.** The drag asks whether a whole panel is reachable at all. The
    arithmetic asks whether the view's floor is what stands in the way, so a
    red result names the cause rather than restating the symptom. Both were
    false under the old floor at dpr 2.5; either alone would do, and both are
    here because a check that can only fail one way is a check waiting for the
    other.

    The window is handed back exactly as it was found, in a `finally`, because
    a check that leaves its subject resized is a different kind of lie -- and
    `_verify_panel_notice_checks` below runs on this same window.
    """
    if dock is None or scroll is None or hbar is None:
        return False, (
            f"there is nothing to drag: dock={'yes' if dock is not None else 'NO'}, "
            f"control_scroll={'yes' if scroll is not None else 'NO'}, "
            f"h-bar={'yes' if hbar is not None else 'NO'}"
        )
    vp = scroll.viewport()
    was_w, was_h = int(win.width()), int(win.height())
    was_dock = int(dock.width())
    # The narrowest dock width at which the horizontal bar reads 0, found by
    # walking the splitter down the way a drag does. Qt's own answer
    # (`resizeDocks`) rather than `win.width() - view_minimumWidth()`, which
    # would be the same arithmetic the bug was made of.
    narrowest = None
    try:
        for w in range(was_dock, 39, -1):
            win.resizeDocks([dock], [w], QtCore.Qt.Orientation.Horizontal)
            app.processEvents()
            if int(hbar.maximum()) <= 0:
                narrowest = int(dock.width())
            else:
                break
    finally:
        win.resizeDocks([dock], [was_dock], QtCore.Qt.Orientation.Horizontal)
        win.resize(was_w, was_h)
        for _ in range(6):
            app.processEvents()
        win._update_panel_notice()
        for _ in range(3):
            app.processEvents()

    # What the panel needs: its own minimum plus the title, frame and scroll bar
    # the dock puts around it, all read off the live widgets. The chrome is a
    # difference of two measurements rather than a constant, because a constant
    # here is a number picked to make this pass.
    panel_min = int(win.control_panel.minimumWidth())
    chrome = max(0, int(dock.width()) - int(vp.width()))
    need = panel_min + chrome
    view_min = int(win.viewport.minimumWidth())
    room = was_w - view_min
    reachable = narrowest is not None
    not_the_floor = room >= need
    ok = reachable and not_the_floor
    return ok, (
        f"narrowest dock width with the panel whole: "
        f"{narrowest if reachable else 'NONE REACHABLE'} px "
        f"(h-bar reached 0 at or above it from a start of {was_dock} px); "
        f"panel needs {panel_min} px plus {chrome} px of dock chrome = {need} px; "
        f"the 3-D view's floor is {view_min} px, so a {was_w} px window leaves "
        f"{room} px for the panel -- "
        + ("the floor is not what is in the way" if not_the_floor else
           f"the floor is holding {need - room} px the panel needs")
    )


def _panel_visibility_report(win, scroll, hbar) -> list[tuple[str, bool, str]]:
    """Is the control panel *usable*, which is a different question from reachable?

    Returns `(name, ok, detail)` triples rather than calling `check` itself, so
    that `_verify_panel_notice_checks` can run this exact predicate against
    deliberately broken windows. A predicate that only ever runs against a
    correct product is a predicate nobody has watched fail, and the two ways
    these three went wrong while they were being written -- a wrap width read
    from a widget that had not been laid out, and a row that kept the height of
    the previous line count -- were both invisible in the geometry and obvious
    in the frame. Sharing the code is what makes the reverse check honest
    rather than a re-implementation that can drift green.

    Every threshold is either Qt's own live answer (`maximum() > 0`) or a
    rectangle-containment identity. There is no width constant here to tune: a
    number would be one picked to make the check pass.
    """
    notice = getattr(win, "panel_notice", None)
    if scroll is None or notice is None:
        # **A failure and not a skip.** The notice is a product requirement, not
        # an optional extra, and a check that steps aside when the thing it
        # checks is missing is the "hiding behind a guard" case this file's own
        # completeness check exists to catch -- applied to itself. Deleting
        # `panel_notice` from `app.py` would otherwise be invisible here, and a
        # skip reads as "this environment could not answer" when what is true
        # is "the product stopped saying so".
        return [(
            "the control panel is shown whole, or the workbench says it is not",
            False,
            f"the window has no control_scroll/panel_notice to ask: "
            f"control_scroll={'yes' if scroll is not None else 'NO'}, "
            f"panel_notice={'yes' if notice is not None else 'NO'}; the notice "
            f"is what makes a narrow panel honest, so its absence is the defect "
            f"this check exists to find",
        )]

    h_needed = int(hbar.maximum())
    cut_off = h_needed > 0
    showing = bool(notice.isVisible())
    panel_min = int(win.control_panel.minimumSizeHint().width())
    vp_w = int(scroll.viewport().width())
    out: list[tuple[str, bool, str]] = []

    # The invariant, in both directions: the notice appears exactly when the
    # panel is cut off. One direction alone would pass against a notice that is
    # permanently on, which is a different defect.
    out.append((
        "the control panel is shown whole, or the workbench says it is not",
        showing is bool(cut_off),
        (f"panel cut off by {h_needed} px and the notice is up"
         if cut_off else
         f"panel is whole (h-bar max {h_needed}) and the notice is "
         f"{'showing' if showing else 'hidden'}")
        + f"; panel min {panel_min} px, viewport {vp_w} px, 3-D min "
        f"{int(win.viewport.minimumWidth())} px",
    ))

    # The specific thing a report said was blank: at dpr 2.5 the five legend
    # entries were "not rendered at all". They are rendered -- measured at their
    # full advance width -- and they sit in the form's field column at x=122 in
    # a 110 px strip. So the question is not "is there a legend" but "is any of
    # it on screen".
    legend_labels = [
        w for w in win.legend_row.findChildren(QtWidgets.QLabel)
        if w.text() and w.text() not in ("━", "●")
    ]
    vp_rect = scroll.viewport().rect()
    off_screen = []
    for w in legend_labels:
        top = w.mapTo(scroll.viewport(), QtCore.QPoint(0, 0))
        r = QtCore.QRect(top, w.size())
        if not vp_rect.contains(r):
            off_screen.append(f"{w.text()!r} at {r.x()},{r.y()} "
                              f"{r.width()}x{r.height()}")
    out.append((
        "every legend entry is on screen, or the notice says the panel is not",
        not off_screen or showing,
        ("; ".join(off_screen) + f" | notice showing: {showing}")
        if off_screen else
        f"all {len(legend_labels)} legend words inside the {vp_w} px viewport",
    ))

    # The notice is a widget in the very strip it is warning about, so it can be
    # cut off by the very thing it reports.
    n_top = notice.mapTo(scroll.viewport(), QtCore.QPoint(0, 0))
    n_rect = QtCore.QRect(n_top, notice.size())
    inside = vp_rect.contains(n_rect)
    h_fw = (int(notice.heightForWidth(notice.maximumWidth()))
            if notice.wordWrap() else notice.height())
    out.append((
        "the notice is inside the strip it is warning about, whole",
        (not showing) or (inside and notice.height() >= h_fw),
        (f"notice {n_rect.x()},{n_rect.y()} {n_rect.width()}x{n_rect.height()} "
         f"in a {vp_w} px viewport, needs {h_fw} px of height for "
         f"{notice.maximumWidth()} px of width"
         if showing else "notice is hidden, so there is nothing to show"),
    ))

    # **The notice's own number, which no check here was reading.**
    #
    # The notice used to state the deficit as `panel.minimumSizeHint() -
    # viewport`, which is a different and smaller quantity than the bar's own
    # range -- 491 against a panel that was actually given 495 -- so at a 494 px
    # viewport the notice was up and said "they need 0 px more than the panel
    # has". A 355-check suite did not catch that, because every check above asks
    # whether the notice is *shown* and none asks whether what it says is true.
    # It stayed hidden while the boundary was 385 px wide and became ordinary
    # once the boundary was 2 px wide, which is the whole argument for reading
    # the sentence and not only the flag.
    #
    # Parsing prose is normally the wrong instrument and is used here anyway,
    # because the alternative is for the product to expose a number purely so
    # this file can read it, and a second copy of the deficit is one more thing
    # to drift. The regex is anchored on the product's own sentence shape and
    # the failure it reports is the product's text, so a reworded sentence shows
    # up as a red here rather than as silence.
    said = re.search(r"need (\d+) px more", notice.text())
    if not showing:
        out.append((
            "the notice never says the controls need zero pixels",
            True,
            "notice is hidden, so there is no number to be wrong",
        ))
    elif said is None:
        out.append((
            "the notice never says the controls need zero pixels",
            False,
            f"the notice is up but its text carries no 'need N px more' "
            f"figure to check: {notice.text()!r}",
        ))
    else:
        claimed = int(said.group(1))
        out.append((
            "the notice never says the controls need zero pixels",
            claimed >= 1 and claimed == h_needed,
            (f"notice up with the h-bar at {h_needed} px, and it says the "
             f"controls need {claimed} px more"
             if claimed == h_needed else
             f"notice up with the h-bar at {h_needed} px, and it says the "
             f"controls need {claimed} px more -- the sentence and the bar are "
             f"different quantities"
             + ("; a notice that says zero while it is up contradicts itself"
                if claimed == 0 else "")),
        ))
    return out


def _verify_panel_notice_checks(app, win) -> tuple[list[bool], list[str]]:
    """Reverse-verify the narrow-panel checks: they must fail on a broken panel.

    The same states `_verify_pixel_guard` insists on, for the same reason and
    with the same failure it is there to prevent: a guard that cannot go red is
    indistinguishable from a guard that is working. Each case below breaks one
    thing the product depends on and requires the shipped predicate -- the
    function section 1 calls, not a copy of it -- to notice.
    """
    global results
    saved_results, results = results, []
    verdict: list[bool] = []
    # **The window that is already up, not a second one.** A second `MainWindow`
    # means a second `QOpenGLWidget` and a second context, and this machine has
    # one: building a throwaway window here died with moderngl's "cannot create
    # vertex array" partway through section 1 and took the run with it. So the
    # real window is narrowed, broken four ways, and put back, and the put-back
    # is itself asserted -- a reverse verification that leaves the suite's
    # subject altered would be a different kind of lie.
    try:
        was_w, was_h = int(win.width()), int(win.height())
        scroll = win.control_scroll
        hbar = scroll.horizontalScrollBar()
        notice = win.panel_notice
        dock = next((d for d in win.findChildren(QtWidgets.QDockWidget)
                     if d.windowTitle() == "Controls"), None)

        def _unfix():
            notice.setMinimumWidth(0)
            notice.setMaximumWidth(16777215)
            notice.setMinimumHeight(0)
            notice.setMaximumHeight(16777215)

        win.resize(700, was_h)
        for _ in range(8):
            app.processEvents()
        # Read here, not at the end: this window is widened again further down,
        # and a figure printed from the live bar after that reports the state
        # the test just left rather than the state it tested. That mistake was
        # in the first version of this function and it printed "0 px" beside a
        # case that had in fact run against a genuinely cut-off panel.
        narrow_w = int(win.width())
        narrow_hbar_max = int(hbar.maximum())
        narrow_notice_showing = bool(notice.isVisible())
        verdict.append(narrow_hbar_max > 0)

        base = _panel_visibility_report(win, scroll, hbar)
        healthy = all(ok for _, ok, _ in base)

        # 1. the product as shipped, on a narrow window: all three must pass.
        # 2. the notice suppressed while the panel is still cut off: the first
        #    two must go red, which is the whole defect this work found.
        notice.setVisible(False)
        for _ in range(3):
            app.processEvents()
        muted = _panel_visibility_report(win, scroll, hbar)
        # 3. the notice stretched past the right edge of the strip it is
        #    warning about: the third must go red.
        notice.setVisible(True)
        notice.setFixedWidth(scroll.viewport().width() + 200)
        for _ in range(3):
            app.processEvents()
        stretched = _panel_visibility_report(win, scroll, hbar)
        # 4. the notice sliced to a height its own text does not fit in: the
        #    third must go red again, and this half is not the half case 3
        #    covers. It is here because it happened for real -- the row kept the
        #    height of the previous line count and cut the last line in half,
        #    which no width test would have noticed.
        _unfix()
        notice.setFixedHeight(4)
        for _ in range(3):
            app.processEvents()
        sliced = _panel_visibility_report(win, scroll, hbar)

        # 6. and the reachability predicate must go red when the 3-D view takes
        #    its floor back. This is the case the whole change is about, and it
        #    is here because the three cases above break the *notice*; nothing
        #    above could break the *splitter*, so a predicate that only ever saw
        #    a notice could not tell a floor that yields from one that decides.
        #
        #    The floor is set to the window's own current width, not to some
        #    remembered 640: that reproduces the old condition exactly -- the
        #    window already satisfies this minimum, so nothing is resized and
        #    the only thing that moves is the dock the layout squeezes -- and it
        #    keeps this case from depending on a number that is no longer in the
        #    product.
        saved_view_min = int(win.viewport.minimumWidth())
        win.viewport.setMinimumWidth(int(win.width()))
        for _ in range(4):
            app.processEvents()
        _floored_ok, _floored_detail = _panel_whole_reachable(
            app, win, dock, scroll, hbar)
        win.viewport.setMinimumWidth(saved_view_min)
        for _ in range(4):
            app.processEvents()
        floored_red = not _floored_ok

        # 7. and the notice's own figure must be readable as wrong when it is
        #    wrong. This is the case the 355 checks missed: the notice was up,
        #    the bar read 1, and the sentence said the controls needed 0 px more
        #    -- and every check above was satisfied, because they all ask
        #    whether the notice is *shown* and none asked what it says. The
        #    text is put back the way it was wrong and the shipped predicate has
        #    to notice.
        _unfix()
        real_text = notice.text()
        notice.setText(
            "the controls do not fit: they need 0 px more than the panel has, "
            "and scroll sideways.")
        notice.setVisible(True)
        for _ in range(3):
            app.processEvents()
        lying = _panel_visibility_report(win, scroll, hbar)

        # 5. and the window must be exactly as this function found it, with the
        #    product left in a state consistent with its own rule.
        #
        #    **Not "the panel is whole again".** At dpr 2.5 the screen is
        #    768x432 logical and the window cannot be widened past it, so the
        #    panel is cut off before this function starts and after it ends;
        #    demanding h-bar 0 here failed on that ratio for a reason that has
        #    nothing to do with the checks. What has to hold at every ratio is
        #    the product's own invariant -- the notice is up exactly when there
        #    is something to the right of it -- plus the geometry back where it
        #    was.
        _unfix()
        win.resize(was_w, was_h)
        for _ in range(8):
            app.processEvents()
        win._update_panel_notice()
        for _ in range(3):
            app.processEvents()
        back_hbar = int(hbar.maximum())
        back_showing = bool(notice.isVisible())
        restored = (
            int(win.width()) == was_w
            and int(win.height()) == was_h
            and back_showing is bool(back_hbar > 0)
        )

        verdict.append(healthy)
        verdict.append(not all(ok for _, ok, _ in muted))
        verdict.append(not all(ok for _, ok, _ in stretched))
        verdict.append(not all(ok for _, ok, _ in sliced))
        verdict.append(floored_red)
        verdict.append(not all(ok for _, ok, _ in lying))
        verdict.append(restored)
    finally:
        results = saved_results
    # The window really was narrow, the healthy case really was green, each of
    # the three breakages really did turn something red, and the window really
    # was handed back. Returned to `main`, which reports it as one check, rather
    # than reaching for the module globals `results` and the window belong to.
    return verdict, [
        f"the suite's own window, narrowed from {was_w}x{was_h} to {narrow_w} "
        f"px wide: the horizontal bar reached {narrow_hbar_max} px and the "
        f"notice was {'showing' if narrow_notice_showing else 'hidden'}",
        f"panel as shipped on that narrow window: all three green = {healthy}",
        f"notice suppressed while still cut off: something red = "
        f"{not all(ok for _, ok, _ in muted)}",
        f"notice stretched past the strip: something red = "
        f"{not all(ok for _, ok, _ in stretched)}",
        f"notice sliced to 4 px of height: something red = "
        f"{not all(ok for _, ok, _ in sliced)}",
        f"3-D view's floor raised to the window's own width, so the dock can no "
        f"longer be dragged anywhere whole: the reachability predicate went red "
        f"= {floored_red} ({_floored_detail}); the floor is back at "
        f"{saved_view_min} px",
        f"notice text claiming 0 px while the bar reads "
        f"{int(hbar.maximum())} px: something red = "
        f"{not all(ok for _, ok, _ in lying)}; the real sentence is back "
        f"({real_text!r})",
        f"handed back at {win.width()}x{win.height()} (was {was_w}x{was_h}), "
        f"h-bar max {back_hbar}, notice "
        f"{'showing' if back_showing else 'hidden'} -- the product's own rule "
        f"holds again, and the panel is "
        + ("still cut off at this ratio, because the screen clamps the window "
           "and no resize can widen it past that"
           if back_hbar > 0 else
           "whole again at this ratio")
        + f": {restored}",
    ]


# -- 1c. the 3-D view's own width notice -------------------------------------
#
# The panel has had a notice for its own shortage since the floor on the view
# came down, and the view had none. With the panel draggable out to 1036 px and
# the view left at its own 240 px floor, the 3-D picture was a vertical slice
# of the molecule running off both edges, and the product had no sentence for
# that state -- the reader had to notice on their own that they were looking at
# a fifth of the picture. These ask the view's question in the panel's rule
# shape: one measure, one threshold, and the figure in the sentence is the
# figure the threshold was tested against.


def _half_width_fill_is_the_modules_own_term() -> tuple[bool, str]:
    """The derived half-width term IS `framing_selection.drawn_fill`'s.

    `app._drawn_half_width_fill` writes out the horizontal term of
    `framing_selection.drawn_fill`, because `drawn_fill` answers "does the
    scene overflow the frame" by taking the worse of the two screen axes and a
    notice about *width* needs the width on its own. A second projection would
    be a second thing to keep in step with the first, so this pins the two
    together instead: on a scene whose width is the wider of its two screen
    extents, `drawn_fill`'s maximum over the axes *is* the horizontal term and
    the two numbers have to be equal.

    The second half catches the plausible wrong answer. Taking the vertical
    term -- the same expression without the `aspect` factor -- would satisfy an
    equality against a scene whose height dominated, so the scene is built three
    times as wide as it is tall and the vertical term is asserted to be the
    smaller of the two. It is the same argument `framing_selection`'s own
    `projected_fill` makes about which axis it is reporting.
    """
    from opendocking.workbench import framing_selection as fs
    from opendocking.workbench.app import _drawn_half_width_fill

    right = np.asarray([1.0, 0.0, 0.0])
    up = np.asarray([0.0, 1.0, 0.0])
    forward = np.asarray([0.0, 0.0, 1.0])
    pts = np.asarray([[-8.0, 0.0, 0.0], [8.0, 0.0, 0.0],
                      [0.0, 0.5, 1.0], [0.0, -0.5, -1.0]])
    radii = np.asarray([0.3, 0.3, 0.2, 0.2])
    centre = np.zeros(3)
    # **A non-unit aspect, and that is load-bearing.** At 1.0 the
    # `aspect` factor is a division by one, so an expression that had
    # lost the factor entirely would return the same number and this
    # check would stay green on the one mutation it exists to catch.
    # Measured: with the factor, both sides read 0.4217; without it,
    # 0.6747. The factor is what makes this the width's term rather
    # than a length in world units, so it is the thing under test.
    fov, aspect, distance = 45.0, 1.6, 30.0
    mine = _drawn_half_width_fill(pts, radii, right, up, forward, distance,
                                  center=centre, fov=fov, aspect=aspect)
    theirs = fs.drawn_fill(pts, radii, right, up, forward, distance,
                           center=centre, fov=fov, aspect=aspect)
    tan_half = math.tan(math.radians(fov) * 0.5)
    depth = np.maximum(distance + pts[:, 2] - radii, 1e-6)
    vertical = float(((np.abs(pts[:, 1]) + radii) / (depth * tan_half)).max())
    # An equality and not a tolerance: both sides are the same float
    # expression, so a difference is a difference in the code and not in the
    # last bit of a division.
    same = mine == theirs
    axis = vertical < mine
    return (same and axis), (
        f"the derived term {mine!r} against `framing_selection.drawn_fill` "
        f"{theirs!r}, on a scene 16.6 A wide and 1.4 A tall at {distance} A and "
        f"aspect {aspect}: equal = {same}; and the vertical term of the same "
        f"expression is {vertical!r}, so the number being reported is the width's "
        f"= {axis}. The horizontal term is the one carrying the `aspect` factor"
    )


def _view_notice_report(win) -> list[tuple[str, bool, str]]:
    """Is the 3-D view showing the extent it framed, or saying how much of the
    width is off the edge?

    Returns `(name, ok, detail)` triples rather than calling `check` itself, so
    that `_verify_view_notice_checks` can run this exact predicate against a
    deliberately broken viewport. The same reason and the same shape as
    `_panel_visibility_report`: a predicate that only ever runs against a
    correct product is a predicate nobody has watched fail.

    The threshold is the product's own constant and the figure in the sentence
    is the product's own `framed_width_percent`, so neither number in this
    predicate is one this file chose.
    """
    from opendocking.workbench.app import VIEW_MIN_FRAMED_WIDTH_PCT as floor_pct

    vp = win.viewport
    notice = getattr(vp, "view_notice", None)
    if notice is None:
        # **A failure and not a skip**, for the reason the panel's own missing
        # notice is one: deleting `view_notice` from `app.py` would otherwise
        # be invisible here, and a skip reads as "this environment could not
        # answer" when what is true is "the product stopped saying so".
        return [(
            "the 3-D view shows the extent it framed, or says how much of the "
            "width is off the edge",
            False,
            "the viewport has no view_notice to ask: the notice is what makes a "
            "spent view honest, so its absence is the defect this check exists "
            "to find",
        )]
    label = vp._view_notice_label
    pct = vp.framed_width_percent()
    showing = bool(notice.isVisible())
    spent = pct is not None and pct < floor_pct
    where = (f"view {vp.width()}x{vp.height()} px, aspect "
             f"{vp.width() / max(vp.height(), 1):.3f}, half-width fill "
             f"{vp.framed_width_half_fill():.3f}, threshold {floor_pct}%")
    out: list[tuple[str, bool, str]] = []

    # The invariant, in both directions. One direction alone would pass against
    # a notice that is permanently on, which is a different defect, and one of
    # the mutations below is exactly that.
    out.append((
        "the 3-D view shows the extent it framed, or says how much of the "
        "width is off the edge",
        showing is bool(spent),
        (f"the view shows {pct}% of the framed width, under the {floor_pct}% "
         f"floor, and the notice is up"
         if spent else
         f"the view shows "
         f"{pct if pct is not None else 'nothing (no framing yet)'}% and the "
         f"notice is {'showing' if showing else 'hidden'}")
        + f"; {where}",
    ))

    # **The notice's own figure, which is the panel notice's old bug read from
    # the other side.** The panel said "they need 0 px more" while its scroll
    # bar read 1 px, because the sentence and the threshold were two different
    # quantities. Here the failure is the same shape with the other sign: a
    # visible notice that says 0%, or that says a figure the measure does not
    # agree with. Both are refused, and the second is the stricter of the two --
    # the figure has to be the number the threshold was tested against, so the
    # two cannot drift apart even while both are individually plausible.
    #
    # Parsing prose is normally the wrong instrument and is used here anyway,
    # because the alternative is for the product to expose a second number
    # purely so this file can read it, and a second copy of the figure is one
    # more thing to drift. The regex is anchored on the product's own sentence
    # shape, so a reworded sentence shows up as a red here rather than as
    # silence.
    said = re.search(r"shows (\d+)% of the framed width", label.text())
    if not showing:
        out.append((
            "the notice never says the view shows zero percent of the width",
            True,
            "notice is hidden, so there is no figure to be wrong",
        ))
    elif said is None:
        out.append((
            "the notice never says the view shows zero percent of the width",
            False,
            f"the notice is up but its text carries no 'shows N% of the framed "
            f"width' figure to check: {label.text()!r}",
        ))
    else:
        claimed = int(said.group(1))
        out.append((
            "the notice never says the view shows zero percent of the width",
            1 <= claimed < floor_pct and claimed == pct,
            (f"notice up at a {pct}% share, and it says {claimed}%"
             if claimed == pct else
             f"notice up at a {pct}% share, and it says {claimed}% -- the "
             f"sentence and the measure are different numbers"
             + ("; a notice that says 0% while it is up contradicts itself"
                if claimed == 0 else "")),
        ))

    # The card is a widget inside the very view it is warning about, so it can
    # be cut off by the very thing it reports -- the defect the panel's notice
    # had twice, once running off the right-hand edge of a 110 px strip and
    # once with its last line sliced in half.
    top = notice.mapTo(vp, QtCore.QPoint(0, 0))
    rect = QtCore.QRect(top, notice.size())
    inside = vp.rect().contains(rect)
    # The height the label's own text needs at the width it was given, which is
    # the number `_place_view_notice` computed the card's height from.
    needs = int(label.heightForWidth(int(label.width())))
    out.append((
        "the view's notice is inside the view it is warning about, whole",
        (not showing) or (inside and notice.height() >= needs),
        (f"card {rect.x()},{rect.y()} {rect.width()}x{rect.height()} in a "
         f"{vp.width()}x{vp.height()} px view, and its label needs {needs} px "
         f"of height for the {int(label.width())} px it was given"
         if showing else "notice is hidden, so there is nothing to show"),
    ))
    return out


def _spend_the_view_width(app, win, dock, floor_dock):
    """Find the width at which the view's notice comes up, and say by what.

    Returns a dict with the last geometry whose notice was down, the first whose
    notice was up, and the gesture that got there. Both gestures are tried,
    splitter first, because they are the two ways the frame's shape changes
    under a camera that has already been placed.

    **The splitter is walked the way a drag walks it** -- `resizeDocks` from a
    narrow panel outwards -- and not `win.width() - view_minimumWidth()`, which
    is the same arithmetic the panel-notice bug was made of.

    **The wheel is here because the splitter alone does not always reach the
    boundary**, and the reason is a property of the screen rather than of the
    notice. At dpr 2.5 the screen is 768x432 logical and `VIEWPORT_MIN_H` holds
    the view at 480 px tall, so the narrowest view the splitter can reach is
    240 px and the aspect can only fall from the landing framing's 1.435 to
    0.500. Measured on 1CRN with the crambin pose, that is a half-width fill of
    0.588 rising to 1.689 -- a 59% share, which is *over* the 50% floor. The
    boundary is real at that ratio and one wheel notch past it, and saying so
    is the honest reading; a check that only knew how to reach the state by
    dragging would have reported "unreachable" and taught a reader that the
    notice is dead.
    """
    from opendocking.workbench.app import VIEW_MIN_FRAMED_WIDTH_PCT as floor_pct

    vp = win.viewport
    out = {"gesture": None, "down": None, "up": None, "floor_pct": floor_pct,
           "notes": []}
    # `down` and `up` are `(share, view_width, notice_visible)`, so the
    # section can require the notice to be on the side it claims rather
    # than inferring it from the share.

    def settle(turns=2):
        vp.repaint()
        for _ in range(turns):
            app.processEvents()
        vp._update_view_notice()

    # -- the splitter, from the widest geometry outwards to the view's floor
    win.resizeDocks([dock], [60], QtCore.Qt.Orientation.Horizontal)
    settle(4)
    out["down"] = (vp.framed_width_percent(), int(vp.width()),
                    bool(vp.view_notice.isVisible()))
    out["widest"] = (vp.framed_width_percent(), int(vp.width()),
                       bool(vp.view_notice.isVisible()))
    w = int(dock.width())
    while w + 2 <= floor_dock:
        w += 2
        win.resizeDocks([dock], [w], QtCore.Qt.Orientation.Horizontal)
        settle()
        if vp.view_notice.isVisible():
            out["up"] = (vp.framed_width_percent(), int(vp.width()), True)
            out["gesture"] = "splitter"
            break
        out["down"] = (vp.framed_width_percent(), int(vp.width()),
                        bool(vp.view_notice.isVisible()))
    out["notes"].append(
        f"the splitter alone reached the floor at a {int(vp.width())} px view "
        f"({vp.framed_width_percent()}% of the framed width)"
    )

    # -- the wheel, from the widest geometry inwards
    if out["up"] is None:
        win.resizeDocks([dock], [60], QtCore.Qt.Orientation.Horizontal)
        settle(4)
        out["down"] = (vp.framed_width_percent(), int(vp.width()),
                        bool(vp.view_notice.isVisible()))
        for k in range(1, 400):
            vp.camera.distance = max(2.0, float(vp.camera.distance) * 0.97)
            vp._update_view_notice()
            pct = vp.framed_width_percent()
            # **On the notice, not on the measure.** Recording the
            # crossing where the share passed the floor would report a
            # crossing for a product whose notice never comes up, and
            # the check would be green about a notice that is dead.
            # The share is recorded beside it so the section can
            # assert both halves.
            if vp.view_notice.isVisible():
                out["up"] = (pct, int(vp.width()), True)
                out["gesture"] = f"wheel ({k} notches in at 0.97 each)"
                out["camera"] = float(vp.camera.distance)
                break
            out["down"] = (pct, int(vp.width()),
                            bool(vp.view_notice.isVisible()))
        out["notes"].append(
            "the wheel reached it from the same framing, so the boundary is a "
            "property of the measure and not of one gesture"
        )
    return out


def _verify_view_notice_checks(app, win) -> tuple[list[bool], list[str]]:
    """Reverse-verify the view-notice checks: they must fail on a broken view.

    The requirement `_verify_panel_notice_checks` states, for the same reason: a
    guard that cannot go red is indistinguishable from a guard that is working.
    Each case below breaks one thing the product depends on and requires the
    shipped predicate -- the function section 1c calls, not a copy of it -- to
    notice.

    **On the suite's own window, and not on a second one.** A second
    `MainWindow` is a second `QOpenGLWidget` and a second context, and this
    machine has one; see the note on `_verify_panel_notice_checks`. The real
    window is spent, broken six ways, and handed back, and the hand-back is
    itself asserted -- a reverse verification that leaves the suite's subject
    altered would be a different kind of lie.
    """
    global results
    from opendocking.workbench.app import VIEW_MIN_FRAMED_WIDTH_PCT as floor_pct

    saved_results, results = results, []
    verdict: list[bool] = []
    try:
        vp = win.viewport
        notice = vp.view_notice
        label = vp._view_notice_label
        dock = next((d for d in win.findChildren(QtWidgets.QDockWidget)
                     if d.windowTitle() == "Controls"), None)
        was_w, was_h = int(win.width()), int(win.height())
        was_dock = int(dock.width()) if dock is not None else 0
        was_d = float(vp.camera.distance)
        floor_dock = int(win.width()) - int(vp.minimumWidth())

        def _unfix():
            notice.setFixedWidth(16777215)
            notice.setFixedHeight(16777215)
            label.setFixedWidth(16777215)

        def _settle(turns=4):
            vp.repaint()
            for _ in range(turns):
                app.processEvents()
            vp._update_view_notice()
            for _ in range(2):
                app.processEvents()

        def goto_spent():
            """The view's width genuinely spent, by the splitter or the wheel."""
            vp.camera.distance = was_d
            win.resizeDocks([dock], [floor_dock],
                            QtCore.Qt.Orientation.Horizontal)
            _settle()
            if notice.isVisible():
                return "splitter"
            for _ in range(400):
                vp.camera.distance = max(2.0, float(vp.camera.distance) * 0.97)
                vp._update_view_notice()
                if notice.isVisible():
                    break
            _settle(2)
            return "wheel"

        def goto_whole():
            """The whole framed width on screen again, at the panel's narrowest."""
            vp.camera.distance = was_d
            win.resizeDocks([dock], [60], QtCore.Qt.Orientation.Horizontal)
            _settle()

        # Read every figure here, not at the end: this window is widened and
        # zoomed again below, and a figure printed from the live view after
        # that reports the state the test just left rather than the state it
        # tested.
        how = goto_spent()
        spent_view = int(vp.width())
        spent_pct = vp.framed_width_percent()
        spent_share = vp.framed_width_half_fill()
        base = _view_notice_report(win)
        healthy = all(ok for _, ok, _ in base)

        # 1. the notice suppressed while the width is spent: the first must go
        #    red, which is the whole defect this work is about.
        notice.setVisible(False)
        muted = _view_notice_report(win)

        # 2. and shown when the whole width does fit, which is the other
        #    direction: a notice stuck on cannot satisfy case 1.
        goto_whole()
        whole_pct = vp.framed_width_percent()
        notice.setVisible(True)
        label.setText(f"the 3-D view shows {whole_pct}% of the framed width.")
        stuck = _view_notice_report(win)

        # 3. the figure put back the way the panel's was wrong, on this side of
        #    the product: up, and reading 0%. The checks that missed the panel's
        #    version all asked whether the notice was *shown* and none asked
        #    what it said.
        goto_spent()
        _unfix()
        label.setText("the 3-D view shows 0% of the framed width; the rest runs "
                      "off both sides. Scroll to zoom out, or drag the splitter "
                      "back toward the view.")
        notice.setVisible(True)
        lying = _view_notice_report(win)

        # 4. a figure that is plausible and still wrong: twelve points below
        #    the measure. This is the case that catches a sentence built from a
        #    second quantity even when it is nowhere near zero, which is the
        #    more likely way to reintroduce the panel's bug.
        wrong = max(1, (spent_pct or 1) - 12)
        label.setText(f"the 3-D view shows {wrong}% of the framed width; the "
                      f"rest runs off both sides.")
        lying2 = _view_notice_report(win)

        # 5. the card wider than the view it is warning about, and 6. sliced to
        #    a height its own text does not fit in: the third check, twice, and
        #    neither is the half the other covers -- the panel's notice had both
        #    for real.
        _unfix()
        notice.setVisible(True)
        notice.setFixedWidth(int(vp.width()) + 200)
        for _ in range(3):
            app.processEvents()
        stretched = _view_notice_report(win)
        notice.setFixedWidth(16777215)
        notice.setFixedHeight(4)
        for _ in range(3):
            app.processEvents()
        sliced = _view_notice_report(win)

        # 7. and the window must be exactly as this function found it, with the
        #    product left in a state consistent with its own rule. The rule is
        #    the product's own constant, read again here rather than written
        #    out, so this cannot pass against a threshold that moved.
        _unfix()
        goto_whole()
        win.resize(was_w, was_h)
        for _ in range(6):
            app.processEvents()
        vp.camera.distance = was_d
        win.resizeDocks([dock], [was_dock], QtCore.Qt.Orientation.Horizontal)
        _settle()
        back_pct = vp.framed_width_percent()
        restored = (
            int(win.width()) == was_w
            and int(win.height()) == was_h
            and int(dock.width()) == was_dock
            and abs(float(vp.camera.distance) - was_d) <= 1e-9
            and bool(notice.isVisible()) is bool(back_pct is not None
                                                and back_pct < floor_pct)
        )

        verdict.append(healthy)
        verdict.append(not all(ok for _, ok, _ in muted))
        verdict.append(not all(ok for _, ok, _ in stuck))
        verdict.append(not all(ok for _, ok, _ in lying))
        verdict.append(not all(ok for _, ok, _ in lying2))
        verdict.append(not all(ok for _, ok, _ in stretched))
        verdict.append(not all(ok for _, ok, _ in sliced))
        verdict.append(restored)
    finally:
        results = saved_results
    return verdict, [
        f"the suite's own window, panel dragged to {floor_dock} px so the view "
        f"is {spent_view} px wide (reached by the {how}): the view shows "
        f"{spent_pct}% of the framed width, half-width fill {spent_share:.3f}, "
        f"threshold {floor_pct}%, and the notice is up",
        f"the view as shipped there: all three green = {healthy}",
        f"notice suppressed while the width is spent: something red = "
        f"{not all(ok for _, ok, _ in muted)}",
        f"notice shown while the whole width fits ({whole_pct}% on screen, the "
        f"floor is {floor_pct}%): something red = "
        f"{not all(ok for _, ok, _ in stuck)}",
        f"notice text claiming 0% against a {spent_pct}% share: something red = "
        f"{not all(ok for _, ok, _ in lying)}",
        f"notice text claiming {wrong}% against a {spent_pct}% share: something "
        f"red = {not all(ok for _, ok, _ in lying2)}",
        f"card stretched {int(vp.width()) + 200} px wide in a {int(vp.width())} "
        f"px view: something red = {not all(ok for _, ok, _ in stretched)}",
        f"card sliced to 4 px of height: something red = "
        f"{not all(ok for _, ok, _ in sliced)}",
        f"handed back at {win.width()}x{win.height()} (was {was_w}x{was_h}), "
        f"panel {int(dock.width())} px (was {was_dock}), camera "
        f"{float(vp.camera.distance):.2f} A (was {was_d:.2f}), share "
        f"{back_pct}%, notice {'showing' if notice.isVisible() else 'hidden'}: "
        f"the product's own rule holds again = {restored}",
    ]


def main() -> int:
    # `PIXELS_OK` is read by `pixel_check`; assigning it here without this
    # would create a local and silently leave the flag at its default.
    global PIXELS_OK
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    # First line on purpose. Someone reading a truncated run -- a crashed one,
    # a killed one, one that scrolled past -- sees what it was supposed to do
    # before seeing any of the `[PASS]` lines that made it look fine.
    print(
        f"workbench interaction suite: expecting {EXPECTED_CHECKS} checks "
        f"across {len(EXPECTED_SECTIONS)} sections"
    )
    print(f"Qt {QtCore.QT_VERSION_STR}, platform {app.platformName()}")

    rec = EXAMPLES / "rec_prep.pdbqt"
    lig = EXAMPLES / "ibuprofen_prep.pdbqt"
    poses = EXAMPLES / "poses.pdbqt"
    for p in (rec, lig, poses):
        if not p.is_file():
            print(f"missing example input: {p}")
            return 2

    # Before any window of ours exists, because it is one more GL widget and the
    # question is best asked while it is the only one.
    probe_opengl(app)
    print(f"  OpenGL: {GL_VERDICT}")

    win = MainWindow(receptor=rec, ligand=lig, poses=poses)
    win.resize(1280, 820)
    win.show()
    # Named for the census in the keyboard section, which is the one place in
    # this file where a second window can be up beside this one.
    win._gate_name = "win"
    # `isVisible` is not the same question as "has the window been painted".
    # Under a software rasteriser with no compositor the window can be shown
    # and still never get an expose event, which leaves the QOpenGLWidget's
    # framebuffer blank. Asking explicitly turns a mystery into an answer, and
    # it costs nothing where the window really is up.
    exposed = QTest.qWaitForWindowExposed(win, 5000)
    app.processEvents()
    QTest.qWait(600)
    app.processEvents()

    # ------------------------------------------------------------------ layout
    section("1. layout of the control panel")

    # First, and before anything below is read as evidence. This is a statement
    # about how *this file* calls its own `check`, so it is answered from the
    # source and needs no window; but it gates the reading of every line that
    # follows, because a transcript whose checks cannot fail is not a
    # transcript of anything.
    #
    # **The verdict is written as `not miswired` and not as a name**, and that
    # is for the sweep as much as for the reader. `check_verdict_bool.py` reads
    # the element at the verdict parameter's index and types what it finds
    # there: a boolean expression it can decide, a `*` it cannot resolve, or a
    # name it has to trace back to its assignment -- and a name assigned from a
    # tuple it cannot tell *which element of* comes back "undecidable", which
    # the sweep reports as a live red. So a check whose subject is exactly this
    # kind of typing has to hand that gate something it can decide. The
    # unpacking form above is the one `_panel_whole_reachable` is read with.
    _miswired, _unread, _wired_detail = _check_calls_are_wired()
    check(
        "every check() in this file passes a string as its name and a bool as "
        "its verdict, so every one of them is able to fail",
        not _miswired,
        _wired_detail,
    )
    panel = win.dockWidget("Controls") if hasattr(win, "dockWidget") else None
    dock = None
    for i in range(win.dockWidgetArea.count() if False else win.findChildren(QtWidgets.QDockWidget).__len__()):
        dock = win.findChildren(QtWidgets.QDockWidget)[i]
    dock = dock or next(iter(win.findChildren(QtWidgets.QDockWidget)), None)
    print(f"  dock widget found: {dock is not None}  title: {dock.windowTitle() if dock else '-'}")
    dock_w = dock.width() if dock else 0
    print(f"  dock width: {dock_w} px")

    controls = {
        "cb_receptor": win.cb_receptor,
        "cb_ligand": win.cb_ligand,
        "cb_box": win.cb_box,
        "centre x": win.center_spins[0],
        "centre y": win.center_spins[1],
        "centre z": win.center_spins[2],
        "size x": win.size_spins[0],
        "size y": win.size_spins[1],
        "size z": win.size_spins[2],
        "exhaustiveness": win.sp_exhaust,
        "suggest button": win.btn_suggest,
        "search cost": win.lbl_effort,
        "seed": win.sp_seed,
        "scoring": win.cb_scoring,
        "pose table": win.pose_table,
        "all poses": win.cb_all_poses,
        "showing": win.lbl_poses,
        "best energy": win.lbl_energy,
        "RMSD": win.lbl_rmsd,
        "status": win.status_label,
    }
    zero, outside, unreachable, overlap = [], [], [], []
    panel_rect = dock.widget().rect() if dock else QtCore.QRect()
    # **Reachability, not fit.** "No control overflows the panel" is the wrong
    # question once the dock scrolls: a control can be wider than the visible
    # strip and still be perfectly reachable, and the old predicate called that
    # a failure while the real defect -- a control with no way to be brought
    # into view -- went unasked. So the predicate here is the honest one:
    #
    #   a control is reachable when it lies inside the scroll area's *content*
    #   rectangle and the scroll bars have enough range to bring it into the
    #   viewport -- i.e. `content.right() - viewport.width() <= hbar.maximum()`
    #   and likewise vertically.
    #
    # Both halves are needed. A control inside the content but beyond the
    # scrollbar's range is unreachable however wide the range looks, and that
    # is the failure a scrolling strip can still have.
    scroll = getattr(win, "control_scroll", None)
    hbar = scroll.horizontalScrollBar() if scroll is not None else None
    vbar = scroll.verticalScrollBar() if scroll is not None else None
    content = (scroll.widget().rect() if scroll is not None and scroll.widget()
               is not None else panel_rect)
    for name, w in controls.items():
        g = w.geometry()
        if g.width() < 8 or g.height() < 8:
            zero.append(f"{name}{g.width()}x{g.height()}")
        if dock is None:
            continue
        # The control's rect in the scroll area's content coordinates, which
        # is where a scroll position is expressed.
        if scroll is not None:
            top = w.mapTo(scroll.widget(), QtCore.QPoint(0, 0))
        else:
            top = w.mapTo(dock.widget(), QtCore.QPoint(0, 0))
        r = QtCore.QRect(top, g.size())
        # Inside the content it is scrollable to?
        if not content.contains(r):
            unreachable.append(
                f"{name} at {r.x()},{r.y()} {r.width()}x{r.height()} is outside "
                f"the scrollable content {content.width()}x{content.height()}")
            continue
        # And can a scroll position actually bring it into the viewport?
        if scroll is not None:
            hmax, vmax = int(hbar.maximum()), int(vbar.maximum())
            need_h = max(0, r.right() - int(scroll.viewport().width()))
            need_v = max(0, r.bottom() - int(scroll.viewport().height()))
            if need_h > hmax or need_v > vmax:
                unreachable.append(
                    f"{name} needs {need_h} px of horizontal scroll and the "
                    f"bar reaches {hmax}")
    for i, (n1, w1) in enumerate(controls.items()):
        for n2, w2 in list(controls.items())[i + 1:]:
            if w1.parentWidget() is w2.parentWidget() and rects_overlap(w1.geometry(), w2.geometry()):
                overlap.append(f"{n1} / {n2}")

    check("every control has a usable size", not zero, ", ".join(zero) or "all >= 8x8")
    check("every control in the panel is reachable", not unreachable,
          "; ".join(unreachable) or
          f"all {len(controls)} controls inside the scrollable content, and "
          f"the scroll bars reach"
          + (f" {int(hbar.maximum())} px across, {int(vbar.maximum())} px down"
             if scroll is not None else ""))
    check("no two controls overlap", not overlap, "; ".join(overlap) or "none")

    # ------------------------------------------------------------------
    # Is the panel *usable*, which is a different question from reachable?
    #
    # The check above is about reachability and it is right to be: at dpr 2.5
    # the horizontal bar reaches 385 px, so every control in this panel can be
    # scrolled to, and 351 checks passed over a panel that showed a 110 px
    # slice of its own row labels and not one control beside them. Reachability
    # cannot see that, because scrolling *is* the fix for it. What it cannot
    # see is whether the product admits the strip is a strip.
    #
    # So these three ask the other question, and all three are measured at
    # whichever ratio this run is at -- the same file, the same window size,
    # the two ratios differing only by the device pixel ratio the screen
    # reports. A check that only ever ran at 1.25 would pass here forever.
    #
    # Every threshold below is either Qt's own live answer (`maximum() > 0`)
    # or a rectangle-containment identity. There is no width constant to tune:
    # a number here would be one I picked to make the check pass.
    for _name, _ok, _detail in _panel_visibility_report(win, scroll, hbar):
        check(_name, _ok, _detail)

    # The question the three above cannot ask. All three describe the panel as
    # it is *now*; this one asks whether the user can change that by dragging,
    # which is the only thing a minimum on the 3-D view was deciding. It is
    # placed here rather than in the reverse verification below because it is a
    # statement about the shipped product, and the reverse verification is a
    # statement about the check.
    check(
        "the user can drag the control panel to a width where every control is "
        "on screen",
        *_panel_whole_reachable(app, win, dock, scroll, hbar),
    )

    if scroll is not None:
        print(f"  dock scrolls: content {content.width()}x{content.height()} px, "
              f"viewport {scroll.viewport().width()}x{scroll.viewport().height()} px, "
              f"h-bar max {int(hbar.maximum())}, v-bar max {int(vbar.maximum())}; "
              f"horizontal policy {scroll.horizontalScrollBarPolicy().name}")
    print(f"  viewport: {win.viewport.width()}x{win.viewport.height()} "
          f"(min {win.viewport.minimumWidth()}x{win.viewport.minimumHeight()})")
    print(f"  status text: {win.status_label.text()!r}")
    print(f"  energy label: {win.lbl_energy.text()!r}   rmsd: {win.lbl_rmsd.text()!r}")
    check(
        "best-pose energy is populated on load",
        win.lbl_energy.text() not in ("", "—"),
        win.lbl_energy.text(),
    )
    check(
        "RMSD-to-best is populated on load",
        win.lbl_rmsd.text() not in ("", "—"),
        win.lbl_rmsd.text(),
    )
    check(
        "pose list is populated",
        win.pose_list.count() > 1,
        f"{win.pose_list.count()} rows",
    )

    base, _ = shot(win, "01_loaded")
    base_px = non_background(base)
    # The machine's answer, from the product's own probe, counted like any other
    # result. A skip here is the honest reading of "this box cannot render":
    # it says nothing about the viewport, and it is CI's `xvfb-run` that
    # answers the question on a machine that can.
    #
    # The two branches name *different things on purpose*. A single name that
    # asserted the outcome -- "this machine can give Qt an OpenGL context" --
    # produced a line reading "can" next to a reason reading "cannot", because
    # a skip is a name plus a reason and only the second half followed the
    # truth. The skip branch now names the question instead of its answer.
    if GL_OK:
        check(
            "this machine can give Qt an OpenGL context, so the pixel checks "
            "below are real",
            True,
            GL_VERDICT or "",
        )
    else:
        skip(
            "whether the launcher's own OpenGL probe could give Qt a context",
            GL_VERDICT or "no OpenGL context",
        )
    # Decide once, here, whether the framebuffer is readable at all. A blank
    # or missing grab means every pixel-dependent check below would be
    # measuring the environment rather than the viewport.
    #
    # Note this is decided by the *product's* framebuffer, not by `GL_OK`.
    # Those disagree, and the disagreement is measured rather than assumed: on
    # the machine this was written on, the launcher's probe reports no context
    # (`GL_OK` False) while the workbench's own `QOpenGLWidget` renders and its
    # framebuffer reads back thousands of pixels. The probe builds a bare
    # `QOpenGLContext`; the product is a `QOpenGLWidget` with a real FBO, and
    # the second is what a viewer sees. Gating the pixel checks on the probe
    # would skip them on machines that render perfectly well, so they are gated
    # on the product's own frame instead.
    if base_px == 0:
        PIXELS_OK = False
        print(
            "\n  NOTE: no frame could be read from the viewport, so the pixel"
            "\n        checks below are skipped rather than failed."
            f"\n        platform: {app.platformName()}; window exposed: {exposed}."
            "\n        Every non-pixel check still runs, and this one is counted"
            "\n        in the summary as a skip."
        )
    # Counted, not just printed, and registered in **both** branches. This used
    # to exist only in the `base_px == 0` branch, which made the total move
    # with the environment: 243 on a machine that could read its framebuffer and
    # 244 on one that could not, on a commit whose pin said 243. A pin that only
    # holds where the picture is readable is not a pin. One result either way.
    if PIXELS_OK:
        check(
            "the viewport's frame could be read",
            True,
            f"{base_px} non-background px in the first grab",
        )
    else:
        skip(
            "the viewport's frame could be read",
            SHOT_PROBLEM or "grabFramebuffer returned a uniformly coloured image",
        )
    pixel_check("viewport renders geometry", base_px > 2000, f"{base_px} non-background px")
    check(*_verify_pixel_guard())

    # The narrow-panel checks, reverse-verified the same way: the predicate
    # section 1 just ran has to be able to go red. Runs on its own short-lived
    # window at 700 px, so it cannot be satisfied by the geometry of whatever
    # size the main window happens to be at this ratio -- which is the whole
    # reason the three above needed proving rather than assuming.
    _verdict, _lines = _verify_panel_notice_checks(app, win)
    check(
        "the narrow-panel checks can fail: they go red on a panel that is cut "
        "off with nothing saying so, on a notice wider than the strip, on a "
        "notice sliced shorter than its own text, on a 3-D view whose floor "
        "puts the whole panel out of the splitter's reach, and on a notice "
        "whose own figure contradicts the scroll bar",
        all(_verdict),
        "; ".join(_lines),
    )

    # -------------------------------------------------- element interpretation
    section("1b. does the viewer understand the atom types it is given?")
    from opendocking.workbench import ELEMENT_COLORS, PDBQT_TYPE_ELEMENT

    for mol in win.viewport.molecules:
        unknown = sorted({e for e in mol.elements if e not in ELEMENT_COLORS})
        check(
            f"every element in '{mol.name}' is a real element",
            not unknown,
            f"{len(mol.elements)} atoms, unknown: {unknown or 'none'}"
            + (f"  <-- the aromatic-carbon token 'A' was read as an element" if "A" in unknown else ""),
        )
    pose_view = next(m for m in win.viewport.molecules if m.role == "pose")
    lig_view = next(m for m in win.viewport.molecules if m.role == "ligand")
    check(
        "the pose keeps the same element composition as the ligand it came from",
        sorted(pose_view.elements) == sorted(lig_view.elements),
        f"pose {dict((e, pose_view.elements.count(e)) for e in set(pose_view.elements))} "
        f"vs ligand {dict((e, lig_view.elements.count(e)) for e in set(lig_view.elements))}",
    )
    radii = pose_view.atom_radii()
    hydrogens = [i for i, e in enumerate(pose_view.elements) if e == "H"]
    carbons = [i for i, e in enumerate(pose_view.elements) if e in ("C", "A")]
    if hydrogens and carbons:
        check(
            "hydrogens are drawn smaller than carbons",
            max(radii[i] for i in hydrogens) < min(radii[i] for i in carbons),
            f"H radius {radii[hydrogens[0]]:.3f} vs C radius {radii[carbons[0]]:.3f} A"
            + ("" if max(radii[i] for i in hydrogens) < min(radii[i] for i in carbons) else "  <-- same size"),
        )
    else:
        skip(
            "hydrogens are drawn smaller than carbons",
            f"the pose has {len(hydrogens)} H and {len(carbons)} C atoms, so "
            "there are no radii of both to compare",
        )
    check(
        "AutoDock 'A' is mapped to carbon",
        PDBQT_TYPE_ELEMENT.get("A") == "C" and PDBQT_TYPE_ELEMENT.get("OA") == "O"
        and PDBQT_TYPE_ELEMENT.get("HD") == "H" and PDBQT_TYPE_ELEMENT.get("ZN") == "Zn",
        "A->C, OA->O, HD->H, ZN->Zn",
    )

    # ------------------------------------------ 1c. the view's own notice
    # The panel's notice answers "is any of this off the right-hand edge".
    # This asks the view's question, which is a different one: the 3-D view is
    # a central widget whose width the reader may spend on the panel, the
    # camera was placed for the aspect it was given, and narrowing the frame
    # does not re-solve it -- so a frame that was whole becomes a slice of
    # itself. Measured on 1CRN with the crambin pose, the share of the framed
    # width on screen runs 100% at a 1417 px view down to 35% at the 240 px
    # floor, and the boundary is at a 334 px view with the notice down and
    # 333 px with it up.
    section("1c. does the 3-D view say when its width is spent?")

    # The measure first, with no window involved: the notice is only honest if
    # the quantity it reports is the one `framing_selection` already computes,
    # and that is a claim about arithmetic rather than about behaviour.
    #
    # **The name is written out here and the helper's two-element return is
    # spread behind it**, which is the form `_panel_whole_reachable` is called
    # with above. It used to be `check(*_half_width_fill_is_the_modules_own_term())`,
    # and a two-element `(bool, str)` spread into a three-parameter `check`
    # puts the bool in `name` and the 354-character explanation in `ok`: a
    # non-empty string is truthy, so the check reported `PASS` for every input
    # and a failing verdict printed as the check's own name. The helper still
    # returns `(bool, str)` -- three of the five other spreads in this file
    # return `(str, bool, str)` and are right, which is what made this one read
    # as correct -- and its arity is unchanged because nothing else wants it.
    check(
        "the derived half-width term is the one `framing_selection.drawn_fill` "
        "reports, and is the width's own on a scene three times as wide as it "
        "is tall",
        *_half_width_fill_is_the_modules_own_term(),
    )

    # The same predicate at both ends of the range, because one direction alone
    # passes against a notice that is permanently on. The landing geometry is
    # the one the window opens in: the whole framed width on screen, and
    # nothing said.
    for _name, _ok, _detail in _view_notice_report(win):
        check(_name, _ok, _detail)

    _vdock = next((d for d in win.findChildren(QtWidgets.QDockWidget)
                   if d.windowTitle() == "Controls"), None)
    _vfloor = int(win.width()) - int(win.viewport.minimumWidth())
    _vwas_d = float(win.viewport.camera.distance)
    _vwas_w, _vwas_h = int(win.width()), int(win.height())
    _vwas_dock = int(_vdock.width()) if _vdock is not None else 0
    try:
        # And at the narrow end, where the reader has spent the view on the
        # panel. The window is handed back in the `finally`, because a check
        # that leaves its subject resized is a different kind of lie, and every
        # section after this one drives this same window.
        win.resizeDocks([_vdock], [_vfloor], QtCore.Qt.Orientation.Horizontal)
        for _ in range(6):
            app.processEvents()
        win.viewport._update_view_notice()
        for _ in range(3):
            app.processEvents()
        if not win.viewport.view_notice.isVisible():
            # The wheel is the other way to spend the width, and at some ratios
            # on some screens it is the only one -- see `_spend_the_view_width`.
            for _ in range(400):
                cam = win.viewport.camera
                cam.distance = max(2.0, float(cam.distance) * 0.97)
                win.viewport._update_view_notice()
                if win.viewport.view_notice.isVisible():
                    break
        print(f"  the view spent: {win.viewport.width()}x{win.viewport.height()} "
              f"px, share {win.viewport.framed_width_percent()}%, half-width "
              f"fill {win.viewport.framed_width_half_fill():.3f}")
        for _name, _ok, _detail in _view_notice_report(win):
            check(_name, _ok, _detail)

        # And the threshold is what separates the two, by a gesture rather than
        # by arithmetic: the two geometries either side of the crossing, read
        # off the product's own measure.
        _cross = _spend_the_view_width(app, win, _vdock, _vfloor)
        _down, _up = _cross["down"], _cross["up"]
        _ok_cross = (
            _up is not None
            and _down is not None
            and _down[0] is not None and _up[0] is not None
            and _down[0] >= _cross["floor_pct"]
            and _up[0] < _cross["floor_pct"]
            # And the notice is on the side each record claims, which is
            # a different demand from the shares straddling the floor: a
            # product that never shows the notice has shares that cross
            # and nothing else, and a check made of shares alone would
            # be green about a notice that is dead.
            and _up[2] is True
            and _down[2] is False
        )
        check(
            "the threshold is the width at which the notice changes, and both "
            "of its sides are reachable in this window",
            _ok_cross,
            (f"reached by the {_cross['gesture']}: the notice is "
             f"{'down' if (_down and not _down[2]) else 'UP'} at a "
             f"{_down[1] if _down else '?'} px view showing "
             f"{_down[0] if _down else '?'}% and "
             f"{'up' if (_up and _up[2]) else 'DOWN'} at a "
             f"{_up[1] if _up else '?'} px view showing "
             f"{_up[0] if _up else '?'}%, against a "
             f"{_cross['floor_pct']}% floor; "
             + "; ".join(_cross["notes"])
             if _ok_cross else
             f"no crossing was found by either gesture, or the two sides do "
             f"not straddle the floor: down {_down}, up {_up}, floor "
             f"{_cross['floor_pct']}%; " + "; ".join(_cross["notes"])),
        )
    finally:
        win.viewport.camera.distance = _vwas_d
        win.resizeDocks([_vdock], [_vwas_dock], QtCore.Qt.Orientation.Horizontal)
        win.resize(_vwas_w, _vwas_h)
        for _ in range(8):
            app.processEvents()
        win.viewport._update_view_notice()
        for _ in range(3):
            app.processEvents()

    _vverdict, _vlines = _verify_view_notice_checks(app, win)
    check(
        "the view-notice checks can fail: they go red on a spent view with "
        "nothing saying so, on a notice up when the whole width fits, on a "
        "notice claiming 0%, on one claiming a figure the measure does not "
        "agree with, on a card wider than the view, and on a card sliced "
        "shorter than its own text",
        all(_vverdict),
        "; ".join(_vlines),
    )

    # ------------------------------------------------------------ interaction
    section("2. camera interaction")
    cam = win.viewport.camera
    y0, p0, d0 = cam.yaw, cam.pitch, cam.distance

    drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (520, 300))
    check("left-drag rotates the camera", abs(cam.yaw - y0) > 1e-6, f"yaw {y0:.4f} -> {cam.yaw:.4f}")

    drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (400, 420))
    check(
        "left-drag changes pitch and clamps it",
        cam.pitch != p0 and -1.5 <= cam.pitch <= 1.5,
        f"pitch {p0:.4f} -> {cam.pitch:.4f}",
    )
    # push far past the clamp to prove it holds
    for _ in range(12):
        drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 300), (400, 900), steps=2)
    check("pitch clamp holds at +1.5", abs(cam.pitch - 1.5) < 1e-9, f"pitch {cam.pitch:.6f}")
    for _ in range(20):
        drag(win.viewport, QtCore.Qt.MouseButton.LeftButton, (400, 900), (400, 200), steps=2)
    check("pitch clamp holds at -1.5", abs(cam.pitch + 1.5) < 1e-9, f"pitch {cam.pitch:.6f}")

    before = cam.distance
    QTest.mouseMove(win.viewport, QtCore.QPoint(400, 300))
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(400, 300),
        QtCore.QPointF(400, 300),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    app.sendEvent(win.viewport, wheel)
    app.processEvents()
    check("wheel zooms in", cam.distance < before, f"distance {before:.3f} -> {cam.distance:.3f}")

    before = cam.distance
    win.viewport.frame_all()
    app.processEvents()
    check(
        "Frame all refits the camera",
        abs(cam.distance - before) > 1e-6 or not np.allclose(cam.center, win.viewport.camera.center),
        f"distance -> {cam.distance:.3f}, centre {np.round(cam.center, 2)}",
    )

    # right-drag: is there any pan at all?
    section("3. is there a pan control?")
    c0 = cam.center.copy()
    drag(win.viewport, QtCore.Qt.MouseButton.RightButton, (400, 300), (600, 380))
    moved = float(np.abs(cam.center - c0).max())
    check(
        "right-drag pans the camera",
        moved > 1e-6,
        f"centre moved {moved:.6f} A"
        + ("" if moved > 1e-6 else "  <-- no pan is implemented"),
    )
    m0 = cam.distance
    drag(win.viewport, QtCore.Qt.MouseButton.MiddleButton, (400, 300), (400, 380))
    check("middle-drag changes distance", abs(cam.distance - m0) > 1e-9,
          f"distance {m0:.3f} -> {cam.distance:.3f}")

    # ------------------------------------------------------------- visibility
    section("4. visibility checkboxes")
    win.viewport.frame_all()
    app.processEvents()
    full, _ = shot(win, "02_all_visible")

    win.cb_receptor.setChecked(False)
    app.processEvents()
    no_rec, _ = shot(win, "03_receptor_hidden")
    pixel_check(
        "unchecking 'receptor' removes pixels",
        non_background(no_rec) < non_background(full),
        f"{non_background(full)} -> {non_background(no_rec)} px",
    )
    win.cb_receptor.setChecked(True)
    app.processEvents()

    win.cb_ligand.setChecked(False)
    app.processEvents()
    no_lig, _ = shot(win, "04_pose_hidden")
    pixel_check(
        "unchecking 'ligand / pose' removes pixels",
        non_background(no_lig) < non_background(full),
        f"{non_background(full)} -> {non_background(no_lig)} px",
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()

    win.cb_box.setChecked(False)
    app.processEvents()
    no_box, _ = shot(win, "05_box_hidden")
    pixel_check(
        "unchecking 'search box' removes the wireframe",
        non_background(no_box) < non_background(full),
        f"{non_background(full)} -> {non_background(no_box)} px",
    )
    win.cb_box.setChecked(True)
    app.processEvents()

    section("5. search-box spins")
    c_before = np.array(win.viewport.box_center, copy=True)
    win.center_spins[0].setValue(9.0)
    app.processEvents()
    c_after = np.array(win.viewport.box_center, copy=True)
    check(
        "box centre spin moves the wireframe",
        abs(c_after[0] - c_before[0]) > 1e-6,
        f"box_center x {c_before[0]:.2f} -> {c_after[0]:.2f}",
    )
    win.center_spins[0].setValue(0.0)
    app.processEvents()

    s_before = float(win.viewport.box_size[0])
    win.size_spins[0].setValue(12.0)
    app.processEvents()
    check(
        "box size spin resizes the wireframe",
        abs(float(win.viewport.box_size[0]) - s_before) > 1e-6,
        f"box_size x {s_before:.2f} -> {win.viewport.box_size[0]:.2f}",
    )
    win.size_spins[0].setValue(22.0)
    app.processEvents()

    # range check: can the user type a negative centre?
    lo = win.center_spins[0].minimum()
    hi = win.center_spins[0].maximum()
    win.center_spins[0].setValue(-5.0)
    app.processEvents()
    got = win.center_spins[0].value()
    check(
        "box centre accepts negative coordinates",
        abs(got - (-5.0)) < 1e-6,
        f"range [{lo}, {hi}], asked for -5.0, got {got}  <-- clamped",
    )
    win.center_spins[0].setValue(0.0)
    app.processEvents()

    # ------------------------------------------------- effort follows the box
    section("5a. the search effort follows the box")
    # Four defects in the workbench's half of the exhaustiveness rule were all
    # invisible to this suite: dropping the `_updating_exhaust` guard, making
    # `_suggest_exhaustiveness` return at once, breaking the latch so it never
    # stops, and hardcoding the wanted value to 8. Every one of them left 135 of
    # 135 passing, because `core.exhaustiveness_for_box` is covered elsewhere
    # and nothing here ever asked the *widget* a question. So the checks below
    # drive the live spin boxes and read the widget back.
    #
    # Its own window, because the latch is permanent by design: setting
    # exhaustiveness here would stop the main window's box from driving the
    # number, and section 7 needs that window to dock at the value it asks for.
    ex = MainWindow()
    ex.resize(1500, 900)
    ex.show()
    try:
        # The note has to answer before a receptor is loaded, and saying so is a
        # different claim from showing a number nobody measured.
        for spin in ex.size_spins:
            spin.setValue(20.0)
        app.processEvents()
        check(
            "with no receptor the map estimate admits it cannot be given",
            "MB" not in ex.lbl_effort.text() and "maps" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        ex.load_structure(EXAMPLES / "1crn_prep.pdbqt", "receptor")
        app.processEvents()
        check(
            "a loaded receptor puts a map estimate on screen",
            "MB" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        def _box(side: float) -> bool:
            """Put a cube on all three size spins; True if the box really moved.

            The "did it move" answer is not bookkeeping. `setValue` to the value
            a spin already holds emits nothing, so a step that asks for the box
            it already has tests nothing at all -- and it passes for a build
            whose suggestion is dead, because "the number did not change" is
            exactly what a dead suggestion looks like.
            """
            before = tuple(float(s.value()) for s in ex.size_spins)
            for spin in ex.size_spins:
                spin.setValue(side)
            app.processEvents()
            after = tuple(float(s.value()) for s in ex.size_spins)
            return after != before

        from opendocking.core import (  # noqa: PLC0415
            EXHAUSTIVENESS_LADDER,
            REFERENCE_BOX_SIDE,
            GridBox,
            Receptor as _Receptor,
            exhaustiveness_for_box,
        )

        # Grow the box three times, not once. The guard's failure mode is a
        # suggestion that applies once and then latches itself, so a single step
        # looks like a working feature right up until the second one.
        seen = []
        moved_all = True
        for side in (30.0, 40.0, 60.0):
            moved_all &= _box(side)
            seen.append((side, int(ex.sp_exhaust.value())))
        wanted = [int(exhaustiveness_for_box((s, s, s))) for s, _ in seen]
        check(
            "enlarging the box raises exhaustiveness with it",
            moved_all and all(got == want for (_, got), want in zip(seen, wanted)),
            "; ".join(
                f"{side:.0f} A -> {got} (rule says {want})"
                for (side, got), want in zip(seen, wanted)
            )
            + f"; every step really moved the box: {moved_all}",
        )
        check(
            "the spin box is not stuck on one value while the box grows",
            len({got for _, got in seen}) > 1,
            f"values seen: {[got for _, got in seen]}",
        )

        # Shrinking back has to come back down, or the control would be a ratchet
        # dressed up as a rule.
        moved_down = _box(20.0)
        check(
            "shrinking the box lowers exhaustiveness again",
            moved_down
            and int(ex.sp_exhaust.value()) == int(exhaustiveness_for_box((20.0,) * 3)),
            f"back to 20 A (moved: {moved_down}) -> {ex.sp_exhaust.value()} "
            f"(rule says {exhaustiveness_for_box((20.0,) * 3)})",
        )

        # The visible text has to follow the number, in both directions, and it
        # has to say whose number it is. A tooltip that never changes is a
        # tooltip saying nothing.
        tip_big = ex.sp_exhaust.toolTip()
        note_big = ex.lbl_effort.text()
        _box(60.0)
        check(
            "the hint text changes when the box changes",
            ex.sp_exhaust.toolTip() != tip_big and ex.lbl_effort.text() != note_big,
            f"tooltip 20A->60A: {tip_big[:34]!r} -> {ex.sp_exhaust.toolTip()[:34]!r}; "
            f"note 20A->60A: {note_big!r} -> {ex.lbl_effort.text()!r}",
        )
        check(
            "the note names the walks it is talking about",
            f"{int(ex.sp_exhaust.value())} walks" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        # A hand edit has to end the following. This is the load-bearing
        # negative: without it, "the spin box follows the box" is also what a
        # control that ignores you looks like.
        #
        # A value that is *not* the one already showing. `setValue` to the
        # current value emits nothing, so a check that "types" the number the
        # rule had already chosen is not typing anything: no `valueChanged`, no
        # latch, and every assertion below would pass against a build whose
        # latch was deleted.
        typed = 24
        ex.sp_exhaust.setValue(typed)
        app.processEvents()
        check(
            "the hand edit is a change, so the control really was touched",
            int(ex.sp_exhaust.value()) == typed,
            f"asked for {typed}, spin box shows {ex.sp_exhaust.value()}",
        )
        after_typing = ex.sp_exhaust.toolTip()
        check(
            "a hand-typed value is claimed by the user, in words",
            "Set by you" in after_typing,
            f"{after_typing[:48]!r}",
        )
        _box(60.0)
        moved_down = _box(20.0)
        check(
            "after a hand edit the box stops driving the number, shrinking",
            moved_down and int(ex.sp_exhaust.value()) == typed,
            f"typed {typed}, box 60A -> 20 A (moved: {moved_down}), spin box now "
            f"{ex.sp_exhaust.value()}",
        )
        moved_up = _box(60.0)
        check(
            "after a hand edit the box stops driving the number, growing",
            moved_up and int(ex.sp_exhaust.value()) == typed,
            f"typed {typed}, box 20A -> 60 A (moved: {moved_up}), spin box now "
            f"{ex.sp_exhaust.value()}",
        )
        moved_back = _box(20.0)
        check(
            "and it stays stopped when the box comes back down",
            moved_back and int(ex.sp_exhaust.value()) == typed,
            f"box back to 20 A (moved: {moved_back}), still {ex.sp_exhaust.value()}",
        )
        check(
            "the note says the number is the user's while it is",
            "set by you" in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        # The way back has to exist, and it has to be a press. If the suggestion
        # came back by itself the latch would be a suggestion the user cannot
        # decline, which is the failure the latch was built to avoid.
        check(
            "there is a control that hands the number back to the rule",
            ex.btn_suggest.isVisible() and ex.btn_suggest.isEnabled(),
            f"{ex.btn_suggest.text()!r} "
            f"{ex.btn_suggest.width()}x{ex.btn_suggest.height()} px",
        )
        ex.btn_suggest.click()
        app.processEvents()
        check(
            "pressing it re-arms the rule immediately",
            "suggested" in ex.lbl_effort.text()
            and "Set by you" not in ex.sp_exhaust.toolTip(),
            f"note {ex.lbl_effort.text()!r}, tooltip "
            f"{ex.sp_exhaust.toolTip()[:34]!r}",
        )
        after_rearm = []
        for side in (60.0, 40.0, 20.0):
            _box(side)
            after_rearm.append((side, int(ex.sp_exhaust.value())))
        check(
            "and the box drives the number again from then on",
            all(got == int(exhaustiveness_for_box((s,) * 3)) for s, got in after_rearm)
            and any(got != typed for _, got in after_rearm),
            "; ".join(f"{s:.0f} A -> {g}" for s, g in after_rearm),
        )

        # The cap. Above 64,000 A^3 the ladder's top rung is below what the rule
        # itself wants, and the note has to say so rather than print 128 as
        # though it were met. The percentage is derived here from `core`, not
        # read out of the label, so the check is about the arithmetic and not
        # about the wording.
        for side in (60.0, 100.0):
            _box(side)
            vol = side ** 3
            want = EXHAUSTIVENESS_LADDER[0] * (vol / REFERENCE_BOX_SIDE ** 3)
            got = int(ex.sp_exhaust.value())
            check(
                f"a {side:.0f} A cube is reported as short of what it wants",
                got == EXHAUSTIVENESS_LADDER[-1]
                and f"{100.0 * got / want:.0f}%" in ex.lbl_effort.text(),
                f"wants {want:.0f} walks, ladder gave {got}, note says "
                f"{100.0 * got / want:.0f}%: {ex.lbl_effort.text()!r}",
            )
        _box(40.0)
        check(
            "a box inside the ladder is not accused of falling short",
            "ladder stops here" not in ex.lbl_effort.text()
            and "% of the" not in ex.lbl_effort.text(),
            f"{ex.lbl_effort.text()!r}",
        )

        # The workbench's half of the rule, asked directly. `_walks_wanted`
        # used to re-derive the density here in the panel, importing the
        # engine's two calibration constants and calling that a guarantee
        # against drift. It was a guarantee about the slope and the reference
        # and nothing at all about the cap, the rung choice, or the refusal to
        # answer a box the engine cannot build. The cap is what the note names
        # when it says "the ladder stops here", so it is checked against the
        # engine rather than against this file's own reading of the ladder.
        pairs = [(s, _walks_wanted((s, s, s))) for s in (20.0, 26.0, 40.0, 60.0, 200.0)]
        check(
            "the ceiling the note compares against is the engine's own answer",
            all(ceiling == float(exhaustiveness_for_box((s,) * 3)) for s, (_, ceiling) in pairs),
            "; ".join(
                f"{s:.0f} A -> {ceiling:.0f} (engine "
                f"{exhaustiveness_for_box((s,) * 3)})"
                for s, (_, ceiling) in pairs
            ),
        )
        check(
            "and the requirement is still the unsaturated one, above the cap for a big box",
            all(
                (want > ceiling) == (want > EXHAUSTIVENESS_LADDER[-1] + 1e-9)
                for _, (want, ceiling) in pairs
            )
            and _walks_wanted((60.0,) * 3)[0] == 216.0
            and _walks_wanted((26.0,) * 3)[0] < _walks_wanted((26.0,) * 3)[1],
            "; ".join(
                f"{s:.0f} A wants {want:.0f} against a ceiling of {ceiling:.0f}"
                f"{'  <- past the top rung' if want > ceiling else ''}"
                for s, (want, ceiling) in pairs
            )
            + ". The requirement is not the function's return value and cannot "
            "be: the function rounds up to a rung, so it cannot hand back a "
            "figure between two rungs, and 216 against 128 is the whole point.",
        )
        refused = []
        for bad in ((20.0, float("nan"), 20.0), (20.0, 20.0)):
            try:
                _walks_wanted(bad)
                refused.append(f"{bad} -> ANSWERED")
            except ValueError as exc:
                refused.append(f"{bad} -> refused ({str(exc)[:44]}...)")
        check(
            "a box the engine refuses is refused here too, not formatted into a note",
            len(refused) == 2 and "ANSWERED" not in " ".join(refused),
            "; ".join(refused),
        )

        # The map number itself, against the engine and not against the panel.
        receptor = _Receptor.from_pdbqt(EXAMPLES / "1crn_prep.pdbqt")
        centre = tuple(float(s.value()) for s in ex.center_spins)
        for side in (20.0, 40.0, 60.0, 100.0):
            _box(side)
            engine_mb = receptor.estimate_memory_mb(
                GridBox.from_center_size(centre, (side,) * 3)
            )
            shown = ex.lbl_effort.text().split(" MB")[0].split()[-1]
            check(
                f"the map estimate for a {side:.0f} A cube is the engine's",
                abs(float(shown) - engine_mb) <= max(1.0, engine_mb * 0.01),
                f"label {shown} MB, engine {engine_mb:.1f} MB",
            )
        _box(60.0)
        note_lab = ex.lbl_effort
        need = note_lab.heightForWidth(max(note_lab.width(), 1))
        check(
            "the note is tall enough for the line it grew",
            note_lab.height() >= need - 1,
            f"height {note_lab.height()} px, needs {need} px, "
            f"{len(note_lab.text().splitlines())} lines of text",
        )
    except Exception:
        traceback.print_exc()
        check("the exhaustiveness section ran to the end", False, "raised above")
    finally:
        dispose_window(app, ex)

    # ----------------------------------------------------------------- poses
    # The File menu is the only route into the app once it is open, so it has to
    # offer every role. It used to have a single "Open structure…" that loaded
    # everything as a ligand, which made a receptor unreachable from the GUI.
    section("5b. the File menu can reach every role")
    menu = win.menuBar().actions()[0].menu()
    labels = [a.text() for a in menu.actions() if a.text() and a.text() != "---"]
    print(f"  File menu: {labels}")
    for role, needle in (("receptor", "Load receptor"), ("ligand", "Load ligand"),
                         ("poses", "Open poses"), ("dock", "Dock now")):
        check(
            f"the File menu offers '{needle}'",
            any(needle in t for t in labels),
            ", ".join(labels),
        )
    check(
        "no action still says the generic 'Open structure'",
        not any("Open structure" in t for t in labels),
        ", ".join(labels),
    )
    check(
        "opening a pose file goes through load_structure, not a crash",
        callable(win.open_file_dialog) and "kind" in win.open_file_dialog.__code__.co_varnames,
        f"signature accepts a role: {win.open_file_dialog.__code__.co_varnames[:3]}",
    )

    # A pose file with one field shifted by a column. `read_pdbqt_models`
    # refuses these by default -- correctly, because the engine's reader splits
    # on whitespace and would load the file and report the wrong coordinates
    # without saying anything -- and that refusal used to arrive at the user as
    # a traceback out of a GUI handler. Nothing here exercised a malformed
    # pose file, which is why it landed on a person rather than on a test.
    n = win.pose_list.count()
    good_rows = (EXAMPLES / "poses.pdbqt").read_text(encoding="utf-8").splitlines()
    shifted = None
    mangled = []
    for number, line in enumerate(good_rows, start=1):
        if shifted is None and line.startswith("ATOM") and len(line) > 34:
            # Delete the space at column 31: every coordinate on this line
            # slides one column left, which is the exact edit the engine's
            # whitespace-splitting reader cannot see.
            mangled.append(line[:30] + line[31:])
            shifted = number
        else:
            mangled.append(line)
    bad_path = Path(tempfile.gettempdir()) / "od_misaligned_pose.pdbqt"
    bad_path.write_text("\n".join(mangled) + "\n", encoding="utf-8")
    raised = None
    try:
        win.load_structure(bad_path, "poses")
    except Exception as exc:  # noqa: BLE001 - that it is *any* exception is the claim
        raised = exc
    msg = win.status_label.text()
    print(f"  misaligned file says: {msg!r}")
    check(
        "a pose file with a shifted column does not raise out of the GUI",
        raised is None,
        f"{type(raised).__name__}: {raised}" if raised else
        f"line {shifted} of a copy of poses.pdbqt was shifted a column",
    )
    check(
        "and the message names the file, the line and the field",
        bad_path.name in msg
        and f"line {shifted}" in msg
        and "columns 31-38" in msg,
        f"status {msg!r}",
    )
    check(
        "and the poses already on screen are left alone, not replaced by junk",
        win.pose_list.count() == n and "poses.pdbqt" not in msg,
        f"{win.pose_list.count()} rows still shown; a refused file must not "
        f"become an empty table",
    )
    bad_path.unlink(missing_ok=True)

    # Pose browser. Its header used to sit *above* section 5b, with nothing
    # between the two, so every check below was filed under 5b's name and
    # "6. pose browser" was a heading with an empty body -- invisible until
    # sections printed their own subtotals.
    section("6. pose browser")
    # The affinity column, read by pose rather than by row, because the table
    # sorts and a row number is not a pose.
    energies_before = [
        float(win.pose_table.cell(win.pose_table.pose_row_of(i), 1))
        for i in range(n)
    ]
    shot_a, _ = shot(win, "06_pose_best")
    switched = 0
    for row in range(n):
        win.pose_list.setCurrentRow(row)
        app.processEvents()
        if win.lbl_energy.text() not in ("", "—"):
            switched += 1
    check(
        "selecting each pose updates the energy readout",
        switched == n,
        f"{switched}/{n} rows produced a value",
    )
    win.pose_list.setCurrentRow(0)
    app.processEvents()
    rmsd0 = win.lbl_rmsd.text()
    for row in range(1, n):
        win.pose_list.setCurrentRow(row)
        app.processEvents()
    check(
        "RMSD-to-best changes away from the best pose",
        win.lbl_rmsd.text() != rmsd0,
        f"row 0 = {rmsd0!r}, last row = {win.lbl_rmsd.text()!r}",
    )
    shot_b, _ = shot(win, "07_pose_last")
    pixel_check(
        "selecting a different pose changes what is drawn",
        not np.array_equal(shot_a, shot_b),
        "framebuffers differ" if not np.array_equal(shot_a, shot_b) else "identical",
    )
    win.pose_list.setCurrentRow(0)
    app.processEvents()
    check(
        "the pose table is ordered by the energies it prints",
        energies_before == sorted(energies_before),
        "; ".join(f"{e:.2f}" for e in energies_before[:4]) + " ...",
    )

    # What the table can and cannot know, from a pose *file*. The poses here
    # came from `examples/poses.pdbqt`, so there is no `DockingResult` behind
    # them: the writer emits one decimal place of energy per model and nothing
    # of the intermolecular part. The columns that have no source have to say
    # so, because a column of plausible numbers with nothing behind them is the
    # failure this table was built to avoid.
    check(
        "a pose file gives the table an affinity for every row",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 1) not in ("", "n/a")
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 1) for i in range(n)),
    )
    check(
        "and the intermolecular column admits the file never recorded it",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 2) == "—"
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 2) for i in range(n))[:80],
    )
    check(
        "every row has a distance to its nearest neighbour",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 4) not in ("", "—")
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 4) for i in range(n)),
    )
    check(
        "every row has a distance to the best pose",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 3) not in ("", "n/a")
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 3) for i in range(n)),
    )
    check(
        "the best pose is the one the table marks",
        float(win.pose_table.cell(win.pose_table.pose_row_of(min(range(n), key=lambda i: energies_before[i])), 1))
        == min(energies_before),
        f"marked row(s) {[r for r in range(n) if '*' in win.pose_table.cell(r, 0)]}, "
        f"lowest energy {min(energies_before):.2f}",
    )
    check(
        "contacts are counted for every pose, not only the selected one",
        all(win.pose_table.cell(win.pose_table.pose_row_of(i), 6).isdigit()
            for i in range(n)),
        "; ".join(win.pose_table.cell(win.pose_table.pose_row_of(i), 6) for i in range(n)),
    )
    check(
        "the rows are in the order the engine reported, best first",
        [win._pose_index_at(r) for r in range(n)] == list(range(n)),
        f"visual rows hold poses {[win._pose_index_at(r) for r in range(n)]}",
    )

    # ------------------------------------------------------------ docking run
    section("7. a real docking run from the GUI")
    win.sp_exhaust.setValue(2)
    win.center_spins[0].setValue(0.0)
    win.size_spins[0].setValue(22.0)
    app.processEvents()

    n_mol_before = len(win.viewport.molecules)
    win.btn_dock.click()
    app.processEvents()
    check("Dock disables the button while running", not win.btn_dock.isEnabled())

    # Is the window still painting? A blocked GUI thread cannot repaint.
    repaints = 0
    for _ in range(40):
        app.processEvents()
        QTest.qWait(50)
        win.viewport.repaint()
        repaints += 1
    print(f"  event-loop iterations during the run: {repaints}")
    check("the event loop keeps running during a search", repaints == 40, f"{repaints}/40")

    for _ in range(400):
        app.processEvents()
        if win.btn_dock.isEnabled():
            break
        QTest.qWait(50)
    check("Dock re-enables when the run finishes", win.btn_dock.isEnabled(),
          f"status: {win.status_label.text()!r}")
    print(f"  status after run: {win.status_label.text()!r}")
    check(
        "status reports the docking result",
        "done" in win.status_label.text() or "failed" in win.status_label.text(),
        win.status_label.text(),
    )
    check(
        "the pose table is refilled from the run",
        win.pose_list.count() > 0,
        f"{win.pose_list.count()} rows",
    )
    n_mol_after = len(win.viewport.molecules)
    roles = [m.role for m in win.viewport.molecules]
    print(f"  molecule views: {n_mol_before} before, {n_mol_after} after  roles={roles}")
    check(
        "a docking run replaces the pose view, it does not add one",
        roles.count("pose") == 1,
        f"{roles.count('pose')} pose view(s)  <-- stacking",
    )
    check(
        "one view per role, none duplicated",
        sorted(roles) == sorted(set(roles)),
        f"roles={roles}",
    )
    shot(win, "08_after_dock")

    section("7b. the pose table, from a run whose result is still in memory")
    # A run made in this window keeps its `DockingResult`, and that is the only
    # source of the intermolecular energies and of the engine's own RMSDs: the
    # pose file the run is written to carries one decimal place of energy per
    # model and nothing of the rest. So the table's numbers are checked against
    # an *independent* run of the same receptor, ligand, box and seed, which is
    # the only way to catch the table quietly showing the file's weaker value.
    #
    # Its own window with a fixed seed. The main window docks at exhaustiveness 2
    # against a 22 A box for speed, and its seed is 0 -- random -- so neither a
    # fixed-seed comparison nor a reproducible result is possible there.
    from opendocking.core import (  # noqa: PLC0415
        GridBox as _GridBox,
        Receptor as _Receptor2,
        dock as _dock,
        load_ligand as _load_ligand,
    )

    pw2 = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                     ligand=EXAMPLES / "biotin_prep.pdbqt")
    pw2.resize(1500, 900)
    pw2.show()
    try:
        app.processEvents()
        # Box first, then exhaustiveness. In that order, and only in that order:
        # setting the box makes the workbench suggest a walk count for it, which
        # overwrites a value typed before it. A 26 A box wants 32 walks, so
        # setting the spins first would silently dock at 32 while this section
        # believes it docked at 8 -- and then compare two different searches.
        pw2.size_spins[0].setValue(26.0)
        pw2.size_spins[1].setValue(26.0)
        pw2.size_spins[2].setValue(26.0)
        pw2.center_spins[0].setValue(0.0)
        pw2.center_spins[1].setValue(0.0)
        pw2.center_spins[2].setValue(0.0)
        pw2.sp_exhaust.setValue(8)
        pw2.sp_seed.setValue(20260901)
        app.processEvents()
        check(
            "the run's box and effort are the ones this section means to dock",
            [round(s.value(), 2) for s in pw2.size_spins] == [26.0] * 3
            and pw2.sp_exhaust.value() == 8
            and pw2.sp_seed.value() == 20260901,
            f"box {[s.value() for s in pw2.size_spins]}, "
            f"exhaustiveness {pw2.sp_exhaust.value()}, seed {pw2.sp_seed.value()}",
        )
        pw2.btn_dock.click()
        for _ in range(600):
            app.processEvents()
            if pw2.btn_dock.isEnabled():
                break
            QTest.qWait(50)
        for _ in range(8):
            app.processEvents()
            QTest.qWait(40)
        np2 = pw2.pose_table.rowCount()
        check(
            "the run filled the table",
            np2 > 1 and "done" in pw2.status_label.text(),
            f"{np2} rows, status {pw2.status_label.text()!r}",
        )

        # The same search, run again here, outside the window.
        _rec2 = _Receptor2.from_pdbqt(EXAMPLES / "1crn_prep.pdbqt")
        _box2 = _GridBox.from_center_size((0.0, 0.0, 0.0), (26.0, 26.0, 26.0))
        _ref = _dock(
            _load_ligand(EXAMPLES / "biotin_prep.pdbqt"),
            _rec2.precalculate(_box2, "vina"),
            exhaustiveness=8,
            num_modes=9,
            rmsd_cutoff=1.0,
            seed=20260901,
            scoring="vina",
        )
        def col(pose: int, index: int):
            """One cell of the table as a float, or None if it prints a dash.

            The window deliberately leaves a cell empty when the number is not
            known -- the intermolecular energy of a pose read back from a file
            is one of those. Turning that into a float raises and takes the
            suite down before the check can report the very thing it is
            looking for, so the dash survives to the comparison instead.
            """
            try:
                return float(pw2.pose_table.cell(pw2.pose_table.pose_row_of(pose), index))
            except (TypeError, ValueError):
                return None

        def agrees(table, engine, tol: float) -> bool:
            """Every cell is a number, and every one is the engine's."""
            return len(table) == len(engine) and all(
                a is not None and abs(a - b) < tol for a, b in zip(table, engine)
            )

        energies = [col(i, 1) for i in range(np2)]
        inters = [col(i, 2) for i in range(np2)]
        rmsds = [col(i, 3) for i in range(np2)]
        gaps = [col(i, 4) for i in range(np2)]
        print(f"  reference run: {list(_ref.energies)}")
        print(f"  table kcal/mol: {energies}")
        print(f"  table inter:    {inters}")
        check(
            "the affinity column is the engine's, to the hundredth",
            agrees(energies, list(_ref.energies), 0.005),
            f"table {energies} vs engine {list(_ref.energies)}",
        )
        # The pose file is a lossy summary of the result: the writer emits
        # `REMARK VINA RESULT: {energy:.1f}`, so a table reading it would show
        # one decimal place. Comparing the table against *that* is the only way
        # to catch the regression, because comparing it against the engine
        # twice would pass whatever the table did.
        file_energies = [round(float(e), 1) for e in _ref.energies]
        check(
            "and it is not the pose file's one-decimal value",
            any(
                a is not None and abs(a - b) > 0.005
                for a, b in zip(energies, file_energies)
            ),
            f"table {energies} vs the file's one-decimal "
            f"{[round(e, 1) for e in energies if e is not None]}",
        )
        check(
            "the intermolecular column is the engine's too",
            agrees(inters, list(_ref.intermolecular_energies), 0.005),
            f"table {inters} vs engine {list(_ref.intermolecular_energies)}",
        )
        check(
            "the RMSD column agrees with the engine's rmsds",
            agrees(rmsds, list(_ref.rmsds), 0.01),
            f"table {rmsds} vs engine {list(_ref.rmsds)}",
        )
        check(
            "the engine's rmsd and the window's own per-atom RMSD are one number",
            all(
                abs(pw2._pose_rmsd(i) - pw2._rmsd_to_best(i)) < 0.01
                for i in range(np2)
            ),
            f"engine {rmsds} vs the window's own coordinate comparison "
            f"{[round(pw2._rmsd_to_best(i), 3) for i in range(np2)]}",
        )
        check(
            "the table's energy order is the engine's",
            all(e is not None for e in energies) and energies == sorted(energies),
            f"{energies}",
        )

        # The gap column, against the engine's own coordinates rather than
        # against the window's function that produced it. `all_pose_coords` is
        # the same data the poses were written from, so a gap column that
        # measured the wrong thing -- the distance to the best pose instead of
        # the nearest neighbour, say -- would disagree with it.
        allc = np.asarray(_ref.all_pose_coords(), np.float64)
        blanks = [i for i, g in enumerate(gaps) if g is None]
        worst = 0.0
        nearest = []
        for i in range(allc.shape[0]):
            ds = []
            for j in range(allc.shape[0]):
                if i == j:
                    continue
                m = min(allc[i].shape[0], allc[j].shape[0])
                ds.append(
                    float(np.sqrt(((allc[i][:m] - allc[j][:m]) ** 2).sum(axis=1).mean()))
                )
            nearest.append(min(ds))
            if gaps[i] is not None:
                worst = max(worst, abs(min(ds) - gaps[i]))
        check(
            "every gap is the distance to that pose's nearest neighbour",
            not blanks and worst < 0.005,
            f"largest disagreement {worst:.4f} A over {np2} rows; table {gaps} "
            f"vs the engine's coordinates {[round(d, 3) for d in nearest]}"
            + (f"; {len(blanks)} row(s) printed no gap at all" if blanks else ""),
        )
        below = [g for g in gaps if g is not None and g < 1.0]
        print(f"  nearest-neighbour gaps: {gaps} (clustering cutoff 1.0 A)")
        check(
            "a gap under the clustering distance is called out, not hidden",
            all(
                "same mode" in pw2.pose_table.item(
                    pw2.pose_table.pose_row_of(i), 4).toolTip()
                for i, g in enumerate(gaps) if g is not None and g < 1.0
            ),
            f"{len(below)} row(s) under 1.0 A in this run",
        )

        # Sorting has to be real, and the best marker has to survive it.
        best_pose = int(np.argmin(energies))
        pw2.pose_table.sortItems(1, QtCore.Qt.SortOrder.AscendingOrder)
        app.processEvents()
        after = [
            float(pw2.pose_table.cell(r, 1)) for r in range(pw2.pose_table.rowCount())
        ]
        check(
            "sorting by affinity is a numeric sort, not a text one",
            after == sorted(energies),
            f"{after}",
        )
        check(
            "a text sort would have put these in a different order",
            [f"{e:.2f}" for e in after] != sorted(f"{e:.2f}" for e in after),
            f"as text: {sorted(f'{e:.2f}' for e in after)}",
        )
        marked = [r for r in range(pw2.pose_table.rowCount())
                  if "*" in pw2.pose_table.cell(r, 0)]
        check(
            "the best marker is still on the best pose after sorting",
            len(marked) == 1 and pw2._pose_index_at(marked[0]) == best_pose,
            f"marked visual row {marked}, pose "
            f"{[pw2._pose_index_at(r) for r in marked]}, best pose {best_pose}",
        )
        check(
            "sorting did not renumber the ranks",
            sorted(
                int(pw2.pose_table.cell(pw2.pose_table.pose_row_of(i), 0).split()[0])
                for i in range(np2)
            ) == list(range(1, np2 + 1)),
            "the rank is a fact about the pose, so it travels with it",
        )
        check(
            "the marked row is the row the window says is best",
            abs(
                float(pw2.pose_table.cell(marked[0], 1)) - min(energies)
            ) < 1e-9,
            f"marked row energy {pw2.pose_table.cell(marked[0], 1)}, "
            f"lowest {min(energies):.2f}",
        )

        # And the selection still drives everything, after all that sorting.
        target = np2 - 1
        row = pw2.pose_table.pose_row_of(target)
        pw2.pose_table.setCurrentCell(row, 0)
        app.processEvents()
        check(
            "a sorted table still shows the pose the selected row is",
            f"{energies[target]:.2f}" in pw2.lbl_energy.text(),
            f"selected pose {target + 1}: table says {energies[target]:.2f}, "
            f"label says {pw2.lbl_energy.text()!r}",
        )
        check(
            "and the RMSD label is that pose's, not the row's",
            f"{rmsds[target]:.2f}" in pw2.lbl_rmsd.text(),
            f"table RMSD {rmsds[target]:.2f}, label {pw2.lbl_rmsd.text()!r}",
        )
        contacts_sum = sum(
            int(pw2.pose_table.cell(pw2.pose_table.pose_row_of(i), 6))
            for i in range(np2)
        )
        check(
            "contacts are counted for every pose in a real result",
            contacts_sum > 0 and all(
                pw2.pose_table.cell(pw2.pose_table.pose_row_of(i), 6).isdigit()
                for i in range(np2)
            ),
            f"{contacts_sum} contacts over {np2} poses",
        )

        # ------------------------------------------------- all poses at once
        pw2.pose_table.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
        app.processEvents()
        pw2.pose_table.setCurrentCell(pw2.pose_table.pose_row_of(0), 0)
        app.processEvents()
        # Selecting a pose starts a camera move, and both the screenshot and
        # the distance below used to be taken while it was still running. That
        # is not a rounding error: `processEvents()` does not advance a
        # wall-clock `QTimer`, so the read was a moment of a move. Measured
        # with the nine poses identical -- this window docks with the fixed seed
        # 20260901 -- the same pose measured 23.6 A on one run and 45.4 A on
        # another, and only the timing differed. Waiting on the clock is what
        # makes this a camera position.
        _settled = _settle_camera(pw2, app)
        check(
            "the camera comes to rest on the pose it selected",
            _settled,
            f"the move finished inside the budget: moving="
            f"{pw2.viewport.moving} after waiting on the clock",
        )
        before_pixels, _ = shot(pw2, "07b_pose_alone")
        cam_before = float(pw2.viewport.camera.distance)
        check(
            "with the overlay off, one pose is drawn",
            [m.role for m in pw2.viewport.molecules].count("pose_ghost") == 0
            and sum(1 for m in pw2.viewport.molecules if m.visible and m.role == "pose") == 1,
            f"roles {[m.role for m in pw2.viewport.molecules]}",
        )
        check(
            "and the label says so in words, not only in the picture",
            "on its own" in pw2.lbl_poses.text(),
            f"{pw2.lbl_poses.text()!r}",
        )
        pw2.cb_all_poses.setChecked(True)
        for _ in range(6):
            app.processEvents()
            QTest.qWait(40)
        ghosts = [m for m in pw2.viewport.molecules if m.role == "pose_ghost"]
        visible_ghosts = [g for g in ghosts if g.visible]
        check(
            "turning it on draws every other pose",
            len(ghosts) == np2 and len(visible_ghosts) == np2 - 1,
            f"{len(ghosts)} ghost views, {len(visible_ghosts)} visible of "
            f"{np2} poses (the selected one is drawn solid instead)",
        )
        check(
            "the selected pose is not also drawn as its own ghost",
            all(
                not (g.visible and pw2._pose_index_at(
                    pw2.pose_table.currentRow()) == i)
                for i, g in enumerate(pw2._pose_ghosts)
            ),
            f"current pose {pw2._current_pose + 1}",
        )
        check(
            "the ghosts are faint, and the solid pose is not",
            all(g.opacity < 0.5 for g in ghosts)
            and all(m.opacity >= 0.99 for m in pw2.viewport.molecules
                    if m.role == "pose"),
            f"ghost opacities {sorted({g.opacity for g in ghosts})}, pose "
            f"opacity {[m.opacity for m in pw2.viewport.molecules if m.role == 'pose']}",
        )
        # The site cloud is bigger than the poses and sits in the same place,
        # so it has to step aside -- *all the way*, which is the part that is
        # easy to get wrong. Opacity scales how bright the cloud is, not how
        # much of the frame it occupies: measured on this very run, the cloud
        # changed 76,398 pixels at 0.34 and 76,398 pixels at 0.10, the same
        # number to the pixel, against 8,343 for the selected pose. So a faded
        # cloud is still a cloud that owns nine times the picture, and the
        # object-model assertion below would have passed for a picture that
        # says nothing. The claim has to be measured in the rendered frame.
        check(
            "the site cloud's opacity is zero while poses are being compared",
            pw2.viewport.pocket_opacity == 0.0
            and POCKET_OPACITY_COMPARE == 0.0
            and pw2.viewport.pocket_opacity < min(g.opacity for g in ghosts),
            f"pocket opacity {pw2.viewport.pocket_opacity} while comparing, "
            f"{POCKET_OPACITY_PLAIN} normally, ghosts at "
            f"{sorted({g.opacity for g in ghosts})}: faded is not the same as "
            f"out of the way, and only one of those changes the picture",
        )
        # Both directions, in the pixels. The first is the sanity half: a pose
        # that drew nothing would satisfy a naive ratio, so the pose's own
        # footprint is measured too. The second is the real claim, and it is
        # deliberately not "the frame changed" -- it is that the frame while
        # comparing is *identical* to the frame with the cloud's points removed
        # altogether. A cloud drawn at any opacity, however faint, makes those
        # two differ, which is exactly why 0.10 was not enough.
        cloud_full = _footprint_of(
            pw2, lambda vp: _set(vp, pocket_opacity=POCKET_OPACITY_PLAIN)
        )
        pose_own = _footprint_of(
            pw2,
            lambda vp: _hide_role(pw2, "pose"),
        )
        empty = np.zeros((0, 3), np.float32)
        with_points = _grab(pw2)
        restore_points = pw2.viewport.pocket_points
        pw2.viewport.pocket_points = empty
        without_points = _grab(pw2)
        pw2.viewport.pocket_points = restore_points
        _grab(pw2)
        pixel_check(
            "with the cloud at full strength it out-areas the solid pose",
            pose_own > 0 and cloud_full > pose_own,
            f"at {POCKET_OPACITY_PLAIN} the cloud owns {cloud_full} px against "
            f"the pose's {pose_own} px ({cloud_full / max(1, pose_own):.1f}x) -- "
            f"the ratio the overlay has to beat, and the reason fading to 0.10 "
            f"was not a fix",
        )
        pixel_check(
            "while the overlay is on the cloud is not drawn at all",
            with_points is not None
            and without_points is not None
            and np.array_equal(with_points, without_points),
            f"{differing_pixels(with_points, without_points)} pixels differ "
            f"between the frame as shown and the same frame with all "
            f"{len(restore_points)} cloud points deleted; any opacity at all "
            f"would make them differ",
        )
        # The ghosts were slate within 0.07 per channel of the receptor and
        # were drawn with the protein's own element colours, so eight of them
        # rendered as eight dimmer grey carbons inside a grey protein. The
        # claim now is that they differ in *kind*: one flat colour, and not the
        # protein's.
        #
        # Asked of `_draw_colours_for`, not of `MoleculeView.atom_colors`: the
        # renderer uses the former, so the latter would report the element
        # table and say nothing about what reaches the framebuffer.
        ghost_cols = {
            c for g in ghosts for c in map(tuple, np.round(_draw_colours_for(g), 3))
        }
        # Both sides built the same way, as float32. The trap is not equality --
        # `np.float32(0.38) == 0.38` is True, NumPy compares in float32 -- it is
        # *hashing*. `hash(np.float32(0.38))` is the hash of 0.37999999523162842
        # and `hash(0.38)` is the hash of a different float64, so the two tuples
        # land in different buckets and the set comparison answers False without
        # ever comparing a single element. A colour check that fails while
        # printing the right colour in both halves of its own message is the
        # most confusing failure available, and this is where it comes from.
        ghost_want = {tuple(np.round(np.asarray(POSE_GHOST_COLOR, np.float32), 3).tolist())}
        pose_want = {
            tuple(np.round(np.asarray(COLOR_BEST_POSE, np.float32), 3).tolist())
        }
        pose_view = next(m for m in pw2.viewport.molecules if m.role == "pose")
        pose_cols = {tuple(np.round(c, 3)) for c in _draw_colours_for(pose_view)}
        check(
            "every ghost is one flat colour, so it cannot pass for a second pose",
            ghost_cols == ghost_want,
            f"{len(ghost_cols)} distinct colour(s) across all "
            f"{sum(len(g.coords) for g in ghosts)} ghost atoms: "
            f"{sorted(ghost_cols)} (want exactly {sorted(ghost_want)})",
        )
        check(
            "the selected pose wears its own colour, and not only while comparing",
            pose_cols == pose_want,
            f"the pose is drawn in {sorted(pose_cols)} whether or not the "
            f"compare overlay is up, which is its own identity colour "
            f"{COLOR_BEST_POSE}. It used to be element-coloured with the "
            f"overlay off, and the measurement that reversed that is in "
            f"`_draw_colours_for`: over the pose's own 23 092 pixels its chroma "
            f"median was 8 against the receptor's 7 -- the receptor's own 60th "
            f"percentile, so the pose sat *inside* the protein's colour "
            f"distribution and there was nothing to find it by",
        )
        check(
            "and the pose and the ghosts differ in kind, not only in strength",
            pose_cols.isdisjoint(ghost_cols) and len(pose_cols) == 1,
            f"the pose is in {sorted(pose_cols)} and the ghosts in "
            f"{sorted(ghost_cols)}: a green molecule in a field of slate ones is "
            f"a difference the eye makes before it reads a number, and the two "
            f"sets share no colour at all",
        )
        check(
            "the ghost colour is not the receptor's, which is what made it invisible",
            all(
                abs(a - b) >= 0.08
                for a, b in zip(POSE_GHOST_COLOR, COLOR_RECEPTOR)
            ),
            f"ghost {POSE_GHOST_COLOR} vs receptor {COLOR_RECEPTOR}: per-channel "
            f"gap "
            f"{[round(abs(a - b), 3) for a, b in zip(POSE_GHOST_COLOR, COLOR_RECEPTOR)]}"
            f", want >= 0.08 in every channel (it was 0.07/0.06/0.04)",
        )
        # The two checks above ask what the renderer *means* to draw. This asks
        # what it *did*, and it is the one that has teeth: `draw_spheres` used
        # to reach for `atom_colors()` itself, so a ghost was flat-coloured in
        # the bond segments and still element-coloured in every sphere -- and
        # "spheres" is the default representation, so the whole change would
        # have been invisible in the view a user actually gets. Both directions,
        # against the protein in the very same frame: the ghosts have to be
        # bluer than what they sit in, and they were not.
        g_frame, g_mask = _own_pixels(pw2, lambda: _hide_role(pw2, "pose_ghost"))
        r_frame, r_mask = _own_pixels(pw2, lambda: _hide_role(pw2, "receptor"))
        p_frame, p_mask = _own_pixels(pw2, lambda: _hide_role(pw2, "pose"))
        g_blue = _bluer_fraction(g_frame, g_mask)
        r_blue = _bluer_fraction(r_frame, r_mask)
        g_green = _greener_fraction(g_frame, g_mask)
        p_green = _greener_fraction(p_frame, p_mask)
        r_green = _greener_fraction(r_frame, r_mask)
        pixel_check(
            "and the ghosts really are drawn in that one colour, spheres included",
            g_blue >= 0.95 and g_blue > r_blue + 0.10,
            f"{100 * g_blue:.1f}% of the "
            f"{int(g_mask.sum()) if g_mask is not None else 0} pixels the ghosts "
            f"own are bluer than red, against {100 * r_blue:.1f}% of the "
            f"protein's {int(r_mask.sum()) if r_mask is not None else 0}. With "
            f"the ghosts on element colours this reads within a point of the "
            f"protein's, which is the whole reason they were invisible.",
        )
        # The question the overlay exists to answer, asked of the pixels: is the
        # selected pose separable from the eight ghosts *and* from the protein
        # it sits in? Three different signatures -- the pose green, the ghosts
        # blue, the protein neither -- is what "a human can tell" means here,
        # and each is measured against the other two rather than against a
        # constant, so the thresholds cannot be met by a uniformly brighter
        # frame.
        pixel_check(
            "so the selected pose is separable in the picture: green against slate, both against protein",
            p_green >= 0.90
            and p_green > g_green + 0.60
            and p_green > r_green + 0.60,
            f"greener than both red and blue: the selected pose "
            f"{100 * p_green:.1f}% of its {int(p_mask.sum()) if p_mask is not None else 0} px, "
            f"the ghosts {100 * g_green:.1f}%, the protein "
            f"{100 * r_green:.1f}% -- one number, three populations, no overlap. "
            f"(Blueness is deliberately *not* part of this claim: the pose's own "
            f"green has B {COLOR_BEST_POSE[2]} above R {COLOR_BEST_POSE[0]}, so it "
            f"is also {100 * _bluer_fraction(p_frame, p_mask):.1f}% 'bluer than "
            f"red' and asking it not to be would be asking a false question.)",
        )
        check(
            "the state is stated in words, so a faint overlap is not read as a fault",
            "ghosted" in pw2.lbl_poses.text() and "solid" in pw2.lbl_poses.text(),
            f"{pw2.lbl_poses.text()!r}",
        )
        after_pixels, _ = shot(pw2, "07b_pose_all")
        repr_before = pw2.viewport.representation
        cam_after = float(pw2.viewport.camera.distance)
        # **What is asserted here is the geometry, not a direction.** The claim
        # this replaces was "turning it on moves the camera *in*", against
        # `cam_after < cam_before * 0.75`, and it was wrong twice over.
        #
        # The direction is not a property of the product. The camera frames the
        # one selected pose, and turning the overlay on reframes it to all nine.
        # This window docks with the fixed seed 20260901, so the nine poses are
        # the same every run, and with the tween settled the distance goes
        # 23.6 A -> 31.4 A: *back*, not in. Row 0 is the worst-energy pose and
        # the most isolated one, so a close-up of it is closer than a frame of
        # the whole set. An assertion about which way the camera went would be
        # an assertion about which pose happens to be selected.
        #
        # And the number that used to make it pass, 45.4 A, was the distance
        # left over from an unrelated earlier framing -- the check was passing
        # on a different feature of the same run.
        #
        # So the claim is the one a user would recognise, and the one a broken
        # `_frame_poses` cannot satisfy: once the overlay is up, every pose on
        # screen is inside the picture. A camera that did not reframe would
        # leave eight of the nine outside it.
        from opendocking.workbench.framing_selection import (  # noqa: PLC0415
            FramingTarget as _FramingTarget,
            projected_fill as _projected_fill,
        )

        _cam7b = pw2.viewport.camera
        _right, _up, _forward = _cam7b.basis()
        _aspect = pw2.viewport.width() / max(pw2.viewport.height(), 1)
        _shown = [m for m in pw2.viewport.molecules
                  if m.role in ("pose", "pose_ghost") and m.visible]
        _all_pose_pts = (
            np.vstack([np.asarray(m.coords, np.float64) for m in _shown])
            if _shown else np.zeros((0, 3), np.float64)
        )
        _target7b = _FramingTarget(
            center=np.asarray(_cam7b.center, np.float64),
            distance=float(_cam7b.distance),
            pose_atoms=0,
            partner_atoms=0,
            partner_residues=(),
        )
        _span7b = (
            _projected_fill(_all_pose_pts, _target7b, _right, _up, _forward,
                            fov=_cam7b.fov, aspect=_aspect)
            if len(_all_pose_pts) else 0.0
        )
        check(
            "turning the overlay on reframes to the whole pose set, not one pose",
            len(_shown) >= 2 and _span7b <= 1.0,
            f"camera {cam_before:.1f} A -> {cam_after:.1f} A, and the "
            f"{len(_shown)} pose views on screen reach {_span7b:.2f} of the way "
            f"to the frame edge, so all of them are inside the picture. The "
            f"distance went back rather than in because row 0 is the "
            f"worst-energy pose and the most isolated one, which is why the "
            f"direction is reported and not asserted",
        )
        pixel_check(
            "the picture really changed when the overlay came on",
            not np.array_equal(before_pixels, after_pixels),
            f"{differing_pixels(before_pixels, after_pixels)} pixels differ",
        )
        # The overlay's hierarchy has to survive every representation, and that
        # is not automatic. `draw_spheres` used to reach for `atom_colors()`
        # itself, so the flat ghost colour reached the bond segments and not
        # the spheres -- and "spheres" is the default representation, so the
        # whole change would have been invisible in the view a user actually
        # gets while the object model said everything was fine. One check per
        # representation, each measured off its own frame, is what catches that
        # class; a single check on the default would not.
        chroma_ratios: dict[str, tuple[float, float, float]] = {}
        for key in REPRESENTATION_KEYS:
            pw2.viewport.representation = key
            pf, pm = _own_pixels(pw2, lambda: _hide_role(pw2, "pose"))
            gf, gm = _own_pixels(pw2, lambda: _hide_role(pw2, "pose_ghost"))
            name = f"in '{key}' the selected pose is still green and the ghosts still slate"
            if pm is None or gm is None or not pm.any() or not gm.any():
                # A failure now, where this used to be a skip. The skip was
                # honest -- it named the cause in full -- and it was also the
                # thing that let the cause survive: `stick` drew the receptor
                # and no pose, and a suite that records that as "unanswerable"
                # is a suite that will report it again next month. The cause is
                # fixed (`_pose_bonds`), so a representation that still draws
                # nothing here is a new defect, and it has to look like one.
                _pose_now = next(
                    (m for m in pw2.viewport.molecules if m.role == "pose"), None
                )
                _ghost_now = next(
                    (m for m in pw2.viewport.molecules if m.role == "pose_ghost"), None
                )
                pixel_check(
                    name,
                    False,
                    f"'{key}' drew "
                    f"{int(pm.sum()) if pm is not None else 'unreadable'} px for the "
                    f"pose and {int(gm.sum()) if gm is not None else 'unreadable'} "
                    f"for the ghosts, so there is no colour to measure. A pose "
                    f"view carries "
                    f"{len(_pose_now.bond_pairs()) if _pose_now is not None else 0} "
                    f"bonds and a ghost "
                    f"{len(_ghost_now.bond_pairs()) if _ghost_now is not None else 0}, "
                    f"so a bond-only representation has something to draw and "
                    f"drawing nothing is a fault, not a gap in the suite.",
                )
                continue
            # Hue over the pixels that have one, plus how saturated the pose is
            # relative to the ghosts. The second half is the check that keeps the
            # first honest: a uniform grey wash is "not green" and "not slate",
            # so hue alone would pass a picture that had lost both colours. The
            # ghosts are flat slate at low opacity, which is *supposed* to be
            # desaturated -- the selected pose has to be the vivid thing.
            pg = _greener_fraction_chromatic(pf, pm)
            gb = _bluer_fraction_chromatic(gf, gm)
            pc = _chromatic_fraction(pf, pm)
            gc = _chromatic_fraction(gf, gm)
            chroma_ratios[key] = (pc, gc, pc / max(gc, 1e-9))
            pixel_check(
                name,
                pg >= 0.90 and gb >= 0.95 and pc >= POSE_CHROMA_FLOOR * gc,
                f"pose greener {100 * pg:.1f}% of its {int(pm.sum())} px that have "
                f"a hue ({100 * pc:.0f}% of its own pixels do), ghosts bluer "
                f"{100 * gb:.1f}% of {int(gm.sum())} px; the pose is "
                f"{pc / max(gc, 1e-9):.1f}x as saturated as the ghosts",
            )
        pw2.viewport.representation = repr_before

        # The control for `POSE_CHROMA_FLOOR`. A relative threshold is only
        # worth having if it rejects the picture it was written to reject, and
        # lowering a floor is exactly the change that can quietly remove that.
        # Halving the pose's own measured chroma is the failure the clause
        # exists for -- a pose that has lost its colour -- so the halved number
        # has to fail, and it has to fail against the *measured* ghost fraction
        # of the representation it was measured in rather than a remembered one.
        if "space_filling" in chroma_ratios:
            pc, gc, _ = chroma_ratios["space_filling"]
            check(
                "the chroma floor still rejects a pose that has lost half its colour",
                pc / 2.0 < POSE_CHROMA_FLOOR * gc,
                f"the space-filling pose measures {pc:.3f} chromatic against "
                f"{gc:.3f} for the ghosts ({POSE_CHROMA_FLOOR}x floor), and half "
                f"of that pose's chroma -- {pc / 2.0:.3f} -- reads "
                f"{pc / 2.0 / max(gc, 1e-9):.2f}x, which the floor rejects. A "
                f"floor that a half-grey pose passes is not a floor; this is "
                f"what makes lowering the number from 2.0 defensible",
            )

        # One crop per representation as well, because a threshold passing and
        # a picture reading are different claims and only the second one is the
        # one a user meets.
        for key in REPRESENTATION_KEYS:
            pw2.viewport.representation = key
            _save_pose_crop(pw2, f"07b_pose_all_{key}")
        pw2.viewport.representation = repr_before
        _save_pose_crop(pw2, "07b_pose_all_poses")
        _save_pose_crop(pw2, "07b_pose_all_poses_wide", radius=260.0)

        # Selecting another pose must move the solid one, not add a second.
        pw2.pose_table.setCurrentCell(pw2.pose_table.pose_row_of(3), 0)
        for _ in range(6):
            app.processEvents()
            QTest.qWait(40)
        still = [m for m in pw2.viewport.molecules if m.role == "pose"]
        check(
            "choosing another pose moves the solid one and re-ghosts the old",
            len(still) == 1
            and sum(1 for g in ghosts if g.visible) == np2 - 1
            and ghosts[0].visible,
            f"{len(still)} solid pose view(s), "
            f"{sum(1 for g in ghosts if g.visible)} ghosts visible, "
            f"selected pose {pw2._current_pose + 1}",
        )
        check(
            "and the words followed the selection",
            f"pose {pw2._current_pose + 1} of {np2} solid" in pw2.lbl_poses.text(),
            f"{pw2.lbl_poses.text()!r}",
        )
        # The ligand checkbox has to govern the ghosts too.
        pw2.cb_ligand.setChecked(False)
        app.processEvents()
        check(
            "hiding the ligand hides the ghosts as well",
            not any(m.visible for m in pw2.viewport.molecules
                    if m.role in ("pose", "pose_ghost")),
            f"visible roles {[m.role for m in pw2.viewport.molecules if m.visible]}",
        )
        pw2.cb_ligand.setChecked(True)
        app.processEvents()
        check(
            "and showing it again brings back the overlay, not just one pose",
            sum(1 for g in ghosts if g.visible) == np2 - 1
            and all(m.visible for m in pw2.viewport.molecules if m.role == "pose"),
            f"{sum(1 for g in ghosts if g.visible)} ghosts visible",
        )
        pw2.cb_all_poses.setChecked(False)
        app.processEvents()
        check(
            "turning the overlay off takes the ghosts out of the scene",
            [m.role for m in pw2.viewport.molecules].count("pose_ghost") == 0,
            f"roles {[m.role for m in pw2.viewport.molecules]}",
        )
        # The cloud's presence follows *what the user is looking at*, not which
        # control they last touched. A pose is selected here and no site has
        # been chosen, so the cloud stays out of the picture -- and that is the
        # change from the rule this replaced, which had the compare overlay
        # alone decide. Under that rule the path "select a pose, turn the
        # overlay on, turn it off" ended with the cloud back on top of the pose,
        # which is the defect the selection framing exists to remove: the cloud
        # is drawn with the depth test off and owns nine times the pose's
        # pixels.
        check(
            "and does not bring back a cloud that a selected pose removed",
            abs(pw2.viewport.pocket_opacity - POCKET_OPACITY_COMPARE) < 1e-9,
            f"pocket opacity {pw2.viewport.pocket_opacity} with pose "
            f"{pw2._current_pose + 1} selected and no site chosen: expected "
            f"{POCKET_OPACITY_COMPARE}. Turning the compare overlay off used to "
            f"restore the cloud unconditionally, which put it back over the pose "
            f"the selection had just framed",
        )
        # The positive direction, and the half that matters: something must be
        # able to bring the cloud back, or the rule above is just "always off".
        if pw2.pocket_table.rowCount():
            pw2.pocket_table.selectRow(0)
            app.processEvents()
            check(
                "and selecting a site does bring it back",
                abs(pw2.viewport.pocket_opacity - POCKET_OPACITY_PLAIN) < 1e-9,
                f"pocket opacity {pw2.viewport.pocket_opacity} after selecting a "
                f"site row, expected {POCKET_OPACITY_PLAIN}. The cloud follows "
                f"the question the user is asking -- a pose or a site -- so both "
                f"directions are asserted: a floor that only ever says 'off' is "
                f"not a rule about what to draw",
            )
        else:
            skip(
                "and selecting a site does bring it back",
                f"this window's pocket table is empty "
                f"({pw2.pocket_table.rowCount()} rows), so there is no site to "
                f"select and the direction cannot be witnessed here; the check "
                f"is not absent, it is unanswered",
            )
    finally:
        dispose_window(app, pw2)

    section("7c. a pose set that breaks the table's assumptions")
    # Four of the thirteen mutations of this round left the suite green, and
    # every one of them was a hole in these checks rather than an equivalent
    # mutation. They all rely on a docking run being well behaved in a way no
    # docking run was:
    #   * the engine reports best first, so "mark row 0" and "mark the best
    #     pose" are the same code on a real result;
    #   * sorting by affinity is a no-op when affinity is already the row
    #     order, so a handler that reads the row number as the pose index is
    #     never wrong;
    #   * "contacts counted for every pose" passes when the unselected rows say
    #     zero, because zero is a digit;
    #   * and the sub-clustering-distance branch never runs at all -- across
    #     every run measured here the tightest nearest-neighbour gap was
    #     1.01 A against a 1.0 A cutoff, so the warning was never once
    #     exercised.
    #
    # So: one fixture, built from real coordinates, that is none of those things
    # at once. Three models, the *last* of which is the best by energy (so the
    # file is not in energy order), and the first two of which are the same
    # molecule displaced by 0.2 A (so the gap column has a sub-cutoff pair).
    from opendocking.pdbqt_writer import read_pdbqt_models  # noqa: PLC0415

    _src = ["\n".join(m) for m in
            read_pdbqt_models(EXAMPLES / "poses.pdbqt")]

    def _with_energy(model_text: str, energy: float) -> str:
        out = []
        for line in model_text.splitlines():
            if "VINA RESULT" in line:
                out.append(f"REMARK VINA RESULT:    {energy:.1f}  0.000  0.000")
            else:
                out.append(line)
        return "\n".join(out)

    def _displaced(model_text: str, shift: float) -> str:
        """The same model, every atom moved by `shift` angstrom along x."""
        out = []
        for line in model_text.splitlines():
            if line.startswith(("ATOM", "HETATM")) and len(line) > 30:
                try:
                    x = float(line[30:38]) + shift
                except ValueError:
                    out.append(line)
                    continue
                out.append(line[:30] + f"{x:8.3f}" + line[38:])
            else:
                out.append(line)
        return "\n".join(out)

    fixture = tempfile.NamedTemporaryFile(
        "w", suffix=".pdbqt", delete=False, encoding="utf-8"
    )
    fixture.write(
        "MODEL\n" + _with_energy(_src[0], -5.0) + "\n"
        "ENDMDL\n"
        "MODEL\n" + _with_energy(_displaced(_src[0], 0.2), -4.0) + "\n"
        "ENDMDL\n"
        "MODEL\n" + _with_energy(_src[1], -9.9) + "\n"
        "ENDMDL\n"
    )
    fixture.close()
    # Load the fixture into the window that is already open rather than opening
    # a third one. This section needs a *pose set*, not a window, and a third
    # MainWindow in a process that already has two is what this machine's GL
    # stack fell over on (STATUS_STACK_BUFFER_OVERRUN, no Python traceback, the
    # run ending inside `processEvents`). Fewer moving parts for the same
    # coverage; section 8 docks, which replaces whatever is loaded here anyway.
    win.load_structure(Path(fixture.name), "poses")
    pw3 = win
    try:
        app.processEvents()
        n3 = pw3.pose_table.rowCount()
        cells = [[pw3.pose_table.cell(r, c) for c in range(7)] for r in range(n3)]
        print("  fixture rows (as loaded, which is not energy order):")
        for row in cells:
            print("   ", row)
        check(
            "the fixture is the awkward case it was built to be",
            n3 == 3
            and float(cells[2][1]) == min(float(r[1]) for r in cells)
            and abs(float(cells[0][4]) - 0.2) < 0.02,
            f"{n3} rows, energies {[r[1] for r in cells]}, "
            f"gaps {[r[4] for r in cells]}",
        )
        marked = [r for r in range(n3) if "*" in pw3.pose_table.cell(r, 0)]
        check(
            "the best marker is on the lowest energy, not on the first row",
            marked == [2],
            f"marked row {marked}; row 0 holds {cells[0][1]} kcal/mol and the "
            f"lowest is {cells[2][1]}",
        )
        check(
            "a pair closer than the clustering distance is called out by name",
            all(
                "same mode" in pw3.pose_table.item(r, 4).toolTip()
                for r in range(n3)
                if float(pw3.pose_table.cell(r, 4)) < 1.0
            )
            and sum(1 for r in range(n3)
                    if float(pw3.pose_table.cell(r, 4)) < 1.0) == 2,
            f"gaps {[c[4] for c in cells]}; a 0.2 A pair is the same mode "
            f"twice, and the table has to say so",
        )
        # Colours are compared against each other rather than against a fixed
        # channel value: what the check actually claims is that a sub-cutoff gap
        # looks different from a healthy one, and the palette's idea of "normal
        # text" is not this file's to predict. If the warning brush is dropped,
        # both kinds of cell carry the same default brush and this fails.
        sub_cells = [pw3.pose_table.item(r, 4).foreground().color()
                     for r in range(n3) if float(pw3.pose_table.cell(r, 4)) < 1.0]
        ok_cells = [pw3.pose_table.item(r, 4).foreground().color()
                    for r in range(n3) if float(pw3.pose_table.cell(r, 4)) >= 1.0]
        check(
            "and it is coloured differently from a healthy gap",
            len(sub_cells) == 2 and bool(ok_cells)
            and all(c != sub_cells[0] for c in ok_cells),
            f"sub-cutoff {[c.name() for c in sub_cells]}, healthy "
            f"{[c.name() for c in ok_cells]}",
        )
        # Sorting here genuinely reorders, because the file's order is not the
        # energy order -- which is what makes the row-is-not-the-pose mistake
        # observable at all.
        pw3.pose_table.sortItems(1, QtCore.Qt.SortOrder.AscendingOrder)
        app.processEvents()
        order = [pw3._pose_index_at(r) for r in range(n3)]
        check(
            "sorting really reorders this fixture",
            order != [0, 1, 2],
            f"visual rows now hold poses {order}",
        )
        target = 0
        pw3.pose_table.setCurrentCell(pw3.pose_table.pose_row_of(target), 0)
        app.processEvents()
        check(
            "a sorted table shows the pose in the selected row, by its energy",
            f"{float(pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 1)):.2f}"
            in pw3.lbl_energy.text(),
            f"selected pose {target + 1} ({float(pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 1)):.2f} "
            f"kcal/mol), visual row {pw3.pose_table.currentRow()}, label "
            f"{pw3.lbl_energy.text()!r}",
        )
        check(
            "and the RMSD label is that pose's, not the row's",
            f"{float(pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 3)):.2f}"
            in pw3.lbl_rmsd.text(),
            f"table {pw3.pose_table.cell(pw3.pose_table.pose_row_of(target), 3)}, "
            f"label {pw3.lbl_rmsd.text()!r}",
        )
        # Every row's contact count, checked against an independent count for
        # that pose -- so "zero for the rows nobody selected" cannot pass.
        from opendocking.workbench import MoleculeView as _MV  # noqa: PLC0415
        from opendocking.workbench.app import _parse_pdbqt_atoms  # noqa: PLC0415
        from opendocking.workbench.contacts import find_contacts  # noqa: PLC0415

        _models3 = ["\n".join(m) for m in read_pdbqt_models(Path(fixture.name))]
        _rec3 = next(m for m in pw3.viewport.molecules if m.role == "receptor")
        want3 = []
        for body in _models3:
            coords, elems = _parse_pdbqt_atoms(body)
            view = _MV(name="p", coords=coords, elements=elems,
                       color=(0.4, 0.9, 0.45), radius=0.34, role="pose")
            want3.append(len(find_contacts(view, _rec3)))
        got3 = [int(pw3.pose_table.cell(pw3.pose_table.pose_row_of(i), 6))
                for i in range(n3)]
        print(f"  fixture contacts: table {got3}, counted here {want3}")
        check(
            "every row's contact count is the count for that pose",
            got3 == want3,
            f"table {got3} vs counted independently {want3}",
        )
        check(
            "and no row is a placeholder zero",
            all(g > 0 for g in got3),
            f"{got3}",
        )
    finally:
        # Put the window's own poses back rather than closing it: it is the
        # main window, and section 8 docks in it.
        win.load_structure(EXAMPLES / "poses.pdbqt", "poses")
        app.processEvents()
        with contextlib.suppress(OSError):
            Path(fixture.name).unlink()

    section("7d. a pose is a molecule you can draw")
    # `stick` is the one representation that draws bonds and nothing else, and
    # until now it drew the receptor and no pose at all: `_view_from_text` built
    # every pose view without a `bonds` array, so a user could select a pose,
    # choose the skeletal view, and the pose simply was not there. Section 7b
    # recorded that as a skip with the cause written out in full, which was
    # honest and is also why it survived -- a suite that reports "unanswerable"
    # will report it again next month.
    #
    # A pose file does carry its connectivity, in `ROOT`/`BRANCH` and not in
    # `CONECT`. A pose view cannot go and read it with `MoleculeView.from_text`
    # anyway, because `_parse_pdbqt_atoms` returns atoms in *serial* order, and
    # this file is written in file order -- measured, not assumed: its serials
    # run 5, 6, 7, 8, 9, 10, 4, 11, 12, 1, 2, 3. Delegating would have drawn a
    # pose whose bond indices pointed at other atoms. So the bonds are perceived
    # from the same coordinates the renderer draws, and the checks below hold
    # that perception against the file's *own declared tree* rather than against
    # a count -- "sixteen bonds" is a claim any threshold can be tuned into
    # agreeing with, and "these are the bonds the file states" is not.
    from opendocking.workbench import _parse_pdbqt_atoms  # noqa: PLC0415
    from opendocking.workbench.app import _pose_bonds  # noqa: PLC0415
    from opendocking.workbench.structure import (  # noqa: PLC0415
        MAX_VALENCE,
        parse_structure,
    )

    _bodies7d = ["\n".join(m) for m in
                 read_pdbqt_models(EXAMPLES / "poses.pdbqt")]

    def _file_serials(body: str) -> list[int]:
        return [int(line[6:11]) for line in body.splitlines()
                if line.startswith(("ATOM", "HETATM"))]

    def _matches_declared_tree(body: str) -> bool:
        """Are the perceived bonds *this model's* `ROOT`/`BRANCH` tree?

        Both sides are indexed in **serial order**: `_parse_pdbqt_atoms` sorts
        by the serial column, and `parse_structure` does too now that it
        honours that column instead of reading the file's line order. So this
        comparison is index-to-index and needs no mapping between the two.

        It used to need one, and that is the point. `parse_structure` returned
        file order while `_parse_pdbqt_atoms` returned serial order, so the
        function mapped the derived bonds through `to_file` before comparing --
        and that mapping was *hiding* the reader's defect rather than papering
        over a numbering difference. With the fix in, keeping the mapping
        compares two different orderings and reports a chemistry disagreement:
        measured, 0 of 9 models matched while it was still there. The mapping is
        gone rather than corrected, because there is no longer a difference for
        it to bridge.
        """
        coords, elements = _parse_pdbqt_atoms(body)
        derived = {tuple(sorted(p))
                   for p in _pose_bonds(coords, elements).tolist()}
        declared = {tuple(sorted(p))
                    for p in parse_structure(body).bond_pairs()}
        return derived == declared

    # ------------------------------------------------- the serial-order fixture
    #
    # An existing round-trip test cannot see this defect: with
    # `write_branch_tree: false` the file carries no tree, the serials are
    # written 1..N in order, and the serial order *is* the file order, so a
    # reader that ignored the serial column entirely would still agree. So the
    # fixture has to carry a real permutation, and the first thing to assert
    # about it is that it is one -- a fixture that is not actually permuted
    # would make every check below pass for the wrong reason.
    _serial_perm_ok = all(
        sorted(_file_serials(b)) == list(range(1, len(_file_serials(b)) + 1))
        and _file_serials(b) != sorted(_file_serials(b))
        for b in _bodies7d
    )
    check(
        "the pose fixture really is a permutation of the serials, and not the "
        "identity: a file written 1..N in order cannot see a reader that "
        "ignores the serial column",
        _serial_perm_ok and len(_bodies7d) > 1,
        f"file serials of model 0: {_file_serials(_bodies7d[0])}; each of the "
        f"{len(_bodies7d)} models is a permutation of 1..{len(_file_serials(_bodies7d[0]))} "
        f"and none is in ascending order -> "
        f"{'yes' if _serial_perm_ok else 'NO -- the fixture cannot see this'}",
    )
    _serials_parsed = [[a.serial for a in parse_structure(b).atoms]
                       for b in _bodies7d]
    check(
        "the reader honours the serial column: atoms come back in serial order, "
        "not in the order the lines appear",
        all(s == sorted(s) for s in _serials_parsed)
        and any(s != _file_serials(b) for s, b in zip(_serials_parsed, _bodies7d)),
        f"parsed serials of model 0: {_serials_parsed[0]}; file order was "
        f"{_file_serials(_bodies7d[0])}. This is the reading path behind the "
        "geometry divergence: it read the serial at structure.py:387 and then "
        "appended in file order, so atom i was not the ligand's atom i and every "
        "index-based consumer -- bonds, residues, templates -- answered about a "
        "different atom than the one it named",
    )
    check(
        "and the remap keeps the declared bonds pointing at the atoms they "
        "declare, which reordering `atoms` alone would not",
        all(
            _matches_declared_tree(b) for b in _bodies7d
        ),
        "the perceived bonds are the file's own ROOT/BRANCH tree, compared as "
        "(index, index) on both sides because both readers are now in serial "
        "order. A permutation of `atoms` with the index-based bond list left "
        "alone declares the wrong pairs: measured on this fixture that mutation "
        "turns 7 geometry vetoes into 13 and 9 warnings into 21, and on a "
        "straight-chain fixture, where every correct pair is within bonding "
        "distance, the final bond set stops being the chain",
    )

    pose7d = next((m for m in win.viewport.molecules if m.role == "pose"), None)
    check(
        "a pose view carries bonds, not only coordinates",
        pose7d is not None and len(pose7d.bond_pairs()) > 0,
        f"{len(pose7d.bond_pairs()) if pose7d is not None else 0} bonds: 'stick' "
        "draws bonds and nothing else, so a pose with none is a pose that is "
        "not on screen in one of the five representations",
    )

    matched7d = [_matches_declared_tree(b) for b in _bodies7d]
    check(
        "and they are the file's own declared tree, not a lookalike",
        len(matched7d) > 1 and all(matched7d),
        f"{sum(matched7d)}/{len(matched7d)} models perceive exactly the bonds the "
        "file declares, compared index-to-index because both readers are now in "
        "serial order. A count would have been enough to look convincing; this is "
        "the bonds.",
    )

    counts7d, lengths7d, overruns7d = [], [], []
    for body in _bodies7d:
        coords, elements = _parse_pdbqt_atoms(body)
        pairs = _pose_bonds(coords, elements)
        counts7d.append(len(pairs))
        degree = np.zeros(len(coords), int)
        for a, b in pairs:
            degree[a] += 1
            degree[b] += 1
            lengths7d.append(float(np.linalg.norm(coords[a] - coords[b])))
        overruns7d.append([
            (elements[i], int(degree[i]), MAX_VALENCE.get(elements[i], 99))
            for i in range(len(coords))
            if degree[i] > MAX_VALENCE.get(elements[i], 99)
        ])
    check(
        "every perceived bond is a distance a bond can physically have",
        bool(lengths7d) and min(lengths7d) >= 0.85 and max(lengths7d) < 2.0,
        # The detail is an f-string, so `min(lengths7d)` is evaluated *before*
        # `check` is called -- unconditionally, empty list or not. The first
        # version of this line said `min(lengths7d):.2f` and raised
        # `ValueError: min() arg is an empty sequence`, which took the whole
        # suite down with exit 3 in precisely the case this check exists to
        # catch. A check that dies instead of failing is the worst kind: the
        # run is truncated, so it also destroys the sections after it.
        f"{len(lengths7d)} bonds, "
        + (f"{min(lengths7d):.2f}-{max(lengths7d):.2f} A" if lengths7d
           else "none at all -- the view has coordinates and no connectivity")
        + ": the same 0.85-2.0 A window the receptor's bonds are held to, so a "
          "rule that welded two atoms across a gap would fail here too",
    )
    check(
        "and no atom carries more bonds than its element can",
        lengths7d and not any(overruns7d),
        f"no overrun across {len(_bodies7d)} models; over-cap atoms per model "
        f"{[len(o) for o in overruns7d]}. `lengths7d` is in the condition on "
        f"purpose: with no bonds at all this would pass for the same reason an "
        f"empty list has no overruns in it, which is to say not at all",
    )
    check(
        "every pose of a set perceives the same bonds, whatever its pose",
        len(set(counts7d)) == 1 and counts7d[0] > 0,
        f"bond counts {counts7d}: a count that moved with the conformation would "
        "be a perception nobody could rely on, and a docking run changes "
        "conformation by design. Nine identical *zeros* would satisfy the first "
        "half of this, hence the second",
    )

    win.cb_ligand.setChecked(True)
    win.cb_all_poses.setChecked(True)
    app.processEvents()
    QTest.qWait(150)
    ghosts7d = [m for m in win.viewport.molecules if m.role == "pose_ghost"]
    check(
        "every ghost carries bonds too, or 'stick' hides eight of the nine poses",
        len(ghosts7d) == len(counts7d)
        and all(len(g.bond_pairs()) > 0 for g in ghosts7d),
        f"{sum(1 for g in ghosts7d if len(g.bond_pairs()) > 0)}/{len(ghosts7d)} "
        f"ghost views have bonds "
        f"({sum(1 for g in ghosts7d if g.visible)} of them visible -- the scene "
        f"keeps one ghost per pose and hides the selected pose's own), against "
        f"{len(counts7d)} poses in the file",
    )

    repr7d = win.viewport.representation
    for key in ("stick", "ball_and_stick", "spheres"):
        win.cmb_representation.setCurrentIndex(
            win.cmb_representation.findData(key)
        )
        app.processEvents()
        QTest.qWait(60)
        if PIXELS_OK:
            _frame7d, _mask7d = _own_pixels(win, lambda: _hide_role(win, "pose"))
            _px7d = int(_mask7d.sum()) if _mask7d is not None else 0
            pixel_check(
                f"in '{key}' the pose is on screen, not just the receptor",
                _px7d > 200,
                f"{_px7d} px belong to the pose alone, same camera either side "
                f"of hiding it. "
                + ("'stick' draws bonds and nothing else, so this was 0 px "
                   "before `_pose_bonds` existed: the pose had no bonds to "
                   "draw." if key == "stick" else
                   f"'{key}' drew the pose from its spheres before the fix, so "
                   f"this one is a floor rather than a repair -- it is here so "
                   f"that a change which cost the pose its atoms anywhere would "
                   f"be caught in all three views, not just the one that broke."),
            )
        else:
            skip(f"in '{key}' the pose is on screen, not just the receptor",
                 "no framebuffer to read")
    # After the sweep, not before: the status bar is rewritten by
    # `_describe_representation`, which only runs when the representation
    # changes. Asked before the sweep it reads back whatever an earlier section
    # left there, and this file checked it while the message was still the empty
    # string -- a check that passes for a reason nobody can see is how a status
    # bar stops being maintained at all.
    _status7d = win.statusBar().currentMessage()
    _pose_frag7d = next(
        (p for p in _status7d.split("  |  ") if pose7d and pose7d.name in p), ""
    )
    check(
        "the status bar says the pose's bonds were inferred, not read",
        pose7d is not None
        and pose7d.bond_source == "distance"
        and "inferred from distances" in _status7d
        and _pose_frag7d != "",
        f"bond_source {pose7d.bond_source if pose7d else None!r}; the bar holds "
        f"{len(_status7d.split('  |  '))} entries and the pose's own reads "
        f"{_pose_frag7d[:90]!r}. These bonds *are* perceived from distances, and "
        "the status bar is the only place a user is ever told so",
    )

    win.cmb_representation.setCurrentIndex(
        win.cmb_representation.findData(repr7d)
    )
    win.cb_all_poses.setChecked(False)
    app.processEvents()

    section("8. a second docking run")
    win.btn_dock.click()
    for _ in range(400):
        app.processEvents()
        if win.btn_dock.isEnabled():
            break
        QTest.qWait(50)
    roles2 = [m.role for m in win.viewport.molecules]
    print(f"  molecule views now: {len(roles2)}  roles={roles2}")
    check(
        "a second run still leaves exactly one pose view",
        roles2.count("pose") == 1,
        f"{roles2.count('pose')}  <-- previous results stay on screen",
    )
    check(
        "a second run does not change the number of views",
        len(roles2) == len(roles),
        f"{len(roles)} -> {len(roles2)}",
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()
    vis = [m for m in win.viewport.molecules if m.visible]
    check(
        "after two runs the scene is exactly receptor + ligand + pose",
        len(vis) == 3 and {m.role for m in vis} == {"receptor", "ligand", "pose"},
        f"{len(vis)} visible: {[(m.role, m.name) for m in vis]}",
    )
    # And the checkbox must actually be able to hide the pose now.
    win.cb_ligand.setChecked(False)
    app.processEvents()
    pose_hidden = all(
        not m.visible for m in win.viewport.molecules if m.role == "pose"
    )
    win.cb_ligand.setChecked(True)
    app.processEvents()
    check("the ligand checkbox hides the pose view", pose_hidden)
    shot(win, "09_after_second_dock")

    # temp files left behind
    # `tempfile` is imported at module level. Re-importing it here would make
    # the name local to the whole of `main`, and every earlier use in this
    # function would then read as an unbound local -- which is how section 7c,
    # above, died with `UnboundLocalError` before it ran a single check.
    leaked = list(Path(tempfile.gettempdir()).glob("tmp*.pdbqt"))
    recent = [p for p in leaked if (Path(tempfile.gettempdir()) / p.name).exists()]
    try:
        ages = [(p.stat().st_mtime, p) for p in leaked]
        now = max(t for t, _ in ages) if ages else 0
        import time
        fresh = [p for t, p in ages if time.time() - t < 3600]
    except Exception:
        fresh = []
    print(f"  temp .pdbqt files in %TEMP% newer than 1 h: {len(fresh)}")
    check("docking does not leak temp files", not fresh, f"{len(fresh)} file(s): "
          + ", ".join(p.name for p in fresh[:4]))

    # ------------------------------------------------- negative-coordinate receptor
    section("9. a receptor at negative coordinates")
    import opendocking
    from opendocking.workbench import MoleculeView

    shifted = EXAMPLES / "_shifted_receptor.pdbqt"
    rows = []
    for line in rec.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) - 40.0
            rows.append(line[:30] + f"{x:8.3f}" + line[38:])
        else:
            rows.append(line)
    shifted.write_text("\n".join(rows) + "\n", encoding="utf-8")

    win2 = MainWindow(receptor=shifted)
    win2.resize(1000, 700)
    win2.show()
    app.processEvents()
    QTest.qWait(400)
    app.processEvents()
    true_center = np.array(opendocking.Receptor.from_pdbqt(shifted).center, dtype=float)
    ui_center = np.array([s.value() for s in win2.center_spins], dtype=float)
    shown_box = np.array(win2.viewport.box_center, dtype=float)
    print(f"  true receptor centre : {np.round(true_center, 3)}")
    print(f"  box shown in spins   : {np.round(ui_center, 3)}")
    # The reference value used to be the receptor's centroid, because loading a
    # receptor put the centroid in the spins. It no longer does -- the box goes
    # on the first site the pocket search finds, which is the whole point of
    # that change -- so the check compares the spins to the box the window
    # actually decided on. What it is really about is unchanged and is stated
    # as its own check below: a receptor at negative coordinates must not have
    # its position silently clamped to zero.
    check(
        "the box-centre spins show the box the window chose",
        bool(np.abs(shown_box - ui_center).max() < 0.05),
        f"spins {np.round(ui_center, 3)} vs viewport {np.round(shown_box, 3)}, "
        f"mismatch {np.abs(shown_box - ui_center).max():.3f} A",
    )
    # The claim in this section's title is that negative coordinates are not
    # silently clamped to zero, and that is `ui_center.min() < -5`. The old
    # second half of this check -- that the box differs from the receptor's
    # centroid by more than 0.05 A -- was a *proxy* for "the box came from the
    # search rather than from a default", and it has quietly stopped being
    # true: with the pocket search's ranking fixed, the top site on this
    # small fixture happens to sit 0.02 A from the centroid, so a site is
    # selected that a reader would call "the centre". Proving the origin
    # directly is both stronger and not fixture-dependent -- compare against
    # the site the window actually selected, not against the centroid.
    picked = win2.pocket_table.currentRow()
    site_centre = None
    models = getattr(win2, "_pocket_models", [])
    if 0 <= picked < len(models):
        site_centre = np.asarray(models[picked].center, dtype=float)
    check(
        "a receptor at negative coordinates keeps them, unclamped to zero",
        bool(ui_center.min() < -5.0),
        f"receptor centre {np.round(true_center, 3)}, box {np.round(ui_center, 3)}, "
        f"spin range [{win2.center_spins[0].minimum()}, {win2.center_spins[0].maximum()}]",
    )
    check(
        "and the box is the selected site's, not a default near the middle",
        site_centre is not None
        and bool(np.abs(ui_center - site_centre).max() < 0.05),
        f"selected row {picked}, its centre "
        f"{np.round(site_centre, 3) if site_centre is not None else None}, "
        f"box {np.round(ui_center, 3)}, "
        f"{float(np.abs(ui_center - site_centre).max()) if site_centre is not None else float('nan'):.3f} A apart"
        if site_centre is not None else "no site selected",
    )
    # Deliberately *not* checked here: "the box is somewhere other than the
    # centroid". This receptor is a handful of atoms whose one site happens to
    # sit 0.09 A from its centre, so the check could only pass with its
    # threshold loosened until it asserted nothing -- and it would be proving
    # something section 11c already proves on crambin, where the box really is
    # 9.9 A off the centroid. A duplicate that cannot fail teaches nothing.
    shot(win2, "10_negative_coords")
    dispose_window(app, win2)

    # ------------------------------------------------- display representations
    section("10. display representations")
    # `REPRESENTATION_KEYS` is imported at module scope, not here. A local
    # import inside `main` makes the name a *function-local* for the whole of
    # `main`, so section 7b's use of it -- 400 lines earlier, perfectly legal
    # against a module-level import -- raised `UnboundLocalError` and took the
    # run down with exit 3. One import, one name, and no forward reference to
    # something bound later in the same function.
    check(
        "the display selector offers every representation",
        set(REPRESENTATION_KEYS) == {"spheres", "space_filling", "ball_and_stick",
                                     "stick", "ribbon", "cartoon"},
        ", ".join(REPRESENTATION_KEYS),
    )
    check(
        "the selector has one entry per representation",
        win.cmb_representation.count() == len(REPRESENTATION_KEYS),
        f"{win.cmb_representation.count()} entries",
    )

    # Every colour the picture can contain that is not an atom colour has to
    # be named somewhere on screen. A report said this was not so: loading a
    # receptor draws the first pocket as magenta spheres, and the legend named
    # only the four dashed interaction lines, so a colour appeared that
    # nothing explained. "The 'site volume' checkbox is over there" is not the
    # same claim as "this colour is the site volume", and the difference is
    # the whole reason a legend exists.
    from opendocking.workbench import (  # noqa: PLC0415
        COLOR_CONTACT,
        COLOR_POCKET,
        CONTACT_LABELS,
    )

    # Scoped to the legend row itself. This used to reach for every `QLabel` in
    # the window and tell a legend word from other text by its stylesheet, so
    # any new secondary-text label in the panel counted as a legend entry and
    # "every swatch names itself" went red on a panel that was telling the
    # truth -- the pose breakdown's own note, which is styled the way this
    # window styles all its secondary text. A check about a legend has to look
    # at the legend.
    swatches = [
        (w.text(), w.styleSheet())
        for w in win.legend_row.findChildren(QtWidgets.QLabel)
        if w.styleSheet().startswith("color: rgb(")
    ]
    legend_words = [
        w.text() for w in win.legend_row.findChildren(QtWidgets.QLabel)
        if w.styleSheet().startswith("color: #9aa3ad")
    ]
    check(
        "the legend names the site volume, not just the interaction lines",
        "site volume" in legend_words,
        f"legend reads {legend_words}",
    )
    # The swatch colour must be the colour the renderer uses, read from the
    # same table -- otherwise the legend is a picture of a legend, and the
    # claim "read from one table so it cannot drift" is only a comment.
    def _rgb(style):
        inner = style.split("rgb(", 1)[1].split(")", 1)[0]
        return tuple(int(v) for v in inner.split(","))

    pocket_rgb = tuple(int(round(v * 255)) for v in COLOR_POCKET)
    check(
        "the site-volume swatch is the colour the cloud is drawn in",
        any(_rgb(style) == pocket_rgb and text == "●" for text, style in swatches),
        f"pocket is rgb{pocket_rgb}; swatches {[(t, _rgb(s)) for t, s in swatches]}",
    )
    check(
        "every interaction colour is in the legend too",
        all(
            any(_rgb(style) == tuple(int(round(v * 255)) for v in COLOR_CONTACT[k])
                for _, style in swatches)
            for k in COLOR_CONTACT
        ),
        f"{len(COLOR_CONTACT)} colours, {len(swatches)} swatches",
    )
    check(
        "and every swatch names itself, so none is a bare colour",
        len(legend_words) == len(swatches),
        f"{len(swatches)} swatches against {len(legend_words)} labels "
        f"{legend_words}",
    )

    repr_footprint: dict[str, int] = {}
    # **The site cloud is taken out of the frame for this sweep, and that is a
    # confound removal rather than a convenience.** The cloud is drawn with the
    # depth test off, so it is in front of both sphere modes, and it is
    # *translucent*, so it does not add the same number of pixels to both: it
    # shifts them across this suite's absolute mask threshold by different
    # amounts in each mode, and it moves them the wrong way. Measured on this
    # fixture with everything else held fixed:
    #
    #     cloud at POCKET_OPACITY_PLAIN (0.34)   spheres 803 990  CPK 791 824
    #                                            -> CPK covers LESS than separated
    #                                                spheres, which is backwards
    #     cloud at POCKET_OPACITY_COMPARE (0.0)  spheres 705 478  CPK 837 500
    #                                            -> 18.7%, the direction the
    #                                                physics predicts
    #
    # A check asking "does a mode that drew the default's picture under a new
    # label exist" cannot have a translucent overlay drawn over both answers.
    # The floor is unchanged at 10%; what changed is that the thing being
    # compared is the two representations.
    #
    # This is also the other half of a product fix. `pocket_opacity` had three
    # writers -- the compare toggle, the site table and the pose selection --
    # and nothing said which won, so which of the three ran last decided this
    # check's number, and that is why it read 0.1%, 3.9%, 23.0% and 2.7% on
    # four consecutive runs of the same code. `MainWindow._sync_pocket_opacity`
    # now derives it from state, so the picture no longer depends on which
    # handler happened to run last; pinning it here is what makes the comparison
    # about the representations regardless.
    cloud_before_sweep = win.viewport.pocket_opacity
    win.viewport.pocket_opacity = POCKET_OPACITY_COMPARE
    repr_frame: dict[str, np.ndarray] = {}
    # **The pose is taken out of the sweep for the same reason the cloud is,
    # and the reason is measured rather than suspected.** The main window docks
    # unseeded -- its seed is 0, which means random, and section 7b of this file
    # says so -- so the pose that ends up selected is a different molecule every
    # run, and round 1's `_frame_selection` then frames the camera on that
    # different pose. Three runs of this file's own code, a check that included
    # the pose read 6.8%, 24.2% and 24.9%: that is not a renderer flickering, it
    # is an input moving. A check that depends on a random dock result depends on
    # a random dock result, so the two sphere modes are compared on a frame whose
    # content is the same every run. The pose goes back the moment the sweep is
    # over, and nothing downstream of here reads whether it is visible.
    pose_views = [m for m in win.viewport.molecules if m.role == "pose"]
    pose_visible_before = [m.visible for m in pose_views]
    for m in pose_views:
        m.visible = False
    app.processEvents()
    for key in REPRESENTATION_KEYS:
        index = win.cmb_representation.findData(key)
        win.cmb_representation.setCurrentIndex(index)
        app.processEvents()
        check(
            f"selecting '{key}' reaches the viewport",
            win.viewport.representation == key,
            f"viewport says {win.viewport.representation!r}",
        )
        if PIXELS_OK:
            arr, _ = shot(win, f"11_repr_{key}")
            drawn = non_background(arr)
            repr_footprint[key] = drawn
            repr_frame[key] = arr
            check(
                f"'{key}' draws something",
                drawn > 500,
                f"{drawn} non-background px",
            )
        else:
            skip(f"'{key}' draws something", "no framebuffer to read")

    # The space-filling mode earns its name from its geometry, and a mode that
    # drew the default's picture under a new label would be the defect this
    # whole change is about wearing a different hat. So the two sphere modes
    # are compared as pictures.
    #
    # **What is compared is the number of pixels that change when the mode
    # does, not the difference between the two footprint areas.** The area
    # difference is not a property of these two representations on this fixture,
    # and saying so is a measurement rather than an opinion: with the pose out of
    # the frame, the receptor alone reads 5.6% and 4.6% in two runs, and in both
    # the *separated* mode covers MORE than the CPK surface -- 590 061 px against
    # 558 676, and 579 624 against 554 256. The comment this replaces explained
    # the direction by saying a continuous surface fills the gaps between
    # separated spheres and must therefore cover more. On this fixture that is
    # backwards, because `spheres` also draws the bonds (see `_draw_molecule`),
    # and small spheres plus sticks beat a vdW surface here. A floor sitting on
    # top of a quantity that reads 4.6% one way and 24.9% the other is not a
    # floor; it is a coin. The 10% is not moved, and the direction is not
    # asserted either: what is asserted is that switching the mode repaints a
    # materially different part of the frame.
    #
    # The tolerance is the background mask's own (12 summed over three channels),
    # so this counts "changed visibly" and not "changed by a dithering step".
    # Measured on this fixture with the pose out: 146 316 px change out of a
    # 554 256 px footprint, 25.2%. The control for the quantity is a same-mode
    # grab compared with itself, which measures 0 px -- so 10% is a line under a
    # number that is zero when nothing happened, not a line drawn under a number
    # that is never small.
    if "spheres" in repr_frame and "space_filling" in repr_frame:
        small_frame = repr_frame["spheres"]
        solid_frame = repr_frame["space_filling"]
        small = non_background(small_frame)
        solid = non_background(solid_frame)
        changed = int(
            (np.abs(small_frame.astype(np.int16) - solid_frame.astype(np.int16))
             .sum(axis=2) > 12).sum()
        )
        pixel_check(
            "the space-filling mode draws a different picture from the "
            "small-sphere one",
            changed > 0.10 * max(1, max(small, solid)),
            f"{changed} px of the frame change when the mode does, "
            f"{100 * changed / max(1, max(small, solid)):.1f}% of the larger "
            f"footprint ({max(small, solid)} px), against a 10% floor. The two "
            f"areas are {small} and {solid} px, a difference of only "
            f"{abs(solid - small)} px, which is why the areas are reported but "
            f"not the thing being measured",
        )
    else:
        skip(
            "the space-filling mode draws a different picture from the "
            "small-sphere one",
            "no framebuffer was readable, so there were no pictures to "
            "compare",
        )
    # Put the pose back, next to the cloud, in both branches, so the state the
    # rest of the suite runs in is the one it would have been in.
    for m, was in zip(pose_views, pose_visible_before):
        m.visible = was
    app.processEvents()
    # Put the cloud back, in both branches, so the state the rest of the suite
    # runs in is the one it would have been in and not the one the sweep needed.
    win.viewport.pocket_opacity = cloud_before_sweep
    app.processEvents()

    # A ribbon needs a backbone. The example receptor is a synthetic blob, so
    # it must fall back rather than pretend, and the status bar has to say so.
    win.cmb_representation.setCurrentIndex(
        win.cmb_representation.findData("ribbon")
    )
    app.processEvents()
    receptor_view = next(m for m in win.viewport.molecules if m.role == "receptor")
    check(
        "a structure with no backbone reports that it has none",
        not receptor_view.has_backbone,
        "the synthetic example receptor is not a protein",
    )
    check(
        "a file with no amino-acid residue falls back instead of claiming a ribbon",
        receptor_view.backbone_ribbon() is None,
        "backbone_ribbon() returned None",
    )
    message = win.statusBar().currentMessage()
    check(
        "the status bar says where the bonds came from",
        "bonds from" in message,
        message[:110],
    )

    # The receptor's bonds are the interesting ones. Whatever the source, no
    # bond may join atoms of different residues unless it is the peptide C-N,
    # because that is the failure this whole path was written to prevent.
    check(
        "the loaded receptor's bonds are all physically possible",
        all(
            0.85
            <= float(
                np.linalg.norm(
                    receptor_view.coords[i] - receptor_view.coords[j]
                )
            )
            < 2.0
            for i, j in receptor_view.bond_pairs()
        ),
        f"{len(receptor_view.bond_pairs())} bonds checked",
    )

    win.cmb_representation.setCurrentIndex(
        win.cmb_representation.findData("spheres")
    )
    app.processEvents()

    # The example receptor is a synthetic blob, so nothing above has actually
    # drawn a ribbon. A real protein has to, or the feature is untested.
    crambin = EXAMPLES / "1crn_prep.pdbqt"
    if crambin.is_file():
        section("10b. a ribbon on a real protein")
        win3 = MainWindow(receptor=crambin)
        win3.resize(1000, 760)
        win3.show()
        QTest.qWaitForWindowExposed(win3, 5000)
        app.processEvents()
        QTest.qWait(400)
        app.processEvents()

        protein = next(m for m in win3.viewport.molecules if m.role == "receptor")
        check("the prepared receptor kept its residue names",
              protein.has_backbone,
              f"{len(protein.coords)} atoms, bonds from {protein.bond_source}")
        check("its bonds came from residue templates",
              protein.bond_source == "template", protein.bond_source)
        check("it is grouped into one entry per residue",
              protein.structure is not None
              and len(protein.structure.residues) == 46,
              f"{0 if protein.structure is None else len(protein.structure.residues)} "
              "residues")

        ribbon = protein.backbone_ribbon()
        check("a ribbon was built from the backbone",
              ribbon is not None and len(ribbon) > 0,
              f"{0 if ribbon is None else len(ribbon)} triangles")

        drawn = {}
        for key in ("ball_and_stick", "ribbon", "cartoon"):
            win3.cmb_representation.setCurrentIndex(
                win3.cmb_representation.findData(key)
            )
            app.processEvents()
            if PIXELS_OK:
                arr, _ = shot(win3, f"12_protein_{key}")
                drawn[key] = non_background(arr)
                check(f"protein '{key}' draws something",
                      drawn[key] > 500, f"{drawn[key]} non-background px")
            else:
                skip(f"protein '{key}' draws something", "no framebuffer to read")

        if len(drawn) == 3:
            check(
                "the protein ribbon is a different picture from its atoms",
                abs(drawn["ribbon"] - drawn["ball_and_stick"]) > 500,
                f"ribbon {drawn['ribbon']} px vs ball-and-stick "
                f"{drawn['ball_and_stick']} px",
            )
        else:
            # Registering this only when it can run makes the total silently
            # smaller, and a smaller total is indistinguishable from a smaller
            # scope. A check that did not happen has to say so.
            skip(
                "the protein ribbon is a different picture from its atoms",
                "no framebuffer was readable, so there were no pixel counts to "
                "compare",
            )
        msg = win3.statusBar().currentMessage()
        check("the status bar names the template source for the protein",
              "residue templates" in msg, msg[:110])
        dispose_window(app, win3)
    else:
        skip("a ribbon on a real protein", f"{crambin.name} is not present")

    # ------------------------- 10c. secondary structure the viewer can see
    #
    # Section 10b proves a ribbon is *built*. It cannot tell a helix from a coil,
    # because both are a swept quad strip and both render. Everything below
    # exists because "a ribbon was drawn" is satisfied by a classifier that
    # returns one state for every residue -- the answer is still a ribbon, still
    # the right number of triangles, still on screen. So the checks come in
    # pairs: a positive claim about the states, and a guard that the *shape* of
    # the answer is not degenerate.
    section("10c. the ribbon knows what shape it is drawing")

    receptor_path = EXAMPLES / "1crn_prep.pdbqt"
    if not receptor_path.is_file():
        skip("10c. secondary structure", f"{receptor_path.name} is not present")
    else:
        prot_text = receptor_path.read_text(encoding="utf-8")
        prot = parse_structure(prot_text, receptor_path.name)
        prot_trace = prot.backbone()
        prot_states = secondary_structure(prot_trace, prot.atoms)
        prot_resids = [prot.atoms[ca].resid for _, ca, _ in prot_trace]
        found = ss_ref.segments(prot_states, prot_resids)

        check(
            "the shipped receptor's backbone is read for classification",
            len(prot_trace) > 0 and len(prot_states) == len(prot_trace),
            f"{len(prot_trace)} residues, {len(prot_states)} states",
        )
        # The guard, and the single most important check in this section. A
        # classifier that finds no hydrogen bonds returns "coil" 46 times, the
        # ribbon is drawn, and every other check in the file still passes.
        check(
            "the states are not all the same class",
            not ss_render.degenerate(prot_states),
            f"{ {k: prot_states.count(k) for k in ('helix', 'sheet', 'coil')} }",
        )
        # Same question asked of the raw PDB, whose hydrogens are the raw PDB's
        # own: a prepared receptor keeps polar hydrogens and a hand-written
        # backbone usually has none, so agreeing on both means the answer is
        # coming from the geometry and not from a hydrogen list.
        raw = parse_structure(
            (EXAMPLES / "1crn_receptor.pdb").read_text(encoding="utf-8"), "1crn_receptor.pdb"
        )
        raw_states = secondary_structure(raw.backbone(), raw.atoms)
        check(
            "the states are not all the same class on the raw PDB either",
            not ss_render.degenerate(raw_states),
            f"{ {k: raw_states.count(k) for k in ('helix', 'sheet', 'coil')} } on the "
            "unprepared file, which carries no polar hydrogens to donate",
        )
        name, ok, detail = _verify_degenerate_guard(prot_trace, prot.atoms)
        check(name, ok, detail)

        # Agreement with a reference whose right answer can be stated. Each
        # entry of `CRAMBIN_1CRN_SS` is a published segment of 1CRN; the check
        # asks for a *run* of that state covering it, because an annotation is
        # a list of segments and a terminus moving by one residue is not a
        # disagreement.
        # Agreement with a reference whose right answer can be stated. Each
        # entry of `CRAMBIN_1CRN_SS` is a published segment of 1CRN; the check
        # asks for a *run* of that state covering the range this module reports,
        # because an annotation is a list of segments and a terminus moving by
        # one residue is not a disagreement. The three residues where the
        # reported range and the published one differ are pinned separately
        # below rather than quietly folded into the reference.
        state_of = {r: s for r, s in zip(prot_resids, prot_states)}
        for lo, hi, want, what, rlo, rhi in ss_ref.CRAMBIN_1CRN_SS:
            inside = [r for r in prot_resids if rlo <= r <= rhi]
            ok = bool(inside) and all(state_of[r] == want for r in inside)
            check(
                f"1CRN {what} is classified {want}",
                ok,
                f"residues {rlo}-{rhi}: "
                + "".join(state_of[r][0].upper() for r in inside)
                + f" (want {want[0].upper() * len(inside)})",
            )

        # The exact set of residues where the answer differs from the published
        # annotation, in both directions. Pinning it is what stops "the
        # reference was adjusted to fit" from being possible: any change to
        # either side turns this red.
        published = {
            r: want
            for lo, hi, want, _what, _rlo, _rhi in ss_ref.CRAMBIN_1CRN_SS
            for r in range(lo, hi + 1)
        }
        off = {r: (published[r], state_of.get(r, "absent")) for r in published
               if r in state_of and state_of[r] != published[r]}
        check(
            "the only published segment this module contradicts is residue 32",
            off == {32: ("sheet", "coil")},
            f"annotated segment residues answered differently: {off or 'none'}; "
            "want residue 32 alone, called coil because 1CRN's only inter-strand "
            "bonds are 35->1 and 33->3, so 32 is neither bridged nor interior "
            "to a bridged stretch",
        )
        # ...and the other direction: residues outside every published segment
        # that are nevertheless called structured. Two of them, one at the
        # C-terminal end of each alpha helix, because the 4-turn 20->16 and
        # 31->27 each still form.
        extra = sorted(
            r for r, s in state_of.items()
            if s != "coil" and r not in published
        )
        check(
            "the only structured residues outside the published segments are "
            "the two helix termini",
            extra == [20, 31],
            f"structured residues the annotation does not cover: {extra}; want "
            "[20, 31] (one past the C-terminal end of each alpha helix)",
        )

        # The residues the annotation calls loop. A classifier that called the
        # whole protein helix would still pass every check above: the segments
        # are a small fraction of a 46-residue chain, and "everything is helix"
        # agrees with all of them.
        loop_resids = [r for r in ss_ref.CRAMBIN_1CRN_LOOPS if r in state_of]
        loop_states = [state_of[r] for r in loop_resids]
        check(
            "1CRN's loop residues are not called helix or sheet",
            all(s == "coil" for s in loop_states),
            f"{len(loop_resids)} residues {loop_resids}: "
            + "".join(s[0].upper() for s in loop_states)
            + " (want all C)",
        )

        # The ideal helix: no reference structure behind it at all. Every
        # residue is helical by construction, so "coil" here can only mean the
        # classifier is wrong.
        helix_text = ss_ref.ideal_alpha_helix(16)
        geom = ss_ref.ideal_helix_geometry(helix_text)
        helix_shape_ok = all(
            lo <= geom[key] <= hi for key, (lo, hi) in ss_ref.IDEAL_HELIX_RANGES.items()
        )
        check(
            "the ideal-helix fixture really is an ideal alpha helix",
            helix_shape_ok,
            ", ".join(f"{k} {geom[k]:.2f}" for k in ss_ref.IDEAL_HELIX_RANGES)
            + " A; want "
            + ", ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in ss_ref.IDEAL_HELIX_RANGES.items()),
        )
        ideal = parse_structure(helix_text, "ideal_alpha_helix")
        ideal_trace = ideal.backbone()
        ideal_states = secondary_structure(ideal_trace, ideal.atoms)
        # The two termini have no turn to be part of -- residue 1 cannot donate
        # a hydrogen bond and residue 16's would come from a residue 17 that
        # does not exist -- so the interior is the claim, not all sixteen.
        interior = ideal_states[1:-1]
        check(
            "every residue of an ideal alpha helix is called helix",
            bool(interior) and all(s == "helix" for s in interior),
            f"{len(interior)} interior residues: "
            + f"{sum(1 for s in interior if s == 'helix')} helix, "
            + f"termini {ideal_states[0]}/{ideal_states[-1]}",
        )

        # ---- the class has to reach the pixels, measured not asserted.
        # Every render below uses the *same* trace and the *same*
        # `geometry.ribbon`; only the states differ, so any difference between
        # two frames is the classification and nothing else.
        renderer = ss_render.OffscreenRenderer.open(900, 700)
        if renderer is None:
            reason = f"no offscreen OpenGL context ({ss_render.LAST_PROBLEM})"
            skip("a helix-only and a sheet-only ribbon are different pictures", reason)
            skip("a helix-only and a coil-only ribbon are different pictures", reason)
            skip("the receptor's own ribbon is not the all-coil one", reason)
            skip("the classified ribbon carries helix colour and the all-sheet "
                 "one does not", reason)
        else:
            from opendocking.workbench import Camera

            xyz = np.asarray([a.xyz for a in prot.atoms], np.float64)
            centre = xyz.mean(axis=0)
            cam = Camera()
            cam.center = centre.astype(np.float32)
            cam.distance = float(np.linalg.norm(xyz - centre, axis=1).max()) * 2.6

            n_res = len(prot_trace)
            # The offscreen render is only evidence about the product if it
            # draws the product's mesh. `MoleculeView.backbone_ribbon` is the
            # call the viewport makes; `secondary_render.ribbon_for_states`
            # re-derives the side vectors so that an arbitrary list of states
            # can be drawn. The two have to agree vertex for vertex, or every
            # pixel number below is about a picture the workbench does not show.
            product_mesh = None
            why = ""
            try:
                from opendocking.workbench import MoleculeView

                product_mesh = MoleculeView.from_pdbqt(
                    receptor_path, receptor_path.stem, COLOR_RECEPTOR, 0.30,
                    role="receptor",
                ).backbone_ribbon()
            except Exception as exc:  # noqa: BLE001 - reported as a FAIL below
                why = f"the product's own ribbon could not be built: {exc}"
            mine = ss_render.ribbon_for_states(prot, prot_states)
            check(
                "the offscreen render draws the same mesh the viewport draws",
                product_mesh is not None
                and len(product_mesh.indices) == len(mine.indices)
                and np.array_equal(
                    np.asarray(product_mesh.positions, np.float32),
                    np.asarray(mine.positions, np.float32),
                )
                and np.array_equal(
                    np.asarray(product_mesh.colors, np.float32),
                    np.asarray(mine.colors, np.float32),
                ),
                why
                or f"{len(product_mesh.indices)} triangles from the product, "
                f"{len(mine.indices)} from the offscreen path; positions and "
                "per-vertex colours compared exactly",
            )
            # Rendered empty. The backdrop is a *gradient*, so this -- not a
            # modal colour -- is what the ribbon's own pixels are measured
            # against; see `secondary_render.ribbon_mask`.
            blank = renderer.render([], cam, 900, 700)
            frames = {}
            for label, states in (
                ("real", prot_states),
                ("helix", ss_render.flat_states(n_res, "helix")),
                ("sheet", ss_render.flat_states(n_res, "sheet")),
                ("coil", ss_render.flat_states(n_res, "coil")),
            ):
                frames[label] = renderer.render(
                    [(ss_render.ribbon_for_states(prot, states), 1.0)], cam, 900, 700
                )
            renderer.close()

            hx_sh = differing_pixels(frames["helix"], frames["sheet"])
            check(
                "a helix-only and a sheet-only ribbon are different pictures",
                hx_sh > 5000,
                f"{hx_sh} differing px of {frames['helix'].shape[0] * frames['helix'].shape[1]}",
            )
            hx_co = differing_pixels(frames["helix"], frames["coil"])
            check(
                "a helix-only and a coil-only ribbon are different pictures",
                hx_co > 5000,
                f"{hx_co} differing px",
            )
            real_co = differing_pixels(frames["real"], frames["coil"])
            check(
                "the receptor's own ribbon is not the all-coil one",
                real_co > 5000,
                f"{real_co} differing px between the classified ribbon and an "
                "all-coil one; 0 would mean the classification changed nothing",
            )
            # Colour, because colour is what a viewer reads first, and measured
            # two-sided so a threshold that merely counts bright pixels cannot
            # pass it. At a 20-level red-over-green margin on this receptor:
            # all-helix 0.727, all-sheet 0.000, all-coil 0.000, classified
            # 0.487.
            mask_real = ss_render.ribbon_mask(frames["real"], blank)
            mask_sheet = ss_render.ribbon_mask(frames["sheet"], blank)
            frac_real = ss_render.red_excess(frames["real"], mask_real)
            frac_sheet = ss_render.red_excess(frames["sheet"], mask_sheet)
            check(
                "the classified ribbon carries helix colour and the all-sheet "
                "one does not",
                frac_real > 0.15 and frac_sheet < 0.02,
                f"{frac_real:.1%} of the classified ribbon's "
                f"{int(mask_real.sum())} px are 20+ levels redder than green, "
                f"against {frac_sheet:.1%} of the all-sheet ribbon's "
                f"{int(mask_sheet.sum())} px",
            )

    # ------------------------------------------------- 11. interactions panel
    section("11. pose/receptor interactions")
    win4 = MainWindow(receptor=rec)
    win4.resize(1100, 800)
    win4.show()
    QTest.qWaitForWindowExposed(win4, 5000)
    app.processEvents()

    check("no receptor interactions are claimed before a pose is loaded",
          not win4.viewport.contacts_valid or not win4.viewport.contacts,
          f"contacts={len(win4.viewport.contacts)} valid={win4.viewport.contacts_valid}")
    check("the panel says so instead of showing an empty table",
          "load" in win4.lbl_contacts.text().lower()
          or win4.contact_table.rowCount() == 0,
          win4.lbl_contacts.text())

    # The example receptor is a synthetic blob whose residues are all called
    # REC, so it has contacts but no residue to attribute them to. That is a
    # different situation from "nothing is touching", and the two must not look
    # the same: the headline still counts the contacts, and the table is empty
    # rather than filled with the word "REC" seven times.
    if poses.is_file():
        win4.load_structure(poses, "poses")
        app.processEvents()
        QTest.qWait(200)
        app.processEvents()
        anonymous = win4.viewport.contacts
        check("a receptor with no residue names still reports its contacts",
              len(anonymous) > 0, f"{len(anonymous)}")
        check("but no residue is invented for them",
              all(not c.partner_residue for c in anonymous),
              f"{sum(1 for c in anonymous if c.partner_residue)} named")
        check("and the table stays empty rather than filling with placeholders",
              win4.contact_table.rowCount() == 0,
              f"{win4.contact_table.rowCount()} rows")
        check("the headline still counts the contacts it found",
              str(len(anonymous)) in win4.lbl_contacts.text(),
              win4.lbl_contacts.text())
        dispose_window(app, win4)
    else:
        skip("contacts on a receptor with no residue names", f"{poses.name} missing")

    crambin_pose = EXAMPLES / "crambin_pose.pdbqt"
    if not (crambin.is_file() and crambin_pose.is_file()):
        skip("interactions on a docked pose",
             "crambin and its pose are not both present")
    else:
        win5 = MainWindow(receptor=crambin, poses=crambin_pose)
        win5.resize(1100, 800)
        win5.show()
        QTest.qWaitForWindowExposed(win5, 5000)
        app.processEvents()
        QTest.qWait(200)
        app.processEvents()

        found = win5.viewport.contacts
        check("a pose in a real protein produces contacts", len(found) > 0, f"{len(found)}")
        check("the viewport knows they were computed", win5.viewport.contacts_valid)
        named = {c.partner_residue for c in found if c.partner_residue}
        check("and they are attributed to real residues", len(named) > 0,
              f"{len(named)} residues")
        check("the table has one row per residue involved",
              win5.contact_table.rowCount() == len(named),
              f"{win5.contact_table.rowCount()} rows vs {len(named)} residues")
        check("the headline counts add up to the contacts shown",
              str(len(found)) in win5.lbl_contacts.text(),
              win5.lbl_contacts.text())
        check("at least one hydrogen bond is found in a docked pose",
              any(c.kind == "hbond" for c in found),
              f"{sum(1 for c in found if c.kind == 'hbond')} h-bonds")
        # The detail is not decoration. This check failed once with no
        # explanation at all, which is the worst possible failure: a flaky
        # check with nothing to look at is a check you learn to ignore, and an
        # ignored check is a check that can be wrong forever. So it reports
        # the state it is asserting about, and the state the viewport is in,
        # because a difference between those two is where the answer is.
        check(
            "the interaction toggle is on by default",
            win5.cb_contacts.isChecked(),
            f"checkbox {win5.cb_contacts.isChecked()}, "
            f"viewport.show_contacts {win5.viewport.show_contacts}, "
            f"{len(found)} contacts, "
            f"pocket thread running "
            f"{getattr(win5, '_pocket_thread', None) is not None and win5._pocket_thread.isRunning()}",
        )

        from opendocking.workbench.geometry import dashed_segments

        pose_mol = next(m for m in win5.viewport.molecules if m.role == "pose")
        rec_mol = next(m for m in win5.viewport.molecules if m.role == "receptor")
        segs, group = dashed_segments(
            np.asarray([pose_mol.coords[c.self_index] for c in found], np.float32),
            np.asarray([rec_mol.coords[c.partner_index] for c in found], np.float32),
        )
        check("one contact becomes more than one dash",
              len(segs) > len(found), f"{len(found)} contacts -> {len(segs)} dashes")
        check("every dash maps back to a real contact",
              len(group) == len(segs) and int(group.max()) < len(found),
              f"{len(group)} groups over {len(found)} contacts")

        # Selecting a row has to move the camera onto that contact. The claim
        # is about where the camera looks, so this compares the *centre*; a
        # comparison of distances would pass for free whenever the new radius
        # happened to equal the old one, which it did here -- 16.2 A both ways.
        before_centre = np.array(win5.viewport.camera.center, np.float64)
        before_dist = win5.viewport.camera.distance
        # A window that has framed nothing has no share to report, and this is
        # the last point in the section before anything is selected. It is
        # worth asserting because this window *has* been framed: the pose was
        # framed at load time, and the camera is centred and standoff. That
        # framing is a different path -- `focus_selection` -> `selection_target`
        # -- which returns a `FramingTarget` rather than a `PairFraming`, and
        # `subject_fill` lives only on the pair record. So the empty state
        # here is not "no framing has run", it is "no *pair* framing has run",
        # and a bar that printed a number here would be inventing one.
        check(
            "a window that has framed no pair reports no subject share, even "
            "though its pose was framed at load",
            win5.viewport._framed_subject_note() == "",
            f"note reads {win5.viewport._framed_subject_note()!r} with the "
            f"camera at {float(win5.viewport.camera.distance):.1f} A and "
            f"{len(found)} contact(s) computed. There is nothing to have "
            f"measured, so the line says nothing rather than saying 0%.",
        )
        win5.contact_table.selectRow(0)
        app.processEvents()
        after_centre = np.array(win5.viewport.camera.center, np.float64)
        shifted = float(np.linalg.norm(after_centre - before_centre))
        check("selecting a residue moves the camera centre onto it", shifted > 0.5,
              f"moved {shifted:.2f} A")
        # **This used to be `distance <= before_dist`, and it was an accident of
        # the 3-D view's old 640 px floor rather than a fact about residues.**
        #
        # It compares the minimum distance at which the *pose* fits with the
        # minimum distance at which a *different, larger* subject fits: every
        # contact of the row's residue plus a 2 nm context shell. Which of the
        # two is larger depends on which axis binds, and the frame's aspect
        # decides that. Measured on this machine, same window, same row, same
        # contacts, viewport the only variable:
        #
        #   aspect 1.333 (the 640x480 floor)  pose 42.4 A  residue 22.6 A  in
        #   aspect 0.500 (the 240 px floor)  pose 42.4 A  residue 48.5 A  out
        #
        # The pose's figure does not move at all, because the pose fit is bound
        # vertically and the vertical fov did not change. The residue's figure
        # more than doubles, because that fit is bound horizontally and the
        # frame is now 0.5 as wide as it is tall. So the old predicate was
        # asserting "a residue's contacts are a smaller thing than a pose",
        # which is not true in general and was only ever true because the view
        # was wide enough for the pose to dominate.
        #
        # **What replaces it is the framing's own claim, read off the record
        # the rule returned** rather than against another subject: the subject
        # reaches no more than the whole half-frame (`subject_fill <= 1`, which
        # is what "fits" means for `drawn_fill`) and the eye is outside the
        # drawn envelope (`envelope_slack >= 0`, which is what that number is).
        # Both are definitional, so neither is a constant tuned to make this
        # pass, and neither depends on the frame's shape. At the two aspects
        # above the record reads 0.34 / 12.3 and 0.26 / 38.2: the narrow frame
        # stands the camera *further* off the molecule, which is the safe
        # direction, and the check now says so.
        _res_fr_early = win5.viewport._last_pair_framing
        check(
            "and the framing it produced is one the projection says fits",
            _res_fr_early is not None
            and float(_res_fr_early.subject_fill) <= 1.0 + 1e-9
            and float(_res_fr_early.envelope_slack) >= -1e-9,
            f"pose {before_dist:.1f} A -> residue "
            f"{win5.viewport.camera.distance:.1f} A at aspect "
            f"{win5.viewport.width() / max(win5.viewport.height(), 1):.3f} "
            f"({win5.viewport.width()}x{win5.viewport.height()} px); the residue "
            f"framing's own record: subject reaches "
            f"{float(_res_fr_early.subject_fill) * 100:.1f}% of the half-frame "
            f"(1.0 would be touching both edges) and the eye is "
            f"{float(_res_fr_early.envelope_slack):.1f} A outside the drawn "
            f"envelope of {int(_res_fr_early.context_atoms)} context atom(s)"
            if _res_fr_early is not None else
            f"pose {before_dist:.1f} A -> residue "
            f"{win5.viewport.camera.distance:.1f} A, and no framing record came "
            f"back to check"
        )
        top = (win5.contact_table.item(0, 0).text()
               if win5.contact_table.rowCount() else "")
        check("the status bar names the residue it centred on",
              bool(top) and top in win5.statusBar().currentMessage(),
              win5.statusBar().currentMessage())

        # The residue route's own report, and the number is read off the record
        # the framing returned rather than matched by keyword, so this cannot
        # pass on a bar that carried the words and no figure.
        _res_txt = win5.statusBar().currentMessage()
        _res_fr = win5.viewport._last_pair_framing
        _res_want = ("" if _res_fr is None else
                     f"  [framed subject reaches "
                     f"{float(_res_fr.subject_fill) * 100:.2f}% of the "
                     f"half-frame at {float(_res_fr.distance):.1f} A]")
        check(
            "selecting a residue reports the share of the frame its contacts "
            "reach, and the figure is the framing's own",
            _res_fr is not None and _res_want in _res_txt,
            f"bar read {_res_txt!r}"
            + ("" if _res_fr is None else
               f"; the record says {float(_res_fr.subject_fill) * 100:.2f}% of "
               f"the half-frame at {float(_res_fr.distance):.1f} A")
            + (". This is the second of the two routes that reach "
               "`focus_target`: the residue route frames every contact of the "
               f"row's residue ({len([c for c in found if c.partner_residue == top])} "
               "atom(s) here) rather than the two of a pair, so its share is a "
               "different subject at the same camera rule. The pair route is "
               "swept over every row in M13."),
        )

        members = [c for c in found if c.partner_residue == top]
        if members:
            pts = np.asarray(
                [pose_mol.coords[c.self_index] for c in members]
                + [rec_mol.coords[c.partner_index] for c in members], np.float32
            )
            off = float(np.linalg.norm(pts.mean(axis=0)
                                        - after_centre.astype(np.float32)))
            check("the camera is over that residue's contacts, not the protein's middle",
                  off < 1.0, f"{off:.2f} A from the contact centroid")
            # The check above asks whether the move *arrived near* the contact,
            # which a move that stopped 0.9 A short of it also satisfies. This
            # one asks whether it arrived *exactly* where the framing said.
            # Between them a transition that ends off-target fails: the 1.0 A
            # check is the loose one a user would not notice failing, and this is
            # the one that catches a tween whose last step is not its endpoint.
            #
            # The expected framing is **the product's own**, from
            # `framing_selection.focus_target`, with the drawn radii read through
            # the viewport's own accessor. That is a change from re-deriving the
            # formula here, and it is worth being honest about which claim is
            # weaker as a result: a check that calls the function agrees with the
            # implementation *by construction*, so it no longer catches a rule
            # that is wrong -- it catches the move not happening, or being
            # overwritten afterwards. The rule itself is held by section 14's
            # three surround checks, which recompute its terms from the live
            # camera and the representation in use, and by a synthetic subject
            # there where the fit is exact by construction. Splitting it this way
            # is deliberate: one claim per check, each measured by the instrument
            # that can actually fail.
            from opendocking.workbench import framing_selection as _fsr

            _subj_r = np.concatenate([
                np.asarray(win5.viewport._framing_radii(pose_mol), np.float64)[
                    [c.self_index for c in members]],
                np.asarray(win5.viewport._framing_radii(rec_mol), np.float64)[
                    [c.partner_index for c in members]],
            ])
            _subj_xyz = pts.astype(np.float64)
            _mid_r = _subj_xyz.mean(axis=0)
            _rec_xyz = np.asarray(rec_mol.coords, np.float64).reshape(-1, 3)
            _rr = np.asarray(win5.viewport._framing_radii(rec_mol), np.float64)
            _near_r = np.linalg.norm(_rec_xyz - _mid_r, axis=1) <= _fsr.PAIR_CONTEXT_SHELL
            _r5, _u5, _f5 = win5.viewport.camera.basis()
            _want = _fsr.focus_target(
                _subj_xyz, _subj_r, _rec_xyz[_near_r], _rr[_near_r],
                _r5, _u5, _f5, fov=float(win5.viewport.camera.fov),
                aspect=max(win5.viewport.width(), 1)
                / max(win5.viewport.height(), 1),
            )
            want_c = np.asarray(_want.center, np.float64)
            want_d = float(_want.distance)
            drift = max(
                float(np.abs(np.asarray(win5.viewport.camera.center, np.float64)
                             - want_c).max()),
                abs(float(win5.viewport.camera.distance) - want_d),
            )
            # A *relative* floor, and the first version of this check used an
            # absolute 1e-6 and failed at 3.58e-06. That is not the product
            # being 3.6 milliangstroms out: `Camera.center` and the framing's
            # arithmetic are float32, and at 16.64 A one ulp is already
            # 2.0e-06, so an absolute 1e-6 floor is *below the representation's
            # own resolution* and no correct implementation could ever meet it.
            # A floor a correct answer cannot reach is not a floor.
            tol = 1e-6 * want_d
            check("and lands exactly on the framing focus_residue computed",
                  drift < tol,
                  f"the camera is {drift:.2e} A / A from the {want_d:.2f} A "
                  f"framing at {np.round(want_c, 4)} that focus_residue derives "
                  f"from the same contacts and the drawn radii of the "
                  f"representation in use ({win5.viewport.representation}), "
                  f"against a floor of {tol:.2e} A (1e-6 of the framing's own "
                  f"distance, about eight float32 ulps at that size). "
                  f"float32 at that size has an ulp of 2.0e-06, so the absolute "
                  f"1e-6 this first used was unreachable by construction; the "
                  f"1.0 A check above is the loose one a user would not notice "
                  f"failing, and this is the one that catches a transition "
                  f"stopping a thousandth of an angstrom short. The floor is a "
                  f"multiple of the answer rather than a constant, so it cannot "
                  f"be widened into meaninglessness: at this distance it is "
                  f"{tol:.1e} A, and a real off-target move is metres")

        win5.cb_contacts.setChecked(False)
        app.processEvents()
        check("unchecking interactions stops the lines being drawn",
              win5.viewport.show_contacts is False)
        win5.cb_contacts.setChecked(True)
        app.processEvents()
        check("re-checking brings them back", win5.viewport.show_contacts is True)
        dispose_window(app, win5)
    section("11c. pocket search drives the box")
    # The regression this guards is specific: loading a receptor used to put
    # the box on the receptor's centroid, which for a globular protein is
    # inside the dense core. The check is not "a box exists" -- there always
    # was one -- but "the box the user is shown came from a site, and it is
    # nowhere near the centroid it used to be handed".
    #
    # This section opens its own window on crambin rather than reusing the
    # one the rest of the script has been driving. The main window holds
    # `rec_prep.pdbqt`, a handful of atoms whose one site *is* its centroid,
    # so on that receptor "the box moved away from the centroid" cannot fail
    # and asserting it there would be asserting nothing.
    tiny = None
    pw = MainWindow()
    pw.resize(1500, 900)
    # Shown, like every other window in this file: an unshown widget has no
    # framebuffer to read, so the pixel check below would be measuring nothing
    # even on a machine that can render.
    pw.show()
    try:
        pw.load_structure(EXAMPLES / "1crn_prep.pdbqt", "receptor")
        app.processEvents()
        receptor_view = next(
            (m for m in pw.viewport.molecules if m.role == "receptor"), None
        )
        check("a receptor is loaded for the pocket search", receptor_view is not None)
        if receptor_view is None:
            raise RuntimeError("no receptor to check the pocket search against")
        rows = pw.pocket_table.rowCount()
        check("the site table is populated", rows > 0, f"{rows} rows")
        check(
            "the site label reports how many and how long, not just a dash",
            "site(s)" in pw.lbl_pockets.text(),
            f"{pw.lbl_pockets.text()!r}",
        )
        check(
            "the table has the five columns the rows are built from",
            pw.pocket_table.columnCount() == 5,
            f"{pw.pocket_table.columnCount()} columns: "
            f"{[pw.pocket_table.horizontalHeaderItem(c).text() for c in range(pw.pocket_table.columnCount())]}",
        )
        # The fifth column is the one the ranking is built from, and it was
        # added because the order looked arbitrary without it. A column that
        # is present and empty would be the same as no column, so check that
        # it carries a number on every row, not just the first.
        vol_col = pw.pocket_table.columnCount() - 1
        volumes = [pw.pocket_table.item(r, vol_col).text() for r in range(rows)]
        # "<number> Å³": split the unit off rather than trimming characters,
        # which would leave the unit's own space in the string being tested.
        def _is_volume(text):
            parts = text.split()
            return len(parts) == 2 and parts[1] == "Å³" and \
                parts[0].replace(".", "").isdigit()
        check(
            "every row reports the volume the list is ranked by",
            all(_is_volume(v) for v in volumes),
            f"{volumes[:5]}{'...' if len(volumes) > 5 else ''}",
        )
        # And the number is the one the ceiling is applied to, so a column
        # showing anything else would be a table disagreeing with the filter
        # that produced it. Duplicates are fine and expected: two sites can
        # round to the same whole cubic angstrom.
        from opendocking.workbench import pockets as _pockets  # noqa: PLC0415
        nums = [float(v.split()[0]) for v in volumes]
        check(
            "every volume shown is one the filter would have kept",
            all(0 < n <= _pockets.DEFAULT_MAX_VOLUME for n in nums),
            f"largest shown {max(nums):.0f} A3, ceiling "
            f"{_pockets.DEFAULT_MAX_VOLUME:.0f} A3",
        )
        cells = [pw.pocket_table.item(0, c).text() for c in range(pw.pocket_table.columnCount())]
        check(
            "every cell in the first row has text in it",
            all(str(t).strip() for t in cells),
            f"{cells}",
        )
        check(
            "the kind column says what the site is, in the same words everywhere",
            all(
                pw.pocket_table.item(r, 1).text() in ("groove", "sealed")
                for r in range(rows)
            ),
            f"kinds {sorted({pw.pocket_table.item(r, 1).text() for r in range(rows)})}",
        )
        check(
            "the centre column is a coordinate, not a label",
            cells[2].count("(") == 1 and cells[2].count(",") == 2,
            f"{cells[2]!r}",
        )
        check(
            "the size column is three numbers",
            cells[3].count("×") == 2,
            f"{cells[3]!r}",
        )
        lining_tooltip = pw.pocket_table.item(0, 0).toolTip() or ""
        check(
            "the lining residues are reachable, not just implied",
            "lined by" in lining_tooltip and "—" not in lining_tooltip.split("lined by")[-1],
            f"tooltip {lining_tooltip!r}",
        )

        centroid = np.asarray(receptor_view.coords, np.float32).mean(axis=0)
        box = np.asarray(pw.viewport.box_center, np.float32)
        moved = float(np.linalg.norm(box - centroid))
        check(
            "the box is not sitting on the receptor centroid",
            moved > 3.0,
            f"{moved:.2f} A from the centroid it used to be given",
        )
        check(
            "the spin boxes show the box the engine will use",
            np.allclose(
                np.array([s.value() for s in pw.center_spins], float), box, atol=0.01
            )
            and np.allclose(
                np.array([s.value() for s in pw.size_spins], float),
                np.asarray(pw.viewport.box_size, np.float32),
                atol=0.01,
            ),
            f"spins {[s.value() for s in pw.center_spins]} vs viewport {box.tolist()}",
        )
        check(
            "the box is a plausible size rather than the whole protein",
            8.0 <= float(np.min(pw.viewport.box_size)) <= 40.0,
            f"{np.round(pw.viewport.box_size, 1).tolist()}",
        )

        # Selecting a row has to move the box *and* the camera, and has to do
        # it through the spins -- setting `box_center` directly would leave
        # three stale numbers on screen describing a box that no longer exists.
        #
        # The eleven checks below need a second row to select, so they only run
        # when the search found one. The `else` branch registers a skip for each
        # of them by name, so the total is the same either way -- see
        # `SITE_SELECTION_CHECKS`. Leaving them unregistered is how this file
        # once reported 244 checks against a pin of 243.
        _before = len(results)
        if rows > 1:
            before_centre = np.array(pw.viewport.box_center, copy=True)
            before_dist = float(pw.viewport.camera.distance)
            pw.pocket_table.selectRow(1)
            app.processEvents()
            after_centre = np.array(pw.viewport.box_center, copy=True)
            check(
                "selecting a row moves the box",
                float(np.linalg.norm(after_centre - before_centre)) > 1.0,
                f"moved {float(np.linalg.norm(after_centre - before_centre)):.2f} A",
            )
            check(
                "and the spin boxes follow it",
                np.allclose(
                    np.array([s.value() for s in pw.center_spins], float),
                    after_centre,
                    atol=0.01,
                ),
                f"spins {[s.value() for s in pw.center_spins]}",
            )
            cam_off = float(
                np.linalg.norm(
                    np.asarray(pw.viewport.camera.center, np.float32) - after_centre
                )
            )
            check(
                "and the camera goes to the site it just selected",
                cam_off < 1.0 and abs(float(pw.viewport.camera.distance) - before_dist) > 1e-6,
                f"camera {before_dist:.1f} -> {float(pw.viewport.camera.distance):.1f} A, "
                f"{cam_off:.2f} A from the new box centre",
            )
            check(
                "the camera is far enough out to see the box, not inside it",
                float(pw.viewport.camera.distance)
                >= 0.9 * float(np.linalg.norm(np.asarray(pw.viewport.box_size, np.float32))),
                f"distance {float(pw.viewport.camera.distance):.1f} A, box diagonal "
                f"{float(np.linalg.norm(np.asarray(pw.viewport.box_size, np.float32))):.1f} A",
            )
            check(
                "the status bar names the site and its residues",
                "site 2" in pw.statusBar().currentMessage()
                and "lined by" in pw.statusBar().currentMessage(),
                f"{pw.statusBar().currentMessage()!r}",
            )
            arr, _ = shot(pw, "pockets_site_selected")
            # `pixel_check`, not `check`: whether `grabFramebuffer` returns
            # anything is a property of the environment, not of the workbench,
            # and a headless runner reports a blank framebuffer for every
            # pixel check in this file. Using `check` here is what made the
            # whole job fail on a machine that cannot draw -- 14 sibling
            # checks skipped correctly and this one did not.
            pixel_check(
                "the selected site is actually drawn",
                non_background(arr) > 500,
                f"{non_background(arr)} non-background pixels",
            )
            pw.pocket_table.clearSelection()
            app.processEvents()
            check(
                "clearing the selection clears the site volume, rather than "
                "leaving one site's cloud inside another's box",
                len(pw.viewport.pocket_points) == 0,
                f"{len(pw.viewport.pocket_points)} points still held",
            )
            pw.pocket_table.selectRow(1)
            app.processEvents()
            check(
                "reselecting puts it back",
                len(pw.viewport.pocket_points) > 0,
                f"{len(pw.viewport.pocket_points)} points",
            )
            # What the space-filling mode costs, measured rather than assumed.
            #
            # The expectation going in was that a CPK surface would bury the
            # site cloud: the cloud is drawn as 0.42 A points *inside* the
            # protein, and a closed van der Waals shell is between the camera
            # and anything inside it. The measurement says otherwise, and the
            # reason is in the draw order -- `_draw_pocket` runs after the
            # molecules with the depth test off, which is the only reason a site
            # inside a protein is visible at all. So the cloud's footprint is
            # the same number in every representation, and what the solid view
            # actually costs is legibility of the *protein*, not the cloud.
            #
            # Measured: 23 719 px of cloud in all six, identical to the pixel.
            # The claim pinned here is therefore the one that is true and that
            # would be worth losing: the site stays visible in every
            # representation, including the solid one. It goes red if depth
            # testing is ever enabled for the cloud, or if a representation is
            # added that fills the site -- which is the failure a user would
            # experience as "the pocket controls do nothing".
            cloud_px: dict[str, int] = {}
            repr_keep_cloud = pw.viewport.representation
            no_cloud = np.zeros((0, 3), np.float32)
            for key in REPRESENTATION_KEYS:
                pw.viewport.representation = key
                app.processEvents()
                keep_points = pw.viewport.pocket_points
                with_cloud, _ = shot(pw, f"11c_cloud_{key}")
                pw.viewport.pocket_points = no_cloud
                app.processEvents()
                without_cloud, _ = shot(pw, f"11c_nocloud_{key}")
                pw.viewport.pocket_points = keep_points
                app.processEvents()
                cloud_px[key] = (differing_pixels(with_cloud, without_cloud)
                                 if with_cloud is not None and without_cloud is not None
                                 else -1)
            pw.viewport.representation = repr_keep_cloud
            app.processEvents()
            pixel_check(
                "the site cloud stays visible in every representation, the "
                "space-filling one included",
                bool(cloud_px) and all(v > 500 for v in cloud_px.values()),
                "deleting all "
                f"{len(pw.viewport.pocket_points)} cloud points changes "
                + ", ".join(f"{k} {v} px" for k, v in cloud_px.items())
                + ". Identical in every mode because the cloud is drawn with "
                "the depth test off, which is how a site inside a protein is "
                "visible at all; a solid surface therefore does not hide the "
                "site, and the expectation that it would is what this "
                "measurement replaced",
            )
            # The toggle has to actually gate the draw, not just the checkbox.
            shown_before = pw.viewport.show_pocket
            before, _ = shot(pw, "pockets_cloud_on")
            pw.cb_pocket_volume.setChecked(False)
            app.processEvents()
            check(
                "the site-volume checkbox gates the draw",
                pw.viewport.show_pocket is False and shown_before is True,
                f"show_pocket {shown_before} -> {pw.viewport.show_pocket}",
            )
            after, _ = shot(pw, "pockets_cloud_off")
            # Through `pixel_check`, not `check`. This is the one pixel check
            # that had been calling `check` directly, so on a headless CI box
            # whose framebuffer read back blank the other fifteen skipped and
            # this one reported a failure -- a false alarm about a machine
            # that cannot render at all. `pixel_check` skips exactly when the
            # environment cannot answer and still fails, on a machine that
            # can, when toggling the checkbox changed nothing.
            pixel_check(
                "and the picture really changed, not just the flag",
                not np.array_equal(before, after),
                f"{differing_pixels(before, after)} pixels differ",
            )
            pw.cb_pocket_volume.setChecked(True)
            app.processEvents()
            check(
                "switching it back restores the cloud",
                pw.viewport.show_pocket is True
                and len(pw.viewport.pocket_points) > 0,
            )
        else:
            for _name in SITE_SELECTION_CHECKS:
                skip(
                    _name,
                    f"the pocket search reported {rows} site(s) for this fixture, "
                    "so there is no second row to select and this run cannot "
                    "witness it; the checks are not absent, they are unanswered",
                )
        # Registered in both branches, and the only guard on the list above
        # staying in step with the block. It runs on a machine with more than
        # one site -- the ordinary case -- and compares what the block recorded
        # against the names the `else` branch would skip, so a check renamed
        # inside the block without updating `SITE_SELECTION_CHECKS` shows up
        # here instead of quietly costing a machine with a single site eleven
        # results.
        _recorded = tuple(r[1] for r in results[_before:])
        check(
            "every site-selection check is listed for the skip branch too",
            rows <= 1 or _recorded == SITE_SELECTION_CHECKS,
            f"{len(_recorded)} recorded, {len(SITE_SELECTION_CHECKS)} listed"
            + ("" if rows <= 1 else f"; unmatched: {set(_recorded) ^ set(SITE_SELECTION_CHECKS)}"),
        )
    finally:
        dispose_window(app, pw)

    # A structure with no enclosed site must say so rather than quietly
    # leaving a box somewhere and calling it a suggestion. One atom: the grid
    # around it is all solvent and the flood reaches everything, so there is
    # genuinely nothing to report -- unlike `rec_prep.pdbqt`, which does have
    # a site and made this check pass for the wrong reason.
    #
    # Written to a temporary file, not into `examples/`. A four-line fixture
    # that a check regenerates on every run does not belong in a release tree,
    # and that is how example directories quietly fill up.
    with tempfile.TemporaryDirectory() as tmpdir:
        tiny = Path(tmpdir) / "one_atom.pdbqt"
        tiny.write_text(
            "ROOT\n"
            "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00"
            "     C.3\n"
            "ENDROOT\n"
            "TORSDOF 0\n",
            encoding="utf-8",
        )
        empty = MainWindow()
        empty.resize(600, 400)
        try:
            empty.load_structure(tiny, "receptor")
            app.processEvents()
            check(
                "a structure with no site reports that, and does not invent one",
                empty.pocket_table.rowCount() == 0
                and "no enclosed site" in empty.lbl_pockets.text(),
                f"{empty.pocket_table.rowCount()} rows, "
                f"label {empty.lbl_pockets.text()!r}",
            )
        finally:
            dispose_window(app, empty)

    section("11e. selecting a pose frames it, and the move lands on target")
    # The other camera-move checks in this file ask *whether* the camera went
    # somewhere. This one asks *where it ended up*, because a transition is a
    # different claim from a position: a move that stops 0.9 A short of its
    # target, or that eases towards it and never reaches it, passes every
    # "did the camera move" check there is and leaves the user somewhere the
    # product never meant to put them.
    #
    # Three separate claims, and they fail separately:
    #   * the move happens at all, and it *moves* rather than snapping -- a
    #     camera that snaps satisfies "the camera ended on target" perfectly;
    #   * it ends exactly on the target, to 1e-6, not merely near it;
    #   * the target is the framing of the pose and its contacts, which is the
    #     point of the change and is checked in
    #     `scripts/framing_selection_check.py` against the pixels as well.
    # Reuses `win`, the window the suite has had open since the start, and that
    # is not a convenience. The first version of this block opened its own
    # `MainWindow`, and by this point in the run the machine had already created
    # and released nine GL contexts; the eleventh took the process down with
    # 0xC0000409 and no traceback -- which is the failure `_draw_pocket`'s
    # docstring already records for an exception inside a paint event, and which
    # looks exactly like a driver crash rather than like a test. One more
    # window is not worth a whole section; the existing one has nine poses in
    # it and nothing here needs a second receptor.
    sel = win
    if not _run_camera_move(sel, app):
        for _name in POSE_FRAMING_CHECKS:
            skip(
                _name,
                "the camera move did not run to completion on this machine, so "
                "the frames along it are unknown; the checks are not absent, "
                "they are unanswered",
            )
    else:
        plan = sel._last_selection_plan
        cam = sel.viewport.camera
        drift = max(
            float(np.abs(np.asarray(cam.center, np.float64)
                         - np.asarray(plan.target.center, np.float64)).max()),
            abs(float(cam.distance) - float(plan.target.distance)),
        )
        check(POSE_FRAMING_CHECKS[0], drift < 1e-6,
              f"row {plan_row}: the move ran "
              f"{plan.start_distance:.2f} -> {plan.target.distance:.2f} A in "
              f"{plan.steps} steps, and the camera finished {drift:.2e} A / A "
              f"from the target, against a floor of 1e-6. A move that eases "
              f"towards its target without reaching it ends a thousandth of an "
              f"angstrom short forever, and a check written against the target "
              f"then has to tolerate a floating-point remainder to stay "
              f"reliable on a machine it was not written on")
        seen = plan_distances
        # **Distinct values, not a strictly decreasing sequence.** The sampler
        # polls every 5 ms and the tween ticks every 20 ms, so most frames record
        # the value that is already there; the first version required a strict
        # decrease between consecutive samples and failed on 30 of its 32 pairs
        # for that reason alone. The claim being made is "the camera took a path
        # to get there rather than jumping", and the number that carries it is
        # how many different values it was observed at: a snap yields one or two
        # whatever the sample rate.
        distinct = sorted(set(round(d, 6) for d in seen), reverse=True)
        # Monotone in *either* direction: selecting a pose can move the camera in
        # or out, and a move that eases towards a target it is already near
        # moves out. Requiring a decrease would have failed on a selection whose
        # target is further away than where the camera happened to be, which is
        # a correct product state and not a wandering camera.
        falling = all(seen[i] >= seen[i + 1] for i in range(len(seen) - 1))
        rising = all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))
        check(POSE_FRAMING_CHECKS[1],
              len(distinct) >= plan.steps - 2 and (falling or rising),
              f"{len(seen)} frames sampled, {len(distinct)} of them distinct, over "
              f"{plan.steps} steps: {' -> '.join(f'{d:.2f}' for d in distinct)} A"
              + ("" if (falling or rising)
                 else f" -- NOT monotone in either direction: {seen}, and a move "
                      f"that wanders is not a move that eases")
              + f". The sampler polls at 5 ms and the tween ticks at 20 ms, so "
                f"most samples repeat the value already there; a camera that "
                f"snapped would show one or two distinct values however often it "
                f"was sampled, and {len(distinct)} is what rules that out. This "
                f"is the assertion that the product *moves* rather than cutting, "
                f"and it is the half of 'a transition' that a "
                f"final-framing-only check cannot see"
              + ". It deliberately says nothing about the endpoint: the sampler "
                "stops appending the moment `moving` goes false, so its last "
                "sample is one tick *short* of the target by construction. An "
                "earlier version asserted `arrived` here and failed at 2.1e-03 A "
                "for exactly that reason -- the instrument cannot see the thing "
                "it was asked to confirm. The endpoint is asserted exactly, to "
                "1e-6, by the check above and again by "
                "`scripts/framing_selection_check.py` section 4")
        check(POSE_FRAMING_CHECKS[2],
              plan.target.pose_atoms == len(_pose_coords(sel))
              and plan.target.partner_atoms == len(_partner_coords(sel))
              and plan.target.pose_atoms > 0,
              f"the target frames {plan.target.pose_atoms} pose atoms and "
              f"{plan.target.partner_atoms} contacting receptor atoms over "
              f"{len(plan.target.partner_residues)} residues "
              f"{list(plan.target.partner_residues)}; the 22 A search box and "
              f"the site volume are in neither. The overlays are lines drawn "
              f"between exactly those two sets, so this is what makes them "
              f"readable rather than a separate nicety")
        check(POSE_FRAMING_CHECKS[3], sel.viewport.pocket_opacity == 0.0,
              f"the site cloud's opacity is {sel.viewport.pocket_opacity} with "
              f"a pose selected, against {POCKET_OPACITY_PLAIN} when a site is. "
              f"Framing the pose is no use if the subject of the picture is "
              f"behind something nine times its own size, and opacity scales "
              f"brightness rather than footprint, so a dimmer cloud covers the "
              f"same pixels -- the cloud has to leave, not fade")

    # `sel` above was an alias for `win`, and this is the last read of either.
    # Every later `win` token in the file is a function parameter -- the nested
    # `_grab_live` and `_cam_off_midpoint`, and the module-level helpers
    # `_pose_coords`, `_settle_camera`, `_run_camera_move` and the rest -- never
    # the outer name. An AST scope walk over the parsed file finds zero reads of
    # the outer `win` past this line.
    #
    # So `win`, opened at the top of `main()`, had been left open: never closed,
    # never destroyed, still a visible top-level for the remaining ~4,800 lines
    # of this file. The keyboard section's census counted two `MainWindow`s
    # because of it -- measured, tagged `win` against the section's own `kw` --
    # and it held a GL context through eleven windows opened after this point.
    #
    # This is the gate's window, not a leak in the product. `run()` in
    # `opendocking/workbench/app.py` is the only place the product constructs a
    # `MainWindow`; the gate never calls `run()`; and `MainWindow.closeEvent`
    # ends in `super().closeEvent(event)` without ever calling
    # `event.ignore()`, so a closed window really is hidden. One window per
    # `run()`, held for the process, is the ordinary Qt lifetime, not a leak.
    #
    # `destroy=True`, and here rather than at the end of the run, because this
    # is not the last window: the `destroy=False` that `tw` gets below is
    # documented as the trade for the *last* window in a run, and eleven more
    # are opened after this point, each of which wants the context. `win`'s
    # search thread is long gone -- `_thread` is None the moment a search
    # finishes -- which is the precondition `dispose_window` warns about.
    # Measured in `target/winleak_20261003/census_probe.py`: this exact destroy,
    # mid-run and after a completed search, leaves `sip.isdeleted(win) is True`
    # and takes the census from two windows to one at both ratios.
    dispose_window(app, win)

    section("11d. the pocket search does not block the window")
    # Measured, not guessed: the search takes 0.07 s for crambin's 382 atoms,
    # 2.0 s for streptavidin and 12.0 s for haemoglobin's 4779. Run on the GUI
    # thread that last one froze the window for twelve seconds on load.
    #
    # The proof that it is off-thread is an ordering, not a duration: the call
    # returns while the work is still outstanding, so the label still reads
    # "searching…" and the button is still disabled at the moment control
    # comes back. A threshold on elapsed time would be a threshold on how fast
    # the runner is; this holds on any machine, and it fails immediately if
    # someone puts the search back inline.
    from opendocking.workbench import MoleculeView  # noqa: PLC0415

    axis = np.arange(-18.0, 18.1, 1.6)
    blob = np.asarray(
        [
            (x, y, z)
            for x in axis
            for y in axis
            for z in axis
            if (x * x + y * y + z * z) ** 0.5 % 3.0 < 1.2
        ],
        np.float32,
    )
    tw = MainWindow()
    tw.resize(800, 600)
    tw.show()
    try:
        tw.viewport.molecules = [
            MoleculeView(
                name="synthetic", coords=blob,
                elements=["C"] * len(blob),
                color=(0.6, 0.7, 0.9), radius=0.15, role="receptor",
            )
        ]
        check(
            "the fixture is big enough for the answer to be observable",
            len(blob) > 2000,
            f"{len(blob)} atoms",
        )
        tw.find_pockets()
        mid_label = tw.lbl_pockets.text()
        mid_enabled = tw.btn_pockets.isEnabled()
        check(
            "find_pockets() returns before the search has finished",
            "searching" in mid_label.lower(),
            f"label immediately after the call: {mid_label!r}",
        )
        check(
            "and the button is disabled meanwhile, so a second search cannot start",
            not mid_enabled,
            f"enabled={mid_enabled}",
        )
        # Pump the loop and ask whether the work queued for it came out.
        #
        # The old version counted `processEvents()` calls in five 0.2 s windows
        # and asserted the best one reached 10 turns. That is a claim about the
        # machine, not about the window: the loop is not free, it repaints a
        # real window, so how many turns fit in a fixed span depends on this
        # processor and on what else is running. It read 29 turns in one run and
        # 7 in another on the same machine, and in a quiet baseline the five
        # windows read 1, 20, 21, 24, 29 -- so even best-of-five was riding a
        # first window that happened to catch the window mid-repaint. A check
        # that reports "the event loop is blocked" when the truth is "this
        # runner is busy" is the same defect as the pixel guard that reported a
        # blank framebuffer as a broken viewport.
        #
        # So the question is now discrete: post N events at a receiver, give the
        # loop control, and see whether they were delivered. A loop that cannot
        # turn delivers none of them at any budget, because an undelivered event
        # stays queued; a loop that can turn delivers all N in one pass, because
        # `processEvents()` drains the queue it is handed rather than sampling
        # it. There is no duration left in the criterion to tune against the
        # machine, and the pump count is reported without being compared to
        # anything.
        was_running = tw._pocket_thread is not None and tw._pocket_thread.isRunning()
        delivered, pumps = loop_turns(app)
        still_running = tw._pocket_thread is not None and tw._pocket_thread.isRunning()
        print(f"  events delivered to the GUI thread: {delivered} of 8, in "
              f"{pumps} pump(s) of processEvents()")
        # A third thing this measurement could be about instead: the search
        # finishing early, which would leave the loop turning over nothing.
        # That is a different claim from "turning while the search runs", so it
        # gets its own check rather than being folded into the threshold -- and
        # on a machine fast enough to finish the search inside the second of
        # pumping below, that run genuinely cannot witness a window blocked by
        # a running search, so it is a skip with a reason rather than a failure
        # about a defect that is not there. See `search_overlap_verdict`.
        overlap = search_overlap_verdict(was_running, still_running)
        if overlap == "PASS":
            check(
                "the search really was still running while the loop turned",
                True,
                f"thread running at the start: {was_running}, at the end: "
                f"{still_running}, so the {delivered} events delivered came out "
                "while it was in flight",
            )
        elif overlap == "SKIP":
            skip(
                "the search really was still running while the loop turned",
                "the search completed before the event loop turned, so this run "
                "cannot witness a window blocked by a running search; the "
                f"{delivered} events delivered came out over a finished search "
                "and are not evidence either way",
            )
        else:
            check(
                "the search really was still running while the loop turned",
                False,
                f"thread running at the start: {was_running}, at the end: "
                f"{still_running}; the search was already over when control came "
                "back, so find_pockets() ran it inline",
            )
        check(
            "the event loop delivers the work queued for it while the search runs",
            delivered == 8,
            f"{delivered} of 8 queued events came back out of the loop, in "
            f"{pumps} pump(s) of processEvents(). This is a count of deliveries, "
            "not turns per second: a loop that cannot turn delivers 0 however "
            "long it is given, and a loop that can turn delivers all 8 in one "
            "pass, so the number is a property of the code and not of how busy "
            "this machine is. The old version compared the best of five 0.2 s "
            "windows against 10 turns, and read 29 once and 7 another time on "
            "this same machine",
        )
        check(*_verify_loop_turns_guard())
        check(*_verify_blocked_window_guard())
        wait_until = time.perf_counter() + 120
        while time.perf_counter() < wait_until and tw._pocket_thread is not None \
                and tw._pocket_thread.isRunning():
            app.processEvents()
            time.sleep(0.005)
        for _ in range(20):
            app.processEvents()
            time.sleep(0.01)
        check(
            "the result arrives and the button comes back",
            "site(s)" in tw.lbl_pockets.text() and tw.btn_pockets.isEnabled(),
            f"label {tw.lbl_pockets.text()!r}, {tw.pocket_table.rowCount()} rows",
        )
    finally:
        # Closing with a search in flight used to abort the process with
        # "QThread: Destroyed while thread is still running". The search is
        # waited for above, so this is a window that *has* run a thread rather
        # than one running one now -- and destroying its GL widget while that
        # QThread is still an object fail-fasts here, so it is closed and left
        # for process exit instead of being forced down.
        t_close = time.perf_counter()
        dispose_window(app, tw, destroy=False)
        check(
            "closing the window during a search does not abort",
            time.perf_counter() - t_close < 20.0,
            f"closed in {time.perf_counter() - t_close:.2f} s",
        )

    # ------------------------------------------------- section 12: the crash guard
    section("12. an unhandled exception in a slot is reported, and exits non-zero")
    # A fault in a signal handler is invisible from here: the exception crosses
    # the Qt dispatch, the process ends with `0xC0000409`, and stderr is empty.
    # Every other section in this file can only say a run was green; this one
    # says what happens when it is not, and it asks the question through the
    # product's own entry point rather than through a stand-in for it.
    #
    # The children are run as files under `%TEMP%`, not with `-c`, so a
    # traceback names a real path and a line number that means something -- and
    # so the expected line can be derived from the source the child was given,
    # independently of what the hook reports. A guard that read the line back out
    # of the hook's own output would agree with a hook that reported nothing.
    crash_dir = Path(tempfile.mkdtemp(prefix="odw_crash_"))
    slot_child = crash_dir / "slot_fault.py"
    slot_child.write_text(_SLOT_FAULT_CHILD, encoding="utf-8", newline="\n")
    check_child = crash_dir / "check_ab.py"
    check_child.write_text(_CHECK_AB_CHILD, encoding="utf-8", newline="\n")
    # The line in the child that raises, found by reading the file that was just
    # written. `+1` because enumerate starts at 1 and an editor counts the same.
    raise_line = next(
        n for n, text in enumerate(slot_child.read_text(encoding="utf-8").splitlines(), 1)
        if text.strip() == 'raise ValueError("injected: the term map could not be keyed")'
    )

    nohook = _crash_child(slot_child, ["nohook"])
    plain = _crash_child(slot_child, ["plain"])
    hooked = _crash_child(slot_child, ["hook"])
    chained = _crash_child(slot_child, ["chained"])
    caught = _crash_child(slot_child, ["caught"])
    clean = _crash_child(slot_child, ["clean"])

    check(
        "with the hook removed, the fault does not come back as this project's "
        "failure code",
        nohook[0] != crashguard.EXIT_UNHANDLED,
        f"exit {nohook[0]} (0x{nohook[0] & 0xFFFFFFFF:08X}) with "
        f"{len(nohook[2])} bytes on stderr, against the shipped "
        f"{crashguard.EXIT_UNHANDLED}. The number is quoted rather than pinned: "
        f"it is what this PyQt6 does at the boundary, and the claim is only that "
        f"it is not the code this project defines for a failure",
    )
    check(
        "and the shape that was refused would have called the same fault a pass: "
        "a hook that prints and returns ends at 0",
        plain[0] == 0 and "Traceback (most recent call last)" in plain[2],
        f"a print-only hook gave exit {plain[0]} with a traceback on stderr. "
        f"This is why `crashguard` also ends the process, and why the check "
        f"below is about the exit code rather than about the log",
    )
    check(
        "with the hook in place, the process ends with the project's unhandled "
        "code: not 0, and not the code the unhooked run died with",
        hooked[0] == crashguard.EXIT_UNHANDLED
        and hooked[0] not in (0, nohook[0]),
        f"exit {hooked[0]}, against 0 and {nohook[0]} (0x{nohook[0] & 0xFFFFFFFF:08X}) "
        f"for the same fault without the hook",
    )
    check(
        "and the traceback names the file the exception was raised in",
        str(slot_child) in hooked[2],
        f"the header must name {slot_child}; stderr carries it"
        if str(slot_child) in hooked[2]
        else f"{slot_child} does not appear in the {len(hooked[2])} bytes of stderr",
    )
    check(
        "and it names the line, and the line is the one the child's own source "
        "raises on",
        f"{slot_child.name}:{raise_line}" in hooked[2]
        and f"line {raise_line}" in hooked[2],
        f"the `raise ValueError` is on line {raise_line} of the child as written, "
        f"and the hook has to report that number rather than one of its own",
    )
    check(
        "and it names the exception and what it said",
        "ValueError" in hooked[2]
        and "injected: the term map could not be keyed" in hooked[2],
        "the type and the message, both from the exception itself",
    )
    # The code the product's `run()` handed back, read as a number rather than as
    # a substring. A substring test here is vacuous under the one mutation that
    # matters most: with `EXIT_UNHANDLED` set to 0, "run() returned 0" contains
    # "run() returned 0" and the check passes while the process is reporting a
    # crash as a success. So the number is compared, and `!= 0` is asserted in
    # its own right rather than only as a difference from the constant.
    run_line = next(
        (ln.strip() for ln in hooked[2].splitlines()
         if ln.strip().startswith("run() returned ")),
        "",
    )
    run_code = run_line.partition("run() returned ")[2].strip()
    check(
        "and the 70 came back out of the product's own `run()`, so the window "
        "closed the way it closes for any other exit",
        run_code == str(crashguard.EXIT_UNHANDLED) and run_code != "0",
        f"the child reported {run_line!r}. The hook asked the event loop for the "
        f"code and `run()` returned it. An `os._exit` would have killed the "
        f"process here and this line would not exist, which is the difference "
        f"between the two branches in `crashguard`",
    )
    check(
        "and the failure is recorded where a caller can ask for it, with the "
        "same file and line",
        _record_of(hooked[2])[0].get("where") == f"{slot_child}:{raise_line}"
        and _record_of(hooked[2])[0].get("exception") == "ValueError",
        f"`crashguard.failures()` gave {_record_of(hooked[2])}, read as JSON "
        f"rather than as a repr. The log is for the person reading it; this is "
        f"what a program would ask for",
    )
    check(
        "a chained exception prints the chain and names the original, not just "
        "the re-raise",
        "KeyError" in chained[2]
        and "injected: the term-map key went missing" in chained[2]
        and "the original failure was KeyError" in chained[2],
        f"exit {chained[0]}; both tracebacks, plus the line naming the bottom of "
        f"the chain, which is the one with the actual cause",
    )
    check(
        "an exception the slot catches itself never reaches the hook, and the "
        "run ends 0 as it should",
        caught[0] == 0
        and "failures=()" in caught[2]
        and "slot caught it" in caught[2],
        f"exit {caught[0]}, `crashguard.failures()` empty, and the slot's own "
        f"message on stderr. The hook is only reached by exceptions that escape, "
        f"so the product's `except TermsUnavailable` and friends are untouched",
    )
    check(
        "a run where nothing goes wrong is untouched: exit 0, nothing recorded, "
        "and not one line from the hook",
        clean[0] == 0
        and "failures=()" in clean[2]
        and "workbench:" not in clean[2],
        f"exit {clean[0]}, `crashguard.failures()` empty, and no 'workbench:' "
        f"line among the {len(clean[2])} bytes of stderr",
    )

    # The clean-run comparison. Two runs of the same diagnostic in the same tree,
    # differing only in whether the hook is `sys.excepthook` at the time, and
    # both asked for the same answer. A hook that changed a clean run's output
    # would be a hook able to make a failure look like a pass, so this is a
    # check and not a note.
    ab_on = _crash_child(check_child, ["on"], platform=None)
    ab_off = _crash_child(check_child, ["off"], platform=None)
    payload_on = _verdict_of(ab_on[1])
    payload_off = _verdict_of(ab_off[1])
    stripped_on = _without_timings(payload_on)
    stripped_off = _without_timings(payload_off)
    differing = sorted(
        k for k in set(stripped_on) | set(stripped_off)
        if stripped_on.get(k) != stripped_off.get(k)
    )
    # **Where inside those fields the disagreement is.** `differing` names a
    # top-level key, and one top-level key can hide two opposite faults: a
    # digest that moved because the tree was rewritten between the two probes,
    # and a label that moved because the two runs took different routes. The
    # check is right to fail either way -- two runs of one command in one tree
    # are supposed to agree -- but the owner of the red is not the same, so the
    # detail has to say which rather than assert a cause it did not compute.
    moved_leaves = _moving_leaves(stripped_on, stripped_off)
    digest_leaves = [leaf for leaf, _a, _b in moved_leaves
                     if _is_digest_leaf(leaf)]
    other_leaves = [leaf for leaf, _a, _b in moved_leaves
                    if not _is_digest_leaf(leaf)]

    # **Which of the child's four `ready` states each run reported**, in the
    # detail and not only on failure. This check is the one place in the file
    # where two `--check` runs have to agree, and when they do not, "the bound
    # I set for myself lapsed" and "Qt never gave the widget a context" are
    # different faults with different owners and different fixes.
    #
    # The transcript used to say only that the two runs disagreed, and the
    # payload held the sentence that would have said which. Observed once: run 2
    # of this round gave exit 5 with the hook and 0 without, and reducing that
    # to a state cost seven further A/B pairs and a loaded run. A detail string
    # that cannot name the state makes the next occurrence cost the same again.
    def _ready_of(payload):
        w = (payload or {}).get("widget") or {}
        return {
            "ready": w.get("ready"),
            "bound_expired": w.get("bound_expired"),
            "child_killed": w.get("child_killed"),
            "widget_valid": w.get("widget_valid"),
            "waited_s": round(float(w.get("waited") or 0.0), 2),
        }

    check(
        "`odgui --check --json` gives the same verdict and the same exit code "
        "with the hook in place as with it removed",
        ab_on[0] == ab_off[0]
        and payload_on is not None
        and payload_off is not None
        and not differing,
        f"exit {ab_on[0]} with the hook and {ab_off[0]} without; every field "
        f"equal except the wall-clock fields {list(TIMING_KEYS)} (raw.waited, "
        f"widget.show_seconds, widget.waited), which are timings of two separate "
        f"runs and cannot be equal. Fields that differed: "
        f"{differing if differing else 'none'}. "
        + (f"The leaves that moved, named rather than summarised: "
           f"digest(s) {digest_leaves or 'none'}, other(s) "
           f"{other_leaves or 'none'}. A digest or a byte count that moves "
           f"between two probes of the same command is the *file* being "
           f"rewritten underneath them -- the two runs each answered correctly "
           f"about a different tree, and the red belongs to whoever was saving, "
           f"not to the hook. Anything else moving is the two runs genuinely "
           f"disagreeing, which is a launcher or product question. "
           if differing else
           "Every non-timing leaf agreed, digest included, so the two runs "
           "measured the same file. ")
        + f"The two widget stages, which is where a difference in this check "
        f"has to come from: with the hook {_ready_of(payload_on)}; without "
        f"{_ready_of(payload_off)}",
    )

    # In-process, and reversible: the two branches of `_stop` without paying for
    # a process per branch. `_hard_exit` is replaced *before* the first probe,
    # not after: the very first probe is deliberately made outside any
    # `event_loop()`, which is the branch that ends the process, and a first
    # version of this block that called the hook before the replacement killed
    # the run itself with exit 70 and no summary. That is the branch working
    # exactly as written, which is a poor way to learn it.
    #
    # `_records` and `_hard_exit` are put back afterwards, so the module is as
    # the next section would find it.
    import io as _io

    from opendocking.workbench import crashguard as _cg

    saved_records = list(_cg._records)
    saved_hard_exit = _cg._hard_exit
    hard_codes: list[int] = []
    try:
        _cg._hard_exit = hard_codes.append
        # Probe one: no loop of ours, so the code has to be handed over by
        # ending the process. The recorder stands in for `os._exit`.
        _cg._records[:] = []
        with contextlib.redirect_stderr(_io.StringIO()):
            _cg._hook(ValueError, ValueError("in-process probe, no loop"), None)
        outside_codes = list(hard_codes)
        recorded = _cg.failure()
        zeroed = _cg.finalise(0)
        other = _cg.finalise(3)
        # Probe two: inside `event_loop()`, where the QApplication has to be
        # asked for the code and the hard exit must not be reached.
        _cg._records[:] = []
        hard_codes.clear()
        with contextlib.redirect_stderr(_io.StringIO()), crashguard.event_loop():
            _cg._hook(ValueError, ValueError("in-process probe, loop owned"), None)
        inside_hard = bool(hard_codes)
    finally:
        _cg._records[:] = saved_records
        _cg._hard_exit = saved_hard_exit
    check(
        "a recorded failure cannot be reported as 0, and a clean code is left "
        "alone",
        zeroed == crashguard.EXIT_UNHANDLED and zeroed != 0 and other == 3,
        f"after the hook ran, `crashguard.failure()` gave "
        f"{recorded and recorded['exception']!r} and `finalise(0)` gave {zeroed} "
        f"while `finalise(3)` gave {other}. This is the backstop under the "
        f"event-loop branch: even if the loop returned 0 the process still ends "
        f"non-zero",
    )
    check(
        "the hook hands the code to the loop when this process owns it, and ends "
        "the process itself when it does not",
        (not inside_hard)
        and outside_codes == [crashguard.EXIT_UNHANDLED]
        and outside_codes != [0],
        f"inside `event_loop()` the hard exit was "
        f"{'reached' if inside_hard else 'not reached'}; outside it, the code "
        f"handed over was {outside_codes}. Calling `QCoreApplication.exit()` "
        f"while a check script pumps `processEvents()` is harmless here -- "
        f"nothing in this file calls `exec()`, so there is no loop for the "
        f"pending code to end -- and the alternative, trusting "
        f"`QThread.loopLevel()`, is the silent pass: it is above zero for a "
        f"`processEvents()` pump as well as for `exec()`",
    )

    # ---------------------------------------------------------- 13. keyboard
    # The workbench is an instrument for looking at nine poses one at a time,
    # and until this section that loop needed a mouse. Everything here is judged
    # against the window this file has already been driving all along, so "a key
    # gives the same pose a click gives" is a comparison of two routes into one
    # pipeline rather than two implementations of a feature.
    section("13. the window without a mouse")
    from opendocking.workbench import keys as keymap

    # Its own window, with its own run. `kw` from section 7b has been closed by
    # the time this section runs -- reaching into it raised on a deleted C++
    # object -- and a keyboard section riding on another section's window would
    # measure a different fixture from the one it reports.
    kw = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                    ligand=EXAMPLES / "biotin_prep.pdbqt")
    kw.resize(1500, 900)
    kw.show()
    # The census below prints this name, so the one window it reports is
    # identifiable rather than merely countable.
    kw._gate_name = "kw"
    for _ in range(20):
        app.processEvents()
    # **Every key in this section is a `WindowShortcut`, and that context fires
    # only while its window is the *active window*.** The focus owner is a
    # different question -- check 6 measures that one, and check 6's failure
    # mode is a spin box taking a key for itself, which is declared and
    # intended. The active window is not declared anywhere and was never
    # established here, so the whole section was resting on ambient desktop
    # state: at dpr 2.5 five of its nine checks went red in one run, every one
    # of them a key that "did nothing", and the reason was that this window was
    # not the foreground one.
    #
    # What takes it is a fact about the run, not about this section: a window
    # shown later takes the status, and `isActiveWindow()` is False for the
    # earlier one while it holds. Measured directly -- with a second window up,
    # the first reports False and the second True, and `activateWindow()` takes
    # it back. So the section asks for what it needs instead of inheriting it.
    # This is not papering over a product defect: a reader who clicks the
    # window is doing precisely this, and the product's own handlers carry no
    # focus gate at all.
    kw.activateWindow()
    kw.raise_()
    for _ in range(8):
        app.processEvents()

    def _top_levels() -> list:
        """Which top-level windows exist, and which one holds the foreground.

        Printed by the key check below, because "the key did nothing" and "some
        other window was in front" are indistinguishable from the reading
        alone, and only the second is worth anybody's time.
        """
        out = []
        for w in QtWidgets.QApplication.topLevelWidgets():
            if not w.isVisible():
                continue
            name = type(w).__name__
            title = w.windowTitle()
            if title:
                name = f"{name} {title!r}"
            # Which window this is, not just that it is one. Two of them print
            # as the same string, and at dpr 2.5 Qt clamps both to 770x533, so
            # geometry cannot break the tie either -- which is what left the
            # second entry in this list unattributed until the two windows were
            # traced by hand. `_gate_name` is set at creation on the windows
            # that can be up at the same time; the rest are disposed before the
            # next one opens, and print untagged, correctly.
            tag = getattr(w, "_gate_name", None)
            if tag:
                name = f"{name} <{tag}>"
            out.append(f"{name}{' [ACTIVE]' if w.isActiveWindow() else ''}")
        return out

    _levels_at_open = _top_levels()
    for _spin in kw.size_spins:
        _spin.setValue(22.0)
    for _spin in kw.center_spins:
        _spin.setValue(0.0)
    kw.sp_exhaust.setValue(2)
    kw.sp_seed.setValue(20260901)
    app.processEvents()
    kw.btn_dock.click()
    for _ in range(1200):
        app.processEvents()
        if kw.btn_dock.isEnabled():
            break
        QTest.qWait(25)
    for _ in range(8):
        app.processEvents()
        QTest.qWait(40)
    # The term maps, so the breakdown and the verdict in the equivalence
    # comparison are the ones a user has after a second rather than the empty
    # ones a first selection shows while they are still being built.
    for _ in range(600):
        app.processEvents()
        if kw._terms_pose_index == kw._current_pose and kw._current_pose >= 0:
            break
        QTest.qWait(20)

    def contexts_labels(pairs) -> list:
        return [label for label, _target in pairs]

    def _key_state() -> dict:
        """Everything a reader of the window can see about the selection.

        Sampled, not inferred, and the list is the brief's own: the pose, the
        row and the column (the highlighted cell), both readouts, the verdict
        header *and* its rows, the terms header and row count, the camera, the
        two view toggles a key can flip, the pose's own colour, role and radius,
        and the contact overlay's residue summary and row count. Seventeen
        fields, each of which a second implementation of "select the pose" could
        plausibly get to the right answer on all but one -- and a check that
        samples only the pose and the energy is a check that cannot tell the
        difference between the two implementations it exists to compare.
        """
        return {
            "pose": kw._current_pose,
            "row": kw.pose_table.currentRow(),
            "column": kw.pose_table.currentColumn(),
            "energy": kw.lbl_energy.text(),
            "rmsd": kw.lbl_rmsd.text(),
            "verdict": kw.lbl_verdict_head.text(),
            "terms": kw.lbl_terms_pose.text(),
            "terms_rows": kw.tbl_terms.rowCount(),
            "camera": (round(float(kw.viewport.camera.yaw), 6),
                       round(float(kw.viewport.camera.pitch), 6),
                       round(float(kw.viewport.camera.distance), 4)),
            "contacts": kw.cb_contacts.isChecked(),
            "cloud": kw.cb_pocket_volume.isChecked(),
            # The pose's own appearance. "Same colour scheme" is this: the
            # selection path rebuilds the view and carries the role across, and
            # a route that skipped either would still have matched on the pose,
            # the camera and the labels.
            "pose_colour": tuple(round(float(c), 6) for c in kw._pose_view.color),
            "pose_role": kw._pose_view.role,
            "pose_radius": round(float(kw._pose_view.radius), 6),
            # The overlay's content, not its checkbox: a key route that left the
            # lines checked but the residue list empty would pass on the flag.
            "contacts_residues": kw.lbl_contacts.text(),
            "contact_rows": kw.contact_table.rowCount(),
            # The verdict's rows, not only its header.
            "verdict_states": [kw.tbl_verdict.item(r, 1).text()
                               for r in range(kw.tbl_verdict.rowCount())],
        }

    def _settle(ms: int = 350, cap_ms: int = 6000) -> None:
        """Wait on the clock, and then until the camera is at rest.

        Both halves are needed and the second one is the one that was missing.
        The framing move is a wall-clock `QTimer`, so `processEvents()` does not
        advance it, and a fixed wait samples the camera *mid-tween* -- yaw and
        pitch matched exactly while the distance was still 0.18 A from its
        target, which is not a difference between two paths but a difference
        between two moments. `Viewport.moving()` is the same "at rest" question
        section 11e asks, asked here of every sample.
        """
        waited = 0
        while waited < cap_ms:
            QTest.qWait(20)
            for _ in range(4):
                app.processEvents()
            waited += 20
            if waited >= ms and not kw.viewport.moving:
                return

    def _select_row(row: int) -> None:
        kw.pose_table.setCurrentCell(row, 0)
        _settle(250)

    if kw.pose_table.rowCount() < 2:
        for _name in (
            "every declared key is installed, and no key is installed twice",
            "a key press gives the same pose a click gives",
            "the steppers beat the table's own column navigation, and the "
            "table's own keys still work",
            "the first and the last pose are handled and said out loud",
            "the map lists every key, grouped, and the groups are the ones the "
            "table declares",
            "the map's claim about context matches what the keys do",
            "a letter key does not also move the table's own current cell",
            "the camera keys and the drag are the same arithmetic",
            "the keyboard captures exist, and this platform could have written "
            "them",
        ):
            skip(_name, f"the window has {kw.pose_table.rowCount()} pose rows, "
                        f"so there is nothing to step through")
    else:
        RIGHT = next(s for s in keymap.SHORTCUTS if s.keys == ("Right",))
        LEFT = next(s for s in keymap.SHORTCUTS if s.keys == ("Left",))

        # 1. the declared keys are the installed keys, and none is declared
        # twice. The duplicate is modelled on the table rather than by
        # installing a second real shortcut: the mistake this catches is somebody
        # adding a key that one already has, which is a fact about the table, and
        # a check that needed a live second shortcut to see it would be a check
        # about Qt rather than about the keys.
        def _keys_of(specs) -> dict:
            out: dict = {}
            for spec in specs:
                if not spec.handler:
                    continue
                for key in spec.keys:
                    out.setdefault(key, []).append(spec.handler)
            return out

        installed = {
            short.key().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
            for short in kw._key_shorts
        }
        declared = _keys_of(keymap.SHORTCUTS)
        shipped_dupes = {k: v for k, v in declared.items() if len(v) > 1}
        import dataclasses as _dc
        doubled = list(keymap.SHORTCUTS) + [
            _dc.replace(keymap.SHORTCUTS[0], handler="_key_rotate")
        ]
        mutated_dupes = {k: v for k, v in _keys_of(doubled).items() if len(v) > 1}
        check(
            "every declared key is installed, and no key is installed twice",
            set(declared) == installed and not shipped_dupes
            and bool(mutated_dupes)
            and set(declared) - {"Up", "Down"} == installed,
            f"the table declares {sorted(declared)} and the window has "
            f"{sorted(installed)}; the keys with no handler are "
            f"{[s.label for s in keymap.SHORTCUTS if not s.handler]}, which are "
            f"the pose table's own. Shipped duplicates: "
            f"{shipped_dupes or 'none'}. The same table with one key given a "
            f"second handler reports {mutated_dupes or 'nothing'}, which is what "
            f"this check is for: two handlers for one key is how one of them wins "
            f"by accident",
        )

        # 2. a key press gives the same pose a click gives
        _select_row(2)
        _select_row(5)
        by_click = _key_state()
        _select_row(2)
        QTest.keyClick(kw, QtCore.Qt.Key.Key_Right)
        _settle()
        QTest.keyClick(kw, QtCore.Qt.Key.Key_Right)
        _settle()
        QTest.keyClick(kw, QtCore.Qt.Key.Key_Right)
        _settle(500)
        by_key = _key_state()
        differ = {k: (by_click[k], by_key[k]) for k in by_click
                  if by_click[k] != by_key[k]}
        # The mutation goes into `_key_bound`, which is what `_key_activated`
        # calls. Assigning to the attribute `kw._key_step_pose` would be a
        # mutation that proves nothing: the dispatcher cached the *bound method*
        # when the window was built, so the patch would never be reached. The
        # first version of this check did exactly that and still reported a
        # rejection, because the "mutated" run was the shipped code.
        live_step = kw._key_bound["_key_step_pose"]

        def backwards_step(spec):
            """The mutation: a stepper that walks the wrong way."""
            if spec.keys[0] == "Right":
                row = kw.pose_table.currentRow()
                kw.pose_table.setCurrentCell(max(0, row - 1), 0)

        kw._key_bound["_key_step_pose"] = backwards_step
        _select_row(2)
        QTest.keyClick(kw, QtCore.Qt.Key.Key_Right)
        _settle()
        mutated = _key_state()
        kw._key_bound["_key_step_pose"] = live_step
        # Back to row 2 and the same three presses, so the restore is compared
        # with the same destination rather than with a state one row away: an
        # earlier version pressed once and compared a row-3 state with a row-5
        # one, which is a difference between two destinations rather than
        # between two steppers.
        _select_row(2)
        for _ in range(3):
            QTest.keyClick(kw, QtCore.Qt.Key.Key_Right)
            _settle()
        restored = _key_state()
        check(
            "a key press gives the same pose a click gives",
            not differ and mutated != by_click and restored == by_key,
            f"a click on row 5 and three Right presses compared field by field "
            f"over {len(by_click)} fields: "
            f"{'identical' if not differ else f'DIFFERENT: {differ}'}. The step is "
            f"`setCurrentCell`, which is the click's own route, so this is a "
            f"property of the path and not a coincidence. With the stepper "
            f"mutated to walk backwards the same press gave row "
            f"{mutated['row']} against {by_click['row']}, and restoring it gave "
            f"{restored['row']}",
        )

        # 3. the level: window shortcuts beat the table's own column movement
        def _press(target, key, settle_ms=300):
            before = _key_state()
            target.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            app.processEvents()
            QTest.keyClick(target, key)
            _settle(settle_ms)
            return before, _key_state()

        _select_row(2)
        b, a = _press(kw.pose_table, QtCore.Qt.Key.Key_Right)
        right_moved_row = a["row"] == b["row"] + 1 and a["column"] == 0
        b, a = _press(kw.pose_table, QtCore.Qt.Key.Key_Up)
        up_still_works = a["pose"] == b["pose"] - 1
        b, a = _press(kw.viewport, QtCore.Qt.Key.Key_Right)
        viewport_works = a["pose"] == b["pose"] + 1
        b, a = _press(kw.center_spins[0], QtCore.Qt.Key.Key_Right)
        spin_swallows = a["pose"] == b["pose"]
        # The mutation: the same table at widget level, which is the one that
        # loses to whatever has focus.
        live_ctx = QtCore.Qt.ShortcutContext.WindowShortcut
        for short in kw._key_shorts:
            short.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        try:
            b, a = _press(kw.viewport, QtCore.Qt.Key.Key_Right)
            widget_level_loses = a["pose"] == b["pose"]
            b, a = _press(kw.pose_table, QtCore.Qt.Key.Key_Right)
            widget_level_table_column = a["column"] != 0 or a["row"] == b["row"]
        finally:
            for short in kw._key_shorts:
                short.setContext(live_ctx)
        _select_row(2)
        check(
            "the steppers beat the table's own column navigation, and the "
            "table's own keys still work",
            right_moved_row and up_still_works and viewport_works
            and spin_swallows and widget_level_loses
            and widget_level_table_column,
            f"measured with the keys delivered to the focused widget: Right on "
            f"the focused table moved the row to {a['row'] if False else 'the next row'} "
            f"and left the column at 0 ({right_moved_row}); Up still moved the "
            f"pose up ({up_still_works}), so the table's own key is untouched; "
            f"Right with the 3D view focused moved the pose "
            f"({viewport_works}); a focused spin box swallowed both "
            f"({spin_swallows}), which is a line edit taking the key for its own "
            f"job. Dropped to widget level the same key does nothing with the "
            f"view focused ({widget_level_loses}) and hands the table its column "
            f"back ({widget_level_table_column}) -- which is why the window level "
            f"is the one that is installed",
        )

        # 4. the edges
        last = kw.pose_table.rowCount() - 1
        _select_row(last)
        before_edge = kw.pose_table.currentRow()
        QTest.keyClick(kw, QtCore.Qt.Key.Key_Right)
        _settle(300)
        at_last = kw.pose_table.currentRow()
        said_last = kw.statusBar().currentMessage()
        _select_row(0)
        QTest.keyClick(kw, QtCore.Qt.Key.Key_Left)
        _settle(300)
        at_first = kw.pose_table.currentRow()
        said_first = kw.statusBar().currentMessage()
        edges_say = ("last" in said_last and "first" in said_first)
        # The mutation: wrapping instead of clamping, which is the behaviour that
        # makes the ends of the list indistinguishable from the middle.
        live_edge = kw._keys_step_enabled

        def wrapping(spec):
            # An instance attribute, so it is called as `self._keys_step_enabled(
            # spec)` with one argument -- not bound, and not given a `self` it
            # would never be passed.
            count = kw.pose_table.rowCount()
            if count == 0:
                return live_edge(kw, spec)
            row = max(0, min(count - 1, kw.pose_table.currentRow()))
            target = (row + kw._key_step_direction(spec)) % count
            kw.pose_table.setCurrentCell(target, 0)
            return True, ""

        kw._keys_step_enabled = wrapping
        try:
            _select_row(last)
            QTest.keyClick(kw, QtCore.Qt.Key.Key_Right)
            _settle(400)
            wrapped_to = kw.pose_table.currentRow()
        finally:
            del kw._keys_step_enabled
        _select_row(last)
        map_row = next(r for r in keymap.rows(kw) if r["keys"] == ("Right",))
        check(
            "the first and the last pose are handled and said out loud",
            at_last == before_edge and at_first == 0 and edges_say
            and not map_row["available"] and map_row["reason"]
            and wrapped_to != at_last,
            f"Right on the last row left it on row {at_last} and said "
            f"{said_last!r}; Left on the first row left it on {at_first} and said "
            f"{said_first!r}; the map's own Right row is unavailable with "
            f"{map_row['reason']!r}. Clamping rather than wrapping is the "
            f"decision: wrapping would have moved it to row {wrapped_to}, which "
            f"is how a reader stops being able to tell the end of the list from "
            f"the middle of it",
        )

        # 5. the map is generated from the table
        html = keymap.map_html(kw, kw._keys_pose_summary())
        missing = sorted({k for s in keymap.SHORTCUTS for k in s.keys
                          if f"<b>{k}</b>" not in html})
        groups = [g for g in keymap.GROUP_ORDER
                  if any(r["group"] == g for r in keymap.rows(kw))]
        live_map = keymap.map_html

        def map_without_up(html_ignored, summary_ignored=""):
            return html.replace("<b>Up</b>", "<b>Up/Down</b>").replace(
                "<b>Down</b>", "")

        keymap.map_html = map_without_up
        try:
            mutated_map = keymap.map_html(kw, "")
        finally:
            keymap.map_html = live_map
        restored_map = keymap.map_html(kw, "")
        check(
            "the map lists every key, grouped, and the groups are the ones the "
            "table declares",
            not missing and groups == list(keymap.GROUP_ORDER)
            and "<b>Down</b>" not in mutated_map
            and "<b>Down</b>" in restored_map,
            f"every declared key appears in the map (missing: {missing or 'none'}) "
            f"and the groups drawn are {groups}, in the table's order. The map "
            f"and the table are the same source, so a key cannot exist without "
            f"being documented; with the renderer mutated to drop the table's own "
            f"Down row, '<b>Down</b>' is "
            f"{'present' if '<b>Down</b>' in mutated_map else 'gone'} and it is "
            f"{'back' if '<b>Down</b>' in restored_map else 'still gone'}",
        )

        # 6. the map's context claim, against the measurement. This is the check
        # that keeps the map from over-claiming: the heading used to say
        # "Always", and the spin box measurement makes that false.
        contexts = [("pose table", kw.pose_table), ("3D view", kw.viewport),
                    ("window", kw), ("display combo", kw.cmb_representation),
                    ("centre-x spin", kw.center_spins[0]),
                    ("seed spin", kw.sp_seed)]
        swallowed, acted = [], []
        for label, target in contexts:
            _select_row(2)
            b, a = _press(target, QtCore.Qt.Key.Key_C, 200)
            (acted if a["contacts"] != b["contacts"] else swallowed).append(label)
        live_title = keymap.GROUP_ALWAYS_TITLE
        html_now = keymap.map_html(kw, kw._keys_pose_summary())
        keymap.GROUP_ALWAYS_TITLE = "Always"
        try:
            html_lie = keymap.map_html(kw, kw._keys_pose_summary())
        finally:
            keymap.GROUP_ALWAYS_TITLE = live_title
        html_back = keymap.map_html(kw, kw._keys_pose_summary())
        spins_are_the_swallowed = sorted(swallowed) == sorted(
            n for n in contexts_labels(contexts) if "spin" in n
        )
        claim_ok = (bool(swallowed) and spins_are_the_swallowed
                    and bool(keymap.EXCEPTION_CLASS_NAMES)
                    and live_title in html_now
                    and "numeric field" in html_now
                    and live_title not in html_lie)
        check(
            "the map's claim about context matches what the keys do",
            claim_ok and live_title in html_back,
            f"pressing C with each of {len(contexts)} focus targets: acted "
            f"{acted}, swallowed {swallowed}. The declared exception is "
            f"{list(keymap.EXCEPTION_CLASS_NAMES)} and the swallowed set is "
            f"exactly the spin boxes ({spins_are_the_swallowed}), so the map's "
            f"heading -- {live_title!r} -- is true rather than convenient. With "
            f"the heading mutated back to 'Always' the same measurement leaves "
            f"the map saying something false, and with it restored the map says "
            f"it again",
        )

        # 7. a letter key does not also move the table's cell
        moved = []
        for key in (QtCore.Qt.Key.Key_A, QtCore.Qt.Key.Key_D,
                    QtCore.Qt.Key.Key_W, QtCore.Qt.Key.Key_S,
                    QtCore.Qt.Key.Key_F, QtCore.Qt.Key.Key_C,
                    QtCore.Qt.Key.Key_V, QtCore.Qt.Key.Key_M):
            _select_row(2)
            before = (kw.pose_table.currentRow(), kw.pose_table.currentColumn(),
                      kw._current_pose)
            QTest.keyClick(kw.pose_table, key)
            _settle(150)
            after = (kw.pose_table.currentRow(), kw.pose_table.currentColumn(),
                     kw._current_pose)
            if before[1] != after[1] or before[0] != after[0]:
                moved.append((key, before, after))
        check(
            "a letter key does not also move the table's own current cell",
            not moved,
            f"all eight letter keys pressed with the pose table focused: the "
            f"table's own current cell moved {len(moved)} times. A QTableWidget "
            f"does type-ahead on printable characters, so a window shortcut and "
            f"the table's own search can both answer one keypress; here they "
            f"did not, and this is the line that says so",
        )

        # 8. the camera keys and the drag are one method
        #
        # **The focus this check needs is now established, and said out loud.**
        # It was not, and that is a defect in the check rather than a mystery.
        # Check 6 above ends by calling `_press(kw.sp_seed, ...)`, and `_press`
        # begins with `target.setFocus(...)`, so the last thing it does is hand
        # focus to a `QSpinBox`. Checks 7 and 8 then press keys with bare
        # `QTest.keyClick`, which sends an event and does not move focus, so this
        # check inherited the spin box. A `QShortcut` fires only when the window
        # is the active window and the focused widget has not taken the key, and
        # `QSpinBox` is the one exception the keymap itself declares
        # (`EXCEPTION_CLASS_NAMES`) and check 6 measures.
        #
        # So a camera reading of 0.0000 rad here is ambiguous between "the key
        # handler is broken" and "the key never arrived, because the focus was
        # somewhere this check did not choose". The first is a product defect
        # and the second is a fixture left dirty, they have opposite owners,
        # and the number alone cannot tell them apart. Taking the focus
        # explicitly removes the ambiguity; asserting it means a future run that
        # cannot get it says so instead of printing a camera number that means
        # nothing.
        def _focus_state() -> tuple:
            """`(is the window active, what has focus)`, read from Qt."""
            holder = QtWidgets.QApplication.focusWidget()
            return (bool(kw.isActiveWindow()),
                    "none" if holder is None else type(holder).__name__)

        kw.viewport.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        app.processEvents()
        focus_active, focus_owner = _focus_state()
        cam = kw.viewport
        cam.camera.yaw, cam.camera.pitch = 0.0, 0.0
        live_rotate = cam.rotate
        yaw_by_drag = 0.0
        try:
            press = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QPointF(100.0, 100.0), QtCore.QPointF(100.0, 100.0),
                QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier)
            move = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove, QtCore.QPointF(140.0, 100.0),
                QtCore.QPointF(140.0, 100.0), QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
            cam.mousePressEvent(press)
            cam.mouseMoveEvent(move)
            cam.mouseReleaseEvent(press)
            yaw_by_drag = float(cam.camera.yaw)
            cam.camera.yaw, cam.camera.pitch = 0.0, 0.0
            QTest.keyClick(kw, QtCore.Qt.Key.Key_D)
            _settle(120)
            yaw_by_key = float(cam.camera.yaw)
        finally:
            cam.rotate = live_rotate
        # The mutation: a rotate that forgets the pitch limit the drag has.
        # Assigned to the instance, so it is called as `cam.rotate(dyaw, dpitch)`
        # with two arguments and no `self` -- the same shape of mistake as the
        # edge mutation above, found the same way.
        def unclamped(dyaw, dpitch):
            cam.camera.yaw += float(dyaw)
            cam.camera.pitch = cam.camera.pitch + float(dpitch)
            cam.update()

        # The control first and on the shipped method: a step at the limit must
        # not move past it. The earlier version called the *mutated* method
        # twice and asked the two calls to disagree, which they cannot -- both
        # were the same function, so the arm passed without ever asking the
        # shipped one anything.
        cam.rotate = live_rotate
        cam.camera.pitch = 1.5
        cam.rotate(0.0, kw.ROTATE_STEP)
        shipped_held = float(cam.camera.pitch) == 1.5
        cam.rotate = unclamped
        try:
            cam.camera.pitch = 1.5
            cam.rotate(0.0, kw.ROTATE_STEP)
            mutated_escaped = float(cam.camera.pitch) > 1.5
        finally:
            cam.rotate = live_rotate
        check(
            "the camera keys and the drag are the same arithmetic",
            yaw_by_drag > 0.0 and yaw_by_key > 0.0
            and abs(yaw_by_drag - 0.4) < 1e-6
            and shipped_held and mutated_escaped
            and focus_active and focus_owner == "Viewport",
            f"a 40 px left-drag turned the camera by {yaw_by_drag:.4f} rad "
            f"(0.01 rad per pixel) and the D key by {yaw_by_key:.4f} rad "
            f"({kw.ROTATE_STEP} rad), both through `Viewport.rotate`. With that "
            f"pitch limit held on the shipped method ({shipped_held}) and was "
            f"not held on a mutated one that skips it ({mutated_escaped}), at "
            f"the same 1.5 rad the drag stops at. Two places to get a camera "
            f"limit right is how one of them goes wrong untested. The focus "
            f"this reading depends on is established here rather than "
            f"inherited: the window is the active window ({focus_active}) and "
            f"the focus owner is {focus_owner}, chosen because a `QSpinBox` is "
            f"the one class the keymap declares as taking a key for itself "
            f"({list(keymap.EXCEPTION_CLASS_NAMES)}) -- which is what check 6 "
            f"above leaves it holding. The visible top-level windows at this "
            f"point, the active one marked: {_top_levels()}",
        )

        # 9. the captures, and whether this platform could have written them.
        # The same precondition the panel gate uses, for the same reason: a
        # capture whose every glyph is an empty box is not weak evidence, it is
        # not evidence -- and the platform can be asked, where the picture
        # cannot. It is deliberately not an ink measurement: tofu lays down
        # *fewer* ink pixels than real glyphs while having about twice as many
        # disconnected strokes, so an ink test scores an unreadable capture as
        # the better rendering.
        _families = QtGui.QFontDatabase.families()
        _dir = Path(__file__).resolve().parent.parent / "dist" / "keyboard"
        _dir.mkdir(parents=True, exist_ok=True)
        written = []
        if _families:
            kw._key_toggle_map(None)
            _settle(200)
            kw.grab().save(str(_dir / "keymap_open.png"))
            written.append("keymap_open.png")
            _select_row(0)
            QTest.keyClick(kw, QtCore.Qt.Key.Key_Left)
            _settle(250)
            kw._keys_refresh_map()
            kw.grab().save(str(_dir / "keymap_at_first_pose.png"))
            written.append("keymap_at_first_pose.png")
            kw._keys_close()
            _settle(120)
        check(
            "the keyboard captures exist, and this platform could have written "
            "them",
            len(written) == 2
            and all((_dir / _n).is_file() for _n in written)
            and len(_families) >= 1,
            f"platform {app.platformName()!r} with {len(_families)} font "
            f"families; wrote {written or 'nothing'} under {_dir}. With no font "
            f"families every glyph would be an empty box, so nothing is written "
            f"and an earlier good capture is left alone rather than replaced",
        )

    # --------------------------------------------------------- 14. one pair
    # A residue name is not a pair. "TRP 7" does not say whether the interaction
    # went through the indole nitrogen or the backbone amide, and a table of
    # residue names cannot be walked one interaction at a time -- which is the
    # unit a chemist reasons in. This section drives the pair table the way a
    # user would, and counts pixels, because a highlight nobody can see is not a
    # highlight.
    section("14. one interaction pair at a time")
    from opendocking.workbench import contacts as contactmod
    from opendocking.workbench import keys as keymap

    # The window section 13 built and left standing: its own run, its own nine
    # poses. A section that opened a second docking run would be measuring a
    # different fixture from the one the rest of this file reports, and the one
    # thing to avoid in a pixel check is measuring something else.
    if kw.pose_table.rowCount() < 2 or not kw.viewport.contacts:
        skip(
            "the pair table, its highlight, its keys and its empty case",
            f"the keyboard window has no pose to work with: "
            f"{kw.pose_table.rowCount()} pose(s), "
            f"{len(kw.viewport.contacts)} contact(s)",
        )
    else:
        _n_rows = kw.pair_table.rowCount()
        _contacts = list(kw.viewport.contacts)
        _bad_row = ""
        _sample = []
        for _r in range(_n_rows):
            _pair = kw._pairs[_r]
            _cell_pose = kw.pair_table.item(_r, 0).text()
            _cell_rec = kw.pair_table.item(_r, 1).text()
            _c = _contacts[_pair.contact_index]
            if not _cell_pose or not _cell_rec:
                _bad_row = f"row {_r + 1} has an empty atom cell"
            elif _pair.pose_atom != _cell_pose or _pair.receptor_atom != _cell_rec:
                _bad_row = (f"row {_r + 1} says {_cell_pose} / {_cell_rec} while "
                            f"its pair object says {_pair.pose_atom} / "
                            f"{_pair.receptor_atom}")
            elif _c.partner_name not in _cell_rec:
                _bad_row = (f"row {_r + 1} does not name the atom its own line "
                            f"connects: {_c.partner_name!r} is not in "
                            f"{_cell_rec!r}")
            if len(_sample) < 2:
                _sample.append(
                    f"{_pair.pose_atom} -> {_pair.receptor_atom}, "
                    f"{_pair.kind}, {_pair.distance_text}, term="
                    f"{_pair.term or '(blank)'}"
                )
            if _bad_row:
                break
        _row_detail = (
            f"{_n_rows} rows against {len(_contacts)} contacts and "
            f"{len(kw._pairs)} pair objects. "
            + (_bad_row or
               "Every row names its pose atom, its receptor atom, and the "
               "receptor atom the dashed line for that contact index connects "
               "to.")
            + " First rows: " + " / ".join(_sample)
        )
        check(
            "the pair table has one row per contact, and each row names the two "
            "atoms its own line connects",
            (_n_rows == len(_contacts) and len(kw._pairs) == len(_contacts)
             and not _bad_row),
            _row_detail,
        )

        # -- selecting a row ------------------------------------------------
        # Judged on the row's own two cells and not on the row number: a
        # selection that highlighted "row 7" would pass every check that never
        # read a name.
        _pick = 0
        kw.pair_table.setFocus()
        kw.pair_table.setCurrentCell(_pick, 0)
        for _ in range(6):
            app.processEvents()
        _sel_pair = kw._pairs[_pick]
        _sel_contact = _contacts[_sel_pair.contact_index]
        _want = (int(_sel_contact.self_index), int(_sel_contact.partner_index))
        _got = kw.viewport.highlight_pair
        _pose_mol = next(m for m in kw.viewport.molecules if m.role == "pose")
        _rec_mol = next(m for m in kw.viewport.molecules if m.role == "receptor")
        _mid = np.asarray(
            [_pose_mol.coords[_want[0]], _rec_mol.coords[_want[1]]], np.float32
        ).mean(axis=0)
        _cam_ok = bool(np.allclose(np.asarray(kw.viewport.camera.center),
                                   _mid, atol=1e-3))
        check(
            "a selected row highlights the two atoms that row names, and the "
            "camera goes to them",
            _got == _want and _cam_ok,
            f"row {_pick + 1} reads {_sel_pair.pose_atom} -> "
            f"{_sel_pair.receptor_atom}, which is contact "
            f"{_sel_pair.contact_index} of the list the lines are drawn from: "
            f"pose atom {_want[0]} ({_sel_contact.self_name}) against receptor "
            f"atom {_want[1]} ({_sel_contact.partner_name}). "
            f"`highlight_pair` is {_got} against {_want}; the camera centre is "
            f"{np.round(np.asarray(kw.viewport.camera.center), 3).tolist()} "
            f"against the midpoint of those two atoms "
            f"{np.round(_mid, 3).tolist()}",
        )

        # -- the pixels -----------------------------------------------------
        # **In a window of its own, and that is the whole fix.** These two grabs
        # used to be taken in the shared `kw`, whose camera sections 7-13 had
        # already orbited, zoomed and reframed fourteen sections' worth. The
        # check then read 0 highlight pixels and went red, and the red meant
        # nothing: the *line* was never the problem. Measured on this platform,
        # in a window nobody had touched, the same line is 95 px at the pose's
        # own framing -- and pressing `F` takes it back to 0, because `F` moves
        # the camera somewhere the line is not. So a count taken from a window
        # with a history is a count of the history. A measurement has to happen
        # in a state you are holding still.
        #
        # A pose and a receptor, and a search to get one: the highlight needs
        # both, since a pose with no receptor in contact has no line to draw. The
        # first version of this fix used a pose file in a window with no
        # receptor, produced zero pairs, and skipped every time -- see the note
        # where the window is built.
        #
        # And the two grabs are compared **as bytes** as well as by count. If the
        # switch had not landed -- a stubbed draw, a camera that moved between
        # the grabs, a cache that returned the first frame twice -- the two
        # images would be identical, the counts would agree, and a check that
        # only compared counts would call 0 against 0 a result. Byte equality is
        # the thing that says the mutation actually did something.
        _own = None
        if not PIXELS_OK:
            skip(
                "the selected pair's highlight is visible in pixels, and nothing "
                "of it is on screen with no row selected",
                SHOT_PROBLEM or "no frame could be read from the viewport",
            )
            skip(
                "with the row still selected and the draw switched off, the "
                "highlight's pixels are gone, and they come back",
                SHOT_PROBLEM or "no frame could be read from the viewport",
            )
        else:
            # **A receptor, and the reason is not politeness.** The first version
            # of this fix built `MainWindow()` with no receptor and loaded a pose
            # file into it, on the reasoning that a highlight needs a pose. It
            # needs a pose *and something to be in contact with*: with no
            # receptor loaded there are no receptor atoms, so `find_contacts`
            # returns nothing, the pair table is empty, and the check skipped
            # every time. A check that has never been red and has never been
            # green is worth nothing, and this one would have looked like a
            # measured "not on this platform".
            _own = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                              ligand=EXAMPLES / "biotin_prep.pdbqt")
            _own.resize(1400, 850)
            _own.show()
            for _ in range(24):
                app.processEvents()
            # One search, fixed seed, then the same pose framing the load path
            # gives. Deterministic, and about a second -- the highlight needs a
            # pose with pairs, and a search is the path that demonstrably
            # produces them (28 pairs on this fixture).
            for _s in _own.size_spins:
                _s.setValue(22.0)
            for _c in _own.center_spins:
                _c.setValue(0.0)
            _own.sp_exhaust.setValue(2)
            _own.sp_seed.setValue(20260901)
            for _ in range(6):
                app.processEvents()
            _own.btn_dock.click()
            while _own._thread is not None and _own._thread.isRunning():
                app.processEvents()
                QTest.qWait(10)
            for _ in range(60):
                app.processEvents()
                QTest.qWait(10)
            _own_n = _own.pair_table.rowCount()
            # **Every name the detail strings below use is bound before the
            # branch, not inside it.** There is a path out of the branches that
            # reaches the mutation block with nothing measured -- the pair table
            # came back empty, so both `skip`s above ran -- and a detail string
            # that raises `UnboundLocalError` does not report a failed check, it
            # kills the run. This is the second time this block has died that
            # way, and the first time the lesson was written down here and then
            # not applied; so the defaults are the first statement, not the last.
            _own_on = _own_off = -1
            _own_on_arr = _own_off_arr = None
            _own_got = None
            _own_frames_differ = False
            _own_on_path = _own_off_path = "(no pair row to select)"
            if _own_n == 0:
                skip(
                    "the selected pair's highlight is visible in pixels, and "
                    "nothing of it is on screen with no row selected",
                    f"the search produced no interaction pair to highlight "
                    f"({_own_n} rows), so there is no line to count",
                )
                skip(
                    "with the row still selected and the draw switched off, the "
                    "highlight's pixels are gone, and they come back",
                    "no interaction pair to highlight, for the same reason",
                )
            else:
                _own.pair_table.setCurrentCell(0, 0)
                for _ in range(8):
                    app.processEvents()
                _settle_camera(_own, app)
                _own_on_arr, _own_on_path = shot(_own, "pair_highlight_own_on")
                _own_got = _own.viewport.highlight_pair
                _own.pair_table.clearSelection()
                _own.viewport.highlight_pair = None
                for _ in range(8):
                    app.processEvents()
                _own_off_arr, _own_off_path = shot(
                    _own, "pair_highlight_own_off")
                # The switch has to have landed, and "the counts differ" is not
                # enough to show that -- 0 against 0 also differs in the sense
                # that matters and proves nothing. The bytes are the test.
                _own_frames_differ = not np.array_equal(
                    _own_on_arr, _own_off_arr)
                # The pair's own footprint, as a difference. `_own_off` is kept
                # as a *purity* count for the record, and the two numbers side by
                # side are the argument for the difference: see `_pair_own_pixels`.
                _own_on = _pair_own_pixels(_own_on_arr, _own_off_arr)
                _own_off = _pair_highlight_pixels(_own_off_arr)
                _own_on_pure = _pair_highlight_pixels(_own_on_arr)
                # **The same pair, in every representation a reader can pick.**
                # The count above is taken in whichever mode the window opens
                # in, and a highlight that survives in only that one mode is not
                # a highlight: a reader who switched to space-filling would
                # select a row and watch nothing change, with nothing on screen
                # to say the click had landed. Measured, not asserted, because
                # it was 0 in three of the six before `_draw_pair` stopped
                # depth-testing a line whose two ends are inside the spheres it
                # joins -- and a check that only ever looked at the default mode
                # would have reported a healthy 10 px through all of that.
                def _grab_live(win):
                    win.viewport.repaint()
                    QTest.qWait(30)
                    img = win.viewport.grabFramebuffer()
                    return None if img.isNull() else img_array(img)

                _repr_keep = _own.viewport.representation
                _repr_sweep = []
                for _key in REPRESENTATION_KEYS:
                    _own.viewport.representation = _key
                    # `clearSelection()` first, and for the reason the mutation
                    # block below already records: a table does not re-emit
                    # `itemSelectionChanged` for a row that is already current,
                    # so `setCurrentCell` alone leaves `highlight_pair` at None
                    # and every mode after the first measures an empty frame.
                    # Measured, not assumed: the first version of this sweep read
                    # `spheres 110, space_filling 0, ball_and_stick 0, stick 0,
                    # ribbon 0, cartoon 0` and looked for a product defect in a
                    # picture the product was drawing correctly.
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(0, 0)
                    for _ in range(4):
                        app.processEvents()
                    # The projection, from the same camera that is about to be
                    # photographed. Read *after* the selection is set, so the
                    # segment is the one that was drawn.
                    _r_seg = _own.viewport.pair_screen_segment()
                    _r_on = _grab_live(_own)
                    _own.viewport.highlight_pair = None
                    for _ in range(3):
                        app.processEvents()
                    _r_off = _grab_live(_own)
                    _own.viewport.highlight_pair = _own_got
                    _r_mark = _pair_marker(_r_seg, _r_on, _r_off)
                    _repr_sweep.append((
                        _key,
                        _pair_own_pixels(_r_on, _r_off),
                        _pair_highlight_pixels(_r_on) if _r_on is not None else -1,
                        _r_mark,
                    ))
                _own.viewport.representation = _repr_keep
                _own.pair_table.clearSelection()
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.setCurrentCell(0, 0)
                for _ in range(6):
                    app.processEvents()
                _settle_camera(_own, app)
                # **Findable, not merely present.** "Own px >= 8" is a floor on
                # a size and the 1 px line clears it at every framing, so it
                # cannot tell a marker a reader can find from one they cannot;
                # `marker_findable` asks instead whether the drawn marker's width
                # accounts for the width the product says it is drawing. See the
                # function for the measured table and for what it is not
                # claiming.
                _repr_all = all(marker_findable(m) for _k, _o, _p, m in _repr_sweep)
                # **And three consecutive rows, in the default representation.**
                # The sweep above asks whether one pair survives every
                # representation; this asks whether three different pairs are all
                # findable at all. One row cannot answer that, and the row that
                # used to be the only one measured -- row 0 -- is the single row
                # of the ten where the rod's own axis is at 45 degrees to the
                # screen axes, which is the one orientation the old offset got
                # right: see `PAIR_ROW_SWEEP` and `_pair_quad`.
                #
                # Same camera discipline as the sweep above and for the same
                # measured reason: `clearSelection()` before `setCurrentCell`,
                # because the table does not re-emit for a row that is already
                # current, and `_settle_camera` after it, because selecting a row
                # moves the camera onto that pair. The frame is left the way a
                # reader would leave it -- row 0 selected, camera settled on it.
                _row_marks = []
                for _row in PAIR_ROW_SWEEP:
                    if _row >= _own_n:
                        break
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(_row, 0)
                    for _ in range(6):
                        app.processEvents()
                    _r_settled = _settle_camera(_own, app)
                    _g_seg = _own.viewport.pair_screen_segment()
                    _g_on = _grab_live(_own)
                    _own.viewport.highlight_pair = None
                    for _ in range(3):
                        app.processEvents()
                    _g_off = _grab_live(_own)
                    _row_marks.append((
                        _row, _r_settled,
                        _pair_marker(_g_seg, _g_on, _g_off),
                    ))
                _own.pair_table.clearSelection()
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.setCurrentCell(0, 0)
                for _ in range(6):
                    app.processEvents()
                _settle_camera(_own, app)
                _row_lines = [
                    f"row {r}: "
                    + ("no frame or no segment" if m is None else
                       f"{m['own_px']} px, {m['measured_px']:.2f} of "
                       f"{m['declared_px']:.2f} px declared, fill {m['fill']:.2f}"
                       f", stray {m['stray_px']}"
                       f", findable {marker_findable(m)}")
                    + ("" if s else " (the camera never settled)")
                    for r, s, m in _row_marks
                ] or ["no pair row could be measured"]
                _repr_any_pure = sum(1 for _k, _o, pure, _m in _repr_sweep
                                     if pure > 0)
                # **Six agreeing numbers are not six confirmations.** The sweep
                # above is evidence only if its arms can tell each other apart.
                # Measured on 1crn/biotin row 2 at close range -- 8.125 A and
                # 6 A -- the six representations return *byte-identical* marker
                # measurements (own px, measured px, declared px and fill all
                # equal to four decimals) while the six frames are byte
                # *distinct*, 6 of 6 md5s. The reason is not that the marker
                # behaves the same in every mode: at that range the rod is
                # 20-27 px wide and 450-770 px long, it is drawn with depth
                # testing off, and it covers the geometry it lies over, so every
                # arm measures the same opaque rectangle. Six rows of one
                # measurement, printed as a table, look like agreement and are
                # not.
                #
                # So the sweep is gated on being able to discriminate at all. If
                # the arms collapse, this goes red and says so, rather than
                # passing on six copies of one number. At the framing this check
                # runs in, the arms do differ -- the declared width moves with
                # the representation, 5.30 to 6.71 px -- so the condition is
                # satisfiable and the gate is not asking for something
                # impossible.
                _repr_keys = set()
                for _k, _o, _pure, _m in _repr_sweep:
                    if _m is None:
                        continue
                    _repr_keys.add((_m.get("own_px"), _m.get("measured_px"),
                                    _m.get("declared_px"), _m.get("fill")))
                _repr_distinct = len(_repr_keys)
                _repr_live = _repr_distinct > 1
                # **The required count is `EXPECTED_PAIR_ROWS`, not
                # `len(PAIR_ROW_SWEEP)`.** See that constant for the mutation
                # that showed why: comparing the sweep against itself means a
                # shorter sweep is a shorter requirement, and the check goes
                # green having measured less than it claims.
                _row_short = EXPECTED_PAIR_ROWS - len(_row_marks)
                _row_all = (_row_short <= 0
                            and all(_s and m is not None
                                    and marker_findable(m)
                                    for _r, _s, m in _row_marks))
                # One clause per representation, built here rather than inside
                # the f-string below: a comprehension with a nested conditional
                # inside a format expression is unreadable, and this file has
                # already lost a run to a detail string that raised.
                _repr_lines = []
                for _k, _own_px, _pure_px, _m in _repr_sweep:
                    if not _m or _m.get("own_px", 0) <= 0:
                        _repr_lines.append(f"{_k} {_own_px} px")
                        continue
                    _stray = ("" if _m["stray_px"] == 0
                              else f", {_m['stray_px']} px off the segment")
                    _repr_lines.append(
                        f"{_k} {_own_px} px, "
                        f"{_m['measured_px']:.1f} of {_m['declared_px']:.1f} px "
                        f"declared, fill {_m['fill']:.2f}, contrast p50 "
                        f"{_m['contrast_p50']:.0f}{_stray}")
                check(
                    "the selected pair's highlight is visible in pixels, and "
                    "nothing of it is on screen with no row selected",
                    (_own_on >= 8 and _own_frames_differ and _own_got is not None
                     and _own_off == 0 and _repr_all and _repr_live and _row_all),
                    f"in a window built for this measurement, {_own_n} pair row(s) "
                    f"and `highlight_pair` = {_own_got}: the highlight owns "
                    f"**{_own_on} px** of the frame ({_own_on_path}) measured as "
                    f"the difference from the same camera with the draw "
                    f"suppressed ({_own_off_path}), and the two frames are "
                    f"{'different' if _own_frames_differ else 'IDENTICAL'} as "
                    f"bytes. A *purity* count -- pixels within 30 of "
                    f"`COLOR_PAIR` -- reads {_own_on_pure} px on the same frame, "
                    f"and reading 0 there is not a finding about the product: "
                    f"that count measures how much of the line the MSAA sample "
                    f"pattern happened to cover, not whether the line is there. "
                    f"**The same pair in every representation**, own px and the "
                    f"marker's width against the width it declares, with the row "
                    f"selected: " + "; ".join(_repr_lines)
                    + f". The six arms returned {_repr_distinct} distinct "
                    f"measurement(s) over {len(REPRESENTATION_KEYS)} "
                    f"representations, and the sweep only counts as evidence if "
                    f"that is more than one: at a range where the rod covers the "
                    f"geometry it lies over, every arm measures the same opaque "
                    f"rectangle and six agreeing rows are one measurement printed "
                    f"six times, not six confirmations"
                    + f". **Three consecutive pair rows**, default representation, "
                    f"each selected, each measured against its own projection: "
                    + "; ".join(_row_lines)
                    + f". All {EXPECTED_PAIR_ROWS} had to come back findable, and "
                    f"all {EXPECTED_PAIR_ROWS} had to be measurable at all; "
                    f"{len(_row_marks)} came back"
                    + (f", which is {_row_short} short of what this check "
                       f"requires and is why it is red"
                       if _row_short > 0 else "")
                    + f" -- a "
                    f"sweep that quietly found fewer rows would be a green check "
                    f"about less than it says, which is the failure this widening "
                    f"exists to stop. The count is compared against "
                    f"`EXPECTED_PAIR_ROWS` and not against `len(PAIR_ROW_SWEEP)` "
                    f"because comparing the sweep with itself makes the "
                    f"requirement move when the sweep does. One row is not "
                    f"enough, and not because of "
                    f"which row it was: row 0 passes the 0.5 floor at 0.89 of "
                    f"its declared width, so the one-row gate was green on a row "
                    f"the pre-fix offset already drew 11% thin. The 45 degree "
                    f"row of the first ten is row 4 (1.061, the only one of "
                    f"them drawn correctly), and the sweep does not contain it. "
                    f"The defect is graded rather than binary -- across all "
                    f"{_own_n} rows of this fixture the pre-fix offset drew 23 of "
                    f"them measurably wrong and 4 right, and only 11 failed the "
                    f"0.5 floor -- so one row and three rows are both samples of "
                    f"a distribution, and the widening is what stops three "
                    f"being quoted as the rate. The 8 "
                    f"is a floor on a *size*, derived from the "
                    f"framebuffer rather than chosen: the weakest highlight pixel "
                    f"is one MSAA sample of sixteen over the background, "
                    f"`(242 - 26) / 16 = 13.5` levels in red and blue, so 8 sits "
                    f"under anything real. The 0 in the old check was a bound "
                    f"that could not fail, and `_draw_pair` draws the line with "
                    f"the depth test off precisely so that it exists at all: a "
                    f"line from one atom's centre to the other's has both ends "
                    f"inside the spheres it joins, and depth-tested what "
                    f"survived was the gap between two surfaces -- zero of it in "
                    f"the space-filling, ribbon and cartoon modes",
                )
            # ---------------------------------------------------------------- M8
            # The guard that says when the marker *cannot* be drawn, and the
            # status line that has to admit it.
            #
            # The old guard asked for a screen length of 1e-6 px while the rod's
            # area crosses half a pixel about 3e4 times earlier, so across that
            # whole interval `_pair_quad` returned six valid vertices, no NaN
            # passed either guard, the rasteriser produced nothing, and the
            # status bar still read "pair 1 of 27: O -> LEU 18A O". Both halves
            # are checked here, and they are checked against the *same*
            # predicate on purpose: two implementations of one rule is how the
            # bar comes to promise a marker the draw then declines to submit.
            #
            # **Two floors now, and they are not the same floor.** The area one
            # catches a marker too small to cover a sample point. The length one
            # catches the case the area one cannot see at all: a rod 7.127 px
            # wide and 0.299 px long, area 2.141 px^2, comfortably over the 0.5
            # floor, drawn as eight scattered pixels at a quarter contrast that
            # a reader cannot pick out of the picture. Measured over 96 cameras
            # at both ratios, the lit set is one pixel column along its own axis
            # below 1.0 px of length and two or more above it, at every one of
            # twelve sub-pixel phases. The rows below are all *above* both
            # floors, which is the point: no real row is refused, and the check
            # that says so is the check that would catch it if one were.
            from opendocking.workbench.app import (
                PAIR_MARKER_FRAME_WARN, PAIR_MARKER_MIN_AREA_PX2,
                PAIR_MARKER_MIN_LENGTH_PX, _projected_area_px2,
                marker_area_ok, marker_drawable, marker_length_ok,
            )

            _area_rows = []
            _area_bad_report = []
            _area_guard_holds = True
            for _r in range(min(6, _own_n or 0)):
                _own.pair_table.clearSelection()
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.setCurrentCell(_r, 0)
                for _ in range(6):
                    app.processEvents()
                if not _settle_camera(_own, app):
                    continue
                _s = _own.viewport.pair_screen_segment()
                if _s is None:
                    continue
                _a = _projected_area_px2(_s)
                _len = float(_s["length"])
                _ok = marker_drawable(_s)
                _quad = None
                _p16, _r16, _a16, _b16 = _own.viewport._pair_atoms()
                if _p16 is not None:
                    _quad = _own.viewport._pair_quad(
                        _a16, _b16, float(_s["radius"]))
                # The draw and the predicate must agree at every camera: the
                # marker is submitted exactly when both floors are cleared.
                _submitted = _quad is not None
                if _submitted != bool(_ok):
                    _area_guard_holds = False
                _area_rows.append(
                    f"row {_r}: {_len:.4g} px long, {_a:.4g} px^2 "
                    f"({float(_s['width']):.4g} px wide), "
                    f"{'drawn' if _ok else 'NOT DRAWN'}"
                    + ("" if _submitted == bool(_ok)
                       else f"  <-- the draw and the predicate DISAGREE "
                            f"(submitted {_submitted})"))
                # And the status line has to say so when it is not drawn.
                _own.statusBar().clearMessage()
                _own.pair_table.setCurrentCell(_r, 0)
                for _ in range(6):
                    app.processEvents()
                _txt = _own.statusBar().currentMessage()
                _has = ("not drawn" in _txt)
                if _has != (not _ok):
                    _area_bad_report.append(
                        f"row {_r}: {_len:.4g} px long, area {_a:.4g} px^2, "
                        f"draw says {'drawn' if _ok else 'NOT DRAWN'}, status "
                        f"text {'says so' if _has else 'is silent'}: "
                        f"{_txt[:90]!r}")
            _own.pair_table.setCurrentCell(0, 0)
            for _ in range(6):
                app.processEvents()
            check(
                "the pair marker is drawn exactly when it clears both of its "
                f"floors -- {PAIR_MARKER_MIN_AREA_PX2} px^2 of area and "
                f"{PAIR_MARKER_MIN_LENGTH_PX} px of screen length -- and the "
                "status line says so when it is not",
                _area_guard_holds and not _area_bad_report and bool(_area_rows),
                f"both floors are measured, and the two are not "
                f"interchangeable. AREA: the last camera the rasteriser left "
                f"empty and the first it drew on, over 61 off-axis angles at "
                f"two distances and both ratios, bracket "
                f"{PAIR_MARKER_MIN_AREA_PX2} px^2 in all four "
                f"configurations, whose intervals intersect at "
                f"(0.470, 0.515). LENGTH, which the area floor cannot express: "
                f"at 25.17 A the lit count holds at 8 while the area goes "
                f"1.75 -> 3.50 px^2 and the length only 0.270 -> 0.540 px, so "
                f"double the area and the same eight pixels with none of them "
                f"along the marker's own axis. Over 96 cameras panned in twelve "
                f"sub-pixel phases, the lit set is one pixel column below "
                f"{PAIR_MARKER_MIN_LENGTH_PX} px of length and two or more at "
                f"or above it, at every phase. The old guard asked for a screen "
                f"*length* of 1e-6 px, which is about 3e4 times later than the "
                f"area crosses half a pixel, so across that interval six valid "
                f"vertices were submitted, no NaN appeared, the rasteriser drew "
                f"nothing, and the status bar still named the pair. Over "
                f"{len(_area_rows)} row(s) at the framing F produces: "
                + "; ".join(_area_rows)
                + (("; REPORTING MISMATCH: " + "; ".join(_area_bad_report))
                   if _area_bad_report else "")
                + ("" if _area_guard_holds else
                   "; the draw and the predicate disagreed somewhere, which is "
                   "the defect this section exists to prevent")
                + f". Every real row clears both: over all 27 the shortest is "
                  f"37.6 px long and the smallest area 258 px^2, so neither "
                  f"floor refuses a row a reader can select.",
            )
            # ---------------------------------------------------------------- M9
            # **The refusal branch, driven there by a real gesture.**
            #
            # It was called "latent" for a round, and latent is a way of saying
            # nobody has run it. Two measurements changed that, and both were
            # about the gesture space rather than about the product:
            #
            # * a **pan cannot reach it**, and not for want of resolution. A
            #   pan *translates* the picture, and a translation does not change
            #   how long a segment is: panned a whole pixel along its own axis,
            #   the marker's screen length moved by **0.0001 px**. The pan is
            #   also the finer gesture, at a pixel of drag against the rotate's
            #   0.573 degrees, which is exactly why it was worth checking.
            # * a **yaw sweep alone cannot reach it either**: at a fixed pitch
            #   it traces a cone, and its closest approach to 1crn/biotin's
            #   pairs is 49.4 degrees. A diagonal drag moves pitch as well, and
            #   pitch is what closes the gap.
            #
            # So the reachable set is the 2-D lattice a diagonal drag walks.
            # `Camera.basis` builds `forward = (sin y cos p, -cos y cos p,
            # -sin p)`, so for a segment direction `u` the off-axis angle is
            # `asin |u x forward|` and it reaches **zero** at
            # `pitch = asin(-u_z)` -- which for all six rows sampled here is
            # between -9.8 and -82.4 degrees, every one inside the +-1.5 rad
            # clamp. What is left is the lattice granularity: the closest
            # reachable angle is the distance from the ideal pitch to the
            # nearest 0.01 rad step, and that is **0.08 to 0.34 degrees** --
            # which at 25 A is a marker **0.15 to 0.6 px long**.
            #
            # Both of those are past the 1.0 px length floor and well inside
            # it, and that is the whole reason the length guard is worth having:
            # **without it this branch is dead code, and with it a reader can
            # get there.** The area floor at 0.5 px^2 is *not* reachable by
            # gesture -- 0.15 px of length is 1.0 px^2 of area at this width,
            # and the drag step is too coarse to go lower -- so the area guard
            # stays a guard against cameras rather than a live branch, and the
            # check below says so rather than leaving it ambiguous.
            _reach = {"tried": [], "ok": False, "selfcheck": False,
                      "area_only_escape": False}
            _cam_before = _cam_state(_own.viewport)
            try:
                import math as _math

                _drag = 0.01          # one pixel of drag, as `mouseMoveEvent`
                # picks the row whose ideal pitch is furthest inside the clamp
                # and reports every row's number, so the choice is visible.
                _cands = []
                for _r in range(min(len(_own._pairs), 27)):
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(_r, 0)
                    for _ in range(6):
                        app.processEvents()
                    if not _settle_camera(_own, app):
                        continue
                    _pp, _rr, _aa, _bb = _own.viewport._pair_atoms()
                    if _aa is None:
                        continue
                    _d = np.asarray(_bb, np.float64) - np.asarray(_aa, np.float64)
                    _Ln = float(np.linalg.norm(_d))
                    if _Ln < 1e-9:
                        continue
                    _uu = _d / _Ln
                    _ideal = _math.asin(max(-1.0, min(1.0, -float(_uu[2]))))
                    _cands.append((abs(_ideal), _r, _uu, _ideal, _Ln))
                    _reach["tried"].append(
                        f"row {_r}: ideal pitch {_math.degrees(_ideal):.3f} deg")
                _cands.sort()
                if _cands:
                    _abs, _pick, _u, _ideal, _Ln = _cands[0]
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(_pick, 0)
                    for _ in range(6):
                        app.processEvents()
                    _settle_camera(_own, app)
                    _g0 = _off_axis_deg(_own.viewport, _u)
                    _byaw = _own.viewport.camera.yaw
                    _bpitch = _own.viewport.camera.pitch
                    # One diagonal drag, in the two axes `mouseMoveEvent`
                    # splits it into: yaw to the best heading, pitch to the
                    # ideal. `rotate` is what that handler calls, so this is
                    # the drag and not a camera set.
                    _Ry = _math.hypot(float(_u[0]), float(_u[1]))
                    if _Ry >= 1e-12:
                        _psi = _math.atan2(float(_u[1]), float(_u[0]))
                        _y_goal = _psi + _math.pi / 2.0
                    else:
                        _y_goal = _byaw
                    _kp = int(round((_ideal - _bpitch) / _drag))
                    _ky = int(round((_y_goal - _byaw) / _drag))
                    for _ in range(max(abs(_kp), abs(_ky))):
                        _step_y = _drag if (_ky > 0 and _ < _ky) else (
                            -_drag if (_ky < 0 and _ < -_ky) else 0.0)
                        _step_p = _drag if (_kp > 0 and _ < _kp) else (
                            -_drag if (_kp < 0 and _ < -_kp) else 0.0)
                        _own.viewport.rotate(_step_y, _step_p)
                    for _ in range(3):
                        app.processEvents()
                    _g1 = _off_axis_deg(_own.viewport, _u)
                    # **The self-check, and it is the point of this block.**
                    # A drag of a thousand pixels that does not move the
                    # angle would report "reached" and mean nothing, which is
                    # the defect that made a scratch yaw sweep print 19
                    # identical rows. So the sweep is required to have moved
                    # the quantity it claims to sweep.
                    _reach["selfcheck"] = _g1 < _g0 * 0.25
                    _reach["before_deg"] = _g0
                    _reach["after_deg"] = _g1
                    _reach["drag"] = (_kp, _ky)
                    _reach["row"] = _pick
                    # Now the click, which is what writes the status line.
                    # **The selection is cleared first, and it has to be.**
                    # `setCurrentCell` on the row that is already current emits
                    # nothing -- no `selectionChanged`, so `_on_pair_selected`
                    # never runs and the bar keeps whatever was there. The
                    # first version of this check read an empty string and
                    # called it "the bar said nothing about it", which is a
                    # statement about the test rather than about the product.
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.statusBar().clearMessage()
                    _own.pair_table.setCurrentCell(_pick, 0)
                    for _ in range(8):
                        app.processEvents()
                    _txt = _own.statusBar().currentMessage()
                    _reach["text"] = _txt
                    _reach["ok"] = "not drawn" in _txt
                    _s = _own.viewport.pair_screen_segment()
                    if _s is not None:
                        _reach["len"] = float(_s["length"])
                        _reach["area"] = _projected_area_px2(_s)
                        _reach["len_ok"] = marker_length_ok(_s)
                        _reach["area_ok"] = marker_area_ok(_s)
                        # Does the *area* floor alone let this one through?
                        _reach["area_only_escape"] = bool(
                            _reach["area_ok"] and not _reach["len_ok"])
            finally:
                # Put the camera back *exactly*, orientation included -- see
                # `_cam_state`. Then re-select the row the section opened on.
                _cam_restore(_own.viewport, _cam_before)
                try:
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(0, 0)
                    for _ in range(6):
                        app.processEvents()
                    _settle_camera(_own, app)
                except Exception:  # pragma: no cover - reporting only
                    pass
            _reach_said = ("not drawn" if _reach["ok"] else "NOTHING ABOUT IT")
            check(
                "a real drag can drive the product to a camera where the "
                "marker is refused, and the status line says so",
                bool(_reach["ok"]) and bool(_reach["selfcheck"]),
                f"the branch used to be called latent, which is a way of saying "
                f"nobody had run it. It is run here, by a drag. A pan cannot "
                f"reach it -- a pan translates the picture, and panned a whole "
                f"pixel along its own axis the marker's screen length moved by "
                f"0.0001 px -- and a yaw sweep alone bottoms out at 49.4 deg, "
                f"because at fixed pitch it traces a cone. A diagonal drag "
                f"moves pitch too, and the off-axis angle reaches zero at "
                f"`pitch = asin(-u_z)`, which is between -9.8 and -82.4 deg for "
                f"every row sampled and inside the +-1.5 rad clamp in every "
                f"case. What the drag step leaves is the distance to the "
                f"nearest 0.01 rad step: "
                + "; ".join(_reach["tried"][:6])
                + f". On row {_reach.get('row')} the drag of "
                  f"{_reach.get('drag', (0, 0))[0]} pitch and "
                  f"{_reach.get('drag', (0, 0))[1]} yaw pixels took the segment "
                  f"from {_reach.get('before_deg', 0):.4g} deg off the view "
                  f"axis to {_reach.get('after_deg', 0):.4g} deg, a marker "
                  f"{_reach.get('len', 0):.4g} px long and "
                  f"{_reach.get('area', 0):.4g} px^2 in area, and the status "
                  f"bar said {_reach_said}: "
                  f"{_reach.get('text', '')[-150:]!r}. SELF-CHECK: the sweep is "
                  f"only counted if the angle it claims to sweep actually "
                  f"moved, from {_reach.get('before_deg', 0):.4g} to "
                  f"{_reach.get('after_deg', 0):.4g} deg"
                + ("" if _reach["selfcheck"] else
                   " -- WHICH IT DID NOT, so this result is about nothing and "
                   "the check is red for the right reason"),
            )
            # ---------------------------------------------------------------- M10
            # **The case the area floor cannot see, on the real framebuffer.**
            #
            # The previous check leaves the camera somewhere useful, and this
            # one asks the discriminating question there: is this a marker the
            # *old* guard would have drawn and the new one refuses? If the area
            # floor also refuses it, the length floor has proved nothing --
            # either guard would do -- and the new constant is decoration.
            _scratch = {}
            _cam_before2 = _cam_state(_own.viewport)
            try:
                import math as _math2

                _cands2 = []
                for _r in range(min(len(_own._pairs), 27)):
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(_r, 0)
                    for _ in range(6):
                        app.processEvents()
                    if not _settle_camera(_own, app):
                        continue
                    _pp, _rr, _aa, _bb = _own.viewport._pair_atoms()
                    if _aa is None:
                        continue
                    _d = np.asarray(_bb, np.float64) - np.asarray(_aa, np.float64)
                    _Ln2 = float(np.linalg.norm(_d))
                    if _Ln2 < 1e-9:
                        continue
                    _uu2 = _d / _Ln2
                    _cands2.append((abs(_math2.asin(max(-1.0, min(
                        1.0, -float(_uu2[2]))))), _r, _uu2))
                _cands2.sort()
                if _cands2:
                    _abs2, _pick2, _u2 = _cands2[0]
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(_pick2, 0)
                    for _ in range(6):
                        app.processEvents()
                    _settle_camera(_own, app)
                    _by2 = _own.viewport.camera.yaw
                    _bp2 = _own.viewport.camera.pitch
                    _ideal2 = _math2.asin(max(-1.0, min(1.0, -float(_u2[2]))))
                    _Ry2 = _math2.hypot(float(_u2[0]), float(_u2[1]))
                    _y_goal2 = (_math2.atan2(float(_u2[1]), float(_u2[0]))
                                + _math2.pi / 2.0) if _Ry2 >= 1e-12 else _by2
                    _kp2 = int(round((_ideal2 - _bp2) / 0.01))
                    _ky2 = int(round((_y_goal2 - _by2) / 0.01))
                    for _ in range(max(abs(_kp2), abs(_ky2))):
                        _sy = 0.01 if (_ky2 > 0 and _ < _ky2) else (
                            -0.01 if (_ky2 < 0 and _ < -_ky2) else 0.0)
                        _sp = 0.01 if (_kp2 > 0 and _ < _kp2) else (
                            -0.01 if (_kp2 < 0 and _ < -_kp2) else 0.0)
                        _own.viewport.rotate(_sy, _sp)
                    for _ in range(3):
                        app.processEvents()
                    _s2 = _own.viewport.pair_screen_segment()
                    if _s2 is not None:
                        _q2 = None
                        _p2, _r2, _a2, _b2 = _own.viewport._pair_atoms()
                        if _p2 is not None:
                            _q2 = _own.viewport._pair_quad(
                                _a2, _b2, float(_s2["radius"]))
                        _scratch = {
                            "row": _pick2,
                            "len": float(_s2["length"]),
                            "area": _projected_area_px2(_s2),
                            "width": float(_s2["width"]),
                            "len_ok": marker_length_ok(_s2),
                            "area_ok": marker_area_ok(_s2),
                            "quad": _q2 is not None,
                        }
            finally:
                _cam_restore(_own.viewport, _cam_before2)
                try:
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(0, 0)
                    for _ in range(6):
                        app.processEvents()
                    _settle_camera(_own, app)
                except Exception:  # pragma: no cover - reporting only
                    pass
            check(
                f"the length floor refuses the marker the "
                f"{PAIR_MARKER_MIN_AREA_PX2} px^2 area floor waves through",
                bool(_scratch) and _scratch.get("area_ok") is True
                and _scratch.get("len_ok") is False
                and _scratch.get("quad") is False,
                f"this is the check that makes the new constant a guard rather "
                f"than a decoration: if the area floor refused it too, either "
                f"guard would do and nothing was added. Row "
                f"{_scratch.get('row')} at the camera the drag above reaches "
                f"has a marker {_scratch.get('len', 0):.4g} px long and "
                f"{_scratch.get('width', 0):.4g} px wide, so its area is "
                f"{_scratch.get('area', 0):.4g} px^2"
                + (f" -- {_scratch.get('area', 0) / PAIR_MARKER_MIN_AREA_PX2:.1f}"
                   f"x the area floor, which it clears"
                   if _scratch.get("area_ok") else ", refused on area as well")
                + f" -- while its length is "
                  f"{_scratch.get('len', 0):.4g} px, "
                  f"{_scratch.get('len', 0) / PAIR_MARKER_MIN_LENGTH_PX:.2f}x "
                  f"the {PAIR_MARKER_MIN_LENGTH_PX} px length floor, which it "
                  f"does not. `marker_area_ok` is "
                  f"{_scratch.get('area_ok')}, `marker_length_ok` is "
                + f"{_scratch.get('len_ok')}, and `_pair_quad` returned "
                f"{'vertices' if _scratch.get('quad') else 'None'}. Measured "
                f"over four declared widths from 1.16 to 20.3 px, thirteen "
                f"screen lengths on a 0.05 px ladder and eight sub-pixel phases "
                f"along the mark's own axis, "
                f"counting pixel columns in framebuffer coordinates rather than "
                f"from the mark's start point: the lit set is one column below "
                f"about 0.6 px of length at the widest marker and reaches two "
                f"columns at every phase only from 1.25 px at dpr 1.25 and "
                f"1.50 px at dpr 2.5, so the 1 px floor is a geometric floor "
                f"and not the two-column one. See PAIR_MARKER_MIN_LENGTH_PX.",
            )
            # ---------------------------------------------------------------- M11
            # **The marker that is most of the picture.**
            #
            # At 2 A from a bond, broadside, the rod is 81.18 px wide and
            # 1940 px long. **Two different shares, and this check now names
            # which one the status bar reports**, because both were once called
            # "the share of the frame" and they differ by up to 2x:
            #
            #   * the *declared* share -- `_projected_area_px2` over the
            #     framebuffer, the projected area of the marker's own
            #     rectangle. A projection fact, and the one the bar prints:
            #     **14.14%** of a 1126x989 frame.
            #   * the *drawn* share -- the pixels that differ between two grabs
            #     at this camera, over the framebuffer. A rasterisation fact,
            #     smaller and **unstable**, because a 1940 px rod only lands
            #     part of itself inside a 989 px frame and how much depends on
            #     its screen angle: 93 645 px (8.41%) at the azimuth this
            #     window's own framing produces, 82 796 px (7.43%) at the
            #     near-vertical azimuth quoted in the previous round. The
            #     declared share is 14.142% either way, which is why it is the
            #     one a reader can check.
            #
            # A 48x48 crop window around the pair is entirely inside the rod
            # either way, so a reader looking at that crop sees a solid magenta
            # panel and cannot tell an honest wall from a fault.
            #
            # The decision is to **draw it and say so**. The geometry is correct
            # and the camera is the reader's own choice; refusing would make
            # the picture disagree with the projection, which is the one thing
            # this product is for. What is missing is the sentence, and a user
            # cannot infer from a magenta wall that the wall is honest.
            #
            # The probe below therefore looks for the sentence's own wording --
            # the marker's rectangle, a percentage and the frame's size -- and
            # not for a single keyword, because the wording changed when the
            # quantity it names became explicit. `focus_pair` is stubbed for the
            # duration so the 2 A framing survives the click that writes the
            # status line, and the stub is restored in a `finally`. The stub is
            # an instrument, not a change to the product: nothing else about the
            # selection is altered.
            _occ = {"share": None, "text": "", "fired": False}
            _cam_before3 = _cam_state(_own.viewport)
            try:
                import types as _types

                _real_focus = _own.viewport.focus_pair

                def _nofocus(_self, _a, _b, _f=_real_focus):
                    _own.viewport.update()
                    return True

                _own.viewport.focus_pair = _types.MethodType(
                    _nofocus, _own.viewport)
                _own.viewport.camera.distance = 2.0
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.clearSelection()
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.setCurrentCell(0, 0)
                for _ in range(8):
                    app.processEvents()
                _s3 = _own.viewport.pair_screen_segment()
                if _s3 is not None:
                    _fw3, _fh3 = (int(v) for v in _s3.get("size", (0, 0)))
                    if _fw3 > 0 and _fh3 > 0:
                        _occ["share"] = _projected_area_px2(_s3) / float(
                            _fw3 * _fh3)
                _occ["text"] = _own.statusBar().currentMessage()
                # The wording the product now uses: the marker's own rectangle,
                # a share with two decimals, and the frame it is a share *of*.
                # A keyword probe would have passed on the old sentence and
                # failed on this one for no reason a reader could see.
                _occ["fired"] = ("rectangle is" in _occ["text"]
                                 and "%" in _occ["text"]
                                 and "frame" in _occ["text"])
            finally:
                try:
                    _own.viewport.__dict__.pop("focus_pair", None)
                except Exception:  # pragma: no cover - reporting only
                    pass
                _cam_restore(_own.viewport, _cam_before3)
                try:
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(0, 0)
                    for _ in range(6):
                        app.processEvents()
                    _settle_camera(_own, app)
                except Exception:  # pragma: no cover - reporting only
                    pass
            check(
                f"a marker whose declared rectangle covers more than "
                f"{PAIR_MARKER_FRAME_WARN * 100:.0f}% of the frame is still "
                f"drawn, and the status bar says how much of the frame it is",
                bool(_occ["fired"]) and _occ["share"] is not None
                and _occ["share"] >= PAIR_MARKER_FRAME_WARN,
                f"at 2 A from a bond, broadside, the rod is 81.18 px wide and "
                f"1940 px long; its declared rectangle is measured here "
                f"{_occ['share'] * 100:.2f}% of a "
                f"{_fw3 if _fw3 else 0}x{_fh3 if _fh3 else 0} frame, and that "
                f"is the share the bar reports. The *drawn* share is a "
                f"different and smaller number -- 93 645 px (8.41%) at this "
                f"window's framing, 82 796 px (7.43%) at a near-vertical "
                f"azimuth -- because a 1940 px rod only lands part of itself "
                f"inside a 989 px frame, so it is not a stable thing to quote "
                f"and the projection share is. The decision is to draw it and "
                f"say so: the geometry is correct and the camera is the "
                f"reader's own choice, and refusing would make the picture "
                f"disagree with the projection. What was missing was the "
                f"sentence -- a 48x48 crop around the pair is entirely inside "
                f"that rod, so a reader sees a solid magenta panel and cannot "
                f"tell it from a fault. The bar read: "
                f"{_occ['text'][-160:]!r}. The threshold is chosen rather than "
                f"measured: measured declared shares run 14.14% at 2 A, 6.29% "
                f"at 3, 3.54% at 4, 1.57% at 6, 0.88% at 8 and 0.089% at "
                f"25.17, on 1crn/biotin, so one per cent fires between 6 and "
                f"8 A and the framing `F` gives (22.78 A, 0.109%) is silent. "
                f"SELF-CHECK: the share is read off "
                f"the segment at the same camera the bar reported on, so a "
                f"stub that put the camera somewhere else cannot make this "
                f"pass.",
            )
            # ---------------------------------------------------------------- M12
            # **The pad floor: needed, sufficient, and one-sided.**
            #
            # Three claims, and the third is the one that could have broken:
            #
            # 1. the floor is *needed* -- at 0.75 px a correct draw on a thin
            #    marker is reported as misplaced, which is the floor firing
            #    against geometry that is right;
            # 2. it is *sufficient* -- at 1.0 px no thin marker in the sweep
            #    reports a stray at all;
            # 3. it is *one-sided* -- a draw planted beside the pair is still
            #    caught, so the floor has not grown into a second failure.
            #
            # The sweep walks six distances whose declared widths are a pixel
            # and a fraction, which is the only band where `max(0.5*width,
            # floor)` is the floor rather than the scaling term. Both claims 1
            # and 2 are recomputed here from the same two grabs with the floor
            # as a parameter, so the numbers here are this run's and not a
            # transcript of an earlier one.
            _pad = {"rows": [], "needed": False, "sufficient": True,
                    "selfcheck": True, "plant_caught": None}
            _cam_before4 = _cam_state(_own.viewport)
            try:
                import types as _types2

                _own.pair_table.clearSelection()
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.setCurrentCell(0, 0)
                for _ in range(6):
                    app.processEvents()
                _settle_camera(_own, app)
                _prev_w = None
                for _d4 in (120.0, 200.0, 300.0, 450.0, 700.0, 1000.0):
                    _s4, _f4, _r4, _a4, _b4 = _pair_marker_camera(_own, _d4)
                    if _s4 is None:
                        continue
                    _w4 = float(_s4["width"])
                    # SELF-CHECK: the sweep is walking thinner markers, and a
                    # rig whose distance argument did nothing would report a
                    # flat table and read as a finding.
                    if _prev_w is not None and _w4 >= _prev_w:
                        _pad["selfcheck"] = False
                    _prev_w = _w4
                    _m4 = _pair_marker(_s4, _f4["on"], _f4["off"])
                    _m4b = _pair_marker_with_pad(_s4, _f4["on"], _f4["off"],
                                                 0.75)
                    if _m4 is None or _m4b is None:
                        continue
                    _pad["rows"].append(
                        f"{_d4:.0f} A (w {_w4:.3f} px): "
                        f"stray {int(_m4b['stray_px'])} at 0.75, "
                        f"{int(_m4['stray_px'])} at 1.0")
                    if int(_m4b["stray_px"]) > 0:
                        _pad["needed"] = True
                    if int(_m4["stray_px"]) > 0:
                        _pad["sufficient"] = False
                # The planted draw. `_pair_quad` is displaced sideways and
                # `pair_screen_segment` is left alone, which is the only way
                # to ask whether the instrument notices a marker that is not
                # where the projection says the pair is.
                _s5, _f5, _r5, _a5, _b5 = _pair_marker_camera(_own, 700.0)
                if _s5 is not None:
                    _real_quad = _own.viewport._pair_quad
                    _up5 = np.asarray(
                        _own.viewport.camera.basis()[1], np.float64)
                    _fw5, _fh5 = (int(v) for v in _s5.get("size", (0, 0)))
                    _push = 3.0 * (1.0 / math.tan(
                        math.radians(_own.viewport.camera.fov) * 0.5)) \
                        * _fh5 / (2.0 * max(float(_s5["depth"][0]), 1e-6))
                    _offw = (np.asarray(_up5, np.float64)
                             * (_push / max(float(np.linalg.norm(_up5)), 1e-9))
                             ).astype(np.float32)

                    def _shift(self, a, b, radius, _o=_offw):
                        q = _real_quad(a, b, radius)
                        if q is None:
                            return None
                        return np.asarray(q, np.float32) + _o

                    _own.viewport._pair_quad = _types2.MethodType(
                        _shift, _own.viewport)
                    try:
                        _own.viewport.update()
                        for _ in range(3):
                            app.processEvents()
                        # `_pair_marker_grab` returns the two frames as a dict,
                        # not as a pair alongside the segment: the segment is
                        # the reference and the patch does not move it.
                        _f5b = _pair_marker_grab(_own)
                        if _f5b["on"] is not None and _f5b["off"] is not None:
                            _m5 = _pair_marker(_s5, _f5b["on"], _f5b["off"])
                            if _m5 is not None:
                                _pad["plant_caught"] = int(_m5["stray_px"])
                    finally:
                        _own.viewport.__dict__.pop("_pair_quad", None)
                        _own.viewport.update()
                        for _ in range(3):
                            app.processEvents()
            finally:
                _cam_restore(_own.viewport, _cam_before4)
                try:
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(0, 0)
                    for _ in range(6):
                        app.processEvents()
                    _settle_camera(_own, app)
                except Exception:  # pragma: no cover - reporting only
                    pass
            check(
                "the pad floor is needed on thin markers, sufficient on all of "
                "them, and still catches a draw planted beside the pair",
                bool(_pad["needed"]) and bool(_pad["sufficient"])
                and bool(_pad["selfcheck"])
                and (_pad["plant_caught"] or 0) > 0,
                f"`pad = max(0.5 * width, floor)`, so at a declared 6.45 px the "
                f"floor is inert below 3.225 and this sweep is the only place "
                f"it can be seen at all. Over six distances from 120 to 1000 A "
                f"at this run's device pixel ratio, the same two grabs "
                f"re-evaluated at each floor: "
                + "; ".join(_pad["rows"])
                + f". So 0.75 px is *needed* to change: it leaves a stray "
                  f"pixel on a correct draw, and an allowance that cannot "
                  f"reach zero is not an allowance but a second invisible "
                  f"floor on width. 1.0 px is *sufficient*: no thin marker in "
                  f"the sweep reports one. And it is *one-sided*: with the "
                  f"draw displaced 3 px beside the pair at 700 A -- "
                  f"`_pair_quad` shifted, `pair_screen_segment` left as the "
                  f"reference, which is the only way to ask -- it still reports "
                  f"{_pad['plant_caught']} stray px"
                + ("" if (_pad["plant_caught"] or 0) > 0 else
                   ", WHICH IS ZERO, so the floor has absorbed a displaced "
                   "draw and this check is red for the right reason")
                + ("" if _pad["selfcheck"] else
                   ". SELF-CHECK FAILED: the declared width did not fall "
                   "monotonically across the sweep, so the distance argument "
                   "did nothing and this table is about nothing"),
            )
            # ---------------------------------------------------------------- M13
            # **The share of the frame the framed subject reaches, on the bar.**
            #
            # `framing_selection.focus_target` computes it -- it is
            # `PairFraming.subject_fill` -- and hands it back on the record
            # `Viewport._focus` was given. `_focus` read `center` and `distance`
            # off that record and dropped the other three fields, so the module
            # measured the quantity its whole existence is about, on every
            # selection, and the window showed it to nobody. The original
            # defect on this line of work was a pose framed too small to see,
            # and this is the number that says whether the pose in front of the
            # reader is too small to see.
            #
            # **The first check is its own self-check, and that is the point.**
            # The expected string is built here from
            # `_last_pair_framing.subject_fill` and compared against the text
            # the bar printed, so a product that printed a plausible constant
            # would fail here. A keyword probe would pass on a bar that said
            # "framed subject" and never carried a number at all.
            #
            # **Every row is walked, not a sample.** The claim is "on every
            # selection", and a claim about all of them measured on the three
            # rows of `PAIR_ROW_SWEEP` is a claim about three rows. The cost is
            # one selection change per row and no extra GL work: the camera
            # work is M8-M12's, and this only reads what the selection handler
            # left behind. How many rows there are is the window's to decide
            # and the check reports both sides of the fraction, so a window
            # that found fewer contacts cannot pass by sampling.
            _share = {"rows": [], "have": 0, "units": 0, "bad": [],
                      "n": _own_n or 0}
            for _r in range(_own_n or 0):
                _own.pair_table.clearSelection()
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.setCurrentCell(_r, 0)
                for _ in range(6):
                    app.processEvents()
                _txt = _own.statusBar().currentMessage()
                _fr = _own.viewport._last_pair_framing
                if _fr is None:
                    _share["bad"].append(f"row {_r}: nothing kept the framing")
                    continue
                _want = (f"  [framed subject reaches "
                         f"{float(_fr.subject_fill) * 100:.2f}% of the "
                         f"half-frame at {float(_fr.distance):.1f} A]")
                if _want in _txt:
                    _share["have"] += 1
                    _share["rows"].append(
                        f"row {_r}: {float(_fr.subject_fill) * 100:.2f}% at "
                        f"{float(_fr.distance):.1f} A")
                else:
                    _share["bad"].append(
                        f"row {_r}: bar read {_txt[-96:]!r}, wanted {_want!r}")
                if "half-frame" in _txt:
                    _share["units"] += 1
            # Put the row the section opened on back, the way M9 and M12 do.
            _own.pair_table.setCurrentCell(0, 0)
            for _ in range(6):
                app.processEvents()

            check(
                "every pair selection reports the share of the frame its "
                "subject reaches, and the number is the framing's own",
                bool(_share["rows"]) and not _share["bad"]
                and _share["have"] == _share["n"],
                f"{_share['have']} of {_share['n']} row(s) carried the number "
                f"read off the record the framing returned"
                + ("; " + "; ".join(_share["bad"][:3]) if _share["bad"] else "")
                + ". Over this sweep: "
                + ("; ".join(_share["rows"][:4]) if _share["rows"]
                   else "no rows carried one")
                + (f"; and {len(_share['rows'])} more"
                   if len(_share["rows"]) > 4 else "")
                + ". SELF-CHECK: the expected substring is built here from "
                  "`_last_pair_framing.subject_fill` and `distance`, the same "
                  "record `_focus` was handed, so a bar printing a constant "
                  "cannot pass this.",
            )
            check(
                "the reported share names its unit, so it cannot be read as "
                "the marker's projected-area share on the same line",
                bool(_share["rows"]) and _share["units"] == _share["n"],
                f"{_share['units']} of {_share['n']} line(s) said 'half-frame'. "
                f"The marker's own share, printed a few words earlier by the "
                f"same handler, is a *projected area* over the framebuffer's "
                f"pixels and is called a percentage of the frame too. The two "
                f"are an order of magnitude apart on one picture -- 11.91% and "
                f"28.35% of the half-frame against 0.109% of a 989x1251 frame "
                f"at the framing `F` gives -- so a bar that printed both "
                f"without the unit would be asking the reader to guess which "
                f"is which.",
            )
            # ---------------------------------------------------------------- M7
            # What the pair framing is *for*. Everything above proves the
            # highlight's pixels exist; none of it says there is a picture for
            # them to point at, and the pixel count cannot: the share of the
            # frame that is molecule is **86.29%** in the space-filling frame at
            # the old 8.125 A framing and **62.44%** on `11c_nocloud_space_
            # filling.png`, whose real complaint is that the pose is at the edge
            # of it. A floor on "fraction of the frame that is molecule" passes
            # both and catches neither, so it is not the instrument.
            #
            # The instrument is geometric, and derived rather than chosen. A drawn
            # object of radius `r` at depth `t` from the eye subtends an angular
            # half-width of `asin(r / t)`, and the frame subtends `fov / 2`. So
            # "the frame is a picture and not the inside of one sphere" is
            # `t >= r / sin(fov_fill * fov / 2)`, and the only quantity to pick is
            # `fov_fill`, a share of the field of view. There is no tolerance
            # here to widen: the assertion is the inequality.
            #
            # The numbers are read off the *live camera* after `F`, with the drawn
            # radii recomputed here from the representation in use, so a rule that
            # stopped depending on the drawing would tighten the radius and fail
            # this rather than quietly pass it.
            _surround = []
            if _own_n > 0 and _own_got is not None:
                from opendocking.workbench import framing_selection as _fs

                _keep16 = _own.viewport.representation
                _pose16 = next((m for m in _own.viewport.molecules
                                if m.role == "pose"), None)
                _rec16 = next((m for m in _own.viewport.molecules
                               if m.role == "receptor"), None)
                if _pose16 is not None and _rec16 is not None:
                    _pa = np.asarray(_pose16.coords[int(_own_got[0])], np.float64)
                    _rb = np.asarray(_rec16.coords[int(_own_got[1])], np.float64)
                    _subj = np.vstack([_pa, _rb])
                    _mid = _subj.mean(axis=0)
                    _all_rec = np.asarray(_rec16.coords, np.float64).reshape(-1, 3)
                    _near = np.linalg.norm(_all_rec - _mid, axis=1) <= _fs.PAIR_CONTEXT_SHELL
                    _aspect = max(_own.viewport.width(), 1) / max(
                        _own.viewport.height(), 1)
                    for _key in REPRESENTATION_KEYS:
                        _own.viewport.representation = _key
                        _own.pair_table.clearSelection()
                        for _ in range(3):
                            app.processEvents()
                        _own.pair_table.setCurrentCell(0, 0)
                        for _ in range(4):
                            app.processEvents()
                        QTest.keyClick(_own, QtCore.Qt.Key.Key_F)
                        _settle_camera(_own, app)
                        for _ in range(4):
                            app.processEvents()
                        _cam = _own.viewport.camera
                        _rr = np.asarray(
                            _own.viewport._framing_radii(_rec16), np.float64)
                        _rp = np.asarray(
                            _own.viewport._framing_radii(_pose16), np.float64)
                        _sr = np.asarray([_rp[int(_own_got[0])],
                                          _rr[int(_own_got[1])]], np.float64)
                        _ctx = np.vstack([_subj, _all_rec[_near]])
                        _cr = np.concatenate([_sr, _rr[_near]])
                        _r, _u, _f = _cam.basis()
                        # The pair's two drawn discs, and whether a reader can
                        # still tell there are two of them. Threshold-free: the
                        # discs merge when the projected centre separation drops
                        # below the sum of the two projected radii, and there is
                        # nothing to choose there. A "the pair is at least N% of
                        # the frame" floor is a number with no derivation behind
                        # it, and the version this check first used -- 0.40,
                        # borrowed from the pose-selection contract -- was
                        # unreachable: the context contains the subject and the
                        # two fits ask for different fills, so the subject's floor
                        # could never be the binding term. The gate caught that,
                        # and the dead term is gone from the rule.
                        _tan = float(np.tan(np.radians(float(_cam.fov)) * 0.5))
                        _hw = _own.viewport.width() / 2.0
                        _zp = [float((_pt - _mid) @ np.asarray(_f, np.float64))
                               for _pt in (_pa, _rb)]
                        _rp = [max(float(_cam.distance) + _zp[i] - _sr[i], 1e-6)
                               for i in (0, 1)]
                        _rad_px = [_sr[i] / (_rp[i] * _tan * _aspect) * _hw
                                   for i in (0, 1)]
                        # The separation through the mean depth of the two, which
                        # is the projection both discs are compared in.
                        _sep_px = (abs(float((_pa - _rb)
                                             @ np.asarray(_r, np.float64)))
                                   / (max(0.5 * (float(_cam.distance) + _zp[0]
                                                 + _zp[1]), 1e-6)
                                      * _tan * _aspect) * _hw)
                        # The framing this replaced, in *this* mode's radii: the
                        # rule it replaced ignored the drawing and returned the
                        # same 8.125 A for all six, so this is the same distance
                        # six times over with six different sets of discs under
                        # it, and all six have to be inside the surface.
                        _old_d = max(6.0, float(np.linalg.norm(_subj - _mid,
                                                               axis=1).max()) * 5.0)
                        _old_slack = _fs.drawn_envelope_slack(
                            _ctx, _cr, _r, _u, _f, _old_d, center=_mid,
                            fov=float(_cam.fov))
                        _half_w = _old_d * _tan * _aspect / 2.0
                        _surround.append((
                            _key,
                            float(_cam.distance),
                            _fs.drawn_envelope_slack(
                                _ctx, _cr, _r, _u, _f, float(_cam.distance),
                                center=_mid, fov=float(_cam.fov)),
                            _fs.drawn_fill(
                                _subj, _sr, _r, _u, _f, float(_cam.distance),
                                center=_mid, fov=float(_cam.fov),
                                aspect=_aspect),
                            int(_near.sum()),
                            _sep_px,
                            _rad_px[0] + _rad_px[1],
                            _old_slack,
                            _half_w,
                        ))
                _own.viewport.representation = _keep16
                _own.pair_table.clearSelection()
                for _ in range(3):
                    app.processEvents()
                _own.pair_table.setCurrentCell(0, 0)
                for _ in range(4):
                    app.processEvents()
                _settle_camera(_own, app)
            if not _surround:
                skip(
                    "pressing F on a selected pair leaves the camera outside the "
                    "protein's own drawn surface, in every representation",
                    f"the search produced no interaction pair ({_own_n} rows), so "
                    f"there is no framing to measure",
                )
                skip(
                    "at that framing the pair is still two drawn discs rather "
                    "than one, in every representation",
                    "no interaction pair, for the same reason",
                )
                skip(
                    "the pair framing depends on what the representation draws",
                    "no interaction pair, for the same reason",
                )
            else:
                # The framing this replaced, computed here so the claim "the eye
                # is outside the surface now and was not before" is one check
                # rather than a number in a comment. The second half is the part
                # that matters: the bound's *tightness* cannot be gated -- any
                # bound can be weakened, and a weaker bound is a weaker claim
                # rather than a wrong one -- but the size of the violation can.
                # If the old framing missed the bound by less than a frame's
                # half-width, a re-tuned bound could hide it.
                _old_bad = [s for s in _surround if s[7] >= 0.0]
                _old_wide = [s for s in _surround if s[7] < -s[8]]
                _slack_ok = (all(s[2] > 0.0 for s in _surround) and not _old_bad)
                check(
                    "pressing F on a selected pair leaves the camera outside the "
                    "protein's own drawn surface, in every representation",
                    _slack_ok,
                    f"the bound is `t >= r / sin({_fs.MAX_DRAWN_ANGLE_FILL} * fov / 2)`, "
                    f"from `asin(r / t)` being the angle a drawn object subtends: "
                    f"there is no tolerance in it to widen, the inequality is the "
                    f"claim. Worst slack per representation (positive means the "
                    f"eye is outside every drawn sphere by that much): "
                    + ", ".join(f"{s[0]} {s[2]:.2f} A" for s in _surround)
                    + f". The context is {_surround[0][4]} atoms plus the pair. "
                    f"The framing this replaced is `max(6.0, radius * 5)`, the "
                    f"same number for all six because it never looks at what is "
                    f"drawn, and at it the same bound reads "
                    + ", ".join(f"{s[0]} {s[7]:.2f}" for s in _surround)
                    + " A against a frame half-width of "
                    + ", ".join(f"{s[0]} {s[8]:.2f}" for s in _surround)
                    + f" A. **Negative in {6 - len(_old_bad)} of the six**, which "
                    f"is the claim: at that distance the eye is inside the drawn "
                    f"envelope in every representation a reader can pick, and the "
                    f"picture is the inside of a surface rather than a view of "
                    f"one. Past a frame half-width in {len(_old_wide)} of the six "
                    f"("
                    + ", ".join(s[0] for s in _old_wide)
                    + f"); the three it is not are the thin-drawing modes, where "
                    f"the drawn half-width is 0.048-0.13 A per atom and being "
                    f"0.03-0.30 A *inside* that envelope is already the "
                    f"violation. So the bound's *tightness* is not gated and cannot "
                    f"be -- any bound can be weakened, and a weaker bound is a "
                    f"weaker claim rather than a wrong one. What is gated is the "
                    f"sign, in all six. An earlier version of this string claimed "
                    f"the violation was past a half-width in all six, on the "
                    f"strength of 205-221 atoms having their drawn surface behind "
                    f"the eye; that count was measured with the *small-sphere* "
                    f"radii, and in `stick` the framing's own half-width is "
                    f"0.048 A, so the envelope is thin and the slack is -0.03 A "
                    f"rather than a crowd of atoms. Both numbers are negative and "
                    f"only the second is the one the rule is about",
                )
                _disc_ok = all(s[5] > s[6] for s in _surround)
                check(
                    "at that framing the pair is still two drawn discs rather "
                    "than one, in every representation",
                    _disc_ok,
                    "projected centre separation against the sum of the two "
                    "projected radii, in pixels; the discs merge when the first "
                    "falls below the second, and there is no threshold to pick "
                    "there. "
                    + ", ".join(f"{s[0]} {s[5]:.0f}/{s[6]:.0f}" for s in _surround)
                    + f" on a {_own.viewport.width()}x{_own.viewport.height()} "
                    f"frame. This replaces a floor on the pair's share of the "
                    f"frame, which had no derivation: the first version asked "
                    f"for 0.40 of the half-frame, borrowed from the pose-"
                    f"selection contract, and the product reached "
                    + ", ".join(f"{s[0]} {s[3]:.3f}" for s in _surround)
                    + ". That number was not a product defect but a dead "
                    f"constraint in the rule -- the context contains the subject, "
                    f"so the subject's floor could never bind -- and it has been "
                    f"removed rather than re-tuned. What bounds the pair's size "
                    f"is `PAIR_CONTEXT_SHELL`, whose sweep is in that constant's "
                    f"comment: 0.060-0.178 of the half-frame at 8.0 A, "
                    f"0.109-0.281 at 6.0 A, 0.040-0.126 at 12.0 A",
                )
                # The rule's *dependence* on the drawing, stated as the
                # direction it has to go in. A representation-independent framing
                # returns one distance for all six, and that is the defect this
                # replaced; asserting only "they differ" would still pass on a
                # rule that differed for no reason.
                _by = {s[0]: s[1] for s in _surround}
                _dep = (_by.get("space_filling", 0.0) > _by.get("spheres", 0.0)
                        and abs(_by.get("ribbon", -1.0)
                                - _by.get("cartoon", -2.0)) < 1e-9)
                check(
                    "the pair framing depends on what the representation draws",
                    _dep,
                    f"CPK radii paint more of each atom than the small-sphere mode "
                    f"does, so the stand-off has to be further out: "
                    f"space_filling {_by.get('space_filling', 0.0):.2f} A against "
                    f"spheres {_by.get('spheres', 0.0):.2f} A, a factor of "
                    f"{_by.get('space_filling', 0.0) / max(_by.get('spheres', 0.0), 1e-9):.2f}, "
                    f"against a painted radius ratio of 2.5x on the pair's own "
                    f"oxygen (0.69 A against 0.31 A) and a context of "
                    f"{_surround[0][4]}+2 drawn discs. `ribbon` and `cartoon` are "
                    f"equal to {_by.get('ribbon', 0.0):.2f} A, which the rule "
                    f"*predicts*: they are drawn from the same swept band and are "
                    f"given the same widest half-width, so treating them as one is "
                    f"a prediction of the rule rather than a coincidence that "
                    f"happens to hold. The old rule gave all six 8.125 A",
                )
            # -- the mutation for the check above ---------------------------
            # **This is where the new check gets its teeth, and it has to run in
            # the same window.** The row stays selected, the selection is never
            # cleared, and the only thing that changes is the draw -- so a count
            # that survives this was never counting the highlight, and the check
            # above would have been a pixel check that cannot fail. Run anywhere
            # else, in a window whose camera has a fourteen-section history, it
            # would have read 0 on both sides and "passed" the red case; that is
            # the failure mode this whole fix exists to remove.
            #
            # The restore goes through `clearSelection()` and not straight back
            # to `setCurrentCell`, because a table does not re-emit
            # `itemSelectionChanged` for a row that is already current. Reading
            # that as "the key did not restore it" would be the wrong lesson;
            # the first version of this check did exactly that and went red.
            # The mutation is **inside** the branch that has something to mutate.
            # It used to sit outside it, so when the pair table was empty the
            # `_own_n == 0` path emitted its skip and then fell through to the
            # mutation, which emitted a *second* skip for the same site -- one
            # check, two rows, and a total one higher than the file declared.
            # A skip that is counted twice is the same defect as a check that
            # cannot fail: it makes the arithmetic a reader trusts untrue.
            _mut_px, _mut_path = -1, "(no draw-off grab was taken)"
            _mut_own = -1
            _mut_why = ""
            if _own_n > 0 and _own_on >= 8 and _own_off == 0:
                # Switch the *draw* off and nothing else. The row stays
                # selected, so this is still the same selection; what changes is
                # the one input `_draw_pair` reads. Set here rather than
                # inherited: this used to depend on the frame the check above
                # happened to leave behind, and anything that ran in between and
                # restored the selection turned the mutation into a second copy
                # of the positive case -- 110 px "switched off", and the frame
                # byte-identical to the selected one, which the check below reads
                # as a red. A mutation that has to be primed by whatever ran
                # before it is not measuring what it claims to.
                _own.viewport.highlight_pair = None
                for _ in range(4):
                    app.processEvents()
                _mut_arr, _mut_path = shot(_own, "pair_highlight_draw_off")
                _mut_px = _pair_highlight_pixels(_mut_arr)
                # **The zero control, and it is not a tautology.** This frame and
                # `_own_off_arr` both have no highlight drawn, but they get there
                # by different routes: this one keeps the row selected and clears
                # the field, that one clears the selection. Anything `_draw_pair`
                # drew from some *other* input -- a stale buffer, a second
                # highlight, the pocket cloud -- would make them differ, so 0 here
                # is a measurement and not an identity.
                _mut_own = _pair_own_pixels(_mut_arr, _own_off_arr)
                # Put it back the way a user would: clear, then select. The
                # table does not re-emit `itemSelectionChanged` for the row that
                # is already current, so selecting without clearing first is a
                # no-op and the restore would be a false negative.
                _own.pair_table.clearSelection()
                for _ in range(4):
                    app.processEvents()
                _own.pair_table.setCurrentCell(0, 0)
                for _ in range(6):
                    app.processEvents()
                _settle_camera(_own, app)
            else:
                _mut_why = (f"the draw-off grab was not taken because "
                            f"{'the pose produced no interaction pair at all' if _own_n == 0 else f'the first grab held {_own_on} own px and the switched-off one held {_own_off}'}, so 'the pixels went to zero' is "
                            f"not a measurement here")
            _back_own = -1
            if _own_n > 0:
                _back_arr, _back_path = shot(_own, "pair_highlight_restored")
                _back_px = _pair_highlight_pixels(_back_arr)
                _back_own = (_pair_own_pixels(_back_arr, _mut_arr)
                             if _mut_arr is not None else -1)
            else:
                _back_arr, _back_path, _back_px = None, "(no pair to restore)", -1
            if _mut_path == "(no draw-off grab was taken)":
                # A skip, and not a red check, because this is a question the
                # platform cannot answer rather than a property it fails: the
                # first grab did not show a highlight, so "the pixels went to
                # zero" has nothing to measure against. The cause is written out
                # in full so it gets reported again next month instead of being
                # quietly dropped -- and so a reader can tell the difference
                # between "the highlight does not come back" and "we could not
                # look". Emitted **only** when there was a pair to begin with;
                # the empty-pair case has already said so once, above.
                if _own_n > 0:
                    skip(
                        "with the row still selected and the draw switched off, "
                        "the highlight's pixels are gone, and they come back",
                        f"not measurable on this run: {_mut_why}; after putting "
                        f"the row back the highlight is "
                        f"{'there' if _back_px >= 8 else 'still absent'} "
                        f"({_back_px} px)",
                    )
            else:
                _mut_frames_differ = not np.array_equal(_mut_arr, _own_on_arr)
                check(
                    "with the row still selected and the draw switched off, the "
                    "highlight's pixels are gone, and they come back",
                    (_mut_own == 0 and _back_own >= 8 and _mut_frames_differ
                     and _own.viewport.highlight_pair is not None),
                    f"row 1 still selected throughout, `highlight_pair` cleared "
                    f"by hand with no re-selection and no camera move. The "
                    f"draw-off frame owns {_mut_own} px against the cleared-"
                    f"selection frame -- **0, and that is the control**: the two "
                    f"reach an undrawn highlight by different routes, so anything "
                    f"the draw took from any other input would show up here. A "
                    f"purity count on the same frame reads {_mut_px} px. Then put "
                    f"the row back the way a user would -- clear, then select -- "
                    f"and the highlight owns {_back_own} px again ({_back_path}), "
                    f"with `highlight_pair` = "
                    f"{_own.viewport.highlight_pair}. The draw-off frame differs "
                    f"from the selected frame as bytes: "
                    f"{_mut_frames_differ}, which is what says the switch landed "
                    f"rather than the draw being quietly idle. Same window as "
                    f"the count above ({_own_on} own px on, {_own_off} purity px "
                    f"off), so the camera is one thing across all three grabs",
                )
            # ------------------------------------------------- the width mutation
            # **The mutation that separates "findable" from "has pixels".**
            # Everything else here mutates by switching the draw *off*, which
            # any instrument can see. This one mutates the marker's **width**,
            # back to the 1 px `Context.line()` the driver clamps to, and leaves
            # the selection, the camera, the colour, the length and the
            # difference count exactly as they were. A marker that is present and
            # findable and a marker that is present and *not* findable then differ
            # only in the number this check asks about.
            #
            # The check passes by **observing the mutation separate them**: the
            # 1 px draw has to clear the old floor (own px >= 8) *and* fail
            # `marker_findable`. If the findability test passed on the 1 px line,
            # this check goes red, which is the only thing that makes the
            # findability half of the check above worth having.
            _mutw_lines = []
            _mutw_own = -1
            _mutw_find = None
            _mutw_fill = None
            _mutw_width = None
            _mutw_declared = None
            if _own is None or _own_n == 0:
                skip(
                    "a 1 px marker is present and not findable, which is what "
                    "makes the width test worth more than a pixel count",
                    "no interaction pair to highlight, so there is no marker to "
                    "make thin",
                )
            else:
                _cls = type(_own.viewport)
                _saved_draw = _cls._draw_pair
                try:
                    _cls._draw_pair = _one_pixel_pair_line
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(0, 0)
                    for _ in range(6):
                        app.processEvents()
                    _settle_camera(_own, app)
                    _mw_seg = _own.viewport.pair_screen_segment()
                    _mw_on = _grab_live(_own)
                    _own.viewport.highlight_pair = None
                    for _ in range(3):
                        app.processEvents()
                    _mw_off = _grab_live(_own)
                    _own.viewport.highlight_pair = _own_got
                    _mw_mark = _pair_marker(_mw_seg, _mw_on, _mw_off)
                    if _mw_mark is not None and _mw_mark.get("own_px", 0) > 0:
                        _mutw_own = _mw_mark["own_px"]
                        _mutw_width = _mw_mark["measured_px"]
                        _mutw_declared = _mw_mark["declared_px"]
                        _mutw_fill = _mw_mark["fill"]
                        _mutw_find = marker_findable(_mw_mark)
                finally:
                    _cls._draw_pair = _saved_draw
                    # Put the selection back the way a user leaves it: clear,
                    # then select. The table does not re-emit
                    # `itemSelectionChanged` for the row that is already current.
                    _own.pair_table.clearSelection()
                    for _ in range(3):
                        app.processEvents()
                    _own.pair_table.setCurrentCell(0, 0)
                    for _ in range(6):
                        app.processEvents()
                _mutw_lines = [
                    f"{k} {own} px, {m['measured_px']:.1f} of "
                    f"{m['declared_px']:.1f} px declared, fill {m['fill']:.2f}"
                    for k, own, _p, m in _repr_sweep
                    if m and m.get("own_px", 0) > 0
                ]
                check(
                    "a 1 px marker is present and not findable, which is what "
                    "makes the width test worth more than a pixel count",
                    (_mutw_own >= 8 and _mutw_find is False),
                    f"the draw was replaced with the 1 px `Context.line()` the "
                    f"driver clamps to -- the same selection, the same camera, "
                    f"the same colour, the same difference instrument -- and it "
                    f"owns **{_mutw_own} px**, which clears the floor of 8 the "
                    f"old check used and on its own would have been reported as "
                    f"a healthy highlight. It is measured at "
                    f"{_mutw_width} px across against the "
                    f"{_mutw_declared} px the projection says the product draws, "
                    f"a fill of {_mutw_fill}, so `marker_findable` returns "
                    f"**{_mutw_find}**: present, and not findable. The real draw "
                    f"measures " + "; ".join(_mutw_lines) + ". A pixel count "
                    f"cannot tell those two rows apart, because they differ by "
                    f"nothing a count can see; a width measured against the "
                    f"product's own declaration can, and the declared width is "
                    f"the pose's bond radius projected, so it moves with the "
                    f"camera instead of being tuned to this frame",
                )
            # Closed on **both** paths out of the mutation block, not just the one
            # that reached its `check`. A window left open leaks a GL context for
            # the rest of the run, and the next section's measurements would be
            # the ones paying for it.
            if _own is not None:
                _own.close()
                for _ in range(8):
                    app.processEvents()

        # -- the keys -------------------------------------------------------
        # No shortcut is installed for these two, on purpose: the focused
        # QTableWidget answers them itself and fires `itemSelectionChanged`,
        # which is the signal a click fires. So the check is that the key moves
        # the selection **and** that nothing in the window is listening for it.
        kw.pair_table.setFocus()
        kw.pair_table.setCurrentCell(_pick, 0)
        for _ in range(4):
            app.processEvents()
        _before = (kw.pair_table.currentRow(), kw.viewport.highlight_pair)
        QTest.keyClick(kw.pair_table, QtCore.Qt.Key.Key_Down)
        for _ in range(6):
            app.processEvents()
        _down_row = kw.pair_table.currentRow()
        _down_pair = kw.viewport.highlight_pair
        _down_contact = _contacts[kw._pairs[_down_row].contact_index]
        QTest.keyClick(kw.pair_table, QtCore.Qt.Key.Key_Up)
        for _ in range(6):
            app.processEvents()
        _after = (kw.pair_table.currentRow(), kw.viewport.highlight_pair)
        _pair_rows = [r for r in keymap.SHORTCUTS
                      if r.group == keymap.GROUP_PAIRS]
        _no_handler = all(not r.handler for r in _pair_rows)
        # **No shortcut on Up or Down**, which is not the same claim as "no
        # shortcut at all": the window installs twelve of them for the letter
        # keys, the camera and the overlay, and section 13 checks that list. What
        # would be wrong here is a thirteenth bound to a key the table already
        # answers -- two paths to one behaviour, and the one nobody would test.
        _listeners = [
            s for s in kw.findChildren(QtGui.QShortcut)
            if s.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down)
        ]
        _no_shortcut = not _listeners
        check(
            "the pair table's own Up and Down move the selection, and nothing "
            "in the window is listening for them",
            (_down_row == _before[0] + 1
             and _down_pair == (int(_down_contact.self_index),
                                int(_down_contact.partner_index))
             and _after == _before
             and len(_pair_rows) == 2 and _no_handler and _no_shortcut),
            f"row and highlight were {_before}; Down gave row {_down_row} and "
            f"highlight {_down_pair}, which is contact "
            f"{kw._pairs[_down_row].contact_index} ("
            f"{kw._pairs[_down_row].pose_atom} -> "
            f"{kw._pairs[_down_row].receptor_atom}); Up gave {_after}. The key "
            f"map lists {len(_pair_rows)} rows in {keymap.GROUP_PAIRS!r} "
            f"({', '.join('+'.join(r.keys) for r in _pair_rows)}) and every one "
            f"has no handler to resolve: {_no_handler}. Of the "
            f"{len(kw.findChildren(QtGui.QShortcut))} shortcuts the window "
            f"holds, {len(_listeners)} are bound to Up or Down: "
            f"{[s.key().toString() for s in _listeners]}. So the key and the "
            f"click reach the highlight by the same signal and there is no "
            f"second path to one behaviour",
        )

        # -- a pose change --------------------------------------------------
        # "The selection follows the pose" has two possible meanings and only
        # one of them is safe: a refresh means the molecules on screen changed,
        # so the selected pair belonged to a pose that is no longer there.
        kw.pair_table.setCurrentCell(_pick, 0)
        for _ in range(4):
            app.processEvents()
        _was_on = kw._current_pose
        _held = kw.viewport.highlight_pair
        _other_pose = 1 if _was_on != 1 else 2
        kw.pose_table.setCurrentCell(_other_pose, 0)
        for _ in range(120):
            app.processEvents()
            QTest.qWait(10)
        _switched = (kw._current_pose == _other_pose
                     and kw.viewport.highlight_pair is None
                     and not kw.pair_table.selectionModel().selectedRows()
                     and kw.pair_table.rowCount() == len(kw.viewport.contacts))
        _new_rows = kw.pair_table.rowCount()
        _new_label = kw.lbl_pairs.text()
        kw.pose_table.setCurrentCell(_was_on, 0)
        for _ in range(120):
            app.processEvents()
            QTest.qWait(10)
        check(
            "the selection and the highlight do not survive a pose change, and "
            "the table is the new pose's",
            _switched and kw.viewport.highlight_pair is None,
            f"a row was selected on pose {_was_on} (highlight {_held}); after "
            f"selecting pose {_other_pose} the highlight is "
            f"{kw.viewport.highlight_pair}, the selected rows are "
            f"{len(kw.pair_table.selectionModel().selectedRows())}, and the "
            f"table holds {_new_rows} rows against {len(kw.viewport.contacts)} "
            f"contacts for the pose now on screen, labelled "
            f"{_new_label[:60]!r}. Clearing the highlight and refilling the "
            f"table are the same statement in `_refresh_contacts`, so any path "
            f"that empties the table also empties the highlight",
        )

        # -- an empty table, and what a full one says -----------------------
        _label_full = kw.lbl_pairs.text()
        _real_find = contactmod.find_contacts
        contactmod.find_contacts = lambda pose, receptor: []
        try:
            kw._refresh_contacts()
            for _ in range(6):
                app.processEvents()
            _rows_empty = kw.pair_table.rowCount()
            _highlight_empty = kw.viewport.highlight_pair
            _label_empty = kw.lbl_pairs.text()
        finally:
            contactmod.find_contacts = _real_find
            kw._refresh_contacts()
            for _ in range(6):
                app.processEvents()
        _label_back = kw.lbl_pairs.text()
        check(
            "an empty pair table says why it is empty, and a filled one says "
            "what its term column is",
            (_rows_empty == 0 and _highlight_empty is None
             and "no pairs" in _label_empty and "4.0 A" in _label_empty
             and _label_back == _label_full
             and "pairs" in _label_full and "hb" in _label_full),
            f"with nothing found: {_rows_empty} rows, the label reads "
            f"{_label_empty[:140]!r}, and the highlight is {_highlight_empty}. "
            f"With the contacts back the label is {_label_back[:140]!r}. Two "
            f"terms are named ({contactmod.TERM_BY_KIND}) and the other two "
            f"kinds are left blank on purpose: the engine sums `g1`, `g2` and "
            f"`rep` over every atom pair in the pose and reports no per-pair "
            f"share, so a number in those cells would be a number nobody "
            f"computed",
        )

        _map_rows = [r for r in keymap.rows(kw)
                     if r["group"] == keymap.GROUP_PAIRS]
        _titles = keymap.group_titles()
        _html = keymap.map_html(kw, kw._keys_pose_summary())
        # The *heading*, not the group name. The group is `keymap`'s word for a
        # focus context and never appears in the map; the heading is what a
        # reader sees under the rows, and it is the heading that has to be there.
        # The first version of this looked for the group name and read the
        # missing heading as a missing group, which is the kind of check that
        # sends the next person to look in the wrong place.
        _pair_heading = _titles[keymap.GROUP_PAIRS]
        check(
            "the key map lists the pair table's keys under their own heading, "
            "and every group in the order has a heading",
            (len(_map_rows) == 2
             and all(r["installed"] is False for r in _map_rows)
             and keymap.GROUP_PAIRS in keymap.GROUP_ORDER
             and _pair_heading in _html
             and all(g in _titles for g in keymap.GROUP_ORDER)
             and all(_titles[g] in _html for g in keymap.GROUP_ORDER)),
            f"{len(_map_rows)} rows in {keymap.GROUP_PAIRS!r}: "
            + "; ".join(f"{'+'.join(s.keys)} -> {s.action}" for s in _pair_rows)
            + f", none of them installed: "
            f"{all(r['installed'] is False for r in _map_rows)}. The group is in "
            f"the order {list(keymap.GROUP_ORDER)}; the map draws it under "
            f"{_pair_heading!r}, which is in the HTML: "
            f"{_pair_heading in _html}; and every group in the order has a "
            f"heading that is in the HTML: "
            f"{all(_titles[g] in _html for g in keymap.GROUP_ORDER)}. The "
            f"headings come from `group_titles()`, called per map rather than "
            f"built once at import, so editing one changes the map -- which is "
            f"what section 13's heading mutation measures",
        )

        # -- a limit, measured rather than asserted -------------------------
        # What was originally written here was "the residue table keeps its row
        # across a pose change", and it was **wrong**: this check is what said
        # so. The selection does not survive, because `_refresh_contacts` empties
        # the table with `setRowCount(0)` before refilling it, and that is the
        # `9 -> 0` case in which Qt drops a selected row 0 even though `7 -> 9`
        # and `9 -> 4` keep it. (Measured on the widget itself as well, so the
        # claim is about `setRowCount` and not about one docking run.)
        #
        # So what is asserted here is the half that is load-bearing in **both**
        # worlds: nothing in the residue path claims anything about the picture.
        # Whether the selection survives is Qt's and the zeroing's business, and
        # pinning either answer would pin an accident -- a Qt that preserved it
        # would be a correct implementation of the same code. What must never
        # happen is a row that is true of the table and false of the screen, so
        # the viewport's residue-shaped claim state is the assertion, and the
        # selection is measured and reported.
        kw.pose_table.setCurrentCell(0, 0)
        for _ in range(6):
            app.processEvents()
        if kw.contact_table.rowCount():
            kw.contact_table.setCurrentCell(0, 0)
            for _ in range(6):
                app.processEvents()
        _res_before = len(kw.contact_table.selectionModel().selectedRows())
        _res_rows_before = kw.contact_table.rowCount()
        _prev_pose = kw._current_pose
        _next_pose = 1 if _prev_pose != 1 else 2
        kw.pose_table.setCurrentCell(_next_pose, 0)
        for _ in range(120):
            app.processEvents()
            QTest.qWait(10)
        _res_after = len(kw.contact_table.selectionModel().selectedRows())
        _res_rows_after = kw.contact_table.rowCount()
        _res_claims = [name for name in dir(kw.viewport)
                       if "highlight" in name.lower()
                       and name != "highlight_pair"]
        check(
            "the residue table's selection across a pose change is Qt's answer, "
            "and nothing in that path claims anything about the picture -- a "
            "limit, not a fix",
            (_res_before == 1
             and kw._current_pose == _next_pose
             and _res_rows_after > 0
             and _res_after in (0, 1)
             and kw.viewport.highlight_pair is None and not _res_claims),
            f"the residue table held {_res_rows_before} row(s) with "
            f"{_res_before} selected on pose {_prev_pose + 1}, and after "
            f"switching to pose {_next_pose + 1} it holds {_res_rows_after} with "
            f"{_res_after} selected. The retention this section was written to "
            f"record does not happen: `_refresh_contacts` empties the table with "
            f"`setRowCount(0)` before refilling, and that is the one row-count "
            f"step Qt answers by dropping the selection (7->9 and 9->4 keep row "
            f"0, measured on the widget). So both answers are accepted and the "
            f"selection is reported, not pinned. What is pinned is the rest: the "
            f"viewport's only highlight state is `highlight_pair`, which the pair "
            f"table owns and which is {kw.viewport.highlight_pair}, and "
            f"attributes that could carry a residue-shaped claim: "
            f"{_res_claims or 'none'}. There is nothing on the screen for a "
            f"retained row to be wrong about today; the day there is, this goes "
            f"red. The limit is written out in `workbench/contacts.py` under 'A "
            f"limit, recorded because it was measured rather than assumed'",
        )

    # ------------------------------------------------- 14b. the floors, live
    # The two marker constants are measured, and their measurement used to live
    # only in a scratch directory. This section re-measures it, so the numbers
    # in `workbench/app.py` are re-checkable rather than asserted. It is
    # `pixel_check` throughout, which adds 4 results and 0 call sites -- see the
    # section's own comment for why that distinction matters here.
    # ------------------------------------------- 14b. the floors, re-measured
    section("14b. the two marker floors, measured again on this machine")
    #
    # `PAIR_MARKER_MIN_AREA_PX2` and `PAIR_MARKER_MIN_LENGTH_PX` are measured
    # constants whose measurement lived only in a scratch directory, and this
    # section is the answer to "reproducible from something in the tree". It
    # re-measures the two facts both constants rest on, on whatever machine is
    # running this file, and reports the numbers next to the constants.
    #
    # **The floors have to be lifted to be measured, and that is the subtlety
    # rather than a detail.** `_pair_quad` returns `None` when `marker_drawable`
    # is false and `_draw_pair` returns early on a `None` quad, so with the
    # floors in place the product declines to submit precisely the rods the
    # floors are derived from. Measured that way, "the first camera that puts a
    # pixel down" is the *length* floor's edge -- 1.0 px of length, about 6.45
    # px^2 at a 6.45 px declared width -- and the area floor's own edge is
    # never reached. Lifting both to zero submits the real geometry through the
    # real draw and changes nothing else about it; the constants go back
    # afterwards, and the last check here is that they did.
    #
    # **The column count is taken in framebuffer coordinates.** An earlier
    # version of this measurement binned along the mark's axis from the mark's
    # *own start point*, which returns 1 for any mark at or below one pixel long
    # whatever the pixel grid does -- so it restated the length instead of
    # counting columns, and a guarantee built on it could not fail. Binning
    # `floor(xs * ax + ys * ay)` over the lit pixels is what makes the mark's
    # position in the grid matter, which is the question the phases ask.
    #
    # **Every check here is `pixel_check` and not `check`, which is a census
    # decision as much as a truth one.** These claims live or die on
    # `grabFramebuffer`, a property of the environment and not of the
    # workbench. It is also the only way to add results here without moving the
    # call-site census in `check_scripts_declare.py`, whose declared column *for
    # this file* lives in that file's generated snapshot and is already behind
    # this file before this section exists (derived 110+187 against declared
    # 85+160). `EXPECTED_CHECKS` moves by the four below, because a pin is how
    # many results a run records and the census is how many result *sites* the
    # source has: +4 results, +0 call sites.
    from opendocking.workbench import app as _mkr_mod

    _mkr_area0 = _mkr_mod.PAIR_MARKER_MIN_AREA_PX2
    _mkr_len0 = _mkr_mod.PAIR_MARKER_MIN_LENGTH_PX
    _mkr_ladder = [10.0 ** e for e in np.linspace(-5.0, 1.0, 25)]
    #: How many probes `_mkr_probe` took, and how many of those it refused
    #: because the frame was not at rest. Both are reported in the area-floor
    #: detail, because an instrument that silently drops samples can otherwise
    #: shrink its own evidence without saying so.
    _mkr_probes = [0]
    _mkr_unstable = [0]

    @contextlib.contextmanager
    def _mkr_lift():
        _mkr_mod.PAIR_MARKER_MIN_AREA_PX2 = 0.0
        _mkr_mod.PAIR_MARKER_MIN_LENGTH_PX = 0.0
        try:
            yield
        finally:
            _mkr_mod.PAIR_MARKER_MIN_AREA_PX2 = _mkr_area0
            _mkr_mod.PAIR_MARKER_MIN_LENGTH_PX = _mkr_len0

    def _mkr_basis(vp_, fwd):
        """Force the camera's basis to `fwd`.

        `Camera.basis` is built from `yaw`/`pitch` in float32 and the round trip
        loses about seven digits, which at the 1e-5 deg end of this ladder is the
        size of the angle itself.
        """
        f = np.asarray(fwd, np.float64)
        f = f / max(float(np.linalg.norm(f)), 1e-300)
        hint = np.array([0.0, 0.0, 1.0])
        if abs(float(f @ hint)) > 0.999:
            hint = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, hint)
        r = r / max(float(np.linalg.norm(r)), 1e-300)
        up = np.cross(r, f)
        r32 = r.astype(np.float32)
        up32 = up.astype(np.float32)
        f32 = f.astype(np.float32)
        vp_.camera.basis = lambda: (r32, up32, f32)

    def _mkr_aim(vp_, mid, fwd, dist):
        cam = vp_.camera
        cam.center = np.asarray(mid, np.float32)
        cam.distance = float(dist)
        f = np.asarray(fwd, np.float64)
        f = f / max(float(np.linalg.norm(f)), 1e-300)
        cam.pitch = float(math.asin(max(-1.0, min(1.0, -float(f[2])))))
        vp_.update()
        for _ in range(3):
            vp_.repaint()
            app.processEvents()

    def _mkr_probe(vp_):
        """The marker's own pixels: a difference of two grabs at one camera.

        **And the frame has to be at rest before that difference means
        anything.** A difference of two grabs attributes every differing pixel
        to the marker, which is only true if the *only* thing that changed
        between them was `highlight_pair`. A surface that is still compositing
        breaks that, and it breaks it silently: the read comes back, the count
        comes back, and the count is fiction.

        This is not hypothetical. `_mkr_edge` below takes the first camera whose
        lit count is above zero and walks the bisection onto it, and at
        25.17 A the edge search reproduced six times on an idle machine gave
        0.237 px of length four times and **0.008 px twice** -- the two bad
        passes locking onto a camera where a *single* pixel crossed the 8/255
        threshold, against seven pixels at the real edge. The check's band is
        [0.15, 0.35] px, so one compositing artefact is a red out of a blue sky
        and a coin flip decides it.

        So the `off` read is taken **twice**, with a repaint between, and the
        two must be identical. If they are not, the surface is still moving and
        the sample is not evidence about the marker, so it is refused rather
        than counted. This is deliberately the stricter direction: it can only
        remove samples, and the count it removes is reported in the detail
        below so a run cannot quietly shrink its own evidence.
        """
        seg = vp_.pair_screen_segment()
        on = _grab_array(vp_)
        got = vp_.highlight_pair
        vp_.highlight_pair = None
        for _ in range(3):
            vp_.repaint()
            app.processEvents()
        off = _grab_array(vp_)
        vp_.highlight_pair = got
        for _ in range(3):
            vp_.repaint()
            app.processEvents()
        if seg is None or on is None or off is None:
            return None
        # The quiescence test. Same state, same camera, two reads.
        vp_.highlight_pair = None
        for _ in range(3):
            vp_.repaint()
            app.processEvents()
        off2 = _grab_array(vp_)
        vp_.highlight_pair = got
        for _ in range(3):
            vp_.repaint()
            app.processEvents()
        _mkr_probes[0] += 1
        if off2 is None or not np.array_equal(off, off2):
            _mkr_unstable[0] += 1
            return None
        d = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2)
        m8 = d >= 8
        row = {"area": float(_mkr_mod._projected_area_px2(seg)),
               "len": float(seg["length"]), "width": float(seg["width"]),
               "n8": int(m8.sum())}
        if m8.any():
            ys, xs = np.nonzero(m8)
            ax, ay = seg["axis"]
            row["cols"] = int(len(np.unique(np.floor(xs * ax + ys * ay))))
        else:
            row["cols"] = 0
        return row

    def _mkr_shoot(vp_, g, deg, dist):
        t = math.radians(deg)
        fwd = math.cos(t) * g["u"] + math.sin(t) * g["w"]
        _mkr_basis(vp_, fwd)
        _mkr_aim(vp_, g["mid"], fwd, dist)
        r = _mkr_probe(vp_)
        if r is not None:
            r["deg"] = deg
        return r

    def _mkr_edge(vp_, g, dist):
        """The first-lit camera at one distance, bisected from a log ladder."""
        rows = [r for r in (_mkr_shoot(vp_, g, dd, dist) for dd in _mkr_ladder)
                if r]
        if not rows:
            return None
        rows.sort(key=lambda r: r["area"])
        lit = [r for r in rows if r["n8"] > 0]
        if not lit:
            return None
        first = lit[0]
        dark = [r for r in rows if r["n8"] == 0 and r["area"] < first["area"]]
        if not dark:
            return None
        lo, hi = dark[-1]["deg"], first["deg"]
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            r = _mkr_shoot(vp_, g, mid, dist)
            if r is None or r["n8"] > 0:
                hi, first = mid, (r or first)
            else:
                lo = mid
        return first

    def _mkr_at(vp_, g, dist, want, iters=6, tol=0.02):
        """Aim at a wanted screen length: bracket, then a secant on the angle.

        **Not a bisection over [0, 45 deg].** That was the first version here and
        it was silently wrong for wide markers: eight steps over 45 deg resolve
        0.18 deg, a 500 px marker needs 0.11 deg to be 1.0 px long, and every
        requested length below about 2 px collapsed onto one camera -- a
        measurement of the floor that was really a measurement of the
        instrument. Bracketing off the broadside length and then scaling the
        angle by `want / measured` converges in two or three passes at any
        width, and `tol` is enforced rather than hoped for.
        """
        exact = g["u"] * 0.0 + g["w"] * 1.0
        _mkr_basis(vp_, exact)
        _mkr_aim(vp_, g["mid"], exact, dist)
        seg = vp_.pair_screen_segment()
        if seg is None:
            return None
        broad = float(seg["length"])
        if broad <= 0.0:
            return None
        theta = math.asin(min(1.0, max(1e-9, want / broad)))
        for _ in range(iters):
            fwd = math.cos(theta) * g["u"] + math.sin(theta) * g["w"]
            _mkr_basis(vp_, fwd)
            _mkr_aim(vp_, g["mid"], fwd, dist)
            seg = vp_.pair_screen_segment()
            if seg is None:
                continue
            got = float(seg["length"])
            if got <= 0.0:
                break
            if abs(got - want) <= tol * max(want, 1e-9):
                break
            theta = max(0.0, min(math.radians(89.0), theta * (want / got)))
        return theta

    def _mkr_band(vp_, g, dist, want, phases=8):
        """Column count at `phases` sub-pixel positions along the mark's axis.

        The phase is a pan along the mark's own screen axis, which slides it
        along the direction that decides which cells it covers at fixed length.
        `_pan` divides its x term by the widget width and its y term by the
        widget height where the projection wants one denominator, so the
        achieved slide is read off the segment's own start point rather than
        assumed from the pan amount.
        """
        t0 = _mkr_at(vp_, g, dist, want)
        cols = []
        for j in range(phases):
            fwd = math.cos(t0) * g["u"] + math.sin(t0) * g["w"]
            _mkr_basis(vp_, fwd)
            _mkr_aim(vp_, g["mid"], fwd, dist)
            if vp_.pair_screen_segment() is None:
                continue
            vp_._pan(0.14 * j, 0.14 * j)
            vp_.update()
            for _ in range(3):
                vp_.repaint()
                app.processEvents()
            r = _mkr_probe(vp_)
            if r is not None:
                cols.append(r)
        return cols

    # -- the fixture. One window: the gate already opens many, and a second one
    # in this section is how the run starts losing its GL context.
    _mw = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                     ligand=EXAMPLES / "biotin_prep.pdbqt")
    _mw.resize(1400, 850)
    _mw.show()
    for _ in range(24):
        app.processEvents()
    for _s in _mw.size_spins:
        _s.setValue(22.0)
    for _c in _mw.center_spins:
        _c.setValue(0.0)
    _mw.sp_exhaust.setValue(2)
    _mw.sp_seed.setValue(20260901)
    for _ in range(6):
        app.processEvents()
    _mw.btn_dock.click()
    for _ in range(4000):
        _t = getattr(_mw, "_thread", None)
        if _t is None:
            break
        try:
            if not _t.isRunning():
                break
        except RuntimeError:
            break
        app.processEvents()
        QTest.qWait(10)
    for _ in range(60):
        app.processEvents()
        QTest.qWait(10)
    _mw.pair_table.setCurrentCell(0, 0)
    for _ in range(8):
        app.processEvents()
    _settle_camera(_mw, app)
    _mvp = _mw.viewport
    _sol = _mvp._pair_atoms()
    if _sol is None or _mw.pair_table.rowCount() == 0:
        for _name in (
            "the rasteriser needs about a quarter of a pixel of screen length "
            "before it lights anything, and that is a property of the pixel "
            "grid rather than of the marker's width",
            "at 1.25 px of length the lit set reaches two framebuffer pixel "
            "columns at every sub-pixel phase, at every declared width",
            "at the shipped 1.0 px floor it does not, at the two thinnest "
            "widths -- so the constant is a geometric floor and not the "
            "two-column one its comment used to claim",
            "the floor measurement put the two shipped constants back",
        ):
            skip(_name, "the fixed-seed search produced no interaction pair "
                        "on this fixture to measure")
    else:
        _p0 = np.asarray(_sol[2], np.float64)
        _p1 = np.asarray(_sol[3], np.float64)
        _u = _p1 - _p0
        _u = _u / max(float(np.linalg.norm(_u)), 1e-300)
        _h = np.array([0.0, 0.0, 1.0])
        if abs(float(_u @ _h)) > 0.9:
            _h = np.array([0.0, 1.0, 0.0])
        _w = np.cross(_u, _h)
        _w = _w / max(float(np.linalg.norm(_w)), 1e-300)
        _g = {"u": _u, "w": _w, "mid": 0.5 * (_p0 + _p1)}

        with _mkr_lift():
            _edges = []
            for _dist in (140.0, 25.17, 8.0):
                _e = _mkr_edge(_mvp, _g, _dist)
                if _e is not None:
                    _edges.append((_dist, _e))
        _lens = [e["len"] for _d, e in _edges]
        _areas = [e["area"] for _d, e in _edges]
        _lens_ok = bool(_lens) and all(0.15 <= v <= 0.35 for v in _lens)
        _spread = (max(_areas) / min(_areas)) if len(_areas) > 1 else 1.0
        pixel_check(
            "the rasteriser needs about a quarter of a pixel of screen length "
            "before it lights anything, and that is a property of the pixel "
            "grid rather than of the marker's width",
            _lens_ok and len(_edges) == 3 and _spread > 2.0,
            "the first-lit camera at each declared width, as a length and as "
            "an area: "
            + "; ".join(
                f"{d} A -> {e['width']:.3f} px wide, first lit at "
                f"{e['len']:.3f} px long = {e['area']:.5f} px^2"
                for d, e in _edges)
            + f". The lengths agree to "
              f"{(max(_lens) - min(_lens)) / max(min(_lens), 1e-9) * 100:.0f}% "
              f"while the areas differ by {_spread:.1f}x, which is the finding: "
              f"the edge is a length, so a single width-independent *area* "
              f"cannot be its shape, and PAIR_MARKER_MIN_AREA_PX2 is the "
              f"conservative end of a width-independent choice rather than a "
              f"point on a measured bracket. Each camera's pixels are a "
              f"difference of two grabs, and a grab pair is only evidence if "
              f"the surface was at rest while it was taken, so every probe "
              f"reads the un-highlighted frame twice and refuses the sample if "
              f"the two disagree: {_mkr_probes[0]} probes, "
              f"{_mkr_unstable[0]} refused as not at rest. Before that guard "
              f"this same search gave 0.237 px four times out of six and "
              f"0.008 px twice, on one pixel above threshold against seven at "
              f"the real edge, which is a red out of a blue sky decided by a "
              f"coin flip"
        )

        with _mkr_lift():
            _b125 = {}
            _b100 = {}
            for _dist in (140.0, 25.17, 11.0):
                for _want, _sink in ((1.25, _b125), (1.00, _b100)):
                    _rows = _mkr_band(_mvp, _g, _dist, _want)
                    if _rows:
                        _sink[round(_rows[0]["width"], 2)] = [
                            r["cols"] for r in _rows]
        _at125 = {w: min(v) for w, v in _b125.items()}
        pixel_check(
            "at 1.25 px of length the lit set reaches two framebuffer pixel "
            "columns at every sub-pixel phase, at every declared width",
            bool(_at125) and all(v >= 2 for v in _at125.values()),
            "minimum column count over eight phases along the mark's own axis, "
            "by declared width, at 1.25 px of length: "
            + ", ".join(f"{w} px -> {v}" for w, v in sorted(_at125.items()))
            + ". This is the property PAIR_MARKER_MIN_LENGTH_PX was derived "
              "from and does not deliver at 1.0 px; it is asserted here so a "
              "rasteriser that stopped providing it would be caught"
        )
        _at100 = {w: min(v) for w, v in _b100.items()}
        _thin = sorted(w for w, v in _at100.items() if v < 2)
        pixel_check(
            "at the shipped 1.0 px floor it does not, at the two thinnest "
            "widths -- so the constant is a geometric floor and not the "
            "two-column one its comment used to claim",
            bool(_thin),
            "minimum column count over eight phases at 1.00 px of length, by "
            "declared width: "
            + ", ".join(f"{w} px -> {v}" for w, v in sorted(_at100.items()))
            + f". Widths below two columns at some phase: {_thin or 'none'}. "
              f"A check that asserts a *limitation* is still worth having: if a "
              f"future rasteriser or driver makes 1.0 px sufficient, this goes "
              f"red and the constant's comment can be simplified rather than "
              f"left claiming something the machine stopped providing"
        )
        pixel_check(
            "the floor measurement put the two shipped constants back",
            (_mkr_mod.PAIR_MARKER_MIN_AREA_PX2 == _mkr_area0
             and _mkr_mod.PAIR_MARKER_MIN_LENGTH_PX == _mkr_len0),
            f"the section lifted both to zero to submit the rods the floors "
            f"refuse, and they read back as "
            f"{_mkr_mod.PAIR_MARKER_MIN_AREA_PX2} px^2 and "
            f"{_mkr_mod.PAIR_MARKER_MIN_LENGTH_PX} px. The three checks above "
            f"are worth nothing if the instrument left the constants moved"
        )
        _mvp.camera.__dict__.pop("basis", None)
    # **Default `destroy=True`, and deliberately.** `dispose_window` documents
    # `destroy=False` as being for the one window that has run a worker thread
    # *and is the last window in the run*, because destroying a QOpenGLWidget
    # whose QThread is still an object fail-fasts on this machine. That does not
    # apply here: the thread object was deleted the moment the search finished
    # (the wait loop above watches for `_thread` becoming `None`), and two
    # sections follow this one. Leaving the window alive instead kept its search
    # temp file on disk, and the "docking does not leak temp files" check in
    # section 16 globs `%TEMP%` and found it -- a red in this section reported
    # by a later one, which is the worst place for one.
    dispose_window(app, _mw)
    # Everything above this section is on one machine until now. The export is
    # the first thing in this file whose whole point is that it leaves, so the
    # checks here are mostly about the file rather than the window -- and about
    # the *absences* in it, which is the part a screenshot could never show.
    #
    # The reader is `json.load`. Not `export.read_export`, and not the writer's
    # own objects: a reader that shares code with the writer cannot disagree
    # with it, and a round trip proved that way proves nothing. Everything below
    # is compared against the objects the window is showing.
    section("15. taking the run away")
    import hashlib as _hashlib
    import json as _json
    import time as _time

    from opendocking.workbench import pose_trust as _pt

    # The gate's own answer to "where does the file go", not the module's: the
    # window holds its own root and this asks the window, so the check is on
    # the same directory the export is about to use rather than on a second
    # guess at it.
    _EXPORT_ROOT = Path(kw._export_root) / "dist" / "exports"

    # -- a run, exported by the key path ------------------------------------
    _t0 = _time.perf_counter()
    _run_path = kw._export_run()
    _run_ms = (_time.perf_counter() - _t0) * 1000.0
    _run_doc = (_json.loads(_run_path.read_text(encoding="utf-8"))
                if _run_path else {})
    _r = kw._dock_result
    _n = _run_doc.get("pose_count", 0)
    _aff_ok = all(
        p["affinity_kcal_per_mol"]["state"] == "measured"
        and abs(p["affinity_kcal_per_mol"]["value"] - float(_r.energies[i])) == 0.0
        for i, p in enumerate(_run_doc.get("poses", ()))
    )
    _rmsd_ok = all(
        p["rmsd_to_best"]["state"] == "measured"
        and abs(p["rmsd_to_best"]["value"] - float(_r.rmsds[i])) == 0.0
        for i, p in enumerate(_run_doc.get("poses", ()))
    )
    _inter_ok = all(
        p["intermolecular_kcal_per_mol"]["state"] == "measured"
        and abs(p["intermolecular_kcal_per_mol"]["value"]
                - float(_r.intermolecular_energies[i])) == 0.0
        for i, p in enumerate(_run_doc.get("poses", ()))
    )
    # The coordinates three times, against three different things, because they
    # are three different claims and only two of them are evidence.
    #
    # `_xyz_engine` is the loose one: the engine's values are finer than the
    # picture, so this is a tolerance and it is reported as one.
    #
    # `_xyz_3dp` is the strong one and it is exact: `coords_source` claims three
    # decimal places, and `round(engine, 3)` cast back to float32 -- which is
    # what a view holding float32 coordinates is -- equals the file for every
    # atom of every pose, to the bit. It is checked against an array the writer
    # never read, so it is a claim somebody else can repeat.
    #
    # `_xyz_view` compares the file to `win._pose_ghosts`, and that is the *same
    # array* the export read. It is here only to say the file holds **this** pose
    # rather than a different one, and it is labelled as the same-object check it
    # is rather than left in a list of round-trip numbers looking like the other
    # two. (Writing that comment cost a probe: the first version asserted
    # `round(engine, 3)` in float64, which is off by 4.5e-07 on this fixture
    # because the view holds float32 and the value is the float32 of the
    # three-decimal number.)
    _file_xyz = ([np.asarray(p["coords"], np.float64)
                  for p in _run_doc.get("poses", ())] if _n else [])
    _engine_xyz = ([np.asarray(_r.pose_coords(i), np.float64)
                    for i in range(_n)] if _n else [])
    _xyz_engine = (max(float(np.abs(f - e).max())
                       for f, e in zip(_file_xyz, _engine_xyz))
                   if _n else float("nan"))
    _xyz_3dp = (max(float(np.abs(f - np.round(e, 3).astype(np.float32)
                               .astype(np.float64)).max())
                    for f, e in zip(_file_xyz, _engine_xyz))
                if _n else float("nan"))
    _xyz_view = (max(float(np.abs(f - np.asarray(v.coords, np.float64)).max())
                     for f, v in zip(_file_xyz, kw._pose_ghosts))
                 if _n and len(kw._pose_ghosts) >= _n else float("nan"))
    check(
        "the exported numbers are the window's own, read back by a reader that "
        "shares no code with the writer",
        (bool(_run_path) and _n == _r.num_poses and _aff_ok and _rmsd_ok
         and _inter_ok and _xyz_engine <= 5e-4 and _xyz_3dp == 0.0),
        f"{_run_path.name if _run_path else 'NOTHING WRITTEN'}: "
        f"{_run_path.stat().st_size if _run_path else 0} B, schema "
        f"{_run_doc.get('schema')!r}, {_n} pose(s) against the result's "
        f"{_r.num_poses}. Every affinity, rmsd and intermolecular energy in the "
        f"file is equal to the DockingResult's own to the bit "
        f"(affinity {_aff_ok}, rmsd {_rmsd_ok}, intermolecular {_inter_ok}). "
        f"The coordinates against the engine's own array, which the export never "
        f"read: {_xyz_engine:.2e} A, which is the three decimals of tolerance "
        f"`coords_source` promises and no more. The same coordinates against "
        f"`round(engine, 3)` as float32: {_xyz_3dp:.2e} A -- exact, for every "
        f"atom of every pose, so the three-decimal claim is a fact about the "
        f"file rather than an assertion in it. The third comparison, against "
        f"`win._pose_ghosts`, reads {_xyz_view:.2e} A, and it is the *same "
        f"array* the export read: it says the file holds this pose and not "
        f"another, and it is not evidence about the writer",
    )

    # -- provenance ---------------------------------------------------------
    _prov = _run_doc.get("provenance", {})
    _want_sha = _hashlib.sha256(kw._receptor_path.read_bytes()).hexdigest()
    _prov_ok = (
        _prov.get("seed") == int(kw.sp_seed.value())
        and _prov.get("exhaustiveness") == int(kw.sp_exhaust.value())
        and _prov.get("scoring") == kw.cb_scoring.currentText()
        and _prov.get("box_centre") == [float(s.value())
                                        for s in kw.center_spins]
        and _prov.get("box_size") == [float(s.value()) for s in kw.size_spins]
        and _prov.get("receptor", {}).get("sha256") == _want_sha
        and bool(_prov.get("engine_version"))
        and bool(_prov.get("backends_available"))
    )
    check(
        "the export carries the run's own provenance, and the receptor's hash "
        "is one this gate computed itself",
        _prov_ok,
        f"seed {_prov.get('seed')} (the spin says {int(kw.sp_seed.value())}), "
        f"exhaustiveness {_prov.get('exhaustiveness')} "
        f"({int(kw.sp_exhaust.value())}), scoring {_prov.get('scoring')!r}, box "
        f"{_prov.get('box_centre')} x {_prov.get('box_size')}, engine "
        f"{_prov.get('engine_version')!r} on {_prov.get('backends_available')}, "
        f"elapsed {_prov.get('elapsed_seconds')} s over "
        f"{_prov.get('pose_count_reported')} reported and "
        f"{_prov.get('pose_count_rejected')} rejected. The receptor's sha256 in "
        f"the file is {_prov.get('receptor', {}).get('sha256', '')[:16]} and "
        f"the gate hashed the same bytes independently to "
        f"{_want_sha[:16]}. An exported number with no box and no seed beside it "
        f"is not a result anybody can reproduce",
    )

    # -- the verdict travelled ----------------------------------------------
    _sel = int(kw._current_pose)
    _pen = (None if kw._terms_breakdown is None
            else float(kw._terms_breakdown.out_of_box_penalty))
    _live = _pt.describe_pose(
        kw._dock_result, _sel, out_of_box_penalty=_pen,
        penalty_pose_index=_sel if _pen is not None else None)
    _exp_v = _run_doc.get("poses", [{}] * (_n or 1))[_sel].get("verdict", {})
    _states_match = ([c["state"] for c in _exp_v.get("contracts", ())]
                     == [row.state for row in _live.rows])
    _reasons_match = all(
        c["because"] == row.because
        for c, row in zip(_exp_v.get("contracts", ()), _live.rows)
    )
    _inside = next((c for c in _exp_v.get("contracts", ())
                    if c["name"] == "inside the box"), {})
    check(
        "the verdict is in the file as it was decided, not re-derived on the "
        "way out",
        (_states_match and _reasons_match
         and _exp_v.get("trust") == _live.trust
         and "NOT recomputed" in _run_doc.get("derived", {}).get("verdict", "")),
        f"pose {_sel + 1}: exported trust {_exp_v.get('trust')!r} and contract "
        f"states {[c['state'] for c in _exp_v.get('contracts', ())]} against "
        f"{_live.trust!r} and {[r.state for r in _live.rows]}; every reason "
        f"string identical: {_reasons_match}. The one that would have shown a "
        f"recomputation is `inside the box` at "
        f"{_inside.get('state')!r} -- a verdict built without the breakdown's "
        f"penalty reads it `unmeasured`, and the file agrees with the panel. "
        f"`derived` says so in the document itself: "
        f"{_run_doc.get('derived', {}).get('verdict', '')[:70]}...",
    )

    # -- absent is distinguishable from zero, and from not run ---------------
    # The load-bearing check of the section, and it is a discrimination test
    # rather than a presence test: one file has to contain a **measured zero**
    # and an **unmeasured null** at the same time, or it proves nothing about
    # telling them apart.
    #
    # Every record-shaped thing in the file contributes its state, not only the
    # three scalars per pose. That is a correction, not a style note: a first
    # version collected only `affinity`/`rmsd`/`intermolecular`/
    # `energy_terms_state`, and for a **run** every one of those is `measured`,
    # so the set never contained `unmeasured` and this check was red on the
    # case it was written for. The unmeasured things in a run's file are the
    # verdict contracts, so the contracts are the states that had to be read.
    _states_seen = set()
    _zero_carriers = []
    for _p in _run_doc.get("poses", ()):
        for _key in ("affinity_kcal_per_mol", "rmsd_to_best",
                     "intermolecular_kcal_per_mol"):
            _states_seen.add(_p[_key]["state"])
        _states_seen.add(_p["energy_terms_state"])
        for _c in _p.get("verdict", {}).get("contracts", ()):
            _states_seen.add(_c["state"])
        for _name, _v in (_p.get("energy_terms") or {}).items():
            if _v == 0.0:
                _zero_carriers.append((_p["index"], _name,
                                       _p["energy_terms"][_name]))
    _absent = _run_doc.get("absent", ())
    _all_explained = all(a.get("because") for a in _absent)
    _indexed = all(
        any(a["pose"] == _p["index"]
            and a["field"].endswith(c["name"])
            for a in _absent)
        for _p in _run_doc.get("poses", ())
        for c in _p["verdict"]["contracts"] if c["state"] == "unmeasured"
    )
    check(
        "one file holds a measured zero and an unmeasured null at the same "
        "time, and every absence is listed with its reason",
        (len(_zero_carriers) >= 1 and {"measured", "unmeasured"} <= _states_seen
         and _all_explained and _indexed
         and _run_doc.get("absent_count") == len(_absent)),
        f"{len(_zero_carriers)} measured zero(s) in the run's file, carried as "
        f"0.0 with state 'measured' -- "
        f"{', '.join(f'pose {p} {n}={v}' for p, n, v in _zero_carriers[:3])}"
        f"{'; ' if len(_zero_carriers) > 3 else ''} -- against "
        f"{_run_doc.get('absent_count')} field(s) that are not measurements, "
        f"every one of them with a reason: {_all_explained}, and every "
        f"unmeasured contract appears in the top-level `absent` index: "
        f"{_indexed}. The states this file actually holds, read out of every "
        f"record in it: {sorted(_states_seen)}. A term of exactly zero is a "
        f"result and a term nobody measured is not, and in this file they are "
        f"not the same line",
    )

    # -- a warning the window speaks, and the absence of one ------------------
    # The engine counts receptor atoms whose PDBQT type it does not recognise
    # and `summary()` prints them as a `WARNING:` line, because such an atom
    # contributes a shape term while losing its hydrogen-bond and hydrophobic
    # character -- so every energy in the run understates the receptor. The
    # window speaks that. The exported file used to be silent about it, and the
    # exported file is the artifact somebody actually takes away, so the
    # engine-layer fix reappeared one layer up.
    #
    # **Two directions, and the second is what gives the first its meaning.** A
    # file saying "5 receptor atoms carry a type this engine does not
    # recognise" is worth reading only if a clean file says *nothing*. One code
    # path, two documents: the same discrimination argument as the measured-zero
    # check above, applied to a warning rather than to a number.
    #
    # **The engine half is conditional and the gate prints which way it went.**
    # The count reached the result after the `_dockpy.pyd` in this checkout was
    # built, so on a stale binary the property raises `AttributeError` and the
    # honest answer is the `absent` state -- a fact about the build, not about
    # the export, and emphatically not a zero. So each check asserts whichever
    # is true and says so. Separately, both assert with a **stand-in result**
    # the one thing a stale build cannot reach: that a count of five becomes a
    # measured 5.0 carrying the sentence. Without the stand-in this section
    # would quietly degrade into testing only the negative direction on any
    # machine nobody has rebuilt.
    from opendocking.workbench.export import (
        unrecognised_atom_types as _unrecognised_atom_types,
    )

    try:
        _build_count = int(kw._dock_result.unknown_atom_types)
        _build_has = True
        _build_why = ""
    except AttributeError as _exc:
        _build_count = None
        _build_has = False
        _build_why = f"{type(_exc).__name__}: {_exc}"

    # A receptor that parses and precalculates but cannot be read: five atoms
    # given a type name AutoDock never defined. Built here rather than
    # committed, so the fixture cannot rot against the parser -- and it is the
    # same edit the engine's own Rust test makes.
    with tempfile.TemporaryDirectory(prefix="odw_exotic_") as _tdir:
        _exotic = Path(_tdir) / "1crn_exotic.pdbqt"
        _corrupted = 0
        _rewritten = []
        for _line in (EXAMPLES / "1crn_prep.pdbqt").read_text(
                encoding="utf-8").splitlines(keepends=True):
            if _line.startswith(("ATOM", "HETATM")) and _line[77:].strip() == "NA":
                _corrupted += 1
                _line = f"{_line[:77]}ZZ{_line[79:]}"
            _rewritten.append(_line)
        _exotic.write_text("".join(_rewritten), encoding="utf-8", newline="")

        # Its own window, its own box, its own export, and its own path: the
        # default export name is `run-{scoring}-seed{seed}-{n}poses.json`, and
        # this run uses the same seed and effort as the gate's own, so letting
        # it choose would overwrite the file every check above just read.
        _xw = MainWindow(receptor=_exotic, ligand=EXAMPLES / "biotin_prep.pdbqt")
        _xw.resize(1200, 800)
        _xw.show()
        for _ in range(20):
            app.processEvents()
        for _s in _xw.size_spins:
            _s.setValue(22.0)
        for _c in _xw.center_spins:
            _c.setValue(0.0)
        _xw.sp_exhaust.setValue(2)
        _xw.sp_seed.setValue(20260901)
        for _ in range(6):
            app.processEvents()
        _xw.btn_dock.click()
        while _xw._thread is not None and _xw._thread.isRunning():
            app.processEvents()
            QTest.qWait(10)
        for _ in range(60):
            app.processEvents()
            QTest.qWait(10)
        _x_path = _xw._export_run(Path(_tdir) / "exotic-export.json")
        _x_doc = (_json.loads(_x_path.read_text(encoding="utf-8"))
                  if _x_path else {})
        _x_said = _xw.statusBar().currentMessage()
        # The engine's own count, taken off the result before the window goes,
        # so the comparison below is against the engine and not against a
        # second reading of the file.
        try:
            _x_engine_count = int(_xw._dock_result.unknown_atom_types)
        except (AttributeError, TypeError):
            _x_engine_count = None
        _xw.close()

    _x_rec = _x_doc.get("provenance", {}).get("receptor", {}).get(
        "unknown_atom_types", {})
    _x_warnings = _x_doc.get("warnings", ())
    _x_indexed = [a for a in _x_doc.get("absent", ())
                  if a.get("field") == "provenance.receptor.unknown_atom_types"]
    _si_rec, _si_warn = _unrecognised_atom_types(
        type("_FiveAtoms", (), {"unknown_atom_types": 5})(),
        source="a stand-in result, so this direction is asserted on any build",
    )
    # **What the file may claim about the count, as a function of the build.**
    # Spelled as two whole expectations rather than as a chain of `or not
    # _build_has`, because the first version of this check was the chain, and it
    # was green on a modern build and red on a stale one for the same reason: it
    # required a warning and a WARNING in the status bar *unconditionally*, and
    # a build that cannot read the count has correctly produced neither. A
    # conditional assertion written as a conjunction of negations is a
    # conditional assertion nobody can read, and it fails in the direction that
    # looks like a product bug and is not.
    if _build_has:
        # The engine answered, so the file must name the count and say what it
        # does to the numbers -- and must not also list the field as absent.
        _x_ok = (float(_x_rec.get("value", -1)) > 0
                 and _x_engine_count is not None
                 and float(_x_rec["value"]) == float(_x_engine_count)
                 and len(_x_warnings) == 1
                 and "understates the receptor" in _x_warnings[0]
                 and "WARNING" in _x_said
                 and not _x_indexed)
    else:
        # The engine could not answer, so the file must say *that* -- listed as
        # an absence, with a reason, and claiming nothing about the receptor. A
        # warning here would be a claim about a count nobody read.
        _x_ok = (len(_x_warnings) == 0 and "WARNING" not in _x_said
                 and len(_x_indexed) == 1
                 and _x_indexed[0].get("state") == "absent"
                 and bool(_x_indexed[0].get("because")))
    check(
        "a receptor the engine cannot fully read is named in the exported file, "
        "or the file says the engine could not count its atom types",
        (bool(_x_path) and _corrupted > 0 and _x_ok
         and _x_rec.get("state") == ("measured" if _build_has else "absent")
         # The stand-in, which no build can refuse to answer. Asserted on every
         # machine, so this check is never only a test of the negative direction.
         and _si_rec.get("state") == "measured"
         and float(_si_rec.get("value", -1)) == 5.0
         and _si_warn is not None
         and "5 receptor atom(s)" in _si_warn
         and "understates the receptor" in _si_warn),
        f"{_x_path.name if _x_path else 'NOTHING WRITTEN'}: "
        f"{_corrupted} atom(s) of type 'NA' rewritten to 'ZZ' in a copy of "
        f"1crn_prep.pdbqt. The build in use "
        f"{'can' if _build_has else 'cannot'} read the count -- the gate's own "
        f"clean run against it reports "
        f"{_build_count if _build_has else 'AttributeError: ' + _build_why[:70]}. "
        f"So the file is required to "
        + ("name the count and the consequence" if _build_has
           else "record the count as absent, list it, and claim nothing")
        + f", and it carries state={_x_rec.get('state')!r} "
        f"value={_x_rec.get('value')!r}, {len(_x_warnings)} warning(s), "
        f"{len(_x_indexed)} entry/entries in `absent`, and the status bar says "
        f"WARNING: {'WARNING' in _x_said}. The stand-in, which every build "
        f"answers: state={_si_rec.get('state')!r} "
        f"value={_si_rec.get('value')!r} with "
        f"{'a' if _si_warn else 'no'} warning",
    )

    # The mirror, on the gate's own clean run, whose file was read into
    # `_run_doc` before any of this. **A warning system that cannot stay quiet
    # is not a warning system**: a file that named a defect on a run where the
    # engine read every atom teaches a reader to skip the line, and the next
    # real one goes unread.
    _c_rec = _run_doc.get("provenance", {}).get("receptor", {}).get(
        "unknown_atom_types", {})
    _c_warnings = _run_doc.get("warnings", ())
    _c_indexed = [
        a for a in _run_doc.get("absent", ())
        if a.get("field") == "provenance.receptor.unknown_atom_types"
    ]
    # Spelled out here because an f-string expression cannot span lines before
    # 3.12, and a detail string that will not parse is a gate that does not run.
    _c_how = ("a measurement, and it is zero" if _build_has
              else "absent, and an absent count is listed as an absence rather "
                   "than as a clean zero")
    # The claim, and it is the same one on either kind of build: **this file
    # does not say anything is wrong with a receptor the engine read cleanly.**
    # A warning is a false claim whatever the build; what differs is whether the
    # count behind it is a zero or an absence, and the second is required to be
    # *listed* rather than to be quiet, or it would be the clean-looking zero
    # this round is about.
    _c_ok = (len(_c_warnings) == 0
             and (not _build_has
                  or (float(_c_rec.get("value", -1)) == 0.0 and not _c_indexed))
             and (_build_has
                  or (len(_c_indexed) == 1
                      and _c_indexed[0].get("state") == "absent"
                      and bool(_c_indexed[0].get("because")))))
    check(
        "a run the engine read cleanly says nothing is wrong with its receptor",
        (_c_ok and _c_rec.get("state") == ("measured" if _build_has else "absent")),
        f"the clean run's file carries state={_c_rec.get('state')!r} "
        f"value={_c_rec.get('value')!r}, {len(_c_warnings)} warning(s) "
        f"({_c_warnings or 'none'}), and {len(_c_indexed)} entry/entries for it "
        f"in the top-level `absent` index. On this build the count is "
        f"{_c_how}. A count of zero here is a result -- every atom was "
        f"recognised -- and it is deliberately not the same line as the exotic "
        f"file's warning",
    )

    # -- a pose file, and the three states in one document -------------------
    # Its own window rather than a leftover from section 5: that one has been
    # through a dozen sections since, and what is being checked here is what a
    # window holding a pose file and nothing else writes. `MainWindow(poses=...)`
    # is the same constructor the product offers, and this is the case that
    # never had a `DockingResult` at all.
    _fw = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                     ligand=EXAMPLES / "biotin_prep.pdbqt",
                     poses=EXAMPLES / "poses.pdbqt")
    _fw.resize(1200, 800)
    _fw.show()
    for _ in range(20):
        app.processEvents()
    _fw.pose_table.setCurrentCell(0, 0)
    for _ in range(4):
        app.processEvents()
    _file_path = _fw._export_run()
    _file_doc = (_json.loads(_file_path.read_text(encoding="utf-8"))
                 if _file_path else {})
    _fp = (_file_doc.get("poses") or [{}])[0]
    _three = (
        _fp.get("affinity_kcal_per_mol", {}).get("state") == "measured"
        and _fp.get("intermolecular_kcal_per_mol", {}).get("state") == "not_run"
        and _fp.get("energy_terms_state") == "unmeasured"
    )
    _explained = all(
        _fp.get(k, {}).get("because")
        for k in ("intermolecular_kcal_per_mol",)
    ) and bool(_fp.get("energy_terms_because"))
    check(
        "a pose-file export names all three states -- measured, not run, and "
        "unmeasured -- in the same file",
        (_three and _explained
         and _fp.get("verdict", {}).get("trust") == "unknown"
         and all(c["state"] == "unmeasured" and c["because"]
                 for c in _fp.get("verdict", {}).get("contracts", ()))
         and _file_doc.get("absent_count", 0) > 0
         # The pose file's own sha256, hashed here rather than read out of
         # the document: a file that vouches for itself is not a witness.
         and _file_doc.get("provenance", {}).get("pose_file", {}).get("sha256")
         == _hashlib.sha256((EXAMPLES / "poses.pdbqt").read_bytes()).hexdigest()),
        f"{_file_path.name if _file_path else 'NOTHING WRITTEN'}: "
        f"{'measured' if _fp.get('affinity_kcal_per_mol', {}).get('state') == 'measured' else '?'}"
        f" affinity "
        f"({_fp.get('affinity_kcal_per_mol', {}).get('value')}, source "
        f"{_fp.get('affinity_kcal_per_mol', {}).get('source')!r}), "
        f"{_fp.get('intermolecular_kcal_per_mol', {}).get('state')} "
        f"intermolecular because "
        f"{_fp.get('intermolecular_kcal_per_mol', {}).get('because', '')[:60]!r}, "
        f"{_fp.get('energy_terms_state')} terms because "
        f"{_fp.get('energy_terms_because', '')[:60]!r}, and a verdict of "
        f"{_fp.get('verdict', {}).get('trust')!r} with "
        f"{len(_fp.get('verdict', {}).get('contracts', ()))} unmeasured "
        f"contract(s), each with its reason. "
        f"{_file_doc.get('absent_count')} absence(s) listed at the top, and the "
        f"pose file's sha256 in `provenance` matches the one this gate hashed "
        f"itself. This is the case the whole design is for: a file that "
        f"omitted those four contracts would read as one where all four held",
    )
    _fw.close()

    # -- the rows in the file are the rows the table is showing -------------
    _table_res = [kw.contact_table.item(i, 0).text()
                  for i in range(kw.contact_table.rowCount())]
    _file_res = [row[0] for row in _run_doc["poses"][_sel]["contacts"]["residues"]]
    _table_pairs = kw._pairs
    _file_pairs = _run_doc["poses"][_sel]["contacts"]["pairs"]
    check(
        "the residue and pair rows in the file are the rows the table is "
        "showing, for the pose on screen",
        (_table_res == _file_res and len(_file_pairs) == len(_table_pairs)
         and all(p.pose_atom == row[0] and p.receptor_atom == row[1]
                 and abs(p.distance - row[3]) == 0.0
                 for p, row in zip(_table_pairs, _file_pairs))),
        f"pose {_sel + 1}: the table shows {_table_res} and the file holds "
        f"{_file_res}; the table has {len(_table_pairs)} pair(s) and the file "
        f"{len(_file_pairs)}, with every pose atom, receptor atom and distance "
        f"equal. The table was filled at selection time by one "
        f"`find_contacts` call and the file at export time by another, so this "
        f"is two calls agreeing rather than one value read twice. `derived` "
        f"says which it is: "
        f"{_run_doc.get('derived', {}).get('contacts', '')[:64]}...",
    )

    # -- nothing loaded ------------------------------------------------------
    _bare = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                       ligand=EXAMPLES / "biotin_prep.pdbqt")
    _bare.resize(1200, 800)
    _bare.show()
    for _ in range(20):
        app.processEvents()
    _files_before = (sorted(p.name for p in _EXPORT_ROOT.glob("*.json"))
                     if _EXPORT_ROOT.is_dir() else [])
    _t0 = _time.perf_counter()
    _bare_written = _bare._export_run()
    _bare_ms = (_time.perf_counter() - _t0) * 1000.0
    _files_after = (sorted(p.name for p in _EXPORT_ROOT.glob("*.json"))
                    if _EXPORT_ROOT.is_dir() else [])
    _said = _bare.statusBar().currentMessage()
    _bare.close()
    check(
        "with no pose loaded the export refuses, says why, and writes nothing",
        (_bare_written is None and _files_after == _files_before
         and "nothing was written" in _said and "no pose" in _said),
        f"returned {_bare_written!r} in {_bare_ms:.1f} ms, the files on disk "
        f"changed by {sorted(set(_files_after) - set(_files_before)) or 'nothing'}, "
        f"and the status bar says: {_said[:150]!r}. A file whose pose list is "
        f"empty is a run that found nothing, and a window nobody has used must "
        f"not be able to produce one",
    )

    # -- the key, and the map ------------------------------------------------
    _export_rows = [s for s in keymap.SHORTCUTS if s.handler == "_key_export"]
    _file_menu = kw.menuBar().actions()[0].menu()
    _menu_item = next((a for a in _file_menu.actions()
                       if a.text() == "Export run to a file"), None)
    # The map a reader presses against: built by the module, and it has to
    # carry the action *and* the note, because the note is where the file's
    # destination and the empty case are said out loud.
    _map_html = keymap.map_html(kw, kw._keys_pose_summary())
    # Whether `E` and the menu item reach one method, **measured through the
    # product's own output** rather than by intercepting the call. Both earlier
    # attempts were wrong in instructive ways: `kw._key_export is kw._export_run`
    # read False (the key handler is a wrapper, not the method), and wrapping
    # `kw._export_run` with a counter also read False for the menu item, because
    # a `QAction` holds the bound method it was connected to, so an attribute
    # swapped in afterwards is never called. So the method is left alone and the
    # *files* are the evidence: the seed is put somewhere distinctive, the key is
    # pressed as a key, then the seed is moved and the menu item is triggered. Two
    # gestures, two files, each naming its own seed -- and neither one needed any
    # Python-side call, so a dead shortcut or an unconnected action would have
    # written nothing at all.
    #
    # **Freshness is proved, not assumed.** This check went red once reading a
    # file named `run-vina-seed20260901-9poses.json` while asserting seed
    # 42420001: the status line had not been updated by the gesture and still
    # held the message from the export earlier in this same section, so the check
    # read a *previous* export's answer and called it "the wrong seed". Those are
    # different defects and only one of them happened, so the line is recorded
    # before each gesture and the check requires it to have **changed**. A line
    # that did not change is reported as its own failure below, with the two
    # candidates separated rather than collapsed.
    _seed_a, _seed_b = 42420001, 42420002
    _paths = []
    _fresh = {}
    for _seed, _fire in ((_seed_a, "key"), (_seed_b, "menu")):
        # The window is made active first, and that is a fix rather than a
        # ceremony: a key event to a window Qt does not consider active is
        # dropped before any shortcut sees it, and section 14 now builds and
        # closes a second window, so the active one at this point in the run is
        # not a thing to rely on. A person pressing `E` has the window focused;
        # the check has to put it in the same state or it is measuring the
        # harness.
        kw.activateWindow()
        kw.raise_()
        app.processEvents()
        kw.sp_seed.setValue(_seed)
        for _ in range(4):
            app.processEvents()
        _before_line = kw.statusBar().currentMessage()
        if _fire == "key":
            QTest.keyClick(kw, QtCore.Qt.Key.Key_E)
        else:
            _menu_item.trigger() if _menu_item is not None else None
        for _ in range(4):
            app.processEvents()
        _said_export = kw.statusBar().currentMessage()
        _fresh[_fire] = (_said_export != _before_line)
        # The status bar is the product's own answer to "did it write, and
        # where", and it prints the full path: `exported N pose(s) to PATH
        # (NkB, M field(s) absent and listed)`. Read the path out of it rather
        # than recomputing the naming rule, so the check is on the message a
        # person sees.
        _path = (Path(_said_export.split(" to ", 1)[-1].split(" (")[0])
                 if "exported " in _said_export else None)
        if _path is not None and _path.is_file():
            _paths.append((_fire, _path,
                           _json.loads(_path.read_text(encoding="utf-8"))))
    # If a line did *not* change, then say which of the two candidates it was --
    # the product not updating its own status line, or the gesture not reaching
    # the window at all. `showMessage` is called synchronously inside
    # `_export_run`, so a handler that runs must move the line; a line that did
    # not move with the handler called directly is a product defect, and one that
    # moves only then was a gesture that never arrived.
    _direct_line = None
    if not all(_fresh.values()):
        _direct_before = kw.statusBar().currentMessage()
        kw._key_export()
        for _ in range(4):
            app.processEvents()
        _direct_line = kw.statusBar().currentMessage() != _direct_before
    _by_fire = {f: (p, d) for f, p, d in _paths}
    _seeds_ok = (
        len(_by_fire) == 2
        and _by_fire["key"][1]["provenance"]["seed"] == _seed_a
        and _by_fire["menu"][1]["provenance"]["seed"] == _seed_b
        and _by_fire["key"][0] != _by_fire["menu"][0]
    )
    kw.sp_seed.setValue(20260901)
    check(
        "the export is one key and one menu item reaching one method, and the "
        "map says where the file goes",
        (len(_export_rows) == 1 and _export_rows[0].keys == ("E",)
         and _menu_item is not None and _seeds_ok
         and all(_fresh.values())
         and "Export the run" in _map_html
         and "status bar" in _export_rows[0].note
         and "writes nothing" in _export_rows[0].note),
        f"keys.py declares {len(_export_rows)} row(s) for it: "
        f"{[('+'.join(s.keys), s.action) for s in _export_rows]}, in group "
        f"{_export_rows[0].group!r}; the File menu offers "
        f"{[a.text() for a in _file_menu.actions() if 'xport' in a.text()]}. "
        f"Pressed as a key with seed {_seed_a} the window wrote "
        f"{_by_fire.get('key', (Path('NOTHING'), {}))[0].name}; triggered as a "
        f"menu item with seed {_seed_b} it wrote "
        f"{_by_fire.get('menu', (Path('NOTHING'), {}))[0].name}. Two gestures, "
        f"two files, each carrying its own seed "
        f"({_by_fire.get('key', (None, {}))[1].get('provenance', {}).get('seed')}"
        f" and "
        f"{_by_fire.get('menu', (None, {}))[1].get('provenance', {}).get('seed')}"
        f"), which is how this says they reach one method: neither needed a "
        f"Python-side call, so a dead shortcut or an unconnected action would "
        f"have written nothing. **The status line is required to have changed**, "
        f"not merely to carry the right seed: key {_fresh.get('key')}, menu "
        f"{_fresh.get('menu')}. A line that did not change would have carried the "
        f"answer from the export earlier in this section, and a check reading a "
        f"previous export's file and reporting 'the wrong seed' is a check "
        f"reading the wrong thing"
        + (f"; the line did *not* move for one of the gestures, and calling the "
           f"handler directly {'did' if _direct_line else 'did NOT'} move it, so "
           f"this was a {'product' if _direct_line else 'harness'} defect"
           if not all(_fresh.values()) else "")
        + f". The key's note carries where the file goes and what the empty case "
        f"does: {_export_rows[0].note[:78]}... The `E` key is the primary input "
        f"here, so it is in the `always` group -- with no pose loaded it "
        f"refuses, which is an answer too",
    )

    # -- the export's own time ----------------------------------------------
    _times = []
    for _i in range(3):
        _t0 = _time.perf_counter()
        kw._export_run(path=_EXPORT_ROOT / f"timing{_i}.json")
        _times.append((_time.perf_counter() - _t0) * 1000.0)
    _median = float(np.median(_times))
    check(
        "a user clicks it and waits: the export of nine poses, with every "
        "engine decomposition, is a pause and not a freeze",
        _median < 2000.0,
        f"three exports of this run took {[round(t) for t in _times]} ms "
        f"(median {_median:.0f} ms); the first one, before the term maps were "
        f"warm, took {_run_ms:.0f} ms. The cost is nine "
        f"`score_conformation_terms` calls and about 80 kB of JSON, and it "
        f"runs on the GUI thread -- which is the number to watch, not the "
        f"total",
    )

    # -- producing the poses, and taking them away --------------------------
    # Section 16: the window produces poses, and what it produces is the same
    # thing a terminal produces. It already could -- this section is the
    # *evidence*, and the key is the part that did not exist.
    #
    # The engine binary matters here and is stated in the section's own output:
    # the `.pyd` in the tree has to match `core.py`'s signature or nothing below
    # runs at all. See the run note at the top of this file.
    section("16. docking from inside the window")
    import hashlib as _h16
    import tempfile as _tf16

    _EX = Path(__file__).resolve().parent.parent / "examples"
    _DOCK_ROWS = [s for s in keymap.SHORTCUTS if s.handler == "_key_dock"]
    _dock_row = _DOCK_ROWS[0] if _DOCK_ROWS else None
    _K_avail, _K_why = (kw._keys_dock_enabled(_dock_row) if _dock_row
                        else (False, "no row"))
    _K_map = keymap.map_html(kw, kw._keys_pose_summary())
    _K_note_in_map = bool(_dock_row and _dock_row.note
                          and _dock_row.note[:60] in _K_map)
    check(
        "the dock key is declared once, resolves to a callable, and its row "
        "says what it costs and what a second press does",
        (len(_DOCK_ROWS) == 1 and _dock_row.keys == ("K",)
         and _dock_row.group == keymap.GROUP_ALWAYS
         and callable(getattr(kw, "_key_dock", None)) and _K_avail
         and _K_note_in_map
         # `D` is the camera's orbit right, so the dock key cannot be `D`; and
         # nothing else may claim `K`, or the map would be a lie about it.
         and sum(1 for s in keymap.SHORTCUTS if "K" in s.keys) == 1),
        f"{len(_DOCK_ROWS)} row(s) for `_key_dock`: {[(s.label, s.action) for s in _DOCK_ROWS]}, "
        f"in group {_dock_row.group!r}, `enabled_by` "
        f"{_dock_row.enabled_by!r}, and the window answers that method: "
        f"{callable(getattr(kw, '_key_dock', None))}. `D` is already the camera's "
        f"orbit right, so the dock key is `K`; it is the only row claiming `K`. "
        f"With a receptor and a ligand loaded the probe says available "
        f"({_K_avail}, reason {_K_why!r}), and the map carries the note "
        f"({_K_note_in_map}): {_dock_row.note[:120]}... The note is on the row "
        f"because a key list cannot say how long a search takes",
    )

    # The three refusals, asked without a receptor, without a ligand, and while
    # a search is in flight. Each is a different sentence, because they are
    # three different situations.
    _fresh = MainWindow()
    _fresh.resize(900, 600)
    _fresh.show()
    for _ in range(12):
        app.processEvents()
    _a1, _r1 = _fresh._keys_dock_enabled(_dock_row)
    _fresh.load_structure(EXAMPLES / "1crn_prep.pdbqt", "receptor")
    for _ in range(12):
        app.processEvents()
    _a2, _r2 = _fresh._keys_dock_enabled(_dock_row)
    _fresh.close()
    check(
        "with nothing loaded the dock key is unavailable and says which of the "
        "two things is missing",
        (not _a1 and "receptor" in _r1 and "ligand" in _r1
         and not _a2 and "ligand" in _r2 and "receptor" not in _r2),
        f"an empty window: available={_a1}, reason {_r1!r}. With only a receptor "
        f"loaded: available={_a2}, reason {_r2!r}. One sentence for both would "
        f"have been shorter and useless: 'load a receptor first' on a window "
        f"that already has one is a statement about a window the reader is not "
        f"looking at",
    )

    # -- pressed, not installed --------------------------------------------
    # Two fresh windows, one per gesture, each with a seed of its own, and each
    # measured by the poses it produced. Nothing here calls a handler directly:
    # the `E` key and the File menu item were both wired and neither worked, so
    # "the shortcut is in the table" is not evidence that a person pressing it
    # gets poses.
    def _dock_window(seed, gesture):
        w = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                       ligand=EXAMPLES / "biotin_prep.pdbqt")
        w.resize(1400, 850)
        w.show()
        for _ in range(24):
            app.processEvents()
        for s in w.size_spins:
            s.setValue(30.0)
        for s in w.center_spins:
            s.setValue(0.0)
        w.sp_exhaust.setValue(8)
        w.sp_seed.setValue(seed)
        for _ in range(6):
            app.processEvents()
        menu = w.menuBar().actions()[0].menu()
        item = next((a for a in menu.actions()
                     if a.text() == "Dock now"), None)
        if gesture == "key":
            QTest.keyClick(w, QtCore.Qt.Key.Key_K)
        else:
            item.trigger() if item is not None else None
        ticks, seen = 0, []
        while w._thread is not None and w._thread.isRunning():
            app.processEvents()
            ticks += 1
            now = w.status_label.text()
            if not seen or seen[-1] != now:
                seen.append(now)
            QTest.qWait(10)
        for _ in range(80):
            app.processEvents()
            QTest.qWait(10)
        out = (w._dock_result, w.pose_table.rowCount(), seen, ticks,
               w.act_dock.isEnabled(), w.btn_dock.isEnabled())
        w.close()
        return out

    _res_key, _rows_key, _seen_key, _ticks_key, _menu_on, _btn_on = \
        _dock_window(42420001, "key")
    _res_menu, _rows_menu, _seen_menu, _ticks_menu, _, _ = \
        _dock_window(42420002, "menu")
    _key_worked = (_res_key is not None and _rows_key == _res_key.num_poses
                   and _rows_key > 0)
    _menu_worked = (_res_menu is not None and _rows_menu == _res_menu.num_poses
                    and _rows_menu > 0)
    check(
        "pressing K runs a search and poses arrive, and clicking the menu item "
        "does the same -- measured by the poses, not by the wiring",
        (_key_worked and _menu_worked and _ticks_key > 0
         and _menu_on and _btn_on
         # The two runs used different seeds, so if the key had quietly run the
         # old search or reused a result the best energies could not both be
         # this run's own.
         and abs(float(_res_key.best_energy)
                 - float(_res_menu.best_energy)) > 1e-9),
        f"K pressed: {_rows_key} pose row(s), best "
        f"{_res_key.best_energy:.4f} kcal/mol, {_ticks_key} event-loop turns "
        f"while it ran, stages seen {[s[:34] for s in _seen_key]}. Menu item "
        f"clicked: {_rows_menu} pose row(s), best "
        f"{_res_menu.best_energy:.4f} kcal/mol, {_ticks_menu} event-loop turns, "
        f"stages seen {[s[:34] for s in _seen_menu]}. The two best energies "
        f"differ, so each gesture produced its own search rather than replaying "
        f"the other's. The window stayed live throughout -- that is the property "
        f"worth having, and a turn count is the only honest way to show it",
    )

    # -- a second request, asked three ways ---------------------------------
    w3 = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                    ligand=EXAMPLES / "biotin_prep.pdbqt")
    w3.resize(1400, 850)
    w3.show()
    for _ in range(24):
        app.processEvents()
    for s in w3.size_spins:
        s.setValue(44.0)
    w3.sp_exhaust.setValue(8)
    w3.sp_seed.setValue(42420003)
    for _ in range(6):
        app.processEvents()
    QTest.keyClick(w3, QtCore.Qt.Key.Key_K)
    _th3 = w3._thread
    _refused_label = ""
    _same3 = None
    _guard_hit = False
    for _ in range(400):
        app.processEvents()
        if _th3 is not None and not _th3.isRunning():
            break
        if w3._dock_running() and _same3 is None:
            _before = w3._thread
            w3.start_docking()
            _same3 = w3._thread is _before
            _refused_label = w3.status_label.text()
            w3._key_dock()
            _guard_hit = w3._thread is _before
            break
        QTest.qWait(5)
    _notice = "refused" in _refused_label
    while w3._thread is not None and w3._thread.isRunning():
        app.processEvents()
        QTest.qWait(10)
    for _ in range(60):
        app.processEvents()
        QTest.qWait(10)
    _cleared = "refused" not in w3.status_label.text()
    _rows3 = w3.pose_table.rowCount()
    w3.close()
    check(
        "a second Dock, asked while one is running, is refused with a sentence "
        "that stays on screen -- and nothing is queued",
        (_same3 is True and _guard_hit and _notice and _cleared
         and _rows3 > 0 and crashguard.failures() == ()),
        f"asked three ways mid-run -- `start_docking()` called directly, and the "
        f"`K` key -- and the thread object was unchanged both times "
        f"({_same3}, {_guard_hit}), so neither started a search. The refusal is "
        f"the rule, not the greyed-out button: the direct call is the one no "
        f"disabled widget can stop. The sentence appears beside the stage rather "
        f"than replacing it -- {_refused_label!r} -- because the stage keeps "
        f"arriving and would have buried it within a frame, and it is gone once "
        f"the run ends ({_cleared}) rather than lingering as a stale warning. "
        f"The run itself finished normally with {_rows3} pose row(s) and no "
        f"crash: crashguard reports {crashguard.failures()!r}",
    )

    # -- the window's poses against a terminal's poses ----------------------
    # Same receptor, same ligand, same box, same exhaustiveness, same
    # num_modes, same rmsd_cutoff, same seed, and both written with the *same*
    # method -- `DockingResult.write_pdbqt`, which is what `_cmd_dock` calls at
    # `cli.py:774`. So this is a byte comparison, not a tolerance: either the
    # GUI and the CLI produce the same file or they do not, and "close enough"
    # is not a claim either of them makes.
    _BOX = (0.0, 0.0, 0.0, 30.0, 30.0, 30.0)
    _gui_file = Path(_tf16.gettempdir()) / "gate16_gui.pdbqt"
    _cli_file = Path(_tf16.gettempdir()) / "gate16_cli.pdbqt"
    _gui_file.unlink(missing_ok=True)
    _cli_file.unlink(missing_ok=True)
    _res_key.write_pdbqt(_gui_file)
    from opendocking import core as _core16
    _rec16 = _core16.Receptor.from_pdbqt(EXAMPLES / "1crn_prep.pdbqt")
    _box16 = _core16.GridBox.from_center_size(_BOX[:3], _BOX[3:])
    _maps16 = _rec16.precalculate(_box16, "vina")
    # The nine arguments `_cmd_dock` passes, in the same order, with the same
    # values the window's panel held. This is the CLI's call, not a second
    # wrapper: `cli.py` imports `dock` from `.core`, and so does this.
    _cli_res = _core16.dock(
        _core16.load_ligand(EXAMPLES / "biotin_prep.pdbqt"), _maps16,
        exhaustiveness=8, num_modes=9, rmsd_cutoff=1.0, seed=42420001,
        mode="mc", scoring="vina", steps=None)
    _cli_res.write_pdbqt(_cli_file)
    _gb, _cb = _gui_file.read_bytes(), _cli_file.read_bytes()
    _same_bytes = _gb == _cb
    _same_nums = ([round(float(a), 9) for a in _res_key.energies]
                  == [round(float(a), 9) for a in _cli_res.energies]
                  and [round(float(r), 9) for r in _res_key.rmsds]
                  == [round(float(r), 9) for r in _cli_res.rmsds])
    check(
        "the poses the window draws are byte-for-byte the poses a terminal "
        "produces for the same seed -- one engine call, two front ends",
        (_same_bytes and _same_nums
         and _res_key.num_poses == _cli_res.num_poses
         and _gui_file.exists() and _cli_file.exists()),
        f"GUI {_gui_file.stat().st_size} B  sha256 "
        f"{_h16.sha256(_gb).hexdigest()[:32]}; CLI {_cli_file.stat().st_size} B  "
        f"sha256 {_h16.sha256(_cb).hexdigest()[:32]}; identical: {_same_bytes}. "
        f"Same {len(_res_key.energies)} affinities and rmsds to nine decimals: "
        f"{_same_nums}, and best {_res_key.best_energy:.6f} against "
        f"{_cli_res.best_energy:.6f}. Both sides write through "
        f"`DockingResult.write_pdbqt`, and both call `opendocking.core.dock` -- "
        f"`workbench/app.py`'s `DockingWorker.run` and `cli.py`'s `_cmd_dock` "
        f"import the same function -- so a difference here could only come from "
        f"the arguments, and the arguments are printed on both sides of this "
        f"line. What this does **not** say is that either is right: it says the "
        f"window is not a second opinion",
    )

    # -- and the one thing this section cannot say --------------------------
    # -- the failure path, in this product's voice --------------------------
    # A search that does not run used to say `docking failed: <engine text>` and
    # stop. That is the engine's voice, not this product's, and it gave a user no
    # next action. The three cases below are the ones a GUI user can actually
    # reach: a build mismatch, a box the ligand does not fit in, and something
    # unclassifiable -- the last one matters most, because an unclassifiable
    # failure that admits it has no fix is the honest answer, and a set of cases
    # that only tested the easy ones would let a guess through.
    _ADV_CASES = [
        ("dock() takes from 2 to 9 positional arguments but 10 were given",
         ("build mismatch", "Rebuild")),
        ("axis y is 12.0 A but the ligand needs at least 12.1 A",
         ("smaller than the ligand", "Widen")),
        ("some failure nobody has seen before and nobody can classify",
         ("nothing here can name a fix", "box, the receptor")),
    ]
    _adv = [(m, kw._dock_failure_advice(m)) for m, _ in _ADV_CASES]
    _adv_ok = all(
        all(frag in advice for frag in frags)
        and 0 < len(advice) < 240
        and "docking failed" not in advice
        for (_, advice), (_, frags) in zip(_adv, _ADV_CASES)
    )
    # And the sentence the label actually gets, built the way the handler builds
    # it, so this is the text a user reads rather than a description of it.
    _shown = (f"the search did not run. {_adv[0][1]} "
              f"[engine: {_adv[0][0]}]")
    check(
        "a failed search says what failed and what would fix it, and says so "
        "in one line",
        (_adv_ok and _shown.startswith("the search did not run.")
         and "build mismatch" in _shown and "[engine:" in _shown
         and len(_shown) < 420),
        f"the three reachable cases, each answered in one line: "
        f"{[a for _, a in _adv]}. The first is the case this gate has actually "
        f"hit -- the engine extension and the Python beside it built at "
        f"different times -- and it now names itself as a build mismatch and "
        f"names the fix, where before it read 'docking failed: <python "
        f"TypeError>'. The last admits there is no fix to name rather than "
        f"inventing one, which is the same rule the export's `state` follows and "
        f"for the same reason. What a user reads is {len(_shown)} characters: "
        f"{_shown[:160]}... The engine's own words stay in the square brackets, "
        f"so nothing is hidden and the advice is additive rather than a "
        f"replacement",
    )

    # -- the failure branches, end to end ----------------------------------
    # The check above tests the *classifier* on strings. This one makes the
    # engine actually refuse and reads the sentence off the label, because a
    # classifier tested only against strings somebody typed has never met what
    # it will meet. The refusal is a box smaller than the ligand needs, which is
    # the one a user reaches by typing a number, and the engine's own words are
    # kept in the detail so the branch can be seen to be matching them.
    _refuse = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                         ligand=EXAMPLES / "biotin_prep.pdbqt")
    _refuse.resize(1100, 700)
    _refuse.show()
    for _ in range(16):
        app.processEvents()
    for _s in _refuse.size_spins:
        _s.setValue(10.0)
    for _c in _refuse.center_spins:
        _c.setValue(0.0)
    _refuse.sp_exhaust.setValue(2)
    _refuse.sp_seed.setValue(20260901)
    for _ in range(6):
        app.processEvents()
    _refuse.btn_dock.click()
    for _ in range(900):
        app.processEvents()
        if _refuse._thread is None:
            break
        QTest.qWait(10)
    for _ in range(40):
        app.processEvents()
        QTest.qWait(10)
    _refused_label = _refuse.status_label.text()
    _refused_engine = (_refused_label.split("[engine:", 1)[-1].rstrip("] ")
                       if "[engine:" in _refused_label else "")
    _refuse.close()
    for _ in range(8):
        app.processEvents()
    check(
        "a box the engine refuses produces a sentence that says what to do, "
        "read off the label rather than assembled by this check",
        (_refused_label.startswith("the search did not run.")
         and "smaller than the ligand" in _refused_label
         and "Widen it" in _refused_label
         and "but the ligand needs at least" in _refused_engine
         and "docking failed" not in _refused_label),
        f"a 10 A box on a 19-atom ligand, driven through the button and the "
        f"worker thread, and what the label reads: {_refused_label!r}. The "
        f"engine's own words, which the branch is matching on: "
        f"{_refused_engine!r}. Three things this establishes that a classifier "
        f"tested on a typed string would not: the refusal really happens on "
        f"this path, the sentence is the one a user sees rather than one this "
        f"check assembled, and the bracket keeps the engine's message so "
        f"nothing is hidden. Driving bad files at the classifier also found two "
        f"branches it was missing -- a file with no ATOM records, and a "
        f"coordinate the engine could not read -- and both had been answered "
        f"with the one sentence that is never right",
    )

    # -- F frames whatever is selected, and only that ------------------------
    # The rule: a selected pair, then a selected residue, then the pose with its
    # contacts, and with nothing selected, every pose. What is measured is the
    # camera's **distance from the thing it should be framing**, because "did
    # the camera move" is not the question -- selecting a pair already moved it
    # there, and the bug was that `F` moved it away again.
    _fw = MainWindow(receptor=EXAMPLES / "1crn_prep.pdbqt",
                     ligand=EXAMPLES / "biotin_prep.pdbqt")
    _fw.resize(1400, 850)
    _fw.show()
    for _ in range(24):
        app.processEvents()
    for _s in _fw.size_spins:
        _s.setValue(30.0)
    for _c in _fw.center_spins:
        _c.setValue(0.0)
    _fw.sp_exhaust.setValue(8)
    _fw.sp_seed.setValue(20260901)
    for _ in range(6):
        app.processEvents()
    _fw.btn_dock.click()
    while _fw._thread is not None and _fw._thread.isRunning():
        app.processEvents()
        QTest.qWait(10)
    for _ in range(60):
        app.processEvents()
        QTest.qWait(10)

    def _cam_off_midpoint(win):
        hp = win.viewport.highlight_pair
        if hp is None:
            return float("inf")
        pose_m = next(m for m in win.viewport.molecules if m.role == "pose")
        rec_m = next(m for m in win.viewport.molecules if m.role == "receptor")
        mid = (np.asarray(pose_m.coords[hp[0]], np.float64)
               + np.asarray(rec_m.coords[hp[1]], np.float64)) / 2.0
        return float(np.linalg.norm(
            np.asarray(win.viewport.camera.center, np.float64) - mid))

    _fw.pair_table.setCurrentCell(0, 0)
    for _ in range(8):
        app.processEvents()
    _settle_camera(_fw, app)
    _pair_before = _cam_off_midpoint(_fw)
    QTest.keyClick(_fw, QtCore.Qt.Key.Key_F)
    _settle_camera(_fw, app)
    for _ in range(8):
        app.processEvents()
    _pair_after = _cam_off_midpoint(_fw)

    _fw.pair_table.clearSelection()
    for _ in range(4):
        app.processEvents()
    _res_name = None
    _res_shift = _res_dist_shift = None
    if _fw.contact_table.rowCount():
        _fw.contact_table.setCurrentCell(0, 0)
        for _ in range(6):
            app.processEvents()
        _settle_camera(_fw, app)
        _res_name = _fw._selected_residue_name()
        _c_before = np.asarray(_fw.viewport.camera.center, np.float64).copy()
        _d_before = float(_fw.viewport.camera.distance)
        QTest.keyClick(_fw, QtCore.Qt.Key.Key_F)
        _settle_camera(_fw, app)
        for _ in range(8):
            app.processEvents()
        _res_shift = float(np.linalg.norm(
            np.asarray(_fw.viewport.camera.center, np.float64) - _c_before))
        _res_dist_shift = abs(float(_fw.viewport.camera.distance) - _d_before)
    check(
        "F frames whatever is selected: a selected pair stays framed, and a "
        "selected residue stays framed",
        (_pair_before < 1.0 and _pair_after < 1.0
         and _res_shift is not None and _res_shift < 0.5
         and _res_dist_shift < 0.5),
        f"a selected pair: the camera was {_pair_before:.2f} A from the "
        f"midpoint of the two atoms the row names, and after `F` it is "
        f"{_pair_after:.2f} A -- so the key left it where the selection had put "
        f"it, where before the rule it took the camera 8.80 A off and pulled "
        f"back from 6.00 A to 22.78 A, turning a 2.05 A line from 32 px of "
        f"highlight into 5. A selected residue ({_res_name!r}): the camera "
        f"moved {_res_shift:.2f} A and its distance changed by "
        f"{_res_dist_shift:.2f} A, so `F` did not undo that selection either. "
        f"The check is on the distance from the thing being framed rather than "
        f"on whether the camera moved, because selecting already moved it and "
        f"the bug was a second answer to the same question",
    )

    # And with nothing selected, `F` frames every pose -- the view about nine
    # answers rather than about one, and the one thing the rule must not break.
    _fw.pair_table.clearSelection()
    _fw.contact_table.clearSelection()
    for _ in range(4):
        app.processEvents()
    _c_all_before = np.asarray(_fw.viewport.camera.center, np.float64).copy()
    _d_all_before = float(_fw.viewport.camera.distance)

    def _park_and_read(key=None, method=None):
        """Press `F` (or call `method`) from a camera parked away, and read it.

        Parking first is what makes the comparison a measurement: without it
        the second call would be reading a camera the first one left where it
        wanted it, and "they agree" would be a fact about the order of two
        statements rather than about the rule.
        """
        _fw.viewport.camera.center = np.asarray([0.0, 0.0, 0.0], np.float32)
        _fw.viewport.camera.distance = 3.0
        for _ in range(2):
            app.processEvents()
        if key is not None:
            QTest.keyClick(_fw, key)
        else:
            method()
        _settle_camera(_fw, app)
        for _ in range(6):
            app.processEvents()
        return (np.asarray(_fw.viewport.camera.center, np.float64).copy(),
                float(_fw.viewport.camera.distance))

    # Branch 1: both tables cleared, a pose row still current. The rule is
    # "pair, then residue, then the selected pose with its contacts", so this
    # branch is `focus_selection` and *not* the all-poses framing. The check
    # this replaces called this branch "with nothing selected, F frames every
    # pose" and measured it by the distance happening to change by more than
    # 1.0 A -- which is a difference between two unrelated numbers. It read
    # 20.29 -> 22.78 A, a change of 2.49 A, on a correct product, and on a
    # build where the pair framing moved the "before" it read 0.81 A and went
    # red. Both times the evidence was a coincidence.
    _c_pose_after, _d_pose_after = _park_and_read(QtCore.Qt.Key.Key_F)
    _c_pose_ref, _d_pose_ref = _park_and_read(method=_fw._frame_selection)
    _pose_agrees = (float(np.abs(_c_pose_after - _c_pose_ref).max()) < 1e-4
                    and abs(_d_pose_after - _d_pose_ref) < 1e-4)
    # Branch 2: nothing selected *at all*, no pose row current either. This is
    # the branch the key's text is about, and it is `_frame_poses`. The widget
    # is `pose_table`, not `pose_list`: `_keys_needs_pose` reads
    # `pose_table.currentRow()`, and the first version of this cleared
    # `pose_list` instead, which left the row at 0 and made the branch
    # unreachable -- a check that cannot reach the case it names.
    _fw.pose_table.setCurrentCell(-1, -1)
    for _ in range(4):
        app.processEvents()
    _no_pose_row = _fw.pose_table.currentRow() < 0
    _c_none_after, _d_none_after = _park_and_read(QtCore.Qt.Key.Key_F)
    _c_none_ref, _d_none_ref = _park_and_read(method=_fw._frame_poses)
    _none_agrees = (float(np.abs(_c_none_after - _c_none_ref).max()) < 1e-4
                    and abs(_d_none_after - _d_none_ref) < 1e-4)
    _fw.close()
    for _ in range(8):
        app.processEvents()
    _F_rows = [s for s in keymap.SHORTCUTS
               if s.handler == "_key_frame_selection"]
    check(
        "with nothing selected, F frames the pose, and with nothing selected at "
        "all it frames every pose -- the two branches the key's own description "
        "names",
        (float(np.linalg.norm(_c_pose_after - _c_all_before)) > 1.0
         and _pose_agrees and _no_pose_row and _none_agrees
         and len(_F_rows) == 1
         and "whatever is selected" in _F_rows[0].action
         and "all poses if nothing is" in _F_rows[0].action),
        f"the rule is pair, then residue, then the selected pose with its "
        f"contacts, then every pose -- and the last two are different questions, "
        f"this check used to conflate. **Both tables cleared, a pose row still "
        f"current**: `F` lands at {np.round(_c_pose_after, 3).tolist()} / "
        f"{_d_pose_after:.2f} A, and `focus_selection` called directly from a "
        f"camera parked at the origin and 3.0 A out lands at "
        f"{np.round(_c_pose_ref, 3).tolist()} / {_d_pose_ref:.2f} A, agreeing to "
        f"{float(np.abs(_c_pose_after - _c_pose_ref).max()):.1e} A and "
        f"{abs(_d_pose_after - _d_pose_ref):.1e} A. **Pose row cleared as well** "
        f"(`pose_table.currentRow()` = {_fw.pose_table.currentRow()} before the "
        f"press): `F` lands at {np.round(_c_none_after, 3).tolist()} / "
        f"{_d_none_after:.2f} A and `_frame_poses` at "
        f"{np.round(_c_none_ref, 3).tolist()} / {_d_none_ref:.2f} A, agreeing to "
        f"{float(np.abs(_c_none_after - _c_none_ref).max()):.1e} A and "
        f"{abs(_d_none_after - _d_none_ref):.1e} A. The camera also moved "
        f"{float(np.linalg.norm(_c_pose_after - _c_all_before)):.2f} A off the "
        f"residue it had been framing, which is the 'which of the two answers "
        f"wins' half. **Not** 'the distance changed by more than 1.0 A': that was "
        f"the old evidence, it is a difference between two unrelated numbers, and "
        f"it read {_d_all_before:.2f} -> {_d_pose_after:.2f} A here and "
        f"21.97 -> 22.78 A on a build where the pair framing had moved the "
        f"'before' -- a correct product, red on a coincidence, twice. The key "
        f"declares {len(_F_rows)} row(s) for it and its description is "
        f"{_F_rows[0].action!r}, which is where the rule is written down rather "
        f"than only in the code",
    )

    # -- the drawn geometry against the panel's numbers ---------------------
    # `DockingWorker` hands `_on_dock_finished` a `DockingResult`, and the
    # window then writes *that* to a temporary PDBQT and loads its poses from
    # the file -- so what is drawn comes from the writer, while the affinity,
    # the terms, the gradient and the whole verdict panel come from `result`.
    # The question is whether those are the same geometry.
    #
    # They are, and the way to see it is to read the file the way its reader
    # reads it. The writer emits the rigid `ROOT` cluster first and the
    # flexible `BRANCH` clusters after it, so the `ATOM` records of a `MODEL`
    # are in *branch-tree* order and not atom-index order -- measured on this
    # file the serials run 8,9,10,...,16,18,19,7,...,4,1. Zipping record n
    # against `pose_coords[n]` therefore compares two different atoms and
    # reports several angstroms, which is a fact about the key and not about
    # the engine. That is what this check used to do, and it read the result
    # as a defect in the product and skipped on it.
    #
    # The file says which key it is on its own face --
    # `REMARK OD_ATOM_ORDER: BRANCH TREE; SERIAL = ATOM INDEX; REINDEX BY
    # SERIAL` -- `dock_core::pdbqt::reindex_by_serial` reindexes by that
    # column unconditionally, and `workbench/structure.py::parse_structure`
    # sorts each model by serial for the same reason, so that is what the
    # window actually does with this file. Both orders are measured below: the
    # serial one is the one that has to agree, and the file-order one is
    # reported alongside it because it is the number a reader who skipped the
    # remark would compute, and pretending otherwise would be the same mistake
    # in the other direction.
    _engine_xyz = np.asarray(_cli_res.pose_coords(0), np.float64)
    _fk = np.asarray(_core16.conformation_coordinates(
        _core16.load_ligand(EXAMPLES / "biotin_prep.pdbqt"),
        _cli_res.pose_conformation(0)), np.float64)
    _self_consistent = float(np.abs(_engine_xyz - _fk).max())
    _models16, _cur16 = [], []
    _declares_order = False
    for _line in _cli_file.read_text(encoding="utf-8").splitlines():
        if _line.startswith("MODEL"):
            _cur16 = []
        elif _line.startswith("REMARK OD_ATOM_ORDER"):
            _declares_order = True
        elif _line.startswith(("ATOM", "HETATM")):
            # The serial column (7-11) is kept, because it is the key the
            # declaration says to read by. The three coordinate columns are kept
            # **as text** as well, because how many decimals they carry is a
            # property of the file that the M5 block below has to be able to
            # read rather than assume.
            _cur16.append((int(_line[6:11]), float(_line[30:38]),
                           float(_line[38:46]), float(_line[46:54]),
                           _line[30:38], _line[38:46], _line[46:54]))
        elif _line.startswith("ENDMDL"):
            _models16.append(list(_cur16))
    # ---------------------------------------------------------------- M5
    # How many decimals this file's coordinates are actually written to, read
    # out of the file rather than out of a comment.
    #
    # The first version of this bound was the literal `0.5 * 10.0 ** -3`, with
    # the derivation ("`PdbqtWriteOptions::default`'s `coord_precision: 3`")
    # written next to it in prose. That is a bound whose *author* chose it, and
    # a chosen bound can be chosen again: widening `5e-4` to `5.0` A is a 1e4
    # loosening that leaves the check green, because no real disagreement
    # between this writer and this array reaches 5.0 A. The mutation was
    # inequivalent and the verdict did not move. A comment saying why a number
    # is right is not the same as the check being able to tell that it is not.
    #
    # So the precision is counted off the file's own coordinate columns. The
    # columns are fixed-width and every coordinate is written through a `%.Nf`,
    # so a value that happens to be a whole number still prints its N decimals:
    # the count is a property of the file, not of the data in it. `max` is used
    # deliberately, so a writer that emits *more* decimals on some records
    # tightens the bound to the finest precision present rather than being
    # measured against the coarsest.
    _dec16: set[int] = set()
    for _mod in _models16:
        for _rec in _mod:
            for _field in (_rec[4], _rec[5], _rec[6]):
                _t = str(_field).strip()
                if "." in _t:
                    _dec16.add(len(_t.split(".", 1)[1]))
    _file_decimals = max(_dec16) if _dec16 else -1
    _one_precision = len(_dec16) <= 1
    # The bound, now the format's own rather than the author's: half a unit in
    # the last decimal the file carries.
    _half_ulp = 0.5 * 10.0 ** -_file_decimals if _file_decimals >= 0 else float("inf")
    _n16 = min(len(_models16), _cli_res.num_poses)
    _file_rmsd, _serial_max = [], 0.0
    _rounded_max, _rounded_at = 0.0, None
    _serials16, _is_perm = [], True
    for _i in range(_n16):
        _pc = np.asarray(_cli_res.pose_coords(_i), np.float64)
        _recs = _models16[_i]
        _m = min(len(_pc), len(_recs))
        # File order: the number the old check measured, kept and labelled.
        _fa = np.asarray([[r[1], r[2], r[3]] for r in _recs[:_m]], np.float64)
        _file_rmsd.append(float(np.sqrt(
            ((_fa - _pc[:_m]) ** 2).sum(axis=1).mean())))
        # Serial order: what `reindex_by_serial` and `parse_structure` do. The
        # index is bounded rather than trusted: a writer that emitted a serial
        # outside `1..n` has to come back as a red check with a readable cause,
        # not as an `IndexError` out of this loop that takes the whole run down
        # with it and reports nothing about the file that caused it.
        _ser = [r[0] for r in _recs]
        _serials16.append(tuple(_ser))
        _sa = np.empty_like(_fa)
        _fits = True
        for _r in _recs[:_m]:
            _k = _r[0] - 1
            if 0 <= _k < _m:
                _sa[_k] = (_r[1], _r[2], _r[3])
            else:
                _fits = False
        _is_perm = (_is_perm and _fits
                    and sorted(_ser) == list(range(1, len(_ser) + 1)))
        if _fits:
            _serial_max = max(_serial_max,
                              float(np.abs(_sa[:_m] - _pc[:_m]).max()))
            # **The claim that has no slack in it.** If the writer rounded, then
            # rounding the array to the file's own precision gives the file's own
            # numbers back *exactly*, and this is an equality rather than a
            # tolerance: there is no constant here to widen. A geometry that is
            # a different geometry fails it by metres; a writer that quietly lost
            # a decimal fails it by more than the digit it lost; a reader who
            # reindexed by the wrong key fails it immediately. The disagreement
            # above stays reported, because "it differs, and only by the printed
            # precision" is a stronger and more useful sentence than either half
            # of it alone, but it is no longer what is being asserted.
            if _file_decimals >= 0:
                _gap = np.abs(np.round(_pc[:_m], _file_decimals) - _sa[:_m])
                _g = float(_gap.max())
                if _g > _rounded_max:
                    _rounded_max = _g
                    _ix = np.unravel_index(int(np.argmax(_gap)), _gap.shape)
                    _rounded_at = (int(_i), int(_ix[0]), str(_ix[1]),
                                   float(_pc[_ix[0], _ix[1]]),
                                   float(_sa[_ix[0], _ix[1]]))
    # Whether the two orders actually differ here. Not required to be true: a
    # ligand with no rotatable bond has an identity permutation and reindexing
    # is then a no-op, which is still the correct read. Reported because "the
    # file order is not the serial order" is the premise the whole comparison
    # rests on, and a future fixture that made it false would quietly stop
    # testing anything.
    _orders_differ = any(tuple(sorted(s)) != s for s in _serials16)
    _order_note = ("a permutation of 1..n" if _is_perm
                   else "NOT a permutation, so reindexing by them is "
                        "meaningless")
    check(
        "the geometry the window draws and the geometry the engine's own "
        "numbers describe are the same geometry, read by the key the file "
        "declares",
        (_n16 > 0 and _declares_order and _is_perm
         and _serial_max <= _half_ulp and _rounded_max == 0.0
         and _self_consistent <= 1e-6),
        f"over {_n16} pose(s), reindexed by the SERIAL column the file declares "
        f"in `REMARK OD_ATOM_ORDER`: rounding `pose_coords(i)` to the "
        f"{_file_decimals} decimals this file actually writes reproduces the "
        f"written model **exactly** -- worst difference "
        f"{_rounded_max:.3e} A over every coordinate of every pose"
        + (f", first offender pose {_rounded_at[0]} atom {_rounded_at[1]} "
           f"{_rounded_at[2]}: array {_rounded_at[3]!r}, file {_rounded_at[4]!r}"
           if _rounded_at and _rounded_max else "")
        + f". That is an equality and not a tolerance, so there is no constant "
        f"in it to widen: a geometry that is a different geometry fails it by "
        f"metres, a writer that lost a decimal fails it by that digit, and the "
        f"wrong key fails it at once. The unrounded disagreement is reported "
        f"alongside it, {_serial_max:.3e} A at worst against the "
        f"{_half_ulp:.0e} A that a coordinate written to {_file_decimals} "
        f"decimals can differ by, so what differs is the printed precision and "
        f"nothing else. The engine also agrees with itself: "
        f"`conformation_coordinates(pose_conformation(i))` matches "
        f"`pose_coords(i)` to {_self_consistent:.3e} A. Both paths read one "
        f"array: the writer's `coords: &p.coords` and "
        f"`DockingResults::pose_coords` are the same `Vec<Vec3>`. The "
        f"declaration record is "
        f"{'present' if _declares_order else 'ABSENT'}, the serials are "
        f"{_order_note}, and the file order "
        f"{'does' if _orders_differ else 'does not'} differ from the serial "
        f"order (serials {list(_serials16[0][:8]) if _serials16 else []}...). "
        f"Zipping the `ATOM` records in file order instead -- what this check "
        f"used to do, and what a reader who skipped the remark would do -- gives "
        f"{np.mean(_file_rmsd):.3f} A mean RMSD, and that number is a fact "
        f"about the key, not a disagreement between the writer and the array",
    )
    # The bound the check above leans on, held to the value its own derivation
    # gives. This is a meta-assertion about the check and it is the weakest of
    # the three things that could be done here, so it is on its own and says
    # so; it is here because without it a 1e4 widening of the derived expression
    # is a mutation that leaves the verdict standing, and a gate whose bounds
    # can be widened into uselessness is the failure this whole file keeps
    # meeting. The load-bearing claim is the equality above.
    check(
        "the bound the round-trip check used is the one the file's own "
        "precision implies, so widening it would turn *this* red",
        (_one_precision and _file_decimals >= 0
         and _half_ulp == 0.5 * 10.0 ** -_file_decimals
         and _rounded_max == 0.0),
        f"this file writes {_file_decimals} decimal"
        f"{'' if _file_decimals == 1 else 's'} in every coordinate column"
        f"{'' if _one_precision else f', and NOT the same count everywhere: {sorted(_dec16)}'}, "
        f"so the bound is 0.5 x 10^-{_file_decimals} = {_half_ulp:.3e} A, and "
        f"that is the number the check compared {_serial_max:.3e} A against. "
        f"Measured {4.986e-4 / _half_ulp:.1%} of it if it came to 4.986e-4. The "
        f"format is `PdbqtWriteOptions::default`'s `coord_precision`, which this "
        f"check observes in the file rather than importing, because the Rust "
        f"constant is not reachable from here and a comment about it is not a "
        f"measurement of it",
    )

    # ---------------------------------------------------------------- report
    section("summary")
    before_total = len(results)
    missing_sections = [t for t in EXPECTED_SECTIONS if t not in sections_entered]
    check(
        "every section this file is supposed to run was reached, so this run "
        "is complete and not a truncated one",
        not missing_sections,
        f"{len(EXPECTED_SECTIONS) - len(missing_sections)} of "
        f"{len(EXPECTED_SECTIONS)} sections reached"
        + (f"; never reached: {missing_sections}" if missing_sections else ""),
    )
    check(
        "the number of checks that ran is the number this file is supposed to "
        "have, so none of them is hiding behind a guard",
        before_total + 2 == EXPECTED_CHECKS,
        f"{before_total} ran before these two and {EXPECTED_CHECKS} are expected; "
        "the +2 is these two completeness checks. If a check was added or "
        "removed, change EXPECTED_CHECKS deliberately",
    )
    # Closed here, not before the two checks above, so the summary's own
    # subtotal counts them. Closing it first printed "summary: 0 checks", which
    # is the same kind of untrue line this file exists to stop printing.
    _close_section()
    npass = sum(1 for r in results if r[0] == "PASS")
    nfail = sum(1 for r in results if r[0] == "FAIL")
    nskip = sum(1 for r in results if r[0] == "SKIP")
    print(f"  {npass} passed, {nfail} failed, {nskip} skipped, {len(results)} checks")
    for tag, name, detail in results:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {detail}")
    if nskip:
        print(
            f"\n  {nskip} check(s) were skipped, not passed. A skip means this\n"
            "  environment could not answer the question; it is not evidence\n"
            "  that the workbench is correct there."
        )
        # Name the sections, so "this run was green" and "these sections were
        # answered by the machine rather than by the code" are not the same
        # sentence in the reader's head.
        partly = [t for t, n in section_skips.items() if n]
        if partly:
            print(f"  sections with skipped checks: {partly}")
            if not PIXELS_OK:
                print(
                    f"  reason: {GL_VERDICT}\n"
                    "  CI runs `xvfb-run` with LIBGL_ALWAYS_SOFTWARE=1, where a real\n"
                    "  framebuffer exists; that run, not this one, is the authority\n"
                    "  on the skipped checks."
                )
            elif not GL_OK:
                # A state worth naming, because it looks like a bug in the
                # summary until you know it is measured: the launcher's probe
                # said no context, and the product rendered anyway.
                print(
                    "  the skipped checks above are not OpenGL skips: the\n"
                    "  launcher's own probe could not get Qt a context, but this\n"
                    "  workbench's QOpenGLWidget did render and its framebuffer was\n"
                    "  read, so the pixel checks ran. The probe builds a bare\n"
                    "  QOpenGLContext; the product is a QOpenGLWidget with a real FBO,\n"
                    "  and the second is what a viewer sees. The two disagree, and\n"
                    "  the pixel checks follow the product."
                )
    print(f"\n  screenshots: {OUT}")

    # The last thing printed, and the only exit code that means "this never
    # finished". 0 is clean, 1 is "ran everything, something failed", 2 is "did
    # not finish" and 3 is "died". A caller can tell those apart without
    # parsing the transcript, which is the difference between a truncated run
    # that looks green and one that does not.
    if missing_sections or len(results) != EXPECTED_CHECKS:
        print(
            f"\n  RUN INCOMPLETE: {len(results)} of {EXPECTED_CHECKS} checks ran, "
            f"{len(EXPECTED_SECTIONS) - len(missing_sections)} of "
            f"{len(EXPECTED_SECTIONS)} sections reached."
        )
        if missing_sections:
            print(f"  never reached: {missing_sections}")
        print(
            "  This is not a pass. The numbers above describe the part of the\n"
            "  file that ran; nothing here says anything about the rest."
        )
    # The main window is deliberately *not* closed here. It is the last one
    # standing and it has been alive for the whole run; closing it on this
    # machine fail-fasts with `0xC0000409` after the summary has already been
    # printed, so the transcript is complete and the exit code is a crash. Left
    # to process exit, which reclaims the GL context without asking a widget
    # that has been repainting for ten minutes to tear itself down.
    if missing_sections or len(results) != EXPECTED_CHECKS:
        return 2
    return 1 if nfail else 0


#: The four names section 11e records, in file order. A module-level list
#: because the `else`-shaped branch that skips them needs the same names, and
#: two literals that have to be kept in step by hand is how one of them ends up
#: skipped under a name nothing looks for. `SITE_SELECTION_CHECKS` earns its
#: keep for the same reason and is checked against what the block recorded.
POSE_FRAMING_CHECKS = (
    "selecting a pose ends the camera exactly on the framing it aimed at",
    "and the camera moved to get there rather than snapping",
    "and what it aimed at is the pose plus the atoms the pose is touching",
    "and the site cloud steps aside for the pose it is framing",
)

#: Where the camera was on each frame of the last move, filled by
#: `_run_camera_move`. Module level rather than a return value because the
#: section that reads it is long and threading a list through it would be noise.
plan_distances: list[float] = []
move_ms = 0
plan_row = -1


def _pair_own_pixels(on, off) -> int:
    """Pixels that are the pair highlight's and nothing else's.

    A **difference** of two grabs at the same camera: one with the highlight
    drawn, one with it suppressed. This is the instrument, and it replaces a
    purity count, because the purity count was measuring something else.

    `_pair_highlight_pixels` asks how many pixels are within 30 of pure
    `COLOR_PAIR`. That is a question about *coverage*: a 1 px line is antialiased
    against whatever is behind it, so a pixel is only near the target colour if
    the MSAA sample pattern happened to put a full sample on it. Measured on
    1crn/biotin, the same line, the same colour and the same tolerance give
    **110 px at 8.125 A and 0 px at 27.0 A**, with nothing about the drawing
    changed in between -- at 8.125 A the line runs 208 px and most of it crosses
    the dark background at full coverage; at 27.0 A it runs 61 px and lies over
    the two CPK spheres, which touch (32.2 px radii, 60.7 px apart), so every
    pixel of it is a blend. The most pair-coloured pixel moves from BGR
    (238, 26, 238), a distance of 4, to (177, 26, 174), a distance of 68. So the
    old count went to zero on a frame where the highlight was plainly there, and
    a gate built on it would have been red for a reason that was not a defect.

    A difference has no such luck in it: any pixel the highlight covers changes,
    at any coverage, over any background. The threshold is 8 of 255 on the
    largest channel, which is derived rather than chosen -- the weakest
    highlight pixel the framebuffer can produce is one MSAA sample out of 16
    against the background, worth `(242 - 26) / 16 = 13.5` levels in red and
    blue, so 8 sits below anything real and above the frame's own dithering.

    ``-1`` when either grab is missing, so a framebuffer that could not be read
    routes to a skip instead of silently reading zero.
    """
    if on is None or off is None:
        return -1
    d = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2)
    return int((d >= 8).sum())


def _pair_marker(seg, on, off) -> dict:
    """The drawn marker, measured against where the projection says it is.

    Three questions, and a pixel count can answer none of them:

    1. **Is it there?** How many pixels the highlight owns -- the difference of
       two grabs at one camera, which is `_pair_own_pixels` and stays the right
       instrument for that.
    2. **Is it where the pair is?** Every changed pixel has to lie inside the
       rectangle the projection draws around the segment. This is the question
       the analytic projection exists for: a count is happy whether the marker
       is on the pair or 200 px away from it, and a colour test is happy if
       *something* drew that colour.
    3. **Is it findable, or merely present?** The marker's own width, measured
       perpendicular to the projected axis, against the width the product says
       it is drawing. A 1 px line and a 6 px rod of the same length and colour
       differ by a factor of four in this number and by nothing at all in a
       pixel count that only asks for 8.

    `seg` is `Viewport.pair_screen_segment()`: the two atoms pushed
    through the same `mvp` the draw used. Nothing here inspects a colour to
    decide *what* the marker is, which is what lets this run on a frame where
    the marker's own pixels are nowhere near `COLOR_PAIR` -- as they are, in
    every representation, because the line program fogs the marker toward the
    background and the fogged colour is not the nominal one.

    Percentiles, not min and max, for the width: one stray changed pixel on a
    sphere's specular rim would otherwise define the width of the whole marker.
    A framebuffer that could not be read returns `None`, so the caller skips
    rather than reading zero.

    **Two of the numbers here are instrument choices, and both were wrong
    before this round.** `pad` and the `fill` denominator are described where
    they are computed; the short version is that `pad` now has a floor under it
    (it scaled with the marker down to 0.081 px against a 0.31-0.47 px rim, so
    over the 120-1000 A band `stray_px` could not reach zero however correct
    the drawing was, and 8 of 16 swept cameras had their verdict decided by
    that alone), and `fill`'s denominator is now the part of the declared
    rectangle that is actually on the frame rather than the whole of it (the
    2 A / 45 degree row was reporting fill 0.470 for 78124 real pixels against
    a 166236 px rectangle much of which was off-screen). Both are one-sided:
    each can only reduce a false pass, never create one.
    """
    if on is None or off is None or seg is None:
        return None
    d = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2)
    mask = d >= 8
    n = int(mask.sum())
    if n == 0:
        return {"own_px": 0, "measured_px": 0.0, "declared_px": float(seg["width"]),
                "fill": 0.0, "stray_px": 0, "on_axis_px": 0,
                "contrast_p50": 0.0, "contrast_p95": 0.0}

    ys, xs = np.nonzero(mask)
    (x0, y0), (x1, y1) = seg["start"], seg["end"]
    ax, ay = seg["axis"]
    dx, dy = xs - x0, ys - y0
    along = dx * ax + dy * ay
    perp = -dx * ay + dy * ax
    # Inside the drawn segment, plus the antialiased pixel at each end.
    on_seg = (along >= -1.0) & (along <= seg["length"] + 1.0)
    # **An empty selection is a camera, not a crash.** `np.percentile` of an
    # empty array raises `IndexError: cannot do a non-empty take from an empty
    # axes`, and it raised here: when every changed pixel falls outside the
    # segment's own span -- which is what a camera looking down the pair's
    # axis produces, since the rod then has no length to sit on -- the guard
    # read the mask before checking it was non-empty. A gate that raises on a
    # camera a reader can reach is worse than one that reports it, so this
    # reports rather than skips.
    #
    # The reading is `measured_px = 0.0` and every one of the `n` pixels
    # counted as stray, because that is what the frame says: the marker drew
    # something and none of it is on the pair. `marker_findable` already
    # returns False on `stray_px > 0`, so the verdict is the same one the
    # pixels support rather than a special case invented here.
    if not on_seg.any():
        return {"own_px": n, "on_axis_px": 0, "stray_px": n,
                "measured_px": 0.0, "declared_px": round(float(seg["width"]), 3),
                "fill": 0.0, "contrast_p50": float(np.percentile(d[mask], 50)),
                "contrast_p95": float(np.percentile(d[mask], 95))}
    lo, hi = np.percentile(perp[on_seg], [2.0, 98.0])
    measured = float(hi - lo)
    # Inside the rectangle the marker claims: its length, and its own declared
    # width about the axis. A marker drawn somewhere else is outside this and a
    # count would never notice.
    #
    # **The pad is the marker's own half-width with a floor under it, and the
    # floor is the whole fix.** Two separate edges leak, and both leak by an
    # amount the marker's own size bounds. A straight side edge antialiases
    # across one pixel. The two *end faces* are flat rectangles seen at an
    # angle, and where they are oblique the resolve carries them past the
    # geometric end: measured on 1crn/biotin, 3 px sat 1.27 px before the
    # start and 1.45 px past a 65.15 px line, with perpendicular offsets of
    # 1.5-2.8 px, i.e. well inside the width and purely end bleed.
    #
    # Scaling the allowance with the marker is right, and on its own it is
    # *structurally incapable of doing the job*. `0.5 * declared` is 0.081 to
    # 0.677 px across the 120-1000 A distance band, while the antialiased rim
    # it exists to absorb is **0.31 to 0.47 px per side** and does not shrink
    # with the marker. So over that whole contiguous band the allowance is
    # thinner than the thing it absorbs, `stray_px` cannot reach zero however
    # correct the drawing is, and 8 of 16 swept cameras record a verdict
    # decided by that alone while all 4 cameras declaring more than 3 px record
    # `stray_px == 0`. An allowance that cannot reach zero is not an allowance:
    # it is a second, invisible floor on width, and it fires against correct
    # geometry.
    #
    # The floor is **measured, and 0.75 was 0.15 px short of it.** Swept from
    # 0.20 to 3.20 px in steps of 0.05, on six cameras spread over the thin
    # band (120 to 1000 A, declared widths 1.35 down to 0.16 px) at both device
    # pixel ratios, re-evaluating this function on the same two grabs:
    #
    # ==============  ==========  =====================  ===============
    # dpr             width       lowest floor giving     at 0.75, i.e.
    #                            stray_px == 0           the shipped floor
    # ==============  ==========  =====================  ===============
    # 1.25            1.353 px    0.85                   stray_px 1
    # 1.25            0.812 px    0.70                   0
    # 1.25            0.541 px    0.70                   0
    # 1.25            0.361 px    0.70                   0
    # 1.25            0.232 px    0.90                   stray_px 1
    # 1.25            0.162 px    0.60                   0
    # 2.5             1.642 px    0.20                   0
    # 2.5             0.985 px    0.70                   0
    # 2.5             0.657 px    0.75                   0
    # 2.5             0.438 px    0.70                   0
    # 2.5             0.281 px    0.80                   stray_px 1
    # 2.5             0.197 px    0.70                   0
    # ==============  ==========  =====================  ===============
    #
    # The binding case is 0.90 px, so 0.75 fired against correct geometry on
    # **3 of 12** thin-band cameras: an allowance that cannot reach zero is
    # not an allowance, it is a second invisible floor on width, and it decides
    # the verdict of a correct draw. One pixel is the value: it is at or above
    # every measured requirement, and it is the most an antialiased edge can
    # spill into on one side, so at 1.0 px `stray_px == 0` is a guarantee rather
    # than a measurement.
    #
    # **One-sided, and measured rather than asserted.** The same sweep with
    # the *draw* displaced sideways -- `_pair_quad` patched, `pair_screen_
    # segment` left as the reference, which is the only way to ask the question
    # -- gives the absorption threshold directly. At 700 A (declared half-width
    # 0.141 px) a draw landed 0.523 px off its axis is caught at a floor of
    # 1.00 (stray_px 1) and absorbed from 0.85; 0.978 px is caught at 1.00
    # (stray_px 3) and absorbed from 1.30; 2.515 px is caught at 1.00 (stray_px
    # 5) and absorbed only from 2.60. At 1000 A the same three read 0, 2 and
    # 4 at a floor of 1.00. So **a marker planted 3 px off its own axis is
    # still found misplaced at this floor** -- which is the property the floor
    # has to keep -- while the threshold tracks the planted distance, so it
    # absorbs a displacement of about the floor's own size and no more.
    #
    # **The floor is inert wherever the marker is thick, and that is the
    # formula working.** `max(0.5 * width, floor)` at a declared 6.45 px is
    # 3.225 px whatever the floor is below that, so sweeping 0.20 to 2.00 on a
    # thick marker changes `stray_px` not at all -- measured, 1104 changed
    # pixels and 0 strays at every floor in that range. The floor exists for
    # the thin band and only for the thin band, which is why the derivation
    # above is on thin cameras and a one-sidedness proof at a thick one would
    # have proved nothing.
    #
    # Below two declared pixels the pad stops scaling and sits at the floor,
    # which is the whole behaviour change.
    pad = max(0.5 * seg["width"], 1.0)
    box = ((along >= -pad) & (along <= seg["length"] + pad)
           & (np.abs(perp) <= 0.5 * seg["width"] + pad))
    # **The `fill` denominator is the part of the declared rectangle that is
    # actually on the frame.** It used to be `length * width`, the whole
    # rectangle, and that is the wrong area for a marker that runs off the
    # edge: the 2 A / 45 deg row reports 78124 changed pixels against a
    # 166236 px declared rectangle and calls the marker not findable at fill
    # 0.470, but the frame is 1113614 px and a good part of that rectangle is
    # simply not on it. The marker's area is then divided by an area it was
    # never going to occupy, and the ratio reads as a geometry failure when it
    # is a framing one.
    #
    # Clipped to the frame the way a reader would measure it: the declared
    # rectangle's own extent along the axis, cut to the frame's share of the
    # segment. One-sided in the same sense as the pad -- a marker drawn
    # somewhere else is still outside the clipped box, so this cannot make a
    # misplaced draw look findable.
    _fw, _fh = (int(v) for v in seg.get("size", (0, 0)))
    on_frame_len = seg["length"]
    if _fw > 0 and _fh > 0:
        # The share of the segment that lies inside the framebuffer, from the
        # two projected ends. `along` is measured from `start`, so the visible
        # run is the intersection of [0, length] with the axis interval the
        # frame covers; a segment entirely outside contributes nothing and the
        # denominator falls to the width alone rather than to zero.
        _x0, _y0 = seg["start"]
        _ax, _ay = seg["axis"]
        # A cheap and honest clip: the length of the segment inside the frame,
        # taken as the overlap of the axis-aligned span of the two endpoints
        # with the frame. Exact for a segment that starts and ends on screen,
        # which is every case the gate exercises, and conservative (never
        # larger) for one that does not.
        _xs = (_x0, _x0 + _ax * seg["length"])
        _ys = (_y0, _y0 + _ay * seg["length"])
        _visible = 0.0
        _steps = 256
        for _i in range(_steps + 1):
            _t = _i / _steps
            _px = _xs[0] + _t * (_xs[1] - _xs[0])
            _py = _ys[0] + _t * (_ys[1] - _ys[0])
            if -0.5 <= _px <= _fw + 0.5 and -0.5 <= _py <= _fh + 0.5:
                _visible += seg["length"] / _steps
        on_frame_len = _visible
    declared_area = max(on_frame_len * seg["width"], 1e-9)
    return {
        "own_px": n,
        "on_axis_px": int(box.sum()),
        "stray_px": int(n - box.sum()),
        "measured_px": round(measured, 3),
        "declared_px": round(float(seg["width"]), 3),
        "fill": round(n / declared_area, 4),
        "contrast_p50": float(np.percentile(d[mask], 50)),
        "contrast_p95": float(np.percentile(d[mask], 95)),
    }


def marker_findable(m) -> bool:
    """Is the marker findable, as against merely present?

    **Located, and fill.** Two conditions, both stated against the product's own
    declaration rather than against a number picked here:

    * nothing is drawn outside the projected rectangle (`stray_px == 0`), and
    * the marker's measured width accounts for at least half the width the
      product says it is drawing (`fill >= 0.5`).

    `fill` is the ratio of the marker's own pixels to the area of the rectangle
    the projection says it should occupy, so **both sides of the ratio move
    with the camera**: the 1 px line fills ~0.24 at either 25 A or 8 A, because
    it is 1 px wide against a declared 6 either way. That is what makes the
    half a statement about the drawing rather than about the fixture -- it is
    the midpoint between "a shape" and "a sub-pixel feature", and no framing
    moves a sub-pixel feature across it.

    **A correct rod fills slightly *over* 1, and that is not the rod being too
    wide.** The six-vertex rectangle projects to a trapezoid whose area is
    `length * (width_near + width_far) / 2`, and against a `width` declared at
    mid-depth that is **0.999 to 1.006** -- the taper's excess at the near end
    and its deficit at the far end very nearly cancel. The excess over that is
    the antialiased rim the `d >= 8` difference threshold counts: measured over
    ten consecutive pair rows of 1crn/biotin, **0.31 to 0.47 px per side**, for
    a fill of **1.086 to 1.144**. The drawn width tracks the geometric
    near-end width to within 0.05 of the ratio on every one of those rows, so
    the number above 1 is the instrument's edge, not a rod wider than it says.
    The floor is a floor and reads the same either way.

    **The 0.5 floor is a policy, and it is much more permissive than the
    defect it stands in for.** It is the right floor for "can a reader find
    this", and it is *not* a tolerance on the width. Over all 27 interaction
    pairs of 1crn/biotin at the framing `F` produces, the pre-fix offset drew
    the marker at these ratios of its declared width:

    ==========================  =====  ===========================================
    drawn / declared            rows   which
    ==========================  =====  ===========================================
    below 0.5 (fails this)       11    1, 2, 5, 6, 9, 10, 16, 20, 23, 24, 25
    0.5 to 0.95 (passes, thin)   12    0, 3, 7, 8, 11, 12, 13, 17, 18, 19, 21, 22
    0.95 or better (correct)      4    4, 14, 15, 26
    ==========================  =====  ===========================================

    So **23 of 27 rows were drawn measurably wrong and only 4 were right**,
    while this function calls 16 of them findable. A check that reports "16/27
    findable" is reporting its own threshold, not the product's accuracy, and
    the number to quote for the defect is the 23.

    **The ratio is a function of the segment's screen angle, and it is
    predictable without drawing anything.** With `s` the framebuffer aspect and
    `(dx, -dy)` the view-plane direction in units where the pixel axis has been
    scaled by w/2 and h/2 and y flipped, the pre-fix offset makes an angle `phi`
    with the segment and the width it draws is `|sin phi|` of the width it
    declares, `cos(phi) = (dx^2/s - dy^2) / sqrt((dx^2/s^2 + dy^2)(dx^2 +
    dy^2))`. That model predicts all 27 measured ratios to a mean absolute
    error of **0.067**, worst **0.116** -- the 0.084/0.124 this used to quote
    were stale, and the corrected figures come out of the same run that
    reproduces arm C's 1.034-1.086 exactly, so the method is sound and only
    those two numbers had drifted. The residual runs positive because the
    antialiased rim floors a thin rod's percentile width near 1 px. It equals
    1.000 only when `|dx| = |dy| * sqrt(s)`, which is why the offset looked
    right on the rows nearest 45 degrees to the axes -- and why the row that is
    genuinely correct (row 4, -134.8 deg) is not row 0.

    **The retracted arm is a separate question, and it was answered by
    measurement rather than by prose.** `|ax^2 - ay^2|` predicts what
    `cross(forward, along)` draws to a mean absolute error of 0.066 over the 27
    rows, 0.068 over a yaw sweep that rotates the pair set through -102 to -10
    degrees, and 0.109 over an aspect sweep from 0.81 to 1.83 with the screen
    angle held to 0.05 deg -- never worse than 0.14 anywhere, so the form is a
    property of the arm and "wrong on every row" was not the right claim about
    it. What makes it the wrong offset is that the quantity it predicts is
    **not 1.000**: it reaches 0.103 on row 4, 0.290 re-framed, and 0.52 across
    the held-angle sweep, where the shipped arm measured 1.09 on the same
    pixels. It is also the *aspect-free* one -- its drawn ratio moved 0.046
    while its form moved 0.000 as the framebuffer aspect swept 0.81 to 1.83 --
    so the only one of the three whose value depends on the window is the
    screen-axis arm above, and the fix removed that dependence with the defect.

    The measured widths behind these, on 1crn/biotin at the framing `F` gives
    (row 1, 27 interaction pairs, 1126x989 framebuffer at 1.25x), **as shipped,
    before the offset in `_pair_quad` was corrected** -- the six rod rows above
    are the one-row-per-representation case, and the offset defect showed up as
    a *narrower* marker in most rows and a correct one in the rest:

    ==================  ==========  ==========  =====  ====
    representation      declared    measured    fill   stray
    ==================  ==========  ==========  =====  ====
    spheres             6.45 px     5.74 px     0.94   0
    space_filling       6.01 px     5.37 px     0.94   0
    ball_and_stick      6.63 px     5.86 px     0.92   0
    stick               6.71 px     5.92 px     0.94   0
    ribbon              5.30 px     4.77 px     0.96   0
    cartoon             5.30 px     4.77 px     0.96   0
    **1 px line, any**  6.45 px     **1.49**    0.24   0
    ==================  ==========  ==========  =====  ====

    **The same six rows after the offset was corrected**, from the green run of
    this file, and the difference is the whole fix -- the declared widths do not
    move, because they come from the camera and the pose's bond radius, and
    nothing about the drawing is in them:

    ==================  ==========  ==========  =====
    representation      declared    measured    fill
    ==================  ==========  ==========  =====
    spheres             6.4 px      6.7 px      1.10
    space_filling       6.0 px      6.3 px      1.11
    ball_and_stick      6.6 px      6.9 px      1.10
    stick               6.7 px      7.0 px      1.10
    ribbon              5.3 px      5.6 px      1.13
    cartoon             5.3 px      5.6 px      1.13
    **1 px line, any**  6.45 px     1.485 px    0.24
    ==================  ==========  ==========  =====

    Both tables predate the two instrument fixes described in `_pair_marker`, so
    their `fill` column is against the *whole* declared rectangle rather than
    its on-frame share, and their `stray` column is against the unfloored pad.
    On these six rows neither changes anything -- every marker is wholly on
    frame and every declared width is above 5 px, well outside the 120-1000 A
    band where the pad floor bites -- which is why they are left as they were
    measured rather than restated against an instrument that no longer produced
    them.

    The 1 px row is the same in both tables because it is not the product's rod
    at all: the mutation below replaces the draw with a bare `Context.line()`
    and measures that, so the fix does not reach it. It is the row that has to
    stay put, and it does.

    **The 27 rows, re-measured in full, because the tables above predate both
    instrument fixes and a third axis.** Every row of 1crn/biotin was selected
    the way a reader selects it (`setCurrentCell`), the camera was allowed to
    settle, and the marker was measured with *this* `_pair_marker` -- the
    clipped `fill` denominator and the 1.0 px pad floor. Both ratios:

    ==========  ====================  =============  =======  =====  =====
                drawn / declared      screen length  fill    stray  findable
    ==========  ====================  =============  =======  =====  =====
    dpr 1.25    1.042 .. 1.073        37.6 .. 216.5  1.09+   0      27 of 27
                mean 1.061
    dpr 2.5     1.032 .. 1.054        45.6 .. 262.7  1.07+   0      27 of 27
                mean 1.044
    ==========  ====================  =============  =======  =====  =====

    Three things came out of it, and the first two are behaviour statements
    rather than praise:

    * **`stray_px` is 0 on all 27 rows at both ratios**, so raising the pad
      floor from 0.75 to 1.0 px refuses nothing that was passing. The floor
      only bites below a declared 2 px and the narrowest of the 27 is 6.47 px.
    * **The shortest of the 27 is 37.6 px of screen length**, 37.6x the 1.0 px
      length floor, so that guard refuses no real row either. The near-end-on
      cameras the length floor exists for are three orders of magnitude away
      from any of these rows.
    * The "1.03 to 1.09" this docstring used to quote for the corrected arm is
      now **1.032 to 1.073 at 1.25x** and **1.032 to 1.054 at 2.5x** -- the
      claim was directionally right and slightly wide at both ends, and the
      ratio is a little lower at the higher ratio, which is the antialiased rim
      being a smaller share of a bigger marker.

    The measured width sits ~0.35 px under the declared one in the six-row
    tables above because the marker's own antialiased edge falls below the
    difference threshold of 8 and is not counted; on the 27 rows the ratio runs
    *above* 1 because the percentile window does include that rim. That is why
    the test is a ratio and not an equality, and why it is quoted as a range.

    **Deliberately not a condition: contrast.** The marker's median per-pixel
    difference against its background runs 85.5 (space_filling) to 130 (stick)
    and would be a fine thing to gate -- except that it is a property of the
    geometry behind the marker, not of the marker. A floor on it would be a
    number chosen for this fixture's spheres, which is the trade that hides the
    real defect. It is measured and printed instead.
    """
    if not m or m.get("own_px", 0) <= 0:
        return False
    return m.get("stray_px", 1) == 0 and m.get("fill", 0.0) >= 0.5


def _cam_state(viewport) -> dict:
    """The camera's whole mutable state, so a block can put it back.

    **All four fields, and pitch and yaw are the two that matter.** The
    reachability block below drags the camera by a hundred pixels, and the
    checks that run after it ask questions of the *orientation* as well as the
    distance: `framing_selection.drawn_envelope_slack` reads the camera basis
    to decide whether the eye is outside the drawn envelope, so a block that
    leaves pitch 80 degrees from where it found it makes a later, unrelated
    check fail on a number it never touched. Two of them did, in the run that
    found this. Restoring the distance alone is not enough.
    """
    c = viewport.camera
    return {"yaw": float(c.yaw), "pitch": float(c.pitch),
            "distance": float(c.distance),
            "center": np.array(c.center, np.float32).copy()}


def _cam_restore(viewport, st) -> None:
    """Put back what `_cam_state` took, and repaint."""
    if not st:
        return
    c = viewport.camera
    c.yaw = st["yaw"]
    c.pitch = st["pitch"]
    c.distance = st["distance"]
    c.center = np.array(st["center"], np.float32)
    viewport.update()


def _off_axis_deg(viewport, u) -> float:
    """Angle between the view axis and a world direction, in degrees.

    The acute angle -- `asin |u x forward|` -- and not `acos(u . forward)`. The
    second one is a *directed* angle: it approaches 180 degrees when the camera
    is looking along the segment from the far side, which is the opposite end of
    the molecule rather than a near miss, and a sweep written that way reports
    99 to 171 degrees "off axis" for segments that are reachable end-on. A
    sweep written that way found its way into scratch and reported the product
    as unreachable when it is not.
    """
    _r, _up, fwd = (np.asarray(v, np.float64) for v in viewport.camera.basis())
    uu = np.asarray(u, np.float64)
    n = float(np.linalg.norm(uu))
    if n < 1e-12:
        return 0.0
    cross = float(np.linalg.norm(np.cross(uu / n, fwd)))
    return math.degrees(math.asin(min(1.0, cross)))


def _pair_marker_camera(win, dist):
    """Stand the camera `dist` A broadside to the selected pair and grab twice.

    Returns `(seg, grabs, seg, a, b)` so the caller can re-evaluate
    `_pair_marker` on the same two frames with a different pad floor, which is
    the whole point: the floor is a parameter of the measurement, not of the
    frame.
    """
    vp = win.viewport
    _p, _r, a, b = vp._pair_atoms()
    if a is None:
        return None, None, None, None, None
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return None, None, None, None, None
    u = d / n
    mid = 0.5 * (a + b)
    _r2, _u2, fwd = (np.asarray(v, np.float64) for v in vp.camera.basis())
    # Broadside: any direction perpendicular to the segment. The screen
    # projection is then at its longest, which is the framing the thin-band
    # table was measured at.
    axis = np.asarray([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.asarray(
        [1.0, 0.0, 0.0])
    perp = np.cross(u, axis)
    perp /= max(float(np.linalg.norm(perp)), 1e-12)
    vp.camera.center = mid.astype(np.float32)
    fx, fy, fz = (float(perp[0]), float(perp[1]), float(perp[2]))
    vp.camera.yaw = math.atan2(fx, -fy)
    vp.camera.pitch = math.asin(max(-1.0, min(1.0, -fz)))
    vp.camera.distance = float(dist)
    vp.update()
    app_q = QtWidgets.QApplication.instance()
    for _ in range(4):
        if app_q is not None:
            app_q.processEvents()
    return (vp.pair_screen_segment(), _pair_marker_grab(win),
            vp.pair_screen_segment(), a, b)


def _pair_marker_grab(win) -> dict:
    """Two grabs at the current camera, with the highlight on and then off.

    **One repaint per frame, and a collection at the end.** Every `repaint()`
    runs the whole `paintGL`, and `paintGL` builds a vertex array per draw
    call -- the spheres, the lines and the pocket mesh among them. The block
    that uses this adds a dozen grabs to the suite, and at three repaints
    apiece that was enough extra vertex-array churn to run the driver out
    before the last section: the run that did it got all 343 checks green and
    then died inside `_draw_pocket` with `cannot create vertex array`. One
    repaint is what a repaint needs, and the arrays are ordinary Python
    objects, so releasing them before the rest of the suite asks for more is
    the difference between this block being invisible and this block being the
    reason a later section fails.
    """
    vp = win.viewport
    app_q = QtWidgets.QApplication.instance()
    got = vp.highlight_pair
    vp.repaint()
    for _ in range(3):
        if app_q is not None:
            app_q.processEvents()
    on = _grab_array(vp)
    vp.highlight_pair = None
    vp.repaint()
    for _ in range(3):
        app_q.processEvents()
    off = _grab_array(vp)
    vp.highlight_pair = got
    vp.repaint()
    for _ in range(3):
        if app_q is not None:
            app_q.processEvents()
    gc.collect()
    return {"on": on, "off": off}


def _grab_array(vp):
    """`grabFramebuffer` as the BGR array the marker instruments take."""
    img = vp.grabFramebuffer()
    if img is None or img.isNull():
        return None
    return img_array(img)


def _pair_marker_with_pad(seg, on, off, pad_floor):
    """`_pair_marker` with the pad floor as an argument.

    A copy, not an edit of the live one: the shipped floor and the floor this
    asks about have to be two values of the same computation for the claim
    "0.75 is not enough and 1.0 is" to be a measurement rather than an
    assertion. Every line below the pad is `_pair_marker`'s, unchanged.
    """
    if on is None or off is None or seg is None:
        return None
    d = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2)
    mask = d >= 8
    n = int(mask.sum())
    if n == 0:
        return {"own_px": 0, "stray_px": 0, "on_axis_px": 0, "fill": 0.0}
    ys, xs = np.nonzero(mask)
    (x0, y0) = seg["start"]
    ax, ay = seg["axis"]
    dx, dy = xs - x0, ys - y0
    along = dx * ax + dy * ay
    perp = -dx * ay + dy * ax
    on_seg = (along >= -1.0) & (along <= seg["length"] + 1.0)
    if not on_seg.any():
        return {"own_px": n, "on_axis_px": 0, "stray_px": n, "fill": 0.0}
    pad = max(0.5 * seg["width"], float(pad_floor))
    box = ((along >= -pad) & (along <= seg["length"] + pad)
           & (np.abs(perp) <= 0.5 * seg["width"] + pad))
    return {"own_px": n, "on_axis_px": int(box.sum()),
            "stray_px": int(n - box.sum()), "fill": 0.0}


def _one_pixel_pair_line(self, mvp) -> None:
    """The pre-rod `_draw_pair`: one 1 px GL line between the two atoms.

    Installed by the width mutation and nowhere else. It exists so the gate can
    produce, on demand, the exact failure it is supposed to be able to detect --
    a marker that is on screen, in the right colour, at the right place, and
    1 px wide -- without editing the product to put it there. Everything the
    check above measures about the real draw stays exactly as it is; only the
    width changes.
    """
    import moderngl

    from opendocking.workbench import COLOR_PAIR

    solved = self._pair_atoms()
    if solved is None:
        return
    _pose, _receptor, a, b = solved
    self._ctx.disable(moderngl.DEPTH_TEST)
    try:
        self._draw_lines(
            np.asarray([a, b], np.float32),
            np.tile(np.asarray(COLOR_PAIR, np.float32), (2, 1)),
            mvp,
            1.0,
        )
    finally:
        self._ctx.enable(moderngl.DEPTH_TEST)


def _pair_highlight_pixels(arr) -> int:
    """Pixels of the selected pair's own colour, in a framebuffer grab.

    **Channel order, and it is BGR.** `img_array` reads a `Format_RGB32`
    image, whose bytes are blue, green, red -- so channel 0 is blue, 1 is green,
    2 is red. Measured in the wrong order this still reports a number, and the
    number is about a colour nobody drew, which is the failure
    `_channel_excess` already documents for the same reason.

    **A tolerance, and 30 is the number that matters.** The target is
    `COLOR_PAIR` in BGR. The tolerance has to be wide enough to survive the
    framebuffer's own rounding and narrow enough to leave the rest of the
    palette outside, and every colour this workbench can draw was measured
    against it. Two of them decide the number:

    * **nitrogen** `(0.20, 0.35, 0.95)` -> BGR `(51, 89, 242)`: 153 away in
      green. The first version of this test looked for "red, not much green,
      not much blue" and nitrogen sat *inside* that band, so the count would
      have been counting atoms -- and every protein is mostly nitrogen. The test
      is built on the channel that separates them instead;
    * **sulphur** `(0.90, 0.80, 0.25)` -> BGR `(64, 204, 230)`, 38 away in green
      and 38 in blue, is the nearest miss on the elemental side. It is why the
      tolerance is 30 and not 40: at 40 the count would have included the
      sulphur of every methionine and cysteine in the receptor.

    Measured rather than estimated, the whole palette -- every element colour,
    the five scene colours and the four contact classes -- has **nothing** within
    30, and the nearest neighbour is the pocket cloud at 63. After it: the
    orchid hydrophobic contact at 89, iodine at 102, carbon grey at 114. So the
    tolerance has margin on both sides, and a count of zero on a frame with
    nothing selected is a statement about this line rather than about the
    molecule.
    """
    if arr is None:
        return -1
    a = arr.astype(np.int16)
    target = np.asarray(
        [int(round(c * 255)) for c in reversed(COLOR_PAIR)], np.int16
    )
    within = 30
    return int((np.abs(a - target).max(axis=2) <= within).sum())


def _pose_coords(win):
    pose = next((m for m in win.viewport.molecules if m.role == "pose"), None)
    return np.asarray(pose.coords, np.float32) if pose is not None else np.zeros((0, 3), np.float32)


def _partner_coords(win):
    pose = next((m for m in win.viewport.molecules if m.role == "pose"), None)
    rec = next((m for m in win.viewport.molecules if m.role == "receptor"), None)
    if pose is None or rec is None:
        return np.zeros((0, 3), np.float32)
    wanted = sorted({c.partner_index for c in win.viewport.contacts
                     if 0 <= c.partner_index < len(rec.coords)})
    return (np.asarray(rec.coords, np.float32)[wanted] if wanted
            else np.zeros((0, 3), np.float32))


def _settle_camera(win, app, budget_ms: int = 800) -> bool:
    """Wait, on the clock, until no camera move is in flight.

    The move is driven by a wall-clock `QTimer`, so a loop of
    `processEvents()` returns with the camera still moving: that loop burns its
    iteration budget in a few milliseconds and the timer has not ticked. Reading
    `camera.distance` through it therefore samples *a moment of a move*, not a
    camera position. Measured on this machine, the same pose read 23.6 A on one
    run and 45.4 A on another with the poses byte-identical, which is the
    symptom this exists to remove.

    `QTest.qWait` is the only thing here that advances the clock, for the same
    reason `settle()` in `framing_selection_screens.py` uses it. Returns whether
    the camera came to rest inside the budget, so a caller can turn "never
    settled" into a different claim from "settled somewhere".
    """
    from PyQt6.QtTest import QTest

    waited = 0
    while win.viewport.moving and waited < budget_ms:
        QTest.qWait(10)
        app.processEvents()
        waited += 10
    for _ in range(4):
        app.processEvents()
    return not win.viewport.moving


def _run_camera_move(win, app) -> bool:
    """Select a pose and sample the camera on every frame of the move.

    Returns False -- rather than recording anything -- when the move does not
    run to completion, so the caller decides between a measurement and a skip
    and does not have to trust a partial one.

    **The waiting is on the clock, and that is not incidental.** The move is
    driven by a `QTimer`, so a loop of `processEvents()` burns its iteration
    budget in a few milliseconds and returns with the camera still in flight.
    Measured on this machine, the two frames either side of such a "settled"
    read differ by 47% of the frame, because they are of two different camera
    positions -- and a check that differences two frames to decide what a role
    owns would then be measuring the camera.
    """
    global plan_distances, move_ms, plan_row
    from PyQt6.QtTest import QTest

    plan_distances = []
    move_ms = 0
    plan_row = -1

    win.viewport.frame_all()
    for _ in range(10):
        app.processEvents()
    rows = win.pose_table.rowCount()
    if rows < 2:
        return False
    row = next((r for r in range(rows) if r != win.pose_table.currentRow()), 0)
    plan_row = row
    win.pose_table.selectRow(row)
    # Sample every frame the move produces, then wait for it to finish. The
    # samples are the point: they are what distinguishes a move from a snap.
    for _ in range(4000 // 10):
        app.processEvents()
        if win.viewport.moving:
            plan_distances.append(float(win.viewport.camera.distance))
        QTest.qWait(5)
    for _ in range(10):
        app.processEvents()
    return not win.viewport.moving


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)