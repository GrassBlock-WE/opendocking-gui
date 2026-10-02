"""Every key this window answers to, in one table, and the map that prints it.

The table is the whole feature's contract. A shortcut and the line of text that
describes it come from the same row, so a key cannot exist without being
documented and a documented key cannot be missing from the map -- which is the
failure mode of the usual arrangement, where the map is a label somebody
remembers to update.

Three things about how the table is written, each one a defect it prevents.

**The handler is named, and the name is resolved at install time.** A typo in a
string would otherwise be discovered the first time a user pressed the key, as
an exception out of a Qt slot -- which ends the process at `0xC0000409` with
nothing on stderr. `resolve` raises here, at construction, naming the group and
the key.

**Availability is a question, not a guess.** `enabled_by` names a method on the
window that answers "can this act right now, and if not, what is missing". A
shortcut that cannot do anything is drawn dimmed with the reason next to it,
because "Left does nothing" and "Left is not available because no pose is
selected" are different facts and a user can only act on the second one.

**This module imports no Qt.** The workbench package must import on a machine
with no PyQt6 installed -- `app.py` says so at the top and calls
`_require_gui()` for exactly that reason. So the table and the map text live
here, and `app.py` builds the `QShortcut` objects out of them. That also means
the map can be rendered in a check that has no display.

The existing keyboard inventory this table starts from, so that a duplicate
cannot be added without being noticed:

* five `Ctrl` chords on the File menu -- `Ctrl+R` receptor, `Ctrl+L` ligand,
  `Ctrl+P` poses, `Ctrl+D` dock, `Ctrl+Q` quit (`app.py`'s `_build_ui`);
* the pose table's **own** `Up`/`Down`, which move the current row and so fire
  `currentCellChanged` and the whole selection pipeline. That one is not
  re-implemented here: it is listed in the map as what it already is, and the
  map says it only applies while the table has focus. Measured, not assumed: on
  this machine `Up` and `Down` with the table focused changed the selected pose
  and the RMSD readout, and with the viewport, the window, the display combo or
  a spin box focused they changed nothing at all.

Nothing else in the package handles a key: there is no `keyPressEvent`, no
`QShortcut`, no `QKeySequence` and no event filter anywhere under `workbench/`.
So every key in the table below is either one of those two or new, and the new
ones were chosen to be letters no existing chord uses.

**`Up` and `Down` appear twice, and that is two tables rather than a duplicate.**
One pair of rows is the pose table's, one is the interaction pair table's, and
they are in different groups because they apply under different focus. A key
listed twice in the *same* group would be the defect this file is for; the same
key doing the same job in two different widgets is two widgets. Neither is
installed as a shortcut: `QTableWidget` answers both on its own and fires
`itemSelectionChanged`, which is the signal a click fires too, so the keyboard
and the mouse reach the highlight by the same path.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The three groups the map is drawn in. `always` means the key acts whatever
#: has focus, which is what a window-level shortcut gives; `pose table` and
#: `pair table` are the tables' own key handling, listed so a user can see they
#: are focus-dependent rather than broken.
GROUP_ALWAYS = "always"
GROUP_TABLE = "pose table focused"
GROUP_PAIRS = "pair table focused"
GROUP_OVERLAY = "map open"
GROUP_ORDER = (GROUP_ALWAYS, GROUP_TABLE, GROUP_PAIRS, GROUP_OVERLAY)

#: The colour of a row that cannot act, and of the group headings. Chosen for
#: contrast on the white card this is drawn on, not on the dark viewport it is
#: drawn over -- see `map_html`.
DIM = "#5c5c5c"

#: The heading for the group whose name would otherwise over-claim. "Always" was
#: wrong: a focused numeric field never offers its keys to the shortcut map, and
#: a heading that says otherwise is worse than no heading.
GROUP_ALWAYS_TITLE = "Wherever focus is, except a field taking the key for itself"


def group_titles() -> dict:
    """The heading for each group, **built on demand**.

    Not a module-level dict, and that is the whole reason it is a function. A
    dict literal next to the constants captures `GROUP_ALWAYS_TITLE` at import
    time, so a later edit to the constant -- including the one
    `scripts/workbench_interaction_check.py` section 13 makes on purpose, to
    prove the map's heading is a live claim rather than a frozen one -- would
    leave the map printing the old sentence while the constant said the new
    one. That check passed on the nested-conditional version of this, and would
    have failed here for a reason that has nothing to do with the heading.

    Built per call because it is called once per map, and a four-entry dict
    costs nothing next to the HTML around it.
    """
    return {
        GROUP_ALWAYS: GROUP_ALWAYS_TITLE,
        GROUP_TABLE: "While the pose table has focus",
        GROUP_PAIRS: "While the pair table has focus",
        GROUP_OVERLAY: "While this map is open",
    }

#: The one measured context where a key does not reach the window, and the
#: sentence the map prints about it.
#:
#: Declared as data as well as prose, because a prose claim about behaviour
#: cannot be checked and this one is the kind that goes stale silently: a future
#: widget that eats keys would leave the map saying "always" and nothing would
#: notice. `scripts/workbench_interaction_check.py` measures the swallow set and
#: compares it with `EXCEPTION_CLASS_NAMES`, so a new offender goes red there
#: rather than in a user's hands.
#:
#: Measured on 1crn after a real run, with `A` and with `Right`, on the window
#: as shipped: the pose table, the 3D view, the window itself, the display combo
#: and the status bar all act. Both spin boxes swallow. The reason is a spin
#: box's own focus living on a `QLineEdit`, and a line edit taking letters and
#: arrows for its own job -- the shortcut map is never offered the key. That is
#: the field doing its job rather than a key being lost: the arrows increment
#: the box, which is what a numeric field is for, and the letters are rejected
#: by its validator. So the fix is not a second route for the same key, which
#: would leave two paths and one of them untested; the fix is to say it, and to
#: name the gesture that gets the keys back.
EXCEPTION_CLASS_NAMES = ("QAbstractSpinBox",)
CONTEXT_EXCEPTION_NOTE = (
    "These keys act whatever has focus, with one measured exception: a focused "
    "numeric field. A spin box keeps its focus on a line edit, and a line edit "
    "takes letters and arrows for its own job, so the key never reaches the "
    "window -- the arrows step that box's value instead. That is the field "
    "doing what a numeric field is for, not a lost key, and there is no second "
    "route for the same key here on purpose. Tab or a click on the 3D view "
    "hands the keys back."
)


@dataclass(frozen=True)
class Shortcut:
    """One key, what it does, and what has to be true for it to do it."""

    keys: tuple[str, ...]
    action: str
    group: str
    handler: str
    enabled_by: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        return " / ".join(self.keys)


#: The table. Read it as the specification: every key the window answers to, in
#: one place, with the context it applies in.
#:
#: `Left`/`Right` are the pose steppers rather than `Up`/`Down` because the
#: table already owns `Up`/`Down` and re-implementing them here would be the
#: duplicate path this file exists to prevent. They are registered as window
#: shortcuts, and the measurement that they beat the table's own *column*
#: navigation -- `Right` on the focused table moved column 0 to 1 and left the
#: pose alone -- is in `scripts/workbench_interaction_check.py`.
#: What the export key does when there is nothing to export, on the row itself
#: rather than only in this file. The map is the place a reader looks before
#: pressing something, so the two things they cannot see from a key list -- where
#: the file goes, and what happens with no run -- belong next to the key. Above
#: the table because the table is built at import and a note defined after it
#: would be a `NameError` rather than a missing sentence.
EXPORT_NOTE = (
    "Writes every pose with its energies, terms, verdict, contacts and "
    "provenance, and prints the path in the status bar; with no pose loaded it "
    "says so and writes nothing"
)

#: What the dock key does, including the two things a reader cannot find out from
#: a key list: how long it takes, and what pressing it again does. The window does
#: not block while a search runs -- the work and the map precalculation are on a
#: worker thread and the event loop keeps turning -- but a search that returns
#: nothing looks exactly like a search that is still going, so the note says both.
#: Asking again while one is running is refused with a reason, never queued: two
#: searches sharing one pair of thread slots is how a result goes missing, and a
#: queued request that never runs is the silent failure this project has named
#: twice already.
DOCK_NOTE = (
    "Runs the search for the box, exhaustiveness and seed in the panel, on a "
    "worker thread, so the window stays live; it takes seconds to minutes "
    "depending on the box, and the panel says which stage it is at. Pressed "
    "again while a search is running it says so and starts nothing, rather than "
    "queuing a second search behind the first"
)

SHORTCUTS: tuple[Shortcut, ...] = (
    Shortcut(("Left",), "Previous pose", GROUP_ALWAYS,
             "_key_step_pose", "_keys_step_enabled"),
    Shortcut(("Right",), "Next pose", GROUP_ALWAYS,
             "_key_step_pose", "_keys_step_enabled"),
    Shortcut(("Up",), "Previous pose (the table's own key)",
             GROUP_TABLE, ""),
    Shortcut(("Down",), "Next pose (the table's own key)", GROUP_TABLE, ""),
    # The pair table's own keys, and the reason they are listed rather than
    # wired. `QTableWidget` moves its current row on Up/Down itself, and that
    # move fires `itemSelectionChanged` -- the same signal a click fires and the
    # same slot answers -- so the selected pair is highlighted, centred and named
    # in the status bar by the key exactly as it is by the mouse. Re-implementing
    # that as a window shortcut is the duplicate path this file exists to
    # prevent: two handlers for one gesture, and the one nobody tested.
    # `handler` is empty for the same reason the pose table's rows are empty:
    # there is no callable here to resolve, because the table is the handler.
    Shortcut(("Up",), "Previous interaction pair (the table's own key)",
             GROUP_PAIRS, ""),
    Shortcut(("Down",), "Next interaction pair (the table's own key)",
             GROUP_PAIRS, ""),
    Shortcut(("A",), "Orbit the camera left", GROUP_ALWAYS, "_key_rotate"),
    Shortcut(("D",), "Orbit the camera right", GROUP_ALWAYS, "_key_rotate"),
    Shortcut(("W",), "Orbit the camera up", GROUP_ALWAYS, "_key_rotate"),
    Shortcut(("S",), "Orbit the camera down", GROUP_ALWAYS, "_key_rotate"),
    # The rule is in the description because it is the whole content of the key,
    # and because it used to be two rules. Selecting a pair put the camera
    # exactly on that pair -- 0.00 A from its midpoint -- and pressing this key
    # then took the camera 8.80 A away and pulled back 3.8x, so a 2.05 A line
    # fell from 32 px of highlight to 5. "I selected this, then asked to frame
    # it, and it is now tiny" is a gesture that undoes itself, and a key whose
    # own text claims to frame the selection is the wrong place for that. The
    # note says what it does instead, which is nothing: it re-runs the framing
    # the selection already earned, through the same viewport call.
    Shortcut(("F",), "Frame whatever is selected -- pair, residue or pose; "
                      "all poses if nothing is",
             GROUP_ALWAYS, "_key_frame_selection", "_keys_needs_pose",
             note="Re-runs the camera move the selection already made, through "
                  "the same call, so the key cannot undo the selection it is "
                  "meant to show"),
    # The one key that produces something rather than looking at it, so it is
    # first among the letters: `K` because `D` is already the camera's orbit
    # right and this window will not give a key two meanings. The File menu's
    # "Dock now" and `Ctrl+D` call the same method as this row does -- the same
    # one the button calls -- and the measurement that pressing it actually
    # starts a search is in `scripts/workbench_interaction_check.py`, by pressing
    # it and watching the poses arrive rather than by finding the shortcut
    # installed.
    Shortcut(("K",), "Run a docking search for this box", GROUP_ALWAYS,
             "_key_dock", "_keys_dock_enabled", note=DOCK_NOTE),
    Shortcut(("C",), "Show or hide the interaction lines", GROUP_ALWAYS,
             "_key_toggle_contacts"),
    Shortcut(("V",), "Show or hide the site volume cloud", GROUP_ALWAYS,
             "_key_toggle_site_volume"),
    Shortcut(("M",), "Next display representation", GROUP_ALWAYS,
             "_key_cycle_representation"),
    # Taking the run away. `E` for export, a letter no other row here claims,
    # and it is in the `always` group because it does not need a pose to be
    # worth having: with none, it refuses and says why, which is an answer too.
    # The `File` menu's "Export run to a file" calls the same method, so the key
    # and the menu item cannot write two different files -- and the note is on
    # the row because "where did it go" is the first question a reader has.
    Shortcut(("E",), "Export the run to a file", GROUP_ALWAYS, "_key_export",
             note=EXPORT_NOTE),
    Shortcut(("?",), "Show or hide this map", GROUP_ALWAYS, "_key_toggle_map",
             note="Shift and /"),
    Shortcut(("Esc",), "Close this map", GROUP_OVERLAY, "_key_close_map",
             enabled_by="_keys_map_open"),
)

#: The key that opens the map, and why this one. It is not a letter, so it
#: cannot collide with the eight single-letter view keys above, and it is the
#: platform-independent convention for "what can I press" -- `F1` is the
#: Windows one but is claimed by window managers more often than `?` is, and
#: the five chords this window already owns are all `Ctrl`, so nothing here is
#: taken. Shift is required, which is why the row says so.
MAP_KEY = "?"

#: The key that closes it. `Esc` is the one key no list widget eats and no
#: shortcut in this table competes with.
CLOSE_KEY = "Esc"

#: Why each of the four camera keys is a letter and not a symbol: the view is
#: driven by holding a mouse button, and the letters are the first-person
#: camera's own convention, so the key that rotates the camera is the key a
#: reader reaches for without being told. They are one key per direction rather
#: than one key with a modifier, because a key that needs a modifier is a key
#: that needs a second hand.
CAMERA_NOTE = (
    "Orbit is the same arithmetic the left-drag does, through the same method; "
    "one degree per press, and the pitch stops at the same +/-1.5 rad the drag "
    "stops at."
)


def keys_for(handler: str) -> tuple[tuple[str, ...], ...]:
    """Every key tuple bound to ``handler``, in table order."""
    return tuple(s.keys for s in SHORTCUTS if s.handler == handler)


def rows(win) -> list[dict]:
    """The map's rows, with availability answered *now*.

    ``win`` is duck-typed on purpose -- anything with the named methods answers,
    so this runs in a check with no display. A row whose ``enabled_by`` method
    is missing is reported as unavailable with the reason, never skipped: a
    missing method is a defect in the window, and a map that quietly omits a row
    hides it.
    """
    out = []
    for s in SHORTCUTS:
        available, reason = True, ""
        if s.enabled_by:
            probe = getattr(win, s.enabled_by, None)
            if probe is None:
                available, reason = False, (
                    f"the window has no {s.enabled_by}, so this key cannot be "
                    f"answered for"
                )
            else:
                available, reason = probe(s)
        out.append({
            "keys": s.keys,
            "label": s.label,
            "action": s.action,
            "group": s.group,
            "note": s.note,
            "available": bool(available),
            "reason": reason,
            "installed": bool(s.handler),
        })
    return out


def resolve(win) -> dict[str, object]:
    """Bind every handler name to a real callable, or say which one is missing.

    Called once, while the window is being built. A missing attribute is a
    `AttributeError` at construction rather than inside a Qt slot later, which
    is the difference between a window that opens and a process that dies
    without a word on stderr.
    """
    bound: dict[str, object] = {}
    for s in SHORTCUTS:
        if not s.handler:
            continue
        fn = getattr(win, s.handler, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"the shortcut table names {s.handler!r} for "
                f"{s.label} ({s.action}) and the window has no such callable. "
                f"A key that cannot be wired is a key that silently does "
                f"nothing, so this is raised at construction instead."
            )
        bound[s.handler] = fn
    return bound


def map_html(win, pose_summary: str = "") -> str:
    """The map, as the HTML one label renders.

    Grouped, not flat: a flat list of sixteen rows is a list nobody reads, and
    the grouping is the part that answers "does this key apply to me right now".
    Availability is drawn in the same row as the key rather than in a separate
    column, so a dimmed key is dimmed *with* its reason.

    The dimmed colour is `#5c5c5c` and not the `#9aa3ad` the sidebar uses for its
    captions, because this text is drawn on a **white card** over a dark
    viewport, and 2.6:1 is not a contrast ratio a reader can use. A shortcut map
    whose "unavailable" rows are the least legible rows is a map that hides the
    thing it exists to show.
    """
    data = rows(win)
    parts = [
        "<div style='font-weight:bold; margin-bottom:2px;'>Keyboard</div>",
        f"<div style='color:{DIM}; margin-bottom:6px;'>"
        "A dimmed key cannot act at this moment, and says why. "
        + (f"{pose_summary}. " if pose_summary else "")
        + "Pose stepping goes through the same selection path as clicking the "
        "row, so a key and a click give the same pose, camera, overlays, "
        "breakdown and verdict.</div>",
    ]
    for group in GROUP_ORDER:
        group_rows = [r for r in data if r["group"] == group]
        if not group_rows:
            continue
        title = group_titles().get(group, group)
        parts.append(
            f"<div style='margin-top:6px; color:{DIM};'>{title}</div>"
            "<table cellspacing='0' cellpadding='1'>"
        )
        for r in group_rows:
            # A table rather than a padded span: Qt's rich text ignores
            # `display:inline-block`, so the first version of this printed
            # "LeftPrevious pose" -- the key and its own description with
            # nothing between them, which is the one thing a shortcut map must
            # not do. Qt renders tables faithfully, so the two columns are the
            # reliable way to put a gutter between them.
            key_cell = " ".join(f"<b>{k}</b>" for k in r["keys"])
            note = (f" <i>{r['note']}</i>" if r["note"] else "")
            if r["available"]:
                parts.append(
                    "<tr>"
                    f"<td width='64' align='left' valign='top'>{key_cell}</td>"
                    f"<td align='left' valign='top'>{r['action']}{note}</td>"
                    "</tr>"
                )
            else:
                parts.append(
                    f"<tr style='color:{DIM};'>"
                    f"<td width='64' align='left' valign='top'>{key_cell}</td>"
                    f"<td align='left' valign='top'>{r['action']} "
                    f"<i>- unavailable: {r['reason']}</i></td>"
                    "</tr>"
                )
        parts.append("</table>")
    if CAMERA_NOTE:
        parts.append(
            f"<div style='color:{DIM}; margin-top:6px;'>{CAMERA_NOTE}</div>"
        )
    if CONTEXT_EXCEPTION_NOTE:
        parts.append(
            f"<div style='color:{DIM}; margin-top:6px;'>"
            f"{CONTEXT_EXCEPTION_NOTE}</div>"
        )
    return "".join(parts)
