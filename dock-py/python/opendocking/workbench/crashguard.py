"""What the workbench does when an exception escapes a Qt slot.

# The fault

An uncaught exception raised inside a PyQt6 slot ends the process with
``0xC0000409`` and prints **nothing**: no traceback, and every statement after
the raise is skipped. Measured in three separate offscreen processes on this
machine, and it is not one exception type -- ``raise ValueError`` in a slot
behaves exactly as ``NameError`` does. The same exception raised in a plain
function and caught by its caller is harmless, so the boundary is the Qt
dispatch, not the exception. ``scripts/workbench_interaction_check.py`` section
12 drives the product's own entry point and asserts every property claimed
here, in both directions.

That failure mode is the worst kind for a gate. The process dies, so the run
ends non-zero and nobody calls it a pass -- but the log is empty, so a reader
cannot tell a driver fault from a bug in a slot, and a shell script that only
keeps stdout sees nothing at all.

# What this module does, and what it refuses to do

`install()` puts a `sys.excepthook` in place that does two things: it prints the
traceback, and it makes the process end **non-zero**.

The second half is the half that is easy to get wrong in the direction that
hurts. A hook that prints the traceback and returns leaves the process alive
and lets it exit 0, and a gate that used to abort now reports green: the crash
has been converted into a silent pass, with a traceback nobody reads. So the
traceback is not the point; the **exit code** is, and everything below is in
service of not losing it.

So:

* the exception is named with its file and line, and `traceback` prints the
  chain when there is one;
* the failure is **recorded**, so a caller can ask what happened
  (`failure()`, `failures()`) rather than infer it from a log;
* the process is ended with `EXIT_UNHANDLED` (70). `70` is `sysexits`
  `EX_SOFTWARE`; it is deliberately far from this project's 2/3/4/5 argument and
  diagnostics codes so a script that switches on those cannot mistake an
  internal failure for a usage error or an OpenGL verdict.

# How the code gets back out, and why that is not automatic

`sys.excepthook` is called *after* the interpreter has already unwound to the
top, so there is nothing left to return a code through. Two ways out, and the
choice between them is the whole design:

1. Inside a loop the product owns (`event_loop()`, used by `app.run`),
   `QCoreApplication.exit(70)` is called, `exec()` returns 70, and the window
   closes the way it closes for any other exit. This is the graceful path.
2. Everywhere else -- a check script pumping `processEvents()`, a probe child
   that calls `sys.exit()` itself, anything that never entered `event_loop()` --
   the code cannot be handed anywhere, so the process is ended directly with
   `os._exit(70)`.

Path 2 is the important one, and it is why the hook does not simply look for a
`QApplication` and call `exit()`. `QThread.loopLevel()` is > 0 for a
`processEvents()` pump as well as for `exec()`, so a hook that trusted it would
call `exit()`, watch the process carry on, and exit 0 -- the silent pass this
module exists to prevent. Only the product's own `run()` opts into the graceful
path, and it says so by saying it.

`finalise()` is the backstop under both: a recorded failure can never be
reported as 0, whatever `exec()` happened to return.

# What it deliberately does not touch

* **An exception that is caught does not reach a hook.** `sys.excepthook` is
  only called for exceptions that escape; Python code with a `try` never
  reaches it, so a product `except` block -- `TermsUnavailable` and the other
  handled paths -- is unaffected. This is a property of where the hook is
  installed, and section 12 asserts it rather than leaving it to be believed.
* **Clean runs are untouched.** The hook is a function the interpreter calls
  only when something has already gone wrong, so a run where nothing raises
  executes no line of it. The installed hook still prints nothing and the
  process still exits 0; section 12 compares a real `--check` run with and
  without the hook in place and requires the payloads to agree.
* **`odcli` is not a GUI entry point** and is not covered. `install()` is called
  from `app`, which every GUI entry point reaches -- `odgui`,
  `python -m opendocking.workbench.launcher` and `odcli workbench` all go
  through `workbench.launch` -> `app.run` -- and which the `odgui --check`
  stage-2 child imports as well. `workbench/__init__` was the other candidate
  and was rejected: `cli` imports it for `MoleculeView` and `pockets` on the
  `sites` and `info` paths, which have no event loop and no slots, so a hook
  there would change a CLI's behaviour for a fault the CLI cannot have.
"""

from __future__ import annotations

import contextlib
import os
import sys
import traceback

__all__ = [
    "EXIT_UNHANDLED",
    "active",
    "event_loop",
    "failure",
    "failures",
    "finalise",
    "install",
    "installed",
    "replaced_hook",
]

#: The exit code an unhandled exception earns. `sysexits`' ``EX_SOFTWARE``, and
#: far enough from the 2/3/4/5 codes this project already returns to be
#: unambiguous to a script that switches on them.
EXIT_UNHANDLED = 70

_replaced = None
_installed = False
_records: list[dict] = []
_loop_owned = False


def install() -> bool:
    """Put the hook in place. Idempotent; returns whether it did anything.

    Called once from `app` at module scope, immediately after `_require_gui()`,
    which is the earliest point every GUI entry point shares and the latest
    point that is still guaranteed to be a GUI process. `_require_gui` comes
    first on purpose: a machine without PyQt6 has no slots to raise in, and the
    diagnostic that says so is better than a hook guarding nothing.
    """
    global _replaced, _installed
    if _installed:
        return False
    _replaced = sys.excepthook
    sys.excepthook = _hook
    _installed = True
    return True


def installed() -> bool:
    """Whether `install()` has run in this process."""
    return _installed


def active() -> bool:
    """Whether this hook is the one `sys.excepthook` will call *right now*.

    Not the same question as `installed()`. A caller can install the hook and
    then put the old one back -- which is exactly what the clean-run comparison
    in section 12 does, so that "with the hook" and "without it" can be two runs
    of the same process rather than two beliefs about two processes. `installed`
    stays true while `active` goes false, and a check that confuses the two
    would compare a hook against itself.
    """
    return sys.excepthook is _hook


def replaced_hook():
    """The hook `install()` displaced, for a run that wants the old behaviour.

    Kept rather than discarded so the clean-run comparison in section 12 is a
    comparison and not a belief: the same process, the same payloads, the hook
    present or not.
    """
    return _replaced


@contextlib.contextmanager
def event_loop():
    """Mark the span where this process owns the event loop.

    Only inside it does the hook end the process by asking the loop to return
    `EXIT_UNHANDLED`; outside it, the hook ends the process itself, because no
    loop is going to hand a code back to anybody.
    """
    global _loop_owned
    previous = _loop_owned
    _loop_owned = True
    try:
        yield
    finally:
        _loop_owned = previous


def failures() -> tuple:
    """Every unhandled exception this process saw, oldest first."""
    return tuple(_records)


def failure() -> dict | None:
    """The first one, or `None`. What a caller should act on."""
    return _records[0] if _records else None


def finalise(code: int) -> int:
    """The code to return, with a recorded failure always winning.

    The backstop under `_stop`. `QCoreApplication.exit(70)` is supposed to make
    `exec()` return 70, and a caller that trusts its own `exec()` would be right
    most of the time; this makes "a recorded failure is never reported as 0"
    true regardless of what the loop returned.
    """
    if _records and code == 0:
        return EXIT_UNHANDLED
    return code


def _bottom_of_chain(exc: BaseException) -> BaseException:
    """The original exception behind a chain, the way `launcher._root_cause` walks it."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        nxt = exc.__cause__ or (
            None if exc.__suppress_context__ else exc.__context__
        )
        if nxt is None:
            break
        exc = nxt
    return exc


def _where(exc_type, tb) -> str:
    """`file:line` of the raise, from the traceback the interpreter handed over."""
    frames = traceback.extract_tb(tb)
    if not frames:
        return "an unknown location"
    last = frames[-1]
    return f"{last.filename}:{last.lineno}"


def _hard_exit(code: int) -> None:
    """End the process now, skipping interpreter shutdown.

    A named function rather than a bare `os._exit` at the call site, for one
    reason: it is the seam section 12 drives. The "no loop of ours is running"
    branch is otherwise unobservable from outside, because observing it means
    letting it happen -- so a check replaces this one function, calls the hook,
    and asks whether it was reached. `os._exit` patched instead would be
    patching the `os` module for the whole process, which is the kind of
    mutation that outlives the check that wanted it.
    """
    os._exit(code)


def _stop(code: int) -> None:
    """End the process with `code`, gracefully if this process owns the loop."""
    if _loop_owned:
        try:
            from PyQt6 import QtCore

            app = QtCore.QCoreApplication.instance()
            if app is not None:
                app.exit(code)
                return
        except Exception:  # noqa: BLE001 - the hard exit below is the fallback
            pass
    # Nothing will hand this code back: no loop of ours is running, or there is
    # no QApplication at all. The traceback is already on stderr and flushed,
    # so what is lost is only interpreter shutdown, on a process that has
    # already lost the thing it was doing.
    _hard_exit(code)


def _hook(exc_type, exc, tb) -> None:
    """`sys.excepthook`: print it, record it, and end the process non-zero."""
    global _records
    try:
        where = _where(exc_type, tb)
        name = getattr(exc_type, "__name__", str(exc_type))
        bottom = _bottom_of_chain(exc)
        chained = bottom is not exc
        _records.append(
            {
                "where": where,
                "exception": name,
                "message": str(exc),
                "chained_from": (
                    f"{type(bottom).__name__}: {bottom}" if chained else None
                ),
            }
        )
        print(
            f"\nworkbench: unhandled exception in a Qt slot -- {where}: {name}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        if chained:
            print(
                f"workbench: the original failure was {type(bottom).__name__}: "
                f"{bottom}",
                file=sys.stderr,
            )
        print(
            f"workbench: ending with exit code {EXIT_UNHANDLED}. An unhandled "
            f"exception is a failure; the process does not continue and does not "
            f"report success.",
            file=sys.stderr,
        )
        sys.stderr.flush()
        sys.stdout.flush()
    except BaseException:  # noqa: BLE001 - a hook that raises loses the exit code
        pass
    _stop(EXIT_UNHANDLED)
