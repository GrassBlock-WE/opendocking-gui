"""One place that asks `odgui --check` whether this machine can make a context.

**Why this file lives here and not somewhere else.**

It is a sibling of the check scripts rather than part of the installed package,
for three reasons that each cost something if ignored:

* Python puts a script's own directory at `sys.path[0]`, so `python
  scripts/workbench_smoke.py` imports this with no path juggling at all -- and
  the four consumers already disagree about how to reach `opendocking` itself
  (some prefer the installed wheel, some prepend the source tree), so a helper
  that needed its own `sys.path` fix would be a fifth thing to get right.
* It is a test helper, not product code. Inside `opendocking/` it would become
  part of the shipped wheel and of the public API, and `workbench/` is not this
  file's to edit in any case.
* The leading underscore marks it as a helper rather than a check, which matters
  because `CONTRIBUTING.md` and `.github/workflows/ci.yml` both enumerate
  `scripts/*.py` as things to *run*. A file called `gui_check.py` would be
  mistaken for a gate; `_gui_check.py` cannot be.

**Why it exists at all.** Four of the check scripts had their own byte-identical
copy of the same twenty-line `odgui --check` wrapper, each parsing the *first
line of prose* as the reason. Four copies of one heuristic is three too many:
a fix to the wrapper would have had to be made four times, and three of them
would have been missed. The heuristic was also wrong in the case that mattered
most -- on success the first line of stdout is the banner
`Open Docking Workbench: GUI stack OK`, which is not a *reason* for anything, so
a caller that took it as one would have been handed a title where it asked for
a cause. `odgui --check --json` removes the guessing: the prose report and the
JSON object are formatted from the same `found` dictionary in `launcher.py`, so
there is a named field to read (`problem` for both failure verdicts) rather than
a line to interpret.

**Degrading rather than duplicating.** If `--json` is unavailable -- an older
`odgui` on `PATH`, which is a real state when the installed wheel lags the
source tree -- the prose path below still answers the question, and says in the
reason that it did so and why. The fallback lives here and nowhere else, which
is the whole point: degrading must not turn into a second copy reappearing in
four files.

Run this file directly to audit that the four consumers still converge on it:

    python scripts/_gui_check.py
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

#: Generous, because the honest answer costs real time on a machine that cannot
#: supply a context: Qt blocks *inside* `show()` for seconds before it gives up,
#: and that wait belongs to Qt. Measured 16.7 s end to end on this machine.
TIMEOUT = 120

VERDICT_OK = "ok"
VERDICT_NO_CONTEXT = "no-context"
VERDICT_NO_GUI_STACK = "no-gui-stack"

#: `odgui --check` exit codes, unchanged from the launcher's own contract.
EXIT_OK = 0
EXIT_NO_GUI_STACK = 3
EXIT_NO_CONTEXT = 4


@dataclass(frozen=True)
class Check:
    """What `odgui --check` said, in a form a caller can act on.

    `verdict` is the field to branch on, not `code`: the two agree today, but
    `verdict` is the one the launcher documents as its answer, and it is the one
    that stays meaningful if a future exit code is added for a reason the
    numbers do not convey.
    """

    code: int
    verdict: str
    reason: str
    #: ``"json"`` when the object came from `--check --json`; ``"prose"`` when it
    #: was recovered from an older launcher; ``"unavailable"`` when `odgui` could
    #: not be run at all. A caller that wants to warn about an old launcher can
    #: read this instead of re-deriving it from the reason text.
    source: str
    detail: dict = field(default_factory=dict)

    @property
    def can_make_context(self) -> bool:
        return self.verdict == VERDICT_OK

    def describe(self) -> str:
        """A reason fit to print, with the provenance of the answer attached.

        The `source` is included even on success, because "this machine cannot
        make a context" and "we could not find out whether this machine can make
        a context" are different sentences and a reader of a skipped check needs
        to know which one they are looking at.
        """
        return f"{self.reason} [odgui --check {self.source}, exit {self.code}]"


def _run(argv: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None


def _reason_from_json(payload: dict) -> str:
    """A reason for every verdict, so `reason` is never empty.

    On failure that is the `problem` field the launcher writes, which carries
    the facts rather than a paraphrase -- on a machine whose driver dies
    natively it includes the child's own exit status, e.g. `3221226505`
    (`0xC0000409`), which is the difference between "Qt could not create a
    context" and "the probe process was killed outright".
    """
    problem = payload.get("problem")
    if problem:
        return str(problem)
    if payload.get("verdict") == VERDICT_OK:
        bits = [str(payload.get("platform", "?"))]
        if payload.get("gl"):
            bits.append(f"OpenGL {payload['gl']}")
        bits.append("the GUI stack imported")
        return ", ".join(bits)
    return f"`odgui --check --json` reported verdict {payload.get('verdict')!r} " \
           "with no problem field"


def odgui_check() -> Check:
    """Ask whether this machine can give Qt the context the viewport needs.

    Tries ``odgui --check --json`` first and reads the object. If that does not
    come back as a JSON object -- an older launcher without the flag, or one
    where ``--json`` is rejected -- it falls back to the prose report and says
    so in the reason, because a degraded answer that does not announce itself is
    worse than no answer.
    """
    proc = _run(["odgui", "--check", "--json"])
    if proc is None:
        return Check(
            code=-1,
            verdict="unavailable",
            reason="`odgui --check` could not be run, so whether this machine can "
                   "create an OpenGL context is unknown",
            source="unavailable",
        )

    payload = None
    text = (proc.stdout or "").strip()
    if text.startswith("{"):
        try:
            loaded = json.loads(text)
        except ValueError:
            loaded = None
        if isinstance(loaded, dict) and loaded.get("verdict"):
            payload = loaded

    if payload is not None:
        return Check(
            code=int(payload.get("exit", proc.returncode)),
            verdict=str(payload["verdict"]),
            reason=_reason_from_json(payload),
            source="json",
            detail=payload,
        )

    # ------------------------------------------------------------------ prose
    # Reached when `--json` is not there. The exit code is still the launcher's
    # documented contract, so the verdict can be derived from it even though the
    # reason cannot.
    old = _run(["odgui", "--check"])
    if old is None:
        return Check(
            code=-1,
            verdict="unavailable",
            reason="`odgui --check` could not be run, so whether this machine can "
                   "create an OpenGL context is unknown",
            source="unavailable",
        )
    code = old.returncode
    verdict = {
        EXIT_OK: VERDICT_OK,
        EXIT_NO_GUI_STACK: VERDICT_NO_GUI_STACK,
    }.get(code, VERDICT_NO_CONTEXT)
    if verdict == VERDICT_OK:
        # Deliberately not the first line of stdout: on success that is the
        # banner, and a title is not a reason. Saying so is more useful than
        # quoting it.
        reason = (
            "the GUI stack imported and Qt created a context (this launcher has "
            "no --json, so the report carries no machine-readable reason)"
        )
    else:
        report = (old.stderr or old.stdout or "").strip().splitlines()
        reason = (
            (report[0].strip() if report else "no output")
            + " (this launcher has no --json, so this is the first line of its "
            "prose report rather than a named field)"
        )
    return Check(code=code, verdict=verdict, reason=reason, source="prose")


# --------------------------------------------------------------------- audit

#: The files that must converge on this helper. `workbench_interaction_check.py`
#: is deliberately **not** here: it carries a fifth copy of the old wrapper, it
#: is owned elsewhere, and it is reported as a known exception rather than
#: silently ignored -- so when its owner converges it, this audit says so
#: instead of quietly passing over it forever.
CONSUMERS = (
    "workbench_smoke.py",
    "odgui_launch_check.py",
    "viewport_framing_check.py",
    "contacts_screenshot.py",
    "pockets_screenshot.py",
)

#: Named so the audit's own output is honest about what it did not cover.
KNOWN_ELSEWHERE = ("workbench_interaction_check.py",)

HELPER_MODULE = "_gui_check"


def _helper_calls_are_live(text: str) -> tuple[int, list[str]]:
    """Count calls to `odgui_check` that can actually run, and say why not.

    A textual `text.count("odgui_check(")` is not enough, and finding that out
    cost a mutation: writing `answer = odgui_check() if False else None` leaves
    the call sitting in the file, so a count-based audit reports the consumer as
    converged while the question is never asked. A check that cannot tell a live
    call from a dead one is the same defect as a check that cannot fail, so this
    parses the file and rejects a call that is

    * in a branch whose condition is a literal false (`if False:`, `x if False
      else y`), which is how a call gets neutered without being deleted, or
    * a bare expression statement, i.e. its answer is thrown away.

    Returns ``(live count, reasons)``; a non-empty ``reasons`` fails the audit.
    """
    import ast

    dead: list[str] = []

    class Walker(ast.NodeVisitor):
        def __init__(self) -> None:
            self.false_depth = 0
            self.live = 0

        def _is_literal_false(self, node: ast.AST) -> bool:
            return isinstance(node, ast.Constant) and not bool(node.value)

        def visit_If(self, node: ast.If) -> None:
            if self._is_literal_false(node.test):
                self.false_depth += 1
                for child in node.body:
                    self.visit(child)
                self.false_depth -= 1
                for child in node.orelse:
                    self.visit(child)
                return
            self.generic_visit(node)

        def visit_IfExp(self, node: ast.IfExp) -> None:
            if self._is_literal_false(node.test):
                # The call is in the untaken arm; visit the taken one only.
                self.visit(node.orelse)
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "odgui_check":
                if self.false_depth:
                    dead.append(
                        "a call to odgui_check() sits in a branch that can never "
                        "run, so the question is never asked"
                    )
                else:
                    self.live += 1
            self.generic_visit(node)

        def visit_Expr(self, node: ast.Expr) -> None:
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "odgui_check"
            ):
                dead.append(
                    "a call to odgui_check() is made as a bare statement, so its "
                    "answer is discarded"
                )
                return
            self.generic_visit(node)

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return 0, [f"the file does not parse ({exc})"]

    walker = Walker()
    walker.visit(tree)
    return walker.live, dead


def audit() -> int:
    """Check that every consumer imports this helper and calls it for real.

    Exists because convergence is not a property that maintains itself. Files
    that each *could* ask `odgui --check` will drift back to private copies the
    first time someone edits one of them under deadline, and the symptom is
    invisible: each file still works, still prints a reason, and still exits the
    same way. A deleted call site is the same failure with one fewer question
    asked, so this checks for the call and not only the import -- and, because a
    text search cannot tell a live call from a disabled one, it parses the file.
    """
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    problems: list[str] = []
    lines: list[str] = []

    for name in CONSUMERS:
        path = here / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        imports = f"from {HELPER_MODULE} import" in text or (
            f"import {HELPER_MODULE}" in text
        )
        live, dead = _helper_calls_are_live(text)
        own_def = "def _odgui_check" in text or "def odgui_check" in text
        notes = []
        if own_def:
            problems.append(
                f"{name}: defines its own odgui_check -- that is the copy this "
                f"file exists to delete"
            )
            notes.append("OWN COPY")
        if not imports:
            problems.append(f"{name}: does not import {HELPER_MODULE}")
            notes.append("NO IMPORT")
        if live == 0:
            problems.append(
                f"{name}: no live call to odgui_check(), so its no-context "
                f"reason is gone"
            )
            notes.append("NO LIVE CALL")
        else:
            notes.append(f"{live} live call(s)")
        for reason in dead:
            problems.append(f"{name}: {reason}")
            notes.append("DEAD CALL")
        lines.append(f"  {name:<28} {', '.join(notes)}")

    print("gui check convergence")
    for line in lines:
        print(line)
    for name in KNOWN_ELSEWHERE:
        print(f"  {name:<28} KNOWN EXCEPTION -- has its own copy, owned elsewhere")
    print()
    if problems:
        for problem in problems:
            print(f"  FAIL {problem}")
        print(f"\nRESULT: FAIL — {len(problems)} convergence problem(s)")
        return 1
    print(f"RESULT: PASS — all {len(CONSUMERS)} consumers use {HELPER_MODULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(audit())
