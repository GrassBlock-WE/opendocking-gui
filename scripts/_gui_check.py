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
* The leading underscore marks it as a helper rather than a check, because
  `CONTRIBUTING.md` and `.github/workflows/ci.yml` treat a `scripts/*.py` whose
  name reads like a gate as a thing to *run*. A file called `gui_check.py` would
  be mistaken for one; `_gui_check.py` cannot be.

  That enumeration is no longer "every file in `scripts/`", and the audit below
  does not depend on it being so -- `audit()` reads nothing but the five
  `CONSUMERS` below and never opens either file. The claim is kept because the
  naming still matters, not because anything here checks it: CI now runs the
  audit scripts from a separate job with no display, and the workbench job runs
  the windowed ones, so a file is run for being a gate in one job or the other.
  An earlier version of this sentence said both files "enumerate `scripts/*.py`
  as things to run", which stopped being true when the workbench job was
  restructured, and a rationale that has quietly stopped being true is the kind
  of thing that later gets relied on.

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
import sys
from dataclasses import dataclass, field

#: Generous, because the honest answer costs real time on a machine that cannot
#: supply a context: Qt blocks *inside* `show()` for seconds before it gives up,
#: and that wait belongs to Qt. Measured 16.7 s end to end on this machine.
TIMEOUT = 120

VERDICT_OK = "ok"
VERDICT_NO_CONTEXT = "no-context"
VERDICT_NO_WIDGET = "no-widget"
VERDICT_NO_GUI_STACK = "no-gui-stack"

#: `odgui --check` exit codes, unchanged from the launcher's own contract.
EXIT_OK = 0
EXIT_NO_GUI_STACK = 3
EXIT_NO_CONTEXT = 4
#: Exit 5: OpenGL works on this machine, but the **product's own viewport**
#: (`app.Viewport`) did not come up, so the viewer window would still open with
#: nothing rendered in it. Its own code because the remedy for 4 ("get a machine
#: with a GPU") is wrong for 5 -- the driver is fine. Mapping it onto 4 is
#: exactly how this file's consumers ended up telling users to go and buy
#: hardware they did not need.
#:
#: The *wording* of 5 changed when the launcher's widget stage stopped using a
#: bare stand-in widget. It used to mean "Qt cannot realise an OpenGL widget",
#: which was a claim about the stand-in and not about the viewer, and on a
#: machine where the viewer rendered it told users their viewer was broken. The
#: code and the `no-widget` spelling are deliberately unchanged -- a script
#: branching on either keeps working -- but the sentence below now describes
#: what the launcher actually tested.
EXIT_NO_WIDGET = 5

#: Exit code -> verdict, used on the prose fallback path where there is no object
#: to read a `verdict` from. Explicit rather than a "anything else is
#: no-context" default: a new exit code must not be silently absorbed into the
#: wrong verdict, which is the failure mode this table exists to prevent.
_EXIT_TO_VERDICT = {
    EXIT_OK: VERDICT_OK,
    EXIT_NO_GUI_STACK: VERDICT_NO_GUI_STACK,
    EXIT_NO_CONTEXT: VERDICT_NO_CONTEXT,
    EXIT_NO_WIDGET: VERDICT_NO_WIDGET,
}


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

    `no-widget` is the one verdict whose `problem` field, quoted bare, would
    say the wrong thing. The problem is the *viewport stage's*, while OpenGL on
    that machine works -- so a consumer that took the field at face value would
    tell a user their machine has no OpenGL when it demonstrably does, and send
    them off to buy a GPU. The raw stage's facts are folded in so the sentence
    survives being passed around without its payload.

    The subject is named as the workbench's own viewport rather than as "an
    OpenGL widget", because that is what the launcher exercises now, and a
    sentence that said the latter would be reporting the probe rather than the
    product.
    """
    problem = payload.get("problem")
    if payload.get("verdict") == VERDICT_NO_WIDGET:
        raw = payload.get("raw") or {}
        detail = (
            f"OpenGL works (raw context GL {raw.get('gl', '?')}, function table "
            f"{'usable' if raw.get('functions') else 'UNUSABLE'}) but the "
            f"workbench's own viewport did not come up"
        )
        return f"{detail}: {problem}" if problem else detail
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
    verdict = _EXIT_TO_VERDICT.get(code)
    if verdict is None:
        # An exit code this table has never heard of. Naming it is strictly
        # better than folding it into `no-context`: the caller is told the
        # answer is unrecognised, and `can_make_context` is False so nobody acts
        # on it as success.
        return Check(
            code=code,
            verdict=f"unknown-exit-{code}",
            reason=(
                f"`odgui --check` exited {code}, which is not a code this file "
                f"knows; treat the answer as unknown rather than as a verdict"
            ),
            source="prose",
        )
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


# ----------------------------------------------------------------- copy audit

#: The payload fields compared against the tree, and where this tree keeps the
#: same file. The paths are relative to the `opendocking` package's parent, so
#: one table serves both layouts -- this checkout's `dock-py/python` and an
#: installed `site-packages` -- and adding a third copy of the package does not
#: mean writing the layout out again.
PROVENANCE_FILES = (
    ("launcher", "opendocking/workbench/launcher.py"),
    ("app", "opendocking/workbench/app.py"),
)


@dataclass(frozen=True)
class CopyAudit:
    """What comparing the loaded copy against this tree established.

    `comparable` is False when nothing could be compared, so a caller can SKIP
    with `why` instead of passing a run that measured nothing -- the same
    three-way discipline `pixel_tag` exists for, applied to a different thing
    that can also fail to answer.
    """

    comparable: bool
    divergences: tuple[str, ...]
    compared: tuple[str, ...]
    why: str = ""


def audit_probe_copy(detail: dict, source_root) -> CopyAudit:
    """Compare the files `odgui --check` loaded with this checkout's copies.

    `detail` is a `Check.detail` -- the `--check --json` payload, whose
    `provenance` block names the resolved files -- and `source_root` is the
    `dock-py/python` directory of the tree the caller lives in.

    **This is the half that cannot live in the probe.** The measurement belongs
    to the process, because only it knows which file it imported; the comparison
    belongs to the check, because only it knows where the tree being edited is.
    A probe that tried to compare would be guessing at a path it cannot see, and
    a check that tried to measure would be guessing at a module it never
    imported. The split is why this is two functions in two files and not one
    clever function in either.

    What it buys is the difference between a *confident wrong answer* and a
    named one. A stale installed copy used to make `odgui --check` report the
    verdict of a launcher that no longer existed in the tree, with no evidence in
    its output at all -- the divergence was only ever visible as two paths in a
    traceback, if a person noticed. Here the two are compared byte for byte and
    a difference comes back as a sentence that says *which file, which digest,
    against which*, which is a cause and not a symptom.
    """
    import hashlib
    from pathlib import Path

    prov = dict((detail or {}).get("provenance") or {})
    if not prov:
        return CopyAudit(
            False,
            (),
            (),
            "the launcher reported no `provenance` block, so the copy that "
            "answered is unknown (an older launcher, or one from before the "
            "field existed) -- which is not evidence that the copies agree",
        )

    root = Path(source_root)
    compared: list[str] = []
    divergences: list[str] = []
    unknown: list[str] = []
    for key, relative in PROVENANCE_FILES:
        facts = prov.get(key) or {}
        name = str(facts.get("module") or relative)
        reported = facts.get("sha256")
        if not reported:
            unknown.append(
                f"{name}: the copy that answered reported no digest, so it "
                f"cannot be compared"
            )
            continue
        reference = root / relative
        if not reference.is_file():
            unknown.append(f"{name}: this tree has no {relative} to compare it with")
            continue
        # Read the reference's bytes once and hash them once. A digest computed
        # twice for a message is a digest that can disagree with itself.
        reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
        if reference_sha == reported:
            compared.append(f"{name} is byte-identical to {relative}")
        else:
            divergences.append(
                f"{name} differs from {relative} -- the copy that answered is "
                f"{facts.get('path')} (sha256 {str(reported)[:12]}) and this "
                f"tree's is {reference} (sha256 {reference_sha[:12]})"
            )

    why = "; ".join(unknown)
    if compared or divergences:
        return CopyAudit(True, tuple(divergences), tuple(compared), why)
    return CopyAudit(
        False,
        (),
        (),
        why or "no file named in the payload could be compared against the tree",
    )


#: The child for `resolve_module_copy`. It reports the interpreter it ran under
#: as well as the module's file, because the answer is only worth having if the
#: thing that produced it is the same interpreter this check is running.
_RESOLVE_CHILD = """
import importlib, json, sys
module = importlib.import_module(sys.argv[1])
print(json.dumps({"executable": sys.executable,
                  "file": getattr(module, "__file__", None)}))
"""


def resolve_module_copy(module_name: str) -> dict:
    """Which file `module_name` resolves to, measured in a fresh interpreter.

    The fallback for a launcher too old to report its own provenance, which is
    the state this mechanism most has to survive: the stale copy that made
    `odgui_launch_check` fail here predates the field entirely, so a comparison
    that only reads the payload would decline to compare exactly when there is
    something to say. The child runs the same interpreter as this process, in
    this process's environment, so it sees the `PYTHONPATH` a console script
    would inherit.

    Two guards, and both of them can only make the answer *unavailable* rather
    than wrong -- which is the direction to fail in, because the failure this
    exists to prevent is a confident wrong answer:

    * the child names the interpreter it ran under, and its answer is discarded
      unless that is this process's own executable, so a copy belonging to a
      different interpreter is never described using this one's `sys.path`;
    * a child that fails to import the module, or answers with no file, yields
      a `problem` and no path, rather than a path inferred from the module name.
    """
    proc = _run([sys.executable, "-c", _RESOLVE_CHILD, module_name])
    if proc is None:
        return {
            "module": module_name,
            "path": None,
            "problem": "a fresh interpreter could not be run to ask",
        }
    try:
        answer = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        answer = None
    if not isinstance(answer, dict) or not answer.get("file"):
        return {
            "module": module_name,
            "path": None,
            "problem": f"the asking interpreter could not import {module_name} "
            f"(exit {proc.returncode})",
        }
    from pathlib import Path

    try:
        same = Path(str(answer.get("executable"))).resolve() == Path(
            sys.executable
        ).resolve()
    except OSError:
        same = False
    if not same:
        return {
            "module": module_name,
            "path": None,
            "problem": f"the copy would have to be named by "
            f"{answer.get('executable')!r}, which is not the interpreter this "
            f"check runs under, so it is left unmeasured rather than guessed",
        }

    import hashlib

    path = Path(str(answer["file"]))
    facts = {
        "module": module_name,
        "path": str(path.resolve()),
        "measured_in": "a fresh interpreter in this check's environment (the "
        "probe itself reported no provenance)",
    }
    try:
        data = path.read_bytes()
    except OSError as exc:
        facts["problem"] = f"the file could not be read ({exc})"
        return facts
    facts["bytes"] = len(data)
    facts["sha256"] = hashlib.sha256(data).hexdigest()
    return facts


# ----------------------------------------------------------------- pixel guard

#: The three-way answer every pixel-reading GUI check has to give, in one place.
#:
#: A pixel check asks about something the *environment* may be unable to show:
#: whether the viewer drew anything at all. So there are three answers, not two,
#: and the middle one is the one that gets forgotten.
#:
#: * the framebuffer cannot answer (a headless runner's uniformly coloured grab,
#:   or a null image) -- **SKIP**: nothing was measured and nothing is claimed;
#: * it can answer and the thing asked is true -- **PASS**;
#: * it can answer and the thing asked is false -- **FAIL**.
#:
#: Collapsing the first into the third is the defect this exists to prevent, and
#: it is not a small one: a suite that reports "nothing was drawn at all" for
#: three frames on a runner that cannot draw has told the reader their viewer is
#: broken when the only true statement is that the runner drew nothing readable.
#: `workbench_interaction_check.py` hit exactly that and its owner fixed it there;
#: `viewport_framing_check.py` hit it independently, in the same CI run, which is
#: what makes this a shared decision rather than two private ones.
#:
#: Deliberately free of Qt, numpy and any display, so this file stays importable
#: by the static audit CI runs before anything needs a screen.
PIXEL_SKIP = "SKIP"
PIXEL_PASS = "PASS"
PIXEL_FAIL = "FAIL"


def pixel_tag(framebuffer_can_answer: bool, condition: bool) -> str:
    """Which of the three answers this measurement earns.

    `framebuffer_can_answer` is the machine's answer, and it must come from the
    **product's own frame** -- never from `odgui --check` and never from
    `GL_OK`. Those two disagree with the frame, and the disagreement is measured
    rather than assumed: on the machine this was written on, the launcher's probe
    reports no context while the workbench's `QOpenGLWidget` renders and its
    framebuffer reads back 723 226 pixels. Gating a pixel check on either of
    them skips it on machines that render perfectly well.

    `condition` is the thing actually being asserted, and is only consulted when
    the framebuffer can answer. A blank frame is not evidence about `condition`
    in either direction, which is why the first branch comes first and returns
    without looking at it.
    """
    if not framebuffer_can_answer:
        return PIXEL_SKIP
    return PIXEL_PASS if condition else PIXEL_FAIL


def verify_pixel_tag() -> tuple[bool, str]:
    """Reverse-verify `pixel_tag`, so a guard that cannot fail is visible.

    All three states are probed, because a guard that skipped unconditionally
    would be indistinguishable from a working one: the first probe is the state
    this function exists for, the second proves a good frame is still measured,
    and the third proves a readable-but-empty frame is still a failure.

    Returns `(held, reason)`.
    """
    got = tuple(
        pixel_tag(answerable, condition)
        for answerable, condition in ((False, False), (True, True), (True, False))
    )
    want = (PIXEL_SKIP, PIXEL_PASS, PIXEL_FAIL)
    return (
        got == want,
        f"blank framebuffer -> {got[0]}, good render -> {got[1]}, "
        f"empty render -> {got[2]} (want {'/'.join(want)})",
    )


#: Files that must converge on `pixel_tag` rather than carry their own copy.
#: `workbench_interaction_check.py` has its own `pixel_check`/`PIXELS_OK` pair,
#: which is the correct three-way behaviour implemented locally. It is not
#: listed here because it is owned elsewhere and is mid-convergence by another
#: agent; when its owner points it at `pixel_tag`, this list grows by one and the
#: two implementations collapse to one.
PIXEL_GUARD_CONSUMERS = ("viewport_framing_check.py",)


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
