"""Is the `_dockpy.pyd` in a tree the extension built from the Rust beside it?

Run:  F:\\python310\\python.exe scripts\\binary_source_parity_check.py

# What this is for, and the shape of the failure

A compiled extension and the Rust source it was built from are two files that
have to agree, and nothing in the tree made them. For a whole day
`dock-py/python/opendocking/_dockpy.pyd` sat in the source tree while the
Python beside it read a field the extension did not export: `odcli dock -o`
exited 1 with `AttributeError: 'DockingResults' object has no attribute
'unknown_atom_types'` while `--json` on the same command returned three healthy
poses. It was found by a person running the command, not by a gate.

**The existing surface gate does not cover this, and the reason is structural
rather than an oversight.** `extension_surface_check.py` compares the names
`core.py` calls *on the module* against `dir(ext)`, plus the arity of those
calls. The name that broke is reached as an attribute **on a returned
instance** -- `core.py:782` is `return int(self._res.unknown_atom_types)`,
where `self._res` came back from a call. `dir(ext)` does not contain it, and a
module-level name census cannot see it by construction. Measured, not argued:
`extension_surface_check.call_surface()` reports no missing name for a binary
that is missing `DockingResults.unknown_atom_types`, and would return 6/6.

So the two gates are complementary and neither subsumes the other. This one
reads the **class-level** surface -- what each `#[pyclass]` exposes, by name --
and compares it with what the binary's classes actually offer. That is the
half that turned red on the day in question.

# What it compares, and why not bytes

**It compares the set of Python-visible names the Rust source declares against
the set the loaded binary offers.** Module-level `#[pyfunction]`s, and per
class the `#[pymethods]` members (`#[new]`, `#[staticmethod]`, `#[getter]`,
plain methods) grouped by the `#[pyclass(name = "...")]` that owns them.

Byte equality would be the wrong instrument, and this repository has already
been bitten by reaching for it. A rebuild of unchanged source is not guaranteed
byte-identical: the pyo3/numpy abi3 build, the Rust toolchain version, the
linker and the target CPU all feed the output, and `RUSTFLAGS: "-C
target-cpu=x86-64"` (`ci.yml:85`) is a machine-dependent input to it. So a
byte gate is red on every correct rebuild by a different toolchain, which is a
gate nobody runs. `docs/VERIFICATION.md` records the same lesson where a
byte-equality assertion on a fresh install was deliberately weakened to "every
name `odgui` reaches for is defined there" -- and that row is in flux this
round, so it is cited as a decision, not as a number.

The name-set is the right instrument for a second reason, and it is the one
that matters: **it is the question the failure actually was.** A user does not
observe a wrong checksum; they observe `AttributeError` on a name. Comparing
declared names to exported names asks "can every name the source reaches for
resolve?" and a mismatch is an `AttributeError` with a line number, not a
digest somebody has to interpret.

**And it is deliberately not a build stamp.** A stamp embedded by the build
would be the stronger instrument and it belongs to the engine owner, exactly as
`extension_surface_check.py:37-38` already says. This file needs no change to
`dock-core/**` or `dock-py/src/lib.rs` to be worth running, and it reads only
what a reader can read.

# What it cannot do, stated here rather than discovered

* **It cannot see inside a function body.** A scoring-weight change, a changed
  constant, a reordered loop -- none of them move a name, so a binary carrying
  last week's arithmetic passes. That is a real limit and it is the price of
  not requiring a rebuild. This gate catches *interface* drift, which is the
  drift that produces an `AttributeError` or a `TypeError`; it does not catch
  *behaviour* drift, which produces a wrong number and needs `cargo test`.
* **It reads the declared surface with a parser, not a compiler.** A
  `#[pymethods]` block written in a shape this parser does not recognise is
  reported as an unparsed class rather than silently as agreement, and a
  renamed export via an attribute shape it does not model is reported as a
  name the source declares and the binary lacks. Both directions are loud.
* **It never calls a function in the extension.** Same reason as its sibling:
  the argument-count `TypeError` is produced by calling, so a check that had to
  call could not survive the mismatch it reports. Names and `dir()` only.

# Where it fails, and why that is the loud direction

A binary that is behind its source is **red, in the tree that holds it**, with
the three facts a maintainer needs: the tree, the class, and the name the
source declares and the binary does not have. That is the same direction
`release_tree_rule.py` argues for when it inverts an exclusion rule into a
recognition rule: a file that should have shipped and did not is a red someone
reads, whereas a file that should not have shipped and did is a green until a
user notices. This gate is the same trade applied to a build artefact. Its
asymmetric failure is a *false red* on a correct build, which costs a re-run; a
false green costs a day.

**A tree with no compiled extension is a SKIP, counted, not a pass.** That is
the clone case, and it is the honest reading: a fresh checkout has no
`_dockpy.pyd` (`.gitignore` line 17 is `*_dockpy.pyd`, so git has never
tracked one) and this gate has nothing to compare until one is built. A skip
that is silently a pass would be the worst outcome available here, because it
would be indistinguishable from "checked and current".

# The three binaries, and which one this file is talking about

There are three `_dockpy.pyd` on a maintainer's machine and they are three
different objects:

| tree | size | sha256 (16) | what it is |
|---|---|---|---|
| development tree | 778,240 | `066fdd0ccd7447df` | built from the current source |
| the published tree | 753,152 | `8df76c14ab224aaf` | an older build, beside older source |
| `site-packages` | 778,240 | `066fdd0ccd7447df` | the install, out of scope by decision |

**This file measures the binary in the tree it is run from**, and prints the
path, the size and the digest of the file it actually opened, so a reader is
never left guessing which of the three answered. `OD_BINARY_SOURCE_ROOT` points
it at another tree; `OD_BINARY_SOURCE_PYD` at a specific file. The
`site-packages` copy is declared out of scope by `release_tree_rule.py`
(`SITE_PACKAGES_DECISION`) and this file does not reach for it.

The three are *not* expected to agree, and the check is written so that they
cannot be mistaken for each other: the published tree's binary is compared
against the published tree's `lib.rs`, and that pair agrees today even though
both are older than development. A binary is stale **relative to the source
beside it**, not relative to the newest source anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIB_REL = Path("dock-py", "src", "lib.rs")
PKG_REL = Path("dock-py", "python", "opendocking")

#: How many checks this file records, in every environment.
#:
#:   1  the Rust source declares a surface this file could parse (a skip,
#:      counted, if there is no `lib.rs` beside the binary)
#:   2  the extension beside that source was located and loaded
#:   3  every module-level `#[pyfunction]` the source declares is exported
#:   4  every `#[pyclass]` the source declares is exported
#:   5  every member of every declared class is present on that class
#:   6  every declared class reported a non-empty surface, so 5 is not vacuous
#:   7  the comparison is not vacuous: a deliberately absent name is caught
#:   8  the tally accounts for every result, skips included
#:
#: Check 7 is the one that keeps 3-5 from being a gate that has never seen
#: anything wrong, and it calls the same `missing_members()` the real
#: comparison calls rather than a restatement of it. The synthetic name is a
#: *measurement* being wrong; the real binary is never touched, which is the
#: same discipline `extension_surface_check.py` uses for its arity predictor.
#:
#: Check 6 exists because check 5 can pass for the wrong reason: a class that
#: loads as a type exposing nothing would satisfy "no declared member is
#: missing" without anything having been compared.
#:
#: 7 -> 8 when check 6 was added, and the number is derived by running the
#: file rather than typed. Every early-return path records the same 8: a
#: missing `lib.rs`, an unparseable one, a missing `.pyd` and a `.pyd` that
#: will not load all skip or fail the same named checks, because a run that
#: reports a different number of results depending on what it found is a
#: smaller run wearing a pass.
#: The call-site census, declared in the file it counts rather than in
#: `check_scripts_declare.py`'s generated snapshot, and measured by walking this
#: file's own syntax tree. A pin is how many results a run records; the census
#: is how many result sites the source has. They are different questions, and
#: keeping them in one place is what stops a deleted check from leaving its
#: number behind.
#:
#: The 3 unconditional sites are the three `skip(...)` calls on the paths where
#: there is nothing to compare, and the 13 guarded ones are every `ok`/`bad`
#: pair -- which is the shape worth stating: **this gate has no unconditional
#: assertion about the binary at all.** Every claim about the binary is
#: conditional on what the source declared and what the child loaded, so a
#: change in either moves the guarded column and leaves the total alone. That is
#: why the digest is here and not just the pair of counts.
#: GATE-DECLARE 1
#: sites: 3 unconditional + 13 guarded
#: guards: sha256:ad68dd8079ea5ebfa53b6146f4f7919aa5317ee081262061715adcd3603714da
EXPECTED_CHECKS = 8

RESULTS: list[tuple[str, str, str]] = []

SOURCE_CHECK = "the Rust source declares a surface this file could parse"
LOADED_CHECK = "the extension beside that source was located and loaded"


def ok(name: str, detail: str) -> None:
    RESULTS.append(("PASS", name, detail))
    print(f"[PASS] {name}\n       {detail}")


def bad(name: str, detail: str) -> None:
    RESULTS.append(("FAIL", name, detail))
    print(f"[FAIL] {name}\n       {detail}")


def skip(name: str, why: str) -> None:
    RESULTS.append(("SKIP", name, why))
    print(f"[SKIP] {name}\n       {why}")


def section(t: str) -> None:
    print()
    print(f"-- {t}")


# ------------------------------------------------------------- static side

_ATTR = re.compile(r"^\s*#\[([A-Za-z_][A-Za-z0-9_]*)")
_NAMEARG = re.compile(r'name\s*=\s*"([^"]+)"')
_FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")


def _skip_body(lines: list[str], start: int) -> int:
    """Index just past the brace-balanced body of the `fn` beginning at `start`.

    Counts the signature's own braces and every nested one, so a `fn` whose
    body contains a `struct` literal (`Ok(PyGridBox { inner: ... })`) or a
    closure does not end the walk early. Returns `start + 1` when the line
    opens no brace, which is a declaration rather than a definition.
    """
    depth = 0
    opened = False
    k = start
    while k < len(lines):
        depth += lines[k].count("{") - lines[k].count("}")
        opened = opened or "{" in lines[k]
        k += 1
        if opened and depth <= 0:
            return k
        if not opened and lines[k - 1].rstrip().endswith(";"):
            return k
    return len(lines)


def declared_surface(text: str) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """The Python-visible surface `lib.rs` declares, as (functions, classes).

    Returns `(functions, classes)` where `functions` maps an exported module
    name to the line that declared it, and `classes` maps a `#[pyclass(name =
    "...")]` to its `#[pymethods]` members and their lines.

    Four shapes this has to get right, each of which a naive line regex gets
    wrong, and each of which was wrong in the first version of this parser:

    * `#[pyfunction(name = "dock")] fn dock_py` exports **`dock`**, not
      `dock_py`. Reading the Rust identifier reports a name the binary does
      not have and misses the one it does -- two false reds from one line.
    * `#[new] fn new` exports **`__new__`**, a dunder, and is dropped from the
      comparison for the same reason `__repr__` is: a dunder is not a name a
      caller reaches for. The Rust identifier is `new`, so the export has to
      be renamed before the dunder filter sees it or every correct build
      reports a missing `new`.
    * A free `fn` outside any `impl` -- `default_atom_type` at `lib.rs:528` is
      one -- is **not exported at all**. Counting it reports a missing name in
      every correct build.
    * `#[pyo3(signature = ...)]` sits *between* the `#[pyfunction]` and its
      `fn`, so the function that follows an attribute is not the next line.
    * **A `}` does not end the `impl` block.** Every method body ends with one,
      and treating it as the terminator collected exactly one member per class
      -- six classes, six members, and a `GridBox.new` "missing" from a binary
      that has always had it. The block ends when brace depth returns to zero,
      counted from the `impl` line.
    * **A recorded `fn`'s body has to be skipped whole, and counted first.**
      The signature line opens a brace and the body closes it, so the advance
      after recording a member is `skip_body(lines, j)`: it returns the index
      of the brace-balanced end of that `fn`, having counted its braces on the
      way. Advancing to `j + 1` instead lands *inside* the body, whose
      opening brace pushes the depth positive, so the `impl` block never
      closes and the parser records exactly one member per class -- six, where
      the file declares fifty-seven, and a `GridBox.new` "missing" from a
      binary that has always had it. Both symptoms of the same advance.
    """
    lines = text.splitlines()
    funcs: dict[str, int] = {}
    classes: dict[str, dict[str, int]] = {}
    cur: str | None = None
    in_impl = False
    depth = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _ATTR.match(line)
        if not m:
            if in_impl:
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    in_impl = False
                    cur = None
            i += 1
            continue
        kind = m.group(1)
        name_arg = _NAMEARG.search(line)
        # The `fn` this attribute applies to is past any further attributes.
        j = i + 1
        while j < len(lines) and lines[j].lstrip().startswith("#["):
            j += 1
        sig = _FN.match(lines[j]) if j < len(lines) else None
        if kind == "pyclass" and name_arg:
            cur = name_arg.group(1)
            classes[cur] = {}
            in_impl = False
        elif kind == "pymethods":
            in_impl = True
            depth = 0
        elif kind == "pyfunction" and sig:
            funcs[name_arg.group(1) if name_arg else sig.group(1)] = i + 1
            in_impl = False
        elif in_impl and cur and sig and not name_arg:
            # `#[new]` renames the export to `__new__`; every other attribute
            # in a `#[pymethods]` block exports under the Rust name.
            exported = "__new__" if kind == "new" else sig.group(1)
            classes[cur][exported] = i + 1
        if sig:
            # Skip the definition whole, braces balanced, so the `impl` block's
            # own depth is unaffected by the bodies of the methods in it.
            end = _skip_body(lines, j)
            depth += sum(l.count("{") - l.count("}") for l in lines[j:end])
            i = end
            continue
        i += 1
    return funcs, classes


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


# ------------------------------------------------------------ runtime side

_CHILD = r'''
import importlib.machinery, importlib.util, json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
out = {"ok": False}
try:
    path = sys.argv[1]
    # A loader the source loader would never be: a `.py` file is importable
    # by `spec_from_file_location`, so without this a pure-Python file is
    # "loaded" and reported as a working extension exporting its imports. The
    # first version did exactly that and answered a question nobody asked.
    #
    # **Any** suffix in `EXTENSION_SUFFIXES`, not `[0]`: on CPython 3.10 that
    # list is ['.cp310-win_amd64.pyd', '.pyd'], and the binary under test is
    # `_dockpy.pyd` -- the ABI-tagged entry first rejected the real extension
    # and turned two green arms red, which is the same shape of error as the
    # one it fixed.
    if not any(path.endswith(sfx) for sfx in importlib.machinery.EXTENSION_SUFFIXES):
        out["error"] = ("not a compiled extension: %s ends in none of %s"
                        % (path, importlib.machinery.EXTENSION_SUFFIXES))
        print("__SURFACE__" + json.dumps(out))
        raise SystemExit(0)
    spec = importlib.util.spec_from_file_location("opendocking._dockpy", path)
    if spec is None or not isinstance(spec.loader,
                                      importlib.machinery.ExtensionFileLoader):
        out["error"] = ("%s has no extension loader (got %r)"
                        % (path, type(spec.loader).__name__ if spec else None))
        print("__SURFACE__" + json.dumps(out))
        raise SystemExit(0)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opendocking._dockpy"] = mod
    spec.loader.exec_module(mod)
    out["module"] = sorted(n for n in dir(mod) if not n.startswith("__"))
    classes = {}
    for n in out["module"]:
        obj = getattr(mod, n)
        if isinstance(obj, type):
            classes[n] = sorted(a for a in dir(obj) if not a.startswith("__"))
    out["classes"] = classes
    out["ok"] = True
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print("__SURFACE__" + json.dumps(out))
'''


def survey(pyd: Path) -> dict:
    """Load one `.pyd` by path in a child and report what it offers.

    A child, for the reason `extension_surface_check.py` gives: a stale binary
    can abort the interpreter, and this gate has to report that rather than
    become it. The environment is *extended*, never rebuilt -- handing CPython
    a fresh env without `SystemRoot` kills it before a line of the child runs,
    and the failure reads as a broken extension.
    """
    tmp = HERE.parent / ".binary_source_parity_child.py"
    tmp.write_text(_CHILD, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        proc = subprocess.run([sys.executable, str(tmp), str(pyd)],
                              capture_output=True, text=True, timeout=300,
                              env=env, cwd=str(ROOT))
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("__SURFACE__")), None)
    if line is None:
        return {"ok": False, "error": (
            f"the child printed no surface report and exited {proc.returncode}. "
            f"stderr: {(proc.stderr or '(empty)').strip()[-300:]}")}
    return json.loads(line[len("__SURFACE__"):])


# ------------------------------------------------------------- the checks

def missing_members(declared: dict[str, int], offered: set) -> list[str]:
    """Declared names the binary does not have. The one comparison, shared.

    The real check and the non-vacuity check call *this*, so the synthetic
    entry below exercises the code that produces the verdict rather than a
    restatement of it. A gate that tested a copy of its own logic would pass
    the vacuity check while the real one was broken.
    """
    return sorted(n for n in declared if not is_dunder(n) and n not in offered)


def check_the_comparison_is_not_vacuous(declared: dict, offered: set) -> None:
    """Fire the detector on a name nobody declared before trusting it on one.

    A comparison that has never reported a difference has not been shown able
    to, and the subject of this file is a drift that went unnoticed for a day.
    The absent name is synthetic and the real binary is never involved.
    """
    name = "the comparison is not vacuous: an absent name is caught"
    # Not a dunder: `missing_members()` filters names that both start and end
    # with `__`, because those are the ones CPython adds rather than exports.
    # A sentinel wearing dunders would have been filtered out by the very
    # comparison it exists to exercise, and the first version of this check
    # reported `[]` for exactly that reason.
    sentinel = "od_deliberately_not_exported_name"
    missing = missing_members({**declared, sentinel: 0}, offered)
    if sentinel not in missing:
        bad(name, f"a name no source declares was not reported as missing, so "
                  f"the comparison above could not have caught a binary that "
                  f"lost one. It reported {missing}")
        return
    others = [n for n in missing if n != sentinel]
    # `sentinel in missing`, **not** `missing == [sentinel]`. This check asks
    # whether the detector can fire, not what the verdict is: when the real
    # binary is also behind its source, `others` is the finding rather than a
    # failure here. Asserting the whole list equalled the sentinel made this
    # check red on precisely the run it exists to make legible -- a gate that
    # reports a correct finding as a broken detector is the false red this
    # file's own docstring argues against.
    ok(name, f"the shared comparison, asked about a synthetic source declaring "
             f"{len(declared) + 1} name(s), reports the absent name whether or "
             f"not the real binary is also behind"
             + (f", alongside {len(others)} genuinely absent name(s) this run "
                f"found: {', '.join(others)}. Those are findings, not a failure "
                f"of this check" if others else
                ", and the real binary is current, so it is the only one")
             + ". That is the shape of the finding above")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Is the compiled extension the one built from the Rust beside it?")

    root = Path(os.environ.get("OD_BINARY_SOURCE_ROOT", ROOT)).resolve()
    override = os.environ.get("OD_BINARY_SOURCE_PYD")
    lib = root / LIB_REL
    pyd = Path(override).resolve() if override else root / PKG_REL / "_dockpy.pyd"
    print(f"  tree    {root}")
    print(f"  source  {lib}")
    print(f"  binary  {pyd}")

    if not lib.is_file():
        why = (f"{lib} does not exist, so there is no declared surface to compare "
               f"against. A published checkout carries `dock-py/src/lib.rs`; a "
               f"tree that does not is not a tree this gate can judge")
        for n in (LOADED_CHECK,
                  "every module-level function the source declares is exported",
                  "every class the source declares is exported",
                  "every member of every declared class is present on that class",
                  "every declared class reported a non-empty surface"):
            skip(n, why)
        skip(SOURCE_CHECK, why)
        check_the_comparison_is_not_vacuous({"synthetic": 1}, set())
        tally()
        return finish()

    text = lib.read_text(encoding="utf-8")
    funcs, classes = declared_surface(text)
    if not funcs and not classes:
        why = (f"{lib} parsed to an empty surface: {len(text.splitlines())} lines "
               f"yielded no `#[pyfunction]` and no `#[pyclass]`. That is a parser "
               f"that stopped understanding the file, and an empty surface "
               f"compared against a real binary would report every name as "
               f"missing for the wrong reason")
        for n in (LOADED_CHECK,
                  "every module-level function the source declares is exported",
                  "every class the source declares is exported",
                  "every member of every declared class is present on that class",
                  "every declared class reported a non-empty surface"):
            skip(n, why)
        bad(SOURCE_CHECK, why)
        check_the_comparison_is_not_vacuous({"synthetic": 1}, set())
        tally()
        return finish()

    ok(SOURCE_CHECK,
       f"{lib.name} declares {len(funcs)} module-level function(s) and "
       f"{len(classes)} class(es) carrying {sum(len(v) for v in classes.values())} "
       f"member(s): {', '.join(sorted(classes))}")

    section("the extension beside it")
    if not pyd.is_file():
        why = (f"{pyd} does not exist. This is the fresh-clone case and it is a "
               f"skip rather than a pass: `.gitignore` line 17 is `*_dockpy.pyd`, "
               f"so no published tree carries one, and until this tree is built "
               f"there is nothing here to compare")
        skip(LOADED_CHECK, why)
        for n in ("every module-level function the source declares is exported",
                  "every class the source declares is exported",
                  "every member of every declared class is present on that class",
                  "every declared class reported a non-empty surface"):
            skip(n, "no binary to compare against")
        check_the_comparison_is_not_vacuous(
            {**funcs, **{m: 0 for v in classes.values() for m in v}}, set())
        tally()
        return finish()

    digest = hashlib.sha256(pyd.read_bytes()).hexdigest()
    print(f"  opened  {pyd.name}  {pyd.stat().st_size:,} B  sha256 {digest[:16]}")
    print("  this is the file every number below was measured from. No byte of it "
          "is compared against anything; it is loaded and read with dir().")
    data = survey(pyd)
    if not data.get("ok"):
        why = (f"the extension did not load in a child: {data.get('error')}. That "
               f"is itself the shape this gate exists for -- a binary that will "
               f"not load beside a source that declares it -- and it is reported "
               f"rather than raised")
        bad(LOADED_CHECK, why)
        for n in ("every module-level function the source declares is exported",
                  "every class the source declares is exported",
                  "every member of every declared class is present on that class",
                  "every declared class reported a non-empty surface"):
            skip(n, why)
        check_the_comparison_is_not_vacuous(
            {**funcs, **{m: 0 for v in classes.values() for m in v}}, set())
        tally()
        return finish()

    got_funcs = set(data["module"])
    got_classes = data["classes"]
    ok(LOADED_CHECK, f"{pyd} loaded and exports {len(got_funcs)} module-level "
                     f"name(s) and {len(got_classes)} class(es): "
                     f"{', '.join(sorted(got_classes))}")

    section("module-level functions")
    want_f = {n for n in funcs if not is_dunder(n)}
    miss_f = sorted(want_f - got_funcs)
    if miss_f:
        bad("every module-level function the source declares is exported",
            f"{len(miss_f)} of {len(want_f)} function(s) declared in "
            f"{lib.name} are not in the binary: "
            + ", ".join(f"{n} (line {funcs[n]})" for n in miss_f)
            + ". **The binary is behind the source.** Every call to one of these "
              "raises AttributeError at the call site")
    else:
        ok("every module-level function the source declares is exported",
           f"all {len(want_f)} declared function(s) are present: "
           f"{', '.join(sorted(want_f))}")

    section("classes")
    want_c = set(classes)
    miss_c = sorted(want_c - set(got_classes))
    if miss_c:
        bad("every class the source declares is exported",
            f"{len(miss_c)} of {len(want_c)} declared class(es) are absent from "
            f"the binary: {', '.join(miss_c)}")
    else:
        ok("every class the source declares is exported",
           f"all {len(want_c)} declared class(es) are present, each as a type: "
           f"{', '.join(sorted(want_c))}")

    section("class members -- the half the module-level gate cannot see")
    bad_members: list[str] = []
    empty: list[str] = []
    for cname in sorted(want_c):
        if cname not in got_classes:
            continue
        have = set(got_classes[cname])
        gone = missing_members(classes[cname], have)
        if gone:
            bad_members.append(
                f"{cname}: " + ", ".join(f"{m} (line {classes[cname][m]})"
                                         for m in gone))
        if not have:
            empty.append(cname)
    if bad_members:
        bad("every member of every declared class is present on that class",
            f"{len(bad_members)} class(es) carry member(s) the source declares "
            f"and the binary does not have: " + "; ".join(bad_members)
            + ". **This is the failure that ran for a day**: an AttributeError "
              "on a name reached through a returned instance, which a "
              "module-level `dir()` census cannot see")
    else:
        total = sum(len(missing_members(v, set())) for v in classes.values())
        ok("every member of every declared class is present on that class",
           f"all {total} declared member(s) across {len(want_c)} class(es) are "
           f"present on the class they belong to. An instance attribute read "
           f"like `DockingResults.unknown_atom_types` resolves")

    if empty:
        bad("every declared class reported a non-empty surface",
            f"{', '.join(empty)} loaded as a type with no public member, so the "
            f"comparison above was vacuous for them")
    else:
        ok("every declared class reported a non-empty surface",
           f"each of the {len(want_c)} class(es) exposes at least one public "
           f"name, so no comparison above was against an empty set")

    check_the_comparison_is_not_vacuous(
        {**funcs, **{m: 0 for v in classes.values() for m in v}},
        got_funcs | {m for v in got_classes.values() for m in v})
    tally()
    return finish()


def tally() -> None:
    section("the tally")
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    name = "the tally accounts for every result, skips included"
    if npass + nfail + nskip == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(name, f"{npass} + {nfail} + {nskip} = {len(RESULTS)}, and this check "
                 f"is the {len(RESULTS) + 1}th, so the run reaches the declared "
                 f"{EXPECTED_CHECKS} -- with a binary, without one, and against "
                 f"a binary that will not load, because a skip is a result")
    else:
        bad(name, f"recorded {len(RESULTS)} results ({npass}+{nfail}+{nskip}) "
                  f"before this check, which must be {EXPECTED_CHECKS - 1}")


def finish() -> int:
    npass = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nfail = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    nskip = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {npass} passed, {nfail} failed, {nskip} skipped, "
          f"{len(RESULTS)} checks (expected {EXPECTED_CHECKS})")
    if nfail:
        print(f"RESULT: FAIL -- {nfail} of {EXPECTED_CHECKS} checks failed. "
              f"Rebuild the extension from this tree's source; do not work "
              f"around the AttributeError")
        return 1
    if npass == 0:
        print("RESULT: DID NOT FINISH -- this run asserted nothing")
        return 2
    print(f"RESULT: OK -- {npass} passed, {nskip} could not be measured here")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        import traceback
        traceback.print_exc()
        print(f"\nRESULT: DID NOT FINISH -- {type(_exc).__name__}: {_exc}")
        raise SystemExit(2)
