"""Is the extension binary the one this tree's `core.py` was written against?

Run:  F:\\python310\\python.exe scripts\\extension_surface_check.py

# The failure this exists for

`dock-py/python/opendocking/_dockpy.pyd` was rebuilt three times in one
evening -- 753,152 then 753,664 then 755,200 B -- and after each rebuild two
gates died with an argument-count `TypeError` raised from three frames below the
call that caused it. The mechanism is one line of policy, in
`workbench_smoke.py:73-76`:

    if (_SOURCE / "opendocking" / "_dockpy.pyd").exists() or ...:
        sys.path.insert(0, str(_SOURCE))

Put the source tree first *if a compiled extension happens to be sitting in it*
and Python imports that binary beside whatever `core.py` is on disk right now. A
rebuild moves one side; nothing moves the other; the next call into the engine
raises. The rebuild is the engine owner's step and the mismatch is invisible
until something calls a function whose signature changed.

**Nothing in this repository checks for it, and the existing check is not it.**
`core_check.py:1828-1839` derives `#[pyclass(name = "...")]` class names from
`dock-py/src/lib.rs` and compares them with `dir(opendocking._dockpy)`. That is
a real currency check and it covers *classes*; its own comment records that a
pyclass taking its name from its Rust type is not counted, and it says nothing
about **arity**, which is the thing that actually raised. A name the source
reaches for and the binary lacks, and a call site that supplies more arguments
than the binary accepts, are both unmeasured today.

# What this file does with the binary

**It imports it. It never opens it.** No read, no hash, no byte comparison, no
size comparison: the two questions those would answer are a *build* question and
an *origin* question, and this gate is a *contract* question -- does the surface
the source declares match the surface the binary offers. `inspect.signature`
and `dir()` answer it from a loaded module. A build stamp would answer a
different question better, and it belongs to the engine owner.

**And it never calls a function in the extension.** Calling one is how the
argument-count `TypeError` is produced, so a check that had to call the thing it
is checking could not survive the mismatch it exists to report. Arity comes from
the reported signature. A name whose signature cannot be read is reported as
unreadable, which is a fourth answer and is not counted as agreement.

# Where it is meant to run, and why that is stated rather than discovered

**Locally, before the gates that force the source tree to the front.** On CI
this gate has nothing to compare: every job builds the wheel from the same
source in the same job (`ci.yml:149-150`, `249-250`, `590-591`) and no checkout
carries a compiled extension to shadow anything (`cli_check.py:233-238`,
`core_check.py:68-72`), so a step for it would be green by construction. That is
the same trap as a gate with nothing to be wrong about, and the honest thing is
to keep it out of the suite and run it where the risk is. If it *is* wired in,
expect a skip with a stated reason on every runner, and expect it to be trusted
while measuring nothing -- which is why the reason for its absence belongs in
the workflow rather than in a reader's memory.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
PKG_REL = Path("dock-py", "python", "opendocking")
CORE_REL = PKG_REL / "core.py"
EXT_STEMS = ("_dockpy",)

#: How many checks this file records, in every environment:
#:
#:   1  the extension Python would import was located (a skip, counted, if not)
#:   2  every name `core.py` calls is exported by that extension
#:   3  no call site supplies more arguments than the signature accepts
#:   4  every call site's arity was readable, so 2 and 3 are not vacuous
#:   5  the arity predictor is not vacuous: a deliberately wrong arity is caught
#:   6  the tally accounts for every result, skips included
#:
#: Check 5 is the one that keeps this file from being a gate that has never
#: seen anything wrong: the predictor is exercised on a synthetic entry before
#: it is trusted about a real one.
#:
#: The call-site census, declared in the file it counts rather than in
#: `check_scripts_declare.py`'s generated snapshot. It is measured by walking this
#: file's own syntax tree and compared against this block, so there is no path by
#: which the declaration satisfies the auditor on its own. **It was NOT declared
#: anywhere until 2026-10-02**: the auditor's own census loop already walked this
#: file, and reported `NOT DECLARED ANYWHERE` for it on every run, because an
#: `EXPECTED_CHECKS` is a pin and a pin is not a call-site census. Two numbers
#: about a gate, kept in two files, both describing the same gate, neither
#: checking the other -- and the one that was missing was the one that would have
#: noticed a result being dropped.
#:
#: 2 of the 12 sites are unconditional and 10 are guarded. That ratio is the
#: interesting half: most of this gate is a loop over symbols and call sites, so
#: a site that is not reached is a site that is not wrong, and the guards digest
#: is what makes that visible -- the counts alone would not notice a site moving
#: between guard shapes.
#: GATE-DECLARE 1
#: sites: 2 unconditional + 10 guarded
#: guards: sha256:5c0e5f610e103c4de2d46783790ed4bbfc7c8cb87b4a5ed23726ca0e8e4911fb
EXPECTED_CHECKS = 6

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


def section(t: str) -> None:
    print()
    print(f"-- {t}")


# ---------------------------------------------------------------- static side

def extension_aliases(tree: ast.AST) -> set[str]:
    """Local names in `core.py` bound to the extension module.

    `from . import _dockpy as _core` parses as an `ImportFrom` with
    **`module is None`** and `level == 1`: the name lives in `names`, not in
    `module`. Reading `module` and comparing it to `"_dockpy"` -- which is what
    the first version of the surface comparison did -- finds nothing at all, on
    a file with eleven live call sites, and reports a clean match. A detector
    that cannot see the thing it detects is worse than none, because it is
    green.
    """
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported = [a.name for a in n.names]
            if n.module in EXT_STEMS or any(i in EXT_STEMS for i in imported):
                for a in n.names:
                    if a.name in EXT_STEMS or n.module in EXT_STEMS:
                        out.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if Path(a.name).name in EXT_STEMS:
                    out.add(a.asname or a.name.split(".")[0])
    return out


def call_surface(path: Path) -> dict[str, dict]:
    """Per extension entry point: the most arguments any call site supplies.

    `star` records a call site using `*args`/`**kwargs`, where a count cannot
    predict anything -- a fact check 3 must respect rather than average away.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = extension_aliases(tree)
    out: dict[str, dict] = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if not isinstance(f, ast.Attribute) or not isinstance(f.value, ast.Name):
            continue
        if f.value.id not in aliases:
            continue
        supplied = len(n.args) + len(n.keywords)
        star = any(isinstance(a, ast.Starred) for a in n.args) or any(
            k.arg is None for k in n.keywords)
        rec = out.setdefault(f.attr, {"supplied": 0, "line": n.lineno, "star": False})
        if supplied > rec["supplied"]:
            rec["supplied"] = supplied
        rec["star"] = rec["star"] or star
    return out


# --------------------------------------------------------------- runtime side

CHILD = r'''
import inspect, json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
out = {"ok": False}
try:
    import opendocking
    from opendocking import _dockpy as ext
    out["package_from"] = opendocking.__file__
    out["extension_file"] = getattr(ext, "__file__", None)
    out["members"] = {}
    for n in dir(ext):
        if n.startswith("__"):
            continue
        obj = getattr(ext, n)
        rec = {"kind": type(obj).__name__}
        try:
            sig = inspect.signature(obj)
            rec["params"] = len(sig.parameters)
            rec["varargs"] = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
            rec["varkw"] = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
            rec["signature"] = str(sig)
        except (ValueError, TypeError) as exc:
            rec["params"] = None
            rec["why"] = f"{type(exc).__name__}: {exc}"
        out["members"][n] = rec
    out["ok"] = True
except BaseException as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("__SURFACE__" + json.dumps(out))
'''


def survey_extension() -> dict:
    """Import the extension in a child and report what it offers.

    A child, for two reasons. One is containment: a stale binary can abort the
    interpreter, and this gate must report that rather than become it. The other
    is `sys.path`: the whole question is which package wins, and answering it in
    a child means this file's own imports are not part of the answer.

    The environment is *extended*, never rebuilt. Handing CPython a fresh env
    without `SystemRoot` kills it in `_Py_HashRandomization_Init` before a line
    of the child runs, and the failure reads as a broken extension.
    """
    tmp = SCRIPTS.parent / ".extension_surface_child.py"
    tmp.write_text(CHILD, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT / "dock-py" / "python")})
    try:
        proc = subprocess.run([sys.executable, str(tmp)], capture_output=True,
                              text=True, timeout=300, env=env, cwd=str(ROOT))
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    line = next((l for l in proc.stdout.splitlines() if l.startswith("__SURFACE__")), None)
    if line is None:
        return {"ok": False, "error": (
            f"the child printed no surface report and exited {proc.returncode}. "
            f"stderr: {(proc.stderr or '(empty)').strip()[-300:]}")}
    return json.loads(line[len("__SURFACE__"):])


# ------------------------------------------------------------------- verdict

def over_supplied(surface: dict, members: dict) -> tuple[list, list, list]:
    """The three answers this comparison can give. Never a default of agreement.

    Returns `(too_many, unreadable, unpredictable)`: call sites that provably
    supply more than the signature accepts, call sites whose arity could not be
    read, and call sites that use `*args`/`**kwargs` where no count can predict
    a raise. The second and third are not passes and are not merged into one.
    """
    too_many, unreadable, unpredictable = [], [], []
    for name in sorted(surface):
        rec = surface[name]
        if name not in members:
            continue
        got = members[name]
        if got.get("params") is None:
            unreadable.append((name, got.get("why", "no signature reported")))
            continue
        if rec["star"] or got.get("varargs") or got.get("varkw"):
            unpredictable.append((name, rec["supplied"], got["params"]))
            continue
        if rec["supplied"] > got["params"]:
            too_many.append((name, rec["supplied"], got["params"], got.get("signature")))
    return too_many, unreadable, unpredictable


def check_the_predictor_is_not_vacuous() -> None:
    """Exercise the predictor on a synthetic entry before trusting it on a real one.

    A detector that has never fired has not been shown able to fire, and this
    file's subject is a failure that has bitten three times. The synthetic entry
    is a *measurement* being wrong, never the binary being touched: it stands in
    for a rebuilt extension whose `dock` takes fewer arguments than the source
    passes.
    """
    name = "the arity predictor fires on a deliberately mismatched arity"
    synthetic = {"dock": {"supplied": 10, "line": 0, "star": False}}
    members = {"dock": {"params": 8, "varargs": False, "varkw": False,
                        "signature": "(a, b, c, d, e, f, g, h)"}}
    too_many, unreadable, unpredictable = over_supplied(synthetic, members)
    if not too_many or unreadable or unpredictable:
        bad(name, f"a source supplying 10 arguments to a binary declaring 8 "
                  f"produced {too_many}, so the comparison this file exists to "
                  f"make would not have caught the mismatch it was written for")
        return
    n, supplied, params, sig = too_many[0]
    ok(name, f"a synthetic call site supplying {supplied} arguments to an entry "
             f"declaring {params} is reported as {n}: {supplied} > {params} "
             f"({sig}). The real surface below is only meaningful because this "
             f"fired first")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Is the extension binary the one this tree's core.py was written against?")
    print()

    core = ROOT / CORE_REL
    surface = call_surface(core)

    section("the surface this tree's source declares")
    print(f"  {core.relative_to(ROOT)} calls {len(surface)} distinct extension "
          f"entry point(s)")
    for name in sorted(surface):
        rec = surface[name]
        print(f"    {name:<28} max {rec['supplied']} argument(s), first at "
              f"core.py:{rec['line']}" + ("  *starred" if rec["star"] else ""))
    print("  the binary is imported in a child process and read with dir() and "
          "inspect.signature() only. No byte of it is read, hashed or compared.")

    section("the extension Python would import")
    data = survey_extension()
    located = ("the extension Python would import was located",)
    if not data.get("ok"):
        why = (f"the tree's package did not import in a child with the source "
               f"tree first on sys.path: {data.get('error')}. That is itself the "
               f"shape this gate exists for -- a source tree whose extension will "
               f"not load -- and it is reported rather than raised")
        for n in ("every name core.py calls is exported by that extension",
                  "no call site supplies more arguments than the signature accepts",
                  "every call site's arity was readable"):
            skip(n, why)
        skip(*located)
        check_the_predictor_is_not_vacuous()
        tally()
        return finish()

    members = data["members"]
    ok(*located, f"{data['extension_file']}, reached through "
                 f"{data['package_from']}; it exports {len(members)} public name(s)")

    missing = sorted(n for n in surface if n not in members)
    if missing:
        bad("every name core.py calls is exported by that extension",
            f"{len(missing)} name(s) core.py calls that the binary does not "
            f"export: {', '.join(missing)}. **The binary is behind the source.** "
            f"An AttributeError at the first call, named here instead of there")
    else:
        ok("every name core.py calls is exported by that extension",
           f"all {len(surface)} name(s) core.py reaches for are present, so the "
           f"binary is not behind the source in names")

    too_many, unreadable, unpredictable = over_supplied(surface, members)
    if too_many:
        detail = "; ".join(f"{n}: core.py supplies {s}, the binary accepts {p}"
                           f" ({sig})" for n, s, p, sig in too_many)
        bad("no call site supplies more arguments than the signature accepts",
            f"{len(too_many)} call site(s) would raise TypeError: {detail}. "
            f"**The binary is behind the source in arity**, and this is the "
            f"exact failure that has taken three rebuilds to diagnose")
    else:
        ok("no call site supplies more arguments than the signature accepts",
           f"{len(surface)} call site(s) compared against the reported "
           f"signatures; the largest is dock() at "
           f"{max((r['supplied'] for r in surface.values()), default=0)} arguments")

    if unreadable:
        bad("every call site's arity was readable",
            f"{len(unreadable)} entry point(s) reported no signature this file "
            f"could read: "
            + "; ".join(f"{n} ({w})" for n, w in unreadable)
            + ". The two checks above are weaker than they look for these, and "
              "'unchecked' is not 'in agreement'")
    elif unpredictable:
        ok("every call site's arity was readable",
           f"every one of the {len(surface)} entry points reported a signature. "
           f"{len(unpredictable)} call site(s) use *args/**kwargs or land on a "
           f"variadic function, where no count can predict a raise: "
           + ", ".join(f"{n}" for n, _s, _p in unpredictable)
           + ". Those are reported rather than counted as agreement")
    else:
        ok("every call site's arity was readable",
           f"all {len(surface)} entry points reported a signature, and no call "
           f"site uses *args/**kwargs, so every comparison above was a real "
           f"comparison")

    check_the_predictor_is_not_vacuous()
    tally()
    return finish()


def tally() -> None:
    section("the tally")
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    name = "the tally accounts for every result, skips included"
    if np_ + nf + ns == len(RESULTS) == EXPECTED_CHECKS - 1:
        ok(name, f"{np_} + {nf} + {ns} = {len(RESULTS)}, and this check is the "
                 f"{len(RESULTS) + 1}th, so the run reaches the declared "
                 f"{EXPECTED_CHECKS} -- with the extension and without it, "
                 f"because a skip is a result")
    else:
        bad(name, f"recorded {len(RESULTS)} results ({np_}+{nf}+{ns}) before "
                  f"this check, which must be {EXPECTED_CHECKS - 1}")


def finish() -> int:
    np_ = sum(1 for t, _n, _d in RESULTS if t == "PASS")
    nf = sum(1 for t, _n, _d in RESULTS if t == "FAIL")
    ns = sum(1 for t, _n, _d in RESULTS if t == "SKIP")
    print()
    print(f"--- {np_} passed, {nf} failed, {ns} skipped, {len(RESULTS)} checks "
          f"(expected {EXPECTED_CHECKS})")
    if nf:
        print(f"RESULT: FAIL -- {nf} of {EXPECTED_CHECKS} checks failed. "
              f"Rebuild the extension; do not work around the TypeError")
        return 1
    if np_ == 0:
        print("RESULT: DID NOT FINISH -- this run asserted nothing")
        return 2
    print(f"RESULT: OK -- {np_} passed, {ns} could not be measured here")
    return 0


def report_crash(exc: BaseException) -> int:
    import traceback

    traceback.print_exc()
    print(f"\nRESULT: DID NOT FINISH -- {type(exc).__name__}: {exc}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001 - the point is to report it
        raise SystemExit(report_crash(_exc))
