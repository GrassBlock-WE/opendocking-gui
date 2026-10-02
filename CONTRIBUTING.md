# Contributing to Open Docking

Thanks for considering a contribution. This document is short on purpose: it
tells you what the project *is*, what it is **not** yet, and what a change has
to look like to be accepted.

Please read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before you file an
issue or a paper reference. The single most important fact about this project
is:

> **Open Docking has never been validated by redocking.** There is no evidence
> that it places a known ligand back into its crystallographic pocket, and its
> absolute kcal/mol numbers have not been cross-checked against AutoDock Vina.
> It is a structurally complete, well-tested engine — not a validated affinity
> predictor. See [`docs/VERIFICATION.md`](docs/VERIFICATION.md) §2 for the full
> list of what is unverified.

Contributions that strengthen that honesty are as welcome as code.

## Reporting a bug

Open an issue and include, if you can:

- what you ran, verbatim, including the exact command line;
- the input files (or enough of them to reproduce);
- what you expected and what happened, with the full traceback;
- platform, `rustc --version`, `python --version`, GPU model.

If a molecule is involved, attach the `.pdbqt` or `.sdf` you used. A docking
result is only meaningful with the exact box and the exact search parameters.

## Before you open a pull request

Run the full local gate. `.github/workflows/ci.yml` is the authority on what runs;
this list is a summary of it, and where the two disagree, the workflow is right and
this file is the bug. That file is not going to be re-derived by hand here on
purpose — a second copy of the gate list is a second thing to forget to update.

```bash
cargo fmt --all --check                   # formatting is enforced in CI
cargo test --workspace
cargo test -p dock-core --features gpu
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p dock-core --features gpu --all-targets -- -D warnings

# Two targets sit behind a *named* feature, so plain `cargo test` cannot reach
# them and prints nothing about them. Run them by name, in release -- they run
# real searches and the measurement is the product, not the wall clock.
cargo test --release --features convergence-budget --test convergence_budget
cargo test --release --features convergence-budget --test budget_monotonicity

python -m pip install -r requirements.txt
python -m maturin build --release -m dock-py/Cargo.toml --out dist
python -m pip install --force-reinstall --no-pip-deps dist/opendocking-*.whl
python -m pytest --pyargs opendocking.tests -q

python scripts/check_text_encoding.py         # 源码树 UTF-8 完整性（.py/.rs/.wgsl/.md…，够得着 161 个、判 159 个）
python scripts/check_repo_docs.py              # 链接 + 本机路径泄露
python scripts/check_scripts_declare.py        # 每个门禁都自述数量与出处
python scripts/docs_claims_check.py            # 文档里引用的数字是否还对得上
```

The earlier version of this list carried passing-test counts in comments. Both were
wrong by the time anyone read them, and a wrong count in a contributor guide reads
exactly like a right one. **Run the suite and read what it prints.** Where a gate has
a pinned total it is pinned in the gate itself as `EXPECTED_CHECKS`, and that number
must never move because a check failed — a total that slides when something breaks
tells you nothing about what broke.

CI runs 38 scripts under `scripts/`. Do not take that number or the list from this
file; print them from the workflow, which is the authority. Python is already a
prerequisite, so this works the same on Windows, macOS and Linux:

```bash
python -c "import re,pathlib;print(chr(10).join(sorted(set(re.findall(r'scripts/[a-z0-9_]+\.py',pathlib.Path('.github/workflows/ci.yml').read_text(encoding='utf-8'))))))"
```

A hand-copied list here is a second list to forget. The command above cannot drift
from what CI actually executes, which is the whole point of asking it rather than
listing the gates. If it does not print 38 for you, the count in this paragraph is
the stale one.

### Running them

Most of the display-dependent gates need a real GL context. On Linux that means
`xvfb-run -a -s "-screen 0 1280x1024x24"`, which is how the workflow invokes them;
`scripts/qt_gl_probe.py` exists to tell you whether your platform gives you one, and
it has four exit codes for a reason — `0` pass, `2` no context, `4` inconclusive,
`3` defect. Exit `0` there means "checked", not "there is nothing wrong", and the
workflow runs that one step with `continue-on-error: true` precisely because a
platform without a context is not a code defect.

One gate is deliberately **absent** from the workflow:

- `extension_surface_check.py` — on a runner every job builds and installs from the
  same source, so a source-versus-binary arity mismatch cannot arise and the step
  would be green by construction. Run it locally, where a stale extension lives.
  It is the gate that answers "is the binary I am testing the one this source
  was written against", and it caught exactly that: the 755,200 B extension
  sitting in the source tree for a whole session, missing a field the Python
  layer read, which made `odcli dock -o` exit 1 while `--json` on the same
  command returned three healthy poses.

A step that is green by construction is not a check, and the two gates that were
once in that position are worth naming as a pair. `installed_copy_check.py` was
excluded for the same reason as above and then **included** once the tautology
was removed: it now manufactures a deliberately stale copy of the package on
every run and requires its own predicate to refuse it, so the green beside that
self-test means something. A gate that cannot be made to fail has to be made to
fail, not excused.

If you touched the viewer, CLI, or anything under `examples/`, also run the
`examples/` checks. **They live outside the installed package and are not collected
by `pytest`**, which is exactly how a real regression slipped through once (see
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) §3.10):

```bash
cd examples && python robustness_check.py && python determinism_check.py
```

## House rules

These are not negotiable; a patch that breaks them will not be merged.

1. **`#![forbid(unsafe_code)]` stays.** `dock-core` compiles under it. Do not
   remove it, and do not add `unsafe` anywhere. PyO3 boundary code lives in
   `dock-py`, which is a different crate.

2. **Parallelism goes through rayon.** No hand-rolled threads, no `std::thread`
   fan-out, no `unsafe` sharing. If you need a new parallel loop, add it to
   rayon and keep the crate clean under clippy with `-D warnings`.

3. **Clippy must be silent.** `cargo clippy -- -D warnings`, both with and
   without `--features gpu`. Zero tolerance, in both configurations.
   `cargo fmt --all --check` is enforced the same way; run `cargo fmt --all`
   before you push rather than letting CI tell you.

4. **Tests assert behaviour, not recorded numbers.** Assert that a force is
   the negative gradient of an energy, that a symmetry operation leaves a pose
   unchanged, that a malformed input raises instead of aborting. A test that
   hard-codes `assert_eq!(energy, -7.3123)` is a change-detector, not a test,
   and it will be asked to change in every legitimate physics fix.

5. **Multi-line `assert!` needs parentheses.** Plain `assert!(a\n and b)`
   parses as a call of the `and` of `a`. Write `assert!(\n a and b\n)`.

6. **Never edit the Chinese documentation from PowerShell.** PowerShell 5.1
   reads UTF-8 as GBK, so `Get-Content` / `Set-Content` will silently destroy
   those files. Use the editor, and run `scripts/check_text_encoding.py` after.

7. **Windows is the only verified platform.** If your change touches anything
   platform-sensitive (paths, GPU adapter selection, DPI, threading), say so
   in the PR. We cannot verify macOS or Linux here, and we will not claim to.

## What would be genuinely useful

Ordered by how much they would move the project, highest first:

1. **A redocking benchmark.** Any holo complex with a well-defined pocket, a
   reference ligand, and a reproducible pipeline. This is the single missing
   validation and it is worth more than any amount of new code.
2. **Cross-checking against AutoDock Vina.** Same receptor, same ligand, same
   box: compare poses, RMSD to the crystal pose, and the energy relationship.
   A discrepancy report is as valuable as a fix.
3. **macOS and Linux CI.** The project is only verified on Windows 11.
4. **A real force-field parameter provenance note** in
   [`docs/SCORING.md`](docs/SCORING.md), so the XS radii and weight choices can
   be audited against published values.

## Licence of contributions

By opening a pull request you agree that your contribution is licensed under
**GPL-3.0-or-later**, the same as the rest of the project. Contributors are
listed in the commit history; there is no separate `AUTHORS` file to update.

If a contribution is derived from another work, state the source and its
licence in the file header. `docs/ARCHITECTURE.md` and the module-level
`Provenance` sections record what this project already derives from where;
follow that pattern.
