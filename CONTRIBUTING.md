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

Run the full local gate. It is the same list CI runs:

```bash
cargo fmt --all --check                   # formatting is enforced in CI
cargo test -p dock-core                        # 104 passed
cargo test -p dock-core --features gpu         # 114 passed
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p dock-core --features gpu --all-targets -- -D warnings

python -m pip install -r requirements.txt
python -m maturin build --release -m dock-py/Cargo.toml --out dist
python -m pip install --force-reinstall --no-deps dist/odockmcode-*.whl
python -m pytest --pyargs odockmcode.tests -q

python scripts/check_doc_encoding.py           # 中文文档 UTF-8 完整性
python scripts/check_repo_docs.py              # 链接 + 本机路径泄露
```

If you touched the viewer, CLI, or anything under `examples/`, also run these.
**They live outside the installed package and are not collected by `pytest`**,
which is exactly how a real regression slipped through once (see
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) §3.10):

```bash
python scripts/workbench_interaction_check.py   # 50 checks, needs a display
python scripts/odgui_launch_check.py
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
   those files. Use the editor, and run `scripts/check_doc_encoding.py` after.

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
