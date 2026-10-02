## What this changes

<!-- One or two sentences. What is different after this patch? -->

## Why

<!-- The problem it solves. If it fixes a bug, link the issue. -->

## How it was verified

<!--
The full local gate, and the real numbers. Not "tests pass" — what did you run
and what came out?

    cargo test -p dock-core                    # 104 passed
    cargo clippy --workspace --all-targets -- -D warnings
    python -m pytest --pyargs opendocking.tests -q

If you touched the viewer, the CLI, or anything under `examples/`, show that
too. Those live outside the installed package and are not collected by
`pytest`; that gap is how a real regression got through before
(docs/VERIFICATION.md §3.10).

    python scripts/workbench_interaction_check.py
    cd examples && python robustness_check.py
-->

## Checklist

- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean, **and**
      clean again with `--features gpu`
- [ ] `cargo test -p dock-core` passes, and with `--features gpu`
- [ ] `python -m pytest --pyargs opendocking.tests -q` passes against a
      **rebuilt and reinstalled** wheel
- [ ] `examples/` and `scripts/` re-run if this touched them
- [ ] `python scripts/check_text_encoding.py` reports 0 damaged files across the source tree
- [ ] `python scripts/check_repo_docs.py` reports no broken links
- [ ] `#![forbid(unsafe_code)]` still present in `dock-core`
- [ ] new tests assert behaviour or physics, not recorded output values
- [ ] any new file derived from another work states its source and licence

## Platform

<!--
Windows 11 x86_64 is the only verified platform. If this patch touches paths,
GPU adapter selection, DPI, threading or anything else platform-sensitive, say
so explicitly here.
-->

- [ ] Windows-only, or: tested on Windows and I understand macOS/Linux remain unverified
