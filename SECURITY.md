# Security Policy

## Supported versions

Open Docking is at **0.1.0**, a pre-1.0 research release. There is no long-term
support branch. Fixes land on `main` and are released as the next patch version.

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## What counts as a vulnerability here

The project parses untrusted files. Please report:

- a **memory-safety** issue in the Rust engine — the crates build with
  `#![forbid(unsafe_code)]`, so any such finding is a serious defect;
- a **panic that kills the host process.** `dock-core` is built with
  `panic = "abort"`, so a reachable panic takes the *calling interpreter* with
  it instead of raising a Python exception. Feeding a crafted or simply
  malformed `.pdbqt` should produce a catchable error, never a dead process.
  `examples/robustness_check.py` exists to police this and is part of the CI
  gate;
- a **path traversal** or arbitrary file write reachable from a `.pdbqt`,
  `.pdb`, `.sdf` or map file, or from a directory passed to `odockmcode dock`;
- a **resource exhaustion** issue reachable without the caller opting in — for
  example an allocation that ignores an explicit size argument;
- anything that would let a prepared file execute code.

Out of scope: crashes from inputs that are plainly out of the documented
parameter range when the tool already reports them as errors, and the
scientific limitations in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). A
docking result that is physically wrong is a research problem, not a
vulnerability — please report those as issues.

## How to report

**Do not open a public issue for a security problem.**

Use GitHub's private reporting for this repository if it is enabled. If it is
not, open a regular issue that contains **only** a description of the class of
bug and a request for a private channel — no proof-of-concept code, no
exploit details.

Please include:

- the affected version, platform, and whether GPU or CPU code is involved;
- the input that triggers it, ideally the file itself;
- the observed behaviour (exit code, crash signature);
- whether you found it by reading the code or by running something.

Expect an acknowledgement within a week. Fixes for confirmed issues are
released as a patch version and credited in `CHANGELOG.md` unless you prefer
otherwise.

## Hardening notes for users of this software

- The engine runs your input. Treat `.pdbqt` files as untrusted input, and
  prefer converting your own structure files with `odockmcode prep-receptor` /
  `prep-ligand` rather than consuming third-party `.pdbqt` directly.
- Search boxes are explicit on purpose. A box covering a whole protein needs
  tens of gigabytes; there is a hard cap (`MAX_GRID_POINTS`) and exceeding it
  is a clean error, not a silent allocation.
- `write_autodock_map_files` writes into a directory you name. It does not
  create paths from receptor atom records.
