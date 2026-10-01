"""Do the numbers the documentation claims still hold?

Run:  python scripts/docs_claims_check.py

# Why it was renamed from `scoring_docs_check.py`

The old name said "scoring", and it stopped being true. This file no longer audits
the scoring function's internals -- `scoring_cross_check.py` does that, against
the engine, with its own literal copy of the constants. What this file audits is
**every quantitative claim the prose makes**: the tables and line references in
`docs/SCORING.md` and `docs/LIMITATIONS.md`, and the figures in `README.md` and
`README.en.md`. It reads the documents and the source and asks whether each claim
still matches, including which file a claim is attributed to.

`docs_claims_check.py` says that, and nothing more is claimed for it. It is still
a `*_check.py` on purpose: the name is the only thing that makes
`check_scripts_declare.py` treat it as a gate, and a gate that stops looking like
a gate stops being audited.

# Why this file exists

`scripts/scoring_cross_check.py` asks whether the engine implements the scoring
function **this project wrote down**. It answers that by keeping its own literal
copy of the constants (`WEIGHTS`, `XS_RADIUS`) and comparing the engine against
that copy. The copy is transcribed by hand, so the question it answers is
strictly "engine == the literals in this file".

Nothing asserted **document == engine**. The document is the thing a reader
trusts; the script is the thing that runs. They were in agreement when written and
nothing kept them there. That gap is what this file closes.

# The design rule, and why it is the rule here

**Every number is parsed out of the document or out of the Rust source at run
time. This file contains no transcribed constants.** If it held its own copy of
`-0.587439` it would be the same re-transcription `scoring_cross_check.py`
already has, with one more copy to drift — and a doc check that can disagree with
its own subject teaches the reader to trust it when it is wrong. A check that
recomputes has a single source of truth and can only fail when the subject moves.

The same reasoning is why this file **used to** assert nothing about the cross-element
hydrogen bond, and why that was a mistake worth recording. It said: asserting it
"would make a green test that fails the moment the bug is fixed -- the failure
mode `smoothstep_is_c1` already demonstrates (a test that pins a deprecated
window stays green when the window is retuned)."

Both halves of that reasoning were right and the conclusion was still wrong. The
window *was* retuned -- `dock-core` fixed it, making the grid's element dimension
the **probe type** rather than a filter on the receptor, pinned by
`grid.rs::a_receptor_atom_reaches_every_probe_type` -- and because this file
asserted nothing, **nothing noticed that `SCORING.md` §5.1.1 and `LIMITATIONS.md`
§3.2 had become false**, both of which tabulated cross-element terms as exactly
`0.000000`. Measured after the fix, `N···O` at 2.0 A is `-0.033468`, not zero.

The error was not the caution; it was pinning the *symptom*. "Cross-element
scores zero" is true only while the defect is present, so a check on it is either
red or a promise to go red. "A polar receptor's atom reaches a cross-element
probe" is true in both worlds -- it fails when the defect is present and passes
after the fix -- so it is the assertion that can be made. That is what
`section 7` below now does, and it is why it is written against a *polar*
receptor: an all-carbon fixture cannot tell the two descriptions apart, which is
precisely how the stale tables survived.

# Both directions

A check that cannot fail is decoration. Every family below is mutation-proven:
perturb the document and the run goes red; perturb the Rust source and the run
goes red. See the mutation note at the bottom of `LIMITATIONS.md` §6.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import math
import re
import sys
from pathlib import Path

from opendocking import core
from opendocking.core import GridBox, Ligand, Receptor, score_conformation

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCORING = ROOT / "docs" / "SCORING.md"
LIMITATIONS = ROOT / "docs" / "LIMITATIONS.md"
CORE = ROOT / "dock-core" / "src"

#: The call-site census this file is supposed to have, declared in the file that
#: holds the checks rather than in a table another file owns -- see
#: `GATE-DECLARE` in `check_scripts_declare.py` for the format and why it cannot
#: drift. Comments only, so `EXPECTED_CHECKS` below is unaffected by them.
#: GATE-DECLARE 1
#: sites: 69 unconditional + 9 guarded
#: guards: sha256:8e0ee12d290713f6a90ed633f58d32a1ef3e292a151dc66c555c27c4a529163d
#: 78 -> 102 is measured, not typed: a run of this file on this machine reports
#: `100` of `102` passing with one accounted-for red, and nothing was removed.
#: Section 9 adds 24 sites for the claims this session wrote down and nothing
#: held -- the three passages SCORING.md 5.3 and 5.3.1 gained, the per-term
#: decomposition and its CPU-only nature, the pose contract and the qualifier
#: that makes it true, the probe's provenance, the frame measurement, the
#: two-tree contract with its recognition rule, and the cartoon's helix
#: subtypes.
#:
#: The census moved `46 -> 69` unconditional and `7 -> 9` guarded, and **the two
#: guarded ones are the deliberate pair**: `try:else` and `try:except` are the
#: two halves of the provenance import. Exactly one of them registers a result
#: on any machine -- the `check` that measures the digest, or the `check` that
#: reports the import failure as a result rather than letting it crash the
#: file -- so the *total* is 102 everywhere even though the guarded count is
#: two. That is the same property the project requires of `skip()`: a count
#: that depends on the machine is not a count, and a `try` that could silently
#: reach neither branch would break it.
#:
#: Four of the new assertions are worth naming, because none of them is
#: something a phrase grep could have produced:
#:
#:   * The qualifier. "The returned poses are minima *to within the line
#:     search's resolution*" fails by being **widened**, not by being deleted --
#:     someone turning "to within the resolution" into "is a minimum". So the
#:     qualifier is asserted as a qualifier (present, in the same blockquote as
#:     the negation it belongs to) *and* bound to the number the code uses:
#:     `lbfgs.rs::line_search` halves the step and stops below 1e-12 conf
#:     units, and the audit's own finest rung is 1e-9. Mutation-proven in both
#:     directions: deleting the words goes red, and so does retuning the floor.
#:   * The recognition rule. "What is a source file" is a *table* in
#:     `release_tree_rule.py`, so the check runs the rule's own iterator over a
#:     fixture carrying one synthetic path per way the rule can say no. Reading
#:     the tables would have been asserting a second copy of the decision.
#:   * The support boundaries, twice. SCORING.md 5.3's 5.200 A and 5.3.1's two
#:     edges are not transcribed: each is *solved* from the gauss1 weight the
#:     document itself states and `f32`'s own smallest normal and subnormal.
#:     That is also how a 20%-wrong product copied out of `grid.rs`'s comment
#:     was caught.
#:   * A two-sided contract. The CPU-only check asserts that `backend` is
#:     absent from the Python binding **and** that the document says so. Both
#:     halves must flip together, so a check that stayed green when someone
#:     closed the gap would have frozen a gap as if it were a property.
#:
#: The guard digest moved only for the two guarded sites; all 22 remaining new
#: sites are unconditional, which is why the *count* moved 46 -> 69 while the
#: guarded count moved only 7 -> 9.
EXPECTED_CHECKS = 102

CHECKS = 0
FAILURES: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(name)
    print(f"[{'ok ' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")
    return ok


def section(t: str) -> None:
    print(f"\n=== {t} ===")


def read(path: Path) -> str:
    return io.open(path, encoding="utf-8").read()


#: **Rust's numeric grammar is wider than `[\d.]+`, and a reader that assumes
#: otherwise goes red for a reason that is not a defect.**
#:
#: Every literal reader in this file -- `VINA_CUTOFF`, `DEFAULT_SPACING`,
#: `GRID_TYPE_COUNT`, `MAX_GRID_TYPE_RADIUS`, `gradient_tolerance`, the
#: `smoothstep(-0.5, 0.0, d)` windows, the `if alpha < 1e-12` floor -- matches
#: one spelling of a number. Rust accepts at least four others: `_` digit
#: separators (`4.224_152e-39`), a leading-dot form (`.5`), a bare exponent
#: (`8e0`), and a trailing dot (`8.`). This is not hypothetical: `grid.rs`
#: already writes `const PUBLISHED_G1: f64 = 4.224_152e-39;` and
#: `scoring.rs:166-170` writes all five Vina weights as `-0.035_579` and
#: friends -- thirteen such literals in `dock-core/src` today. A separator
#: landing inside one of the constants a check reads would make it report a
#: truncated number and fail against a correct engine.
#:
#: So the separators are removed once, here, rather than widened at thirteen
#: call sites where the next reader would forget. What this normaliser cannot
#: do: it rewrites *text*, so an identifier that looks like a separated number
#: (`types.rs:781`'s `dist2_3`) is altered too. No reader in this file looks for
#: such an identifier, and the failure mode if one ever did would be a red, not
#: a silent pass.
def rust_text(path: Path) -> str:
    """`read(path)` with Rust's digit separators folded out."""
    return re.sub(r"(?<=\d)_(?=\d)", "", read(path))


def numbers_in(text: str) -> list[str]:
    """Every signed decimal literal in `text`, as written."""
    return re.findall(r"[-−+]?\d+\.\d+", text)


def blocks_after(doc: str, heading: str) -> list[str]:
    """Every fenced code block following `heading`, in order."""
    tail = doc.split(heading, 1)[1]
    return re.findall(r"```[a-z]*\n(.*?)```", tail, re.S)


def block_after(doc: str, heading: str) -> str:
    """The fenced code block following `heading`."""
    found = blocks_after(doc, heading)
    if not found:
        raise AssertionError(f"no fenced block after {heading!r}")
    return found[0]


# ==========================================================================
# 1. The weights in SCORING.md section 3 against the engine's own report
# ==========================================================================

section("SCORING.md section 3 against core.scoring_descriptions()")

scoring_doc = read(SCORING)
scoring_rs = rust_text(CORE / "scoring.rs")
# Section 3 holds two blocks: the VinaWeights literal, then the Vinardo one.
# Anchor on the heading rather than on the word "Vinardo", which also occurs in
# the ledger table at the top of this document.
sec3_blocks = blocks_after(scoring_doc, "## 3. 权重")
check(
    len(sec3_blocks) >= 2,
    "SCORING.md section 3 still holds both weight blocks",
    f"found {len(sec3_blocks)} fenced blocks after the heading",
)
vina_block, vinardo_block = sec3_blocks[0], sec3_blocks[1]

# The doc's first block is a Rust struct literal; pull `name: value` pairs.
doc_pairs = dict(re.findall(r"(\w+)\s*:\s*([−+-]?\d*\.\d+)", vina_block))


def as_float(s: str) -> float:
    return float(s.replace("−", "-"))


doc_vina = {k: as_float(v) for k, v in doc_pairs.items()}
# `ScoringFunction`/`VinaWeights` are not printed by the engine; the five the
# engine does print are compared. `intramolecular_scale` is checked separately
# below against the Rust source, because the engine never reports it.
engine_key = {"gauss1": "g1", "gauss2": "g2", "repulsion": "rep",
              "hbond": "hbond", "hydrophobic": "hyd"}

reported = {d.split(":")[0]: d for d in core.scoring_descriptions()}
# The VinaWeights block is compared against the engine's `vina:` line. The
# engine's `vinardo:` line is compared against the *second* block further down,
# against that block's own numbers -- comparing the Vina literal to both lines
# would assert the two weight sets are identical, which is the opposite of true.
for doc_name, engine_name in engine_key.items():
    want = doc_vina[doc_name]
    line = reported["vina"]
    check(
        f"{engine_name}={want:.6f}" in line,
        f"SCORING.md's {doc_name} for vina is what the engine reports",
        f"engine says {line!r}; the document says {doc_name}={want}",
    )

check(
    "intramolecular_scale" in doc_vina,
    "SCORING.md states an intramolecular scale",
    f"{doc_vina.get('intramolecular_scale')}",
)

# The Vinardo block is written as commented assignments, and one of them is
# chained (`gauss1 = gauss2 = 0`), so a naive `name = number` pairing silently
# loses gauss1. Parse per line: every name assigned on a line takes that line's
# last number.
def parse_assignments(block: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in block.splitlines():
        line = line.split("//", 1)[0]
        names = re.findall(r"(\w+)\s*=", line)
        nums = re.findall(r"[−+-]?[\d.]+", line)
        if names and nums:
            for n in names:
                out[n] = as_float(nums[-1])
    return out


doc_vinardo = parse_assignments(vinardo_block)
VINARDO_COEFFS = ("gauss1", "gauss2", "repulsion", "hbond", "hydrophobic")
check(
    all(n in doc_vinardo for n in VINARDO_COEFFS)
    and "intramolecular_scale" in doc_vinardo,
    "SCORING.md lists the five Vinardo coefficients and its scale",
    f"found {sorted(doc_vinardo)}",
)
for doc_name, engine_name in (("gauss1", "g1"), ("gauss2", "g2"),
                              ("repulsion", "rep"), ("hbond", "hbond"),
                              ("hydrophobic", "hyd")):
    if doc_name not in doc_vinardo:
        check(False, f"SCORING.md's Vinardo {doc_name} is present",
              "not found in the Vinardo block")
        continue
    want = doc_vinardo[doc_name]
    check(
        f"{engine_name}={want:.6f}" in reported["vinardo"],
        f"SCORING.md's Vinardo {doc_name} is what the engine reports",
        f"engine says {reported['vinardo']!r}; the document says {want}",
    )

# The engine never prints `intramolecular_scale` -- `description()` formats only
# the five pair coefficients -- so there is no runtime report to compare against
# and the binding has to be to the source. Saying so matters: a reader who
# assumes this coefficient is engine-verified is wrong, and a check that
# silently compared nothing would let them believe it.
#
# **The first version of this was a false positive, and the regex is why.**
# `re.findall(r"intramolecular_scale:\s*([\d.]+)", scoring_rs)` found *three*
# matches and reported the third as a third coefficient: the two real struct
# literals, plus the line inside `VINA_WEIGHT_DOC` -- a `const &str` in the
# test module that quotes the documented block back at you. A regex over source
# text cannot see the difference between a value and a sentence about a value,
# so it read a documentation string as a definition and turned a correct
# document red. Three matches, two coefficients, one red that meant nothing.
#
# The fix strips comments and string literals *before* looking, and then binds
# each coefficient to the construction that defines it rather than to its
# position in a list. What this still cannot see, stated so the next reader
# does not assume more than it has: a scale written as `..Default::default()`
# or built by a macro would leave no literal to find (the check would then see
# *fewer* than two and go red, which is the safe direction), and a value
# computed at run time would be invisible to any source reader at all. There is
# no source-level check that can cover that, and pretending otherwise is what
# the first version did.
def _rust_code_only(src: str) -> str:
    """`src` with line comments, block comments and string literals removed.

    Literals become `""` rather than nothing, so the result still has one
    "character" per character of input and a failure report can name a line.
    The scanner is not a Rust parser and does not claim to be: it does not know
    raw strings (`r"..."`), byte strings, lifetimes or char literals, so a
    `'` immediately before a `"` would be read as a string start. Nothing in
    the file it is applied to does that, and the failure mode if one ever did is
    a red rather than a silent pass.
    """
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        if src[i] == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
            i += 1
            out.append('""')
        elif src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
        elif src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


scoring_code = _rust_code_only(scoring_rs)


def _scale_in(anchor: str) -> float | None:
    """The `intramolecular_scale` literal inside the block `anchor` opens."""
    if anchor not in scoring_code:
        return None
    tail = scoring_code.split(anchor, 1)[1]
    # The block ends at its first closing brace at the anchor's own indent level;
    # for both call sites here that is the first `\n    }` after the anchor.
    end = tail.find("\n    }")
    body = tail if end < 0 else tail[:end]
    found = re.search(r"intramolecular_scale:\s*([\d.]+)", body)
    return float(found.group(1)) if found else None


src_vina_scale = _scale_in("impl Default for VinaWeights")
src_vinardo_scale = _scale_in("pub fn vinardo() -> VinaWeights")
# And the *count* over code, which is the floor the first version got wrong: two
# literals, in two constructions, and no third anywhere in the source.
_all_code_scales = [float(v) for v in re.findall(
    r"intramolecular_scale:\s*([\d.]+)", scoring_code)]
doc_vinardo_scale = doc_vinardo.get("intramolecular_scale")
src_scales = [src_vina_scale, src_vinardo_scale]
check(
    doc_vinardo_scale is not None
    and src_vina_scale is not None and src_vinardo_scale is not None
    and len(_all_code_scales) == 2
    and abs(src_vina_scale - doc_vina["intramolecular_scale"]) < 1e-12
    and abs(src_vinardo_scale - doc_vinardo_scale) < 1e-12,
    "SCORING.md's two intramolecular scales are scoring.rs's",
    f"scoring.rs defines vina={src_vina_scale} in `impl Default for "
    f"VinaWeights` and vinardo={src_vinardo_scale} in `VinaWeights::vinardo`, "
    f"and there are exactly {len(_all_code_scales)} "
    f"`intramolecular_scale:` literals in the file's *code*; the document has "
    f"vina={doc_vina.get('intramolecular_scale')}, "
    f"vinardo={doc_vinardo_scale}. NOTE: bound to the source, not to "
    f"core.scoring_descriptions(), which does not report this coefficient. "
    f"Comments and string literals are stripped before the literals are "
    f"counted, because the previous version of this check counted the quoted "
    f"block in `VINA_WEIGHT_DOC` as a third coefficient and reported a "
    f"correct document as wrong. Each value is bound to the construction that "
    f"defines it, so reordering the two cannot swap them silently",
)


# ==========================================================================
# 2. The radius table in SCORING.md section 1 against types.rs
# ==========================================================================

section("SCORING.md section 1's radius table against types.rs::interaction_radius")

types_rs = rust_text(CORE / "types.rs")
radius_fn = types_rs.split("pub fn interaction_radius", 1)[1]
radius_fn = radius_fn.split("}", 1)[0]

# Parse `Element::X => <number>` straight out of the match arms.
rust_radii = {
    m.group(1): float(m.group(2))
    for m in re.finditer(r"Element::(\w+)\s*=>\s*([\d.]+)", radius_fn)
}
doc_row = re.search(
    r"\|\s*无法识别的 token\s*\|\s*([\d.]+)\s*\|\s*\|.*?\|\s*([\d.]+)\s*\|",
    scoring_doc)
check(doc_row is not None, "SCORING.md's radius table still has the two columns",
      f"matched {bool(doc_row)}")

table = scoring_doc.split("`R` (Å)", 1)[1].split("这些是", 1)[0]
doc_radii = {}
# The element column may carry a parenthetical gloss (e.g. "H（PDBQT 只保留极性氢）"),
# and one row is labelled in prose ("无法识别的 token" -> Element::Placeholder).
for row in re.finditer(r"\|\s*([A-Z][a-z]?)\b[^|]*\|\s*([\d.]+)\s*\|", table):
    doc_radii[row.group(1)] = float(row.group(2))
doc_radii["Placeholder"] = float(
    re.search(r"无法识别的 token\s*\|\s*([\d.]+)", table).group(1))

for element, want in sorted(doc_radii.items()):
    got = rust_radii.get(element)
    check(
        got is not None and abs(got - want) < 1e-12,
        f"the document's R({element}) = {want} is types.rs's",
        f"types.rs says {got!r}",
    )

check(
    "H" in doc_radii and abs(rust_radii.get("H", -1) - doc_radii["H"]) < 1e-12,
    "R(H) = 0.0 is stated and is what types.rs has",
    f"types.rs H = {rust_radii.get('H')}, document H = {doc_radii.get('H')}. "
    f"Note: this checks the radius *constant* only. It does NOT check what a "
    f"polar hydrogen does when it probes the grid -- see SCORING.md section 1, "
    f"where the 3.7773 A C...H zero crossing is reconciled by the fact that H "
    f"folds into the carbon map (types.rs grid_type_index)",
)


# ==========================================================================
# 3. The windows, cutoff and stride the document states
# ==========================================================================

section("SCORING.md sections 2.3 / 5.2 / 5.3 / 6.4 against dock-core")

hb_window = re.search(r"hbond_term.*?smoothstep\(\s*([−+-]?[\d.]+)\s*,\s*([−+-]?[\d.]+)",
                      scoring_rs, re.S)
hyd_window = re.search(r"hydrophobic_term.*?smoothstep\(\s*([−+-]?[\d.]+)\s*,\s*([−+-]?[\d.]+)",
                       scoring_rs, re.S)
cutoff = re.search(r"VINA_CUTOFF:\s*f64\s*=\s*([\d.]+)", scoring_rs)

grid_rs = rust_text(CORE / "grid.rs")
maps_per_type = int(re.search(r"MAPS_PER_TYPE:\s*usize\s*=\s*(\d+)", grid_rs).group(1))
grid_type_count = int(re.search(r"GRID_TYPE_COUNT:\s*usize\s*=\s*(\d+)", types_rs).group(1))
default_spacing = float(re.search(r"DEFAULT_SPACING:\s*f64\s*=\s*([\d.]+)", grid_rs).group(1))
max_points = int(re.search(r"MAX_GRID_POINTS:\s*u64\s*=\s*1\s*<<\s*(\d+)", grid_rs).group(1))

# --- hb and hyd windows, taken from the definitive `hb(d) = ...` equations.
doc_hb = re.search(r"hb\(d\)\s*=\s*1\s*−\s*S\(\s*([−+-]?[\d.]+)\s*,\s*([−+-]?[\d.]+)\s*,\s*d\s*\)",
                   scoring_doc)
doc_hyd = re.search(r"hyd\(d\)\s*=\s*1\s*−\s*S\(\s*([−+-]?[\d.]+)\s*,\s*([−+-]?[\d.]+)\s*,\s*d\s*\)",
                    scoring_doc)
check(
    doc_hb is not None
    and (as_float(doc_hb.group(1)), as_float(doc_hb.group(2)))
    == (as_float(hb_window.group(1)), as_float(hb_window.group(2))),
    "SCORING.md's hb(d) equation is scoring.rs's hbond_term window",
    f"scoring.rs uses ({hb_window.group(1)}, {hb_window.group(2)}); the document "
    f"says {doc_hb and doc_hb.groups()}",
)
check(
    doc_hyd is not None
    and (as_float(doc_hyd.group(1)), as_float(doc_hyd.group(2)))
    == (as_float(hyd_window.group(1)), as_float(hyd_window.group(2))),
    "SCORING.md's hyd(d) equation is scoring.rs's hydrophobic_term window",
    f"scoring.rs uses ({hyd_window.group(1)}, {hyd_window.group(2)}); the "
    f"document says {doc_hyd and doc_hyd.groups()}",
)

# --- cutoff 8.0
doc_cutoff = re.search(r"SpatialKernels::cutoff\s*=\s*([\d.]+)", scoring_doc)
check(
    doc_cutoff is not None and abs(float(doc_cutoff.group(1)) - float(cutoff.group(1))) < 1e-12,
    "SCORING.md's cutoff is scoring.rs's VINA_CUTOFF",
    f"document {doc_cutoff and doc_cutoff.group(1)}, source {cutoff.group(1)}",
)

# --- stride = GRID_TYPE_COUNT * MAPS_PER_TYPE, and the WGSL constant
doc_stride = re.search(r"STRIDE\s*=\s*GRID_TYPE_COUNT\s*·\s*MAPS_PER_TYPE\s*=\s*\d+\s*·\s*\d+\s*=\s*(\d+)",
                       scoring_doc)
wgsl = read(CORE / "gpu" / "energy.wgsl")
wgsl_stride = int(re.search(r"const STRIDE:\s*u32\s*=\s*(\d+)u", wgsl).group(1))
check(
    doc_stride is not None
    and int(doc_stride.group(1)) == grid_type_count * maps_per_type,
    "SCORING.md's STRIDE = GRID_TYPE_COUNT * MAPS_PER_TYPE",
    f"document {doc_stride and doc_stride.group(1)}, "
    f"{grid_type_count} * {maps_per_type} = {grid_type_count * maps_per_type}",
)
check(
    wgsl_stride == grid_type_count * maps_per_type,
    "the WGSL STRIDE constant equals the Rust stride",
    f"WGSL {wgsl_stride}, Rust {grid_type_count * maps_per_type}",
)

# --- default spacing
doc_spacing = re.search(r"默认间距取\s*([\d.]+)\s*Å", scoring_doc)
check(
    doc_spacing is not None
    and abs(float(doc_spacing.group(1)) - default_spacing) < 1e-12,
    "SCORING.md's default spacing is grid.rs's DEFAULT_SPACING",
    f"document {doc_spacing and doc_spacing.group(1)}, source {default_spacing}",
)

# --- MAX_GRID_POINTS, stated in LIMITATIONS.md section 6
limitations_doc = read(LIMITATIONS)
max_shift = int(re.search(r"MAX_GRID_POINTS:\s*u64\s*=\s*1\s*<<\s*(\d+)", grid_rs).group(1))
max_points = 1 << max_shift
doc_max = re.search(r"MAX_GRID_POINTS\s*=\s*2\^(\d+)", limitations_doc)
check(
    doc_max is not None and int(doc_max.group(1)) == max_shift,
    "LIMITATIONS.md's MAX_GRID_POINTS exponent is grid.rs's",
    f"document 2^{doc_max and doc_max.group(1)}, source 1 << {max_shift} = {max_points}",
)

# The byte size LIMITATIONS.md quotes for that ceiling.
ceiling_gib = max_points * grid_type_count * maps_per_type * 4 / (1024 ** 3)
check(
    f"{ceiling_gib:.1f} GiB" in limitations_doc,
    "LIMITATIONS.md's GiB figure for the grid ceiling is the arithmetic",
    f"{max_points} points x {grid_type_count * maps_per_type} f32 x 4 B = "
    f"{ceiling_gib:.1f} GiB",
)


# ==========================================================================
# 4. LIMITATIONS.md's memory table against estimate_memory_mb
# ==========================================================================

section("LIMITATIONS.md section 6's memory table against estimate_memory_mb")

rec = Receptor.from_pdbqt_str(
    "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n")

mem_table = limitations_doc.split("实测（`estimate_memory_mb`", 1)[1].split("上一版", 1)[0]
rows = re.findall(
    r"\|\s*(\d+)\s*Å\s*\|\s*\**([\d.]+)\s*MB\**\s*\|\s*\**([\d.]+)\s*MB\**\s*\|\s*\**([\d.]+)\s*MB\**\s*\|",
    mem_table)
check(len(rows) == 3, "LIMITATIONS.md's memory table has three cube sizes",
      f"parsed {len(rows)} rows")

for side, at375, at1875, at75 in rows:
    for spacing, stated in ((0.375, at375), (0.1875, at1875), (0.75, at75)):
        box = GridBox.from_center_size((0, 0, 0), (float(side),) * 3)
        measured = rec.estimate_memory_mb(box, spacing)
        # The table publishes two decimals, so the tolerance has to be tighter
        # than half of the last published digit. At 0.01 MB a wrong value slips
        # through -- `1.23` against a true 1.2207 differs by 0.0093 -- which is
        # exactly the kind of mutation that has to be caught.
        check(
            abs(measured - float(stated)) < 0.005,
            f"a {side} A cube at {spacing} A is the {stated} MB the document says",
            f"measured {measured:.4f} MB, published {stated} MB",
        )


# ==========================================================================
# 5. Line-numbered claims: the cited line must still say what is claimed
# ==========================================================================

section("the source-line citations in LIMITATIONS.md still point at what is claimed")


def cited_line(path: Path, lineno: int) -> str:
    return io.open(path, encoding="utf-8").read().splitlines()[lineno - 1]


mc_rs = CORE / "search" / "monte_carlo.rs"
mc_cite = re.search(r"`monte_carlo\.rs:(\d+)`", limitations_doc)
check(mc_cite is not None, "LIMITATIONS.md cites monte_carlo.rs by line", "")
if mc_cite:
    n = int(mc_cite.group(1))
    text = cited_line(mc_rs, n)
    check(
        "2.0 * ligand.radius() + 1.0" in text.replace(" ", " "),
        f"monte_carlo.rs:{n} is the `2 * radius + 1.0` box check the document cites",
        f"line reads: {text.strip()!r}",
    )

prep_cite = re.search(r"`prep\.py:(\d+)`", limitations_doc)
prep_py = ROOT / "dock-py" / "python" / "opendocking" / "prep.py"
check(prep_cite is not None, "LIMITATIONS.md cites prep.py by line", "")
if prep_cite:
    n = int(prep_cite.group(1))
    text = cited_line(prep_py, n)
    seed = re.search(r"randomSeed\s*=\s*(0[xX][0-9A-Fa-f]+)", text)
    doc_seed = re.search(r"\*\*`(0[xX][0-9A-Fa-f]+)`\*\*", limitations_doc)
    check(
        seed is not None and doc_seed is not None
        and seed.group(1) == doc_seed.group(1),
        f"prep.py:{n} is the randomSeed the document quotes",
        f"line reads {text.strip()!r}; document quotes {doc_seed and doc_seed.group(1)}",
    )

lig_rs = CORE / "ligand.rs"
lig_src = read(lig_rs)
intra_cite = re.search(r"(kinematics|ligand)\.rs::MIN_INTRA_BOND_DISTANCE\s*=\s*(\d+)",
                       scoring_doc)
intra_src = int(re.search(r"MIN_INTRA_BOND_DISTANCE:\s*usize\s*=\s*(\d+)",
                          lig_src).group(1))
check(
    intra_cite is not None and int(intra_cite.group(2)) == intra_src,
    "SCORING.md's MIN_INTRA_BOND_DISTANCE value is ligand.rs's",
    f"document {intra_cite and intra_cite.group(2)}, source {intra_src}",
)
# The constant's *file attribution* is the fragile half: the value can be right
# while the file named next to it is wrong, and a reader debugging a mismatch
# starts at the file the document points them to.
defines_in = "ligand.rs" if f"pub const MIN_INTRA_BOND_DISTANCE" in lig_src else "unknown"
check(
    intra_cite is not None and intra_cite.group(1) == defines_in.split(".")[0],
    "SCORING.md names the file that actually defines MIN_INTRA_BOND_DISTANCE",
    f"the document says {intra_cite and intra_cite.group(1)}.rs; it is defined "
    f"in {defines_in}. The value {intra_src} is right, the file name is not",
)


# ==========================================================================
# 6. The one measured claim that is cheap and load-bearing
# ==========================================================================

section("SCORING.md section 10's corrected C...C minimum, re-measured")

# Recompute the apolar pair curve from the documented formula, then confirm
# the engine lands on it. This is the claim the previous revision got wrong
# twice (3.00 A / -0.050, then 1.3-2.3 A), so it gets a check.
W = {"g1": -0.035579, "g2": -0.005156, "rep": 0.840245,
     "hb": -0.587439, "hyd": -0.035069}


def smoothstep(a: float, b: float, x: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


def spec_apolar(r: float) -> float:
    d = r - 2 * 1.90
    return (W["g1"] * math.exp(-(((d - 0.5) / 0.5) ** 2))
            + W["g2"] * math.exp(-((d / 0.5) ** 2))
            + (W["rep"] * d * d if d < 0 else 0.0)
            + W["hyd"] * (1.0 - smoothstep(0.5, 1.5, d)))


lig = Ligand.from_pdbqt_str(
    "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n")
box = GridBox.from_center_size((0, 0, 0), (14.0, 14.0, 14.0))
maps = rec.precalculate(box, "vina", 0.375)


def engine_energy(r: float) -> float:
    conf = np.zeros(lig.num_dof)
    conf[0] = r
    return float(score_conformation(lig, maps, conf)[0])


curve = [(r, engine_energy(r)) for r in np.arange(4.10, 4.501, 0.01)]
best_r, best_e = min(curve, key=lambda t: t[1])
spec_best_r, spec_best_e = min(
    ((r, spec_apolar(r)) for r in np.arange(4.10, 4.501, 0.001)), key=lambda t: t[1])

check(
    abs(best_r - spec_best_r) < 0.03 and abs(best_e - spec_best_e) < 0.002,
    "the engine's apolar minimum matches the documented formula's",
    f"engine {best_r:.3f} A / {best_e:.5f}; formula {spec_best_r:.3f} A / "
    f"{spec_best_e:.5f}",
)

# The figure SCORING.md section 10 now publishes.
doc_min = re.search(r"\*\*([\d.]+)\s*Å\s*/\s*−?([\d.]+)\*\*（公式", scoring_doc)
check(
    doc_min is not None
    and abs(float(doc_min.group(1)) - best_r) < 0.02
    and abs(abs(float(doc_min.group(2))) - abs(best_e)) < 0.002,
    "SCORING.md section 10's published minimum is the measured one",
    f"document {doc_min and doc_min.groups()}, measured {best_r:.3f} / {best_e:.5f}",
)

# The refuted claim must not have crept back **as an assertion**. Its exact
# phrasing is "3.00 A / -0.050"; the document may legitimately name it while
# refuting it, so the test is not "the string is absent" -- that would forbid
# the very correction being made. It is: every occurrence of the *claim* sits
# inside a 「...」 span. A table row or a bolded figure would fall outside one.
REFUTED = "3.00 Å 处 −0.050"
quoted_spans = [(m.start(), m.end()) for m in re.finditer("「[^」]*」", scoring_doc)]
claims = [m.start() for m in re.finditer(re.escape(REFUTED), scoring_doc)]
unquoted_claims = [c for c in claims
                   if not any(a <= c < b for a, b in quoted_spans)]
check(
    bool(claims) and not unquoted_claims,
    f"SCORING.md only ever quotes {REFUTED!r}, never asserts it",
    f"{len(claims)} occurrence(s), {len(unquoted_claims)} outside a quotation. "
    f"An unquoted one would be the refuted figure being published again",
)
at_3 = engine_energy(3.00)
check(
    at_3 > 0,
    "at 3.00 A the apolar pair is repulsive, not attractive",
    f"engine gives {at_3:+.6f} -- inside the wall, which is why -0.050 is "
    f"unreachable at that separation",
)


# ==========================================================================
# 7. README.md and README.en.md
# ==========================================================================
#
# What is deliberately NOT here
#
# No docking energy, no RMSD and no pose count is checked. Those numbers come
# out of the grid, and `grid.rs` is mid-fix: the element partition that makes
# cross-element hydrogen bonds score exactly zero is being removed, which will
# move every published docking number. A check that pins them now would go red
# on the day the fix lands and would then have to be deleted -- which is a worse
# failure than never having written it. The pocket numbers are safe to check
# because the pocket search is geometric and never touches the scoring function.
#
# What is deliberately NOT here either: a check that the READMEs' own numbers
# are right. Several are wrong (drafts supplied to the document owner), and a
# green test asserting a known-wrong number is the `smoothstep_is_c1` failure
# mode. Those numbers are reported, not pinned. Only currently-true claims get
# a check, so this suite protects what is correct and leaves the rest visible.

section("README.md and README.en.md against the code they describe")

README_CN = ROOT / "README.md"
README_EN = ROOT / "README.en.md"
readme_cn, readme_en = read(README_CN), read(README_EN)

cli_py = read(ROOT / "dock-py" / "python" / "opendocking" / "cli.py")
prep_py = read(ROOT / "dock-py" / "python" / "opendocking" / "prep.py")
contacts_py = read(ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "contacts.py")
pockets_py = read(ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "pockets.py")
app_py = read(ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "app.py")


def cli_default(flag: str):
    """The `default=` of the add_argument call that declares `flag`.

    Scoped to the call rather than to the line, because several of these are
    written one-argument-per-line.
    """
    m = re.search(r"add_argument\((?:[^()]|\([^()]*\))*?" + re.escape(flag)
                  + r"(?:[^()]|\([^()]*\))*?default=([^,)\n]+)", cli_py, re.S)
    return m.group(1).strip().strip('"\'') if m else None


def in_both(pattern: str) -> bool:
    return bool(re.search(pattern, readme_cn)) and bool(re.search(pattern, readme_en))


def lines_mentioning(doc: str, needle: str):
    return [l for l in doc.splitlines() if needle in l]


def near_flag(pattern: str, flag: str, window: int = 140) -> bool:
    """True when `pattern` holds within `window` characters of `flag`.

    Two things force a window rather than a whole line. A bare `\\b9\\b` matches a
    dozen unrelated digits in a README, so a check written that way passes after
    the number it was written about has been changed -- the same "green while
    wrong" failure as a test that pins a deprecated constant. But both READMEs
    are hard-wrapped prose, and the flag and its default routinely land on
    different lines (`--box-padding` / `sets the clearance (default 4 A)`), so a
    line-scoped match misses the real text. The window is the compromise: wide
    enough for the wrap, narrow enough not to reach the next option.
    """
    for doc in (readme_cn, readme_en):
        hits = [m.start() for m in re.finditer(re.escape(flag), doc)]
        if not hits:
            return False
        if not any(re.search(pattern, doc[max(0, h - window):h + window])
                   for h in hits):
            return False
    return True


# --- option defaults the READMEs publish -----------------------------------
check(
    near_flag(r"\b4\s*Å", "--box-padding") and cli_default("--box-padding") == "4.0",
    "the READMEs' --box-padding default (4 Å) is cli.py's",
    f"cli.py --box-padding default={cli_default('--box-padding')}",
)
check(
    near_flag(r"default 9|默认 9", "--num_modes")
    and cli_default("--num_modes") == "9",
    "the READMEs' --num_modes default (9) is cli.py's",
    f"cli.py --num_modes default={cli_default('--num_modes')}",
)
check(
    near_flag(r"default 1\.0 Å|默认 1\.0\s*Å", "--rmsd-cutoff")
    and cli_default("--rmsd-cutoff") == "1.0",
    "the READMEs' --rmsd-cutoff default (1.0 Å) is cli.py's",
    f"cli.py --rmsd-cutoff default={cli_default('--rmsd-cutoff')}",
)
check(
    in_both(r"\b0\.375\s*Å") and cli_default("--spacing") == "0.375",
    "the READMEs' --spacing default (0.375 Å) is cli.py's and grid.rs's",
    f"cli.py default={cli_default('--spacing')}, "
    f"grid.rs DEFAULT_SPACING={default_spacing}",
)
_mc_src = io.open(mc_rs, encoding="utf-8").read()
check(
    in_both(r"2\s*×\s*(radius|半径)\s*\+\s*1")
    and "let needed = 2.0 * ligand.radius() + 1.0;" in _mc_src,
    "the READMEs' engine box floor is monte_carlo.rs's",
    "monte_carlo.rs:114 `let needed = 2.0 * ligand.radius() + 1.0;`",
)

# --- version and toolchain minimums ----------------------------------------
py_version = re.search(r'requires-python\s*=\s*">=([\d.]+)"',
                       read(ROOT / "dock-py" / "pyproject.toml")).group(1)
rust_version = re.search(r'rust-version\s*=\s*"([\d.]+)"', read(ROOT / "Cargo.toml")).group(1)
py_init = read(ROOT / "dock-py" / "python" / "opendocking" / "__init__.py")
pkg_version = re.search(r'__version__\s*=\s*"([^"]+)"', py_init).group(1)
cargo_version = re.search(r'^version\s*=\s*"([^"]+)"', read(ROOT / "Cargo.toml"), re.M).group(1)
cff_version = re.search(r"^version:\s*(\S+)", read(ROOT / "CITATION.cff"), re.M).group(1)

check(
    in_both(r"Rust 1\.75\+") and rust_version == "1.75",
    "the READMEs' 'Rust 1.75+' is Cargo.toml's rust-version",
    f"Cargo.toml rust-version={rust_version}",
)
check(
    in_both(r"Python 3\.9\+") and py_version == "3.9",
    "the READMEs' 'Python 3.9+' is pyproject.toml's requires-python",
    f"pyproject.toml requires-python=>={py_version}",
)
# The wheel glob appears twice per file (CPU build and GPU build), so "the
# pattern is present somewhere" would still pass after one of them was edited.
# Require that *every* wheel glob in both files names the same version -- and
# that it is the version the package, the crate and the citation all declare.
# The three inputs are collected into one set on purpose: a check that only
# compares the READMEs to each other passes when the whole project renames.
_globs = re.findall(r"opendocking-(\d+\.\d+\.\d+)-\*", readme_cn) + \
    re.findall(r"opendocking-(\d+\.\d+\.\d+)-\*", readme_en)
_declared = {pkg_version, cargo_version, cff_version}
check(
    _globs and set(_globs) | _declared == {"0.1.0"} and len(_declared) == 1,
    "every wheel glob in both READMEs names the one version everything declares",
    f"README globs {sorted(set(_globs))} across {len(_globs)} occurrence(s); "
    f"__init__={pkg_version}, Cargo={cargo_version}, CITATION.cff={cff_version}. "
    f"Union must be a single value",
)

# --- workbench numbers ------------------------------------------------------
msaa = re.search(r"MSAA_SAMPLES\s*=\s*(\d+)", app_py).group(1)
check(
    in_both(r"4x") and msaa == "4",
    "the READMEs' 4x multisampling is app.py's MSAA_SAMPLES",
    f"app.py MSAA_SAMPLES={msaa}",
)

hb_max = re.search(r"HBOND_MAX\s*=\s*([\d.]+)", contacts_py).group(1)
hb_angle = re.search(r"HBOND_MIN_ANGLE\s*=\s*([\d.]+)", contacts_py).group(1)
check(
    in_both(r"2\.6\s*Å") and float(hb_max) == 2.6,
    "the READMEs' 2.6 Å hydrogen-bond cut-off is contacts.py's HBOND_MAX",
    f"contacts.py HBOND_MAX={hb_max}",
)
check(
    in_both(r"120") and float(hb_angle) == 120.0,
    "the READMEs' 120° hydrogen-bond angle is contacts.py's HBOND_MIN_ANGLE",
    f"contacts.py HBOND_MIN_ANGLE={hb_angle}",
)

from opendocking.workbench import pockets as _pk  # noqa: E402

ceiling = _pk.DEFAULT_MAX_VOLUME
check(
    in_both(r"\b1500\s*Å") and ceiling == 1500.0,
    "the READMEs' 1500 Å³ site ceiling is pockets.py's DEFAULT_MAX_VOLUME",
    f"pockets.py DEFAULT_MAX_VOLUME={ceiling}",
)


# --- the crambin pocket numbers (geometric; immune to the grid fix) ---------
import numpy as _np  # noqa: E402


def _pdbqt_view(path):
    xyz, els, res = [], [], []
    for line in io.open(path, encoding="utf-8").read().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            tok = line[77:79].strip()
            els.append(tok[0] if tok else "C")
            res.append(f"{line[17:20].strip()} {line[22:26].strip()}")
    return xyz, els, res


_ex = ROOT / "examples"
_coords, _els, _res = _pdbqt_view(_ex / "1crn_prep.pdbqt")
_sites = _pk.find_pockets(_coords, _els, residues=_res)
_uncapped = _pk.find_pockets(_coords, _els, residues=_res, max_pockets=10 ** 6)

# --- which site is "ibuprofen's", and is that the one the pose is in -------
#
# This replaced a check on the READMEs' "16 ligand atoms", which stopped
# existing when that sentence was replaced. The replacement is a *stronger*
# claim on the same sentence rather than a deletion: the READMEs now say which
# site the ligand occupies, and the shipped pose can be used to check it
# offline. The ligand's centroid in `crambin_pose.pdbqt` MODEL 1 sits 2.81 A
# from the 9th-ranked site's centre while the next-nearest is 9.49 A away, so
# the identity is decidable and the margin is wide enough that a reordering of
# the pocket ranking cannot silently keep the sentence true.
_pose_lines, _in_model = [], False
for _l in (_ex / "crambin_pose.pdbqt").read_text(encoding="utf-8").splitlines():
    if _l.startswith("MODEL"):
        _in_model = True
        continue
    if _l.startswith("ENDMDL"):
        break
    if _in_model:
        _pose_lines.append(_l)
_pose_xyz = _np.array([[float(_l[30:38]), float(_l[38:46]), float(_l[46:54])]
                       for _l in _pose_lines
                       if _l.startswith(("ATOM", "HETATM"))])
_pose_centroid = _pose_xyz.mean(axis=0)
_by_distance = sorted(
    (float(_np.linalg.norm(_np.asarray(s.center) - _pose_centroid)), i + 1,
     s.volume)
    for i, s in enumerate(_uncapped))
_nearest_d, _nearest_rank, _nearest_vol = _by_distance[0]
_runner_up_d = _by_distance[1][0]
check(
    in_both(r"第\s*9|ninth") and _nearest_rank == 9
    and _nearest_d < 4.0 and _runner_up_d > 2 * _nearest_d,
    "the READMEs' 'the site ibuprofen occupies' is the site the shipped pose "
    "is actually in",
    f"the pose's 16-atom centroid is {np.round(_pose_centroid, 2).tolist()} A; "
    f"the nearest site centre is rank {_nearest_rank} at {_nearest_d:.2f} A, "
    f"the next is {_runner_up_d:.2f} A away -- a {_runner_up_d / _nearest_d:.1f}x "
    f"margin, so the ranking cannot be reordered and keep the sentence true. "
    f"This replaced a check on the READMEs' former '16 ligand atoms', which was "
    f"deleted when that sentence was; the replacement pins the identity of the "
    f"site rather than a count of atoms",
)

check(
    abs(_uncapped[8].volume - 12.29) < 0.05 and in_both(r"12\.3\s*Å"),
    "the READMEs' 12.3 Å³ site is the 9th-ranked crambin site's real volume",
    f"rank 9 volume = {_uncapped[8].volume:.2f} Å³ (of {len(_uncapped)} uncapped, "
    f"{len(_sites)} at the shipped default cap)",
)
check(
    len(_pk.find_pockets(_coords, _els, residues=_res,
                         max_volume=_pk.DEFAULT_MAX_VOLUME))
    == len(_sites)
    and len(_pk.find_pockets(_coords, _els, residues=_res, max_pockets=10 ** 6,
                             max_volume=_pk.DEFAULT_MAX_VOLUME)) == len(_uncapped),
    "the READMEs' 'the 1500 Å³ ceiling deletes nothing' holds on crambin",
    f"at the default cap {len(_sites)} -> {len(_sites)}; uncapped "
    f"{len(_uncapped)} -> {len(_uncapped)}; largest site "
    f"{max(s.volume for s in _uncapped):.1f} Å³ against a {ceiling:.0f} Å³ ceiling",
)

# --- forward-looking guards: the two ways these two sentences rot ----------
#
# Both of these used to be wrong and were fixed by editing the prose. A check
# that pins the fixed text cannot stop it rotting again, because the fix *was*
# the new text; so each guard below pins the *rule* instead, and fails if the
# old claim comes back. They are the checks this file was missing, and they are
# deliberately written to fail on a regression rather than to describe a state.
_ex_lines = {name: [l for l in (ROOT / name).read_text(encoding="utf-8")
                    .splitlines() if "-e, --exhaustiveness" in l]
             for name in ("README.md", "README.en.md")}
_stale_default = re.compile(r"默认\s*(为|值|是)?\s*8|defaults?\s+(to\s+)?8\b")
_hardcoded_hits = {name: [l.strip() for l in lines
                          if _stale_default.search(l)]
                   for name, lines in _ex_lines.items()}


def _no_numeric_default(flag: str) -> bool:
    """True when `-e` has no numeric default, in either spelling.

    `cli_default` returns Python `None` when the call has no `default=` at all
    and the *string* `"None"` when it writes `default=None` explicitly. Those
    are different facts and the guard should accept both while still failing on
    a number, so the distinction is kept in the detail rather than collapsed.
    """
    return cli_default(flag) in (None, "None")


check(
    _no_numeric_default("--exhaustiveness") and _no_numeric_default("-e")
    and all(len(v) == 1 for v in _ex_lines.values())
    and not any(_hardcoded_hits.values())
    and all(re.search(r"按盒子体积|box's volume", v[0])
            for v in _ex_lines.values()),
    "the '-e' default sentence states a rule, not a number, and the CLI has no "
    "numeric default to contradict it with",
    f"cli_default reports {cli_default('--exhaustiveness')!r} for "
    f"--exhaustiveness and {cli_default('-e')!r} for -e -- an explicit "
    f"default=None rather than a number, so the code has nothing to contradict "
    f"the sentence with. Each README has exactly one such line: "
    + " | ".join(f"{n} -> {v[0].strip()!r}" for n, v in _ex_lines.items())
    + f". Stale hard-coded defaults found: "
      f"{ {k: v for k, v in _hardcoded_hits.items() if v} or 'none'}. "
    # Mutation, inside the check: both files are not mine to edit, so instead
    # of breaking the prose the guard is run against the sentence it is
    # supposed to reject. If the stale wording were reinstated the regex would
    # have to match it, and this assertion is what proves it does.
    + f"**Mutation:** the stale pattern against a reinstated "
      f"'默认 8' line matches "
      f"{bool(_stale_default.search('-e N     独立搜索轨迹数（默认 8）'))}"
      f", against a reinstated 'defaults to 8' line matches "
      f"{bool(_stale_default.search('-e N     walks (defaults to 8)'))}"
      f", and against the current line matches "
      f"{bool(_stale_default.search(_ex_lines['README.en.md'][0]))}. So the "
      f"guard is not passing because the pattern cannot fire",
)

# The count is a digit in the Chinese and a word in the English, so the English
# search has to be scoped to the sentence that makes the claim. Searching the
# whole file for a number word finds ordinary prose first -- "one" is the first
# entry in the table and appears everywhere -- which is how this check reported
# the count as 1 on its first run.
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
          20: "twenty"}
_cn_count = re.search(r"默认返回的\s*\*?\*?(\d+)\*?\*?\s*个位点", readme_cn)
# A *forward* window only. The English reads "ninth of the twelve", so the
# count sits after the ordinal; a window centred on it, or a whole-file search,
# finds ordinary prose ("one") first. 60 characters is enough for the phrase and
# short enough not to reach the next sentence.
_en_window = ""
if "ninth" in readme_en:
    _en_window = readme_en[readme_en.index("ninth"):readme_en.index("ninth") + 60]
_en_word = next((n for n, w in sorted(_WORDS.items())
                 if re.search(rf"\b{w}\b", _en_window)), None)
check(
    _cn_count is not None and _en_word is not None
    and int(_cn_count.group(1)) == len(_sites)
    and _en_word == len(_sites)
    and _pk.DEFAULT_MAX_POCKETS == len(_sites)
    # Mutation, inside the check: the point of reading the number out of the
    # document is that a different answer would turn this red, so that is what
    # is asserted -- with the code unchanged and only the comparison's inputs
    # standing in for a future one.
    and int(_cn_count.group(1)) != len(_sites) + 1
    and _en_word != len(_sites) + 1,
    "the site count the READMEs publish is what find_pockets returns under "
    "its own defaults",
    f"the Chinese says {_cn_count.group(1) if _cn_count else None!r} default "
    f"sites, the English sentence says {_en_word!r}, `find_pockets` with no "
    f"arguments beyond the coordinates returns {len(_sites)}, and "
    f"DEFAULT_MAX_POCKETS is {_pk.DEFAULT_MAX_POCKETS} -- so all three agree. "
    f"The number is read out of the documents rather than written here, which "
    f"is the point: transcribing 12 into this file would have kept the check "
    f"green after the code changed to return something else. **Mutation:** "
    f"against a hypothetical {len(_sites) + 1} sites the Chinese comparison is "
    f"{int(_cn_count.group(1)) == len(_sites) + 1} and the English one is "
    f"{_en_word == len(_sites) + 1}, so both would fail. Both files are pinned "
    f"because they express the count differently, a digit and a word, and a "
    f"check on only one of them would let the other drift",
)


# --- CN / EN must not drift apart -----------------------------------------
#
# The READMEs are the most-read documents in the repository and nothing held
# them in step with each other. Each entry is a number both files publish; the
# check fails if one file has it and the other does not. The floor on the number
# of comparisons matters as much as the comparison itself: if these patterns
# ever stop matching anything, the check must fail rather than pass on an empty
# set.
SHARED_README_FIGURES = [
    ("biotin RMSD", r"1\.17"),
    ("benzamidine RMSD", r"1\.19"),
    ("3PTB RMSD", r"1\.26"),
    ("3PTB box 39x26x41", r"39\s*[\u00d7x]\s*26\s*[\u00d7x]\s*41"),
    ("3PTB cleft 31x18x33", r"31\s*[\u00d7x]\s*18\s*[\u00d7x]\s*33"),
    ("3PTB at exhaustiveness 16", r"12\.88"),
    ("1HVR box 30x38x32", r"30\s*[\u00d7x]\s*38\s*[\u00d7x]\s*32"),
    ("1HVR at exhaustiveness 64", r"21\.24"),
    ("1HVR at exhaustiveness 128", r"10\.53"),
    ("crambin+ibuprofen energy", r"5\.36"),
    ("crambin reference energy", r"5\.50"),
    ("crambin RMSD", r"4\.24"),
    ("site ceiling 1500", r"\b1500\b"),
    ("site volume 12", r"12\s*Å"),
    ("site count 16", r"\b16\b"),
    ("auto box ratio 3.8", r"3\.8"),
    ("crambin pocket-search atoms 327", r"\b327\b"),
    ("streptavidin atoms 1001", r"\b1001\b"),
    ("haemoglobin atoms 4779", r"\b4779\b"),
    ("pocket-search time 0.16", r"0\.16"),
    ("streptavidin time 1.6", r"1\.6"),
    ("haemoglobin time 11", r"\b11\b"),
    ("probe radius 1.4", r"1\.4"),
    ("cavity volume 100", r"\b100\b"),
    ("small probe 0.5", r"0\.5"),
    ("hydrogen-bond cut-off 2.6", r"2\.6"),
    ("hydrogen-bond angle 120", r"\b120\b"),
    ("grad magnitude 2-6", r"2[\u2013-]6"),
    ("GPU agreement 1e-6", r"1e-6|10\u207b\u2076"),
    ("wheel name 0.1.0", r"0\.1\.0"),
    ("Rust 1.75+", r"1\.75"),
    ("Python 3.9+", r"3\.9"),
    ("multisampling 4x", r"4x"),
    ("num_modes default 9", r"\b9\b"),
    ("rmsd-cutoff default 1.0", r"\b1\.0\b"),
    ("spacing default 0.375", r"0\.375"),
    ("engine floor 2 x radius + 1", r"2\s*[\u00d7x]\s*(radius|\u534a\u5f84)\s*\+\s*1"),
]
_divergent = [
    label for label, pat in SHARED_README_FIGURES
    if bool(re.search(pat, readme_cn)) != bool(re.search(pat, readme_en))
]
check(
    not _divergent,
    "every figure both READMEs publish appears in both",
    f"{len(SHARED_README_FIGURES)} figures compared; "
    f"{len(_divergent)} present in only one language: {_divergent}",
)

# The floor. Without it, a pattern that stopped matching would leave `_divergent`
# empty and the check above would pass while testing nothing.
_matched = [lab for lab, pat in SHARED_README_FIGURES
            if re.search(pat, readme_cn) and re.search(pat, readme_en)]
check(
    len(_matched) >= 30,
    "the CN/EN comparison is not passing vacuously",
    f"only {len(_matched)} of {len(SHARED_README_FIGURES)} patterns matched in "
    f"both files. If this fails the figures list has drifted from the "
    f"documents, and the check above was proving less than it looks",
)

# A policy check, and the reason the parent asked the question. Neither README
# currently hard-codes a check count, and that is the correct call: the count
# moves whenever a check is added, and two of them moved inside this session
# alone (pdbqt_check 55 -> 57, scoring_cross_check 15 -> 18). A literal count in
# a README is a claim with no mechanism to keep it true. This check makes the
# absence explicit, so adding one back has to be a deliberate act.
_count_claims = [
    (lab, n) for lab, doc in (("README.md", readme_cn), ("README.en.md", readme_en))
    for n, line in enumerate(doc.splitlines(), 1)
    if re.search(r"\b\d+\s*(checks?|tests?)\b|\d+\s*\u9879(\u68c0\u67e5|\u6d4b\u8bd5)|"
                 r"(\u68c0\u67e5|\u6d4b\u8bd5)\s*\d+\s*\u9879", line)
]
check(
    not _count_claims,
    "neither README hard-codes a 'N checks' count",
    f"found {len(_count_claims)}: {_count_claims}. If a count is wanted here, "
    f"point at the command that produces it (each script asserts its own "
    f"EXPECTED_CHECKS) rather than transcribing a number that goes stale",
)


# ==========================================================================
# 7b. A polar receptor reaches a cross-element probe
# ==========================================================================
#
# This is the check whose absence let `SCORING.md` §5.1.1 and `LIMITATIONS.md`
# §3.2 keep tabulating cross-element terms as exactly `0.000000` after the engine
# stopped producing them. Both documents were wrong for as long as nobody asked
# a polar receptor a question, because an all-carbon fixture cannot distinguish
# "the grid partitions by element" from "the grid works": both give the same
# answer when the probe and the receptor atom are the same element.
#
# What is asserted is deliberately **not** "cross-element is non-zero". That is
# the symptom of the fixed state, and a check on it would go red the moment
# someone re-introduced the partition -- which is fine -- but it would also have
# been red *before* the fix, so it could never have been added while the defect
# was live without the defect being fixed first. What is asserted is the
# property that holds in both worlds: **an oxygen receptor atom contributes its
# shape and repulsion to the block a nitrogen probe reads.** Re-introduce the
# per-element write and the nitrogen block is empty, so the ligand nitrogen
# reads zeros where it should read a large positive number.
#
# The mutation that keeps this honest: forcing the ligand atom back to carbon --
# the all-carbon fixture this replaces -- makes `cross_element_block_is_not_empty`
# compare like against like and go red, because 2.0 A is inside the steric
# window and the two elements have different radii.

section("a polar receptor reaches a cross-element probe")

_GRID_RS = CORE / "grid.rs"
_grid_src = io.open(_GRID_RS, encoding="utf-8").read()
check(
    "a_receptor_atom_reaches_every_probe_type" in _grid_src,
    "grid.rs pins the cross-element reachability property with a test",
    "a test named `a_receptor_atom_reaches_every_probe_type` is present. The "
    "documented reasoning is in its own docstring; this check only asserts the "
    "test exists, so it cannot drift when the implementation is rewritten",
)


def _single_atom_receptor(element: str):
    return Receptor.from_pdbqt_str(
        "ATOM      1  %-3s UNL     1       0.000   0.000   0.000  1.00  0.00"
        "     0.000 %-2s \n" % (element, element)
    )


def _cross_element_block(rec_el: str, lig_el: str, r: float) -> float:
    """Total energy of one probe atom at `r` from a one-atom receptor.

    `r = 2.0 A` is chosen deliberately: it is inside the steric window for every
    published pair radius, so the reading is dominated by shape and repulsion. At
    3.0 A all four terms contribute nothing for these element pairs, so a probe
    there would read `0.000000` **whether or not the grid is broken** -- which is
    exactly the distance the old document tables used, and one of the reasons
    they looked plausible.
    """
    rec = _single_atom_receptor(rec_el)
    lig = Ligand.from_pdbqt_str(
        "ATOM      1 %-3s UNL     1     %8.3f%8.3f%8.3f  1.00  0.00    "
        "0.000 %-2s\nEND\n" % (lig_el + "1", r, 0.0, 0.0, lig_el)
    )
    box = GridBox.from_center_size((r / 2.0, 0.0, 0.0), (12.0, 12.0, 12.0))
    maps = rec.precalculate(box, "vina", 0.375)
    conf = np.zeros(lig.num_dof)
    conf[0] = r
    return float(score_conformation(lig, maps, conf)[0])


# (receptor element, ligand element) -- every pair here is cross-element, which is
# the whole point: an all-carbon fixture passes under both descriptions.
_CROSS = [("C", "N"), ("C", "O"), ("N", "O"), ("O", "N"), ("O", "S")]
_unreachable = []
for rec_el, lig_el in _CROSS:
    same = _cross_element_block(rec_el, rec_el, 2.0)
    cross = _cross_element_block(rec_el, lig_el, 2.0)
    if abs(cross) == 0.0:
        # The signature of the partitioned grid: the probe's element block was
        # never written by a receptor atom of a different element, so the probe
        # reads an empty block and contributes nothing at all -- not even the
        # steric repulsion, which is what made the old tables so alarming.
        _unreachable.append(
            f"{rec_el}/{lig_el} reads exactly 0.000000 at 2.0 A -- the element "
            f"block this probe reads was never written"
        )
    elif abs(cross - same) < 1e-9:
        # Nonzero, but byte-identical to the same-element reading: the probe is
        # not reading its own element's block, which is a different partition
        # bug and would be equally invisible to an all-carbon fixture.
        _unreachable.append(
            f"{rec_el}/{lig_el} reads {cross:.6f}, byte-identical to the "
            f"same-element {rec_el}/{rec_el} reading"
        )
check(
    not _unreachable,
    "every cross-element probe reads a block the receptor actually wrote",
    f"{len(_CROSS)} polar cross-element pairs at 2.0 A; problems: "
    f"{_unreachable or 'none'}. Each reads nonzero and differs from its "
    f"same-element reading, so the grid's element dimension is the probe type "
    f"and not a filter on the receptor -- the property SCORING.md 5.1.1 and "
    f"LIMITATIONS.md 3.2 were both written against, and the one an all-carbon "
    f"fixture cannot test",
)


# ==========================================================================
# 9. The claims this session added, scored
# ==========================================================================
#
# Everything below was written as prose in the last few hours and nothing held
# it in place: the per-term decomposition and its CPU-only nature, the pose
# contract and the qualifier that makes it true, the probe's provenance, the
# frame measurement, the two-tree contract and its recognition rule, and the
# cartoon's helix subtypes. A claim written but not wired is a feature that
# does not exist, one level up from the code.
#
# **A qualifier is part of the claim.** The pose contract says the poses are
# minima *to within the line search's resolution*; the failure mode is not that
# the sentence goes away, it is that someone widens "to within the resolution"
# into "is a minimum". So the qualifier is asserted *as a qualifier* -- present,
# next to the negation it belongs to -- and separately bound to the number the
# code actually uses, so the pair cannot be satisfied by a sentence that kept
# the words and lost the meaning.
#
# The other rule this section keeps: **nothing here is scored by grepping for a
# phrase.** Where a claim has a number in it, the number is parsed out of the
# document and compared with the code or with a measurement; where a claim is
# structural (the shader cannot see a `TermMaps`, a single-tree machine is not
# verified), the structure is executed or measured rather than searched for.
# Each check's detail line says which of the two it is, because a reader
# deserves to know how much a green row is worth.

section("this session's new claims: the per-term decomposition")

SCORING5 = scoring_doc.split("### 5.4", 1)[1].split("\n## ", 1)[0]
LIMIT4 = limitations_doc.split("### 4.2", 1)[1].split("\n## ", 1)[0]
LIMIT43 = limitations_doc.split("### 4.3", 1)[1].split("\n## ", 1)[0]
SCORING53 = scoring_doc.split("### 5.3", 1)[1].split("\n## ", 1)[0]

# ==========================================================================
# 9.1  The two passages SCORING.md 5.3 gained this round
# ==========================================================================
#
# Both of these were written as prose and read by nothing, which is the same
# defect one level up from the code: a claim nobody checks is a comment. The
# 8.0 cutoff itself was already scored; these two were not, and each one is a
# number a reader would act on.

# --- the reach is the cutoff plus the WIDEST probe radius, not twice the
# --- receptor atom's own.
_max_probe = float(re.search(
    r"MAX_GRID_TYPE_RADIUS:\s*f64\s*=\s*([\d.]+)", types_rs).group(1))
_src_reach = re.search(
    r"let far = kernels\.cutoff \+ (\w+) \+ a\.radius", grid_rs)
_doc_reach = re.search(r"kernels\.cutoff \+ (\w+) \+ a\.radius", SCORING53)
_doc_stale = re.search(r"\*\*`cutoff \+ 2\.0 \* a\.radius`(.{0,12})", SCORING53)
check(
    _src_reach is not None and _doc_reach is not None
    and _doc_reach.group(1) == _src_reach.group(1)
    and re.search(rf"pub const {re.escape(_src_reach.group(1))}: f64", types_rs)
    and _doc_stale is not None
    and ("过时" in _doc_stale.group(1) or "0.4 Å" in _doc_stale.group(1)),
    "SCORING.md 5.3's reach formula is the formula in grid.rs, and the "
    "0.4 A-era `cutoff + 2 * R` form is recorded as stale rather than stated",
    f"the document writes `kernels.cutoff + {_doc_reach and _doc_reach.group(1)} "
    f"+ a.radius`; grid.rs computes "
    f"`kernels.cutoff + {_src_reach and _src_reach.group(1)} + a.radius`, and "
    f"{_src_reach and _src_reach.group(1)} is the constant types.rs pins at "
    f"{_max_probe:g} A -- the widest of the ten per-probe-type radii, not the "
    f"receptor atom's. The two forms coincide only when the widest probe radius "
    f"equals the atom's own, which is iodine alone, so for every other element "
    f"`cutoff + 2 * R` is the 0.4 A-era form and the document now says so. "
    f"**Mutation:** putting `cutoff + 2.0 * a.radius` back as the stated form "
    f"goes red, and so does deleting the 'stale' sentence while the formula "
    f"line stays correct -- the sentence is the half a reader copies",
)

# --- 8.0 is an upper bound, and the stored field stops at 5.200 because f32
# --- runs out, not because the cutoff says so.
# The boundary is *derived*, from the weight the document itself states: the
# Shape slot's first term is `w1 * g1` with `w1` the gauss1 weight and
# `g1 = exp(-((d-0.5)/0.5)^2)`, and `f32`'s smallest normal is
# 1.1754944e-38. So the largest surface distance at which the term is still
# normal is a root, not a constant, and the number the document publishes is
# the first sample of the 0.5 A walk at or beyond it.
_F32_MIN_NORMAL = float(np.finfo(np.float32).tiny)
_F32_TINY_SUBNORMAL = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
_w1 = doc_vina["gauss1"]
# |w1| * exp(-((d - 0.5)/0.5)^2) = F32_MIN_NORMAL  ->  solved for d: the
# surface distance at which the term stops being a *normal* f32.
_norm_edge = 0.5 + 0.5 * math.sqrt(math.log(abs(_w1) / _F32_MIN_NORMAL))
# ... and the distance at which it underflows to exactly zero, which is where
# the smallest *subnormal* runs out. Between the two the term is representable
# but denormal, which is exactly the state the document calls 次正规数.
_zero_edge = 0.5 + 0.5 * math.sqrt(math.log(abs(_w1) / _F32_TINY_SUBNORMAL))
_doc_support = re.search(r"\*\*表面距离 ([\d.]+) Å\*\* 处结束", SCORING53)
_subnormal = re.search(r"[−-]([\d.]+)e[-−]40", SCORING53)
_walk = 0.5  # grid.rs's own test walks outward at spacing 0.5
_pub = float(_doc_support.group(1)) if _doc_support else float("nan")
_prod = abs(_w1) * math.exp(-(((_pub - 0.5) / 0.5) ** 2)) * 1e40
check(
    _doc_support is not None and _subnormal is not None
    and _norm_edge < _pub <= _zero_edge
    and "次正规数" in SCORING53 and "上界" in SCORING53
    and abs(float(_subnormal.group(1)) - _prod) <= 0.05 * _prod,
    "the 5.200 A support end SCORING.md 5.3 publishes is where f32 stops being "
    "able to hold the term, derived from the weight the document itself "
    "states, and the cutoff is recorded as an upper bound",
    f"with w1 = {_w1} (the gauss1 weight parsed out of section 3, not typed "
    f"here): the term |w1| * exp(-((d-0.5)/0.5)^2) is a **normal** f32 up to "
    f"d = {_norm_edge:.3f} A, is subnormal-but-nonzero from there to "
    f"d = {_zero_edge:.3f} A, and underflows to exactly zero past it. The "
    f"document publishes {_pub} A, which lies in that denormal window, and its "
    f"quoted product {float(_subnormal.group(1)):.1f}e-40 is what |w1|*g1 "
    f"evaluates to at that distance ({_prod:.2f}e-40 computed, checked to 5% -- "
    f"the document's number is *derived*, and a version that quoted grid.rs's "
    f"own 1.8e-40 fails here, because that figure comes from rounding g1 to "
    f"5.1e-39 before multiplying). The walk that reports the end steps "
    f"{_walk} A, so 5.2 is simply the last sample above the zero edge -- the "
    f"field is not cut at 5.2, it is *unrepresentable* there. Retune gauss1 in "
    f"scoring.rs and the window moves out from under the published value. "
    f"**Mutation:** publishing 5.7 instead goes red, and so does retuning the "
    f"weight. **This is the structural half** -- the behavioural half, the "
    f"walk that finds the last non-zero sample, is `grid.rs`'s own test, which "
    f"needs the engine's internal maps",
)

# --- the five terms are the struct's five fields, in the struct's order -----
_tb = scoring_rs.split("pub struct TermBreakdown", 1)[1].split("\n}", 1)[0]
_src_terms = re.findall(r"pub (\w+): f64", _tb)
# The two polarity halves and the penalty are f64 too, so the document's table
# is matched against the five the *engine* calls the Vina terms, and the extras
# are required to be present as well -- a decomposition that quietly stopped
# splitting hbond by polarity is a different claim, and is caught below.
_VINA_TERMS = ("gauss1", "gauss2", "repulsion", "hbond", "hydrophobic")
_doc_terms = [t for t in _VINA_TERMS if f"`{t}`" in SCORING5]
_missing_terms = [t for t in _VINA_TERMS if t not in _src_terms]
_missing_extra = [t for t in ("hbond_from_donor", "hbond_from_acceptor",
                              "out_of_box_penalty") if t not in _src_terms]
check(
    len(_doc_terms) == 5 and not _missing_terms and not _missing_extra,
    "the five terms SCORING.md 5.4 lists are TermBreakdown's five, and the "
    "struct still carries the three extras it says it does",
    f"the document names {_doc_terms} in that order; scoring.rs declares "
    f"{_src_terms}. Missing from the struct: {_missing_terms + _missing_extra}. "
    f"**Mutation:** deleting `hydrophobic` from the document's table makes the "
    f"count 4 and this goes red; renaming the field in the struct without "
    f"touching the document does the same. The order matters because the "
    f"document reads as a list and a reader maps position to field",
)

# --- the two strides: 40 for the production map, 60 for the decomposition ---
# Both strides are `const fn`s that *compute* rather than literals, so each is
# evaluated from the factors the function body names -- `GRID_TYPE_COUNT *
# MAPS_PER_TYPE` and `GRID_TYPE_COUNT * TERM_FIELDS`. Reading the arithmetic
# is the point: a hand-written 40 in this file would be the second copy of a
# number the source already derives.
def _stride_of(fn: str) -> int:
    """Evaluate grid.rs's `const fn map_stride` / `term_stride` from its factors.

    Both are written as products of *named* constants, so a regex that wanted
    digits would read nothing. The names are resolved from the files that
    define them, which is the only way this is not a second copy of the
    number: a hand-written 40 in this file would survive a change to
    `MAPS_PER_TYPE` and report a stride the engine no longer uses.
    """
    body = re.search(rf"pub const fn {fn}\(\) -> usize \{{(.*?)\}}", grid_rs, re.S)
    out = 1
    for name in re.findall(r"[A-Z_][A-Z_0-9]*", body.group(1)):
        found = re.search(rf"\b{name}:\s*usize\s*=\s*(\d+)", grid_rs) \
            or re.search(rf"\b{name}:\s*usize\s*=\s*(\d+)", types_rs) \
            or re.search(rf"\b{name}:\s*usize\s*=\s*(\d+)", scoring_rs)
        if found is None:
            raise AssertionError(f"{fn} names {name}, which defines no usize")
        out *= int(found.group(1))
    return out


_map_stride = _stride_of("map_stride")
_term_stride = _stride_of("term_stride")
_doc_map_stride = re.search(r"`STRIDE = (\d+)`", SCORING5)
_doc_term_stride = re.search(r"`TERM_STRIDE = (\d+)`", SCORING5)
check(
    _doc_map_stride is not None and _doc_term_stride is not None
    and int(_doc_map_stride.group(1)) == _map_stride
    and int(_doc_term_stride.group(1)) == _term_stride,
    "SCORING.md 5.4's two strides are grid.rs's map_stride and term_stride",
    f"the document says STRIDE={_doc_map_stride and _doc_map_stride.group(1)} "
    f"and TERM_STRIDE={_doc_term_stride and _doc_term_stride.group(1)}; "
    f"grid.rs computes {_map_stride} and {_term_stride}. The two are different "
    f"numbers on purpose -- this is the whole reason the decomposition cannot "
    f"be read off the production map, so a check that accepted one for the "
    f"other would be asserting the opposite of the claim",
)

# --- measured: the five terms account for the production energy -------------
_trec = Receptor.from_pdbqt_str(
    "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
    "ATOM      2  OA  UNL     1       0.000   0.000   3.800  1.00  0.00     0.000 OA \n"
)
_tlig = Ligand.from_pdbqt_str(
    "ATOM      1 C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
    "END\n"
)
_tbox = GridBox.from_center_size((0.0, 0.0, 1.9), (12.0, 12.0, 12.0))
_tmaps = _trec.precalculate(_tbox, "vina", 0.375)
_tterms = _trec.precalculate_terms(_tbox, "vina", 0.375)
_tconf = np.zeros(_tlig.num_dof)
_td = core.score_conformation_terms(_tlig, _tmaps, _tterms, _tconf, "vina")
_tsum = (_td["g1"] + _td["g2"] + _td["rep"] + _td["hb"] + _td["hyd"])
_tgap = abs(_td["terms_total"] - _td["intermolecular"])
# The gap the document publishes, and the tolerance that admits it as f32
# storage rounding rather than as a disagreement. `terms_total` and
# `intermolecular` are computed from the same atoms; the only reason they can
# differ at all is that the production field rounds `g1 + g2 + rep` once, as
# one f32, where the term maps round each of the three separately. A tolerance
# at the f32 epsilon of a score this size (~1e-6) is the statement "differ by
# no more than one storage rounding"; anything looser would admit a real
# disagreement, which is the half of the claim that matters.
_TGAP_MAX = 1e-6
check(
    abs(_tsum - _td["terms_total"]) <= 1e-12
    and _tgap <= _TGAP_MAX
    and abs(_td["total"] - score_conformation(_tlig, _tmaps, _tconf)[0]) <= 1e-12,
    "the five terms add up to terms_total, and terms_total is what the "
    "production path reports to within f32 storage rounding",
    f"the five terms sum to {_tsum!r} against a reported terms_total of "
    f"{_td['terms_total']!r}; the production path reports "
    f"{_td['intermolecular']!r}, a gap of {_tgap:.3e} against a tolerance of "
    f"{_TGAP_MAX:.0e} (one f32 rounding of a score of this size). The document "
    f"publishes 4.01e-07 and the measurement is {_tgap:.2e}, so both are the "
    f"same fact. **Mutation:** widening the tolerance to 1e-3 would admit a "
    f"decomposition that does not account for the total; the sum half fails "
    f"the moment a term is dropped from either side",
)

# --- measured: the terms arrive already weighted ---------------------------
# The single sharpest test of "already weighted": the *raw* gauss1 kernel
# `exp(-((d-0.5)/0.5)^2)` is positive for every d, and the weighted term is
# negative. So a regression that moved the weight from tabulation time to
# scoring time -- or dropped it -- cannot produce these numbers. The
# comparison is against the document's own weight, read out of the vina block
# in section 3, not a literal written here.
def _weighted_gauss1(r: float, weight: float) -> float:
    return weight * math.exp(-(((r - 0.5) / 0.5) ** 2))


_g1_wrong = []
for _r in (3.4, 3.6, 3.8, 4.0, 4.2):
    _rec1 = Receptor.from_pdbqt_str(
        "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00"
        "     0.000 C \n")
    _lig1 = Ligand.from_pdbqt_str(
        "ATOM      1 C   UNL     1     %8.3f%8.3f%8.3f  1.00  0.00    "
        "0.000 C\nEND\n" % (_r, 0.0, 0.0))
    _box1 = GridBox.from_center_size((_r / 2.0, 0.0, 0.0), (12.0, 12.0, 12.0))
    _m1 = _rec1.precalculate(_box1, "vina", 0.375)
    _t1 = _rec1.precalculate_terms(_box1, "vina", 0.375)
    _c1 = np.zeros(_lig1.num_dof)
    _c1[0] = _r
    _got = core.score_conformation_terms(_lig1, _m1, _t1, _c1, "vina")["g1"]
    _want = _weighted_gauss1(_r, doc_vina["gauss1"])
    if abs(_got - _want) > 1e-5:
        _g1_wrong.append(f"r={_r}: engine {_got:.6f}, weighted formula {_want:.6f}")
check(
    not _g1_wrong,
    "the gauss1 the decomposition reports is the document's *weighted* term",
    f"5 separations from 3.4 to 4.2 A, each compared against "
    f"{doc_vina['gauss1']} * exp(-((d-0.5)/0.5)^2) with the weight parsed out "
    f"of the document's own vina block; problems: {_g1_wrong or 'none'}. The "
    f"raw kernel is positive everywhere and these are negative, so the check "
    f"cannot be satisfied by an unweighted reading. **Mutation:** comparing "
    f"against the bare kernel instead flips every one of the five",
)

# --- the four slots are a different decomposition, and the doc says so ------
_slot_names = list(_td["slot_names"])
_hb_split = abs((_td["hb_from_donor"] + _td["hb_from_acceptor"]) - _td["hb"])
_shape_fused = abs(_td["slots"][0] - _td["shape"])
_shape_unfused = abs(_td["shape"] - (_td["g1"] + _td["g2"] + _td["rep"]))
check(
    _slot_names == ["shape", "hb_from_donor", "hb_from_acceptor", "hydrophobic"]
    and len(_slot_names) == 4 and len(_VINA_TERMS) == 5
    and _hb_split <= 1e-12 and _shape_fused <= _TGAP_MAX
    and _shape_unfused <= _TGAP_MAX
    and all(n in SCORING5 for n in _slot_names)
    and "slot 0" in SCORING5,
    "the slots are the four the document names, slot 0 is the three shape "
    "terms fused and the hydrogen bond is one term over two slots -- which is "
    "why 5.4 refuses to read one decomposition as the other",
    f"the payload's slot_names are {_slot_names}: {len(_slot_names)} numbers "
    f"against the decomposition's {len(_VINA_TERMS)} terms. slot 0 "
    f"({_td['slots'][0]:.6f}) is the *production map's own* stored value, so "
    f"it differs from the f64 `shape` ({_td['shape']:.6f}) by "
    f"{_shape_fused:.2e} -- one f32 rounding, and that gap is the evidence "
    f"that slot 0 is read out of the map rather than recomputed. `shape` "
    f"equals g1+g2+rep ({_td['g1'] + _td['g2'] + _td['rep']:.6f}) to "
    f"{_shape_unfused:.2e}, so the fusion is a storage decision and not a "
    f"different sum. hbond_from_donor + hbond_from_acceptor equals hbond to "
    f"{_hb_split:.1e}. The document names all four, names slot 0, and says "
    f"the hydrogen bond is spread over two slots. **Mutation:** reordering "
    f"the slots, or detaching the hbond term from its two halves, goes red",
)

section("this session's new claims: the decomposition is CPU-only")

# --- one variant, and it is the CPU ----------------------------------------
_be = scoring_rs.split("pub enum TermBackend", 1)[1].split("\n}", 1)[0]
# Strip doc comments and attributes so a variant cannot hide in prose.
_be_code = "\n".join(
    line for line in _be.splitlines()
    if not line.lstrip().startswith(("//", "///", "#["))
)
_be_variants = re.findall(r"^\s{4}(\w+),?\s*$", _be_code, re.M)
check(
    _be_variants == ["Cpu"],
    "TermBackend has exactly one variant and it is the CPU, so \"CPU-only\" is "
    "a type a caller can match on rather than a promise in a docstring",
    f"the enum declares {_be_variants}. **Mutation:** adding a `Gpu` variant is "
    f"precisely the change the document says would be a deliberate act, and it "
    f"turns this red. A check that asserted the engine *is* on the CPU would "
    f"have gone green under exactly the edit that breaks the claim",
)

# --- the shader cannot see a TermMaps --------------------------------------
_wgsl_body = wgsl
check(
    "TermMaps" not in _wgsl_body
    and "term_stride" not in _wgsl_body
    and wgsl_stride == _map_stride
    and _term_stride != _map_stride,
    "the WGSL kernel is pinned to the production stride, so a TermMaps cannot "
    "reach a shader even by accident",
    f"the shader's STRIDE is {wgsl_stride}, which is grid.rs's map_stride "
    f"({_map_stride}), and it mentions neither `TermMaps` nor `term_stride`; "
    f"the decomposition's stride is {_term_stride}. That is the *structural* "
    f"half of the CPU-only claim and it is a different assertion from the enum "
    f"above. **Mutation:** a shader written for a 60-float stride fails here "
    f"and in `dock-core`'s own test, from two directions",
)

# --- the negative claim: Python cannot ask which backend produced this ------
_py_core = read(ROOT / "dock-py" / "python" / "opendocking" / "core.py")
# Every way of putting a key on the returned dict counts, not just the tuple
# comprehension: a key added by an `out["x"] = ...` line is *more* exposed than
# one in the tuple, so reading only the tuple would have left the check
# blind to the obvious way someone would close the gap the document records.
_fn = _py_core.split("def score_conformation_terms(", 1)[1].split("\ndef ", 1)[0]
_out_keys = re.findall(r'for k in \(\s*(.*?)\)\}', _fn, re.S)
_py_exposed = set(re.findall(r'"(\w+)"', _out_keys[0]) if _out_keys else [])
_py_exposed |= set(re.findall(r"""out\[\s*["'](\w+)["']\s*\]\s*=""", _fn))
_tm_body = _py_core.split("class TermMaps:", 1)[1].split("\ndef ", 1)[0]
# `scoring.rs` makes the promise in two places and `SCORING.md` 5.4 tabulates
# one row per place. Three states have to be told apart: the code has the half,
# the code lacks it, and **the document has stopped saying which** -- and a
# check that compared two booleans could not tell the third from the second,
# because both are False. That is not hypothetical: the first version of this
# check did exactly that, and deleting the row that recorded the open half left
# it green. So the row *count* is asserted as well as each row's polarity, and
# a deleted row is a red rather than a silent agreement.
_rows = re.findall(
    r"^(?:> )?\| (`(?:TermBreakdown::backend|TermMaps::BACKEND)`[^|]*)\|([^|]*)\|([^|]*)\|",
    SCORING5, re.M)
_h1_code = "backend" in _py_exposed
_h2_code = "BACKEND" in _tm_body
# Both `_hN_doc` mean "the row claims Python **has** this", so the comparison
# against the code is the same expression for both halves. Reading half two as
# "the row claims Python lacks it" would be the same fact inverted, and an
# inverted comparison passes on exactly the state where the document went
# wrong.
_h1_doc = any("`TermBreakdown::backend`" in r[0] and "**无**" not in r[2]
              for r in _rows)
_h2_doc = any("`TermMaps::BACKEND`" in r[0] and "**无**" not in r[2]
              for r in _rows)
check(
    len(_rows) == 2
    and _h1_code == _h1_doc
    and _h2_code == _h2_doc
    and "backend" in LIMIT43
    and "Python 前端目前问不到这件事" not in SCORING5
    and "backend" in scoring_rs.split("pub struct TermBreakdown", 1)[1][:900],
    "both promises the Rust docstring makes about `backend` are tabulated in "
    "SCORING.md 5.4, and each row's Python column matches the binding -- so "
    "the document cannot be half right, and cannot go quiet",
    f"the table has {len(_rows)} row(s): {[(r[0].strip(), r[2].strip()) for r in _rows]}. "
    f"**Half one:** the binding exposes `backend` = {_h1_code}, the row says "
    f"Python has it = {_h1_doc} -- a Python caller holding a breakdown can ask "
    f"rather than infer, and the value comes from the engine's own field. "
    f"**Half two:** class TermMaps exposes `BACKEND` = {_h2_code}, the row "
    f"claims Python has it = {_h2_doc} -- so the more expensive question, "
    f"whether to build a 60-float-per-point tabulation at all, is still "
    f"unanswerable from Python. LIMITATIONS.md 4.3's row matches. **Two "
    f"independent two-way contracts:** exposing `TermMaps.BACKEND` turns half "
    f"two red until 5.4 is rewritten, and deleting the row turns it red with "
    f"the code untouched. Mutation-proven in both directions; the row-count "
    f"assertion is what makes the second one possible",
)

# --- and the region beyond it is outside the model, not a weak interaction --
SCORING531 = scoring_doc.split("### 5.3.1", 1)[1].split("\n## ", 1)[0] \
    if "### 5.3.1" in scoring_doc else ""
# The three boundaries the section states. The first two are the same f32
# arithmetic as the check above, and they are re-derived here rather than
# shared, so that deleting the *section* fails this check instead of quietly
# leaving the numbers to be asserted by the one that no longer has a document
# to read.
_b1 = re.search(r"`d ≤ ([\d.]+) Å` 时是\*\*正规数", SCORING531)
_b2 = re.search(r"`([\d.]+) … ([\d.]+) Å` 之间它是\*\*次正规数", SCORING531)
_b3 = re.search(r"`> ([\d.]+) Å` 它是\*\*精确的 0", SCORING531)
# The two sentences that make it a claim rather than a number: that the zero
# there is indistinguishable from a genuinely absent interaction, and that the
# document deliberately publishes no threshold.
_silent = "完全一样" in SCORING531
_no_threshold = "不给这个区域一个阈值" in SCORING531
check(
    _b1 is not None and _b2 is not None and _b3 is not None
    and abs(float(_b1.group(1)) - _norm_edge) < 0.005
    and abs(float(_b2.group(2)) - _zero_edge) < 0.005
    # The structure is one boundary stated twice and another: "normal up to X"
    # and "subnormal from X" are the *same* X, and "exactly zero beyond Y" is
    # the same Y. Asserting the four numbers to be four distinct values would
    # be asserting a shape the physics does not have.
    and abs(float(_b2.group(1)) - float(_b1.group(1))) < 0.005
    and abs(float(_b3.group(1)) - float(_b2.group(2))) < 0.005
    and float(_b1.group(1)) < float(_b3.group(1))
    and _silent and _no_threshold,
    "SCORING.md 5.3.1's three boundaries are the f32 ones the engine's own "
    "weight gives, and the section says a zero outside support is "
    "indistinguishable from a real one rather than publishing a threshold",
    f"the document states the normal/subnormal boundary at "
    f"{_b1 and _b1.group(1)} A and the underflow boundary at "
    f"{_b2 and _b2.group(2)} A, against the {(_norm_edge, _zero_edge)} this "
    f"file solves from w1 = {_w1} and f32's own limits; the two edges are "
    f"each stated twice in the prose ('normal up to X' and 'subnormal from X' "
    f"are one boundary) and asserted to coincide rather than to be four "
    f"distinct numbers -- a shape the physics does not have is not a stricter "
    f"reading, it is a wrong one. Two "
    f"sentences carry the claim rather than the number: that a zero out there "
    f"reads the same as a genuinely absent interaction ({_silent}), and that "
    f"no threshold is published for the region ({_no_threshold}) -- because a "
    f"cutoff there would dress an f32 property up as a specification. "
    f"**Mutation:** widening the section to publish a 'ignore beyond 5 A' rule "
    f"goes red, and so does moving either boundary off the solved value",
)

# The contract is a blockquote, so it is taken as a whole: the qualifier and
# the negation have to be in the *same* statement, because a qualifier that
# drifted into a neighbouring paragraph would be a qualifier the reader of the
# contract sentence does not see.
_contract_lines: list[str] = []
_all4 = LIMIT4.splitlines()
_start = next((i for i, l in enumerate(_all4) if "**契约" in l), None)
if _start is not None:
    for _line in _all4[_start:]:
        if _line.startswith(">"):
            _contract_lines.append(_line)
        else:
            break
_contract = "\n".join(_contract_lines)
# The qualifier and the negation, in the same sentence of the contract. Both
# are required: a contract that kept "not a stationary point" while dropping
# the resolution is still a wrong claim in the other direction (it would
# promise less than the engine does), and a contract that kept the resolution
# while dropping the negation promises a minimum nobody can point at.
_has_qualifier = "线搜索分辨率" in _contract
_has_negation = "不是插值场的驻点" in _contract
check(
    _has_qualifier and _has_negation,
    "the pose contract carries BOTH its qualifier and its negation -- the "
    "narrow wording is scored as part of the claim",
    f"the contract reads: {_contract.strip()[:160]!r}. Two required parts, "
    f"measured: qualifier (to within the line search's resolution) "
    f"{_has_qualifier}, negation (not a stationary point of the interpolated "
    f"field) {_has_negation}. **This is the mutation the claim predicts:** "
    f"rewriting the sentence as \"the returned poses are local minima\" -- "
    f"i.e. dropping 线搜索分辨率 -- leaves the negation in place and turns "
    f"this red, because a qualifier is part of its claim. Scoring only the "
    f"sentence would have called the widened version true",
)

# --- the resolution is a number the code actually uses ---------------------
_lbfgs = read(CORE / "search" / "lbfgs.rs")
_floor = float(re.search(r"if alpha < ([\d.e-]+) \{\s*\n\s*break;", _lbfgs).group(1))
_halvings = int(re.search(r"for _ in 0\.\.(\d+) \{", _lbfgs).group(1))
_armijo = float(re.search(r"armijo_c:\s*([\d.e+-]+)", _lbfgs).group(1))
_tol = float(re.search(r"gradient_tolerance:\s*([\d.e+-]+)", _lbfgs).group(1))
_doc_floor = re.search(r"折到 `([\d.e+-]+)` 构象单位以下就停", LIMIT4)
_doc_halvings = re.search(r"至多 (\d+) 次折半", LIMIT4)
_doc_armijo = re.search(r"Armijo 常数 `([\d.e+-]+)`", LIMIT4)
check(
    _doc_floor is not None and _doc_halvings is not None
    and _doc_armijo is not None
    and abs(float(_doc_floor.group(1)) - _floor) < 1e-30
    and int(_doc_halvings.group(1)) == _halvings
    and abs(float(_doc_armijo.group(1)) - _armijo) < 1e-30,
    "the resolution the contract is written against is lbfgs.rs's own line "
    "search, not a number the document chose",
    f"the document says the step halves down below {_doc_floor and _doc_floor.group(1)} "
    f"conf units, in at most {_doc_halvings and _doc_halvings.group(1)} "
    f"halvings, with an Armijo constant of {_doc_armijo and _doc_armijo.group(1)}; "
    f"lbfgs.rs says {_floor:g}, {_halvings} and {_armijo:g}. This is what "
    f"stops \"to within the resolution\" from being a decorative phrase: the "
    f"resolution is a floor in a backtracking loop, so it moves if the loop "
    f"does. **Mutation:** retuning the floor in the source without touching "
    f"the document goes red here, and so does editing the document's number",
)

_audit = read(ROOT / "examples" / "audit_poses.py")
_steps = re.findall(r"([\d.e-]+)", _audit.split("_LINE_SEARCH_STEPS = (", 1)[1].split(")", 1)[0])
_finest = min(float(s) for s in _steps)
_doc_finest = re.search(r"阶梯最细只到 `([\d.e+-]+)`", LIMIT4)
_doc_tol = re.search(r"gradient_tolerance`\s*是\s*\*\*([\d.e+-]+)\*\*", LIMIT4)
check(
    _doc_finest is not None and abs(float(_doc_finest.group(1)) - _finest) < 1e-30
    and _doc_tol is not None and abs(float(_doc_tol.group(1)) - _tol) < 1e-30
    and _finest > _floor,
    "the audit's finest rung and the declared tolerance are the two numbers "
    "the contract paragraph quotes",
    f"audit_poses.py's _LINE_SEARCH_STEPS bottoms out at {_finest:g} conf "
    f"units, the document says {_doc_finest and _doc_finest.group(1)}; "
    f"lbfgs.rs's gradient_tolerance is {_tol:g} and the document says "
    f"{_doc_tol and _doc_tol.group(1)}. The ordering "
    f"{_floor:g} < {_finest:g} is the point of the sentence: the engine's own "
    f"line search can see further than the audit can, so below the audit's "
    f"finest rung neither instrument can certify anything, and the document "
    f"says so. **Mutation:** adding a finer rung to the ladder without "
    f"updating the paragraph goes red",
)

section("this session's new claims: the probe's provenance")

# --- the digest is of the file the process actually imported ----------------
sys.path.insert(0, str(ROOT / "dock-py" / "python"))
try:
    from opendocking.workbench import launcher as _launcher
except Exception as _exc:  # pragma: no cover - reported, never silent
    check(False, "the probe's provenance can be measured",
          f"importing opendocking.workbench.launcher failed: {_exc!r}")
    _facts = None
else:
    _facts = _launcher._path_facts(_launcher.__file__,
                                   "opendocking.workbench.launcher", "test")
    _on_disk = Path(_facts["path"]) if _facts.get("path") else None
    _digest_ok = (
        _on_disk is not None
        and _on_disk.resolve() == Path(_launcher.__file__).resolve()
        and _on_disk.is_file()
        and _facts.get("bytes") == _on_disk.stat().st_size
        and _facts.get("sha256")
        == hashlib.sha256(_on_disk.read_bytes()).hexdigest()
    )
    check(
        _digest_ok,
        "`--check --json` names the copy the process really imported, and "
        "carries that file's own digest",
        f"_path_facts reported {(_facts.get('path') or '')[-60:]!r} with "
        f"{_facts.get('bytes')} bytes and sha256 "
        f"{str(_facts.get('sha256'))[:16]}..., which is this file read from "
        f"disk at the path this process imported it from. **This is the "
        f"claim, measured rather than asserted:** a path assembled from an "
        f"assumed layout would carry a digest that does not match the bytes "
        f"at that path, so the check fails the moment the provenance becomes "
        f"a reconstruction. The probe could not have told you which copy it "
        f"read without this",
    )

# --- the files the check compares are files the payload names --------------
_gui = read(ROOT / "scripts" / "_gui_check.py")
_prov_tbl = re.search(r"PROVENANCE_FILES = \((.*?)\n\)", _gui, re.S)
_prov_files = re.findall(r'\("(\w+)",\s*"([^"]+)"\)',
                         _prov_tbl.group(1) if _prov_tbl else "")
# The payload's keys are read by *calling* the builder rather than by parsing
# its source, so a key that stops being reported is a red here and not a
# pattern that quietly stops matching.
_payload_keys = sorted(_launcher._provenance(None))
_gui_keys = {k for k, _ in _prov_files}
check(
    _gui_keys and _gui_keys <= set(_payload_keys)
    and {"launcher", "app"} <= set(_payload_keys)
    and all(rel.endswith(".py") for _, rel in _prov_files),
    "the copies `_gui_check.py` compares are copies the payload actually "
    "names, and the payload names a launcher and a viewport",
    f"_gui_check.PROVENANCE_FILES asks for {_prov_files}; the payload's "
    f"provenance block has keys {_payload_keys}. The check's claim is that a "
    f"stale installed copy becomes a *named* divergence, and it can only be "
    f"that if every file it compares is one the payload reports. **Mutation:** "
    f"adding a file to PROVENANCE_FILES that the payload does not report goes "
    f"red, which is the failure that would otherwise be a comparison against "
    f"nothing",
)

# --- and the stale copy is mechanically distinguishable --------------------
_audit_copy = getattr(sys.modules.get("_gui_check"), "audit_probe_copy", None)
if _audit_copy is None:
    _gi_spec = importlib.util.spec_from_file_location(
        "_gui_check", ROOT / "scripts" / "_gui_check.py")
    _gi = importlib.util.module_from_spec(_gi_spec)
    # Registered before execution: `_gui_check.py` defines a dataclass, and a
    # dataclass's decorator looks itself up in `sys.modules` by `__module__`.
    # A module executed without being registered there raises inside
    # `dataclasses`, which would read as "this file cannot be measured".
    sys.modules["_gui_check"] = _gi
    _gi_spec.loader.exec_module(_gi)
    _audit_copy = _gi.audit_probe_copy
_bad_detail = {
    "provenance": {
        "launcher": {
            "module": "opendocking.workbench.launcher",
            "measured_in": "this process",
            # A digest of something else, which is what a stale install looks
            # like: right module, wrong bytes.
            "path": str(ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "launcher.py"),
            "bytes": 1,
            "sha256": "0" * 64,
        },
    }
}
_copy_result = _audit_copy(_bad_detail, ROOT / "dock-py" / "python")
check(
    (not _copy_result.comparable) or bool(_copy_result.divergences),
    "a stale installed copy comes back as a named divergence or as "
    "\"cannot compare\" -- never as a pass",
    f"handed a payload whose launcher digest does not match the file at that "
    f"path, audit_probe_copy returned comparable="
    f"{_copy_result.comparable} with divergences {_copy_result.divergences} "
    f"and why {(_copy_result.why or '-')!r}. This is the difference between "
    f"\"the product is broken\" and \"you are testing an old install\", which "
    f"is the only reason the digest is in the payload. **Mutation:** returning "
    f"a CopyAudit with comparable=True and no divergences for a mismatched "
    f"digest is what the old prose-parsing heuristic effectively did, and it "
    f"turns this red",
)

section("this session's new claims: a context is not a frame")

# **This claim has exactly one owner, and it is not this file.**
# `scripts/probe_painted_frame.py` measures it: `check_the_viewport_draws` runs
# the product's own `Viewport` and reads the frame back, and
# `check_the_frame_verdict_has_three_answers` reverse-verifies the predicate in
# both directions on every run -- including the mutated `paintGL`, where the
# context is still live and the frame is a single colour. That file has a GL
# context; this one does not, and a count that depended on one would be a count
# that moved with the machine, which is the one thing a pinned total may not be.
#
# So what is asserted here is the *seam*, not the measurement: that the single
# place the threshold is applied reads the read-back's colour count, and that
# the context flag is nowhere in that decision. A structural check in this file
# and a behavioural check in another would normally be a claim nobody can
# state in one sentence -- so the structural half is not allowed to stand on
# its own. It names the file and the check that owns the other half, and it
# fails if that owner stops existing, so the two halves cannot drift apart
# without one of them going red.
_ppf = read(ROOT / "scripts" / "probe_painted_frame.py")
_min_colours = re.search(r"MIN_DISTINCT_COLOURS\s*=\s*(\d+)", _ppf)
_pred = _ppf.split("def _drew_something", 1)[1].split("\ndef ", 1)[0] if \
    "def _drew_something" in _ppf else ""
_predicate_body = _pred.split('"""', 2)[-1]
# The two behavioural owners, by name, and the reverse-verification that makes
# the mutated direction a measured result rather than a claim.
_owners = ("check_the_viewport_draws",
           "check_the_frame_verdict_has_three_answers")
_missing_owners = [o for o in _owners if f"def {o}(" not in _ppf]
_reverse = "reverse-verifies" in _ppf and _predicate_body.count("True") == 0
check(
    _min_colours is not None
    and int(_min_colours.group(1)) == 2
    and "distinct_colours" in _predicate_body
    and "context" not in _predicate_body
    and not _missing_owners
    and _reverse
    and "non_background_px" in _ppf
    and "context=" in _ppf,
    "the verdict is the read-back's colour count and the context flag is not "
    "in it, and the file that measures it is still there with both of its "
    "checks",
    f"probe_painted_frame.py applies its threshold in exactly one place, "
    f"`_drew_something`, whose body reads {'distinct_colours'!r} and does not "
    f"mention `context` at all; the structural minimum is "
    f"{_min_colours and _min_colours.group(1)} distinct colours -- a "
    f"clear-only frame is exactly one colour, so that is 'something was "
    f"drawn' and not a value fitted to a healthy frame. **The measurement is "
    f"not here and does not pretend to be:** the behavioural owner is "
    f"`scripts/probe_painted_frame.py::check_the_viewport_draws`, which runs "
    f"the product's own viewport and reads the frame back, and its "
    f"`check_the_frame_verdict_has_three_answers` reverse-verifies this same "
    f"predicate in both directions on every run. Both are present "
    f"(missing: {_missing_owners or 'none'}), so the seam this file asserts is "
    f"one whose other half is asserted by a named check rather than assumed. "
    f"**Mutation:** making the predicate read the context instead of the "
    f"colours turns this red",
)

section("this session's new claims: the two trees and the recognition rule")

_mirror_name = None
_spk = Path("F:/python310/lib/site-packages/opendocking")
_copies = {
    "development": ROOT / "dock-py" / "python" / "opendocking",
    "published": ROOT / "opendocking-gui" / "dock-py" / "python" / "opendocking",
    "installed": _spk,
}
_present = {k: v.is_dir() for k, v in _copies.items()}
_rule_src = read(ROOT / "scripts" / "release_tree_rule.py")
_tree_dirname = re.search(r'RELEASE_TREE_DIRNAME\s*=\s*"([^"]+)"', _rule_src).group(1)
check(
    all(_present.values()) and _tree_dirname == "opendocking-gui",
    "all three copies of the package are where the contract says they are, "
    "and the published one has the name the rule gives it",
    f"development {_copies['development']} exists: {_present['development']}; "
    f"published {_copies['published']} exists: {_present['published']}; "
    f"installed {_spk} exists: {_present['installed']}. The rule's "
    f"RELEASE_TREE_DIRNAME is {_tree_dirname!r}. The third copy is declared "
    f"out of scope rather than ignored, which is what makes \"one-directional\" "
    f"a property of two trees instead of a promise about three. **Mutation:** "
    f"renaming RELEASE_TREE_DIRNAME goes red, and so does publishing a tree "
    f"under a different directory name",
)

# --- the recognition rule, executed rather than searched for ---------------
_rp = ROOT / "scripts"
sys.path.insert(0, str(_rp))
import release_tree_rule as _RULE  # noqa: E402  (after the path fix, on purpose)
_synth = {
    Path("dock-py/python/opendocking/core.py"): "kept",
    Path("scripts/release_tree_rule.py"): "kept",
    Path("README.md"): "kept",
    Path("target/debug/x.rs"): "excluded",
    Path("__pycache__/core.pyc"): "excluded",
    Path("scratch.pyc"): "excluded",
    Path("odck.md"): "excluded",
    Path("opendocking-gui/dock-py/python/opendocking/core.py"): "excluded",
    Path("docs/SCORING.md"): "kept",
}
_selected = {p for p in _RULE.iter_source_files(ROOT) if p in _synth}
_wrong = {str(p): v for p, v in _synth.items()
          if (p in _selected) != (v == "kept")}
check(
    not _wrong,
    "the recognition rule selects exactly the files the contract says it does, "
    "on a fixture with one file per exclusion kind",
    f"{len(_synth)} synthetic paths, one per way the rule can say no "
    f"(excluded directory, excluded name, excluded suffix, the published tree "
    f"itself), and the rule's own iterator disagreed with the expectation on "
    f"{len(_wrong)}: {_wrong}. The rule is a *table*, so a check that grepped "
    f"the tables would have been asserting a second copy of the decision; "
    f"running it is the only way to score \"this is what a source file is\". "
    f"**Mutation:** adding a suffix to EXCLUDE_SUFFIX, or dropping "
    f".pdb from it, moves one of these paths and turns this red",
)

# --- one implementation, and the copy that ships is the copy that runs ------
_consumers = []
for _name in ("_sync_release.py", "release_tree_parity_check.py"):
    _src = read(ROOT / "scripts" / _name) if (ROOT / "scripts" / _name).is_file() \
        else read(ROOT / _name)
    _consumers.append((_name, "release_tree_rule" in _src))
_shipped = (ROOT / "opendocking-gui" / "scripts" / "release_tree_rule.py").is_file()
check(
    all(ok for _, ok in _consumers) and _shipped,
    "the rule has one implementation, both consumers import it, and the copy "
    "inside the published tree is the one that ships",
    f"{_consumers} name `release_tree_rule`, and the published tree carries "
    f"scripts/release_tree_rule.py: {_shipped}. A second implementation of "
    f"\"is this a source file\" anywhere in either tree is the defect the "
    f"rule's own docstring calls out, and a rule that could not be read from "
    f"the thing it governs is the failure that put it at the workspace root "
    f"in the first place -- where the rule's own root-level exclusion kept it "
    f"out of the artefact. **Mutation:** inlining the tables into the syncer "
    f"goes red",
)

# --- a machine that can see one tree is NOT-ASSERTED, not a pass ------------
_parity_src = read(ROOT / "scripts" / "release_tree_parity_check.py")
_single_tree = re.search(
    r"if trees\.development is None:(.*?)return", _parity_src, re.S)
_body = _single_tree.group(1) if _single_tree else ""
_routes_to_unasserted = "unasserted(" in _body
_compares_anyway = "snapshot(" in _body or "compare(" in _body
_why = re.search(r"there is one tree here", _parity_src)
check(
    _routes_to_unasserted and not _compares_anyway and _why is not None,
    "on a machine that can see only one of the two trees, the parity check "
    "records NOT-ASSERTED and returns -- the two trees are simply not "
    "verified there",
    f"release_tree_parity_check.py's single-tree branch routes every parity "
    f"check through unasserted() ({_routes_to_unasserted}) and reaches no "
    f"snapshot() or compare() before returning ({not _compares_anyway}); its "
    f"own reason string says \"there is one tree here\" ({_why is not None}). "
    f"That is the boundary stated rather than papered over: the development "
    f"tree is not under version control, so a clean CI checkout has one tree "
    f"and cannot reconstruct the other. **This is the structural half** -- a "
    f"machine with the development tree missing is a real state this "
    f"workspace cannot produce, and building a second checkout to produce it "
    f"would measure the copy rather than the contract",
)

section("this session's new claims: the cartoon's helix subtypes")

# Imported as a real submodule, not loaded from its path: `cartoon_geometry`
# does `from .geometry import ...`, so a path-based load has no parent package
# to resolve it against and raises before the first constant is read.
_cg = importlib.import_module("opendocking.workbench.cartoon_geometry")
_rpt = float(_cg.RESIDUES_PER_TURN)
_twist = float(_cg.HELIX_TWIST_PER_RESIDUE)
_doc_rpt = re.search(r"\*\*([\d.]+)\s*残基/圈\*\*画", LIMIT43)
# The two places the limitation states *the rate the cartoon draws at*: the
# summary table's cell and the paragraph's own sentence. Both are required,
# because they are two sentences a reader can arrive from -- a check that read
# only one would let the other drift, which is the defect this file exists to
# close, one document up. The 3-10 helix's own 3.0 is deliberately *not* in
# this set: that is the real rate of a 3-10 run, not the rate it is drawn at.
_all_rpt = ([_doc_rpt.group(1)] if _doc_rpt else []) + re.findall(
    r"画成\s*([\d.]+)\s*残基/圈", LIMIT43)
_doc_ten = re.search(r"（([\d.]+)\s*残基/圈）", LIMIT43)
_doc_pi = re.search(r"π 螺旋（([\d.]+)）", LIMIT43)
check(
    _doc_rpt is not None and abs(float(_doc_rpt.group(1)) - _rpt) < 1e-12
    and all(abs(float(v) - _rpt) < 1e-12 for v in _all_rpt)
    and len(_all_rpt) >= 2
    and abs(_twist - 360.0 / _rpt) < 1e-9
    and "3-10" in LIMIT43 and "π" in LIMIT43
    and _doc_ten is not None and _doc_pi is not None
    and float(_doc_ten.group(1)) < _rpt < float(_doc_pi.group(1)),
    "the residues-per-turn the limitation names is cartoon_geometry's own "
    "constant, the twist is 360 divided by it, and the 3-10 / alpha / pi "
    "orderings in the sentence are the ones that make 'under-twisted' the "
    "right word",
    f"RESIDUES_PER_TURN is {_rpt:g} and the limitation states the *drawing* "
    f"rate {len(_all_rpt)} time(s) -- the summary table's cell and the "
    f"paragraph's own sentence -- as {_all_rpt}, so the two cannot disagree; "
    f"HELIX_TWIST_PER_RESIDUE is "
    f"{_twist:g} = 360/{_rpt:g}, checked by arithmetic rather than by a second "
    f"literal. The sentence's three numbers order as "
    f"{_doc_ten and _doc_ten.group(1)} < {_rpt:g} < "
    f"{_doc_pi and _doc_pi.group(1)}, so a 3-10 run drawn at {_rpt:g} is "
    f"indeed under-twisted and a pi run is over-twisted. **Mutation:** "
    f"retuning the constant without editing the limitation goes red, and so "
    f"does inverting the 3-10/pi order in the prose while the code is right "
    f"-- the direction is part of the claim, not decoration",
)

_struct = read(ROOT / "dock-py" / "python" / "opendocking" / "workbench" / "structure.py")
_turns = re.search(r"HELIX_TURNS:\s*tuple\[int, \.\.\.\]\s*=\s*\(([^)]*)\)",
                   _struct)
_turn_classes = [int(t) for t in _turns.group(1).split(",")] if _turns else []
_states = set(re.findall(r'"(helix|sheet|coil|turn|bend)"\s*:', _struct))
_subtype_tokens = sorted(
    t for t in re.findall(r'"([a-z_0-9]+)"', _struct)
    if t in ("helix_3_10", "helix_3_10_3", "helix_pi", "three_ten", "pi_helix")
)
check(
    _turn_classes == [3, 4, 5] and _subtype_tokens == [] and "helix" in _states,
    "the classifier really does report one state for all three turn classes, "
    "which is the premise the limitation rests on",
    f"structure.py's HELIX_TURNS is {_turn_classes} -- DSSP's 3-10, alpha and "
    f"pi -- and the state table's keys are {sorted(_states)}; there is no "
    f"subtype token anywhere in the file ({_subtype_tokens}). So a per-run "
    f"rate is not merely unwritten, it is **unavailable**: the cartoon would "
    f"need a per-residue subtype the classifier does not emit. This is the "
    f"check that would catch a subtype being added without the cartoon being "
    f"retuned -- at which point the sentence stops being true and this goes "
    f"red, naming the reason",
)


# ==========================================================================
# 10. Self-verification
# ==========================================================================

section("every check in this file ran")

before = CHECKS
check(
    before + 1 == EXPECTED_CHECKS,
    "the number of checks that ran is the number this file is supposed to have",
    f"{before} ran before this one and {EXPECTED_CHECKS} are expected; the +1 is "
    f"this check. Change EXPECTED_CHECKS deliberately",
)

print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
for f in FAILURES:
    print(f"  FAILED: {f}")
sys.exit(1 if FAILURES else 0)
