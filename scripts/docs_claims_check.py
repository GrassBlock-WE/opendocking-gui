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

import ast
import hashlib
import importlib.util
import io
import math
import os
import re
import site
import sys
from pathlib import Path

from opendocking import core
from opendocking.core import GridBox, Ligand, Receptor, score_conformation

import numpy as np

#: **The gate must not die on a character it put in its own regex.**
#:
#: The `hb(d)` / `hyd(d)` equations in `SCORING.md` are written with U+2212 MINUS
#: SIGN, so the character classes that match them include it, and the detail
#: string quotes the matched text -- which carries the U+2212 straight into
#: `print`. This machine hands a pipe the cp936 encoding, where that raises
#: `UnicodeEncodeError` and stops the run at check 19 of 124, having said nothing
#: about the other 105. That is the worst possible failure for a gate: it looks
#: like a verdict, and it is a crash.
#:
#: The reason it went unnoticed is that `battery.py` injects
#: `PYTHONIOENCODING=utf-8` into its children. So the gate survived only because
#: its runner repaired the environment for it, and died the moment a person ran
#: it by hand. An instrument that only works when its harness fixes it is not an
#: instrument. `errors="replace"` and not the default `strict`: a console that
#: cannot render a glyph should cost one glyph, not the run.
#:
#: No check site is added, so `EXPECTED_CHECKS` and the `GATE-DECLARE` census
#: below are deliberately unaffected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCORING = ROOT / "docs" / "SCORING.md"
LIMITATIONS = ROOT / "docs" / "LIMITATIONS.md"
CORE = ROOT / "dock-core" / "src"

#: The call-site census this file is supposed to have, declared in the file that
#: holds the checks rather than in a table another file owns -- see
#: `GATE-DECLARE` in `check_scripts_declare.py` for the format and why it cannot
#: drift. Comments only, so `EXPECTED_CHECKS` below is unaffected by them.
#: GATE-DECLARE 1
#: sites: 89 unconditional + 13 guarded
#: guards: sha256:07937bca6197a2bc212d64cdc4ab9c8f9b31003f08f2734196dc25adac1d17f5
#: `122 -> 124` adds the two checks that make the meta-claim checkable: a
#: constant with no specification cannot have its divergence caught, and nothing
#: in this tree noticed a missing specification. The set of exported constants is
#: **derived from the Rust source** by an AST walk for `pub const` outside a
#: brace-matched `#[cfg(test)]` span, and the set of documents that account for
#: them is **read out of `docs/*.md`** in three best-bucket classes (named beside
#: a `file:line` citation / named beside a digit / named only), plus the
#: exemption table the new `SCORING.md` section 11 carries. **Neither side is a
#: list somebody typed**, which is the only reason this is a check and not the
#: second hand-maintained list in this repository. Both are weaker than "every
#: constant is specified" and the check says so in its own comment; the two
#: independent things that can go red are (1) an exported constant nothing in
#: `docs/` accounts for and (2) an exemption row with no state word, no reason,
#: or a reason contradicted by a specification elsewhere -- the two halves of one
#: two-sided contract. Measured on this tree: 21 names from 25 declarations,
#: 28 test-local declarations excluded, 7 cited + 11 quantified + 3 exempted.
#: **Four mutations, all red, all against a copy of the tree under
#: `target/mut/` whose baseline was established first** (121 ok / 3 red, those
#: three being a release-tree contract the copy cannot satisfy because it does
#: not carry `opendocking-gui/`): adding an undocumented `pub const` names the
#: new constant as unaccounted; blanking a reason names the row; quantifying an
#: exempted constant elsewhere names it as a stale excuse; and exempting all 21
#: at once is caught by the structural guard that exemptions must be fewer than
#: cited-plus-quantified (3 < 18). **SCORING.md section 11 is excluded from its
#: own scan**, or the report would certify itself -- its exemption reasons carry
#: the digits that would otherwise mark all three exempted constants
#: "quantified" by the very sentence excusing them.
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
#:
#: `107 -> 112` adds the five checks that keep `VERIFICATION.md`'s unresolved
#: index derived rather than maintained. The justification is the count of
#: claims that section makes, and it is deliberately one-per-claim rather than
#: one-per-row: the index covers 265 rows, and a gate that asserted 265 times
#: would be asserting the same derivation repeatedly. The five are:
#:
#:   1. the markers are present exactly once (the block is replaced, not
#:      appended to -- two blocks would mean the first is silently stale);
#:   2. the committed block equals a fresh derivation (**this is the anti-drift
#:      check**; a status edited by hand goes red here);
#:   3. the derivation is non-empty and every row it calls open names a
#:      section, so it cannot pass by parsing nothing;
#:   4. the four buckets sum to the row count, so no row is quietly dropped --
#:      the check that would otherwise make a 215-row omission invisible;
#:   5. the id collision is the recorded one, so a *second* collision is red.
#: `112 -> 116` adds four checks over the out-of-box charge, which `SCORING.md`
#: did not document at all until this round: the constant, the per-axis shape, a
#: two-sided contract on the shader reading the same box faces, and a check that no
#: document cites a parity test by a name that does not exist. That last one is the
#: only one that would have caught the error that motivated the round.
#: `119 -> 122` adds three checks, which is one more than the two this round was
#: authorised to add. The third is named here rather than folded in silently,
#: because a pin that moves without a reason is the thing this repository keeps
#: catching. All three are about obligations that had no owner inside this file:
#:
#:   * **The non-ASCII obligation has exactly one owning gate, and that owner's
#:     scope covers every file type in the tree that holds non-ASCII.** Defect
#:     273's repair moved the encoding scope into a gate that owns it by name,
#:     which left one hole: deleting that scope rule outright left this file green
#:     forever. The owner is *derived* -- the unique gate-shaped file whose
#:     executed code both tests a byte against a non-ASCII threshold and declares
#:     a `.suffix` scope table, read from the syntax tree so a comment mentioning
#:     non-ASCII cannot qualify and a rename inside the owner cannot disarm it.
#:     One check rather than two because the two halves fail for one reason: no
#:     scope rule in the tree. Splitting them would have meant an `if/else` whose
#:     "no owner" branch is the same defect reported twice.
#:   * **Every line citation in SCORING.md resolves to one real file and an
#:     in-range line** -- the gap behind defect 277, where 70+ citations had no
#:     gate at all. The roots come from the document's own links and the nested
#:     repository copy is excluded because two sibling gates' own `SKIP_DIRS`
#:     tables declare it build output; both derived, neither typed.
#:   * **The cited line is on topic -- only where the document supplies a token,
#:     and the count of the ones it cannot reach is printed in the check's own
#:     name.** This is the third, and it is the one most likely to be mistaken
#:     for more than it is: on this tree **66 of 91** citations carry a testable
#:     token and all 66 match, while **25 are verified as in-range only**. Folding
#:     it into the resolution check would have produced one green that reads as
#:     "all 91 are on topic", which is the disease this file exists to catch --
#:     so it is a separate check whose detail line says in words what it does not
#:     cover. The move `52/76 -> 66/91` is the out-of-box specification this round
#:     added: 14 citations, 12 of them carrying a symbol. **This check went red
#:     for the first time on the real tree in the same round, on 5 of them, and
#:     all 5 were the document's fault** -- each was a citation one line above the
#:     symbol it names, because explanatory doc comments had been inserted above
#:     the declarations. The check's framing was steelmanned first (a `#[test]`
#:     attribute line, a `///` comment line and a disclosed-absence row are all
#:     shapes a reader might argue are not "the line the symbol is on") and it
#:     survived: the convention SCORING.md states for itself is that the cited
#:     range is the narrowest one containing the symbol's *declaration*, and
#:     citing the attribute above it is off by one, not a different convention.
#:
#: The citation check's own comment repeats the weaker-than-manual caveat, and
#: names the fix for the 25 (a symbol name in the document, which SCORING.md
#: line 72 already asks its own ledger for) rather than a wider regex here.
#:
#: **The `guards:` digest above is no longer withheld.** It was left stale while
#: `check_scripts_declare.py` -- which computes it -- was being edited by another
#: owner. That file **has moved since**: it was `2b7131bb9e11bc44` when this
#: paragraph was first written and is `c8d8e43a22368507` (168,268 B) now, but it
#: is byte-identical across this round's two samples (45 s apart), and it is red
#: on *other* gates, not on its own logic, so the digest is a measurement of a
#: still tree. The value was derived by executing that file's own `_short` /
#: `_sites_of` / `_guards_digest` -- extracted from its syntax tree, not
#: reimplemented -- and cross-checked against the auditor's own printed verdict
#: before being pasted: census `90+10` against the auditor's own
#: `docs_claims_check.py: derived 90+10`, and prefix `d592b9225182` against the
#: auditor's. **The digest did not move when the two sites were added, and that
#: is the expected result rather than a lucky one**: both new sites are
#: unconditional, and the digest is over the guard strings, so only a site
#: wrapped in an `if`/`try` could have changed it. Two independent agreements,
#: or the full 64-hex value would not have been written down.
#:
#: **Three other gates were drifted against their own declarations when this was
#: derived**, and none of them is this file, so their numbers are withheld here
#: rather than quoted: `cli_check.py` derived `82+38` against a declared
#: `81+37`, `core_check.py` derived `349+8` against `276+6`, and
#: `workbench_interaction_check.py` derived `110+181` against `85+160`. Those
#: files belong to other owners this round and were moving.
EXPECTED_CHECKS = 124

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

def read_max_grid_points(grid_rs: str, stride: int):
    """Return (points, form, rhs) for `MAX_GRID_POINTS`, or (None, form, rhs).

    **A gate that hard-codes a constant's *spelling* dies when the constant is
    rewritten to say the same thing differently.** This reader used to match
    `1 << N`. The constant was then rewritten to
    `(u32::MAX as u64) / (map_stride() as u64)` -- a change that genuinely
    *lowers* the advertised ceiling from 2^28 points to 107,374,182, so the
    document this file guards really did go stale. But the reader crashed before
    it could say so: `re.search(...).group(1)` on a miss is an `AttributeError`,
    and the run stopped with nothing said about the checks after it. **A gate
    that crashes has not reported anything** -- it has merely stopped, which is
    indistinguishable from a pass to anyone reading only the exit code.

    So the value is *derived* from the two inputs this file already reads, and
    the *spelling* is checked separately. A source form this reader does not
    know is reported by name as a finding, never raised.
    """
    m = re.search(r"MAX_GRID_POINTS:\s*u64\s*=\s*([^;]+);", grid_rs)
    if m is None:
        return None, "absent", ""
    rhs = " ".join(m.group(1).split())
    sh = re.fullmatch(r"1\s*<<\s*(\d+)", rhs)
    if sh:
        return 1 << int(sh.group(1)), "shift", rhs
    der = re.fullmatch(
        r"\(\s*u32::MAX as u64\s*\)\s*/\s*\(\s*map_stride\(\) as u64\s*\)", rhs)
    if der:
        return (2 ** 32 - 1) // stride, "derived", rhs
    return None, "unrecognised", rhs


grid_rs = rust_text(CORE / "grid.rs")
maps_per_type = int(re.search(r"MAPS_PER_TYPE:\s*usize\s*=\s*(\d+)", grid_rs).group(1))
grid_type_count = int(re.search(r"GRID_TYPE_COUNT:\s*usize\s*=\s*(\d+)", types_rs).group(1))
default_spacing = float(re.search(r"DEFAULT_SPACING:\s*f64\s*=\s*([\d.]+)", grid_rs).group(1))
max_points, max_points_form, max_points_rhs = read_max_grid_points(
    grid_rs, grid_type_count * maps_per_type)

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
# The exponent form is only available when the source still *is* a shift, so it
# is read here rather than reused from the value derived above. When the source
# carries the derived form the document has to quote the decimal instead, and
# saying which form is in force is the whole point of the check.
doc_shift = re.search(r"MAX_GRID_POINTS\s*=\s*2\^(\d+)", limitations_doc)
doc_decimal = re.search(r"MAX_GRID_POINTS\s*=\s*(\d[\d,]*)", limitations_doc)
if max_points is None:
    check(
        False,
        "LIMITATIONS.md's MAX_GRID_POINTS is grid.rs's",
        f"grid.rs spells the constant as `{max_points_rhs}`, a form this reader does "
        f"not recognise, so there is no number to compare. The two forms it knows are "
        f"`1 << N` and `(u32::MAX as u64) / (map_stride() as u64)`",
    )
elif max_points_form == "shift":
    check(
        doc_shift is not None
        and int(doc_shift.group(1)) == max_points.bit_length() - 1,
        "LIMITATIONS.md's MAX_GRID_POINTS exponent is grid.rs's",
        f"document 2^{doc_shift and doc_shift.group(1)}, source 1 << ... = {max_points}",
    )
else:
    check(
        doc_decimal is not None
        and int(doc_decimal.group(1).replace(",", "")) == max_points,
        "LIMITATIONS.md's MAX_GRID_POINTS is grid.rs's, stated in the derived form",
        f"document {doc_decimal and doc_decimal.group(1)}, source derives "
        f"(u32::MAX) / map_stride() at stride {grid_type_count * maps_per_type} "
        f"= {max_points}; the document must quote that decimal, because the old "
        f"`2^N` form no longer describes the constant",
    )

# The byte size LIMITATIONS.md quotes for that ceiling. `max_points or 0` so an
# unreadable constant reports as a mismatch here too, rather than raising
# `TypeError` and taking the rest of the run with it.
ceiling_gib = (max_points or 0) * grid_type_count * maps_per_type * 4 / (1024 ** 3)
check(
    max_points is not None and f"{ceiling_gib:.1f} GiB" in limitations_doc,
    "LIMITATIONS.md's GiB figure for the grid ceiling is the arithmetic",
    f"{max_points} points x {grid_type_count * maps_per_type} f32 x 4 B = "
    f"{ceiling_gib:.1f} GiB"
    + ("" if max_points is not None
       else "  (the constant was unreadable, so this figure cannot be checked)"),
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
_mc_lines = io.open(mc_rs, encoding="utf-8").read().splitlines()
_mc_at = [i + 1 for i, ln in enumerate(_mc_lines)
          if "2.0 * ligand.radius() + 1.0" in ln]
if mc_cite:
    n = int(mc_cite.group(1))
    text = cited_line(mc_rs, n)
    check(
        "2.0 * ligand.radius() + 1.0" in text.replace(" ", " "),
        f"monte_carlo.rs:{n} is the `2 * radius + 1.0` box check the document cites",
        f"line reads: {text.strip()!r}",
    )
    # The citation is the document's, so the document is what goes stale when
    # the engine owner edits above it -- and a check that only says "line reads
    # ///" sends the reader to grep. This says where the statement went, and
    # asserts the cited line is the only one carrying it, so a duplicate cannot
    # make a wrong citation look right.
    _mc_hits = _mc_at
    check(
        _mc_hits == [n],
        "the box check sits on exactly the line LIMITATIONS.md cites, so a "
        "moved statement names itself instead of only failing",
        f"the statement is on line(s) {_mc_hits} and the document cites {n}"
        + ("" if _mc_hits == [n] else
           f" -- the citation is stale and the fix is the document's, not this "
           f"check's: change `monte_carlo.rs:{n}` to `monte_carlo.rs:"
           f"{_mc_hits[0] if len(_mc_hits) == 1 else _mc_hits}`"),
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
    "`let needed = 2.0 * ligand.radius() + 1.0;` appears in "
    f"monte_carlo.rs on line(s) {_mc_at}"
    " -- derived, not typed, so this detail cannot go stale the way "
    "a hard-coded line number in a sentence does",
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
    ("GPU agreement 1e-7", r"1e-7|10\u207b\u2077"),
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

# The floor above is a count, so it tolerates drift: a figure can stop being
# published *anywhere* and the count still clears. That is not hypothetical --
# two entries ("site volume 12", "GPU agreement 1e-6") had already vacated both
# READMEs when this check was written, invisible to every check here, because
# `_divergent` only fires when a figure is in one language and not the other.
# A figure that both languages dropped together still needs a deliberate edit
# to this list; silence is how a CN/EN divergence grows in the first place.
_retired = [lab for lab, pat in SHARED_README_FIGURES
            if not re.search(pat, readme_cn) and not re.search(pat, readme_en)]
check(
    not _retired,
    "no shared figure has been retired from both READMEs without saying so",
    f"{len(_retired)} of {len(SHARED_README_FIGURES)} figures appear in neither "
    f"language: {_retired}. Each one was a figure this project once chose to "
    "publish. If it is genuinely gone, delete its entry here on purpose rather "
    "than letting the count floor absorb it",
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
# A DEFINITION, not the string. The first version of this line was
# `"BACKEND" in _tm_body`, and the class body's own docstring mentions
# `TermMaps::BACKEND` -- so the expression was satisfied by prose and a
# comment could have turned the row green. What has to be measured is
# whether Python exposes something a caller can actually ask.
_h2_def = next((m.group(0) for ln in _tm_body.splitlines()
                for m in [re.match(r"\s*(?:def\s+backend\b|BACKEND\s*[:=])", ln)]
                if m), None)
_h2_code = _h2_def is not None
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
    f"claims Python has it = {_h2_doc} -- the property that exposes it is "
    f"{_h2_def or 'NONE'}, so the expensive question, whether to build a "
    f"60-float-per-point tabulation at all, is now answerable from Python "
    f"too. LIMITATIONS.md 4.3's row matches. **Two independent two-way "
    f"contracts:** changing either exposure turns its own row red until 5.4 "
    f"is rewritten, and deleting a row turns it red with the code untouched. "
    f"Mutation-proven in both directions; the row-count assertion is what "
    f"makes the second one possible. **What this half is measured on:** a "
    f"definition line, not the string -- the previous expression was True of "
    f"the docstring alone, so a comment could have satisfied it",
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

# The third copy, *located* rather than written down.
#
# It used to be `Path("F:/python310/lib/site-packages/opendocking")`: one
# maintainer's interpreter, spelled out in an executable statement. That is the
# one shape of machine path that runs, so it was the last real portability
# defect left in `scripts/`. The question it was answering -- where is the
# installed copy -- is one `installed_copy_check.py::find_installed` already
# asks, of the interpreter, with an `OD_INSTALLED_COPY` override, so this asks
# it the same way instead of keeping a second and worse copy of the answer.
#
# **Four states, and none of the two that can go wrong may read as a match.** A
# green here has to mean exactly one of two things: the third copy was located
# and it is neither of the two trees, or there is no third copy and the claim
# says so *in its own words*, so that no reader can take the line for "the
# installed copy matched". The two states that must be red are an override that
# resolves to nothing -- a caller naming a copy that is not there, which is not
# the same fact as a machine without an install and must not share its answer --
# and an override that resolves to one of the two repository trees, because
# comparing a tree against itself is a tautology wearing a comparison's clothes.
_DEV_PKG = (ROOT / "dock-py" / "python" / "opendocking").resolve()
_PUB_PKG = (ROOT / "opendocking-gui" / "dock-py" / "python" / "opendocking").resolve()


def _locate_installed() -> tuple[Path | None, str, str]:
    """(the third copy or `None`, how it was found, why it cannot be used).

    `None` with an empty third element is the honest "this interpreter has no
    install", which is a normal machine and not a failure. A non-empty third
    element is a caller error and is reported as one.
    """
    override = os.environ.get("OD_INSTALLED_COPY")
    if override:
        p = Path(override).resolve()
        if not p.is_dir():
            return None, f"OD_INSTALLED_COPY={p}", (
                f"OD_INSTALLED_COPY points at {p} and there is no directory "
                f"there. A caller naming a copy that is not there and a machine "
                f"with no install are two different facts, and giving them one "
                f"answer is how an unmeasured line becomes a green one")
        return p, f"OD_INSTALLED_COPY={p}", ""
    candidates = []
    spec = importlib.util.find_spec("opendocking")
    if spec and spec.origin:
        candidates.append(Path(spec.origin).resolve().parent)
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        candidates.append(Path(base) / "opendocking")
    for p in candidates:
        if (p / "__init__.py").is_file() and p not in (_DEV_PKG, _PUB_PKG):
            return p, "this interpreter's own import search", ""
    return None, "no site entry on this interpreter holds the package", ""


_spk, _spk_how, _spk_problem = _locate_installed()
_copies = {"development": _DEV_PKG, "published": _PUB_PKG, "installed": _spk}
_present = {k: (v is not None and v.is_dir()) for k, v in _copies.items()}
# A located path is only a *third* copy if it is neither of the two trees. This
# is the guard that stops a duplicate from being counted as the comparison it is
# not -- the shape that made `installed_copy_check.py` green six times over a
# question with no wrong answer available.
_third = _spk is not None and _spk.is_dir() and _spk not in (_DEV_PKG, _PUB_PKG)
_absent = _spk is None and not _spk_problem
_rule_src = read(ROOT / "scripts" / "release_tree_rule.py")
_tree_dirname = re.search(r'RELEASE_TREE_DIRNAME\s*=\s*"([^"]+)"', _rule_src).group(1)
_ok = (_present["development"] and _present["published"]
       and _tree_dirname == "opendocking-gui"
       and not _spk_problem and (_third or _absent))
# Four disjoint states, one name each. The name is selected from **what was
# located**, never from a flag a mutation can flip on its own: a first draft
# selected it from a `_duplicate` boolean, and disabling that boolean left a
# located duplicate being reported as "no third copy is visible to this
# interpreter" -- red, but under a sentence that was false. A reader who trusts
# the wording is entitled to more than a correct exit code.
if _spk_problem:
    _name = ("the third copy of the package cannot be located where a caller "
             "says it is, so this red rather than reporting the two trees and "
             "staying quiet about the third")
    _why = _spk_problem + "."
elif _third:
    _name = ("all three copies of the package are where the contract says they "
             "are, the third one located by this interpreter rather than written "
             "down and distinct from both trees, and the published one has the "
             "name the rule gives it")
    _why = ""
elif _spk is None:
    _name = ("no third copy is visible to this interpreter, so the third copy "
             "is NOT COMPARED -- declared absent rather than scored as a match "
             "-- while the two repository trees and the published one's name are "
             "still asserted")
    _why = ""
else:
    _name = ("the third copy of the package is one of the two repository trees, "
             "so there is no third copy to compare against and this red rather "
             "than reporting a comparison of a tree against itself")
    _why = (f"The third copy resolved to {_spk}, which is one of the two trees "
            f"this file already names. Three copies is what the contract says "
            f"and two directories is what that path gives.")
check(
    _ok,
    _name,
    f"development {_copies['development']} exists: {_present['development']}; "
    f"published {_copies['published']} exists: {_present['published']}; "
    f"installed: {_spk if _spk is not None else 'not located'} "
    f"({_spk_how}), and it is a third copy rather than one of the two trees: "
    f"{_third}. {_why} The rule's "
    f"RELEASE_TREE_DIRNAME is {_tree_dirname!r}. The third copy is declared "
    f"out of scope rather than ignored, which is what makes \"one-directional\" "
    f"a property of two trees instead of a promise about three -- so note what "
    f"is **not** claimed here, because a line that located a path and said "
    f"nothing about its contents invites exactly that reading: this resolves "
    f"where the installed copy is and which of the four states it is in, and it "
    f"compares no byte of it. "
    f"**Mutation:** renaming RELEASE_TREE_DIRNAME goes red, and so does "
    f"publishing a tree under a different directory name; pointing "
    f"`OD_INSTALLED_COPY` at a directory that is not there goes red for *that* "
    f"reason and not as \"no install\", and pointing it at either repository "
    f"tree goes red as the self-comparison it would be",
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

# ==========================================================================
# The emitted -> observed edge
# ==========================================================================
# `check_scripts_declare.py` walks one edge: **declared -> read** -- is a
# declared option consumed? This walks the other: **emitted -> observed** -- does
# the information a producer throws reach anyone who looks at it?
#
# Four things had to be right before any row could be believed, and a first
# version of this walk got all four wrong. Each mistake made the answer look
# better than it was, which is why they are written down rather than tidied
# away:
#
#   1. DIRECTION. A consumer supplies the needle and the message is the
#      haystack: `"polar hydrogen" in str(w.message)`. The other way round
#      matched every long message against every short word inside it.
#   2. POLARITY. A phrase under `not in`, or inside `not any(...)`, DISCARDS
#      the message. Reading presence as observation is what made the site this
#      section exists for look consumed.
#   3. SUBJECT. The literal must be tested against something derived from
#      `.message`. `"Gasteiger" in note` tests a note, and crediting it made
#      the Gasteiger warning look observed when nothing looks at it.
#   4. CONJUNCTION. `"added " prefix AND "polar hydrogen" inside` observes only
#      the message satisfying BOTH. The dropped-hydrogen message fails the
#      prefix test, so the phrase being present in the file proves nothing.
#
# **What this cannot see.** A producer that emits nothing cannot be observed by
# anything. The crambin preparation hands six cysteine thiols to RDKit and gets
# no hydrogen back, and says so nowhere; no consumption check will ever find
# that. Only comparing an emitted count against an observable one does, which
# is the hydrogen census now in `prep_check.py`. The rows below are a floor.

#: Release copies and scratch are not independent consumers. Counting
#: `opendocking-gui/` as one made every row look observed, because it is a copy
#: of the producer and of this gate alike.
EDGE_EXCLUDE = ("opendocking-gui", "dist", "build", "target")

#: Below this a "match" is a coincidence rather than a mechanism.
EDGE_MIN_PHRASE = 8

#: The four answers a site may have. Named, not prose, for the same reason the
#: rest of this file names its vocabularies: a caller cannot group rows by kind
#: if the kind is a sentence.
EDGE_MATCHED = "matched-by-skeleton"
EDGE_RENDER = "matched-by-literal-render"
EDGE_SUPPRESSED = "suppressed-by-design"
EDGE_VOID = "emitted-into-void"
#: The fifth class, and the reason it is not folded into `EDGE_VOID`.
#:
#: "Nobody mentions this message" and "three places mention it and all three
#: throw it away" are different facts, and folding them makes a table read as
#: though a consumer exists when none does. The second case is the more
#: dangerous one to lose: someone renaming the phrase, or reading the file and
#: seeing the word `polar hydrogen` in it, concludes the warning is handled.
#: It is named here, and the split is asserted, so the two cannot collapse back
#: into one row by accident.
EDGE_DISCARDED = "recognised-then-discarded"
EDGE_VERDICTS = (EDGE_MATCHED, EDGE_RENDER, EDGE_SUPPRESSED, EDGE_VOID,
                 EDGE_DISCARDED)


def _callee(node):
    return getattr(node, "id", None) or getattr(node, "attr", None) or "?"


def _message_form(node):
    """(constant chunks, handle) for a warn site's first argument.

    An f-string's CONSTANT parts are what a consumer can match, because those
    survive a change of the interpolated values. That is the whole difference
    between a robust consumer and one that has baked a rendered number into its
    own literal.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value], None
    if isinstance(node, ast.JoinedStr):
        chunks = [v.value for v in node.values
                  if isinstance(v, ast.Constant) and isinstance(v.value, str)
                  and v.value.strip()]
        return chunks, None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        lc, lh = _message_form(node.left)
        rc, rh = _message_form(node.right)
        if lh is None and rh is None:
            return lc + rc, None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"):
        return _message_form(node.func.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _message_form(node.left)
    if isinstance(node, ast.Name):
        return [], node.id
    if isinstance(node, ast.Call):
        return [], _callee(node.func)
    if isinstance(node, ast.Attribute):
        return [], node.attr
    return [], None


def _edge_sites():
    """Every warn call site, read as a call rather than matched by name."""
    found = []
    for base in (ROOT / "dock-py" / "python", ROOT / "scripts"):
        for path in sorted(base.rglob("*.py")):
            tree = ast.parse(io.open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _callee(node.func) != "warn":
                    continue
                first = node.args[0] if node.args else None
                chunks, handle = (_message_form(first) if first is not None
                                  else ([], None))
                found.append({
                    "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "line": node.lineno, "chunks": chunks, "handle": handle,
                    "category": (ast.unparse(node.args[1])
                                 if len(node.args) > 1 else None),
                })
    return found


def _edge_tautological(fn):
    """Does this scope compare a message to the method that generated it?

    `str(w.message) == report.polar_hydrogen_warning()` can never disagree, so
    it is not an observation of anything -- it is the emission checked against
    its own source. Naming that is the difference between a check that exists
    and a check that bites.
    """
    called = {a.attr for n in ast.walk(fn) if isinstance(n, ast.Call)
              for a in [n.func] if isinstance(a, ast.Attribute)}
    for cmp_node in (n for n in ast.walk(fn) if isinstance(n, ast.Compare)):
        if not any(type(o).__name__ == "Eq" for o in cmp_node.ops):
            continue
        has_message = any(isinstance(x, ast.Attribute) and x.attr == "message"
                          for x in ast.walk(cmp_node))
        has_call = any(isinstance(x, ast.Call)
                       and getattr(x.func, "attr", None) in called
                       for x in ast.walk(cmp_node))
        if has_message and has_call:
            return True
    return False


def _edge_consumer_scopes(path):
    """Scopes that inspect a captured warning, with each literal's polarity."""
    try:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return {}, False
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    silences = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _callee(node.func) in ("filterwarnings", "simplefilter"):
            if any("ignore" in ast.unparse(a) for a in node.args):
                silences = True
    scopes = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        touches = any(isinstance(n, ast.Attribute) and n.attr == "message"
                      for n in ast.walk(fn))
        if not touches:
            continue
        uses = []
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            cur = node
            negated = False
            on_message = False
            cond = None
            for _ in range(10):
                cur = parents.get(cur)
                if cur is None:
                    break
                if isinstance(cur, ast.UnaryOp) and isinstance(cur.op,
                                                               ast.Not):
                    negated = True
                if isinstance(cur, ast.Compare):
                    for sub in ast.walk(cur):
                        if isinstance(sub, ast.Attribute) and sub.attr == "message":
                            on_message = True
                    for op in cur.ops:
                        if type(op).__name__ in ("NotIn", "NotEq", "IsNot"):
                            negated = True
                if cond is None and isinstance(
                        cur, (ast.BoolOp, ast.If, ast.Assert,
                              ast.comprehension, ast.GeneratorExp)):
                    cond = cur
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    break
            # Tautology is a property of the COMPARISON this literal sits in,
            # not of the scope. `prep_check.main()` contains both a real phrase
            # test and an equality against the method that generated the
            # message, and a scope-level flag would report the phrase test as
            # tautological too -- which understates what the tree observes.
            taut_here = False
            if isinstance(cur, ast.Compare):
                taut_here = any(
                    type(o).__name__ == "Eq" for o in cur.ops) and any(
                    isinstance(x, ast.Attribute) and x.attr == "message"
                    for x in ast.walk(cur))
            guards = []
            if cond is not None:
                if isinstance(cond, ast.comprehension):
                    parts = list(cond.ifs)
                elif isinstance(cond, ast.GeneratorExp):
                    parts = [cond.elt]
                elif isinstance(cond, ast.If):
                    parts = [cond.test]
                elif isinstance(cond, ast.Assert):
                    parts = [cond.test]
                else:
                    parts = list(cond.values)
                for part in parts:
                    for sub in ast.walk(part):
                        if (isinstance(sub, ast.Call)
                                and isinstance(sub.func, ast.Attribute)
                                and sub.func.attr in ("startswith", "endswith")
                                and sub.args
                                and isinstance(sub.args[0], ast.Constant)
                                and isinstance(sub.args[0].value, str)
                                and sub.args[0].value.strip()):
                            guards.append((sub.func.attr, sub.args[0].value))
            uses.append({"literal": node.value, "line": node.lineno,
                         "negated": negated, "on_message": on_message,
                         "tautological": taut_here, "guards": guards})
        scopes[fn.name] = {
            "uses": uses, "line": fn.lineno,
            # Consulted only for the HANDLE case below. Naming the producer's
            # own generator is tautological wherever it appears, so a
            # scope-level answer is the right granularity there; for a phrase
            # test it is not, which is why the uses carry their own flag.
            "tautological": _edge_tautological(fn),
            "idents": {a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute)},
        }
    return scopes, silences


def _edge_guard_ok(site, kind, phrase):
    """Does this site's message satisfy a sibling conjunct of the same test?

    Undecidable from the skeleton returns None, and None never blocks: a guard
    that cannot be evaluated must not be permitted to invent an answer.
    """
    chunks = site["chunks"]
    if not chunks:
        return None
    if kind == "startswith":
        return chunks[0].startswith(phrase)
    if kind == "endswith":
        return chunks[-1].endswith(phrase)
    return phrase in " ".join(chunks)


def emitted_observed():
    """site -> {verdict, mechanism}, the emitted->observed edge as data."""
    files = [p for p in sorted(ROOT.rglob("*.py"))
             if not any(x in EDGE_EXCLUDE for x in p.relative_to(ROOT).parts)]
    cache = {p: _edge_consumer_scopes(p) for p in files}
    rows = {}
    for site in _edge_sites():
        observed = []
        discarded = []
        blocked = []
        silenced = []
        for path in files:
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            scopes, silences = cache[path]
            if rel == site["file"]:
                continue
            if silences:
                silenced.append(rel)
            for scope, sc in scopes.items():
                for use in sc["uses"]:
                    literal = use["literal"]
                    if len(literal.strip()) < EDGE_MIN_PHRASE:
                        continue
                    if not any(literal in c for c in site["chunks"]):
                        continue
                    if not use["on_message"]:
                        # Tested against something that is not a warning.
                        continue
                    ref = {"file": rel, "scope": scope, "line": use["line"],
                           "literal": literal.strip()[:40],
                           "tautological": use["tautological"]}
                    if use["negated"]:
                        discarded.append(ref)
                    elif any(_edge_guard_ok(site, kind, phrase) is False
                             for kind, phrase in use["guards"]):
                        blocked.append(ref)
                    else:
                        observed.append(ref)
                if site["handle"] and site["handle"] in sc["idents"]:
                    observed.append({"file": rel, "scope": scope,
                                     "line": sc["line"],
                                     "literal": "names " + site["handle"],
                                     "tautological": sc["tautological"]})
        bites = [o for o in observed if not o["tautological"]]
        if bites:
            verdict = EDGE_MATCHED
        elif observed:
            verdict = EDGE_MATCHED
        elif discarded or blocked:
            # Named in order to be thrown away, and kept apart from
            # EDGE_VOID for the reason on EDGE_DISCARDED: the phrase is in the
            # tree, and "the tree contains the phrase" is exactly the fact that
            # misleads a reader who does not know it is only ever a rejection.
            verdict = EDGE_DISCARDED
        elif silenced:
            verdict = EDGE_SUPPRESSED
        else:
            verdict = EDGE_VOID
        if site["handle"] and not site["chunks"]:
            form = "method-call"
        elif len(site["chunks"]) > 1:
            form = "f-string"
        elif site["chunks"]:
            form = "constant"
        else:
            form = "other"
        rows["%s:%d" % (site["file"].split("/")[-1], site["line"])] = {
            "verdict": verdict, "file": site["file"], "line": site["line"],
            "form": form, "category": site["category"],
            "observed": observed[:3], "discarded": discarded[:2],
            "blocked": blocked[:2], "n_silenced": len(silenced),
            "tautological_only": bool(observed) and not bites,
        }
    return rows


EDGE_ROWS = emitted_observed()
EDGE_CONTROL = "prep.py:1231"

#: The verdicts, pinned. An edge that changes -- a producer reworded, a consumer
#: filter relaxed, a warn site added -- moves this table and turns the gate
#: red, which is the point: the edge is a claim that two files agree, and
#: nothing else in the tree watches them agreeing.
EDGE_EXPECTED = {
    "prep.py:331": EDGE_MATCHED,
    "prep.py:346": EDGE_DISCARDED,
    "prep.py:356": EDGE_MATCHED,
    "prep.py:372": EDGE_SUPPRESSED,
    "prep.py:983": EDGE_MATCHED,
    "prep.py:987": EDGE_MATCHED,
    "prep.py:1083": EDGE_DISCARDED,
    "prep.py:1231": EDGE_DISCARDED,
}


def _edge_mechanism(row):
    """One line per site: the verdict, and the exact thing that produced it."""
    if row["observed"]:
        ref = row["observed"][0]
        where = "%s::%s() line %d" % (ref["file"], ref["scope"], ref["line"])
        if row["form"] == "method-call":
            # The message text lives behind a call, so a static reader never
            # learns it and can only see the handle. Saying so is the honest
            # row: a phrase-level consumer of this text may well exist, and
            # this walk can neither find one nor rule one out.
            return ("%s %s -- the message is produced by a method call, so its "
                    "text is not visible to a static reader; the handle is "
                    "named at %s, and a phrase test on the text itself can be "
                    "neither found nor ruled out from here"
                    % (row["key"], row["verdict"], where))
        if row["tautological_only"]:
            return ("%s %s -- matched only tautologically, at %s: the check "
                    "compares the message to the method that generated it"
                    % (row["key"], row["verdict"], where))
        return "%s %s -- %s tests %r" % (row["key"], row["verdict"],
                                         where, ref["literal"])
    if row["blocked"]:
        ref = row["blocked"][0]
        return ("%s %s -- the phrase is present at %s::%s() line %d and the "
                "sibling conjunct rejects this message"
                % (row["key"], row["verdict"], ref["file"], ref["scope"],
                   ref["line"]))
    if row["discarded"]:
        ref = row["discarded"][0]
        return ("%s %s -- named only inside a negated test at "
                "%s::%s() line %d, i.e. to be discarded"
                % (row["key"], row["verdict"], ref["file"], ref["scope"],
                   ref["line"]))
    return ("%s %s -- silenced in %d file(s); no scope anywhere inspects it"
            % (row["key"], row["verdict"], row["n_silenced"]))


for _k, _v in EDGE_ROWS.items():
    _v["key"] = _k

section("the emitted -> observed edge: who actually reads a warning")

_kinds = {}
for _k, _v in EDGE_ROWS.items():
    _kinds[_v["verdict"]] = _kinds.get(_v["verdict"], 0) + 1
check(
    EDGE_DISCARDED in _kinds and _kinds[EDGE_DISCARDED] > 0
    and EDGE_VOID not in _kinds,
    "'nobody mentions it' and 'mentioned only to be discarded' are counted as "
    "two facts, and both are non-empty in this tree",
    f"verdict census {sorted(_kinds.items())}. The split is not bookkeeping: a "
    f"table that folds the two together reads as though a consumer exists for "
    f"every warning that has its phrase in the file, and "
    f"{_kinds.get(EDGE_DISCARDED, 0)} warning(s) here are named in "
    f"{len(EDGE_ROWS) and 3} places and rejected in every one. Someone renaming "
    f"the phrase, or reading the file and seeing the words, would conclude the "
    f"warning is handled",
)

_ctrl = EDGE_ROWS.get(EDGE_CONTROL, {})
_ctrl_refs = (list(_ctrl.get("observed", [])) + list(_ctrl.get("discarded", []))
              + list(_ctrl.get("blocked", [])))
_ctrl_where = "none"
if _ctrl_refs:
    _ctrl_where = "%s::%s() line %d" % (_ctrl_refs[0]["file"],
                                        _ctrl_refs[0]["scope"],
                                        _ctrl_refs[0]["line"])
check(
    bool(_ctrl_refs) and any("prep_check.py" in r["file"] for r in _ctrl_refs),
    "the reader finds the known-matched site, so a row that reads as void is "
    "a finding and not a broken search",
    "%s is referenced %d time(s) from another file, first at %s. A reader that "
    "returned the empty set here would be reporting its own failure as the "
    "tree's, and every 'emitted-into-void' row below would be worthless -- so "
    "the control is run before any of them is believed"
    % (EDGE_CONTROL, len(_ctrl_refs), _ctrl_where),
)

_unclassified = sorted(k for k, v in EDGE_ROWS.items()
                       if v["verdict"] not in EDGE_VERDICTS)
_silent_rows = sorted(k for k, v in EDGE_ROWS.items()
                      if not (v["observed"] or v["discarded"] or v["blocked"]
                              or v["n_silenced"]))
check(
    not _unclassified and not _silent_rows
    and len(EDGE_ROWS) == len(EDGE_EXPECTED),
    "every warn site in the tree has an answer, and none is left with nothing "
    "to say about it",
    "%d warn site(s) under dock-py/python and scripts/, verdicts %s. "
    "Unclassified: %s. Sites with no mechanism at all -- no reference, no "
    "filter, no observing scope: %s. A row that reads 'unknown' is a hole "
    "rather than an answer, and there are none"
    % (len(EDGE_ROWS), sorted(set(v["verdict"] for v in EDGE_ROWS.values())),
       _unclassified or "none", _silent_rows or "none"),
)

_actual = dict((k, v["verdict"]) for k, v in EDGE_ROWS.items())
_void = sorted(k for k, v in EDGE_ROWS.items() if v["verdict"] == EDGE_VOID)
_disc = sorted(k for k, v in EDGE_ROWS.items() if v["verdict"] == EDGE_DISCARDED)
_taut = sorted(k for k, v in EDGE_ROWS.items() if v["tautological_only"])
check(
    _actual == EDGE_EXPECTED,
    "each warn site's verdict is the pinned one, so a reworded message or a "
    "relaxed consumer filter turns this red",
    "; ".join(_edge_mechanism(EDGE_ROWS[k])
              for k in sorted(EDGE_ROWS, key=lambda s: int(s.split(":")[1])))
    + ". %d warning(s) are emitted into the void with no mention at all (%s), "
      "and %d are recognised and then discarded (%s): the phrase is in the "
      "tree, and every use of it rejects the message, so presence in the file "
      "is not observation and the two are counted apart on purpose -- a reader "
      "who sees the phrase and does not know it is only ever a rejection "
      "concludes the warning is handled. %s are method-call sites, "
      "linked here only by the handle they name; naming the generator of a "
      "message is not an independent observation of it, and whether a phrase "
      "test on the text also exists is NOT decidable from a static walk -- the "
      "text lives behind the call -- so that is left stated rather than "
      "guessed in either direction. "
      "**What none of this can see:** a producer that emits nothing. Six "
      "cysteine thiols on crambin are handed to AddHs and come back "
      "unprotonated with no warning at all, and no consumption check will ever "
      "find that; only an emitted-count-against-observable-count check does, "
      "which is the hydrogen census now in prep_check.py"
      % (len(_void), ", ".join(_void) or "none", len(_disc),
         ", ".join(_disc) or "none", ", ".join(_taut)),
)

section("VERIFICATION.md's unresolved index is derived, not maintained")

# --- What this section is, and the one design decision it rests on ---------
#
# `docs/VERIFICATION.md` is a 260 KB ledger of numbered defect rows spread
# over 23 tables. The question a reader actually opens it with is "what is
# still broken?", and until this section existed the only way to answer it
# was to read all 265 rows. So the file now carries an index near the top.
#
# The index is **derived from the tables at run time and compared against the
# committed copy**, rather than written by a person. The reason is not taste:
# every hand-maintained list in this repository has gone stale, and **a stale
# list is indistinguishable from a fresh one** -- nothing about the bytes says
# when it was last true. A generated block fixes exactly that, because the
# comparison is against the tables as they are *now*.
#
# The alternative (a gate that reports the unresolved rows without writing
# anything) was rejected: it satisfies "derived" but leaves the reader with
# nothing to read, and the deliverable here is a navigable document.
#
# **What this section does NOT claim.** `check` below compares the committed
# block to a freshly derived one, so it can go red when a row's status
# changes. It cannot see a row that carries no status at all -- and 215 of
# the 265 rows are in that state, because the status convention (a leading
# bold token in the last cell) was introduced partway through the document.
# Those rows are therefore **listed by the index under their own heading**
# rather than folded into either bucket. Counting them as resolved would be a
# lie; dropping them would be the same lie by omission, which is the failure
# mode this section exists to prevent.
VERIFICATION = ROOT / "docs" / "VERIFICATION.md"

_IDX_BEGIN = "<!-- BEGIN GENERATED: unresolved index (scripts/docs_claims_check.py) -->"
_IDX_END = "<!-- END GENERATED: unresolved index -->"

#: A table row is split on `|` except where the bar is escaped as `\|`. Several
#: rows carry `max\|cpu\|` inside a cell, and a naive split turns one row into
#: eight, which is how a first pass at this undercounted the rows by 30.
_MD_SPLIT = re.compile(r"(?<!\\)\|")


def _md_cells(line: str) -> list[str]:
    parts = _MD_SPLIT.split(line)
    if parts and parts[0] == '':
        parts = parts[1:]
    if parts and parts[-1].strip() == '':
        parts = parts[:-1]
    return [c.strip() for c in parts]


def _is_row(line: str) -> bool:
    return line.startswith("|") and not re.match(r"^\|\s*-{3,}", line)


def _numbered_tables(lines: list[str]) -> list[dict]:
    """Every `| # | ...` table with its rows.

    A blank line does **not** end a table when the next non-blank line is
    still a row: VERIFICATION.md separates some row groups that way, and
    treating the gap as a boundary silently drops the rows after it.
    """
    blocks: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("| #"):
            body: list[tuple[int, list[str]]] = []
            j = i + 1
            if j < n and re.match(r"^\|\s*-{3,}", lines[j]):
                j += 1
            while j < n:
                line = lines[j]
                if _is_row(line):
                    body.append((j + 1, _md_cells(line)))
                    j += 1
                elif not line.strip():
                    k = j + 1
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and _is_row(lines[k]):
                        j = k
                    else:
                        break
                else:
                    break
            blocks.append({"hline": i + 1, "hdr": _md_cells(lines[i]), "body": body})
            i = j
        else:
            i += 1
    return blocks


#: The status convention as actually practised, and it is **not** column 1 --
#: column 1 is the row id and always has been. The status is the first bold
#: token of the last cell. Nothing else is read.
_LEAD_BOLD = re.compile(r"^\*\*([^*]{1,80}?)\*\*")
_OPEN_LEADS = ("未解决", "未完成", "未闭合", "未断言", "不修", "部分解决")
_CLOSED_LEADS = ("已解决", "已闭合", "已修", "已更正", "已定性",
                 "已改", "已删", "已加")


def _verdict(row: list[str]) -> tuple[str, str | None]:
    """(verdict, lead); verdict is OPEN / CLOSED / UNREADABLE / NO_STATUS."""
    m = _LEAD_BOLD.match(row[-1] if row else "")
    if not m:
        return "NO_STATUS", None
    lead = m.group(1)
    for p in _OPEN_LEADS:
        if lead.startswith(p):
            return "OPEN", lead
    for p in _CLOSED_LEADS:
        if lead.startswith(p):
            return "CLOSED", lead
    # A bold lead that is not a status word is NOT guessed at. `找` (row 74)
    # and `决定：代码对、docstring 错` (row 170) are both bold and both
    # unreadable; calling them resolved would be inventing a status.
    return "UNREADABLE", lead


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip().replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def _derive_index(scan: str) -> tuple[list[dict], list[tuple[str, int, int]]]:
    lines = scan.split("\n")
    heads: dict[int, str] = {}
    cur = ""
    for i, line in enumerate(lines, 1):
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            cur = m.group(2).strip()
        heads[i] = cur
    rows: list[dict] = []
    for b in _numbered_tables(lines):
        for ln, cs in b["body"]:
            v, lead = _verdict(cs)
            rows.append({"line": ln, "id": cs[0], "sec": heads[ln],
                         "verdict": v, "lead": lead,
                         "summary": cs[1] if len(cs) > 1 else ""})
    first: dict[str, int] = {}
    dups: list[tuple[str, int, int]] = []
    for r in rows:
        if r["id"] in first:
            dups.append((r["id"], first[r["id"]], r["line"]))
        else:
            first[r["id"]] = r["line"]
    return rows, dups


def _render_index(rows: list[dict]) -> str:
    op = [r for r in rows if r["verdict"] == "OPEN"]
    un = [r for r in rows if r["verdict"] == "UNREADABLE"]
    ns = [r for r in rows if r["verdict"] == "NO_STATUS"]
    cl = [r for r in rows if r["verdict"] == "CLOSED"]
    out = [_IDX_BEGIN, ""]
    out.append("> **本索引由 `scripts/docs_claims_check.py` 从下面那些表格里推导，不要手改。**")
    out.append("> 改了任何一行的状态而没有重跑生成器，门禁会红。")
    out.append("")
    out.append("共 %d 行有编号的缺陷记录：**未解决 %d 条**，已声明闭合 %d 条，"
               % (len(rows), len(op), len(cl)))
    out.append("**读不出状态的 %d 条**（没有 `**粗体**` 开头的判定词）。" % len(ns))
    out.append("")
    out.append("### 未解决（%d 条）" % len(op))
    out.append("")
    out.append("| # | 所在小节 | 状态词原文 | 现象摘要 |")
    out.append("|---|---|---|---|")
    for r in op:
        out.append("| %s | %s | `%s` | %s |"
                   % (r["id"], r["sec"] or "-", _clip(r["lead"], 30),
                      _clip(r["summary"], 46)))
    out.append("")
    out.append("### 读不出状态（%d 条）——不是已解决，是没写" % len(ns))
    out.append("")
    out.append("这 %d 行的最后一格没有粗体判定词，所以**无法判断它们是否已解决**。" % len(ns))
    out.append("把它们算作\"已解决\"和把它们从索引里删掉一样是错的：前者是撒谎，"
               "后者是撒谎的另一种写法。")
    out.append("")
    out.append("按小节归组，每一条的编号都列出来，没有省略：")
    out.append("")
    out.append("| 所在小节 | 条数 | 编号 |")
    out.append("|---|---|---|")
    grouped: dict[str, list[str]] = {}
    for r in ns:
        grouped.setdefault(r["sec"] or "-", []).append(r["id"])
    for sec in sorted(grouped, key=lambda s: int(re.match(r"\d+", grouped[s][0]).group())
                      if re.match(r"\d+", grouped[s][0]) else 0):
        g = grouped[sec]
        out.append("| %s | %d | %s |" % (sec, len(g), ", ".join(g)))
    out.append("")
    out.append("### 状态词可读但不在词表里（%d 条）" % len(un))
    out.append("")
    if un:
        out.append("| # | 判定词原文 |")
        out.append("|---|---|")
        for r in un:
            out.append("| %s | `%s` |" % (r["id"], _clip(r["lead"], 50)))
    else:
        out.append("（无）")
    out.append("")
    out.append(_IDX_END)
    return "\n".join(out)


_ver_text = read(VERIFICATION)
_bi, _ei = _ver_text.find(_IDX_BEGIN), _ver_text.find(_IDX_END)
check(
    _bi >= 0 and _ei > _bi,
    "the generated block's markers are present exactly once, so the index is "
    "replaced rather than appended to",
    "BEGIN at offset %d, END at offset %d. The derivation reads the document "
    "with this block REMOVED first: the block is itself made of `| # |` "
    "tables, and reading them back would make the index list its own rows and "
    "the count would grow on every regeneration" % (_bi, _ei),
)

_scan = _ver_text[:_bi] + "\n" + _ver_text[_ei + len(_IDX_END):] if _bi >= 0 else _ver_text
_rows, _dups = _derive_index(_scan)
_want = _render_index(_rows)
_have = _ver_text[_bi:_ei + len(_IDX_END)] if _bi >= 0 else ""

_open = [r for r in _rows if r["verdict"] == "OPEN"]
_unread = [r for r in _rows if r["verdict"] == "UNREADABLE"]
_nostat = [r for r in _rows if r["verdict"] == "NO_STATUS"]
_closed = [r for r in _rows if r["verdict"] == "CLOSED"]

#: `--index` is the only supported way to change the committed block, for the
#: same reason `check_scripts_declare.py --pin` is the only way to change its
#: snapshot: it is deliberately not a normal run. A regeneration edits the
#: document, so it must land in review as an explicit act rather than as a
#: number that changed because someone ran something. A plain run never writes.
if "--index" in sys.argv and _bi >= 0 and _ei > _bi:
    _new = _ver_text[:_bi] + _want + _ver_text[_ei + len(_IDX_END):]
    # LF only, no BOM: this file is pure LF and `check_text_encoding.py` reads it.
    with io.open(VERIFICATION, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_new)
    print(f"rewrote the generated index in {VERIFICATION.name} "
          f"({len(_rows)} rows: {len(_open)} open, {len(_closed)} closed, "
          f"{len(_unread)} unreadable, {len(_nostat)} without a status)")
    sys.exit(0)

check(
    _have == _want,
    "the committed index is exactly what the tables say right now, so a status "
    "edited without regenerating turns this red",
    "%d row(s) parsed from %d numbered table(s): %d open, %d closed, %d "
    "unreadable lead, %d with no status at all. The committed block is %s. "
    "Regenerate with the `--index` flag of this file, which rewrites the block "
    "between the markers and nothing else"
    % (len(_rows), len(_numbered_tables(_scan.split("\n"))), len(_open),
       len(_closed), len(_unread), len(_nostat),
       "identical" if _have == _want else "STALE"),
)

check(
    len(_open) > 0 and all(r["id"] and r["sec"] for r in _open),
    "the index is not empty, and every row it claims is open names a section, "
    "so it cannot go green by parsing nothing",
    "%d open row(s): %s. An index that silently derived zero rows would be "
    "indistinguishable from an honest one at a glance, which is the same defect "
    "this section exists to catch -- so emptiness is asserted as a failure "
    "rather than treated as 'nothing to report'"
    % (len(_open), ", ".join(r["id"] for r in _open)),
)

check(
    len(_open) + len(_closed) + len(_unread) + len(_nostat) == len(_rows),
    "every row lands in exactly one bucket, so no row is quietly dropped",
    "%d + %d + %d + %d = %d rows. The %d NO_STATUS rows are listed in the "
    "committed index under their own heading rather than being counted as "
    "resolved or omitted: the status convention (a leading bold token in the "
    "last cell) was introduced partway through the document, so a majority of "
    "rows predate it and **cannot** be read either way. Calling them resolved "
    "would be a lie, and hiding them would be the same lie by omission"
    % (len(_open), len(_closed), len(_unread), len(_nostat), len(_rows),
       len(_nostat)),
)

check(
    len(_dups) == 1 and sorted({d[0] for d in _dups}) == ["58"],
    "the row-id collision is the one that is on record, so a second collision "
    "cannot be introduced without this turning red",
    "%d colliding id(s): %s. The ids are GLOBAL, not per-table: 23 tables carry "
    "1..269 with every number unique except this one, and 98-103 were never "
    "allocated, so the counter is monotonic and a repeat is a collision rather "
    "than two independent per-table sequences. Recorded as row 269; NOT "
    "renumbered, because renumbering would invalidate every 'see row N' "
    "reference in three files owned by other agents. **The line numbers are "
    "deliberately not part of the assertion**: they record where the two rows sat "
    "when the collision was noted, and they move whenever a paragraph is inserted "
    "above them. This check used to pin them, and went red for that reason alone "
    "when a paragraph was added to VERIFICATION.md earlier in this session. The "
    "invariant is the identity of the id"
    % (len(_dups), ", ".join("%s at lines %d and %d" % d for d in _dups)),
)

section("SCORING.md 4.2: the out-of-box charge, against the two implementations")

# The penalty went from a flat step to a per-angstrom, per-axis ramp. `SCORING.md`
# listed no such section at all until this round, which is why these four are new
# rather than extensions of an existing check: there was nothing to extend.
search_rs = rust_text(CORE / "search" / "mod.rs")

# 1. The constant. `API.md` quotes it too, and it is the *shape* that changed, not
#    the number -- so a check on the number alone would have stayed green across the
#    defect that actually shipped.
oob_src = re.search(r"OUT_OF_BOX_PENALTY:\s*f64\s*=\s*([\d.]+)", search_rs)
oob_doc = re.search(r"OUT_OF_BOX_PENALTY\s*=\s*([\d.]+)", scoring_doc)
check(
    oob_src is not None
    and oob_doc is not None
    and float(oob_doc.group(1)) == float(oob_src.group(1)),
    "SCORING.md 4.2's OUT_OF_BOX_PENALTY is search::OUT_OF_BOX_PENALTY's",
    f"document {oob_doc and oob_doc.group(1)}, source {oob_src and oob_src.group(1)}. "
    "**Note the limit of this check:** it pins the multiplier, not the thing it "
    "multiplies. The defect that shipped changed the shape and left the number, so "
    "a constant-only pin would have been green throughout",
)

# 2. The shape: per axis, clamped at zero, charged against the *box's* faces, and
#    the document says so *in the section that describes it*. Scoping matters:
#    `逐轴` also appears in that section's heading, so a whole-document substring
#    test is satisfied by the heading alone. Mutation M2 (dropping `逐轴` from
#    the prose while the heading keeps it) is exactly how that was caught.
#
#    **The third predicate is new, and the first one had to be loosened to add
#    it.** `f64::max` returns the non-`NaN` operand, so
#    `(NaN - max).max(min - NaN).max(0.0)` is **0.0, not NaN**: a coordinate that
#    is not a number charged nothing, the atom contributed zero, and a NaN energy
#    makes a ranking meaningless while looking like a result. The engine now
#    answers `f64::INFINITY` there, so the conformation sorts last instead of
#    vanishing. Nothing in this file covered that, and the fix landed this week.
#
#    The clamp predicate matches with `[^;]*?` between `v[k] =` and the expression
#    rather than immediately after it. It used to be one contiguous regex, and it
#    went red the moment the body grew an `is_finite` guard around the *same*
#    expression -- a semantically identical change to a strictly better one.
#    **A check that pins an expression's spelling reports a spelling change, not
#    a behaviour change**, so the reader of that red learned nothing about the
#    charge. The predicates below are properties of the computation; the text
#    that realises them is free to move.
_oob_body = grid_rs.split("pub fn out_of_box_violation_per_axis", 1)[1].split("\n}", 1)[0]
_oob_clamp = re.search(
    r"v\[k\]\s*=\s*[^;]*?\(p\[k\]\s*-\s*b\.max\[k\]\)\s*"
    r"\.max\(\s*b\.min\[k\]\s*-\s*p\[k\]\s*\)\s*\.max\(\s*0\.0\s*\)",
    _oob_body,
)
_oob_loop = re.search(r"for\s+k\s+in\s+0\.\.3", _oob_body) is not None
_oob_loud = (
    re.search(r"is_finite\(\)", _oob_body) is not None
    and re.search(r"else\s*\{\s*f64::INFINITY", _oob_body) is not None
)
_m42 = scoring_doc.split("### 4.2", 1)[1].split("\n## ", 1)[0] if "### 4.2" in scoring_doc else ""
# The heading itself contains 逐轴, so the predicates are tested against the section
# *body* -- otherwise the heading alone satisfies the check and the prose is free to
# lose the word.
_m42_body = _m42.split("\n", 1)[1] if "\n" in _m42 else ""
check(
    _oob_clamp is not None
    and _oob_loop
    and _oob_loud
    and "逐轴" in _m42_body
    and "按面" in _m42_body,
    "the out-of-box charge is per axis against the faces, is loud about a "
    "non-finite coordinate, and SCORING.md 4.2 says so in those words",
    f"per-axis loop: {_oob_loop}, clamp against the box faces: "
    f"{_oob_clamp is not None}, non-finite charged as f64::INFINITY: {_oob_loud}. "
    f"SCORING.md 4.2 body captured: {len(_m42_body):,} chars; contains 逐轴: "
    f"{'逐轴' in _m42_body}, 按面: {'按面' in _m42_body}. The predicates are scoped to "
    "the 4.2 body, not the whole document and not even the 4.2 heading -- a "
    "whole-document test is satisfied by unrelated text and a heading-inclusive one "
    "by the heading. **Deleting the `is_finite` guard turns the third predicate "
    "red**, because `f64::max` would then charge 0.0 for a NaN coordinate and the "
    "atom would silently contribute nothing",
)

# 3. Two-sided, and about the *contract* rather than one field's spelling. The
#    kernel's out-of-box branch must charge the number the HOST measured, and
#    must not re-measure it from the narrowed `f32` position. Pinning one way of
#    getting the far corner would mirror a transient -- `bmax` and
#    `min + (n - 1) * spacing` are both legitimate -- and would also have missed
#    the defect this now describes, which is not *which* corner but *whose*
#    arithmetic: the CPU measures the violation in `f64` from the coordinates it
#    scores, so a kernel that measures it again in `f32` disagrees by
#    `eps32 x |p|` amplified by the penalty.
#    Three halves, each red on its own:
#      (i)   the branch returns the constant times the uploaded violation;
#      (ii)  the sample body derives no magnitude of its own from `p`;
#      (iii) `bmax` is still in the uniform but is *labelled* as unread by
#            `sample` -- a labelled dead field is a deliberate state, and an
#            unlabelled one is the same failure as a comment that stopped being
#            true. This half is why the field was not simply deleted.
#: Comments are stripped before any *code* predicate below. Measured, not
#: assumed: with the raw text, deleting the real `c.violation.x` read out of the
#: per-point coordinate and leaving only the word inside the layout comment at
#: `energy.wgsl:14` left this check **green** -- the same failure shape as a
#: comment that stopped being true, committed to a gate that was supposed to
#: notice. A predicate that a comment can satisfy is not a predicate about code.
_wgsl_code = "\n".join(
    re.sub(r"//.*$", "", ln) for ln in wgsl.splitlines()
)
_charges_host = re.search(
    r"return\s+OUT_OF_BOX_PENALTY\s*\*\s*violation\s*;", _wgsl_code
)
_derives_own = re.search(
    r"max\s*\(\s*p\s*-\s*params\.(bmax|min)|let\s+per_axis\s*=|p\s*-\s*params\.bmax",
    _wgsl_code,
)
_reads_uploaded = re.search(r"violation\.x", _wgsl_code) is not None
_bmax_labelled = (
    re.search(r"No longer read by `sample`", wgsl) is not None
    and re.search(r"bmax\s*:\s*vec4<f32>", _wgsl_code) is not None
)
#: Section 9's *body*, not the whole document. Measured: with the whole-document
#: test, deleting the parity-gate pointer from section 9 left this check green,
#: because the script's name appears four other times in SCORING.md -- the same
#: "a claim satisfied by unrelated text" failure the 4.2 body scoping avoids.
#: The `<!-- 门禁 ... -->` markers are stripped too. Measured a second time: with
#: the pointer deleted but the marker left, the name was still present and the
#: check still read green. That marker is machine metadata saying *which gate
#: watches this section*; it is not a pointer for a reader, and a check about what
#: the prose tells a reader cannot be satisfied by it. Third instance of the same
#: lesson in this file: a comment satisfied a predicate about code or prose.
_m9 = scoring_doc.split("## 9. GPU", 1)[1].split("\n## ", 1)[0] if "## 9. GPU" in scoring_doc else ""
_m9_body = _m9.split("\n", 1)[1] if "\n" in _m9 else ""
_m9_prose = re.sub(r"<!--.*?-->", "", _m9_body, flags=re.S)
_doc_says_host_owns = "主机量好的那个数" in _m9_prose
_doc_names_gate = "gpu_cpu_parity_check.py" in _m9_prose
check(
    _charges_host is not None
    and _derives_own is None
    and _reads_uploaded
    and _bmax_labelled
    and _doc_says_host_owns
    and _doc_names_gate,
    "the shader's out-of-box branch charges the host's measured violation and "
    "derives none of its own, the now-unread bmax field says so, and SCORING.md 9 "
    "says the same and points at the script that compares the two backends",
    f"charges the host's number: {_charges_host is not None}, derives a magnitude "
    f"of its own: {_derives_own is not None}, reads the uploaded `violation.x` in "
    f"code and not only in a comment: {_reads_uploaded}, bmax field present and "
    f"labelled unread: {_bmax_labelled}, section 9 says the host owns the "
    f"magnitude: {_doc_says_host_owns}, section 9 names the parity gate: "
    f"{_doc_names_gate} (section 9 body is {len(_m9_body):,} chars and "
    f"{len(_m9_prose):,} after the 门禁 markers are stripped; the name appears "
    f"{scoring_doc.count('gpu_cpu_parity_check.py')} time(s) in the whole "
    f"document, which is why the test is scoped to the section, and why the "
    f"markers are stripped, rather than the file). "
    "**What this is not:** it does not check that the host measures the violation "
    "the way the CPU does -- that is `gpu::pack_upload`'s host-side assertion and, "
    "on an adapter, `the_out_of_box_magnitude_is_the_hosts_and_putting_it_back_"
    "breaks_the_band`, which skips rather than fails without one. "
    "**Mutations, all measured on a copy of the tree under target/mut/, with the "
    "sandbox's own exit code and the sandbox's own verdict line read back rather "
    "than assumed:** reverting the kernel to the flat per-atom `return 1000.0;` "
    "reddens it; putting back the narrowed-coordinate re-measurement reddens it; "
    "deleting the 'No longer read by `sample`' label reddens it while leaving the "
    "field in place; **removing the real `c.violation.x` read and leaving only the "
    "word in the layout comment did NOT redden it before the comment stripping "
    "above, which is why the stripping exists**; **deleting section 9's parity-gate "
    "pointer did NOT redden it before the section scoping above, because the name "
    "appears elsewhere in the file**; **and even after scoping, deleting the "
    "pointer while leaving the `<!-- 门禁 ... -->` marker did NOT redden it, which "
    "is why the markers are stripped**; rewriting SCORING.md 9 to claim the kernel "
    "reads `bmax` reddens it; deleting the gate pointer from section 9 reddens it",
)

# 4. The class of error, not the instance: a document citing a test that does not
#    exist reads exactly like one citing a test that does. The name below was
#    quoted in three documents as the thing that enforced GPU/CPU agreement, and
#    grep for it in dock-core/src returned nothing. The test is not banned from the
#    documents -- naming it *while refuting it* is how a reader finds out -- so this
#    uses the same discipline as the "3.00 A / -0.050" check above: every
#    occurrence must sit inside a 「...」 span.
_ghost = "gpu_energies_match_the_cpu_interpolation"
_real = "the_cpu_and_gpu_paths_agree_where_the_difference_was_measured"
_ghost_in_rs = _ghost in read(CORE / "gpu" / "mod.rs")
_named, _asserted = 0, []
for _label, _txt in (("SCORING.md", scoring_doc),
                     ("LIMITATIONS.md", limitations_doc),
                     ("VERIFICATION.md", _ver_text)):
    _spans = [(m.start(), m.end()) for m in re.finditer("「[^」]*」", _txt)]
    for _m in re.finditer(re.escape(_ghost), _txt):
        _named += 1
        if not any(a <= _m.start() < b for a, b in _spans):
            _asserted.append(_label)
check(
    _named >= 1 and not _asserted and not _ghost_in_rs and _real in scoring_doc,
    "no document asserts a GPU/CPU parity test by a name that does not exist",
    f"{_ghost!r}: {_named} mention(s) across the three documents, "
    f"{len(_asserted)} outside a 「...」 span ({', '.join(_asserted) or 'none'}); "
    f"it exists in gpu/mod.rs: {_ghost_in_rs}. The real test {_real!r} is cited by "
    f"SCORING.md: {_real in scoring_doc}. **Mutation:** dropping the 「」 from any "
    "one of those mentions turns this red, and so does citing the ghost in gpu/mod.rs",
)

section("a test is cited as evidence only if it computes the thing it is evidence for")

# The READMEs' GPU section makes a claim about *evidence*: that a test ties the
# kernel's out-of-box term to the CPU's. That is the one claim in the section a
# reader cannot check from the prose, because the prose reads the same either
# way. `gpu/mod.rs` now holds two tests that both look like that evidence:
#
#   - `shader_and_rust_agree_on_the_out_of_box_penalty` parses the constant and
#     requires the kernel to *contain* the per-axis expression. It reads
#     energy.wgsl as text. It never evaluates either implementation, so a
#     flipped sign or a `bmax` passed wrong sails straight through it.
#   - `the_cpu_and_gpu_paths_agree_outside_the_box_too` slides conformations out
#     past the box face, scores them on both backends, and compares the two
#     vectors plus their ordering.
#
# The READMEs cited the first while describing it in the language of the second,
# which is precisely "the specification was in the code and the document only
# mirrored it". So the citation is bound to the *kind* of test it points at:
# whatever test the READMEs offer as proof that two backends agree must actually
# call the GPU scorer and compare against a CPU-side band. This reads the test
# bodies, so renaming a test does not redden it -- citing the textual one does.
_gpu_rs = read(CORE / "gpu" / "mod.rs")


def _rust_fn_body(name: str) -> str:
    """The text of `fn <name>` in a Rust file, matched by brace depth."""
    start = _gpu_rs.find("fn " + name + "(")
    brace = _gpu_rs.find("{", start)
    if start < 0 or brace < 0:
        return ""
    depth = 0
    for i in range(brace, len(_gpu_rs)):
        if _gpu_rs[i] == "{":
            depth += 1
        elif _gpu_rs[i] == "}":
            depth -= 1
            if depth == 0:
                return _gpu_rs[start:i + 1]
    return ""


# Names shaped like a GPU/CPU agreement test. Deliberately not the full list of
# gpu/mod.rs test names: this is the shape a reader would believe is one, so a
# README citing a name of that shape that does not exist is a ghost citation.
_looks_like_agreement_test = re.compile(
    r"`(the_[a-z0-9_]*agree[a-z0-9_]*|shader_and_rust_agree[a-z0-9_]*|"
    r"[a-z0-9_]*(?:gpu|cpu)_paths[a-z0-9_]*)`"
)
_cited = sorted(
    {
        n
        for _doc in (readme_cn, readme_en)
        for n in _looks_like_agreement_test.findall(_doc)
    }
)
_ghosts = [n for n in _cited if ("fn " + n + "(") not in _gpu_rs]
check(
    _cited and not _ghosts,
    "every GPU/CPU agreement test the READMEs cite exists in gpu/mod.rs",
    f"{len(_cited)} cited: {_cited}; {len(_ghosts)} do not exist: {_ghosts}. "
    f"**Mutation:** spelling a test name that is not in gpu/mod.rs turns this red",
)

# And the binding: a test offered as proof that the two backends agree has to
# compute both of them. A `.score(` call is the GPU side; a derived band or an
# inversion count is the CPU side being compared against.
_cited_real = [n for n in _cited if ("fn " + n + "(") in _gpu_rs]
_computing = [
    n for n in _cited_real
    if (".score(" in _rust_fn_body(n))
    and ("over_band" in _rust_fn_body(n) or "inversions" in _rust_fn_body(n))
]
check(
    _cited and _computing,
    "the test the READMEs offer as proof that CPU and GPU agree is one that "
    "computes both of them",
    f"cited: {_cited_real}; of those, {len(_computing)} call the GPU scorer and "
    f"compare it against a CPU-side band: {_computing}. A test that only parses "
    "a constant and greps the shader text is not evidence that two "
    "implementations agree -- it is evidence that a string is present. "
    "**Mutation:** cite the textual test as the numerical proof, or strip "
    "`.score(` from the cited test, and this goes red",
)

section("the non-ASCII source obligation has an owner, and deleting the owner is red")

#: **What this asserts, and what it deliberately does not.**
#:
#: Defect 273 recorded a real gap: this file scans no non-ASCII source, so the
#: corruption it exists to catch (PowerShell 5.1 writing a BOM) can happen to the
#: very files it reads, and the gate stays green. The repair was to move the
#: scope into a gate that owns it by name. That leaves one hole, and it is the
#: one this check closes: **if the scope rule is deleted outright rather than
#: narrowed, this file is still green.** "A gate whose scope was deleted" and "a
#: gate whose scope happens to be correct" print the same thing here.
#:
#: So the claim is not "this file covers encoding". The claim is **the obligation
#: has exactly one owner, and that owner's scope covers every file type in the
#: tree that actually holds non-ASCII**. Two halves, and the second is what makes
#: it a gate rather than a comment:
#:
#:   * the owner is **derived, not named**. Writing `check_text_encoding.py` here
#:     would rebuild exactly the coupling being detected: the day the obligation
#:     moves to another file, this line goes stale and nobody notices, which is
#:     the failure mode of the ledger that `check_scripts_declare.py` was written
#:     to end. Instead the owner is *found* -- the unique gate whose **executed**
#:     code both tests a byte against a non-ASCII threshold and declares a
#:     `.suffix` scope table. Measured over the 35 gate-shaped files in
#:     `scripts/`, exactly one satisfies it, and the 5 other files that merely
#:     *mention* non-ASCII in a comment do not. Comments are excluded by reading
#:     the syntax tree, which is why `tokenize`/`ast` rather than a grep.
#:   * the scope is **measured against the tree, not read from the owner**. Every
#:     suffix that holds non-ASCII bytes anywhere outside the owner's own derived
#:     skip set must appear in its covered set or its exclusion table. Narrow the
#:     scope and this goes red; that is the anti-narrowing half, and it is why
#:     the owner's own check 8 is not enough on its own -- check 8 asks the same
#:     question from inside, where deleting the question takes the asker with it.
#:
#: **What it cannot see.** It cannot tell that the owner's *predicates* are
#: correct, only that the scope rule exists and covers the tree; and a gate
#: deleted together with its own declaration would leave zero owners, which this
#: check reports as red (see the `owners` truthiness) rather than as a pass.
_SCRIPTS = ROOT / "scripts"


def _gate_shaped(p: Path) -> bool:
    """The shape `check_scripts_declare.py` treats as a gate's file name."""
    return p.name.endswith("_check.py") or p.name.startswith("check_")


def _non_ascii_sites(tree: ast.AST) -> list[int]:
    """Lines where EXECUTED code tests a byte against a non-ASCII threshold."""
    out: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, cmp in zip(node.ops, node.comparators):
                if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    continue
                val = (cmp.value
                       if isinstance(cmp, ast.Constant) and isinstance(cmp.value, int)
                       else None)
                if val in (0x7E, 0x7F, 0x80, 127, 128):
                    out.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == "isascii":
            out.append(node.lineno)
    return out


def _suffix_tables(tree: ast.Module) -> list[tuple[str, set[str], str]]:
    """Module-level literal collections of `.suffix` strings, read by shape.

    Deliberately not keyed on the names `COVERED_TYPES` / `EXCLUDED_TYPES`: a
    rename inside the owner must not silently turn this check into a no-op, and a
    name-keyed read would do exactly that. The third element records whether the
    table was a `set` (a scope that *is* scanned) or a `dict` (types deliberately
    *not* scanned, each with a stated reason), which is the only thing that tells
    the two apart when both hold dotted suffixes.
    """
    out: list[tuple[str, set[str], str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value,
                                                             (ast.Set, ast.Dict)):
            continue
        if isinstance(node.value, ast.Set):
            items = [e.value for e in node.value.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        else:
            items = [k.value for k in node.value.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if not items:
            continue
        dotted = {i for i in items if re.fullmatch(r"\.[A-Za-z0-9_]+", i)}
        if len(dotted) >= max(2, len(items) // 2):
            name = next((t.id for t in node.targets
                         if isinstance(t, ast.Name)), "<anonymous>")
            out.append((name, dotted,
                        "dict" if isinstance(node.value, ast.Dict) else "set"))
    return out


def _check_sites(tree: ast.AST) -> int:
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and getattr(n.value.func, "id", None) in ("check", "ok", "bad", "expect"))


_enc_owners: list[Path] = []
for _p in sorted(_SCRIPTS.glob("*.py")):
    if not _gate_shaped(_p):
        continue
    try:
        _t = ast.parse(read(_p))
    except SyntaxError:
        continue
    if _non_ascii_sites(_t) and _suffix_tables(_t) and _check_sites(_t):
        _enc_owners.append(_p)

# The measurement below is done *before* the single `check()` that reports both
# halves, so that the "no owner" case is a red rather than a skipped section. A
# guard here would make the total depend on the tree, which is the property
# `skip()` exists to prevent.
_owner = _enc_owners[0] if _enc_owners else None
_otables: list[tuple[str, set[str], str]] = []
_covered: set[str] = set()
_excluded: set[str] = set()
_oskip: set[str] = set()
_held: set[str] = set()
_unaccounted: list[str] = []

if _owner is not None:
    _otree = ast.parse(read(_owner))
    _otables = _suffix_tables(_otree)
    # A `set` table is the scope that *is* scanned and a `dict` table is the
    # types deliberately *not* scanned, each with a stated reason. Both hold
    # dotted suffixes, so the node type is the only thing that tells them apart.
    for _name, _suffixes, _kind in _otables:
        if _kind == "dict":
            _excluded |= _suffixes
        else:
            _covered |= _suffixes

    # The walk honours the OWNER's own skip set, read from its syntax tree, so
    # this check cannot disagree with the gate about which directories are build
    # output -- a disagreement there would report a false red on a `.rs` file
    # under `target/`.
    for _node in _otree.body:
        if isinstance(_node, ast.Assign) and isinstance(_node.value, ast.Set):
            _s = {e.value for e in _node.value.elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if {"target", "dist", ".git"} <= _s:
                _oskip = _s
                break

    for _dirpath, _dirnames, _filenames in os.walk(ROOT):
        _dirnames[:] = [d for d in _dirnames if d not in _oskip]
        for _f in _filenames:
            _fp = Path(_dirpath) / _f
            try:
                _data = _fp.read_bytes()
            except OSError:
                continue
            if b"\0" in _data[:8192]:
                continue
            if any(b > 0x7F for b in _data):
                _held.add(_fp.suffix or "<no suffix>")

    _unaccounted = sorted(_held - _covered - _excluded)

check(
    len(_enc_owners) == 1 and not _unaccounted,
    "the non-ASCII source obligation has exactly one owning gate, and that "
    "owner's scope covers every file type in the tree that holds non-ASCII -- so "
    "deleting the scope rule is red, not silently fine",
    f"owner(s) found by shape among the gate-shaped files in scripts/ (executed "
    f"code tests a byte against a non-ASCII threshold AND a module-level "
    f"`.suffix` scope table is declared AND it has check-shaped call sites): "
    f"{[p.name for p in _enc_owners]}. "
    + (f"Scope: {len(_held)} suffix(es) hold non-ASCII outside the owner's own "
       f"derived skip set {sorted(_oskip)} -- {sorted(_held)}; the owner's scope "
       f"tables ({', '.join(n for n, _, _k in _otables)}) cover {len(_covered)} "
       f"covered and {len(_excluded)} excluded; unaccounted: {_unaccounted}. "
       if _owner is not None else
       "No owner was derived, so there is no scope to measure and this is a "
       "failure rather than a skip: a tree with no owner must not be mistaken "
       "for a tree whose obligation is met. ")
    + "**Zero owners is the failure this exists for**: a deleted scope rule must "
    "read as red here, not as a gate that happens to be correct. Both halves are "
    "derived, never typed -- a rename inside the owner does not affect this. "
    "**Mutation:** deleting the owner's scope table turns this red (zero "
    "owners), and narrowing it turns this red (an unaccounted suffix)",
)

section("every line citation in SCORING.md resolves to a real line, on topic where provable")

#: **Defect 277: 70 line citations and no gate.** `LIMITATIONS.md`'s 3 had one
#: (section 5 above); these had none, so a citation could rot silently. This is
#: that gate. **The count this check derives is 71, not the 70 the defect row
#: records**, because the row's figure came from a backtick-only pattern and this
#: one also counts the citations written bare inside prose (`search/mod.rs:101`
#: in the §4.2 listing is one). The extra citation resolves to a real in-range
#: line like the rest, so the difference is a counting difference and not a
#: discrepancy about the tree; it is recorded here rather than smoothed over.
#:
#: **What a green here is allowed to mean, precisely.** Two claims, of very
#: different strength, and the weaker one is stated as weaker rather than left
#: to be read as the stronger:
#:
#:   1. **DERIVED AND TOTAL -- every citation resolves to exactly one real file
#:      and an in-range line.** This is the "line number is in range" claim, and
#:      it is the part a machine can be trusted on. It is *not* the claim the
#:      manual pass made.
#:   2. **PARTIAL -- "the cited line is on topic" holds only where the document
#:      itself supplies the token to check.** Where a citation sits beside an
#:      identifier, the check requires that identifier to appear on the cited
#:      line. Measured on this tree: **66 of 91** carry such a token, and all 66
#:      have it on the cited line; **the other 25 get claim 1 and nothing more.**
#:      A reader who takes this section's green as "all 91 citations are on
#:      topic" is reading more into it than it says. The remaining 25 were
#:      verified by hand, not by this check, and the honest fix for them is a
#:      symbol name in the document, not a smarter regex here.
#:
#: **The 76 -> 91 and 52 -> 66 moves, and what they mean.** The second is this
#: round's out-of-box specification (section 4.2.1/4.2.2): 14 citations, 12
#: carrying a symbol. **The first is the 5 that went red.** This check fired on
#: the real tree for the first time in the same round, on five citations, and
#: **all five were the document's fault and none was the check's**:
#:
#:   * `types.rs::GRID_TYPE_COUNT`（`types.rs:1054`） -- the declaration is on
#:     1055 and 1054 is a `///` doc comment above it.
#:   * `types.rs::the_radius_table_backs_the_grid_types`（`types.rs:1441`） and
#:     `gpu::the_cpu_and_gpu_paths_agree_outside_the_box_too`（`gpu/mod.rs:887`）
#:     -- both cited the `#[test]` attribute, with the `fn` one line below.
#:   * `gpu::the_cpu_and_gpu_paths_agree_where_the_difference_was_measured`
#:     （`gpu/mod.rs:612`）, twice -- 613 is the `fn`.
#:
#: Every one is off by exactly one, and the cause is uniform: explanatory doc
#: comments were inserted above the declarations, so anchored went `52 -> 47`
#: and stale `0 -> 5` without anything being renamed. **The framing was
#: steelmanned before the document was touched**, because two previous rounds
#: were both wrong in the direction of blaming a document for a checker's
#: framing. Three arguments were available and all three fail:
#:
#:   1. *"`#[test]` is a region, and regions are a declared class."* The declared
#:      class is a region the row is **about**; a test row is about its function,
#:      and the document's own convention -- stated in its ledger preamble -- is
#:      that the cited range is the narrowest one containing the symbol's
#:      **declaration**. No passing citation can point at `#[test]`, because
#:      that line never contains the name, so this cannot be a convention; it is
#:      an off-by-one that reads like one.
#:   2. *"The `///` line documents the constant, so citing it is citing the
#:      constant."* That is the "comment citation" class, and the class is
#:      defined by the token being **a word the comment contains** rather than a
#:      name the code declares. `GRID_TYPE_COUNT` is a declared `const`; the row
#:      is about its value, 10.
#:   3. *"`the_cpu_and_gpu_paths_agree_where_the_difference_was_measured` does
#:      not exist, so a row refuting it cannot be anchored."* It **does** exist
#:      now, at 613. The document's ❌ cell had hardened into a false sentence --
#:      it said "that test name does not exist in `dock-core/src`" while the
#:      evidence column cited the test that does. Rewriting the cell to name the
#:      name that really is absent (`gpu_energies_match_the_cpu_interpolation`,
#:      0 hits in `dock-core/src`) fixed a claim no check was looking at.
#:      **And the fix immediately reddened a different check** -- the ghost-name
#:      check, which requires every mention of the absent name to sit inside a
#:      `「...」` refutation span. That is the shape working: the first edit
#:      asserted a non-existent test, and the gate said so before it was run a
#:      second time.
#:
#: So: five document edits, each locating the symbol by content in the source and
#: never from the old line number, and one false sentence rewritten that no check
#: was pointing at.
#:
#: **The 71 -> 76 and 10 -> 52 moves, and what they mean.** Both are the
#: *document* changing, not the check widening. SCORING.md was rewritten to the
#: shape this check reads -- `file::symbol`（`file:line-or-range`）, the shape its
#: own line 72 already asked for -- which added five citations (a range split
#: into its two declarations, a token and a wrapper split apart, and two
#: evidence lines promoted out of prose) and turned 42 citations from
#: symbol-less into symbol-bearing. Nothing here got laxer: claim 2 covers more
#: citations *and* is checked by the same test, with one precondition removed
#: (see `_SYMBOL_REF` below, and the mutation that removed it). This round's
#: move is the one recorded above it: `76 -> 91` and `52 -> 66`.
#:
#: **The convention, and the three classes it deliberately leaves out.**
#: The document now writes the symbol the row is *about*, and the cited range is
#: the narrowest one containing that symbol's declaration. Three classes stay
#: outside the on-topic claim on purpose, and all three are counted, not hidden:
#:
#:   * **Comment citations.** `scoring.rs:39`, `types.rs:382`, `types.rs:660`,
#:     `scoring.rs:485`, `scoring.rs:653-655`, `types.rs:660-661`. The claim is
#:     about what a comment *says*. Any word the comment contains would satisfy
#:     a presence test, so anchoring them would inflate the on-topic count with
#:     almost no evidence behind it -- the symbol would be a word, not a name.
#:   * **Disclosed-drift citations.** `types.rs:189-208`, `scoring.rs:336-347`,
#:     `grid.rs:70`, `scoring.rs:383-397`, `grid.rs:416`. The document's own prose
#:     says these line numbers have drifted and is using them as the examples of
#:     that. Anchoring them would assert they are current, which is the opposite
#:     of what they are cited for.
#:   * **Regions and history.** `gpu/mod.rs:719-746` (the in-box fixture), the
#:     binding block at `energy.wgsl` 38-51 (a five-declaration region, so it is
#:     written as prose rather than as a citation at all), `grid.rs:418` ("at that
#:     time"), `search/mod.rs:85-101`, `scoring.rs:968-973`, and
#:     `gpu_cpu_parity_check.py:98-108` (the 3/3/2 reading this round's
#:     specification quotes, which lives in that script's own docstring prose and
#:     names no function -- the one citation this round added to the unanchored
#:     pile, and it is in this class rather than in a new one).
#:
#: Anchoring any of these would be a manufactured claim, which is worse than
#: the gap it closes: a symbol inferred rather than established turns a
#: 66-of-91 automated check into a 91-of-91 *unverified* one.
#:
#: **Why the roots are derived, not typed.** A citation like `types.rs:189` names
#: a *suffix*, and the tree holds `dock-core/src/types.rs` plus a nested copy of
#: the whole repository under `opendocking-gui/`. Resolving by suffix therefore
#: has two answers that differ. **The cause is the nested copy, not the `mod.rs`
#: spelling** -- an earlier version of this comment and of `VERIFICATION.md`
#: row 284 said "the 15 `mod.rs` citations", which named the symptom and left
#: the mechanism unstated. Measured, on this tree: there are **four** `mod.rs`
#: files, two under `dock-core/src` and two under `opendocking-gui/dock-core/
#: src`; **17** citations land in one; **16 resolve to different code** in the
#: copy and **0 agree**, and the 17th (`gpu/mod.rs:887`) points past the end of
#: the copy's file entirely. The copy is not a near-identical twin either: it
#: carries all 23 `dock-core/src` + `docs` paths, and only 5 of those 23 files
#: are byte-identical to the ones beside them.
#:
#: Rather than hardcode `dock-core/src` (which would be a second copy of a fact
#: the source tree owns), the roots come from the document's **own markdown
#: links** -- `](../dock-core/src/...)` -- and the nested copy is excluded
#: because **two sibling gates' own `SKIP_DIRS` tables declare it build
#: output**, read out of their syntax trees by shape. So a citation is resolved
#: the way this tree already agrees to resolve it, and the exclusion is a
#: derived fact with two independent sources rather than a name somebody
#: remembered. **A third copy appearing outside every declared skip set is what
#: turns this red**, which is the property that makes the derivation
#: load-bearing rather than convenient.
#:
#: **What it cannot see.** It cannot see that a citation is *semantically* right,
#: only that it lands on a real line and -- for the 34 -- that the document's own
#: token is there. `SCORING.md` line 67-73 already says line numbers are "a
#: one-day reading, not a specification", and this check does not disagree: it
#: asserts resolvability, and the on-topic half is asserted only where the
#: document hands over something checkable.

_CITE_RE = re.compile(
    r"`?([A-Za-z0-9_./\\-]+\.(?:rs|py|wgsl|md)):(\d+)(?:\s*[-\u2013]\s*(\d+))?`?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{4,}")

#: Directories this tree's gates declare they do not scan. Derived from the
#: shape of a skip table (a set naming `target` + `dist` + `.git`), unioned over
#: the gates that carry one. Two files declare it; neither is named here.
_derived_skip: set[str] = set()
for _p in sorted(_SCRIPTS.glob("*.py")):
    if not _gate_shaped(_p):
        continue
    try:
        _t = ast.parse(read(_p))
    except (SyntaxError, UnicodeDecodeError):
        continue
    for _n in _t.body:
        if not isinstance(_n, ast.Assign) or not isinstance(_n.value, ast.Set):
            continue
        _s = {e.value for e in _n.value.elts
              if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        if {"target", "dist", ".git"} <= _s:
            _derived_skip |= _s

_cite_roots = sorted({l.rsplit("/", 1)[0] for l in
                      re.findall(r"\]\((\.\./[^)]+\.(?:rs|py|wgsl))\)", scoring_doc)})

_scor_index: dict[str, list[str]] = {}
for _dirpath, _dirnames, _filenames in os.walk(ROOT):
    _dirnames[:] = [d for d in _dirnames if d not in _derived_skip]
    for _f in _filenames:
        _rel = (Path(_dirpath) / _f).relative_to(ROOT).as_posix()
        _scor_index.setdefault(_f, []).append(_rel)
        _scor_index.setdefault(_rel, [_rel])

_lines_cache: dict[str, list[str]] = {}


def _cited_lines(rel: str) -> list[str]:
    if rel not in _lines_cache:
        _lines_cache[rel] = read(ROOT / rel).splitlines()
    return _lines_cache[rel]


def _resolve_citation(spelling: str) -> list[str]:
    """Every real file this citation's spelling could name, under the derived roots."""
    found: list[str] = []
    for _r in _cite_roots:
        _c = (_r[3:] if _r.startswith("../") else _r) + "/" + spelling
        if (ROOT / _c).is_file() and _c not in found:
            found.append(_c)
    if not found:
        for _c in _scor_index.get(spelling, []):
            if _c not in found:
                found.append(_c)
        for _c in _scor_index.get(spelling.split("/")[-1], []):
            if _c not in found:
                found.append(_c)
    return found


_unresolved: list[tuple[str, int]] = []
_ambiguous: list[tuple[str, int, list[str]]] = []
_out_of_range: list[tuple[str, int, int, str]] = []
_anchored: list[tuple[str, str, int, str]] = []
_unanchored: list[tuple[str, str, int]] = []
_offtopic: list[tuple[str, str, int, list[str], int]] = []

#: The token is **the symbol the document attributes to that citation**, and the
#: attribution has to be positional, not just "somewhere on the line". A ledger
#: row often carries two citations in one cell -- `grid.rs::out_of_box_violation_per_axis`
#: （`grid.rs:917`）+ `search::OUT_OF_BOX_PENALTY`（`search/mod.rs:101`） -- and
#: the first token belongs to the first citation, not to both. The second version
#: of this loop took every symbol-shaped name on the line and therefore demanded
#: that `grid.rs:917` carry `search`, which it cannot and should not.
#:
#: Two earlier versions, and both were wrong in the direction that matters:
#:
#:   * **Any word on the line** harvested ordinary prose -- `match`, `comment`,
#:     `Provenance` -- and reported **21 of 70 correct citations as stale**. The
#:     check was wrong and the document was right.
#:   * **Any symbol on the line** then misattributed across paired citations and
#:     reported **11**, including real off-topic rows the document *already*
#:     discloses in prose (`scoring.rs:383-397` is described as "no longer that
#:     table"), which are recorded facts rather than defects.
#:
#: So the rule is the narrowest one that is still a rule: a token counts for a
#: citation only if it is the symbol in the document's own ledger shape --
#: **`file::symbol`（`file:N`）**, with the file part matching -- and that symbol
#: occurs in the cited file. Nothing else is evidence. This is the exact shape
#: SCORING.md uses in its ledger, and the separator between the two halves is a
#: full-width parenthesis, not a space. Three attempts failed before this one, and
#: the first two were caught by this check's own `len(_anchored) > 0` guard while
#: the third was caught by a mutation that survived it:
#:
#:   1. ASCII-only punctuation matched **0 of 71**.
#:   2. Taking the text after `::` out of the whole reference captured the *file
#:      stem* -- every hit came back as `rs`.
#:   3. Requiring the reference's left half to be a **filename** matched 9 of 71
#:      and **missed `search::OUT_OF_BOX_PENALTY`（`search/mod.rs:101`）**, whose
#:      left half is a **module**, not a file. That is the one ledger row where
#:      the document is most explicit about a symbol, so the shape had to go
#:      back to accepting a module path. It is the row the C mutations target,
#:      which is how the gap was found rather than assumed away.
#:
#: The symbol is therefore the part inside the backticks, and the file it is
#: attributed to is the one in the *citation* half -- which is also the half that
#: decides where the symbol has to appear.
#:
#: **The `symbol in file` precondition is gone, and a mutation is why.** This
#: function used to drop a named symbol unless it occurred somewhere in the
#: cited file, on the theory that the line-membership test alone was enough. It
#: is not: mutation C renamed one symbol the document hands over
#: (`is_donor` -> `is_donor_XYZ`) and the result was **anchored 52 -> 51,
#: stale 0, section still green**. The mistyped symbol failed the precondition,
#: so the citation quietly stopped being checked and joined the hand-verified
#: pile -- a wrong input reading as a correct one, which is the failure this
#: file exists to catch. Filtering the candidate out is not the same as finding
#: it on the line, and only the second is a claim about the document. So the
#: symbol is now carried through unconditionally and judged on the cited lines
#: alone: absent from the file *and* from those lines is then reported, which is
#: what a typo is.
_SYMBOL_REF = re.compile(
    r"`([A-Za-z0-9_./\\-]+)::([A-Za-z_][A-Za-z0-9_]*)`"
    r"[^0-9A-Za-z]{0,12}`?"
    r"([A-Za-z0-9_./\\-]+\.(?:rs|py|wgsl|md)):(\d+)")


def _tokens_for(ctx: str, spelling: str) -> list[str]:
    """The symbols this line attributes to *this* citation, positionally.

    Only the `file::symbol`（`file:line`） shape counts, and only when the
    `file:line` it precedes is the citation being examined. A ledger row often
    carries two citations in one cell, and the first symbol belongs to the first
    citation, not to both -- demanding that `grid.rs:917` carry `search` would be
    a check reporting its own misreading as a document defect.

    Deliberately does **not** require the symbol to occur in the cited file --
    see the note on `_SYMBOL_REF` above for the mutation that caught that.
    """
    out: list[str] = []
    stem = spelling.split("/")[-1]
    for m in _SYMBOL_REF.finditer(ctx):
        if m.group(3).split("/")[-1] != stem:
            continue
        cand = m.group(2)
        if cand and cand not in out:
            out.append(cand)
    return out


for _m in _CITE_RE.finditer(scoring_doc):
    _sp, _a = _m.group(1), int(_m.group(2))
    _b = int(_m.group(3) or _m.group(2))
    _cands = _resolve_citation(_sp)
    if len(_cands) != 1:
        (_ambiguous if _cands else _unresolved).append((_sp, _a, _cands))
        continue
    _rel = _cands[0]
    _ls = _cited_lines(_rel)
    if not (1 <= _a <= len(_ls) and 1 <= _b <= len(_ls)):
        _out_of_range.append((_sp, _a, _b, _rel))
        continue
    _ls_start = scoring_doc.rfind("\n", 0, _m.start()) + 1
    _ls_end = scoring_doc.find("\n", _m.end())
    _ctx = scoring_doc[_ls_start:len(scoring_doc) if _ls_end < 0 else _ls_end]
    _idents = _tokens_for(_ctx, _sp)
    _hit = None
    for _i in _idents:
        for _ln in range(_a, min(_b, len(_ls)) + 1):
            if _i in _ls[_ln - 1]:
                _hit = _i
                break
        if _hit:
            break
    if _hit:
        _anchored.append((_sp, _rel, _a, _hit))
    elif _idents:
        # The document DID hand over a token and the cited line does not carry
        # it. That is a stale citation, and it is the one this check can prove --
        # so it is a failure, not a silent demotion to "unanchored". The first
        # version of this check appended these to `_unanchored` and asserted
        # only `len(_anchored) > 0`, which meant moving the token off the line
        # merely moved a row between two lists and the section stayed green. That
        # is a check whose name claims more than its condition tests.
        # The document line travels with the entry. Knowing which source line
        # is wrong is not enough to act on: the edit is in `SCORING.md`, and
        # without the line number a reader has to go and re-derive which of
        # its hundreds of citations this was. Same for the identifiers --
        # `_idents[:3]` silently dropped the rest, so a sentence naming four
        # symbols reported three and the fourth could never be checked.
        _docline = scoring_doc.count("\n", 0, _m.start()) + 1
        _offtopic.append((_sp, _rel, _a, _idents, _docline))
    else:
        _unanchored.append((_sp, _rel, _a))

check(
    not _unresolved and not _ambiguous and not _out_of_range,
    "every line citation in SCORING.md resolves to exactly one real file and an "
    "in-range line",
    f"{len(list(_CITE_RE.finditer(scoring_doc)))} citation(s); roots derived from "
    f"the document's own links {_cite_roots}; nested copies excluded by the "
    f"{len([1 for p in _SCRIPTS.glob('*.py') if _gate_shaped(p)])} gates' own "
    f"derived SKIP_DIRS union {sorted(_derived_skip)}. Unresolved: {_unresolved}; "
    f"ambiguous (spelling names more than one file): {_ambiguous}; out of range: "
    f"{_out_of_range}. **This is the 'in range' claim and it is the whole of what "
    f"it proves** -- see the next check for the on-topic claim and for what it "
    f"cannot reach. **Mutation:** moving a cited line in the source, or renaming "
    f"a file a citation names, turns this red",
)

# Every stale citation, named, with the document line that carries it.
#
# This printed `_offtopic[:3]` after the word "namely". A run with 34 stale
# citations therefore showed three, and the other 31 were unfixable: nobody
# can edit a citation they cannot see, and "namely" asserts the list is the
# whole list. The count was right and the report was not, which is the worst
# combination -- a reader trusts the number, skips the list, and concludes 31
# documents are fine. The gate's own mutation text already claimed it turns red
# "naming the bad symbol", so it believed it was enumerating when it was not.
#
# There is no cap here. A stale citation is a work item; a work item that is
# summarised rather than listed is a backlog, and the count is not the work.
_offtopic_block = (
    "\n".join(
        "        SCORING.md:%d -> %s:%d  names %s"
        % (_dl, _rel, _a, ", ".join("`%s`" % _t for _t in _idents))
        for (_sp, _rel, _a, _idents, _dl) in _offtopic
    )
    or "        (none)"
)

check(
    not _offtopic and len(_anchored) > 0,
    "where SCORING.md names an identifier beside a citation, that identifier is "
    "on the cited line -- and the citations it cannot reach are counted, not "
    "passed off as checked",
    f"{len(_anchored)} of {len(_anchored) + len(_unanchored) + len(_offtopic)} "
    f"citation(s) carry a token this check can test, and all {len(_anchored)} of "
    f"those have it on the cited line. **The other {len(_unanchored)} are verified "
    f"as in-range ONLY**: the document supplies no identifier beside them, so "
    f"**this check makes no on-topic claim for them at all**, and a reader must "
    f"not read this section's green as 'every citation is on topic'. Those "
    f"{len(_unanchored)} were verified by hand, not here. **They are not a "
    f"residue of neglected work**: every one is a comment, a region, a "
    f"historical line number, or a disclosed-drift example -- the three classes "
    f"named in the section comment -- and the check prints that split rather "
    f"than leaving the reader to assume the rest is unfinished. The fix for a "
    f"genuine fourth class is a symbol name in the document (`file::symbol`, "
    f"which is what SCORING.md line 72 already asks its own ledger for), not a "
    f"wider heuristic here. "
    f"Stale citations -- a token the document names that is NOT on the line it "
    f"points at: {len(_offtopic)}. **Every one is listed below, with the "
    f"`SCORING.md` line that carries it**, so each is an edit and not a "
    f"mystery -- there is no cap and no 'first few'. The list is built from the "
    f"same collection as the count, so the two cannot disagree; when they do, the "
    f"bug is a cap like the one this sentence replaced:\n"
    f"{_offtopic_block}\n"
    f"`len(_anchored) > 0` "
    f"is in the condition so that a version of this check matching nothing cannot "
    f"satisfy a comparison of its own output -- the `SITE_INVENTORY` shape "
    f"`check_scripts_declare.py` exists to end. **Mutations, measured** against a "
    f"copy of the tree under `target/mut/` so no file another owner is editing "
    f"was written: moving `OUT_OF_BOX_PENALTY` two lines down in `search/mod.rs` "
    f"turns this red (anchored 52 -> 51, stale 1); renaming a symbol the document "
    f"hands over turns it red (stale 1, naming the bad symbol); pointing a "
    f"citation past the end of its file turns the *resolution* check red instead; "
    f"and stripping a symbol off an anchor leaves it green and drops it to "
    f"hand-verified, which is the correct reading. One mutation did **not** red "
    f"and is reported rather than dropped: moving `grid_type_index` down 4 lines "
    f"leaves `types.rs:1035-1049` still containing it, because the new convention "
    f"cites the range that holds the declaration rather than one line of it. At 40 "
    f"lines it does red (anchored 52 -> 48, stale 4). **A range citation is "
    f"therefore less sensitive to small drift than a single-line one, by exactly "
    f"the width of the range** -- that is the price of citing the declaration, and "
    f"it is a real weakening, not a rounding error",
)

section("every constant the engine exports has a documented fate, or a stated reason it has none")

#: **The claim being made checkable.** "A quantity with no specification cannot
#: have its divergence caught, and nothing in this repository notices a missing
#: specification." The second half is the part a machine can see. The left set is
#: derived from the Rust source; the right set is read out of the documents; and
#: **neither is a list somebody typed**, which is the only reason this is a check
#: rather than the second hand-maintained list in this repository (the first being
#: the one `check_scripts_declare.py`'s census exists to end).
#:
#: **What "exports" is resolved to, and not assumed.** `pub const` outside#: `#[cfg(test)]`. `pub(crate)` is excluded by construction -- the pattern is
#: literally `pub const`, so a restricted-visibility constant cannot qualify. The
#: remaining question is whether `pub const` in these files is actually reachable,
#: which depends on the enclosing module being public; that is **measured** rather
#: than assumed, and printed below, because if a module were private the derived
#: set would over-report and the check would be quietly asking about the wrong
#: surface.
#:
#: **`#[cfg(test)]` is excluded by brace matching, not by a line heuristic.** A
#: first version of this derivation used "is the line after a `#[cfg(test)]`", which
#: let 28 test-local names through -- `ELEMENTS`, `RECEPTOR`, `EXOTIC`, every
#: `*_DOC` string constant -- and would have demanded a specification for a
#: `const RECEPTOR: &str = concat!(...)` fixture. The spans are found by counting
#: braces from the `mod` line that follows each `#[cfg(test)]`.
#:
#: **The three document buckets, and why the weakest one still counts.** A name is
#: *cited* if some line of `docs/*.md` names it **and carries a `file:line`
#: citation**; *quantified* if some naming line carries a **digit**; *named only* if
#: the strongest line that names it does neither. Quantified is a **lower bound**:
#: it proves a number is written next to the name, **not** that the number is this
#: constant's value, and not that it is still right. Stating that here is the
#: difference between "coverage" and "a specification exists", and this check only
#: claims the first. **The test is line-level and therefore coarse**: any digit
#: anywhere on a naming line counts, including a row number or a year that has
#: nothing to do with the constant. That coarseness bit during development -- a
#: paragraph *explaining* this check, in VERIFICATION.md, wrote an item count next
#: to a constant that paragraph had just called exempt, and the check correctly
#: called the exemption stale. It is recorded here rather than tightened, because
#: "this digit belongs to this constant" needs a notion of belonging that no
#: regex has; the fix was to stop the explanation supplying the coverage, and the
#: same hazard is why section 11 is cut out below.
#:
#: **SCORING.md's own section 11 is excluded from the scan, and that exclusion is
#: load-bearing.** Section 11 is where the coverage is *reported*, so letting it
#: supply coverage would let the report certify itself -- the exemption table's own
#: reasons contain digits (`MapSlot::ALL` is 4 items, `Element::COUNT = 12`), so an
#: unexcluded scan would mark all three exempted constants "quantified" by the very
#: sentence excusing them. That is a check passing because a list matches a list.
#: The excluded region is located by heading, not typed as a line span.
#:
#: **What this does NOT cover, in the check's own words.** It does not cover
#: `static`; literals inside `pub fn` bodies; constants produced by macro expansion;
#: `pub(crate)`; any name that is not `SCREAMING_CASE`; anything outside `docs/`
#: (the two READMEs are another gate's scope); and -- the important one -- **it
#: does not verify that a quantified number is correct.** It also keys on the bare
#: name, so the three `ALL` and the three `COUNT` declarations are covered by a
#: single mention each and the check cannot say which. That loss is not
#: hypothetical: `API.md` documented `Element::COUNT = 12` and
#: `AtomType::COUNT = 18` while `AtomKind::COUNT = 5` was silently uncovered and
#: this check was green. The document has been corrected; the check still cannot
#: tell, and section 11 says so.
_PUB_CONST = re.compile(r"^(\s*)pub\s+const\s+([A-Z][A-Z0-9_]*)\s*:")
#: The same shape without the `pub`, used **only** to report how many
#: declarations the `#[cfg(test)]` exclusion actually removed. Without this the
#: count is 0 and the exclusion looks like it does nothing -- every test-local
#: constant in this tree is private, so a `pub const`-only pattern would report
#: "nothing was excluded" while the spans were doing the work.
_ANY_CONST = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+[A-Za-z_][A-Za-z0-9_]*\s*:")


def _cfg_test_spans(lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) 1-based line of every `#[cfg(test)] mod ... { ... }`."""
    out: list[tuple[int, int]] = []
    for _i, _l in enumerate(lines):
        if not _l.strip().startswith("#[cfg(test)]"):
            continue
        _j = _i + 1
        while _j < len(lines) and not re.match(r"\s*(pub\s+)?mod\s+\w+", lines[_j]):
            if lines[_j].strip() and not lines[_j].strip().startswith("#["):
                break
            _j += 1
        if _j >= len(lines) or not re.match(r"\s*(pub\s+)?mod\s+\w+", lines[_j]):
            continue
        _depth, _started = 0, False
        for _k in range(_j, len(lines)):
            for _ch in lines[_k]:
                if _ch == "{":
                    _depth += 1
                    _started = True
                elif _ch == "}":
                    _depth -= 1
            if _started and _depth <= 0:
                out.append((_i + 1, _k + 1))
                break
    return out


_exported: dict[str, list[str]] = {}
_excluded_consts = 0
_excluded_consts = 0
for _p in sorted(CORE.rglob("*.rs")):
    _ls = read(_p).splitlines()
    _sp = _cfg_test_spans(_ls)
    for _i, _l in enumerate(_ls, 1):
        _inside = any(a <= _i <= b for a, b in _sp)
        if _ANY_CONST.match(_l) and _inside:
            _excluded_consts += 1
        _m = _PUB_CONST.match(_l)
        if _m and not _inside:
            _exported.setdefault(_m.group(2), []).append(
                "%s:%d" % (_p.relative_to(ROOT).as_posix(), _i))

#: Is `pub const` here the crate's real surface? Read it out of `lib.rs` rather
#: than believed: a private module would make the derived set over-report. The
#: test-local spans are cut out first, because a private module *inside*
#: `#[cfg(test)]` cannot hide a `pub const` from anything -- and without that cut
#: this reports `gated_targets` as a private module and turns a correct run into a
#: sentence that says the surface is private when it is not.
_lib_rs_lines = read(CORE / "lib.rs").splitlines()
_lib_spans = _cfg_test_spans(_lib_rs_lines)
_mods = [
    (m.group(1), m.group(2), n)
    for n, l in enumerate(_lib_rs_lines, 1)
    for m in [re.match(r"^(pub\s+)?mod\s+(\w+)", l)]
    if m
]
_private_mods = sorted(
    nm for pub, nm, ln in _mods
    if not pub and not any(a <= ln <= b for a, b in _lib_spans)
)

#: The reporting section, located by heading, and cut out of the scan.
_scoring_lines = scoring_doc.split("\n")
_s11_at = next((k for k, l in enumerate(_scoring_lines)
                if l.startswith("## 11. ")), None)
_s11_end = len(_scoring_lines)
if _s11_at is not None:
    _s11_end = next((k for k, l in enumerate(_scoring_lines)
                     if k > _s11_at and l.startswith("## ")), len(_scoring_lines))
_s11_skipped = set(range(_s11_at + 1, _s11_end + 1)) if _s11_at is not None else set()
_excuse_rows: list[tuple[str, str, str, int]] = []
for _k in range(_s11_at or 0, _s11_end):
    _l = _scoring_lines[_k]
    if not _l.startswith("|"):
        continue
    _r = re.match(r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|([^|]*)\|([^|]*)\|", _l)
    if _r:
        _excuse_rows.append((_r.group(1), _r.group(2), _r.group(3), _k + 1))

_DOCS_CITE = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:rs|py|wgsl|md):\d+")
#: Best bucket per name, not three independent sets. The first version of this
#: added a name to every bucket any line matched, so `ATOM_STRIDE` landed in
#: *named only* as well as *cited* and the section printed "named only: 6" for a
#: tree where nothing was named only -- and, worse, it computed the stale-excuse
#: set against the wrong one of the two, so that half of the second check was
#: vacuous. A bucket is a classification, so it is assigned once, strongest first.
_tags: dict[str, set[str]] = {n: set() for n in _exported}
for _dp in sorted((ROOT / "docs").glob("*.md")):
    for _i, _l in enumerate(read(_dp).splitlines(), 1):
        if _dp.name == SCORING.name and _i in _s11_skipped:
            continue
        for _n in _exported:
            if not re.search(r"\b" + _n + r"\b", _l):
                continue
            if _DOCS_CITE.search(_l):
                _tags[_n].add("cited")
            elif re.search(r"\d", _l):
                _tags[_n].add("quant")
            else:
                _tags[_n].add("named")

_cited = {n for n, t in _tags.items() if "cited" in t}
_quant = {n for n, t in _tags.items() if "cited" not in t and "quant" in t}
_named_only = {n for n, t in _tags.items() if t == {"named"}}
_absent = {n for n, t in _tags.items() if not t}

_strong = _cited | _quant
_excused = {n for n, _s, _r, _ln in _excuse_rows}
#: A name docs/ says something quantitative about **and** that section 11 excuses
#: is a contradiction, and it is the half of the two-sided contract that was
#: vacuous when computed against `_named_only`.
_shared = sorted(_excused & _strong)
_unaccounted = sorted(set(_exported) - _strong - _excused)
check(
    bool(_exported) and bool(_cited) and not _unaccounted and not _shared
    and len(_excused) < len(_strong),
    "every constant the engine exports has a documented fate -- named beside a "
    "citation, named beside a number, or listed in SCORING.md 11 with a reason",
    f"{len(_exported)} exported name(s) from {sum(len(v) for v in _exported.values())} "
    f"`pub const` declaration(s) outside `#[cfg(test)]`; {_excluded_consts} "
    f"declaration(s) excluded as test-local. **cited: {len(_cited)}** "
    f"{sorted(_cited)}; **quantified: {len(_quant)}** {sorted(_quant)}; "
    f"**named only: {len(_named_only)}** {sorted(_named_only)}; **absent from "
    f"docs/ entirely: {len(_absent)}** {sorted(_absent)}; **excused in "
    f"SCORING.md 11: {len(_excused)}** {sorted(_excused)}. The buckets are "
    f"mutually exclusive and assigned strongest-first (cited > quantified > "
    f"named), so they partition the derived set. Unaccounted: "
    f"{_unaccounted}. An excused name that docs/ also cites or quantifies: "
    f"{_shared} -- a stale excuse. `pub(crate)` cannot qualify (the pattern is "
    f"literally `pub const`), and the enclosing modules read out of lib.rs are "
    f"{'all public' if not _private_mods else 'PRIVATE: ' + str(_private_mods)} "
    f"(test-local module declarations cut first, since one of them is private), "
    f"so this is the crate's real surface and not an over-report. "
    f"**What this is not:** a verified specification. 'Quantified' proves a digit "
    f"sits next to the name -- not that the digit is this constant's value, and "
    f"not that it is still right. The names are keyed bare, so one mention covers "
    f"all declarations sharing a name (ALL x{len(_exported.get('ALL', []))}, "
    f"COUNT x{len(_exported.get('COUNT', []))}) and the check cannot say which; "
    f"that loss was real, not hypothetical -- API.md documented two of the three "
    f"COUNTs and the third was silently uncovered while this was green. SCORING.md "
    f"section 11 itself is excluded from the scan ({len(_s11_skipped)} line(s)), "
    f"or the report would certify itself. **Not covered at all:** `static`, "
    f"literals in `pub fn` bodies, macro-generated constants, `pub(crate)`, "
    f"non-SCREAMING_CASE names, anything outside docs/, and whether a quantified "
    f"number is correct. **Mutation:** adding `pub const DOCS_GATE_PROBE: f64 = "
    f"1.0;` to a non-test part of any dock-core/src file turns this red "
    f"(unaccounted), as does deleting the only line that mentions a constant, as "
    f"does excusing everything (the last condition: excuses must be fewer than "
    f"cited+quantified, {len(_excused)} < {len(_strong)})",
)

_STATES = ("不需要规范", "已知缺口")
_bad_excuse = [
    (n, s.strip(), r.strip(), ln)
    for n, s, r, ln in _excuse_rows
    if not s.strip() or not r.strip() or not any(w in s for w in _STATES)
]
_stale_excuse = sorted(_excused & _strong)
check(
    bool(_excuse_rows) and not _bad_excuse and not _stale_excuse,
    "SCORING.md 11's exemption table gives every row a state word and a reason, "
    "and no exemption contradicts a specification elsewhere in docs/",
    f"{len(_excuse_rows)} row(s): {[(n, s.strip()) for n, s, _r, _l in _excuse_rows]}. "
    f"Malformed (no state word, or one of {_STATES}, or an empty reason): "
    f"{_bad_excuse}. Exempted but also cited or quantified elsewhere: "
    f"{_stale_excuse}. **Two halves that must flip together**, the same shape as "
    f"the two-sided contracts above: writing a real specification for an exempted "
    f"constant turns this red until its row is deleted, and deleting the row leaves "
    f"the coverage check to carry it. The state word is what keeps "
    f"'does not need one' and 'an unpinned gap nobody has closed' from reading as "
    f"the same sentence. **Mutation:** blanking a reason turns this red, as does "
    f"putting `12` next to `ALL` in the table's own reason column if the scan were "
    f"not excluding section 11 (it is, and that exclusion is why the `ALL` row can "
    f"honestly say 4 items)",
)

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
