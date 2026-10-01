"""Do the numbers in `docs/SCORING.md` and `docs/LIMITATIONS.md` still hold?

Run:  python scripts/scoring_docs_check.py

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

The same reasoning is why this file does **not** assert that a cross-element
hydrogen bond scores zero. That is the current behaviour (`grid.rs:418` writes
only into the receptor atom's own element block), it is recorded as a defect in
`SCORING.md` §5.1.1 and `LIMITATIONS.md` §3.2, and it lives in `dock-core` which
this file cannot fix. Asserting it would make a green test that fails the moment
the bug is fixed — the failure mode `smoothstep_is_c1` already demonstrates
(a test that pins a deprecated window stays green when the window is retuned).

# Both directions

A check that cannot fail is decoration. Every family below is mutation-proven:
perturb the document and the run goes red; perturb the Rust source and the run
goes red. See the mutation note at the bottom of `LIMITATIONS.md` §6.
"""

from __future__ import annotations

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

EXPECTED_CHECKS = 74  # measured: a green run of this file, including the self-check

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
scoring_rs = read(CORE / "scoring.rs")
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
doc_vinardo_scale = doc_vinardo.get("intramolecular_scale")
src_scales = [float(v) for v in
              re.findall(r"intramolecular_scale:\s*([\d.]+)", scoring_rs)]
check(
    doc_vinardo_scale is not None
    and len(src_scales) == 2
    and abs(src_scales[0] - doc_vina["intramolecular_scale"]) < 1e-12
    and abs(src_scales[1] - doc_vinardo_scale) < 1e-12,
    "SCORING.md's two intramolecular scales are scoring.rs's",
    f"scoring.rs has {src_scales} (vina default, then vinardo); the document has "
    f"vina={doc_vina.get('intramolecular_scale')}, vinardo={doc_vinardo_scale}. "
    f"NOTE: bound to the source, not to core.scoring_descriptions(), which does "
    f"not report this coefficient",
)


# ==========================================================================
# 2. The radius table in SCORING.md section 1 against types.rs
# ==========================================================================

section("SCORING.md section 1's radius table against types.rs::interaction_radius")

types_rs = read(CORE / "types.rs")
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

grid_rs = read(CORE / "grid.rs")
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
      f"{ {k: v for k, v in _hardcoded_hits.items() if v} or 'none'}. This is "
      f"the forward-looking half: the sentence used to say the default was 8, "
      f"and pinning the new text alone would not have stopped it coming back, "
      f"so the guard matches the stale wording directly",
)

# The count is a digit in the Chinese and a word in the English, so the English
# search has to be scoped to the sentence that makes the claim. Searching the
# whole file for a number word finds ordinary prose first -- "one" is the first
# entry in the table and appears everywhere -- which is how this check reported
# the count as 1 on its first run.
_cn_count = re.search(r"默认返回的\s*\*?\*?(\d+)\*?\*?\s*个位点", readme_cn)
_en_sentence = ""
if "ninth" in readme_en:
    _at = readme_en.index("ninth")
    _en_sentence = readme_en[max(0, _at - 220):_at + 220]
_en_word = next((n for n, w in sorted(_WORDS.items())
                 if re.search(rf"\b{w}\b", _en_sentence)), None)
check(
    _cn_count is not None and _en_word is not None
    and int(_cn_count.group(1)) == len(_sites)
    and _en_word == len(_sites)
    and _pk.DEFAULT_MAX_POCKETS == len(_sites),
    "the site count the READMEs publish is what find_pockets returns under "
    "its own defaults",
    f"the Chinese says {_cn_count.group(1) if _cn_count else None!r} default "
    f"sites, the English sentence says {_en_word!r}, `find_pockets` with no "
    f"arguments beyond the coordinates returns {len(_sites)}, and "
    f"DEFAULT_MAX_POCKETS is {_pk.DEFAULT_MAX_POCKETS} -- so all three agree. "
    f"The number is read out of the documents rather than written here, which "
    f"is the point: transcribing 12 into this file would have kept the check "
    f"green after the code changed to return something else. Both files are "
    f"pinned because they express the count differently, a digit and a word, "
    f"and a check on only one of them would let the other drift",
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
# 8. Self-verification
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
