"""Check that the viewer bonds the right atoms to each other.

Headless on purpose. The bond logic is the part of a molecular viewer that can
be *silently* wrong -- a wrong bond still draws a line, still looks like a
molecule, and nothing crashes -- so it is verified without Qt or a display, and
the assertions are about chemistry rather than about pixels.

Run:  python scripts/structure_bond_check.py
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dock-py" / "python"))

import numpy as np  # noqa: E402

from opendocking.workbench import structure as st_mod  # noqa: E402

EXAMPLES = ROOT / "examples"
FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def dist(a, b) -> float:
    return math.dist(a, b)


def prepare_receptor_text(pdb: Path) -> str:
    """Prepare a receptor with the *current* source, not a stored fixture.

    The committed `*_prep.pdbqt` files were written before residue identity was
    preserved; preparing here means this check always tests the code as it is.
    """
    from opendocking.prep import prepare_receptor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return prepare_receptor(pdb)


# ---------------------------------------------------------------------------
# Protein
# ---------------------------------------------------------------------------


def check_protein() -> None:
    section("protein: bonds come from residue templates")
    text = prepare_receptor_text(EXAMPLES / "1crn_receptor.pdb")
    st = st_mod.parse_structure(text, "1crn")
    check("parsed", len(st.atoms) > 0, f"{len(st.atoms)} atoms")
    check("recognised as protein", st.is_protein())

    aas = [r for r in st.residues if r.is_amino_acid]
    check(
        "every residue matched a template",
        len(aas) == len([r for r in st.residues if r.name not in st_mod.ION_RESIDUES]),
        f"{len(aas)} amino-acid residues of {len(st.residues)} total",
    )
    check("no unknown-residue warning",
          not any("no connectivity template" in w for w in st.warnings),
          "; ".join(w for w in st.warnings if "template" in w))

    # Residue numbering must be contiguous, because that is what licenses the
    # peptide bonds.
    by_chain: dict[str, list[int]] = {}
    for r in aas:
        by_chain.setdefault(r.chain, []).append(r.resid)
    for chain, ids in by_chain.items():
        ids = sorted(ids)
        gaps = [
            (ids[i], ids[i + 1]) for i in range(len(ids) - 1) if ids[i + 1] - ids[i] != 1
        ]
        check(
            f"chain {chain.strip() or '_'} residue numbering is contiguous",
            not gaps,
            f"{len(ids)} residues, {len(gaps)} gap(s){': ' + str(gaps[:4]) if gaps else ''}",
        )
        check(
            f"chain {chain.strip() or '_'} has no duplicate residue numbers",
            len(set(ids)) == len(ids),
        )

    # Backbone completeness.
    missing = []
    for r in aas:
        names = {st.atoms[i].name for i in r.atoms}
        for key in ("N", "CA", "C"):
            if key not in names:
                missing.append(f"{r.name}{r.resid}.{key}")
    check("every residue has N, CA and C", not missing, ", ".join(missing[:6]))

    trace = st.backbone()
    check("backbone trace covers every residue", len(trace) == len(aas),
          f"{len(trace)} residues traced")

    # --- the important one: no bond may cross a residue except the peptide C-N
    crosses: list[str] = []
    peptide = 0
    for i, j in st.bond_pairs():
        a, b = st.atoms[i], st.atoms[j]
        if a.residue_key == b.residue_key:
            continue
        if {a.name, b.name} == {"C", "N"} and b.resid - a.resid == 1 and a.chain == b.chain:
            peptide += 1
            continue
        crosses.append(f"{a.name}{a.resname}{a.resid}-{b.name}{b.resname}{b.resid}")
    check(
        "no bond crosses a residue except the peptide C-N",
        not crosses,
        f"{peptide} peptide bonds; offenders: {'; '.join(crosses[:6])}",
    )
    expected_peptide = sum(
        sum(1 for a, b in zip(ids, ids[1:]) if b - a == 1)
        for ids in (sorted(r.resid for r in aas if r.chain == c) for c in by_chain)
    )
    check(
        "one peptide bond per consecutive residue pair",
        peptide == expected_peptide,
        f"{peptide} drawn, {expected_peptide} expected",
    )

    # --- side chains match their templates exactly
    sidechain_errors: list[str] = []
    for r in aas:
        canon = st_mod.RESIDUE_ALIASES.get(r.name, r.name)
        template = set(st_mod.SIDECHAIN_BONDS.get(canon, ()))
        by_name: dict[str, int] = {}
        for i in r.atoms:
            by_name.setdefault(st.atoms[i].name, i)
        present = {
            (p, c)
            for p, c in template
            if p in by_name
            and c in by_name
            and by_name[c] in st.atoms[by_name[p]].bonds
        }
        expected = {
            (p, c) for p, c in template if p in by_name and c in by_name
        }
        if present != expected:
            missing_bonds = sorted(expected - present)
            sidechain_errors.append(f"{r.name}{r.resid} missing {missing_bonds}")
    check(
        "every side-chain bond of every residue is drawn",
        not sidechain_errors,
        "; ".join(sidechain_errors[:5]),
    )

    proline = next((r for r in aas if r.name == "PRO"), None)
    if proline is not None:
        by_name = {st.atoms[i].name: i for i in proline.atoms}
        has_ring = "CD" in by_name and "N" in by_name and by_name["N"] in st.atoms[by_name["CD"]].bonds
        check("proline's side chain closes back onto its backbone N", has_ring,
              f"PRO{proline.resid}")

    # --- geometry sanity
    lengths = [dist(st.atoms[i].xyz, st.atoms[j].xyz) for i, j in st.bond_pairs()]
    if lengths:
        check(
            "every bond length is physically possible",
            min(lengths) > 0.85 and max(lengths) < 2.0,
            f"min {min(lengths):.2f} A, max {max(lengths):.2f} A, {len(lengths)} bonds",
        )
    over = [
        f"{st.atoms[i].element}{st.atoms[i].name}{st.atoms[i].resid} has {len(st.atoms[i].bonds)} bonds"
        for i in range(len(st.atoms))
        if st_mod.MAX_VALENCE.get(st.atoms[i].element, 99) < len(st.atoms[i].bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))

    ca_d = [
        dist(st.atoms[trace[i][1]].xyz, st.atoms[trace[i + 1][1]].xyz)
        for i in range(len(trace) - 1)
    ]
    if ca_d:
        check(
            "consecutive CA atoms are a residue apart",
            min(ca_d) > 2.8 and max(ca_d) < 4.6,
            f"min {min(ca_d):.2f} A, max {max(ca_d):.2f} A",
        )

    ss = st_mod.secondary_structure(trace, st.atoms)
    check(
        "secondary structure is estimated for the interior residues",
        len(ss) == len(trace) and set(ss) <= {"helix", "sheet", "coil"},
        f"{ss.count('helix')} helix, {ss.count('sheet')} sheet, {ss.count('coil')} coil",
    )

    # --- and the reason templates are used at all
    naive = st_mod.Structure(name=st.name, atoms=st.atoms, residues=st.residues)
    st_mod._apply_distance_bonds(naive)
    naive_cross = sum(
        1
        for i, j in naive.bond_pairs()
        if st.atoms[i].residue_key != st.atoms[j].residue_key
    )
    check(
        "a pure distance rule really would have made cross-residue bonds here",
        naive_cross > 0,
        f"{naive_cross} cross-residue bonds from distance alone, "
        f"against {len(crosses)} from templates",
    )

    check(
        "auditor raised no bond-length complaints",
        not any("too short" in w or "longer than" in w for w in st.warnings),
        "; ".join(w for w in st.warnings if "Å" in w)[:160],
    )


# ---------------------------------------------------------------------------
# Small molecule
# ---------------------------------------------------------------------------


def check_small_molecule() -> None:
    section("small molecule: flat file, bonds inferred and audited")
    st = st_mod.parse_structure(
        (EXAMPLES / "ibuprofen_prep.pdbqt").read_text(encoding="utf-8"), "ibuprofen"
    )
    check("parsed", len(st.atoms) == 16, f"{len(st.atoms)} atoms")
    check("not treated as protein", not st.is_protein())

    # RDKit knows this molecule's real connectivity, so it is the oracle. A
    # hand-written expectation would only encode what I already believe, and
    # the first version of this check did exactly that and was simply wrong
    # about which carbons are in the ring.
    truth = _rdkit_bond_count(EXAMPLES / "biotin_prep.pdbqt")
    st_b = st_mod.parse_structure(
        (EXAMPLES / "biotin_prep.pdbqt").read_text(encoding="utf-8"), "biotin"
    )
    check(
        "perceived bonds match RDKit's own perception",
        len(st_b.bond_pairs()) == truth,
        f"{len(st_b.bond_pairs())} perceived, {truth} from RDKit, "
        f"{len(st_b.atoms)} atoms",
    )

    deg = [len(a.bonds) for a in st.atoms]
    check(
        "every atom has at least one bond",
        all(d >= 1 for d in deg),
        "degrees " + ", ".join(str(d) for d in deg),
    )
    over = [
        f"{a.element}{i} has {len(a.bonds)} bonds"
        for i, a in enumerate(st.atoms)
        if st_mod.MAX_VALENCE.get(a.element, 99) < len(a.bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))

    # A connected graph on n atoms has at least n-1 bonds; more means cycles,
    # which for this molecule means the aromatic ring was closed.
    rings = len(st.bond_pairs()) - (len(st.atoms) - 1)
    check("the graph is connected with one independent ring", rings == 1,
          f"{len(st.bond_pairs())} bonds over {len(st.atoms)} atoms -> {rings} ring(s)")

    check(
        "the carboxyl carbon is bonded to both oxygens",
        _carboxyl_ok(st),
    )

    lengths = [dist(st.atoms[i].xyz, st.atoms[j].xyz) for i, j in st.bond_pairs()]
    check(
        "every bond length is physically possible",
        min(lengths) > 0.85 and max(lengths) < 2.0,
        f"min {min(lengths):.2f} A, max {max(lengths):.2f} A, {len(lengths)} bonds",
    )
    check(
        "connectivity was reported as inferred, not declared",
        any("inferred from interatomic distances" in w for w in st.warnings),
    )


def _rdkit_bond_count(prepared: Path) -> int:
    """How many bonds RDKit perceives in the *prepared* file itself.

    The oracle has to be run on the same file the check reads. Comparing bonds
    perceived from `biotin_prep.pdbqt` against bonds RDKit sees in
    `biotin.sdf` compares two different atom sets -- the prepared file has its
    non-polar hydrogens merged into their carbons -- so the counts differ for a
    reason that has nothing to do with either perception being wrong. RDKit
    reading the prepared file applies its own distance rules to the same atoms,
    which is the comparison worth making.
    """
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(str(prepared), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not read {prepared}")
    return mol.GetNumBonds()


def _carboxyl_ok(st) -> bool:
    """A carboxyl carbon has exactly two oxygens, and they are not bonded to
    each other -- the classic way a distance rule over-bonds a carboxyl group."""
    for a in st.atoms:
        if a.element != "C":
            continue
        oxygens = [j for j in a.bonds if st.atoms[j].element == "O"]
        if len(oxygens) == 2:
            return oxygens[1] not in st.atoms[oxygens[0]].bonds
    return False


# ---------------------------------------------------------------------------
# Docked pose: declared connectivity
# ---------------------------------------------------------------------------


def check_pose() -> None:
    section("docked pose: bonds come from ROOT/BRANCH, not geometry")
    models = st_mod.parse_structure(
        (EXAMPLES / "poses.pdbqt").read_text(encoding="utf-8"), "poses"
    )
    check("parsed", len(models.atoms) > 0, f"{len(models.atoms)} atoms in the last model")
    check("file declared its own connectivity", models.declared)

    lig = st_mod.parse_structure(
        (EXAMPLES / "ibuprofen_prep.pdbqt").read_text(encoding="utf-8"), "ibuprofen"
    )
    check(
        "a pose keeps the ligand's atom count and elements",
        len(models.atoms) == len(lig.atoms)
        and [a.element for a in models.atoms] == [a.element for a in lig.atoms],
        f"{len(models.atoms)} vs {len(lig.atoms)} atoms",
    )

    lengths = [dist(models.atoms[i].xyz, models.atoms[j].xyz) for i, j in models.bond_pairs()]
    check(
        "every declared bond is physically possible",
        bool(lengths) and min(lengths) > 0.85 and max(lengths) < 2.0,
        f"min {min(lengths):.2f} A, max {max(lengths):.2f} A, {len(lengths)} bonds",
    )
    over = [
        f"{a.element}{i} has {len(a.bonds)} bonds"
        for i, a in enumerate(models.atoms)
        if st_mod.MAX_VALENCE.get(a.element, 99) < len(a.bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))


# ---------------------------------------------------------------------------
# Non-protein receptor blob
# ---------------------------------------------------------------------------


def check_synthetic_receptor() -> None:
    section("synthetic receptor: no template, no invented bonds")
    st = st_mod.parse_structure(
        (EXAMPLES / "rec_prep.pdbqt").read_text(encoding="utf-8"), "rec"
    )
    check("parsed", len(st.atoms) == 30, f"{len(st.atoms)} atoms")
    check("not mistaken for a protein", not st.is_protein())
    check(
        "its bonds were inferred and that is reported",
        any("inferred from interatomic distances" in w for w in st.warnings),
    )
    over = [
        f"{a.element}{i} has {len(a.bonds)} bonds"
        for i, a in enumerate(st.atoms)
        if st_mod.MAX_VALENCE.get(a.element, 99) < len(a.bonds)
    ]
    check("no atom exceeds its maximum valency", not over, "; ".join(over[:5]))


def main() -> int:
    print(f"verifying bond perception against {EXAMPLES}")
    check_protein()
    check_small_molecule()
    check_pose()
    check_synthetic_receptor()

    print("\n=== summary ===")
    print(f"  {CHECKS - len(FAILURES)} passed, {len(FAILURES)} failed, {CHECKS} checks")
    for f in FAILURES:
        print(f"    FAIL {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
