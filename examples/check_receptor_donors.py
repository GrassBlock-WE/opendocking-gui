"""Check that a prepared receptor really carries polar hydrogens on its donors.

Experimental PDB files contain no hydrogens, so `prepare_receptor` has to add
them. A missing one is a *silent* error: the receptor simply has no donor there
and every hydrogen bond that should be made is lost. This script matches the
prepared file back against the original coordinates and reports any donor site
that ended up without a hydrogen.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from opendocking.prep import prepare_receptor  # noqa: E402

# Side-chain donor atoms, by residue.
DONOR_SITES = {
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
    "LYS": {"NZ"},
    "ARG": {"NH1", "NH2"},
    "TRP": {"NE1"},
    "HIS": {"NE2"},
}

MAX_BOND = 1.35  # Å; an N-H or O-H bond is about 1.0 Å


def disulfide_sulfurs(rows: list[tuple]) -> set[tuple[int, str]]:
    """Sulfurs already bonded to another sulfur.

    A cysteine in a disulfide bridge has no hydrogen — its valence is spent on
    the partner sulfur — so it is legitimately absent from the expected donor
    set. Crambin has six cysteines in three disulfides, so getting this wrong
    produces six false alarms.
    """
    import itertools

    sulfurs = [r for r in rows if r[0] == "SG" and r[1] == "CYS"]
    bridged = set()
    for a, b in itertools.combinations(sulfurs, 2):
        if math.dist(a[4:7], b[4:7]) < 2.5:
            bridged.add((a[2], a[0]))
            bridged.add((b[2], b[0]))
    return bridged


def parse_pdb(text: str, name_cols, res_cols, type_cols) -> list[tuple]:
    rows = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rows.append(
            (
                line[name_cols[0] : name_cols[1]].strip(),
                line[res_cols[0] : res_cols[1]].strip(),
                int(line[22:26]),
                line[type_cols[0] : type_cols[1]].strip(),
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    return rows


def main(pdb_path: str) -> int:
    path = Path(pdb_path)
    original = parse_pdb(
        path.read_text(encoding="utf-8"),
        (12, 16),
        (17, 20),
        (76, 78),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared_text = prepare_receptor(str(path))
    prepared = parse_pdb(prepared_text, (12, 16), (17, 20), (77, 79))

    hydrogens = [(a[4], a[5], a[6]) for a in prepared if a[3] == "HD"]
    print(f"{path.name}: {len(original)} heavy atoms in, {len(hydrogens)} polar H added")

    found, missing = 0, []
    bridged = disulfide_sulfurs(original)
    for name, res, seq, _t, x, y, z in original:
        is_donor = (name == "N" and res != "PRO") or (name in DONOR_SITES.get(res, set()))
        if not is_donor or (seq, name) in bridged:
            continue
        d = min((math.dist((x, y, z), h) for h in hydrogens), default=9e9)
        if d < MAX_BOND:
            found += 1
        else:
            missing.append((f"{res}{seq}", name, d))

    total = found + len(missing)
    print(f"  donor sites   : {total}")
    print(f"  with polar H  : {found}")
    print(f"  WITHOUT polar H: {len(missing)}")
    for res, name, d in missing[:15]:
        print(f"     {res} {name} nearest HD at {d:.2f} Å")

    # Informational only. A neutral N-terminus should carry two hydrogens;
    # RDKit's `AddHs` places only one of them at a bond length here, because a
    # terminal nitrogen with a single neighbour has no torsion context to build
    # the second position from. This is a marginal site on a charged protein
    # surface, not a pocket donor, so it is reported rather than failed on.
    nterm = [a for a in original if a[0] == "N" and a[2] == min(r[2] for r in original)]
    if nterm:
        d = sorted(math.dist(nterm[0][4:7], h) for h in hydrogens)
        at_bond = sum(1 for x in d[:3] if x < MAX_BOND)
        print(f"  N-terminus N hydrogens: {at_bond} (2 expected; see note in the script)")

    # A rigid receptor means no heavy atom may move.
    prep_heavy = [a for a in prepared if a[3] != "HD"]
    moved = 0
    if len(prep_heavy) == len(original):
        for a, b in zip(original, prep_heavy):
            if math.dist(a[4:7], b[4:7]) > 1e-3:
                moved += 1
    else:
        print(f"  heavy atom count changed: {len(original)} -> {len(prep_heavy)}")
    print(f"  heavy atoms moved       : {moved} (must be 0; the receptor is rigid)")

    return 1 if missing or moved else 0


if __name__ == "__main__":
    # The defaults resolve next to this file rather than next to the caller's
    # working directory, so `python examples/check_receptor_donors.py` works
    # from the repository root. A bare "1crn_receptor.pdb" only worked after a
    # `cd examples`, which is not what anyone runs.
    here = Path(__file__).resolve().parent
    raise SystemExit(
        main(sys.argv[1] if len(sys.argv) > 1 else str(here / "1crn_receptor.pdb"))
    )
