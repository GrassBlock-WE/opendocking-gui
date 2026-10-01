"""Reference answers for the secondary-structure checks, and one built peptide.

Two things live here, both of them *inputs to a check* rather than behaviour:

* :data:`CRAMBIN_1CRN_SS` -- the published secondary structure of 1CRN, the
  receptor this repository ships. Written down so a check can state the answer
  it expects and fail with the number it got, instead of comparing the code
  against itself.
* :func:`ideal_alpha_helix` -- a synthetic peptide built from ideal alpha-helix
  internal coordinates, so there is a case with **no** reference structure
  behind it at all: every residue is helical by construction, and if the
  classifier does not say so then the classifier is wrong, not the fixture.

The construction is validated by :func:`ideal_helix_geometry`, which measures
the distances that define an ideal alpha helix and is itself checked. A fixture
whose geometry has silently drifted would otherwise make the classification
check meaningless: an all-coil answer would be blamed on the classifier when
the peptide had stopped being a helix.
"""

from __future__ import annotations

import math

__all__ = [
    "CRAMBIN_1CRN_SS",
    "ideal_alpha_helix",
    "ideal_helix_geometry",
    "ca_distances",
    "segments",
]

#: Published secondary structure of crambin, PDB entry 1CRN, as
#: ``(published_lo, published_hi, state, what it is, reported_lo, reported_hi)``.
#:
#: 1CRN is a 46-residue plant protein with a short two-stranded antiparallel
#: beta sheet running through it and two alpha helices; the published range of
#: each segment is the first two numbers and is the annotation this repository's
#: answers are measured against.
#:
#: The last two numbers are the range this module is expected to report, and
#: they differ from the published one for two segments. That difference is
#: **not** a threshold somebody moved to make a check green, and it is written
#: down here rather than absorbed into the published range:
#:
#: * **32-35 -> 33-35.** Residue 32 is called coil. The reason is a documented
#:   simplification in `structure._sheet_segments`: it needs a *bridge* to put
#:   a residue in a strand, and a residue is only in a strand if it is bridged
#:   or interior to a bridged stretch. 1CRN has exactly two inter-strand
#:   hydrogen bonds, 35->1 and 33->3, so 32 is neither. Adding 32 would mean
#:   extending a strand residue by residue past its own evidence, which is the
#:   failure mode that same comment rules out.
#: * **7-19 -> 7-20** and **23-30 -> 23-31.** One residue too many at the
#:   C-terminal end of each alpha helix. A turn is the *second and following*
#:   residue of an i->i+4 bond, so the 4-turn 20->16 puts 17-20 in the segment.
#:   Whether the last residue of a helix keeps a clean i->i+4 bond depends on
#:   the last turn's angle, and at a helix end that bond is the one most likely
#:   to miss the 3.5 A N...O gate.
#:
#: A check asserts the reported range, and a second one asserts the exact set of
#: residues where the two differ, so neither can drift without turning red.
CRAMBIN_1CRN_SS: tuple[tuple[int, int, str, str, int, int], ...] = (
    (1, 4, "sheet", "beta strand 1 of the two-stranded antiparallel sheet", 1, 4),
    (7, 19, "helix", "alpha helix 1", 7, 20),
    (23, 30, "helix", "alpha helix 2", 23, 31),
    (32, 35, "sheet", "beta strand 2 of the two-stranded antiparallel sheet", 33, 35),
    (42, 44, "helix", "C-terminal 3-10 helix (a 3-residue turn, not an alpha helix)", 42, 44),
)

#: Residues of 1CRN that the published annotation calls loop and that are
#: inside none of the segments above. Checked as coil, because a classifier that
#: called the whole protein helix would pass every per-segment check above: the
#: segments are a small fraction of a 46-residue chain, and "everything is
#: helix" agrees with all of them.
CRAMBIN_1CRN_LOOPS: tuple[int, ...] = (5, 6, 21, 22, 32, 36, 37, 38, 39, 40, 41)

# ---------------------------------------------------------------------------
# Ideal alpha helix, from internal coordinates
# ---------------------------------------------------------------------------

#: Ideal alpha-helix internal coordinates, in the usual textbook values.
#: Distances in angstrom, angles and dihedrals in degrees.
_PHI, _PSI = -57.0, -47.0
_C_N = 1.329
_N_CA = 1.458
_CA_C = 1.525
_C_O = 1.231
_ANGLE_CA_C_N = 116.2
_ANGLE_C_N_CA = 121.7
_ANGLE_N_CA_C = 111.2
_ANGLE_CA_C_O = 120.8
_OMEGA = 180.0


def _place(a, b, c, bond: float, angle: float, dihedral: float):
    """NeRF: the point `bond` away from `b`, at `angle` b-c, `dihedral` a-b-c-d.

    The standard natural-extension-reference-frame construction: the new atom is
    built in a local frame on `b` and `c` and then rotated about the `b`-`c` axis
    by the torsion. Angles are measured the way the text writes them, from the
    *incoming* direction `a->b`, so the sign conventions here match the ones
    `structure.dihedral` uses.
    """
    ang = math.radians(angle)
    tor = math.radians(dihedral)

    def sub(p, q):
        return [p[k] - q[k] for k in range(3)]

    def cross(u, v):
        return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]

    def norm(v):
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    bc = norm(sub(c, b))
    n_vec = norm(cross(sub(b, a), bc))
    m_vec = cross(n_vec, bc)
    d2 = [
        -bond * math.cos(ang),
        bond * math.sin(ang) * math.cos(tor),
        bond * math.sin(ang) * math.sin(tor),
    ]
    return [c[k] + bc[k] * d2[0] + m_vec[k] * d2[1] + n_vec[k] * d2[2] for k in range(3)]


def ideal_alpha_helix(n: int = 16) -> str:
    """A right-handed ideal alpha helix of `n` residues, as PDBQT text.

    Built in the order N, CA, C, O per residue from the internal coordinates
    above, so the backbone is a real peptide with real carbonyls and the
    hydrogen-bond pass has everything it needs. No side chains: a side chain
    would add atoms the classifier does not read, and their absence is what
    makes this a test of the backbone rule rather than of the parser.

    The chain starts at residue 1 and is written with a terminal ``OXT`` on the
    last residue, so it parses as an ordinary single-chain protein.
    """
    if n < 2:
        raise ValueError("an ideal helix needs at least two residues to have a turn")

    coords: list[tuple[str, int, tuple[float, float, float]]] = []
    n_i = (0.0, 0.0, 0.0)
    ca_i = (1.458, 0.0, 0.0)
    # Residue 1 is seeded rather than derived, then every later residue is
    # placed from the one before it. The seed needs three *distinct* points:
    # NeRF takes its reference plane from `a`, `b`, `c`, so seeding `a` on top
    # of `b` makes the plane undefined and every later distance comes out wrong
    # -- a helix whose CA-CA is 2.17 A, which the geometry checks below catch.
    seed = (0.0, 0.0, 0.0)
    n_i = (0.0, 0.0, 1.0)
    ca_i = (_N_CA, 0.0, 1.0)
    c_i = _place(seed, n_i, ca_i, _CA_C, _ANGLE_N_CA_C, 0.0)
    o_i = _place(n_i, ca_i, c_i, _C_O, _ANGLE_CA_C_O, _PSI)

    for i in range(n):
        res = i + 1
        last = i == n - 1
        coords.append(("N", res, n_i))
        coords.append(("CA", res, ca_i))
        coords.append(("C", res, c_i))
        if last:
            coords.append(("OXT", res, o_i))
            continue
        coords.append(("O", res, o_i))
        # Each atom is placed from (a, b, c) with the *new* atom as `d`, so the
        # angle is always the one at the third of the three, and the torsion is
        # the named one the four atoms form:
        #   N(i+1)  -- angle at C(i),  torsion is psi(i)
        #   CA(i+1) -- angle at N(i+1), torsion is omega
        #   C(i+1)  -- angle at CA(i+1), torsion is phi(i+1)
        #   O(i+1)  -- angle at C(i+1), torsion is psi(i+1)
        # The psi/omega mix-up is not subtle and is not caught by the angles:
        # with `_OMEGA` in the first line the chain still has phi exactly -57,
        # and psi comes out 180, which straightens the chain until CA-CA is
        # 3.8 A. Only the CA separations say so, which is why
        # `ideal_helix_geometry` exists and is checked.
        n_next = _place(n_i, ca_i, c_i, _C_N, _ANGLE_CA_C_N, _PSI)
        ca_next = _place(ca_i, c_i, n_next, _N_CA, _ANGLE_C_N_CA, _OMEGA)
        c_next = _place(c_i, n_next, ca_next, _CA_C, _ANGLE_N_CA_C, _PHI)
        o_next = _place(n_next, ca_next, c_next, _C_O, _ANGLE_CA_C_O, _PSI)
        n_i, ca_i, c_i, o_i = n_next, ca_next, c_next, o_next

    return _as_pdbqt(coords)


def _as_pdbqt(records) -> str:
    """Fixed-column ATOM records, so the product's own parser reads them.

    Column positions are the PDB ones, and they are exact rather than
    approximately right: `structure.parse_structure` slices the line at fixed
    offsets, so a residue name written one column early makes the whole file
    read as a structure with no amino acid in it, and the classifier is then
    handed an empty backbone. That is a fixture that fails for a reason that has
    nothing to do with the classifier, which is the worst kind.
    """
    lines = ["REMARK  synthetic ideal alpha helix, built by scripts/secondary_reference.py"]
    serial = 0
    for name, resid, xyz in records:
        serial += 1
        element = name[0]
        # 1-6 record, 7-11 serial, 12 blank, 13-16 name, 17 altLoc,
        # 18-20 resName, 21 blank, 22 chain, 23-26 resSeq, 27 iCode, 28-30
        # blank, 31-54 xyz, 55-66 occupancy/temp, 67-76 blank, 77-78 element.
        lines.append(
            "ATOM  "
            + f"{serial:5d}"
            + " "
            + f"{name:>4s}"
            + " "
            + f"{'ALA':>3s}"
            + " "
            + "A"
            + f"{resid:4d}"
            + "    "
            + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            + "  1.00  0.00"
            + " " * 10
            + f"{element:>2s}"
        )
    lines.append("TER")
    return "\n".join(lines) + "\n"


def ca_distances(text: str) -> dict:
    """The CA separations of a parsed peptide, as the ones that define a helix.

    Returns a dict with the keys ``ca_ca``, ``i3`` (CA(i)-CA(i+3)) and
    ``i4`` (CA(i)-CA(i+4)), each a list over `i` in the peptide. These are the
    measurements the fixture is validated against, so they are computed from the
    written file rather than from the values used to build it -- otherwise a
    formatting error in the writer would be invisible.
    """
    import numpy as np

    from opendocking.workbench.structure import parse_structure

    st = parse_structure(text, "ideal")
    trace = st.backbone()
    ca = [st.atoms[i].xyz for _, i, _ in trace]
    out = {"ca_ca": [], "i3": [], "i4": []}
    for a in range(len(ca)):
        if a + 1 < len(ca):
            out["ca_ca"].append(float(np.linalg.norm(np.subtract(ca[a + 1], ca[a]))))
        if a + 3 < len(ca):
            out["i3"].append(float(np.linalg.norm(np.subtract(ca[a + 3], ca[a]))))
        if a + 4 < len(ca):
            out["i4"].append(float(np.linalg.norm(np.subtract(ca[a + 4], ca[a]))))
    return out


def ideal_helix_geometry(text: str) -> dict:
    """The measurements an ideal alpha helix has to have, from the file itself.

    Measured on the *written* file rather than on the values used to build it,
    so a formatting error in the writer cannot hide:

    * CA-CA about 3.8 A. That is the C-alpha-C-alpha **virtual bond** of a
      helix, not a bond: 3.6 residues per turn at 2.3 A radius and 1.5 A rise
      puts consecutive CAs 3.8 A apart even though no atom is between them. A
      1.5 A figure here would be a chain of bonded carbons, i.e. a coil.
    * CA(i)-CA(i+3) about 5.0-5.4 A, and CA(i)-CA(i+4) about 6.2-6.4 A, the
      separations that say a residue has 3.6 neighbours per turn.
    * a rise of about 1.5 A per residue along the helix axis, measured from the
      principal axis rather than from a particular residue's z coordinate,
      because nothing here knows which way the build placed the helix.

    All four are consequences of one another: a construction that got psi
    wrong still has every bond length exactly right and a phi of exactly -57,
    and only the separations show the chain has been straightened. That is why
    this function exists and is itself checked.
    """
    import numpy as np

    from opendocking.workbench.structure import parse_structure

    st = parse_structure(text, "ideal")
    trace = st.backbone()
    ca = np.asarray([st.atoms[i].xyz for _, i, _ in trace], dtype=np.float64)
    d = ca_distances(text)
    centred = ca - ca.mean(axis=0)
    # Largest spread of principal components is the helix axis.
    _, sv, vh = np.linalg.svd(centred, full_matrices=False)
    along = centred @ vh[0]
    return {
        "n": len(trace),
        "ca_ca": float(np.mean(d["ca_ca"])),
        "i3": float(np.mean(d["i3"])),
        "i4": float(np.mean(d["i4"])),
        "rise": float((along[-1] - along[0]) / (len(ca) - 1)),
        "ca_spread": float(sv[0]),
    }


#: What :func:`ideal_helix_geometry` has to measure for the fixture to be an
#: ideal alpha helix at all, as ``key -> (low, high)``. Checked, because a
#: fixture that is not a helix makes every classification check meaningless.
IDEAL_HELIX_RANGES: dict[str, tuple[float, float]] = {
    "ca_ca": (3.6, 4.0),
    "i3": (4.8, 5.6),
    "i4": (6.0, 6.6),
    "rise": (1.3, 1.7),
}


def segments(states, residue_numbers) -> list[tuple[int, int, str]]:
    """Maximal runs of one state, as ``(first_resid, last_resid, state)``.

    The unit a check asserts on. A per-residue list is the wrong thing to
    compare against a published annotation, because an annotation is a list of
    *segments*: "residues 7 to 19 are one alpha helix" survives a terminus
    moving by one residue, and "residue 14 is helix" does not.
    """
    out: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append((int(residue_numbers[start]), int(residue_numbers[i - 1]), states[start]))
            start = i
    return out
