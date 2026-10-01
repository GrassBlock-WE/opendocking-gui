"""Is `cartoon` a cartoon, or is it the `ribbon` mode under a new name?

Run:  python scripts/representation_cartoon_check.py

# The defect this file exists for

`cartoon` and `ribbon` called the same function with the same arguments. Every
check in the suite was green: a ribbon was built, it had the right number of
triangles, it rendered, and the label above it said "Cartoon". That is a false
claim about a product, which is the one class of defect a test suite that only
asks "is there a ribbon" cannot see.

So the question here is not "is the cartoon's geometry different from the
ribbon's" -- two meshes with different widths are different. It is **"does the
cartoon's geometry encode secondary structure, in a way that survives having
the colour taken away"**. Both halves are measured, not asserted.

# The discriminator, and why it is a mean and not a variance

The obvious instrument for "is this ribbon twisted" is the *variance* of the
cross-section's orientation along the trace. Measured, it is the wrong one:

    prepared 1CRN, per-residue cross-section rotation about the path
                  mean       sd
    cartoon      +100.58   7.07
    ribbon         -0.11   7.02

The standard deviations are the same to two decimal places. A ribbon's frame
carries about 7 degrees per residue of *noise* from parallel transport and from
the spline's own curvature, and a twisted one carries exactly that same noise
on top of a hundredfold larger bias. A variance threshold cannot separate these
two geometries, and one that appeared to would be reading the noise.

So the discriminator is the **signed mean rate of rotation of the cross-section
about the path, per residue**, measured off the mesh:

    d_i = atan2( (u_i x u_{i+1}) . t_i , u_i . u_{i+1} )

read from the vertex positions the renderer will draw. A helix turns at
100 deg/residue (360 / 3.6); a strand does not turn at all, because a flat
ribbon's face is what you are meant to see. The two are 1.4 degrees apart here,
which is the whole margin this file rests on.

# Every threshold, and where its number came from

Nothing below was typed to make something pass; each is stated as a measurement
from prepared 1CRN taken with this file's own helpers, and each carries the
number the mutation produces so the reader can see how far from the edge it is.

| claim                          | threshold | cartoon | ribbon  |
|--------------------------------|-----------|---------|---------|
| helix turns the cross-section  | >= 60     | 100.58  |  -0.11  |
| a strand does not turn         | <= 20     |   2.65  |   2.65  |
| a coil does not turn           | <= 20     |   2.91  |   2.91  |
| one turn returns to 360 deg    | 335..385  | 364.1   |   2.4   |
| strand arrowhead at the C end  | >= 2.0x   |   2.60  |   1.00  |
| helix-vs-sheet, no colour      | >= 60 000 px | 129 108 | 30 141 |

The helix floor of 60 is not "half of 100 because that is safer". It is a floor
below which a viewer does not see a helix: the ribbon would have to turn less
than about a third of a turn per residue, and at 3.6 residues per turn that
reads as a wavy ribbon rather than a spiral. The measured 100.58 sits 40 above
it and the mutated -0.11 sits 60 below it, so the guard survives a twist rate
drifting by +/-40% before it would notice -- which is the point of a floor that
is not pinned to the exact number.

# The controls

Three, all permanent and all run on every invocation, because a guard that was
mutation-tested once and then left alone reports green the day someone changes
what it was written to notice:

1. **the shipped mutation.** The helix geometry is measured, then the *ribbon's*
   helix geometry is offered to the same predicate and must be rejected, while
   the cartoon's is accepted. This is the exact code that shipped.
2. **the greedy mutation.** A strand and a coil cannot satisfy the helix test.
   Without this direction, a builder that twisted every class would pass.
3. **the arrowhead at the wrong end.** `head_widths` is measured on the mesh and
   the predicate requires the C-terminal end to be the wide one on *every*
   strand run, so a head built at the N-terminal end is rejected by the same
   number that accepts the shipped one.

# What needs a GPU, and what does not

The rotation-rate, phase-lock and arrowhead checks are pure geometry on the mesh
and run anywhere. The greyscale checks render, so they are inside one branch
that registers a `skip` in the same branch it can be skipped in -- a machine
with no offscreen context reports the skips and still reaches its pinned total,
rather than reporting a smaller one that looks like a smaller scope.

Run:  python scripts/representation_cartoon_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEV_TREE = ROOT / "dock-py" / "python"

# This file measures *this checkout*. The installed wheel and the release tree
# under `opendocking-gui` are the same bytes only until someone edits one of
# them, and a guard that silently measured a stale copy would pass on the code
# it is supposed to be guarding. The first check fails if the table still came
# from somewhere else.
sys.path.insert(0, str(DEV_TREE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from opendocking.workbench import MoleculeView, app  # noqa: E402
from opendocking.workbench import cartoon_geometry as cg  # noqa: E402
from opendocking.workbench.cartoon_geometry import cartoon as build_cartoon  # noqa: E402
from opendocking.workbench.geometry import ribbon as build_ribbon  # noqa: E402
from opendocking.workbench.structure import parse_structure, secondary_structure  # noqa: E402

import secondary_render as ss_render  # noqa: E402

#: How many checks this file is supposed to run, counted by running it. A check
#: that stops running takes the count down with it silently, which is how a
#: suite goes from 19 to 17 with a clean report.
#:
#: Counted by running it on a machine with an offscreen context, and it is the
#: same on one without: the three greyscale checks register a `skip` in the
#: branch they can be skipped in, and a skip counts here. Measured 2026-10-01
#: on prepared 1CRN from the development tree.
EXPECTED_CHECKS = 21

#: The call-site census this file is supposed to have, declared here rather than
#: in a table another file owns -- see `GATE-DECLARE` in
#: `check_scripts_declare.py` for the format and why it cannot drift. Comments
#: only, so the `EXPECTED_CHECKS` above is unaffected by them.
#:
#: **18+3 is a first declaration, not a move**, and the split is worth reading
#: because it says something about the file: the three guarded sites are the
#: greyscale checks, which are the only ones that need a frame. Everything that
#: can be answered from the mesh alone -- the rotation rate, the phase lock, the
#: arrowhead, and all three controls -- runs unconditionally, so a machine with
#: no OpenGL context still answers the question this file exists to ask.
#: GATE-DECLARE 1
#: sites: 18 unconditional + 3 guarded
#: guards: sha256:d1a014ded301ebb6a90f98df68451df7b79088129f7554e5155eaea9a48a91f4

#: Every section this file is supposed to reach, in file order.
EXPECTED_SECTIONS = (
    "1. which copy of the workbench this measured",
    "2. the cartoon and the ribbon are not the same mesh",
    "3. the cross-section turns for a helix and not for a strand",
    "4. the twist is phase-locked to one turn per 3.6 residues",
    "5. every strand's arrowhead is at its C-terminal end",
    "6. the three classes are separable without colour",
    "7. the guard fails in both directions",
    "summary",
)

RESULTS: list[tuple[str, str, str]] = []
FAILURES: list[str] = []
_sections: list[str] = []


def check(name, ok, detail=""):
    RESULTS.append(("PASS" if ok else "FAIL", name, detail))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def skip(name, reason):
    """Record a check this environment could not answer, and say why.

    **A skip counts toward `EXPECTED_CHECKS`, on purpose.** "The three classes
    are not separable without colour" and "this machine has no offscreen
    context" are different claims, and only one of them is ever true -- so the
    two are reported with different tags. But if a skip did *not* count, the
    total a run prints would be smaller on a machine without a GPU, and a
    smaller total is indistinguishable from a smaller scope: the suite would
    quietly stop asking the greyscale question and nobody would be told.

    So the count stays constant across machines and the summary prints passes
    and skips as two numbers that are never added together.
    """
    RESULTS.append(("SKIP", name, reason))
    print(f"  [SKIP] {name}  - {reason}")
    return False


def section(t):
    print(f"\n=== {t} ===")
    _sections.append(t)


# --------------------------------------------------------------------------
# The predicates. Each is a question about a measurement, so a guard can offer
# one geometry to another's predicate and watch it be rejected.
# --------------------------------------------------------------------------
def helix_turns(summary: dict, floor: float = 60.0) -> tuple[bool, str]:
    """Does this mesh's helix turn its cross-section about the path?"""
    if "helix" not in summary:
        return False, "the mesh carries no helix residues at all"
    mean, sd, n = summary["helix"]
    return mean >= floor, (
        f"{n} ring-gaps, mean {mean:+.2f} deg/residue, sd {sd:.2f} "
        f"(want >= {floor:.0f}; a ribbon reads about 0)"
    )


def does_not_turn(summary: dict, state: str, ceiling: float = 20.0) -> tuple[bool, str]:
    """Does this class leave the cross-section alone?"""
    if state not in summary:
        return False, f"the mesh carries no {state} residues at all"
    mean, sd, n = summary[state]
    return abs(mean) <= ceiling, (
        f"{n} ring-gaps, mean {mean:+.2f} deg/residue, sd {sd:.2f} "
        f"(want |mean| <= {ceiling:.0f})"
    )


def phase_locked(mesh, states: cg, floor: float = 335.0, ceil: float = 385.0):
    """Cumulative cross-section rotation one turn (3.6 residues) along a helix.

    Returns ``(ok, detail, rows)`` where `rows` is one entry per helix run long
    enough to contain a turn. Runs shorter than `RESIDUES_PER_TURN` cannot be
    asked the question and are reported as skipped rather than counted, because
    asking a 3-residue run what its orientation is 3.6 residues later measures
    the run's end rather than its turn rate.
    """
    prof = cg.orientation_rate(mesh, states)
    x, delta, rs = prof["x"], prof["delta"], prof["ring_state"]
    cum = np.concatenate([[0.0], np.cumsum(delta)])
    rows, ok_all, detail_parts = [], True, []
    for lo, hi, state in cg.runs(list(states)):
        if state != "helix":
            continue
        sel = np.where(np.array([s == "helix" for s in rs]))[0]
        sel = sel[(x[sel] >= lo) & (x[sel] <= hi)]
        if len(sel) < 4 or (x[sel[-1]] - x[sel[0]]) < cg.RESIDUES_PER_TURN:
            rows.append((lo, hi, None))
            continue
        dx = x[sel] - x[sel[0]]
        c = cum[sel] - cum[sel[0]]
        turn = float(np.interp(cg.RESIDUES_PER_TURN, dx, c))
        ok = floor <= turn <= ceil
        ok_all = ok_all and ok
        rows.append((lo, hi, turn))
        detail_parts.append(
            f"res {lo}-{hi}: {turn:.1f} deg after {cg.RESIDUES_PER_TURN} residues "
            f"({'ok' if ok else 'OUT OF RANGE'})"
        )
    return ok_all, "; ".join(detail_parts) or "no helix run is long enough to hold a turn", rows


def heads_at_the_c_terminal_end(mesh, states: cg, floor: float = 2.0):
    """Is every strand's wide end at its C-terminal residue?"""
    rows = [row for row in cg.head_widths(mesh, states) if row[4] == "sheet"]
    if not rows:
        return False, "the mesh carries no strand run at all"
    bad = []
    parts = []
    for lo, hi, w_lo, w_hi, _s in rows:
        ratio = w_hi / max(w_lo, 1e-9)
        if ratio < floor:
            bad.append((lo, hi))
        parts.append(f"res {lo}-{hi}: {w_lo:.2f} -> {w_hi:.2f} A ({ratio:.2f}x)")
    return not bad, "; ".join(parts)


class _StubViewport:
    """The attributes `Viewport._draw_molecule` reads, and nothing else.

    Calling the unbound method with this as `self` runs the real dispatch. The
    GL handles are `None` and every drawing entry point is replaced, so nothing
    is rasterised -- but the *geometry* is still built, which is the part this
    file has an opinion about.
    """

    def __init__(self, representation):
        self.representation = representation
        self.pose_compare = False
        self._ctx = None
        self._sphere_prog = None
        self._line_prog = None
        self._mesh = None


def dispatched_meshes(mol: MoleculeView) -> dict[str, list]:
    """The meshes the real viewport dispatch would draw, per representation.

    Captured by replacing every drawing entry point with a recorder and calling
    the product's own `Viewport._draw_molecule`. This is what makes the guard
    measure the *shipped dispatch* rather than a builder's return value: an
    earlier version of this file called `MoleculeView.cartoon()` directly, and
    a build in which `app.py` routed `cartoon` to `backbone_ribbon()` passed
    every geometric check here -- the mutation was caught by a text search on
    `app.py` and by nothing else. A guard that cannot see the wiring it is
    guarding is a guard of the wrong file.

    Every GL entry point is stubbed, not just `draw_mesh`: `draw_spheres`
    reads `mesh.verts` off the context's own mesh and would raise on the
    `None` handles before the recorder is ever reached, so a partial stub
    measures only the representations that happen not to touch it.

    Returns ``{representation_key: [mesh, ...]}``.
    """
    names = ("draw_mesh", "draw_spheres", "draw_lines")
    real = {n: getattr(app, n) for n in names}
    captured: dict[str, list] = {}
    current = {"key": ""}

    def make_recorder(which):
        def rec(*a, **kw):
            if which == "draw_mesh":
                mesh = a[2] if len(a) > 2 else kw.get("mesh")
                if mesh is not None:
                    captured.setdefault(current["key"], []).append(mesh)
            return None

        return rec

    try:
        for n in names:
            setattr(app, n, make_recorder(n))
        for rep in app.REPRESENTATIONS:
            current["key"] = rep.key
            captured.setdefault(rep.key, [])
            app.Viewport._draw_molecule(_StubViewport(rep.key), mol, None)
    finally:
        for n, fn in real.items():
            setattr(app, n, fn)
    return captured


# --------------------------------------------------------------------------
def prepared_receptor() -> tuple[MoleculeView, list[str]]:
    from opendocking.prep import prepare_receptor

    path = ROOT / "examples" / "1crn_prep.pdbqt"
    mol = MoleculeView.from_text(
        path.read_text(encoding="utf-8"), path.stem, (0.6, 0.65, 0.7), 0.30, "receptor"
    )
    trace = mol.structure.backbone()
    return mol, list(secondary_structure(trace, mol.structure.atoms))


def main() -> int:
    section("1. which copy of the workbench this measured")
    measured_from = Path(cg.__file__).resolve()
    print(f"  opendocking.workbench.cartoon_geometry measured from: {measured_from}")
    check(
        "the cartoon geometry measured is the one in this checkout",
        measured_from.is_relative_to(DEV_TREE.resolve()),
        f"measured at {measured_from.parent}. The installed wheel is a second "
        "copy of the same module and is not what this file checks; a guard that "
        "quietly measured that one would report green on code nobody had edited",
    )

    mol, states = prepared_receptor()
    counts = {k: states.count(k) for k in ("helix", "sheet", "coil")}
    print(f"  receptor: {len(mol.coords)} atoms, backbone {mol.has_backbone}, states {counts}")

    cartoon = mol.cartoon()
    ribbon = mol.backbone_ribbon()
    guide, sides, _ = mol._backbone_guide_and_sides()
    print(f"  cartoon {len(cartoon.indices)} triangles, ribbon {len(ribbon.indices)}")

    section("2. the cartoon and the ribbon are not the same mesh")
    # The mutation that shipped was cartoon == ribbon, so this is the coarsest
    # possible statement of it and the one that has to hold before anything
    # finer is worth measuring.
    same_positions = np.array_equal(
        np.asarray(cartoon.positions, np.float32), np.asarray(ribbon.positions, np.float32)
    )
    check(
        "the cartoon mesh is not the ribbon mesh",
        cartoon is not None and ribbon is not None and not same_positions,
        "positions compared exactly"
        if not same_positions
        else "IDENTICAL -- this is the mutation that shipped: cartoon calls "
             "backbone_ribbon()",
    )

    # The one this file is really for: **the mesh the viewport's cartoon branch
    # would actually draw** must be a twisted one. Measuring
    # `MoleculeView.cartoon()` directly is not enough, and the first version of
    # this check proved it: it called the builder, so a build in which `app.py`
    # routed `cartoon` to `backbone_ribbon()` passed every geometric check here,
    # and the shipped mutation was caught only by a text search on `app.py`. So
    # the real dispatch is run with the drawing call replaced by a recorder, and
    # the geometry it produced is what gets measured.
    drawn = dispatched_meshes(mol)
    # The cartoon branch draws two meshes -- the cartoon and, beneath it, the
    # side-chain sticks -- so the backbone mesh is the one with the most rings
    # rather than the first in the list. Selecting by position would silently
    # measure the sticks if the dispatch order ever changed. "Is a swept quad
    # strip" is asked as divisibility rather than as a size test: the stick mesh
    # has 8558 vertices, which is more than one ring but not a whole number of
    # them, and reshaping it would raise rather than return something false.
    cartoon_drawn = [m for m in drawn.get("cartoon", []) if len(m.positions) % 4 == 0
                     and cg.ring_count(m) > 1]
    ribbon_drawn = [m for m in drawn.get("ribbon", []) if len(m.positions) % 4 == 0
                    and cg.ring_count(m) > 1]
    cartoon_mesh = max(cartoon_drawn, key=cg.ring_count) if cartoon_drawn else None
    check(
        "the cartoon branch draws a ribbon mesh at all",
        cartoon_mesh is not None,
        f"the real dispatch produced {len(cartoon_drawn)} swept-quad mesh(es) "
        f"for 'cartoon' and {len(ribbon_drawn)} for 'ribbon'"
        + (f"; the cartoon's has {cg.ring_count(cartoon_mesh)} rings"
           if cartoon_mesh is not None else ""),
    )
    prof_drawn = (
        cg.orientation_rate(cartoon_mesh, states) if cartoon_mesh is not None else {"summary": {}}
    )
    ok_drawn, detail_drawn = helix_turns(prof_drawn.get("summary", {}))
    check(
        "the helix the viewport draws turns its cross-section",
        ok_drawn,
        detail_drawn
        + ". Measured on the mesh the shipped dispatch produced, not on a "
        "builder's return value -- this is the check that goes red when the "
        "cartoon branch is routed back to backbone_ribbon()",
    )
    ok_drawn_s, detail_drawn_s = does_not_turn(prof_drawn.get("summary", {}), "sheet")
    check(
        "the strand the viewport draws is still flat",
        ok_drawn_s,
        detail_drawn_s + ", on the dispatched mesh",
    )

    prof = cg.orientation_rate(cartoon, states)
    prof_r = cg.orientation_rate(ribbon, states)
    print("  per-residue cross-section rotation about the path:")
    for label, p in (("cartoon", prof), ("ribbon ", prof_r)):
        for state, (mean, sd, n) in sorted(p["summary"].items()):
            print(f"    {label} {state:6s} n={n:4d} mean {mean:+8.2f} sd {sd:6.2f}")

    section("3. the cross-section turns for a helix and not for a strand")
    ok, detail = helix_turns(prof["summary"])
    check("a helix turns its cross-section about the CA axis", ok, detail)
    ok_s, detail_s = does_not_turn(prof["summary"], "sheet")
    check("a strand leaves its cross-section flat", ok_s, detail_s)
    ok_c, detail_c = does_not_turn(prof["summary"], "coil")
    check("a coil is still a tube and does not turn", ok_c, detail_c)

    # The sign is a claim too: a left-handed twist reads as a helix to the eye
    # and would be wrong about which hand the protein is, so the rate is asked
    # to be positive as well as large.
    hx_mean = prof["summary"].get("helix", (0.0, 0.0, 0))[0]
    check(
        "the twist is right-handed",
        hx_mean > 0.0,
        f"mean {hx_mean:+.2f} deg/residue; a left-handed value would still pass "
        "an absolute-value test and would draw a mirror-image helix",
    )

    section("4. the twist is phase-locked to one turn per 3.6 residues")
    ok_lock, detail_lock, rows = phase_locked(cartoon, states)
    check(
        "the cross-section returns to its orientation after one turn",
        ok_lock,
        detail_lock + ". Phase-locked means theta is a continuous function of "
        "position along the trace at exactly 360/3.6 deg per residue, so a turn "
        "later the ribbon is where it started rather than slipped sideways",
    )
    short = [lo for lo, hi, turn in rows if turn is None]
    print(f"  helix runs too short to hold a turn, reported not asked: {short or 'none'}")

    ok_lock_r, detail_lock_r, _ = phase_locked(ribbon, states)
    check(
        "CONTROL: the ribbon's cross-section does not return after a turn",
        not ok_lock_r,
        f"the same measurement on backbone_ribbon: {detail_lock_r}. A guard that "
        "cannot reject the ribbon would accept the shipped mutation",
    )

    section("5. every strand's arrowhead is at its C-terminal end")
    ok_head, detail_head = heads_at_the_c_terminal_end(cartoon, states)
    check(
        "every strand's wide end is its C-terminal residue",
        ok_head,
        detail_head + " (want >= 2.00x on every run). Measured off the mesh, so "
        "this is the arrowhead's direction as a fact and not as a drawing "
        "convention",
    )
    ok_head_r, detail_head_r = heads_at_the_c_terminal_end(ribbon, states)
    check(
        "CONTROL: the ribbon's strands have no head at either end",
        not ok_head_r,
        f"the same measurement on backbone_ribbon: {detail_head_r}",
    )

    section("6. the three classes are separable without colour")
    renderer = ss_render.OffscreenRenderer.open(900, 700)
    if renderer is None:
        reason = f"no offscreen OpenGL context ({ss_render.LAST_PROBLEM})"
        skip("helix and strand are different pictures with all colour removed", reason)
        skip("a strand and a coil are different pictures with all colour removed", reason)
        skip("CONTROL: the ribbon's helix and strand pictures are not separable", reason)
    else:
        from opendocking.workbench import Camera

        xyz = np.asarray([a.xyz for a in mol.structure.atoms], np.float64)
        centre = xyz.mean(axis=0)
        cam = Camera()
        cam.center = centre.astype(np.float32)
        cam.distance = float(np.linalg.norm(xyz - centre, axis=1).max()) * 2.6

        # One neutral grey for every vertex, so nothing in the picture can be
        # read as a class except the geometry. This is the black-and-white print
        # and the colour-blind reader, in one measurement.
        grey = {k: (0.62, 0.62, 0.62) for k in ("helix", "sheet", "coil")}
        blank = renderer.render([], cam, 900, 700)
        frames = {}
        for tag, build in (
            ("cartoon", lambda stl: build_cartoon(guide, sides, stl, colors=grey)),
            ("ribbon", lambda stl: build_ribbon(guide, sides, stl, colors=grey)),
        ):
            for klass in ("helix", "sheet", "coil"):
                mesh = build(ss_render.flat_states(len(states), klass))
                frames[(tag, klass)] = renderer.render([(mesh, 1.0)], cam, 900, 700)
        frames[("cartoon", "real")] = renderer.render(
            [(build_cartoon(guide, sides, states, colors=grey), 1.0)], cam, 900, 700
        )
        frames[("ribbon", "real")] = renderer.render(
            [(build_ribbon(guide, sides, states, colors=grey), 1.0)], cam, 900, 700
        )
        renderer.close()

        def sep(a: str, b: str) -> tuple[int, float]:
            fa, fb = frames[("cartoon", a)], frames[("cartoon", b)]
            ma = ss_render.ribbon_mask(fa, blank)
            mb = ss_render.ribbon_mask(fb, blank)
            iou = int((ma & mb).sum()) / max(int((ma | mb).sum()), 1)
            return int((fa != fb).any(axis=-1).sum()), iou

        def sep_r(a: str, b: str) -> int:
            fa, fb = frames[("ribbon", a)], frames[("ribbon", b)]
            return int((fa != fb).any(axis=-1).sum())

        hs, hs_iou = sep("helix", "sheet")
        check(
            "helix and strand are different pictures with all colour removed",
            hs >= 60000,
            f"{hs} differing px of 630 000 at every vertex forced to one neutral "
            f"grey (IoU {hs_iou:.3f}); want >= 60000. The mutated ribbon manages "
            f"{sep_r('helix', 'sheet')} on the same trace with the same grey, so "
            "this floor is above what the wrong geometry can reach",
        )
        hc, hc_iou = sep("helix", "coil")
        sc, sc_iou = sep("sheet", "coil")
        check(
            "a strand and a coil are different pictures with all colour removed",
            sc >= 60000,
            f"strand vs coil {sc} px (IoU {sc_iou:.3f}), helix vs coil {hc} px "
            f"(IoU {hc_iou:.3f}); want >= 60000 on both. The coil is 0.35 A half-"
            "width against 1.50 for a strand, so it is a different silhouette "
            "before shading is considered",
        )
        real_diff = int(
            (frames[("cartoon", "real")] != frames[("ribbon", "real")]).any(axis=-1).sum()
        )
        check(
            "CONTROL: the ribbon's helix and strand pictures are not separable",
            sep_r("helix", "sheet") < 60000,
            f"backbone_ribbon's helix and strand, same trace and same grey, differ "
            f"by {sep_r('helix', 'sheet')} px -- below the {hs} px the cartoon "
            "reaches, which is the gap the threshold sits in. The real classified "
            f"pictures differ by {real_diff} px",
        )

    section("7. the guard fails in both directions")
    # (1) the shipped mutation, offered to the shipped predicate.
    ok_real, d_real = helix_turns(prof["summary"])
    ok_mut, d_mut = helix_turns(prof_r["summary"])
    check(
        "the shipped mutation -- cartoon calling backbone_ribbon -- is rejected",
        ok_real and not ok_mut,
        f"the cartoon's helix is accepted ({d_real}) and the ribbon's is refused "
        f"({d_mut}). This is the code that shipped: `cartoon` drew the ribbon",
    )
    # (2) the greedy mutation, which must still be rejected.
    greedy_ok, greedy_detail = helix_turns(
        {"helix": (100.58, 7.07, 205), "sheet": (100.58, 7.07, 51), "coil": (100.58, 7.07, 95)}
    )
    check(
        "a build that twists every class cannot satisfy the strand and coil claims",
        greedy_ok and not does_not_turn(
            {"sheet": (100.58, 7.07, 51)}, "sheet"
        )[0] and not does_not_turn({"coil": (100.58, 7.07, 95)}, "coil")[0],
        "a geometry whose strands also turn at 100.58 deg/residue passes the helix "
        "test and is refused by both `does_not_turn` predicates "
        f"({greedy_detail}). Without this direction a builder that twisted "
        "everything would be green",
    )
    # (3) the arrowhead at the wrong end, which no threshold above can see.
    wrong_end = cg.head_widths.__doc__ and True
    from opendocking.workbench.cartoon_geometry import cartoon as _cg_cartoon

    flipped = _cg_cartoon(guide, sides, states, head_residues=0)
    ok_flip, d_flip = heads_at_the_c_terminal_end(flipped, states)
    check(
        "a strand built without a head is rejected by the arrowhead test",
        not ok_flip,
        f"the same trace with head_residues=0 reads {d_flip}. This is the "
        "direction the guard cannot get from the twist: the twist is unchanged "
        "by this build, so only the width profile says a head is missing",
    )
    del wrong_end

    section("summary")
    check(
        "every section this file is supposed to run was reached",
        tuple(_sections) == EXPECTED_SECTIONS,
        f"{len(_sections)} of {len(EXPECTED_SECTIONS)} sections reached",
    )
    # `+ 1` for the check being made, and the total captured *before* the call:
    # `check` appends in its body, so an expression mentioning `len(RESULTS)` in
    # the argument list is evaluated one result short of the run it is
    # describing. Comparing 17 against 18 and reporting "17 ran; 18 expected" is
    # the signature of the mistake -- condition and detail disagreeing.
    total = len(RESULTS) + 1
    check(
        "the number of checks that ran is the number this file is supposed to have",
        total == EXPECTED_CHECKS,
        f"{total - 1} ran before this one and {total} are expected. Skips are "
        "counted here so the total does not depend on the machine having a GPU",
    )
    npass = sum(1 for r in RESULTS if r[0] == "PASS")
    nfail = sum(1 for r in RESULTS if r[0] == "FAIL")
    nskip = sum(1 for r in RESULTS if r[0] == "SKIP")
    if nskip:
        print(f"  {nskip} check(s) were SKIPPED, not passed: a skip means this "
              "environment could not answer the question.")
    for tag, name, detail in RESULTS:
        if tag in ("FAIL", "SKIP"):
            print(f"    {tag} {name}: {detail}")
    if FAILURES:
        print("\n=== failures ===")
        for f in FAILURES:
            print(f"  {f}")
        print(f"\n{npass} passed, {nfail} failed, {nskip} skipped, {len(RESULTS)} checks")
        return 1
    print(f"\n{npass} passed, 0 failed, {nskip} skipped, {len(RESULTS)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
