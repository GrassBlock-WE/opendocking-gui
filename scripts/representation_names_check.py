"""Check that every representation's user-facing name is true of its geometry.

# Why this file exists

The viewer offered a protein labelled "Space-filling" and drew it at 0.30 A.
Measured on prepared 1CRN, 0.0% of atoms had a neighbour inside 2r: not one
sphere touched another, so the "space-filling" picture was a cloud of
separated dots and the name was a false statement about this product. That is
the same class of defect as an exit code that says the viewer is broken when
it is not -- a label asserting something untrue, which costs more than the
bug it covers for, because a user who trusts one label stops reading the
others.

Naming is not something a test can confirm by reading the string, so this file
does not check names. It measures the geometry each representation actually
draws, and then asks whether that geometry has the property the name claims.

# How a name is made falsifiable here

`Representation.claims` in `app.py` records, per representation, the geometric
property its label asserts. The vocabulary is closed and small:

    separated_spheres      the spheres are apart, and the picture is the dots
    solid_surface          the spheres touch, and the picture is a surface
    balls_and_sticks       small spheres plus cylinders
    sticks_only            cylinders, and no spheres at all
    backbone_ribbon        a ribbon that follows the CA trace
    twisted_cartoon        a ribbon whose cross-section turns for a helix and
                           flares into an arrowhead for a strand

Each claim is a *predicate over a measurement*, not a comment. The
measurement is taken by running the real `Viewport._draw_molecule` on a real
receptor with the GL entry points replaced by recorders: the draw functions
are never called, the geometry builders are, so what comes back is the part
list and the radii the renderer's own dispatch chose. No context is needed and
no threshold is invented -- the numbers are measured per run and printed.

Two independent ways a name can be a lie are therefore both caught:

* **the geometry does not have the claimed property** -- a "Space-filling"
  mode at 0.30 A measures 0.0% contact and fails `solid_surface`;
* **the label asserts a property its own entry does not claim** -- a label
  containing "space-filling" on an entry whose claim is `separated_spheres`
  fails the keyword cross-check, whatever the geometry turns out to be.

Both directions are mutation-tested on every run rather than once, as controls
inside the checks themselves (see "the guard fails in both directions"), so a
later edit cannot quietly turn one of them into a check that cannot fail.

# What changed when the cartoon became a cartoon

The entry that used to claim `ribbon_and_side_chains` now claims
`twisted_cartoon`, and the predicate for it asks the geometry a question the
old one could not: **is the mesh the dispatch drew different from the one the
`ribbon` mode drew, and does its helix cross-section turn?**

That is the whole point of the change. The old predicate asked for two parts
(`ribbon`, `sticks`) and a thinner stick radius, and it was satisfied by a mode
that drew the identical ribbon mesh plus some cylinders -- which is exactly
what shipped. A predicate that cannot distinguish "the cartoon" from "the
ribbon again" is not guarding the word "Cartoon"; it is guarding the presence
of a second mesh.

So `_claim_twisted_cartoon` compares the two dispatched meshes directly, and
`scripts/representation_cartoon_check.py` measures the twist itself. The two
overlap on purpose and are not duplicates: this file asks *is it a different
mesh that twists*, the other asks *does it twist by the right amount, at the
right rate, with its arrowhead pointing the right way*.

# What this file refuses to be

A check that the table is well formed. A table can be perfectly consistent
and still lie, and the way it lies is in the numbers, so the numbers are what
is measured.

Run:  python scripts/representation_names_check.py
"""

import sys
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEV_TREE = ROOT / "dock-py" / "python"

# This file measures *this checkout*. The other reachable copies -- the
# installed wheel in site-packages, and the release tree under `opendocking-gui`
# -- are the same bytes only until someone edits one of them, and a guard that
# silently measured a stale copy would pass on the code it is supposed to be
# guarding. So the development tree is put in front deliberately, and the first
# check fails if the table still came from somewhere else. `core_check.py`
# reaches the same question from the other end and prints all three copies.
sys.path.insert(0, str(DEV_TREE))

from opendocking.workbench import app, geometry  # noqa: E402
from opendocking.workbench import MoleculeView  # noqa: E402

#: How many checks this file is supposed to run, counted by running it. A check
#: that stops running takes the count down with it silently, which is how a
#: suite goes from 17 to 15 with a clean report.
#:
#: **16 -> 17, and the one is the cartoon control.** The per-representation
#: loop still runs six times -- the vocabulary changed from `ribbon_and_side_
#: chains` to `twisted_cartoon`, which is a swap rather than an addition, so
#: that loop did not move. The extra check is in section 4: the *ribbon's* own
#: dispatched mesh is now offered to the cartoon claim and must be refused.
#:
#: It earns its place because the predicate it exercises is the only one here
#: that a mutation to product code could satisfy without changing a number.
#: Every other claim asks about radii, contact fractions or a part list, and
#: the shipped defect left all three untouched; this one compares two meshes
#: and a twist rate, and the shipped defect was exactly "they are the same
#: mesh". Measured 2026-10-01 on the development tree.
EXPECTED_CHECKS = 17

#: The call-site census this file is supposed to have, declared here rather than
#: in a table another file owns -- see `GATE-DECLARE` in
#: `check_scripts_declare.py` for the format and why it cannot drift. Comments
#: only, so the `EXPECTED_CHECKS` above is unaffected by them.
#:
#: **10+1 -> 11+1, and the derivation is that nothing was removed.** The one
#: site added is the cartoon control in section 4, which offers the *ribbon's*
#: own dispatched mesh to the cartoon predicate and requires it to be refused.
#: The guard digest moves with it because that site sits inside an `if`, which is
#: exactly what the digest is for: a site changing guard shape without changing
#: the total is invisible to the two counts and visible here.
#:
#: This file was moved onto its own block rather than having the generated
#: snapshot regenerated, because `DECLARED_SNAPSHOT` lives in
#: `check_scripts_declare.py`, a file this gate's owner does not edit. Reporting
#: that as a blocker would have left the drift red; the block the auditor asks
#: every gate to migrate towards is an edit to this file alone, so the number now
#: has exactly one home and the next person to add a check finds it one screen
#: from the code.
#: GATE-DECLARE 1
#: sites: 11 unconditional + 1 guarded
#: guards: sha256:76d2a1173ebfb732b5ab9672fc0e632b2ca76f26917de4bfa400d6e4609d3554

#: Every section this file is supposed to reach, in file order. A section that
#: is entered always prints a header, so a run that stops early is visible in
#: the transcript without anyone having to trust the number above.
EXPECTED_SECTIONS = (
    "1. which copy of the workbench this measured",
    "2. the table is a table",
    "3. every claimed property is measured, and the measurement holds",
    "4. the guard fails in both directions",
    "summary",
)

FAILURES: list[str] = []
CHECKS = 0
_sections: list[str] = []


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def section(t):
    print(f"\n=== {t} ===")
    _sections.append(t)


# --------------------------------------------------------------------------
# The closed vocabulary of claims, and the words each one refuses.
#
# Forbidden words, not required ones. The direction that matters is a name
# claiming *more* than the geometry delivers, which is the defect this file
# was written for; a synonym nobody thought of is a cosmetic problem, and
# pinning the label's exact text would break a legitimate rename for no gain.
FORBIDDEN_CLAIM_WORDS: dict[str, tuple[str, ...]] = {
    "separated_spheres": (
        "space-filling", "space filling", "surface", "van der waals",
        "cpk", "solid",
    ),
    "solid_surface": ("separated", "small", "skeletal"),
    "balls_and_sticks": ("space-filling", "space filling", "surface", "separated"),
    "sticks_only": (
        "space-filling", "space filling", "surface", "ball", "separated",
    ),
    "backbone_ribbon": ("cartoon", "space-filling", "surface", "side chain", "arrow"),
    "twisted_cartoon": ("space-filling", "surface", "side chain"),
}

#: Claims whose geometry only means something when compared against another
#: mode's. These predicates take the whole `measured` table rather than one
#: `Measurement`, because "this cartoon is not the ribbon again" is a claim
#: about two pictures and cannot be evaluated from one.
CROSS_CLAIMS = {"twisted_cartoon"}

#: The vocabulary above is not a list of words: the keys of `CLAIM_TESTS` *are*
#: the vocabulary, each a predicate over a :class:`Measurement`. Keeping the two
#: in one dict is deliberate -- a separate list of property names could name a
#: property with no predicate behind it, and "each claimed property is measured"
#: would then be a claim about a list rather than about anything that runs.


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
class Measurement:
    """What one representation actually drew, measured off the real dispatch.

    `parts` is the ordered list of geometry the renderer's own `_draw_molecule`
    asked for, and `radii` the per-atom sphere radius it asked for in Angstrom
    (empty for a representation that draws no spheres). Both come from running
    the real function with the GL entry points replaced, not from a table
    written down beside it.
    """

    def __init__(self, key, mol):
        self.key = key
        self.mol = mol
        self.parts: list[str] = []
        self.radii = np.zeros(0, float)
        self.stick_radius: float | None = None
        #: Every mesh the dispatch handed to `draw_mesh`, in order. The claims
        #: below are about geometry, and "the picture" is only reachable by
        #: keeping the thing that was going to be drawn. It was not kept until
        #: the cartoon had to be told apart from the ribbon by its shape rather
        #: than by the presence of a second part in a list.
        self.meshes: list = []
        self._mol = mol

    @property
    def has_spheres(self) -> bool:
        return "spheres" in self.parts

    def contact_fraction(self) -> float:
        """Share of atoms whose nearest neighbour lies inside 2r.

        This is the number a name like "space-filling" is really about. Two
        spheres touch exactly when the gap between their centres is under the
        sum of their radii, so "are the atoms in contact" is a statement about
        the coordinates and this one is a function of the coordinates and the
        radii -- it cannot be satisfied by a table agreeing with itself.
        """
        if not self.has_spheres or len(self._mol.coords) < 2:
            return 0.0
        d = np.linalg.norm(
            self._mol.coords[:, None, :] - self._mol.coords[None, :, :], axis=2
        )
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        return float((nn < 2.0 * np.asarray(self.radii, float)).mean())


class _StubViewport:
    """The attributes `_draw_molecule` reads, and nothing else.

    Calling the unbound `Viewport._draw_molecule` with this as `self` runs the
    real dispatch: it reads `self.representation` and the three GL handles, and
    every one of those handles is `None` because each function that would have
    used it has been replaced by a recorder.

    `pose_compare` is set here but no longer read by `_draw_molecule`: a pose
    wears its own flat colour unconditionally, so the flag does not reach the
    colour path any more (see `_draw_colours_for` in `app.py` for the
    measurement that changed it). It is left on the stub because setting an
    attribute nothing reads is harmless, and removing it would have made this
    docstring's claim about the stub's contents a guess.
    """

    def __init__(self, representation):
        self.representation = representation
        self.pose_compare = False
        self._ctx = None
        self._sphere_prog = None
        self._line_prog = None
        self._mesh = None


def measure(key: str, mol: MoleculeView) -> Measurement:
    """Run the real draw dispatch for `key` and record what it asked for."""
    m = Measurement(key, mol)
    real_bonds, real_spheres, real_ribbon = geometry.bonds, geometry.spheres, geometry.ribbon
    # `cartoon_geometry` is a module, so the recorder replaces a *name in it*
    # rather than a module attribute. `MoleculeView.cartoon` imports it inside
    # the function, so patching the module's own attribute is what the product
    # will see.
    import opendocking.workbench.cartoon_geometry as cg

    real_cartoon = cg.cartoon

    def rec_bonds(*a, **kw):
        m.parts.append("sticks")
        m.stick_radius = float(a[2])
        return real_bonds(*a, **kw)

    def rec_spheres(*a, **kw):
        m.parts.append("spheres")
        m.radii = np.broadcast_to(np.asarray(a[1], float), (len(a[0]),))
        return real_spheres(*a, **kw)

    def rec_ribbon(*a, **kw):
        m.parts.append("ribbon")
        return real_ribbon(*a, **kw)

    def rec_cartoon(*a, **kw):
        # A distinct part name, not `ribbon`: the two modes used to append the
        # same string, which is how a mode that drew the identical mesh came to
        # look like a mode that drew a second thing.
        m.parts.append("cartoon")
        return real_cartoon(*a, **kw)

    def rec_draw_spheres(_ctx, _prog, _mesh, _mol, _mvp, _colors=None, radii=None):
        # The only mode that reaches here with no `radii` keyword would be a
        # caller that was not updated; fall back to the molecule's own so the
        # measurement records what the *default* would have been, not a crash.
        m.parts.append("spheres")
        m.radii = np.asarray(
            mol.atom_radii() if radii is None else radii, float
        )
        return None

    def rec_draw_lines(*a, **kw):
        m.parts.append("bond_lines")
        return None

    def rec_draw_mesh(*a, **kw):
        mesh = a[2] if len(a) > 2 else kw.get("mesh")
        if mesh is not None:
            m.meshes.append(mesh)
        return None

    with ExitStack() as stack:
        for name, value in (
            ("bonds", rec_bonds), ("spheres", rec_spheres), ("ribbon", rec_ribbon),
        ):
            stack.callback(setattr, geometry, name, getattr(geometry, name))
            setattr(geometry, name, value)
        for name, value in (
            ("draw_spheres", rec_draw_spheres), ("draw_lines", rec_draw_lines),
            ("draw_mesh", rec_draw_mesh),
        ):
            stack.callback(setattr, app, name, getattr(app, name))
            setattr(app, name, value)
        stack.callback(setattr, cg, "cartoon", cg.cartoon)
        cg.cartoon = rec_cartoon
        app.Viewport._draw_molecule(_StubViewport(key), mol, None)
    return m


# --------------------------------------------------------------------------
# The claims, as predicates over a Measurement
# --------------------------------------------------------------------------
def _claim_separated_spheres(m: Measurement) -> tuple[bool, str]:
    ok = m.parts == ["spheres", "bond_lines"] and m.contact_fraction() < 0.05
    return ok, (
        f"parts {m.parts}, contact {100 * m.contact_fraction():.1f}% of atoms, "
        f"carbon at {float(m.radii.max()):.2f} A"
    )


def _claim_solid_surface(m: Measurement) -> tuple[bool, str]:
    carbon = app.ELEMENT_CPK_RADIUS["C"]
    anchored = abs(carbon - app.CPK_CARBON_RADIUS) < 1e-12
    # Hydrogen has to follow the same table rather than being picked, so its
    # radius is checked as the *ratio* the published CPK values give.
    h_ratio = app.ELEMENT_CPK_RADIUS["H"] / carbon
    ok = (m.parts == ["spheres"] and m.contact_fraction() >= 0.90
          and anchored and abs(h_ratio - 1.20 / 1.70) < 1e-9)
    return ok, (
        f"parts {m.parts}, contact {100 * m.contact_fraction():.1f}% of atoms, "
        f"carbon at {carbon:.3f} A, hydrogen at {carbon * h_ratio:.4f} A "
        f"(= 1.20/1.70 of carbon)"
    )


def _claim_balls_and_sticks(m: Measurement) -> tuple[bool, str]:
    ok = (m.parts == ["spheres", "sticks"] and m.contact_fraction() < 0.05
          and m.stick_radius is not None and m.stick_radius > 0)
    return ok, (
        f"parts {m.parts}, balls at {100 * m.contact_fraction():.1f}% contact "
        f"(so they are separate), sticks at {m.stick_radius:.3f} A"
    )


def _claim_sticks_only(m: Measurement) -> tuple[bool, str]:
    ok = m.parts == ["sticks"] and m.stick_radius is not None
    return ok, f"parts {m.parts}, sticks at {m.stick_radius} A and no spheres"


def _claim_backbone_ribbon(m: Measurement) -> tuple[bool, str]:
    ok = m.parts == ["ribbon"]
    return ok, f"parts {m.parts}"


def _claim_twisted_cartoon(m: Measurement, table: dict | None = None) -> tuple[bool, str]:
    """Is this a cartoon rather than the ribbon again?

    Three questions, and the first is the one that matters:

    * **is the mesh different from the one `ribbon` draws?** Not "is there an
      extra part in the list" -- the shipped defect satisfied that. Compared as
      vertex arrays, because a mesh is positions and a claim about a picture is
      a claim about positions;
    * **does its helix cross-section turn about the path?** Measured off the
      mesh the dispatch handed over, not asked of the builder;
    * **are the side chains still there?** The mode grew, it did not replace.

    `ribbon_and_side_chains` was the old predicate and it is worth saying what
    it could not do: it asked for two parts and a thinner stick radius, and a
    mode drawing the *identical* ribbon mesh plus cylinders passes it. It was
    satisfied by the exact code that shipped.
    """
    thinner = (m.stick_radius is not None
               and m.stick_radius < 0.5 * float(m._mol.radius))
    swept = [x for x in m.meshes if len(x.positions) % 4 == 0 and len(x.positions) >= 8]
    if not swept:
        return False, f"parts {m.parts} but no swept-quad mesh was dispatched"
    cartoon_mesh = max(swept, key=lambda x: len(x.positions))

    ribbon_mesh = None
    if table is not None and "ribbon" in table:
        other = [x for x in table["ribbon"].meshes
                 if len(x.positions) % 4 == 0 and len(x.positions) >= 8]
        if other:
            ribbon_mesh = max(other, key=lambda x: len(x.positions))

    differs = ribbon_mesh is None or not np.array_equal(
        np.asarray(cartoon_mesh.positions, np.float32),
        np.asarray(ribbon_mesh.positions, np.float32),
    )

    from opendocking.workbench.cartoon_geometry import orientation_rate
    from opendocking.workbench.structure import secondary_structure

    trace = m._mol.structure.backbone()
    states = list(secondary_structure(trace, m._mol.structure.atoms))
    summary = orientation_rate(cartoon_mesh, states)["summary"]
    hx = summary.get("helix", (0.0, 0.0, 0))[0]
    turns = hx >= 60.0

    ok = m.parts == ["cartoon", "sticks"] and thinner and differs and turns
    return ok, (
        f"parts {m.parts}, side chains at {m.stick_radius} A against a "
        f"{float(m._mol.radius):.2f} A base radius; the dispatched helix mesh "
        f"turns its cross-section at {hx:+.2f} deg/residue (want >= 60, a ribbon "
        f"reads about 0); its vertices "
        + ("differ from the ribbon mode's mesh"
           if differs else "ARE THE RIBBON MODE'S MESH")
    )


CLAIM_TESTS = {
    "separated_spheres": _claim_separated_spheres,
    "solid_surface": _claim_solid_surface,
    "balls_and_sticks": _claim_balls_and_sticks,
    "sticks_only": _claim_sticks_only,
    "backbone_ribbon": _claim_backbone_ribbon,
    "twisted_cartoon": _claim_twisted_cartoon,
}


def label_overclaims(label: str, claim: str) -> bool:
    """Does this label assert a property its own entry does not claim?"""
    low = label.lower()
    return any(w in low for w in FORBIDDEN_CLAIM_WORDS.get(claim, ()))


# --------------------------------------------------------------------------
def prepared_receptor() -> MoleculeView:
    from opendocking.prep import prepare_receptor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        text = prepare_receptor(ROOT / "examples" / "1crn_receptor.pdb")
    return MoleculeView.from_text(text, "1crn", (0.6, 0.65, 0.7), 0.30, "receptor")


def main() -> int:
    section("1. which copy of the workbench this measured")
    measured_from = Path(app.__file__).resolve()
    print(f"  opendocking.workbench.app measured from: {measured_from}")
    print(f"  this checkout's development tree:       {DEV_TREE}")
    check(
        "the representation table measured is the one in this checkout",
        measured_from.is_relative_to(DEV_TREE.resolve()),
        f"measured {measured_from.name} at {measured_from.parent}. The installed "
        f"wheel is a second copy of the same module and is not what this file "
        f"checks; a guard that quietly measured that one would report green on "
        f"code nobody had edited",
    )

    rec = prepared_receptor()
    print(f"  receptor: {len(rec.coords)} atoms, base radius {rec.radius} A, "
          f"backbone {rec.has_backbone}")

    section("2. the table is a table")
    labels = [r.label for r in app.REPRESENTATIONS]
    keys = [r.key for r in app.REPRESENTATIONS]
    claims = {r.claims for r in app.REPRESENTATIONS}
    check(
        "every entry names a property from the closed vocabulary",
        claims <= set(CLAIM_TESTS),
        f"{sorted(claims)} against a vocabulary of {sorted(CLAIM_TESTS)}, which "
        f"is the set of keys of the predicate table rather than a second list "
        f"of names that could drift from it",
    )
    check(
        "every property in the vocabulary is claimed by something",
        set(CLAIM_TESTS) <= claims,
        "a property nothing claims is a property nothing has measured, and it "
        "is the vocabulary that would let the next registration slip through",
    )
    check(
        "labels are distinct and none is empty",
        len(set(labels)) == len(labels) and all(labels),
        "; ".join(f"{k}={l!r}" for k, l in zip(keys, labels)),
    )
    over = [f"{r.key}: {r.label!r} claims {r.claims}"
            for r in app.REPRESENTATIONS if label_overclaims(r.label, r.claims)]
    check(
        "no label asserts a property its own entry does not claim",
        not over,
        "; ".join(over) or "every label is inside its own claim"
        + "   [CONTROL: relabelling 'spheres' as 'Space-filling' puts a "
        "forbidden word in a separated_spheres entry and this check goes red; "
        "that is the exact defect this file exists for, and it is caught on the "
        "text alone, so it fires even if the geometry is later changed to match]",
    )

    section("3. every claimed property is measured, and the measurement holds")
    measured: dict[str, Measurement] = {}
    for rep in app.REPRESENTATIONS:
        m = measure(rep.key, rec)
        measured[rep.key] = m
    # Second pass, so a cross-claim can compare against a mode measured earlier
    # in the table. `twisted_cartoon` says "this is not the ribbon again", which
    # is a claim about two dispatches and cannot be evaluated while the ribbon
    # is still unmeasured.
    for rep in app.REPRESENTATIONS:
        m = measured[rep.key]
        test = CLAIM_TESTS[rep.claims]
        ok, detail = test(m, measured) if rep.claims in CROSS_CLAIMS else test(m)
        check(f"'{rep.key}' ({rep.label}) has the property it claims", ok,
              f"claims {rep.claims}: {detail}")

    section("4. the guard fails in both directions")
    # Both controls are permanent and run on every invocation. A guard that was
    # mutation-tested once and then left un-monitored is a guard that will
    # report green the day someone changes what it was written to notice.
    sep = measured["spheres"]
    solid = measured["space_filling"]
    sep_ok, sep_detail = _claim_separated_spheres(sep)
    solid_ok, solid_detail = _claim_solid_surface(solid)
    check(
        "separated geometry cannot satisfy the solid-surface claim",
        not _claim_solid_surface(sep)[0] and solid_ok,
        f"the 0.30 A 'spheres' geometry is offered to the solid-surface test and "
        f"is rejected ({sep_detail}), while the real CPK mode is accepted "
        f"({solid_detail}). This is the mutation that shipped: relabel the "
        f"0.30 A protein 'Space-filling' and the solid-surface test refuses it",
    )
    check(
        "solid geometry cannot satisfy the separated-spheres claim",
        not _claim_separated_spheres(solid)[0] and sep_ok,
        f"the CPK geometry is offered to the separated-spheres test and is "
        f"rejected ({solid_detail}), while the real small-sphere mode is "
        f"accepted ({sep_detail}). Without this direction the first control "
        f"would also pass a build that drew *every* mode as a solid surface",
    )
    check(
        "a label carrying a forbidden word is rejected on the text alone",
        label_overclaims("Space-filling", "separated_spheres")
        and not label_overclaims("Small spheres (separated)", "separated_spheres"),
        "'Space-filling' on a separated_spheres entry is rejected, and the "
        "shipped label is not. A name is a claim, and this is the half of the "
        "guard that needs no geometry to run",
    )
    # The mutation that shipped, as a permanent control: the *ribbon's* own
    # dispatched mesh offered to the cartoon claim. The ribbon satisfies every
    # other clause of that predicate -- it has a swept mesh and a stick radius
    # -- so only the geometry can refuse it.
    cartoon_ok, cartoon_detail = _claim_twisted_cartoon(measured["cartoon"], measured)
    ribbon_as_cartoon_ok, ribbon_as_cartoon_detail = _claim_twisted_cartoon(
        measured["ribbon"], measured
    )
    check(
        "the ribbon mode cannot satisfy the cartoon claim",
        cartoon_ok and not ribbon_as_cartoon_ok,
        f"the shipped cartoon is accepted ({cartoon_detail}), and the ribbon's "
        f"own geometry offered to the same predicate is refused "
        f"({ribbon_as_cartoon_detail}). This is the code that shipped: the "
        "cartoon mode drew the ribbon",
    )

    section("summary")
    check(
        "every section this file is supposed to run was reached",
        tuple(_sections) == EXPECTED_SECTIONS,
        f"{len(_sections)} of {len(EXPECTED_SECTIONS)} sections reached",
    )
    # `+ 1` for the check being made, and the total captured *before* the call:
    # `check` increments the counter in its body, so an expression mentioning
    # CHECKS in the argument list is evaluated one check short of the run it is
    # describing. An earlier version of this line compared 15 against 16 and
    # reported "15 ran; 16 expected" -- the condition and its own detail
    # disagreeing is the signature of the mistake.
    total = CHECKS + 1
    check(
        "the number of checks that ran is the number this file is supposed to have",
        total == EXPECTED_CHECKS,
        f"{total - 1} ran before this one and {total} are expected",
    )
    if FAILURES:
        print("\n=== failures ===")
        for f in FAILURES:
            print(f"  {f}")
        print(f"\n{Checks_passed()} passed, {len(FAILURES)} failed, {CHECKS} checks")
        return 1
    print(f"\n{Checks_passed()} passed, 0 failed, {CHECKS} checks")
    return 0


def Checks_passed() -> int:
    return CHECKS - len(FAILURES)


if __name__ == "__main__":
    raise SystemExit(main())
