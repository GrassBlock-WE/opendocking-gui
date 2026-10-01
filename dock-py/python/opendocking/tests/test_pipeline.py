"""End-to-end tests for the Open Docking Python stack.

These exercise the whole pipeline the way a user does: prepare a structure with
RDKit, precalculate maps, dock, and check the *physics* — that the ligand ends
up in the pocket, that the energy is negative and finite, that poses are
distinct, and that the analytic gradient agrees with finite differences.

Assertions are about behaviour, not about memorised numbers. A docking engine
whose energies changed by a few tenths after a refactor should not fail its
test suite; one that put the ligand outside the pocket should.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from opendocking import (
    GridBox,
    Ligand,
    Receptor,
    dock,
    evaluate_conformations,
    gpu_status,
    load_ligand,
    score_conformation,
)
from opendocking.tests.make_data import make_receptor_pdb, write_all

pytest.importorskip("rdkit", reason="RDKit is required for structure preparation")

# A box comfortably around the synthetic receptor's pocket.
BOX = GridBox.from_center_size((0.0, 0.0, 1.5), (18.0, 18.0, 18.0))

# ---------------------------------------------------------------------------
# The line search, and why it is a ladder
# ---------------------------------------------------------------------------
#
# A pose is checked for descent by stepping along the normalised ``-grad``
# direction and asking whether the energy went down. "Can this pose still
# descend" is therefore not a number, it is a function of the step, and a
# single step answers a question about the step as much as about the pose.
#
# The rungs below are conformation units, chosen from what the pose audit
# measures (`examples/audit_poses.py`, whose `_LINE_SEARCH_STEPS` this
# mirrors) rather than from what reads well. The two that decide the
# argument are the ends:
#
#   FINEST (1e-9)
#       Below the step this file used to probe at. The maps are trilinearly
#       interpolated, so the field is C0 across a cell face -- continuous in
#       value, with a jump in its one-sided derivative -- and a pose sitting
#       on a face has a descent region only a few times 1e-9 wide. A 1e-9 step
#       is inside every one of those regions, so the descent it reports is a
#       first-order term and nothing else.
#   COARSE (1e-4 upward)
#       The step this file used to probe at, and the one the audit reported
#       zeros for. It is *wider* than the region, so it lands past the
#       turn-up and reads zero. That is a fact about the probe, and the two
#       tests below exist to keep the two apart.
FINEST_STEP = 1e-9
#: The intermediate rungs exist to locate the turn-over, not to sample the
#: descent: two decades either side of the old probe step is enough to bracket
#: a region that is demonstrably narrower than the old probe step.
LADDER_STEPS = (1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
#: The steps the old assertion used, kept verbatim: the claim "no descent at
#: 1e-4 through 0.3" is still worth making, and it is only honest if the
#: resolution it was made at is named next to it.
COARSE_STEPS = (1e-4, 1e-3, 0.01, 0.1, 0.3)
#: The optimiser's own declared stopping tolerance, `gradient_tolerance` in
#: `dock-core/src/search/lbfgs.rs`. A pose whose gradient is far above this is
#: not a stationary point of the field the engine scores, whatever any line
#: search says about it -- which is what makes it the one rung of this
#: argument that does not depend on the step at all.
GRADIENT_TOLERANCE = 1e-4
#: How close the measured descent/step at the finest rung has to sit against
#: the pose's own |grad|_2 to count as a first-order term with no intercept.
#: Not a fitted value: the measured relative spread over the poses the audit
#: reports is 1.7e-06, which is the double-precision noise floor of a 1e-9
#: conf-unit step, and this sits three decades above it. A value jump across a
#: cell face is two to three decades larger than this, which is what lets the
#: same measurement tell the two mechanisms apart.
LINEARITY_TOL = 1e-3


def _descent(ligand, maps, conf, unit, step):
    """``E(conf) - E(conf - step * unit)``, in kcal/mol.

    Positive means the pose can be moved downhill from here. The direction is
    normalised by the caller, so `step` is in conformation units rather than
    being scaled by the gradient magnitude.

    A conformation the engine refuses to score (one driven outside the box)
    is reported as a descent of 0.0, not as a crash: at the coarse rungs a
    rejected conformation is a pose that has left the basin, which is the
    opposite of descending, and a test that raised there would be reporting a
    crash instead of a reading.
    """
    e0 = score_conformation(ligand, maps, conf)[0]
    try:
        moved = score_conformation(ligand, maps, conf - step * unit)[0]
    except ValueError:
        return 0.0
    return float(e0) - float(moved)



@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """Generate the test structures once for the whole session.

    Returns the *directory*; `write_all` reports what it wrote, which is only
    useful for checking that generation actually did something.
    """
    out = tmp_path_factory.mktemp("opendocking-data")
    written = write_all(out)
    assert written, "no test structures were generated"
    assert (out / "receptor.pdbqt").is_file(), "receptor preparation produced nothing"
    return out


@pytest.fixture(scope="session")
def receptor(data_dir) -> Receptor:
    return Receptor.from_pdbqt(data_dir / "receptor.pdbqt")


@pytest.fixture(scope="session")
def maps(receptor):
    return receptor.precalculate(BOX, "vina", 0.5)


@pytest.fixture(scope="session")
def ligand(data_dir) -> Ligand:
    return load_ligand(data_dir / "ligands" / "ibuprofen.pdbqt")


# ---------------------------------------------------------------------------
# Structure preparation
# ---------------------------------------------------------------------------


class TestPreparation:
    def test_receptor_is_a_single_connected_fragment(self):
        """The synthetic receptor must survive fragment selection intact.

        If the generated structure is not one covalent component, `prepare_receptor`
        legitimately keeps only the largest piece and every downstream test is
        quietly scoring against a handful of isolated atoms.
        """
        import tempfile

        from rdkit import Chem

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rec.pdb"
            src.write_text(make_receptor_pdb(), encoding="utf-8")
            mol = Chem.MolFromPDBFile(str(src), sanitize=False, removeHs=False)
            frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
            assert len(frags) == 1, f"receptor fell into {len(frags)} fragments"
            assert mol.GetNumAtoms() == 30
            # A connected 30-atom graph needs at least 29 bonds; the designed
            # structure has exactly 18 ring + 6 C-O + 6 O-H.
            assert mol.GetNumBonds() == 30, "spurious or missing bonds"

    def test_receptor_preparation_keeps_atoms(self):
        import tempfile

        from opendocking.prep import prepare_receptor

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rec.pdb"
            src.write_text(make_receptor_pdb(), encoding="utf-8")
            with warnings.catch_warnings():
                # No fragment should be dropped; say so if one is.
                warnings.simplefilter("error", RuntimeWarning)
                text = prepare_receptor(src)
        n_atoms = sum(1 for line in text.splitlines() if line.startswith("ATOM"))
        assert n_atoms == 30, "the synthetic receptor has 30 atoms"
        assert text.rstrip().endswith("TER")
        # The receptor must be able to donate *and* accept hydrogen bonds, or
        # the donor map is empty and the scoring path is untested.
        types = [line[77:79].strip() for line in text.splitlines() if line.startswith("ATOM")]
        assert "HD" in types, f"no polar hydrogen in the receptor: {set(types)}"
        assert "OA" in types, f"no acceptor oxygen in the receptor: {set(types)}"

    def test_pdbqt_output_is_column_aligned(self, data_dir):
        """Every written ATOM line must be exactly 79 characters.

        PDBQT is a fixed-column format; a one-character slip produces a file
        that still loads but puts the charge in the wrong place, which is the
        kind of bug that silently corrupts every downstream score.
        """
        for line in (data_dir / "ligands" / "ibuprofen.pdbqt").read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                assert len(line) == 79, f"line is {len(line)} chars: {line!r}"
                # Coordinates must parse out of their columns.
                float(line[30:38])
                float(line[38:46])
                float(line[46:54])

    def test_loading_from_text_matches_loading_from_a_file(
        self, data_dir, maps, tmp_path
    ):
        """``from_pdbqt_str`` must parse the text, not open it as a path.

        Regression: the binding handed the text to the file-opening reader, so
        this entry point failed on *every* input -- on Windows with os error
        123 ("the filename ... is incorrect", because a PDBQT is full of
        newlines), on Linux by looking for a file named after the whole
        document. Nothing caught it, because the only callers passed
        deliberately malformed text and a catchable ValueError looks the same
        whether it came from the parser or from a failed open. Asserting the
        successful path is the only thing that distinguishes them.
        """
        text = (data_dir / "ligands" / "ibuprofen.pdbqt").read_text()
        from_text = Ligand.from_pdbqt_str(text)

        copied = tmp_path / "ibuprofen.pdbqt"
        copied.write_text(text, encoding="utf-8")
        from_file = load_ligand(copied)

        assert from_text.num_atoms == from_file.num_atoms
        assert from_text.num_torsions == from_file.num_torsions
        assert from_text.num_dof == from_file.num_dof

        # And it has to be a working ligand, not just a parsed shell: the two
        # must score identically.
        conf = np.zeros(from_text.num_dof)
        a = score_conformation(from_text, maps, conf)[0]
        b = score_conformation(from_file, maps, conf)[0]
        assert a == pytest.approx(b, abs=1e-12), f"{a} vs {b}"

    @pytest.mark.parametrize(
        "name,min_torsions", [("benzene", 0), ("toluene", 0), ("ibuprofen", 4)]
    )
    def test_torsion_counts_are_chemically_sensible(self, data_dir, name, min_torsions):
        lig = load_ligand(data_dir / "ligands" / f"{name}.pdbqt")
        assert lig.num_torsions >= min_torsions, (
            f"{name}: expected at least {min_torsions} rotatable bonds, "
            f"got {lig.num_torsions}"
        )

    def test_hydroxyl_oxygen_is_donor_and_acceptor(self, data_dir):
        """A carboxylic acid oxygen must be both, not just one.

        Typing it as a donor alone loses every hydrogen bond it accepts, which
        is one of the classic AutoDock preparation bugs.
        """
        lig = load_ligand(data_dir / "ligands" / "ibuprofen.pdbqt")
        kinds = lig.atom_kinds
        assert "donoracceptor" in kinds, f"no donor-acceptor oxygen found: {set(kinds)}"
        assert "acceptor" in kinds
        assert "hydrophobic" in kinds

    def test_nonpolar_hydrogens_are_merged(self, data_dir):
        """Non-polar hydrogens must be gone, or the search cost triples."""
        lig = load_ligand(data_dir / "ligands" / "ibuprofen.pdbqt")
        elements, _charges, _coords, _bonds, _names = _prepared(data_dir, "ibuprofen")
        n_h = sum(1 for e in elements if e == "H")
        # Ibuprofen has one acidic OH; only its hydrogen should survive.
        assert n_h <= 1, f"found {n_h} explicit hydrogens: {elements}"
        del lig


def _prepared(data_dir, name):
    from opendocking.prep import prepare_ligand

    return prepare_ligand(data_dir / f"{name}.sdf")


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


class TestGrid:
    def test_dimensions_match_the_box(self, maps):
        assert maps.dims == (37, 37, 37), maps.dims
        assert maps.spacing == 0.5

    def test_memory_estimate_is_close(self, receptor, maps):
        estimated = receptor.estimate_memory_mb(BOX, 0.5)
        assert estimated > 0
        # The estimate and the allocation use the same formula, so they must
        # agree to rounding.
        assert abs(estimated - maps.memory_mb) / maps.memory_mb < 0.01

    def test_map_files_are_written(self, maps, tmp_path):
        maps.write_map_files(tmp_path)
        files = list(tmp_path.glob("*.map"))
        assert len(files) == 40, f"expected 10 element types x 4 slots, got {len(files)}"
        text = files[0].read_text()
        assert "GRID_PARAMETER_FILE" in text
        assert "SPACING" in text

    def test_too_small_box_is_rejected(self, receptor, ligand):
        tiny = GridBox.from_center_size((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        with pytest.raises(ValueError, match="needs at least"):
            dock(ligand, receptor.precalculate(tiny, "vina", 0.5),
                 exhaustiveness=1, num_modes=1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_energy_is_finite_and_scaled(self, ligand, maps):
        conf = np.zeros(ligand.num_dof)
        energy, grad = score_conformation(ligand, maps, conf)
        assert math.isfinite(energy)
        assert np.all(np.isfinite(grad))
        # The pose here is `zeros` -- the ligand at the origin with the identity
        # rotation, which is *inside* the receptor rather than in a pocket. So
        # this is a clash energy and always was; what changed is how big it is.
        # Before the grid's element partition was fixed (defect 190) the zero
        # pose only felt receptor atoms of its own element, so this number sat
        # under 100. Now the ligand feels every receptor atom, and it measures
        # 151.38. The assertion never claimed a docked pose; it claimed "finite
        # and not blown up", and 1e6 is the signature of blown up.
        #
        # So the bound moves to 1e3: 6.6x above the worst value measured here and
        # three orders of magnitude below the failure mode it exists to catch.
        # It is deliberately *not* pinned to 151.38 -- pinning the exact value
        # would make an ordinary tuning change look like an arithmetic bug.
        assert abs(energy) < 1e3, f"implausible energy {energy}"
        # And it should read as the clash it is: a zero pose inside the receptor
        # is repulsive and large. Without this, a number that came back
        # *negative* would pass the check above just as happily.
        assert energy > 0.0, (
            f"the zero pose sits inside the receptor, so it must be repulsive; "
            f"got {energy}, which reads as an attractive energy for a clash"
        )

    def test_analytic_gradient_matches_finite_difference(self, ligand, maps):
        """The gradient is the engine's core optimisation input.

        If it is wrong, L-BFGS still "converges" — to the wrong place — so this
        is checked directly against central differences.
        """
        rng = np.random.default_rng(20240917)
        conf = np.zeros(ligand.num_dof)
        conf[:3] = rng.uniform(-2.0, 2.0, 3)
        conf[3:6] = rng.uniform(-1.0, 1.0, 3)
        conf[6:] = rng.uniform(-np.pi, np.pi, max(0, ligand.num_dof - 6))

        _e, grad = score_conformation(ligand, maps, conf)
        h = 1e-5
        for k in range(ligand.num_dof):
            up = conf.copy()
            down = conf.copy()
            up[k] += h
            down[k] -= h
            e_up, _ = score_conformation(ligand, maps, up)
            e_down, _ = score_conformation(ligand, maps, down)
            numeric = (e_up - e_down) / (2.0 * h)
            assert abs(numeric - grad[k]) < 5e-3, (
                f"degree of freedom {k}: numeric {numeric:.6f} "
                f"vs analytic {grad[k]:.6f}"
            )

    def test_batch_evaluation_is_zero_copy(self, ligand, maps):
        """A batch must give exactly what the single-conformation path gives."""
        rng = np.random.default_rng(7)
        n = 16
        confs = np.zeros((n, ligand.num_dof))
        confs[:, :3] = rng.uniform(-3.0, 3.0, (n, 3))
        confs[:, 3:6] = rng.uniform(-np.pi, np.pi, (n, 3))
        if ligand.num_dof > 6:
            confs[:, 6:] = rng.uniform(-np.pi, np.pi, (n, ligand.num_dof - 6))

        # The default is the CPU path, so this is a bit-for-bit comparison and
        # it stays one on a machine with no GPU at all.
        batch = evaluate_conformations(ligand, maps, confs)
        assert batch.shape == (n,)
        for i in range(n):
            single, _ = score_conformation(ligand, maps, confs[i])
            assert abs(batch[i] - single) < 1e-9, (
                f"conformation {i}: batch {batch[i]} vs single {single}"
            )

    def test_the_gpu_batch_path_agrees_and_reports_itself(self, ligand, maps):
        """Opting into the GPU must be explicit, and must be close.

        The kernel accumulates in single precision, so it agrees with the CPU
        path to single precision and not bit-for-bit. That is why the default
        is off: a default that silently changed the last digits of a result
        depending on the machine would make runs irreproducible across
        hardware.
        """
        rng = np.random.default_rng(11)
        n = 32
        confs = np.zeros((n, ligand.num_dof))
        confs[:, :3] = rng.uniform(-3.0, 3.0, (n, 3))
        confs[:, 3:6] = rng.uniform(-np.pi, np.pi, (n, 3))
        if ligand.num_dof > 6:
            confs[:, 6:] = rng.uniform(-np.pi, np.pi, (n, ligand.num_dof - 6))

        cpu = evaluate_conformations(ligand, maps, confs, use_gpu=False)
        report: dict = {}
        gpu = evaluate_conformations(
            ligand, maps, confs, use_gpu=True, report_backend=report
        )
        assert set(report) >= {"backend", "adapter", "gpu_skip_reason"}
        assert report["backend"] in ("cpu", "gpu")
        assert report["num_conformations"] == n

        # The tolerance has to be relative, and that is not a detail. These
        # conformations are placed at random, so a good share of them sit
        # outside the box and score in the hundreds -- where a single ulp of a
        # single-precision number is already around 1e-4. A fixed absolute
        # threshold below that would be smaller than the arithmetic it is
        # meant to check, so it would pass on the machine it was written on
        # and fail on every other shader compiler. Measured worst case is
        # about one f32 ulp of the score, so the meaningful criterion is the
        # error relative to the size of the score.
        scale = max(1.0, float(np.max(np.abs(cpu))))
        worst = float(np.max(np.abs(gpu - cpu)))
        assert worst <= 1e-5 * scale, (
            f"GPU and CPU disagree by {worst:g} absolute "
            f"({worst / scale:g} relative) on scores of magnitude {scale:g}"
        )
        if report["backend"] == "gpu":
            assert report["gpu_skip_reason"] is None
            assert report["adapter"]

    def test_batch_rejects_wrong_shape(self, ligand, maps):
        with pytest.raises(ValueError, match="shape"):
            evaluate_conformations(ligand, maps, np.zeros((4, ligand.num_dof + 3)))

    def test_vinardo_runs(self, receptor, ligand):
        vinardo_maps = receptor.precalculate(BOX, "vinardo", 0.5)
        result = dock(ligand, vinardo_maps, exhaustiveness=1, num_modes=1)
        assert math.isfinite(result.best_energy)


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------


class TestDocking:
    def test_a_donor_acceptor_ligand_is_actually_rewarded(self, receptor, data_dir):
        """The hydrogen-bond term has to reach a real contact distance.

        Regression guard. Every interaction radius used to be a flat 0.4 Å, so
        the ``hb`` window (`d ≤ -0.5`) was only reachable at a heavy-atom
        separation of 0.3 Å. The term was therefore exactly zero at every
        distance a hydrogen bond can form, the 18 Å box around the crown ether
        returned poses scoring -0.00, and the "no effective repulsion" problem
        that was later patched over with a clash filter was really this.

        Phenol has one donor and the receptor is all acceptors, so if the polar
        term is alive this pose must be worth something well beyond rounding
        noise. Asserted as a magnitude, not as a memorised value.
        """
        phenol = load_ligand(data_dir / "ligands" / "phenol.pdbqt")
        tight = GridBox.from_center_size((0.0, 0.0, 0.0), (14.0, 14.0, 14.0))
        maps = receptor.precalculate(tight, "vina", 0.375)
        result = dock(phenol, maps, exhaustiveness=8, num_modes=3, seed=11)
        assert result.rejected_pose_count == 0
        assert (
            result.best_energy < -0.1
        ), f"a donor-acceptor contact must be rewarded, got {result.best_energy}"

    def test_solid_matter_is_repulsive_and_a_fallback_says_so(self, data_dir):
        """The steric wall must stop a ligand being buried in the receptor.

        The other half of the same regression: with the radii wrong, burying a
        ligand was the *cheapest* thing the scorer could do, which is why
        clashing poses used to outrank clean ones by a full kcal/mol.

        The receptor is a lattice with no void wide enough for an atom, so a
        physically possible pose does not exist. That makes the clash-reporting
        path reachable for a structural reason rather than by hoping a weak
        search happens to bury the ligand — and it keeps this test from going
        vacuous, because it asserts the warning actually fires.
        """
        from opendocking.tests.make_data import make_solid_receptor_pdbqt

        solid = Receptor.from_pdbqt_str(make_solid_receptor_pdbqt())
        benzene = load_ligand(data_dir / "ligands" / "benzene.pdbqt")
        box = GridBox.from_center_size((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        maps = solid.precalculate(box, "vina", 0.5)
        result = dock(benzene, maps, exhaustiveness=2, num_modes=3, seed=3)

        assert result.best_energy > 0.0, (
            "burying a ligand in solid matter must be repulsive, not attractive; "
            f"got {result.best_energy}"
        )
        assert result.rejected_pose_count == result.num_poses > 0
        assert "WARNING" in result.summary()

    def test_returned_poses_are_not_minima(self, ligand, maps):
        """The poses are **not** minima, measured at a step that can see it.

        This test used to be `test_returned_poses_are_actual_minima` and
        asserted the opposite. It was green, and it was green for a reason that
        had nothing to do with the poses: it asked for descent at steps of 1e-4
        and above, and these poses' descent regions are narrower than 1e-4, so
        every probe landed past the turn-up and read zero. "No descent found"
        and "no descent there" are different readings, and only one of them was
        being asserted -- by a test whose name claimed the stronger one.

        Its docstring carried the same error one level up. The zeros were
        explained by the C0 nature of the interpolated field: a pose sitting on
        a cell face carries two large opposite-signed one-sided gradients, so a
        gradient-norm threshold cannot tell a genuine local minimum from a
        failed optimisation, and "no downhill step exists" can. That is true of
        the gradient and irrelevant to the zeros -- and it has the C0 claim
        backwards, since C0 is continuity of *value*: the value does not jump
        across a face, the one-sided derivative does. The measurement below is
        the one that decides the question, and it says these poses are not
        stationary points.

        What is asserted, in the order the argument needs it:

        * `|grad|_2` is above the optimiser's own declared tolerance, which is
          a statement about the engine and does not depend on any step;
        * the descent at 1e-9 conf units is strictly positive, so the pose can
          be moved downhill -- the opposite of the old test's claim, and
          measured rather than inferred from the gradient;
        * that descent equals `|grad|_2 * step` to within `LINEARITY_TOL`,
          which is one measurement carrying two claims: the slope is the
          pose's own gradient, and the intercept is zero. The second is what
          a field whose *value* jumped could not produce at any step, and it
          is the direct replacement for the mechanism the old docstring gave.
        """
        result = dock(ligand, maps, exhaustiveness=4, num_modes=3, seed=21)
        assert result.num_poses >= 1
        for i in range(result.num_poses):
            conf = np.asarray(result.pose_conformation(i), dtype=np.float64)
            _e0, grad = score_conformation(ligand, maps, conf)
            gnorm2 = float(np.linalg.norm(grad))
            assert gnorm2 > GRADIENT_TOLERANCE, (
                f"pose {i} has |grad|2 {gnorm2:.3e}, at or below the optimiser's "
                f"own declared {GRADIENT_TOLERANCE:.0e}: it is a stationary "
                f"point and this test's premise no longer holds"
            )
            unit = np.asarray(grad, dtype=np.float64) / max(gnorm2, 1e-30)
            descent = _descent(ligand, maps, conf, unit, FINEST_STEP)
            assert descent > 0.0, (
                f"pose {i} cannot descend by {FINEST_STEP:.0e} conf units along "
                f"-|grad| even though |grad|2 is {gnorm2:.3e}; the two "
                f"measurements disagree and the fine one has stopped resolving "
                f"the descent region"
            )
            predicted = gnorm2 * FINEST_STEP
            assert abs(descent - predicted) <= LINEARITY_TOL * predicted, (
                f"pose {i}: descent {descent:+.4e} at {FINEST_STEP:.0e} against "
                f"|grad|2 * step {predicted:+.4e}. A slope that is not the "
                f"pose's own |grad|2, or an offset between them, is a value "
                f"jump across a cell face rather than a first-order term"
            )

    def test_a_coarse_line_search_steps_over_the_descent_region(self, ligand, maps):
        """The old "no descent" reading is a fact about the probe, and is measured.

        The claim the old test made -- no descent at 1e-4 through 0.3 -- is
        still true of these poses, and it is still not evidence that they are
        minima. This test keeps the claim and names what it is: a statement
        about a step that is wider than the region it is asking about.

        Both halves are measured, and both are needed. "No descent at any coarse
        step" alone is what the old test asserted, and it is satisfied by any
        pose at all whenever the probe is coarse enough. "The region closes at
        or before the coarsest probe step" is the part that ties the zero to a
        width, and it is what makes the zero interpretable: the field turns up
        before the probe arrives, so the probe reports the turn-up and not the
        absence of a descent.

        The bracket is a bracket, not a measurement to more digits than the
        ladder has -- the descent is known to be positive at one rung and not at
        the next, and nothing between them was sampled. So the assertion is on
        the rung that closes the region, and it is the coarse probe step that
        is compared against it, not the geometric mean the audit reports.
        """
        result = dock(ligand, maps, exhaustiveness=4, num_modes=3, seed=21)
        assert result.num_poses >= 1
        probe = COARSE_STEPS[0]
        for i in range(result.num_poses):
            conf = np.asarray(result.pose_conformation(i), dtype=np.float64)
            _e0, grad = score_conformation(ligand, maps, conf)
            unit = np.asarray(grad, dtype=np.float64) / max(
                float(np.linalg.norm(grad)), 1e-30
            )
            for step in COARSE_STEPS:
                descent = _descent(ligand, maps, conf, unit, step)
                assert descent <= 0.0, (
                    f"pose {i} descends by {descent:+.6f} kcal/mol at step "
                    f"{step}: the coarse probe is inside its descent region, so "
                    f"'no descent at these steps' is no longer true of it"
                )
            ladder = [
                (step, _descent(ligand, maps, conf, unit, step))
                for step in LADDER_STEPS
            ]
            first_flat = next(
                (n for n, (_s, descent) in enumerate(ladder) if descent <= 0.0),
                None,
            )
            assert first_flat is not None, (
                f"pose {i} descends at every rung of the ladder up to "
                f"{LADDER_STEPS[-1]:.0e}, so its region is wider than the probe "
                f"and the coarse zeros are not explained by the probe being too "
                f"coarse"
            )
            closes_at = ladder[first_flat][0]
            opens_at = ladder[first_flat - 1][0] if first_flat else 0.0
            assert closes_at <= probe, (
                f"pose {i}'s descent region closes between {opens_at:.0e} and "
                f"{closes_at:.0e}, which is wider than the {probe:.0e} probe "
                f"step -- so the coarse steps should have found the descent"
            )

    def test_finds_a_negative_energy_pose(self, ligand, maps):
        result = dock(ligand, maps, exhaustiveness=2, num_modes=3, seed=1)
        assert result.num_poses >= 1
        assert result.best_energy < -1.0, (
            f"expected a clearly favourable pose, got {result.best_energy}"
        )
        assert result.scoring_function == "vina"
        assert result.elapsed_seconds > 0

    def test_poses_are_inside_the_box(self, ligand, maps):
        """A pose outside the box means the out-of-box penalty failed."""
        result = dock(ligand, maps, exhaustiveness=2, num_modes=3, seed=2)
        for i in range(result.num_poses):
            coords = result.pose_coords(i)
            for p in coords:
                assert BOX.contains(p), f"pose {i} atom at {p} is outside the box"

    def test_poses_are_distinct(self, ligand, maps):
        """Clustering must not report ten copies of one basin as ten modes."""
        result = dock(ligand, maps, exhaustiveness=4, num_modes=5, rmsd_cutoff=1.0, seed=3)
        allc = result.all_pose_coords()
        assert allc.shape[0] >= 1
        for i in range(allc.shape[0]):
            for j in range(i + 1, allc.shape[0]):
                rmsd = np.sqrt(((allc[i] - allc[j]) ** 2).sum(axis=1).mean())
                assert rmsd >= 1.0 - 1e-6, f"poses {i} and {j} are {rmsd:.2f} Å apart"

    def test_energies_are_sorted(self, ligand, maps):
        result = dock(ligand, maps, exhaustiveness=2, num_modes=4, seed=4)
        energies = result.energies
        assert np.all(np.diff(energies) >= -1e-9), energies

    def test_seed_is_reproducible(self, ligand, maps):
        a = dock(ligand, maps, exhaustiveness=2, num_modes=3, seed=42)
        b = dock(ligand, maps, exhaustiveness=2, num_modes=3, seed=42)
        assert abs(a.best_energy - b.best_energy) < 1e-9
        assert np.allclose(a.pose_coords(0), b.pose_coords(0))

    def test_higher_exhaustiveness_does_not_worsen(self, receptor, data_dir):
        """More search effort must not return a worse *valid* pose.

        The qualifier matters, and so does the ligand. A weak search can find
        only poses that overlap the receptor, and those are dropped in favour
        of physically possible ones — which can leave the low-exhaustiveness
        run reporting a worse number than it otherwise would. Comparing a
        fallback against a clean result compares two different things.

        A small rigid ligand is used so that even the weak run has somewhere
        valid to go, which makes the premise true rather than conditional.
        """
        small = load_ligand(data_dir / "ligands" / "benzene.pdbqt")
        maps = receptor.precalculate(BOX, "vina", 0.5)
        low = dock(small, maps, exhaustiveness=2, num_modes=1, seed=5)
        high = dock(small, maps, exhaustiveness=8, num_modes=1, seed=5)
        assert low.rejected_pose_count == 0, "the weak run fell back to clashing poses"
        assert high.rejected_pose_count == 0
        assert high.best_energy <= low.best_energy + 1e-6

    def test_rigid_ligand_docks(self, data_dir, maps):
        benzene = load_ligand(data_dir / "ligands" / "benzene.pdbqt")
        assert benzene.num_torsions == 0
        result = dock(benzene, maps, exhaustiveness=2, num_modes=2, seed=6)
        assert result.best_energy < 0.0
        assert result.pose_coords(0).shape == (benzene.num_atoms, 3)

    def test_lga_mode_runs(self, ligand, maps):
        result = dock(ligand, maps, exhaustiveness=1, num_modes=2, seed=7, mode="lga")
        assert result.num_poses >= 1
        assert math.isfinite(result.best_energy)

    def test_invalid_parameters_are_rejected(self, ligand, maps):
        with pytest.raises(ValueError):
            dock(ligand, maps, exhaustiveness=0)
        with pytest.raises(ValueError):
            dock(ligand, maps, num_modes=0)
        with pytest.raises(ValueError):
            dock(ligand, maps, rmsd_cutoff=0.0)

    def test_writes_pdbqt_and_xyz(self, ligand, maps, tmp_path):
        result = dock(ligand, maps, exhaustiveness=1, num_modes=2, seed=8)
        pdbqt = tmp_path / "out.pdbqt"
        result.write_pdbqt(pdbqt)
        text = pdbqt.read_text()
        assert text.count("MODEL") >= 2
        assert "VINA RESULT" in text

        xyz = tmp_path / "out.xyz"
        result.write_xyz(xyz)
        xyz_text = xyz.read_text()
        assert xyz_text.count("kcal/mol") == result.num_poses
        # The first line of each frame declares the atom count.
        first = xyz_text.splitlines()[0].strip()
        assert int(first) == ligand.num_atoms

    def test_round_trips_through_pdbqt(self, ligand, maps, tmp_path):
        """A written pose must reload with the same coordinates and energy."""
        result = dock(ligand, maps, exhaustiveness=1, num_modes=1, seed=9)
        path = tmp_path / "round.pdbqt"
        result.write_pdbqt(path)
        reloaded = Ligand.from_pdbqt(path)
        assert reloaded.num_atoms == ligand.num_atoms

        from opendocking.workbench import _parse_pdbqt_atoms

        coords, _ = _parse_pdbqt_atoms(path.read_text())
        original = result.pose_coords(0)
        assert coords.shape == original.shape
        # PDBQT holds 3 decimals, so allow half a milliångström.
        assert np.abs(coords - original).max() < 5e-4


# ---------------------------------------------------------------------------
# Capability reporting
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_gpu_status_is_a_dict(self):
        status = gpu_status()
        assert set(status) == {"compiled", "available"}
        assert isinstance(status["compiled"], bool)

    def test_engine_version(self):
        import opendocking

        assert opendocking.core.engine_version() == opendocking.__version__

    def test_python_wrappers_only_use_names_the_extension_exports(self):
        """A typo between the two layers is a runtime error, not a compile error.

        `core.py` is a thin wrapper over the compiled module, and nothing but
        the interpreter checks the two sides agree. One mismatch is enough to
        kill an otherwise fine run on a property access — which is exactly what
        `rmds` versus `rmsds` did.
        """
        import inspect
        import re

        from opendocking import _dockpy
        from opendocking import core as core_mod

        def check(wrapper, rust_class, attr):
            exported = {n for n in dir(rust_class) if not n.startswith("_")}
            used = set(re.findall(rf"{re.escape(attr)}\.(\w+)", inspect.getsource(wrapper)))
            # Methods defined on the wrapper itself are fine; only the raw
            # object accesses are checked.
            missing = {n for n in used if n not in exported}
            assert not missing, (
                f"{wrapper.__name__} reads {sorted(missing)} from "
                f"{rust_class.__name__}, which the extension does not export"
            )

        check(core_mod.DockingResult, _dockpy.DockingResults, "_res")
        check(core_mod.Ligand, _dockpy.Ligand, "_lig")
        check(core_mod.GridMaps, _dockpy.GridMaps, "_maps")

    def test_docking_result_exposes_every_documented_property(self):
        from opendocking import core as core_mod

        for name in (
            "num_poses",
            "energies",
            "best_energy",
            "intermolecular_energies",
            "rmsds",
            "elapsed_seconds",
            "raw_pose_count",
            "scoring_function",
            "pose_coords",
            "all_pose_coords",
            "write_pdbqt",
            "write_xyz",
            "summary",
        ):
            assert hasattr(core_mod.DockingResult, name), f"DockingResult lost {name}"

    def test_docking_result_values_are_readable(self, ligand, maps):
        """The regression guard for the `rmds`/`rmsds` typo."""
        result = dock(ligand, maps, exhaustiveness=1, num_modes=2, seed=11)
        assert isinstance(result.rmsds, np.ndarray)
        assert result.rmsds.shape == (result.num_poses,)
        assert np.all(np.isfinite(result.rmsds))
        # The first pose is the reference, so it cannot differ from itself.
        assert abs(result.rmsds[0]) < 1e-9
        assert result.summary()
