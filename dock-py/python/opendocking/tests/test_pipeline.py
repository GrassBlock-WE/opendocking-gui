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

    def test_returned_poses_are_actual_minima(self, ligand, maps):
        """A reported pose must be a stationary point, checked by line search.

        Not by the size of its gradient. The maps are trilinearly interpolated,
        which is only C0 across a cell face, so a pose with an atom sitting on a
        face carries two large, opposite-signed one-sided gradients while still
        being a genuine local minimum. A gradient-norm threshold cannot tell
        that apart from an optimiser that failed to converge; "no downhill step
        exists" can, and it is the property that actually matters.
        """
        result = dock(ligand, maps, exhaustiveness=4, num_modes=3, seed=21)
        assert result.num_poses >= 1
        for i in range(result.num_poses):
            conf = np.asarray(result.pose_conformation(i), dtype=np.float64)
            _, grad = score_conformation(ligand, maps, conf)
            unit = grad / max(float(np.linalg.norm(grad)), 1e-30)
            e0 = score_conformation(ligand, maps, conf)[0]
            for step in (1e-4, 1e-3, 0.01, 0.1, 0.3):
                moved = score_conformation(ligand, maps, conf - step * unit)[0]
                assert moved > e0 - 1e-6, (
                    f"pose {i} can still descend by {e0 - moved:+.6f} kcal/mol "
                    f"at step {step}: it is not a minimum"
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
