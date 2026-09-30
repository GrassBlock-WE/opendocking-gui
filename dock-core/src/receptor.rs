//! The receptor: a rigid structure that maps are precalculated against.

use std::path::Path;

use serde::Serialize;

use crate::error::Result;
use crate::grid::{GridBox, GridMaps};
use crate::pdbqt::read_pdbqt;
use crate::scoring::ScoringFunction;
use crate::types::{Molecule, Vec3};

/// A rigid receptor prepared for docking.
#[derive(Debug, Clone, Serialize)]
pub struct Receptor {
    /// Atoms and covalent graph, in file order.
    pub molecule: Molecule,
    /// Number of atoms.
    pub num_atoms: usize,
    /// Number of polar hydrogens, a common quality indicator after preparation.
    pub num_polar_hydrogens: usize,
}

impl Receptor {
    /// Build a receptor from a molecule.
    pub fn from_molecule(molecule: Molecule) -> Result<Receptor> {
        if molecule.is_empty() {
            return Err(crate::DockError::molecule("receptor contains no atoms"));
        }
        let num_atoms = molecule.len();
        let num_polar_hydrogens = molecule
            .atoms
            .iter()
            .filter(|a| a.element == crate::types::Element::H)
            .count();
        Ok(Receptor {
            molecule,
            num_atoms,
            num_polar_hydrogens,
        })
    }

    /// Read a receptor from a PDBQT file.
    pub fn from_pdbqt(path: impl AsRef<Path>) -> Result<Receptor> {
        let parsed = read_pdbqt(path)?;
        Receptor::from_molecule(parsed.molecule)
    }

    /// Read a receptor from PDBQT text.
    pub fn from_pdbqt_str(text: &str) -> Result<Receptor> {
        let parsed = crate::pdbqt::parse_pdbqt(text)?;
        Receptor::from_molecule(parsed.molecule)
    }

    /// Geometric centre of all atoms.
    pub fn centroid(&self) -> Vec3 {
        self.molecule.centroid()
    }

    /// Axis-aligned bounding box as `(min, max)`.
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        self.molecule.bounding_box()
    }

    /// Precalculate the affinity maps for a search box.
    ///
    /// `spacing` of exactly `0.0` means "use [`crate::grid::DEFAULT_SPACING`]".
    /// A *negative* spacing is a mistake, not a request for the default, and
    /// is rejected: silently turning a typo into a different grid would give
    /// results nobody asked for.
    pub fn precalculate(
        &self,
        box_: &GridBox,
        scoring: &dyn ScoringFunction,
        spacing: f64,
    ) -> Result<GridMaps> {
        if spacing.is_nan() || spacing < 0.0 {
            return Err(crate::DockError::param(
                "spacing",
                spacing,
                "must be zero (meaning the default) or a positive value",
            ));
        }
        let spacing = if spacing == 0.0 {
            crate::grid::DEFAULT_SPACING
        } else {
            spacing
        };
        GridMaps::precalculate(&self.molecule, box_, scoring, spacing, 0)
    }

    /// Number of atoms whose PDBQT type this engine does not recognise.
    ///
    /// An unrecognised type is not an error — another tool may emit types
    /// AutoDock never defined, and refusing the file would make the engine
    /// unusable with them. It does mean the atom contributes only its shape
    /// term, losing its hydrogen-bond and hydrophobic character, which is a
    /// silent degradation. Reporting the count lets a caller notice.
    pub fn unknown_atom_types(&self) -> usize {
        self.molecule
            .atoms
            .iter()
            .filter(|a| a.atom_type == crate::types::AtomType::Unknown)
            .count()
    }

    /// A box of the given size centred on `center`.
    pub fn box_centered(&self, center: Vec3, size: Vec3) -> Result<GridBox> {
        GridBox::centered(center, size)
    }

    /// A box that tightly encloses all receptor atoms, padded by `padding`.
    ///
    /// A docking box must contain a *binding site*, not the whole protein;
    /// this helper is meant for small receptors and for tests, and the CLI
    /// offers explicit `--center_x/--size_x` arguments for real use.
    pub fn box_around_all(&self, padding: f64) -> Result<GridBox> {
        let (lo, hi) = self.bounding_box();
        GridBox::new(
            [lo[0] - padding, lo[1] - padding, lo[2] - padding],
            [hi[0] + padding, hi[1] + padding, hi[2] + padding],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdbqt::parse_pdbqt;

    const RECEPTOR: &str = concat!(
        "ATOM      1  C   ALA A   1      -1.204  14.120   1.234  1.00  0.00     0.000 C \n",
        "ATOM      2  O   ALA A   1      -0.100  13.500   1.900  1.00  0.00    -0.500 OA \n",
        "ATOM      3  N   ALA A   1      -2.300  13.800   1.000  1.00  0.00    -0.300 N \n",
        "ATOM      4  C   GLY A   2      -2.000  15.000   3.100  1.00  0.00     0.100 C \n",
    );

    #[test]
    fn reads_a_receptor() {
        let r = Receptor::from_pdbqt_str(RECEPTOR).unwrap();
        assert_eq!(r.num_atoms, 4);
        assert_eq!(r.num_polar_hydrogens, 0);
    }

    #[test]
    fn precalculates_maps() {
        let r = Receptor::from_pdbqt_str(RECEPTOR).unwrap();
        let b = r.box_around_all(4.0).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        let maps = r.precalculate(&b, &scoring, 0.5).unwrap();
        assert!(maps.point_count() > 0);
        assert!(maps.dims()[0] > 2);
    }

    #[test]
    fn empty_receptor_is_rejected() {
        let err = Receptor::from_molecule(Molecule::default()).unwrap_err();
        assert!(matches!(err, crate::DockError::InvalidMolecule(_)));
    }

    #[test]
    fn a_negative_spacing_is_rejected_rather_than_replaced() {
        // Regression: `spacing <= 0.0` used to fall back to the default, so a
        // typo silently became a different grid. Zero is a genuine "use the
        // default" request; a negative number is not.
        let r = Receptor::from_pdbqt_str(RECEPTOR).unwrap();
        let b = GridBox::centered([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        assert!(r.precalculate(&b, &scoring, -0.5).is_err());
        assert!(r.precalculate(&b, &scoring, f64::NAN).is_err());
        let maps = r
            .precalculate(&b, &scoring, 0.0)
            .expect("zero means the default");
        assert!((maps.spacing[0] - crate::grid::DEFAULT_SPACING).abs() < 1e-12);
    }

    #[test]
    fn an_absurd_spacing_is_a_parameter_error_not_an_abort() {
        // Regression: 1e-9 Ångström passes the `(0, 1]` range check but asks
        // for a 2e10-point axis. The product then overflowed and `vec!` aborted
        // on capacity overflow — and because the release profile is
        // `panic = "abort"`, that killed the *host process*, taking a Python
        // interpreter down with it instead of raising.
        let r = Receptor::from_pdbqt_str(RECEPTOR).unwrap();
        let b = GridBox::centered([0.0, 0.0, 0.0], [20.0, 20.0, 20.0]).unwrap();
        let scoring = crate::scoring::VinaScoring::new();
        let err = r
            .precalculate(&b, &scoring, 1e-9)
            .expect_err("a 1e-9 Ångström grid must be refused, not attempted");
        assert!(err.to_string().contains("spacing"), "message: {err}");
    }

    #[test]
    fn a_non_finite_box_corner_is_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                GridBox::new([bad, 0.0, 0.0], [1.0, 1.0, 1.0]).is_err(),
                "a {bad} min corner must be refused"
            );
            assert!(
                GridBox::new([0.0, 0.0, 0.0], [1.0, 1.0, bad]).is_err(),
                "a {bad} max corner must be refused"
            );
        }
    }

    #[test]
    fn unrecognised_atom_types_are_counted_not_hidden() {
        let with_exotic = format!(
            "{RECEPTOR}ATOM      5  X1  ALA A   3       4.000  15.000   3.100  1.00  0.00     0.000 ZZ\n"
        );
        let r = Receptor::from_pdbqt_str(&with_exotic).unwrap();
        assert_eq!(r.num_atoms, 5);
        assert_eq!(r.unknown_atom_types(), 1, "the exotic type must be visible");
        assert_eq!(
            Receptor::from_pdbqt_str(RECEPTOR)
                .unwrap()
                .unknown_atom_types(),
            0
        );
    }

    #[test]
    fn centroid_is_mean_of_atoms() {
        let mol = parse_pdbqt(RECEPTOR).unwrap().molecule;
        let rec = Receptor::from_molecule(mol).unwrap();
        let c = rec.centroid();
        assert!((c[0] - (-1.401)).abs() < 1e-3, "{c:?}");
    }
}
