//! A docking ligand, prepared for repeated scoring.
//!
//! Everything that depends only on the ligand — the covalent graph, the
//! rotatable-bond tree, the per-atom grid lookup table, and the intramolecular
//! partner list — is computed once here, so the inner loop of the search does
//! no setup work at all.

use serde::Serialize;

use crate::error::{DockError, Result};
use crate::grid::MAPS_PER_TYPE;
use crate::kinematics::KinematicTree;
use crate::pdbqt::{read_pdbqt, ParsedStructure};
use crate::scoring::weights_for_kind;
use crate::types::{dist3, grid_type_index, Atom, Molecule, Vec3};

/// Intramolecular interactions closer than this many bonds are excluded.
///
/// AutoDock omits the 1-2, 1-3 and 1-4 covalent contacts because they are
/// already accounted for by the bonded terms; scoring them again would let a
/// torsion collapse. This matches Vina's exclusion rule.
pub const MIN_INTRA_BOND_DISTANCE: usize = 4;

/// Two atoms of one ligand may not be closer than this, in ångström.
///
/// The shortest real contact in any molecule is an H–H bond at about 0.74 Å,
/// so 0.5 Å cannot reject a genuine structure; it only catches two atoms
/// sitting on the same coordinate, which is what a truncated or mis-prepared
/// file looks like. Such a ligand used to be accepted silently: the pair is
/// either a bonded neighbour (excluded from the intramolecular term) or gets
/// a hard-coded short-range clamp, so every number came out finite and the
/// failure was invisible. A docking run then returned plausible-looking poses
/// built from geometry that does not exist.
pub const MIN_ATOM_SEPARATION: f64 = 0.5;

/// A prepared docking ligand.
#[derive(Debug, Clone, Serialize)]
pub struct Ligand {
    /// Atoms and covalent graph, in file order.
    pub molecule: Molecule,
    /// The rotatable-bond tree.
    pub tree: KinematicTree,
    /// Dense element index per atom, for the grid.
    pub type_index: Vec<usize>,
    /// Per-atom map weights encoding its interaction class.
    pub weights: Vec<[f32; MAPS_PER_TYPE]>,
    /// Flat list of intramolecular partner pairs, stored as `i * n + j`.
    intra_pairs: Vec<(usize, usize)>,
    /// Number of rotatable degrees of freedom.
    pub num_torsions: usize,
}

impl Ligand {
    /// Prepare a ligand from an already-parsed molecule.
    ///
    /// `declared_torsions` should carry the `TORSION` records from the input
    /// file when present, so the ligand docks with exactly the torsions its
    /// preparer intended.
    pub fn from_molecule_with(
        molecule: Molecule,
        declared_torsions: &[(usize, usize)],
    ) -> Result<Ligand> {
        if molecule.is_empty() {
            return Err(DockError::molecule("ligand contains no atoms"));
        }
        if let Some((i, j, d)) = closest_atom_pair(&molecule) {
            return Err(DockError::molecule(format!(
                "atoms {i} ({}) and {j} ({}) are {d:.3} Å apart, below the \
                 {MIN_ATOM_SEPARATION:.1} Å floor: this is a duplicate or a \
                 broken structure, and scoring it would return finite but \
                 meaningless energies",
                molecule.atoms[i].name, molecule.atoms[j].name,
            )));
        }
        let mut mol = molecule;
        // Re-derive the interaction classes even if the caller built the
        // molecule by hand, so weights always match the current rules.
        mol.assign_vina_atom_kinds();

        let tree = KinematicTree::from_molecule(&mol, declared_torsions)?;
        let type_index = mol
            .atoms
            .iter()
            .map(|a| grid_type_index(a.element))
            .collect();
        let weights: Vec<[f32; crate::grid::MAPS_PER_TYPE]> =
            mol.atoms.iter().map(|a| weights_for_kind(a.kind)).collect();
        let intra_pairs = collect_intramolecular_pairs(&mol, MIN_INTRA_BOND_DISTANCE);

        Ok(Ligand {
            num_torsions: tree.num_torsions(),
            molecule: mol,
            tree,
            type_index,
            weights,
            intra_pairs,
        })
    }

    /// Prepare a ligand, perceiving torsions from the covalent graph.
    pub fn from_molecule(molecule: Molecule) -> Result<Ligand> {
        Ligand::from_molecule_with(molecule, &[])
    }

    /// Prepare a ligand by reading a PDBQT file.
    pub fn from_pdbqt(path: impl AsRef<std::path::Path>) -> Result<Ligand> {
        let parsed: ParsedStructure = read_pdbqt(path)?;
        let torsions = parsed.active_torsion_pairs();
        Ligand::from_molecule_with(parsed.molecule, &torsions)
    }

    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.molecule.len()
    }

    /// True if the ligand has no atoms.
    pub fn is_empty(&self) -> bool {
        self.molecule.is_empty()
    }

    /// Number of rotatable bonds.
    pub fn num_torsions(&self) -> usize {
        self.num_torsions
    }

    /// Total number of rotatable degrees of freedom including the rigid body.
    pub fn ndof(&self) -> usize {
        6 + self.num_torsions
    }

    /// Intramolecular partner pairs (both indices, `i < j`).
    pub fn intramolecular_pairs(&self) -> &[(usize, usize)] {
        &self.intra_pairs
    }

    /// Atoms whose positions follow the given cluster.
    pub fn atom(&self, i: usize) -> &Atom {
        &self.molecule.atoms[i]
    }

    /// Axis-aligned extent of the reference conformation.
    pub fn extent(&self) -> (Vec3, Vec3) {
        self.molecule.bounding_box()
    }

    /// Largest distance from the centroid to any atom — a useful lower bound
    /// on the grid box half-size needed to contain the ligand.
    pub fn radius(&self) -> f64 {
        let c = self.molecule.centroid();
        self.molecule
            .atoms
            .iter()
            .map(|a| {
                let dx = a.coord[0] - c[0];
                let dy = a.coord[1] - c[1];
                let dz = a.coord[2] - c[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(0.0f64, f64::max)
    }

    /// The `(proximal, distal)` atom-index pairs of the rotatable bonds, for
    /// writing a PDBQT `BRANCH` tree.
    pub fn torsion_pairs(&self) -> Vec<(usize, usize)> {
        self.tree
            .torsions
            .iter()
            .map(|t| (t.proximal_atom, t.distal_atom))
            .collect()
    }
}

/// The closest pair of atoms closer than [`MIN_ATOM_SEPARATION`], if any.
///
/// Returns `(i, j, distance)`. Ligands are tens of atoms, so the quadratic
/// scan is free next to the grid precalculation that follows it, and it runs
/// once per ligand rather than per conformation.
fn closest_atom_pair(mol: &Molecule) -> Option<(usize, usize, f64)> {
    let mut worst: Option<(usize, usize, f64)> = None;
    for i in 0..mol.len() {
        for j in (i + 1)..mol.len() {
            let d = dist3(mol.atoms[i].coord, mol.atoms[j].coord);
            // `map_or(true, ..)` rather than `is_none_or`: the latter is stable
            // from 1.82 and this crate's MSRV is 1.75.
            if d < MIN_ATOM_SEPARATION && worst.map_or(true, |(_, _, w)| d < w) {
                worst = Some((i, j, d));
            }
        }
    }
    worst
}

/// All atom pairs at least `min_dist` bonds apart, stored once each.
///
/// Vina reaches the same total by tabulating a per-conformation ligand grid
/// and summing the rotating branch against the rest; summing the pair list
/// directly gives the identical result without the interpolation error.
fn collect_intramolecular_pairs(mol: &Molecule, min_dist: usize) -> Vec<(usize, usize)> {
    let n = mol.len();
    let mut out = Vec::new();
    let mut distance = vec![usize::MAX; n];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    for start in 0..n {
        distance.iter_mut().for_each(|d| *d = usize::MAX);
        distance[start] = 0;
        queue.clear();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            if distance[u] >= min_dist {
                break; // everything still in the queue is also far enough
            }
            for &v in &mol.neighbors[u] {
                if distance[v] == usize::MAX {
                    distance[v] = distance[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        for other in (start + 1)..n {
            if distance[other] >= min_dist {
                out.push((start, other));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Atom, AtomKind, AtomType, Element};

    /// A four-carbon chain: 1 rotatable bond, and two 1-4 pairs.
    fn butane() -> Molecule {
        let c = |i: u32, p: Vec3| Atom::new(i, p, Element::C, AtomType::CH);
        Molecule::from_atoms(vec![
            c(1, [0.000, 0.000, 0.000]),
            c(2, [1.500, 0.000, 0.000]),
            c(3, [2.150, 1.300, 0.000]),
            c(4, [3.650, 1.300, 0.000]),
        ])
        .unwrap()
    }

    /// Two atoms placed on the same coordinate, bonded neighbours.
    fn coincident_neighbours() -> Molecule {
        let c = |i: u32, p: Vec3| Atom::new(i, p, Element::C, AtomType::CH);
        Molecule::from_atoms(vec![
            c(1, [0.000, 0.000, 0.000]),
            c(2, [0.000, 0.000, 0.000]),
            c(3, [1.500, 0.000, 0.000]),
        ])
        .unwrap()
    }

    #[test]
    fn two_atoms_on_the_same_coordinate_are_refused() {
        // Regression: this used to be accepted silently. Because the pair is
        // also a bonded neighbour it was dropped from the intramolecular term
        // altogether, so every energy came out finite and the malformed
        // structure produced plausible-looking poses. A crash would at least
        // have been visible.
        let err = Ligand::from_molecule(coincident_neighbours())
            .expect_err("coincident atoms must be refused, not scored");
        let message = err.to_string();
        assert!(message.contains("0.0"), "message: {message}");
        assert!(message.contains("below the"), "message: {message}");
    }

    #[test]
    fn the_separation_floor_does_not_reject_a_real_molecule() {
        // H2 has the shortest bond that exists (0.74 Å), and this chain is
        // built from realistic C-C and C-H distances. If the floor ever
        // rejects something like this it is set far too high.
        // C1 at the origin, C4 one real C-C bond away, and both hydrogens of
        // the methyl group in the plane perpendicular to that bond. Placing an
        // H along the bond axis instead would put it 0.45 Å from C4, which is
        // the floor being tested for -- real geometry has to stay clear of it.
        let h = |i: u32, p: Vec3| Atom::new(i, p, Element::H, AtomType::HD);
        let mol = Molecule::from_atoms(vec![
            Atom::new(1, [0.000, 0.000, 0.000], Element::C, AtomType::CH),
            h(2, [0.000, 1.090, 0.000]),
            h(3, [0.000, -0.545, 0.943]),
            Atom::new(4, [1.540, 0.000, 0.000], Element::C, AtomType::CH),
        ])
        .unwrap();
        let tightest = (0..mol.len())
            .flat_map(|i| ((i + 1)..mol.len()).map(move |j| (i, j)))
            .map(|(i, j)| dist3(mol.atoms[i].coord, mol.atoms[j].coord))
            .fold(f64::INFINITY, f64::min);
        assert!(
            tightest > MIN_ATOM_SEPARATION,
            "real geometry has a {tightest:.3} Å contact, inside the {MIN_ATOM_SEPARATION:.1} Å floor"
        );
        // The floor is not close to any real bond: the tightest contact here is
        // a normal C-H bond. If this ever approaches 0.5 the floor is wrong.
        assert!(tightest > 1.0, "tightest contact is {tightest:.3} Å");
        assert!(closest_atom_pair(&mol).is_none());
        assert!(Ligand::from_molecule(mol).is_ok());
    }

    #[test]
    fn butane_has_one_torsion_and_no_intra_pairs() {
        let lig = Ligand::from_molecule(butane()).unwrap();
        assert_eq!(lig.num_torsions(), 1);
        assert_eq!(lig.ndof(), 7);
        // The terminal C1–C4 contact is only *three* bonds apart, so it is
        // excluded along with the covalent 1-2 and 1-3 contacts: a four-atom
        // chain has no non-bonded intramolecular pair at all.
        assert!(
            lig.intramolecular_pairs().is_empty(),
            "got {:?}",
            lig.intramolecular_pairs()
        );
    }

    /// A six-atom chain: the 1-5 pairs (1–5, 2–6) are four bonds apart and
    /// *are* scored, which is exactly the torsional term that keeps a flexible
    /// ligand from folding through itself.
    #[test]
    fn hexane_has_intra_pairs_at_four_bonds() {
        let mut atoms = Vec::new();
        for i in 0..6u32 {
            let mut a = Atom::new(i + 1, [1.5 * i as f64, 0.0, 0.0], Element::C, AtomType::CH);
            a.kind = AtomKind::Hydrophobic;
            atoms.push(a);
        }
        let mol = Molecule::from_atoms(atoms).unwrap();
        let lig = Ligand::from_molecule(mol).unwrap();
        assert_eq!(lig.num_torsions(), 3);
        assert_eq!(
            lig.intramolecular_pairs(),
            &[(0, 4), (0, 5), (1, 5)],
            "only 1-5 contacts should be scored"
        );
    }

    #[test]
    fn ethane_has_no_torsions_and_no_intra_pairs() {
        let mol = Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [1.5, 0.0, 0.0], Element::C, AtomType::CH),
        ])
        .unwrap();
        let lig = Ligand::from_molecule(mol).unwrap();
        assert_eq!(lig.num_torsions(), 0);
        assert!(lig.intramolecular_pairs().is_empty());
    }

    #[test]
    fn benzene_pair_count() {
        // Every atom is within 3 bonds of the others in a 6-ring, so a benzene
        // has no intramolecular partners at all.
        let atoms = (0..6)
            .map(|i| {
                let a = i as f64 * std::f64::consts::PI / 3.0;
                Atom::new(
                    i + 1,
                    [1.39 * a.cos(), 1.39 * a.sin(), 0.0],
                    Element::C,
                    AtomType::CP,
                )
            })
            .collect();
        let mol = Molecule::from_atoms(atoms).unwrap();
        let lig = Ligand::from_molecule(mol).unwrap();
        assert!(lig.intramolecular_pairs().is_empty());
    }

    #[test]
    fn per_atom_weights_follow_kinds() {
        let lig = Ligand::from_molecule(butane()).unwrap();
        for w in &lig.weights {
            assert_eq!(w[crate::grid::MapSlot::Shape.index()], 1.0);
            assert_eq!(w[crate::grid::MapSlot::Hydrophobic.index()], 1.0);
        }
    }

    #[test]
    fn empty_ligand_is_rejected() {
        // A structure with no atoms is not constructible at all, so build the
        // error path explicitly.
        let err = Ligand::from_molecule(Molecule::default()).unwrap_err();
        assert!(matches!(err, crate::DockError::InvalidMolecule(_)));
        // A single-atom ligand, by contrast, is legal.
        let one = Molecule::from_atoms(vec![Atom::new(
            1,
            [0.0, 0.0, 0.0],
            Element::C,
            AtomType::CH,
        )])
        .unwrap();
        assert!(Ligand::from_molecule(one).is_ok());
    }

    #[test]
    fn radius_covers_every_atom() {
        let lig = Ligand::from_molecule(butane()).unwrap();
        let c = lig.molecule.centroid();
        for a in &lig.molecule.atoms {
            let d = ((a.coord[0] - c[0]).powi(2)
                + (a.coord[1] - c[1]).powi(2)
                + (a.coord[2] - c[2]).powi(2))
            .sqrt();
            assert!(d <= lig.radius() + 1e-12);
        }
    }
}
