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
    ///
    /// # Why this repeats the coordinate check [`Molecule::from_atoms`] makes
    ///
    /// Because this is the last gate before a ligand can be docked, so it is
    /// where the rule belongs even though it is now the second place it is
    /// written. One definition,
    /// [`crate::types::non_finite_coordinate_refusal`], called from both.
    ///
    /// The doors that made the repeat *necessary* are now shut. `Molecule` has
    /// private fields and no `Default`, so a caller outside this crate cannot
    /// assemble one without going through a constructor; and its `Deserialize`
    /// is hand-written rather than derived, so
    /// `serde_json::from_str::<Molecule>(..)` hands the raw shape to the same
    /// validator the constructors use instead of building a molecule from
    /// whatever the file says. What is still open, and named on the `Molecule`
    /// type, is the crate-internal literals. The check stays anyway, for the
    /// door that is not shut and for the one case this function is the right
    /// place for: it is public, so a caller can hand it a molecule that was
    /// legitimately constructed but is *empty* or has coincident atoms,
    /// neither of which is this crate's business to have rejected in a
    /// constructor that does not score.
    ///
    /// The claim this function can honestly make is the narrow one: *whatever
    /// molecule reaches it, an empty one and one with two atoms on the same
    /// coordinate are refused here.* It is not a claim that the molecule
    /// reached the search intact.
    pub fn from_molecule_with(
        molecule: Molecule,
        declared_torsions: &[(usize, usize)],
    ) -> Result<Ligand> {
        if molecule.is_empty() {
            return Err(DockError::molecule("ligand contains no atoms"));
        }
        if let Some(err) = crate::types::non_finite_coordinate_refusal("ligand", &molecule.atoms) {
            return Err(err);
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
        Ligand::from_parsed(parsed)
    }

    /// Prepare a ligand from an already-parsed PDBQT structure.
    ///
    /// The one place a parsed file becomes a dockable ligand, so that
    /// [`Ligand::from_pdbqt`] and a caller holding PDBQT *text* cannot end up
    /// with different notions of what a valid ligand is.
    ///
    /// # Why the declared torsion count is a refusal and not a note
    ///
    /// A Meeko-dialect file states its own flexibility in `TORSDOF n`, and
    /// `TORSION`/`BRANCH` records name the rotatable bonds. The engine builds
    /// its own tree from the covalent graph and those records, and the two can
    /// disagree — a `TORSION` line naming an atom the file does not contain is
    /// dropped silently, and a terminal bond is not rotatable however the
    /// preparer wrote it. When they disagree the search runs over a different
    /// number of degrees of freedom than the file asked for, and the caller
    /// cannot tell: the poses come back finite, ranked, and plausible, and
    /// nothing in a [`crate::docking::DockingResult`] says the flexibility was
    /// not the flexibility requested. That is a docking run reporting success
    /// for work it did not do, and unlike an unrecognised *atom* type — which
    /// is a graded degradation the caller may legitimately accept, and which
    /// [`crate::receptor::Receptor::unknown_atom_types`] reports rather than
    /// refuses — a wrong torsion count is not something a caller can weigh.
    ///
    /// The message names both numbers, because "the file is broken" is not
    /// actionable on its own and "re-export the ligand" is.
    ///
    /// Files that declare no `TORSDOF` are unaffected: there is nothing to
    /// disagree with, and a receptor read through this path has no tree to
    /// promise.
    pub fn from_parsed(parsed: ParsedStructure) -> Result<Ligand> {
        let declared = parsed.torsdof;
        let torsions = parsed.active_torsion_pairs();
        let ligand = Ligand::from_molecule_with(parsed.molecule, &torsions)?;
        if let Some(declared) = declared {
            let derived = ligand.num_torsions();
            if derived != declared {
                return Err(DockError::molecule(format!(
                    "the file declares {declared} rotatable bond(s) (TORSDOF) and \
                     carries {} resolvable torsion record(s), but the ligand prepared \
                     from it has {derived} rotatable bond(s): the search would run over \
                     {} degrees of freedom rather than the {} the file asked for, and \
                     nothing in the returned poses would show that. Re-export the \
                     ligand from its source structure.",
                    torsions.len(),
                    6 + derived,
                    6 + declared,
                )));
            }
        }
        Ok(ligand)
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

    /// A Meeko-dialect three-carbon file: `TORSDOF` and one `BRANCH` naming a
    /// bond that is *terminal* (C3 has no further neighbour), so the engine
    /// builds a zero-torsion tree from it.
    fn declared_torsion_that_cannot_be_built(declared: usize) -> ParsedStructure {
        let text = format!(
            "ROOT\n\
             ATOM      1  C1  UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\n\
             ATOM      2  C2  UNL     1       1.500   0.000   0.000  1.00  0.00     0.000 C\n\
             ENDROOT\n\
             BRANCH   2   3\n\
             ATOM      3  C3  UNL     1       3.000   1.300   0.000  1.00  0.00     0.000 C\n\
             ENDBRANCH\n\
             TORSDOF {declared}\n"
        );
        crate::pdbqt::parse_pdbqt(&text).expect("the fixture should parse")
    }

    #[test]
    fn a_declared_torsion_the_engine_cannot_build_is_refused() {
        // The defect this closes. The file says one rotatable bond, the engine
        // builds none, and before the refusal it docked the rigid body and
        // returned nine ranked poses with no field anywhere saying the
        // flexibility was not the flexibility requested.
        let err = Ligand::from_parsed(declared_torsion_that_cannot_be_built(1))
            .expect_err("a declared torsion the engine cannot build must be refused");
        let message = err.to_string();
        assert!(message.contains("1 rotatable bond"), "message: {message}");
        assert!(message.contains("0 rotatable bond"), "message: {message}");
    }

    #[test]
    fn a_declaration_the_engine_can_honour_is_accepted() {
        // The other direction, and the one that decides whether the refusal is
        // a check or a blanket ban: the same file declaring what the engine
        // actually derives has to go through, or this would be refusing
        // perfectly good ligands.
        let lig = Ligand::from_parsed(declared_torsion_that_cannot_be_built(0))
            .expect("a file whose TORSDOF matches the derived tree must be accepted");
        assert_eq!(lig.num_torsions(), 0);
    }

    #[test]
    fn a_file_that_declares_no_torsdof_is_never_refused_for_it() {
        // Receptors are read through a molecule path and carry no TORSDOF, and
        // an AutoDock-4 style file declares torsions only as TORSION records.
        // Neither has a count to disagree with, so the check must be silent
        // rather than defaulting either number to zero and inventing a
        // mismatch.
        let mut parsed = declared_torsion_that_cannot_be_built(1);
        parsed.torsdof = None;
        let lig = Ligand::from_parsed(parsed).expect("no declared count, nothing to refuse");
        assert_eq!(lig.num_torsions(), 0);
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
    fn a_non_finite_coordinate_is_refused_by_a_ligand_built_in_memory() {
        // The hole this closes, from the caller's side. `Ligand::from_arrays` is
        // the documented way to build a ligand from a table of atoms and the
        // RDKit front-end's route into the engine; before the check in
        // `Molecule::from_atoms` a NaN here produced a ligand whose
        // `reference_coords` were NaN, whose bond perception had silently lost
        // whatever the NaN atom was bonded to -- so the derived torsion count
        // changed -- and which docked to a result blaming the box.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut atoms = butane().atoms;
            atoms[2].coord[2] = bad;
            // Either gate refusing is the claim: `Molecule::from_atoms` is the
            // earlier one and says "structure", `Ligand::from_molecule` is the
            // later one and says "ligand". Which of the two notices is a
            // separate claim, and it is the wrong one to assert through this
            // path; `a_molecule_that_skipped_the_constructor_is_still_refused`
            // below is the one that pins the ligand's own gate.
            let err = match Molecule::from_atoms(atoms) {
                Err(e) => e.to_string(),
                Ok(mol) => match Ligand::from_molecule(mol) {
                    Ok(_) => panic!("a {bad} coordinate must be refused"),
                    Err(e) => e.to_string(),
                },
            };
            assert!(err.contains("atom 2"), "message: {err}");
            assert!(err.contains('z'), "message: {err}");
            assert!(err.contains("non-finite"), "message: {err}");
        }
    }

    #[test]
    fn a_molecule_that_skipped_the_constructor_is_still_refused() {
        // The case the constructor check cannot cover, and the reason
        // `from_molecule_with` repeats it. `Molecule`'s fields are now
        // `pub(crate)`, so the *external* struct literal is gone -- but a
        // molecule that was legitimately built and then had a coordinate
        // written over is a `NaN` in a structure that looks fine, and that is
        // this crate's own unit-test door rather than a caller's. The rule
        // belongs at the last gate before docking either way.
        let mut mol = butane();
        mol.atoms[1].coord[0] = f64::NAN;
        let err = match Ligand::from_molecule(mol) {
            Ok(_) => panic!("a molecule mutated after construction must still be refused"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("ligand atom 1"), "message: {err}");
    }

    #[test]
    fn an_ordinary_in_memory_ligand_is_still_accepted() {
        // The accept direction for the check that was just added here. Butane
        // and a structure far from the origin both have to keep working; if the
        // finiteness rule had become a magnitude limit, this is what would catch
        // it.
        assert_eq!(Ligand::from_molecule(butane()).unwrap().num_torsions(), 1);
        let offset = Molecule {
            atoms: butane()
                .atoms
                .iter()
                .map(|a| {
                    Atom::new(
                        a.serial,
                        [a.coord[0] + 9000.0, a.coord[1], a.coord[2]],
                        a.element,
                        a.atom_type,
                    )
                })
                .collect(),
            bonds: butane().bonds,
            neighbors: butane().neighbors,
        };
        assert!(Ligand::from_molecule(offset).is_ok());
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
        // error path explicitly. The literal is in-crate, which is the one
        // remaining way to make one -- `Molecule::default()` is no longer
        // public API. See the `Molecule` struct documentation.
        let empty = Molecule {
            atoms: Vec::new(),
            bonds: Vec::new(),
            neighbors: Vec::new(),
        };
        let err = Ligand::from_molecule(empty).unwrap_err();
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
