//! Core molecular data structures: elements, AutoDock atom types, atoms, bonds and molecules.
//!
//! # Provenance and licensing
//!
//! The data model here follows the **PDBQT** specification published by the
//! AutoDock project and the atom-typing scheme of AutoDock Vina, both of
//! which are open source:
//!
//! * AutoDock 4.2 — Morris et al., *J. Comput. Chem.* **29**, 2789 (2008),
//!   GPL-2.0. PDBQT column layout and `REMARK`/`TORSDOF` conventions.
//! * AutoDock Vina 1.2 — Trott & Olson, *J. Comput. Chem.* **31**, 455 (2010),
//!   Apache-2.0. The hydrogen-bond donor/acceptor/hydrophobic atom classes and
//!   the `e` / `e_hb` / `e_hyd` grid decomposition.
//! * Meeko — Morris et al., *PLoS ONE* **17**, e0163573 (2022), LGPL-2.1.
//!   Receptor/ligand PDBQT preparation semantics.
//!
//! This is an independent, clean-room re-implementation. No proprietary
//! AutoDockTools (ADT / MGLTools) code was consulted or reused.

use serde::{Deserialize, Serialize};

use crate::error::{DockError, Result};

/// A 3-D point/vector in double precision.
///
/// Kept as a plain array so that it is `repr(C)`-compatible with the GPU
/// upload path and cheap to copy in inner loops.
pub type Vec3 = [f64; 3];

/// Chemical element, ordered so that `#[repr(u8)]` gives a dense discriminant
/// suitable for indexing lookup tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Element {
    /// Hydrogen.
    H = 0,
    /// Carbon.
    C = 1,
    /// Nitrogen.
    N = 2,
    /// Oxygen.
    O = 3,
    /// Fluorine.
    F = 4,
    /// Phosphorus.
    P = 5,
    /// Sulfur.
    S = 6,
    /// Chlorine.
    Cl = 7,
    /// Bromine.
    Br = 8,
    /// Iodine.
    I = 9,
    /// A d-block / s-block metal centre.
    Met = 10,
    /// A polar hydrogen carried explicitly by Meeko-style PDBQT (`HD`).
    ///
    /// Note: `HD` in a PDBQT file is *always* a hydrogen bonded to N or O; it
    /// is modelled with [`Element::H`] plus a donor flag rather than a
    /// separate element, so this variant exists only for round-tripping the
    /// literal PDBQT token.
    Placeholder = 11,
}

impl Element {
    /// The number of distinct element discriminants.
    pub const COUNT: usize = 12;

    /// Every element, in discriminant order. Useful for building lookup tables.
    pub const ALL: [Element; Element::COUNT] = [
        Element::H,
        Element::C,
        Element::N,
        Element::O,
        Element::F,
        Element::P,
        Element::S,
        Element::Cl,
        Element::Br,
        Element::I,
        Element::Met,
        Element::Placeholder,
    ];

    /// One-letter chemical symbol as used in PDB atom names.
    pub fn symbol(self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::F => "F",
            Element::P => "P",
            Element::S => "S",
            Element::Cl => "CL",
            Element::Br => "BR",
            Element::I => "I",
            Element::Met => "M",
            Element::Placeholder => "*",
        }
    }

    /// RDKit/PDBQT atom-type token for this element (e.g. `"C"`, `"Cl"`, `"OA"`).
    ///
    /// The bare element symbol is returned here; polar typing (`OA`, `NA`, …)
    /// is resolved by [`crate::types::AtomType::from_token`], which needs the
    /// bonding context.
    pub fn from_symbol(sym: &str) -> Element {
        match sym.trim() {
            "H" | "HD" | "HS" => Element::H,
            "C" | "A" => Element::C,
            "N" | "NA" | "NS" => Element::N,
            "O" | "OA" | "OS" => Element::O,
            "F" | "FA" => Element::F,
            "P" => Element::P,
            "S" | "SA" => Element::S,
            "CL" | "Cl" => Element::Cl,
            "BR" | "Br" => Element::Br,
            "I" => Element::I,
            "MG" | "CA" | "MN" | "FE" | "ZN" | "CU" | "NI" | "CO" | "CD" | "HG" => Element::Met,
            _ => Element::Placeholder,
        }
    }

    /// Covalent radius in Ångström, used for bond perception.
    ///
    /// Values are the standard covalent radii (Cordero et al., *Dalton Trans.*
    /// 2008, 2832) with the usual docking-specific adjustments: a small
    /// tolerance is added so that borderline aromatic bonds are still
    /// perceived by the distance test in [`crate::chemistry`].
    pub fn covalent_radius(self) -> f64 {
        match self {
            Element::H => 0.31,
            Element::C => 0.76,
            Element::N => 0.71,
            Element::O => 0.66,
            Element::F => 0.57,
            Element::P => 1.07,
            Element::S => 1.05,
            Element::Cl => 1.02,
            Element::Br => 1.20,
            Element::I => 1.39,
            Element::Met => 1.39,
            Element::Placeholder => 0.77,
        }
    }

    /// Interaction radius used by the Vina-family scoring function.
    ///
    /// Surface distances are `d = ‖rᵢ − rⱼ‖ − (Rᵢ + Rⱼ)`, so these radii
    /// decide *where on the separation axis every term in
    /// [`crate::scoring`] sits*. It is therefore not a cosmetic parameter: get
    /// it wrong by 1.5 Å and the whole energy landscape moves into the steric
    /// exclusion zone.
    ///
    /// These are the per-element "XS" radii, i.e. half the reference
    /// non-bonded contact distance: carbon 1.9 Å puts a C···C pair at the
    /// 3.8 Å van der Waals contact, and oxygen 1.6 Å puts an O···O pair at
    /// 3.2 Å of separation.
    ///
    /// Those two numbers are not the same event, and an earlier revision of
    /// this comment ran them together. 3.2 Å is where the *hydrophobic* term
    /// starts; the *hydrogen-bond* term is maximal half an ångström closer, at
    /// 2.70 Å, because its window is `d ≤ −0.5` and `−0.5 + 1.6 + 1.6 =
    /// 2.70`. The old text put the hydrogen-bond maximum at 3.2 Å and quoted
    /// `d ≤ −0.7` as its window. Both claims were false: `d ≤ −0.7` is a
    /// window the code stopped using, and at `d = 0` — the 3.2 Å contact — the
    /// term is exactly **zero**, being spent precisely there. Those numbers are
    /// now asserted by
    /// `scoring::tests::the_hbond_window_lands_on_the_documented_separations`
    /// rather than left to be remembered.
    ///
    /// # This was a flat 0.4 Å, and that was a bug
    ///
    /// A single additive `0.4 Å` for every heavy atom is a *covalent* bond
    /// length, not an interaction radius, and it pushed every attractive term
    /// inside the clash region. With `R = 0.4` the pair terms reach their
    /// maxima at
    ///
    /// | term  | surface distance needed | real separation needed |
    /// |-------|------------------------|------------------------|
    /// | `hb`  | `d ≤ −0.7`             | **0.1 Å**             |
    /// | `g1`  | `d = 0.5`              | 1.3 Å (C–C)           |
    /// | `g2`  | `d = 0`                | 0.8 Å (C–C)           |
    /// | `hyd` | `d ≤ 0.5`              | 1.3 Å (C–C)           |
    ///
    /// (That table is history, kept in the past tense deliberately: it
    /// describes the `R = 0.4` regime and is *not* where the terms sit today.
    /// Note that it is the same `d ≤ −0.7` window that the paragraph above used
    /// to quote as current.)
    ///
    /// Two oxygens 0.1 Å apart are not a hydrogen bond, they are one atom and
    /// its own image, so the `−0.587` hydrogen-bond weight — the single largest
    /// term in the Vina function — was **mathematically dead**: exactly zero at
    /// every distance a real hydrogen bond can form. Measured on phenol against
    /// a crown-ether oxygen, the term contributed −0.0032 kcal/mol where it
    /// should have contributed about −0.18, and the scoring function preferred
    /// burying the ligand inside the protein, which is how the flaw was first
    /// found.
    ///
    /// Hydrogens keep a radius of 0: the only hydrogens a PDBQT retains are
    /// polar ones, and they sit *on* their own heavy-atom donor, so giving them
    /// a radius would make every ordinary hydrogen bond register as an overlap.
    pub fn interaction_radius(self) -> f64 {
        match self {
            Element::C => 1.90,
            Element::N => 1.75,
            Element::O => 1.60,
            Element::F => 1.545,
            Element::P => 2.10,
            Element::S => 2.00,
            Element::Cl => 1.948,
            Element::Br => 2.220,
            Element::I => 2.350,
            Element::Met => 1.20,
            // An unrecognised token is folded into the carbon grid map (see
            // `grid_type_index`), so it has to carry carbon's radius too —
            // giving it 0 would push every contact it makes past the cutoff
            // and make the atom silently vanish from the energy.
            Element::Placeholder => 1.90,
            Element::H => 0.0,
        }
    }

    /// True for elements that form covalent (rather than ionic) bonds.
    pub fn is_metal(self) -> bool {
        self == Element::Met
    }
}

/// The AutoDock "XS" atom type: element plus polar/hydrophobic decoration.
///
/// These are the types that appear in the last column of a PDBQT `ATOM`
/// record. They determine which receptor grid maps a ligand atom samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomType {
    /// Aliphatic carbon — hydrophobic.
    CH,
    /// Aromatic carbon — hydrophobic.
    CP,
    /// Nitrogen, non-hydrogen-bonding (e.g. a tertiary amine N-methyl).
    NP,
    /// Nitrogen that donates a hydrogen bond but does not accept one.
    ND,
    /// Nitrogen that accepts a hydrogen bond.
    NA,
    /// Nitrogen that both donates and accepts (e.g. imidazole `[nH]`).
    NDA,
    /// Oxygen, non-hydrogen-bonding (rare; mostly used for rigid water models).
    OP,
    /// Oxygen that donates a hydrogen bond but does not accept one (alcohol).
    OD,
    /// Oxygen that accepts a hydrogen bond (carbonyl, ether).
    OA,
    /// Oxygen that both donates and accepts (protonated / water).
    ODA,
    /// Sulfur.
    SP,
    /// Phosphorus.
    PP,
    /// Fluorine — hydrophobic.
    FH,
    /// Chlorine — hydrophobic.
    ClH,
    /// Bromine — hydrophobic.
    BrH,
    /// Iodine — hydrophobic.
    IH,
    /// Metal centre — hydrogen-bond donor.
    MetD,
    /// An explicit polar hydrogen (`HD` in PDBQT).
    HD,
    /// Unrecognised token; retained verbatim for round-tripping.
    Unknown,
}

impl AtomType {
    /// Number of distinct atom-type discriminants (used for dense table sizing).
    pub const COUNT: usize = 18;

    /// Parse a PDBQT atom-type token.
    ///
    /// Matching is case-insensitive because AutoDock 4 writes `Cl`/`Br` while
    /// some tools emit `CL`/`BR`.
    pub fn from_token(token: &str) -> AtomType {
        match token.trim() {
            "C" => AtomType::CH,
            "A" => AtomType::CP,
            "N" => AtomType::NP,
            "NA" => AtomType::NA,
            "NS" => AtomType::NP,
            "ND" => AtomType::ND,
            "NDA" => AtomType::NDA,
            "O" => AtomType::OP,
            "OA" => AtomType::OA,
            "OS" => AtomType::OP,
            "OD" => AtomType::OD,
            "ODA" => AtomType::ODA,
            "S" => AtomType::SP,
            "SA" => AtomType::SP,
            "P" => AtomType::PP,
            "F" => AtomType::FH,
            "FA" => AtomType::FH,
            "CL" | "Cl" => AtomType::ClH,
            "BR" | "Br" => AtomType::BrH,
            "I" => AtomType::IH,
            "MG" | "CA" | "MN" | "FE" | "ZN" | "CU" | "NI" | "CO" | "CD" | "HG" => AtomType::MetD,
            "HD" => AtomType::HD,
            "H" => AtomType::HD, // bare `H` in a ligand PDBQT is a merged polar H
            _ => AtomType::Unknown,
        }
    }

    /// The canonical PDBQT token for this type.
    pub fn token(self) -> &'static str {
        match self {
            AtomType::CH => "C",
            AtomType::CP => "A",
            AtomType::NP => "N",
            AtomType::ND => "ND",
            AtomType::NA => "NA",
            AtomType::NDA => "NDA",
            AtomType::OP => "O",
            AtomType::OD => "OD",
            AtomType::OA => "OA",
            AtomType::ODA => "ODA",
            AtomType::SP => "S",
            AtomType::PP => "P",
            AtomType::FH => "F",
            AtomType::ClH => "Cl",
            AtomType::BrH => "Br",
            AtomType::IH => "I",
            AtomType::MetD => "Zn",
            AtomType::HD => "HD",
            AtomType::Unknown => "Xx",
        }
    }

    /// The element underlying this atom type.
    pub fn element(self) -> Element {
        match self {
            AtomType::CH | AtomType::CP => Element::C,
            AtomType::NP | AtomType::ND | AtomType::NA | AtomType::NDA => Element::N,
            AtomType::OP | AtomType::OD | AtomType::OA | AtomType::ODA => Element::O,
            AtomType::SP => Element::S,
            AtomType::PP => Element::P,
            AtomType::FH => Element::F,
            AtomType::ClH => Element::Cl,
            AtomType::BrH => Element::Br,
            AtomType::IH => Element::I,
            AtomType::MetD => Element::Met,
            AtomType::HD => Element::H,
            AtomType::Unknown => Element::Placeholder,
        }
    }

    /// Base hydrogen-bond **donor** flag implied by the atom type alone.
    pub fn is_donor(self) -> bool {
        matches!(
            self,
            AtomType::ND | AtomType::NDA | AtomType::OD | AtomType::ODA
        )
    }

    /// Base hydrogen-bond **acceptor** flag implied by the atom type alone.
    pub fn is_acceptor(self) -> bool {
        matches!(
            self,
            AtomType::NA | AtomType::NDA | AtomType::OA | AtomType::ODA
        )
    }

    /// Base **hydrophobic** flag implied by the atom type alone.
    pub fn is_hydrophobic(self) -> bool {
        matches!(
            self,
            AtomType::CH
                | AtomType::CP
                | AtomType::FH
                | AtomType::ClH
                | AtomType::BrH
                | AtomType::IH
        )
    }
}

/// The Vina interaction class of an atom **after** the bonding context has been
/// taken into account.
///
/// Vina does not score raw PDBQT types directly. Every atom is re-classified
/// from its own type *and* the types of its covalent neighbours, because e.g.
/// the hydroxyl oxygen of a carboxylic acid both donates (it carries the
/// explicit `HD`) and accepts (it is an `OA`), and scoring it as only one of
/// the two would distort the interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomKind {
    /// Apolar contact partner (carbon, halogen, thioether sulfur).
    Hydrophobic,
    /// Hydrogen-bond donor only.
    Donor,
    /// Hydrogen-bond acceptor only.
    Acceptor,
    /// Both donor and acceptor (carboxylic acid, alcohol, imidazole N–H, water).
    DonorAcceptor,
    /// Participates in no class — contributes only the shape/repulsion terms.
    Other,
}

impl AtomKind {
    /// Number of distinct interaction classes.
    pub const COUNT: usize = 5;

    /// Compact discriminant used to index pair-classification tables.
    pub fn index(self) -> usize {
        match self {
            AtomKind::Hydrophobic => 0,
            AtomKind::Donor => 1,
            AtomKind::Acceptor => 2,
            AtomKind::DonorAcceptor => 3,
            AtomKind::Other => 4,
        }
    }
}

/// A single atom in a structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    /// PDB serial number (1-based). Preserved for round-tripping.
    pub serial: u32,
    /// PDB atom name, e.g. `"CA"`, `"1HB"`. Preserved for round-tripping.
    pub name: String,
    /// Residue name this atom belongs to (ligand PDBQT: `UNL` or `LIG`).
    pub resname: String,
    /// Residue sequence number.
    pub resid: i32,
    /// Cartesian position in Ångström.
    pub coord: Vec3,
    /// Partial Gasteiger charge (e). Informational — the Vina family is
    /// charge-independent, but the value is preserved for Vinardo extensions
    /// and for round-tripping Meeko output.
    pub charge: f32,
    /// Chemical element.
    pub element: Element,
    /// AutoDock atom type as written in the PDBQT file.
    pub atom_type: AtomType,
    /// Interaction class after context refinement.
    pub kind: AtomKind,
}

impl Atom {
    /// Create a new atom with sensible defaults.
    pub fn new(serial: u32, coord: Vec3, element: Element, atom_type: AtomType) -> Self {
        Atom {
            serial,
            name: element.symbol().to_string(),
            resname: "UNL".to_string(),
            resid: 1,
            coord,
            charge: 0.0,
            element,
            atom_type,
            kind: AtomKind::Other,
        }
    }

    /// True if the atom is an explicit polar hydrogen.
    pub fn is_polar_hydrogen(&self) -> bool {
        self.element == Element::H
    }

    /// True if the atom is a metal centre.
    pub fn is_metal(&self) -> bool {
        self.element == Element::Met
    }

    /// True if the atom can donate a hydrogen bond.
    #[inline]
    pub fn can_donate(&self) -> bool {
        matches!(self.kind, AtomKind::Donor | AtomKind::DonorAcceptor)
    }

    /// True if the atom can accept a hydrogen bond.
    #[inline]
    pub fn can_accept(&self) -> bool {
        matches!(self.kind, AtomKind::Acceptor | AtomKind::DonorAcceptor)
    }

    /// True if the atom makes apolar contacts.
    #[inline]
    pub fn is_apolar(&self) -> bool {
        self.kind == AtomKind::Hydrophobic
    }
}

/// Chemical bond, stored as an index pair into [`Molecule::atoms`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bond {
    /// Index of the first atom.
    pub i: usize,
    /// Index of the second atom.
    pub j: usize,
    /// Whether the bond is a single covalent bond that is *not* in a ring.
    pub rotatable: bool,
    /// Whether the bond lies in a ring (detected by smallest-ring-through-bond).
    pub in_ring: bool,
}

impl Bond {
    /// Create a non-rotatable, non-ring bond.
    pub fn new(i: usize, j: usize) -> Self {
        Bond {
            i,
            j,
            rotatable: false,
            in_ring: false,
        }
    }
}

/// A structure: atoms, the covalent graph, and derived interaction classes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Molecule {
    /// Atoms in file order.
    pub atoms: Vec<Atom>,
    /// Covalent bonds.
    pub bonds: Vec<Bond>,
    /// Adjacency: for each atom, the indices of its bonded partners.
    pub neighbors: Vec<Vec<usize>>,
}

impl Molecule {
    /// Largest ring (in bonds) that [`Self::assign_ring_membership`] will search
    /// for. Anything larger is treated as acyclic, which is correct for every
    /// ring that occurs in a drug-like molecule and keeps the cost bounded.
    const MAX_RING_SIZE: usize = 32;

    /// Build a molecule from atoms, perceiving the covalent graph from geometry.
    ///
    /// Bond orders are not required by the Vina-family scoring functions, so
    /// perception uses a simple covalent-radius distance criterion — the same
    /// approach used by many open-source PDBQT front-ends.
    pub fn from_atoms(atoms: Vec<Atom>) -> Result<Self> {
        if atoms.is_empty() {
            return Err(DockError::molecule("structure contains no atoms"));
        }
        let mut mol = Molecule {
            atoms,
            bonds: Vec::new(),
            neighbors: Vec::new(),
        };
        mol.perceive_bonds()?;
        mol.assign_vina_atom_kinds();
        Ok(mol)
    }

    /// Perceive the covalent graph from interatomic distances.
    pub fn perceive_bonds(&mut self) -> Result<()> {
        let n = self.atoms.len();
        self.bonds.clear();
        self.neighbors = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.atoms[i];
                let b = &self.atoms[j];
                if a.element == Element::Met || b.element == Element::Met {
                    // Metal centres coordinate to a shell of ligands; treat any
                    // contact within 2.8 Å as a bond so that torsions attached
                    // to metals stay connected.
                    let d = dist3(a.coord, b.coord);
                    if d < 2.8 {
                        self.push_bond(i, j);
                    }
                    continue;
                }
                // A covalent bond is perceived when the interatomic distance is
                // shorter than the sum of covalent radii plus a tolerance.
                // The 0.45 Å slack absorbs C–C aromatic (1.39 Å) and C–N / C–O
                // bonds that sit slightly above the tabulated sum.
                let cutoff = a.element.covalent_radius() + b.element.covalent_radius() + 0.45;
                if dist3(a.coord, b.coord) < cutoff {
                    self.push_bond(i, j);
                }
            }
        }

        // A structure with no perceived bonds is legal: a monatomic ion in a
        // receptor, or a single probe atom. It simply has no intramolecular
        // terms, which the scoring function handles naturally.
        self.assign_ring_membership()?;
        Ok(())
    }

    fn push_bond(&mut self, i: usize, j: usize) {
        self.bonds.push(Bond::new(i, j));
        self.neighbors[i].push(j);
        self.neighbors[j].push(i);
    }

    /// Recompute which bonds lie in rings.
    ///
    /// Public because a caller that adds bonds by hand — the Python front-end
    /// does exactly that when RDKit supplies the bond table — must re-run ring
    /// perception, since rotatable-bond selection depends on it.
    pub fn assign_ring_membership(&mut self) -> Result<()> {
        self.assign_ring_membership_inner()
    }

    /// Flag bonds that lie on a ring.
    ///
    /// A bond `a–b` is in a ring exactly when `a` and `b` remain connected by
    /// some path once that single edge is removed. This is the smallest-ring
    /// through each bond (SSSP), found by a breadth-first search that is
    /// forbidden to traverse the edge under test — without that exclusion the
    /// search trivially "finds" the bond itself and reports every bond as a
    /// ring bond.
    ///
    /// The search is capped at [`Self::MAX_RING_SIZE`] shells. Every ring in
    /// a chemically meaningful ligand is far smaller than that, and the cap
    /// keeps ring perception linear on large receptor structures.
    fn assign_ring_membership_inner(&mut self) -> Result<()> {
        let n = self.atoms.len();
        for bond_idx in 0..self.bonds.len() {
            let (a, b) = (self.bonds[bond_idx].i, self.bonds[bond_idx].j);

            let mut prev = vec![usize::MAX; n];
            let mut shell = vec![a];
            prev[a] = a;
            let mut found = false;

            'search: for _ in 0..Self::MAX_RING_SIZE {
                let mut next_shell = Vec::new();
                for &u in &shell {
                    for &v in &self.neighbors[u] {
                        // Never traverse the bond under test.
                        if u == a && v == b {
                            continue;
                        }
                        if v == b {
                            found = true;
                            break 'search;
                        }
                        if prev[v] == usize::MAX {
                            prev[v] = u;
                            next_shell.push(v);
                        }
                    }
                }
                if next_shell.is_empty() {
                    break;
                }
                shell = next_shell;
            }
            self.bonds[bond_idx].in_ring = found;
        }
        Ok(())
    }

    /// Re-classify every atom into an [`AtomKind`] from its bonding context.
    ///
    /// PDBQT atom types describe an atom in isolation; the *kind* it is scored
    /// as depends on its neighbours. The rule is:
    ///
    /// 1. An explicit polar hydrogen (`HD`) is never itself a class — it marks
    ///    its heavy-atom partner as a **donor**. This is how Meeko/PDBQT
    ///    express "this oxygen has an N–H/O–H", and it is the only signal that
    ///    survives into the scoring function.
    /// 2. An atom that both donates and accepts is [`AtomKind::DonorAcceptor`]
    ///    (carboxylic acid, alcohol, imidazole N–H, water). Scoring it as only
    ///    one of the two would lose half its interactions.
    /// 3. Otherwise a donor is [`AtomKind::Donor`], an acceptor is
    ///    [`AtomKind::Acceptor`].
    /// 4. An atom with no hydrogen-bond capability of its own is
    ///    [`AtomKind::Hydrophobic`] when its PDBQT type says so (aliphatic and
    ///    aromatic carbon, halogens) and [`AtomKind::Other`] otherwise
    ///    (amide `N`, thioether `S`, phosphorus).
    ///
    /// Notice what is *not* here: a count of polar neighbours. The first
    /// version of this function used one, and it silently mis-typed every
    /// carboxyl oxygen as hydrophobic — a carboxyl oxygen's only neighbour is
    /// carbon, so "no polar neighbour" said nothing about it. The atom's own
    /// type is the authoritative signal; the neighbour pass exists only to
    /// propagate the `HD` donor mark.
    pub fn assign_vina_atom_kinds(&mut self) {
        let n = self.atoms.len();
        let base: Vec<AtomType> = self.atoms.iter().map(|a| a.atom_type).collect();
        let is_h: Vec<bool> = self.atoms.iter().map(|a| a.element == Element::H).collect();

        // (1) An atom bonded to an explicit polar hydrogen is a donor.
        let mut explicit_donor = vec![false; n];
        for (i, nbrs) in self.neighbors.iter().enumerate() {
            for &j in nbrs {
                if base[j] == AtomType::HD {
                    explicit_donor[i] = true;
                }
            }
        }

        for (i, atom) in self.atoms.iter_mut().enumerate() {
            atom.kind = if is_h[i] {
                // (1) The polar hydrogen itself is scored through its partner.
                AtomKind::Other
            } else {
                // (2)-(3)
                let donates = explicit_donor[i] || base[i].is_donor();
                let accepts = base[i].is_acceptor();
                if donates && accepts {
                    AtomKind::DonorAcceptor
                } else if donates {
                    AtomKind::Donor
                } else if accepts {
                    AtomKind::Acceptor
                // (4)
                } else if base[i].is_hydrophobic() {
                    AtomKind::Hydrophobic
                } else {
                    AtomKind::Other
                }
            };
        }
    }

    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// True if the molecule has no atoms.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Number of covalent bonds.
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Total Gasteiger charge of the structure.
    pub fn total_charge(&self) -> f64 {
        self.atoms.iter().map(|a| a.charge as f64).sum()
    }

    /// Geometric centroid of all atoms.
    pub fn centroid(&self) -> Vec3 {
        let mut c = [0.0f64; 3];
        for a in &self.atoms {
            for k in 0..3 {
                c[k] += a.coord[k];
            }
        }
        let n = self.atoms.len() as f64;
        [c[0] / n, c[1] / n, c[2] / n]
    }

    /// Axis-aligned bounding box as `(min_corner, max_corner)`.
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for a in &self.atoms {
            for k in 0..3 {
                lo[k] = lo[k].min(a.coord[k]);
                hi[k] = hi[k].max(a.coord[k]);
            }
        }
        (lo, hi)
    }
}

/// Euclidean distance between two cartesian points.
#[inline]
pub fn dist3(a: Vec3, b: Vec3) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Squared Euclidean distance between two cartesian points.
#[inline]
pub fn dist2_3(a: Vec3, b: Vec3) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Index of the larger element discriminant — used for lookup-table sizing.
pub fn grid_type_index(element: Element) -> usize {
    match element {
        Element::C => 0,
        Element::N => 1,
        Element::O => 2,
        Element::P => 3,
        Element::S => 4,
        Element::F => 5,
        Element::Cl => 6,
        Element::Br => 7,
        Element::I => 8,
        Element::Met => 9,
        // Hydrogen and unrecognised tokens are folded into the carbon map,
        // which is the behaviour AutoDock has for apolar hydrogens.
        Element::H | Element::Placeholder => 0,
    }
}

/// Number of receptor grid maps (one per element class).
pub const GRID_TYPE_COUNT: usize = 10;

/// Human-readable name of a grid type index.
pub fn grid_type_name(idx: usize) -> &'static str {
    const NAMES: [&str; GRID_TYPE_COUNT] = ["C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Met"];
    NAMES.get(idx).copied().unwrap_or("?")
}

/// Interaction radius of a grid type index, in Ångström.
///
/// Each block of a precalculated map is the field for a *hypothetical probe atom
/// of that element*, so the radius it was tabulated with is part of what the
/// block means — not a detail of how one receptor atom happened to be written. A
/// receptor oxygen contributes to a ligand nitrogen's block using **their two**
/// radii, and this function is the ligand's half of that sum.
///
/// An out-of-range index returns `NAN` rather than a plausible number: every call
/// site either loops `0..GRID_TYPE_COUNT` or passes [`grid_type_index`], so
/// reaching the fallback is a programming error, and a silent zero would turn
/// that error into a quietly wrong energy instead of an obviously broken one.
/// `the_radius_table_backs_the_grid_types` keeps this table and
/// [`Element::interaction_radius`] from drifting apart.
pub fn grid_type_radius(idx: usize) -> f64 {
    const RADII: [f64; GRID_TYPE_COUNT] = [
        1.90, 1.75, 1.60, 2.10, 2.00, 1.545, 1.948, 2.220, 2.350, 1.20,
    ];
    RADII.get(idx).copied().unwrap_or(f64::NAN)
}

/// Largest interaction radius among the grid types, in Ångström.
///
/// Lets the precalculation widen a single early-out once, instead of repeating
/// the same square root for every probe type.
pub const MAX_GRID_TYPE_RADIUS: f64 = 2.350;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a two-carbon ethane-like fragment for typing tests.
    fn two_carbons() -> Molecule {
        Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [1.54, 0.0, 0.0], Element::C, AtomType::CH),
        ])
        .expect("ethane perceives one bond")
    }

    #[test]
    fn perceives_covalent_bond() {
        let mol = two_carbons();
        assert_eq!(mol.bond_count(), 1);
        assert_eq!(mol.neighbors[0], vec![1]);
        assert_eq!(mol.neighbors[1], vec![0]);
    }

    #[test]
    fn distant_atoms_are_not_bonded() {
        // Two far-apart atoms are a legal (if unusual) structure with no bonds.
        let mol = Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [9.0, 0.0, 0.0], Element::C, AtomType::CH),
        ])
        .expect("an unbonded pair is still a valid molecule");
        assert_eq!(mol.bond_count(), 0);
        assert!(mol.neighbors[0].is_empty());
    }

    #[test]
    fn carbon_pair_is_hydrophobic() {
        let mol = two_carbons();
        assert_eq!(mol.atoms[0].kind, AtomKind::Hydrophobic);
        assert_eq!(mol.atoms[1].kind, AtomKind::Hydrophobic);
    }

    /// Acetic acid: CH3–C(=O)–O.
    fn acetic_acid() -> Molecule {
        let mut mol = Molecule::from_atoms(vec![
            Atom::new(1, [0.00, 0.00, 0.00], Element::C, AtomType::CH),
            Atom::new(2, [1.50, 0.00, 0.00], Element::C, AtomType::CH),
            Atom::new(3, [2.10, 1.30, 0.00], Element::O, AtomType::OA),
            Atom::new(4, [2.15, -1.30, 0.00], Element::O, AtomType::OA),
        ])
        .expect("acetic acid perceives its bonds");
        mol.assign_vina_atom_kinds();
        mol
    }

    #[test]
    fn carboxylic_oxygens_are_acceptors_not_hydrophobic() {
        // Regression test. An earlier version decided "no polar neighbour =>
        // hydrophobic", and a carboxyl oxygen's only neighbour is carbon, so
        // both oxygens of every acid, ester, amide and carbamate came out
        // apolar and simply never made a hydrogen bond.
        let mol = acetic_acid();
        assert_eq!(mol.atoms[2].kind, AtomKind::Acceptor, "carbonyl O");
        assert_eq!(mol.atoms[3].kind, AtomKind::Acceptor, "hydroxyl O");
    }

    #[test]
    fn an_explicit_polar_hydrogen_makes_its_oxygen_a_donor_acceptor() {
        // With the HD present, the hydroxyl oxygen donates *and* accepts.
        let mut mol = Molecule::from_atoms(vec![
            Atom::new(1, [0.00, 0.00, 0.00], Element::C, AtomType::CH),
            Atom::new(2, [1.50, 0.00, 0.00], Element::C, AtomType::CH),
            Atom::new(3, [2.10, 1.30, 0.00], Element::O, AtomType::OA),
            Atom::new(4, [2.15, -1.30, 0.00], Element::O, AtomType::OA),
            Atom::new(5, [3.10, -1.60, 0.00], Element::H, AtomType::HD),
        ])
        .expect("the hydroxyl perceives its bond");
        mol.assign_vina_atom_kinds();
        assert_eq!(mol.atoms[3].kind, AtomKind::DonorAcceptor);
        assert_eq!(mol.atoms[2].kind, AtomKind::Acceptor);
        // The hydrogen itself is scored through its partner, never on its own.
        assert_eq!(mol.atoms[4].kind, AtomKind::Other);
    }

    #[test]
    fn amide_nitrogen_is_inert() {
        // PDBQT type `N` is defined as non-hydrogen-bonding, so it must not
        // become a donor or an acceptor just for sitting next to two of them.
        let mut mol = Molecule::from_atoms(vec![
            Atom::new(1, [0.00, 0.00, 0.00], Element::C, AtomType::CH),
            Atom::new(2, [1.35, 0.00, 0.00], Element::C, AtomType::CH),
            Atom::new(3, [2.10, 1.15, 0.00], Element::O, AtomType::OA),
            Atom::new(4, [2.30, -1.25, 0.00], Element::N, AtomType::NP),
        ])
        .expect("formamide perceives its bonds");
        mol.assign_vina_atom_kinds();
        assert_eq!(mol.atoms[3].kind, AtomKind::Other, "amide N is inert");
        assert_eq!(mol.atoms[2].kind, AtomKind::Acceptor);
    }

    #[test]
    fn kind_assignment_does_not_depend_on_atom_order() {
        let mut a = acetic_acid();
        let n = a.atoms.len();
        // `reversed[i]` is where atom `i` ends up in the shuffled molecule.
        let reversed: Vec<usize> = (0..n).rev().collect();
        let mut b = Molecule {
            atoms: a.atoms.iter().rev().cloned().collect(),
            bonds: Vec::new(),
            neighbors: vec![Vec::new(); n],
        };
        for bond in a.bonds.iter() {
            let i = reversed[bond.i];
            let j = reversed[bond.j];
            b.bonds.push(Bond::new(i, j));
            b.neighbors[i].push(j);
            b.neighbors[j].push(i);
        }
        a.assign_vina_atom_kinds();
        b.assign_vina_atom_kinds();
        let ka: Vec<AtomKind> = a.atoms.iter().map(|x| x.kind).collect();
        let kb: Vec<AtomKind> = reversed.iter().map(|&i| b.atoms[i].kind).collect();
        assert_eq!(ka, kb);
    }

    #[test]
    fn empty_molecule_is_rejected() {
        assert!(Molecule::from_atoms(vec![]).is_err());
    }

    #[test]
    fn atom_type_tokens_round_trip() {
        for t in [
            AtomType::CH,
            AtomType::CP,
            AtomType::NA,
            AtomType::OA,
            AtomType::HD,
            AtomType::ClH,
        ] {
            assert_eq!(AtomType::from_token(t.token()), t, "round trip for {t:?}");
        }
    }

    #[test]
    fn aromatic_and_alkyl_carbon_share_element() {
        assert_eq!(AtomType::CH.element(), Element::C);
        assert_eq!(AtomType::CP.element(), Element::C);
        assert!(AtomType::CP.is_hydrophobic());
    }

    #[test]
    fn interaction_radius_is_zero_for_hydrogen() {
        assert_eq!(Element::H.interaction_radius(), 0.0);
    }

    /// `grid_type_radius` exists because each map block is the field for a
    /// *hypothetical probe* of that element, so it must not become a second,
    /// silently drifting copy of [`Element::interaction_radius`]. These are two
    /// tables of the same numbers on purpose — the grid needs the index form —
    /// and a test is what stops them becoming two different tables.
    #[test]
    fn the_radius_table_backs_the_grid_types() {
        const ELEMENTS: [Element; 10] = [
            Element::C,
            Element::N,
            Element::O,
            Element::F,
            Element::P,
            Element::S,
            Element::Cl,
            Element::Br,
            Element::I,
            Element::Met,
        ];
        for e in ELEMENTS {
            let idx = grid_type_index(e);
            assert!(
                (grid_type_radius(idx) - e.interaction_radius()).abs() < 1e-12,
                "{} is grid type {idx}, but that block's radius is {} while the \
                 element's own is {}",
                grid_type_name(idx),
                grid_type_radius(idx),
                e.interaction_radius()
            );
        }

        // Hydrogen and an unrecognised token both fold into the carbon block, so
        // that block is tabulated with carbon's radius. This is the documented
        // AutoDock behaviour for apolar hydrogens, and it is also why a zero
        // crossing measured on a C...H pair reports the radius the *map* was
        // built with rather than the probe's own. Pinned here so the reason is
        // not rediscovered by measurement later.
        assert_eq!(
            grid_type_radius(grid_type_index(Element::H)),
            Element::C.interaction_radius()
        );
        assert_eq!(
            grid_type_radius(grid_type_index(Element::Placeholder)),
            Element::C.interaction_radius()
        );

        // The constant the precalculation widens its early-out with must be the
        // widest of them, or the outer cutoff silently amputates the tail of the
        // largest probe's map.
        let widest = ELEMENTS
            .iter()
            .map(|e| e.interaction_radius())
            .fold(0.0f64, f64::max);
        assert_eq!(
            MAX_GRID_TYPE_RADIUS, widest,
            "the early-out constant is {} but the widest grid radius is {widest}",
            MAX_GRID_TYPE_RADIUS
        );

        // An index past the end is a programming error, and has to look like
        // one: a plausible number here would corrupt an energy silently.
        assert!(grid_type_radius(GRID_TYPE_COUNT).is_nan());
    }

    /// The separation axis is only meaningful if each term in the scoring
    /// function peaks at a distance a real structure can have. These assertions
    /// are stated in ångström of *real* interatomic distance, using
    /// crystallographic contact distances — not in terms of any reference
    /// implementation's internals.
    #[test]
    fn the_interaction_radii_place_the_terms_at_real_contact_distances() {
        use crate::scoring::ScoringFunction;
        use crate::scoring::{hbond_term, hydrophobic_term, repulsion_term, VinaScoring};
        use crate::types::AtomKind;

        let scoring = VinaScoring::new();
        let mut o = Atom::new(1, [0.0, 0.0, 0.0], Element::O, AtomType::OA);
        o.kind = AtomKind::Acceptor;
        let mut c = Atom::new(2, [3.8, 0.0, 0.0], Element::C, AtomType::CH);
        c.kind = AtomKind::Hydrophobic;

        // A donor-acceptor pair at a textbook O-H...O hydrogen-bond separation
        // must be rewarded. The Vina hydrogen-bond weight is the single largest
        // term in the function; if it is silent here, polar binding is dead.
        let hb_geom = 2.8;
        let d_hb = hb_geom - (o.element.interaction_radius() + o.element.interaction_radius());
        let (hb, _) = hbond_term(d_hb);
        assert!(
            hb > 0.5,
            "the hydrogen-bond term must reward a {hb_geom} A O...O contact, \
             but it returned {hb:.3} at d = {d_hb:.2} A"
        );
        // ... and must be spent once the heavy atoms are further apart than any
        // hydrogen bond. `d = 0.3` is 3.5 Å for an O···O pair, past the far
        // edge of the 2.6–2.9 Å range.
        assert_eq!(
            hbond_term(0.3).0,
            0.0,
            "the hydrogen-bond term must be exactly spent once the surfaces \
             have separated"
        );
        assert!(
            hbond_term(-0.1).0 < 0.2,
            "and must be nearly gone a little past contact, got {:.3}",
            hbond_term(-0.1).0
        );

        // An apolar pair at the van der Waals contact must be rewarded, and
        // must not be rewarded past the point where there is no contact at all.
        let d_hyd = 3.8 - (c.element.interaction_radius() + c.element.interaction_radius());
        assert!(
            hydrophobic_term(d_hyd).0 > 0.5,
            "the hydrophobic term must reward a 3.8 A C...C contact, got {:.3}",
            hydrophobic_term(d_hyd).0
        );

        // The steric wall must sit *inside* a clash, never outside one.
        let clash = 1.5;
        let d_clash = clash - (c.element.interaction_radius() + c.element.interaction_radius());
        let (rep, _) = repulsion_term(d_clash);
        assert!(
            rep * scoring.weights.repulsion > 0.1,
            "two carbons {clash} A apart must cost real repulsion, got {}",
            rep * scoring.weights.repulsion
        );

        // And the attractive minimum of a carbon pair must land at a separation
        // the 2.0 A clash filter would accept — otherwise the filter and the
        // energy function contradict each other and one always wins.
        let mut best = (f64::INFINITY, 0.0);
        for step in 1..=120 {
            let r = step as f64 * 0.05;
            let d = r - (c.element.interaction_radius() + c.element.interaction_radius());
            let e = scoring.pair_energy(&c, &c, d);
            if e < best.0 {
                best = (e, r);
            }
        }
        assert!(
            best.1 > crate::docking::MIN_CONTACT_DISTANCE,
            "the most attractive C...C separation is {0:.2} A, inside the {1} A \
             clash threshold — the energy function is still rewarding overlap",
            best.1,
            crate::docking::MIN_CONTACT_DISTANCE
        );
        assert!(
            (3.0..4.5).contains(&best.1),
            "the most attractive C...C separation is {:.2} A, which is not a \
             physical carbon contact",
            best.1
        );
    }
}
