//! # Open Docking (`dock-core`)
//!
//! A modern, from-scratch molecular docking engine written in safe Rust.
//!
//! `dock-core` is the compute half of [Open Docking](https://github.com/GrassBlock-WE/opendocking-gui).
//! It provides:
//!
//! * [`types`] — atoms, elements, AutoDock atom types and the covalent graph.
//! * [`pdbqt`] — tolerant PDBQT reading and writing.
//! * [`kinematics`] — rigid clusters, the rotatable-bond tree and forward kinematics.
//! * [`grid`] — precalculated affinity maps and trilinear interpolation with analytic gradients.
//! * [`scoring`] — the Vina and Vinardo empirical scoring functions with first-order
//!   analytic derivatives.
//! * [`search`] — L-BFGS local optimisation, iterated local search and an island-model
//!   hybrid genetic algorithm.
//! * [`gpu`] (feature `gpu`) — a cross-platform `wgpu`/WGSL compute backend.
//!
//! ## Quick start
//!
//! ```no_run
//! use dock_core::prelude::*;
//!
//! // Pick a scoring function once and use it for both the maps and the search:
//! // grids tabulated with one weight set are meaningless under another.
//! let scoring = VinaScoring::new();
//!
//! // 1. Receptor and the search box.
//! let receptor = Receptor::from_pdbqt("receptor.pdbqt")?;
//! let box_ = GridBox::centered(receptor.centroid(), [22.0, 22.0, 22.0])?;
//! let maps = receptor.precalculate(&box_, &scoring, 0.375)?;
//!
//! // 2. Ligand: read the PDBQT, or build a `Ligand` from RDKit tables.
//! let ligand = Ligand::from_pdbqt("ligand.pdbqt")?;
//!
//! // 3. Search. `dock` returns distinct poses sorted best-first.
//! let cfg = DockingConfig::fast();
//! let result = dock(&ligand, &maps, &scoring, &cfg)?;
//!
//! println!("best affinity: {} kcal/mol", result.best().energy);
//! # Ok::<(), dock_core::DockError>(())
//! ```
//!
//! ## Threading
//!
//! Multi-ligand batches and genetic-algorithm population evolution are parallelised with
//! [`rayon`]. The engine is `Send + Sync` throughout; a [`Receptor`] and its [`GridMaps`]
//! can be shared across threads behind an `Arc` without copying.
//!
//! ## Licensing
//!
//! GPL-3.0-or-later. The algorithms re-implemented here derive from published work by the
//! AutoDock developers; see the module headers for the specific sources and licences.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

pub mod docking;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod grid;
pub mod kinematics;
pub mod ligand;
pub mod pdbqt;
pub mod receptor;
pub mod scoring;
pub mod search;
pub mod types;

pub use docking::{dock, DockingConfig, DockingResult, Pose, SearchMode};
pub use error::{DockError, Result};
pub use grid::{GridBox, GridMaps};
pub use kinematics::{Conformation, KinematicTree};
pub use ligand::Ligand;
pub use receptor::Receptor;
pub use scoring::{ScoringFunction, VinaScoring, VinaWeights};
pub use types::{Atom, AtomKind, AtomType, Bond, Element, Molecule, Vec3};

/// Frequently used types, re-exported for `use dock_core::prelude::*;`.
pub mod prelude {
    pub use crate::docking::{dock, DockingConfig, DockingResult, Pose, SearchMode};
    pub use crate::error::{DockError, Result};
    pub use crate::grid::{GridBox, GridMaps};
    pub use crate::kinematics::{Conformation, KinematicTree};
    pub use crate::ligand::Ligand;
    pub use crate::receptor::Receptor;
    pub use crate::scoring::{ScoringFunction, VinaScoring, VinaWeights};
    pub use crate::search::lbfgs::{LbfgsConfig, LbfgsOutcome};
    pub use crate::search::lga::LgaConfig;
    pub use crate::search::monte_carlo::MonteCarloConfig;
    pub use crate::types::{Atom, AtomKind, AtomType, Bond, Element, Molecule, Vec3};
}
