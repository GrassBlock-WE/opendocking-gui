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

#[cfg(test)]
mod gated_targets {
    //! The opt-in measurement target has to stay opt-in, and -- more to the
    //! point -- has to stay *visible* as opt-in.
    //!
    //! Measured on cargo 1.98.1: a target declared with `required-features` and
    //! not enabled is **omitted from `cargo test`'s output entirely**. There is
    //! no "skipping", no warning, no mention of the name. The run reports the
    //! lib target's count and stops, and a reader of that output cannot tell the
    //! difference between "the convergence measurement passed" and "the
    //! convergence measurement does not exist". That is the same failure shape
    //! this project has refused twice elsewhere: a count that cannot distinguish
    //! a check from its own absence.
    //!
    //! So the gate is asserted here, from inside the suite that the gate hides
    //! the target from. Deleting `required-features` from `Cargo.toml` -- the one
    //! edit that would silently re-admit a minutes-long target to every plain
    //! `cargo test` -- turns this red. It cannot be dropped by editing CI,
    //! because CI has nothing to edit: the step names a target that cargo builds
    //! or fails to build, and this is the half of the guarantee that survives
    //! someone never running the opt-in at all.

    /// The manifest as written, embedded at compile time so the assertion
    /// cannot be satisfied by a file that is not the one cargo read.
    const MANIFEST: &str = include_str!("../Cargo.toml");

    #[test]
    fn the_convergence_measurement_is_gated_by_a_named_feature() {
        assert!(
            MANIFEST.contains("convergence-budget = []"),
            "the `convergence-budget` feature is gone from dock-core/Cargo.toml, so \
             the convergence target is either running on every plain `cargo test` \
             (minutes, unoptimised) or has stopped existing. Measured behaviour of a \
             `required-features` target that is not enabled: cargo omits it from the \
             output with no message at all, so nothing else here would notice."
        );
        assert!(
            MANIFEST.contains("required-features = [\"convergence-budget\"]"),
            "the `convergence_budget` [[test]] target is no longer gated on the \
             feature. Without this line cargo auto-discovers tests/*.rs and every \
             plain `cargo test` pays for the docking searches, silently."
        );
    }

    #[test]
    fn the_gated_target_is_still_declared_rather_than_deleted() {
        // The gate and the target are two different edits. Someone can remove
        // `required-features` (caught above) or delete the whole `[[test]]`
        // stanza (caught here), and the second one is the more tempting: it looks
        // like tidying up a file nothing runs.
        assert!(
            MANIFEST.contains("name = \"convergence_budget\"")
                && MANIFEST.contains("path = \"tests/convergence_budget.rs\""),
            "the [[test]] stanza for convergence_budget is gone from \
             dock-core/Cargo.toml. The file is still on disk, so the only trace \
             would be a stale source file that nothing builds."
        );
    }
}
