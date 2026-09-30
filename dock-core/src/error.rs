//! Error types for the Open Docking core engine.

use thiserror::Error;

/// The result type used throughout `dock-core`.
pub type Result<T> = std::result::Result<T, DockError>;

/// All failures produced by the docking engine.
#[derive(Debug, Error)]
pub enum DockError {
    /// Underlying filesystem / stream failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A malformed record was encountered while reading a structure file.
    #[error("parse error (line {line}): {message}")]
    Parse {
        /// 1-based line number of the offending record.
        line: usize,
        /// Human-readable description of what was expected.
        message: String,
    },

    /// The structure is chemically inconsistent (e.g. no atoms, unconnected graph).
    #[error("invalid molecule: {0}")]
    InvalidMolecule(String),

    /// A grid / map could not be constructed or sampled.
    #[error("grid error: {0}")]
    Grid(String),

    /// The GPU backend failed to initialise or dispatch.
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    Gpu(String),

    /// An optimiser hit its iteration or tolerance budget without converging.
    #[error("optimiser did not converge: {0}")]
    Convergence(String),

    /// A user-supplied parameter was outside its documented domain.
    #[error("invalid parameter `{name}` = {value}: {reason}")]
    Parameter {
        /// Name of the offending parameter, e.g. `"spacing"`.
        name: String,
        /// The value that was supplied, rendered for display.
        value: String,
        /// Why that value is outside the documented domain.
        reason: String,
    },
}

impl DockError {
    /// Build a [`DockError::Parameter`] without spelling out the struct literal.
    pub fn param(
        name: impl Into<String>,
        value: impl std::fmt::Display,
        reason: impl Into<String>,
    ) -> Self {
        DockError::Parameter {
            name: name.into(),
            value: value.to_string(),
            reason: reason.into(),
        }
    }

    /// Build a [`DockError::Parse`] without spelling out the struct literal.
    pub fn parse(line: usize, message: impl Into<String>) -> Self {
        DockError::Parse {
            line,
            message: message.into(),
        }
    }

    /// Build a [`DockError::InvalidMolecule`] without spelling out the struct literal.
    pub fn molecule(message: impl Into<String>) -> Self {
        DockError::InvalidMolecule(message.into())
    }
}
