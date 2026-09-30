//! Python bindings for the Open Docking engine.
//!
//! Exposes the docking pipeline to Python through PyO3, with **zero-copy**
//! NumPy interop: every array that crosses the boundary as *input* is borrowed,
//! not copied.
//!
//! # The zero-copy contract
//!
//! | direction | path | copies |
//! |-----------|------|--------|
//! | Python → Rust, conformations | [`evaluate_conformations`] | **0** (borrowed `PyReadonlyArray2`) |
//! | Python → Rust, torsions | [`score`] | **0** (borrowed `PyReadonlyArray1`) |
//! | Rust → Python, pose coordinates | [`DockingResults::pose_coords`] | 1 (moved `Vec` into a new array) |
//!
//! The output path cannot be a borrow without handing out a Python object whose
//! lifetime is tied to the Rust result, which would be unsound to do
//! ergonomically. One move of `n × 3` doubles — a few kilobytes per pose — is
//! the right trade, and the returned array *owns* its data, so it stays valid
//! after the `DockingResults` is collected.
//!
//! # Example
//!
//! ```python
//! from opendocking import core
//!
//! receptor = core.Receptor.from_pdbqt("rec.pdbqt")
//! box = core.GridBox.from_center_size((0., 0., 0.), (22., 22., 22.))
//! maps = receptor.precalculate(box, "vina", 0.375)
//!
//! ligand = core.Ligand.from_pdbqt("lig.pdbqt")
//! result = core.dock(ligand, maps, exhaustiveness=8, num_modes=9, seed=42)
//!
//! print(result.best_energy)
//! coords = result.pose_coords(0)      # (n_atoms, 3) float64
//! result.write_pdbqt("poses.pdbqt")
//! ```

#![warn(missing_docs)]

use std::sync::Arc;

use dock_core::docking::{dock, DockingConfig, DockingResult, SearchMode};
use dock_core::grid::{GridBox, GridMaps};
use dock_core::kinematics::Conformation;
use dock_core::ligand::Ligand;
use dock_core::pdbqt::{parse_pdbqt, ParsedStructure};
use dock_core::receptor::Receptor;
use dock_core::scoring::{ScoringFunction, VinaScoring};
use dock_core::search::ScoringContext;
use dock_core::types::{Atom, Element, Molecule};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert a Rust error into a Python exception with a useful message.
fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Build a [`VinaScoring`] from its Python name.
fn scoring_from_name(name: &str) -> PyResult<VinaScoring> {
    match name.to_ascii_lowercase().as_str() {
        "vina" => Ok(VinaScoring::new()),
        "vinardo" => Ok(VinaScoring::vinardo()),
        other => Err(PyValueError::new_err(format!(
            "unknown scoring function {other:?}; expected 'vina' or 'vinardo'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// GridBox
// ---------------------------------------------------------------------------

/// An axis-aligned search box.
#[pyclass(name = "GridBox", module = "opendocking._dockpy")]
#[derive(Debug, Clone, Copy)]
pub struct PyGridBox {
    inner: GridBox,
}

#[pymethods]
impl PyGridBox {
    /// Build a box from its minimum and maximum corners.
    #[new]
    fn new(min: [f64; 3], max: [f64; 3]) -> PyResult<Self> {
        Ok(PyGridBox {
            inner: GridBox::new(min, max).map_err(to_py_err)?,
        })
    }

    /// Build a box of the given size centred on `center`.
    #[staticmethod]
    fn from_center_size(center: [f64; 3], size: [f64; 3]) -> PyResult<Self> {
        Ok(PyGridBox {
            inner: GridBox::centered(center, size).map_err(to_py_err)?,
        })
    }

    /// Minimum corner.
    #[getter]
    fn min(&self) -> [f64; 3] {
        self.inner.min
    }

    /// Maximum corner.
    #[getter]
    fn max(&self) -> [f64; 3] {
        self.inner.max
    }

    /// Edge lengths.
    #[getter]
    fn size(&self) -> [f64; 3] {
        self.inner.size()
    }

    /// Geometric centre.
    #[getter]
    fn center(&self) -> [f64; 3] {
        self.inner.center()
    }

    fn __repr__(&self) -> String {
        format!(
            "GridBox(min={:?}, max={:?})",
            self.inner.min, self.inner.max
        )
    }
}

// ---------------------------------------------------------------------------
// Receptor
// ---------------------------------------------------------------------------

/// A rigid receptor prepared for map precalculation.
#[pyclass(name = "Receptor", module = "opendocking._dockpy")]
pub struct PyReceptor {
    inner: Receptor,
}

#[pymethods]
impl PyReceptor {
    /// Read a receptor from a PDBQT file.
    #[staticmethod]
    fn from_pdbqt(path: std::path::PathBuf) -> PyResult<Self> {
        Ok(PyReceptor {
            inner: Receptor::from_pdbqt(path).map_err(to_py_err)?,
        })
    }

    /// Read a receptor from PDBQT text.
    #[staticmethod]
    fn from_pdbqt_str(text: &str) -> PyResult<Self> {
        Ok(PyReceptor {
            inner: Receptor::from_pdbqt_str(text).map_err(to_py_err)?,
        })
    }

    /// Number of atoms.
    #[getter]
    fn num_atoms(&self) -> usize {
        self.inner.num_atoms
    }

    /// Number of polar hydrogens carried explicitly.
    #[getter]
    fn num_polar_hydrogens(&self) -> usize {
        self.inner.num_polar_hydrogens
    }

    /// Atoms whose PDBQT type this engine does not recognise.
    ///
    /// Not an error — another tool may emit types AutoDock never defined — but
    /// such an atom contributes only its shape term, silently losing its
    /// hydrogen-bond and hydrophobic character.
    #[getter]
    fn unknown_atom_types(&self) -> usize {
        self.inner.unknown_atom_types()
    }

    /// Geometric centre of the receptor.
    #[getter]
    fn center(&self) -> [f64; 3] {
        self.inner.centroid()
    }

    /// Axis-aligned bounding box as `(min, max)`.
    #[getter]
    fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        self.inner.bounding_box()
    }

    /// Precalculate the affinity maps for a search box.
    #[pyo3(signature = (box_, scoring = "vina".to_string(), spacing = 0.375))]
    fn precalculate(
        &self,
        box_: &PyGridBox,
        scoring: String,
        spacing: f64,
    ) -> PyResult<PyGridMaps> {
        let sc = scoring_from_name(&scoring)?;
        let maps = self
            .inner
            .precalculate(&box_.inner, &sc, spacing)
            .map_err(to_py_err)?;
        Ok(PyGridMaps {
            inner: Arc::new(maps),
        })
    }

    /// Estimate the memory the maps for `box_` will occupy, in megabytes.
    ///
    /// Worth calling before precalculating on a laptop: a generous 40 Å box at
    /// 0.375 Å spacing is already tens of megabytes per map set.
    fn estimate_memory_mb(&self, box_: &PyGridBox, spacing: f64) -> f64 {
        let dims = dock_core::grid::estimate_dims(&box_.inner, spacing);
        let points = (dims[0] as f64) * (dims[1] as f64) * (dims[2] as f64);
        // 10 element types × 4 slots × 4 bytes (f32)
        points * 10.0 * 4.0 * 4.0 / (1024.0 * 1024.0)
    }
}

// ---------------------------------------------------------------------------
// GridMaps
// ---------------------------------------------------------------------------

/// Precalculated receptor affinity maps.
#[pyclass(name = "GridMaps", module = "opendocking._dockpy")]
pub struct PyGridMaps {
    inner: Arc<GridMaps>,
}

#[pymethods]
impl PyGridMaps {
    /// Grid dimensions `[nx, ny, nz]`.
    #[getter]
    fn dims(&self) -> [usize; 3] {
        self.inner.dims()
    }

    /// The raw map storage as a flat float32 array.
    ///
    /// Layout is `((ix + nx·(iy + ny·iz)) · STRIDE) + type·4 + slot`, with
    /// `STRIDE = GRID_TYPE_COUNT · MAPS_PER_TYPE = 40`. Exposed so a caller
    /// can write the grid out, plot a slice of it, or compare two grids
    /// exactly — which is how the thread-safety of the precalculation loop is
    /// checked.
    fn raw_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.raw_slice().to_vec().into_pyarray(py)
    }

    /// Grid spacing in Ångström.
    #[getter]
    fn spacing(&self) -> f64 {
        self.inner.spacing()
    }

    /// Number of tabulated grid points.
    #[getter]
    fn num_points(&self) -> usize {
        self.inner.point_count()
    }

    /// Approximate memory footprint, in megabytes.
    #[getter]
    fn memory_mb(&self) -> f64 {
        (self.inner.data_len() * 4) as f64 / (1024.0 * 1024.0)
    }

    /// The box these maps cover.
    #[getter]
    fn box_(&self) -> PyGridBox {
        PyGridBox {
            inner: self.inner.grid_box(),
        }
    }

    /// Write AutoDock-compatible `.map` files into `directory`.
    fn write_map_files(&self, directory: std::path::PathBuf) -> PyResult<()> {
        self.inner
            .write_autodock_map_files(directory)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Ligand
// ---------------------------------------------------------------------------

/// A prepared docking ligand.
#[pyclass(name = "Ligand", module = "opendocking._dockpy")]
pub struct PyLigand {
    inner: Arc<Ligand>,
}

#[pymethods]
impl PyLigand {
    /// Read a ligand from a PDBQT file.
    #[staticmethod]
    fn from_pdbqt(path: std::path::PathBuf) -> PyResult<Self> {
        Ok(PyLigand {
            inner: Arc::new(Ligand::from_pdbqt(path).map_err(to_py_err)?),
        })
    }

    /// Read a ligand from PDBQT text.
    #[staticmethod]
    fn from_pdbqt_str(text: &str) -> PyResult<Self> {
        // `parse_pdbqt`, not `read_pdbqt`: the latter takes a *path* and
        // opens a file. Passing the PDBQT text to it made this entry point
        // fail on every input -- on Windows with os error 123 ("the
        // filename ... is incorrect", because a PDBQT is full of newlines),
        // on Linux by looking for a file whose name is the whole document.
        // Nothing caught it: every caller passed malformed text and saw a
        // catchable ValueError either way, which is exactly what it wanted.
        let parsed: ParsedStructure = parse_pdbqt(text).map_err(to_py_err)?;
        let torsions = parsed.active_torsion_pairs();
        Ok(PyLigand {
            inner: Arc::new(
                Ligand::from_molecule_with(parsed.molecule, &torsions).map_err(to_py_err)?,
            ),
        })
    }

    /// Build a ligand from an element/charge/coordinate description.
    ///
    /// This is the path used by the RDKit front-end: RDKit decides the
    /// chemistry (aromaticity, protonation, Gasteiger charges) and hands over a
    /// plain table, so the engine never has to re-derive it.
    #[staticmethod]
    #[pyo3(signature = (elements, charges, coords, bonds = None, atom_names = None))]
    fn from_arrays(
        elements: Vec<String>,
        charges: Vec<f64>,
        coords: PyReadonlyArray2<'_, f64>,
        bonds: Option<Vec<(usize, usize)>>,
        atom_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let n = elements.len();
        if charges.len() != n {
            return Err(PyValueError::new_err(format!(
                "elements has {n} entries but charges has {}",
                charges.len()
            )));
        }
        let coords = coords.as_array();
        if coords.shape()[0] != n || coords.shape()[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "coords must have shape ({n}, 3), got {:?}",
                coords.shape()
            )));
        }
        let mut atoms: Vec<Atom> = Vec::with_capacity(n);
        for i in 0..n {
            let element = match elements[i].trim() {
                "C" => Element::C,
                "N" => Element::N,
                "O" => Element::O,
                "S" => Element::S,
                "P" => Element::P,
                "F" => Element::F,
                "Cl" => Element::Cl,
                "Br" => Element::Br,
                "I" => Element::I,
                "H" => Element::H,
                "Mg" | "Zn" | "Fe" | "Ca" | "Mn" => Element::Met,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported element {other:?}"
                    )))
                }
            };
            let mut atom = Atom::new(
                i as u32 + 1,
                [coords[[i, 0]], coords[[i, 1]], coords[[i, 2]]],
                element,
                default_atom_type(element),
            );
            atom.charge = charges[i] as f32;
            if let Some(names) = &atom_names {
                if let Some(name) = names.get(i) {
                    atom.name = name.clone();
                }
            }
            atoms.push(atom);
        }

        let mut mol = Molecule::from_atoms(atoms).map_err(to_py_err)?;
        if let Some(pairs) = bonds {
            // The caller's graph *replaces* the distance-perceived one. Appending
            // it instead would leave every bond twice, and a duplicated
            // rotatable bond makes the cluster graph cyclic — which then breaks
            // the invariant that one torsion owns exactly one child cluster.
            mol.bonds.clear();
            mol.neighbors = vec![Vec::new(); n];
            for (i, j) in pairs {
                if i < n && j < n && i != j {
                    if mol.neighbors[i].contains(&j) {
                        continue; // the same bond twice in the caller's table
                    }
                    mol.bonds.push(dock_core::types::Bond::new(i, j));
                    mol.neighbors[i].push(j);
                    mol.neighbors[j].push(i);
                }
            }
            // Ring membership drives rotatable-bond perception, so recompute it
            // now that the graph is final.
            mol.assign_ring_membership().map_err(to_py_err)?;
        }
        mol.assign_vina_atom_kinds();
        Ok(PyLigand {
            inner: Arc::new(Ligand::from_molecule(mol).map_err(to_py_err)?),
        })
    }

    /// Number of atoms.
    #[getter]
    fn num_atoms(&self) -> usize {
        self.inner.len()
    }

    /// Number of rotatable bonds.
    #[getter]
    fn num_torsions(&self) -> usize {
        self.inner.num_torsions()
    }

    /// Total degrees of freedom: 6 rigid-body plus one per torsion.
    #[getter]
    fn num_dof(&self) -> usize {
        self.inner.ndof()
    }

    /// Approximate radius of the ligand about its centroid, in Ångström.
    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius()
    }

    /// The interaction class of every atom, as a list of strings.
    #[getter]
    fn atom_kinds(&self) -> Vec<String> {
        self.inner
            .molecule
            .atoms
            .iter()
            .map(|a| format!("{:?}", a.kind).to_lowercase())
            .collect()
    }

    /// Reference coordinates as a fresh `(n, 3)` array.
    fn reference_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self
            .inner
            .molecule
            .atoms
            .iter()
            .flat_map(|a| a.coord)
            .collect();
        let n = self.inner.len();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("shape (n, 3) matches the flattened coordinates");
        arr.into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Ligand(num_atoms={}, num_torsions={})",
            self.inner.len(),
            self.inner.num_torsions()
        )
    }
}

/// The PDBQT type RDKit-side typing should default to for an element.
fn default_atom_type(e: Element) -> dock_core::types::AtomType {
    use dock_core::types::AtomType as T;
    match e {
        Element::C => T::CH,
        Element::N => T::NP,
        Element::O => T::OP,
        Element::S => T::SP,
        Element::P => T::PP,
        Element::F => T::FH,
        Element::Cl => T::ClH,
        Element::Br => T::BrH,
        Element::I => T::IH,
        Element::H => T::HD,
        _ => T::MetD,
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The outcome of a docking run.
#[pyclass(name = "DockingResults", module = "opendocking._dockpy")]
pub struct PyDockingResults {
    result: Arc<DockingResult>,
    ligand: Arc<Ligand>,
}

#[pymethods]
impl PyDockingResults {
    /// Number of distinct poses.
    #[getter]
    fn num_poses(&self) -> usize {
        self.result.poses.len()
    }

    /// Energy of the best pose, in kcal/mol.
    #[getter]
    fn best_energy(&self) -> f64 {
        self.result
            .poses
            .first()
            .map(|p| p.energy)
            .unwrap_or(f64::NAN)
    }

    /// All pose energies, best first.
    #[getter]
    fn energies(&self) -> Vec<f64> {
        self.result.poses.iter().map(|p| p.energy).collect()
    }

    /// All intermolecular energies, best first.
    #[getter]
    fn intermolecular_energies(&self) -> Vec<f64> {
        self.result.poses.iter().map(|p| p.intermolecular).collect()
    }

    /// RMSD of each pose to the best one, best first.
    #[getter]
    fn rmsds(&self) -> Vec<f64> {
        self.result
            .poses
            .iter()
            .map(|p| p.rmsd.unwrap_or(0.0))
            .collect()
    }

    /// Wall-clock seconds spent searching.
    #[getter]
    fn elapsed_seconds(&self) -> f64 {
        self.result.elapsed_seconds
    }

    /// Number of raw conformations produced before clustering.
    #[getter]
    fn raw_pose_count(&self) -> usize {
        self.result.raw_pose_count
    }

    /// Reported poses that overlap the receptor.
    ///
    /// Non-zero means the search never found a physically possible pose and
    /// the engine fell back to reporting the best of what it had. That is
    /// almost always a sign the search box contains solid protein rather than
    /// a pocket, or that `exhaustiveness` is too low.
    #[getter]
    fn rejected_pose_count(&self) -> usize {
        self.result.rejected_pose_count
    }

    /// Name of the scoring function used.
    #[getter]
    fn scoring_function(&self) -> String {
        self.result.scoring_function.clone()
    }

    /// Coordinates of pose `index` as a fresh `(n_atoms, 3)` array.
    fn pose_coords<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pose = self
            .result
            .poses
            .get(index)
            .ok_or_else(|| PyValueError::new_err(format!("pose index {index} out of range")))?;
        let n = pose.coords.len();
        let flat: Vec<f64> = pose.coords.iter().flat_map(|c| c.iter().copied()).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// The degree-of-freedom vector of pose `index`.
    ///
    /// Layout is `[tx, ty, tz, θx, θy, θz, τ₀, …]`, the same one
    /// `score_conformation` takes. Exposing it means a caller can ask *why* a
    /// pose scored what it did: re-score it, check that the analytic gradient
    /// vanishes there, or continue optimising from it.
    fn pose_conformation(&self, index: usize) -> PyResult<Vec<f64>> {
        let pose = self
            .result
            .poses
            .get(index)
            .ok_or_else(|| PyValueError::new_err(format!("pose index {index} out of range")))?;
        Ok(dock_core::search::conf_to_vec(
            dock_core::docking::pose_conformation(pose),
        ))
    }

    /// Every pose stacked into an `(n_poses, n_atoms, 3)` array.
    fn all_pose_coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let np = self.result.poses.len();
        let n = self.ligand.len();
        let mut flat = Vec::with_capacity(np * n * 3);
        for p in &self.result.poses {
            for c in &p.coords {
                flat.extend_from_slice(c);
            }
        }
        let arr = numpy::ndarray::Array3::from_shape_vec((np, n, 3), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Write the poses as a multi-model PDBQT file.
    fn write_pdbqt(&self, path: std::path::PathBuf) -> PyResult<()> {
        dock_core::docking::write_result(&self.result, &self.ligand, path)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Write the poses as an XYZ file, for viewers that read that format.
    fn write_xyz(&self, path: std::path::PathBuf) -> PyResult<()> {
        use std::io::Write;
        let file = std::fs::File::create(&path)
            .map_err(|e| PyIOError::new_err(format!("{}: {e}", path.display())))?;
        let mut w = std::io::BufWriter::new(file);
        let n = self.ligand.len();
        for (rank, pose) in self.result.poses.iter().enumerate() {
            writeln!(w, "{n}").map_err(to_py_err)?;
            writeln!(
                w,
                "pose {rank}  energy {:.3} kcal/mol  rmsd {:.2} A",
                pose.energy,
                pose.rmsd.unwrap_or(0.0)
            )
            .map_err(to_py_err)?;
            for (atom, c) in self.ligand.molecule.atoms.iter().zip(pose.coords.iter()) {
                writeln!(
                    w,
                    "{:<3} {:>10.4} {:>10.4} {:>10.4}",
                    atom.element.symbol(),
                    c[0],
                    c[1],
                    c[2]
                )
                .map_err(to_py_err)?;
            }
        }
        w.flush().map_err(to_py_err)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "DockingResults(num_poses={}, best_energy={:.3})",
            self.result.poses.len(),
            self.best_energy()
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Dock a ligand against precalculated maps.
///
/// `exhaustiveness` is the number of independent search walks (AutoDock's
/// `--exhaustiveness`); higher finds better poses and costs proportionally
/// more. `mode` is `"mc"`, `"lga"` or `"both"`.
#[pyfunction(name = "dock")]
#[pyo3(signature = (
    ligand,
    maps,
    exhaustiveness = 8,
    num_modes = 9,
    rmsd_cutoff = 1.0,
    seed = None,
    mode = "mc".to_string(),
    scoring = None,
    steps = None,
))]
#[allow(clippy::too_many_arguments)]
fn dock_py(
    ligand: &PyLigand,
    maps: &PyGridMaps,
    exhaustiveness: u32,
    num_modes: usize,
    rmsd_cutoff: f64,
    seed: Option<u64>,
    mode: String,
    scoring: Option<String>,
    steps: Option<u32>,
) -> PyResult<PyDockingResults> {
    let sc = match scoring {
        Some(name) => scoring_from_name(&name)?,
        // The maps carry no record of which function built them, so fall back
        // to the default and let the caller notice a mismatch via the energies.
        None => VinaScoring::new(),
    };
    let search_mode = match mode.to_ascii_lowercase().as_str() {
        "mc" | "montecarlo" | "monte_carlo" => SearchMode::MonteCarlo,
        "lga" | "island" => SearchMode::IslandLga,
        "both" => SearchMode::Both,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown mode {other:?}; expected 'mc', 'lga' or 'both'"
            )))
        }
    };

    let mut config = DockingConfig {
        mode: search_mode,
        num_modes,
        rmsd_cutoff,
        ..DockingConfig::default()
    };
    config.monte_carlo.exhaustiveness = exhaustiveness;
    config.monte_carlo.seed = seed;
    config.lga.seed = seed;
    if let Some(s) = steps {
        config.monte_carlo.steps = s;
    }

    let result = dock(&ligand.inner, &maps.inner, &sc, &config).map_err(to_py_err)?;
    Ok(PyDockingResults {
        result: Arc::new(result),
        ligand: Arc::clone(&ligand.inner),
    })
}

/// Evaluate the docking energy of many conformations at once.
///
/// `conformations` is a borrowed `(n_conf, 6 + num_torsions)` float64 array in
/// the packed layout `[tx, ty, tz, θx, θy, θz, τ₀, …]`. The returned energies
/// are a fresh `(n_conf,)` array. **No input data is copied.**
#[pyfunction]
#[pyo3(signature = (ligand, maps, scoring, conformations, use_gpu=false, report_backend=None))]
fn evaluate_conformations<'py>(
    py: Python<'py>,
    ligand: &PyLigand,
    maps: &PyGridMaps,
    scoring: &str,
    conformations: PyReadonlyArray2<'_, f64>,
    use_gpu: bool,
    report_backend: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sc = scoring_from_name(scoring)?;
    let arr = conformations.as_array();
    let n_conf = arr.shape()[0];
    let n_torsions = ligand.inner.num_torsions();
    let n_dof = 6 + n_torsions;
    if arr.shape()[1] != n_dof {
        return Err(PyValueError::new_err(format!(
            "conformations must have shape (n, {n_dof}), got {:?}",
            arr.shape()
        )));
    }
    let confs: Vec<Conformation> = (0..n_conf)
        .map(|i| Conformation {
            position: [arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]],
            orientation: [arr[[i, 3]], arr[[i, 4]], arr[[i, 5]]],
            torsions: (0..n_torsions).map(|t| arr[[i, 6 + t]]).collect(),
        })
        .collect();
    for (i, c) in confs.iter().enumerate() {
        let mut flat = Vec::with_capacity(6 + n_torsions);
        flat.extend_from_slice(&c.position);
        flat.extend_from_slice(&c.orientation);
        flat.extend_from_slice(&c.torsions);
        check_finite(&flat, &format!("conformations[{i}]"))?;
    }

    // The batched path hands the per-atom grid interpolation to the compute
    // kernel when it can. Falling back is not a silent event: a caller who
    // asked for the GPU and quietly got the CPU would read it as an
    // unexplained slowdown, so the outcome is reported back explicitly.
    let (energies, backend, skip) =
        dock_core::search::evaluate_population(&ligand.inner, &maps.inner, &sc, &confs, use_gpu);

    if let Some(dict) = report_backend {
        let (name, adapter) = match &backend {
            dock_core::search::PopulationBackend::Cpu => ("cpu", None),
            dock_core::search::PopulationBackend::Gpu { adapter } => ("gpu", Some(adapter)),
        };
        dict.set_item("backend", name)?;
        match adapter {
            Some(a) => {
                dict.set_item("adapter", a.as_str())?;
            }
            None => {
                dict.set_item("adapter", py.None())?;
            }
        }
        match &skip {
            Some(s) => {
                dict.set_item("gpu_skip_reason", s.reason)?;
            }
            None => {
                dict.set_item("gpu_skip_reason", py.None())?;
            }
        }
        dict.set_item("num_conformations", n_conf)?;
    }

    Ok(energies.into_pyarray(py))
}

/// Check a packed degree-of-freedom vector before it reaches the optimiser.
///
/// A NaN or infinite component propagates silently: the interpolation returns
/// `None`, the atom is treated as "outside the box", and the caller gets a
/// large finite penalty instead of a recognisable mistake. Rejecting here
/// turns a wrong answer into a catchable error.
fn check_finite(values: &[f64], what: &str) -> PyResult<()> {
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{what}[{i}] is {}; every degree of freedom must be finite",
            values[i]
        )));
    }
    Ok(())
}

/// Energy and analytic gradient of a single conformation.
///
/// `torsions` is borrowed, not copied. Returns `(energy, gradient)` where the
/// gradient is a fresh list of `6 + num_torsions` floats.
#[pyfunction]
#[pyo3(signature = (ligand, maps, scoring, position, orientation, torsions))]
fn score(
    ligand: &PyLigand,
    maps: &PyGridMaps,
    scoring: &str,
    position: [f64; 3],
    orientation: [f64; 3],
    torsions: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, Vec<f64>)> {
    let sc = scoring_from_name(scoring)?;
    if torsions.len() != ligand.inner.num_torsions() {
        return Err(PyValueError::new_err(format!(
            "expected {} torsions, got {}",
            ligand.inner.num_torsions(),
            torsions.len()
        )));
    }
    let conf = Conformation {
        position,
        orientation,
        torsions: torsions.as_slice()?.to_vec(),
    };
    let mut flat = Vec::with_capacity(6 + ligand.inner.num_torsions());
    flat.extend_from_slice(&position);
    flat.extend_from_slice(&orientation);
    flat.extend_from_slice(&conf.torsions);
    check_finite(&flat, "conformation")?;
    let mut ctx = ScoringContext::new(&ligand.inner, &maps.inner, &sc);
    Ok(ctx.evaluate(&conf))
}

/// Version of the engine, as a string.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A human-readable description of the available scoring functions.
#[pyfunction]
fn scoring_functions() -> Vec<String> {
    vec![
        VinaScoring::new().description(),
        VinaScoring::vinardo().description(),
    ]
}

/// Whether the GPU backend was compiled in and an adapter could be opened.
///
/// The Python front-end uses this to decide whether to advertise GPU support,
/// so it must not raise when the feature is absent.
#[pyfunction]
fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        dock_core::gpu::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// `True` if this build includes the `wgpu` compute backend.
#[pyfunction]
fn gpu_compiled() -> bool {
    cfg!(feature = "gpu")
}

/// World-space coordinates of a conformation.
///
/// Forward kinematics turns the degree-of-freedom vector into Cartesian
/// coordinates. Exposing it means a caller can *look at* an arbitrary
/// conformation — to draw it, to measure distances, or to check for steric
/// clashes — without having to run a search first.
#[pyfunction]
#[pyo3(signature = (ligand, conformation))]
fn conformation_coordinates<'py>(
    py: Python<'py>,
    ligand: &PyLigand,
    conformation: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = conformation.as_array();
    let n_dof = 6 + ligand.inner.num_torsions();
    if arr.len() != n_dof {
        return Err(PyValueError::new_err(format!(
            "conformation must have {n_dof} entries, got {}",
            arr.len()
        )));
    }
    let conf = Conformation {
        position: [arr[0], arr[1], arr[2]],
        orientation: [arr[3], arr[4], arr[5]],
        torsions: (0..ligand.inner.num_torsions())
            .map(|t| arr[6 + t])
            .collect(),
    };
    let coords = ligand.inner.tree.coordinates(&conf);
    let flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();
    let arr = numpy::ndarray::Array2::from_shape_vec((coords.len(), 3), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// The native extension module.
#[pymodule]
fn _dockpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGridBox>()?;
    m.add_class::<PyReceptor>()?;
    m.add_class::<PyGridMaps>()?;
    m.add_class::<PyLigand>()?;
    m.add_class::<PyDockingResults>()?;
    m.add_function(wrap_pyfunction!(dock_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_conformations, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(scoring_functions, m)?)?;
    m.add_function(wrap_pyfunction!(conformation_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_compiled, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
