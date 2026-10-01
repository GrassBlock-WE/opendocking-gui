//! Cross-platform GPU compute backend (`wgpu` / WGSL).
//!
//! # What runs on the GPU
//!
//! The inner loop of a docking search is *"for each of thousands of ligand
//! conformations, interpolate the receptor grid at every atom and sum the
//! weighted values"*. That is embarrassingly parallel over conformations and
//! is exactly what a compute shader is good at, so it is what this module
//! moves to the GPU.
//!
//! Deliberately **not** on the GPU: the local optimiser. L-BFGS is
//! sequential — every step depends on the last — and the per-conformation work
//! is small, so shipping one conformation at a time to a device would be
//! latency-bound. The GPU accelerates *scoring a large batch*, which is what
//! the genetic algorithm and the Monte-Carlo candidate pool need.
//!
//! # Portability
//!
//! `wgpu` targets Vulkan, Metal and DirectX 12 from one API, so this backend
//! runs on Linux, macOS and Windows without a CUDA toolchain. The feature is
//! **opt-in** (`--features gpu`) because it substantially increases build time
//! and requires a working adapter; the CPU engine is fully functional without
//! it.
//!
//! # Provenance
//!
//! The batched grid-interpolation idea follows AutoDock-GPU (Morris et al.,
//! *JCAMD* **25**, 10 (2011), LGPL-2.1). The WGSL kernel and the `wgpu`
//! integration are original to this project.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::error::Result;

/// One workgroup size. 64 is a good default: it maps to two 32-wide warps on
/// every major backend, so no subgroup capability is assumed.
///
/// This number appears three times in `energy.wgsl` — the workgroup size
/// attribute, the shared-memory reduction array, and the initial binary-tree
/// stride — and a mismatch is not a compile error in any of those positions,
/// so `shader_and_rust_agree_on_the_workgroup_size` binds them together.
pub const WORKGROUP_SIZE: u32 = 64;

/// The WGSL compute shader that scores a batch of conformations.
///
/// Grid layout matches [`crate::grid::GridMaps`]: `value[((ix + nx*(iy +
/// ny*iz)) * STRIDE) + type*SLOTS + slot]`, with `STRIDE = 10 * SLOTS` and
/// `SLOTS = 4`.
pub const ENERGY_WGSL: &str = include_str!("energy.wgsl");

/// `u32` values of packed per-atom data: element index, then the four map
/// weights as bit-cast floats, padded to a 32-byte stride.
///
/// The types and the weights share one buffer because wgpu's *downlevel*
/// default allows only four storage buffers per shader stage and this kernel
/// would need five. `energy.wgsl` reads the same layout.
pub const ATOM_STRIDE: usize = 8;

/// A batch of pre-computed atom coordinates to score.
///
/// Forward kinematics runs on the CPU: torsions form a recursive tree, so
/// evaluating them per-conformation on the GPU would need one workgroup per
/// conformation rather than one per atom, and would serialise the recursion.
/// What *is* worth moving to the device is the part that is embarrassingly
/// parallel and bandwidth-bound — interpolating the grid at every atom of every
/// conformation — which is exactly what this batch represents.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Flattened coordinates, `n_conf * n_atoms * 3` values, atom-major within
    /// each conformation.
    pub coords: Vec<f32>,
    /// Packed per-atom data, `n_atoms * ATOM_STRIDE` values.
    pub atom_data: Vec<u32>,
    /// Number of conformations.
    pub n_conf: usize,
    /// Number of atoms per conformation.
    pub n_atoms: usize,
}

impl Batch {
    /// Build a batch from flattened coordinates and a prepared ligand.
    pub fn from_coords(
        coords: Vec<f32>,
        n_conf: usize,
        ligand: &crate::ligand::Ligand,
    ) -> Result<Batch> {
        let n_atoms = ligand.len();
        if coords.len() != n_conf * n_atoms * 3 {
            return Err(gpu_err(&format!(
                "expected {} coordinate values, got {}",
                n_conf * n_atoms * 3,
                coords.len()
            )));
        }
        let mut atom_data = vec![0u32; n_atoms * ATOM_STRIDE];
        for i in 0..n_atoms {
            let base = i * ATOM_STRIDE;
            atom_data[base] = ligand.type_index[i] as u32;
            for (k, w) in ligand.weights[i].iter().enumerate() {
                atom_data[base + 1 + k] = w.to_bits();
            }
        }
        Ok(Batch {
            coords,
            atom_data,
            n_conf,
            n_atoms,
        })
    }
}

/// A GPU device and the buffers it owns.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Reused output buffer, grown on demand.
    energy_buffer: wgpu::Buffer,
    energy_capacity: usize,
    /// Human-readable adapter description, kept for diagnostics.
    adapter_label: String,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.adapter_label)
            .finish()
    }
}

impl GpuContext {
    /// Open the default adapter and compile the pipeline.
    ///
    /// Returns an error rather than panicking when no adapter exists — a
    /// machine without a usable GPU is normal, not exceptional, and the caller
    /// should fall back to the CPU engine.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| gpu_err("no suitable GPU adapter found"))?;

        let info = adapter.get_info();
        let adapter_label = if info.name.is_empty() {
            format!("{:?} backend", info.backend)
        } else {
            info.name.clone()
        };

        // wgpu 0.20 takes the device request as a descriptor plus a trace path.
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("opendocking"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|e| gpu_err(&format!("could not open a device: {e}")))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opendocking-energy"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ENERGY_WGSL)),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("opendocking-layout"),
            entries: &[
                binding(0, wgpu::BufferBindingType::Storage { read_only: false }, 4),
                binding(1, wgpu::BufferBindingType::Storage { read_only: true }, 4),
                binding(2, wgpu::BufferBindingType::Storage { read_only: true }, 4),
                binding(3, wgpu::BufferBindingType::Storage { read_only: true }, 4),
                // The uniform block is `GridParams`, which is 64 bytes once
                // the WGSL alignment padding is included. wgpu validates
                // this against the shader and refuses the pipeline if the
                // declared minimum is smaller.
                binding(4, wgpu::BufferBindingType::Uniform, 64),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opendocking-pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("opendocking-energy"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        const INITIAL_CAPACITY: usize = 4096;
        let energy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("opendocking-energies"),
            size: INITIAL_CAPACITY as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(GpuContext {
            device,
            queue,
            pipeline,
            bind_group_layout,
            energy_buffer,
            energy_capacity: INITIAL_CAPACITY,
            adapter_label,
        })
    }

    /// Name of the adapter in use, for diagnostics.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_label
    }

    /// Score a batch of conformations, returning one energy per conformation.
    ///
    /// The GPU currently reduces at most [`WORKGROUP_SIZE`] atoms per
    /// conformation. A larger ligand is rejected explicitly rather than
    /// silently mis-scored; the CPU path has no such limit and is what the
    /// search actually uses today.
    pub fn score(&mut self, batch: &Batch, grid: &crate::grid::GridMaps) -> Result<Vec<f32>> {
        if batch.n_conf == 0 || batch.n_atoms == 0 {
            return Ok(Vec::new());
        }
        if batch.n_atoms > WORKGROUP_SIZE as usize {
            return Err(gpu_err(&format!(
                "the GPU kernel handles at most {WORKGROUP_SIZE} atoms per conformation, got {}",
                batch.n_atoms
            )));
        }
        let out_bytes = batch.n_conf * 4;
        if out_bytes > self.energy_capacity {
            // Storage buffers must be a multiple of 4 bytes; round up so the
            // copy size stays valid for any batch size.
            let grown = out_bytes.next_multiple_of(4);
            self.energy_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("opendocking-energies"),
                size: grown as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.energy_capacity = grown;
        }

        let dims = grid.dims();
        let params = GridParams {
            nx: dims[0] as u32,
            ny: dims[1] as u32,
            nz: dims[2] as u32,
            n_atoms: batch.n_atoms as u32,
            n_conf: batch.n_conf as u32,
            _pad: [0; 3],
            min: [
                grid.min[0] as f32,
                grid.min[1] as f32,
                grid.min[2] as f32,
                0.0,
            ],
            spacing: [
                grid.spacing[0] as f32,
                grid.spacing[1] as f32,
                grid.spacing[2] as f32,
                0.0,
            ],
        };

        let conf_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opendocking-coords"),
                contents: bytemuck::cast_slice(&batch.coords),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let atom_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opendocking-atom-data"),
                contents: bytemuck::cast_slice(&batch.atom_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let grid_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opendocking-grid"),
                contents: bytemuck::cast_slice(grid.raw_slice()),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let param_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opendocking-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("opendocking-bind"),
            layout: &self.bind_group_layout,
            entries: &[
                binding_entry(0, &self.energy_buffer),
                binding_entry(1, &conf_buf),
                binding_entry(2, &atom_buf),
                binding_entry(3, &grid_buf),
                binding_entry(4, &param_buf),
            ],
        });

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("opendocking-readback"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("opendocking"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("opendocking-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per conformation, capped at 64 atoms each. A ligand
            // with more than 64 atoms is handled by looping inside the kernel
            // in a later revision; for now the batch is rejected rather than
            // silently mis-scored.
            pass.dispatch_workgroups(batch.n_conf as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.energy_buffer, 0, &readback, 0, out_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| gpu_err("the GPU readback channel closed unexpectedly"))?
            .map_err(|e| gpu_err(&format!("GPU readback failed: {e}")))?;

        let data = slice.get_mapped_range();
        // The mapped range is raw bytes; reinterpret them as the f32 the
        // shader wrote. `chunks_exact` never panics because the copy size is a
        // multiple of 4 by construction.
        let out: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(data);
        readback.unmap();
        Ok(out)
    }
}

/// Uniform block describing the grid and the batch shape.
///
/// The field order and the padding are dictated by the WGSL uniform address
/// space: `vec4<f32>` is 16-byte aligned, and the five leading `u32`s occupy
/// only 20 bytes, so the shader expects 12 bytes of padding before `min`. This
/// struct must match `struct GridParams` in `energy.wgsl` byte for byte; the
/// test below checks the offsets so a reordering cannot silently corrupt every
/// sampled coordinate.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    n_atoms: u32,
    n_conf: u32,
    /// Matches the 12-byte hole WGSL leaves before the first `vec4`.
    _pad: [u32; 3],
    min: [f32; 4],
    spacing: [f32; 4],
}

/// Whether a GPU backend is compiled in *and* an adapter can actually be opened.
pub fn is_available() -> bool {
    GpuContext::new().is_ok()
}

fn binding(
    index: u32,
    ty: wgpu::BufferBindingType,
    min_binding_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: index,
        visibility: wgpu::ShaderStages::COMPUTE,
        // wgpu 0.20 nests the buffer type inside `BindingType`.
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(min_binding_size),
        },
        count: None,
    }
}

// The output borrows from `buffer`, so both lifetimes are named rather than
// elided. Eliding the input while the return type carries the same lifetime
// compiles, but newer clippy (`mismatched_lifetime_syntaxes`) rejects it.
fn binding_entry<'a>(index: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(not(feature = "gpu"))]
fn gpu_err(msg: &str) -> crate::DockError {
    crate::DockError::param("gpu", "unavailable", msg)
}

#[cfg(feature = "gpu")]
fn gpu_err(msg: &str) -> crate::DockError {
    crate::DockError::Gpu(msg.to_string())
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn grid_params_matches_the_wgsl_uniform_layout() {
        // WGSL lays `vec4<f32>` out on a 16-byte boundary, so the five leading
        // `u32`s are followed by 12 bytes of padding. If this ever drifts, the
        // shader reads the wrong grid origin and every energy is quietly wrong
        // — which no runtime assertion would catch.
        let p = GridParams {
            nx: 0,
            ny: 0,
            nz: 0,
            n_atoms: 0,
            n_conf: 0,
            _pad: [0; 3],
            min: [0.0; 4],
            spacing: [0.0; 4],
        };
        assert_eq!(std::mem::size_of::<GridParams>(), 64);

        // `addr_of!` is a safe macro, so this needs no `unsafe` and the crate
        // keeps its `#![forbid(unsafe_code)]`.
        fn at<T>(f: *const T, base: usize) -> usize {
            f as usize - base
        }
        let base = std::ptr::addr_of!(p) as usize;
        assert_eq!(at(std::ptr::addr_of!(p.nx), base), 0);
        assert_eq!(at(std::ptr::addr_of!(p.ny), base), 4);
        assert_eq!(at(std::ptr::addr_of!(p.nz), base), 8);
        assert_eq!(at(std::ptr::addr_of!(p.n_atoms), base), 12);
        assert_eq!(at(std::ptr::addr_of!(p.n_conf), base), 16);
        assert_eq!(at(std::ptr::addr_of!(p.min), base), 32);
        assert_eq!(at(std::ptr::addr_of!(p.spacing), base), 48);
    }

    #[test]
    fn availability_check_does_not_panic() {
        // Either there is a device or there is not; the call must simply
        // return a bool and never panic or abort.
        let _ = is_available();
    }

    #[test]
    fn shader_and_rust_agree_on_the_grid_stride() {
        // `STRIDE` in the kernel is hard-coded; it must match
        // `GRID_TYPE_COUNT * MAPS_PER_TYPE` on this side or every lookup is
        // out by a map.
        let expected = crate::grid::map_stride();
        assert!(
            ENERGY_WGSL.contains(&format!("const STRIDE: u32 = {expected}u")),
            "energy.wgsl does not declare STRIDE = {expected}"
        );
    }

    #[test]
    fn shader_and_rust_agree_on_the_atom_stride() {
        assert!(
            ENERGY_WGSL.contains(&format!("const ATOM_STRIDE: u32 = {ATOM_STRIDE}u")),
            "energy.wgsl does not declare ATOM_STRIDE = {ATOM_STRIDE}"
        );
    }

    #[test]
    fn shader_and_rust_agree_on_the_workgroup_size() {
        assert!(
            ENERGY_WGSL.contains(&format!("@compute @workgroup_size({WORKGROUP_SIZE})")),
            "energy.wgsl workgroup_size differs from WORKGROUP_SIZE = {WORKGROUP_SIZE}"
        );
        assert!(
            ENERGY_WGSL.contains(&format!("array<f32, {WORKGROUP_SIZE}>")),
            "the shared-memory reduction array is not sized to WORKGROUP_SIZE"
        );
        assert!(
            ENERGY_WGSL.contains(&format!("var stride = {}u;", WORKGROUP_SIZE / 2)),
            "the reduction tree does not start at half the workgroup size"
        );
    }

    #[test]
    fn the_kernel_stays_within_the_downlevel_storage_buffer_limit() {
        // wgpu's downlevel default is four storage buffers per shader stage.
        // `wgpu` panics rather than erroring when a layout exceeds it, so the
        // count is checked here where the failure is readable.
        let storage_bindings = ENERGY_WGSL
            .lines()
            .filter(|l| l.contains("var<storage"))
            .count();
        assert!(
            storage_bindings <= 4,
            "energy.wgsl declares {storage_bindings} storage buffers, \
             the downlevel limit is 4"
        );
    }

    /// A small receptor with both apolar and polar atoms, so every map slot the
    /// kernel reads is non-zero.
    fn test_receptor() -> crate::types::Molecule {
        use crate::types::{Atom, AtomType, Element, Molecule};
        // Four carbons and two oxygens around the origin, 1.5 Å apart.
        let atoms: Vec<Atom> = vec![
            Atom::new(1, [-1.5, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(3, [1.5, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(4, [0.0, 1.5, 0.0], Element::C, AtomType::CH),
            Atom::new(5, [-0.75, 0.75, 1.2], Element::O, AtomType::OA),
            Atom::new(6, [0.75, 0.75, 1.2], Element::O, AtomType::OA),
        ];
        let n = atoms.len();
        // A permissive graph: the grid only needs coordinates and atom kinds.
        Molecule {
            atoms,
            bonds: Vec::new(),
            neighbors: vec![Vec::new(); n],
        }
    }

    #[test]
    fn the_cpu_and_gpu_paths_agree_where_the_difference_was_measured() {
        // The whole point of the compute shader is to reproduce the CPU
        // trilinear sum. A kernel that compiles but reads the wrong slot, the
        // wrong stride or the wrong grid origin still returns plausible-looking
        // numbers, so the only real check is a numerical one against the CPU
        // path on the same grid.
        //
        // What this test is and is not. It is a *measurement*, and it is the
        // only statement in the suite about the shader's arithmetic. It cannot
        // run on a machine with no adapter, and there are two honest ways to
        // deal with that: fail, or do not claim a result. Failing would make a
        // headless CI runner permanently red over something it cannot do, and
        // would make this project's counts a property of the machine they ran
        // on, which is the one thing a count must not be. So this test does not
        // fail, and it does not claim a result either — and the count below is
        // the same with and without an adapter because there is exactly one
        // `#[test]` either way, no `#[ignore]`, and no `skip()` to be counted.
        // What changes between the two machines is a **word in the output**,
        // never a number: that is the whole trick, and it is why the wording
        // below is this long and does not contain the word "pass".
        //
        // The structural claim that does hold everywhere is
        // `grid::tests::under_the_gpu_feature_the_terms_are_still_cpu_only`, in
        // the same feature arm: it reads this module's own `ENERGY_WGSL` and
        // asserts the kernel's stride is the production one, that the two
        // per-point strides differ, and that `TermMaps::BACKEND` is the CPU. It
        // needs no adapter, so it runs on a machine that cannot run this one.
        // Between them the arm says something true about the GPU path either
        // way — which is why this test does not have to assert a structural
        // fact of its own to avoid being vacuous, and why an earlier version of
        // this comment claiming it had to was wrong: that version described a
        // duplicate of a guard that already existed.
        if !is_available() {
            eprintln!(
                "NOT MEASURED: this machine has no GPU adapter, so the CPU/GPU \
                 difference was not computed and no comparison was made. This line \
                 reports an absence, not a result. What was still asserted on this \
                 machine, in this same arm, is the structural claim: \
                 `under_the_gpu_feature_the_terms_are_still_cpu_only` checks that \
                 the kernel's stride is the production one and that the two \
                 per-term and production layouts cannot be interchanged. The \
                 numerical claim above is the one that did not run here."
            );
            return;
        }
        use crate::grid::{GridBox, GridMaps, MapSlot};
        use crate::scoring::VinaScoring;

        let receptor = test_receptor();
        let scoring = VinaScoring::new();
        let box_ = GridBox::new([-6.0, -6.0, -6.0], [6.0, 6.0, 6.0]).expect("box");
        let maps = GridMaps::precalculate(&receptor, &box_, &scoring, 0.5, 0).expect("maps");

        // Two "conformations" of a two-atom fragment: one carbon, one oxygen.
        let ligand_atoms: Vec<crate::types::Atom> = vec![
            crate::types::Atom::new(
                1,
                [0.0, 0.0, 0.0],
                crate::types::Element::C,
                crate::types::AtomType::CH,
            ),
            crate::types::Atom::new(
                2,
                [1.2, 0.0, 0.0],
                crate::types::Element::O,
                crate::types::AtomType::OA,
            ),
        ];
        let n_atoms = ligand_atoms.len();
        let lig_mol = crate::types::Molecule {
            atoms: ligand_atoms,
            bonds: Vec::new(),
            neighbors: vec![Vec::new(); n_atoms],
        };
        let ligand = crate::ligand::Ligand::from_molecule_with(lig_mol, &[]).expect("ligand");

        // The sample. The two placements this test used to check are kept, and
        // 125 more are added on a 1.7 A lattice spanning -3.4..3.4 A, which is
        // off the 0.5 A grid everywhere except by accident -- so most of the
        // sample is a real trilinear blend rather than a node lookup, and the
        // spread includes the corners where the partial sums are largest and
        // the deep interior where they cancel hardest. Two conformations cannot
        // tell a stride error from a correct kernel: both are one number, and
        // the number is wrong in the same way every time. The size of the
        // sample is what makes "the worst difference" a statement about the
        // kernel rather than about one pose.
        let mut placements: Vec<[f32; 3]> = vec![[-1.0, -1.0, -1.0], [2.0, 1.5, 0.5]];
        let mut a = -3.4f32;
        while a <= 3.4 {
            let mut b = -3.4f32;
            while b <= 3.4 {
                let mut c = -3.4f32;
                while c <= 3.4 {
                    placements.push([a, b, c]);
                    c += 1.7;
                }
                b += 1.7;
            }
            a += 1.7;
        }
        // The far corner plus the 1.2 A offset on the second atom must still be
        // inside the box, or `interpolate_with_gradient` would refuse the point
        // and the CPU side of the comparison would never run.
        let reach = placements
            .iter()
            .map(|p| p[0] + 1.2)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            reach < 6.0,
            "the sample reaches x = {reach}, which is outside the 6 A box"
        );

        let mut coords: Vec<f32> = Vec::new();
        for p in &placements {
            coords.extend_from_slice(p);
            coords.extend_from_slice(&[p[0] + 1.2, p[1], p[2]]);
        }
        let n_conf = placements.len();
        let batch = Batch::from_coords(coords, n_conf, &ligand).expect("batch");

        let mut ctx = GpuContext::new().expect("gpu context");
        let got = ctx.score(&batch, &maps).expect("gpu score");
        assert_eq!(got.len(), n_conf, "one energy per conformation");

        // The same sum, on the CPU, from the same maps: the weighted trilinear
        // value of every atom.
        let mut want = Vec::with_capacity(n_conf);
        for p in &placements {
            let mut total = 0.0f64;
            for (i, _atom) in ligand.molecule.atoms.iter().enumerate() {
                let mut xyz = [p[0] as f64, p[1] as f64, p[2] as f64];
                xyz[0] += if i == 1 { 1.2 } else { 0.0 };
                let (e, _g) = maps
                    .interpolate_with_gradient(ligand.type_index[i], &ligand.weights[i], xyz)
                    .expect("point is inside the box");
                total += e;
            }
            want.push(total);
        }

        // The scale the tolerance is derived from: the magnitude of the `f32`
        // grid data the shader actually reads, over the eight corners of every
        // atom's cell, weighted by that atom's slot weights.
        //
        // The first version of this band scaled by the *interpolated* energy,
        // and the 1.7 A lattice is what showed it was wrong: a conformation whose
        // neighbouring nodes nearly cancel is evaluated from nodes orders of
        // magnitude larger than its own total, and the shader's `f32` weights are
        // applied to those nodes, not to the result. A band proportional to the
        // total collapses exactly where the kernel is hardest to get right, and
        // two hand-picked poses at 1.5 A and 3.0 A never reached such a point.
        // The data magnitude does not collapse: it is the only thing the `f32`
        // error can be proportional to, since both paths read the same values
        // and the representation error is common to the two.
        let corners = [
            [0usize, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        let mut read_magnitude = Vec::with_capacity(n_conf);
        for p in &placements {
            let mut magnitude = 0.0f64;
            for (i, _atom) in ligand.molecule.atoms.iter().enumerate() {
                let mut xyz = [p[0] as f64, p[1] as f64, p[2] as f64];
                xyz[0] += if i == 1 { 1.2 } else { 0.0 };
                // The cell this atom lands in, by the same arithmetic
                // `GridMaps::fractional` uses; it is private, and it is restated
                // here rather than reached through, so the two can be seen to
                // agree instead of one hiding the other.
                let mut cell = [0usize; 3];
                for (a, cell_a) in cell.iter_mut().enumerate() {
                    let u = (xyz[a] - maps.min[a]) / maps.spacing[a];
                    *cell_a = u.floor() as usize;
                }
                for c in corners {
                    for (s, slot) in MapSlot::ALL.iter().enumerate() {
                        let w = ligand.weights[i][s] as f64;
                        if w == 0.0 {
                            continue;
                        }
                        let v = maps.raw(
                            ligand.type_index[i],
                            *slot,
                            cell[0] + c[0],
                            cell[1] + c[1],
                            cell[2] + c[2],
                        );
                        magnitude += (v as f64 * w).abs();
                    }
                }
            }
            read_magnitude.push(magnitude);
        }

        // `ROUNDINGS` is the number of `f32` roundings a value passes through on
        // its way from a grid node to the conformation's energy: one per grid
        // value entering the sum. It is a count, not a fudge factor, and the two
        // things it counts are asserted next to it so it cannot drift away from
        // them silently.
        assert_eq!(
            crate::grid::MAPS_PER_TYPE,
            4,
            "ROUNDINGS counts four slots per corner"
        );
        assert_eq!(n_atoms, 2, "ROUNDINGS counts two atoms per conformation");
        const ROUNDINGS: f64 = 64.0; // 8 corners x 4 slots x 2 atoms
        let eps32 = f32::EPSILON as f64;

        let mut worst = 0.0f64;
        let mut worst_at = 0usize;
        for (c, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let diff = ((*g as f64) - *w).abs();
            let tol = ROUNDINGS * eps32 * read_magnitude[c];
            assert!(
                diff <= tol,
                "conformation {c}: GPU {g} vs CPU {w}, a difference of {diff:e} \
                 against a derived band of {tol:e} ({ROUNDINGS} f32 roundings of \
                 a read magnitude of {:e}); the worst seen so far is {worst:e} \
                 (adapter {})",
                read_magnitude[c],
                ctx.adapter_name()
            );
            if diff > worst {
                worst = diff;
                worst_at = c;
            }
        }
        // Report the number the assertion above only prints when it fails. A
        // cross-check that says "they agree" without saying by how much cannot
        // be used to notice the agreement getting worse.
        let widest = ROUNDINGS * eps32 * read_magnitude[worst_at];
        eprintln!(
            "MEASURED on adapter {}: {} conformations, worst |GPU - CPU| = \
             {worst:e} at conformation {worst_at}, against a derived band of \
             {widest:e} there -- the difference used {:.1}% of the band. This is \
             the number the no-adapter branch above declines to produce, so its \
             absence there is visible rather than implied",
            ctx.adapter_name(),
            n_conf,
            100.0 * worst / widest
        );
    }
}
