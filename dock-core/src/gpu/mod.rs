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
//! sequential —every step depends on the last —and the per-conformation work
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
/// This number appears three times in `energy.wgsl` —the workgroup size
/// attribute, the shared-memory reduction array, and the initial binary-tree
/// stride —and a mismatch is not a compile error in any of those positions,
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

/// `f32` values uploaded per scored point: two `vec4<f32>`.
///
/// The narrowed position and the host's in-box flag in the first, the host's
/// out-of-box violation in the second. `energy.wgsl`'s `Coord` is the same
/// pair, and the binding's `min_binding_size` is this number times four bytes.
pub const COORD_FLOATS: usize = 8;

/// A batch of pre-computed atom coordinates to score.
///
/// Forward kinematics runs on the CPU: torsions form a recursive tree, so
/// evaluating them per-conformation on the GPU would need one workgroup per
/// conformation rather than one per atom, and would serialise the recursion.
/// What *is* worth moving to the device is the part that is embarrassingly
/// parallel and bandwidth-bound —interpolating the grid at every atom of every
/// conformation —which is exactly what this batch represents.
///
/// # Why this carries `f64` coordinates as well as `f32` ones
///
/// The kernel's coordinate buffer is `f32`, so a `f64` position has to be
/// rounded to be uploaded, and the CPU scores the *unrounded* one. That is a
/// difference of a couple of bits in the input, which is normally harmless —/// except that the in-box test is a **discontinuous** function of it, so the
/// two backends could be handed coordinates one ULP apart and take different
/// branches: one interpolates a grid value and the other charges the
/// out-of-box penalty. That is a disagreement of the size of the map value
/// (~15.5 kcal/mol measured, against ~4.8e-4 for placements that agree on the
/// branch), not the size of a rounding error, and no tolerance derived from
/// `f32` epsilon can be wide enough to hold it without also admitting the
/// rounding differences it exists to police.
///
/// So the host takes the decision and hands the kernel the answer. In
/// [`GpuContext::score`], each atom's `f64` position is passed to
/// [`crate::grid::GridMaps::fractional`] —the same call
/// `ScoringContext::evaluate_full` makes, on the un-narrowed coordinate, in
/// `f64` —and the resulting in-box flag is uploaded beside the narrowed one.
/// The kernel branches on that flag. One evaluation, two consumers, so the
/// two backends cannot disagree about whether a point is in the box.
///
/// # Why the violation is uploaded too, and not only the flag
///
/// The flag and the size of the charge are one decision, and the host now
/// takes both. It measures the violation with
/// [`crate::grid::out_of_box_violation_per_axis`] -- the same function, in the
/// same per-axis form, on the same un-narrowed `f64` -- and uploads the sum
/// beside the flag, so the kernel multiplies the CPU's number by the CPU's
/// constant.
///
/// It used to recompute it, and the expression was right. What was wrong was
/// the input: the kernel's coordinate is `f32`, so the same expression on the
/// same box returned a different number. At `x = 3.000000119209` the CPU
/// charged `1000 x 1.19e-7` and the kernel charged `0`, because
/// `f32(3.000000119209)` rounds to exactly `3.0` and the narrowed coordinate
/// was then *on* the face. Both sides took the out-of-box branch; they agreed
/// on the function and disagreed on its argument by `eps32 x |p|`, which the
/// penalty amplified to `3.6e-4` kcal/mol.
///
/// That number is small, which is the problem. A tolerance wide enough to
/// admit it cannot police the ~`2e-4` the two backends legitimately differ by
/// inside the box, so it forced the whole kernel's band open. It was also a
/// live ranking input, because a batch scorer exists to order poses. It is
/// pre-existing rather than a regression -- byte-identical before and after the
/// in-box flag -- and it is now closed. `energy.wgsl` states the same thing
/// from the kernel side, and
/// `the_out_of_box_magnitude_is_the_hosts_and_putting_it_back_breaks_the_band`
/// holds the band it closed to.
///
/// What is left for the two sides to round independently is the interpolation
/// *arithmetic* and now the `f32` narrowing of the violation, where a
/// disagreement is `O(eps)` of the quantity being summed rather than `O(|p|)`
/// or `O(|g|)`. That is the distinction the whole change turns on, and
/// `energy.wgsl` states it again on the kernel side, where a reader will
/// actually need it.
///
/// # Why the host does not nudge the uploaded coordinate instead
///
/// It used to. The uploaded `f32` was walked in whole ULPs until a Rust
/// transcription of `sample()`'s `f32` expression agreed with the CPU's, on
/// the reasoning that a coordinate within a few ULPs (2e-6 Å) of the one the
/// CPU scored is within the rounding band. That depends on the transcription
/// reproducing the *device's* rounding of `(p - min) / spacing`, and it was
/// measured not to: the transcription put `u` at `17.999998093` against
/// `n - 1 = 18.0` where the kernel branched the other way, a ~2-ulp gap
/// consistent with the driver contracting the division into a reciprocal
/// multiply. A fix that holds only on an adapter that rounds the way the
/// transcription guessed, and only in the one place where being wrong is worth
/// 22.773 kcal/mol, is the same defect at a smaller scale. The walk and its
/// `RECONCILE_STEPS` bound are gone; there is nothing left to reconcile once
/// the kernel is not deciding anything.
///
/// Narrowing the *CPU* to `f32` instead would also make the two expressions
/// identical, and is the wrong trade: `u` then quantises to ~3.6e-7 Å
/// and the energy becomes a staircase, so finite-difference gradients stop
/// meaning anything. `GridMaps::fractional` records that experiment.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Flattened coordinates, `n_conf * n_atoms * 3` values, atom-major within
    /// each conformation, narrowed to `f32` for upload.
    pub coords: Vec<f32>,
    /// The same coordinates before narrowing, `n_conf * n_atoms * 3` values.
    ///
    /// Kept so [`GpuContext::score`] can take the in-box decision on the value
    /// the CPU scores, before narrowing, and upload it as a flag. This is the
    /// CPU's input, not a second source of truth: `coords` is derived from it
    /// by [`Batch::from_coords`] and nothing else writes either.
    pub coords64: Vec<f64>,
    /// Packed per-atom data, `n_atoms * ATOM_STRIDE` values.
    pub atom_data: Vec<u32>,
    /// Number of conformations.
    pub n_conf: usize,
    /// Number of atoms per conformation.
    pub n_atoms: usize,
}

impl Batch {
    /// Build a batch from flattened `f64` coordinates and a prepared ligand.
    ///
    /// Takes `f64` rather than `f32` so that the narrowing happens here, in one
    /// place, next to the un-narrowed copy that [`GpuContext::score`] needs to
    /// take the in-box decision on. A caller that narrowed first and handed
    /// over `f32` would leave nothing to decide from, and the flag would be
    /// computed from the very rounding the kernel used to disagree with.
    pub fn from_coords(
        coords: Vec<f64>,
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
        // The narrowing itself. It is the same `as f32` the search used to do
        // inline, and it is the reason this function takes `f64`.
        let coords32: Vec<f32> = coords.iter().map(|c| *c as f32).collect();
        let mut atom_data = vec![0u32; n_atoms * ATOM_STRIDE];
        for i in 0..n_atoms {
            let base = i * ATOM_STRIDE;
            atom_data[base] = ligand.type_index[i] as u32;
            for (k, w) in ligand.weights[i].iter().enumerate() {
                atom_data[base + 1 + k] = w.to_bits();
            }
        }
        Ok(Batch {
            coords: coords32,
            coords64: coords,
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

/// Everything [`GpuContext::score_with_violations`] uploads, built with **no
/// device in the call path**.
///
/// # Why this is a free function and not a method
///
/// Because until it was lifted out, every host-side decision on the GPU path
/// -- the in-box flag, the out-of-box magnitude, the narrowing, the parameter
/// block -- lived inside a method on a type that *cannot be constructed
/// without an adapter*. Not "was hard to test": unreachable. Every test that
/// asserted anything about those numbers was therefore behind an
/// `is_available()` early return, printed `NOT MEASURED`, and returned. On a
/// CI runner with no GPU, the whole host half of this path asserted nothing
/// numerical, and the project's GPU/CPU agreement claims rested on the one
/// machine that happened to have a device.
///
/// Splitting the construction out makes it callable, and therefore assertable,
/// unconditionally. It is the same code the production path runs -- this is a
/// move, not a second implementation, and a test that passed by agreeing with
/// a transcription would be worthless.
///
/// Private rather than `pub(crate)` because it returns [`GridParams`], which is
/// a private layout type, and the test module that calls it is a child of this
/// one. Nothing outside `gpu` needs to build an upload buffer.
///
/// # What a green here proves, and what it does not
///
/// **Proves, on any machine, with or without an adapter:**
///
/// * the uploaded in-box flag is exactly `0.0` or `1.0` and equals what
///   [`crate::grid::GridMaps::fractional`] says about the **un-narrowed**
///   `f64` coordinate -- the same call the CPU's own interpolation makes;
/// * the flag is a function of `coords64` and provably *not* of the narrowed
///   `coords`, demonstrated by holding `coords` bit-identical while varying
///   `coords64` and watching the flag move;
/// * the uploaded out-of-box magnitude is the `f64` the CPU would charge,
///   narrowed once, and is exactly `0.0` in the box where the CPU charges
///   nothing;
/// * the uploaded position is the `f32` nearest the `f64`, unaltered -- no
///   ULP nudging;
/// * the parameter block is the grid's own numbers, narrowed, in the layout
///   the shader reads.
///
/// **Does not prove, and cannot:**
///
/// * that the shader does the right arithmetic with any of it. The trilinear
///   weights, the workgroup reduction, the `u32` index arithmetic and the
///   `f32` rounding inside `sample()` are all device-side and all unverified
///   here. Every one of them is a place where correct inputs can produce a
///   wrong energy.
/// * that the buffers are *uploaded*, or bound, or dispatched. Nothing here
///   creates a buffer.
/// * anything at all about any particular adapter's rounding. A host number
///   that is right can still be misread by a driver.
///
/// So a reader who sees this file pass has learned that the host hands the
/// shader the right numbers. They have **not** learned that the GPU path is
/// verified, and the tests in this module that print `NOT MEASURED` on an
/// adapter-less machine are still the only thing standing between this crate
/// and an unchecked shader. The full residue is listed on
/// [`GpuContext::score_with_violations`].
fn pack_upload(
    batch: &Batch,
    grid: &crate::grid::GridMaps,
    violations: Option<&[f64]>,
) -> Result<(Vec<f32>, GridParams)> {
    let n_points = batch.n_conf * batch.n_atoms;
    if let Some(v) = violations {
        if v.len() != n_points {
            return Err(gpu_err(&format!(
                "expected {n_points} out-of-box violations, got {}",
                v.len()
            )));
        }
    }
    let box_ = grid.grid_box();
    let mut coords = Vec::with_capacity(n_points * COORD_FLOATS);
    for i in 0..n_points {
        let p = [
            batch.coords64[i * 3],
            batch.coords64[i * 3 + 1],
            batch.coords64[i * 3 + 2],
        ];
        let in_box = grid.fractional(p).is_some();
        let violation = match violations {
            Some(v) => v[i],
            None if in_box => 0.0,
            None => crate::grid::out_of_box_violation_per_axis(p, &box_)
                .iter()
                .sum(),
        };
        coords.push(batch.coords[i * 3]);
        coords.push(batch.coords[i * 3 + 1]);
        coords.push(batch.coords[i * 3 + 2]);
        // Exactly 0.0 or 1.0. Both are exact in `f32`, so the kernel's
        // `> 0.5` test cannot misread either.
        coords.push(if in_box { 1.0 } else { 0.0 });
        // The host's `f64` violation, narrowed. Narrowed on purpose and
        // not widened back: the kernel's only remaining disagreement on
        // this path is `f32` rounding *of the right number*, which is
        // relative to the penalty and is the same class of error the
        // in-box path spends on the grid values.
        coords.push(violation as f32);
        coords.push(0.0);
        coords.push(0.0);
        coords.push(0.0);
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
        bmax: [
            grid.grid_box().max[0] as f32,
            grid.grid_box().max[1] as f32,
            grid.grid_box().max[2] as f32,
            0.0,
        ],
    };
    Ok((coords, params))
}

impl GpuContext {
    /// Open the default adapter and compile the pipeline.
    ///
    /// Returns an error rather than panicking when no adapter exists —a
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
                // 32, not 4: the shader's `coords` is `array<Coord>`, and
                // `Coord` is a pair of `vec4<f32>` (position + in-box flag, and
                // the out-of-box violation), so its element stride is 32 bytes.
                // wgpu refuses a layout whose `min_binding_size` is smaller than
                // the stride of an unbound trailing array element.
                binding(1, wgpu::BufferBindingType::Storage { read_only: true }, 32),
                binding(2, wgpu::BufferBindingType::Storage { read_only: true }, 4),
                binding(3, wgpu::BufferBindingType::Storage { read_only: true }, 4),
                // The uniform block is `GridParams`, which is 80 bytes once
                // the WGSL alignment padding is included. wgpu validates
                // this against the shader and refuses the pipeline if the
                // declared minimum is smaller.
                binding(4, wgpu::BufferBindingType::Uniform, 80),
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
        self.score_with_violations(batch, grid, None)
    }

    /// [`Self::score`], with the host's out-of-box violation overridable.
    ///
    /// `violations` is `n_conf * n_atoms` values, one per point, in the same
    /// order as `batch.coords64`, and it is the number the kernel will multiply
    /// by `OUT_OF_BOX_PENALTY` instead of the one this function would have
    /// measured. `None` -- what [`Self::score`] always passes -- is the honest
    /// one.
    ///
    /// It exists for one caller, and it is a test: the mutation in
    /// `the_out_of_box_magnitude_is_the_hosts_and_putting_it_back_breaks_the_band`
    /// uses it to restore the defect this method's default removed, measuring
    /// the violation from the *narrowed* coordinate the way the kernel used to.
    /// That has to be reachable through the production upload path rather than
    /// through a second copy of the shader, or the test would be asserting that
    /// a transcription agrees with itself.
    ///
    /// `pub(crate)` rather than a public option for the same reason
    /// [`crate::grid::GridMaps::fractional`] is: the override has no honest use
    /// outside the crate, and a public one would be a way for a caller to make
    /// the two backends disagree on purpose.
    ///
    /// # What still needs a device, and whether CI can be given one
    ///
    /// The host half of this path is asserted with no device at all -- see
    /// [`pack_upload`], which is this method's own upload construction called
    /// directly. The device half is not, and the split is not a matter of
    /// effort:
    ///
    /// * `energy.wgsl`'s `f32` trilinear arithmetic and its eight weights;
    /// * the workgroup reduction over `partial[]`;
    /// * the `u32` flat index `idx * STRIDE` and every derived address;
    /// * the driver's own rounding, which is why the walk-and-reconcile this
    ///   kernel used to carry was removed rather than tightened.
    ///
    /// **The shader half cannot be verified without hardware, ever.** It is
    /// real GPU code; a software rasteriser can compile and run it, but a
    /// result from lavapipe or SwiftShader is evidence about *that*
    /// implementation's arithmetic, not about any adapter a user has, and the
    /// rounding differences that motivated the last two rounds of work are
    /// exactly the ones a software implementation would not reproduce. wgpu has
    /// no conformance suite, and `WGPU_BACKEND` cannot conjure a conformant
    /// device. A CI runner with a real GPU is a self-hosted runner or a
    /// dedicated GPU host, and this repository has neither; the
    /// `docker`/`linux` runners on GitHub Actions have no device. The honest
    /// options are a self-hosted GPU runner that is *not* a PR gate, or
    /// accepting the residue and saying so in prose -- which is what
    /// `docs/` exists for, and is a documentation task rather than a testing
    /// one.
    ///
    /// Until one of those happens, a green `--features gpu` run in CI means
    /// the host hands the shader correct numbers, and **nothing else**. Every
    /// `NOT MEASURED` line in this module's tests is that sentence, printed.
    pub(crate) fn score_with_violations(
        &mut self,
        batch: &Batch,
        grid: &crate::grid::GridMaps,
        violations: Option<&[f64]>,
    ) -> Result<Vec<f32>> {
        if batch.n_conf == 0 || batch.n_atoms == 0 {
            return Ok(Vec::new());
        }
        if batch.n_atoms > WORKGROUP_SIZE as usize {
            return Err(gpu_err(&format!(
                "the GPU kernel handles at most {WORKGROUP_SIZE} atoms per conformation, got {}",
                batch.n_atoms
            )));
        }
        // Before anything is uploaded, and before the pipeline is dispatched.
        // `energy.wgsl` computes `idx * STRIDE` in `u32` (checked against the
        // shader text by grid.rs's
        // `the_shader_flat_index_is_u32_so_this_limit_is_the_right_one` test, not
        // by a line number in a comment), so a grid the kernel cannot address
        // does not overflow in any way the driver would report -- it wraps, and
        // the kernel returns a plausible number read from the wrong part of the
        // buffer. `precalculate` keeps a built grid under the limit, but `dims`
        // is public and the type is `Deserialize`, so this is the only place
        // that can catch one assembled by hand.
        grid.check_gpu_index_range()?;
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

        // The two decisions the host makes about each point, taken once each
        // and uploaded beside the position they were taken for. See `Batch`
        // for why they cannot be left to the kernel.
        //
        // `grid.fractional` is the same call `ScoringContext::evaluate_full`
        // makes through `interpolate_with_gradient`, on the same un-narrowed
        // `f64` the CPU scores, so the flag is the CPU's decision rather than a
        // second opinion that has to be argued into agreement.
        //
        // The violation is the other half of the same branch, and it is taken
        // the same way: `out_of_box_violation_per_axis` against the grid's own
        // box, in `f64`, summed per axis, and exactly the sum the CPU's `None`
        // arm would have charged. The kernel used to recompute it from the
        // narrowed coordinate, which agreed on the branch and disagreed on the
        // size by up to `eps32 x |p|`; measured at `x = 3.000000119209 A` as
        // `1000 x 1.19e-7` on the CPU against `0` on the GPU. That is a
        // difference in the argument of a function both sides agreed to
        // evaluate, not a disagreement about which function, and it is bounded
        // by the coordinate rather than by the penalty -- so it does not shrink
        // as the penalty grows, and it is wide enough to stop any tolerance
        // policing the in-box path. See the header of `energy.wgsl`.
        //
        // In the box the CPU interpolates and charges nothing, so the uploaded
        // violation is `0.0` there. It is not required to be: the kernel only
        // reads it on the branch where the host set the flag to `0.0`.
        //
        // The uploaded position is `batch.coords` unaltered. It used to be
        // nudged by ULPs to match the branch; the flag makes that unnecessary,
        // and a coordinate the kernel never asked to move is one fewer way for
        // the two sides to end up somewhere different.
        //
        // This is a fresh buffer rather than an edit of `batch.coords` in
        // place: `score` takes `&Batch`, the layout is 32 bytes per point, and
        // a caller that reuses one batch across two grids must not find its
        // coordinates silently rewritten by the first call.
        // Everything the shader will be handed is decided here, on the host,
        // with no device in the call path. That is not a refactor for its own
        // sake: it is what lets `pack_upload` be asserted on a machine with no
        // adapter, which is the only place most of these numbers are ever
        // checked. See `pack_upload`'s own documentation for what that does and
        // does not establish.
        let (coords, params) = pack_upload(batch, grid, violations)?;

        let conf_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("opendocking-coords"),
                contents: bytemuck::cast_slice(&coords),
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
    /// The search box's maximum corner, for the out-of-box charge.
    ///
    /// Not derivable from `min` and `spacing` in the shader: `estimate_dims`
    /// rounds the point count *up*, so the last tabulated point sits at or past
    /// `box.max` and usually not on it. The CPU charges the violation against
    /// the box faces, so the shader needs the faces and not the tabulated edge —    /// deriving them here would put the two paths one cell apart at the boundary.
    bmax: [f32; 4],
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
        // —which no runtime assertion would catch.
        let p = GridParams {
            nx: 0,
            ny: 0,
            nz: 0,
            n_atoms: 0,
            n_conf: 0,
            _pad: [0; 3],
            min: [0.0; 4],
            spacing: [0.0; 4],
            bmax: [0.0; 4],
        };
        assert_eq!(std::mem::size_of::<GridParams>(), 80);

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
        assert_eq!(at(std::ptr::addr_of!(p.bmax), base), 64);
    }

    /// Bind the two transcriptions of the out-of-box penalty together.
    ///
    /// `OUT_OF_BOX_PENALTY` lives in `search/mod.rs` for the CPU and in
    /// `energy.wgsl` for the GPU, and nothing in either language checks the
    /// other. They are the same constant by convention, which is exactly the
    /// kind of convention that drifts: the two had already diverged once, when
    /// the CPU's flat step became a per-angstrom ramp and the shader was left
    /// holding `1000.0` as a flat charge per atom. The same test therefore also
    /// pins the *shape* —a per-angstrom ramp, not a flat step —because a
    /// constant alone cannot tell those two apart.
    #[test]
    fn shader_and_rust_agree_on_the_out_of_box_penalty() {
        let declared = ENERGY_WGSL
            .lines()
            .find_map(|l| l.trim().strip_prefix("const OUT_OF_BOX_PENALTY: f32 = "))
            .map(|rest| rest.trim_end_matches(';').trim().to_string())
            .expect("energy.wgsl must declare OUT_OF_BOX_PENALTY for the shader and Rust to agree");
        let shader: f64 = declared
            .parse()
            .unwrap_or_else(|e| panic!("{declared:?} in energy.wgsl is not a number: {e}"));
        assert_eq!(
            shader,
            crate::search::OUT_OF_BOX_PENALTY,
            "the shader's out-of-box penalty has drifted from the CPU's"
        );

        // The charge must scale with the violation, not with the number of
        // protruding atoms. A flat step would still pass the assertion above.
        //
        // The ramp itself is no longer written in the shader: the host
        // measures it with `out_of_box_violation_per_axis` -- the crate's one
        // definition -- and uploads the sum, and the shader multiplies it by
        // the constant. So the shape is now asserted where it is *computed*,
        // and what is asserted of the shader is that it multiplies the host's
        // violation rather than substituting a number of its own.
        assert!(
            ENERGY_WGSL.contains("OUT_OF_BOX_PENALTY * violation"),
            "the shader must charge the host's out-of-box violation, not a number \
             of its own. If it stopped doing that, the term it returns is either a \
             flat per-atom step (which caps the GPU side at 1000 x protruding atoms \
             while the CPU's is unbounded, and inverts the ranking of poses outside \
             the box) or a re-measurement of the violation from the narrowed f32 \
             coordinate, which disagrees with the CPU by eps32 x |p| amplified by \
             the penalty"
        );
        assert!(
            ENERGY_WGSL.contains("violation.x"),
            "the shader no longer reads the uploaded violation out of the \
             per-point coordinate; binding 1 is a pair of vec4s and the second \
             one is where it lives"
        );
        assert!(
            !ENERGY_WGSL.contains("return 1000.0;"),
            "the flat per-atom out-of-box sentinel is back; it caps the GPU term \
             at 1000 x protruding atoms while the CPU's is unbounded, which \
             inverts the ranking of poses outside the box"
        );
        // The host half of the same claim is not asserted by reading this file
        // against itself. It is measured instead:
        // `the_out_of_box_magnitude_is_the_hosts_and_putting_it_back_breaks_the_band`
        // overrides the uploaded violation and requires the result to change,
        // which is what "the host's number is the one being used" means.
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
        // fail, and it does not claim a result either —and the count below is
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
        // way —which is why this test does not have to assert a structural
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

        // `f64` in, because `Batch::from_coords` does the narrowing and keeps
        // the un-narrowed copy that `score` takes the in-box flag from. The
        // placements are `f32` literals, so the values are exactly the ones
        // this test always used.
        let mut coords: Vec<f64> = Vec::new();
        for p in &placements {
            coords.extend_from_slice(&[p[0] as f64, p[1] as f64, p[2] as f64]);
            coords.extend_from_slice(&[p[0] as f64 + 1.2, p[1] as f64, p[2] as f64]);
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

    /// The out-of-box half of the parity claim, and the half that was missing.
    ///
    /// [`the_cpu_and_gpu_paths_agree_where_the_difference_was_measured`] asserts
    /// `reach < 6.0` and calls `interpolate_with_gradient(...).expect("point is
    /// inside the box")` on the CPU side, so by construction it never asked the
    /// kernel a question it had not already excluded. The out-of-box term was
    /// therefore not untested so much as *unasked*: the one place the two
    /// backends implemented different functions was the one place the fixture
    /// was forbidden to look. The kernel returned a flat `1000.0` per protruding
    /// atom where the CPU charges `OUT_OF_BOX_PENALTY` per angstrom, which caps
    /// the GPU at `1000 x protruding atoms` against an unbounded CPU, and inverts
    /// the ranking of poses outside the box while every in-box pose still agrees
    /// to eight figures.
    ///
    /// This asks it, on the same adapter, and the CPU side below is the same
    /// branch `ScoringContext::evaluate_full` takes —the `None` arm of
    /// `interpolate_with_gradient`, the per-axis violation against the box
    /// faces, and the `> 0.0` guard. It is transcribed rather than reached
    /// through, so the two can be seen to agree instead of one hiding the other.
    ///
    /// The band is derived the same way the in-box one is, and for the same
    /// reason: it scales with the magnitude of the `f32` quantity the kernel
    /// summed. For an out-of-box atom that magnitude is the penalty itself, not
    /// the grid data —`read_magnitude` is zero there by construction, and a band
    /// proportional to it would be zero, so it is widened by the penalty the two
    /// paths are both computing.
    #[test]
    fn the_cpu_and_gpu_paths_agree_outside_the_box_too() {
        if !is_available() {
            eprintln!(
                "NOT MEASURED: this machine has no GPU adapter, so the out-of-box \
                 difference between the two backends was not computed and no \
                 comparison was made. This line reports an absence, not a result."
            );
            return;
        }
        use crate::grid::{GridBox, GridMaps};
        use crate::scoring::VinaScoring;

        let receptor = test_receptor();
        let scoring = VinaScoring::new();
        let box_ = GridBox::new([-6.0, -6.0, -6.0], [6.0, 6.0, 6.0]).expect("box");
        let maps = GridMaps::precalculate(&receptor, &box_, &scoring, 0.5, 0).expect("maps");

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

        // A slide out through +x, and a diagonal out through two faces, both
        // spaced finely enough to cross the box face by hundredths. The
        // in-box fixture cannot supply any of these: its `reach < 6.0` guard is
        // the assertion that made this region unaskable.
        let mut placements: Vec<[f32; 3]> = vec![[-1.0, -1.0, -1.0], [2.0, 1.5, 0.5]];
        let mut x = 5.9f32;
        while x <= 20.0 {
            placements.push([x, 0.5, -0.5]);
            placements.push([6.0, x, 1.0]); // out through a second face too
            x += 0.37;
        }

        // `f64` in, because `Batch::from_coords` does the narrowing and keeps
        // the un-narrowed copy that `score` takes the in-box flag from. The
        // placements are `f32` literals, so the values are exactly the ones
        // this test always used.
        let mut coords: Vec<f64> = Vec::new();
        for p in &placements {
            coords.extend_from_slice(&[p[0] as f64, p[1] as f64, p[2] as f64]);
            coords.extend_from_slice(&[p[0] as f64 + 1.2, p[1] as f64, p[2] as f64]);
        }
        let n_conf = placements.len();
        let batch = Batch::from_coords(coords, n_conf, &ligand).expect("batch");

        let mut ctx = GpuContext::new().expect("gpu context");
        let got = ctx.score(&batch, &maps).expect("gpu score");
        assert_eq!(got.len(), n_conf, "one energy per conformation");

        // The CPU branch, transcribed from `ScoringContext::evaluate_full`.
        let b = maps.grid_box();
        let mut want = Vec::with_capacity(n_conf);
        let mut penalty_magnitude = Vec::with_capacity(n_conf);
        for p in &placements {
            let mut total = 0.0f64;
            let mut magnitude = 0.0f64;
            for (i, _atom) in ligand.molecule.atoms.iter().enumerate() {
                let mut xyz = [p[0] as f64, p[1] as f64, p[2] as f64];
                xyz[0] += if i == 1 { 1.2 } else { 0.0 };
                match maps.interpolate_with_gradient(ligand.type_index[i], &ligand.weights[i], xyz)
                {
                    Some((e, _g)) => total += e,
                    None => {
                        let per_axis = crate::grid::out_of_box_violation_per_axis(xyz, &b);
                        let violation: f64 = per_axis.iter().sum();
                        if violation > 0.0 {
                            let penalty = crate::search::OUT_OF_BOX_PENALTY * violation;
                            total += penalty;
                            magnitude += penalty;
                        }
                    }
                }
            }
            want.push(total);
            penalty_magnitude.push(magnitude);
        }

        const ROUNDINGS: f64 = 64.0; // 8 corners x 4 slots x 2 atoms
        let eps32 = f32::EPSILON as f64;
        let mut worst = 0.0f64;
        let mut worst_at = 0usize;
        let mut worst_band = 0.0f64;
        let mut over_band = 0usize;
        for (c, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let diff = ((*g as f64) - *w).abs();
            // The magnitude the `f32` kernel summed, which for these rows is
            // the out-of-box penalty and not any grid value.
            let tol = ROUNDINGS * eps32 * penalty_magnitude[c].max(1.0);
            if diff > worst {
                worst = diff;
                worst_at = c;
                worst_band = tol;
            }
            if diff > tol {
                over_band += 1;
            }
        }
        assert_eq!(
            over_band,
            0,
            "the two backends charge a different function outside the box: \
             {over_band} of {n_conf} conformations differ by more than the band \
             derived from the penalty each of them is summing. Worst {worst:e} at \
             conformation {worst_at} (gpu {}, cpu {}) against a band of {worst_band:e} \
             (adapter {}). The GPU is bounded by 1000 x protruding atoms; the CPU \
             is not",
            got[worst_at],
            want[worst_at],
            ctx.adapter_name()
        );
        // The ranking claim, which is the one a batch scorer exists for and the
        // one the flat per-atom step inverted.
        //
        // Stated as a claim about *separated* pairs, not as equality of two
        // orderings. `f32` and `f64` are not expected to be equal, and this
        // lattice deliberately produces near-degenerate neighbours 0.37 A apart
        // whose energies differ by less than the band, so demanding that every
        // adjacent pair keep its place would be demanding bit-identity —the
        // first version of this assertion did, and it failed on pairs
        // `(4,5), (6,7), (8,9), ...` whose energies agree to nine figures, which
        // is a property of rounding and not of the out-of-box term. An
        // inversion only means something when the two backends disagree about
        // the order of a pair they can actually tell apart.
        let mut compared = 0usize;
        let mut inversions: Vec<(usize, usize)> = Vec::new();
        for i in 0..n_conf {
            for j in i + 1..n_conf {
                let gap = (want[i] - want[j]).abs();
                if gap <= ROUNDINGS * eps32 * penalty_magnitude[i].max(penalty_magnitude[j]) {
                    continue;
                }
                compared += 1;
                if (want[i] - want[j]).is_sign_positive() != (got[i] - got[j]).is_sign_positive() {
                    inversions.push((i, j));
                }
            }
        }
        assert!(
            inversions.is_empty(),
            "the two backends order {} of {compared} separated out-of-box \
             conformation pairs differently, so a batch scorer ranking them by the \
             GPU would propose a different pose than the search ranking them by the \
             CPU. First inversion at {inversions:?} -- cpu {} vs {}, gpu {} vs {} \
             (adapter {})",
            inversions.len(),
            want[inversions[0].0],
            want[inversions[0].1],
            got[inversions[0].0],
            got[inversions[0].1],
            ctx.adapter_name()
        );
        eprintln!(
            "MEASURED out-of-box on adapter {}: {n_conf} conformations, worst \
             |GPU - CPU| = {worst:e} at conformation {worst_at} (gpu {}, cpu {}) \
             against a derived band of {worst_band:e} there -- the difference used \
             {:.1}% of the band. Ranking: {compared} pairs separated by more than \
             that band, {} of them ordered differently",
            ctx.adapter_name(),
            got[worst_at],
            want[worst_at],
            100.0 * worst / worst_band,
            inversions.len()
        );
    }

    // -----------------------------------------------------------------
    // The in-box flag: the branch the host takes and the kernel obeys.
    // -----------------------------------------------------------------

    /// The CPU scorer's per-atom loop for one conformation, transcribed from
    /// `ScoringContext::evaluate_full`.
    ///
    /// A copy rather than a call: `ScoringContext` owns search state (torsion
    /// trees, a convergence budget) that this test has no reason to build, and
    /// every branch in it -- interpolate, refuse, charge -- is two lines.
    fn cpu_sum(
        maps: &crate::grid::GridMaps,
        ligand: &crate::ligand::Ligand,
        coords: &[[f64; 3]],
    ) -> f64 {
        let b = maps.grid_box();
        let mut total = 0.0f64;
        for (i, p) in coords.iter().enumerate() {
            match maps.interpolate_with_gradient(ligand.type_index[i], &ligand.weights[i], *p) {
                Some((e, _)) => total += e,
                None => {
                    let v: f64 = crate::grid::out_of_box_violation_per_axis(*p, &b)
                        .iter()
                        .sum();
                    if v > 0.0 {
                        total += crate::search::OUT_OF_BOX_PENALTY * v;
                    }
                }
            }
        }
        total
    }

    /// A one-carbon probe: one atom, so a conformation's energy is one
    /// interpolation and a disagreement is attributable to one predicate.
    fn one_carbon() -> crate::ligand::Ligand {
        let m = crate::types::Molecule {
            atoms: vec![crate::types::Atom::new(
                1,
                [0.0, 0.0, 0.0],
                crate::types::Element::C,
                crate::types::AtomType::CH,
            )],
            bonds: Vec::new(),
            neighbors: vec![Vec::new()],
        };
        crate::ligand::Ligand::from_molecule_with(m, &[]).expect("ligand")
    }

    /// A grid whose `+x` face lands *exactly* on the box face, with receptor
    /// mass 1.5 Å behind it.
    ///
    /// The face being on the box face is the reachable case, and the one that
    /// matters: `estimate_dims` rounds the point count up, so a box whose side
    /// is not a whole multiple of the spacing leaves the last tabulated point
    /// outside the box, and a flip there is in a sliver no search can place an
    /// atom in. `9.0 Å` at `0.375 Å` is exactly 24 steps, so `+x` ends on the
    /// face and a placement a rounding step inside the tabulated volume is also
    /// inside the box.
    fn face_fixture() -> crate::grid::GridMaps {
        use crate::grid::{GridBox, GridMaps};
        use crate::scoring::VinaScoring;
        let box_ = GridBox::new([-6.0, -6.0, -6.0], [3.0, 6.0, 6.0]).expect("box");
        GridMaps::precalculate(&test_receptor(), &box_, &VinaScoring::new(), 0.375, 0)
            .expect("maps")
    }

    /// Placements straddling the `+x` tabulated face, at `f32` resolution.
    ///
    /// The kernel only ever sees an `f32`, so every `f64` a caller can pass
    /// collapses onto one of these. Within a rounding bucket the CPU's `u` is
    /// monotone in `p`, so the extreme decisions in a bucket are at its
    /// endpoints, and those plus the bucket's own value cover the whole bucket
    /// rather than a sample of it.
    fn face_placements(maps: &crate::grid::GridMaps, window: i32) -> Vec<f64> {
        let face = maps.min[0] + (maps.dims()[0] - 1) as f64 * maps.spacing[0];
        let base = (face as f32).to_bits() as i64;
        let mut out = Vec::new();
        for k in -window..=window {
            let v = f32::from_bits((base + k as i64) as u32);
            let h = (f32::from_bits(v.to_bits() + 1) - v) as f64 / 2.0;
            out.push(v as f64 - h * (1.0 - 1e-9));
            out.push(v as f64);
            out.push(v as f64 + h * (1.0 - 1e-9));
        }
        out
    }

    /// The band the kernel's `f32` arithmetic can spend on one placement, from
    /// the same derivation the two tests above use: every corner value times
    /// this atom's slot weight, summed, times the number of `f32` roundings
    /// those values pass through. 8 corners x 4 slots, one atom.
    fn read_band(maps: &crate::grid::GridMaps, ligand: &crate::ligand::Ligand, p: [f64; 3]) -> f64 {
        const CORNERS: [[usize; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        // The crate's own cell, so the magnitude is read from the nodes this
        // placement actually touches rather than from a restatement of them.
        let Some((_frac, cell)) = maps.fractional(p) else {
            return 0.0;
        };
        let mut magnitude = 0.0f64;
        for c in CORNERS {
            for (s, slot) in crate::grid::MapSlot::ALL.iter().enumerate() {
                let w = ligand.weights[0][s] as f64;
                if w == 0.0 {
                    continue;
                }
                let v = maps.raw(
                    ligand.type_index[0],
                    *slot,
                    cell[0] + c[0],
                    cell[1] + c[1],
                    cell[2] + c[2],
                );
                magnitude += (v as f64 * w).abs();
            }
        }
        const ROUNDINGS: f64 = 32.0; // 8 corners x 4 slots x 1 atom
        ROUNDINGS * f32::EPSILON as f64 * magnitude
    }

    /// The band one placement may legitimately miss by, in full.
    ///
    /// One term now, and it is the same term on both sides of the face.
    ///
    ///  * in the box, the `f32` sum of grid values (`read_band`);
    ///  * out of the box, the same shape of bound applied to the penalty the
    ///    two sides are both summing.
    ///
    /// The second used to be `OUT_OF_BOX_PENALTY * eps32 * |p[k]|` per
    /// offending axis, and that line is the reason this function changed.
    /// `sample()` used to measure the violation from the `f32` it was handed,
    /// so the two violators could differ by the half-ULP of the narrowing,
    /// which the penalty multiplied into an *absolute* error proportional to
    /// the coordinate. A band with an absolute floor that does not shrink as
    /// the term it is policing gets larger is a band that cannot police
    /// anything else: at `x = 3.000000119209` it admitted `3.6e-4`, against
    /// an in-box disagreement of the same order, so the whole kernel had to be
    /// held to the wider of the two.
    ///
    /// Now the kernel multiplies the *host's* violation by the same constant,
    /// so what is left is `f32` rounding of a quantity both sides computed
    /// identically -- `16 x eps32 x penalty`, the same shape as `read_band` and
    /// proportional to the term rather than to the position. It shrinks in
    /// relative terms as the penalty grows, which is the direction a rounding
    /// band has to shrink in.
    ///
    /// `the_out_of_box_magnitude_is_the_hosts_and_putting_it_back_breaks_the_band`
    /// is the other half of this claim: it puts the old magnitude back and
    /// requires the band here to go red.
    fn band_for(maps: &crate::grid::GridMaps, ligand: &crate::ligand::Ligand, p: [f64; 3]) -> f64 {
        let mut band = read_band(maps, ligand, p);
        if maps.fractional(p).is_none() {
            let b = maps.grid_box();
            let violation: f64 = crate::grid::out_of_box_violation_per_axis(p, &b)
                .iter()
                .sum();
            if violation > 0.0 {
                // The same derivation as `read_band`, applied to the other
                // quantity, and counted roundings rather than guessed at.
                //
                // The penalty passes through exactly two `f32` roundings: the
                // host's `f64` violation narrowed on the way in, and the
                // multiply by the constant here. Each is at most `eps32 / 2`
                // relative, so `eps32` relative in total, and the factor of 2
                // is a 2x margin on that -- deliberately not larger, because a
                // band with slack in it is a band that stops noticing.
                //
                // Nothing else rounds: `partial[lid]` is the whole term for a
                // one-atom conformation, and the workgroup reduction then adds
                // it to zeros, which is exact. That is why this constant is
                // small where `read_band`'s `ROUNDINGS` is 32: an in-box
                // placement sums eight corners times four slots of unrelated
                // magnitudes, and this one sums a single number.
                //
                // What it scales with is the point. The term it used to add
                // scaled with the *coordinate*, so a pose 1e-7 A outside a face
                // 3 A away was held to a tolerance set by that 3 A. This one
                // scales with the penalty, so that pose is held to a tolerance
                // set by its own 1e-4 kcal/mol. It is therefore not uniformly
                // the narrower of the two -- on a pose protruding 8 A the
                // penalty is comparable to the coordinate and the two bands are
                // within a factor of 1.5 of each other. It is narrower by
                // orders of magnitude where the defect was, and it no longer
                // has to be widened to hold a term that is now the host's.
                const PENALTY_ROUNDINGS: f64 = 2.0;
                band += PENALTY_ROUNDINGS
                    * f32::EPSILON as f64
                    * crate::search::OUT_OF_BOX_PENALTY
                    * violation;
            }
        }
        band
    }

    /// The branch the *removed* `f32` derivation took, kept only so this test
    /// can prove the fixture is one where that derivation actually flipped.
    ///
    /// Transcribed from `energy.wgsl` as it stood before the flag: `u` from the
    /// `f32` coordinate, against `f32(n - 1)`. Nothing in the kernel uses this
    /// any more.
    fn old_f32_inside(maps: &crate::grid::GridMaps, p: f64) -> bool {
        let d = maps.dims();
        let u = ((p as f32) - (maps.min[0] as f32)) / (maps.spacing[0] as f32);
        u >= 0.0 && u < (d[0] as u32 - 1) as f32
    }

    /// Score `placements` on the GPU, one conformation each, and return the
    /// energies alongside the CPU's.
    fn both_sides(
        ctx: &mut GpuContext,
        maps: &crate::grid::GridMaps,
        ligand: &crate::ligand::Ligand,
        placements: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let coords: Vec<f64> = placements.iter().flat_map(|&x| [x, 0.0, 0.0]).collect();
        let batch = Batch::from_coords(coords, placements.len(), ligand).expect("batch");
        let got = ctx.score(&batch, maps).expect("gpu score");
        assert_eq!(got.len(), placements.len(), "one energy per conformation");
        let want = placements
            .iter()
            .map(|&x| cpu_sum(maps, ligand, &[[x, 0.0, 0.0]]))
            .collect();
        (got.iter().map(|g| *g as f64).collect(), want)
    }

    /// The branch is decided once, on the host, and the two backends agree.
    ///
    /// This is the test the flag exists for. It walks `f32`-resolution
    /// placements across a tabulated face that lands on the box face, and
    /// requires every one of them to agree with the CPU to within the band the
    /// kernel's own `f32` arithmetic can spend. Before the flag the worst
    /// disagreement over this same walk was the map value at the face --
    /// measured 15.540 kcal/mol on a 24x12x12 Å box at 0.375 Å, against a band
    /// of ~1e-3.
    ///
    /// The first assertion is the one that stops it being vacuous: it requires
    /// the removed `f32` derivation to have disagreed with the CPU on at least
    /// one of these placements. If a change to the fixture ever moved the
    /// disagreement out of the window, this test would still pass its band --
    /// and would then be asserting nothing about the branch. So the fixture is
    /// required to be one where the bug was live.
    #[test]
    fn the_in_box_flag_holds_the_branch_across_a_face_the_f32_derivation_flipped() {
        if !is_available() {
            eprintln!(
                "NOT MEASURED: this machine has no GPU adapter, so the CPU/GPU \
                 agreement across a tabulated face was not computed and no \
                 comparison was made. This line reports an absence, not a result."
            );
            return;
        }
        let maps = face_fixture();
        let ligand = one_carbon();
        let placements = face_placements(&maps, 6);
        let face = maps.min[0] + (maps.dims()[0] - 1) as f64 * maps.spacing[0];
        assert_eq!(
            face, 3.0,
            "the fixture's tabulated +x face must land exactly on the box face, \
             or the flip it is meant to expose falls outside the box where no \
             search places an atom"
        );

        // Would the removed derivation have flipped on any of these?
        let would_flip = placements
            .iter()
            .filter(|&&p| maps.fractional([p, 0.0, 0.0]).is_some() != old_f32_inside(&maps, p))
            .count();
        assert!(
            would_flip > 0,
            "this fixture no longer reproduces the defect: the f32 derivation \
             agreed with the CPU on all {} placements, so the band assertion \
             below would pass without the flag doing anything",
            placements.len()
        );

        let mut ctx = GpuContext::new().expect("gpu context");
        let (got, want) = both_sides(&mut ctx, &maps, &ligand, &placements);
        let mut worst = 0.0f64;
        let mut worst_at = 0usize;
        let mut worst_band = 0.0f64;
        for (i, (&p, (&g, &w))) in placements
            .iter()
            .zip(got.iter().zip(want.iter()))
            .enumerate()
        {
            let diff = (g - w).abs();
            let band = band_for(&maps, &ligand, [p, 0.0, 0.0]);
            assert!(
                diff <= band,
                "placement {i} at x = {p:.12} ({:+.3e} A from the face): GPU {g} \
                 vs CPU {w}, a difference of {diff:e} against a derived band of \
                 {band:e}. The map value at that face is what a wrong branch \
                 costs, and it is ~1.5e4 times the in-box part of that band \
                 (adapter {})",
                p - face,
                ctx.adapter_name()
            );
            if diff > worst {
                worst = diff;
                worst_at = i;
                worst_band = band;
            }
        }
        let span = placements
            .iter()
            .map(|&p| (p - face).abs())
            .fold(0.0f64, f64::max);
        eprintln!(
            "MEASURED across a face on adapter {}: {} placements within {:.1e} A of \
             the tabulated +x face, {would_flip} of which the removed f32 \
             derivation would have branched differently on; worst |GPU - CPU| = \
             {worst:e} at placement {worst_at} against a derived band of \
             {worst_band:e} there -- {:.1}% of the band",
            ctx.adapter_name(),
            placements.len(),
            span,
            100.0 * worst / worst_band
        );
    }

    /// The mutation: take the decision from the *narrowed* coordinate.
    ///
    /// A test that passes when the two sides are deliberately made to disagree
    /// is not holding anything, so this builds the same batch with `coords64`
    /// replaced by the `f32` values widened back -- identical uploaded
    /// positions, and a flag that is now decided from the rounding the kernel
    /// used to disagree with. That is the whole bug, injected through the
    /// production path with no test-only branch in it: `score` reads
    /// `coords64` and knows nothing about this test.
    ///
    /// It must produce a disagreement far outside the band. If a future change
    /// made it agree, the band assertion in the test above would be holding
    /// nothing and this one would say so.
    #[test]
    fn deciding_the_flag_from_the_narrowed_coordinate_breaks_the_agreement() {
        if !is_available() {
            eprintln!(
                "NOT MEASURED: this machine has no GPU adapter, so the mutation \
                 was not applied and no disagreement was observed. This line \
                 reports an absence, not a result."
            );
            return;
        }
        let maps = face_fixture();
        let ligand = one_carbon();
        let placements = face_placements(&maps, 6);

        let mut ctx = GpuContext::new().expect("gpu context");
        let (honest_got, _) = both_sides(&mut ctx, &maps, &ligand, &placements);

        // The mutation. Same coordinates uploaded; the decision is taken from
        // them after the narrowing, which is the one thing the flag is for.
        let coords: Vec<f64> = placements.iter().flat_map(|&x| [x, 0.0, 0.0]).collect();
        let honest = Batch::from_coords(coords, placements.len(), &ligand).expect("batch");
        let mutated = Batch {
            coords64: honest.coords.iter().map(|&c| c as f64).collect(),
            ..honest.clone()
        };
        let mutated_got = ctx.score(&mutated, &maps).expect("gpu score");

        let mut worst_honest = 0.0f64;
        let mut worst_mutated = 0.0f64;
        let mut mutated_at = 0usize;
        let mut outsides = 0usize;
        for (i, (&p, (&h, &m))) in placements
            .iter()
            .zip(honest_got.iter().zip(mutated_got.iter()))
            .enumerate()
        {
            let w = maps.fractional([p, 0.0, 0.0]).is_some();
            let w_mut = maps
                .fractional([honest.coords[i * 3] as f64, 0.0, 0.0])
                .is_some();
            if w != w_mut {
                outsides += 1;
            }
            let cpu = cpu_sum(&maps, &ligand, &[[p, 0.0, 0.0]]);
            // `honest_got` is already `f64` (both_sides widens); `mutated_got` is
            // the raw `f32` `score` returns, so it still needs the cast.
            let d_h = (h - cpu).abs();
            let d_m = (m as f64 - cpu).abs();
            if d_h > worst_honest {
                worst_honest = d_h;
            }
            if d_m > worst_mutated {
                worst_mutated = d_m;
                mutated_at = i;
            }
        }
        assert!(
            outsides > 0,
            "the mutation changed no branch: the f64 predicate applied to the \
             narrowed coordinate agreed with itself on all {} placements, so \
             this test would pass without ever making the two sides disagree",
            placements.len()
        );
        let band = band_for(&maps, &ligand, [placements[mutated_at], 0.0, 0.0]);
        assert!(
            worst_mutated > band,
            "deciding the flag from the narrowed coordinate changed {outsides} \
             branches but the worst energy difference stayed at {worst_mutated:e}, \
             inside the {band:e} band. The mutation is supposed to reproduce the \
             original defect, where the wrong branch costs the map value at the \
             face (~1.5e1 kcal/mol) rather than a rounding step; if it no longer \
             does, the band assertion in \
             `the_in_box_flag_holds_the_branch_across_a_face_the_f32_derivation_flipped` \
             is not holding anything and this fixture no longer reproduces the bug"
        );
        eprintln!(
            "MEASURED mutation on adapter {}: {outsides} of {} placements took the \
             other branch once the flag was decided from the narrowed coordinate; \
             worst |GPU - CPU| rose from {worst_honest:e} to {worst_mutated:e} \
             (placement {mutated_at}), against a derived band of {band:e} there",
            ctx.adapter_name(),
            placements.len()
        );
    }

    // -----------------------------------------------------------------
    // The out-of-box magnitude: the host's number, and the band that
    // follows from it.
    // -----------------------------------------------------------------

    /// One row of the out-of-box sweep, in both arms.
    ///
    /// `honest` is what the kernel returns now, with the host's `f64`
    /// violation uploaded. `from_f32` is what it returned when the kernel
    /// measured the violation from the narrowed coordinate itself -- the same
    /// uploaded positions, the same in-box flag, and only the size of the
    /// charge different. `old_band` is the tolerance this crate had to carry
    /// for that arm; `band` is the one it carries now.
    #[derive(Debug, Clone, Copy)]
    struct OutRow {
        p: [f64; 3],
        violation: f64,
        cpu: f64,
        honest: f64,
        from_f32: f64,
        old_band: f64,
        band: f64,
    }

    /// The band this crate had to use before the violation moved to the host.
    ///
    /// Written out rather than deleted, because the comparison in the test
    /// below is the only evidence that removing it was worth anything: a band
    /// nobody ever held is not a band anybody gave up.
    fn old_band_for(maps: &crate::grid::GridMaps, p: [f64; 3]) -> f64 {
        let b = maps.grid_box();
        let mut band = read_band(maps, &one_carbon(), p);
        for (k, v) in crate::grid::out_of_box_violation_per_axis(p, &b)
            .iter()
            .enumerate()
        {
            if *v > 0.0 {
                band += crate::search::OUT_OF_BOX_PENALTY * f32::EPSILON as f64 * p[k].abs();
            }
        }
        band
    }

    /// Placements concentrated on the face where the defect lives.
    ///
    /// The in-box fixture cannot supply these: its `reach < 6.0` guard is the
    /// assertion that made the out-of-box region unaskable, and 44 of its 47
    /// rows never crossed the face at all. Three groups:
    ///
    ///  * the `f32`-resolution neighbourhood of the `+x` face, both sides,
    ///    which is where the narrowing decides whether the kernel can see a
    ///    violation at all;
    ///  * further out along `+x`, where the violation is large enough that the
    ///    defect is a clean fraction of the penalty rather than a rounding of
    ///    zero;
    ///  * out through two faces at once, where the per-axis sum matters and a
    ///    centre-distance would charge the wrong axes.
    fn out_of_box_placements(maps: &crate::grid::GridMaps) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        // The f32-resolution neighbourhood of the +x face, both sides. The
        // inside half is kept deliberately: it is the control that says the
        // sweep did not simply become "everything is out of the box", and it
        // is where the in-box test's rows live.
        for x in face_placements(maps, 6) {
            out.push([x, 0.0, 0.0]);
        }
        // Immediately outside the face, at a tenth-ULP pitch, which is where
        // the narrowing decides whether the kernel can see a violation at all.
        let face = 3.0f32;
        for d in 1..=64u32 {
            out.push([f32::from_bits(face.to_bits() + d) as f64, 0.0, 0.0]);
        }
        // Further out along +x, where the violation is large enough that the
        // defect is a clean fraction of the penalty rather than a rounding of
        // zero.
        let mut x = 3.0f64;
        while x <= 12.0 {
            out.push([x, 0.0, 0.0]);
            x += 0.11;
        }
        // And out through two faces at once, where the per-axis sum matters and
        // a centre-distance would charge the wrong axes.
        let mut y = 6.0f64;
        while y <= 8.0 {
            out.push([4.0, y, 0.0]);
            y += 0.11;
        }
        out
    }

    /// Run one out-of-box sweep in both arms and return the rows.
    ///
    /// The mutated arm is reached through
    /// [`GpuContext::score_with_violations`], which is the production upload
    /// path with one number replaced. Using a second copy of the shader or a
    /// second transcription of the violation would make the test assert that
    /// two hand-written things agree, which is the mistake it is meant to
    /// catch.
    fn sweep_out_of_box(
        ctx: &mut GpuContext,
        maps: &crate::grid::GridMaps,
        ligand: &crate::ligand::Ligand,
        placements: &[[f64; 3]],
    ) -> Vec<OutRow> {
        let b = maps.grid_box();
        let coords: Vec<f64> = placements.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let batch = Batch::from_coords(coords, placements.len(), ligand).expect("batch");

        // What the kernel used to compute: the same ramp, against the same
        // box, from the coordinate it was actually handed. Done in `f64` from
        // the narrowed value rather than in `f32` from it, which is the same
        // number up to one ulp of a `max` and is *exactly* the same number on
        // every row that matters, because on those the narrowed coordinate
        // lands on a face and the difference is the whole term.
        let from_f32: Vec<f64> = (0..placements.len())
            .map(|i| {
                let p = [
                    batch.coords[i * 3] as f64,
                    batch.coords[i * 3 + 1] as f64,
                    batch.coords[i * 3 + 2] as f64,
                ];
                crate::grid::out_of_box_violation_per_axis(p, &b)
                    .iter()
                    .sum()
            })
            .collect();

        let honest: Vec<f64> = ctx
            .score(&batch, maps)
            .expect("gpu score")
            .iter()
            .map(|&v| v as f64)
            .collect();
        let mutated: Vec<f64> = ctx
            .score_with_violations(&batch, maps, Some(&from_f32))
            .expect("gpu score")
            .iter()
            .map(|&v| v as f64)
            .collect();

        placements
            .iter()
            .enumerate()
            .map(|(i, &p)| OutRow {
                p,
                violation: crate::grid::out_of_box_violation_per_axis(p, &b)
                    .iter()
                    .sum(),
                cpu: cpu_sum(maps, ligand, &[p]),
                honest: honest[i],
                from_f32: mutated[i],
                old_band: old_band_for(maps, p),
                band: band_for(maps, ligand, p),
            })
            .collect()
    }

    /// The band is honest in both directions, and the sweep that produced the
    /// number is on the face the defect lived on.
    ///
    /// The previous round closed the in-box branch and was left with a residual
    /// it could not remove without widening the band: the kernel measured the
    /// out-of-box violation from the narrowed `f32` coordinate and the CPU
    /// measured it from the `f64`, so the two agreed on *which* function to
    /// evaluate and disagreed on its argument by `eps32 x |p|`, amplified by
    /// the penalty. The band's response to that was to add a derived term --
    /// `penalty x eps32 x |p|` per offending axis -- and a band widened to
    /// accommodate a known residual stops being able to police anything else.
    ///
    /// So the host now measures the violation, in `f64`, with the crate's own
    /// `out_of_box_violation_per_axis`, and uploads it. The branch is still the
    /// host's flag; only the *magnitude* moved, and it is the side that knows
    /// it. What is left is `f32` rounding of a number both sides computed
    /// identically, which is the same shape of bound as the in-box one and is
    /// proportional to the penalty rather than to the coordinate.
    ///
    /// Direction one, below: with the fix in place the sweep is inside the
    /// tightened band. Direction two, in the mutation test that follows: put
    /// the old magnitude back and the same band goes red. A band that cannot
    /// go red is a comment.
    #[test]
    fn the_out_of_box_magnitude_is_the_hosts_and_putting_it_back_breaks_the_band() {
        if !is_available() {
            eprintln!(
                "NOT MEASURED: this machine has no GPU adapter, so the out-of-box \
                 magnitude the host now uploads was not compared against the one \
                 the kernel used to measure and no band was exercised. This line \
                 reports an absence, not a result."
            );
            return;
        }
        let maps = face_fixture();
        let ligand = one_carbon();
        let placements = out_of_box_placements(&maps);
        let mut ctx = GpuContext::new().expect("gpu context");
        let rows = sweep_out_of_box(&mut ctx, &maps, &ligand, &placements);

        // Only the rows the CPU refused. The ones just inside the face are in
        // `placements` so that the sweep straddles it, and they belong to the
        // other test.
        let outside: Vec<&OutRow> = rows.iter().filter(|r| r.violation > 0.0).collect();
        let inside = rows.len() - outside.len();
        assert!(
            outside.len() > 100,
            "only {} of {} rows were out of the box, so this sweep is not \
             concentrated on the face the defect lived on",
            outside.len(),
            rows.len()
        );

        let mut worst_honest = (0.0f64, 0usize);
        let mut worst_from_f32 = (0.0f64, 0usize);
        let mut over_new_band = 0usize;
        let mut outside_old_band = 0usize;
        let mut worst_ratio = (0.0f64, 0usize);
        let mut honest_ratio = (0.0f64, 0usize);
        let mut narrowest = f64::INFINITY;
        let mut narrowest_at = 0usize;
        for (i, r) in outside.iter().enumerate() {
            let d_honest = (r.honest - r.cpu).abs();
            let d_f32 = (r.from_f32 - r.cpu).abs();
            if d_honest > worst_honest.0 {
                worst_honest = (d_honest, i);
            }
            if d_f32 > worst_from_f32.0 {
                worst_from_f32 = (d_f32, i);
            }
            if d_honest > r.band {
                over_new_band += 1;
            }
            if d_f32 > r.old_band {
                outside_old_band += 1;
            }
            let ratio = d_f32 / r.old_band.max(f64::MIN_POSITIVE);
            if ratio > worst_ratio.0 {
                worst_ratio = (ratio, i);
            }
            let hr = d_honest / r.band.max(f64::MIN_POSITIVE);
            if hr > honest_ratio.0 {
                honest_ratio = (hr, i);
            }
            let n = r.old_band / r.band.max(f64::MIN_POSITIVE);
            if n < narrowest {
                narrowest = n;
                narrowest_at = i;
            }
        }
        let (wd_h, wi_h) = worst_honest;
        let r_h = outside[wi_h];
        let (wd_f, wi_f) = worst_from_f32;
        let r_f = outside[wi_f];

        assert_eq!(
            over_new_band,
            0,
            "{} of {} out-of-box rows are outside the band derived from the \
             penalty the two sides are summing. Worst {wd_h:e} at {:?} \
             (gpu {} vs cpu {}) against {}. The violation is the host's own \
             `f64` number now, so what is left is `f32` rounding of the right \
             quantity and the band is the same shape as the in-box one",
            over_new_band,
            outside.len(),
            r_h.p,
            r_h.honest,
            r_h.cpu,
            r_h.band
        );

        // How much narrower, and where. Reported rather than asserted: these
        // are properties of this sweep, and an assertion on them would only
        // fail when a *fix* made things worse.
        //
        // The three numbers, because "narrower" on its own would be a lie:
        // the band is orders of magnitude narrower next to the face, and
        // slightly *wider* on the deepest protrusions, where the penalty and
        // the coordinate are the same size and a band proportional to the term
        // cannot be smaller than one proportional to the coordinate. The widest
        // it ever gets is the number that says the in-box path is not
        // re-widened by this.
        let new_vs_old = r_h.old_band / r_h.band.max(f64::MIN_POSITIVE);
        let widest = 1.0 / narrowest;
        let nr = outside[narrowest_at];
        let hr = outside[honest_ratio.1];
        eprintln!(
            "MEASURED out-of-box face on adapter {}: {} rows, {} out of the box \
             and {} just inside it (the in-box half is kept as a control). The \
             fixture's +x face is at 3.0 A and its +y face at 6.0 A. With the \
             host's violation uploaded, worst |GPU - CPU| = {:.3e} at {:?}, which \
             is {:.1}% of its {:.3e} band; the worst any row reaches as a \
             fraction of its own band is {:.1}% (at {:?}, violation {:.3e} A). \
             With the old f32-derived magnitude, worst |GPU - CPU| = {:.3e} at \
             {:?}, which is {:.1}% of the {:.3e} band that row used to need. \
             Band width against the old one: {:.4e}x narrower on the worst honest \
             row, and never more than {:.2}x wider anywhere in the sweep (that \
             widest point is {:?}, where the penalty is {:.0} kcal/mol and the \
             two scales have converged). The old magnitude sat inside its own \
             band on {} of {} rows, so this defect was never catchable by a \
             row-by-row comparison at the old tolerance either -- it was \
             catchable only in what that tolerance had to be, which is why the \
             band had to be widened rather than the defect fixed.",
            ctx.adapter_name(),
            rows.len(),
            outside.len(),
            inside,
            wd_h,
            r_h.p,
            100.0 * honest_ratio.0,
            r_h.band,
            100.0 * honest_ratio.0,
            hr.p,
            hr.violation,
            wd_f,
            r_f.p,
            100.0 * wd_f / r_f.old_band,
            r_f.old_band,
            new_vs_old,
            widest,
            nr.p,
            nr.cpu,
            outside.len() - outside_old_band,
            outside.len()
        );
    }

    /// The other direction: put the old magnitude back, and the tightened band
    /// goes red.
    ///
    /// Same coordinates, same in-box flag, same kernel -- only the number
    /// multiplied by `OUT_OF_BOX_PENALTY` is the one the kernel used to work
    /// out for itself. If this did not exceed the band in
    /// `band_for`, that band would be holding nothing, and the tightening above
    /// would be an assertion that `0 <= 0`.
    #[test]
    fn the_tightened_band_rejects_the_magnitude_the_kernel_used_to_measure() {
        if !is_available() {
            eprintln!(
                "NOT MEASURED: this machine has no GPU adapter, so the mutation was \
                 not applied and no band was shown to go red. This line reports an \
                 absence, not a result."
            );
            return;
        }
        let maps = face_fixture();
        let ligand = one_carbon();
        let placements = out_of_box_placements(&maps);
        let mut ctx = GpuContext::new().expect("gpu context");
        let rows = sweep_out_of_box(&mut ctx, &maps, &ligand, &placements);
        let outside: Vec<&OutRow> = rows.iter().filter(|r| r.violation > 0.0).collect();

        let mut worst = (0.0f64, 0usize);
        let mut changed = 0usize;
        for (i, r) in outside.iter().enumerate() {
            let d = (r.from_f32 - r.cpu).abs();
            if d > worst.0 {
                worst = (d, i);
            }
            if r.from_f32 != r.honest {
                changed += 1;
            }
        }
        assert!(
            changed > 0,
            "the mutation changed no row: the kernel returned the same energy \
             whichever violation it was handed, so the upload is not being read \
             and the band below is not being tested against anything"
        );
        let r = outside[worst.1];
        // A band is violated when *any* row misses by more than it, not when
        // the largest absolute miss does. The largest absolute miss is always
        // the deepest-protruding pose, where the penalty is thousands of
        // kcal/mol and even a tight relative band is wide in absolute terms;
        // asserting on that row would be asserting that a band proportional
        // to the term it polices is narrower than the term. The ratio is the
        // honest measure, and it is the rows nearest the face that fail it --
        // which is where the defect is.
        let mut over = 0usize;
        let mut worst_ratio = (0.0f64, 0usize);
        for (i, r) in outside.iter().enumerate() {
            let d = (r.from_f32 - r.cpu).abs();
            if d > r.band {
                over += 1;
            }
            let ratio = d / r.band.max(f64::MIN_POSITIVE);
            if ratio > worst_ratio.0 {
                worst_ratio = (ratio, i);
            }
        }
        let wr = outside[worst_ratio.1];
        assert!(
            over > 0,
            "restoring the f32-derived violation moved {changed} of {} rows, but \
             not one of them lands outside the band: the worst was {:.3e}x its \
             band, at {:?} (miss {:.3e}, band {:.3e}). The band is not tight \
             enough to catch the defect it was tightened for, so \
             `the_out_of_box_magnitude_is_the_hosts_and_putting_it_back_breaks_the_band` \
             is not holding anything either",
            outside.len(),
            worst_ratio.0,
            wr.p,
            (wr.from_f32 - wr.cpu).abs(),
            wr.band
        );
        eprintln!(
            "MEASURED band, tight direction: on adapter {}, {changed} of {} out-of-box rows \
             changed when the kernel was handed the f32-derived violation, and {over} of \
             them landed outside the band -- worst by a factor of {:.3e}, at {:?} \
             (violation {:.3e} A, cpu {:.6}, mutated gpu {:.6}, band {:.3e}). The row with \
             the largest *absolute* miss is {r:?} at {:.3e}, which is inside its band \
             of {:.3e} because the penalty there is {:.1} kcal/mol: a band \
             proportional to the term it polices has to be wide there. That is the \
             difference between this band and the old one, which was \
             {:.3e} there and still admitted the defect on the near-face rows.",
            ctx.adapter_name(),
            outside.len(),
            worst_ratio.0,
            wr.p,
            wr.violation,
            wr.cpu,
            wr.from_f32,
            wr.band,
            (r.from_f32 - r.cpu).abs(),
            r.band,
            r.cpu,
            r.old_band
        );
    }

    /// Does the residual invert the *ordering* of two poses?
    ///
    /// Energy agreement is not ranking agreement, and the previous round said
    /// so and then measured only energy. This measures the ordering, on the
    /// out-of-box face, for every pair of the sweep whose CPU energies are
    /// separated by more than the band -- below that the two backends are not
    /// expected to tell the pair apart at all, and an "inversion" there is not
    /// a disagreement about the order, it is a disagreement about whether
    /// there is one.
    ///
    /// A batch scorer's entire reason to exist is to order poses, so this is
    /// the claim that matters and the one the magnitude defect was actually
    /// live in: two poses a rounding step apart on the face were charged
    /// `1000 x dv` on the CPU and `1000 x 0` on the GPU, which is a sign
    /// error rather than a magnitude error whenever they straddle it.
    #[test]
    fn the_out_of_box_residual_does_not_reorder_two_poses() {
        if !is_available() {
            eprintln!(
                "NOT MEASURED: this machine has no GPU adapter, so no ordering claim \
                 was computed on the out-of-box face. This line reports an absence, \
                 not a result."
            );
            return;
        }
        let maps = face_fixture();
        let ligand = one_carbon();
        let placements = out_of_box_placements(&maps);
        let mut ctx = GpuContext::new().expect("gpu context");
        let rows = sweep_out_of_box(&mut ctx, &maps, &ligand, &placements);
        let outside: Vec<&OutRow> = rows.iter().filter(|r| r.violation > 0.0).collect();

        let mut compared = 0usize;
        let mut inversions_honest: Vec<(usize, usize)> = Vec::new();
        let mut inversions_f32: Vec<(usize, usize)> = Vec::new();
        let mut tightest_separated = f64::INFINITY;
        for i in 0..outside.len() {
            for j in i + 1..outside.len() {
                let a = outside[i];
                let b = outside[j];
                let gap = (a.cpu - b.cpu).abs();
                if gap <= a.band.max(b.band) {
                    continue;
                }
                compared += 1;
                tightest_separated = tightest_separated.min(gap);
                if (a.cpu - b.cpu).is_sign_positive() != (a.honest - b.honest).is_sign_positive() {
                    inversions_honest.push((i, j));
                }
                if (a.cpu - b.cpu).is_sign_positive()
                    != (a.from_f32 - b.from_f32).is_sign_positive()
                {
                    inversions_f32.push((i, j));
                }
            }
        }
        assert!(
            compared > 100,
            "only {compared} pairs of {} out-of-box rows were separated by more \
             than the band, so this is not a ranking measurement",
            outside.len()
        );
        assert!(
            inversions_honest.is_empty(),
            "the GPU orders {} of {compared} separated out-of-box pose pairs \
             differently from the CPU, so a batch scorer ranking them on the \
             device would propose a different pose than the search ranking them \
             on the host. First at {:?}: cpu {} vs {}, gpu {} vs {} (adapter {})",
            inversions_honest.len(),
            inversions_honest[0].0,
            outside[inversions_honest[0].0].cpu,
            outside[inversions_honest[0].1].cpu,
            outside[inversions_honest[0].0].honest,
            outside[inversions_honest[0].1].honest,
            ctx.adapter_name()
        );
        eprintln!(
            "MEASURED ranking on adapter {}: {} out-of-box rows, {compared} pairs \
             separated by more than the band (tightest such gap {tightest_separated:e}, \
             against a widest band of {:e}). With the host's violation uploaded, {} of \
             them are ordered differently. With the old f32-derived magnitude, {} of \
             them are -- so the defect was a live ranking input, not only a \
             difference in size.",
            ctx.adapter_name(),
            outside.len(),
            outside.iter().map(|r| r.band).fold(0.0f64, f64::max),
            inversions_honest.len(),
            inversions_f32.len()
        );
    }

    /// The host half of the GPU path, asserted on a machine with no adapter.
    ///
    /// A nested module rather than more functions in `mod tests` so that the
    /// device-gated tests above and these read as two groups: one that needs a
    /// device and one that does not. `use super::*` is the only concession, and
    /// it changes nothing about what the fixtures above are visible to.
    mod host_side {
        use super::*;

        //
        // Everything below runs and asserts on a machine with **no adapter**, because
        // it calls `pack_upload` -- the production upload construction, lifted out of
        // `GpuContext::score_with_violations` -- rather than a device.
        //
        // # Read this before counting these as GPU coverage
        //
        // These tests are not the GPU path. They are the CPU's half of it: the numbers
        // the host decides and hands across. A green here says the *inputs* to the
        // shader are right. It says nothing about what the shader does with them --
        // not the trilinear weights, not the workgroup reduction, not the `u32` index
        // arithmetic, not the driver's rounding. Every one of those is device-side,
        // and on a runner without a GPU none of it is executed by anything.
        //
        // So: these tests make an end-to-end claim *possible to check later* on a
        // machine that has an adapter. They do not discharge it. The claim "the GPU
        // path is verified" is still only supported by the seven tests above, and only
        // when `is_available()` is true. The counting this module's house rule demands
        // -- a skip is a result -- is exactly why that distinction has to be written
        // down here rather than left to the reader: the number of green tests went up
        // and the amount of *device* verification did not change at all.
        // ===========================================================================

        /// The `f32` above and below `v`, by bit pattern, so a neighbour is exact.
        fn f32_neighbours(v: f32) -> (f32, f32) {
            let b = v.to_bits();
            (
                f32::from_bits(b.wrapping_sub(1)),
                f32::from_bits(b.wrapping_add(1)),
            )
        }

        /// A one-conformation batch from `xs` along `+x`, for a one-carbon probe.
        fn x_axis_batch(xs: &[f64]) -> Batch {
            let coords: Vec<f64> = xs.iter().flat_map(|&x| [x, 0.0, 0.0]).collect();
            Batch::from_coords(coords, xs.len(), &one_carbon()).expect("a one-carbon batch")
        }

        /// The uploaded flag, as the shader would read it: `> 0.5`.
        fn uploaded_flag(coords: &[f32], i: usize) -> bool {
            coords[i * COORD_FLOATS + 3] > 0.5
        }

        /// The uploaded position of point `i`.
        fn uploaded_xyz(coords: &[f32], i: usize) -> [f32; 3] {
            [
                coords[i * COORD_FLOATS],
                coords[i * COORD_FLOATS + 1],
                coords[i * COORD_FLOATS + 2],
            ]
        }

        /// The uploaded out-of-box magnitude of point `i`.
        fn uploaded_violation(coords: &[f32], i: usize) -> f32 {
            coords[i * COORD_FLOATS + 4]
        }

        /// The flag is the CPU's own `fractional` answer, on the un-narrowed `f64`.
        ///
        /// This is the assertion that is *only* about the host, and it is the one the
        /// seven device-gated tests above could not make on a machine with no adapter.
        /// For every point: the uploaded flag is exactly `0.0` or `1.0`, and it is
        /// `1.0` exactly when [`crate::grid::GridMaps::fractional`] -- the same call
        /// the CPU's own interpolation makes, on the same `f64` -- says the point is
        /// in the box.
        ///
        /// Exactly `0.0`/`1.0` and not merely "positive" is deliberate: the shader
        /// tests `> 0.5`, and a flag of, say, `1e-7` would pass a truthiness check
        /// here and be read as out-of-box there.
        #[test]
        fn the_uploaded_flag_is_the_fractional_answer_on_the_un_narrowed_coordinate() {
            let maps = face_fixture();
            let box_ = maps.grid_box();
            // Straddle the face, and include well inside and well outside, so both
            // branches are exercised rather than just the interesting one.
            let mut xs = face_placements(&maps, 4);
            xs.push(box_.min[0] - 5.0);
            xs.push(0.0);
            xs.push(box_.max[0] + 5.0);

            let batch = x_axis_batch(&xs);
            let (coords, _) = pack_upload(&batch, &maps, None).expect("pack");
            assert_eq!(coords.len(), xs.len() * COORD_FLOATS);

            let mut saw_in = 0usize;
            let mut saw_out = 0usize;
            for (i, &x) in xs.iter().enumerate() {
                let p = [x, 0.0, 0.0];
                let expect_in = maps.fractional(p).is_some();
                let flag = coords[i * COORD_FLOATS + 3];
                assert!(
                    flag == 0.0 || flag == 1.0,
                    "point {i} at x={x:.12}: the uploaded flag is {flag:?}, which is neither \
                 0.0 nor 1.0. energy.wgsl tests `> 0.5`, so a flag outside {{0.0, 1.0}} \
                 could be read as the opposite of what the host meant."
                );
                assert_eq!(
                    uploaded_flag(&coords, i),
                    expect_in,
                    "point {i} at x={x:.12} (face at {}): the host uploaded a flag saying \
                 {} but `fractional` on the un-narrowed f64 says {}. The flag is the \
                 CPU's own decision and it must be the CPU's own answer.",
                    maps.min[0] + (maps.dims()[0] - 1) as f64 * maps.spacing[0],
                    if uploaded_flag(&coords, i) {
                        "in-box"
                    } else {
                        "out-of-box"
                    },
                    if expect_in { "in-box" } else { "out-of-box" },
                );
                if expect_in {
                    saw_in += 1;
                } else {
                    saw_out += 1;
                }
            }
            // Both branches, or the loop above proved only half the predicate.
            assert!(
                saw_in > 0 && saw_out > 0,
                "the fixture produced {saw_in} in-box and {saw_out} out-of-box points; \
             both have to be non-zero for this to be a test of the branch"
            );
            eprintln!(
                "MEASURED (no device required): {} points packed, {} flagged in-box and {} \
             out-of-box, every flag exactly 0.0 or 1.0 and every one equal to \
             `fractional` on the un-narrowed f64",
                xs.len(),
                saw_in,
                saw_out
            );
        }

        /// The strongest thing this file buys, and it needs no adapter at all.
        ///
        /// The older round's finding was that the `f32` narrowing fed a *discontinuous*
        /// predicate: the kernel derived "am I in the box" from the narrowed `f32`
        /// coordinate, and the CPU from the `f64`, so a point within one `f32` ULP of a
        /// face could take **different branches** on the two sides -- and then charge
        /// completely different penalties for it. The host-side flag is what removed
        /// that.
        ///
        /// Whether the flag is narrowing-derived is a property of *which array the
        /// function reads*, and that is checkable with no device: build two batches
        /// whose uploaded positions are **bit-identical** and whose `f64` inputs are
        /// not, and observe the flag.
        ///
        /// * If the flag were narrowing-derived, the two uploads would be identical,
        ///   because `coords` -- the only thing the kernel can see -- is identical.
        /// * It is not, so the flag moved, so the flag is a function of `coords64`.
        ///
        /// Concretely: take the face fixture and a placement one `f32` ULP inside the
        /// `+x` face, which `f64` says is in the box. The `f32` that represents it also
        /// lies inside. Now nudge the `f64` a hair so that *its nearest `f32`* is the
        /// same value but the `f64` itself is on the other side of the face, and check
        /// that the flag follows the `f64`. `old_f32_inside` is the removed derivation,
        /// transcribed from the shader as it stood, kept here so the test can show the
        /// two derivations genuinely disagree on this fixture rather than agreeing by
        /// accident.
        #[test]
        fn the_flag_is_not_narrowing_derived_and_that_needs_no_adapter_to_show() {
            let maps = face_fixture();
            let ligand = one_carbon();
            let face = maps.min[0] + (maps.dims()[0] - 1) as f64 * maps.spacing[0];

            // The single `f32` bucket whose midpoint sits on the face. Anything inside
            // this bucket narrows to the same `f32` regardless of which side of the
            // face the `f64` is on -- which is the whole mechanism of the defect.
            let bucket = face as f32;
            let (below, above) = f32_neighbours(bucket);
            let eps = (above as f64 - below as f64) / 2.0;
            assert!(
                eps > 0.0,
                "the face must not be exactly an f32: got neighbours {below} and {above}"
            );

            // Two `f64` placements inside the *same* f32 bucket, straddling the face.
            let inside = face - eps * 0.25;
            let outside = face + eps * 0.25;
            assert_eq!(
                inside as f32, bucket,
                "the inside placement must narrow to the bucket's f32"
            );
            assert_eq!(
                outside as f32, bucket,
                "the outside placement must narrow to the SAME f32 as the inside one -- \
             that identity is the defect, and without it this fixture proves nothing"
            );

            // The CPU's answer, from the `f64`.
            assert!(
                maps.fractional([inside, 0.0, 0.0]).is_some(),
                "the inside placement must be in the box for the f64"
            );
            assert!(
                maps.fractional([outside, 0.0, 0.0]).is_none(),
                "the outside placement must be out of the box for the f64"
            );
            // The removed shader's answer, from the `f32`. It cannot tell them apart,
            // and that is precisely why a narrowing-derived flag was wrong.
            assert_eq!(
                old_f32_inside(&maps, inside),
                old_f32_inside(&maps, outside),
                "the removed f32 derivation branches the same way on both placements"
            );
            assert!(
                maps.fractional([inside, 0.0, 0.0]).is_some() != old_f32_inside(&maps, inside),
                "the f32 derivation must disagree with the CPU on at least one of these, \
             or this fixture is not one where the bug was live"
            );

            // Now the actual claim, through the production upload path. Both batches
            // carry the same `f64` *narrowed*; only the `f64` differs.
            let a = x_axis_batch(&[inside]);
            let b = x_axis_batch(&[outside]);
            assert_eq!(
                a.coords, b.coords,
                "the two uploaded position buffers must be bit-identical, or the \
             difference below could come from the position rather than the flag"
            );
            let (ua, _) = pack_upload(&a, &maps, None).expect("pack inside");
            let (ub, _) = pack_upload(&b, &maps, None).expect("pack outside");

            assert_eq!(
                uploaded_xyz(&ua, 0),
                uploaded_xyz(&ub, 0),
                "the uploaded positions are the same f32, as asserted above"
            );
            assert_eq!(
                ua[0..3],
                ub[0..3],
                "the first vec4's position half must be identical"
            );
            assert!(
                uploaded_flag(&ua, 0) != uploaded_flag(&ub, 0),
                "two batches with bit-identical uploaded positions and different f64 \
             inputs produced the SAME in-box flag ({}). The flag is therefore a \
             function of the narrowed coordinate, which is the defect this test \
             exists to rule out.",
                ua[3]
            );
            assert!(
                uploaded_flag(&ua, 0),
                "the f64 in-box placement must upload an in-box flag"
            );
            assert!(
                !uploaded_flag(&ub, 0),
                "the f64 out-of-box placement must upload an out-of-box flag"
            );

            // And the violation, which is the other half of the same branch, followed
            // the `f64` too rather than the `f32`.
            let v_out = uploaded_violation(&ub, 0);
            let want =
                crate::grid::out_of_box_violation_per_axis([outside, 0.0, 0.0], &maps.grid_box())
                    .iter()
                    .sum::<f64>();
            assert_eq!(
                v_out as f64, want as f32 as f64,
                "the uploaded violation must be the host's f64 violation, narrowed once"
            );
            assert!(
                v_out > 0.0,
                "the out-of-box point uploaded a zero violation ({v_out}); the branch and \
             the magnitude have to come from the same f64 or the penalty is charged \
             on the wrong side of the face"
            );
            assert_eq!(
                uploaded_violation(&ua, 0),
                0.0,
                "the in-box point must upload a zero violation: the CPU interpolates and \
             charges nothing there"
            );

            eprintln!(
                "MEASURED (no device required): two placements {inside:.12} and \
             {outside:.12} A, straddling the +x face at {face:.6}, narrow to the same \
             f32 {bucket} and upload bit-identical positions, yet carry different \
             in-box flags ({}) and different violations ({v_out:e} vs 0). The flag is \
             a function of the un-narrowed f64, not of the narrowing.",
                ua[3]
            );
            let _ = ligand;
        }

        /// The uploaded violation is the number the CPU would have charged.
        ///
        /// Not "close to" it. The shader multiplies this by `OUT_OF_BOX_PENALTY`, so
        /// the only rounding between the host's decision and the charge is the one
        /// narrowing below, and a test allowing a wider tolerance than that would
        /// absorb exactly the defect (a magnitude measured from the narrowed
        /// coordinate) it is supposed to catch.
        #[test]
        fn the_uploaded_violation_is_the_number_the_cpu_would_charge() {
            let maps = face_fixture();
            let box_ = maps.grid_box();
            // Far enough out on each axis that the per-axis excess is unambiguous, and
            // one well inside so the zero arm is covered too.
            let cases: Vec<[f64; 3]> = vec![
                [0.0, 0.0, 0.0],
                [box_.max[0] + 3.5, 0.0, 0.0],
                [0.0, box_.min[1] - 2.25, 0.0],
                [0.0, 0.0, box_.max[2] + 11.0],
                [box_.max[0] + 3.5, box_.min[1] - 2.25, box_.max[2] + 11.0],
            ];
            let coords: Vec<f64> = cases.iter().flat_map(|p| p.iter().copied()).collect();
            let batch = Batch::from_coords(coords, cases.len(), &one_carbon()).expect("batch");
            let (uploaded, _) = pack_upload(&batch, &maps, None).expect("pack");

            let mut charged = 0usize;
            for (i, p) in cases.iter().enumerate() {
                let in_box = maps.fractional(*p).is_some();
                let want = if in_box {
                    0.0
                } else {
                    crate::grid::out_of_box_violation_per_axis(*p, &box_)
                        .iter()
                        .sum()
                };
                let got = uploaded_violation(&uploaded, i);
                assert_eq!(
                    got as f64, want as f32 as f64,
                    "point {i} at {p:?}: uploaded violation {got:e}, CPU would charge \
                 {want:e}. energy.wgsl multiplies this by OUT_OF_BOX_PENALTY, so the \
                 host's f64 narrowing is the only rounding permitted here."
                );
                if in_box {
                    assert_eq!(got, 0.0, "point {i} is in the box and must charge nothing");
                } else {
                    assert!(
                        got > 0.0,
                        "point {i} is out of the box and must charge something"
                    );
                    charged += 1;
                }
            }
            assert_eq!(
                charged, 4,
                "the out-of-box arm must have been taken four times"
            );
            eprintln!(
                "MEASURED (no device required): {} placements packed, {} out-of-box, every \
             uploaded violation exactly `f64 as f32` of the CPU's own per-axis excess sum",
                cases.len(),
                charged
            );
        }

        /// The uploaded position is the nearest `f32` to the `f64`, and nothing else.
        ///
        /// Three separate claims, because they fail separately:
        ///   * the upload is `f64 as f32` (the saturating cast, not a wrap);
        ///   * that `f32` really is the *nearest* one, checked against both
        ///     bit-adjacent neighbours rather than assumed from the cast;
        ///   * the host did **not** nudge it, which it used to: a coordinate the kernel
        ///     never asked to move is one fewer way for the two sides to diverge.
        #[test]
        fn the_uploaded_position_is_the_nearest_f32_and_is_not_nudged() {
            let maps = face_fixture();
            let xs = face_placements(&maps, 3);
            let batch = x_axis_batch(&xs);
            let (coords, _) = pack_upload(&batch, &maps, None).expect("pack");

            for (i, &x) in xs.iter().enumerate() {
                let want = x as f32;
                let got = uploaded_xyz(&coords, i)[0];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "point {i} at x={x:.17}: uploaded {got:.9} against the f32 nearest \
                 {want:.9}. A position the host has moved is a position the shader \
                 and the CPU are no longer scoring at the same place."
                );
                // Nearest-ness, proven rather than assumed from the cast: no
                // bit-adjacent f32 is *strictly* closer to the f64 than the one
                // uploaded. Non-strict, and that is not a weakening: this fixture
                // walks `f32` bucket boundaries on purpose, so at the boundary
                // between two buckets the two candidates are exactly equidistant
                // and either is a nearest. A strict `>` here went red on that tie
                // and was wrong to -- a tie is not a nearer neighbour. What would
                // make this fail is a *strictly* nearer f32, which is what a
                // truncating cast or a ULP nudge would produce.
                let (below, above) = f32_neighbours(got);
                for n in [below, above] {
                    assert!(
                        (n as f64 - x).abs() >= (got as f64 - x).abs(),
                        "point {i}: the f32 {n:.9} is strictly closer to {x:.17} than the \
                     uploaded {got:.9}, so the upload is not a nearest f32"
                    );
                }
                // The other two axes were 0.0 in and must be 0.0 out.
                assert_eq!(uploaded_xyz(&coords, i)[1], 0.0);
                assert_eq!(uploaded_xyz(&coords, i)[2], 0.0);
                // Padding: the shader reads a `vec4` here, and the fourth component is
                // the flag. Everything after it is the violation vec4's padding.
                assert_eq!(coords[i * COORD_FLOATS + 5], 0.0);
                assert_eq!(coords[i * COORD_FLOATS + 6], 0.0);
                assert_eq!(coords[i * COORD_FLOATS + 7], 0.0);
            }
            eprintln!(
                "MEASURED (no device required): {} positions packed, each the bit-exact \
             `f64 as f32` of its coordinate, with no bit-adjacent f32 strictly \
             closer to it and no ULP nudging (this fixture walks f32 bucket \
             boundaries on purpose, so exact ties are expected and are not \
             evidence of a nearer neighbour)",
                xs.len()
            );
        }

        /// The parameter block is the grid's own numbers, in range, unaltered.
        ///
        /// The shader's `u32` index arithmetic and its trilinear weights are device-side
        /// and untested here, but the *inputs* to both are not: `nx`/`ny`/`nz` decide how
        /// far the kernel can address, `min`/`spacing` decide which cell a coordinate
        /// lands in, and `bmax` is the face the host measures the out-of-box charge
        /// against. A mistake in any of them is a wrong energy with no other test able
        /// to see it on a device-free machine.
        #[test]
        fn the_parameter_block_is_the_grids_own_numbers_in_range() {
            let maps = face_fixture();
            let batch = x_axis_batch(&[0.0]);
            let (_, params) = pack_upload(&batch, &maps, None).expect("pack");
            let dims = maps.dims();
            let box_ = maps.grid_box();

            assert_eq!(params.nx as usize, dims[0]);
            assert_eq!(params.ny as usize, dims[1]);
            assert_eq!(params.nz as usize, dims[2]);
            // The published ceiling, applied to the numbers the shader will actually
            // receive rather than to the `dims` that produced them.
            for (n, d) in [("nx", params.nx), ("ny", params.ny), ("nz", params.nz)] {
                assert!(
                    d >= 2,
                    "{n} is {d}; the shader's `n - 2` clamp has no cell to clamp to"
                );
            }
            let points = u128::from(params.nx) * u128::from(params.ny) * u128::from(params.nz);
            assert!(
                points <= crate::grid::MAX_GRID_POINTS as u128,
                "the uploaded dims address {points} points, above the {} energy.wgsl's u32 \
             flat index can reach",
                crate::grid::MAX_GRID_POINTS
            );
            assert_eq!(params.n_atoms as usize, batch.n_atoms);
            assert_eq!(params.n_conf as usize, batch.n_conf);

            // The `f64` -> `f32` narrowing of each, and the padding each vec4 carries.
            assert_eq!(params.min[0], maps.min[0] as f32);
            assert_eq!(params.min[1], maps.min[1] as f32);
            assert_eq!(params.min[2], maps.min[2] as f32);
            assert_eq!(params.spacing[0], maps.spacing[0] as f32);
            assert_eq!(params.spacing[1], maps.spacing[1] as f32);
            assert_eq!(params.spacing[2], maps.spacing[2] as f32);
            assert_eq!(params.bmax[0], box_.max[0] as f32);
            assert_eq!(params.bmax[1], box_.max[1] as f32);
            assert_eq!(params.bmax[2], box_.max[2] as f32);
            for v in [params.min[3], params.spacing[3], params.bmax[3]] {
                assert_eq!(
                    v, 0.0,
                    "the fourth component of each vec4 is padding and must be 0"
                );
            }
            assert_eq!(
                params._pad, [0u32; 3],
                "the 12-byte hole before the first vec4"
            );

            // Finite, and not silently zeroed by a narrowing overflow.
            for (name, v) in [
                ("min", params.min),
                ("spacing", params.spacing),
                ("bmax", params.bmax),
            ] {
                for (k, f) in v.iter().enumerate() {
                    assert!(
                        f.is_finite(),
                        "params.{name}[{k}] is {f:?}; a non-finite parameter makes every \
                     sampled coordinate NaN on the device"
                    );
                }
            }
            assert!(
                params.spacing[0] > 0.0,
                "a zero spacing divides by zero in the shader"
            );
            eprintln!(
                "MEASURED (no device required): params carry {dims:?} points, n_atoms {}, \
             n_conf {}, min {min:?}, spacing {sp:?}, bmax {bmax:?}",
                params.n_atoms,
                params.n_conf,
                min = params.min,
                sp = params.spacing,
                bmax = params.bmax,
            );
        }

        /// The override is honoured, so the mutation above is a real injection.
        ///
        /// `score_with_violations`'s `violations` argument exists for exactly one
        /// caller, a test, and its only purpose is to let a test put the old defect
        /// back. A test that cannot install it is not testing the production path. This
        /// checks the override reaches the upload, on no adapter, so the mutation the
        /// device-gated tests perform is known to be landing where they think.
        #[test]
        fn the_violation_override_reaches_the_upload_without_a_device() {
            let maps = face_fixture();
            let face = maps.min[0] + (maps.dims()[0] - 1) as f64 * maps.spacing[0];
            // A point *just* outside the face, by a fraction of one `f32` ULP. This
            // is the shape of the original defect and it is why the coordinate is
            // not simply `max + 4.0`: at 4.0 out, 7.0 is exactly representable in
            // `f32`, the narrowing changes nothing, and the injection is a no-op.
            // A first attempt used that point and the test caught it -- the
            // "narrowed" measurement came back bit-identical to the honest one, so
            // the mutation would have injected nothing and proved nothing.
            let (below, above) = f32_neighbours(face as f32);
            let eps = (above as f64 - below as f64) / 2.0;
            assert!(eps > 0.0, "the face must not be exactly an f32");
            let out = face + eps * 0.25;

            // The precondition that makes this the defect and not a rounding
            // curiosity: narrowing moves the point back to the face or inside it,
            // so the old kernel's measurement was zero where the CPU charged a
            // penalty. `<=` and not `<`: a quarter-ULP outside rounds back to
            // *exactly* the face, and the excess at the face is zero, so that is
            // the defective measurement just as much as landing inside is. The
            // assertion below pins the measurement rather than the geometry,
            // which is the thing that actually matters.
            assert!(
                ((out as f32) as f64) <= face,
                "narrowing x={out:.17} must land at or inside the face at {face:.9}, got \
             {:.9}; otherwise the narrowed measurement is not the defective one",
                out as f32
            );
            assert!(
                out > face,
                "the honest placement must be strictly outside the face, or there is no \
             penalty for the mutation to lose"
            );
            assert!(
                maps.fractional([out, 0.0, 0.0]).is_none(),
                "the honest placement must be out of the box"
            );

            let batch = x_axis_batch(&[out]);
            let honest = out_of_box_violation_per_axis_sum(&maps, out);

            let (default, _) = pack_upload(&batch, &maps, None).expect("pack default");
            assert_eq!(
                uploaded_violation(&default, 0) as f64,
                honest as f32 as f64,
                "the default path must upload the host's own measurement"
            );

            // The old kernel measured the violation from the *narrowed* coordinate,
            // which for this point is inside the box, so it charged nothing.
            let from_narrowed = out_of_box_violation_per_axis_sum(&maps, out as f32 as f64);
            assert_eq!(
                from_narrowed, 0.0,
                "the narrowed placement is inside the box, so the old measurement is \
             zero; got {from_narrowed:e}"
            );
            assert!(
                from_narrowed < honest,
                "the narrowed measurement must be smaller than the honest one for this \
             fixture to reproduce the defect, got {from_narrowed:e} vs {honest:e}"
            );

            let (mutated, _) =
                pack_upload(&batch, &maps, Some(&[from_narrowed])).expect("pack override");
            assert_eq!(
                uploaded_violation(&mutated, 0),
                0.0,
                "the override must be what gets uploaded, or the mutation is not injected"
            );
            assert_ne!(
                uploaded_violation(&default, 0),
                uploaded_violation(&mutated, 0),
                "the override and the default produced the same upload, so it is not \
             reaching the buffer at all"
            );
            eprintln!(
                "MEASURED (no device required): a point {out:.17} A, {delta:e} A outside \
             the +x face at {face:.9}, narrows to {:.9} -- the face, where the excess \
             is zero. The host uploads {honest:e} and the old f32-derived \
             measurement was 0, a difference of {honest:e} kcal/mol of penalty the \
             device-gated mutation has something to inject.",
                out as f32,
                delta = out - face
            );
        }

        /// The `f64` per-axis excess sum, the host's own arithmetic.
        fn out_of_box_violation_per_axis_sum(maps: &crate::grid::GridMaps, x: f64) -> f64 {
            crate::grid::out_of_box_violation_per_axis([x, 0.0, 0.0], &maps.grid_box())
                .iter()
                .sum()
        }

        /// A mis-sized `violations` slice is refused, on no adapter.
        #[test]
        fn a_mis_sized_violation_override_is_refused_without_a_device() {
            let maps = face_fixture();
            let batch = x_axis_batch(&[0.0, 1.0, 2.0]);
            let err = pack_upload(&batch, &maps, Some(&[0.0, 0.0]))
                .expect_err("2 violations for 3 points must be refused");
            let text = err.to_string();
            assert!(
                text.contains("3 out-of-box violations") && text.contains("got 2"),
                "the refusal must name the expected and found counts, got: {text}"
            );
        }

        /// The upload is sized for the batch, and an empty batch is a real answer.
        ///
        /// `score_with_violations` short-circuits `n_conf == 0 || n_atoms == 0` before
        /// it ever reaches `pack_upload`, so this checks the *other* end: that the
        /// buffer length is exactly `n_points * COORD_FLOATS` and that the binding's
        /// `min_binding_size` still holds for it.
        #[test]
        fn the_upload_is_sized_for_the_batch_and_no_shorter() {
            let maps = face_fixture();
            for n_conf in 1..=4usize {
                let xs: Vec<f64> = (0..n_conf).map(|k| k as f64 * 1.5).collect();
                let batch = x_axis_batch(&xs);
                let (coords, params) = pack_upload(&batch, &maps, None).expect("pack");
                assert_eq!(
                    coords.len(),
                    n_conf * COORD_FLOATS,
                    "{n_conf} conformations"
                );
                assert_eq!(params.n_conf as usize, n_conf);
                assert!(
                    coords.len() * 4 >= 32,
                    "a one-point upload is {} bytes and the binding declares a 32-byte \
                 minimum, so wgpu would reject the bind group",
                    coords.len() * 4
                );
            }
            assert_eq!(
                COORD_FLOATS * 4,
                32,
                "energy.wgsl's `min_binding_size` is 32 bytes"
            );
            eprintln!(
                "MEASURED (no device required): uploads of 1..=4 conformations are exactly \
             n_points * {COORD_FLOATS} f32, and the smallest is 32 bytes, the binding's \
             declared minimum"
            );
        }
    }
}
