// Open Docking — batched grid interpolation and energy summation.
//
// One workgroup scores one conformation: `workgroup_id.x` picks the
// conformation and the local invocation index picks the atom, so a whole
// conformation is reduced in shared memory and written once. That avoids
// atomics entirely — WGSL has no `atomicAdd` on `f32` — and keeps the write
// pattern to exactly one store per workgroup.
//
// Layouts must match dock_core:
//   conformations: (n_conf * n_atoms * 3) f32, atom-major within a conformation
//   atom_data:     (n_atoms * 8) u32 — [type, w0, w1, w2, w3, pad, pad, pad],
//                  with the weights stored as bit-cast f32
//   grid:          point-major, STRIDE = 40 f32 per point
//
// The per-atom weights are packed alongside the element index rather than
// uploaded as their own storage buffer: wgpu's downlevel default allows only
// four storage buffers per shader stage, and this kernel needs five if the
// types and the weights are separate. One buffer also halves the number of
// binds.

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    n_atoms: u32,
    n_conf: u32,
    // WGSL uniform layout gives a vec3 a full 16-byte slot, so pad both.
    min: vec4<f32>,
    spacing: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> energies: array<f32>;

@group(0) @binding(1)
var<storage, read> coords: array<f32>;

@group(0) @binding(2)
var<storage, read> atom_data: array<u32>;

@group(0) @binding(3)
var<storage, read> grid: array<f32>;

@group(0) @binding(4)
var<uniform> params: GridParams;

const MAPS_PER_TYPE: u32 = 4u;
const STRIDE: u32 = 40u; // 10 element types × 4 slots
const ATOM_STRIDE: u32 = 8u; // 1 type + 4 weights + 3 pad

var<workgroup> partial: array<f32, 64>;

/// Trilinear value of one atom at one position, or a large penalty outside.
fn sample(p: vec3<f32>, atom: u32) -> f32 {
    let u = (p - params.min.xyz) / params.spacing.xyz;
    if (any(u < vec3<f32>(0.0))
        || u.x >= f32(params.nx - 1u)
        || u.y >= f32(params.ny - 1u)
        || u.z >= f32(params.nz - 1u)) {
        // Outside the tabulated volume. A large finite value keeps the
        // conformation rankable while making it lose to any legal one; the
        // CPU-side penalty gradient is what actually steers it back inside.
        return 1000.0;
    }

    let base_cell = floor(u);
    let cell = vec3<u32>(base_cell);
    let f = u - base_cell;

    let a0 = atom * ATOM_STRIDE;
    let type_base = atom_data[a0] * MAPS_PER_TYPE;
    let w0 = bitcast<f32>(atom_data[a0 + 1u]);
    let w1 = bitcast<f32>(atom_data[a0 + 2u]);
    let w2 = bitcast<f32>(atom_data[a0 + 3u]);
    let w3 = bitcast<f32>(atom_data[a0 + 4u]);

    var value = 0.0;
    for (var c = 0u; c < 8u; c = c + 1u) {
        let cx = f32(c & 1u);
        let cy = f32((c >> 1u) & 1u);
        let cz = f32((c >> 2u) & 1u);
        let w = select(1.0 - f.x, f.x, cx > 0.5)
            * select(1.0 - f.y, f.y, cy > 0.5)
            * select(1.0 - f.z, f.z, cz > 0.5);

        let idx = (cell.x + u32(cx))
            + params.nx * ((cell.y + u32(cy)) + params.ny * (cell.z + u32(cz)));
        let g = idx * STRIDE + type_base;
        let v = grid[g] * w0 + grid[g + 1u] * w1 + grid[g + 2u] * w2 + grid[g + 3u] * w3;
        value = value + w * v;
    }
    return value;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let conf = wid.x;
    if conf >= params.n_conf {
        return;
    }
    let atom = lid;
    if atom >= params.n_atoms {
        partial[lid] = 0.0;
    } else {
        let o = (conf * params.n_atoms + atom) * 3u;
        partial[lid] = sample(vec3<f32>(coords[o], coords[o + 1u], coords[o + 2u]), atom);
    }
    workgroupBarrier();

    // Binary tree reduction over the workgroup.
    var stride = 32u;
    loop {
        if stride == 0u {
            break;
        }
        if lid < stride {
            partial[lid] = partial[lid] + partial[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if lid == 0u {
        energies[conf] = partial[0];
    }
}
