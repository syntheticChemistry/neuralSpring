// GPU-evolved tiled matrix multiplication with double-buffered tiles
// C = A * B where A is (M, K), B is (K, N), C is (M, N)
//
// Evolved from BarraCUDA's matmul_tiled.wgsl with GPU-specific optimizations:
//
//   1. Double-buffered shared memory tiles
//      - Two sets of tile arrays: load NEXT while computing on CURRENT
//      - On GPU, memory requests and ALU ops run on separate pipelines
//      - Between barriers, the GPU interleaves loads and FMAs
//
//   2. vec4 B-tile storage (same as CPU-evolved shader)
//      - B tile stored as array<vec4<f32>> for coalesced 16-byte reads
//      - GPU memory subsystem handles vec4 natively (128-bit transactions)
//
//   3. 2x2 micro-kernel per thread (register blocking)
//      - Each thread computes a 2×2 output block (4 accumulators)
//      - Doubles arithmetic intensity per shared-memory read
//      - 16×16 workgroup covers 32×32 output tile
//
//   4. 4x k-loop unroll for ILP
//      - GPU warp scheduler overlaps independent FMA chains
//
// Workgroup: 16×16 = 256 threads, each computing a 2×2 output
// Per workgroup output: 32×32 elements
// Shared memory: 2 × (32×32 + 32×8 vec4) = 10 KB

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct MatmulParams {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(3) var<uniform> params: MatmulParams;

const TILE: u32 = 32u;
const WG: u32 = 16u;
const WG_THREADS: u32 = 256u;
const MR: u32 = 2u;
const NR: u32 = 2u;
const TILE_ELEMS: u32 = 1024u;
const A_LOADS: u32 = 4u;   // 1024 / 256
const BV_PER_ROW: u32 = 8u; // 32 / 4
const BV_TOTAL: u32 = 256u; // 32 * 8
const B_LOADS: u32 = 1u;    // 256 / 256

// Double-buffered tiles: ping-pong between 0 and 1
var<workgroup> tileA_0: array<f32, 1024>;
var<workgroup> tileA_1: array<f32, 1024>;
var<workgroup> tileBv_0: array<vec4<f32>, 256>;
var<workgroup> tileBv_1: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let wg_row = wg_id.y * TILE;
    let wg_col = wg_id.x * TILE;
    let tid = lid.y * WG + lid.x;

    // 2×2 micro-kernel: 4 accumulators per thread
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    let num_tiles = (params.K + TILE - 1u) / TILE;

    // Thread's 2×2 output position within the 32×32 tile
    let out_ly = lid.y * MR;
    let out_lx = lid.x * NR;

    // --- Load first tile into buffer 0 ---
    let tile_k_0 = 0u;
    for (var i = 0u; i < A_LOADS; i++) {
        let flat = tid + i * WG_THREADS;
        let lr = flat / TILE;
        let lc = flat % TILE;
        let a_row = wg_row + lr;
        let a_col = tile_k_0 + lc;
        if (a_row < params.M && a_col < params.K) {
            tileA_0[flat] = A[a_row * params.K + a_col];
        } else {
            tileA_0[flat] = 0.0;
        }
    }
    for (var i = 0u; i < B_LOADS; i++) {
        let flat = tid + i * WG_THREADS;
        let row = flat / BV_PER_ROW;
        let col4 = flat % BV_PER_ROW;
        let b_row = tile_k_0 + row;
        let b_col = wg_col + col4 * 4u;
        var v = vec4<f32>(0.0);
        if (b_row < params.K) {
            if (b_col < params.N)      { v.x = B[b_row * params.N + b_col]; }
            if (b_col + 1u < params.N) { v.y = B[b_row * params.N + b_col + 1u]; }
            if (b_col + 2u < params.N) { v.z = B[b_row * params.N + b_col + 2u]; }
            if (b_col + 3u < params.N) { v.w = B[b_row * params.N + b_col + 3u]; }
        }
        tileBv_0[flat] = v;
    }
    workgroupBarrier();

    // --- Main loop: double-buffered compute + prefetch ---
    for (var t = 0u; t < num_tiles - 1u; t++) {
        let next_tile_k = (t + 1u) * TILE;

        // Prefetch NEXT tile into buffer 1 (while computing on buffer 0)
        // GPU memory pipeline runs concurrently with ALU
        if (t % 2u == 0u) {
            // Compute on _0, load _1
            for (var i = 0u; i < A_LOADS; i++) {
                let flat = tid + i * WG_THREADS;
                let lr = flat / TILE;
                let lc = flat % TILE;
                let a_row = wg_row + lr;
                let a_col = next_tile_k + lc;
                if (a_row < params.M && a_col < params.K) {
                    tileA_1[flat] = A[a_row * params.K + a_col];
                } else {
                    tileA_1[flat] = 0.0;
                }
            }
            for (var i = 0u; i < B_LOADS; i++) {
                let flat = tid + i * WG_THREADS;
                let row = flat / BV_PER_ROW;
                let col4 = flat % BV_PER_ROW;
                let b_row = next_tile_k + row;
                let b_col = wg_col + col4 * 4u;
                var v = vec4<f32>(0.0);
                if (b_row < params.K) {
                    if (b_col < params.N)      { v.x = B[b_row * params.N + b_col]; }
                    if (b_col + 1u < params.N) { v.y = B[b_row * params.N + b_col + 1u]; }
                    if (b_col + 2u < params.N) { v.z = B[b_row * params.N + b_col + 2u]; }
                    if (b_col + 3u < params.N) { v.w = B[b_row * params.N + b_col + 3u]; }
                }
                tileBv_1[flat] = v;
            }

            // Compute 2×2 micro-kernel on buffer 0, unrolled 4x
            for (var k = 0u; k < TILE; k += 4u) {
                let a0_k0 = tileA_0[out_ly * TILE + k];
                let a1_k0 = tileA_0[(out_ly + 1u) * TILE + k];
                let bv_k0 = tileBv_0[k * BV_PER_ROW + lid.x / 2u];
                let bi = (lid.x % 2u) * 2u;
                let b0_k0 = select(bv_k0.x, bv_k0.z, bi == 2u);
                let b1_k0 = select(bv_k0.y, bv_k0.w, bi == 2u);
                acc00 = fma(a0_k0, b0_k0, acc00);
                acc01 = fma(a0_k0, b1_k0, acc01);
                acc10 = fma(a1_k0, b0_k0, acc10);
                acc11 = fma(a1_k0, b1_k0, acc11);

                let a0_k1 = tileA_0[out_ly * TILE + k + 1u];
                let a1_k1 = tileA_0[(out_ly + 1u) * TILE + k + 1u];
                let bv_k1 = tileBv_0[(k + 1u) * BV_PER_ROW + lid.x / 2u];
                let b0_k1 = select(bv_k1.x, bv_k1.z, bi == 2u);
                let b1_k1 = select(bv_k1.y, bv_k1.w, bi == 2u);
                acc00 = fma(a0_k1, b0_k1, acc00);
                acc01 = fma(a0_k1, b1_k1, acc01);
                acc10 = fma(a1_k1, b0_k1, acc10);
                acc11 = fma(a1_k1, b1_k1, acc11);

                let a0_k2 = tileA_0[out_ly * TILE + k + 2u];
                let a1_k2 = tileA_0[(out_ly + 1u) * TILE + k + 2u];
                let bv_k2 = tileBv_0[(k + 2u) * BV_PER_ROW + lid.x / 2u];
                let b0_k2 = select(bv_k2.x, bv_k2.z, bi == 2u);
                let b1_k2 = select(bv_k2.y, bv_k2.w, bi == 2u);
                acc00 = fma(a0_k2, b0_k2, acc00);
                acc01 = fma(a0_k2, b1_k2, acc01);
                acc10 = fma(a1_k2, b0_k2, acc10);
                acc11 = fma(a1_k2, b1_k2, acc11);

                let a0_k3 = tileA_0[out_ly * TILE + k + 3u];
                let a1_k3 = tileA_0[(out_ly + 1u) * TILE + k + 3u];
                let bv_k3 = tileBv_0[(k + 3u) * BV_PER_ROW + lid.x / 2u];
                let b0_k3 = select(bv_k3.x, bv_k3.z, bi == 2u);
                let b1_k3 = select(bv_k3.y, bv_k3.w, bi == 2u);
                acc00 = fma(a0_k3, b0_k3, acc00);
                acc01 = fma(a0_k3, b1_k3, acc01);
                acc10 = fma(a1_k3, b0_k3, acc10);
                acc11 = fma(a1_k3, b1_k3, acc11);
            }
        } else {
            // Compute on _1, load _0
            for (var i = 0u; i < A_LOADS; i++) {
                let flat = tid + i * WG_THREADS;
                let lr = flat / TILE;
                let lc = flat % TILE;
                let a_row = wg_row + lr;
                let a_col = next_tile_k + lc;
                if (a_row < params.M && a_col < params.K) {
                    tileA_0[flat] = A[a_row * params.K + a_col];
                } else {
                    tileA_0[flat] = 0.0;
                }
            }
            for (var i = 0u; i < B_LOADS; i++) {
                let flat = tid + i * WG_THREADS;
                let row = flat / BV_PER_ROW;
                let col4 = flat % BV_PER_ROW;
                let b_row = next_tile_k + row;
                let b_col = wg_col + col4 * 4u;
                var v = vec4<f32>(0.0);
                if (b_row < params.K) {
                    if (b_col < params.N)      { v.x = B[b_row * params.N + b_col]; }
                    if (b_col + 1u < params.N) { v.y = B[b_row * params.N + b_col + 1u]; }
                    if (b_col + 2u < params.N) { v.z = B[b_row * params.N + b_col + 2u]; }
                    if (b_col + 3u < params.N) { v.w = B[b_row * params.N + b_col + 3u]; }
                }
                tileBv_0[flat] = v;
            }

            // Compute on buffer 1
            for (var k = 0u; k < TILE; k += 4u) {
                let a0_k0 = tileA_1[out_ly * TILE + k];
                let a1_k0 = tileA_1[(out_ly + 1u) * TILE + k];
                let bv_k0 = tileBv_1[k * BV_PER_ROW + lid.x / 2u];
                let bi = (lid.x % 2u) * 2u;
                let b0_k0 = select(bv_k0.x, bv_k0.z, bi == 2u);
                let b1_k0 = select(bv_k0.y, bv_k0.w, bi == 2u);
                acc00 = fma(a0_k0, b0_k0, acc00);
                acc01 = fma(a0_k0, b1_k0, acc01);
                acc10 = fma(a1_k0, b0_k0, acc10);
                acc11 = fma(a1_k0, b1_k0, acc11);

                let a0_k1 = tileA_1[out_ly * TILE + k + 1u];
                let a1_k1 = tileA_1[(out_ly + 1u) * TILE + k + 1u];
                let bv_k1 = tileBv_1[(k + 1u) * BV_PER_ROW + lid.x / 2u];
                let b0_k1 = select(bv_k1.x, bv_k1.z, bi == 2u);
                let b1_k1 = select(bv_k1.y, bv_k1.w, bi == 2u);
                acc00 = fma(a0_k1, b0_k1, acc00);
                acc01 = fma(a0_k1, b1_k1, acc01);
                acc10 = fma(a1_k1, b0_k1, acc10);
                acc11 = fma(a1_k1, b1_k1, acc11);

                let a0_k2 = tileA_1[out_ly * TILE + k + 2u];
                let a1_k2 = tileA_1[(out_ly + 1u) * TILE + k + 2u];
                let bv_k2 = tileBv_1[(k + 2u) * BV_PER_ROW + lid.x / 2u];
                let b0_k2 = select(bv_k2.x, bv_k2.z, bi == 2u);
                let b1_k2 = select(bv_k2.y, bv_k2.w, bi == 2u);
                acc00 = fma(a0_k2, b0_k2, acc00);
                acc01 = fma(a0_k2, b1_k2, acc01);
                acc10 = fma(a1_k2, b0_k2, acc10);
                acc11 = fma(a1_k2, b1_k2, acc11);

                let a0_k3 = tileA_1[out_ly * TILE + k + 3u];
                let a1_k3 = tileA_1[(out_ly + 1u) * TILE + k + 3u];
                let bv_k3 = tileBv_1[(k + 3u) * BV_PER_ROW + lid.x / 2u];
                let b0_k3 = select(bv_k3.x, bv_k3.z, bi == 2u);
                let b1_k3 = select(bv_k3.y, bv_k3.w, bi == 2u);
                acc00 = fma(a0_k3, b0_k3, acc00);
                acc01 = fma(a0_k3, b1_k3, acc01);
                acc10 = fma(a1_k3, b0_k3, acc10);
                acc11 = fma(a1_k3, b1_k3, acc11);
            }
        }
        workgroupBarrier();
    }

    // --- Compute last tile (whichever buffer it's in) ---
    if (num_tiles > 0u) {
        let last_even = (num_tiles - 1u) % 2u == 0u;
        for (var k = 0u; k < TILE; k++) {
            var a0: f32;
            var a1: f32;
            var bv: vec4<f32>;
            if (last_even) {
                a0 = tileA_0[out_ly * TILE + k];
                a1 = tileA_0[(out_ly + 1u) * TILE + k];
                bv = tileBv_0[k * BV_PER_ROW + lid.x / 2u];
            } else {
                a0 = tileA_1[out_ly * TILE + k];
                a1 = tileA_1[(out_ly + 1u) * TILE + k];
                bv = tileBv_1[k * BV_PER_ROW + lid.x / 2u];
            }
            let bi = (lid.x % 2u) * 2u;
            let b0 = select(bv.x, bv.z, bi == 2u);
            let b1 = select(bv.y, bv.w, bi == 2u);
            acc00 = fma(a0, b0, acc00);
            acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10);
            acc11 = fma(a1, b1, acc11);
        }
    }

    // --- Write 2×2 micro-tile to C ---
    let out_row = wg_row + out_ly;
    let out_col = wg_col + out_lx;

    if (out_row < params.M && out_col < params.N) {
        C[out_row * params.N + out_col] = acc00;
    }
    if (out_row < params.M && out_col + 1u < params.N) {
        C[out_row * params.N + out_col + 1u] = acc01;
    }
    if (out_row + 1u < params.M && out_col < params.N) {
        C[(out_row + 1u) * params.N + out_col] = acc10;
    }
    if (out_row + 1u < params.M && out_col + 1u < params.N) {
        C[(out_row + 1u) * params.N + out_col + 1u] = acc11;
    }
}
