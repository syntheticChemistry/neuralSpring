// BLAS-evolved CPU-optimized tiled matrix multiplication with double-buffered tiles
// C = A * B where A is (M, K), B is (K, N), C is (M, N)
//
// Optimized for CPU execution via llvmpipe (WGSL -> SPIR-V -> LLVM IR -> x86).
// Each technique mirrors a specific OpenBLAS / BLIS optimization:
//
//   1. 32x32 tiles          — L1 cache blocking (same as BLAS panel size)
//   2. vec4 B-tile storage  — aligned 16-byte loads map to SSE/AVX movaps
//   3. 8x4 micro-kernel     — 8 vec4 accumulators, doubles arithmetic intensity
//                              per byte loaded (matches BLAS Mr x Nr register blocking)
//   4. 4x k-loop unroll     — ILP: CPU overlaps independent FMA chains
//   5. Double-buffered tiles — load NEXT tile while computing on CURRENT;
//                              LLVM can schedule main-memory loads early,
//                              overlapping with FMA execution on cached data
//   6. Explicit fma()       — maps to FMA3 instructions via LLVM
//   7. All threads reach barriers (no early-exit for boundary tiles)
//
// Workgroup: 8x4 = 32 threads, each computing an 8x4 micro-tile
// Per workgroup output: 32x32 elements
// Arithmetic intensity: 32 FMAs per k-step per thread (8 rows x 4 cols)
// Shared memory: 2 × (1024 + 256×4) = ~10 KB (double buffered)

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
const MR: u32 = 8u;
const NR: u32 = 4u;
const WG_X: u32 = 8u;   // TILE / NR
const WG_Y: u32 = 4u;   // TILE / MR
const WG_THREADS: u32 = 32u;
const A_LOADS: u32 = 32u;  // 1024 / 32 threads
const B_LOADS: u32 = 8u;   // 256 vec4 / 32 threads
const BV_PER_ROW: u32 = 8u; // TILE / NR = 32 / 4 = 8 vec4s per tile row

// Double-buffered tiles
var<workgroup> tileA_0: array<f32, 1024>;
var<workgroup> tileA_1: array<f32, 1024>;
var<workgroup> tileBv_0: array<vec4<f32>, 256>;
var<workgroup> tileBv_1: array<vec4<f32>, 256>;

fn load_tile_a(dst: u32, tile_k: u32, wg_row: u32, tid: u32) {
    for (var i = 0u; i < A_LOADS; i++) {
        let flat = tid + i * WG_THREADS;
        let lr = flat / TILE;
        let lc = flat % TILE;
        let a_row = wg_row + lr;
        let a_col = tile_k + lc;
        let val = select(0.0, A[a_row * params.K + a_col],
                         a_row < params.M && a_col < params.K);
        if (dst == 0u) { tileA_0[flat] = val; } else { tileA_1[flat] = val; }
    }
}

fn load_tile_bv(dst: u32, tile_k: u32, wg_col: u32, tid: u32) {
    for (var i = 0u; i < B_LOADS; i++) {
        let flat = tid + i * WG_THREADS;
        let row = flat / BV_PER_ROW;
        let col4 = flat % BV_PER_ROW;
        let b_row = tile_k + row;
        let b_col = wg_col + col4 * NR;
        var v = vec4<f32>(0.0);
        if (b_row < params.K) {
            if (b_col < params.N)      { v.x = B[b_row * params.N + b_col]; }
            if (b_col + 1u < params.N) { v.y = B[b_row * params.N + b_col + 1u]; }
            if (b_col + 2u < params.N) { v.z = B[b_row * params.N + b_col + 2u]; }
            if (b_col + 3u < params.N) { v.w = B[b_row * params.N + b_col + 3u]; }
        }
        if (dst == 0u) { tileBv_0[flat] = v; } else { tileBv_1[flat] = v; }
    }
}

// Micro-kernel: 8 rows × 4 cols (vec4), 4x k-unroll, reading from buffer `src`
fn micro_kernel_8x4(src: u32, base_a: u32, col_idx: u32,
                    a0: ptr<function, vec4<f32>>, a1: ptr<function, vec4<f32>>,
                    a2: ptr<function, vec4<f32>>, a3: ptr<function, vec4<f32>>,
                    a4: ptr<function, vec4<f32>>, a5: ptr<function, vec4<f32>>,
                    a6: ptr<function, vec4<f32>>, a7: ptr<function, vec4<f32>>) {
    for (var k = 0u; k < TILE; k += 4u) {
        // Unrolled k+0
        var bv0: vec4<f32>;
        var ak0: array<f32, 8>;
        if (src == 0u) {
            bv0 = tileBv_0[k * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak0[r] = tileA_0[(base_a + r) * TILE + k]; }
        } else {
            bv0 = tileBv_1[k * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak0[r] = tileA_1[(base_a + r) * TILE + k]; }
        }
        *a0 = fma(vec4(ak0[0]), bv0, *a0);
        *a1 = fma(vec4(ak0[1]), bv0, *a1);
        *a2 = fma(vec4(ak0[2]), bv0, *a2);
        *a3 = fma(vec4(ak0[3]), bv0, *a3);
        *a4 = fma(vec4(ak0[4]), bv0, *a4);
        *a5 = fma(vec4(ak0[5]), bv0, *a5);
        *a6 = fma(vec4(ak0[6]), bv0, *a6);
        *a7 = fma(vec4(ak0[7]), bv0, *a7);

        // Unrolled k+1
        var bv1: vec4<f32>;
        var ak1: array<f32, 8>;
        if (src == 0u) {
            bv1 = tileBv_0[(k + 1u) * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak1[r] = tileA_0[(base_a + r) * TILE + k + 1u]; }
        } else {
            bv1 = tileBv_1[(k + 1u) * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak1[r] = tileA_1[(base_a + r) * TILE + k + 1u]; }
        }
        *a0 = fma(vec4(ak1[0]), bv1, *a0);
        *a1 = fma(vec4(ak1[1]), bv1, *a1);
        *a2 = fma(vec4(ak1[2]), bv1, *a2);
        *a3 = fma(vec4(ak1[3]), bv1, *a3);
        *a4 = fma(vec4(ak1[4]), bv1, *a4);
        *a5 = fma(vec4(ak1[5]), bv1, *a5);
        *a6 = fma(vec4(ak1[6]), bv1, *a6);
        *a7 = fma(vec4(ak1[7]), bv1, *a7);

        // Unrolled k+2
        var bv2: vec4<f32>;
        var ak2: array<f32, 8>;
        if (src == 0u) {
            bv2 = tileBv_0[(k + 2u) * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak2[r] = tileA_0[(base_a + r) * TILE + k + 2u]; }
        } else {
            bv2 = tileBv_1[(k + 2u) * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak2[r] = tileA_1[(base_a + r) * TILE + k + 2u]; }
        }
        *a0 = fma(vec4(ak2[0]), bv2, *a0);
        *a1 = fma(vec4(ak2[1]), bv2, *a1);
        *a2 = fma(vec4(ak2[2]), bv2, *a2);
        *a3 = fma(vec4(ak2[3]), bv2, *a3);
        *a4 = fma(vec4(ak2[4]), bv2, *a4);
        *a5 = fma(vec4(ak2[5]), bv2, *a5);
        *a6 = fma(vec4(ak2[6]), bv2, *a6);
        *a7 = fma(vec4(ak2[7]), bv2, *a7);

        // Unrolled k+3
        var bv3: vec4<f32>;
        var ak3: array<f32, 8>;
        if (src == 0u) {
            bv3 = tileBv_0[(k + 3u) * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak3[r] = tileA_0[(base_a + r) * TILE + k + 3u]; }
        } else {
            bv3 = tileBv_1[(k + 3u) * BV_PER_ROW + col_idx];
            for (var r = 0u; r < 8u; r++) { ak3[r] = tileA_1[(base_a + r) * TILE + k + 3u]; }
        }
        *a0 = fma(vec4(ak3[0]), bv3, *a0);
        *a1 = fma(vec4(ak3[1]), bv3, *a1);
        *a2 = fma(vec4(ak3[2]), bv3, *a2);
        *a3 = fma(vec4(ak3[3]), bv3, *a3);
        *a4 = fma(vec4(ak3[4]), bv3, *a4);
        *a5 = fma(vec4(ak3[5]), bv3, *a5);
        *a6 = fma(vec4(ak3[6]), bv3, *a6);
        *a7 = fma(vec4(ak3[7]), bv3, *a7);
    }
}

@compute @workgroup_size(8, 4)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let wg_row = wg_id.y * TILE;
    let wg_col = wg_id.x * TILE;
    let tid = lid.y * WG_X + lid.x;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var acc4 = vec4<f32>(0.0);
    var acc5 = vec4<f32>(0.0);
    var acc6 = vec4<f32>(0.0);
    var acc7 = vec4<f32>(0.0);

    let num_tiles = (params.K + TILE - 1u) / TILE;
    let base_a = lid.y * MR;
    let col_idx = lid.x;

    // Load first tile into buffer 0
    load_tile_a(0u, 0u, wg_row, tid);
    load_tile_bv(0u, 0u, wg_col, tid);
    workgroupBarrier();

    // Double-buffered main loop: load NEXT while computing CURRENT
    for (var t = 0u; t < num_tiles - 1u; t++) {
        let next_k = (t + 1u) * TILE;
        let cur = t % 2u;
        let nxt = 1u - cur;

        // Prefetch next tile into alternate buffer
        load_tile_a(nxt, next_k, wg_row, tid);
        load_tile_bv(nxt, next_k, wg_col, tid);

        // Compute on current buffer (LLVM can interleave loads with FMAs)
        micro_kernel_8x4(cur, base_a, col_idx,
                         &acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7);

        workgroupBarrier();
    }

    // Process last tile
    if (num_tiles > 0u) {
        let last = (num_tiles - 1u) % 2u;
        micro_kernel_8x4(last, base_a, col_idx,
                         &acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7);
    }

    // --- Write 8x4 micro-tile to C (bounds-checked for edge workgroups) ---
    let out_row = wg_row + lid.y * MR;
    let out_col = wg_col + lid.x * NR;

    if (out_row < params.M) {
        if (out_col < params.N)      { C[out_row * params.N + out_col] = acc0.x; }
        if (out_col + 1u < params.N) { C[out_row * params.N + out_col + 1u] = acc0.y; }
        if (out_col + 2u < params.N) { C[out_row * params.N + out_col + 2u] = acc0.z; }
        if (out_col + 3u < params.N) { C[out_row * params.N + out_col + 3u] = acc0.w; }
    }
    if (out_row + 1u < params.M) {
        if (out_col < params.N)      { C[(out_row + 1u) * params.N + out_col] = acc1.x; }
        if (out_col + 1u < params.N) { C[(out_row + 1u) * params.N + out_col + 1u] = acc1.y; }
        if (out_col + 2u < params.N) { C[(out_row + 1u) * params.N + out_col + 2u] = acc1.z; }
        if (out_col + 3u < params.N) { C[(out_row + 1u) * params.N + out_col + 3u] = acc1.w; }
    }
    if (out_row + 2u < params.M) {
        if (out_col < params.N)      { C[(out_row + 2u) * params.N + out_col] = acc2.x; }
        if (out_col + 1u < params.N) { C[(out_row + 2u) * params.N + out_col + 1u] = acc2.y; }
        if (out_col + 2u < params.N) { C[(out_row + 2u) * params.N + out_col + 2u] = acc2.z; }
        if (out_col + 3u < params.N) { C[(out_row + 2u) * params.N + out_col + 3u] = acc2.w; }
    }
    if (out_row + 3u < params.M) {
        if (out_col < params.N)      { C[(out_row + 3u) * params.N + out_col] = acc3.x; }
        if (out_col + 1u < params.N) { C[(out_row + 3u) * params.N + out_col + 1u] = acc3.y; }
        if (out_col + 2u < params.N) { C[(out_row + 3u) * params.N + out_col + 2u] = acc3.z; }
        if (out_col + 3u < params.N) { C[(out_row + 3u) * params.N + out_col + 3u] = acc3.w; }
    }
    if (out_row + 4u < params.M) {
        if (out_col < params.N)      { C[(out_row + 4u) * params.N + out_col] = acc4.x; }
        if (out_col + 1u < params.N) { C[(out_row + 4u) * params.N + out_col + 1u] = acc4.y; }
        if (out_col + 2u < params.N) { C[(out_row + 4u) * params.N + out_col + 2u] = acc4.z; }
        if (out_col + 3u < params.N) { C[(out_row + 4u) * params.N + out_col + 3u] = acc4.w; }
    }
    if (out_row + 5u < params.M) {
        if (out_col < params.N)      { C[(out_row + 5u) * params.N + out_col] = acc5.x; }
        if (out_col + 1u < params.N) { C[(out_row + 5u) * params.N + out_col + 1u] = acc5.y; }
        if (out_col + 2u < params.N) { C[(out_row + 5u) * params.N + out_col + 2u] = acc5.z; }
        if (out_col + 3u < params.N) { C[(out_row + 5u) * params.N + out_col + 3u] = acc5.w; }
    }
    if (out_row + 6u < params.M) {
        if (out_col < params.N)      { C[(out_row + 6u) * params.N + out_col] = acc6.x; }
        if (out_col + 1u < params.N) { C[(out_row + 6u) * params.N + out_col + 1u] = acc6.y; }
        if (out_col + 2u < params.N) { C[(out_row + 6u) * params.N + out_col + 2u] = acc6.z; }
        if (out_col + 3u < params.N) { C[(out_row + 6u) * params.N + out_col + 3u] = acc6.w; }
    }
    if (out_row + 7u < params.M) {
        if (out_col < params.N)      { C[(out_row + 7u) * params.N + out_col] = acc7.x; }
        if (out_col + 1u < params.N) { C[(out_row + 7u) * params.N + out_col + 1u] = acc7.y; }
        if (out_col + 2u < params.N) { C[(out_row + 7u) * params.N + out_col + 2u] = acc7.z; }
        if (out_col + 3u < params.N) { C[(out_row + 7u) * params.N + out_col + 3u] = acc7.w; }
    }
}
