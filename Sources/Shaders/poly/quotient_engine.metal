// Quotient Engine GPU kernels
//
// Fused kernels for the vanishing polynomial quotient pipeline:
// 1. Fused Z_H evaluation + element-wise quotient (avoids separate Z_H buffer)
// 2. Quotient splitting into degree-bounded chunks for FRI
//
// The fused kernel computes q(omega^i) = p(omega^i) / z_H(omega^i) in a single
// pass when vanishing inverses are precomputed, or computes Z_H inline.
//
// Supports BN254 Fr (256-bit Montgomery) and BabyBear (32-bit Barrett).

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================
// Fused quotient: out[i] = evals[i] / (coset_point[i]^n - 1)
// ============================================================
// Computes Z_H inline per-thread, avoiding a separate buffer.
// coset_points[i] = g * omega^i (precomputed)
// subgroup_log = log2(trace_length)
// Each thread: compute x^n via repeated squaring, then invert, then multiply.

kernel void quotient_fused_bn254(
    device const Fr* evals              [[buffer(0)]],  // constraint poly evaluations
    device const Fr* coset_points       [[buffer(1)]],  // g * omega^i
    device Fr* out                      [[buffer(2)]],  // quotient evaluations
    constant uint& domain_size          [[buffer(3)]],
    constant uint& subgroup_log         [[buffer(4)]],  // log2(trace_length)
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= domain_size) return;

    // Compute Z_H(coset_points[gid]) = coset_points[gid]^n - 1
    Fr x = coset_points[gid];
    for (uint i = 0; i < subgroup_log; i++) {
        x = fr_sqr(x);
    }
    Fr zh = fr_sub(x, fr_one());

    // Invert Z_H value (per-thread Fermat inversion)
    Fr zh_inv = fr_inv(zh);

    // Quotient = eval * zh_inv
    out[gid] = fr_mul(evals[gid], zh_inv);
}

kernel void quotient_fused_babybear(
    device const Bb* evals              [[buffer(0)]],
    device const Bb* coset_points       [[buffer(1)]],
    device Bb* out                      [[buffer(2)]],
    constant uint& domain_size          [[buffer(3)]],
    constant uint& subgroup_log         [[buffer(4)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= domain_size) return;

    Bb x = coset_points[gid];
    for (uint i = 0; i < subgroup_log; i++) {
        x = bb_sqr(x);
    }
    Bb zh = bb_sub(x, bb_one());
    Bb zh_inv = bb_inv(zh);
    out[gid] = bb_mul(evals[gid], zh_inv);
}

// ============================================================
// Quotient with precomputed vanishing inverses
// ============================================================
// When vanishing inverses are precomputed (via batch inverse on CPU or separate
// GPU pass), this kernel is a simple element-wise multiply.
// Identical to poly_div_by_vanishing but provided here for API clarity.

kernel void quotient_precomputed_bn254(
    device const Fr* evals              [[buffer(0)]],
    device const Fr* zh_inv             [[buffer(1)]],  // precomputed 1/Z_H(coset_point_i)
    device Fr* out                      [[buffer(2)]],
    constant uint& n                    [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = fr_mul(evals[gid], zh_inv[gid]);
}

kernel void quotient_precomputed_babybear(
    device const Bb* evals              [[buffer(0)]],
    device const Bb* zh_inv             [[buffer(1)]],
    device Bb* out                      [[buffer(2)]],
    constant uint& n                    [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = bb_mul(evals[gid], zh_inv[gid]);
}

// ============================================================
// Batch inverse via Montgomery's trick for vanishing inverses
// ============================================================
// Chunk-based batch inverse. Each threadgroup processes one chunk.
// Uses a single Fermat inversion per chunk (amortized cost).

#define QE_BATCH_INV_CHUNK_FR 256
#define QE_BATCH_INV_CHUNK_BB 1024

kernel void quotient_batch_inverse_bn254(
    device const Fr* a                  [[buffer(0)]],
    device Fr* out                      [[buffer(1)]],
    constant uint& n                    [[buffer(2)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tgid                           [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * QE_BATCH_INV_CHUNK_FR;
    uint chunk = min(uint(QE_BATCH_INV_CHUNK_FR), n - base);
    if (chunk == 0) return;

    // Phase 1: prefix products
    out[base] = a[base];
    for (uint i = 1; i < chunk; i++) {
        out[base + i] = fr_mul(out[base + i - 1], a[base + i]);
    }

    // Phase 2: invert total product
    Fr inv = fr_inv(out[base + chunk - 1]);

    // Phase 3: backward sweep
    for (uint i = chunk - 1; i > 0; i--) {
        Fr ai = a[base + i];
        out[base + i] = fr_mul(inv, out[base + i - 1]);
        inv = fr_mul(inv, ai);
    }
    out[base] = inv;
}

kernel void quotient_batch_inverse_babybear(
    device const Bb* a                  [[buffer(0)]],
    device Bb* out                      [[buffer(1)]],
    constant uint& n                    [[buffer(2)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tgid                           [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * QE_BATCH_INV_CHUNK_BB;
    uint chunk = min(uint(QE_BATCH_INV_CHUNK_BB), n - base);
    if (chunk == 0) return;

    out[base] = a[base];
    for (uint i = 1; i < chunk; i++) {
        out[base + i] = bb_mul(out[base + i - 1], a[base + i]);
    }

    Bb inv = bb_inv(out[base + chunk - 1]);

    for (uint i = chunk - 1; i > 0; i--) {
        Bb ai = a[base + i];
        out[base + i] = bb_mul(inv, out[base + i - 1]);
        inv = bb_mul(inv, ai);
    }
    out[base] = inv;
}

// ============================================================
// Z_H evaluation kernel (for precomputing vanishing values)
// ============================================================
// Computes Z_H(coset_points[i]) = coset_points[i]^n - 1

kernel void quotient_zh_eval_bn254(
    device const Fr* coset_points       [[buffer(0)]],
    device Fr* zh_vals                  [[buffer(1)]],
    constant uint& domain_size          [[buffer(2)]],
    constant uint& subgroup_log         [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= domain_size) return;

    Fr x = coset_points[gid];
    for (uint i = 0; i < subgroup_log; i++) {
        x = fr_sqr(x);
    }
    zh_vals[gid] = fr_sub(x, fr_one());
}

kernel void quotient_zh_eval_babybear(
    device const Bb* coset_points       [[buffer(0)]],
    device Bb* zh_vals                  [[buffer(1)]],
    constant uint& domain_size          [[buffer(2)]],
    constant uint& subgroup_log         [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= domain_size) return;

    Bb x = coset_points[gid];
    for (uint i = 0; i < subgroup_log; i++) {
        x = bb_sqr(x);
    }
    zh_vals[gid] = bb_sub(x, bb_one());
}

// ============================================================
// Quotient chunk extraction: extract chunk k from quotient evals
// ============================================================
// For FRI, the quotient polynomial of degree < kN is split into k chunks
// of degree < N. In evaluation form:
//   q_k(omega^i) = sum_{j=0}^{numChunks-1} (omega^i)^{j*chunkSize} * chunk_j(omega^i)
//
// This kernel extracts interleaved samples for degree reduction.
// chunk_out[i] = quotient[i * numChunks + chunkIdx]
// (used after iNTT -> coefficient split -> NTT, or direct evaluation extraction)

kernel void quotient_extract_chunk_bn254(
    device const Fr* quotient           [[buffer(0)]],
    device Fr* chunk_out                [[buffer(1)]],
    constant uint& total_size           [[buffer(2)]],
    constant uint& num_chunks           [[buffer(3)]],
    constant uint& chunk_idx            [[buffer(4)]],
    uint gid                            [[thread_position_in_grid]]
) {
    uint chunk_size = total_size / num_chunks;
    if (gid >= chunk_size) return;
    chunk_out[gid] = quotient[gid * num_chunks + chunk_idx];
}

kernel void quotient_extract_chunk_babybear(
    device const Bb* quotient           [[buffer(0)]],
    device Bb* chunk_out                [[buffer(1)]],
    constant uint& total_size           [[buffer(2)]],
    constant uint& num_chunks           [[buffer(3)]],
    constant uint& chunk_idx            [[buffer(4)]],
    uint gid                            [[thread_position_in_grid]]
) {
    uint chunk_size = total_size / num_chunks;
    if (gid >= chunk_size) return;
    chunk_out[gid] = quotient[gid * num_chunks + chunk_idx];
}
