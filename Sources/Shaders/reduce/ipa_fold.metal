// GPU-accelerated IPA (Inner Product Argument) fold kernels for BN254 Fr
//
// Kernels:
//   ipa_fold_vectors  — fold two half-vectors: out_i = lo_i + challenge * hi_i
//   ipa_fold_dual     — fold both a and b vectors simultaneously (saves dispatch)
//   ipa_cross_products — compute cross inner products <a_lo, b_hi> and <a_hi, b_lo>
//
// Used by MultiScalarInnerProduct for GPU-accelerated IPA proving/verifying.
// Each round of the IPA protocol halves the vectors using a random challenge.
//
// Architecture:
//   - ipa_fold_vectors: embarrassingly parallel, one thread per output element
//   - ipa_fold_dual: folds a with (x, x_inv) and b with (x_inv, x) in one dispatch
//   - ipa_cross_products: fused multiply-reduce for cross-term inner products

#include "../fields/bn254_fr.metal"

// ============================================================================
// ipa_fold_vectors — fold a pair of half-vectors using a challenge scalar
//
//   out[i] = lo[i] + challenge * hi[i]    for i in 0..halfLen-1
//
// This is the core fold operation in IPA: given vectors of length n,
// split into lo = v[0..n/2-1] and hi = v[n/2..n-1], produce a vector
// of length n/2.
// ============================================================================

kernel void ipa_fold_vectors(
    device const Fr* lo             [[buffer(0)]],
    device const Fr* hi             [[buffer(1)]],
    device Fr* output               [[buffer(2)]],
    constant Fr& challenge          [[buffer(3)]],
    constant uint& halfLen          [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]]
) {
    if (tid >= halfLen) return;

    Fr lo_val = lo[tid];
    Fr hi_val = hi[tid];
    Fr scaled = fr_mul(hi_val, challenge);
    output[tid] = fr_add(lo_val, scaled);
}

// ============================================================================
// ipa_fold_dual — fold both a and b vectors in a single dispatch
//
//   a_out[i] = a_lo[i] + x * a_hi[i]
//   b_out[i] = b_lo[i] + x_inv * b_hi[i]
//
// The a vector is folded with challenge x, the b vector with x^{-1}.
// Buffers layout:
//   a_data: [a_lo_0, ..., a_lo_{h-1}, a_hi_0, ..., a_hi_{h-1}]
//   b_data: [b_lo_0, ..., b_lo_{h-1}, b_hi_0, ..., b_hi_{h-1}]
//   a_out:  [a_folded_0, ..., a_folded_{h-1}]
//   b_out:  [b_folded_0, ..., b_folded_{h-1}]
// ============================================================================

kernel void ipa_fold_dual(
    device const Fr* a_data         [[buffer(0)]],
    device const Fr* b_data         [[buffer(1)]],
    device Fr* a_out                [[buffer(2)]],
    device Fr* b_out                [[buffer(3)]],
    constant Fr& x                  [[buffer(4)]],
    constant Fr& x_inv              [[buffer(5)]],
    constant uint& halfLen          [[buffer(6)]],
    uint tid                        [[thread_position_in_grid]]
) {
    if (tid >= halfLen) return;

    // Fold a: a'[i] = a[i] + x * a[halfLen + i]
    Fr a_lo = a_data[tid];
    Fr a_hi = a_data[halfLen + tid];
    Fr a_scaled = fr_mul(a_hi, x);
    a_out[tid] = fr_add(a_lo, a_scaled);

    // Fold b: b'[i] = b[i] + x_inv * b[halfLen + i]
    Fr b_lo = b_data[tid];
    Fr b_hi = b_data[halfLen + tid];
    Fr b_scaled = fr_mul(b_hi, x_inv);
    b_out[tid] = fr_add(b_lo, b_scaled);
}

// ============================================================================
// SIMD shuffle helper for Fr (8x uint32) — for cross product reduction
// ============================================================================

inline Fr ipa_fr_simd_shuffle_down(Fr a, uint offset) {
    Fr r;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        r.v[k] = simd_shuffle_down(a.v[k], offset);
    }
    return r;
}

inline Fr ipa_fr_simd_reduce_sum(Fr val, uint lane) {
    #pragma unroll
    for (uint off = 16; off > 0; off >>= 1) {
        Fr other = ipa_fr_simd_shuffle_down(val, off);
        if (lane < off) {
            val = fr_add(val, other);
        }
    }
    return val;
}

// ============================================================================
// ipa_cross_products — compute both cross inner products for one IPA round
//
//   crossL = sum(a_lo[i] * b_hi[i])   (used for L commitment)
//   crossR = sum(a_hi[i] * b_lo[i])   (used for R commitment)
//
// Buffers:
//   a_data: [a_lo, a_hi] contiguous (total length = 2 * halfLen)
//   b_data: [b_lo, b_hi] contiguous
//   output: [crossL, crossR] — two Fr values
//
// Each threadgroup computes partial sums; multi-pass reduces partials.
// output[2*tgid] = partial crossL, output[2*tgid+1] = partial crossR
// ============================================================================

kernel void ipa_cross_products(
    device const Fr* a_data         [[buffer(0)]],
    device const Fr* b_data         [[buffer(1)]],
    device Fr* output               [[buffer(2)]],
    constant uint& halfLen          [[buffer(3)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup Fr simd_partialsL[32];
    threadgroup Fr simd_partialsR[32];

    Fr accL = fr_zero();
    Fr accR = fr_zero();

    // Each thread accumulates strided elements
    uint base = tgid * tg_size;
    for (uint i = base + tid; i < halfLen; i += tg_size) {
        // crossL = <a_lo, b_hi> = sum(a[i] * b[halfLen + i])
        Fr prodL = fr_mul(a_data[i], b_data[halfLen + i]);
        accL = fr_add(accL, prodL);

        // crossR = <a_hi, b_lo> = sum(a[halfLen + i] * b[i])
        Fr prodR = fr_mul(a_data[halfLen + i], b_data[i]);
        accR = fr_add(accR, prodR);
    }

    // SIMD shuffle reduction
    accL = ipa_fr_simd_reduce_sum(accL, simd_lane);
    accR = ipa_fr_simd_reduce_sum(accR, simd_lane);

    uint n_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_partialsL[simd_id] = accL;
        simd_partialsR[simd_id] = accR;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        accL = (simd_lane < n_simd_groups) ? simd_partialsL[simd_lane] : fr_zero();
        accR = (simd_lane < n_simd_groups) ? simd_partialsR[simd_lane] : fr_zero();
        accL = ipa_fr_simd_reduce_sum(accL, simd_lane);
        accR = ipa_fr_simd_reduce_sum(accR, simd_lane);
        if (simd_lane == 0) {
            output[2 * tgid] = accL;
            output[2 * tgid + 1] = accR;
        }
    }
}
