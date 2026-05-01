// P^1 Rational Function FRI fold kernels for Mersenne31 field
//
// Standard FRI folding on multiplicative coset domain:
//   - All folds use t → t² (same fold type, unlike Circle FRI's y-fold + x-fold)
//   - Domain points are elements of a multiplicative subgroup
//   - Butterfly pairs: f[t] with f[-t] (sign pairs square to same value)
//
// Fold formula:
//   g[i] = (f[i] + f[i + n/2]) / 2 + alpha * (f[i] - f[i + n/2]) * inv_2t[i]
// where inv_2t[i] = 1 / (2 * t_i) and t_i are domain points at current level.

#include "../fields/mersenne31.metal"

#ifndef M31_INV2_DEFINED
#define M31_INV2_DEFINED
// Precomputed inverse of 2 mod p = (p+1)/2
constant uint M31_INV2 = 1073741824u;  // (2^31 - 1 + 1) / 2 = 2^30
#endif

// P^1 FRI fold: standard t → t² folding.
// Pairs f[i] with f[i + n/2] where domain points form sign pairs (±t).
// inv_2t[i] = 1/(2 * t_i) precomputed on CPU.
//
// folded[i] = (f[i] + f[i + n/2]) / 2 + alpha * (f[i] - f[i + n/2]) * inv_2t[i]
kernel void p1_fri_fold(
    device const M31* evals         [[buffer(0)]],
    device M31* folded              [[buffer(1)]],
    device const M31* inv_2t        [[buffer(2)]],  // 1/(2 * t_i) for i in [0, n/2)
    constant M31* alpha             [[buffer(3)]],   // random challenge
    constant uint& n                [[buffer(4)]],   // current domain size (must be even)
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    M31 a = evals[gid];
    M31 b = evals[gid + half_n];

    // sum = (a + b) / 2
    M31 sum_raw = m31_add(a, b);
    M31 half_sum = m31_mul(sum_raw, M31{M31_INV2});

    // diff_term = alpha * (a - b) * inv_2t[i]
    M31 diff = m31_sub(a, b);
    M31 alpha_val = alpha[0];
    M31 alpha_diff = m31_mul(alpha_val, diff);
    M31 diff_term = m31_mul(alpha_diff, inv_2t[gid]);

    folded[gid] = m31_add(half_sum, diff_term);
}

// ============================================================================
// FOLD-BY-4 CASCADE KERNEL
// Fuses 4 consecutive fold rounds into a single GPU dispatch.
// Uses threadgroup memory to store intermediate fold results.
//
// Strategy: Each threadgroup handles 512 elements of the INPUT (n/2 elements).
// Within the threadgroup, we accumulate fold results for each quarter:
// - Q0 threads compute fold positions [0, n/4)
// - Q1 threads compute fold positions [n/4, n/2)
// - Q2 threads compute fold positions [n/2, 3n/4)
// - Q3 threads compute fold positions [3n/4, n)
//
// After reduction within threadgroup, each thread has its final value for
// round 0 (size n/2). Then we repeat for rounds 1, 2, 3.
//
// Output: n/16 elements
// ============================================================================

kernel void p1_fri_fold_by4(
    device const M31* input           [[buffer(0)]],
    device M31* output                 [[buffer(1)]],
    device const M31* inv_2t_0          [[buffer(2)]],  // inv_2t for round 0: size n/2
    device const M31* inv_2t_1          [[buffer(3)]],  // inv_2t for round 1: size n/4
    device const M31* inv_2t_2          [[buffer(4)]],  // inv_2t for round 2: size n/8
    device const M31* inv_2t_3          [[buffer(5)]],  // inv_2t for round 3: size n/16
    constant M31* alphas               [[buffer(6)]],   // 4 alpha values
    constant uint& n                    [[buffer(7)]],   // original domain size
    uint gid                            [[thread_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tg_size                        [[threads_per_threadgroup]]
) {
    uint n0 = n >> 1;      // n/2
    uint n1 = n >> 2;      // n/4
    uint n2 = n >> 3;      // n/8
    uint n3 = n >> 4;      // n/16 (output size)

    if (gid >= n0) return;  // We have n/2 threads

    // Threadgroup memory for intermediate results
    // Each thread needs to store its fold result for all 4 rounds
    threadgroup M31 stage0[512];  // After round 0: n/2 elements
    threadgroup M31 stage1[512];  // After round 1: n/4 elements
    threadgroup M31 stage2[256];  // After round 2: n/8 elements

    // ========================================================================
    // Round 0: n -> n/2
    // Each thread reads its pair and computes one element of the n/2 output
    // ========================================================================
    M31 a0 = input[gid];
    M31 b0 = input[gid + n0];
    M31 sum0 = m31_add(a0, b0);
    M31 half_sum0 = m31_mul(sum0, M31{M31_INV2});
    M31 diff0 = m31_sub(a0, b0);
    M31 alpha0_diff = m31_mul(alphas[0], diff0);
    M31 diff_term0 = m31_mul(alpha0_diff, inv_2t_0[gid]);
    stage0[tid] = m31_add(half_sum0, diff_term0);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Round 1: n/2 -> n/4
    // Threads 0..n/4-1 process pairs from stage0
    // ========================================================================
    if (tid < n1) {
        uint r1_idx = tid;
        uint src_idx = r1_idx + (tid & 1) * n1;  // Pair indices within quarter
        M31 a1 = stage0[src_idx];
        M31 b1 = stage0[src_idx + n1];
        M31 sum1 = m31_add(a1, b1);
        M31 half_sum1 = m31_mul(sum1, M31{M31_INV2});
        M31 diff1 = m31_sub(a1, b1);
        M31 alpha1_diff = m31_mul(alphas[1], diff1);
        M31 diff_term1 = m31_mul(alpha1_diff, inv_2t_1[r1_idx]);
        stage1[tid] = m31_add(half_sum1, diff_term1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Round 2: n/4 -> n/8
    // Threads 0..n/8-1 process pairs from stage1
    // ========================================================================
    if (tid < n2) {
        uint r2_idx = tid;
        uint src_idx = r2_idx + (tid & 1) * n2;
        M31 a2 = stage1[src_idx];
        M31 b2 = stage1[src_idx + n2];
        M31 sum2 = m31_add(a2, b2);
        M31 half_sum2 = m31_mul(sum2, M31{M31_INV2});
        M31 diff2 = m31_sub(a2, b2);
        M31 alpha2_diff = m31_mul(alphas[2], diff2);
        M31 diff_term2 = m31_mul(alpha2_diff, inv_2t_2[r2_idx]);
        stage2[tid] = m31_add(half_sum2, diff_term2);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Round 3: n/8 -> n/16
    // Threads 0..n/16-1 process pairs from stage2
    // Write final output
    // ========================================================================
    if (tid < n3) {
        uint r3_idx = tid;
        uint src_idx = r3_idx + (tid & 1) * n3;
        M31 a3 = stage2[src_idx];
        M31 b3 = stage2[src_idx + n3];
        M31 sum3 = m31_add(a3, b3);
        M31 half_sum3 = m31_mul(sum3, M31{M31_INV2});
        M31 diff3 = m31_sub(a3, b3);
        M31 alpha3_diff = m31_mul(alphas[3], diff3);
        M31 diff_term3 = m31_mul(alpha3_diff, inv_2t_3[r3_idx]);
        output[r3_idx + tgid * n3] = m31_add(half_sum3, diff_term3);
    }
}

// ============================================================================
// FOLD-BY-8 CASCADE KERNEL
// Fuses 8 consecutive fold rounds into a single GPU dispatch.
// Uses threadgroup memory to store intermediate fold results.
//
// Each threadgroup handles 512 elements of the INPUT (n/2 elements).
// Within the threadgroup, we accumulate fold results for each eighth:
// - Q0 threads compute fold positions [0, n/4)
// - Q1 threads compute fold positions [n/4, n/2)
// - Q2 threads compute fold positions [n/2, 3n/4)
// - Q3 threads compute fold positions [3n/4, n)
// Then repeat through rounds 1-7.
//
// Output: n/256 elements
// ============================================================================

// ============================================================================
// FOLD-BY-8 CASCADE KERNEL
// Fuses 8 consecutive fold rounds into a single GPU dispatch.
// Uses threadgroup memory to store intermediate fold results.
//
// WARNING: This kernel has structural issues with threadgroup indexing
// for larger n values. Use fold-by-4 or single-fold for correctness.
// ============================================================================

kernel void p1_fri_fold_by8(
    device const M31* input           [[buffer(0)]],
    device M31* output               [[buffer(1)]],
    device const M31* inv_2t_0       [[buffer(2)]],  // inv_2t for round 0: size n/2
    device const M31* inv_2t_1       [[buffer(3)]],  // inv_2t for round 1: size n/4
    device const M31* inv_2t_2       [[buffer(4)]],  // inv_2t for round 2: size n/8
    device const M31* inv_2t_3       [[buffer(5)]],  // inv_2t for round 3: size n/16
    device const M31* inv_2t_4       [[buffer(6)]],  // inv_2t for round 4: size n/32
    device const M31* inv_2t_5       [[buffer(7)]],  // inv_2t for round 5: size n/64
    device const M31* inv_2t_6       [[buffer(8)]],  // inv_2t for round 6: size n/128
    device const M31* inv_2t_7       [[buffer(9)]],  // inv_2t for round 7: size n/256
    constant M31* alphas             [[buffer(10)]],  // 8 alpha values
    constant uint& n                 [[buffer(11)]],  // original domain size
    // Output buffers for intermediate stages (buffer indices 12-18)
    device M31* stage0_out           [[buffer(12)]],  // n/2 elements
    device M31* stage1_out           [[buffer(13)]],  // n/4 elements
    device M31* stage2_out           [[buffer(14)]],  // n/8 elements
    device M31* stage3_out           [[buffer(15)]],  // n/16 elements
    device M31* stage4_out           [[buffer(16)]],  // n/32 elements
    device M31* stage5_out           [[buffer(17)]],  // n/64 elements
    device M31* stage6_out           [[buffer(18)]],  // n/128 elements
    uint gid                         [[thread_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tgid                        [[threadgroup_position_in_grid]],
    uint tg_size                     [[threads_per_threadgroup]]
) {
    uint n0 = n >> 1;   // n/2
    uint n1 = n >> 2;   // n/4
    uint n2 = n >> 3;   // n/8
    uint n3 = n >> 4;   // n/16
    uint n4 = n >> 5;   // n/32
    uint n5 = n >> 6;   // n/64
    uint n6 = n >> 7;   // n/128
    uint n7 = n >> 8;   // n/256 (output size)

    if (gid >= n0) return;  // We have n/2 threads

    // Threadgroup memory for intermediate results
    threadgroup M31 stage0[512];  // After round 0: n/2 elements
    threadgroup M31 stage1[512];  // After round 1: n/4 elements
    threadgroup M31 stage2[256];  // After round 2: n/8 elements
    threadgroup M31 stage3[256];  // After round 3: n/16 elements
    threadgroup M31 stage4[128];  // After round 4: n/32 elements
    threadgroup M31 stage5[128];  // After round 5: n/64 elements
    threadgroup M31 stage6[64];   // After round 6: n/128 elements

    // ========================================================================
    // Round 0: n -> n/2
    // ========================================================================
    M31 a0 = input[gid];
    M31 b0 = input[gid + n0];
    M31 sum0 = m31_add(a0, b0);
    M31 half_sum0 = m31_mul(sum0, M31{M31_INV2});
    M31 diff0 = m31_sub(a0, b0);
    M31 alpha0_diff = m31_mul(alphas[0], diff0);
    M31 diff_term0 = m31_mul(alpha0_diff, inv_2t_0[gid]);
    stage0[tid] = m31_add(half_sum0, diff_term0);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write stage 0 output
    if (tid < n1) {
        stage0_out[tid + tgid * n1] = stage0[tid];
    }

    // ========================================================================
    // Round 1: n/2 -> n/4 (NOTE: has threadgroup indexing issues)
    // ========================================================================
    if (tid < n1) {
        uint r1_idx = tid;
        uint src_idx = r1_idx + (r1_idx & 1) * n1;
        M31 a1 = stage0[src_idx];
        M31 b1 = stage0[src_idx + n1];
        M31 sum1 = m31_add(a1, b1);
        M31 half_sum1 = m31_mul(sum1, M31{M31_INV2});
        M31 diff1 = m31_sub(a1, b1);
        M31 alpha1_diff = m31_mul(alphas[1], diff1);
        M31 diff_term1 = m31_mul(alpha1_diff, inv_2t_1[r1_idx]);
        stage1[tid] = m31_add(half_sum1, diff_term1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write stage 1 output
    if (tid < n2) {
        stage1_out[tid + tgid * n2] = stage1[tid];
    }

    // ========================================================================
    // Round 2: n/4 -> n/8
    // ========================================================================
    if (tid < n2) {
        uint r2_idx = tid;
        uint src_idx = r2_idx + (r2_idx & 1) * n2;
        M31 a2 = stage1[src_idx];
        M31 b2 = stage1[src_idx + n2];
        M31 sum2 = m31_add(a2, b2);
        M31 half_sum2 = m31_mul(sum2, M31{M31_INV2});
        M31 diff2 = m31_sub(a2, b2);
        M31 alpha2_diff = m31_mul(alphas[2], diff2);
        M31 diff_term2 = m31_mul(alpha2_diff, inv_2t_2[r2_idx]);
        stage2[tid] = m31_add(half_sum2, diff_term2);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write stage 2 output
    if (tid < n3) {
        stage2_out[tid + tgid * n3] = stage2[tid];
    }

    // ========================================================================
    // Round 3: n/8 -> n/16
    // ========================================================================
    if (tid < n3) {
        uint r3_idx = tid;
        uint src_idx = r3_idx + (r3_idx & 1) * n3;
        M31 a3 = stage2[src_idx];
        M31 b3 = stage2[src_idx + n3];
        M31 sum3 = m31_add(a3, b3);
        M31 half_sum3 = m31_mul(sum3, M31{M31_INV2});
        M31 diff3 = m31_sub(a3, b3);
        M31 alpha3_diff = m31_mul(alphas[3], diff3);
        M31 diff_term3 = m31_mul(alpha3_diff, inv_2t_3[r3_idx]);
        stage3[tid] = m31_add(half_sum3, diff_term3);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write stage 3 output
    if (tid < n4) {
        stage3_out[tid + tgid * n4] = stage3[tid];
    }

    // ========================================================================
    // Round 4: n/16 -> n/32
    // ========================================================================
    if (tid < n4) {
        uint r4_idx = tid;
        uint src_idx = r4_idx + (r4_idx & 1) * n4;
        M31 a4 = stage3[src_idx];
        M31 b4 = stage3[src_idx + n4];
        M31 sum4 = m31_add(a4, b4);
        M31 half_sum4 = m31_mul(sum4, M31{M31_INV2});
        M31 diff4 = m31_sub(a4, b4);
        M31 alpha4_diff = m31_mul(alphas[4], diff4);
        M31 diff_term4 = m31_mul(alpha4_diff, inv_2t_4[r4_idx]);
        stage4[tid] = m31_add(half_sum4, diff_term4);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write stage 4 output
    if (tid < n5) {
        stage4_out[tid + tgid * n5] = stage4[tid];
    }

    // ========================================================================
    // Round 5: n/32 -> n/64
    // ========================================================================
    if (tid < n5) {
        uint r5_idx = tid;
        uint src_idx = r5_idx + (r5_idx & 1) * n5;
        M31 a5 = stage4[src_idx];
        M31 b5 = stage4[src_idx + n5];
        M31 sum5 = m31_add(a5, b5);
        M31 half_sum5 = m31_mul(sum5, M31{M31_INV2});
        M31 diff5 = m31_sub(a5, b5);
        M31 alpha5_diff = m31_mul(alphas[5], diff5);
        M31 diff_term5 = m31_mul(alpha5_diff, inv_2t_5[r5_idx]);
        stage5[tid] = m31_add(half_sum5, diff_term5);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write stage 5 output
    if (tid < n6) {
        stage5_out[tid + tgid * n6] = stage5[tid];
    }

    // ========================================================================
    // Round 6: n/64 -> n/128
    // ========================================================================
    if (tid < n6) {
        uint r6_idx = tid;
        uint src_idx = r6_idx + (r6_idx & 1) * n6;
        M31 a6 = stage5[src_idx];
        M31 b6 = stage5[src_idx + n6];
        M31 sum6 = m31_add(a6, b6);
        M31 half_sum6 = m31_mul(sum6, M31{M31_INV2});
        M31 diff6 = m31_sub(a6, b6);
        M31 alpha6_diff = m31_mul(alphas[6], diff6);
        M31 diff_term6 = m31_mul(alpha6_diff, inv_2t_6[r6_idx]);
        stage6[tid] = m31_add(half_sum6, diff_term6);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write stage 6 output
    if (tid < n7) {
        stage6_out[tid + tgid * n7] = stage6[tid];
    }

    // ========================================================================
    // Round 7: n/128 -> n/256
    // ========================================================================
    if (tid < n7) {
        uint r7_idx = tid;
        uint src_idx = r7_idx + (r7_idx & 1) * n7;
        M31 a7 = stage6[src_idx];
        M31 b7 = stage6[src_idx + n7];
        M31 sum7 = m31_add(a7, b7);
        M31 half_sum7 = m31_mul(sum7, M31{M31_INV2});
        M31 diff7 = m31_sub(a7, b7);
        M31 alpha7_diff = m31_mul(alphas[7], diff7);
        M31 diff_term7 = m31_mul(alpha7_diff, inv_2t_7[r7_idx]);
        output[r7_idx + tgid * n7] = m31_add(half_sum7, diff_term7);
    }
}

// ============================================================================
// FOLD-BY-2 CASCADE KERNEL (for cases where we can't do 4)
// ============================================================================

kernel void p1_fri_fold_by2(
    device const M31* input           [[buffer(0)]],
    device M31* output                 [[buffer(1)]],
    device const M31* inv_2t_0          [[buffer(2)]],  // inv_2t for round 0
    device const M31* inv_2t_1          [[buffer(3)]],  // inv_2t for round 1
    constant M31* alphas               [[buffer(4)]],   // 2 alpha values
    constant uint& n                   [[buffer(5)]],   // original domain size
    uint gid                           [[thread_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tg_size                       [[threads_per_threadgroup]]
) {
    uint n1 = n >> 1;      // n/2
    uint n2 = n >> 2;      // n/4

    if (gid >= n1) return;  // n/2 threads

    threadgroup M31 stage0[512];
    threadgroup M31 stage1[256];

    // Round 0: n -> n/2
    M31 a0 = input[gid];
    M31 b0 = input[gid + n1];
    M31 sum0 = m31_add(a0, b0);
    M31 half_sum0 = m31_mul(sum0, M31{M31_INV2});
    M31 diff0 = m31_sub(a0, b0);
    M31 alpha0_diff = m31_mul(alphas[0], diff0);
    M31 diff_term0 = m31_mul(alpha0_diff, inv_2t_0[gid]);
    stage0[tid] = m31_add(half_sum0, diff_term0);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Round 1: n/2 -> n/4, write output
    if (tid < n2) {
        uint r1_idx = tid;
        uint src_idx = r1_idx + (tid & 1) * n2;
        M31 a1 = stage0[src_idx];
        M31 b1 = stage0[src_idx + n2];
        M31 sum1 = m31_add(a1, b1);
        M31 half_sum1 = m31_mul(sum1, M31{M31_INV2});
        M31 diff1 = m31_sub(a1, b1);
        M31 alpha1_diff = m31_mul(alphas[1], diff1);
        M31 diff_term1 = m31_mul(alpha1_diff, inv_2t_1[r1_idx]);
        output[r1_idx + tgid * n2] = m31_add(half_sum1, diff_term1);
    }
}
