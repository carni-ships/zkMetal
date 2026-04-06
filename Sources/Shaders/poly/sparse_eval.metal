// Sparse polynomial evaluation kernels for Metal GPU
//
// Evaluates p(x) = sum_i coeff_i * x^index_i at multiple points simultaneously.
// Optimized for polynomials with few non-zero coefficients relative to degree.
//
// Kernels:
//   sparse_eval_bn254        -- evaluate sparse poly at many points (1 thread per point)
//   sparse_eval_multi_bn254  -- evaluate sparse poly at many points with TG caching
//   sparse_mul_dense_bn254   -- multiply sparse poly by dense poly

#include "../fields/bn254_fr.metal"

// Each sparse term is stored as (index, coeff) where index is a uint32 and coeff is Fr.
struct SparseTerm {
    uint index;
    uint _pad[7]; // align to 32 bytes
    Fr coeff;
};

// ============================================================
// Sparse polynomial evaluated at many points
// ============================================================
// One thread per evaluation point. For each point, compute:
//   result = sum_{j=0..nnz-1} terms[j].coeff * x^terms[j].index
//
// Uses incremental power computation: sorts terms by index (assumed pre-sorted),
// tracks z^prevIdx and multiplies up to z^curIdx to avoid re-computing from scratch.

kernel void sparse_eval_bn254(
    device const uint* term_indices     [[buffer(0)]],  // nnz indices
    device const Fr*   term_coeffs      [[buffer(1)]],  // nnz coefficients
    device const Fr*   points           [[buffer(2)]],  // evaluation points
    device Fr*         results          [[buffer(3)]],  // output evaluations
    constant uint&     nnz              [[buffer(4)]],  // number of non-zero terms
    constant uint&     num_points       [[buffer(5)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Fr x = points[gid];
    Fr result = fr_zero();
    Fr x_pow = fr_one();  // x^0
    uint prev_idx = 0;

    for (uint j = 0; j < nnz; j++) {
        uint cur_idx = term_indices[j];
        // Advance x_pow from x^prev_idx to x^cur_idx
        uint gap = cur_idx - prev_idx;
        for (uint g = 0; g < gap; g++) {
            x_pow = fr_mul(x_pow, x);
        }
        prev_idx = cur_idx;
        result = fr_add(result, fr_mul(term_coeffs[j], x_pow));
    }
    results[gid] = result;
}

// ============================================================
// Sparse eval with threadgroup caching of terms
// ============================================================
// For sparse polys with many evaluation points, cache the sparse terms
// in threadgroup memory so all threads in a group share the same load.

#define SPARSE_TG_MAX_TERMS 256

kernel void sparse_eval_cached_bn254(
    device const uint* term_indices     [[buffer(0)]],
    device const Fr*   term_coeffs      [[buffer(1)]],
    device const Fr*   points           [[buffer(2)]],
    device Fr*         results          [[buffer(3)]],
    constant uint&     nnz              [[buffer(4)]],
    constant uint&     num_points       [[buffer(5)]],
    uint gid                            [[thread_position_in_grid]],
    uint lid                            [[thread_position_in_threadgroup]],
    uint tg_size                        [[threads_per_threadgroup]]
) {
    if (gid >= num_points) return;

    threadgroup uint tg_indices[SPARSE_TG_MAX_TERMS];
    threadgroup Fr   tg_coeffs[SPARSE_TG_MAX_TERMS];

    uint terms_to_cache = min(nnz, uint(SPARSE_TG_MAX_TERMS));

    // Cooperatively load terms into threadgroup memory
    for (uint i = lid; i < terms_to_cache; i += tg_size) {
        tg_indices[i] = term_indices[i];
        tg_coeffs[i]  = term_coeffs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Fr x = points[gid];
    Fr result = fr_zero();
    Fr x_pow = fr_one();
    uint prev_idx = 0;

    // Evaluate using cached terms
    for (uint j = 0; j < terms_to_cache; j++) {
        uint cur_idx = tg_indices[j];
        uint gap = cur_idx - prev_idx;
        for (uint g = 0; g < gap; g++) {
            x_pow = fr_mul(x_pow, x);
        }
        prev_idx = cur_idx;
        result = fr_add(result, fr_mul(tg_coeffs[j], x_pow));
    }

    // Handle overflow terms from device memory (if nnz > SPARSE_TG_MAX_TERMS)
    for (uint j = terms_to_cache; j < nnz; j++) {
        uint cur_idx = term_indices[j];
        uint gap = cur_idx - prev_idx;
        for (uint g = 0; g < gap; g++) {
            x_pow = fr_mul(x_pow, x);
        }
        prev_idx = cur_idx;
        result = fr_add(result, fr_mul(term_coeffs[j], x_pow));
    }

    results[gid] = result;
}

// ============================================================
// Sparse * Dense polynomial multiplication
// ============================================================
// Output[k] = sum over (i,j) where sparse_index[i] + j == k of sparse_coeff[i] * dense[j]
//
// One thread per output coefficient. For each output index k,
// iterate over sparse terms and check if dense[k - sparse_index[i]] exists.

kernel void sparse_mul_dense_bn254(
    device const uint* term_indices     [[buffer(0)]],  // sparse indices (sorted)
    device const Fr*   term_coeffs      [[buffer(1)]],  // sparse coefficients
    device const Fr*   dense            [[buffer(2)]],  // dense polynomial coefficients
    device Fr*         output           [[buffer(3)]],  // output polynomial coefficients
    constant uint&     nnz              [[buffer(4)]],  // number of sparse terms
    constant uint&     dense_len        [[buffer(5)]],  // length of dense poly
    constant uint&     output_len       [[buffer(6)]],  // length of output poly
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= output_len) return;

    Fr sum = fr_zero();

    for (uint i = 0; i < nnz; i++) {
        uint si = term_indices[i];
        if (si > gid) break;  // indices are sorted, no more valid contributions
        uint di = gid - si;
        if (di < dense_len) {
            sum = fr_add(sum, fr_mul(term_coeffs[i], dense[di]));
        }
    }

    output[gid] = sum;
}
