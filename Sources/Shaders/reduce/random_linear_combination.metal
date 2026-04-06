// GPU-accelerated random linear combination (RLC) kernels for BN254 Fr
//
// Kernels:
//   rlc_combine         — weighted sum of k vectors: result[i] = sum(j, vectors[j][i] * powers[j])
//   rlc_combine_strided — same but with vectors packed contiguously (row-major: k vectors of length n)
//
// Architecture:
//   - Each thread computes one output element by accumulating across all k vectors.
//   - Fully parallel over the vector length n (no cross-thread reduction needed).
//   - Powers buffer is read-only and broadcast across all threads (fits in L1).
//   - For large k, each thread iterates over vectors serially (k is typically small: 2-64).

#include "../fields/bn254_fr.metal"

// ============================================================================
// rlc_combine_strided — random linear combination of k vectors
//
// Layout:
//   vectors[j * n + i] = j-th vector, i-th element (row-major, k rows of n elements)
//   powers[j]          = weight for j-th vector
//   output[i]          = sum over j of vectors[j * n + i] * powers[j]
//
// Dispatch: n threads total (one per output element).
// ============================================================================

kernel void rlc_combine_strided(
    device const Fr* vectors     [[buffer(0)]],   // k*n packed row-major
    device const Fr* powers      [[buffer(1)]],   // k weights
    device Fr* output            [[buffer(2)]],   // n results
    constant uint& n             [[buffer(3)]],   // vector length
    constant uint& k             [[buffer(4)]],   // number of vectors
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    Fr acc = fr_zero();
    for (uint j = 0; j < k; j++) {
        Fr val = vectors[j * n + gid];
        Fr w = powers[j];
        acc = fr_add(acc, fr_mul(val, w));
    }
    output[gid] = acc;
}

// ============================================================================
// rlc_combine_ptrs — random linear combination with per-vector buffer offsets
//
// Layout:
//   vectors[]           = concatenated vector data (may have different offsets)
//   offsets[j]          = byte offset into vectors for j-th vector's start
//   powers[j]           = weight for j-th vector
//   output[i]           = sum over j of vectors[offsets[j]/sizeof(Fr) + i] * powers[j]
//
// Dispatch: n threads total.
// ============================================================================

kernel void rlc_combine_ptrs(
    device const Fr* vectors     [[buffer(0)]],   // concatenated data
    device const uint* offsets   [[buffer(1)]],   // element offsets per vector
    device const Fr* powers      [[buffer(2)]],   // k weights
    device Fr* output            [[buffer(3)]],   // n results
    constant uint& n             [[buffer(4)]],   // vector length
    constant uint& k             [[buffer(5)]],   // number of vectors
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    Fr acc = fr_zero();
    for (uint j = 0; j < k; j++) {
        uint base = offsets[j];
        Fr val = vectors[base + gid];
        Fr w = powers[j];
        acc = fr_add(acc, fr_mul(val, w));
    }
    output[gid] = acc;
}

// ============================================================================
// rlc_alpha_powers — precompute [1, alpha, alpha^2, ..., alpha^(count-1)]
//
// Sequential on GPU is fine for small counts; for large counts this is
// a prefix-product (scan) but RLC power counts are typically < 1000.
// We use a simple serial kernel launched with 1 thread.
// ============================================================================

kernel void rlc_alpha_powers(
    device const Fr* alpha       [[buffer(0)]],   // single Fr element
    device Fr* output            [[buffer(1)]],   // count elements
    constant uint& count         [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    Fr a = alpha[0];
    Fr cur = fr_one();
    for (uint i = 0; i < count; i++) {
        output[i] = cur;
        cur = fr_mul(cur, a);
    }
}
