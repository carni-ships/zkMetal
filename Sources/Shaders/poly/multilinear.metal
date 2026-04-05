// GPU-accelerated multilinear extension (MLE) kernels
// Supports BN254 Fr (8x uint32 Montgomery) and BabyBear (uint32).
//
// Kernel families:
// 1. mle_evaluate_*:      Full MLE evaluation via sequential halving (bind all variables)
// 2. mle_eq_*:            Equality polynomial eq(r, x) for all x in {0,1}^n
// 3. mle_bind_*:          Bind one variable (partial evaluation), halving the table
// 4. mle_tensor_product_*: Tensor product of two MLE evaluation tables

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BN254 Fr field (8x uint32 Montgomery)
// ============================================================================

#include "../fields/bn254_fr.metal"

// --- MLE Bind (partial evaluation) ---
// Given 2^n evaluations, bind variable 0 at value r:
//   out[i] = evals[i] + r * (evals[i + half] - evals[i])
// Halves the table from 2^n to 2^(n-1).
kernel void mle_bind_bn254(
    device const Fr* evals          [[buffer(0)]],
    device Fr* out                  [[buffer(1)]],
    constant Fr* challenge          [[buffer(2)]],
    constant uint& half_n           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    Fr a = evals[gid];
    Fr b = evals[gid + half_n];
    Fr r = challenge[0];

    Fr diff = fr_sub(b, a);
    Fr r_diff = fr_mul(r, diff);
    out[gid] = fr_add(a, r_diff);
}

// --- MLE Evaluate (full evaluation via sequential halving) ---
// This is the same as mle_bind but used iteratively from the Swift side.
// Each dispatch halves the table by one variable. After n dispatches on a
// 2^n-element table, a single Fr element remains: the MLE evaluation.
// (Reuses mle_bind_bn254 kernel — the Swift engine calls it in a loop.)

// --- Equality polynomial: eq(r, x) = prod_i (r_i * x_i + (1 - r_i)(1 - x_i)) ---
// For each x in {0,1}^n (encoded as the thread's global index), compute eq(r, x).
// The point r is passed as an array of n Fr elements.
// Output: 2^n Fr evaluations.
kernel void mle_eq_bn254(
    constant Fr* point              [[buffer(0)]],
    device Fr* out                  [[buffer(1)]],
    constant uint& num_vars         [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint n = num_vars;
    uint total = 1u << n;
    if (gid >= total) return;

    // eq(r, x) = prod_{i=0}^{n-1} (r_i * x_i + (1 - r_i)(1 - x_i))
    // For boolean x_i: if x_i = 1, factor = r_i; if x_i = 0, factor = 1 - r_i
    // Bit i of gid encodes x_{n-1-i} (MSB = variable 0).
    Fr result = fr_one();
    Fr one = fr_one();

    for (uint i = 0; i < n; i++) {
        // Variable i corresponds to bit (n-1-i) of gid
        uint bit = (gid >> (n - 1 - i)) & 1u;
        Fr ri = point[i];
        // factor = bit ? r_i : (1 - r_i)
        Fr factor;
        if (bit) {
            factor = ri;
        } else {
            factor = fr_sub(one, ri);
        }
        result = fr_mul(result, factor);
    }

    out[gid] = result;
}

// --- Tensor product: out[i * size_b + j] = a[i] * b[j] ---
// Total output size = size_a * size_b.
kernel void mle_tensor_product_bn254(
    device const Fr* a              [[buffer(0)]],
    device const Fr* b              [[buffer(1)]],
    device Fr* out                  [[buffer(2)]],
    constant uint& size_a           [[buffer(3)]],
    constant uint& size_b           [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint total = size_a * size_b;
    if (gid >= total) return;

    uint i = gid / size_b;
    uint j = gid % size_b;
    out[gid] = fr_mul(a[i], b[j]);
}


// ============================================================================
// BabyBear field (single uint32, p = 0x78000001)
// ============================================================================

#include "../fields/babybear.metal"

// --- MLE Bind BabyBear ---
kernel void mle_bind_babybear(
    device const uint* evals        [[buffer(0)]],
    device uint* out                [[buffer(1)]],
    constant uint* challenge        [[buffer(2)]],
    constant uint& half_n           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    Bb a = Bb{evals[gid]};
    Bb b_val = Bb{evals[gid + half_n]};
    Bb r = Bb{challenge[0]};

    Bb diff = bb_sub(b_val, a);
    Bb r_diff = bb_mul(r, diff);
    out[gid] = bb_add(a, r_diff).v;
}

// --- Equality polynomial BabyBear ---
kernel void mle_eq_babybear(
    constant uint* point            [[buffer(0)]],
    device uint* out                [[buffer(1)]],
    constant uint& num_vars         [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint n = num_vars;
    uint total = 1u << n;
    if (gid >= total) return;

    Bb result = bb_one();
    Bb one = bb_one();

    for (uint i = 0; i < n; i++) {
        uint bit = (gid >> (n - 1 - i)) & 1u;
        Bb ri = Bb{point[i]};
        Bb factor;
        if (bit) {
            factor = ri;
        } else {
            factor = bb_sub(one, ri);
        }
        result = bb_mul(result, factor);
    }

    out[gid] = result.v;
}

// --- MLE Evaluate BabyBear (same bind kernel, called iteratively from Swift) ---
// Alias: mle_evaluate_babybear is mle_bind_babybear called in a loop.
