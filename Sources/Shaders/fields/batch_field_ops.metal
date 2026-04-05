// Batch field operations for BN254 Fr and BabyBear
//
// Kernels:
//   BN254 (256-bit, 8x32-bit Montgomery form):
//     batch_inverse_bn254  — Montgomery's trick: N inverses with 1 Fermat inverse
//     batch_mul_bn254      — element-wise multiply two arrays
//     batch_add_bn254      — element-wise add two arrays
//     batch_eval_bn254     — evaluate polynomial at N points (Horner per thread)
//
//   BabyBear (32-bit):
//     batch_inverse_bb     — Montgomery's trick for BabyBear
//     batch_mul_bb         — element-wise multiply
//     batch_add_bb         — element-wise add
//     batch_eval_bb        — polynomial evaluation at N points

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// ============================================================
// BN254 Fr batch operations
// ============================================================

// Batch inverse using Montgomery's trick.
// Each threadgroup processes a chunk of up to BATCH_INV_BN254_CHUNK elements.
// Only 1 Fermat inverse (expensive) per chunk; rest is 3*(chunk-1) multiplications.
//
// Algorithm:
//   Phase 1: prefix[0] = a[0], prefix[i] = prefix[i-1] * a[i]
//   Phase 2: inv = prefix[chunk-1]^(-1) via Fermat
//   Phase 3: out[i] = inv * prefix[i-1]; inv *= a[i] (backward sweep)
#define BATCH_INV_BN254_CHUNK 512

kernel void batch_inverse_bn254(
    device const Fr* a          [[buffer(0)]],
    device Fr* out              [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgid                   [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * BATCH_INV_BN254_CHUNK;
    if (base >= n) return;
    uint chunk = min(uint(BATCH_INV_BN254_CHUNK), n - base);

    // Phase 1: prefix products stored in out[]
    out[base] = a[base];
    for (uint i = 1; i < chunk; i++) {
        out[base + i] = fr_mul(out[base + i - 1], a[base + i]);
    }

    // Phase 2: single Fermat inverse of the total product
    Fr inv = fr_inv(out[base + chunk - 1]);

    // Phase 3: backward sweep
    for (uint i = chunk - 1; i > 0; i--) {
        Fr ai = a[base + i];
        out[base + i] = fr_mul(inv, out[base + i - 1]);
        inv = fr_mul(inv, ai);
    }
    out[base] = inv;
}

// Element-wise multiply: out[i] = a[i] * b[i]
kernel void batch_mul_bn254(
    device const Fr* a          [[buffer(0)]],
    device const Fr* b          [[buffer(1)]],
    device Fr* out              [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = fr_mul(a[gid], b[gid]);
}

// Element-wise add: out[i] = a[i] + b[i]
kernel void batch_add_bn254(
    device const Fr* a          [[buffer(0)]],
    device const Fr* b          [[buffer(1)]],
    device Fr* out              [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = fr_add(a[gid], b[gid]);
}

// Evaluate polynomial at N different points using Horner's method.
// One thread per evaluation point — each thread does O(degree) work.
// coeffs[0..degree-1] in ascending order: c0 + c1*x + c2*x^2 + ...
kernel void batch_eval_bn254(
    device const Fr* coeffs     [[buffer(0)]],
    device const Fr* points     [[buffer(1)]],
    device Fr* results          [[buffer(2)]],
    constant uint& degree       [[buffer(3)]],   // number of coefficients
    constant uint& num_points   [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Fr x = points[gid];
    // Horner: start from highest coefficient, work down
    Fr result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = fr_add(fr_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}

// ============================================================
// BabyBear batch operations
// ============================================================

// Batch inverse using Montgomery's trick for BabyBear.
// Same algorithm, but with 32-bit field elements (much cheaper per-op).
#define BATCH_INV_BB_CHUNK 2048

kernel void batch_inverse_bb(
    device const Bb* a          [[buffer(0)]],
    device Bb* out              [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgid                   [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * BATCH_INV_BB_CHUNK;
    if (base >= n) return;
    uint chunk = min(uint(BATCH_INV_BB_CHUNK), n - base);

    // Phase 1: prefix products
    out[base] = a[base];
    for (uint i = 1; i < chunk; i++) {
        out[base + i] = bb_mul(out[base + i - 1], a[base + i]);
    }

    // Phase 2: invert total product via Fermat
    Bb inv = bb_inv(out[base + chunk - 1]);

    // Phase 3: backward sweep
    for (uint i = chunk - 1; i > 0; i--) {
        Bb ai = a[base + i];
        out[base + i] = bb_mul(inv, out[base + i - 1]);
        inv = bb_mul(inv, ai);
    }
    out[base] = inv;
}

// Element-wise multiply: out[i] = a[i] * b[i]
kernel void batch_mul_bb(
    device const Bb* a          [[buffer(0)]],
    device const Bb* b          [[buffer(1)]],
    device Bb* out              [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = bb_mul(a[gid], b[gid]);
}

// Element-wise add: out[i] = a[i] + b[i]
kernel void batch_add_bb(
    device const Bb* a          [[buffer(0)]],
    device const Bb* b          [[buffer(1)]],
    device Bb* out              [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = bb_add(a[gid], b[gid]);
}

// Evaluate polynomial at N different points using Horner's method (BabyBear).
// coeffs[0..degree-1] in ascending order: c0 + c1*x + c2*x^2 + ...
kernel void batch_eval_bb(
    device const Bb* coeffs     [[buffer(0)]],
    device const Bb* points     [[buffer(1)]],
    device Bb* results          [[buffer(2)]],
    constant uint& degree       [[buffer(3)]],
    constant uint& num_points   [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= num_points) return;

    Bb x = points[gid];
    Bb result = coeffs[degree - 1];
    for (uint i = degree - 1; i > 0; i--) {
        result = bb_add(bb_mul(result, x), coeffs[i - 1]);
    }
    results[gid] = result;
}
