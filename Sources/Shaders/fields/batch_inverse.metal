// Batch modular inverse engine — Montgomery's trick for BN254, BabyBear, Goldilocks
//
// Montgomery's trick: compute N field inverses using only 1 actual inversion + 3(N-1) multiplications.
// Algorithm (per chunk):
//   Phase 1 (forward):  prefix[0] = a[0]; prefix[i] = prefix[i-1] * a[i]
//   Phase 2 (invert):   inv = prefix[chunk-1]^(-1) via Fermat's little theorem
//   Phase 3 (backward): out[i] = inv * prefix[i-1]; inv *= a[i]
//
// Zero handling: zeros are skipped in the product chain (replaced with 1).
// The inverse of 0 is defined as 0 (convention in ZK systems).

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"
#include "../fields/goldilocks.metal"

// ============================================================
// Goldilocks helpers (gl_pow and gl_inv not in base goldilocks.metal)
// ============================================================

Gl gl_pow_u64(Gl base, ulong exp) {
    Gl result = gl_one();
    Gl b = base;
    ulong e = exp;
    while (e > 0) {
        if (e & 1UL) result = gl_mul(result, b);
        b = gl_sqr(b);
        e >>= 1;
    }
    return result;
}

// Inverse via Fermat: a^(p-2) mod p, where p = 2^64 - 2^32 + 1
Gl gl_inv(Gl a) {
    // p - 2 = 0xFFFFFFFEFFFFFFFF
    return gl_pow_u64(a, 0xFFFFFFFEFFFFFFFFUL);
}

// ============================================================
// BN254 Fr batch inverse (with zero handling)
// ============================================================
#define BATCH_INV_BN254_SZ 512

kernel void batch_inverse_bn254_safe(
    device const Fr* a          [[buffer(0)]],
    device Fr* out              [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgid                   [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * BATCH_INV_BN254_SZ;
    if (base >= n) return;
    uint chunk = min(uint(BATCH_INV_BN254_SZ), n - base);

    // Phase 1: prefix products, skipping zeros
    Fr running = fr_one();
    for (uint i = 0; i < chunk; i++) {
        Fr ai = a[base + i];
        if (!fr_is_zero(ai)) {
            running = fr_mul(running, ai);
        }
        out[base + i] = running;  // store prefix product (zeros don't contribute)
    }

    // Phase 2: single Fermat inverse of the total product
    Fr inv = fr_inv(running);

    // Phase 3: backward sweep
    for (uint i = chunk; i > 0; i--) {
        uint idx = base + i - 1;
        Fr ai = a[idx];
        if (fr_is_zero(ai)) {
            out[idx] = fr_zero();  // inverse of 0 = 0
        } else {
            if (i > 1) {
                out[idx] = fr_mul(inv, out[idx - 1]);
            } else {
                out[idx] = inv;
            }
            inv = fr_mul(inv, ai);
        }
    }
}

// ============================================================
// BabyBear batch inverse (with zero handling)
// ============================================================
#define BATCH_INV_BB_SZ 2048

kernel void batch_inverse_bb_safe(
    device const Bb* a          [[buffer(0)]],
    device Bb* out              [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgid                   [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * BATCH_INV_BB_SZ;
    if (base >= n) return;
    uint chunk = min(uint(BATCH_INV_BB_SZ), n - base);

    // Phase 1: prefix products, skipping zeros
    Bb running = bb_one();
    for (uint i = 0; i < chunk; i++) {
        Bb ai = a[base + i];
        if (!bb_is_zero(ai)) {
            running = bb_mul(running, ai);
        }
        out[base + i] = running;
    }

    // Phase 2: single Fermat inverse
    Bb inv = bb_inv(running);

    // Phase 3: backward sweep
    for (uint i = chunk; i > 0; i--) {
        uint idx = base + i - 1;
        Bb ai = a[idx];
        if (bb_is_zero(ai)) {
            out[idx] = bb_zero();
        } else {
            if (i > 1) {
                out[idx] = bb_mul(inv, out[idx - 1]);
            } else {
                out[idx] = inv;
            }
            inv = bb_mul(inv, ai);
        }
    }
}

// ============================================================
// Goldilocks batch inverse (with zero handling)
// ============================================================
#define BATCH_INV_GL_SZ 1024

kernel void batch_inverse_goldilocks(
    device const Gl* a          [[buffer(0)]],
    device Gl* out              [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgid                   [[threadgroup_position_in_grid]]
) {
    if (tid != 0) return;

    uint base = tgid * BATCH_INV_GL_SZ;
    if (base >= n) return;
    uint chunk = min(uint(BATCH_INV_GL_SZ), n - base);

    // Phase 1: prefix products, skipping zeros
    Gl running = gl_one();
    for (uint i = 0; i < chunk; i++) {
        Gl ai = a[base + i];
        if (!gl_is_zero(ai)) {
            running = gl_mul(running, ai);
        }
        out[base + i] = running;
    }

    // Phase 2: single Fermat inverse
    Gl inv = gl_inv(running);

    // Phase 3: backward sweep
    for (uint i = chunk; i > 0; i--) {
        uint idx = base + i - 1;
        Gl ai = a[idx];
        if (gl_is_zero(ai)) {
            out[idx] = gl_zero();
        } else {
            if (i > 1) {
                out[idx] = gl_mul(inv, out[idx - 1]);
            } else {
                out[idx] = inv;
            }
            inv = gl_mul(inv, ai);
        }
    }
}
