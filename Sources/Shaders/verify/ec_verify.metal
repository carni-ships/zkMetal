// Batched EC point-on-curve verification on GPU
// Each thread checks one affine point: y^2 == x^3 + b (BN254: b=3)
// Exploits Apple Silicon unified memory: CPU writes points into shared MTLBuffer,
// GPU reads and verifies immediately without any data transfer.

#include "../fields/bn254_fp.metal"

// Compare two Fp values for equality
bool fp_eq(Fp a, Fp b) {
    for (uint i = 0; i < 8; i++) {
        if (a.v[i] != b.v[i]) return false;
    }
    return true;
}

// BN254 curve parameter b=3 in Montgomery form
// 3 * R mod p
Fp fp_three_mont() {
    // Precomputed: 3 * (2^256 mod p) mod p
    // = 3 * R_MOD_P mod p
    Fp three_r;
    // 3 in Montgomery form for BN254
    // R_MOD_P = {0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d, 0x666ea36f7879462c, 0x0e0a77c19a07df2f}
    // 3*R mod p computed as fp_add(R, fp_add(R, R))
    // We use a constant here for efficiency
    three_r.v[0] = 0x50ad28d7;
    three_r.v[1] = 0x7a17caa9;
    three_r.v[2] = 0xe15521b9;
    three_r.v[3] = 0x1f6ac17a;
    three_r.v[4] = 0x696bd284;
    three_r.v[5] = 0x334bea4e;
    three_r.v[6] = 0xce179d8e;
    three_r.v[7] = 0x2a1f6744;
    return three_r;
}

// Batch on-curve check: y^2 == x^3 + 3
// Each thread verifies one point independently
kernel void batch_ec_oncurve_check(
    device const Fp* points_x   [[buffer(0)]],
    device const Fp* points_y   [[buffer(1)]],
    device uint* results        [[buffer(2)]],
    constant uint& count        [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Fp x = points_x[tid];
    Fp y = points_y[tid];

    // Identity point (0, 0) is valid
    if (fp_is_zero(x) && fp_is_zero(y)) {
        results[tid] = 1;
        return;
    }

    // y^2
    Fp y2 = fp_sqr(y);

    // x^3 = x * x * x
    Fp x2 = fp_sqr(x);
    Fp x3 = fp_mul_karatsuba(x2, x);

    // x^3 + b (b=3 for BN254, in Montgomery form)
    Fp b = fp_three_mont();
    Fp rhs = fp_add(x3, b);

    // fp_sqr, fp_mul, fp_add all produce fully reduced results
    // so we can compare directly
    results[tid] = fp_eq(y2, rhs) ? 1u : 0u;
}

// Batch result reduction: AND all individual results into a single bool
// Uses threadgroup reduction for efficiency
kernel void reduce_verify_results(
    device const uint* results  [[buffer(0)]],
    device uint* output         [[buffer(1)]],
    constant uint& count        [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]],
    uint tg_size                [[threads_per_threadgroup]],
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint lid                    [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared_result[256];

    // Each thread loads its element (default to 1 = pass)
    uint val = (tid < count) ? results[tid] : 1u;
    shared_result[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction: AND within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_result[lid] = shared_result[lid] & shared_result[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 of each threadgroup writes to output
    if (lid == 0) {
        output[tg_id] = shared_result[0];
    }
}
