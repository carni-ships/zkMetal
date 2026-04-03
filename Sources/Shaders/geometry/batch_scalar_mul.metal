// Batch scalar multiplication: multiply N points by a single scalar
// Used by IPA generator folding: G'_i = xInv * GL_i + x * GR_i
//
// Each GPU thread independently computes scalar * point[tid] using double-and-add.
// Scalars are in INTEGER form (not Montgomery) — 8x32-bit limbs, little-endian.

#include "bn254_curve.metal"

struct ScalarU32 {
    uint v[8]; // 256-bit integer as 8x32-bit limbs (little-endian)
};

// Double-and-add scalar multiplication on GPU
PointProjective gpu_scalar_mul(PointProjective p, ScalarU32 scalar) {
    PointProjective result = point_identity();
    PointProjective base = p;

    for (uint limb = 0; limb < 8; limb++) {
        uint word = scalar.v[limb];
        for (uint bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                result = point_add(result, base);
            }
            base = point_double(base);
            word >>= 1;
        }
    }
    return result;
}

// Kernel: batch_fold_generators
// Computes G'_i = xInv * GL_i + x * GR_i for all i in one GPU dispatch
// Input: GL (halfLen affine points), GR (halfLen affine points)
//        xInv, x as integer-form scalars
// Output: halfLen projective points
kernel void batch_fold_generators(
    device const PointAffine* GL         [[buffer(0)]],
    device const PointAffine* GR         [[buffer(1)]],
    constant ScalarU32& xInv             [[buffer(2)]],
    constant ScalarU32& x                [[buffer(3)]],
    device PointProjective* out          [[buffer(4)]],
    constant uint& halfLen               [[buffer(5)]],
    uint tid                             [[thread_position_in_grid]]
) {
    if (tid >= halfLen) return;
    PointProjective gL = point_from_affine(GL[tid]);
    PointProjective gR = point_from_affine(GR[tid]);
    PointProjective scaled_L = gpu_scalar_mul(gL, xInv);
    PointProjective scaled_R = gpu_scalar_mul(gR, x);
    out[tid] = point_add(scaled_L, scaled_R);
}
