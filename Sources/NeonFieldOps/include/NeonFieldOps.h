#ifndef NEON_FIELD_OPS_H
#define NEON_FIELD_OPS_H

#include <stdint.h>

/// Forward NTT on BabyBear field (p = 0x78000001) using ARM NEON intrinsics.
/// Cooley-Tukey DIT radix-2. Data is modified in-place.
/// @param data Array of n = 2^logN uint32_t elements in [0, p).
/// @param logN Log2 of the transform size (1..27).
void babybear_ntt_neon(uint32_t *data, int logN);

/// Inverse NTT on BabyBear field using ARM NEON intrinsics.
/// Gentleman-Sande DIF radix-2, followed by bit-reversal and 1/n scaling.
/// @param data Array of n = 2^logN uint32_t elements in [0, p).
/// @param logN Log2 of the transform size (1..27).
void babybear_intt_neon(uint32_t *data, int logN);

/// Forward NTT on Goldilocks field (p = 2^64 - 2^32 + 1).
/// Optimized ARM64 scalar with __uint128_t mul+umulh pipelining.
/// @param data Array of n = 2^logN uint64_t elements in [0, p).
/// @param logN Log2 of the transform size (1..32).
void goldilocks_ntt(uint64_t *data, int logN);

/// Inverse NTT on Goldilocks field.
/// DIF + bit-reversal + 1/n scaling.
void goldilocks_intt(uint64_t *data, int logN);

/// Forward NTT on BN254 Fr field (256-bit Montgomery form).
/// Fully unrolled 4-limb CIOS Montgomery multiplication.
/// @param data Array of n * 4 uint64_t values (n elements, 4 limbs each, little-endian).
/// @param logN Log2 of the transform size (1..28).
void bn254_fr_ntt(uint64_t *data, int logN);

/// Inverse NTT on BN254 Fr field.
/// DIF + bit-reversal + 1/n scaling.
void bn254_fr_intt(uint64_t *data, int logN);

/// BN254 G1 Pippenger MSM using optimized C field arithmetic.
/// Multi-threaded windows, mixed affine addition, batch-to-affine via Montgomery's trick.
/// @param points  n affine points as n×8 uint64_t (x[4], y[4] per point, Montgomery form).
/// @param scalars n scalars as n×8 uint32_t (little-endian limbs).
/// @param n       Number of points.
/// @param result  Output projective point: 12 uint64_t (x[4], y[4], z[4]).
void bn254_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                          int n, uint64_t *result);

/// Batch fold generators: result[i] = scalarMul(GL[i], xInv) + scalarMul(GR[i], x)
/// Multi-threaded double-and-add for IPA generator folding.
/// @param GL       halfLen projective points (12 uint64_t each: x[4], y[4], z[4]).
/// @param GR       halfLen projective points.
/// @param x        Scalar (8 × uint32_t, non-Montgomery integer form).
/// @param xInv     Scalar (8 × uint32_t, non-Montgomery integer form).
/// @param halfLen  Number of generator points to fold.
/// @param result   Output halfLen projective points.
void bn254_fold_generators(const uint64_t *GL, const uint64_t *GR,
                            const uint32_t *x, const uint32_t *xInv,
                            int halfLen, uint64_t *result);

/// Point scalar multiplication using C CIOS field arithmetic.
/// @param p       Projective point (12 uint64_t: x[4], y[4], z[4]).
/// @param scalar  8 × uint32_t scalar (non-Montgomery integer form).
/// @param r       Output projective point (12 uint64_t).
void bn254_point_scalar_mul(const uint64_t *p, const uint32_t *scalar, uint64_t *r);

/// Fr inner product: result = sum(a[i] * b[i]).
/// @param a, b   Arrays of n Fr elements (4 uint64_t each, Montgomery form).
/// @param n      Number of elements.
/// @param result Output Fr element (4 uint64_t).
void bn254_fr_inner_product(const uint64_t *a, const uint64_t *b, int n, uint64_t *result);

/// Fr vector fold: out[i] = a[i]*x + b[i]*xInv.
/// @param a, b    Fr arrays (4 uint64_t per element, Montgomery form).
/// @param x, xInv Fr elements (4 uint64_t, Montgomery form).
/// @param n       Number of elements.
/// @param out     Output Fr array (4 uint64_t per element).
void bn254_fr_vector_fold(const uint64_t *a, const uint64_t *b,
                           const uint64_t *x, const uint64_t *xInv,
                           int n, uint64_t *out);

#endif // NEON_FIELD_OPS_H
