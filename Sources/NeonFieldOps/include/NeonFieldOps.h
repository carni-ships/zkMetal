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

/// Convert projective point to affine using C CIOS field ops.
/// @param p      Projective point (12 uint64_t).
/// @param affine Output affine point (8 uint64_t: x[4], y[4]).
/// Batch convert projective points to affine using C CIOS field ops.
/// Uses Montgomery's batch inversion trick (1 inversion for n points).
void bn254_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n);

void bn254_projective_to_affine(const uint64_t *p, uint64_t *affine);

/// MSM from projective points (direct scalar-mul accumulation, no affine conversion).
/// Optimal for small n (IPA rounds). Points in projective form (12 uint64_t each).
void bn254_msm_projective(const uint64_t *points, const uint32_t *scalars,
                           int n, uint64_t *result);

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

/// Batch convert Fr elements from Montgomery form to uint32 limbs.
/// @param mont  n Fr elements (4 uint64_t each, Montgomery form).
/// @param limbs Output: n × 8 uint32_t (integer form, little-endian).
/// @param n     Number of elements.
void bn254_fr_batch_to_limbs(const uint64_t *mont, uint32_t *limbs, int n);

/// Fr synthetic division: quotient = (p(x) - p(z)) / (x - z).
/// @param coeffs Polynomial coefficients (n elements, 4 uint64_t each, Montgomery form).
/// @param z      Evaluation point (4 uint64_t, Montgomery form).
/// @param n      Number of coefficients.
/// @param quotient Output n-1 elements (4 uint64_t each).
void bn254_fr_synthetic_div(const uint64_t *coeffs, const uint64_t *z,
                             int n, uint64_t *quotient);

/// secp256k1 point scalar multiplication using C CIOS field arithmetic.
/// @param p       Projective point (12 uint64_t: x[4], y[4], z[4], Montgomery form).
/// @param scalar  4 × uint64_t scalar (non-Montgomery integer form, little-endian).
/// @param r       Output projective point (12 uint64_t).
void secp256k1_point_scalar_mul(const uint64_t *p, const uint64_t *scalar, uint64_t *r);

/// secp256k1 full projective point addition.
void secp256k1_point_add(const uint64_t *p, const uint64_t *q, uint64_t *r);

/// secp256k1 projective to affine conversion.
void secp256k1_point_to_affine(const uint64_t *p, uint64_t *ax, uint64_t *ay);

/// secp256k1 G1 Pippenger MSM using optimized C field arithmetic.
/// Multi-threaded windows, mixed affine addition, batch-to-affine via Montgomery's trick.
/// @param points  n affine points as n×8 uint64_t (x[4], y[4] per point, Montgomery form).
/// @param scalars n scalars as n×8 uint32_t (little-endian limbs).
/// @param n       Number of points.
/// @param result  Output projective point: 12 uint64_t (x[4], y[4], z[4]).
void secp256k1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                              int n, uint64_t *result);

/// Forward NTT on Goldilocks field using ARM NEON intrinsics.
/// NEON-vectorized add/sub butterflies with interleaved scalar mul for ILP.
/// @param data Array of n = 2^logN uint64_t elements in [0, p).
/// @param logN Log2 of the transform size (1..32).
void goldilocks_ntt_neon(uint64_t *data, int logN);

/// Inverse NTT on Goldilocks field using ARM NEON intrinsics.
/// DIF + bit-reversal + 1/n scaling.
void goldilocks_intt_neon(uint64_t *data, int logN);

/// Batch add: out[i] = a[i] + b[i] mod p, using NEON uint64x2_t.
void gl_batch_add_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/// Batch sub: out[i] = a[i] - b[i] mod p, using NEON uint64x2_t.
void gl_batch_sub_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/// Batch mul: out[i] = a[i] * b[i] mod p, using interleaved scalar for ILP.
void gl_batch_mul_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/// Batch DIT butterfly for NTT stages using NEON add/sub.
void gl_batch_butterfly_neon(uint64_t *data, const uint64_t *twiddles, int halfBlock, int nBlocks);

/// Batch DIF butterfly for inverse NTT stages using NEON add/sub.
void gl_batch_butterfly_dif_neon(uint64_t *data, const uint64_t *twiddles, int halfBlock, int nBlocks);

/// ARM64 assembly CIOS Montgomery multiplication for BN254 Fr.
/// Computes result = a * b * R^{-1} mod p (4-limb, little-endian).
/// @param result  Output 4 x uint64_t.
/// @param a       Input 4 x uint64_t (Montgomery form).
/// @param b       Input 4 x uint64_t (Montgomery form).
void mont_mul_asm(uint64_t *result, const uint64_t *a, const uint64_t *b);

/// C CIOS Montgomery multiplication for BN254 Fr (for benchmarking comparison).
/// Same interface as mont_mul_asm.
void mont_mul_c(uint64_t *result, const uint64_t *a, const uint64_t *b);

/// Batch multiply: data[i] *= multiplier for i in 0..n-1 (ARM64 assembly).
/// Amortizes function call overhead over n multiplications.
/// @param data       Array of n elements (4 uint64_t each, modified in-place).
/// @param multiplier Single element (4 uint64_t).
/// @param n          Number of elements.
void mont_mul_batch_asm(uint64_t *data, const uint64_t *multiplier, int n);

/// Batch pairwise multiply: result[i] = a[i] * b[i] for i in 0..n-1.
void mont_mul_pair_batch_asm(uint64_t *result, const uint64_t *a, const uint64_t *b, int n);

/// C batch multiply (for fair comparison): data[i] *= multiplier.
void mont_mul_batch_c(uint64_t *data, const uint64_t *multiplier, int n);

/// C pair batch multiply: result[i] = a[i] * b[i].
void mont_mul_pair_batch_c(uint64_t *result, const uint64_t *a, const uint64_t *b, int n);

/// Test ASM correctness from C. Returns 0 on success, -1 on failure.
int mont_mul_asm_test(void);

/// Keccak-f1600 permutation (24 rounds on 25x64-bit state) using ARM NEON.
/// @param state  25 x uint64_t (200 bytes), modified in-place.
void keccak_f1600_neon(uint64_t state[25]);

/// Keccak-256 hash of arbitrary input using NEON-optimized f1600.
/// @param input  Input bytes.
/// @param len    Input length in bytes.
/// @param output 32-byte output buffer.
void keccak256_hash_neon(const uint8_t *input, size_t len, uint8_t output[32]);

/// Keccak-256 hash of two concatenated 32-byte inputs (Merkle inner node).
/// Optimized: 64 bytes < 136-byte rate, so only one f1600 call needed.
void keccak256_hash_pair_neon(const uint8_t a[32], const uint8_t b[32], uint8_t output[32]);

/// Batch Keccak-256 hash of n pairs of 32-byte inputs.
/// @param inputs  n x 64 bytes (pairs concatenated).
/// @param outputs n x 32 bytes.
/// @param n       Number of pairs.
void keccak256_batch_hash_pairs_neon(const uint8_t *inputs, uint8_t *outputs, size_t n);

/// Blake3 NEON-optimized parent node hash.
/// Hashes left(32B) || right(32B) -> output(32B) using Blake3 parent compression.
/// @param left   32-byte left child hash.
/// @param right  32-byte right child hash.
/// @param output 32-byte parent hash output.
void blake3_hash_pair_neon(const uint8_t left[32], const uint8_t right[32],
                           uint8_t output[32]);

/// Batch Blake3 parent hashing: n pairs -> n parent hashes.
/// @param inputs  n * 64 bytes (pairs of 32-byte child hashes, contiguous).
/// @param outputs n * 32 bytes output.
/// @param n       Number of pairs.
void blake3_batch_hash_pairs_neon(const uint8_t *inputs, uint8_t *outputs, size_t n);

/// Batch modular add: result[i] = (a[i] + b[i]) mod p for BN254 Fr.
/// All arrays are n elements, each 4 uint64_t in Montgomery form.
/// Branchless conditional subtract, 2x unrolled with prefetch.
void bn254_fr_batch_add_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n);

/// Batch modular subtract: result[i] = (a[i] - b[i]) mod p.
void bn254_fr_batch_sub_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n);

/// Batch negate: result[i] = (-a[i]) mod p = p - a[i].
void bn254_fr_batch_neg_neon(uint64_t *result, const uint64_t *a, int n);

/// Batch scalar multiply: result[i] = a[i] * scalar mod p (Montgomery).
void bn254_fr_batch_mul_scalar_neon(uint64_t *result, const uint64_t *a,
                                     const uint64_t *scalar, int n);

/// Multi-threaded batch add (auto-threads for n >= 4096).
void bn254_fr_batch_add_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n);

/// Multi-threaded batch subtract.
void bn254_fr_batch_sub_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n);

/// Multi-threaded batch negate.
void bn254_fr_batch_neg_parallel(uint64_t *result, const uint64_t *a, int n);

/// Multi-threaded batch scalar multiply.
void bn254_fr_batch_mul_scalar_parallel(uint64_t *result, const uint64_t *a,
                                         const uint64_t *scalar, int n);

#endif // NEON_FIELD_OPS_H
