#ifndef NEON_FIELD_OPS_H
#define NEON_FIELD_OPS_H

#include <stdint.h>
#include <stddef.h>

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

/// Mixed addition: projective P + affine Q (saves 2 muls + 1 sqr vs full projective add).
/// @param p       Projective point (12 uint64_t: x[4], y[4], z[4]).
/// @param q_aff   Affine point (8 uint64_t: x[4], y[4]).
/// @param r       Output projective point (12 uint64_t).
void bn254_point_add_mixed(const uint64_t *p, const uint64_t *q_aff, uint64_t *r);

/// Projective point addition.
/// @param p, q    Projective points (12 uint64_t each).
/// @param r       Output projective point (12 uint64_t).
void bn254_point_add(const uint64_t *p, const uint64_t *q, uint64_t *r);

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

/// Multiply by successive powers: result[i] = a[i] * base^i.
/// @param result Output Fr array (4 uint64_t per element).
/// @param a      Input Fr array (4 uint64_t per element, Montgomery form).
/// @param base   Fr element (4 uint64_t, Montgomery form) — the base of powers.
/// @param n      Number of elements.
void bn254_fr_batch_mul_powers(uint64_t *result, const uint64_t *a,
                               const uint64_t *base, int n);

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

/// Fr Horner polynomial evaluation: result = coeffs[0] + coeffs[1]*z + ... + coeffs[n-1]*z^(n-1).
/// @param coeffs Polynomial coefficients (n elements, 4 uint64_t each, Montgomery form).
/// @param n      Number of coefficients.
/// @param z      Evaluation point (4 uint64_t, Montgomery form).
/// @param result Output single Fr element (4 uint64_t, Montgomery form).
void bn254_fr_horner_eval(const uint64_t *coeffs, int n, const uint64_t z[4],
                           uint64_t result[4]);

/// Fused polynomial evaluation + synthetic division in one pass.
/// Computes eval_out = p(z) and quotient q(x) = (p(x) - p(z)) / (x - z) simultaneously.
/// @param coeffs Polynomial coefficients (n elements, 4 uint64_t each, Montgomery form).
/// @param n      Number of coefficients (must be >= 2 for quotient).
/// @param z      Evaluation point (4 uint64_t, Montgomery form).
/// @param eval_out Output p(z) (4 uint64_t, Montgomery form).
/// @param quotient Output n-1 elements (4 uint64_t each, Montgomery form).
void bn254_fr_eval_and_div(const uint64_t *coeffs, int n, const uint64_t z[4],
                            uint64_t eval_out[4], uint64_t *quotient);

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

/// General-purpose Blake3 hash (single chunk, <=64B input -> 32B output, NEON).
void blake3_hash_neon(const uint8_t *input, size_t len, uint8_t output[32]);

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

/// Batch scalar-minus-vector: result[i] = scalar - a[i].
void bn254_fr_batch_scalar_sub_neon(uint64_t *result, const uint64_t *scalar,
                                     const uint64_t *a, int n);

/// Batch element-wise multiply: result[i] = a[i] * b[i].
void bn254_fr_batch_mul_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n);

/// Parallel batch element-wise multiply: result[i] = a[i] * b[i]. Auto-threads for n>=4096.
void bn254_fr_batch_mul_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n);

/// Fused scalar-multiply-accumulate: result[i] += scalar * a[i].
void bn254_fr_batch_mac_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *scalar, int n);

/// Sumcheck reduce: result[i] = evals[i] + challenge * (evals[halfN+i] - evals[i]).
void bn254_fr_sumcheck_reduce(const uint64_t *evals, const uint64_t *challenge,
                               uint64_t *result, int halfN);

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

/// Vector sum: result = sum(a[i]) for i=0..n-1.
/// a: array of n Fr elements (4 uint64 each, Montgomery form).
/// result: single Fr element (4 uint64).
void bn254_fr_vector_sum(const uint64_t *a, int n, uint64_t result[4]);

/// Batch beta+value: result[i] = beta + values[indices[i]] for i=0..m-1.
/// beta: single Fr element (4 uint64, Montgomery form).
/// values: subtable array (4 uint64 each).
/// indices: m integers indexing into values.
/// result: output array of m Fr elements.
void bn254_fr_batch_beta_add(const uint64_t *beta, const uint64_t *values,
                              const int *indices, int m, uint64_t *result);

/// Batch range-check decomposition: extracts chunk indices from Fr elements.
/// lookups: m Fr elements (4 uint64 each, Montgomery form).
/// indices: output array of size numChunks * m, stored as indices[k*m + i].
void bn254_fr_batch_decompose(const uint64_t *lookups, int m,
                               int numChunks, int bitsPerChunk,
                               int *indices);

/// Batch inverse using Montgomery's trick: out[i] = a[i]^(-1).
/// O(3n) muls + 1 Fermat inversion. Much faster than n individual inversions.
void bn254_fr_batch_inverse(const uint64_t *a, int n, uint64_t *out);

/// Full sumcheck protocol for a single multilinear polynomial.
/// Input: evals[2^numVars], challenges[numVars] (pre-derived).
/// Output: rounds[numVars * 12] = (S(0), S(1), S(2)) per round as 3 Fr each,
///         finalEval[4] = final evaluation. Parallelized for large rounds.
void bn254_fr_full_sumcheck(const uint64_t *evals, int numVars,
                             const uint64_t *challenges,
                             uint64_t *rounds, uint64_t *finalEval);

/// Evaluate multilinear extension at a point.
/// evals: 2^numVars Fr elements. point: numVars Fr elements. result: single Fr.
void bn254_fr_mle_eval(const uint64_t *evals, int numVars,
                        const uint64_t *point, uint64_t result[4]);

/// Compute inverse evaluations: out[i] = 1/(beta + values[i]).
/// Fused beta-add + batch inverse via Montgomery's trick.
void bn254_fr_inverse_evals(const uint64_t beta[4], const uint64_t *values,
                             int n, uint64_t *out);

/// Fused gather + beta-add + batch inverse: out[i] = 1/(beta + subtable[indices[i]]).
/// Avoids separate gather allocation.
void bn254_fr_inverse_evals_indexed(const uint64_t beta[4], const uint64_t *subtable,
                                     const int *indices, int n, uint64_t *out);

/// Compute weighted inverse evaluations: out[j] = weights[j]/(beta + values[j]).
void bn254_fr_weighted_inverse_evals(const uint64_t beta[4], const uint64_t *values,
                                      const uint64_t *weights, int n, uint64_t *out);

/// Fused: compute MLE(1/(beta + subtable[indices[x]]))(point) without materializing inverses.
/// result = sum_x eq(x, point) / (beta + subtable[indices[x]])
void bn254_fr_inverse_mle_eval(const uint64_t beta[4], const uint64_t *subtable,
                                const int *indices, int n, int numVars,
                                const uint64_t *point, uint64_t result[4]);

// ============================================================
// GKR-specific accelerated operations
// ============================================================

/// Compute eq polynomial evaluations for GKR.
/// point: n Fr elements (4 uint64 each). eq: output 2^n Fr elements.
void gkr_eq_poly(const uint64_t *point, int n, uint64_t *eq);

/// Accumulate wiring entries from gates with eq polynomial values.
/// gates: numGates * 3 int32 [type(0=add,1=mul), left, right].
/// eqVals: numGates Fr elements. weight: scaling Fr element.
/// accum: direct-indexed accumulator (9 uint64 per slot: [valid, add[4], mul[4]]).
/// nonzeroIndices: tracks populated slots. numNonzero: in/out count.
void gkr_accumulate_wiring(
    const int32_t *gates, int numGates,
    const uint64_t *eqVals, const uint64_t weight[4],
    int inSize,
    uint64_t *accum, int accumCapacity,
    int32_t *nonzeroIndices, int *numNonzero);

/// In-place MLE fold: v[j] = (1-c)*v[j] + c*v[j+half].
void gkr_mle_fold(uint64_t *v, int half, const uint64_t challenge[4]);

/// GKR sumcheck round evaluation (X-phase).
/// wiring: sorted entries (9 uint64 per entry: [idx, add[4], mul[4]]).
void gkr_sumcheck_round_x(
    const uint64_t *wiring, int numEntries,
    const uint64_t *vx, int vxSize,
    const uint64_t *vy, int vySize,
    int nIn, int halfSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

/// GKR sumcheck round evaluation (Y-phase).
void gkr_sumcheck_round_y(
    const uint64_t *wiring, int numEntries,
    const uint64_t vxScalar[4],
    const uint64_t *vy, int vySize,
    int halfSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

/// Reduce sparse wiring after a sumcheck round. Returns number of output entries.
int gkr_wiring_reduce(
    const uint64_t *wiring, int numEntries,
    const uint64_t challenge[4],
    int halfSize,
    uint64_t *outWiring);

/// Single-round step for GKR sumcheck (dispatches to X or Y phase).
void gkr_sumcheck_step(
    const uint64_t *wiring, int numEntries,
    const uint64_t *curVx, int vxSize,
    const uint64_t *curVy, int vySize,
    int round, int nIn, int currentTableSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

/// Dual projective MSM: compute two MSMs with shared thread pool.
/// Avoids double thread-creation overhead.
void bn254_dual_msm_projective(
    const uint64_t *points1, const uint32_t *scalars1, int n1,
    const uint64_t *points2, const uint32_t *scalars2, int n2,
    uint64_t result1[12], uint64_t result2[12]);

/// Fr modular inverse via Fermat's little theorem: r = a^(p-2) mod p.
/// Uses C CIOS Montgomery mul (~100x faster than Swift frInverse).
void bn254_fr_inverse(const uint64_t a[4], uint64_t r[4]);

/// Fr CIOS Montgomery multiplication: r = a * b * R^{-1} mod p.
void bn254_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/// Fr modular addition: r = (a + b) mod p.
void bn254_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/// Fr modular subtraction: r = (a - b) mod p.
void bn254_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/// Fr modular negation: r = (-a) mod p.
void bn254_fr_neg(const uint64_t a[4], uint64_t r[4]);

/// Fr power: r = a^exp mod p (exp is 64-bit).
void bn254_fr_pow(const uint64_t a[4], uint64_t exp, uint64_t r[4]);

/// Fr equality check: returns 1 if a == b, 0 otherwise.
int bn254_fr_eq(const uint64_t a[4], const uint64_t b[4]);

/// Batch KZG verification for Marlin.
/// Verifies n tuples via random linear combination using C field ops.
/// @param srsG1        First SRS point (affine, 8 uint64_t: x[4], y[4]).
/// @param srsSecret    SRS secret scalar (4 uint64_t, Montgomery form).
/// @param commitments  n projective points (12 uint64_t each).
/// @param points       n Fr evaluation points (4 uint64_t each).
/// @param evaluations  n Fr claimed evaluations (4 uint64_t each).
/// @param witnesses    n projective points (12 uint64_t each).
/// @param batchChallenge Fr element for random linear combination.
/// @param n            Number of tuples.
/// @return 1 if verification passes, 0 otherwise.
int bn254_batch_kzg_verify(const uint64_t *srsG1, const uint64_t *srsSecret,
                           const uint64_t *commitments, const uint64_t *points,
                           const uint64_t *evaluations, const uint64_t *witnesses,
                           const uint64_t *batchChallenge, int n);

/// Fused sparse matvec + MLE eval: MLE(M*z)(point) in one C call.
/// Avoids Swift intermediate array allocation for M*z.
/// CSR: rowPtr[rows+1], colIdx[nnz], values[nnz*4 uint64 Fr].
/// z: n Fr elements. point: numVars Fr elements.
void bn254_sparse_matvec_mle(
    const int *rowPtr, const int *colIdx, const uint64_t *values,
    int rows, const uint64_t *z,
    const uint64_t *point, int numVars, int padM,
    uint64_t result[4]);

// ============================================================
// secp256k1 Fr (scalar field) CIOS Montgomery operations
// ============================================================

/// secp256k1 Fr Montgomery multiplication: r = a * b * R^{-1} mod n.
void secp256k1_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
/// secp256k1 Fr Montgomery addition: r = (a + b) mod n.
void secp256k1_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
/// secp256k1 Fr Montgomery subtraction: r = (a - b) mod n.
void secp256k1_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
/// secp256k1 Fr negation: r = -a mod n.
void secp256k1_fr_neg(const uint64_t a[4], uint64_t r[4]);

/// secp256k1 Fr inverse via Fermat: r = a^{n-2} mod n.
void secp256k1_fr_inverse(const uint64_t a[4], uint64_t r[4]);

/// secp256k1 Fr batch inverse via Montgomery's trick.
void secp256k1_fr_batch_inverse(const uint64_t *a, int n, uint64_t *out);

// ============================================================
// secp256k1 Fp (base field) CIOS Montgomery operations
// ============================================================

/// secp256k1 Fp Montgomery multiplication: r = a * b * R^{-1} mod p.
void secp256k1_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
/// secp256k1 Fp Montgomery squaring: r = a^2 * R^{-1} mod p.
void secp256k1_fp_sqr(const uint64_t a[4], uint64_t r[4]);
/// secp256k1 Fp Montgomery addition: r = (a + b) mod p.
void secp256k1_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
/// secp256k1 Fp Montgomery subtraction: r = (a - b) mod p.
void secp256k1_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
/// secp256k1 Fp negation: r = -a mod p.
void secp256k1_fp_neg(const uint64_t a[4], uint64_t r[4]);
/// secp256k1 Fp inverse via Fermat: r = a^{p-2} mod p.
void secp256k1_fp_inv(const uint64_t a[4], uint64_t r[4]);

/// Prepare MSM inputs for probabilistic ECDSA batch verification.
/// All CPU scalar work (batch inverse, random weights, liftX) done in C.
/// @param sigs      n * 12 uint64_t: per sig [r[4], s[4], z[4]] in Fr Montgomery form.
/// @param pubkeys   n * 8 uint64_t: per key [x[4], y[4]] in Fp Montgomery form.
/// @param recov     n bytes: y-parity for lifting r to curve point (NULL for all 0).
/// @param n         Number of signatures.
/// @param out_points  Output (2n+1) * 8 uint64_t affine points (Fp Montgomery).
/// @param out_scalars Output (2n+1) * 8 uint32_t scalars (integer form).
/// @return 0 on success, -1 if any r_i is not a valid x-coordinate.
int secp256k1_ecdsa_batch_prepare(
    const uint64_t *sigs, const uint64_t *pubkeys, const uint8_t *recov,
    int n, uint64_t *out_points, uint32_t *out_scalars);

// ============================================================
// Spartan-specific accelerated operations
// ============================================================

/// Sparse matrix-vector multiply for Spartan R1CS.
void spartan_sparse_matvec(const uint64_t *entries, int numEntries,
                           const uint64_t *z, int zLen,
                           uint64_t *result, int numRows);

/// Spartan SC1 round: degree-3 sumcheck over eq*[az*bz - cz].
void spartan_sc1_round(uint64_t *eqC, uint64_t *azC, uint64_t *bzC, uint64_t *czC,
                       int halfSize,
                       uint64_t s0[4], uint64_t s1[4], uint64_t s2[4], uint64_t s3[4]);

/// Fold an Fr array in-place: arr[j] = arr[j] + ri*(arr[j+half] - arr[j]).
void spartan_fold_array(uint64_t *arr, int halfSize, const uint64_t ri[4]);

/// Spartan SC2 round: degree-2 sumcheck over w*z.
void spartan_sc2_round(uint64_t *wC, uint64_t *zC, int halfSize,
                       uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

/// Build combined weight vector for SC2.
void spartan_build_weight_vec(const uint64_t *entries, int numEntries,
                              const uint64_t *eqRx, int eqRxLen,
                              const uint64_t weight[4],
                              uint64_t *wVec, int paddedN);

/// Compute eq polynomial for Spartan.
void spartan_eq_poly(const uint64_t *point, int n, uint64_t *eq);

/// MLE evaluation via successive halving for Spartan.
void spartan_mle_eval(const uint64_t *evals, int numVars,
                      const uint64_t *point, uint64_t result[4]);

/// Basefold single-round fold on CPU with CIOS Montgomery arithmetic.
/// result[i] = evals[i] + alpha * (evals[i + halfN] - evals[i])
/// Multi-threaded via GCD for halfN >= 4096.
/// @param evals  Input array: 2*halfN Fr elements (4 uint64_t each, Montgomery form).
/// @param result Output array: halfN Fr elements (4 uint64_t each).
/// @param alpha  Folding challenge (4 uint64_t, Montgomery form).
/// @param halfN  Half the input size.
void bn254_fr_basefold_fold(const uint64_t *evals, uint64_t *result,
                             const uint64_t *alpha, uint32_t halfN);

/// Basefold fold all rounds on CPU with CIOS Montgomery arithmetic.
/// Folds evals (2^numVars elements) through numVars rounds using point[] as challenges.
/// Stores each intermediate layer contiguously in out_layers:
///   layer0 (n/2 elements), layer1 (n/4), ..., layerK (1 element).
/// Total output = n-1 elements (each 4 uint64_t, Montgomery form).
/// Multi-threaded via GCD for large rounds.
void bn254_fr_basefold_fold_all(const uint64_t *evals, int numVars,
                                 const uint64_t *point, uint64_t *out_layers);

// ============================================================
// Tensor Proof Compression operations
// ============================================================

/// Matrix-vector multiply: result[i] = sum_j M[i*cols+j] * vec[j].
void tensor_mat_vec_mul(const uint64_t *M, const uint64_t *vec,
                        int rows, int cols, uint64_t *result);

/// Full inner-product sumcheck: prove sum_i a[i]*b[i].
/// rounds: numVars * 3 Fr (s0,s1,s2 per round). finalEval: a_final*b_final.
void tensor_inner_product_sumcheck(
    const uint64_t *evalsA, const uint64_t *evalsB,
    int numVars, const uint64_t *challenges,
    uint64_t *rounds, uint64_t *finalEval);

/// Fused eq polynomial + weighted matrix row evaluation.
/// result[j] = sum_i eq(rowPoint)[i] * M[i*cols + j].
void tensor_eq_weighted_row(const uint64_t *M, const uint64_t *rowPoint,
                            int rows, int cols, uint64_t *result);

// ============================================================
// WHIR polynomial folding
// ============================================================

/// WHIR polynomial fold: result[j] = sum_{k=0}^{r-1} beta^k * evals[j*r + k]
/// Uses Horner's method with CIOS Montgomery mul. Multi-threaded for n >= 2048.
/// @param evals  n Fr elements (4 uint64_t each, Montgomery form).
/// @param n      Number of input elements (must be divisible by reductionFactor).
/// @param beta   Folding challenge (4 uint64_t, Montgomery form).
/// @param reductionFactor  Fold factor (2, 4, 8, ...). Specialized fast path for 4.
/// @param result Output n/reductionFactor Fr elements (4 uint64_t each).
void bn254_fr_whir_fold(const uint64_t *evals, int n,
                         const uint64_t beta[4],
                         int reductionFactor,
                         uint64_t *result);

/// CPU Poseidon2 Merkle tree builder (avoids GPU command buffer overhead).
/// Layout: tree[0..n-1] = leaves, tree[n..2n-2] = internal, tree[2n-2] = root.
/// Each element is 4 x uint64_t (32 bytes, BN254 Fr Montgomery form).
/// Multi-threaded for levels with >= 256 pairs.
/// @param leaves  n Fr elements (4 uint64_t each).
/// @param n       Number of leaves (must be power of 2).
/// @param tree    Output buffer for 2n-1 Fr elements (4 uint64_t each).
void poseidon2_merkle_tree_cpu(const uint64_t *leaves, int n, uint64_t *tree);

// ============================================================
// CCS (Customizable Constraint System) accelerated operations
// ============================================================

/// CSR sparse matrix-vector multiply: result = M * z.
/// rowPtr: nRows+1 ints. colIdx/values: nnz entries. z: n Fr elements.
/// result: nRows Fr elements (4 uint64_t each, Montgomery form).
void ccs_sparse_matvec(uint64_t *result,
                       const int *rowPtr, const int *colIdx,
                       const uint64_t *values, const uint64_t *z,
                       int nRows);

/// Fused hadamard product + coefficient-weighted accumulation for CCS.
/// acc[i] += sum_j coeff_j * product_k matResultPtrs[j*maxDegree+k][i].
/// matResultPtrs: flat array of pointers to m-element Fr vectors.
/// nMatricesPerTerm[j]: degree of term j. coefficients: q Fr elements.
void ccs_hadamard_accumulate(uint64_t *acc,
                             const uint64_t * const *matResultPtrs,
                             const int *nMatricesPerTerm,
                             const uint64_t *coefficients,
                             int nTerms, int maxDegree, int m);

/// Compute single CCS term: result[i] = coeff * product_k matVecResults[k][i].
void ccs_compute_term(uint64_t *result,
                      const uint64_t * const *matVecResults,
                      int nMatrices,
                      const uint64_t coeff[4],
                      int m);

// ============================================================
// Plonk permutation Z accumulator
// ============================================================

/// Fused Plonk permutation Z accumulator computation.
/// Computes zEvals[0..n-1] with batch inverse and running product.
/// All arrays are BN254 Fr in Montgomery form (4 x uint64_t per element).
void plonk_compute_z_accumulator(
    const uint64_t *aEvals, const uint64_t *bEvals, const uint64_t *cEvals,
    const uint64_t *sigma1, const uint64_t *sigma2, const uint64_t *sigma3,
    const uint64_t *domain,
    const uint64_t beta[4], const uint64_t gamma[4],
    const uint64_t k1[4], const uint64_t k2[4],
    int n, uint64_t *zEvals);

// ============================================================
// Grumpkin curve operations (y^2 = x^3 - 17 over BN254 Fr)
// ============================================================

/// Grumpkin point addition (Jacobian projective, BN254 Fr CIOS).
/// @param p  First projective point (12 uint64_t: x[4], y[4], z[4], Montgomery form).
/// @param q  Second projective point.
/// @param r  Output projective point.
void grumpkin_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]);

/// Grumpkin point doubling (a=0 curve, Jacobian projective).
/// @param p  Input projective point (12 uint64_t).
/// @param r  Output projective point.
void grumpkin_point_double(const uint64_t p[12], uint64_t r[12]);

/// Grumpkin scalar multiplication using windowed method (w=4).
/// @param p       Projective point (12 uint64_t: x[4], y[4], z[4], BN254 Fr Montgomery form).
/// @param scalar  4 x uint64_t scalar (non-Montgomery integer form, little-endian).
/// @param r       Output projective point (12 uint64_t).
void grumpkin_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]);

/// Grumpkin mixed addition: projective P + affine Q (Z_Q = 1).
void grumpkin_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]);

/// Grumpkin Pippenger MSM (multi-threaded, mixed affine, batch-to-affine).
void grumpkin_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                             int n, uint64_t *result);

// ============================================================
// BN254 optimized squaring (upper-triangle, 37.5% fewer muls)
// ============================================================

void bn254_fr_sqr(const uint64_t a[4], uint64_t r[4]);
void bn254_fp_sqr(const uint64_t a[4], uint64_t r[4]);

// ============================================================
// BN254 Fp inverse and square root
// ============================================================

/// Fp inverse via Fermat: a^(p-2) mod p. CIOS Montgomery.
void bn254_fp_inv(const uint64_t a[4], uint64_t r[4]);

/// Fp sqrt via a^((p+1)/4). Returns 1 if sqrt exists, 0 otherwise.
/// BN254 Fp has p ≡ 3 mod 4.
int  bn254_fp_sqrt(const uint64_t a[4], uint64_t r[4]);

// ============================================================
// BN254 Fr fused pointwise ops (Groth16 computeH)
// ============================================================

/// Fused pointwise mul-sub: result[i] = a[i]*b[i] - c[i]. Multi-threaded for n >= 4096.
void bn254_fr_pointwise_mul_sub(const uint64_t *a, const uint64_t *b,
                                 const uint64_t *c, uint64_t *result, int n);

/// Coefficient division by vanishing polynomial Z_H(x) = x^n - 1.
void bn254_fr_coeff_div_vanishing(const uint64_t *pCoeffs, int domainN,
                                   uint64_t *hCoeffs);

// ============================================================
// BN254 Fr linear combine (HyperNova witness fold)
// ============================================================

/// Linear combine: result[i] = running[i] + rho * new_vals[i].
/// Multi-threaded for count >= 4096.
void bn254_fr_linear_combine(const uint64_t *running, const uint64_t *new_vals,
                              const uint64_t rho[4], uint64_t *result, int count);

// ============================================================
// CPU Poseidon2 (BN254 Fr CIOS)
// ============================================================

/// CPU Poseidon2 full permutation on 3 Fr elements.
void poseidon2_permutation_cpu(const uint64_t state[12], uint64_t result[12]);

/// CPU Poseidon2 hash of two Fr elements (2-to-1 compression).
void poseidon2_hash_cpu(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]);

/// Batch CPU Poseidon2 hash of pairs (multi-threaded for count >= 256).
void poseidon2_hash_batch_cpu(const uint64_t *input, int count, uint64_t *output);

// ============================================================
// secp256k1 Shamir's trick
// ============================================================

/// Shamir's trick: compute s1*P1 + s2*P2 using simultaneous double-and-add.
void secp256k1_shamir_double_mul(const uint64_t *p1, const uint64_t *s1,
                                  const uint64_t *p2, const uint64_t *s2,
                                  uint64_t *r);

// ============================================================
// BabyJubjub twisted Edwards curve (over BN254 Fr)
// ============================================================

void babyjubjub_point_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]);
void babyjubjub_point_double(const uint64_t p[16], uint64_t r[16]);
void babyjubjub_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]);

// ============================================================
// Ed25519 Fp Solinas reduction (p = 2^255 - 19)
// ============================================================

void ed25519_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void ed25519_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fp_neg(const uint64_t a[4], uint64_t r[4]);
void ed25519_fp_inverse(const uint64_t a[4], uint64_t r[4]);
void ed25519_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]);
void ed25519_point_add_c(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]);
void ed25519_point_double_c(const uint64_t p[16], uint64_t r[16]);
void ed25519_point_to_affine(const uint64_t p[16], uint64_t aff[8]);
void ed25519_mont_to_direct(const uint64_t mont[4], uint64_t direct[4]);
void ed25519_direct_to_mont(const uint64_t direct[4], uint64_t mont[4]);
void ed25519_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                            int n, uint64_t *result);

// Ed25519 scalar field Fq (order q = 2^252 + 27742317777372353535851937790883648493)
void ed25519_fq_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fq_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fq_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fq_from_raw(const uint64_t raw[4], uint64_t mont[4]);
void ed25519_fq_to_raw(const uint64_t mont[4], uint64_t raw[4]);
void ed25519_fq_from_bytes64(const uint8_t bytes[64], uint64_t mont[4]);
void ed25519_fq_to_bytes(const uint64_t mont[4], uint8_t bytes[32]);

// Ed25519 Shamir's trick: simultaneous s*G + h*A
void ed25519_shamir_double_mul(const uint64_t G[16], const uint64_t s[4],
                                const uint64_t A[16], const uint64_t h[4],
                                uint64_t result[16]);

// Ed25519 EdDSA sign/verify helpers
void ed25519_eddsa_sign_compute_r(const uint64_t gen[16], const uint64_t r_scalar[4],
                                   uint64_t r_point[16]);
void ed25519_eddsa_sign_compute_s(const uint64_t r_mont[4], const uint64_t k_mont[4],
                                   const uint64_t a_mont[4], uint64_t s_mont[4]);
int ed25519_eddsa_verify(const uint64_t gen[16], const uint64_t s_raw[4],
                          const uint64_t r_point[16], const uint64_t h_raw[4],
                          const uint64_t pub_key[16]);

// ============================================================
// Pallas curve (y^2 = x^3 + 5 over Fp)
// ============================================================

void pallas_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void pallas_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void pallas_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void pallas_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void pallas_fp_neg(const uint64_t a[4], uint64_t r[4]);
void pallas_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]);
void pallas_point_double(const uint64_t p[12], uint64_t r[12]);
void pallas_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]);
void pallas_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]);
void pallas_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                           int n, uint64_t *result);

// ============================================================
// Vesta curve (y^2 = x^3 + 5 over Fp)
// ============================================================

void vesta_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void vesta_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void vesta_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void vesta_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void vesta_fp_neg(const uint64_t a[4], uint64_t r[4]);
void vesta_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]);
void vesta_point_double(const uint64_t p[12], uint64_t r[12]);
void vesta_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]);
void vesta_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]);
void vesta_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                          int n, uint64_t *result);

// ============================================================
// BLS12-381 Fr (scalar field, 4-limb)
// ============================================================

void bls12_381_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void bls12_381_fr_sqr(const uint64_t a[4], uint64_t r[4]);
void bls12_381_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void bls12_381_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void bls12_381_fr_neg(const uint64_t a[4], uint64_t r[4]);

// ============================================================
// BLS12-381 Fp (base field, 6-limb)
// ============================================================

void bls12_381_fp_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);
void bls12_381_fp_sqr(const uint64_t a[6], uint64_t r[6]);
void bls12_381_fp_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);
void bls12_381_fp_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);
void bls12_381_fp_neg(const uint64_t a[6], uint64_t r[6]);
void bls12_381_fp_inv_ext(const uint64_t a[6], uint64_t r[6]);
int  bls12_381_fp_sqrt(const uint64_t a[6], uint64_t r[6]);

// ============================================================
// BLS12-381 G1 curve ops (Jacobian projective, 6-limb coords)
// ============================================================

void bls12_381_g1_point_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]);
void bls12_381_g1_point_double(const uint64_t p[18], uint64_t r[18]);
void bls12_381_g1_point_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]);
void bls12_381_g1_scalar_mul(const uint64_t p[18], const uint64_t scalar[4], uint64_t r[18]);
void bls12_381_g1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                                 int n, uint64_t *result);

// ============================================================
// BLS12-381 Fp2 tower (Fp[u]/(u²+1), 12 uint64_t per element)
// ============================================================

void bls12_381_fp2_add(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]);
void bls12_381_fp2_sub(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]);
void bls12_381_fp2_neg(const uint64_t a[12], uint64_t r[12]);
void bls12_381_fp2_mul(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]);
void bls12_381_fp2_sqr(const uint64_t a[12], uint64_t r[12]);
void bls12_381_fp2_conj(const uint64_t a[12], uint64_t r[12]);
void bls12_381_fp2_mul_by_nonresidue(const uint64_t a[12], uint64_t r[12]);

// ============================================================
// BLS12-381 G2 curve ops (Fp2 coords, 36 uint64_t projective)
// ============================================================

void bls12_381_g2_point_add(const uint64_t p[36], const uint64_t q[36], uint64_t r[36]);
void bls12_381_g2_point_double(const uint64_t p[36], uint64_t r[36]);
void bls12_381_g2_point_add_mixed(const uint64_t p[36], const uint64_t q_aff[24], uint64_t r[36]);
void bls12_381_g2_scalar_mul(const uint64_t p[36], const uint64_t scalar[4], uint64_t r[36]);
void bls12_381_g2_scalar_mul_wide(const uint64_t p[36], const uint64_t *scalar, int n_limbs, uint64_t r[36]);

// ============================================================
// BLS12-381 Fp2 inverse + higher tower + pairing
// ============================================================

void bls12_381_fp2_inv(const uint64_t a[12], uint64_t r[12]);
void bls12_381_fp6_mul(const uint64_t a[36], const uint64_t b[36], uint64_t r[36]);
void bls12_381_fp6_sqr(const uint64_t a[36], uint64_t r[36]);
void bls12_381_fp12_mul(const uint64_t a[72], const uint64_t b[72], uint64_t r[72]);
void bls12_381_fp12_sqr(const uint64_t a[72], uint64_t r[72]);
void bls12_381_fp12_inv(const uint64_t a[72], uint64_t r[72]);
void bls12_381_fp12_conj(const uint64_t a[72], uint64_t r[72]);
void bls12_381_miller_loop(const uint64_t p_aff[12], const uint64_t q_aff[24], uint64_t result[72]);
void bls12_381_final_exp(const uint64_t f[72], uint64_t result[72]);
void bls12_381_pairing(const uint64_t p_aff[12], const uint64_t q_aff[24], uint64_t result[72]);
int bls12_381_pairing_check(const uint64_t *pairs, int n);

// ============================================================
// BN254 pairing (Fp2/Fp6/Fp12 tower + Miller loop + final exp)
// Fp = 4x64-bit, Fp2 = 8x64, Fp6 = 24x64, Fp12 = 48x64
// G1Affine = 8x64 (x[4], y[4]), G2Affine = 16x64 (x[8], y[8])
// ============================================================

void bn254_miller_loop(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]);
void bn254_final_exp(const uint64_t f[48], uint64_t result[48]);
void bn254_pairing(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]);
void bn254_pairing_precomp(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]);
int bn254_pairing_check(const uint64_t *pairs, int n);
int bn254_pairing_check_precomp(const uint64_t *pairs, int n);

// ============================================================
// BLS12-381 hash-to-curve G2 (RFC 9380, SSWU + 3-isogeny)
// ============================================================

void bls12_381_hash_to_g2(const uint8_t *msg, size_t msg_len,
                           const uint8_t *dst, size_t dst_len,
                           uint64_t result[36]);
void bls12_381_hash_to_g2_default(const uint8_t *msg, size_t msg_len, uint64_t result[36]);
void bls12_381_g2_clear_cofactor(const uint64_t p[36], uint64_t r[36]);

// ============================================================
// Jubjub twisted Edwards curve (over BLS12-381 Fr)
// ============================================================

void jubjub_point_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]);
void jubjub_point_double(const uint64_t p[16], uint64_t r[16]);
void jubjub_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]);

// ============================================================
// BLS12-377 Fq (base field, 377-bit, 6x64-bit limbs)
// ============================================================

void bls12_377_fq_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);
void bls12_377_fq_sqr(const uint64_t a[6], uint64_t r[6]);
void bls12_377_fq_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);
void bls12_377_fq_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);
void bls12_377_fq_neg(const uint64_t a[6], uint64_t r[6]);
void bls12_377_fq_inverse(const uint64_t a[6], uint64_t r[6]);

// ============================================================
// BLS12-377 Fr (scalar field, 253-bit, 4x64-bit limbs)
// ============================================================

void bls12_377_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void bls12_377_fr_sqr(const uint64_t a[4], uint64_t r[4]);
void bls12_377_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void bls12_377_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void bls12_377_fr_neg(const uint64_t a[4], uint64_t r[4]);

/// Forward NTT on BLS12-377 Fr field (Cooley-Tukey DIT).
/// @param data Array of n * 4 uint64_t values (n elements, 4 limbs each, little-endian Montgomery form).
/// @param logN Log2 of the transform size (1..47).
void bls12_377_fr_ntt(uint64_t *data, int logN);

/// Inverse NTT on BLS12-377 Fr field (Gentleman-Sande DIF + bit-reversal + 1/n scaling).
void bls12_377_fr_intt(uint64_t *data, int logN);

// ============================================================
// BLS12-377 G1 point ops (Jacobian projective, y²=x³+1)
// ============================================================

void bls12_377_g1_point_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]);
void bls12_377_g1_point_double(const uint64_t p[18], uint64_t r[18]);
void bls12_377_g1_point_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]);
void bls12_377_g1_scalar_mul(const uint64_t p[18], const uint64_t scalar[6], uint64_t r[18]);
void bls12_377_g1_to_affine(const uint64_t p[18], uint64_t aff[12]);
void bls12_377_g1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                                 int n, uint64_t *result);

// ============================================================
// BGMW fixed-base scalar multiplication
// ============================================================

/// Precompute BGMW lookup tables for fixed-base MSM.
/// For each generator g_i, window w, digit d (1..2^window_bits-1):
///   table[i][w][d-1] = d * (2^(w*window_bits)) * g_i
/// @param generators_affine  n affine points (8 uint64_t each: x[4], y[4], Montgomery form).
/// @param n                  Number of generators.
/// @param window_bits        Window width in bits (e.g. 7 or 8).
/// @param table_out          Output: n * ceil(256/w) * (2^w-1) affine points (8 uint64_t each).
void bgmw_precompute(const uint64_t *generators_affine, int n,
                     int window_bits, uint64_t *table_out);

/// BGMW fixed-base MSM using precomputed tables.
/// Decomposes each scalar into base-2^w digits and sums table lookups (additions only).
/// @param table       Precomputed table from bgmw_precompute.
/// @param n           Number of scalars (must match n used in precompute).
/// @param window_bits Window width (must match precompute).
/// @param scalars     n scalars as n*8 uint32_t (little-endian integer limbs, NOT Montgomery).
/// @param result      Output projective point (12 uint64_t: x[4], y[4], z[4]).
void bgmw_msm(const uint64_t *table, int n, int window_bits,
              const uint32_t *scalars, uint64_t *result);

// ============================================================
// Stark252 field (p = 2^251 + 17*2^192 + 1)
// ============================================================

void stark252_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void stark252_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void stark252_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void stark252_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void stark252_fp_neg(const uint64_t a[4], uint64_t r[4]);

// Stark252 NTT (Cooley-Tukey DIT forward, Gentleman-Sande DIF inverse)
// data: array of n elements (n = 2^logN), each 4×64-bit limbs in Montgomery form
void stark252_ntt(uint64_t *data, int logN);
void stark252_intt(uint64_t *data, int logN);

// ============================================================
// Binary tower field arithmetic (GF(2) towers for Binius STARKs)
// Uses ARM64 PMULL carry-less multiply intrinsics
// ============================================================

/// Initialize GF(2^8) log/exp/inverse lookup tables. Call once before using bt_gf8_*.
void bt_gf8_init(void);

/// GF(2^8) multiply via log/exp table (AES polynomial x^8+x^4+x^3+x+1).
uint8_t bt_gf8_mul(uint8_t a, uint8_t b);
/// GF(2^8) square.
uint8_t bt_gf8_sqr(uint8_t a);
/// GF(2^8) inverse (0 maps to 0).
uint8_t bt_gf8_inv(uint8_t a);

/// GF(2^16) multiply using PMULL (irreducible x^16+x^5+x^3+x+1).
uint16_t bt_gf16_mul(uint16_t a, uint16_t b);
/// GF(2^16) square.
uint16_t bt_gf16_sqr(uint16_t a);
/// GF(2^16) inverse via Itoh-Tsujii (0 maps to 0).
uint16_t bt_gf16_inv(uint16_t a);

/// GF(2^32) multiply using PMULL (irreducible x^32+x^7+x^3+x^2+1).
uint32_t bt_gf32_mul(uint32_t a, uint32_t b);
/// GF(2^32) square.
uint32_t bt_gf32_sqr(uint32_t a);
/// GF(2^32) inverse via Itoh-Tsujii (0 maps to 0).
uint32_t bt_gf32_inv(uint32_t a);

/// GF(2^64) multiply using PMULL (irreducible x^64+x^4+x^3+x+1).
uint64_t bt_gf64_mul(uint64_t a, uint64_t b);
/// GF(2^64) square.
uint64_t bt_gf64_sqr(uint64_t a);
/// GF(2^64) inverse via Itoh-Tsujii (0 maps to 0).
uint64_t bt_gf64_inv(uint64_t a);

/// GF(2^128) multiply (flat polynomial, AES-GCM irreducible x^128+x^7+x^2+x+1).
/// Elements are 2 x uint64_t: [lo, hi].
void bt_gf128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
/// GF(2^128) square.
void bt_gf128_sqr(const uint64_t a[2], uint64_t r[2]);
/// GF(2^128) add (XOR).
void bt_gf128_add(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
/// GF(2^128) inverse via Itoh-Tsujii.
void bt_gf128_inv(const uint64_t a[2], uint64_t r[2]);

/// Tower-form GF(2^128) = GF(2^64)[X]/(X^2+X+2): multiply.
/// Elements are [lo, hi] representing lo + hi*X.
void bt_tower128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
/// Tower-form GF(2^128) square.
void bt_tower128_sqr(const uint64_t a[2], uint64_t r[2]);
/// Tower-form GF(2^128) add.
void bt_tower128_add(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
/// Tower-form GF(2^128) inverse.
void bt_tower128_inv(const uint64_t a[2], uint64_t r[2]);

/// Batch GF(2^64) multiply: out[i] = a[i] * b[i].
void bt_gf64_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);
/// Batch GF(2^64) add (XOR): out[i] = a[i] ^ b[i], NEON-vectorized.
void bt_gf64_batch_add(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);
/// Batch GF(2^64) square: out[i] = a[i]^2.
void bt_gf64_batch_sqr(const uint64_t *a, uint64_t *out, int n);
/// Batch GF(2^128) multiply: out[i] = a[i] * b[i]. Each element is 2 x uint64_t.
void bt_gf128_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);
/// Batch GF(2^128) add: out[i] = a[i] ^ b[i]. Each element is 2 x uint64_t.
void bt_gf128_batch_add(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);
/// Batch tower-form GF(2^128) multiply.
void bt_tower128_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

// ============================================================
// Additive FFT for binary fields (Lin-Chung-Han / Cantor algorithm)
// ============================================================
// For binary fields, the standard multiplicative NTT doesn't work (no 2-power
// multiplicative subgroup). Additive FFT evaluates polynomials at all points
// of an affine subspace using subspace polynomials s(x) = x^2 + x.
// Complexity: O(n log^2 n) field muls, O(n log n) XOR additions.
// Required for Binius-style binary field polynomial commitments and STARKs.

/// Generate the standard Frobenius basis for GF(2^64) additive FFT.
/// basis: output array of k uint64_t elements.
/// k: number of basis vectors (transform size is 2^k).
void bt_afft_basis_64(uint64_t *basis, int k);

/// Generate the standard Frobenius basis for GF(2^128) additive FFT.
/// basis: output array of k elements, each 2 x uint64_t (interleaved lo/hi).
/// k: number of basis vectors (transform size is 2^k).
void bt_afft_basis_128(uint64_t *basis, int k);

/// Forward additive FFT over GF(2^64) — recursive.
/// Evaluates polynomial (in novel basis) at all 2^k subspace points.
/// coeffs: in-place, n = 2^k uint64_t elements. basis: k elements.
void bt_afft_forward_64(uint64_t *coeffs, size_t n, const uint64_t *basis);

/// Inverse additive FFT over GF(2^64) — recursive.
/// Interpolates: evaluations -> novel basis coefficients.
void bt_afft_inverse_64(uint64_t *evals, size_t n, const uint64_t *basis);

/// Forward additive FFT over GF(2^128) — recursive.
/// coeffs: n elements, each 2 x uint64_t (2*n uint64_t total). basis: k elements (2*k uint64_t).
void bt_afft_forward_128(uint64_t *coeffs, size_t n, const uint64_t *basis);

/// Inverse additive FFT over GF(2^128) — recursive.
void bt_afft_inverse_128(uint64_t *evals, size_t n, const uint64_t *basis);

/// Forward additive FFT over GF(2^64) — iterative (cache-friendly, no recursion overhead).
void bt_afft_forward_64_iter(uint64_t *data, size_t n, const uint64_t *basis);

/// Inverse additive FFT over GF(2^64) — iterative.
void bt_afft_inverse_64_iter(uint64_t *data, size_t n, const uint64_t *basis);

/// Forward additive FFT over GF(2^128) — iterative.
void bt_afft_forward_128_iter(uint64_t *data, size_t n, const uint64_t *basis);

/// Inverse additive FFT over GF(2^128) — iterative.
void bt_afft_inverse_128_iter(uint64_t *data, size_t n, const uint64_t *basis);

/// Forward additive FFT over GF(2^64) — NEON-accelerated (vectorized XOR propagation).
void bt_afft_forward_64_neon(uint64_t *data, size_t n, const uint64_t *basis);

/// Inverse additive FFT over GF(2^64) — NEON-accelerated.
void bt_afft_inverse_64_neon(uint64_t *data, size_t n, const uint64_t *basis);

// ============================================================
// M31 (Mersenne-31, p = 2^31 - 1) NEON batch operations
// All elements are uint32_t in [0, p). 4-wide NEON vectorized.
// ============================================================

/// Batch add: out[i] = (a[i] + b[i]) mod p.
void m31_batch_add_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n);

/// Batch sub: out[i] = (a[i] - b[i]) mod p.
void m31_batch_sub_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n);

/// Batch mul: out[i] = a[i] * b[i] mod p, using vmull_u32 + Mersenne reduction.
void m31_batch_mul_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n);

/// Batch negate: out[i] = -a[i] mod p.
void m31_batch_neg_neon(const uint32_t *a, uint32_t *out, int n);

/// Batch scalar multiply: out[i] = a[i] * scalar mod p.
void m31_batch_mul_scalar_neon(const uint32_t *a, uint32_t scalar, uint32_t *out, int n);

/// Inner product: result = sum(a[i] * b[i]) mod p.
uint32_t m31_inner_product_neon(const uint32_t *a, const uint32_t *b, int n);

/// Batch dot products: results[v] = inner_product(a + v*vecLen, b + v*vecLen).
void m31_batch_dot_product_neon(const uint32_t *a, const uint32_t *b,
                                 int vecLen, int nVecs, uint32_t *results);

/// M31 extension field Fp3 = M31[x]/(x^3 - 5): add.
void m31_ext3_add_neon(const uint32_t a[3], const uint32_t b[3], uint32_t r[3]);

/// M31 extension field Fp3: sub.
void m31_ext3_sub_neon(const uint32_t a[3], const uint32_t b[3], uint32_t r[3]);

/// M31 extension field Fp3: mul.
void m31_ext3_mul_neon(const uint32_t a[3], const uint32_t b[3], uint32_t r[3]);

/// M31 DIT butterfly: (a, b) -> (a + b*w, a - b*w). NEON-accelerated.
void m31_butterfly_neon(uint32_t *data, int halfBlock, int nBlocks,
                        const uint32_t *twiddles);

/// M31 DIF butterfly: (a, b) -> (a + b, (a - b)*w). NEON-accelerated.
void m31_butterfly_dif_neon(uint32_t *data, int halfBlock, int nBlocks,
                            const uint32_t *twiddles);

// ============================================================
// BabyBear NEON batch operations (Montgomery form)
// Extends babybear_ntt.c with general-purpose batch ops.
// ============================================================

/// Batch add: out[i] = (a[i] + b[i]) mod p (Montgomery form).
void bb_batch_add_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n);

/// Batch sub: out[i] = (a[i] - b[i]) mod p.
void bb_batch_sub_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n);

/// Batch mul: out[i] = a[i] * b[i] mod p (Plonky3 7-instruction Montgomery).
void bb_batch_mul_neon(const uint32_t *a, const uint32_t *b, uint32_t *out, int n);

/// Batch negate: out[i] = -a[i] mod p.
void bb_batch_neg_neon(const uint32_t *a, uint32_t *out, int n);

/// Batch scalar multiply: out[i] = a[i] * scalar mod p.
void bb_batch_mul_scalar_neon(const uint32_t *a, uint32_t scalar, uint32_t *out, int n);

/// Inner product: result = sum(a[i] * b[i]) mod p (Montgomery form).
uint32_t bb_inner_product_neon(const uint32_t *a, const uint32_t *b, int n);

/// Linear combination: out[i] = alpha * a[i] + beta * b[i] (Montgomery form).
void bb_linear_combine_neon(const uint32_t *a, const uint32_t *b,
                             uint32_t alpha, uint32_t beta,
                             uint32_t *out, int n);

/// Batch convert to Montgomery form.
void bb_batch_to_monty_neon(const uint32_t *in, uint32_t *out, int n);

/// Batch convert from Montgomery form.
void bb_batch_from_monty_neon(const uint32_t *in, uint32_t *out, int n);

// ============================================================
// Extended Goldilocks batch operations
// ============================================================

/// Batch scale: out[i] = a[i] * scalar mod p.
void gl_batch_scale_neon(const uint64_t *a, uint64_t scalar, uint64_t *out, int n);

/// Inner product: result = sum(a[i] * b[i]) mod p.
uint64_t gl_inner_product_neon(const uint64_t *a, const uint64_t *b, int n);

/// Linear combination: out[i] = alpha*a[i] + beta*b[i] mod p.
void gl_linear_combine_neon(const uint64_t *a, const uint64_t *b,
                             uint64_t alpha, uint64_t beta,
                             uint64_t *out, int n);

/// Batch negate: out[i] = -a[i] mod p.
void gl_batch_neg_neon(const uint64_t *a, uint64_t *out, int n);

/// Goldilocks Fp2 = Fp[w]/(w^2 - 7): add. Elements are 2 x uint64_t.
void gl_ext2_add_neon(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);

/// Goldilocks Fp2: sub.
void gl_ext2_sub_neon(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);

/// Goldilocks Fp2: mul.
void gl_ext2_mul_neon(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);

/// Goldilocks Fp2: square.
void gl_ext2_sqr_neon(const uint64_t a[2], uint64_t r[2]);

/// Batch Goldilocks Fp2 mul: out[i] = a[i] * b[i] in Fp2.
void gl_ext2_batch_mul_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

// ============================================================
// Small-field Poseidon2 permutations (NEON-accelerated)
// ============================================================

/// BabyBear Poseidon2 permutation (width=16, x^7 S-box).
/// @param state            16 uint32_t in Montgomery form, modified in-place.
/// @param round_constants  (num_full_rounds * 16 + num_partial_rounds) constants.
/// @param internal_diag    16 diagonal constants for internal matrix.
/// @param num_full_rounds  Total full rounds (e.g. 8 = 4 before + 4 after partials).
/// @param num_partial_rounds  Number of partial rounds (e.g. 13).
void bb_poseidon2_permutation_neon(uint32_t state[16],
                                    const uint32_t *round_constants,
                                    const uint32_t internal_diag[16],
                                    int num_full_rounds,
                                    int num_partial_rounds);

/// M31 Poseidon2 permutation (width=16, x^5 S-box).
/// Same interface as BabyBear but elements in [0, 2^31 - 1).
void m31_poseidon2_permutation_neon(uint32_t state[16],
                                     const uint32_t *round_constants,
                                     const uint32_t internal_diag[16],
                                     int num_full_rounds,
                                     int num_partial_rounds);

/// Goldilocks Poseidon2 permutation (width=12, x^7 S-box).
/// @param state            12 uint64_t elements in [0, p).
/// @param round_constants  (num_full_rounds * 12 + num_partial_rounds) constants.
/// @param internal_diag    12 diagonal constants.
void gl_poseidon2_permutation_neon(uint64_t state[12],
                                    const uint64_t *round_constants,
                                    const uint64_t internal_diag[12],
                                    int num_full_rounds,
                                    int num_partial_rounds);

// ============================================================
// Packed NEON small-field arithmetic (4 lanes, uint32x4_t)
// ============================================================
#ifdef __ARM_NEON
#include <arm_neon.h>

/// Packed BabyBear modular add/sub/mul: 4 parallel ops mod p=2013265921.
/// Uses Plonky3-style Montgomery reduction (7 NEON instructions per 4 muls).
uint32x4_t packed_bb_add(uint32x4_t a, uint32x4_t b);
uint32x4_t packed_bb_sub(uint32x4_t a, uint32x4_t b);
uint32x4_t packed_bb_mul(uint32x4_t a, uint32x4_t b);

/// Barrett reduction of 4 products from 64-bit to 32-bit BabyBear.
uint32x4_t packed_bb_reduce(uint64x2_t lo, uint64x2_t hi);

/// Packed M31 modular add/sub/mul: 4 parallel ops mod p=2^31-1.
/// Mul uses vmull_u32 widening + Mersenne bit-trick reduction.
uint32x4_t packed_m31_add(uint32x4_t a, uint32x4_t b);
uint32x4_t packed_m31_sub(uint32x4_t a, uint32x4_t b);
uint32x4_t packed_m31_mul(uint32x4_t a, uint32x4_t b);

/// BabyBear NTT butterfly (vectorized): a' = a + tw*b, b' = a - tw*b.
void packed_bb_ntt_butterfly(uint32_t *a_arr, uint32_t *b_arr,
                              const uint32_t *tw, int n);

/// BabyBear batch dot product and vector add.
void packed_bb_dot_product(const uint32_t *a, const uint32_t *b, int n, uint32_t *result);
void packed_bb_vector_add(const uint32_t *a, const uint32_t *b, uint32_t *out, int n);

/// M31 batch dot product.
void packed_m31_dot_product(const uint32_t *a, const uint32_t *b, int n, uint32_t *result);

/// BabyBear DIT butterfly: (u, v) -> (u + tw*v, u - tw*v).
/// Data in Montgomery form. Twiddles has halfBlock entries.
void packed_bb_butterfly_dit(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles);

/// BabyBear DIF butterfly: (a, b) -> (a+b, (a-b)*tw).
void packed_bb_butterfly_dif(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles);

/// M31 DIT butterfly using packed NEON.
void packed_m31_butterfly_dit(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles);

/// M31 DIF butterfly using packed NEON.
void packed_m31_butterfly_dif(uint32_t *data, int halfBlock, int nBlocks,
                              const uint32_t *twiddles);

/// Packed BabyBear forward NTT (Cooley-Tukey DIT, in-place).
/// Input/output: array of 2^logN uint32_t in [0, p).
void packed_bb_ntt(uint32_t *data, int logN);

/// Packed BabyBear inverse NTT (Gentleman-Sande DIF, in-place).
void packed_bb_intt(uint32_t *data, int logN);
#endif // __ARM_NEON

// ============================================================
// Lattice NTT — Kyber (q=3329) and Dilithium (q=8380417)
// NEON-accelerated Barrett reduction + vectorized butterflies.
// ============================================================

/// Forward NTT for Kyber (q=3329, n=256) using ARM NEON.
/// Cooley-Tukey DIT, in-place. Values must be in [0, 3329).
/// @param poly Array of 256 uint32_t elements.
void lattice_ntt_kyber_neon(uint32_t *poly);

/// Inverse NTT for Kyber (q=3329, n=256) using ARM NEON.
/// Gentleman-Sande DIF + 1/128 scaling. Values must be in [0, 3329).
void lattice_intt_kyber_neon(uint32_t *poly);

/// Forward NTT for Dilithium (q=8380417, n=256) using ARM NEON.
/// Cooley-Tukey DIT, in-place. Values must be in [0, 8380417).
void lattice_ntt_dilithium_neon(uint32_t *poly);

/// Inverse NTT for Dilithium (q=8380417, n=256) using ARM NEON.
/// Gentleman-Sande DIF + 1/128 scaling.
void lattice_intt_dilithium_neon(uint32_t *poly);

/// Batch-4 forward NTT for Kyber: 4 independent NTTs in 4 NEON lanes.
/// @param polys Interleaved layout: polys[j*4 + lane] = polynomial[lane][j].
///              Must have 256*4 = 1024 elements.
void lattice_ntt_kyber_batch4(uint32_t *polys);

/// Batch-4 inverse NTT for Kyber.
void lattice_intt_kyber_batch4(uint32_t *polys);

/// Batch-4 forward NTT for Dilithium: 4 independent NTTs in 4 NEON lanes.
void lattice_ntt_dilithium_batch4(uint32_t *polys);

/// Batch-4 inverse NTT for Dilithium.
void lattice_intt_dilithium_batch4(uint32_t *polys);

/// Interleave 4 polynomials (256 elements each) into batch-4 layout.
/// @param p0..p3 Input polynomials (256 uint32 each).
/// @param out    Output buffer (1024 uint32, interleaved).
void lattice_interleave4(const uint32_t *p0, const uint32_t *p1,
                         const uint32_t *p2, const uint32_t *p3,
                         uint32_t *out);

/// Deinterleave batch-4 layout back into 4 separate polynomials.
void lattice_deinterleave4(const uint32_t *interleaved,
                            uint32_t *p0, uint32_t *p1,
                            uint32_t *p2, uint32_t *p3);

// ============================================================
// Signed-coefficient lattice NTT — standard PQC representations
// Kyber:     int16_t coefficients in [0, q), uses vmull_s16 for 8-way SIMD
// Dilithium: int32_t coefficients in [0, q), uses vmull_s32 for 4-way SIMD
// ============================================================

/// Forward NTT for Kyber (q=3329) with signed int16_t coefficients.
/// Uses vmull_s16 for NEON-accelerated Barrett reduction.
/// @param data Array of 2^logN int16_t elements in [0, 3329).
/// @param logN Log2 of transform size (max 8 for n=256).
void kyber_ntt_neon(int16_t *data, int logN);

/// Inverse NTT for Kyber with signed int16_t coefficients.
/// Gentleman-Sande DIF + 1/(n/2) scaling.
void kyber_intt_neon(int16_t *data, int logN);

/// Forward NTT for Dilithium (q=8380417) with signed int32_t coefficients.
/// Uses vmull_s32 for NEON-accelerated Barrett reduction.
/// @param data Array of 2^logN int32_t elements in [0, 8380417).
/// @param logN Log2 of transform size (max 8 for n=256).
void dilithium_ntt_neon(int32_t *data, int logN);

/// Inverse NTT for Dilithium with signed int32_t coefficients.
/// Gentleman-Sande DIF + 1/(n/2) scaling.
void dilithium_intt_neon(int32_t *data, int logN);

/// Pairwise multiply adjacent elements: result[i] = a[2i] * a[2i+1].
/// Used for binary multiplication tree in grand product GKR.
void bn254_fr_batch_mul_adjacent(uint64_t *result, const uint64_t *a, int half);

/// Grand product degree-3 sumcheck round.
/// Computes s0,s1,s2,s3 from eq/left/right arrays of size 2*half.
void bn254_fr_gp_sumcheck_round(
    const uint64_t *eq, const uint64_t *left, const uint64_t *right,
    int half, uint64_t s0[4], uint64_t s1[4], uint64_t s2[4], uint64_t s3[4]);

#endif // NEON_FIELD_OPS_H
