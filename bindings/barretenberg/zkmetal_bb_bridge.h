/*
 * zkmetal_bb_bridge.h -- C++ bridge between Barretenberg and zkMetal
 *
 * Wraps zkMetal's C API with interfaces compatible with BB's internal types.
 * All functions are gated behind HAS_ZKMETAL so BB compiles cleanly without it.
 *
 * Memory layout note:
 *   BB uses uint64_t[4] limbs in Montgomery form (little-endian).
 *   zkMetal uses the same layout (uint64_t[4] Montgomery, little-endian).
 *   On ARM64 (always little-endian), reinterpret_cast between the two is a no-op.
 */

#pragma once

#ifdef HAS_ZKMETAL

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <atomic>

extern "C" {
#include <zkmetal.h>
}

namespace zkmetal {

// ============================================================
// Initialization
// ============================================================

/// Lazy-initialize zkMetal (Metal device, pipeline cache, etc.)
/// Thread-safe. Returns true on success.
bool ensure_initialized();

/// Returns true if zkMetal GPU acceleration is available.
bool is_gpu_available();

// ============================================================
// BN254 Fr field arithmetic
// ============================================================
// BB's bb::fr is a 4x uint64_t Montgomery element -- identical
// layout to zkMetal's zk_bn254_fr. Direct passthrough.

inline void fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    bn254_fr_mul(a, b, r);
}

inline void fr_sqr(const uint64_t a[4], uint64_t r[4]) {
    bn254_fr_sqr(a, r);
}

inline void fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    bn254_fr_add(a, b, r);
}

inline void fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    bn254_fr_sub(a, b, r);
}

inline void fr_neg(const uint64_t a[4], uint64_t r[4]) {
    bn254_fr_neg(a, r);
}

inline void fr_inverse(const uint64_t a[4], uint64_t r[4]) {
    bn254_fr_inverse(a, r);
}

inline void fr_pow(const uint64_t a[4], uint64_t exp, uint64_t r[4]) {
    bn254_fr_pow(a, exp, r);
}

inline int fr_eq(const uint64_t a[4], const uint64_t b[4]) {
    return bn254_fr_eq(a, b);
}

// ============================================================
// BN254 Fq base field arithmetic
// ============================================================

inline void fq_sqr(const uint64_t a[4], uint64_t r[4]) {
    bn254_fp_sqr(a, r);
}

inline void fq_inv(const uint64_t a[4], uint64_t r[4]) {
    bn254_fp_inv(a, r);
}

inline int fq_sqrt(const uint64_t a[4], uint64_t r[4]) {
    return bn254_fp_sqrt(a, r);
}

// ============================================================
// Batch field operations (NEON-vectorized, multi-threaded)
// ============================================================

inline void fr_batch_inverse(const uint64_t* a, int n, uint64_t* out) {
    bn254_fr_batch_inverse(a, n, out);
}

inline void fr_batch_add(uint64_t* result, const uint64_t* a, const uint64_t* b, int n) {
    bn254_fr_batch_add_parallel(result, a, b, n);
}

inline void fr_batch_sub(uint64_t* result, const uint64_t* a, const uint64_t* b, int n) {
    bn254_fr_batch_sub_parallel(result, a, b, n);
}

inline void fr_batch_neg(uint64_t* result, const uint64_t* a, int n) {
    bn254_fr_batch_neg_parallel(result, a, n);
}

inline void fr_batch_mul_scalar(uint64_t* result, const uint64_t* a, const uint64_t* scalar, int n) {
    bn254_fr_batch_mul_scalar_parallel(result, a, scalar, n);
}

// ============================================================
// Polynomial operations
// ============================================================

inline void fr_horner_eval(const uint64_t* coeffs, int n, const uint64_t z[4], uint64_t result[4]) {
    bn254_fr_horner_eval(coeffs, n, z, result);
}

inline void fr_inner_product(const uint64_t* a, const uint64_t* b, int n, uint64_t* result) {
    bn254_fr_inner_product(a, b, n, result);
}

// ============================================================
// MSM -- Multi-Scalar Multiplication
// ============================================================

/// Minimum point count to route to GPU MSM.
/// Below this threshold, CPU scalar-mul is faster due to GPU dispatch overhead.
constexpr size_t MSM_GPU_THRESHOLD = 1 << 14;  // 16384

/// BN254 G1 MSM.
/// points: array of n affine points, each 8 uint64_t (x[4], y[4]).
/// scalars: array of n scalars, each 4 uint64_t in Montgomery form.
///          zkMetal expects uint32_t* -- we reinterpret (same memory, LE).
/// result: Jacobian projective output, 12 uint64_t (X[4], Y[4], Z[4]).
void msm_bn254_g1(const uint64_t* points_affine,
                   const uint64_t* scalars,
                   size_t n,
                   uint64_t* result_jacobian);

/// Grumpkin MSM (same layout conventions).
void msm_grumpkin(const uint64_t* points_affine,
                  const uint64_t* scalars,
                  size_t n,
                  uint64_t* result_jacobian);

// ============================================================
// NTT -- Number Theoretic Transform
// ============================================================

/// Forward NTT on BN254 Fr, in-place.
/// data: n field elements (n must be power of 2).
/// logN: log2(n).
void ntt_forward(uint64_t* data, size_t n);

/// Inverse NTT on BN254 Fr, in-place (includes 1/n scaling).
void ntt_inverse(uint64_t* data, size_t n);

// ============================================================
// Poseidon2 hash
// ============================================================

/// Poseidon2 2-to-1 compression: hash two Fr elements to one.
inline void poseidon2_hash(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    poseidon2_hash_cpu(a, b, out);
}

/// Poseidon2 full permutation on state width 3 (12 uint64_t).
inline void poseidon2_permute(const uint64_t state[12], uint64_t result[12]) {
    poseidon2_permutation_cpu(state, result);
}

/// Batch Poseidon2: hash count pairs in parallel.
/// input: count * 2 field elements (pairs of [a, b]).
/// output: count field elements.
inline void poseidon2_hash_batch(const uint64_t* input, size_t count, uint64_t* output) {
    poseidon2_hash_batch_cpu(input, static_cast<int>(count), output);
}

/// Poseidon2 Merkle tree builder.
inline void poseidon2_merkle_tree(const uint64_t* leaves, int n, uint64_t* tree) {
    poseidon2_merkle_tree_cpu(leaves, n, tree);
}

// ============================================================
// BN254 Pairing
// ============================================================

/// Full pairing: e(P, Q) -> Fp12.
/// p_aff: G1 affine point (8 uint64_t).
/// q_aff: G2 affine point (16 uint64_t).
/// result: Fp12 element (48 uint64_t).
inline void pairing(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]) {
    bn254_pairing(p_aff, q_aff, result);
}

/// Multi-pairing check: product(e(P_i, Q_i)) == 1.
/// pairs: interleaved [G1_aff(8), G2_aff(16)] * n.
/// Returns true if the pairing product is the identity.
inline bool pairing_check(const uint64_t* pairs, int n) {
    return bn254_pairing_check(pairs, n) == 1;
}

/// Miller loop only.
inline void miller_loop(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]) {
    bn254_miller_loop(p_aff, q_aff, result);
}

/// Final exponentiation only.
inline void final_exp(const uint64_t f[48], uint64_t result[48]) {
    bn254_final_exp(f, result);
}

// ============================================================
// BN254 G1 point operations
// ============================================================

inline void point_add(const uint64_t* p, const uint64_t* q, uint64_t* r) {
    bn254_point_add(p, q, r);
}

inline void point_add_mixed(const uint64_t* p, const uint64_t* q_aff, uint64_t* r) {
    bn254_point_add_mixed(p, q_aff, r);
}

inline void projective_to_affine(const uint64_t* p, uint64_t* affine) {
    bn254_projective_to_affine(p, affine);
}

inline void batch_to_affine(const uint64_t* proj, uint64_t* aff, int n) {
    bn254_batch_to_affine(proj, aff, n);
}

// ============================================================
// Grumpkin point operations (for Goblin/Eccvm)
// ============================================================

inline void grumpkin_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]) {
    ::grumpkin_point_add(p, q, r);
}

inline void grumpkin_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]) {
    ::grumpkin_point_add_mixed(p, q_aff, r);
}

// ============================================================
// Keccak-256 (for transcript hashing)
// ============================================================

inline void keccak256(const uint8_t* input, size_t len, uint8_t output[32]) {
    keccak256_hash_neon(input, len, output);
}

inline void keccak256_pair(const uint8_t a[32], const uint8_t b[32], uint8_t output[32]) {
    keccak256_hash_pair_neon(a, b, output);
}

// ============================================================
// Protocol accelerators (sumcheck, MLE)
// ============================================================

inline void full_sumcheck(const uint64_t* evals, int numVars,
                          const uint64_t* challenges,
                          uint64_t* rounds, uint64_t* finalEval) {
    bn254_fr_full_sumcheck(evals, numVars, challenges, rounds, finalEval);
}

inline void mle_eval(const uint64_t* evals, int numVars,
                     const uint64_t* point, uint64_t result[4]) {
    bn254_fr_mle_eval(evals, numVars, point, result);
}

} // namespace zkmetal

#endif // HAS_ZKMETAL
