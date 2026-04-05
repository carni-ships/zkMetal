/*
 * zkmetal.h -- Public C API for zkMetal cryptographic primitives
 *
 * ARM64/Apple Silicon only (NEON + PMULL intrinsics required).
 * All 256-bit field elements use Montgomery representation unless noted.
 * Little-endian limb ordering throughout.
 *
 * Link: -lzkmetal  or  pkg-config --cflags --libs zkmetal
 */

#ifndef ZKMETAL_H
#define ZKMETAL_H

#include <stdint.h>
#include <stddef.h>

#define ZKMETAL_VERSION_MAJOR 0
#define ZKMETAL_VERSION_MINOR 1
#define ZKMETAL_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Type aliases
 *
 * Field elements are fixed-size arrays of uint64_t limbs (Montgomery form).
 * Points use Jacobian projective coordinates: (X, Y, Z).
 * Affine points omit Z (implicitly 1).
 * ================================================================ */

/* BN254 / Grumpkin / secp256k1 -- 4-limb (256-bit) fields */
typedef uint64_t zk_bn254_fr[4];
typedef uint64_t zk_bn254_fq[4];
typedef uint64_t zk_secp256k1_fp[4];
typedef uint64_t zk_secp256k1_fr[4];

/* BLS12-381 -- Fr is 4-limb, Fp is 6-limb (384-bit) */
typedef uint64_t zk_bls12_381_fr[4];
typedef uint64_t zk_bls12_381_fp[6];

/* BLS12-377 -- Fr is 4-limb, Fq is 6-limb (384-bit) */
typedef uint64_t zk_bls12_377_fr[4];
typedef uint64_t zk_bls12_377_fq[6];

/* Pasta curves -- 4-limb */
typedef uint64_t zk_pallas_fp[4];
typedef uint64_t zk_vesta_fp[4];

/* Stark252 -- 4-limb */
typedef uint64_t zk_stark252_fp[4];

/* Ed25519 -- 4-limb Fp (base) and Fq (scalar) */
typedef uint64_t zk_ed25519_fp[4];
typedef uint64_t zk_ed25519_fq[4];

/* ================================================================
 * BN254 Fr (scalar field) Arithmetic
 * p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
 * ================================================================ */

/** Montgomery multiplication: r = a * b * R^{-1} mod p. */
void bn254_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Modular squaring (upper-triangle, fewer muls than mul). */
void bn254_fr_sqr(const uint64_t a[4], uint64_t r[4]);

/** Modular addition: r = (a + b) mod p. */
void bn254_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Modular subtraction: r = (a - b) mod p. */
void bn254_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Modular negation: r = -a mod p. */
void bn254_fr_neg(const uint64_t a[4], uint64_t r[4]);

/** Modular exponentiation: r = a^exp mod p (64-bit exponent). */
void bn254_fr_pow(const uint64_t a[4], uint64_t exp, uint64_t r[4]);

/** Modular inverse via Fermat's little theorem: r = a^{p-2} mod p. */
void bn254_fr_inverse(const uint64_t a[4], uint64_t r[4]);

/** Equality check: returns 1 if a == b, 0 otherwise. */
int bn254_fr_eq(const uint64_t a[4], const uint64_t b[4]);

/** Batch inverse via Montgomery's trick: out[i] = a[i]^{-1}. */
void bn254_fr_batch_inverse(const uint64_t *a, int n, uint64_t *out);

/** Horner polynomial evaluation: result = sum(coeffs[i]*z^i). */
void bn254_fr_horner_eval(const uint64_t *coeffs, int n, const uint64_t z[4],
                           uint64_t result[4]);

/** Synthetic division: quotient = (p(x) - p(z)) / (x - z). */
void bn254_fr_synthetic_div(const uint64_t *coeffs, const uint64_t *z,
                             int n, uint64_t *quotient);

/** Fused eval + synthetic division in one pass. */
void bn254_fr_eval_and_div(const uint64_t *coeffs, int n, const uint64_t z[4],
                            uint64_t eval_out[4], uint64_t *quotient);

/** Inner product: result = sum(a[i] * b[i]). */
void bn254_fr_inner_product(const uint64_t *a, const uint64_t *b, int n, uint64_t *result);

/** Vector fold: out[i] = a[i]*x + b[i]*xInv. */
void bn254_fr_vector_fold(const uint64_t *a, const uint64_t *b,
                           const uint64_t *x, const uint64_t *xInv,
                           int n, uint64_t *out);

/** Vector sum: result = sum(a[i]). */
void bn254_fr_vector_sum(const uint64_t *a, int n, uint64_t result[4]);

/** Batch convert Fr from Montgomery to uint32 limbs. */
void bn254_fr_batch_to_limbs(const uint64_t *mont, uint32_t *limbs, int n);

/* -- BN254 Fr NTT ------------------------------------------------ */

/** Forward NTT on BN254 Fr (Cooley-Tukey DIT, in-place). */
void bn254_fr_ntt(uint64_t *data, int logN);

/** Inverse NTT on BN254 Fr (DIF + bit-reversal + 1/n scaling). */
void bn254_fr_intt(uint64_t *data, int logN);

/* -- BN254 Fr batch ops (NEON vectorized, multi-threaded) --------- */

/** Batch add: result[i] = (a[i] + b[i]) mod p. */
void bn254_fr_batch_add_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n);

/** Batch subtract: result[i] = (a[i] - b[i]) mod p. */
void bn254_fr_batch_sub_neon(uint64_t *result, const uint64_t *a,
                              const uint64_t *b, int n);

/** Batch negate: result[i] = -a[i] mod p. */
void bn254_fr_batch_neg_neon(uint64_t *result, const uint64_t *a, int n);

/** Batch scalar multiply: result[i] = a[i] * scalar mod p. */
void bn254_fr_batch_mul_scalar_neon(uint64_t *result, const uint64_t *a,
                                     const uint64_t *scalar, int n);

/** Multi-threaded batch add (auto-threads for n >= 4096). */
void bn254_fr_batch_add_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n);

/** Multi-threaded batch subtract. */
void bn254_fr_batch_sub_parallel(uint64_t *result, const uint64_t *a,
                                  const uint64_t *b, int n);

/** Multi-threaded batch negate. */
void bn254_fr_batch_neg_parallel(uint64_t *result, const uint64_t *a, int n);

/** Multi-threaded batch scalar multiply. */
void bn254_fr_batch_mul_scalar_parallel(uint64_t *result, const uint64_t *a,
                                         const uint64_t *scalar, int n);

/** Fused pointwise mul-sub: result[i] = a[i]*b[i] - c[i]. */
void bn254_fr_pointwise_mul_sub(const uint64_t *a, const uint64_t *b,
                                 const uint64_t *c, uint64_t *result, int n);

/** Coefficient division by vanishing polynomial Z_H(x). */
void bn254_fr_coeff_div_vanishing(const uint64_t *pCoeffs, int domainN,
                                   uint64_t *hCoeffs);

/** Linear combine: result[i] = running[i] + rho * new_vals[i]. */
void bn254_fr_linear_combine(const uint64_t *running, const uint64_t *new_vals,
                              const uint64_t rho[4], uint64_t *result, int count);

/** Basefold single-round fold. */
void bn254_fr_basefold_fold(const uint64_t *evals, uint64_t *result,
                             const uint64_t *alpha, uint32_t halfN);

/** Basefold fold all rounds. */
void bn254_fr_basefold_fold_all(const uint64_t *evals, int numVars,
                                 const uint64_t *point, uint64_t *out_layers);

/** WHIR polynomial fold: result[j] = sum(beta^k * evals[j*r+k]). */
void bn254_fr_whir_fold(const uint64_t *evals, int n,
                         const uint64_t beta[4],
                         int reductionFactor,
                         uint64_t *result);

/* -- BN254 Fr Montgomery assembly (ARM64) ------------------------- */

/** ARM64 asm CIOS Montgomery multiply for BN254 Fr. */
void mont_mul_asm(uint64_t *result, const uint64_t *a, const uint64_t *b);

/** C CIOS Montgomery multiply for BN254 Fr (reference). */
void mont_mul_c(uint64_t *result, const uint64_t *a, const uint64_t *b);

/** Batch multiply via asm: data[i] *= multiplier. */
void mont_mul_batch_asm(uint64_t *data, const uint64_t *multiplier, int n);

/** Batch pairwise multiply via asm: result[i] = a[i] * b[i]. */
void mont_mul_pair_batch_asm(uint64_t *result, const uint64_t *a, const uint64_t *b, int n);

/** Batch multiply via C: data[i] *= multiplier. */
void mont_mul_batch_c(uint64_t *data, const uint64_t *multiplier, int n);

/** Batch pairwise multiply via C: result[i] = a[i] * b[i]. */
void mont_mul_pair_batch_c(uint64_t *result, const uint64_t *a, const uint64_t *b, int n);

/* -- BN254 Fq (base field) --------------------------------------- */

/** Fq optimized squaring. */
void bn254_fp_sqr(const uint64_t a[4], uint64_t r[4]);

/** Fq inverse via Fermat. */
void bn254_fp_inv(const uint64_t a[4], uint64_t r[4]);

/** Fq square root: returns 1 if exists, 0 otherwise. */
int bn254_fp_sqrt(const uint64_t a[4], uint64_t r[4]);

/* ================================================================
 * BN254 G1 Curve Operations (y^2 = x^3 + 3 over Fq)
 * Jacobian projective: 12 uint64_t (X[4], Y[4], Z[4])
 * Affine: 8 uint64_t (X[4], Y[4])
 * ================================================================ */

/** Projective point addition. */
void bn254_point_add(const uint64_t *p, const uint64_t *q, uint64_t *r);

/** Mixed addition: projective P + affine Q. */
void bn254_point_add_mixed(const uint64_t *p, const uint64_t *q_aff, uint64_t *r);

/** Scalar multiplication (windowed). */
void bn254_point_scalar_mul(const uint64_t *p, const uint32_t *scalar, uint64_t *r);

/** Projective to affine conversion. */
void bn254_projective_to_affine(const uint64_t *p, uint64_t *affine);

/** Batch projective to affine (Montgomery's trick). */
void bn254_batch_to_affine(const uint64_t *proj, uint64_t *aff, int n);

/** Pippenger MSM on BN254 G1 (multi-threaded). */
void bn254_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                          int n, uint64_t *result);

/** MSM from projective points (direct accumulation, for small n). */
void bn254_msm_projective(const uint64_t *points, const uint32_t *scalars,
                           int n, uint64_t *result);

/** Dual MSM: two MSMs with shared thread pool. */
void bn254_dual_msm_projective(
    const uint64_t *points1, const uint32_t *scalars1, int n1,
    const uint64_t *points2, const uint32_t *scalars2, int n2,
    uint64_t result1[12], uint64_t result2[12]);

/** Fold generators for IPA: result[i] = scalarMul(GL[i], xInv) + scalarMul(GR[i], x). */
void bn254_fold_generators(const uint64_t *GL, const uint64_t *GR,
                            const uint32_t *x, const uint32_t *xInv,
                            int halfLen, uint64_t *result);

/** BGMW fixed-base precompute: build lookup tables. */
void bgmw_precompute(const uint64_t *generators_affine, int n,
                     int window_bits, uint64_t *table_out);

/** BGMW fixed-base MSM using precomputed tables. */
void bgmw_msm(const uint64_t *table, int n, int window_bits,
              const uint32_t *scalars, uint64_t *result);

/* ================================================================
 * BN254 Pairing (Fp2/Fp6/Fp12 tower + Miller loop + final exp)
 * G1Affine = 8x64, G2Affine = 16x64, Fp12 = 48x64
 * ================================================================ */

/** Miller loop: e(P, Q) before final exponentiation. */
void bn254_miller_loop(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]);

/** Final exponentiation: f^{(p^12-1)/r}. */
void bn254_final_exp(const uint64_t f[48], uint64_t result[48]);

/** Full pairing: e(P, Q). */
void bn254_pairing(const uint64_t p_aff[8], const uint64_t q_aff[16], uint64_t result[48]);

/** Multi-pairing check: product(e(P_i, Q_i)) == 1. Returns 1 if true. */
int bn254_pairing_check(const uint64_t *pairs, int n);

/** Batch KZG verification for Marlin. Returns 1 on success. */
int bn254_batch_kzg_verify(const uint64_t *srsG1, const uint64_t *srsSecret,
                           const uint64_t *commitments, const uint64_t *points,
                           const uint64_t *evaluations, const uint64_t *witnesses,
                           const uint64_t *batchChallenge, int n);

/* ================================================================
 * Grumpkin Curve (y^2 = x^3 - 17 over BN254 Fr)
 * ================================================================ */

/** Grumpkin projective point addition. */
void grumpkin_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]);

/** Grumpkin point doubling. */
void grumpkin_point_double(const uint64_t p[12], uint64_t r[12]);

/** Grumpkin mixed addition: projective P + affine Q. */
void grumpkin_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]);

/** Grumpkin scalar multiplication (windowed, w=4). */
void grumpkin_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]);

/** Grumpkin Pippenger MSM (multi-threaded). */
void grumpkin_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                             int n, uint64_t *result);

/* ================================================================
 * secp256k1 Curve Operations
 * ================================================================ */

/** secp256k1 projective point addition. */
void secp256k1_point_add(const uint64_t *p, const uint64_t *q, uint64_t *r);

/** secp256k1 scalar multiplication. */
void secp256k1_point_scalar_mul(const uint64_t *p, const uint64_t *scalar, uint64_t *r);

/** secp256k1 projective to affine. */
void secp256k1_point_to_affine(const uint64_t *p, uint64_t *ax, uint64_t *ay);

/** secp256k1 Pippenger MSM (multi-threaded). */
void secp256k1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                              int n, uint64_t *result);

/** Shamir's trick: r = s1*P1 + s2*P2 (simultaneous double-and-add). */
void secp256k1_shamir_double_mul(const uint64_t *p1, const uint64_t *s1,
                                  const uint64_t *p2, const uint64_t *s2,
                                  uint64_t *r);

/* -- secp256k1 Fr (scalar field) ---------------------------------- */

/** secp256k1 Fr Montgomery multiplication. */
void secp256k1_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** secp256k1 Fr inverse via Fermat. */
void secp256k1_fr_inverse(const uint64_t a[4], uint64_t r[4]);

/** secp256k1 Fr batch inverse. */
void secp256k1_fr_batch_inverse(const uint64_t *a, int n, uint64_t *out);

/** ECDSA batch verification: prepare MSM inputs. Returns 0 on success. */
int secp256k1_ecdsa_batch_prepare(
    const uint64_t *sigs, const uint64_t *pubkeys, const uint8_t *recov,
    int n, uint64_t *out_points, uint32_t *out_scalars);

/* ================================================================
 * BLS12-381 Fr (scalar field, 4-limb, 255-bit)
 * ================================================================ */

/** Fr Montgomery multiplication. */
void bls12_381_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Fr squaring. */
void bls12_381_fr_sqr(const uint64_t a[4], uint64_t r[4]);

/** Fr addition. */
void bls12_381_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Fr subtraction. */
void bls12_381_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Fr negation. */
void bls12_381_fr_neg(const uint64_t a[4], uint64_t r[4]);

/* ================================================================
 * BLS12-381 Fp (base field, 6-limb, 381-bit)
 * ================================================================ */

/** Fp Montgomery multiplication. */
void bls12_381_fp_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);

/** Fp squaring. */
void bls12_381_fp_sqr(const uint64_t a[6], uint64_t r[6]);

/** Fp addition. */
void bls12_381_fp_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);

/** Fp subtraction. */
void bls12_381_fp_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);

/** Fp negation. */
void bls12_381_fp_neg(const uint64_t a[6], uint64_t r[6]);

/** Fp inverse. */
void bls12_381_fp_inv_ext(const uint64_t a[6], uint64_t r[6]);

/** Fp square root: returns 1 if exists, 0 otherwise. */
int bls12_381_fp_sqrt(const uint64_t a[6], uint64_t r[6]);

/* ================================================================
 * BLS12-381 Fp2 Tower (Fp[u]/(u^2+1), 12 uint64_t per element)
 * ================================================================ */

/** Fp2 addition. */
void bls12_381_fp2_add(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]);

/** Fp2 subtraction. */
void bls12_381_fp2_sub(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]);

/** Fp2 negation. */
void bls12_381_fp2_neg(const uint64_t a[12], uint64_t r[12]);

/** Fp2 multiplication. */
void bls12_381_fp2_mul(const uint64_t a[12], const uint64_t b[12], uint64_t r[12]);

/** Fp2 squaring. */
void bls12_381_fp2_sqr(const uint64_t a[12], uint64_t r[12]);

/** Fp2 conjugation. */
void bls12_381_fp2_conj(const uint64_t a[12], uint64_t r[12]);

/** Fp2 multiply by non-residue. */
void bls12_381_fp2_mul_by_nonresidue(const uint64_t a[12], uint64_t r[12]);

/** Fp2 inverse. */
void bls12_381_fp2_inv(const uint64_t a[12], uint64_t r[12]);

/* ================================================================
 * BLS12-381 Higher Tower (Fp6/Fp12) + Pairing
 * ================================================================ */

/** Fp6 multiplication (Fp6 = 36 uint64_t). */
void bls12_381_fp6_mul(const uint64_t a[36], const uint64_t b[36], uint64_t r[36]);

/** Fp6 squaring. */
void bls12_381_fp6_sqr(const uint64_t a[36], uint64_t r[36]);

/** Fp12 multiplication (Fp12 = 72 uint64_t). */
void bls12_381_fp12_mul(const uint64_t a[72], const uint64_t b[72], uint64_t r[72]);

/** Fp12 squaring. */
void bls12_381_fp12_sqr(const uint64_t a[72], uint64_t r[72]);

/** Fp12 inverse. */
void bls12_381_fp12_inv(const uint64_t a[72], uint64_t r[72]);

/** Fp12 conjugation. */
void bls12_381_fp12_conj(const uint64_t a[72], uint64_t r[72]);

/** Miller loop: e(P, Q) before final exponentiation. */
void bls12_381_miller_loop(const uint64_t p_aff[12], const uint64_t q_aff[24], uint64_t result[72]);

/** Final exponentiation. */
void bls12_381_final_exp(const uint64_t f[72], uint64_t result[72]);

/** Full pairing: e(P, Q). */
void bls12_381_pairing(const uint64_t p_aff[12], const uint64_t q_aff[24], uint64_t result[72]);

/** Multi-pairing check: product(e(P_i, Q_i)) == 1. Returns 1 if true. */
int bls12_381_pairing_check(const uint64_t *pairs, int n);

/* ================================================================
 * BLS12-381 G1 Curve (Jacobian projective, 6-limb coords)
 * Projective: 18 uint64_t, Affine: 12 uint64_t
 * ================================================================ */

/** G1 point addition. */
void bls12_381_g1_point_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]);

/** G1 point doubling. */
void bls12_381_g1_point_double(const uint64_t p[18], uint64_t r[18]);

/** G1 mixed addition: projective + affine. */
void bls12_381_g1_point_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]);

/** G1 scalar multiplication. */
void bls12_381_g1_scalar_mul(const uint64_t p[18], const uint64_t scalar[4], uint64_t r[18]);

/** G1 Pippenger MSM (multi-threaded). */
void bls12_381_g1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                                 int n, uint64_t *result);

/* ================================================================
 * BLS12-381 G2 Curve (Fp2 coords, 36 uint64_t projective)
 * ================================================================ */

/** G2 point addition. */
void bls12_381_g2_point_add(const uint64_t p[36], const uint64_t q[36], uint64_t r[36]);

/** G2 point doubling. */
void bls12_381_g2_point_double(const uint64_t p[36], uint64_t r[36]);

/** G2 mixed addition. */
void bls12_381_g2_point_add_mixed(const uint64_t p[36], const uint64_t q_aff[24], uint64_t r[36]);

/** G2 scalar multiplication (4-limb scalar). */
void bls12_381_g2_scalar_mul(const uint64_t p[36], const uint64_t scalar[4], uint64_t r[36]);

/** G2 scalar multiplication (arbitrary-width scalar). */
void bls12_381_g2_scalar_mul_wide(const uint64_t p[36], const uint64_t *scalar, int n_limbs, uint64_t r[36]);

/* ================================================================
 * BLS12-381 Hash-to-Curve G2 (RFC 9380, SSWU + 3-isogeny)
 * ================================================================ */

/** Hash arbitrary message to G2 with custom DST. */
void bls12_381_hash_to_g2(const uint8_t *msg, size_t msg_len,
                           const uint8_t *dst, size_t dst_len,
                           uint64_t result[36]);

/** Hash to G2 with default BLS DST. */
void bls12_381_hash_to_g2_default(const uint8_t *msg, size_t msg_len, uint64_t result[36]);

/** Clear G2 cofactor. */
void bls12_381_g2_clear_cofactor(const uint64_t p[36], uint64_t r[36]);

/* ================================================================
 * BLS12-377 Fq (base field, 6-limb, 377-bit)
 * ================================================================ */

/** Fq Montgomery multiplication. */
void bls12_377_fq_mul(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);

/** Fq squaring. */
void bls12_377_fq_sqr(const uint64_t a[6], uint64_t r[6]);

/** Fq addition. */
void bls12_377_fq_add(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);

/** Fq subtraction. */
void bls12_377_fq_sub(const uint64_t a[6], const uint64_t b[6], uint64_t r[6]);

/** Fq negation. */
void bls12_377_fq_neg(const uint64_t a[6], uint64_t r[6]);

/** Fq inverse via Fermat. */
void bls12_377_fq_inverse(const uint64_t a[6], uint64_t r[6]);

/* ================================================================
 * BLS12-377 Fr (scalar field, 4-limb, 253-bit)
 * ================================================================ */

/** Fr Montgomery multiplication. */
void bls12_377_fr_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Fr squaring. */
void bls12_377_fr_sqr(const uint64_t a[4], uint64_t r[4]);

/** Fr addition. */
void bls12_377_fr_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Fr subtraction. */
void bls12_377_fr_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);

/** Fr negation. */
void bls12_377_fr_neg(const uint64_t a[4], uint64_t r[4]);

/* -- BLS12-377 Fr NTT --------------------------------------------- */

/** Forward NTT on BLS12-377 Fr. */
void bls12_377_fr_ntt(uint64_t *data, int logN);

/** Inverse NTT on BLS12-377 Fr. */
void bls12_377_fr_intt(uint64_t *data, int logN);

/* ================================================================
 * BLS12-377 G1 Curve (Jacobian projective, y^2 = x^3 + 1)
 * Projective: 18 uint64_t, Affine: 12 uint64_t
 * ================================================================ */

/** G1 point addition. */
void bls12_377_g1_point_add(const uint64_t p[18], const uint64_t q[18], uint64_t r[18]);

/** G1 point doubling. */
void bls12_377_g1_point_double(const uint64_t p[18], uint64_t r[18]);

/** G1 mixed addition. */
void bls12_377_g1_point_add_mixed(const uint64_t p[18], const uint64_t q_aff[12], uint64_t r[18]);

/** G1 scalar multiplication. */
void bls12_377_g1_scalar_mul(const uint64_t p[18], const uint64_t scalar[6], uint64_t r[18]);

/** G1 projective to affine. */
void bls12_377_g1_to_affine(const uint64_t p[18], uint64_t aff[12]);

/** G1 Pippenger MSM (multi-threaded). */
void bls12_377_g1_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                                 int n, uint64_t *result);

/* ================================================================
 * Pasta Curves (Pallas & Vesta)
 * ================================================================ */

/* -- Pallas (y^2 = x^3 + 5) -------------------------------------- */

void pallas_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void pallas_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void pallas_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void pallas_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void pallas_fp_neg(const uint64_t a[4], uint64_t r[4]);
void pallas_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]);
void pallas_point_double(const uint64_t p[12], uint64_t r[12]);
void pallas_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]);
void pallas_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]);

/** Pallas Pippenger MSM. */
void pallas_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                           int n, uint64_t *result);

/* -- Vesta (y^2 = x^3 + 5) --------------------------------------- */

void vesta_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void vesta_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void vesta_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void vesta_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void vesta_fp_neg(const uint64_t a[4], uint64_t r[4]);
void vesta_point_add(const uint64_t p[12], const uint64_t q[12], uint64_t r[12]);
void vesta_point_double(const uint64_t p[12], uint64_t r[12]);
void vesta_point_add_mixed(const uint64_t p[12], const uint64_t q_aff[8], uint64_t r[12]);
void vesta_scalar_mul(const uint64_t p[12], const uint64_t scalar[4], uint64_t r[12]);

/** Vesta Pippenger MSM. */
void vesta_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                          int n, uint64_t *result);

/* ================================================================
 * Ed25519 (twisted Edwards, base field p = 2^255 - 19)
 * Extended coordinates: 16 uint64_t (X[4], Y[4], Z[4], T[4])
 * Affine: 8 uint64_t (X[4], Y[4])
 * ================================================================ */

/* -- Fp (base field, Solinas reduction) --------------------------- */

void ed25519_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void ed25519_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fp_neg(const uint64_t a[4], uint64_t r[4]);
void ed25519_fp_inverse(const uint64_t a[4], uint64_t r[4]);

/* -- Fq (scalar field) ------------------------------------------- */

void ed25519_fq_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fq_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fq_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void ed25519_fq_from_raw(const uint64_t raw[4], uint64_t mont[4]);
void ed25519_fq_to_raw(const uint64_t mont[4], uint64_t raw[4]);
void ed25519_fq_from_bytes64(const uint8_t bytes[64], uint64_t mont[4]);
void ed25519_fq_to_bytes(const uint64_t mont[4], uint8_t bytes[32]);

/* -- Curve ops ---------------------------------------------------- */

/** Point addition (extended coordinates). */
void ed25519_point_add_c(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]);

/** Point doubling. */
void ed25519_point_double_c(const uint64_t p[16], uint64_t r[16]);

/** Scalar multiplication. */
void ed25519_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]);

/** Extended to affine. */
void ed25519_point_to_affine(const uint64_t p[16], uint64_t aff[8]);

/** Montgomery to direct field representation. */
void ed25519_mont_to_direct(const uint64_t mont[4], uint64_t direct[4]);

/** Direct to Montgomery field representation. */
void ed25519_direct_to_mont(const uint64_t direct[4], uint64_t mont[4]);

/** Pippenger MSM on Ed25519. */
void ed25519_pippenger_msm(const uint64_t *points, const uint32_t *scalars,
                            int n, uint64_t *result);

/** Shamir's trick: r = s*G + h*A (for EdDSA verify). */
void ed25519_shamir_double_mul(const uint64_t G[16], const uint64_t s[4],
                                const uint64_t A[16], const uint64_t h[4],
                                uint64_t result[16]);

/* -- EdDSA sign/verify -------------------------------------------- */

/** Compute R = r_scalar * G for EdDSA signing. */
void ed25519_eddsa_sign_compute_r(const uint64_t gen[16], const uint64_t r_scalar[4],
                                   uint64_t r_point[16]);

/** Compute s = r + k*a for EdDSA signing. */
void ed25519_eddsa_sign_compute_s(const uint64_t r_mont[4], const uint64_t k_mont[4],
                                   const uint64_t a_mont[4], uint64_t s_mont[4]);

/** EdDSA verify: returns 1 if valid, 0 otherwise. */
int ed25519_eddsa_verify(const uint64_t gen[16], const uint64_t s_raw[4],
                          const uint64_t r_point[16], const uint64_t h_raw[4],
                          const uint64_t pub_key[16]);

/* ================================================================
 * BabyJubjub (twisted Edwards over BN254 Fr)
 * Extended coordinates: 16 uint64_t
 * ================================================================ */

/** Point addition. */
void babyjubjub_point_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]);

/** Point doubling. */
void babyjubjub_point_double(const uint64_t p[16], uint64_t r[16]);

/** Scalar multiplication. */
void babyjubjub_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]);

/* ================================================================
 * Jubjub (twisted Edwards over BLS12-381 Fr)
 * ================================================================ */

/** Point addition. */
void jubjub_point_add(const uint64_t p[16], const uint64_t q[16], uint64_t r[16]);

/** Point doubling. */
void jubjub_point_double(const uint64_t p[16], uint64_t r[16]);

/** Scalar multiplication. */
void jubjub_scalar_mul(const uint64_t p[16], const uint64_t scalar[4], uint64_t r[16]);

/* ================================================================
 * Small Fields
 * ================================================================ */

/* -- BabyBear (p = 0x78000001, 31-bit) --------------------------- */

/** Forward NTT on BabyBear (NEON DIT, in-place). */
void babybear_ntt_neon(uint32_t *data, int logN);

/** Inverse NTT on BabyBear (NEON DIF + bit-reversal + scaling). */
void babybear_intt_neon(uint32_t *data, int logN);

/* -- Goldilocks (p = 2^64 - 2^32 + 1) --------------------------- */

/** Forward NTT on Goldilocks (scalar). */
void goldilocks_ntt(uint64_t *data, int logN);

/** Inverse NTT on Goldilocks (scalar). */
void goldilocks_intt(uint64_t *data, int logN);

/** Forward NTT on Goldilocks (NEON vectorized). */
void goldilocks_ntt_neon(uint64_t *data, int logN);

/** Inverse NTT on Goldilocks (NEON vectorized). */
void goldilocks_intt_neon(uint64_t *data, int logN);

/** Batch add: out[i] = a[i] + b[i] mod p. */
void gl_batch_add_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/** Batch subtract: out[i] = a[i] - b[i] mod p. */
void gl_batch_sub_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/** Batch multiply: out[i] = a[i] * b[i] mod p. */
void gl_batch_mul_neon(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/** NTT DIT butterfly stage. */
void gl_batch_butterfly_neon(uint64_t *data, const uint64_t *twiddles, int halfBlock, int nBlocks);

/** NTT DIF butterfly stage. */
void gl_batch_butterfly_dif_neon(uint64_t *data, const uint64_t *twiddles, int halfBlock, int nBlocks);

/* -- Stark252 (p = 2^251 + 17*2^192 + 1) ------------------------ */

void stark252_fp_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void stark252_fp_sqr(const uint64_t a[4], uint64_t r[4]);
void stark252_fp_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void stark252_fp_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
void stark252_fp_neg(const uint64_t a[4], uint64_t r[4]);

/** Forward NTT on Stark252. */
void stark252_ntt(uint64_t *data, int logN);

/** Inverse NTT on Stark252. */
void stark252_intt(uint64_t *data, int logN);

/* ================================================================
 * Binary Tower Fields (GF(2) towers for Binius, ARM64 PMULL)
 * ================================================================ */

/** Initialize GF(2^8) log/exp/inv tables. Call once before use. */
void bt_gf8_init(void);

uint8_t bt_gf8_mul(uint8_t a, uint8_t b);
uint8_t bt_gf8_sqr(uint8_t a);
uint8_t bt_gf8_inv(uint8_t a);

uint16_t bt_gf16_mul(uint16_t a, uint16_t b);
uint16_t bt_gf16_sqr(uint16_t a);

uint32_t bt_gf32_mul(uint32_t a, uint32_t b);
uint32_t bt_gf32_sqr(uint32_t a);

uint64_t bt_gf64_mul(uint64_t a, uint64_t b);
uint64_t bt_gf64_sqr(uint64_t a);
uint64_t bt_gf64_inv(uint64_t a);

void bt_gf128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
void bt_gf128_sqr(const uint64_t a[2], uint64_t r[2]);
void bt_gf128_add(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
void bt_gf128_inv(const uint64_t a[2], uint64_t r[2]);

void bt_tower128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
void bt_tower128_sqr(const uint64_t a[2], uint64_t r[2]);
void bt_tower128_add(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);
void bt_tower128_inv(const uint64_t a[2], uint64_t r[2]);

/** Batch GF(2^64) multiply. */
void bt_gf64_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/** Batch GF(2^64) add (XOR, NEON). */
void bt_gf64_batch_add(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/** Batch GF(2^64) square. */
void bt_gf64_batch_sqr(const uint64_t *a, uint64_t *out, int n);

/** Batch GF(2^128) multiply. */
void bt_gf128_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/** Batch GF(2^128) add. */
void bt_gf128_batch_add(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/** Batch tower GF(2^128) multiply. */
void bt_tower128_batch_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, int n);

/* ================================================================
 * Hash Functions
 * ================================================================ */

/** Keccak-f1600 permutation (24 rounds, in-place, NEON). */
void keccak_f1600_neon(uint64_t state[25]);

/** Keccak-256 hash. */
void keccak256_hash_neon(const uint8_t *input, size_t len, uint8_t output[32]);

/** Keccak-256 of two 32-byte inputs (Merkle node). */
void keccak256_hash_pair_neon(const uint8_t a[32], const uint8_t b[32], uint8_t output[32]);

/** Batch Keccak-256 of n pairs. */
void keccak256_batch_hash_pairs_neon(const uint8_t *inputs, uint8_t *outputs, size_t n);

/** Blake3 hash (single chunk, NEON). */
void blake3_hash_neon(const uint8_t *input, size_t len, uint8_t output[32]);

/** Blake3 parent hash: left(32B) || right(32B) -> 32B. */
void blake3_hash_pair_neon(const uint8_t left[32], const uint8_t right[32],
                           uint8_t output[32]);

/** Batch Blake3 parent hashing. */
void blake3_batch_hash_pairs_neon(const uint8_t *inputs, uint8_t *outputs, size_t n);

/** Poseidon2 full permutation on 3 BN254 Fr elements. */
void poseidon2_permutation_cpu(const uint64_t state[12], uint64_t result[12]);

/** Poseidon2 2-to-1 compression hash. */
void poseidon2_hash_cpu(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]);

/** Batch Poseidon2 hash of pairs (multi-threaded). */
void poseidon2_hash_batch_cpu(const uint64_t *input, int count, uint64_t *output);

/** Poseidon2 Merkle tree builder (multi-threaded). */
void poseidon2_merkle_tree_cpu(const uint64_t *leaves, int n, uint64_t *tree);

/* ================================================================
 * Protocol-Specific Accelerators
 *
 * These are higher-level fused operations for specific ZK protocols.
 * They combine multiple primitives to avoid intermediate allocations.
 * ================================================================ */

/* -- Sumcheck / MLE ----------------------------------------------- */

/** Full sumcheck for a single multilinear polynomial. */
void bn254_fr_full_sumcheck(const uint64_t *evals, int numVars,
                             const uint64_t *challenges,
                             uint64_t *rounds, uint64_t *finalEval);

/** Evaluate multilinear extension at a point. */
void bn254_fr_mle_eval(const uint64_t *evals, int numVars,
                        const uint64_t *point, uint64_t result[4]);

/* -- Lookup / inverse evals -------------------------------------- */

void bn254_fr_inverse_evals(const uint64_t beta[4], const uint64_t *values,
                             int n, uint64_t *out);

void bn254_fr_inverse_evals_indexed(const uint64_t beta[4], const uint64_t *subtable,
                                     const int *indices, int n, uint64_t *out);

void bn254_fr_weighted_inverse_evals(const uint64_t beta[4], const uint64_t *values,
                                      const uint64_t *weights, int n, uint64_t *out);

void bn254_fr_inverse_mle_eval(const uint64_t beta[4], const uint64_t *subtable,
                                const int *indices, int n, int numVars,
                                const uint64_t *point, uint64_t result[4]);

void bn254_fr_batch_beta_add(const uint64_t *beta, const uint64_t *values,
                              const int *indices, int m, uint64_t *result);

void bn254_fr_batch_decompose(const uint64_t *lookups, int m,
                               int numChunks, int bitsPerChunk,
                               int *indices);

/* -- GKR --------------------------------------------------------- */

void gkr_eq_poly(const uint64_t *point, int n, uint64_t *eq);

void gkr_accumulate_wiring(
    const int32_t *gates, int numGates,
    const uint64_t *eqVals, const uint64_t weight[4],
    int inSize,
    uint64_t *accum, int accumCapacity,
    int32_t *nonzeroIndices, int *numNonzero);

void gkr_mle_fold(uint64_t *v, int half, const uint64_t challenge[4]);

void gkr_sumcheck_round_x(
    const uint64_t *wiring, int numEntries,
    const uint64_t *vx, int vxSize,
    const uint64_t *vy, int vySize,
    int nIn, int halfSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

void gkr_sumcheck_round_y(
    const uint64_t *wiring, int numEntries,
    const uint64_t vxScalar[4],
    const uint64_t *vy, int vySize,
    int halfSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

int gkr_wiring_reduce(
    const uint64_t *wiring, int numEntries,
    const uint64_t challenge[4],
    int halfSize,
    uint64_t *outWiring);

void gkr_sumcheck_step(
    const uint64_t *wiring, int numEntries,
    const uint64_t *curVx, int vxSize,
    const uint64_t *curVy, int vySize,
    int round, int nIn, int currentTableSize,
    uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

/* -- Spartan R1CS ------------------------------------------------ */

void spartan_sparse_matvec(const uint64_t *entries, int numEntries,
                           const uint64_t *z, int zLen,
                           uint64_t *result, int numRows);

void spartan_sc1_round(uint64_t *eqC, uint64_t *azC, uint64_t *bzC, uint64_t *czC,
                       int halfSize,
                       uint64_t s0[4], uint64_t s1[4], uint64_t s2[4], uint64_t s3[4]);

void spartan_fold_array(uint64_t *arr, int halfSize, const uint64_t ri[4]);

void spartan_sc2_round(uint64_t *wC, uint64_t *zC, int halfSize,
                       uint64_t s0[4], uint64_t s1[4], uint64_t s2[4]);

void spartan_build_weight_vec(const uint64_t *entries, int numEntries,
                              const uint64_t *eqRx, int eqRxLen,
                              const uint64_t weight[4],
                              uint64_t *wVec, int paddedN);

void spartan_eq_poly(const uint64_t *point, int n, uint64_t *eq);

void spartan_mle_eval(const uint64_t *evals, int numVars,
                      const uint64_t *point, uint64_t result[4]);

/* -- CCS (Customizable Constraint System) ------------------------ */

void ccs_sparse_matvec(uint64_t *result,
                       const int *rowPtr, const int *colIdx,
                       const uint64_t *values, const uint64_t *z,
                       int nRows);

void ccs_hadamard_accumulate(uint64_t *acc,
                             const uint64_t * const *matResultPtrs,
                             const int *nMatricesPerTerm,
                             const uint64_t *coefficients,
                             int nTerms, int maxDegree, int m);

void ccs_compute_term(uint64_t *result,
                      const uint64_t * const *matVecResults,
                      int nMatrices,
                      const uint64_t coeff[4],
                      int m);

/* -- Sparse matvec + MLE ----------------------------------------- */

void bn254_sparse_matvec_mle(
    const int *rowPtr, const int *colIdx, const uint64_t *values,
    int rows, const uint64_t *z,
    const uint64_t *point, int numVars, int padM,
    uint64_t result[4]);

/* -- Plonk ------------------------------------------------------- */

void plonk_compute_z_accumulator(
    const uint64_t *aEvals, const uint64_t *bEvals, const uint64_t *cEvals,
    const uint64_t *sigma1, const uint64_t *sigma2, const uint64_t *sigma3,
    const uint64_t *domain,
    const uint64_t beta[4], const uint64_t gamma[4],
    const uint64_t k1[4], const uint64_t k2[4],
    int n, uint64_t *zEvals);

/* -- Tensor proof ------------------------------------------------ */

void tensor_mat_vec_mul(const uint64_t *M, const uint64_t *vec,
                        int rows, int cols, uint64_t *result);

void tensor_inner_product_sumcheck(
    const uint64_t *evalsA, const uint64_t *evalsB,
    int numVars, const uint64_t *challenges,
    uint64_t *rounds, uint64_t *finalEval);

void tensor_eq_weighted_row(const uint64_t *M, const uint64_t *rowPoint,
                            int rows, int cols, uint64_t *result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ZKMETAL_H */
