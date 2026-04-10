// zkmetal.h — C FFI for zkMetal GPU-accelerated ZK primitives
// Apple Silicon only (Metal GPU backend)
//
// All field elements are 32 bytes (8x u32 limbs, little-endian Montgomery form).
// Points are affine: 64 bytes (x: 32B, y: 32B) in Montgomery form.
// Scalars for MSM are 32 bytes (8x u32 limbs, standard form, NOT Montgomery).
//
// Two API styles:
//   1. Engine-based: create/destroy engine handles, pass to compute functions.
//      Best for repeated calls — avoids re-initialization.
//   2. Auto (convenience): functions with _auto suffix use a lazy singleton.
//      Simpler API, engine created on first call, persists for process lifetime.

#ifndef ZKMETAL_H
#define ZKMETAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Error codes
// ============================================================================

typedef int32_t ZkMetalStatus;

#define ZKMETAL_SUCCESS           0
#define ZKMETAL_ERR_NO_GPU       -1
#define ZKMETAL_ERR_INVALID_INPUT -2
#define ZKMETAL_ERR_GPU_ERROR    -3
#define ZKMETAL_ERR_ALLOC_FAILED -4

// ============================================================================
// Opaque engine handles
// ============================================================================

typedef void* ZkMetalMSMEngine;
typedef void* ZkMetalPallasMSMEngine;
typedef void* ZkMetalVestaMSMEngine;
typedef void* ZkMetalNTTEngine;
typedef void* ZkMetalPoseidon2Engine;
typedef void* ZkMetalKeccakEngine;
typedef void* ZkMetalFRIEngine;
typedef void* ZkMetalPairingEngine;

// ============================================================================
// Engine lifecycle
// ============================================================================

/// Create a BN254 MSM engine. Returns ZKMETAL_SUCCESS or error code.
/// The engine handle is written to *out.
ZkMetalStatus zkmetal_msm_engine_create(ZkMetalMSMEngine* out);

/// Destroy an MSM engine, releasing GPU resources.
void zkmetal_msm_engine_destroy(ZkMetalMSMEngine engine);

/// Create a Pallas MSM engine (Pasta curve cycle — Mina/Kimchi).
ZkMetalStatus zkmetal_pallas_msm_engine_create(ZkMetalPallasMSMEngine* out);

/// Destroy a Pallas MSM engine.
void zkmetal_pallas_msm_engine_destroy(ZkMetalPallasMSMEngine engine);

/// Create a Vesta MSM engine (Pasta curve cycle — Mina/Kimchi).
ZkMetalStatus zkmetal_vesta_msm_engine_create(ZkMetalVestaMSMEngine* out);

/// Destroy a Vesta MSM engine.
void zkmetal_vesta_msm_engine_destroy(ZkMetalVestaMSMEngine engine);

/// Create a BN254 NTT engine.
ZkMetalStatus zkmetal_ntt_engine_create(ZkMetalNTTEngine* out);

/// Destroy an NTT engine.
void zkmetal_ntt_engine_destroy(ZkMetalNTTEngine engine);

/// Create a BN254 Poseidon2 engine.
ZkMetalStatus zkmetal_poseidon2_engine_create(ZkMetalPoseidon2Engine* out);

/// Destroy a Poseidon2 engine.
void zkmetal_poseidon2_engine_destroy(ZkMetalPoseidon2Engine engine);

/// Create a Keccak-256 GPU engine.
ZkMetalStatus zkmetal_keccak_engine_create(ZkMetalKeccakEngine* out);

/// Destroy a Keccak-256 engine.
void zkmetal_keccak_engine_destroy(ZkMetalKeccakEngine engine);

/// Create a FRI folding engine.
ZkMetalStatus zkmetal_fri_engine_create(ZkMetalFRIEngine* out);

/// Destroy a FRI engine.
void zkmetal_fri_engine_destroy(ZkMetalFRIEngine engine);

/// Create a BN254 pairing engine.
ZkMetalStatus zkmetal_pairing_engine_create(ZkMetalPairingEngine* out);

/// Destroy a pairing engine.
void zkmetal_pairing_engine_destroy(ZkMetalPairingEngine engine);

// ============================================================================
// MSM — Multi-Scalar Multiplication (BN254 G1)
// ============================================================================

/// Compute MSM: result = sum(scalars[i] * points[i]) for i in 0..n_points.
///
/// - engine:     MSM engine handle from zkmetal_msm_engine_create()
/// - points:     n_points affine points, each 64 bytes (x,y in Montgomery form, 8x u32 LE)
/// - scalars:    n_points scalars, each 32 bytes (8x u32 LE, standard form)
/// - n_points:   number of point-scalar pairs
/// - result_x:   output buffer for projective X coordinate (32 bytes, Montgomery form)
/// - result_y:   output buffer for projective Y coordinate (32 bytes, Montgomery form)
/// - result_z:   output buffer for projective Z coordinate (32 bytes, Montgomery form)
ZkMetalStatus zkmetal_bn254_msm(
    ZkMetalMSMEngine engine,
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
);

/// Convenience MSM using a lazy singleton engine (no engine handle needed).
ZkMetalStatus zkmetal_bn254_msm_auto(
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
);

// ============================================================================
// MSM — Multi-Scalar Multiplication (Pallas)
// ============================================================================
// Pallas: y^2 = x^3 + 5 over Fp where
//   p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// Field elements: 32 bytes (8x u32 limbs, little-endian Montgomery form).
// Affine points: 64 bytes (x: 32B, y: 32B). Scalars: 32 bytes (standard form).
// Results: projective (X, Y, Z), 32 bytes each, Montgomery form.

/// Pallas MSM (engine-based).
ZkMetalStatus zkmetal_pallas_msm(
    ZkMetalPallasMSMEngine engine,
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
);

/// Pallas MSM (convenience — lazy singleton).
ZkMetalStatus zkmetal_pallas_msm_auto(
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
);

// ============================================================================
// MSM — Multi-Scalar Multiplication (Vesta)
// ============================================================================
// Vesta: y^2 = x^3 + 5 over Fq where
//   q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// Same layout conventions as Pallas.

/// Vesta MSM (engine-based).
ZkMetalStatus zkmetal_vesta_msm(
    ZkMetalVestaMSMEngine engine,
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
);

/// Vesta MSM (convenience — lazy singleton).
ZkMetalStatus zkmetal_vesta_msm_auto(
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
);

// ============================================================================
// NTT — Number Theoretic Transform (BN254 Fr)
// ============================================================================

/// Forward NTT in-place. data is n field elements (n = 2^log_n), each 32 bytes.
/// Elements are in Montgomery form (8x u32 LE).
ZkMetalStatus zkmetal_bn254_ntt(
    ZkMetalNTTEngine engine,
    uint8_t* data,
    uint32_t log_n
);

/// Inverse NTT in-place. Same layout as forward NTT.
ZkMetalStatus zkmetal_bn254_intt(
    ZkMetalNTTEngine engine,
    uint8_t* data,
    uint32_t log_n
);

/// Convenience forward NTT using a lazy singleton engine.
ZkMetalStatus zkmetal_bn254_ntt_auto(
    uint8_t* data,
    uint32_t log_n
);

/// Convenience inverse NTT using a lazy singleton engine.
ZkMetalStatus zkmetal_bn254_intt_auto(
    uint8_t* data,
    uint32_t log_n
);

// ============================================================================
// Poseidon2 Hash (BN254 Fr)
// ============================================================================

/// Batch hash pairs of field elements.
/// input:  2*n_pairs field elements (each 32 bytes, Montgomery form)
/// output: n_pairs field elements (each 32 bytes, Montgomery form)
ZkMetalStatus zkmetal_bn254_poseidon2_hash_pairs(
    ZkMetalPoseidon2Engine engine,
    const uint8_t* input,
    uint32_t n_pairs,
    uint8_t* output
);

/// Convenience Poseidon2 batch hash using a lazy singleton engine.
ZkMetalStatus zkmetal_bn254_poseidon2_hash_pairs_auto(
    const uint8_t* input,
    uint32_t n_pairs,
    uint8_t* output
);

// ============================================================================
// Keccak-256 Hash (GPU)
// ============================================================================

/// Batch Keccak-256 hash of 64-byte inputs.
/// - input:    n_inputs * 64 bytes (each input is 64 bytes)
/// - n_inputs: number of 64-byte inputs to hash
/// - output:   n_inputs * 32 bytes (each output is a 32-byte hash)
ZkMetalStatus zkmetal_keccak256_hash(
    ZkMetalKeccakEngine engine,
    const uint8_t* input,
    uint32_t n_inputs,
    uint8_t* output
);

/// Convenience Keccak-256 batch hash using a lazy singleton engine.
ZkMetalStatus zkmetal_keccak256_hash_auto(
    const uint8_t* input,
    uint32_t n_inputs,
    uint8_t* output
);

// ============================================================================
// FRI Fold (BN254 Fr)
// ============================================================================

/// Single FRI fold round: fold n=2^log_n evaluations with challenge beta.
/// Output has n/2 elements.
///
/// - evals:   n field elements (each 32 bytes, Montgomery form)
/// - log_n:   log2 of the number of evaluation points (must be >= 1)
/// - beta:    fold challenge (32 bytes, Montgomery form)
/// - result:  output buffer for n/2 field elements (each 32 bytes)
ZkMetalStatus zkmetal_fri_fold(
    ZkMetalFRIEngine engine,
    const uint8_t* evals,
    uint32_t log_n,
    const uint8_t* beta,
    uint8_t* result
);

/// Convenience FRI fold using a lazy singleton engine.
ZkMetalStatus zkmetal_fri_fold_auto(
    const uint8_t* evals,
    uint32_t log_n,
    const uint8_t* beta,
    uint8_t* result
);

// ============================================================================
// Batch Pairing (BN254)
// ============================================================================

/// Batch pairing: compute product of e(g1[i], g2[i]) for i in 0..n_pairs.
/// GPU-accelerated parallel Miller loops + CPU final exponentiation.
///
/// - engine:     Pairing engine handle from zkmetal_pairing_engine_create()
/// - g1_points:  n_pairs affine G1 points, each 64 bytes (x,y in Montgomery form)
/// - g2_points:  n_pairs affine G2 points, each 128 bytes (x0,x1,y0,y1 Montgomery Fp2)
/// - n_pairs:    number of (G1, G2) pairs
/// - result:     output Fp12 element (384 bytes = 12 * 32B, Montgomery form)
ZkMetalStatus zkmetal_bn254_batch_pairing(
    ZkMetalPairingEngine engine,
    const uint8_t* g1_points,
    const uint8_t* g2_points,
    uint32_t n_pairs,
    uint8_t* result
);

/// Pairing check: verify product of e(g1[i], g2[i]) == 1 (Gt identity).
/// Returns ZKMETAL_SUCCESS if check passes, ZKMETAL_ERR_INVALID_INPUT if fails.
ZkMetalStatus zkmetal_bn254_pairing_check(
    ZkMetalPairingEngine engine,
    const uint8_t* g1_points,
    const uint8_t* g2_points,
    uint32_t n_pairs
);

/// Convenience batch pairing using a lazy singleton engine.
ZkMetalStatus zkmetal_bn254_batch_pairing_auto(
    const uint8_t* g1_points,
    const uint8_t* g2_points,
    uint32_t n_pairs,
    uint8_t* result
);

/// Convenience pairing check using a lazy singleton engine.
ZkMetalStatus zkmetal_bn254_pairing_check_auto(
    const uint8_t* g1_points,
    const uint8_t* g2_points,
    uint32_t n_pairs
);

// ============================================================================
// Small-Scalar MSM Variants (BN254 G1)
// ============================================================================

/// MSM with 1-byte scalars (u8). Each scalar is 1 byte, zero-extended internally.
/// More efficient than full 256-bit MSM for small scalar ranges.
ZkMetalStatus zkmetal_bn254_msm_u8(
    ZkMetalMSMEngine engine,
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x, uint8_t* result_y, uint8_t* result_z
);

/// MSM with 2-byte scalars (u16, little-endian).
ZkMetalStatus zkmetal_bn254_msm_u16(
    ZkMetalMSMEngine engine,
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x, uint8_t* result_y, uint8_t* result_z
);

/// MSM with 4-byte scalars (u32, little-endian).
ZkMetalStatus zkmetal_bn254_msm_u32(
    ZkMetalMSMEngine engine,
    const uint8_t* points,
    const uint8_t* scalars,
    uint32_t n_points,
    uint8_t* result_x, uint8_t* result_y, uint8_t* result_z
);

/// Convenience small-scalar MSM variants using a lazy singleton engine.
ZkMetalStatus zkmetal_bn254_msm_u8_auto(
    const uint8_t* points, const uint8_t* scalars, uint32_t n_points,
    uint8_t* result_x, uint8_t* result_y, uint8_t* result_z
);
ZkMetalStatus zkmetal_bn254_msm_u16_auto(
    const uint8_t* points, const uint8_t* scalars, uint32_t n_points,
    uint8_t* result_x, uint8_t* result_y, uint8_t* result_z
);
ZkMetalStatus zkmetal_bn254_msm_u32_auto(
    const uint8_t* points, const uint8_t* scalars, uint32_t n_points,
    uint8_t* result_x, uint8_t* result_y, uint8_t* result_z
);

// ============================================================================
// Batch MSM (BN254 G1)
// ============================================================================

/// Batch MSM: compute multiple independent MSMs in one call.
/// Amortizes Swift/GPU dispatch overhead for many small MSMs.
///
/// - engine:      MSM engine handle
/// - all_points:  concatenated affine points for all MSMs (64 bytes each)
/// - all_scalars: concatenated 32-byte scalars for all MSMs
/// - counts:      array of n_msms uint32_t values, counts[i] = number of points in MSM i
/// - n_msms:      number of MSMs to compute
/// - results:     n_msms * 96 bytes output (projective X,Y,Z per MSM, 32 bytes each)
ZkMetalStatus zkmetal_bn254_msm_batch(
    ZkMetalMSMEngine engine,
    const uint8_t* all_points,
    const uint8_t* all_scalars,
    const uint32_t* counts,
    uint32_t n_msms,
    uint8_t* results
);

/// Convenience batch MSM using a lazy singleton engine.
ZkMetalStatus zkmetal_bn254_msm_batch_auto(
    const uint8_t* all_points,
    const uint8_t* all_scalars,
    const uint32_t* counts,
    uint32_t n_msms,
    uint8_t* results
);

// ============================================================================
// Version / capability query
// ============================================================================

/// Set the path to the Shaders directory (must contain fields/, ntt/, etc.).
/// Call this before creating any engines if the binary is not run from the
/// zkMetal source tree. Pass NULL to clear.
void zkmetal_set_shader_dir(const char* path);

/// Returns 1 if a Metal GPU is available, 0 otherwise.
int32_t zkmetal_gpu_available(void);

/// Returns the library version as a null-terminated string.
const char* zkmetal_version(void);

#ifdef __cplusplus
}
#endif

#endif // ZKMETAL_H
