// zkmetal.h — C FFI for zkMetal GPU-accelerated ZK primitives
// Apple Silicon only (Metal GPU backend)
//
// All field elements are 32 bytes (8x u32 limbs, little-endian Montgomery form).
// Points are affine: 64 bytes (x: 32B, y: 32B) in Montgomery form.
// Scalars for MSM are 32 bytes (8x u32 limbs, standard form, NOT Montgomery).

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
typedef void* ZkMetalNTTEngine;
typedef void* ZkMetalPoseidon2Engine;

// ============================================================================
// Engine lifecycle
// ============================================================================

/// Create a BN254 MSM engine. Returns ZKMETAL_SUCCESS or error code.
/// The engine handle is written to *out.
ZkMetalStatus zkmetal_msm_engine_create(ZkMetalMSMEngine* out);

/// Destroy an MSM engine, releasing GPU resources.
void zkmetal_msm_engine_destroy(ZkMetalMSMEngine engine);

/// Create a BN254 NTT engine.
ZkMetalStatus zkmetal_ntt_engine_create(ZkMetalNTTEngine* out);

/// Destroy an NTT engine.
void zkmetal_ntt_engine_destroy(ZkMetalNTTEngine engine);

/// Create a BN254 Poseidon2 engine.
ZkMetalStatus zkmetal_poseidon2_engine_create(ZkMetalPoseidon2Engine* out);

/// Destroy a Poseidon2 engine.
void zkmetal_poseidon2_engine_destroy(ZkMetalPoseidon2Engine engine);

// ============================================================================
// MSM — Multi-Scalar Multiplication (BN254 G1)
// ============================================================================

/// Compute MSM: result = sum(scalars[i] * points[i]) for i in 0..n_points.
///
/// - points:     n_points affine points, each 64 bytes (x,y in Montgomery form, 8x u32 LE)
/// - scalars:    n_points scalars, each 32 bytes (8x u32 LE, standard form)
/// - n_points:   number of point-scalar pairs
/// - result_x:   output buffer for projective X coordinate (32 bytes, Montgomery form)
/// - result_y:   output buffer for projective Y coordinate (32 bytes, Montgomery form)
/// - result_z:   output buffer for projective Z coordinate (32 bytes, Montgomery form)
///
/// Returns ZKMETAL_SUCCESS or error code.
ZkMetalStatus zkmetal_bn254_msm(
    ZkMetalMSMEngine engine,
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
