// ANEOps Poseidon2 S-box acceleration
// BabyBear: x^7 S-box, width=16, 8 full + 13 partial = 21 rounds
// M31: x^5 S-box, width=16, 14 full + 21 partial = 35 rounds
//
// ANE approach: leverage FP16 GEMM units for diagonal matmul patterns
// x^7 = x^3 * x^4 via 4 ANE matmuls
// x^5 = x * x^4 via 3 ANE matmuls

#ifndef ANE_POSEDON2_H
#define ANE_POSEDON2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// ANE Poseidon2 Lifecycle
// ============================================================

/// Initialize ANE Poseidon2 GPU support
/// Must be called before any GPU-accelerated functions.
/// Returns 0 on success, -1 on failure.
int ane_poseidon2_init(void);

/// Shutdown ANE Poseidon2 GPU support and release resources
void ane_poseidon2_shutdown(void);

/// Check if ANE Poseidon2 GPU acceleration is available
/// Returns true if GPU path can be used, false otherwise.
bool ane_poseidon2_gpu_available(void);

/// Debug: get pipeline status bitmask
int ane_poseidon2_debug_pipeline_status(void);

// ============================================================
// BabyBear Poseidon2 ANE S-box
// ============================================================

/// BabyBear Poseidon2 S-box via ANE matmul
/// Computes x^7 for 16 elements using 4 ANE matmul operations:
///   x^2 = diag(x) * x      (1 matmul)
///   x^4 = diag(x^2) * x^2  (1 matmul)
///   x^3 = diag(x) * x^2    (1 matmul)
///   x^7 = diag(x^3) * x^4  (1 matmul)
/// @param state 16 uint32_t BabyBear elements in [0, p), modified in-place.
void bb_poseidon2_sbox_ane(uint32_t state[16]);

/// BabyBear Poseidon2 full permutation via ANE S-box + Metal matrix layers
/// Width=16, x^7 S-box, 8 full + 13 partial rounds = 21 total
/// @param state 16 uint32_t in Montgomery form, modified in-place.
/// @param round_constants (8*16 + 13) = 141 uint32_t constants in Montgomery form.
/// @param internal_diag 16 diagonal constants for internal matrix (Montgomery form).
void bb_poseidon2_permutation_ane(uint32_t state[16],
                                   const uint32_t *round_constants,
                                   const uint32_t internal_diag[16]);

// ============================================================
// M31 Poseidon2 ANE S-box
// ============================================================

/// M31 Poseidon2 S-box via ANE matmul
/// Computes x^5 for 16 elements using 3 ANE matmul operations:
///   x^2 = diag(x) * x      (1 matmul)
///   x^4 = diag(x^2) * x^2  (1 matmul)
///   x^5 = diag(x) * x^4    (1 matmul)
/// @param state 16 uint32_t M31 elements in [0, p), modified in-place.
void m31_poseidon2_sbox_ane(uint32_t state[16]);

/// M31 Poseidon2 full permutation via ANE S-box + Metal matrix layers
/// Width=16, x^5 S-box, 14 full + 21 partial rounds = 35 total
/// @param state 16 uint32_t in Montgomery form, modified in-place.
/// @param round_constants (14*16 + 21) = 245 uint32_t constants in [0, p).
/// @param internal_diag 16 diagonal constants for internal matrix.
void m31_poseidon2_permutation_ane(uint32_t state[16],
                                    const uint32_t *round_constants,
                                    const uint32_t internal_diag[16]);

// ============================================================
// Batch Poseidon2 (ANE-accelerated)
// ============================================================

/// Batch BabyBear Poseidon2 S-box via ANE
/// @param states Batch of n * 16 uint32_t elements (n permutations).
/// @param n Number of independent permutations.
/// @param output Output buffer for n * 16 uint32_t results.
void bb_poseidon2_sbox_batch_ane(const uint32_t *states, int n, uint32_t *output);

/// Batch M31 Poseidon2 S-box via ANE
/// @param states Batch of n * 16 uint32_t elements (n permutations).
/// @param n Number of independent permutations.
/// @param output Output buffer for n * 16 uint32_t results.
void m31_poseidon2_sbox_batch_ane(const uint32_t *states, int n, uint32_t *output);

// ============================================================
// Batched Full Permutations (NEW - processes N perms in one GPU call)
// ============================================================

/// Batch BabyBear Poseidon2 full permutation via ANE
/// Processes n_perms complete Poseidon2 permutations (21 rounds each) in a single GPU dispatch.
/// This is the preferred API for bulk hashing as it minimizes per-call overhead.
///
/// @param states Input: n_perms * 16 uint32_t elements (each 16-element block is one permutation state)
/// @param n_perms Number of permutations to process
/// @param round_constants 21 * 16 = 336 uint32_t round constants in Montgomery form
/// @param internal_diag 16 uint32_t internal diagonal constants in Montgomery form
/// @param output Output: n_perms * 16 uint32_t results
void bb_poseidon2_permutation_batch_ane(const uint32_t *states, int n_perms,
                                         const uint32_t *round_constants,
                                         const uint32_t *internal_diag,
                                         uint32_t *output);

/// Batch M31 Poseidon2 full permutation via ANE
/// Processes n_perms complete Poseidon2 permutations (35 rounds each) in a single GPU dispatch.
///
/// @param states Input: n_perms * 16 uint32_t elements
/// @param n_perms Number of permutations to process
/// @param round_constants 35 * 16 = 560 uint32_t round constants in [0, p)
/// @param output Output: n_perms * 16 uint32_t results
void m31_poseidon2_permutation_batch_ane(const uint32_t *states, int n_perms,
                                          const uint32_t *round_constants,
                                          uint32_t *output);

#ifdef __cplusplus
}
#endif

#endif // ANE_POSEDON2_H
