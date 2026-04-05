// MetalLibLoader.h — C API for loading a precompiled zkMetal .metallib
//
// FFI consumers (Rust, Go, Python) call these functions to initialize the
// Metal GPU and run compute kernels from a pre-built .metallib, without
// needing Xcode or runtime shader compilation.
//
// Thread safety: zkmetal_gpu_init() must be called once before other functions.
// After initialization, compute functions are thread-safe (each call creates
// its own command buffer).

#ifndef ZKMETAL_METALLIB_LOADER_H
#define ZKMETAL_METALLIB_LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Status codes
// ============================================================================

typedef int32_t ZkGpuStatus;

#define ZK_GPU_SUCCESS            0
#define ZK_GPU_ERR_NO_DEVICE     -1
#define ZK_GPU_ERR_METALLIB      -2  // Failed to load .metallib
#define ZK_GPU_ERR_KERNEL        -3  // Kernel function not found
#define ZK_GPU_ERR_BUFFER        -4  // Buffer allocation failed
#define ZK_GPU_ERR_DISPATCH      -5  // Command buffer error
#define ZK_GPU_ERR_NOT_INIT      -6  // zkmetal_gpu_init() not called
#define ZK_GPU_ERR_INVALID_INPUT -7

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the Metal GPU context from a precompiled .metallib file.
/// Must be called once before any compute functions.
///
/// @param metallib_path  Path to zkmetal.metallib (absolute or relative)
/// @return ZK_GPU_SUCCESS or an error code
ZkGpuStatus zkmetal_gpu_init(const char* metallib_path);

/// Release all GPU resources. Safe to call multiple times.
void zkmetal_gpu_shutdown(void);

/// Check if the GPU context is initialized and ready.
/// @return 1 if ready, 0 if not
int32_t zkmetal_gpu_is_ready(void);

// ============================================================================
// Device info
// ============================================================================

/// Get the GPU device name (e.g., "Apple M3 Pro").
/// Returns NULL if not initialized. The returned string is valid until shutdown.
const char* zkmetal_gpu_device_name(void);

/// Get the number of kernel functions available in the loaded .metallib.
uint32_t zkmetal_gpu_kernel_count(void);

/// Get the name of a kernel function by index (0-based).
/// Returns NULL if index is out of range or not initialized.
const char* zkmetal_gpu_kernel_name(uint32_t index);

// ============================================================================
// BN254 MSM — Multi-Scalar Multiplication
// ============================================================================

/// GPU-accelerated BN254 G1 MSM using the precompiled metallib.
///
/// @param scalars    n_points * 32 bytes (256-bit scalars, little-endian)
/// @param points     n_points * 64 bytes (affine points: x,y each 32B Montgomery)
/// @param n_points   Number of scalar-point pairs
/// @param result_x   Output: 32 bytes (projective X, Montgomery form)
/// @param result_y   Output: 32 bytes (projective Y, Montgomery form)
/// @param result_z   Output: 32 bytes (projective Z, Montgomery form)
ZkGpuStatus zkmetal_gpu_msm_bn254(
    const uint8_t* scalars,
    const uint8_t* points,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
);

// ============================================================================
// BN254 NTT — Number Theoretic Transform
// ============================================================================

/// In-place forward NTT on BN254 Fr field elements.
///
/// @param data    2^log_n field elements, each 32 bytes (Montgomery form)
/// @param log_n   log2 of the number of elements (must be >= 1)
ZkGpuStatus zkmetal_gpu_ntt_bn254(
    uint8_t* data,
    uint32_t log_n
);

/// In-place inverse NTT on BN254 Fr field elements.
ZkGpuStatus zkmetal_gpu_intt_bn254(
    uint8_t* data,
    uint32_t log_n
);

// ============================================================================
// Poseidon2 Hash
// ============================================================================

/// Batch Poseidon2 hash of pairs of BN254 Fr elements.
///
/// @param input    2 * n_pairs field elements (each 32B, Montgomery form)
/// @param n_pairs  Number of pairs to hash
/// @param output   n_pairs field elements (each 32B, Montgomery form)
ZkGpuStatus zkmetal_gpu_poseidon2_hash_pairs(
    const uint8_t* input,
    uint32_t n_pairs,
    uint8_t* output
);

// ============================================================================
// Keccak-256 Hash
// ============================================================================

/// Batch Keccak-256 of 64-byte inputs.
///
/// @param input     n_inputs * 64 bytes
/// @param n_inputs  Number of 64-byte inputs
/// @param output    n_inputs * 32 bytes (32-byte digests)
ZkGpuStatus zkmetal_gpu_keccak256(
    const uint8_t* input,
    uint32_t n_inputs,
    uint8_t* output
);

// ============================================================================
// Low-level: run any kernel by name
// ============================================================================

/// Create a compute pipeline for a named kernel function.
/// Returns an opaque handle, or NULL on failure.
void* zkmetal_gpu_create_pipeline(const char* kernel_name);

/// Destroy a pipeline handle.
void zkmetal_gpu_destroy_pipeline(void* pipeline);

#ifdef __cplusplus
}
#endif

#endif // ZKMETAL_METALLIB_LOADER_H
