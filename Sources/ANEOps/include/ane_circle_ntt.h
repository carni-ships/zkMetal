// ane_circle_ntt.h — ANE-accelerated Circle NTT for Mersenne31
// p = 2^31 - 1 = 0x7FFFFFFF
//
// Circle NTT differs from standard NTT:
// - Layer 0 uses y-coordinate twiddles (twin-coset decomposition)
// - Layers 1+ use x-coordinate twiddles with the squaring map
//
// ANE (Apple Neural Engine) is programmed via Metal compute shaders.
// Standard Metal compute automatically offloads to ANE on Apple Silicon.

#ifndef ANE_CIRCLE_NTT_H
#define ANE_CIRCLE_NTT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to ANE Circle NTT state.
typedef struct ANECircleNTTState opaque_ane_circle_ntt_state_t;

/// Initialize ANE Circle NTT GPU support.
/// Must be called before any GPU-accelerated functions.
/// Returns 0 on success, -1 on failure.
int ane_circle_ntt_init(void);

/// Shutdown ANE Circle NTT GPU support and release resources.
void ane_circle_ntt_shutdown(void);

/// Check if ANE Circle NTT GPU acceleration is available.
/// Returns true if GPU path can be used, false otherwise.
bool ane_circle_ntt_gpu_available(void);

/// Create ANE Circle NTT state for the given transform size.
/// @param logN Log2 of transform size (supported: 8 for N=256).
/// @return Opaque state handle, or NULL on failure.
void* ane_circle_ntt_create(int logN);

/// Destroy ANE Circle NTT state.
void ane_circle_ntt_destroy(void* state);

/// Forward Circle NTT (DIT) on Mersenne31 field using ANE acceleration.
/// @param state  ANE state from ane_circle_ntt_create.
/// @param data   Array of n = 2^logN uint32_t elements in [0, p).
///                Modified in-place.
/// @param twiddles Twiddle factors (precomputed externally).
/// @param logN  Log2 of transform size.
/// @return 0 on success, -1 on error.
int ane_circle_ntt(void* state, uint32_t* data, const uint32_t* twiddles, int logN);

/// Inverse Circle NTT (DIF) on Mersenne31 field using ANE acceleration.
/// @param state  ANE state from ane_circle_ntt_create.
/// @param data   Array of n = 2^logN uint32_t elements in [0, p).
///                Modified in-place.
/// @param inv_twiddles Inverse twiddle factors (precomputed externally).
/// @param inv_n Inverse of n modulo p.
/// @param logN  Log2 of transform size.
/// @return 0 on success, -1 on error.
int ane_circle_intt(void* state, uint32_t* data, const uint32_t* inv_twiddles,
                   uint32_t inv_n, int logN);

/// High-level forward Circle NTT: compute in-place.
/// @param data  Array of n = 2^logN uint32_t elements in [0, p).
///              Modified in-place.
/// @param twiddles Twiddle factors (logN * n/2 elements).
/// @param logN  Log2 of transform size.
/// @return 0 on success, -1 on ANE error.
int ane_circle_ntt_forward(uint32_t* data, const uint32_t* twiddles, int logN);

/// High-level inverse Circle NTT: compute in-place.
/// @param data  Array of n = 2^logN uint32_t elements in [0, p).
///              Modified in-place.
/// @param inv_twiddles Inverse twiddle factors.
/// @param inv_n Inverse of n modulo p.
/// @param logN  Log2 of transform size.
/// @return 0 on success, -1 on ANE error.
int ane_circle_ntt_inverse(uint32_t* data, const uint32_t* inv_twiddles,
                          uint32_t inv_n, int logN);

#ifdef __cplusplus
}
#endif

#endif // ANE_CIRCLE_NTT_H