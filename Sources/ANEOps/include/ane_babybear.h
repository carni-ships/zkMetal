// ane_babybear.h — ANE-accelerated BabyBear NTT
// p = 0x78000001 = 2^31 - 2^27 + 1 (31-bit prime)
//
// ANE (Apple Neural Engine) is programmed via Metal compute shaders.
// This implementation uses Metal as substrate to target ANE hardware.
//
// Forward NTT:  Cooley-Tukey radix-2 DIT (bit-reversal + butterfly stages)
// Inverse NTT:  Gentleman-Sande radix-2 DIF (butterfly stages + bit-reversal + 1/N scale)
//
// Start with N=256 (logN=8) as a working baseline.
// Threadgroup-based fused kernels minimize global memory traffic.

#ifndef ANE_BABYBEAR_H
#define ANE_BABYBEAR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to ANE BabyBear NTT state.
typedef struct ANEBabyBearNTTState opaque_ane_bb_ntt_state_t;

/// Create ANE BabyBear NTT state for the given transform size.
/// @param logN Log2 of transform size (supported: 8 for N=256 to start).
/// @return Opaque state handle, or NULL on failure (no ANE available).
void* ane_babybear_ntt_create(int logN);

/// Destroy ANE BabyBear NTT state.
void ane_babybear_ntt_destroy(void* state);

/// Check if ANE is available on this device.
bool ane_babybear_ntt_available(void);

/// Forward NTT on BabyBear field using ANE acceleration.
/// @param state  ANE state from ane_babybear_ntt_create.
/// @param data   Array of n = 2^logN uint32_t elements in [0, p).
///                Modified in-place.
/// @return 0 on success, -1 on error.
int ane_babybear_ntt(void* state, uint32_t* data, int logN);

/// Inverse NTT on BabyBear field using ANE acceleration.
/// @param state  ANE state from ane_babybear_ntt_create.
/// @param data   Array of n = 2^logN uint32_t elements in [0, p).
///                Modified in-place.
/// @return 0 on success, -1 on error.
int ane_babybear_intt(void* state, uint32_t* data, int logN);

/// High-level forward NTT: allocate, compute, free.
/// @param data  Array of n = 2^logN uint32_t elements in [0, p).
///              Modified in-place.
/// @param logN  Log2 of transform size.
/// @return 0 on success, -1 on ANE error.
int ane_babybear_ntt_forward(uint32_t* data, int logN);

/// High-level inverse NTT: allocate, compute, free.
/// @param data  Array of n = 2^logN uint32_t elements in [0, p).
///              Modified in-place.
/// @param logN  Log2 of transform size.
/// @return 0 on success, -1 on ANE error.
int ane_babybear_ntt_inverse(uint32_t* data, int logN);

#ifdef __cplusplus
}
#endif

#endif // ANE_BABYBEAR_H
