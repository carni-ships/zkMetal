// ane_mimc.h — ANE-accelerated MiMC hash for BN254
//
// MiMC is a block cipher / hash function using the x^7 S-box (for BN254,
// where gcd(7, p-1) = 1).
//
// Field: BN254 Fr (~254-bit prime, ~36 bytes)
// S-box: x^7 via 3 multiplies: x^2, x^4, x^7 = x * x^2 * x^4
// Rounds: 91 (full rounds, no partial)
// Width: 1 element (single field element per round)
// Mode: Miyaguchi-Preneel: h = Enc(h,m) + m + h
//
// ANE approach: Single element per round, so ANE GEMM is less directly
// applicable than for width-16 Poseidon2. The x^7 decomposition can
// potentially be accelerated via FP16 operations if field elements
// can be safely represented.

#ifndef ANE_MIMC_H
#define ANE_MIMC_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to ANE MiMC state.
typedef struct ANEMiMCState opaque_ane_mimc_state_t;

/// Check if ANE MiMC acceleration is available on this device.
/// @return true if ANE is available, false otherwise.
bool ane_mimc_available(void);

/// Create ANE MiMC state for the given number of rounds.
/// @param rounds Number of MiMC rounds (default: 91).
/// @return Opaque state handle, or NULL on failure.
void* ane_mimc_create(int rounds);

/// Destroy ANE MiMC state.
void ane_mimc_destroy(void* state);

/// Compute MiMC hash of a single field element using ANE acceleration.
/// @param state  ANE state from ane_mimc_create.
/// @param input  Single BN254 Fr element in Montgomery form.
/// @param key    Round key (BN254 Fr in Montgomery form).
/// @param output Output buffer for the hash result.
/// @return 0 on success, -1 on error.
int ane_mimc_hash(void* state, const uint8_t* input, const uint8_t* key, uint8_t* output);

/// Compute MiMC hash of a batch of field elements using ANE acceleration.
/// @param state   ANE state from ane_mimc_create.
/// @param inputs  Array of n BN254 Fr elements in Montgomery form.
/// @param keys    Array of n round keys (NULL to use state keys).
/// @param n       Number of elements to hash.
/// @param outputs Output buffer for n hash results.
/// @return 0 on success, -1 on error.
int ane_mimc_batch_hash(void* state, const uint8_t* inputs, const uint8_t* keys,
                        int n, uint8_t* outputs);

#ifdef __cplusplus
}
#endif

#endif // ANE_MIMC_H
