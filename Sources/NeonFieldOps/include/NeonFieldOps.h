#ifndef NEON_FIELD_OPS_H
#define NEON_FIELD_OPS_H

#include <stdint.h>

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

#endif // NEON_FIELD_OPS_H
