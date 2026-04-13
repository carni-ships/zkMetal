// ane_lattice.h — ANE-accelerated Kyber NTT
//
// Kyber-768: q = 3329, n = 256, int16 coefficients
//
// ANE (Apple Neural Engine) is programmed via Metal compute shaders.
// This implementation uses Metal as substrate to target ANE hardware.
//
// Montgomery multiplication (instead of Barrett) for ANE-friendly computation:
//   - Barrett reduction: q = 3329, m = 5039, s = 24
//     t = (x * 5039) >> 24, r = x - t * 3329
//     Requires 64-bit multiply which ANE can't do directly
//   - Montgomery multiplication: a * b * R^{-1} mod p
//     R = 2^16 = 65536 (fits in 16-bit, matches ANE FP16)
//     CiOS algorithm maps naturally to ANE matmul
//     R mod p = 2184, p_inv = 3361
//
// Forward NTT:  Cooley-Tukey radix-2 DIT (Decimation-In-Time)
// Inverse NTT:  Gentleman-Sande radix-2 DIF + 1/N scaling
//
// Batch-64: 64 polynomials × 256 elements = 16384 coefficients per ANE dispatch
// ANE advantage: FP16 matmul for Montgomery multiply >> NEON Barrett for large batches

#ifndef ANE_LATTICE_H
#define ANE_LATTICE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Opaque State Handle
// ============================================================

/// Opaque handle to ANE Kyber NTT state.
typedef struct ANEKyberNTTState opaque_ane_kyber_ntt_state_t;

// ============================================================
// ANE Device Management
// ============================================================

/// Create ANE Kyber NTT state.
/// @param logN Log2 of transform size (supported: 8 for n=256).
/// @return Opaque state handle, or NULL on failure (no ANE available).
void* ane_kyber_ntt_create(int logN);

/// Destroy ANE Kyber NTT state.
void ane_kyber_ntt_destroy(void* state);

/// Check if ANE is available on this device.
bool ane_kyber_ntt_available(void);

// ============================================================
// Single-polynomial NTT
// ============================================================

/// Forward NTT for Kyber (q=3329, n=256) using ANE acceleration.
/// @param state  ANE state from ane_kyber_ntt_create.
/// @param data   Array of 256 uint16_t elements in [0, 3329).
///               Modified in-place.
/// @return 0 on success, -1 on error.
int ane_kyber_ntt(void* state, uint16_t* data, int logN);

/// Inverse NTT for Kyber using ANE acceleration.
/// @param state  ANE state from ane_kyber_ntt_create.
/// @param data   Array of 256 uint16_t elements in [0, 3329).
///               Modified in-place.
/// @return 0 on success, -1 on error.
int ane_kyber_intt(void* state, uint16_t* data, int logN);

// ============================================================
// Batch-64 NTT (ANE-accelerated, processes 64 polynomials at once)
// ============================================================

/// Forward NTT for 64 Kyber polynomials via ANE.
/// @param state     ANE state from ane_kyber_ntt_create.
/// @param polys     Flat array of 64*256 = 16384 uint16_t elements in [0, 3329).
///                  Layout: polys[polyIdx*256 + coeffIdx].
///                  Modified in-place.
/// @return 0 on success, -1 on error.
int ane_kyber_ntt_batch64(void* state, uint16_t* polys);

/// Inverse NTT for 64 Kyber polynomials via ANE.
/// @param state     ANE state from ane_kyber_ntt_create.
/// @param polys     Flat array of 64*256 = 16384 uint16_t elements in [0, 3329).
///                  Modified in-place.
/// @return 0 on success, -1 on error.
int ane_kyber_intt_batch64(void* state, uint16_t* polys);

// ============================================================
// High-level API (allocate → compute → return)
// ============================================================

/// Forward NTT for single polynomial: allocate, compute, free.
/// @param data  Array of 256 uint16_t elements in [0, 3329).
///             Modified in-place.
/// @return 0 on success, -1 on ANE error.
int ane_kyber_ntt_forward(uint16_t* data);

/// Forward NTT for batch-64: allocate, compute, free.
/// @param polys  Flat array of 64*256 uint16_t elements in [0, 3329).
///              Modified in-place.
/// @return 0 on success, -1 on ANE error.
int ane_kyber_ntt_forward_batch64(uint16_t* polys);

/// Inverse NTT for single polynomial: allocate, compute, free.
/// @param data  Array of 256 uint16_t elements in [0, 3329).
///             Modified in-place.
/// @return 0 on success, -1 on ANE error.
int ane_kyber_ntt_inverse(uint16_t* data);

/// Inverse NTT for batch-64: allocate, compute, free.
/// @param polys  Flat array of 64*256 uint16_t elements in [0, 3329).
///              Modified in-place.
/// @return 0 on success, -1 on ANE error.
int ane_kyber_ntt_inverse_batch64(uint16_t* polys);

// ============================================================
// Montgomery Arithmetic Helpers (for C/NEON fallback path)
// ============================================================

/// R mod p for Kyber Montgomery form (R = 2^16).
/// Returns 65536 mod 3329 = 2184.
uint16_t kyber_mont_r_mod_p(void);

/// p_inv = -(p^{-1}) mod R for Kyber Montgomery form.
/// Returns -(3329^{-1}) mod 65536 = 3361.
uint16_t kyber_mont_p_inv(void);

/// Compute (a * b) * R^{-1} mod p using Montgomery multiplication.
/// CiOS Montgomery multiplication algorithm.
/// @param a     Operand in [0, p).
/// @param b     Operand in [0, p).
/// @param p_inv Precomputed p_inv = -(p^{-1}) mod R.
/// @return (a * b * R^{-1}) mod p, in [0, p).
uint16_t kyber_mont_mul(uint16_t a, uint16_t b, uint16_t p_inv);

/// Convert a value to Montgomery form: a * R mod p.
/// @param a        Input in [0, p).
/// @param r_mod_p  Precomputed R mod p = 2184.
/// @param p_inv    Precomputed p_inv = 3361.
/// @return a * R mod p, in [0, p).
uint16_t kyber_to_mont(uint16_t a, uint16_t r_mod_p, uint16_t p_inv);

/// Convert from Montgomery form: a * R^{-1} mod p.
/// @param a     Montgomery form input in [0, p).
/// @param p_inv Precomputed p_inv = 3361.
/// @return a * R^{-1} mod p, in [0, p).
uint16_t kyber_from_mont(uint16_t a, uint16_t p_inv);

/// 128^{-1} mod 3329 = 3073 (for final scaling in INTT).
uint16_t kyber_inv128(void);

#ifdef __cplusplus
}
#endif

#endif // ANE_LATTICE_H
