// ane_binary_tower.h — ANE-accelerated Binary Tower Fields (Binius)
//
// Binary Tower construction: GF(2) → GF(2^8) → GF(2^16) → GF(2^32) → GF(2^64) → GF(2^128)
//
// Tower extension: GF(2^{2k}) = GF(2^k)[X] / (X^2 + X + alpha_k)
// Addition = XOR at every level (free)
// Multiplication:
//   - GF(2^64): single PMULL instruction (ARM NEON carry-less multiply)
//   - GF(2^128): 3× GF(2^64) multiply + Karatsuba recombination
// S-box: x^3 (GF(2^8)), x^5 (GF(2^8) base for Binius)
//
// ANE (Apple Neural Engine) advantages:
//   - GF(2^8) multiply maps to 8-bit integer matmul on ANE tiles
//   - ANE can process thousands of GF(2^8) muls in parallel via FP16 matmul
//   - Batch operations across many tower elements for data parallelism
//
// This file provides the C API declarations and fallback scalar
// implementation for when ANE is not available.

#ifndef ANE_BINARY_TOWER_H
#define ANE_BINARY_TOWER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Opaque State Handle
// ============================================================

/// Opaque handle to ANE Binary Tower state.
typedef struct ANEBinaryTowerState opaque_ane_bt_state_t;

// ============================================================
// ANE Device Management
// ============================================================

/// Check if ANE is available on this device.
/// @return true if ANE is available, false otherwise.
bool ane_bt_available(void);

/// Create ANE Binary Tower state.
/// @param logN Log2 of the tower size (supported values TBD).
/// @return Opaque state handle, or NULL on failure (no ANE available).
void* ane_bt_create(int logN);

/// Destroy ANE Binary Tower state.
/// @param state State handle from ane_bt_create.
void ane_bt_destroy(void* state);

// ============================================================
// GF(2^64) Operations
// ============================================================

/// GF(2^64) multiply using ANE acceleration.
/// @param a First operand (64-bit).
/// @param b Second operand (64-bit).
/// @return Product a * b in GF(2^64).
uint64_t ane_bt_gf64_mul(uint64_t a, uint64_t b);

/// GF(2^64) add (XOR) using ANE acceleration.
/// @param a First operand.
/// @param b Second operand.
/// @return Sum a ^ b in GF(2^64).
uint64_t ane_bt_gf64_add(uint64_t a, uint64_t b);

// ============================================================
// GF(2^128) Operations
// ============================================================

/// GF(2^128) multiply using ANE acceleration.
/// @param a First operand (128-bit, two uint64_t).
/// @param b Second operand (128-bit, two uint64_t).
/// @param r Result buffer (128-bit, two uint64_t).
void ane_bt_gf128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);

// ============================================================
// GF(2^8) Operations (for Binius base layer)
// ============================================================

/// GF(2^8) multiply using log/exp tables.
/// Uses AES polynomial x^8 + x^4 + x^3 + x + 1 (0x11B).
/// @param a First operand (8-bit).
/// @param b Second operand (8-bit).
/// @return Product a * b in GF(2^8).
uint8_t ane_bt_gf8_mul(uint8_t a, uint8_t b);

/// GF(2^8) add (XOR).
/// @param a First operand.
/// @param b Second operand.
/// @return Sum a ^ b in GF(2^8).
uint8_t ane_bt_gf8_add(uint8_t a, uint8_t b);

/// Batch GF(2^8) multiply for multiple pairs.
/// @param a Array of n uint8_t operands.
/// @param b Array of n uint8_t operands.
/// @param r Result array of n uint8_t products.
/// @param n Number of elements in each array.
/// @return 0 on success, -1 on error.
int ane_bt_batch_gf8_mul(const uint8_t* a, const uint8_t* b, uint8_t* r, int n);

// ============================================================
// Batch Operations
// ============================================================

/// Batch GF(2^64) multiply for multiple pairs.
/// @param state ANE state from ane_bt_create.
/// @param a Array of n uint64_t operands.
/// @param b Array of n uint64_t operands.
/// @param r Result array of n uint64_t products (same size as a, b).
/// @param n Number of elements in each array.
/// @return 0 on success, -1 on error.
int ane_bt_batch_gf64_mul(void* state, const uint64_t* a, const uint64_t* b, uint64_t* r, int n);

// ============================================================
// Scalar Fallback Helpers (available even without ANE)
// ============================================================

/// GF(2^64) multiply using scalar PMULL equivalent.
/// Uses ARM NEON vmull_p64 for carry-less multiply.
/// @param a First operand.
/// @param b Second operand.
/// @return Product a * b in GF(2^64).
uint64_t bt_gf64_mul_scalar(uint64_t a, uint64_t b);

/// GF(2^128) multiply using 3× GF(2^64) + Karatsuba.
/// @param a First operand (128-bit, two uint64_t).
/// @param b Second operand (128-bit, two uint64_t).
/// @param r Result buffer (128-bit, two uint64_t).
void bt_gf128_mul_scalar(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]);

/// GF(2^64) add (XOR).
/// @param a First operand.
/// @param b Second operand.
/// @return Sum a ^ b in GF(2^64).
uint64_t bt_gf64_add_scalar(uint64_t a, uint64_t b);

#ifdef __cplusplus
}
#endif

#endif // ANE_BINARY_TOWER_H
