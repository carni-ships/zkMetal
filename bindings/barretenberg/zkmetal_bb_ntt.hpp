/*
 * zkmetal_bb_ntt.hpp -- Drop-in NTT replacement for Barretenberg
 *
 * Replaces BB's polynomial_arithmetic::fft with zkMetal's NEON-accelerated
 * NTT for BN254 Fr.
 *
 * Usage:
 *   #include "zkmetal_bb_ntt.hpp"
 *   // In BB's FFT function:
 *   if (zkmetal::ntt::try_gpu_fft(coeffs, domain_size, inverse))
 *       return;
 *   // else fall through to CPU
 *
 * BB's FFT interface:
 *   - polynomial_arithmetic::fft(fr* coeffs, EvaluationDomain& domain)
 *   - polynomial_arithmetic::ifft(fr* coeffs, EvaluationDomain& domain)
 *   - domain.size is a power of 2
 *   - coeffs are in Montgomery form (same as zkMetal)
 */

#pragma once

#ifdef HAS_ZKMETAL

#include "zkmetal_bb_bridge.h"
#include <cstddef>
#include <cstdint>

namespace zkmetal {
namespace ntt {

/// Minimum domain size to use zkMetal NTT.
/// Below this, BB's CPU radix-4 NTT is competitive.
constexpr size_t NTT_THRESHOLD = 1 << 12;  // 4096

/**
 * try_fft -- Attempt accelerated forward NTT.
 *
 * @param data  Array of n Fr elements (4 uint64_t each), in-place.
 * @param n     Domain size (must be power of 2).
 * @return true if handled, false for CPU fallback.
 */
inline bool try_fft(uint64_t* data, size_t n)
{
    if (n < NTT_THRESHOLD) {
        return false;
    }
    zkmetal::ntt_forward(data, n);
    return true;
}

/**
 * try_ifft -- Attempt accelerated inverse NTT.
 *
 * Includes 1/n scaling (same as BB's ifft convention).
 *
 * @param data  Array of n Fr elements (4 uint64_t each), in-place.
 * @param n     Domain size (must be power of 2).
 * @return true if handled, false for CPU fallback.
 */
inline bool try_ifft(uint64_t* data, size_t n)
{
    if (n < NTT_THRESHOLD) {
        return false;
    }
    zkmetal::ntt_inverse(data, n);
    return true;
}

/**
 * try_coset_fft -- Attempt coset FFT (multiply by coset generator, then FFT).
 *
 * BB performs coset FFT by multiplying each coefficient by g^i before FFT,
 * where g is a multiplicative generator. This cannot be fused into zkMetal's
 * NTT directly, so we do the multiply + NTT.
 *
 * @param data         Array of n Fr elements.
 * @param n            Domain size.
 * @param coset_gen    Coset generator (Montgomery form, 4 uint64_t).
 * @return true if handled.
 */
inline bool try_coset_fft(uint64_t* data, size_t n, const uint64_t coset_gen[4])
{
    if (n < NTT_THRESHOLD) {
        return false;
    }

    // Multiply coeffs[i] by coset_gen^i using batch scalar mul with running product.
    // Build powers: pow[0]=1, pow[i]=pow[i-1]*g
    // Then pointwise multiply data[i] *= pow[i].
    //
    // For large n this is a single pass over memory, O(n) field muls.
    // We use zkMetal's batch operations for the pointwise multiply.

    // Accumulate coset powers and multiply in a fused loop
    uint64_t power[4] = {0, 0, 0, 0};
    // Montgomery form of 1 for BN254 Fr:
    // R mod p = 0x0e0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb
    // But we can compute it: 1 * R mod p. For correctness, use fr_pow(R_mont, 0)
    // or just set to the known R value. Since we're bridging, use the identity:
    // mont(1) can be obtained by squaring then taking sqrt, but simpler to just
    // pass through. BB will provide the coset_gen already in Montgomery form.

    // Actually, the simplest approach: just do the coset multiply on BB's side
    // since BB already has the coset generator powers cached, then call our NTT.
    // This function is a convenience -- the main win is the NTT itself.

    // For now, return false to let BB handle the coset multiply + we just do NTT.
    // The caller can split: BB does coset multiply, then calls try_fft().
    (void)coset_gen;
    return false;
}

} // namespace ntt
} // namespace zkmetal

#endif // HAS_ZKMETAL
