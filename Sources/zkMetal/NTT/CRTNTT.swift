// CRTNTT.swift — CRT-based NTT for BN254 Fr: Feasibility Analysis
//
// VERDICT: NOT FEASIBLE — CRT overhead eliminates the compute savings.
//
// The idea: decompose 256-bit BN254 Fr elements into residues modulo k small
// (31-bit) NTT-friendly primes via CRT, run k independent single-word NTTs,
// then reconstruct via CRT. Each small NTT butterfly is ~5 ops (1 mul32 +
// Barrett) vs ~128 mul32 for a full Fr butterfly.
//
// === PRIME SELECTION ===
//
// BN254 Fr = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
// 2-adicity of Fr: 28 (supports NTT up to 2^28)
//
// We need primes p_i where:
//   (a) p_i < 2^31 (fits in uint32)
//   (b) p_i - 1 divisible by 2^22 (2-adicity >= 22, to support NTT size 2^22)
//   (c) product(p_i) > Fr (no information loss)
//
// Available primes (p = k * 2^22 + 1, prime, < 2^31): 40 candidates exist.
// Minimum 10 primes needed: product of 9 largest 31-bit candidates = 277.7 bits,
// but product of 9 smallest = 247.2 bits < 254 bits of Fr.
// 10 primes give product of 276.0 bits > 254 bits. OK.
//
// Example set (10 primes, 2-adicity >= 22):
//   104857601, 113246209, 138412033, 155189249, 163577857,
//   167772161, 230686721, 377487361, 415236097, 469762049
//
// === COST ANALYSIS AT N = 2^22 (4M elements) ===
//
// Current BN254 NTT: ~26ms on M3 Pro GPU
//   - fr_mul CIOS: 128 mul32 per butterfly
//   - Highly optimized: radix-4, fused threadgroup kernels, column/row decomposition
//   - Bandwidth: ~3.2 GB total (12 global stages * N * 32 bytes * 2 r/w)
//   - Bandwidth floor: ~23ms at 150 GB/s -> NTT is near bandwidth limit
//
// CRT NTT estimate:
//   - 10 small NTTs, each at 4 bytes/element
//   - Compute: 10 * N * logN/2 * 5 ops = 2.3G ops (vs 5.9G for current) -> 2.5x less
//   - Bandwidth: 10 * (9 global stages * N * 4B * 2) = 3.0 GB -> similar to current
//   - Plus CRT conversion: ~0.6 GB bandwidth, ~0.6ms compute
//   - Plus dispatch overhead for 10 NTTs: ~0.5ms
//   - BEST CASE TOTAL: ~24ms (vs 26ms current = only 8% improvement)
//
// === WHY CRT DOES NOT HELP HERE ===
//
// 1. BN254 NTT is BANDWIDTH-BOUND, not compute-bound.
//    The expensive fr_mul (128 mul32) is largely hidden behind memory latency.
//    CRT eliminates compute but cannot reduce bandwidth below current levels.
//
// 2. 10 small NTTs = 10x the memory traffic per stage.
//    Even though each element is 8x smaller (4B vs 32B), there are 10 NTTs,
//    so total bandwidth is 10*4/32 = 1.25x MORE than current.
//
// 3. CRT conversion overhead is non-trivial:
//    Forward:  256-bit mod p_i for 10 primes = 160 mul32 per element
//    Inverse:  Mixed-radix CRT reconstruction = ~100 mul32 per element
//    For 4M elements: ~0.6ms compute + ~4ms bandwidth round-trip
//
// 4. Small NTTs cannot fuse as many stages when batched:
//    Batched: 10 primes * 512 elements = 5120 uint32 = 20KB (fits in 32KB threadgroup)
//    -> only 9 stages fused vs 13 for sequential -> more global passes
//
// 5. Memory expansion: 32B/element -> 40B/element (10 * 4B), 1.25x more GPU memory
//
// === WHERE CRT/RNS IS BENEFICIAL ===
//
// - Homomorphic encryption: elements are ALREADY in RNS form (see RNSNTTEngine.swift)
// - Larger fields (384+ bit, e.g., BLS12-381 Fq): mul cost grows quadratically
// - CPU implementations: different cache/compute balance
// - When NTT is truly compute-bound (not the case for BN254 on Metal GPU)
//
// === CONCLUSION ===
//
// Do not implement. The existing RNSNTTEngine already serves the HE use case.
// For BN254 Fr NTT, continue optimizing the direct approach (fused kernels,
// radix-4/8, coalesced access patterns).

// No implementation — this file documents the infeasibility analysis only.
// The types below are stubs to preserve the file as a reference.

import Foundation

/// CRT-based NTT for BN254 Fr — ANALYSIS ONLY, NOT IMPLEMENTED.
/// See file header for detailed feasibility analysis showing ~8% best-case
/// improvement does not justify the complexity.
public enum CRTNTT {
    /// NTT-friendly primes with 2-adicity >= 22, fitting in 31 bits.
    /// Product of all 10 = 276 bits > 254 bits of BN254 Fr.
    public static let crtPrimes: [UInt32] = [
        104_857_601,  // 25 * 2^22 + 1, 27 bits
        113_246_209,  // 27 * 2^22 + 1, 27 bits
        138_412_033,  // 33 * 2^22 + 1, 28 bits
        155_189_249,  // 37 * 2^22 + 1, 28 bits
        163_577_857,  // 39 * 2^22 + 1, 28 bits
        167_772_161,  // 40 * 2^22 + 1, 28 bits
        230_686_721,  // 55 * 2^22 + 1, 28 bits
        377_487_361,  // 90 * 2^22 + 1, 29 bits
        415_236_097,  // 99 * 2^22 + 1, 29 bits
        469_762_049,  // 112 * 2^22 + 1, 29 bits
    ]

    /// Estimated performance comparison at N = 2^22 on M3 Pro GPU.
    public static let estimatedCurrentMs: Double = 26.0
    public static let estimatedCRTBestCaseMs: Double = 24.0
    public static let estimatedSpeedup: Double = 1.08  // ~8%, not worth it
}
