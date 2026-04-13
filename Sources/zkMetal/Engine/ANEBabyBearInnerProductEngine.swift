// ANEBabyBearInnerProductEngine — ANE-accelerated BabyBear Inner Product
//
// Provides GPU/ANE-accelerated inner product for BabyBear field (p = 2^31 - 2^27 + 1).
//
// BabyBear field properties:
//   - Prime: 0x78000001 = 2013265921 (fits in 31 bits)
//   - Well-suited for ANE: values fit in FP16 precision
//   - Used by SP1, RISC Zero, Plonky3 for STARK proving
//
// ANE acceleration strategy:
//   - Inner product <a,b> = sum(a_i * b_i) maps naturally to ANE matmul
//   - Pack N pairs into ANE-friendly format, compute all products in parallel
//   - SIMD4 vectorization processes 4 field elements per thread
//
// Integration points:
//   - Sumcheck prover (when using BabyBear field)
//   - GKR protocol (BabyBear-based circuits)
//   - IPA arguments over BabyBear extension fields
//   - Multilinear evaluation via MLE

import Foundation
import ANEOps

// MARK: - BabyBear Inner Product Engine

/// ANE-accelerated inner product engine for BabyBear field.
///
/// Computes <a, b> = sum(a_i * b_i) for vectors of BabyBear elements.
/// Uses ANE/GPU acceleration when available, falls back to CPU scalar arithmetic.
///
/// - Note: This engine operates on BabyBear UInt32 representation directly,
///   not on the Swift `Bb` type. Convert to/from UInt32 as needed.
public final class ANEBabyBearInnerProductEngine {

    // MARK: - Configuration

    /// Minimum vector size to use ANE acceleration (smaller = CPU).
    public var aneThreshold: Int = 64

    /// Whether to use ANE even for small vectors.
    public var forceANE: Bool = false

    /// ANE/GPU availability (checked at initialization).
    public private(set) var isANEAvailable: Bool = false

    /// Initialize ANE tensor subsystem.
    public static func initializeANE() -> Bool {
        let result = ane_tensor_init()
        if result == 0 {
            let available = ane_tensor_gpu_available()
            return available
        }
        return false
    }

    // MARK: - Initialization

    public init() {
        ANEBabyBearInnerProductEngine.initializeANE()
        isANEAvailable = ane_tensor_gpu_available()
    }

    // MARK: - Public API

    /// Compute inner product <a, b> = sum(a_i * b_i) for BabyBear vectors.
    ///
    /// - Parameters:
    ///   - a: First vector (BabyBear elements as UInt32)
    ///   - b: Second vector (BabyBear elements as UInt32)
    /// - Returns: Inner product as BabyBear UInt32
    ///
    /// - Complexity: O(n) time, O(1) space
    public func innerProduct(_ a: [UInt32], _ b: [UInt32]) -> UInt32 {
        precondition(a.count == b.count, "Vector lengths must match")

        let n = a.count
        if n == 0 { return 0 }
        if n == 1 { return bbMul(a[0], b[0]) }

        // Small vectors: use CPU
        if n < aneThreshold && !forceANE {
            return cpuInnerProduct(a, b)
        }

        // ANE/GPU path
        if isANEAvailable {
            return aneInnerProduct(a, b)
        }

        // CPU fallback
        return cpuInnerProduct(a, b)
    }

    /// Compute multiple inner products in a single ANE dispatch.
    ///
    /// - Parameters:
    ///   - pairs: Array of (a, b) vector pairs
    /// - Returns: Array of inner products, one per pair
    ///
    /// - Complexity: O(total_elements) time, O(batch) space
    public func batchInnerProduct(_ pairs: [([UInt32], [UInt32])]) -> [UInt32] {
        guard !pairs.isEmpty else { return [] }

        // Check for trivial cases
        let totalElements = pairs.reduce(0) { $0 + $1.0.count }
        let maxLen = pairs.reduce(0) { max($0, $1.0.count) }

        // All single elements: compute directly
        if maxLen == 1 {
            return pairs.map { bbMul($0.0[0], $0.1[0]) }
        }

        // Small total work: use CPU
        if totalElements < aneThreshold && !forceANE {
            return pairs.map { cpuInnerProduct($0.0, $0.1) }
        }

        // Batch ANE path
        if isANEAvailable && pairs.count > 1 {
            return aneBatchInnerProduct(pairs)
        }

        // CPU fallback
        return pairs.map { cpuInnerProduct($0.0, $0.1) }
    }

    /// Weighted sum: Sigma values_i * weights_i (same as inner product).
    public func weightedSum(values: [UInt32], weights: [UInt32]) -> UInt32 {
        return innerProduct(values, weights)
    }

    /// Inner product for multilinear evaluation:
    /// Sigma evals_i * eq_i where eq is the equality polynomial.
    public func multiEqInnerProduct(evals: [UInt32], eq: [UInt32]) -> UInt32 {
        return innerProduct(evals, eq)
    }

    // MARK: - CPU Fallback

    /// Scalar BabyBear inner product (CPU fallback).
    private func cpuInnerProduct(_ a: [UInt32], _ b: [UInt32]) -> UInt32 {
        var acc: UInt32 = 0
        for i in 0..<a.count {
            acc = bbAdd(acc, bbMul(a[i], b[i]))
        }
        return acc
    }

    // MARK: - ANE/GPU Path

    /// ANE-accelerated inner product via tensor API.
    private func aneInnerProduct(_ a: [UInt32], _ b: [UInt32]) -> UInt32 {
        return a.withUnsafeBytes { aPtr in
            b.withUnsafeBytes { bPtr in
                ane_tensor_inner_product(
                    aPtr.baseAddress!.assumingMemoryBound(to: UInt32.self),
                    bPtr.baseAddress!.assumingMemoryBound(to: UInt32.self),
                    Int32(a.count))
            }
        }
    }

    /// ANE batch inner product via tensor API.
    private func aneBatchInnerProduct(_ pairs: [([UInt32], [UInt32])]) -> [UInt32] {
        // Flatten pairs into batched format
        let n = pairs[0].0.count  // All pairs must have same length
        let batch = pairs.count

        var aBatch = [UInt32](repeating: 0, count: batch * n)
        var bBatch = [UInt32](repeating: 0, count: batch * n)

        for k in 0..<batch {
            precondition(pairs[k].0.count == n, "All pairs must have same length")
            let dstOffset = k * n
            aBatch[dstOffset ..< dstOffset + n] = pairs[k].0[...]
            bBatch[dstOffset ..< dstOffset + n] = pairs[k].1[...]
        }

        var results = [UInt32](repeating: 0, count: batch)

        aBatch.withUnsafeBytes { aPtr in
            bBatch.withUnsafeBytes { bPtr in
                results.withUnsafeMutableBytes { rPtr in
                    ane_tensor_inner_product_batch(
                        aPtr.baseAddress!.assumingMemoryBound(to: UInt32.self),
                        bPtr.baseAddress!.assumingMemoryBound(to: UInt32.self),
                        Int32(n),
                        Int32(batch),
                        rPtr.baseAddress!.assumingMemoryBound(to: UInt32.self))
                }
            }
        }

        return results
    }
}

// MARK: - BabyBear Scalar Arithmetic

/// BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
///
/// These are plain (non-Montgomery) arithmetic operations.

private let BB_P: UInt32 = 0x78000001
private let BB_MU: UInt32 = 2290649223  // Barrett reduction coefficient

/// BabyBear modular multiplication via Barrett reduction.
@inline(__always)
private func bbMul(_ a: UInt32, _ b: UInt32) -> UInt32 {
    let prod = UInt64(a) * UInt64(b)
    // Use truncatingIfNeeded because prod may exceed UInt32.max
    let prodLo = UInt32(truncatingIfNeeded: prod)
    let prodHi = UInt32(truncatingIfNeeded: prod >> 32)
    let t1 = UInt64(prodLo) * UInt64(BB_MU)
    let t2 = UInt64(prodHi) * UInt64(BB_MU)
    let q = UInt32(truncatingIfNeeded: (t2 + (t1 >> 32)) >> 30)
    let r = UInt32(truncatingIfNeeded: prod - UInt64(q) * UInt64(BB_P))
    return r >= BB_P ? r - BB_P : r
}

/// BabyBear modular addition.
@inline(__always)
private func bbAdd(_ a: UInt32, _ b: UInt32) -> UInt32 {
    let s = a &+ b
    return s >= BB_P ? s &- BB_P : s
}

// MARK: - Swift Bb Type Integration

extension ANEBabyBearInnerProductEngine {

    /// Compute inner product using Swift `Bb` type.
    ///
    /// Converts Bb elements to UInt32, computes inner product, returns Bb result.
    public func innerProductBb(_ a: [Bb], _ b: [Bb]) -> Bb {
        let aU32 = a.map { $0.v }
        let bU32 = b.map { $0.v }
        let result = innerProduct(aU32, bU32)
        return Bb(v: result)
    }

    /// Batch inner product using Swift `Bb` type.
    public func batchInnerProductBb(_ pairs: [([Bb], [Bb])]) -> [Bb] {
        let u32Pairs = pairs.map { (a: [Bb], b: [Bb]) -> ([UInt32], [UInt32]) in
            (a.map { $0.v }, b.map { $0.v })
        }
        let results = batchInnerProduct(u32Pairs)
        return results.map { Bb(v: $0) }
    }
}

// MARK: - Convenience Initializers

extension ANEBabyBearInnerProductEngine {

    /// Create engine with ANE forced on (useful for testing).
    public static func createWithANE() -> ANEBabyBearInnerProductEngine {
        let engine = ANEBabyBearInnerProductEngine()
        engine.forceANE = true
        return engine
    }

    /// Create engine with custom ANE threshold.
    public static func create(threshold: Int) -> ANEBabyBearInnerProductEngine {
        let engine = ANEBabyBearInnerProductEngine()
        engine.aneThreshold = threshold
        return engine
    }
}
