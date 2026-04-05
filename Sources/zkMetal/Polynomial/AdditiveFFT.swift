// Additive FFT for binary tower fields (Lin-Chung-Han / Cantor algorithm)
//
// For binary fields GF(2^m), the standard multiplicative NTT doesn't work because
// there is no multiplicative subgroup of 2-power order. Instead, additive FFT
// evaluates polynomials at all points of an affine subspace using the subspace
// polynomial s(x) = x^2 + x, which is GF(2)-linear.
//
// This is the "NTT equivalent" required for Binius-style binary field STARKs
// and polynomial commitment schemes.
//
// Complexity: O(n log^2 n) field multiplications, O(n log n) XOR additions
// (XOR is effectively free on ARM64).
//
// References:
// - Lin, Chung, Han: "Novel Polynomial Basis and Its Application" (FOCS 2014)
// - Cantor: "On arithmetical algorithms over finite fields" (1989)

import Foundation
import NeonFieldOps

// MARK: - Additive FFT over GF(2^64)

/// Additive FFT engine for GF(2^64) binary field.
/// Wraps the C implementation using PMULL-accelerated GF(2^64) arithmetic.
public struct AdditiveFFT64 {
    /// The subspace basis vectors (k elements for a 2^k-point transform).
    public let basis: [UInt64]
    /// Transform size: 2^k.
    public let size: Int
    /// Log of transform size.
    public let logSize: Int

    /// Initialize with a given transform size (must be power of 2).
    /// Generates the standard Frobenius basis automatically.
    public init(logSize k: Int) {
        precondition(k >= 0 && k <= 63, "logSize must be in [0, 63]")
        self.logSize = k
        self.size = 1 << k

        var b = [UInt64](repeating: 0, count: k)
        if k > 0 {
            b.withUnsafeMutableBufferPointer { buf in
                bt_afft_basis_64(buf.baseAddress!, Int32(k))
            }
        }
        self.basis = b
    }

    /// Initialize with an explicit basis.
    /// basis.count determines the transform size (2^basis.count).
    public init(basis: [UInt64]) {
        self.basis = basis
        self.logSize = basis.count
        self.size = basis.isEmpty ? 1 : (1 << basis.count)
    }

    /// Forward additive FFT: evaluate polynomial at all subspace points.
    /// Input: coefficients in novel polynomial basis (n = 2^k elements).
    /// Output: evaluations at the 2^k points of the subspace (in-place).
    /// Uses the NEON-accelerated iterative implementation.
    public func forward(_ data: inout [UInt64]) {
        precondition(data.count == size, "Data size must equal transform size \(size)")
        if size <= 1 { return }
        data.withUnsafeMutableBufferPointer { buf in
            basis.withUnsafeBufferPointer { basisBuf in
                bt_afft_forward_64_neon(buf.baseAddress!, size_t(size), basisBuf.baseAddress!)
            }
        }
    }

    /// Inverse additive FFT: interpolate from evaluations to novel basis coefficients.
    /// Input: evaluations at 2^k subspace points (n elements).
    /// Output: novel basis coefficients (in-place).
    public func inverse(_ data: inout [UInt64]) {
        precondition(data.count == size, "Data size must equal transform size \(size)")
        if size <= 1 { return }
        data.withUnsafeMutableBufferPointer { buf in
            basis.withUnsafeBufferPointer { basisBuf in
                bt_afft_inverse_64_neon(buf.baseAddress!, size_t(size), basisBuf.baseAddress!)
            }
        }
    }

    /// Forward FFT using unsafe pointers (zero-copy for performance-critical paths).
    @inline(__always)
    public func forward(_ data: UnsafeMutablePointer<UInt64>, count: Int) {
        precondition(count == size)
        if size <= 1 { return }
        basis.withUnsafeBufferPointer { basisBuf in
            bt_afft_forward_64_neon(data, size_t(size), basisBuf.baseAddress!)
        }
    }

    /// Inverse FFT using unsafe pointers.
    @inline(__always)
    public func inverse(_ data: UnsafeMutablePointer<UInt64>, count: Int) {
        precondition(count == size)
        if size <= 1 { return }
        basis.withUnsafeBufferPointer { basisBuf in
            bt_afft_inverse_64_neon(data, size_t(size), basisBuf.baseAddress!)
        }
    }
}

// MARK: - Additive FFT over GF(2^128)

/// Additive FFT engine for GF(2^128) binary field.
/// Each element is represented as a pair of UInt64 (lo, hi).
public struct AdditiveFFT128 {
    /// The subspace basis vectors. Each element is 2 x UInt64 (interleaved: [lo0, hi0, lo1, hi1, ...]).
    public let basis: [UInt64]
    /// Transform size: 2^k.
    public let size: Int
    /// Log of transform size.
    public let logSize: Int

    /// Initialize with a given transform size.
    public init(logSize k: Int) {
        precondition(k >= 0 && k <= 127, "logSize must be in [0, 127]")
        self.logSize = k
        self.size = 1 << k

        var b = [UInt64](repeating: 0, count: 2 * k)
        if k > 0 {
            b.withUnsafeMutableBufferPointer { buf in
                bt_afft_basis_128(buf.baseAddress!, Int32(k))
            }
        }
        self.basis = b
    }

    /// Initialize with an explicit basis (flat array: [lo0, hi0, lo1, hi1, ...]).
    public init(basis: [UInt64]) {
        precondition(basis.count % 2 == 0, "Basis must have even number of uint64 (pairs of lo/hi)")
        self.basis = basis
        self.logSize = basis.count / 2
        self.size = basis.isEmpty ? 1 : (1 << (basis.count / 2))
    }

    /// Forward additive FFT for GF(2^128).
    /// data: flat array of 2*n uint64 values (n elements, each lo/hi pair).
    public func forward(_ data: inout [UInt64]) {
        precondition(data.count == 2 * size, "Data must have \(2 * size) uint64 values (\(size) GF(2^128) elements)")
        if size <= 1 { return }
        data.withUnsafeMutableBufferPointer { buf in
            basis.withUnsafeBufferPointer { basisBuf in
                bt_afft_forward_128_iter(buf.baseAddress!, size_t(size), basisBuf.baseAddress!)
            }
        }
    }

    /// Inverse additive FFT for GF(2^128).
    public func inverse(_ data: inout [UInt64]) {
        precondition(data.count == 2 * size, "Data must have \(2 * size) uint64 values (\(size) GF(2^128) elements)")
        if size <= 1 { return }
        data.withUnsafeMutableBufferPointer { buf in
            basis.withUnsafeBufferPointer { basisBuf in
                bt_afft_inverse_128_iter(buf.baseAddress!, size_t(size), basisBuf.baseAddress!)
            }
        }
    }

    /// Forward FFT using unsafe pointers (zero-copy).
    /// data must point to 2*count uint64 values.
    @inline(__always)
    public func forward(_ data: UnsafeMutablePointer<UInt64>, count: Int) {
        precondition(count == size)
        if size <= 1 { return }
        basis.withUnsafeBufferPointer { basisBuf in
            bt_afft_forward_128_iter(data, size_t(size), basisBuf.baseAddress!)
        }
    }

    /// Inverse FFT using unsafe pointers.
    @inline(__always)
    public func inverse(_ data: UnsafeMutablePointer<UInt64>, count: Int) {
        precondition(count == size)
        if size <= 1 { return }
        basis.withUnsafeBufferPointer { basisBuf in
            bt_afft_inverse_128_iter(data, size_t(size), basisBuf.baseAddress!)
        }
    }
}

// MARK: - Convenience: polynomial multiply via additive FFT

extension AdditiveFFT64 {
    /// Multiply two polynomials using additive FFT.
    /// Both polynomials must have at most size/2 coefficients (so their product
    /// has at most size-1 coefficients and fits in the transform).
    public func multiply(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        precondition(a.count + b.count - 1 <= size,
                     "Product degree exceeds transform size")

        // Pad to transform size
        var aData = a + [UInt64](repeating: 0, count: size - a.count)
        var bData = b + [UInt64](repeating: 0, count: size - b.count)

        // Forward FFT both
        forward(&aData)
        forward(&bData)

        // Pointwise multiply
        for i in 0..<size {
            aData[i] = bt_gf64_mul(aData[i], bData[i])
        }

        // Inverse FFT
        inverse(&aData)

        return aData
    }
}

extension AdditiveFFT128 {
    /// Multiply two GF(2^128) polynomials using additive FFT.
    /// a, b: flat arrays of 2*n uint64 each (n GF(2^128) elements as lo/hi pairs).
    public func multiply(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        let aCount = a.count / 2
        let bCount = b.count / 2
        precondition(aCount + bCount - 1 <= size,
                     "Product degree exceeds transform size")

        var aData = a + [UInt64](repeating: 0, count: 2 * size - a.count)
        var bData = b + [UInt64](repeating: 0, count: 2 * size - b.count)

        forward(&aData)
        forward(&bData)

        // Pointwise multiply
        var tmp = [UInt64](repeating: 0, count: 2)
        for i in 0..<size {
            bt_gf128_mul([aData[2*i], aData[2*i+1]], [bData[2*i], bData[2*i+1]], &tmp)
            aData[2*i] = tmp[0]
            aData[2*i+1] = tmp[1]
        }

        inverse(&aData)

        return aData
    }
}
