// Reed-Solomon Engine — GPU-accelerated encode/decode/verify for DAS, FRI, and STARK provers
//
// Systematic RS encoding: data as polynomial coefficients, NTT-evaluate at codeRate*n points.
// Erasure decoding: recover from any k-of-n symbols via Lagrange interpolation.
// Verify: check that codeword lies on a polynomial of degree < dataLen.
//
// Supports BN254 Fr (via GPU NTT) and BabyBear (via GPU NTT).
// CPU fallback for small inputs (< 64 elements).

import Foundation
import Metal

// MARK: - RSEngineError

public enum RSEngineError: Error {
    case noGPU
    case invalidCodeRate
    case dataTooLarge
    case insufficientSymbols
    case invalidCodeword
    case nttFailed(String)
}

// MARK: - Field Protocol for Generic RS Operations

/// Minimal protocol abstracting field operations needed by the RS engine.
/// Enables generic encode/decode/verify over BN254 Fr and BabyBear.
public protocol RSField: Equatable {
    static var zero: Self { get }
    static var one: Self { get }
    var isZero: Bool { get }
    static var twoAdicity: Int { get }
}

extension Fr: RSField {
    public static var twoAdicity: Int { Fr.TWO_ADICITY }
}

extension Bb: RSField {
    public static var twoAdicity: Int { Bb.TWO_ADICITY }
}

// MARK: - ReedSolomonEngine (BN254 Fr)

/// GPU-accelerated Reed-Solomon engine over BN254 Fr.
/// Uses NTT for O(n log n) encoding and iNTT-assisted decoding.
///
/// Encoding: interpret data[0..k) as polynomial coefficients, evaluate at n = codeRate*k
/// points (n-th roots of unity) via forward NTT. Result is systematic in frequency domain.
///
/// Decoding: given any k-of-n (index, value) pairs, recover original k coefficients
/// via Lagrange interpolation (CPU for small k, or NTT-assisted for large k).
///
/// Verification: check that a codeword is a valid RS encoding (i.e., iNTT yields
/// a polynomial with degree < dataLen, meaning high coefficients are zero).
public class ReedSolomonEngine {
    public static let version = Versions.reedSolomon
    public let nttEngine: NTTEngine

    /// CPU fallback threshold: below this size, use direct polynomial evaluation.
    public static let cpuThreshold = 64

    public init() throws {
        self.nttEngine = try NTTEngine()
    }

    // MARK: - Encode

    /// Systematic RS encoding: interpret data as polynomial coefficients,
    /// evaluate at codeRate * nextPow2(data.count) points via NTT.
    ///
    /// - Parameters:
    ///   - data: Original data as Fr field elements (polynomial coefficients).
    ///   - codeRate: Blowup factor (2, 4, 8). Total codeword length = codeRate * nextPow2(data.count).
    /// - Returns: Codeword of length n = codeRate * nextPow2(data.count), evaluations at n-th roots of unity.
    public func encode(data: [Fr], codeRate: Int = 2) throws -> [Fr] {
        guard codeRate >= 2 && (codeRate & (codeRate - 1)) == 0 else {
            throw RSEngineError.invalidCodeRate
        }
        let k = data.count
        guard k > 0 else { return [] }

        let kPow2 = nextPow2RS(k)
        let n = kPow2 * codeRate
        let logN = logBase2(n)

        guard logN <= Fr.TWO_ADICITY else {
            throw RSEngineError.dataTooLarge
        }

        // Pad to n with zeros (higher-degree coefficients = 0)
        var padded = [Fr](repeating: .zero, count: n)
        for i in 0..<k { padded[i] = data[i] }

        // Small inputs: CPU polynomial evaluation
        if n <= ReedSolomonEngine.cpuThreshold {
            return encodeCPU(coeffs: padded, logN: logN)
        }

        // GPU: forward NTT evaluates polynomial at n-th roots of unity
        return try nttEngine.ntt(padded)
    }

    /// CPU encode: evaluate polynomial at each n-th root of unity via Horner's method.
    private func encodeCPU(coeffs: [Fr], logN: Int) -> [Fr] {
        let n = coeffs.count
        let omega = frRootOfUnity(logN: logN)
        var result = [Fr](repeating: .zero, count: n)

        var omegaI = Fr.one
        for i in 0..<n {
            // Horner evaluation of poly at omega^i
            var acc = Fr.zero
            for j in stride(from: coeffs.count - 1, through: 0, by: -1) {
                acc = frAdd(frMul(acc, omegaI), coeffs[j])
            }
            result[i] = acc
            omegaI = frMul(omegaI, omega)
        }
        return result
    }

    // MARK: - Decode

    /// Erasure decoding: recover original k coefficients from any k-of-n (index, value) pairs.
    ///
    /// - Parameters:
    ///   - codeword: Array of (index, value) tuples. Index is position in the codeword (0..<n).
    ///   - dataLen: Number of original polynomial coefficients (k).
    /// - Returns: Original k polynomial coefficients.
    public func decode(codeword: [(Int, Fr)], dataLen: Int) throws -> [Fr] {
        guard codeword.count >= dataLen else {
            throw RSEngineError.insufficientSymbols
        }
        guard dataLen > 0 else { return [] }

        let k = dataLen
        let usable = Array(codeword.prefix(k))

        // Determine codeword length from context
        let maxIdx = usable.map { $0.0 }.max()! + 1
        let n = nextPow2RS(max(maxIdx, k * 2))
        let logN = logBase2(n)
        let omega = frRootOfUnity(logN: logN)

        // Compute evaluation points: x_i = omega^(index_i)
        let points = usable.map { (idx, _) -> Fr in
            frPow(omega, UInt64(idx))
        }
        let values = usable.map { $0.1 }

        // Lagrange interpolation to recover polynomial coefficients
        let coeffs = lagrangeInterpolateFr(points: points, values: values)
        return Array(coeffs.prefix(k))
    }

    // MARK: - Verify

    /// Verify that a codeword is a valid RS encoding of a degree < dataLen polynomial.
    ///
    /// Algorithm: iNTT the codeword to get coefficients, check that coefficients
    /// at positions dataLen..<n are all zero.
    ///
    /// - Parameters:
    ///   - codeword: Full RS codeword (all n evaluations).
    ///   - dataLen: Expected polynomial degree bound.
    /// - Returns: true if codeword is a valid RS encoding.
    public func verify(codeword: [Fr], dataLen: Int) throws -> Bool {
        let n = codeword.count
        guard n > 0 && (n & (n - 1)) == 0 else { return false }
        guard dataLen > 0 && dataLen <= n else { return false }

        // Small inputs: CPU verification
        if n <= ReedSolomonEngine.cpuThreshold {
            return verifyCPU(codeword: codeword, dataLen: dataLen)
        }

        // GPU iNTT to recover coefficients
        let coeffs = try nttEngine.intt(codeword)

        // Check that high-degree coefficients are zero
        for i in dataLen..<n {
            if !coeffs[i].isZero { return false }
        }
        return true
    }

    /// CPU verification: iNTT via direct computation, check high coefficients.
    private func verifyCPU(codeword: [Fr], dataLen: Int) -> Bool {
        let n = codeword.count
        let logN = logBase2(n)
        let omega = frRootOfUnity(logN: logN)

        // Build evaluation points
        var points = [Fr](repeating: .zero, count: n)
        var omegaI = Fr.one
        for i in 0..<n {
            points[i] = omegaI
            omegaI = frMul(omegaI, omega)
        }

        // Interpolate to get coefficients
        let coeffs = lagrangeInterpolateFr(points: points, values: codeword)

        // Check high-degree coefficients are zero
        for i in dataLen..<n {
            if !coeffs[i].isZero { return false }
        }
        return true
    }

    // MARK: - Lagrange Interpolation (BN254 Fr)

    /// Lagrange interpolation: given (x_i, y_i), recover polynomial coefficients.
    /// O(k^2) — suitable for moderate k. For large k, use NTT-based methods.
    private func lagrangeInterpolateFr(points: [Fr], values: [Fr]) -> [Fr] {
        let n = points.count
        var result = [Fr](repeating: .zero, count: n)

        // Precompute all Lagrange denominators and batch-invert
        var denoms = [Fr](repeating: Fr.one, count: n)
        for i in 0..<n {
            for j in 0..<n where j != i {
                denoms[i] = frMul(denoms[i], frSub(points[i], points[j]))
            }
        }
        // Montgomery batch inversion: n inverses → 1 inverse + 3(n-1) muls
        var prefix = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { prefix[i] = frMul(prefix[i - 1], denoms[i - 1]) }
        var acc = frInverse(frMul(prefix[n - 1], denoms[n - 1]))
        var denomInvs = [Fr](repeating: Fr.zero, count: n)
        for i in Swift.stride(from: n - 1, through: 0, by: -1) {
            denomInvs[i] = frMul(acc, prefix[i])
            acc = frMul(acc, denoms[i])
        }

        for i in 0..<n {
            var basis = [Fr](repeating: .zero, count: n)
            basis[0] = .one
            var basisDeg = 0

            for j in 0..<n {
                if j == i { continue }
                basisDeg += 1
                for d in Swift.stride(from: basisDeg, through: 1, by: -1) {
                    basis[d] = frSub(basis[d - 1], frMul(points[j], basis[d]))
                }
                basis[0] = frSub(.zero, frMul(points[j], basis[0]))
            }

            let scale = frMul(values[i], denomInvs[i])
            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(scale, basis[d]))
            }
        }
        return result
    }
}

// MARK: - ReedSolomonBbEngine (BabyBear)

/// GPU-accelerated Reed-Solomon engine over BabyBear field.
/// Same algorithm as the BN254 engine but using BabyBear NTT.
public class ReedSolomonBbEngine {
    public static let version = Versions.reedSolomon
    public let nttEngine: BabyBearNTTEngine

    public static let cpuThreshold = 64

    public init() throws {
        self.nttEngine = try BabyBearNTTEngine()
    }

    // MARK: - Encode

    /// Systematic RS encoding over BabyBear: evaluate polynomial at codeRate * n roots of unity.
    public func encode(data: [Bb], codeRate: Int = 2) throws -> [Bb] {
        guard codeRate >= 2 && (codeRate & (codeRate - 1)) == 0 else {
            throw RSEngineError.invalidCodeRate
        }
        let k = data.count
        guard k > 0 else { return [] }

        let kPow2 = nextPow2RS(k)
        let n = kPow2 * codeRate
        let logN = logBase2(n)

        guard logN <= Bb.TWO_ADICITY else {
            throw RSEngineError.dataTooLarge
        }

        var padded = [Bb](repeating: .zero, count: n)
        for i in 0..<k { padded[i] = data[i] }

        if n <= ReedSolomonBbEngine.cpuThreshold {
            return encodeCPU(coeffs: padded, logN: logN)
        }

        return try nttEngine.ntt(padded)
    }

    private func encodeCPU(coeffs: [Bb], logN: Int) -> [Bb] {
        let n = coeffs.count
        let omega = bbRootOfUnity(logN: logN)
        var result = [Bb](repeating: .zero, count: n)

        var omegaI = Bb.one
        for i in 0..<n {
            var acc = Bb.zero
            for j in stride(from: coeffs.count - 1, through: 0, by: -1) {
                acc = bbAdd(bbMul(acc, omegaI), coeffs[j])
            }
            result[i] = acc
            omegaI = bbMul(omegaI, omega)
        }
        return result
    }

    // MARK: - Decode

    /// Erasure decoding from any k-of-n (index, value) pairs over BabyBear.
    public func decode(codeword: [(Int, Bb)], dataLen: Int) throws -> [Bb] {
        guard codeword.count >= dataLen else {
            throw RSEngineError.insufficientSymbols
        }
        guard dataLen > 0 else { return [] }

        let k = dataLen
        let usable = Array(codeword.prefix(k))

        let maxIdx = usable.map { $0.0 }.max()! + 1
        let n = nextPow2RS(max(maxIdx, k * 2))
        let logN = logBase2(n)
        let omega = bbRootOfUnity(logN: logN)

        let points = usable.map { (idx, _) -> Bb in
            bbPow(omega, UInt32(idx))
        }
        let values = usable.map { $0.1 }

        let coeffs = lagrangeInterpolateBb(points: points, values: values)
        return Array(coeffs.prefix(k))
    }

    // MARK: - Verify

    /// Verify that a BabyBear codeword is a valid RS encoding with degree < dataLen.
    public func verify(codeword: [Bb], dataLen: Int) throws -> Bool {
        let n = codeword.count
        guard n > 0 && (n & (n - 1)) == 0 else { return false }
        guard dataLen > 0 && dataLen <= n else { return false }

        if n <= ReedSolomonBbEngine.cpuThreshold {
            return verifyCPU(codeword: codeword, dataLen: dataLen)
        }

        let coeffs = try nttEngine.intt(codeword)
        for i in dataLen..<n {
            if !coeffs[i].isZero { return false }
        }
        return true
    }

    private func verifyCPU(codeword: [Bb], dataLen: Int) -> Bool {
        let n = codeword.count
        let logN = logBase2(n)
        let omega = bbRootOfUnity(logN: logN)

        var points = [Bb](repeating: .zero, count: n)
        var omegaI = Bb.one
        for i in 0..<n {
            points[i] = omegaI
            omegaI = bbMul(omegaI, omega)
        }

        let coeffs = lagrangeInterpolateBb(points: points, values: codeword)
        for i in dataLen..<n {
            if !coeffs[i].isZero { return false }
        }
        return true
    }

    // MARK: - Lagrange Interpolation (BabyBear)

    private func lagrangeInterpolateBb(points: [Bb], values: [Bb]) -> [Bb] {
        let n = points.count
        var result = [Bb](repeating: .zero, count: n)

        // Precompute all Lagrange denominators and batch-invert
        var denoms = [Bb](repeating: Bb.one, count: n)
        for i in 0..<n {
            for j in 0..<n where j != i {
                denoms[i] = bbMul(denoms[i], bbSub(points[i], points[j]))
            }
        }
        // Montgomery batch inversion: n inverses → 1 inverse + 3(n-1) muls
        var prefix = [Bb](repeating: Bb.one, count: n)
        for i in 1..<n { prefix[i] = bbMul(prefix[i - 1], denoms[i - 1]) }
        var acc = bbInverse(bbMul(prefix[n - 1], denoms[n - 1]))
        var denomInvs = [Bb](repeating: .zero, count: n)
        for i in Swift.stride(from: n - 1, through: 0, by: -1) {
            denomInvs[i] = bbMul(acc, prefix[i])
            acc = bbMul(acc, denoms[i])
        }

        for i in 0..<n {
            var basis = [Bb](repeating: .zero, count: n)
            basis[0] = .one
            var basisDeg = 0

            for j in 0..<n {
                if j == i { continue }
                basisDeg += 1
                for d in Swift.stride(from: basisDeg, through: 1, by: -1) {
                    basis[d] = bbSub(basis[d - 1], bbMul(points[j], basis[d]))
                }
                basis[0] = bbSub(.zero, bbMul(points[j], basis[0]))
            }

            let scale = bbMul(values[i], denomInvs[i])
            for d in 0..<n {
                result[d] = bbAdd(result[d], bbMul(scale, basis[d]))
            }
        }
        return result
    }
}

// MARK: - Utility

/// Next power of 2 >= n.
func nextPow2RS(_ n: Int) -> Int {
    guard n > 0 else { return 1 }
    if n & (n - 1) == 0 { return n }
    var v = n
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    return v + 1
}

/// Log base 2 of a power-of-2 integer.
public func logBase2(_ n: Int) -> Int {
    precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
    var v = n
    var log = 0
    while v > 1 { v >>= 1; log += 1 }
    return log
}
