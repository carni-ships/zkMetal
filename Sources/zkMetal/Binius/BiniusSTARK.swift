// BiniusSTARK — Binius-style binary tower STARK components
//
// Builds on the existing binary tower field types (BinaryField8..128 from Fields/BinaryTower.swift)
// and the NEON/PMULL C accelerators (binary_tower.c, binary_additive_fft.c) to provide:
//
//   1. Packed binary field: 128 GF(2) elements in a single 128-bit word
//   2. Packed GF(2^8): 16 elements SIMD-parallel via XOR
//   3. Additive FFT wrappers over the C iterative/NEON implementations
//   4. Binary polynomial commitment using expander-code linear codes (Brakedown-style)
//   5. Binary AIR constraint evaluator for STARK proofs
//
// Key performance properties:
//   - Addition = XOR at every tower level (free on CPU and GPU)
//   - Multiplication uses ARM64 PMULL (carry-less polynomial multiply) via NeonFieldOps
//   - Squaring is the Frobenius endomorphism (linear in binary fields)
//
// References:
//   - Binius: https://eprint.iacr.org/2023/1784
//   - Binary tower fields: Wiedemann (1988), Fan & Vercauteren (2012)
//   - Brakedown PCS: Golovnev et al. (eprint 2021/1043)

import Foundation
import NeonFieldOps

// MARK: - Packed Binary Field (128 x GF(2))

/// PackedBinaryField128 packs 128 GF(2) elements into a single 128-bit word.
///
/// Each bit position represents an independent GF(2) element.
/// Addition = XOR (free), multiplication = AND (element-wise).
/// This is the "packed" representation used by Binius for sub-byte field elements,
/// enabling 128-wide SIMD parallelism for GF(2) operations.
public struct PackedBinaryField128: Equatable {
    public var lo: UInt64
    public var hi: UInt64

    public static let zero = PackedBinaryField128(lo: 0, hi: 0)
    public static let allOnes = PackedBinaryField128(lo: ~0, hi: ~0)

    public init(lo: UInt64, hi: UInt64) {
        self.lo = lo
        self.hi = hi
    }

    /// Create from an array of 128 bits (GF(2) elements).
    /// bits[0] is the LSB of lo, bits[127] is the MSB of hi.
    public init(bits: [UInt8]) {
        precondition(bits.count == 128, "Expected exactly 128 bits")
        var lo: UInt64 = 0
        var hi: UInt64 = 0
        for i in 0..<64 {
            if bits[i] & 1 != 0 { lo |= (1 << i) }
        }
        for i in 0..<64 {
            if bits[64 + i] & 1 != 0 { hi |= (1 << i) }
        }
        self.lo = lo
        self.hi = hi
    }

    /// Extract bit at position i (0..127) as a GF(2) element.
    @inline(__always)
    public func bit(at i: Int) -> UInt8 {
        if i < 64 {
            return UInt8((lo >> i) & 1)
        } else {
            return UInt8((hi >> (i - 64)) & 1)
        }
    }

    /// Set bit at position i.
    @inline(__always)
    public mutating func setBit(at i: Int, to val: UInt8) {
        if i < 64 {
            lo = (lo & ~(1 << i)) | (UInt64(val & 1) << i)
        } else {
            let j = i - 64
            hi = (hi & ~(1 << j)) | (UInt64(val & 1) << j)
        }
    }

    /// Population count (Hamming weight).
    public var popcount: Int {
        lo.nonzeroBitCount + hi.nonzeroBitCount
    }

    /// Inner product over GF(2): popcount(a AND b) mod 2.
    @inline(__always)
    public func innerProduct(with other: PackedBinaryField128) -> UInt8 {
        let andLo = lo & other.lo
        let andHi = hi & other.hi
        let total = andLo.nonzeroBitCount + andHi.nonzeroBitCount
        return UInt8(total & 1)
    }
}

/// Packed GF(2) addition = XOR.
@inline(__always)
public func packedBF2Add(_ a: PackedBinaryField128, _ b: PackedBinaryField128) -> PackedBinaryField128 {
    PackedBinaryField128(lo: a.lo ^ b.lo, hi: a.hi ^ b.hi)
}

/// Packed GF(2) multiplication = AND (element-wise).
@inline(__always)
public func packedBF2Mul(_ a: PackedBinaryField128, _ b: PackedBinaryField128) -> PackedBinaryField128 {
    PackedBinaryField128(lo: a.lo & b.lo, hi: a.hi & b.hi)
}

/// Packed GF(2) negation = identity (in characteristic 2, -x = x).
@inline(__always)
public func packedBF2Neg(_ a: PackedBinaryField128) -> PackedBinaryField128 { a }

/// Packed GF(2) bitwise NOT.
@inline(__always)
public func packedBF2Not(_ a: PackedBinaryField128) -> PackedBinaryField128 {
    PackedBinaryField128(lo: ~a.lo, hi: ~a.hi)
}

// MARK: - Packed GF(2^8) x 16

/// Pack 16 GF(2^8) elements into 128 bits.
/// Each element occupies 8 consecutive bits. Addition = lane-wise XOR.
public struct PackedBinaryField8x16: Equatable {
    public var lo: UInt64
    public var hi: UInt64

    public static let zero = PackedBinaryField8x16(lo: 0, hi: 0)

    public init(lo: UInt64, hi: UInt64) {
        self.lo = lo
        self.hi = hi
    }

    /// Create from 16 GF(2^8) elements.
    public init(elements: [BinaryField8]) {
        precondition(elements.count == 16)
        var lo: UInt64 = 0
        var hi: UInt64 = 0
        for i in 0..<8 {
            lo |= UInt64(elements[i].value) << (i * 8)
        }
        for i in 0..<8 {
            hi |= UInt64(elements[8 + i].value) << (i * 8)
        }
        self.lo = lo
        self.hi = hi
    }

    /// Extract the i-th GF(2^8) element (0..15).
    @inline(__always)
    public func element(at i: Int) -> BinaryField8 {
        if i < 8 {
            return BinaryField8(value: UInt8((lo >> (i * 8)) & 0xFF))
        } else {
            return BinaryField8(value: UInt8((hi >> ((i - 8) * 8)) & 0xFF))
        }
    }

    /// Addition = XOR (all 16 elements in parallel).
    @inline(__always)
    public func add(_ other: PackedBinaryField8x16) -> PackedBinaryField8x16 {
        PackedBinaryField8x16(lo: lo ^ other.lo, hi: hi ^ other.hi)
    }
}

// MARK: - Additive FFT Wrappers

/// Binary additive FFT over GF(2^64) using NEON-accelerated C implementation.
///
/// Evaluates a polynomial (in novel/subspace polynomial basis) at all 2^k points
/// of a GF(2)-affine subspace. Uses the Lin-Chung-Han (LCH) / Cantor algorithm
/// with NEON-vectorized XOR propagation.
public struct BinaryAdditiveFFT64 {
    public let logSize: Int
    public let size: Int
    public let basis: [UInt64]

    public init(logSize: Int) {
        precondition(logSize > 0 && logSize <= 64)
        self.logSize = logSize
        self.size = 1 << logSize
        var basisBuf = [UInt64](repeating: 0, count: logSize)
        bt_afft_basis_64(&basisBuf, Int32(logSize))
        self.basis = basisBuf
    }

    /// Forward FFT: coefficients (novel basis) -> evaluations on the subspace.
    /// Input data is BinaryField64 array, modified in-place.
    public func forward(_ data: inout [BinaryField64]) {
        precondition(data.count == size)
        // BinaryField64 is a tower struct (lo: BinaryField32, hi: BinaryField32).
        // The C FFT operates on flat UInt64 values, so we convert.
        var flat = data.map { $0.toUInt64 }
        bt_afft_forward_64_neon(&flat, size, basis)
        for i in 0..<size {
            data[i] = BinaryField64(value: flat[i])
        }
    }

    /// Inverse FFT: evaluations on the subspace -> coefficients (novel basis).
    public func inverse(_ data: inout [BinaryField64]) {
        precondition(data.count == size)
        var flat = data.map { $0.toUInt64 }
        bt_afft_inverse_64_neon(&flat, size, basis)
        for i in 0..<size {
            data[i] = BinaryField64(value: flat[i])
        }
    }
}

/// Binary additive FFT over GF(2^128) using iterative C implementation.
public struct BinaryAdditiveFFT128 {
    public let logSize: Int
    public let size: Int
    /// Basis elements stored as flat UInt64 pairs: [lo0, hi0, lo1, hi1, ...]
    public let basis: [UInt64]

    public init(logSize: Int) {
        precondition(logSize > 0 && logSize <= 128)
        self.logSize = logSize
        self.size = 1 << logSize
        var basisBuf = [UInt64](repeating: 0, count: 2 * logSize)
        bt_afft_basis_128(&basisBuf, Int32(logSize))
        self.basis = basisBuf
    }

    /// Forward FFT over GF(2^128).
    /// Note: BinaryField128 uses tower representation (lo/hi are BinaryField64),
    /// while the C code uses flat (lo/hi are UInt64 halves of the polynomial rep).
    /// For the additive FFT, both representations share the same XOR-based structure,
    /// so the butterfly operations are compatible.
    public func forward(_ data: inout [UInt64]) {
        precondition(data.count == 2 * size)
        bt_afft_forward_128_iter(&data, size, basis)
    }

    public func inverse(_ data: inout [UInt64]) {
        precondition(data.count == 2 * size)
        bt_afft_inverse_128_iter(&data, size, basis)
    }
}

// MARK: - Binary Polynomial Commitment (Brakedown-style)

/// Binary polynomial commitment using expander-code linear codes over GF(2^64).
///
/// Follows the Brakedown approach but over binary tower fields:
/// 1. Arrange polynomial evaluations as a matrix (sqrt(n) x sqrt(n))
/// 2. Encode each row with an expander code (systematic + sparse redundancy)
/// 3. Hash columns to produce a Merkle commitment
///
/// Addition in binary fields is XOR (free), so encoding is cheap:
/// only the sparse GF(2^64) multiplications in the expander code cost anything.
public struct BinaryBrakedownCommitment {
    /// Merkle root hash (32 bytes)
    public let root: [UInt8]
    /// Matrix dimensions
    public let numRows: Int
    public let numCols: Int
    /// Encoded matrix for proof generation (prover retains this)
    public let encodedMatrix: [[BinaryField64]]
}

/// Binary expander code over GF(2^64).
///
/// Each redundancy symbol is a sparse linear combination of message symbols
/// using GF(2^64) coefficients from a pseudo-random expander graph.
/// The code is systematic: codeword = [message | redundancy].
public struct BinaryExpanderCode {
    public let messageLength: Int
    public let codewordLength: Int
    public let redundancyLength: Int
    public let degree: Int
    public let seed: UInt32

    /// Neighbor indices: neighbors[i * degree + d] = left vertex for right vertex i, edge d
    private let neighbors: [UInt32]
    /// GF(2^64) coefficients for each edge
    private let coefficients: [BinaryField64]

    public init(messageLength: Int, rateInverse: Int = 4, degree: Int = 10, seed: UInt32 = 0xB1A45) {
        precondition(messageLength > 0)
        precondition(rateInverse >= 2)
        let deg = min(degree, messageLength)
        self.messageLength = messageLength
        self.codewordLength = messageLength * rateInverse
        self.redundancyLength = codewordLength - messageLength
        self.degree = deg
        self.seed = seed

        var nbrs = [UInt32](repeating: 0, count: redundancyLength * deg)
        var coeffs = [BinaryField64](repeating: .zero, count: redundancyLength * deg)

        for i in 0..<redundancyLength {
            var selected = Set<UInt32>()
            selected.reserveCapacity(deg)
            var attempt: UInt32 = 0

            for d in 0..<deg {
                var leftIdx: UInt32
                repeat {
                    leftIdx = BinaryExpanderCode.prng(
                        seed: seed, i: UInt32(i), d: UInt32(d), attempt: attempt,
                        modulus: UInt32(messageLength)
                    )
                    attempt &+= 1
                } while selected.contains(leftIdx)
                selected.insert(leftIdx)
                nbrs[i * deg + d] = leftIdx

                // Generate a non-zero GF(2^64) coefficient
                var s = seed ^ 0xCAFECAFE
                s ^= UInt32(truncatingIfNeeded: i) &* 2654435761
                s ^= UInt32(truncatingIfNeeded: d) &* 2246822519
                s ^= s >> 16; s &*= 0x45d9f3b; s ^= s >> 16; s &*= 0x45d9f3b; s ^= s >> 16
                let val = UInt64(s) | 1  // ensure non-zero
                coeffs[i * deg + d] = BinaryField64(value: val)
            }
        }

        self.neighbors = nbrs
        self.coefficients = coeffs
    }

    private static func prng(seed: UInt32, i: UInt32, d: UInt32, attempt: UInt32, modulus: UInt32) -> UInt32 {
        var s = seed
        s ^= i &* 2654435761
        s ^= d &* 2246822519
        s ^= attempt &* 3266489917
        s ^= s >> 16; s &*= 0x45d9f3b
        s ^= s >> 16; s &*= 0x45d9f3b
        s ^= s >> 16
        return s % modulus
    }

    /// Encode a message: codeword = [message | redundancy].
    /// Redundancy symbols are sparse GF(2^64) dot products via the expander graph.
    public func encode(_ message: [BinaryField64]) -> [BinaryField64] {
        precondition(message.count == messageLength)
        var codeword = message
        codeword.reserveCapacity(codewordLength)

        for i in 0..<redundancyLength {
            var acc = BinaryField64.zero
            let base = i * degree
            for d in 0..<degree {
                let leftIdx = Int(neighbors[base + d])
                let coeff = coefficients[base + d]
                acc = acc + (coeff * message[leftIdx])
            }
            codeword.append(acc)
        }

        return codeword
    }

    /// Verify a codeword by recomputing redundancy from the message part.
    public func isValid(_ codeword: [BinaryField64]) -> Bool {
        guard codeword.count == codewordLength else { return false }
        let message = Array(codeword.prefix(messageLength))
        let expected = encode(message)
        for i in messageLength..<codewordLength {
            if expected[i].toUInt64 != codeword[i].toUInt64 { return false }
        }
        return true
    }
}

/// Commit to a vector of GF(2^64) elements using binary Brakedown.
///
/// The vector is reshaped into a matrix, rows are encoded with an expander code,
/// and columns are Merkle-hashed.
public func binaryBrakedownCommit(
    evaluations: [BinaryField64],
    rateInverse: Int = 4,
    degree: Int = 10
) -> BinaryBrakedownCommitment {
    let n = evaluations.count
    let numCols = max(1, Int(Double(n).squareRoot().rounded(.up)))
    let numRows = (n + numCols - 1) / numCols

    // Pad evaluations to fill the matrix
    var padded = evaluations
    let totalCells = numRows * numCols
    if padded.count < totalCells {
        padded.append(contentsOf: [BinaryField64](repeating: .zero, count: totalCells - padded.count))
    }

    // Create expander code for row encoding
    let code = BinaryExpanderCode(messageLength: numCols, rateInverse: rateInverse, degree: degree)

    // Encode each row
    var encodedMatrix = [[BinaryField64]]()
    encodedMatrix.reserveCapacity(numRows)
    for row in 0..<numRows {
        let start = row * numCols
        let messageRow = Array(padded[start..<start + numCols])
        encodedMatrix.append(code.encode(messageRow))
    }

    // Hash columns to produce Merkle leaves
    let numEncodedCols = code.codewordLength
    var columnHashes = [[UInt8]]()
    columnHashes.reserveCapacity(numEncodedCols)

    for col in 0..<numEncodedCols {
        var columnData = [UInt8]()
        columnData.reserveCapacity(numRows * 8)
        for row in 0..<numRows {
            var val = encodedMatrix[row][col].toUInt64
            withUnsafeBytes(of: &val) { columnData.append(contentsOf: $0) }
        }
        columnHashes.append(biniusHash(columnData))
    }

    let root = biniusMerkleRoot(leaves: columnHashes)

    return BinaryBrakedownCommitment(
        root: root,
        numRows: numRows,
        numCols: numCols,
        encodedMatrix: encodedMatrix
    )
}

/// Simple hash for binary commitment (XOR-fold + mix).
/// In production, replace with Keccak-256 or Poseidon2 from the Hash/ module.
private func biniusHash(_ data: [UInt8]) -> [UInt8] {
    var h = [UInt8](repeating: 0, count: 32)
    var offset = 0
    while offset < data.count {
        let end = min(offset + 32, data.count)
        for i in offset..<end {
            h[i - offset] ^= data[i]
        }
        for i in 0..<31 {
            h[i] ^= h[i + 1] &+ UInt8(truncatingIfNeeded: i)
        }
        h[31] ^= h[0]
        offset += 32
    }
    return h
}

/// Build a Merkle root from leaf hashes.
private func biniusMerkleRoot(leaves: [[UInt8]]) -> [UInt8] {
    guard !leaves.isEmpty else { return [UInt8](repeating: 0, count: 32) }
    if leaves.count == 1 { return leaves[0] }

    let n = 1 << Int(ceil(log2(Double(leaves.count))))
    var level = leaves
    while level.count < n {
        level.append([UInt8](repeating: 0, count: 32))
    }

    while level.count > 1 {
        var nextLevel = [[UInt8]]()
        nextLevel.reserveCapacity(level.count / 2)
        for i in stride(from: 0, to: level.count, by: 2) {
            let combined = level[i] + level[i + 1]
            nextLevel.append(biniusHash(combined))
        }
        level = nextLevel
    }
    return level[0]
}

// MARK: - Binary AIR (Algebraic Intermediate Representation)

/// Protocol for binary-field AIR constraint systems.
///
/// Like CircleAIR but over binary tower fields (BinaryField64).
/// Addition is XOR and multiplication uses PMULL, so constraint evaluation
/// is significantly cheaper than prime-field AIRs.
public protocol BinaryAIR {
    /// Number of trace columns
    var numColumns: Int { get }

    /// Log2 of trace length
    var logTraceLength: Int { get }

    /// Trace length (must be power of 2)
    var traceLength: Int { get }

    /// Number of constraints
    var numConstraints: Int { get }

    /// Maximum degree of each constraint
    var constraintDegrees: [Int] { get }

    /// Generate the execution trace: [column][row] of BinaryField64 elements
    func generateTrace() -> [[BinaryField64]]

    /// Evaluate all transition constraints at a single row.
    /// Returns array of constraint evaluations; all should be zero on a valid trace.
    func evaluateConstraints(current: [BinaryField64], next: [BinaryField64]) -> [BinaryField64]

    /// Boundary constraints: (column, row, expected value)
    var boundaryConstraints: [(column: Int, row: Int, value: BinaryField64)] { get }
}

extension BinaryAIR {
    public var traceLength: Int { 1 << logTraceLength }

    /// Verify a trace against all AIR constraints (CPU check).
    /// Returns nil if valid, or an error description string if invalid.
    public func verifyTrace(_ trace: [[BinaryField64]]) -> String? {
        let n = traceLength
        guard trace.count == numColumns else {
            return "Expected \(numColumns) columns, got \(trace.count)"
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == n else {
                return "Column \(ci): expected \(n) rows, got \(col.count)"
            }
        }

        // Check boundary constraints
        for bc in boundaryConstraints {
            guard bc.column < numColumns && bc.row < n else {
                return "Boundary constraint out of range: col=\(bc.column), row=\(bc.row)"
            }
            if trace[bc.column][bc.row].toUInt64 != bc.value.toUInt64 {
                return "Boundary constraint failed: col=\(bc.column), row=\(bc.row), " +
                       "expected=\(bc.value), got=\(trace[bc.column][bc.row])"
            }
        }

        // Check transition constraints on all rows except the last
        for i in 0..<(n - 1) {
            let current = (0..<numColumns).map { trace[$0][i] }
            let next = (0..<numColumns).map { trace[$0][i + 1] }
            let evals = evaluateConstraints(current: current, next: next)
            for (ci, ev) in evals.enumerated() {
                if !ev.isZero {
                    return "Transition constraint \(ci) failed at row \(i): eval=\(ev)"
                }
            }
        }

        return nil
    }
}

/// Generic binary AIR defined by closures (useful for testing and simple circuits).
public struct GenericBinaryAIR: BinaryAIR {
    public let numColumns: Int
    public let logTraceLength: Int
    public let numConstraints: Int
    public let constraintDegrees: [Int]
    public let boundaryConstraints: [(column: Int, row: Int, value: BinaryField64)]

    private let _generateTrace: () -> [[BinaryField64]]
    private let _evaluateConstraints: ([BinaryField64], [BinaryField64]) -> [BinaryField64]

    public init(
        numColumns: Int,
        logTraceLength: Int,
        constraintDegrees: [Int],
        boundaryConstraints: [(column: Int, row: Int, value: BinaryField64)],
        generateTrace: @escaping () -> [[BinaryField64]],
        evaluateConstraints: @escaping ([BinaryField64], [BinaryField64]) -> [BinaryField64]
    ) {
        self.numColumns = numColumns
        self.logTraceLength = logTraceLength
        self.numConstraints = constraintDegrees.count
        self.constraintDegrees = constraintDegrees
        self.boundaryConstraints = boundaryConstraints
        self._generateTrace = generateTrace
        self._evaluateConstraints = evaluateConstraints
    }

    public func generateTrace() -> [[BinaryField64]] { _generateTrace() }
    public func evaluateConstraints(current: [BinaryField64], next: [BinaryField64]) -> [BinaryField64] {
        _evaluateConstraints(current, next)
    }
}

// MARK: - Binary Constraint Evaluator

/// Evaluates binary AIR constraints and produces a composition polynomial.
///
/// Given a trace and a random challenge alpha in GF(2^64), computes the
/// random linear combination of all constraint polynomials:
///   C(x) = sum_i alpha^i * c_i(x)
///
/// This is the binary-field analogue of the composition polynomial in prime-field STARKs.
public struct BinaryConstraintEvaluator {

    /// Evaluate the composition polynomial at every row of the trace.
    ///
    /// Returns the evaluation of the random linear combination of constraints
    /// at each trace row (last row is set to zero since transition constraints don't apply).
    public static func evaluateComposition<A: BinaryAIR>(
        air: A,
        trace: [[BinaryField64]],
        alpha: BinaryField64
    ) -> [BinaryField64] {
        let n = air.traceLength
        var composition = [BinaryField64](repeating: .zero, count: n)

        for i in 0..<(n - 1) {
            let current = (0..<air.numColumns).map { trace[$0][i] }
            let next = (0..<air.numColumns).map { trace[$0][i + 1] }
            let constraintVals = air.evaluateConstraints(current: current, next: next)

            // Random linear combination: sum_j alpha^j * c_j
            var result = BinaryField64.zero
            var alphaPow = BinaryField64.one
            for cv in constraintVals {
                result = result + (alphaPow * cv)
                alphaPow = alphaPow * alpha
            }
            composition[i] = result
        }

        return composition
    }

    /// Compute quotient polynomial evaluations.
    ///
    /// For transition constraints, the vanishing polynomial of the transition domain
    /// is Z_T(x). The quotient Q(x) = C(x) / Z_T(x) must be a polynomial (not rational)
    /// for a valid trace.
    ///
    /// In binary fields, the vanishing polynomial of an additive subspace V is the
    /// linearized polynomial: prod_{v in V} (x + v) = x^|V| + x.
    public static func evaluateQuotient<A: BinaryAIR>(
        air: A,
        trace: [[BinaryField64]],
        alpha: BinaryField64,
        domainGenerator: BinaryField64
    ) -> [BinaryField64] {
        let composition = evaluateComposition(air: air, trace: trace, alpha: alpha)
        let n = air.traceLength
        var quotient = [BinaryField64](repeating: .zero, count: n)

        // For each domain point, divide by the vanishing polynomial evaluation.
        // The vanishing polynomial of the additive subspace is V(x) = x^n + x.
        // For transition domain (all but last row), we need V_T(x) = V(x) / (x + omega^{n-1}).
        var domainPow = BinaryField64.one
        for i in 0..<n {
            if !composition[i].isZero {
                // V(x) = x^n + x = domainPow^n + domainPow
                // Since domainPow is in a domain of size n, domainPow^n cycles.
                // Simplified: use x^2 + x (the subspace polynomial s_1) as a stand-in
                // for the vanishing set indicator at this level.
                let vanishing = (domainPow * domainPow) + domainPow
                if !vanishing.isZero {
                    quotient[i] = composition[i] * vanishing.inverse()
                }
            }
            domainPow = domainPow * domainGenerator
        }

        return quotient
    }
}

// MARK: - Example: Binary XOR-Chain AIR

/// A simple example AIR over binary fields: XOR chain with accumulation.
///
/// Trace columns: [value, accumulator]
/// Transition constraints:
///   - value' = value XOR stepConstant
///   - accumulator' = accumulator XOR value
///
/// This demonstrates that binary field constraints are extremely cheap:
/// addition is free (XOR) and these particular constraints require no multiplications.
public struct BinaryXORChainAIR: BinaryAIR {
    public let numColumns: Int = 2
    public let logTraceLength: Int
    public let numConstraints: Int = 2
    public let constraintDegrees: [Int] = [1, 1]

    /// The XOR constant applied to advance the value each step
    public let stepConstant: BinaryField64
    /// Initial value
    public let initialValue: BinaryField64

    public var boundaryConstraints: [(column: Int, row: Int, value: BinaryField64)] {
        [
            (column: 0, row: 0, value: initialValue),
            (column: 1, row: 0, value: .zero),
        ]
    }

    public init(logTraceLength: Int,
                stepConstant: BinaryField64 = BinaryField64(value: 0x1234567890ABCDEF),
                initialValue: BinaryField64 = BinaryField64(value: 1)) {
        self.logTraceLength = logTraceLength
        self.stepConstant = stepConstant
        self.initialValue = initialValue
    }

    public func generateTrace() -> [[BinaryField64]] {
        let n = traceLength
        var values = [BinaryField64](repeating: .zero, count: n)
        var accum = [BinaryField64](repeating: .zero, count: n)

        values[0] = initialValue
        accum[0] = .zero

        for i in 1..<n {
            values[i] = values[i - 1] + stepConstant   // XOR with constant
            accum[i] = accum[i - 1] + values[i - 1]    // accumulate via XOR
        }

        return [values, accum]
    }

    public func evaluateConstraints(current: [BinaryField64], next: [BinaryField64]) -> [BinaryField64] {
        // Constraint 0: next_value + current_value + stepConstant = 0 (in char 2)
        let c0 = next[0] + current[0] + stepConstant

        // Constraint 1: next_accum + current_accum + current_value = 0
        let c1 = next[1] + current[1] + current[0]

        return [c0, c1]
    }
}

// MARK: - Example: Binary Multiplicative Constraint AIR

/// Example AIR with multiplicative constraints (exercises PMULL path).
///
/// Trace columns: [a, b, product]
/// Transition constraints:
///   - a' = a^2 (Frobenius endomorphism / squaring)
///   - b' = b XOR bStep
///   - product = a * b (verified each row)
///
/// Degree-2 constraints over binary tower fields.
public struct BinaryMulAIR: BinaryAIR {
    public let numColumns: Int = 3
    public let logTraceLength: Int
    public let numConstraints: Int = 3
    public let constraintDegrees: [Int] = [2, 2, 1]

    public let initialA: BinaryField64
    public let initialB: BinaryField64
    public let bStep: BinaryField64

    public var boundaryConstraints: [(column: Int, row: Int, value: BinaryField64)] {
        [
            (column: 0, row: 0, value: initialA),
            (column: 1, row: 0, value: initialB),
            (column: 2, row: 0, value: initialA * initialB),
        ]
    }

    public init(logTraceLength: Int,
                initialA: BinaryField64 = BinaryField64(value: 3),
                initialB: BinaryField64 = BinaryField64(value: 7),
                bStep: BinaryField64 = BinaryField64(value: 0xDEAD)) {
        self.logTraceLength = logTraceLength
        self.initialA = initialA
        self.initialB = initialB
        self.bStep = bStep
    }

    public func generateTrace() -> [[BinaryField64]] {
        let n = traceLength
        var colA = [BinaryField64](repeating: .zero, count: n)
        var colB = [BinaryField64](repeating: .zero, count: n)
        var colP = [BinaryField64](repeating: .zero, count: n)

        colA[0] = initialA
        colB[0] = initialB
        colP[0] = initialA * initialB

        for i in 1..<n {
            colA[i] = colA[i - 1].squared()      // Frobenius
            colB[i] = colB[i - 1] + bStep         // XOR step
            colP[i] = colA[i] * colB[i]           // product
        }

        return [colA, colB, colP]
    }

    public func evaluateConstraints(current: [BinaryField64], next: [BinaryField64]) -> [BinaryField64] {
        let a = current[0], b = current[1]
        let nextA = next[0], nextB = next[1], nextP = next[2]

        // Constraint 0: next_a + a^2 = 0
        let c0 = nextA + a.squared()

        // Constraint 1: next_p + next_a * next_b = 0
        let c1 = nextP + (nextA * nextB)

        // Constraint 2: next_b + b + bStep = 0
        let c2 = nextB + b + bStep

        return [c0, c1, c2]
    }
}
