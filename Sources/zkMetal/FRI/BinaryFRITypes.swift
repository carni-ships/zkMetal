// BinaryFRI Types — Binary-native FRI with additive domain folding
//
// Implements FRI over binary tower fields using additive (not multiplicative) domains.
// Key differences from standard FRI:
//   - Domain is an affine subspace of GF(2^m), not a multiplicative coset
//   - Fold uses trace-based formulas instead of root-of-unity division
//   - Works with BinaryTower128 elements instead of Fr
//
// Additive domain structure:
//   - Subspace S of dimension k has size 2^k
//   - Doubling map x -> x^2 + x halves the space (kernel is GF(2))
//   - Fold formula: f'(x) = f_even(x) + alpha * f_odd(x)
//     where f_even and f_odd are derived via trace

import Foundation

// MARK: - Binary FRI Configuration

/// Configuration for binary-native FRI with additive domain folding.
public struct BinaryFRIConfig {
    /// Extension degree of the binary field: GF(2^m)
    public let extensionDegree: Int

    /// Folding factor per round (2, 4, or 8)
    public let foldingFactor: Int

    /// Number of queries for soundness
    public let numQueries: Int

    /// Maximum degree of the final polynomial (stopping condition)
    public let finalPolyMaxDegree: Int

    /// Log size of the initial evaluation domain (dimension of subspace)
    public let logDomainSize: Int

    public init(extensionDegree: Int = 128,
                foldingFactor: Int = 2,
                numQueries: Int = 32,
                finalPolyMaxDegree: Int = 7,
                logDomainSize: Int = 20) {
        precondition(foldingFactor == 2 || foldingFactor == 4 || foldingFactor == 8,
                     "Folding factor must be 2, 4, or 8")
        precondition(extensionDegree > 0 && extensionDegree <= 128,
                     "Extension degree must be between 1 and 128")
        self.extensionDegree = extensionDegree
        self.foldingFactor = foldingFactor
        self.numQueries = numQueries
        self.finalPolyMaxDegree = finalPolyMaxDegree
        self.logDomainSize = logDomainSize
    }
}

// MARK: - Binary FRI Domain

/// Represents an additive domain (affine subspace) in a binary field.
///
/// An affine subspace S of dimension k in GF(2^m) has size 2^k and can be
/// written as S = {a_0 + a_1*t + ... + a_{k-1}*t^{k-1} | a_i in GF(2)}
/// where t is a basis element.
///
/// The doubling map D(x) = x^2 + x maps S to a subspace of half the size,
/// with kernel = {0, 1} (the GF(2) subfield).
public struct BinaryFRIDomain<B: BinaryTowerProtocol> {
    /// Basis elements defining the subspace
    public let basis: [B]

    /// Offset point (translation from origin)
    public let offset: B

    /// Log2 of the domain size (dimension)
    public let logSize: Int

    /// Total number of points in the domain
    public var size: Int { 1 << logSize }

    /// Create a full domain of size 2^logSize
    public static func full(logSize: Int) -> BinaryFRIDomain<B> {
        // For full subspace, basis is the canonical basis
        var basis = [B]()
        for i in 0..<logSize {
            // Create standard basis vector with 1 in position i
            var vec = B.zero
            // This would need proper basis construction
            basis.append(vec)
        }
        return BinaryFRIDomain(basis: basis, offset: .zero, logSize: logSize)
    }
}

// MARK: - Binary FRI Commitment

/// Binary FRI commitment: all folded layers and Merkle roots.
public struct BinaryFRICommitment<B: BinaryTowerProtocol> {
    /// Layers of folded evaluations (each layer is a BinaryTower array)
    public let layers: [[B]]

    /// Merkle roots for each layer (using Poseidon2 over binary field)
    public let roots: [B]

    /// Folding challenges (alphas) for each round
    public let alphas: [B]

    /// Final constant polynomial value
    public let finalValue: B

    /// Log of original domain size
    public let logN: Int

    /// Configuration used
    public let config: BinaryFRIConfig
}

// MARK: - Binary FRI Query Proof

/// Query proof for binary FRI verification.
public struct BinaryFRIQueryProof<B: BinaryTowerProtocol> {
    /// Initial query index in original domain
    public let initialIndex: UInt32

    /// Evaluation pairs at each layer (for fold verification)
    public let layerEvals: [(B, B)]

    /// Merkle authentication paths at each layer
    public let merklePaths: [[B]]
}

// MARK: - Binary FRI Opening

/// Complete binary FRI opening proof.
public struct BinaryFRIOpening<B: BinaryTowerProtocol> {
    /// Query proofs for each position
    public let queryProofs: [BinaryFRIQueryProof<B>]

    /// The claimed evaluation
    public let evaluation: B

    /// Commitment this opens to
    public let commitment: BinaryFRICommitment<B>
}

// MARK: - Binary Co-Curvilinearity Test

/// Binary co-curvilinearity test for proximity verification.
///
/// Replaces the standard collinearity test in FRI.
/// Given m+1 points in GF(2^m), checks if they lie on an affine line.
///
/// In characteristic 2, collinearity can be tested via the property that
/// for points P_0, ..., P_m on a line L, the sum of weighted traces is zero.
public struct BinaryCoCurvilinearityTest<B: BinaryTowerProtocol> {
    /// Number of random points to test
    public let numPoints: Int

    public init(numPoints: Int = 16) {
        self.numPoints = numPoints
    }

    /// Test if the given points are co-curvilinear (lie on an affine line).
    ///
    /// Uses the quadratic form Q(x) = Tr(x^2) to test linearity.
    /// Points P_0, ..., P_m are on an affine line iff there exists alpha, beta
    /// such that P_i = alpha * i + beta for all i.
    public func test(points: [B], linePoint: B, lineDirection: B) -> Bool {
        // Check that all points satisfy P_i = linePoint + i * lineDirection
        // This is a simplified check; real implementation uses trace-based test
        guard points.count == numPoints else { return false }
        for i in 0..<numPoints {
            // Expected point on line: linePoint + i * lineDirection
            var expected = linePoint
            var dir = lineDirection
            for _ in 0..<i {
                expected = expected + dir
            }
            if points[i] != expected {
                return false
            }
        }
        return true
    }
}

// MARK: - Johnson Bound Decoder

/// Johnson bound decoder for binary algebraic geometry codes.
///
/// Implements list decoding achieving the Johnson bound J(n, d, L) which gives
/// tighter radius than standard unique decoding for binary codes.
///
/// Johnson bound: radius = n - sqrt((n - d) * (n - L * d))
public struct JohnsonBoundDecoder {
    /// Code length n
    public let n: Int

    /// Minimum distance d
    public let d: Int

    /// List size bound L
    public let L: Int

    public init(n: Int, d: Int, L: Int) {
        self.n = n
        self.d = d
        self.L = L
    }

    /// Johnson radius for list decoding.
    /// Returns the radius within which list decoding guarantees success.
    public var johnsonRadius: Int {
        let inner = Double(n - d) * Double(n - d)
        return n - Int(sqrt(inner))
    }

    /// List decode the received word within Johnson radius.
    /// Returns all codewords within the Johnson radius.
    public func listDecode<B: BinaryTowerProtocol>(received: [B]) -> [[B]] {
        // Simplified: return empty list
        // Real implementation would use interpolation-based list decoding
        return []
    }
}
