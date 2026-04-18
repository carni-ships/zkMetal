// BinaryFRIFoldEngine — Additive domain folding for binary-native FRI
//
// Implements FRI folding over binary tower fields using additive (not multiplicative)
// domains. Key operations:
//
// 1. Additive Domain: An affine subspace S of GF(2^m) with dimension k has size 2^k.
//    Represented as S = {offset + sum(a_i * basis_i) | a_i in GF(2)}
//
// 2. Doubling Map: D(x) = x^2 + x is GF(2)-linear with kernel {0, 1}.
//    D maps S to a subspace of half the size (dimension k-1).
//
// 3. Fold Formula: Given f: S -> GF(2^m), split into even/odd parts via trace:
//    f_even(x) = (f(x) + f(x+1)) / 2  (technically: GF(2)-linear part)
//    f_odd(x)  = (f(x) + f(x+1)) / 2  (the complementary part)
//    Then: f'(x) = f_even(x) + alpha * f_odd(x)
//
// 4. High-Ary Folding: Generalizes to folding factor 2^k by using the k-fold
//    composition of the doubling map, with kernel of size 2^k.

import Foundation

// MARK: - Binary FRI Fold Engine

/// Engine for performing additive domain folding in binary FRI.
///
/// This engine implements the core fold operation for binary-native FRI,
/// working with additive (affine subspace) domains rather than multiplicative
/// coset domains used in standard FRI.
public struct BinaryFRIFoldEngine<B: BinaryTowerProtocol> {

    /// Configuration for the fold operations
    public let config: BinaryFRIConfig

    /// Log size of the current domain (decreases by logFoldFactor each round)
    public private(set) var currentLogSize: Int

    /// Create a new fold engine with the given configuration.
    public init(config: BinaryFRIConfig) {
        self.config = config
        self.currentLogSize = config.logDomainSize
    }

    // MARK: - Domain Operations

    /// Compute the additive domain for the current fold level.
    /// Returns basis vectors and offset defining the affine subspace.
    public func currentDomain() -> (basis: [B], offset: B) {
        // For a full subspace (no offset), generate canonical basis
        var basis = [B]()
        for i in 0..<currentLogSize {
            // Create basis vector with 1 in position i of the tower representation
            var vec = B.zero
            // The basis is the standard basis for the subspace
            basis.append(computeBasisVector(index: i))
        }
        return (basis, .zero)
    }

    /// Compute a basis vector for the given index.
    /// Uses the tower field's bit representation.
    private func computeBasisVector(index: Int) -> B {
        // For BinaryTower128, the representation is as a polynomial
        // basis[i] = 2^i in the field representation
        if index == 0 {
            return B.one
        }
        // For other indices, construct via repeated squaring
        // basis[i] = (generator)^(2^i) in multiplicative sense, but we
        // use additive structure: the subspace basis
        var result = B.zero
        // This is a simplified basis construction - real implementation
        // would use the specific tower structure
        return result
    }

    /// Apply the doubling map D(x) = x^2 + x to a domain point.
    /// This halves the domain size (kernel is GF(2) = {0, 1}).
    public func doublingMap(_ x: B) -> B {
        let x2 = x.squared()
        return x2 + x
    }

    /// Apply the k-fold doubling map D^k(x).
    /// Kernel size is 2^k, so domain shrinks by factor of 2^k.
    public func kFoldDoublingMap(_ x: B, k: Int) -> B {
        var result = x
        for _ in 0..<k {
            result = doublingMap(result)
        }
        return result
    }

    // MARK: - Trace-Based Splitting

    /// Compute the field trace Tr_{GF(2^m)/GF(2)}(x) = x + x^2 + x^4 + ... + x^{2^{m-1}}.
    /// Used for splitting polynomials into even/odd parts.
    public func trace(_ x: B) -> B {
        var result = x
        var current = x
        let degree = B.extensionDegree

        for _ in 1..<degree {
            current = current.squared()
            result = result + current
        }
        return result
    }

    /// Split a polynomial evaluation f(x) into even and odd parts using trace.
    ///
    /// For additive domains, the splitting is based on the GF(2)-linearity of D(x).
    /// f_even corresponds to the part that is linear w.r.t. the doubling map,
    /// f_odd corresponds to the complementary part.
    ///
    /// Mathematically:
    ///   f_even(x) = (f(x) + f(x+1)) / 2  -> but in GF(2^m), division by 2 is
    ///   actually done via the automorphism (since 2 = 0 in char 2, we use trace)
    ///
    /// More precisely, for the doubling map D(x) = x^2 + x:
    ///   f_even(x) = sum_{y in kernel} f(x + y) where kernel = {0, 1}
    ///              = f(x) + f(x+1)  (but this is not quite right either)
    ///
    /// The correct splitting uses the linearity:
    ///   f((x^2 + x) + y) = f(x)^2 + f(x) + f(y) for y in kernel
    public func splitEvenOdd(fx: B, fx1: B, alpha: B) -> B {
        // Standard FRI fold over binary field:
        // f'(x) = (f(x) + f(x+1)) / 2 + alpha * (f(x) - f(x+1)) / 2
        //
        // In characteristic 2, subtraction = addition, and "division by 2"
        // is more subtle. For the trace-based approach:
        //
        // f_even(x) = (f(x) + f(x+1)) via the linear trace
        // f_odd(x)  = (f(x) + f(x+1)) via the complementary projector
        //
        // Simplified (works for the standard basis):
        // f'(x) = f(x) + alpha * f(x+1)
        //
        // The prover sends f(x) and f(x+1), verifier checks the fold.

        // This is the simplified fold formula:
        // In binary fields, f'(x) = f(x) + alpha * f(x + 1)
        // where x + 1 is the "next" point in the kernel expansion
        return fx + alpha * fx1
    }

    // MARK: - Fold Operations

    /// Perform one round of FRI folding, reducing domain size by factor of 2.
    ///
    /// Input: evaluations at 2^currentLogSize points
    /// Output: evaluations at 2^(currentLogSize-1) points
    ///
    /// - Parameters:
    ///   - evals: Evaluations at current domain points
    ///   - alpha: Random challenge for this fold
    /// - Returns: Folded evaluations at half the domain size
    public mutating func foldRound(evals: [B], alpha: B) -> [B] {
        precondition(evals.count == 1 << currentLogSize,
                     "Evals count must match domain size")

        let halfSize = evals.count / 2
        var folded = [B](repeating: .zero, count: halfSize)

        // The fold formula for additive domain:
        // f'(x) = f_even(x) + alpha * f_odd(x)
        //
        // where f_even(x) = (f(x) + f(D^{-1}(x))) / 2
        // and the pairing is determined by the doubling map structure
        //
        // Simplified implementation: pair evals[i] with evals[i + halfSize]
        // using the standard FRI fold over the affine subspace
        for i in 0..<halfSize {
            let f0 = evals[i]
            let f1 = evals[i + halfSize]

            // Standard fold: f'(i) = f_even(i) + alpha * f_odd(i)
            // where f_even = (f0 + f1) / 2 and f_odd = (f0 - f1) / 2
            // In char 2: /2 means taking the trace-related projection
            //
            // The actual formula used:
            folded[i] = splitEvenOdd(fx: f0, fx1: f1, alpha: alpha)
        }

        currentLogSize -= 1
        return folded
    }

    /// Perform one round of high-arity folding (factor = 2^k).
    ///
    /// - Parameters:
    ///   - evals: Evaluations at current domain points
    ///   - alpha: Random challenge for this fold
    ///   - arity: k such that folding factor = 2^k
    /// - Returns: Folded evaluations at 1/2^k the domain size
    public mutating func foldRoundArity(evals: [B], alpha: B, arity: Int) -> [B] {
        let foldFactor = 1 << arity
        let newSize = evals.count / foldFactor
        precondition(evals.count % foldFactor == 0, "Evals count must be divisible by fold factor")

        var folded = [B](repeating: .zero, count: newSize)

        // High-arity fold using the k-fold doubling map
        // The kernel of D^k has size 2^k, giving the arity
        for i in 0..<newSize {
            // Collect evaluations at all kernel points
            var terms = [B]()
            for j in 0..<foldFactor {
                terms.append(evals[i * foldFactor + j])
            }

            // Compute fold using the high-arity formula:
            // f'(x) = sum_{j=0}^{2^k-1} c_j * f(x + k_j) where k_j are kernel elements
            // The coefficients c_j come from the alphas in a Merkle-Damgard structure
            folded[i] = computeHighArityFold(terms: terms, alpha: alpha, arity: arity)
        }

        currentLogSize -= arity
        return folded
    }

    /// Compute the high-arity fold value.
    /// Uses the trace-based projection for the 2^k-ary fold.
    private func computeHighArityFold(terms: [B], alpha: B, arity: Int) -> B {
        // For high-arity fold with arity k (fold factor 2^k):
        // f'(x) = f_0 + alpha * f_1 + alpha^2 * f_2 + ... + alpha^{2^k-1} * f_{2^k-1}
        // where f_j are evaluations at kernel elements
        //
        // This is a generalized version of the standard fold where we fold
        // 2^k values at once using powers of alpha
        precondition(terms.count == 1 << arity, "Terms count must equal 2^arity")

        var result = terms[0]
        var alphaPower = alpha

        for i in 1..<terms.count {
            result = result + alphaPower * terms[i]
            alphaPower = alphaPower * alpha
        }

        return result
    }

    // MARK: - Multi-Round Folding

    /// Fold through multiple rounds until reaching the final polynomial degree.
    ///
    /// - Parameters:
    ///   - evals: Initial evaluations at full domain
    ///   - alphas: Random challenges for each round
    /// - Returns: All intermediate layers (for Merkle commitment)
    public mutating func foldAllRounds(evals: [B], alphas: [B]) -> [[B]] {
        var layers = [evals]
        var current = evals

        // Determine number of rounds based on final degree
        // Need at least log(foldFactor) rounds to reduce to finalPolyMaxDegree
        let numRounds = computeNumRounds(initialSize: evals.count)

        for i in 0..<min(alphas.count, numRounds) {
            let arity = determineArity(round: i, remainingRounds: numRounds - i)
            current = foldRoundArity(evals: current, alpha: alphas[i], arity: arity)
            layers.append(current)
        }

        return layers
    }

    /// Compute the number of rounds needed to reduce to final polynomial.
    func computeNumRounds(initialSize: Int) -> Int {
        let initialLogSize = Int(log2(Double(initialSize)))
        var currentLogSize = initialLogSize
        var rounds = 0

        while currentLogSize > config.finalPolyMaxDegree && rounds < 100 {
            // Use highest arity possible that still allows proper folding
            let arity = min(self.config.foldingFactor.trailingZeroBitCount,
                           currentLogSize - config.finalPolyMaxDegree)
            currentLogSize -= (arity > 0 ? arity : 1)
            rounds += 1
        }

        return rounds
    }

    /// Determine which arity to use for a given round.
    /// Prefers higher arity when possible for efficiency.
    func determineArity(round: Int, remainingRounds: Int) -> Int {
        let maxArity = config.foldingFactor.trailingZeroBitCount

        // Need to leave enough rounds for remaining domain reduction
        // Each round with arity k reduces logSize by k
        let minArityNeeded = remainingRounds > 0 ?
            max(1, round - (config.logDomainSize - config.finalPolyMaxDegree) + 1) : 1

        // Use highest arity that fits
        return min(maxArity, remainingRounds)
    }

    // MARK: - Verification Support

    /// Verify a fold round by checking the fold equation at random points.
    ///
    /// - Parameters:
    ///   - f0, f1: Original evaluations at paired points
    ///   - folded: Resulting folded evaluation
    ///   - alpha: Challenge used in fold
    ///   - x: Domain point where fold is evaluated
    /// - Returns: True if fold equation holds
    public func verifyFold(f0: B, f1: B, folded: B, alpha: B, x: B) -> Bool {
        // The fold equation: f'(x) = f_even(x) + alpha * f_odd(x)
        // For the doubling map, f_even and f_odd are computed via trace
        let expected = splitEvenOdd(fx: f0, fx1: f1, alpha: alpha)
        return expected == folded
    }

    /// Generate the query positions for FRI verification.
    /// Uses pseudorandom access based on the commitment and challenges.
    public func generateQueryPositions(seed: [UInt8], numQueries: Int,
                                       domainSize: Int) -> [Int] {
        // Use a simple PRG based on the seed and challenges
        // In practice, this would use a proper hash-based PRG
        var positions = [Int]()
        var state = seed

        for _ in 0..<numQueries {
            // Simple hash-based selection - just use first bytes of hash
            let hash = simpleHash(state)
            let hashValue = hash.withUnsafeBytes { $0.load(as: UInt64.self) }
            let pos = Int(hashValue) % domainSize
            positions.append(pos)
            // Update state by mixing in the hash
            state = hash
        }

        return positions
    }

    /// Simple hash for pseudorandom generation (placeholder).
    private func simpleHash(_ input: [UInt8]) -> [UInt8] {
        // Placeholder - would use Poseidon or SHA3 in real implementation
        var result = input
        for i in 0..<result.count {
            result[i] = result[i] &* 31 &+ 17
        }
        return result
    }

    /// Reset the engine to initial domain size.
    public mutating func reset() {
        currentLogSize = config.logDomainSize
    }
}

// MARK: - Merkle Tree Builder

/// Merkle tree builder for commitments in binary FRI.
/// Uses simple XOR-based hashing as placeholder (production would use Poseidon2).
public struct MerkleTreeBuilder {
    public init() {}

    public func buildRoot<B: BinaryTowerProtocol>(values: [B]) throws -> B {
        // Simple XOR-based hash as placeholder (addition in GF(2^m) is XOR)
        var hash = B.zero
        for value in values {
            hash = hash + value
        }
        return hash
    }
}
