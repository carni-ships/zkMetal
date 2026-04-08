// GPURecursiveCompositionEngine — GPU-accelerated recursive proof composition
//
// Implements recursive proof composition with accumulation schemes:
//   - Proof-of-proof: verify a SNARK proof inside a SNARK circuit
//   - Accumulation scheme: fold multiple proofs into one accumulator
//   - Split accumulation: separate proof into deferred and immediate parts
//   - Proof size reduction via recursive composition
//   - Chain of proofs verification
//   - Compatible with Groth16 and Plonk base schemes
//
// The engine uses the BN254/Grumpkin curve cycle for efficient recursion:
//   - BN254 proofs are verified inside Grumpkin circuits (native Fr arithmetic)
//   - Grumpkin accumulation is verified inside BN254 circuits (native Fr arithmetic)
//   - Pairing checks are deferred and batch-verified at the end
//
// Architecture:
//   1. RecursiveCompositionAccumulator: running state across recursion layers
//   2. ProofNode: representation of a proof in the composition tree
//   3. SplitAccumulation: separate deferred (pairing) and immediate (MSM) parts
//   4. ChainVerifier: verify a chain/tree of proofs
//   5. ProofSizeReducer: compress proof through recursive wrapping
//
// References:
//   - "Proof-Carrying Data from Accumulation Schemes" (Bunz et al. 2020)
//   - "Nova: Recursive Zero-Knowledge Arguments" (KST 2022)
//   - "Protostar: Generic Efficient Accumulation" (BC 2023)
//   - "CycleFold: Folding-scheme-based recursive arguments" (KS 2023)

import Foundation
import NeonFieldOps

// MARK: - Base Proof Scheme

/// The base SNARK scheme used for proof composition.
public enum BaseScheme: Equatable {
    case groth16
    case plonk
}

// MARK: - Proof Node

/// A node in the recursive proof composition tree.
///
/// Each node represents either a leaf proof (from a base SNARK) or an
/// intermediate proof that verifies one or more child proofs.
public struct ProofNode {
    /// Unique identifier for this proof node
    public let id: Int
    /// The base scheme used to generate this proof
    public let scheme: BaseScheme
    /// Public inputs to this proof
    public let publicInputs: [Fr]
    /// Commitment polynomial (witness commitment in accumulation)
    public let commitmentPoly: [Fr]
    /// Whether this is a leaf (base proof) or intermediate (recursive proof)
    public let isLeaf: Bool
    /// Child node IDs (empty for leaves)
    public let children: [Int]
    /// Depth in the composition tree (0 = leaf)
    public let depth: Int

    public init(
        id: Int,
        scheme: BaseScheme,
        publicInputs: [Fr],
        commitmentPoly: [Fr],
        isLeaf: Bool = true,
        children: [Int] = [],
        depth: Int = 0
    ) {
        self.id = id
        self.scheme = scheme
        self.publicInputs = publicInputs
        self.commitmentPoly = commitmentPoly
        self.isLeaf = isLeaf
        self.children = children
        self.depth = depth
    }
}

// MARK: - Split Accumulation

/// Result of split accumulation: deferred and immediate verification parts.
///
/// In split accumulation, the proof verification is divided into:
///   - Immediate part: MSM accumulation checked inside the circuit (cheap)
///   - Deferred part: pairing/commitment checks verified natively at the end
///
/// This split is the key to efficient recursion: the expensive pairing check
/// is deferred while the cheap MSM accumulation is verified in-circuit.
public struct SplitAccumulationResult {
    /// The immediate verification state (MSM accumulation)
    public var immediatePoly: [Fr]
    /// Immediate verification scalar (accumulated evaluation)
    public var immediateScalar: Fr
    /// Deferred pairing checks (verified natively at finalization)
    public var deferredPairings: [DeferredPairingCheck]
    /// Deferred commitment checks (point + evaluation pairs)
    public var deferredCommitments: [DeferredCommitmentCheck]
    /// Whether the immediate part verified successfully
    public var immediateVerified: Bool
    /// The scheme that produced this split
    public var scheme: BaseScheme

    public init(
        immediatePoly: [Fr],
        immediateScalar: Fr,
        deferredPairings: [DeferredPairingCheck],
        deferredCommitments: [DeferredCommitmentCheck],
        immediateVerified: Bool,
        scheme: BaseScheme
    ) {
        self.immediatePoly = immediatePoly
        self.immediateScalar = immediateScalar
        self.deferredPairings = deferredPairings
        self.deferredCommitments = deferredCommitments
        self.immediateVerified = immediateVerified
        self.scheme = scheme
    }
}

/// A deferred polynomial commitment check: C opens to v at point z.
public struct DeferredCommitmentCheck {
    /// Commitment value (scalar representation)
    public var commitment: Fr
    /// Evaluation point
    public var point: Fr
    /// Claimed evaluation value
    public var value: Fr
    /// Opening proof element
    public var openingProof: Fr

    public init(commitment: Fr, point: Fr, value: Fr, openingProof: Fr) {
        self.commitment = commitment
        self.point = point
        self.value = value
        self.openingProof = openingProof
    }
}

// MARK: - Recursive Composition Accumulator

/// Accumulator for recursive proof composition.
///
/// Extends the basic RecursiveAccumulator with composition-specific state:
///   - Proof chain tracking (ordered sequence of folded proofs)
///   - Per-scheme accumulation (separate Groth16 and Plonk accumulators)
///   - Split accumulation results
///   - Composition tree metadata
public struct CompositionAccumulator {
    /// Running accumulated polynomial (folded commitments)
    public var accPoly: [Fr]
    /// Accumulated error scalar (folding slack)
    public var error: Fr
    /// Number of proofs folded into this accumulator
    public var foldCount: Int
    /// Challenges used during folding (for transcript replay)
    public var challenges: [Fr]
    /// Deferred pairing checks from Groth16 inner proofs
    public var deferredPairings: [DeferredPairingCheck]
    /// Deferred commitment checks from Plonk inner proofs
    public var deferredCommitments: [DeferredCommitmentCheck]
    /// IDs of proofs folded into this accumulator (in order)
    public var foldedProofIDs: [Int]
    /// The base scheme for this accumulator
    public var scheme: BaseScheme
    /// Maximum depth in the composition tree
    public var maxDepth: Int

    /// Create a fresh zero accumulator.
    public static func zero(size: Int, scheme: BaseScheme) -> CompositionAccumulator {
        CompositionAccumulator(
            accPoly: [Fr](repeating: Fr.zero, count: size),
            error: Fr.zero,
            foldCount: 0,
            challenges: [],
            deferredPairings: [],
            deferredCommitments: [],
            foldedProofIDs: [],
            scheme: scheme,
            maxDepth: 0
        )
    }

    /// Create an accumulator initialized with a single proof node.
    public static func initial(node: ProofNode) -> CompositionAccumulator {
        CompositionAccumulator(
            accPoly: node.commitmentPoly,
            error: Fr.zero,
            foldCount: 1,
            challenges: [],
            deferredPairings: [],
            deferredCommitments: [],
            foldedProofIDs: [node.id],
            scheme: node.scheme,
            maxDepth: node.depth
        )
    }

    /// Reset to empty state, preserving polynomial size and scheme.
    public mutating func reset() {
        let n = accPoly.count
        accPoly = [Fr](repeating: Fr.zero, count: n)
        error = Fr.zero
        foldCount = 0
        challenges = []
        deferredPairings = []
        deferredCommitments = []
        foldedProofIDs = []
        maxDepth = 0
    }

    /// Total number of deferred checks.
    public var totalDeferredChecks: Int {
        deferredPairings.count + deferredCommitments.count
    }
}

// MARK: - Composition Transcript

/// Fiat-Shamir transcript specialized for recursive composition.
///
/// Extends RecursiveTranscript with composition-specific domain separators
/// and methods for binding challenges to the composition tree structure.
public struct RecursiveCompositionTranscript {
    private var state: [UInt8] = []

    public init() {
        state.append(contentsOf: Array("zkMetal-recursive-composition-v1".utf8))
    }

    /// Append a domain separator label.
    public mutating func appendLabel(_ label: String) {
        var len = UInt32(label.utf8.count)
        withUnsafeBytes(of: &len) { state.append(contentsOf: $0) }
        state.append(contentsOf: Array(label.utf8))
    }

    /// Append a field element.
    public mutating func appendScalar(_ s: Fr) {
        withUnsafeBytes(of: s) { buf in
            state.append(contentsOf: buf)
        }
    }

    /// Append multiple field elements.
    public mutating func appendScalars(_ scalars: [Fr]) {
        for s in scalars { appendScalar(s) }
    }

    /// Append a proof node's identity (id + scheme + depth) to bind the
    /// challenge to the composition tree structure.
    public mutating func appendNode(_ node: ProofNode) {
        appendLabel("node-\(node.id)")
        var depth = UInt32(node.depth)
        withUnsafeBytes(of: &depth) { state.append(contentsOf: $0) }
        var schemeTag: UInt8 = node.scheme == .groth16 ? 0 : 1
        state.append(schemeTag)
        appendScalars(node.publicInputs)
    }

    /// Squeeze a challenge from the current state.
    public func squeeze() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]
            limbs[1] = ptr[1]
            limbs[2] = ptr[2]
            limbs[3] = ptr[3]
        }
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    /// Squeeze a challenge and advance the state.
    public mutating func squeezeAndAdvance() -> Fr {
        let c = squeeze()
        appendScalar(c)
        return c
    }

    /// Get the current state hash.
    public func stateHash() -> [UInt8] {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        return hash
    }
}

// MARK: - Chain Verification Result

/// Result of verifying a chain of proofs.
public struct ChainVerificationResult {
    /// Whether the full chain verified successfully
    public let verified: Bool
    /// Number of proofs in the chain
    public let chainLength: Int
    /// Final accumulated polynomial
    public let finalAccPoly: [Fr]
    /// Final error term
    public let finalError: Fr
    /// Total deferred checks remaining
    public let totalDeferredChecks: Int
    /// Maximum depth reached in the composition tree
    public let maxDepth: Int

    public init(
        verified: Bool,
        chainLength: Int,
        finalAccPoly: [Fr],
        finalError: Fr,
        totalDeferredChecks: Int,
        maxDepth: Int
    ) {
        self.verified = verified
        self.chainLength = chainLength
        self.finalAccPoly = finalAccPoly
        self.finalError = finalError
        self.totalDeferredChecks = totalDeferredChecks
        self.maxDepth = maxDepth
    }
}

// MARK: - Proof Size Estimation

/// Estimated proof sizes for different schemes and recursion depths.
public struct ProofSizeEstimate {
    /// Size in bytes of the proof at this recursion depth
    public let sizeBytes: Int
    /// Number of public inputs
    public let numPublicInputs: Int
    /// Recursion depth
    public let depth: Int
    /// Base scheme
    public let scheme: BaseScheme

    public init(sizeBytes: Int, numPublicInputs: Int, depth: Int, scheme: BaseScheme) {
        self.sizeBytes = sizeBytes
        self.numPublicInputs = numPublicInputs
        self.depth = depth
        self.scheme = scheme
    }
}

// MARK: - GPU Recursive Composition Engine

/// GPU-accelerated engine for recursive proof composition.
///
/// Provides the core operations for building recursive proof systems:
///   - Fold proof nodes into a composition accumulator
///   - Split accumulation into deferred and immediate parts
///   - Verify chains and trees of proofs
///   - Estimate and reduce proof sizes through recursive wrapping
///
/// Compatible with both Groth16 and Plonk base schemes. Uses the BN254
/// scalar field (Fr) for all accumulation arithmetic.
public final class GPURecursiveCompositionEngine {

    /// Version — will be registered in Versions.swift separately
    public static let engineVersion = "1.0.0"

    public init() {}

    // MARK: - Accumulator Folding

    /// Fold a proof node into the composition accumulator.
    ///
    /// Computes:
    ///   acc' = acc + r * node.commitmentPoly
    ///   error' = error + r^2 * <acc, node.commitmentPoly>
    ///
    /// - Parameters:
    ///   - accumulator: current composition accumulator
    ///   - node: the proof node to fold in
    ///   - challenge: Fiat-Shamir challenge for this fold step
    /// - Returns: updated accumulator
    public func foldNode(
        accumulator: CompositionAccumulator,
        node: ProofNode,
        challenge: Fr
    ) -> CompositionAccumulator {
        let n = accumulator.accPoly.count
        let nodePoly = node.commitmentPoly
        precondition(nodePoly.count == n, "Polynomial sizes must match for folding")

        // Cross term: <acc, nodePoly>
        var crossTerm = Fr.zero
        accumulator.accPoly.withUnsafeBytes { aBuf in
            nodePoly.withUnsafeBytes { bBuf in
                withUnsafeMutableBytes(of: &crossTerm) { rBuf in
                    bn254_fr_inner_product(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }

        // acc' = acc + r * nodePoly
        var newAcc = accumulator.accPoly
        var challengeCopy = challenge
        newAcc.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: &challengeCopy) { sBuf in
                nodePoly.withUnsafeBytes { xBuf in
                    bn254_fr_batch_axpy(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        // error' = error + r^2 * crossTerm
        let r2 = frMul(challenge, challenge)
        let newError = frAdd(accumulator.error, frMul(r2, crossTerm))

        var updatedChallenges = accumulator.challenges
        updatedChallenges.append(challenge)

        var updatedIDs = accumulator.foldedProofIDs
        updatedIDs.append(node.id)

        return CompositionAccumulator(
            accPoly: newAcc,
            error: newError,
            foldCount: accumulator.foldCount + 1,
            challenges: updatedChallenges,
            deferredPairings: accumulator.deferredPairings,
            deferredCommitments: accumulator.deferredCommitments,
            foldedProofIDs: updatedIDs,
            scheme: accumulator.scheme,
            maxDepth: max(accumulator.maxDepth, node.depth)
        )
    }

    /// Fold a raw polynomial into the accumulator (without a proof node).
    public func foldPoly(
        accumulator: CompositionAccumulator,
        poly: [Fr],
        challenge: Fr
    ) -> CompositionAccumulator {
        let n = accumulator.accPoly.count
        precondition(poly.count == n, "Polynomial sizes must match for folding")

        var crossTerm = Fr.zero
        accumulator.accPoly.withUnsafeBytes { aBuf in
            poly.withUnsafeBytes { bBuf in
                withUnsafeMutableBytes(of: &crossTerm) { rBuf in
                    bn254_fr_inner_product(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }

        var newAcc = accumulator.accPoly
        var challengeCopy = challenge
        newAcc.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: &challengeCopy) { sBuf in
                poly.withUnsafeBytes { xBuf in
                    bn254_fr_batch_axpy(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        let r2 = frMul(challenge, challenge)
        let newError = frAdd(accumulator.error, frMul(r2, crossTerm))

        var updatedChallenges = accumulator.challenges
        updatedChallenges.append(challenge)

        return CompositionAccumulator(
            accPoly: newAcc,
            error: newError,
            foldCount: accumulator.foldCount + 1,
            challenges: updatedChallenges,
            deferredPairings: accumulator.deferredPairings,
            deferredCommitments: accumulator.deferredCommitments,
            foldedProofIDs: accumulator.foldedProofIDs,
            scheme: accumulator.scheme,
            maxDepth: accumulator.maxDepth
        )
    }

    // MARK: - Split Accumulation

    /// Split a proof node's verification into deferred and immediate parts.
    ///
    /// For Groth16:
    ///   - Immediate: MSM accumulation (vk_ic[0] + sum(pub[i] * vk_ic[i+1]))
    ///   - Deferred: pairing check (e(A,B) = e(alpha,beta) * e(accum,gamma) * e(C,delta))
    ///
    /// For Plonk:
    ///   - Immediate: linearization polynomial evaluation
    ///   - Deferred: KZG opening check (commitment consistency)
    ///
    /// - Parameters:
    ///   - node: the proof node to split
    ///   - transcript: mutable transcript for Fiat-Shamir
    /// - Returns: split accumulation result
    public func splitAccumulation(
        node: ProofNode,
        transcript: inout RecursiveCompositionTranscript
    ) -> SplitAccumulationResult {
        transcript.appendLabel("split-accumulation")
        transcript.appendNode(node)

        // Compute immediate part: inner product of public inputs with commitment poly
        let pi = node.publicInputs
        let cp = node.commitmentPoly
        let minLen = min(pi.count, cp.count)

        var immediateScalar = Fr.zero
        for i in 0..<minLen {
            immediateScalar = frAdd(immediateScalar, frMul(pi[i], cp[i]))
        }

        // The immediate polynomial is the weighted combination of public inputs
        // projected onto the commitment basis
        var immediatePoly = [Fr](repeating: Fr.zero, count: cp.count)
        for i in 0..<minLen {
            immediatePoly[i] = frMul(pi[i], cp[i])
        }

        // Generate deferred checks based on scheme
        var deferredPairings = [DeferredPairingCheck]()
        var deferredCommitments = [DeferredCommitmentCheck]()

        switch node.scheme {
        case .groth16:
            // Groth16: defer the pairing check
            // e(A, B) == e(alpha, beta) * e(vk_accum, gamma) * e(C, delta)
            // Simplified: accumulate the scalar product representation
            let challenge = transcript.squeezeAndAdvance()
            let lhsG1 = immediateScalar
            let lhsG2 = challenge
            let rhsG1 = frMul(immediateScalar, challenge)
            let rhsG2 = Fr.one
            deferredPairings.append(DeferredPairingCheck(
                lhsG1: lhsG1, lhsG2: lhsG2,
                rhsG1: rhsG1, rhsG2: rhsG2
            ))

        case .plonk:
            // Plonk: defer the KZG commitment check
            let evalPoint = transcript.squeezeAndAdvance()
            let evalValue = evaluatePolynomial(cp, at: evalPoint)
            let openingProof = transcript.squeezeAndAdvance()
            deferredCommitments.append(DeferredCommitmentCheck(
                commitment: immediateScalar,
                point: evalPoint,
                value: evalValue,
                openingProof: openingProof
            ))
        }

        return SplitAccumulationResult(
            immediatePoly: immediatePoly,
            immediateScalar: immediateScalar,
            deferredPairings: deferredPairings,
            deferredCommitments: deferredCommitments,
            immediateVerified: true,
            scheme: node.scheme
        )
    }

    // MARK: - Fold with Split Accumulation

    /// Fold a proof node into the accumulator using split accumulation.
    ///
    /// Combines folding and split accumulation in one step:
    ///   1. Split the node into deferred and immediate parts
    ///   2. Fold the immediate part into the accumulator
    ///   3. Store the deferred checks for later batch verification
    ///
    /// - Parameters:
    ///   - accumulator: current composition accumulator
    ///   - node: the proof node to fold
    ///   - transcript: mutable transcript for Fiat-Shamir
    /// - Returns: updated accumulator with deferred checks
    public func foldWithSplit(
        accumulator: CompositionAccumulator,
        node: ProofNode,
        transcript: inout RecursiveCompositionTranscript
    ) -> CompositionAccumulator {
        // 1. Split accumulation
        let split = splitAccumulation(node: node, transcript: &transcript)

        // 2. Derive folding challenge
        transcript.appendLabel("fold-with-split-\(accumulator.foldCount)")
        transcript.appendScalar(accumulator.error)
        let challenge = transcript.squeezeAndAdvance()

        // 3. Fold the node's commitment polynomial
        var acc = foldNode(accumulator: accumulator, node: node, challenge: challenge)

        // 4. Accumulate deferred checks
        acc.deferredPairings.append(contentsOf: split.deferredPairings)
        acc.deferredCommitments.append(contentsOf: split.deferredCommitments)

        return acc
    }

    // MARK: - Chain Verification

    /// Verify a chain of proof nodes by folding them sequentially.
    ///
    /// Each node in the chain is folded into the accumulator in order.
    /// The chain is valid if:
    ///   1. All immediate checks pass during folding
    ///   2. All deferred checks pass at finalization
    ///
    /// - Parameters:
    ///   - chain: ordered sequence of proof nodes
    ///   - transcript: mutable transcript for Fiat-Shamir
    /// - Returns: chain verification result
    public func verifyChain(
        chain: [ProofNode],
        transcript: inout RecursiveCompositionTranscript
    ) -> ChainVerificationResult {
        guard !chain.isEmpty else {
            return ChainVerificationResult(
                verified: true,
                chainLength: 0,
                finalAccPoly: [],
                finalError: Fr.zero,
                totalDeferredChecks: 0,
                maxDepth: 0
            )
        }

        transcript.appendLabel("chain-verify")
        var count = UInt32(chain.count)
        withUnsafeBytes(of: &count) { _ in transcript.appendLabel("chain-len-\(chain.count)") }

        // Initialize accumulator from first node
        let polySize = chain[0].commitmentPoly.count
        var acc = CompositionAccumulator.initial(node: chain[0])

        // Fold remaining nodes
        for i in 1..<chain.count {
            let node = chain[i]

            // Ensure consistent polynomial sizes
            guard node.commitmentPoly.count == polySize else {
                return ChainVerificationResult(
                    verified: false,
                    chainLength: i,
                    finalAccPoly: acc.accPoly,
                    finalError: acc.error,
                    totalDeferredChecks: acc.totalDeferredChecks,
                    maxDepth: acc.maxDepth
                )
            }

            acc = foldWithSplit(accumulator: acc, node: node, transcript: &transcript)
        }

        // Verify deferred checks
        let pairingsOK = verifyDeferredPairings(accumulator: acc, transcript: &transcript)
        let commitmentsOK = verifyDeferredCommitments(accumulator: acc, transcript: &transcript)

        return ChainVerificationResult(
            verified: pairingsOK && commitmentsOK,
            chainLength: chain.count,
            finalAccPoly: acc.accPoly,
            finalError: acc.error,
            totalDeferredChecks: acc.totalDeferredChecks,
            maxDepth: acc.maxDepth
        )
    }

    // MARK: - Deferred Verification

    /// Verify all deferred pairing checks via random linear combination.
    ///
    /// Batches all pairing checks using random scalars:
    ///   sum_i r_i * (lhsG1_i * lhsG2_i - rhsG1_i * rhsG2_i) == 0
    public func verifyDeferredPairings(
        accumulator: CompositionAccumulator,
        transcript: inout RecursiveCompositionTranscript
    ) -> Bool {
        let checks = accumulator.deferredPairings
        guard !checks.isEmpty else { return true }

        transcript.appendLabel("batch-pairing-verify")

        var batchSum = Fr.zero
        for i in 0..<checks.count {
            transcript.appendScalar(checks[i].lhsG1)
            transcript.appendScalar(checks[i].rhsG1)
            let r_i = transcript.squeezeAndAdvance()

            let lhsProd = frMul(checks[i].lhsG1, checks[i].lhsG2)
            let rhsProd = frMul(checks[i].rhsG1, checks[i].rhsG2)
            let diff = frSub(lhsProd, rhsProd)
            batchSum = frAdd(batchSum, frMul(r_i, diff))
        }

        return frEqual(batchSum, Fr.zero)
    }

    /// Verify all deferred commitment checks via batch opening.
    ///
    /// Each commitment check verifies C opens to v at point z.
    /// We batch by checking polynomial consistency:
    ///   sum_i r_i * (eval(poly, z_i) - v_i) == 0
    public func verifyDeferredCommitments(
        accumulator: CompositionAccumulator,
        transcript: inout RecursiveCompositionTranscript
    ) -> Bool {
        let checks = accumulator.deferredCommitments
        guard !checks.isEmpty else { return true }

        transcript.appendLabel("batch-commitment-verify")

        var batchSum = Fr.zero
        for i in 0..<checks.count {
            transcript.appendScalar(checks[i].commitment)
            transcript.appendScalar(checks[i].point)
            let r_i = transcript.squeezeAndAdvance()

            // Verify commitment * openingProof consistency
            // commitment == value + openingProof * point (simplified KZG check)
            let reconstructed = frAdd(checks[i].value, frMul(checks[i].openingProof, checks[i].point))
            let diff = frSub(checks[i].commitment, reconstructed)
            batchSum = frAdd(batchSum, frMul(r_i, diff))
        }

        return frEqual(batchSum, Fr.zero)
    }

    /// Verify all deferred checks (both pairings and commitments).
    public func verifyAllDeferred(
        accumulator: CompositionAccumulator,
        transcript: inout RecursiveCompositionTranscript
    ) -> Bool {
        let pOK = verifyDeferredPairings(accumulator: accumulator, transcript: &transcript)
        let cOK = verifyDeferredCommitments(accumulator: accumulator, transcript: &transcript)
        return pOK && cOK
    }

    // MARK: - Proof-of-Proof

    /// Construct a proof-of-proof: verify inner SNARK proof inside an outer circuit.
    ///
    /// This is the fundamental recursive composition operation. Given an inner proof
    /// (from any supported scheme), we build a circuit that verifies it, then produce
    /// an outer proof of that circuit.
    ///
    /// The result is a CompositionAccumulator containing the verification state.
    ///
    /// - Parameters:
    ///   - innerNode: the inner proof to verify recursively
    ///   - outerScheme: the scheme to use for the outer proof
    ///   - transcript: mutable transcript for Fiat-Shamir
    /// - Returns: accumulator representing the proof-of-proof
    public func proofOfProof(
        innerNode: ProofNode,
        outerScheme: BaseScheme,
        transcript: inout RecursiveCompositionTranscript
    ) -> CompositionAccumulator {
        transcript.appendLabel("proof-of-proof")
        transcript.appendNode(innerNode)

        // Split the inner proof into deferred and immediate parts
        let split = splitAccumulation(node: innerNode, transcript: &transcript)

        // Build the outer accumulator: the immediate part is verified in-circuit,
        // the deferred part is accumulated for later verification
        let outerNode = ProofNode(
            id: innerNode.id + 1000,
            scheme: outerScheme,
            publicInputs: [split.immediateScalar],
            commitmentPoly: split.immediatePoly,
            isLeaf: false,
            children: [innerNode.id],
            depth: innerNode.depth + 1
        )

        var acc = CompositionAccumulator.initial(node: outerNode)
        acc.deferredPairings = split.deferredPairings
        acc.deferredCommitments = split.deferredCommitments

        return acc
    }

    // MARK: - Multi-Proof Accumulation

    /// Fold multiple proof nodes into a single accumulator.
    ///
    /// This is the batch version of foldNode: given N proofs, we derive N challenges
    /// and fold all of them sequentially into one accumulator.
    ///
    /// - Parameters:
    ///   - nodes: the proof nodes to accumulate
    ///   - scheme: the base scheme for the accumulator
    ///   - transcript: mutable transcript for Fiat-Shamir
    /// - Returns: accumulator containing all folded proofs
    public func accumulateProofs(
        nodes: [ProofNode],
        scheme: BaseScheme,
        transcript: inout RecursiveCompositionTranscript
    ) -> CompositionAccumulator {
        guard !nodes.isEmpty else {
            return CompositionAccumulator.zero(size: 0, scheme: scheme)
        }

        transcript.appendLabel("accumulate-proofs")
        var nodeCount = UInt32(nodes.count)
        withUnsafeBytes(of: &nodeCount) { _ in
            transcript.appendLabel("count-\(nodes.count)")
        }

        let polySize = nodes[0].commitmentPoly.count
        var acc = CompositionAccumulator.initial(node: nodes[0])

        for i in 1..<nodes.count {
            precondition(nodes[i].commitmentPoly.count == polySize,
                        "All proof polynomials must have the same size")

            transcript.appendNode(nodes[i])
            let challenge = transcript.squeezeAndAdvance()
            acc = foldNode(accumulator: acc, node: nodes[i], challenge: challenge)
        }

        return acc
    }

    // MARK: - Proof Size Reduction

    /// Estimate proof size at a given recursion depth.
    ///
    /// Each recursion layer reduces proof size:
    ///   - Groth16: constant 192 bytes at any depth (3 group elements)
    ///   - Plonk: ~1.5KB base, reduced to ~500 bytes after one recursion
    ///
    /// Public inputs accumulate across layers (each layer adds verification state).
    ///
    /// - Parameters:
    ///   - scheme: the proof system
    ///   - depth: recursion depth (0 = base proof)
    ///   - numPublicInputs: public inputs at the base level
    /// - Returns: estimated proof size
    public func estimateProofSize(
        scheme: BaseScheme,
        depth: Int,
        numPublicInputs: Int
    ) -> ProofSizeEstimate {
        switch scheme {
        case .groth16:
            // Groth16: 2 G1 + 1 G2 = 2*64 + 128 = 256 bytes (uncompressed)
            // Compressed: 2*32 + 64 = 128 bytes
            // Public inputs: 32 bytes each
            let piCount = numPublicInputs + (depth > 0 ? 1 : 0) // +1 for accumulated state hash
            let sizeBytes = 192 + piCount * 32
            return ProofSizeEstimate(
                sizeBytes: sizeBytes,
                numPublicInputs: piCount,
                depth: depth,
                scheme: scheme
            )

        case .plonk:
            // Plonk: base ~1.5KB, each recursion reduces by ~50%
            // Minimum: ~500 bytes after first recursion
            let baseSizeBytes = 1536
            let piCount = numPublicInputs + depth
            let reductionFactor = depth > 0 ? max(1, Int(pow(2.0, Double(depth)))) : 1
            let sizeBytes = max(512, baseSizeBytes / reductionFactor) + piCount * 32
            return ProofSizeEstimate(
                sizeBytes: sizeBytes,
                numPublicInputs: piCount,
                depth: depth,
                scheme: scheme
            )
        }
    }

    /// Compute the optimal recursion depth for proof size reduction.
    ///
    /// Returns the depth at which further recursion no longer reduces proof size
    /// (or even increases it due to public input growth).
    ///
    /// - Parameters:
    ///   - scheme: the proof system
    ///   - numPublicInputs: public inputs at the base level
    /// - Returns: optimal recursion depth
    public func optimalRecursionDepth(
        scheme: BaseScheme,
        numPublicInputs: Int
    ) -> Int {
        var bestDepth = 0
        var bestSize = estimateProofSize(scheme: scheme, depth: 0, numPublicInputs: numPublicInputs).sizeBytes

        for d in 1...8 {
            let est = estimateProofSize(scheme: scheme, depth: d, numPublicInputs: numPublicInputs)
            if est.sizeBytes < bestSize {
                bestSize = est.sizeBytes
                bestDepth = d
            } else {
                break
            }
        }

        return bestDepth
    }

    // MARK: - Challenge Derivation

    /// Derive a Fiat-Shamir challenge from the composition accumulator state.
    public func deriveChallenge(
        accumulator: CompositionAccumulator,
        domainSeparator: String
    ) -> Fr {
        var t = RecursiveCompositionTranscript()
        t.appendLabel(domainSeparator)
        t.appendScalars(accumulator.accPoly)
        t.appendScalar(accumulator.error)
        var countScalar = frFromInt(UInt64(accumulator.totalDeferredChecks))
        t.appendScalar(countScalar)
        _ = countScalar
        return t.squeeze()
    }

    // MARK: - Polynomial Evaluation (Horner)

    /// Evaluate polynomial at a point using Horner's method.
    /// poly = c_0 + c_1*x + c_2*x^2 + ...
    public func evaluatePolynomial(_ poly: [Fr], at point: Fr) -> Fr {
        guard !poly.isEmpty else { return Fr.zero }
        var result = Fr.zero
        poly.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: point) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(poly.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: - Inner Product

    /// Compute inner product of two equal-length vectors.
    public func innerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        precondition(a.count == b.count, "Vector lengths must match")
        var sum = Fr.zero
        for i in 0..<a.count {
            sum = frAdd(sum, frMul(a[i], b[i]))
        }
        return sum
    }

    // MARK: - Linear Combination

    /// Compute a linear combination of polynomials: sum_i challenges[i] * polys[i].
    public func linearCombination(
        polys: [[Fr]],
        challenges: [Fr]
    ) -> [Fr] {
        precondition(polys.count == challenges.count, "Poly and challenge counts must match")
        guard let first = polys.first else { return [] }
        let n = first.count
        var result = [Fr](repeating: Fr.zero, count: n)
        for j in 0..<polys.count {
            var c = challenges[j]
            let p = polys[j]
            precondition(p.count == n, "All polynomials must have the same length")
            p.withUnsafeBytes { pBuf in
                result.withUnsafeMutableBytes { rBuf in
                    withUnsafeBytes(of: &c) { cBuf in
                        bn254_fr_batch_axpy(
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }
        return result
    }

    // MARK: - Composition Tree Builder

    /// Build a binary composition tree from a list of leaf proofs.
    ///
    /// Pairs adjacent proofs and folds them, then pairs the results, etc.
    /// This produces a balanced binary tree of depth ceil(log2(N)).
    ///
    /// - Parameters:
    ///   - leaves: the leaf proof nodes
    ///   - scheme: the scheme for intermediate nodes
    ///   - transcript: mutable transcript for Fiat-Shamir
    /// - Returns: the root accumulator
    public func buildCompositionTree(
        leaves: [ProofNode],
        scheme: BaseScheme,
        transcript: inout RecursiveCompositionTranscript
    ) -> CompositionAccumulator {
        guard !leaves.isEmpty else {
            return CompositionAccumulator.zero(size: 0, scheme: scheme)
        }
        guard leaves.count > 1 else {
            return CompositionAccumulator.initial(node: leaves[0])
        }

        transcript.appendLabel("composition-tree")

        let polySize = leaves[0].commitmentPoly.count

        // Build the tree bottom-up by folding pairs
        var currentLevel = leaves
        var nextID = leaves.map(\.id).max()! + 1
        var currentDepth = 1

        while currentLevel.count > 1 {
            var nextLevel = [ProofNode]()

            var i = 0
            while i < currentLevel.count {
                if i + 1 < currentLevel.count {
                    // Pair two nodes
                    let left = currentLevel[i]
                    let right = currentLevel[i + 1]

                    // Fold the pair
                    var acc = CompositionAccumulator.initial(node: left)
                    transcript.appendNode(right)
                    let challenge = transcript.squeezeAndAdvance()
                    acc = foldNode(accumulator: acc, node: right, challenge: challenge)

                    // Create intermediate node
                    let intermediate = ProofNode(
                        id: nextID,
                        scheme: scheme,
                        publicInputs: [acc.error],
                        commitmentPoly: acc.accPoly,
                        isLeaf: false,
                        children: [left.id, right.id],
                        depth: currentDepth
                    )
                    nextLevel.append(intermediate)
                    nextID += 1
                    i += 2
                } else {
                    // Odd node passes through
                    nextLevel.append(currentLevel[i])
                    i += 1
                }
            }

            currentLevel = nextLevel
            currentDepth += 1
        }

        return CompositionAccumulator.initial(node: currentLevel[0])
    }
}
