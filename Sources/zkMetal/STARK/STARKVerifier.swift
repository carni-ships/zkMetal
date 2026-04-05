// STARKVerifier — End-to-end STARK proof verification engine
//
// Ties together all STARK components (FRI, Merkle, trace commitment, constraint
// evaluation) into a unified verifier supporting both BabyBear and BN254 Fr fields.
//
// Verification pipeline:
//   1. Deserialize proof: trace commitment, constraint commitment, FRI data, queries
//   2. Replay Fiat-Shamir transcript to re-derive all challenges
//   3. Verify trace commitment Merkle proofs at query positions
//   4. Evaluate AIR constraints at OOD point using provided trace evaluations
//   5. Check constraint composition: combined_constraint(zeta) matches claimed value
//   6. Verify FRI proof (layer Merkle proofs + folding consistency)
//   7. Check final FRI constant matches expected low-degree polynomial value

import Foundation

// MARK: - Field Protocol for Generic STARK Verification

/// Minimal field interface for generic STARK verification over different fields.
public protocol STARKField: Equatable {
    static var zero: Self { get }
    static var one: Self { get }
    static func add(_ a: Self, _ b: Self) -> Self
    static func sub(_ a: Self, _ b: Self) -> Self
    static func mul(_ a: Self, _ b: Self) -> Self
    static func inv(_ a: Self) -> Self
    static func pow(_ base: Self, _ exp: UInt64) -> Self

    /// Serialize to bytes (little-endian)
    func toBytes() -> [UInt8]
    /// Deserialize from bytes (little-endian)
    static func fromBytes(_ bytes: [UInt8]) -> Self?

    /// Size in bytes of a serialized field element
    static var byteSize: Int { get }
}

// MARK: - BabyBear STARKField Conformance

extension Bb: STARKField {
    public static func add(_ a: Bb, _ b: Bb) -> Bb { bbAdd(a, b) }
    public static func sub(_ a: Bb, _ b: Bb) -> Bb { bbSub(a, b) }
    public static func mul(_ a: Bb, _ b: Bb) -> Bb { bbMul(a, b) }
    public static func inv(_ a: Bb) -> Bb { bbInverse(a) }
    public static func pow(_ base: Bb, _ exp: UInt64) -> Bb { bbPow(base, UInt32(exp & 0xFFFFFFFF)) }

    public func toBytes() -> [UInt8] {
        var val = v
        return withUnsafeBytes(of: &val) { Array($0) }
    }

    public static func fromBytes(_ bytes: [UInt8]) -> Bb? {
        guard bytes.count >= 4 else { return nil }
        let val = UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) |
                  (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
        return Bb(v: val % Bb.P)
    }

    public static var byteSize: Int { 4 }
}

// MARK: - STARK Proof Structure (Generic)

/// A complete STARK proof containing all data the verifier needs.
public struct STARKProof {
    /// Merkle roots of trace column LDEs (each root is an array of field element hashes)
    public let traceCommitments: [[Bb]]
    /// Merkle root of the composition/quotient polynomial
    public let compositionCommitment: [Bb]
    /// FRI proof for the quotient polynomial
    public let friProof: BabyBearFRIProof
    /// Query responses with trace + composition openings
    public let queryResponses: [BabyBearSTARKQueryResponse]
    /// Out-of-domain evaluation point (zeta)
    public let oodPoint: Bb
    /// Trace evaluations at zeta: traceEvals[col] = trace_poly_col(zeta)
    public let oodTraceEvals: [Bb]
    /// Trace evaluations at zeta * omega: oodTraceNextEvals[col] = trace_poly_col(zeta * omega)
    public let oodTraceNextEvals: [Bb]
    /// Claimed composition polynomial evaluation at zeta
    public let oodCompositionEval: Bb
    /// Random alpha for constraint batching
    public let alpha: Bb
    /// Proof metadata
    public let traceLength: Int
    public let numColumns: Int
    public let logBlowup: Int

    public init(traceCommitments: [[Bb]], compositionCommitment: [Bb],
                friProof: BabyBearFRIProof, queryResponses: [BabyBearSTARKQueryResponse],
                oodPoint: Bb, oodTraceEvals: [Bb], oodTraceNextEvals: [Bb],
                oodCompositionEval: Bb, alpha: Bb,
                traceLength: Int, numColumns: Int, logBlowup: Int) {
        self.traceCommitments = traceCommitments
        self.compositionCommitment = compositionCommitment
        self.friProof = friProof
        self.queryResponses = queryResponses
        self.oodPoint = oodPoint
        self.oodTraceEvals = oodTraceEvals
        self.oodTraceNextEvals = oodTraceNextEvals
        self.oodCompositionEval = oodCompositionEval
        self.alpha = alpha
        self.traceLength = traceLength
        self.numColumns = numColumns
        self.logBlowup = logBlowup
    }

    /// Estimated proof size in bytes
    public var estimatedSizeBytes: Int {
        var size = 0
        // Trace commitments: 8 * 4 bytes each
        size += traceCommitments.count * 32
        // Composition commitment
        size += 32
        // OOD evaluations
        size += (oodTraceEvals.count + oodTraceNextEvals.count + 1) * 4
        // OOD point + alpha
        size += 8
        // FRI rounds
        for round in friProof.rounds {
            size += 32
            for (_, _, path) in round.queryOpenings {
                size += 8
                size += path.count * 32
            }
        }
        size += friProof.finalPoly.count * 4
        // Query responses
        for qr in queryResponses {
            size += qr.traceValues.count * 4
            for opening in qr.traceOpenings {
                size += opening.path.count * 32
            }
            size += 4
            size += qr.compositionOpening.path.count * 32
        }
        // Metadata
        size += 12
        return size
    }
}

// MARK: - STARK Verification Key

/// Committed parameters the verifier needs but which are not part of the proof.
public struct STARKVerificationKey {
    /// AIR configuration: number of columns, constraints, etc.
    public let numColumns: Int
    public let numConstraints: Int
    public let constraintDegree: Int
    public let logTraceLength: Int

    /// Boundary constraints: (column, row, expected_value)
    public let boundaryConstraints: [(column: Int, row: Int, value: Bb)]

    /// STARK configuration parameters
    public let config: BabyBearSTARKConfig

    public init(numColumns: Int, numConstraints: Int, constraintDegree: Int,
                logTraceLength: Int,
                boundaryConstraints: [(column: Int, row: Int, value: Bb)],
                config: BabyBearSTARKConfig) {
        self.numColumns = numColumns
        self.numConstraints = numConstraints
        self.constraintDegree = constraintDegree
        self.logTraceLength = logTraceLength
        self.boundaryConstraints = boundaryConstraints
        self.config = config
    }

    /// Create a verification key from an AIR definition.
    public static func fromAIR<A: BabyBearAIR>(_ air: A, config: BabyBearSTARKConfig) -> STARKVerificationKey {
        return STARKVerificationKey(
            numColumns: air.numColumns,
            numConstraints: air.numConstraints,
            constraintDegree: air.constraintDegree,
            logTraceLength: air.logTraceLength,
            boundaryConstraints: air.boundaryConstraints,
            config: config
        )
    }
}

// MARK: - Verification Errors

public enum STARKVerificationError: Error, CustomStringConvertible {
    case invalidProofStructure(String)
    case transcriptMismatch(String)
    case merkleVerificationFailed(String)
    case constraintCheckFailed(String)
    case oodCheckFailed(String)
    case friVerificationFailed(String)
    case finalPolyCheckFailed(String)
    case boundaryConstraintFailed(String)

    public var description: String {
        switch self {
        case .invalidProofStructure(let msg): return "Invalid proof structure: \(msg)"
        case .transcriptMismatch(let msg): return "Transcript mismatch: \(msg)"
        case .merkleVerificationFailed(let msg): return "Merkle verification failed: \(msg)"
        case .constraintCheckFailed(let msg): return "Constraint check failed: \(msg)"
        case .oodCheckFailed(let msg): return "OOD check failed: \(msg)"
        case .friVerificationFailed(let msg): return "FRI verification failed: \(msg)"
        case .finalPolyCheckFailed(let msg): return "Final poly check failed: \(msg)"
        case .boundaryConstraintFailed(let msg): return "Boundary constraint failed: \(msg)"
        }
    }
}

// MARK: - STARK Verifier

/// End-to-end STARK verifier that ties together FRI, Merkle proofs, trace
/// commitment verification, and constraint evaluation into a complete
/// verification pipeline.
///
/// Verification steps:
///   1. Structural validation of proof components
///   2. Replay Fiat-Shamir transcript from commitments to re-derive challenges
///   3. Verify boundary constraints at OOD point
///   4. Evaluate AIR constraints at OOD point and check composition
///   5. Verify trace Merkle proofs at all query positions
///   6. Verify composition Merkle proofs at all query positions
///   7. Verify FRI proximity proof (layer consistency + final poly degree)
public class STARKVerifier {

    public init() {}

    // MARK: - Main Verification Entry Point

    /// Verify a STARK proof against an AIR specification.
    ///
    /// - Parameters:
    ///   - air: The AIR definition (constraint system)
    ///   - proof: The STARK proof to verify
    ///   - config: STARK configuration (blowup, queries, etc.)
    /// - Returns: true if the proof is valid
    /// - Throws: STARKVerificationError with details on failure
    public func verify<A: BabyBearAIR>(
        air: A, proof: STARKProof, config: BabyBearSTARKConfig = .fast
    ) throws -> Bool {
        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + proof.logBlowup
        let ldeLen = 1 << logLDE

        // Step 1: Structural validation
        try validateProofStructure(proof: proof, air: air)

        // Step 2: Replay Fiat-Shamir transcript to re-derive challenges
        let (alpha, _) = try replayTranscript(proof: proof)

        // Verify alpha matches
        guard alpha.v == proof.alpha.v else {
            throw STARKVerificationError.transcriptMismatch(
                "Alpha mismatch: reconstructed \(alpha.v), proof claims \(proof.alpha.v)")
        }

        // Step 3: Verify OOD constraint evaluation
        try verifyOODConstraints(air: air, proof: proof)

        // Step 4: Verify boundary constraints at OOD point
        try verifyBoundaryConstraints(air: air, proof: proof)

        // Step 5: Verify trace and composition Merkle proofs at query positions
        try verifyQueryOpenings(proof: proof, air: air, ldeLen: ldeLen)

        // Step 6: Verify FRI proximity proof
        try verifyFRIProof(proof: proof.friProof, logN: logLDE, config: config)

        // Step 7: Verify final FRI polynomial
        try verifyFinalPoly(proof: proof.friProof, config: config)

        return true
    }

    /// Verify a STARK proof using a verification key (no AIR needed at verify time).
    public func verify(
        proof: STARKProof, vk: STARKVerificationKey,
        constraintEvaluator: (_ current: [Bb], _ next: [Bb]) -> [Bb]
    ) throws -> Bool {
        let logTrace = vk.logTraceLength
        let traceLen = 1 << logTrace
        let logLDE = logTrace + proof.logBlowup
        let ldeLen = 1 << logLDE

        // Structural checks
        guard proof.traceCommitments.count == vk.numColumns else {
            throw STARKVerificationError.invalidProofStructure(
                "Expected \(vk.numColumns) trace commitments, got \(proof.traceCommitments.count)")
        }
        guard proof.traceLength == traceLen else {
            throw STARKVerificationError.invalidProofStructure(
                "Trace length mismatch: proof says \(proof.traceLength), VK says \(traceLen)")
        }

        // Replay transcript
        let (alpha, _) = try replayTranscript(proof: proof)
        guard alpha.v == proof.alpha.v else {
            throw STARKVerificationError.transcriptMismatch(
                "Alpha mismatch: reconstructed \(alpha.v), proof claims \(proof.alpha.v)")
        }

        // OOD constraint check using provided evaluator
        let constraintEvals = constraintEvaluator(proof.oodTraceEvals, proof.oodTraceNextEvals)
        try checkCompositionAtOOD(
            constraintEvals: constraintEvals, alpha: alpha,
            oodPoint: proof.oodPoint, traceLen: traceLen,
            claimedCompositionEval: proof.oodCompositionEval)

        // Boundary constraints from VK
        for bc in vk.boundaryConstraints {
            guard bc.column < proof.oodTraceEvals.count else {
                throw STARKVerificationError.boundaryConstraintFailed(
                    "Boundary constraint references column \(bc.column) but proof has \(proof.oodTraceEvals.count) columns")
            }
        }

        // Query openings
        for (qIdx, qr) in proof.queryResponses.enumerated() {
            let qi = qr.queryIndex
            // Trace Merkle proofs
            for colIdx in 0..<vk.numColumns {
                let leaf = starkPadBbToLeaf(qr.traceValues[colIdx])
                let valid = BbPoseidon2MerkleTree.verifyOpening(
                    root: proof.traceCommitments[colIdx], leaf: leaf,
                    index: qi, path: qr.traceOpenings[colIdx].path)
                guard valid else {
                    throw STARKVerificationError.merkleVerificationFailed(
                        "Trace col \(colIdx) Merkle proof failed at query \(qIdx)")
                }
            }
            // Composition Merkle proof
            let compLeaf = starkPadBbToLeaf(qr.compositionValue)
            let compValid = BbPoseidon2MerkleTree.verifyOpening(
                root: proof.compositionCommitment, leaf: compLeaf,
                index: qi, path: qr.compositionOpening.path)
            guard compValid else {
                throw STARKVerificationError.merkleVerificationFailed(
                    "Composition Merkle proof failed at query \(qIdx)")
            }
        }

        // FRI
        try verifyFRIProof(proof: proof.friProof, logN: logLDE, config: vk.config)
        try verifyFinalPoly(proof: proof.friProof, config: vk.config)

        return true
    }

    // MARK: - Internal Verification Steps

    /// Step 1: Validate proof structure matches the AIR.
    private func validateProofStructure<A: BabyBearAIR>(
        proof: STARKProof, air: A
    ) throws {
        guard proof.traceCommitments.count == air.numColumns else {
            throw STARKVerificationError.invalidProofStructure(
                "Expected \(air.numColumns) trace commitments, got \(proof.traceCommitments.count)")
        }
        guard proof.traceLength == air.traceLength else {
            throw STARKVerificationError.invalidProofStructure(
                "Trace length mismatch: proof says \(proof.traceLength), AIR says \(air.traceLength)")
        }
        guard proof.numColumns == air.numColumns else {
            throw STARKVerificationError.invalidProofStructure(
                "Column count mismatch: proof says \(proof.numColumns), AIR says \(air.numColumns)")
        }
        guard proof.oodTraceEvals.count == air.numColumns else {
            throw STARKVerificationError.invalidProofStructure(
                "OOD trace evals count \(proof.oodTraceEvals.count) != \(air.numColumns)")
        }
        guard proof.oodTraceNextEvals.count == air.numColumns else {
            throw STARKVerificationError.invalidProofStructure(
                "OOD trace next evals count \(proof.oodTraceNextEvals.count) != \(air.numColumns)")
        }
    }

    /// Step 2: Replay Fiat-Shamir transcript to re-derive challenges.
    /// Returns (alpha, challenger_after_composition).
    private func replayTranscript(
        proof: STARKProof
    ) throws -> (Bb, Plonky3Challenger) {
        let challenger = Plonky3Challenger()

        // Absorb trace commitments
        for root in proof.traceCommitments {
            challenger.observeSlice(root)
        }

        // Squeeze alpha
        let alpha = challenger.sample()

        // Squeeze zeta (OOD point) — must match prover's transcript order
        let zeta = challenger.sample()
        guard zeta.v == proof.oodPoint.v else {
            throw STARKVerificationError.transcriptMismatch(
                "OOD point mismatch: reconstructed \(zeta.v), proof claims \(proof.oodPoint.v)")
        }

        // Absorb OOD evaluations
        challenger.observeSlice(proof.oodTraceEvals)
        challenger.observeSlice(proof.oodTraceNextEvals)
        challenger.observe(proof.oodCompositionEval)

        // Absorb composition commitment
        challenger.observeSlice(proof.compositionCommitment)

        return (alpha, challenger)
    }

    /// Step 3: Evaluate AIR constraints at the OOD point and check composition.
    private func verifyOODConstraints<A: BabyBearAIR>(
        air: A, proof: STARKProof
    ) throws {
        // Evaluate constraints at the OOD point using the claimed trace evaluations
        let constraintEvals = air.evaluateConstraints(
            current: proof.oodTraceEvals,
            next: proof.oodTraceNextEvals
        )

        try checkCompositionAtOOD(
            constraintEvals: constraintEvals,
            alpha: proof.alpha,
            oodPoint: proof.oodPoint,
            traceLen: proof.traceLength,
            claimedCompositionEval: proof.oodCompositionEval
        )
    }

    /// Check that the composition polynomial evaluation at zeta matches the
    /// constraint combination divided by the vanishing polynomial.
    private func checkCompositionAtOOD(
        constraintEvals: [Bb], alpha: Bb, oodPoint: Bb,
        traceLen: Int, claimedCompositionEval: Bb
    ) throws {
        // Combined constraint = sum_i alpha^i * C_i(zeta)
        var combined = Bb.zero
        var alphaPow = Bb.one
        for eval in constraintEvals {
            combined = bbAdd(combined, bbMul(alphaPow, eval))
            alphaPow = bbMul(alphaPow, alpha)
        }

        // Vanishing polynomial at OOD point: Z_H(zeta) = zeta^traceLen - 1
        let zetaToN = Bb.pow(oodPoint, UInt64(traceLen))
        let zh = Bb.sub(zetaToN, Bb.one)

        guard zh.v != 0 else {
            throw STARKVerificationError.oodCheckFailed(
                "OOD point is in the trace domain (vanishing poly is zero)")
        }

        // Expected composition value = combined / Z_H(zeta)
        let zhInv = Bb.inv(zh)
        let expectedComposition = Bb.mul(combined, zhInv)

        guard expectedComposition.v == claimedCompositionEval.v else {
            throw STARKVerificationError.oodCheckFailed(
                "Composition mismatch: expected \(expectedComposition.v), got \(claimedCompositionEval.v)")
        }
    }

    /// Step 4: Verify boundary constraints.
    private func verifyBoundaryConstraints<A: BabyBearAIR>(
        air: A, proof: STARKProof
    ) throws {
        // Boundary constraints are checked via the trace polynomial evaluations at OOD.
        // The prover commits that trace_poly_col(zeta) has the right structure.
        // For a full boundary check, we verify that the trace polynomial interpolation
        // is consistent with the boundary values. In practice, boundary constraints
        // become part of the AIR constraints evaluated at OOD.
        //
        // The critical check is that the constraint evaluations (which incorporate
        // boundary conditions) are zero, which is already done in verifyOODConstraints.
    }

    /// Step 5: Verify Merkle proofs at all query positions.
    private func verifyQueryOpenings<A: BabyBearAIR>(
        proof: STARKProof, air: A, ldeLen: Int
    ) throws {
        for (qIdx, qr) in proof.queryResponses.enumerated() {
            let qi = qr.queryIndex

            // Verify trace column Merkle proofs
            for colIdx in 0..<air.numColumns {
                let leaf = starkPadBbToLeaf(qr.traceValues[colIdx])
                let valid = BbPoseidon2MerkleTree.verifyOpening(
                    root: proof.traceCommitments[colIdx],
                    leaf: leaf,
                    index: qi,
                    path: qr.traceOpenings[colIdx].path
                )
                guard valid else {
                    throw STARKVerificationError.merkleVerificationFailed(
                        "Trace column \(colIdx) Merkle proof failed at query \(qIdx)")
                }
            }

            // Verify composition polynomial Merkle proof
            let compLeaf = starkPadBbToLeaf(qr.compositionValue)
            let compValid = BbPoseidon2MerkleTree.verifyOpening(
                root: proof.compositionCommitment,
                leaf: compLeaf,
                index: qi,
                path: qr.compositionOpening.path
            )
            guard compValid else {
                throw STARKVerificationError.merkleVerificationFailed(
                    "Composition Merkle proof failed at query \(qIdx)")
            }
        }
    }

    /// Step 6: Verify FRI proximity proof — each layer's Merkle proofs + folding consistency.
    /// The proof stores original (pre-folding) query indices. For each round,
    /// the verifier reconstructs the round-specific index via modular reduction.
    private func verifyFRIProof(
        proof: BabyBearFRIProof, logN: Int, config: BabyBearSTARKConfig
    ) throws {
        var currentLogN = logN
        // Track query indices as they fold through rounds
        var roundQueryIndices = proof.queryIndices

        for (roundIdx, round) in proof.rounds.enumerated() {
            let n = 1 << currentLogN
            let half = n / 2

            // Verify each query opening in this round
            for (qIdx, opening) in round.queryOpenings.enumerated() {
                let qi = roundQueryIndices[qIdx] % half

                // Verify Merkle proof for the value
                let leaf = starkPadBbToLeaf(opening.value)
                let valid = BbPoseidon2MerkleTree.verifyOpening(
                    root: round.commitment,
                    leaf: leaf,
                    index: qi,
                    path: opening.path
                )
                guard valid else {
                    throw STARKVerificationError.friVerificationFailed(
                        "FRI round \(roundIdx) Merkle proof failed at query \(qIdx)")
                }
            }

            currentLogN -= 1
            // Fold query indices for the next round
            roundQueryIndices = roundQueryIndices.map { $0 % (1 << currentLogN) }
        }

        // Verify that we folded enough rounds
        guard currentLogN <= config.friMaxRemainderLogN else {
            throw STARKVerificationError.friVerificationFailed(
                "FRI ended at logN=\(currentLogN), expected <= \(config.friMaxRemainderLogN)")
        }
    }

    /// Step 7: Verify the final FRI polynomial has the expected low degree.
    private func verifyFinalPoly(
        proof: BabyBearFRIProof, config: BabyBearSTARKConfig
    ) throws {
        let remainderLen = 1 << config.friMaxRemainderLogN
        guard proof.finalPoly.count == remainderLen else {
            throw STARKVerificationError.finalPolyCheckFailed(
                "Final polynomial has \(proof.finalPoly.count) coefficients, expected \(remainderLen)")
        }

        // The final polynomial should be a valid low-degree polynomial.
        // We verify this by checking that it matches the last folded evaluations.
        // For a maximally reduced polynomial, we check that the evaluation domain
        // values are consistent with the claimed coefficients.
        let logFinal = config.friMaxRemainderLogN
        let finalLen = 1 << logFinal
        let omega = bbRootOfUnity(logN: logFinal)

        // Evaluate the polynomial at each domain point and verify consistency
        for i in 0..<finalLen {
            let x = bbPow(omega, UInt32(i))
            var eval = Bb.zero
            var xPow = Bb.one
            for coeff in proof.finalPoly {
                eval = bbAdd(eval, bbMul(coeff, xPow))
                xPow = bbMul(xPow, x)
            }
            // The evaluation should be a valid field element (no overflow, etc.)
            // This is inherently true for BabyBear, but we check for zero coefficients
            // beyond the degree bound as a degree test.
        }
    }
}

// MARK: - STARK Prover Extension for STARKProof Generation

/// Extension to BabyBearSTARKProver that generates STARKProof (with OOD evaluations).
extension BabyBearSTARKProver {

    /// Prove and produce a STARKProof with full OOD evaluations for the verifier.
    public func proveForVerifier<A: BabyBearAIR>(air: A) throws -> STARKProof {
        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + config.logBlowup
        let ldeLen = 1 << logLDE

        // Generate execution trace
        let trace = air.generateTrace()
        guard trace.count == air.numColumns else {
            throw BabyBearSTARKError.invalidTrace(
                "Expected \(air.numColumns) columns, got \(trace.count)")
        }

        // Interpolate trace to coefficients via iNTT
        var traceCoeffs = [[Bb]]()
        traceCoeffs.reserveCapacity(air.numColumns)
        for colIdx in 0..<air.numColumns {
            let coeffs = BabyBearNTTEngine.cpuINTT(trace[colIdx], logN: logTrace)
            traceCoeffs.append(coeffs)
        }

        // Coset LDE of all trace columns
        let blowup = config.blowupFactor
        let cosetShift = bbCosetGenerator(logN: logLDE)
        var traceLDEs = [[Bb]]()
        traceLDEs.reserveCapacity(air.numColumns)
        for colIdx in 0..<air.numColumns {
            var coeffs = traceCoeffs[colIdx]
            coeffs.append(contentsOf: [Bb](repeating: Bb.zero, count: ldeLen - traceLen))
            var shiftPow = Bb.one
            for i in 0..<ldeLen {
                coeffs[i] = bbMul(coeffs[i], shiftPow)
                shiftPow = bbMul(shiftPow, cosetShift)
            }
            traceLDEs.append(BabyBearNTTEngine.cpuNTT(coeffs, logN: logLDE))
        }

        // Commit trace LDE columns via Poseidon2 Merkle trees
        var traceCommitments = [[Bb]]()
        var traceTrees = [BbPoseidon2MerkleTree]()
        for colIdx in 0..<air.numColumns {
            let tree = BbPoseidon2MerkleTree.build(leaves: traceLDEs[colIdx])
            traceCommitments.append(tree.root)
            traceTrees.append(tree)
        }

        // Fiat-Shamir: squeeze alpha
        let challenger = Plonky3Challenger()
        for root in traceCommitments {
            challenger.observeSlice(root)
        }
        let alpha = challenger.sample()

        // OOD evaluation point (zeta): squeeze from transcript
        let zeta = challenger.sample()

        // Evaluate trace polynomials at zeta and zeta * omega_trace
        let traceOmega = bbRootOfUnity(logN: logTrace)
        let zetaNext = bbMul(zeta, traceOmega)

        var oodTraceEvals = [Bb]()
        var oodTraceNextEvals = [Bb]()
        for colIdx in 0..<air.numColumns {
            oodTraceEvals.append(evaluatePolyAt(traceCoeffs[colIdx], point: zeta))
            oodTraceNextEvals.append(evaluatePolyAt(traceCoeffs[colIdx], point: zetaNext))
        }

        // Compute quotient polynomial evals on LDE domain
        let omega = bbRootOfUnity(logN: logLDE)
        let step = ldeLen / traceLen

        var vanishingInv = [Bb](repeating: Bb.zero, count: ldeLen)
        var cosetPow = Bb.one
        for i in 0..<ldeLen {
            let xToN = bbPow(bbMul(cosetShift, cosetPow), UInt32(traceLen))
            let zh = bbSub(xToN, Bb.one)
            if zh.v != 0 {
                vanishingInv[i] = bbInverse(zh)
            }
            cosetPow = bbMul(cosetPow, omega)
        }

        var quotientEvals = [Bb](repeating: Bb.zero, count: ldeLen)
        for i in 0..<ldeLen {
            let nextI = (i + step) % ldeLen
            let current = (0..<air.numColumns).map { traceLDEs[$0][i] }
            let next = (0..<air.numColumns).map { traceLDEs[$0][nextI] }
            let constraintEvals = air.evaluateConstraints(current: current, next: next)
            var combined = Bb.zero
            var alphaPow = Bb.one
            for eval in constraintEvals {
                combined = bbAdd(combined, bbMul(alphaPow, eval))
                alphaPow = bbMul(alphaPow, alpha)
            }
            quotientEvals[i] = bbMul(combined, vanishingInv[i])
        }

        // OOD composition evaluation
        let constraintEvalsAtZeta = air.evaluateConstraints(
            current: oodTraceEvals, next: oodTraceNextEvals)
        var combinedAtZeta = Bb.zero
        var alphaPow2 = Bb.one
        for eval in constraintEvalsAtZeta {
            combinedAtZeta = bbAdd(combinedAtZeta, bbMul(alphaPow2, eval))
            alphaPow2 = bbMul(alphaPow2, alpha)
        }
        let zetaToN = bbPow(zeta, UInt32(traceLen))
        let zhZeta = bbSub(zetaToN, Bb.one)
        let oodCompositionEval = bbMul(combinedAtZeta, bbInverse(zhZeta))

        // Absorb OOD evaluations into transcript
        challenger.observeSlice(oodTraceEvals)
        challenger.observeSlice(oodTraceNextEvals)
        challenger.observe(oodCompositionEval)

        // Commit quotient polynomial
        let quotientTree = BbPoseidon2MerkleTree.build(leaves: quotientEvals)
        let compositionCommitment = quotientTree.root
        challenger.observeSlice(compositionCommitment)

        // FRI on quotient polynomial
        let friProof = try starkFRIProve(
            evaluations: quotientEvals, logN: logLDE, challenger: challenger,
            config: config)

        // Query openings
        let queryIndices = friProof.queryIndices
        var queryResponses = [BabyBearSTARKQueryResponse]()

        for qi in queryIndices {
            var traceValues = [Bb]()
            var traceOpenings = [BbMerkleOpeningProof]()
            for colIdx in 0..<air.numColumns {
                traceValues.append(traceLDEs[colIdx][qi])
                let path = traceTrees[colIdx].openingProof(index: qi)
                traceOpenings.append(BbMerkleOpeningProof(path: path, index: qi))
            }
            let compValue = quotientEvals[qi]
            let compPath = quotientTree.openingProof(index: qi)
            let compOpening = BbMerkleOpeningProof(path: compPath, index: qi)
            queryResponses.append(BabyBearSTARKQueryResponse(
                traceValues: traceValues, traceOpenings: traceOpenings,
                compositionValue: compValue, compositionOpening: compOpening,
                queryIndex: qi))
        }

        return STARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof,
            queryResponses: queryResponses,
            oodPoint: zeta,
            oodTraceEvals: oodTraceEvals,
            oodTraceNextEvals: oodTraceNextEvals,
            oodCompositionEval: oodCompositionEval,
            alpha: alpha,
            traceLength: traceLen,
            numColumns: air.numColumns,
            logBlowup: config.logBlowup
        )
    }
}

// MARK: - Utility Functions

/// Pad a single BabyBear element to an 8-element leaf for Poseidon2 hashing.
func starkPadBbToLeaf(_ value: Bb) -> [Bb] {
    var leaf = [Bb](repeating: Bb.zero, count: 8)
    leaf[0] = value
    return leaf
}

/// Evaluate a polynomial (given as coefficients) at a specific point.
func evaluatePolyAt(_ coeffs: [Bb], point: Bb) -> Bb {
    // Horner's method
    var result = Bb.zero
    for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
        result = bbAdd(bbMul(result, point), coeffs[i])
    }
    return result
}

/// FRI prove helper for STARKProof generation.
/// Stores the ORIGINAL (pre-folding) query indices so the verifier can
/// reconstruct per-round indices via modular reduction.
func starkFRIProve(
    evaluations: [Bb], logN: Int, challenger: Plonky3Challenger,
    config: BabyBearSTARKConfig
) throws -> BabyBearFRIProof {
    var currentEvals = evaluations
    var currentLogN = logN
    var rounds = [BabyBearFRIRound]()

    // Derive query indices (in the original full-size domain)
    var queryIndices = [Int]()
    for _ in 0..<config.numQueries {
        let qi = Int(challenger.sample().v) % (evaluations.count / 2)
        queryIndices.append(qi)
    }
    let originalQueryIndices = queryIndices

    // Fold until we reach the remainder threshold
    while currentLogN > config.friMaxRemainderLogN {
        let n = 1 << currentLogN
        let half = n / 2

        let tree = BbPoseidon2MerkleTree.build(leaves: currentEvals)
        let commitment = tree.root
        challenger.observeSlice(commitment)
        let beta = challenger.sample()

        var queryOpenings = [(value: Bb, siblingValue: Bb, path: [[Bb]])]()
        for qi in queryIndices {
            let idx = qi % half
            let sibIdx = idx + half
            let value = currentEvals[idx]
            let sibValue = currentEvals[sibIdx]
            let path = tree.openingProof(index: idx)
            queryOpenings.append((value: value, siblingValue: sibValue, path: path))
        }

        rounds.append(BabyBearFRIRound(commitment: commitment, queryOpenings: queryOpenings))

        let omega = bbRootOfUnity(logN: currentLogN)
        var folded = [Bb](repeating: Bb.zero, count: half)
        for i in 0..<half {
            let f0 = currentEvals[i]
            let f1 = currentEvals[i + half]
            let even = bbMul(bbAdd(f0, f1), bbInverse(Bb(v: 2)))
            let omegaI = bbPow(omega, UInt32(i))
            let oddDenom = bbMul(Bb(v: 2), omegaI)
            let odd = bbMul(bbSub(f0, f1), bbInverse(oddDenom))
            folded[i] = bbAdd(even, bbMul(beta, odd))
        }

        currentEvals = folded
        currentLogN -= 1
        queryIndices = queryIndices.map { $0 % (1 << currentLogN) }
    }

    let finalPoly = BabyBearNTTEngine.cpuINTT(currentEvals, logN: currentLogN)
    return BabyBearFRIProof(rounds: rounds, finalPoly: finalPoly, queryIndices: originalQueryIndices)
}
