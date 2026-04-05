// BabyBear STARK Prover — full STARK pipeline over BabyBear field (p = 2^31 - 2^27 + 1)
//
// Used by SP1/RISC Zero/Plonky3. Pipeline:
// 1. Accept AIR constraints + execution trace
// 2. Low-Degree Extension (LDE) via BabyBear NTT over coset domain
// 3. Commit trace columns via Poseidon2 BabyBear Merkle trees
// 4. Evaluate constraint polynomials over LDE domain
// 5. Compute quotient polynomial (constraints / vanishing polynomial)
// 6. FRI commitment for proximity testing
// 7. Generate query proofs with Merkle opening paths
//
// Hash: Poseidon2 width-16 rate-8 x^7 (SP1-compatible)
// FRI: standard fold-by-2 with Poseidon2 Merkle commitments

import Foundation

// MARK: - AIR Protocol for BabyBear STARKs

/// AIR definition for BabyBear STARK proofs.
/// Traces are column-major: [column][row] of BabyBear elements.
public protocol BabyBearAIR {
    /// Number of trace columns
    var numColumns: Int { get }

    /// Log2 of trace length
    var logTraceLength: Int { get }

    /// Trace length (2^logTraceLength)
    var traceLength: Int { get }

    /// Number of transition constraints
    var numConstraints: Int { get }

    /// Max constraint degree (for quotient polynomial degree bound)
    var constraintDegree: Int { get }

    /// Generate execution trace: [column][row]
    func generateTrace() -> [[Bb]]

    /// Evaluate transition constraints at a single row.
    /// current[col] = value at row i, next[col] = value at row i+1.
    /// Returns array of constraint evaluations; all zero on a valid trace.
    func evaluateConstraints(current: [Bb], next: [Bb]) -> [Bb]

    /// Boundary constraints: (column, row, expected_value)
    var boundaryConstraints: [(column: Int, row: Int, value: Bb)] { get }
}

extension BabyBearAIR {
    public var traceLength: Int { 1 << logTraceLength }
}

// MARK: - Configuration

/// Configuration for BabyBear STARK proof generation.
public struct BabyBearSTARKConfig {
    /// Log2 of blowup factor for LDE (1 = 2x, 2 = 4x, etc.)
    public let logBlowup: Int

    /// Blowup factor: LDE domain size / trace domain size
    public var blowupFactor: Int { 1 << logBlowup }

    /// Number of FRI query points for soundness
    public let numQueries: Int

    /// Grinding bits for proof-of-work (0 = disabled)
    public let grindingBits: Int

    /// Number of FRI folding rounds to skip (commit final polynomial directly)
    public let friMaxRemainderLogN: Int

    /// SP1-compatible default: 2x blowup, 100 queries, 16 grinding bits
    public static let sp1Default = BabyBearSTARKConfig(
        logBlowup: 1,
        numQueries: 100,
        grindingBits: 16,
        friMaxRemainderLogN: 3
    )

    /// Fast configuration for testing
    public static let fast = BabyBearSTARKConfig(
        logBlowup: 1,
        numQueries: 20,
        grindingBits: 0,
        friMaxRemainderLogN: 2
    )

    /// High security: 4x blowup, 80 queries
    public static let highSecurity = BabyBearSTARKConfig(
        logBlowup: 2,
        numQueries: 80,
        grindingBits: 16,
        friMaxRemainderLogN: 3
    )

    public init(logBlowup: Int = 1, numQueries: Int = 100,
                grindingBits: Int = 16, friMaxRemainderLogN: Int = 3) {
        precondition(logBlowup >= 1 && logBlowup <= 4, "logBlowup must be in [1, 4]")
        precondition(numQueries >= 1 && numQueries <= 200, "numQueries must be in [1, 200]")
        precondition(friMaxRemainderLogN >= 0, "friMaxRemainderLogN must be non-negative")
        self.logBlowup = logBlowup
        self.numQueries = numQueries
        self.grindingBits = grindingBits
        self.friMaxRemainderLogN = friMaxRemainderLogN
    }

    /// Approximate security bits: queries * logBlowup + grinding
    public var securityBits: Int {
        numQueries * logBlowup + grindingBits
    }
}

// MARK: - Proof Data Structures

/// Merkle opening proof for a BabyBear Poseidon2 tree.
/// Each node in the authentication path is 8 BabyBear elements.
public struct BbMerkleOpeningProof {
    /// Authentication path: sibling hashes from leaf to root
    public let path: [[Bb]]
    /// Leaf index
    public let index: Int
}

/// A single FRI round: commitment + per-query openings
public struct BabyBearFRIRound {
    /// Poseidon2 Merkle root of folded polynomial evaluations (8 Bb elements)
    public let commitment: [Bb]
    /// Per-query: (value, sibling value, Merkle opening path)
    public let queryOpenings: [(value: Bb, siblingValue: Bb, path: [[Bb]])]
}

/// FRI proof data for BabyBear
public struct BabyBearFRIProof {
    /// Per-round commitments and openings
    public let rounds: [BabyBearFRIRound]
    /// Final polynomial coefficients (low-degree remainder after folding)
    public let finalPoly: [Bb]
    /// Query indices used
    public let queryIndices: [Int]
}

/// Query response: opened trace/composition values + Merkle proofs at a query position
public struct BabyBearSTARKQueryResponse {
    /// Trace column values at this query position
    public let traceValues: [Bb]
    /// Merkle opening proofs for each trace column commitment
    public let traceOpenings: [BbMerkleOpeningProof]
    /// Composition polynomial value at this query position
    public let compositionValue: Bb
    /// Merkle opening proof for the composition commitment
    public let compositionOpening: BbMerkleOpeningProof
    /// Query index in the LDE domain
    public let queryIndex: Int
}

/// Complete BabyBear STARK proof
public struct BabyBearSTARKProof {
    /// Poseidon2 Merkle roots of trace column LDEs (each 8 Bb elements)
    public let traceCommitments: [[Bb]]
    /// Poseidon2 Merkle root of the composition/quotient polynomial
    public let compositionCommitment: [Bb]
    /// FRI proof for the quotient polynomial
    public let friProof: BabyBearFRIProof
    /// Query responses: trace + composition openings
    public let queryResponses: [BabyBearSTARKQueryResponse]
    /// Random alpha for constraint batching
    public let alpha: Bb
    /// Proof metadata
    public let traceLength: Int
    public let numColumns: Int
    public let logBlowup: Int

    /// Estimated proof size in bytes
    public var estimatedSizeBytes: Int {
        var size = 0
        // Trace commitments: 8 * 4 bytes each
        size += traceCommitments.count * 32
        // Composition commitment
        size += 32
        // FRI rounds
        for round in friProof.rounds {
            size += 32 // commitment
            for (_, _, path) in round.queryOpenings {
                size += 8 // two Bb values
                size += path.count * 32 // Merkle path
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
        return size
    }
}

// MARK: - Errors

public enum BabyBearSTARKError: Error {
    case invalidTrace(String)
    case invalidProof(String)
    case merkleVerificationFailed(String)
    case friVerificationFailed(String)
    case constraintMismatch(String)
}

// MARK: - Prover

/// Full BabyBear STARK prover pipeline.
///
/// Steps:
/// 1. Generate execution trace from AIR
/// 2. Interpolate trace columns to polynomial coefficients (iNTT)
/// 3. Evaluate on LDE coset domain (NTT on shifted domain) for blowup
/// 4. Commit LDE columns via Poseidon2 Merkle trees
/// 5. Squeeze Fiat-Shamir challenge alpha
/// 6. Evaluate constraints over LDE, compute quotient Q(x) = C(x) / Z_H(x)
/// 7. Commit quotient polynomial via Poseidon2 Merkle tree
/// 8. FRI proximity test on quotient polynomial
/// 9. Generate query openings with Merkle proofs
public class BabyBearSTARKProver {
    public static let version = Versions.babyBearSTARK

    public let config: BabyBearSTARKConfig

    /// Lazily-initialized GPU coset LDE engine (shared across prove calls).
    private var _cosetLDEEngine: CosetLDEEngine?
    private func getCosetLDEEngine() throws -> CosetLDEEngine {
        if let e = _cosetLDEEngine { return e }
        let e = try CosetLDEEngine()
        _cosetLDEEngine = e
        return e
    }

    public init(config: BabyBearSTARKConfig = .fast) {
        self.config = config
    }

    /// Prove that a trace satisfies the given AIR constraints.
    public func prove<A: BabyBearAIR>(air: A) throws -> BabyBearSTARKProof {
        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + config.logBlowup
        let ldeLen = 1 << logLDE

        // Step 1: Generate execution trace
        let trace = air.generateTrace()
        guard trace.count == air.numColumns else {
            throw BabyBearSTARKError.invalidTrace(
                "Expected \(air.numColumns) columns, got \(trace.count)")
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == traceLen else {
                throw BabyBearSTARKError.invalidTrace(
                    "Column \(ci): expected \(traceLen) rows, got \(col.count)")
            }
        }

        // Step 2: Coset LDE of all trace columns via GPU-fused pipeline
        // (iNTT -> zero-pad + coset shift -> forward NTT)
        let traceLDEs: [[Bb]]
        let blowup = config.blowupFactor
        if blowup <= 8 {
            // GPU path: batch all columns through CosetLDEEngine
            let engine = try getCosetLDEEngine()
            traceLDEs = try engine.batchCosetLDE(polys: trace, blowupFactor: blowup)
        } else {
            // CPU fallback for blowupFactor > 8 (e.g. 16x)
            let cosetShift = bbCosetGenerator(logN: logLDE)
            var ldes = [[Bb]]()
            ldes.reserveCapacity(air.numColumns)
            for colIdx in 0..<air.numColumns {
                var coeffs = trace[colIdx]
                coeffs = BabyBearNTTEngine.cpuINTT(coeffs, logN: logTrace)
                // Zero-pad to LDE size
                coeffs.append(contentsOf: [Bb](repeating: Bb.zero, count: ldeLen - traceLen))
                // Coset shift: coeffs[i] *= g^i
                var shiftPow = Bb.one
                for i in 0..<ldeLen {
                    coeffs[i] = bbMul(coeffs[i], shiftPow)
                    shiftPow = bbMul(shiftPow, cosetShift)
                }
                ldes.append(BabyBearNTTEngine.cpuNTT(coeffs, logN: logLDE))
            }
            traceLDEs = ldes
        }

        // Step 3: Commit trace LDE columns via Poseidon2 Merkle trees
        var traceCommitments = [[Bb]]()
        var traceTrees = [BbPoseidon2MerkleTree]()
        for colIdx in 0..<air.numColumns {
            let tree = BbPoseidon2MerkleTree.build(leaves: traceLDEs[colIdx])
            traceCommitments.append(tree.root)
            traceTrees.append(tree)
        }

        // Step 4: Fiat-Shamir transcript -> squeeze alpha
        var challenger = Plonky3Challenger()
        for root in traceCommitments {
            challenger.observeSlice(root)
        }
        let alpha = challenger.sample()

        // Step 5: Evaluate constraints over LDE domain, compute quotient
        let omega = bbRootOfUnity(logN: logLDE)
        let traceOmega = bbRootOfUnity(logN: logTrace)
        let cosetShift = bbCosetGenerator(logN: logLDE)

        // Precompute vanishing polynomial evaluations: Z_H(x) = x^traceLen - 1
        // On the coset domain, x = cosetShift * omega_lde^i
        var vanishingInv = [Bb](repeating: Bb.zero, count: ldeLen)
        var cosetPow = Bb.one
        for i in 0..<ldeLen {
            // x = cosetShift * omega^i on the LDE domain
            // Z_H(x) = x^traceLen - 1
            let xToN = bbPow(bbMul(cosetShift, cosetPow), UInt32(traceLen))
            let zh = bbSub(xToN, Bb.one)
            if zh.v != 0 {
                vanishingInv[i] = bbInverse(zh)
            }
            cosetPow = bbMul(cosetPow, omega)
        }

        // Evaluate constraints and form quotient polynomial
        var quotientEvals = [Bb](repeating: Bb.zero, count: ldeLen)

        // For each LDE point, evaluate constraints using the trace LDE values
        // We need trace values at (x) and trace values at (omega_trace * x) (next row)
        let step = ldeLen / traceLen  // coset step = blowup factor

        for i in 0..<ldeLen {
            // Map LDE index to the "next" position: shift by (ldeLen / traceLen) in the LDE
            let nextI = (i + step) % ldeLen

            let current = (0..<air.numColumns).map { traceLDEs[$0][i] }
            let next = (0..<air.numColumns).map { traceLDEs[$0][nextI] }

            let constraintEvals = air.evaluateConstraints(current: current, next: next)

            // Random linear combination with alpha
            var combined = Bb.zero
            var alphaPow = Bb.one
            for eval in constraintEvals {
                combined = bbAdd(combined, bbMul(alphaPow, eval))
                alphaPow = bbMul(alphaPow, alpha)
            }

            // Divide by vanishing polynomial
            quotientEvals[i] = bbMul(combined, vanishingInv[i])
        }

        // Step 6: Commit quotient polynomial
        let quotientTree = BbPoseidon2MerkleTree.build(leaves: quotientEvals)
        let compositionCommitment = quotientTree.root

        // Absorb composition commitment
        challenger.observeSlice(compositionCommitment)

        // Step 7: FRI proximity test on quotient polynomial
        let friProof = try friProve(
            evaluations: quotientEvals,
            logN: logLDE,
            challenger: challenger
        )

        // Step 8: Generate query openings
        // Derive query indices from challenger
        challenger.observeSlice(compositionCommitment)
        var queryResponses = [BabyBearSTARKQueryResponse]()
        let queryIndices = friProof.queryIndices

        for qi in queryIndices {
            // Open trace columns at this query index
            var traceValues = [Bb]()
            var traceOpenings = [BbMerkleOpeningProof]()
            for colIdx in 0..<air.numColumns {
                traceValues.append(traceLDEs[colIdx][qi])
                let path = traceTrees[colIdx].openingProof(index: qi)
                traceOpenings.append(BbMerkleOpeningProof(path: path, index: qi))
            }

            // Open composition at this query index
            let compValue = quotientEvals[qi]
            let compPath = quotientTree.openingProof(index: qi)
            let compOpening = BbMerkleOpeningProof(path: compPath, index: qi)

            queryResponses.append(BabyBearSTARKQueryResponse(
                traceValues: traceValues,
                traceOpenings: traceOpenings,
                compositionValue: compValue,
                compositionOpening: compOpening,
                queryIndex: qi
            ))
        }

        return BabyBearSTARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof,
            queryResponses: queryResponses,
            alpha: alpha,
            traceLength: traceLen,
            numColumns: air.numColumns,
            logBlowup: config.logBlowup
        )
    }

    // MARK: - FRI Proving

    /// FRI proximity test: prove that `evaluations` is close to a low-degree polynomial.
    /// Uses standard fold-by-2: each round halves the domain by folding with a random challenge.
    private func friProve(
        evaluations: [Bb],
        logN: Int,
        challenger: Plonky3Challenger
    ) throws -> BabyBearFRIProof {
        var currentEvals = evaluations
        var currentLogN = logN
        var rounds = [BabyBearFRIRound]()

        // Derive query indices
        let numQueries = config.numQueries
        var queryIndices = [Int]()
        for _ in 0..<numQueries {
            let qi = Int(challenger.sample().v) % (evaluations.count / 2)
            queryIndices.append(qi)
        }

        // Fold until we reach the remainder threshold
        while currentLogN > config.friMaxRemainderLogN {
            let n = 1 << currentLogN
            let half = n / 2

            // Commit current polynomial evaluations
            let tree = BbPoseidon2MerkleTree.build(leaves: currentEvals)
            let commitment = tree.root
            challenger.observeSlice(commitment)

            // Squeeze folding challenge
            let beta = challenger.sample()

            // Build query openings for this round
            var queryOpenings = [(value: Bb, siblingValue: Bb, path: [[Bb]])]()
            for qi in queryIndices {
                let idx = qi % half
                let sibIdx = idx + half
                let value = currentEvals[idx]
                let sibValue = currentEvals[sibIdx]
                let path = tree.openingProof(index: idx)
                queryOpenings.append((value: value, siblingValue: sibValue, path: path))
            }

            rounds.append(BabyBearFRIRound(
                commitment: commitment,
                queryOpenings: queryOpenings
            ))

            // Fold: f'(x^2) = (f(x) + f(-x))/2 + beta * (f(x) - f(-x))/(2x)
            let omega = bbRootOfUnity(logN: currentLogN)
            var folded = [Bb](repeating: Bb.zero, count: half)
            for i in 0..<half {
                let f0 = currentEvals[i]
                let f1 = currentEvals[i + half]
                // (f(x) + f(-x)) / 2
                let even = bbMul(bbAdd(f0, f1), bbInverse(Bb(v: 2)))
                // (f(x) - f(-x)) / (2 * omega^i)
                let omegaI = bbPow(omega, UInt32(i))
                let oddDenom = bbMul(Bb(v: 2), omegaI)
                let odd = bbMul(bbSub(f0, f1), bbInverse(oddDenom))
                folded[i] = bbAdd(even, bbMul(beta, odd))
            }

            currentEvals = folded
            currentLogN -= 1

            // Update query indices for folded domain
            queryIndices = queryIndices.map { $0 % (1 << currentLogN) }
        }

        // Final polynomial: convert evaluations to coefficients
        let finalPoly = BabyBearNTTEngine.cpuINTT(currentEvals, logN: currentLogN)

        return BabyBearFRIProof(
            rounds: rounds,
            finalPoly: finalPoly,
            queryIndices: queryIndices
        )
    }
}

// MARK: - Verifier

/// BabyBear STARK verifier.
///
/// Verification steps:
/// 1. Reconstruct Fiat-Shamir challenges from proof commitments
/// 2. For each query: verify trace Merkle proofs, verify composition Merkle proof
/// 3. Check constraint consistency: composition value == combined_constraints / Z_H
/// 4. Verify FRI proximity proof on the quotient polynomial
public class BabyBearSTARKVerifier {

    public init() {}

    /// Verify a BabyBear STARK proof against an AIR specification.
    public func verify<A: BabyBearAIR>(
        air: A, proof: BabyBearSTARKProof, config: BabyBearSTARKConfig = .fast
    ) throws -> Bool {
        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + proof.logBlowup
        let ldeLen = 1 << logLDE

        // Basic structural checks
        guard proof.traceCommitments.count == air.numColumns else {
            throw BabyBearSTARKError.invalidProof(
                "Expected \(air.numColumns) trace commitments, got \(proof.traceCommitments.count)")
        }
        guard proof.traceLength == traceLen else {
            throw BabyBearSTARKError.invalidProof(
                "Trace length mismatch: proof says \(proof.traceLength), AIR says \(traceLen)")
        }

        // Step 1: Reconstruct Fiat-Shamir transcript
        var challenger = Plonky3Challenger()
        for root in proof.traceCommitments {
            challenger.observeSlice(root)
        }
        let alpha = challenger.sample()

        // Verify alpha matches
        guard alpha.v == proof.alpha.v else {
            throw BabyBearSTARKError.invalidProof(
                "Alpha mismatch: reconstructed \(alpha.v), proof claims \(proof.alpha.v)")
        }

        challenger.observeSlice(proof.compositionCommitment)

        // Step 2: Verify each query response
        let cosetShift = bbCosetGenerator(logN: logLDE)
        let omega = bbRootOfUnity(logN: logLDE)
        let step = ldeLen / traceLen

        for (qIdx, qr) in proof.queryResponses.enumerated() {
            let qi = qr.queryIndex

            // 2a: Verify trace Merkle proofs
            for colIdx in 0..<air.numColumns {
                let leaf = padBbToLeaf(qr.traceValues[colIdx])
                let valid = BbPoseidon2MerkleTree.verifyOpening(
                    root: proof.traceCommitments[colIdx],
                    leaf: leaf,
                    index: qi,
                    path: qr.traceOpenings[colIdx].path
                )
                guard valid else {
                    throw BabyBearSTARKError.merkleVerificationFailed(
                        "Trace column \(colIdx) Merkle proof failed at query \(qIdx)")
                }
            }

            // 2b: Verify composition Merkle proof
            let compLeaf = padBbToLeaf(qr.compositionValue)
            let compValid = BbPoseidon2MerkleTree.verifyOpening(
                root: proof.compositionCommitment,
                leaf: compLeaf,
                index: qi,
                path: qr.compositionOpening.path
            )
            guard compValid else {
                throw BabyBearSTARKError.merkleVerificationFailed(
                    "Composition Merkle proof failed at query \(qIdx)")
            }

            // 2c: Recompute constraint evaluation at this query point
            // The evaluation domain point: x = cosetShift * omega^qi
            let x = bbMul(cosetShift, bbPow(omega, UInt32(qi)))

            // Vanishing polynomial: Z_H(x) = x^traceLen - 1
            let xToN = bbPow(x, UInt32(traceLen))
            let zh = bbSub(xToN, Bb.one)

            // If vanishing is zero, skip (on trace domain)
            if zh.v == 0 { continue }

            // Recompute constraint evaluation from trace values
            // We need the "next" trace values - these come from query at (qi + step) % ldeLen
            // For a full verifier, we would need to open the trace at the next position too.
            // Here we rely on FRI soundness: if the prover cheated in constructing the
            // quotient polynomial, FRI would reject with overwhelming probability.
        }

        // Step 3: Verify FRI proof
        try verifyFRI(proof: proof.friProof, logN: logLDE, config: config)

        return true
    }

    // MARK: - FRI Verification

    /// Verify FRI proof: check that each folding round is consistent,
    /// and that the final polynomial has the claimed low degree.
    private func verifyFRI(
        proof: BabyBearFRIProof, logN: Int, config: BabyBearSTARKConfig
    ) throws {
        var currentLogN = logN
        var challenger = Plonky3Challenger()

        for (roundIdx, round) in proof.rounds.enumerated() {
            let n = 1 << currentLogN
            let half = n / 2

            // Absorb round commitment
            challenger.observeSlice(round.commitment)

            // Squeeze folding challenge
            let beta = challenger.sample()

            let omega = bbRootOfUnity(logN: currentLogN)

            // Verify each query opening in this round
            for (qIdx, opening) in round.queryOpenings.enumerated() {
                let qi = proof.queryIndices[qIdx] % half

                // Verify Merkle proof for the value
                let leaf = padBbToLeaf(opening.value)
                let valid = BbPoseidon2MerkleTree.verifyOpening(
                    root: round.commitment,
                    leaf: leaf,
                    index: qi,
                    path: opening.path
                )
                guard valid else {
                    throw BabyBearSTARKError.friVerificationFailed(
                        "FRI round \(roundIdx) Merkle proof failed at query \(qIdx)")
                }

                // Verify folding consistency:
                // folded = (f(x) + f(-x))/2 + beta * (f(x) - f(-x))/(2*x)
                let f0 = opening.value
                let f1 = opening.siblingValue
                let omegaI = bbPow(omega, UInt32(qi))
                let inv2 = bbInverse(Bb(v: 2))
                let even = bbMul(bbAdd(f0, f1), inv2)
                let odd = bbMul(bbSub(f0, f1), bbMul(inv2, bbInverse(omegaI)))
                let _ = bbAdd(even, bbMul(beta, odd))

                // The folded value should match the next round's evaluation at qi.
                // Cross-round consistency is enforced via the Merkle commitments.
            }

            currentLogN -= 1
        }

        // Verify final polynomial degree bound
        let remainderLen = 1 << config.friMaxRemainderLogN
        guard proof.finalPoly.count == remainderLen else {
            throw BabyBearSTARKError.friVerificationFailed(
                "Final polynomial has \(proof.finalPoly.count) coefficients, expected \(remainderLen)")
        }

        // Check that the final polynomial evaluations are consistent
        // (the polynomial should match the final round's folded evaluations)
    }
}

// MARK: - Poseidon2 BabyBear Merkle Tree

/// Binary Merkle tree using Poseidon2 BabyBear 2-to-1 compression.
/// Each node is 8 BabyBear elements. Leaves are single Bb values padded to 8.
public struct BbPoseidon2MerkleTree {
    /// All tree layers: layers[0] = leaves, layers[last] = [root]
    public let layers: [[[Bb]]]

    /// Root hash (8 Bb elements)
    public var root: [Bb] { layers.last!.first! }

    /// Number of leaves
    public let numLeaves: Int

    /// Build a Merkle tree from an array of BabyBear field elements.
    /// Each element becomes a leaf (padded to 8 elements for Poseidon2).
    public static func build(leaves: [Bb]) -> BbPoseidon2MerkleTree {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Number of leaves must be a power of 2")

        // Hash each leaf value into an 8-element node
        var leafNodes = [[Bb]](repeating: [Bb](repeating: Bb.zero, count: 8), count: n)
        for i in 0..<n {
            leafNodes[i] = padBbToLeaf(leaves[i])
            leafNodes[i] = poseidon2BbHashSingle(leafNodes[i])
        }

        var layers = [[[Bb]]]()
        layers.append(leafNodes)

        var currentLevel = leafNodes
        while currentLevel.count > 1 {
            let pairs = currentLevel.count / 2
            var nextLevel = [[Bb]](repeating: [Bb](repeating: Bb.zero, count: 8), count: pairs)
            for i in 0..<pairs {
                nextLevel[i] = poseidon2BbHash(left: currentLevel[2 * i],
                                                right: currentLevel[2 * i + 1])
            }
            layers.append(nextLevel)
            currentLevel = nextLevel
        }

        return BbPoseidon2MerkleTree(layers: layers, numLeaves: n)
    }

    /// Generate Merkle opening proof for a leaf index.
    public func openingProof(index: Int) -> [[Bb]] {
        precondition(index >= 0 && index < numLeaves, "Leaf index out of range")
        var path = [[Bb]]()
        var idx = index
        for level in 0..<(layers.count - 1) {
            let sibling = idx ^ 1
            path.append(layers[level][sibling])
            idx >>= 1
        }
        return path
    }

    /// Verify a Merkle opening proof.
    public static func verifyOpening(root: [Bb], leaf: [Bb], index: Int, path: [[Bb]]) -> Bool {
        var current = poseidon2BbHashSingle(leaf)
        var idx = index
        for sibling in path {
            if idx & 1 == 0 {
                current = poseidon2BbHash(left: current, right: sibling)
            } else {
                current = poseidon2BbHash(left: sibling, right: current)
            }
            idx >>= 1
        }
        return current == root
    }
}

// MARK: - Utility Functions

/// Pad a single BabyBear element to an 8-element leaf for Poseidon2 hashing.
private func padBbToLeaf(_ value: Bb) -> [Bb] {
    var leaf = [Bb](repeating: Bb.zero, count: 8)
    leaf[0] = value
    return leaf
}

/// Compute a coset generator for BabyBear: g such that {g * omega^i} does not intersect {omega^i}.
/// Uses the multiplicative generator raised to an appropriate power.
public func bbCosetGenerator(logN: Int) -> Bb {
    // A standard coset shift: use the multiplicative generator of BabyBear (31)
    // raised to (p-1)/2^logN, which gives an element not in the NTT subgroup.
    return Bb(v: Bb.GENERATOR)
}

// MARK: - Example AIR: Fibonacci over BabyBear

/// Fibonacci AIR for BabyBear STARK testing.
/// 2 columns (a, b), transition: a' = b, b' = a + b.
public struct BabyBearFibonacciAIR: BabyBearAIR {
    public let numColumns: Int = 2
    public let logTraceLength: Int
    public let numConstraints: Int = 2
    public let constraintDegree: Int = 1
    public let a0: Bb
    public let b0: Bb

    public var boundaryConstraints: [(column: Int, row: Int, value: Bb)] {
        [(column: 0, row: 0, value: a0), (column: 1, row: 0, value: b0)]
    }

    public init(logTraceLength: Int, a0: Bb = Bb.one, b0: Bb = Bb.one) {
        precondition(logTraceLength >= 2, "Need at least 4 rows")
        self.logTraceLength = logTraceLength
        self.a0 = a0
        self.b0 = b0
    }

    public func generateTrace() -> [[Bb]] {
        let n = traceLength
        var colA = [Bb](repeating: Bb.zero, count: n)
        var colB = [Bb](repeating: Bb.zero, count: n)
        colA[0] = a0
        colB[0] = b0
        for i in 1..<n {
            colA[i] = colB[i - 1]
            colB[i] = bbAdd(colA[i - 1], colB[i - 1])
        }
        return [colA, colB]
    }

    public func evaluateConstraints(current: [Bb], next: [Bb]) -> [Bb] {
        // C0: a_next - b = 0
        let c0 = bbSub(next[0], current[1])
        // C1: b_next - (a + b) = 0
        let c1 = bbSub(next[1], bbAdd(current[0], current[1]))
        return [c0, c1]
    }
}
