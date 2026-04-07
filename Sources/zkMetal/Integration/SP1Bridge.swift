// SP1 (Succinct) Prover Integration Bridge
//
// Connects SP1's proving pipeline to zkMetal's GPU-accelerated kernels.
// SP1 uses Plonky3 under the hood with the BabyBearPoseidon2 configuration:
//   - Field: BabyBear (p = 2^31 - 2^27 + 1)
//   - Hash: Poseidon2 width-16, rate-8, x^7 S-box
//   - FRI: fold-by-4, 2x blowup, 100 queries, Poseidon2 Merkle commitments
//   - Challenger: Poseidon2 duplex sponge
//
// This bridge provides GPU-accelerated implementations of SP1's three
// hottest proving phases: trace commitment, FRI folding, and constraint
// evaluation. Each component can be used standalone or composed into
// the full SP1ProverPipeline.

import Foundation
import Metal

// MARK: - SP1 Prover Configuration

/// Configuration matching SP1's default BabyBearPoseidon2 proving parameters.
///
/// SP1 uses Plonky3 with fold-by-4 FRI, 2x blowup, 100 queries, and
/// Poseidon2 width-16 for both hashing and the Fiat-Shamir transcript.
public struct SP1ProverConfig {
    // -- Field --
    /// BabyBear prime: p = 2^31 - 2^27 + 1
    public static let fieldModulus: UInt32 = Bb.P

    // -- Hash --
    /// Poseidon2 state width (BabyBear elements per permutation)
    public static let poseidon2Width: Int = Poseidon2BabyBearConfig.t         // 16
    /// Poseidon2 rate (elements absorbed per permutation call)
    public static let poseidon2Rate: Int = Poseidon2BabyBearConfig.rate       // 8
    /// Poseidon2 capacity (elements reserved for domain separation)
    public static let poseidon2Capacity: Int = Poseidon2BabyBearConfig.capacity // 8
    /// Hash digest: 8 BabyBear elements (256 bits)
    public static let digestElements: Int = Poseidon2BabyBearEngine.nodeSize  // 8

    // -- FRI --
    /// SP1 fold-by-4 (log2 = 2), matching Plonky3's FRI configuration
    public static let friLogFoldFactor: Int = 2
    /// FRI folding factor: 4 evaluations folded per round
    public static let friFoldFactor: Int = 4

    /// Log2 of the blowup factor for the low-degree extension
    public let logBlowup: Int
    /// Blowup factor: LDE domain size / trace domain size
    public var blowupFactor: Int { 1 << logBlowup }

    /// Number of FRI query points for soundness amplification
    public let numQueries: Int

    /// Grinding bits for proof-of-work (SP1 default: 16)
    public let grindingBits: Int

    /// Maximum log-size of the FRI remainder polynomial
    public let friMaxRemainderLogN: Int

    /// Number of BabyBear extension field elements (EF4)
    public static let extensionDegree: Int = 4

    /// Whether to use Metal GPU acceleration (falls back to CPU if false or unavailable)
    public let useGPU: Bool

    /// SP1 default: 2x blowup, 100 FRI queries, 16 grinding bits
    public static let sp1Default = SP1ProverConfig(
        logBlowup: 1,
        numQueries: 100,
        grindingBits: 16,
        friMaxRemainderLogN: 3,
        useGPU: true
    )

    /// Fast configuration for integration testing
    public static let fast = SP1ProverConfig(
        logBlowup: 1,
        numQueries: 20,
        grindingBits: 0,
        friMaxRemainderLogN: 2,
        useGPU: true
    )

    public init(logBlowup: Int = 1, numQueries: Int = 100,
                grindingBits: Int = 16, friMaxRemainderLogN: Int = 3,
                useGPU: Bool = true) {
        precondition(logBlowup >= 1 && logBlowup <= 4, "logBlowup must be in [1, 4]")
        precondition(numQueries >= 1 && numQueries <= 200, "numQueries must be in [1, 200]")
        precondition(friMaxRemainderLogN >= 0, "friMaxRemainderLogN must be non-negative")
        self.logBlowup = logBlowup
        self.numQueries = numQueries
        self.grindingBits = grindingBits
        self.friMaxRemainderLogN = friMaxRemainderLogN
        self.useGPU = useGPU
    }

    /// Approximate security bits: queries * logBlowup + grinding
    public var securityBits: Int {
        numQueries * logBlowup + grindingBits
    }
}

// MARK: - SP1 Trace Committer

/// GPU-accelerated trace commitment for SP1's execution traces.
///
/// SP1 represents execution traces as BabyBear matrices (columns x rows).
/// The commitment pipeline:
///   1. Interpolate each column to coefficient form (GPU iNTT)
///   2. Low-degree extend onto a coset domain (GPU NTT at shifted domain)
///   3. Hash each row of the LDE via Poseidon2 sponge into a Merkle leaf
///   4. Build a Poseidon2 Merkle tree over the leaves (GPU batch hashing)
///
/// The result is a Poseidon2 Merkle root (8 BabyBear elements) that
/// commits to the entire trace, plus the LDE data for later opening proofs.
public class SP1TraceCommitter {
    private let config: SP1ProverConfig
    private var nttEngine: BabyBearNTTEngine?
    private var hashEngine: Poseidon2BabyBearEngine?

    /// Committed trace data, retained for opening proofs.
    public struct TraceCommitmentData {
        /// Poseidon2 Merkle root: 8 BabyBear elements
        public let root: [Bb]
        /// LDE columns: [column][row] over the extended domain
        public let ldeColumns: [[Bb]]
        /// Merkle tree layers for opening proofs
        public let merkleLayers: [[[Bb]]]
        /// Log2 of LDE domain size
        public let logLDESize: Int
        /// Number of original trace columns
        public let numColumns: Int
        /// Original trace length (before extension)
        public let traceLength: Int
    }

    public init(config: SP1ProverConfig = .sp1Default) {
        self.config = config
        if config.useGPU {
            self.nttEngine = try? BabyBearNTTEngine()
            self.hashEngine = try? Poseidon2BabyBearEngine()
        }
    }

    /// Commit to an SP1 execution trace.
    ///
    /// - Parameter columns: Trace data as [column][row] of BabyBear elements.
    ///   All columns must have the same power-of-2 length.
    /// - Returns: Commitment data including the Merkle root and retained LDE.
    public func commit(columns: [[Bb]]) throws -> TraceCommitmentData {
        guard !columns.isEmpty else {
            throw BabyBearSTARKError.invalidTrace("Empty trace")
        }
        let numCols = columns.count
        let traceLen = columns[0].count
        guard traceLen > 0 && (traceLen & (traceLen - 1)) == 0 else {
            throw BabyBearSTARKError.invalidTrace("Trace length must be a power of 2")
        }
        for (i, col) in columns.enumerated() {
            guard col.count == traceLen else {
                throw BabyBearSTARKError.invalidTrace(
                    "Column \(i) has \(col.count) rows, expected \(traceLen)")
            }
        }

        let logTrace = traceLen.trailingZeroBitCount
        let logLDE = logTrace + config.logBlowup
        let ldeLen = 1 << logLDE
        let cosetShift = bbCosetGenerator(logN: logLDE)

        // Step 1-2: Interpolate + LDE for each column
        var ldeColumns = [[Bb]]()
        ldeColumns.reserveCapacity(numCols)

        for colIdx in 0..<numCols {
            let lde = try computeLDE(
                column: columns[colIdx],
                logTrace: logTrace,
                logLDE: logLDE,
                ldeLen: ldeLen,
                cosetShift: cosetShift
            )
            ldeColumns.append(lde)
        }

        // Step 3-4: Row-wise Poseidon2 hash into Merkle leaves, then build tree
        let commitment = try Plonky3TraceCommitment.commit(
            columns: ldeColumns,
            engine: hashEngine
        )

        return TraceCommitmentData(
            root: commitment.root,
            ldeColumns: ldeColumns,
            merkleLayers: commitment.layers,
            logLDESize: logLDE,
            numColumns: numCols,
            traceLength: traceLen
        )
    }

    /// Compute the low-degree extension of a single trace column.
    ///
    /// Pipeline: iNTT (eval -> coeffs) -> pad -> coset shift -> NTT (coeffs -> LDE evals)
    private func computeLDE(
        column: [Bb], logTrace: Int, logLDE: Int, ldeLen: Int, cosetShift: Bb
    ) throws -> [Bb] {
        let traceLen = column.count

        // iNTT: evaluation form -> coefficient form
        var coeffs: [Bb]
        if let engine = nttEngine {
            coeffs = try engine.intt(column)
        } else {
            coeffs = BabyBearNTTEngine.cpuINTT(column, logN: logTrace)
        }

        // Pad coefficients to LDE size
        coeffs.append(contentsOf: [Bb](repeating: Bb.zero, count: ldeLen - traceLen))

        // Apply coset shift: multiply coefficient i by cosetShift^i
        var shifted = [Bb](repeating: Bb.zero, count: ldeLen)
        var shiftPow = Bb.one
        for i in 0..<traceLen {
            shifted[i] = bbMul(coeffs[i], shiftPow)
            shiftPow = bbMul(shiftPow, cosetShift)
        }

        // NTT on shifted coefficients -> LDE evaluations
        let lde: [Bb]
        if let engine = nttEngine {
            lde = try engine.ntt(shifted)
        } else {
            lde = BabyBearNTTEngine.cpuNTT(shifted, logN: logLDE)
        }

        return lde
    }

    /// Generate a Merkle opening proof for a specific row in the committed trace.
    public func openingProof(data: TraceCommitmentData, row: Int) -> [[Bb]] {
        precondition(row >= 0 && row < (1 << data.logLDESize), "Row index out of range")
        var path = [[Bb]]()
        var idx = row
        for level in 0..<(data.merkleLayers.count - 1) {
            let sibling = idx ^ 1
            path.append(data.merkleLayers[level][sibling])
            idx >>= 1
        }
        return path
    }
}

// MARK: - SP1 FRI Folder

/// GPU-accelerated FRI folding for SP1's fold-by-4 protocol.
///
/// SP1 uses Plonky3's FRI with fold-by-4: each round reduces the polynomial
/// degree by a factor of 4 using three random challenges (or equivalently,
/// one extension-field challenge decomposed into base-field components).
///
/// Each folding round:
///   1. Commit current evaluations via Poseidon2 Merkle tree (GPU)
///   2. Squeeze folding challenge from Fiat-Shamir transcript
///   3. Fold 4 evaluations into 1 using the challenge
///   4. Repeat until remainder is small enough
///
/// The final small polynomial is sent in the clear.
public class SP1FRIFolder {
    private let config: SP1ProverConfig
    private var hashEngine: Poseidon2BabyBearEngine?

    /// A single FRI round's data.
    public struct FRIRoundData {
        /// Poseidon2 Merkle root of this round's evaluations (8 Bb elements)
        public let commitment: [Bb]
        /// Merkle tree layers for opening proofs
        public let merkleLayers: [[[Bb]]]
        /// Folding challenge used in this round
        public let beta: Bb
        /// Per-query openings: (values at 4 coset positions, Merkle paths)
        public let queryOpenings: [(values: [Bb], paths: [[[Bb]]])]
    }

    /// Complete FRI proof for SP1.
    public struct SP1FRIProof {
        /// Per-round data
        public let rounds: [FRIRoundData]
        /// Final remainder polynomial coefficients
        public let finalPoly: [Bb]
        /// Query indices used
        public let queryIndices: [Int]
    }

    public init(config: SP1ProverConfig = .sp1Default) {
        self.config = config
        if config.useGPU {
            self.hashEngine = try? Poseidon2BabyBearEngine()
        }
    }

    /// Run the SP1 FRI protocol on polynomial evaluations.
    ///
    /// - Parameters:
    ///   - evaluations: Polynomial evaluations over the LDE domain
    ///   - logN: Log2 of the evaluation domain size
    ///   - challenger: Fiat-Shamir transcript for challenge generation
    /// - Returns: Complete FRI proof
    public func fold(
        evaluations: [Bb],
        logN: Int,
        challenger: Plonky3Challenger
    ) throws -> SP1FRIProof {
        var currentEvals = evaluations
        var currentLogN = logN
        var rounds = [FRIRoundData]()

        // Derive initial query indices
        let numQueries = config.numQueries
        var queryIndices = [Int]()
        for _ in 0..<numQueries {
            let qi = Int(challenger.sample().v) % (evaluations.count / SP1ProverConfig.friFoldFactor)
            queryIndices.append(qi)
        }
        let originalQueryIndices = queryIndices

        // Fold-by-4: each round reduces domain by factor of 4 (2 bits of logN)
        while currentLogN > config.friMaxRemainderLogN {
            let n = 1 << currentLogN
            let quarter = n / SP1ProverConfig.friFoldFactor

            // Commit current evaluations via Poseidon2 Merkle tree
            let tree = BbPoseidon2MerkleTree.build(leaves: currentEvals)
            let commitment = tree.root
            challenger.observeSlice(commitment)

            // Squeeze folding challenge
            let beta = challenger.sample()

            // Build query openings for this round
            var queryOpenings = [(values: [Bb], paths: [[[Bb]]])]()
            for qi in queryIndices {
                let baseIdx = qi % quarter
                // Gather the 4 coset siblings
                var values = [Bb]()
                var paths = [[[Bb]]]()
                for k in 0..<SP1ProverConfig.friFoldFactor {
                    let idx = baseIdx + k * quarter
                    values.append(currentEvals[idx])
                    paths.append(tree.openingProof(index: idx))
                }
                queryOpenings.append((values: values, paths: paths))
            }

            // Retain Merkle layers for later verification
            rounds.append(FRIRoundData(
                commitment: commitment,
                merkleLayers: tree.layers,
                beta: beta,
                queryOpenings: queryOpenings
            ))

            // Fold-by-4: combine 4 evaluations into 1
            let omega = bbRootOfUnity(logN: currentLogN)
            let inv4 = bbInverse(Bb(v: 4))
            let beta2 = bbSqr(beta)
            let beta3 = bbMul(beta2, beta)
            let betaInv4 = bbMul(beta, inv4)
            let beta2Inv4 = bbMul(beta2, inv4)
            let beta3Inv4 = bbMul(beta3, inv4)

            // Precompute omega powers via chain multiply, then batch-invert omega^i, omega^2i, omega^3i
            var omegaPows = [Bb](repeating: Bb.one, count: quarter)
            for i in 1..<quarter { omegaPows[i] = bbMul(omegaPows[i - 1], omega) }

            var omegaDenoms = [Bb](repeating: Bb.zero, count: 3 * quarter)
            for i in 0..<quarter {
                omegaDenoms[3 * i] = omegaPows[i]
                omegaDenoms[3 * i + 1] = bbSqr(omegaPows[i])
                omegaDenoms[3 * i + 2] = bbMul(omegaDenoms[3 * i + 1], omegaPows[i])
            }
            var oPrefix = [Bb](repeating: Bb.one, count: 3 * quarter)
            for i in 1..<(3 * quarter) {
                oPrefix[i] = omegaDenoms[i - 1].v == 0 ? oPrefix[i - 1] : bbMul(oPrefix[i - 1], omegaDenoms[i - 1])
            }
            let oLast = omegaDenoms[3 * quarter - 1].v == 0 ? oPrefix[3 * quarter - 1] : bbMul(oPrefix[3 * quarter - 1], omegaDenoms[3 * quarter - 1])
            var oInv = bbInverse(oLast)
            var omegaInvs = [Bb](repeating: Bb.zero, count: 3 * quarter)
            for i in stride(from: 3 * quarter - 1, through: 0, by: -1) {
                if omegaDenoms[i].v != 0 {
                    omegaInvs[i] = bbMul(oInv, oPrefix[i])
                    oInv = bbMul(oInv, omegaDenoms[i])
                }
            }

            var folded = [Bb](repeating: Bb.zero, count: quarter)
            for i in 0..<quarter {
                let f0 = currentEvals[i]
                let f1 = currentEvals[i + quarter]
                let f2 = currentEvals[i + 2 * quarter]
                let f3 = currentEvals[i + 3 * quarter]

                let sum0 = bbAdd(bbAdd(f0, f1), bbAdd(f2, f3))
                let sum1 = bbAdd(bbSub(f0, f1), bbSub(f2, f3))
                let sum2 = bbSub(bbAdd(f0, f1), bbAdd(f2, f3))
                let sum3 = bbSub(bbSub(f0, f1), bbSub(f2, f3))

                var result = bbMul(sum0, inv4)
                result = bbAdd(result, bbMul(betaInv4, bbMul(sum1, omegaInvs[3 * i])))
                result = bbAdd(result, bbMul(beta2Inv4, bbMul(sum2, omegaInvs[3 * i + 1])))
                result = bbAdd(result, bbMul(beta3Inv4, bbMul(sum3, omegaInvs[3 * i + 2])))

                folded[i] = result
            }

            currentEvals = folded
            currentLogN -= SP1ProverConfig.friLogFoldFactor

            // Update query indices for folded domain
            if currentLogN > 0 {
                queryIndices = queryIndices.map { $0 % (1 << currentLogN) }
            }
        }

        // Final polynomial: convert remaining evaluations to coefficients
        let finalPoly: [Bb]
        if currentLogN > 0 {
            finalPoly = BabyBearNTTEngine.cpuINTT(currentEvals, logN: currentLogN)
        } else {
            finalPoly = currentEvals
        }

        return SP1FRIProof(
            rounds: rounds,
            finalPoly: finalPoly,
            queryIndices: originalQueryIndices
        )
    }
}

// MARK: - SP1 Constraint Evaluator

/// GPU-accelerated AIR constraint evaluator for SP1.
///
/// SP1 programs compile down to a set of AIR chips (CPU chip, memory chip,
/// ALU chip, etc.), each with its own trace and constraints. This evaluator
/// handles a single chip's constraint evaluation over its LDE domain.
///
/// The evaluator computes the quotient polynomial:
///   Q(x) = sum_i alpha^i * C_i(x) / Z_H(x)
/// where C_i are the constraint polynomials and Z_H is the vanishing polynomial.
///
/// Supports SP1's three trace sections:
///   - Preprocessed: fixed columns (selectors, wiring)
///   - Main: witness columns from program execution
///   - Permutation: interaction/lookup columns from challenges
public class SP1ConstraintEvaluator {
    private let config: SP1ProverConfig

    public init(config: SP1ProverConfig = .sp1Default) {
        self.config = config
    }

    /// Evaluate all AIR constraints and produce the quotient polynomial.
    ///
    /// - Parameters:
    ///   - air: The Plonky3-compatible AIR definition
    ///   - preprocessed: Preprocessed (fixed) trace columns, or nil
    ///   - main: Main execution trace columns
    ///   - permutation: Permutation/interaction columns, or nil
    ///   - challenges: Interaction challenges from the Fiat-Shamir transcript
    ///   - alpha: Random linear combination challenge
    /// - Returns: Quotient polynomial evaluations over the trace domain
    public func evaluateQuotient(
        air: any Plonky3AIR,
        preprocessed: [[Bb]]?,
        main: [[Bb]],
        permutation: [[Bb]]?,
        challenges: [Bb],
        alpha: Bb
    ) -> [Bb] {
        let evaluator = Plonky3ConstraintEvaluator(
            air: air,
            config: Plonky3AIRConfig(
                logBlowup: config.logBlowup,
                numQueries: config.numQueries,
                grindingBits: config.grindingBits,
                useGPU: config.useGPU
            )
        )
        return evaluator.evaluateQuotient(
            preprocessed: preprocessed,
            main: main,
            permutation: permutation,
            challenges: challenges,
            alpha: alpha
        )
    }

    /// Evaluate constraints over the full LDE domain for the quotient polynomial.
    ///
    /// This is the LDE-domain version: constraints are evaluated at every point
    /// in the extended domain, then divided by the vanishing polynomial.
    /// Used when the trace has already been LDE-extended.
    ///
    /// - Parameters:
    ///   - air: The Plonky3-compatible AIR definition
    ///   - traceLDEs: LDE columns [column][row] over the extended domain
    ///   - alpha: Random linear combination challenge
    ///   - logTrace: Log2 of the original trace length
    ///   - logLDE: Log2 of the LDE domain size
    /// - Returns: Quotient polynomial evaluations over the LDE domain
    public func evaluateQuotientOverLDE(
        air: any Plonky3AIR,
        traceLDEs: [[Bb]],
        alpha: Bb,
        logTrace: Int,
        logLDE: Int
    ) -> [Bb] {
        let traceLen = 1 << logTrace
        let ldeLen = 1 << logLDE
        let step = ldeLen / traceLen
        let cosetShift = bbCosetGenerator(logN: logLDE)
        let omega = bbRootOfUnity(logN: logLDE)

        // Precompute vanishing polynomial values at each LDE point, then batch-invert
        var zhVals = [Bb](repeating: Bb.zero, count: ldeLen)
        var cosetPow = Bb.one
        for i in 0..<ldeLen {
            let xToN = bbPow(bbMul(cosetShift, cosetPow), UInt32(traceLen))
            zhVals[i] = bbSub(xToN, Bb.one)
            cosetPow = bbMul(cosetPow, omega)
        }
        // Montgomery batch inversion
        var zhPrefix = [Bb](repeating: Bb.one, count: ldeLen)
        for i in 1..<ldeLen {
            zhPrefix[i] = zhVals[i - 1].v == 0 ? zhPrefix[i - 1] : bbMul(zhPrefix[i - 1], zhVals[i - 1])
        }
        let zhLast = zhVals[ldeLen - 1].v == 0 ? zhPrefix[ldeLen - 1] : bbMul(zhPrefix[ldeLen - 1], zhVals[ldeLen - 1])
        var zhInv = bbInverse(zhLast)
        var vanishingInv = [Bb](repeating: Bb.zero, count: ldeLen)
        for i in stride(from: ldeLen - 1, through: 0, by: -1) {
            if zhVals[i].v != 0 {
                vanishingInv[i] = bbMul(zhInv, zhPrefix[i])
                zhInv = bbMul(zhInv, zhVals[i])
            }
        }

        // Evaluate constraints at each LDE point
        let numCols = traceLDEs.count
        var quotientEvals = [Bb](repeating: Bb.zero, count: ldeLen)

        for i in 0..<ldeLen {
            let nextI = (i + step) % ldeLen

            let current = (0..<numCols).map { traceLDEs[$0][i] }
            let next = (0..<numCols).map { traceLDEs[$0][nextI] }

            let constraintEvals = air.evaluateConstraints(
                preprocessed: nil,
                main: (current: current, next: next),
                permutation: nil,
                challenges: []
            )

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

        return quotientEvals
    }
}

// MARK: - SP1 Prover Pipeline

/// Full SP1-compatible STARK prover pipeline using zkMetal GPU acceleration.
///
/// Orchestrates the complete proving flow:
///   1. Generate execution trace from the AIR
///   2. Commit trace via GPU-accelerated Poseidon2 Merkle (SP1TraceCommitter)
///   3. Evaluate constraints and compute quotient (SP1ConstraintEvaluator)
///   4. Commit quotient polynomial
///   5. FRI proximity test with fold-by-4 (SP1FRIFolder)
///   6. Generate query responses with Merkle opening proofs
///
/// Usage:
///   let pipeline = SP1ProverPipeline()
///   let proof = try pipeline.prove(air: myAIR)
public class SP1ProverPipeline {
    public static let version = Versions.babyBearSTARK

    public let config: SP1ProverConfig
    private let traceCommitter: SP1TraceCommitter
    private let friFolder: SP1FRIFolder
    private let constraintEvaluator: SP1ConstraintEvaluator

    public init(config: SP1ProverConfig = .sp1Default) {
        self.config = config
        self.traceCommitter = SP1TraceCommitter(config: config)
        self.friFolder = SP1FRIFolder(config: config)
        self.constraintEvaluator = SP1ConstraintEvaluator(config: config)
    }

    /// Prove that a Plonky3 AIR is satisfied, producing an SP1-compatible STARK proof.
    ///
    /// - Parameter air: A Plonky3-compatible AIR definition
    /// - Returns: A BabyBear STARK proof
    public func prove(air: any Plonky3AIR) throws -> BabyBearSTARKProof {
        let logTrace = air.logTraceLength
        let traceLen = air.traceLength
        let logLDE = logTrace + config.logBlowup
        let ldeLen = 1 << logLDE

        // Step 1: Generate execution trace
        let mainTrace = air.generateMainTrace()
        guard mainTrace.count == air.numMainColumns else {
            throw BabyBearSTARKError.invalidTrace(
                "Expected \(air.numMainColumns) main columns, got \(mainTrace.count)")
        }

        // Step 2: Commit trace via GPU Poseidon2 Merkle
        let traceData = try traceCommitter.commit(columns: mainTrace)

        // Step 3: Fiat-Shamir transcript
        var challenger = Plonky3Challenger()
        challenger.observeDigest(traceData.root)
        let alpha = challenger.sample()

        // Step 4: Evaluate constraints over LDE, compute quotient
        let quotientEvals = constraintEvaluator.evaluateQuotientOverLDE(
            air: air,
            traceLDEs: traceData.ldeColumns,
            alpha: alpha,
            logTrace: logTrace,
            logLDE: logLDE
        )

        // Step 5: Commit quotient polynomial
        let quotientTree = BbPoseidon2MerkleTree.build(leaves: quotientEvals)
        let compositionCommitment = quotientTree.root
        challenger.observeDigest(compositionCommitment)

        // Step 6: FRI proximity test (fold-by-4)
        let friProof = try friFolder.fold(
            evaluations: quotientEvals,
            logN: logLDE,
            challenger: challenger
        )

        // Step 7: Generate query responses
        challenger.observeDigest(compositionCommitment)
        let queryIndices = friProof.queryIndices

        var queryResponses = [BabyBearSTARKQueryResponse]()
        queryResponses.reserveCapacity(queryIndices.count)

        for qi in queryIndices {
            let boundedQI = qi % ldeLen

            // Open trace at this query index
            var traceValues = [Bb]()
            var traceOpenings = [BbMerkleOpeningProof]()
            for colIdx in 0..<air.numMainColumns {
                traceValues.append(traceData.ldeColumns[colIdx][boundedQI])
                let path = traceCommitter.openingProof(data: traceData, row: boundedQI)
                traceOpenings.append(BbMerkleOpeningProof(path: path, index: boundedQI))
            }

            // Open quotient at this query index
            let compValue = quotientEvals[boundedQI]
            let compPath = quotientTree.openingProof(index: boundedQI)
            let compOpening = BbMerkleOpeningProof(path: compPath, index: boundedQI)

            queryResponses.append(BabyBearSTARKQueryResponse(
                traceValues: traceValues,
                traceOpenings: traceOpenings,
                compositionValue: compValue,
                compositionOpening: compOpening,
                queryIndex: boundedQI
            ))
        }

        // Convert SP1 FRI proof to BabyBear FRI proof format
        let babybearFRIRounds = friProof.rounds.map { round in
            BabyBearFRIRound(
                commitment: round.commitment,
                queryOpenings: round.queryOpenings.map { opening in
                    (value: opening.values[0],
                     siblingValue: opening.values.count > 1 ? opening.values[1] : Bb.zero,
                     path: opening.paths[0])
                }
            )
        }

        let babybearFRIProof = BabyBearFRIProof(
            rounds: babybearFRIRounds,
            finalPoly: friProof.finalPoly,
            queryIndices: friProof.queryIndices
        )

        return BabyBearSTARKProof(
            traceCommitments: [traceData.root],
            compositionCommitment: compositionCommitment,
            friProof: babybearFRIProof,
            queryResponses: queryResponses,
            alpha: alpha,
            traceLength: traceLen,
            numColumns: air.numMainColumns,
            logBlowup: config.logBlowup
        )
    }
}
