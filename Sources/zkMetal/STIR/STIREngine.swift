// STIR Engine — Shift To Improve Rate proximity testing (Prover)
//
// STIR (Arnon, Chiesa, Fenzi, Yogev — eprint 2024/390) is an alternative
// to FRI for Reed-Solomon proximity testing with O(log^2 n) query
// complexity. Key idea: after each fold round, evaluate the folded polynomial
// on a *shifted* domain {alpha * omega^i}, decorrelating errors across rounds
// and improving the effective rate of the tested code.
//
// Protocol overview (each round):
//   1. Prover commits to current evaluations via Poseidon2 Merkle tree
//   2. Verifier sends folding challenge beta (via Fiat-Shamir)
//   3. Prover computes folded polynomial (degree halved)
//   4. Verifier sends shift challenge alpha (via Fiat-Shamir)
//   5. Prover shifts domain: iNTT -> coeff[j] *= alpha^j -> NTT
//   6. Repeat until polynomial is small; send final poly in the clear
//   7. Verifier sends query positions
//   8. Prover opens evaluations at query positions with Merkle proofs
//   9. Verifier checks fold+shift consistency at opened positions
//
// The domain shift step is what distinguishes STIR from FRI. Shifts decorrelate
// the evaluation domain between rounds, achieving better soundness per query:
//   FRI:  error ~ rho       per query  -> O(lambda * log n) queries
//   STIR: error ~ rho^1.5   per query  -> O(log^2 n) queries
//
// GPU acceleration: Merkle tree via Metal Poseidon2 for large domains;
// NTT/iNTT via Metal for domain shifts; C CIOS for polynomial folding.

import Foundation
import NeonFieldOps

// MARK: - STIR Prover

public class STIRProver {
    public static let version = Versions.stir

    /// Number of queries per round
    public let numQueries: Int
    /// Folding factor (must be power of 2)
    public let reductionFactor: Int
    /// log2(reductionFactor)
    public let logReduction: Int

    private let merkleEngine: Poseidon2MerkleEngine

    /// Optional NTT engine for GPU-accelerated domain shifts.
    /// If nil, CPU-only shifts are used.
    private var nttEngine: NTTEngine?

    /// CPU Merkle threshold: use CPU Poseidon2 for small trees to avoid
    /// GPU command buffer overhead (~5-9ms per dispatch).
    /// GCD dispatch_apply gives near-zero threading overhead, so CPU is
    /// competitive up to ~4096 leaves on Apple Silicon.
    private static let cpuMerkleThreshold = 4096

    /// Initialize STIR prover.
    /// - Parameters:
    ///   - numQueries: queries per round (default 4; more = higher security)
    ///   - reductionFactor: degree reduction per round (default 4; must be power of 2)
    ///   - useGPU: whether to use Metal GPU for NTT domain shifts (default false)
    public init(numQueries: Int = 4, reductionFactor: Int = 4, useGPU: Bool = false) throws {
        precondition(reductionFactor >= 2 && (reductionFactor & (reductionFactor - 1)) == 0,
                     "reductionFactor must be a power of 2")
        self.numQueries = numQueries
        self.reductionFactor = reductionFactor
        self.logReduction = Int(log2(Double(reductionFactor)))
        self.merkleEngine = try Poseidon2MerkleEngine()
        if useGPU {
            self.nttEngine = try NTTEngine()
        }
    }

    // MARK: - Commit

    /// Commit to polynomial evaluations via Poseidon2 Merkle tree.
    /// Uses CPU path for small trees, GPU for large ones.
    public func commit(evaluations: [Fr]) throws -> STIRRoundCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")

        let tree: [Fr]
        if n <= STIRProver.cpuMerkleThreshold {
            let treeSize = 2 * n - 1
            var treeArr = [Fr](repeating: Fr.zero, count: treeSize)
            evaluations.withUnsafeBytes { evPtr in
                treeArr.withUnsafeMutableBytes { treePtr in
                    poseidon2_merkle_tree_cpu(
                        evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        treePtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
            tree = treeArr
        } else {
            tree = try merkleEngine.buildTree(evaluations)
        }

        let root = tree[2 * n - 2]
        return STIRRoundCommitment(root: root, tree: tree, evaluations: evaluations)
    }

    // MARK: - Domain Shift

    /// Shift evaluation domain by alpha: f evaluated on {alpha * omega^i}.
    /// Algorithm: iNTT(evals) -> coeff[j] *= alpha^j -> NTT.
    ///
    /// This is the key operation that distinguishes STIR from FRI. By shifting
    /// the domain between rounds, the proximity gap improves multiplicatively.
    public func domainShift(evals: [Fr], alpha: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // Get coefficients via iNTT
        var coeffs: [Fr]
        if let ntt = nttEngine {
            coeffs = try ntt.intt(evals)
        } else {
            let omega = frRootOfUnity(logN: logN)
            let omegaInv = frInverse(omega)
            coeffs = STIRProver.cpuINTT(evals: evals, omegaInv: omegaInv, n: n)
        }

        // Multiply coefficient j by alpha^j
        var alphaPow = Fr.one
        for j in 0..<n {
            coeffs[j] = frMul(coeffs[j], alphaPow)
            alphaPow = frMul(alphaPow, alpha)
        }

        // NTT back to evaluation domain
        if let ntt = nttEngine {
            return try ntt.ntt(coeffs)
        } else {
            let omega = frRootOfUnity(logN: logN)
            return STIRProver.cpuNTT(coeffs: coeffs, omega: omega, n: n)
        }
    }

    /// CPU domain shift for verification at small sizes.
    public static func cpuDomainShift(evals: [Fr], alpha: Fr) -> [Fr] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let omega = frRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        var coeffs = cpuINTT(evals: evals, omegaInv: omegaInv, n: n)

        var alphaPow = Fr.one
        for j in 0..<n {
            coeffs[j] = frMul(coeffs[j], alphaPow)
            alphaPow = frMul(alphaPow, alpha)
        }

        return cpuNTT(coeffs: coeffs, omega: omega, n: n)
    }

    // MARK: - Prove

    /// Generate a STIR proof for polynomial evaluations.
    ///
    /// The proof demonstrates that the committed polynomial is close to one of
    /// degree < d, using iterative folding with domain shifts. Each round:
    ///   1. Commit evaluations via Poseidon2 Merkle tree
    ///   2. Fold by reductionFactor using verifier challenge beta
    ///   3. Shift domain by verifier challenge alpha (distinguishes STIR from FRI)
    ///   4. Repeat until polynomial is small
    ///
    /// - Parameters:
    ///   - evaluations: polynomial evaluations on the domain (length must be power of 2)
    ///   - transcript: optional Fiat-Shamir transcript (created internally if nil)
    /// - Returns: STIR proof with Merkle commitments, query openings, and shift data
    public func prove(evaluations: [Fr], transcript: Transcript? = nil) throws -> STIRProofData {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // Fold until <= 16 elements remain
        let rounds = max(1, (logN - 4) / logReduction)

        let ts = transcript ?? Transcript(label: "stir-v2")

        // Phase 1: Build all layers (commit -> derive beta+alpha -> fold -> shift)
        var layers: [STIRRoundCommitment] = []
        var betas: [Fr] = []
        var alphas: [Fr] = []
        var currentEvals = evaluations

        for round in 0..<rounds {
            let currentN = currentEvals.count
            if currentN <= reductionFactor { break }

            let commitment = try commit(evaluations: currentEvals)
            layers.append(commitment)

            // Transcript: absorb root, squeeze folding challenge beta
            ts.absorb(commitment.root)
            ts.absorbLabel("stir-fold-r\(round)")
            let beta = ts.squeeze()
            betas.append(beta)

            // Transcript: squeeze shift challenge alpha
            ts.absorbLabel("stir-shift-r\(round)")
            let alpha = ts.squeeze()
            alphas.append(alpha)

            // Fold polynomial using C CIOS arithmetic (Horner's method)
            let folded = STIRProver.cpuFold(
                evals: currentEvals, challenge: beta,
                reductionFactor: reductionFactor)

            // Apply domain shift (the key STIR operation)
            // Skip shift on last round (final poly is sent in the clear)
            if round < rounds - 1 && folded.count > reductionFactor {
                currentEvals = try domainShift(evals: folded, alpha: alpha)
            } else {
                currentEvals = folded
            }
        }

        let finalPoly = currentEvals
        let actualRounds = betas.count

        // Transcript: absorb final polynomial
        ts.absorbLabel("stir-final")
        for v in finalPoly { ts.absorb(v) }

        // Phase 2: Query phase — open positions via Fiat-Shamir
        var layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]] = []

        for round in 0..<actualRounds {
            let layer = layers[round]
            let layerN = layer.evaluations.count
            let foldedN = layerN / reductionFactor

            // Derive query positions (in folded domain)
            let effectiveQ = min(numQueries, foldedN)
            var queryIndices = [UInt32]()
            var used = Set<UInt32>()
            for _ in 0..<effectiveQ {
                let c = ts.squeeze()
                var idx = UInt32(frToInt(c)[0] % UInt64(foldedN))
                while used.contains(idx) {
                    idx = (idx + 1) % UInt32(foldedN)
                }
                queryIndices.append(idx)
                used.insert(idx)
            }

            // Open reductionFactor positions per query in the current layer
            let layerTree = layer.tree
            let layerEvals = layer.evaluations
            var roundOpenings: [(index: UInt32, values: [Fr], merklePaths: [[Fr]])] = []
            roundOpenings.reserveCapacity(effectiveQ)

            for qi in 0..<effectiveQ {
                let foldedIdx = Int(queryIndices[qi])
                var values = [Fr]()
                values.reserveCapacity(reductionFactor)
                var paths = [[Fr]]()
                paths.reserveCapacity(reductionFactor)
                for k in 0..<reductionFactor {
                    let origIdx = foldedIdx * reductionFactor + k
                    values.append(layerEvals[origIdx])
                    paths.append(extractMerklePath(tree: layerTree,
                                                    leafCount: layerN,
                                                    index: origIdx))
                }
                roundOpenings.append((index: queryIndices[qi], values: values, merklePaths: paths))
            }
            layerOpenings.append(roundOpenings)
        }

        return STIRProofData(
            roots: layers.map { $0.root },
            betas: betas,
            alphas: alphas,
            layerOpenings: layerOpenings,
            finalPoly: finalPoly,
            numRounds: actualRounds
        )
    }

    // MARK: - Verify (convenience, delegates to STIRVerifier)

    /// Succinct verify without original evaluations.
    public func verify(proof: STIRProofData) -> Bool {
        let verifier = STIRVerifier(numQueries: numQueries, reductionFactor: reductionFactor)
        return verifier.verify(proof: proof)
    }

    /// Succinct verify with known domain size.
    public func verify(proof: STIRProofData, evaluations: [Fr]) -> Bool {
        let verifier = STIRVerifier(numQueries: numQueries, reductionFactor: reductionFactor)
        return verifier.verify(proof: proof, domainSize: evaluations.count)
    }

    /// Full verify with original evaluations (checks every value including shift consistency).
    public func verifyFull(proof: STIRProofData, evaluations: [Fr]) -> Bool {
        let verifier = STIRVerifier(numQueries: numQueries, reductionFactor: reductionFactor)
        return verifier.verifyFull(proof: proof, evaluations: evaluations)
    }

    // MARK: - Soundness Analysis

    /// Compute queries needed for a given security level.
    /// STIR achieves better soundness per query than FRI:
    ///   FRI:  error ~ rho       per query
    ///   STIR: error ~ rho^1.5   per query (from domain shifts)
    public static func queriesNeeded(securityBits: Int, rate: Double, useSTIR: Bool) -> Int {
        let rho = rate
        if useSTIR {
            let errorPerQuery = pow(rho, 1.5)
            let queriesF = Double(securityBits) * log(2.0) / (-log(errorPerQuery))
            return max(1, Int(ceil(queriesF)))
        } else {
            let errorPerQuery = rho
            let queriesF = Double(securityBits) * log(2.0) / (-log(errorPerQuery))
            return max(1, Int(ceil(queriesF)))
        }
    }

    /// Estimate proof size for given parameters.
    public static func estimateProofSize(logN: Int, numQueries: Int, reductionFactor: Int) -> Int {
        let frSize = MemoryLayout<Fr>.stride
        let logReduction = Int(log2(Double(reductionFactor)))
        let numRounds = max(1, (logN - 4) / logReduction)
        let perQueryPerRound = reductionFactor * frSize + reductionFactor * logN * frSize
        let commitSize = numRounds * frSize  // roots
        let challengeSize = numRounds * 2 * frSize  // betas + alphas
        let finalSize = max(1, 1 << max(0, logN - numRounds * logReduction)) * frSize
        return commitSize + challengeSize + numQueries * numRounds * perQueryPerRound + finalSize
    }

    // MARK: - Merkle Helpers

    func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
        var path = [Fr]()
        var idx = index
        var levelStart = 0
        var levelSize = leafCount

        while levelSize > 1 {
            let siblingIdx = idx ^ 1
            if levelStart + siblingIdx < tree.count {
                path.append(tree[levelStart + siblingIdx])
            }
            idx /= 2
            levelStart += levelSize
            levelSize /= 2
        }
        return path
    }

    // MARK: - CPU Helpers

    /// Fold polynomial evaluations by reductionFactor using C CIOS Montgomery arithmetic.
    /// result[j] = sum_{k=0}^{r-1} beta^k * evals[j*r + k]  (computed via Horner's method)
    public static func cpuFold(evals: [Fr], challenge: Fr, reductionFactor: Int) -> [Fr] {
        let n = evals.count
        let newN = n / reductionFactor
        var result = [Fr](repeating: Fr.zero, count: newN)
        evals.withUnsafeBytes { evalsPtr in
            result.withUnsafeMutableBytes { resPtr in
                var betaLimbs = challenge.to64()
                betaLimbs.withUnsafeBufferPointer { betaPtr in
                    bn254_fr_whir_fold(
                        evalsPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        betaPtr.baseAddress!,
                        Int32(reductionFactor),
                        resPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    /// Simple CPU NTT (Cooley-Tukey) for correctness testing.
    static func cpuNTT(coeffs: [Fr], omega: Fr, n: Int) -> [Fr] {
        if n == 1 { return coeffs }
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            var val = Fr.zero
            var omegaPow = Fr.one
            let omegaI = frPow(omega, UInt64(i))
            for j in 0..<n {
                val = frAdd(val, frMul(coeffs[j], omegaPow))
                omegaPow = frMul(omegaPow, omegaI)
            }
            result[i] = val
        }
        return result
    }

    /// Simple CPU iNTT for correctness testing.
    static func cpuINTT(evals: [Fr], omegaInv: Fr, n: Int) -> [Fr] {
        let nInv = frInverse(frFromInt(UInt64(n)))
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            var val = Fr.zero
            var omegaPow = Fr.one
            let omegaInvI = frPow(omegaInv, UInt64(i))
            for j in 0..<n {
                val = frAdd(val, frMul(evals[j], omegaPow))
                omegaPow = frMul(omegaPow, omegaInvI)
            }
            result[i] = frMul(val, nInv)
        }
        return result
    }
}
