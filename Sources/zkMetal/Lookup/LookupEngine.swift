// LogUp Lookup Argument Engine
// Proves that every element in a lookup vector exists in a table using the LogUp protocol.
//
// Protocol (LogUp / logarithmic derivative):
//   Given table T[0..N-1] and lookups f[0..m-1] where each f[i] ∈ T:
//   1. Prover computes multiplicities mult[j] = #{i : f[i] = T[j]}
//   2. Verifier sends random challenge β
//   3. Prove: Σᵢ 1/(β + f[i]) = Σⱼ mult[j]/(β + T[j])
//   4. Both sides proven via sumcheck on multilinear extensions
//
// References: LogUp (Haböck 2022), Lasso (Setty et al. 2023)

import Foundation
import Metal
import NeonFieldOps

/// Proof that all lookups exist in the table
public struct LookupProof {
    /// Multiplicities: mult[j] = number of times T[j] appears in f
    public let multiplicities: [Fr]
    /// The random challenge β used in the protocol
    public let beta: Fr
    /// Sumcheck rounds for the lookup side: Σ 1/(β + f[i])
    public let lookupSumcheckRounds: [(Fr, Fr, Fr)]
    /// Sumcheck rounds for the table side: Σ mult[j]/(β + T[j])
    public let tableSumcheckRounds: [(Fr, Fr, Fr)]
    /// Claimed sum value S = Σ 1/(β + f[i])
    public let claimedSum: Fr
    /// Final evaluation of lookup inverse polynomial at random point
    public let lookupFinalEval: Fr
    /// Final evaluation of table inverse polynomial at random point
    public let tableFinalEval: Fr

    public init(multiplicities: [Fr], beta: Fr,
                lookupSumcheckRounds: [(Fr, Fr, Fr)],
                tableSumcheckRounds: [(Fr, Fr, Fr)],
                claimedSum: Fr, lookupFinalEval: Fr, tableFinalEval: Fr) {
        self.multiplicities = multiplicities
        self.beta = beta
        self.lookupSumcheckRounds = lookupSumcheckRounds
        self.tableSumcheckRounds = tableSumcheckRounds
        self.claimedSum = claimedSum
        self.lookupFinalEval = lookupFinalEval
        self.tableFinalEval = tableFinalEval
    }
}

public class LookupEngine {
    public static let version = Versions.lookup
    public let polyEngine: PolyEngine
    public let sumcheckEngine: SumcheckEngine

    /// Enable timing instrumentation for profiling hot paths
    public var profileLogUp = false

    // Grow-only cached GPU buffers to avoid per-call allocation
    private var cachedBufA: MTLBuffer?     // for betaPlusF or betaPlusT input
    private var cachedBufB: MTLBuffer?     // for batchInverse output
    private var cachedBufC: MTLBuffer?     // for hadamard second input (mult)
    private var cachedBufD: MTLBuffer?     // for hadamard output (ht)
    private var cachedBufCapacity: Int = 0 // max element count currently allocated

    public init() throws {
        self.polyEngine = try PolyEngine()
        self.sumcheckEngine = try SumcheckEngine()
    }

    /// Ensure cached buffers can hold at least `count` Fr elements.
    /// Grow-only: only reallocates when the requested size exceeds current capacity.
    private func ensureCachedBuffers(count: Int) {
        guard count > cachedBufCapacity else { return }
        let bytes = count * MemoryLayout<Fr>.stride
        let device = polyEngine.device
        cachedBufA = device.makeBuffer(length: bytes, options: .storageModeShared)
        cachedBufB = device.makeBuffer(length: bytes, options: .storageModeShared)
        cachedBufC = device.makeBuffer(length: bytes, options: .storageModeShared)
        cachedBufD = device.makeBuffer(length: bytes, options: .storageModeShared)
        cachedBufCapacity = count
    }

    /// Compute multiplicities: for each table entry T[j], count how many times it appears in f.
    /// Returns array of length table.count with multiplicity values as Fr elements.
    ///
    /// Uses sorted-array binary search instead of Dictionary for cache-friendly access.
    public static func computeMultiplicities(table: [Fr], lookups: [Fr]) -> [Fr] {
        // Build sorted array of (key, originalIndex) for binary search
        let N = table.count
        var sortedEntries = [(limbs: [UInt64], idx: Int)](repeating: ([], 0), count: N)
        for j in 0..<N {
            sortedEntries[j] = (frToInt(table[j]), j)
        }
        sortedEntries.sort { a, b in
            for k in stride(from: a.limbs.count - 1, through: 0, by: -1) {
                if a.limbs[k] != b.limbs[k] { return a.limbs[k] < b.limbs[k] }
            }
            return false
        }

        // Extract sorted keys for cache-friendly binary search
        let sortedKeys = sortedEntries.map { $0.limbs }
        let sortedIndices = sortedEntries.map { $0.idx }

        var mult = [UInt64](repeating: 0, count: N)
        for i in 0..<lookups.count {
            let key = frToInt(lookups[i])
            // Binary search in sorted keys
            var lo = 0, hi = N - 1
            var found = false
            while lo <= hi {
                let mid = (lo + hi) >> 1
                let cmp = compareLimbs(key, sortedKeys[mid])
                if cmp == 0 {
                    mult[sortedIndices[mid]] += 1
                    found = true
                    break
                } else if cmp < 0 {
                    hi = mid - 1
                } else {
                    lo = mid + 1
                }
            }
            precondition(found, "Lookup value not in table at index \(i)")
        }

        return mult.map { frFromInt($0) }
    }

    /// Compare two limb arrays lexicographically (big-endian limb order).
    /// Returns -1, 0, or 1.
    private static func compareLimbs(_ a: [UInt64], _ b: [UInt64]) -> Int {
        for k in stride(from: a.count - 1, through: 0, by: -1) {
            if a[k] < b[k] { return -1 }
            if a[k] > b[k] { return 1 }
        }
        return 0
    }

    /// Create a LogUp lookup proof.
    /// Proves that every element in `lookups` exists in `table`.
    /// The `beta` challenge would normally come from Fiat-Shamir; here it's passed explicitly.
    ///
    /// Optimizations applied:
    /// - Fused GPU command buffer: batchInverse(f) + batchInverse(T) + hadamard in one submit
    /// - Grow-only buffer caching: avoids per-call GPU allocation
    /// - Sorted-array multiplicities: cache-friendly binary search
    public func prove(table: [Fr], lookups: [Fr], beta: Fr) throws -> LookupProof {
        let m = lookups.count
        let N = table.count
        precondition(m > 0 && (m & (m - 1)) == 0, "Lookup count must be power of 2")
        precondition(N > 0 && (N & (N - 1)) == 0, "Table size must be power of 2")

        let _tTotal = profileLogUp ? CFAbsoluteTimeGetCurrent() : 0
        var _tPhase = profileLogUp ? CFAbsoluteTimeGetCurrent() : 0

        // Step 1: Compute multiplicities
        let mult = LookupEngine.computeMultiplicities(table: table, lookups: lookups)

        if profileLogUp { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] multiplicities: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 2+3: Compute h_f and h_t via fused GPU command buffer
        // h_f[i] = 1/(β + f[i]),  h_t[j] = mult[j]/(β + T[j])
        let (hf, ht) = try fusedInverseAndHadamard(
            lookups: lookups, table: table, mult: mult, beta: beta)

        if profileLogUp { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] fused GPU (batchInv x2 + hadamard): %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 4: Compute the claimed sum S = Σ h_f[i]
        var sum = Fr.zero
        hf.withUnsafeBytes { buf in
            withUnsafeMutableBytes(of: &sum) { r in
                bn254_fr_vector_sum(buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    Int32(m), r.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Verify table side matches (sanity check)
        var tableSum = Fr.zero
        ht.withUnsafeBytes { buf in
            withUnsafeMutableBytes(of: &tableSum) { r in
                bn254_fr_vector_sum(buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    Int32(N), r.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }
        precondition(frEqual(sum, tableSum), "LogUp sum mismatch — lookup values not all in table")

        if profileLogUp { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] sum computation: %.2f ms\n", (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 5: Run sumcheck on h_f (lookup side)
        var transcript = [UInt8]()
        appendFr(&transcript, beta)
        appendFr(&transcript, sum)

        let logM = Int(log2(Double(m)))
        let (lookupRounds, lookupFinalEval, lookupChallenges) = try runSumcheck(
            evals: hf, numVars: logM, transcript: &transcript)

        if profileLogUp { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] sumcheck lookup (logM=%d): %.2f ms\n", logM, (_t - _tPhase) * 1000), stderr); _tPhase = _t }

        // Step 6: Run sumcheck on h_t (table side)
        let logN = Int(log2(Double(N)))
        let (tableRounds, tableFinalEval, tableChallenges) = try runSumcheck(
            evals: ht, numVars: logN, transcript: &transcript)
        _ = tableChallenges  // used implicitly via transcript

        if profileLogUp { let _t = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [logup] sumcheck table (logN=%d): %.2f ms\n", logN, (_t - _tPhase) * 1000), stderr)
            fputs(String(format: "  [logup] TOTAL prove: %.2f ms\n", (_t - _tTotal) * 1000), stderr)
        }

        return LookupProof(
            multiplicities: mult,
            beta: beta,
            lookupSumcheckRounds: lookupRounds,
            tableSumcheckRounds: tableRounds,
            claimedSum: sum,
            lookupFinalEval: lookupFinalEval,
            tableFinalEval: tableFinalEval
        )
    }

    /// Fused GPU dispatch: batchInverse(lookups) + batchInverse(table) + hadamard(mult, invT)
    /// Uses a single command buffer and cached grow-only buffers.
    private func fusedInverseAndHadamard(
        lookups: [Fr], table: [Fr], mult: [Fr], beta: Fr
    ) throws -> (hf: [Fr], ht: [Fr]) {
        let m = lookups.count
        let N = table.count
        let maxN = max(m, N)
        let stride = MemoryLayout<Fr>.stride
        let device = polyEngine.device

        // Ensure cached buffers are large enough
        ensureCachedBuffers(count: maxN)

        // Prepare betaPlusF on CPU, write directly into cached buffer
        let bufBetaPlusF = cachedBufA!
        let bufHf = cachedBufB!
        do {
            let dst = bufBetaPlusF.contents().bindMemory(to: Fr.self, capacity: m)
            for i in 0..<m {
                dst[i] = frAdd(beta, lookups[i])
            }
        }

        // We need separate buffers for table side since m and N may differ
        // Use temporary buffers for table side, reusing cached ones when possible
        let bufBetaPlusT: MTLBuffer
        let bufInvT: MTLBuffer
        let bufMult: MTLBuffer
        let bufHt: MTLBuffer
        if N <= cachedBufCapacity && m <= cachedBufCapacity {
            // Reuse cachedBufC for mult input, cachedBufD for ht output
            bufBetaPlusT = device.makeBuffer(length: N * stride, options: .storageModeShared)!
            bufInvT = device.makeBuffer(length: N * stride, options: .storageModeShared)!
            bufMult = cachedBufC!
            bufHt = cachedBufD!
        } else {
            bufBetaPlusT = device.makeBuffer(length: N * stride, options: .storageModeShared)!
            bufInvT = device.makeBuffer(length: N * stride, options: .storageModeShared)!
            bufMult = device.makeBuffer(length: N * stride, options: .storageModeShared)!
            bufHt = device.makeBuffer(length: N * stride, options: .storageModeShared)!
        }

        // Prepare betaPlusT and mult into GPU buffers
        do {
            let dst = bufBetaPlusT.contents().bindMemory(to: Fr.self, capacity: N)
            for i in 0..<N {
                dst[i] = frAdd(beta, table[i])
            }
        }
        mult.withUnsafeBytes { src in
            memcpy(bufMult.contents(), src.baseAddress!, N * stride)
        }

        // Single command buffer for all GPU work
        guard let cmdBuf = polyEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let chunkSize = 512
        let biTG = min(64, Int(polyEngine.batchInverseFunction.maxTotalThreadsPerThreadgroup))

        // Encode batchInverse for lookup side
        do {
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(polyEngine.batchInverseFunction)
            enc.setBuffer(bufBetaPlusF, offset: 0, index: 0)
            enc.setBuffer(bufHf, offset: 0, index: 1)
            var nVal = UInt32(m)
            enc.setBytes(&nVal, length: 4, index: 2)
            let numGroups = (m + chunkSize - 1) / chunkSize
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: biTG, height: 1, depth: 1))
            enc.endEncoding()
        }

        // Encode batchInverse for table side
        do {
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(polyEngine.batchInverseFunction)
            enc.setBuffer(bufBetaPlusT, offset: 0, index: 0)
            enc.setBuffer(bufInvT, offset: 0, index: 1)
            var nVal = UInt32(N)
            enc.setBytes(&nVal, length: 4, index: 2)
            let numGroups = (N + chunkSize - 1) / chunkSize
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: biTG, height: 1, depth: 1))
            enc.endEncoding()
        }

        // Encode hadamard: ht = mult * invT
        do {
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(polyEngine.hadamardFunction)
            enc.setBuffer(bufMult, offset: 0, index: 0)
            enc.setBuffer(bufInvT, offset: 0, index: 1)
            enc.setBuffer(bufHt, offset: 0, index: 2)
            var nVal = UInt32(N)
            enc.setBytes(&nVal, length: 4, index: 3)
            let tg = min(256, Int(polyEngine.hadamardFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Read results
        let hf = polyEngine.readBuffer(bufHf, count: m)
        let ht = polyEngine.readBuffer(bufHt, count: N)
        return (hf, ht)
    }

    /// Verify a LogUp lookup proof.
    /// Requires the original table and lookups (or their commitments in a real system).
    public func verify(proof: LookupProof, table: [Fr], lookups: [Fr]) throws -> Bool {
        let m = lookups.count
        let N = table.count
        let logM = Int(log2(Double(m)))
        let logN = Int(log2(Double(N)))

        guard proof.lookupSumcheckRounds.count == logM else { return false }
        guard proof.tableSumcheckRounds.count == logN else { return false }
        guard proof.multiplicities.count == N else { return false }

        // Reconstruct transcript and challenges (must match prover's derivation)
        var transcript = [UInt8]()
        appendFr(&transcript, proof.beta)
        appendFr(&transcript, proof.claimedSum)

        // Verify lookup-side sumcheck and reconstruct challenges
        let lookupChallenges: [Fr]
        let useGPULookup = m >= 256
        if useGPULookup {
            // GPU path: challenges derived before seeing round polys
            var lc = [Fr]()
            for _ in 0..<logM {
                let c = deriveChallenge(transcript)
                lc.append(c)
                appendFr(&transcript, c)
            }
            lookupChallenges = lc
        } else {
            // CPU path: challenges derived after each round poly
            var lc = [Fr]()
            for k in 0..<logM {
                let (s0, s1, s2) = proof.lookupSumcheckRounds[k]
                appendFr(&transcript, s0)
                appendFr(&transcript, s1)
                appendFr(&transcript, s2)
                let c = deriveChallenge(transcript)
                lc.append(c)
                appendFr(&transcript, c)
            }
            lookupChallenges = lc
        }

        // Check round 0: S(0) + S(1) = claimed sum
        let (s0_0, s1_0, _) = proof.lookupSumcheckRounds[0]
        if !frEqual(frAdd(s0_0, s1_0), proof.claimedSum) { return false }

        // Check subsequent rounds
        for k in 1..<logM {
            let (s0, s1, _) = proof.lookupSumcheckRounds[k]
            let prevEval = evaluateQuadratic(proof.lookupSumcheckRounds[k-1], at: lookupChallenges[k-1])
            if !frEqual(frAdd(s0, s1), prevEval) { return false }
        }

        // Check final eval
        let lastLookupEval = evaluateQuadratic(
            proof.lookupSumcheckRounds[logM - 1], at: lookupChallenges[logM - 1])
        if !frEqual(lastLookupEval, proof.lookupFinalEval) { return false }

        // Verify table-side sumcheck
        let tableChallenges: [Fr]
        let useGPUTable = N >= 256
        if useGPUTable {
            var tc = [Fr]()
            for _ in 0..<logN {
                let c = deriveChallenge(transcript)
                tc.append(c)
                appendFr(&transcript, c)
            }
            tableChallenges = tc
        } else {
            var tc = [Fr]()
            for k in 0..<logN {
                let (s0, s1, s2) = proof.tableSumcheckRounds[k]
                appendFr(&transcript, s0)
                appendFr(&transcript, s1)
                appendFr(&transcript, s2)
                let c = deriveChallenge(transcript)
                tc.append(c)
                appendFr(&transcript, c)
            }
            tableChallenges = tc
        }

        let (t0_0, t1_0, _) = proof.tableSumcheckRounds[0]
        if !frEqual(frAdd(t0_0, t1_0), proof.claimedSum) { return false }

        for k in 1..<logN {
            let (s0, s1, _) = proof.tableSumcheckRounds[k]
            let prevEval = evaluateQuadratic(proof.tableSumcheckRounds[k-1], at: tableChallenges[k-1])
            if !frEqual(frAdd(s0, s1), prevEval) { return false }
        }

        let lastTableEval = evaluateQuadratic(
            proof.tableSumcheckRounds[logN - 1], at: tableChallenges[logN - 1])
        if !frEqual(lastTableEval, proof.tableFinalEval) { return false }

        // Verify final evaluations against the actual polynomials
        // (In a real system, these would be polynomial commitment openings)
        let hfEvals = try computeHfEvals(lookups: lookups, beta: proof.beta)
        let hfAtR = evaluateMLE(hfEvals, at: lookupChallenges)
        if !frEqual(hfAtR, proof.lookupFinalEval) { return false }

        let htEvals = try computeHtEvals(table: table, multiplicities: proof.multiplicities, beta: proof.beta)
        let htAtR = evaluateMLE(htEvals, at: tableChallenges)
        if !frEqual(htAtR, proof.tableFinalEval) { return false }

        return true
    }

    // MARK: - Sumcheck execution

    /// Run a sumcheck protocol with round-by-round Fiat-Shamir challenge derivation.
    /// Uses GPU for sizes ≥ 256, CPU for smaller.
    /// Returns: (round polynomials, final evaluation, challenges used)
    private func runSumcheck(evals: [Fr], numVars: Int,
                             transcript: inout [UInt8]) throws
        -> (rounds: [(Fr, Fr, Fr)], finalEval: Fr, challenges: [Fr])
    {
        let useGPU = evals.count >= 256
        var rounds = [(Fr, Fr, Fr)]()
        var challenges = [Fr]()
        rounds.reserveCapacity(numVars)
        challenges.reserveCapacity(numVars)

        if useGPU {
            // Generate all challenges first (Fiat-Shamir from transcript seed)
            for _ in 0..<numVars {
                let c = deriveChallenge(transcript)
                challenges.append(c)
                appendFr(&transcript, c)
            }
            let (gpuRounds, finalEval) = try sumcheckEngine.fullSumcheck(
                evals: evals, challenges: challenges)
            return (gpuRounds, finalEval, challenges)
        } else {
            // CPU round-by-round with proper Fiat-Shamir
            var current = evals
            for _ in 0..<numVars {
                let roundPoly = SumcheckEngine.cpuRoundPoly(evals: current)
                rounds.append(roundPoly)

                // Derive challenge from transcript including this round's polynomial
                appendFr(&transcript, roundPoly.0)
                appendFr(&transcript, roundPoly.1)
                appendFr(&transcript, roundPoly.2)
                let challenge = deriveChallenge(transcript)
                challenges.append(challenge)
                appendFr(&transcript, challenge)

                current = SumcheckEngine.cpuReduce(evals: current, challenge: challenge)
            }
            precondition(current.count == 1)
            return (rounds, current[0], challenges)
        }
    }

    // MARK: - Helpers

    /// Compute h_f[i] = 1/(β + f[i])
    private func computeHfEvals(lookups: [Fr], beta: Fr) throws -> [Fr] {
        var betaPlusF = [Fr](repeating: Fr.zero, count: lookups.count)
        for i in 0..<lookups.count {
            betaPlusF[i] = frAdd(beta, lookups[i])
        }
        return try polyEngine.batchInverse(betaPlusF)
    }

    /// Compute h_t[j] = mult[j]/(β + T[j])
    private func computeHtEvals(table: [Fr], multiplicities: [Fr], beta: Fr) throws -> [Fr] {
        var betaPlusT = [Fr](repeating: Fr.zero, count: table.count)
        for i in 0..<table.count {
            betaPlusT[i] = frAdd(beta, table[i])
        }
        let inv = try polyEngine.batchInverse(betaPlusT)
        return try polyEngine.hadamard(multiplicities, inv)
    }

    /// Evaluate a degree-2 polynomial given by (S(0), S(1), S(2)) at point x.
    /// Uses Lagrange interpolation over {0, 1, 2}.
    private func evaluateQuadratic(_ triple: (Fr, Fr, Fr), at x: Fr) -> Fr {
        let (s0, s1, s2) = triple
        // L_0(x) = (x-1)(x-2)/2, L_1(x) = x(x-2)/(-1), L_2(x) = x(x-1)/2
        let one = Fr.one
        let two = frAdd(one, one)
        let xm1 = frSub(x, one)
        let xm2 = frSub(x, two)
        let inv2 = frInverse(two)
        let negOne = frSub(Fr.zero, one)

        let l0 = frMul(frMul(xm1, xm2), inv2)
        let l1 = frMul(frMul(x, xm2), negOne)
        let l2 = frMul(frMul(x, xm1), inv2)

        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }

    /// Evaluate multilinear extension at a point using repeated folding.
    /// evals: evaluations over {0,1}^k, point: [r_0, ..., r_{k-1}]
    private func evaluateMLE(_ evals: [Fr], at point: [Fr]) -> Fr {
        var current = evals
        for r in point {
            let half = current.count / 2
            var next = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                // f(r) = (1-r)*f(0) + r*f(1)
                let oneMinusR = frSub(Fr.one, r)
                next[i] = frAdd(frMul(oneMinusR, current[i]), frMul(r, current[half + i]))
            }
            current = next
        }
        precondition(current.count == 1)
        return current[0]
    }

    // MARK: - Fiat-Shamir

    private func appendFr(_ transcript: inout [UInt8], _ v: Fr) {
        let vInt = frToInt(v)
        for limb in vInt {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func deriveChallenge(_ transcript: [UInt8]) -> Fr {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

/// Hashable wrapper for Fr elements (used in multiplicity computation)
struct FrKey: Hashable {
    let limbs: [UInt64]
    init(_ fr: Fr) {
        self.limbs = frToInt(fr)
    }
    static func == (lhs: FrKey, rhs: FrKey) -> Bool {
        lhs.limbs == rhs.limbs
    }
    func hash(into hasher: inout Hasher) {
        for l in limbs { hasher.combine(l) }
    }
}

/// Check if two Fr elements are equal
public func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    frToInt(a) == frToInt(b)
}
