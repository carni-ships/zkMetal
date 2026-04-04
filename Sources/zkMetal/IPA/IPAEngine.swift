// IPA (Inner Product Argument) Engine
// Implements Pedersen commitment + Bulletproofs-style IPA opening proof.
//
// Protocol:
//   Commit: C = MSM(G, a) + v*Q  where v = <a, b>
//   Prove: log(n) rounds of halving — split vectors, compute L/R, get challenge, fold
//   Verify: reconstruct folded commitment from L/R and challenges, check against final scalar
//
// References: Bulletproofs (Bünz et al. 2018), Halo (Bowe et al. 2019)

import Foundation
import Metal
import NeonFieldOps

public struct IPAProof {
    public let L: [PointProjective]  // left commitments per round
    public let R: [PointProjective]  // right commitments per round
    public let a: Fr                  // final scalar (length-1 vector)

    public init(L: [PointProjective], R: [PointProjective], a: Fr) {
        self.L = L
        self.R = R
        self.a = a
    }
}

public class IPAEngine {
    public static let version = Versions.ipa
    public let msmEngine: MetalMSM
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let foldGeneratorsFunction: MTLComputePipelineState?

    /// Generator points G_0, G_1, ..., G_{n-1}
    public let generators: [PointAffine]
    /// Blinding generator Q (for binding inner product value)
    public let Q: PointAffine

    // GPU threshold: use GPU fold for halfLen >= this
    // GPU dispatch + batchToAffine overhead dominates below ~4096 points
    static let gpuFoldThreshold = 4096

    /// Profiling flag — when true, prints per-round timing to stderr
    public var profileIPA = false

    // Grow-only buffer cache for GPU fold kernel
    private var bufferCache: [String: (MTLBuffer, Int)] = [:]

    /// Get or grow a cached buffer for the given slot. Reuses existing buffer if large enough.
    private func getCachedBuffer(slot: String, minBytes: Int) -> MTLBuffer {
        if let (buf, cap) = bufferCache[slot], cap >= minBytes {
            return buf
        }
        let buf = device.makeBuffer(length: minBytes, options: .storageModeShared)!
        bufferCache[slot] = (buf, minBytes)
        return buf
    }

    /// Create an IPA engine with the given generators and blinding point.
    /// generators: n points in affine form (n must be power of 2)
    /// Q: separate generator for inner product binding
    public init(generators: [PointAffine], Q: PointAffine) throws {
        precondition(generators.count > 0 && (generators.count & (generators.count - 1)) == 0,
                     "Generator count must be power of 2")
        self.generators = generators
        self.Q = Q
        self.msmEngine = try MetalMSM()
        self.device = msmEngine.device
        self.commandQueue = msmEngine.commandQueue

        // Compile batch fold kernel
        do {
            let shaderDir = IPAEngine.findShaderDir()
            let curveSource = try String(contentsOfFile: shaderDir + "/geometry/bn254_curve.metal", encoding: .utf8)
            let fpSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fp.metal", encoding: .utf8)
            let foldSource = try String(contentsOfFile: shaderDir + "/geometry/batch_scalar_mul.metal", encoding: .utf8)
            // Inline includes: remove #include lines and concatenate
            let fpClean = fpSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
            let curveClean = curveSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
            let foldClean = foldSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
            let combined = fpClean + "\n" + curveClean + "\n" + foldClean
            let options = MTLCompileOptions()
            options.fastMathEnabled = true
            let library = try device.makeLibrary(source: combined, options: options)
            if let fn = library.makeFunction(name: "batch_fold_generators") {
                self.foldGeneratorsFunction = try device.makeComputePipelineState(function: fn)
            } else {
                self.foldGeneratorsFunction = nil
            }
        } catch {
            self.foldGeneratorsFunction = nil
        }
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for path in ["\(execDir)/../Sources/Shaders", "./Sources/Shaders"] {
            if FileManager.default.fileExists(atPath: "\(path)/geometry/batch_scalar_mul.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    /// Generate test generators deterministically (NOT secure — for testing only).
    /// Creates n+1 points: n generators + Q, by scalar-multiplying the BN254 generator.
    public static func generateTestGenerators(count n: Int) -> (generators: [PointAffine], Q: PointAffine) {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let g = PointAffine(x: gx, y: gy)
        let gProj = pointFromAffine(g)

        // Generate n+1 distinct points: (i+1)*G for i in 0..n
        var points = [PointProjective]()
        points.reserveCapacity(n + 1)
        var acc = gProj
        for _ in 0..<(n + 1) {
            points.append(acc)
            acc = pointAdd(acc, gProj)
        }
        let affine = batchToAffine(points)
        return (generators: Array(affine.prefix(n)), Q: affine[n])
    }

    /// Commit to a vector: C = MSM(G, a)
    public func commit(_ a: [Fr]) throws -> PointProjective {
        precondition(a.count == generators.count)
        let n = a.count
        // C batch Fr-to-limbs + Pippenger
        var flatScalars = [UInt32](repeating: 0, count: n * 8)
        a.withUnsafeBytes { aBuf in
            flatScalars.withUnsafeMutableBufferPointer { lBuf in
                bn254_fr_batch_to_limbs(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    lBuf.baseAddress!,
                    Int32(n)
                )
            }
        }
        var result = PointProjective(x: .one, y: .one, z: .zero)
        generators.withUnsafeBytes { ptsBuf in
            flatScalars.withUnsafeBufferPointer { scBuf in
                withUnsafeMutableBytes(of: &result) { resBuf in
                    bn254_pippenger_msm(
                        ptsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        scBuf.baseAddress!,
                        Int32(n),
                        resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    /// CPU multi-scalar multiplication using C Pippenger for speed.
    /// Uses C batch-to-affine and C batch Fr-to-limbs to avoid slow Swift ops.
    private func cpuMSM(points: [PointProjective], scalars: [Fr]) -> PointProjective {
        let n = points.count
        if n == 0 { return pointIdentity() }

        // C batch-to-affine: much faster than Swift batchToAffine
        var affPts = [PointAffine](repeating: PointAffine(x: .one, y: .one), count: n)
        points.withUnsafeBytes { projBuf in
            affPts.withUnsafeMutableBytes { affBuf in
                bn254_batch_to_affine(
                    projBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    affBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n)
                )
            }
        }

        // C batch Fr-to-limbs: much faster than per-element Swift frToLimbs
        var flatScalars = [UInt32](repeating: 0, count: n * 8)
        scalars.withUnsafeBytes { sBuf in
            flatScalars.withUnsafeMutableBufferPointer { lBuf in
                bn254_fr_batch_to_limbs(
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    lBuf.baseAddress!,
                    Int32(n)
                )
            }
        }

        var result = PointProjective(x: .one, y: .one, z: .zero)
        affPts.withUnsafeBytes { ptsBuf in
            flatScalars.withUnsafeBufferPointer { scBuf in
                withUnsafeMutableBytes(of: &result) { resBuf in
                    bn254_pippenger_msm(
                        ptsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        scBuf.baseAddress!,
                        Int32(n),
                        resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    /// Compute inner product <a, b> = sum(a_i * b_i)
    public static func innerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        precondition(a.count == b.count)
        var result = Fr.zero
        for i in 0..<a.count {
            result = frAdd(result, frMul(a[i], b[i]))
        }
        return result
    }

    /// Create an IPA opening proof.
    /// Proves that commitment C opens to vector `a` with inner product <a, b> = v
    /// where b is the evaluation vector (known to both prover and verifier).
    ///
    /// The commitment includes the inner product binding: C = MSM(G, a) + v*Q
    public func createProof(a inputA: [Fr], b inputB: [Fr]) throws -> IPAProof {
        let n = inputA.count
        precondition(n == inputB.count)
        precondition(n == generators.count)
        precondition(n > 0 && (n & (n - 1)) == 0)

        let proofStart = profileIPA ? CFAbsoluteTimeGetCurrent() : 0

        var a = inputA
        var b = inputB
        var G = generators.map { pointFromAffine($0) }
        let qProj = pointFromAffine(Q)

        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        let logN = Int(log2(Double(n)))
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        // Transcript for Fiat-Shamir challenges
        var transcript = [UInt8]()
        // Seed with bound commitment C_bound = MSM(G, a) + v*Q
        // Must match what the verifier seeds with
        let C = try commit(inputA)
        let v = cFrInnerProduct(inputA, inputB)
        let vQ = cPointScalarMul(qProj, v)
        let Cbound = pointAdd(C, vQ)
        appendPoint(&transcript, Cbound)
        appendFr(&transcript, v)

        if profileIPA {
            let t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [ipa-profile] commit+setup: %.2f ms\n", (t - proofStart) * 1000), stderr)
        }

        var halfLen = n / 2

        for round in 0..<logN {
            let roundStart = profileIPA ? CFAbsoluteTimeGetCurrent() : 0

            // Split vectors
            let aL = Array(a.prefix(halfLen))
            let aR = Array(a.suffix(halfLen))
            let bL = Array(b.prefix(halfLen))
            let bR = Array(b.suffix(halfLen))
            let GL = Array(G.prefix(halfLen))
            let GR = Array(G.suffix(halfLen))

            // Compute cross inner products (C-accelerated)
            let cL = cFrInnerProduct(aL, bR)
            let cR = cFrInnerProduct(aR, bL)

            // L = MSM(GR, aL) + cL * Q
            let L = try computeLR(generators: GR, scalars: aL, crossIP: cL, Q: qProj)
            // R = MSM(GL, aR) + cR * Q
            let R = try computeLR(generators: GL, scalars: aR, crossIP: cR, Q: qProj)

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge
            appendPoint(&transcript, L)
            appendPoint(&transcript, R)
            let x = deriveChallenge(transcript)
            let xInv = frInverse(x)

            let foldStart = profileIPA ? CFAbsoluteTimeGetCurrent() : 0

            // Fold: a' = aL * x + aR * x^(-1), b' = bL * x^(-1) + bR * x
            let newA = cFrVectorFold(aL, aR, x: x, xInv: xInv)
            let newB = cFrVectorFold(bL, bR, x: xInv, xInv: x)

            // Fold generators: G'_i = xInv * GL_i + x * GR_i
            let newG: [PointProjective]
            if halfLen >= IPAEngine.gpuFoldThreshold, let pipeline = foldGeneratorsFunction {
                // GPU path with cached buffers and setBytes for scalars
                newG = try gpuFoldGenerators(GL: GL, GR: GR, x: x, xInv: xInv,
                                             halfLen: halfLen, pipelineState: pipeline)
            } else {
                // CPU path: C multi-threaded Straus fold
                let xLimbs = frToLimbs(x)
                let xInvLimbs = frToLimbs(xInv)
                var cpuG = [PointProjective](repeating: pointIdentity(), count: halfLen)
                GL.withUnsafeBytes { glBuf in
                    GR.withUnsafeBytes { grBuf in
                        xLimbs.withUnsafeBufferPointer { xBuf in
                            xInvLimbs.withUnsafeBufferPointer { xiBuf in
                                cpuG.withUnsafeMutableBytes { outBuf in
                                    bn254_fold_generators(
                                        glBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                        grBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                        xBuf.baseAddress!,
                                        xiBuf.baseAddress!,
                                        Int32(halfLen),
                                        outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                                    )
                                }
                            }
                        }
                    }
                }
                newG = cpuG
            }

            if profileIPA {
                let t = CFAbsoluteTimeGetCurrent()
                fputs(String(format: "  [ipa-profile] round %d (halfLen=%d): LR=%.2f ms, fold=%.2f ms, total=%.2f ms\n",
                             round, halfLen,
                             (foldStart - roundStart) * 1000,
                             (t - foldStart) * 1000,
                             (t - roundStart) * 1000), stderr)
            }

            a = newA
            b = newB
            G = newG
            halfLen /= 2
        }

        if profileIPA {
            let t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [ipa-profile] total prove: %.2f ms\n", (t - proofStart) * 1000), stderr)
        }

        precondition(a.count == 1)
        return IPAProof(L: Ls, R: Rs, a: a[0])
    }

    /// Verify an IPA proof.
    /// Given commitment C (= MSM(G, a) + v*Q), evaluation vector b, inner product value v,
    /// and proof (L[], R[], final_a), checks:
    ///   1. Reconstruct folded commitment using challenges
    ///   2. Check: C_final == final_a * G_final + (final_a * b_final) * Q
    public func verify(commitment C: PointProjective, b inputB: [Fr],
                       innerProductValue v: Fr, proof: IPAProof) -> Bool {
        let n = generators.count
        let logN = Int(log2(Double(n)))
        guard proof.L.count == logN, proof.R.count == logN else { return false }
        guard inputB.count == n else { return false }

        let verifyStart = profileIPA ? CFAbsoluteTimeGetCurrent() : 0

        let qProj = pointFromAffine(Q)

        // Reconstruct challenges from transcript
        var transcript = [UInt8]()
        appendPoint(&transcript, C)
        appendFr(&transcript, v)

        var challenges = [Fr]()
        challenges.reserveCapacity(logN)

        for round in 0..<logN {
            appendPoint(&transcript, proof.L[round])
            appendPoint(&transcript, proof.R[round])
            let x = deriveChallenge(transcript)
            challenges.append(x)
        }

        // Fold commitment: C' = C + sum(x_i^2 * L_i + x_i^(-2) * R_i)
        // Use C point scalar mul for massive speedup over Swift
        var Cprime = C
        for round in 0..<logN {
            let x = challenges[round]
            let x2 = frMul(x, x)
            let xInv = frInverse(x)
            let xInv2 = frMul(xInv, xInv)
            let lTerm = cPointScalarMul(proof.L[round], x2)
            let rTerm = cPointScalarMul(proof.R[round], xInv2)
            Cprime = pointAdd(Cprime, pointAdd(lTerm, rTerm))
        }

        // Fold generators: G_final = MSM(G, s) where s_i = product of x_j^(±1)
        // Precompute challenges and inverses, then use C Fr operations
        var challengeInvs = [Fr]()
        challengeInvs.reserveCapacity(logN)
        for round in 0..<logN {
            challengeInvs.append(frInverse(challenges[round]))
        }

        // Compute s[] — use Swift frMul (C inner_product used for larger ops)
        // This is n×logN muls, manageable even in Swift for typical IPA sizes
        var s = [Fr](repeating: Fr.one, count: n)
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                if bit == 0 {
                    s[i] = frMul(s[i], xInv)
                } else {
                    s[i] = frMul(s[i], x)
                }
            }
        }

        // G_final = MSM(G, s) — use C Pippenger
        let sLimbs = s.map { frToLimbs($0) }
        let gFinal = cPippengerMSM(points: generators, scalars: sLimbs)

        // Fold b using C vector fold
        var bFolded = inputB
        var halfLen = n / 2
        for round in 0..<logN {
            let bL = Array(bFolded.prefix(halfLen))
            let bR = Array(bFolded.suffix(halfLen))
            bFolded = cFrVectorFold(bL, bR, x: challengeInvs[round], xInv: challenges[round])
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Check: C' == proof.a * G_final + (proof.a * bFinal) * Q
        let aG = cPointScalarMul(gFinal, proof.a)
        let ab = frMul(proof.a, bFinal)
        let abQ = cPointScalarMul(qProj, ab)
        let expected = pointAdd(aG, abQ)

        if profileIPA {
            let t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [ipa-profile] total verify: %.2f ms\n", (t - verifyStart) * 1000), stderr)
        }

        return pointEqual(Cprime, expected)
    }

    // MARK: - GPU batch fold with cached buffers and single command buffer

    private func gpuFoldGenerators(GL: [PointProjective], GR: [PointProjective],
                                   x: Fr, xInv: Fr, halfLen: Int,
                                   pipelineState: MTLComputePipelineState) throws -> [PointProjective] {
        // Convert projective points to affine for GPU input
        let glAffine = batchToAffine(GL)
        let grAffine = batchToAffine(GR)

        // Convert scalars to integer form (non-Montgomery) as 8×UInt32
        let xLimbs = frToLimbs(x)
        let xInvLimbs = frToLimbs(xInv)

        // Use grow-only cached buffers to avoid re-allocation across rounds
        let pointStride = MemoryLayout<PointAffine>.stride
        let projStride = MemoryLayout<PointProjective>.stride

        let glBuf = getCachedBuffer(slot: "foldGL", minBytes: halfLen * pointStride)
        let grBuf = getCachedBuffer(slot: "foldGR", minBytes: halfLen * pointStride)
        let outBuf = getCachedBuffer(slot: "foldOut", minBytes: halfLen * projStride)

        // Copy input data into cached buffers
        glAffine.withUnsafeBytes { src in
            glBuf.contents().copyMemory(from: src.baseAddress!, byteCount: halfLen * pointStride)
        }
        grAffine.withUnsafeBytes { src in
            grBuf.contents().copyMemory(from: src.baseAddress!, byteCount: halfLen * pointStride)
        }

        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw MSMError.noCommandBuffer
        }

        enc.setComputePipelineState(pipelineState)
        enc.setBuffer(glBuf, offset: 0, index: 0)
        enc.setBuffer(grBuf, offset: 0, index: 1)
        // setBytes for small scalar constants — avoids buffer allocation overhead
        var xInvScalar = xInvLimbs  // [UInt32] × 8
        var xScalar = xLimbs
        enc.setBytes(&xInvScalar, length: 32, index: 2)
        enc.setBytes(&xScalar, length: 32, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var halfLenVal = UInt32(halfLen)
        enc.setBytes(&halfLenVal, length: 4, index: 5)

        let tgSize = min(halfLen, Int(pipelineState.maxTotalThreadsPerThreadgroup))
        let numGroups = (halfLen + tgSize - 1) / tgSize
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }

        // Read results
        let ptr = outBuf.contents().bindMemory(to: PointProjective.self, capacity: halfLen)
        return Array(UnsafeBufferPointer(start: ptr, count: halfLen))
    }

    // MARK: - Private helpers

    private func computeLR(generators G: [PointProjective], scalars a: [Fr],
                           crossIP c: Fr, Q qProj: PointProjective) throws -> PointProjective {
        // MSM(G, a) + c * Q
        let msmResult: PointProjective
        if G.count >= 64 {
            // For larger MSMs, Pippenger is more efficient despite affine conversion cost
            msmResult = cpuMSM(points: G, scalars: a)
        } else {
            // For small MSMs, direct projective scalar-mul avoids overhead
            msmResult = cMSMProjective(points: G, scalars: a)
        }
        let cQ = cPointScalarMul(qProj, c)
        return pointAdd(msmResult, cQ)
    }

    private func appendPoint(_ transcript: inout [UInt8], _ p: PointProjective) {
        // Use C CIOS field ops for projective-to-affine (much faster than Swift fpInverse)
        var affine = [UInt64](repeating: 0, count: 8)
        withUnsafeBytes(of: p) { pBuf in
            affine.withUnsafeMutableBufferPointer { affBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    affBuf.baseAddress!
                )
            }
        }
        // Append x and y coordinates as bytes (already in Montgomery form)
        // We need to convert from Montgomery to integer form for hashing
        for limb in affine {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

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
        // Reduce hash to Fr: interpret as 256-bit integer and reduce mod r
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        // Convert to Fr (handles modular reduction)
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))  // to Montgomery form
    }
}

// MARK: - Point scalar multiplication (CPU, double-and-add)

public func pointScalarMul(_ p: PointProjective, _ scalar: Fr) -> PointProjective {
    let limbs = frToInt(scalar)
    var result = pointIdentity()
    var base = p
    for limb in limbs {
        var word = limb
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = pointAdd(result, base)
            }
            base = pointDouble(base)
            word >>= 1
        }
    }
    return result
}

// MARK: - Point equality check

public func pointEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
    // Compare in projective: a.X * b.Z^2 == b.X * a.Z^2 and a.Y * b.Z^3 == b.Y * a.Z^3
    if pointIsIdentity(a) && pointIsIdentity(b) { return true }
    if pointIsIdentity(a) || pointIsIdentity(b) { return false }

    let aZ2 = fpSqr(a.z)
    let bZ2 = fpSqr(b.z)
    let aZ3 = fpMul(a.z, aZ2)
    let bZ3 = fpMul(b.z, bZ2)

    let lhsX = fpMul(a.x, bZ2)
    let rhsX = fpMul(b.x, aZ2)
    let lhsY = fpMul(a.y, bZ3)
    let rhsY = fpMul(b.y, aZ3)

    return fpToInt(lhsX) == fpToInt(rhsX) && fpToInt(lhsY) == fpToInt(rhsY)
}
