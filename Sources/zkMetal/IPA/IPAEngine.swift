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

        let qProj = pointFromAffine(Q)

        let logN = Int(log2(Double(n)))
        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        // Transcript for Fiat-Shamir challenges
        var transcript = [UInt8]()
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

        // Flat working buffers to avoid Swift array copies in the hot loop.
        // a, b as raw UInt64 (Fr = 4 x UInt64), G as projective (12 x UInt64).
        // Fold operates in-place on the first halfLen elements.
        var aFlat = [UInt64](repeating: 0, count: n * 4)
        var bFlat = [UInt64](repeating: 0, count: n * 4)
        var gFlat = [UInt64](repeating: 0, count: n * 12)
        var gFoldBuf = [UInt64](repeating: 0, count: n * 12)  // scratch for fold output

        // Copy inputs into flat buffers
        inputA.withUnsafeBytes { src in
            aFlat.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: src.baseAddress, count: n * 32))
            }
        }
        inputB.withUnsafeBytes { src in
            bFlat.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: src.baseAddress, count: n * 32))
            }
        }
        // Convert affine generators to projective in flat buffer
        // FP_ONE = R mod p (Montgomery form of 1) for BN254 Fp
        let fpOne: [UInt64] = [0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d, 0x666ea36f7879462c, 0x0e0a77c19a07df2f]
        generators.withUnsafeBytes { src in
            for i in 0..<n {
                gFlat.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.advanced(by: i * 96).copyMemory(from: src.baseAddress!.advanced(by: i * 64), byteCount: 64)
                }
                gFlat[i * 12 + 8] = fpOne[0]
                gFlat[i * 12 + 9] = fpOne[1]
                gFlat[i * 12 + 10] = fpOne[2]
                gFlat[i * 12 + 11] = fpOne[3]
            }
        }

        var halfLen = n / 2

        // Pre-allocate scalar limb buffers to avoid per-round allocation
        var aLoLimbs = [UInt32](repeating: 0, count: n * 8)
        var aHiLimbs = [UInt32](repeating: 0, count: n * 8)

        for round in 0..<logN {
            let roundStart = profileIPA ? CFAbsoluteTimeGetCurrent() : 0

            // Compute cross inner products directly from flat buffers (no array copy)
            var crossL = Fr.zero
            var crossR = Fr.zero
            aFlat.withUnsafeBufferPointer { aPtr in
                bFlat.withUnsafeBufferPointer { bPtr in
                    withUnsafeMutableBytes(of: &crossL) { clBuf in
                        // cL = <a_lo, b_hi>
                        bn254_fr_inner_product(
                            aPtr.baseAddress!,
                            bPtr.baseAddress! + halfLen * 4,
                            Int32(halfLen),
                            clBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                    withUnsafeMutableBytes(of: &crossR) { crBuf in
                        // cR = <a_hi, b_lo>
                        bn254_fr_inner_product(
                            aPtr.baseAddress! + halfLen * 4,
                            bPtr.baseAddress!,
                            Int32(halfLen),
                            crBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                }
            }

            // Convert a_lo, a_hi to uint32 limbs for MSM
            aFlat.withUnsafeBufferPointer { aPtr in
                aLoLimbs.withUnsafeMutableBufferPointer { loBuf in
                    bn254_fr_batch_to_limbs(aPtr.baseAddress!, loBuf.baseAddress!, Int32(halfLen))
                }
                aHiLimbs.withUnsafeMutableBufferPointer { hiBuf in
                    bn254_fr_batch_to_limbs(aPtr.baseAddress! + halfLen * 4, hiBuf.baseAddress!, Int32(halfLen))
                }
            }

            // L = MSM(G_hi, a_lo) + cL*Q,  R = MSM(G_lo, a_hi) + cR*Q
            // Use dual MSM to share a single thread pool for both computations
            var msmL = PointProjective(x: .one, y: .one, z: .zero)
            var msmR = PointProjective(x: .one, y: .one, z: .zero)

            gFlat.withUnsafeBufferPointer { gPtr in
                aLoLimbs.withUnsafeBufferPointer { alBuf in
                    aHiLimbs.withUnsafeBufferPointer { ahBuf in
                        withUnsafeMutableBytes(of: &msmL) { lBuf in
                            withUnsafeMutableBytes(of: &msmR) { rBuf in
                                bn254_dual_msm_projective(
                                    UnsafeRawPointer(gPtr.baseAddress! + halfLen * 12).assumingMemoryBound(to: UInt64.self),
                                    alBuf.baseAddress!,
                                    Int32(halfLen),
                                    UnsafeRawPointer(gPtr.baseAddress!).assumingMemoryBound(to: UInt64.self),
                                    ahBuf.baseAddress!,
                                    Int32(halfLen),
                                    lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                            }
                        }
                    }
                }
            }

            let cLQ = cPointScalarMul(qProj, crossL)
            let cRQ = cPointScalarMul(qProj, crossR)
            let L = pointAdd(msmL, cLQ)
            let R = pointAdd(msmR, cRQ)

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge
            appendPoint(&transcript, L)
            appendPoint(&transcript, R)
            let x = deriveChallenge(transcript)
            // Use C CIOS Fr inverse (~100x faster than Swift Fermat inversion)
            var xInv = Fr.zero
            withUnsafeBytes(of: x) { xBuf in
                withUnsafeMutableBytes(of: &xInv) { rBuf in
                    bn254_fr_inverse(
                        xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }

            let foldStart = profileIPA ? CFAbsoluteTimeGetCurrent() : 0

            // Fold a, b in-place (output overwrites first halfLen elements)
            aFlat.withUnsafeMutableBufferPointer { aPtr in
                bFlat.withUnsafeMutableBufferPointer { bPtr in
                    withUnsafeBytes(of: x) { xBuf in
                        withUnsafeBytes(of: xInv) { xiBuf in
                            let xp = xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            let xip = xiBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            bn254_fr_vector_fold(aPtr.baseAddress!, aPtr.baseAddress! + halfLen * 4,
                                                 xp, xip, Int32(halfLen), aPtr.baseAddress!)
                            bn254_fr_vector_fold(bPtr.baseAddress!, bPtr.baseAddress! + halfLen * 4,
                                                 xip, xp, Int32(halfLen), bPtr.baseAddress!)
                        }
                    }
                }
            }

            // Fold generators using C multi-threaded Straus
            let xLimbs = frToLimbs(x)
            let xInvLimbs = frToLimbs(xInv)
            gFlat.withUnsafeBufferPointer { glBuf in
                xLimbs.withUnsafeBufferPointer { xBuf in
                    xInvLimbs.withUnsafeBufferPointer { xiBuf in
                        gFoldBuf.withUnsafeMutableBufferPointer { outBuf in
                            bn254_fold_generators(
                                UnsafeRawPointer(glBuf.baseAddress!).assumingMemoryBound(to: UInt64.self),
                                UnsafeRawPointer(glBuf.baseAddress! + halfLen * 12).assumingMemoryBound(to: UInt64.self),
                                xBuf.baseAddress!,
                                xiBuf.baseAddress!,
                                Int32(halfLen),
                                UnsafeMutableRawPointer(outBuf.baseAddress!).assumingMemoryBound(to: UInt64.self))
                        }
                    }
                }
            }
            // Copy folded generators back (only halfLen entries)
            gFoldBuf.withUnsafeBytes { src in
                gFlat.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: halfLen * 96)
                }
            }

            if profileIPA {
                let t = CFAbsoluteTimeGetCurrent()
                fputs(String(format: "  [ipa-profile] round %d (halfLen=%d): LR=%.2f ms, fold=%.2f ms, total=%.2f ms\n",
                             round, halfLen,
                             (foldStart - roundStart) * 1000,
                             (t - foldStart) * 1000,
                             (t - roundStart) * 1000), stderr)
            }

            halfLen /= 2
        }

        if profileIPA {
            let t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [ipa-profile] total prove: %.2f ms\n", (t - proofStart) * 1000), stderr)
        }

        // Extract final scalar
        let finalA: Fr = aFlat.withUnsafeBytes { buf in
            buf.load(as: Fr.self)
        }
        return IPAProof(L: Ls, R: Rs, a: finalA)
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
        // Append projective coordinates directly (x, y, z) — avoids expensive
        // Fp inversion for affine conversion. Both prover and verifier use this
        // representation consistently.
        withUnsafeBytes(of: p) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            transcript.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 96))
        }
    }

    private func appendFr(_ transcript: inout [UInt8], _ v: Fr) {
        // Append raw Montgomery representation directly (consistent for both prover/verifier)
        withUnsafeBytes(of: v) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            transcript.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 32))
        }
    }

    private func deriveChallenge(_ transcript: [UInt8]) -> Fr {
        // Use C NEON blake3 for speed
        var hash = [UInt8](repeating: 0, count: 32)
        transcript.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        // Reduce hash to Fr: interpret as 256-bit integer and reduce mod r
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]
            limbs[1] = ptr[1]
            limbs[2] = ptr[2]
            limbs[3] = ptr[3]
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
