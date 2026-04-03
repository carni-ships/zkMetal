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
}

public class IPAEngine {
    public let msmEngine: MetalMSM
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let foldGeneratorsFunction: MTLComputePipelineState?

    /// Generator points G_0, G_1, ..., G_{n-1}
    public let generators: [PointAffine]
    /// Blinding generator Q (for binding inner product value)
    public let Q: PointAffine

    // GPU threshold: use GPU fold for halfLen >= this
    static let gpuFoldThreshold = 16

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
        // Use C Pippenger for all sizes (faster than GPU for typical IPA sizes ≤ 4096)
        let scalars = a.map { frToLimbs($0) }
        return cPippengerMSM(points: generators, scalars: scalars)
    }

    /// CPU multi-scalar multiplication using C Pippenger for speed
    private func cpuMSM(points: [PointProjective], scalars: [Fr]) -> PointProjective {
        let affPts = batchToAffine(points)
        let limbScalars = scalars.map { frToLimbs($0) }
        return cPippengerMSM(points: affPts, scalars: limbScalars)
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

        var halfLen = n / 2

        for _ in 0..<logN {
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

            // Fold: a' = aL * x + aR * x^(-1), b' = bL * x^(-1) + bR * x
            let newA = cFrVectorFold(aL, aR, x: x, xInv: xInv)
            let newB = cFrVectorFold(bL, bR, x: xInv, xInv: x)

            // Fold generators: G'_i = xInv * GL_i + x * GR_i
            // Use C multi-threaded fold (much faster than GPU dispatch or Swift scalar mul)
            let xLimbs = frToLimbs(x)
            let xInvLimbs = frToLimbs(xInv)
            var newG = [PointProjective](repeating: pointIdentity(), count: halfLen)
            GL.withUnsafeBytes { glBuf in
                GR.withUnsafeBytes { grBuf in
                    xLimbs.withUnsafeBufferPointer { xBuf in
                        xInvLimbs.withUnsafeBufferPointer { xiBuf in
                            newG.withUnsafeMutableBytes { outBuf in
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

            a = newA
            b = newB
            G = newG
            halfLen /= 2
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

        return pointEqual(Cprime, expected)
    }

    // MARK: - GPU batch fold

    private func gpuFoldGenerators(GL: [PointProjective], GR: [PointProjective],
                                   x: Fr, xInv: Fr, halfLen: Int,
                                   pipelineState: MTLComputePipelineState) throws -> [PointProjective] {
        // Convert projective points to affine for GPU input
        let glAffine = batchToAffine(GL)
        let grAffine = batchToAffine(GR)

        // Convert scalars to integer form (non-Montgomery) as 8×UInt32
        let xLimbs = frToLimbs(x)
        let xInvLimbs = frToLimbs(xInv)

        // Create GPU buffers
        let pointStride = MemoryLayout<PointAffine>.stride
        let projStride = MemoryLayout<PointProjective>.stride

        let glBuf = device.makeBuffer(bytes: glAffine, length: halfLen * pointStride, options: .storageModeShared)!
        let grBuf = device.makeBuffer(bytes: grAffine, length: halfLen * pointStride, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: halfLen * projStride, options: .storageModeShared)!

        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw MSMError.noCommandBuffer
        }

        enc.setComputePipelineState(pipelineState)
        enc.setBuffer(glBuf, offset: 0, index: 0)
        enc.setBuffer(grBuf, offset: 0, index: 1)
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
        // MSM(G, a) + c * Q — always use C Pippenger (faster than GPU for IPA sizes)
        let msmResult = cpuMSM(points: G, scalars: a)
        let cQ = cpuMSM(points: [qProj], scalars: [c])
        return pointAdd(msmResult, cQ)
    }

    private func appendPoint(_ transcript: inout [UInt8], _ p: PointProjective) {
        let aff = batchToAffine([p])[0]
        let xInt = fpToInt(aff.x)
        let yInt = fpToInt(aff.y)
        for limb in xInt {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
        for limb in yInt {
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

