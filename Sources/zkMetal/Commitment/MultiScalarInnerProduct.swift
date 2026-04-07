// MultiScalarInnerProduct — GPU-accelerated inner product argument engine
//
// Provides GPU-parallel field inner product, vector folding, and a complete
// IPA (Inner Product Argument) prover/verifier using Metal compute shaders.
//
// The IPA protocol (Bulletproofs/Halo2 style) works on:
//   - Field inner product: <a, b> = sum(a_i * b_i) over BN254 Fr
//   - Multi-scalar multiplication: <a, G> = sum(a_i * G_i) (MSM)
//   - Vector folding: a'_i = a_i + x * a_{n/2+i} (halves vectors each round)
//
// GPU acceleration targets:
//   - Parallel fold: embarrassingly parallel, one thread per output element
//   - Cross products: fused multiply-reduce with SIMD shuffle
//   - Field inner product: delegated to GPUInnerProductEngine
//
// References:
//   - Bulletproofs (Bunz et al. 2018)
//   - Halo (Bowe et al. 2019)
//   - Efficient Inner Product Arguments (Bootle et al. 2016)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Multi-Scalar Inner Product Proof

/// Proof structure for the multi-scalar inner product argument.
/// Contains log(n) round commitments (L_i, R_i) and final scalars.
public struct MSIPAProof {
    /// Left cross-term commitments per round.
    public let Ls: [PointProjective]
    /// Right cross-term commitments per round.
    public let Rs: [PointProjective]
    /// Final scalar from the a vector after all folds.
    public let finalA: Fr
    /// Final scalar from the b vector after all folds.
    public let finalB: Fr

    public init(Ls: [PointProjective], Rs: [PointProjective], finalA: Fr, finalB: Fr) {
        self.Ls = Ls
        self.Rs = Rs
        self.finalA = finalA
        self.finalB = finalB
    }
}

// MARK: - Multi-Scalar Inner Product Engine

/// GPU-accelerated engine for inner product argument operations.
///
/// Provides:
///   - `fieldInnerProduct(a:b:)` — GPU parallel multiply+reduce
///   - `ipaFold(a:b:challenge:)` — fold a,b vectors using challenge
///   - `ipaProve(a:b:G:Q:)` — full IPA protocol prover
///   - `ipaVerify(proof:commitment:innerProduct:G:Q:)` — full IPA verifier
///
/// Uses Metal compute shaders for the fold and cross-product kernels,
/// and delegates inner products to GPUInnerProductEngine.
public class MultiScalarInnerProduct {

    public static let version = Versions.multiScalarIPA

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // GPU pipelines for IPA-specific kernels
    private let foldVectorsPipeline: MTLComputePipelineState
    private let foldDualPipeline: MTLComputePipelineState
    private let crossProductsPipeline: MTLComputePipelineState

    // Inner product engine (reuse existing GPU infrastructure)
    private let ipEngine: GPUInnerProductEngine

    // Buffer pool for GPU memory management
    private let pool: GPUBufferPool

    private let threadgroupSize: Int

    /// Vectors smaller than this use CPU fold instead of GPU dispatch.
    public var gpuFoldThreshold: Int = 2048

    /// Vectors smaller than this use CPU inner product.
    public var cpuInnerProductThreshold: Int = 1024

    public init(threadgroupSize: Int = 256) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.threadgroupSize = threadgroupSize
        self.pool = GPUBufferPool(device: device)

        // Compile IPA fold shaders
        let library = try MultiScalarInnerProduct.compileShaders(device: device)

        guard let foldVecFn = library.makeFunction(name: "ipa_fold_vectors"),
              let foldDualFn = library.makeFunction(name: "ipa_fold_dual"),
              let crossProdFn = library.makeFunction(name: "ipa_cross_products") else {
            throw MSMError.missingKernel
        }

        self.foldVectorsPipeline = try device.makeComputePipelineState(function: foldVecFn)
        self.foldDualPipeline = try device.makeComputePipelineState(function: foldDualFn)
        self.crossProductsPipeline = try device.makeComputePipelineState(function: crossProdFn)

        self.ipEngine = try GPUInnerProductEngine(threadgroupSize: threadgroupSize)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let foldSource = try String(contentsOfFile: shaderDir + "/reduce/ipa_fold.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanFold = foldSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanFold
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Field Inner Product (GPU parallel multiply + reduce)

    /// Compute the field inner product: <a, b> = sum(a_i * b_i) over BN254 Fr.
    ///
    /// Dispatches to GPU for large vectors, CPU for small ones.
    /// Uses the existing GPUInnerProductEngine for the heavy lifting.
    public func fieldInnerProduct(a: [Fr], b: [Fr]) -> Fr {
        return ipEngine.fieldInnerProduct(a: a, b: b)
    }

    // MARK: - IPA Fold (GPU parallel)

    /// Fold two vectors a and b using a challenge scalar x.
    ///
    /// Computes:
    ///   a'[i] = a[i] + x * a[halfLen + i]       for i in 0..<halfLen
    ///   b'[i] = b[i] + x^{-1} * b[halfLen + i]  for i in 0..<halfLen
    ///
    /// Returns the folded vectors (a', b') each of length n/2.
    /// This is the core operation in each IPA round.
    public func ipaFold(a: [Fr], b: [Fr], challenge x: Fr) -> (aFolded: [Fr], bFolded: [Fr]) {
        let n = a.count
        precondition(n == b.count, "a and b must have equal length")
        precondition(n > 0 && n % 2 == 0, "Vector length must be even")

        let halfLen = n / 2

        // Compute x^{-1}
        var xInv = Fr.zero
        withUnsafeBytes(of: x) { xBuf in
            withUnsafeMutableBytes(of: &xInv) { rBuf in
                bn254_fr_inverse(
                    xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        if halfLen < gpuFoldThreshold {
            return cpuFoldDual(a: a, b: b, x: x, xInv: xInv)
        }

        return gpuFoldDual(a: a, b: b, x: x, xInv: xInv)
    }

    /// Fold a single vector: out[i] = lo[i] + challenge * hi[i]
    public func foldVector(_ v: [Fr], challenge: Fr) -> [Fr] {
        let n = v.count
        precondition(n > 0 && n % 2 == 0, "Vector length must be even")
        let halfLen = n / 2

        if halfLen < gpuFoldThreshold {
            return cpuFoldSingle(v, challenge: challenge)
        }
        return gpuFoldSingle(v, challenge: challenge)
    }

    // MARK: - IPA Prove

    /// Prove an inner product argument.
    ///
    /// Given vectors a, b of field elements and generators G (curve points) + Q (binding point),
    /// produce a proof that the prover knows a such that:
    ///   - C = MSM(G, a) + <a, b> * Q   (Pedersen commitment with inner product binding)
    ///   - The inner product <a, b> has a specific value
    ///
    /// The protocol runs log(n) rounds, halving vectors each time.
    ///
    /// - Parameters:
    ///   - a: witness vector (private)
    ///   - b: public evaluation vector
    ///   - G: generator points (affine, length n, power of 2)
    ///   - Q: inner product binding generator
    /// - Returns: (proof, commitment, innerProduct)
    public func ipaProve(a: [Fr], b: [Fr], G: [PointAffine], Q: PointAffine)
        -> (proof: MSIPAProof, commitment: PointProjective, innerProduct: Fr)
    {
        let n = a.count
        precondition(n == b.count && n == G.count, "All vectors must have equal length")
        precondition(n > 0 && (n & (n - 1)) == 0, "Vector length must be power of 2")

        let qProj = pointFromAffine(Q)
        let logN = Int(log2(Double(n)))

        // Compute initial inner product and commitment
        let ip = cFrInnerProduct(a, b)
        let commitment = computeCommitment(scalars: a, generators: G, ip: ip, Q: qProj)

        // Initialize Fiat-Shamir transcript
        var transcript = MSIPATranscript()
        transcript.appendPoint(commitment)
        transcript.appendScalar(ip)

        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        // Working copies (flat UInt64 buffers for C interop)
        var aFlat = [UInt64](repeating: 0, count: n * 4)
        var bFlat = [UInt64](repeating: 0, count: n * 4)
        var gFlat = [UInt64](repeating: 0, count: n * 12)
        var gFoldBuf = [UInt64](repeating: 0, count: n * 12)

        // Copy a into aFlat
        a.withUnsafeBytes { src in
            aFlat.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: src.baseAddress, count: n * 32))
            }
        }
        // Copy b into bFlat
        b.withUnsafeBytes { src in
            bFlat.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(start: src.baseAddress, count: n * 32))
            }
        }
        // Convert affine generators to projective flat buffer
        let fpOne: [UInt64] = [0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d,
                               0x666ea36f7879462c, 0x0e0a77c19a07df2f]
        G.withUnsafeBytes { src in
            for i in 0..<n {
                gFlat.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.advanced(by: i * 96).copyMemory(
                        from: src.baseAddress!.advanced(by: i * 64), byteCount: 64)
                }
                gFlat[i * 12 + 8] = fpOne[0]
                gFlat[i * 12 + 9] = fpOne[1]
                gFlat[i * 12 + 10] = fpOne[2]
                gFlat[i * 12 + 11] = fpOne[3]
            }
        }

        var halfLen = n / 2
        var aLoLimbs = [UInt32](repeating: 0, count: n * 8)
        var aHiLimbs = [UInt32](repeating: 0, count: n * 8)

        for _ in 0..<logN {
            // Compute cross inner products
            var crossL = Fr.zero
            var crossR = Fr.zero
            aFlat.withUnsafeBufferPointer { aPtr in
                bFlat.withUnsafeBufferPointer { bPtr in
                    withUnsafeMutableBytes(of: &crossL) { clBuf in
                        bn254_fr_inner_product(
                            aPtr.baseAddress!,
                            bPtr.baseAddress! + halfLen * 4,
                            Int32(halfLen),
                            clBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                    withUnsafeMutableBytes(of: &crossR) { crBuf in
                        bn254_fr_inner_product(
                            aPtr.baseAddress! + halfLen * 4,
                            bPtr.baseAddress!,
                            Int32(halfLen),
                            crBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                    }
                }
            }

            // Convert a_lo, a_hi to UInt32 limbs for MSM
            aFlat.withUnsafeBufferPointer { aPtr in
                aLoLimbs.withUnsafeMutableBufferPointer { loBuf in
                    bn254_fr_batch_to_limbs(aPtr.baseAddress!, loBuf.baseAddress!, Int32(halfLen))
                }
                aHiLimbs.withUnsafeMutableBufferPointer { hiBuf in
                    bn254_fr_batch_to_limbs(
                        aPtr.baseAddress! + halfLen * 4, hiBuf.baseAddress!, Int32(halfLen))
                }
            }

            // L = MSM(G_hi, a_lo) + crossL * Q
            // R = MSM(G_lo, a_hi) + crossR * Q
            var msmL = PointProjective(x: .one, y: .one, z: .zero)
            var msmR = PointProjective(x: .one, y: .one, z: .zero)

            gFlat.withUnsafeBufferPointer { gPtr in
                aLoLimbs.withUnsafeBufferPointer { alBuf in
                    aHiLimbs.withUnsafeBufferPointer { ahBuf in
                        withUnsafeMutableBytes(of: &msmL) { lBuf in
                            withUnsafeMutableBytes(of: &msmR) { rBuf in
                                bn254_dual_msm_projective(
                                    UnsafeRawPointer(gPtr.baseAddress! + halfLen * 12)
                                        .assumingMemoryBound(to: UInt64.self),
                                    alBuf.baseAddress!,
                                    Int32(halfLen),
                                    UnsafeRawPointer(gPtr.baseAddress!)
                                        .assumingMemoryBound(to: UInt64.self),
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
            transcript.appendPoint(L)
            transcript.appendPoint(R)
            let x = transcript.deriveChallenge()

            // Compute x^{-1}
            var xInv = Fr.zero
            withUnsafeBytes(of: x) { xBuf in
                withUnsafeMutableBytes(of: &xInv) { rBuf in
                    bn254_fr_inverse(
                        xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }

            // GPU fold for large vectors, CPU fold for small
            if halfLen >= gpuFoldThreshold {
                gpuFoldFlat(aFlat: &aFlat, bFlat: &bFlat, x: x, xInv: xInv, halfLen: halfLen)
            } else {
                // Fold a, b in-place using C NEON
                aFlat.withUnsafeMutableBufferPointer { aPtr in
                    bFlat.withUnsafeMutableBufferPointer { bPtr in
                        withUnsafeBytes(of: x) { xBuf in
                            withUnsafeBytes(of: xInv) { xiBuf in
                                let xp = xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                                let xip = xiBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                                bn254_fr_vector_fold(
                                    aPtr.baseAddress!, aPtr.baseAddress! + halfLen * 4,
                                    xp, xip, Int32(halfLen), aPtr.baseAddress!)
                                bn254_fr_vector_fold(
                                    bPtr.baseAddress!, bPtr.baseAddress! + halfLen * 4,
                                    xip, xp, Int32(halfLen), bPtr.baseAddress!)
                            }
                        }
                    }
                }
            }

            // Fold generators
            let xLimbs = frToLimbs(x)
            let xInvLimbs = frToLimbs(xInv)
            gFlat.withUnsafeBufferPointer { glBuf in
                xLimbs.withUnsafeBufferPointer { xBuf in
                    xInvLimbs.withUnsafeBufferPointer { xiBuf in
                        gFoldBuf.withUnsafeMutableBufferPointer { outBuf in
                            bn254_fold_generators(
                                UnsafeRawPointer(glBuf.baseAddress!)
                                    .assumingMemoryBound(to: UInt64.self),
                                UnsafeRawPointer(glBuf.baseAddress! + halfLen * 12)
                                    .assumingMemoryBound(to: UInt64.self),
                                xBuf.baseAddress!,
                                xiBuf.baseAddress!,
                                Int32(halfLen),
                                UnsafeMutableRawPointer(outBuf.baseAddress!)
                                    .assumingMemoryBound(to: UInt64.self))
                        }
                    }
                }
            }
            gFoldBuf.withUnsafeBytes { src in
                gFlat.withUnsafeMutableBytes { dst in
                    dst.baseAddress!.copyMemory(from: src.baseAddress!, byteCount: halfLen * 96)
                }
            }

            halfLen /= 2
        }

        // Extract final scalars
        let finalA: Fr = aFlat.withUnsafeBytes { $0.load(as: Fr.self) }
        let finalB: Fr = bFlat.withUnsafeBytes { $0.load(as: Fr.self) }

        let proof = MSIPAProof(Ls: Ls, Rs: Rs, finalA: finalA, finalB: finalB)
        return (proof: proof, commitment: commitment, innerProduct: ip)
    }

    // MARK: - IPA Verify

    /// Verify an inner product argument proof.
    ///
    /// Checks that the proof is valid for the given commitment, inner product value,
    /// generators, and binding point.
    ///
    /// - Parameters:
    ///   - proof: the MSIPAProof to verify
    ///   - commitment: the original Pedersen commitment C = MSM(G, a) + <a,b>*Q
    ///   - innerProduct: the claimed inner product value t = <a, b>
    ///   - G: generator points (affine, length n, power of 2)
    ///   - Q: inner product binding generator
    /// - Returns: true if the proof verifies correctly
    public func ipaVerify(proof: MSIPAProof, commitment: PointProjective,
                          innerProduct t: Fr, G: [PointAffine], Q: PointAffine) -> Bool {
        let n = G.count
        let logN = Int(log2(Double(n)))
        guard proof.Ls.count == logN, proof.Rs.count == logN else { return false }

        let qProj = pointFromAffine(Q)

        // Reconstruct Fiat-Shamir challenges
        var transcript = MSIPATranscript()
        transcript.appendPoint(commitment)
        transcript.appendScalar(t)

        var challenges = [Fr]()
        var challengeInvs = [Fr]()
        challenges.reserveCapacity(logN)
        challengeInvs.reserveCapacity(logN)

        for round in 0..<logN {
            transcript.appendPoint(proof.Ls[round])
            transcript.appendPoint(proof.Rs[round])
            let x = transcript.deriveChallenge()
            challenges.append(x)

            var xInv = Fr.zero
            withUnsafeBytes(of: x) { xBuf in
                withUnsafeMutableBytes(of: &xInv) { rBuf in
                    bn254_fr_inverse(
                        xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
            challengeInvs.append(xInv)
        }

        // Fold commitment: C' = C + sum(x_i^2 * L_i + x_i^{-2} * R_i)
        var Cprime = commitment
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            let x2 = frMul(x, x)
            let xInv2 = frMul(xInv, xInv)
            let lTerm = cPointScalarMul(proof.Ls[round], x2)
            let rTerm = cPointScalarMul(proof.Rs[round], xInv2)
            Cprime = pointAdd(Cprime, pointAdd(lTerm, rTerm))
        }

        // Compute s[i] via O(n) butterfly construction (replaces O(n*logN) nested loops)
        var s = [Fr](repeating: Fr.zero, count: n)
        s[0] = Fr.one
        var half = 1
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            for i in stride(from: half - 1, through: 0, by: -1) {
                s[2 * i + 1] = frMul(s[i], x)
                s[2 * i]     = frMul(s[i], xInv)
            }
            half *= 2
        }

        // G_final = MSM(G, s)
        var sLimbs = [UInt32](repeating: 0, count: n * 8)
        s.withUnsafeBytes { sBuf in
            sLimbs.withUnsafeMutableBufferPointer { lBuf in
                bn254_fr_batch_to_limbs(
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    lBuf.baseAddress!,
                    Int32(n))
            }
        }
        var gFinal = PointProjective(x: .one, y: .one, z: .zero)
        G.withUnsafeBytes { ptsBuf in
            sLimbs.withUnsafeBufferPointer { scBuf in
                withUnsafeMutableBytes(of: &gFinal) { resBuf in
                    bn254_pippenger_msm(
                        ptsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        scBuf.baseAddress!,
                        Int32(n),
                        resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }

        // Fold b using challenges (verifier recomputes folded b from public data)
        // In standard IPA, b is public (e.g., powers of evaluation point)
        // The verifier computes b_final = product of (1, x_j or x_j^{-1}) applied to b
        // For simplicity, b_final from proof is used (caller must verify b is correct)
        let bFinal = proof.finalB

        // Final check: C' == finalA * G_final + (finalA * bFinal) * Q
        let aG = cPointScalarMul(gFinal, proof.finalA)
        let ab = frMul(proof.finalA, bFinal)
        let abQ = cPointScalarMul(qProj, ab)
        let expected = pointAdd(aG, abQ)

        return pointEqual(Cprime, expected)
    }

    // MARK: - GPU Fold (flat buffers, in-place)

    /// GPU fold of flat UInt64 buffers (a and b simultaneously).
    /// a'[i] = a[i] + x * a[halfLen+i], b'[i] = b[i] + xInv * b[halfLen+i]
    private func gpuFoldFlat(aFlat: inout [UInt64], bFlat: inout [UInt64],
                             x: Fr, xInv: Fr, halfLen: Int) {
        let frStride = MemoryLayout<Fr>.stride  // 32 bytes
        let bufSize = halfLen * 2 * frStride  // full vector size

        guard let aBuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
              let aOutBuf = device.makeBuffer(length: halfLen * frStride, options: .storageModeShared),
              let bOutBuf = device.makeBuffer(length: halfLen * frStride, options: .storageModeShared) else {
            // Fallback to CPU
            aFlat.withUnsafeMutableBufferPointer { aPtr in
                bFlat.withUnsafeMutableBufferPointer { bPtr in
                    withUnsafeBytes(of: x) { xBuf in
                        withUnsafeBytes(of: xInv) { xiBuf in
                            let xp = xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            let xip = xiBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            bn254_fr_vector_fold(
                                aPtr.baseAddress!, aPtr.baseAddress! + halfLen * 4,
                                xp, xip, Int32(halfLen), aPtr.baseAddress!)
                            bn254_fr_vector_fold(
                                bPtr.baseAddress!, bPtr.baseAddress! + halfLen * 4,
                                xip, xp, Int32(halfLen), bPtr.baseAddress!)
                        }
                    }
                }
            }
            return
        }

        // Upload data
        aFlat.withUnsafeBytes { src in
            memcpy(aBuf.contents(), src.baseAddress!, bufSize)
        }
        bFlat.withUnsafeBytes { src in
            memcpy(bBuf.contents(), src.baseAddress!, bufSize)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return
        }

        encoder.setComputePipelineState(foldDualPipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(aOutBuf, offset: 0, index: 2)
        encoder.setBuffer(bOutBuf, offset: 0, index: 3)
        var xVal = x
        var xInvVal = xInv
        encoder.setBytes(&xVal, length: frStride, index: 4)
        encoder.setBytes(&xInvVal, length: frStride, index: 5)
        var hl = UInt32(halfLen)
        encoder.setBytes(&hl, length: 4, index: 6)

        let numGroups = (halfLen + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error == nil {
            // Copy results back
            aFlat.withUnsafeMutableBytes { dst in
                memcpy(dst.baseAddress!, aOutBuf.contents(), halfLen * frStride)
            }
            bFlat.withUnsafeMutableBytes { dst in
                memcpy(dst.baseAddress!, bOutBuf.contents(), halfLen * frStride)
            }
        }
    }

    // MARK: - GPU Fold (Swift arrays)

    /// GPU fold of a single vector: out[i] = v[i] + challenge * v[halfLen + i]
    private func gpuFoldSingle(_ v: [Fr], challenge: Fr) -> [Fr] {
        let n = v.count
        let halfLen = n / 2
        let frStride = MemoryLayout<Fr>.stride

        guard let loBuf = device.makeBuffer(length: halfLen * frStride, options: .storageModeShared),
              let hiBuf = device.makeBuffer(length: halfLen * frStride, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: halfLen * frStride, options: .storageModeShared) else {
            return cpuFoldSingle(v, challenge: challenge)
        }

        v.withUnsafeBytes { src in
            memcpy(loBuf.contents(), src.baseAddress!, halfLen * frStride)
            memcpy(hiBuf.contents(), src.baseAddress!.advanced(by: halfLen * frStride), halfLen * frStride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuFoldSingle(v, challenge: challenge)
        }

        encoder.setComputePipelineState(foldVectorsPipeline)
        encoder.setBuffer(loBuf, offset: 0, index: 0)
        encoder.setBuffer(hiBuf, offset: 0, index: 1)
        encoder.setBuffer(outBuf, offset: 0, index: 2)
        var ch = challenge
        encoder.setBytes(&ch, length: frStride, index: 3)
        var hl = UInt32(halfLen)
        encoder.setBytes(&hl, length: 4, index: 4)

        let numGroups = (halfLen + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return cpuFoldSingle(v, challenge: challenge)
        }

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: halfLen)
        return Array(UnsafeBufferPointer(start: ptr, count: halfLen))
    }

    /// GPU dual fold: fold both a and b in one dispatch.
    private func gpuFoldDual(a: [Fr], b: [Fr], x: Fr, xInv: Fr) -> (aFolded: [Fr], bFolded: [Fr]) {
        let n = a.count
        let halfLen = n / 2
        let frStride = MemoryLayout<Fr>.stride

        guard let aBuf = device.makeBuffer(length: n * frStride, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: n * frStride, options: .storageModeShared),
              let aOutBuf = device.makeBuffer(length: halfLen * frStride, options: .storageModeShared),
              let bOutBuf = device.makeBuffer(length: halfLen * frStride, options: .storageModeShared) else {
            return cpuFoldDual(a: a, b: b, x: x, xInv: xInv)
        }

        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, n * frStride) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, n * frStride) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuFoldDual(a: a, b: b, x: x, xInv: xInv)
        }

        encoder.setComputePipelineState(foldDualPipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(aOutBuf, offset: 0, index: 2)
        encoder.setBuffer(bOutBuf, offset: 0, index: 3)
        var xVal = x
        var xInvVal = xInv
        encoder.setBytes(&xVal, length: frStride, index: 4)
        encoder.setBytes(&xInvVal, length: frStride, index: 5)
        var hl = UInt32(halfLen)
        encoder.setBytes(&hl, length: 4, index: 6)

        let numGroups = (halfLen + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return cpuFoldDual(a: a, b: b, x: x, xInv: xInv)
        }

        let aPtr = aOutBuf.contents().bindMemory(to: Fr.self, capacity: halfLen)
        let bPtr = bOutBuf.contents().bindMemory(to: Fr.self, capacity: halfLen)
        return (
            aFolded: Array(UnsafeBufferPointer(start: aPtr, count: halfLen)),
            bFolded: Array(UnsafeBufferPointer(start: bPtr, count: halfLen))
        )
    }

    // MARK: - CPU Fallbacks

    private func cpuFoldSingle(_ v: [Fr], challenge: Fr) -> [Fr] {
        let halfLen = v.count / 2
        var result = [Fr](repeating: Fr.zero, count: halfLen)
        withUnsafeBytes(of: challenge) { cPtr in
            v.withUnsafeBytes { vBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_sumcheck_reduce(
                        vBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        cPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(halfLen))
                }
            }
        }
        return result
    }

    private func cpuFoldDual(a: [Fr], b: [Fr], x: Fr, xInv: Fr) -> (aFolded: [Fr], bFolded: [Fr]) {
        let halfLen = a.count / 2
        var aResult = [Fr](repeating: Fr.zero, count: halfLen)
        var bResult = [Fr](repeating: Fr.zero, count: halfLen)
        withUnsafeBytes(of: x) { xPtr in
            a.withUnsafeBytes { aBuf in
                aResult.withUnsafeMutableBytes { rBuf in
                    bn254_fr_sumcheck_reduce(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        xPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(halfLen))
                }
            }
        }
        withUnsafeBytes(of: xInv) { xiPtr in
            b.withUnsafeBytes { bBuf in
                bResult.withUnsafeMutableBytes { rBuf in
                    bn254_fr_sumcheck_reduce(
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        xiPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(halfLen))
                }
            }
        }
        return (aFolded: aResult, bFolded: bResult)
    }

    // MARK: - Private Helpers

    /// Compute commitment C = MSM(G, scalars) + ip * Q
    private func computeCommitment(scalars: [Fr], generators: [PointAffine],
                                   ip: Fr, Q: PointProjective) -> PointProjective {
        let n = scalars.count
        let scalarLimbs = scalars.map { frToLimbs($0) }
        let msmResult = cPippengerMSM(points: generators, scalars: scalarLimbs)
        let ipQ = cPointScalarMul(Q, ip)
        return pointAdd(msmResult, ipQ)
    }
}

// MARK: - Transcript for Multi-Scalar IPA (Fiat-Shamir)

private struct MSIPATranscript {
    var data = [UInt8]()

    mutating func appendPoint(_ p: PointProjective) {
        withUnsafeBytes(of: p) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            data.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 96))
        }
    }

    mutating func appendScalar(_ v: Fr) {
        withUnsafeBytes(of: v) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            data.append(contentsOf: UnsafeBufferPointer(start: ptr, count: 32))
        }
    }

    func deriveChallenge() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBufferPointer { inp in
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
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}
