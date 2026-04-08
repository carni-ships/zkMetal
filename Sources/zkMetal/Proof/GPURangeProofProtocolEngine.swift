// GPURangeProofProtocolEngine — GPU-accelerated range proof protocol engine
//
// Full Bulletproofs-style range proof protocol with:
//   - Bit decomposition approach: proves v in [0, 2^n) without revealing v
//   - GPU-accelerated vector Pedersen commitments via Metal MSM
//   - GPU-accelerated inner product argument (IPA) with halving rounds
//   - Batch range proofs for multiple values (aggregated proof)
//   - Sigma protocol structure: commit -> challenge -> response -> verify
//
// Protocol flow (single proof):
//   1. Prover bit-decomposes v into a_L in {0,1}^n, sets a_R = a_L - 1^n
//   2. Prover commits A = <a_L, G> + <a_R, H> + alpha*h (GPU MSM)
//   3. Prover commits S = <s_L, G> + <s_R, H> + rho*h   (GPU MSM)
//   4. Verifier sends challenges y, z (Fiat-Shamir)
//   5. Prover computes polynomial t(X) = <l(X), r(X)>, commits T1, T2
//   6. Verifier sends challenge x (Fiat-Shamir)
//   7. Prover evaluates l(x), r(x), sends taux, mu, tHat
//   8. Inner product argument proves <l, r> = tHat (GPU MSM per round)
//
// Batch proof flow:
//   - Aggregates m values into a single proof of size O(log(m*n))
//   - Uses random linear combination with powers of a challenge
//   - Single IPA over concatenated/combined vectors
//
// References:
//   - Bulletproofs: Short Proofs for Confidential Transactions (Bunz et al. 2018)
//   - https://eprint.iacr.org/2017/1066

import Foundation
import Metal
import NeonFieldOps

// MARK: - Protocol Configuration

/// Configuration for the range proof protocol engine.
public struct RangeProofProtocolConfig {
    /// Bit size of the range (must be power of 2 in [8, 64])
    public let bitSize: Int
    /// Whether to use GPU acceleration when available
    public let useGPU: Bool
    /// Minimum vector size to dispatch to GPU MSM (below this, use CPU)
    public let gpuThreshold: Int

    public init(bitSize: Int = 32, useGPU: Bool = true, gpuThreshold: Int = 16) {
        precondition(bitSize >= 8 && bitSize <= 64 && (bitSize & (bitSize - 1)) == 0,
                     "bitSize must be power of 2 in [8, 64]")
        self.bitSize = bitSize
        self.useGPU = useGPU
        self.gpuThreshold = gpuThreshold
    }
}

// MARK: - Protocol Transcript

/// Fiat-Shamir transcript for the range proof protocol.
/// Maintains running state for deterministic challenge derivation.
public struct RangeProofTranscript {
    private var state: [UInt8] = []
    private var challengeCount: Int = 0

    public init() {}

    /// Append a domain separator label.
    public mutating func appendLabel(_ label: String) {
        state.append(contentsOf: Array(label.utf8))
        state.append(0) // null separator
    }

    /// Append a field element.
    public mutating func appendScalar(_ s: Fr) {
        withUnsafeBytes(of: s) { buf in
            state.append(contentsOf: buf)
        }
    }

    /// Append a point (serialized as x, y, z coordinates).
    public mutating func appendPoint(_ p: PointProjective) {
        withUnsafeBytes(of: p.x) { state.append(contentsOf: $0) }
        withUnsafeBytes(of: p.y) { state.append(contentsOf: $0) }
        withUnsafeBytes(of: p.z) { state.append(contentsOf: $0) }
    }

    /// Append raw bytes.
    public mutating func appendBytes(_ bytes: [UInt8]) {
        state.append(contentsOf: bytes)
    }

    /// Squeeze a challenge from the current state via Blake3.
    public mutating func challenge() -> Fr {
        challengeCount += 1
        // Include counter to ensure distinct challenges
        withUnsafeBytes(of: challengeCount) { state.append(contentsOf: $0) }

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
        // Reduce to field element
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - Protocol Proof Structures

/// A single range proof produced by the protocol engine.
public struct ProtocolRangeProof {
    /// Pedersen commitment to the value: V = v*g + gamma*h
    public let V: PointProjective
    /// Bit decomposition commitment A
    public let A: PointProjective
    /// Blinding commitment S
    public let S: PointProjective
    /// Polynomial commitment T1 (linear coefficient)
    public let T1: PointProjective
    /// Polynomial commitment T2 (quadratic coefficient)
    public let T2: PointProjective
    /// Blinding factor for tHat evaluation
    public let taux: Fr
    /// Combined blinding: mu = alpha + rho*x
    public let mu: Fr
    /// Inner product evaluation: tHat = t(x)
    public let tHat: Fr
    /// Inner product argument proof
    public let ipProof: ProtocolIPAProof
    /// Bit size of the range
    public let bitSize: Int

    public init(V: PointProjective, A: PointProjective, S: PointProjective,
                T1: PointProjective, T2: PointProjective,
                taux: Fr, mu: Fr, tHat: Fr,
                ipProof: ProtocolIPAProof, bitSize: Int) {
        self.V = V
        self.A = A
        self.S = S
        self.T1 = T1
        self.T2 = T2
        self.taux = taux
        self.mu = mu
        self.tHat = tHat
        self.ipProof = ipProof
        self.bitSize = bitSize
    }
}

/// Inner product argument proof within the protocol.
public struct ProtocolIPAProof {
    /// Left commitments per halving round
    public let Ls: [PointProjective]
    /// Right commitments per halving round
    public let Rs: [PointProjective]
    /// Final scalar a
    public let a: Fr
    /// Final scalar b
    public let b: Fr

    public init(Ls: [PointProjective], Rs: [PointProjective], a: Fr, b: Fr) {
        self.Ls = Ls
        self.Rs = Rs
        self.a = a
        self.b = b
    }
}

/// Aggregated batch range proof for multiple values.
public struct BatchProtocolRangeProof {
    /// Individual value commitments V_i = v_i*g + gamma_i*h
    public let commitments: [PointProjective]
    /// Individual A commitments (one per value)
    public let As: [PointProjective]
    /// Individual S commitments (one per value)
    public let Ss: [PointProjective]
    /// Combined polynomial commitments T1, T2
    public let T1: PointProjective
    public let T2: PointProjective
    /// Combined blinding factor
    public let taux: Fr
    /// Combined mu
    public let mu: Fr
    /// Combined inner product evaluation
    public let tHat: Fr
    /// Single inner product proof over the aggregated vectors
    public let ipProof: ProtocolIPAProof
    /// Number of values in the batch
    public let count: Int
    /// Bit size of the range per value
    public let bitSize: Int

    public init(commitments: [PointProjective], As: [PointProjective],
                Ss: [PointProjective], T1: PointProjective, T2: PointProjective,
                taux: Fr, mu: Fr, tHat: Fr,
                ipProof: ProtocolIPAProof, count: Int, bitSize: Int) {
        self.commitments = commitments
        self.As = As
        self.Ss = Ss
        self.T1 = T1
        self.T2 = T2
        self.taux = taux
        self.mu = mu
        self.tHat = tHat
        self.ipProof = ipProof
        self.count = count
        self.bitSize = bitSize
    }
}

// MARK: - Generator Set

/// Extended generator set for the range proof protocol.
/// Contains G[], H[] of size n plus Pedersen base points g, h, and optional u for IPA.
public struct ProtocolGenerators {
    /// Generator vector G (length n)
    public let G: [PointAffine]
    /// Generator vector H (length n)
    public let H: [PointAffine]
    /// Pedersen base point g (for value)
    public let g: PointAffine
    /// Pedersen blinding point h
    public let h: PointAffine
    /// IPA base point u (derived from transcript)
    public let u: PointAffine
    /// Bit size n
    public let n: Int

    /// Generate deterministic generators for n-bit range proofs.
    public static func generate(n: Int) -> ProtocolGenerators {
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be power of 2")
        let totalNeeded = 2 * n + 3  // G[n], H[n], g, h, u
        let seed = pointFromAffine(PointAffine(x: fpFromInt(1), y: fpFromInt(2)))

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(totalNeeded)
        var acc = seed
        for _ in 0..<totalNeeded {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, seed))
        }
        let affine = batchToAffine(projPoints)

        return ProtocolGenerators(
            G: Array(affine[0..<n]),
            H: Array(affine[n..<(2 * n)]),
            g: affine[2 * n],
            h: affine[2 * n + 1],
            u: affine[2 * n + 2],
            n: n)
    }

    /// Generate extended generators for batch proofs of m values, each n bits.
    /// Returns generators with G[], H[] of size m*n.
    public static func generateBatch(n: Int, count m: Int) -> ProtocolGenerators {
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be power of 2")
        precondition(m > 0, "count must be positive")
        let totalN = m * n
        let totalNeeded = 2 * totalN + 3
        let seed = pointFromAffine(PointAffine(x: fpFromInt(1), y: fpFromInt(2)))

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(totalNeeded)
        var acc = seed
        for _ in 0..<totalNeeded {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, seed))
        }
        let affine = batchToAffine(projPoints)

        return ProtocolGenerators(
            G: Array(affine[0..<totalN]),
            H: Array(affine[totalN..<(2 * totalN)]),
            g: affine[2 * totalN],
            h: affine[2 * totalN + 1],
            u: affine[2 * totalN + 2],
            n: totalN)
    }
}

// MARK: - GPU Range Proof Protocol Engine

/// GPU-accelerated range proof protocol engine.
///
/// Implements the full Bulletproofs range proof protocol with GPU acceleration
/// for the compute-intensive multi-scalar multiplication steps. Supports:
///   - Single value range proofs
///   - Batch/aggregated range proofs for multiple values
///   - Configurable bit size (8, 16, 32, 64)
///   - Automatic GPU/CPU fallback
public class GPURangeProofProtocolEngine {

    public static let version = Versions.gpuRangeProofProtocol

    private let msmEngine: MetalMSM?
    private let config: RangeProofProtocolConfig
    private let usesGPU: Bool

    /// Initialize the protocol engine with the given configuration.
    public init(config: RangeProofProtocolConfig = RangeProofProtocolConfig()) {
        self.config = config
        if config.useGPU {
            do {
                self.msmEngine = try MetalMSM()
                self.usesGPU = true
            } catch {
                self.msmEngine = nil
                self.usesGPU = false
            }
        } else {
            self.msmEngine = nil
            self.usesGPU = false
        }
    }

    /// Whether the engine is using GPU acceleration.
    public var isGPUEnabled: Bool { usesGPU }

    // MARK: - Single Value Prove

    /// Prove that `value` is in [0, 2^n) where n is determined by generators.
    ///
    /// - Parameters:
    ///   - value: the secret value to prove in range
    ///   - blinding: blinding factor gamma for V = v*g + gamma*h
    ///   - generators: protocol generator set (determines bit size)
    /// - Returns: a ProtocolRangeProof
    public func prove(value: UInt64, blinding: Fr,
                      generators: ProtocolGenerators) -> ProtocolRangeProof {
        let n = generators.n
        precondition(n >= 8 && n <= 64 && (n & (n - 1)) == 0,
                     "n must be power of 2 in [8, 64]")
        precondition(n == 64 || value < (1 << n),
                     "value \(value) out of range [0, 2^\(n))")

        let gProj = pointFromAffine(generators.g)
        let hProj = pointFromAffine(generators.h)

        // Step 1: Commit to value
        let vFr = frFromInt(value)
        let V = pointAdd(cPointScalarMul(gProj, vFr), cPointScalarMul(hProj, blinding))

        // Step 2: Bit decomposition
        var aL = [Fr](repeating: Fr.zero, count: n)
        var aR = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            if (value >> i) & 1 == 1 {
                aL[i] = Fr.one
            }
            // a_R[i] = a_L[i] - 1
            aR[i] = frSub(aL[i], Fr.one)
        }

        // Blinding scalars
        let alpha = deterministicScalar(seed: 1, base: blinding)
        let rho   = deterministicScalar(seed: 2, base: blinding)
        let tau1  = deterministicScalar(seed: 3, base: blinding)
        let tau2  = deterministicScalar(seed: 4, base: blinding)

        var sL = [Fr](repeating: Fr.zero, count: n)
        var sR = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            sL[i] = deterministicScalar(seed: UInt64(100 + i), base: blinding)
            sR[i] = deterministicScalar(seed: UInt64(200 + i), base: blinding)
        }

        // Step 3: A = <a_L, G> + <a_R, H> + alpha*h (GPU MSM)
        let A = gpuVectorPedersenCommit(
            leftScalars: aL, leftPoints: generators.G,
            rightScalars: aR, rightPoints: generators.H,
            blindScalar: alpha, blindPoint: generators.h)

        // Step 4: S = <s_L, G> + <s_R, H> + rho*h (GPU MSM)
        let S = gpuVectorPedersenCommit(
            leftScalars: sL, leftPoints: generators.G,
            rightScalars: sR, rightPoints: generators.H,
            blindScalar: rho, blindPoint: generators.h)

        // Fiat-Shamir transcript
        var transcript = RangeProofTranscript()
        transcript.appendLabel("range-proof-protocol")
        transcript.appendPoint(V)
        transcript.appendPoint(A)
        transcript.appendPoint(S)
        transcript.appendLabel("y")
        let y = transcript.challenge()
        transcript.appendScalar(y)
        transcript.appendLabel("z")
        let z = transcript.challenge()

        // Precompute powers
        var yPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { yPow[i] = frMul(yPow[i - 1], y) }

        let two = frFromInt(2)
        var twoPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { twoPow[i] = frMul(twoPow[i - 1], two) }

        let z2 = frMul(z, z)

        // Compute polynomial coefficient vectors (Hadamard products)
        var l0 = [Fr](repeating: Fr.zero, count: n)
        var l1 = sL  // l1 = sL
        var r0 = [Fr](repeating: Fr.zero, count: n)
        var r1 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            // l0[i] = aL[i] - z
            l0[i] = frSub(aL[i], z)
            // r0[i] = y^i * (aR[i] + z) + z^2 * 2^i
            let aRpZ = frAdd(aR[i], z)
            r0[i] = frAdd(frMul(yPow[i], aRpZ), frMul(z2, twoPow[i]))
            // r1[i] = y^i * sR[i]
            r1[i] = frMul(yPow[i], sR[i])
        }

        // Inner products for t(X) = <l(X), r(X)> = t0 + t1*X + t2*X^2
        let t0 = innerProduct(l0, r0)
        let t1 = frAdd(innerProduct(l0, r1), innerProduct(l1, r0))
        let t2 = innerProduct(l1, r1)

        // T1 = t1*g + tau1*h, T2 = t2*g + tau2*h
        let T1 = pointAdd(cPointScalarMul(gProj, t1), cPointScalarMul(hProj, tau1))
        let T2 = pointAdd(cPointScalarMul(gProj, t2), cPointScalarMul(hProj, tau2))

        // Fiat-Shamir: challenge x
        transcript.appendPoint(T1)
        transcript.appendPoint(T2)
        transcript.appendLabel("x")
        let x = transcript.challenge()

        // Evaluate l, r at x: lVec[i] = l0[i] + x*l1[i], rVec[i] = r0[i] + x*r1[i]
        var lVec = [Fr](repeating: Fr.zero, count: n)
        var rVec = [Fr](repeating: Fr.zero, count: n)
        l0.withUnsafeBytes { l0Buf in
            withUnsafeBytes(of: x) { xBuf in
                l1.withUnsafeBytes { l1Buf in
                    lVec.withUnsafeMutableBytes { lBuf in
                        bn254_fr_batch_linear_combine(
                            l0Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            l1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }
        r0.withUnsafeBytes { r0Buf in
            withUnsafeBytes(of: x) { xBuf in
                r1.withUnsafeBytes { r1Buf in
                    rVec.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_linear_combine(
                            r0Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            r1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
        }

        let tHat = innerProduct(lVec, rVec)

        // Blinding factors
        let x2 = frMul(x, x)
        let taux = frAdd(frAdd(frMul(tau2, x2), frMul(tau1, x)), frMul(z2, blinding))
        let mu = frAdd(alpha, frMul(rho, x))

        // Prepare H' = H[i] * y^{-i} for inner product argument
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { yInvPow[i] = frMul(yInvPow[i - 1], yInv) }

        var HPrime = [PointProjective](repeating: pointIdentity(), count: n)
        for i in 0..<n {
            HPrime[i] = cPointScalarMul(pointFromAffine(generators.H[i]), yInvPow[i])
        }
        let HPrimeAffine = batchToAffine(HPrime)

        // Derive IPA base point from transcript
        transcript.appendScalar(tHat)
        transcript.appendLabel("ipa_u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        // Run GPU-accelerated inner product argument
        let ipProof = gpuInnerProductArgument(
            G: generators.G, H: HPrimeAffine, U: uPoint,
            a: lVec, b: rVec, transcript: &transcript)

        return ProtocolRangeProof(
            V: V, A: A, S: S, T1: T1, T2: T2,
            taux: taux, mu: mu, tHat: tHat,
            ipProof: ipProof, bitSize: n)
    }

    // MARK: - Single Value Verify

    /// Verify a single range proof.
    ///
    /// - Parameters:
    ///   - proof: the range proof to verify
    ///   - commitment: expected commitment V (must match proof.V)
    ///   - generators: protocol generator set
    /// - Returns: true if the proof is valid
    public func verify(proof: ProtocolRangeProof,
                       commitment: PointProjective,
                       generators: ProtocolGenerators) -> Bool {
        let n = proof.bitSize
        guard n == generators.n else { return false }
        guard pointEqual(proof.V, commitment) else { return false }

        let gProj = pointFromAffine(generators.g)
        let hProj = pointFromAffine(generators.h)

        // Reconstruct Fiat-Shamir challenges
        var transcript = RangeProofTranscript()
        transcript.appendLabel("range-proof-protocol")
        transcript.appendPoint(proof.V)
        transcript.appendPoint(proof.A)
        transcript.appendPoint(proof.S)
        transcript.appendLabel("y")
        let y = transcript.challenge()
        transcript.appendScalar(y)
        transcript.appendLabel("z")
        let z = transcript.challenge()

        transcript.appendPoint(proof.T1)
        transcript.appendPoint(proof.T2)
        transcript.appendLabel("x")
        let x = transcript.challenge()

        let z2 = frMul(z, z)
        let x2 = frMul(x, x)

        // Check 1: tHat consistency
        // g^{tHat} * h^{taux} = V^{z^2} * g^{delta(y,z)} * T1^x * T2^{x^2}
        // where delta(y,z) = (z - z^2) * <1^n, y^n> - z^3 * <1^n, 2^n>
        var yPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { yPow[i] = frMul(yPow[i - 1], y) }

        let two = frFromInt(2)
        var twoPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { twoPow[i] = frMul(twoPow[i - 1], two) }

        // <1, y^n> = sum of y^i
        var sumY = Fr.zero
        for i in 0..<n { sumY = frAdd(sumY, yPow[i]) }

        // <1, 2^n> = sum of 2^i = 2^n - 1
        var sum2 = Fr.zero
        for i in 0..<n { sum2 = frAdd(sum2, twoPow[i]) }

        let z3 = frMul(z2, z)
        // delta = (z - z^2) * sumY - z^3 * sum2
        let delta = frSub(frMul(frSub(z, z2), sumY), frMul(z3, sum2))

        // LHS: g^{tHat} * h^{taux}
        let lhs = pointAdd(cPointScalarMul(gProj, proof.tHat),
                           cPointScalarMul(hProj, proof.taux))

        // RHS: V^{z^2} * g^{delta} * T1^x * T2^{x^2}
        let rhs = pointAdd(
            pointAdd(cPointScalarMul(proof.V, z2),
                     cPointScalarMul(gProj, delta)),
            pointAdd(cPointScalarMul(proof.T1, x),
                     cPointScalarMul(proof.T2, x2)))

        guard pointEqual(lhs, rhs) else { return false }

        // Check 2: Inner product argument verification
        // Reconstruct H' = H[i] * y^{-i}
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { yInvPow[i] = frMul(yInvPow[i - 1], yInv) }

        var HPrime = [PointProjective](repeating: pointIdentity(), count: n)
        for i in 0..<n {
            HPrime[i] = cPointScalarMul(pointFromAffine(generators.H[i]), yInvPow[i])
        }
        let HPrimeAffine = batchToAffine(HPrime)

        // Derive IPA base point
        transcript.appendScalar(proof.tHat)
        transcript.appendLabel("ipa_u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        // Compute P = A * S^x * h^{-mu} + <-z, G> + <z*y^i + z^2*2^i, H'>
        // Then verify IPA on P with respect to <l, r> = tHat
        let negMu = frNeg(proof.mu)
        var P = pointAdd(proof.A, cPointScalarMul(proof.S, x))
        P = pointAdd(P, cPointScalarMul(hProj, negMu))

        // Add generator offsets from z
        let negZ = frNeg(z)
        for i in 0..<n {
            P = pointAdd(P, cPointScalarMul(pointFromAffine(generators.G[i]), negZ))
            let hCoeff = frAdd(frMul(z, yPow[i]), frMul(z2, twoPow[i]))
            P = pointAdd(P, cPointScalarMul(HPrime[i], hCoeff))
        }

        // Add tHat * U
        P = pointAdd(P, cPointScalarMul(uPoint, proof.tHat))

        // Verify IPA
        return verifyIPA(
            proof: proof.ipProof, P: P,
            G: generators.G, H: HPrimeAffine, U: uPoint,
            transcript: &transcript)
    }

    // MARK: - Batch Prove

    /// Create an aggregated range proof for multiple values.
    /// All values must be in [0, 2^bitSize).
    ///
    /// - Parameters:
    ///   - values: array of secret values
    ///   - blindings: array of blinding factors (one per value)
    ///   - generators: protocol generators (G, H must have size >= count * bitSize)
    ///   - bitSize: bit size per value
    /// - Returns: a BatchProtocolRangeProof
    public func batchProve(values: [UInt64], blindings: [Fr],
                           generators: ProtocolGenerators,
                           bitSize: Int) -> BatchProtocolRangeProof {
        let m = values.count
        precondition(m == blindings.count, "values and blindings must have same count")
        precondition(m > 0, "must have at least one value")
        precondition(bitSize >= 8 && bitSize <= 64 && (bitSize & (bitSize - 1)) == 0)
        let totalN = m * bitSize
        precondition(generators.n >= totalN,
                     "generators must have at least \(totalN) points")

        let gProj = pointFromAffine(generators.g)
        let hProj = pointFromAffine(generators.h)

        // Commit to each value
        var commitments = [PointProjective]()
        commitments.reserveCapacity(m)
        for i in 0..<m {
            let vFr = frFromInt(values[i])
            let V = pointAdd(cPointScalarMul(gProj, vFr),
                             cPointScalarMul(hProj, blindings[i]))
            commitments.append(V)
        }

        // Concatenated bit decomposition across all values
        var aL = [Fr](repeating: Fr.zero, count: totalN)
        var aR = [Fr](repeating: Fr.zero, count: totalN)
        for j in 0..<m {
            let v = values[j]
            precondition(bitSize == 64 || v < (1 << bitSize))
            for i in 0..<bitSize {
                let idx = j * bitSize + i
                if (v >> i) & 1 == 1 {
                    aL[idx] = Fr.one
                }
                aR[idx] = frSub(aL[idx], Fr.one)
            }
        }

        // Blinding scalars for the concatenated vectors
        let combinedBlinding = blindings[0]
        let alpha = deterministicScalar(seed: 10001, base: combinedBlinding)
        let rho   = deterministicScalar(seed: 10002, base: combinedBlinding)
        let tau1  = deterministicScalar(seed: 10003, base: combinedBlinding)
        let tau2  = deterministicScalar(seed: 10004, base: combinedBlinding)

        var sL = [Fr](repeating: Fr.zero, count: totalN)
        var sR = [Fr](repeating: Fr.zero, count: totalN)
        for i in 0..<totalN {
            sL[i] = deterministicScalar(seed: UInt64(20000 + i), base: combinedBlinding)
            sR[i] = deterministicScalar(seed: UInt64(30000 + i), base: combinedBlinding)
        }

        // A and S commitments over the full concatenated vectors
        let useG = Array(generators.G[0..<totalN])
        let useH = Array(generators.H[0..<totalN])

        let A = gpuVectorPedersenCommit(
            leftScalars: aL, leftPoints: useG,
            rightScalars: aR, rightPoints: useH,
            blindScalar: alpha, blindPoint: generators.h)

        let S = gpuVectorPedersenCommit(
            leftScalars: sL, leftPoints: useG,
            rightScalars: sR, rightPoints: useH,
            blindScalar: rho, blindPoint: generators.h)

        // Fiat-Shamir
        var transcript = RangeProofTranscript()
        transcript.appendLabel("batch-range-proof-protocol")
        for V in commitments { transcript.appendPoint(V) }
        transcript.appendPoint(A)
        transcript.appendPoint(S)
        transcript.appendLabel("y")
        let y = transcript.challenge()
        transcript.appendScalar(y)
        transcript.appendLabel("z")
        let z = transcript.challenge()

        // Precompute y powers over totalN
        var yPow = [Fr](repeating: Fr.one, count: totalN)
        for i in 1..<totalN { yPow[i] = frMul(yPow[i - 1], y) }

        let two = frFromInt(2)
        var twoPow = [Fr](repeating: Fr.one, count: bitSize)
        for i in 1..<bitSize { twoPow[i] = frMul(twoPow[i - 1], two) }

        // z powers: z^2, z^3, ..., z^{m+1}
        var zPow = [Fr](repeating: Fr.zero, count: m + 2)
        zPow[0] = Fr.one
        zPow[1] = z
        for i in 2..<(m + 2) { zPow[i] = frMul(zPow[i - 1], z) }

        // Compute l0, r0, r1 for the aggregated case
        var l0 = [Fr](repeating: Fr.zero, count: totalN)
        var r0 = [Fr](repeating: Fr.zero, count: totalN)
        var r1 = [Fr](repeating: Fr.zero, count: totalN)
        for j in 0..<m {
            let zj2 = zPow[j + 2]  // z^{j+2}
            for i in 0..<bitSize {
                let idx = j * bitSize + i
                l0[idx] = frSub(aL[idx], z)
                let aRpZ = frAdd(aR[idx], z)
                r0[idx] = frAdd(frMul(yPow[idx], aRpZ), frMul(zj2, twoPow[i]))
                r1[idx] = frMul(yPow[idx], sR[idx])
            }
        }

        let t0 = innerProduct(l0, r0)
        let t1 = frAdd(innerProduct(l0, r1), innerProduct(sL, r0))
        let t2 = innerProduct(sL, r1)

        let T1 = pointAdd(cPointScalarMul(gProj, t1), cPointScalarMul(hProj, tau1))
        let T2 = pointAdd(cPointScalarMul(gProj, t2), cPointScalarMul(hProj, tau2))

        transcript.appendPoint(T1)
        transcript.appendPoint(T2)
        transcript.appendLabel("x")
        let x = transcript.challenge()

        // Evaluate l, r at x
        var lVec = [Fr](repeating: Fr.zero, count: totalN)
        var rVec = [Fr](repeating: Fr.zero, count: totalN)
        var xx = x
        l0.withUnsafeBytes { l0Buf in
            sL.withUnsafeBytes { sLBuf in
                lVec.withUnsafeMutableBytes { lBuf in
                    withUnsafeBytes(of: &xx) { xBuf in
                        bn254_fr_linear_combine(
                            l0Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            sLBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(totalN))
                    }
                }
            }
        }
        r0.withUnsafeBytes { r0Buf in
            r1.withUnsafeBytes { r1Buf in
                rVec.withUnsafeMutableBytes { rBuf in
                    withUnsafeBytes(of: &xx) { xBuf in
                        bn254_fr_linear_combine(
                            r0Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            r1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(totalN))
                    }
                }
            }
        }

        let tHat = innerProduct(lVec, rVec)

        // Aggregated blinding: taux = tau2*x^2 + tau1*x + sum_j z^{j+2} * gamma_j
        let x2 = frMul(x, x)
        var tauxVal = frAdd(frMul(tau2, x2), frMul(tau1, x))
        for j in 0..<m {
            tauxVal = frAdd(tauxVal, frMul(zPow[j + 2], blindings[j]))
        }
        let mu = frAdd(alpha, frMul(rho, x))

        // Prepare H' for IPA
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: totalN)
        for i in 1..<totalN { yInvPow[i] = frMul(yInvPow[i - 1], yInv) }

        var HPrime = [PointProjective](repeating: pointIdentity(), count: totalN)
        for i in 0..<totalN {
            HPrime[i] = cPointScalarMul(pointFromAffine(generators.H[i]), yInvPow[i])
        }
        let HPrimeAffine = batchToAffine(HPrime)

        transcript.appendScalar(tHat)
        transcript.appendLabel("ipa_u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        let ipProof = gpuInnerProductArgument(
            G: Array(generators.G[0..<totalN]),
            H: HPrimeAffine,
            U: uPoint,
            a: lVec, b: rVec,
            transcript: &transcript)

        // Collect per-value A/S (for batch proof, single A and S cover all)
        // In the aggregated protocol, A and S are single commitments
        let As = [A]
        let Ss = [S]

        return BatchProtocolRangeProof(
            commitments: commitments, As: As, Ss: Ss,
            T1: T1, T2: T2,
            taux: tauxVal, mu: mu, tHat: tHat,
            ipProof: ipProof, count: m, bitSize: bitSize)
    }

    // MARK: - Batch Verify

    /// Verify an aggregated batch range proof.
    ///
    /// - Parameters:
    ///   - proof: the batch range proof
    ///   - commitments: expected commitments (must match proof.commitments)
    ///   - generators: protocol generators
    /// - Returns: true if the batch proof is valid
    public func batchVerify(proof: BatchProtocolRangeProof,
                            commitments: [PointProjective],
                            generators: ProtocolGenerators) -> Bool {
        let m = proof.count
        let bitSize = proof.bitSize
        let totalN = m * bitSize
        guard commitments.count == m else { return false }
        guard generators.n >= totalN else { return false }

        // Check commitments match
        for i in 0..<m {
            guard pointEqual(proof.commitments[i], commitments[i]) else { return false }
        }

        let gProj = pointFromAffine(generators.g)
        let hProj = pointFromAffine(generators.h)

        // Reconstruct Fiat-Shamir challenges
        var transcript = RangeProofTranscript()
        transcript.appendLabel("batch-range-proof-protocol")
        for V in proof.commitments { transcript.appendPoint(V) }
        guard proof.As.count == 1, proof.Ss.count == 1 else { return false }
        let A = proof.As[0]
        let S = proof.Ss[0]
        transcript.appendPoint(A)
        transcript.appendPoint(S)
        transcript.appendLabel("y")
        let y = transcript.challenge()
        transcript.appendScalar(y)
        transcript.appendLabel("z")
        let z = transcript.challenge()

        transcript.appendPoint(proof.T1)
        transcript.appendPoint(proof.T2)
        transcript.appendLabel("x")
        let x = transcript.challenge()

        let x2 = frMul(x, x)

        // z powers
        var zPow = [Fr](repeating: Fr.zero, count: m + 2)
        zPow[0] = Fr.one
        zPow[1] = z
        for i in 2..<(m + 2) { zPow[i] = frMul(zPow[i - 1], z) }

        // y powers
        var yPow = [Fr](repeating: Fr.one, count: totalN)
        for i in 1..<totalN { yPow[i] = frMul(yPow[i - 1], y) }

        let two = frFromInt(2)
        var twoPow = [Fr](repeating: Fr.one, count: bitSize)
        for i in 1..<bitSize { twoPow[i] = frMul(twoPow[i - 1], two) }

        // Compute delta for the aggregated case
        var sumY = Fr.zero
        for i in 0..<totalN { sumY = frAdd(sumY, yPow[i]) }

        var sum2 = Fr.zero
        for i in 0..<bitSize { sum2 = frAdd(sum2, twoPow[i]) }

        // delta = (z - z^2) * sumY - sum_j z^{j+3} * sum2
        var delta = frMul(frSub(z, frMul(z, z)), sumY)
        for j in 0..<m {
            let zj3 = frMul(zPow[j + 2], z)
            delta = frSub(delta, frMul(zj3, sum2))
        }

        // Check: g^{tHat} * h^{taux} = sum_j V_j^{z^{j+2}} * g^{delta} * T1^x * T2^{x^2}
        let lhs = pointAdd(cPointScalarMul(gProj, proof.tHat),
                           cPointScalarMul(hProj, proof.taux))

        var rhs = cPointScalarMul(gProj, delta)
        for j in 0..<m {
            rhs = pointAdd(rhs, cPointScalarMul(proof.commitments[j], zPow[j + 2]))
        }
        rhs = pointAdd(rhs, cPointScalarMul(proof.T1, x))
        rhs = pointAdd(rhs, cPointScalarMul(proof.T2, x2))

        guard pointEqual(lhs, rhs) else { return false }

        // IPA verification
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: totalN)
        for i in 1..<totalN { yInvPow[i] = frMul(yInvPow[i - 1], yInv) }

        var HPrime = [PointProjective](repeating: pointIdentity(), count: totalN)
        for i in 0..<totalN {
            HPrime[i] = cPointScalarMul(pointFromAffine(generators.H[i]), yInvPow[i])
        }
        let HPrimeAffine = batchToAffine(HPrime)

        transcript.appendScalar(proof.tHat)
        transcript.appendLabel("ipa_u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        // Reconstruct P
        let negMu = frNeg(proof.mu)
        var P = pointAdd(A, cPointScalarMul(S, x))
        P = pointAdd(P, cPointScalarMul(hProj, negMu))

        let negZ = frNeg(z)
        for j in 0..<m {
            let zj2 = zPow[j + 2]
            for i in 0..<bitSize {
                let idx = j * bitSize + i
                P = pointAdd(P, cPointScalarMul(pointFromAffine(generators.G[idx]), negZ))
                let hCoeff = frAdd(frMul(z, yPow[idx]), frMul(zj2, twoPow[i]))
                P = pointAdd(P, cPointScalarMul(HPrime[idx], hCoeff))
            }
        }

        P = pointAdd(P, cPointScalarMul(uPoint, proof.tHat))

        return verifyIPA(
            proof: proof.ipProof, P: P,
            G: Array(generators.G[0..<totalN]),
            H: HPrimeAffine, U: uPoint,
            transcript: &transcript)
    }

    // MARK: - GPU Vector Pedersen Commitment

    /// Compute <leftScalars, leftPoints> + <rightScalars, rightPoints> + blind * blindPoint
    /// using GPU MSM when available.
    private func gpuVectorPedersenCommit(
        leftScalars: [Fr], leftPoints: [PointAffine],
        rightScalars: [Fr], rightPoints: [PointAffine],
        blindScalar: Fr, blindPoint: PointAffine
    ) -> PointProjective {
        let n = leftScalars.count
        precondition(n == leftPoints.count && n == rightScalars.count && n == rightPoints.count)

        var points = [PointAffine]()
        var scalars = [[UInt32]]()
        points.reserveCapacity(2 * n + 1)
        scalars.reserveCapacity(2 * n + 1)

        for i in 0..<n {
            if !leftScalars[i].isZero {
                points.append(leftPoints[i])
                scalars.append(frToLimbs(leftScalars[i]))
            }
        }
        for i in 0..<n {
            if !rightScalars[i].isZero {
                points.append(rightPoints[i])
                scalars.append(frToLimbs(rightScalars[i]))
            }
        }
        points.append(blindPoint)
        scalars.append(frToLimbs(blindScalar))

        if let engine = msmEngine, points.count >= config.gpuThreshold {
            if let result = try? engine.msm(points: points, scalars: scalars) {
                return result
            }
        }

        // CPU fallback
        let hProj = pointFromAffine(blindPoint)
        var result = cPointScalarMul(hProj, blindScalar)
        for i in 0..<n {
            if !leftScalars[i].isZero {
                result = pointAdd(result, cPointScalarMul(pointFromAffine(leftPoints[i]), leftScalars[i]))
            }
            if !rightScalars[i].isZero {
                result = pointAdd(result, cPointScalarMul(pointFromAffine(rightPoints[i]), rightScalars[i]))
            }
        }
        return result
    }

    // MARK: - GPU Inner Product Argument

    /// GPU-accelerated inner product argument prover.
    /// Halves the vector length each round, producing L, R commitments via GPU MSM.
    private func gpuInnerProductArgument(
        G: [PointAffine], H: [PointAffine], U: PointProjective,
        a inputA: [Fr], b inputB: [Fr],
        transcript: inout RangeProofTranscript
    ) -> ProtocolIPAProof {
        var n = inputA.count
        precondition(n > 0 && (n & (n - 1)) == 0)

        var a = inputA
        var b = inputB
        var gPts = G.map { pointFromAffine($0) }
        var hPts = H.map { pointFromAffine($0) }
        var Ls = [PointProjective]()
        var Rs = [PointProjective]()

        while n > 1 {
            let halfN = n / 2

            let cL = innerProduct(Array(a[0..<halfN]), Array(b[halfN..<n]))
            let cR = innerProduct(Array(a[halfN..<n]), Array(b[0..<halfN]))

            // GPU MSM for L and R
            let L = gpuComputeIPACommitment(
                aScalars: Array(a[0..<halfN]), gPoints: Array(gPts[halfN..<n]),
                bScalars: Array(b[halfN..<n]), hPoints: Array(hPts[0..<halfN]),
                c: cL, U: U, halfN: halfN)

            let R = gpuComputeIPACommitment(
                aScalars: Array(a[halfN..<n]), gPoints: Array(gPts[0..<halfN]),
                bScalars: Array(b[0..<halfN]), hPoints: Array(hPts[halfN..<n]),
                c: cR, U: U, halfN: halfN)

            Ls.append(L)
            Rs.append(R)

            transcript.appendPoint(L)
            transcript.appendPoint(R)
            transcript.appendLabel("ipa_round")
            let u = transcript.challenge()
            let uInv = frInverse(u)

            // Fold vectors
            var newA = [Fr](repeating: Fr.zero, count: halfN)
            var newB = [Fr](repeating: Fr.zero, count: halfN)
            var newG = [PointProjective](repeating: pointIdentity(), count: halfN)
            var newH = [PointProjective](repeating: pointIdentity(), count: halfN)

            for i in 0..<halfN {
                newA[i] = frAdd(frMul(u, a[i]), frMul(uInv, a[halfN + i]))
                newB[i] = frAdd(frMul(uInv, b[i]), frMul(u, b[halfN + i]))
                newG[i] = pointAdd(cPointScalarMul(gPts[i], uInv),
                                   cPointScalarMul(gPts[halfN + i], u))
                newH[i] = pointAdd(cPointScalarMul(hPts[i], u),
                                   cPointScalarMul(hPts[halfN + i], uInv))
            }

            a = newA
            b = newB
            gPts = newG
            hPts = newH
            n = halfN
        }

        return ProtocolIPAProof(Ls: Ls, Rs: Rs, a: a[0], b: b[0])
    }

    /// Compute an IPA round commitment: <a, G> + <b, H> + c*U via GPU MSM.
    private func gpuComputeIPACommitment(
        aScalars: [Fr], gPoints: [PointProjective],
        bScalars: [Fr], hPoints: [PointProjective],
        c: Fr, U: PointProjective, halfN: Int
    ) -> PointProjective {
        if let engine = msmEngine, halfN >= config.gpuThreshold / 2 {
            var points = [PointAffine]()
            var scalars = [[UInt32]]()
            let allG = batchToAffine(Array(gPoints))
            let allH = batchToAffine(Array(hPoints))
            let uAff = batchToAffine([U])

            for i in 0..<halfN {
                points.append(allG[i])
                scalars.append(frToLimbs(aScalars[i]))
            }
            for i in 0..<halfN {
                points.append(allH[i])
                scalars.append(frToLimbs(bScalars[i]))
            }
            points.append(uAff[0])
            scalars.append(frToLimbs(c))

            if let result = try? engine.msm(points: points, scalars: scalars) {
                return result
            }
        }

        // CPU fallback
        var result = cPointScalarMul(U, c)
        for i in 0..<halfN {
            result = pointAdd(result, cPointScalarMul(gPoints[i], aScalars[i]))
            result = pointAdd(result, cPointScalarMul(hPoints[i], bScalars[i]))
        }
        return result
    }

    // MARK: - IPA Verification

    /// Verify an inner product argument proof.
    /// Reconstructs the final check: P' = a*G' + b*H' + (a*b)*U
    private func verifyIPA(
        proof: ProtocolIPAProof, P: PointProjective,
        G: [PointAffine], H: [PointAffine], U: PointProjective,
        transcript: inout RangeProofTranscript
    ) -> Bool {
        var n = G.count
        guard proof.Ls.count == proof.Rs.count else { return false }
        let logN = proof.Ls.count
        guard (1 << logN) == n else { return false }

        var gPts = G.map { pointFromAffine($0) }
        var hPts = H.map { pointFromAffine($0) }
        var challenges = [Fr]()
        var challengeInvs = [Fr]()

        // Reconstruct challenges and fold P
        var currentP = P
        for round in 0..<logN {
            transcript.appendPoint(proof.Ls[round])
            transcript.appendPoint(proof.Rs[round])
            transcript.appendLabel("ipa_round")
            let u = transcript.challenge()
            let uInv = frInverse(u)
            challenges.append(u)
            challengeInvs.append(uInv)

            // P' = u^2 * L + P + u^{-2} * R
            let u2 = frMul(u, u)
            let uInv2 = frMul(uInv, uInv)
            currentP = pointAdd(pointAdd(cPointScalarMul(proof.Ls[round], u2), currentP),
                                cPointScalarMul(proof.Rs[round], uInv2))

            // Fold generators
            let halfN = n / 2
            var newG = [PointProjective](repeating: pointIdentity(), count: halfN)
            var newH = [PointProjective](repeating: pointIdentity(), count: halfN)
            for i in 0..<halfN {
                newG[i] = pointAdd(cPointScalarMul(gPts[i], uInv),
                                   cPointScalarMul(gPts[halfN + i], u))
                newH[i] = pointAdd(cPointScalarMul(hPts[i], u),
                                   cPointScalarMul(hPts[halfN + i], uInv))
            }
            gPts = newG
            hPts = newH
            n = halfN
        }

        // Final check: P' == a * G'[0] + b * H'[0] + (a*b) * U
        let ab = frMul(proof.a, proof.b)
        let expected = pointAdd(
            pointAdd(cPointScalarMul(gPts[0], proof.a),
                     cPointScalarMul(hPts[0], proof.b)),
            cPointScalarMul(U, ab))

        return pointEqual(currentP, expected)
    }

    // MARK: - Utility: Proof Size

    /// Compute the proof size in group elements and field elements.
    /// Returns (groupElements, fieldElements) counts.
    public static func proofSize(bitSize n: Int) -> (groups: Int, fields: Int) {
        // A, S, T1, T2, V = 5 group elements
        // taux, mu, tHat = 3 field elements
        // IPA: log2(n) L's + log2(n) R's = 2*log2(n) group elements, 2 field elements (a, b)
        let logN = Int(log2(Double(n)))
        return (groups: 5 + 2 * logN, fields: 3 + 2)
    }

    /// Compute the batch proof size.
    public static func batchProofSize(bitSize n: Int, count m: Int) -> (groups: Int, fields: Int) {
        let totalN = m * n
        let logTotalN = Int(log2(Double(totalN)))
        // commitments: m, A: 1, S: 1, T1: 1, T2: 1 = m + 4 group elements
        // taux, mu, tHat = 3 field elements
        // IPA: 2*log2(m*n) group elements, 2 field elements
        return (groups: m + 4 + 2 * logTotalN, fields: 3 + 2)
    }

    // MARK: - Helpers

    /// Fr inner product: <a, b> = sum(a[i] * b[i])
    private func innerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        precondition(a.count == b.count)
        let n = a.count
        var result = Fr.zero
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_inner_product(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }
        return result
    }

    /// Deterministic scalar from seed and base blinding.
    private func deterministicScalar(seed: UInt64, base: Fr) -> Fr {
        var data = [UInt8](repeating: 0, count: 40)
        withUnsafeBytes(of: seed) { buf in
            data.replaceSubrange(0..<8, with: buf)
        }
        withUnsafeBytes(of: base) { buf in
            data.replaceSubrange(8..<40, with: buf)
        }
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
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

}
